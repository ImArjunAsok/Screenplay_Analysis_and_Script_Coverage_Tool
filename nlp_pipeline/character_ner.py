"""
Week 2 -- spaCy NER character-extraction check
------------------------------------------------
Your parser already knows "who speaks" from character cues (ALL-CAPS names
followed by dialogue). This script adds the other half: "who's mentioned or
described," using spaCy NER over the action lines. The two signals matter
for different things -- cues are enough for dialogue analysis, but the
character-relationship graph (Week 5) needs to know when a character is
PRESENT in a scene even if they don't speak, which only shows up in action
text.

IMPORTANT -- a real finding, not a guess: spaCy's NER is trained on ordinary
mixed-case text and is noticeably worse at ALL-CAPS input. Tested directly:

    "CAPTAIN REYES shouts orders as Ensign Lopez scrambles."   -> []  (nothing found)
    "Captain Reyes shouts orders as Ensign Lopez scrambles."   -> [('Reyes', 'PERSON')]
    "Frank and Jesse ride toward the farmhouse."               -> [('Frank', 'PERSON'), ('Jesse', 'PERSON')]

Same sentence, same names -- the ALL-CAPS version loses the entity almost
entirely. That's why this script runs NER ONLY over action_lines (which are
normal mixed-case prose), never over headings, cues, or full_text (which
mixes cues back in). If you see this pipeline missing characters, the first
thing to check is whether they only ever appear in the parser's own
ALL-CAPS output (cues) and not in any action-line prose -- that's expected,
not a bug, and cue-detection already has you covered for those characters.

Run:
    python nlp_pipeline/character_ner.py

Reads dataset/corpus.jsonl (from build_dataset.py) and writes:
    dataset/ner_report.json   -- per-screenplay NER vs. cue-character comparison
    dataset/ner_summary.csv   -- one row per screenplay, for a quick skim
"""

import csv
import json
import re
import sys
from pathlib import Path

import spacy

CORPUS_FILE = Path(__file__).parent.parent / "dataset" / "corpus.jsonl"
OUT_JSON = Path(__file__).parent.parent / "dataset" / "ner_report.json"
OUT_CSV = Path(__file__).parent.parent / "dataset" / "ner_summary.csv"

# Strip a trailing (V.O.)/(O.S.)/(CONT'D) etc. and possessive 's before
# comparing a cue-name to an NER-detected name.
CUE_SUFFIX = re.compile(r"\s*\((?:V\.O\.|O\.S\.|O\.C\.|CONT'D|CONTINUED)\)\s*$", re.IGNORECASE)
POSSESSIVE = re.compile(r"'s$", re.IGNORECASE)


def normalize(name: str) -> str:
    name = CUE_SUFFIX.sub("", name)
    name = POSSESSIVE.sub("", name)
    return name.strip().upper()


def name_tokens(name: str) -> set[str]:
    """Individual name tokens, for loose matching -- 'Captain Reyes' cue vs
    NER's 'Reyes' should still count as a match on the surname alone.
    Also strips a possessive 's from each token individually (not just the
    end of the whole string), so a cue like 'ANNE'S VOICE' still matches an
    NER hit on plain 'Anne'."""
    return {POSSESSIVE.sub("", tok) for tok in normalize(name).split()}


def match_cue_to_ner(cue_characters: list[str], ner_names: set[str]) -> dict:
    """For each cue-derived character, check whether ANY NER-detected name
    shares a token with it (e.g. cue 'CAPTAIN REYES' matches NER 'Reyes').
    Returns per-character match status plus overall coverage."""
    ner_token_pool: set[str] = set()
    for n in ner_names:
        ner_token_pool |= name_tokens(n)

    matched = []
    unmatched = []
    for cue in cue_characters:
        cue_toks = name_tokens(cue)
        if cue_toks & ner_token_pool:
            matched.append(cue)
        else:
            unmatched.append(cue)

    return {
        "matched": matched,
        "unmatched": unmatched,
        "match_rate": round(len(matched) / len(cue_characters), 3) if cue_characters else None,
    }


def main():
    if not CORPUS_FILE.exists():
        print(f"No corpus found at {CORPUS_FILE}. Run build_dataset.py first.")
        sys.exit(1)

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "parser"])
    print("Model ready.\n")

    screenplays = []
    with open(CORPUS_FILE, encoding="utf-8") as f:
        for line in f:
            screenplays.append(json.loads(line))

    print(f"Running NER over {len(screenplays)} screenplays...\n")

    report = []
    for sp in screenplays:
        # Batch every scene's action lines through the pipeline together --
        # much faster than calling nlp() once per scene.
        action_texts = [" ".join(scene["action_lines"]) for scene in sp["scenes"]]
        ner_names: set[str] = set()
        extra_hits_per_scene = []

        for doc in nlp.pipe(action_texts):
            scene_people = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
            ner_names |= scene_people
            extra_hits_per_scene.append(len(scene_people))

        comparison = match_cue_to_ner(sp["characters"], ner_names)

        # Names NER found that never showed up as a speaking cue at all --
        # candidates for "present but doesn't speak" in the character graph.
        cue_token_pool: set[str] = set()
        for c in sp["characters"]:
            cue_token_pool |= name_tokens(c)
        ner_only = sorted(n for n in ner_names if not (name_tokens(n) & cue_token_pool))

        entry = {
            "title": sp["title"],
            "file": sp["file"],
            "cue_character_count": len(sp["characters"]),
            "ner_person_count": len(ner_names),
            "matched_count": len(comparison["matched"]),
            "match_rate": comparison["match_rate"],
            "unmatched_cue_characters": comparison["unmatched"],
            "ner_only_names": ner_only[:20],  # cap for readability
        }
        report.append(entry)

        print(f"  {sp['title']:<35} cue={entry['cue_character_count']:>3}  "
              f"NER={entry['ner_person_count']:>3}  "
              f"matched={entry['matched_count']:>3}  "
              f"match_rate={entry['match_rate']}")

    OUT_JSON.write_text(json.dumps(report, indent=2))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "title", "file", "cue_character_count", "ner_person_count",
            "matched_count", "match_rate",
        ])
        writer.writeheader()
        for entry in report:
            writer.writerow({k: entry[k] for k in writer.fieldnames})

    rates = [e["match_rate"] for e in report if e["match_rate"] is not None]
    if rates:
        print(f"\nAverage match rate (cue characters also found by NER): "
              f"{sum(rates) / len(rates):.1%}")
    print(f"\nWritten to {OUT_JSON} and {OUT_CSV}")


if __name__ == "__main__":
    main()