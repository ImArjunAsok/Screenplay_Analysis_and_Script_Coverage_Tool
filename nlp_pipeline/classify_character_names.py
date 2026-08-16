"""
Character-name cleanup: real names vs. generic role labels
------------------------------------------------------------
FIRST VERSION OF THIS SCRIPT USED ISOLATED-WORD POS TAGGING AND IT DIDN'T
WORK WELL ENOUGH -- keeping this note because it's a real finding, not
hiding the dead end. Tagging a character name by itself, with no
sentence context ("Sykes" -> tagged NOUN, "Secretary" -> tagged PROPN),
turned out to be close to a coin flip on real script data: several real
surnames (Sykes, Watson, O'Neal, Minkler) got miscategorized as role
labels, while some genuine role labels (Secretary, Human Cop) got kept
as names. Verified directly against Alien Nation's real character list
before rejecting this approach.

CURRENT APPROACH: reuse the in-context NER check from character_ner.py.
If a cue name matches a PERSON entity spaCy found while reading the
actual action-line prose (i.e. a real sentence, not an isolated word),
that's a far stronger signal -- real sentence context is what NER
actually needs to work well, per the earlier ALL-CAPS finding.

This still isn't perfect: characters who ONLY ever appear in a speaking
cue and are never named in action-line prose (common for minor
characters) won't get an NER match either way, real name or not. For
those, this script falls back to the (weaker) POS + role-word check,
and flags them as lower-confidence so they're easy to spot-check rather
than silently trusted.

Run:
    python nlp_pipeline/classify_character_names.py

Reads dataset/corpus.jsonl, writes:
    dataset/character_classification.json
    dataset/corpus_clean_characters.jsonl   -- use this, not corpus.jsonl,
                                                for Week 5 onward
"""

import json
import re
import sys
from pathlib import Path

import spacy

CORPUS_FILE = Path(__file__).parent.parent / "dataset" / "corpus.jsonl"
OUT_REPORT = Path(__file__).parent.parent / "dataset" / "character_classification.json"
OUT_CORPUS = Path(__file__).parent.parent / "dataset" / "corpus_clean_characters.jsonl"

CUE_SUFFIX = re.compile(r"\s*\((?:V\.O\.|O\.S\.|O\.C\.|CONT'D|CONTINUED)\)\s*$", re.IGNORECASE)
POSSESSIVE = re.compile(r"'s$", re.IGNORECASE)

ROLE_WORDS = {
    "cop", "officer", "security", "pilot", "co-pilot", "operator", "driver",
    "doctor", "nurse", "secretary", "mayor", "guard", "soldier", "waiter",
    "waitress", "detective", "agent", "voice", "narrator", "scientist",
    "teacher", "dealer", "chief", "captain", "lieutenant", "sergeant",
    "private", "general", "president", "senator", "judge", "lawyer",
    "clerk", "manager", "worker", "man", "woman", "boy", "girl", "kid",
    "stranger", "passerby", "bystander", "crowd", "reporter", "anchor",
    "host", "human", "alien", "priest", "rabbi", "student", "professor",
}


def normalize(name: str) -> str:
    name = CUE_SUFFIX.sub("", name)
    name = POSSESSIVE.sub("", name)
    return name.strip().upper()


def name_tokens(name: str) -> set[str]:
    return {POSSESSIVE.sub("", tok) for tok in normalize(name).split()}


def get_ner_names_for_screenplay(sp: dict, nlp) -> set[str]:
    action_texts = [" ".join(scene["action_lines"]) for scene in sp["scenes"]]
    ner_names: set[str] = set()
    for doc in nlp.pipe(action_texts):
        ner_names |= {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
    return ner_names


def classify_name(cue: str, ner_token_pool: set[str], pos_nlp) -> tuple[str, str]:
    """Returns (verdict, confidence). confidence 'high' = NER-backed (real
    sentence context). 'low' = POS/stoplist fallback only -- worth a glance."""
    cue_toks = name_tokens(cue)

    if cue_toks & ner_token_pool:
        return "likely_name", "high"

    cleaned = CUE_SUFFIX.sub("", cue).strip()
    if not cleaned:
        return "uncertain", "low"

    words = cleaned.lower().split()
    doc = pos_nlp(cleaned.title())
    has_propn = any(tok.pos_ == "PROPN" for tok in doc)

    if has_propn and not any(w in words for w in ROLE_WORDS):
        return "likely_name", "low"
    if any(w in words for w in ROLE_WORDS):
        return "likely_role_label", "low"
    return "uncertain", "low"


def main():
    if not CORPUS_FILE.exists():
        print(f"No corpus found at {CORPUS_FILE}. Run build_dataset.py first.")
        sys.exit(1)

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    screenplays = [json.loads(line) for line in open(CORPUS_FILE, encoding="utf-8")]
    print(f"Classifying character names across {len(screenplays)} screenplays...\n")

    report = []
    totals = {"likely_name": 0, "likely_role_label": 0, "uncertain": 0}
    low_confidence_count = 0

    with open(OUT_CORPUS, "w", encoding="utf-8") as clean_f:
        for sp in screenplays:
            ner_names = get_ner_names_for_screenplay(sp, nlp)
            ner_token_pool: set[str] = set()
            for n in ner_names:
                ner_token_pool |= name_tokens(n)

            names, roles, uncertain, needs_review = [], [], [], []
            for char in sp["characters"]:
                verdict, confidence = classify_name(char, ner_token_pool, nlp)
                totals[verdict] += 1
                if confidence == "low":
                    low_confidence_count += 1
                    needs_review.append({"name": char, "verdict": verdict})

                if verdict == "likely_name":
                    names.append(char)
                elif verdict == "likely_role_label":
                    roles.append(char)
                else:
                    uncertain.append(char)

            entry = {
                "title": sp["title"],
                "original_character_count": len(sp["characters"]),
                "likely_names": names,
                "likely_role_labels": roles,
                "uncertain": uncertain,
                "low_confidence_needs_review": needs_review,
                "real_name_count": len(names),
            }
            report.append(entry)
            print(f"  {sp['title']:<35} {len(sp['characters']):>3} total  "
                  f"-> {len(names):>3} names, {len(roles):>3} role labels, "
                  f"{len(uncertain):>2} uncertain  "
                  f"({len(needs_review)} low-confidence, worth a glance)")

            sp_clean = dict(sp)
            sp_clean["characters"] = names
            sp_clean["role_label_characters"] = roles
            sp_clean["uncertain_characters"] = uncertain
            clean_f.write(json.dumps(sp_clean) + "\n")

    OUT_REPORT.write_text(json.dumps(report, indent=2))

    print(f"\n{'='*60}")
    print(f"  Likely real names   : {totals['likely_name']}")
    print(f"  Likely role labels  : {totals['likely_role_label']}")
    print(f"  Uncertain           : {totals['uncertain']}")
    print(f"  Low-confidence (no NER match, worth checking): {low_confidence_count}")
    print(f"{'='*60}")
    print(f"\nWritten to {OUT_REPORT} and {OUT_CORPUS}")


if __name__ == "__main__":
    main()