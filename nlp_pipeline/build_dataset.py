"""
Week 2 -- Dataset builder
-------------------------
Parses every .txt screenplay in data/, sorts each into one of three tiers
using the same quality signals notebooks/explore_parser.py surfaces, and
writes out a consolidated corpus ready for the NLP pipeline -- this is the
"cleaned IMSDB dataset" deliverable for Week 2.

Tiers:
  clean    - no flags at all. Use freely.
  review   - flagged (used a fallback, e.g. indentation or bare-heading),
             but scene_count is high enough (>=20) that the output is very
             likely fine. Included in the corpus, but marked so you can
             spot-check a sample before trusting it blindly.
  excluded - zero scenes, non-standard format, or so few scenes relative to
             dialogue that most headings were clearly missed. NOT included
             in corpus.jsonl. Logged with a reason instead, so you have a
             record for the dissertation's data-quality section rather than
             just silently dropping files.

Run:
    python nlp_pipeline/build_dataset.py

Outputs (into dataset/):
    corpus.jsonl      - one JSON object per included screenplay (clean + review)
    excluded_log.csv   - file, title, reason(s), counts -- for everything left out
    manifest.json       - summary counts
"""

import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from parser.screenplay_parser import ScreenplayParser

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "dataset"
CORPUS_FILE = OUT_DIR / "corpus.jsonl"
EXCLUDED_LOG = OUT_DIR / "excluded_log.csv"
MANIFEST_FILE = OUT_DIR / "manifest.json"


def classify(parsed) -> tuple[str, list[str]]:
    """Returns (tier, reasons). Mirrors the flag logic in explore_parser.py
    so the two tools never disagree about what counts as broken."""
    reasons = []

    if parsed.scene_count == 0:
        reasons.append("no_scenes_found")
    elif parsed.scene_count <= 5 and parsed.dialogue_count > 50:
        reasons.append("suspiciously_few_scenes")

    for note in parsed.parser_notes:
        if "No scene headings" in note:
            reasons.append("not_standard_screenplay_format")
        elif "Indentation was unreliable" in note:
            reasons.append("indent_fallback_used")
        elif "bare" in note.lower():
            reasons.append("bare_heading_fallback_used")

    hard_exclude = {"no_scenes_found", "suspiciously_few_scenes", "not_standard_screenplay_format"}
    if hard_exclude & set(reasons):
        return "excluded", reasons
    if reasons:
        return "review", reasons
    return "clean", reasons


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parser = ScreenplayParser()
    scripts = sorted(DATA_DIR.glob("*.txt"))

    print(f"\nFound {len(scripts)} scripts in {DATA_DIR}\n")

    counts = {"clean": 0, "review": 0, "excluded": 0}
    reason_counts: dict[str, int] = {}

    with open(CORPUS_FILE, "w", encoding="utf-8") as corpus_f, \
         open(EXCLUDED_LOG, "w", encoding="utf-8", newline="") as excl_f:

        excl_writer = csv.writer(excl_f)
        excl_writer.writerow(["file", "title", "reasons", "scene_count", "character_count", "dialogue_count"])

        for script_path in scripts:
            try:
                parsed = parser.parse_file(script_path)
            except Exception as e:
                counts["excluded"] += 1
                reason_counts["parse_exception"] = reason_counts.get("parse_exception", 0) + 1
                excl_writer.writerow([script_path.name, script_path.stem, "parse_exception: " + str(e), "", "", ""])
                print(f"  ERROR  {script_path.name}: {e}")
                continue

            tier, reasons = classify(parsed)
            counts[tier] += 1
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1

            if tier == "excluded":
                excl_writer.writerow([
                    script_path.name, parsed.title, ";".join(reasons),
                    parsed.scene_count, len(parsed.characters), parsed.dialogue_count,
                ])
                print(f"  EXCLUDE  {parsed.title:<40} {','.join(reasons)}")
                continue

            record = parsed.to_dict()
            record["file"] = script_path.name
            record["tier"] = tier
            record["exclusion_reasons"] = reasons  # empty for "clean"
            corpus_f.write(json.dumps(record) + "\n")
            print(f"  {tier.upper():<8} {parsed.title:<40} scenes={parsed.scene_count} chars={len(parsed.characters)} dialogue={parsed.dialogue_count}")

    manifest = {
        "total_scripts": len(scripts),
        "clean": counts["clean"],
        "review": counts["review"],
        "excluded": counts["excluded"],
        "included_in_corpus": counts["clean"] + counts["review"],
        "exclusion_reasons": reason_counts,
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'='*60}")
    print(f"  Clean            : {counts['clean']}")
    print(f"  Review (usable)  : {counts['review']}")
    print(f"  Excluded         : {counts['excluded']}")
    print(f"  --> corpus.jsonl has {counts['clean'] + counts['review']} screenplays")
    print(f"{'='*60}")
    print(f"\nWritten to {OUT_DIR}/:")
    print(f"  corpus.jsonl     -- one parsed screenplay per line (clean + review)")
    print(f"  excluded_log.csv -- everything left out, with reasons")
    print(f"  manifest.json    -- summary counts")


if __name__ == "__main__":
    main()