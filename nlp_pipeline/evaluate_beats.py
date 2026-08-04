"""
Week 4 -- Evaluate predicted beats against manual annotations
------------------------------------------------------------------
Takes the CSV template from story_structure.py --export-template, AFTER
you've filled in the "actual_scene_index" column by reading the real
script, and measures how close the predictions were.

This is the "comparison of predicted beats against manual annotations"
evidence your timeline asks for.

Error is reported two ways:
  - scenes off       : simple, easy to read, but not comparable across
                        scripts of very different lengths
  - % of runtime off  : (scenes off / scene_count) -- THIS is the number
                        to report and compare across scripts, since being
                        5 scenes off in a 50-scene script is a much bigger
                        miss than 5 scenes off in a 300-scene script

Run (after filling in actual_scene_index by hand):
    python nlp_pipeline/evaluate_beats.py Black_Panther_beats_template.csv
"""

import csv
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python nlp_pipeline/evaluate_beats.py <filled_in_template.csv> [more.csv ...]")
        sys.exit(1)

    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            print(f"Skipping {path_str} (not found)")
            continue

        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        annotated = [r for r in rows if r["actual_scene_index"].strip() != ""]
        skipped = len(rows) - len(annotated)

        if not annotated:
            print(f"\n{path.name}: no beats annotated yet -- fill in actual_scene_index first.")
            continue

        # We don't know scene_count directly from the CSV, so estimate it
        # from the largest scene index referenced (predicted or actual) --
        # good enough for computing % of runtime.
        all_indices = [int(r["predicted_scene_index"]) for r in rows] + \
                      [int(r["actual_scene_index"]) for r in annotated]
        scene_count_estimate = max(all_indices) + 1

        print(f"\n{'='*72}")
        print(f"  BEAT PREDICTION ACCURACY -- {path.stem}")
        print(f"{'='*72}")
        if skipped:
            print(f"  ({skipped} beat(s) not yet annotated -- skipped)")

        errors_scenes = []
        errors_pct = []
        for r in annotated:
            pred = int(r["predicted_scene_index"])
            actual = int(r["actual_scene_index"])
            diff = abs(pred - actual)
            pct = diff / scene_count_estimate
            errors_scenes.append(diff)
            errors_pct.append(pct)
            flag = "  <-- off by >10% of runtime" if pct > 0.10 else ""
            print(f"  {r['beat']:<22} predicted={pred:>4}  actual={actual:>4}  "
                  f"off by {diff:>3} scenes ({pct:.1%}){flag}")

        mean_scenes = sum(errors_scenes) / len(errors_scenes)
        mean_pct = sum(errors_pct) / len(errors_pct)
        within_10pct = sum(1 for p in errors_pct if p <= 0.10)

        print(f"\n  Beats annotated       : {len(annotated)} / {len(rows)}")
        print(f"  Mean error            : {mean_scenes:.1f} scenes ({mean_pct:.1%} of runtime)")
        print(f"  Within 10% of runtime : {within_10pct} / {len(annotated)}")


if __name__ == "__main__":
    main()
