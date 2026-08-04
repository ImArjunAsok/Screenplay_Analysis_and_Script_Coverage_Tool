"""
Helper for manual beat annotation
-------------------------------------
Prints every scene in a script with its index and heading, so you can
look up "what scene number is this?" while reading the real screenplay
and filling in actual_scene_index in a *_beats_template.csv.

Run:
    python nlp_pipeline/list_scenes.py data/A_Real_Pain.txt

Tip: pipe it to a file and keep it open in a second tab/window while you
annotate, or use --search to jump straight to a rough area:
    python nlp_pipeline/list_scenes.py data/A_Real_Pain.txt --search "hotel"
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from parser.screenplay_parser import ScreenplayParser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script_path")
    ap.add_argument("--search", help="Only show scenes whose heading contains this text (case-insensitive)")
    args = ap.parse_args()

    parser = ScreenplayParser()
    screenplay = parser.parse_file(args.script_path)

    print(f"{screenplay.title} -- {screenplay.scene_count} scenes\n")
    try:
        for scene in screenplay.scenes:
            if args.search and args.search.lower() not in scene.heading.lower():
                continue
            first_line = scene.action_lines[0][:60] if scene.action_lines else ""
            print(f"  [{scene.index:>4}]  {scene.heading:<55} {first_line}")
    except BrokenPipeError:
        sys.stderr.close()


if __name__ == "__main__":
    main()
