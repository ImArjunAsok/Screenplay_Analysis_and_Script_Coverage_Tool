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
