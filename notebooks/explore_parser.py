"""
It will parse all .txt files in data/, print a summary table,
and save a results JSON.
"""

import json
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from parser.screenplay_parser import ScreenplayParser

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "parsed_summary.json"

parser = ScreenplayParser()
results = []
failed = []

scripts = list(DATA_DIR.glob("*.txt"))
print(f"\nFound {len(scripts)} scripts in {DATA_DIR}\n")
print(f"{'#':<4} {'Title':<40} {'Scenes':>7} {'Chars':>7} {'Dialogue':>9}")
print("-" * 72)

for i, script_path in enumerate(sorted(scripts), 1):
    try:
        parsed = parser.parse_file(script_path)
        row = {
            "title": parsed.title,
            "file": script_path.name,
            "scene_count": parsed.scene_count,
            "character_count": len(parsed.characters),
            "dialogue_count": parsed.dialogue_count,
            "top_characters": parsed.characters[:5],
        }
        results.append(row)
        print(f"{i:<4} {parsed.title:<40} {parsed.scene_count:>7} {len(parsed.characters):>7} {parsed.dialogue_count:>9}")
    except Exception as e:
        failed.append({"file": script_path.name, "error": str(e)})
        print(f"{i:<4} {script_path.stem:<40} {'ERROR':>7}  — {e}")

print("-" * 72)
print(f"\n✅ Parsed successfully : {len(results)}")
print(f"❌ Failed              : {len(failed)}")

if results:
    avg_scenes = sum(r["scene_count"] for r in results) / len(results)
    avg_chars  = sum(r["character_count"] for r in results) / len(results)
    avg_dial   = sum(r["dialogue_count"] for r in results) / len(results)
    print(f"\nAverages across {len(results)} scripts:")
    print(f"  Scenes per script    : {avg_scenes:.1f}")
    print(f"  Characters per script: {avg_chars:.1f}")
    print(f"  Dialogue lines       : {avg_dial:.1f}")

# Save full summary
summary = {"parsed": results, "failed": failed}
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text(json.dumps(summary, indent=2))
print(f"\n💾 Full summary saved to {OUTPUT_FILE}")

if failed:
    print(f"\n⚠️  Failed files — fix these:")
    for f in failed:
        print(f"  {f['file']}: {f['error']}")
