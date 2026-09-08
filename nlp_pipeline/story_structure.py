import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from parser.screenplay_parser import ScreenplayParser

BEATS = [
    {"name": "Opening Image",        "position": 0.01, "signal": None,
     "description": "A snapshot of the story's starting world/tone."},
    {"name": "Theme Stated",         "position": 0.05, "signal": None,
     "description": "Someone states, in passing, what the story is really about."},
    {"name": "Catalyst",             "position": 0.10, "signal": "low",
     "description": "The inciting incident -- the event that kicks the story into motion."},
    {"name": "Debate",               "position": 0.17, "signal": None,
     "description": "The protagonist hesitates / weighs whether to act."},
    {"name": "Break into Two",       "position": 0.20, "signal": "turning_point",
     "description": "The protagonist commits and the story shifts into its main body."},
    {"name": "B Story",              "position": 0.22, "signal": None,
     "description": "A secondary storyline (often relational) begins."},
    {"name": "Fun and Games",        "position": 0.35, "signal": "high",
     "description": "The 'promise of the premise' -- the core appeal plays out."},
    {"name": "Midpoint",             "position": 0.50, "signal": "turning_point",
     "description": "A false victory or false defeat; stakes escalate."},
    {"name": "Bad Guys Close In",    "position": 0.62, "signal": "low",
     "description": "Pressure mounts, internally and externally."},
    {"name": "All Is Lost",          "position": 0.75, "signal": "low",
     "description": "The lowest point -- the darkest moment of the story."},
    {"name": "Dark Night of the Soul", "position": 0.80, "signal": "low",
     "description": "The protagonist processes the loss before finding a way forward."},
    {"name": "Break into Three",     "position": 0.82, "signal": "turning_point",
     "description": "A new idea/resolve emerges; the final push begins."},
    {"name": "Finale",               "position": 0.90, "signal": "high",
     "description": "The climax -- the protagonist confronts the story's central problem."},
    {"name": "Final Image",          "position": 0.99, "signal": None,
     "description": "A closing snapshot, often mirroring/contrasting the Opening Image."},
]


def refine_with_arc(expected_scene: int, signal: str, arc: list[float],
                     scene_count: int, turning_points: list[int],
                     window_frac: float = 0.06) -> tuple[int, str]:
    if signal is None or not arc:
        return expected_scene, "position-only"

    window = max(2, int(scene_count * window_frac))
    lo = max(0, expected_scene - window)
    hi = min(scene_count - 1, expected_scene + window)

    if signal == "turning_point":
        nearby = [tp for tp in turning_points if lo <= tp <= hi]
        if nearby:
            best = min(nearby, key=lambda tp: abs(tp - expected_scene))
            return best, "sentiment-refined (nearest turning point)"
        return expected_scene, "position-only (no turning point nearby)"

    window_scores = arc[lo:hi + 1]
    if not window_scores:
        return expected_scene, "position-only"

    if signal == "low":
        best_offset = window_scores.index(min(window_scores))
    else:
        best_offset = window_scores.index(max(window_scores))
    best_scene = lo + best_offset

    if abs(window_scores[best_offset] - arc[expected_scene]) < 0.05:
        return expected_scene, "position-only (arc too flat nearby to refine)"

    return best_scene, f"sentiment-refined (local {signal})"


def predict_beats(scene_count: int, arc_data: dict | None = None) -> list[dict]:
    smoothed = arc_data["smoothed_arc"] if arc_data else []
    turning_points = [tp["scene_index"] for tp in arc_data["turning_points"]] if arc_data else []

    predictions = []
    for beat in BEATS:
        expected_scene = min(scene_count - 1, max(0, round(beat["position"] * scene_count)))
        scene_index, method = refine_with_arc(
            expected_scene, beat["signal"], smoothed, scene_count, turning_points
        )
        predictions.append({
            "beat": beat["name"],
            "description": beat["description"],
            "expected_position_pct": beat["position"],
            "predicted_scene_index": scene_index,
            "method": method,
        })
    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script_path")
    ap.add_argument("--arc", help="Path to a *_sentiment_arc.json file from sentiment_arc.py (optional)")
    ap.add_argument("--export-template", action="store_true",
                     help="Also write a CSV template for manual annotation")
    args = ap.parse_args()

    parser = ScreenplayParser()
    screenplay = parser.parse_file(args.script_path)

    if screenplay.scene_count == 0:
        print("Error: no scenes found. Check the parser output first.")
        sys.exit(1)

    arc_data = None
    if args.arc:
        arc_path = Path(args.arc)
        if arc_path.exists():
            arc_data = json.loads(arc_path.read_text())
            print(f"Using sentiment arc from {arc_path} to refine beat predictions.")
        else:
            print(f"WARNING: --arc file {arc_path} not found, falling back to position-only predictions.")

    predictions = predict_beats(screenplay.scene_count, arc_data)

    for p in predictions:
        scene = screenplay.scenes[p["predicted_scene_index"]]
        p["predicted_heading"] = scene.heading

    print(f"\n{'='*70}")
    print(f"  PREDICTED BEATS -- {screenplay.title}  ({screenplay.scene_count} scenes)")
    print(f"{'='*70}")
    for p in predictions:
        print(f"  {p['beat']:<22} scene {p['predicted_scene_index']:>4}  "
              f"[{p['method']}]")
        print(f"      {p['predicted_heading']}")

    out_path = Path(args.script_path).stem + "_beats_predicted.json"
    Path(out_path).write_text(json.dumps({
        "title": screenplay.title,
        "scene_count": screenplay.scene_count,
        "used_sentiment_arc": arc_data is not None,
        "predictions": predictions,
    }, indent=2))
    print(f"\nSaved predictions to {out_path}")

    if args.export_template:
        template_path = Path(args.script_path).stem + "_beats_template.csv"
        with open(template_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "beat", "description", "predicted_scene_index", "predicted_heading",
                "actual_scene_index", "notes",
            ])
            for p in predictions:
                writer.writerow([
                    p["beat"], p["description"], p["predicted_scene_index"],
                    p["predicted_heading"], "", "",
                ])
        print(f"Saved annotation template to {template_path}")
        print(f"  --> Open this in a spreadsheet, read the actual script, and fill in")
        print(f"      'actual_scene_index' for each beat by hand. Then run")
        print(f"      evaluate_beats.py on the filled-in file.")


if __name__ == "__main__":
    main()
