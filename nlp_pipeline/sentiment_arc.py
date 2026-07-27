"""
Sentiment Arc Module
--------------------
Takes a parsed screenplay and scores each scene's emotional tone
using a pre-trained HuggingFace model.

Run standalone:
    python nlp_pipeline/sentiment_arc.py data/127_Hours.txt

Output: JSON with per-scene sentiment scores + arc summary
"""

import json
import sys
from pathlib import Path

# Add project root so we can import the parser
sys.path.append(str(Path(__file__).parent.parent))
from parser.screenplay_parser import ScreenplayParser

from transformers import pipeline


# ── Load model ───────────────────────────────────────────────────────────────
# We use a pre-trained sentiment model fine-tuned on SST-2 (movie reviews).
# First run will download ~500MB — after that it's cached locally.

print("Loading sentiment model... (first run downloads ~500MB)")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512,
)
print("Model ready.")


# ── Scorer ───────────────────────────────────────────────────────────────────

def score_scene(scene) -> float:
    """
    Returns a sentiment score from -1.0 (very negative) to +1.0 (very positive).
    Uses the scene's full text — action lines + dialogue combined.
    """
    text = scene.full_text.strip()
    if not text:
        return 0.0

    # Model can only handle 512 tokens — truncate long scenes
    text = text[:1000]

    result = sentiment_model(text)[0]
    score = result["score"]  # confidence 0.0 → 1.0

    # Convert to -1 to +1 scale
    if result["label"] == "POSITIVE":
        return round(score, 4)
    else:
        return round(-score, 4)


def build_sentiment_arc(screenplay) -> dict:
    """
    Scores every scene and returns the full arc as structured data.
    """
    arc = []

    print(f"\nScoring {screenplay.scene_count} scenes for '{screenplay.title}'...")

    for i, scene in enumerate(screenplay.scenes):
        score = score_scene(scene)
        arc.append({
            "scene_index": i,
            "heading": scene.heading,
            "location": scene.location,
            "time_of_day": scene.time_of_day,
            "sentiment_score": score,
            "label": "POSITIVE" if score >= 0 else "NEGATIVE",
            "dialogue_count": len(scene.dialogue),
            "action_line_count": len(scene.action_lines),
        })

        # Progress every 20 scenes
        if (i + 1) % 20 == 0:
            print(f"  Scored {i + 1}/{screenplay.scene_count} scenes...")

    print(f"  Done. All {screenplay.scene_count} scenes scored.")

    # ── Arc statistics ────────────────────────────────────────────────────────
    scores = [s["sentiment_score"] for s in arc]

    # Smooth the arc using a rolling average (window=5)
    # This removes noise and shows the real emotional shape
    smoothed = []
    window = 5
    for i in range(len(scores)):
        start = max(0, i - window // 2)
        end = min(len(scores), i + window // 2 + 1)
        smoothed.append(round(sum(scores[start:end]) / (end - start), 4))

    # Find the emotional turning points
    # A turning point is where sentiment crosses from positive to negative or vice versa
    turning_points = []
    for i in range(1, len(smoothed)):
        if (smoothed[i - 1] >= 0 and smoothed[i] < 0) or \
           (smoothed[i - 1] < 0 and smoothed[i] >= 0):
            turning_points.append({
                "scene_index": i,
                "heading": arc[i]["heading"],
                "from": smoothed[i - 1],
                "to": smoothed[i],
            })

    # Find the lowest point (darkest moment) and highest point
    lowest_idx = scores.index(min(scores))
    highest_idx = scores.index(max(scores))

    return {
        "title": screenplay.title,
        "scene_count": screenplay.scene_count,
        "arc": arc,
        "smoothed_arc": smoothed,
        "statistics": {
            "average_sentiment": round(sum(scores) / len(scores), 4),
            "most_positive_scene": arc[highest_idx]["heading"],
            "most_positive_score": scores[highest_idx],
            "most_negative_scene": arc[lowest_idx]["heading"],
            "most_negative_score": scores[lowest_idx],
            "positive_scene_count": sum(1 for s in scores if s >= 0),
            "negative_scene_count": sum(1 for s in scores if s < 0),
            "turning_point_count": len(turning_points),
        },
        "turning_points": turning_points,
    }


def print_summary(arc_data: dict):
    stats = arc_data["statistics"]
    print(f"\n{'='*60}")
    print(f"  SENTIMENT ARC — {arc_data['title']}")
    print(f"{'='*60}")
    print(f"  Scenes analysed   : {arc_data['scene_count']}")
    print(f"  Average sentiment : {stats['average_sentiment']:+.4f}")
    print(f"  Positive scenes   : {stats['positive_scene_count']}")
    print(f"  Negative scenes   : {stats['negative_scene_count']}")
    print(f"  Turning points    : {stats['turning_point_count']}")
    print(f"\n  Most positive scene:")
    print(f"    {stats['most_positive_scene']} ({stats['most_positive_score']:+.4f})")
    print(f"  Most negative scene (darkest moment):")
    print(f"    {stats['most_negative_scene']} ({stats['most_negative_score']:+.4f})")

    print(f"\n  Emotional arc (every 10th scene):")
    arc = arc_data["smoothed_arc"]
    step = max(1, len(arc) // 20)
    for i in range(0, len(arc), step):
        bar_len = int(abs(arc[i]) * 20)
        bar = "█" * bar_len
        sign = "+" if arc[i] >= 0 else "-"
        print(f"    Scene {i:>3}: [{sign}{bar:<20}] {arc[i]:+.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nlp_pipeline/sentiment_arc.py <script.txt>")
        sys.exit(1)

    script_path = sys.argv[1]

    # Parse the screenplay
    parser = ScreenplayParser()
    screenplay = parser.parse_file(script_path)

    if screenplay.scene_count == 0:
        print("Error: no scenes found. Check the parser output first.")
        sys.exit(1)

    # Build the arc
    arc_data = build_sentiment_arc(screenplay)

    # Print summary
    print_summary(arc_data)

    # Save JSON
    output_path = Path(script_path).stem + "_sentiment_arc.json"
    Path(output_path).write_text(json.dumps(arc_data, indent=2))
    print(f"\n  Full arc saved to: {output_path}")
