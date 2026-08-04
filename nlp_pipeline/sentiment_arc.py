"""
Sentiment Arc Module
--------------------
Takes a parsed screenplay and scores each scene's emotional tone.

Uses the fine-tuned RoBERTa model
(models/roberta-sentiment-finetuned/) if it exists. If you haven't run
train_sentiment_model.py yet, falls back to the original pretrained
SST-2 placeholder model, with a clear warning. So this module keeps
working at every stage, but you always know which model actually
produced a given score.

Two things changed from the Week-1 placeholder version, both discussed
in review:

1. Score is now P(positive) - P(negative), a genuine continuous value in
   [-1, +1], instead of "confidence of whichever label won." A model
   being 99% sure something is negative isn't the same as it being very
   negative -- confidence measures certainty, not intensity. The
   softmax difference is the standard way to get an actual continuous
   signal out of a binary classifier.
2. Long scenes are truncated by the tokenizer itself (subword-aware,
   respects the model's real token budget) instead of a crude
   text[:1000] character slice, which could cut mid-word and doesn't
   actually correspond to the model's token limit.

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

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FINE_TUNED_DIR = Path(__file__).parent.parent / "models" / "roberta-sentiment-finetuned"
FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 256  # must match what train_sentiment_model.py trained with


# ── Load model ───────────────────────────────────────────────────────────────

def load_model():
    if FINE_TUNED_DIR.exists():
        print(f"Loading fine-tuned sentiment model from {FINE_TUNED_DIR} ...")
        tokenizer = AutoTokenizer.from_pretrained(str(FINE_TUNED_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(str(FINE_TUNED_DIR))
        source = "fine-tuned-roberta"
    else:
        print(
            "\n*** No fine-tuned model found in models/roberta-sentiment-finetuned/. ***\n"
            "Falling back to the pretrained SST-2 placeholder "
            "(distilbert-base-uncased-finetuned-sst-2-english).\n"
            "Run nlp_pipeline/train_sentiment_model.py first if you want scene "
            "scores from the film-domain fine-tuned model instead.\n"
        )
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
        source = "pretrained-sst2-fallback"

    model.eval()
    return tokenizer, model, source


TOKENIZER, MODEL, MODEL_SOURCE = load_model()


# ── Scorer ───────────────────────────────────────────────────────────────────

def score_scene(scene) -> float:
    """
    Returns a sentiment score from -1.0 (very negative) to +1.0 (very
    positive): P(positive) - P(negative) from the model's own softmax
    output, not the raw "confidence" of whichever label won.
    """
    text = scene.full_text.strip()
    if not text:
        return 0.0

    inputs = TOKENIZER(
        text, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    with torch.no_grad():
        logits = MODEL(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    # Both models used here follow the convention label 0 = negative,
    # label 1 = positive (true for SST-2 and for Rotten Tomatoes).
    p_negative, p_positive = probs[0].item(), probs[1].item()
    score = p_positive - p_negative
    return round(score, 4)


def build_sentiment_arc(screenplay) -> dict:
    """
    Scores every scene and returns the full arc as structured data.
    """
    arc = []

    print(f"\nScoring {screenplay.scene_count} scenes for '{screenplay.title}' "
          f"(model: {MODEL_SOURCE})...")

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

        if (i + 1) % 20 == 0:
            print(f"  Scored {i + 1}/{screenplay.scene_count} scenes...")

    print(f"  Done. All {screenplay.scene_count} scenes scored.")

    # ── Arc statistics ────────────────────────────────────────────────────────
    scores = [s["sentiment_score"] for s in arc]

    # Smooth the arc using a rolling average (window=5)
    smoothed = []
    window = 5
    for i in range(len(scores)):
        start = max(0, i - window // 2)
        end = min(len(scores), i + window // 2 + 1)
        smoothed.append(round(sum(scores[start:end]) / (end - start), 4))

    # Turning points: where the smoothed arc crosses from positive to
    # negative or vice versa
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

    lowest_idx = scores.index(min(scores))
    highest_idx = scores.index(max(scores))

    return {
        "title": screenplay.title,
        "scene_count": screenplay.scene_count,
        "model_source": MODEL_SOURCE,
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
    print(f"  SENTIMENT ARC — {arc_data['title']}  (model: {arc_data['model_source']})")
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

    parser = ScreenplayParser()
    screenplay = parser.parse_file(script_path)

    if screenplay.scene_count == 0:
        print("Error: no scenes found. Check the parser output first.")
        sys.exit(1)

    arc_data = build_sentiment_arc(screenplay)
    print_summary(arc_data)

    output_path = Path(script_path).stem + "_sentiment_arc.json"
    Path(output_path).write_text(json.dumps(arc_data, indent=2))
    print(f"\n  Full arc saved to: {output_path}")