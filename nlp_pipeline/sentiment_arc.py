import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from parser.screenplay_parser import ScreenplayParser

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FINE_TUNED_DIR = Path(__file__).parent.parent / "models" / "roberta-sentiment-finetuned"
FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 256


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



def score_scene(scene) -> float:
    text = scene.full_text.strip()
    if not text:
        return 0.0

    inputs = TOKENIZER(
        text, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    with torch.no_grad():
        logits = MODEL(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    p_negative, p_positive = probs[0].item(), probs[1].item()
    score = p_positive - p_negative
    return round(score, 4)


def build_sentiment_arc(screenplay) -> dict:
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

    scores = [s["sentiment_score"] for s in arc]

    smoothed = []
    window = 5
    for i in range(len(scores)):
        start = max(0, i - window // 2)
        end = min(len(scores), i + window // 2 + 1)
        smoothed.append(round(sum(scores[start:end]) / (end - start), 4))

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