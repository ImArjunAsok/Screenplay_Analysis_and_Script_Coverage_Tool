"""
Week 3 -- Sentiment arc visualization
----------------------------------------
Plots the scene-by-scene emotional arc produced by sentiment_arc.py:
raw scores, the smoothed curve, and marked turning points. This is the
"sentiment visualisations" evidence for Week 3.

Run (after sentiment_arc.py has produced a *_sentiment_arc.json file):
    python nlp_pipeline/plot_sentiment_arc.py Black_Panther_sentiment_arc.json

Or on several at once, to eyeball shapes across scripts:
    python nlp_pipeline/plot_sentiment_arc.py *_sentiment_arc.json

Saves a PNG next to each input JSON.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def plot_arc(json_path: Path):
    data = json.loads(json_path.read_text())

    raw = [s["sentiment_score"] for s in data["arc"]]
    smoothed = data["smoothed_arc"]
    x = list(range(len(raw)))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, raw, color="#c9c9c9", linewidth=1, label="raw per-scene score")
    ax.plot(x, smoothed, color="#1f6feb", linewidth=2.5, label="smoothed arc (window=5)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # Mark turning points
    for tp in data["turning_points"]:
        ax.axvline(tp["scene_index"], color="#e05252", alpha=0.25, linewidth=1)

    # Mark the darkest and lightest moments
    stats = data["statistics"]
    ax.annotate(
        "darkest moment",
        xy=(raw.index(stats["most_negative_score"]), stats["most_negative_score"]),
        xytext=(0, -25), textcoords="offset points",
        ha="center", fontsize=9, color="#e05252",
        arrowprops=dict(arrowstyle="->", color="#e05252", alpha=0.7),
    )
    ax.annotate(
        "brightest moment",
        xy=(raw.index(stats["most_positive_score"]), stats["most_positive_score"]),
        xytext=(0, 20), textcoords="offset points",
        ha="center", fontsize=9, color="#2ea043",
        arrowprops=dict(arrowstyle="->", color="#2ea043", alpha=0.7),
    )

    ax.set_title(f"{data['title']} -- sentiment arc  (model: {data.get('model_source', 'unknown')})")
    ax.set_xlabel("Scene index")
    ax.set_ylabel("Sentiment  (-1 negative  →  +1 positive)")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.15)

    fig.tight_layout()
    out_path = json_path.with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python nlp_pipeline/plot_sentiment_arc.py <arc.json> [more.json ...]")
        sys.exit(1)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"  Skipping {arg} (not found)")
            continue
        plot_arc(path)


if __name__ == "__main__":
    main()