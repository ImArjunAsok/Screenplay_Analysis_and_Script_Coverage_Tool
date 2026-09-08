import sys
from pathlib import Path

import joblib
import numpy as np
import spacy
from scipy.sparse import hstack, csr_matrix

sys.path.append(str(Path(__file__).parent.parent))
from parser.screenplay_parser import ScreenplayParser
from nlp_pipeline.sentiment_arc import build_sentiment_arc 
from nlp_pipeline.story_structure import predict_beats
from nlp_pipeline.character_graph import build_graph, compute_centrality
from nlp_pipeline.classify_character_names import (
    get_ner_names_for_screenplay, classify_name, name_tokens,
)

DATASET_DIR = Path(__file__).parent.parent / "dataset"

print("Loading spaCy model...")
NLP = spacy.load("en_core_web_sm")

print("Loading genre model...")
_genre_bundle = joblib.load(DATASET_DIR / "genre_model.joblib")
GENRE_MODELS = _genre_bundle["models"]
GENRE_VECTORIZER = _genre_bundle["vectorizer"]
GENRE_LABELS = _genre_bundle["genres"]

print("Loading viability model...")
_viability_bundle = joblib.load(DATASET_DIR / "viability_model.joblib")
VIABILITY_MODEL = _viability_bundle["model"]
VIABILITY_VECTORIZER = _viability_bundle["vectorizer"]
VIABILITY_INCLUDES_GENRE = _viability_bundle["includes_genre"]
VIABILITY_GENRE_MLB = _viability_bundle["genre_mlb"]

print("All models loaded. Pipeline ready.")


def _structural_features(screenplay) -> list[float]:
    dialogue_count = screenplay.dialogue_count
    action_count = sum(len(s.action_lines) for s in screenplay.scenes)
    ratio = dialogue_count / action_count if action_count else 0.0
    return [screenplay.scene_count, len(screenplay.characters), dialogue_count, ratio]


def _full_text(screenplay) -> str:
    return " ".join(s.full_text for s in screenplay.scenes)


def predict_genres(screenplay) -> dict:
    text = _full_text(screenplay)
    tfidf = GENRE_VECTORIZER.transform([text])
    structural = np.array([_structural_features(screenplay)], dtype=float)
    X = hstack([tfidf, csr_matrix(structural)])

    predicted = []
    confidence = []
    for genre, model in zip(GENRE_LABELS, GENRE_MODELS):
        proba = float(model.predict_proba(X)[0][1])
        confidence.append({"genre": genre, "probability": round(proba, 3)})
        if model.predict(X)[0]:
            predicted.append(genre)

    confidence.sort(key=lambda c: -c["probability"])
    return {"predicted": predicted, "confidence": confidence}


def predict_viability(screenplay, predicted_genres: list[str]) -> dict:
    text = _full_text(screenplay)
    tfidf = VIABILITY_VECTORIZER.transform([text])
    structural = np.array([_structural_features(screenplay)], dtype=float)
    parts = [tfidf, csr_matrix(structural)]

    if VIABILITY_INCLUDES_GENRE and VIABILITY_GENRE_MLB is not None:
        genre_vec = VIABILITY_GENRE_MLB.transform([predicted_genres])
        parts.append(csr_matrix(genre_vec, dtype=float))

    X = hstack(parts)
    rating = float(VIABILITY_MODEL.predict(X)[0])
    return {
        "predicted_imdb_rating": round(rating, 2),
        "confidence": "low",
        "caveat": "Script text alone explains only a small fraction of what determines "
                  "audience reception (R^2 ~0.08 in cross-validation) -- acting, directing, "
                  "marketing, and other non-textual factors are the larger drivers and are "
                  "not captured by this model. Treat this number as a rough signal, not a forecast.",
    }


def _emotional_volatility(scores: list[float]) -> tuple[float, str]:
    if len(scores) < 2:
        return 0.0, "Insufficient data"
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance ** 0.5
    if std < 0.15:
        label = "Low"
    elif std < 0.35:
        label = "Moderate"
    else:
        label = "High"
    return round(std, 3), label


def _arc_interpretation(sentiment_label: str, volatility_label: str) -> str:
    if volatility_label == "High":
        return (
            f"The screenplay shows high emotional volatility despite a "
            f"{sentiment_label.lower()} overall tone -- the average score alone does not "
            f"fully represent the emotional trajectory; see the scene-by-scene chart."
        )
    return (
        f"The screenplay maintains a relatively {volatility_label.lower()}-volatility, "
        f"{sentiment_label.lower()} emotional tone throughout."
    )


def _sentiment_label(score: float) -> str:
    direction = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
    magnitude = abs(score)
    if magnitude < 0.05:
        return "Neutral / Balanced"
    elif magnitude < 0.2:
        return f"Mildly {direction}"
    elif magnitude < 0.45:
        return f"Moderately {direction}"
    else:
        return f"Strongly {direction}"


SENTIMENT_LABEL_CAVEAT = (
    "This label reflects distance from a neutral score, not narrative quality or "
    "genre expectations -- a low score does not indicate poor writing, and this "
    "does not yet account for genre norms (e.g. thrillers trending more negative "
    "than comedies on average). Treat it as a rough descriptive label, not a "
    "genre-adjusted benchmark."
)


def analyze_characters(screenplay) -> dict:
    scenes_for_ner = [{"action_lines": s.action_lines} for s in screenplay.scenes]
    ner_names = get_ner_names_for_screenplay(
        {"scenes": [{"action_lines": s.action_lines} for s in screenplay.scenes]}, NLP
    )
    ner_token_pool = set()
    for n in ner_names:
        ner_token_pool |= name_tokens(n)

    real_names, role_labels, uncertain = [], [], []
    for char in screenplay.characters:
        verdict, _ = classify_name(char, ner_token_pool, NLP)
        if verdict == "likely_name":
            real_names.append(char)
        elif verdict == "likely_role_label":
            role_labels.append(char)
        else:
            uncertain.append(char)

    return {
        "all_characters": screenplay.characters,
        "likely_real_names": real_names,
        "likely_role_labels": role_labels,
        "uncertain": uncertain,
    }


def analyze_relationships(screenplay) -> dict:
    allowed = set(screenplay.characters)
    scenes = [
        {"dialogue": [{"character": d.character} for d in s.dialogue if d.character in allowed]}
        for s in screenplay.scenes
    ]
    G = build_graph(scenes)
    centrality = compute_centrality(G)

    ranked = sorted(centrality.items(), key=lambda x: -x[1]["weighted_degree"])
    ranked_bridges = sorted(centrality.items(), key=lambda x: -x[1]["betweenness_centrality"])

    likely_protagonist = ranked[0][0] if ranked else None
    top_bridge_name = ranked_bridges[0][0] if ranked_bridges else None
    top_bridge_score = ranked_bridges[0][1]["betweenness_centrality"] if ranked_bridges else None

    network_interpretation = None
    if top_bridge_name and top_bridge_score is not None:
        if top_bridge_score >= 0.15:
            network_interpretation = (
                f"{top_bridge_name} has the highest betweenness centrality ({top_bridge_score:.3f}), "
                f"suggesting they frequently connect otherwise-separate character interactions."
            )
        else:
            network_interpretation = (
                f"No character shows strong bridging behaviour (highest betweenness centrality is "
                f"{top_bridge_name} at {top_bridge_score:.3f}) -- this cast is not strongly organised "
                f"around a single connector character."
            )

    return {
        "character_count_in_network": G.number_of_nodes(),
        "relationship_count": G.number_of_edges(),
        "likely_protagonist": likely_protagonist,
        "top_bridge_character": top_bridge_name,
        "network_interpretation": network_interpretation,
        "most_central_characters": [{"name": n, **c} for n, c in ranked[:5]],
        "top_bridge_characters": [{"name": n, "betweenness": c["betweenness_centrality"]} for n, c in ranked_bridges[:5]],
    }


def _beat_confidence(method: str) -> str:
    return "High" if method.startswith("sentiment-refined") else "Low"


STANDARD_LIMITATIONS = [
    "Character identification may contain false positives or negatives; low-confidence "
    "cases are flagged separately rather than silently trusted.",
    "Beat detection uses heuristic positional and sentiment signals -- it estimates where "
    "beats likely fall, it does not semantically verify that a scene IS that narrative beat.",
    "The sentiment model may misinterpret sarcasm, irony, and comedic dialogue, which read "
    "differently in tone than the film-review text it was trained on.",
    "Genre classification reflects textual word-choice signals, not filmmaking execution.",
    "The IMDb rating prediction does not account for casting, directing, marketing, or "
    "cultural context -- see the Viability Assessment for the measured explanatory ceiling.",
    "Scores in this report are automated estimates and should not be interpreted as "
    "objective measures of screenplay quality.",
]


def _pacing_analysis(screenplay) -> dict:
    scene_stats = []
    for i, scene in enumerate(screenplay.scenes):
        action_lines = len(scene.action_lines)
        dialogue_lines = len(scene.dialogue)
        total = action_lines + dialogue_lines
        density = dialogue_lines / total if total else 0.0
        scene_stats.append({
            "scene_index": i, "heading": scene.heading,
            "length_lines": total, "dialogue_density": round(density, 3),
        })

    if not scene_stats:
        return {"scene_count": 0}

    lengths = [s["length_lines"] for s in scene_stats]
    densities = [s["dialogue_density"] for s in scene_stats]
    avg_length = sum(lengths) / len(lengths)
    avg_density = sum(densities) / len(densities)

    shortest = min(scene_stats, key=lambda s: s["length_lines"])
    longest = max(scene_stats, key=lambda s: s["length_lines"])

    std_d = 0.0
    if len(densities) > 1:
        var_d = sum((d - avg_density) ** 2 for d in densities) / len(densities)
        std_d = var_d ** 0.5

    outliers = []
    if std_d > 0:
        for s in scene_stats:
            z = (s["dialogue_density"] - avg_density) / std_d
            if abs(z) >= 1.5:
                outliers.append({
                    "scene_index": s["scene_index"], "heading": s["heading"],
                    "dialogue_density": s["dialogue_density"],
                    "type": "high" if z > 0 else "low",
                })

    return {
        "scene_count": len(scene_stats),
        "average_scene_length_lines": round(avg_length, 1),
        "shortest_scene": {k: shortest[k] for k in ("scene_index", "heading", "length_lines")},
        "longest_scene": {k: longest[k] for k in ("scene_index", "heading", "length_lines")},
        "average_dialogue_density": round(avg_density, 3),
        "pacing_outliers": outliers[:8],
        "pacing_note": "Scene 'length' is a line-count proxy (action + dialogue lines), not "
                       "actual screen time. Outlier scenes are flagged using a simple statistical "
                       "threshold, not a validated pacing model.",
    }


def _dialogue_distribution(screenplay, real_names: list[str]) -> list[dict]:
    """Dialogue line share per REAL character -- role labels and
    uncertain cases excluded, matching the character-cleanup filter used
    everywhere else in the response."""
    counts: dict[str, int] = {}
    for scene in screenplay.scenes:
        for d in scene.dialogue:
            if d.character in real_names:
                counts[d.character] = counts.get(d.character, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return []

    distribution = [
        {"character": name, "dialogue_lines": count, "share_pct": round(count / total * 100, 1)}
        for name, count in counts.items()
    ]
    distribution.sort(key=lambda d: -d["dialogue_lines"])
    return distribution


def _character_arcs(screenplay, sentiment: dict, character_names: list[str]) -> list[dict]:
    scene_sentiment = {a["scene_index"]: a["sentiment_score"] for a in sentiment["arc"]}

    arcs = []
    for char in character_names:
        appearances = [
            i for i, scene in enumerate(screenplay.scenes)
            if any(d.character == char for d in scene.dialogue)
        ]
        if len(appearances) < 3:
            continue

        scores = [scene_sentiment.get(i, 0.0) for i in appearances]
        n = len(scores)
        third = max(1, n // 3)
        intro = scores[:third]
        mid = scores[third:n - third] or scores[third:third + 1]
        final = scores[-third:]

        intro_avg = sum(intro) / len(intro)
        mid_avg = sum(mid) / len(mid)
        final_avg = sum(final) / len(final)
        delta = final_avg - intro_avg

        if delta > 0.08:
            direction = "Positive"
        elif delta < -0.08:
            direction = "Negative"
        else:
            direction = "Flat"

        arcs.append({
            "character": char,
            "scene_appearances": n,
            "introduction_sentiment": round(intro_avg, 3),
            "midpoint_sentiment": round(mid_avg, 3),
            "final_sentiment": round(final_avg, 3),
            "arc_direction": direction,
            "arc_strength": round(abs(delta), 3),
        })

    arcs.sort(key=lambda a: -a["arc_strength"])
    return arcs


def analyze_screenplay(file_path: str, title_override: str = None) -> dict:
    parser = ScreenplayParser()
    screenplay = parser.parse_file(file_path)
    if title_override:
        screenplay.title = title_override

    if screenplay.scene_count == 0:
        return {
            "success": False,
            "error": "No scenes could be parsed from this file. It may not be a "
                     "standard screenplay format -- see parser_notes for detail.",
            "parser_notes": screenplay.parser_notes,
        }

    sentiment = build_sentiment_arc(screenplay)
    beats = predict_beats(screenplay.scene_count, sentiment)
    characters = analyze_characters(screenplay)
    relationships = analyze_relationships(screenplay)
    genre_result = predict_genres(screenplay)
    viability = predict_viability(screenplay, genre_result["predicted"])

    scene_scores = [s["sentiment_score"] for s in sentiment["arc"]]
    volatility_value, volatility_label = _emotional_volatility(scene_scores)
    sentiment_label = _sentiment_label(sentiment["statistics"]["average_sentiment"])

    pacing = _pacing_analysis(screenplay)
    dialogue_distribution = _dialogue_distribution(screenplay, characters["likely_real_names"])
    arc_character_names = [c["name"] for c in relationships["most_central_characters"]]
    character_arcs = _character_arcs(screenplay, sentiment, arc_character_names)

    return {
        "success": True,
        "title": screenplay.title,
        "overview": {
            "scene_count": screenplay.scene_count,
            "character_count": len(screenplay.characters),
            "dialogue_count": screenplay.dialogue_count,
        },
        "parser_notes": screenplay.parser_notes,
        "characters": characters,
        "dialogue_distribution": dialogue_distribution,
        "pacing": pacing,
        "sentiment_arc": {
            "average_sentiment": sentiment["statistics"]["average_sentiment"],
            "sentiment_label": sentiment_label,
            "sentiment_label_caveat": SENTIMENT_LABEL_CAVEAT,
            "emotional_volatility": volatility_value,
            "emotional_volatility_label": volatility_label,
            "arc_interpretation": _arc_interpretation(sentiment_label, volatility_label),
            "most_positive_scene": sentiment["statistics"]["most_positive_scene"],
            "most_negative_scene": sentiment["statistics"]["most_negative_scene"],
            "turning_point_count": sentiment["statistics"]["turning_point_count"],
            "model_source": sentiment["model_source"],
            "scene_scores": scene_scores,
            "smoothed_scores": sentiment["smoothed_arc"],
        },
        "character_arcs": character_arcs,
        "story_structure": {
            "detection_type": "Heuristic Story Structure Detection",
            "detection_note": "Beat positions are estimated from screenplay length and sentiment "
                               "signals -- this is evidence, not semantic proof that a given scene "
                               "IS that narrative beat.",
            "predicted_beats": [
                {
                    "beat": b["beat"], "scene_index": b["predicted_scene_index"], "method": b["method"],
                    "confidence": _beat_confidence(b["method"]),
                }
                for b in beats
            ],
        },
        "character_relationships": relationships,
        "predicted_genres": genre_result["predicted"],
        "genre_confidence": genre_result["confidence"],
        "viability": viability,
        "limitations": STANDARD_LIMITATIONS,
    }