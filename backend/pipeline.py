"""
Week 7 -- Backend pipeline orchestration
--------------------------------------------
This is the core logic behind the FastAPI service: takes an uploaded
screenplay file and runs it through every analysis module built across
Weeks 1-6, combining everything into one response.

WHY MODELS ARE LOADED HERE, AT IMPORT TIME, NOT INSIDE THE REQUEST
HANDLER: loading a fine-tuned transformer or an XGBoost model takes real
time (seconds). If that happened on every request, the API would be
unusably slow and would reload the same files over and over for no
reason. Loading everything ONCE when this module is first imported --
which happens once, when the FastAPI server starts up -- means every
actual request is fast, since it's just running inference on
already-loaded models.

WHY VIABILITY PREDICTION USES THE GENRE MODEL'S OWN PREDICTION, NOT REAL
GENRE LABELS: during training, the viability model learned from REAL
IMSDB genre tags. But a brand-new script someone uploads here has no
such tag yet -- nothing external has classified it. So at prediction
time, this pipeline runs the genre classifier FIRST, and feeds ITS
prediction into the viability model as a stand-in for a real genre
label. This is a legitimate, common pattern (one model's output feeding
another), but it does mean a wrong genre prediction could nudge the
viability prediction slightly -- worth knowing, not a hidden flaw.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import spacy
from scipy.sparse import hstack, csr_matrix

sys.path.append(str(Path(__file__).parent.parent))
from parser.screenplay_parser import ScreenplayParser
from nlp_pipeline.sentiment_arc import build_sentiment_arc  # loads the sentiment model at import time
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


def predict_genres(screenplay) -> list[str]:
    text = _full_text(screenplay)
    tfidf = GENRE_VECTORIZER.transform([text])
    structural = np.array([_structural_features(screenplay)], dtype=float)
    # NOTE: at training time structural features were normalized against
    # the whole training set's mean/std. A single new script has no
    # "set" to normalize against, so this uses the raw values -- a known
    # simplification. Good enough for a first working version; a more
    # careful version would persist the training set's mean/std and
    # reuse them here.
    X = hstack([tfidf, csr_matrix(structural)])

    predicted = []
    for genre, model in zip(GENRE_LABELS, GENRE_MODELS):
        if model.predict(X)[0]:
            predicted.append(genre)
    return predicted


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


def _sentiment_label(score: float) -> str:
    """Plain-language bucket for a raw -1..+1 sentiment score. Thresholds
    are a reasonable, disclosed heuristic based on distance from zero --
    NOT derived from genre-specific norms. Deliberately doesn't claim to
    answer "is this normal for a comedy" -- that would need a separate
    analysis comparing scores across genres in the training corpus,
    which doesn't exist yet. See the caveat text shipped alongside this
    in the API response."""
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
    # Use the SAME filtered character list as everywhere else in the
    # response (2+ speaking appearances) -- without this, the graph
    # silently included one-line background characters (e.g. a single
    # "EMPLOYEE" line) that the main character list correctly excludes,
    # producing a node count that didn't match character_count anywhere
    # else in the response.
    allowed = set(screenplay.characters)
    scenes = [
        {"dialogue": [{"character": d.character} for d in s.dialogue if d.character in allowed]}
        for s in screenplay.scenes
    ]
    G = build_graph(scenes)
    centrality = compute_centrality(G)

    ranked = sorted(centrality.items(), key=lambda x: -x[1]["weighted_degree"])
    ranked_bridges = sorted(centrality.items(), key=lambda x: -x[1]["betweenness_centrality"])

    # "Most scenes shared" is a reasonable proxy for narrative centrality,
    # but it isn't guaranteed to be the actual protagonist -- an ensemble
    # hub, mentor figure, or antagonist can also score highest. Labelled
    # "likely" deliberately, not asserted as fact.
    likely_protagonist = ranked[0][0] if ranked else None

    return {
        "character_count_in_network": G.number_of_nodes(),
        "relationship_count": G.number_of_edges(),
        "likely_protagonist": likely_protagonist,
        "most_central_characters": [{"name": n, **c} for n, c in ranked[:5]],
        "top_bridge_characters": [{"name": n, "betweenness": c["betweenness_centrality"]} for n, c in ranked_bridges[:5]],
    }


def analyze_screenplay(file_path: str, title_override: str = None) -> dict:
    """The main entry point: parse a script and run the full analysis
    pipeline, returning one combined result. title_override is used by
    the API layer to pass through the ORIGINAL uploaded filename, since
    file_path itself is usually a randomly-named temp file on disk."""
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
    genres = predict_genres(screenplay)
    viability = predict_viability(screenplay, genres)

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
        "sentiment_arc": {
            "average_sentiment": sentiment["statistics"]["average_sentiment"],
            "sentiment_label": _sentiment_label(sentiment["statistics"]["average_sentiment"]),
            "sentiment_label_caveat": SENTIMENT_LABEL_CAVEAT,
            "most_positive_scene": sentiment["statistics"]["most_positive_scene"],
            "most_negative_scene": sentiment["statistics"]["most_negative_scene"],
            "turning_point_count": sentiment["statistics"]["turning_point_count"],
            "model_source": sentiment["model_source"],
            "scene_scores": [s["sentiment_score"] for s in sentiment["arc"]],
            "smoothed_scores": sentiment["smoothed_arc"],
        },
        "story_structure": {
            "predicted_beats": [
                {"beat": b["beat"], "scene_index": b["predicted_scene_index"], "method": b["method"]}
                for b in beats
            ],
        },
        "character_relationships": relationships,
        "predicted_genres": genres,
        "viability": viability,
    }