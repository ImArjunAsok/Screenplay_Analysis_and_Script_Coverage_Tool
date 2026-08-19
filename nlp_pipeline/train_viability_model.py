"""
Week 6 (part 2) -- Viability prediction
--------------------------------------------
Predicts a screenplay's IMDb rating from its text and structure -- the
same TF-IDF + structural feature approach as the genre classifier
(train_genre_classifier.py), but regression instead of classification,
since rating is a continuous number (0-10), not a category.

WHY A REGRESSOR, NOT A "HIT/FLOP" CLASSIFIER: turning rating into a
binary label would need an arbitrary cutoff (is 6.5 a "hit"? 7.0?) that
throws away information. Predicting the actual number is more honest
about what the model does and doesn't know, and you can always bucket
the output afterward if a simple category is more useful for the report.

WHY A NAIVE BASELINE IS INCLUDED: "mean absolute error of 0.8" means
nothing on its own. This script also reports the error you'd get by
just always guessing the average rating, with no model at all -- the
real model is only meaningfully useful if it beats that.

GENRE-AS-A-FEATURE EXPERIMENT: by default, this now runs TWO versions
back to back -- text+structure only, and text+structure+genre -- and
reports both, so the effect of adding genre is a direct, honest
before/after rather than something you have to run twice yourself and
compare by hand. Uses the REAL genre labels from IMSDB (already in the
data), not the genre classifier's predictions -- that isolates "does
genre information help predict rating" from "how accurate is my genre
classifier," which are two different questions. Use --no-genre-compare
to skip this and just run the base version.

Run:
    python nlp_pipeline/train_viability_model.py dataset/corpus_with_viability.jsonl
    python nlp_pipeline/train_viability_model.py dataset/corpus_with_viability.jsonl --no-genre-compare

Outputs:
    dataset/viability_report.json    -- CV metrics (both versions),
                                         baseline comparison, sample predictions
    dataset/viability_model.joblib    -- trained model + vectorizer
                                         (the genre-augmented version, if run)
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

OUT_DIR = Path(__file__).parent.parent / "dataset"
NON_GENRE_LABELS = {"Short"}


def load_data(corpus_path: str):
    screenplays = [json.loads(l) for l in open(corpus_path, encoding="utf-8")]
    usable = [
        sp for sp in screenplays
        if sp.get("viability", {}).get("matched") and sp["viability"].get("imdb_rating") is not None
    ]
    dropped = len(screenplays) - len(usable)
    return usable, dropped


def extract_features(screenplays: list[dict], vectorizer: TfidfVectorizer = None, fit=True,
                      include_genre: bool = False, mlb: MultiLabelBinarizer = None):
    """Same TF-IDF + structural approach as train_genre_classifier.py's
    extract_features -- duplicated here (not imported) so this script
    stays runnable on its own. If include_genre=True, also appends a
    multi-hot encoded genre vector (real IMSDB labels, not predictions)."""
    texts, structural, genre_lists = [], [], []
    for sp in screenplays:
        scene_texts = [s.get("full_text", "") for s in sp["scenes"]]
        texts.append(" ".join(scene_texts))

        dialogue_count = sp.get("dialogue_count", sum(len(s.get("dialogue", [])) for s in sp["scenes"]))
        action_count = sum(len(s.get("action_lines", [])) for s in sp["scenes"])
        ratio = dialogue_count / action_count if action_count else 0.0

        structural.append([
            sp.get("scene_count", len(sp["scenes"])),
            len(sp.get("characters", [])),
            dialogue_count,
            ratio,
        ])
        genre_lists.append([g for g in sp.get("genres", []) if g not in NON_GENRE_LABELS])

    if fit:
        vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", min_df=2)
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)

    structural = np.array(structural, dtype=float)
    structural = (structural - structural.mean(axis=0)) / (structural.std(axis=0) + 1e-6)

    parts = [tfidf, csr_matrix(structural)]

    if include_genre:
        if fit:
            mlb = MultiLabelBinarizer()
            genre_matrix = mlb.fit_transform(genre_lists)
        else:
            genre_matrix = mlb.transform(genre_lists)
        parts.append(csr_matrix(genre_matrix, dtype=float))

    X = hstack(parts)
    return X, vectorizer, mlb


def run_cross_validation(X, y, folds: int, label: str) -> dict:
    """Runs K-fold CV and returns the metrics dict. Pulled out as its own
    function so the with-genre and without-genre runs use IDENTICAL
    logic -- the only thing that can differ between them is the feature
    matrix X itself, which keeps the comparison honest."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    fold_mae, fold_rmse, fold_r2 = [], [], []

    print(f"\nRunning {folds}-fold cross-validation -- {label}...")
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X.toarray()), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        fold_mae.append(mae); fold_rmse.append(rmse); fold_r2.append(r2)
        print(f"  Fold {fold_i}: MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

    return {
        "mean_mae": float(np.mean(fold_mae)),
        "std_mae": float(np.std(fold_mae)),
        "mean_rmse": float(np.mean(fold_rmse)),
        "mean_r2": float(np.mean(fold_r2)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_path")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-genre-compare", action="store_true",
                     help="Skip the genre-as-a-feature comparison, just run the base version")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    screenplays, dropped = load_data(args.corpus_path)
    print(f"Screenplays with a usable IMDb rating: {len(screenplays)} "
          f"({dropped} dropped -- no OMDb match or no rating)")

    if len(screenplays) < args.folds * 3:
        print(f"\nWARNING: only {len(screenplays)} labeled screenplays -- results at this "
              f"volume will be noisy. Treat as a preliminary result, not a final one.")

    y = np.array([sp["viability"]["imdb_rating"] for sp in screenplays])

    # Naive baseline: always predict the mean rating, no model at all.
    baseline_mae = mean_absolute_error(y, [y.mean()] * len(y))
    print(f"\nNaive baseline MAE (always guess the average): {baseline_mae:.3f}")

    X_base, vectorizer, _ = extract_features(screenplays, fit=True, include_genre=False)
    base_metrics = run_cross_validation(X_base, y, args.folds, "text + structure only")

    genre_metrics = None
    if not args.no_genre_compare:
        X_genre, _, mlb = extract_features(screenplays, fit=True, include_genre=True)
        genre_metrics = run_cross_validation(X_genre, y, args.folds, "text + structure + genre")

    def pct_improve(mae):
        return (baseline_mae - mae) / baseline_mae * 100

    print(f"\n{'='*60}")
    print(f"  Naive baseline MAE:                {baseline_mae:.3f}")
    print(f"  Text+structure MAE:                {base_metrics['mean_mae']:.3f}  "
          f"(R²={base_metrics['mean_r2']:.3f}, {pct_improve(base_metrics['mean_mae']):.1f}% over baseline)")
    if genre_metrics:
        print(f"  Text+structure+genre MAE:          {genre_metrics['mean_mae']:.3f}  "
              f"(R²={genre_metrics['mean_r2']:.3f}, {pct_improve(genre_metrics['mean_mae']):.1f}% over baseline)")
        genre_effect = base_metrics['mean_mae'] - genre_metrics['mean_mae']
        print(f"  Effect of adding genre:            {genre_effect:+.4f} MAE "
              f"({'helps' if genre_effect > 0 else 'no meaningful help' if abs(genre_effect) < 0.01 else 'hurts slightly'})")
    print(f"{'='*60}")

    # Final model -- use the genre-augmented version if we ran it (best
    # available), otherwise the base version
    use_genre_final = genre_metrics is not None
    X_final = X_genre if use_genre_final else X_base
    final_model = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
    final_model.fit(X_final, y)

    sample_predictions = []
    for i in range(min(8, len(screenplays))):
        pred = float(final_model.predict(X_final[i])[0])
        actual = float(y[i])
        sample_predictions.append({
            "title": screenplays[i]["title"],
            "actual_rating": actual,
            "predicted_rating": round(pred, 2),
            "error": round(abs(pred - actual), 2),
        })
        print(f"\n  {screenplays[i]['title']}: actual={actual}, predicted={pred:.2f}")

    joblib.dump({
        "model": final_model, "vectorizer": vectorizer,
        "includes_genre": use_genre_final, "genre_mlb": mlb if use_genre_final else None,
    }, OUT_DIR / "viability_model.joblib")

    report = {
        "screenplay_count": len(screenplays),
        "dropped_no_rating": dropped,
        "naive_baseline_mae": float(baseline_mae),
        "cv_folds": args.folds,
        "text_and_structure_only": base_metrics,
        "text_structure_and_genre": genre_metrics,
        "sample_predictions": sample_predictions,
    }
    (OUT_DIR / "viability_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved report to {OUT_DIR / 'viability_report.json'}")
    print(f"Saved model to {OUT_DIR / 'viability_model.joblib'}")


if __name__ == "__main__":
    main()