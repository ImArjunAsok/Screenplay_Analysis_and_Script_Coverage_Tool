"""
Week 6 -- Genre classification
----------------------------------
Predicts a screenplay's genre(s) from its text and structure. Multi-label
by design (a film can genuinely be both "Action" and "Comedy" -- IMSDB's
own genre pages already reflect this, a title can appear on several).

WHY TF-IDF + XGBoost, not a transformer: full screenplays are 20,000+
words -- far past what a BERT-style model can read in one pass (~500
words) without heavy chunking. And at current corpus sizes, a large
neural model would overfit badly. TF-IDF (word-frequency features) +
XGBoost is far more sample-efficient and is what your own proposal's
tech stack already specifies for this stage.

Features used, combined into one vector per screenplay:
  - TF-IDF over all scene text (captures word-choice patterns -- e.g.
    "blood"/"gun" cluster differently than "wedding"/"laugh")
  - scene_count, character_count, dialogue_count (raw structural signals)
  - dialogue_to_action_ratio (talk-heavy vs. description-heavy scripts
    tend to skew by genre -- dramas/comedies are more dialogue-heavy,
    action/horror often lean more on action description)

WHY CROSS-VALIDATION, NOT A SINGLE TRAIN/TEST SPLIT: with a small corpus,
one random split could easily land you a lucky (or unlucky) test set by
chance. K-fold cross-validation trains and evaluates K times on
different splits and averages the result -- a much more honest estimate
of real performance at this data size, and standard practice when data
is this limited.

WHY SOME GENRES GET DROPPED: a genre with only 1-2 example scripts can't
be meaningfully learned or evaluated -- the model would just be
memorizing single examples, and any "accuracy" number would be
meaningless. Genres below --min-examples (default 5) are excluded, and
exactly which ones is printed and saved, not hidden. "Short" is always
excluded regardless of count -- on IMSDB it means "this is a short film"
(a format), not a narrative genre, so it doesn't belong in this list at all.

WHY scale_pos_weight (added after the first real run on 1,117 scripts):
rare genres (Family, War, Musical...) were scoring exactly 0 -- not
because the model was confused, but because a classifier trained on,
say, 35 positive examples out of 1,117 can get 97%+ raw accuracy by just
always predicting "not this genre," so it has almost no pressure to ever
say "yes." scale_pos_weight tells XGBoost "a missed positive costs N
times more than a false positive," where N is that genre's actual
imbalance ratio (negatives/positives) -- computed separately per genre,
since Drama's imbalance (629/1117) is nothing like Western's (20/1117).
This needs a per-genre model rather than one shared MultiOutputClassifier,
since each genre needs its own weight.

Run:
    python nlp_pipeline/train_genre_classifier.py dataset/corpus_with_genres.jsonl
    python nlp_pipeline/train_genre_classifier.py dataset/corpus_with_genres.jsonl --min-examples 8

Outputs:
    dataset/genre_classifier_report.json   -- per-genre metrics, dropped
                                               genres, sample predictions
    dataset/genre_model.joblib              -- the trained model + vectorizer
                                               + label list, for later use
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
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

OUT_DIR = Path(__file__).parent.parent / "dataset"

# Not a real narrative genre -- IMSDB uses "Short" to mean "this is a
# short film" (a format/length tag), unrelated to story-type genres like
# Drama or Horror. Excluded unconditionally, not subject to --min-examples.
NON_GENRE_LABELS = {"Short"}


def load_data(corpus_path: str, min_examples: int):
    screenplays = [json.loads(l) for l in open(corpus_path, encoding="utf-8")]
    screenplays = [sp for sp in screenplays if sp.get("genres")]

    if not screenplays:
        print("No screenplays with genre labels found. Run fetch_genre_labels.py --join first.")
        sys.exit(1)

    from collections import Counter
    genre_counts = Counter(
        g for sp in screenplays for g in sp["genres"] if g not in NON_GENRE_LABELS
    )
    kept_genres = {g for g, c in genre_counts.items() if c >= min_examples}
    dropped_genres = {g: c for g, c in genre_counts.items() if c < min_examples}

    for sp in screenplays:
        sp["genres"] = [g for g in sp["genres"] if g in kept_genres]
    screenplays = [sp for sp in screenplays if sp["genres"]]

    return screenplays, sorted(kept_genres), dropped_genres


def extract_features(screenplays: list[dict], vectorizer: TfidfVectorizer = None, fit=True):
    texts = []
    structural = []
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

    if fit:
        vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", min_df=2)
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)

    structural = np.array(structural, dtype=float)
    # Normalize structural features roughly to TF-IDF's scale so they
    # don't get drowned out or dominate purely from having larger raw numbers
    structural = (structural - structural.mean(axis=0)) / (structural.std(axis=0) + 1e-6)

    X = hstack([tfidf, csr_matrix(structural)])
    return X, vectorizer


def compute_scale_pos_weight(y_col: np.ndarray) -> float:
    """negatives / positives for this one genre column. E.g. Western with
    20 positives out of 1117 -> weight ~54, meaning XGBoost treats missing
    a real Western as ~54x worse than a false positive. Without this, the
    model has almost no incentive to ever predict a rare genre at all."""
    positives = y_col.sum()
    negatives = len(y_col) - positives
    if positives == 0:
        return 1.0
    return float(negatives / positives)


def fit_weighted_multilabel(X_train, y_train, **xgb_kwargs) -> list[XGBClassifier]:
    """One XGBClassifier per genre column, each with its OWN
    scale_pos_weight -- this is why we can't use MultiOutputClassifier,
    which shares one fixed set of hyperparameters across every column."""
    models = []
    for i in range(y_train.shape[1]):
        col = y_train[:, i]
        spw = compute_scale_pos_weight(col)
        clf = XGBClassifier(scale_pos_weight=spw, **xgb_kwargs)
        clf.fit(X_train, col)
        models.append(clf)
    return models


def predict_weighted_multilabel(models: list[XGBClassifier], X) -> np.ndarray:
    return np.column_stack([m.predict(X) for m in models])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_path")
    ap.add_argument("--min-examples", type=int, default=5,
                     help="Minimum screenplays needed for a genre to be included (default 5)")
    ap.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    screenplays, genres, dropped = load_data(args.corpus_path, args.min_examples)

    print(f"Screenplays with usable genre labels: {len(screenplays)}")
    print(f"Genres kept (>= {args.min_examples} examples): {genres}")
    if dropped:
        print(f"Genres dropped (too few examples): {dropped}")

    if len(screenplays) < args.folds * 2:
        print(f"\nWARNING: only {len(screenplays)} labeled screenplays for {args.folds}-fold "
              f"cross-validation -- results will be noisy. Consider --folds 3 or scraping more data.")

    mlb = MultiLabelBinarizer(classes=genres)
    y = mlb.fit_transform([sp["genres"] for sp in screenplays])

    X, vectorizer = extract_features(screenplays, fit=True)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    fold_f1_scores = []
    all_true, all_pred = [], []

    print(f"\nRunning {args.folds}-fold cross-validation...")
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X.toarray()), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        xgb_kwargs = dict(n_estimators=100, max_depth=4, learning_rate=0.1,
                           eval_metric="logloss", random_state=42)
        models = fit_weighted_multilabel(X_train, y_train, **xgb_kwargs)
        y_pred = predict_weighted_multilabel(models, X_test)

        fold_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        fold_f1_scores.append(fold_f1)
        all_true.append(y_test)
        all_pred.append(y_pred)
        print(f"  Fold {fold_i}: macro F1 = {fold_f1:.3f}")

    y_true_all = np.vstack(all_true)
    y_pred_all = np.vstack(all_pred)

    report = classification_report(
        y_true_all, y_pred_all, target_names=genres, zero_division=0, output_dict=True
    )

    print(f"\n{'='*60}")
    print(f"  Mean macro F1 across {args.folds} folds: {np.mean(fold_f1_scores):.3f} "
          f"(+/- {np.std(fold_f1_scores):.3f})")
    print(f"{'='*60}")
    print(f"\n  Per-genre performance:")
    for g in genres:
        m = report[g]
        print(f"    {g:<12} precision={m['precision']:.2f}  recall={m['recall']:.2f}  "
              f"f1={m['f1-score']:.2f}  support={int(m['support'])}")

    # Train a final model on ALL data for actual future predictions, and
    # show sample predictions on a handful of real screenplays
    xgb_kwargs = dict(n_estimators=100, max_depth=4, learning_rate=0.1,
                       eval_metric="logloss", random_state=42)
    final_models = fit_weighted_multilabel(X, y, **xgb_kwargs)
    weights_used = {g: compute_scale_pos_weight(y[:, i]) for i, g in enumerate(genres)}

    sample_predictions = []
    sample_idx = list(range(min(5, len(screenplays))))
    for i in sample_idx:
        pred_row = predict_weighted_multilabel(final_models, X[i])[0]
        predicted_genres = [g for g, flag in zip(genres, pred_row) if flag]
        sample_predictions.append({
            "title": screenplays[i]["title"],
            "actual_genres": screenplays[i]["genres"],
            "predicted_genres": predicted_genres,
        })
        print(f"\n  Sample: {screenplays[i]['title']}")
        print(f"    Actual   : {screenplays[i]['genres']}")
        print(f"    Predicted: {predicted_genres}")

    joblib.dump({
        "models": final_models, "vectorizer": vectorizer, "genres": genres,
    }, OUT_DIR / "genre_model.joblib")

    report_out = {
        "screenplay_count": len(screenplays),
        "genres_kept": genres,
        "genres_dropped_insufficient_data": dropped,
        "scale_pos_weight_used": {g: round(w, 2) for g, w in weights_used.items()},
        "cv_folds": args.folds,
        "mean_macro_f1": float(np.mean(fold_f1_scores)),
        "std_macro_f1": float(np.std(fold_f1_scores)),
        "per_genre_metrics": {g: report[g] for g in genres},
        "sample_predictions": sample_predictions,
    }
    (OUT_DIR / "genre_classifier_report.json").write_text(json.dumps(report_out, indent=2))
    print(f"\nSaved report to {OUT_DIR / 'genre_classifier_report.json'}")
    print(f"Saved model to {OUT_DIR / 'genre_model.joblib'}")


if __name__ == "__main__":
    main()