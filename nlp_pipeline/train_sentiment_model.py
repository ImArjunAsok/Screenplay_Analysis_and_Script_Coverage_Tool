"""
Week 3 -- Fine-tune RoBERTa for sentiment
-------------------------------------------
Fine-tunes roberta-base on a COMBINATION of two datasets: HuggingFace's
Rotten Tomatoes (film-domain critic snippets) and the Stanford Large
Movie Review Dataset ("stanfordnlp/imdb" on HuggingFace -- the same data as the
popular "IMDB Dataset of 50K Movie Reviews" on Kaggle, just accessed
directly instead of via a manual CSV download).

WHY ADD IMDB, SPECIFICALLY: not just "more data." Rotten Tomatoes'
snippets are short, single sentences. IMDB reviews are full paragraphs
-- much closer in LENGTH to an actual screenplay scene (which mixes
several sentences of action and dialogue) than a single critic
sentence is. This directly targets a documented weakness: the
placeholder/RT-only model scored noisily on long, sparse-dialogue
scenes (127 Hours) but cleanly on short, conversational ones (A Real
Pain) -- consistent with training data length mismatch. Longer,
paragraph-scale training examples are a better length match for what
the model actually has to score.

WHAT THIS DOESN'T FIX: IMDB reviews are still REVIEW/OPINION text about
a film, not narrative prose describing action inside a story. This
narrows the domain gap, it doesn't close it -- worth stating plainly
rather than overselling.

Both datasets share the same schema (a "text" column, a "label" column,
0=negative/1=positive), so they concatenate directly with no relabeling.
Rotten Tomatoes' own validation/test sets are kept as the PRIMARY
benchmark (backward-comparable to earlier training runs). IMDB's test
set is evaluated separately as a SECOND benchmark, to confirm adding it
to training didn't come at the cost of doing worse on either domain.

This is meant to run on Colab with a GPU (per your proposal's plan) --
fine-tuning roberta-base on CPU is painfully slow. The script checks for a
GPU and warns loudly if it doesn't find one.

NOTE ON TRAINING TIME: the combined training set is roughly 4x larger
than the Rotten-Tomatoes-only version (~33,500 vs ~8,500 examples), so
expect training to take roughly 4x as long -- around 20-25 minutes on a
T4 GPU instead of ~6, based on the earlier RT-only run. Still very
manageable on Colab, just don't expect the same ~6-minute turnaround.

Run:
    python nlp_pipeline/train_sentiment_model.py
    python nlp_pipeline/train_sentiment_model.py --epochs 4 --batch-size 32
    python nlp_pipeline/train_sentiment_model.py --rt-only   # old behaviour, for comparison

Outputs:
    models/roberta-sentiment-finetuned/   -- the fine-tuned model + tokenizer
    models/training_log.json               -- per-epoch loss/accuracy/F1,
                                               hyperparameters, dataset
                                               composition, and BOTH test
                                               benchmarks (RT and IMDB)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "roberta-base"
OUTPUT_DIR = Path(__file__).parent.parent / "models" / "roberta-sentiment-finetuned"
LOG_FILE = Path(__file__).parent.parent / "models" / "training_log.json"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rt-only", action="store_true",
                     help="Train on Rotten Tomatoes alone (old behaviour), for comparison")
    args = ap.parse_args()

    if torch.cuda.is_available():
        print(f"GPU found: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "\n*** WARNING: no GPU detected. ***\n"
            "Fine-tuning roberta-base on CPU will take a very long time.\n"
            "On Colab: Runtime -> Change runtime type -> GPU, then re-run.\n"
        )

    print(f"\nLoading Rotten Tomatoes dataset...")
    rt = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    print(f"  train: {len(rt['train'])}  validation: {len(rt['validation'])}  test: {len(rt['test'])}")

    if args.rt_only:
        train_data = rt["train"]
        dataset_desc = "rotten_tomatoes only"
        imdb_test = None
    else:
        print(f"\nLoading IMDB dataset (Stanford Large Movie Review Dataset)...")
        imdb = load_dataset("stanfordnlp/imdb")
        print(f"  train: {len(imdb['train'])}  test: {len(imdb['test'])}")

        rt_train = rt["train"].remove_columns([c for c in rt["train"].column_names if c not in ("text", "label")])
        imdb_train = imdb["train"].remove_columns([c for c in imdb["train"].column_names if c not in ("text", "label")])
        train_data = concatenate_datasets([rt_train, imdb_train]).shuffle(seed=42)
        imdb_test = imdb["test"]
        dataset_desc = "rotten_tomatoes + imdb (combined)"

        print(f"\nCombined training set: {len(rt_train)} (RT) + {len(imdb_train)} (IMDB) "
              f"= {len(train_data)} total")

    print(f"\nLoading tokenizer and model ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def tokenize(batch):
        # Standard tokenizer truncation (proper subword-aware cutoff),
        # not the crude character-slicing the placeholder model used.
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    def prep(ds):
        t = ds.map(tokenize, batched=True)
        t = t.rename_column("label", "labels")
        t.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return t

    tokenized_train = prep(train_data)
    tokenized_val = prep(rt["validation"])
    tokenized_test = prep(rt["test"])
    tokenized_imdb_test = prep(imdb_test) if imdb_test is not None else None

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,   # RT validation throughout -- keeps
                                       # "best model" selection comparable
                                       # to earlier RT-only training runs
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"Training finished in {elapsed / 60:.1f} minutes.")

    print("\nEvaluating on Rotten Tomatoes test set (primary benchmark)...")
    rt_test_metrics = trainer.evaluate(tokenized_test)
    print(f"  RT test accuracy: {rt_test_metrics['eval_accuracy']:.4f}")
    print(f"  RT test F1      : {rt_test_metrics['eval_f1']:.4f}")

    imdb_test_metrics = None
    if tokenized_imdb_test is not None:
        print("\nEvaluating on IMDB test set (secondary benchmark)...")
        imdb_test_metrics = trainer.evaluate(tokenized_imdb_test)
        print(f"  IMDB test accuracy: {imdb_test_metrics['eval_accuracy']:.4f}")
        print(f"  IMDB test F1      : {imdb_test_metrics['eval_f1']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nModel saved to {OUTPUT_DIR}")

    log = {
        "model_name": MODEL_NAME,
        "dataset": dataset_desc,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": 256,
        },
        "train_examples": len(train_data),
        "validation_examples": len(rt["validation"]),
        "rt_test_examples": len(rt["test"]),
        "imdb_test_examples": len(imdb_test) if imdb_test is not None else None,
        "training_time_minutes": round(elapsed / 60, 2),
        "gpu_used": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)",
        "epoch_log": trainer.state.log_history,   # per-epoch/step loss + eval metrics
        "final_rt_test_metrics": rt_test_metrics,
        "final_imdb_test_metrics": imdb_test_metrics,
    }
    LOG_FILE.write_text(json.dumps(log, indent=2))
    print(f"Training log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()