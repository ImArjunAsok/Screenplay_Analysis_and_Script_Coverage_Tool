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