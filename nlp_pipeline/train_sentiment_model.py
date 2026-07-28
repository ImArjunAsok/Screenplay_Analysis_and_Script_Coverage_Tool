"""
Week 3 -- Fine-tune RoBERTa for sentiment
-------------------------------------------
Fine-tunes roberta-base on the HuggingFace Rotten Tomatoes dataset
(film-domain critic snippets, binary pos/neg) so the sentiment-arc module
uses a model trained on film language instead of the generic SST-2
placeholder from earlier weeks.

This is meant to run on Colab with a GPU (per your proposal's plan) --
fine-tuning roberta-base on CPU is painfully slow. The script checks for a
GPU and warns loudly if it doesn't find one.

Run:
    python nlp_pipeline/train_sentiment_model.py
    python nlp_pipeline/train_sentiment_model.py --epochs 4 --batch-size 32

Outputs:
    models/roberta-sentiment-finetuned/   -- the fine-tuned model + tokenizer
    models/training_log.json               -- per-epoch loss/accuracy/F1,
                                               hyperparameters, and final test
                                               metrics. This file IS the
                                               "recorded training results"
                                               evidence for Week 3.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
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
    dataset = load_dataset("rotten_tomatoes")
    print(f"  train: {len(dataset['train'])}  validation: {len(dataset['validation'])}  test: {len(dataset['test'])}")

    print(f"\nLoading tokenizer and model ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def tokenize(batch):
        # Standard tokenizer truncation (proper subword-aware cutoff),
        # not the crude character-slicing the placeholder model used.
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

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
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...")
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"Training finished in {elapsed / 60:.1f} minutes.")

    print("\nEvaluating on held-out test set...")
    test_metrics = trainer.evaluate(tokenized["test"])
    print(f"  Test accuracy: {test_metrics['eval_accuracy']:.4f}")
    print(f"  Test F1      : {test_metrics['eval_f1']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"\nModel saved to {OUTPUT_DIR}")

    log = {
        "model_name": MODEL_NAME,
        "dataset": "rotten_tomatoes",
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": 256,
        },
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "test_examples": len(dataset["test"]),
        "training_time_minutes": round(elapsed / 60, 2),
        "gpu_used": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)",
        "epoch_log": trainer.state.log_history,   # per-epoch/step loss + eval metrics
        "final_test_metrics": test_metrics,
    }
    LOG_FILE.write_text(json.dumps(log, indent=2))
    print(f"Training log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()