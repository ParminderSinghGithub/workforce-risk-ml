"""GPU/CPU training pipeline and checkpointing for DistilBERT + PEFT/LoRA text classifier."""

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW

from workforce_risk.config import get_config
from workforce_risk.models.evaluate import find_optimal_threshold
from workforce_risk.models.train import get_device, print_startup_banner
from workforce_risk.nlp.dataset import create_text_data_loaders
from workforce_risk.nlp.evaluate import evaluate_text_transformer
from workforce_risk.nlp.model import (
    create_lora_text_classifier,
    get_trainable_parameters_summary,
    load_lora_text_model,
    save_lora_text_model,
)
from workforce_risk.utils.seed import set_seed


def train_text_transformer(
    train_path: Optional[str | Path] = None,
    val_path: Optional[str | Path] = None,
    test_path: Optional[str | Path] = None,
    artifacts_dir: Optional[str | Path] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    base_model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: float = 0.01,
    epochs: Optional[int] = None,
    patience: int = 3,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    device_str: str = "auto",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute end-to-end training for DistilBERT with LoRA adapters."""
    config = get_config()
    t0 = time.time()
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    seed = seed if seed is not None else config.project.seed
    set_seed(seed)

    batch_size = batch_size if batch_size is not None else config.models.text.batch_size
    learning_rate = learning_rate if learning_rate is not None else config.models.text.learning_rate
    epochs = epochs if epochs is not None else config.models.text.epochs

    splits_dir = Path(config.paths.data_splits_dir)
    train_path = Path(train_path or splits_dir / "text_train.parquet").resolve()
    val_path = Path(val_path or splits_dir / "text_validation.parquet").resolve()
    test_path = Path(test_path or splits_dir / "text_test.parquet").resolve()

    artifacts_dir = Path(artifacts_dir or "artifacts/text_transformer").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = artifacts_dir / "best_model"
    history_path = artifacts_dir / "training_history.json"
    summary_path = artifacts_dir / "evaluation_summary.json"

    # 1. Device selection
    device, device_info = get_device(device_str)
    print_startup_banner(device_info, max_train_samples, batch_size)

    # 2. DataLoaders
    print(f"[Training] Loading Text DataLoaders (limits: train={max_train_samples}, val={max_val_samples})...")
    train_loader, val_loader, test_loader, tokenizer = create_text_data_loaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        tokenizer_name=base_model_name,
        max_length=max_length,
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        random_seed=seed,
    )

    actual_train_rows = len(train_loader.dataset)
    actual_val_rows = len(val_loader.dataset)
    actual_test_rows = len(test_loader.dataset)

    print(f"[Training] Datasets: Train={actual_train_rows:,} | Val={actual_val_rows:,} | Test={actual_test_rows:,}")

    # 3. Model with LoRA
    print(f"[Training] Initializing DistilBERT with LoRA (r={lora_r}, alpha={lora_alpha})...")
    model = create_lora_text_classifier(
        base_model_name=base_model_name,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_labels=1,
    ).to(device)

    param_summary = get_trainable_parameters_summary(model)
    print(f"[Training] Trainable Params: {param_summary['trainable_params']:,} / {param_summary['all_params']:,} ({param_summary['trainable_percent']}%)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 4. Training Loop
    best_val_roc_auc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    print(f"[Training] Starting training for {epochs} max epochs (patience={patience})...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=(device.type == "cuda"))
            attention_mask = batch["attention_mask"].to(device, non_blocking=(device.type == "cuda"))
            targets = batch["labels"].to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            batch_sz = input_ids.size(0)
            running_train_loss += loss.item() * batch_sz
            train_samples += batch_sz

        avg_train_loss = running_train_loss / max(train_samples, 1)

        # Validation Step
        val_loss, val_metrics, _, _ = evaluate_text_transformer(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = round(time.time() - epoch_start, 2)
        val_roc = val_metrics["roc_auc"]

        history_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_roc_auc": val_roc,
            "val_pr_auc": val_metrics["pr_auc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "epoch_seconds": epoch_time,
        }
        history.append(history_entry)

        improved = val_roc > best_val_roc_auc
        status_tag = "[BEST]" if improved else ""

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:4.1f}s] | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val ROC-AUC: {val_roc:.4f} | Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} {status_tag}"
        )

        if improved:
            best_val_roc_auc = val_roc
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best LoRA model and tokenizer
            save_lora_text_model(
                model=model,
                tokenizer=tokenizer,
                output_dir=checkpoint_dir,
                training_config={
                    "base_model": base_model_name,
                    "max_length": max_length,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "best_epoch": best_epoch,
                    "best_val_roc_auc": best_val_roc_auc,
                    "seed": seed,
                },
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Training] Early stopping triggered at epoch {epoch}.")
                break

    # 5. Reload Best Checkpoint and Evaluate on Test Set
    print(f"[Training] Reloading best LoRA model from {checkpoint_dir}...")
    best_model, _ = load_lora_text_model(checkpoint_dir, device=device)

    _, reloaded_val_metrics, val_targets, val_probs = evaluate_text_transformer(
        model=best_model,
        data_loader=val_loader,
        criterion=criterion,
        device=device,
    )

    best_thresh, best_val_f1 = find_optimal_threshold(val_targets, val_probs, metric="f1")

    # Evaluate on final holdout test set at default and optimal thresholds
    test_loss_default, test_metrics_default, test_targets, test_probs = evaluate_text_transformer(
        model=best_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=0.5,
    )
    _, test_metrics_optimal, _, _ = evaluate_text_transformer(
        model=best_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=best_thresh,
    )

    print("-" * 72)
    print(f"[Test Evaluation] ROC-AUC: {test_metrics_default['roc_auc']:.4f} | PR-AUC: {test_metrics_default['pr_auc']:.4f}")
    print(f"[Test Evaluation] Default F1 (t=0.50): {test_metrics_default['f1']:.4f} | Optimal F1 (t={best_thresh:.2f}): {test_metrics_optimal['f1']:.4f}")
    print("-" * 72)

    total_time = round(time.time() - t0, 2)

    # 6. Save comprehensive history and summary JSON
    history_data = {
        "experiment": "DistilBERT_LoRA_Burnout_Risk",
        "timestamp_utc": timestamp_utc,
        "environment": device_info,
        "dataset_sizes": {
            "train_rows": actual_train_rows,
            "val_rows": actual_val_rows,
            "test_rows": actual_test_rows,
        },
        "model_architecture": {
            "base_model": base_model_name,
            "peft_type": "LoRA",
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "parameter_summary": param_summary,
        },
        "hyperparameters": {
            "max_length": max_length,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "patience": patience,
            "seed": seed,
        },
        "results": {
            "best_epoch": best_epoch,
            "best_val_roc_auc": best_val_roc_auc,
            "optimal_val_threshold": best_thresh,
            "optimal_val_f1": best_val_f1,
            "val_metrics": reloaded_val_metrics,
            "test_metrics_default_0_5": test_metrics_default,
            "test_metrics_optimal_thresh": test_metrics_optimal,
            "total_seconds": total_time,
        },
        "epochs_history": history,
    }

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    summary_data = {
        "timestamp_utc": timestamp_utc,
        "device": device_info["selected_device"],
        "train_rows": actual_train_rows,
        "best_epoch": best_epoch,
        "val_roc_auc": best_val_roc_auc,
        "test_roc_auc": test_metrics_default["roc_auc"],
        "test_pr_auc": test_metrics_default["pr_auc"],
        "test_f1_optimal": test_metrics_optimal["f1"],
        "checkpoint_dir": str(checkpoint_dir),
        "total_seconds": total_time,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[Artifacts] Best LoRA model saved: {checkpoint_dir}")
    print(f"[Artifacts] Training history saved:  {history_path}")
    print(f"[Artifacts] Evaluation summary saved:{summary_path}")

    return {
        "status": "SUCCESS",
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_val_roc_auc,
        "test_roc_auc": test_metrics_default["roc_auc"],
        "test_pr_auc": test_metrics_default["pr_auc"],
        "checkpoint_dir": str(checkpoint_dir),
        "total_seconds": total_time,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for transformer text training."""
    parser = argparse.ArgumentParser(description="Train DistilBERT + LoRA Text Risk Classifier")
    parser.add_argument("--smoke-test", action="store_true", help="Run bounded local smoke test (e.g. 200 train / 50 val / 50 test / 1 epoch)")
    parser.add_argument("--train-sample-size", type=int, default=None, help="Number of training samples")
    parser.add_argument("--val-sample-size", type=int, default=None, help="Number of validation samples")
    parser.add_argument("--test-sample-size", type=int, default=None, help="Number of test samples")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence tokenization length")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Artifacts output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.smoke_test:
        print("[CLI] Executing bounded CPU smoke test (200 train / 50 val / 50 test / 1 epoch)...")
        train_text_transformer(
            max_train_samples=args.train_sample_size or 200,
            max_val_samples=args.val_sample_size or 50,
            max_test_samples=args.test_sample_size or 50,
            max_length=args.max_length,
            batch_size=args.batch_size or 16,
            epochs=args.epochs or 1,
            learning_rate=args.lr or 2e-4,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            device_str=args.device,
            artifacts_dir=args.artifacts_dir or "artifacts/text_transformer_smoke",
        )
    else:
        train_text_transformer(
            max_train_samples=args.train_sample_size,
            max_val_samples=args.val_sample_size,
            max_test_samples=args.test_sample_size,
            max_length=args.max_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            device_str=args.device,
            artifacts_dir=args.artifacts_dir,
        )
