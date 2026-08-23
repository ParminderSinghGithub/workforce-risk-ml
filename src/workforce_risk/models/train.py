"""Training pipeline, early stopping, and checkpoint management for StructuredMLP."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from workforce_risk.config import get_config
from workforce_risk.models.dataset import create_data_loaders
from workforce_risk.models.evaluate import evaluate_model
from workforce_risk.models.model import StructuredMLP
from workforce_risk.utils.seed import set_seed


def get_device(device_setting: str = "auto") -> torch.device:
    """Select compute device based on configuration and hardware availability."""
    setting = device_setting.lower().strip()
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif setting == "cpu":
        return torch.device("cpu")
    elif setting == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available on this system.")
        return torch.device("cuda")
    else:
        raise ValueError(f"Unknown device setting '{device_setting}'. Choose from 'auto', 'cpu', 'cuda'.")


def train_structured_model(
    train_path: Optional[str | Path] = None,
    val_path: Optional[str | Path] = None,
    test_path: Optional[str | Path] = None,
    artifacts_dir: Optional[str | Path] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: float = 1e-4,
    epochs: Optional[int] = None,
    patience: int = 5,
    hidden_dims: Optional[List[int]] = None,
    dropout: Optional[float] = None,
    device_str: str = "auto",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute end-to-end PyTorch training for the structured attrition model."""
    config = get_config()
    t0 = time.time()

    # Resolve defaults from config
    seed = seed if seed is not None else config.project.seed
    set_seed(seed)

    batch_size = batch_size if batch_size is not None else config.models.structured.batch_size
    learning_rate = learning_rate if learning_rate is not None else config.models.structured.learning_rate
    epochs = epochs if epochs is not None else config.models.structured.epochs
    hidden_dims = hidden_dims if hidden_dims is not None else config.models.structured.hidden_dims
    dropout = dropout if dropout is not None else config.models.structured.dropout

    splits_dir = Path(config.paths.data_splits_dir)
    train_path = Path(train_path or splits_dir / "structured_train.parquet").resolve()
    val_path = Path(val_path or splits_dir / "structured_validation.parquet").resolve()
    test_path = Path(test_path or splits_dir / "structured_test.parquet").resolve()

    artifacts_dir = Path(artifacts_dir or "artifacts/structured_model").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / "best_checkpoint.pt"
    history_path = artifacts_dir / "training_history.json"

    device = get_device(device_str)
    print(f"[Training] Initializing StructuredMLP on device: {device}")

    # 1. Create DataLoaders
    print(f"[Training] Loading DataLoaders (smoke sample limits: train={max_train_samples}, val={max_val_samples})...")
    train_loader, val_loader, test_loader, feature_names = create_data_loaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        random_seed=seed,
    )

    input_dim = len(feature_names)
    model = StructuredMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    print(f"[Training] Architecture: {input_dim} -> {hidden_dims} -> 1 (Total params: {model.total_parameters})")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 2. Training loop with early stopping
    best_val_roc_auc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    print(f"[Training] Starting training loop for {epochs} max epochs (patience={patience})...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            batch_sz = batch_features.size(0)
            running_train_loss += loss.item() * batch_sz
            train_samples += batch_sz

        avg_train_loss = running_train_loss / max(train_samples, 1)

        # Validation step
        val_loss, val_metrics, _, _ = evaluate_model(
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
        status_tag = "*" if improved else " "

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time}s] | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val ROC-AUC: {val_roc:.4f} | Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} {status_tag}"
        )

        if improved:
            best_val_roc_auc = val_roc
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best checkpoint
            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_dim": input_dim,
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                },
                "feature_names": feature_names,
                "training_config": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "seed": seed,
                },
                "best_epoch": best_epoch,
                "best_val_roc_auc": best_val_roc_auc,
                "val_metrics": val_metrics,
                "project_version": config.project.version,
            }
            torch.save(checkpoint_data, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Training] Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # 3. Reload best checkpoint and verify consistency
    print(f"[Training] Reloading best checkpoint from epoch {best_epoch} ({checkpoint_path})...")
    saved_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    best_model = StructuredMLP(
        input_dim=saved_ckpt["model_config"]["input_dim"],
        hidden_dims=saved_ckpt["model_config"]["hidden_dims"],
        dropout=saved_ckpt["model_config"]["dropout"],
    ).to(device)
    best_model.load_state_dict(saved_ckpt["model_state_dict"])
    best_model.eval()

    # Re-evaluate reloaded model on validation set to verify consistency
    _, reloaded_val_metrics, _, reloaded_val_probs = evaluate_model(
        model=best_model,
        data_loader=val_loader,
        criterion=criterion,
        device=device,
    )

    # Evaluate on final test set (untouched during training)
    test_loss, test_metrics, _, test_probs = evaluate_model(
        model=best_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"[Test Evaluation] Test Loss: {test_loss:.4f} | Test ROC-AUC: {test_metrics['roc_auc']:.4f} | Test PR-AUC: {test_metrics['pr_auc']:.4f} | Test F1: {test_metrics['f1']:.4f}")

    total_time = round(time.time() - t0, 2)

    # Save training history JSON
    history_data = {
        "model_type": "StructuredMLP",
        "total_parameters": model.total_parameters,
        "input_dim": input_dim,
        "device": str(device),
        "total_training_seconds": total_time,
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_val_roc_auc,
        "best_val_metrics": reloaded_val_metrics,
        "test_metrics": test_metrics,
        "epochs_history": history,
    }
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    return {
        "status": "SUCCESS",
        "device": str(device),
        "total_parameters": model.total_parameters,
        "input_dim": input_dim,
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_val_roc_auc,
        "reloaded_val_metrics": reloaded_val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "training_history_path": str(history_path),
        "elapsed_seconds": total_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PyTorch StructuredMLP Attrition Model")
    parser.add_argument("--smoke-test", action="store_true", help="Run bounded smoke test with 5k train / 1k val / 1k test")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    args = parser.parse_args()

    if args.smoke_test:
        print("[CLI] Executing bounded CPU smoke test (5,000 train / 1,000 val / 1,000 test)...")
        train_structured_model(
            max_train_samples=5000,
            max_val_samples=1000,
            max_test_samples=1000,
            epochs=args.epochs or 5,
            batch_size=args.batch_size or 128,
            learning_rate=args.lr or 0.001,
            device_str=args.device,
        )
    else:
        train_structured_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device_str=args.device,
        )
