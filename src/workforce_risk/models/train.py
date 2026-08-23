"""GPU-ready training pipeline, early stopping, checkpointing, and experiment tracking."""

import argparse
import datetime
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW

from workforce_risk.config import get_config
from workforce_risk.features.definitions import FEATURE_DEFINITIONS
from workforce_risk.models.dataset import create_data_loaders
from workforce_risk.models.evaluate import evaluate_model
from workforce_risk.models.model import StructuredMLP
from workforce_risk.models.preprocessor import TabularPreprocessor
from workforce_risk.utils.seed import set_seed


def get_git_commit_hash() -> str:
    """Attempt to retrieve current git commit hash, returning fallback if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unversioned"


def get_device(device_setting: str = "auto") -> Tuple[torch.device, Dict[str, Any]]:
    """Select compute device and extract comprehensive hardware/CUDA metadata."""
    setting = device_setting.lower().strip()
    cuda_available = torch.cuda.is_available()

    if setting == "auto":
        target_device = torch.device("cuda" if cuda_available else "cpu")
    elif setting == "cpu":
        target_device = torch.device("cpu")
    elif setting == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA device requested (--device cuda) but CUDA is not available on this system.")
        target_device = torch.device("cuda")
    else:
        raise ValueError(f"Unknown device setting '{device_setting}'. Choose from 'auto', 'cpu', 'cuda'.")

    gpu_name = torch.cuda.get_device_name(0) if cuda_available and target_device.type == "cuda" else None
    cuda_version = torch.version.cuda if cuda_available else None
    gpu_count = torch.cuda.device_count() if cuda_available else 0

    device_info = {
        "selected_device": str(target_device),
        "device_type": target_device.type,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    return target_device, device_info


def print_startup_banner(device_info: Dict[str, Any], sample_size: Optional[int], batch_size: int) -> None:
    """Print structured startup information for execution logs (Kaggle & local)."""
    print("=" * 72)
    print(" WORKFORCE RISK ML SYSTEM -- PYTORCH STRUCTURED MODEL TRAINING")
    print("=" * 72)
    print(f" PyTorch Version:      {device_info['pytorch_version']}")
    print(f" Python Version:       {device_info['python_version']} ({device_info['platform']})")
    print(f" CUDA Available:       {device_info['cuda_available']}")
    if device_info["cuda_available"]:
        print(f" CUDA Version:         {device_info['cuda_version']}")
        print(f" GPU Device:           {device_info['gpu_name']} ({device_info['gpu_count']} device(s))")
    print(f" Selected Device:      {device_info['selected_device']}")
    print(f" Train Sample Target:  {sample_size if sample_size is not None else 'Full Partition'}")
    print(f" Mini-Batch Size:      {batch_size}")
    print("=" * 72)


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
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1. Resolve configuration parameters and seed
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
    evaluation_summary_path = artifacts_dir / "evaluation_summary.json"

    # 2. Select compute device and display environment banner
    device, device_info = get_device(device_str)
    print_startup_banner(device_info, max_train_samples, batch_size)

    # 3. Create DataLoaders: fit preprocessor on COMPLETE train partition, then apply sampling
    print(f"[Training] Fitting preprocessor on complete training partition ({train_path})...")
    train_loader, val_loader, test_loader, preprocessor = create_data_loaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
        fit_on_full_train=True,
        random_seed=seed,
    )

    actual_train_rows = len(train_loader.dataset)
    actual_val_rows = len(val_loader.dataset)
    actual_test_rows = len(test_loader.dataset)
    input_dim = preprocessor.feature_dim

    print(f"[Training] Datasets loaded: Train={actual_train_rows:,} | Val={actual_val_rows:,} | Test={actual_test_rows:,}")
    print(f"[Training] Encoded Features: {input_dim} (24 continuous + {input_dim - 24} one-hot categorical)")

    # 4. Instantiate Model, Loss, and Optimizer
    model = StructuredMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    print(f"[Training] Model Architecture: {input_dim} -> {hidden_dims} -> 1 (Total params: {model.total_parameters:,})")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 5. Training loop with early stopping on validation ROC-AUC
    best_val_roc_auc = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    print(f"[Training] Beginning training for {epochs} max epochs (early stopping patience={patience})...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device, non_blocking=(device.type == "cuda"))
            batch_targets = batch_targets.to(device, non_blocking=(device.type == "cuda"))

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

            # Save best checkpoint with full self-contained contract
            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_dim": input_dim,
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                },
                "preprocessing_config": preprocessor.to_dict(),
                "encoded_feature_names": preprocessor.encoded_feature_names,
                "raw_feature_names": list(FEATURE_DEFINITIONS.keys()),
                "training_config": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "seed": seed,
                    "train_samples": actual_train_rows,
                },
                "best_epoch": best_epoch,
                "best_val_roc_auc": best_val_roc_auc,
                "val_metrics": val_metrics,
                "project_version": config.project.version,
                "timestamp_utc": timestamp_utc,
            }
            torch.save(checkpoint_data, checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Training] Early stopping triggered at epoch {epoch} (no improvement for {patience} consecutive epochs).")
                break

    # 6. Reload best checkpoint from disk and evaluate on holdout test set
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
    _, reloaded_val_metrics, _, _ = evaluate_model(
        model=best_model,
        data_loader=val_loader,
        criterion=criterion,
        device=device,
    )

    # Evaluate on final test set (strictly untouched during training decisions)
    test_loss, test_metrics, _, _ = evaluate_model(
        model=best_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("-" * 72)
    print(f"[Test Evaluation] Test Loss: {test_loss:.4f} | Test ROC-AUC: {test_metrics['roc_auc']:.4f} | Test PR-AUC: {test_metrics['pr_auc']:.4f} | Test F1: {test_metrics['f1']:.4f}")
    print("-" * 72)

    total_time = round(time.time() - t0, 2)
    git_hash = get_git_commit_hash()

    # 7. Save comprehensive training history and metadata JSON
    history_data = {
        "experiment": "StructuredMLP_Attrition_Prediction",
        "timestamp_utc": timestamp_utc,
        "git_commit": git_hash,
        "environment": device_info,
        "dataset_sizes": {
            "train_rows": actual_train_rows,
            "validation_rows": actual_val_rows,
            "test_rows": actual_test_rows,
        },
        "model_architecture": {
            "model_type": "StructuredMLP",
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "total_parameters": model.total_parameters,
        },
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_epochs": epochs,
            "early_stopping_patience": patience,
            "seed": seed,
        },
        "training_results": {
            "total_training_seconds": total_time,
            "best_epoch": best_epoch,
            "best_val_roc_auc": best_val_roc_auc,
            "best_val_metrics": reloaded_val_metrics,
            "test_metrics": test_metrics,
        },
        "epochs_history": history,
    }

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    # Save summary artifact for quick evaluation review
    summary_data = {
        "timestamp_utc": timestamp_utc,
        "device": device_info["selected_device"],
        "train_rows": actual_train_rows,
        "best_epoch": best_epoch,
        "val_roc_auc": best_val_roc_auc,
        "val_pr_auc": reloaded_val_metrics["pr_auc"],
        "val_f1": reloaded_val_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_f1": test_metrics["f1"],
        "test_loss": test_metrics["loss"],
        "checkpoint_path": str(checkpoint_path),
        "total_seconds": total_time,
    }
    with open(evaluation_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[Artifacts] Checkpoint saved: {checkpoint_path} ({checkpoint_path.stat().st_size:,} bytes)")
    print(f"[Artifacts] Training history: {history_path}")
    print(f"[Artifacts] Evaluation summary: {evaluation_summary_path}")

    return {
        "status": "SUCCESS",
        "device": device_info["selected_device"],
        "total_parameters": model.total_parameters,
        "input_dim": input_dim,
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_val_roc_auc,
        "reloaded_val_metrics": reloaded_val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "training_history_path": str(history_path),
        "evaluation_summary_path": str(evaluation_summary_path),
        "elapsed_seconds": total_time,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training script."""
    parser = argparse.ArgumentParser(description="Train PyTorch StructuredMLP Attrition Model")
    parser.add_argument("--train-path", type=str, default=None, help="Path to structured train parquet dataset")
    parser.add_argument("--val-path", type=str, default=None, help="Path to structured validation parquet dataset")
    parser.add_argument("--test-path", type=str, default=None, help="Path to structured test parquet dataset")
    parser.add_argument("--train-sample-size", type=int, default=None, help="Subsample count for training (e.g. 100000)")
    parser.add_argument("--val-sample-size", type=int, default=None, help="Subsample count for validation (default: full)")
    parser.add_argument("--test-sample-size", type=int, default=None, help="Subsample count for test (default: full)")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size (default: 256)")
    parser.add_argument("--epochs", type=int, default=None, help="Maximum number of training epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=None, help="AdamW learning rate (default: 0.001)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: auto, cpu, cuda (default: auto)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Directory to store checkpoints and reports")
    parser.add_argument("--smoke-test", action="store_true", help="Run bounded local smoke test (5k train / 1k val / 1k test / 5 epochs)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.smoke_test:
        print("[CLI] Executing bounded CPU smoke test (5,000 train / 1,000 val / 1,000 test)...")
        train_structured_model(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            artifacts_dir=args.artifacts_dir,
            max_train_samples=args.train_sample_size or 5000,
            max_val_samples=args.val_sample_size or 1000,
            max_test_samples=args.test_sample_size or 1000,
            epochs=args.epochs or 5,
            batch_size=args.batch_size or 128,
            learning_rate=args.lr or 0.001,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device_str=args.device,
            seed=args.seed,
        )
    else:
        train_structured_model(
            train_path=args.train_path,
            val_path=args.val_path,
            test_path=args.test_path,
            artifacts_dir=args.artifacts_dir,
            max_train_samples=args.train_sample_size,
            max_val_samples=args.val_sample_size,
            max_test_samples=args.test_sample_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device_str=args.device,
            seed=args.seed,
        )
