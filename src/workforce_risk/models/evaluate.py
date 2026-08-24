"""Evaluation metrics, threshold sweeps, and evaluation loop for structured classification models."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate quantitative classification performance metrics.

    Args:
        y_true: Ground truth binary target array [N] or [N, 1].
        y_prob: Predicted probability array in [0, 1] [N] or [N, 1].
        threshold: Decision threshold for discrete classification (default: 0.5).

    Returns:
        Dictionary containing ROC-AUC, PR-AUC, Precision, Recall, F1, and Log-Loss.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Clip probabilities to prevent log(0)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    # Check for single-class edge cases
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_true))
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        bce_loss = float(log_loss(y_true, y_prob_clipped))
    except Exception:
        bce_loss = 0.0

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "loss": round(bce_loss, 4),
        "threshold": round(threshold, 4),
    }


def evaluate_threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float] | np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Evaluate precision, recall, F1, and predicted positive counts across a threshold grid.

    Args:
        y_true: Ground truth binary array.
        y_prob: Continuous predicted probabilities in [0, 1].
        thresholds: List or array of probability thresholds (default: 0.10 to 0.90 in steps of 0.01).

    Returns:
        List of metric dictionaries for each threshold.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 81)

    results: List[Dict[str, Any]] = []
    for t in thresholds:
        t_val = float(t)
        y_pred = (y_prob >= t_val).astype(int)
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        pos_pct = float(y_pred.mean() * 100)

        results.append({
            "threshold": round(t_val, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "pred_pos_pct": round(pos_pct, 2),
            "pred_count": int(y_pred.sum()),
        })

    return results


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    num_thresholds: int = 100,
) -> Tuple[float, float]:
    """Find decision threshold that optimizes target metric (e.g. F1) on validation data.

    Args:
        y_true: Ground truth binary target array.
        y_prob: Predicted probability array.
        metric: Optimization criterion ('f1', 'precision', 'recall').
        num_thresholds: Granularity of threshold search grid.

    Returns:
        Tuple of (optimal_threshold, best_metric_score).
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    p_min = max(0.001, float(y_prob.min()))
    p_max = min(0.999, float(y_prob.max()))
    if p_min >= p_max:
        return 0.5, 0.0

    thresholds = np.linspace(p_min, p_max, num_thresholds)
    best_threshold = 0.5
    best_score = -1.0

    for t in thresholds:
        t_val = float(t)
        y_pred = (y_prob >= t_val).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true, y_pred, zero_division=0))
        elif metric == "precision":
            score = float(precision_score(y_true, y_pred, zero_division=0))
        elif metric == "recall":
            score = float(recall_score(y_true, y_pred, zero_division=0))
        else:
            raise ValueError(f"Unsupported metric '{metric}' for threshold optimization.")

        if score > best_score:
            best_score = score
            best_threshold = t_val

    return round(best_threshold, 4), round(best_score, 4)


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Execute evaluation loop over DataLoader and calculate comprehensive metrics.

    Args:
        model: PyTorch model module.
        data_loader: Evaluation DataLoader.
        criterion: Loss function (e.g. BCEWithLogitsLoss).
        device: Active torch device (CPU or CUDA).
        threshold: Classification decision boundary.

    Returns:
        Tuple of (average_loss, metrics_dict, y_true_array, y_prob_array).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch_features, batch_targets in data_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            logits = model(batch_features)
            loss = criterion(logits, batch_targets)

            batch_size = batch_features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.sigmoid(logits)

            all_targets.append(batch_targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    y_true = np.vstack(all_targets).ravel()
    y_prob = np.vstack(all_probs).ravel()

    metrics = calculate_classification_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = round(avg_loss, 4)

    return avg_loss, metrics, y_true, y_prob
