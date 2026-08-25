"""Inference evaluation, metrics calculation, and visualization plotting for Multimodal Late Fusion."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from workforce_risk.models.evaluate import calculate_classification_metrics, evaluate_threshold_sweep, find_optimal_threshold
from workforce_risk.models.model import StructuredMLP
from workforce_risk.models.preprocessor import TabularPreprocessor


def generate_unimodal_predictions(
    df: pd.DataFrame,
    structured_model: StructuredMLP,
    preprocessor: TabularPreprocessor,
    text_model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate aligned inference probabilities from both unimodal models for a dataset DataFrame.

    Args:
        df: Aligned DataFrame containing tabular features and `recent_feedback`.
        structured_model: Evaluated PyTorch StructuredMLP instance.
        preprocessor: Fitted TabularPreprocessor.
        text_model: Fine-tuned DistilBERT + LoRA model instance.
        tokenizer: Pre-trained DistilBERT tokenizer.
        device: Active torch device (CPU or CUDA).
        batch_size: Mini-batch size for text tokenization.
        max_length: Max sequence length for tokenizer.

    Returns:
        Tuple of (p_structured, p_text, y_true).
    """
    structured_model.eval()
    text_model.eval()

    # 1. Structured Tabular Inference
    print(f"[Inference] Processing {len(df):,} structured records...")
    X_struct = preprocessor.transform(df)
    X_tensor = torch.tensor(X_struct, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits_struct = structured_model(X_tensor)
        p_struct = torch.sigmoid(logits_struct).cpu().numpy().ravel()

    # 2. Text NLP Inference in Mini-Batches
    print(f"[Inference] Processing {len(df):,} feedback text reviews...")
    texts = df["recent_feedback"].tolist()
    p_text_list = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
            logits_text = outputs.logits
            probs = torch.sigmoid(logits_text).cpu().numpy().ravel()
            p_text_list.append(probs)

    p_text = np.concatenate(p_text_list).ravel()
    y_true = df["left_company"].to_numpy().astype(int)

    return p_struct, p_text, y_true


def generate_fusion_plots(
    y_true: np.ndarray,
    p_struct: np.ndarray,
    p_text: np.ndarray,
    p_fusion: np.ndarray,
    optimal_threshold: float,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Generate high-resolution benchmark plots substantiating multimodal fusion evidence.

    Creates:
    - roc_curve.png: Comparative ROC curves with AUC annotations.
    - precision_recall_curve.png: Comparative Precision-Recall curves.
    - confusion_matrix.png: Heatmap matrix at optimal decision threshold.
    - calibration_curve.png: Empirical reliability diagram across probability deciles.
    """
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    generated_plots = {}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # =========================================================================
    # 1. Comparative ROC Curves
    # =========================================================================
    plt.figure(figsize=(8, 6), dpi=300)
    for name, probs, color in [
        ("Structured Tabular", p_struct, "#1f77b4"),
        ("DistilBERT Text", p_text, "#ff7f0e"),
        ("Multimodal Fused", p_fusion, "#2ca02c"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, probs)
        score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {score:.4f})", color=color, linewidth=2.2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.5000)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    plt.title("Multimodal Late Fusion vs Unimodal ROC Curves (Holdout Test)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    roc_plot_file = out_path / "roc_curve.png"
    plt.savefig(roc_plot_file)
    plt.close()
    generated_plots["roc_curve"] = str(roc_plot_file)

    # =========================================================================
    # 2. Comparative Precision-Recall Curves
    # =========================================================================
    plt.figure(figsize=(8, 6), dpi=300)
    baseline_pr = float(y_true.mean())
    for name, probs, color in [
        ("Structured Tabular", p_struct, "#1f77b4"),
        ("DistilBERT Text", p_text, "#ff7f0e"),
        ("Multimodal Fused", p_fusion, "#2ca02c"),
    ]:
        precision, recall, _ = precision_recall_curve(y_true, probs)
        score = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {score:.4f})", color=color, linewidth=2.2)

    plt.axhline(y=baseline_pr, color="k", linestyle="--", alpha=0.6, label=f"Base Rate (PR-AUC = {baseline_pr:.4f})")
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall Curves (Holdout Test)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    pr_plot_file = out_path / "precision_recall_curve.png"
    plt.savefig(pr_plot_file)
    plt.close()
    generated_plots["precision_recall_curve"] = str(pr_plot_file)

    # =========================================================================
    # 3. Confusion Matrix Heatmap at Operating Threshold
    # =========================================================================
    plt.figure(figsize=(7, 6), dpi=300)
    y_pred = (p_fusion >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Multimodal Fusion Confusion Matrix (Test Set @ tau = {optimal_threshold:.2f})", fontsize=12, fontweight="bold", pad=12)
    plt.colorbar(im)
    classes = ["Retained (0)", "Exited (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}\n({(cm[i, j]/cm.sum())*100:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
                fontweight="bold",
            )

    plt.ylabel("Ground Truth Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Risk Decision", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_plot_file = out_path / "confusion_matrix.png"
    plt.savefig(cm_plot_file)
    plt.close()
    generated_plots["confusion_matrix"] = str(cm_plot_file)

    # =========================================================================
    # 4. Calibration Curve (Reliability Diagram)
    # =========================================================================
    plt.figure(figsize=(8, 6), dpi=300)
    prob_true, prob_pred = calibration_curve(y_true, p_fusion, n_bins=10, strategy="uniform")
    plt.plot(prob_pred, prob_true, "s-", color="#2ca02c", linewidth=2.2, label="Multimodal Fused Model")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Observed Attrition Fraction", fontsize=11, fontweight="bold")
    plt.title("Reliability Diagram / Calibration Curve (Holdout Test)", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    calib_plot_file = out_path / "calibration_curve.png"
    plt.savefig(calib_plot_file)
    plt.close()
    generated_plots["calibration_curve"] = str(calib_plot_file)

    return generated_plots
