"""Multimodal Late Fusion training pipeline, threshold optimization, and artifact generation."""

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import torch

from workforce_risk.config import get_config
from workforce_risk.fusion.dataset import load_aligned_multimodal_data
from workforce_risk.fusion.evaluate import generate_fusion_plots, generate_unimodal_predictions
from workforce_risk.fusion.model import MultimodalLateFusion, save_fusion_model
from workforce_risk.models.evaluate import calculate_classification_metrics, evaluate_threshold_sweep, find_optimal_threshold
from workforce_risk.models.model import StructuredMLP
from workforce_risk.models.preprocessor import TabularPreprocessor
from workforce_risk.models.train import get_device, get_git_commit_hash, print_startup_banner
from workforce_risk.nlp.baseline import load_text_baseline
from workforce_risk.nlp.model import load_lora_text_model
from workforce_risk.utils.seed import set_seed


def run_multimodal_fusion(
    structured_checkpoint_path: Optional[str | Path] = None,
    text_model_dir: Optional[str | Path] = None,
    text_baseline_path: Optional[str | Path] = None,
    splits_dir: Optional[str | Path] = None,
    artifacts_dir: Optional[str | Path] = None,
    c_param: float = 1.0,
    use_logit_transform: bool = True,
    device_str: str = "auto",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute end-to-end multimodal late fusion training, validation thresholding, and test benchmarking."""
    config = get_config()
    t0 = time.time()
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git_hash = get_git_commit_hash()

    seed = seed if seed is not None else config.project.seed
    set_seed(seed)

    # 1. Resolve Paths
    artifacts_dir = Path(artifacts_dir or "artifacts/fusion").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = artifacts_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    struct_ckpt = Path(structured_checkpoint_path or "artifacts/structured_model/best_checkpoint.pt").resolve()
    text_dir = Path(text_model_dir or "artifacts/text_transformer/best_model").resolve()
    baseline_path = Path(text_baseline_path or "artifacts/text_baseline/tfidf_baseline.joblib").resolve()

    if not struct_ckpt.exists():
        raise FileNotFoundError(f"Missing structured checkpoint: {struct_ckpt}")
    if not text_dir.exists():
        raise FileNotFoundError(f"Missing text transformer model dir: {text_dir}")

    # 2. Select Device
    device, device_info = get_device(device_str)
    print_startup_banner(device_info, sample_size=None, batch_size=64)

    # 3. Load Trained Unimodal Models
    print(f"[Fusion] Loading Structured MLP checkpoint from: {struct_ckpt}")
    saved_struct = torch.load(struct_ckpt, map_location=device, weights_only=False)
    preprocessor = TabularPreprocessor.from_dict(saved_struct["preprocessing_config"])
    structured_model = StructuredMLP(
        input_dim=saved_struct["model_config"]["input_dim"],
        hidden_dims=saved_struct["model_config"]["hidden_dims"],
        dropout=saved_struct["model_config"]["dropout"],
    ).to(device)
    structured_model.load_state_dict(saved_struct["model_state_dict"])
    structured_model.eval()

    print(f"[Fusion] Loading DistilBERT + LoRA model from: {text_dir}")
    text_model, tokenizer = load_lora_text_model(text_dir, device=device)
    text_model.eval()

    print(f"[Fusion] Loading TF-IDF Baseline model from: {baseline_path}")
    tfidf_baseline = load_text_baseline(baseline_path) if baseline_path.exists() else None

    # 4. Load Aligned Dual-Holdout Datasets
    print("[Fusion] Loading aligned dual-holdout validation and test datasets...")
    val_df, test_df = load_aligned_multimodal_data(splits_dir=splits_dir)
    print(f"[Fusion] Aligned Datasets Loaded: Validation={len(val_df):,} | Test={len(test_df):,}")

    # 5. Generate Unimodal Validation Predictions
    print("[Fusion] Generating unimodal predictions on Dual Validation set...")
    p_struct_val, p_text_val, y_val = generate_unimodal_predictions(
        df=val_df,
        structured_model=structured_model,
        preprocessor=preprocessor,
        text_model=text_model,
        tokenizer=tokenizer,
        device=device,
    )
    p_tfidf_val = tfidf_baseline.predict_proba(val_df["recent_feedback"].tolist()) if tfidf_baseline else np.zeros_like(y_val)

    # 6. Fit Multimodal Late Fusion Model on Validation Data
    print("[Fusion] Fitting Multimodal Late Fusion meta-classifier on validation log-odds...")
    fusion_model = MultimodalLateFusion(
        use_logit_transform=use_logit_transform,
        c_param=c_param,
        random_seed=seed,
    )
    fusion_model.fit(p_structured=p_struct_val, p_text=p_text_val, y_true=y_val)
    p_fusion_val = fusion_model.predict_proba(p_struct_val, p_text_val)

    # 7. Optimize Operating Thresholds on Validation Data ONLY
    print("[Fusion] Optimizing operating thresholds on Validation partition...")
    tau_struct_val, f1_struct_val = find_optimal_threshold(y_val, p_struct_val, metric="f1")
    tau_text_val, f1_text_val = find_optimal_threshold(y_val, p_text_val, metric="f1")
    tau_tfidf_val, f1_tfidf_val = find_optimal_threshold(y_val, p_tfidf_val, metric="f1")
    tau_fusion_val, f1_fusion_val = find_optimal_threshold(y_val, p_fusion_val, metric="f1")

    fusion_model.optimal_threshold = tau_fusion_val
    val_threshold_sweep = evaluate_threshold_sweep(y_val, p_fusion_val)

    print(f"[Validation Results] Optimal Thresholds: Structured={tau_struct_val:.2f} (F1={f1_struct_val:.4f}) | Text={tau_text_val:.2f} (F1={f1_text_val:.4f}) | Fusion={tau_fusion_val:.2f} (F1={f1_fusion_val:.4f})")

    # 8. Generate Unimodal & Fused Predictions on Untouched Holdout Test Set
    print("[Fusion] Generating predictions on Untouched Holdout Test partition...")
    p_struct_test, p_text_test, y_test = generate_unimodal_predictions(
        df=test_df,
        structured_model=structured_model,
        preprocessor=preprocessor,
        text_model=text_model,
        tokenizer=tokenizer,
        device=device,
    )
    p_tfidf_test = tfidf_baseline.predict_proba(test_df["recent_feedback"].tolist()) if tfidf_baseline else np.zeros_like(y_test)
    p_fusion_test = fusion_model.predict_proba(p_struct_test, p_text_test)

    # 9. Evaluate All Models on Holdout Test Set Using Validation-Fixed Thresholds
    print("[Fusion] Calculating quantitative test performance benchmarks...")
    metrics_struct_test = calculate_classification_metrics(y_test, p_struct_test, threshold=tau_struct_val)
    metrics_tfidf_test = calculate_classification_metrics(y_test, p_tfidf_test, threshold=tau_tfidf_val)
    metrics_text_test = calculate_classification_metrics(y_test, p_text_test, threshold=tau_text_val)
    metrics_fusion_test = calculate_classification_metrics(y_test, p_fusion_test, threshold=tau_fusion_val)
    metrics_fusion_default = calculate_classification_metrics(y_test, p_fusion_test, threshold=0.50)

    # 10. Generate High-Resolution Benchmark Evidence Plots
    print("[Fusion] Generating comparative benchmark visualizations...")
    plot_files = generate_fusion_plots(
        y_true=y_test,
        p_struct=p_struct_test,
        p_text=p_text_test,
        p_fusion=p_fusion_test,
        optimal_threshold=tau_fusion_val,
        output_dir=plots_dir,
    )

    # 11. Save Aligned Prediction Tables
    val_pred_df = pd.DataFrame({
        "employee_id": val_df["employee_id"],
        "p_structured": np.round(p_struct_val, 5),
        "p_text_lora": np.round(p_text_val, 5),
        "p_text_tfidf": np.round(p_tfidf_val, 5),
        "p_fusion": np.round(p_fusion_val, 5),
        "left_company": y_val,
    })
    val_pred_path = artifacts_dir / "validation_predictions.parquet"
    val_pred_df.to_parquet(val_pred_path, index=False)

    test_pred_df = pd.DataFrame({
        "employee_id": test_df["employee_id"],
        "p_structured": np.round(p_struct_test, 5),
        "p_text_lora": np.round(p_text_test, 5),
        "p_text_tfidf": np.round(p_tfidf_test, 5),
        "p_fusion": np.round(p_fusion_test, 5),
        "left_company": y_test,
    })
    test_pred_path = artifacts_dir / "test_predictions.parquet"
    test_pred_df.to_parquet(test_pred_path, index=False)

    # 12. Save Trained Fusion Model
    fusion_model_path = artifacts_dir / "fusion_model.joblib"
    save_fusion_model(fusion_model, fusion_model_path)

    # 13. Save Comprehensive Metadata & Summaries
    coefs = fusion_model.get_coefficients()
    config_data = {
        "fusion_type": "MultimodalLateFusion_LogitRegression",
        "timestamp_utc": timestamp_utc,
        "git_commit": git_hash,
        "hyperparameters": coefs,
        "input_modalities": {
            "structured": str(struct_ckpt),
            "text_transformer": str(text_dir),
            "text_baseline": str(baseline_path),
        },
        "dataset_sizes": {
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "test_positive_base_rate": float(y_test.mean()),
        },
    }
    config_path = artifacts_dir / "fusion_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    total_time = round(time.time() - t0, 2)
    eval_summary = {
        "timestamp_utc": timestamp_utc,
        "git_commit": git_hash,
        "device": device_info["selected_device"],
        "total_seconds": total_time,
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "optimal_thresholds": {
            "structured": tau_struct_val,
            "text_tfidf": tau_tfidf_val,
            "text_lora": tau_text_val,
            "fusion": tau_fusion_val,
        },
        "model_comparison_holdout_test": {
            "1_structured_tabular_mlp": metrics_struct_test,
            "2_text_tfidf_baseline": metrics_tfidf_test,
            "3_text_distilbert_lora": metrics_text_test,
            "4_multimodal_late_fusion_default_0_5": metrics_fusion_default,
            "5_multimodal_late_fusion_optimal_thresh": metrics_fusion_test,
        },
        "coefficients": coefs,
        "plots": plot_files,
    }
    summary_path = artifacts_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # 14. Generate Formatted Markdown Run Summary
    delta_roc = metrics_fusion_test["roc_auc"] - metrics_struct_test["roc_auc"]
    delta_pr = metrics_fusion_test["pr_auc"] - metrics_struct_test["pr_auc"]
    delta_f1 = metrics_fusion_test["f1"] - metrics_struct_test["f1"]

    md_content = f"""# Multimodal Late Fusion Benchmark Report

- **Timestamp (UTC)**: `{timestamp_utc}`
- **Git Commit**: `{git_hash}`
- **Device**: `{device_info['selected_device']}`
- **Dual Validation Samples**: `{len(val_df):,}`
- **Dual Holdout Test Samples**: `{len(test_df):,}`
- **Total Pipeline Runtime**: `{total_time}s`

## Fusion Architecture & Meta-Model Parameters
- **Meta-Classifier**: Logistic Regression on Log-Odds ($C = {c_param}$)
- **Formula**: $\\text{{logit}}(\\hat{{y}}) = {coefs['intercept_b0']} + {coefs['w_structured']} \\cdot \\text{{logit}}(p_{{\\text{{structured}}}}) + {coefs['w_text']} \\cdot \\text{{logit}}(p_{{\\text{{text}}}})$
- **Validation-Selected Optimal Threshold ($\\tau^*$)**: **`{tau_fusion_val:.2f}`** (Val F1: `{f1_fusion_val:.4f}`)

## Comparative Performance on Untouched Holdout Test Partition ($N = {len(test_df):,}$)

| Model / Architecture | Operating Threshold ($\\tau^*$) | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | Log-Loss | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Structured Tabular MLP** | `{tau_struct_val:.2f}` | `{metrics_struct_test['roc_auc']:.4f}` | `{metrics_struct_test['pr_auc']:.4f}` | `{metrics_struct_test['precision']:.4f}` | `{metrics_struct_test['recall']:.4f}` | `{metrics_struct_test['f1']:.4f}` | `{metrics_struct_test['loss']:.4f}` | `{metrics_struct_test['brier_score']:.4f}` |
| **2. TF-IDF Text Baseline** | `{tau_tfidf_val:.2f}` | `{metrics_tfidf_test['roc_auc']:.4f}` | `{metrics_tfidf_test['pr_auc']:.4f}` | `{metrics_tfidf_test['precision']:.4f}` | `{metrics_tfidf_test['recall']:.4f}` | `{metrics_tfidf_test['f1']:.4f}` | `{metrics_tfidf_test['loss']:.4f}` | `{metrics_tfidf_test['brier_score']:.4f}` |
| **3. DistilBERT + PEFT/LoRA** | `{tau_text_val:.2f}` | `{metrics_text_test['roc_auc']:.4f}` | `{metrics_text_test['pr_auc']:.4f}` | `{metrics_text_test['precision']:.4f}` | `{metrics_text_test['recall']:.4f}` | `{metrics_text_test['f1']:.4f}` | `{metrics_text_test['loss']:.4f}` | `{metrics_text_test['brier_score']:.4f}` |
| **4. Multimodal Late Fusion** | **`{tau_fusion_val:.2f}`** | **`{metrics_fusion_test['roc_auc']:.4f}`** | **`{metrics_fusion_test['pr_auc']:.4f}`** | **`{metrics_fusion_test['precision']:.4f}`** | **`{metrics_fusion_test['recall']:.4f}`** | **`{metrics_fusion_test['f1']:.4f}`** | **`{metrics_fusion_test['loss']:.4f}`** | **`{metrics_fusion_test['brier_score']:.4f}`** |

### Multimodal Fusion Gains vs Structured Baseline:
- **$\\Delta$ ROC-AUC**: `+{delta_roc:.4f}`
- **$\\Delta$ PR-AUC**: `+{delta_pr:.4f}`
- **$\\Delta$ F1 Score**: `+{delta_f1:.4f}`

## Final Multimodal Test Confusion Matrix (@ $\\tau^* = {tau_fusion_val:.2f}$)
- **True Negatives**: `{metrics_fusion_test['confusion_matrix']['true_negatives']:,}`
- **False Positives**: `{metrics_fusion_test['confusion_matrix']['false_positives']:,}`
- **False Negatives**: `{metrics_fusion_test['confusion_matrix']['false_negatives']:,}`
- **True Positives**: `{metrics_fusion_test['confusion_matrix']['true_positives']:,}`
"""
    run_summary_path = artifacts_dir / "run_summary.md"
    with open(run_summary_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print("-" * 72)
    print(f"[Summary] Fusion Model saved:      {fusion_model_path}")
    print(f"[Summary] Validation predictions: {val_pred_path}")
    print(f"[Summary] Test predictions:       {test_pred_path}")
    print(f"[Summary] Evaluation summary:     {summary_path}")
    print(f"[Summary] Run summary (Markdown): {run_summary_path}")
    print(f"[Summary] Benchmark plots:        {plots_dir}")
    print("-" * 72)

    return eval_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Multimodal Late Fusion Pipeline")
    parser.add_argument("--structured-checkpoint", type=str, default=None, help="Path to best_checkpoint.pt")
    parser.add_argument("--text-model-dir", type=str, default=None, help="Path to best_model directory")
    parser.add_argument("--text-baseline", type=str, default=None, help="Path to tfidf_baseline.joblib")
    parser.add_argument("--splits-dir", type=str, default=None, help="Directory containing Parquet splits")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Output directory for fusion artifacts")
    parser.add_argument("--c-param", type=float, default=1.0, help="Logistic regression regularization parameter C")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_multimodal_fusion(
        structured_checkpoint_path=args.structured_checkpoint,
        text_model_dir=args.text_model_dir,
        text_baseline_path=args.text_baseline,
        splits_dir=args.splits_dir,
        artifacts_dir=args.artifacts_dir,
        c_param=args.c_param,
        device_str=args.device,
        seed=args.seed,
    )
