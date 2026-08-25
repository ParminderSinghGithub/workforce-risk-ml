"""TF-IDF + Logistic Regression text baseline for burnout risk classification."""

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from workforce_risk.config import get_config
from workforce_risk.models.evaluate import (
    calculate_classification_metrics,
    evaluate_threshold_sweep,
    find_optimal_threshold,
)
from workforce_risk.utils.seed import set_seed


class TfidfTextBaseline:
    """Leakage-safe TF-IDF Vectorizer + Logistic Regression classifier for text risk modeling."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 1,
        c_param: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.c_param = c_param
        self.random_seed = random_seed

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=min_df,
            stop_words="english",
        )
        self.classifier = LogisticRegression(
            C=c_param,
            max_iter=300,
            random_state=random_seed,
        )
        self.is_fitted: bool = False

    def fit(self, texts: List[str] | np.ndarray, labels: List[int] | np.ndarray) -> "TfidfTextBaseline":
        """Fit vectorizer and classifier strictly on training texts."""
        texts_clean = [str(t) if t is not None else "" for t in texts]
        y = np.asarray(labels).astype(int)

        x_mat = self.vectorizer.fit_transform(texts_clean)
        self.classifier.fit(x_mat, y)
        self.is_fitted = True
        return self

    def predict_proba(self, texts: List[str] | np.ndarray) -> np.ndarray:
        """Predict probability of high burnout risk [N]."""
        if not self.is_fitted:
            raise RuntimeError("TfidfTextBaseline must be fitted before predict_proba().")
        texts_clean = [str(t) if t is not None else "" for t in texts]
        x_mat = self.vectorizer.transform(texts_clean)
        return self.classifier.predict_proba(x_mat)[:, 1]

    def evaluate(
        self,
        texts: List[str] | np.ndarray,
        labels: List[int] | np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Compute comprehensive quantitative classification metrics."""
        probs = self.predict_proba(texts)
        y = np.asarray(labels).astype(int)
        metrics = calculate_classification_metrics(y, probs, threshold=threshold)
        return metrics, probs

    def get_top_features(self, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Extract top positive and negative indicative n-grams."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        coefs = self.classifier.coef_[0]

        top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
        top_neg_idx = np.argsort(coefs)[:top_n]

        return {
            "top_positive": [(str(feature_names[i]), round(float(coefs[i]), 4)) for i in top_pos_idx],
            "top_negative": [(str(feature_names[i]), round(float(coefs[i]), 4)) for i in top_neg_idx],
        }

    def save(self, filepath: str | Path) -> None:
        """Persist fitted baseline model artifact to disk."""
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "config": {
                    "max_features": self.max_features,
                    "ngram_range": self.ngram_range,
                    "min_df": self.min_df,
                    "c_param": self.c_param,
                    "random_seed": self.random_seed,
                },
                "is_fitted": self.is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, filepath: str | Path) -> "TfidfTextBaseline":
        """Load persisted baseline model artifact from disk."""
        path = Path(filepath).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at: {path}")
        data = joblib.load(path)
        model = cls(
            max_features=data["config"]["max_features"],
            ngram_range=tuple(data["config"]["ngram_range"]),
            min_df=data["config"].get("min_df", 1),
            c_param=data["config"]["c_param"],
            random_seed=data["config"]["random_seed"],
        )
        model.vectorizer = data["vectorizer"]
        model.classifier = data["classifier"]
        model.is_fitted = data.get("is_fitted", True)
        return model


def load_text_baseline(filepath: str | Path) -> TfidfTextBaseline:
    """Load persisted TF-IDF text baseline model artifact from disk."""
    return TfidfTextBaseline.load(filepath)


def train_text_baseline(
    train_path: Optional[str | Path] = None,
    val_path: Optional[str | Path] = None,
    test_path: Optional[str | Path] = None,
    artifacts_dir: Optional[str | Path] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    max_features: int = 5000,
    min_df: int = 1,
    c_param: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Train and evaluate TF-IDF + Logistic Regression baseline pipeline."""
    config = get_config()
    t0 = time.time()
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()

    seed = seed if seed is not None else config.project.seed
    set_seed(seed)

    splits_dir = Path(config.paths.data_splits_dir)
    train_path = Path(train_path or splits_dir / "text_train.parquet").resolve()
    val_path = Path(val_path or splits_dir / "text_validation.parquet").resolve()
    test_path = Path(test_path or splits_dir / "text_test.parquet").resolve()

    artifacts_dir = Path(artifacts_dir or "artifacts/text_baseline").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = artifacts_dir / "tfidf_baseline.joblib"
    summary_path = artifacts_dir / "text_baseline_summary.json"

    print("=" * 72)
    print(" WORKFORCE RISK ML SYSTEM -- TEXT MODALITY TF-IDF BASELINE")
    print("=" * 72)

    # 1. Load Parquet splits
    print(f"[Loading] Train: {train_path}")
    print(f"[Loading] Val:   {val_path}")
    print(f"[Loading] Test:  {test_path}")

    train_table = pq.read_table(str(train_path), columns=["recent_feedback", "high_burnout_risk"])
    val_table = pq.read_table(str(val_path), columns=["recent_feedback", "high_burnout_risk"])
    test_table = pq.read_table(str(test_path), columns=["recent_feedback", "high_burnout_risk"])

    # Optional subsampling
    if max_train_samples and max_train_samples < train_table.num_rows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(train_table.num_rows, size=max_train_samples, replace=False)
        train_texts = train_table["recent_feedback"].to_numpy()[idx]
        train_labels = train_table["high_burnout_risk"].to_numpy()[idx]
    else:
        train_texts = train_table["recent_feedback"].to_numpy()
        train_labels = train_table["high_burnout_risk"].to_numpy()

    if max_val_samples and max_val_samples < val_table.num_rows:
        val_texts = val_table["recent_feedback"].to_numpy()[:max_val_samples]
        val_labels = val_table["high_burnout_risk"].to_numpy()[:max_val_samples]
    else:
        val_texts = val_table["recent_feedback"].to_numpy()
        val_labels = val_table["high_burnout_risk"].to_numpy()

    if max_test_samples and max_test_samples < test_table.num_rows:
        test_texts = test_table["recent_feedback"].to_numpy()[:max_test_samples]
        test_labels = test_table["high_burnout_risk"].to_numpy()[:max_test_samples]
    else:
        test_texts = test_table["recent_feedback"].to_numpy()
        test_labels = test_table["high_burnout_risk"].to_numpy()

    print(f"[Data] Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")

    # 2. Fit TF-IDF Baseline
    baseline = TfidfTextBaseline(
        max_features=max_features,
        min_df=min_df,
        c_param=c_param,
        random_seed=seed,
    )
    print(f"[Training] Fitting TfidfTextBaseline (max_features={max_features}, C={c_param})...")
    baseline.fit(train_texts, train_labels)

    # 3. Evaluate on Validation Set
    val_metrics_default, val_probs = baseline.evaluate(val_texts, val_labels, threshold=0.5)
    best_thresh, best_f1 = find_optimal_threshold(val_labels, val_probs, metric="f1")
    val_metrics_optimal, _ = baseline.evaluate(val_texts, val_labels, threshold=best_thresh)

    print(f"[Validation] ROC-AUC: {val_metrics_default['roc_auc']:.4f} | PR-AUC: {val_metrics_default['pr_auc']:.4f}")
    print(f"[Validation] Default F1 (t=0.50): {val_metrics_default['f1']:.4f} | Optimal F1 (t={best_thresh:.2f}): {val_metrics_optimal['f1']:.4f}")

    # 4. Evaluate on Final Holdout Test Set (using validation-selected threshold)
    test_metrics_default, test_probs = baseline.evaluate(test_texts, test_labels, threshold=0.5)
    test_metrics_optimal, _ = baseline.evaluate(test_texts, test_labels, threshold=best_thresh)

    print(f"[Test]       ROC-AUC: {test_metrics_default['roc_auc']:.4f} | PR-AUC: {test_metrics_default['pr_auc']:.4f}")
    print(f"[Test]       Default F1 (t=0.50): {test_metrics_default['f1']:.4f} | Optimal F1 (t={best_thresh:.2f}): {test_metrics_optimal['f1']:.4f}")

    # 5. Persist Model and Summary Artifacts
    baseline.save(model_save_path)
    top_words = baseline.get_top_features(top_n=10)

    total_time = round(time.time() - t0, 2)
    summary_data = {
        "model_type": "Tfidf_LogisticRegression_Baseline",
        "timestamp_utc": timestamp_utc,
        "dataset_sizes": {
            "train_rows": len(train_texts),
            "val_rows": len(val_texts),
            "test_rows": len(test_texts),
        },
        "vocabulary_size": len(baseline.vectorizer.vocabulary_),
        "hyperparameters": {
            "max_features": max_features,
            "min_df": min_df,
            "c_param": c_param,
            "seed": seed,
        },
        "validation_metrics": {
            "roc_auc": val_metrics_default["roc_auc"],
            "pr_auc": val_metrics_default["pr_auc"],
            "default_threshold_0_5": val_metrics_default,
            "optimal_threshold": best_thresh,
            "optimal_threshold_metrics": val_metrics_optimal,
        },
        "test_metrics": {
            "roc_auc": test_metrics_default["roc_auc"],
            "pr_auc": test_metrics_default["pr_auc"],
            "default_threshold_0_5": test_metrics_default,
            "evaluated_at_optimal_threshold": best_thresh,
            "optimal_threshold_metrics": test_metrics_optimal,
        },
        "top_feature_n_grams": top_words,
        "model_artifact_path": str(model_save_path),
        "total_seconds": total_time,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[Artifacts] Model saved:   {model_save_path}")
    print(f"[Artifacts] Summary saved: {summary_path}")

    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate TF-IDF text baseline")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast smoke test with reduced sample")
    parser.add_argument("--train-samples", type=int, default=None, help="Sample size for training")
    parser.add_argument("--max-features", type=int, default=5000, help="Maximum TF-IDF n-gram vocabulary size")
    parser.add_argument("--c-param", type=float, default=1.0, help="Logistic regression regularization parameter C")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Output artifacts directory")
    args = parser.parse_args()

    if args.smoke_test:
        train_text_baseline(
            max_train_samples=args.train_samples or 10000,
            max_val_samples=2000,
            max_test_samples=2000,
            max_features=2000,
            artifacts_dir=args.artifacts_dir or "artifacts/text_baseline_smoke",
        )
    else:
        train_text_baseline(
            max_train_samples=args.train_samples,
            max_features=args.max_features,
            c_param=args.c_param,
            artifacts_dir=args.artifacts_dir,
        )
