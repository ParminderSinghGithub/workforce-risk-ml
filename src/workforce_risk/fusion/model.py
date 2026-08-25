"""Multimodal Late Fusion model combining structured tabular risk and unstructured text distress probabilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute numerical logit transform log(p / (1 - p)) clipped to prevent infinity."""
    p_clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped))


class MultimodalLateFusion:
    """Calibrated late-fusion classifier combining structured attrition and text burnout probabilities.

    Uses a logistic meta-classifier trained on either calibrated log-odds or raw probabilities
    to output the final enterprise attrition probability:
        logit(p_fused) = w0 + w1 * logit(p_structured) + w2 * logit(p_text)
    """

    def __init__(
        self,
        use_logit_transform: bool = True,
        c_param: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        self.use_logit_transform = use_logit_transform
        self.c_param = c_param
        self.random_seed = random_seed
        self.model = LogisticRegression(
            C=c_param,
            random_state=random_seed,
            solver="lbfgs",
            max_iter=1000,
        )
        self.optimal_threshold: float = 0.5
        self.is_fitted: bool = False

    def _transform_features(self, p_struct: np.ndarray, p_text: np.ndarray) -> np.ndarray:
        p_s = np.asarray(p_struct).ravel()
        p_t = np.asarray(p_text).ravel()
        if len(p_s) != len(p_t):
            raise ValueError(f"Length mismatch: p_structured ({len(p_s)}) vs p_text ({len(p_t)})")

        if self.use_logit_transform:
            f_s = safe_logit(p_s)
            f_t = safe_logit(p_t)
        else:
            f_s = p_s
            f_t = p_t

        return np.column_stack([f_s, f_t])

    def fit(
        self,
        p_structured: np.ndarray,
        p_text: np.ndarray,
        y_true: np.ndarray,
    ) -> "MultimodalLateFusion":
        """Fit late-fusion meta-model on validation probabilities and true attrition labels."""
        X = self._transform_features(p_structured, p_text)
        y = np.asarray(y_true).ravel().astype(int)

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, p_structured: np.ndarray, p_text: np.ndarray) -> np.ndarray:
        """Compute fused attrition probability in [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("Fusion model must be fitted before predict_proba.")
        X = self._transform_features(p_structured, p_text)
        # Return probability of positive class (column 1)
        return self.model.predict_proba(X)[:, 1]

    def predict(
        self,
        p_structured: np.ndarray,
        p_text: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict binary attrition decision at operating threshold."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(p_structured, p_text)
        return (probs >= thresh).astype(int)

    def get_coefficients(self) -> Dict[str, float]:
        """Extract learned weights and intercept."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        coefs = self.model.coef_.ravel()
        intercept = float(self.model.intercept_[0])
        return {
            "w_structured": round(float(coefs[0]), 4),
            "w_text": round(float(coefs[1]), 4),
            "intercept_b0": round(intercept, 4),
            "use_logit_transform": self.use_logit_transform,
            "c_param": self.c_param,
            "optimal_threshold": round(self.optimal_threshold, 4),
        }


def save_fusion_model(
    fusion_model: MultimodalLateFusion,
    output_path: str | Path,
) -> None:
    """Serialize fitted fusion model to disk."""
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fusion_model, path)


def load_fusion_model(
    model_path: str | Path,
) -> MultimodalLateFusion:
    """Deserialize fitted fusion model from disk."""
    path = Path(model_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Fusion model not found at: {path}")
    return joblib.load(path)
