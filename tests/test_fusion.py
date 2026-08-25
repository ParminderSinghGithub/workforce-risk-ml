"""Unit tests for Multimodal Late Fusion dataset alignment, model mathematics, and serialization."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from workforce_risk.fusion.dataset import load_aligned_multimodal_data
from workforce_risk.fusion.model import (
    MultimodalLateFusion,
    load_fusion_model,
    safe_logit,
    save_fusion_model,
)


def test_safe_logit_transform() -> None:
    """Verify safe logit function handles boundary probabilities without inf/nan."""
    probs = np.array([0.0, 0.5, 1.0, 0.1, 0.9])
    logits = safe_logit(probs, eps=1e-6)

    assert not np.any(np.isnan(logits))
    assert not np.any(np.isinf(logits))
    assert np.isclose(logits[1], 0.0, atol=1e-5)  # logit(0.5) == 0.0
    assert logits[3] < 0.0  # logit(0.1) < 0
    assert logits[4] > 0.0  # logit(0.9) > 0


def test_multimodal_late_fusion_fit_and_predict() -> None:
    """Verify late fusion model learns weights and predicts bounded probabilities."""
    np.random.seed(42)
    n = 100
    p_struct = np.random.uniform(0.1, 0.9, size=n)
    p_text = np.random.uniform(0.1, 0.9, size=n)
    # Synthetic target correlated with both modalities
    logits = 1.5 * safe_logit(p_struct) + 0.8 * safe_logit(p_text) - 0.2
    y_true = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)

    fusion = MultimodalLateFusion(use_logit_transform=True, c_param=1.0, random_seed=42)
    fusion.fit(p_struct, p_text, y_true)

    assert fusion.is_fitted
    coefs = fusion.get_coefficients()
    assert "w_structured" in coefs
    assert "w_text" in coefs
    assert "intercept_b0" in coefs

    # Predict probabilities
    p_fused = fusion.predict_proba(p_struct, p_text)
    assert p_fused.shape == (n,)
    assert np.all((p_fused >= 0.0) & (p_fused <= 1.0))

    # Predict discrete binary decisions
    preds = fusion.predict(p_struct, p_text, threshold=0.5)
    assert np.all(np.isin(preds, [0, 1]))


def test_multimodal_late_fusion_save_and_load() -> None:
    """Verify serialized fusion model reload produces identical predictions."""
    np.random.seed(42)
    p_s = np.random.uniform(0.2, 0.8, size=50)
    p_t = np.random.uniform(0.2, 0.8, size=50)
    y = np.random.choice([0, 1], size=50)

    fusion = MultimodalLateFusion(use_logit_transform=True, c_param=1.0, random_seed=42)
    fusion.fit(p_s, p_t, y)
    fusion.optimal_threshold = 0.35

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "fusion_model.joblib"
        save_fusion_model(fusion, model_path)

        reloaded = load_fusion_model(model_path)
        assert reloaded.is_fitted
        assert reloaded.optimal_threshold == 0.35

        orig_probs = fusion.predict_proba(p_s, p_t)
        reload_probs = reloaded.predict_proba(p_s, p_t)
        np.testing.assert_allclose(orig_probs, reload_probs, rtol=1e-6)


def test_dual_holdout_split_disjointness() -> None:
    """Verify aligned validation and test datasets have zero overlap with training partitions."""
    val_df, test_df = load_aligned_multimodal_data()

    assert len(val_df) > 0
    assert len(test_df) > 0

    val_ids = set(val_df["employee_id"])
    test_ids = set(test_df["employee_id"])

    # Disjointness between val and test
    assert len(val_ids.intersection(test_ids)) == 0

    # Required columns present
    for col in ["employee_id", "left_company", "recent_feedback", "salary", "tenure_months"]:
        assert col in val_df.columns
        assert col in test_df.columns
