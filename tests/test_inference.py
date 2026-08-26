"""Unit tests for the end-to-end multimodal risk inference pipeline."""

import pytest
from pathlib import Path
import numpy as np
import pandas as pd

from workforce_risk.inference.predictor import WorkforceRiskPredictor
from workforce_risk.inference.schemas import EmployeeInput, RiskPredictionResult, RiskTier


@pytest.fixture(scope="module")
def predictor() -> WorkforceRiskPredictor:
    """Fixture providing initialized WorkforceRiskPredictor loaded from artifacts."""
    return WorkforceRiskPredictor.from_artifacts(device_str="cpu")


def test_employee_input_defaults_and_serialization() -> None:
    """Verify EmployeeInput initialization with defaults and dictionary conversion."""
    emp = EmployeeInput(employee_id="EMP-9999")
    assert emp.employee_id == "EMP-9999"
    assert emp.tenure_months == 24.0
    assert emp.department == "Engineering"

    emp_dict = emp.to_dict()
    assert isinstance(emp_dict, dict)
    assert emp_dict["employee_id"] == "EMP-9999"
    assert "recent_feedback" in emp_dict


def test_predictor_single_prediction_schema_and_bounds(predictor: WorkforceRiskPredictor) -> None:
    """Verify single prediction output schema, probability bounds, and tier assignment."""
    emp = EmployeeInput(
        employee_id="EMP-12345",
        department="Engineering",
        job_level="Senior",
        role="Senior Software Engineer",
        tenure_months=30.0,
        salary=120000.0,
        performance_score=0.85,
        satisfaction_score=0.80,
        recent_feedback="Positive quarter, strong team alignment and great project results.",
    )

    result = predictor.predict_single(emp)

    assert isinstance(result, RiskPredictionResult)
    assert result.employee_id == "EMP-12345"
    assert 0.0 <= result.fused_risk_probability <= 1.0
    assert 0.0 <= result.structured_risk_probability <= 1.0
    assert 0.0 <= result.text_risk_probability <= 1.0
    assert result.risk_prediction in [0, 1]
    assert result.risk_tier in [t.value for t in RiskTier]
    assert "structured_weight" in result.modality_breakdown
    assert "text_weight" in result.modality_breakdown
    assert len(result.summary) > 0


def test_predictor_robustness_on_malformed_and_missing_inputs(
    predictor: WorkforceRiskPredictor,
) -> None:
    """Verify inference pipeline handles missing text, unseen categories, and raw dicts safely."""
    # 1. Empty feedback text and unseen department string
    raw_dict = {
        "employee_id": "EMP-UNKNOWN",
        "department": "NonExistentDept123",
        "role": "UnknownRole456",
        "tenure_months": 12.0,
        "salary": 80000.0,
        "recent_feedback": None,  # None/empty text
        "technical_skills": None,
        "soft_skills": None,
    }

    result = predictor.predict_single(raw_dict)

    assert result.employee_id == "EMP-UNKNOWN"
    assert 0.0 <= result.fused_risk_probability <= 1.0
    assert not np.isnan(result.fused_risk_probability)

    # 2. Extreme out-of-bound numerical scores (should be safely clipped)
    extreme_dict = {
        "performance_score": 99.0,  # > 1.0
        "satisfaction_score": -5.0,  # < 0.0
        "overtime_hours": -10.0,  # < 0.0
        "recent_feedback": "Extreme test case.",
    }
    result_extreme = predictor.predict_single(extreme_dict)
    assert 0.0 <= result_extreme.fused_risk_probability <= 1.0


def test_predictor_threshold_override(predictor: WorkforceRiskPredictor) -> None:
    """Verify custom threshold override alters binary decision accordingly."""
    emp = EmployeeInput(
        employee_id="EMP-THRESH",
        recent_feedback="Everything is proceeding normally.",
    )

    res_default = predictor.predict_single(emp)
    res_high_thresh = predictor.predict_single(emp, threshold_override=0.99)
    res_low_thresh = predictor.predict_single(emp, threshold_override=0.01)

    assert res_high_thresh.risk_prediction == 0
    assert res_low_thresh.risk_prediction == 1
    assert res_high_thresh.decision_threshold == 0.99
    assert res_low_thresh.decision_threshold == 0.01


def test_predictor_batch_prediction(predictor: WorkforceRiskPredictor) -> None:
    """Verify batch prediction on DataFrame and list of inputs."""
    records = [
        EmployeeInput(employee_id="EMP-B1", satisfaction_score=0.9, recent_feedback="Happy and motivated!"),
        EmployeeInput(employee_id="EMP-B2", satisfaction_score=0.2, recent_feedback="Stressed and leaving soon."),
    ]

    results = predictor.predict_batch(records)
    assert len(results) == 2
    assert results[0].employee_id == "EMP-B1"
    assert results[1].employee_id == "EMP-B2"

    # Also test DataFrame input
    df_records = pd.DataFrame([r.to_dict() for r in records])
    results_df = predictor.predict_batch(df_records)
    assert len(results_df) == 2
    assert results_df[0].fused_risk_probability == results[0].fused_risk_probability


def test_predictor_from_artifacts_hf_download_fallback(tmp_path: Path) -> None:
    """Verify that predictor triggers snapshot_download from Hugging Face when artifacts are missing."""
    from unittest.mock import patch

    empty_dir = tmp_path / "missing_artifacts"

    with patch("huggingface_hub.snapshot_download") as mock_download:
        with pytest.raises(FileNotFoundError):
            # Since mock doesn't create real files, it fails at the FileNotFoundError check after attempting download
            WorkforceRiskPredictor.from_artifacts(artifacts_dir=empty_dir)

        mock_download.assert_called_once()

