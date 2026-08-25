"""Sample CLI and programmatic demonstration of multimodal workforce risk inference."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workforce_risk.inference.predictor import WorkforceRiskPredictor
from workforce_risk.inference.schemas import EmployeeInput


def run_sample_predictions() -> None:
    """Demonstrate inference on synthetic employee profiles (low-risk vs high-risk)."""
    print("=" * 72)
    print(" WORKFORCE RISK ML SYSTEM -- MULTIMODAL INFERENCE SMOKE TEST")
    print("=" * 72)

    # 1. Instantiate Predictor from saved artifacts
    print("[1/3] Initializing WorkforceRiskPredictor from saved disk artifacts...")
    predictor = WorkforceRiskPredictor.from_artifacts(device_str="cpu")
    print("      Predictor loaded successfully (Offline inference ready).")

    # 2. Construct Test Profiles
    print("[2/3] Constructing sample employee profiles...")
    
    # Profile A: Satisfied, low burnout
    profile_a = EmployeeInput(
        employee_id="EMP-1001",
        department="Engineering",
        job_level="Senior",
        role="Senior Software Engineer",
        tenure_months=36.0,
        salary=135000.0,
        performance_score=0.88,
        satisfaction_score=0.85,
        workload_score=0.45,
        team_sentiment=0.82,
        project_completion_rate=0.92,
        overtime_hours=2.0,
        training_participation=0.75,
        collaboration_score=0.80,
        stress_level=0.30,
        recent_feedback="Great quarter! Feeling very supported by management and love the project direction.",
    )

    # Profile B: Stressed, high workload, burnout indicators
    profile_b = EmployeeInput(
        employee_id="EMP-2002",
        department="Sales",
        job_level="Mid",
        role="Sales Representative",
        tenure_months=14.0,
        salary=72000.0,
        performance_score=0.62,
        satisfaction_score=0.32,
        workload_score=0.95,
        team_sentiment=0.40,
        project_completion_rate=0.68,
        overtime_hours=22.0,
        training_participation=0.10,
        collaboration_score=0.45,
        stress_level=0.90,
        recent_feedback="Completely exhausted from constant overtime and unattainable quarterly targets. Severe burnout.",
    )

    # 3. Predict Single and Batch
    print("[3/3] Executing multimodal prediction pipeline...")
    result_a = predictor.predict_single(profile_a)
    result_b = predictor.predict_single(profile_b)

    print("\n" + "-" * 72)
    print(" RESULTS: Profile A (Satisfied Engineer)")
    print("-" * 72)
    print(json.dumps(result_a.to_dict(), indent=2))

    print("\n" + "-" * 72)
    print(" RESULTS: Profile B (Burnt-out Representative)")
    print("-" * 72)
    print(json.dumps(result_b.to_dict(), indent=2))
    print("-" * 72)

    # Quick Assertions for smoke test
    assert 0.0 <= result_a.fused_risk_probability <= 1.0
    assert 0.0 <= result_b.fused_risk_probability <= 1.0
    assert result_a.risk_tier in ["LOW", "ELEVATED", "HIGH", "CRITICAL"]
    assert result_b.risk_tier in ["LOW", "ELEVATED", "HIGH", "CRITICAL"]
    print("\n[Verification] All inference assertions passed successfully!")


if __name__ == "__main__":
    run_sample_predictions()
