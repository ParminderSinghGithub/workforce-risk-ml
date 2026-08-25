"""Unit and integration tests for the FastAPI serving endpoints."""

import pytest
from fastapi.testclient import TestClient

from workforce_risk.serving.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture providing initialized FastAPI TestClient with loaded lifespan models."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient) -> None:
    """Verify root / endpoint returns operational identity."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "service" in data


def test_health_endpoints(client: TestClient) -> None:
    """Verify /health and /api/v1/health return model readiness status."""
    for endpoint in ["/health", "/api/v1/health"]:
        response = client.get(endpoint)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["offline_mode"] is True
        assert data["models_loaded"]["structured_mlp"] is True
        assert data["models_loaded"]["text_distilbert_lora"] is True
        assert data["models_loaded"]["multimodal_late_fusion"] is True
        assert 0.0 < data["decision_threshold"] < 1.0


def test_predict_single_endpoint(client: TestClient) -> None:
    """Verify /predict endpoint returns calibrated multimodal prediction output."""
    payload = {
        "employee_id": "EMP-API-001",
        "department": "Engineering",
        "job_level": "Senior",
        "role": "Senior Software Engineer",
        "tenure_months": 36.0,
        "salary": 140000.0,
        "performance_score": 0.90,
        "satisfaction_score": 0.85,
        "workload_score": 0.50,
        "team_sentiment": 0.80,
        "recent_feedback": "Excellent performance, strong collaboration and exciting deliverables.",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["employee_id"] == "EMP-API-001"
    assert 0.0 <= data["fused_risk_probability"] <= 1.0
    assert 0.0 <= data["structured_risk_probability"] <= 1.0
    assert 0.0 <= data["text_risk_probability"] <= 1.0
    assert data["risk_prediction"] in [0, 1]
    assert data["risk_tier"] in ["LOW", "ELEVATED", "HIGH", "CRITICAL"]
    assert "modality_breakdown" in data
    assert "structured_weight" in data["modality_breakdown"]
    assert "text_weight" in data["modality_breakdown"]


def test_predict_batch_endpoint(client: TestClient) -> None:
    """Verify /predict/batch endpoint processes multiple employee records."""
    payload = {
        "employees": [
            {
                "employee_id": "EMP-B1",
                "satisfaction_score": 0.90,
                "recent_feedback": "Very satisfied with current projects and team.",
            },
            {
                "employee_id": "EMP-B2",
                "satisfaction_score": 0.20,
                "workload_score": 0.95,
                "recent_feedback": "Extreme burnout from continuous overtime.",
            },
        ]
    }

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_predictions"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["employee_id"] == "EMP-B1"
    assert data["predictions"][1]["employee_id"] == "EMP-B2"


def test_predict_validation_errors(client: TestClient) -> None:
    """Verify malformed requests trigger clear 422 HTTP validation errors."""
    # 1. Invalid score range (performance_score > 1.0)
    bad_payload_1 = {
        "performance_score": 5.0,  # ge=0.0, le=1.0 constraint
    }
    res1 = client.post("/predict", json=bad_payload_1)
    assert res1.status_code == 422

    # 2. Empty batch list (min_length=1 constraint)
    bad_payload_2 = {
        "employees": []
    }
    res2 = client.post("/predict/batch", json=bad_payload_2)
    assert res2.status_code == 422
