"""FastAPI serving package for Workforce Risk ML System."""

from workforce_risk.serving.app import app, create_app
from workforce_risk.serving.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    EmployeePredictionRequest,
    HealthResponse,
    ModalityBreakdown,
    PredictionResponse,
)

__all__ = [
    "app",
    "create_app",
    "EmployeePredictionRequest",
    "PredictionResponse",
    "BatchPredictionRequest",
    "BatchPredictionResponse",
    "HealthResponse",
    "ModalityBreakdown",
]
