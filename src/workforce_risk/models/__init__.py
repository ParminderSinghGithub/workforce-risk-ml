"""PyTorch structured modeling package for employee attrition prediction."""

from workforce_risk.models.dataset import StructuredDataset, create_data_loaders
from workforce_risk.models.model import StructuredMLP
from workforce_risk.models.evaluate import calculate_classification_metrics, evaluate_model
from workforce_risk.models.train import get_device, train_structured_model

__all__ = [
    "StructuredDataset",
    "create_data_loaders",
    "StructuredMLP",
    "calculate_classification_metrics",
    "evaluate_model",
    "get_device",
    "train_structured_model",
]
