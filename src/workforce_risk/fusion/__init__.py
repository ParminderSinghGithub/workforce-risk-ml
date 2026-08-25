"""Multimodal Late Fusion module for Workforce Risk ML System."""

from workforce_risk.fusion.model import MultimodalLateFusion, load_fusion_model, save_fusion_model
from workforce_risk.fusion.train import run_multimodal_fusion

__all__ = [
    "MultimodalLateFusion",
    "load_fusion_model",
    "save_fusion_model",
    "run_multimodal_fusion",
]
