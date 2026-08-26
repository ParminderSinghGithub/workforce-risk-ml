"""Multimodal Late Fusion module for Workforce Risk ML System."""

from workforce_risk.fusion.model import (
    MultimodalLateFusion,
    load_fusion_model,
    safe_logit,
    save_fusion_model,
)

__all__ = [
    "MultimodalLateFusion",
    "load_fusion_model",
    "save_fusion_model",
    "safe_logit",
]
