"""Feature engineering, leakage control, and dataset splitting module."""

from workforce_risk.features.definitions import (
    ALLOWLISTED_PREDICTORS,
    EXCLUDED_LEAKAGE_COLUMNS,
    FEATURE_DEFINITIONS,
    generate_feature_manifest,
)
from workforce_risk.features.engineer import (
    build_feature_pipeline,
    extract_structured_features,
    prepare_text_dataset,
)
from workforce_risk.features.split import (
    create_grouped_text_split,
    create_stratified_structured_split,
)
from workforce_risk.features.pipeline import run_feature_pipeline

__all__ = [
    "ALLOWLISTED_PREDICTORS",
    "EXCLUDED_LEAKAGE_COLUMNS",
    "FEATURE_DEFINITIONS",
    "generate_feature_manifest",
    "build_feature_pipeline",
    "extract_structured_features",
    "prepare_text_dataset",
    "create_stratified_structured_split",
    "create_grouped_text_split",
    "run_feature_pipeline",
]
