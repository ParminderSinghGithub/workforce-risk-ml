"""Configuration module for Workforce Risk ML System.

Loads and validates frozen project constants from configs/config.yaml using Pydantic.
"""

from pathlib import Path
from typing import Any, List, Optional
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ProjectInfoConfig(BaseModel):
    name: str = "workforce-risk-ml-system"
    version: str = "0.1.0"
    seed: int = 42


class DatasetConfig(BaseModel):
    repository: str
    revision: str
    raw_file: str = "synthetic-employee-dataset.json"


class TargetsConfig(BaseModel):
    structured_target: str = "left_company"
    text_target_derived: str = "high_burnout_risk"
    burnout_threshold: float = 0.75

    @field_validator("burnout_threshold")
    @classmethod
    def validate_burnout_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"burnout_threshold must be between 0.0 and 1.0, got {v}")
        return v


class FeaturesConfig(BaseModel):
    text_column: str = "recent_feedback"
    leakage_exclusions: List[str] = Field(default_factory=list)

    @field_validator("leakage_exclusions")
    @classmethod
    def validate_leakage_exclusions(cls, v: List[str]) -> List[str]:
        required_exclusions = {
            "employee_id",
            "turnover_reason",
            "turnover_probability_generated",
            "risk_factors_summary",
            "burnout_risk",
        }
        missing = required_exclusions - set(v)
        if missing:
            raise ValueError(f"leakage_exclusions missing required fields: {missing}")
        return v


class SplitsConfig(BaseModel):
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    structured_strategy: str = "stratified"
    text_strategy: str = "group_template"

    @model_validator(mode="after")
    def validate_split_sum(self) -> "SplitsConfig":
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        if total != 1.0:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {self.train_ratio} + "
                f"{self.val_ratio} + {self.test_ratio} = {total}"
            )
        return self


class SamplingConfig(BaseModel):
    structured_train_size: int = 100000
    text_sample_min: int = 5000
    text_sample_max: int = 10000
    text_sample_target: int = 10000


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = Field(default_factory=lambda: ["q_lin", "v_lin"])
    lora_dropout: float = 0.05
    bias: str = "none"


class TextModelConfig(BaseModel):
    base_model: str = "distilbert-base-uncased"
    max_length: int = 256
    batch_size: int = 32
    learning_rate: float = 0.0002
    epochs: int = 3
    lora: LoRAConfig = Field(default_factory=LoRAConfig)


class StructuredModelConfig(BaseModel):
    architecture: str = "mlp"
    hidden_dims: List[int] = Field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.2
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 20


class FusionModelConfig(BaseModel):
    architecture: str = "late_fusion_mlp"
    input_dim: int = 2  # [p_structured, p_text]
    hidden_dims: List[int] = Field(default_factory=lambda: [16, 8])
    learning_rate: float = 0.001
    epochs: int = 15


class ModelsConfig(BaseModel):
    structured: StructuredModelConfig = Field(default_factory=StructuredModelConfig)
    text: TextModelConfig = Field(default_factory=TextModelConfig)
    fusion: FusionModelConfig = Field(default_factory=FusionModelConfig)


class PathsConfig(BaseModel):
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    data_splits_dir: str = "data/splits"
    models_dir: str = "models"
    reports_dir: str = "reports"


class ProjectConfig(BaseModel):
    project: ProjectInfoConfig = Field(default_factory=ProjectInfoConfig)
    dataset: DatasetConfig
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"


def load_config(config_path: Optional[str | Path] = None) -> ProjectConfig:
    """Load and validate configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file. Defaults to configs/config.yaml.

    Returns:
        Validated ProjectConfig instance.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f)

    if not isinstance(raw_dict, dict):
        raise ValueError(f"Invalid YAML content in {path}, expected dictionary.")

    return ProjectConfig.model_validate(raw_dict)


# Global singleton instance for easy import across modules
_cached_config: Optional[ProjectConfig] = None


def get_config(reload: bool = False, config_path: Optional[str | Path] = None) -> ProjectConfig:
    """Get the cached global configuration or load it if not yet initialized."""
    global _cached_config
    if _cached_config is None or reload:
        _cached_config = load_config(config_path)
    return _cached_config
