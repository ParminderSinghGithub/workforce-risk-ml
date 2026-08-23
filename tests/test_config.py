"""Tests for project configuration, constraints, and seed utilities."""

import random
from pathlib import Path
import pytest
import numpy as np
from workforce_risk.config import load_config, get_config, ProjectConfig, SplitsConfig, TargetsConfig, FeaturesConfig
from workforce_risk.utils.seed import set_seed


def test_config_loads_successfully():
    """Verify that the central YAML configuration loads and parses without errors."""
    config = load_config()
    assert isinstance(config, ProjectConfig)
    assert config.project.name == "workforce-risk-ml-system"
    assert config.project.seed == 42


def test_required_dataset_and_target_fields():
    """Verify dataset identifiers, targets, and burnout threshold."""
    config = get_config()
    assert config.dataset.repository == "Umer112233/employee-burnout-turnover-prediction-800k"
    assert config.dataset.revision == "8d35f5bdcd0b7ff0ea04a5d5e93132eaae630e52"
    assert config.targets.structured_target == "left_company"
    assert config.targets.text_target_derived == "high_burnout_risk"
    assert config.targets.burnout_threshold == 0.75


def test_leakage_exclusions_contain_all_required_columns():
    """Verify that all 5 critical leakage columns are explicitly excluded."""
    config = get_config()
    expected_leakage = {
        "employee_id",
        "turnover_reason",
        "turnover_probability_generated",
        "risk_factors_summary",
        "burnout_risk",
    }
    actual_exclusions = set(config.features.leakage_exclusions)
    assert expected_leakage.issubset(actual_exclusions), (
        f"Missing leakage exclusions: {expected_leakage - actual_exclusions}"
    )


def test_split_proportions_sum_to_one():
    """Verify that train, validation, and test proportions sum exactly to 1.0."""
    config = get_config()
    total = config.splits.train_ratio + config.splits.val_ratio + config.splits.test_ratio
    assert round(total, 6) == 1.0
    assert config.splits.train_ratio == 0.80
    assert config.splits.val_ratio == 0.10
    assert config.splits.test_ratio == 0.10
    assert config.splits.structured_strategy == "stratified"
    assert config.splits.text_strategy == "group_template"


def test_sampling_and_model_specifications():
    """Verify bounded sample sizes and model architectures."""
    config = get_config()
    assert config.sampling.structured_train_size == 100000
    assert config.sampling.text_sample_min == 5000
    assert config.sampling.text_sample_max == 10000
    assert config.models.text.base_model == "distilbert-base-uncased"
    assert config.models.text.lora.r == 16
    assert config.models.text.lora.lora_alpha == 32
    assert "q_lin" in config.models.text.lora.target_modules
    assert "v_lin" in config.models.text.lora.target_modules
    assert config.models.fusion.input_dim == 2


def test_seed_utility_reproducibility():
    """Verify that set_seed ensures deterministic random number generation."""
    set_seed(42)
    val_random_1 = random.random()
    val_numpy_1 = np.random.rand()

    set_seed(42)
    val_random_2 = random.random()
    val_numpy_2 = np.random.rand()

    assert val_random_1 == val_random_2
    assert val_numpy_1 == val_numpy_2


def test_config_validation_catches_invalid_splits():
    """Verify that invalid split ratios trigger a Pydantic validation error."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        SplitsConfig(train_ratio=0.80, val_ratio=0.20, test_ratio=0.10)


def test_config_validation_catches_invalid_burnout_threshold():
    """Verify that an out-of-bounds burnout threshold triggers an error."""
    with pytest.raises(ValueError, match="burnout_threshold must be between 0.0 and 1.0"):
        TargetsConfig(burnout_threshold=1.5)


def test_config_validation_catches_missing_leakage_columns():
    """Verify that omitting a mandatory leakage column triggers an error."""
    with pytest.raises(ValueError, match="leakage_exclusions missing required fields"):
        FeaturesConfig(leakage_exclusions=["employee_id", "turnover_reason"])
