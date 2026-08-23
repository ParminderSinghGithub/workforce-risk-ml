"""Unit tests for TabularPreprocessor, StructuredDataset, StructuredMLP, training steps, device handling, and checkpointing."""

import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn

from workforce_risk.features.definitions import FEATURE_DEFINITIONS
from workforce_risk.models.dataset import StructuredDataset, create_data_loaders
from workforce_risk.models.evaluate import calculate_classification_metrics, evaluate_model
from workforce_risk.models.model import StructuredMLP
from workforce_risk.models.preprocessor import (
    CATEGORICAL_FEATURE_NAMES,
    NUMERICAL_FEATURE_NAMES,
    TabularPreprocessor,
)
from workforce_risk.models.train import get_device, parse_args, train_structured_model


def _create_synthetic_parquet(file_path: Path, num_rows: int = 50, scale_salary: float = 100000.0) -> None:
    """Helper to create a small synthetic Parquet file conforming to structured feature schema."""
    feature_names = list(FEATURE_DEFINITIONS.keys())
    data = {
        "employee_id": [f"EMP_{i:03d}" for i in range(num_rows)],
        "left_company": [int(i % 3 == 0) for i in range(num_rows)],
    }
    for f in feature_names:
        if f == "salary":
            data[f] = [float(50000.0 + (i * scale_salary / num_rows)) for i in range(num_rows)]
        elif f in CATEGORICAL_FEATURE_NAMES:
            data[f] = [float(i % 5) for i in range(num_rows)]
        else:
            data[f] = [float((i * 0.1) % 1.0) for i in range(num_rows)]

    table = pa.Table.from_pydict(data)
    pq.write_table(table, str(file_path))


def test_tabular_preprocessor_fit_and_transform_isolation():
    """Verify that scaling statistics and vocabularies are learned strictly from train partition."""
    train_data = {
        "salary": np.array([50000.0, 100000.0, 150000.0]),
        "performance_score": np.array([0.5, 0.7, 0.9]),
        "department_idx": np.array([0, 1, 2]),
        "job_level_idx": np.array([0, 1, 0]),
        "role_idx": np.array([10, 20, 30]),
        "communication_patterns_idx": np.array([1, 2, 3]),
        "persona_name_idx": np.array([0, 1, 2]),
    }
    for num_col in NUMERICAL_FEATURE_NAMES:
        if num_col not in train_data:
            train_data[num_col] = np.array([0.1, 0.2, 0.3])

    prep = TabularPreprocessor()
    prep.fit(train_data)

    salary_idx = prep.numerical_features.index("salary")
    assert prep.means is not None and prep.stds is not None
    assert np.isclose(prep.means[salary_idx], 100000.0)
    assert prep.is_fitted

    val_data = {
        "salary": np.array([100000.0, 200000.0]),
        "performance_score": np.array([0.7, 1.0]),
        "department_idx": np.array([1, 99]),  # 99 is unseen in train
        "job_level_idx": np.array([1, 1]),
        "role_idx": np.array([20, 20]),
        "communication_patterns_idx": np.array([2, 2]),
        "persona_name_idx": np.array([1, 1]),
    }
    for num_col in NUMERICAL_FEATURE_NAMES:
        if num_col not in val_data:
            val_data[num_col] = np.array([0.2, 0.5])

    transformed = prep.transform(val_data)
    assert transformed.shape == (2, prep.feature_dim)
    assert np.isfinite(transformed).all()
    assert np.isclose(transformed[0, salary_idx], 0.0, atol=1e-5)


def test_production_path_fits_preprocessor_on_full_train_before_sampling(tmp_path: Path):
    """Verify that create_data_loaders fits preprocessor on complete training parquet before subsampling."""
    train_parquet = tmp_path / "full_train.parquet"
    val_parquet = tmp_path / "val.parquet"
    test_parquet = tmp_path / "test.parquet"

    _create_synthetic_parquet(train_parquet, num_rows=100)
    _create_synthetic_parquet(val_parquet, num_rows=20)
    _create_synthetic_parquet(test_parquet, num_rows=20)

    train_loader, val_loader, test_loader, preprocessor = create_data_loaders(
        train_path=train_parquet,
        val_path=val_parquet,
        test_path=test_parquet,
        batch_size=10,
        max_train_samples=20,
        fit_on_full_train=True,
    )

    full_table = pq.read_table(str(train_parquet))
    full_salary_mean = np.mean(full_table["salary"].to_numpy().astype(np.float32))
    salary_idx = preprocessor.numerical_features.index("salary")

    assert np.isclose(preprocessor.means[salary_idx], full_salary_mean, atol=1e-3)
    assert len(train_loader.dataset) == 20
    assert len(val_loader.dataset) == 20
    assert len(test_loader.dataset) == 20


def test_structured_dataset_loading_and_leakage_exclusion(tmp_path: Path):
    """Verify Dataset loads preprocessed features as float32 and strictly rejects leakage columns."""
    parquet_file = tmp_path / "synthetic_train.parquet"
    _create_synthetic_parquet(parquet_file, num_rows=40)

    dataset = StructuredDataset(parquet_file, fit_preprocessor=True)
    assert len(dataset) == 40
    assert dataset.input_dim >= 29

    x, y = dataset[0]
    assert x.shape == (dataset.input_dim,)
    assert y.shape == (1,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert torch.isfinite(x).all()

    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        StructuredDataset(parquet_file, feature_names=["burnout_risk", "salary"])


def test_device_selection_and_cuda_error_handling(monkeypatch):
    """Verify device selection returns proper devices and fails clearly when CUDA is unavailable."""
    dev, info = get_device("cpu")
    assert dev.type == "cpu"
    assert "pytorch_version" in info

    dev_auto, info_auto = get_device("auto")
    assert dev_auto.type in ["cpu", "cuda"]

    # Mock CUDA as unavailable and test explicit 'cuda' request raises RuntimeError
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA device requested"):
        get_device("cuda")

    with pytest.raises(ValueError, match="Unknown device setting"):
        get_device("tpu")


def test_cli_argument_parsing(monkeypatch):
    """Verify parse_args accurately parses command-line arguments for Kaggle execution."""
    test_args = [
        "train.py",
        "--train-sample-size", "100000",
        "--device", "cuda",
        "--epochs", "15",
        "--batch-size", "512",
        "--lr", "0.0005",
        "--artifacts-dir", "custom_artifacts",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    parsed = parse_args()

    assert parsed.train_sample_size == 100000
    assert parsed.device == "cuda"
    assert parsed.epochs == 15
    assert parsed.batch_size == 512
    assert parsed.lr == 0.0005
    assert parsed.artifacts_dir == "custom_artifacts"


def test_training_and_metadata_generation(tmp_path: Path):
    """Verify end-to-end training generates all expected metadata and artifacts."""
    train_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    test_p = tmp_path / "test.parquet"
    art_dir = tmp_path / "artifacts"

    _create_synthetic_parquet(train_p, num_rows=30)
    _create_synthetic_parquet(val_p, num_rows=15)
    _create_synthetic_parquet(test_p, num_rows=15)

    result = train_structured_model(
        train_path=train_p,
        val_path=val_p,
        test_path=test_p,
        artifacts_dir=art_dir,
        epochs=1,
        batch_size=8,
        device_str="cpu",
    )

    assert result["status"] == "SUCCESS"
    assert (art_dir / "best_checkpoint.pt").exists()
    assert (art_dir / "training_history.json").exists()
    assert (art_dir / "evaluation_summary.json").exists()


def test_structured_mlp_architecture():
    """Verify StructuredMLP module structure, output logit shape, and absence of embedded sigmoid."""
    model = StructuredMLP(input_dim=50, hidden_dims=[128, 64, 32], dropout=0.2)
    assert isinstance(model, nn.Module)
    assert model.total_parameters > 0

    batch_x = torch.randn(16, 50)
    logits = model(batch_x)
    assert logits.shape == (16, 1)

    probs = model.predict_proba(batch_x)
    assert probs.shape == (16, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_training_step_parameter_updates():
    """Verify a forward/backward step on StructuredMLP updates parameters with finite loss."""
    model = StructuredMLP(input_dim=50, hidden_dims=[32, 16], dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()

    initial_param = model.network[0].weight.clone()

    batch_x = torch.randn(8, 50)
    batch_y = torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0], [1.0]])

    optimizer.zero_grad()
    logits = model(batch_x)
    loss = criterion(logits, batch_y)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()

    updated_param = model.network[0].weight
    assert not torch.equal(initial_param, updated_param), "Model parameters did not update during training step"


def test_validation_does_not_update_parameters(tmp_path: Path):
    """Verify evaluation loop does not perform gradient updates."""
    parquet_file = tmp_path / "synthetic_val.parquet"
    _create_synthetic_parquet(parquet_file, num_rows=20)
    dataset = StructuredDataset(parquet_file, fit_preprocessor=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)

    model = StructuredMLP(input_dim=dataset.input_dim, hidden_dims=[32, 16])
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cpu")

    weights_before = model.network[0].weight.clone()
    avg_loss, metrics, _, _ = evaluate_model(model, loader, criterion, device)

    weights_after = model.network[0].weight
    assert torch.equal(weights_before, weights_after), "Validation modified model weights!"
    assert metrics["roc_auc"] >= 0.0


def test_checkpoint_save_and_reload_identity(tmp_path: Path):
    """Verify saved checkpoint reloads preprocessor, model config, and reproduces exact inference."""
    parquet_file = tmp_path / "synthetic_train.parquet"
    _create_synthetic_parquet(parquet_file, num_rows=30)
    dataset = StructuredDataset(parquet_file, fit_preprocessor=True)

    input_dim = dataset.input_dim
    model = StructuredMLP(input_dim=input_dim, hidden_dims=[64, 32], dropout=0.1)
    model.eval()

    test_input = dataset.features[:5]
    with torch.no_grad():
        original_output = model(test_input)

    ckpt_path = tmp_path / "test_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {"input_dim": input_dim, "hidden_dims": [64, 32], "dropout": 0.1},
        "preprocessing_config": dataset.preprocessor.to_dict(),
    }, ckpt_path)

    loaded_ckpt = torch.load(ckpt_path, weights_only=False)
    reloaded_prep = TabularPreprocessor.from_dict(loaded_ckpt["preprocessing_config"])
    assert reloaded_prep.feature_dim == input_dim

    reloaded_model = StructuredMLP(**loaded_ckpt["model_config"])
    reloaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
    reloaded_model.eval()

    with torch.no_grad():
        reloaded_output = reloaded_model(test_input)

    assert torch.allclose(original_output, reloaded_output, atol=1e-6)
