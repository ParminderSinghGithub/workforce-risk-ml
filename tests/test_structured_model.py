"""Unit tests for PyTorch dataset construction, StructuredMLP architecture, training steps, and checkpointing."""

from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn

from workforce_risk.features.definitions import FEATURE_DEFINITIONS
from workforce_risk.models.dataset import StructuredDataset
from workforce_risk.models.evaluate import calculate_classification_metrics, evaluate_model
from workforce_risk.models.model import StructuredMLP


def _create_synthetic_parquet(file_path: Path, num_rows: int = 50) -> None:
    """Helper to create a small synthetic Parquet file conforming to structured feature schema."""
    feature_names = list(FEATURE_DEFINITIONS.keys())
    data = {
        "employee_id": [f"EMP_{i:03d}" for i in range(num_rows)],
        "left_company": [int(i % 3 == 0) for i in range(num_rows)],
    }
    for f in feature_names:
        data[f] = [float((i * 0.1) % 1.0) for i in range(num_rows)]

    table = pa.Table.from_pydict(data)
    pq.write_table(table, str(file_path))


def test_structured_dataset_loading_and_leakage_exclusion(tmp_path: Path):
    """Verify Dataset loads exact 29 features as float32 and strictly rejects leakage columns."""
    parquet_file = tmp_path / "synthetic_train.parquet"
    _create_synthetic_parquet(parquet_file, num_rows=40)

    dataset = StructuredDataset(parquet_file)
    assert len(dataset) == 40
    assert dataset.input_dim == 29

    x, y = dataset[0]
    assert x.shape == (29,)
    assert y.shape == (1,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32

    # Leakage exclusion assertion: passing forbidden column raises ValueError
    with pytest.raises(ValueError, match="LEAKAGE VIOLATION"):
        StructuredDataset(parquet_file, feature_names=["burnout_risk", "salary"])


def test_structured_dataset_deterministic_sampling(tmp_path: Path):
    """Verify deterministic max_samples subsampling produces identical rows for same seed."""
    parquet_file = tmp_path / "synthetic_train.parquet"
    _create_synthetic_parquet(parquet_file, num_rows=100)

    ds1 = StructuredDataset(parquet_file, max_samples=25, random_seed=42)
    ds2 = StructuredDataset(parquet_file, max_samples=25, random_seed=42)
    ds3 = StructuredDataset(parquet_file, max_samples=25, random_seed=99)

    assert len(ds1) == 25
    assert len(ds2) == 25
    assert len(ds3) == 25

    # ds1 and ds2 must be identical
    assert torch.equal(ds1.features, ds2.features)
    assert torch.equal(ds1.targets, ds2.targets)

    # ds3 with different seed should differ
    assert not torch.equal(ds1.features, ds3.features)


def test_structured_mlp_architecture():
    """Verify StructuredMLP module structure, output logit shape, and absence of embedded sigmoid."""
    model = StructuredMLP(input_dim=29, hidden_dims=[128, 64, 32], dropout=0.2)
    assert isinstance(model, nn.Module)
    assert model.total_parameters > 0

    batch_x = torch.randn(16, 29)
    logits = model(batch_x)
    assert logits.shape == (16, 1)

    # Predict proba applies sigmoid
    probs = model.predict_proba(batch_x)
    assert probs.shape == (16, 1)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_training_step_parameter_updates():
    """Verify a forward/backward step on StructuredMLP updates parameters with finite loss."""
    model = StructuredMLP(input_dim=29, hidden_dims=[32, 16], dropout=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCEWithLogitsLoss()

    initial_param = model.network[0].weight.clone()

    batch_x = torch.randn(8, 29)
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
    dataset = StructuredDataset(parquet_file)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)

    model = StructuredMLP(input_dim=29, hidden_dims=[32, 16])
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cpu")

    weights_before = model.network[0].weight.clone()
    avg_loss, metrics, _, _ = evaluate_model(model, loader, criterion, device)

    weights_after = model.network[0].weight
    assert torch.equal(weights_before, weights_after), "Validation modified model weights!"
    assert metrics["roc_auc"] >= 0.0


def test_metrics_calculation_and_edge_cases():
    """Verify calculate_classification_metrics returns accurate bounds and handles edge cases."""
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.2, 0.3, 0.85, 0.15])

    metrics = calculate_classification_metrics(y_true, y_prob, threshold=0.5)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert metrics["roc_auc"] > 0.9  # Good predictions should have high ROC-AUC

    # Single class edge case should not crash
    single_class_metrics = calculate_classification_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert "roc_auc" in single_class_metrics


def test_checkpoint_save_and_reload_identity(tmp_path: Path):
    """Verify saved checkpoint reloads exactly and produces identical predictions."""
    model = StructuredMLP(input_dim=29, hidden_dims=[64, 32], dropout=0.1)
    model.eval()

    test_input = torch.randn(10, 29)
    with torch.no_grad():
        original_output = model(test_input)

    ckpt_path = tmp_path / "test_checkpoint.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {"input_dim": 29, "hidden_dims": [64, 32], "dropout": 0.1},
    }, ckpt_path)

    loaded_ckpt = torch.load(ckpt_path, weights_only=False)
    reloaded_model = StructuredMLP(**loaded_ckpt["model_config"])
    reloaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
    reloaded_model.eval()

    with torch.no_grad():
        reloaded_output = reloaded_model(test_input)

    assert torch.allclose(original_output, reloaded_output, atol=1e-6)
