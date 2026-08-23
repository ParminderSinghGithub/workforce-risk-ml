"""PyTorch Dataset and DataLoader construction for structured tabular features."""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from workforce_risk.features.definitions import (
    EXCLUDED_LEAKAGE_COLUMNS,
    FEATURE_DEFINITIONS,
)

CANONICAL_STRUCTURED_FEATURES: List[str] = list(FEATURE_DEFINITIONS.keys())


class StructuredDataset(Dataset):
    """PyTorch tabular Dataset consuming partitioned Parquet files with leakage enforcement."""

    def __init__(
        self,
        parquet_path: str | Path,
        feature_names: Optional[List[str]] = None,
        target_name: str = "left_company",
        max_samples: Optional[int] = None,
        random_seed: int = 42,
    ) -> None:
        self.parquet_path = Path(parquet_path).resolve()
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet dataset not found at: {self.parquet_path}")

        self.target_name = target_name
        self.feature_names = feature_names or CANONICAL_STRUCTURED_FEATURES

        # 1. Strict leakage prevention assertions
        forbidden_in_features = set(self.feature_names).intersection(
            set(EXCLUDED_LEAKAGE_COLUMNS)
        )
        if forbidden_in_features:
            raise ValueError(
                f"LEAKAGE VIOLATION: Forbidden columns present in feature list: {forbidden_in_features}"
            )

        # 2. Read only requested feature columns and target from Parquet
        required_columns = self.feature_names + [self.target_name]
        table = pq.read_table(str(self.parquet_path), columns=required_columns)

        # 3. Convert to float32 NumPy arrays
        features_dict = {col: table[col].to_numpy().astype(np.float32) for col in self.feature_names}
        features_matrix = np.column_stack([features_dict[col] for col in self.feature_names])
        target_array = table[self.target_name].to_numpy().astype(np.float32)

        total_rows = len(target_array)

        # 4. Deterministic sampling if bounded sample requested (e.g. smoke test)
        if max_samples is not None and max_samples < total_rows:
            rng = np.random.RandomState(random_seed)
            indices = rng.choice(total_rows, size=max_samples, replace=False)
            indices.sort()  # Maintain deterministic index order
            features_matrix = features_matrix[indices]
            target_array = target_array[indices]

        self.features = torch.from_numpy(features_matrix).float()
        self.targets = torch.from_numpy(target_array).float().unsqueeze(1)  # [N, 1]

        # Verify dimensionality
        if self.features.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature matrix width ({self.features.shape[1]}) does not match feature count ({len(self.feature_names)})"
            )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)


def create_data_loaders(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    batch_size: int = 256,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train, validation, and test PyTorch DataLoaders with deterministic settings."""
    train_dataset = StructuredDataset(
        parquet_path=train_path,
        max_samples=max_train_samples,
        random_seed=random_seed,
    )
    val_dataset = StructuredDataset(
        parquet_path=val_path,
        max_samples=max_val_samples,
        random_seed=random_seed,
    )
    test_dataset = StructuredDataset(
        parquet_path=test_path,
        max_samples=max_test_samples,
        random_seed=random_seed,
    )

    # Deterministic DataLoader generator
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, train_dataset.feature_names
