"""PyTorch Dataset and DataLoader construction with leakage-safe tabular preprocessing."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from workforce_risk.features.definitions import (
    EXCLUDED_LEAKAGE_COLUMNS,
    FEATURE_DEFINITIONS,
)
from workforce_risk.models.preprocessor import (
    CATEGORICAL_FEATURE_NAMES,
    NUMERICAL_FEATURE_NAMES,
    TabularPreprocessor,
)

CANONICAL_STRUCTURED_FEATURES: List[str] = list(FEATURE_DEFINITIONS.keys())


class StructuredDataset(Dataset):
    """PyTorch tabular Dataset consuming Parquet partitions with strict preprocessing and leakage enforcement."""

    def __init__(
        self,
        parquet_path: str | Path,
        preprocessor: Optional[TabularPreprocessor] = None,
        fit_preprocessor: bool = False,
        feature_names: Optional[List[str]] = None,
        target_name: str = "left_company",
        max_samples: Optional[int] = None,
        random_seed: int = 42,
    ) -> None:
        self.parquet_path = Path(parquet_path).resolve()
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet dataset not found at: {self.parquet_path}")

        self.target_name = target_name
        self.raw_feature_names = feature_names or CANONICAL_STRUCTURED_FEATURES

        # 1. Strict leakage prevention assertions
        forbidden_in_features = set(self.raw_feature_names).intersection(
            set(EXCLUDED_LEAKAGE_COLUMNS)
        )
        if forbidden_in_features:
            raise ValueError(
                f"LEAKAGE VIOLATION: Forbidden columns present in feature list: {forbidden_in_features}"
            )

        # 2. Read only requested raw feature columns and target from Parquet
        required_columns = self.raw_feature_names + [self.target_name]
        table = pq.read_table(str(self.parquet_path), columns=required_columns)

        # 3. Deterministic subsampling if bounded sample requested (e.g. smoke test)
        total_rows = table.num_rows
        if max_samples is not None and max_samples < total_rows:
            rng = np.random.RandomState(random_seed)
            indices = rng.choice(total_rows, size=max_samples, replace=False)
            indices.sort()  # Maintain deterministic index order
            raw_data_dict = {
                col: table[col].to_numpy()[indices] for col in self.raw_feature_names
            }
            target_array = table[self.target_name].to_numpy()[indices].astype(np.float32)
        else:
            raw_data_dict = {
                col: table[col].to_numpy() for col in self.raw_feature_names
            }
            target_array = table[self.target_name].to_numpy().astype(np.float32)

        # 4. Fit or apply tabular preprocessor
        if fit_preprocessor:
            self.preprocessor = TabularPreprocessor(
                numerical_features=[c for c in self.raw_feature_names if c in NUMERICAL_FEATURE_NAMES],
                categorical_features=[c for c in self.raw_feature_names if c in CATEGORICAL_FEATURE_NAMES],
            )
            self.preprocessor.fit(raw_data_dict)
        elif preprocessor is not None:
            self.preprocessor = preprocessor
        else:
            # Standalone unscaled fallback (e.g. if explicitly passed pre-fitted)
            self.preprocessor = TabularPreprocessor(
                numerical_features=[c for c in self.raw_feature_names if c in NUMERICAL_FEATURE_NAMES],
                categorical_features=[c for c in self.raw_feature_names if c in CATEGORICAL_FEATURE_NAMES],
            )
            self.preprocessor.fit(raw_data_dict)

        # 5. Transform raw features to float32 tensor
        features_matrix = self.preprocessor.transform(raw_data_dict)
        self.features = torch.from_numpy(features_matrix).float()
        self.targets = torch.from_numpy(target_array).float().unsqueeze(1)  # [N, 1]

        # Verify finite values
        if not torch.isfinite(self.features).all():
            raise ValueError("Feature tensor contains non-finite values (NaN or Inf).")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    @property
    def input_dim(self) -> int:
        return self.preprocessor.feature_dim

    @property
    def encoded_feature_names(self) -> List[str]:
        return self.preprocessor.encoded_feature_names


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
) -> Tuple[DataLoader, DataLoader, DataLoader, TabularPreprocessor]:
    """Create train, validation, and test PyTorch DataLoaders with train-fitted preprocessing."""
    # 1. Fit preprocessor strictly on training split
    train_dataset = StructuredDataset(
        parquet_path=train_path,
        fit_preprocessor=True,
        max_samples=max_train_samples,
        random_seed=random_seed,
    )
    preprocessor = train_dataset.preprocessor

    # 2. Apply fitted preprocessor to validation and test splits (zero data leakage)
    val_dataset = StructuredDataset(
        parquet_path=val_path,
        preprocessor=preprocessor,
        max_samples=max_val_samples,
        random_seed=random_seed,
    )
    test_dataset = StructuredDataset(
        parquet_path=test_path,
        preprocessor=preprocessor,
        max_samples=max_test_samples,
        random_seed=random_seed,
    )

    # 3. Deterministic DataLoader generators
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

    return train_loader, val_loader, test_loader, preprocessor
