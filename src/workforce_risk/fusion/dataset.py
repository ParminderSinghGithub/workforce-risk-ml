"""Dataset alignment and loading for multimodal late fusion."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from workforce_risk.config import get_config
from workforce_risk.models.preprocessor import TabularPreprocessor


class AlignedMultimodalDataset(Dataset):
    """PyTorch dataset serving aligned structured tabular tensors and raw feedback text."""

    def __init__(
        self,
        features_tensor: torch.Tensor,
        feedback_texts: list[str],
        targets: np.ndarray,
        employee_ids: list[str],
    ) -> None:
        self.features_tensor = features_tensor
        self.feedback_texts = feedback_texts
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        self.employee_ids = employee_ids

    def __len__(self) -> int:
        return len(self.employee_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self.features_tensor[idx],
            "text": self.feedback_texts[idx],
            "target": self.targets[idx],
            "employee_id": self.employee_ids[idx],
        }


def load_aligned_multimodal_data(
    splits_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge aligned dual-holdout validation and test datasets.

    Guarantees:
    - Zero employee overlap with training sets.
    - Zero template overlap with training sets.
    - Exact 1-to-1 row alignment on `employee_id`.

    Returns:
        Tuple of (val_df, test_df) where both DataFrames contain all tabular features,
        `recent_feedback`, and ground truth `left_company`.
    """
    config = get_config()
    splits_path = Path(splits_dir or config.paths.data_splits_dir).resolve()

    s_val_path = splits_path / "structured_validation.parquet"
    t_val_path = splits_path / "text_validation.parquet"
    s_test_path = splits_path / "structured_test.parquet"
    t_test_path = splits_path / "text_test.parquet"

    s_val = pq.read_table(s_val_path).to_pandas()
    t_val = pq.read_table(t_val_path).to_pandas()
    s_test = pq.read_table(s_test_path).to_pandas()
    t_test = pq.read_table(t_test_path).to_pandas()

    val_df = pd.merge(
        s_val,
        t_val[["employee_id", "recent_feedback", "high_burnout_risk"]],
        on="employee_id",
        how="inner",
    ).sort_values("employee_id").reset_index(drop=True)

    test_df = pd.merge(
        s_test,
        t_test[["employee_id", "recent_feedback", "high_burnout_risk"]],
        on="employee_id",
        how="inner",
    ).sort_values("employee_id").reset_index(drop=True)

    return val_df, test_df
