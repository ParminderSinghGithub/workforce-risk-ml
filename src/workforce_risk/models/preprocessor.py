"""Numerical scaling and categorical one-hot encoding for tabular PyTorch model."""

from typing import Any, Dict, List, Optional
import numpy as np

from workforce_risk.features.definitions import FEATURE_DEFINITIONS

# Canonical column groupings
CATEGORICAL_FEATURE_NAMES: List[str] = [
    "department_idx",
    "job_level_idx",
    "role_idx",
    "communication_patterns_idx",
    "persona_name_idx",
]

NUMERICAL_FEATURE_NAMES: List[str] = [
    f for f in FEATURE_DEFINITIONS.keys() if f not in CATEGORICAL_FEATURE_NAMES
]


class TabularPreprocessor:
    """Numerical standardizer and categorical one-hot encoder fitted strictly on training data.

    Preserves exact learned parameters (means, standard deviations, categorical vocabularies)
    for deterministic inference and checkpoint reproducibility.
    """

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> None:
        self.numerical_features = numerical_features or NUMERICAL_FEATURE_NAMES
        self.categorical_features = categorical_features or CATEGORICAL_FEATURE_NAMES

        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.cat_vocabs: Dict[str, List[int]] = {}
        self.encoded_feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, data_dict: Dict[str, np.ndarray]) -> "TabularPreprocessor":
        """Fit standardization statistics and categorical vocabularies on training partition."""
        # 1. Numerical scaling statistics
        num_matrix = np.column_stack(
            [data_dict[col].astype(np.float32) for col in self.numerical_features]
        )
        self.means = np.mean(num_matrix, axis=0)
        self.stds = np.std(num_matrix, axis=0)
        # Avoid division by zero
        self.stds = np.where(self.stds < 1e-6, 1.0, self.stds)

        # 2. Categorical vocabularies
        self.cat_vocabs = {}
        encoded_names = list(self.numerical_features)

        for col in self.categorical_features:
            col_vals = data_dict[col].astype(int)
            vocab = sorted(list(set(col_vals.tolist())))
            self.cat_vocabs[col] = vocab
            for v in vocab:
                encoded_names.append(f"{col}_{v}")

        self.encoded_feature_names = encoded_names
        self.is_fitted = True
        return self

    def transform(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Transform tabular features into a standardized, one-hot encoded float32 matrix."""
        if not self.is_fitted or self.means is None or self.stds is None:
            raise RuntimeError("TabularPreprocessor must be fitted before calling transform().")

        # 1. Standardize numeric columns
        num_matrix = np.column_stack(
            [data_dict[col].astype(np.float32) for col in self.numerical_features]
        )
        num_scaled = (num_matrix - self.means) / self.stds

        # 2. One-hot encode categorical columns against fitted vocabularies
        cat_oh_blocks: List[np.ndarray] = []
        num_rows = num_matrix.shape[0]

        for col in self.categorical_features:
            vocab = self.cat_vocabs[col]
            col_vals = data_dict[col].astype(int)

            oh_block = np.zeros((num_rows, len(vocab)), dtype=np.float32)
            for idx, val in enumerate(vocab):
                oh_block[col_vals == val, idx] = 1.0
            cat_oh_blocks.append(oh_block)

        # 3. Concatenate continuous and one-hot blocks
        if cat_oh_blocks:
            final_matrix = np.hstack([num_scaled] + cat_oh_blocks)
        else:
            final_matrix = num_scaled

        return final_matrix.astype(np.float32)

    @property
    def feature_dim(self) -> int:
        """Return total output feature dimensionality after preprocessing."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted.")
        return len(self.encoded_feature_names)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fitted preprocessor state into JSON-compatible dictionary."""
        return {
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "means": self.means.tolist() if self.means is not None else [],
            "stds": self.stds.tolist() if self.stds is not None else [],
            "cat_vocabs": {k: [int(v) for v in vals] for k, vals in self.cat_vocabs.items()},
            "encoded_feature_names": self.encoded_feature_names,
            "feature_dim": self.feature_dim,
            "is_fitted": self.is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TabularPreprocessor":
        """Deserialize preprocessor state from dictionary."""
        preprocessor = cls(
            numerical_features=data["numerical_features"],
            categorical_features=data["categorical_features"],
        )
        preprocessor.means = np.array(data["means"], dtype=np.float32)
        preprocessor.stds = np.array(data["stds"], dtype=np.float32)
        preprocessor.cat_vocabs = {k: list(v) for k, v in data["cat_vocabs"].items()}
        preprocessor.encoded_feature_names = data["encoded_feature_names"]
        preprocessor.is_fitted = data.get("is_fitted", True)
        return preprocessor
