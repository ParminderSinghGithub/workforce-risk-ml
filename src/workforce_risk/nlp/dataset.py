"""PyTorch Dataset and DataLoader constructors for text tokenization and batching."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class FeedbackTextDataset(Dataset):
    """PyTorch Dataset for tokenizing employee feedback texts with target labels."""

    def __init__(
        self,
        texts: List[str] | np.ndarray,
        labels: Optional[List[int] | np.ndarray] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 256,
    ) -> None:
        self.texts = [str(t) if t is not None else "" for t in texts]
        self.labels = np.asarray(labels, dtype=np.float32) if labels is not None else None
        self.max_length = max_length

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor([self.labels[idx]], dtype=torch.float32)

        return item

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str | Path,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        tokenizer_name: str = "distilbert-base-uncased",
        max_length: int = 256,
        text_column: str = "recent_feedback",
        target_column: str = "high_burnout_risk",
        max_samples: Optional[int] = None,
        random_seed: int = 42,
    ) -> "FeedbackTextDataset":
        """Construct Dataset directly from a Parquet split with optional deterministic sampling."""
        path = Path(parquet_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Text dataset not found at: {path}")

        table = pq.read_table(str(path), columns=[text_column, target_column])
        total_rows = table.num_rows

        if max_samples is not None and max_samples < total_rows:
            rng = np.random.RandomState(random_seed)
            indices = rng.choice(total_rows, size=max_samples, replace=False)
            indices.sort()
            texts = table[text_column].to_numpy()[indices]
            labels = table[target_column].to_numpy()[indices]
        else:
            texts = table[text_column].to_numpy()
            labels = table[target_column].to_numpy()

        return cls(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
        )


def create_text_data_loaders(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 256,
    batch_size: int = 32,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, PreTrainedTokenizerBase]:
    """Create train, validation, and test PyTorch DataLoaders with tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_ds = FeedbackTextDataset.from_parquet(
        parquet_path=train_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_train_samples,
        random_seed=random_seed,
    )
    val_ds = FeedbackTextDataset.from_parquet(
        parquet_path=val_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_val_samples,
        random_seed=random_seed,
    )
    test_ds = FeedbackTextDataset.from_parquet(
        parquet_path=test_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_test_samples,
        random_seed=random_seed,
    )

    generator = torch.Generator()
    generator.manual_seed(random_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, tokenizer
