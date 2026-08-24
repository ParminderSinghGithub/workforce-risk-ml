"""Kaggle execution wrapper for Workforce Risk DistilBERT + PEFT/LoRA text classifier training.

Validates input datasets, forwards execution parameters, and runs the official training pipeline.
Source of truth remains `workforce_risk.nlp.train`.
"""

import sys
from pathlib import Path

# Add project root to sys.path if invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workforce_risk.nlp.train import parse_args, train_text_transformer


def main() -> None:
    args = parse_args()
    print("[Runner] Initializing DistilBERT + LoRA Text Training Execution Wrapper...")

    # Default to 10,000 training samples on GPU if not explicitly provided (per project spec bounds 5k-10k)
    train_sample_size = args.train_sample_size if args.train_sample_size is not None else (200 if args.smoke_test else 10000)

    train_text_transformer(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        artifacts_dir=args.artifacts_dir,
        max_train_samples=train_sample_size,
        max_val_samples=args.val_sample_size,
        max_test_samples=args.test_sample_size,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
