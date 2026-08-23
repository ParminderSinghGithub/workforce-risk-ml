"""Kaggle execution wrapper for Workforce Risk PyTorch structured model training.

Validates input datasets, forwards execution parameters, and runs the official training pipeline.
Source of truth remains `workforce_risk.models.train`.
"""

import sys
from pathlib import Path

# Add project root to sys.path if invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workforce_risk.models.train import parse_args, train_structured_model


def main() -> None:
    args = parse_args()
    print("[Runner] Initializing Structured Training Execution Wrapper...")

    # Default to 100K training samples on GPU if not explicitly provided
    train_sample_size = args.train_sample_size if args.train_sample_size is not None else (5000 if args.smoke_test else 100000)

    train_structured_model(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        artifacts_dir=args.artifacts_dir,
        max_train_samples=train_sample_size,
        max_val_samples=args.val_sample_size,
        max_test_samples=args.test_sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
