"""Execution runner for Multimodal Late Fusion pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workforce_risk.fusion.train import parse_args, run_multimodal_fusion

if __name__ == "__main__":
    args = parse_args()
    run_multimodal_fusion(
        structured_checkpoint_path=args.structured_checkpoint,
        text_model_dir=args.text_model_dir,
        text_baseline_path=args.text_baseline,
        splits_dir=args.splits_dir,
        artifacts_dir=args.artifacts_dir,
        c_param=args.c_param,
        device_str=args.device,
        seed=args.seed,
    )
