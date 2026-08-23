"""Seed management module for reproducible execution across all pipeline components."""

import os
import random
from typing import Optional


def set_seed(seed: int = 42, deterministic_torch: bool = True) -> int:
    """Set global random seeds across Python random, NumPy, and PyTorch (if available).

    Args:
        seed: The integer seed value. Defaults to 42.
        deterministic_torch: Whether to configure PyTorch for deterministic algorithms.

    Returns:
        The integer seed value that was configured.
    """
    # 1. Python built-in random & hash seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. NumPy (if installed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    # 3. PyTorch (if installed in later milestones)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic_torch:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
