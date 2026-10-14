"""Configuration loading and reproducibility helpers."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config file into a plain dictionary.

    Args:
        path: Optional path to a YAML file. Defaults to ``config.yaml`` at the repo root.

    Returns:
        The configuration as nested dictionaries.
    """
    import yaml

    with open(path or DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base``.

    Args:
        base: The base configuration.
        overrides: Values that replace or extend ``base``.

    Returns:
        A new merged dictionary; inputs are not modified.
    """
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_seed(seed: int) -> None:
    """Fix all random seeds (Python, NumPy, PyTorch) for reproducible runs.

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # keep cuDNN fast; seeds still fix init and data order
    torch.backends.cudnn.benchmark = True
