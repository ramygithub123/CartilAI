"""Shared fixtures: a tiny synthetic MRNet dataset and a quickly trained run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.config import load_config, merge_config
from src.synthetic import make_synthetic_mrnet
from src.train import QUICK_OVERRIDES, run_pipeline

TINY = {"data": {"image_size": 64}}


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Quick CPU config."""
    return merge_config(merge_config(load_config(), QUICK_OVERRIDES), TINY)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic dataset nested like a Kaggle input folder."""
    base = tmp_path_factory.mktemp("kaggle_input")
    make_synthetic_mrnet(base / "mrnet-v1" / "MRNet-v1.0", n_train=28, n_valid=12, image_size=64, seed=1)
    return base


@pytest.fixture(scope="session")
def trained_run(config: dict[str, Any], synthetic_root: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the full training pipeline once and return its output folder."""
    out = tmp_path_factory.mktemp("run")
    run_pipeline(config, synthetic_root, out, log=lambda *_: None)
    return out
