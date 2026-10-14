"""The Kaggle notebook is generated from src/ and must run top to bottom."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from scripts.build_notebook import NOTEBOOK, build


def test_committed_notebook_is_up_to_date(tmp_path, monkeypatch):
    committed = NOTEBOOK.read_text(encoding="utf-8")
    try:
        build()
        assert NOTEBOOK.read_text(encoding="utf-8") == committed, "Run: python scripts/build_notebook.py"
    finally:
        NOTEBOOK.write_text(committed, encoding="utf-8")


def test_notebook_runs_end_to_end(synthetic_root, tmp_path, monkeypatch):
    monkeypatch.setenv("CARTILAI_INPUT_DIR", str(synthetic_root))
    monkeypatch.setenv("CARTILAI_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("CARTILAI_QUICK", "1")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(tmp_path)}})
    with zipfile.ZipFile(tmp_path / "cartilai_outputs.zip") as zf:
        names = zf.namelist()
    assert "cartilai_outputs/models/cartilai_resnet18.pt" in names
    assert "cartilai_outputs/results/metrics.json" in names
    metrics = json.loads((tmp_path / "cartilai_outputs/results/metrics.json").read_text())
    assert metrics["project"] == "CartilAI"
