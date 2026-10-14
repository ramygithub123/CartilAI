"""Tests for evaluation metrics and plots."""

from __future__ import annotations

import json

import numpy as np

from src.evaluate import compute_metrics, plot_confusion_matrices, plot_roc_curves, safe_auc, save_json, youden_threshold

FINDINGS = ["abnormal", "acl", "meniscus"]


def test_perfect_predictions():
    y = np.array([[0, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=float)
    m = compute_metrics(y, y * 0.8 + 0.1, FINDINGS, [0.5] * 3, seed=0)
    for f in FINDINGS:
        assert m[f]["auc"] == 1.0 and m[f]["sensitivity"] == 1.0 and m[f]["specificity"] == 1.0
        assert m[f]["tp"] + m[f]["fn"] == m[f]["n_positive"]


def test_threshold_and_auc_edge_cases():
    assert np.isnan(safe_auc(np.ones(4), np.random.rand(4)))
    assert youden_threshold(np.ones(4), np.random.rand(4)) == 0.5
    t = youden_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.2 < t <= 0.8


def test_save_json_turns_nan_into_null(tmp_path):
    save_json({"a": float("nan"), "b": np.float32(1.5), "c": [np.int64(2)]}, tmp_path / "m.json")
    assert json.loads((tmp_path / "m.json").read_text()) == {"a": None, "b": 1.5, "c": [2]}


def test_plots_are_written(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (30, 3)).astype(float)
    p = rng.random((30, 3))
    names = dict(zip(FINDINGS, FINDINGS))
    plot_roc_curves(y, p, FINDINGS, names, tmp_path / "roc.png")
    plot_confusion_matrices(compute_metrics(y, p, FINDINGS, [0.5] * 3), FINDINGS, names, tmp_path / "cm.png")
    assert (tmp_path / "roc.png").stat().st_size > 1000 and (tmp_path / "cm.png").stat().st_size > 1000


def test_youden_never_returns_predict_nothing_threshold():
    # an uninformative model must not get a threshold that labels everything negative
    t = youden_threshold(np.array([0, 1, 0, 1]), np.array([0.9, 0.1, 0.8, 0.2]))
    assert t == 0.5
