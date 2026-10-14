"""Tests for the draft report."""

from __future__ import annotations

import numpy as np

from src.report import DISCLAIMER, categorize, generate_report, make_calls, top_slices

NAMES = {"abnormal": "Abnormal", "acl": "ACL tear", "meniscus": "Meniscal tear"}


def _report(probs, demo=False):
    imp = np.zeros((3, 10))
    imp[:, 4] = 1.0
    calls = make_calls(probs, [0.5, 0.5, 0.5], list(NAMES), NAMES, imp, margin=0.15)
    return calls, generate_report(calls, n_slices=10, demo_mode=demo)


def test_categorize_bands():
    assert categorize(0.9, 0.5, 0.15) == "likely present"
    assert categorize(0.55, 0.5, 0.15) == "indeterminate"
    assert categorize(0.1, 0.5, 0.15) == "not detected"


def test_top_slices_are_one_based_and_sorted():
    assert top_slices(np.array([0.1, 0.5, 0.0, 0.4]), k=2) == [2, 4]
    assert top_slices(np.zeros(4)) == []


def test_report_positive_acl():
    calls, text = _report([0.95, 0.9, 0.1])
    assert "suggestive of a tear of the anterior cruciate ligament (ACL)" in text
    assert "Most influential slices: 5" in text
    assert "No meniscal tear detected" in text
    assert DISCLAIMER in text and "not for clinical use" in text
    assert "future injury risk" in text


def test_report_normal_and_demo_flags():
    _, text = _report([0.05, 0.05, 0.05], demo=True)
    assert "DEMO MODE" in text
    assert "1. No ACL tear, meniscal tear or other abnormality detected by the model." in text


def test_report_abnormal_without_tear():
    _, text = _report([0.9, 0.1, 0.1])
    assert "Abnormal exam without a specific ACL or meniscal tear" in text
