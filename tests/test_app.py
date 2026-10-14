"""Smoke tests for the Gradio app in demo mode (no browser needed)."""

from __future__ import annotations

import gradio as gr
import numpy as np
import pytest

import app


def test_app_builds_and_reports_mode():
    assert isinstance(app.build_demo(), gr.Blocks)
    assert "CartilAI" in app.header_html() or "Cartil" in app.header_html()
    assert "not for clinical use" in app.header_html()
    assert "About the model" in app.about_markdown()


def test_analyze_sample_end_to_end():
    out = app.analyze_sample(app.SAMPLE_TITLES[1], 0.45)
    analysis, findings_html, _, _, image, strip, report, _ = out
    assert image.shape == (512, 512, 3)
    assert "ACL tear" in findings_html and "Slice relevance" in strip
    assert "DISCLAIMER" in report
    for name in [c.name for c in analysis.calls] + [app.NO_OVERLAY]:
        s = app.best_slice(analysis, name)
        assert 1 <= s <= analysis.volume.shape[0]
        assert app.render(analysis, name, s, 0.5).shape == (512, 512, 3)


def test_upload_validation(tmp_path):
    good = tmp_path / "scan.npy"
    np.save(good, np.random.default_rng(0).integers(0, 255, (12, 256, 256), dtype=np.uint8))
    assert app.analyze_upload(str(good), 0.4)[0].volume.shape == (12, 256, 256)
    bad = tmp_path / "bad.npy"
    bad.write_text("nope")
    with pytest.raises(gr.Error):
        app.analyze_upload(str(bad), 0.4)
    with pytest.raises(gr.Error):
        app.analyze_upload(None, 0.4)
