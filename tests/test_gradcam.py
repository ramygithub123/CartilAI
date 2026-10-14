"""Tests for Grad-CAM and overlays."""

from __future__ import annotations

import numpy as np
import torch

from src.gradcam import GradCAM, overlay, slice_importance
from src.model import CartilAINet


def test_gradcam_shapes_and_range():
    torch.manual_seed(0)
    model = CartilAINet(pretrained=False)
    probs, cams = GradCAM(model)(torch.randn(5, 3, 64, 64))
    assert probs.shape == (3,) and np.all((probs >= 0) & (probs <= 1))
    assert cams.shape == (3, 5, 64, 64)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6
    imp = slice_importance(cams)
    assert imp.shape == (3, 5)
    for row in imp:
        assert row.sum() == 0 or abs(row.sum() - 1) < 1e-5


def test_gradcam_does_not_break_no_grad_inference():
    model = CartilAINet(pretrained=False).eval()
    GradCAM(model)
    with torch.no_grad():
        assert model(torch.randn(2, 3, 32, 32)).shape == (3,)


def test_overlay_output():
    img = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    cam = np.linspace(0, 1, 64 * 64).reshape(64, 64)
    out = overlay(img, cam, alpha=0.5)
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
    plain = overlay(img, None)
    assert np.all(plain[..., 0] == plain[..., 1])
