"""Tests for the CartilAI network."""

from __future__ import annotations

import torch

from src.model import CartilAINet


def test_forward_handles_any_slice_count():
    torch.manual_seed(0)
    model = CartilAINet(num_findings=3, pretrained=False).eval()
    for s in (1, 7, 30):
        assert model(torch.randn(s, 3, 64, 64)).shape == (3,)


def test_max_pool_ignores_slice_order():
    torch.manual_seed(0)
    model = CartilAINet(pretrained=False).eval()
    x = torch.randn(6, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x.flip(0)), atol=1e-5)


def test_gradients_reach_backbone():
    model = CartilAINet(pretrained=False)
    model(torch.randn(3, 3, 64, 64)).sum().backward()
    assert model.features[0].weight.grad is not None
