"""Grad-CAM for the slice-wise CartilAI network, plus heatmap overlays.

For finding *k* we back-propagate its logit to the last convolutional block
(ResNet18 ``layer4``) of every slice. Per slice, each feature map is weighted
by its average gradient, summed and passed through ReLU (Selvaraju et al.,
2017). Because the exam score comes from a max over slices, slices that did
not contribute receive (near) zero gradient, so the maps also show *which
slices* drove the prediction, not only where inside a slice.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.model import CartilAINet


class GradCAM:
    """Compute per-slice Grad-CAM maps for every finding of one exam."""

    def __init__(self, model: CartilAINet) -> None:
        """Attach hooks to the model's last convolutional block.

        Args:
            model: A :class:`CartilAINet` (evaluation mode is set during use).
        """
        self.model = model
        self._acts: torch.Tensor | None = None
        model.gradcam_layer.register_forward_hook(self._save_acts)

    def _save_acts(self, _module: torch.nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        if out.requires_grad:  # the hook also fires during plain no-grad inference
            out.retain_grad()
        self._acts = out

    def __call__(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Run the model and compute Grad-CAM for every output.

        Args:
            x: Preprocessed volume ``(S, 3, H, W)``.

        Returns:
            ``(probs, cams)`` where ``probs`` has shape ``(k,)`` and ``cams`` has shape
            ``(k, S, H, W)`` with values in [0, 1], normalised per finding over the
            whole volume.
        """
        self.model.eval()
        x = x.detach().requires_grad_(False)
        with torch.enable_grad():
            logits = self.model(x)
            acts = self._acts
            assert acts is not None, "forward hook did not fire"
            cams = []
            for k in range(logits.shape[0]):
                if acts.grad is not None:
                    acts.grad = None
                logits[k].backward(retain_graph=k < logits.shape[0] - 1)
                grads = acts.grad.detach()                          # (S, C, h, w)
                weights = grads.mean(dim=(2, 3), keepdim=True)      # (S, C, 1, 1)
                cam = F.relu((weights * acts.detach()).sum(dim=1))  # (S, h, w)
                cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
                peak = cam.max()
                cams.append((cam / peak if peak > 0 else cam).cpu().numpy())
        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        self.model.zero_grad(set_to_none=True)
        return probs, np.stack(cams).astype(np.float32)


def slice_importance(cams: np.ndarray) -> np.ndarray:
    """Relative importance of each slice for each finding.

    Args:
        cams: ``(k, S, H, W)`` Grad-CAM maps.

    Returns:
        ``(k, S)`` array; each row sums to 1 (or is all zeros).
    """
    mass = cams.reshape(cams.shape[0], cams.shape[1], -1).sum(-1)
    total = mass.sum(1, keepdims=True)
    return np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)


def window_slice(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Contrast-stretch a slice to 0..1 using intensity window ``[lo, hi]``."""
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def overlay(
    slice_img: np.ndarray,
    cam: np.ndarray | None,
    alpha: float = 0.45,
    window: tuple[float, float] = (0.0, 255.0),
    cmap: str = "jet",
    min_visible: float = 0.15,
) -> np.ndarray:
    """Blend a heatmap onto a grayscale slice.

    Args:
        slice_img: ``(H, W)`` uint8 slice.
        cam: ``(H, W)`` map in [0, 1], or ``None`` for the plain slice.
        alpha: Maximum heatmap opacity.
        window: Intensity window for the grayscale image.
        cmap: Matplotlib colormap name.
        min_visible: Heat values below this are not drawn (keeps anatomy readable).

    Returns:
        ``(H, W, 3)`` uint8 RGB image.
    """
    import matplotlib

    gray = window_slice(slice_img, *window)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if cam is not None and alpha > 0:
        heat = matplotlib.colormaps[cmap](np.clip(cam, 0, 1))[..., :3]
        a = alpha * np.clip((cam - min_visible) / (1 - min_visible), 0, 1)[..., None]
        rgb = rgb * (1 - a) + heat * a
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
