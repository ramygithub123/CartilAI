"""High-level inference used by the app: load weights (or demo mode), analyse a volume."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import PROJECT_ROOT, load_config, set_seed
from src.data import preprocess, to_uint8_volume
from src.gradcam import GradCAM, slice_importance
from src.model import CartilAINet, build_model
from src.report import FindingCall, generate_report, make_calls


@dataclass
class Analysis:
    """Everything the UI needs to display one analysed exam."""

    volume: np.ndarray                 # (S, H, W) uint8
    probs: np.ndarray                  # (k,)
    cams: np.ndarray                   # (k, S, H, W) in [0, 1]
    importance: np.ndarray             # (k, S)
    calls: list[FindingCall]
    report: str
    window: tuple[float, float]
    exam_label: str = "Uploaded exam"
    meta: dict[str, Any] = field(default_factory=dict)


class Predictor:
    """Wraps the model and switches automatically between demo mode and the trained model.

    If the weights file from ``config.yaml`` (``models/cartilai_resnet18.pt``)
    exists and loads, the trained model is used; otherwise the app runs in
    demo mode with seeded, untrained weights.
    """

    def __init__(self, config: dict[str, Any] | None = None, weights_path: str | Path | None = None) -> None:
        """Load the model.

        Args:
            config: CartilAI config (defaults to ``config.yaml``).
            weights_path: Override the weights file location.
        """
        self.config = copy.deepcopy(config) if config else load_config()
        self.findings: list[str] = list(self.config["findings"])
        self.names: dict[str, str] = dict(self.config["finding_names"])
        self.weights_path = Path(weights_path) if weights_path else PROJECT_ROOT / self.config["paths"]["weights_file"]
        self.thresholds = [0.5] * len(self.findings)
        self.load_error: str | None = None
        self.checkpoint_info: dict[str, Any] = {}
        self.device = torch.device("cpu")
        torch.set_num_threads(max(1, torch.get_num_threads()))

        self.model: CartilAINet = build_model(self.config, pretrained=False)
        self.demo_mode = True
        if self.weights_path.exists():
            try:
                ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
                state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                self.model.load_state_dict(state)
                if isinstance(ckpt, dict):
                    if ckpt.get("findings") and list(ckpt["findings"]) != self.findings:
                        raise ValueError(f"checkpoint findings {ckpt['findings']} != config {self.findings}")
                    self.thresholds = [float(t) for t in ckpt.get("thresholds", self.thresholds)]
                    size = ckpt.get("config", {}).get("data", {}).get("image_size")
                    if size:  # analyse at the resolution the model was trained on
                        self.config["data"]["image_size"] = int(size)
                    self.checkpoint_info = {k: v for k, v in ckpt.items() if k not in {"state_dict"}}
                self.demo_mode = False
            except Exception as exc:  # fall back to demo mode, but tell the user why
                self.load_error = f"Could not load {self.weights_path.name}: {exc}"
                set_seed(self.config["seed"])
                self.model = build_model(self.config, pretrained=False)
        else:
            set_seed(self.config["seed"])
            self.model = build_model(self.config, pretrained=False)
        self.model.eval().to(self.device)
        self.gradcam = GradCAM(self.model)

    @property
    def mode_label(self) -> str:
        """Short human-readable description of the active mode."""
        return "Demo mode — untrained weights" if self.demo_mode else "Trained model"

    def analyze(self, volume: np.ndarray, exam_label: str = "Uploaded exam") -> Analysis:
        """Score a volume, compute Grad-CAM for each finding and draft a report.

        Args:
            volume: Any 3D array accepted by :func:`src.data.to_uint8_volume`.
            exam_label: Label shown in the report.

        Returns:
            An :class:`Analysis`.
        """
        vol = to_uint8_volume(volume, self.config["data"]["image_size"])
        x = preprocess(torch.from_numpy(vol).to(self.device))
        probs, cams = self.gradcam(x)
        importance = slice_importance(cams)
        calls = make_calls(probs, self.thresholds, self.findings, self.names, importance,
                           self.config["report"]["likely_threshold_margin"])
        report = generate_report(calls, n_slices=vol.shape[0], plane=self.config["data"]["plane"],
                                 exam_label=exam_label, demo_mode=self.demo_mode)
        lo, hi = np.percentile(vol, [0.5, 99.5])
        return Analysis(vol, probs, cams, importance, calls, report, (float(lo), float(max(hi, lo + 1))), exam_label)


def list_samples(samples_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """List bundled sample scans: real MRNet exports first (if present), then synthetic ones.

    Each sample folder holds ``.npy`` files and a ``samples.json`` manifest.

    Args:
        samples_dir: Root samples folder (defaults to ``samples/``).

    Returns:
        Manifest entries with an absolute ``path`` added.
    """
    root = Path(samples_dir) if samples_dir else PROJECT_ROOT / "samples"
    out: list[dict[str, Any]] = []
    for sub in ("mrnet", "synthetic"):
        folder = root / sub
        manifest = folder / "samples.json"
        entries: list[dict[str, Any]] = []
        if manifest.exists():
            with open(manifest, encoding="utf-8") as fh:
                entries = json.load(fh)
        else:
            entries = [{"file": p.name, "title": p.stem} for p in sorted(folder.glob("*.npy"))]
        for e in entries:
            path = folder / e["file"]
            if path.exists():
                out.append({**e, "path": str(path), "kind": "real" if sub == "mrnet" else "synthetic"})
    return out


def load_metrics(path: str | Path | None = None) -> dict[str, Any] | None:
    """Read ``results/metrics.json`` if it exists.

    Args:
        path: Override the metrics file location.

    Returns:
        The parsed metrics or ``None``.
    """
    p = Path(path) if path else PROJECT_ROOT / load_config()["paths"]["metrics_file"]
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
