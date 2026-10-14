"""Evaluation: AUC with bootstrap CIs, threshold selection, ROC curves and confusion matrices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.data import preprocess


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the model over a loader and collect probabilities.

    Args:
        model: Trained network.
        loader: Yields ``(volume, labels, case_id)`` with batch size 1.
        device: Device to run on.
        amp: Use autocast (CUDA only).

    Returns:
        ``(y_true, y_prob, case_ids)``; arrays have shape ``(n, n_findings)``.
    """
    model.eval()
    ys, ps, ids = [], [], []
    for vol, label, case in loader:
        x = preprocess(vol.to(device, non_blocking=True))
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(x)
        ps.append(torch.sigmoid(logits.float()).cpu().numpy())
        ys.append(label.numpy())
        ids.append(case)
    return np.stack(ys), np.stack(ps), ids


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC AUC that returns NaN instead of raising when only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick the probability threshold that maximises sensitivity + specificity - 1.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.

    Returns:
        The threshold, or 0.5 if it cannot be computed.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[best], 1e-6, 1 - 1e-6))


def bootstrap_auc_ci(
    y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap confidence interval for the AUC.

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        n_boot: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        ``(low, high)``; NaNs if the AUC is undefined.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    findings: Sequence[str],
    thresholds: Sequence[float],
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Compute per-finding metrics at the given decision thresholds.

    Args:
        y_true: ``(n, k)`` binary labels.
        y_prob: ``(n, k)`` probabilities.
        findings: Finding keys (length ``k``).
        thresholds: Decision threshold per finding.
        seed: Seed for the bootstrap.

    Returns:
        ``{finding: {auc, auc_ci_low, auc_ci_high, accuracy, sensitivity, specificity,
        threshold, tp, fp, tn, fn, n_positive, n_negative}}``.
    """
    out: dict[str, dict[str, Any]] = {}
    for k, finding in enumerate(findings):
        t, p = y_true[:, k], y_prob[:, k]
        pred = (p >= thresholds[k]).astype(int)
        tp = int(((pred == 1) & (t == 1)).sum())
        tn = int(((pred == 0) & (t == 0)).sum())
        fp = int(((pred == 1) & (t == 0)).sum())
        fn = int(((pred == 0) & (t == 1)).sum())
        lo, hi = bootstrap_auc_ci(t, p, seed=seed)
        out[finding] = {
            "auc": safe_auc(t, p),
            "auc_ci_low": lo,
            "auc_ci_high": hi,
            "accuracy": (tp + tn) / max(1, len(t)),
            "sensitivity": tp / max(1, tp + fn),
            "specificity": tn / max(1, tn + fp),
            "threshold": float(thresholds[k]),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_positive": int(t.sum()),
            "n_negative": int(len(t) - t.sum()),
        }
    return out


def plot_roc_curves(
    y_true: np.ndarray, y_prob: np.ndarray, findings: Sequence[str], names: dict[str, str], path: str | Path
) -> None:
    """Save one ROC curve per finding to a single PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#0f766e", "#2563eb", "#b45309"]
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=130)
    for k, finding in enumerate(findings):
        t, p = y_true[:, k], y_prob[:, k]
        if len(np.unique(t)) < 2:
            continue
        fpr, tpr, _ = roc_curve(t, p)
        ax.plot(fpr, tpr, lw=2, color=colors[k % len(colors)],
                label=f"{names.get(finding, finding)} (AUC {safe_auc(t, p):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#94a3b8", lw=1)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("CartilAI — ROC curves (held-out MRNet validation set)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrices(
    metrics: dict[str, dict[str, Any]], findings: Sequence[str], names: dict[str, str], path: str | Path
) -> None:
    """Save a row of 2x2 confusion matrices (one per finding) to a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(findings), figsize=(4.2 * len(findings), 3.9), dpi=130)
    axes = np.atleast_1d(axes)
    for ax, finding in zip(axes, findings):
        m = metrics[finding]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "#0f172a", fontsize=14)
        ax.set_xticks([0, 1], ["Pred. negative", "Pred. positive"])
        ax.set_yticks([0, 1], ["Negative", "Positive"])
        ax.set_title(f"{names.get(finding, finding)}\n(threshold {m['threshold']:.2f})")
    fig.suptitle("CartilAI — confusion matrices (held-out MRNet validation set)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_training_curves(history: list[dict[str, float]], path: str | Path) -> None:
    """Save training loss and internal-validation AUC per epoch to a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=130)
    a1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    a1.plot(epochs, [h["val_loss"] for h in history], marker="o", label="internal val")
    a1.set_xlabel("epoch"); a1.set_ylabel("BCE loss"); a1.legend(frameon=False); a1.grid(alpha=0.25)
    a2.plot(epochs, [h["val_mean_auc"] for h in history], marker="o", color="#0f766e")
    a2.set_xlabel("epoch"); a2.set_ylabel("mean AUC"); a2.set_title("Internal validation"); a2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` as pretty JSON, converting NaN to null."""

    def clean(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clean(obj), fh, indent=2)
