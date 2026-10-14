"""Training pipeline: train on MRNet, select the best epoch, evaluate, export everything.

The same code runs locally (synthetic data, CPU) and in the Kaggle notebook
(real MRNet, GPU); ``scripts/build_notebook.py`` inlines it into the notebook.

Local usage::

    python -m src.train --synthetic --quick          # smoke test on generated data
    python -m src.train --data-root /path/to/MRNet-v1.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config, merge_config, set_seed
from src.data import (
    MRNetDataset, augment, collate_single, find_mrnet_root, preprocess, read_labels, stratified_split,
)
from src.evaluate import (
    compute_metrics, plot_confusion_matrices, plot_roc_curves, plot_training_curves,
    predict_dataset, safe_auc, save_json, youden_threshold,
)
from src.model import build_model

WEIGHTS_NAME = "cartilai_resnet18.pt"
N_EXPORTED_SAMPLES = 6

# Tiny settings for fast smoke tests (CPU, synthetic data).
QUICK_OVERRIDES: dict[str, Any] = {
    "data": {"image_size": 96, "num_workers": 0, "internal_val_fraction": 0.25},
    "model": {"pretrained": False},
    "train": {"epochs": 2, "lr": 1.0e-4, "max_train_minutes": 10},
}


def _loader(ds: MRNetDataset, shuffle: bool, workers: int, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=1, shuffle=shuffle, num_workers=workers, collate_fn=collate_single,
        pin_memory=torch.cuda.is_available(), generator=g, persistent_workers=workers > 0,
    )


def train_model(
    config: dict[str, Any],
    root: Path,
    device: torch.device,
    log: Any = print,
) -> tuple[nn.Module, list[float], list[dict[str, float]], dict[str, Any]]:
    """Train CartilAINet on the MRNet ``train`` split with an internal validation split.

    The best epoch (highest mean internal-validation AUC) is kept. Decision
    thresholds are chosen on the internal split (Youden's J) so that the
    official ``valid`` split stays an untouched test set.

    Args:
        config: Full CartilAI config.
        root: MRNet root folder.
        device: Training device.
        log: Logging function.

    Returns:
        ``(model, thresholds, history, info)``.
    """
    seed = config["seed"]
    findings = config["findings"]
    dcfg, tcfg = config["data"], config["train"]
    set_seed(seed)

    case_ids, labels = read_labels(root, "train", findings)
    tr_idx, va_idx = stratified_split(labels, dcfg["internal_val_fraction"], seed)
    train_ds = MRNetDataset(root, "train", dcfg["plane"], [case_ids[i] for i in tr_idx], labels[tr_idx], dcfg["image_size"])
    val_ds = MRNetDataset(root, "train", dcfg["plane"], [case_ids[i] for i in va_idx], labels[va_idx], dcfg["image_size"])
    log(f"Train exams: {len(train_ds)} | internal validation exams: {len(val_ds)}")
    train_dl = _loader(train_ds, True, dcfg["num_workers"], seed)
    val_dl = _loader(val_ds, False, dcfg["num_workers"], seed)

    model = build_model(config).to(device)
    # Class-imbalance weighting: pos_weight = negatives / positives per finding.
    pos = labels[tr_idx].sum(0)
    neg = len(tr_idx) - pos
    pos_weight = torch.tensor(np.where(pos > 0, neg / np.maximum(pos, 1), 1.0), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = max(1, tcfg["epochs"] * len(train_dl))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=tcfg["lr"], total_steps=total_steps, pct_start=0.1,
    )
    use_amp = bool(tcfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    aug_gen = torch.Generator().manual_seed(seed)

    best_auc, best_state, best_epoch = -1.0, None, 0
    history: list[dict[str, float]] = []
    start = time.time()
    budget_s = tcfg["max_train_minutes"] * 60
    stopped_early = False

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for vol, label, _ in train_dl:
            x = preprocess(vol.to(device, non_blocking=True))
            if tcfg["augment"]:
                x = augment(x, aug_gen, tcfg["rotation_degrees"], tcfg["translate_pixels"], tcfg["hflip"])
            y = label.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            n += 1

        y_val, p_val, _ = predict_dataset(model, val_dl, device, use_amp)
        with torch.no_grad():
            val_loss = float(criterion(torch.logit(torch.tensor(p_val).clamp(1e-6, 1 - 1e-6)).to(device),
                                       torch.tensor(y_val).to(device)))
        aucs = [safe_auc(y_val[:, k], p_val[:, k]) for k in range(len(findings))]
        mean_auc = float(np.nanmean(aucs)) if not np.all(np.isnan(aucs)) else float(-val_loss)
        history.append({"epoch": epoch, "train_loss": running / max(1, n), "val_loss": val_loss,
                        "val_mean_auc": mean_auc, **{f"val_auc_{f}": a for f, a in zip(findings, aucs)}})
        log(f"Epoch {epoch:02d}/{tcfg['epochs']} | train loss {running / max(1, n):.4f} | "
            f"val loss {val_loss:.4f} | val AUC " + " ".join(f"{f}={a:.3f}" for f, a in zip(findings, aucs))
            + f" | {time.time() - t0:.0f}s")
        if mean_auc > best_auc:
            best_auc, best_epoch = mean_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epoch_time = time.time() - t0
        if time.time() - start + epoch_time > budget_s and epoch < tcfg["epochs"]:
            log(f"Stopping after epoch {epoch}: next epoch would exceed the {tcfg['max_train_minutes']} min budget.")
            stopped_early = True
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    y_val, p_val, _ = predict_dataset(model, val_dl, device, use_amp)
    thresholds = [youden_threshold(y_val[:, k], p_val[:, k]) for k in range(len(findings))]
    info = {
        "best_epoch": best_epoch,
        "best_internal_val_mean_auc": best_auc,
        "epochs_completed": len(history),
        "stopped_early_for_time": stopped_early,
        "train_minutes": (time.time() - start) / 60,
        "n_train": len(train_ds),
        "n_internal_val": len(val_ds),
    }
    log(f"Best epoch {best_epoch} (internal val mean AUC {best_auc:.3f}); thresholds "
        + ", ".join(f"{f}={t:.2f}" for f, t in zip(findings, thresholds)))
    return model, thresholds, history, info


def save_checkpoint(
    path: Path, model: nn.Module, config: dict[str, Any], thresholds: list[float], extra: dict[str, Any]
) -> None:
    """Save weights plus everything the app needs to use them (findings, thresholds, config)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "cartilai-v1",
        "state_dict": {k: v.detach().cpu().float() if v.is_floating_point() else v.detach().cpu()
                       for k, v in model.state_dict().items()},
        "findings": list(config["findings"]),
        "thresholds": [float(t) for t in thresholds],
        "config": {"model": config["model"], "data": config["data"]},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }, path)


def export_samples(
    root: Path, config: dict[str, Any], case_ids: list[str], labels: np.ndarray,
    probs: np.ndarray, out_dir: Path, n: int = N_EXPORTED_SAMPLES,
) -> list[dict[str, Any]]:
    """Copy a few held-out exams (mixed findings) and a manifest for the app's demo gallery.

    Args:
        root: MRNet root.
        config: Full config.
        case_ids: Validation case ids.
        labels: Their labels ``(n, k)``.
        probs: Model probabilities ``(n, k)``.
        out_dir: Destination folder.
        n: Number of exams to export.

    Returns:
        The manifest entries written to ``samples.json``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    findings, names = config["findings"], config["finding_names"]
    rng = np.random.default_rng(config["seed"])
    # Aim for variety: normal, abnormal-only, ACL tear, meniscal tear, both.
    patterns = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]
    chosen: list[int] = []
    for pat in patterns:
        idx = [i for i in range(len(case_ids)) if tuple(int(v) for v in labels[i]) == pat and i not in chosen]
        if idx:
            chosen.append(int(rng.choice(idx)))
    for i in rng.permutation(len(case_ids)):
        if len(chosen) >= n:
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    manifest = []
    plane = config["data"]["plane"]
    for i in chosen[:n]:
        case = case_ids[i]
        fname = f"mrnet_{plane}_{case}.npy"
        shutil.copy(root / "valid" / plane / f"{case}.npy", out_dir / fname)
        truth = [names[f] for f, v in zip(findings, labels[i]) if v == 1]
        manifest.append({
            "file": fname,
            "title": f"MRNet exam {case} ({plane})",
            "source": "MRNet v1.0 validation set (held out from training)",
            "ground_truth": {f: int(v) for f, v in zip(findings, labels[i])},
            "description": "Reference labels: " + (", ".join(truth) if truth else "normal"),
            "model_probabilities": {f: round(float(p), 4) for f, p in zip(findings, probs[i])},
        })
    with open(out_dir / "samples.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def run_pipeline(
    config: dict[str, Any],
    input_dir: str | Path,
    output_dir: str | Path,
    log: Any = print,
) -> dict[str, Any]:
    """End-to-end: find data, train, evaluate on the held-out split, save all outputs.

    Output layout (mirrors the project folders so it can be copied in directly)::

        <output_dir>/models/cartilai_resnet18.pt
        <output_dir>/results/{metrics.json, roc_curves.png, confusion_matrices.png,
                              training_curves.png, predictions.csv}
        <output_dir>/samples/mrnet/{*.npy, samples.json}

    Args:
        config: Full CartilAI config.
        input_dir: Folder containing an MRNet dataset (searched recursively).
        output_dir: Where to write the outputs.
        log: Logging function.

    Returns:
        The metrics dictionary saved to ``metrics.json``.
    """
    t_start = time.time()
    set_seed(config["seed"])
    output_dir = Path(output_dir)
    models_dir, results_dir, samples_dir = output_dir / "models", output_dir / "results", output_dir / "samples" / "mrnet"
    for d in (models_dir, results_dir, samples_dir):
        d.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    root = find_mrnet_root(input_dir, config["data"]["plane"])
    log(f"MRNet root: {root}")
    findings, names = config["findings"], config["finding_names"]

    model, thresholds, history, info = train_model(config, root, device, log)

    # Held-out evaluation on the official MRNet validation split (never used for training or selection).
    test_ids, test_labels = read_labels(root, "valid", findings)
    test_ds = MRNetDataset(root, "valid", config["data"]["plane"], test_ids, test_labels, config["data"]["image_size"])
    test_dl = _loader(test_ds, False, config["data"]["num_workers"], config["seed"])
    y_true, y_prob, ids = predict_dataset(model, test_dl, device, bool(config["train"]["amp"]))
    per_finding = compute_metrics(y_true, y_prob, findings, thresholds, seed=config["seed"])

    metrics = {
        "project": "CartilAI",
        "evaluation_set": f"MRNet v1.0 validation split ({len(ids)} exams, {config['data']['plane']} plane), held out from training",
        "model": f"{config['model']['backbone']} per-slice CNN + max-pool across slices",
        "plane": config["data"]["plane"],
        "findings": {f: {"name": names[f], **per_finding[f]} for f in findings},
        "mean_auc": float(np.nanmean([per_finding[f]["auc"] for f in findings])),
        "thresholds_selected_on": "internal validation split (10% of MRNet train), Youden's J",
        "training": {**info, "epochs_planned": config["train"]["epochs"], "lr": config["train"]["lr"],
                     "seed": config["seed"], "device": str(device),
                     "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
        "history": history,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_json(metrics, results_dir / "metrics.json")
    plot_roc_curves(y_true, y_prob, findings, names, results_dir / "roc_curves.png")
    plot_confusion_matrices(per_finding, findings, names, results_dir / "confusion_matrices.png")
    plot_training_curves(history, results_dir / "training_curves.png")
    with open(results_dir / "predictions.csv", "w", encoding="utf-8") as fh:
        fh.write("case_id," + ",".join(f"label_{f}" for f in findings) + "," + ",".join(f"prob_{f}" for f in findings) + "\n")
        for c, t, p in zip(ids, y_true, y_prob):
            fh.write(c + "," + ",".join(str(int(v)) for v in t) + "," + ",".join(f"{v:.5f}" for v in p) + "\n")

    save_checkpoint(models_dir / WEIGHTS_NAME, model, config, thresholds,
                    {"test_metrics": {f: {"auc": per_finding[f]["auc"]} for f in findings}, "training": info})
    export_samples(root, config, ids, y_true, y_prob, samples_dir)

    log("\n=== CartilAI held-out results ===")
    for f in findings:
        m = per_finding[f]
        log(f"{names[f]:<14} AUC {m['auc']:.3f} (95% CI {m['auc_ci_low']:.3f}-{m['auc_ci_high']:.3f}) | "
            f"sens {m['sensitivity']:.2f} | spec {m['specificity']:.2f} | acc {m['accuracy']:.2f}")
    log(f"Total time: {(time.time() - t_start) / 60:.1f} min. Outputs in {output_dir}")
    return metrics


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train and evaluate CartilAI")
    parser.add_argument("--data-root", type=str, help="Folder containing an MRNet dataset")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Where to write models/results/samples")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic MRNet-like dataset and train on it")
    parser.add_argument("--quick", action="store_true", help="Tiny settings for a fast smoke test")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.quick:
        config = merge_config(config, QUICK_OVERRIDES)
    data_root = args.data_root
    if args.synthetic:
        from src.synthetic import make_synthetic_mrnet

        data_root = str(Path(args.output_dir) / "synthetic_mrnet")
        make_synthetic_mrnet(data_root, n_train=40 if args.quick else 200, n_valid=16 if args.quick else 60,
                             image_size=config["data"]["image_size"], seed=config["seed"])
    if not data_root:
        parser.error("Pass --data-root or --synthetic")
    run_pipeline(config, data_root, args.output_dir)


if __name__ == "__main__":
    main()
