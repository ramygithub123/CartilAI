"""Synthetic knee-MRI-like volumes in MRNet format, for tests and the demo.

These are cartoons, not real anatomy: soft-edged shapes that resemble a
sagittal knee MRI (femur, tibia, patella, cartilage, menisci, ACL) plus
simple, visible "lesions" (a disrupted ACL, a bright line through a meniscus,
joint effusion, a bone bruise). They let the whole pipeline (training,
evaluation, Grad-CAM, the app) run end to end without the real dataset.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FINDINGS = ("abnormal", "acl", "meniscus")


@dataclass
class SyntheticCase:
    """Ground truth for one synthetic exam."""

    acl_tear: bool = False
    meniscal_tear: bool = False
    effusion: bool = False
    bone_bruise: bool = False

    @property
    def labels(self) -> dict[str, int]:
        """MRNet-style labels: any finding makes the exam abnormal."""
        acl, men = int(self.acl_tear), int(self.meniscal_tear)
        abnormal = int(acl or men or self.effusion or self.bone_bruise)
        return {"abnormal": abnormal, "acl": acl, "meniscus": men}


def _inside_ellipse(u: np.ndarray, v: np.ndarray, cx: float, cy: float, rx: float, ry: float, edge: float) -> np.ndarray:
    """Soft (0..1) mask of an ellipse; ``edge`` controls the blur of the border."""
    d = np.sqrt(((u - cx) / rx) ** 2 + ((v - cy) / ry) ** 2) - 1.0
    return 1.0 / (1.0 + np.exp(np.clip(d / edge, -50, 50)))


def _segment_distance(u: np.ndarray, v: np.ndarray, p0: tuple[float, float], p1: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each pixel to segment p0-p1, and the position (0..1) along it."""
    ax, ay = p0
    bx, by = p1
    dx, dy = bx - ax, by - ay
    t = np.clip(((u - ax) * dx + (v - ay) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return np.hypot(u - (ax + t * dx), v - (ay + t * dy)), t


def make_volume(
    case: SyntheticCase,
    n_slices: int = 24,
    image_size: int = 256,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render one synthetic sagittal knee volume.

    Args:
        case: Which findings to draw.
        n_slices: Number of sagittal slices (medial to lateral).
        image_size: In-plane size in pixels.
        rng: Random generator (controls anatomy jitter and noise).

    Returns:
        uint8 array of shape ``(n_slices, image_size, image_size)``.
    """
    rng = rng or np.random.default_rng(0)
    lin = np.linspace(-1, 1, image_size, dtype=np.float32)
    v, u = np.meshgrid(lin, lin, indexing="ij")  # v: down, u: anterior (right)
    e = 2.0 / image_size  # one-pixel soft edge

    ox, oy = rng.uniform(-0.06, 0.06, 2)
    joint = 0.08 + oy                       # joint line height
    contrast = rng.uniform(0.85, 1.15)
    tear_slices = rng.uniform(-0.2, 0.2)    # where in z the ACL tear is centred
    men_side = rng.choice([-1.0, 1.0])      # meniscal tear in anterior or posterior horn
    men_z = rng.choice([-1.0, 1.0])         # medial or lateral compartment
    bruise = (rng.uniform(-0.25, 0.25) + ox, rng.choice([joint - 0.25, joint + 0.25]), rng.uniform(-0.6, 0.6))

    vol = np.zeros((n_slices, image_size, image_size), dtype=np.float32)
    for s, z in enumerate(np.linspace(-1, 1, n_slices)):
        width = np.sqrt(max(0.05, 1 - (0.85 * z) ** 2))  # bones narrow towards the edges
        img = np.full_like(u, 12.0)
        leg = _inside_ellipse(u, v, ox, oy, 0.82 * width + 0.05, 1.4, 4 * e)
        img += leg * 55
        # subcutaneous fat (bright rim)
        img += (leg - _inside_ellipse(u, v, ox, oy, 0.74 * width + 0.04, 1.35, 4 * e)) * 85
        # femur with rounded condyle, tibia with flat plateau
        femur = _inside_ellipse(u, v, ox, joint - 0.55, 0.42 * width, 0.55, e)
        tibia = np.clip((v - (joint + 0.07)) / e, 0, 1) * _inside_ellipse(u, v, ox + 0.03, joint + 0.75, 0.40 * width, 0.70, e)
        marrow = 95 + 10 * np.sin(17 * u + 3 * z) * np.cos(13 * v)
        img = img * (1 - femur) + femur * marrow
        img = img * (1 - tibia) + tibia * marrow
        # cartilage: bright rim on the femoral condyle and the tibial plateau
        fem_cart = (_inside_ellipse(u, v, ox, joint - 0.55, 0.42 * width + 0.025, 0.575, e) - femur) * (v > joint - 0.45)
        tib_cart = np.clip(((joint + 0.075) - v) / e, 0, 1) * np.clip((v - (joint + 0.045)) / e, 0, 1) * (np.abs(u - ox - 0.03) < 0.38 * width)
        img += np.clip(fem_cart, 0, 1) * 75 + tib_cart * 70
        # patella
        patella = _inside_ellipse(u, v, ox + 0.5 * width + 0.08, joint - 0.42, 0.08, 0.17, e)
        img = img * (1 - patella) + patella * 100
        # menisci: dark wedges in the joint space, away from the central (notch) slices
        if abs(z) > 0.25:
            for side in (-1.0, 1.0):
                hx = ox + side * 0.27 * width
                horn = _inside_ellipse(u, v, hx, joint + 0.025, 0.10, 0.035, e)
                img = img * (1 - horn) + horn * 22
                if case.meniscal_tear and side == men_side and np.sign(z) == men_z:
                    d, _ = _segment_distance(u, v, (hx - 0.07, joint + 0.055), (hx + 0.07, joint - 0.005))
                    line = np.clip(1 - d / (2.5 * e), 0, 1) * horn
                    img = img + line * 190
        # ACL: dark band through the intercondylar notch on central slices
        if abs(z) < 0.3:
            p0, p1 = (ox + 0.14, joint + 0.06), (ox - 0.16, joint - 0.32)
            d, t = _segment_distance(u, v, p0, p1)
            band = np.clip(1 - d / 0.03, 0, 1)
            if case.acl_tear and abs(z - tear_slices) < 0.3:
                gap = np.exp(-((t - 0.5) / 0.18) ** 2)  # disrupted mid-substance, fluid signal
                img = img * (1 - band) + band * (35 * (1 - gap) + 200 * gap)
                wavy = np.clip(1 - np.abs(d - 0.05 - 0.02 * np.sin(40 * t)) / 0.012, 0, 1) * (t > 0.55)
                img = img + wavy * 90
            else:
                img = img * (1 - band) + band * 30
        if case.effusion:
            pouch = _inside_ellipse(u, v, ox + 0.38 * width + 0.05, joint - 0.78, 0.13, 0.17, 3 * e)
            img = img * (1 - pouch) + pouch * 210
        if case.bone_bruise and abs(z - bruise[2]) < 0.35:
            blob = _inside_ellipse(u, v, bruise[0], bruise[1], 0.11, 0.08, 6 * e) * np.maximum(femur, tibia)
            img = img + blob * 90
        bias = 1 + 0.12 * (u * 0.5 - v * 0.3)
        img = img * bias * contrast + rng.normal(0, 7, img.shape)
        vol[s] = img
    return np.clip(vol, 0, 255).astype(np.uint8)


def random_case(rng: np.random.Generator) -> SyntheticCase:
    """Draw a synthetic case with MRNet-like class mix."""
    r = rng.uniform()
    if r < 0.22:
        return SyntheticCase()
    if r < 0.42:
        return SyntheticCase(effusion=rng.uniform() < 0.6, bone_bruise=True) if rng.uniform() < 0.5 else SyntheticCase(effusion=True)
    if r < 0.62:
        return SyntheticCase(acl_tear=True, effusion=rng.uniform() < 0.5)
    if r < 0.84:
        return SyntheticCase(meniscal_tear=True)
    return SyntheticCase(acl_tear=True, meniscal_tear=True, effusion=rng.uniform() < 0.5)


def make_synthetic_mrnet(
    root: str | Path,
    n_train: int = 60,
    n_valid: int = 20,
    image_size: int = 256,
    plane: str = "sagittal",
    seed: int = 42,
) -> Path:
    """Write a small synthetic dataset with the exact MRNet folder and CSV layout.

    Args:
        root: Output folder (created).
        n_train: Number of training exams.
        n_valid: Number of validation exams.
        image_size: In-plane size.
        plane: Plane folder name to create.
        seed: Random seed.

    Returns:
        The dataset root.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    case_no = 0
    for split, n in (("train", n_train), ("valid", n_valid)):
        (root / split / plane).mkdir(parents=True, exist_ok=True)
        rows: dict[str, list[tuple[str, int]]] = {f: [] for f in FINDINGS}
        for _ in range(n):
            case = random_case(rng)
            cid = f"{case_no:04d}"
            vol = make_volume(case, int(rng.integers(18, 30)), image_size, rng)
            np.save(root / split / plane / f"{cid}.npy", vol)
            for f, lab in case.labels.items():
                rows[f].append((cid, lab))
            case_no += 1
        for f in FINDINGS:
            with open(root / f"{split}-{f}.csv", "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows(rows[f])
    return root


DEMO_SAMPLES: list[tuple[str, str, SyntheticCase]] = [
    ("synthetic_normal.npy", "Synthetic knee — normal", SyntheticCase()),
    ("synthetic_acl_tear.npy", "Synthetic knee — ACL tear + effusion", SyntheticCase(acl_tear=True, effusion=True)),
    ("synthetic_meniscal_tear.npy", "Synthetic knee — meniscal tear", SyntheticCase(meniscal_tear=True)),
    ("synthetic_bone_bruise.npy", "Synthetic knee — bone bruise (abnormal, no tear)", SyntheticCase(bone_bruise=True, effusion=True)),
]


def write_demo_samples(out_dir: str | Path, n_slices: int = 24, image_size: int = 256, seed: int = 7) -> list[dict]:
    """Create the bundled synthetic demo scans plus a ``samples.json`` manifest.

    Args:
        out_dir: Destination folder.
        n_slices: Slices per volume.
        image_size: In-plane size.
        seed: Random seed.

    Returns:
        The manifest entries.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (fname, title, case) in enumerate(DEMO_SAMPLES):
        vol = make_volume(case, n_slices, image_size, np.random.default_rng(seed + i))
        np.save(out_dir / fname, vol)
        truth = [n for n, k in (("ACL tear", "acl"), ("Meniscal tear", "meniscus")) if case.labels[k]]
        extras = [n for n, flag in (("effusion", case.effusion), ("bone bruise", case.bone_bruise)) if flag]
        manifest.append({
            "file": fname,
            "title": title,
            "source": "Synthetic cartoon volume generated by src/synthetic.py (not a real patient)",
            "ground_truth": case.labels,
            "description": "Drawn findings: " + (", ".join(truth + extras) if truth or extras else "none"),
        })
    with open(out_dir / "samples.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


if __name__ == "__main__":
    write_demo_samples(Path(__file__).resolve().parent.parent / "samples" / "synthetic")
    print("Wrote synthetic demo samples to samples/synthetic/")
