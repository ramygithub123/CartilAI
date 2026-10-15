"""MRNet data loading, preprocessing and augmentation.

MRNet layout (Stanford ML Group, v1.0)::

    MRNet-v1.0/
        train/{axial,coronal,sagittal}/0000.npy ... 1129.npy
        valid/{axial,coronal,sagittal}/1130.npy ... 1249.npy
        train-abnormal.csv  train-acl.csv  train-meniscus.csv
        valid-abnormal.csv  valid-acl.csv  valid-meniscus.csv

Each ``.npy`` is a uint8 volume of shape ``(slices, 256, 256)``; each CSV has
rows ``case_id,label`` without a header.
"""

from __future__ import annotations

import csv
import io
import math
import os
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAX_SLICES = 128


def find_mrnet_root(search_dir: str | Path, plane: str = "sagittal") -> Path:
    """Locate the MRNet root folder anywhere below ``search_dir``.

    Kaggle mirrors nest the data differently (``/kaggle/input/<name>/MRNet-v1.0``,
    ``/kaggle/input/<name>/data``...). We look for the folder that contains
    ``train-acl.csv`` and a ``train/<plane>`` sub-folder.

    Args:
        search_dir: Directory to search recursively.
        plane: MRNet plane that must be present.

    Returns:
        Path to the MRNet root.

    Raises:
        FileNotFoundError: If no MRNet-shaped folder exists below ``search_dir``.
    """
    search_dir = Path(search_dir)
    for dirpath, dirnames, filenames in os.walk(search_dir):
        lower = {f.lower() for f in filenames}
        if "train-acl.csv" in lower and (Path(dirpath) / "train" / plane).is_dir():
            return Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in {"axial", "coronal", "sagittal"}]
    raise FileNotFoundError(
        f"Could not find an MRNet dataset (train-acl.csv + train/{plane}/) under {search_dir}. "
        "Attach an MRNet dataset to the notebook (see KAGGLE_STEPS.md)."
    )


def _find_csv(root: Path, name: str) -> Path:
    """Return the CSV ``name`` inside ``root`` regardless of filename case."""
    for path in root.iterdir():
        if path.name.lower() == name.lower():
            return path
    raise FileNotFoundError(f"Missing label file {name} in {root}")


def read_labels(root: str | Path, split: str, findings: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Read MRNet label CSVs for a split.

    Args:
        root: MRNet root folder.
        split: ``"train"`` or ``"valid"``.
        findings: Finding keys, e.g. ``["abnormal", "acl", "meniscus"]``.

    Returns:
        ``(case_ids, labels)`` where ``labels`` has shape ``(n_cases, n_findings)``.
    """
    root = Path(root)
    per_finding: list[dict[str, int]] = []
    for finding in findings:
        mapping: dict[str, int] = {}
        with open(_find_csv(root, f"{split}-{finding}.csv"), newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) < 2 or not row[0].strip().isdigit():
                    continue  # skip an optional header line
                mapping[row[0].strip().zfill(4)] = int(float(row[1]))
        per_finding.append(mapping)
    case_ids = sorted(set.intersection(*(set(m) for m in per_finding)))
    labels = np.array([[m[c] for m in per_finding] for c in case_ids], dtype=np.float32)
    return case_ids, labels


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickler that can only rebuild NumPy arrays, so an uploaded pickle cannot run code."""

    ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
        ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("_codecs", "encode"),  # protocol-2 pickles store raw bytes via codecs.encode
    }

    def find_class(self, module: str, name: str):  # type: ignore[override]
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}: only NumPy arrays are allowed")


def load_volume_file(path: str | Path) -> np.ndarray:
    """Read a scan from disk: MRNet ``.npy`` or a pickled NumPy array (``.pck``/``.pkl``).

    ``.pck`` is the format of the KneeMRI dataset (Clinical Hospital Centre Rijeka),
    a useful outside test set. Pickles are loaded with an unpickler that only
    accepts NumPy arrays.

    Args:
        path: File to read.

    Returns:
        The array stored in the file.

    Raises:
        ValueError: If the file is not a supported scan.
    """
    path = Path(path)
    if path.suffix.lower() in {".pck", ".pkl", ".pickle"}:
        data = path.read_bytes()
        try:
            try:
                arr = _NumpyOnlyUnpickler(io.BytesIO(data)).load()
            except UnicodeDecodeError:  # Python 2 pickles (KneeMRI) need latin1
                arr = _NumpyOnlyUnpickler(io.BytesIO(data), encoding="latin1").load()
        except Exception as exc:
            raise ValueError(f"Could not read {path.name} as a pickled NumPy volume ({exc})") from exc
        if not isinstance(arr, np.ndarray):
            raise ValueError(f"{path.name} does not contain an image volume")
        return arr
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Could not read {path.name} as a NumPy .npy volume ({exc})") from exc


def to_uint8_volume(volume: np.ndarray, image_size: int = 256) -> np.ndarray:
    """Validate and normalise an arbitrary 3D array into a ``(S, H, W)`` uint8 volume.

    Accepts MRNet-style ``(S, H, W)`` arrays, squeezes singleton dimensions,
    rescales non-uint8 intensities to 0..255 and resizes slices to
    ``image_size`` if needed.

    Args:
        volume: Input array.
        image_size: Target in-plane size.

    Returns:
        A uint8 array of shape ``(S, image_size, image_size)``.

    Raises:
        ValueError: If the array cannot be interpreted as a 3D volume.
    """
    arr = np.asarray(volume)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D volume (slices, height, width); got shape {np.shape(volume)}")
    # MRNet stores slices first. If the smallest axis is last, move it to the front.
    if arr.shape[2] < min(arr.shape[0], arr.shape[1]) and arr.shape[0] == arr.shape[1]:
        arr = np.moveaxis(arr, 2, 0)
    if arr.shape[0] > MAX_SLICES:
        raise ValueError(f"Volume has {arr.shape[0]} slices; at most {MAX_SLICES} are supported")
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr.astype(np.float32))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = ((arr - lo) / (hi - lo) * 255.0) if hi > lo else np.zeros_like(arr)
        arr = arr.round().astype(np.uint8)
    if arr.shape[1] != image_size or arr.shape[2] != image_size:
        t = torch.from_numpy(arr).float()[:, None]
        t = F.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
        arr = t[:, 0].clamp(0, 255).round().to(torch.uint8).numpy()
    return np.ascontiguousarray(arr)


def preprocess(volume: torch.Tensor) -> torch.Tensor:
    """Turn a uint8 volume ``(S, H, W)`` into a normalised network input ``(S, 3, H, W)``.

    Slices are scaled to [0, 1], replicated to 3 channels and normalised with
    ImageNet statistics so they match what the pretrained backbone expects.

    Args:
        volume: uint8 (or float 0..255) tensor of shape ``(S, H, W)``.

    Returns:
        Float tensor of shape ``(S, 3, H, W)`` on the same device.
    """
    x = volume.float().div(255.0).unsqueeze(1).expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(
    x: torch.Tensor,
    generator: torch.Generator | None = None,
    rotation_degrees: float = 25.0,
    translate_pixels: float = 25.0,
    hflip: bool = True,
) -> torch.Tensor:
    """Apply one random affine transform (and optional flip) to every slice of a volume.

    Mirrors the MRNet paper augmentation: random rotation, translation and
    horizontal flip, shared by all slices of the exam.

    Args:
        x: Tensor ``(S, C, H, W)``.
        generator: Optional CPU random generator for reproducibility.
        rotation_degrees: Max absolute rotation.
        translate_pixels: Max absolute shift in pixels.
        hflip: Whether to randomly flip horizontally.

    Returns:
        Augmented tensor with the same shape.
    """
    s, _, h, w = x.shape
    r = torch.rand(4, generator=generator)
    angle = math.radians((r[0].item() * 2 - 1) * rotation_degrees)
    tx = (r[1].item() * 2 - 1) * translate_pixels * 2 / w
    ty = (r[2].item() * 2 - 1) * translate_pixels * 2 / h
    flip = -1.0 if (hflip and r[3].item() < 0.5) else 1.0
    cos, sin = math.cos(angle), math.sin(angle)
    theta = torch.tensor(
        [[cos * flip, -sin, tx], [sin * flip, cos, ty]], dtype=x.dtype, device=x.device
    ).unsqueeze(0).expand(s, -1, -1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    # pad with the normalised value of a black pixel
    fill = x.amin(dim=(0, 2, 3), keepdim=True)
    out = F.grid_sample(x - fill, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out + fill


class MRNetDataset(Dataset):
    """One item per exam: ``(uint8 volume (S, H, W), labels (n_findings,), case_id)``."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        plane: str,
        case_ids: Sequence[str],
        labels: np.ndarray,
        image_size: int = 256,
    ) -> None:
        """Create the dataset.

        Args:
            root: MRNet root folder.
            split: Folder name, ``"train"`` or ``"valid"``.
            plane: ``"sagittal"``, ``"coronal"`` or ``"axial"``.
            case_ids: Case identifiers (zero-padded strings).
            labels: Array ``(len(case_ids), n_findings)``.
            image_size: In-plane size fed to the network.
        """
        self.dir = Path(root) / split / plane
        self.case_ids = list(case_ids)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        case = self.case_ids[idx]
        vol = to_uint8_volume(np.load(self.dir / f"{case}.npy"), self.image_size)
        return torch.from_numpy(vol), torch.from_numpy(self.labels[idx]), case


def collate_single(batch: list[tuple[torch.Tensor, torch.Tensor, str]]) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Collate function for batch size 1 (exams have different slice counts)."""
    assert len(batch) == 1, "CartilAI trains one exam at a time (variable slice counts)"
    return batch[0]


def stratified_split(
    labels: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into train / internal-validation, stratified on the label pattern.

    Args:
        labels: ``(n, n_findings)`` binary labels.
        fraction: Fraction of cases to hold out.
        seed: Random seed.

    Returns:
        ``(train_idx, val_idx)`` sorted index arrays.
    """
    rng = np.random.default_rng(seed)
    keys = ["".join(str(int(v)) for v in row) for row in labels]
    train_idx: list[int] = []
    val_idx: list[int] = []
    for key in sorted(set(keys)):
        idx = np.array([i for i, k in enumerate(keys) if k == key])
        rng.shuffle(idx)
        n_val = int(round(len(idx) * fraction))
        if len(idx) >= 2:
            n_val = max(1, n_val)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(val_idx))
