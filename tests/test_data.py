"""Tests for data loading and preprocessing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data import (
    MRNetDataset, augment, find_mrnet_root, preprocess, read_labels, stratified_split, to_uint8_volume,
)

FINDINGS = ["abnormal", "acl", "meniscus"]


def test_find_root_in_nested_kaggle_layout(synthetic_root):
    root = find_mrnet_root(synthetic_root)
    assert root.name == "MRNet-v1.0"


def test_find_root_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_mrnet_root(tmp_path)


def test_labels_follow_mrnet_rules(synthetic_root):
    ids, labels = read_labels(find_mrnet_root(synthetic_root), "train", FINDINGS)
    assert len(ids) == 28 and labels.shape == (28, 3)
    assert all(len(i) == 4 for i in ids)
    # an ACL or meniscal tear always makes the exam abnormal
    assert np.all(labels[:, 0] >= labels[:, 1:].max(axis=1))


def test_read_labels_skips_header(tmp_path):
    for f in FINDINGS:
        (tmp_path / f"valid-{f}.csv").write_text("case,label\n0001,1\n2,0\n")
    ids, labels = read_labels(tmp_path, "valid", FINDINGS)
    assert ids == ["0001", "0002"]
    assert labels.tolist() == [[1, 1, 1], [0, 0, 0]]


def test_dataset_item(synthetic_root):
    root = find_mrnet_root(synthetic_root)
    ids, labels = read_labels(root, "valid", FINDINGS)
    ds = MRNetDataset(root, "valid", "sagittal", ids, labels, image_size=64)
    vol, lab, case = ds[0]
    assert vol.dtype == torch.uint8 and vol.ndim == 3 and vol.shape[1:] == (64, 64)
    assert lab.shape == (3,) and case == ids[0]


@pytest.mark.parametrize("shape,dtype", [((20, 256, 256), np.uint8), ((20, 128, 128), np.float32),
                                          ((256, 256, 20), np.int16), ((1, 30, 256, 256), np.uint8), ((256, 256), np.uint8)])
def test_to_uint8_volume_accepts_variants(shape, dtype):
    arr = (np.random.default_rng(0).random(shape) * 1000).astype(dtype)
    out = to_uint8_volume(arr, 256)
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[1:] == (256, 256)


def test_to_uint8_volume_rejects_bad_input():
    with pytest.raises(ValueError):
        to_uint8_volume(np.zeros((2, 3, 4, 5, 6)))
    with pytest.raises(ValueError):
        to_uint8_volume(np.zeros((500, 8, 8)))


def test_preprocess_shape_and_normalisation():
    vol = torch.full((5, 32, 32), 255, dtype=torch.uint8)
    x = preprocess(vol)
    assert x.shape == (5, 3, 32, 32)
    assert torch.allclose(x[0, :, 0, 0], torch.tensor([(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]), atol=1e-4)


def test_augment_is_seeded_and_shape_preserving():
    x = preprocess(torch.randint(0, 255, (4, 32, 32), dtype=torch.uint8))
    a = augment(x, torch.Generator().manual_seed(0))
    b = augment(x, torch.Generator().manual_seed(0))
    assert a.shape == x.shape and torch.equal(a, b)


def test_stratified_split_is_disjoint_and_complete():
    labels = np.array([[1, 1, 0]] * 10 + [[0, 0, 0]] * 10 + [[1, 0, 1]] * 10, dtype=np.float32)
    tr, va = stratified_split(labels, 0.2, seed=0)
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == 30 and len(va) == 6
