"""End-to-end tests: training pipeline, checkpoint loading, Kaggle output installation."""

from __future__ import annotations

import json
import shutil
import zipfile

import numpy as np

from src.inference import Predictor, list_samples, load_metrics
from src.install_outputs import README_END, README_START, install


def test_pipeline_writes_all_outputs(trained_run):
    for rel in ["models/cartilai_resnet18.pt", "results/metrics.json", "results/roc_curves.png",
                "results/confusion_matrices.png", "results/training_curves.png", "results/predictions.csv",
                "samples/mrnet/samples.json"]:
        assert (trained_run / rel).exists(), rel
    m = json.loads((trained_run / "results/metrics.json").read_text())
    assert set(m["findings"]) == {"abnormal", "acl", "meniscus"}
    assert m["training"]["epochs_completed"] >= 1
    assert len(list((trained_run / "samples/mrnet").glob("*.npy"))) == 6


def test_predictor_switches_to_trained_mode(trained_run, config):
    pred = Predictor(config, weights_path=trained_run / "models/cartilai_resnet18.pt")
    assert not pred.demo_mode and pred.mode_label == "Trained model"
    assert len(pred.thresholds) == 3 and pred.thresholds != [0.5, 0.5, 0.5]
    vol = np.load(next((trained_run / "samples/mrnet").glob("*.npy")))
    a = pred.analyze(vol, "test")
    assert a.cams.shape[:2] == (3, vol.shape[0]) and "DEMO MODE" not in a.report


def test_predictor_demo_mode_and_bad_weights(tmp_path, config):
    assert Predictor(config, weights_path=tmp_path / "missing.pt").demo_mode
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    pred = Predictor(config, weights_path=bad)
    assert pred.demo_mode and pred.load_error


def test_install_outputs_from_kaggle_zip(trained_run, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text(f"# x\n{README_START}\npending\n{README_END}\nend\n")
    # Kaggle's "download all" zip contains the folder *and* the inner zip
    inner = tmp_path / "cartilai_outputs.zip"
    shutil.make_archive(str(inner.with_suffix("")), "zip", trained_run, ".")
    outer = tmp_path / "kaggle_download.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(inner, "cartilai_outputs.zip")
    installed = install(outer, project)
    assert "models/cartilai_resnet18.pt" in installed and "results/metrics.json" in installed
    assert (project / "samples/mrnet/samples.json").exists()
    readme = (project / "README.md").read_text()
    assert "| ACL tear |" in readme and "pending" not in readme and readme.endswith("end\n")
    assert load_metrics(project / "results/metrics.json")["project"] == "CartilAI"
    kinds = [s["kind"] for s in list_samples(project / "samples")]
    assert kinds and kinds[0] == "real"


def test_bundled_synthetic_samples_exist():
    samples = list_samples()
    synthetic = [s for s in samples if s["kind"] == "synthetic"]
    assert len(synthetic) == 4
    for s in synthetic:
        vol = np.load(s["path"])
        assert vol.dtype == np.uint8 and vol.shape[1:] == (256, 256)
