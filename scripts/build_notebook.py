"""Generate notebooks/cartilai_kaggle_training.ipynb from the source modules.

The Kaggle notebook must be a single self-contained file (Kaggle only receives
the .ipynb), so the training code from ``src/`` is inlined into its cells.
Building it from the real modules keeps the notebook and the tested code in
sync. Run ``python scripts/build_notebook.py`` after changing ``src/``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "cartilai_kaggle_training.ipynb"
MODULES = ["config", "data", "model", "evaluate", "train"]


def strip_internal_imports(source: str) -> str:
    """Remove ``from src... import ...`` statements and anything from ``def main`` onwards."""
    tree = ast.parse(source)
    lines = source.splitlines()
    drop: set[int] = set()
    cut = len(lines)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "src":
            drop.update(range(node.lineno - 1, node.end_lineno))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            cut = min(cut, node.lineno - 1)
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            cut = min(cut, node.lineno - 1)
    kept = [line for i, line in enumerate(lines[:cut]) if i not in drop]
    return "\n".join(kept).rstrip() + "\n"


def module_source(name: str) -> str:
    """Return a ``src`` module's code, ready to paste into a notebook cell."""
    src = (ROOT / "src" / f"{name}.py").read_text(encoding="utf-8")
    body = strip_internal_imports(src)
    if name == "config":
        # Paths relative to the repo make no sense on Kaggle; the notebook embeds the config instead.
        body = body.replace('PROJECT_ROOT = Path(__file__).resolve().parent.parent\n', 'PROJECT_ROOT = Path(".")\n')
        body = body.replace('DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"\n', 'DEFAULT_CONFIG_PATH = Path("config.yaml")\n')
    return f"# ===== src/{name}.py (inlined) =====\n{body}"


INTRO = """# CartilAI — Kaggle training notebook
**AI-assisted knee MRI analysis.** Research prototype — not for clinical use.

This notebook trains the CartilAI model on the **MRNet** knee MRI dataset and saves everything the app needs.
It is self-contained: attach an MRNet dataset, turn on a **GPU (T4)** and **Internet**, then click **Run All**. No edits needed.

**What it does**
1. Finds the MRNet data under `/kaggle/input` automatically (any mirror with the standard MRNet layout).
2. Trains a ResNet18 (ImageNet-pretrained) applied to every sagittal slice, max-pooled across slices, with one output per finding: *Abnormal*, *ACL tear*, *Meniscal tear* (the MRNet-paper approach).
3. Picks the best epoch and decision thresholds on an internal validation split carved out of MRNet *train*.
4. Evaluates once on the official MRNet *validation* split (never seen during training): AUC with 95% CIs, sensitivity, specificity, ROC curves, confusion matrices.
5. Exports the weights, the results and a few sample scans to `/kaggle/working/cartilai_outputs/`, zipped as **`cartilai_outputs.zip`**.

Training stops by itself before a fixed time budget, so the whole run finishes in well under 2 hours on a T4.

_Generated from the CartilAI repository by `scripts/build_notebook.py`; do not edit by hand._"""

SETUP = '''# ===== Setup: environment, paths and configuration =====
import os, sys, time, json, shutil, zipfile, warnings
from pathlib import Path
import numpy as np
import torch
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_DIR = Path(os.environ.get("CARTILAI_INPUT_DIR", "/kaggle/input"))
WORK_DIR = Path(os.environ.get("CARTILAI_OUTPUT_DIR", "/kaggle/working"))
OUTPUT_DIR = WORK_DIR / "cartilai_outputs"
QUICK = os.environ.get("CARTILAI_QUICK") == "1"   # tiny smoke-test settings (used by the repo's tests)

CONFIG = json.loads(r"""__CONFIG_JSON__""")
if QUICK:
    CONFIG["data"].update({"image_size": 96, "num_workers": 0, "internal_val_fraction": 0.25})
    CONFIG["model"]["pretrained"] = False
    CONFIG["train"].update({"epochs": 2, "lr": 1e-4, "max_train_minutes": 10})

print(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name}")
    try:  # some PyTorch builds no longer support older GPUs such as the P100
        (torch.ones(8, device="cuda") * 2).sum().item()
    except Exception as exc:
        raise RuntimeError(
            f"The GPU '{name}' does not work with this PyTorch version ({exc}). "
            "In the right-hand panel choose Settings > Accelerator > 'GPU T4 x2' and Run All again."
        )
elif not QUICK:
    raise RuntimeError(
        "No GPU found. In the right-hand panel open 'Session options' (or Settings) > Accelerator, "
        "choose 'GPU T4 x2', then click Run All again. Training on CPU would take many hours."
    )
print("Input folder:", INPUT_DIR)
print("Outputs will be written to:", OUTPUT_DIR)'''

RUN = '''# ===== Train, evaluate and export =====
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
metrics = run_pipeline(CONFIG, INPUT_DIR, OUTPUT_DIR)'''

SHOW = '''# ===== Show the results =====
from IPython.display import Image as IPImage, display, Markdown

rows = ["| Finding | AUC (95% CI) | Sensitivity | Specificity | Accuracy |", "|---|---|---|---|---|"]
for key, m in metrics["findings"].items():
    rows.append(f"| {m['name']} | {m['auc']:.3f} ({m['auc_ci_low']:.3f}-{m['auc_ci_high']:.3f}) | "
                f"{m['sensitivity']:.1%} | {m['specificity']:.1%} | {m['accuracy']:.1%} |")
display(Markdown("### Held-out MRNet validation results\\n" + "\\n".join(rows)))
for name in ["roc_curves.png", "confusion_matrices.png", "training_curves.png"]:
    display(IPImage(filename=str(OUTPUT_DIR / "results" / name)))'''

PACKAGE = '''# ===== Package everything into one zip for download =====
zip_path = WORK_DIR / "cartilai_outputs.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            zf.write(path, Path("cartilai_outputs") / path.relative_to(OUTPUT_DIR))
print(f"Created {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB) containing:")
for path in sorted(OUTPUT_DIR.rglob("*")):
    if path.is_file():
        print("  ", path.relative_to(WORK_DIR))
print("\\nDONE. Download 'cartilai_outputs.zip' from the Output panel and follow KAGGLE_STEPS.md.")'''

OUTRO = """## Next steps
1. In the right-hand **Output** panel (or the notebook's *Output* tab after a saved run), download **`cartilai_outputs.zip`**.
2. Put the zip file in the CartilAI project folder (next to `app.py`), without unzipping it.
3. Run the launch command (`python start.py`). It installs the files into `models/`, `results/` and `samples/mrnet/` and the app switches from demo mode to the trained model.

MRNet data are subject to the Stanford MRNet research use agreement: keep the exported sample scans private."""


def build() -> Path:
    """Write the notebook and return its path."""
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    nb = nbformat.v4.new_notebook()
    cells = [nbformat.v4.new_markdown_cell(INTRO),
             nbformat.v4.new_code_cell(SETUP.replace("__CONFIG_JSON__", json.dumps(config, indent=2)))]
    cells.append(nbformat.v4.new_markdown_cell("## Pipeline code\nThe next cells contain the CartilAI source code "
                                               "(data loading, model, evaluation, training). They only define functions."))
    cells += [nbformat.v4.new_code_cell(module_source(m)) for m in MODULES]
    cells += [nbformat.v4.new_markdown_cell("## Run training and evaluation"), nbformat.v4.new_code_cell(RUN),
              nbformat.v4.new_code_cell(SHOW), nbformat.v4.new_code_cell(PACKAGE), nbformat.v4.new_markdown_cell(OUTRO)]
    for i, cell in enumerate(cells):
        cell.id = f"cartilai-{i:02d}"  # stable ids keep the generated file diff-friendly
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NOTEBOOK)
    return NOTEBOOK


if __name__ == "__main__":
    print(f"Wrote {build().relative_to(ROOT)}")
