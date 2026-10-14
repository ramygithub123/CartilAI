"""Install the files produced by the Kaggle notebook into this project.

Accepts ``cartilai_outputs.zip`` (or Kaggle's full "download all" zip, or an
unzipped folder) and copies:

* ``models/cartilai_resnet18.pt``  -> ``models/``
* ``results/*``                    -> ``results/``
* ``samples/mrnet/*``              -> ``samples/mrnet/``

then fills the results table in ``README.md``. ``start.py`` calls this
automatically when it finds ``cartilai_outputs*.zip`` in the project folder.

Usage::

    python -m src.install_outputs path/to/cartilai_outputs.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

WEIGHTS_NAME = "cartilai_resnet18.pt"
README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


def _extract_all(src: Path, dest: Path) -> None:
    """Extract ``src`` into ``dest``, including zips nested inside it."""
    with zipfile.ZipFile(src) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"Unsafe path in zip: {member}")
        zf.extractall(dest)
    for inner in list(dest.rglob("*.zip")):
        sub = inner.with_suffix("")
        sub.mkdir(exist_ok=True)
        _extract_all(inner, sub)
        inner.unlink()


def locate_output_root(folder: Path) -> Path:
    """Find the folder that contains ``models/cartilai_resnet18.pt`` (or ``results/metrics.json``)."""
    for pattern in (f"models/{WEIGHTS_NAME}", "results/metrics.json"):
        hits = sorted(folder.rglob(pattern), key=lambda p: len(p.parts))
        if hits:
            return hits[0].parent.parent
    raise FileNotFoundError(f"No CartilAI outputs (models/{WEIGHTS_NAME} or results/metrics.json) found in {folder}")


def results_table(metrics: dict[str, Any]) -> str:
    """Markdown table of held-out metrics for the README."""
    lines = [
        f"Evaluated on the {metrics.get('evaluation_set', 'held-out set')}.",
        "",
        "| Finding | AUC (95% CI) | Sensitivity | Specificity | Accuracy |",
        "|---|---|---|---|---|",
    ]
    for m in metrics["findings"].values():
        auc = "n/a" if m.get("auc") is None else f"{m['auc']:.3f}"
        ci = "" if m.get("auc_ci_low") is None else f" ({m['auc_ci_low']:.3f}–{m['auc_ci_high']:.3f})"
        lines.append(f"| {m['name']} | {auc}{ci} | {m['sensitivity']:.1%} | {m['specificity']:.1%} | {m['accuracy']:.1%} |")
    t = metrics.get("training", {})
    lines += ["", f"Mean AUC {metrics.get('mean_auc', float('nan')):.3f}. Thresholds chosen on "
              f"{metrics.get('thresholds_selected_on', 'an internal validation split')}. Best epoch "
              f"{t.get('best_epoch')} of {t.get('epochs_completed')}, {t.get('train_minutes', 0):.0f} min on "
              f"{t.get('gpu') or t.get('device')}. ROC curves: `results/roc_curves.png`; confusion matrices: "
              "`results/confusion_matrices.png`."]
    return "\n".join(lines)


def update_readme(metrics: dict[str, Any], readme: Path) -> bool:
    """Replace the README section between the RESULTS markers. Returns True if changed."""
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    readme.write_text(f"{head}{README_START}\n{results_table(metrics)}\n{README_END}{tail}", encoding="utf-8")
    return True


def install(source: str | Path, project_root: str | Path = PROJECT_ROOT) -> list[str]:
    """Copy Kaggle outputs into the project.

    Args:
        source: ``cartilai_outputs.zip``, a Kaggle download zip, or an extracted folder.
        project_root: The CartilAI project folder.

    Returns:
        Project-relative paths of the installed files.
    """
    source, project_root = Path(source).expanduser(), Path(project_root)
    installed: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        if source.is_file():
            _extract_all(source, Path(tmp))
            base = locate_output_root(Path(tmp))
        else:
            base = locate_output_root(source)
        for sub in ("models", "results", "samples/mrnet"):
            src_dir = base / sub
            if not src_dir.is_dir():
                continue
            dst_dir = project_root / sub
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in sorted(src_dir.iterdir()):
                if f.is_file():
                    shutil.copy2(f, dst_dir / f.name)
                    installed.append(f"{sub}/{f.name}")
    metrics_path = project_root / "results" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as fh:
            if update_readme(json.load(fh), project_root / "README.md"):
                installed.append("README.md (results table)")
    return installed


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Install CartilAI Kaggle outputs into this project")
    parser.add_argument("source", help="cartilai_outputs.zip or the extracted folder")
    args = parser.parse_args()
    for path in install(args.source):
        print("installed", path)


if __name__ == "__main__":
    main()
