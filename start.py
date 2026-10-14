"""One-command launcher for CartilAI: ``python start.py``

1. Creates a private Python environment in ``.venv`` (first run only).
2. Installs the requirements into it (first run, or when requirements.txt changes).
3. Installs Kaggle outputs if a ``cartilai_outputs*.zip`` sits in this folder.
4. Starts the web app and opens it in your browser.

Extra arguments are passed to ``app.py`` (e.g. ``python start.py --share``).
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQS = ROOT / "requirements.txt"
STAMP = VENV / ".cartilai-requirements.sha256"


def venv_python() -> Path:
    """Path of the Python executable inside ``.venv``."""
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd: list[str]) -> None:
    """Run a command, exiting with a readable message if it fails."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    if subprocess.call([str(c) for c in cmd], cwd=ROOT) != 0:
        sys.exit(f"\nCartilAI setup failed while running: {' '.join(str(c) for c in cmd)}")


def ensure_environment() -> Path:
    """Create the virtual environment and install requirements when needed."""
    if sys.version_info < (3, 10):
        sys.exit(f"CartilAI needs Python 3.10 or newer (found {platform.python_version()}). Install it from python.org.")
    if not venv_python().exists():
        print("Creating a private Python environment in .venv (first run only) ...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    py = venv_python()
    digest = hashlib.sha256(REQS.read_bytes()).hexdigest()
    if not STAMP.exists() or STAMP.read_text().strip() != digest:
        print("Installing CartilAI's dependencies (first run takes a few minutes) ...")
        run([py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
        if sys.platform.startswith("linux"):
            # The default Linux PyTorch wheel bundles ~3 GB of CUDA libraries; the app only needs the CPU build.
            subprocess.call([str(py), "-m", "pip", "install", "--quiet", "torch", "torchvision",
                             "--index-url", "https://download.pytorch.org/whl/cpu"], cwd=ROOT)
        run([py, "-m", "pip", "install", "--quiet", "-r", REQS])
        STAMP.write_text(digest)
    return py


def install_kaggle_outputs(py: Path) -> None:
    """Install any ``cartilai_outputs*.zip`` found in the project folder, then mark it as installed."""
    for zip_path in sorted(ROOT.glob("cartilai_outputs*.zip")):
        print(f"Found {zip_path.name}: installing the trained model, results and sample scans ...")
        run([py, "-m", "src.install_outputs", zip_path])
        zip_path.rename(zip_path.with_name(zip_path.stem + ".installed.zip"))


def main() -> None:
    """Set up (if needed) and launch the app."""
    py = ensure_environment()
    install_kaggle_outputs(py)
    print("\nStarting CartilAI at http://127.0.0.1:7860  (press Ctrl+C to stop)\n", flush=True)
    sys.exit(subprocess.call([str(py), "app.py", *sys.argv[1:]], cwd=ROOT))


if __name__ == "__main__":
    main()
