"""Standalone Colab launcher for the repository-controlled NAR–GRIT experiment.

Paste this complete file into one Colab cell and run it.  The scientific implementation lives in
``src/graph_specialisation_metrics/synthetic/nar_grit.py`` so fixes made in the central repository
are picked up on the next Colab run.  Checkpoints, analysis tensors, tables, and PNG/PDF figures
are cached on Drive.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")


def command(*parts: str, check: bool = True) -> None:
    print(f"[cmd] {' '.join(parts)}", flush=True)
    subprocess.run(list(parts), check=check)


def bootstrap() -> None:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except ImportError as exc:
        raise RuntimeError("This launcher is intended for Google Colab") from exc

    if not (COLAB_REPOSITORY / ".git").exists():
        command("git", "clone", REPOSITORY_URL, str(COLAB_REPOSITORY))
    else:
        # Preserve any deliberate Colab-side edit: a non-fast-forward pull fails rather than
        # overwriting it, and the existing checkout remains usable.
        command("git", "-C", str(COLAB_REPOSITORY), "pull", "--ff-only", check=False)
    command(sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPOSITORY))
    if str(COLAB_REPOSITORY / "src") not in sys.path:
        sys.path.insert(0, str(COLAB_REPOSITORY / "src"))


bootstrap()

from graph_specialisation_metrics.synthetic.nar_grit import main


# Normal run: trains only missing checkpoints, then caches mechanisms and makes every figure.
# Cheap reruns after training:
#   CELL_ARGS = ["--run-name", "nar_grit_v1", "--phase", "analyze"]
#   CELL_ARGS = ["--run-name", "nar_grit_v1", "--phase", "figures"]
# Installation/plumbing only:
#   CELL_ARGS = ["--run-name", "nar_grit_smoke", "--fast-dev-run", "--allow-low-accuracy"]
CELL_ARGS = [
    "--run-name",
    "nar_grit_v1",
    "--phase",
    "all",
    "--models",
    "1hop,2hop,dense",
    "--widths",
    "64,128",
    "--analysis-width",
    "128",
    "--heads",
    "8",
    "--layers",
    "2",
    "--train-ns",
    "4,8,16,32,64",
    "--eval-ns",
    "4,8,12,16,24,32,48,64",
    "--mechanistic-ns",
    "4,16,32,64",
    "--seeds",
    "0,1,2",
]

main(CELL_ARGS)

