"""Standalone Colab launcher for the paper-aligned fixed-N NAR–GRIT experiment.

Paste this complete file into one Colab cell and run it.  The scientific implementation lives in
``src/graph_specialisation_metrics/synthetic/nar_grit_fixed.py`` so fixes made in the central
repository are picked up on the next Colab run. Checkpoints, analysis tensors, tables, and
PNG/PDF figures are cached on Drive.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPOSITORY_BRANCH = "codex/cfim-grit-experiments"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"


def command(*parts: str, check: bool = True) -> None:
    shown = [
        "https://***@github.com/" + part.split("@github.com/", 1)[1]
        if "@github.com/" in part
        else part
        for part in parts
    ]
    print(f"[cmd] {' '.join(shown)}", flush=True)
    subprocess.run(list(parts), check=check)


def bootstrap() -> None:
    try:
        from google.colab import drive, userdata

        drive.mount("/content/drive", force_remount=False)
    except ImportError as exc:
        raise RuntimeError("This launcher is intended for Google Colab") from exc

    token = userdata.get(SECRET_NAME) or os.environ.get(SECRET_NAME)
    if not token:
        raise RuntimeError(f"Colab secret {SECRET_NAME!r} is missing or empty")
    suffix = REPOSITORY_URL.removeprefix("https://github.com/")
    authenticated = (
        f"https://x-access-token:{quote(str(token).strip(), safe='')}@github.com/{suffix}"
    )

    if (COLAB_REPOSITORY / ".git").exists():
        command(
            "git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", authenticated
        )
        command("git", "-C", str(COLAB_REPOSITORY), "fetch", "origin", REPOSITORY_BRANCH)
        command("git", "-C", str(COLAB_REPOSITORY), "checkout", REPOSITORY_BRANCH)
        command(
            "git",
            "-C",
            str(COLAB_REPOSITORY),
            "reset",
            "--hard",
            f"origin/{REPOSITORY_BRANCH}",
        )
    else:
        if COLAB_REPOSITORY.exists():
            shutil.rmtree(COLAB_REPOSITORY)
        command(
            "git",
            "clone",
            "--branch",
            REPOSITORY_BRANCH,
            "--single-branch",
            authenticated,
            str(COLAB_REPOSITORY),
        )
    # Never leave the Colab secret in the checkout's persistent Git configuration.
    command(
        "git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", REPOSITORY_URL
    )
    command(sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPOSITORY))
    if str(COLAB_REPOSITORY / "src") not in sys.path:
        sys.path.insert(0, str(COLAB_REPOSITORY / "src"))


bootstrap()

from graph_specialisation_metrics.synthetic.nar_grit_fixed import main


# Normal run: trains only missing checkpoints, then caches mechanisms and makes every figure.
# Cheap reruns after training:
#   Retain --additional-seeds 3,4 when regenerating figures, so every performance cell contains
#   the same five repeats.
# Installation/plumbing only:
#   CELL_ARGS = ["--run-name", "nar_grit_smoke", "--fast-dev-run", "--allow-low-accuracy"]
CELL_ARGS = [
    "--run-name",
    "nar_grit_fixed_n_v3",
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
    "--ns",
    "4,8,16,32,64",
    "--mechanistic-ns",
    "4,16,64",
    "--seeds",
    "0,1,2",
    "--steps",
    "10000",
    "--early-stopping-loss-threshold",
    "0.001",
    # Two unconditional new runs for every model x width x N cell (five total repeats).
    "--additional-seeds",
    "3,4",
]

main(CELL_ARGS)
