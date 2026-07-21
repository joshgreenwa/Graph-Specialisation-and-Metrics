"""Standalone Colab launcher for NAR attention-faithfulness/transport analysis.

Paste this complete file into one Google Colab cell and run it. It mounts Drive,
checks out the repository, installs the package and pinned official GRIT stack,
loads the validation-selected fixed-N checkpoints, and runs the resumable
metric, causal, follow-up and figure phases. Expensive results are cached per checkpoint
under ``transport_mechanisms_v2/d<width>``; rerunning a figures phase never
loads a model.

The default analyses completed width-64 checkpoints. Change only
``--analysis-width`` to ``128`` when those checkpoints are ready.
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
    except ImportError as exc:  # pragma: no cover - Colab-only path
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
            "git", "-C", str(COLAB_REPOSITORY), "reset", "--hard",
            f"origin/{REPOSITORY_BRANCH}",
        )
    else:
        if COLAB_REPOSITORY.exists():
            shutil.rmtree(COLAB_REPOSITORY)
        command(
            "git", "clone", "--branch", REPOSITORY_BRANCH, "--single-branch",
            authenticated, str(COLAB_REPOSITORY),
        )
    # Do not persist the Colab secret in Git configuration.
    command(
        "git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", REPOSITORY_URL
    )
    command(sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPOSITORY))
    source = str(COLAB_REPOSITORY / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


bootstrap()

from graph_specialisation_metrics.synthetic.nar_transport_mechanisms import main


# Resumable alternatives after the first run:
#   CELL_ARGS = ["--analysis-width", "64", "--phase", "analyze"]
#   CELL_ARGS = ["--analysis-width", "64", "--phase", "causal"]
#   CELL_ARGS = ["--analysis-width", "64", "--phase", "followups"]
#   CELL_ARGS = ["--analysis-width", "64", "--phase", "figures", "--skip-install"]
# Installation/plumbing check (requires the corresponding N=4 seed-0 checkpoints):
#   CELL_ARGS = ["--analysis-width", "64", "--phase", "all", "--fast-dev-run"]
CELL_ARGS = [
    "--drive-root",
    "/content/drive/MyDrive/graph_specialisation_metrics/nar_grit",
    "--run-name",
    "nar_grit_fixed_n_v3",
    "--analysis-width",
    "64",
    "--phase",
    "all",
    "--models",
    "1hop,2hop,dense",
    "--ns",
    "4,8,16,32,64",
    "--anchor-ns",
    "4,16,64",
    "--seeds",
    "0,1,2,3,4",
    "--donors",
    "4",
    "--discovery-graphs",
    "32",
    "--mechanism-graphs",
    "96",
    "--robustness-graphs",
    "48",
    "--causal-graphs",
    "256",
    "--causal-donors",
    "4",
    "--followup-graphs",
    "128",
    "--followup-donors",
    "2",
    "--ablation-random-rankings",
    "8",
]

main(CELL_ARGS)
