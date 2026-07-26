"""Standalone Colab frontend for canonical NAR specialisation and carriage analysis.

Paste this complete file into one Colab cell. It mounts Drive, checks out the repository, installs
the package and pinned official GRIT environment, loads the fixed-N checkpoints from their default
Drive savepoints, runs the canonical README methodology, and saves atomic caches plus PNG/PDF
figures back to Drive.

The accuracy figure always uses every value in ``--all-ns``. Expensive score/causal/carriage work
uses only ``--analysis-ns``. Functional carriage is computed for the validation-selected seed in
each model-by-N cell; Beneficial carriage is deliberately not computed.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPOSITORY_BRANCH = "expansion/graphormer_specialisation"
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
            "git",
            "-C",
            str(COLAB_REPOSITORY),
            "remote",
            "set-url",
            "origin",
            authenticated,
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
    # Do not leave the private token in the checkout configuration.
    command(
        "git",
        "-C",
        str(COLAB_REPOSITORY),
        "remote",
        "set-url",
        "origin",
        REPOSITORY_URL,
    )
    command(sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPOSITORY))
    source = str(COLAB_REPOSITORY / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    # A failed Colab cell leaves imported modules alive.  Drop only this repository's modules so
    # rerunning after a branch update executes the newly checked-out analysis rather than stale
    # in-memory code.
    for module_name in tuple(sys.modules):
        if module_name == "graph_specialisation_metrics" or module_name.startswith(
            "graph_specialisation_metrics."
        ):
            del sys.modules[module_name]
    importlib.invalidate_caches()


bootstrap()

from graph_specialisation_metrics.synthetic.nar_canonical_analysis import main


# Clear, paper-strength defaults. All expensive stages are resumable and cache-bound to the
# protocol, checkpoint hash, graph/source/donor manifests, and bootstrap policy.
#
# Useful reruns:
#   CELL_ARGS = ["--phase", "figures", "--skip-install"]
#   CELL_ARGS = ["--phase", "performance"]  # no model inference
#   CELL_ARGS = ["--phase", "all", "--fast-dev-run", "--skip-install"]
CELL_ARGS = [
    "--phase",
    "all",
    "--drive-root",
    "/content/drive/MyDrive/graph_specialisation_metrics/nar_grit",
    "--training-run-name",
    "nar_grit_fixed_n_v3",
    "--analysis-name",
    "canonical_nar_analysis_d128",
    "--models",
    "1hop,2hop,dense",
    "--all-ns",
    "4,8,16,32,64,80",
    "--analysis-ns",
    "4,16,64",
    "--seeds",
    "0,1,2",
    "--analysis-width",
    "128",
    "--discovery-graphs",
    "48",
    "--causal-graphs",
    "48",
    "--clean-ablation-graphs",
    "64",
    "--semantic-donor-graphs",
    "256",
    "--source-nodes-per-graph",
    "2",
    "--donors-per-source",
    "8",
    "--graphs-per-batch",
    "48",
    "--analysis-seed",
    "31415",
    "--bootstrap-seed",
    "17071",
    "--activity-floor",
    "0.20",
    "--tail-fraction",
    "0.20",
    "--central-fraction",
    "0.20",
    "--accelerator",
    "cuda:0",
    "--num-threads",
    "4",
]

main(CELL_ARGS)
