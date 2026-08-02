"""Standalone Colab frontend for apparent reach versus task necessity.

Paste this complete file into one Colab cell. It trains a tiny minimum-norm graph
filter on real ZINC molecular topologies with either redundant or essential
distant cues. Jacobian range and finite Functional carriage measure apparent
learned use; frozen deletion and a local-only refit separate reliance from task
necessity. Results and figures are cached to Drive.

Set ``PHASE = "figures"`` to regenerate the paper figure from cached CSV files.
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
REPOSITORY_BRANCH = "expansion/carriage_experiments"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"

# ----------------------------- experiment controls -----------------------------

PHASE = "all"  # "all", "measure", or "figures"
DRIVE_OUTPUT = (
    "/content/drive/MyDrive/graph_specialisation_metrics/"
    "molecular_redundancy_reach_v1"
)
DATA_ROOT = "/content/zinc_subset"
TRAIN_GRAPHS = 512
VAL_GRAPHS = 96
TEST_GRAPHS = 96
COPY_DISTANCES = "2,4,6"
SEEDS = "0,1,2,3"
TRAIN_STEPS = 500
BOOTSTRAP_REPLICATES = 2_000


def command(*parts: str) -> None:
    shown = [
        "https://***@github.com/" + part.split("@github.com/", 1)[1]
        if "@github.com/" in part
        else part
        for part in parts
    ]
    print(f"[cmd] {' '.join(shown)}", flush=True)
    subprocess.run(list(parts), check=True)


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
    command(
        "git",
        "-C",
        str(COLAB_REPOSITORY),
        "remote",
        "set-url",
        "origin",
        REPOSITORY_URL,
    )
    if PHASE != "figures":
        command(sys.executable, "-m", "pip", "install", "-q", "torch_geometric")
    command(sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPOSITORY))

    source = str((COLAB_REPOSITORY / "src").resolve())
    sys.path[:] = [entry for entry in sys.path if entry != source]
    sys.path.insert(0, source)
    for module_name in tuple(sys.modules):
        if module_name == "graph_specialisation_metrics" or module_name.startswith(
            "graph_specialisation_metrics."
        ):
            del sys.modules[module_name]
    importlib.invalidate_caches()


bootstrap()

from graph_specialisation_metrics.synthetic.molecular_redundancy_reach import main


CELL_ARGS = [
    "--phase",
    PHASE,
    "--output-dir",
    DRIVE_OUTPUT,
    "--data-root",
    DATA_ROOT,
    "--train-graphs",
    str(TRAIN_GRAPHS),
    "--val-graphs",
    str(VAL_GRAPHS),
    "--test-graphs",
    str(TEST_GRAPHS),
    "--copy-distances",
    COPY_DISTANCES,
    "--seeds",
    SEEDS,
    "--train-steps",
    str(TRAIN_STEPS),
    "--bootstrap-replicates",
    str(BOOTSTRAP_REPLICATES),
]

result = main(CELL_ARGS)
for path in result.get("figures", {}).values():
    if str(path).lower().endswith(".png"):
        try:
            from IPython.display import Image, display

            display(Image(filename=str(path)))
        except Exception as error:
            print(f"[display:warning] {type(error).__name__}: {error}")
