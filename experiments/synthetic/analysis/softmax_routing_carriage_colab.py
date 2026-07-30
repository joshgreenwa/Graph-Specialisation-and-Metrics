"""Standalone Colab frontend for learned softmax-routing carriage.

Paste this complete file into one Colab cell. It mounts Drive, checks out the
carriage-experiments branch, installs the package, trains four tiny categorical
key-value routers, caches checkpoints and measurements, saves the publication
figure as PNG/PDF, and displays the PNG inline.

For a figure-only rerun, change ``PHASE`` to ``"figures"``. This reads cached
measurements and performs no training or model inference.
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

PHASE = "all"  # "all", "train", "measure", or "figures"
DRIVE_OUTPUT = (
    "/content/drive/MyDrive/graph_specialisation_metrics/"
    "softmax_routing_carriage_v1"
)
SEEDS = "0,1,2,3"
SHARPNESS_MULTIPLIERS = "0.125,0.25,0.5,1,2,4"
NUM_KEYS = 6
FAR_DISTANCE = 8
TRAIN_STEPS = 900
EVALUATION_GRAPHS = 96


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
    source_path = (COLAB_REPOSITORY / "src").resolve()
    backend_path = (
        source_path
        / "graph_specialisation_metrics"
        / "synthetic"
        / "softmax_routing_carriage.py"
    )
    if not backend_path.is_file():
        raise RuntimeError(
            f"Checked-out branch {REPOSITORY_BRANCH!r} does not contain {backend_path}"
        )

    source = str(source_path)
    sys.path[:] = [entry for entry in sys.path if entry != source]
    sys.path.insert(0, source)
    for module_name in tuple(sys.modules):
        if module_name == "graph_specialisation_metrics" or module_name.startswith(
            "graph_specialisation_metrics."
        ):
            del sys.modules[module_name]
    importlib.invalidate_caches()


bootstrap()

from graph_specialisation_metrics.synthetic.softmax_routing_carriage import main


CELL_ARGS = [
    "--phase",
    PHASE,
    "--output-dir",
    DRIVE_OUTPUT,
    "--seeds",
    SEEDS,
    "--sharpness-multipliers",
    SHARPNESS_MULTIPLIERS,
    "--num-keys",
    str(NUM_KEYS),
    "--far-distance",
    str(FAR_DISTANCE),
    "--train-steps",
    str(TRAIN_STEPS),
    "--evaluation-graphs",
    str(EVALUATION_GRAPHS),
    "--num-threads",
    "2",
]

result = main(CELL_ARGS)

if "figures" in result:
    from IPython.display import Image, display

    figure_path = result["figures"]["png"]
    print(f"[display] {figure_path}", flush=True)
    display(Image(filename=figure_path))
