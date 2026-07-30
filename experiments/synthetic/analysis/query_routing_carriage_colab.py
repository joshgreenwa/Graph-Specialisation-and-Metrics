"""One-cell Google Colab frontend for the query-routing carriage experiment.

Paste this complete file into one Colab cell. It:

1. mounts Google Drive;
2. checks out the pinned repository branch;
3. installs the package;
4. generates and caches every dataset split on Drive;
5. trains and caches all checkpoints and histories;
6. caches donor-resolved oracle, direction-matched/entrywise Jacobian, Functional, Beneficial,
   and off-protocol dose-ladder fields;
7. saves all CSV/JSON summaries and PNG/PDF figures to Drive; and
8. displays every PNG plus the health and primary-contrast tables inline.

Set ``PHASE = "figures"`` to regenerate and display figures entirely from the Drive caches,
without loading checkpoints or running model inference.
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

PHASE = "all"  # "all", "data", "train", "measure", or "figures"
DRIVE_OUTPUT = (
    "/content/drive/MyDrive/graph_specialisation_metrics/"
    "query_routing_carriage_v1"
)
DEVICE = "auto"
SEEDS = "0,1,2,3"
BETA_MULTIPLIERS = "0.25,0.5,1,2,4"
DOSE_ALPHAS = "0.001,0.01,0.05,0.1,0.25,0.5,1"
NUM_KEYS = 8
HIDDEN_DIM = 96
LAYERS = 3
HEADS = 4
TRAIN_GRAPHS = 12_288
VALIDATION_GRAPHS = 1_024
ID_GRAPHS = 256
OOD_GRAPHS = 256
DONOR_GRAPHS = 512
DONORS_PER_SOURCE = 16
BATCH_SIZE = 32
MAX_EPOCHS = 100
LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 1.0e-4
COMPUTE_BENEFICIAL = True
COMPUTE_DOSE_LADDER = True
PRODUCTION_BOOTSTRAP = True
FAST_DEV_RUN = False  # engineering smoke only; never use its outputs in the paper


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

    # Never retain the private token in the checked-out repository configuration.
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
        / "query_routing_carriage.py"
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

from graph_specialisation_metrics.synthetic.query_routing_carriage import main


CELL_ARGS = [
    "--phase",
    PHASE,
    "--output-dir",
    DRIVE_OUTPUT,
    "--device",
    DEVICE,
    "--seeds",
    SEEDS,
    "--beta-multipliers",
    BETA_MULTIPLIERS,
    "--dose-alphas",
    DOSE_ALPHAS,
    "--num-keys",
    str(NUM_KEYS),
    "--hidden-dim",
    str(HIDDEN_DIM),
    "--layers",
    str(LAYERS),
    "--heads",
    str(HEADS),
    "--train-graphs",
    str(TRAIN_GRAPHS),
    "--validation-graphs",
    str(VALIDATION_GRAPHS),
    "--id-graphs",
    str(ID_GRAPHS),
    "--ood-graphs",
    str(OOD_GRAPHS),
    "--donor-graphs",
    str(DONOR_GRAPHS),
    "--donors-per-source",
    str(DONORS_PER_SOURCE),
    "--batch-size",
    str(BATCH_SIZE),
    "--max-epochs",
    str(MAX_EPOCHS),
    "--learning-rate",
    str(LEARNING_RATE),
    "--weight-decay",
    str(WEIGHT_DECAY),
    "--num-threads",
    "2",
]
if not COMPUTE_BENEFICIAL:
    CELL_ARGS.append("--skip-beneficial")
if not COMPUTE_DOSE_LADDER:
    CELL_ARGS.append("--skip-dose-ladder")
if not PRODUCTION_BOOTSTRAP:
    CELL_ARGS.append("--no-bootstrap")
if FAST_DEV_RUN:
    CELL_ARGS.append("--fast-dev-run")

result = main(CELL_ARGS)

from IPython.display import Image, display

if "figures" in result:
    for name, files in result["figures"].items():
        print(f"[display:{name}] {files['png']}", flush=True)
        display(Image(filename=files["png"]))

results_dir = Path(DRIVE_OUTPUT) / "results"
health_path = results_dir / "model_health.csv"
contrast_path = results_dir / "primary_contrasts.csv"
if health_path.is_file() and contrast_path.is_file():
    import pandas as pd

    print("\nModel health", flush=True)
    display(pd.read_csv(health_path))
    print("\nPrimary contrasts", flush=True)
    display(pd.read_csv(contrast_path))

print(f"\nAll persistent artifacts: {DRIVE_OUTPUT}", flush=True)
