"""Standalone Colab frontend for finite-versus-local NAR reach analysis.

Paste this complete file into one Colab cell. It reads existing fixed-N NAR
checkpoints, measures semantic and structural reach for 1-hop, 2-hop, and dense
GRIT at N=8,16,64, and writes only a new analysis extension. Graph-level shards
make measurement safe to resume.

Set ``PHASE = "figures"`` to regenerate and display all figures from cached CSV
results without loading GRIT or any checkpoint.
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
DRIVE_ROOT = Path("/content/drive/MyDrive/graph_specialisation_metrics/nar_grit")
TRAINING_RUN_NAME = "nar_grit_fixed_n_v3"
BASE_ANALYSIS_NAME = "canonical_nar_analysis_d128"
REACH_EXTENSION_NAME = "nar_reach_analysis_v1"
GRIT_DIR = Path("/content/GRIT")

MODELS = "1hop,2hop,dense"
NS = "8,16,64"
SEEDS = "0,1,2"
WIDTH = 128
GRAPHS = 16
DONORS_PER_SOURCE = 4
SEMANTIC_DONOR_GRAPHS = 256
ANALYSIS_SEED = 73021
ACCELERATOR = "cuda:0"
NUM_THREADS = 4


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
        / "nar_reach_analysis.py"
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

from graph_specialisation_metrics.synthetic.nar_reach_analysis import main


training_run_dir = DRIVE_ROOT / TRAINING_RUN_NAME
output_dir = (
    training_run_dir
    / BASE_ANALYSIS_NAME
    / "extensions"
    / REACH_EXTENSION_NAME
)

print(
    "\n[scope] Literal prior-work reference: Bamberger semantic encoded-input to "
    "central-output Jacobian range.\n"
    "[scope] Matched finite/local comparison: layer-1 routed-message distance profiles "
    "for the same semantic or structural donor events.\n"
    "[scope] Companion: layer-2 arrival/readout profiles, which should collapse to the "
    "central carrier under output projection.\n"
    "[scope] Reference: architecture support ceilings and known query/readout (d=2) "
    "and record/readout (d=1) distances; there is no unique learned-route oracle.\n"
    "[scope] Beneficial carriage: separate signed central-readout task benefit, not a "
    "distributed reach ground truth.\n",
    flush=True,
)

CELL_ARGS = [
    "--phase",
    PHASE,
    "--output-dir",
    str(output_dir),
    "--training-run-dir",
    str(training_run_dir),
    "--grit-dir",
    str(GRIT_DIR),
    "--models",
    MODELS,
    "--ns",
    NS,
    "--seeds",
    SEEDS,
    "--width",
    str(WIDTH),
    "--graphs",
    str(GRAPHS),
    "--donors-per-source",
    str(DONORS_PER_SOURCE),
    "--semantic-donor-graphs",
    str(SEMANTIC_DONOR_GRAPHS),
    "--analysis-seed",
    str(ANALYSIS_SEED),
    "--accelerator",
    ACCELERATOR,
    "--num-threads",
    str(NUM_THREADS),
]

result = main(CELL_ARGS)

if "figures" in result:
    from IPython.display import Image, display

    display_order = (
        "semantic_reach_layer1",
        "structural_reach_layer1",
        "expected_reach_layer1",
        "bamberger_semantic_range",
        "semantic_arrival_layer2",
        "structural_arrival_layer2",
        "beneficial",
    )
    for name in display_order:
        paths = result["figures"][name]
        print(f"\n[display] {name}: {paths['png']}", flush=True)
        display(Image(filename=paths["png"]))

print(f"\n[done] Reach analysis saved under {output_dir}", flush=True)
