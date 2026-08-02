"""Standalone Colab frontend for Peptides-struct finite/Jacobian reach.

Paste this complete file into one Colab cell.  The pilot preset deliberately
uses a tiny semantic-only sample because Peptides graphs are much larger than
ZINC/QM9.  Every completed graph is cached independently to Drive.  Set
``PHASE = "figures"`` to rebuild plots without loading a checkpoint.
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
PILOT = True
# In pilot mode, sample only the required official-split graphs before RRWP is
# materialised.  The base molecular archive is read once, but the 15,535-graph
# RRWP transform and full-split checkpoint evaluation are avoided.
LIMIT_DATASET_TO_ANALYSIS = PILOT
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/graph_specialisation_metrics/"
    "peptides_struct_bamberger_functional_reach_v1"
)

# Both presets compare the trained seed-41 1-hop and dense checkpoints on the
# same held-out graph IDs.  To validate only one checkpoint, shorten this string.
TASKS = "peptides_struct_1hop,peptides_struct"
SEED = 41
CHANNELS = "semantic" if PILOT else "semantic,structural"
GRAPHS = 4 if PILOT else 32
SOURCES_PER_GRAPH = 2 if PILOT else 4
DONORS_PER_SOURCE = 1 if PILOT else 2
SEMANTIC_DONOR_GRAPHS = 32 if PILOT else 128
BAMBERGER_OUTPUT_NODES = 2 if PILOT else 4
BAMBERGER_OUTPUT_CHANNELS = 2 if PILOT else 8
INTERPOLATION_DOSES = "0.01,0.1,0.5,1.0" if PILOT else "0.01,0.02,0.05,0.1,0.25,0.5,1.0"
INTERPOLATION_BATCH_SIZE = 4 if PILOT else 8
BOOTSTRAP_REPLICATES = 500 if PILOT else 2_000

# These extensions are not needed for the finite/Jacobian trajectory headline
# and are expensive on ~150-node Peptides graphs.  They can be enabled later;
# completed core graph shards will be reused.
RUN_OUTPUT_CARRIAGE = False
RUN_BENEFICIAL = False
RUN_SURVIVAL = False
RUN_SCALE_ANALYSIS = False

# Extension controls (used only when the corresponding switch above is True).
SURVIVAL_CARRIERS_PER_GRAPH = 1
SURVIVAL_DRAWS = 1
SURVIVAL_TAIL_RADII = "2,3,4,5"
SURVIVAL_REPLACEMENT_CANDIDATES = 16
SURVIVAL_EXACT_LIMIT = 8
SURVIVAL_RANDOM_ATTEMPTS = 128
SURVIVAL_REPLICA_BATCH_SIZE = 32
BENEFICIAL_DONORS_PER_SOURCE = 1
BENEFICIAL_ATOL = 1.0e-5
BENEFICIAL_RTOL = 1.0e-4
BENEFICIAL_MAX_INTERVALS = 32

ANALYSIS_SEED = 91_021
ACCELERATOR = "cuda:0"
NUM_THREADS = 4
INSTALL_DEPENDENCIES = True


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
        command("git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", authenticated)
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
    command("git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", REPOSITORY_URL)
    command(sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPOSITORY))

    source_path = (COLAB_REPOSITORY / "src").resolve()
    backend_path = (
        source_path
        / "graph_specialisation_metrics"
        / "peptides_struct_reach_analysis.py"
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

from graph_specialisation_metrics.peptides_struct_reach_analysis import main


print(
    "\n[scope] Dataset: Peptides-struct, seed-41 trained GRIT checkpoints.\n"
    "[scope] Semantic donor swaps replace all nine OGB atom fields together.\n"
    "[scope] Bamberger uses the equivalent differentiable concatenated one-hot "
    "input and final pre-pooling hidden-state Jacobian.\n"
    "[scope] Functional carriage and the interpolation sweep use the identical "
    "source/donor events and task projection used for ZINC and QM9.\n"
    "[scope] Disconnected node pairs have undefined SPD and are excluded from "
    "both radial profiles.\n"
    f"[scope] Pilot={PILOT}; tasks={TASKS}; graphs={GRAPHS}; channels={CHANNELS}.\n"
    f"[scope] Analysis-only dataset loading={LIMIT_DATASET_TO_ANALYSIS}; pilot "
    "samples the required official-split graphs before RRWP.\n"
    "[scope] Core graph shards are cached independently, so increasing GRAPHS or "
    "enabling extensions reuses completed compatible work.\n",
    flush=True,
)

CELL_ARGS = [
    "--phase", PHASE,
    "--output-dir", str(OUTPUT_DIR),
    "--tasks", TASKS,
    "--channels", CHANNELS,
    "--seed", str(SEED),
    "--graphs", str(GRAPHS),
    "--sources-per-graph", str(SOURCES_PER_GRAPH),
    "--donors-per-source", str(DONORS_PER_SOURCE),
    "--semantic-donor-graphs", str(SEMANTIC_DONOR_GRAPHS),
    "--bamberger-output-nodes", str(BAMBERGER_OUTPUT_NODES),
    "--bamberger-output-channels", str(BAMBERGER_OUTPUT_CHANNELS),
    "--interpolation-doses", INTERPOLATION_DOSES,
    "--interpolation-batch-size", str(INTERPOLATION_BATCH_SIZE),
    "--survival-carriers-per-graph", str(SURVIVAL_CARRIERS_PER_GRAPH),
    "--survival-draws", str(SURVIVAL_DRAWS),
    "--survival-tail-radii", SURVIVAL_TAIL_RADII,
    "--survival-replacement-candidates", str(SURVIVAL_REPLACEMENT_CANDIDATES),
    "--survival-exact-limit", str(SURVIVAL_EXACT_LIMIT),
    "--survival-random-attempts", str(SURVIVAL_RANDOM_ATTEMPTS),
    "--survival-replica-batch-size", str(SURVIVAL_REPLICA_BATCH_SIZE),
    "--beneficial-donors-per-source", str(BENEFICIAL_DONORS_PER_SOURCE),
    "--beneficial-atol", str(BENEFICIAL_ATOL),
    "--beneficial-rtol", str(BENEFICIAL_RTOL),
    "--beneficial-max-intervals", str(BENEFICIAL_MAX_INTERVALS),
    "--bootstrap-replicates", str(BOOTSTRAP_REPLICATES),
    "--analysis-seed", str(ANALYSIS_SEED),
    "--accelerator", ACCELERATOR,
    "--num-threads", str(NUM_THREADS),
]
if not RUN_OUTPUT_CARRIAGE:
    CELL_ARGS.append("--no-output-carriage")
if not RUN_BENEFICIAL:
    CELL_ARGS.append("--no-beneficial")
if not RUN_SURVIVAL:
    CELL_ARGS.append("--no-survival")
if not RUN_SCALE_ANALYSIS:
    CELL_ARGS.append("--no-scale-analysis")
if LIMIT_DATASET_TO_ANALYSIS:
    CELL_ARGS.append("--limit-dataset-to-analysis")
if not INSTALL_DEPENDENCIES:
    CELL_ARGS.append("--skip-dependency-install")

result = main(CELL_ARGS)

try:
    import pandas as pd
    from IPython.display import Image, display

    for key, title in (
        ("health", "Model health"),
        ("core_failures", "Soft core-reach failure audit"),
        ("trajectory_summary", "Finite profile-trajectory decomposition"),
        ("semantic_usage_contrasts", "Paired semantic-estimand contrasts"),
        ("interpolation_rows", "Interpolation-sweep summary"),
        ("figure_failures", "Soft figure-generation failure audit"),
    ):
        rows = result.get(key) or result.get("measurement", {}).get(key)
        if rows:
            print(f"\n{title}", flush=True)
            display(pd.DataFrame(rows))

    for name, formats in result.get("figures", {}).items():
        path = formats["png"]
        print(f"\n[display] {name}: {path}", flush=True)
        display(Image(filename=path))
except ImportError:
    pass

print(f"\n[done] Analysis saved under {OUTPUT_DIR}", flush=True)
