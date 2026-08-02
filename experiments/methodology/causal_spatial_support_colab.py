"""Standalone Colab frontend for causal spatial support of specialist heads.

The first run measures dense ZINC and dense QM9 using a small held-out event
sample and caches every graph/channel shard to Drive. Set ``PHASE='figures'``
afterward to regenerate and display the paper figures without loading GRIT,
checkpoints, datasets, or RRWP features.
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
DRIVE_ROOT = Path("/content/drive/MyDrive")
CANONICAL_ROOT_CANDIDATES = (
    DRIVE_ROOT / "graph_specialisation_metrics/canonical_methodology_v4_zinc_qm9",
    DRIVE_ROOT / "graph_specialisation_metrics/canonical_methodology",
)
OUTPUT_DIR = DRIVE_ROOT / "graph_specialisation_metrics/causal_spatial_support_v1"
TASKS = "zinc,qm9_gap_dense"
TRAIN_SEED = 42
GRAPHS = 8
SOURCES_PER_GRAPH = 2
DONORS_PER_SOURCE = 1
HEADS_PER_FAMILY = 3
LONG_RANGE_RADIUS = 2
BOOTSTRAP_REPLICATES = 2_000
ANALYSIS_SEED = 72_019
ACCELERATOR = "cuda:0"
INSTALL_DEPENDENCIES = True
FORCE = False


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
    from google.colab import drive, userdata

    drive.mount("/content/drive", force_remount=False)
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
        command("git", "-C", str(COLAB_REPOSITORY), "reset", "--hard", f"origin/{REPOSITORY_BRANCH}")
    else:
        if COLAB_REPOSITORY.exists():
            shutil.rmtree(COLAB_REPOSITORY)
        command("git", "clone", "--branch", REPOSITORY_BRANCH, "--single-branch", authenticated, str(COLAB_REPOSITORY))
    command("git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", REPOSITORY_URL)
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


def complete_root(root: Path) -> bool:
    for task in tuple(value.strip() for value in TASKS.split(",") if value.strip()):
        task_root = root / task / f"seed_{TRAIN_SEED}"
        if not (task_root / "cache/scores/raw.pt").is_file():
            return False
        if not (task_root / "model.json").is_file():
            return False
    return (root / "protocol.json").is_file()


available = [root for root in CANONICAL_ROOT_CANDIDATES if complete_root(root)]
canonical_root = available[0] if available else CANONICAL_ROOT_CANDIDATES[0]
if not available:
    print(f"[cache:warning] canonical ZINC/QM9 root was not found; expected {canonical_root}")
if PHASE != "figures" and INSTALL_DEPENDENCIES:
    from graph_specialisation_metrics.carriage import env

    env.install_dependencies(pyg_version="2.2.0")

from graph_specialisation_metrics.methodology.causal_spatial_support import main


print(
    "\n[question] Which spatial portion of a causally important specialist head "
    "actually mediates the output?\n"
    "[A] Clean direct attention from each intervened source to receivers by pristine SPD.\n"
    "[S] Immutable canonical discovery-split internal response by source-carrier SPD.\n"
    "[M] Held-out symmetric injection/restoration when only one carrier shell is patched.\n"
    "[families] Three semantic specialists, three structural specialists, three high-J "
    "generalists, and registered same-layer central controls.\n"
    "[overlap] Each family is patched jointly and individually; joint/sum below one "
    "indicates overlapping realised pathways, not minimal task necessity.\n"
    "[cache] Every task/graph/channel shard is resumable on Drive. PHASE='figures' "
    "does not load GRIT, checkpoints, datasets, or RRWP.\n",
    flush=True,
)

args = [
    "--canonical-root", str(canonical_root),
    "--output-dir", str(OUTPUT_DIR),
    "--tasks", TASKS,
    "--train-seed", str(TRAIN_SEED),
    "--phase", PHASE,
    "--graphs", str(GRAPHS),
    "--sources-per-graph", str(SOURCES_PER_GRAPH),
    "--donors-per-source", str(DONORS_PER_SOURCE),
    "--heads-per-family", str(HEADS_PER_FAMILY),
    "--long-range-radius", str(LONG_RANGE_RADIUS),
    "--bootstrap-replicates", str(BOOTSTRAP_REPLICATES),
    "--analysis-seed", str(ANALYSIS_SEED),
    "--accelerator", ACCELERATOR,
]
if FORCE:
    args.append("--force")
result = main(args)

from IPython.display import Image, display

for key in ("headline_png", "overlap_png"):
    path = result.get("figures", {}).get(key)
    if path and Path(path).is_file():
        display(Image(filename=path))

try:
    import pandas as pd

    table = OUTPUT_DIR / "results/causal_spatial_support_statistics.csv"
    if table.is_file():
        display(pd.read_csv(table))
except Exception as error:
    print(f"[display:warning] {type(error).__name__}: {error}")
print(f"[saved] {OUTPUT_DIR}")
