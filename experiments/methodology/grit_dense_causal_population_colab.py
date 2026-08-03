"""Paste/run this cell in Colab for dense ZINC and QM9 GRIT causal populations.

The two tasks use separate per-task caches under one Drive root and resume at the
per-graph level.  Their registered training checkpoint and dataset roots are reused
read-only; only the causal-population outputs are new.
"""

# ============================ paste from here ============================
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPO_REVISION = "expansion/graphormer_causal_population"
REPO_DIR = Path("/content/Graph-Specialisation-and-Metrics")
GITHUB_SECRET = "dissertation_key"

DRIVE_ROOT = Path("/content/drive/MyDrive")
OUTPUT_ROOT = (
    DRIVE_ROOT / "graph_specialisation_metrics/grit_dense_causal_population_paper"
)

TASKS = ("zinc", "qm9_gap_dense")
TRAIN_SEED = 42
CHECKPOINTS = {}
TASK_OVERRIDES = {
    # The registrations already use the training Drive roots. Override only when
    # relocating an exact checkpoint, dataset cache, config, or GRIT clone.
    # "zinc": {
    #     "drive_dir": "/content/drive/MyDrive/grit_zinc_official",
    # },
    # "qm9_gap_dense": {
    #     "drive_dir": "/content/drive/MyDrive/grit_qm9_gap_dense",
    #     "dataset_dir": "/content/drive/MyDrive/grit_qm9_gap_data",
    # },
}

PHASE = "all"  # "run", "figures", or "all"
FORCE = False
FORCE_FRESH_GRIT = False
SKIP_DEPENDENCY_INSTALL = False
ACCELERATOR = "cuda:0"

# Match the PCQM4Mv2 paper contract: disjoint discovery, intervention, and
# clean-ablation molecules plus a disjoint semantic donor population.
DISCOVERY_GRAPHS = 256
CAUSAL_GRAPHS = 256
CLEAN_ABLATION_GRAPHS = 256
SEMANTIC_DONOR_GRAPHS = 2_000
SOURCES_PER_GRAPH = 6
DONORS_PER_SOURCE = 8

# A100 80 GB-oriented throughput preset. Head chunks back off automatically on
# CUDA OOM; reduce GRAPHS_PER_BATCH first on a smaller GPU.
GRAPHS_PER_BATCH = 16
HEAD_BATCH_SIZE = 64
EVENT_BATCH_SIZE = SOURCES_PER_GRAPH * DONORS_PER_SOURCE

# The same discovery-only matching policy as the PCQM4Mv2 population analysis.
# Keep these fixed before inspecting causal outcomes. If an 80-head GRIT model is
# not estimable, the gate is saved and reported rather than silently relaxed.
POPULATION_HEAD_PAIRS = 16
POPULATION_MINIMUM_PAIRS = 12

from google.colab import drive, userdata

drive.mount("/content/drive", force_remount=False)
token = userdata.get(GITHUB_SECRET)
if not token:
    raise RuntimeError(f"Colab secret {GITHUB_SECRET!r} is missing or empty")
suffix = REPO_URL.removeprefix("https://github.com/")
authenticated = f"https://x-access-token:{token.strip()}@github.com/{suffix}"
if not (REPO_DIR / ".git").is_dir():
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            REPO_REVISION,
            "--single-branch",
            authenticated,
            str(REPO_DIR),
        ],
        check=True,
    )
else:
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "remote", "set-url", "origin", authenticated],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "fetch", "origin", REPO_REVISION],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "switch", "--detach", "FETCH_HEAD"],
        check=True,
    )
subprocess.run(
    ["git", "-C", str(REPO_DIR), "remote", "set-url", "origin", REPO_URL],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)],
    check=True,
)

source = str(REPO_DIR / "src")
if source not in sys.path:
    sys.path.insert(0, source)
for module_name in tuple(sys.modules):
    if module_name == "graph_specialisation_metrics" or module_name.startswith(
        "graph_specialisation_metrics."
    ):
        del sys.modules[module_name]
os.chdir(REPO_DIR)

if not SKIP_DEPENDENCY_INSTALL and PHASE != "figures":
    from graph_specialisation_metrics.carriage.env import install_dependencies

    install_dependencies(pyg_version="2.2.0")

from graph_specialisation_metrics.methodology.grit_causal_population import run

result = run(
    phase=PHASE,
    output_dir=str(OUTPUT_ROOT),
    tasks=TASKS,
    train_seed=TRAIN_SEED,
    checkpoints=CHECKPOINTS,
    task_overrides=TASK_OVERRIDES,
    accelerator=ACCELERATOR,
    force=FORCE,
    force_fresh_grit=FORCE_FRESH_GRIT,
    graphs_per_batch=GRAPHS_PER_BATCH,
    head_batch_size=HEAD_BATCH_SIZE,
    event_batch_size=EVENT_BATCH_SIZE,
    population_head_pairs=POPULATION_HEAD_PAIRS,
    population_minimum_pairs=POPULATION_MINIMUM_PAIRS,
    discovery_graphs=DISCOVERY_GRAPHS,
    causal_graphs=CAUSAL_GRAPHS,
    clean_ablation_graphs=CLEAN_ABLATION_GRAPHS,
    semantic_donor_graphs=SEMANTIC_DONOR_GRAPHS,
    sources_per_graph=SOURCES_PER_GRAPH,
    donors_per_source=DONORS_PER_SOURCE,
)
print(json.dumps(result, indent=2, default=str))

from IPython.display import Image, display

for task_name in TASKS:
    manifest_path = (
        OUTPUT_ROOT
        / task_name
        / f"seed_{TRAIN_SEED}"
        / "focused_causal_population_manifest.json"
    )
    if not manifest_path.is_file():
        print(f"{task_name}: no figure manifest (not estimable, or use PHASE='all').")
        continue
    manifest = json.loads(manifest_path.read_text())
    print(f"\n{task_name} matching balance:")
    print(json.dumps(manifest["matching_balance"], indent=2, default=str))
    for name, paths in manifest["figures"].items():
        png = next(path for path in paths if path.endswith(".png"))
        print(task_name, name, png)
        display(Image(filename=png))
# ============================ paste to here ============================
