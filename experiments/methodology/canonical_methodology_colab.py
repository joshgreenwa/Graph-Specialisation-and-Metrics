"""Paste/run this lightweight cell in Colab to execute the final methodology.

Edit only TASKS, TRAIN_SEEDS/TASK_TRAIN_SEEDS, PHASES, CHECKPOINTS, TASK_OVERRIDES,
SIZES, FAMILIES, and EXECUTION. Scientific definitions live in
``src/graph_specialisation_metrics/README.md``
and the canonical package; this front end merely checks out the chosen repository revision,
mounts Drive, and dispatches registered tasks.
"""

# ============================ paste from here ============================
import os
import subprocess
import sys

REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
BRANCH = "main"
REPO_DIR = "/content/Graph-Specialisation-and-Metrics"
SECRET_NAME = "dissertation_key"

# First production run: all registered dense task families.
TASKS = ("zinc", "qm9_gap_dense", "peptides_func", "peptides_struct")
# Receptive-field control suite (uncomment instead when building missing score caches):
# TASKS = (
#     "zinc_1hop",
#     "zinc_1hop_vnode",
#     "zinc_2hop",
#     "zinc_2hop_vnode",
#     "qm9_gap_1hop",
#     "qm9_gap_1hop_vnode",
# )
# Their registered task specifications already point at the corresponding Drive checkpoint roots;
# leave CHECKPOINTS empty unless overriding one exact checkpoint file.
TRAIN_SEEDS = (42,)
# A public checkpoint has no training-seed ensemble; use a stable seed label for its cache.
# Example: TASKS = ("graphormer_pcqm4mv2",)
TASK_TRAIN_SEEDS = {"graphormer_pcqm4mv2": (0,)}
PHASES = ("scores", "causal", "carriage", "figures")
CHECKPOINTS = {}  # e.g. {"graphormer_zinc:42": "/content/drive/MyDrive/.../checkpoint.pt"}
TASK_OVERRIDES = {
    # Optional PCQM cache location or Hugging Face cache/offline controls:
    # "graphormer_pcqm4mv2": {
    #     "dataset_root": "/content/drive/MyDrive/datasets/pcqm4mv2",
    #     "cache_dir": "/content/drive/MyDrive/huggingface",
    #     "local_files_only": False,
    # },
}
SIZES = {
    "discovery_graphs": 48,
    "causal_graphs": 24,
    "clean_ablation_graphs": 64,
    "semantic_donor_graphs": 2_000,
    "sources_per_graph": 6,
    "donors_per_source": 8,
    "bootstrap_replicates": 2_000,  # fixed by the normative protocol
}
# Preregister these margins before looking at causal outcomes.  D_rel and the reference-scaled
# family-by-channel causal interaction live on different scales, so they have separate regions.
# Equivalence requires the complete nested-bootstrap interval to fit inside the region.
FAMILIES = {
    "activity_floor": 0.20,
    "tail_fraction": 0.20,
    "central_fraction": 0.20,
    "central_pool_fraction": 0.50,
    "equivalence_half_width": 0.10,
    "causal_equivalence_half_width": 0.20,
    "membership_stability_floor": 0.60,
    "generalist_fraction_floor": 0.50,
    "importance_correlation_floor": 0.10,
    "causal_response_floor": 0.10,
}
EXECUTION = {
    # Runtime-only: increase on large GPUs; CUDA OOM automatically retries smaller groups.
    "graphs_per_batch": 4,
    "oom_backoff": True,
}

from google.colab import drive, userdata  # noqa: E402

drive.mount("/content/drive", force_remount=False)
token = userdata.get(SECRET_NAME)
if not token:
    raise RuntimeError(f"Colab secret {SECRET_NAME!r} is missing or empty")
suffix = REPO_URL.removeprefix("https://github.com/")
authenticated = f"https://x-access-token:{token.strip()}@github.com/{suffix}"
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", authenticated, REPO_DIR],
        check=True,
    )
else:
    subprocess.run(
        ["git", "-C", REPO_DIR, "remote", "set-url", "origin", authenticated], check=True
    )
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(
        ["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"], check=True
    )
subprocess.run(
    ["git", "-C", REPO_DIR, "remote", "set-url", "origin", REPO_URL], check=True
)

source = os.path.join(REPO_DIR, "src")
if source not in sys.path:
    sys.path.insert(0, source)
for module in [
    name for name in tuple(sys.modules) if name.startswith("graph_specialisation_metrics")
]:
    del sys.modules[module]

from graph_specialisation_metrics.methodology.colab import run  # noqa: E402

run(
    tasks=TASKS,
    train_seeds=TRAIN_SEEDS,
    task_train_seeds={
        task: seeds for task, seeds in TASK_TRAIN_SEEDS.items() if task in TASKS
    },
    phases=PHASES,
    checkpoints=CHECKPOINTS,
    task_overrides=TASK_OVERRIDES,
    sizes=SIZES,
    families=FAMILIES,
    execution=EXECUTION,
    mount=False,
)
# On a runtime where GRIT/PyG dependencies are already installed, add skip_install=True.
# ============================ paste to here ============================
