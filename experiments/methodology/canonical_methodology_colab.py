"""Paste/run this lightweight cell in Colab to complete canonical QM9 scores and carriage.

Edit only TASKS, TRAIN_SEEDS/TASK_TRAIN_SEEDS, PHASES, CHECKPOINTS, TASK_OVERRIDES,
OUTPUT_DIR, SIZES, FAMILIES, and EXECUTION. Scientific definitions live in
``src/graph_specialisation_metrics/README.md``
and the canonical package; this front end merely checks out the chosen repository revision,
mounts Drive, dispatches registered tasks, and verifies the completed consolidated caches.
"""

# ============================ paste from here ============================
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
BRANCH = "expansion/carriage_experiments"
REPO_DIR = "/content/Graph-Specialisation-and-Metrics"
SECRET_NAME = "dissertation_key"
OUTPUT_DIR = "/content/drive/MyDrive/graph_specialisation_metrics/canonical_methodology"

# Complete every registered QM9 attention/RRWP/VNode control under one scientific contract.
TASKS = (
    "qm9_gap_1hop",
    "qm9_gap_1hop_local",
    "qm9_gap_2hop",
    "qm9_gap_1hop_vnode",
    "qm9_gap_2hop_vnode",
    "qm9_gap_dense",
)
TRAIN_SEEDS = (42,)
TASK_TRAIN_SEEDS = {}
PHASES = ("scores", "carriage")
CHECKPOINTS = {}  # e.g. {"graphormer_zinc:42": "/content/drive/MyDrive/.../checkpoint.pt"}
TASK_OVERRIDES = {
    # Override only if a trained checkpoint directory differs from the registered default:
    # "qm9_gap_2hop": {
    #     "drive_dir": "/content/drive/MyDrive/your_actual_qm9_2hop_directory",
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
    # Runtime-only: an OOM automatically halves the failing group without duplicating results.
    "graphs_per_batch": 8,
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

results = run(
    tasks=TASKS,
    train_seeds=TRAIN_SEEDS,
    task_train_seeds={
        task: seeds for task, seeds in TASK_TRAIN_SEEDS.items() if task in TASKS
    },
    phases=PHASES,
    output_dir=OUTPUT_DIR,
    checkpoints=CHECKPOINTS,
    task_overrides=TASK_OVERRIDES,
    sizes=SIZES,
    families=FAMILIES,
    execution=EXECUTION,
    mount=False,
)
# On a runtime where GRIT/PyG dependencies are already installed, add skip_install=True.

# Fail closed unless every requested task has internally valid, complete 48-graph consolidated
# score and carriage caches under the requested seed/config/checkpoint contract.
from graph_specialisation_metrics.methodology.cache import (  # noqa: E402
    load_cache_artifact_file,
)

expected_graphs = int(SIZES["discovery_graphs"])
expected_channels = {"semantic", "structural"}
expected_runs = sum(
    len(TASK_TRAIN_SEEDS.get(task, TRAIN_SEEDS))
    for task in TASKS
)
assert len(results) == expected_runs, (len(results), expected_runs)

for task in TASKS:
    for seed in TASK_TRAIN_SEEDS.get(task, TRAIN_SEEDS):
        seed_root = Path(OUTPUT_DIR) / task / f"seed_{int(seed)}"
        artifacts = {
            "scores": load_cache_artifact_file(
                seed_root / "cache" / "scores" / "raw.pt"
            ),
            "carriage": load_cache_artifact_file(
                seed_root / "cache" / "carriage" / "fields.pt"
            ),
        }
        for stage, artifact in artifacts.items():
            contract = artifact.metadata["contract"]
            assert contract["task"] == task, (stage, contract["task"], task)
            assert int(contract["train_seed"]) == int(seed)
            assert int(contract["source_cap"]) == int(SIZES["sources_per_graph"])
            assert int(contract["donors_per_source"]) == int(
                SIZES["donors_per_source"]
            )
            assert int(contract["bootstrap_replicates"]) == int(
                SIZES["bootstrap_replicates"]
            )
            assert set(artifact.value["channels"]) == expected_channels

        for channel in sorted(expected_channels):
            score_graphs = len(
                artifacts["scores"].value["channels"][channel]["graph_scores"]
            )
            carriage_graphs = len(
                artifacts["carriage"].value["channels"][channel]["graph_fields"]
            )
            assert score_graphs == expected_graphs, (
                task,
                seed,
                channel,
                "scores",
                score_graphs,
            )
            assert carriage_graphs == expected_graphs, (
                task,
                seed,
                channel,
                "carriage",
                carriage_graphs,
            )

        score_contract = artifacts["scores"].metadata["contract"]
        carriage_contract = artifacts["carriage"].metadata["contract"]
        assert score_contract["protocol_fingerprint"] == carriage_contract[
            "protocol_fingerprint"
        ]
        assert score_contract["checkpoint_sha256"] == carriage_contract[
            "checkpoint_sha256"
        ]
        partials = list(seed_root.glob("cache/**/*.partial"))
        assert not partials, partials
        print(
            f"[complete] {task}:seed{seed} "
            f"scores={expected_graphs}/{expected_graphs}, "
            f"carriage={expected_graphs}/{expected_graphs}",
            flush=True,
        )

print(f"[complete] Validated all {expected_runs} QM9 canonical runs.", flush=True)
# ============================ paste to here ============================
