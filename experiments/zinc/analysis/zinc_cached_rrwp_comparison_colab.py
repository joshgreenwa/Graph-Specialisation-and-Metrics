"""Paste/run this cell in Colab for the six-model cached ZINC comparison.

This frontend reads canonical ``scores/raw.pt``, ``model.json``, and optional
``carriage/fields.pt`` artifacts. It never loads ZINC, a checkpoint, or a GRIT
model and never recomputes specialisation scores.
"""

# ============================ paste from here ============================
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
BRANCH = "expansion/carriage_experiments"
REPO_DIR = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"

DRIVE_ROOT = Path("/content/drive/MyDrive")
CANONICAL_ROOTS = (
    DRIVE_ROOT / "graph_specialisation_metrics/canonical_methodology_v4_zinc_qm9",
    DRIVE_ROOT / "graph_specialisation_metrics/canonical_methodology",
)
OUTPUT_DIR = DRIVE_ROOT / "graph_specialisation_metrics/zinc_cached_rrwp_comparison"
TRAIN_SEED = 42
REQUIRE_CARRIAGE = False

TASKS = (
    "zinc_1hop_localrrwp",
    "zinc_1hop",
    "zinc_1hop_vnode",
    "zinc_2hop",
    "zinc_2hop_vnode",
    "zinc",
)
TRAINING_DRIVE_ROOTS = {
    "zinc_1hop_localrrwp": DRIVE_ROOT / "grit_zinc_1hop_localrrwp",
    "zinc_1hop": DRIVE_ROOT / "grit_zinc_1hop",
    "zinc_1hop_vnode": DRIVE_ROOT / "grit_zinc_1hop_vnode",
    "zinc_2hop": DRIVE_ROOT / "grit_zinc_2hop",
    "zinc_2hop_vnode": DRIVE_ROOT / "grit_zinc_2hop_vnode",
    "zinc": DRIVE_ROOT / "grit_zinc_official",
}
KNOWN_RECOVERY_CHECKPOINTS = {
    "zinc_1hop_localrrwp": (
        TRAINING_DRIVE_ROOTS["zinc_1hop_localrrwp"]
        / "results/_recovery_checkpoints/seed0_ColabDrive.1hopLocalRRWP.GRITwRRWP"
    ),
    "zinc_1hop": (
        TRAINING_DRIVE_ROOTS["zinc_1hop"]
        / "results/_recovery_checkpoints/seed0_ColabDrive.1hop.GRITwRRWP"
    ),
    "zinc_1hop_vnode": (
        TRAINING_DRIVE_ROOTS["zinc_1hop_vnode"]
        / "results/_recovery_checkpoints/seed0_ColabDrive.1hop.GRITwRRWP.VNode"
    ),
    "zinc_2hop": (
        TRAINING_DRIVE_ROOTS["zinc_2hop"]
        / "results/_recovery_checkpoints/seed0_ColabDrive.2hop.GRITwRRWP"
    ),
    "zinc_2hop_vnode": (
        TRAINING_DRIVE_ROOTS["zinc_2hop_vnode"]
        / "results/_recovery_checkpoints/seed0_ColabDrive.2hop.GRITwRRWP.VNode"
    ),
}

from google.colab import drive, userdata

drive.mount("/content/drive", force_remount=False)
token = userdata.get(SECRET_NAME)
if token:
    suffix = REPO_URL.removeprefix("https://github.com/")
    authenticated = (
        f"https://x-access-token:{quote(str(token).strip(), safe='')}@github.com/{suffix}"
    )
else:
    print(f"[setup:warning] Colab secret {SECRET_NAME!r} is missing; using public clone")
    authenticated = REPO_URL

if not (REPO_DIR / ".git").is_dir():
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", authenticated, str(REPO_DIR)],
        check=True,
    )
else:
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "remote", "set-url", "origin", authenticated],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "checkout", BRANCH],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "merge", "--ff-only", f"origin/{BRANCH}"],
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
sys.path[:] = [entry for entry in sys.path if entry != source]
sys.path.insert(0, source)
for module_name in tuple(sys.modules):
    if module_name == "graph_specialisation_metrics" or module_name.startswith(
        "graph_specialisation_metrics."
    ):
        del sys.modules[module_name]
os.chdir(REPO_DIR)

from graph_specialisation_metrics.zinc_cached_rrwp_comparison import (
    cache_inventory,
    load_compatible_score_artifact,
    run,
)

inventory = cache_inventory(CANONICAL_ROOTS, tasks=TASKS, train_seed=TRAIN_SEED)


def missing_tasks(records):
    return [
        record["task"]
        for record in records
        if int(record["complete_score_locations"]) < 1
    ]


inventory_rows = []
protocol_rows = []
for task_record in inventory:
    if task_record["matches"]:
        for match in task_record["matches"]:
            inventory_rows.append(
                {
                    "task": task_record["task"],
                    "artifact_task": match["artifact_task"],
                    "root": match["root"],
                    "score": match["score_exists"],
                    "model": match["model_exists"],
                    "carriage": match["carriage_exists"],
                }
            )
            if match["score_exists"] and match["model_exists"]:
                artifact = load_compatible_score_artifact(
                    match["score"], expected_task=match["artifact_task"]
                )
                contract = artifact.metadata["contract"]
                protocol_rows.append(
                    {
                        "task": task_record["task"],
                        "artifact_task": match["artifact_task"],
                        "protocol": artifact.metadata["protocol_version"],
                        "semantic_donor_law": contract.get(
                            "semantic_donor_law", "unknown"
                        ),
                        "structural_donor_law": contract.get(
                            "structural_donor_law", "unknown"
                        ),
                        "score_cache": match["score"],
                    }
                )
    else:
        inventory_rows.append(
            {
                "task": task_record["task"],
                "artifact_task": "not found",
                "root": "not found",
                "score": False,
                "model": False,
                "carriage": False,
            }
        )

for task in TASKS:
    task_rows = [row for row in protocol_rows if row["task"] == task]
    has_v4 = any(
        row["protocol"] == "donor-swap-specialisation-carriage-v4"
        for row in task_rows
    )
    for row in task_rows:
        row["preferred"] = (
            row["protocol"] == "donor-swap-specialisation-carriage-v4"
            if has_v4
            else row["protocol"] == "donor-swap-specialisation-carriage-v3"
        )


def checkpoint_from_canonical_record(task_record):
    for match in task_record["matches"]:
        if not match["model_exists"]:
            continue
        model_record = json.loads(Path(match["model"]).read_text(encoding="utf-8"))
        checkpoint = model_record.get("checkpoint")
        if checkpoint:
            return Path(str(checkpoint)), model_record.get("checkpoint_epoch")
    return None, None


checkpoint_rows = []
for task_record in inventory:
    task = task_record["task"]
    checkpoint, checkpoint_epoch = checkpoint_from_canonical_record(task_record)
    source_kind = "canonical model.json"
    if checkpoint is None and task in KNOWN_RECOVERY_CHECKPOINTS:
        recovery_root = KNOWN_RECOVERY_CHECKPOINTS[task]
        candidates = tuple(
            recovery_root / name
            for name in ("best.ckpt", "latest.ckpt", "first_after_resume.ckpt")
        )
        checkpoint = next((path for path in candidates if path.is_file()), candidates[0])
        checkpoint_epoch = -1
        source_kind = "training runner stable path"
    checkpoint_rows.append(
        {
            "task": task,
            "training_drive_root": str(TRAINING_DRIVE_ROOTS[task]),
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_exists": bool(checkpoint is not None and checkpoint.is_file()),
            "path_source": source_kind if checkpoint is not None else "not recorded",
        }
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_manifest = OUTPUT_DIR / "checkpoint_paths.json"
checkpoint_manifest.write_text(
    json.dumps(checkpoint_rows, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
protocol_manifest = OUTPUT_DIR / "cache_protocol_inventory.json"
protocol_manifest.write_text(
    json.dumps(protocol_rows, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

import pandas as pd
from IPython.display import Image, display

print("Exact training checkpoint inventory (no recursive Drive search; checkpoints not loaded)")
display(pd.DataFrame(checkpoint_rows))
print(f"[checkpoint-manifest] {checkpoint_manifest}")
print("Canonical cache inventory")
display(pd.DataFrame(inventory_rows))
print("Canonical score protocol inventory")
display(pd.DataFrame(protocol_rows))
print(f"[protocol-manifest] {protocol_manifest}")
for protocol in sorted({row["protocol"] for row in protocol_rows}):
    protocol_tasks = [
        task
        for task in TASKS
        if any(
            row["task"] == task and row["protocol"] == protocol
            for row in protocol_rows
        )
    ]
    print(f"[{protocol.rsplit('-', 1)[-1]} available] " + ", ".join(protocol_tasks))
v4_tasks = {
    row["task"]
    for row in protocol_rows
    if row["protocol"] == "donor-swap-specialisation-carriage-v4"
}
print("[v4 selected by default] " + ", ".join(task for task in TASKS if task in v4_tasks))
recompute_v4 = [task for task in TASKS if task not in v4_tasks]
if recompute_v4:
    print("[recompute-v4] " + ", ".join(recompute_v4))
missing = missing_tasks(inventory)
if missing:
    missing_checkpoint_state = [
        row for row in checkpoint_rows if row["task"] in missing
    ]
    print(
        "The rows below distinguish an available trained checkpoint from a missing "
        "canonical specialisation-score cache. A checkpoint alone is not a raw.pt score artifact."
    )
    display(pd.DataFrame(missing_checkpoint_state))
    print(
        "[skip] Missing canonical score caches; these models will not be included: "
        + ", ".join(missing)
    )

available_tasks = tuple(
    record["task"]
    for record in inventory
    if int(record["complete_score_locations"]) >= 1
)
if not available_tasks:
    raise FileNotFoundError(
        "No complete canonical score+model artifacts were found for any requested ZINC model."
    )
print("[run] Models with validated canonical scores: " + ", ".join(available_tasks))


print("[scope] Cache-only: no dataset, checkpoint, model, forward pass, or score recomputation.")
print("[scope] All intervals are canonical cached one-checkpoint intervals, not seed uncertainty.")
print(
    "[scope] Layerwise head co-location uses cached exact per-head score-distance tensors; "
    "medians, IQRs, ranges, and descriptive low-overlap outliers are fingerprint-cached."
)
print(f"[scope] Output directory: {OUTPUT_DIR}")
try:
    result = run(
        CANONICAL_ROOTS,
        OUTPUT_DIR,
        tasks=available_tasks,
        train_seed=TRAIN_SEED,
        require_carriage=REQUIRE_CARRIAGE,
        verbose=True,
    )
except Exception as error:
    print(
        "[analysis-failed] "
        f"{type(error).__name__}: {error}\n"
        f"[analysis-failed] tasks={list(available_tasks)}\n"
        f"[analysis-failed] roots={[str(root) for root in CANONICAL_ROOTS]}\n"
        f"[analysis-failed] output={OUTPUT_DIR}"
    )
    raise

compatibility = result["cache_compatibility"]
print("Cache compatibility")
display(pd.DataFrame([compatibility]))
if not compatibility["same_structural_donor_law"]:
    print(
        "[compatibility-warning] Score formulas match, but stored structural donor laws differ. "
        "Treat small cross-protocol structural-score differences as exploratory; per-model "
        "protocol and donor-law columns are retained in the tables."
    )

print("Model summary")
display(pd.read_csv(OUTPUT_DIR / "model_summary.csv"))
print("Local/global and cross-architecture comparisons")
comparison_table = OUTPUT_DIR / "pairwise_comparisons.csv"
if comparison_table.stat().st_size:
    display(pd.read_csv(comparison_table))
else:
    print("[skip] Pairwise comparisons require at least two available models.")
alignment_result = result["head_profile_alignment"]
print(
    "Head-profile co-location cache: "
    f"{alignment_result['cache_status']} ({alignment_result['cache']})"
)
print("Layerwise within-head semantic–structural profile overlap")
alignment_table = pd.read_csv(OUTPUT_DIR / "head_profile_alignment_layerwise.csv")
display(
    alignment_table.loc[
        :,
        [
            "task",
            "profile_kind",
            "layer",
            "valid_heads",
            "overlap_median",
            "overlap_iqr",
            "overlap_min",
            "overlap_min_head",
            "low_overlap_outlier_count",
            "low_overlap_outlier_heads",
        ],
    ]
)
outlier_table = OUTPUT_DIR / "head_profile_alignment_outliers.csv"
if outlier_table.stat().st_size:
    print("Descriptive within-layer low-overlap outlier heads")
    display(pd.read_csv(outlier_table))
else:
    print("[outliers] No heads fall below their layer's Tukey low-overlap fence.")
for path in result["figures"]:
    if str(path).endswith(".png") and Path(path).is_file():
        display(Image(filename=path))
print(f"[saved] {OUTPUT_DIR}")
# ============================= paste to here =============================
