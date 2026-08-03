"""Paste/run this cell in Colab for the six-model cached ZINC comparison.

This frontend reads canonical ``scores/raw.pt``, ``model.json``, and optional
``carriage/fields.pt`` artifacts. It never loads ZINC, a checkpoint, or a GRIT
model and never recomputes specialisation scores.
"""

# ============================ paste from here ============================
from __future__ import annotations

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
    run,
)

inventory = cache_inventory(CANONICAL_ROOTS, tasks=TASKS, train_seed=TRAIN_SEED)
inventory_rows = []
for task_record in inventory:
    if task_record["matches"]:
        for match in task_record["matches"]:
            inventory_rows.append(
                {
                    "task": task_record["task"],
                    "root": match["root"],
                    "score": match["score_exists"],
                    "model": match["model_exists"],
                    "carriage": match["carriage_exists"],
                }
            )
    else:
        inventory_rows.append(
            {
                "task": task_record["task"],
                "root": "not found",
                "score": False,
                "model": False,
                "carriage": False,
            }
        )

import pandas as pd
from IPython.display import Image, display

print("Canonical cache inventory")
display(pd.DataFrame(inventory_rows))
missing = [
    record["task"]
    for record in inventory
    if int(record["complete_score_locations"]) < 1
]
if missing:
    expected = "\n".join(
        str(root / task / f"seed_{TRAIN_SEED}" / "cache/scores/raw.pt")
        for task in missing
        for root in CANONICAL_ROOTS
    )
    raise FileNotFoundError(
        f"Missing canonical score+model artifacts for {missing}. Checked:\n{expected}"
    )

print("[scope] Cache-only: no dataset, checkpoint, model, forward pass, or score recomputation.")
print("[scope] All intervals are canonical cached one-checkpoint intervals, not seed uncertainty.")
result = run(
    CANONICAL_ROOTS,
    OUTPUT_DIR,
    tasks=TASKS,
    train_seed=TRAIN_SEED,
    require_carriage=REQUIRE_CARRIAGE,
)

print("Model summary")
display(pd.read_csv(OUTPUT_DIR / "model_summary.csv"))
print("Local/global and cross-architecture comparisons")
display(pd.read_csv(OUTPUT_DIR / "pairwise_comparisons.csv"))
for path in result["figures"]:
    if str(path).endswith(".png") and Path(path).is_file():
        display(Image(filename=path))
print(f"[saved] {OUTPUT_DIR}")
# ============================= paste to here =============================
