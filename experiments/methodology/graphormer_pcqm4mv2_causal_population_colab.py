"""Paste/run this cell in Colab for the PCQM4Mv2 causal population analysis.

The analysis reuses the focused score, causal-event, and all-head clean-
ablation shards already stored under OUTPUT_ROOT.  Only selected heads absent
from the population cache are measured.
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
# Keep the existing focused-causal root: compatible graph/head shards are read through.
OUTPUT_ROOT = DRIVE_ROOT / "graph_specialisation_metrics/graphormer_pcqm4mv2_causal"
PCQM_DATASET_ROOT = DRIVE_ROOT / "graph_specialisation_metrics/cache/pcqm4mv2"
HF_CACHE_DIR = DRIVE_ROOT / "graph_specialisation_metrics/cache/huggingface"

PHASE = "all"  # "run", "figures", or "all"
FORCE = False
ACCELERATOR = "cuda:0"
GRAPHS_PER_BATCH = 2
HEAD_BATCH_SIZE = 4
EVENT_BATCH_SIZE = 8

# Primary paper population: 12 J-matched semantic/structural pairs (24 heads)
# plus one distinct J-matched null for every selected specialist (24 controls).
POPULATION_HEAD_PAIRS = 12
POPULATION_MINIMUM_PAIRS = 8

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
    [sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO_DIR}[graphormer]"],
    check=True,
)
for module_name in tuple(sys.modules):
    if module_name == "graph_specialisation_metrics" or module_name.startswith(
        "graph_specialisation_metrics."
    ):
        del sys.modules[module_name]
source = str(REPO_DIR / "src")
if source not in sys.path:
    sys.path.insert(0, source)
os.chdir(REPO_DIR)

from graph_specialisation_metrics.methodology.graphormer_causal_population import (
    run,
)

result = run(
    phase=PHASE,
    output_dir=str(OUTPUT_ROOT),
    dataset_root=str(PCQM_DATASET_ROOT),
    cache_dir=str(HF_CACHE_DIR),
    accelerator=ACCELERATOR,
    force=FORCE,
    graphs_per_batch=GRAPHS_PER_BATCH,
    head_batch_size=HEAD_BATCH_SIZE,
    event_batch_size=EVENT_BATCH_SIZE,
    population_head_pairs=POPULATION_HEAD_PAIRS,
    population_minimum_pairs=POPULATION_MINIMUM_PAIRS,
)
print(json.dumps(result, indent=2, default=str))

seed_root = OUTPUT_ROOT / "graphormer_pcqm4mv2/seed_0"
manifest_path = seed_root / "focused_causal_population_manifest.json"
if manifest_path.is_file():
    from IPython.display import Image, display

    manifest = json.loads(manifest_path.read_text())
    for name, paths in manifest["figures"].items():
        png = next(path for path in paths if path.endswith(".png"))
        print(name, png)
        display(Image(filename=png))
    print("Matching balance:")
    print(json.dumps(manifest["matching_balance"], indent=2, default=str))
else:
    print("No figure manifest yet; use PHASE='all' or PHASE='figures'.")
# ============================ paste to here ============================
