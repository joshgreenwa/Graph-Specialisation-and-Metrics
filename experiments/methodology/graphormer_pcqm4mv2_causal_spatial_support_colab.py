"""PCQM4Mv2 Colab frontend for causal spatial support of Graphormer heads.

The immutable canonical score cache supplies head families and the internal
response profile ``S``. Only held-out source-conditioned attention ``A`` and
finite shell mediation ``M`` require model forwards; each graph/channel shard is
cached independently on Drive. Set ``PHASE='figures'`` to redraw from the final
supplemental cache without loading Graphormer or PCQM4Mv2.
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
REPOSITORY_BRANCH = "expansion/graphormer_causal_spatial_support"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"

# ----------------------------- experiment controls -----------------------------

PHASE = "all"  # "all", "measure", or "figures"
PILOT = True  # 4-graph engineering run; set False for the registered 8-graph run.
DRIVE_ROOT = Path("/content/drive/MyDrive")
CANONICAL_ROOT = DRIVE_ROOT / "graph_specialisation_metrics/canonical_methodology"
PCQM_DATASET_ROOT = DRIVE_ROOT / "graph_specialisation_metrics/cache/pcqm4mv2"
HF_CACHE_DIR = DRIVE_ROOT / "graph_specialisation_metrics/cache/huggingface"
OUTPUT_DIR = (
    DRIVE_ROOT / "graph_specialisation_metrics/graphormer_pcqm4mv2_causal_spatial_support_v1"
)
TASK = "graphormer_pcqm4mv2"
TRAIN_SEED = 0
GRAPHS = 4 if PILOT else 8
SOURCES_PER_GRAPH = 1 if PILOT else 2
DONORS_PER_SOURCE = 1
HEADS_PER_FAMILY = 2 if PILOT else 3
LONG_RANGE_RADIUS = 2
BOOTSTRAP_REPLICATES = 1_000 if PILOT else 2_000
ANALYSIS_SEED = 72_019
ACCELERATOR = "cuda:0"
PATCH_BATCH_SIZE = 32  # Reduce if the Colab GPU runs out of memory.
LOCAL_FILES_ONLY = False
FORCE = False


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
        command(
            "git",
            "-C",
            str(COLAB_REPOSITORY),
            "remote",
            "set-url",
            "origin",
            authenticated,
        )
        command(
            "git",
            "-C",
            str(COLAB_REPOSITORY),
            "fetch",
            "origin",
            REPOSITORY_BRANCH,
        )
        command(
            "git",
            "-C",
            str(COLAB_REPOSITORY),
            "checkout",
            REPOSITORY_BRANCH,
        )
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
    command(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-e",
        f"{COLAB_REPOSITORY}[graphormer]",
    )
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

score_cache = CANONICAL_ROOT / TASK / f"seed_{TRAIN_SEED}" / "cache/scores/raw.pt"
model_record = CANONICAL_ROOT / TASK / f"seed_{TRAIN_SEED}" / "model.json"
if not score_cache.is_file() or not model_record.is_file():
    raise FileNotFoundError(
        "The canonical Graphormer score cache/model record was not found at "
        f"{score_cache.parent.parent.parent}"
    )

from graph_specialisation_metrics.methodology.causal_spatial_support import main

print(
    f"\n[preset] {'pilot' if PILOT else 'full'} Graphormer PCQM4Mv2: "
    f"graphs={GRAPHS}; sources/graph={SOURCES_PER_GRAPH}; "
    f"donors/source={DONORS_PER_SOURCE}; heads/family={HEADS_PER_FAMILY}\n"
    "[reuse] Frozen families and S(distance) are read from scores/raw.pt.\n"
    "[new] A(distance) and symmetric shell-specific M(distance) are measured "
    "on the held-out causal split.\n"
    "[carrier] Molecular nodes use pristine SPD; the graph token is retained "
    "as a separate non-SPD carrier.\n"
    "[cache] Each graph/channel shard is resumable. PHASE='figures' performs no "
    "model or dataset load.\n",
    flush=True,
)

args = [
    "--canonical-root",
    str(CANONICAL_ROOT),
    "--output-dir",
    str(OUTPUT_DIR),
    "--tasks",
    TASK,
    "--train-seed",
    str(TRAIN_SEED),
    "--phase",
    PHASE,
    "--graphs",
    str(GRAPHS),
    "--sources-per-graph",
    str(SOURCES_PER_GRAPH),
    "--donors-per-source",
    str(DONORS_PER_SOURCE),
    "--heads-per-family",
    str(HEADS_PER_FAMILY),
    "--long-range-radius",
    str(LONG_RANGE_RADIUS),
    "--bootstrap-replicates",
    str(BOOTSTRAP_REPLICATES),
    "--analysis-seed",
    str(ANALYSIS_SEED),
    "--accelerator",
    ACCELERATOR,
    "--patch-batch-size",
    str(PATCH_BATCH_SIZE),
    "--dataset-root",
    str(PCQM_DATASET_ROOT),
    "--model-cache-dir",
    str(HF_CACHE_DIR),
]
if LOCAL_FILES_ONLY:
    args.append("--local-files-only")
if FORCE:
    args.append("--force")
result = main(args)

from IPython.display import Image, Markdown, display

for key in ("headline_png", "overlap_png"):
    path = result.get("figures", {}).get(key)
    if path and Path(path).is_file():
        display(Markdown(f"### {key.removesuffix('_png').replace('_', ' ').title()}"))
        display(Image(filename=path))

try:
    import pandas as pd

    table = OUTPUT_DIR / "results/causal_spatial_support_statistics.csv"
    if table.is_file():
        display(pd.read_csv(table))
except Exception as error:  # noqa: BLE001 - display failure must not hide saved outputs
    print(f"[display:warning] {type(error).__name__}: {error}")

print(f"[saved] {OUTPUT_DIR}")
