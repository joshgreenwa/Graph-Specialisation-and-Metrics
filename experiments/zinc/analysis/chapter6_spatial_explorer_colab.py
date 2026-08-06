"""Single-cell Colab frontend for the Chapter 6 spatial explorer.

Paste this file into one Colab cell.  The default run is cache-only: it finds
every available canonical score cache, creates the spatial tables and figures,
and rebuilds interaction figures only when their existing cache is present.

Set ``INTERACTION_PHASE = "all"`` only when a full 2x2 semantic/structural
interaction measurement is intended.  Missing attention, carriage, interaction,
or individual model artifacts disable only the corresponding outputs.
"""

# The backend import intentionally follows the Colab clone/install step.
# ruff: noqa: I001

# ============================ paste from here ============================
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from IPython.display import Image, display


# ------------------------------- repository ---------------------------------

REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPOSITORY_BRANCH = "expansion/carriage_experiments"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"  # Optional for a public repository.


# --------------------------- experiment controls -----------------------------

DRIVE_ROOT = Path("/content/drive/MyDrive")
CANONICAL_ROOTS = (
    DRIVE_ROOT / "graph_specialisation_metrics/canonical_methodology_v4_zinc_qm9",
    DRIVE_ROOT / "graph_specialisation_metrics/canonical_methodology",
)
OUTPUT_DIR = DRIVE_ROOT / "graph_specialisation_metrics/chapter6_spatial_explorer"

TASKS = (
    "zinc_1hop_localrrwp",
    "zinc_1hop",
    "zinc_1hop_vnode",
    "zinc_2hop",
    "zinc_2hop_vnode",
    "zinc",
)
TRAIN_SEED = 42
REPRESENTATIVE_ACTIVITY_QUANTILE = 0.25

# "off" does nothing. "figures" is cache-only. "all" measures the full 2x2
# semantic/structural interaction and then plots it. A missing figures cache is
# reported and skipped; it never blocks the spatial explorer.
INTERACTION_PHASE = "figures"  # "off", "figures", "measure", or "all"
INTERACTION_OUTPUT_DIR = (
    DRIVE_ROOT
    / "graph_specialisation_metrics/zinc_semantic_structural_interaction_carriage_v2"
)
INTERACTION_TASKS = "zinc_1hop,zinc_2hop,zinc_1hop_vnode,zinc"
INTERACTION_SEED = 0
INTERACTION_GRAPHS = 64
INTERACTION_SOURCES_PER_GRAPH = 6
INTERACTION_DONOR_PAIRS_PER_SOURCE = 4
INTERACTION_SEMANTIC_DONOR_GRAPHS = 256
INTERACTION_BOOTSTRAP_REPLICATES = 2_000
ACCELERATOR = "cuda:0"
NUM_THREADS = 4


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
    try:
        from google.colab import drive, userdata

        drive.mount("/content/drive", force_remount=False)
    except ImportError as error:
        raise RuntimeError("This launcher is intended for Google Colab") from error

    token = userdata.get(SECRET_NAME) or os.environ.get(SECRET_NAME)
    if token:
        suffix = REPOSITORY_URL.removeprefix("https://github.com/")
        remote = (
            f"https://x-access-token:{quote(str(token).strip(), safe='')}@github.com/"
            f"{suffix}"
        )
    else:
        print(
            f"[setup:warning] Colab secret {SECRET_NAME!r} is missing; using public clone",
            flush=True,
        )
        remote = REPOSITORY_URL

    if not (COLAB_REPOSITORY / ".git").is_dir():
        command(
            "git",
            "clone",
            "--branch",
            REPOSITORY_BRANCH,
            "--single-branch",
            remote,
            str(COLAB_REPOSITORY),
        )
    else:
        command(
            "git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", remote
        )
        command(
            "git", "-C", str(COLAB_REPOSITORY), "fetch", "origin", REPOSITORY_BRANCH
        )
        command(
            "git", "-C", str(COLAB_REPOSITORY), "checkout", REPOSITORY_BRANCH
        )
        command(
            "git",
            "-C",
            str(COLAB_REPOSITORY),
            "merge",
            "--ff-only",
            f"origin/{REPOSITORY_BRANCH}",
        )
    command(
        "git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", REPOSITORY_URL
    )
    command(sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPOSITORY))

    source_path = (COLAB_REPOSITORY / "src").resolve()
    source = str(source_path)
    sys.path[:] = [entry for entry in sys.path if entry != source]
    sys.path.insert(0, source)
    for module_name in tuple(sys.modules):
        if module_name == "graph_specialisation_metrics" or module_name.startswith(
            "graph_specialisation_metrics."
        ):
            del sys.modules[module_name]
    importlib.invalidate_caches()
    os.chdir(COLAB_REPOSITORY)

bootstrap()

from graph_specialisation_metrics.chapter6_spatial_explorer import inventory, run


print(
    "\n[scope] Core analysis is cache-only: no dataset, checkpoint, model, or forward pass.\n"
    "[scope] Semantic and structural distance profiles are read for every head.\n"
    "[scope] Profile overlap (1 - total variation) is the primary alignment measure; "
    "expected distance and peak agreement remain descriptive summaries.\n"
    "[scope] Spatial width is profile variance. Estimation uncertainty is separately "
    "computed across cached held-out graphs.\n"
    "[scope] Clean attention mass is compared with layerwise score distance whenever it "
    "is present in raw.pt; attention is not treated as a causal score.\n"
    "[scope] Final-state carriage is overlaid when fields.pt is present. It measures "
    "learned response, not task necessity.\n"
    "[scope] Cache protocol differences produce warnings only. Missing components skip "
    "their own table or panel rather than stopping the run.\n",
    flush=True,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
inventory_rows = inventory(CANONICAL_ROOTS, TASKS, seed=TRAIN_SEED)
print("Cache inventory", flush=True)
display(pd.DataFrame(inventory_rows))

result = run(
    CANONICAL_ROOTS,
    OUTPUT_DIR,
    tasks=TASKS,
    seed=TRAIN_SEED,
    activity_quantile=REPRESENTATIVE_ACTIVITY_QUANTILE,
    verbose=True,
)

print("\nArtifact availability", flush=True)
display(pd.DataFrame(result["model_artifacts"]))

head_table = pd.read_csv(OUTPUT_DIR / "head_spatial_metrics.csv")
head_table["expected_distance_gap"] = (
    head_table["structural_expected_distance"]
    - head_table["semantic_expected_distance"]
)
alignment_summary = (
    head_table.groupby("task", as_index=False)
    .agg(
        heads=("overlap", "count"),
        profile_overlap_median=("overlap", "median"),
        wasserstein_similarity_median=("wasserstein_similarity", "median"),
        expected_distance_gap_median=("expected_distance_gap", "median"),
    )
)
print("\nModel-level alignment summary", flush=True)
display(alignment_summary)

print("\nRepresentative high- and low-alignment heads", flush=True)
display(pd.read_csv(OUTPUT_DIR / "representative_heads.csv"))

layer_table = pd.read_csv(OUTPUT_DIR / "layer_spatial_summary.csv")
print("\nLayerwise expected distance", flush=True)
display(
    layer_table.loc[
        :,
        [
            "task",
            "layer",
            "source",
            "expected_distance_mean",
            "expected_distance_q1",
            "expected_distance_q3",
            "spatial_variance_mean",
            "estimation_sem_mean",
        ],
    ]
)

for path in result["figures"]:
    if str(path).endswith(".png") and Path(path).is_file():
        print(f"\n[display] {Path(path).stem}", flush=True)
        display(Image(filename=path))


# ------------------------- optional interaction stage -------------------------

if INTERACTION_PHASE != "off":
    if INTERACTION_PHASE not in {"figures", "measure", "all"}:
        raise ValueError("INTERACTION_PHASE must be 'off', 'figures', 'measure', or 'all'")
    print(
        "\n[interaction] Running the matched clean / semantic-only / structural-only / "
        f"joint stage in {INTERACTION_PHASE!r} mode.",
        flush=True,
    )
    try:
        from graph_specialisation_metrics.zinc_interaction_pilot import main as interaction_main

        interaction_args = [
            "--phase",
            INTERACTION_PHASE,
            "--output-dir",
            str(INTERACTION_OUTPUT_DIR),
            "--tasks",
            INTERACTION_TASKS,
            "--seed",
            str(INTERACTION_SEED),
            "--graphs",
            str(INTERACTION_GRAPHS),
            "--sources-per-graph",
            str(INTERACTION_SOURCES_PER_GRAPH),
            "--donor-pairs-per-source",
            str(INTERACTION_DONOR_PAIRS_PER_SOURCE),
            "--semantic-donor-graphs",
            str(INTERACTION_SEMANTIC_DONOR_GRAPHS),
            "--bootstrap-replicates",
            str(INTERACTION_BOOTSTRAP_REPLICATES),
            "--accelerator",
            ACCELERATOR,
            "--num-threads",
            str(NUM_THREADS),
        ]
        interaction_result = interaction_main(interaction_args)
        for name, paths in interaction_result.get("figures", {}).items():
            path = paths.get("png") if isinstance(paths, dict) else None
            if path and Path(path).is_file():
                print(f"\n[display:interaction] {name}", flush=True)
                display(Image(filename=path))
    except (OSError, RuntimeError, ValueError, KeyError, ImportError) as error:
        print(
            "[interaction:skip] "
            f"{type(error).__name__}: {error}\n"
            "Set INTERACTION_PHASE='all' to deliberately compute a missing cache. "
            "The completed spatial analysis is unaffected.",
            flush=True,
        )

print(f"\n[done] Chapter 6 exploration saved under {OUTPUT_DIR}", flush=True)
# ============================= paste to here =============================
