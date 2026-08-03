"""Standalone Colab frontend for QM9 semantic--structural output modulation.

Paste this complete file into one Colab cell. It compares the seed-0 1-hop,
1-hop+virtual-node, and dense GRIT checkpoints for the QM9 HOMO--LUMO gap
target using ordinary batched predictions only. Every clean, semantic-only,
structural-only, and joint endpoint is cached to Drive.

After a completed run, set ``PHASE = "output-figures"`` to rebuild summaries
and figures from cached endpoints without loading a model.
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

PHASE = "output-all"  # "output-all", "output-measure", or "output-figures"
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/graph_specialisation_metrics/"
    "qm9_semantic_structural_output_modulation_v1"
)
OUTPUT_TASKS = "qm9_gap_1hop,qm9_gap_1hop_vnode,qm9_gap_dense"
SEED = 0
OUTPUT_GRAPHS = 128
OUTPUT_SOURCES_PER_GRAPH = 6
OUTPUT_DONOR_PAIRS_PER_SOURCE = 2
OUTPUT_SEMANTIC_DONOR_GRAPHS = 64
OUTPUT_EFFECT_FLOOR = 1.0e-6
OUTPUT_GRAPHS_PER_BATCH = 16
BOOTSTRAP_REPLICATES = 2_000
ANALYSIS_SEED = 260_803
ACCELERATOR = "cuda:0"
NUM_THREADS = 4

# Set False only when this runtime already has the canonical GRIT/PyG stack.
INSTALL_DEPENDENCIES = True


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
    command("git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", REPOSITORY_URL)
    command(sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPOSITORY))

    source_path = (COLAB_REPOSITORY / "src").resolve()
    backend_path = source_path / "graph_specialisation_metrics" / "zinc_interaction_pilot.py"
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

from graph_specialisation_metrics.zinc_interaction_pilot import main

print(
    "\n[scope] This phase uses predictions only: no Jacobians or layer hooks.\n"
    "[scope] Each source uses matched clean, semantic-only, structural-only, and joint "
    "endpoints.\n"
    "[scope] M measures how much the semantic output effect changes after a structural "
    "swap; it is bounded from 0 to 2.\n"
    "[scope] Graph, source, and donor identities are sampled under QM9 1-hop and replayed "
    "exactly for 1-hop+VN and dense.\n"
    "[scope] M describes model response, not task necessity. Uncertainty bootstraps graphs "
    "from one checkpoint per architecture.\n",
    flush=True,
)

CELL_ARGS = [
    "--phase",
    PHASE,
    "--output-dir",
    str(OUTPUT_DIR),
    "--output-tasks",
    OUTPUT_TASKS,
    "--seed",
    str(SEED),
    "--output-graphs",
    str(OUTPUT_GRAPHS),
    "--output-sources-per-graph",
    str(OUTPUT_SOURCES_PER_GRAPH),
    "--output-donor-pairs-per-source",
    str(OUTPUT_DONOR_PAIRS_PER_SOURCE),
    "--output-semantic-donor-graphs",
    str(OUTPUT_SEMANTIC_DONOR_GRAPHS),
    "--output-effect-floor",
    str(OUTPUT_EFFECT_FLOOR),
    "--output-graphs-per-batch",
    str(OUTPUT_GRAPHS_PER_BATCH),
    "--bootstrap-replicates",
    str(BOOTSTRAP_REPLICATES),
    "--analysis-seed",
    str(ANALYSIS_SEED),
    "--accelerator",
    ACCELERATOR,
    "--num-threads",
    str(NUM_THREADS),
]
if not INSTALL_DEPENDENCIES:
    CELL_ARGS.append("--skip-dependency-install")

result = main(CELL_ARGS)

try:
    import pandas as pd
    from IPython.display import Image, display

    cache_value = result.get("output_modulation_cache_dir")
    if cache_value:
        cache_dir = Path(cache_value)
        health_path = cache_dir / "model_health.csv"
        if health_path.is_file():
            print("\nQM9 model health", flush=True)
            display(pd.read_csv(health_path))
        summary_path = cache_dir / "output_modulation_summary.csv"
        if summary_path.is_file():
            print("\nQM9 output-M summary", flush=True)
            display(pd.read_csv(summary_path))
        contrasts_path = cache_dir / "output_modulation_paired_contrasts.csv"
        if contrasts_path.is_file():
            contrasts = pd.read_csv(contrasts_path)
            print("\nPaired M contrasts", flush=True)
            display(contrasts[contrasts["metric"] == "modulation_m"])

    figure = result.get("output_modulation_figures", {}).get("output_modulation", {}).get("png")
    if figure:
        print(f"\n[display] output_modulation: {figure}", flush=True)
        display(Image(filename=figure))
except ImportError:
    pass

print(f"\n[done] QM9 output-modulation analysis saved under {OUTPUT_DIR}", flush=True)
