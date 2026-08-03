"""Standalone Colab frontend for ZINC interaction carriage and output modulation.

The default lightweight phase compares local-RRWP 1-hop, global-RRWP 1-hop,
1-hop+VN, 2-hop, 2-hop+VN, and dense GRIT using exact scalar-output endpoints
only. It caches every clean, semantic-only, structural-only, and joint
prediction to Drive. The original full carriage phases remain available below.

After a completed lightweight run, set ``PHASE = "output-figures"`` to rebuild
all M tables and figures from cached CSV files without loading the models.
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
    "zinc_semantic_structural_interaction_carriage_v2"
)
TASKS = "zinc_1hop,zinc_2hop,zinc_1hop_vnode,zinc"
SEED = 0
GRAPHS = 64
SOURCES_PER_GRAPH = 6
DONOR_PAIRS_PER_SOURCE = 4
SEMANTIC_DONOR_GRAPHS = 256
ABSOLUTE_EFFECT_FLOOR = 1.0e-6
RELATIVE_EFFECT_FLOOR = 1.0e-3
FAR_DISTANCE = 4
BOOTSTRAP_REPLICATES = 2_000
ANALYSIS_SEED = 260_803
ACCELERATOR = "cuda:0"
NUM_THREADS = 4
DISPLAY_MAX_DISTANCE = 0  # 0 keeps the complete observed distance axis.

# Lightweight exact-output M pilot. These controls do not run carriage or Jacobians.
OUTPUT_TASKS = "zinc_1hop_localrrwp,zinc_1hop,zinc_1hop_vnode,zinc_2hop,zinc_2hop_vnode,zinc"
OUTPUT_GRAPHS = 128
OUTPUT_SOURCES_PER_GRAPH = 6
OUTPUT_DONOR_PAIRS_PER_SOURCE = 2
OUTPUT_SEMANTIC_DONOR_GRAPHS = 64
OUTPUT_EFFECT_FLOOR = 1.0e-6
OUTPUT_GRAPHS_PER_BATCH = 8

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
    "\n[scope] The output-M phase uses predictions only: no Jacobians or layer hooks.\n"
    "[scope] Each source uses matched clean, semantic-only, structural-only, and "
    "joint donor endpoints.\n"
    "[scope] M asks how much the semantic output effect changes after the structural "
    "context is swapped; it is bounded from 0 to 2.\n"
    "[scope] Graph, source, and donor identities are sampled once under local RRWP "
    "and replayed exactly for all sparse, virtual-node, and dense models.\n"
    "[scope] M measures model response, not task necessity. All four signed endpoint "
    "predictions are cached for alternative summaries without model reruns.\n"
    "[scope] The legacy carriage phases retain the following interpretation.\n"
    "[scope] Semantic and structural carriage are clean-condition marginals; "
    "interaction carriage is the signed 2x2 contrast before any norm is taken.\n"
    "[scope] Final-state profiles show aggregate functional allocation. Layer profiles "
    "apply the identical contrasts at native head-transport sites and the same clean "
    "output Jacobians used by the specialisation scores.\n"
    "[scope] Layers are diagnostic sites, not additive causal stages. The virtual "
    "carrier is retained as a separate VN column rather than assigned a graph distance.\n"
    "[scope] Interaction distance is shown only above absolute and marginal-relative "
    "estimability floors. Raw interaction strength and the estimable fraction are "
    "reported alongside it.\n"
    "[scope] Dense versus sparse comparisons describe trained model organisation. "
    "Near-equal test MAE supports architectural sufficiency, not task necessity.\n"
    "[scope] Uncertainty is a held-out-graph bootstrap from one checkpoint per "
    "architecture; it does not include training-seed variance.\n",
    flush=True,
)

CELL_ARGS = [
    "--phase",
    PHASE,
    "--output-dir",
    str(OUTPUT_DIR),
    "--tasks",
    TASKS,
    "--seed",
    str(SEED),
    "--graphs",
    str(GRAPHS),
    "--sources-per-graph",
    str(SOURCES_PER_GRAPH),
    "--donor-pairs-per-source",
    str(DONOR_PAIRS_PER_SOURCE),
    "--semantic-donor-graphs",
    str(SEMANTIC_DONOR_GRAPHS),
    "--absolute-effect-floor",
    str(ABSOLUTE_EFFECT_FLOOR),
    "--relative-effect-floor",
    str(RELATIVE_EFFECT_FLOOR),
    "--far-distance",
    str(FAR_DISTANCE),
    "--bootstrap-replicates",
    str(BOOTSTRAP_REPLICATES),
    "--analysis-seed",
    str(ANALYSIS_SEED),
    "--accelerator",
    ACCELERATOR,
    "--num-threads",
    str(NUM_THREADS),
    "--display-max-distance",
    str(DISPLAY_MAX_DISTANCE),
    "--output-tasks",
    OUTPUT_TASKS,
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
]
if not INSTALL_DEPENDENCIES:
    CELL_ARGS.append("--skip-dependency-install")

result = main(CELL_ARGS)

try:
    import pandas as pd
    from IPython.display import Image, display

    health_path = OUTPUT_DIR / "results" / "model_health.csv"
    if health_path.is_file():
        print("\nModel health", flush=True)
        display(pd.read_csv(health_path))

    metric_path = OUTPUT_DIR / "results" / "layer_metric_summary.csv"
    if metric_path.is_file():
        metrics = pd.read_csv(metric_path)
        selected = metrics[
            metrics["metric"].isin(
                [
                    "interaction_relative_mass",
                    "interaction_estimable",
                    "semantic_expected_distance",
                    "structural_expected_distance",
                    "interaction_expected_distance",
                ]
            )
        ]
        print("\nLayer summary", flush=True)
        display(selected)

    output_cache_value = result.get("output_modulation_cache_dir")
    if output_cache_value:
        output_cache = Path(output_cache_value)
        output_health = output_cache / "model_health.csv"
        if output_health.is_file():
            print("\nOutput-M model health", flush=True)
            display(pd.read_csv(output_health))
        output_summary = output_cache / "output_modulation_summary.csv"
        if output_summary.is_file():
            print("\nOutput-M summary", flush=True)
            display(pd.read_csv(output_summary))
        output_contrasts = output_cache / "output_modulation_paired_contrasts.csv"
        if output_contrasts.is_file():
            contrasts = pd.read_csv(output_contrasts)
            print("\nPaired model contrasts", flush=True)
            display(contrasts[contrasts["metric"] == "modulation_m"])

    output_figure = (
        result.get("output_modulation_figures", {}).get("output_modulation", {}).get("png")
    )
    if output_figure:
        print(f"\n[display] output_modulation: {output_figure}", flush=True)
        display(Image(filename=output_figure))

    for name in (
        "final_profiles",
        "layer_heatmaps",
        "layer_reach",
        "interaction_strength",
    ):
        path = result.get("figures", {}).get(name, {}).get("png")
        if path:
            print(f"\n[display] {name}: {path}", flush=True)
            display(Image(filename=path))
except ImportError:
    pass

print(f"\n[done] Analysis saved under {OUTPUT_DIR}", flush=True)
