"""Single Colab frontend for the dissertation's layer-confound correction.

Run this complete file in one Colab cell.  It searches Google Drive for the exact
cached runs behind Figures 4.5(a) and 5.6, computes the equal-weight mean of
within-layer Spearman correlations, bootstraps the molecular panels over held-out
molecules, and writes dissertation-matched replacement figures.

GraphBench is deliberately excluded because its cache resides on HPC.  No model,
checkpoint, dataset, GRIT runtime, Graphormer runtime, GPU, or forward pass is used.
"""

# ruff: noqa: BLE001, I001

# ============================ paste from here ============================
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


# ------------------------------- repository ---------------------------------

REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPOSITORY_REVISION = "codex/layer-controlled-ablation"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
GITHUB_SECRET = "dissertation_key"  # Optional when the repository is public.


# --------------------------- correction controls -----------------------------

MODE = os.environ.get("LAYER_CONTROL_MODE", "inventory").strip().lower()
# "inventory" only lists compatible cache roots; "run" computes and renders;
# "verify" also recomputes and performs the same strict output checks as "run".
if MODE not in {"inventory", "run", "verify"}:
    raise ValueError("MODE must be 'inventory', 'run', or 'verify'")

TASKS = (
    "synthetic",
    "graphormer_pcqm4mv2",
    "zinc",
    "qm9_gap_dense",
)

DRIVE_ROOT = Path("/content/drive/MyDrive")
METRICS_ROOT = DRIVE_ROOT / "graph_specialisation_metrics"
OUTPUT_ROOT = METRICS_ROOT / "layer_controlled_head_ablation_correction_v1"

# Set only when an exact dissertation cache has been relocated.  Values may point
# to the experiment root, task/seed root, or a primary cache file.  Ambiguity is an
# error: the frontend never picks a candidate merely because it is newest.
CACHE_OVERRIDES = {
    "synthetic": "",
    "graphormer_pcqm4mv2": "",
    "zinc": "",
    "qm9_gap_dense": "",
}

BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 17_071
PERMUTATION_REPLICATES = 10_000
PERMUTATION_SEED = 72_019
STRICT = True
# Keep False for the dissertation correction. Setting True explicitly permits a
# labelled 256-molecule population-cache robustness rerun when the original cache
# is unavailable; it never masquerades as the original Figure 5.6 analysis.
ALLOW_NON_DISSERTATION_FALLBACKS = False


def command(*parts: str) -> None:
    shown = [
        "https://***@github.com/" + part.split("@github.com/", 1)[1]
        if "@github.com/" in part
        else part
        for part in parts
    ]
    print(f"[cmd] {' '.join(shown)}", flush=True)
    subprocess.run(list(parts), check=True)


def bootstrap() -> str:
    from google.colab import drive, userdata

    drive.mount("/content/drive", force_remount=False)
    try:
        token = userdata.get(GITHUB_SECRET)
    except Exception as error:
        token = None
        print(
            f"[setup] Colab secret {GITHUB_SECRET!r} unavailable "
            f"({type(error).__name__}); trying a public clone",
            flush=True,
        )
    token = token or os.environ.get(GITHUB_SECRET)
    suffix = REPOSITORY_URL.removeprefix("https://github.com/")
    remote = (
        f"https://x-access-token:{quote(str(token).strip(), safe='')}@github.com/{suffix}"
        if token
        else REPOSITORY_URL
    )
    if not token:
        print(
            f"[setup:warning] Colab secret {GITHUB_SECRET!r} is missing; using public clone",
            flush=True,
        )
    try:
        if not (COLAB_REPOSITORY / ".git").is_dir():
            command("git", "clone", remote, str(COLAB_REPOSITORY))
        else:
            command(
                "git",
                "-C",
                str(COLAB_REPOSITORY),
                "remote",
                "set-url",
                "origin",
                remote,
            )
        command(
            "git",
            "-C",
            str(COLAB_REPOSITORY),
            "fetch",
            "origin",
            REPOSITORY_REVISION,
        )
        command(
            "git",
            "-C",
            str(COLAB_REPOSITORY),
            "switch",
            "--detach",
            "FETCH_HEAD",
        )
        commit = subprocess.check_output(
            ["git", "-C", str(COLAB_REPOSITORY), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    finally:
        # Never leave a credential in .git/config, including on fetch/switch errors.
        if (COLAB_REPOSITORY / ".git").is_dir():
            command(
                "git",
                "-C",
                str(COLAB_REPOSITORY),
                "remote",
                "set-url",
                "origin",
                REPOSITORY_URL,
            )
    # Base analysis dependencies only. No graph/model extras are installed.
    command(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "-e",
        str(COLAB_REPOSITORY),
    )
    command(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "pillow",
        "pypdf",
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
    os.chdir(COLAB_REPOSITORY)
    return commit


repository_commit = bootstrap()

from graph_specialisation_metrics.methodology.layer_controlled_ablation import (
    discover_cache_candidates,
    inventory_rows,
    run_layer_controlled_correction,
)


resolved_overrides = {task: value for task, value in CACHE_OVERRIDES.items() if value}
candidates = discover_cache_candidates(
    METRICS_ROOT,
    cache_overrides=resolved_overrides,
)
inventory = inventory_rows(candidates, strict=STRICT)

try:
    import pandas as pd
    from IPython.display import Image, Markdown, display

    display(Markdown("## Drive cache inventory"))
    display(pd.DataFrame(inventory))
except Exception:
    print(json.dumps(inventory, indent=2, default=str))

if MODE == "inventory":
    print(
        "[inventory] No estimates or figures were written. Resolve any missing or "
        "ambiguous exact caches with CACHE_OVERRIDES, then set MODE='run'."
    )
else:
    result = run_layer_controlled_correction(
        metrics_root=METRICS_ROOT,
        output_root=OUTPUT_ROOT,
        tasks=TASKS,
        cache_overrides=resolved_overrides,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        permutation_replicates=PERMUTATION_REPLICATES,
        permutation_seed=PERMUTATION_SEED,
        strict=STRICT,
        allow_non_dissertation_fallbacks=ALLOW_NON_DISSERTATION_FALLBACKS,
    )
    manifest_path = Path(result["manifest_path"])
    # The backend records the checked-out commit as well; keep the frontend value
    # visible in the cell output for a quick provenance audit.
    print(f"[repository] {repository_commit}")
    print(f"[manifest] {manifest_path}")

    try:
        comparison = pd.read_csv(result["tables"]["pooled_vs_layer_controlled"])
        display(Markdown("## Pooled versus layer-controlled estimates"))
        display(comparison)
        display(Markdown("## Primary confound-free panels"))
        for task in TASKS:
            paths = result["tasks"][task]["primary_within_layer_rank_figures"]
            png = next(Path(path) for path in paths if str(path).endswith(".png"))
            display(Markdown(f"### {task}"))
            display(Image(filename=str(png)))
        display(Markdown("## Dissertation-coordinate companions"))
        for task in TASKS:
            paths = result["tasks"][task]["raw_coordinate_companion_figures"]
            png = next(Path(path) for path in paths if str(path).endswith(".png"))
            display(Markdown(f"### {task}"))
            display(Image(filename=str(png)))
    except Exception as error:
        print(f"[display:warning] {type(error).__name__}: {error}")
    print(f"[complete] Cache-only correction finished: model_forwards=0; outputs={OUTPUT_ROOT}")
# ============================ paste to here ============================
