"""Single-cell Colab frontend for the Chapter 6 multi-seed figure suite.

Set ``DATASET`` to ``"zinc"`` or ``"qm9"`` and run the cell.  The two values
write to separate Drive directories, so two Colab sessions can run in parallel.
Score and carriage figures are cache-only.  The first run also computes the focused
clean single-head ablation endpoint and, for ZINC, any missing checkpoint-trajectory
scores.  Both stages are resumable, so later runs return to cache-only figure generation.
"""

# ============================ paste from here ============================
# ruff: noqa: I001
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from IPython.display import Image, display
from PIL import Image as PILImage


# ------------------------------- repository ---------------------------------

REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPOSITORY_BRANCH = "expansion/carriage_experiments"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"  # Optional for a public repository.


# --------------------------- experiment controls -----------------------------

# Run one Colab with "zinc" and another with "qm9". Their outputs never overlap.
# The environment override is used by the small .ipynb launcher beside this file.
DATASET = os.environ.get("CHAPTER6_DATASET", "zinc").strip().lower()
SEEDS = (0, 1, 2)
STRICT_CACHE_INVENTORY = True
RELIABLE_HEAD_QUANTILE = 0.25
COMPUTE_MISSING_ABLATIONS = True
COMPUTE_MISSING_TRAJECTORY_SCORES = True
GENERATE_SPECIAL_HEAD_ANALYSIS = True
SPECIAL_HEAD_COMPUTE_MISSING = True
SPECIAL_HEAD_FORCE = False
SPECIAL_HEAD_PCA_GRAPHS = 500
SPECIAL_HEAD_EXAMPLES_PER_PAGE = 5  # two readable pages = ten molecules/head
SPECIAL_HEAD_RENDER_GRAPH_INDICES = None  # later choose a subset of the cached ten
SPECIAL_HEAD_GENERALIST_MAX_ABS_DREL = 0.10
DISPLAY_ALL_SPECIAL_HEAD_FIGURES = False  # full bundles remain saved in Drive

DRIVE_ROOT = Path("/content/drive/MyDrive")
MULTI_SEED_ROOT = DRIVE_ROOT / "graph_specialisation_metrics/multi_seed_models"
CANONICAL_ROOT = MULTI_SEED_ROOT / "canonical_outputs"
OUTPUT_ROOT = MULTI_SEED_ROOT / "chapter6_multiseed_analysis"
OUTPUT_DIR = OUTPUT_ROOT / DATASET.lower()
ABLATION_ROOT = MULTI_SEED_ROOT / "chapter6_clean_head_ablation"
TRAJECTORY_ROOT = MULTI_SEED_ROOT / "multiple_checkpoints_zinc"


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
        remote = f"https://x-access-token:{quote(str(token).strip(), safe='')}@github.com/{suffix}"
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
        command("git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", remote)
        command("git", "-C", str(COLAB_REPOSITORY), "fetch", "origin", REPOSITORY_BRANCH)
        command("git", "-C", str(COLAB_REPOSITORY), "checkout", REPOSITORY_BRANCH)
        command(
            "git",
            "-C",
            str(COLAB_REPOSITORY),
            "merge",
            "--ff-only",
            f"origin/{REPOSITORY_BRANCH}",
        )
    command("git", "-C", str(COLAB_REPOSITORY), "remote", "set-url", "origin", REPOSITORY_URL)
    command(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-e",
        str(COLAB_REPOSITORY),
        "pillow",
        "rdkit",
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


if DATASET.lower() not in {"zinc", "qm9"}:
    raise ValueError("DATASET must be 'zinc' or 'qm9'")

bootstrap()

from graph_specialisation_metrics.chapter6_multiseed import (
    cache_inventory,
    run,
)
from graph_specialisation_metrics.chapter6_clean_ablation import (
    compute_dataset as compute_clean_ablations,
    ensure_completion_manifest as finalize_clean_ablations,
    missing_runs as missing_ablation_runs,
)


missing_ablations = missing_ablation_runs(ABLATION_ROOT, dataset=DATASET, seeds=SEEDS)
if missing_ablations:
    if not COMPUTE_MISSING_ABLATIONS:
        detail = ", ".join(f"{task}/seed_{seed}" for task, seed in missing_ablations)
        raise FileNotFoundError(f"missing clean-head ablation summaries: {detail}")
    print(
        f"[ablation] {len(missing_ablations)} run(s) are missing; "
        "loading checkpoints and computing only clean single-head ablations.",
        flush=True,
    )
    from experiments.methodology.zinc_qm9_canonical_colab_worker import (
        build_production_config,
        ensure_runtime_dependencies,
        load_prepared_corpus,
        require_requested_accelerator,
    )

    require_requested_accelerator("cuda:0")
    ensure_runtime_dependencies()
    corpus = load_prepared_corpus(MULTI_SEED_ROOT)
    methodology_config = build_production_config(
        MULTI_SEED_ROOT,
        corpus,
        accelerator="cuda:0",
    )
    compute_clean_ablations(
        methodology_config,
        CANONICAL_ROOT,
        ABLATION_ROOT,
        dataset=DATASET,
        seeds=SEEDS,
        verbose=True,
    )
ablation_completion = finalize_clean_ablations(
    ABLATION_ROOT,
    dataset=DATASET,
    seeds=SEEDS,
)
print(f"[ablation] persistent completion manifest: {ablation_completion}", flush=True)


trajectory_inventory_rows = []
if DATASET == "zinc":
    from graph_specialisation_metrics.chapter6_score_trajectory import (
        cache_inventory as trajectory_cache_inventory,
    )
    from graph_specialisation_metrics.chapter6_score_trajectory import (
        missing_architectures as missing_trajectory_architectures_from_cache,
    )

    trajectory_inventory_rows = trajectory_cache_inventory(TRAJECTORY_ROOT)
    missing_trajectory_architectures = missing_trajectory_architectures_from_cache(
        TRAJECTORY_ROOT
    )
    if missing_trajectory_architectures:
        if not COMPUTE_MISSING_TRAJECTORY_SCORES:
            detail = ", ".join(
                str(row["score_path"])
                for row in trajectory_inventory_rows
                if not bool(row["score_exists"])
            )
            raise FileNotFoundError(f"missing ZINC trajectory score cache(s): {detail}")
        print(
            "[trajectory] missing score caches for "
            f"{', '.join(missing_trajectory_architectures)}; "
            "computing or resuming only those architectures.",
            flush=True,
        )
        from experiments.methodology.zinc_checkpoint_trajectory_colab import (
            run_architecture as run_trajectory_architecture,
            setup_drive as setup_trajectory_drive,
        )
        from experiments.methodology.zinc_qm9_canonical_colab_worker import (
            ensure_runtime_dependencies,
            require_requested_accelerator,
        )

        require_requested_accelerator("cuda:0")
        ensure_runtime_dependencies()
        prepared_trajectory = setup_trajectory_drive(TRAJECTORY_ROOT)
        for architecture in missing_trajectory_architectures:
            run_trajectory_architecture(
                prepared_trajectory,
                architecture,
                accelerator="cuda:0",
            )
        trajectory_inventory_rows = trajectory_cache_inventory(TRAJECTORY_ROOT)
        still_missing = [
            str(row["score_path"])
            for row in trajectory_inventory_rows
            if not bool(row["score_exists"])
        ]
        if still_missing:
            raise RuntimeError(
                "trajectory computation returned without all score caches: "
                + ", ".join(still_missing)
            )


print(
    f"\n[scope] Dataset: {DATASET.upper()}\n"
    f"[scope] Seeds: {SEEDS}\n"
    "[scope] Five models: 1-hop, 1-hop + VNode, 2-hop, 2-hop + VNode, dense.\n"
    "[scope] Scores/carriage are cache-only; missing ablations and ZINC trajectory "
    "scores resume or compute once.\n"
    "[scope] Head identities are never matched or averaged across seeds.\n"
    f"[scope] Output directory: {OUTPUT_DIR}\n",
    flush=True,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
inventory_rows = cache_inventory(CANONICAL_ROOT, dataset=DATASET, seeds=SEEDS)
inventory_frame = pd.DataFrame(inventory_rows)
print("Cache inventory", flush=True)
display(
    inventory_frame.loc[
        :,
        [
            "task",
            "seed",
            "score_exists",
            "model_exists",
            "carriage_exists",
            "score_path",
        ],
    ]
)

if DATASET == "zinc":
    print("\nZINC checkpoint-trajectory cache inventory", flush=True)
    display(pd.DataFrame(trajectory_inventory_rows))

manifest = run(
    CANONICAL_ROOT,
    OUTPUT_DIR,
    dataset=DATASET,
    seeds=SEEDS,
    strict_inventory=STRICT_CACHE_INVENTORY,
    ablation_root=ABLATION_ROOT,
    strict_ablation=True,
    trajectory_root=TRAJECTORY_ROOT if DATASET == "zinc" else None,
    strict_trajectory=DATASET == "zinc",
    activity_quantile=RELIABLE_HEAD_QUANTILE,
    verbose=True,
)

special_head_manifest = None
if GENERATE_SPECIAL_HEAD_ANALYSIS:
    # Attention and routed-output PCA are the only Chapter 6 multi-seed figures
    # that require model reconstruction. Contract caches make subsequent runs
    # checkpoint-free unless the requested heads or analysis settings change.
    from experiments.methodology.zinc_qm9_canonical_colab_worker import (
        ensure_runtime_dependencies,
        require_requested_accelerator,
    )
    from graph_specialisation_metrics.chapter6_special_heads import (
        generate_special_head_analysis,
    )

    require_requested_accelerator("cuda:0")
    ensure_runtime_dependencies()
    special_head_manifest = generate_special_head_analysis(
        CANONICAL_ROOT,
        ABLATION_ROOT,
        OUTPUT_DIR,
        dataset=DATASET,
        seeds=SEEDS,
        render_graph_indices=SPECIAL_HEAD_RENDER_GRAPH_INDICES,
        n_pca_graphs=SPECIAL_HEAD_PCA_GRAPHS,
        examples_per_page=SPECIAL_HEAD_EXAMPLES_PER_PAGE,
        accelerator="cuda:0",
        compute_missing=SPECIAL_HEAD_COMPUTE_MISSING,
        force=SPECIAL_HEAD_FORCE,
        generalist_max_abs_drel=SPECIAL_HEAD_GENERALIST_MAX_ABS_DREL,
        verbose=True,
    )
    print("\nControlled special-head selection", flush=True)
    selected_columns = [
        "model_label",
        "role",
        "display_family",
        "seed",
        "layer",
        "head",
        "selectivity",
        "joint_sensitivity",
        "head_ablation_impact",
    ]
    display(pd.DataFrame(special_head_manifest["selected_heads"])[selected_columns])

print("\nSemantic--structural distance alignment", flush=True)
display(pd.read_csv(OUTPUT_DIR / "semantic_structural_alignment.csv"))

print("\nGenerated figures", flush=True)
pngs = [Path(path) for path in manifest["figures"] if str(path).endswith(".png")]
for path in pngs:
    print(path.name, flush=True)
    display(Image(filename=str(path)))

if special_head_manifest is not None:
    completion = pd.DataFrame(special_head_manifest["completion"])
    print("\nControlled special-head completion (all rows must be True)", flush=True)
    display(
        completion[
            [
                "model_label",
                "role",
                "seed",
                "layer",
                "head",
                "attention_pages",
                "pca_figures",
                "distance_figures",
                "complete",
            ]
        ]
    )

    if DISPLAY_ALL_SPECIAL_HEAD_FIGURES:
        print("\nAll controlled special-head figures", flush=True)
        for record in special_head_manifest["outputs"]:
            path = Path(record["png"])
            print(
                f"{record['task']} · seed {record['seed']} · {record['role']} · "
                f"{record['figure']}: {path.name}",
                flush=True,
            )
            display(Image(filename=str(path)))
    else:
        # Five compact contact sheets prove that all architectures were rendered
        # without asking Colab to inline roughly sixty publication-resolution files.
        preview_dir = OUTPUT_DIR / "special_head_analysis/previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        role_order = ("semantic", "structural", "highest_joint")
        print("\nSpecial-head attention previews (one sheet per model)", flush=True)
        for task in special_head_manifest["models"]:
            panels = []
            for role in role_order:
                candidates = [
                    record
                    for record in special_head_manifest["outputs"]
                    if record["task"] == task
                    and record["role"] == role
                    and record["figure"] == "attention"
                ]
                if not candidates:
                    raise RuntimeError(f"missing attention preview for {task}/{role}")
                with PILImage.open(candidates[0]["png"]) as source:
                    panel = source.convert("RGB")
                    panel.thumbnail((1200, 900), PILImage.Resampling.LANCZOS)
                    panels.append(panel.copy())
            width = max(panel.width for panel in panels)
            gap = 20
            height = sum(panel.height for panel in panels) + gap * (len(panels) - 1)
            sheet = PILImage.new("RGB", (width, height), "white")
            cursor = 0
            for panel in panels:
                sheet.paste(panel, ((width - panel.width) // 2, cursor))
                cursor += panel.height + gap
            preview_path = preview_dir / f"{task}_selected_heads.jpg"
            sheet.save(preview_path, quality=88, optimize=True)
            sheet.close()
            print(f"{task}: {preview_path.name}", flush=True)
            display(Image(filename=str(preview_path), width=900))
        print(
            "Full attention, PCA, and distance-profile bundles: "
            f"{OUTPUT_DIR / 'special_head_analysis/figures'}",
            flush=True,
        )

print(f"\n[done] {DATASET.upper()} outputs: {OUTPUT_DIR}", flush=True)
# ============================== paste to here ==============================
