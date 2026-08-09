"""Single-cell Colab frontend for the Chapter 6 Peptides cache analysis.

Set ``DATASET`` to ``"peptides_func"`` or ``"peptides_struct"``.  This frontend
is CPU-only and cache-only: it never reconstructs checkpoints and never runs
attention-example or PCA analysis.
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
from IPython.display import FileLink, display


# ------------------------------- repository ---------------------------------

REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPOSITORY_BRANCH = "expansion/carriage_experiments"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"  # Optional while the repository is public.


# --------------------------- experiment controls -----------------------------

DATASET = os.environ.get("CHAPTER6_PEPTIDES_DATASET", "peptides_func").strip().lower()
FORCE_REBUILD = os.environ.get("CHAPTER6_PEPTIDES_FORCE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
SEEDS = (0, 1, 2)
MINIMUM_GRAPHS = 10
MINIMUM_PAIRS = 50

DRIVE_ROOT = Path("/content/drive/MyDrive")
MULTI_SEED_ROOT = DRIVE_ROOT / "graph_specialisation_metrics/multi_seed_models"
CANONICAL_ROOT = (
    MULTI_SEED_ROOT
    / "peptides_func_struct_checkpoints"
    / "canonical_outputs"
)
DATASET_KEY = DATASET.replace("-", "_")
if DATASET_KEY in {"func", "peptides_func"}:
    DATASET_KEY = "peptides_func"
elif DATASET_KEY in {"struct", "peptides_struct"}:
    DATASET_KEY = "peptides_struct"
else:
    raise ValueError("DATASET must be 'peptides_func' or 'peptides_struct'")
OUTPUT_DIR = MULTI_SEED_ROOT / "chapter6_peptides_analysis" / DATASET_KEY


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
    command(sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPOSITORY))

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


bootstrap()

from graph_specialisation_metrics.chapter6_peptides import (
    FIGURE_FILENAMES,
    cache_inventory,
    read_manifest,
    run,
)


inventory_rows = cache_inventory(CANONICAL_ROOT, dataset=DATASET_KEY, seeds=SEEDS)
inventory_frame = pd.DataFrame(inventory_rows)
display(
    inventory_frame[
        [
            "task",
            "seed",
            "score_path",
            "score_exists",
            "carriage_exists",
        ]
    ]
)
if len(inventory_frame) != 15:
    raise RuntimeError(f"expected 15 Peptides cache records, found {len(inventory_frame)}")
if not bool(inventory_frame["score_exists"].all()):
    raise FileNotFoundError("one or more consolidated score caches are missing")
if not bool(inventory_frame["carriage_exists"].all()):
    raise FileNotFoundError("one or more consolidated carriage caches are missing")

manifest = None if FORCE_REBUILD else read_manifest(OUTPUT_DIR)
if manifest is None:
    print(
        "[scope] CPU/cache-only analysis: 15 score/carriage runs; no checkpoints, "
        "ablations, attention examples, PCA, or GPU work.",
        flush=True,
    )
    manifest = run(
        CANONICAL_ROOT,
        OUTPUT_DIR,
        dataset=DATASET_KEY,
        seeds=SEEDS,
        minimum_graphs=MINIMUM_GRAPHS,
        minimum_pairs=MINIMUM_PAIRS,
        strict_inventory=True,
        verbose=True,
    )
else:
    print(
        "[resume] complete peptide figure suite found; no scientific cache loaded",
        flush=True,
    )

print(f"\n[complete] {manifest['dataset']} figures: {OUTPUT_DIR / 'figures'}")
for filename in FIGURE_FILENAMES:
    path = OUTPUT_DIR / "figures" / filename
    if not path.is_file():
        raise FileNotFoundError(f"expected figure was not written: {path}")
    display(FileLink(str(path)))

print("\nDistance bins:", ", ".join(manifest["distance_bins"]))
print("Virtual node: separate non-distance category")
print("Runtime: CPU is sufficient; a GPU provides no benefit for this frontend.")
# ============================= paste to here =============================
