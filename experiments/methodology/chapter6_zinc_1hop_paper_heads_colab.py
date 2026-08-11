"""CPU-only Colab frontend for the two final ZINC 1-hop head figures.

The frontend reads the existing cross-seed selection and controlled-attention
caches from Drive.  It performs no model, checkpoint, score, PCA, dataset, or
attention recomputation.
"""

# ============================ paste from here ============================
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

from IPython.display import Image, display

REPOSITORY_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
REPOSITORY_BRANCH = "expansion/carriage_experiments"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"

DRIVE_ROOT = Path("/content/drive/MyDrive")
OUTPUT_DIR = (
    DRIVE_ROOT
    / "graph_specialisation_metrics"
    / "multi_seed_models"
    / "chapter6_multiseed_analysis"
    / "zinc"
)
FIGURE_DIR = OUTPUT_DIR / "paper_attention_examples"

# Prevent an accidental CUDA context: this job is intentionally CPU-only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


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
            "https://x-access-token:"
            f"{quote(str(token).strip(), safe='')}@github.com/{suffix}"
        )
    else:
        print(
            f"[setup:warning] Colab secret {SECRET_NAME!r} is missing; "
            "using the public repository",
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


bootstrap()

from pypdf import PdfReader

from graph_specialisation_metrics.chapter6_special_heads import (
    render_cached_zinc_1hop_paper_heads,
)

print(
    "\n[scope] CPU-only figure refresh\n"
    "[scope] Task: ZINC 1-hop\n"
    "[scope] Roles: cross-seed most semantic and most structural\n"
    "[scope] Molecules: 80 and 100\n"
    "[scope] No model-dependent computation is available in this frontend\n"
    f"[scope] Output: {FIGURE_DIR}\n",
    flush=True,
)

result = render_cached_zinc_1hop_paper_heads(
    OUTPUT_DIR,
    graph_indices=(80, 100),
    figure_output_dir=FIGURE_DIR,
    verbose=True,
)

if len(result["figures"]) != 2:
    raise RuntimeError(f"expected exactly two figures, found {len(result['figures'])}")
if {row["role"] for row in result["figures"]} != {"semantic", "structural"}:
    raise RuntimeError("the output roles are not exactly semantic and structural")

print("\nGenerated cache-only paper figures", flush=True)
for row in result["figures"]:
    pdf_path = Path(row["pdf"])
    png_path = Path(row["png"])
    if len(PdfReader(pdf_path).pages) != 1:
        raise RuntimeError(f"expected a one-page PDF: {pdf_path}")
    print(
        f"{row['role']}: seed={row['seed']}; L{row['layer']} H{row['head']}; "
        f"D_rel={row['D_rel']:+.3f}; J={row['J']:.3f}\n{pdf_path}",
        flush=True,
    )
    display(Image(filename=str(png_path)))

print("\n[complete] Two cached ZINC 1-hop figures regenerated on CPU.", flush=True)
# ============================= paste to here =============================
