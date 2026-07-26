"""Standalone Colab frontend for the cache-safe NAR methodology-paper extension.

Paste this complete file into one Colab cell.  The existing
``canonical_nar_analysis_d128/canonical`` tree is opened read-only.  New score-only caches for
N=8 and N=32, role-derived artifacts, counterfactual results, tables, and figures are written to
``canonical_nar_analysis_d128/extensions/nar_role_counterfactual_v1``.

Useful reruns:
    CELL_ARGS = ["--phase", "figures"]        # no GRIT or checkpoint inference
    CELL_ARGS = ["--phase", "roles"]          # cache-only role derivation
    CELL_ARGS = ["--phase", "counterfactual", "--skip-install"]
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
REPOSITORY_BRANCH = "expansion/graphormer_specialisation"
COLAB_REPOSITORY = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"


def command(*parts: str, check: bool = True) -> None:
    shown = [
        "https://***@github.com/" + part.split("@github.com/", 1)[1]
        if "@github.com/" in part
        else part
        for part in parts
    ]
    print(f"[cmd] {' '.join(shown)}", flush=True)
    subprocess.run(list(parts), check=check)


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
        command(
            "git",
            "-C",
            str(COLAB_REPOSITORY),
            "remote",
            "set-url",
            "origin",
            authenticated,
        )
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
    command(
        "git",
        "-C",
        str(COLAB_REPOSITORY),
        "remote",
        "set-url",
        "origin",
        REPOSITORY_URL,
    )
    command(sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_REPOSITORY))
    source = str(COLAB_REPOSITORY / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    for module_name in tuple(sys.modules):
        if module_name == "graph_specialisation_metrics" or module_name.startswith(
            "graph_specialisation_metrics."
        ):
            del sys.modules[module_name]
    importlib.invalidate_caches()


bootstrap()

from graph_specialisation_metrics.synthetic.nar_methodology_extension import main


# Paper-strength defaults.  This frontend does not overwrite canonical caches:
#   existing expensive scores: N=4,16,64
#   isolated new score-only runs: N=8,32
#   exact counterfactual inference: N=4,8,16
#   headline role figure: N=16
CELL_ARGS = [
    "--phase",
    "all",
    "--drive-root",
    "/content/drive/MyDrive/graph_specialisation_metrics/nar_grit",
    "--training-run-name",
    "nar_grit_fixed_n_v3",
    "--base-analysis-name",
    "canonical_nar_analysis_d128",
    "--extension-name",
    "nar_role_counterfactual_v1",
    "--models",
    "1hop,2hop,dense",
    "--seeds",
    "0,1,2",
    "--cached-ns",
    "4,16,64",
    "--additional-score-ns",
    "8,32",
    "--transition-ns",
    "4,8,16,32,64",
    "--counterfactual-ns",
    "4,8,16",
    "--all-performance-ns",
    "4,8,16,32,64,80",
    "--role-figure-n",
    "16",
    "--analysis-width",
    "128",
    "--counterfactual-donors-per-role",
    "8",
    "--accuracy-gate",
    "0.85",
    "--graphs-per-batch",
    "8",
    "--accelerator",
    "cuda:0",
    "--num-threads",
    "4",
]

main(CELL_ARGS)
