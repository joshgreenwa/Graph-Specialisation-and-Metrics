"""Standalone cache-only Colab frontend for the v3 NAR paper synthesis.

Run ``nar_causal_transition_colab.py`` once first.  This frontend then reads every protected
score and causal cell for N=4,8,16,32,64, performs no checkpoint inference, and writes a new
publication tree at ``extensions/nar_methodology_paper_v3``.  Its headline capacity analysis
tests whether absolute semantic/structural score and held-out causal-response retention
co-transition with fixed-N retrieval competence; it never uses normalised ``J`` as a cross-N
magnitude and does not describe independently trained fixed-N checkpoints as an OOD sweep.
The first full run caches source-fingerprinted mechanism-survival estimands. For later
presentation-only iterations, set ``--render-target mechanism`` and
``--mechanism-cache-mode require`` in ``CELL_ARGS``.
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
            "git", "-C", str(COLAB_REPOSITORY),
            "remote", "set-url", "origin", authenticated,
        )
        command("git", "-C", str(COLAB_REPOSITORY), "fetch", "origin", REPOSITORY_BRANCH)
        command("git", "-C", str(COLAB_REPOSITORY), "checkout", REPOSITORY_BRANCH)
        command(
            "git", "-C", str(COLAB_REPOSITORY),
            "reset", "--hard", f"origin/{REPOSITORY_BRANCH}",
        )
    else:
        if COLAB_REPOSITORY.exists():
            shutil.rmtree(COLAB_REPOSITORY)
        command(
            "git", "clone", "--branch", REPOSITORY_BRANCH, "--single-branch",
            authenticated, str(COLAB_REPOSITORY),
        )
    command(
        "git", "-C", str(COLAB_REPOSITORY),
        "remote", "set-url", "origin", REPOSITORY_URL,
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

from graph_specialisation_metrics.synthetic.nar_methodology_paper_v3 import main


CELL_ARGS = [
    "--phase",
    "figures",
    "--render-target",
    "all",
    "--mechanism-cache-mode",
    "auto",
    "--drive-root",
    "/content/drive/MyDrive/graph_specialisation_metrics/nar_grit",
    "--training-run-name",
    "nar_grit_fixed_n_v3",
    "--base-analysis-name",
    "canonical_nar_analysis_d128",
    "--source-extension-name",
    "nar_role_counterfactual_v1",
    "--causal-extension-name",
    "nar_causal_transition_v1",
    "--paper-analysis-name",
    "nar_methodology_paper_v3",
    "--models",
    "1hop,2hop,dense",
    "--seeds",
    "0,1,2",
    "--cached-ns",
    "4,16,64",
    "--score-ns",
    "4,8,16,32,64",
    "--canonical-causal-ns",
    "4,16,64",
    "--transition-causal-ns",
    "8,32",
    "--counterfactual-ns",
    "4,8,16",
    "--carriage-ns",
    "4,16,64",
    "--performance-ns",
    "4,8,16,32,64,80",
    "--headline-n",
    "16",
    "--analysis-width",
    "128",
    "--counterfactual-donors-per-role",
    "8",
    "--accuracy-gate",
    "0.85",
]

main(CELL_ARGS)
