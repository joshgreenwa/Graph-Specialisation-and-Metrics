"""Standalone Colab frontend for ZINC Bamberger-versus-carriage reach.

Paste this complete file into one Colab cell.  It loads the seed-0 checkpoints
for 1-hop, 2-hop, 1-hop+VNode, and dense GRIT, caches graph-level measurements
to Drive, saves PNG/PDF figures, and displays every PNG in the notebook.

After the first completed run, set ``PHASE = "figures"`` to rebuild figures
without reinstalling GRIT or loading checkpoints.
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

PHASE = "all"  # "all", "measure", or "figures"
OUTPUT_DIR = Path(
    "/content/drive/MyDrive/graph_specialisation_metrics/"
    "zinc_bamberger_functional_reach_v4"
)
TASKS = "zinc_1hop,zinc_2hop,zinc_1hop_vnode,zinc"
SEED = 0
GRAPHS = 16
SOURCES_PER_GRAPH = 6
DONORS_PER_SOURCE = 4
SEMANTIC_DONOR_GRAPHS = 256
BAMBERGER_OUTPUT_NODES = 6
BAMBERGER_OUTPUT_CHANNELS = 8
BOOTSTRAP_REPLICATES = 2_000
ANALYSIS_SEED = 91_021
ACCELERATOR = "cuda:0"
NUM_THREADS = 4

# Set False only when this exact Colab runtime already has the canonical GRIT/PyG stack.
INSTALL_DEPENDENCIES = True


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
        str(COLAB_REPOSITORY),
    )

    source_path = (COLAB_REPOSITORY / "src").resolve()
    backend_path = (
        source_path / "graph_specialisation_metrics" / "zinc_reach_analysis.py"
    )
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

from graph_specialisation_metrics.zinc_reach_analysis import main


print(
    "\n[scope] Semantic: literal Bamberger pre-pooling Jacobian range and "
    "finite Functional carriage, decomposed through a donor-direction Jacobian "
    "and raw finite hidden-state response.\n"
    "[scope] Structural: finite Functional carriage only; "
    "Bamberger has no canonical structural intervention analogue.\n"
    "[scope] Fairness: checkpoints, graphs, SPD and carrier site are shared. "
    "The donor-direction, raw-finite and Functional profiles use identical donor "
    "events; literal Bamberger remains output-centric and channel-subsampled.\n"
    "[scope] Interpretation: this is an estimand comparison, not a claim that "
    "the two methods measure the same quantity. There is no learned-route ground "
    "truth on ZINC.\n"
    "[scope] Uncertainty: 95% held-out-graph bootstrap from one seed-0 checkpoint "
    "per architecture; it does not include training-seed variance.\n",
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
    "--donors-per-source",
    str(DONORS_PER_SOURCE),
    "--semantic-donor-graphs",
    str(SEMANTIC_DONOR_GRAPHS),
    "--bamberger-output-nodes",
    str(BAMBERGER_OUTPUT_NODES),
    "--bamberger-output-channels",
    str(BAMBERGER_OUTPUT_CHANNELS),
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

if "health" in result.get("measurement", {}):
    from IPython.display import display

    try:
        import pandas as pd

        print("\nModel health", flush=True)
        display(pd.DataFrame(result["measurement"]["health"]))
    except ImportError:
        print(result["measurement"]["health"], flush=True)

if "expected_rows" in result:
    from IPython.display import display

    try:
        import pandas as pd

        print("\nExpected-distance summary", flush=True)
        display(pd.DataFrame(result["expected_rows"]))
    except ImportError:
        pass

if "figures" in result:
    from IPython.display import Image, display

    for name in (
        "semantic_decomposition",
        "semantic_functional",
        "semantic_bamberger",
        "structural_functional",
        "expected_distance",
    ):
        path = result["figures"][name]["png"]
        print(f"\n[display] {name}: {path}", flush=True)
        display(Image(filename=path))

print(f"\n[done] Analysis saved under {OUTPUT_DIR}", flush=True)
