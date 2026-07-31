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
    "zinc_bamberger_functional_reach_v7"
)
TASKS = "zinc_1hop,zinc_2hop,zinc_1hop_vnode,zinc"
SEED = 0
GRAPHS = 64
SOURCES_PER_GRAPH = 6
DONORS_PER_SOURCE = 4
SEMANTIC_DONOR_GRAPHS = 256
BAMBERGER_OUTPUT_NODES = 6
BAMBERGER_OUTPUT_CHANNELS = 8
INTERPOLATION_DOSES = "0.01,0.02,0.05,0.1,0.25,0.5,1.0"
INTERPOLATION_BATCH_SIZE = 64
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
    "finite Functional carriage.\n"
    "[scope] Structural: finite Functional carriage only; "
    "Bamberger has no canonical structural intervention analogue.\n"
    "[scope] Structural distance is interpreted as source-conditioned usage, "
    "not literal hopwise transport, because RRWP relations may be accessed "
    "directly.\n"
    "[scope] Core check: semantic Functional carriage is recomputed along the "
    "same clean-to-donor event at increasing donor fractions. Departure from the "
    "Bamberger profile tests local linearisation versus finite intervention.\n"
    "[scope] Matched control: every dose is also compared with alpha=0.01 on "
    "identical graphs, donors, carriers and task projections. This separates "
    "finite nonlinear change from the residual Bamberger estimand mismatch.\n"
    "[scope] Scale analysis: MAE uses every test molecule; Functional reach uses "
    "the 64 carriage graphs. Adjacent values are grouped adaptively by data density, "
    "and continuous paired-bootstrap slopes avoid dependence on bin boundaries.\n"
    "[scope] Cancellation analysis: signed scalar-output carriage is integrated "
    "along each finite semantic donor path. Apparent mass sums carrier magnitudes; "
    "coherent mass sums signed carriers before taking magnitude.\n"
    "[scope] Fairness: checkpoints, graphs, SPD and carrier site are shared; "
    "literal Bamberger remains output-centric and channel-subsampled.\n"
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
    "--interpolation-doses",
    INTERPOLATION_DOSES,
    "--interpolation-batch-size",
    str(INTERPOLATION_BATCH_SIZE),
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

if "interpolation_rows" in result:
    from IPython.display import display

    try:
        import pandas as pd

        print("\nInterpolation-sweep summary", flush=True)
        display(pd.DataFrame(result["interpolation_rows"]))
    except ImportError:
        pass

if "scale_rows" in result:
    from IPython.display import display

    try:
        import pandas as pd

        print("\nMolecular-scale summary", flush=True)
        display(pd.DataFrame(result["scale_rows"]))
    except ImportError:
        pass

if "scale_trends" in result:
    from IPython.display import display

    try:
        import pandas as pd

        print("\nContinuous molecular-scale trends", flush=True)
        display(pd.DataFrame(result["scale_trends"]))
    except ImportError:
        pass

if "output_coherence_expected" in result:
    from IPython.display import display

    try:
        import pandas as pd

        print("\nApparent-versus-coherent expected distance", flush=True)
        display(pd.DataFrame(result["output_coherence_expected"]))
    except ImportError:
        pass

if "figures" in result:
    from IPython.display import Image, display

    for name in (
        "interpolation_sweep",
        "semantic_functional",
        "semantic_bamberger",
        "structural_functional",
        "expected_distance",
        "output_coherence",
        "scale_dependence",
        "scale_slopes",
    ):
        path = result["figures"][name]["png"]
        print(f"\n[display] {name}: {path}", flush=True)
        display(Image(filename=path))

print(f"\n[done] Analysis saved under {OUTPUT_DIR}", flush=True)
