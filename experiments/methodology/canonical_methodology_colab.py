"""Paste/run this lightweight cell in Colab to execute the final methodology.

Edit only TASKS, TRAIN_SEEDS, PHASES, CHECKPOINTS, and SIZES.  Scientific definitions live in
``src/graph_specialisation_metrics/README.md`` and the canonical package; this front end merely
checks out the chosen repository revision, mounts Drive, and dispatches registered tasks.
"""

# ============================ paste from here ============================
import os
import subprocess
import sys

REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
BRANCH = "main"
REPO_DIR = "/content/Graph-Specialisation-and-Metrics"
SECRET_NAME = "dissertation_key"

# First production run: all registered dense task families.
TASKS = ("zinc", "qm9_gap_dense", "peptides_func", "peptides_struct")
TRAIN_SEEDS = (42,)
PHASES = ("scores", "causal", "carriage", "figures")
CHECKPOINTS = {}  # e.g. {"zinc:42": "/content/drive/MyDrive/.../best.ckpt"}
SIZES = {
    "discovery_graphs": 48,
    "causal_graphs": 24,
    "clean_ablation_graphs": 64,
    "semantic_donor_graphs": 2_000,
    "sources_per_graph": 6,
    "donors_per_source": 8,
    "bootstrap_replicates": 2_000,  # fixed by the normative protocol
}

from google.colab import drive, userdata  # noqa: E402

drive.mount("/content/drive", force_remount=False)
token = userdata.get(SECRET_NAME)
if not token:
    raise RuntimeError(f"Colab secret {SECRET_NAME!r} is missing or empty")
suffix = REPO_URL.removeprefix("https://github.com/")
authenticated = f"https://x-access-token:{token.strip()}@github.com/{suffix}"
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", authenticated, REPO_DIR],
        check=True,
    )
else:
    subprocess.run(
        ["git", "-C", REPO_DIR, "remote", "set-url", "origin", authenticated], check=True
    )
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(
        ["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"], check=True
    )
subprocess.run(
    ["git", "-C", REPO_DIR, "remote", "set-url", "origin", REPO_URL], check=True
)

source = os.path.join(REPO_DIR, "src")
if source not in sys.path:
    sys.path.insert(0, source)
for module in [
    name for name in tuple(sys.modules) if name.startswith("graph_specialisation_metrics")
]:
    del sys.modules[module]

from graph_specialisation_metrics.methodology.colab import run  # noqa: E402

run(
    tasks=TASKS,
    train_seeds=TRAIN_SEEDS,
    phases=PHASES,
    checkpoints=CHECKPOINTS,
    sizes=SIZES,
    mount=False,
)
# On a runtime where GRIT/PyG dependencies are already installed, add skip_install=True.
# ============================ paste to here ============================
