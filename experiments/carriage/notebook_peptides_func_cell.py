"""
Colab bootstrap cell: carriage analysis of the dense GRIT Peptides-func checkpoint.

Same tiny, stable bootstrap as notebook_bootstrap_cell.py -- it clones this repo via the
`dissertation_key` secret and calls the CENTRAL methodology. Only the task and a couple of
size knobs differ. Peptides-func is a 10-way multilabel classification model, so the
methodology automatically uses the multi-output readout (functional carriage = magnitude
of the output movement over the 10 logits; beneficial carriage = exact per-source change
in the BCE loss, carrier-attributed). The load check recomputes test AP.

Peptides molecules are large (~150 nodes) vs ZINC (~23), so each graph is far heavier:
we default to fewer graphs / donors and a smaller per-forward budget. Raise them if the
A100 has headroom (watch the [mem] line).

Figures collate under /content/drive/MyDrive/graph_specialisation_metrics/carriage_figures/
peptides_func/ alongside the ZINC ones.
"""

# ============================ paste from here ============================
import os
import subprocess
import sys

REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
BRANCH = "codex/cfim-grit-experiments"
REPO_DIR = "/content/Graph-Specialisation-and-Metrics"
SECRET_NAME = "dissertation_key"

from google.colab import drive, userdata  # noqa: E402

drive.mount("/content/drive", force_remount=False)

_token = userdata.get(SECRET_NAME)
if not _token:
    raise RuntimeError(f"Colab secret {SECRET_NAME!r} is missing or empty.")
_suffix = REPO_URL.removeprefix("https://github.com/")
_authed = f"https://x-access-token:{_token.strip()}@github.com/{_suffix}"

if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", _authed, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", _authed], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"], check=True)
subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", REPO_URL], check=True)

# Both <repo>/src (the carriage package) and <repo> (the experiments/ peptides patches).
for _p in (os.path.join(REPO_DIR, "src"), REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _m in [m for m in list(sys.modules) if m.startswith("graph_specialisation_metrics")]:
    del sys.modules[_m]

from graph_specialisation_metrics.carriage import run  # noqa: E402

# Fresh runtime: omit skip_install (installs GRIT deps + RDKit, applies peptides patches).
# Peptides is heavy: start modest and scale up if memory allows.
run(
    task="peptides_func",
    mount=False,
    num_graphs=32,          # ~150-node graphs; raise toward 64 if the A100 has headroom
    donors=16,              # K donor swaps per source
    max_pair_edges=4_000_000,   # smaller per-forward budget for large full-attention graphs
    verify_graphs=2,
)
# ============================ paste to here ============================
