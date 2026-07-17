"""
Colab bootstrap cell: carriage analysis of the dense GRIT Peptides-struct checkpoint.

Same tiny, stable bootstrap as the other carriage cells -- clone this repo via the
`dissertation_key` secret and call the CENTRAL methodology. Peptides-struct is an 11-target
REGRESSION model (l1 loss, metric MAE), which the multi-output methodology handles the same
way as Peptides-func: functional carriage = magnitude of the output movement over the 11
targets; beneficial carriage = exact per-source change in the L1 loss (mean over targets),
carrier-attributed. The load check recomputes test MAE.

Like Peptides-func, molecules are large (~150 nodes), so this defaults to a modest graph /
donor budget and a smaller per-forward budget; raise them if the A100 has headroom.

Figures collate under /content/drive/MyDrive/graph_specialisation_metrics/carriage_figures/
peptides_struct/ alongside the ZINC and Peptides-func ones.
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

for _p in (os.path.join(REPO_DIR, "src"), REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _m in [m for m in list(sys.modules) if m.startswith("graph_specialisation_metrics")]:
    del sys.modules[_m]

from graph_specialisation_metrics.carriage import run  # noqa: E402

run(
    task="peptides_struct",
    mount=False,
    num_graphs=32,          # ~150-node graphs; raise toward 64 if the A100 has headroom
    donors=16,              # K donor swaps per source
    max_pair_edges=4_000_000,   # smaller per-forward budget for large full-attention graphs
    verify_graphs=2,
)
# ============================ paste to here ============================
