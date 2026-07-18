"""
Colab bootstrap cell: carriage analysis of the dense GRIT Peptides-func checkpoint.

Same tiny, stable bootstrap as notebook_bootstrap_cell.py -- it clones this repo via the
`dissertation_key` secret and calls the CENTRAL methodology. Only the task and a couple of
size knobs differ. Peptides-func is a 10-way multilabel classification model, so the
methodology automatically uses the multi-output readout (functional carriage = magnitude
of the output movement over the 10 logits; beneficial carriage = exact per-source change
in the BCE loss, carrier-attributed). The load check recomputes test AP.

Peptides molecules are large (~150 nodes) vs ZINC (~23), so each graph is far heavier. For
statistical power we use >=128 graphs and K>=128 donors (32/32 is under-powered); this is a
multi-hour run on the A100 -- start smaller (num_graphs=16) to sanity-check the green checks,
then scale up. Watch the [mem] line; the OOM backoff shrinks chunks automatically.

Two aggregation choices (both default to the improved behaviour; documented on run()):
  * beneficial_denom="magnitude" (default) -- convex |C| shares, so |B| <= |dL_j| (no
    blow-up) and B vanishes where functional carriage vanishes. Pass "signed" for the
    legacy signed-sum share (can spike at far distances) only if you want to compare.
  * bin_strategy="log" (default) + central="trimmed" -- F/B pooled into adaptive SPD bins
    with a robust central tendency + graph-clustered bootstrap CI, which is what makes the
    large-diameter x-axis legible. Pass bin_strategy="hop" for per-hop.

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
    num_graphs=128,         # >=128 for tight bootstrap CIs; drop to 16 for a quick check
    donors=128,             # K>=128 donor swaps/source (32 is under-powered)
    max_pair_edges=4_000_000,   # smaller per-forward budget for large full-attention graphs
    verify_graphs=2,
    # defaults already applied: beneficial_denom="magnitude", bin_strategy="log", central="trimmed"
    # to compare the legacy attribution: beneficial_denom="signed"
)
# ============================ paste to here ============================
