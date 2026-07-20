"""BETA — causal validation of semantic/structural head specialisation on ZINC.

Run this file as a single Colab cell. It compares:

  * dense GRIT with global RRWP;
  * 1-hop GRIT with global RRWP;
  * 1-hop GRIT with local-only RRWP.

The experiment reuses the established semantic donor swap, mask-frozen structural
transposition, and per-head transport score. It then tests whether those scores predict
pre-head ablation impact, whether score-selected semantic/structural families selectively
rescue their matching corruption, and whether graphwise channel responses explain the paired
accuracy gaps between checkpoints. The previous effective-response-rank beta is not run.

Prerequisites: the ``dissertation_key`` Colab secret and all three checkpoints on Drive.
Outputs are written under:

    MyDrive/graph_specialisation_metrics/beta_zinc_causal_specialisation
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
from IPython.display import Image, display  # noqa: E402

drive.mount("/content/drive", force_remount=False)
_token = userdata.get(SECRET_NAME)
if not _token:
    raise RuntimeError(f"Colab secret {SECRET_NAME!r} is missing or empty.")
_suffix = REPO_URL.removeprefix("https://github.com/")
_authed = f"https://x-access-token:{_token.strip()}@github.com/{_suffix}"

if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", _authed, REPO_DIR],
        check=True,
    )
else:
    subprocess.run(
        ["git", "-C", REPO_DIR, "remote", "set-url", "origin", _authed], check=True
    )
    subprocess.run(
        ["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True
    )
    subprocess.run(
        ["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"], check=True
    )
subprocess.run(
    ["git", "-C", REPO_DIR, "remote", "set-url", "origin", REPO_URL], check=True
)

_src = os.path.join(REPO_DIR, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
for _module in [
    name for name in list(sys.modules)
    if name.startswith("graph_specialisation_metrics")
]:
    del sys.modules[_module]

from graph_specialisation_metrics.specialisation.richness_beta import run  # noqa: E402

result = run(
    num_graphs=128,
    donors=8,
    max_sources=None,             # all ZINC nodes
    rescue_graphs=128,            # independent eval graphs held out from score estimation
    rescue_batch_size=64,
    family_size=3,                # selected from scores only
    n_null=1000,                  # layer/throughput-matched family nulls
    n_boot=1000,                  # graph-paired bootstrap intervals
    skip_install=False,           # True only if dependencies already exist in this runtime
    mount=False,                  # Drive was mounted above
)

print("\nBETA causal-specialisation analysis complete.")
print(f"Raw arrays: {result['raw_npz']}")
print(f"Summary: {result['summary_json']}")
for _path in result["figures"].values():
    display(Image(filename=_path))
# ============================ paste to here ============================
