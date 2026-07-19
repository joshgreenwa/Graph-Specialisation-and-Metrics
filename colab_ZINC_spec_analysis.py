"""Colab entrypoint for the held-out ZINC head-specialisation paper analysis.

The scientific implementation lives in ``src/graph_specialisation_metrics/specialisation``.
Set ``causal_extension=False`` in the final call to recover the legacy score/zero-ablation/
attention-grid analysis without removing any code.
"""

import os
import subprocess
import sys

REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
BRANCH = "codex/cfim-grit-experiments"
REPO_DIR = "/content/Graph-Specialisation-and-Metrics"
SECRET_NAME = "dissertation_key"

from google.colab import drive, userdata

drive.mount("/content/drive", force_remount=False)

_token = userdata.get(SECRET_NAME)
if not _token:
    raise RuntimeError(f"Colab secret {SECRET_NAME!r} is missing or empty.")
_suffix = REPO_URL.removeprefix("https://github.com/")
_authed = f"https://x-access-token:{_token.strip()}@github.com/{_suffix}"

# Fresh checkout each run so central methodology edits always take effect.
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", _authed, REPO_DIR],
        check=True,
    )
else:
    subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", _authed], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(
        ["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"],
        check=True,
    )
# Do not leave the token in .git/config.
subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", REPO_URL], check=True)

_src = os.path.join(REPO_DIR, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
for _module in [m for m in list(sys.modules) if m.startswith("graph_specialisation_metrics")]:
    del sys.modules[_module]

from graph_specialisation_metrics.specialisation import run

run(
    tasks=["zinc"],
    num_graphs=64,          # discovery graphs used only to estimate head scores
    donors=96,              # donor/partner marginalisation for discovery scores
    max_sources=None,
    with_attn_routing=True,

    # Default-on held-out paper extension. False restores the exact legacy pipeline.
    causal_extension=True,
    causal_graphs=64,       # confirmation molecules, disjoint from score discovery
    causal_interventions_per_graph=2,
    causal_topk=(1, 2, 4, 8),
    causal_primary_k=4,
    matched_control_draws=8,
    transport_residual_permutations=4,
    bootstrap_replicates=1000,
    legacy_outputs=False,  # True additionally writes the old ablation/attention-grid appendix

    skip_install=False,
    mount=True,
)
