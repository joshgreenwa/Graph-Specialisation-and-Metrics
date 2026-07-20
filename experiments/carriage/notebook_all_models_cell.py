"""Colab cell: evaluate + carriage + specialise + compare ALL ZINC GRIT models.

Paste this whole cell into Colab and run it. Like the other bootstrap cells it clones this
dissertation repo (via the ``dissertation_key`` Colab secret), puts <repo>/src (and <repo>) on
the path, and calls the CENTRAL cross-model methodology in
``graph_specialisation_metrics.comparison``. All the science lives in the repo, so editing +
pushing there updates this notebook on its next run -- you do not edit this cell.

It runs, for each of the five ZINC GRIT models -- dense, 1-hop, 2-hop, 1-hop+VNode, 2-hop+VNode:
  (1) val + test performance recomputed from the checkpoint;
  (2) functional/beneficial SEMANTIC and STRUCTURAL carriage (integrated beneficial estimator);
  (3) per-head SEMANTIC vs STRUCTURAL specialisation scores;
  (4) OPTIONAL per-head channel-split causal ablation (swap x ablate), I_sem/I_str functional+loss;
and CACHES all of it to Drive, then builds the deliverables:
  (ii)  fig_spec_scatter_grid.png          -- side-by-side per-model score scatter (S_str vs S_sem);
  (ii-b)fig_spec_DJ_grid.png               -- selectivity D_rel vs joint strength J per model;
  (iii) fig_carriage_smallmult_<intv>.png  -- functional/beneficial carriage, standardised y-axis;
  (iv)  fig_carriage_overlay_<intv>.png    -- functional/beneficial carriage overlaid per method;
  (v)   fig_DJ_ablation_validation.png     -- D_rel vs the causal ablation contrast (functional &
        loss), + fig_DJ_quadrants.png + fig_DJ_influence_strength.png (needs with_channel_ablation);
  (+)   fig_performance.png                -- val/test bars.

The new 2-hop / VNode checkpoints are auto-registered from carriage.tasks; their Drive dirs are
grit_zinc_2hop, grit_zinc_1hop_vnode, grit_zinc_2hop_vnode. run_all pins the exact recovery
checkpoints the training runner wrote (see comparison.run.DEFAULT_CKPTS); dense/1-hop
auto-discover their GraphGym ckpt/ dirs.
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

# Fresh checkout each run so central methodology edits always take effect.
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", _authed, REPO_DIR], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", _authed], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"], check=True)
# Restore the non-secret URL so the token is not left in .git/config.
subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", REPO_URL], check=True)

# Both <repo>/src AND <repo> on the path: src holds the package; the repo root holds
# GRIT_khop_ZINC.py (whose patch the 2-hop / VNode env hooks replay) and the experiments/ package.
for _p in (os.path.join(REPO_DIR, "src"), REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Drop any stale modules from a previous run so the fresh checkout is imported.
for _m in [m for m in list(sys.modules) if m.startswith("graph_specialisation_metrics")]:
    del sys.modules[_m]

from graph_specialisation_metrics.comparison import run_all, build_figures  # noqa: E402

# First run on a fresh runtime: omit skip_install so deps install (~a few minutes).
# On a warm runtime (e.g. right after a training run), pass skip_install=True.
# force=False reuses any cached carriage/score results, so re-runs are incremental.
run_all(
    # tasks=["zinc", "zinc_1hop", "zinc_2hop", "zinc_1hop_vnode", "zinc_2hop_vnode"],  # default
    skip_install=False,
    force=False,
    display=True,     # show the figures inline
    # ---- channel-split causal ablation -> the D/J validation figure (deliverable v) ----
    # HEAVY: L*H ablated forwards per model over intervention replicas. Set False to skip it
    # (the score-only quadrant/influence figures still build). Scale the knobs up for tighter CIs.
    with_channel_ablation=True,
    channel_ablation_graphs=48,
    channel_ablation_sources=6,
    channel_ablation_donors=3,
)

# --- Re-draw the deliverables from cache only (no re-compute), e.g. dropping the VNode runs: ---
# build_figures(drop_vnode=True, display=True)
# build_figures(include=["zinc", "zinc_1hop", "zinc_2hop"], display=True)
#
# --- Diagnose why a model is missing from a figure (checkpoint found? which caches exist?): ---
# from graph_specialisation_metrics.comparison import inventory
# inventory()          # prints a per-model table: ckpt | carriage | scores | chanAbl | val/test
# ============================ paste to here ============================
