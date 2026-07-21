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
  (4) per-head channel-split causal ablation (swap x ablate), I_sem/I_str functional+loss;
and CACHES all of it to Drive, then builds the deliverables:
  (ii)  fig_spec_scatter_grid.png          -- side-by-side per-model score scatter (S_str vs S_sem);
  (ii-b)fig_spec_DJ_grid.png               -- selectivity D_rel vs joint strength J per model;
  (iii) fig_carriage_smallmult_<intv>.png  -- functional/beneficial carriage, standardised y-axis;
  (iv)  fig_carriage_overlay_<intv>.png    -- functional/beneficial carriage overlaid per method;
  (v)   fig_DJ_ablation_validation.png     -- D_rel vs the causal ablation contrast (functional &
        loss), + fig_DJ_quadrants.png + fig_DJ_influence_strength.png (needs with_channel_ablation);
  (vi)  fig_DJ_family_ablation_curves.png + fig_DJ_family_ablation_contrasts.png -- held-out,
        cumulative semantic/structural/generalist family ablation crossed with high/low J;
  (vii) fig_semantic_outlier_ablation.png  -- direct held-out test of each model's largest raw
        semantic-score heads against layer-nearest throughput controls;
        fig_structural_outlier_ablation.png -- the identical test for raw structural-score heads;
        fig_semantic_outlier_attention_dense_LxHy.png -- one 4-molecule topology/inflow/raw-
        matrix figure for each of the six dense semantic outliers;
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
# force=False reuses every existing carriage result and computes only missing model/intervention
# pairs. Dense/1-hop are therefore not repeated when completing 2-hop/VNode carriage. The two
# pre-v2 VNode score caches are intentionally refreshed once to include VNode-mediated transport.
# The factorial family stage below consumes those score caches and has its own cache: it computes
# only the new validation-graph group ablations, never carriage or specialisation scores.
# The semantic-outlier stage is cached independently and, under the defaults below, reuses the
# factorial stage's graph IDs, clean predictions, labels, and throughput (no repeated clean pass).
run_all(
    # tasks=["zinc", "zinc_1hop", "zinc_2hop", "zinc_1hop_vnode", "zinc_2hop_vnode"],  # default
    skip_install=False,
    force=False,
    allow_partial=False,  # fail clearly instead of silently drawing a subset of requested models
    display=True,     # show the figures inline
    # ---- channel-split causal ablation -> the D/J validation figure (deliverable v) ----
    # HEAVY: L*H ablated forwards per model over intervention replicas. This is on by default for
    # the paper-analysis run; set False explicitly for a score/carriage-only diagnostic run.
    with_channel_ablation=True,
    channel_ablation_graphs=128,
    channel_ablation_sources=32,
    channel_ablation_donors=16,  # each donor => L*H extra ablated forwards
    integrated_atol=1e-4,
    # ---- cached-score D x J family ablation (enabled by default) ----
    with_factorial_family_ablation=True,
    family_ablation_graphs=256,
    family_size=6,
    family_random_sets=24,  # secondary layer-matched band; generalists are the scientific nulls
    # ---- raw semantic-score outlier ablation (enabled by default; beta) ----
    with_semantic_outlier_ablation=True,
    with_structural_outlier_ablation=True,
    semantic_outlier_graphs=256,
    semantic_outlier_top_k=6,       # applied separately to raw S_sem and raw S_str
    semantic_outlier_random_sets=24,
    semantic_outlier_attention_graphs=4,  # dense only; fixed size-quantile validation examples
    semantic_outlier_attention_heads=6,   # captures all six targets + their six controls
    # A capped path is retained only when both the global <=1% failure-rate gate and this
    # absolute residual gate pass. 1e-2 admits the observed isolated 5.108e-3 ZINC path while
    # still rejecting a materially inaccurate tail; all failures/residuals remain reported.
    integrated_unconverged_error_cap=1e-2,
)

# --- Re-draw the deliverables from cache only (no re-compute), e.g. dropping the VNode runs: ---
# build_figures(drop_vnode=True, display=True)
# build_figures(include=["zinc", "zinc_1hop", "zinc_2hop"], display=True)
#
# --- Diagnose why a model is missing from a figure (checkpoint found? which caches exist?): ---
# from graph_specialisation_metrics.comparison import inventory
# inventory()          # prints a per-model table: ckpt | carriage | scores | chanAbl | val/test
# ============================ paste to here ============================
