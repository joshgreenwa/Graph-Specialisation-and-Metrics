"""Colab cell: full dense-vs-1-hop GRIT analysis for the QM9 HOMO-LUMO gap.

Paste this whole file into one Colab cell and run it. It mirrors
``notebook_all_models_cell.py`` stage-for-stage, but uses the two trained QM9 gap models:

* dense GRIT+RRWP, best checkpoint epoch 294;
* exact <=1-hop GRIT+RRWP, best checkpoint epoch 295.

For both checkpoints the analysis replays the checkpoint-compatible ``GRIT_QM9_gap.py`` model
and data patch before loading. Consequently data/evaluation match training exactly: target 4
(HOMO-LUMO gap, raw eV), atomic-number node content, four categorical bond types, RRWP-21, and
the seeded 110000/10000/remainder train/validation/test split. Both tasks use the shared training
dataset cache at ``/content/drive/MyDrive/grit_qm9_gap_data``.

The run recomputes validation/test MAE and mirrors every ZINC analysis stage: semantic and
structural carriage, per-head semantic/structural scores, channel-split causal ablation, matched
D_rel x J family ablation, raw semantic/structural outlier tests, dense outlier examples, and the
all-model specialist molecule gallery. QM9 caches and figures live in their own Drive namespace,
so they cannot overwrite or be mistaken for ZINC results.
"""

# ============================ paste from here ============================
import os
import subprocess
import sys

REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
BRANCH = "codex/cfim-grit-experiments"
REPO_DIR = "/content/Graph-Specialisation-and-Metrics"
SECRET_NAME = "dissertation_key"

QM9_ANALYSIS_ROOT = "/content/drive/MyDrive/graph_specialisation_metrics/qm9_gap"
CARRIAGE_COLLATE = f"{QM9_ANALYSIS_ROOT}/carriage_figures"
SPEC_COLLATE = f"{QM9_ANALYSIS_ROOT}/specialisation_figures"
COMPARISON_DIR = f"{QM9_ANALYSIS_ROOT}/model_comparison"

QM9_TASKS = ["qm9_gap_dense", "qm9_gap_1hop"]
QM9_CKPTS = {
    "qm9_gap_dense": (
        "/content/drive/MyDrive/grit_qm9_gap_dense/results/"
        "qm9-gap-GRIT-RRWP-QM9Gap.dense.GRITwRRWP/0/ckpt/294.ckpt"
    ),
    "qm9_gap_1hop": (
        "/content/drive/MyDrive/grit_qm9_gap_1hop_real/results/"
        "qm9-gap-GRIT-RRWP-QM9Gap.1hop.GRITwRRWP/0/ckpt/295.ckpt"
    ),
}

from google.colab import drive, userdata  # noqa: E402

drive.mount("/content/drive", force_remount=False)

_token = userdata.get(SECRET_NAME)
if not _token:
    raise RuntimeError(f"Colab secret {SECRET_NAME!r} is missing or empty.")
_suffix = REPO_URL.removeprefix("https://github.com/")
_authed = f"https://x-access-token:{_token.strip()}@github.com/{_suffix}"

# Fresh source each run; Drive caches remain incremental and are fingerprinted by each stage.
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", _authed, REPO_DIR],
        check=True,
    )
else:
    subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", _authed], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH], check=True)
    subprocess.run(
        ["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"], check=True
    )
# Never retain the secret-bearing URL in the checkout.
subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", REPO_URL], check=True)

# <repo>/src contains the package; <repo> contains the exact QM9 training patch replayed by the
# task hooks. Purge stale imports if this cell is rerun in a warm runtime.
for _p in (os.path.join(REPO_DIR, "src"), REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _m in [m for m in list(sys.modules) if m.startswith("graph_specialisation_metrics")]:
    del sys.modules[_m]

from graph_specialisation_metrics.comparison import build_figures, inventory, run_all  # noqa: E402

run_all(
    tasks=QM9_TASKS,
    ckpts=QM9_CKPTS,
    carriage_collate=CARRIAGE_COLLATE,
    spec_collate=SPEC_COLLATE,
    comparison_dir=COMPARISON_DIR,
    dataset_label="QM9 HOMO-LUMO gap",
    skip_install=False,  # set True only in a warm runtime with the training dependencies present
    force=False,
    allow_partial=False,
    display=True,
    # ---- channel-split causal ablation -> D/J validation ----
    with_channel_ablation=True,
    channel_ablation_graphs=128,
    channel_ablation_sources=32,
    channel_ablation_donors=16,
    # ---- semantic + structural carriage ----
    carriage_num_graphs=128,
    carriage_donors=64,
    beneficial_denom="integrated",
    integrated_atol=1e-4,
    integrated_max_intervals=256,
    integrated_max_unconverged_fraction=1e-2,
    # QM9 loss is raw eV. Keep a task-scale-aware numerical gate instead of copying the looser
    # 1e-2 ZINC exception; capped paths above 5e-4 eV abort and remain fully auditable.
    integrated_unconverged_error_cap=5e-4,
    # ---- per-head semantic/structural scores ----
    spec_num_graphs=128,
    spec_donors=64,
    with_attn_routing=True,
    # ---- cached-score D_rel x J family ablation ----
    with_factorial_family_ablation=True,
    family_ablation_graphs=256,
    family_size=6,
    family_random_sets=24,
    # ---- raw semantic/structural score outlier tests ----
    with_semantic_outlier_ablation=True,
    with_structural_outlier_ablation=True,
    semantic_outlier_graphs=256,
    semantic_outlier_top_k=6,
    semantic_outlier_random_sets=24,
    semantic_outlier_attention_graphs=4,
    semantic_outlier_attention_heads=6,
    semantic_outlier_attention_max_nodes=18,
    semantic_outlier_attention_candidates=32,
    semantic_outlier_attention_score_donors=32,
    # ---- separate dense + 1-hop specialist molecule gallery ----
    with_specialist_molecule_examples=True,
    specialist_molecule_heads_per_channel=3,
    specialist_molecule_examples_per_head=4,
    specialist_molecule_max_nodes=18,
    specialist_molecule_candidates=32,
    specialist_molecule_score_donors=32,
)

# Cache-only redraw (no checkpoint/model load):
# build_figures(
#     tasks=QM9_TASKS,
#     carriage_collate=CARRIAGE_COLLATE,
#     spec_collate=SPEC_COLLATE,
#     comparison_dir=COMPARISON_DIR,
#     dataset_label="QM9 HOMO-LUMO gap",
#     display=True,
# )

# Checkpoint/cache diagnosis:
# inventory(
#     tasks=QM9_TASKS,
#     carriage_collate=CARRIAGE_COLLATE,
#     spec_collate=SPEC_COLLATE,
#     mount=False,
# )
# ============================ paste to here ============================
