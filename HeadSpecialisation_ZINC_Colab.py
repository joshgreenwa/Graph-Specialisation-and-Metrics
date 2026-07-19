"""
Colab bootstrap cell for the PER-HEAD SPECIALISATION-SCORE analysis on GRIT ZINC.

Paste this whole cell into Colab and run it. Like the carriage bootstrap it is deliberately
tiny and STABLE: it clones this dissertation repo (via the `dissertation_key` Colab secret),
puts <repo>/src on the path, and calls the central methodology. All the science lives in the
repo under src/graph_specialisation_metrics/specialisation/, so editing + pushing there updates
every notebook on its next run -- you do not edit this cell to change methodology.

What it does, for BOTH pretrained ZINC models (dense GRIT and the parameter-matched 1-hop
control), loading each checkpoint from Drive and writing all figures back to Drive:

  * per-head SEMANTIC and STRUCTURAL specialisation scores at the transport site
    (o^{lh}_i = batch.wV), via the separate donor-swap / node-transposition interventions
    (SPECIALISATION_SCORES.md), plus a complementary semantic attention-routing score;
  * (i)   scatter of structural (x) vs semantic (y) score per head, coloured by layer;
  * (ii)  per-head [layer x head] heatmaps of each score;
  * (iii) attention maps of the most interesting heads across several molecules;
  * (iv)  head-ablation: does zeroing the top semantic / structural / joint head hurt the
          output more than a random head? (rank vs the null, per-graph impact distribution,
          and impact-vs-structural-feature correlations).

The two ZINC models share this cell. The 1-hop variant applies the repo's parameter-matched
1-hop patch to its own GRIT clone (masked RRWP edge encoder, sparsity=one_hop) via the same
env hook the carriage runner uses; the analysis is otherwise identical.

GENERAL ACROSS TASKS. The methodology is task-agnostic: pass ANY task registered in
``carriage.tasks`` (zinc, zinc_1hop, peptides_func, peptides_struct, and any GRIT/1-hop model
you register). Multi-target regression (peptides-struct, 11 targets) and multilabel
classification (peptides-func, 10 labels) are handled via the functional-magnitude combination
over the T outputs; the task loss (l1/mse/BCE) drives the ablation impact. To analyse a NEW GRIT
model, add ONE registry entry to ``carriage/tasks.py`` (config, checkpoint drive_dir, content
adapter) and pass its name here -- no notebook or methodology edit needed. Examples:
  run(tasks=["zinc", "zinc_1hop"])                     # the ZINC dense-vs-sparse comparison
  run(tasks=["peptides_struct"], max_sources=32)       # large graphs -> cap sources per graph
  run(tasks=["peptides_func", "peptides_struct"])      # peptides family

Requires the trained checkpoints on Drive at the locations each task's GritTaskSpec names, e.g.:
  dense ZINC: /content/drive/MyDrive/grit_zinc_official/results/**/ckpt/*.ckpt
  1-hop ZINC: /content/drive/MyDrive/grit_zinc_1hop/results/**/ckpt/*.ckpt
Figures collate under /content/drive/MyDrive/graph_specialisation_metrics/specialisation_figures.
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

_src = os.path.join(REPO_DIR, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
# Drop any stale modules from a previous run so the fresh checkout is imported.
for _m in [m for m in list(sys.modules) if m.startswith("graph_specialisation_metrics")]:
    del sys.modules[_m]

from graph_specialisation_metrics.specialisation import run  # noqa: E402

# First run on a fresh runtime: omit skip_install so deps install (~a few minutes).
# On a runtime that already has the GRIT deps (e.g. right after a carriage run), pass
# skip_install=True. Bump num_graphs / donors / ablation_graphs for tighter estimates on an A100.
run(
    tasks=["zinc", "zinc_1hop"],   # ANY registered carriage.tasks; e.g. ["peptides_struct"]
    num_graphs=200,          # graphs scored per model (per-head scores)
    donors=8,                # K donors/partners per source
    max_sources=None,        # None = all nodes; set (e.g. 32) for large-graph tasks like peptides
    with_attn_routing=True,  # also compute the semantic attention-routing (selection) score
    ablation_graphs=256,     # graphs for the head-ablation impact sweep
    n_attention_molecules=5, # molecules shown in the attention grid
    skip_install=False,
    mount=True,
)
# ============================ paste to here ============================
