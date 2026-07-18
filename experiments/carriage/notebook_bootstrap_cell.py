"""
Colab bootstrap cell for the CENTRAL carriage methodology.

Paste this whole cell into Colab and run it. It is deliberately tiny and STABLE: it clones
this dissertation repo (via the `dissertation_key` Colab secret), puts <repo>/src on the
path, and calls the central methodology. All the science lives in the repo under
src/graph_specialisation_metrics/carriage/, so editing + pushing there updates every
notebook on its next run -- you do not edit this cell to change methodology.

Reproduces the ZINC result with: local version installs, GRIT checkpoint loaded from Drive,
figures written to Drive (collated under one root for all tasks).

Two ZINC models share this cell -- set TASK:
  * "zinc"       official dense GRIT+RRWP (full attention).
  * "zinc_1hop"  parameter-matched 1-hop control. The methodology is identical (same ZINC
                 scalar regression); the only difference is an env hook that applies the
                 repo's 1-hop patch to the GRIT clone (masked RRWP edge encoder,
                 sparsity=one_hop) so the checkpoint's masking is active. Its checkpoints
                 live in /content/drive/MyDrive/grit_zinc_1hop.

To analyse a different GRIT model, register it once in src/.../carriage/tasks.py and call
run(task="<name>"). Everything else is identical.

Aggregation options (both default to the improved behaviour, method-wide):
  * beneficial_denom="magnitude" (default): B[i,j] = dL_j * |C[i,j]| / sum_i |C[i,j]|.
    Convex shares => |B| <= |dL_j| (no blow-up) and B=0 where functional carriage=0. Pass
    run(..., beneficial_denom="signed") for the legacy signed-sum share (kept for comparison).
  * bin_strategy="log" (default) + central="trimmed": F/B pooled into adaptive SPD bins
    ({0},{1},{2},{3},{4-7},{8-15},...) with a robust central tendency + graph-clustered CI.
    Pass bin_strategy="hop" to recover per-hop resolution (fine on ZINC's small diameter).
"""

# ============================ paste from here ============================
import os
import subprocess
import sys

REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
BRANCH = "codex/cfim-grit-experiments"
REPO_DIR = "/content/Graph-Specialisation-and-Metrics"
SECRET_NAME = "dissertation_key"
TASK = "zinc"          # <- set to "zinc_1hop" for the 1-hop masked control

from google.colab import drive, userdata  # noqa: E402

drive.mount("/content/drive", force_remount=False)

_token = userdata.get(SECRET_NAME)
if not _token:
    raise RuntimeError(f"Colab secret {SECRET_NAME!r} is missing or empty.")
_suffix = REPO_URL.removeprefix("https://github.com/")
_authed = f"https://x-access-token:{_token.strip()}@github.com/{_suffix}"

# Fresh checkout each run so central methodology edits always take effect. `reset --hard`
# guarantees the tracked source matches the pushed branch exactly (a stale/dirty clone
# would otherwise silently run OLD code).
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
# Drop any stale carriage modules from a previous run so the fresh checkout is imported.
for _m in [m for m in list(sys.modules) if m.startswith("graph_specialisation_metrics")]:
    del sys.modules[_m]

from graph_specialisation_metrics.carriage import run  # noqa: E402

# First run on a fresh runtime: omit skip_install so deps install (~a few minutes).
# On a runtime that already has the GRIT deps (e.g. right after a training run), pass
# skip_install=True. Bump num_graphs / donors for tighter CIs on the A100.
run(task=TASK, mount=False)
# run(task=TASK, mount=False, skip_install=True, num_graphs=128, donors=64)
# ============================ paste to here ============================
