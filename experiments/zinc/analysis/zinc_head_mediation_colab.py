"""Paste/run this cell in Colab for the dense-ZINC Sx-versus-Mx analysis.

The run is figures-only: it reads the completed canonical dense-ZINC score and
causal caches, loads no model or dataset, and performs no GRIT forward passes.
Figures and tables are saved to Drive and displayed inline.
"""

# ============================ paste from here ============================
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
BRANCH = "expansion/carriage_experiments"
REPO_DIR = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"

DRIVE_ROOT = Path("/content/drive/MyDrive")
CANONICAL_ROOT_CANDIDATES = (
    DRIVE_ROOT / "graph_specialisation_metrics/canonical_methodology_v4_zinc_qm9",
    DRIVE_ROOT / "graph_specialisation_metrics/canonical_methodology",
)
OUTPUT_DIR = DRIVE_ROOT / "graph_specialisation_metrics/zinc_head_mediation"
TRAIN_SEED = 42
TOP_HEADS = 8
BOOTSTRAP_REPLICATES = 2_000
ANALYSIS_SEED = 91_733

from google.colab import drive, userdata  # noqa: E402

drive.mount("/content/drive", force_remount=False)
token = userdata.get(SECRET_NAME)
if not token:
    print(f"[setup:warning] Colab secret {SECRET_NAME!r} is missing; using public clone")
    authenticated = REPO_URL
else:
    suffix = REPO_URL.removeprefix("https://github.com/")
    authenticated = f"https://x-access-token:{token.strip()}@github.com/{suffix}"

if not (REPO_DIR / ".git").is_dir():
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--single-branch", authenticated, str(REPO_DIR)],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "remote", "set-url", "origin", authenticated], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", f"origin/{BRANCH}"], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "remote", "set-url", "origin", REPO_URL], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)], check=True)

source = str(REPO_DIR / "src")
if source not in sys.path:
    sys.path.insert(0, source)
for module_name in tuple(sys.modules):
    if module_name == "graph_specialisation_metrics" or module_name.startswith("graph_specialisation_metrics."):
        del sys.modules[module_name]
os.chdir(REPO_DIR)

def has_required_caches(root: Path) -> bool:
    task_root = root / "zinc" / f"seed_{TRAIN_SEED}" / "cache"
    return (task_root / "scores/raw.pt").is_file() and (task_root / "causal/validation.pt").is_file()

available = [root for root in CANONICAL_ROOT_CANDIDATES if has_required_caches(root)]
canonical_root = available[0] if available else CANONICAL_ROOT_CANDIDATES[0]
if not available:
    print("[cache:warning] No complete dense-ZINC score+causal cache was found in the registered roots.")
    print(f"[cache:warning] Expected under: {canonical_root}/zinc/seed_{TRAIN_SEED}/cache")
else:
    print(f"[cache] {canonical_root}")
print("[scope] Dense ZINC only; cache-only Sx versus symmetric finite head mediation Mx.")
print("[scope] No checkpoint, dataset, RRWP preprocessing, or model forward pass is loaded.")

from graph_specialisation_metrics.methodology.head_mediation import main  # noqa: E402

result = main(
    [
        "--canonical-root", str(canonical_root),
        "--output-dir", str(OUTPUT_DIR),
        "--task", "zinc",
        "--train-seed", str(TRAIN_SEED),
        "--top-heads", str(TOP_HEADS),
        "--bootstrap-replicates", str(BOOTSTRAP_REPLICATES),
        "--analysis-seed", str(ANALYSIS_SEED),
    ]
)

from IPython.display import Image, display  # noqa: E402

for key in ("headline_png", "coordinates_png"):
    path = result.get("figures", {}).get(key)
    if path and Path(path).is_file():
        display(Image(filename=path))

try:
    import pandas as pd

    summary_path = OUTPUT_DIR / "results/channel_summary.csv"
    if summary_path.is_file():
        display(pd.read_csv(summary_path))
except Exception as error:
    print(f"[display:warning] {type(error).__name__}: {error}")

print(f"[saved] {OUTPUT_DIR}")
# ============================= paste to here =============================
