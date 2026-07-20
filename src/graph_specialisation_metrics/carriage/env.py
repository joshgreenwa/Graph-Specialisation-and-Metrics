"""Environment setup shared by every carriage notebook.

Runs entirely IN-PROCESS (a pasted/cloned Colab cell has no __file__ to relaunch as a
subprocess), so the Python-3.12 / GRIT-era compatibility fixes are applied directly here
before GRIT is imported. Also: the exact dependency pin set the training runners used,
cloning + pinning the official GRIT source, and discovering the last checkpoint on Drive.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


def log(msg: str) -> None:
    print(msg, flush=True)


class CommandError(RuntimeError):
    pass


def run_cmd(cmd: Sequence[str], *, cwd: Optional[Path] = None, check: bool = True):
    printable = " ".join(map(str, cmd))
    log(f"\n[cmd] {printable}")
    proc = subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None,
                          text=True, check=False)
    if check and proc.returncode != 0:
        raise CommandError(f"Command failed ({proc.returncode}): {printable}")
    return proc


# ======================================================================================
# Python-3.12 / GRIT compatibility (in-process; mirrors the training runner's sitecustomize)
# ======================================================================================

def apply_compat_patches() -> None:
    """Idempotent, no-op where unneeded. Does not alter any GRIT model/config logic.

    Fixes: (1) pkgutil.ImpImporter / FileFinder.find_module removed in 3.12 but referenced
    by old pkg_resources; (2) torch.load weights_only default flipped in torch>=2.6 vs
    PyG-2.2 saved Data / GraphGym checkpoints (trusted, local); (3) sklearn>=1.6 removed
    mean_squared_error(squared=False) that GRIT's logger calls.
    """
    import importlib.machinery
    import pkgutil
    import site
    import sysconfig

    preferred = []
    for _p in list(site.getsitepackages()) + [
        sysconfig.get_paths().get("purelib", ""),
        sysconfig.get_paths().get("platlib", ""),
    ]:
        if _p and os.path.isdir(_p) and "/usr/local/" in _p and _p not in preferred:
            preferred.append(_p)
    for _p in reversed(preferred):
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)

    if not hasattr(pkgutil, "ImpImporter"):
        class _CompatImpImporter:  # pragma: no cover
            pass
        pkgutil.ImpImporter = _CompatImpImporter

    if not hasattr(importlib.machinery.FileFinder, "find_module"):
        def _compat_find_module(self, fullname, path=None):
            spec = self.find_spec(fullname)
            return None if spec is None else spec.loader
        importlib.machinery.FileFinder.find_module = _compat_find_module

    _mod = sys.modules.get("pkg_resources")
    _mod_file = str(getattr(_mod, "__file__", "") or "") if _mod is not None else ""
    if _mod_file.startswith("/usr/lib/python3/dist-packages/"):
        del sys.modules["pkg_resources"]

    try:
        import torch
        if getattr(torch.load, "__name__", "") != "_compat_torch_load":
            _orig = torch.load

            def _compat_torch_load(*a, **kw):
                kw.setdefault("weights_only", False)
                return _orig(*a, **kw)

            torch.load = _compat_torch_load
            log("[compat] torch.load defaults to weights_only=False for trusted local files.")
    except Exception as exc:  # noqa: BLE001
        log(f"[compat-warning] could not patch torch.load: {exc}")

    try:
        import inspect
        import numpy as _np
        import sklearn.metrics as _skm

        if "squared" not in inspect.signature(_skm.mean_squared_error).parameters:
            _orig_mse = _skm.mean_squared_error

            def _compat_mse(y_true, y_pred, *, sample_weight=None,
                            multioutput="uniform_average", squared=True):
                mse = _orig_mse(y_true, y_pred, sample_weight=sample_weight,
                                multioutput=multioutput)
                return mse if squared else _np.sqrt(mse)

            _skm.mean_squared_error = _compat_mse
            log("[compat] restored legacy sklearn mean_squared_error(squared=False).")
    except Exception:  # noqa: BLE001
        pass


# ======================================================================================
# Dependencies -- identical pin set to train_grit_zinc_official_colab.py, plus matplotlib
# ======================================================================================

def install_dependencies(pyg_version: str = "2.2.0") -> None:
    log("\n[deps] Installing Python dependencies (may take a while on a fresh runtime).")
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    import importlib
    torch = importlib.import_module("torch")
    torch_version = str(torch.__version__).split("+")[0]
    cuda = getattr(torch.version, "cuda", None)
    cuda_tag = ("cu" + cuda.replace(".", "")) if cuda else "cpu"
    log(f"[deps] Python {sys.version.split()[0]} | torch {torch.__version__} | CUDA {cuda}")
    wheel = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    log(f"[deps] PyG wheel index: {wheel}")

    for pkg in ("pyg-lib", "torch-spline-conv"):
        p = run_cmd([sys.executable, "-m", "pip", "install", pkg, "-f", wheel], check=False)
        if p.returncode != 0:
            log(f"[deps-warning] {pkg} unavailable for this stack; GRIT ZINC RRWP does not need it.")
    for pkg in ("torch-scatter", "torch-sparse", "torch-cluster"):
        p = run_cmd([sys.executable, "-m", "pip", "install", pkg, "-f", wheel], check=False)
        if p.returncode != 0:
            raise CommandError(f"Required PyG extension {pkg!r} failed to install from {wheel}.")

    run_cmd([sys.executable, "-m", "pip", "install", f"torch-geometric=={pyg_version}"])
    run_cmd([sys.executable, "-m", "pip", "install",
             "yacs==0.1.8", "pytorch-lightning==1.9.5", "torchmetrics==0.9.1",
             "opt_einsum>=3.3", "tensorboardX>=2.6,<2.7", "ogb==1.3.6",
             "wandb>=0.16,<0.18", "pyyaml>=6.0", "scikit-learn>=1.0", "scipy>=1.9",
             "networkx>=2.8", "matplotlib>=3.6"])


# ======================================================================================
# GRIT source
# ======================================================================================

def clone_grit(repo_dir: Path, repo_url: str, commit: Optional[str],
               force_fresh: bool = False) -> str:
    if force_fresh and repo_dir.exists():
        log(f"[repo] Removing existing GRIT: {repo_dir}")
        shutil.rmtree(repo_dir)
    if not (repo_dir / ".git").exists():
        run_cmd(["git", "clone", "--branch", "main", repo_url, str(repo_dir)])
    if commit:
        cur = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
        if cur != commit:
            run_cmd(["git", "-C", str(repo_dir), "fetch", "origin"], check=False)
            run_cmd(["git", "-C", str(repo_dir), "checkout", commit])
    resolved = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    log(f"[repo] GRIT commit: {resolved}")
    return resolved


def prepare_inprocess_grit(repo_dir: Path) -> None:
    """Make `import grit` work in THIS process and match main.py's cwd expectations."""
    import importlib

    apply_compat_patches()
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_dir), "--no-deps"], check=False)
    importlib.invalidate_caches()
    p = str(repo_dir)
    if p not in sys.path:
        sys.path.insert(0, p)
    # GRIT reads a relative --cfg path and sets cfg.work_dir = os.getcwd().
    os.chdir(p)
    # Drop any stale grit modules from a previous run so the fresh checkout is imported.
    for name in [m for m in sys.modules if m == "grit" or m.startswith("grit.")]:
        del sys.modules[name]
    log(f"[compat] In-process GRIT: sys.path[0]={sys.path[0]} | cwd={os.getcwd()}")


# ======================================================================================
# Checkpoint discovery on Drive
# ======================================================================================

def find_checkpoint(results_root: Path, explicit: Optional[str] = None) -> tuple[Path, int]:
    """Locate the checkpoint to analyse.

    GraphGym writes <out>/<cfg>-<tag>/<seed>/ckpt/<epoch>.ckpt with ckpt_best+ckpt_clean,
    so the surviving file is both the last-saved and best-validation checkpoint. We pick
    the newest ckpt directory, then the highest epoch inside it.
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"ckpt does not exist: {p}")
        epoch = int(p.stem) if p.stem.isdigit() else -1
        log(f"[ckpt] Using explicit checkpoint: {p} (epoch={epoch})")
        return p, epoch

    if not results_root.exists():
        raise FileNotFoundError(
            f"Results root not found: {results_root}\n"
            f"Point drive_dir at the same directory the training runner used."
        )
    candidates = sorted(results_root.glob("**/ckpt/*.ckpt"))
    if not candidates:
        # Fallback: the k-hop/VNode ZINC runner (GRIT_khop_ZINC.py) writes stable, Colab-safe
        # recovery copies under results/_recovery_checkpoints/seed<seed>_<name_tag>/ rather than
        # GraphGym's <cfg>/<seed>/ckpt/<epoch>.ckpt. Prefer best.ckpt (best validation) over
        # latest.ckpt. These are byte-identical GraphGym save_ckpt() outputs, so they load the
        # same way; their stem is non-numeric, so the epoch is reported as -1.
        for stem in ("best.ckpt", "latest.ckpt", "first_after_resume.ckpt"):
            rec = sorted(results_root.glob(f"_recovery_checkpoints/**/{stem}"))
            if rec:
                chosen = max(rec, key=lambda p: p.stat().st_mtime)
                log(f"[ckpt] No GraphGym ckpt/ dir; using recovery checkpoint: {chosen} "
                    f"(prefer={stem})")
                return chosen, -1
        raise FileNotFoundError(
            f"No *.ckpt under {results_root}/**/ckpt/ and no "
            f"{results_root}/_recovery_checkpoints/**/(best|latest).ckpt. "
            f"Has training saved a checkpoint?"
        )
    log(f"[ckpt] Found {len(candidates)} checkpoint file(s) under {results_root}:")
    for c in candidates:
        log(f"[ckpt]   {c}  (mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c.stat().st_mtime))})")
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    in_dir = [p for p in candidates if p.parent == newest.parent]
    numeric = [p for p in in_dir if p.stem.isdigit()]
    chosen = max(numeric, key=lambda p: int(p.stem)) if numeric else newest
    epoch = int(chosen.stem) if chosen.stem.isdigit() else -1
    log(f"[ckpt] Selected last checkpoint: {chosen} (epoch={epoch})")
    log("[ckpt] (ckpt_best+ckpt_clean => this is also the best-validation checkpoint)")
    return chosen, epoch


def ensure_repo_root_on_path() -> None:
    """Put the PROJECT repo root on sys.path so `import experiments...` resolves.

    Task env hooks (peptides patches, the 1-hop patch) live under experiments/ in this
    repo. The bootstrap cell adds <repo>/src (the carriage package); this adds <repo> so
    the experiments package is importable too. Derived from this module's own location:
    .../<repo>/src/graph_specialisation_metrics/carriage/env.py -> parents[3] == <repo>.
    """
    root = Path(__file__).resolve().parents[3]
    if (root / "experiments").is_dir():
        p = str(root)
        if p not in sys.path:
            sys.path.insert(0, p)


def resolve_config(task, repo_dir: Path, out_dir: Path) -> str:
    """Return an absolute config path for the task (repo file, or inline text written out)."""
    if task.config_text:
        p = out_dir / "task_config.yaml"
        p.write_text(task.config_text, encoding="utf-8")
        log(f"[config] Wrote inline task config: {p}")
        return str(p)
    cand = Path(task.config_path)
    if not cand.is_file():
        cand = Path(repo_dir) / task.config_path
    if not cand.is_file():
        raise FileNotFoundError(f"Config not found for task {task.name!r}: {task.config_path}")
    return str(cand)
