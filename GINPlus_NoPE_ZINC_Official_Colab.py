#!/usr/bin/env python3
"""
Colab runner for the official GNN+ / GIN+ no-PE ablation on ZINC.

This script intentionally does not reimplement GIN+. It clones the official
GNNPlus repository from the ICML 2025 paper:

    Can Classic GNNs Be Strong Baselines for Graph-level Tasks?
    https://arxiv.org/abs/2502.09263
    https://github.com/LUOyk1999/GNNPlus

and first validates the untouched official ZINC GINE/GIN+ configuration:

    python -u main.py --cfg configs/gine/zinc.yaml

It then derives a separate no-PE config which changes only:

    dataset.node_encoder_name: TypeDictNode
    posenc_RWSE.enable: False

The official source and original YAML remain untouched. All other model,
dataset, optimizer, scheduler, depth, hidden dimension, and 2000-epoch settings
come from the official config. The runner uses a separate Drive namespace and
enforces the expected no-PE parameter count (475,521) before training proceeds.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Mapping, Sequence


OFFICIAL_REPO = "https://github.com/LUOyk1999/GNNPlus.git"
OFFICIAL_CFG = "configs/gine/zinc.yaml"
NOPE_CFG = "configs/gine/zinc-nope-ablation.yaml"
OFFICIAL_COMMIT = "0e02ad9acc2f1e54b5ad71c051bf5dfb1fcb4f28"
EXPECTED_NOPE_PARAMS = 475_521

EXPECTED_CFG_VALUES = {
    ("metric_best",): "mae",
    ("metric_agg",): "argmin",
    ("dataset", "format"): "PyG-ZINC",
    ("dataset", "name"): "subset",
    ("dataset", "task"): "graph",
    ("dataset", "task_type"): "regression",
    ("dataset", "transductive"): False,
    ("dataset", "node_encoder"): True,
    ("dataset", "node_encoder_name"): "TypeDictNode+RWSE",
    ("dataset", "node_encoder_num_types"): 21,
    ("dataset", "node_encoder_bn"): False,
    ("dataset", "edge_encoder"): True,
    ("dataset", "edge_encoder_name"): "TypeDictEdge",
    ("dataset", "edge_encoder_num_types"): 4,
    ("dataset", "edge_encoder_bn"): False,
    ("posenc_RWSE", "enable"): True,
    ("posenc_RWSE", "kernel", "times_func"): "range(1,21)",
    ("posenc_RWSE", "model"): "Linear",
    ("posenc_RWSE", "dim_pe"): 28,
    ("posenc_RWSE", "raw_norm_type"): "BatchNorm",
    ("train", "mode"): "custom",
    ("train", "batch_size"): 32,
    ("train", "eval_period"): 1,
    ("train", "ckpt_period"): 100,
    ("model", "type"): "custom_gnn",
    ("model", "loss_fun"): "l1",
    ("model", "edge_decoding"): "dot",
    ("model", "graph_pooling"): "add",
    ("gnn", "head"): "san_graph",
    ("gnn", "layer_type"): "gine",
    ("gnn", "layers_mp"): 12,
    ("gnn", "layers_pre_mp"): 0,
    ("gnn", "layers_post_mp"): 3,
    ("gnn", "dim_inner"): 80,
    ("gnn", "act"): "relu",
    ("gnn", "dropout"): 0.0,
    ("gnn", "agg"): "mean",
    ("gnn", "normalize_adj"): False,
    ("gnn", "ffn"): True,
    ("gnn", "residual"): True,
    ("optim", "clip_grad_norm"): True,
    ("optim", "optimizer"): "adamW",
    ("optim", "weight_decay"): 1e-5,
    ("optim", "base_lr"): 1e-3,
    ("optim", "max_epoch"): 2000,
    ("optim", "scheduler"): "cosine_with_warmup",
    ("optim", "num_warmup_epochs"): 50,
    ("optim", "min_lr"): 1e-6,
}

EXPECTED_NOPE_CFG_VALUES = dict(EXPECTED_CFG_VALUES)
EXPECTED_NOPE_CFG_VALUES.update({
    ("dataset", "node_encoder_name"): "TypeDictNode",
    ("posenc_RWSE", "enable"): False,
})


class CommandError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    printable = " ".join(map(str, cmd))
    log(f"\n[cmd] {printable}")
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise CommandError(f"Command failed with exit code {proc.returncode}: {printable}")
    return proc


class ConsoleFilter:
    """Compact notebook stdout while preserving the full raw Drive log."""

    def __init__(self, *, verbosity: str, epoch_period: int) -> None:
        self.verbosity = verbosity
        self.epoch_period = max(1, int(epoch_period))
        self._suppress_until_num_params = False
        self._in_traceback = False
        self._epoch_stats: dict[int, dict[str, dict]] = {}
        self._best_epoch: int | None = None
        self._best_val_mae = float("inf")
        self._best_test_mae = float("nan")

    @staticmethod
    def _fmt(value, digits: int = 5) -> str:
        try:
            v = float(value)
        except Exception:
            return "nan"
        if v != v:
            return "nan"
        return f"{v:.{digits}f}"

    def _parse_split_stats(self, stripped: str) -> bool:
        for split in ("train", "val", "test"):
            prefix = f"{split}: "
            if not stripped.startswith(prefix):
                continue
            payload = stripped[len(prefix):]
            if not payload.startswith("{"):
                return True
            try:
                stats = ast.literal_eval(payload)
                epoch = int(stats.get("epoch"))
            except Exception:
                return True
            self._epoch_stats.setdefault(epoch, {})[split] = stats
            return True
        return False

    def _update_best(self, epoch: int) -> None:
        val = self._epoch_stats.get(epoch, {}).get("val")
        if not val:
            return
        val_mae = float(val.get("mae", val.get("loss", float("inf"))))
        if val_mae < self._best_val_mae:
            self._best_epoch = epoch
            self._best_val_mae = val_mae
            self._best_test_mae = float(
                self._epoch_stats.get(epoch, {}).get("test", {}).get("mae", float("nan"))
            )

    def _emit_epoch_line(self, stripped: str) -> None:
        match = re.search(r"> Epoch\s+(\d+):", stripped)
        if not match:
            return
        epoch = int(match.group(1))
        if epoch >= 2 and ((epoch + 1) % self.epoch_period != 0):
            return
        self._update_best(epoch)
        splits = self._epoch_stats.get(epoch, {})
        train = splits.get("train", {})
        val = splits.get("val", {})
        test = splits.get("test", {})
        m_time = re.search(r"took\s+([0-9.]+)s\s+\(avg\s+([0-9.]+)s\)", stripped)
        took = m_time.group(1) if m_time else "?"
        avg = m_time.group(2) if m_time else "?"
        best_epoch = "?" if self._best_epoch is None else str(self._best_epoch)
        print(
            f"[epoch {epoch:04d}] "
            f"train_loss={self._fmt(train.get('loss'))} train_mae={self._fmt(train.get('mae'))} | "
            f"val_loss={self._fmt(val.get('loss'))} val_mae={self._fmt(val.get('mae'))} | "
            f"test_loss={self._fmt(test.get('loss'))} test_mae={self._fmt(test.get('mae'))} | "
            f"best@{best_epoch} val_mae={self._fmt(self._best_val_mae)} "
            f"test_mae={self._fmt(self._best_test_mae)} | time={took}s avg={avg}s",
            flush=True,
        )

    def should_print(self, line: str) -> bool:
        if self.verbosity == "full":
            return True

        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith("Traceback"):
            self._in_traceback = True
            return True
        if self._in_traceback:
            return True
        if any(tok in stripped for tok in ("ERROR", "Error:", "Exception", "RuntimeError", "TypeError", "AttributeError")):
            return True

        if self._parse_split_stats(stripped):
            return False

        if stripped.startswith(("custom_gnn(", "CustomGNN(")):
            self._suppress_until_num_params = True
            return False
        if self._suppress_until_num_params:
            if re.search(r"Num parameters:\s*[0-9,]+", stripped):
                self._suppress_until_num_params = False
                return True
            return False

        if re.search(r"> Epoch\s+(\d+):", stripped):
            self._emit_epoch_line(stripped)
            return False

        noisy_fragments = (
            "it/s]", "Processing train dataset", "Processing val dataset",
            "Processing test dataset", "UserWarning:", "Downloading ",
            "Extracting ", "Processing...", "Done!",
        )
        if any(fragment in stripped for fragment in noisy_fragments):
            return False

        setup_prefixes = (
            "[*] Run ID", "Starting now:", "[*] Loaded dataset", "Data(",
            "undirected:", "num graphs:", "avg num_nodes/graph:",
            "num node features:", "num edge features:", "num classes:",
            "Parsed RWSE", "Precomputing Positional Encoding statistics",
            "Start from epoch", "Checkpoint found", "Task done",
            "Avg time per epoch", "Total train loop time", "Num parameters:",
        )
        if stripped.startswith(setup_prefixes):
            return True
        return self.verbosity == "standard"


def run_streaming_to_console_and_log(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_file: Path,
    env: Mapping[str, str] | None,
    console_verbosity: str,
    console_epoch_period: int,
    expected_params: int,
) -> int:
    printable = " ".join(map(str, cmd))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"\n[train-cmd] {printable}")
    log(f"[train-log] {log_file}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env["PYTHONUNBUFFERED"] = "1"

    filt = ConsoleFilter(verbosity=console_verbosity, epoch_period=console_epoch_period)
    observed_params: int | None = None
    start = time.perf_counter()
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Launched at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {printable}\n")
        f.write("=" * 100 + "\n")
        f.flush()
        proc = subprocess.Popen(
            list(map(str, cmd)),
            cwd=str(cwd),
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        if console_verbosity != "full":
            msg = f"[console] {console_verbosity} stdout enabled; full raw GNN+ output is saved to {log_file}\n"
            print(msg, end="", flush=True)
            f.write(msg)
        for line in proc.stdout:
            f.write(line)
            f.flush()
            match = re.search(r"Num parameters:\s*([0-9,]+)", line)
            if match:
                observed_params = int(match.group(1).replace(",", ""))
                if observed_params != expected_params:
                    message = (
                        f"\n[param-error] Effective model has {observed_params:,} "
                        f"parameters; expected no-PE GIN+ {expected_params:,}. "
                        "Terminating before an epoch is trained.\n"
                    )
                    print(message, end="", flush=True)
                    f.write(message)
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return 97
            if filt.should_print(line):
                print(line, end="", flush=True)
        rc = proc.wait()
        elapsed = time.perf_counter() - start
        f.write(f"\nReturn code: {rc}\nElapsed seconds: {elapsed:.2f}\n")
    if rc == 0 and observed_params is None:
        log("[param-error] Training exited without reporting a parameter count.")
        return 98
    return rc


def write_py312_compat_shim(base_dir: Path) -> Path:
    shim_dir = base_dir / "python312_compat"
    shim_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = shim_dir / "sitecustomize.py"
    sitecustomize.write_text(
        """
import importlib.machinery
import os
import pkgutil
import site
import sys
import sysconfig

_preferred = []
for _p in list(site.getsitepackages()) + [
    sysconfig.get_paths().get("purelib", ""),
    sysconfig.get_paths().get("platlib", ""),
]:
    if _p and os.path.isdir(_p) and "/usr/local/" in _p and _p not in _preferred:
        _preferred.append(_p)
for _p in reversed(_preferred):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(1, _p)

if not hasattr(pkgutil, "ImpImporter"):
    class _CompatImpImporter:
        pass
    pkgutil.ImpImporter = _CompatImpImporter

if not hasattr(importlib.machinery.FileFinder, "find_module"):
    def _compat_find_module(self, fullname, path=None):
        spec = self.find_spec(fullname)
        return None if spec is None else spec.loader
    importlib.machinery.FileFinder.find_module = _compat_find_module

try:
    import torch
    _orig_torch_load = torch.load
    def _compat_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    if getattr(torch.load, "__name__", "") != "_compat_torch_load":
        torch.load = _compat_torch_load
except Exception:
    pass

try:
    import inspect
    import numpy as _np
    import sklearn.metrics as _sk_metrics
    _mse_sig = inspect.signature(_sk_metrics.mean_squared_error)
    if "squared" not in _mse_sig.parameters:
        _orig_mean_squared_error = _sk_metrics.mean_squared_error
        def _compat_mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput="uniform_average", squared=True):
            mse = _orig_mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
            return mse if squared else _np.sqrt(mse)
        _sk_metrics.mean_squared_error = _compat_mean_squared_error
except Exception:
    pass
""".lstrip(),
        encoding="utf-8",
    )
    log(f"[compat] Wrote Python 3.12 compatibility shim: {sitecustomize}")
    return shim_dir


def env_with_py312_compat(shim_dir: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if shim_dir is not None:
        old = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(shim_dir) + ((os.pathsep + old) if old else "")
    return env


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore # noqa: F401
        return True
    except Exception:
        return False


def mount_drive(mount_point: Path) -> None:
    if in_colab():
        from google.colab import drive  # type: ignore
        if mount_point.exists() and any(mount_point.iterdir()):
            log(f"[drive] Google Drive already mounted at {mount_point}.")
        else:
            log(f"[drive] Mounting Google Drive at {mount_point} ...")
            drive.mount(str(mount_point), force_remount=False)
    else:
        log("[drive] google.colab is unavailable; assuming a non-Colab run.")
        mount_point.mkdir(parents=True, exist_ok=True)


def pip_install(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return run_cmd([sys.executable, "-m", "pip", "install", *args], check=check)


def install_dependencies(args: argparse.Namespace) -> None:
    log("\n[deps] Installing Python dependencies for official GNNPlus.")
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    import importlib
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError("PyTorch is not installed/importable in this runtime.") from exc

    torch_version = str(torch.__version__).split("+")[0]
    cuda_version = getattr(torch.version, "cuda", None)
    cuda_tag = "cu" + cuda_version.replace(".", "") if cuda_version else "cpu"
    pyg_wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    log(f"[deps] Python: {sys.version.split()[0]} | torch: {torch.__version__} | CUDA: {cuda_version}")
    log(f"[deps] PyG wheel index: {pyg_wheel_url}")

    for package in ("pyg-lib", "torch-spline-conv"):
        proc = pip_install([package, "-f", pyg_wheel_url], check=False)
        if proc.returncode != 0:
            log(f"[deps-warning] {package} wheel unavailable; continuing because GIN+ ZINC does not require it directly.")

    for package in ("torch-scatter", "torch-sparse", "torch-cluster"):
        proc = pip_install([package, "-f", pyg_wheel_url], check=False)
        if proc.returncode != 0:
            raise CommandError(
                f"Required PyG extension {package!r} failed to install from {pyg_wheel_url}."
            )

    pip_install([f"torch-geometric=={args.pyg_version}"])
    pip_install([
        "yacs>=0.1.8",
        "pytorch-lightning>=1.9,<2.3",
        "torchmetrics>=0.9,<1.5",
        "tensorboardX>=2.6,<2.7",
        "ogb>=1.3.6",
        "wandb>=0.16,<0.18",
        "pyyaml>=6.0",
        "scikit-learn>=1.4",
        "scipy>=1.9",
        "networkx>=2.8",
        "rdkit",
        "fsspec",
    ])


def clone_or_update_repo(repo_dir: Path, repo_url: str, branch: str, commit: str | None, force_fresh: bool) -> str:
    if force_fresh and repo_dir.exists():
        log(f"[repo] Removing existing repo: {repo_dir}")
        shutil.rmtree(repo_dir)
    if not repo_dir.exists():
        run_cmd(["git", "clone", "--branch", branch, repo_url, str(repo_dir)])
    else:
        log(f"[repo] Existing repo found: {repo_dir}")
        run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
        run_cmd(["git", "checkout", branch], cwd=repo_dir)
        run_cmd(["git", "pull", "--ff-only", "origin", branch], cwd=repo_dir)
    if commit:
        log(f"[repo] Pinning GNNPlus to commit: {commit}")
        run_cmd(["git", "checkout", commit], cwd=repo_dir)
    resolved_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    log(f"[repo] Using GNNPlus commit: {resolved_commit}")
    return resolved_commit


def install_gnnplus_editable(repo_dir: Path) -> None:
    pip_install(["-e", str(repo_dir)])


def apply_nope_edge_attr_compat_patch(repo_dir: Path, drive_dir: Path) -> None:
    """Allow TypeDictEdgeEncoder to consume raw 1-D ZINC bond labels.

    The upstream implementation always indexes edge_attr[:, 0] because its
    released ZINC config enables RWSE, whose preprocessing expands edge_attr to
    two dimensions. In the paper's -PE ablation, raw ZINC edge_attr remains
    one-dimensional. This shape-only compatibility patch preserves the released
    two-dimensional path and adds the equivalent one-dimensional path.
    """
    path = repo_dir / "GNNPlus" / "encoder" / "type_dict_encoder.py"
    marker = "# GINPLUS_NOPE_1D_EDGE_ATTR_COMPAT"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        log(f"[patch] No-PE 1-D ZINC edge_attr compatibility already active: {path}")
        return

    old = """\
        batch.edge_attr = self.encoder(batch.edge_attr[:,0].long())
        
        if cfg.dataset.node_encoder_name == 'TypeDictNode+RWSE':
"""
    new = f"""\
        {marker}
        edge_type = batch.edge_attr if batch.edge_attr.dim() == 1 else batch.edge_attr[:, 0]
        batch.edge_attr = self.encoder(edge_type.long())
        
        if cfg.dataset.node_encoder_name == 'TypeDictNode+RWSE':
"""
    if old not in text:
        raise RuntimeError(
            "Could not apply the audited no-PE edge_attr compatibility patch; "
            f"the pinned upstream source has an unexpected shape: {path}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

    patch_dir = drive_dir / "derived_configs"
    patch_dir.mkdir(parents=True, exist_ok=True)
    note = patch_dir / "nope_edge_attr_compat_patch.txt"
    note.write_text(
        "\n".join(
            [
                "Pinned upstream file: GNNPlus/encoder/type_dict_encoder.py",
                "Scientific condition: GIN+ -PE ablation on ZINC",
                "Reason: raw ZINC edge_attr is 1-D when RWSE preprocessing is disabled.",
                "Change: select edge_attr directly when 1-D; retain edge_attr[:, 0] when 2-D.",
                "Trainable parameter change: none.",
                "Feature change beyond disabling PE: none.",
                "",
                "Replacement:",
                new.rstrip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(f"[patch] Applied no-PE 1-D ZINC edge_attr compatibility: {path}")
    log(f"[patch] Audit note: {note}")


def get_nested(d: Mapping, path: Sequence[str]):
    cur = d
    for key in path:
        cur = cur[key]
    return cur


def _semantic_cfg_equal(actual, expected) -> bool:
    if actual == expected:
        return True
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) <= max(1e-12, 1e-9 * abs(float(expected)))
        except Exception:
            return False
    return False


def validate_official_config(repo_dir: Path, allow_drift: bool) -> None:
    import yaml

    cfg_path = repo_dir / OFFICIAL_CFG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Official GIN+ ZINC config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    errors: list[str] = []
    for path, expected in EXPECTED_CFG_VALUES.items():
        try:
            actual = get_nested(cfg, path)
        except Exception:
            errors.append(f"missing {'.'.join(path)}; expected {expected!r}")
            continue
        if not _semantic_cfg_equal(actual, expected):
            errors.append(f"{'.'.join(path)} = {actual!r}; expected {expected!r}")
    if errors:
        msg = "Official GNNPlus GIN+ ZINC config does not match expected values:\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        if allow_drift:
            log("[config-warning] " + msg)
        else:
            raise RuntimeError(
                msg
                + "\nThe repository is pinned, so upstream drift is not permitted "
                "for this ablation runner."
            )
    log("[config] Untouched upstream GNNPlus GIN+ ZINC config validated.")
    log("[config] Upstream baseline correctly contains TypeDictNode+RWSE with posenc_RWSE.enable=True.")


def prepare_and_validate_nope_config(repo_dir: Path, drive_dir: Path) -> Path:
    """Derive the paper's -PE ablation after validating the pristine YAML."""
    import yaml

    official_path = repo_dir / OFFICIAL_CFG
    cfg = yaml.safe_load(official_path.read_text(encoding="utf-8"))
    cfg["dataset"]["node_encoder_name"] = "TypeDictNode"
    cfg["posenc_RWSE"]["enable"] = False

    derived_path = repo_dir / NOPE_CFG
    derived_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False),
        encoding="utf-8",
    )

    errors: list[str] = []
    for path, expected in EXPECTED_NOPE_CFG_VALUES.items():
        try:
            actual = get_nested(cfg, path)
        except Exception:
            errors.append(f"missing {'.'.join(path)}; expected {expected!r}")
            continue
        if not _semantic_cfg_equal(actual, expected):
            errors.append(f"{'.'.join(path)} = {actual!r}; expected {expected!r}")
    if errors:
        raise RuntimeError(
            "Derived GIN+ no-PE config failed validation:\n"
            + "\n".join(f"  - {error}" for error in errors)
        )

    provenance_dir = drive_dir / "derived_configs"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    snapshot = provenance_dir / "zinc-nope-ablation.yaml"
    snapshot.write_text(derived_path.read_text(encoding="utf-8"), encoding="utf-8")
    log(f"[config] Derived and validated no-PE config: {derived_path}")
    log("[config] Scientific intervention: TypeDictNode+RWSE -> TypeDictNode; RWSE enabled -> False.")
    log("[config] Expected no-PE parameter count: 475,521.")
    log(f"[config] Drive snapshot: {snapshot}")
    return derived_path


def build_training_command(
    args: argparse.Namespace,
    drive_dir: Path,
    nope_cfg_path: Path,
) -> List[str]:
    results_dir = drive_dir / "results"
    dataset_dir = drive_dir / "datasets"
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--cfg",
        str(nope_cfg_path),
        "--repeat",
        str(args.repeat),
        "seed",
        str(args.seed),
        "out_dir",
        str(results_dir),
        "dataset.dir",
        str(dataset_dir),
        "name_tag",
        args.name_tag,
        "wandb.use",
        "True" if args.wandb else "False",
        "optim.max_epoch",
        str(args.max_epoch),
        "train.eval_period",
        "1",
        "train.enable_ckpt",
        "True",
        "train.ckpt_period",
        str(args.ckpt_period),
        "train.auto_resume",
        "True" if args.auto_resume else "False",
        "num_threads",
        str(args.num_threads),
    ]
    if args.save_best_checkpoints:
        cmd.extend([
            "train.ckpt_best",
            "True",
            "train.ckpt_clean",
            "False",
        ])
    if args.accelerator:
        cmd.extend(["accelerator", args.accelerator])
    return cmd


def checkpoint_candidates(root: Path) -> list[Path]:
    patterns = ["*.ckpt", "*checkpoint*.pt", "*checkpoint*.pth", "best*.pt", "best*.pth", "model*.pt"]
    out: list[Path] = []
    if root.exists():
        for pattern in patterns:
            out.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(out), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)


def checkpoint_epoch(path: Path) -> int | None:
    text = str(path).lower()
    for pattern in (r"epoch[=_-]?(\d+)", r"ep[=_-]?(\d+)", r"ckpt[=_-]?(\d+)", r"/(\d+)\.ckpt$"):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    try:
        return int(path.stem)
    except Exception:
        return None


def validate_resume_checkpoint_namespace(args: argparse.Namespace) -> None:
    """Refuse to load a full-PE checkpoint into the no-PE experiment."""
    if not args.auto_resume:
        log("[resume] Disabled for this launch; seed will start at epoch 0.")
        return
    checkpoints = checkpoint_candidates(args.drive_dir / "results")
    if not checkpoints:
        log("[resume] Enabled, but no checkpoint exists in the separate no-PE namespace.")
        return

    import torch

    latest = checkpoints[0]
    try:
        payload = torch.load(latest, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(latest, map_location="cpu")
    state = payload.get("model_state", payload) if isinstance(payload, dict) else {}
    keys = tuple(state.keys()) if isinstance(state, dict) else ()
    full_pe_markers = (
        ".node_encoder.encoder1." in key
        or ".node_encoder.encoder2." in key
        or ".edge_encoder.encoder2." in key
        for key in keys
    )
    if any(full_pe_markers):
        raise RuntimeError(
            "Refusing auto-resume: the newest checkpoint contains full-PE/RWSE "
            f"encoder modules: {latest}. Use the separate no-PE Drive root or "
            "launch with --no-auto-resume."
        )
    if not any(".node_encoder.encoder.weight" in key for key in keys):
        raise RuntimeError(
            "Refusing auto-resume because checkpoint architecture could not be "
            f"verified as GIN+ no-PE: {latest}"
        )
    log(f"[resume] Verified compatible no-PE checkpoint: {latest}")


def prepare_results_namespace(args: argparse.Namespace) -> None:
    """Keep a fresh no-resume launch separate from any earlier attempt."""
    if args.auto_resume:
        return
    results_dir = args.drive_dir / "results"
    if not results_dir.exists() or not any(results_dir.iterdir()):
        results_dir.mkdir(parents=True, exist_ok=True)
        return
    archive_root = args.drive_dir / "attempt_archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive = archive_root / f"results_before_{timestamp}"
    suffix = 1
    while archive.exists():
        archive = archive_root / f"results_before_{timestamp}_{suffix}"
        suffix += 1
    shutil.move(str(results_dir), str(archive))
    results_dir.mkdir(parents=True, exist_ok=True)
    log(f"[recovery] Archived previous non-resumed results to: {archive}")


def prepare_legacy_results_link(repo_dir: Path, drive_dir: Path) -> Path:
    """Route the upstream trainer's hard-coded ./results write to Drive."""
    target = drive_dir / "results"
    target.mkdir(parents=True, exist_ok=True)
    legacy = repo_dir / "results"
    if legacy.is_symlink():
        if legacy.resolve() == target.resolve():
            log(f"[storage] Legacy results path already routes to Drive: {legacy}")
            return legacy
        legacy.unlink()
    elif legacy.exists():
        archive_root = drive_dir / "attempt_archives"
        archive_root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive = archive_root / f"repo_relative_results_{timestamp}"
        suffix = 1
        while archive.exists():
            archive = archive_root / f"repo_relative_results_{timestamp}_{suffix}"
            suffix += 1
        shutil.move(str(legacy), str(archive))
        log(f"[storage] Archived pre-existing repository-relative results: {archive}")
    os.symlink(target, legacy, target_is_directory=True)
    log(f"[storage] Routed upstream hard-coded {legacy} -> {target}")
    return legacy


def parse_training_summary_from_logs(log_paths: Sequence[Path]) -> dict:
    by_epoch: dict[int, dict[str, dict]] = {}
    param_count = None
    for path in log_paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            match = re.search(r"Num parameters:\s*([0-9,]+)", line)
            if match:
                param_count = int(match.group(1).replace(",", ""))
            stripped = line.strip()
            for split in ("train", "val", "test"):
                prefix = f"{split}: "
                if not stripped.startswith(prefix):
                    continue
                payload = stripped[len(prefix):]
                if not payload.startswith("{"):
                    continue
                try:
                    stats = ast.literal_eval(payload)
                    epoch = int(stats.get("epoch"))
                except Exception:
                    continue
                by_epoch.setdefault(epoch, {})[split] = stats

    best_epoch = None
    best_val_mae = float("inf")
    for epoch, splits in sorted(by_epoch.items()):
        val = splits.get("val", {})
        try:
            val_mae = float(val.get("mae", val.get("loss", float("inf"))))
        except Exception:
            val_mae = float("inf")
        if val_mae < best_val_mae:
            best_epoch = epoch
            best_val_mae = val_mae
    best_splits = by_epoch.get(best_epoch, {}) if best_epoch is not None else {}
    return {
        "history_epochs": len(by_epoch),
        "max_logged_epoch": max(by_epoch) if by_epoch else None,
        "best_epoch": best_epoch,
        "best_train": best_splits.get("train", {}),
        "best_val": best_splits.get("val", {}),
        "best_test": best_splits.get("test", {}),
        "param_count": param_count,
        "log_files": [str(path) for path in log_paths],
    }


def write_run_manifest(args: argparse.Namespace, drive_dir: Path, repo_commit: str, wrapper_log: Path) -> Path:
    result_root = drive_dir / "results"
    log_paths = [wrapper_log] if wrapper_log.exists() else []
    log_paths.extend(path for path in (drive_dir / "wrapper_logs").rglob("*.log") if path.is_file())
    log_paths = sorted(set(log_paths), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    summary = parse_training_summary_from_logs(log_paths)
    checkpoints = checkpoint_candidates(result_root)
    manifest = {
        "runner": "GINPlus_NoPE_ZINC_Official_Colab.py",
        "official_repo": OFFICIAL_REPO,
        "official_commit": repo_commit,
        "official_source_config": OFFICIAL_CFG,
        "effective_derived_config": NOPE_CFG,
        "paper": "arXiv:2502.09263",
        "model": "GIN+ / GINE (-PE ablation)",
        "dataset": "ZINC subset",
        "expected_parameter_count": EXPECTED_NOPE_PARAMS,
        "seed": args.seed,
        "name_tag": args.name_tag,
        "drive_dir": str(drive_dir),
        "param_count_from_log": summary.get("param_count"),
        "best_epoch_from_logs": summary.get("best_epoch"),
        "best_val": summary.get("best_val", {}),
        "best_test": summary.get("best_test", {}),
        "latest_checkpoint_by_mtime": str(checkpoints[0]) if checkpoints else None,
        "latest_checkpoint_epoch": checkpoint_epoch(checkpoints[0]) if checkpoints else None,
        "checkpoint_candidates": [
            {
                "path": str(path),
                "epoch": checkpoint_epoch(path),
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
                "bytes": path.stat().st_size,
            }
            for path in checkpoints
        ],
        "parsed_logs": summary.get("log_files", []),
        "history_epochs": summary.get("history_epochs", 0),
    }
    manifest_path = drive_dir / f"ginplus_zinc_manifest_seed{args.seed}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    latest = drive_dir / "latest_ginplus_zinc_manifest.json"
    latest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log(f"[manifest] wrote: {manifest_path}")
    return manifest_path


def finalize_existing_completed_run(args: argparse.Namespace) -> Path:
    """Recover bookkeeping after the upstream final relative-path write fails."""
    wrapper_dir = args.drive_dir / "wrapper_logs"
    logs = sorted(
        (path for path in wrapper_dir.glob("ginplus_nope_zinc_seed*.log") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not logs:
        raise RuntimeError(f"No no-PE wrapper logs found under {wrapper_dir}")

    completed_log: Path | None = None
    completed_summary: dict | None = None
    for candidate in logs:
        summary = parse_training_summary_from_logs([candidate])
        if (
            summary.get("max_logged_epoch") == args.max_epoch - 1
            and summary.get("history_epochs") == args.max_epoch
            and summary.get("param_count") == EXPECTED_NOPE_PARAMS
        ):
            completed_log = candidate
            completed_summary = summary
            break
    if completed_log is None or completed_summary is None:
        raise RuntimeError(
            "Could not find a log proving completion of all configured epochs "
            f"with {EXPECTED_NOPE_PARAMS:,} no-PE parameters."
        )

    checkpoints = checkpoint_candidates(args.drive_dir / "results")
    if not checkpoints:
        raise RuntimeError(
            "The log reached the final epoch, but no retained checkpoint was found "
            f"under {args.drive_dir / 'results'}."
        )
    best_epoch = completed_summary.get("best_epoch")
    best_val = completed_summary.get("best_val", {})
    best_test = completed_summary.get("best_test", {})
    if best_epoch is None or "mae" not in best_val or "mae" not in best_test:
        raise RuntimeError(
            f"Could not recover best validation/test MAE from {completed_log}"
        )

    result_file = args.drive_dir / "results" / "subset_result.txt"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    existing = result_file.read_text(encoding="utf-8") if result_file.exists() else ""
    seed_marker = f"seed_{args.seed}:"
    if seed_marker not in existing:
        with result_file.open("a", encoding="utf-8") as handle:
            handle.write(
                "gine residual_True ffn_True 12 80 0.0 "
                f"seed_{args.seed}: test_mae: {float(best_test['mae']):.4f}\n"
            )
        log(f"[finalize] Reconstructed upstream result line: {result_file}")
    else:
        log(f"[finalize] Result line for seed {args.seed} already exists: {result_file}")

    manifest_path = write_run_manifest(
        args,
        args.drive_dir,
        OFFICIAL_COMMIT,
        completed_log,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "training_complete_bookkeeping_recovered",
        "recovered_from_log": str(completed_log),
        "recovery_reason": (
            "Upstream custom_train.py completed the final epoch, then failed while "
            "writing hard-coded relative results/subset_result.txt."
        ),
        "max_logged_epoch": completed_summary.get("max_logged_epoch"),
        "best_epoch_from_completed_log": best_epoch,
        "best_val_from_completed_log": best_val,
        "best_test_from_completed_log": best_test,
        "verified_checkpoint_count": len(checkpoints),
    })
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.drive_dir / "latest_ginplus_zinc_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    recovery_path = args.drive_dir / f"ginplus_nope_finalize_seed{args.seed}.json"
    recovery_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "parameter_count": EXPECTED_NOPE_PARAMS,
                "max_logged_epoch": completed_summary.get("max_logged_epoch"),
                "best_epoch": best_epoch,
                "best_val_mae": float(best_val["mae"]),
                "best_test_mae": float(best_test["mae"]),
                "completed_log": str(completed_log),
                "checkpoints": [str(path) for path in checkpoints],
                "result_file": str(result_file),
                "manifest": str(manifest_path),
                "finalized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    log("[finalize] Verified completed no-PE training without rerunning any epoch.")
    log(
        f"[finalize] best@{best_epoch} val_mae={float(best_val['mae']):.5f} "
        f"test_mae={float(best_test['mae']):.5f}"
    )
    log(f"[finalize] Recovery record: {recovery_path}")
    return recovery_path


def print_environment_summary(drive_dir: Path, repo_dir: Path, commit: str) -> None:
    log("\n[env] Runtime summary")
    log(f"  platform: {platform.platform()}")
    log(f"  python:   {sys.version.replace(os.linesep, ' ')}")
    try:
        import torch
        log(f"  torch:    {torch.__version__}")
        log(f"  cuda:     {torch.version.cuda} | available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"  gpu:      {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        log(f"  torch:    not importable ({exc})")
    log(f"  repo:     {repo_dir}")
    log(f"  commit:   {commit}")
    log(f"  drive:    {drive_dir}")


def _strip_colab_kernel_args(argv: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-f" and i + 1 < len(argv) and "kernel-" in argv[i + 1] and argv[i + 1].endswith(".json"):
            log(f"[args] Ignoring Colab/Jupyter kernel argument: {a} {argv[i + 1]}")
            i += 2
            continue
        if a.startswith("-f=") and "kernel-" in a and a.endswith(".json"):
            log(f"[args] Ignoring Colab/Jupyter kernel argument: {a}")
            i += 1
            continue
        cleaned.append(a)
        i += 1
    return cleaned


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train the official-paper GNNPlus GIN+ -PE ablation on ZINC in Colab.",
        epilog=textwrap.dedent(
            """
            Examples:
              main([])
              main(["--seed", "42", "--name-tag", "ColabDrive.GINPlus_NoPE.ZINC.s42"])
              main(["--skip-install", "--auto-resume"])
            """
        ),
    )
    parser.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    parser.add_argument(
        "--drive-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/ginplus_zinc_nope_official"),
    )
    parser.add_argument("--repo-dir", type=Path, default=Path("/content/GNNPlus"))
    parser.add_argument("--repo-url", default=OFFICIAL_REPO)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--commit", default=OFFICIAL_COMMIT, help="Pin official GNNPlus. Pass empty string to use branch HEAD.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--name-tag", default="ColabDrive.GINPlus_NoPE.ZINC")
    parser.add_argument("--max-epoch", type=int, default=2000)
    parser.add_argument("--ckpt-period", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--pyg-version", default="2.3.1")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--force-fresh-repo", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help=(
            "Do not train. Recover result/manifest bookkeeping when a complete "
            "epoch-1999 run hit the upstream relative results-path error."
        ),
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--console-verbosity", choices=["compact", "standard", "full"], default="compact")
    parser.add_argument("--console-epoch-period", type=int, default=1)
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=False,
        help="Resume only after the runner verifies a checkpoint as no-PE compatible.",
    )
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")
    parser.add_argument(
        "--save-best-checkpoints",
        action="store_true",
        default=True,
        help="Storage-only override: save validation-best checkpoints and keep old checkpoints. Default enabled for Colab recovery.",
    )
    parser.add_argument("--official-checkpoint-policy", action="store_false", dest="save_best_checkpoints")
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _strip_colab_kernel_args(argv)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    mount_drive(args.drive_mount)
    if args.drive_dir.name == "ginplus_zinc_official":
        raise RuntimeError(
            "Refusing to use the full-PE Drive namespace "
            f"{args.drive_dir}. Use a separate no-PE root such as "
            "/content/drive/MyDrive/ginplus_zinc_nope_official."
        )
    args.drive_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize_existing:
        finalize_existing_completed_run(args)
        return

    compat_shim_dir = None
    if sys.version_info >= (3, 12):
        compat_shim_dir = write_py312_compat_shim(args.drive_dir)

    if not args.skip_install:
        install_dependencies(args)
    else:
        log("[deps] Skipping dependency installation (--skip-install).")

    commit = clone_or_update_repo(args.repo_dir, args.repo_url, args.branch, args.commit or None, args.force_fresh_repo)
    validate_official_config(args.repo_dir, allow_drift=False)
    apply_nope_edge_attr_compat_patch(args.repo_dir, args.drive_dir)
    install_gnnplus_editable(args.repo_dir)
    nope_cfg_path = prepare_and_validate_nope_config(args.repo_dir, args.drive_dir)
    print_environment_summary(args.drive_dir, args.repo_dir, commit)
    validate_resume_checkpoint_namespace(args)
    prepare_results_namespace(args)
    prepare_legacy_results_link(args.repo_dir, args.drive_dir)

    cmd = build_training_command(args, args.drive_dir, nope_cfg_path)
    wrapper_log = args.drive_dir / "wrapper_logs" / f"ginplus_nope_zinc_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    rc = run_streaming_to_console_and_log(
        cmd,
        cwd=args.repo_dir,
        log_file=wrapper_log,
        env=env_with_py312_compat(compat_shim_dir),
        console_verbosity=args.console_verbosity,
        console_epoch_period=args.console_epoch_period,
        expected_params=EXPECTED_NOPE_PARAMS,
    )
    if rc != 0:
        raise SystemExit(rc)

    write_run_manifest(args, args.drive_dir, commit, wrapper_log)
    log("\n[done] GIN+ -PE ZINC training process completed successfully.")
    log(f"[done] Results/checkpoints root: {args.drive_dir / 'results'}")
    log(f"[done] Wrapper log: {wrapper_log}")


if __name__ == "__main__":
    main([
        # Recovery mode for the completed seed-41 run. Remove this option only
        # when intentionally launching a new training run.
        "--finalize-existing",
        # "--skip-install",  # Safe when rerunning training in the same runtime.
        "--seed", "41",
        "--name-tag", "ColabDrive.GINPlus_NoPE.ZINC.s41",
        "--drive-dir", "/content/drive/MyDrive/ginplus_zinc_nope_official",
        "--no-auto-resume",
    ])
