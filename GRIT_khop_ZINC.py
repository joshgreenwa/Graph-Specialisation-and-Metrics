# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""Colab runner for configurable k-hop GRIT on the ZINC subset.

This script intentionally DOES NOT reimplement GRIT. It clones the official
LiamMa/GRIT repository, pins it to the same official commit used by the dense
runner, applies a local k-hop support-restriction patch, and launches GRIT.

The attention support contains every ordered node pair whose shortest-path
distance is at most ``--hops`` (self included). The support is recovered exactly
from the existing RRWP walk channels, so changing ``--hops`` does not require
regenerating the cached ZINC dataset. With the official 21-channel RRWP config,
valid values are 1 through 20.

The k-hop config is matched to the official ZINC RRWP config for dataset, model
width/depth/heads, RRWP dimensions, optimizer, schedule, batch size, loss, and
checkpoint behavior. The main scientific intervention is restricting the RRWP
relative edge representation/attention support to pairs at distance <= k
(`gt.attn.full_attn=False`, `gt.attn.sparsity=k_hop`). Pass ``--global-vnode``
to add one learned global virtual node per graph. It attends bidirectionally with
every real node in every attention layer and is excluded from final graph
pooling. The VNode adds exactly 64 trainable parameters.

Suggested Colab usage:

    # Upload this file to Colab, then run:
    from GRIT_khop_ZINC import main
    main(["--skip-install", "--hops", "2"])

    # A 3-hop model with a global virtual node:
    main(["--skip-install", "--hops", "3", "--global-vnode"])

For a fresh runtime, omit --skip-install:

    main([])

Checkpoints/logs are written by default to:

    /content/drive/MyDrive/grit_zinc_khop/results

Notes:
- The official GRIT README specifies a Python 3.9 / PyTorch 1.12.1 / PyG 2.2.0
  environment. Modern Colab images usually ship newer Python/PyTorch. By default
  this script keeps Colab's preinstalled torch and installs torch-geometric==2.2.0
  plus matching PyG extension wheels for the active torch/CUDA build.
- On Python 3.12 / modern Colab, the runner injects a subprocess-local
  compatibility shim for old Lightning/pkg_resources imports, restores legacy
  torch.load behavior for PyG processed dataset files under PyTorch >=2.6,
  restores the legacy scikit-learn mean_squared_error(..., squared=False)
  logger API expected by GRIT, and filters very large model/config dumps from
  notebook stdout while preserving the full raw log on Drive. This does not alter
  GRIT model/config logic.
- For maximal historical fidelity, run in a Python 3.9/3.10 runtime/container and
  pass --official-torch112. That may not work on current Colab Python versions.
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
from typing import Iterable, List, Mapping, Sequence


OFFICIAL_REPO = "https://github.com/LiamMa/GRIT.git"
DENSE_OFFICIAL_CFG = "configs/GRIT/zinc-GRIT-RRWP.yaml"
OFFICIAL_CFG = "configs/GRIT/zinc-GRIT-RRWP-khop.yaml"
# Current official main-branch commit observed from GitHub commit history.
OFFICIAL_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
EXPECTED_ZINC_GRIT_RRWP_PARAMS = 473_473
VNODE_PARAM_COUNT = 64
KHOP_CFG_TEMPLATE = """\
# Configurable k-hop sparse-control variant of the official ZINC GRIT RRWP config.
#
# This file intentionally keeps the official ZINC subset model, optimizer,
# training schedule, RRWP dimensions, and decoder settings. The only scientific
# intervention is `gt.attn.sparsity: k_hop`. The local patch keeps node pairs
# whose shortest-path distance is <= `gt.attn.hops`. An optional learned global
# VNode is connected bidirectionally to every real node in every attention layer.
out_dir: results
metric_best: mae
metric_agg: argmin
tensorboard_each_run: True
accelerator: "cuda:0"
mlflow:
  use: False
  project: Exp
  name: {run_name}
wandb:
  use: False
  project: ZINC
dataset:
  format: PyG-ZINC
  name: subset
  task: graph
  task_type: regression
  transductive: False
  node_encoder: True
  node_encoder_name: TypeDictNode
  node_encoder_num_types: 21
  node_encoder_bn: False
  edge_encoder: True
  edge_encoder_name: TypeDictEdge
  edge_encoder_num_types: 4
  edge_encoder_bn: False
posenc_RRWP:
  enable: True
  ksteps: 21
  add_identity: True
  add_node_attr: False
  add_inverse: False
train:
  mode: custom
  batch_size: 32
  eval_period: 1
  enable_ckpt: True
  ckpt_best: True
  ckpt_clean: True
model:
  type: GritTransformer
  loss_fun: l1
  edge_decoding: dot
  graph_pooling: add
gt:
  layer_type: GritTransformer
  layers: 10
  n_heads: 8
  dim_hidden: 64
  dropout: 0.0
  layer_norm: False
  batch_norm: True
  update_e: True
  attn_dropout: 0.2
  attn:
    clamp: 5.
    act: 'relu'
    full_attn: False
    sparsity: k_hop
    hops: {hops}
    global_vnode: {global_vnode}
    edge_enhance: True
    O_e: True
    norm_e: True
    fwl: False
gnn:
  head: san_graph
  layers_pre_mp: 0
  layers_post_mp: 3
  dim_inner: 64
  batchnorm: True
  act: relu
  dropout: 0.0
  agg: mean
  normalize_adj: False
optim:
  clip_grad_norm: True
  optimizer: adamW
  weight_decay: 1e-5
  base_lr: 1e-3
  max_epoch: 2000
  num_warmup_epochs: 50
  scheduler: cosine_with_warmup
  min_lr: 1e-6
"""
EXPECTED_CFG_VALUES = {
    ("metric_best",): "mae",
    ("metric_agg",): "argmin",
    ("dataset", "format"): "PyG-ZINC",
    ("dataset", "name"): "subset",
    ("dataset", "task"): "graph",
    ("dataset", "task_type"): "regression",
    ("dataset", "transductive"): False,
    ("dataset", "node_encoder"): True,
    ("dataset", "node_encoder_name"): "TypeDictNode",
    # Official GRIT config comment: ZINC-12k uses 21 atom types; ZINC-full uses 28.
    ("dataset", "node_encoder_num_types"): 21,
    ("dataset", "node_encoder_bn"): False,
    ("dataset", "edge_encoder"): True,
    ("dataset", "edge_encoder_name"): "TypeDictEdge",
    ("dataset", "edge_encoder_num_types"): 4,
    ("dataset", "edge_encoder_bn"): False,
    ("posenc_RRWP", "enable"): True,
    ("posenc_RRWP", "ksteps"): 21,
    ("posenc_RRWP", "add_identity"): True,
    ("posenc_RRWP", "add_node_attr"): False,
    ("posenc_RRWP", "add_inverse"): False,
    ("train", "mode"): "custom",
    ("train", "batch_size"): 32,
    ("train", "eval_period"): 1,
    ("train", "enable_ckpt"): True,
    ("train", "ckpt_best"): True,
    ("train", "ckpt_clean"): True,
    ("model", "type"): "GritTransformer",
    ("model", "loss_fun"): "l1",
    ("model", "graph_pooling"): "add",
    ("gt", "layer_type"): "GritTransformer",
    ("gt", "layers"): 10,
    ("gt", "n_heads"): 8,
    ("gt", "dim_hidden"): 64,
    ("gt", "dropout"): 0.0,
    ("gt", "attn_dropout"): 0.2,
    ("gt", "layer_norm"): False,
    ("gt", "batch_norm"): True,
    ("gt", "update_e"): True,
    ("gt", "attn", "clamp"): 5.0,
    ("gt", "attn", "act"): "relu",
    ("gt", "attn", "full_attn"): False,
    ("gt", "attn", "sparsity"): "k_hop",
    ("gt", "attn", "edge_enhance"): True,
    ("gt", "attn", "O_e"): True,
    ("gt", "attn", "norm_e"): True,
    ("gt", "attn", "fwl"): False,
    ("gnn", "head"): "san_graph",
    ("gnn", "layers_pre_mp"): 0,
    ("gnn", "layers_post_mp"): 3,
    ("gnn", "dim_inner"): 64,
    ("gnn", "batchnorm"): True,
    ("gnn", "act"): "relu",
    ("gnn", "dropout"): 0.0,
    ("gnn", "agg"): "mean",
    ("gnn", "normalize_adj"): False,
    ("optim", "clip_grad_norm"): True,
    ("optim", "optimizer"): "adamW",
    ("optim", "weight_decay"): 1e-5,
    ("optim", "base_lr"): 1e-3,
    ("optim", "max_epoch"): 2000,
    ("optim", "scheduler"): "cosine_with_warmup",
    ("optim", "num_warmup_epochs"): 50,
    ("optim", "min_lr"): 1e-6,
}


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
    """Stateful stdout filter for long GRIT notebook runs.

    The full subprocess output is always written to the Drive log. This filter only
    controls what is echoed into the notebook, which avoids Colab output trimming
    from the model/config dump and tqdm progress bars.

    In compact/standard mode, raw GraphGym `train: {...}`, `val: {...}`, and
    `test: {...}` dictionaries are parsed and replaced with one concise epoch line:
    current train/val/test loss+MAE, best validation MAE/loss so far, test@best,
    and epoch timing. This exposes validation movement even when it is not a new
    best epoch.
    """

    def __init__(self, *, verbosity: str, epoch_period: int) -> None:
        self.verbosity = verbosity
        self.epoch_period = max(1, int(epoch_period))
        self._suppress_until_num_params = False
        self._in_traceback = False
        self._epoch_stats: dict[int, dict[str, dict]] = {}
        self._best_epoch: int | None = None
        self._best_val_mae = float("inf")
        self._best_val_loss = float("inf")
        self._best_test_mae = float("nan")
        self._best_test_loss = float("nan")
        self._best_train_mae = float("nan")
        self._best_train_loss = float("nan")

    @staticmethod
    def _fmt(value, digits: int = 5) -> str:
        try:
            v = float(value)
        except Exception:
            return "nan"
        if v != v:  # NaN
            return "nan"
        return f"{v:.{digits}f}"

    def _should_emit_epoch(self, epoch: int) -> bool:
        # GRIT epochs are zero-indexed internally. Show first two epochs and
        # then every Nth human-readable epoch when thinning output.
        return epoch < 2 or ((epoch + 1) % self.epoch_period == 0)

    def _parse_split_stats(self, stripped: str) -> bool:
        """Parse lines of the form `train: {'epoch': 0, ...}`.

        Returns True iff the line was a split-stat line and should be suppressed
        from raw stdout.
        """
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
        splits = self._epoch_stats.get(epoch, {})
        val = splits.get("val")
        if not val:
            return
        # The official ZINC config uses metric_best=mae and metric_agg=argmin.
        val_mae = float(val.get("mae", val.get("loss", float("inf"))))
        val_loss = float(val.get("loss", float("inf")))
        if val_mae < self._best_val_mae:
            self._best_epoch = epoch
            self._best_val_mae = val_mae
            self._best_val_loss = val_loss
            train = splits.get("train", {})
            test = splits.get("test", {})
            self._best_train_mae = float(train.get("mae", float("nan")))
            self._best_train_loss = float(train.get("loss", float("nan")))
            self._best_test_mae = float(test.get("mae", float("nan")))
            self._best_test_loss = float(test.get("loss", float("nan")))

    def _emit_compact_epoch_line(self, stripped: str) -> None:
        # Official line contains timing and best-so-far. We recompute best from the
        # parsed split stats and use the official line only for timing.
        m_epoch = re.search(r"> Epoch\s+(\d+):", stripped)
        if not m_epoch:
            return
        epoch = int(m_epoch.group(1))
        if not self._should_emit_epoch(epoch):
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
        line = (
            f"[epoch {epoch:04d}] "
            f"cur train_loss={self._fmt(train.get('loss'))} train_mae={self._fmt(train.get('mae'))} | "
            f"val_loss={self._fmt(val.get('loss'))} val_mae={self._fmt(val.get('mae'))} | "
            f"test_loss={self._fmt(test.get('loss'))} test_mae={self._fmt(test.get('mae'))} | "
            f"best@{best_epoch} val_loss={self._fmt(self._best_val_loss)} "
            f"val_mae={self._fmt(self._best_val_mae)} "
            f"test_mae={self._fmt(self._best_test_mae)} | "
            f"time={took}s avg={avg}s"
        )
        print(line, flush=True)

    def should_print(self, line: str) -> bool:
        if self.verbosity == "full":
            return True

        stripped = line.strip()
        if not stripped:
            return False

        # Keep Python errors complete enough to debug.
        if stripped.startswith("Traceback"):
            self._in_traceback = True
            return True
        if self._in_traceback:
            return True
        if any(tok in stripped for tok in (
            "Error:", "ERROR", "Exception", "RuntimeError", "TypeError",
            "AttributeError", "UnpicklingError", "SystemExit",
        )):
            return True

        # Parse current split metrics and suppress their verbose dictionary form.
        if self._parse_split_stats(stripped):
            return False

        # Suppress the very large model repr + full expanded config block. The next
        # important line after that dump is the parameter count, which we keep.
        if stripped.startswith("GraphGymModule("):
            self._suppress_until_num_params = True
            return False
        if self._suppress_until_num_params:
            if re.search(r"Num parameters:\s*[0-9,]+", stripped):
                self._suppress_until_num_params = False
                return True
            return False

        # Suppress tqdm/progress noise and warnings that are not actionable here.
        noisy_fragments = (
            "it/s]", "Processing train dataset", "Processing val dataset",
            "Processing test dataset", "SyntaxWarning:", "UserWarning:",
            "Downloading ", "Extracting ", "Processing...", "Done!",
        )
        if any(fragment in stripped for fragment in noisy_fragments):
            return False

        # Keep concise setup/provenance lines.
        setup_prefixes = (
            "[*] Run ID", "Starting now:", "[*] Loaded dataset", "Data(",
            "undirected:", "num graphs:", "avg num_nodes/graph:",
            "num node features:", "num edge features:", "num classes:",
            "Parsed RWSE", "Parsed RRWP", "Precomputing Positional Encoding statistics", "Computing RRWP", "Precomputing RRWP",
            "Start from epoch", "Checkpoint found", "Task done",
            "Avg time per epoch", "Total train loop time",
            "Forced numbered recovery checkpoint saved",
            "Stable recovery checkpoint copied",
        )
        if stripped.startswith(setup_prefixes):
            return True

        # Replace the official best-only epoch line by a compact current+best line.
        if re.search(r"> Epoch\s+(\d+):", stripped):
            self._emit_compact_epoch_line(stripped)
            return False

        # Keep parameter-count validation.
        if re.search(r"Num parameters:\s*[0-9,]+", stripped):
            return True
        if stripped.startswith("[param-check"):
            return True

        # Standard mode keeps more non-progress one-liners; compact mode drops the
        # rest. Full mode is handled at the top.
        if self.verbosity == "standard":
            return True
        return False

def run_streaming_to_console_and_log(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_file: Path,
    env: Mapping[str, str] | None = None,
    expected_params: int | None = None,
    allow_param_count_drift: bool = False,
    console_verbosity: str = "compact",
    console_epoch_period: int = 1,
) -> int:
    printable = " ".join(map(str, cmd))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"\n[train-cmd] {printable}")
    log(f"[train-log] {log_file}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env["PYTHONUNBUFFERED"] = "1"

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
        saw_param_count = False
        param_mismatch = False

        console_filter = ConsoleFilter(
            verbosity=console_verbosity,
            epoch_period=max(1, int(console_epoch_period)),
        )
        if console_verbosity != "full":
            msg = (
                f"[console] {console_verbosity} stdout enabled; full raw GRIT output is saved to {log_file}\n"
            )
            print(msg, end="", flush=True)
            f.write(msg)
            f.flush()

        for line in proc.stdout:
            f.write(line)
            f.flush()
            if console_filter.should_print(line):
                print(line, end="", flush=True)

            if expected_params is not None:
                match = re.search(r"Num parameters:\s*([0-9,]+)", line)
                if match is not None:
                    saw_param_count = True
                    actual_params = int(match.group(1).replace(",", ""))
                    if actual_params == expected_params:
                        msg = f"[param-check] Matched expected GRIT ZINC+RRWP parameter count: {actual_params}\n"
                        print(msg, end="", flush=True)
                        f.write(msg)
                        f.flush()
                    else:
                        msg = (
                            f"[param-check:ERROR] Model parameter count is {actual_params}, "
                            f"expected {expected_params}. Aborting before training to avoid a non-paper config.\n"
                        )
                        print(msg, end="", flush=True)
                        f.write(msg)
                        f.flush()
                        param_mismatch = True
                        if not allow_param_count_drift:
                            proc.terminate()
                            try:
                                proc.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
                            break

        rc = proc.wait()
        elapsed = time.perf_counter() - start
        if expected_params is not None and not saw_param_count:
            msg = (
                f"[param-check:ERROR] Did not observe a `Num parameters:` line. "
                f"Cannot verify exact paper config parameter count {expected_params}.\n"
            )
            print(msg, end="", flush=True)
            f.write(msg)
            if rc == 0 and not allow_param_count_drift:
                rc = 98
        if param_mismatch and not allow_param_count_drift:
            rc = 97
        f.write(f"\nReturn code: {rc}\nElapsed seconds: {elapsed:.2f}\n")
    return rc



def write_py312_compat_shim(base_dir: Path) -> Path:
    """Create a sitecustomize.py shim for Python 3.12 + older GRIT deps.

    Current Colab runtimes may use Python 3.12, while the official GRIT
    environment was Python 3.10-era. Old setuptools/pkg_resources code imported
    through PyG/Lightning can hit two Python-3.12 removals:

      1. pkgutil.ImpImporter no longer exists.
      2. importlib.machinery.FileFinder no longer exposes find_module().

    Python automatically imports `sitecustomize` from PYTHONPATH during interpreter
    startup. This shim is therefore applied before GRIT imports Lightning/PyG.
    It does not alter any GRIT model, dataset, optimizer, or training config.
    """
    shim_dir = base_dir / "python312_compat"
    shim_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = shim_dir / "sitecustomize.py"
    sitecustomize.write_text(
        """
# Auto-generated by GRIT_khop_ZINC.py.
# Compatibility shim for Python 3.12 running older GRIT dependencies.
# This file is imported automatically by Python at interpreter startup when its
# directory is prepended to PYTHONPATH.

import importlib.machinery
import os
import pkgutil
import site
import sys
import sysconfig

# Prefer pip-installed packages in /usr/local over Debian's /usr/lib copy.
# On some Colab Python-3.12 images, an old Debian pkg_resources shadows the
# newer pip-installed setuptools/pkg_resources and then crashes on removed
# importlib APIs. Reordering here is conservative and subprocess-local.
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
    sys.path.insert(1, _p)  # keep this shim directory at sys.path[0]

# Python 3.12 removed pkgutil.ImpImporter. Very old pkg_resources imports still
# reference it during module initialization. A dummy class is sufficient because
# modern importers are not instances of the old imp importer anyway.
if not hasattr(pkgutil, "ImpImporter"):
    class _CompatImpImporter:  # pragma: no cover - startup compatibility only
        pass
    pkgutil.ImpImporter = _CompatImpImporter

# Old pkg_resources also falls back to importer.find_module(...). Python 3.12's
# FileFinder only has find_spec(...). Restore a minimal equivalent method.
if not hasattr(importlib.machinery.FileFinder, "find_module"):
    def _compat_find_module(self, fullname, path=None):
        spec = self.find_spec(fullname)
        return None if spec is None else spec.loader
    importlib.machinery.FileFinder.find_module = _compat_find_module

# PyTorch >= 2.6 changed torch.load's default to weights_only=True. Older PyG
# datasets, including ZINC in torch_geometric==2.2.0, save processed Data objects
# and later call torch.load(path) without specifying weights_only. That now fails
# because Data is not a weight tensor/state-dict object. For this official ZINC
# run, these files are locally generated by PyG from the benchmark data and are
# trusted. Patch only unspecified calls; explicit weights_only=True/False remains
# respected. This restores the behavior expected by the official GRIT era.
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
    # Avoid breaking interpreter startup if torch is unavailable during early init.
    pass

# If pkg_resources was somehow imported before this shim finished and came from
# Debian's system path, discard it so a later import can prefer /usr/local. This
# branch is normally not taken because sitecustomize runs very early.
_mod = sys.modules.get("pkg_resources")
_mod_file = getattr(_mod, "__file__", "") if _mod is not None else ""
if _mod_file.startswith("/usr/lib/python3/dist-packages/"):
    del sys.modules["pkg_resources"]

# scikit-learn >= 1.6 removed the `squared` keyword from
# sklearn.metrics.mean_squared_error and introduced root_mean_squared_error.
# The official GRIT logger, written against older sklearn, calls
# mean_squared_error(..., squared=False) to report RMSE. Restore that API
# locally for the subprocess so logging works without editing GRIT files.
try:
    import inspect
    import numpy as _np
    import sklearn.metrics as _sk_metrics

    _mse_sig = inspect.signature(_sk_metrics.mean_squared_error)
    if "squared" not in _mse_sig.parameters:
        _orig_mean_squared_error = _sk_metrics.mean_squared_error

        def _compat_mean_squared_error(
            y_true,
            y_pred,
            *,
            sample_weight=None,
            multioutput="uniform_average",
            squared=True,
        ):
            mse = _orig_mean_squared_error(
                y_true,
                y_pred,
                sample_weight=sample_weight,
                multioutput=multioutput,
            )
            if squared:
                return mse
            # Preserve legacy sklearn behavior: squared=False returns RMSE.
            return _np.sqrt(mse)

        _sk_metrics.mean_squared_error = _compat_mean_squared_error
except Exception:
    # Avoid breaking interpreter startup if sklearn is unavailable during early init.
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
        log(f"[compat] Prepending shim directory to PYTHONPATH for GRIT subprocess: {shim_dir}")
    return env

def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def mount_drive(mount_point: Path) -> None:
    if in_colab():
        from google.colab import drive  # type: ignore
        log(f"[drive] Mounting Google Drive at {mount_point} ...")
        drive.mount(str(mount_point), force_remount=False)
    else:
        log("[drive] google.colab is unavailable; assuming a non-Colab run.")
        log(f"[drive] Using local path instead: {mount_point}")
        mount_point.mkdir(parents=True, exist_ok=True)


def pip_install(args: Sequence[str]) -> None:
    run_cmd([sys.executable, "-m", "pip", "install", *args])


def install_dependencies(args: argparse.Namespace) -> None:
    log("\n[deps] Installing Python dependencies. This may take a while on a fresh Colab runtime.")
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    if args.official_torch112:
        if sys.version_info >= (3, 11):
            raise RuntimeError(
                "--official-torch112 requested, but PyTorch 1.12 wheels are not available "
                "for Python >= 3.11. Use a Python 3.9/3.10 runtime/container, or omit this flag "
                "to use Colab's current torch while keeping the official GRIT config."
            )
        pip_install([
            "torch==1.12.1+cu113",
            "torchvision==0.13.1+cu113",
            "torchaudio==0.12.1",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu113",
        ])

    import importlib
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError("PyTorch is not installed/importable in this runtime.") from exc

    torch_version = str(torch.__version__).split("+")[0]
    cuda_version = getattr(torch.version, "cuda", None)
    if cuda_version:
        cuda_tag = "cu" + cuda_version.replace(".", "")
    else:
        cuda_tag = "cpu"

    log(f"[deps] Python: {sys.version.split()[0]} | torch: {torch.__version__} | CUDA: {cuda_version}")
    pyg_wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    log(f"[deps] PyG wheel index: {pyg_wheel_url}")

    # Install compiled PyG extensions matched to the active torch/CUDA build.
    # Current Colab images can move ahead of the full PyG extension matrix. For
    # torch 2.11/cu128, torch-spline-conv may be unavailable while the ZINC GRIT
    # RRWP path only needs the core PyG stack. Keep spline-conv as a best-effort
    # optional install so a missing wheel does not abort the official GRIT run.
    pip_install([
        "pyg-lib",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "-f",
        pyg_wheel_url,
    ])
    spline_proc = run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch-spline-conv",
            "-f",
            pyg_wheel_url,
        ],
        check=False,
    )
    if spline_proc.returncode != 0:
        log("[deps-warning] torch-spline-conv wheel was unavailable for this Torch/CUDA/Python stack; continuing because GRIT ZINC RRWP does not use spline convolution.")

    # GRIT README says PyG v2.2 is required. Keep this explicit and configurable.
    pip_install([f"torch-geometric=={args.pyg_version}"])

    # Runtime/support dependencies used by GRIT/GraphGym/ZINC logging.
    pip_install([
        "yacs==0.1.8",
        "pytorch-lightning==1.9.5",
        "torchmetrics==0.9.1",
        "opt_einsum>=3.3",
        "tensorboardX>=2.6,<2.7",
        "ogb==1.3.6",
        "wandb>=0.16,<0.18",
        "pyyaml>=6.0",
        "scikit-learn>=1.0",
        "scipy>=1.9",
        "networkx>=2.8",
    ])


def clone_or_update_repo(repo_dir: Path, repo_url: str, branch: str, commit: str | None, force_fresh: bool) -> str:
    if force_fresh and repo_dir.exists():
        log(f"[repo] Removing existing repo: {repo_dir}")
        shutil.rmtree(repo_dir)

    if not repo_dir.exists():
        run_cmd(["git", "clone", "--branch", branch, repo_url, str(repo_dir)])
    else:
        log(f"[repo] Existing repo found: {repo_dir}")
        current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
        if commit and current == commit:
            log("[repo] Existing repo is already at the pinned official commit; keeping local k-hop patch in place.")
        else:
            run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
            run_cmd(["git", "checkout", branch], cwd=repo_dir)
            run_cmd(["git", "pull", "--ff-only", "origin", branch], cwd=repo_dir)

    if commit:
        log(f"[repo] Pinning GRIT to commit: {commit}")
        run_cmd(["git", "checkout", commit], cwd=repo_dir)

    resolved_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    log(f"[repo] Using GRIT commit: {resolved_commit}")
    return resolved_commit


def _read_text_preserve_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def _write_text_preserve_newlines(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def _native_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _replace_exact(path: Path, old: str, new: str, marker: str, label: str) -> bool:
    text = _read_text_preserve_newlines(path)
    newline = _native_newline(text)
    marker_native = marker.replace("\n", newline)
    if marker_native in text:
        log(f"[patch] {label}: already present")
        return False
    old_native = old.replace("\n", newline)
    new_native = new.replace("\n", newline)
    if old_native not in text:
        raise RuntimeError(
            f"Could not apply k-hop patch segment `{label}` to {path}. "
            "The official GRIT source differs from the pinned layout."
        )
    _write_text_preserve_newlines(path, text.replace(old_native, new_native, 1))
    log(f"[patch] {label}: applied")
    return True


def _insert_after(path: Path, anchor: str, insertion: str, marker: str, label: str) -> bool:
    text = _read_text_preserve_newlines(path)
    newline = _native_newline(text)
    marker_native = marker.replace("\n", newline)
    if marker_native in text:
        log(f"[patch] {label}: already present")
        return False
    anchor_native = anchor.replace("\n", newline)
    insertion_native = insertion.replace("\n", newline)
    if anchor_native not in text:
        raise RuntimeError(
            f"Could not apply k-hop patch insertion `{label}` to {path}. "
            "The official GRIT source differs from the pinned layout."
        )
    _write_text_preserve_newlines(path, text.replace(anchor_native, anchor_native + insertion_native, 1))
    log(f"[patch] {label}: applied")
    return True


def _replace_if_present(path: Path, old: str, new: str, label: str) -> bool:
    text = _read_text_preserve_newlines(path)
    newline = _native_newline(text)
    old_native = old.replace("\n", newline)
    new_native = new.replace("\n", newline)
    if old_native not in text:
        return False
    _write_text_preserve_newlines(path, text.replace(old_native, new_native, 1))
    log(f"[patch] {label}: updated")
    return True


def apply_khop_patch(repo_dir: Path, drive_dir: Path, args: argparse.Namespace) -> None:
    """Apply configurable k-hop support and optional global-VNode patches."""
    log(
        f"\n[patch] Applying {args.hops}-hop GRIT control patch "
        f"(global_vnode={args.global_vnode})."
    )

    cfg_path = repo_dir / OFFICIAL_CFG
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    config_text = KHOP_CFG_TEMPLATE.format(
        run_name=f"zinc-GRIT-RRWP-{args.hops}hop" + ("-vnode" if args.global_vnode else ""),
        hops=args.hops,
        global_vnode="True" if args.global_vnode else "False",
    )
    current_cfg = _read_text_preserve_newlines(cfg_path) if cfg_path.exists() else ""
    if current_cfg != config_text:
        _write_text_preserve_newlines(cfg_path, config_text)
        log(f"[patch] wrote exact {args.hops}-hop ZINC config: {cfg_path}")
    else:
        log(f"[patch] exact {args.hops}-hop ZINC config already present: {cfg_path}")

    gt_config = repo_dir / "grit" / "config" / "gt_config.py"
    _replace_exact(
        gt_config,
        old=(
            '    cfg.gt.attn.full_attn = True\n'
            '    cfg.gt.attn.norm_e = True\n'
        ),
        new=(
            '    cfg.gt.attn.full_attn = True\n'
            '    cfg.gt.attn.sparsity = "full"\n'
            '    cfg.gt.attn.hops = 1\n'
            '    cfg.gt.attn.global_vnode = False\n'
            '    cfg.gt.attn.norm_e = True\n'
        ),
        marker='cfg.gt.attn.global_vnode = False',
        label="default k-hop/VNode attention config",
    )

    rrwp_encoder = repo_dir / "grit" / "encoder" / "rrwp_encoder.py"
    _replace_exact(
        rrwp_encoder,
        old=(
            '                 mask_index_name="edge_index",\n'
            '                 ):\n'
        ),
        new=(
            '                 mask_index_name="edge_index",\n'
            '                 max_hops=None,\n'
            '                 ):\n'
        ),
        marker="max_hops=None",
        label="masked RRWP max-hops argument",
    )
    _replace_exact(
        rrwp_encoder,
        old=(
            '        torch.nn.init.xavier_uniform_(self.fc.weight)\n'
            '        self.fill_value = 0.\n'
        ),
        new=(
            '        torch.nn.init.xavier_uniform_(self.fc.weight)\n'
            '        self.pad_to_full_graph = False\n'
            '        self.max_hops = None if max_hops is None else int(max_hops)\n'
            '        self.fill_value = 0.\n'
        ),
        marker="self.max_hops = None if max_hops is None else int(max_hops)",
        label="masked RRWP k-hop state",
    )
    _replace_exact(
        rrwp_encoder,
        old=(
            '        rrwp_idx = batch.rrwp_index\n'
            '        rrwp_val = batch.rrwp_val\n'
            '        edge_index = batch.edge_index\n'
            '        edge_attr = batch.edge_attr\n'
            '        rrwp_val = self.fc(rrwp_val)\n'
            '        mask_index = batch.get(self.mask_index_name, None)\n'
            '        num_nodes = batch.num_nodes\n'
        ),
        new=(
            '        rrwp_idx = batch.rrwp_index\n'
            '        raw_rrwp_val = batch.rrwp_val\n'
            '        edge_index = batch.edge_index\n'
            '        edge_attr = batch.edge_attr\n'
            '        if self.max_hops is None:\n'
            '            mask_index = batch.get(self.mask_index_name, None)\n'
            '        else:\n'
            '            needed_channels = self.max_hops + 1  # identity + walks of length 1..k\n'
            '            if self.max_hops < 1 or needed_channels > raw_rrwp_val.size(1):\n'
            '                raise ValueError(\n'
            '                    f"max_hops={self.max_hops} requires {needed_channels} RRWP channels; "\n'
            '                    f"found {raw_rrwp_val.size(1)}."\n'
            '                )\n'
            '            reachable = raw_rrwp_val[:, :needed_channels].abs().sum(dim=-1) > 0\n'
            '            mask_index = rrwp_idx[:, reachable]\n'
            '        rrwp_val = self.fc(raw_rrwp_val)\n'
            '        num_nodes = batch.num_nodes\n'
        ),
        marker="needed_channels = self.max_hops + 1",
        label="derive exact <=k-hop mask from RRWP",
    )

    grit_model = repo_dir / "grit" / "network" / "grit_model.py"
    _replace_exact(
        grit_model,
        old=(
            '        if cfg.posenc_RRWP.enable:\n'
            '            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"]\\\n'
            '                (cfg.posenc_RRWP.ksteps, cfg.gnn.dim_inner)\n'
            '            rel_pe_dim = cfg.posenc_RRWP.ksteps\n'
            '            self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"] \\\n'
            '                (rel_pe_dim, cfg.gnn.dim_edge,\n'
            '                 pad_to_full_graph=cfg.gt.attn.full_attn,\n'
            '                 add_node_attr_as_self_loop=False,\n'
            '                 fill_value=0.\n'
            '                 )\n'
        ),
        new=(
            '        if cfg.posenc_RRWP.enable:\n'
            '            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"]\\\n'
            '                (cfg.posenc_RRWP.ksteps, cfg.gnn.dim_inner)\n'
            '            rel_pe_dim = cfg.posenc_RRWP.ksteps\n'
            '            attn_sparsity = cfg.gt.attn.get("sparsity", "full")\n'
            '            if attn_sparsity == "full":\n'
            '                self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"] \\\n'
            '                    (rel_pe_dim, cfg.gnn.dim_edge,\n'
            '                     pad_to_full_graph=cfg.gt.attn.full_attn,\n'
            '                     add_node_attr_as_self_loop=False,\n'
            '                     fill_value=0.\n'
            '                     )\n'
            '            elif attn_sparsity == "k_hop":\n'
            '                self.rrwp_rel_encoder = register.edge_encoder_dict["masked_rrwp_linear"] \\\n'
            '                    (rel_pe_dim, cfg.gnn.dim_edge,\n'
            '                     add_node_attr_as_self_loop=False,\n'
            '                     fill_value=0.,\n'
            '                     mask_index_name="edge_index",\n'
            '                     max_hops=cfg.gt.attn.hops,\n'
            '                     )\n'
            '            else:\n'
            '                raise ValueError(\n'
            '                    f"Unsupported cfg.gt.attn.sparsity={attn_sparsity!r}; "\n'
            '                    "expected \'full\' or \'k_hop\'."\n'
            '                )\n'
        ),
        marker="max_hops=cfg.gt.attn.hops",
        label="GritTransformer RRWP k-hop switch",
    )

    _replace_exact(
        grit_model,
        old=(
            '\n\n@register_network(\'GritTransformer\')\n'
            'class GritTransformer(torch.nn.Module):\n'
        ),
        new=(
            '\n\nclass GlobalVNode(torch.nn.Module):\n'
            '    """One learned graph token connected both ways to every real node."""\n'
            '\n'
            '    def __init__(self, dim):\n'
            '        super().__init__()\n'
            '        self.embedding = torch.nn.Parameter(torch.zeros(1, dim))\n'
            '\n'
            '    def forward(self, batch):\n'
            '        graph_id = batch.batch\n'
            '        num_real = batch.x.size(0)\n'
            '        num_graphs = int(graph_id.max().item()) + 1\n'
            '        device = batch.x.device\n'
            '        vnode_id = torch.arange(num_graphs, device=device) + num_real\n'
            '        real_id = torch.arange(num_real, device=device)\n'
            '        vnode_for_real = vnode_id[graph_id]\n'
            '\n'
            '        # source->target convention used by GRIT attention.\n'
            '        virtual_edges = torch.cat([\n'
            '            torch.stack([real_id, vnode_for_real]),\n'
            '            torch.stack([vnode_for_real, real_id]),\n'
            '            torch.stack([vnode_id, vnode_id]),\n'
            '        ], dim=1)\n'
            '        virtual_attr = batch.edge_attr.new_zeros(\n'
            '            virtual_edges.size(1), batch.edge_attr.size(1)\n'
            '        )\n'
            '\n'
            '        batch.x = torch.cat([\n'
            '            batch.x, self.embedding.to(dtype=batch.x.dtype).expand(num_graphs, -1)\n'
            '        ], dim=0)\n'
            '        batch.batch = torch.cat([\n'
            '            graph_id, torch.arange(num_graphs, device=device, dtype=graph_id.dtype)\n'
            '        ], dim=0)\n'
            '        batch.edge_index = torch.cat([batch.edge_index, virtual_edges], dim=1)\n'
            '        batch.edge_attr = torch.cat([batch.edge_attr, virtual_attr], dim=0)\n'
            '        batch.real_node_mask = torch.arange(\n'
            '            num_real + num_graphs, device=device\n'
            '        ) < num_real\n'
            '\n'
            '        counts = torch.bincount(graph_id, minlength=num_graphs)\n'
            '        if batch.get("log_deg", None) is not None:\n'
            '            vnode_log_deg = torch.log(counts.to(batch.log_deg.dtype) + 1)\n'
            '            batch.log_deg = torch.cat([batch.log_deg.view(-1), vnode_log_deg])\n'
            '        if batch.get("deg", None) is not None:\n'
            '            batch.deg = torch.cat([batch.deg.view(-1), counts.to(batch.deg.dtype)])\n'
            '        return batch\n'
            '\n'
            '\n'
            '@register_network(\'GritTransformer\')\n'
            'class GritTransformer(torch.nn.Module):\n'
        ),
        marker="class GlobalVNode(torch.nn.Module):",
        label="global virtual-node module",
    )
    _replace_exact(
        grit_model,
        old=(
            '        if cfg.gnn.layers_pre_mp > 0:\n'
            '            self.pre_mp = GNNPreMP(\n'
            '                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)\n'
            '            dim_in = cfg.gnn.dim_inner\n'
            '\n'
            '        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \\\n'
        ),
        new=(
            '        if cfg.gnn.layers_pre_mp > 0:\n'
            '            self.pre_mp = GNNPreMP(\n'
            '                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)\n'
            '            dim_in = cfg.gnn.dim_inner\n'
            '\n'
            '        self.global_vnode = (\n'
            '            GlobalVNode(cfg.gnn.dim_inner)\n'
            '            if cfg.gt.attn.get("global_vnode", False) else None\n'
            '        )\n'
            '\n'
            '        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \\\n'
        ),
        marker='GlobalVNode(cfg.gnn.dim_inner)',
        label="optional global VNode construction",
    )
    _replace_exact(
        grit_model,
        old=(
            '        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)\n'
            '\n'
            '    def forward(self, batch):\n'
            '        for module in self.children():\n'
            '            batch = module(batch)\n'
            '\n'
            '        return batch\n'
        ),
        new=(
            '        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)\n'
            '\n'
            '    def forward(self, batch):\n'
            '        batch = self.encoder(batch)\n'
            '        if hasattr(self, "rrwp_abs_encoder"):\n'
            '            batch = self.rrwp_abs_encoder(batch)\n'
            '            batch = self.rrwp_rel_encoder(batch)\n'
            '        if hasattr(self, "pre_mp"):\n'
            '            batch = self.pre_mp(batch)\n'
            '        if self.global_vnode is not None:\n'
            '            batch = self.global_vnode(batch)\n'
            '        batch = self.layers(batch)\n'
            '        if self.global_vnode is not None:\n'
            '            # The VNode is communication-only: preserve official real-node pooling.\n'
            '            batch.x = batch.x[batch.real_node_mask]\n'
            '            batch.batch = batch.batch[batch.real_node_mask]\n'
            '        return self.post_mp(batch)\n'
        ),
        marker="communication-only: preserve official real-node pooling",
        label="explicit forward with VNode pooling exclusion",
    )

    custom_train = repo_dir / "grit" / "train" / "custom_train.py"
    _replace_exact(
        custom_train,
        old=(
            "import logging\n"
            "import time\n"
        ),
        new=(
            "import logging\n"
            "import os\n"
            "import shutil\n"
            "import time\n"
        ),
        marker="import os\nimport shutil\nimport time",
        label="custom train recovery checkpoint import",
    )
    _replace_if_present(
        custom_train,
        old=(
            "import logging\n"
            "import os\n"
            "import time\n"
        ),
        new=(
            "import logging\n"
            "import os\n"
            "import shutil\n"
            "import time\n"
        ),
        label="custom train recovery checkpoint shutil import",
    )
    old_epoch_recovery_block = (
        "\n"
        "            if os.environ.get('GRIT_FORCE_EPOCH_CKPT', '0') == '1' and cfg.train.enable_ckpt:\n"
        "                save_ckpt(model, optimizer, scheduler, cur_epoch)\n"
        "                logging.info('Forced recovery checkpoint saved: %s', get_ckpt_path(get_ckpt_epoch(cur_epoch)))\n"
    )
    old_period_recovery_block = (
        "\n"
        "            if os.environ.get('GRIT_FORCE_RECOVERY_CKPT', '0') == '1' and cfg.train.enable_ckpt:\n"
        "                try:\n"
        "                    recovery_period = int(os.environ.get('GRIT_RECOVERY_CKPT_PERIOD', '100'))\n"
        "                except ValueError:\n"
        "                    recovery_period = 100\n"
        "                recovery_period = max(0, recovery_period)\n"
        "                recovery_reason = None\n"
        "                if best_epoch == cur_epoch:\n"
        "                    recovery_reason = 'new_best'\n"
        "                elif recovery_period and cur_epoch > 0 and cur_epoch % recovery_period == 0:\n"
        "                    recovery_reason = f'period_{recovery_period}'\n"
        "                if recovery_reason is not None:\n"
        "                    save_ckpt(model, optimizer, scheduler, cur_epoch)\n"
        "                    logging.info('Forced recovery checkpoint saved (%s): %s', recovery_reason, get_ckpt_path(get_ckpt_epoch(cur_epoch)))\n"
    )
    old_stable_recovery_block = (
        "\n"
        "            if os.environ.get('GRIT_FORCE_RECOVERY_CKPT', '0') == '1' and cfg.train.enable_ckpt:\n"
        "                try:\n"
        "                    recovery_period = int(os.environ.get('GRIT_RECOVERY_CKPT_PERIOD', '100'))\n"
        "                except ValueError:\n"
        "                    recovery_period = 100\n"
        "                recovery_period = max(0, recovery_period)\n"
        "                recovery_reason = None\n"
        "                if best_epoch == cur_epoch:\n"
        "                    recovery_reason = 'new_best'\n"
        "                elif recovery_period and cur_epoch > 0 and cur_epoch % recovery_period == 0:\n"
        "                    recovery_reason = f'period_{recovery_period}'\n"
        "                if recovery_reason is not None:\n"
        "                    save_ckpt(model, optimizer, scheduler, cur_epoch)\n"
        "                    recovery_path = get_ckpt_path(get_ckpt_epoch(cur_epoch))\n"
        "                    logging.info('Forced numbered recovery checkpoint saved (%s): %s', recovery_reason, recovery_path)\n"
        "                    recovery_dir = os.environ.get('GRIT_RECOVERY_CKPT_DIR', '')\n"
        "                    if recovery_dir:\n"
        "                        os.makedirs(recovery_dir, exist_ok=True)\n"
        "                        shutil.copy2(recovery_path, os.path.join(recovery_dir, 'latest.ckpt'))\n"
        "                        with open(os.path.join(recovery_dir, 'latest_epoch.txt'), 'w', encoding='utf-8') as f:\n"
        "                            f.write(f'{cur_epoch}\\n')\n"
        "                        if recovery_reason == 'new_best':\n"
        "                            shutil.copy2(recovery_path, os.path.join(recovery_dir, 'best.ckpt'))\n"
        "                            with open(os.path.join(recovery_dir, 'best_epoch.txt'), 'w', encoding='utf-8') as f:\n"
        "                                f.write(f'{cur_epoch}\\n')\n"
        "                        elif recovery_reason.startswith('period_'):\n"
        "                            shutil.copy2(recovery_path, os.path.join(recovery_dir, f'{recovery_reason}_epoch{cur_epoch}.ckpt'))\n"
        "                        logging.info('Stable recovery checkpoint copied (%s): %s', recovery_reason, recovery_dir)\n"
    )
    new_recovery_block = (
        "\n"
        "            if os.environ.get('GRIT_FORCE_RECOVERY_CKPT', '0') == '1' and cfg.train.enable_ckpt:\n"
        "                try:\n"
        "                    recovery_period = int(os.environ.get('GRIT_RECOVERY_CKPT_PERIOD', '100'))\n"
        "                except ValueError:\n"
        "                    recovery_period = 100\n"
        "                recovery_period = max(0, recovery_period)\n"
        "                recovery_reason = None\n"
        "                if os.environ.get('GRIT_SAVE_FIRST_RECOVERY_CKPT', '1') == '1' and cur_epoch == start_epoch:\n"
        "                    recovery_reason = 'first_after_resume'\n"
        "                elif best_epoch == cur_epoch:\n"
        "                    recovery_reason = 'new_best'\n"
        "                elif recovery_period and cur_epoch > 0 and cur_epoch % recovery_period == 0:\n"
        "                    recovery_reason = f'period_{recovery_period}'\n"
        "                if recovery_reason is not None:\n"
        "                    save_ckpt(model, optimizer, scheduler, cur_epoch)\n"
        "                    recovery_path = get_ckpt_path(get_ckpt_epoch(cur_epoch))\n"
        "                    logging.info('Forced numbered recovery checkpoint saved (%s): %s', recovery_reason, recovery_path)\n"
        "                    recovery_dir = os.environ.get('GRIT_RECOVERY_CKPT_DIR', '')\n"
        "                    if recovery_dir:\n"
        "                        os.makedirs(recovery_dir, exist_ok=True)\n"
        "                        shutil.copy2(recovery_path, os.path.join(recovery_dir, 'latest.ckpt'))\n"
        "                        with open(os.path.join(recovery_dir, 'latest_epoch.txt'), 'w', encoding='utf-8') as f:\n"
        "                            f.write(f'{cur_epoch}\\n')\n"
        "                        if recovery_reason == 'new_best':\n"
        "                            shutil.copy2(recovery_path, os.path.join(recovery_dir, 'best.ckpt'))\n"
        "                            with open(os.path.join(recovery_dir, 'best_epoch.txt'), 'w', encoding='utf-8') as f:\n"
        "                                f.write(f'{cur_epoch}\\n')\n"
        "                        elif recovery_reason == 'first_after_resume':\n"
        "                            shutil.copy2(recovery_path, os.path.join(recovery_dir, 'first_after_resume.ckpt'))\n"
        "                            with open(os.path.join(recovery_dir, 'first_after_resume_epoch.txt'), 'w', encoding='utf-8') as f:\n"
        "                                f.write(f'{cur_epoch}\\n')\n"
        "                        elif recovery_reason.startswith('period_'):\n"
        "                            shutil.copy2(recovery_path, os.path.join(recovery_dir, f'{recovery_reason}_epoch{cur_epoch}.ckpt'))\n"
        "                        logging.info('Stable recovery checkpoint copied (%s): %s', recovery_reason, recovery_dir)\n"
    )
    _replace_if_present(
        custom_train,
        old_epoch_recovery_block,
        new_recovery_block,
        "custom train recovery checkpoint semantics",
    )
    _replace_if_present(
        custom_train,
        old_period_recovery_block,
        new_recovery_block,
        "custom train stable best checkpoint semantics",
    )
    _replace_if_present(
        custom_train,
        old_stable_recovery_block,
        new_recovery_block,
        "custom train first-resume checkpoint semantics",
    )
    _insert_after(
        custom_train,
        anchor=(
            "            logging.info(\n"
            "                f\"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s \"\n"
            "                f\"(avg {np.mean(full_epoch_times):.1f}s) | \"\n"
            "                f\"Best so far: epoch {best_epoch}\\t\"\n"
            "                f\"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\\t\"\n"
            "                f\"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\\t\" \n"
            "                f\"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}\\n\"\n"
            "                f\"-----------------------------------------------------------\"\n"
            "            )\n"
        ),
        insertion=new_recovery_block,
        marker="GRIT_FORCE_RECOVERY_CKPT",
        label="custom train guaranteed recovery checkpoint",
    )

    patch_note = drive_dir / "patches" / "zinc_grit_rrwp_khop_patch.txt"
    patch_note.parent.mkdir(parents=True, exist_ok=True)
    patch_note.write_text(
        "\n".join([
            "Configurable k-hop GRIT ZINC control patch",
            f"official_repo: {OFFICIAL_REPO}",
            f"official_commit: {OFFICIAL_COMMIT}",
            f"dense_reference_config: {DENSE_OFFICIAL_CFG}",
            f"k_hop_config: {OFFICIAL_CFG}",
            f"hops: {args.hops}",
            f"global_vnode: {args.global_vnode}",
            f"parameter_count_guard: {expected_param_count(args)}",
            "scientific_change: gt.attn.full_attn=False and gt.attn.sparsity=k_hop",
            "support: exact shortest-path distance <= k plus self after RRWP encoding",
            "vnode: one learned graph token, bidirectional global attention, excluded from pooling",
        ]) + "\n",
        encoding="utf-8",
    )
    log(f"[patch] wrote patch provenance note: {patch_note}")


def install_grit_editable(repo_dir: Path) -> None:
    pip_install(["-e", str(repo_dir)])


def verify_recovery_checkpoint_patch(repo_dir: Path) -> None:
    custom_train = repo_dir / "grit" / "train" / "custom_train.py"
    text = custom_train.read_text(encoding="utf-8", errors="replace")
    required = [
        "GRIT_FORCE_RECOVERY_CKPT",
        "GRIT_RECOVERY_CKPT_DIR",
        "GRIT_SAVE_FIRST_RECOVERY_CKPT",
        "first_after_resume.ckpt",
        "best.ckpt",
        "latest.ckpt",
        "Stable recovery checkpoint copied",
        "Forced numbered recovery checkpoint saved",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(
            "Recovery checkpoint patch is not active in the official GRIT checkout. "
            f"Missing tokens in {custom_train}: {missing}. "
            "Restart Colab with the latest standalone GRIT_khop_ZINC.py and pass --force-fresh-repo."
        )
    lines = text.splitlines()
    hit = next(i for i, line in enumerate(lines) if "GRIT_FORCE_RECOVERY_CKPT" in line)
    lo = max(0, hit - 2)
    hi = min(len(lines), hit + 22)
    log("[checkpoint-guarantee] Verified recovery checkpoint patch in official GRIT custom_train.py:")
    for line_no in range(lo, hi):
        log(f"[checkpoint-guarantee:source] {line_no + 1:04d}: {lines[line_no]}")


def verify_khop_attention_patch(repo_dir: Path) -> None:
    """Fail before training if any required scientific patch segment is absent."""
    required_by_file = {
        repo_dir / "grit" / "config" / "gt_config.py": [
            'cfg.gt.attn.sparsity = "full"',
            "cfg.gt.attn.hops = 1",
            "cfg.gt.attn.global_vnode = False",
        ],
        repo_dir / "grit" / "encoder" / "rrwp_encoder.py": [
            "max_hops=None",
            "needed_channels = self.max_hops + 1",
            "mask_index = rrwp_idx[:, reachable]",
        ],
        repo_dir / "grit" / "network" / "grit_model.py": [
            'attn_sparsity == "k_hop"',
            "max_hops=cfg.gt.attn.hops",
            "class GlobalVNode(torch.nn.Module):",
            'cfg.gt.attn.get("global_vnode", False)',
            "communication-only: preserve official real-node pooling",
        ],
    }
    errors = []
    for path, tokens in required_by_file.items():
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path}: syntax error: {exc}")
        missing = [token for token in tokens if token not in text]
        if missing:
            errors.append(f"{path}: missing {missing}")
    if errors:
        raise RuntimeError(
            "The k-hop/VNode patch is incomplete; use --force-fresh-repo.\n  - "
            + "\n  - ".join(errors)
        )
    log("[patch-check] Verified k-hop support and optional global-VNode source patches.")


def get_nested(d: Mapping, path: Sequence[str]):
    cur = d
    for key in path:
        cur = cur[key]
    return cur


def _semantic_cfg_equal(actual, expected) -> bool:
    """Compare YAML config values semantically, not merely by Python type.

    Some PyYAML versions parse scientific notation such as `1e-5` as a string,
    while others parse it as a float. For config validation we care that the
    official value is numerically identical, not whether the parser returned
    `"1e-5"` or `1e-05`. Booleans are intentionally kept as booleans, since
    bool is a subclass of int in Python.
    """
    if actual == expected:
        return True

    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected

    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) <= max(1e-12, 1e-9 * abs(float(expected)))
        except (TypeError, ValueError):
            return False

    return False


def expected_param_count(args: argparse.Namespace) -> int:
    if args.expected_params is not None:
        return int(args.expected_params)
    return EXPECTED_ZINC_GRIT_RRWP_PARAMS + (VNODE_PARAM_COUNT if args.global_vnode else 0)


def validate_official_config(
    repo_dir: Path,
    allow_drift: bool,
    args: argparse.Namespace,
) -> None:
    import yaml

    cfg_path = repo_dir / OFFICIAL_CFG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Official config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    expected_values = dict(EXPECTED_CFG_VALUES)
    expected_values[("gt", "attn", "hops")] = args.hops
    expected_values[("gt", "attn", "global_vnode")] = args.global_vnode

    errors: List[str] = []
    for path, expected in expected_values.items():
        try:
            actual = get_nested(cfg, path)
        except Exception:
            errors.append(f"missing {'.'.join(path)}; expected {expected!r}")
            continue
        if not _semantic_cfg_equal(actual, expected):
            errors.append(f"{'.'.join(path)} = {actual!r}; expected {expected!r}")

    if errors:
        msg = "Official ZINC config does not match the expected GRIT ZINC+RRWP setup:\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        if allow_drift:
            log("[config-warning] " + msg)
        else:
            raise RuntimeError(msg + "\nPass --allow-upstream-config-drift to run anyway.")

    log(f"[config] {args.hops}-hop GRIT ZINC+RRWP config validated (global_vnode={args.global_vnode}).")
    log(f"[config] Key setup: PyG-ZINC/subset, graph regression, RRWP, GritTransformer, <= {args.hops}-hop attention support, 10 layers, 64 hidden dim, 8 heads, batch 32, L1/MAE, 2000 epochs, eval every epoch.")
    log("[paper-check] Dense-reference GRIT paper Table 9 ZINC settings are preserved: layers=10, hidden_dim=64, heads=8, dropout=0, attn_dropout=0.2, pooling=sum/add, PE=RRWP-21, PE_encoder=linear, batch=32, lr=0.001, epochs=2000, warmup=50, weight_decay=1e-5.")
    log(f"[paper-check] Control intervention: full_attn=False, sparsity=k_hop, hops={args.hops}, global_vnode={args.global_vnode}. No width/depth/head/schedule changes.")
    log(f"[paper-check] Expected trainable parameter count: {expected_param_count(args)}")


def build_training_command(args: argparse.Namespace, drive_dir: Path) -> List[str]:
    results_dir = drive_dir / "results"
    dataset_dir = drive_dir / "datasets"
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = not args.checkpoint_every_epoch
    ckpt_clean = False if (args.keep_all_checkpoints or args.checkpoint_every_epoch or args.guaranteed_checkpoints) else True

    # Only runtime/storage/logging overrides. Model/dataset/task/optimizer hyperparams
    # remain matched to configs/GRIT/zinc-GRIT-RRWP.yaml except for the explicit
    # k-hop/VNode attention intervention in OFFICIAL_CFG. max_epoch=2000 and
    # eval_period=1 are repeated defensively because the requested run length is exact
    # and these are also the paper/config values. By default checkpoint mode matches
    # the official ZINC config. Optional Colab recovery flags change storage policy
    # only, not the model/data/optimizer/schedule.
    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--cfg",
        OFFICIAL_CFG,
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
        "2000",
        "train.eval_period",
        "1",
        "train.enable_ckpt",
        "True",
        "train.ckpt_period",
        str(args.ckpt_period),
        "train.ckpt_best",
        "True" if ckpt_best else "False",
        "train.ckpt_clean",
        "True" if ckpt_clean else "False",
        "train.auto_resume",
        "True" if args.auto_resume else "False",
        "num_threads",
        str(args.num_threads),
    ]
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
    for pattern in (
        r"epoch[=_-]?(\d+)",
        r"ep[=_-]?(\d+)",
        r"ckpt[=_-]?(\d+)",
        r"/(\d+)\.ckpt$",
        r"/(\d+)\.(?:pt|pth)$",
    ):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    try:
        return int(path.stem)
    except Exception:
        return None


def parse_training_summary_from_logs(log_paths: Sequence[Path]) -> dict:
    by_epoch: dict[int, dict[str, dict]] = {}
    for path in log_paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
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

    best_epoch: int | None = None
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
        "best_epoch": best_epoch,
        "best_train": best_splits.get("train", {}),
        "best_val": best_splits.get("val", {}),
        "best_test": best_splits.get("test", {}),
        "log_files": [str(path) for path in log_paths],
    }


def write_checkpoint_audit(drive_dir: Path, wrapper_log: Path, seed: int) -> Path:
    result_root = drive_dir / "results"
    log_root = drive_dir / "wrapper_logs"
    log_paths = []
    if wrapper_log.exists():
        log_paths.append(wrapper_log)
    if log_root.exists():
        log_paths.extend(path for path in log_root.rglob("*.log") if path.is_file())
    log_paths = sorted(set(log_paths), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    summary = parse_training_summary_from_logs(log_paths)

    candidates = checkpoint_candidates(result_root)
    best_epoch = summary.get("best_epoch")
    matching_best = [
        path for path in candidates
        if best_epoch is not None and checkpoint_epoch(path) == int(best_epoch)
    ]
    latest = candidates[0] if candidates else None
    audit = {
        "seed": seed,
        "drive_dir": str(drive_dir),
        "results_root": str(result_root),
        "best_epoch_from_logs": best_epoch,
        "best_val": summary.get("best_val", {}),
        "best_test": summary.get("best_test", {}),
        "latest_checkpoint_by_mtime": str(latest) if latest else None,
        "latest_checkpoint_epoch": checkpoint_epoch(latest) if latest else None,
        "best_epoch_checkpoint_matches": [str(path) for path in matching_best],
        "checkpoint_candidates": [
            {
                "path": str(path),
                "epoch": checkpoint_epoch(path),
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
                "bytes": path.stat().st_size,
            }
            for path in candidates
        ],
        "parsed_logs": summary.get("log_files", []),
        "history_epochs": summary.get("history_epochs", 0),
    }
    audit_path = drive_dir / f"checkpoint_audit_seed{seed}.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)
    latest_audit = drive_dir / "latest_checkpoint_audit.json"
    with latest_audit.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    log(f"[checkpoint-audit] best_epoch_from_logs={best_epoch}")
    if matching_best:
        log(f"[checkpoint-audit] matching best checkpoint: {matching_best[0]}")
    elif best_epoch is not None:
        log(
            "[checkpoint-audit:WARNING] No checkpoint filename matched the parsed best epoch. "
            "GraphGym may have kept a generic best checkpoint, or checkpoint cleaning may have removed older files."
        )
    if latest:
        log(f"[checkpoint-audit] latest checkpoint by mtime: {latest} (epoch={checkpoint_epoch(latest)})")
    else:
        log("[checkpoint-audit:WARNING] No checkpoint files found under Drive results root.")
    log(f"[checkpoint-audit] wrote: {audit_path}")
    return audit_path


def safe_path_fragment(value: str) -> str:
    fragment = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return fragment or "run"


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
    """Remove argv fragments injected by IPython/Colab, but keep real user errors strict.

    When a script is executed inside an IPython kernel via `%run`, `exec(open(...).read())`,
    or a copied notebook cell, `sys.argv` may contain e.g.

        -f /root/.local/share/jupyter/runtime/kernel-....json

    Plain `argparse.parse_args()` treats that as an unknown user argument and exits.
    We remove only this specific kernel-file pattern, then still use parse_args() so genuine
    typos such as `--repat` continue to fail loudly.
    """
    cleaned: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if (
            a == "-f"
            and i + 1 < len(argv)
            and "kernel-" in argv[i + 1]
            and argv[i + 1].endswith(".json")
        ):
            log(f"[args] Ignoring Colab/Jupyter kernel argument: {a} {argv[i + 1]}")
            i += 2
            continue
        if (
            a.startswith("-f=")
            and "kernel-" in a
            and a.endswith(".json")
        ):
            log(f"[args] Ignoring Colab/Jupyter kernel argument: {a}")
            i += 1
            continue
        cleaned.append(a)
        i += 1
    return cleaned


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train configurable k-hop GRIT+RRWP on the ZINC subset in Colab, with an optional global VNode.",
        epilog=textwrap.dedent(
            """
            Examples:
              !python GRIT_khop_ZINC.py --hops 2
              %run GRIT_khop_ZINC.py --hops 3 --global-vnode
              !python GRIT_khop_ZINC.py --skip-install --hops 1 --auto-resume
            """
        ),
    )
    p.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    p.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/grit_zinc_khop"))
    p.add_argument("--repo-dir", type=Path, default=Path("/content/GRIT_khop"))
    p.add_argument("--repo-url", type=str, default=OFFICIAL_REPO)
    p.add_argument("--branch", type=str, default="main")
    p.add_argument("--commit", type=str, default=OFFICIAL_COMMIT, help="Pin the official GRIT repo to this commit. Pass an empty string to use the branch HEAD.")
    p.add_argument("--hops", "-k", type=int, default=1, help="Attention radius in graph hops (1..20 with RRWP-21). Default: 1.")
    p.add_argument("--global-vnode", action="store_true", help="Add one learned global virtual node per graph to all attention layers; exclude it from final pooling.")
    p.add_argument("--expected-params", type=int, default=None, help="Override the automatic parameter-count guard (473473 without VNode; 473537 with VNode).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--name-tag", type=str, default=None, help="Run name. Default is derived from --hops and --global-vnode.")
    p.add_argument("--ckpt-period", type=int, default=1, help="GraphGym checkpoint period in epochs. Default: 1 for reliable Colab/Drive recovery.")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--pyg-version", type=str, default="2.2.0")
    p.add_argument("--skip-install", action="store_true", help="Do not pip-install dependencies.")
    p.add_argument("--official-torch112", action="store_true", help="Try to install torch==1.12.1+cu113. Requires Python <=3.10 and may not work on current Colab.")
    p.add_argument("--force-fresh-repo", action="store_true", help="Delete and reclone the GRIT repo directory before running.")
    p.add_argument("--allow-upstream-config-drift", action="store_true", help="Warn instead of aborting if upstream config differs from expected official values.")
    p.add_argument("--allow-param-count-drift", action="store_true", help="Warn instead of aborting if Num parameters is not the expected ZINC GRIT+RRWP paper count.")
    p.add_argument("--wandb", action="store_true", help="Enable W&B. Default is disabled for unattended Colab runs.")
    p.add_argument("--console-verbosity", choices=["compact", "standard", "full"], default="compact", help="Notebook stdout filtering. Full raw output is always saved to Drive.")
    p.add_argument("--console-epoch-period", type=int, default=1, help="In compact/standard mode, print every Nth official epoch summary. Default: every epoch.")
    p.add_argument("--accelerator", type=str, default="cuda:0", help="Runtime device override passed to GRIT. Default: cuda:0 for Colab GPU.")
    p.add_argument("--auto-resume", action="store_true", default=True, help="Resume from existing run directory/checkpoints if available. Default: true.")
    p.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")
    p.add_argument(
        "--keep-all-checkpoints",
        action="store_true",
        help=(
            "Set train.ckpt_clean=False so GraphGym does not remove older checkpoint files. "
            "Useful for Colab/Drive audit runs; default keeps the official ZINC config behavior."
        ),
    )
    p.add_argument(
        "--checkpoint-every-epoch",
        action="store_true",
        help=(
            "Storage-only recovery mode: set train.ckpt_best=False, train.ckpt_clean=False, "
            "and save every --ckpt-period epochs. This guarantees checkpoint file updates in "
            "Colab/Drive but no longer exactly matches the official checkpointing policy."
        ),
    )
    p.add_argument(
        "--guaranteed-checkpoints",
        action="store_true",
        default=True,
        help=(
            "Patch official GRIT custom_train.py to call GraphGym save_ckpt when validation reaches a new best "
            "and at periodic recovery epochs. Default: enabled for Colab recovery. This changes only storage/recovery behavior."
        ),
    )
    p.add_argument("--no-guaranteed-checkpoints", action="store_false", dest="guaranteed_checkpoints")
    p.add_argument(
        "--recovery-ckpt-period",
        type=int,
        default=100,
        help="When --guaranteed-checkpoints is enabled, also save recovery checkpoints at official epochs divisible by this value. Default: 100. Use 0 to disable periodic recovery checkpoints.",
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _strip_colab_kernel_args(argv)
    args = p.parse_args(argv)
    if not 1 <= args.hops <= 20:
        p.error("--hops must be between 1 and 20 because RRWP-21 stores identity plus walks of lengths 1..20")
    if args.name_tag is None:
        args.name_tag = f"ColabDrive.{args.hops}hop.GRITwRRWP" + (".VNode" if args.global_vnode else "")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Notebook-friendly entry point.

    In a Colab cell, prefer calling e.g.

        main([])
        main(["--skip-install"])
        main(["--skip-install", "--seed", "42"])

    Passing a list avoids accidental parsing of IPython kernel arguments. If
    argv is None, the script still parses sys.argv for normal CLI usage.
    """
    args = parse_args(argv)

    mount_drive(args.drive_mount)
    args.drive_dir.mkdir(parents=True, exist_ok=True)

    compat_shim_dir = None
    if sys.version_info >= (3, 12):
        compat_shim_dir = write_py312_compat_shim(args.drive_dir)

    if not args.skip_install:
        install_dependencies(args)
    else:
        log("[deps] Skipping dependency installation (--skip-install).")

    commit = clone_or_update_repo(args.repo_dir, args.repo_url, args.branch, args.commit or None, args.force_fresh_repo)
    apply_khop_patch(args.repo_dir, args.drive_dir, args)
    verify_khop_attention_patch(args.repo_dir)
    verify_recovery_checkpoint_patch(args.repo_dir)
    install_grit_editable(args.repo_dir)
    validate_official_config(args.repo_dir, args.allow_upstream_config_drift, args)
    print_environment_summary(args.drive_dir, args.repo_dir, commit)

    cmd = build_training_command(args, args.drive_dir)
    variant = f"{args.hops}hop" + ("_vnode" if args.global_vnode else "")
    wrapper_log = args.drive_dir / "wrapper_logs" / f"grit_zinc_{variant}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    train_env = env_with_py312_compat(compat_shim_dir)
    if args.guaranteed_checkpoints:
        recovery_dir = args.drive_dir / "results" / "_recovery_checkpoints" / f"seed{args.seed}_{safe_path_fragment(args.name_tag)}"
        recovery_dir.mkdir(parents=True, exist_ok=True)
        train_env["GRIT_FORCE_RECOVERY_CKPT"] = "1"
        train_env["GRIT_RECOVERY_CKPT_PERIOD"] = str(max(0, int(args.recovery_ckpt_period)))
        train_env["GRIT_RECOVERY_CKPT_DIR"] = str(recovery_dir)
        train_env["GRIT_SAVE_FIRST_RECOVERY_CKPT"] = "1"
        train_env.pop("GRIT_FORCE_EPOCH_CKPT", None)
        log(
            "[checkpoint-guarantee] Enabled: official GRIT will save compatible recovery checkpoints "
            f"after the first completed epoch, on each new best, and at official epochs divisible by {max(0, int(args.recovery_ckpt_period))}."
        )
        log(f"[checkpoint-guarantee] Stable best checkpoint path: {recovery_dir / 'best.ckpt'}")
        log(f"[checkpoint-guarantee] Stable latest checkpoint path: {recovery_dir / 'latest.ckpt'}")
        log(f"[checkpoint-guarantee] First-resume checkpoint path: {recovery_dir / 'first_after_resume.ckpt'}")
    else:
        train_env.pop("GRIT_FORCE_RECOVERY_CKPT", None)
        train_env.pop("GRIT_RECOVERY_CKPT_PERIOD", None)
        train_env.pop("GRIT_RECOVERY_CKPT_DIR", None)
        train_env.pop("GRIT_SAVE_FIRST_RECOVERY_CKPT", None)
        train_env.pop("GRIT_FORCE_EPOCH_CKPT", None)
        log("[checkpoint-guarantee] Disabled: using only official GraphGym checkpoint policy.")
    rc = run_streaming_to_console_and_log(
        cmd,
        cwd=args.repo_dir,
        log_file=wrapper_log,
        env=train_env,
        expected_params=expected_param_count(args),
        allow_param_count_drift=args.allow_param_count_drift,
        console_verbosity=args.console_verbosity,
        console_epoch_period=args.console_epoch_period,
    )
    if rc != 0:
        raise SystemExit(rc)

    log("\n[done] Training process completed successfully.")
    write_checkpoint_audit(args.drive_dir, wrapper_log, args.seed)
    log(f"[done] Results/checkpoints root: {args.drive_dir / 'results'}")
    log(f"[done] Wrapper log: {wrapper_log}")


if __name__ == "__main__":
    main()
