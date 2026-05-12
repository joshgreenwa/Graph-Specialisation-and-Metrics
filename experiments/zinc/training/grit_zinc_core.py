# Extracted from grit_ZINC_core.ipynb
# Keep the notebook and script in sync when changing experiment logic.

#!/usr/bin/env python3
"""
Colab runner for the official GRIT ZINC subset configuration.

This script intentionally DOES NOT reimplement GRIT. It clones the official
repository and launches the official ZINC RRWP config:

    python -u main.py --cfg configs/GRIT/zinc-GRIT-RRWP.yaml wandb.use False

with only non-scientific runtime overrides for Google Drive output/checkpoints,
notebook-safe console filtering, and resume behavior. Compact stdout prints
current train/val/test loss+MAE every selected epoch, plus the best validation
epoch so far. The official ZINC model/task/training config is validated before
launch. The repo is pinned by default to a known official main-branch commit,
and the run aborts before training if the logged model parameter count is not
exactly 473,473, matching the GRIT paper ZINC/ZINC-full hyperparameter table.

Suggested Colab usage:

    # Upload this file to Colab, then run:
    from train_grit_zinc_official_colab import main
    main(["--skip-install"])

For a fresh runtime, omit --skip-install:

    main([])

Checkpoints/logs are written by default to:

    /content/drive/MyDrive/grit_zinc_official/results

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
OFFICIAL_CFG = "configs/GRIT/zinc-GRIT-RRWP.yaml"
# Current official main-branch commit observed from GitHub commit history.
OFFICIAL_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
EXPECTED_ZINC_GRIT_RRWP_PARAMS = 473_473
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
    ("gt", "attn", "full_attn"): True,
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
# Auto-generated by train_grit_zinc_official_colab.py.
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
    pip_install([
        "pyg-lib",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
        "-f",
        pyg_wheel_url,
    ])

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
        run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
        run_cmd(["git", "checkout", branch], cwd=repo_dir)
        run_cmd(["git", "pull", "--ff-only", "origin", branch], cwd=repo_dir)

    if commit:
        log(f"[repo] Pinning GRIT to commit: {commit}")
        run_cmd(["git", "checkout", commit], cwd=repo_dir)

    resolved_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    log(f"[repo] Using GRIT commit: {resolved_commit}")
    return resolved_commit


def install_grit_editable(repo_dir: Path) -> None:
    pip_install(["-e", str(repo_dir)])


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


def validate_official_config(repo_dir: Path, allow_drift: bool) -> None:
    import yaml

    cfg_path = repo_dir / OFFICIAL_CFG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Official config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    errors: List[str] = []
    for path, expected in EXPECTED_CFG_VALUES.items():
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

    log("[config] Official GRIT ZINC+RRWP config validated.")
    log("[config] Key setup: PyG-ZINC/subset, graph regression, RRWP, GritTransformer, full GRIT attention, 10 layers, 64 hidden dim, 8 heads, batch 32, L1/MAE, 2000 epochs, eval every epoch.")
    log("[paper-check] Enforcing GRIT paper Table 9 ZINC settings: layers=10, hidden_dim=64, heads=8, dropout=0, attn_dropout=0.2, pooling=sum/add, PE=RRWP-21, PE_encoder=linear, batch=32, lr=0.001, epochs=2000, warmup=50, weight_decay=1e-5.")
    log(f"[paper-check] Expected paper parameter count: {EXPECTED_ZINC_GRIT_RRWP_PARAMS}")


def build_training_command(args: argparse.Namespace, drive_dir: Path) -> List[str]:
    results_dir = drive_dir / "results"
    dataset_dir = drive_dir / "datasets"
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Only runtime/storage/logging overrides. Model/dataset/task/optimizer hyperparams
    # remain those of configs/GRIT/zinc-GRIT-RRWP.yaml. max_epoch=2000 and
    # eval_period=1 are repeated defensively because the requested run length is exact
    # and these are also the paper/config values. Checkpoint mode is deliberately NOT
    # overridden: the official GRIT config uses ckpt_best=True and ckpt_clean=True.
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
        "train.auto_resume",
        "True" if args.auto_resume else "False",
        "num_threads",
        str(args.num_threads),
    ]
    if args.accelerator:
        cmd.extend(["accelerator", args.accelerator])
    return cmd


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
        description="Train official GRIT+RRWP on the ZINC subset in Colab, saving checkpoints to Drive.",
        epilog=textwrap.dedent(
            """
            Examples:
              !python train_grit_zinc_official_colab.py
              %run train_grit_zinc_official_colab.py
              !python train_grit_zinc_official_colab.py --seed 42 --name-tag GRITwRRWP.seed42
              !python train_grit_zinc_official_colab.py --skip-install --auto-resume
            """
        ),
    )
    p.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    p.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/grit_zinc_official"))
    p.add_argument("--repo-dir", type=Path, default=Path("/content/GRIT"))
    p.add_argument("--repo-url", type=str, default=OFFICIAL_REPO)
    p.add_argument("--branch", type=str, default="main")
    p.add_argument("--commit", type=str, default=OFFICIAL_COMMIT, help="Pin the official GRIT repo to this commit. Pass an empty string to use the branch HEAD.")
    p.add_argument("--expected-params", type=int, default=EXPECTED_ZINC_GRIT_RRWP_PARAMS, help="Abort unless GRIT logs this exact model parameter count.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--name-tag", type=str, default="ColabDrive.official.GRITwRRWP")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--pyg-version", type=str, default="2.2.0")
    p.add_argument("--skip-install", action="store_true", help="Do not pip-install dependencies.")
    p.add_argument("--official-torch112", action="store_true", help="Try to install torch==1.12.1+cu113. Requires Python <=3.10 and may not work on current Colab.")
    p.add_argument("--force-fresh-repo", action="store_true", help="Delete and reclone /content/GRIT before running.")
    p.add_argument("--allow-upstream-config-drift", action="store_true", help="Warn instead of aborting if upstream config differs from expected official values.")
    p.add_argument("--allow-param-count-drift", action="store_true", help="Warn instead of aborting if Num parameters is not the expected ZINC GRIT+RRWP paper count.")
    p.add_argument("--wandb", action="store_true", help="Enable W&B. Default is disabled for unattended Colab runs.")
    p.add_argument("--console-verbosity", choices=["compact", "standard", "full"], default="compact", help="Notebook stdout filtering. Full raw output is always saved to Drive.")
    p.add_argument("--console-epoch-period", type=int, default=1, help="In compact/standard mode, print every Nth official epoch summary. Default: every epoch.")
    p.add_argument("--accelerator", type=str, default="cuda:0", help="Runtime device override passed to GRIT. Default: cuda:0 for Colab GPU.")
    p.add_argument("--auto-resume", action="store_true", default=True, help="Resume from existing run directory/checkpoints if available. Default: true.")
    p.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _strip_colab_kernel_args(argv)
    return p.parse_args(argv)


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
    install_grit_editable(args.repo_dir)
    validate_official_config(args.repo_dir, args.allow_upstream_config_drift)
    print_environment_summary(args.drive_dir, args.repo_dir, commit)

    cmd = build_training_command(args, args.drive_dir)
    wrapper_log = args.drive_dir / "wrapper_logs" / f"grit_zinc_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    train_env = env_with_py312_compat(compat_shim_dir)
    rc = run_streaming_to_console_and_log(
        cmd,
        cwd=args.repo_dir,
        log_file=wrapper_log,
        env=train_env,
        expected_params=args.expected_params,
        allow_param_count_drift=args.allow_param_count_drift,
        console_verbosity=args.console_verbosity,
        console_epoch_period=args.console_epoch_period,
    )
    if rc != 0:
        raise SystemExit(rc)

    log("\n[done] Training process completed successfully.")
    log(f"[done] Results/checkpoints root: {args.drive_dir / 'results'}")
    log(f"[done] Wrapper log: {wrapper_log}")


if __name__ == "__main__":
    main()
