#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab runner for the official Microsoft Graphormer-Slim model on ZINC.

Colab notebook export name: Graphormer_ZINC_Official.ipynb

This script intentionally does not reimplement Graphormer. It clones the
official Microsoft repository:

    https://github.com/microsoft/Graphormer

and validates the official ZINC configuration from:

    examples/property_prediction/zinc.sh

The official ZINC config is a fairseq command-line config rather than YAML. The
original upstream command is:

    fairseq-train --user-dir ../../graphormer \
      --dataset-name zinc --dataset-source pyg --task graph_prediction \
      --criterion l1_loss --arch graphormer_slim --num-classes 1 \
      --attention-dropout 0.1 --act-dropout 0.1 --dropout 0.0 \
      --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 \
      --clip-norm 5.0 --weight-decay 0.01 \
      --lr-scheduler polynomial_decay --power 1 \
      --warmup-updates 60000 --total-num-update 400000 \
      --lr 2e-4 --end-learning-rate 1e-9 \
      --batch-size 64 --fp16 --data-buffer-size 20 \
      --encoder-layers 12 --encoder-embed-dim 80 \
      --encoder-ffn-embed-dim 80 --encoder-attention-heads 8 \
      --max-epoch 10000 --save-dir ./ckpts

This runner then applies a ZINC-small derived default requested for controlled
comparison with GRIT (Ma et al., ICML 2023): 2000 epochs, batch size 32, AdamW
semantics via fairseq's adam optimizer, lr=1e-3, weight_decay=1e-5, 50 warmup
epochs, cosine decay to 1e-6, and eval/checkpointing each epoch. The Graphormer
model stays on the official implementation but defaults to 10 layers, d=80,
8 heads, FFN dim=116, and explicit ZINC-sized embedding vocabularies for an
estimated 475,529 parameters.

Runtime/storage overrides are also applied by default: Google Drive output and
dataset cache paths, seed, checkpoint retention, logging format, CUDA visibility,
and dependency compatibility for modern Colab runtimes. Pass
--dependency-profile official-2021 to request the exact dependency pins from
Microsoft's install.sh; that profile requires a Python version with wheels for
torch 1.9.1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Mapping, MutableSequence, Sequence


OFFICIAL_REPO = "https://github.com/microsoft/Graphormer.git"
OFFICIAL_COMMIT = "a04573c40705fb174db261bb746a8258d00992f5"
OFFICIAL_FAIRSEQ_COMMIT = "98ebe4f1ada75d006717d84f9d603519d8ff5579"
OFFICIAL_ZINC_SCRIPT = Path("examples/property_prediction/zinc.sh")
OFFICIAL_TRAIN_CWD = Path("examples/property_prediction")

DEFAULT_ENCODER_LAYERS = 10
DEFAULT_ENCODER_EMBED_DIM = 80
DEFAULT_ENCODER_FFN_EMBED_DIM = 116
DEFAULT_ENCODER_ATTENTION_HEADS = 8

DEFAULT_NUM_ATOMS = 29
DEFAULT_NUM_EDGES = 6
DEFAULT_NUM_IN_DEGREE = 6
DEFAULT_NUM_OUT_DEGREE = 6
DEFAULT_NUM_SPATIAL = 64
DEFAULT_NUM_EDGE_DIS = 128
DEFAULT_MULTI_HOP_MAX_DIST = 5
DEFAULT_SPATIAL_POS_MAX = 1024

DEFAULT_TRAIN_SIZE = 10_000
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_EPOCH = 2000
DEFAULT_WARMUP_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_MIN_LR = 1e-6
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_CLIP_NORM = 5.0
DEFAULT_PARAM_COUNT_ESTIMATE = 475_529

OFFICIAL_ZINC_ARGS = [
    "--user-dir", "../../graphormer",
    "--num-workers", "16",
    "--ddp-backend=legacy_ddp",
    "--dataset-name", "zinc",
    "--dataset-source", "pyg",
    "--task", "graph_prediction",
    "--criterion", "l1_loss",
    "--arch", "graphormer_slim",
    "--num-classes", "1",
    "--attention-dropout", "0.1",
    "--act-dropout", "0.1",
    "--dropout", "0.0",
    "--optimizer", "adam",
    "--adam-betas", "(0.9, 0.999)",
    "--adam-eps", "1e-8",
    "--clip-norm", "5.0",
    "--weight-decay", "0.01",
    "--lr-scheduler", "polynomial_decay",
    "--power", "1",
    "--warmup-updates", "60000",
    "--total-num-update", "400000",
    "--lr", "2e-4",
    "--end-learning-rate", "1e-9",
    "--batch-size", "64",
    "--fp16",
    "--data-buffer-size", "20",
    "--encoder-layers", "12",
    "--encoder-embed-dim", "80",
    "--encoder-ffn-embed-dim", "80",
    "--encoder-attention-heads", "8",
    "--max-epoch", "10000",
    "--save-dir", "./ckpts",
]

EXPECTED_ZINC_OPTIONS = {
    "--user-dir": "../../graphormer",
    "--num-workers": "16",
    "--ddp-backend": "legacy_ddp",
    "--dataset-name": "zinc",
    "--dataset-source": "pyg",
    "--task": "graph_prediction",
    "--criterion": "l1_loss",
    "--arch": "graphormer_slim",
    "--num-classes": "1",
    "--attention-dropout": "0.1",
    "--act-dropout": "0.1",
    "--dropout": "0.0",
    "--optimizer": "adam",
    "--adam-betas": "(0.9, 0.999)",
    "--adam-eps": "1e-8",
    "--clip-norm": "5.0",
    "--weight-decay": "0.01",
    "--lr-scheduler": "polynomial_decay",
    "--power": "1",
    "--warmup-updates": "60000",
    "--total-num-update": "400000",
    "--lr": "2e-4",
    "--end-learning-rate": "1e-9",
    "--batch-size": "64",
    "--data-buffer-size": "20",
    "--encoder-layers": "12",
    "--encoder-embed-dim": "80",
    "--encoder-ffn-embed-dim": "80",
    "--encoder-attention-heads": "8",
    "--max-epoch": "10000",
    "--save-dir": "./ckpts",
}
EXPECTED_ZINC_FLAGS = {"--fp16"}


def estimate_graphormer_params(
    *,
    layers: int = DEFAULT_ENCODER_LAYERS,
    embed_dim: int = DEFAULT_ENCODER_EMBED_DIM,
    ffn_dim: int = DEFAULT_ENCODER_FFN_EMBED_DIM,
    heads: int = DEFAULT_ENCODER_ATTENTION_HEADS,
    num_atoms: int = DEFAULT_NUM_ATOMS,
    num_edges: int = DEFAULT_NUM_EDGES,
    num_in_degree: int = DEFAULT_NUM_IN_DEGREE,
    num_out_degree: int = DEFAULT_NUM_OUT_DEGREE,
    num_spatial: int = DEFAULT_NUM_SPATIAL,
    num_edge_dis: int = DEFAULT_NUM_EDGE_DIS,
) -> int:
    """Count parameters for the official Graphormer modules under scalar ZINC."""
    total = 0
    total += (num_atoms + 1) * embed_dim
    total += num_in_degree * embed_dim
    total += num_out_degree * embed_dim
    total += embed_dim  # graph token
    total += (num_edges + 1) * heads
    total += num_edge_dis * heads * heads
    total += num_spatial * heads
    total += heads  # graph-token virtual distance
    total += 2 * embed_dim  # embedding LayerNorm

    attention_params = 4 * (embed_dim * embed_dim + embed_dim)
    ffn_params = embed_dim * ffn_dim + ffn_dim + ffn_dim * embed_dim + embed_dim
    layer_norm_params = 4 * embed_dim
    total += layers * (attention_params + ffn_params + layer_norm_params)

    total += embed_dim * embed_dim + embed_dim  # masked_lm_pooler
    total += embed_dim * embed_dim + embed_dim  # lm_head_transform_weight
    total += 2 * embed_dim  # output LayerNorm
    total += embed_dim  # scalar regression output projection, no bias
    total += 1  # learned scalar output bias
    return total


class CommandError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def _format_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    printable = _format_cmd(cmd)
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


def pip_install(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return run_cmd([sys.executable, "-m", "pip", "install", *args], cwd=cwd, env=env, check=check)


class ConsoleFilter:
    """Compact notebook stdout while preserving the full raw fairseq log."""

    def __init__(self, *, verbosity: str, epoch_period: int) -> None:
        self.verbosity = verbosity
        self.epoch_period = max(1, int(epoch_period))
        self._in_traceback = False
        self._best_valid_loss = float("inf")
        self._best_epoch: int | None = None

    @staticmethod
    def _fmt(value, digits: int = 6) -> str:
        try:
            v = float(value)
        except Exception:
            return "nan"
        if v != v:
            return "nan"
        return f"{v:.{digits}f}"

    def _parse_valid_line(self, stripped: str) -> bool:
        if "valid on" not in stripped or " loss " not in f" {stripped} ":
            return False
        loss_match = re.search(r"(?:^|\s)loss\s+([-+0-9.eE]+)", stripped)
        if not loss_match:
            return False
        epoch_match = re.search(r"epoch\s+([0-9]+)", stripped)
        updates_match = re.search(r"num_updates\s+([0-9]+)", stripped)
        epoch = int(epoch_match.group(1)) if epoch_match else None
        if epoch is not None and epoch >= 2 and ((epoch + 1) % self.epoch_period != 0):
            return True
        loss = float(loss_match.group(1))
        if loss < self._best_valid_loss:
            self._best_valid_loss = loss
            self._best_epoch = epoch
        epoch_text = "?" if epoch is None else f"{epoch:04d}"
        updates_text = "?" if updates_match is None else updates_match.group(1)
        best_epoch = "?" if self._best_epoch is None else str(self._best_epoch)
        print(
            f"[valid epoch {epoch_text}] loss={self._fmt(loss)} | "
            f"best@{best_epoch} loss={self._fmt(self._best_valid_loss)} | "
            f"updates={updates_text}",
            flush=True,
        )
        return True

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
        error_tokens = (
            "ERROR", "Error:", "Exception", "RuntimeError", "TypeError",
            "AttributeError", "ImportError", "ModuleNotFoundError", "CUDA out of memory",
        )
        if any(tok in stripped for tok in error_tokens):
            return True

        if self._parse_valid_line(stripped):
            return False

        setup_fragments = (
            "num. shared model params:",
            "num. expert model params:",
            "task: GraphPredictionTask",
            "model: GraphormerModel",
            "criterion: GraphPredictionL1Loss",
            "training on ",
            "max tokens per device",
            "Loaded train with #samples",
            "Loaded valid with #samples",
            "Loaded test with #samples",
            "done training in",
            "saved checkpoint",
            "loaded checkpoint",
        )
        if any(fragment in stripped for fragment in setup_fragments):
            return True

        noisy_fragments = (
            "it/s]", "Downloading ", "Extracting ", "Processing...", "Done!",
            "UserWarning:", "The epoch iterator over",
        )
        if any(fragment in stripped for fragment in noisy_fragments):
            return False

        return self.verbosity == "standard"


def run_streaming_to_console_and_log(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_file: Path,
    env: Mapping[str, str] | None,
    console_verbosity: str,
    console_epoch_period: int,
) -> int:
    printable = _format_cmd(cmd)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log(f"\n[train-cmd] {printable}")
    log(f"[train-cwd] {cwd}")
    log(f"[train-log] {log_file}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env["PYTHONUNBUFFERED"] = "1"

    filt = ConsoleFilter(verbosity=console_verbosity, epoch_period=console_epoch_period)
    start = time.perf_counter()
    with log_file.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"Launched at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {printable}\n")
        f.write(f"CWD: {cwd}\n")
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
            msg = f"[console] {console_verbosity} stdout enabled; full raw Graphormer/fairseq output is saved to {log_file}\n"
            print(msg, end="", flush=True)
            f.write(msg)
        for line in proc.stdout:
            f.write(line)
            f.flush()
            if filt.should_print(line):
                print(line, end="", flush=True)
        rc = proc.wait()
        elapsed = time.perf_counter() - start
        f.write(f"\nReturn code: {rc}\nElapsed seconds: {elapsed:.2f}\n")
    return rc


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


def write_runtime_compat_shim(base_dir: Path) -> Path:
    shim_dir = base_dir / "runtime_compat"
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
    import numpy as _np
    for _name, _value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
    }.items():
        if not hasattr(_np, _name):
            setattr(_np, _name, _value)
except Exception:
    pass

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
""".lstrip(),
        encoding="utf-8",
    )
    log(f"[compat] Wrote runtime compatibility shim: {sitecustomize}")
    return shim_dir


def write_dgl_stub(shim_dir: Path) -> None:
    dgl_dir = shim_dir / "dgl"
    data_dir = dgl_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (dgl_dir / "__init__.py").write_text(
        "from .data import DGLDataset\n__all__ = ['DGLDataset']\n",
        encoding="utf-8",
    )
    (data_dir / "__init__.py").write_text(
        """
class DGLDataset:
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "DGL is not installed in this runtime. The official Graphormer ZINC "
            "configuration uses --dataset-source pyg, so DGL is only stubbed to "
            "satisfy an unused import."
        )
""".lstrip(),
        encoding="utf-8",
    )
    log(f"[compat] Wrote minimal DGL import stub under: {dgl_dir}")


def ensure_dgl_import_path(args: argparse.Namespace, shim_dir: Path) -> None:
    if args.dgl_stub == "never":
        return
    if args.dgl_stub == "always":
        write_dgl_stub(shim_dir)
        return
    try:
        import dgl  # type: ignore # noqa: F401
        log("[deps] DGL imports successfully; no DGL stub needed.")
    except Exception as exc:
        log(f"[compat] DGL import failed ({exc}); adding a PyG-ZINC-only DGL stub.")
        write_dgl_stub(shim_dir)


def env_with_runtime_compat(repo_dir: Path, shim_dir: Path | None, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    py_paths = []
    if shim_dir is not None:
        py_paths.append(str(shim_dir))
    py_paths.extend([str(repo_dir), str(repo_dir / "fairseq")])
    old = env.get("PYTHONPATH", "")
    if old:
        py_paths.append(old)
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    env["PYTHONUNBUFFERED"] = "1"
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def clone_or_update_repo(repo_dir: Path, repo_url: str, branch: str, commit: str | None, force_fresh: bool) -> str:
    if force_fresh and repo_dir.exists():
        log(f"[repo] Removing existing repo: {repo_dir}")
        shutil.rmtree(repo_dir)
    if not repo_dir.exists():
        run_cmd(["git", "clone", "--recursive", "--branch", branch, repo_url, str(repo_dir)])
    else:
        log(f"[repo] Existing repo found: {repo_dir}")
        run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
        run_cmd(["git", "checkout", branch], cwd=repo_dir)
        run_cmd(["git", "pull", "--ff-only", "origin", branch], cwd=repo_dir)
        run_cmd(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_dir)
    if commit:
        log(f"[repo] Pinning Microsoft Graphormer to commit: {commit}")
        run_cmd(["git", "checkout", commit], cwd=repo_dir)
        run_cmd(["git", "submodule", "update", "--init", "--recursive"], cwd=repo_dir)
    resolved_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    fairseq_commit = subprocess.check_output(["git", "-C", str(repo_dir / "fairseq"), "rev-parse", "HEAD"], text=True).strip()
    log(f"[repo] Using Graphormer commit: {resolved_commit}")
    log(f"[repo] Using fairseq submodule commit: {fairseq_commit}")
    return resolved_commit


def install_official_2021_dependencies(args: argparse.Namespace, repo_dir: Path, env: Mapping[str, str]) -> None:
    if sys.version_info >= (3, 10) and not args.allow_unsupported_python:
        raise RuntimeError(
            "The official-2021 dependency profile pins torch==1.9.1+cu111, which has no "
            f"wheels for Python {sys.version_info.major}.{sys.version_info.minor}. "
            "Use --dependency-profile current-colab, or pass --allow-unsupported-python "
            "if you are managing a compatible runtime yourself."
        )
    log("\n[deps] Installing exact dependency pins from Microsoft Graphormer's install.sh.")
    pip_install(["torch==1.9.1+cu111", "torchaudio", "-f", "https://download.pytorch.org/whl/cu111/torch_stable.html"], env=env)
    pip_install(["lmdb"], env=env)
    pip_install(["torch-scatter==2.0.9", "-f", "https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html"], env=env)
    pip_install(["torch-sparse==0.6.12", "-f", "https://pytorch-geometric.com/whl/torch-1.9.1+cu111.html"], env=env)
    pip_install(["torch-geometric==1.7.2"], env=env)
    pip_install(["tensorboardX==2.4.1"], env=env)
    pip_install(["ogb==1.3.2"], env=env)
    pip_install(["rdkit-pypi==2021.9.3"], env=env)
    pip_install(["dgl==0.7.2", "-f", "https://data.dgl.ai/wheels/repo.html"], env=env)

    fairseq_dir = repo_dir / "fairseq"
    proc = pip_install([".", "--use-feature=in-tree-build"], cwd=fairseq_dir, env=env, check=False)
    if proc.returncode != 0:
        log("[deps-warning] pip no longer accepted --use-feature=in-tree-build; retrying fairseq install without it.")
        pip_install(["."], cwd=fairseq_dir, env=env)
    run_cmd([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=fairseq_dir, env=env)


def install_current_colab_dependencies(args: argparse.Namespace, repo_dir: Path, env: Mapping[str, str]) -> None:
    log("\n[deps] Installing modern-Colab-compatible dependencies for the official Microsoft Graphormer code.")
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    pip_install(["setuptools<70", "cython<3", "numpy<2"], env=env)

    import importlib
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError("PyTorch is not installed/importable in this runtime. Select a Colab GPU runtime first.") from exc

    torch_version = str(torch.__version__).split("+")[0]
    cuda_version = getattr(torch.version, "cuda", None)
    cuda_tag = "cu" + cuda_version.replace(".", "") if cuda_version else "cpu"
    pyg_wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    log(f"[deps] Python: {sys.version.split()[0]} | torch: {torch.__version__} | CUDA: {cuda_version}")
    log(f"[deps] PyG wheel index: {pyg_wheel_url}")

    for package in ("pyg-lib", "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv"):
        proc = pip_install([package, "--only-binary=:all:", "-f", pyg_wheel_url], env=env, check=False)
        if proc.returncode != 0:
            log(f"[deps-warning] {package} wheel unavailable for this torch/CUDA pair; continuing.")

    pip_install([
        args.pyg_requirement,
        "lmdb",
        "tensorboardX>=2.6,<2.7",
        "ogb>=1.3.6",
        "rdkit",
        "scikit-learn>=1.3",
        "scipy>=1.9",
        "networkx>=2.8",
        "pandas",
        "tqdm",
        "regex",
        "bitarray",
        "sacrebleu>=1.4.12",
        "hydra-core>=1.3,<1.4",
        "omegaconf>=2.3,<2.4",
    ], env=env)

    if not args.skip_dgl_install:
        proc = pip_install(["dgl"], env=env, check=False)
        if proc.returncode != 0:
            log("[deps-warning] DGL failed to install; the PyG-ZINC-only DGL stub will be used if needed.")

    fairseq_dir = repo_dir / "fairseq"
    proc = pip_install(["-e", str(fairseq_dir), "--no-deps", "--no-build-isolation"], env=env, check=False)
    if proc.returncode != 0:
        log("[deps-warning] Editable fairseq install failed; retrying non-editable install.")
        pip_install([str(fairseq_dir), "--no-deps", "--no-build-isolation"], env=env)
    proc = run_cmd([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=fairseq_dir, env=env, check=False)
    if proc.returncode != 0:
        log("[deps-warning] fairseq build_ext failed. Training may still work if Python fallbacks cover the used code paths.")


def install_dependencies(args: argparse.Namespace, repo_dir: Path, env: Mapping[str, str]) -> None:
    if args.dependency_profile == "official-2021":
        install_official_2021_dependencies(args, repo_dir, env)
    else:
        install_current_colab_dependencies(args, repo_dir, env)


def _logical_shell_tokens(script_text: str) -> list[str]:
    lines = []
    for line in script_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(line.rstrip())
    logical = "\n".join(lines).replace("\\\n", " ")
    return shlex.split(logical, comments=True, posix=True)


def _normalise_option_name(token: str) -> str:
    if "=" in token:
        return token.split("=", 1)[0]
    return token


def _parse_options(tokens: Sequence[str]) -> tuple[dict[str, str], set[str], list[str], str | None]:
    env_assignments: list[str] = []
    exe: str | None = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if exe is None and "=" in token and not token.startswith("--"):
            env_assignments.append(token)
            i += 1
            continue
        if exe is None:
            exe = token
            i += 1
            break
        break

    options: dict[str, str] = {}
    flags: set[str] = set()
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            options[key] = value
            i += 1
            continue
        key = token
        if key in EXPECTED_ZINC_FLAGS:
            flags.add(key)
            i += 1
            continue
        if i + 1 >= len(tokens) or tokens[i + 1].startswith("--"):
            flags.add(key)
            i += 1
            continue
        options[key] = tokens[i + 1]
        i += 2
    return options, flags, env_assignments, exe


def validate_official_zinc_script(repo_dir: Path, allow_drift: bool) -> None:
    script_path = repo_dir / OFFICIAL_ZINC_SCRIPT
    if not script_path.exists():
        raise FileNotFoundError(f"Official Graphormer ZINC script not found: {script_path}")
    tokens = _logical_shell_tokens(script_path.read_text(encoding="utf-8"))
    options, flags, env_assignments, exe = _parse_options(tokens)

    errors: list[str] = []
    if exe != "fairseq-train":
        errors.append(f"launcher = {exe!r}; expected 'fairseq-train'")
    if "CUDA_VISIBLE_DEVICES=0" not in env_assignments:
        errors.append(f"CUDA env assignments = {env_assignments!r}; expected CUDA_VISIBLE_DEVICES=0")
    for key, expected in EXPECTED_ZINC_OPTIONS.items():
        actual = options.get(key)
        if actual != expected:
            errors.append(f"{key} = {actual!r}; expected {expected!r}")
    for flag in EXPECTED_ZINC_FLAGS:
        if flag not in flags:
            errors.append(f"missing flag {flag}")

    if errors:
        msg = "Official Microsoft Graphormer ZINC script does not match expected values:\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        if allow_drift:
            log("[config-warning] " + msg)
        else:
            raise RuntimeError(msg + "\nPass --allow-upstream-config-drift to run anyway.")
    log("[config] Official Microsoft Graphormer ZINC script validated.")
    log("[config] Key setup: PyG ZINC, graph_prediction, L1/MAE, graphormer_slim, 12 layers, width=80, FFN=80, heads=8, Adam, polynomial decay, batch_size=64, FP16, max_epoch=10000.")


def _replace_option(tokens: MutableSequence[str], option: str, value: str) -> None:
    prefix = option + "="
    for idx, token in enumerate(tokens):
        if token == option:
            if idx + 1 >= len(tokens):
                raise ValueError(f"Option {option} has no value slot")
            tokens[idx + 1] = value
            return
        if token.startswith(prefix):
            tokens[idx] = f"{option}={value}"
            return
    tokens.extend([option, value])


def _remove_flag(tokens: MutableSequence[str], flag: str) -> None:
    while flag in tokens:
        tokens.remove(flag)


def _remove_option(tokens: MutableSequence[str], option: str) -> None:
    idx = 0
    prefix = option + "="
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith(prefix):
            del tokens[idx]
            continue
        if token == option:
            del tokens[idx]
            if idx < len(tokens) and not tokens[idx].startswith("--"):
                del tokens[idx]
            continue
        idx += 1


def compute_updates_per_epoch(train_size: int, batch_size: int) -> int:
    return max(1, math.ceil(int(train_size) / int(batch_size)))


def build_training_command(args: argparse.Namespace, repo_dir: Path, drive_dir: Path) -> List[str]:
    save_dir = drive_dir / "ckpts"
    tensorboard_dir = drive_dir / "tensorboard"
    save_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    fairseq_args = list(OFFICIAL_ZINC_ARGS)
    _replace_option(fairseq_args, "--user-dir", str(repo_dir / "graphormer"))
    _replace_option(fairseq_args, "--num-workers", str(args.num_workers))
    _replace_option(fairseq_args, "--save-dir", str(save_dir))
    _replace_option(fairseq_args, "--seed", str(args.seed))
    _replace_option(fairseq_args, "--max-epoch", str(args.max_epoch))
    _replace_option(fairseq_args, "--batch-size", str(args.batch_size))
    _replace_option(fairseq_args, "--encoder-layers", str(args.encoder_layers))
    _replace_option(fairseq_args, "--encoder-embed-dim", str(args.encoder_embed_dim))
    _replace_option(fairseq_args, "--encoder-ffn-embed-dim", str(args.encoder_ffn_embed_dim))
    _replace_option(fairseq_args, "--encoder-attention-heads", str(args.encoder_attention_heads))
    _replace_option(fairseq_args, "--num-atoms", str(args.num_atoms))
    _replace_option(fairseq_args, "--num-edges", str(args.num_edges))
    _replace_option(fairseq_args, "--num-in-degree", str(args.num_in_degree))
    _replace_option(fairseq_args, "--num-out-degree", str(args.num_out_degree))
    _replace_option(fairseq_args, "--num-spatial", str(args.num_spatial))
    _replace_option(fairseq_args, "--num-edge-dis", str(args.num_edge_dis))
    _replace_option(fairseq_args, "--multi-hop-max-dist", str(args.multi_hop_max_dist))
    _replace_option(fairseq_args, "--spatial-pos-max", str(args.spatial_pos_max))

    updates_per_epoch = args.updates_per_epoch or compute_updates_per_epoch(args.train_size, args.batch_size)
    warmup_updates = args.warmup_updates or (args.warmup_epochs * updates_per_epoch)
    max_update = args.max_update or (args.max_epoch * updates_per_epoch)
    lr_period_updates = args.lr_period_updates or max(1, max_update - warmup_updates)

    _remove_option(fairseq_args, "--power")
    _remove_option(fairseq_args, "--total-num-update")
    _remove_option(fairseq_args, "--end-learning-rate")
    _replace_option(fairseq_args, "--optimizer", "adam")
    _replace_option(fairseq_args, "--lr-scheduler", "cosine")
    _replace_option(fairseq_args, "--lr", str(args.lr))
    _replace_option(fairseq_args, "--weight-decay", str(args.weight_decay))
    _replace_option(fairseq_args, "--clip-norm", str(args.clip_norm))
    _replace_option(fairseq_args, "--warmup-updates", str(warmup_updates))
    _replace_option(fairseq_args, "--warmup-init-lr", str(args.min_lr))
    _replace_option(fairseq_args, "--min-lr", str(args.min_lr))
    _replace_option(fairseq_args, "--max-update", str(max_update))
    _replace_option(fairseq_args, "--lr-period-updates", str(lr_period_updates))

    if args.no_fp16 or args.cpu:
        _remove_flag(fairseq_args, "--fp16")
    if args.cpu:
        fairseq_args.append("--cpu")

    if not args.auto_resume:
        _replace_option(fairseq_args, "--restore-file", "_codex_no_auto_resume_checkpoint.pt")

    if args.no_progress_bar:
        fairseq_args.append("--no-progress-bar")
    if args.log_format:
        _replace_option(fairseq_args, "--log-format", args.log_format)
    _replace_option(fairseq_args, "--log-interval", str(args.log_interval))
    _replace_option(fairseq_args, "--save-interval", str(args.save_interval))
    if args.keep_last_epochs >= 0:
        _replace_option(fairseq_args, "--keep-last-epochs", str(args.keep_last_epochs))
    if args.keep_best_checkpoints >= 0:
        _replace_option(fairseq_args, "--keep-best-checkpoints", str(args.keep_best_checkpoints))
    if args.tensorboard:
        _replace_option(fairseq_args, "--tensorboard-logdir", str(tensorboard_dir))
    if args.validate_test:
        _replace_option(fairseq_args, "--valid-subset", "valid,test")
    if args.no_save:
        fairseq_args.append("--no-save")
    if args.extra_fairseq_args:
        fairseq_args.extend(shlex.split(args.extra_fairseq_args))

    if args.launcher == "module":
        return [sys.executable, "-m", "fairseq_cli.train", *fairseq_args]
    if args.launcher == "script":
        return ["fairseq-train", *fairseq_args]
    if shutil.which("fairseq-train"):
        return ["fairseq-train", *fairseq_args]
    log("[launcher] fairseq-train is not on PATH; using python -m fairseq_cli.train.")
    return [sys.executable, "-m", "fairseq_cli.train", *fairseq_args]


def prepare_dataset_cache(train_cwd: Path, drive_dir: Path, relink: bool) -> Path:
    drive_dataset_dir = drive_dir / "datasets" / "pyg"
    drive_dataset_dir.mkdir(parents=True, exist_ok=True)
    local_dataset = train_cwd / "dataset"
    if local_dataset.is_symlink():
        current = os.readlink(local_dataset)
        if current == str(drive_dataset_dir):
            log(f"[data] Dataset symlink already points to Drive cache: {local_dataset}")
            return drive_dataset_dir
        if relink:
            local_dataset.unlink()
        else:
            log(f"[data-warning] Existing dataset symlink points to {current}; leaving it unchanged.")
            return local_dataset
    if local_dataset.exists():
        log(f"[data] Existing local dataset path found; leaving it unchanged: {local_dataset}")
        return local_dataset
    try:
        os.symlink(str(drive_dataset_dir), str(local_dataset), target_is_directory=True)
        log(f"[data] Linked official Graphormer relative dataset path to Drive cache: {local_dataset} -> {drive_dataset_dir}")
    except Exception as exc:
        local_dataset.mkdir(parents=True, exist_ok=True)
        log(f"[data-warning] Could not create dataset symlink ({exc}); using local cache: {local_dataset}")
        return local_dataset
    return drive_dataset_dir


def checkpoint_candidates(root: Path) -> list[Path]:
    patterns = ["*.pt", "*.pth", "*.ckpt"]
    out: list[Path] = []
    if root.exists():
        for pattern in patterns:
            out.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(out), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)


def checkpoint_epoch(path: Path) -> int | None:
    text = str(path).lower()
    for pattern in (r"checkpoint([0-9]+)\.pt$", r"epoch[=_-]?(\d+)", r"ep[=_-]?(\d+)"):
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def parse_training_summary_from_logs(log_paths: Sequence[Path]) -> dict:
    valid_history: list[dict] = []
    param_count = None
    for path in log_paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            param_match = re.search(r"num\. shared model params:\s*([0-9,]+)", line)
            if param_match:
                param_count = int(param_match.group(1).replace(",", ""))
            if "valid on" not in line or " loss " not in f" {line} ":
                continue
            loss_match = re.search(r"(?:^|\s)loss\s+([-+0-9.eE]+)", line)
            if not loss_match:
                continue
            epoch_match = re.search(r"epoch\s+([0-9]+)", line)
            updates_match = re.search(r"num_updates\s+([0-9]+)", line)
            valid_history.append({
                "epoch": int(epoch_match.group(1)) if epoch_match else None,
                "num_updates": int(updates_match.group(1)) if updates_match else None,
                "loss": float(loss_match.group(1)),
                "line": line.strip(),
                "log_file": str(path),
            })
    best = min(valid_history, key=lambda row: row["loss"]) if valid_history else None
    return {
        "history_epochs": len(valid_history),
        "best_valid": best,
        "param_count": param_count,
        "log_files": [str(path) for path in log_paths],
    }


def write_run_manifest(
    args: argparse.Namespace,
    drive_dir: Path,
    repo_commit: str,
    wrapper_log: Path | None,
    command: Sequence[str],
    train_cwd: Path,
    no_train: bool,
) -> Path:
    checkpoint_root = drive_dir / "ckpts"
    updates_per_epoch = args.updates_per_epoch or compute_updates_per_epoch(args.train_size, args.batch_size)
    warmup_updates = args.warmup_updates or (args.warmup_epochs * updates_per_epoch)
    max_update = args.max_update or (args.max_epoch * updates_per_epoch)
    lr_period_updates = args.lr_period_updates or max(1, max_update - warmup_updates)
    estimated_params = estimate_graphormer_params(
        layers=args.encoder_layers,
        embed_dim=args.encoder_embed_dim,
        ffn_dim=args.encoder_ffn_embed_dim,
        heads=args.encoder_attention_heads,
        num_atoms=args.num_atoms,
        num_edges=args.num_edges,
        num_in_degree=args.num_in_degree,
        num_out_degree=args.num_out_degree,
        num_spatial=args.num_spatial,
        num_edge_dis=args.num_edge_dis,
    )
    log_paths = []
    if wrapper_log is not None and wrapper_log.exists():
        log_paths.append(wrapper_log)
    wrapper_dir = drive_dir / "wrapper_logs"
    if wrapper_dir.exists():
        log_paths.extend(path for path in wrapper_dir.rglob("*.log") if path.is_file())
    log_paths = sorted(set(log_paths), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    summary = parse_training_summary_from_logs(log_paths)
    checkpoints = checkpoint_candidates(checkpoint_root)
    manifest = {
        "runner": "Graphormer_ZINC_Official_Colab.py",
        "official_repo": OFFICIAL_REPO,
        "official_commit": repo_commit,
        "official_fairseq_commit_expected": OFFICIAL_FAIRSEQ_COMMIT,
        "official_config_script": str(OFFICIAL_ZINC_SCRIPT),
        "paper": "arXiv:2203.04810",
        "comparison_training_reference": "GRIT, arXiv:2305.17589",
        "model": "Microsoft Graphormer-Slim",
        "dataset": "Official Microsoft Graphormer PyG ZINC configuration with GRIT-aligned ZINC-small training defaults",
        "derived_config": {
            "encoder_layers": args.encoder_layers,
            "encoder_embed_dim": args.encoder_embed_dim,
            "encoder_ffn_embed_dim": args.encoder_ffn_embed_dim,
            "encoder_attention_heads": args.encoder_attention_heads,
            "num_atoms": args.num_atoms,
            "num_edges": args.num_edges,
            "num_in_degree": args.num_in_degree,
            "num_out_degree": args.num_out_degree,
            "num_spatial": args.num_spatial,
            "num_edge_dis": args.num_edge_dis,
            "multi_hop_max_dist": args.multi_hop_max_dist,
            "spatial_pos_max": args.spatial_pos_max,
            "estimated_param_count": estimated_params,
            "target_param_count_window": [470000, 480000],
            "train_size_for_schedule": args.train_size,
            "batch_size": args.batch_size,
            "max_epoch": args.max_epoch,
            "updates_per_epoch": updates_per_epoch,
            "warmup_epochs": args.warmup_epochs,
            "warmup_updates": warmup_updates,
            "max_update": max_update,
            "lr_period_updates": lr_period_updates,
            "optimizer": "fairseq adam (AdamW-style decoupled weight decay)",
            "lr_scheduler": "cosine",
            "lr": args.lr,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "clip_norm": args.clip_norm,
        },
        "dependency_profile": args.dependency_profile,
        "seed": args.seed,
        "name_tag": args.name_tag,
        "drive_dir": str(drive_dir),
        "train_cwd": str(train_cwd),
        "command": list(map(str, command)),
        "command_string": _format_cmd(command),
        "no_train": no_train,
        "param_count_from_log": summary.get("param_count"),
        "best_valid": summary.get("best_valid"),
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
    suffix = "notrain" if no_train else f"seed{args.seed}"
    manifest_path = drive_dir / f"graphormer_zinc_official_manifest_{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    latest = drive_dir / "latest_graphormer_zinc_official_manifest.json"
    latest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log(f"[manifest] wrote: {manifest_path}")
    return manifest_path


def smoke_test_official_imports(repo_dir: Path, env: Mapping[str, str]) -> None:
    code = r"""
import graphormer
import graphormer.models.graphormer
import graphormer.tasks.graph_prediction
from graphormer.data.wrapper import preprocess_item
from fairseq import options
print("official Graphormer + fairseq imports ok")
"""
    run_cmd([sys.executable, "-c", code], cwd=repo_dir, env=env)


def print_environment_summary(drive_dir: Path, repo_dir: Path, commit: str, train_cwd: Path) -> None:
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
    log(f"  train cwd:{train_cwd}")
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
        description="Train official Microsoft Graphormer-Slim on ZINC in Colab.",
        epilog=textwrap.dedent(
            """
            Examples:
              main([])
              main(["--seed", "41", "--name-tag", "ColabDrive.Graphormer.ZINC.GritSched.s41"])
              main(["--no-train"])  # install/validate/smoke-test only
              main(["--dependency-profile", "official-2021"])
            """
        ),
    )
    parser.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    parser.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/graphormer_zinc_official"))
    parser.add_argument("--repo-dir", type=Path, default=Path("/content/Graphormer"))
    parser.add_argument("--repo-url", default=OFFICIAL_REPO)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--commit", default=OFFICIAL_COMMIT, help="Pin official Microsoft Graphormer. Pass empty string to use branch HEAD.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name-tag", default="ColabDrive.Graphormer.ZINC.GritSched")
    parser.add_argument("--dependency-profile", choices=["current-colab", "official-2021"], default="current-colab")
    parser.add_argument("--pyg-requirement", default="torch-geometric>=2.4,<3")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-dgl-install", action="store_true", default=True)
    parser.add_argument("--install-dgl", action="store_false", dest="skip_dgl_install")
    parser.add_argument("--dgl-stub", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--force-fresh-repo", action="store_true")
    parser.add_argument("--allow-upstream-config-drift", action="store_true")
    parser.add_argument("--allow-unsupported-python", action="store_true")
    parser.add_argument("--launcher", choices=["auto", "script", "module"], default="auto")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-epoch", type=int, default=DEFAULT_MAX_EPOCH)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--updates-per-epoch", type=int, default=0, help="Override inferred ceil(train_size / batch_size).")
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_WARMUP_EPOCHS)
    parser.add_argument("--warmup-updates", type=int, default=0, help="Override inferred warmup_epochs * updates_per_epoch.")
    parser.add_argument("--max-update", type=int, default=0, help="Override inferred max_epoch * updates_per_epoch for cosine scheduling.")
    parser.add_argument("--lr-period-updates", type=int, default=0, help="Override inferred max_update - warmup_updates.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--min-lr", type=float, default=DEFAULT_MIN_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--clip-norm", type=float, default=DEFAULT_CLIP_NORM)
    parser.add_argument("--encoder-layers", type=int, default=DEFAULT_ENCODER_LAYERS)
    parser.add_argument("--encoder-embed-dim", type=int, default=DEFAULT_ENCODER_EMBED_DIM)
    parser.add_argument("--encoder-ffn-embed-dim", type=int, default=DEFAULT_ENCODER_FFN_EMBED_DIM)
    parser.add_argument("--encoder-attention-heads", type=int, default=DEFAULT_ENCODER_ATTENTION_HEADS)
    parser.add_argument("--num-atoms", type=int, default=DEFAULT_NUM_ATOMS)
    parser.add_argument("--num-edges", type=int, default=DEFAULT_NUM_EDGES)
    parser.add_argument("--num-in-degree", type=int, default=DEFAULT_NUM_IN_DEGREE)
    parser.add_argument("--num-out-degree", type=int, default=DEFAULT_NUM_OUT_DEGREE)
    parser.add_argument("--num-spatial", type=int, default=DEFAULT_NUM_SPATIAL)
    parser.add_argument("--num-edge-dis", type=int, default=DEFAULT_NUM_EDGE_DIS)
    parser.add_argument("--multi-hop-max-dist", type=int, default=DEFAULT_MULTI_HOP_MAX_DIST)
    parser.add_argument("--spatial-pos-max", type=int, default=DEFAULT_SPATIAL_POS_MAX)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--keep-last-epochs", type=int, default=5)
    parser.add_argument("--keep-best-checkpoints", type=int, default=5)
    parser.add_argument("--official-checkpoint-policy", action="store_true")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--validate-test", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--auto-resume", action="store_true", default=True)
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")
    parser.add_argument("--console-verbosity", choices=["compact", "standard", "full"], default="compact")
    parser.add_argument("--console-epoch-period", type=int, default=1)
    parser.add_argument("--no-progress-bar", action="store_true", default=True)
    parser.add_argument("--progress-bar", action="store_false", dest="no_progress_bar")
    parser.add_argument("--log-format", default="simple")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--extra-fairseq-args", default="", help="Extra fairseq args as one shell-quoted string.")
    parser.add_argument("--relink-dataset-cache", action="store_true")
    parser.add_argument("--no-train", action="store_true", help="Stop after clone/install/validate/import smoke test.")
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _strip_colab_kernel_args(argv)
    args = parser.parse_args(argv)
    if args.official_checkpoint_policy:
        args.keep_last_epochs = -1
        args.keep_best_checkpoints = -1
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    mount_drive(args.drive_mount)
    args.drive_dir.mkdir(parents=True, exist_ok=True)

    commit = clone_or_update_repo(args.repo_dir, args.repo_url, args.branch, args.commit or None, args.force_fresh_repo)
    shim_dir = write_runtime_compat_shim(args.drive_dir)
    env = env_with_runtime_compat(args.repo_dir, shim_dir, args)

    if not args.skip_install:
        install_dependencies(args, args.repo_dir, env)
    else:
        log("[deps] Skipping dependency installation (--skip-install).")

    ensure_dgl_import_path(args, shim_dir)
    env = env_with_runtime_compat(args.repo_dir, shim_dir, args)

    validate_official_zinc_script(args.repo_dir, args.allow_upstream_config_drift)
    smoke_test_official_imports(args.repo_dir, env)

    train_cwd = args.repo_dir / OFFICIAL_TRAIN_CWD
    prepare_dataset_cache(train_cwd, args.drive_dir, args.relink_dataset_cache)
    print_environment_summary(args.drive_dir, args.repo_dir, commit, train_cwd)

    command = build_training_command(args, args.repo_dir, args.drive_dir)
    updates_per_epoch = args.updates_per_epoch or compute_updates_per_epoch(args.train_size, args.batch_size)
    warmup_updates = args.warmup_updates or (args.warmup_epochs * updates_per_epoch)
    max_update = args.max_update or (args.max_epoch * updates_per_epoch)
    estimated_params = estimate_graphormer_params(
        layers=args.encoder_layers,
        embed_dim=args.encoder_embed_dim,
        ffn_dim=args.encoder_ffn_embed_dim,
        heads=args.encoder_attention_heads,
        num_atoms=args.num_atoms,
        num_edges=args.num_edges,
        num_in_degree=args.num_in_degree,
        num_out_degree=args.num_out_degree,
        num_spatial=args.num_spatial,
        num_edge_dis=args.num_edge_dis,
    )
    log(
        "\n[derived-config] "
        f"Graphormer layers={args.encoder_layers}, d={args.encoder_embed_dim}, "
        f"ffn={args.encoder_ffn_embed_dim}, heads={args.encoder_attention_heads}, "
        f"estimated_params={estimated_params:,}"
    )
    log(
        "[derived-config] "
        f"batch={args.batch_size}, epochs={args.max_epoch}, lr={args.lr}, "
        f"AdamW-style wd={args.weight_decay}, cosine warmup_epochs={args.warmup_epochs} "
        f"({warmup_updates} updates), max_update={max_update}, min_lr={args.min_lr}"
    )
    log("\n[resolved-command]")
    log(_format_cmd(command))

    if args.no_train:
        write_run_manifest(args, args.drive_dir, commit, None, command, train_cwd, no_train=True)
        log("\n[done] --no-train completed after official config validation and import smoke test.")
        return

    wrapper_log = args.drive_dir / "wrapper_logs" / f"graphormer_zinc_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    rc = run_streaming_to_console_and_log(
        command,
        cwd=train_cwd,
        log_file=wrapper_log,
        env=env,
        console_verbosity=args.console_verbosity,
        console_epoch_period=args.console_epoch_period,
    )
    if rc != 0:
        raise SystemExit(rc)

    write_run_manifest(args, args.drive_dir, commit, wrapper_log, command, train_cwd, no_train=False)
    log("\n[done] Official Microsoft Graphormer ZINC training process completed successfully.")
    log(f"[done] Results/checkpoints root: {args.drive_dir / 'ckpts'}")
    log(f"[done] Wrapper log: {wrapper_log}")


if __name__ == "__main__":
    cli_argv = _strip_colab_kernel_args(sys.argv[1:])
    if cli_argv:
        main()
    else:
        main([
            "--seed", "1",
            "--name-tag", "ColabDrive.Graphormer.ZINC.GritSched.L10F116.s1",
            "--force-fresh-repo",
            "--console-verbosity", "compact",
            "--console-epoch-period", "1",
        ])
