#!/usr/bin/env python3
"""Colab runner for official Benchmarking-GNNs GIN on ZINC.

This script intentionally does not reimplement GIN training. It clones the
official Benchmarking-GNNs repository and launches the official ZINC GIN
configuration:

    python -u main_molecules_graph_regression.py \
        --dataset ZINC \
        --gpu_id 0 \
        --config configs/molecules_graph_regression_GIN_ZINC_500k.json

The default config is the official 500k-parameter ZINC config from the
Benchmarking-GNNs paper/repository. The official config is validated before
launch, then the runtime copy defaults to 2000 epochs to match the GRIT/1-hop
GRIT ZINC training schedule used in this project. The script otherwise only
changes Colab/runtime behavior: dependency setup, mounted Drive paths, clean
epoch prints, and extra best/latest checkpoint copies. The official rolling
epoch checkpoints are still written by the official training script.

Suggested Colab usage:

    # Upload this file to Colab, then run:
    from gin_zinc_core import main
    main([])

Default Drive outputs:

    /content/drive/MyDrive/gin_zinc_official

Notes:
- The official repo targets an older DGL/PyTorch environment. Current Colab
  images use newer Python/PyTorch, so the runner installs a compatible DGL GPU
  stack by default before launching the official code in a subprocess.
- The official config's ``ZINC_500k`` suffix denotes the model-size budget, not
  ZINC-full. The config's dataset field is ``ZINC``, which the official repo
  loads as the 10k/1k/1k benchmark subset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


OFFICIAL_REPO = "https://github.com/graphdeeplearning/benchmarking-gnns.git"
OFFICIAL_BRANCH = "master"
OFFICIAL_COMMIT = "b6c407712fa576e9699555e1e035d1e327ccae6c"
OFFICIAL_CONFIG = "configs/molecules_graph_regression_GIN_ZINC_500k.json"
OFFICIAL_ZINC_URL = "https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl"
EXPECTED_OFFICIAL_PARAMS = 509_549
DEFAULT_RUNTIME_EPOCHS = 2_000
DEFAULT_RUNTIME_MAX_TIME = 0.0
DEFAULT_RUNTIME_LR_SCHEDULE = "cosine_warmup"
DEFAULT_RUNTIME_WARMUP_EPOCHS = 50
DEFAULT_RUNTIME_COSINE_MIN_LR = 1.0e-6

EXPECTED_CONFIG_VALUES: dict[tuple[str, ...], Any] = {
    ("model",): "GIN",
    ("dataset",): "ZINC",
    ("params", "seed"): 41,
    ("params", "epochs"): 1000,
    ("params", "batch_size"): 128,
    ("params", "init_lr"): 0.001,
    ("params", "lr_reduce_factor"): 0.5,
    ("params", "lr_schedule_patience"): 10,
    ("params", "min_lr"): 1e-5,
    ("params", "weight_decay"): 0.0,
    ("params", "print_epoch_interval"): 5,
    ("params", "max_time"): 12,
    ("net_params", "L"): 16,
    ("net_params", "hidden_dim"): 124,
    ("net_params", "residual"): True,
    ("net_params", "readout"): "sum",
    ("net_params", "n_mlp_GIN"): 2,
    ("net_params", "learn_eps_GIN"): True,
    ("net_params", "neighbor_aggr_GIN"): "sum",
    ("net_params", "in_feat_dropout"): 0.0,
    ("net_params", "dropout"): 0.0,
    ("net_params", "batch_norm"): True,
}


class CommandError(RuntimeError):
    """Raised when a subprocess command fails."""


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    safe_display: str | None = None,
) -> subprocess.CompletedProcess[str]:
    display = safe_display or " ".join(cmd)
    print(f"[cmd] {display}", flush=True)
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if check and proc.returncode != 0:
        raise CommandError(f"command failed with exit code {proc.returncode}: {display}")
    return proc


def run_streaming_to_log(cmd: Sequence[str], *, cwd: Path, log_path: Path, env: Mapping[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    print(f"[log] full raw training log: {log_path}", flush=True)
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    interesting = (
        "[epoch]",
        "[best]",
        "cuda available",
        "cuda not available",
        "MODEL/Total parameters",
        "Training Graphs:",
        "Validation Graphs:",
        "Test Graphs:",
        "Test MAE:",
        "Train MAE:",
        "Convergence Time",
        "TOTAL TIME TAKEN",
        "AVG TIME PER EPOCH",
        "Max_time for training elapsed",
        "LR EQUAL TO MIN LR",
    )
    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n===== command started {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        log_f.write(" ".join(cmd) + "\n")
        for line in proc.stdout:
            log_f.write(line)
            if any(token in line for token in interesting):
                print(line.rstrip(), flush=True)
        code = proc.wait()
        log_f.write(f"===== command exited {code} {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    if code != 0:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-120:])
            print("[error] training failed; last 120 raw log lines follow:", flush=True)
            print(tail, flush=True)
        except Exception as exc:
            print(f"[error] training failed and log tail could not be read: {exc}", flush=True)
        raise CommandError(f"training failed with exit code {code}; see {log_path}")


def running_in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


def path_is_mountpoint(path: Path) -> bool:
    try:
        if os.path.ismount(path):
            return True
    except Exception:
        pass
    try:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return False
    target = str(path)
    return any(len(line.split()) >= 2 and line.split()[1] == target for line in mounts)


def assert_drive_writable(drive_root: Path) -> None:
    if not running_in_colab():
        return
    drive_mount = Path("/content/drive")
    try:
        drive_root.resolve().relative_to(drive_mount.resolve())
    except Exception:
        return
    if not path_is_mountpoint(drive_mount):
        raise RuntimeError(
            "/content/drive is not a mounted Google Drive filesystem. Refusing to train because outputs "
            "would be written to ephemeral Colab disk. Run drive.mount('/content/drive', force_remount=True) "
            "or rerun this script from a fresh runtime."
        )
    my_drive = drive_mount / "MyDrive"
    if not my_drive.exists():
        raise RuntimeError("/content/drive is mounted, but /content/drive/MyDrive is missing.")
    drive_root.mkdir(parents=True, exist_ok=True)
    probe = drive_root / ".drive_write_probe"
    token = f"gin-zinc-drive-probe-{time.time()}"
    probe.write_text(token, encoding="utf-8")
    observed = probe.read_text(encoding="utf-8")
    if observed != token:
        raise RuntimeError(f"Google Drive write probe failed at {probe}")
    probe.unlink(missing_ok=True)
    print(f"[drive-check] verified mounted, writable Drive root: {drive_root}", flush=True)


def stash_unmounted_drive_root(drive_root: Path) -> Path | None:
    if not running_in_colab():
        return None
    drive_mount = Path("/content/drive")
    try:
        drive_root.resolve().relative_to(drive_mount.resolve())
    except Exception:
        return None
    if path_is_mountpoint(drive_mount) or not drive_root.exists():
        return None
    try:
        has_contents = any(drive_root.iterdir())
    except Exception:
        has_contents = False
    if not has_contents:
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stash_root = Path("/content/gin_zinc_unmounted_drive_recovery")
    stash_root.mkdir(parents=True, exist_ok=True)
    stash = stash_root / f"{drive_root.name}_{stamp}"
    print(
        f"[drive-recovery] Found outputs under unmounted fake Drive path {drive_root}. "
        f"Moving them aside before mounting real Drive: {stash}",
        flush=True,
    )
    shutil.move(str(drive_root), str(stash))
    return stash


def restore_shadow_to_drive(stash: Path | None, drive_root: Path) -> None:
    if stash is None:
        return
    if not stash.exists():
        print(f"[drive-recovery-warning] Recovery stash disappeared: {stash}", flush=True)
        return
    print(f"[drive-recovery] Copying recovered local outputs into real Drive: {drive_root}", flush=True)
    shutil.copytree(stash, drive_root, dirs_exist_ok=True)
    marker = drive_root / "RECOVERED_FROM_UNMOUNTED_COLAB_DRIVE.txt"
    marker.write_text(
        "This directory was recovered from a local /content/drive shadow path created before Google Drive was mounted.\n"
        f"Recovery stash: {stash}\n"
        f"Recovered at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )
    print(f"[drive-recovery] Recovery complete. Marker: {marker}", flush=True)


def mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        print("[drive] google.colab is unavailable; assuming Drive is already mounted or not needed.", flush=True)
        return
    mountpoint = Path("/content/drive")
    print("[drive] Mounting Google Drive at /content/drive ...", flush=True)
    try:
        drive.mount("/content/drive", force_remount=False)
    except ValueError as exc:
        if "Mountpoint must not already contain files" not in str(exc):
            raise
        print("[drive-warning] /content/drive is non-empty before mount; retrying with force_remount=True.", flush=True)
        drive.mount("/content/drive", force_remount=True)


def install_dependencies(mode: str) -> None:
    if mode == "none":
        print("[deps] Skipping dependency installation.", flush=True)
        return
    if "torch" in sys.modules:
        print(
            "[deps-warning] torch is already imported in this kernel. If dependency installation changes torch, "
            "restart the runtime and run this file first.",
            flush=True,
        )

    run_cmd([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"])
    if mode == "compatible":
        print("[deps] Installing DGL/PyTorch stack selected for current Colab compatibility.", flush=True)
        run_cmd(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torch==2.4.0",
                "torchvision==0.19.0",
                "torchaudio==2.4.0",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ]
        )
        run_cmd(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torchdata==0.8.0",
                "dgl==2.4.0+cu121",
                "-f",
                "https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html",
            ]
        )
    elif mode == "current":
        print("[deps] Keeping current torch; installing latest available DGL stack.", flush=True)
        run_cmd(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torchdata",
                "dgl",
            ]
        )
    else:
        raise ValueError(f"unknown dependency install mode: {mode}")

    run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "tensorboardX",
            "tqdm",
            "numpy",
            "scipy",
            "networkx",
            "pandas",
            "scikit-learn",
            "ogb",
        ]
    )
    run_cmd(
        [
            sys.executable,
            "-c",
            "import torch, dgl, tensorboardX; "
            "print('[deps] torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available()); "
            "print('[deps] dgl', dgl.__version__)",
        ]
    )


def clone_or_update_repo(repo_dir: Path, *, force: bool, pin_commit: bool) -> str:
    if force and repo_dir.exists():
        print(f"[repo] Removing existing repo: {repo_dir}", flush=True)
        shutil.rmtree(repo_dir)
    if not repo_dir.exists():
        run_cmd(
            [
                "git",
                "clone",
                "--branch",
                OFFICIAL_BRANCH,
                OFFICIAL_REPO,
                str(repo_dir),
            ],
            safe_display=f"git clone --branch {OFFICIAL_BRANCH} {OFFICIAL_REPO} {repo_dir}",
        )
    else:
        run_cmd(["git", "-C", str(repo_dir), "fetch", "origin", OFFICIAL_BRANCH])
    if pin_commit:
        run_cmd(["git", "-C", str(repo_dir), "checkout", OFFICIAL_COMMIT])
    else:
        run_cmd(["git", "-C", str(repo_dir), "checkout", OFFICIAL_BRANCH])
        run_cmd(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", OFFICIAL_BRANCH])
    sha = run_cmd(["git", "-C", str(repo_dir), "rev-parse", "HEAD"]).stdout.strip()
    print(f"[repo] official Benchmarking-GNNs commit: {sha}", flush=True)
    return sha


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"expected JSON object at {path}")
    return data


def nested_get(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            raise KeyError(".".join(path))
        cur = cur[key]
    return cur


def validate_official_config(cfg_path: Path, *, allow_drift: bool) -> None:
    cfg = load_json(cfg_path)
    mismatches: list[str] = []
    for path, expected in EXPECTED_CONFIG_VALUES.items():
        try:
            observed = nested_get(cfg, path)
        except KeyError:
            mismatches.append(f"{'.'.join(path)} missing; expected {expected!r}")
            continue
        if observed != expected:
            mismatches.append(f"{'.'.join(path)} = {observed!r}; expected {expected!r}")
    if mismatches:
        msg = "Official GIN ZINC config does not match expected paper/repo config:\n" + "\n".join(
            f"  - {m}" for m in mismatches
        )
        if allow_drift:
            print("[config-warning] " + msg, flush=True)
        else:
            raise RuntimeError(msg)
    print("[config] Official Benchmarking-GNNs GIN ZINC 500k config validated.", flush=True)
    print(
        "[config] Key setup: dataset=ZINC subset, model=GIN, layers=16, hidden_dim=124, "
        "sum readout, train_eps=True, batch=128, L1/MAE, 1000 epochs, lr=0.001.",
        flush=True,
    )


def ensure_zinc_data(repo_dir: Path, drive_root: Path, *, force: bool) -> Path:
    cache_dir = drive_root / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / "ZINC.pkl"
    if force and cached.exists():
        cached.unlink()
    if not cached.exists():
        tmp = cached.with_suffix(".pkl.tmp")
        run_cmd(["curl", "-L", "--fail", "--retry", "3", OFFICIAL_ZINC_URL, "-o", str(tmp)])
        tmp.replace(cached)
    target_dir = repo_dir / "data" / "molecules"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "ZINC.pkl"
    if force or not target.exists() or target.stat().st_size != cached.stat().st_size:
        shutil.copy2(cached, target)
    print(f"[data] ZINC cache: {cached}", flush=True)
    print(f"[data] ZINC repo copy: {target}", flush=True)
    return target


def patch_official_training_script(repo_dir: Path) -> None:
    path = repo_dir / "main_molecules_graph_regression.py"
    text = path.read_text(encoding="utf-8")
    dgl_marker = "import dgl\n"
    dgl_patch = """import dgl

# COLAB_GIN_ZINC_DGL_COMPAT_START: DGL renamed these helpers after the
# official Benchmarking-GNNs release. Alias only missing names so official model
# code can import unchanged under modern Colab DGL.
import dgl.function as _colab_dgl_fn
if not hasattr(_colab_dgl_fn, "copy_src") and hasattr(_colab_dgl_fn, "copy_u"):
    def _colab_copy_src(src=None, out=None, *args, **kwargs):
        if args:
            src = args[0]
            if len(args) > 1:
                out = args[1]
        src = kwargs.get("src", src)
        out = kwargs.get("out", out)
        return _colab_dgl_fn.copy_u(src, out)
    _colab_dgl_fn.copy_src = _colab_copy_src
if not hasattr(_colab_dgl_fn, "copy_edge") and hasattr(_colab_dgl_fn, "copy_e"):
    def _colab_copy_edge(edge=None, out=None, *args, **kwargs):
        if args:
            edge = args[0]
            if len(args) > 1:
                out = args[1]
        edge = kwargs.get("edge", edge)
        out = kwargs.get("out", out)
        return _colab_dgl_fn.copy_e(edge, out)
    _colab_dgl_fn.copy_edge = _colab_copy_edge
if not hasattr(_colab_dgl_fn, "copy_dst") and hasattr(_colab_dgl_fn, "copy_v"):
    def _colab_copy_dst(dst=None, out=None, *args, **kwargs):
        if args:
            dst = args[0]
            if len(args) > 1:
                out = args[1]
        dst = kwargs.get("dst", dst)
        out = kwargs.get("out", out)
        return _colab_dgl_fn.copy_v(dst, out)
    _colab_dgl_fn.copy_dst = _colab_copy_dst
import importlib as _colab_importlib
import sys as _colab_sys
_colab_dgl_heterograph = _colab_importlib.import_module("dgl.heterograph")
if not hasattr(_colab_dgl_heterograph, "DGLHeteroGraph"):
    if hasattr(_colab_dgl_heterograph, "DGLGraph"):
        _colab_dgl_heterograph.DGLHeteroGraph = _colab_dgl_heterograph.DGLGraph
    else:
        import torch as _colab_torch
        _empty = _colab_torch.tensor([], dtype=_colab_torch.int64)
        _colab_dgl_heterograph.DGLHeteroGraph = type(dgl.graph((_empty, _empty)))
_colab_sys.modules["dgl.heterograph"].DGLHeteroGraph = _colab_dgl_heterograph.DGLHeteroGraph
# COLAB_GIN_ZINC_DGL_COMPAT_END
"""
    compat_pattern = re.compile(
        r"import dgl\n\n# COLAB_GIN_ZINC_DGL_COMPAT_START:.*?# COLAB_GIN_ZINC_DGL_COMPAT_END\n",
        re.DOTALL,
    )
    if "COLAB_GIN_ZINC_DGL_COMPAT_START" in text:
        text, replaced = compat_pattern.subn(dgl_patch, text, count=1)
        if replaced != 1:
            raise RuntimeError("could not replace existing DGL compatibility patch")
    else:
        if dgl_marker not in text:
            raise RuntimeError("could not patch official training script: import dgl marker not found")
        text = text.replace(dgl_marker, dgl_patch, 1)

    if "COLAB_GIN_ZINC_PATCH_START" in text:
        text = patch_official_lr_schedule_text(text)
        text = patch_official_no_early_stop_text(text)
        text = patch_official_checkpoint_cleanup_text(text)
        path.write_text(text, encoding="utf-8")
        print("[patch] Colab compatibility/checkpoint/log patches already present.", flush=True)
        patch_official_molecule_loader(repo_dir)
        return

    marker = "    epoch_train_MAEs, epoch_val_MAEs = [], [] \n"
    inject = marker + """\

    # COLAB_GIN_ZINC_PATCH_START: output-only helpers; model/training math is unchanged.
    best_val_mae = float("inf")
    best_epoch = -1
    history_csv_path = write_file_name + "_history.csv"
    best_summary_path = write_file_name + "_best.json"
    with open(history_csv_path, "w") as hist_f:
        hist_f.write("epoch,train_loss,val_loss,train_mae,val_mae,test_mae,lr,seconds\\n")
    # COLAB_GIN_ZINC_PATCH_END
"""
    if marker not in text:
        raise RuntimeError("could not patch official training script: metrics marker not found")
    text = text.replace(marker, inject, 1)

    marker = "                scheduler.step(epoch_val_loss)\n"
    inject = """\
                # COLAB_GIN_ZINC_PATCH_START: extra Drive-friendly logs/checkpoints.
                latest_state_path = os.path.join(ckpt_dir, "latest_state_dict.pkl")
                torch.save(model.state_dict(), latest_state_path)
                with open(history_csv_path, "a") as hist_f:
                    hist_f.write("{},{:.10g},{:.10g},{:.10g},{:.10g},{:.10g},{:.10g},{:.10g}\\n".format(
                        epoch,
                        epoch_train_loss,
                        epoch_val_loss,
                        epoch_train_mae,
                        epoch_val_mae,
                        epoch_test_mae,
                        optimizer.param_groups[0]['lr'],
                        per_epoch_time[-1],
                    ))
                if epoch_val_mae < best_val_mae:
                    best_val_mae = float(epoch_val_mae)
                    best_epoch = int(epoch)
                    best_state_path = os.path.join(ckpt_dir, "best_val_mae_state_dict.pkl")
                    torch.save(model.state_dict(), best_state_path)
                    with open(best_summary_path, "w") as best_f:
                        json.dump({
                            "best_epoch": best_epoch,
                            "best_val_mae": best_val_mae,
                            "test_mae_at_best_val_epoch": float(epoch_test_mae),
                            "train_mae_at_best_val_epoch": float(epoch_train_mae),
                            "checkpoint": best_state_path,
                        }, best_f, indent=2)
                    print("[best] epoch={} val_mae={:.6f} test_mae={:.6f} checkpoint={}".format(
                        best_epoch, best_val_mae, epoch_test_mae, best_state_path
                    ), flush=True)
                interval = int(params.get('print_epoch_interval', 1))
                if interval <= 0:
                    interval = 1
                if epoch % interval == 0 or epoch == params['epochs'] - 1:
                    print("[epoch] epoch={} train_loss={:.6f} val_loss={:.6f} train_mae={:.6f} val_mae={:.6f} test_mae={:.6f} lr={:.6g} sec={:.2f}".format(
                        epoch,
                        epoch_train_loss,
                        epoch_val_loss,
                        epoch_train_mae,
                        epoch_val_mae,
                        epoch_test_mae,
                        optimizer.param_groups[0]['lr'],
                        per_epoch_time[-1],
                    ), flush=True)
                # COLAB_GIN_ZINC_PATCH_END

                # COLAB_GIN_ZINC_LR_STEP_START: validation-triggered LR changes are
                # used only for the untouched official plateau schedule. The default
                # project runtime uses a GRIT-matched epoch schedule.
                if lr_schedule_type == 'plateau':
                    scheduler.step(epoch_val_loss)
                # COLAB_GIN_ZINC_LR_STEP_END
"""
    if marker not in text:
        raise RuntimeError("could not patch official training script: scheduler marker not found")
    text = text.replace(marker, inject, 1)
    text = patch_official_lr_schedule_text(text)
    text = patch_official_no_early_stop_text(text)
    text = patch_official_checkpoint_cleanup_text(text)
    path.write_text(text, encoding="utf-8")
    print("[patch] Added Drive-friendly best/latest checkpoint and CSV logging patch.", flush=True)
    patch_official_molecule_loader(repo_dir)


def patch_official_lr_schedule_text(text: str) -> str:
    if "COLAB_GIN_ZINC_LR_SCHEDULE_START" not in text:
        old = """\
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
"""
        new = """\
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    # COLAB_GIN_ZINC_LR_SCHEDULE_START: keep the official optimizer and default
    # LR=1e-3, but use the same kind of epoch schedule as GRIT ZINC by default:
    # 50 warmup epochs followed by cosine decay to 1e-6. Set
    # params['lr_schedule']='plateau' to recover the untouched official scheduler,
    # or 'fixed' for a constant LR.
    lr_schedule_type = str(params.get('lr_schedule', 'cosine_warmup'))
    base_lr = float(params['init_lr'])
    warmup_epochs = int(params.get('warmup_epochs', 50))
    cosine_min_lr = float(params.get('cosine_min_lr', 1e-6))

    def _colab_epoch_lr(epoch):
        if lr_schedule_type == 'fixed':
            return base_lr
        if lr_schedule_type == 'cosine_warmup':
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return base_lr * float(epoch + 1) / float(warmup_epochs)
            cosine_epochs = max(1, int(params['epochs']) - warmup_epochs)
            progress = min(1.0, max(0.0, float(epoch - warmup_epochs) / float(cosine_epochs)))
            return cosine_min_lr + 0.5 * (base_lr - cosine_min_lr) * (1.0 + np.cos(np.pi * progress))
        return base_lr

    def _colab_set_epoch_lr(epoch):
        if lr_schedule_type in ['fixed', 'cosine_warmup']:
            lr = float(_colab_epoch_lr(epoch))
            for group in optimizer.param_groups:
                group['lr'] = lr

    scheduler = None
    if lr_schedule_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=params['lr_reduce_factor'],
                                                         patience=params['lr_schedule_patience'],
                                                         verbose=True)
    elif lr_schedule_type not in ['fixed', 'cosine_warmup']:
        raise ValueError("unknown lr_schedule: {}".format(lr_schedule_type))
    print("[lr] schedule={} base_lr={} warmup_epochs={} cosine_min_lr={}".format(
        lr_schedule_type, base_lr, warmup_epochs, cosine_min_lr
    ))
    # COLAB_GIN_ZINC_LR_SCHEDULE_END
"""
        if old not in text:
            raise RuntimeError("could not patch official optimizer/scheduler block")
        text = text.replace(old, new, 1)

    if "COLAB_GIN_ZINC_LR_EPOCH_START" not in text:
        old = """\
                t.set_description('Epoch %d' % epoch)

                start = time.time()
"""
        new = """\
                t.set_description('Epoch %d' % epoch)

                # COLAB_GIN_ZINC_LR_EPOCH_START
                _colab_set_epoch_lr(epoch)
                # COLAB_GIN_ZINC_LR_EPOCH_END

                start = time.time()
"""
        if old not in text:
            raise RuntimeError("could not patch official epoch LR hook")
        text = text.replace(old, new, 1)

    if "COLAB_GIN_ZINC_LR_STEP_START" not in text:
        old = "                scheduler.step(epoch_val_loss)\n"
        new = """\
                # COLAB_GIN_ZINC_LR_STEP_START: validation-triggered LR changes are
                # used only for the untouched official plateau schedule. The default
                # project runtime uses a GRIT-matched epoch schedule.
                if lr_schedule_type == 'plateau':
                    scheduler.step(epoch_val_loss)
                # COLAB_GIN_ZINC_LR_STEP_END
"""
        if old in text:
            text = text.replace(old, new, 1)

    return text


def patch_official_no_early_stop_text(text: str) -> str:
    if "COLAB_GIN_ZINC_NO_TIME_STOP_START" not in text:
        old = """\
                # Stop training after params['max_time'] hours
                if time.time()-t0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
"""
        new = """\
                # COLAB_GIN_ZINC_NO_TIME_STOP_START: max_time <= 0 disables this
                # official wall-time stop so the run continues to the requested
                # epoch count unless interrupted.
                max_time_hours = float(params.get('max_time', 0) or 0)
                if max_time_hours > 0 and time.time()-t0 > max_time_hours*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(max_time_hours))
                    break
                # COLAB_GIN_ZINC_NO_TIME_STOP_END
"""
        if old not in text:
            raise RuntimeError("could not patch official max_time stop block")
        text = text.replace(old, new, 1)

    if "COLAB_GIN_ZINC_NO_MIN_LR_STOP_START" not in text:
        old = """\
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\\n!! LR EQUAL TO MIN LR SET.")
                    break
"""
        new = """\
                # COLAB_GIN_ZINC_NO_MIN_LR_STOP_START: min_lr <= 0 disables this
                # official LR-based stop so training continues to the requested
                # epoch count.
                min_lr_stop = float(params.get('min_lr', 0) or 0)
                if min_lr_stop > 0 and optimizer.param_groups[0]['lr'] < min_lr_stop:
                    print("\\n!! LR EQUAL TO MIN LR SET.")
                    break
                # COLAB_GIN_ZINC_NO_MIN_LR_STOP_END
"""
        if old not in text:
            raise RuntimeError("could not patch official min_lr stop block")
        text = text.replace(old, new, 1)

    return text


def patch_official_checkpoint_cleanup_text(text: str) -> str:
    if "COLAB_GIN_ZINC_CKPT_CLEANUP_START" in text:
        return text
    old = """\
                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)
"""
    new = """\
                # COLAB_GIN_ZINC_CKPT_CLEANUP_START: official cleanup assumes every
                # .pkl file is named epoch_N.pkl. Keep that behavior for official
                # epoch checkpoints, but ignore extra best/latest state dicts.
                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    base_name = os.path.basename(file)
                    if not (base_name.startswith('epoch_') and base_name.endswith('.pkl')):
                        continue
                    epoch_nb = base_name.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)
                # COLAB_GIN_ZINC_CKPT_CLEANUP_END
"""
    if old not in text:
        raise RuntimeError("could not patch official checkpoint cleanup block")
    return text.replace(old, new, 1)


def patch_official_molecule_loader(repo_dir: Path) -> None:
    path = repo_dir / "data" / "molecules.py"
    text = path.read_text(encoding="utf-8")
    marker = "import dgl\n"
    patch = """import dgl

# COLAB_GIN_ZINC_MOLECULE_PICKLE_COMPAT_START: official ZINC.pkl was serialized
# under an older DGL class name. Install the alias in the dataset loader module
# immediately before pickle.load sees graph objects.
import importlib as _colab_importlib
import sys as _colab_sys
_colab_dgl_heterograph = _colab_importlib.import_module("dgl.heterograph")
if not hasattr(_colab_dgl_heterograph, "DGLHeteroGraph"):
    if hasattr(_colab_dgl_heterograph, "DGLGraph"):
        _colab_dgl_heterograph.DGLHeteroGraph = _colab_dgl_heterograph.DGLGraph
    else:
        import torch as _colab_torch
        _empty = _colab_torch.tensor([], dtype=_colab_torch.int64)
        _colab_dgl_heterograph.DGLHeteroGraph = type(dgl.graph((_empty, _empty)))
_colab_sys.modules["dgl.heterograph"].DGLHeteroGraph = _colab_dgl_heterograph.DGLHeteroGraph
# COLAB_GIN_ZINC_MOLECULE_PICKLE_COMPAT_END
"""
    pattern = re.compile(
        r"import dgl\n\n# COLAB_GIN_ZINC_MOLECULE_PICKLE_COMPAT_START:.*?# COLAB_GIN_ZINC_MOLECULE_PICKLE_COMPAT_END\n",
        re.DOTALL,
    )
    if "COLAB_GIN_ZINC_MOLECULE_PICKLE_COMPAT_START" in text:
        text, replaced = pattern.subn(patch, text, count=1)
        if replaced != 1:
            raise RuntimeError("could not replace existing molecule pickle compatibility patch")
    else:
        if marker not in text:
            raise RuntimeError("could not patch official molecule loader: import dgl marker not found")
        text = text.replace(marker, patch, 1)
    text = patch_official_molecule_collate_text(text)
    path.write_text(text, encoding="utf-8")
    print("[patch] Added official ZINC.pkl DGL compatibility/collate patch.", flush=True)


def patch_official_molecule_collate_text(text: str) -> str:
    if "COLAB_GIN_ZINC_LABEL_SHAPE_START" in text:
        return text
    old = "        labels = torch.tensor(np.array(labels)).unsqueeze(1)\n"
    new = """\
        # COLAB_GIN_ZINC_LABEL_SHAPE_START: modern unpickling returns scalar
        # targets as small tensors in some Colab/DGL stacks. Flatten to the
        # official regression shape [batch, 1] so L1/MAE does not broadcast.
        labels = torch.as_tensor(np.array(labels), dtype=torch.float32).reshape(-1, 1)
        # COLAB_GIN_ZINC_LABEL_SHAPE_END
"""
    count = text.count(old)
    if count < 1:
        raise RuntimeError("could not patch official molecule label collation")
    return text.replace(old, new)



def verify_official_runtime_compat(repo_dir: Path) -> None:
    code = r"""
import importlib
import dgl
import data.molecules
hg = importlib.import_module("dgl.heterograph")
cls = getattr(hg, "DGLHeteroGraph", None)
print("[compat] dgl", dgl.__version__, "DGLHeteroGraph_alias", cls is not None, "alias_class", getattr(cls, "__name__", None))
if cls is None:
    raise RuntimeError("DGLHeteroGraph alias was not installed")
"""
    run_cmd([sys.executable, "-c", code], cwd=repo_dir)


def write_runtime_config(
    official_cfg_path: Path,
    run_dir: Path,
    out_dir: Path,
    *,
    seed: int | None,
    epochs: int | None,
    max_time: float | None,
    gpu_id: int,
    print_epoch_interval: int | None,
    lr_schedule: str,
    warmup_epochs: int,
    cosine_min_lr: float,
) -> Path:
    cfg = load_json(official_cfg_path)
    cfg["gpu"]["use"] = True
    cfg["gpu"]["id"] = int(gpu_id)
    cfg["out_dir"] = str(out_dir) + "/"
    if seed is not None:
        cfg["params"]["seed"] = int(seed)
    if epochs is not None:
        cfg["params"]["epochs"] = int(epochs)
    cfg["params"]["max_time"] = float(DEFAULT_RUNTIME_MAX_TIME if max_time is None else max_time)
    cfg["params"]["min_lr"] = 0.0
    cfg["params"]["lr_schedule"] = str(lr_schedule)
    cfg["params"]["warmup_epochs"] = int(warmup_epochs)
    cfg["params"]["cosine_min_lr"] = float(cosine_min_lr)
    if print_epoch_interval is not None:
        cfg["params"]["print_epoch_interval"] = int(print_epoch_interval)
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = config_dir / "molecules_graph_regression_GIN_ZINC_500k.runtime.json"
    runtime_config.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"[config] runtime config: {runtime_config}", flush=True)
    return runtime_config


def find_newest(paths: Sequence[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def parse_param_count(log_path: Path) -> int | None:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"MODEL/Total parameters:\s+GIN\s+(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def collect_artifacts(
    drive_root: Path,
    run_dir: Path,
    official_out_dir: Path,
    raw_log_path: Path,
    runtime_config: Path,
    official_commit: str,
    *,
    allow_param_drift: bool,
) -> None:
    latest_outputs = drive_root / "latest_outputs"
    latest_outputs.mkdir(parents=True, exist_ok=True)

    best_ckpt = find_newest(list(official_out_dir.glob("checkpoints/**/best_val_mae_state_dict.pkl")))
    latest_ckpt = find_newest(list(official_out_dir.glob("checkpoints/**/latest_state_dict.pkl")))
    epoch_ckpt = find_newest(list(official_out_dir.glob("checkpoints/**/*.pkl")))
    history_csv = find_newest(list(official_out_dir.glob("results/*_history.csv")))
    result_txt = find_newest(list(official_out_dir.glob("results/result_*.txt")))
    best_json = find_newest(list(official_out_dir.glob("results/*_best.json")))
    config_txt = find_newest(list(official_out_dir.glob("configs/config_*.txt")))

    param_count = parse_param_count(raw_log_path)
    if param_count is None:
        msg = "could not find official GIN parameter count in raw log"
        if allow_param_drift:
            print(f"[param-warning] {msg}", flush=True)
        else:
            raise RuntimeError(msg)
    elif param_count != EXPECTED_OFFICIAL_PARAMS:
        msg = f"official GIN parameter count {param_count} != expected {EXPECTED_OFFICIAL_PARAMS}"
        if allow_param_drift:
            print(f"[param-warning] {msg}", flush=True)
        else:
            raise RuntimeError(msg)
    else:
        print(f"[param-check] Matched expected official GIN ZINC 500k params: {param_count}", flush=True)

    copied: dict[str, str | None] = {}
    for label, src in {
        "best_checkpoint_pkl": best_ckpt,
        "latest_checkpoint_pkl": latest_ckpt or epoch_ckpt,
        "history_csv": history_csv,
        "result_txt": result_txt,
        "best_json": best_json,
        "official_config_txt": config_txt,
    }.items():
        if src is None:
            copied[label] = None
            continue
        dst = latest_outputs / f"gin_zinc_seed41_{label}{src.suffix}"
        shutil.copy2(src, dst)
        copied[label] = str(dst)
        if label.endswith("_checkpoint_pkl"):
            pt_dst = dst.with_suffix(".pt")
            shutil.copy2(src, pt_dst)
            copied[label.replace("_pkl", "_pt")] = str(pt_dst)

    runtime_copy = latest_outputs / runtime_config.name
    shutil.copy2(runtime_config, runtime_copy)

    summary = {
        "model": "GIN",
        "dataset": "ZINC",
        "official_repo": OFFICIAL_REPO,
        "official_commit": official_commit,
        "official_config": OFFICIAL_CONFIG,
        "expected_parameter_count": EXPECTED_OFFICIAL_PARAMS,
        "observed_parameter_count": param_count,
        "run_dir": str(run_dir),
        "official_out_dir": str(official_out_dir),
        "raw_log": str(raw_log_path),
        "runtime_config": str(runtime_config),
        "latest_outputs": str(latest_outputs),
        "artifacts": copied,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (latest_outputs / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] run summary: {run_dir / 'training_summary.json'}", flush=True)
    print(f"[done] latest outputs: {latest_outputs}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train official Benchmarking-GNNs GIN on ZINC in Colab.")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/gin_zinc_official")
    parser.add_argument("--repo-dir", default="/content/benchmarking-gnns")
    parser.add_argument("--run-id", default="gin_zinc_500k_seed41")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_RUNTIME_EPOCHS,
        help="Runtime training epochs. Defaults to 2000 for GRIT schedule fairness; pass 1000 for the untouched official GIN config schedule.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=None,
        help="Optional wall-time stop in hours. Defaults to disabled so training continues to --epochs.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["cosine_warmup", "fixed", "plateau"],
        default=DEFAULT_RUNTIME_LR_SCHEDULE,
        help="Default cosine_warmup matches GRIT ZINC's warmup+cosine schedule style. Use fixed for constant LR or plateau for official GIN ReduceLROnPlateau.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=DEFAULT_RUNTIME_WARMUP_EPOCHS)
    parser.add_argument("--cosine-min-lr", type=float, default=DEFAULT_RUNTIME_COSINE_MIN_LR)
    parser.add_argument("--print-epoch-interval", type=int, default=5)
    parser.add_argument("--install-mode", choices=["compatible", "current", "none"], default="compatible")
    parser.add_argument("--skip-install", action="store_true", help="Alias for --install-mode none.")
    parser.add_argument("--force-clone", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--no-pin", action="store_true", help="Use current official master instead of the pinned commit.")
    parser.add_argument("--allow-config-drift", action="store_true")
    parser.add_argument("--allow-param-count-drift", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Prepare repo/config/data but do not train.")
    parser.add_argument(
        "--recover-shadow-only",
        action="store_true",
        help=(
            "Recover artifacts from a previous run that accidentally wrote to a local, unmounted "
            "/content/drive shadow directory, then exit without launching training."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    notebook_args = [x for x in unknown if x == "-f" or x.endswith(".json")]
    other_unknown = [x for x in unknown if x not in notebook_args]
    if notebook_args:
        print(f"[args] Ignoring Colab/Jupyter kernel argument(s): {notebook_args}", flush=True)
    if other_unknown:
        raise SystemExit(f"unrecognized arguments: {' '.join(other_unknown)}")
    if args.skip_install:
        args.install_mode = "none"

    drive_root = Path(args.drive_root)
    shadow_root = stash_unmounted_drive_root(drive_root)
    mount_drive()
    assert_drive_writable(drive_root)
    restore_shadow_to_drive(shadow_root, drive_root)
    if args.recover_shadow_only:
        if shadow_root is None:
            print(
                "[drive-recovery] No unmounted local shadow outputs were visible. If Drive is already mounted, "
                "unmount it first to reveal any hidden local /content/drive files.",
                flush=True,
            )
        print("[drive-recovery] Recovery-only mode complete; training was not launched.", flush=True)
        return
    repo_dir = Path(args.repo_dir)
    run_dir = drive_root / "results" / args.run_id
    official_out_dir = run_dir / "official_out"
    raw_log_path = run_dir / "logs" / "raw_training.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    install_dependencies(args.install_mode)
    official_commit = clone_or_update_repo(repo_dir, force=args.force_clone, pin_commit=not args.no_pin)

    official_cfg_path = repo_dir / OFFICIAL_CONFIG
    validate_official_config(official_cfg_path, allow_drift=args.allow_config_drift)
    ensure_zinc_data(repo_dir, drive_root, force=args.force_data)
    patch_official_training_script(repo_dir)
    if not (args.dry_run and args.install_mode == "none"):
        verify_official_runtime_compat(repo_dir)

    runtime_config = write_runtime_config(
        official_cfg_path,
        run_dir,
        official_out_dir,
        seed=args.seed,
        epochs=args.epochs,
        max_time=args.max_time,
        gpu_id=args.gpu_id,
        print_epoch_interval=args.print_epoch_interval,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        cosine_min_lr=args.cosine_min_lr,
    )

    manifest = {
        "runner": "experiments/zinc/training/gin_zinc_core.py",
        "official_repo": OFFICIAL_REPO,
        "official_branch": OFFICIAL_BRANCH,
        "official_commit": official_commit,
        "official_config": OFFICIAL_CONFIG,
        "runtime_config": str(runtime_config),
        "drive_root": str(drive_root),
        "run_dir": str(run_dir),
        "official_out_dir": str(official_out_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "lr_schedule": args.lr_schedule,
        "warmup_epochs": args.warmup_epochs,
        "cosine_min_lr": args.cosine_min_lr,
        "expected_parameter_count": EXPECTED_OFFICIAL_PARAMS,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print("[dry-run] Prepared repo, data, config, and manifest. Training not launched.", flush=True)
        return

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [
        sys.executable,
        "-u",
        "main_molecules_graph_regression.py",
        "--dataset",
        "ZINC",
        "--gpu_id",
        str(args.gpu_id),
        "--config",
        str(runtime_config),
        "--out_dir",
        str(official_out_dir) + "/",
    ]
    run_streaming_to_log(cmd, cwd=repo_dir, log_path=raw_log_path, env=env)
    collect_artifacts(
        drive_root,
        run_dir,
        official_out_dir,
        raw_log_path,
        runtime_config,
        official_commit,
        allow_param_drift=args.allow_param_count_drift,
    )


if __name__ == "__main__":
    main()
