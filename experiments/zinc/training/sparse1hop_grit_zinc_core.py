#!/usr/bin/env python3
"""Colab-friendly training runner for true-sparse 1-hop GRIT on ZINC.

Default run:

    from sparse1hop_grit_zinc_core import main
    main([])

Default Drive output:

    /content/drive/MyDrive/sparse1hop_grit_zinc

This runner trains ``Sparse1HopGRIT`` with the official ZINC GRIT RRWP settings
and ``num_global_tokens=4`` by default.  Pass
``--parameter-match-global-tokens`` to keep the four virtual nodes but make
their initial token/virtual-edge features fixed buffers, giving exactly the
official GRIT ZINC trainable parameter count.  The shared K=0 architecture is a
faithful true-sparse implementation of the official GRIT 1-hop control:
same TypeDict encoders, RRWP absolute/relative encoders, GRIT attention
equations, degree scaler, edge update, BN/FFN stack, add pooling, AdamW,
cosine-with-warmup schedule, and 2000 epochs.

The default K=4 model is intentionally not parameter-matched; it adds four
learned global tokens and three learned virtual-edge embeddings.  Use
``--num-global-tokens 0`` for the strict parity target, or
``--num-global-tokens 4 --parameter-match-global-tokens`` for the separate
parameter-matched global-token experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


DEFAULT_REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
DEFAULT_BRANCH = "main"
DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/sparse1hop_grit_zinc"
DEFAULT_OFFICIAL_1HOP_DIR = "/content/drive/MyDrive/grit_zinc_1hop"


class CommandError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(msg, flush=True)


def run_cmd(cmd: Sequence[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    printable = " ".join(map(str, cmd))
    log(f"[cmd] {printable}")
    proc = subprocess.run(list(map(str, cmd)), cwd=str(cwd) if cwd else None, text=True, check=False)
    if check and proc.returncode != 0:
        raise CommandError(f"command failed with exit code {proc.returncode}: {printable}")
    return proc


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def get_colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        if value:
            return str(value)
    except Exception:
        return None
    return os.environ.get(name)


def authenticated_url(repo_url: str, token: str | None) -> str:
    if not token or not repo_url.startswith("https://github.com/"):
        return repo_url
    return repo_url.replace("https://github.com/", f"https://x-access-token:{token}@github.com/", 1)


def mount_drive() -> None:
    if not in_colab():
        return
    from google.colab import drive  # type: ignore

    drive.mount("/content/drive")


def install_dependencies(skip_install: bool) -> None:
    if skip_install:
        return
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools", "wheel"])
    # The model itself is torch-only; torch-geometric is used for the ZINC dataset
    # and batching.  No torch-scatter/torch-sparse custom ops are required by the
    # true-sparse implementation.
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "torch-geometric", "pandas", "numpy", "pyyaml", "tqdm"])


def clone_or_update_repo(repo_url: str, branch: str, repo_dir: Path, secret_name: str, skip_git: bool) -> None:
    if skip_git:
        sys.path.insert(0, str(repo_dir / "src"))
        return
    token = get_colab_secret(secret_name)
    url = authenticated_url(repo_url, token)
    safe = "<token-authenticated-url>" if token else repo_url
    if repo_dir.exists():
        log(f"[git] updating {repo_dir}")
        run_cmd(["git", "-C", str(repo_dir), "fetch", "origin", branch])
        run_cmd(["git", "-C", str(repo_dir), "checkout", branch])
        run_cmd(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", branch])
    else:
        log(f"[git] cloning {safe} branch={branch} -> {repo_dir}")
        run_cmd(["git", "clone", "--branch", branch, url, str(repo_dir)])
        run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_dir), "--no-deps"])
    sys.path.insert(0, str(repo_dir / "src"))


def ensure_imports() -> None:
    import torch  # noqa: F401
    import torch_geometric  # noqa: F401
    from graph_specialisation_metrics.sparse1hop_grit import Sparse1HopGRIT  # noqa: F401


class Sparse1HopRRWP:
    """Precompute official-orientation RRWP features for sparse support edges."""

    def __init__(self, walk_length: int = 21, add_identity: bool = True) -> None:
        self.walk_length = int(walk_length)
        self.add_identity = bool(add_identity)

    def __call__(self, data: Any) -> Any:
        import torch

        n = int(data.num_nodes)
        edge_index = data.edge_index.long()
        device = edge_index.device
        adj = torch.zeros((n, n), dtype=torch.float32, device=device)
        if edge_index.numel():
            adj[edge_index[0], edge_index[1]] = 1.0
        deg = adj.sum(dim=1)
        deg_inv = torch.zeros_like(deg)
        mask = deg > 0
        deg_inv[mask] = 1.0 / deg[mask]
        trans = adj * deg_inv.view(-1, 1)

        pe_list = []
        if self.add_identity:
            pe_list.append(torch.eye(n, dtype=torch.float32, device=device))
        out = trans
        pe_list.append(out)
        while len(pe_list) < self.walk_length:
            out = out @ trans
            pe_list.append(out)
        pe = torch.stack(pe_list[: self.walk_length], dim=-1)
        data.rrwp = pe.diagonal(dim1=0, dim2=1).transpose(0, 1).contiguous()
        # Official GRIT stores rel_pe_idx as [col,row], so an edge source->dest
        # receives PE[dest, source].
        if edge_index.numel():
            data.edge_rrwp = pe[edge_index[1], edge_index[0]].contiguous()
        else:
            data.edge_rrwp = torch.empty((0, self.walk_length), dtype=torch.float32, device=device)
        data.log_deg = torch.log(deg + 1.0)
        data.deg = deg.long()
        return data


def load_zinc_datasets(data_root: Path, *, force_data: bool) -> tuple[Any, Any, Any]:
    from torch_geometric.datasets import ZINC

    if force_data and data_root.exists():
        shutil.rmtree(data_root)
    pre = Sparse1HopRRWP(walk_length=21, add_identity=True)
    train = ZINC(str(data_root), subset=True, split="train", pre_transform=pre)
    val = ZINC(str(data_root), subset=True, split="val", pre_transform=pre)
    test = ZINC(str(data_root), subset=True, split="test", pre_transform=pre)
    return train, val, test


def make_loaders(train: Any, val: Any, test: Any, batch_size: int, workers: int) -> tuple[Any, Any, Any]:
    from torch_geometric.loader import DataLoader

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True),
        DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
    )


def lr_for_epoch(epoch: int, *, base_lr: float, min_lr: float, max_epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(warmup_epochs)
    if max_epoch <= warmup_epochs:
        return min_lr
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / float(max_epoch - warmup_epochs)))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: Any, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def mae_epoch(model: Any, loader: Any, device: str, *, train: bool, optimizer: Any | None = None, clip_grad: bool = True) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.train(train)
    total_loss = 0.0
    total_abs = 0.0
    total_graphs = 0
    start = time.time()
    if train and optimizer is None:
        raise ValueError("optimizer is required for train=True")
    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            pred = model(batch).view(-1)
            target = batch.y.view(-1).to(pred.dtype)
            loss = F.l1_loss(pred, target, reduction="mean")
            if train:
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        graphs = int(target.numel())
        total_loss += float(loss.detach().cpu().item()) * graphs
        total_abs += float((pred.detach() - target.detach()).abs().sum().cpu().item())
        total_graphs += graphs
    denom = max(1, total_graphs)
    return {"loss": total_loss / denom, "mae": total_abs / denom, "graphs": total_graphs, "seconds": time.time() - start}


def checkpoint_candidates(root: Path) -> list[Path]:
    patterns = ["*.ckpt", "*.pt", "*.pth"]
    cands = sorted({p for pat in patterns for p in root.rglob(pat) if p.is_file()}, key=lambda p: p.stat().st_mtime, reverse=True)
    return cands


def extract_state_dict(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        for key in ("model_state_dict", "model_state", "state_dict", "model"):
            val = obj.get(key)
            if isinstance(val, Mapping):
                return val
        if obj and all(hasattr(v, "shape") for v in obj.values()):
            return obj
    raise ValueError("could not extract model state dict from checkpoint")


def maybe_warm_start_from_official(model: Any, official_dir: Path, device: str, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    import torch
    from graph_specialisation_metrics.sparse1hop_grit import load_official_grit_state_dict

    cands = checkpoint_candidates(official_dir / "results")
    if not cands:
        return {"enabled": True, "status": "no_checkpoint_found", "searched": str(official_dir / "results")}
    ckpt = cands[0]
    log(f"[warm-start] loading shared official 1-hop weights: {ckpt}")
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    state = extract_state_dict(obj)
    report = load_official_grit_state_dict(model, state, strict_shared=True)
    report.update({"enabled": True, "status": "loaded", "checkpoint": str(ckpt)})
    return report


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_history(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(dict(row))


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(path)


def load_resume_checkpoint(path: Path, model: Any, optimizer: Any, device: str) -> tuple[int, float]:
    import torch

    obj = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(obj["model_state_dict"])
    optimizer.load_state_dict(obj["optimizer_state_dict"])
    return int(obj.get("epoch", -1)) + 1, float(obj.get("best_val_mae", float("inf")))


def run_training(args: argparse.Namespace) -> Path:
    import torch
    from graph_specialisation_metrics.sparse1hop_grit import (
        OFFICIAL_ZINC_GRIT_RRWP_PARAMS,
        Sparse1HopGRIT,
        assert_official_parameter_parity,
        parameter_count,
        zinc_sparse1hop_grit_config,
    )

    drive = Path(args.drive_dir)
    match_tag = "_pmatch" if args.parameter_match_global_tokens else ""
    run_dir = drive / "results" / f"sparse1hop_grit_zinc_gtok{args.num_global_tokens}{match_tag}_seed{args.seed}"
    ckpt_dir = run_dir / "ckpt"
    data_root = drive / "datasets" / "zinc_subset_rrwp21_sparse1hop"
    history_path = run_dir / "metrics" / "history_metrics.csv"
    save_json(
        run_dir / "config.json",
        {
            "model": "Sparse1HopGRIT",
            "num_global_tokens": args.num_global_tokens,
            "parameter_match_global_tokens": bool(args.parameter_match_global_tokens),
            "global_tokens_trainable": not bool(args.parameter_match_global_tokens),
            "virtual_edge_features_trainable": not bool(args.parameter_match_global_tokens),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "base_lr": args.base_lr,
            "min_lr": args.min_lr,
            "warmup_epochs": args.warmup_epochs,
            "official_reference_params_k0": OFFICIAL_ZINC_GRIT_RRWP_PARAMS,
            "official_equation_sources": [
                "LiamMa/GRIT grit/encoder/type_dict_encoder.py",
                "LiamMa/GRIT grit/encoder/rrwp_encoder.py",
                "LiamMa/GRIT grit/layer/grit_layer.py",
                "LiamMa/GRIT grit/head/san_graph.py",
            ],
        },
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[device] {device}")

    train, val, test = load_zinc_datasets(data_root, force_data=args.force_data)
    loaders = make_loaders(train, val, test, args.batch_size, args.workers)
    cfg = zinc_sparse1hop_grit_config(
        num_global_tokens=args.num_global_tokens,
        parameter_match_global_tokens=args.parameter_match_global_tokens,
    )
    model = Sparse1HopGRIT(cfg).to(device)
    params = parameter_count(model)
    if args.num_global_tokens == 0 or args.parameter_match_global_tokens:
        assert_official_parameter_parity(model)
    log(f"[model] trainable params={params:,} (official K=0 reference={OFFICIAL_ZINC_GRIT_RRWP_PARAMS:,})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    warm_start_report = maybe_warm_start_from_official(
        model,
        Path(args.official_1hop_drive_dir),
        device,
        enabled=args.warm_start_official_1hop and not args.force_retrain,
    )
    save_json(run_dir / "warm_start_report.json", warm_start_report)

    start_epoch = 0
    best_val = float("inf")
    latest = ckpt_dir / "latest.ckpt"
    if latest.exists() and not args.force_retrain:
        start_epoch, best_val = load_resume_checkpoint(latest, model, optimizer, device)
        log(f"[resume] {latest} -> start_epoch={start_epoch}, best_val_mae={best_val:.6f}")
    elif args.force_retrain and run_dir.exists():
        log("[force] force_retrain set; existing checkpoints/history will be overwritten as training progresses")

    for epoch in range(start_epoch, args.epochs):
        lr = lr_for_epoch(epoch, base_lr=args.base_lr, min_lr=args.min_lr, max_epoch=args.epochs, warmup_epochs=args.warmup_epochs)
        set_lr(optimizer, lr)
        train_stats = mae_epoch(model, loaders[0], device, train=True, optimizer=optimizer, clip_grad=True)
        with torch.no_grad():
            val_stats = mae_epoch(model, loaders[1], device, train=False)
            test_stats = mae_epoch(model, loaders[2], device, train=False)
        is_best = val_stats["mae"] < best_val
        if is_best:
            best_val = val_stats["mae"]
        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_stats["loss"],
            "train_mae": train_stats["mae"],
            "val_loss": val_stats["loss"],
            "val_mae": val_stats["mae"],
            "test_loss": test_stats["loss"],
            "test_mae": test_stats["mae"],
            "best_val_mae": best_val,
            "is_best": int(is_best),
            "epoch_seconds": train_stats["seconds"] + val_stats["seconds"] + test_stats["seconds"],
            "params": params,
        }
        append_history(history_path, row)
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_mae": best_val,
            "config": vars(args),
            "row": row,
        }
        save_checkpoint(latest, payload)
        if is_best and epoch >= args.warmup_epochs:
            save_checkpoint(ckpt_dir / "best.ckpt", payload)
            save_json(ckpt_dir / "best_summary.json", row)
        if epoch == start_epoch or ((epoch + 1) % args.checkpoint_period == 0):
            save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.ckpt", payload)
        if epoch < 3 or is_best or ((epoch + 1) % args.print_period == 0):
            log(
                f"[epoch {epoch:04d}] train={train_stats['mae']:.5f} "
                f"val={val_stats['mae']:.5f} test={test_stats['mae']:.5f} "
                f"best_val={best_val:.5f} lr={lr:.2e} best={int(is_best)}"
            )

    save_json(
        run_dir / "training_summary.json",
        {
            "status": "complete",
            "run_dir": str(run_dir),
            "latest_checkpoint": str(latest),
            "best_checkpoint": str(ckpt_dir / "best.ckpt"),
            "history": str(history_path),
            "params": params,
        },
    )
    log(f"[done] {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--drive-dir", default=DEFAULT_DRIVE_DIR)
    p.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    p.add_argument("--branch", default=DEFAULT_BRANCH)
    p.add_argument("--repo-dir", type=Path, default=Path("/content/Graph-Specialisation-and-Metrics"))
    p.add_argument("--secret-name", default="dissertation_key")
    p.add_argument("--skip-git", action="store_true")
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--force-data", action="store_true")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--num-global-tokens", type=int, default=4)
    p.add_argument(
        "--parameter-match-global-tokens",
        action="store_true",
        help="Keep virtual global nodes but make their initial token/virtual-edge features fixed buffers so trainable params equal official GRIT.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--warmup-epochs", type=int, default=50)
    p.add_argument("--base-lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--checkpoint-period", type=int, default=100)
    p.add_argument("--print-period", type=int, default=10)
    p.add_argument("--official-1hop-drive-dir", default=DEFAULT_OFFICIAL_1HOP_DIR)
    p.add_argument("--warm-start-official-1hop", action="store_true", default=False)
    return p


def main(argv: Sequence[str] | None = None) -> Path:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown and any(arg == "-f" or arg.endswith(".json") for arg in unknown):
        log(f"[args] ignoring notebook launcher arguments: {unknown}")
    elif unknown:
        raise SystemExit(f"unrecognized arguments: {unknown}")
    mount_drive()
    install_dependencies(args.skip_install)
    clone_or_update_repo(args.repo_url, args.branch, args.repo_dir, args.secret_name, args.skip_git)
    ensure_imports()
    return run_training(args)


if __name__ == "__main__":
    main()
