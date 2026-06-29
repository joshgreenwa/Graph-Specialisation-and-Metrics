#!/usr/bin/env python3
"""Standalone Google Colab runner for the ZINC dissertation core procedure.

This runner is intentionally orchestration-only. It clones the current
methodology repo branch with the Colab secret ``dissertation_key``, discovers
the default Drive outputs from the dense GRIT and 1-hop GRIT ZINC Colab
training runners, prepares a ZINC config, and invokes the repo's
``graph_specialisation_metrics.main_procedure`` CLI for Steps 0-5.

The repo implementation runs the full GRIT/1-hop GRIT intervention procedure
when trained checkpoints are present. This file records any missing checkpoints,
dependency failures, or deliberately omitted validation references in Drive and,
by default, fails if the requested complete run cannot be completed. Pass
``--allow-incomplete`` only when you want a discovery/preflight artifact.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote


PUBLIC_REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
DEFAULT_BRANCH = "codex/cfim-grit-experiments"
DEFAULT_REPO_DIR = "/content/Graph-Specialisation-and-Metrics"
DEFAULT_SECRET_NAME = "dissertation_key"

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/graph_specialisation_metrics/zinc_main_procedure_colab"
DEFAULT_DENSE_DRIVE_DIR = "/content/drive/MyDrive/grit_zinc_official"
DEFAULT_ONEHOP_DRIVE_DIR = "/content/drive/MyDrive/grit_zinc_1hop"

OFFICIAL_GRIT_REPO = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
EXPECTED_GRIT_PARAMS = 473_473
ONE_HOP_CFG_TEXT = """\
# Parameter-matched 1-hop sparse-control variant of the official ZINC GRIT RRWP config.
#
# This file intentionally keeps the official ZINC subset model, optimizer,
# training schedule, RRWP dimensions, and decoder settings. The only scientific
# intervention is `gt.attn.sparsity: one_hop`, which selects the masked RRWP edge
# encoder added by the local patch. That encoder keeps attention/message passing
# on original molecular bonds plus self-loops after RRWP encoding instead of the
# official complete-graph attention. The trainable parameter count remains
# 473,473, matching the official GRIT ZINC RRWP config.
out_dir: results
metric_best: mae
metric_agg: argmin
tensorboard_each_run: True
accelerator: "cuda:0"
mlflow:
  use: False
  project: Exp
  name: zinc-GRIT-RRWP-1hop
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
    sparsity: one_hop
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

MARKDOWN_REQUIRED_STEPS = {
    "0": "measurement_model_validation",
    "1": "performance_gap",
    "2": "usage_vs_causal_usage",
    "3": "distance_resolved_overfitting",
    "4": "mediator_patching",
    "5": "non_composable_gap_attribution",
}


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | str | None = None,
    safe_display: str | None = None,
    check: bool = True,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    display = safe_display or " ".join(str(c) for c in cmd)
    print(f"[cmd] {display}", flush=True)
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=dict(env) if env is not None else None,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {display}")
    return proc


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return path


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    return path


def require_colab_token(secret_name: str) -> str:
    try:
        from google.colab import userdata
    except Exception as exc:  # pragma: no cover - outside Colab.
        raise RuntimeError("Run this file in Google Colab; it uses google.colab.userdata.") from exc
    token = userdata.get(secret_name)
    if not token:
        raise RuntimeError(f"Colab secret {secret_name!r} is missing or empty")
    secret = str(token).strip()
    prefix = "github_pat_" if secret.startswith("github_pat_") else secret[:4]
    print(f"[auth] secret {secret_name!r} loaded; length={len(secret)}, prefix={prefix!r}", flush=True)
    return secret


def mount_drive() -> None:
    try:
        from google.colab import drive
    except Exception as exc:  # pragma: no cover - outside Colab.
        raise RuntimeError("Google Drive mounting requires Google Colab") from exc
    drive.mount("/content/drive", force_remount=False)


def sanitize_repo_url(repo_url: str) -> str:
    repo_url = str(repo_url).strip()
    if repo_url.startswith("[") and "](" in repo_url and repo_url.endswith(")"):
        repo_url = repo_url.split("](", 1)[1][:-1]
    return repo_url


def repo_owner(public_url: str) -> str:
    public_url = sanitize_repo_url(public_url)
    if not public_url.startswith("https://github.com/"):
        raise ValueError("only https://github.com repo URLs are supported")
    return public_url.removeprefix("https://github.com/").split("/", 1)[0]


def auth_url_candidates(public_url: str, token: str, username: str | None = None) -> list[str]:
    public_url = sanitize_repo_url(public_url)
    user = quote(username or repo_owner(public_url), safe="")
    token_q = quote(token, safe="")
    suffix = public_url.removeprefix("https://github.com/")
    return [
        f"https://{user}:{token_q}@github.com/{suffix}",
        f"https://x-access-token:{token_q}@github.com/{suffix}",
    ]


def clone_or_update_repo(
    repo_url: str,
    branch: str,
    repo_dir: Path,
    token: str,
    github_username: str | None,
) -> None:
    repo_url = sanitize_repo_url(repo_url)
    candidates = auth_url_candidates(repo_url, token, github_username)
    authed = candidates[0]
    preflight: subprocess.CompletedProcess[str] | None = None
    for idx, candidate in enumerate(candidates, start=1):
        preflight = run_cmd(
            ["git", "ls-remote", "--heads", candidate, branch],
            safe_display=f"git ls-remote --heads <token-authenticated-url> {branch} [auth-form {idx}]",
            check=False,
        )
        if preflight.returncode == 0:
            authed = candidate
            break
    assert preflight is not None
    if preflight.returncode != 0:
        raise RuntimeError(
            "GitHub authentication failed. A fine-grained PAT must select this repo "
            "and grant Repository permissions: Contents=Read-only. If the branch is "
            "not pushed to GitHub, push it or pass --branch for a pushed branch."
        )
    if not preflight.stdout.strip():
        raise RuntimeError(f"GitHub authentication worked, but branch {branch!r} was not found")

    if (repo_dir / ".git").exists():
        run_cmd(
            ["git", "-C", str(repo_dir), "remote", "set-url", "origin", authed],
            safe_display=f"git -C {repo_dir} remote set-url origin <token-authenticated-url>",
        )
        run_cmd(["git", "-C", str(repo_dir), "fetch", "origin", branch])
        run_cmd(["git", "-C", str(repo_dir), "checkout", branch])
        run_cmd(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", branch])
        run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])
        return

    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    run_cmd(
        ["git", "clone", "--branch", branch, "--single-branch", authed, str(repo_dir)],
        safe_display=f"git clone --branch {branch} <token-authenticated-url> {repo_dir}",
    )
    run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])


def install_repo(repo_dir: Path) -> None:
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "networkx", "matplotlib", "numpy", "scipy", "pandas"])
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_dir), "--no-deps"])
    for path in [str(repo_dir), str(repo_dir / "src")]:
        if path not in sys.path:
            sys.path.insert(0, path)


def clone_official_grit(repo_dir: Path, *, force: bool = False) -> None:
    if force and repo_dir.exists():
        shutil.rmtree(repo_dir)
    if not (repo_dir / ".git").exists():
        run_cmd(["git", "clone", "--branch", "main", OFFICIAL_GRIT_REPO, str(repo_dir)])
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    if current != OFFICIAL_GRIT_COMMIT:
        run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
        run_cmd(["git", "checkout", OFFICIAL_GRIT_COMMIT], cwd=repo_dir)
    print(f"[grit] official checkout {repo_dir} @ {OFFICIAL_GRIT_COMMIT}", flush=True)


def _read_text_preserve_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as f:
        return f.read()


def _write_text_preserve_newlines(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(text)


def _native_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _replace_exact(path: Path, old: str, new: str, marker: str, label: str) -> bool:
    text = _read_text_preserve_newlines(path)
    if marker in text:
        print(f"[patch] {label}: already present", flush=True)
        return False
    newline = _native_newline(text)
    old_native = old.replace("\n", newline)
    new_native = new.replace("\n", newline)
    if old_native not in text:
        raise RuntimeError(
            f"Could not apply 1-hop patch segment `{label}` to {path}. "
            "The official GRIT source differs from the pinned layout."
        )
    _write_text_preserve_newlines(path, text.replace(old_native, new_native, 1))
    print(f"[patch] {label}: applied", flush=True)
    return True


def apply_embedded_parameter_matched_onehop_patch(repo_dir: Path, drive_root: Path) -> None:
    """Apply the parameter-matched 1-hop GRIT patch without repo helper imports."""
    print("\n[patch] Applying embedded parameter-matched 1-hop GRIT control patch.", flush=True)
    cfg_path = repo_dir / "configs" / "GRIT" / "zinc-GRIT-RRWP-1hop.yaml"
    current_cfg = _read_text_preserve_newlines(cfg_path) if cfg_path.exists() else ""
    if current_cfg != ONE_HOP_CFG_TEXT:
        _write_text_preserve_newlines(cfg_path, ONE_HOP_CFG_TEXT)
        print(f"[patch] wrote exact 1-hop ZINC config: {cfg_path}", flush=True)
    else:
        print(f"[patch] exact 1-hop ZINC config already present: {cfg_path}", flush=True)

    _replace_exact(
        repo_dir / "grit" / "config" / "gt_config.py",
        old=(
            '    cfg.gt.attn.full_attn = True\n'
            '    cfg.gt.attn.norm_e = True\n'
        ),
        new=(
            '    cfg.gt.attn.full_attn = True\n'
            '    cfg.gt.attn.sparsity = "full"\n'
            '    cfg.gt.attn.norm_e = True\n'
        ),
        marker='cfg.gt.attn.sparsity = "full"',
        label="default gt.attn.sparsity",
    )
    _replace_exact(
        repo_dir / "grit" / "encoder" / "rrwp_encoder.py",
        old=(
            '        torch.nn.init.xavier_uniform_(self.fc.weight)\n'
            '        self.fill_value = 0.\n'
        ),
        new=(
            '        torch.nn.init.xavier_uniform_(self.fc.weight)\n'
            '        self.pad_to_full_graph = False\n'
            '        self.fill_value = 0.\n'
        ),
        marker="self.pad_to_full_graph = False",
        label="masked RRWP encoder repr compatibility",
    )
    _replace_exact(
        repo_dir / "grit" / "encoder" / "rrwp_encoder.py",
        old=(
            '    def __repr__(self):\n'
            '        return f"{self.__class__.__name__}" \\\n'
            '               f"(pad_to_full_graph={self.pad_to_full_graph}," \\\n'
            '               f"fill_value={self.fill_value}," \\\n'
            '               f"{self.fc.__repr__()})"\n'
        ),
        new=(
            '    def __repr__(self):\n'
            '        return f"{self.__class__.__name__}" \\\n'
            '               f"(pad_to_full_graph={self.pad_to_full_graph}," \\\n'
            '               f"fill_value={self.fill_value}," \\\n'
            '               f"mask_index_name={self.mask_index_name}," \\\n'
            '               f"{self.fc.__repr__()})"\n'
        ),
        marker="mask_index_name={self.mask_index_name}",
        label="masked RRWP encoder repr mask label",
    )
    _replace_exact(
        repo_dir / "grit" / "network" / "grit_model.py",
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
            '            elif attn_sparsity == "one_hop":\n'
            '                self.rrwp_rel_encoder = register.edge_encoder_dict["masked_rrwp_linear"] \\\n'
            '                    (rel_pe_dim, cfg.gnn.dim_edge,\n'
            '                     add_node_attr_as_self_loop=False,\n'
            '                     fill_value=0.,\n'
            '                     mask_index_name="edge_index",\n'
            '                     )\n'
            '            else:\n'
            '                raise ValueError(\n'
            '                    f"Unsupported cfg.gt.attn.sparsity={attn_sparsity!r}; "\n'
            '                    "expected \'full\' or \'one_hop\'."\n'
            '                )\n'
        ),
        marker='attn_sparsity = cfg.gt.attn.get("sparsity", "full")',
        label="GritTransformer RRWP sparsity switch",
    )
    write_json(
        drive_root / "patches" / "zinc_grit_rrwp_1hop_patch.json",
        {
            "official_repo": OFFICIAL_GRIT_REPO,
            "official_commit": OFFICIAL_GRIT_COMMIT,
            "one_hop_config": str(cfg_path),
            "parameter_count_guard": EXPECTED_GRIT_PARAMS,
            "scientific_change": "gt.attn.full_attn=False and gt.attn.sparsity=one_hop",
        },
    )


def prepare_grit_repos(repo_dir: Path, drive_root: Path, force: bool = False) -> tuple[Path, Path, Path, Path]:
    dense_repo = Path("/content/GRIT_dense_analysis")
    onehop_repo = Path("/content/GRIT_1hop_analysis")
    clone_official_grit(dense_repo, force=force)
    clone_official_grit(onehop_repo, force=force)

    apply_embedded_parameter_matched_onehop_patch(onehop_repo, drive_root)
    dense_cfg = dense_repo / "configs" / "GRIT" / "zinc-GRIT-RRWP.yaml"
    onehop_cfg = onehop_repo / "configs" / "GRIT" / "zinc-GRIT-RRWP-1hop.yaml"
    if not dense_cfg.exists():
        raise FileNotFoundError(f"missing dense GRIT config: {dense_cfg}")
    if not onehop_cfg.exists():
        raise FileNotFoundError(f"missing 1-hop GRIT config: {onehop_cfg}")
    return dense_repo, onehop_repo, dense_cfg, onehop_cfg


def checkpoint_candidates(root: Path) -> list[Path]:
    patterns = ["*.ckpt", "*checkpoint*.pt", "*checkpoint*.pth", "best*.pt", "best*.pth", "model*.pt"]
    out: list[Path] = []
    if root.exists():
        for pattern in patterns:
            out.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(out), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)


def choose_checkpoint(results_root: Path, label: str) -> Path:
    candidates = checkpoint_candidates(results_root)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for {label} under {results_root}. "
            "The training run must have saved at least one best checkpoint first."
        )
    chosen = candidates[0]
    print(f"[checkpoint] {label}: {chosen}", flush=True)
    return chosen


def wrapper_logs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in [root / "wrapper_logs", root]:
        if base.exists():
            candidates.extend(path for path in base.rglob("*.log") if path.is_file())
            candidates.extend(path for path in base.rglob("*.out") if path.is_file())
    return sorted(set(candidates), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)


def parse_grit_training_logs(log_paths: Sequence[Path], model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_epoch: dict[int, dict[str, Any]] = {}
    source_by_epoch: dict[int, str] = {}
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
                row = by_epoch.setdefault(epoch, {"model": model, "epoch": epoch})
                for key, value in stats.items():
                    if key == "epoch":
                        continue
                    out_key = f"{split}_{key}"
                    if out_key not in row:
                        row[out_key] = value
                source_by_epoch.setdefault(epoch, str(path))

    rows = []
    best_epoch: int | None = None
    best_val = float("inf")
    for epoch in sorted(by_epoch):
        row = by_epoch[epoch]
        row["source"] = source_by_epoch.get(epoch, "")
        rows.append(row)
        try:
            val = float(row.get("val_mae", row.get("val_loss", float("inf"))))
        except Exception:
            val = float("inf")
        if val < best_val:
            best_val = val
            best_epoch = epoch

    best_row = by_epoch.get(best_epoch, {}) if best_epoch is not None else {}
    summary = {
        "model": model,
        "num_rows": len(rows),
        "best_epoch": best_epoch,
        "best_val_mae": best_row.get("val_mae"),
        "best_val_loss": best_row.get("val_loss"),
        "best_test_mae": best_row.get("test_mae"),
        "best_test_loss": best_row.get("test_loss"),
        "log_files": [str(path) for path in log_paths[:20]],
    }
    return rows, summary


def prepare_model_artifact(
    *,
    model: str,
    source_drive_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    prepared_root: Path,
) -> dict[str, Any]:
    prepared_root.mkdir(parents=True, exist_ok=True)
    logs = wrapper_logs(source_drive_dir)
    rows, summary = parse_grit_training_logs(logs, model)
    metrics_csv = write_csv(prepared_root / "metrics" / "history_metrics.csv", rows)
    summary_json = write_json(prepared_root / "metrics" / "training_summary.json", summary)
    pointer = {
        "model": model,
        "source_drive_dir": str(source_drive_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "prepared_root": str(prepared_root),
        "metrics_csv": str(metrics_csv),
        "summary_json": str(summary_json),
        "logs_parsed": len(logs),
        "history_rows": len(rows),
    }
    write_json(prepared_root / "artifact_pointer.json", pointer)
    return pointer


def build_zinc_config(
    *,
    artifact_root: Path,
    dense_prepared: Path,
    onehop_prepared: Path,
    dense_dataset_dir: Path,
    onehop_dataset_dir: Path,
    dense_repo: Path,
    onehop_repo: Path,
    dense_cfg: Path,
    onehop_cfg: Path,
    dense_ckpt: Path,
    onehop_ckpt: Path,
    single_seed: int,
    include_local_references: bool,
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "dense_grit": {
            "adapter": "official_grit",
            "variant": "official",
            "role": "treatment",
            "official_repo": "https://github.com/LiamMa/GRIT",
            "repo_path": str(dense_repo),
            "artifact_root": str(dense_prepared),
            "dataset_dir": str(dense_dataset_dir),
            "config_path": str(dense_cfg),
            "checkpoint_path": str(dense_ckpt),
        },
        "grit_1hop": {
            "adapter": "official_grit",
            "variant": "1hop",
            "role": "parameter_matched_control",
            "official_repo": "https://github.com/LiamMa/GRIT",
            "repo_path": str(onehop_repo),
            "artifact_root": str(onehop_prepared),
            "dataset_dir": str(onehop_dataset_dir),
            "config_path": str(onehop_cfg),
            "checkpoint_path": str(onehop_ckpt),
        },
    }
    if include_local_references:
        models.update(
            {
                "gin": {
                    "adapter": "pyg_gin",
                    "role": "local_validation_reference",
                    "artifact_root": str(artifact_root / "missing_gin_reference"),
                },
                "gcn": {
                    "adapter": "pyg_gcn",
                    "role": "local_validation_reference",
                    "artifact_root": str(artifact_root / "missing_gcn_reference"),
                },
            }
        )

    return {
        "artifact_root": str(artifact_root / "artifacts"),
        "dataset": {"name": "ZINC", "split": "official_subset", "task": "molecular_regression"},
        "seeds": [int(single_seed)],
        "primary_tau": 3,
        "far_thresholds": [2, 3, 4],
        "perturbation": {
            "carriage_primary": "integrated_gradients",
            "ig_baseline": "mean_node_embedding",
            "ig_steps": 32,
            "swap_partners": 8,
        },
        "models": models,
        "steps": {
            "0": {"name": "measurement_model_validation", "sample_graphs": 200},
            "1": {"name": "performance_gap", "reach_sweep": [1, 2, 3, 5, "dense"]},
            "2": {"name": "usage_vs_causal_usage", "sample_graphs": 200},
            "3": {"name": "distance_resolved_overfitting", "sample_graphs": 200},
            "4": {"name": "mediator_patching", "sample_graphs": 100, "max_far_pairs_per_graph": 64},
            "5": {"name": "non_composable_gap_attribution", "sample_graphs": 200, "interaction_pairs": 1000},
        },
        "figures": {"dpi": 180},
        "colab_notes": {
            "requested_single_seed_only": True,
            "markdown_default_requires_three_seeds": True,
            "dense_default_drive_dir": DEFAULT_DENSE_DRIVE_DIR,
            "onehop_default_drive_dir": DEFAULT_ONEHOP_DRIVE_DIR,
        },
    }


def write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def run_main_procedure(repo_dir: Path, config_path: Path, *, force: bool, dry_run: bool) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "graph_specialisation_metrics.main_procedure",
        "run",
        "--config",
        str(config_path),
        "--steps",
        "all",
    ]
    if force:
        cmd.append("--force")
    if dry_run:
        cmd.append("--dry-run")
    proc = run_cmd(cmd, cwd=repo_dir, check=True)
    match = re.search(r"\[done\] main procedure artifacts: (.+)", proc.stdout)
    if not match:
        raise RuntimeError("main_procedure did not report an artifact root")
    return Path(match.group(1).strip())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def assess_completion(artifact_root: Path) -> dict[str, Any]:
    status_path = artifact_root / "metrics" / "main_status.json"
    adapter_path = artifact_root / "metrics" / "adapter_checks.json"
    statuses = load_json(status_path) if status_path.exists() else {}
    adapter_checks = load_json(adapter_path) if adapter_path.exists() else {}
    incomplete = {
        step: status
        for step, status in statuses.items()
        if isinstance(status, Mapping)
        and str(status.get("status"))
        in {"requires_trained_model_intervention_hooks", "waiting_for_training_artifacts", "waiting_for_official_grit_artifacts", "failed"}
    }
    figures = sorted(str(path) for path in (artifact_root / "figures").glob("*") if path.is_file())
    return {
        "artifact_root": str(artifact_root),
        "status_by_step": statuses,
        "adapter_checks": adapter_checks,
        "incomplete_steps": incomplete,
        "figures": figures,
        "complete": not incomplete and all(str(i) in statuses for i in range(6)),
    }


def write_methodology_fidelity_audit(
    path: Path,
    *,
    config: Mapping[str, Any],
    completion: Mapping[str, Any],
    include_local_references: bool,
) -> None:
    notes = []
    if config.get("seeds") != [0]:
        notes.append("Single-seed run requested; markdown default requires three seeds for paper claims.")
    else:
        notes.append("Single-seed run requested by user; this deliberately deviates from the markdown's 3-seed final protocol.")
    if not include_local_references:
        notes.append("GCN/GIN local validation reference omitted by default because only dense GRIT and 1-hop GRIT checkpoints were requested.")
    if completion.get("incomplete_steps"):
        notes.append("One or more required steps did not complete. Inspect completion.status_by_step for missing checkpoints, dependency errors, or failed intervention execution.")
    audit = {
        "markdown_steps_required": MARKDOWN_REQUIRED_STEPS,
        "config_steps": config.get("steps"),
        "config_primary_tau": config.get("primary_tau"),
        "config_far_thresholds": config.get("far_thresholds"),
        "config_perturbation": config.get("perturbation"),
        "models": config.get("models"),
        "completion": completion,
        "fidelity_notes": notes,
    }
    write_json(path, audit)


def copy_latest_outputs(drive_root: Path, artifact_root: Path) -> Path:
    latest = drive_root / "latest_outputs"
    if latest.exists():
        shutil.rmtree(latest)
    latest.mkdir(parents=True, exist_ok=True)
    for subdir in ["figures", "metrics"]:
        source = artifact_root / subdir
        if source.exists():
            shutil.copytree(source, latest / subdir)
    for name in ["manifest.json", "config.yaml"]:
        source = artifact_root / name
        if source.exists():
            shutil.copy2(source, latest / name)
    return latest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-url", default=PUBLIC_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--repo-dir", type=Path, default=Path(DEFAULT_REPO_DIR))
    parser.add_argument("--github-username", default=None)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--drive-root", type=Path, default=Path(DEFAULT_DRIVE_ROOT))
    parser.add_argument("--dense-drive-dir", type=Path, default=Path(DEFAULT_DENSE_DRIVE_DIR))
    parser.add_argument("--onehop-drive-dir", type=Path, default=Path(DEFAULT_ONEHOP_DRIVE_DIR))
    parser.add_argument("--single-seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Force rerun of main_procedure artifact generation.")
    parser.add_argument("--force-official-grit-reclone", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only run main_procedure discovery/status mode.")
    parser.add_argument("--allow-incomplete", action="store_true", help="Do not raise if checkpoints/dependencies are missing or a preflight is requested.")
    parser.add_argument("--include-local-references", action="store_true", help="Include placeholder GCN/GIN entries required by the markdown gate.")
    parser.add_argument("--skip-git", action="store_true", help="Use an already-cloned repo-dir.")
    argv = list(sys.argv[1:] if argv is None else argv)
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "-f" and i + 1 < len(argv) and "kernel-" in argv[i + 1] and argv[i + 1].endswith(".json"):
            print(f"[args] ignoring notebook launcher arguments: {argv[i:i+2]}", flush=True)
            i += 2
            continue
        cleaned.append(argv[i])
        i += 1
    return parser.parse_args(cleaned)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    mount_drive()
    args.drive_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_git:
        token = require_colab_token(args.secret_name)
        clone_or_update_repo(args.repo_url, args.branch, args.repo_dir, token, args.github_username)
    else:
        print(f"[git] using existing repo: {args.repo_dir}", flush=True)
    install_repo(args.repo_dir)

    dense_repo, onehop_repo, dense_cfg, onehop_cfg = prepare_grit_repos(
        args.repo_dir,
        args.drive_root,
        force=bool(args.force_official_grit_reclone),
    )
    dense_results = args.dense_drive_dir / "results"
    onehop_results = args.onehop_drive_dir / "results"
    dense_ckpt = choose_checkpoint(dense_results, "dense_grit")
    onehop_ckpt = choose_checkpoint(onehop_results, "grit_1hop")

    prepared = args.drive_root / "prepared_model_artifacts" / time.strftime("%Y%m%d_%H%M%S")
    dense_pointer = prepare_model_artifact(
        model="dense_grit",
        source_drive_dir=args.dense_drive_dir,
        config_path=dense_cfg,
        checkpoint_path=dense_ckpt,
        prepared_root=prepared / "dense_grit",
    )
    onehop_pointer = prepare_model_artifact(
        model="grit_1hop",
        source_drive_dir=args.onehop_drive_dir,
        config_path=onehop_cfg,
        checkpoint_path=onehop_ckpt,
        prepared_root=prepared / "grit_1hop",
    )

    cfg = build_zinc_config(
        artifact_root=args.drive_root,
        dense_prepared=prepared / "dense_grit",
        onehop_prepared=prepared / "grit_1hop",
        dense_dataset_dir=args.dense_drive_dir / "datasets",
        onehop_dataset_dir=args.onehop_drive_dir / "datasets",
        dense_repo=dense_repo,
        onehop_repo=onehop_repo,
        dense_cfg=dense_cfg,
        onehop_cfg=onehop_cfg,
        dense_ckpt=dense_ckpt,
        onehop_ckpt=onehop_ckpt,
        single_seed=args.single_seed,
        include_local_references=bool(args.include_local_references),
    )
    config_path = write_yaml(args.drive_root / "configs" / "zinc_main_procedure_colab.yaml", cfg)
    write_json(
        args.drive_root / "prepared_model_artifacts" / "latest_pointers.json",
        {"dense_grit": dense_pointer, "grit_1hop": onehop_pointer, "config_path": str(config_path)},
    )

    print(f"[config] wrote {config_path}", flush=True)
    artifact_root = run_main_procedure(args.repo_dir, config_path, force=bool(args.force), dry_run=bool(args.dry_run))
    completion = assess_completion(artifact_root)
    write_methodology_fidelity_audit(
        args.drive_root / "methodology_fidelity_audit.json",
        config=cfg,
        completion=completion,
        include_local_references=bool(args.include_local_references),
    )
    latest = copy_latest_outputs(args.drive_root, artifact_root)

    print("[completion]", json.dumps(completion, indent=2, sort_keys=True), flush=True)
    print(f"[done] artifacts: {artifact_root}", flush=True)
    print(f"[done] latest outputs: {latest}", flush=True)
    print(f"[done] fidelity audit: {args.drive_root / 'methodology_fidelity_audit.json'}", flush=True)

    if completion.get("incomplete_steps") and not args.allow_incomplete:
        raise SystemExit(
            "The current repo did not complete the full markdown procedure. "
            "Artifacts were written; inspect methodology_fidelity_audit.json for the missing or failed step. "
            "Pass --allow-incomplete for discovery/preflight only."
        )


if __name__ == "__main__":
    main()
