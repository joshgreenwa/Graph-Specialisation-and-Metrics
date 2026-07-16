#!/usr/bin/env python3
"""Unified Colab runner for GRIT on Peptides-func and Peptides-struct.

This file is designed to be copied into a fresh Google Colab runtime and run as
one standalone script. It bootstraps the project repo from GitHub, mounts Drive,
installs the GRIT runtime, clones the official LiamMa/GRIT repo, applies only
runtime/data compatibility patches, and launches one of four training runs:

  * Peptides-func, official dense GRIT;
  * Peptides-func, parameter-matched 1-hop masked GRIT;
  * Peptides-struct, official dense GRIT;
  * Peptides-struct, parameter-matched 1-hop masked GRIT.

The dense variants use the official Peptides GRIT RRWP configs. The 1-hop
variants copy those official configs and change only:

  gt.attn.full_attn = False
  gt.attn.sparsity = one_hop

The model-code support for the 1-hop path is the same masked-official GRIT patch
used by the current ZINC 1-hop runner in this repo.
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
from urllib.parse import quote


PROJECT_REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
PROJECT_BRANCH = "codex/cfim-grit-experiments"
PROJECT_REPO_DIR = Path("/content/Graph-Specialisation-and-Metrics")
SECRET_NAME = "dissertation_key"

OFFICIAL_GRIT_REPO = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"

TASKS: dict[str, dict[str, Any]] = {
    "func": {
        "label": "peptides_func",
        "dataset_name": "peptides-functional",
        "official_cfg": "configs/GRIT/peptides-func-GRIT-RRWP.yaml",
        "onehop_cfg": "configs/GRIT/peptides-func-GRIT-RRWP-1hop.yaml",
        "metric_best": "ap",
        # The pinned official Peptides-func config sets metric_best: ap and
        # leaves metric_agg to GraphGym defaults; do not require it here.
        "metric_agg": None,
        "task_type": "classification_multilabel",
        "loss_fun": "cross_entropy",
        "rrwp_ksteps": 17,
        "layers": 4,
        "heads": 4,
        "hidden": 96,
        "dropout": 0.0,
        "attn_dropout": 0.5,
        "layers_post_mp": 1,
        "batch_size": 16,
        "max_epoch": 200,
        "warmup": 10,
        "base_lr": 0.0003,
        "graph_pooling": "mean",
    },
    "struct": {
        "label": "peptides_struct",
        "dataset_name": "peptides-structural",
        "official_cfg": "configs/GRIT/peptides-struct-GRIT-RRWP.yaml",
        "onehop_cfg": "configs/GRIT/peptides-struct-GRIT-RRWP-1hop.yaml",
        "metric_best": "mae",
        "metric_agg": "argmin",
        "task_type": "regression",
        "loss_fun": "l1",
        "rrwp_ksteps": 24,
        "layers": 4,
        "heads": 8,
        "hidden": 96,
        "dropout": 0.05,
        "attn_dropout": 0.2,
        "layers_post_mp": 2,
        "batch_size": 16,
        "max_epoch": 200,
        "warmup": 10,
        "base_lr": 0.0003,
        "graph_pooling": "mean",
    },
}


class CommandError(RuntimeError):
    pass


def log(message: str) -> None:
    print(message, flush=True)


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
    safe: str | None = None,
) -> subprocess.CompletedProcess:
    printable = safe or " ".join(map(str, cmd))
    log(f"[cmd] {printable}")
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env else None,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise CommandError(f"command failed with exit code {proc.returncode}: {printable}")
    return proc


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def get_secret(name: str) -> str | None:
    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        if value:
            return str(value).strip()
    except Exception:
        pass
    value = os.environ.get(name)
    return value.strip() if value else None


def auth_url(repo_url: str, token: str | None) -> str:
    if not token or not repo_url.startswith("https://github.com/"):
        return repo_url
    suffix = repo_url.removeprefix("https://github.com/")
    return f"https://x-access-token:{quote(token, safe='')}@github.com/{suffix}"


def premount_drive(mount_point: Path) -> None:
    if not in_colab():
        return
    try:
        from google.colab import drive  # type: ignore

        log(f"[drive] Mounting Google Drive at {mount_point} before repository setup ...")
        drive.mount(str(mount_point), force_remount=False)
    except Exception as exc:
        raise RuntimeError(f"failed to mount Google Drive at {mount_point}") from exc


def strip_colab_kernel_args(argv: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "-f" and i + 1 < len(argv) and "kernel-" in argv[i + 1] and argv[i + 1].endswith(".json"):
            log(f"[args] ignoring notebook launcher arguments: {arg} {argv[i + 1]}")
            i += 2
            continue
        if arg.startswith("-f=") and "kernel-" in arg and arg.endswith(".json"):
            log(f"[args] ignoring notebook launcher argument: {arg}")
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return cleaned


def bootstrap_project_repo(repo_url: str, branch: str, repo_dir: Path, secret_name: str, *, skip_git: bool) -> Path:
    local_root = Path(__file__).resolve().parents[3] if "__file__" in globals() else None
    if local_root and (local_root / "src" / "graph_specialisation_metrics").exists():
        for path in [str(local_root), str(local_root / "src")]:
            if path not in sys.path:
                sys.path.insert(0, path)
        log(f"[project] using local project repo: {local_root}")
        return local_root

    if skip_git:
        for path in [str(repo_dir), str(repo_dir / "src")]:
            if path not in sys.path:
                sys.path.insert(0, path)
        log(f"[project] using existing project repo without git refresh: {repo_dir}")
        return repo_dir

    if not in_colab():
        raise RuntimeError("project repo bootstrap needs Colab, --skip-project-git, or in-repo execution")

    token = get_secret(secret_name)
    if not token:
        raise RuntimeError(f"missing Colab secret {secret_name!r}")
    log(f"[auth] secret {secret_name!r} loaded; length={len(token)}, prefix={token[:10]!r}")
    authed = auth_url(repo_url, token)

    if (repo_dir / ".git").exists():
        run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", authed], safe=f"git -C {repo_dir} remote set-url origin <token-authenticated-url>")
        run_cmd(["git", "-C", str(repo_dir), "fetch", "origin", branch])
        run_cmd(["git", "-C", str(repo_dir), "checkout", branch])
        run_cmd(["git", "-C", str(repo_dir), "reset", "--hard", f"origin/{branch}"])
        run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])
    else:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        run_cmd(["git", "clone", "--branch", branch, "--single-branch", authed, str(repo_dir)], safe=f"git clone --branch {branch} <token-authenticated-url> {repo_dir}")
        run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])

    for path in [str(repo_dir), str(repo_dir / "src")]:
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_dir


def split_bootstrap_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-repo-url", default=PROJECT_REPO_URL)
    parser.add_argument("--project-branch", default=PROJECT_BRANCH)
    parser.add_argument("--project-repo-dir", type=Path, default=PROJECT_REPO_DIR)
    parser.add_argument("--secret-name", default=SECRET_NAME)
    parser.add_argument("--skip-project-git", action="store_true")
    parser.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    raw = strip_colab_kernel_args(list(sys.argv[1:] if argv is None else argv))
    args, rest = parser.parse_known_args(raw)
    return args, rest


def default_drive_dir(task: str, variant: str) -> Path:
    suffix = "official" if variant == "official" else "1hop"
    return Path(f"/content/drive/MyDrive/grit_{TASKS[task]['label']}_{suffix}")


def default_grit_repo_dir(task: str, variant: str) -> Path:
    suffix = "official" if variant == "official" else "1hop"
    return Path(f"/content/GRIT_{TASKS[task]['label']}_{suffix}")


def default_name_tag(task: str, variant: str, seed: int) -> str:
    variant_tag = "official" if variant == "official" else "1hop"
    return f"ColabDrive.{variant_tag}.GRITwRRWP.{TASKS[task]['label']}.s{seed}"


def nested_get(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        current = current[key]
    return current


def nested_set(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = payload
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def semantically_equal(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        try:
            return abs(float(actual) - float(expected)) <= max(1e-12, abs(float(expected)) * 1e-8)
        except Exception:
            return False
    return False


def expected_config_values(task: str, variant: str) -> dict[tuple[str, ...], Any]:
    info = TASKS[task]
    values: dict[tuple[str, ...], Any] = {
        ("metric_best",): info["metric_best"],
        ("dataset", "format"): "OGB",
        ("dataset", "name"): info["dataset_name"],
        ("dataset", "task"): "graph",
        ("dataset", "task_type"): info["task_type"],
        ("dataset", "transductive"): False,
        ("dataset", "node_encoder"): True,
        ("dataset", "node_encoder_name"): "Atom",
        ("dataset", "node_encoder_bn"): False,
        ("dataset", "edge_encoder"): True,
        ("dataset", "edge_encoder_name"): "Bond",
        ("dataset", "edge_encoder_bn"): False,
        ("posenc_RRWP", "enable"): True,
        ("posenc_RRWP", "ksteps"): info["rrwp_ksteps"],
        ("posenc_RRWP", "add_identity"): True,
        ("posenc_RRWP", "add_node_attr"): False,
        ("train", "mode"): "custom",
        ("train", "batch_size"): info["batch_size"],
        ("model", "type"): "GritTransformer",
        ("model", "loss_fun"): info["loss_fun"],
        ("model", "graph_pooling"): info["graph_pooling"],
        ("gt", "layer_type"): "GritTransformer",
        ("gt", "layers"): info["layers"],
        ("gt", "n_heads"): info["heads"],
        ("gt", "dim_hidden"): info["hidden"],
        ("gt", "dropout"): info["dropout"],
        ("gt", "attn_dropout"): info["attn_dropout"],
        ("gt", "layer_norm"): False,
        ("gt", "batch_norm"): True,
        ("gt", "attn", "clamp"): 5.0,
        ("gt", "attn", "act"): "relu",
        ("gt", "attn", "edge_enhance"): True,
        ("gt", "attn", "O_e"): True,
        ("gt", "attn", "norm_e"): True,
        ("gt", "attn", "signed_sqrt"): True,
        ("gnn", "head"): "default",
        ("gnn", "layers_pre_mp"): 0,
        ("gnn", "layers_post_mp"): info["layers_post_mp"],
        ("gnn", "dim_inner"): info["hidden"],
        ("gnn", "batchnorm"): True,
        ("gnn", "act"): "relu",
        ("gnn", "dropout"): 0.0,
        ("optim", "clip_grad_norm"): True,
        ("optim", "optimizer"): "adamW",
        ("optim", "weight_decay"): 0.0,
        ("optim", "base_lr"): info["base_lr"],
        ("optim", "max_epoch"): info["max_epoch"],
        ("optim", "scheduler"): "cosine_with_warmup",
        ("optim", "num_warmup_epochs"): info["warmup"],
        ("gt", "attn", "full_attn"): variant == "official",
    }
    if info.get("metric_agg") is not None:
        values[("metric_agg",)] = info["metric_agg"]
    if variant == "1hop":
        values[("gt", "attn", "sparsity")] = "one_hop"
    return values


def make_onehop_config(repo_dir: Path, task: str) -> Path:
    import yaml

    info = TASKS[task]
    official = repo_dir / info["official_cfg"]
    onehop = repo_dir / info["onehop_cfg"]
    if not official.exists():
        raise FileNotFoundError(f"official Peptides config is missing: {official}")
    cfg = yaml.safe_load(official.read_text(encoding="utf-8"))
    nested_set(cfg, ("gt", "attn", "full_attn"), False)
    nested_set(cfg, ("gt", "attn", "sparsity"), "one_hop")
    nested_set(cfg, ("mlflow", "name"), onehop.stem)
    nested_set(cfg, ("wandb", "use"), False)
    header = (
        f"# Parameter-matched 1-hop sparse-control variant of {info['official_cfg']}.\n"
        "# Generated by experiments/peptides/training/grit_peptides_colab.py.\n"
        "# Only gt.attn.full_attn=False and gt.attn.sparsity=one_hop change the scientific model path.\n"
    )
    onehop.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log(f"[config] wrote parameter-matched 1-hop config: {onehop}")
    return onehop


def validate_config(repo_dir: Path, task: str, variant: str, *, allow_drift: bool) -> Path:
    import yaml

    info = TASKS[task]
    cfg_rel = info["official_cfg"] if variant == "official" else info["onehop_cfg"]
    cfg_path = repo_dir / cfg_rel
    if not cfg_path.exists():
        raise FileNotFoundError(f"GRIT config not found: {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for key_path, expected in expected_config_values(task, variant).items():
        try:
            actual = nested_get(cfg, key_path)
        except Exception:
            errors.append(f"missing {'.'.join(key_path)}; expected {expected!r}")
            continue
        if not semantically_equal(actual, expected):
            errors.append(f"{'.'.join(key_path)}={actual!r}; expected {expected!r}")
    if errors:
        msg = f"{task}/{variant} config does not match expected official setup:\n" + "\n".join(f"  - {e}" for e in errors)
        if not allow_drift:
            raise RuntimeError(msg + "\nPass --allow-upstream-config-drift to run anyway.")
        log("[config-warning] " + msg)

    info = TASKS[task]
    log(
        "[config] validated: "
        f"task={task} ({info['dataset_name']}), variant={variant}, RRWP-{info['rrwp_ksteps']}, "
        f"{info['layers']} layers, hidden={info['hidden']}, heads={info['heads']}, "
        f"batch={info['batch_size']}, max_epoch={info['max_epoch']}, "
        f"scheduler=cosine_with_warmup, warmup={info['warmup']}."
    )
    if variant == "1hop":
        log("[control] 1-hop intervention: attention support is molecular self/bonds; model dimensions/config remain official.")
    else:
        log("[official] dense variant uses the unmodified official LiamMa/GRIT Peptides config.")
    return cfg_path


def build_train_command(args: argparse.Namespace, cfg_path: Path) -> list[str]:
    results_dir = args.drive_dir / "results"
    dataset_dir = args.drive_dir / "datasets"
    results_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = not args.checkpoint_every_epoch
    ckpt_clean = False if (args.keep_all_checkpoints or args.checkpoint_every_epoch or args.guaranteed_checkpoints) else True
    cmd = [
        sys.executable,
        "-u",
        "main.py",
        "--cfg",
        str(cfg_path),
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
    cmd.extend(args.cfg_overrides)
    return cmd


def checkpoint_inventory(drive_dir: Path, wrapper_log: Path, seed: int) -> Path:
    patterns = ("*.ckpt", "*.pt", "*.pth")
    result_root = drive_dir / "results"
    candidates = sorted(
        {path for pattern in patterns for path in result_root.rglob(pattern) if path.is_file()},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    sidecars: list[dict[str, Any]] = []
    for sidecar in result_root.rglob("best_epoch.txt") if result_root.exists() else []:
        try:
            sidecars.append({"path": str(sidecar), "best_epoch": sidecar.read_text(encoding="utf-8").strip()})
        except Exception:
            pass
    payload = {
        "seed": seed,
        "drive_dir": str(drive_dir),
        "wrapper_log": str(wrapper_log),
        "latest_checkpoint_by_mtime": str(candidates[0]) if candidates else None,
        "checkpoint_candidates": [
            {
                "path": str(path),
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
                "bytes": path.stat().st_size,
            }
            for path in candidates
        ],
        "best_epoch_sidecars": sidecars,
    }
    out = drive_dir / f"checkpoint_inventory_seed{seed}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (drive_dir / "latest_checkpoint_inventory.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log(f"[checkpoint-inventory] latest={payload['latest_checkpoint_by_mtime']}")
    log(f"[checkpoint-inventory] wrote: {out}")
    return out


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train official dense or 1-hop masked GRIT on Peptides-func/Peptides-struct in Colab.",
    )
    parser.add_argument("--task", choices=sorted(TASKS), default="func")
    parser.add_argument("--variant", choices=["official", "1hop"], default="official")
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--repo-dir", type=Path, default=None)
    parser.add_argument("--repo-url", default=OFFICIAL_GRIT_REPO)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--commit", default=OFFICIAL_GRIT_COMMIT)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--name-tag", default=None)
    parser.add_argument("--max-epoch", type=int, default=None, help="Default is the official config value for the selected task.")
    parser.add_argument("--ckpt-period", type=int, default=100)
    parser.add_argument("--recovery-ckpt-period", type=int, default=100)
    parser.add_argument("--rrwp-stream-chunk-size", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--pyg-version", default="2.2.0")
    parser.add_argument("--official-torch112", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--force-fresh-repo", action="store_true")
    parser.add_argument("--allow-upstream-config-drift", action="store_true")
    parser.add_argument("--expected-params", type=int, default=None)
    parser.add_argument("--allow-param-count-drift", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--console-verbosity", choices=["compact", "standard", "full"], default="full")
    parser.add_argument("--console-epoch-period", type=int, default=1)
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument("--auto-resume", action="store_true", default=True)
    parser.add_argument("--no-auto-resume", action="store_false", dest="auto_resume")
    parser.add_argument("--keep-all-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-every-epoch", action="store_true")
    parser.add_argument("--guaranteed-checkpoints", action="store_true", default=True)
    parser.add_argument("--no-guaranteed-checkpoints", action="store_false", dest="guaranteed_checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("cfg_overrides", nargs=argparse.REMAINDER, help="Optional trailing GraphGym config overrides.")
    args = parser.parse_args(strip_colab_kernel_args(list(sys.argv[1:] if argv is None else argv)))
    if args.cfg_overrides and args.cfg_overrides[0] == "--":
        args.cfg_overrides = args.cfg_overrides[1:]
    if args.drive_dir is None:
        args.drive_dir = default_drive_dir(args.task, args.variant)
    if args.repo_dir is None:
        args.repo_dir = default_grit_repo_dir(args.task, args.variant)
    if args.name_tag is None:
        args.name_tag = default_name_tag(args.task, args.variant, args.seed)
    if args.max_epoch is None:
        args.max_epoch = int(TASKS[args.task]["max_epoch"])
    return args


def main(argv: Sequence[str] | None = None) -> None:
    bootstrap, rest = split_bootstrap_args(argv)
    premount_drive(bootstrap.drive_mount)
    project_root = bootstrap_project_repo(
        bootstrap.project_repo_url,
        bootstrap.project_branch,
        bootstrap.project_repo_dir,
        bootstrap.secret_name,
        skip_git=bootstrap.skip_project_git,
    )
    log(f"[project] ready: {project_root}")

    from experiments.peptides_struct.training import grit_peptides_struct_common as peptides_common
    from experiments.zinc.training import grit_zinc_core as base
    from experiments.zinc.training import grit_zinc_1hop_core as onehop_base

    args = parse_args(rest)
    args.drive_dir.mkdir(parents=True, exist_ok=True)

    compat_shim_dir = None
    if sys.version_info >= (3, 12):
        compat_shim_dir = base.write_py312_compat_shim(args.drive_dir)

    if not args.skip_install:
        base.install_dependencies(args)
        peptides_common.install_peptides_dependencies(base)
    else:
        log("[deps] skipping dependency installation (--skip-install).")

    commit = base.clone_or_update_repo(args.repo_dir, args.repo_url, args.branch, args.commit or None, args.force_fresh_repo)
    base.run_cmd(["git", "reset", "--hard", commit], cwd=args.repo_dir)

    peptides_common.apply_peptides_dataset_compat_patch(base, args.repo_dir)
    peptides_common.apply_peptides_streaming_rrwp_patch(base, args.repo_dir)

    if args.variant == "1hop":
        onehop_base.apply_parameter_matched_onehop_patch(args.repo_dir, args.drive_dir)
        make_onehop_config(args.repo_dir, args.task)
        base.verify_recovery_checkpoint_patch(args.repo_dir)
    elif args.guaranteed_checkpoints:
        base.apply_recovery_checkpoint_patch(args.repo_dir)
        base.verify_recovery_checkpoint_patch(args.repo_dir)

    base.install_grit_editable(args.repo_dir)
    cfg_path = validate_config(args.repo_dir, args.task, args.variant, allow_drift=args.allow_upstream_config_drift)
    base.print_environment_summary(args.drive_dir, args.repo_dir, commit)
    log(f"[run] task={args.task} variant={args.variant} seed={args.seed} tag={args.name_tag}")
    log(f"[run] drive_dir={args.drive_dir}")
    log(f"[run] repo_dir={args.repo_dir}")

    cmd = build_train_command(args, cfg_path)
    if args.dry_run:
        log("[dry-run] training command:")
        log(" ".join(map(str, cmd)))
        return

    wrapper_log = args.drive_dir / "wrapper_logs" / f"grit_{TASKS[args.task]['label']}_{args.variant}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    train_env = base.env_with_py312_compat(compat_shim_dir)
    train_env["GRIT_PE_STREAM_CHUNK_SIZE"] = str(max(1, int(args.rrwp_stream_chunk_size)))
    if args.guaranteed_checkpoints:
        recovery_dir = (
            args.drive_dir
            / "results"
            / "_recovery_checkpoints"
            / f"{args.task}_{args.variant}_seed{args.seed}_{base.safe_path_fragment(args.name_tag)}"
        )
        recovery_dir.mkdir(parents=True, exist_ok=True)
        train_env["GRIT_FORCE_RECOVERY_CKPT"] = "1"
        train_env["GRIT_RECOVERY_CKPT_PERIOD"] = str(max(0, int(args.recovery_ckpt_period)))
        train_env["GRIT_RECOVERY_CKPT_DIR"] = str(recovery_dir)
        train_env["GRIT_SAVE_FIRST_RECOVERY_CKPT"] = "1"
        log("[checkpoint] recovery checkpoints enabled")
        log(f"[checkpoint] stable best path: {recovery_dir / 'best.ckpt'}")
        log(f"[checkpoint] first completed epoch after resume: {recovery_dir / 'first_after_resume.ckpt'}")
        log(f"[checkpoint] numbered snapshots every {args.recovery_ckpt_period} epochs")

    rc = base.run_streaming_to_console_and_log(
        cmd,
        cwd=args.repo_dir,
        log_file=wrapper_log,
        env=train_env,
        expected_params=args.expected_params,
        allow_param_count_drift=args.allow_param_count_drift,
        console_verbosity=args.console_verbosity,
        console_epoch_period=args.console_epoch_period,
    )
    checkpoint_inventory(args.drive_dir, wrapper_log, args.seed)
    if rc != 0:
        raise SystemExit(rc)
    log("[done] training completed.")
    log(f"[done] full raw log: {wrapper_log}")
    log(f"[done] Drive results: {args.drive_dir / 'results'}")


if __name__ == "__main__":
    main([
        "--task", "func",
        "--variant", "official",
        "--seed", "41",
        "--name-tag", "ColabDrive.official.GRITwRRWP.peptides_func.s41",
        "--console-verbosity", "full",
        "--console-epoch-period", "1",
        "--rrwp-stream-chunk-size", "32",
        "--force-fresh-repo",
    ])
