#!/usr/bin/env python3
"""Colab runner for ZINC RRWP structural probes on 1-hop GRIT variants.

This is a narrow runner for the current Step-4 structural/RRWP analysis:

* 1-hop GRIT with global RRWP;
* 1-hop GRIT with local-only RRWP;
* pair-RRWP, node-RRWP, and combined RRWP ablation probes;
* symbolic/structural carriage distance profiles;
* global-vs-local RRWP paired contrast figures.

It deliberately skips mediator patching and the full dense/GIN analysis, so it
can be used to iterate on the structural-RRWP question without rerunning the
complete dissertation procedure.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote


PUBLIC_REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
DEFAULT_BRANCH = "codex/cfim-grit-experiments"
DEFAULT_REPO_DIR = "/content/Graph-Specialisation-and-Metrics"
DEFAULT_SECRET_NAME = "dissertation_key"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/graph_specialisation_metrics/zinc_rrwp_structural_1hop_colab"
DEFAULT_ONEHOP_DRIVE_DIR = "/content/drive/MyDrive/grit_zinc_1hop"
DEFAULT_ONEHOP_LOCALRRWP_DRIVE_DIR = "/content/drive/MyDrive/grit_zinc_1hop_localrrwp"
DEFAULT_PYG_VERSION = "2.2.0"


class CommandError(RuntimeError):
    pass


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    safe_display: str | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    printable = safe_display or " ".join(map(str, cmd))
    print(f"[cmd] {printable}", flush=True)
    proc = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if check and proc.returncode != 0:
        raise CommandError(f"command failed with exit code {proc.returncode}: {printable}")
    return proc


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return path


def require_colab_token(secret_name: str) -> str:
    try:
        from google.colab import userdata
    except Exception as exc:  # pragma: no cover
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
    except Exception as exc:  # pragma: no cover
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
    if preflight is None or preflight.returncode != 0:
        raise RuntimeError(
            "GitHub authentication failed. The Colab secret must be a PAT with Contents=Read access "
            "to this repository, and the requested branch must be pushed."
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
        run_cmd(["git", "-C", str(repo_dir), "reset", "--hard", f"origin/{branch}"])
        run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])
        return

    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    run_cmd(
        ["git", "clone", "--branch", branch, "--single-branch", authed, str(repo_dir)],
        safe_display=f"git clone --branch {branch} <token-authenticated-url> {repo_dir}",
    )
    run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])


def import_base_runner(repo_dir: Path) -> Any:
    for path in [str(repo_dir), str(repo_dir / "src")]:
        if path not in sys.path:
            sys.path.insert(0, path)
    from experiments.methodology import colab_zinc_main_procedure as base

    return base


def assert_targeted_step4_support(repo_dir: Path) -> None:
    source = repo_dir / "src" / "graph_specialisation_metrics" / "grit_intervention_procedure.py"
    text = source.read_text(encoding="utf-8") if source.exists() else ""
    required = "run_mediator_patching"
    if required not in text:
        raise RuntimeError(
            "The cloned GitHub branch does not contain the Step-4 run_mediator_patching skip switch. "
            "Push the current methodology branch before running this targeted Colab file, otherwise it "
            "would fall back to the full mediator-patching workload."
        )


def build_rrwp_structural_config(
    *,
    artifact_root: Path,
    onehop_prepared: Path,
    localrrwp_prepared: Path,
    onehop_dataset_dir: Path,
    localrrwp_dataset_dir: Path,
    onehop_repo: Path,
    localrrwp_repo: Path,
    onehop_cfg: Path,
    localrrwp_cfg: Path,
    onehop_ckpt: Path,
    localrrwp_ckpt: Path,
    seed: int,
    symbolic_sample_graphs: int,
    rrwp_sample_graphs: int,
    global_channel_sample_graphs: int,
    graph_metric_cut_pairs: int,
    dpi: int,
) -> dict[str, Any]:
    bins = [
        {"label": "d=2-3", "min": 2, "max": 3},
        {"label": "d=4-6", "min": 4, "max": 6},
        {"label": "d=7-10", "min": 7, "max": 10},
        {"label": "d=11-14", "min": 11, "max": 14},
        {"label": "d>14", "min": 15, "max": None},
    ]
    return {
        "artifact_root": str(artifact_root / "artifacts"),
        "dataset": {"name": "ZINC", "split": "official_subset", "task": "molecular_regression"},
        "seeds": [int(seed)],
        "primary_tau": 3,
        "far_thresholds": [2, 3, 4],
        "perturbation": {
            "carriage_primary": "integrated_gradients",
            "ig_baseline": "mean_node_embedding",
            "ig_steps": 32,
            "baseline_sample_graphs": 32,
            "swap_partners": 4,
            "batched_vjp": True,
            "carriage_readout_ig": False,
            "swap_partner_policy": "different_type",
        },
        "models": {
            "grit_1hop": {
                "adapter": "official_grit",
                "variant": "1hop",
                "role": "parameter_matched_control_global_rrwp",
                "official_repo": "https://github.com/LiamMa/GRIT",
                "repo_path": str(onehop_repo),
                "artifact_root": str(onehop_prepared),
                "dataset_dir": str(onehop_dataset_dir),
                "config_path": str(onehop_cfg),
                "checkpoint_path": str(onehop_ckpt),
            },
            "grit_1hop_localrrwp": {
                "adapter": "official_grit",
                "variant": "1hop_localrrwp",
                "role": "strict_local_pe_parameter_matched_control",
                "official_repo": "https://github.com/LiamMa/GRIT",
                "repo_path": str(localrrwp_repo),
                "artifact_root": str(localrrwp_prepared),
                "dataset_dir": str(localrrwp_dataset_dir),
                "config_path": str(localrrwp_cfg),
                "checkpoint_path": str(localrrwp_ckpt),
            },
        },
        "runtime": {"skip_failed_optional_adapters": True},
        "steps": {
            "4": {
                "name": "mediator_patching",
                "sample_graphs": 1,
                "run_mediator_patching": False,
                "run_analytic_patching_check": False,
                "run_clamp_negative_control": False,
                "run_clamp_mode_comparison": False,
                "run_symbolic_structural_carriage": True,
                "symbolic_structural_sample_graphs": int(symbolic_sample_graphs),
                "symbolic_structural_max_sources": "all",
                "symbolic_structural_min_distance": 1,
                "symbolic_structural_rrwp_channel_start": 2,
                "symbolic_structural_rrwp_replacement": "zero",
                "run_rrwp_distance_ablation": True,
                "rrwp_ablation_sample_graphs": int(rrwp_sample_graphs),
                "rrwp_ablation_channel_start": 2,
                "rrwp_ablation_replacement": "zero",
                "rrwp_ablation_types": ["node", "pair", "both"],
                "rrwp_distance_ablation_types": ["pair"],
                "global_rrwp_channel_ablation_types": ["node", "pair", "both"],
                "run_global_rrwp_channel_ablation": True,
                "global_rrwp_channel_ablation_sample_graphs": int(global_channel_sample_graphs),
                "rrwp_ablation_distance_bins": bins,
                "rrwp_contrast_global_model": "grit_1hop",
                "rrwp_contrast_local_model": "grit_1hop_localrrwp",
                "rrwp_graph_metric_cut_pairs": int(graph_metric_cut_pairs),
            }
        },
        "figures": {"dpi": int(dpi)},
        "colab_notes": {
            "targeted_runner": "zinc_rrwp_structural_1hop",
            "full_methodology_runner": "experiments/methodology/colab_zinc_main_procedure.py",
            "mediator_patching_skipped": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-url", default=PUBLIC_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--repo-dir", type=Path, default=Path(DEFAULT_REPO_DIR))
    parser.add_argument("--github-username", default=None)
    parser.add_argument("--secret-name", default=DEFAULT_SECRET_NAME)
    parser.add_argument("--drive-root", type=Path, default=Path(DEFAULT_DRIVE_ROOT))
    parser.add_argument("--onehop-drive-dir", type=Path, default=Path(DEFAULT_ONEHOP_DRIVE_DIR))
    parser.add_argument("--onehop-localrrwp-drive-dir", type=Path, default=Path(DEFAULT_ONEHOP_LOCALRRWP_DRIVE_DIR))
    parser.add_argument("--prepared-id", default="zinc_rrwp_structural_1hop_v1")
    parser.add_argument("--single-seed", type=int, default=0)
    parser.add_argument("--pyg-version", default=DEFAULT_PYG_VERSION)
    parser.add_argument("--analysis-preset", default="high", choices=["quick", "pilot", "medium", "high", "full"])
    parser.add_argument("--steps", default="4")
    parser.add_argument("--symbolic-sample-graphs", type=int, default=16)
    parser.add_argument("--rrwp-sample-graphs", type=int, default=64)
    parser.add_argument("--global-channel-sample-graphs", type=int, default=96)
    parser.add_argument("--graph-metric-cut-pairs", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-git", action="store_true", help="Delete and reclone the methodology repo.")
    parser.add_argument("--force-official-repos", action="store_true", help="Delete and reclone official GRIT checkouts.")
    parser.add_argument("--skip-git", action="store_true", help="Use the existing repo-dir checkout without fetch/reset.")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip dependency installation.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    if argv is None:
        argv = [arg for arg in sys.argv[1:] if not arg.endswith(".json") and arg != "-f"]
    args = parse_args(argv)

    mount_drive()
    args.drive_root.mkdir(parents=True, exist_ok=True)

    token = require_colab_token(args.secret_name)
    if args.force_git and args.repo_dir.exists():
        shutil.rmtree(args.repo_dir)
    if not args.skip_git:
        clone_or_update_repo(args.repo_url, args.branch, args.repo_dir, token, args.github_username)

    assert_targeted_step4_support(args.repo_dir)
    base = import_base_runner(args.repo_dir)
    shim_dir = base.write_py312_compat_shim(args.drive_root)
    env = base.env_with_py312_compat(shim_dir)
    if not args.skip_install:
        base.install_repo(args.repo_dir, pyg_version=args.pyg_version)

    _, onehop_repo, localrrwp_repo, _, onehop_cfg, localrrwp_cfg = base.prepare_grit_repos(
        args.repo_dir,
        args.drive_root,
        force=bool(args.force_official_repos),
        include_localrrwp=True,
    )
    if localrrwp_repo is None or localrrwp_cfg is None:
        raise RuntimeError("local-RRWP official GRIT checkout/config was not prepared")

    onehop_ckpt = base.choose_checkpoint(args.onehop_drive_dir / "results", "grit_1hop")
    localrrwp_ckpt = base.choose_checkpoint(args.onehop_localrrwp_drive_dir / "results", "grit_1hop_localrrwp")

    prepared_root = args.drive_root / "prepared_model_artifacts" / args.prepared_id
    onehop_prepared = prepared_root / "grit_1hop"
    localrrwp_prepared = prepared_root / "grit_1hop_localrrwp"
    print(f"[prepared] {prepared_root}", flush=True)

    base.prepare_model_artifact(
        model="grit_1hop",
        source_drive_dir=args.onehop_drive_dir,
        config_path=onehop_cfg,
        checkpoint_path=onehop_ckpt,
        prepared_root=onehop_prepared,
    )
    base.prepare_model_artifact(
        model="grit_1hop_localrrwp",
        source_drive_dir=args.onehop_localrrwp_drive_dir,
        config_path=localrrwp_cfg,
        checkpoint_path=localrrwp_ckpt,
        prepared_root=localrrwp_prepared,
    )

    artifact_root = args.drive_root / "runs" / args.prepared_id
    config = build_rrwp_structural_config(
        artifact_root=artifact_root,
        onehop_prepared=onehop_prepared,
        localrrwp_prepared=localrrwp_prepared,
        onehop_dataset_dir=args.onehop_drive_dir / "datasets",
        localrrwp_dataset_dir=args.onehop_localrrwp_drive_dir / "datasets",
        onehop_repo=onehop_repo,
        localrrwp_repo=localrrwp_repo,
        onehop_cfg=onehop_cfg,
        localrrwp_cfg=localrrwp_cfg,
        onehop_ckpt=onehop_ckpt,
        localrrwp_ckpt=localrrwp_ckpt,
        seed=args.single_seed,
        symbolic_sample_graphs=args.symbolic_sample_graphs,
        rrwp_sample_graphs=args.rrwp_sample_graphs,
        global_channel_sample_graphs=args.global_channel_sample_graphs,
        graph_metric_cut_pairs=args.graph_metric_cut_pairs,
        dpi=args.dpi,
    )
    config_path = args.drive_root / "configs" / f"{args.prepared_id}.yaml"
    base.write_yaml(config_path, config)
    write_json(
        args.drive_root / "rrwp_structural_run_inputs.json",
        {
            "config_path": str(config_path),
            "onehop_checkpoint": str(onehop_ckpt),
            "localrrwp_checkpoint": str(localrrwp_ckpt),
            "prepared_root": str(prepared_root),
            "artifact_root": str(artifact_root),
            "branch": args.branch,
            "analysis_preset": args.analysis_preset,
            "steps": args.steps,
        },
    )
    print(f"[config] {config_path}", flush=True)

    produced_artifact_root = base.run_main_procedure(
        args.repo_dir,
        config_path,
        steps=args.steps,
        force=bool(args.force),
        analysis_preset=args.analysis_preset,
        fast_dev_run=False,
        dry_run=bool(args.dry_run),
        env=env,
    )
    latest = base.copy_latest_outputs(args.drive_root, produced_artifact_root)
    print(f"[done] artifacts: {produced_artifact_root}", flush=True)
    print(f"[done] latest outputs: {latest}", flush=True)
    print("[done] key figures:", flush=True)
    for rel in [
        "figures/step4_global_to_local_rrwp_ablation_main.png",
        "figures/step4_global_to_local_rrwp_ablation.png",
        "figures/step4_rrwp_distance_bin_ablation.png",
        "figures/step4_symbolic_structural_carriage_absolute_by_distance.png",
        "figures/step4_symbolic_structural_carriage_by_distance.png",
        "figures/step4_symbolic_global_vs_local_rrwp_contrast.png",
        "figures/step4_global_vs_local_rrwp_paired_contrast.png",
    ]:
        path = latest / rel
        print(f"  [{'ok' if path.exists() else 'missing'}] {path}", flush=True)


if __name__ == "__main__":
    main()
