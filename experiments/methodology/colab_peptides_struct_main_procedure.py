#!/usr/bin/env python3
"""Standalone Google Colab runner for Peptides-struct methodology analysis.

This mirrors ``colab_zinc_main_procedure.py`` but is intentionally scoped to
the seed-41 Peptides-struct dense GRIT and 1-hop global-RRWP GRIT checkpoints.
GIN and local-RRWP 1-hop controls are not included until those Peptides models
exist.

Default Colab run:

    main([
        "--steps", "7",
        "--analysis-preset", "medium",
    ])

The runner clones the current methodology repo branch using the Colab secret
``dissertation_key``, prepares official LiamMa/GRIT analysis checkouts, applies
the same Peptides loader/RRWP compatibility patches used by the training
notebooks, discovers the seed-41 Drive checkpoints by name tag, writes a
Peptides-specific config, and invokes ``graph_specialisation_metrics``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
DEFAULT_PYG_VERSION = "2.2.0"

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/graph_specialisation_metrics/peptides_struct_main_procedure_colab"
DEFAULT_DENSE_DRIVE_DIR = "/content/drive/MyDrive/grit_peptides_struct_official"
DEFAULT_ONEHOP_DRIVE_DIR = "/content/drive/MyDrive/grit_peptides_struct_1hop"
DEFAULT_DENSE_REPO_DIR = "/content/GRIT_peptides_struct_dense_analysis"
DEFAULT_ONEHOP_REPO_DIR = "/content/GRIT_peptides_struct_1hop_analysis"

DEFAULT_DENSE_NAME_TAG = "ColabDrive.official.GRITwRRWP.peptides_struct.s41"
DEFAULT_ONEHOP_NAME_TAG = "ColabDrive.1hop.GRITwRRWP.peptides_struct.s41"

OFFICIAL_GRIT_REPO = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
DENSE_CFG_REL = "configs/GRIT/peptides-struct-GRIT-RRWP.yaml"
ONEHOP_CFG_REL = "configs/GRIT/peptides-struct-GRIT-RRWP-1hop.yaml"


class CommandError(RuntimeError):
    pass


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | str | None = None,
    safe_display: str | None = None,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    stream: bool = False,
) -> subprocess.CompletedProcess[str]:
    printable = safe_display or " ".join(map(str, cmd))
    print(f"[cmd] {printable}", flush=True)
    if stream:
        proc = subprocess.Popen(
            list(map(str, cmd)),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert proc.stdout is not None
        out_lines: list[str] = []
        for line in proc.stdout:
            out_lines.append(line)
            print(line, end="", flush=True)
        code = proc.wait()
        stdout = "".join(out_lines)
        result = subprocess.CompletedProcess(list(map(str, cmd)), code, stdout=stdout, stderr=None)
    else:
        result = subprocess.run(
            list(map(str, cmd)),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if check and result.returncode != 0:
        raise CommandError(f"command failed with exit code {result.returncode}: {printable}")
    return result


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
    suffix = public_url.removeprefix("https://github.com/")
    token_q = quote(token, safe="")
    user = quote(username or repo_owner(public_url), safe="")
    return [
        f"https://{user}:{token_q}@github.com/{suffix}",
        f"https://x-access-token:{token_q}@github.com/{suffix}",
    ]


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
    print("[drive] Mounting Google Drive at /content/drive ...", flush=True)
    drive.mount("/content/drive", force_remount=False)
    if not Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Google Drive did not mount at /content/drive/MyDrive")


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
    ok = False
    for idx, candidate in enumerate(candidates, start=1):
        preflight = run_cmd(
            ["git", "ls-remote", "--heads", candidate, branch],
            safe_display=f"git ls-remote --heads <token-authenticated-url> {branch} [auth-form {idx}]",
            check=False,
        )
        if preflight.returncode == 0 and preflight.stdout.strip():
            authed = candidate
            ok = True
            break
    if not ok:
        raise RuntimeError(
            "GitHub authentication failed, or the requested branch was not found. "
            "The Colab secret must be a GitHub PAT with repository Contents=Read access."
        )

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


def import_repo_modules(repo_dir: Path) -> tuple[Any, Any, Any]:
    for path in [str(repo_dir), str(repo_dir / "src")]:
        if path not in sys.path:
            sys.path.insert(0, path)
    from experiments.methodology import colab_zinc_main_procedure as base
    from experiments.peptides_struct.training import grit_peptides_struct_common as pep_common
    from experiments.zinc.training import grit_zinc_1hop_core as onehop_base

    return base, pep_common, onehop_base


def ensure_peptides_helper_api(base: Any) -> None:
    if not hasattr(base, "log"):
        base.log = lambda message="": print(str(message), flush=True)  # type: ignore[attr-defined]


def safe_path_fragment(value: str) -> str:
    fragment = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return fragment or "run"


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return path


def write_yaml(path: Path, payload: Mapping[str, Any]) -> Path:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def clone_official_grit(base: Any, repo_dir: Path, *, force: bool = False) -> None:
    if force and repo_dir.exists():
        shutil.rmtree(repo_dir)
    if not (repo_dir / ".git").exists():
        run_cmd(["git", "clone", "--branch", "main", OFFICIAL_GRIT_REPO, str(repo_dir)])
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_dir), text=True).strip()
    if current != OFFICIAL_GRIT_COMMIT:
        run_cmd(["git", "fetch", "origin"], cwd=repo_dir)
        run_cmd(["git", "checkout", OFFICIAL_GRIT_COMMIT], cwd=repo_dir)
    print(f"[grit] official checkout {repo_dir} @ {OFFICIAL_GRIT_COMMIT}", flush=True)


def prepare_peptides_grit_repos(
    *,
    base: Any,
    pep_common: Any,
    onehop_base: Any,
    dense_repo: Path,
    onehop_repo: Path,
    drive_root: Path,
    force: bool,
) -> tuple[Path, Path, Path, Path]:
    clone_official_grit(base, dense_repo, force=force)
    clone_official_grit(base, onehop_repo, force=force)

    # Peptides dataset compatibility and streaming RRWP collation are analysis-safe:
    # they change neither model equations nor trained checkpoints.
    pep_common.apply_peptides_dataset_compat_patch(base, dense_repo)
    pep_common.apply_peptides_streaming_rrwp_patch(base, dense_repo)
    pep_common.apply_peptides_dataset_compat_patch(base, onehop_repo)
    pep_common.apply_peptides_streaming_rrwp_patch(base, onehop_repo)

    # Reuse the same 1-hop GRIT patch used for training, with Peptides config globals.
    pep_common.configure_base(onehop_base, onehop=True)
    onehop_base.apply_parameter_matched_onehop_patch(onehop_repo, drive_root)

    dense_cfg = dense_repo / DENSE_CFG_REL
    onehop_cfg = onehop_repo / ONEHOP_CFG_REL
    if not dense_cfg.exists():
        raise FileNotFoundError(f"missing dense Peptides GRIT config: {dense_cfg}")
    if not onehop_cfg.exists():
        raise FileNotFoundError(f"missing 1-hop Peptides GRIT config: {onehop_cfg}")
    return dense_repo, onehop_repo, dense_cfg, onehop_cfg


def checkpoint_candidates(root: Path) -> list[Path]:
    patterns = ["*.ckpt", "*checkpoint*.pt", "*checkpoint*.pth", "best*.pt", "best*.pth", "model*.pt"]
    out: list[Path] = []
    if root.exists():
        for pattern in patterns:
            out.extend(p for p in root.rglob(pattern) if p.is_file())
    return sorted(set(out), key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)


def choose_seed_tag_checkpoint(
    *,
    base: Any,
    drive_dir: Path,
    seed: int,
    name_tag: str,
    label: str,
) -> Path:
    results_root = drive_dir / "results"
    fragment = safe_path_fragment(name_tag)
    recovery_dir = results_root / "_recovery_checkpoints" / f"seed{int(seed)}_{fragment}"
    preferred = [
        recovery_dir / "best.ckpt",
        recovery_dir / "latest.ckpt",
        recovery_dir / "first_after_resume.ckpt",
    ]
    for path in preferred:
        if path.exists() and path.is_file():
            print(f"[checkpoint] {label}: {path} (seed/name-tag recovery checkpoint)", flush=True)
            return path

    candidates = checkpoint_candidates(results_root)
    tagged = [p for p in candidates if fragment in str(p) and f"seed{int(seed)}" in str(p)]
    if tagged:
        best_named = [p for p in tagged if p.name.lower().startswith("best")]
        chosen = sorted(best_named or tagged, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)[0]
        print(f"[checkpoint] {label}: {chosen} (tag-filtered fallback)", flush=True)
        return chosen

    print(
        f"[checkpoint-warning] {label}: no checkpoint matched seed={seed}, name_tag={name_tag!r}; "
        "falling back to generic discovery inside this model's Drive root.",
        flush=True,
    )
    return base.choose_checkpoint(results_root, label)


def checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "checkpoint_path": str(path),
        "checkpoint_size_bytes": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
    }


def prepare_or_refresh_pointer(
    *,
    base: Any,
    model: str,
    source_drive_dir: Path,
    config_path: Path,
    checkpoint_path: Path,
    prepared_root: Path,
) -> dict[str, Any]:
    pointer = base.prepare_model_artifact(
        model=model,
        source_drive_dir=source_drive_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        prepared_root=prepared_root,
    )
    pointer.update(checkpoint_fingerprint(checkpoint_path))
    write_json(prepared_root / "artifact_pointer.json", pointer)
    return pointer


def load_pointer(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    ckpt = Path(str(payload.get("checkpoint_path", "")))
    cfg = Path(str(payload.get("config_path", "")))
    return payload if ckpt.exists() and cfg.exists() else None


def build_peptides_config(
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
    seed: int,
    target_index: int,
) -> dict[str, Any]:
    bins = [
        {"label": "d=2-3", "min": 2, "max": 3},
        {"label": "d=4-6", "min": 4, "max": 6},
        {"label": "d=7-10", "min": 7, "max": 10},
        {"label": "d=11-14", "min": 11, "max": 14},
        {"label": "d>14", "min": 15, "max": None},
    ]
    models = {
        "dense_grit": {
            "adapter": "official_grit",
            "variant": "official",
            "role": "treatment",
            "official_repo": "https://github.com/LiamMa/GRIT",
            "official_commit": OFFICIAL_GRIT_COMMIT,
            "repo_path": str(dense_repo),
            "artifact_root": str(dense_prepared),
            "dataset_dir": str(dense_dataset_dir),
            "config_path": str(dense_cfg),
            "checkpoint_path": str(dense_ckpt),
            **checkpoint_fingerprint(dense_ckpt),
        },
        "grit_1hop": {
            "adapter": "official_grit",
            "variant": "1hop",
            "role": "parameter_matched_control",
            "official_repo": "https://github.com/LiamMa/GRIT",
            "official_commit": OFFICIAL_GRIT_COMMIT,
            "repo_path": str(onehop_repo),
            "artifact_root": str(onehop_prepared),
            "dataset_dir": str(onehop_dataset_dir),
            "config_path": str(onehop_cfg),
            "checkpoint_path": str(onehop_ckpt),
            **checkpoint_fingerprint(onehop_ckpt),
        },
    }
    return {
        "artifact_root": str(artifact_root / "artifacts"),
        "dataset": {
            "name": "Peptides-struct",
            "split": "official",
            "task": "multi-target_graph_regression",
            "analysis_target_index": int(target_index),
        },
        "seeds": [int(seed)],
        "primary_tau": 3,
        "far_thresholds": [2, 3, 4],
        "perturbation": {
            "carriage_primary": "integrated_gradients",
            "ig_baseline": "mean_node_embedding",
            "ig_steps": 16,
            "baseline_sample_graphs": 16,
            "swap_partners": 2,
            "batched_vjp": True,
            "carriage_readout_ig": False,
            "swap_partner_policy": "different_type",
            "target_index": int(target_index),
        },
        "models": models,
        "runtime": {"skip_failed_optional_adapters": True},
        "steps": {
            "0": {
                "name": "measurement_model_validation",
                "sample_graphs": 12,
                "ig_step_sweep": [8, 16, 32],
                "diagnostic_sample_graphs": 3,
                "baseline_sweep": ["mean_node_embedding"],
                "matched_target_ig_steps": 16,
                "matched_target_sources_per_graph": 2,
                "matched_target_partners_per_source": 1,
            },
            "1": {"name": "performance_gap", "reach_sweep": [1, "dense"]},
            "2": {
                "name": "usage_vs_causal_usage",
                "sample_graphs": 12,
                "compare_attention_to_swaps": False,
                "run_layer_channel_split": False,
                "run_attention_erasure": True,
                "erasure_models": ["dense_grit"],
                "erasure_sample_graphs": 6,
                "erasure_fractions": [0.0, 0.05, 0.10, 0.20, 0.35, 0.50],
                "erasure_include_near_control": True,
            },
            "3": {"name": "distance_resolved_overfitting", "sample_graphs": 12},
            "4": {
                "name": "mediator_patching",
                "sample_graphs": 6,
                "max_far_pairs_per_graph": 3,
                "depth_pairs_per_graph": 1,
                "all_distance": True,
                "min_distance": 2,
                "stratify_by_distance": True,
                "min_effect_abs": 1.0e-6,
                "signal_gate": True,
                "signal_gate_quantile": 0.90,
                "clamp_mode": "detach",
                "run_analytic_patching_check": True,
                "run_clamp_negative_control": True,
                "run_clamp_mode_comparison": True,
                "clamp_mode_comparison_modes": ["detach", "overwrite"],
                "clamp_mode_comparison_max_pairs_per_model": 4,
            },
            "5": {
                "name": "non_composable_gap_attribution",
                "sample_graphs": 12,
                "max_far_pairs_per_graph": 3,
                "interaction_pairs": 64,
                "min_effect_abs": 1.0e-6,
                "signal_gate": True,
                "signal_gate_quantile": 0.90,
                "clamp_mode": "detach",
                "reference_models": ["dense_grit", "grit_1hop"],
                "load_bearing_ablation_fractions": [0.0, 0.05, 0.10, 0.25, 0.50, 1.0],
                "load_bearing_random_draws": 4,
                "dense_excess_control_model": "grit_1hop",
                "dense_excess_distance_bins": bins[1:],
                "distance_binned_ablation_bins": bins,
            },
            "6": {
                "name": "beneficial_carriage",
                "sample_graphs": 12,
                "donors": 4,
                "max_distance": 8,
                "resamplers": ["matched", "marginal"],
                "splits": ["test"],
            },
            "7": {
                "name": "symbolic_structural_carriage",
                "run_symbolic_structural_carriage": True,
                "symbolic_structural_sample_graphs": 8,
                "symbolic_structural_max_sources": "all",
                "symbolic_structural_min_distance": 1,
                "symbolic_structural_rrwp_channel_start": 2,
                "symbolic_structural_rrwp_replacement": "zero",
                "symbolic_structural_donor_samples": 2,
                "run_structural_carriage_ig": True,
                "run_rrwp_distance_ablation": True,
                "rrwp_ablation_sample_graphs": 8,
                "rrwp_ablation_channel_start": 2,
                "rrwp_ablation_replacement": "zero",
                "rrwp_ablation_types": ["node", "pair", "both"],
                "rrwp_distance_ablation_types": ["pair"],
                "global_rrwp_channel_ablation_types": ["node", "pair", "both"],
                "run_global_rrwp_channel_ablation": True,
                "global_rrwp_channel_ablation_sample_graphs": 12,
                "rrwp_ablation_distance_bins": bins,
                "rrwp_graph_metric_cut_pairs": 24,
            },
        },
        "figures": {"dpi": 180},
        "colab_notes": {
            "runner": "colab_peptides_struct_main_procedure",
            "models_included": ["dense_grit", "grit_1hop"],
            "omitted_until_trained": ["gin", "grit_1hop_localrrwp"],
            "dense_name_tag": DEFAULT_DENSE_NAME_TAG,
            "onehop_name_tag": DEFAULT_ONEHOP_NAME_TAG,
            "analysis_target_index": int(target_index),
        },
    }


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


def assess_completion(artifact_root: Path, requested_steps: Sequence[str]) -> dict[str, Any]:
    status_path = artifact_root / "metrics" / "main_status.json"
    statuses = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    incomplete = {
        step: status
        for step, status in statuses.items()
        if str(step) in set(requested_steps)
        and isinstance(status, Mapping)
        and str(status.get("status")) in {"failed", "waiting_for_training_artifacts", "waiting_for_official_grit_artifacts"}
    }
    missing = [step for step in requested_steps if str(step) not in statuses]
    return {
        "artifact_root": str(artifact_root),
        "requested_steps": list(requested_steps),
        "status_by_step": statuses,
        "incomplete_steps": incomplete,
        "missing_steps": missing,
        "complete": not incomplete and not missing,
    }


def parse_requested_steps(raw: str) -> list[str]:
    if str(raw).strip() == "all":
        return [str(i) for i in range(8)]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def clean_colab_argv(argv: Sequence[str] | None) -> list[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    out: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "-f" and i + 1 < len(raw) and "kernel-" in raw[i + 1] and raw[i + 1].endswith(".json"):
            print(f"[args] ignoring notebook launcher arguments: {raw[i:i+2]}", flush=True)
            i += 2
            continue
        out.append(raw[i])
        i += 1
    return out


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
    parser.add_argument("--dense-repo-dir", type=Path, default=Path(DEFAULT_DENSE_REPO_DIR))
    parser.add_argument("--onehop-repo-dir", type=Path, default=Path(DEFAULT_ONEHOP_REPO_DIR))
    parser.add_argument("--single-seed", type=int, default=41)
    parser.add_argument("--dense-name-tag", default=DEFAULT_DENSE_NAME_TAG)
    parser.add_argument("--onehop-name-tag", default=DEFAULT_ONEHOP_NAME_TAG)
    parser.add_argument("--target-index", type=int, default=0, help="Peptides-struct output dimension to analyze.")
    parser.add_argument("--prepared-id", default="peptides_struct_seed41_dense_and_1hop")
    parser.add_argument(
        "--refresh-model-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rediscover seed/name-tag checkpoints and rewrite prepared pointers before analysis.",
    )
    parser.add_argument("--pyg-version", default=DEFAULT_PYG_VERSION)
    parser.add_argument("--steps", default="7", help="Steps to run, e.g. 7 or 2,4,7 or all.")
    parser.add_argument("--analysis-preset", default="medium", choices=["smoke", "quick", "pilot", "medium", "high", "full"])
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force recomputation of requested main-procedure steps.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--skip-git", action="store_true")
    parser.add_argument("--force-official-grit-reclone", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-onehop-locality-check", action="store_true")
    parser.add_argument("--onehop-locality-check-graphs", type=int, default=1)
    parser.add_argument("--onehop-locality-tolerance", type=float, default=1.0e-12)
    return parser.parse_args(clean_colab_argv(argv))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    mount_drive()
    args.drive_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_git:
        token = require_colab_token(args.secret_name)
        clone_or_update_repo(args.repo_url, args.branch, args.repo_dir, token, args.github_username)
    else:
        print(f"[git] using existing repo: {args.repo_dir}", flush=True)

    base, pep_common, onehop_base = import_repo_modules(args.repo_dir)
    ensure_peptides_helper_api(base)
    compat_shim_dir = base.write_py312_compat_shim(args.drive_root)
    analysis_env = base.env_with_py312_compat(compat_shim_dir)
    if not args.skip_install:
        base.install_repo(args.repo_dir, pyg_version=str(args.pyg_version))
        pep_common.install_peptides_dependencies(base)
    else:
        print("[deps] skipping dependency installation (--skip-install)", flush=True)

    dense_repo, onehop_repo, dense_cfg, onehop_cfg = prepare_peptides_grit_repos(
        base=base,
        pep_common=pep_common,
        onehop_base=onehop_base,
        dense_repo=args.dense_repo_dir,
        onehop_repo=args.onehop_repo_dir,
        drive_root=args.drive_root,
        force=bool(args.force_official_grit_reclone),
    )

    prepared = args.drive_root / "prepared_model_artifacts" / str(args.prepared_id)
    dense_pointer_path = prepared / "dense_grit" / "artifact_pointer.json"
    onehop_pointer_path = prepared / "grit_1hop" / "artifact_pointer.json"
    print(f"[prepared] using stable prepared artifact root: {prepared}", flush=True)

    dense_pointer = None if args.refresh_model_artifacts else load_pointer(dense_pointer_path)
    onehop_pointer = None if args.refresh_model_artifacts else load_pointer(onehop_pointer_path)

    if dense_pointer is None:
        dense_ckpt = choose_seed_tag_checkpoint(
            base=base,
            drive_dir=args.dense_drive_dir,
            seed=int(args.single_seed),
            name_tag=str(args.dense_name_tag),
            label="dense_grit",
        )
        dense_pointer = prepare_or_refresh_pointer(
            base=base,
            model="dense_grit",
            source_drive_dir=args.dense_drive_dir,
            config_path=dense_cfg,
            checkpoint_path=dense_ckpt,
            prepared_root=prepared / "dense_grit",
        )
    else:
        dense_ckpt = Path(str(dense_pointer["checkpoint_path"]))
        print(f"[prepared] reusing dense checkpoint: {dense_ckpt}", flush=True)

    if onehop_pointer is None:
        onehop_ckpt = choose_seed_tag_checkpoint(
            base=base,
            drive_dir=args.onehop_drive_dir,
            seed=int(args.single_seed),
            name_tag=str(args.onehop_name_tag),
            label="grit_1hop",
        )
        onehop_pointer = prepare_or_refresh_pointer(
            base=base,
            model="grit_1hop",
            source_drive_dir=args.onehop_drive_dir,
            config_path=onehop_cfg,
            checkpoint_path=onehop_ckpt,
            prepared_root=prepared / "grit_1hop",
        )
    else:
        onehop_ckpt = Path(str(onehop_pointer["checkpoint_path"]))
        print(f"[prepared] reusing 1-hop checkpoint: {onehop_ckpt}", flush=True)

    cfg = build_peptides_config(
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
        seed=int(args.single_seed),
        target_index=int(args.target_index),
    )
    config_path = write_yaml(args.drive_root / "configs" / "peptides_struct_main_procedure_colab.yaml", cfg)
    print(f"[config] wrote {config_path}", flush=True)
    print(f"[config] models included: {', '.join(sorted(cfg['models']))}", flush=True)
    write_json(
        args.drive_root / "prepared_model_artifacts" / "latest_pointers.json",
        {
            "dense_grit": dense_pointer,
            "grit_1hop": onehop_pointer,
            "config_path": str(config_path),
        },
    )

    if not args.skip_onehop_locality_check:
        try:
            base.run_onehop_locality_preflight(
                args.repo_dir,
                config_path,
                args.drive_root,
                sample_graphs=int(args.onehop_locality_check_graphs),
                tolerance=float(args.onehop_locality_tolerance),
                env=analysis_env,
            )
        except Exception as exc:  # noqa: BLE001
            write_json(
                args.drive_root / "preflight" / "onehop_locality_certification_failed.json",
                {"status": "failed", "error": str(exc), "config_path": str(config_path)},
            )
            if not args.allow_incomplete:
                raise
            print(f"[preflight-warning] 1-hop locality check failed but --allow-incomplete is set: {exc}", flush=True)

    artifact_root = base.run_main_procedure(
        args.repo_dir,
        config_path,
        steps=str(args.steps),
        force=bool(args.force),
        analysis_preset=str(args.analysis_preset),
        fast_dev_run=bool(args.fast_dev_run),
        dry_run=bool(args.dry_run),
        env=analysis_env,
    )
    requested_steps = parse_requested_steps(str(args.steps))
    completion = assess_completion(artifact_root, requested_steps)
    latest = copy_latest_outputs(args.drive_root, artifact_root)
    write_json(
        args.drive_root / "methodology_fidelity_audit.json",
        {
            "dataset": cfg["dataset"],
            "requested_steps": requested_steps,
            "models": cfg["models"],
            "completion": completion,
            "notes": [
                "Single seed 41 only; no GIN or local-RRWP Peptides models included yet.",
                f"Peptides-struct scalar analysis target index: {int(args.target_index)}.",
            ],
        },
    )

    print("[completion]", json.dumps(completion, indent=2, sort_keys=True), flush=True)
    print(f"[done] artifacts: {artifact_root}", flush=True)
    print(f"[done] latest outputs: {latest}", flush=True)
    print(f"[done] fidelity audit: {args.drive_root / 'methodology_fidelity_audit.json'}", flush=True)

    if not completion.get("complete") and not args.allow_incomplete:
        raise SystemExit(
            "The Peptides-struct methodology run did not complete the requested steps. "
            f"Completion: {json.dumps(completion, sort_keys=True)}"
        )


if __name__ == "__main__":
    main([
        "--steps", "7",
        "--analysis-preset", "medium",
    ])
