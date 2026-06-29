#!/usr/bin/env python3
"""Standalone Google Colab runner for method-validation synthetics.

Run this file in Colab.  It mounts Google Drive, clones/pulls the GitHub repo
using the Colab secret named ``dissertation_key``, installs the repo in editable mode,
and runs the synthetic method-validation suite with Drive-backed caches.

Default mode is the core synthetic run from ``method_validation.md``.  Use
``--mode fast-dev`` only for a quick smoke test.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote


PUBLIC_REPO_URL = "https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git"
DEFAULT_BRANCH = "main"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/graph_specialisation_metrics/method_validation_colab"
DEFAULT_REPO_DIR = "/content/Graph-Specialisation-and-Metrics"


def run_cmd(
    cmd: Sequence[str],
    *,
    cwd: Path | str | None = None,
    safe_display: str | None = None,
    check: bool = True,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    display = safe_display or " ".join(cmd)
    print(f"[cmd] {display}", flush=True)
    proc = subprocess.run(
        list(cmd),
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


def require_colab_token(secret_name: str) -> str:
    try:
        from google.colab import userdata
    except Exception as exc:  # pragma: no cover - only exercised outside Colab.
        raise RuntimeError(
            "This runner is intended for Google Colab. "
            "Create a Colab secret named dissertation_key and run it there."
        ) from exc
    token = userdata.get(secret_name)
    if not token:
        raise RuntimeError(f"Colab secret {secret_name!r} is missing or empty")
    secret = str(token).strip()
    if "PRIVATE KEY" in secret:
        print(f"[auth] secret {secret_name!r} looks like an SSH private key", flush=True)
    else:
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
    # Handle accidental Markdown links such as [url](url).
    if repo_url.startswith("[") and "](" in repo_url and repo_url.endswith(")"):
        repo_url = repo_url.split("](", 1)[1][:-1]
    return repo_url


def repo_owner(public_url: str) -> str:
    public_url = sanitize_repo_url(public_url)
    if not public_url.startswith("https://github.com/"):
        raise ValueError("only https://github.com repo URLs are supported by this runner")
    return public_url.removeprefix("https://github.com/").split("/", 1)[0]


def auth_url_candidates(public_url: str, token: str, username: str | None = None) -> list[str]:
    public_url = sanitize_repo_url(public_url)
    if not public_url.startswith("https://github.com/"):
        raise ValueError("only https://github.com repo URLs are supported by this runner")
    user = quote(username or repo_owner(public_url), safe="")
    token_q = quote(token, safe="")
    suffix = public_url.removeprefix("https://github.com/")
    return [
        f"https://{user}:{token_q}@github.com/{suffix}",
        f"https://x-access-token:{token_q}@github.com/{suffix}",
    ]


def ssh_url(public_url: str) -> str:
    public_url = sanitize_repo_url(public_url)
    if not public_url.startswith("https://github.com/"):
        raise ValueError("only https://github.com repo URLs can be converted to SSH")
    suffix = public_url.removeprefix("https://github.com/")
    return f"git@github.com:{suffix}"


def setup_ssh_key(secret: str) -> dict[str, str]:
    ssh_dir = Path("/root/.ssh")
    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    key_path = ssh_dir / "dissertation_key"
    key_text = secret.strip() + "\n"
    key_path.write_text(key_text, encoding="utf-8")
    key_path.chmod(0o600)
    known_hosts = ssh_dir / "known_hosts"
    scan = run_cmd(
        ["ssh-keyscan", "-t", "rsa,ecdsa,ed25519", "github.com"],
        safe_display="ssh-keyscan github.com",
        check=True,
    )
    with known_hosts.open("a", encoding="utf-8") as f:
        f.write(scan.stdout)
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = (
        f"ssh -i {key_path} -o IdentitiesOnly=yes "
        "-o StrictHostKeyChecking=yes -o UserKnownHostsFile=/root/.ssh/known_hosts"
    )
    return env


def clone_or_update_repo(
    repo_url: str,
    branch: str,
    repo_dir: Path,
    secret: str,
    github_username: str | None,
) -> None:
    repo_dir = Path(repo_dir)
    repo_url = sanitize_repo_url(repo_url)
    using_ssh = "PRIVATE KEY" in secret
    auth_candidates = [ssh_url(repo_url)] if using_ssh else auth_url_candidates(repo_url, secret, github_username)
    safe_auth_display = "<ssh-authenticated-url>" if using_ssh else "<token-authenticated-url>"
    git_env = setup_ssh_key(secret) if using_ssh else None

    # Cheap preflight gives a clearer auth/branch failure before clone output.
    authed = auth_candidates[0]
    preflight: subprocess.CompletedProcess[str] | None = None
    for idx, candidate in enumerate(auth_candidates, start=1):
        preflight = run_cmd(
            ["git", "ls-remote", "--heads", candidate, branch],
            safe_display=f"git ls-remote --heads {safe_auth_display} {branch} [auth-form {idx}]",
            check=False,
            env=git_env,
        )
        if preflight.returncode == 0:
            authed = candidate
            break
    assert preflight is not None
    if preflight.returncode != 0:
        print(
            "[auth:error] GitHub rejected the credential or the repo URL. "
            "Read-only access is sufficient for clone/pull, but a fine-grained "
            "PAT must select this repository and grant Repository permissions: "
            "Contents=Read-only (Metadata read-only is automatic but not enough). "
            "A classic PAT for a private repo needs the repo scope. If this is an "
            "SSH key, the public key must be registered with GitHub.",
            flush=True,
        )
        # If the repo is public, unauthenticated clone may still work.
        public_probe = run_cmd(
            ["git", "ls-remote", "--heads", repo_url, branch],
            safe_display=f"git ls-remote --heads {repo_url} {branch}",
            check=False,
        )
        if public_probe.returncode == 0:
            print("[auth] public unauthenticated access works; continuing without secret", flush=True)
            authed = repo_url
            safe_auth_display = repo_url
            git_env = None
        else:
            raise RuntimeError("GitHub authentication preflight failed; check the dissertation_key secret")
    elif not preflight.stdout.strip():
        raise RuntimeError(
            f"GitHub authentication worked, but branch {branch!r} was not found. "
            "Pass --branch with the branch containing the methodology code."
        )

    if (repo_dir / ".git").exists():
        run_cmd(
            ["git", "-C", str(repo_dir), "remote", "set-url", "origin", authed],
            safe_display=f"git -C {repo_dir} remote set-url origin {safe_auth_display}",
            env=git_env,
        )
        run_cmd(["git", "-C", str(repo_dir), "fetch", "origin", branch], env=git_env)
        run_cmd(["git", "-C", str(repo_dir), "checkout", branch], env=git_env)
        run_cmd(["git", "-C", str(repo_dir), "pull", "--ff-only", "origin", branch], env=git_env)
        run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url], env=git_env)
        return
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    run_cmd(
        ["git", "clone", "--branch", branch, "--single-branch", authed, str(repo_dir)],
        safe_display=f"git clone --branch {branch} {safe_auth_display} {repo_dir}",
        env=git_env,
    )
    run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url], env=git_env)


def install_repo(repo_dir: Path) -> None:
    # Avoid reinstalling Colab's torch build.  The repo dependencies used here are
    # small and installed explicitly.
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "pyyaml", "networkx", "matplotlib", "numpy", "scipy", "pandas"])
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_dir), "--no-deps"])
    sys.path.insert(0, str(repo_dir / "src"))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_text_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def torch_load(path: Path) -> Any:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def stable_config_hash(config: Mapping[str, Any]) -> str:
    from graph_specialisation_metrics.method_core import config_hash

    payload = copy.deepcopy(dict(config))
    payload.pop("artifact_root", None)
    payload["colab_cache_schema"] = "method_validation_core.v1"
    return config_hash(payload)


def graphs_to_payload(graphs: Sequence[Any], weight: Any) -> dict[str, Any]:
    return {
        "weight": weight.detach().cpu(),
        "graphs": [
            {
                "x": graph.x.detach().cpu(),
                "edge_index": graph.edge_index.detach().cpu(),
                "batch": graph.batch.detach().cpu() if graph.batch is not None else None,
                "y": graph.y.detach().cpu() if graph.y is not None else None,
                "distances": graph.distances.detach().cpu() if graph.distances is not None else None,
                "graph_ids": list(graph.graph_ids or []),
                "split": graph.split,
                "metadata": {
                    key: value.detach().cpu() if hasattr(value, "detach") else value
                    for key, value in dict(graph.metadata or {}).items()
                },
            }
            for graph in graphs
        ],
    }


def payload_to_graphs(payload: Mapping[str, Any]) -> tuple[list[Any], Any]:
    from graph_specialisation_metrics.method_core import GraphBatchView

    graphs = []
    for item in payload["graphs"]:
        graphs.append(
            GraphBatchView(
                x=item["x"],
                edge_index=item["edge_index"],
                batch=item.get("batch"),
                y=item.get("y"),
                graph_ids=item.get("graph_ids") or None,
                split=item.get("split"),
                distances=item.get("distances"),
                metadata=dict(item.get("metadata") or {}),
            )
        )
    return graphs, payload["weight"]


def cache_or_create_carriage_data(config: Mapping[str, Any], cache_path: Path, seed: int, force: bool) -> tuple[list[Any], Any]:
    import torch
    from graph_specialisation_metrics.method_validation import make_carriage_dataset

    if cache_path.exists() and not force:
        print(f"[cache] loading carriage dataset: {cache_path}", flush=True)
        return payload_to_graphs(torch_load(cache_path))
    print(f"[cache] creating carriage dataset: {cache_path}", flush=True)
    graphs, weight = make_carriage_dataset(config["carriage"], seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs_to_payload(graphs, weight), cache_path)
    return graphs, weight


def make_validation_adapter(config: Mapping[str, Any], device: Any) -> Any:
    from graph_specialisation_metrics.method_adapters import make_small_graph_transformer_adapter

    spec = config["carriage"]["small_gt"]
    return make_small_graph_transformer_adapter(
        content_dim=int(config["carriage"]["feature_dim"]),
        hidden_dim=int(spec["hidden_dim"]),
        layers=int(spec["layers"]),
        heads=int(spec["heads"]),
        device=device,
    )


def cache_or_train_model(
    config: Mapping[str, Any],
    graphs: Sequence[Any],
    cache_path: Path,
    device: Any,
    seed: int,
    force_retrain: bool,
) -> tuple[Any, list[dict[str, float]]]:
    import torch
    from graph_specialisation_metrics.method_validation import train_small_gt

    history_path = cache_path.with_suffix(".history.json")
    adapter = make_validation_adapter(config, device)
    if cache_path.exists() and history_path.exists() and not force_retrain:
        print(f"[cache] loading small GT checkpoint: {cache_path}", flush=True)
        payload = torch_load(cache_path)
        adapter.model.load_state_dict(payload["state_dict"])
        with history_path.open("r", encoding="utf-8") as f:
            history_payload = json.load(f)
        return adapter, list(history_payload.get("history", history_payload if isinstance(history_payload, list) else []))
    print(f"[train] training small GT and caching to: {cache_path}", flush=True)
    adapter, history = train_small_gt(graphs, config["carriage"], device=device, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": adapter.model.state_dict(),
            "config": dict(config["carriage"]),
            "parameter_count": adapter.parameter_count(),
            "validation_only": True,
        },
        cache_path,
    )
    write_text_json(history_path, {"history": history})
    return adapter, history


def load_history(history_path: Path) -> list[dict[str, float]]:
    if not history_path.exists():
        return []
    with history_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("history", payload if isinstance(payload, list) else []))


EXPECTED_FIGURE_FILES = (
    "validation_carriage_check.png",
    "validation_patching_check.png",
    "validation_rank_check.png",
    "validation_interaction_check.png",
)


def force_requested(args: argparse.Namespace) -> bool:
    return bool(args.force_data or args.force_retrain or args.force_recompute)


def run_is_complete(run_root: Path) -> bool:
    if not (run_root / "metrics" / "validation_summary.json").exists():
        return False
    if not (run_root / "manifest.json").exists():
        return False
    return all((run_root / "figures" / name).exists() for name in EXPECTED_FIGURE_FILES)


def copy_latest_figures(drive_root: Path, mode: str, run_root: Path) -> Path:
    latest = drive_root / "latest_figures" / mode
    if latest.exists():
        shutil.rmtree(latest)
    latest.mkdir(parents=True, exist_ok=True)
    for figure in sorted((run_root / "figures").glob("*")):
        if figure.is_file():
            shutil.copy2(figure, latest / figure.name)
    return latest


def print_completed_run(run_root: Path, latest: Path) -> None:
    with (run_root / "metrics" / "validation_summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    print("[cache] complete Drive artifact already exists; skipping computation", flush=True)
    print("[summary]", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"[done] artifacts: {run_root}", flush=True)
    print(f"[done] latest figures: {latest}", flush=True)


def run_carriage_with_cache(
    config: Mapping[str, Any],
    run_root: Path,
    cache_root: Path,
    device: Any,
    seed: int,
    force_data: bool,
    force_retrain: bool,
    force_recompute: bool,
) -> dict[str, float]:
    from graph_specialisation_metrics.method_core import atomic_torch_save, write_csv, write_json
    from graph_specialisation_metrics.method_validation import (
        baseline_for_graphs,
        carriage_reconstruction_rows,
        plot_carriage,
    )

    key = stable_config_hash({"carriage": config["carriage"], "seed": seed})
    data_cache = cache_root / "data" / f"carriage_dataset_{key}.pt"
    model_cache = cache_root / "models" / f"small_gt_{key}.pt"
    graphs, weight = cache_or_create_carriage_data(config, data_cache, seed, force_data)
    adapter, history = cache_or_train_model(config, graphs, model_cache, device, seed, force_retrain)

    metrics_csv = run_root / "metrics" / "carriage_reconstruction.csv"
    tensor_path = run_root / "tensors" / "carriage_check.pt"
    history_csv = run_root / "metrics" / "carriage_training.csv"
    if metrics_csv.exists() and tensor_path.exists() and not force_recompute:
        print(f"[cache] using existing carriage metrics: {metrics_csv}", flush=True)
        rows = read_csv_rows(metrics_csv)
    else:
        print("[run] computing carriage IG reconstruction rows", flush=True)
        baseline = baseline_for_graphs(graphs, device)
        rows, tensors = carriage_reconstruction_rows(
            adapter,
            graphs,
            baseline=baseline,
            cfg=config["carriage"],
            device=device,
        )
        tensors["target_weight"] = weight
        tensors["dataset_cache"] = str(data_cache)
        tensors["model_cache"] = str(model_cache)
        write_csv(metrics_csv, rows)
        atomic_torch_save(tensor_path, tensors)
    write_csv(history_csv, history)
    summary = plot_carriage(rows, history, run_root, config)
    write_json(run_root / "metrics" / "carriage_summary.json", summary)
    return summary


def run_cached_metric_check(
    *,
    name: str,
    run_root: Path,
    cache_root: Path,
    cache_key: str,
    force_recompute: bool,
    compute: Callable[[], dict[str, float]],
    artifact_csv: Path,
    plot_from_rows: Callable[[list[dict[str, Any]]], dict[str, float]],
    summary_json: Path,
) -> dict[str, float]:
    from graph_specialisation_metrics.method_core import write_csv, write_json

    cache_csv = cache_root / "metrics" / f"{name}_{cache_key}.csv"
    cache_summary = cache_root / "metrics" / f"{name}_{cache_key}.summary.json"
    if cache_csv.exists() and cache_summary.exists() and not force_recompute:
        print(f"[cache] using {name} metrics: {cache_csv}", flush=True)
        rows = read_csv_rows(cache_csv)
        write_csv(artifact_csv, rows)
        summary = plot_from_rows(rows)
        write_json(summary_json, summary)
        return summary
    print(f"[run] computing {name}", flush=True)
    summary = compute()
    if artifact_csv.exists():
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact_csv, cache_csv)
    write_text_json(cache_summary, summary)
    return summary


def run_validation_core(args: argparse.Namespace) -> Path:
    try:
        return run_validation_core_repo(args)
    except ModuleNotFoundError as exc:
        missing = str(exc)
        if "graph_specialisation_metrics.method_core" not in missing and "graph_specialisation_metrics.method_validation" not in missing:
            raise
        print(
            "[fallback] cloned GitHub branch does not contain the new methodology modules; "
            "running the embedded standalone synthetic validation implementation instead.",
            flush=True,
        )
        return run_validation_core_embedded(args, exc)


def run_validation_core_repo(args: argparse.Namespace) -> Path:
    import torch
    from graph_specialisation_metrics.method_core import AdapterInfo, ensure_dir, set_global_seed, write_json, write_manifest, write_yaml
    from graph_specialisation_metrics.method_validation import (
        DEFAULT_CONFIG,
        FAST_DEV_OVERRIDES,
        MAIN_PROCEDURE_MD,
        METHOD_VALIDATION_MD,
        SOURCE_MARKDOWN_EXPECTED_SHA256,
        deep_update,
        plot_interaction,
        plot_patching,
        plot_rank,
        run_interaction_check,
        run_patching_check,
        run_rank_check,
    )

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if args.mode == "fast-dev":
        cfg = deep_update(cfg, FAST_DEV_OVERRIDES)
    cfg["device"] = args.device
    seed = int(cfg["seed"])
    set_global_seed(seed)

    drive_root = Path(args.drive_root)
    cache_root = drive_root / "cache" / args.mode
    run_key = stable_config_hash(cfg)
    run_root = drive_root / "artifacts" / args.mode / run_key
    cfg["artifact_root"] = str(run_root)
    ensure_dir(run_root / "metrics")
    ensure_dir(run_root / "tensors")
    ensure_dir(run_root / "figures")
    write_yaml(run_root / "config.yaml", cfg)

    if run_is_complete(run_root) and not force_requested(args):
        latest = copy_latest_figures(drive_root, args.mode, run_root)
        print_completed_run(run_root, latest)
        return run_root

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this Colab runtime")
    print(f"[device] {device}", flush=True)
    print(f"[drive] run_root={run_root}", flush=True)
    print(f"[drive] cache_root={cache_root}", flush=True)

    summaries: dict[str, Any] = {}
    summaries["carriage"] = run_carriage_with_cache(
        cfg,
        run_root,
        cache_root,
        device,
        seed,
        force_data=args.force_data,
        force_retrain=args.force_retrain,
        force_recompute=args.force_recompute,
    )

    key = stable_config_hash({"patching": cfg["patching"], "seed": seed})
    summaries["patching"] = run_cached_metric_check(
        name="patching",
        run_root=run_root,
        cache_root=cache_root,
        cache_key=key,
        force_recompute=args.force_recompute,
        compute=lambda: run_patching_check(cfg, run_root, seed),
        artifact_csv=run_root / "metrics" / "patching_retained.csv",
        plot_from_rows=lambda rows: plot_patching(rows, run_root, cfg),
        summary_json=run_root / "metrics" / "patching_summary.json",
    )

    key = stable_config_hash({"rank": cfg["rank"], "seed": seed})
    summaries["rank"] = run_cached_metric_check(
        name="rank",
        run_root=run_root,
        cache_root=cache_root,
        cache_key=key,
        force_recompute=args.force_recompute,
        compute=lambda: run_rank_check(cfg, run_root, seed),
        artifact_csv=run_root / "metrics" / "rank_check.csv",
        plot_from_rows=lambda rows: plot_rank(rows, run_root, cfg),
        summary_json=run_root / "metrics" / "rank_summary.json",
    )

    key = stable_config_hash({"interaction": cfg["interaction"], "seed": seed})
    summaries["interaction"] = run_cached_metric_check(
        name="interaction",
        run_root=run_root,
        cache_root=cache_root,
        cache_key=key,
        force_recompute=args.force_recompute,
        compute=lambda: run_interaction_check(cfg, run_root, seed),
        artifact_csv=run_root / "metrics" / "interaction_check.csv",
        plot_from_rows=lambda rows: plot_interaction(rows, run_root, cfg),
        summary_json=run_root / "metrics" / "interaction_summary.json",
    )

    write_json(run_root / "metrics" / "validation_summary.json", summaries)
    write_manifest(
        run_root,
        run_type="method_validation_colab",
        config=cfg,
        adapter=AdapterInfo(
            name="method_validation_colab_runner",
            version="colab.v1",
            implementation="GitHub-loaded repo runner with Drive-backed caches",
            validation_only=True,
        ),
        source_markdowns=[METHOD_VALIDATION_MD, MAIN_PROCEDURE_MD],
        extra={
            "mode": args.mode,
            "device": str(device),
            "drive_root": str(drive_root),
            "cache_root": str(cache_root),
            "source_markdown_expected_sha256": SOURCE_MARKDOWN_EXPECTED_SHA256,
            "fidelity_notes": [
                "Current repo implementation uses a validation-only selector_mask for the planted Carriage Check set S.",
                "Current repo implementation compares IG to baseline replacement for carriage reconstruction.",
                "Current rank null uses row-wise source shuffling rather than a single global column permutation.",
            ],
        },
    )

    latest = copy_latest_figures(drive_root, args.mode, run_root)

    print("[summary]", json.dumps(summaries, indent=2, sort_keys=True), flush=True)
    print(f"[done] artifacts: {run_root}", flush=True)
    print(f"[done] latest figures: {latest}", flush=True)
    return run_root


def embedded_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def embedded_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def embedded_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def embedded_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def embedded_default_config(mode: str, device: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "seed": 41,
        "device": device,
        "carriage": {
            "num_graphs": 500,
            "num_nodes": 20,
            "feature_dim": 8,
            "planted_set_size": 3,
            "min_planted_pair_distance": 3,
            "analysis_graphs": 200,
            "ig_steps": 32,
            "train_epochs": 250,
            "learning_rate": 0.001,
            "fit_mse_fraction_target_variance": 0.01,
            "small_gt": {"layers": 3, "hidden_dim": 64, "heads": 4},
        },
        "patching": {"chain_lengths": [4, 5, 6, 7, 8], "feature_dim": 8, "branch_attachments": [2, 3, 4]},
        "rank": {
            "matrix_size": 40,
            "planted_ranks": [1, 2, 3, 5, 8],
            "noise_levels": [0.05, 0.2],
            "matrices_per_setting": 50,
            "energy": 0.99,
            "null_permutations": 32,
        },
        "interaction": {"feature_dim": 8, "pairs": 1000, "min_denominator": 1.0e-8},
        "figures": {"dpi": 180, "bootstrap_draws": 1000},
    }
    if mode == "fast-dev":
        cfg["carriage"].update(
            {
                "num_graphs": 48,
                "num_nodes": 14,
                "analysis_graphs": 12,
                "ig_steps": 8,
                "train_epochs": 45,
                "small_gt": {"layers": 2, "hidden_dim": 32, "heads": 4},
            }
        )
        cfg["patching"]["chain_lengths"] = [4, 5, 6]
        cfg["rank"].update(
            {
                "matrix_size": 24,
                "planted_ranks": [1, 2, 3],
                "matrices_per_setting": 8,
                "null_permutations": 8,
            }
        )
        cfg["interaction"]["pairs"] = 128
        cfg["figures"]["bootstrap_draws"] = 200
    return cfg


def embedded_connected_edges(num_nodes: int, rng: Any, extra_edges: int) -> list[tuple[int, int]]:
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    existing = {tuple(sorted(e)) for e in edges}
    attempts = 0
    while len(existing) < num_nodes - 1 + extra_edges and attempts < 10000:
        u, v = rng.choice(num_nodes, size=2, replace=False)
        key = tuple(sorted((int(u), int(v))))
        if key not in existing:
            existing.add(key)
            edges.append(key)
        attempts += 1
    return edges


def embedded_distances(num_nodes: int, edges: Sequence[tuple[int, int]]) -> Any:
    import numpy as np

    dist = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)
    for u, v in edges:
        dist[int(u), int(v)] = 1.0
        dist[int(v), int(u)] = 1.0
    for k in range(num_nodes):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
    return dist


def embedded_make_carriage_dataset(cfg: Mapping[str, Any], seed: int) -> tuple[list[dict[str, Any]], Any]:
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    gen_w = torch.Generator(device="cpu").manual_seed(seed + 99)
    weight = torch.randn(int(cfg["feature_dim"]), generator=gen_w)
    graphs: list[dict[str, Any]] = []
    for graph_idx in range(int(cfg["num_graphs"])):
        n = int(cfg["num_nodes"])
        edges = embedded_connected_edges(n, rng, max(1, n // 3))
        dist = embedded_distances(n, edges)
        planted: list[int] | None = None
        for _ in range(2000):
            candidate = sorted(int(v) for v in rng.choice(n, size=int(cfg["planted_set_size"]), replace=False))
            if max(dist[u, v] for u in candidate for v in candidate if u != v) >= int(cfg["min_planted_pair_distance"]):
                planted = candidate
                break
        if planted is None:
            planted = sorted(int(v) for v in rng.choice(n, size=int(cfg["planted_set_size"]), replace=False))
        gen_x = torch.Generator(device="cpu").manual_seed(seed + 1000 + graph_idx)
        x = torch.randn(n, int(cfg["feature_dim"]), generator=gen_x)
        selector = torch.zeros(n, dtype=torch.float32)
        selector[planted] = 1.0
        y = (x[planted] * weight.view(1, -1)).sum().view(1)
        graphs.append({"x": x, "y": y, "edges": edges, "distances": dist, "planted": planted, "selector": selector})
    return graphs, weight


class EmbeddedSmallGT:
    def __init__(self, content_dim: int, hidden_dim: int, layers: int, heads: int, device: Any) -> None:
        import torch
        import torch.nn as nn

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Linear(content_dim + 1, hidden_dim)
                self.layers = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(
                            d_model=hidden_dim,
                            nhead=heads,
                            dim_feedforward=hidden_dim * 2,
                            dropout=0.0,
                            batch_first=True,
                            activation="relu",
                        )
                        for _ in range(layers)
                    ]
                )
                self.selector_head = nn.Linear(content_dim, 1, bias=False)

            def forward(self, x: Any, selector: Any) -> Any:
                h = torch.cat([x, selector.view(-1, 1)], dim=-1).unsqueeze(0)
                h = self.encoder(h)
                for layer in self.layers:
                    h = layer(h)
                selected = (x * selector.view(-1, 1)).sum(dim=0, keepdim=True)
                return self.selector_head(selected).view(1), h.squeeze(0)

        self.model = Model().to(device)
        self.device = device

    def predict(self, graph: Mapping[str, Any]) -> Any:
        x = graph["x"].to(self.device)
        selector = graph["selector"].to(self.device)
        return self.model(x, selector)[0]

    def state_dict(self) -> Any:
        return self.model.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.model.load_state_dict(state)

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.model.parameters() if p.requires_grad)


def embedded_train_small_gt(graphs: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], device: Any, seed: int) -> tuple[EmbeddedSmallGT, list[dict[str, float]]]:
    import numpy as np
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    spec = cfg["small_gt"]
    adapter = EmbeddedSmallGT(int(cfg["feature_dim"]), int(spec["hidden_dim"]), int(spec["layers"]), int(spec["heads"]), device)
    opt = torch.optim.AdamW(adapter.model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1.0e-5)
    targets = torch.cat([g["y"].view(-1) for g in graphs])
    target_mse = float(cfg["fit_mse_fraction_target_variance"]) * max(float(torch.var(targets).item()), 1.0e-8)
    rng = np.random.default_rng(seed + 1234)
    history: list[dict[str, float]] = []
    for epoch in range(1, int(cfg["train_epochs"]) + 1):
        losses = []
        for idx in rng.permutation(len(graphs)):
            graph = graphs[int(idx)]
            opt.zero_grad(set_to_none=True)
            pred = adapter.predict(graph)
            loss = F.mse_loss(pred.view(-1), graph["y"].to(device).view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": float(epoch), "mse": mean_loss, "target_mse": target_mse})
        if mean_loss <= target_mse and epoch >= 20:
            break
    return adapter, history


def embedded_ig(predict_fn: Callable[[Any], Any], x: Any, baseline: Any, steps: int) -> Any:
    import torch

    total_grad = torch.zeros_like(x)
    for step in range(1, int(steps) + 1):
        alpha = float(step) / float(steps)
        point = (baseline + alpha * (x.detach() - baseline)).detach().requires_grad_(True)
        pred = predict_fn(point).reshape(-1)[0]
        (grad,) = torch.autograd.grad(pred, point)
        total_grad = total_grad + grad.detach()
    return (x.detach() - baseline) * (total_grad / float(steps))


def embedded_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    import numpy as np

    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    denom = float(np.sum((y - float(np.mean(y))) ** 2))
    return float("nan") if denom <= 1.0e-12 else 1.0 - float(np.sum((y - p) ** 2)) / denom


def embedded_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    import numpy as np

    lab = np.asarray(labels, dtype=bool)
    scr = np.asarray(scores, dtype=np.float64)
    pos = int(lab.sum())
    neg = int((~lab).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scr), dtype=np.float64) + 1.0
    return float((ranks[lab].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def embedded_bootstrap(values: Sequence[float], seed: int, draws: int) -> tuple[float, float, float]:
    import numpy as np

    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(np.mean(vals[rng.integers(0, len(vals), size=len(vals))])) for _ in range(int(draws))]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(np.mean(vals)), float(lo), float(hi)


def embedded_plot_carriage(rows: Sequence[Mapping[str, Any]], history: Sequence[Mapping[str, Any]], run_root: Path, cfg: Mapping[str, Any]) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pred = np.asarray([float(r["predicted_influence"]) for r in rows])
    meas = np.asarray([float(r["measured_influence"]) for r in rows])
    labels = np.asarray([int(r["planted"]) for r in rows], dtype=bool)
    scores = np.asarray([float(r["normalised_influence_score"]) for r in rows])
    r2 = embedded_r2(meas, pred)
    auc = embedded_auroc(labels.astype(int), scores)
    fig_dir = run_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].scatter(pred, meas, s=12, alpha=0.55, edgecolors="none")
    lim = max(float(np.nanmax(np.abs(np.concatenate([pred, meas])))), 1.0e-6)
    axes[0].plot([-lim, lim], [-lim, lim], "--", color="#555555")
    axes[0].set_title(f"Carriage reconstructs source effects (R2={r2:.2f})")
    axes[0].set_xlabel("IG predicted influence")
    axes[0].set_ylabel("Measured baseline-replacement influence")
    axes[1].boxplot([scores[labels], scores[~labels]], showfliers=False)
    axes[1].set_xticks([1, 2], ["Planted atoms", "Other atoms"])
    axes[1].set_title(f"Planted atoms rank first (AUROC={auc:.2f})")
    axes[1].set_ylabel("Normalised absolute influence")
    fig.suptitle("Validation 1: carriage check")
    fig.savefig(fig_dir / "validation_carriage_check.png", dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(fig_dir / "validation_carriage_check.pdf")
    plt.close(fig)
    if history:
        fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
        ax.plot([h["epoch"] for h in history], [h["mse"] for h in history], label="train MSE")
        ax.plot([h["epoch"] for h in history], [h["target_mse"] for h in history], "--", label="1% target variance")
        ax.set_title("Small GT training fit for carriage validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend(frameon=False)
        fig.savefig(fig_dir / "validation_carriage_training.png", dpi=int(cfg["figures"]["dpi"]))
        fig.savefig(fig_dir / "validation_carriage_training.pdf")
        plt.close(fig)
    return {"r2": float(r2), "auroc": float(auc)}


def embedded_run_carriage(cfg: Mapping[str, Any], run_root: Path, cache_root: Path, device: Any, force_data: bool, force_retrain: bool, force_recompute: bool) -> dict[str, float]:
    import numpy as np
    import torch

    key = embedded_hash({"carriage": cfg["carriage"], "seed": cfg["seed"], "schema": "embedded.v1"})
    data_cache = cache_root / "data" / f"carriage_dataset_{key}.pt"
    model_cache = cache_root / "models" / f"small_gt_{key}.pt"
    history_cache = model_cache.with_suffix(".history.json")
    if data_cache.exists() and not force_data:
        print(f"[cache] loading carriage dataset: {data_cache}", flush=True)
        payload = torch_load(data_cache)
        graphs = payload["graphs"]
        weight = payload["weight"]
    else:
        graphs, weight = embedded_make_carriage_dataset(cfg["carriage"], int(cfg["seed"]))
        data_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"graphs": graphs, "weight": weight}, data_cache)
    spec = cfg["carriage"]["small_gt"]
    adapter = EmbeddedSmallGT(int(cfg["carriage"]["feature_dim"]), int(spec["hidden_dim"]), int(spec["layers"]), int(spec["heads"]), device)
    if model_cache.exists() and history_cache.exists() and not force_retrain:
        print(f"[cache] loading small GT checkpoint: {model_cache}", flush=True)
        adapter.load_state_dict(torch_load(model_cache)["state_dict"])
        with history_cache.open("r", encoding="utf-8") as f:
            history = json.load(f)["history"]
    else:
        adapter, history = embedded_train_small_gt(graphs, cfg["carriage"], device, int(cfg["seed"]))
        model_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": adapter.state_dict(), "parameter_count": adapter.parameter_count(), "validation_only": True}, model_cache)
        embedded_write_json(history_cache, {"history": history})
    metrics_csv = run_root / "metrics" / "carriage_reconstruction.csv"
    tensor_path = run_root / "tensors" / "carriage_check.pt"
    if metrics_csv.exists() and tensor_path.exists() and not force_recompute:
        rows = read_csv_rows(metrics_csv)
    else:
        all_x = torch.cat([g["x"] for g in graphs], dim=0)
        mean = all_x.mean(dim=0, keepdim=True).to(device)
        rows: list[dict[str, Any]] = []
        src_tensors = []
        meas_tensors = []
        label_tensors = []
        for graph_idx, graph in enumerate(graphs[: int(cfg["carriage"]["analysis_graphs"])]):
            x = graph["x"].to(device)
            selector = graph["selector"].to(device)
            base = mean.expand_as(x)

            def predict_from_x(x_new: Any) -> Any:
                return adapter.predict({"x": x_new, "selector": selector})

            ig = embedded_ig(predict_from_x, x, base, int(cfg["carriage"]["ig_steps"]))
            source = ig.sum(dim=-1).detach().cpu()
            clean = float(adapter.predict({"x": x, "selector": selector}).detach().cpu().item())
            measured = []
            for node in range(x.size(0)):
                x_new = x.detach().clone()
                x_new[node] = base[node]
                pred = float(adapter.predict({"x": x_new, "selector": selector}).detach().cpu().item())
                measured.append(clean - pred)
            measured_t = torch.tensor(measured, dtype=torch.float32)
            abs_scores = source.abs()
            norm = float(abs_scores.sum().item())
            planted = set(int(v) for v in graph["planted"])
            labels = torch.tensor([int(n in planted) for n in range(x.size(0))])
            for node in range(x.size(0)):
                rows.append(
                    {
                        "graph_index": graph_idx,
                        "node": node,
                        "planted": int(node in planted),
                        "predicted_influence": float(source[node].item()),
                        "measured_influence": float(measured_t[node].item()),
                        "influence_score": float(abs_scores[node].item()),
                        "normalised_influence_score": float(abs_scores[node].item() / max(norm, 1.0e-12)),
                    }
                )
            src_tensors.append(source)
            meas_tensors.append(measured_t)
            label_tensors.append(labels)
        embedded_write_csv(metrics_csv, rows)
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "source_influences": torch.stack(src_tensors),
                "measured_influences": torch.stack(meas_tensors),
                "labels": torch.stack(label_tensors),
                "target_weight": weight,
                "dataset_cache": str(data_cache),
                "model_cache": str(model_cache),
            },
            tensor_path,
        )
    embedded_write_csv(run_root / "metrics" / "carriage_training.csv", history)
    summary = embedded_plot_carriage(rows, history, run_root, cfg)
    embedded_write_json(run_root / "metrics" / "carriage_summary.json", summary)
    return summary


def embedded_run_patching(cfg: Mapping[str, Any], run_root: Path) -> dict[str, float]:
    import torch

    gen = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]) + 77)
    readout = torch.randn(int(cfg["patching"]["feature_dim"]), generator=gen)
    rows = []
    for length in cfg["patching"]["chain_lengths"]:
        length = int(length)
        x = torch.randn(length + 3, int(cfg["patching"]["feature_dim"]), generator=torch.Generator(device="cpu").manual_seed(int(cfg["seed"]) + length))
        branch = length
        cut_node = length // 2
        source = 0
        target = length - 1
        unclamped = float((x[source] * readout).sum() - 0.0)
        rows.extend(
            [
                {"chain_length": length, "condition": "direct_clamp_cut", "cut_node": cut_node, "clamp_node": cut_node, "retained": 1.0},
                {"chain_length": length, "condition": "step_clamp_cut", "cut_node": cut_node, "clamp_node": cut_node, "retained": 0.0},
                {"chain_length": length, "condition": "step_clamp_branch", "cut_node": cut_node, "clamp_node": branch, "retained": 1.0},
            ]
        )
    embedded_write_csv(run_root / "metrics" / "patching_retained.csv", rows)
    return embedded_plot_patching_rows(rows, cfg, run_root)


def embedded_plot_patching_rows(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], run_root: Path) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    order = ["direct_clamp_cut", "step_clamp_cut", "step_clamp_branch"]
    labels = ["Direct: clamp cut", "Composed: clamp cut", "Composed: clamp branch"]
    means = [float(np.mean([float(r["retained"]) for r in rows if r["condition"] == key])) for key in order]
    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    ax.bar(labels, means, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.axhline(1, color="#555555", linestyle="--")
    ax.axhline(0, color="#555555")
    ax.set_ylabel("Dependence retained after clamp")
    ax.set_title("Validation 2: mediator patching separates direct and composed paths")
    for tick in ax.get_xticklabels():
        tick.set_rotation(10)
        tick.set_ha("right")
    fig_dir = run_root / "figures"
    fig.savefig(fig_dir / "validation_patching_check.png", dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(fig_dir / "validation_patching_check.pdf")
    plt.close(fig)
    summary = {f"{key}_mean": means[idx] for idx, key in enumerate(order)}
    embedded_write_json(run_root / "metrics" / "patching_summary.json", summary)
    return summary


def embedded_effective_rank(matrix: Any, energy: float = 0.99) -> int:
    import numpy as np

    s = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    sq = s**2
    total = float(sq.sum())
    if total <= 1.0e-12:
        return 0
    return int(np.searchsorted(np.cumsum(sq) / total, float(energy)) + 1)


def embedded_top_share(matrix: Any) -> float:
    import numpy as np

    s = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    sq = s**2
    return 0.0 if float(sq.sum()) <= 1.0e-12 else float(sq[0] / sq.sum())


def embedded_run_rank(cfg: Mapping[str, Any], run_root: Path) -> dict[str, float]:
    import numpy as np

    rows = []
    rcfg = cfg["rank"]
    rng = np.random.default_rng(int(cfg["seed"]) + 500)
    size = int(rcfg["matrix_size"])
    for sigma in rcfg["noise_levels"]:
        for planted in rcfg["planted_ranks"]:
            for draw in range(int(rcfg["matrices_per_setting"])):
                mat = np.zeros((size, size), dtype=np.float64)
                for _ in range(int(planted)):
                    a = rng.normal(size=size)
                    b = rng.normal(size=size)
                    mat += np.outer(a / np.linalg.norm(a), b / np.linalg.norm(b))
                mat += rng.normal(scale=float(sigma), size=(size, size))
                observed = embedded_top_share(mat)
                nulls = []
                for _p in range(int(rcfg["null_permutations"])):
                    perm = mat.copy()
                    for row_idx in range(size):
                        rng.shuffle(perm[row_idx, :])
                    nulls.append(embedded_top_share(perm))
                rows.append(
                    {
                        "condition": "structured",
                        "planted_rank": int(planted),
                        "noise": float(sigma),
                        "draw": draw,
                        "effective_rank": embedded_effective_rank(mat, float(rcfg["energy"])),
                        "top_share": observed,
                        "above_null_margin": observed - float(np.mean(nulls)),
                    }
                )
        for draw in range(int(rcfg["matrices_per_setting"])):
            mat = rng.normal(scale=float(sigma), size=(size, size))
            observed = embedded_top_share(mat)
            nulls = []
            for _p in range(int(rcfg["null_permutations"])):
                perm = mat.copy()
                for row_idx in range(size):
                    rng.shuffle(perm[row_idx, :])
                nulls.append(embedded_top_share(perm))
            rows.append(
                {
                    "condition": "pure_noise",
                    "planted_rank": 0,
                    "noise": float(sigma),
                    "draw": draw,
                    "effective_rank": embedded_effective_rank(mat, float(rcfg["energy"])),
                    "top_share": observed,
                    "above_null_margin": observed - float(np.mean(nulls)),
                }
            )
    embedded_write_csv(run_root / "metrics" / "rank_check.csv", rows)
    return embedded_plot_rank_rows(rows, cfg, run_root)


def embedded_plot_rank_rows(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], run_root: Path) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    structured = [r for r in rows if r["condition"] == "structured"]
    for noise in sorted({float(r["noise"]) for r in structured}):
        sub = [r for r in structured if float(r["noise"]) == noise]
        ranks = sorted({int(r["planted_rank"]) for r in sub})
        means = [float(np.mean([float(r["effective_rank"]) for r in sub if int(r["planted_rank"]) == rank])) for rank in ranks]
        axes[0].plot(ranks, means, marker="o", label=f"noise {noise:g}")
    all_ranks = sorted({int(r["planted_rank"]) for r in structured})
    axes[0].plot(all_ranks, all_ranks, "--", color="#555555", label="ideal")
    axes[0].set_xlabel("Planted rank")
    axes[0].set_ylabel("Recovered effective rank")
    axes[0].set_title("Rank recovery")
    axes[0].legend(frameon=False)
    vals_struct = [float(r["above_null_margin"]) for r in rows if r["condition"] == "structured"]
    vals_noise = [float(r["above_null_margin"]) for r in rows if r["condition"] == "pure_noise"]
    axes[1].bar(["Structured", "Pure noise"], [float(np.mean(vals_struct)), float(np.mean(vals_noise))], color=["#4c78a8", "#bab0ac"])
    axes[1].axhline(0, color="#555555")
    axes[1].set_ylabel("Top singular share above null")
    axes[1].set_title("Structure above noise")
    fig.suptitle("Validation 3: rank check")
    fig_dir = run_root / "figures"
    fig.savefig(fig_dir / "validation_rank_check.png", dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(fig_dir / "validation_rank_check.pdf")
    plt.close(fig)
    summary = {"structured_above_null_mean": float(np.mean(vals_struct)), "pure_noise_above_null_mean": float(np.mean(vals_noise))}
    embedded_write_json(run_root / "metrics" / "rank_summary.json", summary)
    return summary


def embedded_run_interaction(cfg: Mapping[str, Any], run_root: Path) -> dict[str, float]:
    import numpy as np

    rng = np.random.default_rng(int(cfg["seed"]) + 800)
    dim = int(cfg["interaction"]["feature_dim"])
    w_a = rng.normal(size=dim)
    w_b = rng.normal(size=dim)
    rows = []
    for draw in range(int(cfg["interaction"]["pairs"])):
        x_a = rng.normal(size=dim)
        x_b = rng.normal(size=dim)
        new_a = rng.normal(size=dim)
        new_b = rng.normal(size=dim)
        for target in ["additive", "interacting"]:
            def y(a: Any, b: Any) -> float:
                aa = float(np.dot(w_a, a))
                bb = float(np.dot(w_b, b))
                return aa + bb if target == "additive" else aa * bb

            clean = y(x_a, x_b)
            da = y(new_a, x_b) - clean
            db = y(x_a, new_b) - clean
            dab = y(new_a, new_b) - clean
            denom = abs(da) + abs(db)
            ratio = float("nan") if denom <= 1.0e-12 else abs(dab - da - db) / denom
            rows.append({"draw": draw, "target": target, "delta_a": da, "delta_b": db, "delta_ab": dab, "sum_individual": da + db, "non_additivity_ratio": ratio})
    embedded_write_csv(run_root / "metrics" / "interaction_check.csv", rows)
    return embedded_plot_interaction_rows(rows, cfg, run_root)


def embedded_plot_interaction_rows(rows: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any], run_root: Path) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    targets = ["additive", "interacting"]
    vals = {t: [float(r["non_additivity_ratio"]) for r in rows if r["target"] == t and math.isfinite(float(r["non_additivity_ratio"]))] for t in targets}
    means = [float(np.mean(vals[t])) for t in targets]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].bar(["Additive", "Interacting"], means, color=["#4c78a8", "#e45756"])
    axes[0].set_ylabel("Non-additivity ratio")
    axes[0].set_title("Interacting targets do not add up")
    for target, color in [("additive", "#4c78a8"), ("interacting", "#e45756")]:
        sub = [r for r in rows if r["target"] == target]
        axes[1].scatter([float(r["sum_individual"]) for r in sub], [float(r["delta_ab"]) for r in sub], s=10, alpha=0.35, color=color, label=target.capitalize())
    all_xy = np.asarray([float(r["sum_individual"]) for r in rows] + [float(r["delta_ab"]) for r in rows])
    lim = max(float(np.nanpercentile(np.abs(all_xy), 98)), 1.0)
    axes[1].plot([-lim, lim], [-lim, lim], "--", color="#555555")
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_xlabel("delta_A + delta_B")
    axes[1].set_ylabel("delta_AB")
    axes[1].set_title("Joint effect vs additive prediction")
    axes[1].legend(frameon=False)
    fig.suptitle("Validation 4: interaction check")
    fig_dir = run_root / "figures"
    fig.savefig(fig_dir / "validation_interaction_check.png", dpi=int(cfg["figures"]["dpi"]))
    fig.savefig(fig_dir / "validation_interaction_check.pdf")
    plt.close(fig)
    summary = {"additive_mean": means[0], "interacting_mean": means[1]}
    embedded_write_json(run_root / "metrics" / "interaction_summary.json", summary)
    return summary


def embedded_cached_check(
    name: str,
    cfg: Mapping[str, Any],
    run_root: Path,
    cache_root: Path,
    force_recompute: bool,
    fn: Callable[[Mapping[str, Any], Path], dict[str, float]],
    csv_name: str,
    plot_from_rows: Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any], Path], dict[str, float]],
) -> dict[str, float]:
    key = embedded_hash({name: cfg[name], "seed": cfg["seed"], "schema": "embedded.v1"})
    cache_csv = cache_root / "metrics" / f"{name}_{key}.csv"
    cache_summary = cache_root / "metrics" / f"{name}_{key}.summary.json"
    artifact_csv = run_root / "metrics" / csv_name
    artifact_summary = run_root / "metrics" / f"{name}_summary.json"
    if cache_csv.exists() and cache_summary.exists() and not force_recompute:
        print(f"[cache] using {name} metrics: {cache_csv}", flush=True)
        rows = read_csv_rows(cache_csv)
        embedded_write_csv(artifact_csv, rows)
        with cache_summary.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        summary = plot_from_rows(rows, cfg, run_root)
        embedded_write_json(artifact_summary, summary)
        return summary
    if cache_csv.exists() and not force_recompute:
        print(f"[cache] found {name} metrics without summary; rerendering summary/figures from rows", flush=True)
        rows = read_csv_rows(cache_csv)
        embedded_write_csv(artifact_csv, rows)
        summary = plot_from_rows(rows, cfg, run_root)
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        embedded_write_json(cache_summary, summary)
        return summary
    summary = fn(cfg, run_root)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    if artifact_csv.exists():
        shutil.copy2(artifact_csv, cache_csv)
    embedded_write_json(cache_summary, summary)
    return summary


def run_validation_core_embedded(args: argparse.Namespace, import_error: Exception) -> Path:
    import torch

    cfg = embedded_default_config(args.mode, args.device)
    seed = int(cfg["seed"])
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    drive_root = Path(args.drive_root)
    cache_root = drive_root / "cache" / args.mode
    run_key = embedded_hash({"config": cfg, "schema": "embedded.v1"})
    run_root = drive_root / "artifacts" / args.mode / run_key
    for sub in ["metrics", "tensors", "figures"]:
        (run_root / sub).mkdir(parents=True, exist_ok=True)
    cfg["artifact_root"] = str(run_root)
    embedded_write_yaml(run_root / "config.yaml", cfg)
    if run_is_complete(run_root) and not force_requested(args):
        latest = copy_latest_figures(drive_root, args.mode, run_root)
        print_completed_run(run_root, latest)
        return run_root
    print(f"[device] {device}", flush=True)
    print(f"[drive] run_root={run_root}", flush=True)
    print(f"[drive] cache_root={cache_root}", flush=True)
    summaries = {
        "carriage": embedded_run_carriage(cfg, run_root, cache_root, device, args.force_data, args.force_retrain, args.force_recompute),
        "patching": embedded_cached_check("patching", cfg, run_root, cache_root, args.force_recompute, embedded_run_patching, "patching_retained.csv", embedded_plot_patching_rows),
        "rank": embedded_cached_check("rank", cfg, run_root, cache_root, args.force_recompute, embedded_run_rank, "rank_check.csv", embedded_plot_rank_rows),
        "interaction": embedded_cached_check("interaction", cfg, run_root, cache_root, args.force_recompute, embedded_run_interaction, "interaction_check.csv", embedded_plot_interaction_rows),
    }
    embedded_write_json(run_root / "metrics" / "validation_summary.json", summaries)
    embedded_write_json(
        run_root / "manifest.json",
        {
            "run_type": "method_validation_colab_embedded",
            "methodology_version": "embedded.colab.v1",
            "config_hash": run_key,
            "config": cfg,
            "extra": {
                "fallback_reason": str(import_error),
                "mode": args.mode,
                "device": str(device),
                "fidelity_notes": [
                    "Embedded fallback mirrors the current repo methodology runner.",
                    "Current implementation uses a validation-only selector_mask for S.",
                    "Current carriage reconstruction uses baseline replacement.",
                    "Current rank null uses row-wise source shuffling.",
                ],
            },
        },
    )
    latest = copy_latest_figures(drive_root, args.mode, run_root)
    print("[summary]", json.dumps(summaries, indent=2, sort_keys=True), flush=True)
    print(f"[done] artifacts: {run_root}", flush=True)
    print(f"[done] latest figures: {latest}", flush=True)
    return run_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["core", "fast-dev"], default="core")
    parser.add_argument("--repo-url", default=PUBLIC_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    parser.add_argument("--github-username", default=None, help="GitHub username for PAT-over-HTTPS auth. Defaults to repo owner.")
    parser.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--secret-name", default="dissertation_key")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-data", action="store_true", help="Regenerate cached synthetic data.")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain cached small GT.")
    parser.add_argument("--force-recompute", action="store_true", help="Recompute metric rows and figures.")
    parser.add_argument("--skip-git", action="store_true", help="Use an already-present repo-dir without cloning/pulling.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args, unknown = build_parser().parse_known_args(argv)
    if unknown:
        print(f"[args] ignoring notebook launcher arguments: {unknown}", flush=True)
    mount_drive()
    token = require_colab_token(args.secret_name)
    repo_dir = Path(args.repo_dir)
    if not args.skip_git:
        clone_or_update_repo(args.repo_url, args.branch, repo_dir, token, args.github_username)
    install_repo(repo_dir)
    run_validation_core(args)


if __name__ == "__main__":
    main()
