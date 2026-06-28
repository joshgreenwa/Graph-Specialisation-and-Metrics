#!/usr/bin/env python3
"""Standalone Google Colab runner for method-validation synthetics.

Run this file in Colab.  It mounts Google Drive, clones/pulls the GitHub repo
using the Colab secret named ``diss_key``, installs the repo in editable mode,
and runs the synthetic method-validation suite with Drive-backed caches.

Default mode is the core synthetic run from ``method_validation.md``.  Use
``--mode fast-dev`` only for a quick smoke test.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


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
) -> subprocess.CompletedProcess[str]:
    display = safe_display or " ".join(cmd)
    print(f"[cmd] {display}", flush=True)
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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
            "Create a Colab secret named diss_key and run it there."
        ) from exc
    token = userdata.get(secret_name)
    if not token:
        raise RuntimeError(f"Colab secret {secret_name!r} is missing or empty")
    return str(token)


def mount_drive() -> None:
    try:
        from google.colab import drive
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Google Drive mounting requires Google Colab") from exc
    drive.mount("/content/drive", force_remount=False)


def auth_url(public_url: str, token: str) -> str:
    if not public_url.startswith("https://github.com/"):
        raise ValueError("only https://github.com repo URLs are supported by this runner")
    return public_url.replace("https://github.com/", f"https://x-access-token:{token}@github.com/")


def clone_or_update_repo(repo_url: str, branch: str, repo_dir: Path, token: str) -> None:
    repo_dir = Path(repo_dir)
    authed = auth_url(repo_url, token)
    if (repo_dir / ".git").exists():
        run_cmd(
            ["git", "-C", str(repo_dir), "remote", "set-url", "origin", authed],
            safe_display=f"git -C {repo_dir} remote set-url origin <authenticated-url>",
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
        safe_display=f"git clone --branch {branch} <authenticated-url> {repo_dir}",
    )
    run_cmd(["git", "-C", str(repo_dir), "remote", "set-url", "origin", repo_url])


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

    latest = drive_root / "latest_figures" / args.mode
    if latest.exists():
        shutil.rmtree(latest)
    latest.mkdir(parents=True, exist_ok=True)
    for figure in sorted((run_root / "figures").glob("*")):
        if figure.is_file():
            shutil.copy2(figure, latest / figure.name)

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
    parser.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--secret-name", default="diss_key")
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
        clone_or_update_repo(args.repo_url, args.branch, repo_dir, token)
    install_repo(repo_dir)
    run_validation_core(args)


if __name__ == "__main__":
    main()
