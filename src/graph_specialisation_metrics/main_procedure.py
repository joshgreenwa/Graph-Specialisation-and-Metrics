"""Main dissertation procedure runner.

This module provides the reusable CLI and artifact contract for Steps 0-5.  The
heavy GRIT intervention hooks are intentionally strict: dry runs and discovery
work now, but paper-claim interventions require official checkpoints/adapters.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graph_specialisation_metrics.method_adapters import OfficialGRITAdapter, parameter_count_close
from graph_specialisation_metrics.method_core import (
    AdapterInfo,
    config_hash,
    ensure_dir,
    file_sha256,
    read_yaml,
    write_csv,
    write_json,
    write_manifest,
    write_yaml,
)
from graph_specialisation_metrics.grit_intervention_procedure import run_intervention_steps


METHOD_VALIDATION_MD = Path("/Users/joshgreen/Downloads/method_validation.md")
MAIN_PROCEDURE_MD = Path("/Users/joshgreen/Downloads/dissertation_core_procedure.md")
SOURCE_MARKDOWN_EXPECTED_SHA256 = {
    "method_validation.md": "58e85e0897199e03b8f757f92eb0cb6f13ab3c8f571498834732bb3cb464165a",
    "dissertation_core_procedure.md": "cf567856da55d780b6eb5a10f51e94e8448ac0c4ad1959dcea8839a5127c525f",
}


DEFAULT_CONFIG: dict[str, Any] = {
    "artifact_root": "artifacts/main_procedure",
    "dataset": {"name": "ZINC", "split": "official_subset", "task": "molecular_regression"},
    "seeds": [41, 42, 43],
    "primary_tau": 3,
    "far_thresholds": [2, 3, 4],
    "perturbation": {
        "carriage_primary": "integrated_gradients",
        "ig_baseline": "mean_node_embedding",
        "ig_steps": 32,
        "swap_partners": 8,
    },
    "models": {
        "dense_grit": {
            "adapter": "official_grit",
            "variant": "official",
            "role": "treatment",
            "official_repo": "https://github.com/LiamMa/GRIT",
            "repo_path": "external/GRIT",
            "artifact_root": "/rds/user/jgg45/hpc-work/grit_zinc_results/official",
            "config_path": None,
            "checkpoint_path": None,
        },
        "grit_1hop": {
            "adapter": "official_grit",
            "variant": "1hop",
            "role": "parameter_matched_control",
            "official_repo": "https://github.com/LiamMa/GRIT",
            "repo_path": "external/GRIT",
            "artifact_root": "/rds/user/jgg45/hpc-work/grit_zinc_results/1hop",
            "config_path": None,
            "checkpoint_path": None,
        },
        "gin": {
            "adapter": "pyg_gin",
            "role": "local_validation_reference",
            "artifact_root": "/rds/user/jgg45/hpc-work/grit_zinc_results/gin_reference",
        },
        "gcn": {
            "adapter": "pyg_gcn",
            "role": "local_validation_reference",
            "artifact_root": "/rds/user/jgg45/hpc-work/grit_zinc_results/gcn_reference",
        },
    },
    "steps": {
        "0": {"name": "measurement_model_validation", "sample_graphs": 200},
        "1": {"name": "performance_gap", "reach_sweep": [1, 2, 3, 5, "dense"]},
        "2": {"name": "usage_vs_causal_usage", "sample_graphs": 200},
        "3": {"name": "distance_resolved_overfitting", "sample_graphs": 200},
        "4": {"name": "mediator_patching", "sample_graphs": 100, "max_far_pairs_per_graph": 64},
        "5": {"name": "non_composable_gap_attribution", "sample_graphs": 200, "interaction_pairs": 1000},
    },
    "figures": {"dpi": 180},
}


FAST_DEV_OVERRIDES: dict[str, Any] = {
    "artifact_root": "artifacts/main_procedure_fast_dev",
    "seeds": [41],
    "steps": {
        "0": {"sample_graphs": 8},
        "1": {"reach_sweep": [1, "dense"]},
        "2": {"sample_graphs": 8},
        "3": {"sample_graphs": 8},
        "4": {"sample_graphs": 4, "max_far_pairs_per_graph": 4},
        "5": {"sample_graphs": 8, "interaction_pairs": 32},
    },
}


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_steps(raw: str | None) -> list[str]:
    if raw is None or raw.strip() in {"", "all"}:
        return [str(i) for i in range(6)]
    out = [item.strip() for item in raw.split(",") if item.strip()]
    bad = [item for item in out if item not in {str(i) for i in range(6)}]
    if bad:
        raise ValueError(f"unknown step ids {bad}; expected 0,1,2,3,4,5")
    return out


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def find_files(root: Path, patterns: Sequence[str]) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(set(files))


def discover_model_artifacts(model_name: str, model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(model_cfg.get("artifact_root") or ".")).expanduser()
    config_path = model_cfg.get("config_path")
    checkpoint_path = model_cfg.get("checkpoint_path")
    configs = [Path(config_path)] if config_path else find_files(root, ["*.yaml", "*.yml", "config.json"])
    checkpoints = [Path(checkpoint_path)] if checkpoint_path else find_files(
        root,
        ["*checkpoint*.pt", "*checkpoint*.pth", "*.ckpt", "best*.pt", "best*.pth", "model*.pt"],
    )
    stats = find_files(root, ["*stats*.json", "*metrics*.json", "*history*.json", "*stats*.csv", "*metrics*.csv", "*history*.csv"])
    return {
        "model": model_name,
        "role": model_cfg.get("role"),
        "adapter": model_cfg.get("adapter"),
        "variant": model_cfg.get("variant"),
        "artifact_root": str(root),
        "exists": root.exists(),
        "config_candidates": [str(path) for path in configs[:10] if path.exists()],
        "checkpoint_candidates": [str(path) for path in checkpoints[:10] if path.exists()],
        "stats_candidates": [str(path) for path in stats[:20] if path.exists()],
        "num_configs": len([p for p in configs if p.exists()]),
        "num_checkpoints": len([p for p in checkpoints if p.exists()]),
        "num_stats": len([p for p in stats if p.exists()]),
    }


def extract_metric_from_payload(payload: Any, keys: Sequence[str]) -> float:
    if isinstance(payload, Mapping):
        for key in keys:
            if key in payload:
                return safe_float(payload[key])
        for value in payload.values():
            found = extract_metric_from_payload(value, keys)
            if math.isfinite(found):
                return found
    if isinstance(payload, list):
        for item in reversed(payload):
            found = extract_metric_from_payload(item, keys)
            if math.isfinite(found):
                return found
    return float("nan")


def extract_history_rows(model: str, stats_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        if stats_path.suffix.lower() == ".csv":
            raw_rows = read_csv_rows(stats_path)
            for row in raw_rows:
                step = safe_float(row.get("epoch", row.get("step", row.get("iteration", len(rows)))))
                train = extract_metric_from_payload(row, ["train_loss", "loss_train", "train_mae", "train"])
                val = extract_metric_from_payload(row, ["val_loss", "valid_loss", "val_mae", "valid_mae", "val"])
                if math.isfinite(train) or math.isfinite(val):
                    rows.append({"model": model, "step": step, "train": train, "val": val, "source": str(stats_path)})
        else:
            payload = read_json(stats_path)
            history = payload.get("history", payload.get("epochs", payload.get("stats", payload))) if isinstance(payload, Mapping) else payload
            if isinstance(history, list):
                for idx, item in enumerate(history):
                    step = safe_float(item.get("epoch", item.get("step", idx))) if isinstance(item, Mapping) else idx
                    train = extract_metric_from_payload(item, ["train_loss", "loss_train", "train_mae", "train"])
                    val = extract_metric_from_payload(item, ["val_loss", "valid_loss", "val_mae", "valid_mae", "val"])
                    if math.isfinite(train) or math.isfinite(val):
                        rows.append({"model": model, "step": step, "train": train, "val": val, "source": str(stats_path)})
    except Exception:
        return rows
    return rows


def extract_test_metric(model: str, stats_path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(stats_path) if stats_path.suffix.lower() != ".csv" else read_csv_rows(stats_path)
    except Exception:
        return None
    test = extract_metric_from_payload(payload, ["test_mae", "mae_test", "test_loss", "test", "best_test_mae"])
    val = extract_metric_from_payload(payload, ["val_mae", "valid_mae", "best_val_mae", "val_loss", "valid_loss"])
    train = extract_metric_from_payload(payload, ["train_mae", "train_loss"])
    if not any(math.isfinite(v) for v in [test, val, train]):
        return None
    seed = None
    match = re.search(r"(?:seed|s)(\d+)", str(stats_path), flags=re.IGNORECASE)
    if match:
        seed = int(match.group(1))
    return {"model": model, "seed": seed, "test_metric": test, "val_metric": val, "train_metric": train, "source": str(stats_path)}


def run_step_1(discovery: Sequence[Mapping[str, Any]], artifact_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for entry in discovery:
        model = str(entry["model"])
        for raw_path in entry.get("stats_candidates", []):
            path = Path(str(raw_path))
            metric = extract_test_metric(model, path)
            if metric is not None:
                metric_rows.append(metric)
            history_rows.extend(extract_history_rows(model, path))
    write_csv(artifact_root / "metrics" / "step1_test_metrics.csv", metric_rows)
    write_csv(artifact_root / "metrics" / "step1_training_history.csv", history_rows)
    render_step_1_figures(metric_rows, history_rows, artifact_root, config)
    return {
        "status": "complete" if metric_rows else "waiting_for_training_artifacts",
        "test_metric_rows": len(metric_rows),
        "history_rows": len(history_rows),
    }


def render_step_1_figures(
    metric_rows: Sequence[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
    config: Mapping[str, Any],
) -> None:
    figures = ensure_dir(artifact_root / "figures")
    dpi = int(config["figures"]["dpi"])
    if metric_rows:
        grouped: dict[str, list[float]] = {}
        for row in metric_rows:
            value = safe_float(row.get("test_metric"))
            if math.isfinite(value):
                grouped.setdefault(str(row["model"]), []).append(value)
        if grouped:
            models = sorted(grouped)
            means = [float(np.mean(grouped[m])) for m in models]
            stds = [float(np.std(grouped[m], ddof=1)) if len(grouped[m]) > 1 else 0.0 for m in models]
            fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
            ax.bar(models, means, yerr=stds, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2"][: len(models)], capsize=4)
            ax.set_title("Step 1: test error by model")
            ax.set_ylabel("Test metric (lower is better)")
            ax.set_xlabel("Model")
            for tick in ax.get_xticklabels():
                tick.set_rotation(15)
                tick.set_ha("right")
            fig.savefig(figures / "step1_test_error_dense_vs_1hop.png", dpi=dpi)
            fig.savefig(figures / "step1_test_error_dense_vs_1hop.pdf")
            plt.close(fig)
    if history_rows:
        fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
        by_model: dict[str, list[Mapping[str, Any]]] = {}
        for row in history_rows:
            by_model.setdefault(str(row["model"]), []).append(row)
        for model, rows in sorted(by_model.items()):
            rows_sorted = sorted(rows, key=lambda r: safe_float(r.get("step")))
            x = [safe_float(r.get("step")) for r in rows_sorted]
            train = [safe_float(r.get("train")) for r in rows_sorted]
            val = [safe_float(r.get("val")) for r in rows_sorted]
            if any(math.isfinite(v) for v in train):
                ax.plot(x, train, linewidth=1.5, label=f"{model} train")
            if any(math.isfinite(v) for v in val):
                ax.plot(x, val, linewidth=1.5, linestyle="--", label=f"{model} val")
        ax.set_title("Step 1: training and validation curves")
        ax.set_xlabel("Epoch/step")
        ax.set_ylabel("Loss/metric")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(figures / "step1_training_validation_loss.png", dpi=dpi)
        fig.savefig(figures / "step1_training_validation_loss.pdf")
        plt.close(fig)


def run_adapter_checks(config: Mapping[str, Any], discovery: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dense_cfg = config["models"].get("dense_grit", {})
    hop_cfg = config["models"].get("grit_1hop", {})
    dense_disc = next((d for d in discovery if d["model"] == "dense_grit"), {})
    hop_disc = next((d for d in discovery if d["model"] == "grit_1hop"), {})
    status: dict[str, Any] = {
        "dense_checkpoint_found": bool(dense_disc.get("checkpoint_candidates")),
        "onehop_checkpoint_found": bool(hop_disc.get("checkpoint_candidates")),
        "parameter_match_checked": False,
        "parameter_match": None,
    }
    if dense_disc.get("checkpoint_candidates") and hop_disc.get("checkpoint_candidates"):
        dense_config = dense_disc.get("config_candidates", [None])[0]
        hop_config = hop_disc.get("config_candidates", [None])[0]
        if dense_config and hop_config:
            try:
                dense = OfficialGRITAdapter(
                    repo_path=Path(str(dense_cfg.get("repo_path", "external/GRIT"))),
                    config_path=Path(str(dense_config)),
                    checkpoint_path=Path(str(dense_disc["checkpoint_candidates"][0])),
                    variant="official",
                    dataset_dir=Path(str(dense_cfg["dataset_dir"])) if dense_cfg.get("dataset_dir") else None,
                    device=str(config.get("device", "cpu")),
                    seed=int(config.get("seeds", [0])[0]),
                )
                onehop = OfficialGRITAdapter(
                    repo_path=Path(str(hop_cfg.get("repo_path", "external/GRIT"))),
                    config_path=Path(str(hop_config)),
                    checkpoint_path=Path(str(hop_disc["checkpoint_candidates"][0])),
                    variant="1hop",
                    dataset_dir=Path(str(hop_cfg["dataset_dir"])) if hop_cfg.get("dataset_dir") else None,
                    device=str(config.get("device", "cpu")),
                    seed=int(config.get("seeds", [0])[0]),
                )
                matched, dense_params, onehop_params = parameter_count_close(dense, onehop)
                status.update(
                    {
                        "parameter_match_checked": True,
                        "parameter_match": matched,
                        "dense_parameter_count": dense_params,
                        "onehop_parameter_count": onehop_params,
                    }
                )
            except Exception as exc:
                status["parameter_match_error"] = str(exc)
    return status


def write_step_status(step: str, artifact_root: Path, status: Mapping[str, Any]) -> None:
    write_json(artifact_root / "metrics" / f"step{step}_status.json", status)


def read_step_status(step: str, artifact_root: Path) -> dict[str, Any] | None:
    path = artifact_root / "metrics" / f"step{step}_status.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return dict(payload) if isinstance(payload, Mapping) else None
    except Exception:
        return None


def load_cached_step_statuses(steps: Sequence[str], artifact_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for step in steps:
        status = read_step_status(step, artifact_root)
        if status is not None:
            out[step] = status
    return out


def step_is_complete(status: Mapping[str, Any] | None) -> bool:
    return bool(status) and str(status.get("status")) == "complete"


def progress(message: str) -> None:
    print(f"[main:{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run_main(config: Mapping[str, Any], *, steps: Sequence[str], dry_run: bool = False, force: bool = False) -> Path:
    artifact_root = ensure_dir(Path(str(config["artifact_root"])) / config_hash(config))
    cached_statuses = {} if force else load_cached_step_statuses(steps, artifact_root)
    if (
        (artifact_root / "manifest.json").exists()
        and not force
        and not dry_run
        and all(step_is_complete(cached_statuses.get(step)) for step in steps)
    ):
        progress(f"using cached artifact root: {artifact_root}")
        return artifact_root
    progress(f"artifact root: {artifact_root}")
    ensure_dir(artifact_root / "metrics")
    ensure_dir(artifact_root / "tensors")
    ensure_dir(artifact_root / "figures")
    write_yaml(artifact_root / "config.yaml", config)

    progress("discovering model artifacts")
    discovery = [discover_model_artifacts(name, cfg) for name, cfg in config["models"].items()]
    write_json(artifact_root / "metrics" / "artifact_discovery.json", {"models": discovery})
    for item in discovery:
        progress(
            f"discovered {item.get('model')}: "
            f"{len(item.get('checkpoint_candidates', []))} checkpoint(s), "
            f"{len(item.get('config_candidates', []))} config(s)"
        )
    progress("running adapter checks")
    adapter_checks = run_adapter_checks(config, discovery)
    write_json(artifact_root / "metrics" / "adapter_checks.json", adapter_checks)
    progress(f"adapter checks complete: parameter_match={adapter_checks.get('parameter_match')}")

    status_by_step: dict[str, Any] = dict(cached_statuses)
    if dry_run:
        for step in steps:
            if not force and step_is_complete(status_by_step.get(step)):
                progress(f"Step {step} already complete; skipping dry-run rewrite")
                continue
            progress(f"dry-run status for Step {step}: {config['steps'][step]['name']}")
            status = {
                "status": "dry_run_only",
                "step": step,
                "name": config["steps"][step]["name"],
                "required_artifacts_discovered": discovery,
            }
            write_step_status(step, artifact_root, status)
            status_by_step[step] = status
            write_json(artifact_root / "metrics" / "main_status.json", status_by_step)
    else:
        intervention_steps = [step for step in steps if step in {"0", "2", "3", "4", "5"}]
        for step in steps:
            if not force and step_is_complete(status_by_step.get(step)):
                progress(f"Step {step} already complete; reusing cached outputs")
                continue
            if step == "1":
                progress("starting Step 1: performance_gap")
                status = run_step_1(discovery, artifact_root, config)
            elif step in intervention_steps:
                try:
                    progress(f"starting GRIT intervention Step {step}: {config['steps'][step]['name']}")
                    step_status = run_intervention_steps(config, discovery, artifact_root, [step])
                    status = step_status.get(step, {"status": "unknown_step", "step": step})
                    progress(f"GRIT intervention Step {step} finished")
                except Exception as exc:
                    progress(f"GRIT intervention Step {step} failed: {exc}")
                    status = {
                        "status": "failed",
                        "step": step,
                        "name": config["steps"][step]["name"],
                        "error": str(exc),
                        "methodology_core_available": True,
                    }
            else:  # pragma: no cover
                status = {"status": "unknown_step", "step": step}
            write_step_status(step, artifact_root, status)
            status_by_step[step] = status
            write_json(artifact_root / "metrics" / "main_status.json", status_by_step)
            progress(f"Step {step} status: {status.get('status')}")

    write_json(artifact_root / "metrics" / "main_status.json", status_by_step)
    progress("writing manifest")
    write_manifest(
        artifact_root,
        run_type="main_procedure",
        config=config,
        adapter=AdapterInfo(
            name="main_procedure_runner",
            version="main.v1",
            implementation="official-adapter-compatible analysis runner",
            validation_only=False,
        ),
        source_markdowns=[METHOD_VALIDATION_MD, MAIN_PROCEDURE_MD],
        extra={
            "steps": list(steps),
            "dry_run": dry_run,
            "adapter_checks": adapter_checks,
            "source_markdown_expected_sha256": SOURCE_MARKDOWN_EXPECTED_SHA256,
        },
    )
    return artifact_root


def load_config(path: str | None, *, fast_dev_run: bool, output_root: str | None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        cfg = deep_update(cfg, read_yaml(path))
    if fast_dev_run:
        cfg = deep_update(cfg, FAST_DEV_OVERRIDES)
    if output_root:
        cfg["artifact_root"] = output_root
    return cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run or dry-run dissertation procedure steps.")
    run.add_argument("--config", type=str, default=None)
    run.add_argument("--steps", type=str, default="all")
    run.add_argument("--output-root", type=str, default=None)
    run.add_argument("--fast-dev-run", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = load_config(args.config, fast_dev_run=bool(args.fast_dev_run), output_root=args.output_root)
        root = run_main(
            config,
            steps=parse_steps(args.steps),
            dry_run=bool(args.dry_run),
            force=bool(args.force),
        )
        print(f"[done] main procedure artifacts: {root}", flush=True)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":  # pragma: no cover
    main()
