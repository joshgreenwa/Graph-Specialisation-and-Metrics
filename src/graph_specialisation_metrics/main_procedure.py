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
from graph_specialisation_metrics.grit_intervention_procedure import instantiate_official_models, run_intervention_steps


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
        "baseline_sample_graphs": 200,
        "swap_partners": 8,
        "batched_vjp": True,
        "swap_partner_policy": "different_type",
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
        "grit_1hop_localrrwp": {
            "adapter": "official_grit",
            "variant": "1hop_localrrwp",
            "role": "strict_local_pe_parameter_matched_control",
            "official_repo": "https://github.com/LiamMa/GRIT",
            "repo_path": "external/GRIT",
            "artifact_root": "/rds/user/jgg45/hpc-work/grit_zinc_results/1hop_localrrwp",
            "config_path": None,
            "checkpoint_path": None,
        },
        "gin": {
            "adapter": "pyg_gin",
            "role": "local_validation_reference",
            "artifact_root": "/rds/user/jgg45/hpc-work/grit_zinc_results/gin_reference",
            "dataset_dir": "/rds/user/jgg45/hpc-work/grit_zinc_results/datasets",
            "config_path": None,
            "checkpoint_path": None,
        },
        "gcn": {
            "adapter": "pyg_gcn",
            "role": "local_validation_reference",
            "artifact_root": "/rds/user/jgg45/hpc-work/grit_zinc_results/gcn_reference",
        },
    },
    "steps": {
        "0": {
            "name": "measurement_model_validation",
            "sample_graphs": 200,
            "ig_step_sweep": [16, 32, 64, 128, 256],
            "diagnostic_sample_graphs": 16,
            "baseline_sweep": ["mean_node_embedding", "zero_embedding"],
            "matched_target_ig_steps": 64,
            "matched_target_sources_per_graph": 4,
            "matched_target_partners_per_source": 2,
        },
        "1": {"name": "performance_gap", "reach_sweep": [1, 2, 3, 5, "dense"]},
        "2": {
            "name": "usage_vs_causal_usage",
            "sample_graphs": 200,
            "compare_attention_to_swaps": True,
            "run_layer_channel_split": True,
            "run_attention_erasure": True,
            "erasure_models": ["dense_grit"],
            "erasure_sample_graphs": 64,
            "erasure_fractions": [0.0, 0.05, 0.10, 0.20, 0.35, 0.50],
            "erasure_include_near_control": True,
        },
        "3": {"name": "distance_resolved_overfitting", "sample_graphs": 200},
        "4": {
            "name": "mediator_patching",
            "sample_graphs": 100,
            "max_far_pairs_per_graph": 64,
            "depth_pairs_per_graph": 8,
            "all_distance": True,
            "min_distance": 2,
            "stratify_by_distance": True,
            "onset_fraction_threshold": 0.50,
            "min_effect_abs": 1.0e-6,
            "signal_gate": True,
            "signal_gate_quantile": 0.90,
            "clamp_mode": "detach",
            "run_analytic_patching_check": True,
            "run_clamp_negative_control": True,
            "run_clamp_mode_comparison": True,
            "clamp_mode_comparison_modes": ["detach", "overwrite"],
            "clamp_mode_comparison_max_pairs_per_model": 16,
            "composed_reference_max_direct_fraction": 0.20,
        },
        "5": {
            "name": "non_composable_gap_attribution",
            "sample_graphs": 200,
            "max_far_pairs_per_graph": "all",
            "interaction_pairs": 1000,
            "min_effect_abs": 1.0e-6,
            "signal_gate": True,
            "signal_gate_quantile": 0.90,
            "clamp_mode": "detach",
            "reference_models": ["dense_grit", "grit_1hop", "grit_1hop_localrrwp", "gin"],
        },
    },
    "figures": {"dpi": 180},
}


FAST_DEV_OVERRIDES: dict[str, Any] = {
    "artifact_root": "artifacts/main_procedure_fast_dev",
    "seeds": [41],
    "perturbation": {
        "ig_steps": 4,
        "baseline_sample_graphs": 2,
        "swap_partners": 1,
    },
    "steps": {
        "0": {
            "sample_graphs": 2,
            "ig_step_sweep": [4, 8],
            "diagnostic_sample_graphs": 1,
            "matched_target_ig_steps": 4,
            "matched_target_sources_per_graph": 1,
            "matched_target_partners_per_source": 1,
        },
        "1": {"reach_sweep": [1, "dense"]},
        "2": {"sample_graphs": 2, "compare_attention_to_swaps": False, "run_layer_channel_split": False, "erasure_sample_graphs": 2},
        "3": {"sample_graphs": 2},
        "4": {"sample_graphs": 1, "max_far_pairs_per_graph": 1, "depth_pairs_per_graph": 0},
        "5": {"sample_graphs": 2, "max_far_pairs_per_graph": 2, "interaction_pairs": 16},
    },
}


ANALYSIS_PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    "full": {},
    "smoke": FAST_DEV_OVERRIDES,
    "quick": {
        "seeds": [41],
        "perturbation": {
            "ig_steps": 4,
            "baseline_sample_graphs": 4,
            "swap_partners": 0,
        },
        "steps": {
            "0": {
                "sample_graphs": 4,
                "ig_step_sweep": [4],
                "diagnostic_sample_graphs": 1,
                "baseline_sweep": ["mean_node_embedding"],
                "matched_target_ig_steps": 0,
                "matched_target_sources_per_graph": 0,
                "matched_target_partners_per_source": 0,
            },
            "1": {"reach_sweep": [1, "dense"]},
            "2": {"sample_graphs": 4, "compare_attention_to_swaps": False, "run_layer_channel_split": False, "erasure_sample_graphs": 2},
            "3": {"sample_graphs": 4},
            "4": {
                "sample_graphs": 2,
                "max_far_pairs_per_graph": 1,
                "depth_pairs_per_graph": 0,
                "run_clamp_negative_control": False,
                "run_clamp_mode_comparison": False,
            },
            "5": {"sample_graphs": 4, "max_far_pairs_per_graph": 1, "interaction_pairs": 16},
        },
    },
    "pilot": {
        "seeds": [41],
        "perturbation": {
            "ig_steps": 8,
            "baseline_sample_graphs": 8,
            "swap_partners": 2,
        },
        "steps": {
            "0": {
                "sample_graphs": 8,
                "ig_step_sweep": [8, 16],
                "diagnostic_sample_graphs": 2,
                "baseline_sweep": ["mean_node_embedding"],
                "matched_target_ig_steps": 8,
                "matched_target_sources_per_graph": 2,
                "matched_target_partners_per_source": 1,
            },
            "1": {"reach_sweep": [1, "dense"]},
            "2": {"sample_graphs": 8, "compare_attention_to_swaps": False, "run_layer_channel_split": False, "erasure_sample_graphs": 4},
            "3": {"sample_graphs": 8},
            "4": {
                "sample_graphs": 4,
                "max_far_pairs_per_graph": 2,
                "depth_pairs_per_graph": 0,
                "run_clamp_negative_control": True,
                "run_clamp_mode_comparison": False,
            },
            "5": {"sample_graphs": 8, "max_far_pairs_per_graph": 2, "interaction_pairs": 64},
        },
    },
    "medium": {
        "seeds": [41],
        "perturbation": {
            "ig_steps": 16,
            "baseline_sample_graphs": 16,
            "swap_partners": 2,
        },
        "steps": {
            "0": {
                "sample_graphs": 16,
                "ig_step_sweep": [8, 16, 32],
                "diagnostic_sample_graphs": 4,
                "baseline_sweep": ["mean_node_embedding", "zero_embedding"],
                "matched_target_ig_steps": 16,
                "matched_target_sources_per_graph": 2,
                "matched_target_partners_per_source": 1,
            },
            "1": {"reach_sweep": [1, "dense"]},
            "2": {"sample_graphs": 16, "compare_attention_to_swaps": False, "run_layer_channel_split": False, "erasure_sample_graphs": 8},
            "3": {"sample_graphs": 16},
            "4": {
                "sample_graphs": 8,
                "max_far_pairs_per_graph": 4,
                "depth_pairs_per_graph": 2,
                "run_clamp_negative_control": True,
                "run_clamp_mode_comparison": True,
                "clamp_mode_comparison_max_pairs_per_model": 4,
            },
            "5": {"sample_graphs": 16, "max_far_pairs_per_graph": 4, "interaction_pairs": 128},
        },
    },
    "high": {
        # A step up from "medium" for tighter confidence intervals without going to the
        # full 200-graph run: ~3x the molecules and far-pairs, IG steps 16->32 (past the
        # Step-0 under-convergence point), the depth schedule enabled, and a wider IG-step
        # sweep so the noise-floor-vs-steps question is answerable in one run. Still a
        # single analysis seed (multi-seed needs retrained checkpoints, which we do not have).
        "seeds": [41],
        "perturbation": {
            "ig_steps": 32,
            "baseline_sample_graphs": 32,
            "swap_partners": 4,
        },
        "steps": {
            "0": {
                "sample_graphs": 48,
                "ig_step_sweep": [16, 32, 64],
                "diagnostic_sample_graphs": 8,
                "baseline_sweep": ["mean_node_embedding", "zero_embedding"],
                "matched_target_ig_steps": 32,
                "matched_target_sources_per_graph": 4,
                "matched_target_partners_per_source": 2,
            },
            "1": {"reach_sweep": [1, "dense"]},
            "2": {"sample_graphs": 48, "compare_attention_to_swaps": False, "run_layer_channel_split": False, "erasure_sample_graphs": 24},
            "3": {"sample_graphs": 48},
            "4": {
                "sample_graphs": 24,
                "max_far_pairs_per_graph": 12,
                "depth_pairs_per_graph": 4,
                "run_clamp_negative_control": True,
                "run_clamp_mode_comparison": True,
                "clamp_mode_comparison_max_pairs_per_model": 12,
            },
            "5": {"sample_graphs": 48, "max_far_pairs_per_graph": 12, "interaction_pairs": 256},
        },
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
        [
            "*checkpoint*.pt",
            "*checkpoint*.pth",
            "*checkpoint*.pkl",
            "*state_dict*.pkl",
            "*.ckpt",
            "best*.pt",
            "best*.pth",
            "best*.pkl",
            "model*.pt",
        ],
    )
    stats = find_files(
        root,
        [
            "*stats*.json",
            "*metrics*.json",
            "*history*.json",
            "*summary*.json",
            "training_summary.json",
            "*stats*.csv",
            "*metrics*.csv",
            "*history*.csv",
            "*summary*.csv",
        ],
    )
    return {
        "model": model_name,
        "role": model_cfg.get("role"),
        "adapter": model_cfg.get("adapter"),
        "variant": model_cfg.get("variant"),
        "artifact_root": str(root),
        "exists": root.exists(),
        "config_candidates": [str(path) for path in configs[:10] if path.exists()],
        "checkpoint_candidates": [str(path) for path in checkpoints[:10] if path.exists()],
        "stats_candidates": [str(path) for path in stats[:100] if path.exists()],
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


def metric_source_kind(stats_path: Path) -> str:
    name = stats_path.name.lower()
    if "summary" in name or "best" in name:
        return "summary"
    if stats_path.suffix.lower() == ".csv":
        return "history_csv"
    return "history_json"


def best_validation_row(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any] | None, str]:
    keyed: list[tuple[float, int, Mapping[str, Any]]] = []
    for idx, row in enumerate(rows):
        val = extract_metric_from_payload(row, ["best_val_mae", "val_mae", "valid_mae", "val_loss", "valid_loss", "val"])
        if math.isfinite(val):
            keyed.append((val, idx, row))
    if keyed:
        keyed.sort(key=lambda item: (item[0], item[1]))
        return keyed[0][2], "history_best_val"
    for row in reversed(list(rows)):
        has_metric = any(
            math.isfinite(extract_metric_from_payload(row, keys))
            for keys in (
                ["best_test_mae", "test_mae", "mae_test", "test_loss", "test"],
                ["best_val_mae", "val_mae", "valid_mae", "val_loss", "valid_loss", "val"],
                ["best_train_mae", "train_mae", "train_loss", "train"],
            )
        )
        if has_metric:
            return row, "history_final"
    return None, "history_empty"


def choose_preferred_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int | str], list[Mapping[str, Any]]] = {}
    for row in rows:
        seed = row.get("seed")
        seed_key: int | str = int(seed) if isinstance(seed, int) else "unknown"
        grouped.setdefault((str(row.get("model")), seed_key), []).append(row)
    out: list[dict[str, Any]] = []
    source_rank = {"summary": 0, "history_best_val": 1, "history_final": 2, "history_csv": 3, "history_json": 4}
    for _, items in sorted(grouped.items()):
        def sort_key(item: Mapping[str, Any]) -> tuple[int, float]:
            val = safe_float(item.get("val_metric"))
            return (source_rank.get(str(item.get("source_kind")), 99), val if math.isfinite(val) else float("inf"))

        best = dict(sorted(items, key=sort_key)[0])
        best["selection_rule"] = "prefer_training_summary_then_best_validation_row"
        out.append(best)
    return out


def extract_test_metric(model: str, stats_path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(stats_path) if stats_path.suffix.lower() != ".csv" else read_csv_rows(stats_path)
    except Exception:
        return None
    source_kind = metric_source_kind(stats_path)
    selected_payload: Any = payload
    if isinstance(payload, list):
        selected, row_kind = best_validation_row([row for row in payload if isinstance(row, Mapping)])
        if selected is None:
            return None
        selected_payload = selected
        source_kind = row_kind
    elif isinstance(payload, Mapping) and source_kind != "summary":
        for key in ("history", "epochs", "stats"):
            history = payload.get(key)
            if isinstance(history, list):
                selected, row_kind = best_validation_row([row for row in history if isinstance(row, Mapping)])
                if selected is not None:
                    selected_payload = selected
                    source_kind = row_kind
                break
    test = extract_metric_from_payload(selected_payload, ["best_test_mae", "test_mae", "mae_test", "test_loss", "test"])
    val = extract_metric_from_payload(selected_payload, ["best_val_mae", "val_mae", "valid_mae", "val_loss", "valid_loss", "val"])
    train = extract_metric_from_payload(selected_payload, ["best_train_mae", "train_mae", "train_loss", "train"])
    if not any(math.isfinite(v) for v in [test, val, train]):
        return None
    seed = None
    match = re.search(r"(?:seed|s)(\d+)", str(stats_path), flags=re.IGNORECASE)
    if match:
        seed = int(match.group(1))
    return {
        "model": model,
        "seed": seed,
        "test_metric": test,
        "val_metric": val,
        "train_metric": train,
        "source": str(stats_path),
        "source_kind": source_kind,
    }


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
    metric_rows = choose_preferred_metric_rows(metric_rows)
    write_csv(artifact_root / "metrics" / "step1_test_metrics.csv", metric_rows)
    write_csv(artifact_root / "metrics" / "step1_training_history.csv", history_rows)
    render_step_1_figures(metric_rows, history_rows, artifact_root, config)
    return {
        "status": "complete" if metric_rows else "waiting_for_training_artifacts",
        "test_metric_rows": len(metric_rows),
        "history_rows": len(history_rows),
    }


def canonical_step1_model_name(model: str) -> str:
    """Map common model-label variants onto the Step 1 decomposition keys."""
    raw = str(model)
    normalised = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    compact = normalised.replace("_", "")
    if normalised == "gin" or compact == "gin":
        return "gin"
    if "local" in normalised and "rrwp" in normalised and ("1hop" in compact or "onehop" in compact):
        return "grit_1hop_localrrwp"
    if normalised in {"grit_1hop_localrrwp", "grit_1hop_local_rrwp", "one_hop_local_rrwp"}:
        return "grit_1hop_localrrwp"
    if normalised in {
        "grit_1hop",
        "grit_1_hop",
        "one_hop",
        "onehop",
        "one_hop_global_rrwp",
        "grit_1hop_global_rrwp",
        "grit_1_hop_global_rrwp",
    }:
        return "grit_1hop"
    if ("1hop" in compact or "onehop" in compact) and "rrwp" in compact:
        return "grit_1hop"
    if normalised in {"dense_grit", "grit_dense", "official_grit", "official", "dense"}:
        return "dense_grit"
    return raw


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
                grouped.setdefault(canonical_step1_model_name(str(row["model"])), []).append(value)
        if grouped:
            ladder_order = ["gin", "grit_1hop_localrrwp", "grit_1hop", "dense_grit"]
            ladder_labels = {
                "gin": "GIN",
                "grit_1hop_localrrwp": "1-hop GRIT\nlocal RRWP",
                "grit_1hop": "1-hop GRIT\nglobal RRWP",
                "dense_grit": "dense GRIT",
            }
            mechanism_by_pair = {
                ("gin", "grit_1hop_localrrwp"): "arch",
                ("grit_1hop_localrrwp", "grit_1hop"): "global RRWP",
                ("grit_1hop", "dense_grit"): "global attention",
            }
            configured_models = {
                canonical_step1_model_name(str(model_name))
                for model_name in (config.get("models", {}) or {}).keys()
            }
            if any(model in grouped or model in configured_models for model in ladder_order):
                models = list(ladder_order)
            else:
                models = []
            models.extend(sorted(model for model in grouped if model not in set(models)))

            mean_by_model = {model: float(np.mean(values)) for model, values in grouped.items()}
            std_by_model = {
                model: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                for model, values in grouped.items()
            }
            finite_values = [
                mean_by_model[model] + std_by_model.get(model, 0.0)
                for model in models
                if model in mean_by_model and math.isfinite(mean_by_model[model])
            ]
            xmax = max(finite_values + [0.1])
            pad = 0.025 * xmax
            y = np.arange(len(models), dtype=float)
            fig_height = max(4.8, 0.72 * len(models) + 2.2)
            fig, ax = plt.subplots(figsize=(9.4, fig_height))
            palette = {
                "gin": "#4c78a8",
                "grit_1hop_localrrwp": "#72b7b2",
                "grit_1hop": "#f58518",
                "dense_grit": "#54a24b",
            }
            missing_models: list[str] = []
            for idx, model in enumerate(models):
                label = ladder_labels.get(model, model)
                value = mean_by_model.get(model, float("nan"))
                err = std_by_model.get(model, 0.0)
                if math.isfinite(value):
                    ax.barh(
                        y[idx],
                        value,
                        xerr=err,
                        color=palette.get(model, "#b279a2"),
                        alpha=0.92,
                        capsize=4,
                    )
                    ax.text(value + pad, y[idx], f"{value:.3f}", va="center", ha="left", fontsize=9)
                else:
                    missing_models.append(label.replace("\n", " "))
                    ax.barh(
                        y[idx],
                        max(0.015 * xmax, 0.002),
                        color="#eeeeee",
                        edgecolor="#888888",
                        hatch="//",
                    )
                    ax.text(max(0.03 * xmax, 0.003), y[idx], "metric missing", va="center", ha="left", fontsize=9, color="#555555")
            ax.set_title("Step 1: ZINC test MAE by model", pad=14)
            ax.set_xlabel("Test MAE (lower is better)")
            ax.set_yticks(y)
            ax.set_yticklabels([ladder_labels.get(model, model) for model in models])
            ax.invert_yaxis()
            ax.set_xlim(0.0, xmax * 1.28)
            ax.grid(axis="x", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            gap_lines: list[str] = []
            for left, right in mechanism_by_pair:
                if left in mean_by_model and right in mean_by_model:
                    improvement = mean_by_model[left] - mean_by_model[right]
                    left_label = ladder_labels.get(left, left).replace("\n", " ")
                    right_label = ladder_labels.get(right, right).replace("\n", " ")
                    gap_lines.append(f"{mechanism_by_pair[(left, right)]}: {left_label} to {right_label}, dMAE={improvement:+.3f}")
            note_lines = ["Decomposition order: GIN to 1-hop local RRWP to 1-hop global RRWP to dense GRIT."]
            if gap_lines:
                note_lines.extend(gap_lines)
            if missing_models:
                note_lines.append("Missing metric: " + ", ".join(missing_models) + ".")
            bottom = min(0.36, 0.14 + 0.035 * len(note_lines))
            fig.subplots_adjust(left=0.28, right=0.98, top=0.88, bottom=bottom)
            fig.text(0.28, 0.03, "\n".join(note_lines), ha="left", va="bottom", fontsize=8.5)
            fig.savefig(figures / "step1_test_error_dense_vs_1hop.png", dpi=dpi)
            fig.savefig(figures / "step1_test_error_dense_vs_1hop.pdf", bbox_inches="tight")
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
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L1 / MAE loss")
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


def verify_onehop_locality(
    config: Mapping[str, Any],
    *,
    output: Path,
    sample_graphs: int = 4,
    tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Certify that the 1-hop GRIT checkpoint has no direct non-local attention.

    This is intentionally a hard preflight for the 1-hop control: every captured
    attention layer must use only self/neighbor molecular pairs, and its direct
    attention mass on pairs with molecular distance > 1 must be zero up to the
    provided numerical tolerance.
    """

    from graph_specialisation_metrics.grit_intervention_procedure import (
        attention_support_audit_rows,
        attention_support_failures,
        distance_matrix,
        graph_identity,
        instantiate_official_models,
        select_graphs,
    )

    progress("1-hop locality preflight: discovering model artifacts")
    discovery = [discover_model_artifacts(name, cfg) for name, cfg in config["models"].items()]
    models = instantiate_official_models(config, discovery)
    onehop_models = [
        m
        for m in models
        if "1hop" in m.name.lower() or "1hop" in str(m.variant).lower() or "one_hop" in str(m.variant).lower()
    ]
    if not onehop_models:
        raise RuntimeError("1-hop locality preflight failed: no loadable 1-hop official GRIT adapter was found")

    seed = int(config.get("seeds", [0])[0])
    rows: list[dict[str, Any]] = []
    checked_models: list[str] = []
    sample_counts: dict[str, int] = {}
    for onehop in onehop_models:
        graphs = select_graphs(onehop.adapter, "test", int(sample_graphs), seed=seed)
        if not graphs:
            raise RuntimeError(f"1-hop locality preflight failed: no test graphs were available for {onehop.name}")
        checked_models.append(onehop.name)
        sample_counts[onehop.name] = len(graphs)
        progress(f"1-hop locality preflight: checking {onehop.name} on {len(graphs)} graph(s)")
        for graph_idx, graph in enumerate(graphs):
            gid = graph_identity("test", graph_idx, graph)
            dist = distance_matrix(graph)
            cache = onehop.adapter.forward(graph)
            rows.extend(
                attention_support_audit_rows(
                    cache,
                    dist,
                    onehop.name,
                    gid,
                    expected_max_direct_distance=1,
                )
            )

    csv_path = output.with_suffix(".csv")
    write_csv(csv_path, rows)
    if not rows:
        raise RuntimeError("1-hop locality preflight failed: no attention layers were captured")

    support_failures = list(attention_support_failures(rows))
    mass_failures = [
        row
        for row in rows
        if math.isfinite(safe_float(row.get("direct_attention_mass_distance_gt1")))
        and safe_float(row.get("direct_attention_mass_distance_gt1")) > float(tolerance)
    ]
    max_distance = max(
        [safe_float(row.get("max_direct_attention_distance")) for row in rows if math.isfinite(safe_float(row.get("max_direct_attention_distance")))]
        or [float("nan")]
    )
    max_nonlocal_mass = max(
        [safe_float(row.get("direct_attention_mass_distance_gt1")) for row in rows if math.isfinite(safe_float(row.get("direct_attention_mass_distance_gt1")))]
        or [float("nan")]
    )
    total_nonlocal_edges = int(sum(max(0.0, safe_float(row.get("direct_edges_distance_gt1", 0))) for row in rows))
    total_expected_violations = int(sum(max(0.0, safe_float(row.get("expected_distance_violating_edges", 0))) for row in rows))
    summary = {
        "status": "pass" if not support_failures and not mass_failures else "failed",
        "models": checked_models,
        "sample_graphs_by_model": sample_counts,
        "attention_layers_checked": len(rows),
        "expected_max_direct_distance": 1,
        "tolerance": float(tolerance),
        "max_direct_attention_distance": max_distance,
        "max_direct_attention_mass_distance_gt1": max_nonlocal_mass,
        "total_direct_edges_distance_gt1": total_nonlocal_edges,
        "total_expected_distance_violating_edges": total_expected_violations,
        "support_failures": len(support_failures),
        "mass_failures": len(mass_failures),
        "csv_path": str(csv_path),
    }
    write_json(output, summary)
    progress(
        "1-hop locality preflight: "
        f"status={summary['status']} max_distance={summary['max_direct_attention_distance']} "
        f"max_nonlocal_mass={summary['max_direct_attention_mass_distance_gt1']}"
    )
    if support_failures or mass_failures:
        first = (support_failures or mass_failures)[0]
        raise RuntimeError(
            "1-hop locality preflight failed: a 1-hop GRIT control has direct non-local attention/routing. "
            f"First failure model={first.get('model')} graph={first.get('graph_id')} layer={first.get('layer')} "
            f"max_distance={first.get('max_direct_attention_distance')} "
            f"nonlocal_mass={first.get('direct_attention_mass_distance_gt1')}. "
            f"Audit written to {output} and {csv_path}."
        )
    return summary


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
    run_config = copy.deepcopy(dict(config))
    run_config.setdefault("runtime", {})
    run_config["runtime"]["force_rerun_steps"] = bool(force)
    # --force also busts the per-artifact carriage cache so intervention steps recompute
    # IG carriage from scratch (not just re-render figures from cached tensors). This flag
    # lives under runtime, which is not part of config_hash, so artifact_root is unchanged.
    run_config["runtime"]["force_recompute_carriage"] = bool(force)
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
    discovery = [discover_model_artifacts(name, cfg) for name, cfg in run_config["models"].items()]
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
            progress(f"dry-run status for Step {step}: {run_config['steps'][step]['name']}")
            status = {
                "status": "dry_run_only",
                "step": step,
                "name": run_config["steps"][step]["name"],
                "required_artifacts_discovered": discovery,
            }
            write_step_status(step, artifact_root, status)
            status_by_step[step] = status
            write_json(artifact_root / "metrics" / "main_status.json", status_by_step)
    else:
        intervention_steps = [step for step in steps if step in {"0", "2", "3", "4", "5"}]
        pending_intervention_steps = [
            step for step in intervention_steps if force or not step_is_complete(status_by_step.get(step))
        ]
        intervention_models = None
        intervention_model_error: Exception | None = None
        if pending_intervention_steps:
            try:
                progress(
                    "instantiating intervention adapters once for steps: "
                    f"{','.join(pending_intervention_steps)}"
                )
                intervention_models = instantiate_official_models(run_config, discovery)
                model_names = ", ".join(model.name for model in intervention_models) if intervention_models else "none found"
                progress(f"intervention adapters ready: {model_names}")
            except Exception as exc:
                intervention_model_error = exc
        for step in steps:
            if not force and step_is_complete(status_by_step.get(step)):
                progress(f"Step {step} already complete; reusing cached outputs")
                continue
            if step == "1":
                progress("starting Step 1: performance_gap")
                status = run_step_1(discovery, artifact_root, run_config)
            elif step in intervention_steps:
                try:
                    if intervention_model_error is not None:
                        raise intervention_model_error
                    progress(f"starting intervention Step {step}: {run_config['steps'][step]['name']}")
                    step_status = run_intervention_steps(
                        run_config,
                        discovery,
                        artifact_root,
                        [step],
                        models=intervention_models,
                    )
                    status = step_status.get(step, {"status": "unknown_step", "step": step})
                    progress(f"intervention Step {step} finished")
                except Exception as exc:
                    progress(f"intervention Step {step} failed: {exc}")
                    status = {
                        "status": "failed",
                        "step": step,
                        "name": run_config["steps"][step]["name"],
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


def load_config(
    path: str | None,
    *,
    fast_dev_run: bool,
    output_root: str | None,
    analysis_preset: str = "full",
) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        cfg = deep_update(cfg, read_yaml(path))
    configured_artifact_root = cfg.get("artifact_root") if path else None
    preset = "smoke" if fast_dev_run and analysis_preset == "full" else str(analysis_preset)
    if preset not in ANALYSIS_PRESET_OVERRIDES:
        raise ValueError(f"unknown analysis preset {preset!r}; expected one of {sorted(ANALYSIS_PRESET_OVERRIDES)}")
    if preset != "full":
        cfg = deep_update(cfg, ANALYSIS_PRESET_OVERRIDES[preset])
        if configured_artifact_root is not None:
            cfg["artifact_root"] = configured_artifact_root
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
    run.add_argument("--analysis-preset", choices=sorted(ANALYSIS_PRESET_OVERRIDES), default="full")
    run.add_argument("--fast-dev-run", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--force", action="store_true")
    verify = sub.add_parser("verify-onehop", help="Certify that grit_1hop has no direct non-local attention/routing.")
    verify.add_argument("--config", type=str, required=True)
    verify.add_argument("--output", type=str, required=True)
    verify.add_argument("--sample-graphs", type=int, default=4)
    verify.add_argument("--tolerance", type=float, default=1.0e-12)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = load_config(
            args.config,
            fast_dev_run=bool(args.fast_dev_run),
            output_root=args.output_root,
            analysis_preset=str(args.analysis_preset),
        )
        root = run_main(
            config,
            steps=parse_steps(args.steps),
            dry_run=bool(args.dry_run),
            force=bool(args.force),
        )
        print(f"[done] main procedure artifacts: {root}", flush=True)
    elif args.command == "verify-onehop":
        config = load_config(args.config, fast_dev_run=False, output_root=None, analysis_preset="full")
        summary = verify_onehop_locality(
            config,
            output=Path(args.output),
            sample_graphs=int(args.sample_graphs),
            tolerance=float(args.tolerance),
        )
        print(f"[done] 1-hop locality certification: {args.output}", flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":  # pragma: no cover
    main()
