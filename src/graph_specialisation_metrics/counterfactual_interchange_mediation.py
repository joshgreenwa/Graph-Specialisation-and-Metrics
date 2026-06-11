"""Counterfactual interchange-mediation experiments for GRIT teacher-student tasks.

This module implements the first CFIM experiment sequence described in
``counterfactual_interchange_mediation_plan.md``:

* deterministic graph/data caches for ``ppr_diffusion``,
  ``nearest_anchor_voronoi`` and ``local_mean_gcn``;
* frozen teacher operators with stored ``K``, ``M``, ``b`` and ``Y``;
* official-backed GRIT training for continuous node regression;
* effect-aware graph counterfactual intervention caches;
* clean/counterfactual metrics and gates;
* specialisation-score extraction through the existing official-GRIT adapter;
* source-to-base interchange patches for attention, message, pair state, and
  residual head contributions.

The official model path imports Liam Ma's GRIT implementation at runtime.  The
local backend is only a CPU smoke-test fallback and must not be used for paper
claims.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

# Required when strict PyTorch deterministic algorithms are enabled and CUDA
# matmul reaches CuBLAS before the shell has set a workspace policy.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import yaml
except Exception as exc:  # pragma: no cover - exercised only in incomplete envs.
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


TASKS = ("ppr_diffusion", "nearest_anchor_voronoi", "local_mean_gcn")
MODEL_BACKENDS = ("official", "official_gnnplus", "local")
PPR_FAMILIES = ("ppr_payload_swap", "ppr_struct_swap")
VORONOI_FAMILIES = (
    "voronoi_anchor_marker_swap",
    "voronoi_struct_swap",
    "voronoi_payload_swap",
)
LOCAL_MEAN_GCN_FAMILIES = ("local_mean_payload_swap", "local_mean_struct_swap")
ALL_FAMILIES = PPR_FAMILIES + VORONOI_FAMILIES + LOCAL_MEAN_GCN_FAMILIES
FUNCTIONAL_MODELS = ("grit", "gcn_plus")
FUNCTIONAL_STRATA = ("d1", "d2_to_L", "dL1_to_2L", "d_gt_2L")
FUNCTIONAL_STRATUM_LABELS = {
    "d1": "d = 1",
    "d2_to_L": "2 <= d <= L",
    "dL1_to_2L": "L < d <= 2L",
    "d_gt_2L": "d > 2L",
}
FUNCTIONAL_NODE_STRATA = ("self", "d1", "d2_to_L", "d_gt_L")
FUNCTIONAL_NODE_STRATUM_LABELS = {
    "self": "i in {u,v}",
    "d1": "D_i = 1",
    "d2_to_L": "2 <= D_i <= L",
    "d_gt_L": "D_i > L",
}
RHO_FAR_BINS = ("q1", "q2", "q3", "q4")
RHO_FAR_BIN_LABELS = {
    "q1": "Q1 low rho_far",
    "q2": "Q2",
    "q3": "Q3",
    "q4": "Q4 high rho_far",
}
Q1_GATE_BINS = ("q1", "q2", "q3", "q4")
Q1_GATE_BIN_LABELS = {
    "q1": "Q1 low gate",
    "q2": "Q2",
    "q3": "Q3",
    "q4": "Q4 high gate",
}
Q1_PRIMARY_GATES = {
    "routing": "clean_route_swap_far_mass",
    "transport": "clean_transport_swap_far_share",
}
PATHWAY_BY_FAMILY = {
    "ppr_payload_swap": "M",
    "ppr_struct_swap": "K",
    "voronoi_anchor_marker_swap": "K",
    "voronoi_struct_swap": "K",
    "voronoi_payload_swap": "M",
    "local_mean_payload_swap": "M",
    "local_mean_struct_swap": "K",
}
TEACHER_SEEDS = {"ppr_diffusion": 314159, "nearest_anchor_voronoi": 271828, "local_mean_gcn": 161803}
SPLIT_SEEDS = {
    "train": 1729,
    "val": 1730,
    "test_id": 1731,
    "test_ood_64": 1732,
    "patch_eval": 1733,
}
EPS = 1.0e-12


DEFAULT_CONFIG: dict[str, Any] = {
    "artifacts": {"root": "artifacts/cfim"},
    "graph_generator": {
        "family": "sparse_connected_with_chords",
        "n_train_min": 24,
        "n_train_max": 32,
        "n_eval": 32,
        "n_ood": 64,
        "tree_backbone": True,
        "tree_method": "prufer",
        "chord_edges_per_node": 1.0,
        "chord_sampling": "uniform_non_edges",
        "undirected": True,
        "self_loops_for_model": False,
        "edge_types": {"backbone": 0, "chord": 1, "ring": 2},
        "planted_rings": {"enabled": True, "probability": 0.5, "ring_size": 6, "edge_type": 2},
        "unique_hub": {"enabled": True, "probability": 0.25, "extra_degree": 6},
        "reject_if_disconnected": True,
        "max_rejection_attempts": 100,
    },
    "structural_features": {
        "shortest_path_distance": {"enabled": True, "max_bucket": 16, "disconnected_bucket": 17},
        "rrwp": {
            "enabled": True,
            "steps": 16,
            "normalization": "random_walk_row_stochastic",
        },
    },
    "feature_dimensions": {
        "payload_dim": 16,
        "marker_dim": 2,
        "node_input_dim_ppr": 16,
        "node_input_dim_voronoi": 18,
        "node_input_dim_local_mean_gcn": 16,
        "target_dim": 16,
    },
    "teachers": {
        "ppr": {"alpha": 0.15, "truncation": 8, "teacher_seed": 314159},
        "voronoi": {"n_anchors": 4, "min_anchor_distance": 2, "teacher_seed": 271828},
        "local_mean_gcn": {"teacher_seed": 161803, "include_self": True},
    },
    "model": {
        "name": "grit",
        "backend": "official",
        "num_layers": 2,
        "hidden_dim": 128,
        "num_heads": 8,
        "head_dim": 16,
        "ffn_hidden_dim": 256,
        "pair_hidden_dim": 64,
        "input_dropout": 0.0,
        "attention_dropout": 0.0,
        "residual_dropout": 0.1,
        "ffn_dropout": 0.1,
        "layer_norm": False,
        "batch_norm": True,
        "activation": "relu",
        "use_rrwp": True,
        "rrwp_steps": 16,
        "use_shortest_path_bias": False,
        "use_edge_type": True,
        "use_degree_features": True,
        "use_pair_value_transport": True,
        "use_pair_state_evolution": True,
        "output_head": "node_regression_linear",
        "official_source": {
            "repo": "https://github.com/LiamMa/GRIT",
            "checked_commit": "6c988ea600a606fbb49a2246c64a2d37396b3ab5",
            "config_family": "GRIT-RRWP",
        },
    },
    "training": {
        "optimizer": "adamw",
        "learning_rate": 3.0e-4,
        "weight_decay": 1.0e-5,
        "batch_size_graphs": 64,
        "eval_batch_size_graphs": 1024,
        "max_steps": 5000,
        "warmup_steps": 250,
        "lr_schedule": "cosine_decay_to_10_percent",
        "gradient_clip_norm": 1.0,
        "loss": "mse_node_mean",
        "eval_every_steps": 250,
        "checkpoint_every_steps": 0,
        "select_checkpoint": "lowest_val_relmse",
        "early_stop_val_relmse": 0.005,
        "early_stop_min_steps": 1500,
        "early_stop_patience_evals": 4,
        "mixed_precision": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "dataloader_workers": 4,
        "pin_memory": True,
        "allow_tf32": False,
        "use_cached_train_data": False,
        "train_cache_graphs": 0,
        "progress_every_steps": 0,
    },
    "seeds": {
        "graph_train_seed": 1729,
        "graph_val_seed": 1730,
        "graph_test_id_seed": 1731,
        "graph_test_ood_seed": 1732,
        "graph_patch_eval_seed": 1733,
        "teacher_ppr_seed": 314159,
        "teacher_voronoi_seed": 271828,
        "teacher_local_mean_gcn_seed": 161803,
        "model_init_seed": 1001,
        "training_order_seed": 1002,
        "intervention_seed": 2001,
    },
    "splits": {
        "val": {"num_graphs": 4096, "n": 32},
        "test_id": {"num_graphs": 4096, "n": 32},
        "test_ood_64": {"num_graphs": 1024, "n": 64},
        "patch_eval": {"num_graphs": 256, "n": 32},
    },
    "cf_eval_budget": {
        "graphs_per_task": 1024,
        "candidate_sample_limit_per_graph_family": 64,
        "interventions_per_graph_per_family": 16,
        "bin_allocation": {"null": 2, "low": 2, "medium": 4, "high": 8},
        "progress_every_graphs": 25,
    },
    "patch_eval_budget": {
        "graphs_per_task": 256,
        "candidate_sample_limit_per_graph_family": 256,
        "interventions_per_graph_per_family": 16,
        "bin_allocation": {"null": 2, "low": 2, "medium": 4, "high": 8},
        "progress_every_graphs": 25,
    },
    "minimum_clean_performance": {
        "test_id_relmse_max": 0.02,
        "test_ood_64_relmse_max": 0.10,
    },
    "counterfactual_correctness_gate": {
        "median_CEA_min": 0.80,
        "median_CEE_max": 0.25,
        "median_beta_T_min": 0.70,
        "median_beta_T_max": 1.30,
        "cf_relmse_clean_multiplier_max": 2.0,
    },
    "specialisation": {
        "num_permutations": 32,
        "batch_size_graphs": 64,
        "metrics": "routing,transport,output",
        "interventions": "content,structure",
        "blocks": "all,local,global",
        "centered": "false,true",
        "alpha_tau": 0.1,
        "top_k_single_head_display": 8,
        "top_fraction_group": 0.25,
        "minimum_score_for_candidate": 0.25,
    },
    "patching": {
        "max_interventions_per_family": 32,
        "components": ["attn_probs", "message_pre_weight", "resid_contribution"],
        "group_sizes": [1, 2, 4],
        "random_groups_per_size": 10,
        "batch_size_graphs": 1,
        "allow_failed_gate_patching": False,
    },
    "response_predictivity": {
        "intervention_kind": "cf_eval",
        "families": [],
        "max_interventions_per_family": 4096,
        "batch_size_graphs": 64,
        "folds": 5,
        "ridge_alpha": 1.0,
        "centered": False,
        "seed": 7001,
    },
    "bootstrap": {"resamples": 2000, "confidence_interval": 95, "seed": 6060},
}


def deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def require_yaml() -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML is required for CFIM configs") from _YAML_IMPORT_ERROR
    return yaml


def load_config(path: Path | None, *, task: str | None = None, fast_dev_run: bool = False) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        y = require_yaml()
        loaded = y.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = deep_update(cfg, loaded)
    if task is not None:
        cfg["task"] = task
    if "task" not in cfg:
        raise ValueError("config must define task or --task must be passed")
    if cfg["task"] not in TASKS:
        raise ValueError(f"unsupported task {cfg['task']!r}")
    if os.environ.get("CFIM_ARTIFACT_ROOT"):
        cfg["artifacts"]["root"] = os.environ["CFIM_ARTIFACT_ROOT"]
    if fast_dev_run:
        cfg["splits"]["val"]["num_graphs"] = 8
        cfg["splits"]["test_id"]["num_graphs"] = 8
        cfg["splits"]["test_ood_64"]["num_graphs"] = 4
        cfg["splits"]["patch_eval"]["num_graphs"] = 4
        cfg["training"]["batch_size_graphs"] = 4
        cfg["training"]["eval_batch_size_graphs"] = 8
        cfg["training"]["train_cache_graphs"] = min(int(cfg["training"].get("train_cache_graphs", 0)), 16)
        cfg["training"]["max_steps"] = 2
        cfg["training"]["warmup_steps"] = 1
        cfg["training"]["eval_every_steps"] = 1
        cfg["training"]["checkpoint_every_steps"] = 0
        cfg["cf_eval_budget"]["graphs_per_task"] = 4
        cfg["cf_eval_budget"]["interventions_per_graph_per_family"] = 3
        cfg["cf_eval_budget"]["bin_allocation"] = {"null": 0, "low": 1, "medium": 1, "high": 1}
        cfg["patch_eval_budget"]["graphs_per_task"] = 4
        cfg["patch_eval_budget"]["interventions_per_graph_per_family"] = 3
        cfg["patch_eval_budget"]["bin_allocation"] = {"null": 0, "low": 1, "medium": 1, "high": 1}
        cfg["specialisation"]["num_permutations"] = 2
        cfg["patching"]["max_interventions_per_family"] = 2
        cfg["patching"]["random_groups_per_size"] = 2
    return cfg


def write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    y = require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(y.safe_dump(dict(data), sort_keys=False), encoding="utf-8")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_config(data: Mapping[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def tensor_digest(digest: "hashlib._Hash", value: torch.Tensor) -> None:
    arr = value.detach().cpu().contiguous().numpy()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())


def graph_sha256(record: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("n", "graph_id"):
        digest.update(str(record[key]).encode("utf-8"))
    for key in ("edge_index", "edge_attr", "x", "payload"):
        tensor_digest(digest, record[key])
    for key in ("K", "M", "b", "Y"):
        tensor_digest(digest, record["teacher"][key])
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def dirty_git_state() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return bool(out.strip())
    except Exception:
        return True


def artifact_root(cfg: Mapping[str, Any]) -> Path:
    return Path(str(cfg["artifacts"]["root"])).expanduser()


def data_dir(cfg: Mapping[str, Any], task: str) -> Path:
    return artifact_root(cfg) / "data" / task


def checkpoint_dir(cfg: Mapping[str, Any], task: str, seed: int | None = None) -> Path:
    seed = int(seed if seed is not None else cfg["seeds"]["model_init_seed"])
    return artifact_root(cfg) / "checkpoints" / model_name_for_checkpoint(cfg) / task / f"seed_{seed}"


def model_name_for_checkpoint(cfg: Mapping[str, Any]) -> str:
    return str(cfg.get("model", {}).get("name", "grit"))


def metric_model_name(row: Mapping[str, Any]) -> str:
    return str(row.get("model") or "grit")


def intervention_dir(cfg: Mapping[str, Any], task: str) -> Path:
    return artifact_root(cfg) / "interventions" / task


def metrics_dir(cfg: Mapping[str, Any]) -> Path:
    return artifact_root(cfg) / "metrics"


def figures_main_dir(cfg: Mapping[str, Any]) -> Path:
    return artifact_root(cfg) / "figures" / "main"


def figures_appendix_dir(cfg: Mapping[str, Any]) -> Path:
    return artifact_root(cfg) / "figures" / "appendix"


def functional_root_dir(cfg: Mapping[str, Any]) -> Path:
    return artifact_root(cfg) / "function"


def functional_swaps_dir(cfg: Mapping[str, Any], task: str) -> Path:
    return functional_root_dir(cfg) / "swaps" / task


def functional_responses_dir(cfg: Mapping[str, Any], task: str) -> Path:
    return functional_root_dir(cfg) / "responses" / task


def functional_metrics_dir(cfg: Mapping[str, Any]) -> Path:
    return functional_root_dir(cfg) / "metrics"


def functional_figures_dir(cfg: Mapping[str, Any]) -> Path:
    return functional_root_dir(cfg) / "figures"


def functional_node_swaps_dir(cfg: Mapping[str, Any], task: str) -> Path:
    return functional_root_dir(cfg) / "node_swaps" / task


def functional_node_responses_dir(cfg: Mapping[str, Any], task: str) -> Path:
    return functional_root_dir(cfg) / "node_responses" / task


def functional_q1_dir(cfg: Mapping[str, Any], task: str) -> Path:
    return functional_root_dir(cfg) / "q1" / task


def functional_q1_gate_features_path(cfg: Mapping[str, Any], task: str, seed: int) -> Path:
    return functional_q1_dir(cfg, task) / f"functional_q1_gate_features_seed{int(seed)}.csv"


def functional_validity_gate_path(cfg: Mapping[str, Any]) -> Path:
    return functional_metrics_dir(cfg) / "functional_validity_gate.csv"


def functional_q1_gate_quartile_stats_path(cfg: Mapping[str, Any]) -> Path:
    return functional_metrics_dir(cfg) / "functional_q1_gate_quartile_stats.csv"


def functional_q1_graph_coupling_points_path(cfg: Mapping[str, Any]) -> Path:
    return functional_metrics_dir(cfg) / "functional_q1_graph_coupling_points.csv"


def functional_q1_graph_coupling_stats_path(cfg: Mapping[str, Any]) -> Path:
    return functional_metrics_dir(cfg) / "functional_q1_graph_coupling_stats.csv"


def functional_q1_gate_contrast_stats_path(cfg: Mapping[str, Any]) -> Path:
    return functional_metrics_dir(cfg) / "functional_q1_gate_contrast_stats.csv"


def set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_runtime(cfg: Mapping[str, Any], device: torch.device) -> None:
    training = cfg["training"]
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(training.get("allow_tf32", False))
        torch.backends.cudnn.allow_tf32 = bool(training.get("allow_tf32", False))
    if bool(training.get("deterministic_algorithms", True)):
        torch.use_deterministic_algorithms(
            True,
            warn_only=bool(training.get("deterministic_warn_only", False)),
        )


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def prufer_tree_edges(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if n <= 1:
        return []
    prufer = rng.integers(0, n, size=n - 2).tolist()
    degree = np.ones(n, dtype=np.int64)
    for node in prufer:
        degree[int(node)] += 1
    leaves = sorted(int(i) for i in np.flatnonzero(degree == 1))
    edges: list[tuple[int, int]] = []
    for node in prufer:
        leaf = leaves.pop(0)
        edges.append(tuple(sorted((leaf, int(node)))))
        degree[leaf] -= 1
        degree[int(node)] -= 1
        if degree[int(node)] == 1:
            insert = int(np.searchsorted(leaves, int(node)))
            leaves.insert(insert, int(node))
    edges.append(tuple(sorted((leaves[0], leaves[1]))))
    return edges


def add_edge(edge_types: dict[tuple[int, int], int], a: int, b: int, edge_type: int) -> None:
    if a == b:
        return
    key = tuple(sorted((int(a), int(b))))
    edge_types.setdefault(key, int(edge_type))


def generate_edge_types(n: int, cfg: Mapping[str, Any], rng: np.random.Generator) -> dict[tuple[int, int], int]:
    gcfg = cfg["graph_generator"]
    edge_code = gcfg["edge_types"]
    edge_types: dict[tuple[int, int], int] = {}
    for a, b in prufer_tree_edges(n, rng):
        add_edge(edge_types, a, b, int(edge_code["backbone"]))

    target_chords = int(round(float(gcfg["chord_edges_per_node"]) * n))
    attempts = 0
    while len(edge_types) < (n - 1) + target_chords and attempts < target_chords * 100 + 100:
        a, b = rng.choice(n, size=2, replace=False)
        key = tuple(sorted((int(a), int(b))))
        attempts += 1
        if key not in edge_types:
            edge_types[key] = int(edge_code["chord"])

    ring_cfg = gcfg["planted_rings"]
    if ring_cfg.get("enabled", True) and rng.random() < float(ring_cfg["probability"]):
        size = min(int(ring_cfg["ring_size"]), n)
        ring = rng.choice(n, size=size, replace=False).tolist()
        for idx, node in enumerate(ring):
            add_edge(edge_types, int(node), int(ring[(idx + 1) % size]), int(ring_cfg["edge_type"]))

    hub_cfg = gcfg["unique_hub"]
    if hub_cfg.get("enabled", True) and rng.random() < float(hub_cfg["probability"]):
        hub = int(rng.integers(0, n))
        candidates = [node for node in range(n) if node != hub]
        rng.shuffle(candidates)
        for node in candidates[: int(hub_cfg["extra_degree"])]:
            add_edge(edge_types, hub, int(node), int(edge_code["chord"]))
    return edge_types


def edge_type_dense_from_edges(n: int, edge_types: Mapping[tuple[int, int], int]) -> torch.Tensor:
    dense = torch.full((n, n), -1, dtype=torch.long)
    for (a, b), edge_type in edge_types.items():
        dense[a, b] = int(edge_type)
        dense[b, a] = int(edge_type)
    return dense


def dense_to_edge_index_attr(edge_type_dense: torch.Tensor, num_edge_types: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst = torch.nonzero(edge_type_dense >= 0, as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0).long()
    edge_type = edge_type_dense[src, dst].long().clamp_min(0)
    edge_attr = F.one_hot(edge_type, num_classes=num_edge_types).float()
    return edge_index, edge_attr


def adjacency_from_edge_type(edge_type_dense: torch.Tensor) -> torch.Tensor:
    return (edge_type_dense >= 0).float()


def shortest_path_distances(adj: torch.Tensor) -> torch.Tensor:
    n = int(adj.size(0))
    neighbors = [torch.nonzero(adj[i] > 0, as_tuple=False).reshape(-1).tolist() for i in range(n)]
    dist = torch.full((n, n), -1, dtype=torch.long)
    for source in range(n):
        dist[source, source] = 0
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            for nxt in neighbors[node]:
                if int(dist[source, nxt]) < 0:
                    dist[source, nxt] = int(dist[source, node]) + 1
                    queue.append(int(nxt))
    return dist


def rrwp_tensor(adj: torch.Tensor, steps: int) -> torch.Tensor:
    n = int(adj.size(0))
    deg = adj.sum(dim=1).clamp_min(1.0)
    transition = adj.float() / deg[:, None]
    powers = []
    current = transition.clone()
    for _ in range(int(steps)):
        powers.append(current)
        current = current @ transition
    if not powers:
        return torch.zeros(n, n, 0)
    return torch.stack(powers, dim=-1).float()


def build_struct(edge_type_dense: torch.Tensor, cfg: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    adj = adjacency_from_edge_type(edge_type_dense)
    degree = adj.sum(dim=1).float()
    dist = shortest_path_distances(adj)
    spd_cfg = cfg["structural_features"]["shortest_path_distance"]
    max_bucket = int(spd_cfg["max_bucket"])
    disconnected = int(spd_cfg["disconnected_bucket"])
    spd_bucket = torch.where(dist < 0, torch.full_like(dist, disconnected), dist.clamp(max=max_bucket))
    rrwp_steps = int(cfg["structural_features"]["rrwp"]["steps"])
    return {
        "adjacency": adj,
        "edge_type_dense": edge_type_dense.long(),
        "degree": degree,
        "shortest_path_distance": dist,
        "spd_bucket": spd_bucket.long(),
        "rrwp": rrwp_tensor(adj, rrwp_steps),
        "ring_pair": (edge_type_dense == int(cfg["graph_generator"]["edge_types"]["ring"])).float(),
    }


def orthogonal_teacher_matrix(dim: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    matrix = rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q * signs
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return torch.from_numpy(q.astype(np.float32))


def ppr_kernel(adj: torch.Tensor, alpha: float, truncation: int) -> torch.Tensor:
    n = int(adj.size(0))
    a_tilde = adj.float() + torch.eye(n, dtype=torch.float32)
    transition = a_tilde / a_tilde.sum(dim=1, keepdim=True).clamp_min(1.0)
    current = torch.eye(n, dtype=torch.float32)
    kernel = torch.zeros(n, n, dtype=torch.float32)
    for r in range(int(truncation) + 1):
        kernel = kernel + float(alpha) * ((1.0 - float(alpha)) ** r) * current
        current = current @ transition
    normalizer = 1.0 - (1.0 - float(alpha)) ** (int(truncation) + 1)
    return kernel / normalizer


def local_mean_kernel(adj: torch.Tensor, *, include_self: bool = True) -> torch.Tensor:
    n = int(adj.size(0))
    kernel = adj.float()
    if bool(include_self):
        kernel = kernel + torch.eye(n, dtype=torch.float32)
    return kernel / kernel.sum(dim=1, keepdim=True).clamp_min(1.0)


def voronoi_kernel(
    dist: torch.Tensor,
    anchor_indicator: torch.Tensor,
    anchor_priority: torch.Tensor,
) -> torch.Tensor:
    n = int(dist.size(0))
    anchors = torch.nonzero(anchor_indicator > 0.5, as_tuple=False).reshape(-1).tolist()
    if not anchors:
        raise ValueError("nearest_anchor_voronoi graph has no anchors")
    kernel = torch.zeros(n, n, dtype=torch.float32)
    for query in range(n):
        best_anchor = min(
            anchors,
            key=lambda anchor: (
                int(dist[query, anchor]) if int(dist[query, anchor]) >= 0 else 10**9,
                float(anchor_priority[anchor]),
            ),
        )
        kernel[query, int(best_anchor)] = 1.0
    return kernel


def sample_anchors(
    dist: torch.Tensor,
    n_anchors: int,
    min_distance: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    n = int(dist.size(0))
    last = None
    for _ in range(50):
        anchors = rng.choice(n, size=int(n_anchors), replace=False)
        last = anchors
        ok = True
        for i, a in enumerate(anchors):
            for b in anchors[i + 1 :]:
                if int(dist[int(a), int(b)]) < int(min_distance):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            break
    assert last is not None
    indicator = torch.zeros(n, dtype=torch.float32)
    indicator[torch.as_tensor(last, dtype=torch.long)] = 1.0
    return indicator


def make_teacher(
    task: str,
    payload: torch.Tensor,
    struct: Mapping[str, torch.Tensor],
    cfg: Mapping[str, Any],
    *,
    anchor_indicator: Optional[torch.Tensor] = None,
    anchor_priority: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    dims = cfg["feature_dimensions"]
    payload_dim = int(dims["payload_dim"])
    target_dim = int(dims["target_dim"])
    if task == "ppr_diffusion":
        tc = cfg["teachers"]["ppr"]
        w = orthogonal_teacher_matrix(payload_dim, int(tc["teacher_seed"]))[:, :target_dim]
        k = ppr_kernel(struct["adjacency"], float(tc["alpha"]), int(tc["truncation"]))
        m = payload @ w
    elif task == "local_mean_gcn":
        tc = cfg["teachers"]["local_mean_gcn"]
        w = orthogonal_teacher_matrix(payload_dim, int(tc["teacher_seed"]))[:, :target_dim]
        k = local_mean_kernel(struct["adjacency"], include_self=bool(tc.get("include_self", True)))
        m = payload @ w
    elif task == "nearest_anchor_voronoi":
        if anchor_indicator is None or anchor_priority is None:
            raise ValueError("voronoi teacher requires anchors")
        tc = cfg["teachers"]["voronoi"]
        w = orthogonal_teacher_matrix(payload_dim, int(tc["teacher_seed"]))[:, :target_dim]
        k = voronoi_kernel(struct["shortest_path_distance"], anchor_indicator, anchor_priority)
        m = payload @ w
    else:
        raise ValueError(task)
    b = torch.zeros_like(m)
    y = b + k @ m
    return {"K": k.float(), "M": m.float(), "b": b.float(), "Y": y.float()}


def make_graph_record(task: str, n: int, seed: int, cfg: Mapping[str, Any], graph_id: str | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    edge_types = generate_edge_types(int(n), cfg, rng)
    edge_type_dense = edge_type_dense_from_edges(int(n), edge_types)
    struct = build_struct(edge_type_dense, cfg)
    payload_dim = int(cfg["feature_dimensions"]["payload_dim"])
    payload = torch.from_numpy(rng.normal(size=(int(n), payload_dim)).astype(np.float32))
    anchor_indicator = None
    anchor_priority = None
    if task in {"ppr_diffusion", "local_mean_gcn"}:
        x = payload.clone()
    elif task == "nearest_anchor_voronoi":
        tc = cfg["teachers"]["voronoi"]
        anchor_indicator = sample_anchors(
            struct["shortest_path_distance"],
            int(tc["n_anchors"]),
            int(tc["min_anchor_distance"]),
            rng,
        )
        anchor_priority = torch.ones(int(n), dtype=torch.float32)
        anchors = torch.nonzero(anchor_indicator > 0.5, as_tuple=False).reshape(-1)
        raw = torch.from_numpy(rng.uniform(0.0, 1.0, size=int(anchors.numel())).astype(np.float32))
        order = torch.argsort(raw)
        raw[order] = raw[order] + 1.0e-6 * torch.arange(raw.numel(), dtype=torch.float32)
        anchor_priority[anchors] = raw
        x = torch.cat([payload, anchor_indicator[:, None], anchor_priority[:, None]], dim=1)
    else:
        raise ValueError(task)
    teacher = make_teacher(
        task,
        payload,
        struct,
        cfg,
        anchor_indicator=anchor_indicator,
        anchor_priority=anchor_priority,
    )
    edge_index, edge_attr = dense_to_edge_index_attr(struct["edge_type_dense"])
    gid = graph_id or f"{task}_seed{int(seed)}_n{int(n)}"
    return {
        "graph_id": gid,
        "task": task,
        "n": int(n),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "x": x.float(),
        "payload": payload.float(),
        "anchor_indicator": anchor_indicator,
        "anchor_priority": anchor_priority,
        "struct": struct,
        "teacher": teacher,
        "rng_state_or_seed": int(seed),
    }


def clone_record_with(
    base: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    graph_id: str,
    payload: Optional[torch.Tensor] = None,
    edge_type_dense: Optional[torch.Tensor] = None,
    anchor_indicator: Optional[torch.Tensor] = None,
    anchor_priority: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    task = str(base["task"])
    out = copy.deepcopy(dict(base))
    out["graph_id"] = graph_id
    if edge_type_dense is not None:
        struct = build_struct(edge_type_dense.long(), cfg)
        edge_index, edge_attr = dense_to_edge_index_attr(struct["edge_type_dense"])
        out["struct"] = struct
        out["edge_index"] = edge_index
        out["edge_attr"] = edge_attr
    if payload is not None:
        out["payload"] = payload.float()
    if anchor_indicator is not None:
        out["anchor_indicator"] = anchor_indicator.float()
    if anchor_priority is not None:
        out["anchor_priority"] = anchor_priority.float()
    if task in {"ppr_diffusion", "local_mean_gcn"}:
        out["x"] = out["payload"].float()
    elif task == "nearest_anchor_voronoi":
        out["x"] = torch.cat(
            [
                out["payload"].float(),
                out["anchor_indicator"].float()[:, None],
                out["anchor_priority"].float()[:, None],
            ],
            dim=1,
        )
    out["teacher"] = make_teacher(
        task,
        out["payload"],
        out["struct"],
        cfg,
        anchor_indicator=out.get("anchor_indicator"),
        anchor_priority=out.get("anchor_priority"),
    )
    return out


def transposition_perm(n: int, u: int, v: int) -> torch.Tensor:
    perm = torch.arange(int(n), dtype=torch.long)
    perm[int(u)] = int(v)
    perm[int(v)] = int(u)
    return perm


def apply_pair_permutation(tensor: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return tensor[perm][:, perm]


def source_for_intervention(base: Mapping[str, Any], family: str, u: int, v: int, cfg: Mapping[str, Any]) -> dict[str, Any]:
    n = int(base["n"])
    perm = transposition_perm(n, int(u), int(v))
    graph_id = f"{base['graph_id']}__{family}_{int(u)}_{int(v)}"
    if family in {"ppr_payload_swap", "voronoi_payload_swap", "local_mean_payload_swap"}:
        payload = base["payload"].clone()
        payload[[int(u), int(v)]] = payload[[int(v), int(u)]]
        return clone_record_with(base, cfg, graph_id=graph_id, payload=payload)
    if family in {"ppr_struct_swap", "voronoi_struct_swap", "local_mean_struct_swap"}:
        edge_type_dense = apply_pair_permutation(base["struct"]["edge_type_dense"], perm)
        return clone_record_with(base, cfg, graph_id=graph_id, edge_type_dense=edge_type_dense)
    if family == "voronoi_anchor_marker_swap":
        indicator = base["anchor_indicator"].clone()
        priority = base["anchor_priority"].clone()
        indicator[[int(u), int(v)]] = indicator[[int(v), int(u)]]
        priority[[int(u), int(v)]] = priority[[int(v), int(u)]]
        return clone_record_with(
            base,
            cfg,
            graph_id=graph_id,
            anchor_indicator=indicator,
            anchor_priority=priority,
        )
    raise ValueError(family)


def pathway_deltas(base: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    k_b = base["teacher"]["K"]
    m_b = base["teacher"]["M"]
    y_b = base["teacher"]["Y"]
    k_s = source["teacher"]["K"]
    m_s = source["teacher"]["M"]
    y_s = source["teacher"]["Y"]
    return {
        "K": (k_s - k_b) @ m_b,
        "M": k_b @ (m_s - m_b),
        "KM": y_s - y_b,
        "total": y_s - y_b,
    }


def candidate_pairs_for_family(base: Mapping[str, Any], family: str) -> list[tuple[int, int]]:
    n = int(base["n"])
    if family == "voronoi_anchor_marker_swap":
        anchors = set(torch.nonzero(base["anchor_indicator"] > 0.5, as_tuple=False).reshape(-1).tolist())
        return [(u, v) for u in sorted(anchors) for v in range(n) if v not in anchors]
    return [(u, v) for u in range(n) for v in range(u + 1, n)]


def task_families(task: str) -> tuple[str, ...]:
    if task == "ppr_diffusion":
        return PPR_FAMILIES
    if task == "nearest_anchor_voronoi":
        return VORONOI_FAMILIES
    if task == "local_mean_gcn":
        return LOCAL_MEAN_GCN_FAMILIES
    raise ValueError(task)


def frob_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()).detach().cpu())


def effect_bin_rows(candidates: list[dict[str, Any]]) -> None:
    non_null = [row["delta_T_norm"] for row in candidates if row["delta_T_norm"] > 1.0e-8]
    if not non_null:
        for row in candidates:
            row["effect_bin"] = "null"
        return
    q33, q66 = np.percentile(np.asarray(non_null, dtype=np.float64), [33.0, 66.0])
    for row in candidates:
        norm = float(row["delta_T_norm"])
        if norm <= 1.0e-8:
            row["effect_bin"] = "null"
        elif norm <= q33:
            row["effect_bin"] = "low"
        elif norm <= q66:
            row["effect_bin"] = "medium"
        else:
            row["effect_bin"] = "high"


def sample_by_bins(
    candidates: Sequence[dict[str, Any]],
    allocation: Mapping[str, int],
    total: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_bin[str(row["effect_bin"])].append(row)
    selected: list[dict[str, Any]] = []
    used: set[tuple[int, int, str]] = set()
    for bin_name, count in allocation.items():
        pool = list(by_bin.get(bin_name, []))
        rng.shuffle(pool)
        for row in pool[: int(count)]:
            selected.append(row)
            used.add((int(row["u"]), int(row["v"]), str(row["family"])))
    if len(selected) < int(total):
        remainder = [row for row in candidates if (int(row["u"]), int(row["v"]), str(row["family"])) not in used]
        remainder.sort(key=lambda row: float(row["delta_T_norm"]), reverse=True)
        for row in remainder[: int(total) - len(selected)]:
            selected.append(row)
    return selected[: int(total)]


@dataclass
class CFIMBatch:
    x: torch.Tensor
    payload: torch.Tensor
    anchor_indicator: torch.Tensor
    anchor_priority: torch.Tensor
    node_type: torch.Tensor
    node_mask: torch.Tensor
    adj: torch.Tensor
    edge_value_mat: torch.Tensor
    edge_type_dense: torch.Tensor
    degree: torch.Tensor
    spd: torch.Tensor
    rwse: torch.Tensor
    rrwp: torch.Tensor
    pair_xi: torch.Tensor
    edge_batch: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_attr: torch.Tensor
    node_target: torch.Tensor
    graph_num_nodes: torch.Tensor
    graph_ids: list[str]
    task_type: str = "node_regression"
    num_graphs: int = 0
    max_nodes: int = 0

    @property
    def pair_mask(self) -> torch.Tensor:
        return self.node_mask[:, :, None] & self.node_mask[:, None, :]

    def to(self, device: torch.device) -> "CFIMBatch":
        fields: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            fields[field.name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return CFIMBatch(**fields)


def pad_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype)


def collate_records(records: Sequence[Mapping[str, Any]]) -> CFIMBatch:
    if not records:
        raise ValueError("cannot collate empty records")
    bsz = len(records)
    max_n = max(int(row["n"]) for row in records)
    max_x = max(int(row["x"].size(1)) for row in records)
    payload_dim = int(records[0]["payload"].size(1))
    target_dim = int(records[0]["teacher"]["Y"].size(1))
    rrwp_steps = int(records[0]["struct"]["rrwp"].size(-1))
    x = pad_tensor((bsz, max_n, max_x), torch.float32)
    payload = pad_tensor((bsz, max_n, payload_dim), torch.float32)
    anchor_indicator = pad_tensor((bsz, max_n), torch.float32)
    anchor_priority = torch.ones(bsz, max_n, dtype=torch.float32)
    node_type = torch.zeros(bsz, max_n, dtype=torch.long)
    node_mask = torch.zeros(bsz, max_n, dtype=torch.bool)
    adj = pad_tensor((bsz, max_n, max_n), torch.float32)
    edge_value_mat = pad_tensor((bsz, max_n, max_n), torch.float32)
    edge_type_dense = torch.full((bsz, max_n, max_n), -1, dtype=torch.long)
    degree = pad_tensor((bsz, max_n), torch.float32)
    spd = torch.full((bsz, max_n, max_n), 17, dtype=torch.long)
    rwse = pad_tensor((bsz, max_n, rrwp_steps), torch.float32)
    rrwp = pad_tensor((bsz, max_n, max_n, rrwp_steps), torch.float32)
    pair_xi = pad_tensor((bsz, max_n, max_n, 4), torch.float32)
    node_target = pad_tensor((bsz, max_n, target_dim), torch.float32)
    edge_batches = []
    edge_srcs = []
    edge_dsts = []
    edge_attrs = []
    graph_ids = []
    counts = []
    for graph_idx, row in enumerate(records):
        n = int(row["n"])
        graph_ids.append(str(row["graph_id"]))
        counts.append(n)
        node_mask[graph_idx, :n] = True
        x[graph_idx, :n, : row["x"].size(1)] = row["x"].float()
        payload[graph_idx, :n] = row["payload"].float()
        if row.get("anchor_indicator") is not None:
            anchor_indicator[graph_idx, :n] = row["anchor_indicator"].float()
            anchor_priority[graph_idx, :n] = row["anchor_priority"].float()
            node_type[graph_idx, :n] = row["anchor_indicator"].long()
        adj[graph_idx, :n, :n] = row["struct"]["adjacency"].float()
        edge_value_mat[graph_idx, :n, :n] = row["struct"]["adjacency"].float()
        edge_type_dense[graph_idx, :n, :n] = row["struct"]["edge_type_dense"].long()
        degree[graph_idx, :n] = row["struct"]["degree"].float()
        spd[graph_idx, :n, :n] = row["struct"]["spd_bucket"].long()
        rrwp[graph_idx, :n, :n] = row["struct"]["rrwp"].float()
        diag = row["struct"]["rrwp"][torch.arange(n), torch.arange(n)]
        rwse[graph_idx, :n] = diag.float()
        et = row["struct"]["edge_type_dense"].float()
        pair_xi[graph_idx, :n, :n, 0] = row["struct"]["adjacency"].float()
        pair_xi[graph_idx, :n, :n, 1] = torch.where(et >= 0, (et + 1.0) / 3.0, torch.zeros_like(et))
        pair_xi[graph_idx, :n, :n, 2] = row["struct"]["spd_bucket"].float() / 17.0
        pair_xi[graph_idx, :n, :n, 3] = row["struct"]["ring_pair"].float()
        node_target[graph_idx, :n] = row["teacher"]["Y"].float()
        edge_index = row["edge_index"].long()
        edge_batches.append(torch.full((edge_index.size(1),), graph_idx, dtype=torch.long))
        edge_srcs.append(edge_index[0].long())
        edge_dsts.append(edge_index[1].long())
        edge_attrs.append(row["edge_attr"].float())
    return CFIMBatch(
        x=x,
        payload=payload,
        anchor_indicator=anchor_indicator,
        anchor_priority=anchor_priority,
        node_type=node_type,
        node_mask=node_mask,
        adj=adj,
        edge_value_mat=edge_value_mat,
        edge_type_dense=edge_type_dense,
        degree=degree,
        spd=spd,
        rwse=rwse,
        rrwp=rrwp,
        pair_xi=pair_xi,
        edge_batch=torch.cat(edge_batches) if edge_batches else torch.empty(0, dtype=torch.long),
        edge_src=torch.cat(edge_srcs) if edge_srcs else torch.empty(0, dtype=torch.long),
        edge_dst=torch.cat(edge_dsts) if edge_dsts else torch.empty(0, dtype=torch.long),
        edge_attr=torch.cat(edge_attrs) if edge_attrs else torch.empty(0, 3),
        node_target=node_target,
        graph_num_nodes=torch.tensor(counts, dtype=torch.long),
        graph_ids=graph_ids,
        num_graphs=bsz,
        max_nodes=max_n,
    )


def node_counts_and_offsets(batch: CFIMBatch) -> tuple[list[int], torch.Tensor]:
    counts = [int(value) for value in batch.graph_num_nodes.detach().cpu().tolist()]
    offsets = batch.graph_num_nodes.new_zeros(len(counts))
    if len(counts) > 1:
        offsets[1:] = batch.graph_num_nodes.cumsum(0)[:-1]
    return counts, offsets


def flatten_nodes(batch: CFIMBatch, tensor: torch.Tensor) -> torch.Tensor:
    counts, _ = node_counts_and_offsets(batch)
    return torch.cat([tensor[i, :n] for i, n in enumerate(counts)], dim=0)


def pack_flat_nodes(batch: CFIMBatch, tensor: torch.Tensor) -> torch.Tensor:
    counts, _ = node_counts_and_offsets(batch)
    out = tensor.new_zeros((len(counts), batch.max_nodes, tensor.size(-1)))
    offset = 0
    for graph_idx, n in enumerate(counts):
        out[graph_idx, :n] = tensor[offset : offset + n]
        offset += n
    return out


def external_repo_paths(env_name: str, candidates: Sequence[str]) -> list[Path]:
    paths = []
    if os.environ.get(env_name):
        paths.append(Path(os.environ[env_name]))
    root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).resolve()
    for name in candidates:
        paths.append(root / "external" / name)
        paths.append(root / "graphbench-algoreas-hpc" / "external" / name)
    return paths


def resolve_external_repo_path(env_name: str, candidates: Sequence[str]) -> Path:
    for path in external_repo_paths(env_name, candidates):
        if path.exists():
            return path.resolve()
    raise RuntimeError(
        f"Official repository for {env_name} was not found. Checked: "
        + ", ".join(str(path) for path in external_repo_paths(env_name, candidates))
    )


def add_external_repo_path(env_name: str, candidates: Sequence[str]) -> None:
    for path in external_repo_paths(env_name, candidates):
        if path.exists():
            text = str(path.resolve())
            if text not in sys.path:
                sys.path.insert(0, text)
            return


def require_import(module_name: str, package_hint: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Official backend import failed for {module_name!r}. Install or expose "
            f"{package_hint}. For GRIT set GRIT_ROOT; for GNNPlus set GNNPLUS_ROOT."
        ) from exc


def require_official_file_module(module_name: str, path: Path):
    path = path.resolve()
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not path.exists():
        raise RuntimeError(f"Official backend file is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official backend file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def grit_layer_cfg(update_e: bool):
    yacs_config = require_import("yacs.config", "yacs")
    cn = yacs_config.CfgNode
    cfg = cn()
    cfg.update_e = bool(update_e)
    cfg.bn_momentum = 0.1
    cfg.bn_no_runner = False
    cfg.rezero = False
    cfg.attn = cn()
    cfg.attn.use = True
    cfg.attn.deg_scaler = True
    cfg.attn.use_bias = False
    cfg.attn.clamp = 5.0
    cfg.attn.act = "relu"
    cfg.attn.edge_enhance = True
    cfg.attn.sqrt_relu = False
    cfg.attn.signed_sqrt = True
    cfg.attn.scaled_attn = False
    cfg.attn.no_qk = False
    cfg.attn.graphormer_attn = False
    cfg.attn.norm_e = True
    cfg.attn.O_e = True
    return cfg


def build_pyg_adapter_batch(batch: CFIMBatch, rrwp_steps: int):
    data_mod = require_import("torch_geometric.data", "torch_geometric")
    data = data_mod.Data(num_nodes=int(batch.graph_num_nodes.sum().item()))
    device = batch.x.device
    counts, offsets = node_counts_and_offsets(batch)
    data.batch = torch.cat(
        [torch.full((n,), idx, dtype=torch.long, device=device) for idx, n in enumerate(counts)],
        dim=0,
    )
    if batch.edge_batch.numel():
        edge_offsets = offsets[batch.edge_batch]
        src_global = edge_offsets + batch.edge_src
        dst_global = edge_offsets + batch.edge_dst
        data.edge_index = torch.stack([src_global, dst_global], dim=0)
        data.orig_edge_attr = batch.edge_attr.float()
        data.orig_edge_src = src_global
        data.orig_edge_dst = dst_global
    else:
        data.edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        data.orig_edge_attr = torch.empty(0, batch.edge_attr.size(-1), dtype=torch.float32, device=device)
        data.orig_edge_src = torch.empty(0, dtype=torch.long, device=device)
        data.orig_edge_dst = torch.empty(0, dtype=torch.long, device=device)
    rrwp_node = []
    rrwp_indices = []
    rrwp_values = []
    deg = []
    for graph_idx, n in enumerate(counts):
        local_rrwp = batch.rrwp[graph_idx, :n, :n, :rrwp_steps].float()
        arange = torch.arange(n, dtype=torch.long, device=device)
        rrwp_node.append(local_rrwp[arange, arange])
        src = arange.repeat_interleave(n) + offsets[graph_idx]
        dst = arange.repeat(n) + offsets[graph_idx]
        rrwp_indices.append(torch.stack([src, dst], dim=0))
        rrwp_values.append(local_rrwp.reshape(n * n, rrwp_steps))
        deg.append(batch.degree[graph_idx, :n].float())
    data.rrwp = torch.cat(rrwp_node, dim=0)
    data.rrwp_index = torch.cat(rrwp_indices, dim=1)
    data.rrwp_val = torch.cat(rrwp_values, dim=0)
    data.deg = torch.cat(deg, dim=0)
    data.log_deg = torch.log(data.deg + 1.0)
    data.graph_num_nodes = batch.graph_num_nodes
    return data


class CFIMOfficialGRITModel(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], input_dim: int, edge_attr_dim: int = 3) -> None:
        super().__init__()
        add_external_repo_path("GRIT_ROOT", ("GRIT",))
        grit_layer_mod = require_import("grit.layer.grit_layer", "official GRIT repository")
        rrwp_mod = require_import("grit.encoder.rrwp_encoder", "official GRIT repository")
        model_cfg = cfg["model"]
        dim = int(model_cfg["hidden_dim"])
        heads = int(model_cfg["num_heads"])
        self.rrwp_steps = int(model_cfg["rrwp_steps"])
        self.input_encoder = nn.Linear(int(input_dim), dim)
        self.edge_encoder = nn.Linear(int(edge_attr_dim), dim)
        self.rrwp_node_encoder = rrwp_mod.RRWPLinearNodeEncoder(
            self.rrwp_steps,
            dim,
            batchnorm=False,
            layernorm=False,
        )
        self.rrwp_edge_encoder = rrwp_mod.RRWPLinearEdgeEncoder(
            self.rrwp_steps,
            dim,
            batchnorm=False,
            layernorm=False,
            pad_to_full_graph=True,
            add_node_attr_as_self_loop=False,
            overwrite_old_attr=False,
        )
        layer_cfg = grit_layer_cfg(bool(model_cfg.get("use_pair_state_evolution", True)))
        self.layers = nn.ModuleList(
            grit_layer_mod.GritTransformerLayer(
                dim,
                dim,
                heads,
                dropout=float(model_cfg["residual_dropout"]),
                attn_dropout=float(model_cfg["attention_dropout"]),
                layer_norm=bool(model_cfg.get("layer_norm", False)),
                batch_norm=bool(model_cfg.get("batch_norm", True)),
                residual=True,
                act=str(model_cfg.get("activation", "relu")),
                norm_e=True,
                O_e=True,
                cfg=layer_cfg,
            )
            for _ in range(int(model_cfg["num_layers"]))
        )
        self.output_head = nn.Linear(dim, int(cfg["feature_dimensions"]["target_dim"]))

    def forward(self, batch: CFIMBatch) -> torch.Tensor:
        pyg_batch = build_pyg_adapter_batch(batch, self.rrwp_steps)
        flat_x = flatten_nodes(batch, batch.x.float())
        pyg_batch.x = self.input_encoder(flat_x)
        pyg_batch.edge_attr = self.edge_encoder(pyg_batch.orig_edge_attr.to(pyg_batch.x.dtype))
        pyg_batch = self.rrwp_node_encoder(pyg_batch)
        pyg_batch = self.rrwp_edge_encoder(pyg_batch)
        for layer in self.layers:
            pyg_batch = layer(pyg_batch)
        return pack_flat_nodes(batch, self.output_head(pyg_batch.x))


def build_sparse_pyg_batch(batch: CFIMBatch):
    data_mod = require_import("torch_geometric.data", "torch_geometric")
    data = data_mod.Data(num_nodes=int(batch.graph_num_nodes.sum().item()))
    device = batch.x.device
    counts, offsets = node_counts_and_offsets(batch)
    data.batch = torch.cat(
        [torch.full((n,), idx, dtype=torch.long, device=device) for idx, n in enumerate(counts)],
        dim=0,
    )
    if batch.edge_batch.numel():
        edge_offsets = offsets[batch.edge_batch]
        data.edge_index = torch.stack([edge_offsets + batch.edge_src, edge_offsets + batch.edge_dst], dim=0)
        data.orig_edge_attr = batch.edge_attr.float()
    else:
        data.edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
        data.orig_edge_attr = torch.empty(0, batch.edge_attr.size(-1), dtype=torch.float32, device=device)
    data.graph_num_nodes = batch.graph_num_nodes
    return data


def setup_gnnplus_graphgym_cfg(model_cfg: Mapping[str, Any]) -> None:
    graphgym_config = require_import("torch_geometric.graphgym.config", "torch_geometric GraphGym")
    graphgym_register = require_import("torch_geometric.graphgym.register", "torch_geometric GraphGym")
    yacs_config = require_import("yacs.config", "yacs")
    allow_graphgym_duplicate_registration(graphgym_register)
    cfg = graphgym_config.cfg
    if hasattr(cfg, "defrost"):
        cfg.defrost()
    if hasattr(cfg, "set_new_allowed"):
        cfg.set_new_allowed(True)
    if not hasattr(cfg, "gnn"):
        cfg.gnn = yacs_config.CfgNode(new_allowed=True)
    elif hasattr(cfg.gnn, "set_new_allowed"):
        cfg.gnn.set_new_allowed(True)
    cfg.gnn.act = str(model_cfg.get("activation", "relu"))
    if cfg.gnn.act not in graphgym_register.act_dict:
        try:
            importlib.import_module("GNNPlus.act.example")
        except Exception:
            pass
    if cfg.gnn.act not in graphgym_register.act_dict:
        raise RuntimeError(f"GNNPlus activation {cfg.gnn.act!r} is not registered with GraphGym")


def allow_graphgym_duplicate_registration(graphgym_register: Any) -> None:
    if getattr(graphgym_register, "_cfim_duplicate_registration_ok", False):
        return
    original_register_base = graphgym_register.register_base

    def register_base_idempotent(mapping: dict[str, Any], key: str, module: Any) -> None:
        if key in mapping:
            return
        return original_register_base(mapping, key, module)

    graphgym_register.register_base = register_base_idempotent
    graphgym_register._cfim_duplicate_registration_ok = True


class CFIMOfficialGNNPlusModel(nn.Module):
    """Thin teacher-student adapter around the official GNNPlus layer classes.

    The upstream GNNPlus repository is GraphGym/GPS-oriented and targets graph-level
    heads.  This wrapper keeps its official message-passing layers, but reuses the
    CFIM node-regression data, training loop, and checkpoint format.
    """

    LAYER_MODULES = {
        "gcn": ("gcn_conv_layer.py", "_official_gnnplus_gcn_conv_layer", "GCNConvLayer", False),
        "gcne": ("gcn_conv_layer_e.py", "_official_gnnplus_gcn_conv_layer_e", "GCNConvLayer", True),
        "gine": ("gine_conv_layer.py", "_official_gnnplus_gine_conv_layer", "GINEConvLayer", True),
        "gatedgcn": ("gatedgcn_layer.py", "_official_gnnplus_gatedgcn_layer", "GatedGCNLayer", True),
    }

    def __init__(self, cfg: Mapping[str, Any], input_dim: int, edge_attr_dim: int = 3) -> None:
        super().__init__()
        gnnplus_root = resolve_external_repo_path("GNNPLUS_ROOT", ("GNNPlus",))
        gnnplus_pkg = gnnplus_root / "GNNPlus" if (gnnplus_root / "GNNPlus").exists() else gnnplus_root
        model_cfg = cfg["model"]
        gnnplus_cfg = model_cfg.get("gnnplus", {})
        layer_type = str(gnnplus_cfg.get("layer_type", "gcn")).lower()
        if layer_type not in self.LAYER_MODULES:
            raise ValueError(f"unsupported GNNPlus layer_type {layer_type!r}; expected one of {sorted(self.LAYER_MODULES)}")
        setup_gnnplus_graphgym_cfg(model_cfg)
        file_name, module_name, class_name, edge_aware = self.LAYER_MODULES[layer_type]
        layer_mod = require_official_file_module(module_name, gnnplus_pkg / "layer" / file_name)
        layer_cls = getattr(layer_mod, class_name)

        dim = int(model_cfg["hidden_dim"])
        self.layer_type = layer_type
        self.edge_aware = bool(edge_aware)
        self.rwse_steps = int(model_cfg.get("rrwp_steps", cfg["structural_features"]["rrwp"]["steps"]))
        self.use_rwse = bool(model_cfg.get("use_rwse", model_cfg.get("use_rrwp", True)))
        self.use_degree_features = bool(model_cfg.get("use_degree_features", True))
        node_dim = int(input_dim)
        if self.use_rwse:
            node_dim += self.rwse_steps
        if self.use_degree_features:
            node_dim += 1
        self.input_dropout = nn.Dropout(float(model_cfg.get("input_dropout", 0.0)))
        self.input_encoder = nn.Linear(node_dim, dim)
        self.edge_encoder = nn.Linear(int(edge_attr_dim), dim) if self.edge_aware else None
        dropout = float(model_cfg.get("dropout", model_cfg.get("residual_dropout", 0.0)))
        residual = bool(model_cfg.get("residual", True))
        ffn = bool(model_cfg.get("ffn", True))
        self.layers = nn.ModuleList(
            layer_cls(dim, dim, dropout=dropout, residual=residual, ffn=ffn)
            for _ in range(int(model_cfg["num_layers"]))
        )
        self.output_head = nn.Linear(dim, int(cfg["feature_dimensions"]["target_dim"]))

    def node_features(self, batch: CFIMBatch) -> torch.Tensor:
        pieces = [flatten_nodes(batch, batch.x.float())]
        if self.use_rwse:
            pieces.append(flatten_nodes(batch, batch.rwse[..., : self.rwse_steps].float()))
        if self.use_degree_features:
            pieces.append(torch.log1p(flatten_nodes(batch, batch.degree[..., None].float())))
        return torch.cat(pieces, dim=-1)

    def forward(self, batch: CFIMBatch) -> torch.Tensor:
        pyg_batch = build_sparse_pyg_batch(batch)
        pyg_batch.x = self.input_encoder(self.input_dropout(self.node_features(batch)))
        if self.edge_encoder is not None:
            pyg_batch.edge_attr = self.edge_encoder(pyg_batch.orig_edge_attr.to(pyg_batch.x.dtype))
        else:
            pyg_batch.edge_attr = pyg_batch.orig_edge_attr
        for layer in self.layers:
            pyg_batch = layer(pyg_batch)
        return pack_flat_nodes(batch, self.output_head(pyg_batch.x))


class LocalGRITStyleLayer(nn.Module):
    def __init__(self, dim: int, heads: int, pair_dim: int, dropout: float) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.pair_bias = nn.Linear(pair_dim, heads)
        self.pair_value = nn.Linear(pair_dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, n, dim = h.shape
        hn = self.norm1(h)
        q = self.q(hn).view(bsz, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(hn).view(bsz, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(hn).view(bsz, n, self.heads, self.head_dim).transpose(1, 2)
        pair_bias = self.pair_bias(pair).permute(0, 3, 1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + pair_bias
        scores = scores.masked_fill(~(mask[:, None, None, :] & mask[:, None, :, None]), -1.0e9)
        attn = torch.softmax(scores, dim=-1)
        pair_value = self.pair_value(pair).view(bsz, n, n, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        msg = v[:, :, None, :, :] + pair_value
        delta = (attn.unsqueeze(-1) * msg).sum(dim=3).transpose(1, 2).reshape(bsz, n, dim)
        h = (h + self.dropout(self.out(delta))) * mask.unsqueeze(-1)
        h = (h + self.dropout(self.ffn(self.norm2(h)))) * mask.unsqueeze(-1)
        return h


class CFIMLocalGRITStyleModel(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], input_dim: int) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        dim = int(model_cfg["hidden_dim"])
        heads = int(model_cfg["num_heads"])
        rrwp_steps = int(model_cfg["rrwp_steps"])
        self.input_encoder = nn.Linear(int(input_dim), dim)
        pair_dim = rrwp_steps + 4
        self.layers = nn.ModuleList(
            LocalGRITStyleLayer(dim, heads, pair_dim, float(model_cfg["residual_dropout"]))
            for _ in range(int(model_cfg["num_layers"]))
        )
        self.output_head = nn.Linear(dim, int(cfg["feature_dimensions"]["target_dim"]))

    def forward(self, batch: CFIMBatch) -> torch.Tensor:
        h = self.input_encoder(batch.x.float()) * batch.node_mask.unsqueeze(-1)
        pair = torch.cat([batch.rrwp.float(), batch.pair_xi.float()], dim=-1)
        for layer in self.layers:
            h = layer(h, pair, batch.node_mask)
        return self.output_head(h) * batch.node_mask.unsqueeze(-1)


def input_dim_for_task(cfg: Mapping[str, Any], task: str) -> int:
    dims = cfg["feature_dimensions"]
    if task == "ppr_diffusion":
        return int(dims["node_input_dim_ppr"])
    if task == "nearest_anchor_voronoi":
        return int(dims["node_input_dim_voronoi"])
    if task == "local_mean_gcn":
        return int(dims["node_input_dim_local_mean_gcn"])
    raise ValueError(task)


def build_model(cfg: Mapping[str, Any], task: str, backend: str | None = None) -> nn.Module:
    selected = backend or str(cfg["model"].get("backend", "official"))
    input_dim = input_dim_for_task(cfg, task)
    if selected == "official":
        return CFIMOfficialGRITModel(cfg, input_dim=input_dim)
    if selected == "official_gnnplus":
        return CFIMOfficialGNNPlusModel(cfg, input_dim=input_dim)
    if selected == "local":
        return CFIMLocalGRITStyleModel(cfg, input_dim=input_dim)
    raise ValueError(f"unknown model backend {selected!r}")


def parameter_count(model: nn.Module, *, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return int(sum(param.numel() for param in params if param.requires_grad))
    return int(sum(param.numel() for param in params))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    mask = node_mask.unsqueeze(-1).to(pred.dtype)
    return ((pred - target).square() * mask).sum() / mask.sum().clamp_min(1.0) / pred.size(-1)


def batch_relmse_values(pred: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> list[float]:
    out = []
    for idx in range(pred.size(0)):
        valid = node_mask[idx]
        p = pred[idx, valid].float()
        y = target[idx, valid].float()
        mse = (p - y).square().mean()
        var = y.var(unbiased=False).clamp_min(1.0e-12)
        out.append(float((mse / var).detach().cpu()))
    return out


@torch.no_grad()
def evaluate_records(
    model: nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()
    rels = []
    losses = []
    preds = []
    for start in range(0, len(records), int(batch_size)):
        batch = collate_records(records[start : start + int(batch_size)]).to(device)
        pred = model(batch)
        losses.append(float(masked_mse(pred, batch.node_target, batch.node_mask).detach().cpu()) * batch.num_graphs)
        rels.extend(batch_relmse_values(pred, batch.node_target, batch.node_mask))
        preds.append(pred.detach().cpu())
    pred_tensor = torch.cat(preds, dim=0) if preds else torch.empty(0)
    return {
        "loss": sum(losses) / max(1, len(records)),
        "relmse_mean": float(np.mean(rels)) if rels else float("nan"),
        "relmse_median": float(np.median(rels)) if rels else float("nan"),
        "graphs": len(records),
    }, pred_tensor


def load_records(path: Path) -> list[dict[str, Any]]:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(records), path)


def generate_split_records(task: str, split: str, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = cfg["splits"][split]
    count = int(spec["num_graphs"])
    n = int(spec["n"])
    seed_key = {
        "val": "graph_val_seed",
        "test_id": "graph_test_id_seed",
        "test_ood_64": "graph_test_ood_seed",
        "patch_eval": "graph_patch_eval_seed",
    }[split]
    base_seed = int(cfg["seeds"][seed_key])
    return [
        make_graph_record(task, n, base_seed + idx, cfg, graph_id=f"{task}_{split}_{idx:06d}")
        for idx in range(count)
    ]


def generate_train_pool_records(task: str, cfg: Mapping[str, Any], count: int) -> list[dict[str, Any]]:
    gcfg = cfg["graph_generator"]
    n_min = int(gcfg["n_train_min"])
    n_max = int(gcfg["n_train_max"])
    base_seed = int(cfg["seeds"]["graph_train_seed"])
    rng = np.random.default_rng(base_seed)
    records = []
    for idx in range(int(count)):
        n = int(rng.integers(n_min, n_max + 1))
        seed = base_seed + idx
        records.append(make_graph_record(task, n, seed, cfg, graph_id=f"{task}_train_pool_{idx:06d}"))
    return records


def cache_data(cfg: Mapping[str, Any], task: str, *, force: bool = False) -> None:
    out_dir = data_dir(cfg, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(out_dir / "generator_config.yaml", cfg["graph_generator"])
    write_yaml(out_dir / "teacher_config.yaml", cfg["teachers"])
    manifest_rows = []
    for split in ("val", "test_id", "test_ood_64", "patch_eval"):
        path = out_dir / f"{split}.pt"
        if path.exists() and not force:
            records = load_records(path)
        else:
            print(f"[data] generating {task}/{split}", flush=True)
            records = generate_split_records(task, split, cfg)
            save_records(path, records)
        nodes = [int(row["n"]) for row in records]
        edges = [int(row["edge_index"].size(1) // 2) for row in records]
        manifest_rows.append(
            {
                "split": split,
                "path": str(path),
                "graphs": len(records),
                "nodes_min": min(nodes) if nodes else 0,
                "nodes_max": max(nodes) if nodes else 0,
                "edges_mean": float(np.mean(edges)) if edges else 0.0,
                "sha256": sha256_file(path),
            }
        )
    train_cache_graphs = int(cfg["training"].get("train_cache_graphs", 0))
    if bool(cfg["training"].get("use_cached_train_data", False)) and train_cache_graphs > 0:
        path = out_dir / "train_pool.pt"
        if path.exists() and not force:
            manifest_rows.append(
                {
                    "split": "train_pool",
                    "path": str(path),
                    "graphs": train_cache_graphs,
                    "nodes_min": int(cfg["graph_generator"]["n_train_min"]),
                    "nodes_max": int(cfg["graph_generator"]["n_train_max"]),
                    "edges_mean": "",
                    "sha256": "",
                    "bytes": int(path.stat().st_size),
                }
            )
        else:
            print(f"[data] generating {task}/train_pool graphs={train_cache_graphs}", flush=True)
            records = generate_train_pool_records(task, cfg, train_cache_graphs)
            save_records(path, records)
            nodes = [int(row["n"]) for row in records]
            edges = [int(row["edge_index"].size(1) // 2) for row in records]
            manifest_rows.append(
                {
                    "split": "train_pool",
                    "path": str(path),
                    "graphs": len(records),
                    "nodes_min": min(nodes) if nodes else 0,
                    "nodes_max": max(nodes) if nodes else 0,
                    "edges_mean": float(np.mean(edges)) if edges else 0.0,
                    "sha256": sha256_file(path),
                    "bytes": int(path.stat().st_size),
                }
            )
    write_json(
        out_dir / "data_manifest.json",
        {
            "task": task,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "splits": manifest_rows,
            "teacher_seed": TEACHER_SEEDS[task],
            "config_sha256": sha256_config(cfg),
        },
    )
    write_csv(out_dir / "dataset_summary.csv", manifest_rows)


def online_train_batch(cfg: Mapping[str, Any], task: str, rng: np.random.Generator, batch_size: int) -> list[dict[str, Any]]:
    gcfg = cfg["graph_generator"]
    n_min = int(gcfg["n_train_min"])
    n_max = int(gcfg["n_train_max"])
    seed_base = int(cfg["seeds"]["graph_train_seed"])
    records = []
    for _ in range(int(batch_size)):
        n = int(rng.integers(n_min, n_max + 1))
        seed = seed_base + int(rng.integers(0, 2**31 - 1))
        records.append(make_graph_record(task, n, seed, cfg, graph_id=f"{task}_train_seed{seed}"))
    return records


def cached_train_batch(records: Sequence[Mapping[str, Any]], rng: np.random.Generator, batch_size: int) -> list[Mapping[str, Any]]:
    indices = rng.integers(0, len(records), size=int(batch_size))
    return [records[int(idx)] for idx in indices]


def lr_for_step(base_lr: float, step: int, warmup_steps: int, max_steps: int) -> float:
    if step <= warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base_lr * (0.1 + 0.9 * cosine)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def save_checkpoint(path: Path, model: nn.Module, cfg: Mapping[str, Any], task: str, step: int, val_relmse: float, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "task": task,
        "step": int(step),
        "val_relmse": float(val_relmse),
        "config": dict(cfg),
        "model_backend": cfg["model"].get("backend", "official"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train(cfg: Mapping[str, Any], task: str, *, device_name: str = "auto", backend: Optional[str] = None) -> None:
    cache_data(cfg, task, force=False)
    seed = int(cfg["seeds"]["model_init_seed"])
    set_all_seeds(seed)
    device = choose_device(device_name)
    configure_runtime(cfg, device)
    run_dir = checkpoint_dir(cfg, task, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cfg_path = run_dir / "train_config.yaml"
    resolved_cfg_path = run_dir / "resolved_config.yaml"
    write_yaml(train_cfg_path, cfg)
    write_yaml(resolved_cfg_path, cfg)
    val_records = load_records(data_dir(cfg, task) / "val.pt")
    train_records: Optional[list[dict[str, Any]]] = None
    if bool(cfg["training"].get("use_cached_train_data", False)):
        train_path = data_dir(cfg, task) / "train_pool.pt"
        train_records = load_records(train_path)
        if not train_records:
            raise RuntimeError(f"cached training is enabled but {train_path} is empty")
        print(f"[data] using cached train_pool={train_path} graphs={len(train_records)}", flush=True)
    model = build_model(cfg, task, backend=backend).to(device)
    params_trainable = parameter_count(model, trainable_only=True)
    params_total = parameter_count(model, trainable_only=False)
    print(
        f"[model] name={model_name_for_checkpoint(cfg)} backend={backend or cfg['model'].get('backend', 'official')} "
        f"trainable_params={params_trainable} total_params={params_total}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    rng = np.random.default_rng(int(cfg["seeds"]["training_order_seed"]))
    best_rel = float("inf")
    best_step = 0
    rows = []
    max_steps = int(cfg["training"]["max_steps"])
    batch_size = int(cfg["training"]["batch_size_graphs"])
    eval_batch = int(cfg["training"]["eval_batch_size_graphs"])
    progress_every = int(cfg["training"].get("progress_every_steps", 0))
    start_time = time.time()
    last_step = 0
    early_stop_hits = 0
    for step in range(1, max_steps + 1):
        last_step = step
        model.train()
        lr = lr_for_step(
            float(cfg["training"]["learning_rate"]),
            step,
            int(cfg["training"]["warmup_steps"]),
            max_steps,
        )
        set_optimizer_lr(optimizer, lr)
        if train_records is not None:
            batch_records = cached_train_batch(train_records, rng, batch_size)
        else:
            batch_records = online_train_batch(cfg, task, rng, batch_size)
        batch = collate_records(batch_records).to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = masked_mse(pred, batch.node_target, batch.node_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
        optimizer.step()
        do_eval = step % int(cfg["training"]["eval_every_steps"]) == 0 or step == 1 or step == max_steps
        if progress_every > 0 and step % progress_every == 0 and not do_eval:
            elapsed = time.time() - start_time
            print(
                f"[train-progress] model={model_name_for_checkpoint(cfg)} task={task} step={step}/{max_steps} "
                f"loss={float(loss.detach().cpu()):.6g} lr={lr:.4g} sec_per_step={elapsed / max(step, 1):.3f}",
                flush=True,
            )
        if do_eval:
            metrics, val_pred = evaluate_records(model, val_records, batch_size=eval_batch, device=device)
            improved = metrics["relmse_mean"] < best_rel
            if improved:
                best_rel = float(metrics["relmse_mean"])
                best_step = step
                save_checkpoint(run_dir / "best.pt", model, cfg, task, step, best_rel, optimizer)
                torch.save({"pred": val_pred, "step": step, "val_relmse": best_rel}, run_dir / "val_predictions_best.pt")
            row = {
                "step": step,
                "train_mse": float(loss.detach().cpu()),
                "val_relmse": float(metrics["relmse_mean"]),
                "val_loss": float(metrics["loss"]),
                "learning_rate": lr,
                "seconds": time.time() - start_time,
                "best": improved,
            }
            rows.append(row)
            write_csv(run_dir / "metrics_history.csv", rows)
            print(
                f"[train] task={task} step={step} loss={row['train_mse']:.6g} "
                f"val_relmse={row['val_relmse']:.6g} best={best_rel:.6g}",
                flush=True,
            )
            early_target = float(cfg["training"].get("early_stop_val_relmse", 0.0))
            early_min_steps = int(cfg["training"].get("early_stop_min_steps", max_steps + 1))
            early_patience = int(cfg["training"].get("early_stop_patience_evals", 0))
            if early_target > 0.0 and early_patience > 0 and step >= early_min_steps:
                if float(metrics["relmse_mean"]) <= early_target:
                    early_stop_hits += 1
                else:
                    early_stop_hits = 0
                if early_stop_hits >= early_patience:
                    print(
                        f"[early-stop] task={task} step={step} "
                        f"val_relmse={metrics['relmse_mean']:.6g} target={early_target:.6g}",
                        flush=True,
                    )
                    break
        checkpoint_every = int(cfg["training"].get("checkpoint_every_steps", 0))
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            save_checkpoint(run_dir / f"checkpoint_step{step:06d}.pt", model, cfg, task, step, best_rel, optimizer)
    save_checkpoint(run_dir / "final.pt", model, cfg, task, last_step, best_rel, optimizer)
    best_path = run_dir / "best.pt"
    if not best_path.exists():
        save_checkpoint(best_path, model, cfg, task, last_step, best_rel, optimizer)
    manifest = {
        "task": task,
        "model": model_name_for_checkpoint(cfg),
        "model_seed": seed,
        "teacher_seed": TEACHER_SEEDS[task],
        "git_commit": git_commit(),
        "dirty_git_state": dirty_git_state(),
        "best_step": best_step,
        "best_val_relmse": best_rel,
        "trainable_parameters": params_trainable,
        "total_parameters": params_total,
        "checkpoint_sha256": sha256_file(best_path),
        "config_sha256": sha256_file(resolved_cfg_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_backend": backend or cfg["model"].get("backend", "official"),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    plot_training_curves(run_dir / "metrics_history.csv", figures_appendix_dir(cfg) / f"figA0_training_curves_{task}.pdf")


def load_model_from_checkpoint(cfg: Mapping[str, Any], task: str, checkpoint: Optional[Path], device: torch.device, backend: Optional[str] = None) -> tuple[nn.Module, Mapping[str, Any]]:
    ckpt_path = checkpoint or (checkpoint_dir(cfg, task) / "best.pt")
    checkpoint_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run_cfg = checkpoint_data.get("config", cfg)
    model = build_model(run_cfg, task, backend=backend or checkpoint_data.get("model_backend")).to(device)
    missing, unexpected = model.load_state_dict(checkpoint_data["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint state mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, run_cfg


def build_interventions(cfg: Mapping[str, Any], task: str, kind: str, *, force: bool = False) -> None:
    if kind not in {"cf_eval", "patch_eval"}:
        raise ValueError("--kind must be cf_eval or patch_eval")
    source_split = "test_id" if kind == "cf_eval" else "patch_eval"
    out_path = intervention_dir(cfg, task) / f"{kind}_interventions.pt"
    manifest_path = intervention_dir(cfg, task) / "intervention_manifest.csv"
    if out_path.exists() and not force:
        print(f"[interventions] using existing {out_path}", flush=True)
        return
    records = load_records(data_dir(cfg, task) / f"{source_split}.pt")
    budget = cfg["cf_eval_budget"] if kind == "cf_eval" else cfg["patch_eval_budget"]
    records = records[: int(budget["graphs_per_task"])]
    families = task_families(task)
    rng = random.Random(int(cfg["seeds"]["intervention_seed"]) + (0 if kind == "cf_eval" else 10000))
    selected_records = []
    manifest_rows = []
    candidate_limit = int(budget.get("candidate_sample_limit_per_graph_family", 0))
    progress_every = max(1, int(budget.get("progress_every_graphs", 25)))
    print(
        f"[interventions] start task={task} kind={kind} graphs={len(records)} "
        f"families={','.join(families)} candidate_limit={candidate_limit or 'all'}",
        flush=True,
    )
    for graph_idx, base in enumerate(records, start=1):
        for family in families:
            candidates = []
            pairs = candidate_pairs_for_family(base, family)
            if candidate_limit > 0 and len(pairs) > candidate_limit:
                pairs = rng.sample(pairs, k=candidate_limit)
            for u, v in pairs:
                source = source_for_intervention(base, family, u, v, cfg)
                deltas = pathway_deltas(base, source)
                candidates.append(
                    {
                        "base_graph": base,
                        "source_graph": source,
                        "family": family,
                        "u": int(u),
                        "v": int(v),
                        "delta_T_norm": frob_norm(deltas["total"]),
                        "delta_T_K_norm": frob_norm(deltas["K"]),
                        "delta_T_M_norm": frob_norm(deltas["M"]),
                    }
                )
            effect_bin_rows(candidates)
            chosen = sample_by_bins(
                candidates,
                budget["bin_allocation"],
                int(budget["interventions_per_graph_per_family"]),
                rng,
            )
            for local_idx, row in enumerate(chosen):
                intervention_id = f"{base['graph_id']}__{family}__{local_idx:03d}"
                item = {
                    "intervention_id": intervention_id,
                    "graph_id": base["graph_id"],
                    "task": task,
                    "family": family,
                    "u": row["u"],
                    "v": row["v"],
                    "effect_bin": row["effect_bin"],
                    "pathway_target": PATHWAY_BY_FAMILY[family],
                    "delta_T_norm": row["delta_T_norm"],
                    "delta_T_K_norm": row["delta_T_K_norm"],
                    "delta_T_M_norm": row["delta_T_M_norm"],
                    "base_graph": row["base_graph"],
                    "source_graph": row["source_graph"],
                }
                selected_records.append(item)
                manifest_rows.append(
                    {
                        "intervention_id": intervention_id,
                        "graph_id": base["graph_id"],
                        "task": task,
                        "family": family,
                        "u": row["u"],
                        "v": row["v"],
                        "effect_bin": row["effect_bin"],
                        "pathway_target": PATHWAY_BY_FAMILY[family],
                        "delta_T_norm": row["delta_T_norm"],
                        "delta_T_K_norm": row["delta_T_K_norm"],
                        "delta_T_M_norm": row["delta_T_M_norm"],
                        "source_graph_sha256": graph_sha256(row["source_graph"]),
                        "kind": kind,
                    }
                )
        if graph_idx == 1 or graph_idx % progress_every == 0 or graph_idx == len(records):
            print(
                f"[interventions] task={task} kind={kind} graphs={graph_idx}/{len(records)} "
                f"selected={len(selected_records)}",
                flush=True,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected_records, out_path)
    existing = [row for row in read_csv_dicts(manifest_path) if row.get("kind") != kind]
    write_csv(manifest_path, existing + manifest_rows)
    print(f"[interventions] wrote {len(selected_records)} records to {out_path}", flush=True)


def effect_metrics(delta_f: torch.Tensor, delta_t: torch.Tensor) -> dict[str, float]:
    df = delta_f.float().reshape(-1)
    dt = delta_t.float().reshape(-1)
    denom_t = float((dt * dt).sum().detach().cpu()) + EPS
    denom_f = float((df * df).sum().detach().cpu()) + EPS
    dot = float((df * dt).sum().detach().cpu())
    return {
        "CEE": float(((df - dt).square().sum().detach().cpu()) / denom_t),
        "CEA": dot / math.sqrt(denom_f * denom_t),
        "beta_T": dot / denom_t,
    }


def mediation_metrics(delta_patch: torch.Tensor, target: torch.Tensor, delta_total: torch.Tensor) -> dict[str, float]:
    dp = delta_patch.float().reshape(-1)
    target_flat = target.float().reshape(-1)
    total_flat = delta_total.float().reshape(-1)
    target_norm2 = float((target_flat * target_flat).sum().detach().cpu()) + EPS
    patch_norm2 = float((dp * dp).sum().detach().cpu()) + EPS
    total_norm2 = float((total_flat * total_flat).sum().detach().cpu()) + EPS
    dot = float((dp * target_flat).sum().detach().cpu())
    tcm = dot / target_norm2
    residual = dp - tcm * target_flat
    return {
        "TCM": tcm,
        "TCMA": dot / math.sqrt(patch_norm2 * target_norm2),
        "TPR": float((dp - target_flat).square().sum().detach().cpu()) / target_norm2,
        "MEM": float((dp * total_flat).sum().detach().cpu()) / total_norm2,
        "OC": float(torch.linalg.vector_norm(residual).detach().cpu()) / math.sqrt(target_norm2),
    }


def clean_performance(cfg: Mapping[str, Any], task: str, model: nn.Module, device: torch.device) -> list[dict[str, Any]]:
    rows = []
    model_name = model_name_for_checkpoint(cfg)
    for split in ("test_id", "test_ood_64"):
        split_start = time.time()
        print(f"[clean] start model={model_name} task={task} split={split}", flush=True)
        records = load_records(data_dir(cfg, task) / f"{split}.pt")
        metrics, preds = evaluate_records(
            model,
            records,
            batch_size=int(cfg["training"]["eval_batch_size_graphs"]),
            device=device,
        )
        torch.save({"pred": preds, "metrics": metrics}, checkpoint_dir(cfg, task) / f"{split}_predictions_best.pt")
        rows.append({"model": model_name, "task": task, "split": split, **metrics})
        print(
            f"[clean] done model={model_name} task={task} split={split} relmse_mean={metrics['relmse_mean']:.6g} "
            f"elapsed={time.time() - split_start:.1f}s",
            flush=True,
        )
    return rows


def evaluate_counterfactuals(
    cfg: Mapping[str, Any],
    task: str,
    *,
    device_name: str = "auto",
    checkpoint: Optional[Path] = None,
    backend: Optional[str] = None,
) -> None:
    device = choose_device(device_name)
    configure_runtime(cfg, device)
    model, _run_cfg = load_model_from_checkpoint(cfg, task, checkpoint, device, backend=backend)
    model_name = model_name_for_checkpoint(cfg)
    clean_rows = clean_performance(cfg, task, model, device)
    clean_path = metrics_dir(cfg) / "clean_performance.csv"
    old_clean = [
        row
        for row in read_csv_dicts(clean_path)
        if not (row.get("task") == task and metric_model_name(row) == model_name)
    ]
    write_csv(clean_path, old_clean + clean_rows)
    interventions = torch.load(intervention_dir(cfg, task) / "cf_eval_interventions.pt", map_location="cpu", weights_only=False)
    rows = []
    batch_size = int(cfg["training"]["eval_batch_size_graphs"])
    total_chunks = max(1, math.ceil(len(interventions) / batch_size))
    print(
        f"[counterfactuals] start task={task} interventions={len(interventions)} "
        f"batch_size={batch_size} chunks={total_chunks}",
        flush=True,
    )
    start_time = time.time()
    for start in range(0, len(interventions), batch_size):
        chunk = interventions[start : start + batch_size]
        base_batch = collate_records([row["base_graph"] for row in chunk]).to(device)
        source_batch = collate_records([row["source_graph"] for row in chunk]).to(device)
        with torch.no_grad():
            y_base = model(base_batch)
            y_source = model(source_batch)
        for idx, row in enumerate(chunk):
            n = int(row["base_graph"]["n"])
            pred_b = y_base[idx, :n].detach().cpu()
            pred_s = y_source[idx, :n].detach().cpu()
            target_s = row["source_graph"]["teacher"]["Y"]
            rel_cf = float(((pred_s - target_s).square().mean() / target_s.var(unbiased=False).clamp_min(EPS)).item())
            deltas = pathway_deltas(row["base_graph"], row["source_graph"])
            metrics = effect_metrics(pred_s - pred_b, deltas["total"])
            rows.append(
                {
                    "model": model_name,
                    "task": task,
                    "intervention_id": row["intervention_id"],
                    "graph_id": row["graph_id"],
                    "family": row["family"],
                    "effect_bin": row["effect_bin"],
                    "pathway_target": row["pathway_target"],
                    "delta_T_norm": row["delta_T_norm"],
                    "delta_T_K_norm": row["delta_T_K_norm"],
                    "delta_T_M_norm": row["delta_T_M_norm"],
                    "cf_relmse": rel_cf,
                    **metrics,
                }
            )
        chunk_idx = start // batch_size + 1
        elapsed = time.time() - start_time
        rate = len(rows) / max(elapsed, 1.0)
        print(
            f"[counterfactuals] task={task} chunk={chunk_idx}/{total_chunks} rows={len(rows)} "
            f"rate={rate:.1f}/s elapsed={elapsed:.1f}s",
            flush=True,
        )
    cf_path = metrics_dir(cfg) / "counterfactual_metrics.csv"
    old = [
        row
        for row in read_csv_dicts(cf_path)
        if not (row.get("task") == task and metric_model_name(row) == model_name)
    ]
    write_csv(cf_path, old + rows)
    write_counterfactual_decisions(cfg)
    plot_counterfactual_summary(cfg)


def median(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(clean)) if clean else float("nan")


def task_clean_gate(cfg: Mapping[str, Any], task: str, model_name: Optional[str] = None) -> bool:
    selected_model = model_name or model_name_for_checkpoint(cfg)
    path = metrics_dir(cfg) / "clean_performance.csv"
    rows = [
        row
        for row in read_csv_dicts(path)
        if row.get("task") == task and metric_model_name(row) == selected_model
    ]
    by_split = {row["split"]: float(row["relmse_mean"]) for row in rows}
    gate = cfg["minimum_clean_performance"]
    return (
        by_split.get("test_id", float("inf")) <= float(gate["test_id_relmse_max"])
        and by_split.get("test_ood_64", float("inf")) <= float(gate["test_ood_64_relmse_max"])
    )


def write_counterfactual_decisions(cfg: Mapping[str, Any]) -> None:
    rows = read_csv_dicts(metrics_dir(cfg) / "counterfactual_metrics.csv")
    clean_rows = read_csv_dicts(metrics_dir(cfg) / "clean_performance.csv")
    clean_by_task_model = {
        (row["task"], metric_model_name(row)): float(row["relmse_mean"])
        for row in clean_rows
        if row.get("split") == "test_id" and row.get("relmse_mean")
    }
    model_names = sorted({metric_model_name(row) for row in rows}) or [model_name_for_checkpoint(cfg)]
    decisions = []
    gate = cfg["counterfactual_correctness_gate"]
    for model_name in model_names:
        for task in TASKS:
            families = task_families(task)
            clean_pass = task_clean_gate(cfg, task, model_name=model_name)
            for family in families:
                subset = [
                    row
                    for row in rows
                    if row.get("task") == task
                    and metric_model_name(row) == model_name
                    and row.get("family") == family
                    and row.get("effect_bin") == "high"
                ]
                cea = median([float(row["CEA"]) for row in subset]) if subset else float("nan")
                cee = median([float(row["CEE"]) for row in subset]) if subset else float("nan")
                beta = median([float(row["beta_T"]) for row in subset]) if subset else float("nan")
                cf_rel = median([float(row["cf_relmse"]) for row in subset]) if subset else float("nan")
                clean_rel = clean_by_task_model.get((task, model_name), float("inf"))
                passed = (
                    clean_pass
                    and cea >= float(gate["median_CEA_min"])
                    and cee <= float(gate["median_CEE_max"])
                    and float(gate["median_beta_T_min"]) <= beta <= float(gate["median_beta_T_max"])
                    and cf_rel <= float(gate["cf_relmse_clean_multiplier_max"]) * clean_rel + 1.0e-6
                )
                decisions.append(
                    {
                        "model": model_name,
                        "task": task,
                        "family": family,
                        "hypothesis": "H1_counterfactual_functional_correctness",
                        "decision": "supported" if passed else ("not_supported" if clean_pass else "not_testable_due_to_failed_upstream_gate"),
                        "evidence_metric": "high_bin_median_CEA_CEE_beta",
                        "estimate": cea,
                        "ci_low": "",
                        "ci_high": "",
                        "notes": f"median_CEA={cea:.4g}; median_CEE={cee:.4g}; median_beta_T={beta:.4g}; clean_gate={clean_pass}",
                    }
                )
    path = metrics_dir(cfg) / "hypothesis_decisions.csv"
    old = [row for row in read_csv_dicts(path) if row.get("hypothesis") != "H1_counterfactual_functional_correctness"]
    write_csv(path, old + decisions)


def parse_bool_csv(raw: str) -> tuple[bool, ...]:
    out = []
    for item in raw.split(","):
        value = item.strip().lower()
        if value in {"true", "1", "yes"}:
            out.append(True)
        elif value in {"false", "0", "no"}:
            out.append(False)
        else:
            raise ValueError(f"invalid bool {item!r}")
    return tuple(out)


def parse_csv_tuple(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def run_specialisation(
    cfg: Mapping[str, Any],
    task: str,
    *,
    device_name: str = "auto",
    checkpoint: Optional[Path] = None,
    backend: Optional[str] = None,
) -> None:
    from graph_specialisation_metrics.core_interpretability_specialisation_metrics import (
        MetricOptions,
        OfficialGRITFieldCollector,
        SpecialisationMetricEngine,
    )

    if (backend or cfg["model"].get("backend", "official")) != "official":
        raise RuntimeError("specialisation metrics require the official GRIT backend")
    device = choose_device(device_name)
    configure_runtime(cfg, device)
    model, _run_cfg = load_model_from_checkpoint(cfg, task, checkpoint, device, backend=backend)
    records = load_records(data_dir(cfg, task) / "patch_eval.pt")
    max_nodes = max(int(row["n"]) for row in records)
    collector = OfficialGRITFieldCollector(model, max_nodes=max_nodes)
    scfg = cfg["specialisation"]
    options = MetricOptions(
        metrics=parse_csv_tuple(str(scfg["metrics"])),
        interventions=parse_csv_tuple(str(scfg["interventions"])),
        blocks=parse_csv_tuple(str(scfg["blocks"])),
        centered=parse_bool_csv(str(scfg["centered"])),
        num_permutations=int(scfg["num_permutations"]),
        alpha_tau=float(scfg["alpha_tau"]),
        seed=int(cfg["seeds"]["intervention_seed"]),
    )
    engine = SpecialisationMetricEngine(collector, options)
    batch_size = int(scfg["batch_size_graphs"])
    print(
        f"[specialisation] start task={task} graphs={len(records)} batch_size={batch_size} "
        f"permutations={options.num_permutations}",
        flush=True,
    )
    start_time = time.time()
    for start in range(0, len(records), batch_size):
        batch = collate_records(records[start : start + batch_size]).to(device)
        engine.compute_batch(batch, graph_indices=list(range(start, min(start + batch_size, len(records)))))
        done = min(start + batch_size, len(records))
        elapsed = time.time() - start_time
        print(
            f"[specialisation] task={task} graphs={done}/{len(records)} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
    result = engine.results()
    summary_rows = [dict(row, task=task) for row in result.summary_rows]
    per_graph_rows = [dict(row, task=task) for row in result.per_graph_rows]
    summary_path = metrics_dir(cfg) / "specialisation_scores.csv"
    old = [row for row in read_csv_dicts(summary_path) if row.get("task") != task]
    write_csv(summary_path, old + summary_rows)
    write_csv(metrics_dir(cfg) / f"specialisation_per_graph_{task}.csv", per_graph_rows)
    plot_specialisation_atlas(cfg)


@dataclass
class LayerActivation:
    layer: int
    src: torch.Tensor
    dst: torch.Tensor
    local_src: torch.Tensor
    local_dst: torch.Tensor
    attention: torch.Tensor
    logits: torch.Tensor
    message: torch.Tensor
    pair_message: torch.Tensor
    node_wv: torch.Tensor
    pair_e: Optional[torch.Tensor]
    heads: int
    head_dim: int


class CFIMActivationCapture:
    def __init__(self, model: nn.Module) -> None:
        if not hasattr(model, "layers"):
            raise TypeError("GRIT activation capture expects model.layers")
        self.model = model
        self.records: dict[int, LayerActivation] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> "CFIMActivationCapture":
        from graph_specialisation_metrics import mechanistic_operator_analysis as moa

        self.records = {}
        self.handles = []

        def make_hook(layer_idx: int):
            def hook(module: nn.Module, inputs: tuple[Any, ...], outputs: Any) -> None:
                pyg_batch = inputs[0]
                node_wv, _e_out = outputs
                node_msg, pair_msg, logits, _edge_state = moa.grit_attention_components(module, pyg_batch)
                edge_index = pyg_batch.edge_index.long()
                _graph, local_src, local_dst = moa.edge_local_coordinates(
                    edge_index[0],
                    edge_index[1],
                    [int(v) for v in pyg_batch.graph_num_nodes.detach().cpu().tolist()],
                )
                attention = pyg_batch.attn.squeeze(-1)
                self.records[layer_idx] = LayerActivation(
                    layer=layer_idx,
                    src=edge_index[0].detach(),
                    dst=edge_index[1].detach(),
                    local_src=local_src.detach(),
                    local_dst=local_dst.detach(),
                    attention=attention.detach().float(),
                    logits=logits.detach().float(),
                    message=(node_msg + pair_msg).detach().float(),
                    pair_message=pair_msg.detach().float(),
                    node_wv=node_wv.detach().float(),
                    pair_e=None if getattr(pyg_batch, "E", None) is None else pyg_batch.E.detach().float(),
                    heads=int(attention.size(1)),
                    head_dim=int(node_wv.size(-1)),
                )

            return hook

        for idx, layer in enumerate(self.model.layers):
            self.handles.append(layer.attention.register_forward_hook(make_hook(idx)))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def capture_prediction(model: nn.Module, batch: CFIMBatch) -> tuple[torch.Tensor, dict[int, LayerActivation]]:
    model.eval()
    with CFIMActivationCapture(model) as capture:
        with torch.no_grad():
            pred = model(batch)
    if not capture.records:
        raise RuntimeError("no GRIT activation records captured")
    return pred.detach(), capture.records


def align_source_edges(source: LayerActivation, current_src: torch.Tensor, current_dst: torch.Tensor, n: int) -> torch.Tensor:
    source_keys = source.local_dst.long() * int(n) + source.local_src.long()
    current_keys = current_dst.long() * int(n) + current_src.long()
    order = torch.argsort(source_keys)
    sorted_keys = source_keys[order]
    pos = torch.searchsorted(sorted_keys, current_keys)
    clamped = pos.clamp(max=max(int(sorted_keys.numel()) - 1, 0))
    found = (pos < sorted_keys.numel()) & (sorted_keys[clamped] == current_keys)
    if not bool(found.all()):
        raise RuntimeError(f"could not align {int((~found).sum().item())} source edges")
    return order[clamped]


def get_log_deg(pyg_batch: Any) -> torch.Tensor:
    if hasattr(pyg_batch, "log_deg"):
        return pyg_batch.log_deg.view(-1, 1)
    if hasattr(pyg_batch, "deg"):
        return torch.log(pyg_batch.deg + 1.0).view(-1, 1)
    deg = torch.zeros(pyg_batch.num_nodes, device=pyg_batch.x.device)
    deg.index_add_(0, pyg_batch.edge_index[1], torch.ones_like(pyg_batch.edge_index[1], dtype=torch.float32))
    return torch.log(deg + 1.0).view(-1, 1)


class CFIMPatchContext:
    def __init__(
        self,
        model: nn.Module,
        *,
        batch: CFIMBatch,
        source_cache: Mapping[int, LayerActivation],
        component: str,
        head_group: Sequence[tuple[int, int]],
    ) -> None:
        self.model = model
        self.batch = batch
        self.source_cache = source_cache
        self.component = component
        self.by_layer: dict[int, list[int]] = defaultdict(list)
        for layer, head in head_group:
            self.by_layer[int(layer)].append(int(head))
        self.originals: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> "CFIMPatchContext":
        for layer_idx, heads in self.by_layer.items():
            layer = self.model.layers[layer_idx]
            if self.component == "resid_contribution":
                self.originals.append((layer, "forward", layer.forward))
                layer.forward = self._make_layer_forward(layer_idx, layer, heads)  # type: ignore[method-assign]
            else:
                attention = layer.attention
                self.originals.append((attention, "propagate_attention", attention.propagate_attention))
                attention.propagate_attention = self._make_propagate(layer_idx, attention, heads)
        return self

    def __exit__(self, *_exc: object) -> None:
        for obj, name, original in self.originals:
            setattr(obj, name, original)
        self.originals = []

    def _source_edge_values(self, layer_idx: int, pyg_batch: Any, n: int) -> dict[str, torch.Tensor]:
        source = self.source_cache[layer_idx]
        edge_index = pyg_batch.edge_index.long()
        from graph_specialisation_metrics import mechanistic_operator_analysis as moa

        _graph, local_src, local_dst = moa.edge_local_coordinates(
            edge_index[0],
            edge_index[1],
            [int(v) for v in pyg_batch.graph_num_nodes.detach().cpu().tolist()],
        )
        pos = align_source_edges(source, local_src, local_dst, n).to(edge_index.device)
        out = {
            "attention": source.attention.to(edge_index.device)[pos],
            "message": source.message.to(edge_index.device)[pos],
            "pair_message": source.pair_message.to(edge_index.device)[pos],
        }
        if source.pair_e is not None:
            out["pair_e"] = source.pair_e.to(edge_index.device)[pos]
        return out

    def _make_propagate(self, layer_idx: int, attention_module: nn.Module, heads: Sequence[int]):
        def propagate(pyg_batch: Any) -> None:
            from torch_scatter import scatter
            from graph_specialisation_metrics import mechanistic_operator_analysis as moa

            n = int(self.batch.graph_num_nodes[0].detach().cpu())
            source = self._source_edge_values(layer_idx, pyg_batch, n)
            head_idx = torch.tensor(list(heads), dtype=torch.long, device=pyg_batch.edge_index.device)

            if self.component == "pair_state":
                if getattr(pyg_batch, "E", None) is None or "pair_e" not in source:
                    raise RuntimeError("pair_state patch requested but pair state is unavailable")
                heads_total = int(pyg_batch.V_h.size(1))
                cur = pyg_batch.E.view(pyg_batch.E.size(0), heads_total, -1)
                src = source["pair_e"].view(pyg_batch.E.size(0), heads_total, -1).to(cur.device, cur.dtype)
                cur[:, head_idx] = src[:, head_idx]
                pyg_batch.E = cur.reshape_as(pyg_batch.E)

            node_msg, pair_msg, logits, edge_state = moa.grit_attention_components(attention_module, pyg_batch)
            score = moa.pyg_sparse_softmax(logits.unsqueeze(-1), pyg_batch.edge_index[1], pyg_batch.num_nodes).squeeze(-1)
            if self.component == "attn_probs":
                score[:, head_idx] = source["attention"].to(score.device, score.dtype)[:, head_idx]
            score = attention_module.dropout(score.unsqueeze(-1))
            pyg_batch.attn = score
            if getattr(pyg_batch, "E", None) is not None:
                pyg_batch.wE = edge_state.flatten(1)
            message = node_msg + pair_msg
            if self.component == "message_pre_weight":
                message[:, head_idx] = source["message"].to(message.device, message.dtype)[:, head_idx]
            weighted = message * score
            pyg_batch.wV = torch.zeros_like(pyg_batch.V_h)
            scatter(weighted, pyg_batch.edge_index[1], dim=0, out=pyg_batch.wV, reduce="add")

        return propagate

    def _make_layer_forward(self, layer_idx: int, layer: nn.Module, heads: Sequence[int]):
        def forward(pyg_batch: Any) -> Any:
            h_in1 = pyg_batch.x
            e_in1 = pyg_batch.get("edge_attr", None)
            h_attn_out, e_attn_out = layer.attention(pyg_batch)
            head_idx = torch.tensor(list(heads), dtype=torch.long, device=h_attn_out.device)
            source_wv = self.source_cache[layer_idx].node_wv.to(h_attn_out.device, h_attn_out.dtype)
            h_attn_out = h_attn_out.clone()
            h_attn_out[:, head_idx] = source_wv[:, head_idx]
            h = h_attn_out.view(pyg_batch.num_nodes, -1)
            h = F.dropout(h, layer.dropout, training=layer.training)
            if getattr(layer, "deg_scaler", False):
                log_deg = get_log_deg(pyg_batch)
                h = torch.stack([h, h * log_deg], dim=-1)
                h = (h * layer.deg_coef).sum(dim=-1)
            h = layer.O_h(h)
            e = None
            if e_attn_out is not None:
                e = e_attn_out.flatten(1)
                e = F.dropout(e, layer.dropout, training=layer.training)
                e = layer.O_e(e)
            if layer.residual:
                if getattr(layer, "rezero", False):
                    h = h * layer.alpha1_h
                h = h_in1 + h
                if e is not None:
                    if getattr(layer, "rezero", False):
                        e = e * layer.alpha1_e
                    e = e + e_in1
            if layer.layer_norm:
                h = layer.layer_norm1_h(h)
                if e is not None:
                    e = layer.layer_norm1_e(e)
            if layer.batch_norm:
                h = layer.batch_norm1_h(h)
                if e is not None:
                    e = layer.batch_norm1_e(e)
            h_in2 = h
            h = layer.FFN_h_layer1(h)
            h = layer.act(h)
            h = F.dropout(h, layer.dropout, training=layer.training)
            h = layer.FFN_h_layer2(h)
            if layer.residual:
                if getattr(layer, "rezero", False):
                    h = h * layer.alpha2_h
                h = h_in2 + h
            if layer.layer_norm:
                h = layer.layer_norm2_h(h)
            if layer.batch_norm:
                h = layer.batch_norm2_h(h)
            pyg_batch.x = h
            pyg_batch.edge_attr = e if getattr(layer, "update_e", True) else e_in1
            return pyg_batch

        return forward


def metric_value(row: Mapping[str, Any]) -> float:
    for key in ("mean", "value", "score"):
        if key in row and row[key] not in {"", None}:
            try:
                return float(row[key])
            except Exception:
                pass
    return float("nan")


def ranking_metric_for_family(family: str) -> tuple[str, str, str]:
    if family == "ppr_payload_swap":
        return "content", "transport_follow", "payload_message_transport"
    if family == "ppr_struct_swap":
        return "structure", "routing_follow", "structural_routing"
    if family == "voronoi_anchor_marker_swap":
        return "content", "routing_follow", "symbolic_anchor_routing"
    if family == "voronoi_struct_swap":
        return "structure", "routing_follow", "structural_routing"
    if family == "voronoi_payload_swap":
        return "content", "transport_follow", "payload_message_transport"
    raise ValueError(family)


def load_head_rankings(
    cfg: Mapping[str, Any],
    task: str,
    all_heads: Sequence[tuple[int, int]],
) -> dict[str, tuple[list[tuple[int, int]], str]]:
    rows = [
        row
        for row in read_csv_dicts(metrics_dir(cfg) / "specialisation_scores.csv")
        if row.get("task") == task
    ]
    fallback = (list(all_heads), "fallback_layer_head_order")
    out: dict[str, tuple[list[tuple[int, int]], str]] = {}
    for family in task_families(task):
        intervention, metric, score_name = ranking_metric_for_family(family)
        scored = []
        for row in rows:
            if row.get("intervention") != intervention or row.get("metric") != metric:
                continue
            if row.get("block", "all") != "all":
                continue
            try:
                head = (int(row["layer"]), int(row["head"]))
            except Exception:
                continue
            value = metric_value(row)
            if math.isfinite(value):
                scored.append((head, value))
        if not scored:
            out[family] = fallback
            continue
        scored.sort(key=lambda item: item[1], reverse=True)
        out[family] = ([head for head, _value in scored], score_name)
    return out


def patch_specs_from_scores(cfg: Mapping[str, Any], task: str, model: nn.Module) -> list[dict[str, Any]]:
    layers = len(model.layers)
    first_layer = model.layers[0]
    heads = int(getattr(first_layer, "num_heads", cfg["model"]["num_heads"]))
    all_heads = [(layer, head) for layer in range(layers) for head in range(heads)]
    rankings = load_head_rankings(cfg, task, all_heads)
    rng = random.Random(int(cfg["seeds"]["intervention_seed"]) + 303)
    specs = []
    for family in (task_families(task)):
        ranked_heads, score_name = rankings.get(family, (all_heads, "fallback_layer_head_order"))
        mismatched_family = next(
            other for other in rankings if other != family
        ) if len(rankings) > 1 else family
        mismatched_heads, mismatched_score = rankings.get(mismatched_family, (all_heads, "fallback_layer_head_order"))
        matched = PATHWAY_BY_FAMILY[family]
        for component in cfg["patching"]["components"]:
            if component == "attn_probs":
                target = "K"
            elif component == "message_pre_weight":
                target = "M"
            elif component == "pair_state":
                target = "KM"
            else:
                target = matched
            for layer, head in all_heads:
                specs.append(
                    {
                        "patch_id": f"{family}__{component}__L{layer}H{head}",
                        "task": task,
                        "family": family,
                        "component": component,
                        "layer": layer,
                        "head": head,
                        "head_group": [(layer, head)],
                        "group_name": "single_head",
                        "matched_high_level_target": target,
                        "ranking_score_used": "all_heads",
                        "is_control": False,
                        "control_type": "",
                    }
                )
            for size in cfg["patching"]["group_sizes"]:
                size = min(int(size), len(all_heads))
                group = ranked_heads[:size]
                specs.append(
                    {
                        "patch_id": f"{family}__{component}__top{size}_fallback_order",
                        "task": task,
                        "family": family,
                        "component": component,
                        "layer": "",
                        "head": "",
                        "head_group": group,
                        "group_name": f"top{size}",
                        "matched_high_level_target": target,
                        "ranking_score_used": score_name,
                        "is_control": False,
                        "control_type": "",
                    }
                )
                mismatch_group = mismatched_heads[:size]
                specs.append(
                    {
                        "patch_id": f"{family}__{component}__mismatched_top{size}",
                        "task": task,
                        "family": family,
                        "component": component,
                        "layer": "",
                        "head": "",
                        "head_group": mismatch_group,
                        "group_name": f"mismatched_top{size}",
                        "matched_high_level_target": target,
                        "ranking_score_used": mismatched_score,
                        "is_control": True,
                        "control_type": "mismatched_specialisation",
                    }
                )
                for control_idx in range(int(cfg["patching"].get("random_groups_per_size", 0))):
                    random_group = rng.sample(all_heads, k=size)
                    specs.append(
                        {
                            "patch_id": f"{family}__{component}__random{size}_{control_idx:03d}",
                            "task": task,
                            "family": family,
                            "component": component,
                            "layer": "",
                            "head": "",
                            "head_group": random_group,
                            "group_name": f"random{size}",
                            "matched_high_level_target": target,
                            "ranking_score_used": "random_same_size",
                            "is_control": True,
                            "control_type": "random_head_group",
                        }
                    )
    return specs


def run_patching(
    cfg: Mapping[str, Any],
    task: str,
    *,
    device_name: str = "auto",
    checkpoint: Optional[Path] = None,
    backend: Optional[str] = None,
) -> None:
    if not bool(cfg["patching"].get("allow_failed_gate_patching", False)) and not task_clean_gate(cfg, task):
        raise RuntimeError(f"clean performance gate failed for {task}; refusing mediation patching")
    if (backend or cfg["model"].get("backend", "official")) != "official":
        raise RuntimeError("interchange patching requires official GRIT backend")
    device = choose_device(device_name)
    configure_runtime(cfg, device)
    model, _run_cfg = load_model_from_checkpoint(cfg, task, checkpoint, device, backend=backend)
    interventions = torch.load(intervention_dir(cfg, task) / "patch_eval_interventions.pt", map_location="cpu", weights_only=False)
    max_per_family = int(cfg["patching"]["max_interventions_per_family"])
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interventions:
        if row["effect_bin"] in {"high", "null"} and len(by_family[row["family"]]) < max_per_family:
            by_family[row["family"]].append(row)
    selected = [row for family_rows in by_family.values() for row in family_rows]
    specs = patch_specs_from_scores(cfg, task, model)
    print(
        f"[patching] start task={task} interventions={len(selected)} specs={len(specs)} "
        f"components={','.join(cfg['patching']['components'])}",
        flush=True,
    )
    rows = []
    start_time = time.time()
    for idx, intervention in enumerate(selected):
        base_batch = collate_records([intervention["base_graph"]]).to(device)
        source_batch = collate_records([intervention["source_graph"]]).to(device)
        y_base, _base_cache = capture_prediction(model, base_batch)
        y_source, source_cache = capture_prediction(model, source_batch)
        n = int(intervention["base_graph"]["n"])
        deltas = pathway_deltas(intervention["base_graph"], intervention["source_graph"])
        delta_total_model = (y_source[0, :n] - y_base[0, :n]).detach().cpu()
        for spec in [item for item in specs if item["family"] == intervention["family"]]:
            with CFIMPatchContext(
                model,
                batch=base_batch,
                source_cache=source_cache,
                component=spec["component"],
                head_group=spec["head_group"],
            ):
                with torch.no_grad():
                    y_patch = model(base_batch).detach()
            delta_patch = (y_patch[0, :n].cpu() - y_base[0, :n].cpu())
            target_name = spec["matched_high_level_target"]
            target = deltas["total"] if target_name in {"KM", "total"} else deltas[target_name]
            primary = mediation_metrics(delta_patch, target, delta_total_model)
            wrong_target = "M" if target_name == "K" else "K"
            wrong = mediation_metrics(delta_patch, deltas[wrong_target], delta_total_model)
            rows.append(
                {
                    "task": task,
                    "intervention_id": intervention["intervention_id"],
                    "graph_id": intervention["graph_id"],
                    "family": intervention["family"],
                    "effect_bin": intervention["effect_bin"],
                    "component": spec["component"],
                    "layer": spec["layer"],
                    "head": spec["head"],
                    "group_name": spec["group_name"],
                    "head_group": json.dumps(spec["head_group"]),
                    "matched_high_level_target": target_name,
                    "ranking_score_used": spec["ranking_score_used"],
                    "is_control": spec["is_control"],
                    "control_type": spec["control_type"],
                    **primary,
                    "wrong_pathway_target": wrong_target,
                    "wrong_pathway_TCM": wrong["TCM"],
                    "wrong_pathway_TCMA": wrong["TCMA"],
                }
            )
        elapsed = time.time() - start_time
        rate = (idx + 1) / max(elapsed, 1.0)
        print(
            f"[patching] task={task} intervention={idx + 1}/{len(selected)} "
            f"family={intervention['family']} rows={len(rows)} rate={rate:.2f}/s elapsed={elapsed:.1f}s",
            flush=True,
        )
    single = [row for row in rows if row["group_name"] == "single_head"]
    grouped = [row for row in rows if row["group_name"] != "single_head"]
    single_path = metrics_dir(cfg) / "patch_single_head_metrics.csv"
    group_path = metrics_dir(cfg) / "patch_group_metrics.csv"
    old_single = [row for row in read_csv_dicts(single_path) if row.get("task") != task]
    old_group = [row for row in read_csv_dicts(group_path) if row.get("task") != task]
    write_csv(single_path, old_single + single)
    write_csv(group_path, old_group + grouped)
    write_mediation_decisions(cfg)
    plot_patching_summaries(cfg)


def collect_official_fields_and_prediction(
    model: nn.Module,
    batch: CFIMBatch,
    *,
    max_nodes: int,
) -> tuple[torch.Tensor, list[Any]]:
    from graph_specialisation_metrics.core_interpretability_specialisation_metrics import OfficialGRITFieldCollector

    collector = OfficialGRITFieldCollector(model, max_nodes=max_nodes)
    records: list[Any] = []
    handles = []

    def make_hook(layer_idx: int):
        def hook(module: nn.Module, inputs: tuple[Any, ...], _outputs: Any) -> None:
            pyg_batch = inputs[0]
            records.append(collector._densify(layer_idx, module, pyg_batch, batch.node_mask))

        return hook

    model.eval()
    for layer_idx, layer in enumerate(model.layers):
        handles.append(layer.attention.register_forward_hook(make_hook(layer_idx)))
    try:
        with torch.no_grad():
            pred = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    records.sort(key=lambda item: item.layer)
    if not records:
        raise RuntimeError("no official GRIT fields collected for response predictivity")
    return pred.detach(), records


def masked_query_mean(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weights = valid.to(scores.dtype)
    return (scores * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)


def exact_response_score_features(
    clean_layers: Sequence[Any],
    source_layers: Sequence[Any],
    base_batch: CFIMBatch,
    perm_pos: torch.Tensor,
    swap_u: torch.Tensor,
    swap_v: torch.Tensor,
    *,
    centered: bool,
) -> dict[str, torch.Tensor]:
    from graph_specialisation_metrics.core_interpretability_specialisation_metrics import (
        block_key_membership,
        cosine_by_query,
        expand_mask,
        key_swap_field,
        transposition_valid_for_block,
    )

    block_mask = block_key_membership(base_batch, "all")
    out: dict[str, torch.Tensor] = {}
    for clean, source in zip(clean_layers, source_layers):
        for field_name, clean_field, source_field in [
            ("routing", clean.attention, source.attention),
            ("transport", clean.message, source.message),
        ]:
            ref = key_swap_field(clean_field, perm_pos)
            clean_mask = expand_mask(clean.mask, clean_field)
            source_mask = expand_mask(source.mask, source_field)
            ref_mask = key_swap_field(clean_mask.long(), perm_pos).bool()
            block_m = block_mask[:, None, :, :].to(device=clean_field.device)
            if block_m.size(1) == 1 and clean_field.size(1) != 1:
                block_m = block_m.expand(-1, clean_field.size(1), -1, -1)
            query_valid = transposition_valid_for_block(block_mask, swap_u, swap_v).to(device=clean_field.device)
            if query_valid.size(1) == 1 and clean_field.size(1) != 1:
                query_valid = query_valid.expand(-1, clean_field.size(1), -1)
            stable_mask = clean_mask & source_mask & block_m
            follow_mask = ref_mask & source_mask & block_m
            valid_score = (stable_mask.any(dim=3) | follow_mask.any(dim=3)) & query_valid
            invariant = cosine_by_query(source_field, clean_field, stable_mask, centered=centered)
            follow = cosine_by_query(source_field, ref, follow_mask, centered=centered)
            out[f"L{clean.layer}_{field_name}_invariant"] = masked_query_mean(invariant, valid_score)
            out[f"L{clean.layer}_{field_name}_follow"] = masked_query_mean(follow, valid_score)
    return out


def select_response_predictivity_interventions(
    cfg: Mapping[str, Any],
    task: str,
    *,
    kind: str,
    families: Sequence[str],
    max_per_family: int,
    seed: int,
) -> list[dict[str, Any]]:
    path = intervention_dir(cfg, task) / f"{kind}_interventions.pt"
    interventions = torch.load(path, map_location="cpu", weights_only=False)
    rng = random.Random(int(seed))
    selected: list[dict[str, Any]] = []
    for family in families:
        rows = [row for row in interventions if row.get("family") == family]
        if max_per_family > 0 and len(rows) > max_per_family:
            by_bin: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_bin[str(row.get("effect_bin", ""))].append(row)
            quota = max(1, max_per_family // max(1, len(by_bin)))
            sampled: list[dict[str, Any]] = []
            for bin_rows in by_bin.values():
                pool = list(bin_rows)
                rng.shuffle(pool)
                sampled.extend(pool[:quota])
            if len(sampled) < max_per_family:
                used = {str(row.get("intervention_id")) for row in sampled}
                remainder = [row for row in rows if str(row.get("intervention_id")) not in used]
                rng.shuffle(remainder)
                sampled.extend(remainder[: max_per_family - len(sampled)])
            rows = sampled[:max_per_family]
        selected.extend(rows)
    selected.sort(key=lambda row: (str(row.get("family")), str(row.get("graph_id")), str(row.get("intervention_id"))))
    return selected


def effect_targets_for_intervention(
    row: Mapping[str, Any],
    y_base: torch.Tensor,
    y_source: torch.Tensor,
) -> dict[str, float]:
    n = int(row["base_graph"]["n"])
    family = str(row["family"])
    pathway = PATHWAY_BY_FAMILY[family]
    deltas = pathway_deltas(row["base_graph"], row["source_graph"])
    teacher_path = deltas[pathway][:n].float()
    teacher_total = deltas["total"][:n].float()
    student = (y_source[:n] - y_base[:n]).detach().cpu().float()
    path_flat = teacher_path.reshape(-1)
    total_flat = teacher_total.reshape(-1)
    student_flat = student.reshape(-1)
    path_norm = torch.linalg.vector_norm(path_flat).clamp_min(EPS)
    total_norm = torch.linalg.vector_norm(total_flat).clamp_min(EPS)
    student_norm = torch.linalg.vector_norm(student_flat).clamp_min(EPS)
    dot_path = torch.dot(student_flat, path_flat)
    dot_total = torch.dot(student_flat, total_flat)
    return {
        "teacher_pathway_norm": float(path_norm),
        "teacher_total_norm": float(total_norm),
        "student_delta_norm": float(student_norm),
        "student_teacher_pathway_projection": float(dot_path / path_norm),
        "student_teacher_pathway_beta": float(dot_path / path_norm.square().clamp_min(EPS)),
        "student_teacher_pathway_cea": float(dot_path / (student_norm * path_norm).clamp_min(EPS)),
        "student_teacher_total_projection": float(dot_total / total_norm),
        "student_teacher_total_beta": float(dot_total / total_norm.square().clamp_min(EPS)),
        "student_teacher_total_cea": float(dot_total / (student_norm * total_norm).clamp_min(EPS)),
    }


def compute_response_predictivity_table(
    cfg: Mapping[str, Any],
    task: str,
    *,
    interventions: Sequence[Mapping[str, Any]],
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    centered: bool,
) -> list[dict[str, Any]]:
    max_nodes = max(int(row["base_graph"]["n"]) for row in interventions)
    rows: list[dict[str, Any]] = []
    total_chunks = max(1, math.ceil(len(interventions) / int(batch_size)))
    start_time = time.time()
    for start in range(0, len(interventions), int(batch_size)):
        chunk = list(interventions[start : start + int(batch_size)])
        base_batch = collate_records([row["base_graph"] for row in chunk]).to(device)
        source_batch = collate_records([row["source_graph"] for row in chunk]).to(device)
        perm_pos = torch.stack(
            [transposition_perm(int(row["base_graph"]["n"]), int(row["u"]), int(row["v"])) for row in chunk],
            dim=0,
        ).to(device)
        swap_u = torch.tensor([int(row["u"]) for row in chunk], dtype=torch.long, device=device)
        swap_v = torch.tensor([int(row["v"]) for row in chunk], dtype=torch.long, device=device)
        y_base, clean_layers = collect_official_fields_and_prediction(model, base_batch, max_nodes=max_nodes)
        y_source, source_layers = collect_official_fields_and_prediction(model, source_batch, max_nodes=max_nodes)
        features = exact_response_score_features(
            clean_layers,
            source_layers,
            base_batch,
            perm_pos,
            swap_u,
            swap_v,
            centered=centered,
        )
        feature_cpu = {name: value.detach().cpu() for name, value in features.items()}
        for idx, row in enumerate(chunk):
            target_values = effect_targets_for_intervention(row, y_base[idx], y_source[idx])
            out = {
                "task": task,
                "intervention_id": row["intervention_id"],
                "graph_id": row["graph_id"],
                "family": row["family"],
                "effect_bin": row["effect_bin"],
                "pathway_target": row["pathway_target"],
                "u": int(row["u"]),
                "v": int(row["v"]),
                "centered": bool(centered),
                **target_values,
            }
            for name, tensor in feature_cpu.items():
                for head in range(tensor.size(1)):
                    out[f"{name}_H{head}"] = float(tensor[idx, head])
            rows.append(out)
        chunk_idx = start // int(batch_size) + 1
        elapsed = time.time() - start_time
        print(
            f"[response-predictivity] features task={task} chunk={chunk_idx}/{total_chunks} "
            f"rows={len(rows)} elapsed={elapsed:.1f}s",
            flush=True,
        )
    return rows


def numeric_matrix(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(rows), 0), dtype=np.float64)
    matrix = np.empty((len(rows), len(columns)), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for col_idx, col in enumerate(columns):
            try:
                value = float(row[col])
            except Exception:
                value = float("nan")
            matrix[row_idx, col_idx] = value
    return matrix


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    aa = a - float(np.mean(a))
    bb = b - float(np.mean(b))
    denom = math.sqrt(float(np.dot(aa, aa) * np.dot(bb, bb)))
    if denom <= EPS:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_corr(rank_values(a), rank_values(b))


def partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[mask]
    y = y[mask]
    z = z[mask]
    if len(x) < 3:
        return float("nan")
    design = np.column_stack([np.ones(len(z), dtype=np.float64), z])
    try:
        beta_x = np.linalg.lstsq(design, x, rcond=None)[0]
        beta_y = np.linalg.lstsq(design, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return float("nan")
    return pearson_corr(x - design @ beta_x, y - design @ beta_y)


def distance_stratum(distance: int, gnn_depth: int = 2) -> str:
    d = int(distance)
    depth = max(1, int(gnn_depth))
    if d <= 0:
        return "disconnected"
    if d == 1:
        return "d1"
    if d <= depth:
        return "d2_to_L"
    if d <= 2 * depth:
        return "dL1_to_2L"
    return "d_gt_2L"


def graph_sum_teacher(record: Mapping[str, Any]) -> torch.Tensor:
    return record["teacher"]["Y"].float().sum(dim=0)


def graph_sum_response_delta(base: Mapping[str, Any], source: Mapping[str, Any]) -> torch.Tensor:
    return graph_sum_teacher(source) - graph_sum_teacher(base)


def teacher_far_response_fraction(
    base: Mapping[str, Any],
    source: Mapping[str, Any],
    u: int,
    v: int,
    *,
    gnn_depth: int = 2,
) -> float:
    n = int(base["n"])
    dy_nodes = source["teacher"]["Y"].float()[:n] - base["teacher"]["Y"].float()[:n]
    node_mass = dy_nodes.square().sum(dim=1)
    total = float(node_mass.sum().item())
    if total <= EPS:
        return 0.0
    spd = base["struct"]["shortest_path_distance"]
    dist_to_swapped = torch.minimum(spd[:n, int(u)], spd[:n, int(v)])
    far_mass = float(node_mass[dist_to_swapped > int(gnn_depth)].sum().item())
    return float(far_mass / total)


def effective_resistance_matrix(record: Mapping[str, Any]) -> torch.Tensor:
    adj = record["struct"]["adjacency"].float()
    degree = torch.diag(adj.sum(dim=1))
    lap = degree - adj
    pinv = torch.linalg.pinv(lap)
    diag = torch.diag(pinv)
    resistance = diag[:, None] + diag[None, :] - 2.0 * pinv
    return resistance.clamp_min(0.0)


def default_functional_family(task: str) -> str:
    if task == "ppr_diffusion":
        return "ppr_payload_swap"
    if task == "nearest_anchor_voronoi":
        return "voronoi_payload_swap"
    if task == "local_mean_gcn":
        return "local_mean_payload_swap"
    raise ValueError(task)


def functional_swap_cache_path(cfg: Mapping[str, Any], task: str, seed: int) -> Path:
    return functional_swaps_dir(cfg, task) / f"content_swaps_seed{int(seed)}.pt"


def functional_response_table_path(cfg: Mapping[str, Any], task: str, seed: int) -> Path:
    return functional_responses_dir(cfg, task) / f"functional_response_table_seed{int(seed)}.csv"


def sample_functional_swaps_from_records(
    records: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    task: str,
    *,
    family: str | None = None,
    num_graphs: int = 200,
    swaps_per_stratum: int = 50,
    gnn_depth: int = 2,
    seed: int = 9101,
) -> list[dict[str, Any]]:
    selected_records = list(records)[: int(num_graphs)]
    family = family or default_functional_family(task)
    cap = int(swaps_per_stratum)
    rng = random.Random(int(seed))
    swaps: list[dict[str, Any]] = []
    for graph_idx, base in enumerate(selected_records):
        resistance = effective_resistance_matrix(base)
        grouped: dict[str, list[tuple[int, int, int, float]]] = {key: [] for key in FUNCTIONAL_STRATA}
        spd = base["struct"]["shortest_path_distance"]
        for u, v in candidate_pairs_for_family(base, family):
            d_uv = int(spd[int(u), int(v)])
            stratum = distance_stratum(d_uv, gnn_depth)
            if stratum not in grouped:
                continue
            grouped[stratum].append((int(u), int(v), d_uv, float(resistance[int(u), int(v)].item())))
        for stratum in FUNCTIONAL_STRATA:
            candidates = list(grouped[stratum])
            if len(candidates) > cap:
                candidates = rng.sample(candidates, k=cap)
            candidates.sort(key=lambda item: (item[2], item[0], item[1]))
            for local_idx, (u, v, d_uv, r_eff) in enumerate(candidates):
                source = source_for_intervention(base, family, u, v, cfg)
                dy = graph_sum_response_delta(base, source)
                swap_id = f"{base['graph_id']}__{family}_{u}_{v}"
                rho_far = teacher_far_response_fraction(base, source, u, v, gnn_depth=gnn_depth)
                swaps.append(
                    {
                        "task": task,
                        "family": family,
                        "graph_index": graph_idx,
                        "graph_id": str(base["graph_id"]),
                        "swap_id": swap_id,
                        "swap_index_in_stratum": local_idx,
                        "u": u,
                        "v": v,
                        "d_uv": d_uv,
                        "stratum": stratum,
                        "R_eff_uv": r_eff,
                        "dy": dy.detach().cpu(),
                        "dy_norm": float(torch.linalg.vector_norm(dy.float()).item()),
                        "rho_far": rho_far,
                    }
                )
    return swaps


def build_functional_swaps(
    cfg: Mapping[str, Any],
    task: str,
    *,
    family: str | None = None,
    num_graphs: int = 200,
    swaps_per_stratum: int = 50,
    gnn_depth: int = 2,
    seed: int = 9101,
    force: bool = False,
) -> Path:
    cache_data(cfg, task, force=False)
    out_path = functional_swap_cache_path(cfg, task, seed)
    if out_path.exists() and not force:
        print(f"[functional-swaps] using existing {out_path}", flush=True)
        return out_path
    records = load_records(data_dir(cfg, task) / "test_id.pt")[: int(num_graphs)]
    family = family or default_functional_family(task)
    swaps = sample_functional_swaps_from_records(
        records,
        cfg,
        task,
        family=family,
        num_graphs=num_graphs,
        swaps_per_stratum=swaps_per_stratum,
        gnn_depth=gnn_depth,
        seed=seed,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": task,
            "family": family,
            "seed": int(seed),
            "num_graphs": int(num_graphs),
            "swaps_per_stratum": int(swaps_per_stratum),
            "gnn_depth": int(gnn_depth),
            "base_records": records,
            "swaps": swaps,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
        },
        out_path,
    )
    counts = defaultdict(int)
    for row in swaps:
        counts[str(row["stratum"])] += 1
    print(
        f"[functional-swaps] wrote {out_path} swaps={len(swaps)} "
        + " ".join(f"{key}={counts[key]}" for key in FUNCTIONAL_STRATA),
        flush=True,
    )
    return out_path


def load_functional_swap_cache(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def default_model_config_path(task: str, model_name: str) -> Path | None:
    prefix = "gcn_plus" if model_name == "gcn_plus" else "grit"
    path = Path("experiments/synthetic/cfim/configs") / f"{prefix}_{task}.yaml"
    return path if path.exists() else None


def load_stage1_model_config(
    base_cfg: Mapping[str, Any],
    task: str,
    *,
    model_name: str,
    config_path: Path | None = None,
    fast_dev_run: bool = False,
) -> dict[str, Any]:
    resolved_path = config_path or default_model_config_path(task, model_name)
    if resolved_path is not None:
        cfg = load_config(resolved_path, task=task, fast_dev_run=fast_dev_run)
    elif model_name == "gcn_plus":
        cfg = gnnplus_default_config(task, layer_type="gcn")
    else:
        cfg = copy.deepcopy(dict(base_cfg))
        cfg["task"] = task
    cfg["artifacts"]["root"] = str(artifact_root(base_cfg))
    return cfg


@torch.no_grad()
def predict_node_outputs(
    model: nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    model.eval()
    outputs: list[torch.Tensor] = []
    for start in range(0, len(records), int(batch_size)):
        chunk = records[start : start + int(batch_size)]
        batch = collate_records(chunk).to(device)
        pred = model(batch).detach().cpu()
        for idx, record in enumerate(chunk):
            outputs.append(pred[idx, : int(record["n"])].clone())
    return outputs


def graph_sum_from_node_output(output: torch.Tensor) -> torch.Tensor:
    return output.float().sum(dim=0)


def projection_and_cosine(delta: torch.Tensor, oracle_delta: torch.Tensor) -> tuple[float, float]:
    delta = delta.float()
    oracle_delta = oracle_delta.float()
    oracle_norm = torch.linalg.vector_norm(oracle_delta).clamp_min(float(EPS))
    delta_norm = torch.linalg.vector_norm(delta).clamp_min(float(EPS))
    projection = float(torch.dot(delta, oracle_delta).item() / oracle_norm.item())
    cosine = float(torch.dot(delta, oracle_delta).item() / (oracle_norm.item() * delta_norm.item()))
    return projection, cosine


def clean_performance_context_rows(
    task: str,
    records: Sequence[Mapping[str, Any]],
    base_outputs: Mapping[str, Sequence[torch.Tensor]],
) -> list[dict[str, Any]]:
    rows = []
    teacher_sums = torch.stack([graph_sum_teacher(record) for record in records], dim=0)
    teacher_graph_var = teacher_sums.float().var(unbiased=False).clamp_min(1.0e-12)
    for model_name in FUNCTIONAL_MODELS:
        node_rels = []
        pred_sums = []
        for idx, record in enumerate(records):
            pred = base_outputs[model_name][idx].float()
            target = record["teacher"]["Y"].float()
            mse = (pred - target).square().mean()
            var = target.var(unbiased=False).clamp_min(1.0e-12)
            node_rels.append(float((mse / var).item()))
            pred_sums.append(graph_sum_from_node_output(pred))
        pred_sum_tensor = torch.stack(pred_sums, dim=0)
        graph_relmse = float(((pred_sum_tensor - teacher_sums).square().mean() / teacher_graph_var).item())
        rows.append(
            {
                "task": task,
                "model": model_name,
                "graphs": len(records),
                "node_relmse_mean": float(np.mean(node_rels)) if node_rels else float("nan"),
                "node_relmse_median": float(np.median(node_rels)) if node_rels else float("nan"),
                "graph_sum_relmse": graph_relmse,
            }
        )
    return rows


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def validity_status(value: float, *, pass_if: Any, warn_if: Any | None = None) -> str:
    if math.isnan(value):
        return "fail"
    if bool(pass_if(value)):
        return "pass"
    if warn_if is not None and bool(warn_if(value)):
        return "warn"
    return "fail"


def metric_summary_value(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    stratum: str,
    stat: str,
    field: str = "mean",
) -> float:
    for row in rows:
        if row.get("task") == task and row.get("stratum") == stratum and row.get("stat") == stat:
            return finite_float(row.get(field))
    return float("nan")


def summarise_functional_validity_gate(
    cfg: Mapping[str, Any],
    *,
    tasks: Sequence[str] | None = None,
    e1g_seed: int = 9101,
    e1n_seed: int = 9301,
    gnn_depth: int = 2,
) -> Path:
    tasks = list(tasks or TASKS)
    metrics = functional_metrics_dir(cfg)
    e1_path = metrics / "functional_e1_fingerprint_stats.csv"
    e1_rho_path = metrics / "functional_e1_fingerprint_rhofar_stats.csv"
    clean_path = metrics / "functional_e1_clean_performance_context.csv"
    e1n_path = metrics / "functional_e1n_node_fingerprint_stats.csv"
    e1n_validity_path = metrics / "functional_e1n_validity.csv"
    required_global = [e1_path, e1_rho_path, clean_path, e1n_path, e1n_validity_path]
    e1_rows = read_csv_dicts(e1_path) if e1_path.exists() else []
    e1_rho_rows = read_csv_dicts(e1_rho_path) if e1_rho_path.exists() else []
    clean_rows = read_csv_dicts(clean_path) if clean_path.exists() else []
    e1n_rows = read_csv_dicts(e1n_path) if e1n_path.exists() else []
    e1n_validity_rows = read_csv_dicts(e1n_validity_path) if e1n_validity_path.exists() else []
    out_rows: list[dict[str, Any]] = []

    def add(
        task: str,
        check: str,
        why: str,
        observed: str,
        rule: str,
        status: str,
        failed: str,
    ) -> None:
        out_rows.append(
            {
                "task": task,
                "check": check,
                "why_it_matters": why,
                "observed_value": observed,
                "pass_rule": rule,
                "status": status,
                "interpretation_if_failed": failed,
            }
        )

    for task in tasks:
        task_files = [
            functional_response_table_path(cfg, task, e1g_seed),
            functional_responses_dir(cfg, task) / f"functional_clean_context_seed{int(e1g_seed)}.csv",
            functional_node_graph_stats_path(cfg, task, e1n_seed),
            functional_node_validity_path(cfg, task, e1n_seed),
        ]
        missing = [str(path) for path in required_global + task_files if not path.exists()]
        empty = [
            str(path)
            for path in required_global + task_files
            if path.exists() and path.stat().st_size == 0
        ]
        status = "pass" if not missing and not empty else "fail"
        observed = "all required files present" if status == "pass" else f"missing={len(missing)} empty={len(empty)}"
        add(
            task,
            "Stage 1 artifacts",
            "Q1 reuses E1-G responses, E1-N theorem checks, and clean context.",
            observed,
            "all required CSV/PT artifacts exist and are non-empty",
            status,
            "Q1 can only be run as a diagnostic; first regenerate missing Stage 1 outputs.",
        )

        for model_name in FUNCTIONAL_MODELS:
            clean = next(
                (row for row in clean_rows if row.get("task") == task and row.get("model") == model_name),
                None,
            )
            node_rel = finite_float(clean.get("node_relmse_mean") if clean is not None else None)
            graph_rel = finite_float(clean.get("graph_sum_relmse") if clean is not None else None)
            clean_status = validity_status(
                node_rel,
                pass_if=lambda value: value <= 0.02,
                warn_if=lambda value: value <= 0.05,
            )
            add(
                task,
                f"Clean performance: {model_name}",
                "Functional response comparisons require both students to be competent on clean graphs.",
                f"node_relmse_mean={node_rel:.4g}; graph_sum_relmse={graph_rel:.4g}",
                "pass <= 0.02; warn <= 0.05; fail > 0.05",
                clean_status,
                "Alignment/excess may mostly reflect undertraining rather than mechanism.",
            )

        validity = next((row for row in e1n_validity_rows if row.get("task") == task), None)
        violations = finite_float(validity.get("beyond_L_gcn_violations") if validity is not None else None)
        max_norm = finite_float(validity.get("beyond_L_gcn_max_norm") if validity is not None else None)
        leakage_status = "pass" if violations == 0.0 else "fail"
        add(
            task,
            "GCN+ beyond-L leakage",
            "The E1-N theorem-null only holds if the GCN+ is silent beyond its receptive field.",
            f"violations={violations:.0f}; max_norm={max_norm:.4g}; L={int(gnn_depth)}",
            "beyond_L_gcn_violations == 0",
            leakage_status,
            "Beyond-L GRIT advantage is not interpretable as certified dense-support use.",
        )

        usable_bins = []
        for bin_name in RHO_FAR_BINS:
            bin_rows = [row for row in e1_rho_rows if row.get("task") == task and row.get("rho_far_bin") == bin_name]
            if not bin_rows:
                continue
            swaps = max(finite_float(row.get("swaps"), 0.0) for row in bin_rows)
            low = min(finite_float(row.get("rho_far_low")) for row in bin_rows)
            high = max(finite_float(row.get("rho_far_high")) for row in bin_rows)
            if swaps >= 100 and math.isfinite(low) and math.isfinite(high) and high >= low:
                usable_bins.append((bin_name, swaps, low, high))
        if len(usable_bins) >= 4:
            rho_status = "pass"
        elif len(usable_bins) >= 2:
            rho_status = "warn"
        else:
            rho_status = "fail"
        ranges = "; ".join(f"{name}:{low:.2f}-{high:.2f},n={int(swaps)}" for name, swaps, low, high in usable_bins)
        add(
            task,
            "rho_far bins usable",
            "Consequence-range rebinning is only meaningful when bins have enough non-degenerate mass.",
            ranges or "no usable rho_far bins",
            "pass: 4 bins with >=100 swaps; warn: >=2 bins; fail: <2 bins",
            rho_status,
            "rho_far-conditioned Q1 should be ignored or treated as underpowered.",
        )

    if "local_mean_gcn" in tasks:
        local_demand = metric_summary_value(
            e1n_rows,
            task="local_mean_gcn",
            stratum="d_gt_L",
            stat="oracle_demand_mean",
        )
        local_status = validity_status(
            local_demand,
            pass_if=lambda value: value <= 1.0e-6,
            warn_if=lambda value: value <= 1.0e-4,
        )
        add(
            "local_mean_gcn",
            "Local control beyond-L oracle demand",
            "The local teacher should not require a response outside the GCN+ receptive field.",
            f"E||dy_i||={local_demand:.4g}",
            "pass <= 1e-6; warn <= 1e-4",
            local_status,
            "The negative control is not local under this intervention dictionary.",
        )
        local_spurious = metric_summary_value(
            e1n_rows,
            task="local_mean_gcn",
            stratum="d_gt_L",
            stat="grit_spurious_silent_norm_mean",
        )
        local_excess = metric_summary_value(
            e1n_rows,
            task="local_mean_gcn",
            stratum="d_gt_L",
            stat="grit_excess_partial_corr",
        )
        finite_excess_ok = (not math.isfinite(local_excess)) or abs(local_excess) <= 0.20
        finite_excess_warn = (not math.isfinite(local_excess)) or abs(local_excess) <= 0.40
        if local_spurious <= 1.0e-3 and finite_excess_ok:
            status = "pass"
        elif local_spurious <= 1.0e-2 and finite_excess_warn:
            status = "warn"
        else:
            status = "fail"
        add(
            "local_mean_gcn",
            "Local control GRIT beyond-L signal",
            "GRIT should not show strong global functional sensitivity when the teacher is local.",
            f"spurious_norm={local_spurious:.4g}; excess={local_excess:.4g}",
            "pass: spurious <= 1e-3 and |excess| <= 0.20 if finite",
            status,
            "Q1 gate-excess coupling may reflect generic global sensitivity rather than task-relevant computation.",
        )

    if "nearest_anchor_voronoi" in tasks:
        vor_demand = metric_summary_value(
            e1n_rows,
            task="nearest_anchor_voronoi",
            stratum="d_gt_L",
            stat="oracle_demand_mean",
        )
        vor_status = validity_status(
            vor_demand,
            pass_if=lambda value: value >= 1.0e-3,
            warn_if=lambda value: value >= 1.0e-5,
        )
        add(
            "nearest_anchor_voronoi",
            "Voronoi beyond-L oracle demand",
            "The global positive case must actually require beyond-local node responses.",
            f"E||dy_i||={vor_demand:.4g}",
            "pass >= 1e-3; warn >= 1e-5",
            vor_status,
            "A null Q1 result would be uninformative because the teacher barely demands global response.",
        )

    path = functional_validity_gate_path(cfg)
    write_csv(path, out_rows)
    print(f"[functional-validity] wrote {path}", flush=True)
    return path


def wrap_cell(text: Any, width: int) -> str:
    words = str(text).split()
    lines: list[str] = []
    cur: list[str] = []
    for word in words:
        if sum(len(part) for part in cur) + len(cur) + len(word) > width and cur:
            lines.append(" ".join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def plot_functional_validity_gate(cfg: Mapping[str, Any], *, tasks: Sequence[str] | None = None) -> Path:
    tasks = list(tasks or TASKS)
    rows = read_csv_dicts(functional_validity_gate_path(cfg))
    if not rows:
        raise FileNotFoundError("functional validity gate CSV is missing or empty")
    out = functional_figures_dir(cfg) / "functional_validity_gate_table.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    status_colors = {"pass": "#d9f0e3", "warn": "#fff1c7", "fail": "#f7d6d0"}
    columns = ["Check", "Why it matters", "Observed value", "Pass rule", "Status", "Interpretation if failed"]
    keys = ["check", "why_it_matters", "observed_value", "pass_rule", "status", "interpretation_if_failed"]
    widths = [22, 30, 28, 26, 8, 34]
    with PdfPages(out) as pdf:
        for task in tasks:
            task_rows = [row for row in rows if row.get("task") == task]
            if not task_rows:
                continue
            height = max(4.2, 0.78 * len(task_rows) + 1.3)
            fig, ax = plt.subplots(figsize=(16.0, height))
            ax.axis("off")
            ax.set_title(f"{task}: Stage A validity gate", fontsize=13, loc="left", pad=8)
            cell_text = [[wrap_cell(row.get(key, ""), width) for key, width in zip(keys, widths)] for row in task_rows]
            table = ax.table(
                cellText=cell_text,
                colLabels=columns,
                cellLoc="left",
                colLoc="left",
                loc="upper left",
                colWidths=[0.13, 0.20, 0.17, 0.18, 0.07, 0.25],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.8)
            for (row_idx, col_idx), cell in table.get_celld().items():
                cell.set_edgecolor("#d0d4dc")
                cell.set_linewidth(0.4)
                if row_idx == 0:
                    cell.set_facecolor("#edf2f7")
                    cell.set_text_props(weight="bold")
                else:
                    status = str(task_rows[row_idx - 1].get("status", ""))
                    if col_idx == 4:
                        cell.set_facecolor(status_colors.get(status, "#ffffff"))
                        cell.set_text_props(weight="bold")
                    else:
                        cell.set_facecolor("#ffffff")
            table.scale(1.0, 2.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"[functional-validity-plot] wrote {out}", flush=True)
    return out


@dataclass
class Q1LayerFields:
    layer: int
    attention: torch.Tensor
    message: torch.Tensor
    mask: torch.Tensor


def q1_layers_for_graph(collected_layers: Sequence[Any], batch_idx: int) -> list[Q1LayerFields]:
    return [
        Q1LayerFields(
            layer=int(layer.layer),
            attention=layer.attention[int(batch_idx)].detach().cpu().float(),
            message=layer.message[int(batch_idx)].detach().cpu().float(),
            mask=layer.mask[int(batch_idx)].detach().cpu().bool(),
        )
        for layer in collected_layers
    ]


def q1_gate_scope_values(
    layers: Sequence[Q1LayerFields],
    spd: torch.Tensor,
    *,
    u: int,
    v: int,
    gnn_depth: int = 2,
    layer_scope: int | None = None,
    head_scope: int | None = None,
) -> dict[str, float]:
    n = int(spd.size(0))
    u = int(u)
    v = int(v)
    dist_to_swap = torch.minimum(spd[:n, u], spd[:n, v])
    far_query = dist_to_swap > int(gnn_depth)
    pair_far = spd[:n, :n] > int(gnn_depth)
    swap_key = torch.zeros(n, dtype=torch.bool)
    swap_key[u] = True
    swap_key[v] = True

    route_swap_num = 0.0
    route_swap_den = 0.0
    route_global_num = 0.0
    route_global_den = 0.0
    trans_swap_num = 0.0
    trans_swap_den = 0.0
    trans_global_num = 0.0
    trans_global_den = 0.0
    selected_layers = 0
    selected_heads = 0
    valid_far_queries = 0

    for layer in layers:
        if layer_scope is not None and int(layer.layer) != int(layer_scope):
            continue
        attn = layer.attention[:, :n, :n].float()
        msg = layer.message[:, :n, :n].float()
        mask = layer.mask[:, :n, :n].bool()
        if head_scope is not None:
            attn = attn[int(head_scope) : int(head_scope) + 1]
            msg = msg[int(head_scope) : int(head_scope) + 1]
            mask = mask[int(head_scope) : int(head_scope) + 1]
        if attn.numel() == 0:
            continue
        selected_layers += 1
        selected_heads += int(attn.size(0))
        query_valid = mask.any(dim=2) & far_query[None, :]
        valid_far_queries += int(query_valid.sum().item())
        swap_mass = (attn * mask.to(attn.dtype) * swap_key[None, None, :].to(attn.dtype)).sum(dim=2)
        route_swap_num += float((swap_mass * query_valid.to(attn.dtype)).sum().item())
        route_swap_den += float(query_valid.sum().item())

        pair_mask = mask & pair_far[None, :, :]
        route_global_num += float((attn * pair_mask.to(attn.dtype)).sum().item())
        route_global_den += float((attn * mask.to(attn.dtype)).sum().item())

        msg_norm = torch.linalg.vector_norm(msg, dim=-1)
        weighted = attn * msg_norm * mask.to(attn.dtype)
        far_query_mask = far_query[None, :, None]
        swap_pair_mask = far_query_mask & swap_key[None, None, :]
        trans_swap_num += float(weighted[swap_pair_mask.expand_as(weighted)].sum().item())
        trans_swap_den += float(weighted[far_query_mask & mask].sum().item())
        trans_global_num += float(weighted[pair_mask].sum().item())
        trans_global_den += float(weighted[mask].sum().item())

    return {
        "route_swap_far_mass": route_swap_num / route_swap_den if route_swap_den > 0 else float("nan"),
        "transport_swap_far_share": trans_swap_num / trans_swap_den if trans_swap_den > 0 else float("nan"),
        "route_global_far_share": route_global_num / route_global_den if route_global_den > 0 else float("nan"),
        "transport_global_far_share": trans_global_num / trans_global_den if trans_global_den > 0 else float("nan"),
        "far_query_count": float(int(far_query.sum().item())),
        "valid_far_query_head_count": float(valid_far_queries),
        "layer_count": float(selected_layers),
        "head_count": float(selected_heads),
    }


def q1_scope_rows_for_swap(
    *,
    task: str,
    response_row: Mapping[str, Any],
    swap: Mapping[str, Any],
    base: Mapping[str, Any],
    clean_layers: Sequence[Q1LayerFields],
    source_layers: Sequence[Q1LayerFields],
    gnn_depth: int,
    store_head_gates: bool,
) -> list[dict[str, Any]]:
    scopes: list[tuple[str, str, int | None, int | None]] = [("all", "all", None, None)]
    layer_ids = [int(layer.layer) for layer in clean_layers]
    for layer_idx in layer_ids:
        scopes.append((str(layer_idx), "all", layer_idx, None))
    if store_head_gates and clean_layers:
        heads = int(clean_layers[0].attention.size(0))
        for layer_idx in layer_ids:
            for head_idx in range(heads):
                scopes.append((str(layer_idx), str(head_idx), layer_idx, head_idx))

    spd = base["struct"]["shortest_path_distance"]
    u = int(swap["u"])
    v = int(swap["v"])
    out_rows: list[dict[str, Any]] = []
    base_payload = {
        "task": task,
        "family": str(response_row.get("family", swap.get("family", ""))),
        "graph_id": str(response_row.get("graph_id", swap.get("graph_id", ""))),
        "graph_index": int(swap["graph_index"]),
        "swap_id": str(response_row.get("swap_id", swap.get("swap_id", ""))),
        "u": u,
        "v": v,
        "d_uv": int(response_row.get("d_uv", swap.get("d_uv", -1))),
        "stratum": str(response_row.get("stratum", swap.get("stratum", ""))),
        "R_eff_uv": finite_float(response_row.get("R_eff_uv", swap.get("R_eff_uv"))),
        "rho_far": finite_float(response_row.get("rho_far", swap.get("rho_far"))),
        "dy_norm": finite_float(response_row.get("dy_norm")),
        "grit_delta_norm": finite_float(response_row.get("grit_delta_norm")),
        "gcn_plus_delta_norm": finite_float(response_row.get("gcn_plus_delta_norm")),
        "grit_teacher_projection": finite_float(response_row.get("grit_teacher_projection")),
        "gcn_plus_teacher_projection": finite_float(response_row.get("gcn_plus_teacher_projection")),
        "grit_teacher_cosine": finite_float(response_row.get("grit_teacher_cosine")),
        "gcn_plus_teacher_cosine": finite_float(response_row.get("gcn_plus_teacher_cosine")),
    }
    for layer_label, head_label, layer_scope, head_scope in scopes:
        clean = q1_gate_scope_values(
            clean_layers,
            spd,
            u=u,
            v=v,
            gnn_depth=gnn_depth,
            layer_scope=layer_scope,
            head_scope=head_scope,
        )
        source = q1_gate_scope_values(
            source_layers,
            spd,
            u=u,
            v=v,
            gnn_depth=gnn_depth,
            layer_scope=layer_scope,
            head_scope=head_scope,
        )
        row = dict(base_payload)
        row.update({"layer": layer_label, "head": head_label})
        for key in ("route_swap_far_mass", "transport_swap_far_share", "route_global_far_share", "transport_global_far_share"):
            row[f"clean_{key}"] = clean[key]
            row[f"source_{key}"] = source[key]
            row[f"delta_{key}"] = source[key] - clean[key] if math.isfinite(source[key]) and math.isfinite(clean[key]) else float("nan")
        for key in ("far_query_count", "valid_far_query_head_count", "layer_count", "head_count"):
            row[key] = clean[key]
        out_rows.append(row)
    return out_rows


def run_functional_responses(
    cfg: Mapping[str, Any],
    task: str,
    *,
    swaps_path: Path | None = None,
    seed: int = 9101,
    device_name: str = "auto",
    batch_size: int = 1024,
    gnn_depth: int = 2,
    grit_config: Path | None = None,
    gcn_plus_config: Path | None = None,
    grit_checkpoint: Path | None = None,
    gcn_plus_checkpoint: Path | None = None,
    fast_dev_run: bool = False,
) -> tuple[Path, Path]:
    device = choose_device(device_name)
    configure_runtime(cfg, device)
    swaps_path = swaps_path or functional_swap_cache_path(cfg, task, seed)
    cache = load_functional_swap_cache(swaps_path)
    base_records: list[dict[str, Any]] = list(cache["base_records"])
    swaps: list[dict[str, Any]] = list(cache["swaps"])
    if fast_dev_run:
        swaps = swaps[: min(len(swaps), 32)]
    grit_cfg = load_stage1_model_config(cfg, task, model_name="grit", config_path=grit_config, fast_dev_run=fast_dev_run)
    gcn_cfg = load_stage1_model_config(cfg, task, model_name="gcn_plus", config_path=gcn_plus_config, fast_dev_run=fast_dev_run)
    grit, _ = load_model_from_checkpoint(grit_cfg, task, grit_checkpoint, device, backend="official")
    gcn_plus, _ = load_model_from_checkpoint(gcn_cfg, task, gcn_plus_checkpoint, device, backend="official_gnnplus")
    models = {"grit": grit, "gcn_plus": gcn_plus}
    print(
        f"[functional-responses] task={task} swaps={len(swaps)} graphs={len(base_records)} "
        f"device={device} batch_size={batch_size}",
        flush=True,
    )
    base_outputs = {
        model_name: predict_node_outputs(model, base_records, batch_size=batch_size, device=device)
        for model_name, model in models.items()
    }
    clean_rows = clean_performance_context_rows(task, base_records, base_outputs)
    clean_path = functional_responses_dir(cfg, task) / f"functional_clean_context_seed{int(seed)}.csv"
    write_csv(clean_path, clean_rows)
    rows: list[dict[str, Any]] = []
    chunks = max(1, math.ceil(len(swaps) / int(batch_size)))
    start_time = time.time()
    for start in range(0, len(swaps), int(batch_size)):
        chunk_swaps = swaps[start : start + int(batch_size)]
        source_records = [
            source_for_intervention(
                base_records[int(row["graph_index"])],
                str(row["family"]),
                int(row["u"]),
                int(row["v"]),
                cfg,
            )
            for row in chunk_swaps
        ]
        source_outputs = {
            model_name: predict_node_outputs(model, source_records, batch_size=batch_size, device=device)
            for model_name, model in models.items()
        }
        for row_idx, swap in enumerate(chunk_swaps):
            graph_index = int(swap["graph_index"])
            base = base_records[graph_index]
            dy = swap["dy"].float()
            out: dict[str, Any] = {
                "task": task,
                "family": str(swap["family"]),
                "graph_id": str(swap["graph_id"]),
                "graph_index": graph_index,
                "swap_id": str(swap["swap_id"]),
                "u": int(swap["u"]),
                "v": int(swap["v"]),
                "d_uv": int(swap["d_uv"]),
                "stratum": str(swap["stratum"]),
                "R_eff_uv": float(swap["R_eff_uv"]),
                "dy_norm": float(swap["dy_norm"]),
                "rho_far": float(
                    swap.get(
                        "rho_far",
                        teacher_far_response_fraction(base, source_records[row_idx], int(swap["u"]), int(swap["v"]), gnn_depth=gnn_depth),
                    )
                ),
            }
            for dim_idx, value in enumerate(dy.tolist()):
                out[f"dy_{dim_idx}"] = float(value)
            spd = base["struct"]["shortest_path_distance"]
            dist_to_swapped = torch.minimum(spd[:, int(swap["u"])], spd[:, int(swap["v"])])
            far_mask = dist_to_swapped > int(gnn_depth)
            for model_name in FUNCTIONAL_MODELS:
                delta_nodes = source_outputs[model_name][row_idx].float() - base_outputs[model_name][graph_index].float()
                delta_graph = graph_sum_from_node_output(delta_nodes)
                projection, cosine = projection_and_cosine(delta_graph, dy)
                prefix = "df_grit" if model_name == "grit" else "df_gcn_plus"
                out[f"{model_name}_delta_norm"] = float(torch.linalg.vector_norm(delta_graph).item())
                out[f"{model_name}_teacher_projection"] = projection
                out[f"{model_name}_teacher_cosine"] = cosine
                for dim_idx, value in enumerate(delta_graph.tolist()):
                    out[f"{prefix}_{dim_idx}"] = float(value)
                if model_name == "gcn_plus":
                    far_delta = delta_nodes[far_mask]
                    norms = torch.linalg.vector_norm(far_delta, dim=1) if far_delta.numel() else torch.empty(0)
                    out["gcn_far_node_count"] = int(far_mask.sum().item())
                    out["gcn_far_node_max_norm"] = float(norms.max().item()) if norms.numel() else 0.0
                    out["gcn_far_node_median_norm"] = float(norms.median().item()) if norms.numel() else 0.0
                    out["gcn_far_node_mean_norm"] = float(norms.mean().item()) if norms.numel() else 0.0
            rows.append(out)
        chunk_idx = start // int(batch_size) + 1
        elapsed = time.time() - start_time
        print(
            f"[functional-responses] task={task} chunk={chunk_idx}/{chunks} rows={len(rows)} elapsed={elapsed:.1f}s",
            flush=True,
        )
    out_path = functional_response_table_path(cfg, task, seed)
    write_csv(out_path, rows)
    print(f"[functional-responses] wrote responses={out_path} clean={clean_path}", flush=True)
    return out_path, clean_path


def run_functional_q1_gates(
    cfg: Mapping[str, Any],
    task: str,
    *,
    swaps_path: Path | None = None,
    response_path: Path | None = None,
    seed: int = 9501,
    e1g_seed: int = 9101,
    device_name: str = "auto",
    batch_size: int = 512,
    gnn_depth: int = 2,
    grit_config: Path | None = None,
    grit_checkpoint: Path | None = None,
    store_head_gates: bool = False,
    fast_dev_run: bool = False,
) -> Path:
    device = choose_device(device_name)
    configure_runtime(cfg, device)
    swaps_path = swaps_path or functional_swap_cache_path(cfg, task, e1g_seed)
    response_path = response_path or functional_response_table_path(cfg, task, e1g_seed)
    cache = load_functional_swap_cache(swaps_path)
    base_records: list[dict[str, Any]] = list(cache["base_records"])
    swaps: list[dict[str, Any]] = list(cache["swaps"])
    if fast_dev_run:
        swaps = swaps[: min(len(swaps), 48)]
    response_rows = read_csv_dicts(response_path)
    if not response_rows:
        raise FileNotFoundError(f"functional response table missing or empty: {response_path}")
    response_by_key = {(str(row["graph_id"]), str(row["swap_id"])): row for row in response_rows}
    grit_cfg = load_stage1_model_config(cfg, task, model_name="grit", config_path=grit_config, fast_dev_run=fast_dev_run)
    model, _ = load_model_from_checkpoint(grit_cfg, task, grit_checkpoint, device, backend="official")
    max_nodes = max(int(row["n"]) for row in base_records)

    print(
        f"[functional-q1-gates] task={task} swaps={len(swaps)} graphs={len(base_records)} "
        f"device={device} batch_size={batch_size} store_head_gates={store_head_gates}",
        flush=True,
    )
    base_layer_cache: dict[int, list[Q1LayerFields]] = {}
    for start in range(0, len(base_records), int(batch_size)):
        chunk = base_records[start : start + int(batch_size)]
        batch = collate_records(chunk).to(device)
        _pred, layers = collect_official_fields_and_prediction(model, batch, max_nodes=max_nodes)
        for local_idx, _record in enumerate(chunk):
            base_layer_cache[start + local_idx] = q1_layers_for_graph(layers, local_idx)

    rows: list[dict[str, Any]] = []
    chunks = max(1, math.ceil(len(swaps) / int(batch_size)))
    start_time = time.time()
    for start in range(0, len(swaps), int(batch_size)):
        chunk_swaps = swaps[start : start + int(batch_size)]
        source_records = [
            source_for_intervention(
                base_records[int(row["graph_index"])],
                str(row["family"]),
                int(row["u"]),
                int(row["v"]),
                cfg,
            )
            for row in chunk_swaps
        ]
        source_batch = collate_records(source_records).to(device)
        _pred, source_layers = collect_official_fields_and_prediction(model, source_batch, max_nodes=max_nodes)
        for local_idx, swap in enumerate(chunk_swaps):
            graph_index = int(swap["graph_index"])
            key = (str(swap["graph_id"]), str(swap["swap_id"]))
            response = response_by_key.get(key)
            if response is None:
                raise KeyError(f"missing response row for graph_id={key[0]} swap_id={key[1]}")
            source_q1_layers = q1_layers_for_graph(source_layers, local_idx)
            rows.extend(
                q1_scope_rows_for_swap(
                    task=task,
                    response_row=response,
                    swap=swap,
                    base=base_records[graph_index],
                    clean_layers=base_layer_cache[graph_index],
                    source_layers=source_q1_layers,
                    gnn_depth=gnn_depth,
                    store_head_gates=store_head_gates,
                )
            )
        chunk_idx = start // int(batch_size) + 1
        elapsed = time.time() - start_time
        print(
            f"[functional-q1-gates] task={task} chunk={chunk_idx}/{chunks} "
            f"swaps_done={min(start + int(batch_size), len(swaps))} rows={len(rows)} elapsed={elapsed:.1f}s",
            flush=True,
        )
    out = functional_q1_gate_features_path(cfg, task, seed)
    write_csv(out, rows)
    print(f"[functional-q1-gates] wrote {out}", flush=True)
    return out


def numeric_column(rows: Sequence[Mapping[str, Any]], column: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row[column]))
        except Exception:
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def projection_advantage_mean(rows: Sequence[Mapping[str, Any]]) -> float:
    return float(
        np.nanmean(
            numeric_column(rows, "grit_teacher_projection")
            - numeric_column(rows, "gcn_plus_teacher_projection")
        )
    )


def q1_stat_functions() -> dict[str, Any]:
    base = functional_stat_functions()
    names = [
        "dy_norm_mean",
        "grit_delta_norm_mean",
        "gcn_plus_delta_norm_mean",
        "grit_excess_partial_corr",
        "gcn_plus_converse_partial_corr",
    ]
    fns = {name: base[name] for name in names}
    fns["projection_advantage_mean"] = projection_advantage_mean
    return fns


def assign_task_quantile_bins(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_column: str,
    bin_column: str,
    label_column: str,
    bin_labels: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(dict(row))
    out: list[dict[str, Any]] = []
    for _task, task_rows in sorted(by_task.items()):
        finite_pairs = []
        for idx, row in enumerate(task_rows):
            value = finite_float(row.get(value_column))
            if math.isfinite(value):
                finite_pairs.append((idx, value))
        if not finite_pairs:
            for row in task_rows:
                row[bin_column] = "missing"
                row[label_column] = "missing"
            out.extend(task_rows)
            continue
        values = np.asarray([value for _idx, value in finite_pairs], dtype=np.float64)
        if len(np.unique(values)) < 2:
            for idx, _value in finite_pairs:
                task_rows[idx][bin_column] = "q1"
                task_rows[idx][label_column] = bin_labels.get("q1", "q1")
        else:
            edges = np.quantile(values, [0.25, 0.50, 0.75])
            for idx, value in finite_pairs:
                q_idx = int(np.searchsorted(edges, value, side="right"))
                q_idx = min(max(q_idx, 0), len(Q1_GATE_BINS) - 1)
                bin_name = Q1_GATE_BINS[q_idx]
                task_rows[idx][bin_column] = bin_name
                task_rows[idx][label_column] = bin_labels.get(bin_name, bin_name)
        for row in task_rows:
            row.setdefault(bin_column, "missing")
            row.setdefault(label_column, "missing")
        out.extend(task_rows)
    return out


def q1_scope_all_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("layer")) == "all" and str(row.get("head")) == "all"
    ]


def q1_graph_excess(row_group: Sequence[Mapping[str, Any]]) -> float:
    if len(row_group) < 3:
        return float("nan")
    return partial_corr(
        numeric_column(row_group, "grit_teacher_projection"),
        numeric_column(row_group, "dy_norm"),
        numeric_column(row_group, "gcn_plus_teacher_projection"),
    )


def summarise_functional_q1_gates(
    cfg: Mapping[str, Any],
    *,
    tasks: Sequence[str] | None = None,
    seed: int = 9501,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 9601,
    min_swaps_per_bin: int = 100,
    min_swaps_per_graph: int = 20,
) -> tuple[Path, Path, Path]:
    tasks = list(tasks or TASKS)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        path = functional_q1_gate_features_path(cfg, task, seed)
        if not path.exists():
            raise FileNotFoundError(f"Q1 gate feature table not found: {path}")
        task_rows = read_csv_dicts(path)
        if not task_rows:
            raise RuntimeError(f"Q1 gate feature table is empty: {path}")
        rows.extend(task_rows)
    scope_rows = q1_scope_all_rows(rows)
    stat_fns = q1_stat_functions()
    quartile_rows: list[dict[str, Any]] = []

    def add_quartile_stats(
        base_rows: Sequence[Mapping[str, Any]],
        *,
        gate_type: str,
        gate_metric: str,
        conditioning: str,
        seed_offset: int,
    ) -> None:
        binned = assign_task_quantile_bins(
            base_rows,
            value_column=gate_metric,
            bin_column="q1_gate_bin",
            label_column="q1_gate_bin_label",
            bin_labels=Q1_GATE_BIN_LABELS,
        )
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in binned:
            if str(row.get("q1_gate_bin")) in Q1_GATE_BINS:
                grouped[(str(row["task"]), str(row["q1_gate_bin"]))].append(row)
        for (task, gate_bin), group in sorted(grouped.items()):
            graphs = len({str(row["graph_id"]) for row in group})
            gate_values = numeric_column(group, gate_metric)
            status = "ok" if len(group) >= int(min_swaps_per_bin) and graphs >= 2 else "underpowered"
            for stat_name, stat_fn in stat_fns.items():
                point, lo, hi = cluster_bootstrap_ci(
                    group,
                    stat_fn,
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed + seed_offset + stable_int_seed(task, gate_type, conditioning, gate_bin, stat_name) % 100000,
                )
                quartile_rows.append(
                    {
                        "task": task,
                        "gate_type": gate_type,
                        "gate_metric": gate_metric,
                        "conditioning": conditioning,
                        "gate_bin": gate_bin,
                        "gate_bin_label": Q1_GATE_BIN_LABELS.get(gate_bin, gate_bin),
                        "graphs": graphs,
                        "swaps": len(group),
                        "gate_low": float(np.nanmin(gate_values)) if gate_values.size else float("nan"),
                        "gate_high": float(np.nanmax(gate_values)) if gate_values.size else float("nan"),
                        "gate_mean": float(np.nanmean(gate_values)) if gate_values.size else float("nan"),
                        "stat": stat_name,
                        "mean": point,
                        "ci_low": lo,
                        "ci_high": hi,
                        "status": status,
                    }
                )

    for gate_type, gate_metric in Q1_PRIMARY_GATES.items():
        add_quartile_stats(scope_rows, gate_type=gate_type, gate_metric=gate_metric, conditioning="all", seed_offset=0)
    rho_rows = assign_rho_far_quartiles(scope_rows)
    for rho_bin in RHO_FAR_BINS:
        subset = [row for row in rho_rows if str(row.get("rho_far_bin")) == rho_bin]
        if not subset:
            continue
        for gate_type, gate_metric in Q1_PRIMARY_GATES.items():
            add_quartile_stats(
                subset,
                gate_type=gate_type,
                gate_metric=gate_metric,
                conditioning=f"rho_far:{rho_bin}",
                seed_offset=10000 + stable_int_seed(rho_bin) % 100000,
            )

    graph_points: list[dict[str, Any]] = []
    by_task_graph: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in scope_rows:
        by_task_graph[(str(row["task"]), str(row["graph_id"]))].append(row)
    for (task, graph_id), group in sorted(by_task_graph.items()):
        swaps = len(group)
        graph_excess = q1_graph_excess(group) if swaps >= int(min_swaps_per_graph) else float("nan")
        for gate_type, gate_metric in Q1_PRIMARY_GATES.items():
            values = numeric_column(group, gate_metric)
            gate_mean = float(np.nanmean(values)) if values.size else float("nan")
            status = (
                "ok"
                if swaps >= int(min_swaps_per_graph) and math.isfinite(graph_excess) and math.isfinite(gate_mean)
                else "underpowered"
            )
            graph_points.append(
                {
                    "task": task,
                    "graph_id": graph_id,
                    "gate_type": gate_type,
                    "gate_metric": gate_metric,
                    "graph_gate_mean": gate_mean,
                    "graph_gate_std": float(np.nanstd(values)) if values.size else float("nan"),
                    "graph_excess_partial_corr": graph_excess,
                    "swaps": swaps,
                    "status": status,
                }
            )

    graph_stats: list[dict[str, Any]] = []
    for task in tasks:
        for gate_type, gate_metric in Q1_PRIMARY_GATES.items():
            group = [
                row
                for row in graph_points
                if row["task"] == task and row["gate_type"] == gate_type and row["status"] == "ok"
            ]
            for stat_name, stat_fn in [
                (
                    "pearson",
                    lambda sample: pearson_corr(
                        numeric_column(sample, "graph_gate_mean"),
                        numeric_column(sample, "graph_excess_partial_corr"),
                    ),
                ),
                (
                    "spearman",
                    lambda sample: spearman_corr(
                        numeric_column(sample, "graph_gate_mean"),
                        numeric_column(sample, "graph_excess_partial_corr"),
                    ),
                ),
            ]:
                point, lo, hi = cluster_bootstrap_ci(
                    group,
                    stat_fn,
                    resamples=bootstrap_resamples,
                    seed=bootstrap_seed + stable_int_seed(task, gate_type, stat_name) % 100000,
                )
                graph_stats.append(
                    {
                        "task": task,
                        "gate_type": gate_type,
                        "gate_metric": gate_metric,
                        "stat": stat_name,
                        "graphs": len(group),
                        "mean": point,
                        "ci_low": lo,
                        "ci_high": hi,
                        "status": "ok" if len(group) >= 3 else "underpowered",
                    }
                )

    contrast_rows: list[dict[str, Any]] = []
    for task in tasks:
        task_points = [row for row in graph_points if row["task"] == task and row["status"] == "ok"]

        def contrast_stat(sample: Sequence[Mapping[str, Any]], *, corr: str) -> float:
            route = [row for row in sample if row["gate_type"] == "routing"]
            transport = [row for row in sample if row["gate_type"] == "transport"]
            fn = spearman_corr if corr == "spearman" else pearson_corr
            r_route = fn(numeric_column(route, "graph_gate_mean"), numeric_column(route, "graph_excess_partial_corr"))
            r_trans = fn(numeric_column(transport, "graph_gate_mean"), numeric_column(transport, "graph_excess_partial_corr"))
            return float(r_route - r_trans)

        for corr_name in ("pearson", "spearman"):
            point, lo, hi = cluster_bootstrap_ci(
                task_points,
                lambda sample, name=corr_name: contrast_stat(sample, corr=name),
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + stable_int_seed(task, "route_minus_transport", corr_name) % 100000,
            )
            contrast_rows.append(
                {
                    "task": task,
                    "contrast": "routing_minus_transport",
                    "stat": corr_name,
                    "graphs": len({str(row["graph_id"]) for row in task_points}),
                    "mean": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "status": "ok" if len({str(row["graph_id"]) for row in task_points}) >= 3 else "underpowered",
                }
            )

    quartile_path = functional_q1_gate_quartile_stats_path(cfg)
    points_path = functional_q1_graph_coupling_points_path(cfg)
    stats_path = functional_q1_graph_coupling_stats_path(cfg)
    contrast_path = functional_q1_gate_contrast_stats_path(cfg)
    write_csv(quartile_path, quartile_rows)
    write_csv(points_path, graph_points)
    write_csv(stats_path, graph_stats)
    write_csv(contrast_path, contrast_rows)
    print(
        f"[functional-q1-summary] wrote quartiles={quartile_path} points={points_path} "
        f"stats={stats_path} contrasts={contrast_path}",
        flush=True,
    )
    return quartile_path, points_path, stats_path


def enrich_functional_rows_with_rho_far(
    cfg: Mapping[str, Any],
    task: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 9101,
    gnn_depth: int = 2,
) -> list[dict[str, Any]]:
    all_present = True
    for row in rows:
        try:
            all_present = all_present and np.isfinite(float(row.get("rho_far", "nan")))
        except Exception:
            all_present = False
            break
    if all_present:
        return [dict(row) for row in rows]
    cache = load_functional_swap_cache(functional_swap_cache_path(cfg, task, seed))
    base_records: list[dict[str, Any]] = list(cache["base_records"])
    swaps_by_id = {str(row["swap_id"]): row for row in cache["swaps"]}
    out = []
    for row in rows:
        enriched = dict(row)
        try:
            rho_far = float(enriched.get("rho_far", "nan"))
        except Exception:
            rho_far = float("nan")
        if not np.isfinite(rho_far):
            swap = swaps_by_id.get(str(enriched["swap_id"]))
            if swap is None:
                graph_index = int(enriched["graph_index"])
                family = str(enriched["family"])
                u = int(enriched["u"])
                v = int(enriched["v"])
            else:
                graph_index = int(swap["graph_index"])
                family = str(swap["family"])
                u = int(swap["u"])
                v = int(swap["v"])
            base = base_records[graph_index]
            source = source_for_intervention(base, family, u, v, cfg)
            rho_far = teacher_far_response_fraction(base, source, u, v, gnn_depth=gnn_depth)
        enriched["rho_far"] = rho_far
        out.append(enriched)
    return out


def assign_rho_far_quartiles(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(dict(row))
    out = []
    for task_rows in by_task.values():
        finite = []
        for idx, row in enumerate(task_rows):
            try:
                value = float(row["rho_far"])
            except Exception:
                value = float("nan")
            if np.isfinite(value):
                finite.append((value, idx))
        finite.sort(key=lambda item: (item[0], item[1]))
        n = len(finite)
        for rank, (_value, idx) in enumerate(finite):
            q_idx = min(3, int(rank * 4 / max(1, n)))
            task_rows[idx]["rho_far_bin"] = RHO_FAR_BINS[q_idx]
            task_rows[idx]["rho_far_bin_label"] = RHO_FAR_BIN_LABELS[RHO_FAR_BINS[q_idx]]
        for row in task_rows:
            row.setdefault("rho_far_bin", "missing")
            row.setdefault("rho_far_bin_label", "missing")
            out.append(row)
    return out


def rho_far_histogram_rows(rows: Sequence[Mapping[str, Any]], *, bins: int = 10) -> list[dict[str, Any]]:
    out = []
    for task in sorted({str(row["task"]) for row in rows}):
        vals = np.asarray([float(row["rho_far"]) for row in rows if str(row["task"]) == task], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        counts, edges = np.histogram(vals, bins=int(bins), range=(0.0, 1.0))
        for idx, count in enumerate(counts):
            out.append(
                {
                    "task": task,
                    "bin": idx,
                    "rho_far_low": float(edges[idx]),
                    "rho_far_high": float(edges[idx + 1]),
                    "count": int(count),
                    "fraction": float(count / max(1, vals.size)),
                    "rho_far_mean": float(vals.mean()),
                    "rho_far_median": float(np.median(vals)),
                    "rho_far_unique_rounded_3dp": int(len(set(np.round(vals, 3).tolist()))),
                }
            )
    return out


def row_graph_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row["graph_id"]) for row in rows]


def cluster_bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    stat_fn: Any,
    *,
    resamples: int = 1000,
    seed: int = 9201,
    ci: float = 95.0,
) -> tuple[float, float, float]:
    if not rows:
        return float("nan"), float("nan"), float("nan")
    point = float(stat_fn(list(rows)))
    if int(resamples) <= 0:
        return point, float("nan"), float("nan")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["graph_id"])].append(row)
    graph_ids = sorted(groups)
    if len(graph_ids) < 2:
        return point, float("nan"), float("nan")
    rng = random.Random(int(seed))
    values = []
    for _ in range(int(resamples)):
        sample_rows: list[Mapping[str, Any]] = []
        sampled = [rng.choice(graph_ids) for _ in graph_ids]
        for graph_id in sampled:
            sample_rows.extend(groups[graph_id])
        values.append(float(stat_fn(sample_rows)))
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return point, float("nan"), float("nan")
    alpha = (100.0 - float(ci)) / 2.0
    return point, float(np.percentile(arr, alpha)), float(np.percentile(arr, 100.0 - alpha))


def stable_int_seed(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def functional_stat_functions() -> dict[str, Any]:
    return {
        "dy_norm_mean": lambda rows: float(np.nanmean(numeric_column(rows, "dy_norm"))),
        "grit_delta_norm_mean": lambda rows: float(np.nanmean(numeric_column(rows, "grit_delta_norm"))),
        "gcn_plus_delta_norm_mean": lambda rows: float(np.nanmean(numeric_column(rows, "gcn_plus_delta_norm"))),
        "grit_alignment_pearson": lambda rows: pearson_corr(
            numeric_column(rows, "grit_teacher_projection"), numeric_column(rows, "dy_norm")
        ),
        "gcn_plus_alignment_pearson": lambda rows: pearson_corr(
            numeric_column(rows, "gcn_plus_teacher_projection"), numeric_column(rows, "dy_norm")
        ),
        "grit_alignment_spearman": lambda rows: spearman_corr(
            numeric_column(rows, "grit_teacher_projection"), numeric_column(rows, "dy_norm")
        ),
        "gcn_plus_alignment_spearman": lambda rows: spearman_corr(
            numeric_column(rows, "gcn_plus_teacher_projection"), numeric_column(rows, "dy_norm")
        ),
        "grit_excess_partial_corr": lambda rows: partial_corr(
            numeric_column(rows, "grit_teacher_projection"),
            numeric_column(rows, "dy_norm"),
            numeric_column(rows, "gcn_plus_teacher_projection"),
        ),
        "gcn_plus_converse_partial_corr": lambda rows: partial_corr(
            numeric_column(rows, "gcn_plus_teacher_projection"),
            numeric_column(rows, "dy_norm"),
            numeric_column(rows, "grit_teacher_projection"),
        ),
        "gcn_far_node_max_norm_mean": lambda rows: float(np.nanmean(numeric_column(rows, "gcn_far_node_max_norm"))),
        "gcn_far_node_median_norm_mean": lambda rows: float(np.nanmean(numeric_column(rows, "gcn_far_node_median_norm"))),
    }


def summarise_functional_responses(
    cfg: Mapping[str, Any],
    *,
    tasks: Sequence[str] | None = None,
    seed: int = 9101,
    gnn_depth: int = 2,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 9201,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    tasks = list(tasks or TASKS)
    all_rows: list[dict[str, str]] = []
    for task in tasks:
        path = functional_response_table_path(cfg, task, seed)
        if not path.exists():
            raise FileNotFoundError(f"functional response table not found: {path}")
        rows = read_csv_dicts(path)
        if not rows:
            raise RuntimeError(f"functional response table is empty: {path}")
        all_rows.extend(enrich_functional_rows_with_rho_far(cfg, task, rows, seed=seed, gnn_depth=gnn_depth))
    stat_fns = functional_stat_functions()
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[(str(row["task"]), str(row["stratum"]))].append(row)
    rho_rows = assign_rho_far_quartiles(all_rows)
    rho_grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rho_rows:
        rho_grouped[(str(row["task"]), str(row["rho_far_bin"]))].append(row)
    e0_rows = []
    e1_rows = []
    e1_rho_rows = []
    validity_rows = []
    for (task, stratum), rows in sorted(grouped.items()):
        graphs = len(set(row_graph_ids(rows)))
        e0_point, e0_lo, e0_hi = cluster_bootstrap_ci(
            rows,
            stat_fns["dy_norm_mean"],
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        e0_rows.append(
            {
                "task": task,
                "stratum": stratum,
                "stratum_label": FUNCTIONAL_STRATUM_LABELS.get(stratum, stratum),
                "graphs": graphs,
                "swaps": len(rows),
                "dy_norm_mean": e0_point,
                "dy_norm_ci_low": e0_lo,
                "dy_norm_ci_high": e0_hi,
            }
        )
        for stat_name, stat_fn in stat_fns.items():
            point, lo, hi = cluster_bootstrap_ci(
                rows,
                stat_fn,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + stable_int_seed(task, stratum, stat_name) % 100000,
            )
            target = validity_rows if stat_name.startswith("gcn_far") else e1_rows
            target.append(
                {
                    "task": task,
                    "stratum": stratum,
                    "stratum_label": FUNCTIONAL_STRATUM_LABELS.get(stratum, stratum),
                    "graphs": graphs,
                    "swaps": len(rows),
                    "stat": stat_name,
                    "mean": point,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    for (task, rho_bin), rows in sorted(rho_grouped.items()):
        graphs = len(set(row_graph_ids(rows)))
        rho_values = numeric_column(rows, "rho_far")
        for stat_name, stat_fn in stat_fns.items():
            if stat_name.startswith("gcn_far"):
                continue
            point, lo, hi = cluster_bootstrap_ci(
                rows,
                stat_fn,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + stable_int_seed(task, rho_bin, stat_name) % 100000,
            )
            e1_rho_rows.append(
                {
                    "task": task,
                    "rho_far_bin": rho_bin,
                    "rho_far_bin_label": RHO_FAR_BIN_LABELS.get(rho_bin, rho_bin),
                    "rho_far_low": float(np.nanmin(rho_values)) if rho_values.size else float("nan"),
                    "rho_far_high": float(np.nanmax(rho_values)) if rho_values.size else float("nan"),
                    "rho_far_mean": float(np.nanmean(rho_values)) if rho_values.size else float("nan"),
                    "graphs": graphs,
                    "swaps": len(rows),
                    "stat": stat_name,
                    "mean": point,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    clean_rows = []
    for task in tasks:
        clean_path = functional_responses_dir(cfg, task) / f"functional_clean_context_seed{int(seed)}.csv"
        if not clean_path.exists():
            raise FileNotFoundError(f"functional clean-context table not found: {clean_path}")
        clean_rows.extend(read_csv_dicts(clean_path))
    metrics = functional_metrics_dir(cfg)
    e0_path = metrics / "functional_e0_task_anatomy.csv"
    e1_path = metrics / "functional_e1_fingerprint_stats.csv"
    e1_rho_path = metrics / "functional_e1_fingerprint_rhofar_stats.csv"
    rho_hist_path = metrics / "functional_e0_rho_far_histogram.csv"
    clean_out = metrics / "functional_e1_clean_performance_context.csv"
    validity_path = metrics / "functional_e1_gcn_receptive_field_validity.csv"
    write_csv(e0_path, e0_rows)
    write_csv(e1_path, e1_rows)
    write_csv(e1_rho_path, e1_rho_rows)
    write_csv(rho_hist_path, rho_far_histogram_rows(all_rows))
    write_csv(clean_out, clean_rows)
    write_csv(validity_path, validity_rows)
    print(
        f"[functional-summary] wrote e0={e0_path} e1={e1_path} e1_rho={e1_rho_path} "
        f"rho_hist={rho_hist_path} clean={clean_out} validity={validity_path}",
        flush=True,
    )
    return e0_path, e1_path, e1_rho_path, rho_hist_path, clean_out, validity_path


def rows_by_task_stratum(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Mapping[str, Any]]]:
    out: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        out[(str(row["task"]), str(row["stratum"]))][str(row["stat"])] = row
    return out


def get_metric_value(rows: Mapping[str, Mapping[str, Any]], stat: str, field: str = "mean") -> float:
    try:
        return float(rows[stat][field])
    except Exception:
        return float("nan")


def nonnegative_ci_yerr(values: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    lower = np.where(np.isfinite(values - lows), np.maximum(0.0, values - lows), 0.0)
    upper = np.where(np.isfinite(highs - values), np.maximum(0.0, highs - values), 0.0)
    return np.vstack([lower, upper])


def plot_functional_responses(cfg: Mapping[str, Any], *, tasks: Sequence[str] | None = None, seed: int = 9101) -> None:
    tasks = list(tasks or TASKS)
    fig_dir = functional_figures_dir(cfg)
    fig_dir.mkdir(parents=True, exist_ok=True)
    e0_rows = read_csv_dicts(functional_metrics_dir(cfg) / "functional_e0_task_anatomy.csv")
    e1_rows = read_csv_dicts(functional_metrics_dir(cfg) / "functional_e1_fingerprint_stats.csv")
    e1_rho_rows = read_csv_dicts(functional_metrics_dir(cfg) / "functional_e1_fingerprint_rhofar_stats.csv")
    rho_hist_rows = read_csv_dicts(functional_metrics_dir(cfg) / "functional_e0_rho_far_histogram.csv")
    clean_rows = read_csv_dicts(functional_metrics_dir(cfg) / "functional_e1_clean_performance_context.csv")
    validity_rows = read_csv_dicts(functional_metrics_dir(cfg) / "functional_e1_gcn_receptive_field_validity.csv")
    if not e0_rows or not e1_rows or not clean_rows:
        raise FileNotFoundError("functional summary CSVs are missing or empty; run summarise-functional-responses first")
    strata = list(FUNCTIONAL_STRATA)
    x = np.arange(len(strata), dtype=np.float64)

    fig, axes = plt.subplots(2, max(1, len(tasks)), figsize=(5.2 * max(1, len(tasks)), 6.8), sharex=False)
    axes_arr = np.asarray(axes).reshape(2, -1)
    for col, task in enumerate(tasks):
        ax = axes_arr[0, col]
        rows = {str(row["stratum"]): row for row in e0_rows if row.get("task") == task}
        means = np.asarray([float(rows[s]["dy_norm_mean"]) if s in rows else np.nan for s in strata])
        lows = np.asarray([float(rows[s]["dy_norm_ci_low"]) if s in rows else np.nan for s in strata])
        highs = np.asarray([float(rows[s]["dy_norm_ci_high"]) if s in rows else np.nan for s in strata])
        ax.bar(x, means, color="#3f6f8f", width=0.68)
        ax.errorbar(x, means, yerr=nonnegative_ci_yerr(means, lows, highs), fmt="none", color="black", lw=1.0)
        ax.set_title(task)
        ax.set_xticks(x)
        ax.set_xticklabels([FUNCTIONAL_STRATUM_LABELS[s] for s in strata], rotation=25, ha="right")
        ax.set_ylabel("Oracle ||Δy||")
        ax.grid(axis="y", alpha=0.25)
        ax_hist = axes_arr[1, col]
        hist = [row for row in rho_hist_rows if row.get("task") == task]
        if hist:
            lows_h = np.asarray([float(row["rho_far_low"]) for row in hist])
            highs_h = np.asarray([float(row["rho_far_high"]) for row in hist])
            mids = 0.5 * (lows_h + highs_h)
            widths = highs_h - lows_h
            fractions = np.asarray([float(row["fraction"]) for row in hist])
            ax_hist.bar(mids, fractions, width=widths * 0.92, color="#6b6f7a", align="center")
            mean_val = float(hist[0].get("rho_far_mean", "nan"))
            median_val = float(hist[0].get("rho_far_median", "nan"))
            ax_hist.axvline(mean_val, color="#1b7f79", lw=1.2, label="mean")
            ax_hist.axvline(median_val, color="#b24c3d", lw=1.2, linestyle="--", label="median")
        ax_hist.set_xlim(0.0, 1.0)
        ax_hist.set_xlabel("rho_far: teacher response mass beyond L hops")
        ax_hist.set_ylabel("Swap fraction")
        ax_hist.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_e0_task_anatomy_content_swaps.pdf")
    plt.close(fig)

    indexed = rows_by_task_stratum(e1_rows)
    fig, axes = plt.subplots(2, max(1, len(tasks)), figsize=(5.4 * max(1, len(tasks)), 7.2), sharex=True)
    axes_arr = np.asarray(axes).reshape(2, -1)
    colors = {"grit": "#1b7f79", "gcn_plus": "#b24c3d"}
    for col, task in enumerate(tasks):
        ax_mag = axes_arr[0, col]
        ax_align = axes_arr[1, col]
        for offset, model_name in [(-0.18, "grit"), (0.18, "gcn_plus")]:
            mag = np.asarray([get_metric_value(indexed.get((task, s), {}), f"{model_name}_delta_norm_mean") for s in strata])
            align = np.asarray([get_metric_value(indexed.get((task, s), {}), f"{model_name}_alignment_pearson") for s in strata])
            ax_mag.bar(x + offset, mag, width=0.34, label=model_name, color=colors[model_name])
            ax_align.bar(x + offset, align, width=0.34, label=model_name, color=colors[model_name])
        ax_mag.set_title(task)
        ax_mag.set_ylabel("Mean ||Δf||")
        ax_align.set_ylabel("corr(proj(Δf, Δy), ||Δy||)")
        ax_align.axhline(0.0, color="black", lw=0.8)
        for ax in (ax_mag, ax_align):
            ax.set_xticks(x)
            ax.set_xticklabels([FUNCTIONAL_STRATUM_LABELS[s] for s in strata], rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
    axes_arr[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_e1_response_magnitude_alignment.pdf")
    plt.close(fig)

    rho_indexed: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in e1_rho_rows:
        rho_indexed[(str(row["task"]), str(row["rho_far_bin"]))][str(row["stat"])] = row
    fig, axes = plt.subplots(2, max(1, len(tasks)), figsize=(5.6 * max(1, len(tasks)), 7.4), sharey=True)
    axes_arr = np.asarray(axes).reshape(2, -1)
    for col, task in enumerate(tasks):
        ax = axes_arr[0, col]
        for offset, stat_name, label, color in [
            (-0.18, "grit_excess_partial_corr", "GRIT | GCN+", "#1b7f79"),
            (0.18, "gcn_plus_converse_partial_corr", "GCN+ | GRIT", "#b24c3d"),
        ]:
            vals = np.asarray([get_metric_value(indexed.get((task, s), {}), stat_name) for s in strata])
            lows = np.asarray([get_metric_value(indexed.get((task, s), {}), stat_name, "ci_low") for s in strata])
            highs = np.asarray([get_metric_value(indexed.get((task, s), {}), stat_name, "ci_high") for s in strata])
            ax.bar(x + offset, vals, width=0.34, label=label, color=color)
            ax.errorbar(x + offset, vals, yerr=nonnegative_ci_yerr(vals, lows, highs), fmt="none", color="black", lw=0.8)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(task)
        ax.set_ylabel("Partial correlation")
        ax.set_xticks(x)
        ax.set_xticklabels([FUNCTIONAL_STRATUM_LABELS[s] for s in strata], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("site-distance stratum d(u,v)")
        ax_rho = axes_arr[1, col]
        rho_x = np.arange(len(RHO_FAR_BINS), dtype=np.float64)
        for offset, stat_name, label, color in [
            (-0.18, "grit_excess_partial_corr", "GRIT | GCN+", "#1b7f79"),
            (0.18, "gcn_plus_converse_partial_corr", "GCN+ | GRIT", "#b24c3d"),
        ]:
            vals = np.asarray([get_metric_value(rho_indexed.get((task, s), {}), stat_name) for s in RHO_FAR_BINS])
            lows = np.asarray([get_metric_value(rho_indexed.get((task, s), {}), stat_name, "ci_low") for s in RHO_FAR_BINS])
            highs = np.asarray([get_metric_value(rho_indexed.get((task, s), {}), stat_name, "ci_high") for s in RHO_FAR_BINS])
            ax_rho.bar(rho_x + offset, vals, width=0.34, label=label, color=color)
            ax_rho.errorbar(rho_x + offset, vals, yerr=nonnegative_ci_yerr(vals, lows, highs), fmt="none", color="black", lw=0.8)
        ax_rho.axhline(0.0, color="black", lw=0.8)
        ax_rho.set_title(f"{task}: binned by rho_far")
        ax_rho.set_ylabel("Partial correlation")
        ax_rho.set_xticks(rho_x)
        ax_rho.set_xticklabels([RHO_FAR_BIN_LABELS[s] for s in RHO_FAR_BINS], rotation=25, ha="right")
        ax_rho.set_xlabel("consequence-range quartile")
        ax_rho.grid(axis="y", alpha=0.25)
    axes_arr[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_e1_excess_alignment.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, metric, ylabel in [
        (axes[0], "node_relmse_mean", "Node relMSE"),
        (axes[1], "graph_sum_relmse", "Graph-sum relMSE"),
    ]:
        labels = []
        vals = []
        bar_colors = []
        for task in tasks:
            for model_name in FUNCTIONAL_MODELS:
                row = next((row for row in clean_rows if row.get("task") == task and row.get("model") == model_name), None)
                labels.append(f"{task}\n{model_name}")
                vals.append(float(row[metric]) if row is not None else float("nan"))
                bar_colors.append(colors[model_name])
        ax.bar(np.arange(len(vals)), vals, color=bar_colors)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_e1_clean_performance_context.pdf")
    plt.close(fig)

    response_rows = []
    for task in tasks:
        response_rows.extend(read_csv_dicts(functional_response_table_path(cfg, task, seed)))
    fig, axes = plt.subplots(1, max(1, len(tasks)), figsize=(5.2 * max(1, len(tasks)), 3.8), sharey=True)
    axes_list = np.atleast_1d(axes)
    for ax, task in zip(axes_list, tasks):
        rows = [row for row in response_rows if row.get("task") == task]
        ax.scatter(
            numeric_column(rows, "d_uv"),
            numeric_column(rows, "R_eff_uv"),
            s=8,
            alpha=0.25,
            color="#4d6880",
            linewidths=0,
        )
        ax.set_title(task)
        ax.set_xlabel("Shortest-path distance")
        ax.set_ylabel("Effective resistance")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_swap_distance_resistance_diagnostic.pdf")
    plt.close(fig)

    validity_indexed = rows_by_task_stratum(validity_rows)
    fig, axes = plt.subplots(1, max(1, len(tasks)), figsize=(5.2 * max(1, len(tasks)), 3.8), sharey=True)
    axes_list = np.atleast_1d(axes)
    for ax, task in zip(axes_list, tasks):
        max_vals = np.asarray([get_metric_value(validity_indexed.get((task, s), {}), "gcn_far_node_max_norm_mean") for s in strata])
        med_vals = np.asarray([get_metric_value(validity_indexed.get((task, s), {}), "gcn_far_node_median_norm_mean") for s in strata])
        ax.bar(x - 0.18, max_vals, width=0.34, label="max far-node Δ", color="#6b6f7a")
        ax.bar(x + 0.18, med_vals, width=0.34, label="median far-node Δ", color="#9aa0aa")
        ax.set_title(task)
        ax.set_ylabel("GCN+ change outside L-hop field")
        ax.set_xticks(x)
        ax.set_xticklabels([FUNCTIONAL_STRATUM_LABELS[s] for s in strata], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes_list[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_e1_gcn_receptive_field_validity.pdf")
    plt.close(fig)
    print(f"[functional-plot] wrote figures={fig_dir}", flush=True)


def node_distance_stratum(query_idx: int, u: int, v: int, distance_to_swap: int, gnn_depth: int = 2) -> str:
    if int(query_idx) in {int(u), int(v)}:
        return "self"
    d = int(distance_to_swap)
    depth = max(1, int(gnn_depth))
    if d == 1:
        return "d1"
    if 2 <= d <= depth:
        return "d2_to_L"
    return "d_gt_L"


def functional_node_swap_cache_path(cfg: Mapping[str, Any], task: str, seed: int) -> Path:
    return functional_node_swaps_dir(cfg, task) / f"node_content_swaps_seed{int(seed)}.pt"


def functional_node_response_table_path(cfg: Mapping[str, Any], task: str, seed: int) -> Path:
    return functional_node_responses_dir(cfg, task) / f"functional_e1n_node_response_table_seed{int(seed)}.csv"


def functional_node_graph_stats_path(cfg: Mapping[str, Any], task: str, seed: int) -> Path:
    return functional_node_responses_dir(cfg, task) / f"functional_e1n_graph_stratum_stats_seed{int(seed)}.csv"


def functional_node_validity_path(cfg: Mapping[str, Any], task: str, seed: int) -> Path:
    return functional_node_responses_dir(cfg, task) / f"functional_e1n_validity_seed{int(seed)}.csv"


def sample_functional_node_swaps_from_records(
    records: Sequence[Mapping[str, Any]],
    task: str,
    *,
    family: str | None = None,
    num_graphs: int = 200,
    swaps_per_graph: int = 200,
    seed: int = 9301,
) -> list[dict[str, Any]]:
    selected_records = list(records)[: int(num_graphs)]
    family = family or default_functional_family(task)
    rng = random.Random(int(seed))
    swaps: list[dict[str, Any]] = []
    for graph_idx, base in enumerate(selected_records):
        pairs = candidate_pairs_for_family(base, family)
        if len(pairs) > int(swaps_per_graph):
            pairs = rng.sample(pairs, k=int(swaps_per_graph))
        else:
            pairs = list(pairs)
            rng.shuffle(pairs)
        pairs.sort(key=lambda pair: (int(pair[0]), int(pair[1])))
        for swap_idx, (u, v) in enumerate(pairs):
            swaps.append(
                {
                    "task": task,
                    "family": family,
                    "graph_index": graph_idx,
                    "graph_id": str(base["graph_id"]),
                    "swap_id": f"{base['graph_id']}__{family}_{int(u)}_{int(v)}",
                    "swap_index": swap_idx,
                    "u": int(u),
                    "v": int(v),
                }
            )
    return swaps


def build_functional_node_swaps(
    cfg: Mapping[str, Any],
    task: str,
    *,
    family: str | None = None,
    num_graphs: int = 200,
    swaps_per_graph: int = 200,
    seed: int = 9301,
    force: bool = False,
) -> Path:
    cache_data(cfg, task, force=False)
    out_path = functional_node_swap_cache_path(cfg, task, seed)
    if out_path.exists() and not force:
        print(f"[functional-node-swaps] using existing {out_path}", flush=True)
        return out_path
    records = load_records(data_dir(cfg, task) / "test_id.pt")[: int(num_graphs)]
    family = family or default_functional_family(task)
    swaps = sample_functional_node_swaps_from_records(
        records,
        task,
        family=family,
        num_graphs=num_graphs,
        swaps_per_graph=swaps_per_graph,
        seed=seed,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": task,
            "family": family,
            "seed": int(seed),
            "num_graphs": int(num_graphs),
            "swaps_per_graph": int(swaps_per_graph),
            "base_records": records,
            "swaps": swaps,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
        },
        out_path,
    )
    print(
        f"[functional-node-swaps] wrote {out_path} graphs={len(records)} swaps={len(swaps)}",
        flush=True,
    )
    return out_path


def empty_node_stat_accumulator(task: str, graph_id: str, stratum: str) -> dict[str, Any]:
    return {
        "task": task,
        "graph_id": graph_id,
        "stratum": stratum,
        "observations": 0,
        "channel_n": 0,
        "dy_norm_sum": 0.0,
        "grit_norm_sum": 0.0,
        "gcn_plus_norm_sum": 0.0,
        "grit_cos_sum": 0.0,
        "grit_cos_count": 0,
        "gcn_plus_cos_sum": 0.0,
        "gcn_plus_cos_count": 0,
        "silent_count": 0,
        "grit_silent_norm_sum": 0.0,
        "gcn_plus_silent_norm_sum": 0.0,
        "gcn_plus_max_norm": 0.0,
        "sum_y": 0.0,
        "sum_y2": 0.0,
        "sum_grit": 0.0,
        "sum_grit2": 0.0,
        "sum_gcn_plus": 0.0,
        "sum_gcn_plus2": 0.0,
        "sum_grit_y": 0.0,
        "sum_gcn_plus_y": 0.0,
        "sum_grit_gcn_plus": 0.0,
    }


def update_node_stat_accumulator(
    acc: dict[str, Any],
    *,
    dy: torch.Tensor,
    grit_delta: torch.Tensor,
    gcn_delta: torch.Tensor,
    epsilon: float,
) -> None:
    dy = dy.float()
    grit_delta = grit_delta.float()
    gcn_delta = gcn_delta.float()
    dy_norm = float(torch.linalg.vector_norm(dy).item())
    grit_norm = float(torch.linalg.vector_norm(grit_delta).item())
    gcn_norm = float(torch.linalg.vector_norm(gcn_delta).item())
    acc["observations"] += 1
    acc["channel_n"] += int(dy.numel())
    acc["dy_norm_sum"] += dy_norm
    acc["grit_norm_sum"] += grit_norm
    acc["gcn_plus_norm_sum"] += gcn_norm
    acc["gcn_plus_max_norm"] = max(float(acc["gcn_plus_max_norm"]), gcn_norm)
    if dy_norm > float(epsilon):
        _, grit_cos = projection_and_cosine(grit_delta, dy)
        _, gcn_cos = projection_and_cosine(gcn_delta, dy)
        acc["grit_cos_sum"] += grit_cos
        acc["grit_cos_count"] += 1
        acc["gcn_plus_cos_sum"] += gcn_cos
        acc["gcn_plus_cos_count"] += 1
    else:
        acc["silent_count"] += 1
        acc["grit_silent_norm_sum"] += grit_norm
        acc["gcn_plus_silent_norm_sum"] += gcn_norm
    y = dy.detach().cpu().numpy().astype(np.float64, copy=False)
    grit = grit_delta.detach().cpu().numpy().astype(np.float64, copy=False)
    gcn = gcn_delta.detach().cpu().numpy().astype(np.float64, copy=False)
    acc["sum_y"] += float(y.sum())
    acc["sum_y2"] += float(np.dot(y, y))
    acc["sum_grit"] += float(grit.sum())
    acc["sum_grit2"] += float(np.dot(grit, grit))
    acc["sum_gcn_plus"] += float(gcn.sum())
    acc["sum_gcn_plus2"] += float(np.dot(gcn, gcn))
    acc["sum_grit_y"] += float(np.dot(grit, y))
    acc["sum_gcn_plus_y"] += float(np.dot(gcn, y))
    acc["sum_grit_gcn_plus"] += float(np.dot(grit, gcn))


def teacher_channel_std(records: Sequence[Mapping[str, Any]]) -> float:
    values = torch.cat([record["teacher"]["Y"].float().reshape(-1) for record in records])
    return float(values.std(unbiased=False).item())


def run_functional_node_responses(
    cfg: Mapping[str, Any],
    task: str,
    *,
    swaps_path: Path | None = None,
    seed: int = 9301,
    device_name: str = "auto",
    batch_size: int = 1024,
    gnn_depth: int = 2,
    grit_config: Path | None = None,
    gcn_plus_config: Path | None = None,
    grit_checkpoint: Path | None = None,
    gcn_plus_checkpoint: Path | None = None,
    store_node_rows: bool = True,
    leakage_tol: float = 1.0e-5,
    fail_on_gcn_leakage: bool = False,
    fast_dev_run: bool = False,
) -> tuple[Path, Path, Path]:
    device = choose_device(device_name)
    configure_runtime(cfg, device)
    swaps_path = swaps_path or functional_node_swap_cache_path(cfg, task, seed)
    cache = load_functional_swap_cache(swaps_path)
    base_records: list[dict[str, Any]] = list(cache["base_records"])
    swaps: list[dict[str, Any]] = list(cache["swaps"])
    if fast_dev_run:
        swaps = swaps[: min(len(swaps), 16)]
    grit_cfg = load_stage1_model_config(cfg, task, model_name="grit", config_path=grit_config, fast_dev_run=fast_dev_run)
    gcn_cfg = load_stage1_model_config(cfg, task, model_name="gcn_plus", config_path=gcn_plus_config, fast_dev_run=fast_dev_run)
    grit, _ = load_model_from_checkpoint(grit_cfg, task, grit_checkpoint, device, backend="official")
    gcn_plus, _ = load_model_from_checkpoint(gcn_cfg, task, gcn_plus_checkpoint, device, backend="official_gnnplus")
    models = {"grit": grit, "gcn_plus": gcn_plus}
    epsilon = 1.0e-6 * max(teacher_channel_std(base_records), 1.0e-12)
    print(
        f"[functional-e1n] task={task} swaps={len(swaps)} graphs={len(base_records)} "
        f"epsilon={epsilon:.3e} device={device} batch_size={batch_size}",
        flush=True,
    )
    base_outputs = {
        model_name: predict_node_outputs(model, base_records, batch_size=batch_size, device=device)
        for model_name, model in models.items()
    }
    out_dir = functional_node_responses_dir(cfg, task)
    out_dir.mkdir(parents=True, exist_ok=True)
    node_table_path = functional_node_response_table_path(cfg, task, seed)
    stats_path = functional_node_graph_stats_path(cfg, task, seed)
    validity_path = functional_node_validity_path(cfg, task, seed)
    node_fields = [
        "task",
        "graph_id",
        "swap_id",
        "u",
        "v",
        "query_i",
        "D_i",
        "stratum",
        "dy_norm",
        "grit_delta_norm",
        "gcn_plus_delta_norm",
        "grit_teacher_projection",
        "gcn_plus_teacher_projection",
        "grit_teacher_cosine",
        "gcn_plus_teacher_cosine",
    ]
    node_handle = None
    node_writer = None
    if store_node_rows:
        node_handle = node_table_path.open("w", newline="", encoding="utf-8")
        node_writer = csv.DictWriter(node_handle, fieldnames=node_fields)
        node_writer.writeheader()
    graph_stats: dict[tuple[str, str], dict[str, Any]] = {}
    chunks = max(1, math.ceil(len(swaps) / int(batch_size)))
    start_time = time.time()
    beyond_l_gcn_max = 0.0
    beyond_l_gcn_violations = 0
    for start in range(0, len(swaps), int(batch_size)):
        chunk_swaps = swaps[start : start + int(batch_size)]
        source_records = [
            source_for_intervention(
                base_records[int(row["graph_index"])],
                str(row["family"]),
                int(row["u"]),
                int(row["v"]),
                cfg,
            )
            for row in chunk_swaps
        ]
        source_outputs = {
            model_name: predict_node_outputs(model, source_records, batch_size=batch_size, device=device)
            for model_name, model in models.items()
        }
        for row_idx, swap in enumerate(chunk_swaps):
            graph_index = int(swap["graph_index"])
            base = base_records[graph_index]
            source = source_records[row_idx]
            n = int(base["n"])
            u = int(swap["u"])
            v = int(swap["v"])
            dy_nodes = source["teacher"]["Y"].float()[:n] - base["teacher"]["Y"].float()[:n]
            grit_delta_nodes = source_outputs["grit"][row_idx].float() - base_outputs["grit"][graph_index].float()
            gcn_delta_nodes = source_outputs["gcn_plus"][row_idx].float() - base_outputs["gcn_plus"][graph_index].float()
            spd = base["struct"]["shortest_path_distance"]
            dist_to_swapped = torch.minimum(spd[:n, u], spd[:n, v])
            for query_idx in range(n):
                d_i = int(dist_to_swapped[query_idx])
                stratum = node_distance_stratum(query_idx, u, v, d_i, gnn_depth)
                key = (str(base["graph_id"]), stratum)
                if key not in graph_stats:
                    graph_stats[key] = empty_node_stat_accumulator(task, str(base["graph_id"]), stratum)
                dy = dy_nodes[query_idx]
                grit_delta = grit_delta_nodes[query_idx]
                gcn_delta = gcn_delta_nodes[query_idx]
                update_node_stat_accumulator(graph_stats[key], dy=dy, grit_delta=grit_delta, gcn_delta=gcn_delta, epsilon=epsilon)
                gcn_norm = float(torch.linalg.vector_norm(gcn_delta).item())
                if stratum == "d_gt_L":
                    beyond_l_gcn_max = max(beyond_l_gcn_max, gcn_norm)
                    if gcn_norm > float(leakage_tol):
                        beyond_l_gcn_violations += 1
                if node_writer is not None:
                    grit_projection, grit_cosine = projection_and_cosine(grit_delta, dy)
                    gcn_projection, gcn_cosine = projection_and_cosine(gcn_delta, dy)
                    node_writer.writerow(
                        {
                            "task": task,
                            "graph_id": str(base["graph_id"]),
                            "swap_id": str(swap["swap_id"]),
                            "u": u,
                            "v": v,
                            "query_i": query_idx,
                            "D_i": d_i,
                            "stratum": stratum,
                            "dy_norm": float(torch.linalg.vector_norm(dy).item()),
                            "grit_delta_norm": float(torch.linalg.vector_norm(grit_delta).item()),
                            "gcn_plus_delta_norm": gcn_norm,
                            "grit_teacher_projection": grit_projection,
                            "gcn_plus_teacher_projection": gcn_projection,
                            "grit_teacher_cosine": grit_cosine,
                            "gcn_plus_teacher_cosine": gcn_cosine,
                        }
                    )
        chunk_idx = start // int(batch_size) + 1
        elapsed = time.time() - start_time
        print(
            f"[functional-e1n] task={task} chunk={chunk_idx}/{chunks} swaps_done={min(start + int(batch_size), len(swaps))} "
            f"elapsed={elapsed:.1f}s beyond_L_gcn_max={beyond_l_gcn_max:.3e}",
            flush=True,
        )
    if node_handle is not None:
        node_handle.close()
    elif node_table_path.exists():
        node_table_path.unlink()
    write_csv(stats_path, list(graph_stats.values()))
    validity_rows = [
        {
            "task": task,
            "seed": int(seed),
            "gnn_depth": int(gnn_depth),
            "epsilon": epsilon,
            "leakage_tol": float(leakage_tol),
            "beyond_L_gcn_max_norm": beyond_l_gcn_max,
            "beyond_L_gcn_violations": beyond_l_gcn_violations,
            "status": "pass" if beyond_l_gcn_violations == 0 else "fail",
        }
    ]
    write_csv(validity_path, validity_rows)
    print(
        f"[functional-e1n] wrote node_rows={node_table_path if store_node_rows else 'disabled'} "
        f"stats={stats_path} validity={validity_path}",
        flush=True,
    )
    if fail_on_gcn_leakage and beyond_l_gcn_violations > 0:
        raise RuntimeError(
            f"E1-N GCN+ beyond-L leakage failed: max={beyond_l_gcn_max:.6g}, "
            f"violations={beyond_l_gcn_violations}, tol={leakage_tol}"
        )
    return node_table_path, stats_path, validity_path


def combine_node_graph_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    fields = [
        "observations",
        "channel_n",
        "dy_norm_sum",
        "grit_norm_sum",
        "gcn_plus_norm_sum",
        "grit_cos_sum",
        "grit_cos_count",
        "gcn_plus_cos_sum",
        "gcn_plus_cos_count",
        "silent_count",
        "grit_silent_norm_sum",
        "gcn_plus_silent_norm_sum",
        "sum_y",
        "sum_y2",
        "sum_grit",
        "sum_grit2",
        "sum_gcn_plus",
        "sum_gcn_plus2",
        "sum_grit_y",
        "sum_gcn_plus_y",
        "sum_grit_gcn_plus",
    ]
    out = {field: 0.0 for field in fields}
    out["gcn_plus_max_norm"] = 0.0
    for row in rows:
        for field in fields:
            out[field] += float(row.get(field, 0.0))
        out["gcn_plus_max_norm"] = max(out["gcn_plus_max_norm"], float(row.get("gcn_plus_max_norm", 0.0)))
    return out


def corr_from_sufficient(n: float, sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float) -> float:
    if n < 2:
        return float("nan")
    cov = sum_xy - (sum_x * sum_y / n)
    vx = sum_x2 - (sum_x * sum_x / n)
    vy = sum_y2 - (sum_y * sum_y / n)
    denom = math.sqrt(max(vx, 0.0) * max(vy, 0.0))
    if denom <= EPS:
        return float("nan")
    return float(cov / denom)


def pcorr_from_sufficient(combined: Mapping[str, float]) -> float:
    n = float(combined["channel_n"])
    r_xy = corr_from_sufficient(
        n,
        float(combined["sum_grit"]),
        float(combined["sum_y"]),
        float(combined["sum_grit2"]),
        float(combined["sum_y2"]),
        float(combined["sum_grit_y"]),
    )
    r_zy = corr_from_sufficient(
        n,
        float(combined["sum_gcn_plus"]),
        float(combined["sum_y"]),
        float(combined["sum_gcn_plus2"]),
        float(combined["sum_y2"]),
        float(combined["sum_gcn_plus_y"]),
    )
    r_xz = corr_from_sufficient(
        n,
        float(combined["sum_grit"]),
        float(combined["sum_gcn_plus"]),
        float(combined["sum_grit2"]),
        float(combined["sum_gcn_plus2"]),
        float(combined["sum_grit_gcn_plus"]),
    )
    if not all(np.isfinite([r_xy, r_zy, r_xz])):
        return float("nan")
    denom = math.sqrt(max(1.0 - r_xz * r_xz, 0.0) * max(1.0 - r_zy * r_zy, 0.0))
    if denom <= EPS:
        return float("nan")
    return float((r_xy - r_xz * r_zy) / denom)


def node_stat_value(rows: Sequence[Mapping[str, Any]], stat: str) -> float:
    combined = combine_node_graph_stats(rows)
    obs = max(float(combined["observations"]), 1.0)
    channel_n = float(combined["channel_n"])
    if stat == "occupancy":
        return float(combined["observations"])
    if stat == "oracle_demand_mean":
        return float(combined["dy_norm_sum"] / obs)
    if stat == "grit_delta_norm_mean":
        return float(combined["grit_norm_sum"] / obs)
    if stat == "gcn_plus_delta_norm_mean":
        return float(combined["gcn_plus_norm_sum"] / obs)
    if stat == "grit_channel_corr":
        return corr_from_sufficient(
            channel_n,
            combined["sum_grit"],
            combined["sum_y"],
            combined["sum_grit2"],
            combined["sum_y2"],
            combined["sum_grit_y"],
        )
    if stat == "gcn_plus_channel_corr":
        return corr_from_sufficient(
            channel_n,
            combined["sum_gcn_plus"],
            combined["sum_y"],
            combined["sum_gcn_plus2"],
            combined["sum_y2"],
            combined["sum_gcn_plus_y"],
        )
    if stat == "grit_excess_partial_corr":
        return pcorr_from_sufficient(combined)
    if stat == "grit_cosine_mean":
        count = float(combined["grit_cos_count"])
        return float(combined["grit_cos_sum"] / count) if count > 0 else float("nan")
    if stat == "gcn_plus_cosine_mean":
        count = float(combined["gcn_plus_cos_count"])
        return float(combined["gcn_plus_cos_sum"] / count) if count > 0 else float("nan")
    if stat == "grit_spurious_silent_norm_mean":
        count = float(combined["silent_count"])
        return float(combined["grit_silent_norm_sum"] / count) if count > 0 else float("nan")
    if stat == "gcn_plus_spurious_silent_norm_mean":
        count = float(combined["silent_count"])
        return float(combined["gcn_plus_silent_norm_sum"] / count) if count > 0 else float("nan")
    if stat == "gcn_plus_max_norm":
        return float(combined["gcn_plus_max_norm"])
    raise ValueError(stat)


def summarise_functional_node_responses(
    cfg: Mapping[str, Any],
    *,
    tasks: Sequence[str] | None = None,
    seed: int = 9301,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 9401,
) -> tuple[Path, Path]:
    tasks = list(tasks or TASKS)
    rows: list[dict[str, str]] = []
    validity_rows: list[dict[str, str]] = []
    for task in tasks:
        stats_path = functional_node_graph_stats_path(cfg, task, seed)
        validity_path = functional_node_validity_path(cfg, task, seed)
        if not stats_path.exists():
            raise FileNotFoundError(f"E1-N graph stats not found: {stats_path}")
        rows.extend(read_csv_dicts(stats_path))
        if validity_path.exists():
            validity_rows.extend(read_csv_dicts(validity_path))
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["task"]), str(row["stratum"]))].append(row)
    stat_names = [
        "occupancy",
        "oracle_demand_mean",
        "grit_delta_norm_mean",
        "gcn_plus_delta_norm_mean",
        "grit_channel_corr",
        "gcn_plus_channel_corr",
        "grit_excess_partial_corr",
        "grit_cosine_mean",
        "gcn_plus_cosine_mean",
        "grit_spurious_silent_norm_mean",
        "gcn_plus_spurious_silent_norm_mean",
        "gcn_plus_max_norm",
    ]
    summary_rows = []
    for (task, stratum), group_rows in sorted(grouped.items()):
        graphs = len({str(row["graph_id"]) for row in group_rows})
        for stat_name in stat_names:
            point, lo, hi = cluster_bootstrap_ci(
                group_rows,
                lambda sample, name=stat_name: node_stat_value(sample, name),
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + stable_int_seed(task, stratum, stat_name) % 100000,
            )
            summary_rows.append(
                {
                    "task": task,
                    "stratum": stratum,
                    "stratum_label": FUNCTIONAL_NODE_STRATUM_LABELS.get(stratum, stratum),
                    "graphs": graphs,
                    "stat": stat_name,
                    "mean": point,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    metrics = functional_metrics_dir(cfg)
    summary_path = metrics / "functional_e1n_node_fingerprint_stats.csv"
    validity_out = metrics / "functional_e1n_validity.csv"
    write_csv(summary_path, summary_rows)
    write_csv(validity_out, validity_rows)
    print(f"[functional-e1n-summary] wrote summary={summary_path} validity={validity_out}", flush=True)
    return summary_path, validity_out


def plot_functional_node_responses(cfg: Mapping[str, Any], *, tasks: Sequence[str] | None = None) -> None:
    tasks = list(tasks or TASKS)
    fig_dir = functional_figures_dir(cfg)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv_dicts(functional_metrics_dir(cfg) / "functional_e1n_node_fingerprint_stats.csv")
    validity_rows = read_csv_dicts(functional_metrics_dir(cfg) / "functional_e1n_validity.csv")
    if not rows:
        raise FileNotFoundError("E1-N summary CSV is missing or empty; run summarise-functional-node-responses first")
    indexed = rows_by_task_stratum(rows)
    strata = list(FUNCTIONAL_NODE_STRATA)
    x = np.arange(len(strata), dtype=np.float64)
    colors = {"grit": "#1b7f79", "gcn_plus": "#b24c3d", "oracle": "#3f6f8f"}

    fig, axes = plt.subplots(2, max(1, len(tasks)), figsize=(5.4 * max(1, len(tasks)), 7.2), sharex=True)
    axes_arr = np.asarray(axes).reshape(2, -1)
    for col, task in enumerate(tasks):
        ax_occ = axes_arr[0, col]
        ax_oracle = axes_arr[1, col]
        occ = np.asarray([get_metric_value(indexed.get((task, s), {}), "occupancy") for s in strata])
        demand = np.asarray([get_metric_value(indexed.get((task, s), {}), "oracle_demand_mean") for s in strata])
        ax_occ.bar(x, occ, color="#6b6f7a", width=0.68)
        ax_oracle.bar(x, demand, color=colors["oracle"], width=0.68)
        ax_occ.set_title(task)
        ax_occ.set_ylabel("Node observations")
        ax_oracle.set_ylabel("Mean ||Δy_i||")
        for ax in (ax_occ, ax_oracle):
            ax.set_xticks(x)
            ax.set_xticklabels([FUNCTIONAL_NODE_STRATUM_LABELS[s] for s in strata], rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_e1n_node_oracle_demand_occupancy.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, max(1, len(tasks)), figsize=(5.4 * max(1, len(tasks)), 7.2), sharex=True)
    axes_arr = np.asarray(axes).reshape(2, -1)
    for col, task in enumerate(tasks):
        ax_mag = axes_arr[0, col]
        ax_corr = axes_arr[1, col]
        for offset, model_name in [(-0.18, "grit"), (0.18, "gcn_plus")]:
            norm = np.asarray([get_metric_value(indexed.get((task, s), {}), f"{model_name}_delta_norm_mean") for s in strata])
            corr_name = "grit_channel_corr" if model_name == "grit" else "gcn_plus_channel_corr"
            corr = np.asarray([get_metric_value(indexed.get((task, s), {}), corr_name) for s in strata])
            ax_mag.bar(x + offset, norm, width=0.34, label=model_name, color=colors[model_name])
            ax_corr.bar(x + offset, corr, width=0.34, label=model_name, color=colors[model_name])
        ax_mag.set_title(task)
        ax_mag.set_ylabel("Mean ||Δf_i||")
        ax_corr.set_ylabel("corr(Δf_i,c, Δy_i,c)")
        ax_corr.axhline(0.0, color="black", lw=0.8)
        for ax in (ax_mag, ax_corr):
            ax.set_xticks(x)
            ax.set_xticklabels([FUNCTIONAL_NODE_STRATUM_LABELS[s] for s in strata], rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
    axes_arr[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_e1n_node_response_magnitude_alignment.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, max(1, len(tasks)), figsize=(5.2 * max(1, len(tasks)), 3.9), sharey=True)
    axes_list = np.atleast_1d(axes)
    for ax, task in zip(axes_list, tasks):
        vals = np.asarray([get_metric_value(indexed.get((task, s), {}), "grit_excess_partial_corr") for s in strata])
        ax.bar(x, vals, color=colors["grit"], width=0.68)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(task)
        ax.set_ylabel("pcorr(GRIT, teacher | GCN+)")
        ax.set_xticks(x)
        ax.set_xticklabels([FUNCTIONAL_NODE_STRATUM_LABELS[s] for s in strata], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_e1n_node_excess_alignment.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, max(1, len(tasks)), figsize=(5.2 * max(1, len(tasks)), 3.9), sharey=False)
    axes_list = np.atleast_1d(axes)
    for ax, task in zip(axes_list, tasks):
        gt_signal = get_metric_value(indexed.get((task, "d_gt_L"), {}), "grit_channel_corr")
        gcn_leak = get_metric_value(indexed.get((task, "d_gt_L"), {}), "gcn_plus_max_norm")
        spurious = get_metric_value(indexed.get((task, "d_gt_L"), {}), "grit_spurious_silent_norm_mean")
        ax.bar([0, 1, 2], [gt_signal, gcn_leak, spurious], color=[colors["grit"], colors["gcn_plus"], "#6b6f7a"])
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["GRIT corr\nbeyond L", "GCN+ max\nleakage", "GRIT silent\nnorm"], rotation=20, ha="right")
        ax.set_title(task)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_e1n_beyond_l_validity_and_signal.pdf")
    plt.close(fig)

    if validity_rows:
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        labels = [str(row["task"]) for row in validity_rows]
        vals = [float(row.get("beyond_L_gcn_max_norm", "nan")) for row in validity_rows]
        ax.bar(np.arange(len(vals)), vals, color="#b24c3d")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("max ||Δf_GCN+,i|| for D_i > L")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "functional_e1n_gcn_beyond_l_validity.pdf")
        plt.close(fig)
    print(f"[functional-e1n-plot] wrote figures={fig_dir}", flush=True)


def q1_stat_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    gate_type: str,
    stat: str,
    field: str = "mean",
) -> float:
    for row in rows:
        if row.get("task") == task and row.get("gate_type") == gate_type and row.get("stat") == stat:
            return finite_float(row.get(field))
    return float("nan")


def status_rank(status: str) -> int:
    return {"pass": 0, "warn": 1, "fail": 2}.get(str(status), 2)


def worst_status(statuses: Sequence[str]) -> str:
    if not statuses:
        return "fail"
    return max(statuses, key=status_rank)


def plot_functional_q1_gates(cfg: Mapping[str, Any], *, tasks: Sequence[str] | None = None) -> None:
    tasks = list(tasks or TASKS)
    fig_dir = functional_figures_dir(cfg)
    fig_dir.mkdir(parents=True, exist_ok=True)
    quartile_rows = read_csv_dicts(functional_q1_gate_quartile_stats_path(cfg))
    points = read_csv_dicts(functional_q1_graph_coupling_points_path(cfg))
    graph_stats = read_csv_dicts(functional_q1_graph_coupling_stats_path(cfg))
    contrast_rows = read_csv_dicts(functional_q1_gate_contrast_stats_path(cfg))
    if not quartile_rows or not points or not graph_stats:
        raise FileNotFoundError("Q1 summary CSVs are missing or empty; run summarise-functional-q1-gates first")

    colors = {"routing": "#2b6cb0", "transport": "#805ad5"}
    x = np.arange(len(Q1_GATE_BINS), dtype=np.float64)
    fig, axes = plt.subplots(
        max(1, len(tasks)),
        2,
        figsize=(10.5, 3.3 * max(1, len(tasks))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_idx, task in enumerate(tasks):
        for col_idx, gate_type in enumerate(("routing", "transport")):
            ax = axes[row_idx, col_idx]
            vals = []
            lows = []
            highs = []
            labels = []
            statuses = []
            for gate_bin in Q1_GATE_BINS:
                subset = [
                    row
                    for row in quartile_rows
                    if row.get("task") == task
                    and row.get("gate_type") == gate_type
                    and row.get("conditioning") == "all"
                    and row.get("gate_bin") == gate_bin
                    and row.get("stat") == "grit_excess_partial_corr"
                ]
                row = subset[0] if subset else {}
                vals.append(finite_float(row.get("mean")))
                lows.append(finite_float(row.get("ci_low")))
                highs.append(finite_float(row.get("ci_high")))
                labels.append(Q1_GATE_BIN_LABELS.get(gate_bin, gate_bin).replace(" gate", ""))
                statuses.append(str(row.get("status", "missing")))
            vals_arr = np.asarray(vals, dtype=np.float64)
            lows_arr = np.asarray(lows, dtype=np.float64)
            highs_arr = np.asarray(highs, dtype=np.float64)
            ax.errorbar(
                x,
                vals_arr,
                yerr=nonnegative_ci_yerr(vals_arr, lows_arr, highs_arr),
                marker="o",
                color=colors[gate_type],
                linewidth=1.4,
                capsize=3,
            )
            for xi, status in zip(x, statuses):
                if status == "underpowered":
                    ax.text(xi, 0.02, "u", ha="center", va="bottom", fontsize=8, color="#b24c3d")
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(f"{task}: {gate_type}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel("pcorr(GRIT, teacher | GCN+)")
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Q1: functional excess by GRIT global gate quartile", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_dir / "functional_q1_gate_excess_quartiles.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, max(1, len(tasks)), figsize=(4.6 * max(1, len(tasks)), 3.9), sharey=True)
    axes_list = np.atleast_1d(axes)
    for ax, task in zip(axes_list, tasks):
        vals = np.asarray([q1_stat_row(graph_stats, task=task, gate_type=g, stat="pearson") for g in ("routing", "transport")])
        lows = np.asarray([q1_stat_row(graph_stats, task=task, gate_type=g, stat="pearson", field="ci_low") for g in ("routing", "transport")])
        highs = np.asarray([q1_stat_row(graph_stats, task=task, gate_type=g, stat="pearson", field="ci_high") for g in ("routing", "transport")])
        ax.bar([0, 1], vals, yerr=nonnegative_ci_yerr(vals, lows, highs), capsize=3, color=[colors["routing"], colors["transport"]])
        contrast = next((row for row in contrast_rows if row.get("task") == task and row.get("stat") == "pearson"), None)
        diff = finite_float(contrast.get("mean") if contrast is not None else None)
        ax.text(0.5, 0.96, f"route - transport = {diff:.2f}", ha="center", va="top", transform=ax.transAxes, fontsize=8)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["routing", "transport"])
        ax.set_title(task)
        ax.set_ylabel("corr(graph gate, graph excess)")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_q1_route_vs_transport.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(max(1, len(tasks)), 2, figsize=(10.0, 3.3 * max(1, len(tasks))), squeeze=False)
    for row_idx, task in enumerate(tasks):
        for col_idx, gate_type in enumerate(("routing", "transport")):
            ax = axes[row_idx, col_idx]
            subset = [
                row
                for row in points
                if row.get("task") == task and row.get("gate_type") == gate_type and row.get("status") == "ok"
            ]
            gate = numeric_column(subset, "graph_gate_mean")
            excess = numeric_column(subset, "graph_excess_partial_corr")
            ax.scatter(gate, excess, s=18, alpha=0.68, color=colors[gate_type], edgecolor="none")
            finite = np.isfinite(gate) & np.isfinite(excess)
            if int(finite.sum()) >= 2:
                coef = np.polyfit(gate[finite], excess[finite], deg=1)
                xs = np.linspace(float(np.nanmin(gate[finite])), float(np.nanmax(gate[finite])), 100)
                ax.plot(xs, coef[0] * xs + coef[1], color="#2d3748", linewidth=1.0)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_title(f"{task}: {gate_type}")
            ax.set_xlabel("mean graph gate")
            ax.set_ylabel("graph-level excess")
            ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_q1_graph_coupling_scatter.pdf")
    plt.close(fig)

    validity_rows = read_csv_dicts(functional_validity_gate_path(cfg)) if functional_validity_gate_path(cfg).exists() else []
    summary_rows = []
    for task in tasks:
        statuses = [str(row.get("status", "fail")) for row in validity_rows if row.get("task") == task]
        summary_rows.append(
            [
                task,
                worst_status(statuses),
                f"{q1_stat_row(graph_stats, task=task, gate_type='routing', stat='pearson'):.2f}",
                f"{q1_stat_row(graph_stats, task=task, gate_type='transport', stat='pearson'):.2f}",
                f"{next((finite_float(row.get('mean')) for row in contrast_rows if row.get('task') == task and row.get('stat') == 'pearson'), float('nan')):.2f}",
            ]
        )
    fig, ax = plt.subplots(figsize=(9.2, max(2.6, 0.65 * len(summary_rows) + 1.4)))
    ax.axis("off")
    ax.set_title("Q1 validity and headline graph-coupling summary", fontsize=13, loc="left", pad=8)
    table = ax.table(
        cellText=summary_rows,
        colLabels=["Task", "Validity", "routing corr", "transport corr", "route - transport"],
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        colWidths=[0.30, 0.14, 0.18, 0.20, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d4dc")
        cell.set_linewidth(0.4)
        if row_idx == 0:
            cell.set_facecolor("#edf2f7")
            cell.set_text_props(weight="bold")
        elif col_idx == 1:
            status = summary_rows[row_idx - 1][1]
            cell.set_facecolor({"pass": "#d9f0e3", "warn": "#fff1c7", "fail": "#f7d6d0"}.get(status, "#ffffff"))
            cell.set_text_props(weight="bold")
    table.scale(1.0, 1.6)
    fig.tight_layout()
    fig.savefig(fig_dir / "functional_q1_validity_plus_result_summary.pdf")
    plt.close(fig)
    print(f"[functional-q1-plot] wrote figures={fig_dir}", flush=True)


def grouped_folds(groups: Sequence[str], folds: int, seed: int) -> list[set[str]]:
    unique = sorted(set(groups))
    rng = random.Random(int(seed))
    rng.shuffle(unique)
    k = max(2, min(int(folds), len(unique)))
    out = [set() for _ in range(k)]
    for idx, group in enumerate(unique):
        out[idx % k].add(group)
    return out


def fit_ridge_standardized(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if x_train.shape[1] == 0:
        pred = np.full(x_test.shape[0], float(np.mean(y_train)), dtype=np.float64)
        return pred, np.empty(0), np.empty(0), float(np.mean(y_train))
    mean = np.nanmean(x_train, axis=0)
    std = np.nanstd(x_train, axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    xtr = np.nan_to_num((x_train - mean) / std)
    xte = np.nan_to_num((x_test - mean) / std)
    y_mean = float(np.mean(y_train))
    yc = y_train - y_mean
    gram = xtr.T @ xtr
    rhs = xtr.T @ yc
    weights = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), rhs)
    pred = xte @ weights + y_mean
    return pred, weights, mean, y_mean


def cv_ridge_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_columns: Sequence[str],
    target_column: str,
    folds: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    groups = [str(row["graph_id"]) for row in rows]
    fold_groups = grouped_folds(groups, folds, seed)
    x = numeric_matrix(rows, feature_columns)
    y = np.asarray([float(row[target_column]) for row in rows], dtype=np.float64)
    fold_rows = []
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    for fold_idx, held_groups in enumerate(fold_groups):
        test_mask = np.asarray([group in held_groups for group in groups], dtype=bool)
        train_mask = ~test_mask
        if not bool(test_mask.any()) or not bool(train_mask.any()):
            continue
        pred, _weights, _mean, _y_mean = fit_ridge_standardized(
            x[train_mask],
            y[train_mask],
            x[test_mask],
            alpha=alpha,
        )
        predictions[test_mask] = pred
        y_test = y[test_mask]
        sse = float(np.square(y_test - pred).sum())
        sst = float(np.square(y_test - float(np.mean(y_test))).sum())
        r2 = float("nan") if sst <= EPS else 1.0 - sse / sst
        fold_rows.append(
            {
                "fold": fold_idx,
                "r2": r2,
                "pearson": pearson_corr(y_test, pred),
                "spearman": spearman_corr(y_test, pred),
                "n_test": int(test_mask.sum()),
                "groups_test": len(held_groups),
            }
        )
    valid = np.isfinite(predictions)
    return {
        "n": len(rows),
        "groups": len(set(groups)),
        "folds": len(fold_rows),
        "r2_mean": float(np.nanmean([row["r2"] for row in fold_rows])) if fold_rows else float("nan"),
        "r2_std": float(np.nanstd([row["r2"] for row in fold_rows])) if fold_rows else float("nan"),
        "pearson_oof": pearson_corr(y[valid], predictions[valid]) if bool(valid.any()) else float("nan"),
        "spearman_oof": spearman_corr(y[valid], predictions[valid]) if bool(valid.any()) else float("nan"),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
    }


def final_ridge_coefficients(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_columns: Sequence[str],
    target_column: str,
    alpha: float,
) -> list[dict[str, Any]]:
    if not feature_columns:
        return []
    x = numeric_matrix(rows, feature_columns)
    y = np.asarray([float(row[target_column]) for row in rows], dtype=np.float64)
    _pred, weights, _mean, _y_mean = fit_ridge_standardized(x, y, x, alpha=alpha)
    return [
        {"feature": feature, "standardized_coefficient": float(weight)}
        for feature, weight in zip(feature_columns, weights)
    ]


def response_feature_sets(feature_columns: Sequence[str]) -> dict[str, list[str]]:
    return {
        "intercept_only": [],
        "all_scores": list(feature_columns),
        "routing_scores": [col for col in feature_columns if "_routing_" in col],
        "transport_scores": [col for col in feature_columns if "_transport_" in col],
        "follow_scores": [col for col in feature_columns if "_follow_" in col],
        "invariant_scores": [col for col in feature_columns if "_invariant_" in col],
    }


def payload_gating_controls(row: Mapping[str, Any]) -> dict[str, float]:
    base = row["base_graph"]
    n = int(base["n"])
    u = int(row["u"])
    v = int(row["v"])
    k = base["teacher"]["K"].float()
    m = base["teacher"]["M"].float()
    ku = k[:n, u].float()
    kv = k[:n, v].float()
    k_col_diff_l2 = torch.linalg.vector_norm(ku - kv)
    m_delta_l2 = torch.linalg.vector_norm(m[v].float() - m[u].float())
    degree = base["struct"]["degree"].float()
    spd = base["struct"]["shortest_path_distance"]
    anchor_indicator = base.get("anchor_indicator")
    if anchor_indicator is None:
        u_is_anchor = 0.0
        v_is_anchor = 0.0
    else:
        u_is_anchor = float(anchor_indicator[u].item() > 0.5)
        v_is_anchor = float(anchor_indicator[v].item() > 0.5)
    try:
        same_query_anchor = float(int(torch.argmax(k[u]).item()) == int(torch.argmax(k[v]).item()))
    except Exception:
        same_query_anchor = float("nan")
    spd_uv = int(spd[u, v])
    return {
        "ctrl_k_col_u_mass": float(ku.sum().item()),
        "ctrl_k_col_v_mass": float(kv.sum().item()),
        "ctrl_k_col_mass_sum": float((ku.sum() + kv.sum()).item()),
        "ctrl_k_col_mass_absdiff": float(torch.abs(ku.sum() - kv.sum()).item()),
        "ctrl_k_col_diff_l2": float(k_col_diff_l2.item()),
        "ctrl_k_col_dot": float(torch.dot(ku, kv).item()),
        "ctrl_m_delta_l2": float(m_delta_l2.item()),
        "ctrl_km_product_l2": float((k_col_diff_l2 * m_delta_l2).item()),
        "ctrl_u_is_anchor": u_is_anchor,
        "ctrl_v_is_anchor": v_is_anchor,
        "ctrl_num_anchor_swapped": u_is_anchor + v_is_anchor,
        "ctrl_degree_u": float(degree[u].item()),
        "ctrl_degree_v": float(degree[v].item()),
        "ctrl_degree_absdiff": float(torch.abs(degree[u] - degree[v]).item()),
        "ctrl_spd_uv": float(spd_uv if spd_uv >= 0 else 1.0e6),
        "ctrl_same_query_anchor": same_query_anchor,
        "ctrl_k_row_diff_l2": float(torch.linalg.vector_norm(k[u].float() - k[v].float()).item()),
    }


def gating_control_feature_sets(score_columns: Sequence[str], control_columns: Sequence[str]) -> dict[str, list[str]]:
    response_sets = response_feature_sets(score_columns)
    k_controls = [col for col in control_columns if col.startswith("ctrl_k_") or col in {"ctrl_u_is_anchor", "ctrl_v_is_anchor", "ctrl_num_anchor_swapped", "ctrl_same_query_anchor"}]
    m_controls = [col for col in control_columns if col.startswith("ctrl_m_")]
    structural_controls = [col for col in control_columns if col.startswith("ctrl_degree_") or col == "ctrl_spd_uv"]
    km_controls = list(dict.fromkeys(k_controls + m_controls + ["ctrl_km_product_l2"]))
    return {
        "intercept_only": [],
        "teacher_k_controls": k_controls,
        "teacher_m_controls": m_controls,
        "teacher_struct_controls": structural_controls,
        "teacher_km_controls": km_controls,
        "routing_scores": response_sets["routing_scores"],
        "transport_scores": response_sets["transport_scores"],
        "transport_plus_k_controls": response_sets["transport_scores"] + k_controls,
        "transport_plus_km_controls": response_sets["transport_scores"] + km_controls,
        "routing_plus_k_controls": response_sets["routing_scores"] + k_controls,
        "routing_plus_km_controls": response_sets["routing_scores"] + km_controls,
        "all_scores": response_sets["all_scores"],
        "all_scores_plus_km_controls": response_sets["all_scores"] + km_controls,
    }


def plot_payload_gating_controls(
    cfg: Mapping[str, Any],
    task: str,
    family: str,
    summary_rows: Sequence[Mapping[str, Any]],
) -> Path:
    out = figures_main_dir(cfg) / f"fig8_payload_gating_controls_{task}_{family}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    targets = [
        "teacher_pathway_norm",
        "student_teacher_pathway_projection",
        "student_delta_norm",
    ]
    target_labels = {
        "teacher_pathway_norm": "teacher pathway norm",
        "student_teacher_pathway_projection": "student teacher-aligned",
        "student_delta_norm": "student delta norm",
    }
    feature_order = [
        "intercept_only",
        "teacher_k_controls",
        "teacher_m_controls",
        "teacher_km_controls",
        "transport_scores",
        "routing_scores",
        "transport_plus_k_controls",
        "transport_plus_km_controls",
        "all_scores",
        "all_scores_plus_km_controls",
    ]
    label_map = {
        "intercept_only": "intercept",
        "teacher_k_controls": "K controls",
        "teacher_m_controls": "M controls",
        "teacher_km_controls": "K+M controls",
        "transport_scores": "transport",
        "routing_scores": "routing",
        "transport_plus_k_controls": "transport+K",
        "transport_plus_km_controls": "transport+K+M",
        "all_scores": "all scores",
        "all_scores_plus_km_controls": "all+K+M",
    }
    colors = {
        "intercept_only": "#a0aec0",
        "teacher_k_controls": "#2f855a",
        "teacher_m_controls": "#38a169",
        "teacher_km_controls": "#276749",
        "transport_scores": "#805ad5",
        "routing_scores": "#2b6cb0",
        "transport_plus_k_controls": "#d69e2e",
        "transport_plus_km_controls": "#b7791f",
        "all_scores": "#4a5568",
        "all_scores_plus_km_controls": "#1a202c",
    }
    fig, axes = plt.subplots(1, len(targets), figsize=(17, 5.2), sharey=True, squeeze=False)
    fig.suptitle(f"{task} / {family}: do teacher K-gating controls explain payload-swap predictivity?", fontsize=12)
    for ax, target in zip(axes.reshape(-1), targets):
        values = []
        labels = []
        bar_colors = []
        for feature_set in feature_order:
            row = next(
                (
                    item
                    for item in summary_rows
                    if item.get("target") == target and item.get("feature_set") == feature_set
                ),
                None,
            )
            if row is None:
                continue
            values.append(float(row["r2_mean"]))
            labels.append(label_map.get(feature_set, feature_set))
            bar_colors.append(colors.get(feature_set, "#718096"))
        x = np.arange(len(values), dtype=float)
        ax.bar(x, values, color=bar_colors)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(target_labels.get(target, target), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("mean held-out R2")
        ax.grid(axis="y", linewidth=0.35, alpha=0.35)
        if values:
            top = max(max(values) + 0.08, 0.1)
            ax.set_ylim(min(-0.05, min(values) - 0.05), min(1.05, top))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out)
    plt.close(fig)
    return out


def run_payload_gating_controls(
    cfg: Mapping[str, Any],
    task: str,
    *,
    family: Optional[str] = None,
    kind: str = "cf_eval",
    folds: Optional[int] = None,
    ridge_alpha: Optional[float] = None,
) -> None:
    selected_family = family or (default_functional_family(task))
    if selected_family not in {"ppr_payload_swap", "voronoi_payload_swap"}:
        raise ValueError("payload gating controls are defined for ppr_payload_swap or voronoi_payload_swap")
    feature_path = metrics_dir(cfg) / f"response_predictivity_features_{task}.csv"
    feature_rows = [
        row
        for row in read_csv_dicts(feature_path)
        if row.get("family") == selected_family
    ]
    if not feature_rows:
        raise RuntimeError(f"no response predictivity features found for {selected_family} at {feature_path}")
    interventions = torch.load(intervention_dir(cfg, task) / f"{kind}_interventions.pt", map_location="cpu", weights_only=False)
    intervention_by_id = {
        str(row["intervention_id"]): row
        for row in interventions
        if row.get("family") == selected_family
    }
    rows = []
    missing = 0
    for feature_row in feature_rows:
        intervention = intervention_by_id.get(str(feature_row["intervention_id"]))
        if intervention is None:
            missing += 1
            continue
        merged = dict(feature_row)
        merged.update(payload_gating_controls(intervention))
        rows.append(merged)
    if missing:
        print(f"[payload-gating] skipped missing_interventions={missing}", flush=True)
    if not rows:
        raise RuntimeError(f"no joined rows for {selected_family}")
    score_columns = sorted([key for key in rows[0] if key.startswith("L") and ("_routing_" in key or "_transport_" in key)])
    control_columns = sorted([key for key in rows[0] if key.startswith("ctrl_")])
    feature_sets = gating_control_feature_sets(score_columns, control_columns)
    targets = [
        "teacher_pathway_norm",
        "student_teacher_pathway_projection",
        "student_delta_norm",
    ]
    pcfg = cfg.get("response_predictivity", {})
    n_folds = int(folds if folds is not None else pcfg.get("folds", 5))
    alpha = float(ridge_alpha if ridge_alpha is not None else pcfg.get("ridge_alpha", 1.0))
    seed = int(pcfg.get("seed", 7001))
    summary_rows = []
    for target in targets:
        for set_name, cols in feature_sets.items():
            summary = cv_ridge_summary(
                rows,
                feature_columns=cols,
                target_column=target,
                folds=n_folds,
                alpha=alpha,
                seed=seed,
            )
            summary_rows.append(
                {
                    "task": task,
                    "family": selected_family,
                    "target": target,
                    "feature_set": set_name,
                    "ridge_alpha": alpha,
                    "features": len(cols),
                    **summary,
                }
            )
    safe_family = selected_family.replace("/", "_")
    joined_path = metrics_dir(cfg) / f"payload_gating_controls_features_{task}_{safe_family}.csv"
    cv_path = metrics_dir(cfg) / f"payload_gating_controls_cv_{task}_{safe_family}.csv"
    write_csv(joined_path, rows)
    write_csv(cv_path, summary_rows)
    fig_path = plot_payload_gating_controls(cfg, task, safe_family, summary_rows)
    print(
        f"[payload-gating] wrote features={joined_path} cv={cv_path} figure={fig_path} rows={len(rows)} "
        f"score_features={len(score_columns)} controls={len(control_columns)}",
        flush=True,
    )


def run_response_predictivity(
    cfg: Mapping[str, Any],
    task: str,
    *,
    device_name: str = "auto",
    checkpoint: Optional[Path] = None,
    backend: Optional[str] = None,
    families: Optional[Sequence[str]] = None,
    kind: Optional[str] = None,
    max_interventions_per_family: Optional[int] = None,
    batch_size: Optional[int] = None,
    folds: Optional[int] = None,
    ridge_alpha: Optional[float] = None,
    centered: Optional[bool] = None,
) -> None:
    if (backend or cfg["model"].get("backend", "official")) != "official":
        raise RuntimeError("response predictivity currently requires official GRIT fields")
    pcfg = cfg.get("response_predictivity", {})
    default_families = task_families(task)
    selected_families = list(families or pcfg.get("families") or default_families)
    selected_kind = str(kind or pcfg.get("intervention_kind", "cf_eval"))
    max_per_family = int(max_interventions_per_family if max_interventions_per_family is not None else pcfg.get("max_interventions_per_family", 4096))
    batch = int(batch_size if batch_size is not None else pcfg.get("batch_size_graphs", 64))
    n_folds = int(folds if folds is not None else pcfg.get("folds", 5))
    alpha = float(ridge_alpha if ridge_alpha is not None else pcfg.get("ridge_alpha", 1.0))
    use_centered = bool(centered if centered is not None else pcfg.get("centered", False))
    seed = int(pcfg.get("seed", 7001))
    device = choose_device(device_name)
    configure_runtime(cfg, device)
    model, _run_cfg = load_model_from_checkpoint(cfg, task, checkpoint, device, backend=backend)
    interventions = select_response_predictivity_interventions(
        cfg,
        task,
        kind=selected_kind,
        families=selected_families,
        max_per_family=max_per_family,
        seed=seed,
    )
    print(
        f"[response-predictivity] start task={task} kind={selected_kind} "
        f"families={','.join(selected_families)} interventions={len(interventions)} "
        f"batch_size={batch}",
        flush=True,
    )
    rows = compute_response_predictivity_table(
        cfg,
        task,
        interventions=interventions,
        model=model,
        device=device,
        batch_size=batch,
        centered=use_centered,
    )
    feature_path = metrics_dir(cfg) / f"response_predictivity_features_{task}.csv"
    write_csv(feature_path, rows)
    feature_columns = sorted([key for key in rows[0] if key.startswith("L") and ("_routing_" in key or "_transport_" in key)]) if rows else []
    targets = [
        "teacher_pathway_norm",
        "student_teacher_pathway_projection",
        "student_delta_norm",
    ]
    feature_sets = response_feature_sets(feature_columns)
    summary_rows = []
    coefficient_rows = []
    for family in selected_families:
        family_rows = [row for row in rows if row.get("family") == family]
        if not family_rows:
            continue
        matched_set = "transport_scores" if PATHWAY_BY_FAMILY[family] == "M" else "routing_scores"
        mismatched_set = "routing_scores" if matched_set == "transport_scores" else "transport_scores"
        for target in targets:
            for set_name, cols in feature_sets.items():
                summary = cv_ridge_summary(
                    family_rows,
                    feature_columns=cols,
                    target_column=target,
                    folds=n_folds,
                    alpha=alpha,
                    seed=seed,
                )
                summary_rows.append(
                    {
                        "task": task,
                        "family": family,
                        "target": target,
                        "feature_set": set_name,
                        "matched_set": set_name == matched_set,
                        "mismatched_set": set_name == mismatched_set,
                        "ridge_alpha": alpha,
                        **summary,
                    }
                )
                for coef in final_ridge_coefficients(
                    family_rows,
                    feature_columns=cols,
                    target_column=target,
                    alpha=alpha,
                ):
                    coefficient_rows.append(
                        {
                            "task": task,
                            "family": family,
                            "target": target,
                            "feature_set": set_name,
                            "ridge_alpha": alpha,
                            **coef,
                        }
                    )
    summary_path = metrics_dir(cfg) / f"response_predictivity_cv_{task}.csv"
    coef_path = metrics_dir(cfg) / f"response_predictivity_coefficients_{task}.csv"
    write_csv(summary_path, summary_rows)
    write_csv(coef_path, coefficient_rows)
    plot_response_predictivity_summary(cfg, task)
    contrast_path = write_response_predictivity_contrasts(cfg, task)
    print(
        f"[response-predictivity] wrote features={feature_path} cv={summary_path} "
        f"coefficients={coef_path} contrasts={contrast_path}",
        flush=True,
    )


def write_mediation_decisions(cfg: Mapping[str, Any]) -> None:
    rows = read_csv_dicts(metrics_dir(cfg) / "patch_group_metrics.csv")
    decisions = []
    for task in TASKS:
        families = task_families(task)
        for family in families:
            subset = [
                row
                for row in rows
                if row.get("task") == task
                and row.get("family") == family
                and row.get("effect_bin") == "high"
                and row.get("group_name") in {"top4", "top2", "top1"}
            ]
            tcm = median([float(row["TCM"]) for row in subset]) if subset else float("nan")
            tcma = median([float(row["TCMA"]) for row in subset]) if subset else float("nan")
            oc = median([float(row["OC"]) for row in subset]) if subset else float("nan")
            supported = tcma >= 0.70 and tcm >= 0.20 and oc <= 0.75
            decisions.append(
                {
                    "task": task,
                    "family": family,
                    "hypothesis": "H3_interchange_mediation",
                    "decision": "supported" if supported else "not_supported",
                    "evidence_metric": "high_bin_group_median_TCM_TCMA_OC",
                    "estimate": tcm,
                    "ci_low": "",
                    "ci_high": "",
                    "notes": f"median_TCM={tcm:.4g}; median_TCMA={tcma:.4g}; median_OC={oc:.4g}",
                }
            )
    path = metrics_dir(cfg) / "hypothesis_decisions.csv"
    old = [row for row in read_csv_dicts(path) if row.get("hypothesis") != "H3_interchange_mediation"]
    write_csv(path, old + decisions)


def bootstrap_ci_by_graph(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    graph_key: str = "graph_id",
    resamples: int = 2000,
    seed: int = 6060,
) -> tuple[float, float, float]:
    by_graph: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        try:
            value = float(row[metric])
        except Exception:
            continue
        if math.isfinite(value):
            by_graph[str(row[graph_key])].append(value)
    graph_ids = list(by_graph)
    if not graph_ids:
        return float("nan"), float("nan"), float("nan")
    graph_means = {gid: float(np.mean(vals)) for gid, vals in by_graph.items()}
    estimate = float(np.mean(list(graph_means.values())))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(int(resamples)):
        sampled = rng.choice(graph_ids, size=len(graph_ids), replace=True)
        samples.append(float(np.mean([graph_means[str(gid)] for gid in sampled])))
    return estimate, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def plot_training_curves(csv_path: Path, out_path: Path) -> None:
    rows = read_csv_dicts(csv_path)
    if not rows:
        return
    steps = [int(row["step"]) for row in rows]
    train_loss = [float(row["train_mse"]) for row in rows]
    val = [float(row["val_relmse"]) for row in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(6.2, 3.8))
    ax1.plot(steps, train_loss, label="train MSE", color="#2b6cb0")
    ax1.set_xlabel("step")
    ax1.set_ylabel("train MSE")
    ax2 = ax1.twinx()
    ax2.plot(steps, val, label="val relMSE", color="#b83280")
    ax2.set_ylabel("val relMSE")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_counterfactual_summary(cfg: Mapping[str, Any]) -> None:
    rows = read_csv_dicts(metrics_dir(cfg) / "counterfactual_metrics.csv")
    if not rows:
        return
    out = figures_main_dir(cfg) / "fig1_counterfactual_operator_consistency.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    cea = []
    beta = []
    model_names = sorted({metric_model_name(row) for row in rows})
    for model_name in model_names:
        for task in TASKS:
            families = task_families(task)
            for family in families:
                subset = [
                    row
                    for row in rows
                    if metric_model_name(row) == model_name
                    and row.get("task") == task
                    and row.get("family") == family
                    and row.get("effect_bin") == "high"
                ]
                if subset:
                    labels.append(f"{model_name}\n{task}\n{family}")
                    cea.append(median([float(row["CEA"]) for row in subset]))
                    beta.append(median([float(row["beta_T"]) for row in subset]))
    if not labels:
        return
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(labels) * 1.1), 4), sharex=True)
    x = np.arange(len(labels))
    axes[0].bar(x, cea, color="#2b6cb0")
    axes[0].axhline(0.8, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("median CEA, high effect")
    axes[1].bar(x, beta, color="#2f855a")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].axhspan(0.7, 1.3, color="#2f855a", alpha=0.12)
    axes[1].set_ylabel("median beta_T, high effect")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def specialisation_centered_key(row: Mapping[str, Any]) -> str:
    value = str(row.get("centered", "")).strip().lower()
    if value in {"true", "1", "yes"}:
        return "centered"
    if value in {"false", "0", "no"}:
        return "uncentered"
    return "none"


def specialisation_task(row: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    return str(row.get("task") or cfg.get("task") or "")


def specialisation_score_index(
    rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> dict[tuple[str, str, str, str, str, int, int], dict[str, float]]:
    index: dict[tuple[str, str, str, str, str, int, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        metric = str(row.get("metric", ""))
        if metric not in {"routing_follow", "routing_invariant", "transport_follow", "transport_invariant"}:
            continue
        try:
            layer = int(row["layer"])
            head = int(row["head"])
            value = metric_value(row)
        except Exception:
            continue
        if not math.isfinite(value):
            continue
        key = (
            specialisation_task(row, cfg),
            str(row.get("field", "")),
            str(row.get("intervention", "")),
            str(row.get("block", "")),
            specialisation_centered_key(row),
            layer,
            head,
        )
        index[key][metric] = value
    return index


def specialisation_grid(
    index: Mapping[tuple[str, str, str, str, str, int, int], Mapping[str, float]],
    *,
    task: str,
    field: str,
    intervention: str,
    centered: str,
    layers: Sequence[int],
    heads: Sequence[int],
    metric: str,
    block: str = "all",
    subtract_metric: Optional[str] = None,
) -> np.ndarray:
    grid = np.full((len(layers), len(heads)), np.nan, dtype=float)
    for layer_idx, layer in enumerate(layers):
        for head_idx, head in enumerate(heads):
            values = index.get((task, field, intervention, block, centered, layer, head), {})
            value = values.get(metric, float("nan"))
            if subtract_metric is not None and math.isfinite(value):
                value = value - values.get(subtract_metric, float("nan"))
            grid[layer_idx, head_idx] = value
    return grid


def annotate_specialisation_heatmap(ax: Any, values: np.ndarray) -> None:
    if values.size > 96:
        return
    for layer_idx in range(values.shape[0]):
        for head_idx in range(values.shape[1]):
            value = values[layer_idx, head_idx]
            if math.isfinite(float(value)):
                ax.text(head_idx, layer_idx, f"{value:.2f}", ha="center", va="center", fontsize=7)


def plot_specialisation_heatmap_page(
    pdf: PdfPages,
    index: Mapping[tuple[str, str, str, str, str, int, int], Mapping[str, float]],
    *,
    task: str,
    centered: str,
    layers: Sequence[int],
    heads: Sequence[int],
    mode: str,
) -> None:
    combos = (
        ("routing", "content"),
        ("routing", "structure"),
        ("transport", "content"),
        ("transport", "structure"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, sharey=True)
    fig.suptitle(f"{task}: {centered} specialisation {mode}", fontsize=13)
    cmap_name = "viridis" if mode == "follow scores" else "coolwarm"
    for ax, (field, intervention) in zip(axes.reshape(-1), combos):
        metric = f"{field}_follow"
        subtract = None if mode == "follow scores" else f"{field}_invariant"
        values = specialisation_grid(
            index,
            task=task,
            field=field,
            intervention=intervention,
            centered=centered,
            layers=layers,
            heads=heads,
            metric=metric,
            subtract_metric=subtract,
        )
        masked = np.ma.masked_invalid(values)
        if mode == "follow scores":
            vmin, vmax = 0.0, 1.0
        else:
            vmax = max(0.2, float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.2)
            vmin = -vmax
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#eeeeee")
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        annotate_specialisation_heatmap(ax, values)
        ax.set_title(f"{field} / {intervention}", fontsize=10)
        ax.set_xticks(np.arange(len(heads)))
        ax.set_xticklabels([str(head) for head in heads], fontsize=8)
        ax.set_yticks(np.arange(len(layers)))
        ax.set_yticklabels([str(layer) for layer in layers], fontsize=8)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def plot_specialisation_scatter_page(
    pdf: PdfPages,
    index: Mapping[tuple[str, str, str, str, str, int, int], Mapping[str, float]],
    *,
    task: str,
    centered: str,
    layers: Sequence[int],
    heads: Sequence[int],
) -> None:
    combos = (
        ("routing", "content"),
        ("routing", "structure"),
        ("transport", "content"),
        ("transport", "structure"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, sharey=True)
    fig.suptitle(f"{task}: {centered} follow vs invariant", fontsize=13)
    layer_colors = {layer: plt.get_cmap("tab10")(idx % 10) for idx, layer in enumerate(layers)}
    for ax, (field, intervention) in zip(axes.reshape(-1), combos):
        points = []
        for layer in layers:
            for head in heads:
                values = index.get((task, field, intervention, "all", centered, layer, head), {})
                follow = values.get(f"{field}_follow", float("nan"))
                invariant = values.get(f"{field}_invariant", float("nan"))
                if math.isfinite(follow) and math.isfinite(invariant):
                    points.append((invariant, follow, layer, head, follow - invariant))
        if points:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            colors = [layer_colors[point[2]] for point in points]
            ax.scatter(xs, ys, c=colors, s=48, edgecolors="black", linewidths=0.35, alpha=0.9)
            lower = min(-0.05, min(xs), min(ys)) - 0.03
            upper = max(1.0, max(xs), max(ys)) + 0.03
            ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=0.8)
            for invariant, follow, layer, head, _gap in sorted(points, key=lambda item: item[4], reverse=True)[:3]:
                ax.annotate(
                    f"L{layer}H{head}",
                    (invariant, follow),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
        ax.set_title(f"{field} / {intervention}", fontsize=10)
        ax.set_xlabel("invariant score")
        ax.set_ylabel("follow score")
        ax.grid(True, linewidth=0.4, alpha=0.35)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"layer {layer}", markerfacecolor=color, markeredgecolor="black", markersize=7)
        for layer, color in layer_colors.items()
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def plot_specialisation_atlas(cfg: Mapping[str, Any]) -> None:
    rows = read_csv_dicts(metrics_dir(cfg) / "specialisation_scores.csv")
    if not rows:
        return
    index = specialisation_score_index(rows, cfg)
    if not index:
        return
    out = figures_main_dir(cfg) / "fig2_specialisation_atlas.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = sorted({key[0] for key in index if key[0]})
    centered_values = [value for value in ("uncentered", "centered") if any(key[4] == value for key in index)]
    with PdfPages(out) as pdf:
        for task in tasks:
            task_keys = [key for key in index if key[0] == task]
            layers = sorted({key[5] for key in task_keys})
            heads = sorted({key[6] for key in task_keys})
            if not layers or not heads:
                continue
            for centered in centered_values:
                if not any(key[0] == task and key[4] == centered for key in index):
                    continue
                plot_specialisation_heatmap_page(
                    pdf,
                    index,
                    task=task,
                    centered=centered,
                    layers=layers,
                    heads=heads,
                    mode="follow scores",
                )
                plot_specialisation_heatmap_page(
                    pdf,
                    index,
                    task=task,
                    centered=centered,
                    layers=layers,
                    heads=heads,
                    mode="follow - invariant",
                )
                plot_specialisation_scatter_page(
                    pdf,
                    index,
                    task=task,
                    centered=centered,
                    layers=layers,
                    heads=heads,
                )


def plot_patching_summaries(cfg: Mapping[str, Any]) -> None:
    rows = read_csv_dicts(metrics_dir(cfg) / "patch_group_metrics.csv")
    if not rows:
        return
    out = figures_main_dir(cfg) / "fig3_interchange_mediation_matrix.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    values = []
    for task in TASKS:
        families = task_families(task)
        for family in families:
            subset = [row for row in rows if row.get("task") == task and row.get("family") == family and row.get("effect_bin") == "high"]
            if subset:
                labels.append(f"{task}\n{family}")
                values.append(median([float(row["TCM"]) for row in subset]))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 4))
    ax.bar(np.arange(len(values)), values, color="#c05621")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("median TCM, high effect")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    plot_cumulative_head_recovery(cfg, rows)
    plot_routing_transport_specificity(cfg, rows)


def is_true_csv(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def is_ranked_top_group(value: Any) -> bool:
    text = str(value).strip()
    if not text.startswith("top"):
        return False
    try:
        int(text[3:])
    except Exception:
        return False
    return True


def ranked_top_group_size(value: Any) -> int:
    text = str(value).strip()
    if not text.startswith("top"):
        return 1
    try:
        return int(text[3:])
    except Exception:
        return 1


def finite_metric_values(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[metric])
        except Exception:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def patch_group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    task: str,
    family: str,
    component: str,
    group_name: str,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("task") == task
        and row.get("family") == family
        and row.get("component") == component
        and row.get("group_name") == group_name
        and row.get("effect_bin") == "high"
    ]


def median_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    return median(finite_metric_values(rows, metric))


def patch_plot_tasks(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(row.get("task", "")) for row in rows if row.get("task")})


def plot_cumulative_head_recovery(cfg: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    tasks = patch_plot_tasks(rows)
    if not tasks:
        return
    out = figures_main_dir(cfg) / "fig4_cumulative_head_recovery.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    group_sizes = [1, 2, 4]
    with PdfPages(out) as pdf:
        for task in tasks:
            families = [family for family in (task_families(task)) if any(row.get("task") == task and row.get("family") == family for row in rows)]
            components = sorted({str(row.get("component")) for row in rows if row.get("task") == task and row.get("component")})
            if not families or not components:
                continue
            fig, axes = plt.subplots(
                len(families),
                len(components),
                figsize=(max(9.5, 3.2 * len(components)), max(3.4, 2.7 * len(families))),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            fig.suptitle(f"{task}: cumulative grouped patch recovery", fontsize=13)
            for row_idx, family in enumerate(families):
                for col_idx, component in enumerate(components):
                    ax = axes[row_idx][col_idx]
                    top = [
                        median_metric(
                            patch_group_rows(rows, task=task, family=family, component=component, group_name=f"top{size}"),
                            "TCM",
                        )
                        for size in group_sizes
                    ]
                    random_control = [
                        median_metric(
                            patch_group_rows(rows, task=task, family=family, component=component, group_name=f"random{size}"),
                            "TCM",
                        )
                        for size in group_sizes
                    ]
                    mismatch = [
                        median_metric(
                            patch_group_rows(rows, task=task, family=family, component=component, group_name=f"mismatched_top{size}"),
                            "TCM",
                        )
                        for size in group_sizes
                    ]
                    ax.plot(group_sizes, top, marker="o", linewidth=2.0, color="#2b6cb0", label="ranked top-k")
                    if any(math.isfinite(value) for value in random_control):
                        ax.plot(group_sizes, random_control, marker="s", linestyle="--", color="#718096", label="random control")
                    if any(math.isfinite(value) for value in mismatch):
                        ax.plot(group_sizes, mismatch, marker="^", linestyle=":", color="#c05621", label="mismatched control")
                    ax.axhline(0.0, color="black", linewidth=0.7)
                    ax.set_title(f"{family}\n{component}", fontsize=9)
                    ax.set_xticks(group_sizes)
                    ax.set_xlabel("patched heads")
                    ax.set_ylabel("median TCM")
                    ax.grid(True, linewidth=0.35, alpha=0.35)
            handles, labels = axes[0][0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="lower center", ncol=min(3, len(handles)), frameon=False)
            fig.tight_layout(rect=(0, 0.08, 1, 0.93))
            pdf.savefig(fig)
            plt.close(fig)


def plot_routing_transport_specificity(cfg: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    tasks = patch_plot_tasks(rows)
    out = figures_main_dir(cfg) / "fig5_routing_transport_specificity.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    family_palette = {
        family: plt.get_cmap("tab10")(idx % 10)
        for idx, family in enumerate(ALL_FAMILIES)
    }
    component_markers = {
        "attn_probs": "o",
        "message_pre_weight": "s",
        "resid_contribution": "^",
        "pair_state": "D",
    }
    pages_written = 0
    with PdfPages(out) as pdf:
        if not tasks:
            fig, ax = plt.subplots(figsize=(7, 3.8))
            ax.axis("off")
            ax.text(0.5, 0.55, "No patch rows found in patch_group_metrics.csv", ha="center", va="center", fontsize=12)
            ax.text(0.5, 0.42, str(metrics_dir(cfg) / "patch_group_metrics.csv"), ha="center", va="center", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
            pages_written += 1
        for task in tasks:
            plot_rows = [
                row
                for row in rows
                if row.get("task") == task
                and row.get("effect_bin") == "high"
                and is_ranked_top_group(row.get("group_name"))
                and not is_true_csv(row.get("is_control", ""))
            ]
            if not plot_rows:
                fig, ax = plt.subplots(figsize=(7, 3.8))
                ax.axis("off")
                group_counts: dict[str, int] = defaultdict(int)
                control_counts: dict[str, int] = defaultdict(int)
                for row in rows:
                    if row.get("task") != task or row.get("effect_bin") != "high":
                        continue
                    group_counts[str(row.get("group_name", ""))] += 1
                    control_counts[str(row.get("is_control", ""))] += 1
                groups = ", ".join(f"{name}:{count}" for name, count in sorted(group_counts.items())[:12])
                controls = ", ".join(f"{name}:{count}" for name, count in sorted(control_counts.items()))
                ax.text(0.5, 0.68, f"{task}: no non-control top-k rows for fig5", ha="center", va="center", fontsize=12)
                ax.text(0.5, 0.50, f"group_name counts: {groups or 'none'}", ha="center", va="center", fontsize=8)
                ax.text(0.5, 0.39, f"is_control counts: {controls or 'none'}", ha="center", va="center", fontsize=8)
                ax.text(0.5, 0.25, "Expected group_name values like top1, top2, top4 with is_control=False.", ha="center", va="center", fontsize=8)
                pdf.savefig(fig)
                plt.close(fig)
                pages_written += 1
                continue
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
            fig.suptitle(f"{task}: matched pathway specificity", fontsize=13)
            for ax, metric, wrong_metric, label in [
                (axes[0], "TCMA", "wrong_pathway_TCMA", "alignment"),
                (axes[1], "TCM", "wrong_pathway_TCM", "effect recovery"),
            ]:
                points = []
                for row in plot_rows:
                    try:
                        matched = float(row[metric])
                        wrong = float(row[wrong_metric])
                    except Exception:
                        continue
                    if not (math.isfinite(matched) and math.isfinite(wrong)):
                        continue
                    points.append((wrong, matched, row))
                if points:
                    xs = [point[0] for point in points]
                    ys = [point[1] for point in points]
                    lower = min(-0.1, min(xs), min(ys)) - 0.05
                    upper = max(1.0, max(xs), max(ys)) + 0.05
                    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=0.8)
                    for wrong, matched, row in points:
                        family = str(row.get("family", ""))
                        component = str(row.get("component", ""))
                        group_name = str(row.get("group_name", ""))
                        size = ranked_top_group_size(group_name)
                        ax.scatter(
                            wrong,
                            matched,
                            s=35 + 18 * size,
                            marker=component_markers.get(component, "o"),
                            color=family_palette.get(family, "#2b6cb0"),
                            edgecolors="black",
                            linewidths=0.35,
                            alpha=0.85,
                        )
                    for wrong, matched, row in sorted(points, key=lambda item: item[1] - item[0], reverse=True)[:5]:
                        ax.annotate(
                            f"{row.get('family')}\n{row.get('component')} {row.get('group_name')}",
                            (wrong, matched),
                            textcoords="offset points",
                            xytext=(4, 4),
                            fontsize=7,
                        )
                    ax.set_xlim(lower, upper)
                    ax.set_ylim(lower, upper)
                ax.set_title(label)
                ax.set_xlabel(f"wrong-pathway {metric}")
                ax.set_ylabel(f"matched-pathway {metric}")
                ax.grid(True, linewidth=0.35, alpha=0.35)
            family_handles = [
                plt.Line2D([0], [0], marker="o", color="w", label=family, markerfacecolor=family_palette.get(family, "#2b6cb0"), markeredgecolor="black", markersize=7)
                for family in sorted({str(row.get("family")) for row in plot_rows if row.get("family")})
            ]
            component_handles = [
                plt.Line2D([0], [0], marker=marker, color="black", label=component, linestyle="None", markersize=7)
                for component, marker in component_markers.items()
                if any(row.get("component") == component for row in plot_rows)
            ]
            handles = family_handles + component_handles
            if handles:
                fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False, fontsize=8)
            fig.tight_layout(rect=(0, 0.12, 1, 0.92))
            pdf.savefig(fig)
            plt.close(fig)
            pages_written += 1
    print(f"[plot-patching] wrote {out} pages={pages_written}", flush=True)


def plot_response_predictivity_summary(cfg: Mapping[str, Any], task: str) -> None:
    rows = read_csv_dicts(metrics_dir(cfg) / f"response_predictivity_cv_{task}.csv")
    if not rows:
        return
    out = figures_main_dir(cfg) / f"fig6_response_predictivity_{task}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    targets = [
        "teacher_pathway_norm",
        "student_teacher_pathway_projection",
        "student_delta_norm",
    ]
    feature_sets = ["intercept_only", "routing_scores", "transport_scores", "all_scores"]
    families = [family for family in (task_families(task)) if any(row.get("family") == family for row in rows)]
    if not families:
        return
    with PdfPages(out) as pdf:
        for target in targets:
            fig, axes = plt.subplots(
                1,
                len(families),
                figsize=(max(7.0, 4.0 * len(families)), 4.2),
                sharey=True,
                squeeze=False,
            )
            fig.suptitle(f"{task}: grouped-CV response predictivity ({target})", fontsize=12)
            for ax, family in zip(axes.reshape(-1), families):
                values = []
                labels = []
                colors = []
                matched = "transport_scores" if PATHWAY_BY_FAMILY[family] == "M" else "routing_scores"
                for feature_set in feature_sets:
                    subset = [
                        row
                        for row in rows
                        if row.get("family") == family
                        and row.get("target") == target
                        and row.get("feature_set") == feature_set
                    ]
                    if not subset:
                        continue
                    try:
                        value = float(subset[0]["r2_mean"])
                    except Exception:
                        value = float("nan")
                    values.append(value)
                    labels.append(feature_set.replace("_scores", "").replace("_", "\n"))
                    colors.append("#2b6cb0" if feature_set == matched else ("#718096" if feature_set == "intercept_only" else "#805ad5"))
                x = np.arange(len(values))
                ax.bar(x, values, color=colors)
                ax.axhline(0.0, color="black", linewidth=0.8)
                ax.set_title(family)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_ylabel("mean held-out R2")
                ax.grid(axis="y", linewidth=0.35, alpha=0.35)
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            pdf.savefig(fig)
            plt.close(fig)


def response_cv_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    target: str,
    feature_set: str,
) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if (
            row.get("family") == family
            and row.get("target") == target
            and row.get("feature_set") == feature_set
        ):
            return row
    return None


def response_cv_float(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    target: str,
    feature_set: str,
    key: str,
) -> float:
    row = response_cv_row(rows, family=family, target=target, feature_set=feature_set)
    if row is None:
        return float("nan")
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def response_predictivity_contrast_rows(
    rows: Sequence[Mapping[str, Any]],
    task: str,
) -> list[dict[str, Any]]:
    targets = ["teacher_pathway_norm", "student_teacher_pathway_projection"]
    families = [
        family
        for family in (task_families(task))
        if any(row.get("family") == family for row in rows)
    ]
    out = []
    for family in families:
        matched = "transport_scores" if PATHWAY_BY_FAMILY[family] == "M" else "routing_scores"
        mismatched = "routing_scores" if matched == "transport_scores" else "transport_scores"
        for target in targets:
            intercept_r2 = response_cv_float(rows, family=family, target=target, feature_set="intercept_only", key="r2_mean")
            matched_r2 = response_cv_float(rows, family=family, target=target, feature_set=matched, key="r2_mean")
            mismatched_r2 = response_cv_float(rows, family=family, target=target, feature_set=mismatched, key="r2_mean")
            all_r2 = response_cv_float(rows, family=family, target=target, feature_set="all_scores", key="r2_mean")
            out.append(
                {
                    "task": task,
                    "family": family,
                    "pathway_target": PATHWAY_BY_FAMILY[family],
                    "target": target,
                    "matched_feature_set": matched,
                    "mismatched_feature_set": mismatched,
                    "intercept_r2_mean": intercept_r2,
                    "matched_r2_mean": matched_r2,
                    "mismatched_r2_mean": mismatched_r2,
                    "all_r2_mean": all_r2,
                    "matched_minus_intercept_r2": matched_r2 - intercept_r2,
                    "matched_minus_mismatched_r2": matched_r2 - mismatched_r2,
                    "all_minus_matched_r2": all_r2 - matched_r2,
                    "matched_pearson_oof": response_cv_float(rows, family=family, target=target, feature_set=matched, key="pearson_oof"),
                    "matched_spearman_oof": response_cv_float(rows, family=family, target=target, feature_set=matched, key="spearman_oof"),
                }
            )
    return out


def plot_response_predictivity_contrasts(
    cfg: Mapping[str, Any],
    task: str,
    contrast_rows: Sequence[Mapping[str, Any]],
) -> None:
    if not contrast_rows:
        return
    out = figures_main_dir(cfg) / f"fig7_response_predictivity_teacher_student_{task}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    targets = ["teacher_pathway_norm", "student_teacher_pathway_projection"]
    target_labels = {
        "teacher_pathway_norm": "teacher effect",
        "student_teacher_pathway_projection": "student teacher-aligned effect",
    }
    families = [
        family
        for family in (task_families(task))
        if any(row.get("family") == family for row in contrast_rows)
    ]
    fig, axes = plt.subplots(
        1,
        len(families),
        figsize=(max(7.5, 4.2 * len(families)), 4.4),
        sharey=True,
        squeeze=False,
    )
    fig.suptitle(f"{task}: matched score predictivity for teacher vs student targets", fontsize=12)
    for ax, family in zip(axes.reshape(-1), families):
        family_rows = [row for row in contrast_rows if row.get("family") == family]
        x = np.arange(len(targets), dtype=float)
        matched_delta = []
        specificity_delta = []
        all_delta = []
        for target in targets:
            row = next((item for item in family_rows if item.get("target") == target), None)
            if row is None:
                matched_delta.append(float("nan"))
                specificity_delta.append(float("nan"))
                all_delta.append(float("nan"))
                continue
            matched_delta.append(float(row["matched_minus_intercept_r2"]))
            specificity_delta.append(float(row["matched_minus_mismatched_r2"]))
            all_delta.append(float(row["all_minus_matched_r2"]))
        width = 0.24
        ax.bar(x - width, matched_delta, width=width, label="matched - intercept", color="#2b6cb0")
        ax.bar(x, specificity_delta, width=width, label="matched - mismatched", color="#2f855a")
        ax.bar(x + width, all_delta, width=width, label="all - matched", color="#805ad5")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(family)
        ax.set_xticks(x)
        ax.set_xticklabels([target_labels[target] for target in targets], rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("held-out R2 difference")
        ax.grid(axis="y", linewidth=0.35, alpha=0.35)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(handles)), frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 0.91))
    fig.savefig(out)
    plt.close(fig)


def write_response_predictivity_contrasts(cfg: Mapping[str, Any], task: str) -> Path:
    rows = read_csv_dicts(metrics_dir(cfg) / f"response_predictivity_cv_{task}.csv")
    out = metrics_dir(cfg) / f"response_predictivity_contrasts_{task}.csv"
    contrast_rows = response_predictivity_contrast_rows(rows, task)
    write_csv(out, contrast_rows)
    plot_response_predictivity_contrasts(cfg, task, contrast_rows)
    return out


def run_sequence(
    cfg: Mapping[str, Any],
    task: str,
    *,
    device_name: str,
    backend: Optional[str],
    skip_training: bool,
    force_data: bool,
    force_interventions: bool,
) -> None:
    sequence_start = time.time()
    print(f"[sequence] start task={task} skip_training={skip_training}", flush=True)
    print(f"[sequence] stage=cache-data task={task}", flush=True)
    cache_data(cfg, task, force=force_data)
    if not skip_training:
        print(f"[sequence] stage=train task={task}", flush=True)
        train(cfg, task, device_name=device_name, backend=backend)
    print(f"[sequence] stage=build-interventions kind=cf_eval task={task}", flush=True)
    build_interventions(cfg, task, "cf_eval", force=force_interventions)
    print(f"[sequence] stage=build-interventions kind=patch_eval task={task}", flush=True)
    build_interventions(cfg, task, "patch_eval", force=force_interventions)
    print(f"[sequence] stage=evaluate-counterfactuals task={task}", flush=True)
    evaluate_counterfactuals(cfg, task, device_name=device_name, backend=backend)
    if (backend or cfg["model"].get("backend", "official")) == "official":
        print(f"[sequence] stage=run-specialisation task={task}", flush=True)
        run_specialisation(cfg, task, device_name=device_name, backend=backend)
        if bool(cfg["patching"].get("allow_failed_gate_patching", False)) or task_clean_gate(cfg, task):
            print(f"[sequence] stage=run-patching task={task}", flush=True)
            run_patching(cfg, task, device_name=device_name, backend=backend)
        else:
            print(f"[patching] skipped for {task}: clean gate failed", flush=True)
    print(f"[sequence] done task={task} elapsed={time.time() - sequence_start:.1f}s", flush=True)


def write_default_configs(output_dir: Path) -> None:
    for task in TASKS:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["task"] = task
        cfg["response_predictivity"]["families"] = list(task_families(task))
        cfg["run"] = {
            "name": f"cfim_grit_{task}_seed1001",
            "hardware": "single_a100_80gb",
            "model_note": "2-layer official-backed GRIT-RRWP with continuous node-regression head",
        }
        write_yaml(output_dir / f"grit_{task}.yaml", cfg)
        write_yaml(output_dir / f"gcn_plus_{task}.yaml", gnnplus_default_config(task, layer_type="gcn"))


def gnnplus_default_config(task: str, *, layer_type: str = "gcn") -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["task"] = task
    cfg["model"] = deep_update(
        cfg["model"],
        {
            "name": "gcn_plus" if layer_type == "gcn" else f"{layer_type}_plus",
            "backend": "official_gnnplus",
            "num_layers": 2,
            "hidden_dim": 192,
            "ffn_hidden_dim": 384,
            "input_dropout": 0.0,
            "dropout": 0.1,
            "residual_dropout": 0.1,
            "ffn_dropout": 0.1,
            "activation": "relu",
            "residual": True,
            "ffn": True,
            "use_rwse": True,
            "use_degree_features": True,
            "use_edge_type": False,
            "output_head": "node_regression_linear",
            "parameter_match": {
                "reference_model": "grit",
                "reference_layers": 2,
                "reference_hidden_dim": 128,
                "note": "hidden_dim=192 is chosen to approximately match the parameter budget of the 2-layer GRIT-128 teacher-student baseline.",
            },
            "gnnplus": {
                "layer_type": layer_type,
                "official_name": "GCN+" if layer_type == "gcn" else f"{layer_type.upper()}+",
            },
            "official_source": {
                "repo": "https://github.com/LUOyk1999/GNNPlus",
                "checked_commit": "0e02ad9acc2f1e54b5ad71c051bf5dfb1fcb4f28",
                "paper": "https://arxiv.org/abs/2502.09263",
                "config_family": "GNN+",
            },
        },
    )
    cfg["training"] = deep_update(
        cfg["training"],
        {
            "batch_size_graphs": 512,
            "eval_batch_size_graphs": 4096,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "max_steps": 5000,
            "warmup_steps": 250,
            "eval_every_steps": 250,
            "checkpoint_every_steps": 0,
            "early_stop_val_relmse": 0.005,
            "early_stop_min_steps": 1500,
            "early_stop_patience_evals": 4,
            "use_cached_train_data": True,
            "train_cache_graphs": 32768,
            "progress_every_steps": 25,
        },
    )
    cfg["response_predictivity"]["families"] = list(task_families(task))
    cfg["run"] = {
        "name": f"cfim_gcn_plus_{task}_seed1001",
        "hardware": "single_a100_80gb",
        "model_note": "2-layer official GCN+ adapter using the GNNPlus repository layers, RWSE/degree inputs, residuals, BatchNorm, dropout, and FFN.",
    }
    return cfg


def print_hpc_commands(config_dir: Path) -> None:
    for task in TASKS:
        cfg_path = config_dir / f"grit_{task}.yaml"
        print(f"# {task}")
        print(f"python -m graph_specialisation_metrics.counterfactual_interchange_mediation run-sequence --config {cfg_path} --task {task} --device cuda --backend official")
        gnn_cfg_path = config_dir / f"gcn_plus_{task}.yaml"
        print(f"python -m graph_specialisation_metrics.counterfactual_interchange_mediation train --config {gnn_cfg_path} --task {task} --device cuda --backend official_gnnplus")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=Path)
        p.add_argument("--task", choices=TASKS)
        p.add_argument("--fast-dev-run", action="store_true")

    p = sub.add_parser("write-configs")
    p.add_argument("--output-dir", type=Path, default=Path("experiments/synthetic/cfim/configs"))

    p = sub.add_parser("print-hpc-commands")
    p.add_argument("--config-dir", type=Path, default=Path("experiments/synthetic/cfim/configs"))

    p = sub.add_parser("cache-data")
    common(p)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("train")
    common(p)
    p.add_argument("--device", default="auto")
    p.add_argument("--backend", choices=MODEL_BACKENDS)

    p = sub.add_parser("build-interventions")
    common(p)
    p.add_argument("--kind", choices=("cf_eval", "patch_eval"), required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("evaluate-counterfactuals")
    common(p)
    p.add_argument("--device", default="auto")
    p.add_argument("--backend", choices=MODEL_BACKENDS)
    p.add_argument("--checkpoint", type=Path)

    p = sub.add_parser("run-specialisation")
    common(p)
    p.add_argument("--device", default="auto")
    p.add_argument("--backend", choices=MODEL_BACKENDS)
    p.add_argument("--checkpoint", type=Path)

    p = sub.add_parser("plot-specialisation")
    common(p)

    p = sub.add_parser("run-patching")
    common(p)
    p.add_argument("--device", default="auto")
    p.add_argument("--backend", choices=MODEL_BACKENDS)
    p.add_argument("--checkpoint", type=Path)

    p = sub.add_parser("plot-patching")
    common(p)

    p = sub.add_parser("run-response-predictivity")
    common(p)
    p.add_argument("--device", default="auto")
    p.add_argument("--backend", choices=MODEL_BACKENDS)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--families", default="")
    p.add_argument("--intervention-kind", choices=("cf_eval", "patch_eval"))
    p.add_argument("--max-interventions-per-family", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--folds", type=int)
    p.add_argument("--ridge-alpha", type=float)
    p.add_argument("--centered", action="store_true")

    p = sub.add_parser("run-payload-gating-controls")
    common(p)
    p.add_argument("--family")
    p.add_argument("--intervention-kind", choices=("cf_eval", "patch_eval"), default="cf_eval")
    p.add_argument("--folds", type=int)
    p.add_argument("--ridge-alpha", type=float)

    p = sub.add_parser("plot-response-predictivity")
    common(p)

    p = sub.add_parser("build-functional-swaps")
    common(p)
    p.add_argument("--family")
    p.add_argument("--num-graphs", type=int, default=200)
    p.add_argument("--swaps-per-stratum", type=int, default=50)
    p.add_argument("--gnn-depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=9101)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("run-functional-responses")
    common(p)
    p.add_argument("--swaps-path", type=Path)
    p.add_argument("--seed", type=int, default=9101)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--gnn-depth", type=int, default=2)
    p.add_argument("--grit-config", type=Path)
    p.add_argument("--gcn-plus-config", type=Path)
    p.add_argument("--grit-checkpoint", type=Path)
    p.add_argument("--gcn-plus-checkpoint", type=Path)

    p = sub.add_parser("summarise-functional-responses")
    common(p)
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--seed", type=int, default=9101)
    p.add_argument("--gnn-depth", type=int, default=2)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=9201)

    p = sub.add_parser("plot-functional-responses")
    common(p)
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--seed", type=int, default=9101)

    p = sub.add_parser("run-functional-stage1")
    common(p)
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--family")
    p.add_argument("--num-graphs", type=int, default=200)
    p.add_argument("--swaps-per-stratum", type=int, default=50)
    p.add_argument("--gnn-depth", type=int, default=2)
    p.add_argument("--seed", type=int, default=9101)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--grit-config", type=Path)
    p.add_argument("--gcn-plus-config", type=Path)
    p.add_argument("--grit-checkpoint", type=Path)
    p.add_argument("--gcn-plus-checkpoint", type=Path)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=9201)
    p.add_argument("--force-swaps", action="store_true")

    p = sub.add_parser("build-functional-node-swaps")
    common(p)
    p.add_argument("--family")
    p.add_argument("--num-graphs", type=int, default=200)
    p.add_argument("--swaps-per-graph", type=int, default=200)
    p.add_argument("--seed", type=int, default=9301)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("run-functional-node-responses")
    common(p)
    p.add_argument("--swaps-path", type=Path)
    p.add_argument("--seed", type=int, default=9301)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--gnn-depth", type=int, default=2)
    p.add_argument("--grit-config", type=Path)
    p.add_argument("--gcn-plus-config", type=Path)
    p.add_argument("--grit-checkpoint", type=Path)
    p.add_argument("--gcn-plus-checkpoint", type=Path)
    p.add_argument("--no-store-node-rows", action="store_true")
    p.add_argument("--leakage-tol", type=float, default=1.0e-5)
    p.add_argument("--fail-on-gcn-leakage", action="store_true")

    p = sub.add_parser("summarise-functional-node-responses")
    common(p)
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--seed", type=int, default=9301)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=9401)

    p = sub.add_parser("plot-functional-node-responses")
    common(p)
    p.add_argument("--all-tasks", action="store_true")

    p = sub.add_parser("run-functional-stage1-node")
    common(p)
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--family")
    p.add_argument("--num-graphs", type=int, default=200)
    p.add_argument("--swaps-per-graph", type=int, default=200)
    p.add_argument("--seed", type=int, default=9301)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--gnn-depth", type=int, default=2)
    p.add_argument("--grit-config", type=Path)
    p.add_argument("--gcn-plus-config", type=Path)
    p.add_argument("--grit-checkpoint", type=Path)
    p.add_argument("--gcn-plus-checkpoint", type=Path)
    p.add_argument("--no-store-node-rows", action="store_true")
    p.add_argument("--leakage-tol", type=float, default=1.0e-5)
    p.add_argument("--fail-on-gcn-leakage", action="store_true")
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=9401)
    p.add_argument("--force-swaps", action="store_true")

    p = sub.add_parser("summarise-functional-validity-gate")
    common(p)
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--e1g-seed", type=int, default=9101)
    p.add_argument("--e1n-seed", type=int, default=9301)
    p.add_argument("--gnn-depth", type=int, default=2)

    p = sub.add_parser("plot-functional-validity-gate")
    common(p)
    p.add_argument("--all-tasks", action="store_true")

    p = sub.add_parser("run-functional-q1-gates")
    common(p)
    p.add_argument("--swaps-path", type=Path)
    p.add_argument("--response-path", type=Path)
    p.add_argument("--seed", type=int, default=9501)
    p.add_argument("--e1g-seed", type=int, default=9101)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--gnn-depth", type=int, default=2)
    p.add_argument("--grit-config", type=Path)
    p.add_argument("--grit-checkpoint", type=Path)
    p.add_argument("--store-head-gates", action="store_true")

    p = sub.add_parser("summarise-functional-q1-gates")
    common(p)
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--seed", type=int, default=9501)
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=9601)
    p.add_argument("--min-swaps-per-bin", type=int, default=100)
    p.add_argument("--min-swaps-per-graph", type=int, default=20)

    p = sub.add_parser("plot-functional-q1-gates")
    common(p)
    p.add_argument("--all-tasks", action="store_true")

    p = sub.add_parser("run-functional-q1-stage")
    common(p)
    p.add_argument("--all-tasks", action="store_true")
    p.add_argument("--seed", type=int, default=9501)
    p.add_argument("--e1g-seed", type=int, default=9101)
    p.add_argument("--e1n-seed", type=int, default=9301)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--gnn-depth", type=int, default=2)
    p.add_argument("--grit-config", type=Path)
    p.add_argument("--grit-checkpoint", type=Path)
    p.add_argument("--store-head-gates", action="store_true")
    p.add_argument("--bootstrap-resamples", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=9601)
    p.add_argument("--min-swaps-per-bin", type=int, default=100)
    p.add_argument("--min-swaps-per-graph", type=int, default=20)

    p = sub.add_parser("run-sequence")
    common(p)
    p.add_argument("--device", default="auto")
    p.add_argument("--backend", choices=MODEL_BACKENDS)
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--force-data", action="store_true")
    p.add_argument("--force-interventions", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "write-configs":
        write_default_configs(args.output_dir)
        return 0
    if args.command == "print-hpc-commands":
        print_hpc_commands(args.config_dir)
        return 0
    functional_all_task_commands = {
        "summarise-functional-responses",
        "plot-functional-responses",
        "run-functional-stage1",
        "summarise-functional-node-responses",
        "plot-functional-node-responses",
        "run-functional-stage1-node",
        "summarise-functional-validity-gate",
        "plot-functional-validity-gate",
        "summarise-functional-q1-gates",
        "plot-functional-q1-gates",
        "run-functional-q1-stage",
    }
    load_task = args.task
    if load_task is None and args.command in functional_all_task_commands and args.config is None:
        load_task = "ppr_diffusion"
    cfg = load_config(args.config, task=load_task, fast_dev_run=bool(getattr(args, "fast_dev_run", False)))
    task = str(cfg["task"])
    if args.command == "cache-data":
        cache_data(cfg, task, force=args.force)
    elif args.command == "train":
        train(cfg, task, device_name=args.device, backend=args.backend)
    elif args.command == "build-interventions":
        cache_data(cfg, task, force=False)
        build_interventions(cfg, task, args.kind, force=args.force)
    elif args.command == "evaluate-counterfactuals":
        evaluate_counterfactuals(cfg, task, device_name=args.device, checkpoint=args.checkpoint, backend=args.backend)
    elif args.command == "run-specialisation":
        run_specialisation(cfg, task, device_name=args.device, checkpoint=args.checkpoint, backend=args.backend)
    elif args.command == "plot-specialisation":
        plot_specialisation_atlas(cfg)
    elif args.command == "run-patching":
        run_patching(cfg, task, device_name=args.device, checkpoint=args.checkpoint, backend=args.backend)
    elif args.command == "plot-patching":
        plot_patching_summaries(cfg)
    elif args.command == "run-response-predictivity":
        families = parse_csv_tuple(args.families) if args.families else None
        run_response_predictivity(
            cfg,
            task,
            device_name=args.device,
            checkpoint=args.checkpoint,
            backend=args.backend,
            families=families,
            kind=args.intervention_kind,
            max_interventions_per_family=args.max_interventions_per_family,
            batch_size=args.batch_size,
            folds=args.folds,
            ridge_alpha=args.ridge_alpha,
            centered=True if args.centered else None,
        )
    elif args.command == "plot-response-predictivity":
        plot_response_predictivity_summary(cfg, task)
        contrast_path = write_response_predictivity_contrasts(cfg, task)
        print(f"[plot-response-predictivity] wrote contrasts={contrast_path}", flush=True)
    elif args.command == "build-functional-swaps":
        build_functional_swaps(
            cfg,
            task,
            family=args.family,
            num_graphs=4 if args.fast_dev_run and args.num_graphs == 200 else args.num_graphs,
            swaps_per_stratum=2 if args.fast_dev_run and args.swaps_per_stratum == 50 else args.swaps_per_stratum,
            gnn_depth=args.gnn_depth,
            seed=args.seed,
            force=args.force,
        )
    elif args.command == "run-functional-responses":
        run_functional_responses(
            cfg,
            task,
            swaps_path=args.swaps_path,
            seed=args.seed,
            device_name=args.device,
            batch_size=args.batch_size,
            gnn_depth=args.gnn_depth,
            grit_config=args.grit_config,
            gcn_plus_config=args.gcn_plus_config,
            grit_checkpoint=args.grit_checkpoint,
            gcn_plus_checkpoint=args.gcn_plus_checkpoint,
            fast_dev_run=bool(args.fast_dev_run),
        )
    elif args.command == "summarise-functional-responses":
        tasks = TASKS if args.all_tasks or args.task is None else (task,)
        summarise_functional_responses(
            cfg,
            tasks=tasks,
            seed=args.seed,
            gnn_depth=args.gnn_depth,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
    elif args.command == "plot-functional-responses":
        tasks = TASKS if args.all_tasks or args.task is None else (task,)
        plot_functional_responses(cfg, tasks=tasks, seed=args.seed)
    elif args.command == "run-functional-stage1":
        tasks = TASKS if args.all_tasks or args.task is None else (task,)
        for task_name in tasks:
            task_cfg = load_config(args.config, task=task_name, fast_dev_run=bool(args.fast_dev_run))
            build_functional_swaps(
                task_cfg,
                task_name,
                family=args.family,
                num_graphs=4 if args.fast_dev_run and args.num_graphs == 200 else args.num_graphs,
                swaps_per_stratum=2 if args.fast_dev_run and args.swaps_per_stratum == 50 else args.swaps_per_stratum,
                gnn_depth=args.gnn_depth,
                seed=args.seed,
                force=args.force_swaps,
            )
            run_functional_responses(
                task_cfg,
                task_name,
                seed=args.seed,
                device_name=args.device,
                batch_size=args.batch_size,
                gnn_depth=args.gnn_depth,
                grit_config=args.grit_config,
                gcn_plus_config=args.gcn_plus_config,
                grit_checkpoint=args.grit_checkpoint,
                gcn_plus_checkpoint=args.gcn_plus_checkpoint,
                fast_dev_run=bool(args.fast_dev_run),
            )
        summarise_functional_responses(
            cfg,
            tasks=tasks,
            seed=args.seed,
            gnn_depth=args.gnn_depth,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        plot_functional_responses(cfg, tasks=tasks, seed=args.seed)
    elif args.command == "build-functional-node-swaps":
        build_functional_node_swaps(
            cfg,
            task,
            family=args.family,
            num_graphs=4 if args.fast_dev_run and args.num_graphs == 200 else args.num_graphs,
            swaps_per_graph=4 if args.fast_dev_run and args.swaps_per_graph == 200 else args.swaps_per_graph,
            seed=args.seed,
            force=args.force,
        )
    elif args.command == "run-functional-node-responses":
        run_functional_node_responses(
            cfg,
            task,
            swaps_path=args.swaps_path,
            seed=args.seed,
            device_name=args.device,
            batch_size=args.batch_size,
            gnn_depth=args.gnn_depth,
            grit_config=args.grit_config,
            gcn_plus_config=args.gcn_plus_config,
            grit_checkpoint=args.grit_checkpoint,
            gcn_plus_checkpoint=args.gcn_plus_checkpoint,
            store_node_rows=not args.no_store_node_rows,
            leakage_tol=args.leakage_tol,
            fail_on_gcn_leakage=args.fail_on_gcn_leakage,
            fast_dev_run=bool(args.fast_dev_run),
        )
    elif args.command == "summarise-functional-node-responses":
        tasks = TASKS if args.all_tasks or args.task is None else (task,)
        summarise_functional_node_responses(
            cfg,
            tasks=tasks,
            seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
    elif args.command == "plot-functional-node-responses":
        tasks = TASKS if args.all_tasks or args.task is None else (task,)
        plot_functional_node_responses(cfg, tasks=tasks)
    elif args.command == "run-functional-stage1-node":
        tasks = TASKS if args.all_tasks or args.task is None else (task,)
        for task_name in tasks:
            task_cfg = load_config(args.config, task=task_name, fast_dev_run=bool(args.fast_dev_run))
            build_functional_node_swaps(
                task_cfg,
                task_name,
                family=args.family,
                num_graphs=4 if args.fast_dev_run and args.num_graphs == 200 else args.num_graphs,
                swaps_per_graph=4 if args.fast_dev_run and args.swaps_per_graph == 200 else args.swaps_per_graph,
                seed=args.seed,
                force=args.force_swaps,
            )
            run_functional_node_responses(
                task_cfg,
                task_name,
                seed=args.seed,
                device_name=args.device,
                batch_size=args.batch_size,
                gnn_depth=args.gnn_depth,
                grit_config=args.grit_config,
                gcn_plus_config=args.gcn_plus_config,
                grit_checkpoint=args.grit_checkpoint,
                gcn_plus_checkpoint=args.gcn_plus_checkpoint,
                store_node_rows=not args.no_store_node_rows,
                leakage_tol=args.leakage_tol,
                fail_on_gcn_leakage=args.fail_on_gcn_leakage,
                fast_dev_run=bool(args.fast_dev_run),
            )
        summarise_functional_node_responses(
            cfg,
            tasks=tasks,
            seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        plot_functional_node_responses(cfg, tasks=tasks)
    elif args.command == "summarise-functional-validity-gate":
        tasks = TASKS if args.all_tasks or args.task is None else (task,)
        summarise_functional_validity_gate(
            cfg,
            tasks=tasks,
            e1g_seed=args.e1g_seed,
            e1n_seed=args.e1n_seed,
            gnn_depth=args.gnn_depth,
        )
    elif args.command == "plot-functional-validity-gate":
        tasks = TASKS if args.all_tasks or args.task is None else (task,)
        plot_functional_validity_gate(cfg, tasks=tasks)
    elif args.command == "run-functional-q1-gates":
        run_functional_q1_gates(
            cfg,
            task,
            swaps_path=args.swaps_path,
            response_path=args.response_path,
            seed=args.seed,
            e1g_seed=args.e1g_seed,
            device_name=args.device,
            batch_size=args.batch_size,
            gnn_depth=args.gnn_depth,
            grit_config=args.grit_config,
            grit_checkpoint=args.grit_checkpoint,
            store_head_gates=args.store_head_gates,
            fast_dev_run=bool(args.fast_dev_run),
        )
    elif args.command == "summarise-functional-q1-gates":
        tasks = TASKS if args.all_tasks or args.task is None else (task,)
        summarise_functional_q1_gates(
            cfg,
            tasks=tasks,
            seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            min_swaps_per_bin=args.min_swaps_per_bin,
            min_swaps_per_graph=args.min_swaps_per_graph,
        )
    elif args.command == "plot-functional-q1-gates":
        tasks = TASKS if args.all_tasks or args.task is None else (task,)
        plot_functional_q1_gates(cfg, tasks=tasks)
    elif args.command == "run-functional-q1-stage":
        tasks = TASKS if args.all_tasks or args.task is None else (task,)
        summarise_functional_validity_gate(
            cfg,
            tasks=tasks,
            e1g_seed=args.e1g_seed,
            e1n_seed=args.e1n_seed,
            gnn_depth=args.gnn_depth,
        )
        plot_functional_validity_gate(cfg, tasks=tasks)
        for task_name in tasks:
            task_cfg = load_config(args.config, task=task_name, fast_dev_run=bool(args.fast_dev_run))
            run_functional_q1_gates(
                task_cfg,
                task_name,
                seed=args.seed,
                e1g_seed=args.e1g_seed,
                device_name=args.device,
                batch_size=args.batch_size,
                gnn_depth=args.gnn_depth,
                grit_config=args.grit_config,
                grit_checkpoint=args.grit_checkpoint,
                store_head_gates=args.store_head_gates,
                fast_dev_run=bool(args.fast_dev_run),
            )
        summarise_functional_q1_gates(
            cfg,
            tasks=tasks,
            seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
            min_swaps_per_bin=args.min_swaps_per_bin,
            min_swaps_per_graph=args.min_swaps_per_graph,
        )
        plot_functional_q1_gates(cfg, tasks=tasks)
    elif args.command == "run-payload-gating-controls":
        run_payload_gating_controls(
            cfg,
            task,
            family=args.family,
            kind=args.intervention_kind,
            folds=args.folds,
            ridge_alpha=args.ridge_alpha,
        )
    elif args.command == "run-sequence":
        run_sequence(
            cfg,
            task,
            device_name=args.device,
            backend=args.backend,
            skip_training=args.skip_training,
            force_data=args.force_data,
            force_interventions=args.force_interventions,
        )
    else:  # pragma: no cover
        raise ValueError(args.command)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
