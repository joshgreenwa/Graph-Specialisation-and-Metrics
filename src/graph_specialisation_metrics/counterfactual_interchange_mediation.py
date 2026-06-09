"""Counterfactual interchange-mediation experiments for GRIT teacher-student tasks.

This module implements the first CFIM experiment sequence described in
``counterfactual_interchange_mediation_plan.md``:

* deterministic graph/data caches for ``ppr_diffusion`` and
  ``nearest_anchor_voronoi``;
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

import matplotlib.pyplot as plt
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


TASKS = ("ppr_diffusion", "nearest_anchor_voronoi")
PPR_FAMILIES = ("ppr_payload_swap", "ppr_struct_swap")
VORONOI_FAMILIES = (
    "voronoi_anchor_marker_swap",
    "voronoi_struct_swap",
    "voronoi_payload_swap",
)
ALL_FAMILIES = PPR_FAMILIES + VORONOI_FAMILIES
PATHWAY_BY_FAMILY = {
    "ppr_payload_swap": "M",
    "ppr_struct_swap": "K",
    "voronoi_anchor_marker_swap": "K",
    "voronoi_struct_swap": "K",
    "voronoi_payload_swap": "M",
}
TEACHER_SEEDS = {"ppr_diffusion": 314159, "nearest_anchor_voronoi": 271828}
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
        "target_dim": 16,
    },
    "teachers": {
        "ppr": {"alpha": 0.15, "truncation": 8, "teacher_seed": 314159},
        "voronoi": {"n_anchors": 4, "min_anchor_distance": 2, "teacher_seed": 271828},
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
        "eval_batch_size_graphs": 256,
        "max_steps": 30000,
        "warmup_steps": 1000,
        "lr_schedule": "cosine_decay_to_10_percent",
        "gradient_clip_norm": 1.0,
        "loss": "mse_node_mean",
        "eval_every_steps": 500,
        "checkpoint_every_steps": 2500,
        "select_checkpoint": "lowest_val_relmse",
        "early_stop_val_relmse": 0.01,
        "early_stop_min_steps": 3000,
        "early_stop_patience_evals": 3,
        "mixed_precision": False,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "dataloader_workers": 4,
        "pin_memory": True,
        "allow_tf32": False,
    },
    "seeds": {
        "graph_train_seed": 1729,
        "graph_val_seed": 1730,
        "graph_test_id_seed": 1731,
        "graph_test_ood_seed": 1732,
        "graph_patch_eval_seed": 1733,
        "teacher_ppr_seed": 314159,
        "teacher_voronoi_seed": 271828,
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
        "interventions_per_graph_per_family": 16,
        "bin_allocation": {"null": 2, "low": 2, "medium": 4, "high": 8},
    },
    "patch_eval_budget": {
        "graphs_per_task": 256,
        "interventions_per_graph_per_family": 16,
        "bin_allocation": {"null": 2, "low": 2, "medium": 4, "high": 8},
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
        "num_permutations": 128,
        "batch_size_graphs": 16,
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
        "max_interventions_per_family": 256,
        "components": ["attn_probs", "message_pre_weight", "resid_contribution", "pair_state"],
        "group_sizes": [1, 2, 4],
        "random_groups_per_size": 100,
        "batch_size_graphs": 1,
        "allow_failed_gate_patching": False,
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
        cfg["training"]["max_steps"] = 2
        cfg["training"]["warmup_steps"] = 1
        cfg["training"]["eval_every_steps"] = 1
        cfg["training"]["checkpoint_every_steps"] = 2
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
    return artifact_root(cfg) / "checkpoints" / "grit" / task / f"seed_{seed}"


def intervention_dir(cfg: Mapping[str, Any], task: str) -> Path:
    return artifact_root(cfg) / "interventions" / task


def metrics_dir(cfg: Mapping[str, Any]) -> Path:
    return artifact_root(cfg) / "metrics"


def figures_main_dir(cfg: Mapping[str, Any]) -> Path:
    return artifact_root(cfg) / "figures" / "main"


def figures_appendix_dir(cfg: Mapping[str, Any]) -> Path:
    return artifact_root(cfg) / "figures" / "appendix"


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
    if task == "ppr_diffusion":
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
    if task == "ppr_diffusion":
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
    if family in {"ppr_payload_swap", "voronoi_payload_swap"}:
        payload = base["payload"].clone()
        payload[[int(u), int(v)]] = payload[[int(v), int(u)]]
        return clone_record_with(base, cfg, graph_id=graph_id, payload=payload)
    if family in {"ppr_struct_swap", "voronoi_struct_swap"}:
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
            f"{package_hint}; for GRIT set GRIT_ROOT and run the repository preflight."
        ) from exc


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
    raise ValueError(task)


def build_model(cfg: Mapping[str, Any], task: str, backend: str | None = None) -> nn.Module:
    selected = backend or str(cfg["model"].get("backend", "official"))
    input_dim = input_dim_for_task(cfg, task)
    if selected == "official":
        return CFIMOfficialGRITModel(cfg, input_dim=input_dim)
    if selected == "local":
        return CFIMLocalGRITStyleModel(cfg, input_dim=input_dim)
    raise ValueError(f"unknown model backend {selected!r}")


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
    model = build_model(cfg, task, backend=backend).to(device)
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
        batch_records = online_train_batch(cfg, task, rng, batch_size)
        batch = collate_records(batch_records).to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = masked_mse(pred, batch.node_target, batch.node_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
        optimizer.step()
        if step % int(cfg["training"]["eval_every_steps"]) == 0 or step == 1 or step == max_steps:
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
        if step % int(cfg["training"]["checkpoint_every_steps"]) == 0:
            save_checkpoint(run_dir / f"checkpoint_step{step:06d}.pt", model, cfg, task, step, best_rel, optimizer)
    save_checkpoint(run_dir / "final.pt", model, cfg, task, last_step, best_rel, optimizer)
    best_path = run_dir / "best.pt"
    if not best_path.exists():
        save_checkpoint(best_path, model, cfg, task, last_step, best_rel, optimizer)
    manifest = {
        "task": task,
        "model": "grit",
        "model_seed": seed,
        "teacher_seed": TEACHER_SEEDS[task],
        "git_commit": git_commit(),
        "dirty_git_state": dirty_git_state(),
        "best_step": best_step,
        "best_val_relmse": best_rel,
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
    families = PPR_FAMILIES if task == "ppr_diffusion" else VORONOI_FAMILIES
    rng = random.Random(int(cfg["seeds"]["intervention_seed"]) + (0 if kind == "cf_eval" else 10000))
    selected_records = []
    manifest_rows = []
    for base in records:
        for family in families:
            candidates = []
            for u, v in candidate_pairs_for_family(base, family):
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
    for split in ("test_id", "test_ood_64"):
        records = load_records(data_dir(cfg, task) / f"{split}.pt")
        metrics, preds = evaluate_records(
            model,
            records,
            batch_size=int(cfg["training"]["eval_batch_size_graphs"]),
            device=device,
        )
        torch.save({"pred": preds, "metrics": metrics}, checkpoint_dir(cfg, task) / f"{split}_predictions_best.pt")
        rows.append({"task": task, "split": split, **metrics})
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
    model, run_cfg = load_model_from_checkpoint(cfg, task, checkpoint, device, backend=backend)
    clean_rows = clean_performance(run_cfg, task, model, device)
    clean_path = metrics_dir(cfg) / "clean_performance.csv"
    old_clean = [row for row in read_csv_dicts(clean_path) if row.get("task") != task]
    write_csv(clean_path, old_clean + clean_rows)
    interventions = torch.load(intervention_dir(cfg, task) / "cf_eval_interventions.pt", map_location="cpu", weights_only=False)
    rows = []
    batch_size = int(cfg["training"]["eval_batch_size_graphs"])
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
    cf_path = metrics_dir(cfg) / "counterfactual_metrics.csv"
    old = [row for row in read_csv_dicts(cf_path) if row.get("task") != task]
    write_csv(cf_path, old + rows)
    write_counterfactual_decisions(cfg)
    plot_counterfactual_summary(cfg)


def median(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(clean)) if clean else float("nan")


def task_clean_gate(cfg: Mapping[str, Any], task: str) -> bool:
    path = metrics_dir(cfg) / "clean_performance.csv"
    rows = [row for row in read_csv_dicts(path) if row.get("task") == task]
    by_split = {row["split"]: float(row["relmse_mean"]) for row in rows}
    gate = cfg["minimum_clean_performance"]
    return (
        by_split.get("test_id", float("inf")) <= float(gate["test_id_relmse_max"])
        and by_split.get("test_ood_64", float("inf")) <= float(gate["test_ood_64_relmse_max"])
    )


def write_counterfactual_decisions(cfg: Mapping[str, Any]) -> None:
    rows = read_csv_dicts(metrics_dir(cfg) / "counterfactual_metrics.csv")
    clean_rows = read_csv_dicts(metrics_dir(cfg) / "clean_performance.csv")
    clean_by_task = {
        row["task"]: float(row["relmse_mean"])
        for row in clean_rows
        if row.get("split") == "test_id" and row.get("relmse_mean")
    }
    decisions = []
    gate = cfg["counterfactual_correctness_gate"]
    for task in TASKS:
        families = PPR_FAMILIES if task == "ppr_diffusion" else VORONOI_FAMILIES
        clean_pass = task_clean_gate(cfg, task)
        for family in families:
            subset = [
                row
                for row in rows
                if row.get("task") == task and row.get("family") == family and row.get("effect_bin") == "high"
            ]
            cea = median([float(row["CEA"]) for row in subset]) if subset else float("nan")
            cee = median([float(row["CEE"]) for row in subset]) if subset else float("nan")
            beta = median([float(row["beta_T"]) for row in subset]) if subset else float("nan")
            cf_rel = median([float(row["cf_relmse"]) for row in subset]) if subset else float("nan")
            clean_rel = clean_by_task.get(task, float("inf"))
            passed = (
                clean_pass
                and cea >= float(gate["median_CEA_min"])
                and cee <= float(gate["median_CEE_max"])
                and float(gate["median_beta_T_min"]) <= beta <= float(gate["median_beta_T_max"])
                and cf_rel <= float(gate["cf_relmse_clean_multiplier_max"]) * clean_rel + 1.0e-6
            )
            decisions.append(
                {
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
    model, run_cfg = load_model_from_checkpoint(cfg, task, checkpoint, device, backend=backend)
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
    for start in range(0, len(records), batch_size):
        batch = collate_records(records[start : start + batch_size]).to(device)
        engine.compute_batch(batch, graph_indices=list(range(start, min(start + batch_size, len(records)))))
        print(f"[specialisation] task={task} graphs={min(start + batch_size, len(records))}/{len(records)}", flush=True)
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
    for family in PPR_FAMILIES if task == "ppr_diffusion" else VORONOI_FAMILIES:
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
    for family in (PPR_FAMILIES if task == "ppr_diffusion" else VORONOI_FAMILIES):
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
    model, run_cfg = load_model_from_checkpoint(cfg, task, checkpoint, device, backend=backend)
    interventions = torch.load(intervention_dir(cfg, task) / "patch_eval_interventions.pt", map_location="cpu", weights_only=False)
    max_per_family = int(cfg["patching"]["max_interventions_per_family"])
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interventions:
        if row["effect_bin"] in {"high", "null"} and len(by_family[row["family"]]) < max_per_family:
            by_family[row["family"]].append(row)
    selected = [row for family_rows in by_family.values() for row in family_rows]
    specs = patch_specs_from_scores(run_cfg, task, model)
    rows = []
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
        print(f"[patching] task={task} intervention={idx + 1}/{len(selected)} rows={len(rows)}", flush=True)
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


def write_mediation_decisions(cfg: Mapping[str, Any]) -> None:
    rows = read_csv_dicts(metrics_dir(cfg) / "patch_group_metrics.csv")
    decisions = []
    for task in TASKS:
        families = PPR_FAMILIES if task == "ppr_diffusion" else VORONOI_FAMILIES
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
    for task in TASKS:
        families = PPR_FAMILIES if task == "ppr_diffusion" else VORONOI_FAMILIES
        for family in families:
            subset = [row for row in rows if row.get("task") == task and row.get("family") == family and row.get("effect_bin") == "high"]
            if subset:
                labels.append(f"{task}\n{family}")
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


def plot_specialisation_atlas(cfg: Mapping[str, Any]) -> None:
    rows = read_csv_dicts(metrics_dir(cfg) / "specialisation_scores.csv")
    if not rows:
        return
    out = figures_main_dir(cfg) / "fig2_specialisation_atlas.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_rows = [row for row in rows if row.get("metric") in {"routing_follow", "transport_follow"} and row.get("block") == "all"]
    if not plot_rows:
        plt.close(fig)
        return
    labels = [f"{row.get('task')}\nL{row.get('layer')}H{row.get('head')}\n{row.get('metric')}" for row in plot_rows[:40]]
    values = [float(row.get("mean", row.get("value", 0.0)) or 0.0) for row in plot_rows[:40]]
    ax.bar(np.arange(len(values)), values, color="#805ad5")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("specialisation score")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_patching_summaries(cfg: Mapping[str, Any]) -> None:
    rows = read_csv_dicts(metrics_dir(cfg) / "patch_group_metrics.csv")
    if not rows:
        return
    out = figures_main_dir(cfg) / "fig3_interchange_mediation_matrix.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    values = []
    for task in TASKS:
        families = PPR_FAMILIES if task == "ppr_diffusion" else VORONOI_FAMILIES
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
    for name in [
        "fig4_cumulative_head_recovery.pdf",
        "fig5_routing_transport_specificity.pdf",
    ]:
        target = figures_main_dir(cfg) / name
        if not target.exists():
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.axis("off")
            ax.text(0.5, 0.5, "Generated from patch metrics after full ranking controls", ha="center", va="center")
            fig.savefig(target)
            plt.close(fig)


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
    cache_data(cfg, task, force=force_data)
    if not skip_training:
        train(cfg, task, device_name=device_name, backend=backend)
    build_interventions(cfg, task, "cf_eval", force=force_interventions)
    build_interventions(cfg, task, "patch_eval", force=force_interventions)
    evaluate_counterfactuals(cfg, task, device_name=device_name, backend=backend)
    if (backend or cfg["model"].get("backend", "official")) == "official":
        run_specialisation(cfg, task, device_name=device_name, backend=backend)
        if bool(cfg["patching"].get("allow_failed_gate_patching", False)) or task_clean_gate(cfg, task):
            run_patching(cfg, task, device_name=device_name, backend=backend)
        else:
            print(f"[patching] skipped for {task}: clean gate failed", flush=True)


def write_default_configs(output_dir: Path) -> None:
    for task in TASKS:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["task"] = task
        cfg["run"] = {
            "name": f"cfim_grit_{task}_seed1001",
            "hardware": "single_a100_80gb",
            "model_note": "2-layer official-backed GRIT-RRWP with continuous node-regression head",
        }
        write_yaml(output_dir / f"grit_{task}.yaml", cfg)


def print_hpc_commands(config_dir: Path) -> None:
    for task in TASKS:
        cfg_path = config_dir / f"grit_{task}.yaml"
        print(f"# {task}")
        print(f"python -m graph_specialisation_metrics.counterfactual_interchange_mediation run-sequence --config {cfg_path} --task {task} --device cuda --backend official")


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
    p.add_argument("--backend", choices=("official", "local"))

    p = sub.add_parser("build-interventions")
    common(p)
    p.add_argument("--kind", choices=("cf_eval", "patch_eval"), required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("evaluate-counterfactuals")
    common(p)
    p.add_argument("--device", default="auto")
    p.add_argument("--backend", choices=("official", "local"))
    p.add_argument("--checkpoint", type=Path)

    p = sub.add_parser("run-specialisation")
    common(p)
    p.add_argument("--device", default="auto")
    p.add_argument("--backend", choices=("official", "local"))
    p.add_argument("--checkpoint", type=Path)

    p = sub.add_parser("run-patching")
    common(p)
    p.add_argument("--device", default="auto")
    p.add_argument("--backend", choices=("official", "local"))
    p.add_argument("--checkpoint", type=Path)

    p = sub.add_parser("run-sequence")
    common(p)
    p.add_argument("--device", default="auto")
    p.add_argument("--backend", choices=("official", "local"))
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
    cfg = load_config(args.config, task=args.task, fast_dev_run=bool(getattr(args, "fast_dev_run", False)))
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
    elif args.command == "run-patching":
        run_patching(cfg, task, device_name=args.device, checkpoint=args.checkpoint, backend=args.backend)
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
