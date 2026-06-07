#!/usr/bin/env python3
"""HPC/SLURM runner for compact GraphBench algorithmic hard-OOD base training.

This runner intentionally uses the official ``graphbench.Loader`` datasets.
It implements the compact hard-split base-model protocol used for mechanistic
analysis rather than a full GraphBench reproduction. It is designed for one
task/model/seed per SLURM job on an A100 80GB GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


RUN_NAME = "graphbench_algoreas_hpc_base_v1"
RUN_VERSION = "hpc_base_v1_5task_5k_pe_cache_matched_params"
DATALOADER_NUM_WORKERS = int(os.environ.get("GRAPHBENCH_NUM_WORKERS", "4"))
DATALOADER_PIN_MEMORY = os.environ.get("GRAPHBENCH_PIN_MEMORY", "1").lower() not in {"0", "false", "no"}

BASE_TASK_TYPES = {
    "steinertree": "edge_binary",
    "bipartite_matching": "edge_binary",
    "maxclique": "node_binary",
    "topologicalorder": "node_regression",
    "flow": "graph_regression",
    "mst": "edge_binary",
    "bridges": "edge_binary",
}
DIFFICULTIES = ("easy", "medium", "hard")
TASK_TYPES = {
    f"{task}_{difficulty}": task_type
    for task, task_type in BASE_TASK_TYPES.items()
    for difficulty in DIFFICULTIES
}
DEFAULT_TASKS = ("bipartite_matching_hard", "flow_hard", "mst_hard", "maxclique_hard", "bridges_hard")
CALIBRATION_TASKS = ("mst_easy", "bridges_easy", "flow_easy")
MODEL_NAMES = ("graphormer", "graphgps", "grit", "static_grit", "gcn_plus", "gin_plus", "gatedgcn_plus")
DEFAULT_MODELS = ("graphormer", "graphgps", "static_grit", "grit", "gatedgcn_plus", "gin_plus", "gcn_plus")
RW_STEPS = 16
RRWP_STEPS = 16
GRAPHORMER_NUM_SPATIAL = 512
GRAPHORMER_SPATIAL_POS_MAX = 1024
GRAPHORMER_MULTI_HOP_MAX_DIST = 5
SPD_CAP = GRAPHORMER_NUM_SPATIAL - 1
PAIR_RAW_DIM = 3 + 2 + (RRWP_STEPS + 1)
NODE_VOCAB = 8
MODEL_SIZE_PRESETS = {
    "graphormer": {"hidden_dim": 160},
    "graphgps": {"hidden_dim": 128},
    "grit": {"hidden_dim": 112},
    "static_grit": {"hidden_dim": 168},
    "gcn_plus": {"gnn_hidden_dim": 240},
    "gin_plus": {"gnn_hidden_dim": 224},
    "gatedgcn_plus": {"gnn_hidden_dim": 184},
}
EVAL_BATCH_SIZE_BY_MODEL = {
    "graphormer": 128,
    "graphgps": 64,
    "grit": 32,
    "static_grit": 32,
    "gcn_plus": 256,
    "gin_plus": 256,
    "gatedgcn_plus": 256,
}
TRAIN_BATCH_SIZE_BY_MODEL = {
    "graphormer": 1024,
    "graphgps": 1024,
    "grit": 1024,
    "static_grit": 1024,
    "gcn_plus": 1024,
    "gin_plus": 1024,
    "gatedgcn_plus": 1024,
}
GRAPH_TRANSFORMER_LRS = {
    "mst": 3.0e-4,
    "maxclique": 1.0e-4,
    "flow": 1.0e-4,
    "bipartite_matching": 1.0e-4,
    "bridges": 1.0e-4,
}
GNN_PLUS_LRS = {
    "gcn_plus": {"mst": 2.0e-3, "maxclique": 1.0e-3, "flow": 2.0e-3, "bipartite_matching": 1.0e-3, "bridges": 1.0e-3},
    "gin_plus": {"mst": 2.0e-3, "maxclique": 1.0e-3, "flow": 2.0e-3, "bipartite_matching": 1.0e-3, "bridges": 1.0e-3},
    "gatedgcn_plus": {"mst": 2.0e-3, "maxclique": 1.0e-3, "flow": 2.0e-3, "bipartite_matching": 2.0e-3, "bridges": 1.0e-3},
}
FAIR_FAMILY_LRS = {
    "graph_transformer": 2.0e-4,
    "gnn_plus": 1.0e-3,
}
PROTOCOL_CONFIG = {
    "run_name": RUN_NAME,
    "purpose": "interp_base_models_not_full_benchmark_replication",
    "difficulty": "hard",
    "graph_format": "original_graph",
    "edge_token_transform": False,
    "tasks": DEFAULT_TASKS,
    "split_sizes": {"train": 40000, "val": 4000, "test": 8000},
    "split_nodes": {"train": 16, "val": 128, "test": 128},
    "model_size_preset": "paper_matched_about_2p2m_params",
    "lr_policy": "fair_family_fixed",
    "fair_family_learning_rates": FAIR_FAMILY_LRS,
    "train_batch_size_by_model": TRAIN_BATCH_SIZE_BY_MODEL,
    "eval_batch_size_by_model": EVAL_BATCH_SIZE_BY_MODEL,
    "max_steps": 5000,
    "warmup_steps": 500,
    "eval_every_steps": 500,
    "major_checkpoint_steps": [2500, 3000, 3500, 4000, 4500, 5000],
    "hard_generator_weights": {
        "train": {"ER": 1.0, "PC": 0.0, "NWS": 0.0, "BA": 0.0, "DBA": 0.0, "SBM": 0.0},
        "val_test": {"ER": 0.0, "PC": 0.2, "NWS": 0.2, "BA": 0.2, "DBA": 0.2, "SBM": 0.2},
    },
    "pe_policy": {
        "graphormer": "native_degree_spd_multihop_edge_bias",
        "graphgps": "rwse16_node_pe",
        "grit": "rrwp16_relative_pair_pe_no_spd",
        "static_grit": "rrwp16_relative_pair_pe_no_spd",
        "gnn_plus": "rwse16_node_pe",
    },
}


@dataclass(frozen=True)
class ScreenConfig:
    seed: int = 0
    split_seed: int = 0
    train_size: int = 40000
    val_size: int = 4000
    test_size: int = 8000
    batch_size: int = 0
    eval_batch_size: int = 0
    max_steps: int = 5000
    warmup_steps: int = 500
    eval_every_steps: int = 500
    min_checkpoint_step: int = 2500
    val_watch_size: int = 1000
    save_major_checkpoints: bool = True
    min_delta: float = 1.0e-5
    lr: Optional[float] = None
    lr_policy: str = "fair_family"
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    hidden_dim: int = 128
    gnn_hidden_dim: int = 192
    layers: int = 6
    heads: int = 8
    dropout: float = 0.1
    attn_dropout: float = 0.1
    gnnplus_residual: bool = True
    gnnplus_ffn: bool = True
    gnnplus_act: str = "relu"
    amp: bool = True
    final_eval: bool = False


@dataclass
class OfficialGraph:
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_value: torch.Tensor
    target: torch.Tensor
    task_type: str
    num_nodes: int
    spd: Optional[torch.Tensor] = None
    rwse: Optional[torch.Tensor] = None
    rrwp: Optional[torch.Tensor] = None


@dataclass
class OfficialBatch:
    node_type: torch.Tensor
    node_mask: torch.Tensor
    adj: torch.Tensor
    edge_value_mat: torch.Tensor
    degree: torch.Tensor
    spd: torch.Tensor
    rwse: torch.Tensor
    rrwp: torch.Tensor
    pair_xi: torch.Tensor
    edge_batch: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_value: torch.Tensor
    edge_target: torch.Tensor
    node_target: torch.Tensor
    graph_target: torch.Tensor
    graph_num_nodes: torch.Tensor
    task_type: str
    num_graphs: int
    max_nodes: int

    @property
    def pair_mask(self) -> torch.Tensor:
        return self.node_mask[:, :, None] & self.node_mask[:, None, :]

    def to(self, device: torch.device) -> "OfficialBatch":
        return OfficialBatch(
            node_type=self.node_type.to(device),
            node_mask=self.node_mask.to(device),
            adj=self.adj.to(device),
            edge_value_mat=self.edge_value_mat.to(device),
            degree=self.degree.to(device),
            spd=self.spd.to(device),
            rwse=self.rwse.to(device),
            rrwp=self.rrwp.to(device),
            pair_xi=self.pair_xi.to(device),
            edge_batch=self.edge_batch.to(device),
            edge_src=self.edge_src.to(device),
            edge_dst=self.edge_dst.to(device),
            edge_value=self.edge_value.to(device),
            edge_target=self.edge_target.to(device),
            node_target=self.node_target.to(device),
            graph_target=self.graph_target.to(device),
            graph_num_nodes=self.graph_num_nodes.to(device),
            task_type=self.task_type,
            num_graphs=self.num_graphs,
            max_nodes=self.max_nodes,
        )


class OfficialGraphDataset(Dataset[OfficialGraph]):
    def __init__(self, graphs: Sequence[OfficialGraph], task_name: str, split: str) -> None:
        self.graphs = list(graphs)
        self.task_name = task_name
        self.split = split

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> OfficialGraph:
        return self.graphs[index]


def graph_to_payload(graph: OfficialGraph) -> dict[str, object]:
    return {
        "node_type": graph.node_type,
        "edge_index": graph.edge_index,
        "edge_value": graph.edge_value,
        "target": graph.target,
        "task_type": graph.task_type,
        "num_nodes": graph.num_nodes,
        "spd": graph.spd,
        "rwse": graph.rwse,
        "rrwp": graph.rrwp,
    }


def graph_from_payload(payload: Mapping[str, object]) -> OfficialGraph:
    return OfficialGraph(
        node_type=payload["node_type"],
        edge_index=payload["edge_index"],
        edge_value=payload["edge_value"],
        target=payload["target"],
        task_type=str(payload["task_type"]),
        num_nodes=int(payload["num_nodes"]),
        spd=payload.get("spd"),
        rwse=payload.get("rwse"),
        rrwp=payload.get("rrwp"),
    )


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def require_package(import_name: str, package_name: str, log=print) -> None:
    try:
        __import__(import_name)
    except Exception:
        raise RuntimeError(
            f"Missing required package {package_name!r}. Install it in the HPC environment before submitting jobs."
        )


def import_plotting():
    require_package("matplotlib", "matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str, allow_cpu: bool = False) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError("CUDA GPU is not available. Use a GPU runtime or pass --allow-cpu for smoke tests.")
    device = torch.device(requested)
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError(f"Refusing to train on {device}; pass --allow-cpu for local smoke tests.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return device


def shortest_path_buckets(adj: torch.Tensor, cap: int = SPD_CAP) -> torch.Tensor:
    n = int(adj.size(0))
    inf = cap + 1
    dist = torch.full((n, n), inf, dtype=torch.long)
    neighbors = [torch.nonzero(adj[i] > 0, as_tuple=False).flatten().tolist() for i in range(n)]
    for src in range(n):
        dist[src, src] = 0
        queue: deque[int] = deque([src])
        while queue:
            node = queue.popleft()
            if int(dist[src, node]) >= cap:
                continue
            for nbr in neighbors[node]:
                if dist[src, nbr] > dist[src, node] + 1:
                    dist[src, nbr] = dist[src, node] + 1
                    queue.append(nbr)
    return dist.clamp(max=inf)


def random_walk_features(adj: torch.Tensor, steps: int = RW_STEPS) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(adj.size(0))
    degree = adj.sum(dim=-1, keepdim=True)
    transition = torch.where(degree > 0, adj / degree.clamp_min(1.0), torch.zeros_like(adj))
    power = torch.eye(n, dtype=torch.float32)
    rwse = []
    rrwp = [power.clone()]
    for _ in range(steps):
        power = power @ transition
        rwse.append(torch.diagonal(power, dim1=-2, dim2=-1).float())
        rrwp.append(power.float().clone())
    return torch.stack(rwse, dim=-1), torch.stack(rrwp, dim=-1)


def pe_dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported PE cache dtype {name!r}")


def graph_adjacency(graph: OfficialGraph) -> torch.Tensor:
    adj = torch.zeros(graph.num_nodes, graph.num_nodes, dtype=torch.float32)
    src = graph.edge_index[0].long()
    dst = graph.edge_index[1].long()
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0
    return adj


def compute_graph_pe(graph: OfficialGraph, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adj = graph_adjacency(graph)
    spd = shortest_path_buckets(adj).to(torch.int16)
    rwse, rrwp = random_walk_features(adj)
    return spd, rwse.to(dtype), rrwp.to(dtype)


def graph_with_pe(
    graph: OfficialGraph,
    spd: Optional[torch.Tensor],
    rwse: Optional[torch.Tensor],
    rrwp: Optional[torch.Tensor],
) -> OfficialGraph:
    return OfficialGraph(
        node_type=graph.node_type,
        edge_index=graph.edge_index,
        edge_value=graph.edge_value,
        target=graph.target,
        task_type=graph.task_type,
        num_nodes=graph.num_nodes,
        spd=spd,
        rwse=rwse,
        rrwp=rrwp,
    )


def pe_cache_path(root: Path, namespace: str, task_name: str, split: str, size: int, seed: int, dtype_name: str) -> Path:
    return root / RUN_VERSION / namespace / f"{task_name}_{split}_n{size}_seed{seed}_rw{RW_STEPS}_rrwp{RRWP_STEPS}_{dtype_name}.pt"


def load_pe_cache(
    path: Path,
    dataset: OfficialGraphDataset,
    dtype_name: str,
    log=print,
) -> Optional[OfficialGraphDataset]:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        return None
    if payload.get("version") != RUN_VERSION or payload.get("dtype") != dtype_name:
        return None
    pe_items = payload.get("pe")
    if not isinstance(pe_items, Sequence) or len(pe_items) != len(dataset):
        log(f"[pe] Ignoring incompatible PE cache {path}")
        return None
    graphs = []
    for graph, pe in zip(dataset.graphs, pe_items):
        graphs.append(graph_with_pe(graph, pe["spd"], pe["rwse"], pe["rrwp"]))
    log(f"[pe] Loaded {dataset.task_name}/{dataset.split} PE cache: {path}")
    return OfficialGraphDataset(graphs, task_name=dataset.task_name, split=dataset.split)


def save_pe_cache(path: Path, dataset: OfficialGraphDataset, dtype_name: str, log=print) -> OfficialGraphDataset:
    dtype = pe_dtype_from_name(dtype_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    graphs = []
    pe_items = []
    start = time.time()
    for idx, graph in enumerate(dataset.graphs, start=1):
        spd, rwse, rrwp = compute_graph_pe(graph, dtype)
        graphs.append(graph_with_pe(graph, spd, rwse, rrwp))
        pe_items.append({"spd": spd, "rwse": rwse, "rrwp": rrwp})
        if idx == len(dataset) or idx % max(1000, len(dataset) // 10) == 0:
            log(f"[pe] {dataset.task_name}/{dataset.split}: {idx}/{len(dataset)} graphs precomputed")
    torch.save(
        {
            "version": RUN_VERSION,
            "task": dataset.task_name,
            "split": dataset.split,
            "dtype": dtype_name,
            "rw_steps": RW_STEPS,
            "rrwp_steps": RRWP_STEPS,
            "pe": pe_items,
        },
        path,
    )
    elapsed = time.time() - start
    log(f"[pe] Saved {dataset.task_name}/{dataset.split} PE cache to {path} in {elapsed:.1f}s")
    return OfficialGraphDataset(graphs, task_name=dataset.task_name, split=dataset.split)


def attach_or_build_pe_cache(
    splits: Mapping[str, OfficialGraphDataset],
    root: Path,
    cfg: ScreenConfig,
    namespace: str,
    dtype_name: str,
    force_recompute: bool,
    build_missing: bool,
    require_present: bool,
    log=print,
) -> dict[str, OfficialGraphDataset]:
    split_sizes = {"train": cfg.train_size, "val": cfg.val_size, "test": cfg.test_size}
    split_seeds = {"train": 101, "val": 211, "test": 307}
    out: dict[str, OfficialGraphDataset] = {}
    for split, dataset in splits.items():
        path = pe_cache_path(root, namespace, dataset.task_name, split, split_sizes[split], cfg.split_seed + split_seeds[split], dtype_name)
        cached = None if force_recompute else load_pe_cache(path, dataset, dtype_name, log=log)
        if cached is not None:
            out[split] = cached
            continue
        if require_present:
            raise FileNotFoundError(f"required PE cache is missing or incompatible: {path}")
        if not build_missing:
            log(f"[pe] Missing PE cache for {dataset.task_name}/{split}; falling back to on-the-fly collation")
            out[split] = dataset
            continue
        out[split] = save_pe_cache(path, dataset, dtype_name, log=log)
    return out


def pair_features(adj: torch.Tensor, spd: torch.Tensor, rrwp: torch.Tensor, edge_value_mat: torch.Tensor) -> torch.Tensor:
    del spd
    n = int(adj.size(0))
    eye = torch.eye(n, dtype=torch.bool, device=adj.device)
    semantic = torch.zeros(n, n, 3, dtype=torch.float32, device=adj.device)
    semantic[..., 0] = eye.float()
    semantic[..., 1] = ((adj <= 0) & (~eye)).float()
    semantic[..., 2] = adj.float()
    edge_pair = torch.stack([edge_value_mat.float(), edge_value_mat.float().t()], dim=-1)
    return torch.cat([semantic, edge_pair, rrwp.float()], dim=-1)


def graph_from_pyg(data: object, task_name: str) -> OfficialGraph:
    task_type = TASK_TYPES[task_name]
    num_nodes = int(getattr(data, "num_nodes"))
    edge_index = getattr(data, "edge_index").long().cpu()
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is None:
        edge_value = torch.ones(edge_index.size(1), dtype=torch.float32)
    else:
        edge_value = edge_attr.detach().cpu().float().reshape(-1)
    x = getattr(data, "x", None)
    if x is None:
        node_type = torch.zeros(num_nodes, dtype=torch.long)
    else:
        node_type = x.detach().cpu().long().reshape(-1).clamp(min=0, max=NODE_VOCAB - 1)
        if node_type.numel() != num_nodes:
            node_type = torch.zeros(num_nodes, dtype=torch.long)
    target = getattr(data, "y").detach().cpu()
    return OfficialGraph(
        node_type=node_type,
        edge_index=edge_index,
        edge_value=edge_value,
        target=target,
        task_type=task_type,
        num_nodes=num_nodes,
    )


def deterministic_subset(dataset: Sequence[object], size: int, seed: int) -> list[object]:
    n = len(dataset)
    if size <= 0 or size >= n:
        return [dataset[i] for i in range(n)]
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(n), k=size))
    return [dataset[i] for i in idxs]


def subset_cache_path(root: Path, task_name: str, split: str, size: int, seed: int) -> Path:
    return root / "_hpc_subset_cache" / f"{task_name}_{split}_n{size}_seed{seed}_{RUN_VERSION}.pt"


def load_subset_cache(path: Path, task_name: str, split: str) -> Optional[OfficialGraphDataset]:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("version") != RUN_VERSION:
        return None
    graphs = [graph_from_payload(graph) for graph in payload["graphs"]]
    return OfficialGraphDataset(graphs, task_name=task_name, split=split)


def save_subset_cache(path: Path, task_name: str, split: str, graphs: Sequence[OfficialGraph]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": RUN_VERSION,
            "task": task_name,
            "split": split,
            "graphs": [graph_to_payload(graph) for graph in graphs],
        },
        path,
    )


def edge_value_stats(dataset: OfficialGraphDataset) -> dict[str, float]:
    values = [dataset[i].edge_value.float().reshape(-1) for i in range(len(dataset)) if dataset[i].edge_value.numel()]
    if not values:
        return {"mean": 0.0, "std": 1.0}
    flat = torch.cat(values)
    return {"mean": float(flat.mean()), "std": float(flat.std(unbiased=False).clamp_min(1.0e-6))}


def normalize_edge_values(dataset: OfficialGraphDataset, stats: Mapping[str, float]) -> OfficialGraphDataset:
    mean = float(stats["mean"])
    std = max(float(stats["std"]), 1.0e-6)
    graphs = [
        OfficialGraph(
            node_type=graph.node_type,
            edge_index=graph.edge_index,
            edge_value=(graph.edge_value.float() - mean) / std,
            target=graph.target,
            task_type=graph.task_type,
            num_nodes=graph.num_nodes,
        )
        for graph in dataset.graphs
    ]
    return OfficialGraphDataset(graphs, task_name=dataset.task_name, split=dataset.split)


def normalize_edge_values_in_splits(
    splits: Mapping[str, OfficialGraphDataset],
    log=print,
) -> dict[str, OfficialGraphDataset]:
    stats = edge_value_stats(splits["train"])
    log(f"[data] edge value train normalization={stats}")
    return {split: normalize_edge_values(dataset, stats) for split, dataset in splits.items()}


def get_official_split(split_map: Mapping[str, object], split: str) -> object:
    aliases = {
        "train": ("train",),
        "val": ("val", "valid", "validation"),
        "test": ("test",),
    }
    for key in aliases[split]:
        if key in split_map:
            return split_map[key]
    raise KeyError(f"GraphBench loader did not return split={split!r}; available keys={list(split_map.keys())}")


def load_official_graphbench_task(
    root: Path,
    task_name: str,
    cfg: ScreenConfig,
    force_reload: bool,
    log=print,
) -> dict[str, OfficialGraphDataset]:
    split_sizes = {"train": cfg.train_size, "val": cfg.val_size, "test": cfg.test_size}
    split_seeds = {"train": 101, "val": 211, "test": 307}
    cached: dict[str, OfficialGraphDataset] = {}
    if not force_reload:
        for split, size in split_sizes.items():
            path = subset_cache_path(root, task_name, split, size, cfg.split_seed + split_seeds[split])
            dataset = load_subset_cache(path, task_name, split)
            if dataset is None:
                cached = {}
                break
            cached[split] = dataset
        if cached:
            log(f"[data] Loaded converted subset cache for {task_name}")
            for split, dataset in cached.items():
                node_counts = [graph.num_nodes for graph in dataset]
                log(
                    f"[data] {task_name}/{split}: cached={len(dataset)} "
                        f"nodes={min(node_counts) if node_counts else 0}-{max(node_counts) if node_counts else 0}"
                )
            return normalize_edge_values_in_splits(cached, log=log)

    require_package("graphbench", "graphbench-lib", log=log)
    from graphbench import Loader  # type: ignore

    log(f"[data] Loading official GraphBench task={task_name}")
    loader = Loader(root=str(root), dataset_names=task_name)
    loaded = loader.load()
    if len(loaded) != 1:
        raise RuntimeError(f"expected one loaded task for {task_name}, got {len(loaded)}")
    split_map = loaded[0]
    out = {}
    for split, size in split_sizes.items():
        raw_dataset = get_official_split(split_map, split)
        raw_graphs = deterministic_subset(raw_dataset, size, cfg.split_seed + split_seeds[split])
        graphs = [graph_from_pyg(graph, task_name) for graph in raw_graphs]
        out[split] = OfficialGraphDataset(graphs, task_name=task_name, split=split)
        save_subset_cache(subset_cache_path(root, task_name, split, size, cfg.split_seed + split_seeds[split]), task_name, split, graphs)
        node_counts = [graph.num_nodes for graph in graphs]
        log(
            f"[data] {task_name}/{split}: selected={len(graphs)} "
            f"nodes={min(node_counts) if node_counts else 0}-{max(node_counts) if node_counts else 0}"
        )
    return normalize_edge_values_in_splits(out, log=log)


def collate_graphs(graphs: Sequence[OfficialGraph]) -> OfficialBatch:
    bsz = len(graphs)
    task_type = graphs[0].task_type
    max_nodes = max(graph.num_nodes for graph in graphs)
    node_type = torch.zeros(bsz, max_nodes, dtype=torch.long)
    node_mask = torch.zeros(bsz, max_nodes, dtype=torch.bool)
    adj = torch.zeros(bsz, max_nodes, max_nodes, dtype=torch.float32)
    edge_value_mat = torch.zeros(bsz, max_nodes, max_nodes, dtype=torch.float32)
    degree = torch.zeros(bsz, max_nodes, dtype=torch.float32)
    spd = torch.full((bsz, max_nodes, max_nodes), SPD_CAP, dtype=torch.long)
    rwse = torch.zeros(bsz, max_nodes, RW_STEPS, dtype=torch.float32)
    rrwp = torch.zeros(bsz, max_nodes, max_nodes, RW_STEPS + 1, dtype=torch.float32)
    node_target = torch.zeros(bsz, max_nodes, dtype=torch.float32)
    graph_target = torch.zeros(bsz, dtype=torch.float32)
    graph_num_nodes = []
    edge_batch = []
    edge_src = []
    edge_dst = []
    edge_value = []
    edge_target = []
    for graph_idx, graph in enumerate(graphs):
        n = graph.num_nodes
        node_type[graph_idx, :n] = graph.node_type
        node_mask[graph_idx, :n] = True
        edge_index = graph.edge_index
        src = edge_index[0].long()
        dst = edge_index[1].long()
        vals = graph.edge_value.float()
        adj[graph_idx, src, dst] = 1.0
        adj[graph_idx, dst, src] = 1.0
        edge_value_mat[graph_idx, src, dst] = vals
        graph_num_nodes.append(n)
        edge_batch.append(torch.full((edge_index.size(1),), graph_idx, dtype=torch.long))
        edge_src.append(src)
        edge_dst.append(dst)
        edge_value.append(vals)
        if task_type == "edge_binary":
            edge_target.append(graph.target.float().reshape(-1))
        elif task_type in {"node_binary", "node_regression"}:
            node_target[graph_idx, :n] = graph.target.float().reshape(-1)[:n]
        elif task_type == "graph_regression":
            graph_target[graph_idx] = graph.target.float().reshape(-1)[0]
        else:
            raise ValueError(task_type)
        degree[graph_idx, :n] = adj[graph_idx, :n, :n].sum(dim=-1)
        if graph.spd is not None and graph.rwse is not None and graph.rrwp is not None:
            spd_i = graph.spd.long()
            rwse_i = graph.rwse.float()
            rrwp_i = graph.rrwp.float()
        else:
            spd_i = shortest_path_buckets(adj[graph_idx, :n, :n])
            rwse_i, rrwp_i = random_walk_features(adj[graph_idx, :n, :n])
        spd[graph_idx, :n, :n] = spd_i
        rwse[graph_idx, :n] = rwse_i
        rrwp[graph_idx, :n, :n] = rrwp_i
    if task_type != "edge_binary":
        edge_target = [torch.empty(0, dtype=torch.float32)]
    pair_xi = torch.stack([pair_features(adj[i], spd[i], rrwp[i], edge_value_mat[i]) for i in range(bsz)], dim=0)
    return OfficialBatch(
        node_type=node_type,
        node_mask=node_mask,
        adj=adj,
        edge_value_mat=edge_value_mat,
        degree=degree,
        spd=spd,
        rwse=rwse,
        rrwp=rrwp,
        pair_xi=pair_xi,
        edge_batch=torch.cat(edge_batch) if edge_batch else torch.empty(0, dtype=torch.long),
        edge_src=torch.cat(edge_src) if edge_src else torch.empty(0, dtype=torch.long),
        edge_dst=torch.cat(edge_dst) if edge_dst else torch.empty(0, dtype=torch.long),
        edge_value=torch.cat(edge_value) if edge_value else torch.empty(0, dtype=torch.float32),
        edge_target=torch.cat(edge_target) if task_type == "edge_binary" else torch.empty(0, dtype=torch.float32),
        node_target=node_target,
        graph_target=graph_target,
        graph_num_nodes=torch.tensor(graph_num_nodes, dtype=torch.long),
        task_type=task_type,
        num_graphs=bsz,
        max_nodes=max_nodes,
    )


def make_loader(dataset: Dataset[OfficialGraph], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collate_graphs,
        num_workers=DATALOADER_NUM_WORKERS,
        pin_memory=DATALOADER_PIN_MEMORY,
        persistent_workers=DATALOADER_NUM_WORKERS > 0,
    )


class NodeEncoder(nn.Module):
    def __init__(self, cfg: ScreenConfig, dim: int, use_rwse: bool = True) -> None:
        super().__init__()
        self.use_rwse = use_rwse
        self.node_encoder = nn.Embedding(NODE_VOCAB, dim)
        self.degree_proj = nn.Linear(1, dim, bias=False)
        if use_rwse:
            pe_hidden = max(16, min(dim, 128))
            self.pe_bn = nn.BatchNorm1d(RW_STEPS)
            self.pe_encoder = nn.Sequential(nn.Linear(RW_STEPS, pe_hidden), nn.ReLU(), nn.Linear(pe_hidden, dim))
        else:
            self.pe_bn = None
            self.pe_encoder = None

    def forward(self, batch: OfficialBatch) -> torch.Tensor:
        bsz, n, steps = batch.rwse.shape
        h = (
            self.node_encoder(batch.node_type)
            + self.degree_proj(torch.log1p(batch.degree).unsqueeze(-1))
        )
        if self.use_rwse:
            assert self.pe_bn is not None and self.pe_encoder is not None
            pe = self.pe_bn(batch.rwse.reshape(-1, steps)).view(bsz, n, steps)
            h = h + self.pe_encoder(pe)
        return h * batch.node_mask.unsqueeze(-1)


class PredictionHeads(nn.Module):
    def __init__(self, dim: int, edge_dim: int = 0) -> None:
        super().__init__()
        self.graph_head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.node_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        edge_in = 4 * dim + edge_dim + 1
        self.edge_head = nn.Sequential(nn.LayerNorm(edge_in), nn.Linear(edge_in, dim), nn.GELU(), nn.Linear(dim, 1))

    def graph(self, h: torch.Tensor, batch: OfficialBatch) -> torch.Tensor:
        mask = batch.node_mask.unsqueeze(-1)
        mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        max_pool = h.masked_fill(~batch.node_mask.unsqueeze(-1), torch.finfo(h.dtype).min).max(dim=1).values
        return self.graph_head(torch.cat([mean, max_pool], dim=-1)).squeeze(-1)

    def node(self, h: torch.Tensor) -> torch.Tensor:
        return self.node_head(h).squeeze(-1)

    def edge(self, h: torch.Tensor, batch: OfficialBatch, edge_repr: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = h[batch.edge_batch, batch.edge_src]
        dst = h[batch.edge_batch, batch.edge_dst]
        pieces = [src, dst, torch.abs(src - dst), src * dst]
        if edge_repr is not None:
            pieces.append(edge_repr[batch.edge_batch, batch.edge_src, batch.edge_dst])
        pieces.append(batch.edge_value.unsqueeze(-1).to(h.dtype))
        return self.edge_head(torch.cat(pieces, dim=-1)).squeeze(-1)


def activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation {name!r}")


def apply_node_bn(bn: nn.BatchNorm1d, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    if bool(mask.any()):
        out[mask] = bn(x[mask])
    return out * mask.unsqueeze(-1)


def apply_pair_bn(bn: nn.BatchNorm1d, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    if bool(mask.any()):
        out[mask] = bn(x[mask])
    return out * mask.unsqueeze(-1)


def incoming_edge_mask(batch: OfficialBatch) -> torch.Tensor:
    return (batch.adj.transpose(1, 2) > 0) & batch.pair_mask


def incoming_edge_features(batch: OfficialBatch) -> torch.Tensor:
    incoming = incoming_edge_mask(batch)
    return torch.stack([incoming.float(), batch.edge_value_mat.transpose(1, 2)], dim=-1)


class MultiHeadAttentionWithBias(nn.Module):
    def __init__(self, dim: int, heads: int, attn_dropout: float, out_dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_bias
        scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
        out = torch.matmul(self.attn_dropout(attn), v).transpose(1, 2).reshape(bsz, seq_len, dim)
        return self.out_dropout(self.out_proj(out))


class TransformerBlockWithBias(nn.Module):
    def __init__(self, dim: int, heads: int, cfg: ScreenConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithBias(dim, heads, cfg.attn_dropout, cfg.dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_bias, key_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class GraphormerModel(nn.Module):
    def __init__(self, cfg: ScreenConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.cfg = cfg
        self.node_encoder = nn.Embedding(NODE_VOCAB, dim)
        self.degree_encoder = nn.Embedding(513, dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, dim)
        self.spatial_encoder = nn.Embedding(GRAPHORMER_NUM_SPATIAL, cfg.heads, padding_idx=0)
        self.edge_encoder = nn.Embedding(2, cfg.heads, padding_idx=0)
        self.edge_value_encoder = nn.Linear(1, cfg.heads, bias=False)
        self.virtual_distance = nn.Embedding(1, cfg.heads)
        self.emb_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(TransformerBlockWithBias(dim, cfg.heads, cfg) for _ in range(cfg.layers))
        self.heads = PredictionHeads(dim)

    def build_bias(self, batch: OfficialBatch) -> torch.Tensor:
        bsz, n = batch.node_type.shape
        heads = self.cfg.heads
        bias = torch.zeros(bsz, heads, n + 1, n + 1, device=batch.node_type.device)
        spatial = self.spatial_encoder(batch.spd.long().clamp(max=GRAPHORMER_NUM_SPATIAL - 1)).permute(0, 3, 1, 2)
        edge = self.edge_encoder((batch.adj > 0).long().clamp(max=1)).permute(0, 3, 1, 2)
        edge_value = self.edge_value_encoder(batch.edge_value_mat.unsqueeze(-1)).permute(0, 3, 1, 2)
        bias[:, :, 1:, 1:] = spatial + edge + edge_value
        token_bias = self.virtual_distance.weight.view(1, heads, 1)
        bias[:, :, 1:, 0] = bias[:, :, 1:, 0] + token_bias
        bias[:, :, 0, 1:] = bias[:, :, 0, 1:] + token_bias
        return bias

    def forward(self, batch: OfficialBatch) -> torch.Tensor:
        bsz = batch.node_type.size(0)
        degree = batch.degree.long().clamp(max=511)
        h = self.node_encoder(batch.node_type) + self.degree_encoder(degree + 1)
        token = self.graph_token.weight.unsqueeze(0).expand(bsz, -1, -1)
        h = self.emb_norm(torch.cat([token, h], dim=1))
        key_mask = torch.cat([torch.ones(bsz, 1, dtype=torch.bool, device=h.device), batch.node_mask], dim=1)
        bias = self.build_bias(batch).to(h.dtype)
        for layer in self.layers:
            h = layer(h, bias, key_mask)
        h_nodes = h[:, 1:] * batch.node_mask.unsqueeze(-1)
        if batch.task_type == "edge_binary":
            return self.heads.edge(h_nodes, batch)
        if batch.task_type in {"node_binary", "node_regression"}:
            return self.heads.node(h_nodes)
        if batch.task_type == "graph_regression":
            return self.heads.graph(h_nodes, batch)
        raise ValueError(batch.task_type)


class DenseGINE(nn.Module):
    def __init__(self, dim: int, cfg: ScreenConfig) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(cfg.dropout),
        )
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, h: torch.Tensor, batch: OfficialBatch, mask: torch.Tensor) -> torch.Tensor:
        bsz, n, dim = h.shape
        eye = torch.eye(n, dtype=torch.bool, device=h.device).view(1, n, n)
        incoming = batch.adj.transpose(1, 2) > 0
        route = incoming | (eye & batch.pair_mask)
        edge_input = torch.stack(
            [incoming.float(), batch.edge_value_mat.transpose(1, 2), eye.expand(bsz, -1, -1).float()],
            dim=-1,
        )
        edge_emb = self.edge_mlp(edge_input)
        source_h = h[:, None, :, :].expand(-1, n, -1, -1)
        messages = torch.relu(source_h + edge_emb)
        agg = (route.float().unsqueeze(-1) * messages).sum(dim=2)
        out = self.mlp((1.0 + self.eps) * h + agg)
        out = self.norm(out.reshape(-1, dim)).view_as(out)
        return out * mask.unsqueeze(-1)


class GPSLayer(nn.Module):
    def __init__(self, dim: int, cfg: ScreenConfig) -> None:
        super().__init__()
        self.local = DenseGINE(dim, cfg)
        self.local_norm = nn.BatchNorm1d(dim)
        self.global_attn = nn.MultiheadAttention(dim, cfg.heads, dropout=cfg.attn_dropout, batch_first=True)
        self.global_norm = nn.BatchNorm1d(dim)
        self.ffn_norm = nn.BatchNorm1d(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, h: torch.Tensor, batch: OfficialBatch, mask: torch.Tensor) -> torch.Tensor:
        h = h + self.local(apply_node_bn(self.local_norm, h, mask), batch, mask)
        hn = apply_node_bn(self.global_norm, h, mask)
        out, _attn = self.global_attn(hn, hn, hn, key_padding_mask=~mask, need_weights=False)
        h = h + out * mask.unsqueeze(-1)
        h = h + self.ffn(apply_node_bn(self.ffn_norm, h, mask)) * mask.unsqueeze(-1)
        return h * mask.unsqueeze(-1)


class GraphGPSModel(nn.Module):
    def __init__(self, cfg: ScreenConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.encoder = NodeEncoder(cfg, dim)
        self.layers = nn.ModuleList(GPSLayer(dim, cfg) for _ in range(cfg.layers))
        self.heads = PredictionHeads(dim)

    def forward(self, batch: OfficialBatch) -> torch.Tensor:
        h = self.encoder(batch)
        for layer in self.layers:
            h = layer(h, batch, batch.node_mask)
        if batch.task_type == "edge_binary":
            return self.heads.edge(h, batch)
        if batch.task_type in {"node_binary", "node_regression"}:
            return self.heads.node(h)
        if batch.task_type == "graph_regression":
            return self.heads.graph(h, batch)
        raise ValueError(batch.task_type)


class GritLayer(nn.Module):
    def __init__(self, dim: int, cfg: ScreenConfig, evolve_pairs: bool = True) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = dim // cfg.heads
        self.evolve_pairs = evolve_pairs
        self.node_norm1 = nn.BatchNorm1d(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.edge_to_bias = nn.Linear(dim, cfg.heads)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.out_dropout = nn.Dropout(cfg.dropout)
        self.node_norm2 = nn.BatchNorm1d(dim)
        self.node_ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(dim * 4, dim))
        if evolve_pairs:
            self.edge_norm = nn.BatchNorm1d(dim)
            self.edge_mlp = nn.Sequential(nn.Linear(dim * 3, dim * 4), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(dim * 4, dim))
        else:
            self.edge_norm = None
            self.edge_mlp = None

    def forward(self, h: torch.Tensor, edge_repr: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n, dim = h.shape
        hn = apply_node_bn(self.node_norm1, h, mask)
        q = self.q_proj(hn).view(bsz, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hn).view(bsz, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hn).view(bsz, n, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.edge_to_bias(edge_repr).permute(0, 3, 1, 2)
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        scores = scores.masked_fill(~pair_mask[:, None], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
        delta = torch.matmul(self.attn_dropout(attn), v).transpose(1, 2).reshape(bsz, n, dim)
        h = h + self.out_dropout(self.out_proj(delta)) * mask.unsqueeze(-1)
        h = h + self.node_ffn(apply_node_bn(self.node_norm2, h, mask)) * mask.unsqueeze(-1)
        if self.evolve_pairs:
            assert self.edge_mlp is not None and self.edge_norm is not None
            src = h.unsqueeze(2).expand(-1, -1, n, -1)
            dst = h.unsqueeze(1).expand(-1, n, -1, -1)
            edge_delta = self.edge_mlp(torch.cat([edge_repr, src, dst], dim=-1))
            edge_repr = apply_pair_bn(self.edge_norm, edge_repr + edge_delta, pair_mask)
        return h * mask.unsqueeze(-1), edge_repr


class GritModel(nn.Module):
    def __init__(self, cfg: ScreenConfig, evolve_pairs: bool = True) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.encoder = NodeEncoder(cfg, dim, use_rwse=False)
        self.pair_encoder = nn.Linear(PAIR_RAW_DIM, dim)
        self.layers = nn.ModuleList(GritLayer(dim, cfg, evolve_pairs=evolve_pairs) for _ in range(cfg.layers))
        self.heads = PredictionHeads(dim, edge_dim=dim)

    def forward(self, batch: OfficialBatch) -> torch.Tensor:
        h = self.encoder(batch)
        pair_mask = batch.pair_mask
        edge_repr = self.pair_encoder(batch.pair_xi) * pair_mask.unsqueeze(-1)
        for layer in self.layers:
            h, edge_repr = layer(h, edge_repr, batch.node_mask)
        if batch.task_type == "edge_binary":
            return self.heads.edge(h, batch, edge_repr)
        if batch.task_type in {"node_binary", "node_regression"}:
            return self.heads.node(h)
        if batch.task_type == "graph_regression":
            return self.heads.graph(h, batch)
        raise ValueError(batch.task_type)


class GNNPlusFFN(nn.Module):
    def __init__(self, dim: int, cfg: ScreenConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.act = activation_module(cfg.gnnplus_act)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hn = self.norm1(h) * mask.unsqueeze(-1)
        delta = self.dropout1(self.act(self.linear1(hn)))
        delta = self.dropout2(self.linear2(delta))
        h = (h + delta) * mask.unsqueeze(-1)
        return self.norm2(h) * mask.unsqueeze(-1)


class PaperGNNPlusEdgeEncoder(nn.Module):
    def __init__(self, cfg: ScreenConfig) -> None:
        super().__init__()
        dim = cfg.gnn_hidden_dim
        self.net = nn.Sequential(nn.Linear(2, dim), activation_module(cfg.gnnplus_act), nn.Linear(dim, dim))

    def forward(self, batch: OfficialBatch) -> torch.Tensor:
        mask = incoming_edge_mask(batch)
        return self.net(incoming_edge_features(batch)) * mask.unsqueeze(-1)


class PaperGCNEPlusLayer(nn.Module):
    def __init__(self, dim: int, cfg: ScreenConfig) -> None:
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))
        self.norm = nn.LayerNorm(dim)
        self.act = activation_module(cfg.gnnplus_act)
        self.dropout = nn.Dropout(cfg.dropout)
        self.residual = cfg.gnnplus_residual
        self.ffn = GNNPlusFFN(dim, cfg) if cfg.gnnplus_ffn else None

    def forward(self, h: torch.Tensor, edge_repr: torch.Tensor, batch: OfficialBatch) -> tuple[torch.Tensor, torch.Tensor]:
        mask = batch.node_mask
        route = incoming_edge_mask(batch)
        h_in = h
        x = self.lin(h)
        messages = torch.relu(x[:, None, :, :] + edge_repr) * route.unsqueeze(-1)
        out = messages.sum(dim=2) + self.bias
        out = self.norm(out) * mask.unsqueeze(-1)
        out = self.dropout(self.act(out)) * mask.unsqueeze(-1)
        h = (h_in + out if self.residual else out) * mask.unsqueeze(-1)
        if self.ffn is not None:
            h = self.ffn(h, mask)
        return h, edge_repr


class PaperGINEPlusLayer(nn.Module):
    def __init__(self, dim: int, cfg: ScreenConfig) -> None:
        super().__init__()
        self.nn = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.eps = nn.Parameter(torch.zeros(1))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout)
        self.residual = cfg.gnnplus_residual
        self.ffn = GNNPlusFFN(dim, cfg) if cfg.gnnplus_ffn else None

    def forward(self, h: torch.Tensor, edge_repr: torch.Tensor, batch: OfficialBatch) -> tuple[torch.Tensor, torch.Tensor]:
        mask = batch.node_mask
        route = incoming_edge_mask(batch)
        h_in = h
        messages = torch.relu(h[:, None, :, :] + edge_repr) * route.unsqueeze(-1)
        agg = messages.sum(dim=2)
        out = self.nn((1.0 + self.eps) * h + agg)
        out = self.dropout(self.act(out)) * mask.unsqueeze(-1)
        h = (h_in + out if self.residual else out) * mask.unsqueeze(-1)
        if self.ffn is not None:
            h = self.ffn(h, mask)
        return h, edge_repr


class PaperGatedGCNPlusLayer(nn.Module):
    def __init__(self, dim: int, cfg: ScreenConfig) -> None:
        super().__init__()
        self.a_proj = nn.Linear(dim, dim)
        self.b_proj = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)
        self.d_proj = nn.Linear(dim, dim)
        self.e_proj = nn.Linear(dim, dim)
        self.node_norm = nn.LayerNorm(dim)
        self.edge_norm = nn.LayerNorm(dim)
        self.act_x = activation_module(cfg.gnnplus_act)
        self.act_e = activation_module(cfg.gnnplus_act)
        self.dropout = nn.Dropout(cfg.dropout)
        self.residual = cfg.gnnplus_residual
        self.ffn = GNNPlusFFN(dim, cfg) if cfg.gnnplus_ffn else None

    def forward(self, h: torch.Tensor, edge_repr: torch.Tensor, batch: OfficialBatch) -> tuple[torch.Tensor, torch.Tensor]:
        mask = batch.node_mask
        route = incoming_edge_mask(batch)
        h_in = h
        e_in = edge_repr
        ax = self.a_proj(h)
        bx = self.b_proj(h)
        dx = self.d_proj(h)
        ex = self.e_proj(h)
        e_msg = dx[:, :, None, :] + ex[:, None, :, :] + self.c_proj(edge_repr)
        sigma = torch.sigmoid(e_msg) * route.unsqueeze(-1)
        numerator = (sigma * bx[:, None, :, :]).sum(dim=2)
        denominator = sigma.sum(dim=2)
        node_out = ax + numerator / (denominator + 1.0e-6)
        node_out = self.node_norm(node_out) * mask.unsqueeze(-1)
        edge_out = self.edge_norm(e_msg) * route.unsqueeze(-1)
        node_out = self.dropout(self.act_x(node_out)) * mask.unsqueeze(-1)
        edge_out = self.dropout(self.act_e(edge_out)) * route.unsqueeze(-1)
        h = (h_in + node_out if self.residual else node_out) * mask.unsqueeze(-1)
        edge_repr = (e_in + edge_out if self.residual else edge_out) * route.unsqueeze(-1)
        if self.ffn is not None:
            h = self.ffn(h, mask)
        return h, edge_repr


class PaperGNNPlusModel(nn.Module):
    def __init__(self, cfg: ScreenConfig, layer_cls: type[nn.Module], use_edge_head: bool) -> None:
        super().__init__()
        dim = cfg.gnn_hidden_dim
        self.encoder = NodeEncoder(cfg, dim)
        self.edge_encoder = PaperGNNPlusEdgeEncoder(cfg)
        self.layers = nn.ModuleList(layer_cls(dim, cfg) for _ in range(cfg.layers))
        self.heads = PredictionHeads(dim, edge_dim=dim if use_edge_head else 0)
        self.use_edge_head = use_edge_head

    def forward(self, batch: OfficialBatch) -> torch.Tensor:
        h = self.encoder(batch)
        edge_repr = self.edge_encoder(batch)
        for layer in self.layers:
            h, edge_repr = layer(h, edge_repr, batch)
        if batch.task_type == "edge_binary":
            return self.heads.edge(h, batch, edge_repr if self.use_edge_head else None)
        if batch.task_type in {"node_binary", "node_regression"}:
            return self.heads.node(h)
        if batch.task_type == "graph_regression":
            return self.heads.graph(h, batch)
        raise ValueError(batch.task_type)


class GCNPlusModel(PaperGNNPlusModel):
    def __init__(self, cfg: ScreenConfig) -> None:
        super().__init__(cfg, PaperGCNEPlusLayer, use_edge_head=False)


class GINPlusModel(PaperGNNPlusModel):
    def __init__(self, cfg: ScreenConfig) -> None:
        super().__init__(cfg, PaperGINEPlusLayer, use_edge_head=False)


class GatedGCNPlusModel(PaperGNNPlusModel):
    def __init__(self, cfg: ScreenConfig) -> None:
        super().__init__(cfg, PaperGatedGCNPlusLayer, use_edge_head=True)


def build_model(model_name: str, cfg: ScreenConfig) -> nn.Module:
    if model_name == "graphormer":
        return GraphormerModel(cfg)
    if model_name == "graphgps":
        return GraphGPSModel(cfg)
    if model_name == "grit":
        return GritModel(cfg, evolve_pairs=True)
    if model_name == "static_grit":
        return GritModel(cfg, evolve_pairs=False)
    if model_name == "gcn_plus":
        return GCNPlusModel(cfg)
    if model_name == "gin_plus":
        return GINPlusModel(cfg)
    if model_name == "gatedgcn_plus":
        return GatedGCNPlusModel(cfg)
    raise ValueError(model_name)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def compute_target_stats(dataset: Dataset[OfficialGraph]) -> Optional[dict[str, float]]:
    if len(dataset) == 0 or dataset[0].task_type != "graph_regression":
        return None
    targets = torch.stack([dataset[i].target.float().reshape(-1)[0] for i in range(len(dataset))])
    mean = float(targets.mean())
    std = float(targets.std(unbiased=False).clamp_min(1.0e-6))
    return {"mean": mean, "std": std}


def normalize_graph_target(target: torch.Tensor, target_stats: Optional[Mapping[str, float]]) -> torch.Tensor:
    if target_stats is None:
        return target.float()
    mean = float(target_stats["mean"])
    std = float(target_stats["std"])
    return (target.float() - mean) / max(std, 1.0e-6)


def denormalize_graph_target(pred: torch.Tensor, target_stats: Optional[Mapping[str, float]]) -> torch.Tensor:
    if target_stats is None:
        return pred.float()
    mean = float(target_stats["mean"])
    std = float(target_stats["std"])
    return pred.float() * max(std, 1.0e-6) + mean


def task_loss(
    pred: torch.Tensor,
    batch: OfficialBatch,
    pos_weight: Optional[torch.Tensor],
    target_stats: Optional[Mapping[str, float]],
) -> torch.Tensor:
    if batch.task_type == "edge_binary":
        return F.binary_cross_entropy_with_logits(pred, batch.edge_target.float(), pos_weight=pos_weight)
    if batch.task_type == "node_binary":
        return F.binary_cross_entropy_with_logits(pred[batch.node_mask], batch.node_target[batch.node_mask].float(), pos_weight=pos_weight)
    if batch.task_type == "node_regression":
        return F.mse_loss(torch.sigmoid(pred[batch.node_mask]), batch.node_target[batch.node_mask].float())
    if batch.task_type == "graph_regression":
        return F.mse_loss(pred, normalize_graph_target(batch.graph_target, target_stats))
    raise ValueError(batch.task_type)


def binary_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = (torch.sigmoid(logits.float()) >= 0.5).float()
    target = target.float()
    tp = float(((pred == 1) & (target == 1)).sum())
    fp = float(((pred == 1) & (target == 0)).sum())
    fn = float(((pred == 0) & (target == 1)).sum())
    tn = float(((pred == 0) & (target == 0)).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
    acc = (tp + tn) / max(1.0, tp + fp + fn + tn)
    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": acc}


def ranks(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    out = torch.empty_like(x, dtype=torch.float32)
    out[order] = torch.arange(x.numel(), dtype=torch.float32, device=x.device)
    return out


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    x = x.float()
    y = y.float()
    xm = x.mean()
    ym = y.mean()
    vx = ((x - xm) ** 2).sum()
    vy = ((y - ym) ** 2).sum()
    if float(vx) <= 0 or float(vy) <= 0:
        return float("nan")
    return float(((x - xm) * (y - ym)).sum() / torch.sqrt(vx * vy))


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.float()
    target = target.float()
    err = pred - target
    mse = float((err.square()).mean()) if err.numel() else float("nan")
    mae = float(err.abs().mean()) if err.numel() else float("nan")
    ss_res = float(err.square().sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1.0e-12 else float("nan")
    spearman = pearson(ranks(pred), ranks(target)) if pred.numel() > 1 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2, "spearman": spearman}


def flow_metrics(pred_z: torch.Tensor, target_raw: torch.Tensor, target_stats: Optional[Mapping[str, float]]) -> dict[str, float]:
    pred_raw = denormalize_graph_target(pred_z, target_stats)
    raw = regression_metrics(pred_raw, target_raw)
    z_target = normalize_graph_target(target_raw, target_stats)
    z = regression_metrics(pred_z, z_target)
    rel = (pred_raw - target_raw).abs() / target_raw.abs().clamp_min(1.0)
    raw["raw_mae"] = raw["mae"]
    raw["target_z_mae"] = z["mae"]
    raw["relative_mae"] = float(rel.mean()) if rel.numel() else float("nan")
    return raw


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: Dataset[OfficialGraph],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    seed: int,
    pos_weight: Optional[torch.Tensor],
    target_stats: Optional[Mapping[str, float]],
) -> tuple[dict[str, float], float]:
    model.eval()
    loader = make_loader(dataset, batch_size, shuffle=False, seed=seed)
    preds = []
    targets = []
    losses = []
    start = time.time()
    for batch in loader:
        batch = batch.to(device)
        pw = pos_weight.to(device) if pos_weight is not None else None
        with torch.autocast(device_type="cuda", enabled=use_amp):
            pred = model(batch)
            loss = task_loss(pred, batch, pw, target_stats)
        losses.append(float(loss.detach().cpu()) * batch.num_graphs)
        if batch.task_type == "edge_binary":
            preds.append(pred.detach().cpu())
            targets.append(batch.edge_target.detach().cpu())
        elif batch.task_type == "node_binary":
            preds.append(pred[batch.node_mask].detach().cpu())
            targets.append(batch.node_target[batch.node_mask].detach().cpu())
        elif batch.task_type == "node_regression":
            preds.append(torch.sigmoid(pred[batch.node_mask]).detach().cpu())
            targets.append(batch.node_target[batch.node_mask].detach().cpu())
        elif batch.task_type == "graph_regression":
            preds.append(pred.detach().cpu())
            targets.append(batch.graph_target.detach().cpu())
    seconds = time.time() - start
    pred_all = torch.cat(preds) if preds else torch.empty(0)
    target_all = torch.cat(targets) if targets else torch.empty(0)
    task_type = dataset[0].task_type
    if task_type in {"edge_binary", "node_binary"}:
        metrics = binary_metrics(pred_all, target_all)
        metrics["primary"] = metrics["f1"]
        metrics["higher_is_better"] = 1.0
    else:
        metrics = flow_metrics(pred_all, target_all, target_stats) if task_type == "graph_regression" else regression_metrics(pred_all, target_all)
        metrics["primary"] = metrics["mae"]
        metrics["higher_is_better"] = 0.0
    metrics["loss"] = sum(losses) / max(1, len(dataset))
    metrics["eval_seconds"] = seconds
    metrics["eval_seconds_per_graph"] = seconds / max(1, len(dataset))
    return metrics, seconds


def compute_pos_weight(dataset: Dataset[OfficialGraph]) -> Optional[torch.Tensor]:
    task_type = dataset[0].task_type
    if task_type == "edge_binary":
        targets = torch.cat([dataset[i].target.float().reshape(-1) for i in range(len(dataset))])
    elif task_type == "node_binary":
        targets = torch.cat([dataset[i].target.float().reshape(-1) for i in range(len(dataset))])
    else:
        return None
    pos = float((targets == 1).sum())
    neg = float((targets == 0).sum())
    return torch.tensor([neg / max(1.0, pos)], dtype=torch.float32)


def validation_score(metrics: Mapping[str, float]) -> float:
    if float(metrics.get("higher_is_better", 0.0)) > 0.5:
        return -float(metrics["primary"])
    return float(metrics["primary"])


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({field for row in rows for field in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1.0e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def dataset_stats(dataset: Dataset[OfficialGraph]) -> dict[str, object]:
    nodes = [dataset[i].num_nodes for i in range(len(dataset))]
    edges = [int(dataset[i].edge_index.size(1)) for i in range(len(dataset))]
    return {
        "graphs": len(dataset),
        "task_type": dataset[0].task_type if len(dataset) else "unknown",
        "nodes_min": min(nodes) if nodes else 0,
        "nodes_max": max(nodes) if nodes else 0,
        "edges_mean": sum(edges) / max(1, len(edges)),
    }


def model_config_for(model_name: str, cfg: ScreenConfig, preset: str) -> ScreenConfig:
    if preset == "shared":
        return cfg
    if preset != "paper":
        raise ValueError(f"unknown model size preset {preset!r}")
    updates = MODEL_SIZE_PRESETS.get(model_name, {})
    return replace(cfg, **updates)


def task_base_name(task: str) -> str:
    for difficulty in DIFFICULTIES:
        suffix = f"_{difficulty}"
        if task.endswith(suffix):
            return task[: -len(suffix)]
    return task


def learning_rate_for(model_name: str, task: str, cfg: ScreenConfig) -> float:
    if cfg.lr is not None:
        return float(cfg.lr)
    if cfg.lr_policy == "fair_family":
        if model_name in {"graphormer", "graphgps", "grit", "static_grit"}:
            return FAIR_FAMILY_LRS["graph_transformer"]
        return FAIR_FAMILY_LRS["gnn_plus"]
    if cfg.lr_policy != "graphbench_table":
        raise ValueError(f"unknown lr_policy={cfg.lr_policy!r}")
    base = task_base_name(task)
    if model_name in {"graphormer", "graphgps", "grit", "static_grit"}:
        return GRAPH_TRANSFORMER_LRS[base]
    return GNN_PLUS_LRS[model_name][base]


def eval_batch_size_for(model_name: str, cfg: ScreenConfig) -> int:
    if cfg.eval_batch_size > 0:
        return cfg.eval_batch_size
    return EVAL_BATCH_SIZE_BY_MODEL[model_name]


def train_batch_size_for(model_name: str, cfg: ScreenConfig) -> int:
    if cfg.batch_size > 0:
        return cfg.batch_size
    return TRAIN_BATCH_SIZE_BY_MODEL[model_name]


def resolved_model_configs(models: Sequence[str], cfg: ScreenConfig, preset: str) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for model_name in models:
        model_cfg = model_config_for(model_name, cfg, preset)
        model = build_model(model_name, model_cfg)
        out[model_name] = {
            "config": asdict(model_cfg),
            "trainable_parameters": count_parameters(model),
        }
    return out


def run_signature(task: str, model_name: str, cfg: ScreenConfig, splits: Mapping[str, OfficialGraphDataset]) -> dict[str, object]:
    training_config = asdict(cfg)
    training_config["final_eval"] = False
    return {
        "run_name": RUN_NAME,
        "version": RUN_VERSION,
        "task": task,
        "model": model_name,
        "resolved_learning_rate": learning_rate_for(model_name, task, cfg),
        "resolved_train_batch_size": train_batch_size_for(model_name, cfg),
        "resolved_eval_batch_size": eval_batch_size_for(model_name, cfg),
        "config": training_config,
        "dataset_stats": {split: dataset_stats(dataset) for split, dataset in splits.items()},
    }


def signature_matches(path: Path, signature: Mapping[str, object]) -> bool:
    if not path.exists():
        return False
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return old.get("run_signature") == signature


def train_one(
    task: str,
    model_name: str,
    splits: Mapping[str, OfficialGraphDataset],
    cfg: ScreenConfig,
    output_root: Path,
    device: torch.device,
    force_retrain: bool,
    log=print,
) -> dict[str, object]:
    set_seed(cfg.seed)
    run_dir = output_root / RUN_NAME / task / model_name / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = RunLogger(run_dir / "run.log")
    signature = run_signature(task, model_name, cfg, splits)
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best.pt"
    metrics_path = run_dir / "metrics.csv"
    model = build_model(model_name, cfg).to(device)
    n_params = count_parameters(model)
    pos_weight = compute_pos_weight(splits["train"])
    target_stats = compute_target_stats(splits["train"])
    resolved_lr = learning_rate_for(model_name, task, cfg)
    train_batch_size = train_batch_size_for(model_name, cfg)
    eval_batch_size = eval_batch_size_for(model_name, cfg)
    use_amp = cfg.amp and device.type == "cuda"
    watch_graphs = deterministic_subset(splits["val"], min(cfg.val_watch_size, len(splits["val"])), cfg.split_seed + 4099)
    watch_dataset = OfficialGraphDataset(watch_graphs, task_name=task, split="val_watch")
    run_log(
        f"[run] task={task} model={model_name} seed={cfg.seed} params={n_params:,} "
        f"lr={resolved_lr:g} train_batch={train_batch_size} eval_batch={eval_batch_size} device={device}"
    )
    run_log(f"[data] stats={json.dumps({k: dataset_stats(v) for k, v in splits.items()}, sort_keys=True)}")
    if target_stats is not None:
        run_log(f"[target] train graph target normalization={target_stats}")
    if best_path.exists() and signature_matches(summary_path, signature) and not force_retrain:
        run_log("[resume] matching checkpoint found; skipping training")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=resolved_lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        scheduler = make_scheduler(optimizer, cfg.warmup_steps, cfg.max_steps)
        best_score = float("inf")
        best_step = 0
        global_step = 0
        epoch = 0
        major_checkpoints: list[str] = []
        window_loss = 0.0
        window_graphs = 0
        train_started = time.time()
        fields = ["step", "epoch", "lr", "train_loss", "val_primary", "val_loss", "val_f1", "val_mae", "seconds"]
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while global_step < cfg.max_steps:
                epoch += 1
                train_loader = make_loader(splits["train"], train_batch_size, shuffle=True, seed=cfg.seed + epoch)
                for batch in train_loader:
                    if global_step >= cfg.max_steps:
                        break
                    batch = batch.to(device)
                    pw = pos_weight.to(device) if pos_weight is not None else None
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        pred = model(batch)
                        loss = task_loss(pred, batch, pw, target_stats)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    global_step += 1
                    window_loss += float(loss.detach().cpu()) * batch.num_graphs
                    window_graphs += batch.num_graphs
                    should_eval = global_step % cfg.eval_every_steps == 0 or global_step == cfg.max_steps
                    if not should_eval:
                        continue
                    elapsed = time.time() - train_started
                    val_metrics, _ = evaluate_model(
                        model,
                        watch_dataset,
                        eval_batch_size,
                        device,
                        use_amp,
                        cfg.seed,
                        pos_weight,
                        target_stats,
                    )
                    score = validation_score(val_metrics)
                    row = {
                        "step": global_step,
                        "epoch": epoch,
                        "lr": optimizer.param_groups[0]["lr"],
                        "train_loss": window_loss / max(1, window_graphs),
                        "val_primary": val_metrics["primary"],
                        "val_loss": val_metrics["loss"],
                        "val_f1": val_metrics.get("f1", float("nan")),
                        "val_mae": val_metrics.get("mae", float("nan")),
                        "seconds": elapsed,
                    }
                    writer.writerow(row)
                    handle.flush()
                    train_graphs = global_step * train_batch_size
                    steps_per_sec = global_step / max(elapsed, 1.0e-9)
                    graphs_per_sec = train_graphs / max(elapsed, 1.0e-9)
                    peak_gb = (
                        torch.cuda.max_memory_allocated(device) / 1.0e9
                        if device.type == "cuda"
                        else float("nan")
                    )
                    run_log(
                        f"[step {global_step:05d}] epoch={epoch} train_loss={row['train_loss']:.5f} "
                        f"watch_primary={row['val_primary']:.5f} watch_loss={row['val_loss']:.5f} "
                        f"elapsed={row['seconds']:.1f}s steps/s={steps_per_sec:.3f} "
                        f"graphs/s={graphs_per_sec:.1f} peak_mem_gb={peak_gb:.2f}"
                    )
                    window_loss = 0.0
                    window_graphs = 0
                    improved = score < best_score - cfg.min_delta
                    if global_step >= cfg.min_checkpoint_step and improved:
                        best_score = score
                        best_step = global_step
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "step": global_step,
                                "epoch": epoch,
                                "val_metrics": val_metrics,
                                "run_signature": signature,
                                "trainable_parameters": n_params,
                                "target_stats": target_stats,
                            },
                            best_path,
                        )
                    if cfg.save_major_checkpoints and global_step >= cfg.min_checkpoint_step:
                        major_path = run_dir / f"checkpoint_step{global_step:05d}.pt"
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "step": global_step,
                                "epoch": epoch,
                                "val_metrics": val_metrics,
                                "run_signature": signature,
                                "trainable_parameters": n_params,
                                "target_stats": target_stats,
                            },
                            major_path,
                        )
                        major_checkpoints.append(str(major_path))
                        run_log(f"[checkpoint] saved {major_path.name}")
        if not best_path.exists():
            run_log("[checkpoint] no eligible best checkpoint was written; saving final model")
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": global_step,
                    "epoch": epoch,
                    "val_metrics": {},
                    "run_signature": signature,
                    "trainable_parameters": n_params,
                    "target_stats": target_stats,
                },
                best_path,
            )
        else:
            run_log(f"[checkpoint] selected best_step={best_step}")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    split_metrics = {}
    if cfg.final_eval:
        for split in ("train", "val", "test"):
            eval_size = eval_batch_size if split != "train" else train_batch_size
            metrics, _ = evaluate_model(model, splits[split], eval_size, device, use_amp, cfg.seed, pos_weight, target_stats)
            split_metrics[split] = metrics
    summary = {
        "task": task,
        "task_type": splits["train"][0].task_type,
        "model": model_name,
        "seed": cfg.seed,
        "trainable_parameters": n_params,
        "resolved_learning_rate": resolved_lr,
        "resolved_train_batch_size": train_batch_size,
        "resolved_eval_batch_size": eval_batch_size,
        "target_stats": target_stats,
        "final_eval": cfg.final_eval,
        "best_step": int(ckpt.get("step", 0)),
        "best_epoch": int(ckpt.get("epoch", 0)),
        "best_checkpoint": str(best_path),
        "major_checkpoints": [str(path) for path in sorted(run_dir.glob("checkpoint_step*.pt"))],
        "run_signature": signature,
    }
    for split, metrics in split_metrics.items():
        summary[f"{split}_metrics"] = metrics
    write_json(summary_path, summary)
    if cfg.final_eval:
        run_log(
            f"[done] val_primary={split_metrics['val']['primary']:.5f} "
            f"test_primary={split_metrics['test']['primary']:.5f}"
        )
    else:
        run_log(f"[done] final_eval=skipped best_checkpoint={best_path}")
    return summary


def aggregate_rows(summaries: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for summary in summaries:
        if not all(f"{split}_metrics" in summary for split in ("train", "val", "test")):
            continue
        grouped[(str(summary["task"]), str(summary["model"]))].append(summary)
    for (task, model), runs in sorted(grouped.items()):
        task_type = str(runs[0]["task_type"])
        row: dict[str, object] = {"task": task, "task_type": task_type, "model": model, "n_seeds": len(runs)}
        for split in ("train", "val", "test"):
            vals = [float(run[f"{split}_metrics"]["primary"]) for run in runs]
            mean = sum(vals) / len(vals)
            std = 0.0 if len(vals) == 1 else math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            row[f"{split}_primary_mean"] = mean
            row[f"{split}_primary_std"] = std
            for metric in ("f1", "accuracy", "mae", "raw_mae", "target_z_mae", "relative_mae", "mse", "r2", "spearman", "loss"):
                metric_vals = [float(run[f"{split}_metrics"].get(metric, float("nan"))) for run in runs]
                finite = [v for v in metric_vals if math.isfinite(v)]
                row[f"{split}_{metric}_mean"] = sum(finite) / len(finite) if finite else float("nan")
        rows.append(row)
    return rows


def task_gap_rows(agg: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_task: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in agg:
        by_task[str(row["task"])].append(row)
    gaps = []
    for task, rows in sorted(by_task.items()):
        task_type = str(rows[0]["task_type"])
        grit = next((row for row in rows if row["model"] == "grit"), None)
        gnns = [row for row in rows if row["model"] in {"gcn_plus", "gin_plus", "gatedgcn_plus"}]
        if grit is None or not gnns:
            continue
        if task_type in {"edge_binary", "node_binary"}:
            grit_score = float(grit["test_f1_mean"])
            best_gnn = max(gnns, key=lambda row: float(row["test_f1_mean"]))
            gnn_score = float(best_gnn["test_f1_mean"])
            advantage = grit_score - gnn_score
            metric = "f1"
        else:
            grit_score = float(grit["test_mae_mean"])
            best_gnn = min(gnns, key=lambda row: float(row["test_mae_mean"]))
            gnn_score = float(best_gnn["test_mae_mean"])
            advantage = gnn_score - grit_score
            metric = "mae_gap_gnn_minus_grit"
        gaps.append(
            {
                "task": task,
                "task_type": task_type,
                "metric": metric,
                "grit_score": grit_score,
                "best_gnn_model": best_gnn["model"],
                "best_gnn_score": gnn_score,
                "grit_advantage": advantage,
            }
        )
    return sorted(gaps, key=lambda row: float(row["grit_advantage"]), reverse=True)


def plot_screening(agg: Sequence[Mapping[str, object]], gaps: Sequence[Mapping[str, object]], output_dir: Path, log=print) -> None:
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping plots: {exc}")
        return
    tasks = sorted({str(row["task"]) for row in agg})
    models = [model for model in MODEL_NAMES if any(str(row["model"]) == model for row in agg)]
    score = torch.full((len(tasks), len(models)), float("nan"))
    for i, task in enumerate(tasks):
        for j, model in enumerate(models):
            row = next((r for r in agg if str(r["task"]) == task and str(r["model"]) == model), None)
            if row is None:
                continue
            if str(row["task_type"]) in {"edge_binary", "node_binary"}:
                score[i, j] = float(row["test_f1_mean"])
            else:
                score[i, j] = -float(row["test_mae_mean"])
    fig, ax = plt.subplots(figsize=(5.8, max(3.2, 0.42 * len(tasks))))
    im = ax.imshow(score.numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    for i in range(len(tasks)):
        for j in range(len(models)):
            val = float(score[i, j])
            if math.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_title("Official GraphBench screening score\nF1 for binary tasks, -MAE for regression tasks")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_dir / "screening_score_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if gaps:
        labels = [str(row["task"]) for row in gaps]
        vals = [float(row["grit_advantage"]) for row in gaps]
        fig, ax = plt.subplots(figsize=(7.0, max(3.2, 0.38 * len(labels))))
        colors = ["#4daf4a" if v > 0 else "#e41a1c" for v in vals]
        ax.barh(range(len(labels)), vals, color=colors)
        ax.axvline(0.0, color="black", linewidth=1.0)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("GRIT advantage over best GNN+")
        ax.set_title("Task separation screen")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "grit_advantage_over_best_gnn.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def parse_csv_list(text: str) -> list[str]:
    return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]


def parse_seed_list(text: str) -> list[int]:
    return [int(part) for part in parse_csv_list(text)]


def env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def require_path(value: Optional[Path], label: str, env_name: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required. Pass it explicitly or set {env_name}.")
    return value.expanduser()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HPC/SLURM GraphBench AlgoReas hard-OOD base-training runner.")
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", type=str, default="0,1,2,3")
    parser.add_argument("--split-seed", type=int, default=ScreenConfig.split_seed)
    parser.add_argument("--train-size", type=int, default=ScreenConfig.train_size)
    parser.add_argument("--val-size", type=int, default=ScreenConfig.val_size)
    parser.add_argument("--test-size", type=int, default=ScreenConfig.test_size)
    parser.add_argument("--batch-size", type=int, default=ScreenConfig.batch_size, help="0 uses the A100-oriented per-model train batch table.")
    parser.add_argument("--eval-batch-size", type=int, default=ScreenConfig.eval_batch_size, help="0 uses the A100-oriented per-model eval batch table.")
    parser.add_argument("--max-steps", type=int, default=ScreenConfig.max_steps)
    parser.add_argument("--warmup-steps", type=int, default=ScreenConfig.warmup_steps)
    parser.add_argument("--eval-every-steps", type=int, default=ScreenConfig.eval_every_steps)
    parser.add_argument("--min-checkpoint-step", type=int, default=ScreenConfig.min_checkpoint_step)
    parser.add_argument("--val-watch-size", type=int, default=ScreenConfig.val_watch_size)
    parser.add_argument("--no-major-checkpoints", action="store_true")
    parser.add_argument("--final-eval", action="store_true", help="Run full train/val/test evaluation after training. Default skips this for base-training jobs.")
    parser.add_argument("--min-delta", type=float, default=1.0e-5)
    parser.add_argument("--lr", type=float, default=None, help="Override the fixed GraphBench-derived task/model LR table.")
    parser.add_argument("--lr-policy", choices=("fair_family", "graphbench_table"), default=ScreenConfig.lr_policy)
    parser.add_argument("--weight-decay", type=float, default=ScreenConfig.weight_decay)
    parser.add_argument("--hidden-dim", type=int, default=ScreenConfig.hidden_dim)
    parser.add_argument("--gnn-hidden-dim", type=int, default=ScreenConfig.gnn_hidden_dim)
    parser.add_argument("--layers", type=int, default=ScreenConfig.layers)
    parser.add_argument("--heads", type=int, default=ScreenConfig.heads)
    parser.add_argument("--dropout", type=float, default=ScreenConfig.dropout)
    parser.add_argument("--attn-dropout", type=float, default=ScreenConfig.attn_dropout)
    parser.add_argument("--model-size-preset", choices=("paper", "shared"), default="paper")
    parser.add_argument("--gnnplus-act", choices=("gelu", "relu"), default=ScreenConfig.gnnplus_act)
    parser.add_argument("--no-gnnplus-residual", action="store_true")
    parser.add_argument("--no-gnnplus-ffn", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=env_path("GRAPHBENCH_DATASET_ROOT"))
    parser.add_argument("--pe-cache-root", type=Path, default=env_path("GRAPHBENCH_PE_CACHE_ROOT"))
    parser.add_argument("--pe-cache-namespace", default="base", help="Subdirectory namespace under RUN_VERSION, e.g. base or sizegen_n256.")
    parser.add_argument("--pe-cache-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--precompute-pe-only", action="store_true", help="Build reusable SPD/RWSE/RRWP caches for selected tasks, then exit.")
    parser.add_argument("--force-recompute-pe", action="store_true")
    parser.add_argument("--no-build-missing-pe-cache", action="store_true", help="Use PE caches only when already present; otherwise compute PE in collate.")
    parser.add_argument("--require-pe-cache", action="store_true", help="Fail if a selected split lacks a compatible PE cache.")
    parser.add_argument("--output-root", type=Path, default=env_path("GRAPHBENCH_OUTPUT_ROOT"))
    parser.add_argument("--num-workers", type=int, default=DATALOADER_NUM_WORKERS)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-reload-data", action="store_true")
    parser.add_argument("--array-index", type=int, default=None, help="Run one 0-based task/model/seed job from the Cartesian product.")
    parser.add_argument("--print-jobs", action="store_true", help="Print the task/model/seed job table and exit.")
    parser.add_argument("--skip-suite-summary", action="store_true", help="Skip top-level aggregate writes; use for parallel array jobs.")
    parser.add_argument("--prepare-data-only", action="store_true", help="Load/cache selected official splits, then exit before training.")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args(argv)


def cfg_from_args(args: argparse.Namespace, seed: int) -> ScreenConfig:
    cfg = ScreenConfig(
        seed=seed,
        split_seed=args.split_seed,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        eval_every_steps=args.eval_every_steps,
        min_checkpoint_step=args.min_checkpoint_step,
        val_watch_size=args.val_watch_size,
        save_major_checkpoints=not args.no_major_checkpoints,
        min_delta=args.min_delta,
        lr=args.lr,
        lr_policy=args.lr_policy,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        gnnplus_residual=not args.no_gnnplus_residual,
        gnnplus_ffn=not args.no_gnnplus_ffn,
        gnnplus_act=args.gnnplus_act,
        amp=not args.no_amp,
        final_eval=args.final_eval,
    )
    if args.fast_dev_run:
        cfg = replace(
            cfg,
            train_size=min(cfg.train_size, 256),
            val_size=min(cfg.val_size, 128),
            test_size=min(cfg.test_size, 128),
            batch_size=16,
            eval_batch_size=16,
            max_steps=min(cfg.max_steps, 4),
            warmup_steps=min(cfg.warmup_steps, 1),
            eval_every_steps=1,
            min_checkpoint_step=1,
            val_watch_size=min(cfg.val_watch_size, 32),
        )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    global DATALOADER_NUM_WORKERS, DATALOADER_PIN_MEMORY
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    DATALOADER_NUM_WORKERS = int(args.num_workers)
    DATALOADER_PIN_MEMORY = not args.no_pin_memory
    tasks = parse_csv_list(args.tasks)
    if args.include_calibration:
        tasks.extend([task for task in CALIBRATION_TASKS if task not in tasks])
    unknown = [task for task in tasks if task not in TASK_TYPES]
    if unknown:
        raise ValueError(f"unknown task(s): {unknown}")
    models = parse_csv_list(args.models)
    unknown_models = [model for model in models if model not in MODEL_NAMES]
    if unknown_models:
        raise ValueError(f"unknown model(s): {unknown_models}")
    seeds = parse_seed_list(args.seeds)
    if args.fast_dev_run:
        seeds = seeds[:1]
        tasks = tasks[:1]
    jobs = [(task, model, seed) for task in tasks for model in models for seed in seeds]
    if args.print_jobs:
        for idx, (task, model, seed) in enumerate(jobs):
            print(f"{idx}\t{task}\t{model}\tseed{seed}")
        return
    if args.array_index is not None:
        if args.array_index < 0 or args.array_index >= len(jobs):
            raise IndexError(f"array-index {args.array_index} out of range for {len(jobs)} jobs")
        task, model, seed = jobs[args.array_index]
        tasks = [task]
        models = [model]
        seeds = [seed]
        args.skip_suite_summary = True
    dataset_root = require_path(args.dataset_root, "--dataset-root", "GRAPHBENCH_DATASET_ROOT")
    pe_cache_root = require_path(args.pe_cache_root, "--pe-cache-root", "GRAPHBENCH_PE_CACHE_ROOT")
    output_root = require_path(args.output_root, "--output-root", "GRAPHBENCH_OUTPUT_ROOT")
    suite_dir = output_root / RUN_NAME
    suite_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(suite_dir / "run.log")
    device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log(f"[setup] device={device} tasks={tasks} models={models} seeds={seeds}")
    log(f"[setup] dataset_root={dataset_root}")
    log(f"[setup] pe_cache_root={pe_cache_root}")
    log(f"[setup] output_root={output_root}")
    log(f"[setup] dataloader_workers={DATALOADER_NUM_WORKERS} pin_memory={DATALOADER_PIN_MEMORY}")
    if args.prepare_data_only:
        for task in tasks:
            cfg = cfg_from_args(args, seeds[0])
            _splits = load_official_graphbench_task(dataset_root, task, cfg, force_reload=args.force_reload_data, log=log)
        log("[done] prepared official GraphBench data/subset caches")
        return
    if args.precompute_pe_only:
        for task in tasks:
            cfg = cfg_from_args(args, seeds[0])
            splits = load_official_graphbench_task(dataset_root, task, cfg, force_reload=args.force_reload_data, log=log)
            _splits = attach_or_build_pe_cache(
                splits,
                pe_cache_root,
                cfg,
                namespace=args.pe_cache_namespace,
                dtype_name=args.pe_cache_dtype,
                force_recompute=args.force_recompute_pe,
                build_missing=True,
                require_present=False,
                log=log,
            )
        log("[done] prepared reusable PE caches")
        return
    all_summaries = []
    for task in tasks:
        base_cfg = cfg_from_args(args, seeds[0])
        splits = load_official_graphbench_task(dataset_root, task, base_cfg, force_reload=args.force_reload_data, log=log)
        pe_split_names = ("train", "val", "test") if base_cfg.final_eval else ("train", "val")
        pe_splits = attach_or_build_pe_cache(
            {split: splits[split] for split in pe_split_names},
            pe_cache_root,
            base_cfg,
            namespace=args.pe_cache_namespace,
            dtype_name=args.pe_cache_dtype,
            force_recompute=args.force_recompute_pe,
            build_missing=not args.no_build_missing_pe_cache,
            require_present=args.require_pe_cache,
            log=log,
        )
        splits = {**splits, **pe_splits}
        for seed in seeds:
            cfg = cfg_from_args(args, seed)
            for model_name in models:
                model_cfg = model_config_for(model_name, cfg, args.model_size_preset)
                summary = train_one(task, model_name, splits, model_cfg, output_root, device, force_retrain=args.force_retrain, log=log)
                all_summaries.append(summary)
    if not args.skip_suite_summary:
        agg = aggregate_rows(all_summaries)
        gaps = task_gap_rows(agg)
        flat_metrics = []
        for summary in all_summaries:
            for split in ("train", "val", "test"):
                metrics_key = f"{split}_metrics"
                if metrics_key not in summary:
                    continue
                flat_metrics.append(
                    {
                        "task": summary["task"],
                        "task_type": summary["task_type"],
                        "model": summary["model"],
                        "seed": summary["seed"],
                        "split": split,
                        **summary[metrics_key],
                    }
                )
        base_cfg = cfg_from_args(args, seeds[0])
        write_json(
            suite_dir / "resolved_config.json",
            {
                "tasks": tasks,
                "models": models,
                "seeds": seeds,
                "protocol_config": PROTOCOL_CONFIG,
                "model_size_preset": args.model_size_preset,
                "base_config": asdict(base_cfg),
                "model_configs": resolved_model_configs(models, base_cfg, args.model_size_preset),
                "version": RUN_VERSION,
            },
        )
        write_json(suite_dir / "summary.json", {"runs": all_summaries, "aggregate": agg, "task_gaps": gaps})
        write_csv_rows(suite_dir / "performance_metrics.csv", flat_metrics)
        write_csv_rows(suite_dir / "aggregate_metrics.csv", agg)
        write_csv_rows(suite_dir / "task_gaps.csv", gaps)
        if agg:
            plot_screening(agg, gaps, suite_dir, log=log)
        else:
            log("[summary] no final metrics found; aggregate plots skipped")
    else:
        log("[summary] skipped suite aggregate writes for parallel job")
    log("[done] Official GraphBench AlgoReas HPC run complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
