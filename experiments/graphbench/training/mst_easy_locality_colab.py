#!/usr/bin/env python3
"""Standalone GraphBench MST-Easy locality-ablation pilot for Colab.

This script trains small graph models on a self-contained GraphBench-style
`mst_easy` task. It trains on 16-node graphs and evaluates size generalisation
on 128-node and 256-node graphs.

Models:
- Graphormer, GraphGPS, GRIT, and CSA with full/k1/k2 attention support.
- GCN+ and GatedGCN+ local baselines.

CSA here is intentionally simple: standard dense self-attention with additive
static pair-feature bias, using the same static RRWP/SPD/edge-weight features
as GRIT, but without pairwise evolution or chromatic attention.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


TASK_NAME = "mst_easy_locality"
GT_FAMILIES = ("graphormer", "graphgps", "grit", "csa")
ATTENTION_MODES = ("full", "k1", "k2")
ATTENTION_MODEL_NAMES = tuple(f"{family}_{mode}" for family in GT_FAMILIES for mode in ATTENTION_MODES)
GNN_PLUS_MODEL_NAMES = ("gcn_plus", "gatedgcn_plus")
MODEL_NAMES = (*ATTENTION_MODEL_NAMES, *GNN_PLUS_MODEL_NAMES)

RW_STEPS = 4
SPD_CAP = 8
PAIR_RAW_DIM = 3 + (SPD_CAP + 2) + (RW_STEPS + 1) + 1

GENERATOR_NAMES = (
    "erdos-renyi",
    "newman-watts-strogatz",
    "barabasi-albert",
    "dual-barabasi-albert",
    "powerlaw-cluster",
    "stochastic-block-model",
)

CONFIG_TRAIN_MST = {
    "erdos-renyi": (1, 0.19),
    "newman-watts-strogatz": (1, 4, 0.2),
    "barabasi-albert": (1, 3),
    "dual-barabasi-albert": (1, 4, 2, 0.3),
    "powerlaw-cluster": (1, 5, 0.4),
    "stochastic-block-model": (1, [0.5, 0.5], [[0.5, 0.3], [0.3, 0.5]]),
}
CONFIG_TEST_MST = {
    "erdos-renyi": (1, 0.25),
    "newman-watts-strogatz": (1, 5, 0.8),
    "barabasi-albert": (1, 7),
    "dual-barabasi-albert": (1, 4, 3, 0.6),
    "powerlaw-cluster": (1, 7, 0.7),
    "stochastic-block-model": (1, [0.4, 0.6], [[0.25, 0.65], [0.65, 0.25]]),
}
SAMPLING_TRAIN_EASY = [1, 0, 1, 0, 1, 0]
SAMPLING_TEST_EASY = [1, 1, 1, 1, 1, 1]

MODEL_CONFIG_NOTES = {
    "graphormer": "2 layers, d=48, 4 heads; degree + SPD + edge presence/weight bias.",
    "graphgps": "2 layers, d=48, 4 heads; RWSE + dense GINE using edge weights.",
    "grit": "2 layers, d=48, 4 heads; RRWP/static pair features with pair evolution.",
    "csa": "2 layers, d=48, 4 heads; static pair-feature attention bias, no pair evolution.",
    "gcn_plus": "2 layers, d=56; local GCN+ with RWSE and edge-weight messages.",
    "gatedgcn_plus": "2 layers, d=56; local GatedGCN+ with RWSE and edge-weight edge states.",
}

PE_NOTES = {
    "graphormer": "degree + SPD attention bias + edge-weight bias",
    "graphgps": "RWSE node PE + degree; DenseGINE local branch with edge weights",
    "grit": "RRWP/SPD/edge-weight pair features + degree",
    "csa": "RRWP/SPD/edge-weight static pair features + degree",
    "gcn_plus": "RWSE node PE + degree + edge-weight messages",
    "gatedgcn_plus": "RWSE node PE + degree + learned edge states from edge weights",
}

ATTENTION_MODE_NOTES = {
    "full": "all valid node pairs; graph token enabled for Graphormer.",
    "k1": "<=1-hop node pairs only; Graphormer graph token attends only to itself.",
    "k2": "<=2-hop node pairs only; Graphormer graph token attends only to itself.",
}


@dataclass(frozen=True)
class PilotConfig:
    preset: str = TASK_NAME
    seed: int = 0
    train_graphs: int = 20000
    val_graphs: int = 3000
    test128_graphs: int = 2000
    test256_graphs: int = 400
    train_nodes: int = 16
    test128_nodes: int = 128
    test256_nodes: int = 256
    batch_size: int = 128
    eval_batch_size: int = 4
    max_epochs: int = 50
    patience: int = 12
    min_delta: float = 1.0e-5
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    hidden_dim: int = 48
    layers: int = 2
    heads: int = 4
    attention_mode: str = "full"
    dropout: float = 0.0
    attn_dropout: float = 0.1
    metric_graphs: int = 64
    metric_batch_size: int = 16
    metric_perms: int = 8
    metric_alpha_tau: float = 0.1
    amp: bool = True


@dataclass
class MSTGraph:
    edge_index: torch.Tensor
    edge_label: torch.Tensor
    edge_weight: torch.Tensor
    adj: torch.Tensor
    edge_weight_mat: torch.Tensor
    degree: torch.Tensor
    spd: torch.Tensor
    rwse: torch.Tensor
    rrwp: torch.Tensor
    true_mst_weight: float

    @property
    def num_nodes(self) -> int:
        return int(self.adj.size(0))


@dataclass
class MSTBatch:
    node_type: torch.Tensor
    node_mask: torch.Tensor
    adj: torch.Tensor
    edge_weight_mat: torch.Tensor
    degree: torch.Tensor
    spd: torch.Tensor
    rwse: torch.Tensor
    rrwp: torch.Tensor
    pair_xi: torch.Tensor
    edge_batch: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_label: torch.Tensor
    edge_weight: torch.Tensor
    graph_num_nodes: torch.Tensor
    true_mst_weight: torch.Tensor
    num_graphs: int
    max_nodes: int

    def to(self, device: torch.device) -> "MSTBatch":
        return MSTBatch(
            node_type=self.node_type.to(device),
            node_mask=self.node_mask.to(device),
            adj=self.adj.to(device),
            edge_weight_mat=self.edge_weight_mat.to(device),
            degree=self.degree.to(device),
            spd=self.spd.to(device),
            rwse=self.rwse.to(device),
            rrwp=self.rrwp.to(device),
            pair_xi=self.pair_xi.to(device),
            edge_batch=self.edge_batch.to(device),
            edge_src=self.edge_src.to(device),
            edge_dst=self.edge_dst.to(device),
            edge_label=self.edge_label.to(device),
            edge_weight=self.edge_weight.to(device),
            graph_num_nodes=self.graph_num_nodes.to(device),
            true_mst_weight=self.true_mst_weight.to(device),
            num_graphs=self.num_graphs,
            max_nodes=self.max_nodes,
        )

    @property
    def pair_mask(self) -> torch.Tensor:
        return self.node_mask[:, :, None] & self.node_mask[:, None, :]


class MSTDataset(Dataset[MSTGraph]):
    def __init__(self, graphs: Sequence[MSTGraph]) -> None:
        self.graphs = list(graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> MSTGraph:
        return self.graphs[index]


def graph_to_payload(graph: MSTGraph) -> dict[str, object]:
    return {
        "edge_index": graph.edge_index,
        "edge_label": graph.edge_label,
        "edge_weight": graph.edge_weight,
        "adj": graph.adj,
        "edge_weight_mat": graph.edge_weight_mat,
        "degree": graph.degree,
        "spd": graph.spd,
        "rwse": graph.rwse,
        "rrwp": graph.rrwp,
        "true_mst_weight": graph.true_mst_weight,
    }


def graph_from_payload(payload: object) -> MSTGraph:
    if isinstance(payload, Mapping):
        return MSTGraph(
            edge_index=payload["edge_index"],
            edge_label=payload["edge_label"],
            edge_weight=payload["edge_weight"],
            adj=payload["adj"],
            edge_weight_mat=payload["edge_weight_mat"],
            degree=payload["degree"],
            spd=payload["spd"],
            rwse=payload["rwse"],
            rrwp=payload["rrwp"],
            true_mst_weight=float(payload["true_mst_weight"]),
        )
    return MSTGraph(
        edge_index=payload.edge_index,
        edge_label=payload.edge_label,
        edge_weight=payload.edge_weight,
        adj=payload.adj,
        edge_weight_mat=payload.edge_weight_mat,
        degree=payload.degree,
        spd=payload.spd,
        rwse=payload.rwse,
        rrwp=payload.rrwp,
        true_mst_weight=float(payload.true_mst_weight),
    )


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def install_package_if_missing(import_name: str, package_name: str, log=print) -> None:
    try:
        __import__(import_name)
    except Exception:
        log(f"[deps] Installing {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mount_drive(mount_point: Path, enabled: bool, log=print) -> None:
    if not enabled:
        log("[drive] Drive mount disabled.")
        return
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        log("[drive] google.colab unavailable; continuing without mounting Drive.")
        return
    log(f"[drive] Mounting Google Drive at {mount_point}")
    drive.mount(str(mount_point), force_remount=False)


def resolve_device(requested: str, allow_cpu: bool = False) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA GPU is not available. In Colab, select Runtime > Change runtime type > GPU. "
            "For local smoke tests pass --allow-cpu."
        )
    device = torch.device(requested)
    if device.type != "cuda" and not allow_cpu:
        raise RuntimeError(f"Refusing to train on {device}; pass --allow-cpu for local smoke tests.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return device


def generate_base_graph(num_nodes: int, task_cfg: Mapping[str, tuple], sampling: Sequence[float], rng: random.Random):
    import networkx as nx

    generator = rng.choices(GENERATOR_NAMES, weights=sampling, k=1)[0]
    cfg = task_cfg[generator]
    seed = rng.randrange(2**31 - 1)
    if generator == "erdos-renyi":
        graph = nx.erdos_renyi_graph(num_nodes, float(cfg[1]), seed=seed)
    elif generator == "newman-watts-strogatz":
        k = min(num_nodes - 1, int(cfg[1]))
        graph = nx.newman_watts_strogatz_graph(num_nodes, k, float(cfg[2]), seed=seed)
    elif generator == "barabasi-albert":
        m = max(1, min(num_nodes - 1, int(cfg[1])))
        graph = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
    elif generator == "dual-barabasi-albert":
        m1 = max(1, min(num_nodes - 1, int(cfg[1])))
        m2 = max(1, min(num_nodes - 1, int(cfg[2])))
        graph = nx.dual_barabasi_albert_graph(num_nodes, m1, m2, float(cfg[3]), seed=seed)
    elif generator == "powerlaw-cluster":
        m = max(1, min(num_nodes - 1, int(cfg[1])))
        graph = nx.powerlaw_cluster_graph(num_nodes, m, float(cfg[2]), seed=seed)
    elif generator == "stochastic-block-model":
        first = max(1, int(float(cfg[1][0]) * num_nodes))
        second = max(1, num_nodes - first)
        graph = nx.stochastic_block_model([first, second], cfg[2], seed=seed)
    else:
        raise ValueError(generator)
    components = list(nx.connected_components(graph))
    if len(components) >= 2:
        for idx, component in enumerate(components):
            for _ in range(int(cfg[0])):
                choices = [j for j in range(len(components)) if j != idx]
                other = rng.choice(choices)
                a = rng.choice(list(component))
                b = rng.choice(list(components[other]))
                graph.add_edge(a, b)
    return graph


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
    return dist.clamp(max=cap + 1).to(torch.uint8)


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


def pair_features(adj: torch.Tensor, spd: torch.Tensor, rrwp: torch.Tensor, edge_weight_mat: torch.Tensor) -> torch.Tensor:
    n = int(adj.size(0))
    eye = torch.eye(n, dtype=torch.bool, device=adj.device)
    semantic = torch.zeros(n, n, 3, dtype=torch.float32, device=adj.device)
    semantic[..., 0] = eye.float()
    semantic[..., 1] = ((adj <= 0) & (~eye)).float()
    semantic[..., 2] = adj.float()
    spd_oh = F.one_hot(spd.long().clamp(max=SPD_CAP + 1), num_classes=SPD_CAP + 2).float()
    return torch.cat([semantic, spd_oh, rrwp.float(), edge_weight_mat.float().unsqueeze(-1)], dim=-1)


def mst_graph_to_record(num_nodes: int, graph_idx: int, split: str, seed: int) -> MSTGraph:
    import networkx as nx
    from networkx.algorithms import tree

    is_training = split in {"train", "val"}
    task_cfg = CONFIG_TRAIN_MST if is_training else CONFIG_TEST_MST
    sampling = SAMPLING_TRAIN_EASY if is_training else SAMPLING_TEST_EASY
    rng = random.Random(seed + 104729 * graph_idx + (0 if is_training else 17))
    while True:
        graph = generate_base_graph(num_nodes, task_cfg=task_cfg, sampling=sampling, rng=rng)
        if graph.number_of_edges() < num_nodes - 1:
            continue
        weight_dict = {edge: {"weight": round(rng.uniform(0, 10), 6)} for edge in graph.edges}
        weights = [attrs["weight"] for attrs in weight_dict.values()]
        if len(weights) == len(set(weights)):
            nx.set_edge_attributes(graph, weight_dict)
            break

    mst_edges = {tuple(sorted((u, v))) for u, v, _ in tree.minimum_spanning_edges(graph, weight="weight")}
    undirected_edges = sorted(tuple(sorted(edge)) for edge in graph.edges())
    true_mst_weight = sum(float(graph.edges[u, v]["weight"]) / 10.0 for u, v in mst_edges)

    directed_edges: list[tuple[int, int]] = []
    labels: list[float] = []
    edge_weights: list[float] = []
    for u, v in undirected_edges:
        y = 1.0 if (u, v) in mst_edges else 0.0
        w = float(graph.edges[u, v]["weight"]) / 10.0
        directed_edges.append((u, v))
        labels.append(y)
        edge_weights.append(w)
        directed_edges.append((v, u))
        labels.append(y)
        edge_weights.append(w)

    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    edge_weight_mat = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for u, v in undirected_edges:
        w = float(graph.edges[u, v]["weight"]) / 10.0
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        edge_weight_mat[u, v] = w
        edge_weight_mat[v, u] = w
    degree = adj.sum(dim=-1)
    spd = shortest_path_buckets(adj)
    rwse, rrwp = random_walk_features(adj)
    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    edge_label = torch.tensor(labels, dtype=torch.float32)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    return MSTGraph(
        edge_index=edge_index,
        edge_label=edge_label,
        edge_weight=edge_weight,
        adj=adj.to(torch.uint8),
        edge_weight_mat=edge_weight_mat.to(torch.float16),
        degree=degree.float(),
        spd=spd,
        rwse=rwse.to(torch.float16),
        rrwp=rrwp.to(torch.float16),
        true_mst_weight=float(true_mst_weight),
    )


def cache_path(cache_root: Path, split: str, nodes: int, count: int, seed: int) -> Path:
    return cache_root / f"mst_easy_{split}_n{nodes}_count{count}_seed{seed}_rw{RW_STEPS}.pt"


def load_or_generate_split(
    cache_root: Path,
    split: str,
    nodes: int,
    count: int,
    seed: int,
    force: bool,
    log=print,
) -> MSTDataset:
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_root, split, nodes, count, seed)
    if path.exists() and not force:
        log(f"[data] Loading cached {split} n={nodes}: {path}")
        raw = torch.load(path, map_location="cpu", weights_only=False)["graphs"]
        return MSTDataset([graph_from_payload(item) for item in raw])
    log(f"[data] Generating MST-Easy split={split} n={nodes} count={count}")
    graphs = [mst_graph_to_record(nodes, idx, split, seed) for idx in range(count)]
    torch.save(
        {"graphs": [graph_to_payload(graph) for graph in graphs], "split": split, "nodes": nodes, "count": count, "seed": seed},
        path,
    )
    return MSTDataset(graphs)


def dataset_stats(dataset: Dataset[MSTGraph]) -> dict[str, float]:
    n_graphs = len(dataset)
    nodes = [dataset[i].num_nodes for i in range(n_graphs)]
    edges = [int(dataset[i].edge_label.numel()) for i in range(n_graphs)]
    positives = [float(dataset[i].edge_label.sum()) for i in range(n_graphs)]
    weights = torch.cat([dataset[i].edge_weight for i in range(min(n_graphs, 512))]) if n_graphs else torch.empty(0)
    return {
        "graphs": float(n_graphs),
        "nodes_min": float(min(nodes) if nodes else 0),
        "nodes_max": float(max(nodes) if nodes else 0),
        "directed_edges_mean": float(sum(edges) / max(1, n_graphs)),
        "positive_directed_edges_mean": float(sum(positives) / max(1, n_graphs)),
        "positive_rate": float(sum(positives) / max(1, sum(edges))),
        "edge_weight_mean_sample": float(weights.float().mean()) if weights.numel() else 0.0,
    }


def collate_graphs(graphs: Sequence[MSTGraph]) -> MSTBatch:
    bsz = len(graphs)
    max_nodes = max(graph.num_nodes for graph in graphs)
    node_type = torch.zeros(bsz, max_nodes, dtype=torch.long)
    node_mask = torch.zeros(bsz, max_nodes, dtype=torch.bool)
    adj = torch.zeros(bsz, max_nodes, max_nodes, dtype=torch.float32)
    edge_weight_mat = torch.zeros(bsz, max_nodes, max_nodes, dtype=torch.float32)
    degree = torch.zeros(bsz, max_nodes, dtype=torch.float32)
    spd = torch.full((bsz, max_nodes, max_nodes), SPD_CAP + 1, dtype=torch.long)
    rwse = torch.zeros(bsz, max_nodes, RW_STEPS, dtype=torch.float32)
    rrwp = torch.zeros(bsz, max_nodes, max_nodes, RW_STEPS + 1, dtype=torch.float32)
    edge_batch = []
    edge_src = []
    edge_dst = []
    edge_label = []
    edge_weight = []
    true_mst_weight = []
    graph_num_nodes = []
    for graph_idx, graph in enumerate(graphs):
        n = graph.num_nodes
        node_type[graph_idx, :n] = 1
        node_mask[graph_idx, :n] = True
        adj[graph_idx, :n, :n] = graph.adj.float()
        edge_weight_mat[graph_idx, :n, :n] = graph.edge_weight_mat.float()
        degree[graph_idx, :n] = graph.degree.float()
        spd[graph_idx, :n, :n] = graph.spd.long()
        rwse[graph_idx, :n] = graph.rwse.float()
        rrwp[graph_idx, :n, :n] = graph.rrwp.float()
        e = graph.edge_index
        edge_batch.append(torch.full((e.size(1),), graph_idx, dtype=torch.long))
        edge_src.append(e[0])
        edge_dst.append(e[1])
        edge_label.append(graph.edge_label)
        edge_weight.append(graph.edge_weight)
        true_mst_weight.append(float(graph.true_mst_weight))
        graph_num_nodes.append(n)
    pair_xi = torch.stack([pair_features(adj[i], spd[i], rrwp[i], edge_weight_mat[i]) for i in range(bsz)], dim=0)
    return MSTBatch(
        node_type=node_type,
        node_mask=node_mask,
        adj=adj,
        edge_weight_mat=edge_weight_mat,
        degree=degree,
        spd=spd,
        rwse=rwse,
        rrwp=rrwp,
        pair_xi=pair_xi,
        edge_batch=torch.cat(edge_batch),
        edge_src=torch.cat(edge_src),
        edge_dst=torch.cat(edge_dst),
        edge_label=torch.cat(edge_label),
        edge_weight=torch.cat(edge_weight),
        graph_num_nodes=torch.tensor(graph_num_nodes, dtype=torch.long),
        true_mst_weight=torch.tensor(true_mst_weight, dtype=torch.float32),
        num_graphs=bsz,
        max_nodes=max_nodes,
    )


def make_loader(dataset: Dataset[MSTGraph], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collate_graphs,
        num_workers=0,
    )


def attention_support_mask(batch: MSTBatch, mode: str) -> torch.Tensor:
    if mode == "full":
        return batch.pair_mask
    if mode.startswith("k"):
        hops = int(mode[1:])
        return (batch.spd <= hops) & batch.pair_mask
    raise ValueError(f"unknown attention_mode={mode}")


def parse_model_name(model_name: str) -> tuple[str, Optional[str]]:
    if model_name in GNN_PLUS_MODEL_NAMES:
        return model_name, None
    for family in GT_FAMILIES:
        prefix = f"{family}_"
        if model_name.startswith(prefix):
            mode = model_name[len(prefix) :]
            if mode in ATTENTION_MODES:
                return family, mode
    raise ValueError(f"unknown model {model_name}")


def gather_node_pair(h: torch.Tensor, batch: MSTBatch) -> torch.Tensor:
    hb = h[batch.edge_batch]
    idx = torch.arange(hb.size(0), device=h.device)
    src = hb[idx, batch.edge_src]
    dst = hb[idx, batch.edge_dst]
    return torch.cat([src, dst, src * dst, torch.abs(src - dst)], dim=-1)


class EdgeHead(nn.Module):
    def __init__(self, dim: int, pair_dim: int = 0) -> None:
        super().__init__()
        in_dim = 4 * dim + pair_dim + 1
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, h: torch.Tensor, batch: MSTBatch, pair: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = gather_node_pair(h, batch)
        if pair is not None:
            pair_edge = pair[batch.edge_batch, batch.edge_src, batch.edge_dst]
            x = torch.cat([x, pair_edge], dim=-1)
        x = torch.cat([x, batch.edge_weight.to(h.dtype).unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


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

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        key_mask: torch.Tensor,
        attn_allow: Optional[torch.Tensor] = None,
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_bias
        scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        if attn_allow is not None:
            scores = scores.masked_fill(~attn_allow[:, None, :, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
        out = torch.matmul(self.attn_dropout(attn), v).transpose(1, 2).reshape(bsz, seq_len, dim)
        out = self.out_dropout(self.out_proj(out))
        if collect_attention:
            return out, attn
        return out


class TransformerBlockWithBias(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int, cfg: PilotConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithBias(dim, heads, cfg.attn_dropout, cfg.dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        key_mask: torch.Tensor,
        attn_allow: Optional[torch.Tensor] = None,
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.attn(self.norm1(x), attn_bias, key_mask, attn_allow, collect_attention)
        if collect_attention:
            delta, attn = out
        else:
            delta, attn = out, None
        x = x + delta
        x = x + self.ffn(self.norm2(x))
        if collect_attention:
            return x, attn
        return x


class MSTGraphormer(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.cfg = cfg
        self.node_encoder = nn.Embedding(2, dim, padding_idx=0)
        self.degree_encoder = nn.Embedding(512, dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, dim)
        self.spatial_encoder = nn.Embedding(SPD_CAP + 2, cfg.heads, padding_idx=0)
        self.edge_encoder = nn.Embedding(2, cfg.heads, padding_idx=0)
        self.weight_encoder = nn.Linear(1, cfg.heads, bias=False)
        self.virtual_distance = nn.Embedding(1, cfg.heads)
        self.emb_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(TransformerBlockWithBias(dim, dim * 2, cfg.heads, cfg) for _ in range(cfg.layers))
        self.edge_head = EdgeHead(dim)

    def build_bias(self, batch: MSTBatch) -> torch.Tensor:
        bsz, n = batch.node_type.shape
        heads = self.cfg.heads
        bias = torch.zeros(bsz, heads, n + 1, n + 1, device=batch.node_type.device)
        spatial = self.spatial_encoder(batch.spd.long().clamp(max=SPD_CAP + 1)).permute(0, 3, 1, 2)
        edge = self.edge_encoder(batch.adj.long().clamp(max=1)).permute(0, 3, 1, 2)
        weight = self.weight_encoder(batch.edge_weight_mat.unsqueeze(-1)).permute(0, 3, 1, 2)
        bias[:, :, 1:, 1:] = spatial + edge + weight
        token_bias = self.virtual_distance.weight.view(1, heads, 1)
        bias[:, :, 1:, 0] = bias[:, :, 1:, 0] + token_bias
        bias[:, :, 0, 1:] = bias[:, :, 0, 1:] + token_bias
        return bias

    def build_allow(self, batch: MSTBatch) -> Optional[torch.Tensor]:
        if self.cfg.attention_mode == "full":
            return None
        bsz, n = batch.node_type.shape
        allow = torch.zeros(bsz, n + 1, n + 1, dtype=torch.bool, device=batch.node_type.device)
        allow[:, 0, 0] = True
        allow[:, 1:, 1:] = attention_support_mask(batch, self.cfg.attention_mode)
        return allow

    def forward(self, batch: MSTBatch, collect_attention: bool = False):
        bsz = batch.node_type.size(0)
        degree = batch.degree.long().clamp(max=511)
        h = self.node_encoder(batch.node_type) + self.degree_encoder(degree + 1)
        token = self.graph_token.weight.unsqueeze(0).expand(bsz, -1, -1)
        h = self.emb_norm(torch.cat([token, h], dim=1))
        key_mask = torch.cat([torch.ones(bsz, 1, dtype=torch.bool, device=h.device), batch.node_mask], dim=1)
        bias = self.build_bias(batch).to(h.dtype)
        allow = self.build_allow(batch)
        layers = []
        for idx, layer in enumerate(self.layers):
            if collect_attention:
                h, attn = layer(h, bias, key_mask, allow, collect_attention=True)
                layers.append({"layer": idx, "attn": attn.detach(), "node_mask": key_mask.detach()})
            else:
                h = layer(h, bias, key_mask, allow)
        logits = self.edge_head(h[:, 1:], batch)
        if collect_attention:
            return logits, layers
        return logits


class DenseGINE(nn.Module):
    def __init__(self, dim: int, cfg: PilotConfig) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(2, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(dim * 2, dim))
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, h: torch.Tensor, batch: MSTBatch, mask: torch.Tensor) -> torch.Tensor:
        edge_input = torch.stack([batch.adj, batch.edge_weight_mat], dim=-1)
        edge_emb = self.edge_mlp(edge_input)
        source_h = h[:, None, :, :].expand(-1, h.size(1), -1, -1)
        messages = torch.relu(source_h + edge_emb)
        weights = batch.adj
        agg = (weights.unsqueeze(-1) * messages).sum(dim=2)
        out = self.mlp((1.0 + self.eps) * h + agg)
        out = self.norm(out.reshape(-1, out.size(-1))).view_as(out)
        return out * mask.unsqueeze(-1)


class GPSLayer(nn.Module):
    def __init__(self, dim: int, cfg: PilotConfig) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.local = DenseGINE(dim, cfg)
        self.local_norm = nn.LayerNorm(dim)
        self.global_attn = nn.MultiheadAttention(dim, cfg.heads, dropout=cfg.attn_dropout, batch_first=True)
        self.global_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        h: torch.Tensor,
        batch: MSTBatch,
        mask: torch.Tensor,
        attn_allow: Optional[torch.Tensor] = None,
        collect_attention: bool = False,
    ):
        h = h + self.local(self.local_norm(h), batch, mask)
        hn = self.global_norm(h)
        attn_mask = None
        if attn_allow is not None:
            bsz, n, _ = attn_allow.shape
            attn_mask = ~attn_allow[:, None, :, :].expand(bsz, self.heads, n, n)
            attn_mask = attn_mask.reshape(bsz * self.heads, n, n)
        out, attn = self.global_attn(
            hn,
            hn,
            hn,
            attn_mask=attn_mask,
            key_padding_mask=~mask,
            need_weights=collect_attention,
            average_attn_weights=False,
        )
        h = h + out * mask.unsqueeze(-1)
        h = h + self.ffn(self.ffn_norm(h)) * mask.unsqueeze(-1)
        if collect_attention:
            return h * mask.unsqueeze(-1), attn
        return h * mask.unsqueeze(-1)


class MSTGraphGPS(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.cfg = cfg
        self.node_encoder = nn.Embedding(2, dim, padding_idx=0)
        self.degree_proj = nn.Linear(1, dim, bias=False)
        self.pe_bn = nn.BatchNorm1d(RW_STEPS)
        self.pe_encoder = nn.Sequential(nn.Linear(RW_STEPS, 16), nn.ReLU(), nn.Linear(16, dim))
        self.layers = nn.ModuleList(GPSLayer(dim, cfg) for _ in range(cfg.layers))
        self.edge_head = EdgeHead(dim)

    def encode_pe(self, rwse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, n, steps = rwse.shape
        pe = self.pe_bn(rwse.reshape(-1, steps)).view(bsz, n, steps)
        return self.pe_encoder(pe) * mask.unsqueeze(-1)

    def forward(self, batch: MSTBatch, collect_attention: bool = False):
        mask = batch.node_mask
        h = (
            self.node_encoder(batch.node_type)
            + self.degree_proj(torch.log1p(batch.degree).unsqueeze(-1))
            + self.encode_pe(batch.rwse, mask)
        ) * mask.unsqueeze(-1)
        attn_allow = attention_support_mask(batch, self.cfg.attention_mode) if self.cfg.attention_mode != "full" else None
        layers = []
        for idx, layer in enumerate(self.layers):
            if collect_attention:
                h, attn = layer(h, batch, mask, attn_allow, collect_attention=True)
                layers.append({"layer": idx, "attn": attn.detach(), "node_mask": mask.detach()})
            else:
                h = layer(h, batch, mask, attn_allow)
        logits = self.edge_head(h, batch)
        if collect_attention:
            return logits, layers
        return logits


class GritLayer(nn.Module):
    def __init__(self, dim: int, cfg: PilotConfig, evolve_pairs: bool = True) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = dim // cfg.heads
        self.evolve_pairs = evolve_pairs
        self.node_norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.edge_to_bias = nn.Linear(dim, cfg.heads)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.out_dropout = nn.Dropout(cfg.dropout)
        self.node_norm2 = nn.LayerNorm(dim)
        self.node_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(cfg.dropout),
        )
        if evolve_pairs:
            self.edge_norm = nn.LayerNorm(dim)
            self.edge_mlp = nn.Sequential(
                nn.Linear(dim * 3, dim * 2),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(dim * 2, dim),
            )
        else:
            self.edge_norm = None
            self.edge_mlp = None

    def forward(
        self,
        h: torch.Tensor,
        edge_repr: torch.Tensor,
        mask: torch.Tensor,
        attn_allow: Optional[torch.Tensor] = None,
        collect_attention: bool = False,
    ):
        bsz, n, dim = h.shape
        hn = self.node_norm1(h)
        q = self.q_proj(hn).view(bsz, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hn).view(bsz, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hn).view(bsz, n, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.edge_to_bias(edge_repr).permute(0, 3, 1, 2)
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        if attn_allow is not None:
            scores = scores.masked_fill(~attn_allow[:, None, :, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
        delta = torch.matmul(self.attn_dropout(attn), v).transpose(1, 2).reshape(bsz, n, dim)
        h = h + self.out_dropout(self.out_proj(delta)) * mask.unsqueeze(-1)
        h = h + self.node_ffn(self.node_norm2(h)) * mask.unsqueeze(-1)
        if self.evolve_pairs:
            assert self.edge_mlp is not None and self.edge_norm is not None
            src = h.unsqueeze(2).expand(-1, -1, n, -1)
            dst = h.unsqueeze(1).expand(-1, n, -1, -1)
            edge_delta = self.edge_mlp(torch.cat([edge_repr, src, dst], dim=-1))
            pair_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
            edge_repr = self.edge_norm(edge_repr + edge_delta) * pair_mask
        if collect_attention:
            return h * mask.unsqueeze(-1), edge_repr, attn
        return h * mask.unsqueeze(-1), edge_repr


class MSTGrit(nn.Module):
    def __init__(self, cfg: PilotConfig, evolve_pairs: bool = True) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.cfg = cfg
        self.evolve_pairs = evolve_pairs
        self.node_encoder = nn.Embedding(2, dim, padding_idx=0)
        self.degree_proj = nn.Linear(1, dim, bias=False)
        self.pair_encoder = nn.Linear(PAIR_RAW_DIM, dim)
        self.layers = nn.ModuleList(GritLayer(dim, cfg, evolve_pairs=evolve_pairs) for _ in range(cfg.layers))
        self.edge_head = EdgeHead(dim, pair_dim=dim)

    def forward(self, batch: MSTBatch, collect_attention: bool = False):
        mask = batch.node_mask
        h = (
            self.node_encoder(batch.node_type)
            + self.degree_proj(torch.log1p(batch.degree).unsqueeze(-1))
        ) * mask.unsqueeze(-1)
        edge_repr = self.pair_encoder(batch.pair_xi) * batch.pair_mask.unsqueeze(-1)
        attn_allow = attention_support_mask(batch, self.cfg.attention_mode) if self.cfg.attention_mode != "full" else None
        layers = []
        for idx, layer in enumerate(self.layers):
            if collect_attention:
                h, edge_repr, attn = layer(h, edge_repr, mask, attn_allow, collect_attention=True)
                layers.append({"layer": idx, "attn": attn.detach(), "node_mask": mask.detach()})
            else:
                h, edge_repr = layer(h, edge_repr, mask, attn_allow)
        logits = self.edge_head(h, batch, edge_repr)
        if collect_attention:
            return logits, layers
        return logits


class GNNPlusEncoder(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.node_encoder = nn.Embedding(2, dim, padding_idx=0)
        self.degree_proj = nn.Linear(1, dim, bias=False)
        self.pe_bn = nn.BatchNorm1d(RW_STEPS)
        self.pe_encoder = nn.Sequential(nn.Linear(RW_STEPS, 16), nn.ReLU(), nn.Linear(16, dim))

    def forward(self, batch: MSTBatch) -> torch.Tensor:
        bsz, n, steps = batch.rwse.shape
        pe = self.pe_bn(batch.rwse.reshape(-1, steps)).view(bsz, n, steps)
        h = (
            self.node_encoder(batch.node_type)
            + self.degree_proj(torch.log1p(batch.degree).unsqueeze(-1))
            + self.pe_encoder(pe)
        )
        return h * batch.node_mask.unsqueeze(-1)


class GNNPlusFFN(nn.Module):
    def __init__(self, dim: int, cfg: PilotConfig) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h.reshape(-1, h.size(-1))).view_as(h)
        return (h + self.ffn(h_norm)) * mask.unsqueeze(-1)


class GCNPlusLayer(nn.Module):
    def __init__(self, dim: int, cfg: PilotConfig) -> None:
        super().__init__()
        self.self_proj = nn.Linear(dim, dim)
        self.neigh_proj = nn.Linear(dim, dim)
        self.edge_mlp = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ffn = GNNPlusFFN(dim, cfg)

    def forward(self, h: torch.Tensor, batch: MSTBatch) -> torch.Tensor:
        bsz, n, dim = h.shape
        eye = torch.eye(n, dtype=torch.bool, device=h.device).view(1, n, n)
        adj = batch.adj.bool() & batch.pair_mask
        route = adj | (eye & batch.pair_mask)
        edge_input = torch.stack([batch.adj, batch.edge_weight_mat, eye.expand(bsz, -1, -1).float()], dim=-1)
        edge_emb = self.edge_mlp(edge_input)
        source_h = h[:, None, :, :].expand(-1, n, -1, -1)
        messages = torch.relu(source_h + edge_emb)
        weights = route.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        agg = (weights.unsqueeze(-1) * messages).sum(dim=2)
        out = self.self_proj(h) + self.neigh_proj(agg)
        out = self.bn(out.reshape(-1, dim)).view_as(out)
        out = self.dropout(torch.relu(out)) * batch.node_mask.unsqueeze(-1)
        h = (h + out) * batch.node_mask.unsqueeze(-1)
        return self.ffn(h, batch.node_mask)


class MSTGCNPlus(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.encoder = GNNPlusEncoder(cfg)
        self.layers = nn.ModuleList(GCNPlusLayer(dim, cfg) for _ in range(cfg.layers))
        self.edge_head = EdgeHead(dim)

    def forward(self, batch: MSTBatch) -> torch.Tensor:
        h = self.encoder(batch)
        for layer in self.layers:
            h = layer(h, batch)
        return self.edge_head(h, batch)


class GatedGCNPlusLayer(nn.Module):
    def __init__(self, dim: int, cfg: PilotConfig) -> None:
        super().__init__()
        self.a_proj = nn.Linear(dim, dim)
        self.b_proj = nn.Linear(dim, dim)
        self.c_proj = nn.Linear(dim, dim)
        self.d_proj = nn.Linear(dim, dim)
        self.e_proj = nn.Linear(dim, dim)
        self.node_bn = nn.BatchNorm1d(dim)
        self.edge_bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ffn = GNNPlusFFN(dim, cfg)

    def forward(self, h: torch.Tensor, edge_repr: torch.Tensor, batch: MSTBatch):
        bsz, n, dim = h.shape
        eye = torch.eye(n, dtype=torch.bool, device=h.device).view(1, n, n)
        route_mask = (batch.adj.bool() | eye) & batch.pair_mask
        h_in = h
        e_in = edge_repr
        ah = self.a_proj(h)
        bh = self.b_proj(h)
        dh = self.d_proj(h)
        eh = self.e_proj(h)
        e_msg = dh[:, :, None, :] + eh[:, None, :, :] + self.c_proj(edge_repr)
        gate = torch.sigmoid(e_msg) * route_mask.unsqueeze(-1)
        denom = gate.sum(dim=2).clamp_min(1.0e-6)
        agg = (gate * bh[:, None, :, :]).sum(dim=2) / denom
        node_out = ah + agg
        node_out = self.node_bn(node_out.reshape(-1, dim)).view_as(node_out)
        node_out = self.dropout(torch.relu(node_out)) * batch.node_mask.unsqueeze(-1)
        h = (h_in + node_out) * batch.node_mask.unsqueeze(-1)
        edge_out = self.edge_bn(e_msg.reshape(-1, dim)).view_as(e_msg)
        edge_out = self.dropout(torch.relu(edge_out)) * route_mask.unsqueeze(-1)
        edge_repr = (e_in + edge_out) * route_mask.unsqueeze(-1)
        h = self.ffn(h, batch.node_mask)
        return h, edge_repr


class MSTGatedGCNPlus(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.encoder = GNNPlusEncoder(cfg)
        self.edge_encoder = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList(GatedGCNPlusLayer(dim, cfg) for _ in range(cfg.layers))
        self.edge_head = EdgeHead(dim, pair_dim=dim)

    def initial_edges(self, batch: MSTBatch) -> torch.Tensor:
        bsz, n = batch.node_type.shape
        eye = torch.eye(n, dtype=torch.bool, device=batch.node_type.device).view(1, n, n)
        route_mask = ((batch.adj.bool() | eye) & batch.pair_mask).unsqueeze(-1)
        edge_input = torch.stack([batch.adj, batch.edge_weight_mat, eye.expand(bsz, -1, -1).float()], dim=-1)
        return self.edge_encoder(edge_input) * route_mask

    def forward(self, batch: MSTBatch) -> torch.Tensor:
        h = self.encoder(batch)
        edge_repr = self.initial_edges(batch)
        for layer in self.layers:
            h, edge_repr = layer(h, edge_repr, batch)
        return self.edge_head(h, batch, edge_repr)


def build_model(model_name: str, cfg: PilotConfig) -> nn.Module:
    family, _mode = parse_model_name(model_name)
    if family == "graphormer":
        return MSTGraphormer(cfg)
    if family == "graphgps":
        return MSTGraphGPS(cfg)
    if family == "grit":
        return MSTGrit(cfg, evolve_pairs=True)
    if family == "csa":
        return MSTGrit(cfg, evolve_pairs=False)
    if family == "gcn_plus":
        return MSTGCNPlus(cfg)
    if family == "gatedgcn_plus":
        return MSTGatedGCNPlus(cfg)
    raise ValueError(f"unknown model {model_name}")


def model_config_for(model_name: str, cfg: PilotConfig) -> PilotConfig:
    family, mode = parse_model_name(model_name)
    updates: dict[str, object] = {}
    if mode is not None:
        updates["attention_mode"] = mode
    if family in {"gcn_plus", "gatedgcn_plus"}:
        updates["hidden_dim"] = 56
    return replace(cfg, **updates)


def model_family(model_name: str) -> str:
    return parse_model_name(model_name)[0]


def model_attention_mode(model_name: str) -> str:
    _family, mode = parse_model_name(model_name)
    return mode or "message_passing"


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def bce_pos_weight(dataset: Dataset[MSTGraph]) -> float:
    pos = sum(float(dataset[i].edge_label.sum()) for i in range(len(dataset)))
    total = sum(float(dataset[i].edge_label.numel()) for i in range(len(dataset)))
    neg = max(1.0, total - pos)
    return float(min(50.0, neg / max(1.0, pos)))


def binary_auroc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.float()
    if labels.numel() == 0 or labels.sum() == 0 or labels.sum() == labels.numel():
        return float("nan")
    order = torch.argsort(probs.float(), descending=True)
    y = labels[order]
    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1.0 - y, dim=0)
    tpr = torch.cat([torch.zeros(1), tp / labels.sum().clamp_min(1.0)])
    fpr = torch.cat([torch.zeros(1), fp / (labels.numel() - labels.sum()).clamp_min(1.0)])
    return float(torch.trapz(tpr, fpr))


def binary_auprc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.float()
    total_pos = labels.sum()
    if labels.numel() == 0 or total_pos == 0:
        return float("nan")
    order = torch.argsort(probs.float(), descending=True)
    y = labels[order]
    tp = torch.cumsum(y, dim=0)
    precision = tp / torch.arange(1, y.numel() + 1, dtype=torch.float32)
    recall = tp / total_pos.clamp_min(1.0)
    recall_prev = torch.cat([torch.zeros(1), recall[:-1]])
    return float(((recall - recall_prev) * precision).sum())


def metric_values(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    labels = labels.float()
    probs = torch.sigmoid(logits.float())
    pred = (probs >= threshold).float()
    tp = float(((pred == 1) & (labels == 1)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, tp + fn)
    f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    return {
        "f1": f1,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "positive_rate": float(labels.mean()) if labels.numel() else 0.0,
        "pred_positive_rate": float(pred.mean()) if pred.numel() else 0.0,
        "auroc": binary_auroc(probs, labels),
        "auprc": binary_auprc(probs, labels),
    }


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def f1_from_sets(pred: set[tuple[int, int]], true: set[tuple[int, int]]) -> tuple[float, float, float]:
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
    return f1, precision, recall


def undirected_scores(graph: MSTGraph, logits: torch.Tensor) -> list[tuple[tuple[int, int], float, float, float]]:
    buckets: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
    probs = torch.sigmoid(logits.float())
    for idx, (src, dst) in enumerate(graph.edge_index.t().tolist()):
        key = tuple(sorted((int(src), int(dst))))
        buckets.setdefault(key, []).append(
            (float(probs[idx]), float(graph.edge_label[idx]), float(graph.edge_weight[idx]))
        )
    out = []
    for key, values in buckets.items():
        score = sum(v[0] for v in values) / len(values)
        label = max(v[1] for v in values)
        weight = values[0][2]
        out.append((key, score, label, weight))
    return out


def graph_mst_metrics(graph: MSTGraph, logits: torch.Tensor) -> dict[str, float]:
    rows = undirected_scores(graph, logits)
    n = graph.num_nodes
    true = {edge for edge, _score, label, _weight in rows if label > 0.5}
    k = max(0, n - 1)
    ranked = sorted(rows, key=lambda row: row[1], reverse=True)
    topk = {edge for edge, _score, _label, _weight in ranked[:k]}
    topk_f1, topk_precision, topk_recall = f1_from_sets(topk, true)

    uf = UnionFind(n)
    pred_tree: set[tuple[int, int]] = set()
    pred_weight = 0.0
    for edge, _score, _label, weight in ranked:
        if uf.union(edge[0], edge[1]):
            pred_tree.add(edge)
            pred_weight += weight
            if len(pred_tree) == k:
                break
    kruskal_f1, kruskal_precision, kruskal_recall = f1_from_sets(pred_tree, true)
    weight_gap = (pred_weight - graph.true_mst_weight) / max(1.0e-8, graph.true_mst_weight)
    return {
        "topk_f1": topk_f1,
        "topk_precision": topk_precision,
        "topk_recall": topk_recall,
        "kruskal_f1": kruskal_f1,
        "kruskal_precision": kruskal_precision,
        "kruskal_recall": kruskal_recall,
        "pred_tree_weight": pred_weight,
        "true_tree_weight": graph.true_mst_weight,
        "tree_weight_gap": weight_gap,
        "tree_complete": float(len(pred_tree) == k),
    }


@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits = []
    labels = []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(batch)
        logits.append(out.detach().cpu())
        labels.append(batch.edge_label.detach().cpu())
    return torch.cat(logits), torch.cat(labels)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: Dataset[MSTGraph],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    seed: int,
) -> dict[str, float]:
    model.eval()
    loader = make_loader(dataset, batch_size, shuffle=False, seed=seed)
    logits_all = []
    labels_all = []
    graph_rows = []
    graph_offset = 0
    for batch in loader:
        batch_device = batch.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(batch_device).detach().cpu()
        logits_all.append(logits)
        labels_all.append(batch.edge_label.detach().cpu())
        edge_offset = 0
        for local_idx in range(batch.num_graphs):
            graph = dataset[graph_offset + local_idx]
            count = graph.edge_label.numel()
            graph_rows.append(graph_mst_metrics(graph, logits[edge_offset : edge_offset + count]))
            edge_offset += count
        graph_offset += batch.num_graphs
    edge_metrics = metric_values(torch.cat(logits_all), torch.cat(labels_all))
    graph_metrics = {}
    if graph_rows:
        for key in graph_rows[0].keys():
            vals = [float(row[key]) for row in graph_rows if math.isfinite(float(row[key]))]
            graph_metrics[key] = sum(vals) / max(1, len(vals))
    return edge_metrics | graph_metrics


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_curves(metrics_path: Path, output_path: Path, log=print) -> None:
    rows = read_csv_rows(metrics_path)
    if not rows:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping training curves: {exc}")
        return
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_f1 = [float(row["val_f1"]) for row in rows]
    val_kruskal = [float(row["val_kruskal_f1"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(epochs, train_loss, color="#377eb8")
    axes[0].set_title("Training loss")
    axes[1].plot(epochs, val_f1, color="#4daf4a")
    axes[1].set_title("Val edge F1")
    axes[2].plot(epochs, val_kruskal, color="#984ea3")
    axes[2].set_title("Val Kruskal F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_performance_summary(summaries: Sequence[Mapping[str, object]], output_path: Path, log=print) -> None:
    if not summaries:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping performance summary: {exc}")
        return
    models = [str(summary["model"]) for summary in summaries]
    metrics = [("f1", "Edge F1", 0.0, 1.0), ("kruskal_f1", "Kruskal F1", 0.0, 1.0), ("tree_weight_gap", "Tree weight gap", None, None)]
    fig, axes = plt.subplots(1, 3, figsize=(max(12.0, len(models) * 0.75), 4.2))
    x = torch.arange(len(models)).float()
    width = 0.25
    colors = {"val16": "#80b1d3", "test128": "#8dd3c7", "test256": "#fb8072"}
    for ax, (metric, title, y0, y1) in zip(axes, metrics):
        for label, offset in [("val16", -width), ("test128", 0.0), ("test256", width)]:
            values = [float(summary[f"{label}_metrics"][metric]) for summary in summaries]
            ax.bar((x + offset).numpy(), values, width=width, label=label, color=colors[label])
        ax.set_title(title)
        ax.set_xticks(x.numpy())
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        if y0 is not None:
            ax.set_ylim(y0, y1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle("MST-Easy size generalization")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_size_generalization_gap(summaries: Sequence[Mapping[str, object]], output_path: Path, log=print) -> None:
    if not summaries:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping size-generalization gap: {exc}")
        return
    models = [str(summary["model"]) for summary in summaries]
    x = torch.arange(len(models)).float()
    width = 0.32
    edge128 = [float(s["val16_metrics"]["f1"]) - float(s["test128_metrics"]["f1"]) for s in summaries]
    edge256 = [float(s["val16_metrics"]["f1"]) - float(s["test256_metrics"]["f1"]) for s in summaries]
    kr128 = [float(s["val16_metrics"]["kruskal_f1"]) - float(s["test128_metrics"]["kruskal_f1"]) for s in summaries]
    kr256 = [float(s["val16_metrics"]["kruskal_f1"]) - float(s["test256_metrics"]["kruskal_f1"]) for s in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(max(10.0, len(models) * 0.7), 4.0), sharex=True)
    axes[0].bar((x - width / 2).numpy(), edge128, width=width, label="val16-test128", color="#8dd3c7")
    axes[0].bar((x + width / 2).numpy(), edge256, width=width, label="val16-test256", color="#fb8072")
    axes[1].bar((x - width / 2).numpy(), kr128, width=width, label="val16-test128", color="#8dd3c7")
    axes[1].bar((x + width / 2).numpy(), kr256, width=width, label="val16-test256", color="#fb8072")
    axes[0].set_title("Edge F1 drop")
    axes[1].set_title("Kruskal F1 drop")
    for ax in axes:
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_xticks(x.numpy())
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_family_mode_heatmap(summaries: Sequence[Mapping[str, object]], output_path: Path, metric: str = "kruskal_f1", log=print) -> None:
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping family/mode heatmap: {exc}")
        return
    rows = [s for s in summaries if str(s["model"]) in ATTENTION_MODEL_NAMES]
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.5), sharey=True)
    for ax, split in zip(axes, ["val16", "test128", "test256"]):
        mat = torch.full((len(GT_FAMILIES), len(ATTENTION_MODES)), float("nan"))
        for summary in rows:
            family, mode = parse_model_name(str(summary["model"]))
            mat[GT_FAMILIES.index(family), ATTENTION_MODES.index(mode)] = float(summary[f"{split}_metrics"][metric])
        im = ax.imshow(mat.numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(split)
        ax.set_xticks(range(len(ATTENTION_MODES)))
        ax.set_xticklabels(ATTENTION_MODES)
        ax.set_yticks(range(len(GT_FAMILIES)))
        ax.set_yticklabels(GT_FAMILIES)
        for i in range(len(GT_FAMILIES)):
            for j in range(len(ATTENTION_MODES)):
                val = mat[i, j].item()
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03)
    fig.suptitle(f"{metric} by family and attention support")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def support_density(dataset: Dataset[MSTGraph], mode: str, max_graphs: int = 32) -> float:
    loader = make_loader(Subset(dataset, list(range(min(len(dataset), max_graphs)))), batch_size=min(8, max_graphs), shuffle=False, seed=0)
    num = 0.0
    den = 0.0
    for batch in loader:
        mask = attention_support_mask(batch, mode)
        num += float(mask.sum())
        den += float(batch.pair_mask.sum())
    return num / max(1.0, den)


def model_design_rows(model_names: Sequence[str], cfg: PilotConfig, splits: Mapping[str, MSTDataset]) -> list[dict[str, object]]:
    rows = []
    for model_name in model_names:
        model_cfg = model_config_for(model_name, cfg)
        family = model_family(model_name)
        mode = model_attention_mode(model_name)
        n_params = count_parameters(build_model(model_name, model_cfg))
        rows.append(
            {
                "model": model_name,
                "family": family,
                "layers": model_cfg.layers,
                "hidden_dim": model_cfg.hidden_dim,
                "heads": model_cfg.heads,
                "attention_support": mode,
                "pe_features": PE_NOTES[family],
                "parameters": n_params,
                "support_density_train16": support_density(splits["train16"], mode) if mode in ATTENTION_MODES else support_density(splits["train16"], "k1"),
                "support_density_test128": support_density(splits["test128"], mode) if mode in ATTENTION_MODES else support_density(splits["test128"], "k1"),
                "support_density_test256": support_density(splits["test256"], mode) if mode in ATTENTION_MODES else support_density(splits["test256"], "k1"),
            }
        )
    return rows


def write_and_log_model_design(model_names: Sequence[str], cfg: PilotConfig, splits: Mapping[str, MSTDataset], path: Path, log=print) -> None:
    rows = model_design_rows(model_names, cfg, splits)
    write_csv_rows(path, rows)
    log("[design] model | PE | support | params | support density train/128/256")
    for row in rows:
        log(
            f"[design] {row['model']} | {row['pe_features']} | {row['attention_support']} | "
            f"{int(row['parameters']):,} | {float(row['support_density_train16']):.3f}/"
            f"{float(row['support_density_test128']):.3f}/{float(row['support_density_test256']):.3f}"
        )


def make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1.0e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_signature(model_name: str, model_cfg: PilotConfig, splits: Mapping[str, MSTDataset]) -> dict[str, object]:
    return {
        "task": TASK_NAME,
        "model": model_name,
        "config": asdict(model_cfg),
        "dataset_stats": {name: dataset_stats(dataset) for name, dataset in splits.items()},
    }


def signature_matches(summary_path: Path, signature: Mapping[str, object]) -> bool:
    if not summary_path.exists():
        return False
    try:
        old = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return old.get("run_signature") == signature


def train_one_model(
    model_name: str,
    splits: Mapping[str, MSTDataset],
    cfg: PilotConfig,
    output_root: Path,
    device: torch.device,
    force_retrain: bool,
    log=print,
) -> dict[str, object]:
    set_seed(cfg.seed)
    run_dir = output_root / cfg.preset / model_name / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = RunLogger(run_dir / "run.log")
    model_cfg = model_config_for(model_name, cfg)
    family = model_family(model_name)
    mode = model_attention_mode(model_name)
    signature = run_signature(model_name, model_cfg, splits)
    run_log(f"[run] model={model_name} device={device}")
    run_log(f"[config] {MODEL_CONFIG_NOTES[family]}")
    if mode in ATTENTION_MODES:
        run_log(f"[attention] {mode}: {ATTENTION_MODE_NOTES[mode]}")
    write_json(run_dir / "config.json", asdict(model_cfg) | {"model": model_name, "family": family, "attention_mode": mode})
    write_json(run_dir / "dataset_stats.json", {name: dataset_stats(dataset) for name, dataset in splits.items()})

    train_loader = make_loader(splits["train16"], cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(splits["val16"], cfg.eval_batch_size, shuffle=False, seed=cfg.seed)
    model = build_model(model_name, model_cfg).to(device)
    n_params = count_parameters(model)
    run_log(f"[model] trainable_parameters={n_params:,}")
    pos_weight = bce_pos_weight(splits["train16"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    run_log(f"[loss] pos_weight={pos_weight:.4f}")

    best_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    use_amp = cfg.amp and device.type == "cuda"
    if best_path.exists() and signature_matches(summary_path, signature) and not force_retrain:
        run_log("[resume] found matching best.pt and summary.json; skipping training")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        best_epoch = int(ckpt.get("epoch", 0))
        best_val_score = float(ckpt.get("val_metrics", {}).get("kruskal_f1", float("nan")))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        total_steps = cfg.max_epochs * max(1, len(train_loader))
        warmup_steps = cfg.warmup_epochs * max(1, len(train_loader))
        scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
        best_val_score = -1.0
        best_epoch = 0
        bad_epochs = 0
        fields = [
            "epoch",
            "lr",
            "train_loss",
            "val_f1",
            "val_auprc",
            "val_kruskal_f1",
            "val_tree_weight_gap",
            "seconds",
        ]
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, cfg.max_epochs + 1):
                started = time.time()
                model.train()
                total_loss = 0.0
                total_edges = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        logits = model(batch)
                        loss = loss_fn(logits, batch.edge_label)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    total_loss += float(loss.detach().cpu()) * int(batch.edge_label.numel())
                    total_edges += int(batch.edge_label.numel())
                val_metrics = evaluate_model(model, splits["val16"], cfg.eval_batch_size, device, use_amp, cfg.seed)
                row = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": total_loss / max(1, total_edges),
                    "val_f1": val_metrics["f1"],
                    "val_auprc": val_metrics["auprc"],
                    "val_kruskal_f1": val_metrics["kruskal_f1"],
                    "val_tree_weight_gap": val_metrics["tree_weight_gap"],
                    "seconds": time.time() - started,
                }
                writer.writerow(row)
                handle.flush()
                run_log(
                    f"[epoch {epoch:03d}] loss={row['train_loss']:.5f} "
                    f"val_f1={row['val_f1']:.4f} val_kruskal={row['val_kruskal_f1']:.4f} "
                    f"gap={row['val_tree_weight_gap']:.4f} time={row['seconds']:.1f}s"
                )
                improved = val_metrics["kruskal_f1"] > best_val_score
                if improved:
                    best_val_score = val_metrics["kruskal_f1"]
                    best_epoch = epoch
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "epoch": epoch,
                            "val_metrics": val_metrics,
                            "trainable_parameters": n_params,
                            "run_signature": signature,
                        },
                        best_path,
                    )
                if val_metrics["kruskal_f1"] > best_val_score + cfg.min_delta:
                    bad_epochs = 0
                else:
                    bad_epochs = 0 if improved else bad_epochs + 1
                    if bad_epochs >= cfg.patience:
                        run_log(f"[early-stop] best_epoch={best_epoch}")
                        break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_metrics = evaluate_model(model, splits["val16"], cfg.eval_batch_size, device, use_amp, cfg.seed)
    test128_metrics = evaluate_model(model, splits["test128"], cfg.eval_batch_size, device, use_amp, cfg.seed)
    test256_metrics = evaluate_model(model, splits["test256"], cfg.eval_batch_size, device, use_amp, cfg.seed)
    summary = {
        "model": model_name,
        "seed": cfg.seed,
        "trainable_parameters": n_params,
        "best_epoch": best_epoch,
        "best_val_kruskal_f1": best_val_score,
        "val16_metrics": val_metrics,
        "test128_metrics": test128_metrics,
        "test256_metrics": test256_metrics,
        "best_checkpoint": str(best_path),
        "run_signature": signature,
    }
    write_json(summary_path, summary)
    plot_training_curves(metrics_path, run_dir / "training_curves.png", log=run_log)
    run_log(
        f"[done] val_kruskal={val_metrics['kruskal_f1']:.4f} "
        f"test128_kruskal={test128_metrics['kruskal_f1']:.4f} "
        f"test256_kruskal={test256_metrics['kruskal_f1']:.4f} "
        f"test256_gap={test256_metrics['tree_weight_gap']:.4f}"
    )
    return summary


def strip_colab_kernel_args(argv: Sequence[str]) -> list[str]:
    cleaned = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-f", "--f", "--file"}:
            skip_next = True
            continue
        if "jupyter/runtime/kernel-" in arg or (arg.endswith(".json") and "kernel-" in arg):
            continue
        cleaned.append(arg)
    return cleaned


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone MST-Easy size-generalization locality pilot.")
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-graphs", type=int, default=20000)
    parser.add_argument("--val-graphs", type=int, default=3000)
    parser.add_argument("--test128-graphs", type=int, default=2000)
    parser.add_argument("--test256-graphs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1.0e-5)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument(
        "--drive-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/graph_specialisation_metrics/mst_easy_locality"),
    )
    parser.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args(argv)
    return args


def cfg_from_args(args: argparse.Namespace) -> PilotConfig:
    cfg = PilotConfig(
        seed=args.seed,
        train_graphs=args.train_graphs,
        val_graphs=args.val_graphs,
        test128_graphs=args.test128_graphs,
        test256_graphs=args.test256_graphs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=not args.no_amp,
    )
    if args.fast_dev_run:
        cfg = PilotConfig(
            **{
                **asdict(cfg),
                "train_graphs": min(cfg.train_graphs, 128),
                "val_graphs": min(cfg.val_graphs, 64),
                "test128_graphs": min(cfg.test128_graphs, 32),
                "test256_graphs": min(cfg.test256_graphs, 16),
                "batch_size": min(cfg.batch_size, 32),
                "eval_batch_size": min(cfg.eval_batch_size, 2),
                "max_epochs": min(cfg.max_epochs, 2),
                "patience": 2,
            }
        )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = strip_colab_kernel_args(list(sys.argv[1:] if argv is None else argv))
    args = parse_args(argv)
    install_package_if_missing("networkx", "networkx")
    cfg = cfg_from_args(args)
    set_seed(cfg.seed)
    device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    mount_drive(args.drive_mount, enabled=not args.no_mount_drive)
    cache_root = args.cache_root or args.drive_dir / "cache"
    output_root = args.output_root or args.drive_dir / "results"
    suite_dir = output_root / cfg.preset
    suite_dir.mkdir(parents=True, exist_ok=True)
    top_log = RunLogger(suite_dir / "run.log")
    top_log(f"[setup] device={device} amp={cfg.amp and device.type == 'cuda'}")
    top_log(f"[setup] seed={cfg.seed}")
    top_log(f"[setup] cache_root={cache_root}")
    top_log(f"[setup] output_root={output_root}")
    splits = {
        "train16": load_or_generate_split(cache_root, "train", 16, cfg.train_graphs, cfg.seed, args.force_regenerate, top_log),
        "val16": load_or_generate_split(cache_root, "val", 16, cfg.val_graphs, cfg.seed + 11, args.force_regenerate, top_log),
        "test128": load_or_generate_split(cache_root, "test128", 128, cfg.test128_graphs, cfg.seed + 23, args.force_regenerate, top_log),
        "test256": load_or_generate_split(cache_root, "test256", 256, cfg.test256_graphs, cfg.seed + 37, args.force_regenerate, top_log),
    }
    write_json(suite_dir / "resolved_config.json", asdict(cfg))
    write_json(suite_dir / "dataset_stats.json", {name: dataset_stats(dataset) for name, dataset in splits.items()})
    models = list(MODEL_NAMES) if args.model == "all" else [args.model]
    write_and_log_model_design(models, cfg, splits, suite_dir / "model_design_table.csv", log=top_log)
    summaries = []
    for model_name in models:
        summaries.append(
            train_one_model(
                model_name,
                splits,
                cfg,
                output_root,
                device,
                force_retrain=args.force_retrain,
                log=top_log,
            )
        )
    write_json(suite_dir / "summary.json", {"runs": summaries})
    flat_rows = []
    for summary in summaries:
        for split in ["val16", "test128", "test256"]:
            flat_rows.append({"model": summary["model"], "split": split, **summary[f"{split}_metrics"]})
    write_csv_rows(suite_dir / "performance_metrics.csv", flat_rows)
    plot_performance_summary(summaries, suite_dir / "performance_summary.png", log=top_log)
    plot_size_generalization_gap(summaries, suite_dir / "size_generalization_gap.png", log=top_log)
    plot_family_mode_heatmap(summaries, suite_dir / "family_mode_kruskal_heatmap.png", metric="kruskal_f1", log=top_log)
    plot_family_mode_heatmap(summaries, suite_dir / "family_mode_edge_f1_heatmap.png", metric="f1", log=top_log)
    top_log("[done] MST-Easy locality pilot complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
