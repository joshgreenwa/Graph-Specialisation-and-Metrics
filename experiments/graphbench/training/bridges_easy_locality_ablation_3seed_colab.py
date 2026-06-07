#!/usr/bin/env python3
"""Standalone GraphBench Bridges-Easy locality-ablation 3-seed training run.

The script trains lightweight dense graph-transformer edge classifiers under
different attention-support masks:

- full attention
- <=1-hop attention
- <=2-hop attention
- <=4-hop attention
- graph edges plus deterministic expander chords

It uses a deterministic, self-contained Bridges-Easy generator matching the
GraphBench algorithmic-reasoning bridge-generation recipe, trains on 16-node
graphs, then evaluates on 128-node and 256-node graphs. It uses a symmetric
directed-edge target, so both directions of an undirected bridge are positive.

This version is for a stronger training sweep: larger train/validation samples,
three training seeds per model by default, aggregate performance visualisations,
and a copied best checkpoint per model across repeat trials. Alpha-weighted
specialisation metrics are disabled by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
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


TASK_NAME = "bridges_easy_locality_ablation_3seed"
GT_FAMILIES = ("graphormer", "graphgps", "grit")
ATTENTION_MODES = ("full", "k1", "k2", "k4", "expander")
ATTENTION_MODEL_NAMES = tuple(f"{family}_{mode}" for family in GT_FAMILIES for mode in ATTENTION_MODES)
GNN_PLUS_MODEL_NAMES = ("gcn_plus", "gatedgcn_plus")
MODEL_NAMES = (*ATTENTION_MODEL_NAMES, *GNN_PLUS_MODEL_NAMES)

MODEL_CONFIG_NOTES = {
    "graphormer": "4 layers, d=64, 4 heads; degree + SPD bias + binary edge bias.",
    "graphgps": "4 layers, d=64, 4 heads; DenseGINE + Transformer with RWSE and degree.",
    "grit": "4 layers, d=56, 4 heads; pair attention with RRWP edge states.",
    "gcn_plus": "4 layers, d=80; dense GCN+ with RWSE, edge integration, BN, residual, FFN.",
    "gatedgcn_plus": "4 layers, d=64; dense GatedGCN+ with RWSE, edge gates, BN, residual, FFN.",
}

PE_NOTES = {
    "graphormer": "degree encoder + SPD attention bias + binary edge bias",
    "graphgps": "RWSE node PE + degree; DenseGINE local branch",
    "grit": "RRWP pair PE + degree",
    "gcn_plus": "RWSE node PE + degree",
    "gatedgcn_plus": "RWSE node PE + degree + learned edge states",
}

ATTENTION_MODE_NOTES = {
    "full": "all valid node pairs; graph token enabled if present.",
    "k1": "<=1-hop node pairs only; graph token is not a global shortcut if present.",
    "k2": "<=2-hop node pairs only; graph token is not a global shortcut if present.",
    "k4": "<=4-hop node pairs only; graph token is not a global shortcut if present.",
    "expander": "graph edges + self + deterministic circular +/-1,+/-2,+/-4,+/-8 chords.",
}

NODE_SYMBOLS = 8
RW_STEPS = 4
SPD_CAP = 8
PAIR_RAW_DIM = 3 + (SPD_CAP + 2) + (RW_STEPS + 1)

M_POSITIONAL = "positional_score"
M_SYMBOLIC = "symbolic_score"
M_PE_INVARIANT = "pe_invariance"
M_PE_EQUIVARIANT = "pe_equivariance"
M_JOINT_EQUIVARIANT = "joint_equivariance"
M_POSITIONAL_CENTERED = "positional_score_centered"
M_SYMBOLIC_CENTERED = "symbolic_score_centered"
M_PE_INVARIANT_CENTERED = "pe_invariance_centered"
M_PE_EQUIVARIANT_CENTERED = "pe_equivariance_centered"
M_JOINT_EQUIVARIANT_CENTERED = "joint_equivariance_centered"

GENERATOR_NAMES = (
    "erdos-renyi",
    "newman-watts-strogatz",
    "barabasi-albert",
    "dual-barabasi-albert",
    "powerlaw-cluster",
    "stochastic-block-model",
)

CONFIG_TRAIN = {
    "erdos-renyi": (1, 0.11),
    "newman-watts-strogatz": (1, 1, 1.0),
    "barabasi-albert": (1, 1),
    "dual-barabasi-albert": (1, 3, 1, 0.07),
    "powerlaw-cluster": (1, 1, 0.5),
    "stochastic-block-model": (1, [0.5, 0.5], [[0.5, 0.01], [0.01, 0.5]]),
}
CONFIG_TEST = {
    "erdos-renyi": (1, 0.07),
    "newman-watts-strogatz": (1, 1, 1.0),
    "barabasi-albert": (1, 1),
    "dual-barabasi-albert": (1, 4, 1, 0.6),
    "powerlaw-cluster": (1, 1, 0.8),
    "stochastic-block-model": (1, [0.5, 0.5], [[0.05, 0.001], [0.001, 0.05]]),
}
SAMPLING_TRAIN_EASY = [1, 0, 1, 0, 1, 0]
SAMPLING_TEST_EASY = [1, 1, 1, 1, 1, 1]


@dataclass(frozen=True)
class PilotConfig:
    preset: str = TASK_NAME
    seed: int = 0
    data_seed: int = 0
    train_graphs: int = 10000
    val_graphs: int = 2000
    test128_graphs: int = 1000
    test256_graphs: int = 256
    train_nodes: int = 16
    test128_nodes: int = 128
    test256_nodes: int = 256
    batch_size: int = 128
    eval_batch_size: int = 4
    metric_graphs: int = 64
    metric_batch_size: int = 16
    metric_perms: int = 8
    metric_alpha_tau: float = 0.1
    max_epochs: int = 50
    patience: int = 15
    min_delta: float = 1.0e-4
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    hidden_dim: int = 64
    layers: int = 4
    heads: int = 4
    attention_mode: str = "full"
    pair_dim: int = 24
    pair_rank: int = 4
    dropout: float = 0.0
    attn_dropout: float = 0.1
    amp: bool = True


@dataclass
class BridgeGraph:
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_label: torch.Tensor
    adj: torch.Tensor
    degree: torch.Tensor
    spd: torch.Tensor
    rwse: torch.Tensor
    rrwp: torch.Tensor
    pair_xi: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.node_type.numel())


@dataclass
class BridgeBatch:
    node_type: torch.Tensor
    node_mask: torch.Tensor
    adj: torch.Tensor
    degree: torch.Tensor
    spd: torch.Tensor
    rwse: torch.Tensor
    rrwp: torch.Tensor
    pair_xi: torch.Tensor
    edge_batch: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_label: torch.Tensor
    num_graphs: int
    max_nodes: int

    def to(self, device: torch.device) -> "BridgeBatch":
        return BridgeBatch(
            node_type=self.node_type.to(device),
            node_mask=self.node_mask.to(device),
            adj=self.adj.to(device),
            degree=self.degree.to(device),
            spd=self.spd.to(device),
            rwse=self.rwse.to(device),
            rrwp=self.rrwp.to(device),
            pair_xi=self.pair_xi.to(device),
            edge_batch=self.edge_batch.to(device),
            edge_src=self.edge_src.to(device),
            edge_dst=self.edge_dst.to(device),
            edge_label=self.edge_label.to(device),
            num_graphs=self.num_graphs,
            max_nodes=self.max_nodes,
        )

    @property
    def pair_mask(self) -> torch.Tensor:
        return self.node_mask[:, :, None] & self.node_mask[:, None, :]


class BridgeDataset(Dataset[BridgeGraph]):
    def __init__(self, graphs: Sequence[BridgeGraph]) -> None:
        self.graphs = list(graphs)
        self.sizes = [graph.num_nodes for graph in self.graphs]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> BridgeGraph:
        return self.graphs[index]


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
        return
    except Exception:
        pass
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


def graphbench_like_bridge_graph(num_nodes: int, is_training: bool, rng: random.Random):
    import networkx as nx

    weights = SAMPLING_TRAIN_EASY if is_training else SAMPLING_TEST_EASY
    generator = rng.choices(GENERATOR_NAMES, weights=weights, k=1)[0]
    cfg = CONFIG_TRAIN[generator] if is_training else CONFIG_TEST[generator]
    connect_count = int(cfg[0])
    seed = rng.randrange(2**31 - 1)

    if generator == "erdos-renyi":
        graph = nx.erdos_renyi_graph(num_nodes, float(cfg[1]), seed=seed)
    elif generator == "newman-watts-strogatz":
        k = max(1, min(num_nodes - 1, int(cfg[1])))
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
        frac = cfg[1]
        first = max(1, min(num_nodes - 1, int(float(frac[0]) * num_nodes)))
        sizes = [first, num_nodes - first]
        graph = nx.stochastic_block_model(sizes, cfg[2], seed=seed)
    else:
        raise ValueError(generator)

    graph.add_nodes_from(range(num_nodes))
    components = list(nx.connected_components(graph))
    if len(components) >= 2:
        for idx, comp in enumerate(components):
            for _ in range(connect_count):
                choices = [j for j in range(len(components)) if j != idx]
                other = components[rng.choice(choices)]
                graph.add_edge(rng.choice(tuple(comp)), rng.choice(tuple(other)))
    return graph


def shortest_path_buckets(adj: torch.Tensor, cap: int = SPD_CAP) -> torch.Tensor:
    n = int(adj.size(0))
    dist = torch.full((n, n), cap + 1, dtype=torch.long)
    neighbors = [torch.nonzero(adj[i] > 0, as_tuple=False).flatten().tolist() for i in range(n)]
    for src in range(n):
        dist[src, src] = 0
        queue: deque[int] = deque([src])
        while queue:
            u = queue.popleft()
            if int(dist[src, u]) >= cap:
                continue
            for v in neighbors[u]:
                if dist[src, v] > dist[src, u] + 1:
                    dist[src, v] = dist[src, u] + 1
                    queue.append(v)
    return dist.clamp(max=cap + 1)


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


def pair_features(adj: torch.Tensor, spd: torch.Tensor, rrwp: torch.Tensor) -> torch.Tensor:
    n = int(adj.size(0))
    eye = torch.eye(n, dtype=torch.bool)
    semantic = torch.zeros(n, n, 3, dtype=torch.float32)
    semantic[..., 0] = eye.float()
    semantic[..., 1] = ((adj <= 0) & (~eye)).float()
    semantic[..., 2] = adj.float()
    spd_oh = F.one_hot(spd.clamp(max=SPD_CAP + 1), num_classes=SPD_CAP + 2).float()
    return torch.cat([semantic, spd_oh, rrwp.float()], dim=-1)


def expander_pair_mask(batch: BridgeBatch) -> torch.Tensor:
    bsz, n = batch.node_type.shape
    device = batch.node_type.device
    idx = torch.arange(n, device=device)
    delta = (idx[None, :] - idx[:, None]).remainder(max(1, n))
    offsets = {1, 2, 4, 8}
    allowed = torch.eye(n, dtype=torch.bool, device=device)
    for offset in offsets:
        if offset < n:
            allowed = allowed | (delta == offset) | (delta == (n - offset) % n)
    allowed = allowed.unsqueeze(0).expand(bsz, -1, -1)
    return (allowed | batch.adj.bool()) & batch.pair_mask


def attention_support_mask(batch: BridgeBatch, mode: str) -> torch.Tensor:
    if mode == "full":
        return batch.pair_mask
    if mode.startswith("k"):
        hops = int(mode[1:])
        return (batch.spd <= hops) & batch.pair_mask
    if mode == "expander":
        return expander_pair_mask(batch)
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


def bridge_graph_to_record(num_nodes: int, graph_idx: int, split: str, seed: int) -> BridgeGraph:
    import networkx as nx

    is_training = split in {"train", "val"}
    rng = random.Random(seed + 104729 * graph_idx + (0 if is_training else 17))
    graph = graphbench_like_bridge_graph(num_nodes, is_training=is_training, rng=rng)
    bridge_set = {tuple(sorted(edge)) for edge in nx.bridges(graph)}
    undirected_edges = sorted(tuple(sorted(edge)) for edge in graph.edges())

    directed_edges: list[tuple[int, int]] = []
    labels: list[float] = []
    for u, v in undirected_edges:
        y = 1.0 if (u, v) in bridge_set else 0.0
        directed_edges.append((u, v))
        labels.append(y)
        directed_edges.append((v, u))
        labels.append(y)
    if not directed_edges:
        directed_edges = [(0, 0)]
        labels = [0.0]

    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    for u, v in undirected_edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    degree = adj.sum(dim=-1)
    spd = shortest_path_buckets(adj)
    rwse, rrwp = random_walk_features(adj)
    pair_xi = pair_features(adj, spd, rrwp)

    node_rng = torch.Generator().manual_seed(seed + graph_idx * 8191 + num_nodes)
    node_type = torch.randint(1, NODE_SYMBOLS + 1, (num_nodes,), generator=node_rng)
    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    edge_label = torch.tensor(labels, dtype=torch.float32)
    return BridgeGraph(
        node_type=node_type,
        edge_index=edge_index,
        edge_label=edge_label,
        adj=adj,
        degree=degree,
        spd=spd,
        rwse=rwse,
        rrwp=rrwp,
        pair_xi=pair_xi,
    )


def cache_path(cache_root: Path, split: str, nodes: int, count: int, seed: int) -> Path:
    return cache_root / f"bridges_easy_{split}_n{nodes}_count{count}_seed{seed}_rw{RW_STEPS}.pt"


def load_or_generate_split(
    cache_root: Path,
    split: str,
    nodes: int,
    count: int,
    seed: int,
    force: bool,
    log=print,
) -> BridgeDataset:
    path = cache_path(cache_root, split, nodes, count, seed)
    if path.exists() and not force:
        log(f"[data] Loading cached {split} n={nodes}: {path}")
        return BridgeDataset(torch.load(path, map_location="cpu", weights_only=False)["graphs"])
    cache_root.mkdir(parents=True, exist_ok=True)
    log(f"[data] Generating Bridges-Easy split={split} n={nodes} count={count}")
    graphs = [bridge_graph_to_record(nodes, idx, split, seed) for idx in range(count)]
    torch.save({"graphs": graphs, "split": split, "nodes": nodes, "count": count}, path)
    return BridgeDataset(graphs)


def dataset_stats(dataset: Dataset[BridgeGraph]) -> dict[str, float]:
    n_graphs = len(dataset)
    nodes = [dataset[i].num_nodes for i in range(n_graphs)]
    edges = [int(dataset[i].edge_label.numel()) for i in range(n_graphs)]
    positives = [float(dataset[i].edge_label.sum()) for i in range(n_graphs)]
    total_edges = float(sum(edges))
    return {
        "graphs": float(n_graphs),
        "min_nodes": float(min(nodes)),
        "max_nodes": float(max(nodes)),
        "mean_nodes": float(sum(nodes) / max(1, n_graphs)),
        "mean_directed_edges": float(sum(edges) / max(1, n_graphs)),
        "positive_rate": float(sum(positives) / max(1.0, total_edges)),
    }


def collate_graphs(graphs: Sequence[BridgeGraph]) -> BridgeBatch:
    batch_size = len(graphs)
    max_nodes = max(graph.num_nodes for graph in graphs)
    node_type = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    adj = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.float32)
    degree = torch.zeros(batch_size, max_nodes, dtype=torch.float32)
    spd = torch.full((batch_size, max_nodes, max_nodes), SPD_CAP + 1, dtype=torch.long)
    rwse = torch.zeros(batch_size, max_nodes, RW_STEPS, dtype=torch.float32)
    rrwp = torch.zeros(batch_size, max_nodes, max_nodes, RW_STEPS + 1, dtype=torch.float32)
    pair_xi = torch.zeros(batch_size, max_nodes, max_nodes, PAIR_RAW_DIM, dtype=torch.float32)
    edge_batches = []
    edge_src = []
    edge_dst = []
    edge_label = []
    for bidx, graph in enumerate(graphs):
        n = graph.num_nodes
        node_type[bidx, :n] = graph.node_type
        node_mask[bidx, :n] = True
        adj[bidx, :n, :n] = graph.adj
        degree[bidx, :n] = graph.degree
        spd[bidx, :n, :n] = graph.spd
        rwse[bidx, :n] = graph.rwse
        rrwp[bidx, :n, :n] = graph.rrwp
        pair_xi[bidx, :n, :n] = graph.pair_xi
        e = graph.edge_index.size(1)
        edge_batches.append(torch.full((e,), bidx, dtype=torch.long))
        edge_src.append(graph.edge_index[0])
        edge_dst.append(graph.edge_index[1])
        edge_label.append(graph.edge_label)
    return BridgeBatch(
        node_type=node_type,
        node_mask=node_mask,
        adj=adj,
        degree=degree,
        spd=spd,
        rwse=rwse,
        rrwp=rrwp,
        pair_xi=pair_xi,
        edge_batch=torch.cat(edge_batches),
        edge_src=torch.cat(edge_src),
        edge_dst=torch.cat(edge_dst),
        edge_label=torch.cat(edge_label),
        num_graphs=batch_size,
        max_nodes=max_nodes,
    )


def model_support_density(
    dataset: Dataset[BridgeGraph],
    model_name: str,
    cfg: PilotConfig,
    max_graphs: int = 32,
) -> float:
    family = model_family(model_name)
    mode = model_attention_mode(model_name)
    total = 0.0
    allowed = 0.0
    for idx in range(min(len(dataset), max_graphs)):
        batch = collate_graphs([dataset[idx]])
        if family in GT_FAMILIES:
            support = attention_support_mask(batch, mode)
        else:
            n = batch.max_nodes
            eye = torch.eye(n, dtype=torch.bool).view(1, n, n)
            support = (batch.adj.bool() | eye) & batch.pair_mask
        allowed += float(support.sum())
        total += float(batch.pair_mask.sum())
    return allowed / max(1.0, total)


def model_design_rows(
    models: Sequence[str],
    cfg: PilotConfig,
    splits: Mapping[str, BridgeDataset],
) -> list[dict[str, object]]:
    rows = []
    for model_name in models:
        family = model_family(model_name)
        mode = model_attention_mode(model_name)
        model_cfg = model_config_for(model_name, cfg)
        params = count_parameters(build_model(model_name, model_cfg))
        rows.append(
            {
                "model": model_name,
                "family": family,
                "attention_or_message_support": mode,
                "support_note": ATTENTION_MODE_NOTES.get(mode, "local message passing on graph edges + self"),
                "pe": PE_NOTES[family],
                "hidden_dim": model_cfg.hidden_dim,
                "layers": model_cfg.layers,
                "heads": model_cfg.heads if family in GT_FAMILIES else "",
                "trainable_parameters": params,
                "train16_support_density": model_support_density(splits["train16"], model_name, model_cfg),
                "test128_support_density": model_support_density(splits["test128"], model_name, model_cfg),
                "test256_support_density": model_support_density(splits["test256"], model_name, model_cfg),
                "alpha_metrics": family in GT_FAMILIES,
            }
        )
    return rows


def write_and_log_model_design(
    models: Sequence[str],
    cfg: PilotConfig,
    splits: Mapping[str, BridgeDataset],
    output_path: Path,
    log=print,
) -> None:
    rows = model_design_rows(models, cfg, splits)
    write_csv_rows(output_path, rows)
    log("[design] model | PE | support | params | support density train/128/256")
    for row in rows:
        log(
            "[design] "
            f"{row['model']} | {row['pe']} | {row['attention_or_message_support']} | "
            f"{int(row['trainable_parameters']):,} | "
            f"{float(row['train16_support_density']):.3f}/"
            f"{float(row['test128_support_density']):.3f}/"
            f"{float(row['test256_support_density']):.3f}"
        )


def make_loader(dataset: Dataset[BridgeGraph], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collate_graphs,
        num_workers=0,
    )


def gather_node_pair(h: torch.Tensor, batch: BridgeBatch) -> torch.Tensor:
    hb = h[batch.edge_batch]
    src = hb[torch.arange(hb.size(0), device=h.device), batch.edge_src]
    dst = hb[torch.arange(hb.size(0), device=h.device), batch.edge_dst]
    return torch.cat([src, dst, src * dst, torch.abs(src - dst)], dim=-1)


class EdgeHead(nn.Module):
    def __init__(self, dim: int, pair_dim: int = 0) -> None:
        super().__init__()
        in_dim = 4 * dim + pair_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, h: torch.Tensor, batch: BridgeBatch, pair: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = gather_node_pair(h, batch)
        if pair is not None:
            pair_edge = pair[batch.edge_batch, batch.edge_src, batch.edge_dst]
            x = torch.cat([x, pair_edge], dim=-1)
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


class BridgeGraphormer(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.cfg = cfg
        self.node_encoder = nn.Embedding(NODE_SYMBOLS + 1, dim, padding_idx=0)
        self.degree_encoder = nn.Embedding(258, dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, dim)
        self.spatial_encoder = nn.Embedding(SPD_CAP + 2, cfg.heads, padding_idx=0)
        self.edge_encoder = nn.Embedding(2, cfg.heads, padding_idx=0)
        self.virtual_distance = nn.Embedding(1, cfg.heads)
        self.emb_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            TransformerBlockWithBias(dim, dim * 2, cfg.heads, cfg) for _ in range(cfg.layers)
        )
        self.edge_head = EdgeHead(dim)

    def build_bias(self, batch: BridgeBatch) -> torch.Tensor:
        bsz, n = batch.node_type.shape
        heads = self.cfg.heads
        bias = torch.zeros(bsz, heads, n + 1, n + 1, device=batch.node_type.device)
        spatial = self.spatial_encoder(batch.spd.clamp(max=SPD_CAP + 1)).permute(0, 3, 1, 2)
        edge = self.edge_encoder(batch.adj.long().clamp(max=1)).permute(0, 3, 1, 2)
        bias[:, :, 1:, 1:] = spatial + edge
        token_bias = self.virtual_distance.weight.view(1, heads, 1)
        bias[:, :, 1:, 0] = bias[:, :, 1:, 0] + token_bias
        bias[:, :, 0, 1:] = bias[:, :, 0, 1:] + token_bias
        return bias

    def build_allow(self, batch: BridgeBatch) -> Optional[torch.Tensor]:
        if self.cfg.attention_mode == "full":
            return None
        bsz, n = batch.node_type.shape
        allow = torch.zeros(bsz, n + 1, n + 1, dtype=torch.bool, device=batch.node_type.device)
        allow[:, 0, 0] = True
        allow[:, 1:, 1:] = attention_support_mask(batch, self.cfg.attention_mode)
        return allow

    def forward(
        self,
        batch: BridgeBatch,
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        bsz = batch.node_type.size(0)
        degree = batch.degree.long().clamp(max=257)
        h = self.node_encoder(batch.node_type) + self.degree_encoder(degree + 1)
        token = self.graph_token.weight.unsqueeze(0).expand(bsz, -1, -1)
        h = self.emb_norm(torch.cat([token, h], dim=1))
        key_mask = torch.cat(
            [torch.ones(bsz, 1, dtype=torch.bool, device=h.device), batch.node_mask],
            dim=1,
        )
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
        self.edge_encoder = nn.Embedding(2, dim, padding_idx=0)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        edge_emb = self.edge_encoder(torch.ones((), dtype=torch.long, device=h.device))
        messages = torch.relu(h + edge_emb.view(1, 1, -1))
        agg = torch.matmul(adj, messages)
        out = self.mlp((1.0 + self.eps) * h + agg)
        flat = out.reshape(-1, out.size(-1))
        out = self.norm(flat).view_as(out)
        return out * mask.unsqueeze(-1)


class GPSLayer(nn.Module):
    def __init__(self, dim: int, cfg: PilotConfig) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.local = DenseGINE(dim, cfg)
        self.local_norm = nn.LayerNorm(dim)
        self.global_attn = nn.MultiheadAttention(
            dim,
            cfg.heads,
            dropout=cfg.attn_dropout,
            batch_first=True,
        )
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
        adj: torch.Tensor,
        mask: torch.Tensor,
        attn_allow: Optional[torch.Tensor] = None,
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = h + self.local(self.local_norm(h), adj, mask)
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


class BridgeGraphGPS(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.cfg = cfg
        self.node_encoder = nn.Embedding(NODE_SYMBOLS + 1, dim, padding_idx=0)
        self.degree_proj = nn.Linear(1, dim, bias=False)
        self.pe_bn = nn.BatchNorm1d(RW_STEPS)
        self.pe_encoder = nn.Sequential(nn.Linear(RW_STEPS, 16), nn.ReLU(), nn.Linear(16, dim))
        self.layers = nn.ModuleList(GPSLayer(dim, cfg) for _ in range(cfg.layers))
        self.edge_head = EdgeHead(dim)

    def encode_pe(self, rwse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, n, steps = rwse.shape
        pe = self.pe_bn(rwse.reshape(-1, steps)).view(bsz, n, steps)
        return self.pe_encoder(pe) * mask.unsqueeze(-1)

    def forward(
        self,
        batch: BridgeBatch,
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        mask = batch.node_mask
        h = (
            self.node_encoder(batch.node_type)
            + self.degree_proj(torch.log1p(batch.degree).unsqueeze(-1))
            + self.encode_pe(batch.rwse, mask)
        )
        h = h * mask.unsqueeze(-1)
        attn_allow = None
        if self.cfg.attention_mode != "full":
            attn_allow = attention_support_mask(batch, self.cfg.attention_mode)
        layers = []
        for idx, layer in enumerate(self.layers):
            if collect_attention:
                h, attn = layer(h, batch.adj, mask, attn_allow, collect_attention=True)
                layers.append({"layer": idx, "attn": attn.detach(), "node_mask": mask.detach()})
            else:
                h = layer(h, batch.adj, mask, attn_allow)
        logits = self.edge_head(h, batch)
        if collect_attention:
            return logits, layers
        return logits


class GritLayer(nn.Module):
    def __init__(self, dim: int, cfg: PilotConfig) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = dim // cfg.heads
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
        self.edge_norm = nn.LayerNorm(dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_repr: torch.Tensor,
        mask: torch.Tensor,
        attn_allow: Optional[torch.Tensor] = None,
        collect_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        src = h.unsqueeze(2).expand(-1, -1, n, -1)
        dst = h.unsqueeze(1).expand(-1, n, -1, -1)
        edge_delta = self.edge_mlp(torch.cat([edge_repr, src, dst], dim=-1))
        pair_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
        edge_repr = self.edge_norm(edge_repr + edge_delta) * pair_mask
        if collect_attention:
            return h * mask.unsqueeze(-1), edge_repr, attn
        return h * mask.unsqueeze(-1), edge_repr


class BridgeGrit(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.cfg = cfg
        self.node_encoder = nn.Embedding(NODE_SYMBOLS + 1, dim, padding_idx=0)
        self.degree_proj = nn.Linear(1, dim, bias=False)
        self.rrwp_encoder = nn.Linear(RW_STEPS + 1, dim)
        self.layers = nn.ModuleList(GritLayer(dim, cfg) for _ in range(cfg.layers))
        self.edge_head = EdgeHead(dim, pair_dim=dim)

    def forward(
        self,
        batch: BridgeBatch,
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        mask = batch.node_mask
        h = (
            self.node_encoder(batch.node_type)
            + self.degree_proj(torch.log1p(batch.degree).unsqueeze(-1))
        ) * mask.unsqueeze(-1)
        edge_repr = self.rrwp_encoder(batch.rrwp) * batch.pair_mask.unsqueeze(-1)
        attn_allow = None
        if self.cfg.attention_mode != "full":
            attn_allow = attention_support_mask(batch, self.cfg.attention_mode)
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
        self.node_encoder = nn.Embedding(NODE_SYMBOLS + 1, dim, padding_idx=0)
        self.degree_proj = nn.Linear(1, dim, bias=False)
        self.pe_bn = nn.BatchNorm1d(RW_STEPS)
        self.pe_encoder = nn.Sequential(nn.Linear(RW_STEPS, 16), nn.ReLU(), nn.Linear(16, dim))

    def forward(self, batch: BridgeBatch) -> torch.Tensor:
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
        flat = h.reshape(-1, h.size(-1))
        h_norm = self.norm(flat).view_as(h)
        return (h + self.ffn(h_norm)) * mask.unsqueeze(-1)


class GCNPlusLayer(nn.Module):
    def __init__(self, dim: int, cfg: PilotConfig) -> None:
        super().__init__()
        self.self_proj = nn.Linear(dim, dim)
        self.neigh_proj = nn.Linear(dim, dim)
        self.edge_encoder = nn.Embedding(3, dim, padding_idx=0)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ffn = GNNPlusFFN(dim, cfg)

    def forward(self, h: torch.Tensor, batch: BridgeBatch) -> torch.Tensor:
        bsz, n, dim = h.shape
        eye = torch.eye(n, dtype=torch.bool, device=h.device).view(1, n, n)
        adj = batch.adj.bool() & batch.pair_mask
        adj_self = adj | (eye & batch.pair_mask)
        edge_type = adj.long()
        edge_type = torch.where(eye.expand(bsz, -1, -1), torch.full_like(edge_type, 2), edge_type)
        edge_emb = self.edge_encoder(edge_type)
        source_h = h[:, None, :, :].expand(-1, n, -1, -1)
        messages = torch.relu(source_h + edge_emb)
        weights = adj_self.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        agg = (weights.unsqueeze(-1) * messages).sum(dim=2)
        out = self.self_proj(h) + self.neigh_proj(agg)
        out = self.bn(out.reshape(-1, dim)).view_as(out)
        out = self.dropout(torch.relu(out)) * batch.node_mask.unsqueeze(-1)
        h = (h + out) * batch.node_mask.unsqueeze(-1)
        return self.ffn(h, batch.node_mask)


class BridgeGCNPlus(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.encoder = GNNPlusEncoder(cfg)
        self.layers = nn.ModuleList(GCNPlusLayer(dim, cfg) for _ in range(cfg.layers))
        self.edge_head = EdgeHead(dim)

    def forward(self, batch: BridgeBatch) -> torch.Tensor:
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

    def forward(
        self,
        h: torch.Tensor,
        edge_repr: torch.Tensor,
        batch: BridgeBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n, dim = h.shape
        eye = torch.eye(n, dtype=torch.bool, device=h.device).view(1, n, n)
        pair_mask = batch.pair_mask
        route_mask = (batch.adj.bool() | eye) & pair_mask
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


class BridgeGatedGCNPlus(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.encoder = GNNPlusEncoder(cfg)
        self.edge_encoder = nn.Embedding(3, dim, padding_idx=0)
        self.layers = nn.ModuleList(GatedGCNPlusLayer(dim, cfg) for _ in range(cfg.layers))
        self.edge_head = EdgeHead(dim, pair_dim=dim)

    def initial_edges(self, batch: BridgeBatch) -> torch.Tensor:
        bsz, n = batch.node_type.shape
        eye = torch.eye(n, dtype=torch.bool, device=batch.node_type.device).view(1, n, n)
        edge_type = batch.adj.long()
        edge_type = torch.where(eye.expand(bsz, -1, -1), torch.full_like(edge_type, 2), edge_type)
        route_mask = ((batch.adj.bool() | eye) & batch.pair_mask).unsqueeze(-1)
        return self.edge_encoder(edge_type) * route_mask

    def forward(self, batch: BridgeBatch) -> torch.Tensor:
        h = self.encoder(batch)
        edge_repr = self.initial_edges(batch)
        for layer in self.layers:
            h, edge_repr = layer(h, edge_repr, batch)
        return self.edge_head(h, batch, edge_repr)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


def bn_valid(x: torch.Tensor, mask: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
    out = torch.zeros_like(x)
    if bool(mask.any()):
        out[mask] = bn(x[mask])
    return out


class V24NodeEncoder(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.symbol_emb = nn.Embedding(NODE_SYMBOLS + 1, dim, padding_idx=0)
        self.degree_proj = nn.Linear(1, dim, bias=False)
        self.rwse_proj = nn.Linear(RW_STEPS, dim, bias=False)

    def forward(self, batch: BridgeBatch) -> torch.Tensor:
        h = (
            self.symbol_emb(batch.node_type)
            + self.degree_proj(torch.log1p(batch.degree).unsqueeze(-1))
            + self.rwse_proj(batch.rwse)
        )
        return h * batch.node_mask.unsqueeze(-1)


class V24Layer(nn.Module):
    def __init__(self, cfg: PilotConfig, use_dynamic_read: bool) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        heads = cfg.heads
        head_dim = dim // heads
        self.use_dynamic_read = use_dynamic_read
        self.heads = heads
        self.head_dim = head_dim
        read_dim = PAIR_RAW_DIM + (cfg.pair_dim if use_dynamic_read else 0)
        self.pre_attn_bn = nn.BatchNorm1d(dim)
        self.pre_ffn_bn = nn.BatchNorm1d(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.bias_weight = nn.Parameter(torch.empty(heads, read_dim))
        self.bias_gate = nn.Parameter(torch.ones(heads))
        self.value_weight = nn.Parameter(torch.empty(heads, head_dim, read_dim))
        self.theta1 = nn.Parameter(torch.ones(dim))
        self.theta2 = nn.Parameter(torch.zeros(dim))
        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(cfg.dropout),
        )
        if use_dynamic_read:
            self.pair_norm = RMSNorm(cfg.pair_dim)
            self.dynamic_bias_logit = nn.Parameter(torch.full((1,), -2.9444))
            self.dynamic_value_logit = nn.Parameter(torch.full((1,), -2.9444))
        else:
            self.pair_norm = None
            self.register_parameter("dynamic_bias_logit", None)
            self.register_parameter("dynamic_value_logit", None)
        nn.init.normal_(self.bias_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.value_weight, mean=0.0, std=0.02)

    def make_reads(self, pair_xi: torch.Tensor, z_pair: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_dynamic_read:
            return pair_xi, pair_xi
        assert z_pair is not None and self.pair_norm is not None
        z_read = self.pair_norm(z_pair)
        b_gate = torch.sigmoid(self.dynamic_bias_logit)
        v_gate = torch.sigmoid(self.dynamic_value_logit)
        return torch.cat([pair_xi, b_gate * z_read], dim=-1), torch.cat([pair_xi, v_gate * z_read], dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        z_pair: Optional[torch.Tensor],
        batch: BridgeBatch,
        collect_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        bsz, n, dim = h.shape
        h_in = h
        hn = bn_valid(h, batch.node_mask, self.pre_attn_bn)
        q = self.q_proj(hn).view(bsz, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hn).view(bsz, n, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hn).view(bsz, n, self.heads, self.head_dim).transpose(1, 2)
        dot = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias_in, value_in = self.make_reads(batch.pair_xi, z_pair)
        bias = torch.einsum("bijp,hp->bhij", bias_in, self.bias_weight)
        bias = bias * self.bias_gate.view(1, self.heads, 1, 1)
        logits = dot + bias
        logits = logits.masked_fill(~batch.node_mask[:, None, None, :], torch.finfo(logits.dtype).min)
        alpha = torch.softmax(logits, dim=-1)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
        alpha_used = self.attn_dropout(alpha)
        base = torch.matmul(alpha_used, v)
        pair_avg = torch.einsum("bhij,bijp->bhip", alpha_used, value_in)
        pair_msg = torch.einsum("bhip,hdp->bhid", pair_avg, self.value_weight)
        out = (base + pair_msg).transpose(1, 2).reshape(bsz, n, dim)
        out = self.o_proj(out) * batch.node_mask.unsqueeze(-1)
        degree_log = torch.log1p(batch.degree).unsqueeze(-1)
        scaled = out * self.theta1.view(1, 1, -1) + degree_log * out * self.theta2.view(1, 1, -1)
        h_half = (h_in + self.resid_dropout(scaled)) * batch.node_mask.unsqueeze(-1)
        h_pair_post = h_half
        ffn = self.ffn(bn_valid(h_half, batch.node_mask, self.pre_ffn_bn))
        h_out = (h_half + ffn) * batch.node_mask.unsqueeze(-1)
        return h_out, h_pair_post, alpha.detach() if collect_attention else None


class V24PairEvolution(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        pair_dim = cfg.pair_dim
        rank = cfg.pair_rank
        self.rank = rank
        self.h_norm = RMSNorm(dim)
        self.z_norm = RMSNorm(pair_dim)
        self.u_proj = nn.Linear(dim, rank, bias=False)
        self.v_proj = nn.Linear(dim, rank, bias=False)
        self.u_norm = RMSNorm(rank)
        self.v_norm = RMSNorm(rank)
        self.out_proj = nn.Linear(rank * rank, pair_dim)
        self.proposal_norm = RMSNorm(pair_dim)
        self.write_gate = nn.Linear(pair_dim + PAIR_RAW_DIM, 1)
        self.alpha_raw = nn.Parameter(torch.full((1,), -2.9706))
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, 0.8473)

    def forward(self, z_pair: torch.Tensor, h_post: torch.Tensor, batch: BridgeBatch) -> torch.Tensor:
        h = self.h_norm(h_post) * batch.node_mask.unsqueeze(-1)
        u = self.u_norm(self.u_proj(h))
        v = self.v_norm(self.v_proj(h))
        outer = (u[:, :, None, :, None] * v[:, None, :, None, :]).flatten(start_dim=-2)
        proposal = self.proposal_norm(self.out_proj(outer) * (self.rank ** -0.5))
        proposal = proposal * batch.pair_mask.unsqueeze(-1)
        z_gate = self.z_norm(z_pair) * batch.pair_mask.unsqueeze(-1)
        write = torch.sigmoid(self.write_gate(torch.cat([z_gate, batch.pair_xi], dim=-1)))
        alpha = F.softplus(self.alpha_raw)
        return (z_pair + alpha * write * proposal) * batch.pair_mask.unsqueeze(-1)


class BridgeV24(nn.Module):
    def __init__(self, cfg: PilotConfig, variant: str) -> None:
        super().__init__()
        if variant not in {"v24_static", "v24_evo"}:
            raise ValueError(variant)
        self.variant = variant
        self.use_pair_evolution = variant == "v24_evo"
        self.cfg = cfg
        self.node_encoder = V24NodeEncoder(cfg)
        self.layers = nn.ModuleList(
            V24Layer(cfg, use_dynamic_read=self.use_pair_evolution) for _ in range(cfg.layers)
        )
        self.evolvers = (
            nn.ModuleList(V24PairEvolution(cfg) for _ in range(cfg.layers))
            if self.use_pair_evolution
            else None
        )
        self.final_bn = nn.BatchNorm1d(cfg.hidden_dim)
        self.edge_head = EdgeHead(cfg.hidden_dim, pair_dim=cfg.pair_dim if self.use_pair_evolution else 0)

    def forward(
        self,
        batch: BridgeBatch,
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        h = self.node_encoder(batch)
        z_pair = (
            batch.pair_xi.new_zeros(batch.pair_xi.shape[:-1] + (self.cfg.pair_dim,))
            if self.use_pair_evolution
            else None
        )
        layers = []
        for idx, layer in enumerate(self.layers):
            h, h_pair_post, attn = layer(h, z_pair, batch, collect_attention=collect_attention)
            if collect_attention:
                layers.append({"layer": idx, "attn": attn, "node_mask": batch.node_mask.detach()})
            if self.use_pair_evolution:
                assert self.evolvers is not None and z_pair is not None
                z_pair = self.evolvers[idx](z_pair, h_pair_post, batch)
        h = bn_valid(h, batch.node_mask, self.final_bn)
        logits = self.edge_head(h, batch, z_pair)
        if collect_attention:
            return logits, layers
        return logits


def build_model(model_name: str, cfg: PilotConfig) -> nn.Module:
    family, _mode = parse_model_name(model_name)
    if family == "graphormer":
        return BridgeGraphormer(cfg)
    if family == "graphgps":
        return BridgeGraphGPS(cfg)
    if family == "grit":
        return BridgeGrit(cfg)
    if family == "gcn_plus":
        return BridgeGCNPlus(cfg)
    if family == "gatedgcn_plus":
        return BridgeGatedGCNPlus(cfg)
    raise ValueError(f"unknown model {model_name}")


def model_config_for(model_name: str, cfg: PilotConfig) -> PilotConfig:
    family, mode = parse_model_name(model_name)
    updates: dict[str, object] = {}
    if mode is not None:
        updates["attention_mode"] = mode
    if family == "grit":
        updates["hidden_dim"] = 56
    if family == "gcn_plus":
        updates["hidden_dim"] = 80
    return replace(cfg, **updates)


def model_family(model_name: str) -> str:
    return parse_model_name(model_name)[0]


def model_attention_mode(model_name: str) -> str:
    _family, mode = parse_model_name(model_name)
    return mode or "message_passing"


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def bce_pos_weight(dataset: Dataset[BridgeGraph]) -> float:
    pos = sum(float(dataset[i].edge_label.sum()) for i in range(len(dataset)))
    total = sum(float(dataset[i].edge_label.numel()) for i in range(len(dataset)))
    neg = max(1.0, total - pos)
    return float(min(50.0, neg / max(1.0, pos)))


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
    positive_rate = float(labels.mean()) if labels.numel() else 0.0
    pred_positive_rate = float(pred.mean()) if pred.numel() else 0.0
    return {
        "f1": f1,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "positive_rate": positive_rate,
        "pred_positive_rate": pred_positive_rate,
    }


DETOUR_BINS = ("bridge", "detour_2", "detour_3_4", "detour_5_8", "detour_9_16", "detour_17_plus")


def shortest_path_after_edge_removal(adj: torch.Tensor, src: int, dst: int) -> Optional[int]:
    n = int(adj.size(0))
    seen = [False] * n
    seen[src] = True
    queue: deque[tuple[int, int]] = deque([(src, 0)])
    while queue:
        node, dist = queue.popleft()
        for nbr in torch.nonzero(adj[node] > 0, as_tuple=False).flatten().tolist():
            if (node == src and nbr == dst) or (node == dst and nbr == src):
                continue
            if nbr == dst:
                return dist + 1
            if not seen[nbr]:
                seen[nbr] = True
                queue.append((nbr, dist + 1))
    return None


def detour_bin_for_edge(graph: BridgeGraph, src: int, dst: int) -> str:
    detour = shortest_path_after_edge_removal(graph.adj, src, dst)
    if detour is None:
        return "bridge"
    if detour <= 2:
        return "detour_2"
    if detour <= 4:
        return "detour_3_4"
    if detour <= 8:
        return "detour_5_8"
    if detour <= 16:
        return "detour_9_16"
    return "detour_17_plus"


def detour_bins_for_graph(graph: BridgeGraph) -> list[str]:
    cache = {}
    bins = []
    for src, dst in graph.edge_index.t().tolist():
        key = tuple(sorted((int(src), int(dst))))
        if key not in cache:
            cache[key] = detour_bin_for_edge(graph, key[0], key[1])
        bins.append(cache[key])
    return bins


@torch.no_grad()
def evaluate_detour_metrics(
    model: nn.Module,
    dataset: Dataset[BridgeGraph],
    split_name: str,
    model_name: str,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    seed: int,
) -> list[dict[str, object]]:
    loader = make_loader(dataset, batch_size, shuffle=False, seed=seed)
    buckets: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"logits": [], "labels": []} for name in DETOUR_BINS
    }
    graph_offset = 0
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(batch).detach().cpu()
        edge_offset = 0
        for local_idx in range(batch.num_graphs):
            graph = dataset[graph_offset + local_idx]
            count = graph.edge_label.numel()
            graph_logits = logits[edge_offset : edge_offset + count]
            graph_labels = graph.edge_label.float()
            edge_offset += count
            graph_bins = detour_bins_for_graph(graph)
            for bin_name in DETOUR_BINS:
                mask = torch.tensor([name == bin_name for name in graph_bins], dtype=torch.bool)
                if bool(mask.any()):
                    buckets[bin_name]["logits"].append(graph_logits[mask])
                    buckets[bin_name]["labels"].append(graph_labels[mask])
        graph_offset += batch.num_graphs
    rows = []
    for bin_name in DETOUR_BINS:
        if not buckets[bin_name]["logits"]:
            continue
        logits = torch.cat(buckets[bin_name]["logits"])
        labels = torch.cat(buckets[bin_name]["labels"])
        metrics = metric_values(logits, labels)
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "split": split_name,
                "edge_bin": bin_name,
                "n_edges": int(labels.numel()),
                **metrics,
            }
        )
    return rows


@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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


def transform_pair_reference(z: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    bsz, heads, n, _ = z.shape
    idx = perm_pos.to(z.device)
    row_idx = idx[:, None, :, None].expand(bsz, heads, n, n)
    z_rows = torch.gather(z, 2, row_idx)
    col_idx = idx[:, None, None, :].expand(bsz, heads, n, n)
    return torch.gather(z_rows, 3, col_idx)


def gather_node_axis(t: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    bsz, n = perm_pos.shape
    extra = t.dim() - 2
    idx = perm_pos.to(t.device).view(bsz, n, *([1] * extra)).expand_as(t)
    return torch.gather(t, 1, idx)


def gather_pair_axes(t: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    bsz, n = perm_pos.shape
    extra = t.dim() - 3
    row_idx = perm_pos.to(t.device).view(bsz, n, 1, *([1] * extra)).expand_as(t)
    rows = torch.gather(t, 1, row_idx)
    col_idx = perm_pos.to(t.device).view(bsz, 1, n, *([1] * extra)).expand_as(rows)
    return torch.gather(rows, 2, col_idx)


def expand_head_mask(mask: torch.Tensor, heads: int) -> torch.Tensor:
    if mask.size(1) == 1 and heads != 1:
        return mask.expand(-1, heads, -1, -1)
    return mask


def pair_mask_from_layer(layer: Mapping[str, torch.Tensor]) -> torch.Tensor:
    mask = layer["node_mask"].to(dtype=torch.bool)
    return mask[:, None, :, None] & mask[:, None, None, :]


def row_center_tensor(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = expand_head_mask(mask.to(device=z.device, dtype=torch.bool), z.size(1))
    z0 = torch.where(mask, z.float(), torch.zeros_like(z.float()))
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(z0.dtype)
    mean = z0.sum(dim=-1, keepdim=True) / denom
    return torch.where(mask, z0 - mean, torch.zeros_like(z0))


def cosine_by_query(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, signed: bool) -> torch.Tensor:
    mask = expand_head_mask(mask.to(device=u.device, dtype=torch.bool), u.size(1))
    u0 = torch.where(mask, u.float(), torch.zeros_like(u.float()))
    v0 = torch.where(mask, v.float(), torch.zeros_like(v.float()))
    num = (u0 * v0).sum(dim=-1)
    den = torch.sqrt((u0 * u0).sum(dim=-1).clamp_min(1.0e-12))
    den = den * torch.sqrt((v0 * v0).sum(dim=-1).clamp_min(1.0e-12))
    cos = num / den.clamp_min(1.0e-12)
    if signed:
        return torch.clamp(cos, -1.0, 1.0)
    return torch.clamp(cos, 0.0, 1.0)


def attention_moved_mass(clean_attn: torch.Tensor, perm_pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ref = transform_pair_reference(clean_attn.float(), perm_pos)
    ref_mask = transform_pair_reference(mask, perm_pos).to(dtype=torch.bool)
    moved_mask = expand_head_mask(mask.to(clean_attn.device) | ref_mask, clean_attn.size(1))
    clean0 = torch.where(moved_mask, clean_attn.float(), torch.zeros_like(clean_attn.float()))
    ref0 = torch.where(moved_mask, ref, torch.zeros_like(ref))
    return torch.clamp(0.5 * torch.abs(clean0 - ref0).sum(dim=-1), 0.0, 1.0)


def make_node_permutation(batch: BridgeBatch, generator: torch.Generator) -> torch.Tensor:
    bsz, nmax = batch.node_type.shape
    perm_pos = torch.zeros(bsz, nmax, dtype=torch.long, device=batch.node_type.device)
    for bidx in range(bsz):
        n = int(batch.node_mask[bidx].sum())
        perm = torch.randperm(n, generator=generator).to(batch.node_type.device)
        perm_pos[bidx, :n] = perm
        if n < nmax:
            perm_pos[bidx, n:] = torch.arange(n, nmax, device=batch.node_type.device)
    return perm_pos


def replace_batch_fields(batch: BridgeBatch, **fields: torch.Tensor) -> BridgeBatch:
    data = {
        "node_type": batch.node_type,
        "node_mask": batch.node_mask,
        "adj": batch.adj,
        "degree": batch.degree,
        "spd": batch.spd,
        "rwse": batch.rwse,
        "rrwp": batch.rrwp,
        "pair_xi": batch.pair_xi,
        "edge_batch": batch.edge_batch,
        "edge_src": batch.edge_src,
        "edge_dst": batch.edge_dst,
        "edge_label": batch.edge_label,
        "num_graphs": batch.num_graphs,
        "max_nodes": batch.max_nodes,
    }
    data.update(fields)
    return BridgeBatch(**data)


def node_type_permuted(batch: BridgeBatch, perm_pos: torch.Tensor) -> BridgeBatch:
    return replace_batch_fields(batch, node_type=gather_node_axis(batch.node_type, perm_pos))


def structural_permuted(batch: BridgeBatch, perm_pos: torch.Tensor) -> BridgeBatch:
    return replace_batch_fields(
        batch,
        adj=gather_pair_axes(batch.adj, perm_pos),
        degree=gather_node_axis(batch.degree, perm_pos),
        spd=gather_pair_axes(batch.spd, perm_pos),
        rwse=gather_node_axis(batch.rwse, perm_pos),
        rrwp=gather_pair_axes(batch.rrwp, perm_pos),
        pair_xi=gather_pair_axes(batch.pair_xi, perm_pos),
    )


def joint_permuted(batch: BridgeBatch, perm_pos: torch.Tensor) -> BridgeBatch:
    out = structural_permuted(batch, perm_pos)
    return replace_batch_fields(out, node_type=gather_node_axis(batch.node_type, perm_pos))


def perm_for_attention(attn: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    if attn.size(-1) == perm_pos.size(1) + 1:
        bsz, n = perm_pos.shape
        out = torch.zeros(bsz, n + 1, dtype=torch.long, device=perm_pos.device)
        out[:, 0] = 0
        out[:, 1:] = perm_pos + 1
        return out
    return perm_pos


def aggregate_alpha_rows(
    score_lists: Mapping[str, list[torch.Tensor]],
    moved_masses: Sequence[torch.Tensor],
    node_mask: torch.Tensor,
    meta: Mapping[str, object],
    batch_idx: int,
    graph_offset: int,
    layer: int,
    tau: float,
) -> list[dict[str, object]]:
    moved = torch.stack(list(moved_masses), dim=0)
    alpha = torch.softmax(moved / tau, dim=0)
    alpha_max = alpha.max(dim=0).values
    effective = 1.0 / torch.square(alpha).sum(dim=0).clamp_min(1.0e-12)
    valid = node_mask[:, None, :].to(device=moved.device, dtype=torch.bool)
    valid = valid.expand(-1, moved.size(2), -1)
    rows = []
    for metric, values in score_lists.items():
        scores = torch.stack(values, dim=0)
        weighted = (alpha * scores).sum(dim=0)
        denom = valid.sum(dim=-1).clamp_min(1).to(weighted.dtype)
        head_score = (weighted * valid.to(weighted.dtype)).sum(dim=-1) / denom
        moved_mean = (moved.mean(dim=0) * valid.to(moved.dtype)).sum(dim=-1) / denom
        alpha_max_mean = (alpha_max * valid.to(alpha_max.dtype)).sum(dim=-1) / denom
        effective_mean = (effective * valid.to(effective.dtype)).sum(dim=-1) / denom
        bsz, heads = head_score.shape
        for graph_idx in range(bsz):
            for head in range(heads):
                rows.append(
                    {
                        **meta,
                        "batch": batch_idx,
                        "graph_in_batch": graph_idx,
                        "graph_index": graph_offset + graph_idx,
                        "layer": layer,
                        "head": head,
                        "metric": metric,
                        "score": float(head_score[graph_idx, head].detach().cpu()),
                        "moved_mass_mean": float(moved_mean[graph_idx, head].detach().cpu()),
                        "alpha_max_mean": float(alpha_max_mean[graph_idx, head].detach().cpu()),
                        "effective_perms_mean": float(effective_mean[graph_idx, head].detach().cpu()),
                        "metric_alpha_tau": tau,
                        "num_perms": len(moved_masses),
                    }
                )
    return rows


@torch.no_grad()
def compute_alpha_metrics(
    model: nn.Module,
    dataset: Dataset[BridgeGraph],
    model_name: str,
    cfg: PilotConfig,
    device: torch.device,
    use_amp: bool,
    split_name: str = "val16",
    max_graphs: Optional[int] = None,
    metric_batch_size: Optional[int] = None,
    metric_perms: Optional[int] = None,
) -> list[dict[str, object]]:
    num_graphs = min(len(dataset), max_graphs if max_graphs is not None else cfg.metric_graphs)
    metric_set = Subset(dataset, list(range(num_graphs)))
    batch_size = metric_batch_size if metric_batch_size is not None else cfg.metric_batch_size
    num_perms = metric_perms if metric_perms is not None else cfg.metric_perms
    loader = make_loader(metric_set, batch_size, shuffle=False, seed=cfg.seed)
    split_offset = sum(ord(char) for char in split_name)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 9901 + split_offset)
    rows: list[dict[str, object]] = []
    model.eval()
    graph_offset = 0
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            _logits, clean_layers = model(batch, collect_attention=True)
        layer_state = {
            int(layer["layer"]): {
                "clean": layer,
                "moved": [],
                "scores": {
                    M_POSITIONAL: [],
                    M_SYMBOLIC: [],
                    M_PE_INVARIANT: [],
                    M_PE_EQUIVARIANT: [],
                    M_JOINT_EQUIVARIANT: [],
                    M_POSITIONAL_CENTERED: [],
                    M_SYMBOLIC_CENTERED: [],
                    M_PE_INVARIANT_CENTERED: [],
                    M_PE_EQUIVARIANT_CENTERED: [],
                    M_JOINT_EQUIVARIANT_CENTERED: [],
                },
            }
            for layer in clean_layers
        }
        for _ in range(num_perms):
            perm_pos = make_node_permutation(batch, generator)
            x_batch = node_type_permuted(batch, perm_pos)
            pe_batch = structural_permuted(batch, perm_pos)
            both_batch = joint_permuted(batch, perm_pos)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                _, x_layers = model(x_batch, collect_attention=True)
                _, pe_layers = model(pe_batch, collect_attention=True)
                _, both_layers = model(both_batch, collect_attention=True)
            for clean, x_layer, pe_layer, both_layer in zip(clean_layers, x_layers, pe_layers, both_layers):
                idx = int(clean["layer"])
                clean_attn = clean["attn"].float()
                clean_mask = pair_mask_from_layer(clean).to(clean_attn.device)
                attn_perm = perm_for_attention(clean_attn, perm_pos)
                ref = transform_pair_reference(clean_attn, attn_perm)
                ref_mask = transform_pair_reference(clean_mask, attn_perm).to(dtype=torch.bool)
                layer_state[idx]["moved"].append(attention_moved_mass(clean_attn, attn_perm, clean_mask))
                scores = layer_state[idx]["scores"]
                scores[M_POSITIONAL].append(
                    cosine_by_query(x_layer["attn"].float(), clean_attn, clean_mask, signed=False)
                )
                scores[M_SYMBOLIC].append(
                    cosine_by_query(x_layer["attn"].float(), ref, ref_mask, signed=False)
                )
                scores[M_PE_INVARIANT].append(
                    cosine_by_query(pe_layer["attn"].float(), clean_attn, clean_mask, signed=False)
                )
                scores[M_PE_EQUIVARIANT].append(
                    cosine_by_query(pe_layer["attn"].float(), ref, ref_mask, signed=False)
                )
                scores[M_JOINT_EQUIVARIANT].append(
                    cosine_by_query(both_layer["attn"].float(), ref, ref_mask, signed=False)
                )
                clean_c = row_center_tensor(clean_attn, clean_mask)
                ref_c = transform_pair_reference(clean_c, attn_perm)
                x_c = row_center_tensor(x_layer["attn"].float(), clean_mask)
                pe_c = row_center_tensor(pe_layer["attn"].float(), clean_mask)
                both_c = row_center_tensor(both_layer["attn"].float(), clean_mask)
                scores[M_POSITIONAL_CENTERED].append(cosine_by_query(x_c, clean_c, clean_mask, signed=True))
                scores[M_SYMBOLIC_CENTERED].append(cosine_by_query(x_c, ref_c, ref_mask, signed=True))
                scores[M_PE_INVARIANT_CENTERED].append(cosine_by_query(pe_c, clean_c, clean_mask, signed=True))
                scores[M_PE_EQUIVARIANT_CENTERED].append(cosine_by_query(pe_c, ref_c, ref_mask, signed=True))
                scores[M_JOINT_EQUIVARIANT_CENTERED].append(cosine_by_query(both_c, ref_c, ref_mask, signed=True))
        for layer_idx, state in sorted(layer_state.items()):
            rows.extend(
                aggregate_alpha_rows(
                    state["scores"],
                    state["moved"],
                    state["clean"]["node_mask"],
                    {"model": model_name, "phase": "best", "split": split_name},
                    batch_idx,
                    graph_offset,
                    layer_idx,
                    cfg.metric_alpha_tau,
                )
            )
        graph_offset += batch.num_graphs
    return rows


def summarize_metric_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, int, int], list[float]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["metric"]), int(row["layer"]), int(row["head"]))
        buckets.setdefault(key, []).append(float(row["score"]))
    out = []
    for (model, metric, layer, head), values in sorted(buckets.items()):
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        out.append(
            {
                "model": model,
                "metric": metric,
                "layer": layer,
                "head": head,
                "score_mean": mean,
                "score_std": math.sqrt(var),
                "n": len(values),
            }
        )
    return out


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
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, color="#377eb8")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.25)
    axes[1].plot(epochs, val_f1, color="#4daf4a")
    axes[1].set_title("Validation F1")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.25)
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
    val = [float(summary["val16_metrics"]["f1"]) for summary in summaries]
    t128 = [float(summary["test128_metrics"]["f1"]) for summary in summaries]
    t256 = [float(summary["test256_metrics"]["f1"]) for summary in summaries]
    x = torch.arange(len(models)).float()
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 4.2))
    ax.bar((x - width).numpy(), val, width=width, label="val16", color="#80b1d3")
    ax.bar(x.numpy(), t128, width=width, label="test128", color="#8dd3c7")
    ax.bar((x + width).numpy(), t256, width=width, label="test256", color="#fb8072")
    ax.set_xticks(x.numpy())
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Bridges-Easy size generalization")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def metric_from_summary(summary: Mapping[str, object], split_key: str, metric: str = "f1") -> float:
    values = summary.get(split_key, {})
    if not isinstance(values, Mapping):
        return float("nan")
    try:
        return float(values.get(metric, float("nan")))
    except Exception:
        return float("nan")


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan"), float("nan")
    mean = sum(finite) / len(finite)
    if len(finite) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return mean, math.sqrt(max(0.0, var))


def aggregate_repeat_rows(
    models: Sequence[str],
    summaries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        model_runs = [summary for summary in summaries if str(summary.get("model")) == model]
        if not model_runs:
            continue
        row: dict[str, object] = {"model": model, "n_seeds": len(model_runs)}
        for split_key, prefix in [
            ("val16_metrics", "val16"),
            ("test128_metrics", "test128"),
            ("test256_metrics", "test256"),
        ]:
            for metric in ["f1", "acc", "precision", "recall"]:
                mean, std = mean_std([metric_from_summary(summary, split_key, metric) for summary in model_runs])
                row[f"{prefix}_{metric}_mean"] = mean
                row[f"{prefix}_{metric}_std"] = std
        best = max(model_runs, key=lambda summary: metric_from_summary(summary, "val16_metrics", "f1"))
        row["best_seed_by_val16_f1"] = int(best.get("seed", -1))
        row["best_val16_f1"] = metric_from_summary(best, "val16_metrics", "f1")
        row["best_test128_f1"] = metric_from_summary(best, "test128_metrics", "f1")
        row["best_test256_f1"] = metric_from_summary(best, "test256_metrics", "f1")
        rows.append(row)
    return rows


def select_and_save_best_trials(
    models: Sequence[str],
    summaries: Sequence[Mapping[str, object]],
    suite_dir: Path,
    log=print,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        model_runs = [summary for summary in summaries if str(summary.get("model")) == model]
        if not model_runs:
            continue
        best = max(model_runs, key=lambda summary: metric_from_summary(summary, "val16_metrics", "f1"))
        model_dir = suite_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        source_checkpoint = Path(str(best.get("best_checkpoint", "")))
        dest_checkpoint = model_dir / "best_over_trials.pt"
        if source_checkpoint.exists():
            shutil.copy2(source_checkpoint, dest_checkpoint)
        else:
            log(f"[best] source checkpoint missing for {model}: {source_checkpoint}")
        best_summary = dict(best)
        best_summary["source_best_checkpoint"] = str(source_checkpoint)
        best_summary["copied_best_checkpoint"] = str(dest_checkpoint)
        write_json(model_dir / "best_over_trials_summary.json", best_summary)
        row = {
            "model": model,
            "selected_seed": int(best.get("seed", -1)),
            "selected_epoch": int(best.get("best_epoch", -1)),
            "val16_f1": metric_from_summary(best, "val16_metrics", "f1"),
            "test128_f1": metric_from_summary(best, "test128_metrics", "f1"),
            "test256_f1": metric_from_summary(best, "test256_metrics", "f1"),
            "source_best_checkpoint": str(source_checkpoint),
            "copied_best_checkpoint": str(dest_checkpoint),
        }
        rows.append(row)
        log(
            f"[best] {model}: seed={row['selected_seed']} "
            f"val16_f1={row['val16_f1']:.4f} test128_f1={row['test128_f1']:.4f} "
            f"test256_f1={row['test256_f1']:.4f}"
        )
    return rows


def plot_repeat_performance_summary(
    aggregate_rows: Sequence[Mapping[str, object]],
    output_path: Path,
    log=print,
) -> None:
    if not aggregate_rows:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping repeat performance summary: {exc}")
        return
    models = [str(row["model"]) for row in aggregate_rows]
    x = torch.arange(len(models)).float()
    width = 0.24
    specs = [
        ("val16", -width, "#80b1d3"),
        ("test128", 0.0, "#8dd3c7"),
        ("test256", width, "#fb8072"),
    ]
    fig, ax = plt.subplots(figsize=(max(9.0, len(models) * 0.75), 4.8))
    for label, offset, color in specs:
        means = [float(row[f"{label}_f1_mean"]) for row in aggregate_rows]
        stds = [float(row[f"{label}_f1_std"]) for row in aggregate_rows]
        ax.bar((x + offset).numpy(), means, yerr=stds, width=width, label=label, color=color, capsize=2.5)
    ax.set_xticks(x.numpy())
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("F1, mean +/- seed std")
    ax.set_title("Bridges-Easy 3-seed performance")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_trial_performance(
    best_rows: Sequence[Mapping[str, object]],
    output_path: Path,
    log=print,
) -> None:
    if not best_rows:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping best-trial performance plot: {exc}")
        return
    models = [str(row["model"]) for row in best_rows]
    x = torch.arange(len(models)).float()
    width = 0.24
    specs = [
        ("val16_f1", "val16", -width, "#80b1d3"),
        ("test128_f1", "test128", 0.0, "#8dd3c7"),
        ("test256_f1", "test256", width, "#fb8072"),
    ]
    fig, ax = plt.subplots(figsize=(max(9.0, len(models) * 0.75), 4.8))
    for key, label, offset, color in specs:
        values = [float(row[key]) for row in best_rows]
        ax.bar((x + offset).numpy(), values, width=width, label=label, color=color)
    seed_labels = [f"s{int(row['selected_seed'])}" for row in best_rows]
    for idx, seed_label in enumerate(seed_labels):
        ax.text(float(x[idx]), 1.01, seed_label, rotation=90, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x.numpy())
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("F1")
    ax.set_title("Best validation-seed checkpoint per model")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_seed_repeat_scatter(
    summaries: Sequence[Mapping[str, object]],
    output_path: Path,
    log=print,
) -> None:
    if not summaries:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping seed scatter plot: {exc}")
        return
    models = [model for model in MODEL_NAMES if any(str(summary.get("model")) == model for summary in summaries)]
    model_index = {model: idx for idx, model in enumerate(models)}
    split_specs = [
        ("val16_metrics", "val16", "#80b1d3"),
        ("test128_metrics", "test128", "#1b9e77"),
        ("test256_metrics", "test256", "#d95f02"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(max(10.5, len(models) * 0.85), 3.8), sharey=True)
    for ax, (split_key, label, color) in zip(axes, split_specs):
        for summary in summaries:
            model = str(summary.get("model"))
            idx = model_index[model]
            seed = int(summary.get("seed", 0))
            jitter = ((seed % 17) - 8) * 0.012
            ax.scatter(idx + jitter, metric_from_summary(summary, split_key, "f1"), s=18, color=color, alpha=0.85)
        ax.set_title(label)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
    axes[0].set_ylabel("F1 by seed")
    fig.suptitle("Repeat-trial spread")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_generalization_drop(
    aggregate_rows: Sequence[Mapping[str, object]],
    output_path: Path,
    log=print,
) -> None:
    if not aggregate_rows:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping generalization-drop plot: {exc}")
        return
    models = [str(row["model"]) for row in aggregate_rows]
    x = torch.arange(len(models)).float()
    width = 0.32
    drop128 = [float(row["val16_f1_mean"]) - float(row["test128_f1_mean"]) for row in aggregate_rows]
    drop256 = [float(row["val16_f1_mean"]) - float(row["test256_f1_mean"]) for row in aggregate_rows]
    fig, ax = plt.subplots(figsize=(max(9.0, len(models) * 0.65), 4.2))
    ax.bar((x - width / 2).numpy(), drop128, width=width, color="#8dd3c7", label="val16 - test128")
    ax.bar((x + width / 2).numpy(), drop256, width=width, color="#fb8072", label="val16 - test256")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x.numpy())
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("F1 drop")
    ax.set_title("Mean size-generalization gap")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_detour_performance(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
    split_name: str = "test256",
    log=print,
) -> None:
    data = [row for row in rows if str(row["split"]) == split_name]
    if not data:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping detour performance: {exc}")
        return
    models = [model for model in MODEL_NAMES if any(str(row["model"]) == model for row in data)]
    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, 0.42 * len(models)), 7.0), sharex=True)
    width = 0.12
    x = torch.arange(len(models)).float()
    colors = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462"]
    for bin_idx, bin_name in enumerate(DETOUR_BINS):
        offsets = x + (bin_idx - (len(DETOUR_BINS) - 1) / 2.0) * width
        values_acc = []
        values_pred = []
        for model in models:
            match = [row for row in data if str(row["model"]) == model and str(row["edge_bin"]) == bin_name]
            values_acc.append(mean_std([float(row["acc"]) for row in match])[0] if match else float("nan"))
            values_pred.append(
                mean_std([float(row["pred_positive_rate"]) for row in match])[0] if match else float("nan")
            )
        axes[0].bar(offsets.numpy(), values_acc, width=width, label=bin_name, color=colors[bin_idx])
        axes[1].bar(offsets.numpy(), values_pred, width=width, label=bin_name, color=colors[bin_idx])
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].set_ylabel("predicted bridge rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x.numpy())
    axes[1].set_xticklabels(models, rotation=35, ha="right", fontsize=8)
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle(f"Detour-stratified edge performance on {split_name}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_heatmaps(
    all_summaries: Mapping[str, Sequence[Mapping[str, object]]],
    output_path: Path,
    log=print,
) -> None:
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping alpha heatmaps: {exc}")
        return
    models = [model for model in ATTENTION_MODEL_NAMES if model in all_summaries]
    if not models:
        return
    fig, axes = plt.subplots(len(models), 2, figsize=(8.5, max(2.2, 1.7 * len(models))))
    if len(models) == 1:
        axes = axes.reshape(1, 2)
    for row_idx, model in enumerate(models):
        rows = list(all_summaries[model])
        for col_idx, metric in enumerate([M_POSITIONAL, M_SYMBOLIC]):
            data = [row for row in rows if str(row["metric"]) == metric]
            max_layer = max([int(row["layer"]) for row in data], default=0)
            max_head = max([int(row["head"]) for row in data], default=0)
            mat = torch.full((max_layer + 1, max_head + 1), float("nan"))
            for row in data:
                mat[int(row["layer"]), int(row["head"])] = float(row["score_mean"])
            ax = axes[row_idx][col_idx]
            im = ax.imshow(mat.numpy(), vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
            ax.set_title(f"{model}: {metric}", fontsize=9)
            ax.set_xlabel("head")
            ax.set_ylabel("layer")
            ax.set_xticks(range(max_head + 1))
            ax.set_yticks(range(max_layer + 1))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def alpha_summary_table(rows: Sequence[Mapping[str, object]]) -> dict[tuple[int, int], dict[str, float]]:
    table: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        layer = int(row["layer"])
        head = int(row["head"])
        metric = str(row["metric"])
        table.setdefault((layer, head), {})[metric] = float(row["score_mean"])
    return table


def raw_score_values(
    rows: Sequence[Mapping[str, object]],
    layer: int,
    head: int,
    metric: str,
) -> list[float]:
    return [
        float(row["score"])
        for row in rows
        if int(row["layer"]) == layer and int(row["head"]) == head and str(row["metric"]) == metric
    ]


def raw_row_graph_index(row: Mapping[str, object], fallback_batch_size: int) -> Optional[int]:
    if "graph_index" in row and str(row["graph_index"]) != "":
        return int(row["graph_index"])
    if "batch" in row and "graph_in_batch" in row:
        return int(row["batch"]) * fallback_batch_size + int(row["graph_in_batch"])
    return None


def per_graph_head_scores(
    rows: Sequence[Mapping[str, object]],
    layer: int,
    head: int,
    graph_index: int,
    fallback_batch_size: int,
) -> dict[str, float]:
    out = {}
    for row in rows:
        if int(row["layer"]) != layer or int(row["head"]) != head:
            continue
        row_graph_index = raw_row_graph_index(row, fallback_batch_size)
        if row_graph_index != graph_index:
            continue
        out[str(row["metric"])] = float(row["score"])
    return out


def scored_graph_indices(
    rows: Sequence[Mapping[str, object]],
    fallback_batch_size: int,
) -> list[int]:
    indices = set()
    for row in rows:
        if str(row.get("metric", "")) != M_SYMBOLIC:
            continue
        graph_index = raw_row_graph_index(row, fallback_batch_size)
        if graph_index is not None:
            indices.add(graph_index)
    return sorted(indices)


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    tensor = torch.tensor(list(values), dtype=torch.float32)
    return float(torch.quantile(tensor, q).item())


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    x = torch.tensor(list(xs), dtype=torch.float64)
    y = torch.tensor(list(ys), dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(1.0e-12)
    return float(((x * y).sum() / denom).item())


def masked_corr(values: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.to(dtype=torch.bool)
    if int(valid.sum()) < 3:
        return float("nan")
    x = values[valid].float()
    y = features[valid].float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum()).clamp_min(1.0e-12)
    return float(((x * y).sum() / denom).detach().cpu())


def strip_graph_token_attention(
    attn: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attn.size(-1) == node_mask.size(1) + 1:
        return attn[:, :, 1:, 1:], node_mask
    return attn, node_mask


def edge_label_matrix(batch: BridgeBatch) -> torch.Tensor:
    labels = torch.zeros_like(batch.adj)
    labels[batch.edge_batch, batch.edge_src, batch.edge_dst] = batch.edge_label.float()
    return labels


def select_standout_heads(
    model_name: str,
    val_summary_rows: Sequence[Mapping[str, object]],
    stability_rows: Sequence[Mapping[str, object]],
    limit: int,
) -> list[dict[str, object]]:
    table = alpha_summary_table(val_summary_rows)
    stability_by_head = {
        (int(row["layer"]), int(row["head"])): float(row["combined_abs_delta"])
        for row in stability_rows
        if str(row["model"]) == model_name and str(row["target_split"]) == "test256"
    }
    category_order = [
        "high_symbolic_low_structural",
        "high_structural_low_symbolic",
        "high_both",
        "low_both",
        "high_positional",
        "unstable_on_256",
    ]
    candidates = []
    for (layer, head), metrics in table.items():
        symbolic = metrics.get(M_SYMBOLIC, float("nan"))
        structural = metrics.get(M_PE_EQUIVARIANT, float("nan"))
        positional = metrics.get(M_POSITIONAL, float("nan"))
        if math.isnan(symbolic) or math.isnan(structural):
            continue
        base = {
            "model": model_name,
            "layer": layer,
            "head": head,
            "symbolic_score": symbolic,
            "structural_score": structural,
            "positional_score": positional,
            "instability_256": stability_by_head.get((layer, head), float("nan")),
        }
        candidates.extend(
            [
                (symbolic - structural, "high_symbolic_low_structural", base),
                (structural - symbolic, "high_structural_low_symbolic", base),
                (min(symbolic, structural), "high_both", base),
                (-max(symbolic, structural), "low_both", base),
                (positional, "high_positional", base),
                (stability_by_head.get((layer, head), -1.0), "unstable_on_256", base),
            ]
        )
    selected: list[dict[str, object]] = []
    seen_heads: set[tuple[int, int]] = set()

    def try_add(category: str, candidates_for_category: Sequence[tuple[float, str, Mapping[str, object]]]) -> None:
        for _score, _category, base in sorted(candidates_for_category, key=lambda item: item[0], reverse=True):
            head_key = (int(base["layer"]), int(base["head"]))
            if head_key in seen_heads:
                continue
            selected.append(dict(base) | {"category": category})
            seen_heads.add(head_key)
            return

    for category in category_order:
        try_add(category, [item for item in candidates if item[1] == category])
        if len(selected) >= limit:
            return selected

    for _score, category, base in sorted(candidates, key=lambda item: item[0], reverse=True):
        head_key = (int(base["layer"]), int(base["head"]))
        if head_key in seen_heads:
            continue
        selected.append(dict(base) | {"category": category})
        seen_heads.add(head_key)
        if len(selected) >= limit:
            break
    return selected


def summarize_score_distributions(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, str, int, int], list[float]] = {}
    for row in rows:
        key = (
            str(row["model"]),
            str(row["split"]),
            str(row["metric"]),
            int(row["layer"]),
            int(row["head"]),
        )
        buckets.setdefault(key, []).append(float(row["score"]))
    out = []
    for (model, split, metric, layer, head), values in sorted(buckets.items()):
        mean = sum(values) / len(values)
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        out.append(
            {
                "model": model,
                "split": split,
                "metric": metric,
                "layer": layer,
                "head": head,
                "mean": mean,
                "std": std,
                "p10": quantile(values, 0.10),
                "p25": quantile(values, 0.25),
                "p50": quantile(values, 0.50),
                "p75": quantile(values, 0.75),
                "p90": quantile(values, 0.90),
                "frac_ge_0_75": sum(value >= 0.75 for value in values) / len(values),
                "frac_le_0_25": sum(value <= 0.25 for value in values) / len(values),
                "n": len(values),
            }
        )
    return out


def split_analysis_limits(args: argparse.Namespace, split_name: str, cfg: PilotConfig) -> tuple[int, int]:
    if split_name == "val16":
        graphs, batch_size = args.analysis_graphs16, cfg.metric_batch_size
    elif split_name == "test128":
        graphs, batch_size = args.analysis_graphs128, min(cfg.metric_batch_size, 4)
    elif split_name == "test256":
        graphs, batch_size = args.analysis_graphs256, min(cfg.metric_batch_size, 1)
    else:
        graphs, batch_size = cfg.metric_graphs, cfg.metric_batch_size
    if args.fast_dev_run:
        caps = {"val16": 8, "test128": 4, "test256": 2}
        graphs = min(graphs, caps.get(split_name, 8))
        batch_size = min(batch_size, 2)
    return max(1, graphs), max(1, batch_size)


def load_trained_model(
    model_name: str,
    cfg: PilotConfig,
    output_root: Path,
    device: torch.device,
) -> tuple[nn.Module, PilotConfig]:
    model_cfg = model_config_for(model_name, cfg)
    model = build_model(model_name, model_cfg).to(device)
    best_path = output_root / cfg.preset / model_name / f"seed{cfg.seed}" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for {model_name}: {best_path}")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_cfg


def compute_or_load_split_alpha(
    model: nn.Module,
    model_name: str,
    split_name: str,
    dataset: Dataset[BridgeGraph],
    cfg: PilotConfig,
    model_cfg: PilotConfig,
    output_root: Path,
    device: torch.device,
    use_amp: bool,
    args: argparse.Namespace,
    log=print,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_dir = output_root / cfg.preset / model_name / f"seed{cfg.seed}"
    raw_path = run_dir / f"alpha_permutation_metrics_{split_name}_best.csv"
    summary_path = run_dir / f"alpha_permutation_summary_{split_name}_best.csv"
    legacy_raw = run_dir / "alpha_permutation_metrics_best.csv"
    legacy_summary = run_dir / "alpha_permutation_summary_best.csv"
    if split_name == "val16" and legacy_summary.exists() and not summary_path.exists():
        summary_path = legacy_summary
        raw_path = legacy_raw
    if raw_path.exists() and summary_path.exists() and not args.force_analysis and not args.force_metrics:
        return read_csv_rows(raw_path), read_csv_rows(summary_path)
    max_graphs, batch_size = split_analysis_limits(args, split_name, cfg)
    log(f"[analysis] computing alpha metrics model={model_name} split={split_name} graphs={max_graphs}")
    raw_rows = compute_alpha_metrics(
        model,
        dataset,
        model_name,
        model_cfg,
        device,
        use_amp,
        split_name=split_name,
        max_graphs=max_graphs,
        metric_batch_size=batch_size,
    )
    summary_rows = summarize_metric_rows(raw_rows)
    raw_path = run_dir / f"alpha_permutation_metrics_{split_name}_best.csv"
    summary_path = run_dir / f"alpha_permutation_summary_{split_name}_best.csv"
    write_csv_rows(raw_path, raw_rows)
    write_csv_rows(summary_path, summary_rows)
    if split_name == "val16":
        write_csv_rows(legacy_raw, raw_rows)
        write_csv_rows(legacy_summary, summary_rows)
    return raw_rows, summary_rows


@torch.no_grad()
def compute_attention_feature_profiles(
    model: nn.Module,
    dataset: Dataset[BridgeGraph],
    model_name: str,
    cfg: PilotConfig,
    device: torch.device,
    use_amp: bool,
    max_graphs: int,
    batch_size: int,
) -> list[dict[str, object]]:
    metric_set = Subset(dataset, list(range(min(len(dataset), max_graphs))))
    loader = make_loader(metric_set, batch_size, shuffle=False, seed=cfg.seed)
    buckets: dict[tuple[int, int], dict[str, list[float]]] = {}
    for batch in loader:
        batch = batch.to(device)
        bridge_mat = edge_label_matrix(batch)
        same_symbol = (batch.node_type[:, :, None] == batch.node_type[:, None, :]).float()
        self_mat = torch.eye(batch.max_nodes, device=device).view(1, batch.max_nodes, batch.max_nodes)
        pair_mask = batch.pair_mask
        inv_spd = torch.where(
            batch.spd > 0,
            1.0 / batch.spd.float().clamp_min(1.0),
            torch.zeros_like(batch.spd.float()),
        )
        far_mat = ((batch.spd >= 3) & (batch.spd <= SPD_CAP)).float()
        degree_dst = batch.degree[:, None, :].expand_as(batch.adj)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            _logits, layers = model(batch, collect_attention=True)
        for layer in layers:
            layer_idx = int(layer["layer"])
            attn, node_mask = strip_graph_token_attention(layer["attn"].float(), batch.node_mask)
            valid = pair_mask & node_mask[:, :, None] & node_mask[:, None, :]
            for head in range(attn.size(1)):
                weights = torch.where(valid, attn[:, head], torch.zeros_like(attn[:, head]))
                denom = weights.sum().clamp_min(1.0e-12)
                key = (layer_idx, head)
                target = buckets.setdefault(key, {})

                def add(name: str, value: float) -> None:
                    target.setdefault(name, []).append(float(value))

                add("self_mass", float((weights * self_mat).sum().detach().cpu() / denom.detach().cpu()))
                add("edge_mass", float((weights * batch.adj).sum().detach().cpu() / denom.detach().cpu()))
                add("bridge_mass", float((weights * bridge_mat).sum().detach().cpu() / denom.detach().cpu()))
                nonedge_mass = (weights * (1.0 - batch.adj) * (1.0 - self_mat)).sum()
                add("nonedge_mass", float(nonedge_mass.detach().cpu() / denom.detach().cpu()))
                add("same_symbol_mass", float((weights * same_symbol).sum().detach().cpu() / denom.detach().cpu()))
                add("far_mass", float((weights * far_mat).sum().detach().cpu() / denom.detach().cpu()))
                add("adjacency_corr", masked_corr(attn[:, head], batch.adj, valid))
                add("bridge_corr", masked_corr(attn[:, head], bridge_mat, valid))
                add("same_symbol_corr", masked_corr(attn[:, head], same_symbol, valid))
                add("inverse_spd_corr", masked_corr(attn[:, head], inv_spd, valid))
                add("dst_degree_corr", masked_corr(attn[:, head], degree_dst, valid))
    rows = []
    for (layer, head), values in sorted(buckets.items()):
        row: dict[str, object] = {"model": model_name, "split": "val16", "layer": layer, "head": head}
        for name, vals in values.items():
            vals = [value for value in vals if not math.isnan(value)]
            row[name] = sum(vals) / len(vals) if vals else float("nan")
        rows.append(row)
    return rows


LOCALITY_BINS = ("self", "1-hop", "2-hop", "3-hop", "4-hop", "5+-hop", "unreachable")
LOCALITY_VALUES = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, float(SPD_CAP + 1)])


def locality_bucket_ids(spd: torch.Tensor) -> torch.Tensor:
    buckets = spd.clone()
    buckets = torch.where((buckets >= 5) & (buckets <= SPD_CAP), torch.full_like(buckets, 5), buckets)
    buckets = torch.where(buckets >= SPD_CAP + 1, torch.full_like(buckets, 6), buckets)
    return buckets.clamp(min=0, max=6)


def head_locality_distribution(
    attn: torch.Tensor,
    spd: torch.Tensor,
    pair_mask: torch.Tensor,
) -> tuple[list[float], float, float, float]:
    weights = torch.where(pair_mask, attn.float(), torch.zeros_like(attn.float()))
    denom = weights.sum().clamp_min(1.0e-12)
    buckets = locality_bucket_ids(spd).to(weights.device)
    masses = []
    for bucket in range(len(LOCALITY_BINS)):
        masses.append(float(weights[buckets == bucket].sum().detach().cpu() / denom.detach().cpu()))
    values = LOCALITY_VALUES.to(weights.device)
    expected_hop = float((weights * values[buckets]).sum().detach().cpu() / denom.detach().cpu())
    local_1hop = masses[0] + masses[1]
    local_2hop = local_1hop + masses[2]
    long_range = sum(masses[3:])
    return masses, expected_hop, local_1hop, local_2hop if local_2hop <= 1.0 else 1.0


@torch.no_grad()
def compute_attention_locality_rows(
    model: nn.Module,
    dataset: Dataset[BridgeGraph],
    model_name: str,
    cfg: PilotConfig,
    device: torch.device,
    use_amp: bool,
    max_graphs: int,
    batch_size: int,
    split_name: str = "val16",
) -> list[dict[str, object]]:
    metric_set = Subset(dataset, list(range(min(len(dataset), max_graphs))))
    loader = make_loader(metric_set, batch_size, shuffle=False, seed=cfg.seed)
    rows: list[dict[str, object]] = []
    graph_offset = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            _logits, layers = model(batch, collect_attention=True)
        for layer in layers:
            layer_idx = int(layer["layer"])
            attn, node_mask = strip_graph_token_attention(layer["attn"].float(), batch.node_mask)
            pair_mask = batch.pair_mask & node_mask[:, :, None] & node_mask[:, None, :]
            for graph_idx in range(batch.num_graphs):
                valid = pair_mask[graph_idx]
                spd = batch.spd[graph_idx]
                for head in range(attn.size(1)):
                    masses, expected_hop, local_1hop, local_2hop = head_locality_distribution(
                        attn[graph_idx, head],
                        spd,
                        valid,
                    )
                    row = {
                        "model": model_name,
                        "split": split_name,
                        "graph_index": graph_offset + graph_idx,
                        "layer": layer_idx,
                        "head": head,
                        "expected_hop": expected_hop,
                        "local_1hop_mass": local_1hop,
                        "local_2hop_mass": local_2hop,
                        "long_range_mass": max(0.0, 1.0 - local_2hop),
                    }
                    for name, value in zip(LOCALITY_BINS, masses):
                        row[f"mass_{name}"] = value
                    rows.append(row)
        graph_offset += batch.num_graphs
    return rows


def summarize_locality_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, int, int], dict[str, list[float]]] = {}
    fields = [
        "expected_hop",
        "local_1hop_mass",
        "local_2hop_mass",
        "long_range_mass",
        *[f"mass_{name}" for name in LOCALITY_BINS],
    ]
    for row in rows:
        key = (str(row["model"]), str(row["split"]), int(row["layer"]), int(row["head"]))
        target = buckets.setdefault(key, {field: [] for field in fields})
        for field in fields:
            target[field].append(float(row[field]))
    out = []
    for (model, split, layer, head), values_by_field in sorted(buckets.items()):
        row: dict[str, object] = {"model": model, "split": split, "layer": layer, "head": head}
        for field, values in values_by_field.items():
            mean = sum(values) / len(values)
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
            row[f"{field}_p10"] = quantile(values, 0.10)
            row[f"{field}_p50"] = quantile(values, 0.50)
            row[f"{field}_p90"] = quantile(values, 0.90)
        row["n"] = len(next(iter(values_by_field.values())))
        out.append(row)
    return out


def plot_locality_distributions(
    model_name: str,
    locality_rows: Sequence[Mapping[str, object]],
    output_path: Path,
    log=print,
) -> None:
    rows = [row for row in locality_rows if str(row["model"]) == model_name]
    if not rows:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping locality distributions: {exc}")
        return
    heads = sorted({(int(row["layer"]), int(row["head"])) for row in rows})
    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 0.45 * len(heads)), 6.2), sharex=True)
    for ax, field, title in [
        (axes[0], "expected_hop", "attention expected hop"),
        (axes[1], "long_range_mass", "attention mass at 3+ hops"),
    ]:
        data = [
            [float(row[field]) for row in rows if int(row["layer"]) == layer and int(row["head"]) == head]
            for layer, head in heads
        ]
        box = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch in box["boxes"]:
            patch.set_facecolor("#80b1d3")
            patch.set_alpha(0.82)
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(range(1, len(heads) + 1))
    axes[-1].set_xticklabels([f"L{layer}H{head}" for layer, head in heads], rotation=45, ha="right", fontsize=7)
    fig.suptitle(f"{model_name}: locality distribution across val16 graph inputs")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_locality_metric_connection(
    locality_summary_rows: Sequence[Mapping[str, object]],
    split_summaries: Mapping[str, Sequence[Mapping[str, object]]],
    output_path: Path,
    log=print,
) -> None:
    if not locality_summary_rows:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping locality/metric connection: {exc}")
        return
    colors = {
        "graphormer": "#1f77b4",
        "graphgps": "#2ca02c",
        "grit": "#d62728",
        "v24_static": "#9467bd",
        "v24_evo": "#ff7f0e",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True, sharey=True)
    for metric, ax in [(M_SYMBOLIC, axes[0]), (M_PE_EQUIVARIANT, axes[1])]:
        xs, ys, cs, labels = [], [], [], []
        for row in locality_summary_rows:
            model = str(row["model"])
            table = alpha_summary_table(split_summaries.get(model, []))
            key = (int(row["layer"]), int(row["head"]))
            if metric not in table.get(key, {}):
                continue
            xs.append(float(row["long_range_mass_mean"]))
            ys.append(float(table[key][metric]))
            cs.append(colors.get(model, "#666666"))
            labels.append(model)
        ax.scatter(xs, ys, c=cs, s=48, edgecolor="black", linewidth=0.25)
        ax.set_title(metric)
        ax.set_xlabel("mean attention mass at 3+ hops")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("alpha-weighted score")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=model, markersize=7)
        for model, color in colors.items()
        if any(str(row["model"]) == model for row in locality_summary_rows)
    ]
    axes[1].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Locality vs symbolic/structural specialization")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def graph_from_record(graph: BridgeGraph):
    import networkx as nx

    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(graph.num_nodes))
    for src, dst in graph.edge_index.t().tolist():
        if src < dst:
            graph_nx.add_edge(src, dst)
    return graph_nx


def choose_example_indices(
    dataset: Dataset[BridgeGraph],
    count: int,
    seed: int,
    candidates: Optional[Sequence[int]] = None,
) -> list[int]:
    pool = [idx for idx in (candidates or range(len(dataset))) if 0 <= int(idx) < len(dataset)]
    if not pool:
        pool = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[: min(count, len(pool))]


@torch.no_grad()
def plot_head_examples(
    model: nn.Module,
    dataset: Dataset[BridgeGraph],
    model_name: str,
    head_rows: Sequence[Mapping[str, object]],
    raw_metric_rows: Sequence[Mapping[str, object]],
    cfg: PilotConfig,
    output_dir: Path,
    device: torch.device,
    use_amp: bool,
    example_count: int,
    log=print,
) -> None:
    if not head_rows:
        return
    try:
        plt = import_plotting()
        import networkx as nx
    except Exception as exc:
        log(f"[plot] Skipping head examples: {exc}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_indices = scored_graph_indices(raw_metric_rows, cfg.metric_batch_size)
    indices = choose_example_indices(
        dataset,
        example_count,
        seed=cfg.seed + sum(ord(char) for char in model_name) + 4049,
        candidates=candidate_indices,
    )
    batch = collate_graphs([dataset[idx] for idx in indices]).to(device)
    with torch.autocast(device_type="cuda", enabled=use_amp):
        logits, layers = model(batch, collect_attention=True)
    layer_by_idx = {int(layer["layer"]): layer for layer in layers}
    probs = torch.sigmoid(logits.detach()).cpu()
    edge_offset = 0
    prob_mats = []
    for graph in [dataset[idx] for idx in indices]:
        mat = torch.zeros(graph.num_nodes, graph.num_nodes)
        count = graph.edge_label.numel()
        edge_probs = probs[edge_offset : edge_offset + count]
        edge_offset += count
        for pos, (src, dst) in enumerate(graph.edge_index.t().tolist()):
            mat[src, dst] = edge_probs[pos]
        prob_mats.append(mat)

    for head_row in head_rows:
        layer_idx = int(head_row["layer"])
        head = int(head_row["head"])
        category = str(head_row["category"])
        layer = layer_by_idx.get(layer_idx)
        if layer is None:
            continue
        attn, _node_mask = strip_graph_token_attention(layer["attn"].detach().cpu().float(), batch.node_mask.cpu())
        fig, axes = plt.subplots(len(indices), 3, figsize=(12.8, max(3.2, 3.2 * len(indices))))
        if len(indices) == 1:
            axes = axes.reshape(1, 3)
        for row_idx, graph_idx in enumerate(indices):
            graph = dataset[graph_idx]
            graph_nx = graph_from_record(graph)
            pos = nx.spring_layout(graph_nx, seed=cfg.seed + graph_idx)
            ax_graph = axes[row_idx][0]
            graph_scores = per_graph_head_scores(
                raw_metric_rows,
                layer_idx,
                head,
                graph_idx,
                cfg.metric_batch_size,
            )
            graph_sym = graph_scores.get(M_SYMBOLIC, float("nan"))
            graph_struct = graph_scores.get(M_PE_EQUIVARIANT, float("nan"))
            bridge_edges = set()
            for pos_idx, (src, dst) in enumerate(graph.edge_index.t().tolist()):
                if src < dst and float(graph.edge_label[pos_idx]) > 0.5:
                    bridge_edges.add((src, dst))
            non_bridge_edges = [edge for edge in graph_nx.edges() if tuple(sorted(edge)) not in bridge_edges]
            edge_widths = {}
            for src, dst in graph_nx.edges():
                prob = float(0.5 * (prob_mats[row_idx][src, dst] + prob_mats[row_idx][dst, src]))
                edge_widths[(src, dst)] = 0.8 + 3.0 * prob
            nx.draw_networkx_edges(
                graph_nx,
                pos,
                edgelist=non_bridge_edges,
                width=[edge_widths[edge] for edge in non_bridge_edges],
                edge_color="#b8b8b8",
                ax=ax_graph,
            )
            nx.draw_networkx_edges(
                graph_nx,
                pos,
                edgelist=list(bridge_edges),
                width=[edge_widths[edge] for edge in bridge_edges],
                edge_color="#d62728",
                ax=ax_graph,
            )
            node_colors = [int(graph.node_type[node]) for node in graph_nx.nodes()]
            nx.draw_networkx_nodes(graph_nx, pos, node_color=node_colors, cmap="tab10", node_size=260, ax=ax_graph)
            labels = {node: f"{node}:{int(graph.node_type[node])}" for node in graph_nx.nodes()}
            nx.draw_networkx_labels(graph_nx, pos, labels=labels, font_size=7, ax=ax_graph)
            head_attn = attn[row_idx, head, : graph.num_nodes, : graph.num_nodes]
            top_edges = []
            top_widths = []
            for src in range(graph.num_nodes):
                scores = head_attn[src].clone()
                scores[src] = -1.0
                dst = int(scores.argmax())
                value = float(scores[dst])
                if value > 0:
                    top_edges.append((src, dst))
                    top_widths.append(0.4 + 4.0 * value)
            directed = nx.DiGraph()
            directed.add_nodes_from(graph_nx.nodes())
            directed.add_edges_from(top_edges)
            nx.draw_networkx_edges(
                directed,
                pos,
                edgelist=top_edges,
                width=top_widths,
                edge_color="#1f77b4",
                arrows=True,
                arrowsize=8,
                alpha=0.62,
                connectionstyle="arc3,rad=0.16",
                ax=ax_graph,
            )
            ax_graph.set_title(
                f"graph {graph_idx}: sym={graph_sym:.3f}, struct={graph_struct:.3f}\n"
                "red=true bridge, blue=top attention",
                fontsize=8,
            )
            ax_graph.axis("off")

            ax_heat = axes[row_idx][1]
            im = ax_heat.imshow(head_attn.numpy(), vmin=0.0, vmax=max(0.05, float(head_attn.max())), cmap="magma")
            ax_heat.set_title("attention matrix", fontsize=8)
            ax_heat.set_xlabel("key node")
            ax_heat.set_ylabel("query node")
            fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

            ax_loc = axes[row_idx][2]
            masses, expected_hop, local_1hop, local_2hop = head_locality_distribution(
                head_attn,
                graph.spd,
                torch.ones_like(graph.spd, dtype=torch.bool),
            )
            ax_loc.bar(range(len(LOCALITY_BINS)), masses, color="#8dd3c7")
            ax_loc.set_ylim(0.0, 1.0)
            ax_loc.set_xticks(range(len(LOCALITY_BINS)))
            ax_loc.set_xticklabels(LOCALITY_BINS, rotation=45, ha="right", fontsize=7)
            ax_loc.set_ylabel("attention mass")
            ax_loc.set_title(
                f"locality: E[hop]={expected_hop:.2f}, <=1={local_1hop:.2f}, <=2={local_2hop:.2f}",
                fontsize=8,
            )
            ax_loc.grid(axis="y", alpha=0.25)
        fig.suptitle(
            f"{model_name} L{layer_idx} H{head} {category} | "
            f"sym={float(head_row['symbolic_score']):.3f} "
            f"struct={float(head_row['structural_score']):.3f} "
            f"pos={float(head_row['positional_score']):.3f}",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = output_dir / f"{model_name}_L{layer_idx}_H{head}_{category}.png"
        fig.savefig(out, dpi=170, bbox_inches="tight")
        plt.close(fig)


def plot_specialization_map(
    split_summaries: Mapping[str, Sequence[Mapping[str, object]]],
    selected_rows: Sequence[Mapping[str, object]],
    output_path: Path,
    log=print,
) -> None:
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping specialization map: {exc}")
        return
    models = [model for model in ATTENTION_MODEL_NAMES if model in split_summaries]
    if not models:
        return
    fig, axes = plt.subplots(1, len(models), figsize=(max(4.0 * len(models), 8), 3.8), sharex=True, sharey=True)
    if len(models) == 1:
        axes = [axes]
    selected_keys = {
        (str(row["model"]), int(row["layer"]), int(row["head"])): str(row["category"])
        for row in selected_rows
    }
    for ax, model in zip(axes, models):
        table = alpha_summary_table(split_summaries[model])
        xs, ys, colors, labels = [], [], [], []
        for (layer, head), metrics in table.items():
            if M_SYMBOLIC not in metrics or M_PE_EQUIVARIANT not in metrics:
                continue
            xs.append(metrics[M_SYMBOLIC])
            ys.append(metrics[M_PE_EQUIVARIANT])
            colors.append(metrics.get(M_POSITIONAL, 0.0))
            labels.append((layer, head))
        scatter = ax.scatter(
            xs,
            ys,
            c=colors,
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
            s=52,
            edgecolor="black",
            linewidth=0.2,
        )
        for x, y, (layer, head) in zip(xs, ys, labels):
            if (model, layer, head) in selected_keys:
                ax.text(x + 0.012, y + 0.012, f"L{layer}H{head}", fontsize=7)
        ax.set_title(model)
        ax.set_xlabel("symbolic score")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("structural score")
    fig.colorbar(scatter, ax=axes, label="positional score", fraction=0.03, pad=0.02)
    fig.suptitle("Head specialization map on val16")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_model_feature_profiles(
    model_name: str,
    selected_rows: Sequence[Mapping[str, object]],
    profile_rows: Sequence[Mapping[str, object]],
    output_path: Path,
    log=print,
) -> None:
    selected = [row for row in selected_rows if str(row["model"]) == model_name]
    if not selected:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping feature profile plot: {exc}")
        return
    profile = {
        (int(row["layer"]), int(row["head"])): row
        for row in profile_rows
        if str(row["model"]) == model_name
    }
    features = ["self_mass", "edge_mass", "bridge_mass", "nonedge_mass", "same_symbol_mass", "far_mass"]
    labels = [feat.replace("_mass", "").replace("_", "\n") for feat in features]
    fig, axes = plt.subplots(len(selected), 1, figsize=(7.2, max(2.0, 1.65 * len(selected))), sharex=True)
    if len(selected) == 1:
        axes = [axes]
    for ax, row in zip(axes, selected):
        key = (int(row["layer"]), int(row["head"]))
        values = [float(profile.get(key, {}).get(feat, float("nan"))) for feat in features]
        ax.bar(range(len(features)), values, color="#80b1d3")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(f"L{key[0]}H{key[1]}", rotation=0, ha="right", va="center")
        ax.set_title(str(row["category"]), fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(range(len(features)))
    axes[-1].set_xticklabels(labels)
    fig.suptitle(f"{model_name}: attention feature mass profiles")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_stability_rows(
    split_summaries: Mapping[str, Mapping[str, Sequence[Mapping[str, object]]]],
) -> list[dict[str, object]]:
    rows = []
    for model, by_split in split_summaries.items():
        base = alpha_summary_table(by_split.get("val16", []))
        for target_split in ["test128", "test256"]:
            target = alpha_summary_table(by_split.get(target_split, []))
            for key, base_metrics in sorted(base.items()):
                if key not in target:
                    continue
                layer, head = key
                sym_base = base_metrics.get(M_SYMBOLIC, float("nan"))
                sym_target = target[key].get(M_SYMBOLIC, float("nan"))
                str_base = base_metrics.get(M_PE_EQUIVARIANT, float("nan"))
                str_target = target[key].get(M_PE_EQUIVARIANT, float("nan"))
                pos_base = base_metrics.get(M_POSITIONAL, float("nan"))
                pos_target = target[key].get(M_POSITIONAL, float("nan"))
                rows.append(
                    {
                        "model": model,
                        "target_split": target_split,
                        "layer": layer,
                        "head": head,
                        "symbolic_val16": sym_base,
                        "symbolic_target": sym_target,
                        "symbolic_delta": sym_target - sym_base,
                        "symbolic_abs_delta": abs(sym_target - sym_base),
                        "structural_val16": str_base,
                        "structural_target": str_target,
                        "structural_delta": str_target - str_base,
                        "structural_abs_delta": abs(str_target - str_base),
                        "positional_val16": pos_base,
                        "positional_target": pos_target,
                        "positional_abs_delta": abs(pos_target - pos_base),
                        "combined_abs_delta": abs(sym_target - sym_base) + abs(str_target - str_base),
                    }
                )
    return rows


def build_model_stability_rows(
    stability_rows: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    perf = {str(row["model"]): row for row in summaries}
    buckets: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in stability_rows:
        buckets.setdefault((str(row["model"]), str(row["target_split"])), []).append(row)
    out = []
    for (model, split), rows in sorted(buckets.items()):
        summary = perf.get(model, {})
        val_f1 = float(summary.get("val16_metrics", {}).get("f1", float("nan")))
        target_key = f"{split}_metrics"
        target_f1 = float(summary.get(target_key, {}).get("f1", float("nan")))
        out.append(
            {
                "model": model,
                "target_split": split,
                "mean_symbolic_abs_delta": sum(float(row["symbolic_abs_delta"]) for row in rows) / len(rows),
                "mean_structural_abs_delta": sum(float(row["structural_abs_delta"]) for row in rows) / len(rows),
                "mean_combined_abs_delta": sum(float(row["combined_abs_delta"]) for row in rows) / len(rows),
                "max_combined_abs_delta": max(float(row["combined_abs_delta"]) for row in rows),
                "val16_f1": val_f1,
                "target_f1": target_f1,
                "f1_drop": val_f1 - target_f1,
                "n_heads": len(rows),
            }
        )
    return out


def plot_stability_vs_performance(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
    log=print,
) -> None:
    if not rows:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping stability/performance plot: {exc}")
        return
    splits = ["test128", "test256"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, split in zip(axes, splits):
        data = [row for row in rows if str(row["target_split"]) == split]
        xs = [float(row["mean_combined_abs_delta"]) for row in data]
        ys = [float(row["f1_drop"]) for row in data]
        ax.scatter(xs, ys, s=60, color="#fb8072", edgecolor="black", linewidth=0.35)
        for row, x, y in zip(data, xs, ys):
            ax.text(x + 0.002, y + 0.002, str(row["model"]), fontsize=8)
        corr = pearson(xs, ys)
        ax.set_title(f"{split}: r={corr:.2f}" if not math.isnan(corr) else split)
        ax.set_xlabel("mean |delta symbolic| + |delta structural|")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("F1 drop from val16")
    fig.suptitle("Specialization instability vs size-generalization drop")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_stability_heatmaps(
    stability_rows: Sequence[Mapping[str, object]],
    output_path: Path,
    log=print,
) -> None:
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping stability heatmaps: {exc}")
        return
    models = sorted({str(row["model"]) for row in stability_rows})
    if not models:
        return
    fig, axes = plt.subplots(len(models), 2, figsize=(8.4, max(2.1, 1.8 * len(models))))
    if len(models) == 1:
        axes = axes.reshape(1, 2)
    for row_idx, model in enumerate(models):
        for col_idx, metric in enumerate(["symbolic_abs_delta", "structural_abs_delta"]):
            data = [
                row
                for row in stability_rows
                if str(row["model"]) == model and str(row["target_split"]) == "test256"
            ]
            max_layer = max([int(row["layer"]) for row in data], default=0)
            max_head = max([int(row["head"]) for row in data], default=0)
            mat = torch.full((max_layer + 1, max_head + 1), float("nan"))
            for row in data:
                mat[int(row["layer"]), int(row["head"])] = float(row[metric])
            ax = axes[row_idx][col_idx]
            im = ax.imshow(mat.numpy(), vmin=0.0, vmax=1.0, aspect="auto", cmap="inferno")
            ax.set_title(f"{model}: {metric} to test256", fontsize=8)
            ax.set_xlabel("head")
            ax.set_ylabel("layer")
            ax.set_xticks(range(max_head + 1))
            ax.set_yticks(range(max_layer + 1))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_selected_head_distributions(
    model_name: str,
    selected_rows: Sequence[Mapping[str, object]],
    raw_rows: Sequence[Mapping[str, object]],
    output_path: Path,
    log=print,
) -> None:
    selected = [row for row in selected_rows if str(row["model"]) == model_name]
    if not selected:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping distribution plot: {exc}")
        return
    fig, ax = plt.subplots(figsize=(max(8.0, 1.0 * len(selected)), 4.4))
    positions, data, colors = [], [], []
    labels = []
    for idx, row in enumerate(selected):
        layer = int(row["layer"])
        head = int(row["head"])
        sym = raw_score_values(raw_rows, layer, head, M_SYMBOLIC)
        structural = raw_score_values(raw_rows, layer, head, M_PE_EQUIVARIANT)
        positions.extend([idx * 3.0, idx * 3.0 + 0.9])
        data.extend([sym, structural])
        colors.extend(["#8dd3c7", "#bebada"])
        labels.append(f"L{layer}H{head}\n{str(row['category']).replace('_', ' ')}")
    box = ax.boxplot(data, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    ax.set_xticks([idx * 3.0 + 0.45 for idx in range(len(selected))])
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("per-graph alpha-weighted score")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#8dd3c7", label="symbolic"),
            plt.Rectangle((0, 0), 1, 1, color="#bebada", label="structural"),
        ],
        frameon=False,
        loc="lower right",
    )
    fig.suptitle(f"{model_name}: specialization distribution across val16 graph inputs")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def run_specialization_analysis(
    model_names: Sequence[str],
    splits: Mapping[str, BridgeDataset],
    summaries: Sequence[Mapping[str, object]],
    cfg: PilotConfig,
    output_root: Path,
    device: torch.device,
    args: argparse.Namespace,
    log=print,
) -> None:
    if args.skip_specialization_analysis:
        return
    model_names = [model for model in model_names if model in ATTENTION_MODEL_NAMES]
    if not model_names:
        log("[analysis] no attention-based models selected; skipping specialization analysis")
        return
    use_amp = cfg.amp and device.type == "cuda"
    suite_dir = output_root / cfg.preset
    analysis_dir = suite_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    split_summaries: dict[str, dict[str, Sequence[Mapping[str, object]]]] = {}
    raw_by_model_split: dict[str, dict[str, Sequence[Mapping[str, object]]]] = {}
    all_distribution_rows: list[dict[str, object]] = []
    all_profile_rows: list[dict[str, object]] = []
    all_locality_rows: list[dict[str, object]] = []
    all_locality_summary_rows: list[dict[str, object]] = []

    for model_name in model_names:
        model, model_cfg = load_trained_model(model_name, cfg, output_root, device)
        split_summaries[model_name] = {}
        raw_by_model_split[model_name] = {}
        for split_name in ["val16", "test128", "test256"]:
            raw_rows, summary_rows = compute_or_load_split_alpha(
                model,
                model_name,
                split_name,
                splits[split_name],
                cfg,
                model_cfg,
                output_root,
                device,
                use_amp,
                args,
                log=log,
            )
            raw_by_model_split[model_name][split_name] = raw_rows
            split_summaries[model_name][split_name] = summary_rows
            all_distribution_rows.extend(summarize_score_distributions(raw_rows))

        profile_path = (
            output_root / cfg.preset / model_name / f"seed{cfg.seed}" / "attention_feature_profiles_val16.csv"
        )
        if profile_path.exists() and not args.force_analysis:
            profile_rows = read_csv_rows(profile_path)
        else:
            max_graphs, batch_size = split_analysis_limits(args, "val16", cfg)
            log(f"[analysis] computing attention feature profiles model={model_name}")
            profile_rows = compute_attention_feature_profiles(
                model,
                splits["val16"],
                model_name,
                model_cfg,
                device,
                use_amp,
                max_graphs=max_graphs,
                batch_size=batch_size,
            )
            write_csv_rows(profile_path, profile_rows)
        all_profile_rows.extend(profile_rows)

        run_dir = output_root / cfg.preset / model_name / f"seed{cfg.seed}"
        locality_path = run_dir / "attention_locality_val16.csv"
        locality_summary_path = run_dir / "attention_locality_summary_val16.csv"
        if locality_path.exists() and locality_summary_path.exists() and not args.force_analysis:
            locality_rows = read_csv_rows(locality_path)
            locality_summary_rows = read_csv_rows(locality_summary_path)
        else:
            max_graphs, batch_size = split_analysis_limits(args, "val16", cfg)
            log(f"[analysis] computing attention locality model={model_name}")
            locality_rows = compute_attention_locality_rows(
                model,
                splits["val16"],
                model_name,
                model_cfg,
                device,
                use_amp,
                max_graphs=max_graphs,
                batch_size=batch_size,
                split_name="val16",
            )
            locality_summary_rows = summarize_locality_rows(locality_rows)
            write_csv_rows(locality_path, locality_rows)
            write_csv_rows(locality_summary_path, locality_summary_rows)
        all_locality_rows.extend(locality_rows)
        all_locality_summary_rows.extend(locality_summary_rows)

    stability_rows = build_stability_rows(split_summaries)
    model_stability_rows = build_model_stability_rows(stability_rows, summaries)
    selected_rows: list[dict[str, object]] = []
    for model_name in model_names:
        selected_rows.extend(
            select_standout_heads(
                model_name,
                split_summaries[model_name].get("val16", []),
                stability_rows,
                args.standout_heads_per_model,
            )
        )

    write_csv_rows(analysis_dir / "specialization_distribution_summary.csv", all_distribution_rows)
    write_csv_rows(analysis_dir / "attention_feature_profiles_val16.csv", all_profile_rows)
    write_csv_rows(analysis_dir / "attention_locality_val16.csv", all_locality_rows)
    write_csv_rows(analysis_dir / "attention_locality_summary_val16.csv", all_locality_summary_rows)
    write_csv_rows(analysis_dir / "specialization_size_stability.csv", stability_rows)
    write_csv_rows(analysis_dir / "model_stability_vs_ood.csv", model_stability_rows)
    write_csv_rows(analysis_dir / "standout_heads.csv", selected_rows)

    val_summaries = {model: split_summaries[model].get("val16", []) for model in model_names}
    plot_specialization_map(val_summaries, selected_rows, analysis_dir / "specialization_map_val16.png", log=log)
    plot_locality_metric_connection(
        all_locality_summary_rows,
        val_summaries,
        analysis_dir / "locality_vs_specialization_metrics.png",
        log=log,
    )
    plot_stability_vs_performance(model_stability_rows, analysis_dir / "stability_vs_ood_performance.png", log=log)
    plot_stability_heatmaps(stability_rows, analysis_dir / "specialization_size_stability_heatmaps.png", log=log)
    example_dir = analysis_dir / "head_examples"
    for model_name in model_names:
        model, _model_cfg = load_trained_model(model_name, cfg, output_root, device)
        model_selected = [row for row in selected_rows if str(row["model"]) == model_name]
        plot_head_examples(
            model,
            splits["val16"],
            model_name,
            model_selected,
            raw_by_model_split[model_name].get("val16", []),
            cfg,
            example_dir,
            device,
            use_amp,
            args.analysis_examples,
            log=log,
        )
        plot_selected_head_distributions(
            model_name,
            model_selected,
            raw_by_model_split[model_name].get("val16", []),
            analysis_dir / f"{model_name}_standout_head_distributions.png",
            log=log,
        )
        plot_model_feature_profiles(
            model_name,
            model_selected,
            all_profile_rows,
            analysis_dir / f"{model_name}_attention_feature_profiles.png",
            log=log,
        )
        plot_locality_distributions(
            model_name,
            all_locality_rows,
            analysis_dir / f"{model_name}_attention_locality_distributions.png",
            log=log,
        )
    log(f"[analysis] wrote specialization analysis to {analysis_dir}")


def make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    warmup_steps = max(1, warmup_steps)
    total_steps = max(total_steps, warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_model(
    model_name: str,
    splits: Mapping[str, BridgeDataset],
    cfg: PilotConfig,
    output_root: Path,
    device: torch.device,
    force_retrain: bool,
    force_metrics: bool,
    skip_alpha_metrics: bool,
    log=print,
) -> dict[str, object]:
    set_seed(cfg.seed)
    run_dir = output_root / cfg.preset / model_name / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = RunLogger(run_dir / "run.log")
    model_cfg = model_config_for(model_name, cfg)
    family = model_family(model_name)
    attention_mode = model_attention_mode(model_name)
    run_log(f"[run] model={model_name} device={device}")
    run_log(f"[config] {MODEL_CONFIG_NOTES[family]}")
    if family in GT_FAMILIES:
        run_log(f"[attention] {attention_mode}: {ATTENTION_MODE_NOTES[attention_mode]}")
    write_json(
        run_dir / "config.json",
        asdict(model_cfg)
        | {
            "model": model_name,
            "family": family,
            "attention_mode": attention_mode,
            "model_note": MODEL_CONFIG_NOTES[family],
        },
    )
    write_json(
        run_dir / "dataset_stats.json",
        {name: dataset_stats(dataset) for name, dataset in splits.items()},
    )

    train_loader = make_loader(splits["train16"], cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(splits["val16"], cfg.eval_batch_size, shuffle=False, seed=cfg.seed)
    test128_loader = make_loader(splits["test128"], cfg.eval_batch_size, shuffle=False, seed=cfg.seed)
    test256_loader = make_loader(splits["test256"], cfg.eval_batch_size, shuffle=False, seed=cfg.seed)

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
    if best_path.exists() and summary_path.exists() and not force_retrain:
        run_log("[resume] found best.pt and summary.json; skipping training")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        best_epoch = int(ckpt.get("epoch", 0))
        best_val_f1 = float(ckpt.get("val_metrics", {}).get("f1", float("nan")))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        total_steps = cfg.max_epochs * max(1, len(train_loader))
        warmup_steps = cfg.warmup_epochs * max(1, len(train_loader))
        scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
        best_val_f1 = -1.0
        best_epoch = 0
        bad_epochs = 0
        fields = ["epoch", "lr", "train_loss", "val_f1", "val_acc", "val_precision", "val_recall", "seconds"]
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
                val_logits, val_labels = predict_all(model, val_loader, device, use_amp)
                val_metrics = metric_values(val_logits, val_labels)
                row = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": total_loss / max(1, total_edges),
                    "val_f1": val_metrics["f1"],
                    "val_acc": val_metrics["acc"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "seconds": time.time() - started,
                }
                writer.writerow(row)
                handle.flush()
                run_log(
                    f"[epoch {epoch:03d}] loss={row['train_loss']:.5f} "
                    f"val_f1={row['val_f1']:.4f} val_acc={row['val_acc']:.4f} "
                    f"time={row['seconds']:.1f}s"
                )
                if val_metrics["f1"] > best_val_f1 + cfg.min_delta:
                    best_val_f1 = val_metrics["f1"]
                    best_epoch = epoch
                    bad_epochs = 0
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "epoch": epoch,
                            "val_metrics": val_metrics,
                            "trainable_parameters": n_params,
                        },
                        best_path,
                    )
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg.patience:
                        run_log(f"[early-stop] best_epoch={best_epoch}")
                        break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    val_logits, val_labels = predict_all(model, val_loader, device, use_amp)
    test128_logits, test128_labels = predict_all(model, test128_loader, device, use_amp)
    test256_logits, test256_labels = predict_all(model, test256_loader, device, use_amp)
    val_metrics = metric_values(val_logits, val_labels)
    test128_metrics = metric_values(test128_logits, test128_labels)
    test256_metrics = metric_values(test256_logits, test256_labels)
    detour_rows: list[dict[str, object]] = []
    for split_name, dataset in [
        ("val16", splits["val16"]),
        ("test128", splits["test128"]),
        ("test256", splits["test256"]),
    ]:
        detour_rows.extend(
            evaluate_detour_metrics(
                model,
                dataset,
                split_name,
                model_name,
                cfg.eval_batch_size,
                device,
                use_amp,
                cfg.seed,
            )
        )
    summary = {
        "model": model_name,
        "seed": cfg.seed,
        "trainable_parameters": n_params,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "val16_metrics": val_metrics,
        "test128_metrics": test128_metrics,
        "test256_metrics": test256_metrics,
        "detour_metrics": detour_rows,
        "best_checkpoint": str(best_path),
    }
    write_json(summary_path, summary)
    write_csv_rows(run_dir / "detour_metrics.csv", detour_rows)
    plot_training_curves(metrics_path, run_dir / "training_curves.png", log=run_log)

    alpha_summary_rows = []
    alpha_summary_path = run_dir / "alpha_permutation_summary_best.csv"
    if not skip_alpha_metrics and model_name in ATTENTION_MODEL_NAMES:
        if force_metrics or force_retrain or not alpha_summary_path.exists():
            run_log("[metrics] computing alpha-weighted attention metrics")
            alpha_rows = compute_alpha_metrics(
                model,
                splits["val16"],
                model_name,
                model_cfg,
                device,
                use_amp,
            )
            alpha_summary_rows = summarize_metric_rows(alpha_rows)
            write_csv_rows(run_dir / "alpha_permutation_metrics_best.csv", alpha_rows)
            write_csv_rows(alpha_summary_path, alpha_summary_rows)
        else:
            alpha_summary_rows = read_csv_rows(alpha_summary_path)
    elif model_name not in ATTENTION_MODEL_NAMES:
        run_log("[metrics] skipping alpha-weighted attention metrics for non-attention GNN+ baseline")
    run_log(
        f"[done] val_f1={val_metrics['f1']:.4f} "
        f"test128_f1={test128_metrics['f1']:.4f} test256_f1={test256_metrics['f1']:.4f}"
    )
    return summary | {"alpha_summary_rows": alpha_summary_rows, "detour_rows": detour_rows}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Bridges-Easy GT locality-ablation 3-seed training run.")
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated training seeds.")
    parser.add_argument("--data-seed", type=int, default=0, help="Fixed seed for train/val/test graph generation.")
    parser.add_argument("--train-graphs", type=int, default=10000)
    parser.add_argument("--val-graphs", type=int, default=2000)
    parser.add_argument("--test128-graphs", type=int, default=1000)
    parser.add_argument("--test256-graphs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--metric-graphs", type=int, default=64)
    parser.add_argument("--metric-batch-size", type=int, default=16)
    parser.add_argument("--metric-perms", type=int, default=8)
    parser.add_argument("--metric-alpha-tau", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument(
        "--drive-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/graph_specialisation_metrics/bridges_easy_locality_ablation_3seed"),
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
    parser.add_argument("--force-metrics", action="store_true")
    parser.add_argument("--skip-alpha-metrics", dest="skip_alpha_metrics", action="store_true", default=True)
    parser.add_argument("--run-alpha-metrics", dest="skip_alpha_metrics", action="store_false")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument(
        "--skip-specialization-analysis",
        dest="skip_specialization_analysis",
        action="store_true",
        default=True,
    )
    parser.add_argument("--run-specialization-analysis", dest="skip_specialization_analysis", action="store_false")
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument("--analysis-graphs16", type=int, default=96)
    parser.add_argument("--analysis-graphs128", type=int, default=48)
    parser.add_argument("--analysis-graphs256", type=int, default=24)
    parser.add_argument("--analysis-examples", type=int, default=4)
    parser.add_argument("--standout-heads-per-model", type=int, default=8)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args(argv)
    return args


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


def parse_seed_list(text: str, fallback_seed: int) -> list[int]:
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if not parts:
        return [int(fallback_seed)]
    seeds = []
    for part in parts:
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(part))
    seen: set[int] = set()
    deduped = []
    for seed in seeds:
        if seed not in seen:
            deduped.append(seed)
            seen.add(seed)
    return deduped


def cfg_from_args(args: argparse.Namespace) -> PilotConfig:
    cfg = PilotConfig(
        seed=args.seed,
        data_seed=args.data_seed,
        train_graphs=args.train_graphs,
        val_graphs=args.val_graphs,
        test128_graphs=args.test128_graphs,
        test256_graphs=args.test256_graphs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        metric_graphs=args.metric_graphs,
        metric_batch_size=args.metric_batch_size,
        metric_perms=args.metric_perms,
        metric_alpha_tau=args.metric_alpha_tau,
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
                "metric_graphs": min(cfg.metric_graphs, 16),
                "metric_batch_size": min(cfg.metric_batch_size, 4),
                "metric_perms": min(cfg.metric_perms, 2),
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
    seeds = parse_seed_list(args.seeds, args.seed)
    if args.fast_dev_run:
        seeds = seeds[:1]
    set_seed(cfg.data_seed)
    device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    mount_drive(args.drive_mount, enabled=not args.no_mount_drive)
    cache_root = args.cache_root or args.drive_dir / "cache"
    output_root = args.output_root or args.drive_dir / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    suite_dir = output_root / cfg.preset
    top_log = RunLogger(suite_dir / "run.log")
    top_log(f"[setup] device={device} amp={cfg.amp and device.type == 'cuda'}")
    top_log(f"[setup] training_seeds={seeds} data_seed={cfg.data_seed}")
    top_log(f"[setup] cache_root={cache_root}")
    top_log(f"[setup] output_root={output_root}")
    splits = {
        "train16": load_or_generate_split(
            cache_root, "train", 16, cfg.train_graphs, cfg.data_seed, args.force_regenerate, top_log
        ),
        "val16": load_or_generate_split(
            cache_root, "val", 16, cfg.val_graphs, cfg.data_seed + 11, args.force_regenerate, top_log
        ),
        "test128": load_or_generate_split(
            cache_root, "test128", 128, cfg.test128_graphs, cfg.data_seed + 23, args.force_regenerate, top_log
        ),
        "test256": load_or_generate_split(
            cache_root, "test256", 256, cfg.test256_graphs, cfg.data_seed + 37, args.force_regenerate, top_log
        ),
    }
    write_json(suite_dir / "resolved_config.json", asdict(cfg) | {"training_seeds": seeds})
    write_json(
        suite_dir / "dataset_stats.json",
        {name: dataset_stats(dataset) for name, dataset in splits.items()},
    )

    models = list(MODEL_NAMES) if args.model == "all" else [args.model]
    write_and_log_model_design(models, cfg, splits, suite_dir / "model_design_table.csv", log=top_log)
    summaries = []
    detour_by_model: list[dict[str, object]] = []
    alpha_by_model: dict[str, Sequence[Mapping[str, object]]] = {}
    for model_name in models:
        for seed in seeds:
            seed_cfg = replace(cfg, seed=seed)
            if args.analysis_only:
                run_dir = suite_dir / model_name / f"seed{seed}"
                summary_path = run_dir / "summary.json"
                if not summary_path.exists():
                    raise FileNotFoundError(f"--analysis-only requested but summary is missing: {summary_path}")
                result = json.loads(summary_path.read_text(encoding="utf-8"))
                if model_name in ATTENTION_MODEL_NAMES:
                    alpha_path = run_dir / "alpha_permutation_summary_val16_best.csv"
                    if not alpha_path.exists():
                        alpha_path = run_dir / "alpha_permutation_summary_best.csv"
                    alpha_by_model[model_name] = read_csv_rows(alpha_path)
                else:
                    alpha_by_model[model_name] = []
                detour_path = run_dir / "detour_metrics.csv"
                if detour_path.exists():
                    detour_by_model.extend(read_csv_rows(detour_path))
                else:
                    detour_by_model.extend(result.get("detour_metrics", []))
                summaries.append(result)
            else:
                result = train_one_model(
                    model_name,
                    splits,
                    seed_cfg,
                    output_root,
                    device,
                    force_retrain=args.force_retrain,
                    force_metrics=args.force_metrics,
                    skip_alpha_metrics=args.skip_alpha_metrics,
                    log=top_log,
                )
                alpha_by_model[model_name] = result.pop("alpha_summary_rows", [])
                detour_by_model.extend(result.pop("detour_rows", []))
                summaries.append(result)
    aggregate_rows = aggregate_repeat_rows(models, summaries)
    best_rows = select_and_save_best_trials(models, summaries, suite_dir, log=top_log)
    write_json(
        suite_dir / "summary.json",
        {"runs": summaries, "repeat_summary": aggregate_rows, "best_trials": best_rows},
    )
    write_csv_rows(suite_dir / "repeat_performance_summary.csv", aggregate_rows)
    write_csv_rows(suite_dir / "best_trials.csv", best_rows)
    write_csv_rows(suite_dir / "detour_metrics.csv", detour_by_model)
    plot_repeat_performance_summary(aggregate_rows, suite_dir / "performance_summary_3seed_mean_std.png", log=top_log)
    plot_best_trial_performance(best_rows, suite_dir / "performance_summary_best_trials.png", log=top_log)
    plot_seed_repeat_scatter(summaries, suite_dir / "seed_repeat_scatter.png", log=top_log)
    plot_generalization_drop(aggregate_rows, suite_dir / "size_generalization_gap.png", log=top_log)
    plot_detour_performance(detour_by_model, suite_dir / "detour_performance_test128.png", "test128", log=top_log)
    plot_detour_performance(detour_by_model, suite_dir / "detour_performance_test256.png", "test256", log=top_log)
    if not args.skip_alpha_metrics:
        plot_alpha_heatmaps(alpha_by_model, suite_dir / "alpha_pos_symbol_heatmaps.png", log=top_log)
    run_specialization_analysis(
        models,
        splits,
        summaries,
        cfg,
        output_root,
        device,
        args,
        log=top_log,
    )
    top_log("[done] Bridges-Easy size-generalization pilot complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
