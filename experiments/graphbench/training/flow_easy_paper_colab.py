#!/usr/bin/env python3
"""Standalone GraphBench Flow-Easy paper runner for Colab.

This script trains small graph models on a self-contained GraphBench-style
`flow_easy` task. It trains on 16-node graphs and evaluates size generalisation
on 64, 128, and 256-node graphs with both train-generator and all-generator
OOD suites.

The task is graph-level max-flow regression. Each generated graph is directed,
has a marked source and sink, directed edge capacities, a normalized target
flow fraction, and min-cut metadata saved for later interpretability work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
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


TASK_NAME = "flow_easy_paper"
RUN_IMPLEMENTATION_VERSION = "rigor_v2_common_readout_masked_pairs_dual_eval"
GT_FAMILIES = ("graphormer", "graphgps", "grit", "csa")
ATTENTION_MODES = ("full", "k1", "k2")
ATTENTION_MODEL_NAMES = tuple(f"{family}_{mode}" for family in GT_FAMILIES for mode in ATTENTION_MODES)
GNN_PLUS_MODEL_NAMES = ("gcn_plus", "gatedgcn_plus")
MODEL_NAMES = (*ATTENTION_MODEL_NAMES, *GNN_PLUS_MODEL_NAMES)
DEPTH_SWEEP_MODELS = ("grit_full", "grit_k1", "csa_full", "csa_k1", "graphgps_full", "gatedgcn_plus")
VAL_SPLIT = "val16"
SAMEGEN_SPLITS = ("val16", "samegen64", "samegen128", "samegen256")
ALLGEN_SPLITS = ("val16", "allgen64", "allgen128", "allgen256")
EVAL_SPLITS = ("val16", "samegen64", "samegen128", "samegen256", "allgen64", "allgen128", "allgen256")
OOD_EVAL_SPLITS = tuple(split for split in EVAL_SPLITS if split != VAL_SPLIT)
SPLIT_SIZE = {
    "val16": 16,
    "samegen64": 64,
    "samegen128": 128,
    "samegen256": 256,
    "allgen64": 64,
    "allgen128": 128,
    "allgen256": 256,
}
METRIC_NAMES = ("mae", "mse", "r2", "spearman", "norm_rel_error", "eval_seconds_per_graph")

RW_STEPS = 4
SPD_CAP = 8
NODE_SYMBOLS = 3  # ordinary, source, sink
PAIR_RAW_DIM = 3 + 2 + (SPD_CAP + 2) + (RW_STEPS + 1)

GENERATOR_NAMES = (
    "erdos-renyi",
    "newman-watts-strogatz",
    "barabasi-albert",
    "dual-barabasi-albert",
    "powerlaw-cluster",
    "stochastic-block-model",
)

CONFIG_TRAIN_FLOW = {
    "erdos-renyi": (1, 0.16),
    "newman-watts-strogatz": (1, 4, 0.2),
    "barabasi-albert": (1, 3),
    "dual-barabasi-albert": (1, 4, 2, 0.3),
    "powerlaw-cluster": (1, 5, 0.4),
    "stochastic-block-model": (1, [0.5, 0.5], [[0.35, 0.3], [0.3, 0.35]]),
}
CONFIG_TEST_FLOW = {
    "erdos-renyi": (1, 0.16),
    "newman-watts-strogatz": (1, 4, 0.2),
    "barabasi-albert": (1, 3),
    "dual-barabasi-albert": (1, 4, 2, 0.3),
    "powerlaw-cluster": (1, 5, 0.4),
    "stochastic-block-model": (1, [0.5, 0.5], [[0.35, 0.3], [0.3, 0.35]]),
}
SAMPLING_TRAIN_EASY = [1, 0, 1, 0, 1, 0]
SAMPLING_TEST_EASY = [1, 1, 1, 1, 1, 1]

MODEL_CONFIG_NOTES = {
    "graphormer": "Graphormer with degree, SPD, directed edge presence, and capacity bias.",
    "graphgps": "GraphGPS with RWSE and dense GINE local branch using directed capacities.",
    "grit": "GRIT with static pair features and pair evolution.",
    "csa": "Static pair-feature attention bias, no pair evolution and no chromatic attention.",
    "gcn_plus": "Local GCN+ baseline with RWSE and directed-capacity messages.",
    "gatedgcn_plus": "Local GatedGCN+ baseline with learned directed-capacity edge states.",
}

PE_NOTES = {
    "graphormer": "degree + SPD attention bias + directed capacity bias",
    "graphgps": "RWSE + degree; DenseGINE local branch",
    "grit": "RRWP/SPD/directed-capacity pair features + degree",
    "csa": "RRWP/SPD/directed-capacity static pair features + degree",
    "gcn_plus": "RWSE + degree + local directed-capacity messages",
    "gatedgcn_plus": "RWSE + degree + directed-capacity edge states",
}

ATTENTION_MODE_NOTES = {
    "full": "all valid node pairs; graph token enabled for Graphormer.",
    "k1": "<=1-hop node pairs only over symmetrized topology.",
    "k2": "<=2-hop node pairs only over symmetrized topology.",
}


@dataclass(frozen=True)
class PilotConfig:
    preset: str = TASK_NAME
    seed: int = 0
    train_graphs: int = 50000
    val_graphs: int = 8000
    test64_graphs: int = 3000
    test128_graphs: int = 2000
    test256_graphs: int = 600
    train_nodes: int = 16
    test64_nodes: int = 64
    test128_nodes: int = 128
    test256_nodes: int = 256
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 20
    min_delta: float = 1.0e-5
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    hidden_dim: int = 48
    layers: int = 4
    heads: int = 4
    attention_mode: str = "full"
    dropout: float = 0.0
    attn_dropout: float = 0.1
    amp: bool = True


@dataclass
class FlowGraph:
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_capacity: torch.Tensor
    topo_adj: torch.Tensor
    cap_mat: torch.Tensor
    degree: torch.Tensor
    cap_in: torch.Tensor
    cap_out: torch.Tensor
    spd: torch.Tensor
    rwse: torch.Tensor
    rrwp: torch.Tensor
    source: int
    sink: int
    target: float
    raw_flow: float
    raw_flow_norm: float
    source_out_capacity: float
    sink_in_capacity: float
    bound_capacity: float
    generator: str
    source_side: torch.Tensor
    cut_edge_index: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.node_type.numel())


@dataclass
class FlowBatch:
    node_type: torch.Tensor
    node_mask: torch.Tensor
    topo_adj: torch.Tensor
    cap_mat: torch.Tensor
    degree: torch.Tensor
    cap_in: torch.Tensor
    cap_out: torch.Tensor
    spd: torch.Tensor
    rwse: torch.Tensor
    rrwp: torch.Tensor
    pair_xi: torch.Tensor
    edge_batch: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_capacity: torch.Tensor
    graph_num_nodes: torch.Tensor
    source: torch.Tensor
    sink: torch.Tensor
    target: torch.Tensor
    raw_flow: torch.Tensor
    raw_flow_norm: torch.Tensor
    source_out_capacity: torch.Tensor
    sink_in_capacity: torch.Tensor
    bound_capacity: torch.Tensor
    generator_id: torch.Tensor
    source_side: torch.Tensor
    num_graphs: int
    max_nodes: int

    def to(self, device: torch.device) -> "FlowBatch":
        return FlowBatch(
            node_type=self.node_type.to(device),
            node_mask=self.node_mask.to(device),
            topo_adj=self.topo_adj.to(device),
            cap_mat=self.cap_mat.to(device),
            degree=self.degree.to(device),
            cap_in=self.cap_in.to(device),
            cap_out=self.cap_out.to(device),
            spd=self.spd.to(device),
            rwse=self.rwse.to(device),
            rrwp=self.rrwp.to(device),
            pair_xi=self.pair_xi.to(device),
            edge_batch=self.edge_batch.to(device),
            edge_src=self.edge_src.to(device),
            edge_dst=self.edge_dst.to(device),
            edge_capacity=self.edge_capacity.to(device),
            graph_num_nodes=self.graph_num_nodes.to(device),
            source=self.source.to(device),
            sink=self.sink.to(device),
            target=self.target.to(device),
            raw_flow=self.raw_flow.to(device),
            raw_flow_norm=self.raw_flow_norm.to(device),
            source_out_capacity=self.source_out_capacity.to(device),
            sink_in_capacity=self.sink_in_capacity.to(device),
            bound_capacity=self.bound_capacity.to(device),
            generator_id=self.generator_id.to(device),
            source_side=self.source_side.to(device),
            num_graphs=self.num_graphs,
            max_nodes=self.max_nodes,
        )

    @property
    def pair_mask(self) -> torch.Tensor:
        return self.node_mask[:, :, None] & self.node_mask[:, None, :]


class FlowDataset(Dataset[FlowGraph]):
    def __init__(self, graphs: Sequence[FlowGraph]) -> None:
        self.graphs = list(graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> FlowGraph:
        return self.graphs[index]


def graph_to_payload(graph: FlowGraph) -> dict[str, object]:
    return {
        "node_type": graph.node_type,
        "edge_index": graph.edge_index,
        "edge_capacity": graph.edge_capacity,
        "topo_adj": graph.topo_adj,
        "cap_mat": graph.cap_mat,
        "degree": graph.degree,
        "cap_in": graph.cap_in,
        "cap_out": graph.cap_out,
        "spd": graph.spd,
        "rwse": graph.rwse,
        "rrwp": graph.rrwp,
        "source": graph.source,
        "sink": graph.sink,
        "target": graph.target,
        "raw_flow": graph.raw_flow,
        "raw_flow_norm": graph.raw_flow_norm,
        "source_out_capacity": graph.source_out_capacity,
        "sink_in_capacity": graph.sink_in_capacity,
        "bound_capacity": graph.bound_capacity,
        "generator": graph.generator,
        "source_side": graph.source_side,
        "cut_edge_index": graph.cut_edge_index,
    }


def graph_from_payload(payload: object) -> FlowGraph:
    if not isinstance(payload, Mapping):
        payload = payload.__dict__
    return FlowGraph(
        node_type=payload["node_type"],
        edge_index=payload["edge_index"],
        edge_capacity=payload["edge_capacity"],
        topo_adj=payload["topo_adj"],
        cap_mat=payload["cap_mat"],
        degree=payload["degree"],
        cap_in=payload["cap_in"],
        cap_out=payload["cap_out"],
        spd=payload["spd"],
        rwse=payload["rwse"],
        rrwp=payload["rrwp"],
        source=int(payload["source"]),
        sink=int(payload["sink"]),
        target=float(payload["target"]),
        raw_flow=float(payload["raw_flow"]),
        raw_flow_norm=float(payload["raw_flow_norm"]),
        source_out_capacity=float(payload["source_out_capacity"]),
        sink_in_capacity=float(payload["sink_in_capacity"]),
        bound_capacity=float(payload["bound_capacity"]),
        generator=str(payload["generator"]),
        source_side=payload["source_side"],
        cut_edge_index=payload["cut_edge_index"],
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


def generate_base_graph(
    num_nodes: int,
    task_cfg: Mapping[str, tuple],
    sampling: Sequence[float],
    rng: random.Random,
):
    import networkx as nx

    generator = rng.choices(GENERATOR_NAMES, weights=sampling, k=1)[0]
    cfg = task_cfg[generator]
    seed = rng.randrange(2**31 - 1)
    if generator == "erdos-renyi":
        graph = nx.erdos_renyi_graph(num_nodes, float(cfg[1]), seed=seed)
    elif generator == "newman-watts-strogatz":
        graph = nx.newman_watts_strogatz_graph(num_nodes, min(num_nodes - 1, int(cfg[1])), float(cfg[2]), seed=seed)
    elif generator == "barabasi-albert":
        graph = nx.barabasi_albert_graph(num_nodes, max(1, min(num_nodes - 1, int(cfg[1]))), seed=seed)
    elif generator == "dual-barabasi-albert":
        graph = nx.dual_barabasi_albert_graph(
            num_nodes,
            max(1, min(num_nodes - 1, int(cfg[1]))),
            max(1, min(num_nodes - 1, int(cfg[2]))),
            float(cfg[3]),
            seed=seed,
        )
    elif generator == "powerlaw-cluster":
        graph = nx.powerlaw_cluster_graph(num_nodes, max(1, min(num_nodes - 1, int(cfg[1]))), float(cfg[2]), seed=seed)
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
                other = rng.choice([j for j in range(len(components)) if j != idx])
                graph.add_edge(rng.choice(list(component)), rng.choice(list(components[other])))
    return graph, generator


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


def pair_features(topo_adj: torch.Tensor, spd: torch.Tensor, rrwp: torch.Tensor, cap_mat: torch.Tensor) -> torch.Tensor:
    n = int(topo_adj.size(0))
    eye = torch.eye(n, dtype=torch.bool, device=topo_adj.device)
    any_edge = topo_adj.float()
    semantic = torch.zeros(n, n, 3, dtype=torch.float32, device=topo_adj.device)
    semantic[..., 0] = eye.float()
    semantic[..., 1] = ((any_edge <= 0) & (~eye)).float()
    semantic[..., 2] = any_edge
    cap_pair = torch.stack([cap_mat.float(), cap_mat.float().t()], dim=-1)
    spd_oh = F.one_hot(spd.long().clamp(max=SPD_CAP + 1), num_classes=SPD_CAP + 2).float()
    return torch.cat([semantic, cap_pair, spd_oh, rrwp.float()], dim=-1)


def flow_graph_to_record(num_nodes: int, graph_idx: int, split: str, seed: int) -> FlowGraph:
    import networkx as nx

    uses_train_distribution = split in {"train", "val"} or split.startswith("samegen")
    task_cfg = CONFIG_TRAIN_FLOW if uses_train_distribution else CONFIG_TEST_FLOW
    sampling = SAMPLING_TRAIN_EASY if uses_train_distribution else SAMPLING_TEST_EASY
    split_offset = 0 if split in {"train", "val"} else (31 if split.startswith("samegen") else 17)
    rng = random.Random(seed + 104729 * graph_idx + split_offset)
    undirected, generator = generate_base_graph(num_nodes, task_cfg=task_cfg, sampling=sampling, rng=rng)
    directed = undirected.to_directed()
    for edge in directed.edges:
        directed.edges[edge]["capacity"] = round(rng.uniform(0.0, 3.0), 2)
    nodes = list(range(num_nodes))
    source = rng.choice(nodes)
    sink = rng.choice([node for node in nodes if node != source])
    raw_flow = float(nx.flow.maximum_flow_value(directed, source, sink, capacity="capacity"))
    cut_value, partition = nx.minimum_cut(directed, source, sink, capacity="capacity")
    source_set, sink_set = partition
    source_side = torch.zeros(num_nodes, dtype=torch.uint8)
    source_side[list(source_set)] = 1
    cut_edges = [(u, v) for u in source_set for v in directed.successors(u) if v in sink_set]

    cap_mat = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    topo_adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
    directed_edges: list[tuple[int, int]] = []
    capacities: list[float] = []
    for u, v, data in directed.edges(data=True):
        cap = float(data["capacity"]) / 3.0
        cap_mat[u, v] = cap
        topo_adj[u, v] = 1.0
        topo_adj[v, u] = 1.0
        directed_edges.append((int(u), int(v)))
        capacities.append(cap)
    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    edge_capacity = torch.tensor(capacities, dtype=torch.float32)
    cap_out = cap_mat.sum(dim=1)
    cap_in = cap_mat.sum(dim=0)
    source_out = float(cap_out[source])
    sink_in = float(cap_in[sink])
    bound = max(1.0e-6, min(source_out, sink_in))
    raw_flow_norm = raw_flow / 3.0
    target = max(0.0, min(1.0, raw_flow_norm / bound))
    degree = topo_adj.sum(dim=-1)
    spd = shortest_path_buckets(topo_adj)
    rwse, rrwp = random_walk_features(topo_adj)
    node_type = torch.ones(num_nodes, dtype=torch.long)
    node_type[source] = 2
    node_type[sink] = 3
    cut_edge_index = (
        torch.tensor(cut_edges, dtype=torch.long).t().contiguous()
        if cut_edges
        else torch.empty(2, 0, dtype=torch.long)
    )
    return FlowGraph(
        node_type=node_type,
        edge_index=edge_index,
        edge_capacity=edge_capacity,
        topo_adj=topo_adj.to(torch.uint8),
        cap_mat=cap_mat.to(torch.float16),
        degree=degree.float(),
        cap_in=cap_in.float(),
        cap_out=cap_out.float(),
        spd=spd,
        rwse=rwse.to(torch.float16),
        rrwp=rrwp.to(torch.float16),
        source=int(source),
        sink=int(sink),
        target=float(target),
        raw_flow=float(raw_flow),
        raw_flow_norm=float(raw_flow_norm),
        source_out_capacity=float(source_out),
        sink_in_capacity=float(sink_in),
        bound_capacity=float(bound),
        generator=generator,
        source_side=source_side,
        cut_edge_index=cut_edge_index,
    )


def cache_path(cache_root: Path, split: str, nodes: int, count: int, seed: int) -> Path:
    return cache_root / f"flow_easy_{split}_n{nodes}_count{count}_seed{seed}_rw{RW_STEPS}.pt"


def load_or_generate_split(
    cache_root: Path,
    split: str,
    nodes: int,
    count: int,
    seed: int,
    force: bool,
    log=print,
) -> FlowDataset:
    cache_root.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_root, split, nodes, count, seed)
    if path.exists() and not force:
        log(f"[data] Loading cached {split} n={nodes}: {path}")
        raw = torch.load(path, map_location="cpu", weights_only=False)["graphs"]
        return FlowDataset([graph_from_payload(item) for item in raw])
    log(f"[data] Generating Flow-Easy split={split} n={nodes} count={count}")
    graphs = [flow_graph_to_record(nodes, idx, split, seed) for idx in range(count)]
    torch.save(
        {"graphs": [graph_to_payload(graph) for graph in graphs], "split": split, "nodes": nodes, "count": count, "seed": seed},
        path,
    )
    return FlowDataset(graphs)


def dataset_stats(dataset: Dataset[FlowGraph]) -> dict[str, float]:
    n_graphs = len(dataset)
    nodes = [dataset[i].num_nodes for i in range(n_graphs)]
    edges = [int(dataset[i].edge_index.size(1)) for i in range(n_graphs)]
    targets = torch.tensor([dataset[i].target for i in range(n_graphs)], dtype=torch.float32)
    raw = torch.tensor([dataset[i].raw_flow for i in range(n_graphs)], dtype=torch.float32)
    return {
        "graphs": float(n_graphs),
        "nodes_min": float(min(nodes) if nodes else 0),
        "nodes_max": float(max(nodes) if nodes else 0),
        "directed_edges_mean": float(sum(edges) / max(1, n_graphs)),
        "target_mean": float(targets.mean()) if targets.numel() else 0.0,
        "target_std": float(targets.std(unbiased=False)) if targets.numel() else 0.0,
        "raw_flow_mean": float(raw.mean()) if raw.numel() else 0.0,
        "raw_flow_std": float(raw.std(unbiased=False)) if raw.numel() else 0.0,
    }


def collate_graphs(graphs: Sequence[FlowGraph]) -> FlowBatch:
    bsz = len(graphs)
    max_nodes = max(graph.num_nodes for graph in graphs)
    node_type = torch.zeros(bsz, max_nodes, dtype=torch.long)
    node_mask = torch.zeros(bsz, max_nodes, dtype=torch.bool)
    topo_adj = torch.zeros(bsz, max_nodes, max_nodes, dtype=torch.float32)
    cap_mat = torch.zeros(bsz, max_nodes, max_nodes, dtype=torch.float32)
    degree = torch.zeros(bsz, max_nodes, dtype=torch.float32)
    cap_in = torch.zeros(bsz, max_nodes, dtype=torch.float32)
    cap_out = torch.zeros(bsz, max_nodes, dtype=torch.float32)
    spd = torch.full((bsz, max_nodes, max_nodes), SPD_CAP + 1, dtype=torch.long)
    rwse = torch.zeros(bsz, max_nodes, RW_STEPS, dtype=torch.float32)
    rrwp = torch.zeros(bsz, max_nodes, max_nodes, RW_STEPS + 1, dtype=torch.float32)
    source_side = torch.zeros(bsz, max_nodes, dtype=torch.float32)
    edge_batch = []
    edge_src = []
    edge_dst = []
    edge_capacity = []
    graph_num_nodes = []
    source = []
    sink = []
    target = []
    raw_flow = []
    raw_flow_norm = []
    source_out = []
    sink_in = []
    bound = []
    generator_id = []
    for graph_idx, graph in enumerate(graphs):
        n = graph.num_nodes
        node_type[graph_idx, :n] = graph.node_type.long()
        node_mask[graph_idx, :n] = True
        topo_adj[graph_idx, :n, :n] = graph.topo_adj.float()
        cap_mat[graph_idx, :n, :n] = graph.cap_mat.float()
        degree[graph_idx, :n] = graph.degree.float()
        cap_in[graph_idx, :n] = graph.cap_in.float()
        cap_out[graph_idx, :n] = graph.cap_out.float()
        spd[graph_idx, :n, :n] = graph.spd.long()
        rwse[graph_idx, :n] = graph.rwse.float()
        rrwp[graph_idx, :n, :n] = graph.rrwp.float()
        source_side[graph_idx, :n] = graph.source_side.float()
        e = graph.edge_index
        edge_batch.append(torch.full((e.size(1),), graph_idx, dtype=torch.long))
        edge_src.append(e[0])
        edge_dst.append(e[1])
        edge_capacity.append(graph.edge_capacity.float())
        graph_num_nodes.append(n)
        source.append(graph.source)
        sink.append(graph.sink)
        target.append(graph.target)
        raw_flow.append(graph.raw_flow)
        raw_flow_norm.append(graph.raw_flow_norm)
        source_out.append(graph.source_out_capacity)
        sink_in.append(graph.sink_in_capacity)
        bound.append(graph.bound_capacity)
        generator_id.append(GENERATOR_NAMES.index(graph.generator))
    pair_xi = torch.stack([pair_features(topo_adj[i], spd[i], rrwp[i], cap_mat[i]) for i in range(bsz)], dim=0)
    return FlowBatch(
        node_type=node_type,
        node_mask=node_mask,
        topo_adj=topo_adj,
        cap_mat=cap_mat,
        degree=degree,
        cap_in=cap_in,
        cap_out=cap_out,
        spd=spd,
        rwse=rwse,
        rrwp=rrwp,
        pair_xi=pair_xi,
        edge_batch=torch.cat(edge_batch),
        edge_src=torch.cat(edge_src),
        edge_dst=torch.cat(edge_dst),
        edge_capacity=torch.cat(edge_capacity),
        graph_num_nodes=torch.tensor(graph_num_nodes, dtype=torch.long),
        source=torch.tensor(source, dtype=torch.long),
        sink=torch.tensor(sink, dtype=torch.long),
        target=torch.tensor(target, dtype=torch.float32),
        raw_flow=torch.tensor(raw_flow, dtype=torch.float32),
        raw_flow_norm=torch.tensor(raw_flow_norm, dtype=torch.float32),
        source_out_capacity=torch.tensor(source_out, dtype=torch.float32),
        sink_in_capacity=torch.tensor(sink_in, dtype=torch.float32),
        bound_capacity=torch.tensor(bound, dtype=torch.float32),
        generator_id=torch.tensor(generator_id, dtype=torch.long),
        source_side=source_side,
        num_graphs=bsz,
        max_nodes=max_nodes,
    )


def eval_batch_size_for_split(split: str, requested: Optional[int] = None) -> int:
    if requested is not None and requested > 0:
        return requested
    size = SPLIT_SIZE.get(split, 0)
    if size <= 16:
        return 64
    if size <= 64:
        return 8
    if size <= 128:
        return 2
    return 1


def make_loader(dataset: Dataset[FlowGraph], batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collate_graphs,
        num_workers=0,
    )


def attention_support_mask(batch: FlowBatch, mode: str) -> torch.Tensor:
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


def gather_source_sink(h: torch.Tensor, batch: FlowBatch) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(h.size(0), device=h.device)
    return h[idx, batch.source], h[idx, batch.sink]


class GraphRegressionHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        in_dim = 6 * dim + 3
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, h: torch.Tensor, batch: FlowBatch) -> torch.Tensor:
        mask = batch.node_mask.unsqueeze(-1)
        h_masked = h * mask
        mean = h_masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        max_pool = h.masked_fill(~batch.node_mask.unsqueeze(-1), torch.finfo(h.dtype).min).max(dim=1).values
        source, sink = gather_source_sink(h, batch)
        pieces = [mean, max_pool, source, sink, source * sink, torch.abs(source - sink)]
        scalars = torch.stack([batch.source_out_capacity, batch.sink_in_capacity, batch.bound_capacity], dim=-1).to(h.dtype)
        pieces.append(scalars)
        x = torch.cat(pieces, dim=-1)
        return torch.sigmoid(self.net(x).squeeze(-1))


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

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, key_mask: torch.Tensor, attn_allow: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        return self.out_dropout(self.out_proj(out))


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

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor, key_mask: torch.Tensor, attn_allow: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_bias, key_mask, attn_allow)
        x = x + self.ffn(self.norm2(x))
        return x


class FlowGraphormer(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.cfg = cfg
        self.node_encoder = nn.Embedding(NODE_SYMBOLS + 1, dim, padding_idx=0)
        self.degree_encoder = nn.Embedding(1024, dim, padding_idx=0)
        self.cap_proj = nn.Linear(2, dim, bias=False)
        self.graph_token = nn.Embedding(1, dim)
        self.spatial_encoder = nn.Embedding(SPD_CAP + 2, cfg.heads, padding_idx=0)
        self.edge_encoder = nn.Embedding(2, cfg.heads, padding_idx=0)
        self.cap_bias = nn.Linear(2, cfg.heads, bias=False)
        self.virtual_distance = nn.Embedding(1, cfg.heads)
        self.emb_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(TransformerBlockWithBias(dim, dim * 2, cfg.heads, cfg) for _ in range(cfg.layers))
        self.graph_head = GraphRegressionHead(dim)

    def build_bias(self, batch: FlowBatch) -> torch.Tensor:
        bsz, n = batch.node_type.shape
        heads = self.cfg.heads
        bias = torch.zeros(bsz, heads, n + 1, n + 1, device=batch.node_type.device)
        spatial = self.spatial_encoder(batch.spd.long().clamp(max=SPD_CAP + 1)).permute(0, 3, 1, 2)
        edge = self.edge_encoder(batch.topo_adj.long().clamp(max=1)).permute(0, 3, 1, 2)
        cap_pair = torch.stack([batch.cap_mat, batch.cap_mat.transpose(1, 2)], dim=-1)
        cap = self.cap_bias(cap_pair).permute(0, 3, 1, 2)
        bias[:, :, 1:, 1:] = spatial + edge + cap
        token_bias = self.virtual_distance.weight.view(1, heads, 1)
        bias[:, :, 1:, 0] = bias[:, :, 1:, 0] + token_bias
        bias[:, :, 0, 1:] = bias[:, :, 0, 1:] + token_bias
        return bias

    def build_allow(self, batch: FlowBatch) -> Optional[torch.Tensor]:
        if self.cfg.attention_mode == "full":
            return None
        bsz, n = batch.node_type.shape
        allow = torch.zeros(bsz, n + 1, n + 1, dtype=torch.bool, device=batch.node_type.device)
        allow[:, 0, 0] = True
        allow[:, 1:, 1:] = attention_support_mask(batch, self.cfg.attention_mode)
        return allow

    def forward(self, batch: FlowBatch) -> torch.Tensor:
        bsz = batch.node_type.size(0)
        degree = batch.degree.long().clamp(max=1023)
        cap_node = torch.stack([batch.cap_in, batch.cap_out], dim=-1)
        h = self.node_encoder(batch.node_type) + self.degree_encoder(degree + 1) + self.cap_proj(cap_node)
        token = self.graph_token.weight.unsqueeze(0).expand(bsz, -1, -1)
        h = self.emb_norm(torch.cat([token, h], dim=1))
        key_mask = torch.cat([torch.ones(bsz, 1, dtype=torch.bool, device=h.device), batch.node_mask], dim=1)
        bias = self.build_bias(batch).to(h.dtype)
        allow = self.build_allow(batch)
        for layer in self.layers:
            h = layer(h, bias, key_mask, allow)
        return self.graph_head(h[:, 1:], batch)


class DenseGINE(nn.Module):
    def __init__(self, dim: int, cfg: PilotConfig) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(dim * 2, dim))
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, h: torch.Tensor, batch: FlowBatch, mask: torch.Tensor) -> torch.Tensor:
        bsz, n, _dim = h.shape
        eye = torch.eye(n, dtype=torch.float32, device=h.device).view(1, n, n).expand(bsz, -1, -1)
        route = batch.cap_mat.transpose(1, 2) > 0
        edge_input = torch.stack([route.float(), batch.cap_mat.transpose(1, 2), eye], dim=-1)
        edge_emb = self.edge_mlp(edge_input)
        source_h = h[:, None, :, :].expand(-1, n, -1, -1)
        messages = torch.relu(source_h + edge_emb)
        weights = route.float() + eye
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

    def forward(self, h: torch.Tensor, batch: FlowBatch, mask: torch.Tensor, attn_allow: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = h + self.local(self.local_norm(h), batch, mask)
        hn = self.global_norm(h)
        attn_mask = None
        if attn_allow is not None:
            bsz, n, _ = attn_allow.shape
            attn_mask = ~attn_allow[:, None, :, :].expand(bsz, self.heads, n, n)
            attn_mask = attn_mask.reshape(bsz * self.heads, n, n)
        out, _attn = self.global_attn(hn, hn, hn, attn_mask=attn_mask, key_padding_mask=~mask, need_weights=False)
        h = h + out * mask.unsqueeze(-1)
        h = h + self.ffn(self.ffn_norm(h)) * mask.unsqueeze(-1)
        return h * mask.unsqueeze(-1)


class FlowGraphGPS(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.cfg = cfg
        self.node_encoder = nn.Embedding(NODE_SYMBOLS + 1, dim, padding_idx=0)
        self.degree_proj = nn.Linear(1, dim, bias=False)
        self.cap_proj = nn.Linear(2, dim, bias=False)
        self.pe_bn = nn.BatchNorm1d(RW_STEPS)
        self.pe_encoder = nn.Sequential(nn.Linear(RW_STEPS, 16), nn.ReLU(), nn.Linear(16, dim))
        self.layers = nn.ModuleList(GPSLayer(dim, cfg) for _ in range(cfg.layers))
        self.graph_head = GraphRegressionHead(dim)

    def encode_pe(self, rwse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, n, steps = rwse.shape
        pe = self.pe_bn(rwse.reshape(-1, steps)).view(bsz, n, steps)
        return self.pe_encoder(pe) * mask.unsqueeze(-1)

    def forward(self, batch: FlowBatch) -> torch.Tensor:
        mask = batch.node_mask
        cap_node = torch.stack([batch.cap_in, batch.cap_out], dim=-1)
        h = (
            self.node_encoder(batch.node_type)
            + self.degree_proj(torch.log1p(batch.degree).unsqueeze(-1))
            + self.cap_proj(cap_node)
            + self.encode_pe(batch.rwse, mask)
        ) * mask.unsqueeze(-1)
        attn_allow = attention_support_mask(batch, self.cfg.attention_mode) if self.cfg.attention_mode != "full" else None
        for layer in self.layers:
            h = layer(h, batch, mask, attn_allow)
        return self.graph_head(h, batch)


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
            self.edge_mlp = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(dim * 2, dim))
        else:
            self.edge_norm = None
            self.edge_mlp = None

    def forward(
        self,
        h: torch.Tensor,
        edge_repr: torch.Tensor,
        mask: torch.Tensor,
        pair_allow: torch.Tensor,
        attn_allow: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            edge_repr = self.edge_norm(edge_repr + edge_delta)
        edge_repr = edge_repr * pair_allow.unsqueeze(-1)
        return h * mask.unsqueeze(-1), edge_repr


class FlowGrit(nn.Module):
    def __init__(self, cfg: PilotConfig, evolve_pairs: bool = True) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.cfg = cfg
        self.node_encoder = nn.Embedding(NODE_SYMBOLS + 1, dim, padding_idx=0)
        self.degree_proj = nn.Linear(1, dim, bias=False)
        self.cap_proj = nn.Linear(2, dim, bias=False)
        self.pair_encoder = nn.Linear(PAIR_RAW_DIM, dim)
        self.layers = nn.ModuleList(GritLayer(dim, cfg, evolve_pairs=evolve_pairs) for _ in range(cfg.layers))
        self.graph_head = GraphRegressionHead(dim)

    def forward(self, batch: FlowBatch) -> torch.Tensor:
        mask = batch.node_mask
        cap_node = torch.stack([batch.cap_in, batch.cap_out], dim=-1)
        h = (
            self.node_encoder(batch.node_type)
            + self.degree_proj(torch.log1p(batch.degree).unsqueeze(-1))
            + self.cap_proj(cap_node)
        ) * mask.unsqueeze(-1)
        pair_allow = attention_support_mask(batch, self.cfg.attention_mode)
        edge_repr = self.pair_encoder(batch.pair_xi) * pair_allow.unsqueeze(-1)
        attn_allow = pair_allow if self.cfg.attention_mode != "full" else None
        for layer in self.layers:
            h, edge_repr = layer(h, edge_repr, mask, pair_allow, attn_allow)
        return self.graph_head(h, batch)


class GNNPlusEncoder(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.node_encoder = nn.Embedding(NODE_SYMBOLS + 1, dim, padding_idx=0)
        self.degree_proj = nn.Linear(1, dim, bias=False)
        self.cap_proj = nn.Linear(2, dim, bias=False)
        self.pe_bn = nn.BatchNorm1d(RW_STEPS)
        self.pe_encoder = nn.Sequential(nn.Linear(RW_STEPS, 16), nn.ReLU(), nn.Linear(16, dim))

    def forward(self, batch: FlowBatch) -> torch.Tensor:
        bsz, n, steps = batch.rwse.shape
        pe = self.pe_bn(batch.rwse.reshape(-1, steps)).view(bsz, n, steps)
        cap_node = torch.stack([batch.cap_in, batch.cap_out], dim=-1)
        h = (
            self.node_encoder(batch.node_type)
            + self.degree_proj(torch.log1p(batch.degree).unsqueeze(-1))
            + self.cap_proj(cap_node)
            + self.pe_encoder(pe)
        )
        return h * batch.node_mask.unsqueeze(-1)


class GNNPlusFFN(nn.Module):
    def __init__(self, dim: int, cfg: PilotConfig) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(dim * 2, dim), nn.Dropout(cfg.dropout))

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

    def forward(self, h: torch.Tensor, batch: FlowBatch) -> torch.Tensor:
        bsz, n, dim = h.shape
        eye = torch.eye(n, dtype=torch.bool, device=h.device).view(1, n, n)
        incoming = batch.cap_mat.transpose(1, 2) > 0
        route = incoming | (eye & batch.pair_mask)
        edge_input = torch.stack([incoming.float(), batch.cap_mat.transpose(1, 2), eye.expand(bsz, -1, -1).float()], dim=-1)
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


class FlowGCNPlus(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.encoder = GNNPlusEncoder(cfg)
        self.layers = nn.ModuleList(GCNPlusLayer(dim, cfg) for _ in range(cfg.layers))
        self.graph_head = GraphRegressionHead(dim)

    def forward(self, batch: FlowBatch) -> torch.Tensor:
        h = self.encoder(batch)
        for layer in self.layers:
            h = layer(h, batch)
        return self.graph_head(h, batch)


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

    def forward(self, h: torch.Tensor, edge_repr: torch.Tensor, batch: FlowBatch) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n, dim = h.shape
        eye = torch.eye(n, dtype=torch.bool, device=h.device).view(1, n, n)
        route_mask = ((batch.cap_mat.transpose(1, 2) > 0) | eye) & batch.pair_mask
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


class FlowGatedGCNPlus(nn.Module):
    def __init__(self, cfg: PilotConfig) -> None:
        super().__init__()
        dim = cfg.hidden_dim
        self.encoder = GNNPlusEncoder(cfg)
        self.edge_encoder = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList(GatedGCNPlusLayer(dim, cfg) for _ in range(cfg.layers))
        self.graph_head = GraphRegressionHead(dim)

    def initial_edges(self, batch: FlowBatch) -> torch.Tensor:
        bsz, n = batch.node_type.shape
        eye = torch.eye(n, dtype=torch.bool, device=batch.node_type.device).view(1, n, n)
        incoming = batch.cap_mat.transpose(1, 2) > 0
        route_mask = (incoming | eye) & batch.pair_mask
        edge_input = torch.stack([incoming.float(), batch.cap_mat.transpose(1, 2), eye.expand(bsz, -1, -1).float()], dim=-1)
        return self.edge_encoder(edge_input) * route_mask.unsqueeze(-1)

    def forward(self, batch: FlowBatch) -> torch.Tensor:
        h = self.encoder(batch)
        edge_repr = self.initial_edges(batch)
        for layer in self.layers:
            h, edge_repr = layer(h, edge_repr, batch)
        return self.graph_head(h, batch)


def build_model(model_name: str, cfg: PilotConfig) -> nn.Module:
    family, _mode = parse_model_name(model_name)
    if family == "graphormer":
        return FlowGraphormer(cfg)
    if family == "graphgps":
        return FlowGraphGPS(cfg)
    if family == "grit":
        return FlowGrit(cfg, evolve_pairs=True)
    if family == "csa":
        return FlowGrit(cfg, evolve_pairs=False)
    if family == "gcn_plus":
        return FlowGCNPlus(cfg)
    if family == "gatedgcn_plus":
        return FlowGatedGCNPlus(cfg)
    raise ValueError(model_name)


def model_config_for(model_name: str, cfg: PilotConfig) -> PilotConfig:
    family, mode = parse_model_name(model_name)
    updates: dict[str, object] = {}
    if mode is not None:
        updates["attention_mode"] = mode
    if family in {"gcn_plus", "gatedgcn_plus"}:
        updates["hidden_dim"] = 64
    return replace(cfg, **updates)


def model_family(model_name: str) -> str:
    return parse_model_name(model_name)[0]


def model_attention_mode(model_name: str) -> str:
    _family, mode = parse_model_name(model_name)
    return mode or "message_passing"


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def ranks(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    out = torch.empty_like(order, dtype=torch.float32)
    out[order] = torch.arange(len(x), dtype=torch.float32)
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return float("nan")
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in pairs) / math.sqrt(vx * vy)


def regression_metrics(pred: torch.Tensor, target: torch.Tensor, raw_flow_norm: torch.Tensor, bound: torch.Tensor) -> dict[str, float]:
    pred = pred.float().clamp(0.0, 1.0)
    target = target.float()
    err = pred - target
    mse = float((err.square()).mean()) if err.numel() else float("nan")
    mae = float(err.abs().mean()) if err.numel() else float("nan")
    rmse = math.sqrt(max(0.0, mse)) if math.isfinite(mse) else float("nan")
    mean_y = target.mean()
    ss_res = float(err.square().sum())
    ss_tot = float(((target - mean_y) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1.0e-12 else float("nan")
    spearman = pearson(ranks(pred).tolist(), ranks(target).tolist()) if pred.numel() > 1 else float("nan")
    raw_pred = pred * bound.float()
    raw_err = (raw_pred - raw_flow_norm.float()).abs()
    rel = raw_err / raw_flow_norm.float().abs().clamp_min(1.0e-6)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
        "norm_rel_error": float(rel.mean()) if rel.numel() else float("nan"),
        "pred_mean": float(pred.mean()) if pred.numel() else float("nan"),
        "target_mean": float(target.mean()) if target.numel() else float("nan"),
        "raw_mae_norm_units": float(raw_err.mean()) if raw_err.numel() else float("nan"),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataset: Dataset[FlowGraph],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    seed: int,
    split: str,
) -> tuple[dict[str, float], list[dict[str, object]], float]:
    model.eval()
    loader = make_loader(dataset, batch_size, shuffle=False, seed=seed)
    preds = []
    targets = []
    raw_flow_norm = []
    bounds = []
    rows: list[dict[str, object]] = []
    start = time.time()
    graph_offset = 0
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            pred = model(batch).detach().cpu()
        target = batch.target.detach().cpu()
        preds.append(pred)
        targets.append(target)
        raw_flow_norm.append(batch.raw_flow_norm.detach().cpu())
        bounds.append(batch.bound_capacity.detach().cpu())
        for local_idx in range(batch.num_graphs):
            graph = dataset[graph_offset + local_idx]
            rows.append(
                {
                    "split": split,
                    "graph_index": graph_offset + local_idx,
                    "generator": graph.generator,
                    "num_nodes": graph.num_nodes,
                    "num_edges": int(graph.edge_index.size(1)),
                    "source": graph.source,
                    "sink": graph.sink,
                    "target": float(target[local_idx]),
                    "prediction": float(pred[local_idx].clamp(0.0, 1.0)),
                    "raw_flow": graph.raw_flow,
                    "raw_flow_norm": graph.raw_flow_norm,
                    "bound_capacity": graph.bound_capacity,
                    "abs_error": abs(float(pred[local_idx].clamp(0.0, 1.0)) - float(target[local_idx])),
                }
            )
        graph_offset += batch.num_graphs
    seconds = time.time() - start
    pred_all = torch.cat(preds)
    target_all = torch.cat(targets)
    raw_all = torch.cat(raw_flow_norm)
    bound_all = torch.cat(bounds)
    metrics = regression_metrics(pred_all, target_all, raw_all, bound_all)
    metrics["eval_seconds"] = seconds
    metrics["eval_seconds_per_graph"] = seconds / max(1, len(dataset))
    return metrics, rows, seconds


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
    install_package_if_missing("matplotlib", "matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return float("nan"), float("nan")
    mean = sum(finite) / len(finite)
    if len(finite) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)
    return mean, math.sqrt(max(0.0, var))


def eval_group_splits(suite: str) -> tuple[str, ...]:
    if suite == "size_samegen":
        return SAMEGEN_SPLITS
    if suite == "size_allgen":
        return ALLGEN_SPLITS
    raise ValueError(f"unknown evaluation suite: {suite}")


def split_display_label(split: str) -> str:
    if split == VAL_SPLIT:
        return "val16"
    return split


def metric_from_summary(summary: Mapping[str, object], split: str, metric: str) -> float:
    return float(summary[f"{split}_metrics"][metric])


def aggregate_rows(summaries: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    out = []
    by_model: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for summary in summaries:
        by_model[str(summary["model"])].append(summary)
    for model in MODEL_NAMES:
        runs = by_model.get(model, [])
        if not runs:
            continue
        row: dict[str, object] = {"model": model, "n_seeds": len(runs), "family": model_family(model), "support": model_attention_mode(model)}
        for split in EVAL_SPLITS:
            for metric in METRIC_NAMES:
                mean, std = mean_std([metric_from_summary(run, split, metric) for run in runs])
                row[f"{split}_{metric}_mean"] = mean
                row[f"{split}_{metric}_std"] = std
        out.append(row)
    return out


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
    train = [float(row["train_mse"]) for row in rows]
    val_mae = [float(row["val_mae"]) for row in rows]
    val_r2 = [float(row["val_r2"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    axes[0].plot(epochs, train, color="#377eb8")
    axes[0].set_title("train MSE")
    axes[1].plot(epochs, val_mae, color="#e41a1c")
    axes[1].set_title("val normalized MAE")
    axes[2].plot(epochs, val_r2, color="#4daf4a")
    axes[2].set_title("val R2")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_performance_vs_size(
    agg: Sequence[Mapping[str, object]],
    output_path: Path,
    metric: str = "mae",
    suite: str = "size_samegen",
    log=print,
) -> None:
    if not agg:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping performance-vs-size: {exc}")
        return
    splits = eval_group_splits(suite)
    sizes = [SPLIT_SIZE[split] for split in splits]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for row in agg:
        model = str(row["model"])
        means = [float(row[f"{split}_{metric}_mean"]) for split in splits]
        stds = [float(row[f"{split}_{metric}_std"]) for split in splits]
        ax.plot(sizes, means, marker="o", linewidth=1.4, label=model)
        if len(agg) <= 16:
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            ax.fill_between(sizes, lo, hi, alpha=0.08)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("graph size")
    ax.set_ylabel("normalized MAE" if metric == "mae" else metric)
    ax.set_title(f"Flow-Easy size generalization ({suite})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_locality_ablation(
    agg: Sequence[Mapping[str, object]],
    output_path: Path,
    metric: str = "mae",
    suite: str = "size_samegen",
    log=print,
) -> None:
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping locality ablation: {exc}")
        return
    rows = [row for row in agg if str(row["family"]) in GT_FAMILIES]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    x = torch.arange(len(GT_FAMILIES)).float()
    width = 0.24
    colors = {"full": "#80b1d3", "k1": "#8dd3c7", "k2": "#fb8072"}
    splits = eval_group_splits(suite)[-2:]
    for ax, split in zip(axes, splits):
        for idx, support in enumerate(ATTENTION_MODES):
            vals = []
            errs = []
            for family in GT_FAMILIES:
                match = [row for row in rows if str(row["family"]) == family and str(row["support"]) == support]
                vals.append(float(match[0][f"{split}_{metric}_mean"]) if match else float("nan"))
                errs.append(float(match[0][f"{split}_{metric}_std"]) if match else 0.0)
            ax.bar((x + (idx - 1) * width).numpy(), vals, yerr=errs, width=width, label=support, color=colors[support], capsize=2)
        ax.set_xticks(x.numpy())
        ax.set_xticklabels(GT_FAMILIES, rotation=20, ha="right")
        ax.set_title(split)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("normalized MAE")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Locality ablation ({suite})")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_family_rank(agg: Sequence[Mapping[str, object]], output_path: Path, suite: str = "size_samegen", log=print) -> None:
    if not agg:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping rank plot: {exc}")
        return
    rank_splits = eval_group_splits(suite)[-2:]
    best: dict[str, Mapping[str, object]] = {}
    for row in agg:
        fam = str(row["family"])
        score = sum(float(row[f"{split}_mae_mean"]) for split in rank_splits) / len(rank_splits)
        if fam not in best or score < sum(float(best[fam][f"{split}_mae_mean"]) for split in rank_splits) / len(rank_splits):
            best[fam] = row
    ordered = sorted(best.values(), key=lambda row: sum(float(row[f"{split}_mae_mean"]) for split in rank_splits) / len(rank_splits))
    labels = [str(row["model"]) for row in ordered]
    vals = [sum(float(row[f"{split}_mae_mean"]) for split in rank_splits) / len(rank_splits) for row in ordered]
    errs = [sum(float(row[f"{split}_mae_std"]) for split in rank_splits) / len(rank_splits) for row in ordered]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(range(len(labels)), vals, yerr=errs, color="#80b1d3", capsize=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("mean OOD normalized MAE")
    ax.set_title(f"Best OOD model per family ({suite})")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_compute_scaling(agg: Sequence[Mapping[str, object]], output_path: Path, suite: str = "size_samegen", log=print) -> None:
    if not agg:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping compute scaling: {exc}")
        return
    splits = eval_group_splits(suite)
    sizes = [SPLIT_SIZE[split] for split in splits]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for row in agg:
        model = str(row["model"])
        vals = [float(row[f"{split}_eval_seconds_per_graph_mean"]) for split in splits]
        ax.plot(sizes, vals, marker="o", linewidth=1.2, label=model)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_yscale("log")
    ax.set_xlabel("graph size")
    ax.set_ylabel("eval seconds per graph")
    ax.set_title(f"Evaluation scaling ({suite})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(agg: Sequence[Mapping[str, object]], output_path: Path, metric: str, log=print) -> None:
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping heatmap: {exc}")
        return
    if not agg:
        return
    models = [str(row["model"]) for row in agg]
    splits = list(EVAL_SPLITS)
    mat = torch.tensor([[float(row[f"{split}_{metric}_mean"]) for split in splits] for row in agg])
    fig, ax = plt.subplots(figsize=(6.2, max(4.0, 0.32 * len(models))))
    im = ax.imshow(mat.numpy(), aspect="auto", cmap="viridis_r" if metric in {"mae", "mse", "norm_rel_error"} else "viridis")
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(splits)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7)
    for i in range(mat.size(0)):
        for j in range(mat.size(1)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=6, color="white")
    ax.set_title(f"{metric} mean over seeds")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_seed_scatter(summaries: Sequence[Mapping[str, object]], output_path: Path, metric: str = "mae", log=print) -> None:
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping seed scatter: {exc}")
        return
    if not summaries:
        return
    models = [model for model in MODEL_NAMES if any(str(summary["model"]) == model for summary in summaries)]
    model_idx = {model: idx for idx, model in enumerate(models)}
    scatter_splits = list(OOD_EVAL_SPLITS)
    fig, axes = plt.subplots(2, 3, figsize=(max(10.0, 0.7 * len(models)), 6.6), sharey=True)
    for ax, split in zip(axes.flatten(), scatter_splits):
        for summary in summaries:
            model = str(summary["model"])
            seed = int(summary["seed"])
            x = model_idx[model] + ((seed % 11) - 5) * 0.012
            ax.scatter(x, metric_from_summary(summary, split, metric), s=18, alpha=0.8)
        ax.set_title(split)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel(metric)
    axes[1, 0].set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(rows: Sequence[Mapping[str, object]], output_path: Path, log=print) -> None:
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping calibration: {exc}")
        return
    data = [row for row in rows if str(row.get("split")) in OOD_EVAL_SPLITS and SPLIT_SIZE.get(str(row.get("split")), 0) >= 128]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    sample = data[:: max(1, len(data) // 4000)]
    ax.scatter([float(r["target"]) for r in sample], [float(r["prediction"]) for r in sample], s=6, alpha=0.18)
    ax.plot([0, 1], [0, 1], color="black", linewidth=1.0)
    ax.set_xlabel("target normalized flow")
    ax.set_ylabel("predicted normalized flow")
    ax.set_title("OOD calibration sample (samegen + allgen)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_generator(rows: Sequence[Mapping[str, object]], output_path: Path, log=print) -> None:
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping error by generator: {exc}")
        return
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        split = str(row.get("split"))
        if split not in OOD_EVAL_SPLITS or SPLIT_SIZE.get(split, 0) < 128:
            continue
        buckets[(split, str(row["generator"]))].append(float(row["abs_error"]))
    if not buckets:
        return
    gens = list(GENERATOR_NAMES)
    x = torch.arange(len(gens)).float()
    plot_splits = [split for split in OOD_EVAL_SPLITS if SPLIT_SIZE.get(split, 0) >= 128]
    width = 0.82 / max(1, len(plot_splits))
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    for idx, split in enumerate(plot_splits):
        vals = [mean_std(buckets.get((split, gen), []))[0] for gen in gens]
        ax.bar((x + (idx - (len(plot_splits) - 1) / 2) * width).numpy(), vals, width=width, label=split)
    ax.set_xticks(x.numpy())
    ax.set_xticklabels(gens, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("mean absolute error")
    ax.set_title("OOD error by generator family")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_flow_diagnostics(splits: Mapping[str, FlowDataset], output_path: Path, log=print) -> None:
    try:
        plt = import_plotting()
        import networkx as nx
    except Exception as exc:
        log(f"[plot] Skipping flow diagnostics: {exc}")
        return

    picks: list[tuple[str, FlowGraph]] = []
    for split_name, count in [("val16", 3), ("samegen64", 2), ("allgen64", 2)]:
        dataset = splits.get(split_name)
        if dataset is None or len(dataset) == 0:
            continue
        rng = random.Random(1701 + len(dataset) + count)
        idxs = rng.sample(range(len(dataset)), k=min(count, len(dataset)))
        picks.extend((split_name, dataset[idx]) for idx in idxs)
    if not picks:
        return

    cols = 3
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.9), squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")

    for ax, (split_name, graph) in zip(axes.flatten(), picks):
        n = graph.num_nodes
        directed = nx.DiGraph()
        directed.add_nodes_from(range(n))
        src_edges = graph.edge_index[0].tolist()
        dst_edges = graph.edge_index[1].tolist()
        caps = graph.edge_capacity.tolist()
        for src, dst, cap in zip(src_edges, dst_edges, caps):
            directed.add_edge(int(src), int(dst), capacity=float(cap))
        layout_graph = nx.Graph()
        layout_graph.add_nodes_from(range(n))
        layout_graph.add_edges_from({tuple(sorted((int(src), int(dst)))) for src, dst in zip(src_edges, dst_edges) if src != dst})
        pos = nx.spring_layout(layout_graph, seed=17, iterations=80)

        source_side = {idx for idx, val in enumerate(graph.source_side.tolist()) if val > 0.5}
        node_colors = []
        for idx in range(n):
            if idx == graph.source:
                node_colors.append("#d73027")
            elif idx == graph.sink:
                node_colors.append("#4575b4")
            elif idx in source_side:
                node_colors.append("#fee090")
            else:
                node_colors.append("#e0f3f8")

        cut_edges = {
            (int(graph.cut_edge_index[0, idx]), int(graph.cut_edge_index[1, idx]))
            for idx in range(graph.cut_edge_index.size(1))
        }
        non_cut_edges = [edge for edge in directed.edges() if edge not in cut_edges]
        nx.draw_networkx_edges(directed, pos, edgelist=non_cut_edges, ax=ax, edge_color="#9e9e9e", width=0.7, alpha=0.45, arrows=False)
        if cut_edges:
            nx.draw_networkx_edges(
                directed,
                pos,
                edgelist=list(cut_edges),
                ax=ax,
                edge_color="#f46d43",
                width=1.8,
                alpha=0.9,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=8,
            )
        nx.draw_networkx_nodes(directed, pos, ax=ax, node_color=node_colors, node_size=95 if n <= 24 else 34, linewidths=0.4, edgecolors="#333333")
        if n <= 24:
            nx.draw_networkx_labels(directed, pos, ax=ax, font_size=6, font_color="#111111")
        ax.set_title(
            f"{split_name} n={n} s={graph.source} t={graph.sink}\n"
            f"target={graph.target:.3f} raw_flow={graph.raw_flow:.2f} cut_edges={len(cut_edges)}",
            fontsize=8,
        )
        ax.axis("off")

    fig.suptitle("Flow metadata diagnostics: source, sink, min-cut side, and cut edges", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def support_density(dataset: Dataset[FlowGraph], mode: str, max_graphs: int = 32) -> float:
    loader = make_loader(Subset(dataset, list(range(min(len(dataset), max_graphs)))), batch_size=min(8, max_graphs), shuffle=False, seed=0)
    num = 0.0
    den = 0.0
    for batch in loader:
        mask = attention_support_mask(batch, mode if mode in ATTENTION_MODES else "k1")
        num += float(mask.sum())
        den += float(batch.pair_mask.sum())
    return num / max(1.0, den)


def model_design_rows(model_names: Sequence[str], cfg: PilotConfig, splits: Mapping[str, FlowDataset]) -> list[dict[str, object]]:
    rows = []
    for model_name in model_names:
        model_cfg = model_config_for(model_name, cfg)
        family = model_family(model_name)
        support = model_attention_mode(model_name)
        rows.append(
            {
                "model": model_name,
                "family": family,
                "layers": model_cfg.layers,
                "hidden_dim": model_cfg.hidden_dim,
                "heads": model_cfg.heads,
                "attention_support": support,
                "pe_features": PE_NOTES[family],
                "readout_features": "mean/max/source/sink/source*sink/abs(source-sink)/capacity-bound scalars; no pair-state readout",
                "pair_state_update": "evolved and support-masked" if family == "grit" else ("static and support-masked" if family == "csa" else "not_applicable"),
                "parameters": count_parameters(build_model(model_name, model_cfg)),
                "support_density_train16": support_density(splits["train16"], support),
                "support_density_samegen64": support_density(splits["samegen64"], support),
                "support_density_samegen128": support_density(splits["samegen128"], support),
                "support_density_samegen256": support_density(splits["samegen256"], support),
                "support_density_allgen64": support_density(splits["allgen64"], support),
                "support_density_allgen128": support_density(splits["allgen128"], support),
                "support_density_allgen256": support_density(splits["allgen256"], support),
            }
        )
    return rows


def write_and_log_model_design(model_names: Sequence[str], cfg: PilotConfig, splits: Mapping[str, FlowDataset], path: Path, log=print) -> None:
    rows = model_design_rows(model_names, cfg, splits)
    write_csv_rows(path, rows)
    log("[design] model | PE | support | params | support density train/same64/same128/same256/all64/all128/all256")
    for row in rows:
        log(
            f"[design] {row['model']} | {row['pe_features']} | {row['attention_support']} | "
            f"{int(row['parameters']):,} | {float(row['support_density_train16']):.3f}/"
            f"{float(row['support_density_samegen64']):.3f}/{float(row['support_density_samegen128']):.3f}/"
            f"{float(row['support_density_samegen256']):.3f}/{float(row['support_density_allgen64']):.3f}/"
            f"{float(row['support_density_allgen128']):.3f}/{float(row['support_density_allgen256']):.3f}"
        )


def make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1.0e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_signature(model_name: str, model_cfg: PilotConfig, splits: Mapping[str, FlowDataset]) -> dict[str, object]:
    return {
        "task": TASK_NAME,
        "implementation_version": RUN_IMPLEMENTATION_VERSION,
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
    splits: Mapping[str, FlowDataset],
    cfg: PilotConfig,
    output_root: Path,
    device: torch.device,
    force_retrain: bool,
    log=print,
) -> dict[str, object]:
    set_seed(cfg.seed)
    run_dir = output_root / cfg.preset / f"depth{cfg.layers}" / model_name / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_log = RunLogger(run_dir / "run.log")
    model_cfg = model_config_for(model_name, cfg)
    family = model_family(model_name)
    support = model_attention_mode(model_name)
    signature = run_signature(model_name, model_cfg, splits)
    run_log(f"[run] model={model_name} seed={cfg.seed} device={device}")
    run_log(f"[config] {MODEL_CONFIG_NOTES[family]}")
    if support in ATTENTION_MODES:
        run_log(f"[attention] {support}: {ATTENTION_MODE_NOTES[support]}")
    write_json(run_dir / "config.json", asdict(model_cfg) | {"model": model_name, "family": family, "attention_mode": support})
    write_json(run_dir / "dataset_stats.json", {name: dataset_stats(dataset) for name, dataset in splits.items()})
    train_loader = make_loader(splits["train16"], cfg.batch_size, shuffle=True, seed=cfg.seed)
    model = build_model(model_name, model_cfg).to(device)
    n_params = count_parameters(model)
    run_log(f"[model] trainable_parameters={n_params:,}")
    loss_fn = nn.MSELoss()
    use_amp = cfg.amp and device.type == "cuda"
    best_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    per_graph_rows: list[dict[str, object]] = []

    if best_path.exists() and signature_matches(summary_path, signature) and not force_retrain:
        run_log("[resume] found matching best.pt and summary.json; skipping training")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        best_epoch = int(ckpt.get("epoch", 0))
        best_val_mae = float(ckpt.get("val_metrics", {}).get("mae", float("nan")))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        total_steps = cfg.max_epochs * max(1, len(train_loader))
        warmup_steps = cfg.warmup_epochs * max(1, len(train_loader))
        scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
        best_val_mae = float("inf")
        best_epoch = 0
        bad_epochs = 0
        fields = ["epoch", "lr", "train_mse", "val_mae", "val_mse", "val_r2", "val_spearman", "seconds"]
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for epoch in range(1, cfg.max_epochs + 1):
                started = time.time()
                model.train()
                total_loss = 0.0
                total_graphs = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", enabled=use_amp):
                        pred = model(batch)
                        loss = loss_fn(pred, batch.target)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    total_loss += float(loss.detach().cpu()) * batch.num_graphs
                    total_graphs += batch.num_graphs
                val_metrics, _rows, _seconds = evaluate_model(
                    model,
                    splits["val16"],
                    eval_batch_size_for_split("val16"),
                    device,
                    use_amp,
                    cfg.seed,
                    "val16",
                )
                row = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_mse": total_loss / max(1, total_graphs),
                    "val_mae": val_metrics["mae"],
                    "val_mse": val_metrics["mse"],
                    "val_r2": val_metrics["r2"],
                    "val_spearman": val_metrics["spearman"],
                    "seconds": time.time() - started,
                }
                writer.writerow(row)
                handle.flush()
                run_log(
                    f"[epoch {epoch:03d}] train_mse={row['train_mse']:.5f} "
                    f"val_mae={row['val_mae']:.5f} val_r2={row['val_r2']:.4f} "
                    f"time={row['seconds']:.1f}s"
                )
                improved = val_metrics["mae"] < best_val_mae
                if improved:
                    best_val_mae = val_metrics["mae"]
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
                if val_metrics["mae"] < best_val_mae - cfg.min_delta:
                    bad_epochs = 0
                else:
                    bad_epochs = 0 if improved else bad_epochs + 1
                    if bad_epochs >= cfg.patience:
                        run_log(f"[early-stop] best_epoch={best_epoch}")
                        break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    split_metrics = {}
    split_rows = {}
    for split_name in EVAL_SPLITS:
        metrics, rows, seconds = evaluate_model(
            model,
            splits[split_name],
            eval_batch_size_for_split(split_name),
            device,
            use_amp,
            cfg.seed,
            split_name,
        )
        split_metrics[split_name] = metrics
        split_rows[split_name] = rows
        for row in rows:
            per_graph_rows.append({"model": model_name, "seed": cfg.seed, **row})
    summary = {
        "model": model_name,
        "seed": cfg.seed,
        "trainable_parameters": n_params,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_checkpoint": str(best_path),
        "run_signature": signature,
    }
    for split_name in EVAL_SPLITS:
        summary[f"{split_name}_metrics"] = split_metrics[split_name]
    write_json(summary_path, summary)
    write_csv_rows(run_dir / "graph_predictions.csv", per_graph_rows)
    plot_training_curves(metrics_path, run_dir / "training_curves.png", log=run_log)
    plot_calibration(per_graph_rows, run_dir / "calibration_ood.png", log=run_log)
    plot_error_by_generator(per_graph_rows, run_dir / "error_by_generator_ood.png", log=run_log)
    run_log(
        f"[done] val_mae={split_metrics['val16']['mae']:.5f} "
        f"samegen128_mae={split_metrics['samegen128']['mae']:.5f} "
        f"allgen128_mae={split_metrics['allgen128']['mae']:.5f} "
        f"allgen256_mae={split_metrics['allgen256']['mae']:.5f}"
    )
    return summary | {"per_graph_rows": per_graph_rows}


def parse_seed_list(text: str, fallback_seed: int) -> list[int]:
    parts = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if not parts:
        return [fallback_seed]
    out = []
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a)
            end = int(b)
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(part))
    seen = set()
    deduped = []
    for seed in out:
        if seed not in seen:
            deduped.append(seed)
            seen.add(seed)
    return deduped


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
    parser = argparse.ArgumentParser(description="Standalone Flow-Easy paper training runner.")
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all", "depth_sweep"], default="all")
    parser.add_argument("--depth", type=int, choices=[2, 4], default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default="0,1,2,3")
    parser.add_argument("--train-graphs", type=int, default=50000)
    parser.add_argument("--val-graphs", type=int, default=8000)
    parser.add_argument("--test64-graphs", type=int, default=3000)
    parser.add_argument("--test128-graphs", type=int, default=2000)
    parser.add_argument("--test256-graphs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1.0e-5)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/graph_specialisation_metrics/flow_easy_paper"))
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
    return parser.parse_args(argv)


def cfg_from_args(args: argparse.Namespace, seed: int) -> PilotConfig:
    cfg = PilotConfig(
        seed=seed,
        train_graphs=args.train_graphs,
        val_graphs=args.val_graphs,
        test64_graphs=args.test64_graphs,
        test128_graphs=args.test128_graphs,
        test256_graphs=args.test256_graphs,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        layers=args.depth,
        amp=not args.no_amp,
    )
    if args.fast_dev_run:
        cfg = PilotConfig(
            **{
                **asdict(cfg),
                "train_graphs": min(cfg.train_graphs, 128),
                "val_graphs": min(cfg.val_graphs, 64),
                "test64_graphs": min(cfg.test64_graphs, 32),
                "test128_graphs": min(cfg.test128_graphs, 16),
                "test256_graphs": min(cfg.test256_graphs, 8),
                "batch_size": min(cfg.batch_size, 32),
                "max_epochs": min(cfg.max_epochs, 2),
                "patience": 2,
            }
        )
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = strip_colab_kernel_args(list(sys.argv[1:] if argv is None else argv))
    args = parse_args(argv)
    install_package_if_missing("networkx", "networkx")
    seeds = parse_seed_list(args.seeds, args.seed)
    if args.fast_dev_run:
        seeds = seeds[:1]
    if args.model == "depth_sweep":
        models = list(DEPTH_SWEEP_MODELS)
        if args.seeds == "0,1,2,3":
            seeds = [0, 1, 2]
    else:
        models = list(MODEL_NAMES) if args.model == "all" else [args.model]
    base_cfg = cfg_from_args(args, seeds[0])
    set_seed(base_cfg.seed)
    device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    mount_drive(args.drive_mount, enabled=not args.no_mount_drive)
    cache_root = args.cache_root or args.drive_dir / "cache"
    output_root = args.output_root or args.drive_dir / "results"
    suite_dir = output_root / base_cfg.preset / f"depth{base_cfg.layers}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    top_log = RunLogger(suite_dir / "run.log")
    top_log(f"[setup] device={device} amp={base_cfg.amp and device.type == 'cuda'}")
    top_log(f"[setup] seeds={seeds} depth={base_cfg.layers}")
    top_log(f"[setup] cache_root={cache_root}")
    top_log(f"[setup] output_root={output_root}")
    splits = {
        "train16": load_or_generate_split(cache_root, "train", 16, base_cfg.train_graphs, 0, args.force_regenerate, top_log),
        "val16": load_or_generate_split(cache_root, "val", 16, base_cfg.val_graphs, 11, args.force_regenerate, top_log),
        "samegen64": load_or_generate_split(cache_root, "samegen64", 64, base_cfg.test64_graphs, 67, args.force_regenerate, top_log),
        "samegen128": load_or_generate_split(cache_root, "samegen128", 128, base_cfg.test128_graphs, 79, args.force_regenerate, top_log),
        "samegen256": load_or_generate_split(cache_root, "samegen256", 256, base_cfg.test256_graphs, 97, args.force_regenerate, top_log),
        "allgen64": load_or_generate_split(cache_root, "allgen64", 64, base_cfg.test64_graphs, 23, args.force_regenerate, top_log),
        "allgen128": load_or_generate_split(cache_root, "allgen128", 128, base_cfg.test128_graphs, 37, args.force_regenerate, top_log),
        "allgen256": load_or_generate_split(cache_root, "allgen256", 256, base_cfg.test256_graphs, 53, args.force_regenerate, top_log),
    }
    write_json(suite_dir / "resolved_config.json", asdict(base_cfg) | {"seeds": seeds, "models": models})
    write_json(suite_dir / "dataset_stats.json", {name: dataset_stats(dataset) for name, dataset in splits.items()})
    write_and_log_model_design(models, base_cfg, splits, suite_dir / "model_design_table.csv", log=top_log)
    summaries = []
    all_graph_rows: list[dict[str, object]] = []
    for model_name in models:
        for seed in seeds:
            cfg = replace(base_cfg, seed=seed)
            result = train_one_model(model_name, splits, cfg, output_root, device, force_retrain=args.force_retrain, log=top_log)
            all_graph_rows.extend(result.pop("per_graph_rows", []))
            summaries.append(result)
    agg = aggregate_rows(summaries)
    flat_metrics = []
    for summary in summaries:
        for split in EVAL_SPLITS:
            flat_metrics.append({"model": summary["model"], "seed": summary["seed"], "split": split, **summary[f"{split}_metrics"]})
    write_json(suite_dir / "summary.json", {"runs": summaries, "aggregate": agg})
    write_csv_rows(suite_dir / "performance_metrics.csv", flat_metrics)
    write_csv_rows(suite_dir / "aggregate_metrics.csv", agg)
    write_csv_rows(suite_dir / "graph_predictions.csv", all_graph_rows)
    plot_performance_vs_size(agg, suite_dir / "main_performance_vs_size_samegen_mae.png", metric="mae", suite="size_samegen", log=top_log)
    plot_performance_vs_size(agg, suite_dir / "main_performance_vs_size_allgen_mae.png", metric="mae", suite="size_allgen", log=top_log)
    plot_performance_vs_size(agg, suite_dir / "appendix_performance_vs_size_samegen_r2.png", metric="r2", suite="size_samegen", log=top_log)
    plot_performance_vs_size(agg, suite_dir / "appendix_performance_vs_size_allgen_r2.png", metric="r2", suite="size_allgen", log=top_log)
    plot_locality_ablation(agg, suite_dir / "main_locality_ablation_samegen_mae.png", metric="mae", suite="size_samegen", log=top_log)
    plot_locality_ablation(agg, suite_dir / "main_locality_ablation_allgen_mae.png", metric="mae", suite="size_allgen", log=top_log)
    plot_family_rank(agg, suite_dir / "main_gt_vs_gnn_rank_samegen.png", suite="size_samegen", log=top_log)
    plot_family_rank(agg, suite_dir / "main_gt_vs_gnn_rank_allgen.png", suite="size_allgen", log=top_log)
    plot_compute_scaling(agg, suite_dir / "main_compute_scaling_samegen.png", suite="size_samegen", log=top_log)
    plot_compute_scaling(agg, suite_dir / "main_compute_scaling_allgen.png", suite="size_allgen", log=top_log)
    plot_heatmap(agg, suite_dir / "appendix_mae_heatmap.png", metric="mae", log=top_log)
    plot_heatmap(agg, suite_dir / "appendix_r2_heatmap.png", metric="r2", log=top_log)
    plot_seed_scatter(summaries, suite_dir / "appendix_seed_scatter_mae.png", metric="mae", log=top_log)
    plot_calibration(all_graph_rows, suite_dir / "appendix_calibration_ood.png", log=top_log)
    plot_error_by_generator(all_graph_rows, suite_dir / "appendix_error_by_generator_ood.png", log=top_log)
    plot_flow_diagnostics(splits, suite_dir / "appendix_flow_metadata_diagnostics.png", log=top_log)
    top_log("[done] Flow-Easy paper run complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
