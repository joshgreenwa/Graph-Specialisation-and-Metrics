#!/usr/bin/env python3
"""Standalone wallclock + memory benchmark for graph transformer variants.

This file intentionally loads no project-local or downloaded Python files. It is
meant to be copied to, uploaded to, or run directly in Colab.

Task
----
Synthetic graph-level regression on sparse connected non-molecular graphs. Each
graph has two marked categorical nodes. The target is their shortest-path
distance normalized by graph size. The task is only there to exercise forward /
backward paths on non-ZINC graphs; the output is wallclock and memory, not a
scientific accuracy claim.

Methods
-------
* grit_dense_rrwp_proxy: dense attention + RRWP/SPD pair bias, included as a
  standalone dense GRIT-like timing proxy because official GRIT is a GraphGym
  repo/config runner rather than a portable single-file model.
* csa_static_anchor: fixed symmetric structural anchor with additive bias and
  value addition.
* graphgrape_v21_spd2 / spd3: sparse latent pair memory with support clipped to
  shortest-path distance <= 2 or <= 3.
* graphgrape_v21_triangle_spd2 / spd3: same, with sparse triangle composition.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import resource
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Colab/notebook defaults
# =============================================================================

CELL_ARGS: list[str] | str | None = None

METHOD_CHOICES = (
    "grit_dense_rrwp_proxy",
    "csa_static_anchor",
    "graphgrape_v21_spd2",
    "graphgrape_v21_spd3",
    "graphgrape_v21_triangle_spd2",
    "graphgrape_v21_triangle_spd3",
)


# =============================================================================
# Model/task constants
# =============================================================================

NUM_ATOM_TYPES = 28
NUM_BOND_TYPES = 3

D_MODEL = 64
N_HEADS = 8
N_LAYERS = 10
D_HEAD = D_MODEL // N_HEADS

K_WALK = 12
D_MAX_SPD = 5

STATIC_P_PAIR = (D_MAX_SPD + 1) + K_WALK + NUM_BOND_TYPES

RING_MIN_SIZE = 3
RING_MAX_SIZE = 8
P_RING_SIZES = RING_MAX_SIZE - RING_MIN_SIZE + 1
GG_P_RING_PAIR = 2 * P_RING_SIZES
GG_P_SYM = STATIC_P_PAIR
GG_P_PAIR = GG_P_RING_PAIR + GG_P_SYM
P_LATENT_PAIR = 32

PAIR_UPDATE_RANK = 16
PAIR_UPDATE_SCALE_INIT = 5e-2
TRIANGLE_UPDATE_SCALE_INIT = 5e-2
PAIR_OUT_INIT_STD = 1e-3
TRIANGLE_OUT_INIT_STD = 1e-3
PAIR_WRITE_GATE_INIT = 0.7
DYNAMIC_GATE_INIT = 5e-2
RMS_NORM_EPS = 1e-6

ATTN_DROPOUT = 0.2
RESID_DROPOUT = 0.0
FFN_MULT = 2


@dataclass(frozen=True)
class SizeConfig:
    size_label: str
    min_nodes: int
    max_nodes: int
    seed: int


@dataclass
class MethodResult:
    method: str
    size_label: str
    repeat: int
    device: str
    num_graphs: int
    batch_size: int
    epochs: int
    warmup_epochs: int
    min_nodes: int
    max_nodes: int
    avg_extra_degree: float
    param_count: int
    preprocess_s: float
    train_measured_s: float
    mean_epoch_s: float
    median_epoch_s: float
    std_epoch_s: float
    graphs_per_s: float
    final_loss: float
    avg_nodes: float
    avg_directed_edges: float
    avg_sparse_pairs: float
    avg_triangle_triples: float
    rss_start_mb: float
    rss_peak_mb: float
    rss_end_mb: float
    cuda_peak_allocated_mb: float
    cuda_peak_reserved_mb: float


@dataclass
class StaticBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    rwse: torch.Tensor
    pair_xi: torch.Tensor
    degree: torch.Tensor
    batch_index: torch.Tensor
    node_pos: torch.Tensor
    node_mask: torch.Tensor
    pair_mask: torch.Tensor
    num_graphs: int
    max_nodes: int

    def to(self, device: torch.device) -> "StaticBatch":
        fields = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in self.__dict__.items()
        }
        return StaticBatch(**fields)


@dataclass
class GraphGrapeBatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    rwse: torch.Tensor
    pair_xi: torch.Tensor
    sparse_pair_batch: torch.Tensor
    sparse_pair_i: torch.Tensor
    sparse_pair_j: torch.Tensor
    sparse_pair_src: torch.Tensor
    sparse_pair_dst: torch.Tensor
    triangle_out: torch.Tensor
    triangle_left: torch.Tensor
    triangle_right: torch.Tensor
    degree: torch.Tensor
    degree_log: torch.Tensor
    batch_index: torch.Tensor
    node_pos: torch.Tensor
    node_mask: torch.Tensor
    pair_mask: torch.Tensor
    num_graphs: int
    max_nodes: int

    def to(self, device: torch.device) -> "GraphGrapeBatch":
        fields = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in self.__dict__.items()
        }
        return GraphGrapeBatch(**fields)


class TensorGraphDataset:
    def __init__(self, graphs: list[dict[str, torch.Tensor]]) -> None:
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.graphs[idx]


# =============================================================================
# Utilities
# =============================================================================

def log(msg: str = "") -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def current_rss_mb() -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.Process().memory_info().rss) / (1024.0 ** 2)
    except Exception:
        usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return usage / (1024.0 ** 2)
        return usage / 1024.0


class MemorySampler:
    """Lightweight process-RSS sampler plus CUDA peak-memory wrapper."""

    def __init__(self, device: torch.device, interval_s: float = 0.05) -> None:
        self.device = device
        self.interval_s = interval_s
        self.start_rss_mb = 0.0
        self.peak_rss_mb = 0.0
        self.end_rss_mb = 0.0
        self.cuda_peak_allocated_mb = 0.0
        self.cuda_peak_reserved_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "MemorySampler":
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)
        self.start_rss_mb = current_rss_mb()
        self.peak_rss_mb = self.start_rss_mb
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.end_rss_mb = current_rss_mb()
        self.peak_rss_mb = max(self.peak_rss_mb, self.end_rss_mb)
        if self.device.type == "cuda":
            sync_device(self.device)
            self.cuda_peak_allocated_mb = (
                torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
            )
            self.cuda_peak_reserved_mb = (
                torch.cuda.max_memory_reserved(self.device) / (1024.0 ** 2)
            )
        return False

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_mb = max(self.peak_rss_mb, current_rss_mb())
            self._stop.wait(self.interval_s)


def mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def std(values: list[float]) -> float:
    if not values:
        return float("nan")
    m = mean(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def logit_from_prob(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return torch.logit(torch.tensor(p)).item()


def softplus_inverse(x: float) -> float:
    return torch.log(torch.expm1(torch.tensor(float(x)))).item()


# =============================================================================
# Synthetic graph generation
# =============================================================================

def extra_edge_probability(n: int, avg_extra_degree: float) -> float:
    return max(0.0, min(1.0, avg_extra_degree / max(1, n - 1)))


def make_connected_edges(n: int, rng: random.Random, avg_extra_degree: float) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    order = list(range(n))
    rng.shuffle(order)
    for idx in range(1, n):
        child = order[idx]
        parent = order[rng.randrange(idx)]
        edges.add(tuple(sorted((child, parent))))

    p_extra = extra_edge_probability(n, avg_extra_degree)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edges and rng.random() < p_extra:
                edges.add((i, j))
    return sorted(edges)


def shortest_path_distance(n: int, edges: list[tuple[int, int]], source: int, target: int) -> int:
    neighbours = [[] for _ in range(n)]
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)
    seen = [False] * n
    seen[source] = True
    queue: deque[tuple[int, int]] = deque([(source, 0)])
    while queue:
        node, dist = queue.popleft()
        if node == target:
            return dist
        for nxt in neighbours[node]:
            if not seen[nxt]:
                seen[nxt] = True
                queue.append((nxt, dist + 1))
    return n


def synthetic_graph(n: int, rng: random.Random, avg_extra_degree: float) -> SimpleNamespace:
    edges = make_connected_edges(n, rng, avg_extra_degree)
    directed_edges: list[tuple[int, int]] = []
    directed_attrs: list[int] = []
    for a, b in edges:
        attr = rng.randrange(NUM_BOND_TYPES)
        directed_edges.append((a, b))
        directed_edges.append((b, a))
        directed_attrs.append(attr)
        directed_attrs.append(attr)

    anchor = rng.randrange(n)
    target = rng.randrange(n - 1)
    if target >= anchor:
        target += 1
    dist = shortest_path_distance(n, edges, anchor, target)

    x = [rng.randrange(3, NUM_ATOM_TYPES) for _ in range(n)]
    x[anchor] = 1
    x[target] = 2

    return SimpleNamespace(
        x=torch.tensor(x, dtype=torch.long),
        edge_index=torch.tensor(directed_edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(directed_attrs, dtype=torch.long),
        y=torch.tensor([dist / max(1, n - 1)], dtype=torch.float32),
        num_nodes=n,
    )


def generate_graphs(
    *,
    num_graphs: int,
    min_nodes: int,
    max_nodes: int,
    avg_extra_degree: float,
    seed: int,
) -> list[SimpleNamespace]:
    rng = random.Random(seed)
    graphs = []
    for _ in range(num_graphs):
        n = rng.randint(min_nodes, max_nodes)
        graphs.append(synthetic_graph(n, rng, avg_extra_degree))
    return graphs


def graph_stats(raw_graphs: list[SimpleNamespace]) -> tuple[float, float]:
    avg_nodes = mean([float(g.num_nodes) for g in raw_graphs])
    avg_edges = mean([float(g.edge_index.size(1)) for g in raw_graphs])
    return avg_nodes, avg_edges


# =============================================================================
# Feature construction
# =============================================================================

def bfs_spd_classes(adj_bool: torch.Tensor, dmax: int) -> torch.Tensor:
    n = int(adj_bool.size(0))
    dist = torch.full((n, n), dmax, dtype=torch.long)
    neighbours = [torch.nonzero(adj_bool[i], as_tuple=False).view(-1).tolist() for i in range(n)]
    for source in range(n):
        dist[source, source] = 0
        queue: deque[int] = deque([source])
        while queue:
            node = queue.popleft()
            node_dist = int(dist[source, node].item())
            if node_dist >= dmax:
                continue
            for nxt in neighbours[node]:
                if dist[source, nxt] == dmax and nxt != source:
                    dist[source, nxt] = min(node_dist + 1, dmax)
                    queue.append(int(nxt))
    return dist


def base_structural_features(data: SimpleNamespace) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    x = data.x.view(-1).long()
    edge_index = data.edge_index.long()
    edge_attr = data.edge_attr.view(-1).long()

    n = int(data.num_nodes)
    adj = torch.zeros((n, n), dtype=torch.float32)
    if edge_index.numel() > 0:
        src, dst = edge_index
        adj[src, dst] = 1.0
    adj = torch.maximum(adj, adj.t())
    adj.fill_diagonal_(0.0)

    degree = adj.sum(dim=1)
    walk = torch.zeros_like(adj)
    nz = degree > 0
    walk[nz] = adj[nz] / degree[nz].unsqueeze(1)

    powers = []
    power = walk.clone()
    for _ in range(K_WALK):
        powers.append(power.clone())
        power = power @ walk

    rwse = torch.stack([p.diagonal() for p in powers], dim=-1).float()
    walk_sum = torch.stack([p + p.t() for p in powers], dim=-1).float()
    spd_cls = bfs_spd_classes(adj > 0, D_MAX_SPD)
    spd_oh = F.one_hot(spd_cls.clamp(max=D_MAX_SPD), num_classes=D_MAX_SPD + 1).float()

    bond_oh = torch.zeros((n, n, NUM_BOND_TYPES), dtype=torch.float32)
    if edge_index.numel() > 0:
        src, dst = edge_index
        bond_oh[src, dst, edge_attr] = 1.0
        bond_oh[dst, src, edge_attr] = 1.0

    sym = torch.cat([spd_oh, walk_sum, bond_oh], dim=-1).float()
    return x, edge_index, edge_attr, rwse, sym, degree.float()


def compute_static_features(data: SimpleNamespace) -> dict[str, torch.Tensor]:
    x, edge_index, edge_attr, rwse, pair_xi, degree = base_structural_features(data)
    return {
        "x": x.cpu(),
        "edge_index": edge_index.cpu(),
        "edge_attr": edge_attr.cpu(),
        "y": data.y.view(1).float().cpu(),
        "rwse": rwse.cpu(),
        "pair_xi": pair_xi.cpu(),
        "degree": degree.cpu(),
    }


def compute_ring_pair_features(
    adj_bool: torch.Tensor,
    min_size: int = RING_MIN_SIZE,
    max_size: int = RING_MAX_SIZE,
) -> torch.Tensor:
    n = int(adj_bool.size(0))
    num_sizes = max_size - min_size + 1
    same_ring = torch.zeros((n, n, num_sizes), dtype=torch.float32)
    edge_in_ring = torch.zeros((n, n, num_sizes), dtype=torch.float32)
    neighbours = [torch.nonzero(adj_bool[i], as_tuple=False).view(-1).tolist() for i in range(n)]
    seen_cycles: set[tuple[tuple[int, int], ...]] = set()

    def is_chordless(path: list[int]) -> bool:
        m = len(path)
        if len(set(path)) != m:
            return False
        for a in range(m):
            u = path[a]
            for b in range(a + 1, m):
                if b == a + 1 or (a == 0 and b == m - 1):
                    continue
                if bool(adj_bool[u, path[b]]):
                    return False
        return True

    def mark_cycle(path: list[int]) -> None:
        m = len(path)
        if m < min_size or m > max_size or not is_chordless(path):
            return
        key_edges = []
        for idx in range(m):
            u, v = path[idx], path[(idx + 1) % m]
            key_edges.append((u, v) if u < v else (v, u))
        key = tuple(sorted(key_edges))
        if key in seen_cycles:
            return
        seen_cycles.add(key)
        channel = m - min_size
        for u in path:
            for v in path:
                same_ring[u, v, channel] = 1.0
        for idx in range(m):
            u, v = path[idx], path[(idx + 1) % m]
            edge_in_ring[u, v, channel] = 1.0
            edge_in_ring[v, u, channel] = 1.0

    def dfs(start: int, current: int, path: list[int], visited: set[int]) -> None:
        if len(path) > max_size:
            return
        for nxt in neighbours[current]:
            if nxt == start:
                mark_cycle(path)
            elif nxt > start and nxt not in visited and len(path) < max_size:
                visited.add(nxt)
                path.append(nxt)
                dfs(start, int(nxt), path, visited)
                path.pop()
                visited.remove(nxt)

    for start in range(n):
        dfs(start, start, [start], {start})
    return torch.cat([same_ring, edge_in_ring], dim=-1)


def compute_sparse_relation_support(spd_cls: torch.Tensor, support_max: int) -> dict[str, torch.Tensor]:
    n = int(spd_cls.size(0))
    support = spd_cls <= int(support_max)
    pair_ij = torch.nonzero(support, as_tuple=False).long()
    edge_lookup = {(int(i), int(j)): edge for edge, (i, j) in enumerate(pair_ij.tolist())}

    tri_out: list[int] = []
    tri_left: list[int] = []
    tri_right: list[int] = []
    for edge_out, (i_raw, j_raw) in enumerate(pair_ij.tolist()):
        i = int(i_raw)
        j = int(j_raw)
        for k in range(n):
            if k == i or k == j:
                continue
            edge_left = edge_lookup.get((i, k))
            edge_right = edge_lookup.get((k, j))
            if edge_left is None or edge_right is None:
                continue
            tri_out.append(edge_out)
            tri_left.append(edge_left)
            tri_right.append(edge_right)

    empty = torch.empty((0,), dtype=torch.long)
    return {
        "sparse_pair_index": pair_ij.t().contiguous().cpu(),
        "triangle_out": torch.tensor(tri_out, dtype=torch.long) if tri_out else empty.clone(),
        "triangle_left": torch.tensor(tri_left, dtype=torch.long) if tri_left else empty.clone(),
        "triangle_right": torch.tensor(tri_right, dtype=torch.long) if tri_right else empty.clone(),
    }


def compute_graphgrape_features(data: SimpleNamespace, sparse_spd: int) -> dict[str, torch.Tensor]:
    x, edge_index, edge_attr, rwse, sym, degree = base_structural_features(data)
    n = int(data.num_nodes)
    adj = torch.zeros((n, n), dtype=torch.float32)
    if edge_index.numel() > 0:
        src, dst = edge_index
        adj[src, dst] = 1.0
    adj = torch.maximum(adj, adj.t())
    adj.fill_diagonal_(0.0)
    spd_cls = bfs_spd_classes(adj > 0, D_MAX_SPD)
    ring_pair = compute_ring_pair_features(adj > 0)
    pair_xi = torch.cat([ring_pair, sym], dim=-1).float()
    sparse_support = compute_sparse_relation_support(spd_cls, sparse_spd)
    return {
        "x": x.cpu(),
        "edge_index": edge_index.cpu(),
        "edge_attr": edge_attr.cpu(),
        "y": data.y.view(1).float().cpu(),
        "rwse": rwse.cpu(),
        "pair_xi": pair_xi.cpu(),
        "degree": degree.cpu(),
        **sparse_support,
    }


# =============================================================================
# Collation
# =============================================================================

def collate_static(graphs: list[dict[str, torch.Tensor]]) -> StaticBatch:
    batch_size = len(graphs)
    counts = [int(g["x"].numel()) for g in graphs]
    max_nodes = max(counts)
    total_nodes = sum(counts)

    xs, edge_indices, edge_attrs, ys, rwse_list, degree_list = [], [], [], [], [], []
    batch_index = torch.empty(total_nodes, dtype=torch.long)
    node_pos = torch.empty(total_nodes, dtype=torch.long)
    pair_xi = torch.zeros((batch_size, max_nodes, max_nodes, STATIC_P_PAIR), dtype=torch.float32)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    pair_mask = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)

    node_offset = 0
    for b, graph in enumerate(graphs):
        n = counts[b]
        xs.append(graph["x"])
        ys.append(graph["y"])
        rwse_list.append(graph["rwse"])
        degree_list.append(graph["degree"])
        edge_indices.append(graph["edge_index"] + node_offset)
        edge_attrs.append(graph["edge_attr"])
        batch_index[node_offset:node_offset + n] = b
        node_pos[node_offset:node_offset + n] = torch.arange(n, dtype=torch.long)
        pair_xi[b, :n, :n] = graph["pair_xi"]
        node_mask[b, :n] = True
        pair_mask[b, :n, :n] = True
        node_offset += n

    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.cat(edge_attrs, dim=0) if edge_attrs else torch.empty((0,), dtype=torch.long)
    return StaticBatch(
        x=torch.cat(xs, dim=0).long(),
        edge_index=edge_index.long(),
        edge_attr=edge_attr.long(),
        y=torch.cat(ys, dim=0).float(),
        rwse=torch.cat(rwse_list, dim=0).float(),
        pair_xi=pair_xi.float(),
        degree=torch.cat(degree_list, dim=0).float(),
        batch_index=batch_index,
        node_pos=node_pos,
        node_mask=node_mask,
        pair_mask=pair_mask,
        num_graphs=batch_size,
        max_nodes=max_nodes,
    )


def collate_graphgrape(graphs: list[dict[str, torch.Tensor]]) -> GraphGrapeBatch:
    batch_size = len(graphs)
    counts = [int(g["x"].numel()) for g in graphs]
    max_nodes = max(counts)
    total_nodes = sum(counts)

    xs, edge_indices, edge_attrs, ys, rwse_list, degree_list = [], [], [], [], [], []
    sparse_pair_batches, sparse_pair_is, sparse_pair_js = [], [], []
    sparse_pair_srcs, sparse_pair_dsts = [], []
    triangle_outs, triangle_lefts, triangle_rights = [], [], []
    batch_index = torch.empty(total_nodes, dtype=torch.long)
    node_pos = torch.empty(total_nodes, dtype=torch.long)
    pair_xi = torch.zeros((batch_size, max_nodes, max_nodes, GG_P_PAIR), dtype=torch.float32)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    pair_mask = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)

    node_offset = 0
    sparse_offset = 0
    for b, graph in enumerate(graphs):
        n = counts[b]
        xs.append(graph["x"])
        ys.append(graph["y"])
        rwse_list.append(graph["rwse"])
        degree_list.append(graph["degree"])
        edge_indices.append(graph["edge_index"] + node_offset)
        edge_attrs.append(graph["edge_attr"])
        batch_index[node_offset:node_offset + n] = b
        node_pos[node_offset:node_offset + n] = torch.arange(n, dtype=torch.long)
        pair_xi[b, :n, :n] = graph["pair_xi"]
        node_mask[b, :n] = True
        pair_mask[b, :n, :n] = True

        sparse_pair = graph["sparse_pair_index"].long()
        pi, pj = sparse_pair[0], sparse_pair[1]
        edge_count = int(pi.numel())
        sparse_pair_batches.append(torch.full((edge_count,), b, dtype=torch.long))
        sparse_pair_is.append(pi)
        sparse_pair_js.append(pj)
        sparse_pair_srcs.append(pi + node_offset)
        sparse_pair_dsts.append(pj + node_offset)

        tri_out = graph["triangle_out"].long()
        tri_left = graph["triangle_left"].long()
        tri_right = graph["triangle_right"].long()
        if tri_out.numel() > 0:
            triangle_outs.append(tri_out + sparse_offset)
            triangle_lefts.append(tri_left + sparse_offset)
            triangle_rights.append(tri_right + sparse_offset)

        node_offset += n
        sparse_offset += edge_count

    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.cat(edge_attrs, dim=0) if edge_attrs else torch.empty((0,), dtype=torch.long)
    sparse_pair_batch = torch.cat(sparse_pair_batches, dim=0) if sparse_pair_batches else torch.empty((0,), dtype=torch.long)
    sparse_pair_i = torch.cat(sparse_pair_is, dim=0) if sparse_pair_is else torch.empty((0,), dtype=torch.long)
    sparse_pair_j = torch.cat(sparse_pair_js, dim=0) if sparse_pair_js else torch.empty((0,), dtype=torch.long)
    sparse_pair_src = torch.cat(sparse_pair_srcs, dim=0) if sparse_pair_srcs else torch.empty((0,), dtype=torch.long)
    sparse_pair_dst = torch.cat(sparse_pair_dsts, dim=0) if sparse_pair_dsts else torch.empty((0,), dtype=torch.long)
    triangle_out = torch.cat(triangle_outs, dim=0) if triangle_outs else torch.empty((0,), dtype=torch.long)
    triangle_left = torch.cat(triangle_lefts, dim=0) if triangle_lefts else torch.empty((0,), dtype=torch.long)
    triangle_right = torch.cat(triangle_rights, dim=0) if triangle_rights else torch.empty((0,), dtype=torch.long)
    degree = torch.cat(degree_list, dim=0).float()

    return GraphGrapeBatch(
        x=torch.cat(xs, dim=0).long(),
        edge_index=edge_index.long(),
        edge_attr=edge_attr.long(),
        y=torch.cat(ys, dim=0).float(),
        rwse=torch.cat(rwse_list, dim=0).float(),
        pair_xi=pair_xi.float(),
        sparse_pair_batch=sparse_pair_batch,
        sparse_pair_i=sparse_pair_i,
        sparse_pair_j=sparse_pair_j,
        sparse_pair_src=sparse_pair_src,
        sparse_pair_dst=sparse_pair_dst,
        triangle_out=triangle_out,
        triangle_left=triangle_left,
        triangle_right=triangle_right,
        degree=degree,
        degree_log=torch.log1p(degree),
        batch_index=batch_index,
        node_pos=node_pos,
        node_mask=node_mask,
        pair_mask=pair_mask,
        num_graphs=batch_size,
        max_nodes=max_nodes,
    )


def flat_to_dense(x: torch.Tensor, batch: StaticBatch | GraphGrapeBatch) -> torch.Tensor:
    out = x.new_zeros((batch.num_graphs, batch.max_nodes, x.size(-1)))
    out[batch.batch_index, batch.node_pos] = x
    return out


def dense_to_flat(x: torch.Tensor, batch: StaticBatch | GraphGrapeBatch) -> torch.Tensor:
    return x[batch.node_mask]


# =============================================================================
# Dense GRIT-like proxy
# =============================================================================

class DensePairBiasLayer(nn.Module):
    def __init__(self, pair_dim: int):
        super().__init__()
        self.pre_attn_bn = nn.BatchNorm1d(D_MODEL)
        self.pre_ffn_bn = nn.BatchNorm1d(D_MODEL)
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL)
        self.pair_bias = nn.Linear(pair_dim, N_HEADS, bias=False)
        self.attn_dropout = nn.Dropout(ATTN_DROPOUT)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN_MULT * D_MODEL),
            nn.GELU(),
            nn.Dropout(RESID_DROPOUT),
            nn.Linear(FFN_MULT * D_MODEL, D_MODEL),
            nn.Dropout(RESID_DROPOUT),
        )

    def forward(self, h_flat: torch.Tensor, batch: StaticBatch) -> torch.Tensor:
        h_norm = flat_to_dense(self.pre_attn_bn(h_flat), batch)
        bsz, max_nodes, _ = h_norm.shape
        q = self.q_proj(h_norm).view(bsz, max_nodes, N_HEADS, D_HEAD).transpose(1, 2)
        k = self.k_proj(h_norm).view(bsz, max_nodes, N_HEADS, D_HEAD).transpose(1, 2)
        v = self.v_proj(h_norm).view(bsz, max_nodes, N_HEADS, D_HEAD).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (D_HEAD ** -0.5)
        logits = logits + self.pair_bias(batch.pair_xi).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~batch.node_mask[:, None, None, :], float("-inf"))
        alpha = torch.softmax(logits, dim=-1)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
        alpha = self.attn_dropout(alpha)
        out = torch.matmul(alpha, v).transpose(1, 2).contiguous().view(bsz, max_nodes, D_MODEL)
        out = self.o_proj(out).masked_fill(~batch.node_mask.unsqueeze(-1), 0.0)
        h = h_flat + dense_to_flat(out, batch)
        return h + self.ffn(self.pre_ffn_bn(h))


class DenseRrwpGritProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES, D_MODEL)
        self.bond_emb = nn.Embedding(NUM_BOND_TYPES, D_MODEL)
        self.rwse_proj = nn.Linear(K_WALK, D_MODEL)
        self.layers = nn.ModuleList([DensePairBiasLayer(STATIC_P_PAIR) for _ in range(N_LAYERS)])
        self.final_bn = nn.BatchNorm1d(D_MODEL)
        self.readout = nn.Sequential(
            nn.Linear(D_MODEL, 2 * D_MODEL),
            nn.GELU(),
            nn.Linear(2 * D_MODEL, 1),
        )

    def forward(self, batch: StaticBatch) -> torch.Tensor:
        h = self.atom_emb(batch.x)
        edge_msg = self.bond_emb(batch.edge_attr)
        bond_sum = h.new_zeros(h.shape)
        if batch.edge_index.numel() > 0:
            bond_sum.index_add_(0, batch.edge_index[1], edge_msg)
        h = h + bond_sum + self.rwse_proj(batch.rwse)
        for layer in self.layers:
            h = layer(h, batch)
        h = self.final_bn(h)
        pooled = h.new_zeros((batch.num_graphs, D_MODEL))
        pooled.index_add_(0, batch.batch_index, h)
        return self.readout(pooled).view(-1)


# =============================================================================
# CSA static anchor
# =============================================================================

class StaticAnchorLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attn_bn = nn.BatchNorm1d(D_MODEL)
        self.pre_ffn_bn = nn.BatchNorm1d(D_MODEL)
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL)
        self.bias_weight = nn.Parameter(torch.empty(N_HEADS, STATIC_P_PAIR))
        self.value_weight = nn.Parameter(torch.empty(N_HEADS, D_HEAD, STATIC_P_PAIR))
        self.bias_gate = nn.Parameter(torch.ones(N_HEADS))
        self.theta1 = nn.Parameter(torch.ones(D_MODEL))
        self.theta2 = nn.Parameter(torch.zeros(D_MODEL))
        self.attn_dropout = nn.Dropout(ATTN_DROPOUT)
        self.resid_dropout = nn.Dropout(RESID_DROPOUT)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN_MULT * D_MODEL),
            nn.GELU(),
            nn.Dropout(RESID_DROPOUT),
            nn.Linear(FFN_MULT * D_MODEL, D_MODEL),
            nn.Dropout(RESID_DROPOUT),
        )
        nn.init.normal_(self.bias_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.value_weight, mean=0.0, std=0.02)

    def forward(self, h_flat: torch.Tensor, batch: StaticBatch) -> torch.Tensor:
        h_in = h_flat
        h_norm = flat_to_dense(self.pre_attn_bn(h_flat), batch)
        bsz, max_nodes, _ = h_norm.shape
        q = self.q_proj(h_norm).view(bsz, max_nodes, N_HEADS, D_HEAD).transpose(1, 2)
        k = self.k_proj(h_norm).view(bsz, max_nodes, N_HEADS, D_HEAD).transpose(1, 2)
        v = self.v_proj(h_norm).view(bsz, max_nodes, N_HEADS, D_HEAD).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * (D_HEAD ** -0.5)
        xi = batch.pair_xi.masked_fill(~batch.pair_mask.unsqueeze(-1), 0.0)
        bias = torch.einsum("bijp,hp->bhij", xi, self.bias_weight)
        logits = logits + bias * self.bias_gate.view(1, N_HEADS, 1, 1)
        logits = logits.masked_fill(~batch.node_mask[:, None, None, :], float("-inf"))
        alpha = torch.softmax(logits, dim=-1)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
        alpha = self.attn_dropout(alpha)

        out = torch.matmul(alpha, v)
        value_pair = torch.einsum("bijp,hdp->bhijd", xi, self.value_weight)
        out = out + torch.einsum("bhij,bhijd->bhid", alpha, value_pair)
        out = out.transpose(1, 2).contiguous().view(bsz, max_nodes, D_MODEL)
        out = self.o_proj(out).masked_fill(~batch.node_mask.unsqueeze(-1), 0.0)
        out_flat = dense_to_flat(out, batch)

        degree_log = torch.log1p(batch.degree).view(-1, 1)
        scaled = out_flat * self.theta1.view(1, -1) + degree_log * out_flat * self.theta2.view(1, -1)
        h_half = h_in + self.resid_dropout(scaled)
        return h_half + self.ffn(self.pre_ffn_bn(h_half))


class StaticAnchorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES, D_MODEL)
        self.bond_emb = nn.Embedding(NUM_BOND_TYPES, D_MODEL)
        self.rwse_proj = nn.Linear(K_WALK, D_MODEL)
        self.layers = nn.ModuleList([StaticAnchorLayer() for _ in range(N_LAYERS)])
        self.final_bn = nn.BatchNorm1d(D_MODEL)
        self.readout = nn.Sequential(
            nn.Linear(D_MODEL, 2 * D_MODEL),
            nn.GELU(),
            nn.Linear(2 * D_MODEL, 1),
        )

    def forward(self, batch: StaticBatch) -> torch.Tensor:
        h = self.atom_emb(batch.x)
        edge_msg = self.bond_emb(batch.edge_attr)
        bond_sum = h.new_zeros(h.shape)
        if batch.edge_index.numel() > 0:
            bond_sum.index_add_(0, batch.edge_index[1], edge_msg)
        h = h + bond_sum + self.rwse_proj(batch.rwse)
        for layer in self.layers:
            h = layer(h, batch)
        h = self.final_bn(h)
        pooled = h.new_zeros((batch.num_graphs, D_MODEL))
        pooled.index_add_(0, batch.batch_index, h)
        return self.readout(pooled).view(-1)


# =============================================================================
# GraphGrape v21 standalone implementation
# =============================================================================

class LastDimRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = RMS_NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


class GraphGrapeLayer(nn.Module):
    def __init__(self, *, use_triangle_update: bool):
        super().__init__()
        self.use_triangle_update = bool(use_triangle_update)
        self.pre_attn_bn = nn.BatchNorm1d(D_MODEL)
        self.pre_ffn_bn = nn.BatchNorm1d(D_MODEL)
        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL)

        bias_input_dim = GG_P_PAIR + P_LATENT_PAIR
        self.bias_weight = nn.Parameter(torch.empty(N_HEADS, bias_input_dim))
        self.value_weight = nn.Parameter(torch.empty(N_HEADS, D_HEAD, bias_input_dim))
        self.dynamic_bias_logit = nn.Parameter(torch.full((1,), logit_from_prob(DYNAMIC_GATE_INIT)))
        self.dynamic_value_logit = nn.Parameter(torch.full((1,), logit_from_prob(DYNAMIC_GATE_INIT)))
        self.bias_gate = nn.Parameter(torch.ones(N_HEADS))
        self.theta1 = nn.Parameter(torch.ones(D_MODEL))
        self.theta2 = nn.Parameter(torch.zeros(D_MODEL))
        nn.init.normal_(self.bias_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.value_weight, mean=0.0, std=0.02)

        self.attn_dropout = nn.Dropout(ATTN_DROPOUT)
        self.resid_dropout = nn.Dropout(RESID_DROPOUT)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN_MULT * D_MODEL),
            nn.GELU(),
            nn.Dropout(RESID_DROPOUT),
            nn.Linear(FFN_MULT * D_MODEL, D_MODEL),
            nn.Dropout(RESID_DROPOUT),
        )

        self.pair_read_norm = LastDimRMSNorm(P_LATENT_PAIR)
        self.pair_write_state_norm = LastDimRMSNorm(P_LATENT_PAIR)
        self.pair_proposal_norm = LastDimRMSNorm(P_LATENT_PAIR)
        self.h_pair_norm = LastDimRMSNorm(D_MODEL)
        self.pair_u_proj = nn.Linear(D_MODEL, PAIR_UPDATE_RANK, bias=False)
        self.pair_v_proj = nn.Linear(D_MODEL, PAIR_UPDATE_RANK, bias=False)
        self.pair_u_norm = LastDimRMSNorm(PAIR_UPDATE_RANK)
        self.pair_v_norm = LastDimRMSNorm(PAIR_UPDATE_RANK)
        self.pair_out = nn.Linear(PAIR_UPDATE_RANK * PAIR_UPDATE_RANK, P_LATENT_PAIR)
        self.pair_write_gate = nn.Linear(P_LATENT_PAIR + GG_P_PAIR, 1)
        self.pair_update_log_scale = nn.Parameter(
            torch.full((1,), softplus_inverse(PAIR_UPDATE_SCALE_INIT))
        )

        if self.use_triangle_update:
            self.pair_relation_norm = LastDimRMSNorm(P_LATENT_PAIR)
            self.pair_anchor = nn.Linear(GG_P_PAIR, P_LATENT_PAIR, bias=False)
            self.triangle_left = nn.Linear(P_LATENT_PAIR, P_LATENT_PAIR, bias=False)
            self.triangle_right = nn.Linear(P_LATENT_PAIR, P_LATENT_PAIR, bias=False)
            self.triangle_left_norm = LastDimRMSNorm(P_LATENT_PAIR)
            self.triangle_right_norm = LastDimRMSNorm(P_LATENT_PAIR)
            self.triangle_msg_norm = LastDimRMSNorm(P_LATENT_PAIR)
            self.triangle_out = nn.Linear(P_LATENT_PAIR, P_LATENT_PAIR)
            self.triangle_update_log_scale = nn.Parameter(
                torch.full((1,), softplus_inverse(TRIANGLE_UPDATE_SCALE_INIT))
            )

        nn.init.normal_(self.pair_out.weight, mean=0.0, std=PAIR_OUT_INIT_STD)
        nn.init.zeros_(self.pair_out.bias)
        if self.use_triangle_update:
            nn.init.normal_(self.pair_anchor.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.triangle_out.weight, mean=0.0, std=TRIANGLE_OUT_INIT_STD)
            nn.init.zeros_(self.triangle_out.bias)
        nn.init.zeros_(self.pair_write_gate.weight)
        nn.init.constant_(self.pair_write_gate.bias, logit_from_prob(PAIR_WRITE_GATE_INIT))

    def dynamic_bias_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.dynamic_bias_logit)

    def dynamic_value_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.dynamic_value_logit)

    def pair_update_scale(self) -> torch.Tensor:
        return F.softplus(self.pair_update_log_scale)

    def triangle_update_scale(self) -> torch.Tensor:
        if not self.use_triangle_update:
            return self.pair_update_log_scale.new_tensor(0.0)
        return F.softplus(self.triangle_update_log_scale)

    def sparse_dynamic_value_average(
        self,
        alpha: torch.Tensor,
        value_dynamic_edge: torch.Tensor,
        batch: GraphGrapeBatch,
    ) -> torch.Tensor:
        bsz, heads, max_nodes, _ = alpha.shape
        out = alpha.new_zeros((bsz * heads * max_nodes, P_LATENT_PAIR))
        edge_count = int(batch.sparse_pair_batch.numel())
        if edge_count == 0:
            return out.view(bsz, heads, max_nodes, P_LATENT_PAIR)
        alpha_edge = alpha.permute(0, 2, 3, 1)[
            batch.sparse_pair_batch, batch.sparse_pair_i, batch.sparse_pair_j
        ]
        head_idx = torch.arange(heads, device=alpha.device)
        flat_idx = (
            (batch.sparse_pair_batch[:, None] * heads + head_idx[None, :]) * max_nodes
            + batch.sparse_pair_i[:, None]
        ).reshape(-1)
        msg = (alpha_edge.unsqueeze(-1) * value_dynamic_edge[:, None, :]).reshape(
            -1, P_LATENT_PAIR
        )
        out.index_add_(0, flat_idx, msg)
        return out.view(bsz, heads, max_nodes, P_LATENT_PAIR)

    def forward(
        self,
        h_flat: torch.Tensor,
        p_pair: torch.Tensor,
        batch: GraphGrapeBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_in = h_flat
        h_norm = flat_to_dense(self.pre_attn_bn(h_flat), batch)
        bsz, max_nodes, _ = h_norm.shape
        q = self.q_proj(h_norm).view(bsz, max_nodes, N_HEADS, D_HEAD).transpose(1, 2)
        k = self.k_proj(h_norm).view(bsz, max_nodes, N_HEADS, D_HEAD).transpose(1, 2)
        v = self.v_proj(h_norm).view(bsz, max_nodes, N_HEADS, D_HEAD).transpose(1, 2)

        xi_pair = batch.pair_xi
        xi_edge = xi_pair[batch.sparse_pair_batch, batch.sparse_pair_i, batch.sparse_pair_j]
        p_read = self.pair_read_norm(p_pair)
        bias_dynamic_edge = self.dynamic_bias_gate() * p_read
        value_dynamic_edge = self.dynamic_value_gate() * p_read

        logits_base = torch.matmul(q, k.transpose(-2, -1)) * (D_HEAD ** -0.5)
        bias_static = torch.einsum("bijp,hp->bhij", xi_pair, self.bias_weight[:, :GG_P_PAIR])
        bias_static = bias_static * self.bias_gate.view(1, N_HEADS, 1, 1)

        bias_dynamic = logits_base.new_zeros((bsz, max_nodes, max_nodes, N_HEADS))
        if bias_dynamic_edge.numel() > 0:
            bias_dynamic_edge_h = torch.einsum(
                "ep,hp->eh", bias_dynamic_edge, self.bias_weight[:, GG_P_PAIR:]
            )
            bias_dynamic_edge_h = bias_dynamic_edge_h * self.bias_gate.view(1, N_HEADS)
            bias_dynamic[
                batch.sparse_pair_batch, batch.sparse_pair_i, batch.sparse_pair_j
            ] = bias_dynamic_edge_h
        logits = logits_base + bias_static + bias_dynamic.permute(0, 3, 1, 2).contiguous()
        logits = logits.masked_fill(~batch.node_mask[:, None, None, :], float("-inf"))
        alpha_clean = torch.softmax(logits, dim=-1)
        alpha_clean = torch.where(torch.isfinite(alpha_clean), alpha_clean, torch.zeros_like(alpha_clean))
        alpha_drop = alpha_clean if not self.training else self.attn_dropout(alpha_clean)

        out_base_clean = torch.matmul(alpha_clean, v)
        out_base = out_base_clean if not self.training else torch.matmul(alpha_drop, v)

        static_weight = self.value_weight[:, :, :GG_P_PAIR]
        dynamic_weight = self.value_weight[:, :, GG_P_PAIR:]
        static_avg_clean = torch.einsum("bhij,bijp->bhip", alpha_clean, xi_pair)
        static_avg = static_avg_clean if not self.training else torch.einsum(
            "bhij,bijp->bhip", alpha_drop, xi_pair
        )
        value_static = torch.einsum("bhip,hdp->bhid", static_avg, static_weight)
        dynamic_avg = self.sparse_dynamic_value_average(alpha_drop, value_dynamic_edge, batch)
        value_dynamic = torch.einsum("bhip,hdp->bhid", dynamic_avg, dynamic_weight)
        out = out_base + value_static + value_dynamic

        out_clean = out_base_clean + torch.einsum(
            "bhip,hdp->bhid", static_avg_clean, static_weight
        )
        clean_dynamic_avg = self.sparse_dynamic_value_average(alpha_clean, value_dynamic_edge, batch)
        out_clean = out_clean + torch.einsum("bhip,hdp->bhid", clean_dynamic_avg, dynamic_weight)

        out_clean = out_clean.transpose(1, 2).contiguous().view(bsz, max_nodes, D_MODEL)
        out_clean = self.o_proj(out_clean).masked_fill(~batch.node_mask.unsqueeze(-1), 0.0)
        out_clean_flat = dense_to_flat(out_clean, batch)

        out = out.transpose(1, 2).contiguous().view(bsz, max_nodes, D_MODEL)
        out = self.o_proj(out).masked_fill(~batch.node_mask.unsqueeze(-1), 0.0)
        out_flat = dense_to_flat(out, batch)

        scaled = out_flat * self.theta1.view(1, -1) + batch.degree_log.view(-1, 1) * out_flat * self.theta2.view(1, -1)
        h_half = h_in + self.resid_dropout(scaled)
        scaled_clean = out_clean_flat * self.theta1.view(1, -1) + batch.degree_log.view(-1, 1) * out_clean_flat * self.theta2.view(1, -1)
        h_pair_post = h_in + scaled_clean
        h_out = h_half + self.ffn(self.pre_ffn_bn(h_half))

        h_post = self.h_pair_norm(h_pair_post)
        u = self.pair_u_norm(self.pair_u_proj(h_post))
        vv = self.pair_v_norm(self.pair_v_proj(h_post))
        u_src = u[batch.sparse_pair_src]
        v_dst = vv[batch.sparse_pair_dst]
        outer = (u_src[:, :, None] * v_dst[:, None, :]).flatten(start_dim=-2)
        endpoint = self.pair_out(outer) * (PAIR_UPDATE_RANK ** -0.5)

        triangle = endpoint.new_zeros(endpoint.shape)
        tri_count = int(batch.triangle_out.numel())
        triangle_scale = self.triangle_update_scale()
        if self.use_triangle_update and tri_count > 0:
            relation = self.pair_relation_norm(p_pair + self.pair_anchor(xi_edge))
            left = self.triangle_left_norm(self.triangle_left(relation))
            right = self.triangle_right_norm(self.triangle_right(relation))
            tri_msg = self.triangle_msg_norm(left[batch.triangle_left] * right[batch.triangle_right])
            triangle.index_add_(0, batch.triangle_out, tri_msg)
            denom = endpoint.new_zeros((endpoint.size(0), 1))
            denom.index_add_(0, batch.triangle_out, endpoint.new_ones((tri_count, 1)))
            triangle = self.triangle_out(triangle / denom.clamp_min(1.0))

        proposal = self.pair_proposal_norm(endpoint + triangle_scale * triangle)
        p_gate_state = self.pair_write_state_norm(p_pair)
        write_gate = torch.sigmoid(self.pair_write_gate(torch.cat([p_gate_state, xi_edge], dim=-1)))
        p_next = p_pair + self.pair_update_scale() * write_gate * proposal
        return h_out, p_next


class GraphGrapeModel(nn.Module):
    def __init__(self, *, use_triangle_update: bool):
        super().__init__()
        self.use_triangle_update = bool(use_triangle_update)
        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES, D_MODEL)
        self.bond_emb = nn.Embedding(NUM_BOND_TYPES, D_MODEL)
        self.rwse_proj = nn.Linear(K_WALK, D_MODEL)
        self.layers = nn.ModuleList(
            [GraphGrapeLayer(use_triangle_update=use_triangle_update) for _ in range(N_LAYERS)]
        )
        self.final_bn = nn.BatchNorm1d(D_MODEL)
        self.readout = nn.Sequential(
            nn.Linear(D_MODEL, 2 * D_MODEL),
            nn.GELU(),
            nn.Linear(2 * D_MODEL, 1),
        )

    def forward(self, batch: GraphGrapeBatch) -> torch.Tensor:
        h = self.atom_emb(batch.x)
        edge_msg = self.bond_emb(batch.edge_attr)
        bond_sum = h.new_zeros(h.shape)
        if batch.edge_index.numel() > 0:
            bond_sum.index_add_(0, batch.edge_index[1], edge_msg)
        h = h + bond_sum + self.rwse_proj(batch.rwse)
        p_pair = batch.pair_xi.new_zeros((batch.sparse_pair_batch.numel(), P_LATENT_PAIR))
        for layer in self.layers:
            h, p_pair = layer(h, p_pair, batch)
        h = self.final_bn(h)
        pooled = h.new_zeros((batch.num_graphs, D_MODEL))
        pooled.index_add_(0, batch.batch_index, h)
        return self.readout(pooled).view(-1)


# =============================================================================
# Benchmark runner
# =============================================================================

def summarize_sparse_features(features: list[dict[str, torch.Tensor]]) -> tuple[float, float]:
    sparse_counts = []
    triangle_counts = []
    for graph in features:
        sparse = graph.get("sparse_pair_index")
        triangles = graph.get("triangle_out")
        sparse_counts.append(float(sparse.size(1)) if sparse is not None else 0.0)
        triangle_counts.append(float(triangles.numel()) if triangles is not None else 0.0)
    return mean(sparse_counts), mean(triangle_counts)


def make_method(
    method: str,
    raw_graphs: list[SimpleNamespace],
    batch_size: int,
) -> tuple[nn.Module, torch.utils.data.DataLoader, float, float, float]:
    t0 = time.perf_counter()
    if method == "grit_dense_rrwp_proxy":
        features = [compute_static_features(graph) for graph in raw_graphs]
        model: nn.Module = DenseRrwpGritProxy()
        collate_fn = collate_static
    elif method == "csa_static_anchor":
        features = [compute_static_features(graph) for graph in raw_graphs]
        model = StaticAnchorModel()
        collate_fn = collate_static
    elif method.startswith("graphgrape_v21"):
        sparse_spd = 2 if method.endswith("spd2") else 3
        use_triangle = "_triangle_" in method
        features = [compute_graphgrape_features(graph, sparse_spd=sparse_spd) for graph in raw_graphs]
        model = GraphGrapeModel(use_triangle_update=use_triangle)
        collate_fn = collate_graphgrape
    else:
        raise ValueError(f"Unknown method: {method}")

    preprocess_s = time.perf_counter() - t0
    dataset = TensorGraphDataset(features)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    avg_sparse_pairs, avg_triangle_triples = summarize_sparse_features(features)
    return model, loader, preprocess_s, avg_sparse_pairs, avg_triangle_triples


def train_timed(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    warmup_epochs: int,
) -> tuple[list[float], float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epoch_times: list[float] = []
    final_loss = float("nan")

    for epoch in range(1, epochs + 1):
        model.train()
        sync_device(device)
        t0 = time.perf_counter()
        total_loss = 0.0
        total_graphs = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = F.l1_loss(pred, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n_graphs = int(batch.num_graphs)
            total_loss += float(loss.detach().cpu()) * n_graphs
            total_graphs += n_graphs
        sync_device(device)
        elapsed = time.perf_counter() - t0
        final_loss = total_loss / max(1, total_graphs)
        epoch_times.append(elapsed)
        log(f"    epoch {epoch:02d}/{epochs}: {elapsed:.3f}s loss={final_loss:.5f}")

    measured = epoch_times[warmup_epochs:] if len(epoch_times) > warmup_epochs else epoch_times
    return measured, final_loss


def run_method(
    *,
    method: str,
    raw_graphs: list[SimpleNamespace],
    size_cfg: SizeConfig,
    repeat: int,
    num_graphs: int,
    batch_size: int,
    epochs: int,
    warmup_epochs: int,
    avg_extra_degree: float,
    avg_nodes: float,
    avg_directed_edges: float,
    device: torch.device,
) -> MethodResult:
    log(f"[run] size={size_cfg.size_label} method={method} repeat={repeat}")
    with MemorySampler(device) as mem:
        model, loader, preprocess_s, avg_sparse_pairs, avg_triangle_triples = make_method(
            method, raw_graphs, batch_size
        )
        param_count = count_parameters(model)
        log(
            f"    params={param_count:,} preprocess={preprocess_s:.2f}s "
            f"sparse_pairs={avg_sparse_pairs:.1f} triangles={avg_triangle_triples:.1f}"
        )
        measured_epoch_times, final_loss = train_timed(
            model=model,
            loader=loader,
            device=device,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
        )
        del model, loader

    train_measured_s = sum(measured_epoch_times)
    mean_epoch_s = mean(measured_epoch_times)
    result = MethodResult(
        method=method,
        size_label=size_cfg.size_label,
        repeat=repeat,
        device=str(device),
        num_graphs=num_graphs,
        batch_size=batch_size,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        min_nodes=size_cfg.min_nodes,
        max_nodes=size_cfg.max_nodes,
        avg_extra_degree=avg_extra_degree,
        param_count=param_count,
        preprocess_s=preprocess_s,
        train_measured_s=train_measured_s,
        mean_epoch_s=mean_epoch_s,
        median_epoch_s=median(measured_epoch_times),
        std_epoch_s=std(measured_epoch_times),
        graphs_per_s=num_graphs / mean_epoch_s if mean_epoch_s > 0 else float("inf"),
        final_loss=final_loss,
        avg_nodes=avg_nodes,
        avg_directed_edges=avg_directed_edges,
        avg_sparse_pairs=avg_sparse_pairs,
        avg_triangle_triples=avg_triangle_triples,
        rss_start_mb=mem.start_rss_mb,
        rss_peak_mb=mem.peak_rss_mb,
        rss_end_mb=mem.end_rss_mb,
        cuda_peak_allocated_mb=mem.cuda_peak_allocated_mb,
        cuda_peak_reserved_mb=mem.cuda_peak_reserved_mb,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def print_summary(results: list[MethodResult]) -> None:
    log("")
    log(
        "size  method                         epoch_s  graphs/s  "
        "rss_peak  cuda_alloc  cuda_reserved  sparse_pairs  triangles"
    )
    log(
        "----  -----------------------------  -------  --------  "
        "--------  ----------  -------------  ------------  ---------"
    )
    for result in sorted(results, key=lambda r: (r.min_nodes, r.max_nodes, r.mean_epoch_s)):
        log(
            f"{result.size_label:<4}  "
            f"{result.method:<29}  "
            f"{result.mean_epoch_s:>7.3f}  "
            f"{result.graphs_per_s:>8.1f}  "
            f"{result.rss_peak_mb:>8.0f}  "
            f"{result.cuda_peak_allocated_mb:>10.0f}  "
            f"{result.cuda_peak_reserved_mb:>13.0f}  "
            f"{result.avg_sparse_pairs:>12.1f}  "
            f"{result.avg_triangle_triples:>9.1f}"
        )


def write_results(results: list[MethodResult], output_dir: Path, run_config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]

    csv_path = output_dir / "wallclock_memory_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "wallclock_memory_summary.json"
    env = {
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"config": run_config, "environment": env, "results": rows}, f, indent=2)

    log(f"[write] {csv_path}")
    log(f"[write] {json_path}")


# =============================================================================
# CLI / main
# =============================================================================

def default_output_dir() -> Path:
    root = Path("/content") if Path("/content").exists() else Path.cwd()
    return root / "wallclock_graph_methods" / datetime.now().strftime("%Y%m%d_%H%M%S")


def strip_notebook_args(argv: list[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for idx, tok in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if tok == "-f" and idx + 1 < len(argv) and "kernel-" in argv[idx + 1]:
            skip_next = True
            continue
        if "kernel-" in tok and tok.endswith(".json"):
            continue
        out.append(tok)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        if CELL_ARGS is None:
            argv = sys.argv[1:]
        elif isinstance(CELL_ARGS, str):
            import shlex

            argv = shlex.split(CELL_ARGS)
        else:
            argv = list(CELL_ARGS)
    argv = strip_notebook_args(list(argv))

    parser = argparse.ArgumentParser(
        description="Standalone Colab wallclock + memory benchmark on non-ZINC synthetic graphs."
    )
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=list(METHOD_CHOICES))
    parser.add_argument("--node-sizes", nargs="+", type=int, default=[32, 64, 128, 192])
    parser.add_argument("--num-graphs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--avg-extra-degree", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny smoke run: sizes 16 and 32, 8 graphs, 2 epochs, batch 4.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick:
        args.node_sizes = [16, 32]
        args.num_graphs = 8
        args.batch_size = 4
        args.epochs = 2
        args.warmup_epochs = 1

    output_dir = args.output_dir or default_output_dir()
    device = choose_device(args.device)
    warmup_epochs = max(0, min(int(args.warmup_epochs), int(args.epochs) - 1))
    run_config = {
        "methods": list(args.methods),
        "node_sizes": list(args.node_sizes),
        "num_graphs": args.num_graphs,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "warmup_epochs": warmup_epochs,
        "avg_extra_degree": args.avg_extra_degree,
        "seed": args.seed,
        "repeats": args.repeats,
        "device": str(device),
        "output_dir": str(output_dir),
        "model": {
            "layers": N_LAYERS,
            "hidden_dim": D_MODEL,
            "heads": N_HEADS,
            "k_walk": K_WALK,
            "d_max_spd": D_MAX_SPD,
        },
    }

    set_seed(args.seed)
    log(f"[env] torch={torch.__version__} device={device}")
    if device.type == "cuda":
        log(f"[env] cuda_device={torch.cuda.get_device_name(device)}")
    log(f"[out] {output_dir}")
    log(
        f"[config] sizes={args.node_sizes} graphs/size={args.num_graphs} "
        f"batch={args.batch_size} epochs={args.epochs} measured_after_warmup={warmup_epochs}"
    )

    results: list[MethodResult] = []
    size_configs = [
        SizeConfig(size_label=str(size), min_nodes=int(size), max_nodes=int(size), seed=args.seed + 1000 * i)
        for i, size in enumerate(args.node_sizes)
    ]

    for size_cfg in size_configs:
        raw_graphs = generate_graphs(
            num_graphs=args.num_graphs,
            min_nodes=size_cfg.min_nodes,
            max_nodes=size_cfg.max_nodes,
            avg_extra_degree=args.avg_extra_degree,
            seed=size_cfg.seed,
        )
        avg_nodes, avg_directed_edges = graph_stats(raw_graphs)
        log(
            f"[task] size={size_cfg.size_label} graphs={args.num_graphs} "
            f"avg_nodes={avg_nodes:.1f} avg_directed_edges={avg_directed_edges:.1f}"
        )
        for repeat in range(args.repeats):
            for method in args.methods:
                set_seed(size_cfg.seed + repeat)
                result = run_method(
                    method=method,
                    raw_graphs=raw_graphs,
                    size_cfg=size_cfg,
                    repeat=repeat,
                    num_graphs=args.num_graphs,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    warmup_epochs=warmup_epochs,
                    avg_extra_degree=args.avg_extra_degree,
                    avg_nodes=avg_nodes,
                    avg_directed_edges=avg_directed_edges,
                    device=device,
                )
                results.append(result)
                log(
                    f"[done] size={size_cfg.size_label} {method}: "
                    f"epoch={result.mean_epoch_s:.3f}s graphs/s={result.graphs_per_s:.1f} "
                    f"rss_peak={result.rss_peak_mb:.0f}MB "
                    f"cuda_peak={result.cuda_peak_allocated_mb:.0f}MB"
                )

    print_summary(results)
    write_results(results, output_dir, run_config)


if __name__ == "__main__":
    main()
