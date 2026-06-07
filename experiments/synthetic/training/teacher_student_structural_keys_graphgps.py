#!/usr/bin/env python3
"""Teacher-student structural-key routing benchmark for small GraphGPS models.

The teacher is synthetic and deterministic: for each graph it chooses one source
candidate node for a query node, then the target label is the value attached to
that source candidate. The student only sees graph input and output labels.

The suite compares three routing modes with the same outer format:

* symbolic: route by explicit key tuple equality;
* structural: route by equality of hidden graph-derived structural keys;
* mixed: route by the conjunction of explicit symbolic key and hidden
  structural key, with typed shortcut distractors.

The default run is now a one-head/one-layer starter benchmark for the symbolic
copy task and one structural copy task. The structural starter uses anchor
distance structural keys and PE-only evidence, and value leaves are blank by
default so the easiest route is to attend to the selected candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Colab/notebook hook. If this whole file is pasted into a notebook cell, edit
# CELL_ARGS to override defaults without fighting IPython kernel argv.
CELL_ARGS: list[str] | str | None = None


TeacherTask = Literal["symbolic", "structural", "mixed"]
StructuralKeyFamily = Literal["anchor_distance", "local", "diffusion", "global"]
GraphFamily = Literal["er", "ws", "ba", "sbm", "mixed"]
FeatureSet = Literal[
    "none",
    "anchor_dist",
    "degree",
    "rwse",
    "rwse_anchor_dist",
    "stats",
    "rwse_stats",
    "spd_bias",
    "rwse_spd_bias",
    "oracle_key",
]

TASK_CHOICES = ("symbolic", "structural", "mixed")
STRUCTURAL_KEY_FAMILIES = ("anchor_distance", "local", "diffusion", "global")
GRAPH_FAMILIES = ("er", "ws", "ba", "sbm", "mixed")
FEATURE_SETS = (
    "none",
    "anchor_dist",
    "degree",
    "rwse",
    "rwse_anchor_dist",
    "stats",
    "rwse_stats",
    "spd_bias",
    "rwse_spd_bias",
    "oracle_key",
)
SUITE_PRESETS = ("one_head_one_layer", "single_head_fast", "pilot", "family_scan", "full", "custom")

M_POSITIONAL = "positional_score"
M_SYMBOLIC = "symbolic_score"
M_PE_INVARIANT = "pe_invariance"
M_PE_EQUIVARIANT = "pe_equivariance"
M_ENTANGLEMENT = "entanglement_score"
M_ENTROPY = "entropy_norm"
M_RELABEL_EQUIVARIANT = "relabel_equivariance"

PAD_ROLE = 0
BACKBONE_ROLE = 1
CANDIDATE_ROLE = 2
VALUE_ROLE = 3
QUERY_ROLE = 4
KEY_ROLE = 5
ANCHOR_A_ROLE = 6
ANCHOR_B_ROLE = 7
NUM_ROLES = 8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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


def running_in_notebook() -> bool:
    launcher = Path(sys.argv[0]).name
    if launcher in {"ipykernel_launcher.py", "colab_kernel_launcher.py"}:
        return True
    if "google.colab" in sys.modules or "ipykernel" in sys.modules:
        return True
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except Exception:
        return False
    return shell in {"ZMQInteractiveShell", "Shell"}


def notebook_safe_argv(argv: list[str] | None) -> list[str] | None:
    if argv is not None:
        return argv
    if not running_in_notebook():
        return sys.argv[1:]
    if CELL_ARGS is None:
        return []
    if isinstance(CELL_ARGS, str):
        return shlex.split(CELL_ARGS)
    return list(CELL_ARGS)


def symbol_vocab_size(key_vocab_size: int, value_vocab_size: int) -> int:
    return 1 + key_vocab_size + value_vocab_size


def key_symbol(key: int) -> int:
    return 1 + int(key)


def value_symbol(value: int, key_vocab_size: int) -> int:
    return 1 + int(key_vocab_size) + int(value)


def adjacency_from_edges(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    for a, b in edges:
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    return adj


def neighbours_from_adj(adj: np.ndarray) -> list[list[int]]:
    return [np.flatnonzero(row > 0).astype(np.int64).tolist() for row in adj]


def all_pairs_distances(adj: np.ndarray) -> np.ndarray:
    neighbours = neighbours_from_adj(adj)
    n = len(neighbours)
    out = np.zeros((n, n), dtype=np.int64)
    for source in range(n):
        dist = np.full(n, -1, dtype=np.int64)
        dist[source] = 0
        queue = [source]
        for node in queue:
            for nxt in neighbours[node]:
                if dist[nxt] < 0:
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
        out[source] = dist
    return out


def random_tree_edges(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    order = np.arange(n)
    rng.shuffle(order)
    edges = []
    for idx in range(1, n):
        child = int(order[idx])
        parent = int(order[int(rng.integers(0, idx))])
        edges.append((child, parent))
    return edges


def add_unique_edge(edges: set[tuple[int, int]], a: int, b: int) -> None:
    if a == b:
        return
    left, right = sorted((int(a), int(b)))
    edges.add((left, right))


def graphworld_backbone_edges(
    n: int,
    family: GraphFamily,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], str]:
    if family == "mixed":
        family = str(rng.choice(["er", "ws", "ba", "sbm"]))  # type: ignore[assignment]

    edge_set: set[tuple[int, int]] = set()
    for a, b in random_tree_edges(n, rng):
        add_unique_edge(edge_set, a, b)

    if family == "er":
        p = float(rng.uniform(1.8 / n, 4.5 / n))
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < p:
                    add_unique_edge(edge_set, a, b)
    elif family == "ws":
        k = int(rng.choice([4, 6, 8]))
        beta = float(rng.uniform(0.05, 0.35))
        for a in range(n):
            for step in range(1, k // 2 + 1):
                b = (a + step) % n
                if rng.random() < beta:
                    b = int(rng.integers(0, n))
                add_unique_edge(edge_set, a, b)
    elif family == "ba":
        degrees = np.ones(n, dtype=np.float64)
        m = int(rng.choice([2, 3, 4]))
        for new_node in range(1, n):
            probs = degrees[:new_node] / degrees[:new_node].sum()
            parents = rng.choice(
                np.arange(new_node),
                size=min(m, new_node),
                replace=False,
                p=probs,
            )
            for parent in parents:
                add_unique_edge(edge_set, new_node, int(parent))
                degrees[new_node] += 1
                degrees[int(parent)] += 1
    elif family == "sbm":
        groups = rng.integers(0, 3, size=n)
        p_in = float(rng.uniform(3.5 / n, 7.0 / n))
        p_out = float(rng.uniform(0.35 / n, 1.7 / n))
        for a in range(n):
            for b in range(a + 1, n):
                p = p_in if groups[a] == groups[b] else p_out
                if rng.random() < p:
                    add_unique_edge(edge_set, a, b)
    else:
        raise ValueError(f"Unknown graph family: {family}")
    return sorted(edge_set), family


def compute_rwse(adj: np.ndarray, steps: int) -> np.ndarray:
    n = int(adj.shape[0])
    if steps <= 0:
        return np.zeros((n, 0), dtype=np.float32)
    deg = np.maximum(adj.sum(axis=1), 1.0).astype(np.float32)
    transition = adj / deg[:, None]
    try:
        import scipy.sparse as sp

        p_mat = sp.csr_matrix(transition)
        cur = sp.identity(n, dtype=np.float32, format="csr")
        feats = []
        for _ in range(steps):
            cur = cur @ p_mat
            feats.append(cur.diagonal().astype(np.float32))
        return np.stack(feats, axis=1)
    except Exception:
        cur = np.eye(n, dtype=np.float32)
        feats = []
        for _ in range(steps):
            cur = cur @ transition
            feats.append(np.diag(cur).astype(np.float32))
        return np.stack(feats, axis=1)


def triangle_counts(adj: np.ndarray) -> np.ndarray:
    a2 = adj @ adj
    tri = (a2 * adj).sum(axis=1) / 2.0
    return tri.astype(np.float32)


def clustering_coefficients(adj: np.ndarray, triangles: np.ndarray | None = None) -> np.ndarray:
    if triangles is None:
        triangles = triangle_counts(adj)
    degree = adj.sum(axis=1)
    denom = degree * np.maximum(degree - 1.0, 1.0) / 2.0
    return np.where(degree >= 2, triangles / np.maximum(denom, 1.0), 0.0).astype(np.float32)


def k_core_numbers(adj: np.ndarray) -> np.ndarray:
    n = int(adj.shape[0])
    neighbours = [set(np.flatnonzero(adj[i] > 0).astype(int).tolist()) for i in range(n)]
    remaining = set(range(n))
    degree = np.asarray([len(neighbours[i]) for i in range(n)], dtype=np.int64)
    core = np.zeros(n, dtype=np.float32)
    current_k = 0
    while remaining:
        node = min(remaining, key=lambda x: degree[x])
        current_k = max(current_k, int(degree[node]))
        core[node] = current_k
        remaining.remove(node)
        for nbr in list(neighbours[node]):
            if nbr in remaining:
                neighbours[nbr].discard(node)
                degree[nbr] = max(0, degree[nbr] - 1)
    return core


def closeness_scores(dist: np.ndarray) -> np.ndarray:
    reachable = dist > 0
    denom = np.where(reachable, dist, 0).sum(axis=1).astype(np.float32)
    count = reachable.sum(axis=1).astype(np.float32)
    return np.where(denom > 0, count / denom, 0.0).astype(np.float32)


def personalized_pagerank(
    adj: np.ndarray,
    source: int,
    alpha: float = 0.15,
    steps: int = 30,
) -> np.ndarray:
    n = int(adj.shape[0])
    deg = np.maximum(adj.sum(axis=1), 1.0).astype(np.float32)
    transition = adj / deg[:, None]
    restart = np.zeros(n, dtype=np.float32)
    restart[int(source)] = 1.0
    p = restart.copy()
    for _ in range(steps):
        p = alpha * restart + (1.0 - alpha) * (p @ transition)
    return p.astype(np.float32)


def bucket_by_quantile(values: np.ndarray, buckets: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if buckets <= 1:
        return np.zeros_like(values, dtype=np.int64)
    finite = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if float(finite.max() - finite.min()) < 1.0e-8:
        return np.zeros_like(finite, dtype=np.int64)
    qs = np.linspace(0.0, 1.0, buckets + 1)[1:-1]
    cuts = np.quantile(finite, qs)
    return np.searchsorted(cuts, finite, side="right").astype(np.int64)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    std = float(values.std())
    if std < 1.0e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(values.mean())) / std).astype(np.float32)


@dataclass
class StructuralStats:
    degree: np.ndarray
    triangles: np.ndarray
    clustering: np.ndarray
    two_hop: np.ndarray
    core: np.ndarray
    closeness: np.ndarray
    dist_a: np.ndarray
    dist_b: np.ndarray
    ppr_a: np.ndarray
    ppr_b: np.ndarray
    dist_delta: np.ndarray


def compute_structural_stats(
    adj: np.ndarray,
    anchor_a: int,
    anchor_b: int,
    dist: np.ndarray | None = None,
) -> StructuralStats:
    if dist is None:
        dist = all_pairs_distances(adj)
    degree = adj.sum(axis=1).astype(np.float32)
    triangles = triangle_counts(adj)
    clustering = clustering_coefficients(adj, triangles)
    two_hop = ((dist > 0) & (dist <= 2)).sum(axis=1).astype(np.float32)
    core = k_core_numbers(adj).astype(np.float32)
    closeness = closeness_scores(dist)
    dist_a = dist[anchor_a].astype(np.float32)
    dist_b = dist[anchor_b].astype(np.float32)
    ppr_a = personalized_pagerank(adj, anchor_a)
    ppr_b = personalized_pagerank(adj, anchor_b)
    dist_delta = (dist_a - dist_b).astype(np.float32)
    return StructuralStats(
        degree=degree,
        triangles=triangles,
        clustering=clustering,
        two_hop=two_hop,
        core=core,
        closeness=closeness,
        dist_a=dist_a,
        dist_b=dist_b,
        ppr_a=ppr_a,
        ppr_b=ppr_b,
        dist_delta=dist_delta,
    )


def structural_keys_from_stats(
    stats: StructuralStats,
    family: StructuralKeyFamily,
    buckets: int,
) -> np.ndarray:
    if family == "anchor_distance":
        cap = max(1, int(buckets) - 1)
        components = [
            np.minimum(np.maximum(stats.dist_a, 0), cap).astype(np.int64),
            np.minimum(np.maximum(stats.dist_b, 0), cap).astype(np.int64),
        ]
    elif family == "local":
        components = [
            bucket_by_quantile(stats.degree, buckets),
            bucket_by_quantile(stats.triangles, buckets),
            bucket_by_quantile(stats.two_hop, buckets),
        ]
    elif family == "diffusion":
        components = [
            bucket_by_quantile(stats.ppr_a, buckets),
            bucket_by_quantile(stats.ppr_b, buckets),
            bucket_by_quantile(stats.dist_delta, buckets),
        ]
    elif family == "global":
        components = [
            bucket_by_quantile(stats.core, buckets),
            bucket_by_quantile(stats.clustering, buckets),
            bucket_by_quantile(stats.closeness, buckets),
        ]
    else:
        raise ValueError(f"Unknown structural key family: {family}")
    return np.stack(components, axis=1).astype(np.int64)


def same_key(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(a, b))


def shared_components(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.asarray(a == b, dtype=np.int64).sum())


def sample_wrong_symbol_key(
    query_key: np.ndarray,
    vocab_size: int,
    rng: np.random.Generator,
    *,
    partial: bool,
) -> np.ndarray:
    key = rng.integers(0, vocab_size, size=query_key.shape, dtype=np.int64)
    if partial and query_key.size >= 2:
        keep = int(rng.integers(0, query_key.size))
        key[keep] = query_key[keep]
    while np.array_equal(key, query_key):
        component = int(rng.integers(0, query_key.size))
        value = int(rng.integers(0, vocab_size - 1))
        if value >= int(query_key[component]):
            value += 1
        key[component] = value
    return key


@dataclass
class TeacherGraph:
    role: np.ndarray
    symbol: np.ndarray
    value_id: np.ndarray
    adj: np.ndarray
    pe: np.ndarray
    spd: np.ndarray
    n: int
    candidate_nodes: np.ndarray
    value_nodes: np.ndarray
    query_node: int
    target_node: int
    target_candidate: int
    target_y: int
    candidate_values: np.ndarray
    candidate_types: list[str]
    task: str
    structural_family: str
    feature_set: str
    graph_family: str
    anchor_a: int
    anchor_b: int
    query_structural_key: np.ndarray
    candidate_structural_keys: np.ndarray
    query_symbolic_key: np.ndarray
    candidate_symbolic_keys: np.ndarray


@dataclass
class Batch:
    role: torch.Tensor
    symbol: torch.Tensor
    value_id: torch.Tensor
    mask: torch.Tensor
    adj_norm: torch.Tensor
    pe: torch.Tensor
    spd: torch.Tensor
    candidate_nodes: torch.Tensor
    candidate_mask: torch.Tensor
    value_nodes: torch.Tensor
    query_node: torch.Tensor
    target_node: torch.Tensor
    target_y: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            role=self.role.to(device),
            symbol=self.symbol.to(device),
            value_id=self.value_id.to(device),
            mask=self.mask.to(device),
            adj_norm=self.adj_norm.to(device),
            pe=self.pe.to(device),
            spd=self.spd.to(device),
            candidate_nodes=self.candidate_nodes.to(device),
            candidate_mask=self.candidate_mask.to(device),
            value_nodes=self.value_nodes.to(device),
            query_node=self.query_node.to(device),
            target_node=self.target_node.to(device),
            target_y=self.target_y.to(device),
        )


def feature_set_uses_spd(feature_set: FeatureSet) -> bool:
    return feature_set in {"spd_bias", "rwse_spd_bias"}


def make_student_pe(
    adj: np.ndarray,
    feature_set: FeatureSet,
    rwse_steps: int,
    full_stats: StructuralStats,
    oracle_key: np.ndarray,
    structural_buckets: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    n = int(adj.shape[0])
    if feature_set in {"degree", "stats", "rwse_stats"}:
        degree = adj.sum(axis=1, keepdims=True).astype(np.float32)
        features.append(degree / max(1.0, float(n - 1)))
    if feature_set in {"anchor_dist", "rwse_anchor_dist"}:
        denom = max(1.0, float(structural_buckets - 1))
        anchor_dist = np.stack(
            [
                np.clip(full_stats.dist_a, 0.0, denom) / denom,
                np.clip(full_stats.dist_b, 0.0, denom) / denom,
            ],
            axis=1,
        ).astype(np.float32)
        features.append(anchor_dist)
    if feature_set in {"rwse", "rwse_stats", "rwse_spd_bias", "rwse_anchor_dist"}:
        features.append(compute_rwse(adj, rwse_steps))
    if feature_set in {"stats", "rwse_stats"}:
        stats_matrix = np.stack(
            [
                zscore(full_stats.triangles),
                zscore(full_stats.clustering),
                zscore(full_stats.two_hop),
                zscore(full_stats.core),
                zscore(full_stats.closeness),
                zscore(full_stats.dist_a),
                zscore(full_stats.dist_b),
                zscore(full_stats.ppr_a),
                zscore(full_stats.ppr_b),
                zscore(full_stats.dist_delta),
            ],
            axis=1,
        ).astype(np.float32)
        features.append(stats_matrix)
    if feature_set == "oracle_key":
        features.append(oracle_key.astype(np.float32))
    if not features:
        return np.zeros((n, 0), dtype=np.float32)
    return np.concatenate(features, axis=1).astype(np.float32)


def build_teacher_graph(
    task: TeacherTask,
    structural_family: StructuralKeyFamily,
    feature_set: FeatureSet,
    rng: np.random.Generator,
    *,
    graph_min_nodes: int,
    graph_max_nodes: int,
    graph_family: GraphFamily,
    min_candidates: int,
    max_candidates: int,
    key_vocab_size: int,
    key_tuple_size: int,
    value_vocab_size: int,
    structural_buckets: int,
    rwse_steps: int,
    spd_cap: int,
    value_leaf_carries_value: bool,
    max_attempts: int = 256,
) -> TeacherGraph:
    for _ in range(max_attempts):
        backbone_n = int(rng.integers(graph_min_nodes, graph_max_nodes + 1))
        edges, resolved_family = graphworld_backbone_edges(backbone_n, graph_family, rng)
        backbone_adj = adjacency_from_edges(backbone_n, edges)
        backbone_dist = all_pairs_distances(backbone_adj)
        anchors = rng.choice(np.arange(backbone_n), size=2, replace=False)
        anchor_a, anchor_b = int(anchors[0]), int(anchors[1])
        stats = compute_structural_stats(backbone_adj, anchor_a, anchor_b, backbone_dist)
        structural_keys = structural_keys_from_stats(stats, structural_family, structural_buckets)

        eligible = np.asarray(
            [idx for idx in range(backbone_n) if idx not in {anchor_a, anchor_b}],
            dtype=np.int64,
        )
        num_candidates = int(rng.integers(min_candidates, max_candidates + 1))
        if eligible.size < num_candidates + 1:
            continue

        query_node = -1
        target_base = -1
        candidate_base: list[int] = []
        candidate_types: list[str] = []
        query_symbolic_key = np.zeros(key_tuple_size, dtype=np.int64)
        candidate_symbolic_keys = np.zeros((num_candidates, key_tuple_size), dtype=np.int64)

        if task == "symbolic":
            chosen = rng.choice(eligible, size=num_candidates + 1, replace=False)
            query_node = int(chosen[0])
            candidate_base = [int(x) for x in chosen[1:]]
            target_candidate = int(rng.integers(0, num_candidates))
            target_base = candidate_base[target_candidate]
            query_symbolic_key = rng.integers(
                0,
                key_vocab_size,
                size=key_tuple_size,
                dtype=np.int64,
            )
            for idx in range(num_candidates):
                if idx == target_candidate:
                    candidate_symbolic_keys[idx] = query_symbolic_key
                    candidate_types.append("target")
                else:
                    partial = bool(rng.random() < 0.5)
                    candidate_symbolic_keys[idx] = sample_wrong_symbol_key(
                        query_symbolic_key,
                        key_vocab_size,
                        rng,
                        partial=partial,
                    )
                    candidate_types.append("symbolic_partial" if partial else "neither")

        else:
            groups: dict[tuple[int, ...], list[int]] = {}
            for node in eligible.tolist():
                groups.setdefault(tuple(structural_keys[node].tolist()), []).append(int(node))
            min_group = 3 if task == "mixed" else 2
            usable_groups = [nodes for nodes in groups.values() if len(nodes) >= min_group]
            if not usable_groups:
                continue
            group = list(usable_groups[int(rng.integers(0, len(usable_groups)))])
            rng.shuffle(group)
            query_node = int(group[0])
            target_base = int(group[1])
            query_key = structural_keys[query_node]

            same_struct = [
                int(node)
                for node in eligible.tolist()
                if node not in {query_node, target_base}
                and same_key(structural_keys[node], query_key)
            ]
            wrong_struct = [
                int(node)
                for node in eligible.tolist()
                if node not in {query_node, target_base}
                and not same_key(structural_keys[node], query_key)
            ]
            partial_struct = [
                node
                for node in wrong_struct
                if shared_components(structural_keys[node], query_key) >= 1
            ]
            candidate_base = [target_base]
            candidate_types = ["target"]

            if task == "mixed":
                if not same_struct or len(wrong_struct) < 2:
                    continue
                query_symbolic_key = rng.integers(
                    0,
                    key_vocab_size,
                    size=key_tuple_size,
                    dtype=np.int64,
                )
                structural_only = int(same_struct[int(rng.integers(0, len(same_struct)))])
                symbolic_pool = [node for node in wrong_struct if node != structural_only]
                if not symbolic_pool:
                    continue
                symbolic_only = int(symbolic_pool[int(rng.integers(0, len(symbolic_pool)))])
                candidate_base.extend([symbolic_only, structural_only])
                candidate_types.extend(["symbolic_only", "structural_only"])
                remaining_pool = [
                    node
                    for node in wrong_struct
                    if node not in {symbolic_only, structural_only}
                ]
                rng.shuffle(remaining_pool)
                while len(candidate_base) < num_candidates and remaining_pool:
                    candidate_base.append(int(remaining_pool.pop()))
                    candidate_types.append("neither")
                if len(candidate_base) < num_candidates:
                    continue
                rng_order = rng.permutation(num_candidates)
                candidate_base = [candidate_base[int(i)] for i in rng_order]
                candidate_types = [candidate_types[int(i)] for i in rng_order]
                target_candidate = int(candidate_types.index("target"))
                for idx, ctype in enumerate(candidate_types):
                    if ctype in {"target", "symbolic_only"}:
                        candidate_symbolic_keys[idx] = query_symbolic_key
                    else:
                        candidate_symbolic_keys[idx] = sample_wrong_symbol_key(
                            query_symbolic_key,
                            key_vocab_size,
                            rng,
                            partial=ctype == "structural_only",
                        )
            else:
                if partial_struct:
                    rng.shuffle(partial_struct)
                    for node in partial_struct[: min(2, len(partial_struct))]:
                        candidate_base.append(int(node))
                        candidate_types.append("structural_partial")
                wrong_pool = [
                    node
                    for node in wrong_struct
                    if node not in set(candidate_base)
                ]
                rng.shuffle(wrong_pool)
                while len(candidate_base) < num_candidates and wrong_pool:
                    candidate_base.append(int(wrong_pool.pop()))
                    candidate_types.append("neither")
                if len(candidate_base) < num_candidates:
                    continue
                rng_order = rng.permutation(num_candidates)
                candidate_base = [candidate_base[int(i)] for i in rng_order]
                candidate_types = [candidate_types[int(i)] for i in rng_order]
                target_candidate = int(candidate_types.index("target"))

        candidate_base_arr = np.asarray(candidate_base, dtype=np.int64)
        if len(set(candidate_base_arr.tolist())) != len(candidate_base_arr):
            continue

        values = rng.choice(
            np.arange(value_vocab_size),
            size=num_candidates,
            replace=num_candidates > value_vocab_size,
        ).astype(np.int64)
        target_candidate = int(candidate_types.index("target"))
        target_value = int(values[target_candidate])

        role: list[int] = [BACKBONE_ROLE for _ in range(backbone_n)]
        symbol: list[int] = [0 for _ in range(backbone_n)]
        value_id: list[int] = [0 for _ in range(backbone_n)]
        role[anchor_a] = ANCHOR_A_ROLE
        role[anchor_b] = ANCHOR_B_ROLE
        role[query_node] = QUERY_ROLE
        for node in candidate_base_arr:
            role[int(node)] = CANDIDATE_ROLE
        for idx, node in enumerate(candidate_base_arr):
            value_id[int(node)] = int(values[idx]) + 1

        if task in {"symbolic", "mixed"}:
            symbol[query_node] = key_symbol(int(query_symbolic_key[0]))
            for idx, node in enumerate(candidate_base_arr):
                symbol[int(node)] = key_symbol(int(candidate_symbolic_keys[idx, 0]))

        full_edges = list(edges)
        value_nodes: list[int] = []

        def add_node(role_id: int, symbol_id: int = 0) -> int:
            node_id = len(role)
            role.append(role_id)
            symbol.append(symbol_id)
            value_id.append(0)
            return node_id

        if task in {"symbolic", "mixed"}:
            for component in query_symbolic_key[1:]:
                key_node = add_node(KEY_ROLE, key_symbol(int(component)))
                full_edges.append((query_node, key_node))

        for idx, node in enumerate(candidate_base_arr):
            leaf_symbol = (
                value_symbol(int(values[idx]), key_vocab_size)
                if value_leaf_carries_value
                else 0
            )
            val_node = add_node(VALUE_ROLE, leaf_symbol)
            if value_leaf_carries_value:
                value_id[val_node] = int(values[idx]) + 1
            value_nodes.append(val_node)
            full_edges.append((int(node), val_node))
            if task in {"symbolic", "mixed"}:
                for component in candidate_symbolic_keys[idx, 1:]:
                    key_node = add_node(KEY_ROLE, key_symbol(int(component)))
                    full_edges.append((int(node), key_node))

        full_adj = adjacency_from_edges(len(role), full_edges)
        full_dist = all_pairs_distances(full_adj)
        full_anchor_a = anchor_a
        full_anchor_b = anchor_b
        full_stats = compute_structural_stats(full_adj, full_anchor_a, full_anchor_b, full_dist)
        oracle_key = np.zeros((len(role), structural_keys.shape[1]), dtype=np.float32)
        oracle_key[:backbone_n] = structural_keys.astype(np.float32)
        pe = make_student_pe(
            full_adj,
            feature_set,
            rwse_steps,
            full_stats,
            oracle_key,
            structural_buckets,
        )
        spd = (
            np.clip(full_dist, 0, spd_cap).astype(np.int64)
            if feature_set_uses_spd(feature_set)
            else np.zeros_like(full_dist, dtype=np.int64)
        )

        return TeacherGraph(
            role=np.asarray(role, dtype=np.int64),
            symbol=np.asarray(symbol, dtype=np.int64),
            value_id=np.asarray(value_id, dtype=np.int64),
            adj=full_adj,
            pe=pe,
            spd=spd,
            n=len(role),
            candidate_nodes=candidate_base_arr.astype(np.int64),
            value_nodes=np.asarray(value_nodes, dtype=np.int64),
            query_node=query_node,
            target_node=int(candidate_base_arr[target_candidate]),
            target_candidate=target_candidate,
            target_y=target_value,
            candidate_values=values,
            candidate_types=candidate_types,
            task=task,
            structural_family=structural_family,
            feature_set=feature_set,
            graph_family=resolved_family,
            anchor_a=anchor_a,
            anchor_b=anchor_b,
            query_structural_key=structural_keys[query_node].copy(),
            candidate_structural_keys=structural_keys[candidate_base_arr].copy(),
            query_symbolic_key=query_symbolic_key.copy(),
            candidate_symbolic_keys=candidate_symbolic_keys.copy(),
        )

    raise RuntimeError(
        f"Could not sample a valid {task}/{structural_family} teacher graph "
        f"after {max_attempts} attempts"
    )


def collate_examples(examples: list[TeacherGraph], spd_cap: int) -> Batch:
    batch_size = len(examples)
    max_n = max(ex.n for ex in examples)
    max_candidates = max(len(ex.candidate_nodes) for ex in examples)
    pe_dim = examples[0].pe.shape[1]

    role = np.full((batch_size, max_n), PAD_ROLE, dtype=np.int64)
    symbol = np.zeros((batch_size, max_n), dtype=np.int64)
    value_id = np.zeros((batch_size, max_n), dtype=np.int64)
    mask = np.zeros((batch_size, max_n), dtype=np.bool_)
    adj_norm = np.zeros((batch_size, max_n, max_n), dtype=np.float32)
    pe = np.zeros((batch_size, max_n, pe_dim), dtype=np.float32)
    spd = np.full((batch_size, max_n, max_n), spd_cap + 1, dtype=np.int64)
    candidate_nodes = np.zeros((batch_size, max_candidates), dtype=np.int64)
    candidate_mask = np.zeros((batch_size, max_candidates), dtype=np.bool_)
    value_nodes = np.zeros((batch_size, max_candidates), dtype=np.int64)
    query_node = np.zeros(batch_size, dtype=np.int64)
    target_node = np.zeros(batch_size, dtype=np.int64)
    target_y = np.zeros(batch_size, dtype=np.int64)

    for idx, ex in enumerate(examples):
        n = ex.n
        degree = np.maximum(ex.adj.sum(axis=1, keepdims=True), 1.0)
        role[idx, :n] = ex.role
        symbol[idx, :n] = ex.symbol
        value_id[idx, :n] = ex.value_id
        mask[idx, :n] = True
        adj_norm[idx, :n, :n] = ex.adj / degree
        pe[idx, :n, :] = ex.pe
        spd[idx, :n, :n] = np.clip(ex.spd, 0, spd_cap)
        c = len(ex.candidate_nodes)
        candidate_nodes[idx, :c] = ex.candidate_nodes
        candidate_mask[idx, :c] = True
        value_nodes[idx, :c] = ex.value_nodes
        query_node[idx] = ex.query_node
        target_node[idx] = ex.target_node
        target_y[idx] = ex.target_y

    return Batch(
        role=torch.from_numpy(role),
        symbol=torch.from_numpy(symbol),
        value_id=torch.from_numpy(value_id),
        mask=torch.from_numpy(mask),
        adj_norm=torch.from_numpy(adj_norm),
        pe=torch.from_numpy(pe),
        spd=torch.from_numpy(spd),
        candidate_nodes=torch.from_numpy(candidate_nodes),
        candidate_mask=torch.from_numpy(candidate_mask),
        value_nodes=torch.from_numpy(value_nodes),
        query_node=torch.from_numpy(query_node),
        target_node=torch.from_numpy(target_node),
        target_y=torch.from_numpy(target_y),
    )


def iter_static_batches(
    examples: list[TeacherGraph],
    batch_size: int,
    spd_cap: int,
    device: torch.device,
) -> Iterable[tuple[list[TeacherGraph], Batch]]:
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        yield chunk, collate_examples(chunk, spd_cap).to(device)


def generate_examples(
    count: int,
    seed: int,
    task: TeacherTask,
    structural_family: StructuralKeyFamily,
    feature_set: FeatureSet,
    args: argparse.Namespace,
    *,
    ood: bool = False,
) -> list[TeacherGraph]:
    rng = np.random.default_rng(seed)
    graph_min = args.ood_graph_min_nodes if ood else args.graph_min_nodes
    graph_max = args.ood_graph_max_nodes if ood else args.graph_max_nodes
    cand_min = args.ood_min_candidates if ood else args.min_candidates
    cand_max = args.ood_max_candidates if ood else args.max_candidates
    return [
        build_teacher_graph(
            task=task,
            structural_family=structural_family,
            feature_set=feature_set,
            rng=rng,
            graph_min_nodes=graph_min,
            graph_max_nodes=graph_max,
            graph_family=args.graph_family,
            min_candidates=cand_min,
            max_candidates=cand_max,
            key_vocab_size=args.key_vocab_size,
            key_tuple_size=args.key_tuple_size,
            value_vocab_size=args.value_vocab_size,
            structural_buckets=args.structural_buckets,
            rwse_steps=args.rwse_steps,
            spd_cap=args.spd_cap,
            value_leaf_carries_value=args.value_leaf_carries_value,
        )
        for _ in range(count)
    ]


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseGINEBranch(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_emb = nn.Parameter(torch.zeros(dim))
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = MLP(dim, dim * 2, dim, dropout)

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        msg = F.relu(h + self.edge_emb)
        agg = torch.bmm(adj_norm, msg)
        out = self.mlp((1.0 + self.eps) * h + agg)
        return out * mask.unsqueeze(-1).to(out.dtype)


class BiasedMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        use_spd_bias: bool,
        spd_cap: int,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.spd_cap = spd_cap
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.spd_bias = nn.Embedding(spd_cap + 2, num_heads) if use_spd_bias else None

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        spd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, max_n, _ = h.shape
        qkv = self.qkv(h).view(batch_size, max_n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.spd_bias is not None:
            bias = self.spd_bias(spd.clamp(0, self.spd_cap + 1)).permute(0, 3, 1, 2)
            logits = logits + bias
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        key_mask = mask[:, None, None, :]
        masked_logits = logits.masked_fill(~key_mask, -1.0e9)
        attn = torch.softmax(masked_logits, dim=-1).masked_fill(~pair_mask, 0.0)
        out = torch.matmul(self.dropout(attn), v)
        out = out.transpose(1, 2).contiguous().view(batch_size, max_n, self.dim)
        out = self.out(out) * mask.unsqueeze(-1).to(h.dtype)
        metric_logits = logits.masked_fill(~pair_mask, 0.0)
        return out, metric_logits, attn


class GraphGPSLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        attn_dropout: float,
        use_spd_bias: bool,
        spd_cap: int,
    ) -> None:
        super().__init__()
        self.local = DenseGINEBranch(dim, dropout=dropout)
        self.attn = BiasedMultiheadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            use_spd_bias=use_spd_bias,
            spd_cap=spd_cap,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        batch: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local = self.local(h, batch.adj_norm, batch.mask)
        attn, logits, attn_weights = self.attn(h, batch.mask, batch.spd)
        h = self.norm1(h + self.dropout(local + attn))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        h = h * batch.mask.unsqueeze(-1).to(h.dtype)
        return h, logits, attn_weights


class GraphGPSQueryModel(nn.Module):
    def __init__(
        self,
        depth: int,
        hidden_dim: int,
        num_heads: int,
        symbol_vocab: int,
        pe_dim: int,
        value_vocab_size: int,
        dropout: float,
        attn_dropout: float,
        use_spd_bias: bool,
        spd_cap: int,
        value_decoder: str,
        fixed_value_embeddings: bool,
    ) -> None:
        super().__init__()
        if value_decoder not in {"tied", "mlp"}:
            raise ValueError(f"Unknown value decoder: {value_decoder}")
        self.role_emb = nn.Embedding(NUM_ROLES, hidden_dim)
        self.symbol_emb = nn.Embedding(symbol_vocab, hidden_dim)
        self.value_emb = nn.Embedding(value_vocab_size + 1, hidden_dim)
        self.value_vocab_size = value_vocab_size
        self.value_decoder = value_decoder
        if fixed_value_embeddings:
            self._init_fixed_value_embeddings(hidden_dim, value_vocab_size)
        self.pe_proj = nn.Linear(pe_dim, hidden_dim, bias=False) if pe_dim > 0 else None
        self.layers = nn.ModuleList(
            [
                GraphGPSLayer(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    use_spd_bias=use_spd_bias,
                    spd_cap=spd_cap,
                )
                for _ in range(depth)
            ]
        )
        if value_decoder == "tied":
            self.head_norm = nn.LayerNorm(hidden_dim)
            self.head_proj = nn.Linear(hidden_dim, hidden_dim)
            self.head_bias = nn.Parameter(torch.zeros(value_vocab_size))
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, value_vocab_size),
            )

    def _init_fixed_value_embeddings(self, hidden_dim: int, value_vocab_size: int) -> None:
        with torch.no_grad():
            self.value_emb.weight.zero_()
            eye_dim = min(hidden_dim, value_vocab_size)
            self.value_emb.weight[1 : eye_dim + 1, :eye_dim] = (
                torch.eye(eye_dim) * math.sqrt(float(hidden_dim))
            )
            if value_vocab_size > eye_dim:
                extra = torch.randn(value_vocab_size - eye_dim, hidden_dim)
                extra = F.normalize(extra, dim=-1) * math.sqrt(float(hidden_dim))
                self.value_emb.weight[eye_dim + 1 :] = extra
        self.value_emb.weight.requires_grad_(False)

    def decode_value(self, query_h: torch.Tensor) -> torch.Tensor:
        if self.value_decoder == "mlp":
            return self.head(query_h)
        z = self.head_proj(self.head_norm(query_h))
        value_codes = self.value_emb.weight[1 : self.value_vocab_size + 1]
        return (z @ value_codes.t()) / math.sqrt(float(z.size(-1))) + self.head_bias

    def forward(
        self,
        batch: Batch,
        collect_attention: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        h = (
            self.role_emb(batch.role)
            + self.symbol_emb(batch.symbol)
            + self.value_emb(batch.value_id)
        )
        if self.pe_proj is not None:
            h = h + self.pe_proj(batch.pe)
        h = h * batch.mask.unsqueeze(-1).to(h.dtype)

        layers = []
        for layer in self.layers:
            h, logits, attn = layer(h, batch)
            if collect_attention:
                layers.append(
                    {
                        "logits": logits if self.training else logits.detach(),
                        "attn": attn if self.training else attn.detach(),
                        "node_mask": batch.mask.detach(),
                    }
                )
        batch_idx = torch.arange(h.size(0), device=h.device)
        query_h = h[batch_idx, batch.query_node]
        return self.decode_value(query_h), layers


@dataclass
class EvalStats:
    loss: float
    acc: float
    mrr_value_proxy: float
    n: int


def batch_loss(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    return F.cross_entropy(logits, batch.target_y)


@torch.no_grad()
def evaluate(
    model: GraphGPSQueryModel,
    examples: list[TeacherGraph],
    batch_size: int,
    spd_cap: int,
    device: torch.device,
) -> tuple[EvalStats, dict[str, int]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    value_proxy_rr = []
    error_types: dict[str, int] = {}
    for chunk, batch in iter_static_batches(examples, batch_size, spd_cap, device):
        logits, _ = model(batch, collect_attention=False)
        loss = batch_loss(logits, batch)
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        labels = batch.target_y.detach().cpu().numpy()
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        total_loss += float(loss.item()) * len(chunk)
        correct += int((preds == labels).sum())
        total += len(chunk)
        for pred, label, prob, ex in zip(preds, labels, probs, chunk):
            if int(pred) == int(label):
                error_types["correct"] = error_types.get("correct", 0) + 1
            else:
                matched = [
                    idx
                    for idx, value in enumerate(ex.candidate_values.tolist())
                    if int(value) == int(pred)
                ]
                if matched:
                    ctype = ex.candidate_types[int(matched[0])]
                else:
                    ctype = "non_candidate_value"
                error_types[ctype] = error_types.get(ctype, 0) + 1
            rank = int((prob > prob[int(label)]).sum()) + 1
            value_proxy_rr.append(1.0 / rank)
    return (
        EvalStats(
            loss=total_loss / max(1, total),
            acc=correct / max(1, total),
            mrr_value_proxy=float(np.mean(value_proxy_rr)) if value_proxy_rr else 0.0,
            n=total,
        ),
        error_types,
    )


def pair_mask_from_layer(layer: dict[str, torch.Tensor]) -> torch.Tensor:
    node_mask = layer["node_mask"]
    return node_mask[:, None, :, None] & node_mask[:, None, None, :]


def gather_dense_node_axis(t: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    batch_size, max_n = perm_pos.shape
    extra = t.dim() - 2
    idx = perm_pos.to(t.device).view(batch_size, max_n, *([1] * extra)).expand_as(t)
    return torch.gather(t, dim=1, index=idx)


def gather_dense_pair_axes(t: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    batch_size, max_n = perm_pos.shape
    extra = t.dim() - 3
    idx_r = perm_pos.to(t.device).view(batch_size, max_n, 1, *([1] * extra)).expand_as(t)
    t_rows = torch.gather(t, dim=1, index=idx_r)
    idx_c = perm_pos.to(t.device).view(batch_size, 1, max_n, *([1] * extra)).expand_as(t_rows)
    return torch.gather(t_rows, dim=2, index=idx_c)


def inverse_perm_pos(perm_pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    inv = torch.arange(perm_pos.size(1), device=perm_pos.device).repeat(perm_pos.size(0), 1)
    for graph_idx in range(perm_pos.size(0)):
        n = int(mask[graph_idx].sum().item())
        perm = perm_pos[graph_idx, :n]
        inv_graph = torch.empty_like(perm)
        inv_graph[perm] = torch.arange(n, device=perm_pos.device)
        inv[graph_idx, :n] = inv_graph
    return inv


def transform_pair_reference(z: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    batch_size, heads, max_n, _ = z.shape
    idx = perm_pos.to(z.device)
    row_idx = idx[:, None, :, None].expand(batch_size, heads, max_n, max_n)
    z_rows = torch.gather(z, dim=2, index=row_idx)
    col_idx = idx[:, None, None, :].expand(batch_size, heads, max_n, max_n)
    return torch.gather(z_rows, dim=3, index=col_idx)


def transform_pair_mask(mask: torch.Tensor, perm_pos: torch.Tensor, heads: int) -> torch.Tensor:
    m = mask.to(dtype=torch.bool)
    if m.size(1) == 1 and heads != 1:
        m = m.expand(-1, heads, -1, -1)
    return transform_pair_reference(m, perm_pos).to(dtype=torch.bool)


def comparison_masks(
    clean_layer: dict[str, torch.Tensor],
    variant_layer: dict[str, torch.Tensor],
    perm_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_var = variant_layer["logits"]
    heads = int(z_var.size(1))
    m_clean = pair_mask_from_layer(clean_layer).to(device=z_var.device, dtype=torch.bool)
    m_var = pair_mask_from_layer(variant_layer).to(device=z_var.device, dtype=torch.bool)
    if m_clean.size(1) == 1 and heads != 1:
        m_clean = m_clean.expand(-1, heads, -1, -1)
    if m_var.size(1) == 1 and heads != 1:
        m_var = m_var.expand(-1, heads, -1, -1)
    m_clean_t = transform_pair_mask(m_clean, perm_pos, heads).to(device=z_var.device)
    return (m_clean & m_var), (m_clean_t & m_var)


def row_center_logits(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=z.device, dtype=torch.bool)
    if m.size(1) == 1 and z.size(1) != 1:
        m = m.expand(-1, z.size(1), -1, -1)
    z0 = torch.where(m, torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    denom = m.sum(dim=-1, keepdim=True).clamp_min(1).to(z.dtype)
    mean = z0.sum(dim=-1, keepdim=True) / denom
    return torch.where(m, z0 - mean, torch.zeros_like(z0))


def cosine_by_head_logits(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=u.device, dtype=torch.bool)
    if m.size(1) == 1 and u.size(1) != 1:
        m = m.expand(-1, u.size(1), -1, -1)
    u0 = row_center_logits(u, m)
    v0 = row_center_logits(v, m)
    dims = (0, 2, 3)
    num = (u0 * v0).sum(dim=dims)
    den = (
        torch.sqrt((u0 * u0).sum(dim=dims).clamp_min(1.0e-12))
        * torch.sqrt((v0 * v0).sum(dim=dims).clamp_min(1.0e-12))
    )
    return (num / den.clamp_min(1.0e-12)).detach().cpu()


def cosine_to_score01(cos: torch.Tensor) -> torch.Tensor:
    return torch.clamp(0.5 * (cos + 1.0), 0.0, 1.0)


def make_node_symbol_permutation(
    batch: Batch,
    generator: torch.Generator,
) -> tuple[Batch, torch.Tensor]:
    batch_size, max_n = batch.symbol.shape
    perm_pos = torch.zeros((batch_size, max_n), dtype=torch.long, device=batch.symbol.device)
    symbol_perm = batch.symbol.clone()
    value_perm = batch.value_id.clone()
    for graph_idx in range(batch_size):
        n = int(batch.mask[graph_idx].sum().item())
        perm = torch.randperm(n, generator=generator).to(batch.symbol.device)
        perm_pos[graph_idx, :n] = perm
        if n < max_n:
            perm_pos[graph_idx, n:] = torch.arange(n, max_n, device=batch.symbol.device)
        symbol_perm[graph_idx, :n] = batch.symbol[graph_idx, perm]
        value_perm[graph_idx, :n] = batch.value_id[graph_idx, perm]
    return replace(batch, symbol=symbol_perm, value_id=value_perm), perm_pos


def make_pe_permutation(batch: Batch, perm_pos: torch.Tensor) -> Batch | None:
    if batch.pe.size(-1) == 0:
        return None
    return replace(batch, pe=gather_dense_node_axis(batch.pe, perm_pos))


def make_x_and_pe_permutation(batch: Batch, perm_pos: torch.Tensor) -> Batch | None:
    if batch.pe.size(-1) == 0:
        return None
    symbol_perm = gather_dense_node_axis(batch.symbol.unsqueeze(-1), perm_pos).squeeze(-1)
    value_perm = gather_dense_node_axis(batch.value_id.unsqueeze(-1), perm_pos).squeeze(-1)
    pe_perm = gather_dense_node_axis(batch.pe, perm_pos)
    return replace(batch, symbol=symbol_perm, value_id=value_perm, pe=pe_perm)


def make_relabel_variant(batch: Batch, perm_pos: torch.Tensor) -> Batch:
    inv = inverse_perm_pos(perm_pos, batch.mask)
    gather_index = lambda values: torch.gather(inv, dim=1, index=values.clamp_min(0))
    return replace(
        batch,
        role=gather_dense_node_axis(batch.role.unsqueeze(-1), perm_pos).squeeze(-1),
        symbol=gather_dense_node_axis(batch.symbol.unsqueeze(-1), perm_pos).squeeze(-1),
        value_id=gather_dense_node_axis(batch.value_id.unsqueeze(-1), perm_pos).squeeze(-1),
        pe=gather_dense_node_axis(batch.pe, perm_pos),
        adj_norm=gather_dense_pair_axes(batch.adj_norm, perm_pos),
        spd=gather_dense_pair_axes(batch.spd, perm_pos),
        candidate_nodes=gather_index(batch.candidate_nodes),
        value_nodes=gather_index(batch.value_nodes),
        query_node=torch.gather(inv, dim=1, index=batch.query_node[:, None]).squeeze(1),
        target_node=torch.gather(inv, dim=1, index=batch.target_node[:, None]).squeeze(1),
    )


def head_entropy_rows(
    layers: list[dict[str, torch.Tensor]],
    run_meta: dict[str, str | int],
    batch_idx: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for layer_idx, layer in enumerate(layers):
        z = layer["logits"]
        heads = int(z.size(1))
        mask = pair_mask_from_layer(layer).to(device=z.device, dtype=torch.bool)
        if mask.size(1) == 1 and heads != 1:
            mask = mask.expand(-1, heads, -1, -1)
        z0 = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z_masked = torch.where(mask, z0, torch.full_like(z0, -1.0e9))
        valid_row = mask.any(dim=-1, keepdim=True)
        z_masked = torch.where(valid_row, z_masked, torch.zeros_like(z_masked))
        attn = torch.softmax(z_masked, dim=-1)
        attn = torch.where(mask & valid_row, attn, torch.zeros_like(attn))
        key_count = mask.sum(dim=-1).to(attn.dtype)
        valid = key_count > 1
        ent_raw = -(
            torch.where(attn > 0, attn * torch.log(attn.clamp_min(1.0e-12)), 0.0)
        ).sum(dim=-1)
        ent_norm = torch.where(
            valid,
            ent_raw / torch.log(key_count.clamp_min(2.0)),
            torch.zeros_like(ent_raw),
        )
        valid_f = valid.to(attn.dtype)
        denom = valid_f.sum(dim=(0, 2)).clamp_min(1.0)
        entropy = (ent_norm * valid_f).sum(dim=(0, 2)) / denom
        for head in range(heads):
            rows.append(
                {
                    **run_meta,
                    "batch": batch_idx,
                    "perm": -1,
                    "layer": layer_idx,
                    "head": head,
                    "metric": M_ENTROPY,
                    "score": float(entropy[head].detach().cpu()),
                }
            )
    return rows


def transport_plane_rows(
    clean_layers: list[dict[str, torch.Tensor]],
    variant_layers: list[dict[str, torch.Tensor]],
    perm_pos: torch.Tensor,
    run_meta: dict[str, str | int],
    batch_idx: int,
    perm_idx: int,
    inv_metric: str,
    equi_metric: str,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for layer_idx, (clean, variant) in enumerate(zip(clean_layers, variant_layers)):
        z_clean = clean["logits"]
        z_var = variant["logits"]
        z_ref = transform_pair_reference(z_clean, perm_pos)
        stable_mask, follow_mask = comparison_masks(clean, variant, perm_pos)
        inv_cos = cosine_by_head_logits(z_var, z_clean, stable_mask)
        equi_cos = cosine_by_head_logits(z_var, z_ref, follow_mask)
        inv = cosine_to_score01(inv_cos)
        equi = cosine_to_score01(equi_cos)
        for head in range(int(inv.numel())):
            common = {
                **run_meta,
                "batch": batch_idx,
                "perm": perm_idx,
                "layer": layer_idx,
                "head": head,
            }
            rows.append(
                {
                    **common,
                    "metric": inv_metric,
                    "score": float(inv[head]),
                    "raw_cos": float(inv_cos[head]),
                }
            )
            rows.append(
                {
                    **common,
                    "metric": equi_metric,
                    "score": float(equi[head]),
                    "raw_cos": float(equi_cos[head]),
                }
            )
    return rows


def relabel_equivariance_rows(
    clean_layers: list[dict[str, torch.Tensor]],
    variant_layers: list[dict[str, torch.Tensor]],
    perm_pos: torch.Tensor,
    run_meta: dict[str, str | int],
    batch_idx: int,
    perm_idx: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for layer_idx, (clean, variant) in enumerate(zip(clean_layers, variant_layers)):
        z_clean = clean["logits"]
        z_var = variant["logits"]
        heads = int(z_var.size(1))
        z_ref = transform_pair_reference(z_clean, perm_pos)
        m_clean = pair_mask_from_layer(clean).to(device=z_var.device, dtype=torch.bool)
        m_var = pair_mask_from_layer(variant).to(device=z_var.device, dtype=torch.bool)
        m_ref = transform_pair_mask(m_clean, perm_pos, heads).to(device=z_var.device)
        if m_var.size(1) == 1 and heads != 1:
            m_var = m_var.expand(-1, heads, -1, -1)
        cos = cosine_by_head_logits(z_var, z_ref, m_ref & m_var)
        score = cosine_to_score01(cos)
        for head in range(int(score.numel())):
            rows.append(
                {
                    **run_meta,
                    "batch": batch_idx,
                    "perm": perm_idx,
                    "layer": layer_idx,
                    "head": head,
                    "metric": M_RELABEL_EQUIVARIANT,
                    "score": float(score[head]),
                    "raw_cos": float(cos[head]),
                }
            )
    return rows


def entanglement_rows(
    clean_layers: list[dict[str, torch.Tensor]],
    x_layers: list[dict[str, torch.Tensor]],
    pe_layers: list[dict[str, torch.Tensor]],
    both_layers: list[dict[str, torch.Tensor]],
    run_meta: dict[str, str | int],
    batch_idx: int,
    perm_idx: int,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for layer_idx, (clean, var_x, var_pe, var_both) in enumerate(
        zip(clean_layers, x_layers, pe_layers, both_layers)
    ):
        z_clean = clean["logits"]
        delta_x = var_x["logits"] - z_clean
        delta_pe = var_pe["logits"] - z_clean
        delta_both = var_both["logits"] - z_clean
        additive_pred = delta_x + delta_pe
        mask = pair_mask_from_layer(clean)
        cos = cosine_by_head_logits(delta_both, additive_pred, mask)
        score = torch.clamp(0.5 * (1.0 - cos), 0.0, 1.0)
        for head in range(int(score.numel())):
            rows.append(
                {
                    **run_meta,
                    "batch": batch_idx,
                    "perm": perm_idx,
                    "layer": layer_idx,
                    "head": head,
                    "metric": M_ENTANGLEMENT,
                    "score": float(score[head]),
                    "raw_cos": float(cos[head]),
                }
            )
    return rows


@torch.no_grad()
def compute_permutation_metrics(
    model: GraphGPSQueryModel,
    examples: list[TeacherGraph],
    run_meta: dict[str, str | int],
    batch_size: int,
    spd_cap: int,
    metric_graphs: int,
    num_perms: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    model.eval()
    selected = examples[: min(metric_graphs, len(examples))]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    rows: list[dict[str, float | int | str]] = []
    for batch_idx, (_, batch) in enumerate(
        iter_static_batches(selected, batch_size, spd_cap, device)
    ):
        _, clean_layers = model(batch, collect_attention=True)
        rows.extend(head_entropy_rows(clean_layers, run_meta, batch_idx))
        for perm_idx in range(num_perms):
            x_batch, perm_pos = make_node_symbol_permutation(batch, generator)
            _, x_layers = model(x_batch, collect_attention=True)
            rows.extend(
                transport_plane_rows(
                    clean_layers,
                    x_layers,
                    perm_pos,
                    run_meta,
                    batch_idx,
                    perm_idx,
                    inv_metric=M_POSITIONAL,
                    equi_metric=M_SYMBOLIC,
                )
            )

            relabel_batch = make_relabel_variant(batch, perm_pos)
            _, relabel_layers = model(relabel_batch, collect_attention=True)
            rows.extend(
                relabel_equivariance_rows(
                    clean_layers,
                    relabel_layers,
                    perm_pos,
                    run_meta,
                    batch_idx,
                    perm_idx,
                )
            )

            pe_batch = make_pe_permutation(batch, perm_pos)
            if pe_batch is None:
                continue
            _, pe_layers = model(pe_batch, collect_attention=True)
            rows.extend(
                transport_plane_rows(
                    clean_layers,
                    pe_layers,
                    perm_pos,
                    run_meta,
                    batch_idx,
                    perm_idx,
                    inv_metric=M_PE_INVARIANT,
                    equi_metric=M_PE_EQUIVARIANT,
                )
            )
            both_batch = make_x_and_pe_permutation(batch, perm_pos)
            if both_batch is None:
                continue
            _, both_layers = model(both_batch, collect_attention=True)
            rows.extend(
                entanglement_rows(
                    clean_layers,
                    x_layers,
                    pe_layers,
                    both_layers,
                    run_meta,
                    batch_idx,
                    perm_idx,
                )
            )
    return rows


@torch.no_grad()
def compute_target_attention_metrics(
    model: GraphGPSQueryModel,
    examples: list[TeacherGraph],
    run_meta: dict[str, str | int],
    batch_size: int,
    spd_cap: int,
    metric_graphs: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    model.eval()
    selected = examples[: min(metric_graphs, len(examples))]
    buckets: dict[tuple[int, int, str], list[float]] = {}
    candidate_types = (
        "target",
        "symbolic_only",
        "structural_only",
        "structural_partial",
        "symbolic_partial",
        "neither",
    )
    role_metrics = (
        (CANDIDATE_ROLE, "query_to_role_candidate"),
        (VALUE_ROLE, "query_to_role_value"),
        (KEY_ROLE, "query_to_role_key"),
        (ANCHOR_A_ROLE, "query_to_anchor_a"),
        (ANCHOR_B_ROLE, "query_to_anchor_b"),
    )
    for chunk, batch in iter_static_batches(selected, batch_size, spd_cap, device):
        _, layers = model(batch, collect_attention=True)
        batch_idx = torch.arange(batch.role.size(0), device=device)
        for layer_idx, layer in enumerate(layers):
            attn = layer["attn"]
            for head in range(int(attn.size(1))):
                rows = attn[batch_idx, head, batch.query_node, :]
                target_mass = rows[batch_idx, batch.target_node]
                candidate_attn = torch.gather(
                    rows,
                    dim=1,
                    index=batch.candidate_nodes,
                ).masked_fill(~batch.candidate_mask, -1.0)
                target_scores = target_mass[:, None]
                rank = (candidate_attn > target_scores).sum(dim=1).float() + 1.0
                mrr = 1.0 / rank
                entropy_row = rows.masked_fill(~batch.mask, 0.0)
                entropy = -(entropy_row * entropy_row.clamp_min(1.0e-12).log()).sum(dim=-1)
                entropy = entropy / batch.mask.sum(dim=-1).float().clamp_min(2).log()
                target_candidate_idx = torch.tensor(
                    [ex.target_candidate for ex in chunk],
                    dtype=torch.long,
                    device=device,
                )
                target_value = batch.value_nodes[batch_idx, target_candidate_idx]
                query_to_target_value = rows[batch_idx, target_value]
                source_to_target_value = attn[batch_idx, head, batch.target_node, target_value]
                for metric, values in (
                    ("query_to_teacher_source", target_mass),
                    ("teacher_source_rank_mrr", mrr),
                    ("query_attention_entropy", entropy),
                    ("query_to_target_value_leaf", query_to_target_value),
                    ("source_to_own_value_leaf", source_to_target_value),
                ):
                    buckets.setdefault((layer_idx, head, metric), []).extend(
                        values.detach().cpu().numpy().astype(float).tolist()
                    )
                for role_id, metric in role_metrics:
                    role_mask = (batch.role == role_id) & batch.mask
                    values = rows.masked_fill(~role_mask, 0.0).sum(dim=1)
                    buckets.setdefault((layer_idx, head, metric), []).extend(
                        values.detach().cpu().numpy().astype(float).tolist()
                    )
                for local_idx, ex in enumerate(chunk):
                    row = rows[local_idx]
                    for ctype in candidate_types:
                        nodes = [
                            int(node)
                            for node, node_type in zip(ex.candidate_nodes, ex.candidate_types)
                            if node_type == ctype
                        ]
                        value = float(row[nodes].sum().item()) if nodes else 0.0
                        buckets.setdefault((layer_idx, head, f"query_to_type_{ctype}"), []).append(
                            value
                        )
    rows_out: list[dict[str, float | int | str]] = []
    for (layer, head, metric), values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=np.float64)
        rows_out.append(
            {
                **run_meta,
                "layer": layer,
                "head": head,
                "metric": metric,
                "score_mean": float(arr.mean()),
                "score_std": float(arr.std(ddof=0)),
                "n": int(arr.size),
            }
        )
    return rows_out


def summarize_metric_rows(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    buckets: dict[tuple, list[float]] = {}
    meta_by_key: dict[tuple, dict[str, float | int | str]] = {}
    for row in rows:
        task_label = str(row.get("task_name", row["task"]))
        if task_label in {"structural", "mixed"}:
            task_label = f"{task_label}_{row['structural_family']}"
        key = (
            row["task"],
            task_label,
            row["structural_family"],
            row["feature_set"],
            row["depth"],
            row["layer"],
            row["head"],
            row["metric"],
        )
        buckets.setdefault(key, []).append(float(row["score"]))
        meta_by_key[key] = {
            "task": row["task"],
            "task_name": task_label,
            "structural_family": row["structural_family"],
            "feature_set": row["feature_set"],
            "depth": row["depth"],
            "layer": row["layer"],
            "head": row["head"],
            "metric": row["metric"],
        }
    out = []
    for key, values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=np.float64)
        out.append(
            {
                **meta_by_key[key],
                "score_mean": float(arr.mean()),
                "score_std": float(arr.std(ddof=0)),
                "n": int(arr.size),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_curves(log_rows: list[dict], path: Path) -> None:
    if not log_rows:
        return
    plt = import_plotting()
    epochs = [int(row["epoch"]) for row in log_rows]
    train_loss = [float(row["train_loss"]) for row in log_rows]
    train_acc = [float(row.get("train_acc", 0.0)) for row in log_rows]
    val_acc = [float(row["val_acc"]) for row in log_rows]
    id_acc = [float(row["id_acc"]) for row in log_rows]
    ood_acc = [float(row["ood_acc"]) for row in log_rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].plot(epochs, train_loss, color="#636363")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].plot(epochs, train_acc, label="train", color="#636363", linestyle=":")
    axes[1].plot(epochs, val_acc, label="val", color="#756bb1")
    axes[1].plot(epochs, id_acc, label="ID", color="#31a354")
    axes[1].plot(epochs, ood_acc, label="OOD", color="#de2d26")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_example_graph(example: TeacherGraph, path: Path, key_vocab_size: int) -> None:
    plt = import_plotting()
    import matplotlib.patches as mpatches

    backbone_nodes = [
        idx
        for idx, role in enumerate(example.role)
        if role in {BACKBONE_ROLE, CANDIDATE_ROLE, QUERY_ROLE, ANCHOR_A_ROLE, ANCHOR_B_ROLE}
    ]
    radius = max(2.5, len(backbone_nodes) / 10.0)
    pos: dict[int, tuple[float, float]] = {}
    for idx, node in enumerate(backbone_nodes):
        theta = 2.0 * math.pi * idx / max(1, len(backbone_nodes))
        pos[node] = (radius * math.cos(theta), radius * math.sin(theta))
    parent_counts: dict[int, int] = {}
    for node in range(example.n):
        if node in pos:
            continue
        parents = [idx for idx in range(example.n) if example.adj[node, idx] > 0 and idx in pos]
        parent = parents[0] if parents else example.query_node
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
        px, py = pos[parent]
        norm = math.hypot(px, py) or 1.0
        ux, uy = px / norm, py / norm
        tx, ty = -uy, ux
        offset = (parent_counts[parent] - 1) * 0.18
        pos[node] = (px + 0.55 * ux + offset * tx, py + 0.55 * uy + offset * ty)

    candidate_index = {int(node): idx for idx, node in enumerate(example.candidate_nodes)}
    labels: dict[int, str] = {}
    colors = []
    edgecolors = []
    linewidths = []
    for node in range(example.n):
        role = int(example.role[node])
        if role == QUERY_ROLE:
            if example.task in {"symbolic", "mixed"} and int(example.symbol[node]) > 0:
                labels[node] = f"Q/K{int(example.symbol[node] - 1)}"
            else:
                labels[node] = "Q"
            colors.append("#f7f7f7")
            edgecolors.append("#111111")
            linewidths.append(2.5)
        elif role == CANDIDATE_ROLE:
            idx = candidate_index[node]
            value = int(example.value_id[node] - 1)
            label_prefix = "T" if node == example.target_node else f"C{idx}"
            labels[node] = f"{label_prefix}/V{value}"
            ctype = example.candidate_types[idx]
            palette = {
                "target": ("#9ecae1", "#08519c", 3.0),
                "symbolic_only": ("#fcbba1", "#cb181d", 2.4),
                "structural_only": ("#c7e9c0", "#238b45", 2.4),
                "structural_partial": ("#dadaeb", "#6a51a3", 2.0),
                "symbolic_partial": ("#fee391", "#b58100", 2.0),
                "neither": ("#ffffff", "#737373", 1.1),
            }
            face, edge, lw = palette.get(ctype, ("#ffffff", "#737373", 1.1))
            colors.append(face)
            edgecolors.append(edge)
            linewidths.append(lw)
        elif role == VALUE_ROLE:
            if int(example.symbol[node]) > 0:
                value = int(example.symbol[node] - 1 - key_vocab_size)
                labels[node] = f"V{value}"
            else:
                labels[node] = "v"
            colors.append("#fdd0a2")
            edgecolors.append("#636363")
            linewidths.append(0.9)
        elif role == KEY_ROLE:
            labels[node] = f"K{int(example.symbol[node] - 1)}"
            colors.append("#fff7bc")
            edgecolors.append("#b58100")
            linewidths.append(0.9)
        elif role == ANCHOR_A_ROLE:
            labels[node] = "A"
            colors.append("#c7e9c0")
            edgecolors.append("#238b45")
            linewidths.append(2.0)
        elif role == ANCHOR_B_ROLE:
            labels[node] = "B"
            colors.append("#c7e9c0")
            edgecolors.append("#238b45")
            linewidths.append(2.0)
        else:
            labels[node] = ""
            colors.append("#f2f2f2")
            edgecolors.append("#969696")
            linewidths.append(0.8)

    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    for i in range(example.n):
        for j in range(i + 1, example.n):
            if example.adj[i, j] > 0:
                ax.plot(
                    [pos[i][0], pos[j][0]],
                    [pos[i][1], pos[j][1]],
                    color="#d0d0d0",
                    linewidth=0.85,
                    zorder=1,
                )
    ax.scatter(
        [pos[node][0] for node in range(example.n)],
        [pos[node][1] for node in range(example.n)],
        s=380,
        c=colors,
        edgecolors=edgecolors,
        linewidths=linewidths,
        zorder=3,
    )
    for node in range(example.n):
        if labels[node]:
            ax.text(pos[node][0], pos[node][1], labels[node], ha="center", va="center", fontsize=8)
    ax.set_title(
        f"{example.task}/{example.structural_family}; family={example.graph_family}; "
        f"target C{example.target_candidate} -> V{example.target_y}",
        fontsize=11,
    )
    handles = [
        mpatches.Patch(facecolor="#9ecae1", edgecolor="#08519c", label="target"),
        mpatches.Patch(facecolor="#fcbba1", edgecolor="#cb181d", label="symbolic-only"),
        mpatches.Patch(facecolor="#c7e9c0", edgecolor="#238b45", label="structural-only"),
        mpatches.Patch(facecolor="#ffffff", edgecolor="#737373", label="neither"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _ordered_strings(values: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in values})


def _ordered_ints(values: Iterable[object]) -> list[int]:
    return sorted({int(value) for value in values})


def _plot_heatmap_numbers(ax, matrix: np.ndarray) -> None:
    if matrix.size > 160:
        return
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < 0.45 else "black",
                )


def _task_label(row: dict) -> str:
    if "task_name" in row and row["task_name"] != "":
        return str(row["task_name"])
    task = str(row["task"])
    if task == "symbolic":
        return "symbolic"
    return f"{task}_{row['structural_family']}"


def _run_column_labels(rows: list[dict]) -> list[str]:
    features = _ordered_strings(row["feature_set"] for row in rows)
    depths = _ordered_ints(row["depth"] for row in rows)
    return [f"{feature}\nd{depth}" for feature in features for depth in depths]


def plot_suite_accuracy(summary_rows: list[dict], suite_dir: Path) -> None:
    if not summary_rows:
        return
    plt = import_plotting()
    tasks = _ordered_strings(_task_label(row) for row in summary_rows)
    labels = _run_column_labels(summary_rows)
    metrics = (
        ("best_val_acc", "validation"),
        ("best_id_acc", "ID test"),
        ("best_ood_acc", "OOD test"),
    )
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(max(11.0, len(labels) * 1.6), max(4.0, len(tasks) * 0.55)),
        sharey=True,
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(1, -1)[0]
    last_im = None
    for ax, (metric, title) in zip(axes_arr, metrics):
        matrix = np.full((len(tasks), len(labels)), np.nan, dtype=np.float32)
        for row in summary_rows:
            t = tasks.index(_task_label(row))
            col = labels.index(f"{row['feature_set']}\nd{int(row['depth'])}")
            matrix[t, col] = float(row[metric])
        last_im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        _plot_heatmap_numbers(ax, matrix)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(tasks)))
        ax.set_yticklabels(tasks)
        ax.set_title(title)
    fig.suptitle("Best accuracy by task, feature set, and depth")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes_arr.tolist(), shrink=0.82, label="accuracy")
    fig.savefig(suite_dir / "suite_accuracy_heatmaps.png", dpi=180, bbox_inches="tight")
    fig.savefig(suite_dir / "suite_accuracy_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _group_permutation_heads(
    summary_rows: list[dict],
) -> dict[tuple[str, str, int, int, int], dict[str, float]]:
    grouped: dict[tuple[str, str, int, int, int], dict[str, float]] = {}
    for row in summary_rows:
        key = (
            _task_label(row),
            str(row["feature_set"]),
            int(row["depth"]),
            int(row["layer"]),
            int(row["head"]),
        )
        grouped.setdefault(key, {})[str(row["metric"])] = float(row["score_mean"])
    return grouped


def _depth_color_map(plt, depths: list[int]) -> dict[int, object]:
    if not depths:
        return {}
    if len(depths) == 1:
        return {depths[0]: "#3182bd"}
    cmap = plt.cm.viridis
    return {
        depth: cmap(idx / max(1, len(depths) - 1))
        for idx, depth in enumerate(depths)
    }


def plot_permutation_plane(
    summary_rows: list[dict],
    suite_dir: Path,
    *,
    plane: str,
) -> None:
    if not summary_rows:
        return
    plt = import_plotting()
    grouped = _group_permutation_heads(summary_rows)
    tasks = _ordered_strings(key[0] for key in grouped)
    features = _ordered_strings(key[1] for key in grouped)
    depths = _ordered_ints(key[2] for key in grouped)
    if not tasks or not features:
        return
    if plane == "pe":
        x_metric = M_PE_INVARIANT
        y_metric = M_PE_EQUIVARIANT
        x_label = "PE-invariance"
        y_label = "PE-equivariance"
        title = "PE-channel permutation score plane"
        suffix = "pe"
    else:
        x_metric = M_POSITIONAL
        y_metric = M_SYMBOLIC
        x_label = "positional score"
        y_label = "symbolic score"
        title = "Symbolic-channel permutation score plane"
        suffix = "x"
    colors = _depth_color_map(plt, depths)
    markers = ("o", "s", "^", "D", "P", "X")
    fig, axes = plt.subplots(
        len(tasks),
        len(features),
        figsize=(max(7.5, len(features) * 3.4), max(4.0, len(tasks) * 3.0)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for row_idx, task in enumerate(tasks):
        for col_idx, feature in enumerate(features):
            ax = axes[row_idx, col_idx]
            plotted = False
            for key, metrics in grouped.items():
                task_key, feature_key, depth, layer, head = key
                if task_key != task or feature_key != feature:
                    continue
                if x_metric not in metrics or y_metric not in metrics:
                    continue
                plotted = True
                ax.scatter(
                    metrics[x_metric],
                    metrics[y_metric],
                    color=colors.get(depth, "#636363"),
                    marker=markers[layer % len(markers)],
                    s=34 + 10 * head,
                    alpha=0.82,
                    edgecolor="white",
                    linewidth=0.35,
                )
            ax.axhline(0.5, color="#d9d9d9", linewidth=0.7)
            ax.axvline(0.5, color="#d9d9d9", linewidth=0.7)
            ax.plot([0, 1], [0, 1], color="#efefef", linewidth=0.7)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(color="#f0f0f0", linewidth=0.6)
            ax.set_title(f"{task}\n{feature}", fontsize=9)
            if not plotted:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            if row_idx == len(tasks) - 1:
                ax.set_xlabel(x_label)
            if col_idx == 0:
                ax.set_ylabel(y_label)
    depth_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=colors[depth],
            label=f"depth {depth}",
            markersize=6,
        )
        for depth in depths
    ]
    layer_count = max([key[3] for key in grouped], default=-1) + 1
    layer_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[layer % len(markers)],
            linestyle="",
            color="#636363",
            label=f"layer {layer}",
            markersize=6,
        )
        for layer in range(layer_count)
    ]
    if depth_handles or layer_handles:
        fig.legend(
            handles=depth_handles + layer_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=max(2, min(8, len(depth_handles) + len(layer_handles))),
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(suite_dir / f"suite_permutation_plane_{suffix}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_specialisation_by_depth(summary_rows: list[dict], suite_dir: Path) -> None:
    if not summary_rows:
        return
    plt = import_plotting()
    grouped = _group_permutation_heads(summary_rows)
    tasks = _ordered_strings(key[0] for key in grouped)
    features = _ordered_strings(key[1] for key in grouped)
    depths = _ordered_ints(key[2] for key in grouped)
    if not tasks or not depths:
        return
    metric_sets = (
        (
            "x channel",
            M_POSITIONAL,
            M_SYMBOLIC,
        ),
        (
            "PE channel",
            M_PE_INVARIANT,
            M_PE_EQUIVARIANT,
        ),
    )
    fig, axes = plt.subplots(
        len(metric_sets),
        len(tasks),
        figsize=(max(9.0, len(tasks) * 3.5), 6.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    palette = {feature: plt.cm.Set2(idx % 8) for idx, feature in enumerate(features)}
    for row_idx, (label, structural_metric, symbolic_metric) in enumerate(metric_sets):
        for col_idx, task in enumerate(tasks):
            ax = axes[row_idx, col_idx]
            for feature in features:
                structural_scores = []
                symbolic_scores = []
                for depth in depths:
                    metric_values = [
                        metrics
                        for key, metrics in grouped.items()
                        if key[0] == task and key[1] == feature and key[2] == depth
                    ]
                    s_vals = [m[structural_metric] for m in metric_values if structural_metric in m]
                    y_vals = [m[symbolic_metric] for m in metric_values if symbolic_metric in m]
                    structural_scores.append(max(s_vals) if s_vals else np.nan)
                    symbolic_scores.append(max(y_vals) if y_vals else np.nan)
                ax.plot(
                    depths,
                    structural_scores,
                    color=palette[feature],
                    linestyle="-",
                    marker="o",
                    label=f"{feature} structural",
                )
                ax.plot(
                    depths,
                    symbolic_scores,
                    color=palette[feature],
                    linestyle="--",
                    marker="s",
                    label=f"{feature} symbolic",
                )
            ax.set_title(f"{task}\n{label}", fontsize=9)
            ax.set_xticks(depths)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(color="#eeeeee", linewidth=0.7)
            if row_idx == len(metric_sets) - 1:
                ax.set_xlabel("depth")
            if col_idx == 0:
                ax.set_ylabel("best head score")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=max(2, min(4, len(handles))),
            frameon=False,
            fontsize=8,
        )
    fig.suptitle("Best-head permutation specialisation by depth", y=1.02)
    fig.tight_layout()
    fig.savefig(suite_dir / "suite_specialisation_by_depth.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_target_attention(attention_rows: list[dict], suite_dir: Path) -> None:
    if not attention_rows:
        return
    plt = import_plotting()
    final_rows: list[dict] = []
    for row in attention_rows:
        if str(row["metric"]) != "teacher_source_rank_mrr":
            continue
        final_rows.append(row)
    if not final_rows:
        return
    tasks = _ordered_strings(_task_label(row) for row in final_rows)
    labels = _run_column_labels(final_rows)
    matrix = np.full((len(tasks), len(labels)), np.nan, dtype=np.float32)
    for row in final_rows:
        t = tasks.index(_task_label(row))
        col = labels.index(f"{row['feature_set']}\nd{int(row['depth'])}")
        value = float(row["score_mean"])
        if np.isnan(matrix[t, col]) or value > matrix[t, col]:
            matrix[t, col] = value
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.8), max(3.8, len(tasks) * 0.55))
    )
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma", aspect="auto")
    _plot_heatmap_numbers(ax, matrix)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)
    ax.set_title("Best teacher-source attention MRR by run")
    fig.colorbar(im, ax=ax, label="best head MRR")
    fig.tight_layout()
    fig.savefig(suite_dir / "suite_teacher_source_attention.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_head_diagnostics(attention_rows: list[dict], path: Path) -> None:
    if not attention_rows:
        return
    plt = import_plotting()
    metric_order = [
        "teacher_source_rank_mrr",
        "query_to_teacher_source",
        "query_to_target_value_leaf",
        "source_to_own_value_leaf",
        "query_to_type_target",
        "query_to_type_symbolic_only",
        "query_to_type_structural_only",
        "query_to_type_structural_partial",
        "query_to_type_symbolic_partial",
        "query_to_type_neither",
        "query_to_role_candidate",
        "query_to_role_value",
        "query_to_role_key",
        "query_to_anchor_a",
        "query_to_anchor_b",
        "query_attention_entropy",
    ]
    metric_labels = [
        "source MRR",
        "Q->source",
        "Q->value",
        "source->value",
        "Q->target type",
        "Q->sym-only",
        "Q->struct-only",
        "Q->struct partial",
        "Q->sym partial",
        "Q->neither",
        "Q->candidates",
        "Q->values",
        "Q->keys",
        "Q->anchor A",
        "Q->anchor B",
        "Q entropy",
    ]
    heads = sorted({(int(row["layer"]), int(row["head"])) for row in attention_rows})
    if not heads:
        return
    row_labels = [f"L{layer}H{head}" for layer, head in heads]
    matrix = np.full((len(heads), len(metric_order)), np.nan, dtype=np.float32)
    for row in attention_rows:
        head_idx = heads.index((int(row["layer"]), int(row["head"])))
        metric = str(row["metric"])
        if metric in metric_order:
            matrix[head_idx, metric_order.index(metric)] = float(row["score_mean"])
    fig, ax = plt.subplots(
        figsize=(max(10.0, len(metric_order) * 0.6), max(2.8, len(heads) * 0.55))
    )
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    _plot_heatmap_numbers(ax, matrix)
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(heads)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Head role diagnostics on ID examples")
    fig.colorbar(im, ax=ax, label="mean attention / rank score")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_error_types(error_rows: list[dict], suite_dir: Path) -> None:
    if not error_rows:
        return
    plt = import_plotting()
    run_keys = []
    type_keys = sorted({str(row["prediction_type"]) for row in error_rows})
    counts: dict[tuple[str, str], int] = {}
    for row in error_rows:
        run = f"{row['task_name']}/{row['feature_set']}/d{row['depth']}"
        if run not in run_keys:
            run_keys.append(run)
        counts[(run, str(row["prediction_type"]))] = int(row["count"])
    if len(run_keys) > 36:
        run_keys = run_keys[:36]
    matrix = np.zeros((len(type_keys), len(run_keys)), dtype=np.float32)
    totals = np.zeros(len(run_keys), dtype=np.float32)
    for j, run in enumerate(run_keys):
        totals[j] = sum(counts.get((run, typ), 0) for typ in type_keys)
        for i, typ in enumerate(type_keys):
            matrix[i, j] = counts.get((run, typ), 0) / max(1.0, totals[j])
    fig, ax = plt.subplots(figsize=(max(9.0, len(run_keys) * 0.35), 4.8))
    bottom = np.zeros(len(run_keys), dtype=np.float32)
    cmap = plt.cm.tab20
    for i, typ in enumerate(type_keys):
        ax.bar(np.arange(len(run_keys)), matrix[i], bottom=bottom, label=typ, color=cmap(i % 20))
        bottom += matrix[i]
    ax.set_xticks(np.arange(len(run_keys)))
    ax.set_xticklabels(run_keys, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("fraction")
    ax.set_title("Prediction/error type distribution on ID test")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(suite_dir / "suite_error_type_stacks.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def task_name(task: TeacherTask, structural_family: StructuralKeyFamily) -> str:
    if task == "symbolic":
        return "symbolic"
    return f"{task}_{structural_family}"


def build_specs(args: argparse.Namespace) -> list[tuple[TeacherTask, StructuralKeyFamily]]:
    specs: list[tuple[TeacherTask, StructuralKeyFamily]] = []
    for task in args.tasks:
        if task == "symbolic":
            specs.append(("symbolic", args.structural_key_families[0]))
        else:
            for family in args.structural_key_families:
                specs.append((task, family))
    return specs


def train_one_run(
    task: TeacherTask,
    structural_family: StructuralKeyFamily,
    feature_set: FeatureSet,
    depth: int,
    suite_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    run_name = f"{task_name(task, structural_family)}__{feature_set}__depth_{depth}"
    run_dir = suite_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "task": task,
                "structural_family": structural_family,
                "feature_set": feature_set,
                "depth": depth,
                "device_resolved": str(device),
            },
            f,
            indent=2,
            default=str,
        )

    print(
        f"[setup] {run_name} hidden={args.hidden_dim} heads={args.num_heads}",
        flush=True,
    )
    val_examples = generate_examples(
        args.val_graphs,
        args.seed + 101,
        task,
        structural_family,
        feature_set,
        args,
    )
    id_examples = generate_examples(
        args.id_test_graphs,
        args.seed + 202,
        task,
        structural_family,
        feature_set,
        args,
    )
    ood_examples = generate_examples(
        args.ood_test_graphs,
        args.seed + 303,
        task,
        structural_family,
        feature_set,
        args,
        ood=True,
    )

    model_kwargs = {
        "depth": depth,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "symbol_vocab": symbol_vocab_size(args.key_vocab_size, args.value_vocab_size),
        "pe_dim": val_examples[0].pe.shape[1],
        "value_vocab_size": args.value_vocab_size,
        "dropout": args.dropout,
        "attn_dropout": args.attn_dropout,
        "use_spd_bias": feature_set_uses_spd(feature_set),
        "spd_cap": args.spd_cap,
        "value_decoder": args.value_decoder,
        "fixed_value_embeddings": args.fixed_value_embeddings,
    }
    model = GraphGPSQueryModel(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_rows: list[dict] = []
    best_val_acc = -1.0
    best_epoch = 0
    best_stats: dict[str, EvalStats] = {}
    best_error_types: dict[str, int] = {}
    bad_epochs = 0
    best_state = None
    fixed_train_examples = None
    if args.fixed_train_set:
        fixed_train_examples = generate_examples(
            args.train_graphs_per_epoch,
            args.seed + 9091 + depth * 9173,
            task,
            structural_family,
            feature_set,
            args,
        )

    for epoch in range(1, args.max_epochs + 1):
        if fixed_train_examples is None:
            train_examples = generate_examples(
                args.train_graphs_per_epoch,
                args.seed + epoch * 1009 + depth * 9173,
                task,
                structural_family,
                feature_set,
                args,
            )
        else:
            train_examples = list(fixed_train_examples)
        rng = np.random.default_rng(args.seed + epoch * 1297)
        order = rng.permutation(len(train_examples))
        train_examples = [train_examples[int(i)] for i in order]

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_n = 0
        for _, batch in iter_static_batches(train_examples, args.batch_size, args.spd_cap, device):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch, collect_attention=False)
            loss = batch_loss(logits, batch)
            preds = logits.argmax(dim=-1)
            train_correct += int((preds == batch.target_y).sum().item())
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * int(batch.role.size(0))
            train_n += int(batch.role.size(0))

        if epoch % args.eval_every != 0 and epoch != args.max_epochs:
            continue

        val_stats, _ = evaluate(model, val_examples, args.eval_batch_size, args.spd_cap, device)
        id_stats, id_error_types = evaluate(
            model,
            id_examples,
            args.eval_batch_size,
            args.spd_cap,
            device,
        )
        ood_stats, _ = evaluate(
            model,
            ood_examples,
            args.eval_batch_size,
            args.spd_cap,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_n),
            "train_acc": train_correct / max(1, train_n),
            "val_loss": val_stats.loss,
            "val_acc": val_stats.acc,
            "id_loss": id_stats.loss,
            "id_acc": id_stats.acc,
            "ood_loss": ood_stats.loss,
            "ood_acc": ood_stats.acc,
        }
        log_rows.append(row)
        print(
            f"[{run_name} | epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.4f} train={row['train_acc']:.3f} "
            f"val_loss={val_stats.loss:.4f} val={val_stats.acc:.3f} "
            f"ID={id_stats.acc:.3f} OOD={ood_stats.acc:.3f}",
            flush=True,
        )

        if val_stats.acc > best_val_acc:
            best_val_acc = val_stats.acc
            best_epoch = epoch
            best_stats = {"val": val_stats, "id": id_stats, "ood": ood_stats}
            best_error_types = id_error_types
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += args.eval_every

        if args.stop_on_solved and val_stats.acc >= args.solved_threshold:
            print(f"[early] solved threshold reached for {run_name}", flush=True)
            break
        if bad_epochs >= args.patience:
            print(f"[early] patience reached for {run_name}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    write_csv(run_dir / "train_log.csv", log_rows)
    plot_training_curves(log_rows, run_dir / "training_curves.png")
    plot_example_graph(id_examples[0], run_dir / "task_example.png", args.key_vocab_size)

    run_meta = {
        "task": task,
        "task_name": task_name(task, structural_family),
        "structural_family": structural_family,
        "feature_set": feature_set,
        "depth": depth,
    }
    perm_rows: list[dict[str, float | int | str]] = []
    perm_summary: list[dict[str, float | int | str]] = []
    if not args.skip_permutation_metrics:
        perm_rows = compute_permutation_metrics(
            model,
            id_examples,
            run_meta,
            batch_size=args.metric_batch_size,
            spd_cap=args.spd_cap,
            metric_graphs=args.metric_graphs,
            num_perms=args.metric_perms,
            seed=args.seed + 404 + depth,
            device=device,
        )
        perm_summary = summarize_metric_rows(perm_rows)
        write_csv(run_dir / "symbol_permutation_metrics.csv", perm_rows)
        write_csv(run_dir / "symbol_permutation_summary.csv", perm_summary)

    attention_rows = compute_target_attention_metrics(
        model,
        id_examples,
        run_meta,
        batch_size=args.metric_batch_size,
        spd_cap=args.spd_cap,
        metric_graphs=args.metric_graphs,
        device=device,
    )
    write_csv(run_dir / "teacher_source_attention_summary.csv", attention_rows)
    plot_head_diagnostics(attention_rows, run_dir / "head_role_diagnostics.png")

    error_rows = [
        {
            **run_meta,
            "prediction_type": prediction_type,
            "count": count,
            "split": "id_test",
        }
        for prediction_type, count in sorted(best_error_types.items())
    ]
    write_csv(run_dir / "id_error_types.csv", error_rows)

    summary = {
        **run_meta,
        "best_epoch": best_epoch,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_val_loss": best_stats["val"].loss,
        "best_val_acc": best_stats["val"].acc,
        "best_val_mrr_value_proxy": best_stats["val"].mrr_value_proxy,
        "best_id_loss": best_stats["id"].loss,
        "best_id_acc": best_stats["id"].acc,
        "best_id_mrr_value_proxy": best_stats["id"].mrr_value_proxy,
        "best_ood_loss": best_stats["ood"].loss,
        "best_ood_acc": best_stats["ood"].acc,
        "best_ood_mrr_value_proxy": best_stats["ood"].mrr_value_proxy,
    }
    write_csv(run_dir / "summary.csv", [summary])
    if not args.no_save_checkpoints:
        state_dict_cpu = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        run_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        checkpoint = {
            "format_version": 1,
            "model_class": "GraphGPSQueryModel",
            "model_kwargs": model_kwargs,
            "model_state_dict": state_dict_cpu,
            "run_meta": run_meta,
            "summary": summary,
            "config": run_config
            | {
                "task": task,
                "structural_family": structural_family,
                "feature_set": feature_set,
                "depth": depth,
                "device_resolved": str(device),
            },
            "role_ids": {
                "pad": PAD_ROLE,
                "backbone": BACKBONE_ROLE,
                "candidate": CANDIDATE_ROLE,
                "value": VALUE_ROLE,
                "query": QUERY_ROLE,
                "key": KEY_ROLE,
                "anchor_a": ANCHOR_A_ROLE,
                "anchor_b": ANCHOR_B_ROLE,
            },
            "best_epoch": best_epoch,
            "train_log": log_rows,
        }
        torch.save(state_dict_cpu, run_dir / "model.pt")
        torch.save(checkpoint, run_dir / "checkpoint.pt")
    return summary, perm_summary, attention_rows, error_rows, log_rows


def running_in_colab_env() -> bool:
    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return False
    return True


def resolve_output_root(args: argparse.Namespace) -> Path:
    in_colab = running_in_colab_env()
    if in_colab and not args.no_mount_drive:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount("/content/drive", force_remount=False)
    if args.output_dir is not None:
        return Path(args.output_dir)
    if in_colab:
        drive_root = Path(args.drive_output_root)
        if drive_root.is_absolute():
            return drive_root
        return Path("/content/drive") / drive_root
    return Path("experiments/synthetic/results/teacher_student_structural_keys_graphgps")


def write_design_card(path: Path, args: argparse.Namespace) -> None:
    text = f"""# Teacher-Student Structural-Key Routing Sweep

This suite trains GraphGPS students against deterministic synthetic teachers.
The teacher chooses one source candidate for each query and the label is the
source candidate's attached value. The source candidate directly carries its
copied value, while the value leaf remains in the graph for diagnostics.
Teacher attention is not used for training, but the source node is saved for
diagnostics.

## Resolved Sweep

- Model: GraphGPS only.
- Suite preset: `{args.suite_preset}`.
- Depths: `{args.depths}`.
- Feature sets: `{args.feature_sets}`.
- Tasks: `{args.tasks}`.
- Structural key families: `{args.structural_key_families}`.
- Runs: `{len(build_specs(args)) * len(args.feature_sets) * len(args.depths)}`.
- Graph source: local GraphWorld-style random generator with ER, WS, BA, SBM,
  or mixed graph families.
- Training data: fresh synthetic graphs each epoch by default. Pass
  `--fixed-train-set` only when debugging memorisation capacity.
- Per run: `{args.train_graphs_per_epoch}` train graphs/epoch, `{args.val_graphs}`
  validation graphs, `{args.id_test_graphs}` ID test graphs, and
  `{args.ood_test_graphs}` OOD test graphs.
- Value readout: `{args.value_decoder}` decoder; fixed value embeddings:
  `{args.fixed_value_embeddings}`.
- Value leaves carry copied value tokens: `{args.value_leaf_carries_value}`.

## Teachers

- `symbolic`: explicit key tuple equality selects the source candidate.
- `structural`: hidden graph-derived structural key equality selects the source.
- `mixed`: explicit symbolic key and hidden structural key must both match.

Structural key families:

- `anchor_distance`: clipped shortest-path distance to two anchor nodes. This
  is the intended one-head/one-layer starter structural task: no symbolic key is
  provided, and the recoverable signal is structural PE.
- `local`: degree, triangle count, and two-hop expansion buckets.
- `diffusion`: two-anchor personalized PageRank buckets and anchor-distance
  contrast bucket.
- `global`: k-core, clustering, and closeness buckets.

Mixed distractors are typed as `symbolic_only`, `structural_only`, and
`neither`, so errors can be interpreted as shortcut use.

## Metrics

- Accuracy and value-space MRR on validation, ID, and OOD splits.
- Prediction/error type counts on ID.
- Teacher-source attention mass and teacher-source attention MRR by layer/head.
- Head role diagnostics: attention to candidate types, blank/value leaves, key
  leaves, anchors, and entropy.
- ZINC-style forward-pass permutation metrics on row-centered attention logits:
  `positional_score`, `symbolic_score`, `pe_invariance`, `pe_equivariance`,
  `entanglement_score`, `entropy_norm`, and `relabel_equivariance`.
- Suite figures split symbolic-channel and PE-channel permutation score planes.
"""
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GraphGPS teacher-student routing sweep with structural keys."
    )
    parser.add_argument(
        "--suite-preset",
        choices=SUITE_PRESETS,
        default="one_head_one_layer",
        help=(
            "one_head_one_layer is the fast starter run; pilot is the broader "
            "depth sweep; family_scan checks all structural key families at "
            "depth 2; full runs the larger matrix; custom uses explicit "
            "task/family/feature/depth flags."
        ),
    )
    parser.add_argument("--tasks", nargs="+", choices=TASK_CHOICES, default=None)
    parser.add_argument(
        "--structural-key-families",
        nargs="+",
        choices=STRUCTURAL_KEY_FAMILIES,
        default=None,
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=FEATURE_SETS,
        default=None,
        help="Defaults depend on --suite-preset.",
    )
    parser.add_argument("--depths", type=int, nargs="+", default=None)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument(
        "--value-decoder",
        choices=("tied", "mlp"),
        default="tied",
        help=(
            "tied decodes against value input codes, focusing difficulty on "
            "routing/copying; mlp is the older untied value classifier."
        ),
    )
    parser.add_argument(
        "--fixed-value-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use fixed near-orthogonal value codes. This removes an avoidable "
            "arbitrary class-code alignment problem from the teacher-student task."
        ),
    )
    parser.add_argument(
        "--value-leaf-carries-value",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When enabled, value leaves carry the same value token as their "
            "candidate. The starter keeps them blank so attention cannot solve "
            "the task by bypassing the teacher source candidate."
        ),
    )

    parser.add_argument("--graph-family", choices=GRAPH_FAMILIES, default="mixed")
    parser.add_argument("--graph-min-nodes", type=int, default=24)
    parser.add_argument("--graph-max-nodes", type=int, default=72)
    parser.add_argument("--ood-graph-min-nodes", type=int, default=72)
    parser.add_argument("--ood-graph-max-nodes", type=int, default=120)
    parser.add_argument("--min-candidates", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--ood-min-candidates", type=int, default=12)
    parser.add_argument("--ood-max-candidates", type=int, default=20)
    parser.add_argument("--key-vocab-size", type=int, default=64)
    parser.add_argument("--key-tuple-size", type=int, default=2)
    parser.add_argument("--value-vocab-size", type=int, default=32)
    parser.add_argument("--structural-buckets", type=int, default=4)
    parser.add_argument("--rwse-steps", type=int, default=12)
    parser.add_argument("--spd-cap", type=int, default=32)

    parser.add_argument("--train-graphs-per-epoch", type=int, default=2048)
    parser.add_argument("--val-graphs", type=int, default=512)
    parser.add_argument("--id-test-graphs", type=int, default=1024)
    parser.add_argument("--ood-test-graphs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--stop-on-solved", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--solved-threshold", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--fixed-train-set",
        action="store_true",
        help="Reuse one generated training set per run. Use only for debugging memorisation.",
    )

    parser.add_argument("--metric-graphs", type=int, default=128)
    parser.add_argument("--metric-perms", type=int, default=4)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--skip-permutation-metrics", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--drive-output-root",
        type=Path,
        default=Path("MyDrive/graph_specialisation_metrics/teacher_student_structural_keys"),
    )
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved run matrix and exit without mounting Drive or training.",
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Small static/debug configuration. Not used by default.",
    )
    args = parser.parse_args(notebook_safe_argv(argv))

    apply_suite_preset(args)
    if args.key_vocab_size < 8:
        raise ValueError("--key-vocab-size should be at least 8")
    if args.value_vocab_size < args.ood_max_candidates:
        raise ValueError("--value-vocab-size should be >= --ood-max-candidates")
    if args.key_tuple_size < 1:
        raise ValueError("--key-tuple-size must be positive")
    if args.structural_buckets < 2:
        raise ValueError("--structural-buckets must be at least 2")
    if args.fast_dev_run:
        args.tasks = ["symbolic", "structural"]
        args.structural_key_families = ["anchor_distance"]
        args.feature_sets = args.feature_sets[:1]
        args.depths = args.depths[:1]
        args.train_graphs_per_epoch = 48
        args.val_graphs = 16
        args.id_test_graphs = 16
        args.ood_test_graphs = 16
        args.batch_size = 8
        args.eval_batch_size = 16
        args.metric_graphs = 8
        args.metric_perms = 1
        args.max_epochs = min(args.max_epochs, 2)
        args.rwse_steps = min(args.rwse_steps, 4)
    return args


def apply_suite_preset(args: argparse.Namespace) -> None:
    if args.suite_preset == "one_head_one_layer":
        args.tasks = args.tasks or ["symbolic", "structural"]
        args.structural_key_families = args.structural_key_families or ["anchor_distance"]
        args.feature_sets = args.feature_sets or ["anchor_dist"]
        args.depths = args.depths or [1]
        args.hidden_dim = 48
        args.num_heads = 1
        args.graph_min_nodes = 12
        args.graph_max_nodes = 28
        args.ood_graph_min_nodes = 28
        args.ood_graph_max_nodes = 44
        args.min_candidates = 3
        args.max_candidates = 6
        args.ood_min_candidates = 6
        args.ood_max_candidates = 10
        args.key_vocab_size = 16
        args.key_tuple_size = 1
        args.value_vocab_size = 16
        args.structural_buckets = 4
        args.rwse_steps = 4
        args.train_graphs_per_epoch = 1024
        args.val_graphs = 128
        args.id_test_graphs = 256
        args.ood_test_graphs = 256
        args.batch_size = 128
        args.eval_batch_size = 128
        args.max_epochs = 40
        args.eval_every = 2
        args.patience = 20
        args.lr = 2.0e-3
        args.metric_graphs = 64
        args.metric_perms = 2
        args.metric_batch_size = 128
    elif args.suite_preset == "single_head_fast":
        args.tasks = args.tasks or ["symbolic", "structural", "mixed"]
        args.structural_key_families = args.structural_key_families or ["local"]
        args.feature_sets = args.feature_sets or ["rwse_stats"]
        args.depths = args.depths or [2]
        args.hidden_dim = 48
        args.num_heads = 1
        args.graph_min_nodes = 12
        args.graph_max_nodes = 32
        args.ood_graph_min_nodes = 32
        args.ood_graph_max_nodes = 56
        args.min_candidates = 3
        args.max_candidates = 6
        args.ood_min_candidates = 6
        args.ood_max_candidates = 10
        args.key_vocab_size = 16
        args.key_tuple_size = 1
        args.value_vocab_size = 16
        args.structural_buckets = 3
        args.rwse_steps = 6
        args.train_graphs_per_epoch = 1024
        args.val_graphs = 128
        args.id_test_graphs = 256
        args.ood_test_graphs = 256
        args.batch_size = 128
        args.eval_batch_size = 128
        args.max_epochs = 40
        args.eval_every = 2
        args.patience = 20
        args.lr = 2.0e-3
        args.metric_graphs = 64
        args.metric_perms = 2
        args.metric_batch_size = 128
    elif args.suite_preset == "pilot":
        args.tasks = args.tasks or ["symbolic", "structural", "mixed"]
        args.structural_key_families = args.structural_key_families or ["local"]
        args.feature_sets = args.feature_sets or ["rwse", "rwse_stats"]
        args.depths = args.depths or [2, 3]
    elif args.suite_preset == "family_scan":
        args.tasks = args.tasks or ["symbolic", "structural", "mixed"]
        args.structural_key_families = args.structural_key_families or list(
            STRUCTURAL_KEY_FAMILIES
        )
        args.feature_sets = args.feature_sets or ["rwse_stats"]
        args.depths = args.depths or [2]
    elif args.suite_preset == "full":
        args.tasks = args.tasks or list(TASK_CHOICES)
        args.structural_key_families = args.structural_key_families or list(
            STRUCTURAL_KEY_FAMILIES
        )
        args.feature_sets = args.feature_sets or ["rwse", "rwse_stats"]
        args.depths = args.depths or [1, 2, 3]
    else:
        args.tasks = args.tasks or ["symbolic", "structural", "mixed"]
        args.structural_key_families = args.structural_key_families or ["local"]
        args.feature_sets = args.feature_sets or ["rwse", "rwse_stats"]
        args.depths = args.depths or [2, 3]


def print_run_plan(args: argparse.Namespace) -> None:
    specs = build_specs(args)
    total = len(specs) * len(args.feature_sets) * len(args.depths)
    print(f"suite_preset={args.suite_preset}")
    print(f"tasks={args.tasks}")
    print(f"structural_key_families={args.structural_key_families}")
    print(f"feature_sets={args.feature_sets}")
    print(f"depths={args.depths}")
    print(f"hidden_dim={args.hidden_dim}, num_heads={args.num_heads}")
    print(f"runs={total}")
    print(
        "per_run="
        f"train_graphs_per_epoch={args.train_graphs_per_epoch}, "
        f"max_epochs={args.max_epochs}, val={args.val_graphs}, "
        f"ID={args.id_test_graphs}, OOD={args.ood_test_graphs}"
    )
    print(
        "difficulty="
        f"ID_nodes={args.graph_min_nodes}-{args.graph_max_nodes}, "
        f"ID_candidates={args.min_candidates}-{args.max_candidates}, "
        f"OOD_nodes={args.ood_graph_min_nodes}-{args.ood_graph_max_nodes}, "
        f"OOD_candidates={args.ood_min_candidates}-{args.ood_max_candidates}"
    )
    print(
        f"value_decoder={args.value_decoder}, "
        f"fixed_value_embeddings={args.fixed_value_embeddings}"
    )
    for task, structural_family in specs:
        for feature_set in args.feature_sets:
            for depth in args.depths:
                print(f"- {task_name(task, structural_family)} / {feature_set} / depth {depth}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        print_run_plan(args)
        return
    set_seed(args.seed)
    device = choose_device(args.device)
    output_root = resolve_output_root(args)
    run_name = args.run_name or datetime.now().strftime("graphgps_teacher_student_%Y%m%d_%H%M%S")
    suite_dir = output_root / run_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    with (suite_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"device_resolved": str(device)}, f, indent=2, default=str)
    write_design_card(suite_dir / "design_card.md", args)

    print(f"[setup] output_dir={suite_dir}", flush=True)
    print(f"[setup] device={device}", flush=True)
    print(
        f"[setup] specs={len(build_specs(args))} features={args.feature_sets} "
        f"depths={args.depths}",
        flush=True,
    )

    summaries: list[dict] = []
    all_perm_summary: list[dict] = []
    all_attention_rows: list[dict] = []
    all_error_rows: list[dict] = []

    start = time.time()
    for task, structural_family in build_specs(args):
        for feature_set in args.feature_sets:
            for depth in args.depths:
                summary, perm_summary, attention_rows, error_rows, _ = train_one_run(
                    task=task,
                    structural_family=structural_family,
                    feature_set=feature_set,
                    depth=depth,
                    suite_dir=suite_dir,
                    args=args,
                    device=device,
                )
                summaries.append(summary)
                all_perm_summary.extend(perm_summary)
                all_attention_rows.extend(attention_rows)
                all_error_rows.extend(error_rows)
                write_csv(suite_dir / "summary_all_runs.csv", summaries)
                write_csv(suite_dir / "symbol_permutation_summary_all_runs.csv", all_perm_summary)
                write_csv(suite_dir / "teacher_source_attention_all_runs.csv", all_attention_rows)
                write_csv(suite_dir / "id_error_types_all_runs.csv", all_error_rows)

    plot_suite_accuracy(summaries, suite_dir)
    plot_permutation_plane(all_perm_summary, suite_dir, plane="x")
    plot_permutation_plane(all_perm_summary, suite_dir, plane="pe")
    plot_specialisation_by_depth(all_perm_summary, suite_dir)
    plot_target_attention(all_attention_rows, suite_dir)
    plot_error_types(all_error_rows, suite_dir)

    print("[done] suite summary", flush=True)
    for row in summaries:
        print(
            f"{row['task_name']} {row['feature_set']} depth={row['depth']} "
            f"val={row['best_val_acc']:.3f} ID={row['best_id_acc']:.3f} "
            f"OOD={row['best_ood_acc']:.3f}",
            flush=True,
        )
    print(f"[done] wrote {suite_dir}", flush=True)
    print(f"[done] elapsed_minutes={(time.time() - start) / 60.0:.2f}", flush=True)


if __name__ == "__main__":
    main()
