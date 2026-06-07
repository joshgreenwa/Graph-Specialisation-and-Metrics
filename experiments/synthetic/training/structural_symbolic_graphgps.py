#!/usr/bin/env python3
"""GraphGPS-style baseline for structural and symbolic graph tasks.

The default dataset is GraphWorld-inspired: every sample is a fresh random graph
backbone, drawn from ER, small-world, preferential-attachment, or SBM-like
families. A small set of candidate nodes carry semantic key/value features, and
one optional anchor node defines a positional/structural query.

Two independent graph-level labels are available:

* structural label: classify the anchor-distance shell of a structurally marked
  candidate node, or optionally retrieve a uniquely nearest candidate's value;
* symbolic label: retrieve the value attached to the candidate whose key matches
  the query key at the readout node.

The two targets are deliberately generated independently. The structural target
is unchanged by key/query symbol changes, while the symbolic target is unchanged
by anchor placement. With the default 2-layer, 2-head model, the structural head
reads the marked target node and the symbolic head reads the readout node. The
dual task trains both labels at once, which is useful for encouraging head
specialisation.

Outputs include train/test CSVs, structural-vs-symbolic head scores, target
attention mass metrics, and several PNG visualisations. The older record/motif
generator is still available with ``--dataset-style record``.
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


# Notebook/Colab hook:
# If this whole file is pasted into a notebook cell, kernel argv contains
# IPython flags that would normally break argparse. In a notebook cell we ignore
# those flags and parse CELL_ARGS instead. Edit this value in the pasted cell to
# override defaults, for example:
# CELL_ARGS = "--task dual --max-epochs 50 --structural-channel rwse_spd_bias"
CELL_ARGS: list[str] | str | None = None


TaskName = Literal["structural", "symbolic", "dual"]
StructuralChannel = Literal["none", "degree", "rwse", "spd_bias", "rwse_spd_bias"]
MotifStyle = Literal["branch", "subtle"]
SymbolicDistractorMode = Literal["random", "partial", "mixed"]
DatasetStyle = Literal["graphworld", "record"]
GraphFamily = Literal["er", "ws", "ba", "sbm", "mixed"]
GraphworldStructuralLabel = Literal["distance_bin", "value_at_nearest"]
TASK_CHOICES = ("structural", "symbolic", "dual")

PAD_ROLE = 0
CLS_ROLE = 1
RECORD_ROLE = 2
VALUE_ROLE = 3
MOTIF_ROLE = 4
KEY_ROLE = 5
QUERY_ROLE = 6
ANCHOR_ROLE = 7
STRUCT_TARGET_ROLE = 8
NUM_ROLES = 9


@dataclass
class GraphExample:
    role: np.ndarray
    symbol: np.ndarray
    adj: np.ndarray
    pe: np.ndarray
    spd: np.ndarray
    n: int
    record_nodes: np.ndarray
    value_nodes: np.ndarray
    structural_record: int
    symbolic_record: int
    structural_y: int
    symbolic_y: int
    dataset_style: str
    graph_family: str
    anchor_node: int
    structural_query_type: int
    structural_bridge: bool
    symbolic_partial_distractors: bool
    record_motif_types: np.ndarray
    query_key: int
    query_keys: np.ndarray
    record_keys: np.ndarray
    record_values: np.ndarray


@dataclass
class Batch:
    role: torch.Tensor
    symbol: torch.Tensor
    mask: torch.Tensor
    adj_norm: torch.Tensor
    pe: torch.Tensor
    spd: torch.Tensor
    record_nodes: torch.Tensor
    record_mask: torch.Tensor
    value_nodes: torch.Tensor
    structural_node: torch.Tensor
    symbolic_node: torch.Tensor
    structural_value_node: torch.Tensor
    symbolic_value_node: torch.Tensor
    structural_y: torch.Tensor
    symbolic_y: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            role=self.role.to(device),
            symbol=self.symbol.to(device),
            mask=self.mask.to(device),
            adj_norm=self.adj_norm.to(device),
            pe=self.pe.to(device),
            spd=self.spd.to(device),
            record_nodes=self.record_nodes.to(device),
            record_mask=self.record_mask.to(device),
            value_nodes=self.value_nodes.to(device),
            structural_node=self.structural_node.to(device),
            symbolic_node=self.symbolic_node.to(device),
            structural_value_node=self.structural_value_node.to(device),
            symbolic_value_node=self.symbolic_value_node.to(device),
            structural_y=self.structural_y.to(device),
            symbolic_y=self.symbolic_y.to(device),
        )


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


def symbol_vocab_size(num_keys: int, num_values: int) -> int:
    # 0 is "no symbol"; keys are 1..num_keys; values are offset after keys.
    return 1 + num_keys + num_values


def key_symbol(key: int) -> int:
    return 1 + int(key)


def value_symbol(value: int, num_keys: int) -> int:
    return 1 + int(num_keys) + int(value)


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


def compute_rwse(adj: np.ndarray, steps: int) -> np.ndarray:
    n = int(adj.shape[0])
    if steps <= 0:
        return np.zeros((n, 0), dtype=np.float32)
    deg = np.maximum(adj.sum(axis=1), 1.0).astype(np.float32)
    row_scaled = adj / deg[:, None]

    try:
        import scipy.sparse as sp

        p_mat = sp.csr_matrix(row_scaled)
        cur = sp.identity(n, dtype=np.float32, format="csr")
        feats = []
        for _ in range(steps):
            cur = cur @ p_mat
            feats.append(cur.diagonal().astype(np.float32))
        return np.stack(feats, axis=1)
    except Exception:
        cur_dense = np.eye(n, dtype=np.float32)
        feats = []
        for _ in range(steps):
            cur_dense = cur_dense @ row_scaled
            feats.append(np.diag(cur_dense).astype(np.float32))
        return np.stack(feats, axis=1)


def offset_edges(edges: list[tuple[int, int]], offset: int) -> list[tuple[int, int]]:
    return [(a + offset, b + offset) for a, b in edges]


def random_tree_edges(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    order = np.arange(n)
    rng.shuffle(order)
    edges = []
    for idx in range(1, n):
        child = int(order[idx])
        parent = int(order[int(rng.integers(0, idx))])
        edges.append((child, parent))
    return edges


def add_unique_edge(
    edges: set[tuple[int, int]],
    a: int,
    b: int,
) -> None:
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
        p = float(rng.uniform(1.5 / n, 4.0 / n))
        for a in range(n):
            for b in range(a + 1, n):
                if rng.random() < p:
                    add_unique_edge(edge_set, a, b)
    elif family == "ws":
        k = int(rng.choice([4, 6]))
        beta = float(rng.uniform(0.05, 0.35))
        for a in range(n):
            for step in range(1, k // 2 + 1):
                b = (a + step) % n
                if rng.random() < beta:
                    b = int(rng.integers(0, n))
                add_unique_edge(edge_set, a, b)
    elif family == "ba":
        degrees = np.ones(n, dtype=np.float64)
        m = int(rng.choice([2, 3]))
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
        groups = rng.integers(0, 2, size=n)
        p_in = float(rng.uniform(3.0 / n, 6.0 / n))
        p_out = float(rng.uniform(0.3 / n, 1.5 / n))
        for a in range(n):
            for b in range(a + 1, n):
                p = p_in if groups[a] == groups[b] else p_out
                if rng.random() < p:
                    add_unique_edge(edge_set, a, b)
    else:
        raise ValueError(f"Unknown graph family: {family}")
    return sorted(edge_set), family


def pe_dim_for_channel(channel: StructuralChannel, rwse_steps: int) -> int:
    if channel == "degree":
        return 1
    if channel in {"rwse", "rwse_spd_bias"}:
        return rwse_steps
    return 0


def uses_spd_bias(channel: StructuralChannel) -> bool:
    return channel in {"spd_bias", "rwse_spd_bias"}


def make_record_keys(
    num_records: int,
    symbolic_record: int,
    query_keys: np.ndarray,
    num_keys: int,
    key_tuple_size: int,
    distractor_mode: SymbolicDistractorMode,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_keys < 2:
        raise ValueError("--num-keys must be at least 2")
    keys = rng.integers(0, num_keys, size=(num_records, key_tuple_size), dtype=np.int64)
    query_keys = np.asarray(query_keys, dtype=np.int64)
    for record_idx in range(num_records):
        if record_idx == symbolic_record:
            continue
        if distractor_mode == "partial" and key_tuple_size >= 2:
            keys[record_idx] = query_keys.copy()
            keep_component = int(rng.integers(0, key_tuple_size))
            for component in range(key_tuple_size):
                if component == keep_component:
                    continue
                value = int(rng.integers(0, num_keys - 1))
                if value >= int(query_keys[component]):
                    value += 1
                keys[record_idx, component] = value
        else:
            while np.array_equal(keys[record_idx], query_keys):
                keys[record_idx] = rng.integers(
                    0,
                    num_keys,
                    size=key_tuple_size,
                    dtype=np.int64,
                )
    keys[symbolic_record] = query_keys
    return keys


def add_structural_motif(
    rec: int,
    edges: list[tuple[int, int]],
    add_node,
    motif_type: int,
    motif_style: MotifStyle,
) -> None:
    nodes = [add_node(MOTIF_ROLE) for _ in range(5)]
    a, b, c, d, e = nodes
    motif_type = int(motif_type) % 4
    if motif_style == "branch":
        if motif_type == 0:
            motif_edges = [(rec, a), (a, b), (b, c), (c, d), (d, e)]
        elif motif_type == 1:
            motif_edges = [(rec, a), (rec, b), (a, c), (a, d), (b, e)]
        elif motif_type == 2:
            motif_edges = [(rec, a), (rec, b), (rec, c), (c, d), (c, e)]
        else:
            motif_edges = [(rec, a), (rec, b), (rec, c), (rec, d), (d, e)]
    else:
        if motif_type == 0:
            motif_edges = [(rec, a), (a, b), (a, c), (b, d), (c, e)]
        elif motif_type == 1:
            motif_edges = [(rec, a), (a, b), (b, c), (b, d), (d, e)]
        elif motif_type == 2:
            motif_edges = [(rec, a), (a, b), (b, c), (c, d), (c, e)]
        else:
            motif_edges = [(rec, a), (a, b), (a, c), (a, d), (d, e)]
    edges.extend(motif_edges)


def make_record_motif_types(
    num_records: int,
    structural_record: int,
    query_type: int,
    num_motif_types: int,
    rng: np.random.Generator,
) -> np.ndarray:
    motif_types = rng.integers(0, num_motif_types, size=num_records, dtype=np.int64)
    distractors = np.asarray(
        [idx for idx in range(num_motif_types) if idx != query_type],
        dtype=np.int64,
    )
    for record_idx in range(num_records):
        if record_idx == structural_record:
            motif_types[record_idx] = query_type
        else:
            motif_types[record_idx] = int(distractors[int(rng.integers(0, len(distractors)))])
    return motif_types


def make_graphworld_example(
    num_records: int,
    rng: np.random.Generator,
    num_keys: int,
    num_values: int,
    key_tuple_size: int,
    symbolic_distractor_mode: SymbolicDistractorMode,
    structural_channel: StructuralChannel,
    rwse_steps: int,
    avoid_target_overlap: bool,
    enable_structural_selector: bool,
    enable_query_key: bool,
    graph_min_nodes: int,
    graph_max_nodes: int,
    graph_family: GraphFamily,
    structural_bridge_fraction: float,
    symbolic_partial_fraction: float,
    graphworld_structural_label: GraphworldStructuralLabel,
    graphworld_distance_classes: int,
) -> GraphExample:
    min_backbone_nodes = max(graph_min_nodes, num_records + 3)
    max_backbone_nodes = max(graph_max_nodes, min_backbone_nodes)
    backbone_n = int(rng.integers(min_backbone_nodes, max_backbone_nodes + 1))
    backbone_edges, resolved_family = graphworld_backbone_edges(backbone_n, graph_family, rng)
    backbone_adj = adjacency_from_edges(backbone_n, backbone_edges)
    backbone_dist = all_pairs_distances(backbone_adj)

    structural_record = 0
    anchor_base = int(rng.integers(0, backbone_n))
    candidate_base = np.asarray([], dtype=np.int64)
    structural_easy = False
    structural_distance = 1
    if enable_structural_selector:
        for _ in range(128):
            anchor_base = int(rng.integers(0, backbone_n))
            dist = backbone_dist[anchor_base]
            if graphworld_structural_label == "distance_bin":
                target_distance = int(rng.integers(1, graphworld_distance_classes + 1))
                target_pool = np.flatnonzero(dist == target_distance)
                distractor_pool = np.asarray(
                    [
                        idx
                        for idx in range(backbone_n)
                        if idx != anchor_base and dist[idx] > 0
                    ],
                    dtype=np.int64,
                )
            else:
                use_easy = rng.random() < structural_bridge_fraction
                target_distance = 1 if use_easy else 2
                target_pool = np.flatnonzero(dist == target_distance)
                distractor_pool = np.flatnonzero(dist > target_distance)
                distractor_pool = distractor_pool[distractor_pool != anchor_base]
            if target_pool.size < 1 or distractor_pool.size < num_records - 1:
                continue
            target_base = int(target_pool[int(rng.integers(0, target_pool.size))])
            distractor_pool = distractor_pool[distractor_pool != target_base]
            if distractor_pool.size < num_records - 1:
                continue
            distractors = rng.choice(distractor_pool, size=num_records - 1, replace=False)
            candidate_base = np.concatenate([[target_base], distractors]).astype(np.int64)
            rng.shuffle(candidate_base)
            structural_record = int(np.flatnonzero(candidate_base == target_base)[0])
            structural_easy = target_distance == 1
            structural_distance = target_distance
            break
    if candidate_base.size == 0:
        pool = np.asarray([idx for idx in range(backbone_n) if idx != anchor_base], dtype=np.int64)
        candidate_base = rng.choice(pool, size=num_records, replace=False)
        dists = backbone_dist[anchor_base, candidate_base]
        structural_record = int(np.argmin(dists))
        structural_distance = int(dists[structural_record])

    if avoid_target_overlap:
        choices = [idx for idx in range(num_records) if idx != structural_record]
        symbolic_record = int(choices[int(rng.integers(0, len(choices)))])
    else:
        symbolic_record = int(rng.integers(0, num_records))

    example_key_tuple_size = key_tuple_size
    effective_symbolic_distractor_mode = symbolic_distractor_mode
    if enable_query_key and symbolic_distractor_mode == "mixed":
        use_partial = rng.random() < symbolic_partial_fraction and key_tuple_size >= 2
        effective_symbolic_distractor_mode = "partial" if use_partial else "random"
        example_key_tuple_size = key_tuple_size if use_partial else 1
    query_keys = (
        rng.integers(0, num_keys, size=example_key_tuple_size, dtype=np.int64)
        if enable_query_key
        else np.zeros(example_key_tuple_size, dtype=np.int64)
    )
    query_key = int(query_keys[0]) if query_keys.size else 0
    symbolic_partial_distractors = (
        effective_symbolic_distractor_mode == "partial" and example_key_tuple_size >= 2
    )
    if enable_query_key:
        record_keys = make_record_keys(
            num_records=num_records,
            symbolic_record=symbolic_record,
            query_keys=query_keys,
            num_keys=num_keys,
            key_tuple_size=example_key_tuple_size,
            distractor_mode=effective_symbolic_distractor_mode,
            rng=rng,
        )
    else:
        record_keys = np.zeros((num_records, example_key_tuple_size), dtype=np.int64)
    record_values = rng.integers(0, num_values, size=num_records, dtype=np.int64)
    structural_y = int(record_values[structural_record])
    if graphworld_structural_label == "distance_bin":
        structural_y = max(0, min(graphworld_distance_classes - 1, structural_distance - 1))

    role: list[int] = [CLS_ROLE] + [MOTIF_ROLE for _ in range(backbone_n)]
    symbol: list[int] = [0] * (backbone_n + 1)
    if enable_query_key:
        symbol[0] = key_symbol(query_key)
    edges = offset_edges(backbone_edges, offset=1)
    record_nodes = (candidate_base + 1).astype(np.int64)
    value_nodes: list[int] = []

    if enable_structural_selector:
        role[anchor_base + 1] = ANCHOR_ROLE
    for rec in record_nodes:
        role[int(rec)] = RECORD_ROLE
    if enable_structural_selector and graphworld_structural_label == "distance_bin":
        role[int(record_nodes[structural_record])] = STRUCT_TARGET_ROLE

    def add_node(role_id: int, symbol_id: int = 0) -> int:
        node_id = len(role)
        role.append(role_id)
        symbol.append(symbol_id)
        return node_id

    if enable_query_key:
        for query_component in query_keys:
            query_node = add_node(QUERY_ROLE, key_symbol(int(query_component)))
            edges.append((0, query_node))

    for record_idx, rec in enumerate(record_nodes):
        val = add_node(VALUE_ROLE, value_symbol(int(record_values[record_idx]), num_keys))
        value_nodes.append(val)
        edges.append((int(rec), val))
        if enable_query_key:
            symbol[int(rec)] = key_symbol(int(record_keys[record_idx][0]))
            for key_component in record_keys[record_idx][1:]:
                key_node = add_node(KEY_ROLE, key_symbol(int(key_component)))
                edges.append((int(rec), key_node))

    n = len(role)
    adj = adjacency_from_edges(n, edges)
    if structural_channel == "degree":
        degree = adj.sum(axis=1, keepdims=True)
        pe = (degree / max(1.0, float(n - 1))).astype(np.float32)
    elif structural_channel in {"rwse", "rwse_spd_bias"}:
        pe = compute_rwse(adj, steps=rwse_steps).astype(np.float32)
    else:
        pe = np.zeros((n, pe_dim_for_channel(structural_channel, rwse_steps)), dtype=np.float32)

    if uses_spd_bias(structural_channel):
        spd = all_pairs_distances(adj)
    else:
        spd = np.zeros((n, n), dtype=np.int64)

    return GraphExample(
        role=np.asarray(role, dtype=np.int64),
        symbol=np.asarray(symbol, dtype=np.int64),
        adj=adj,
        pe=pe,
        spd=spd,
        n=n,
        record_nodes=record_nodes,
        value_nodes=np.asarray(value_nodes, dtype=np.int64),
        structural_record=structural_record,
        symbolic_record=symbolic_record,
        structural_y=structural_y,
        symbolic_y=int(record_values[symbolic_record]),
        dataset_style="graphworld",
        graph_family=resolved_family,
        anchor_node=anchor_base + 1 if enable_structural_selector else -1,
        structural_query_type=int(structural_distance),
        structural_bridge=structural_easy,
        symbolic_partial_distractors=symbolic_partial_distractors,
        record_motif_types=np.zeros(num_records, dtype=np.int64),
        query_key=query_key,
        query_keys=query_keys,
        record_keys=record_keys,
        record_values=record_values,
    )


def make_example(
    num_records: int,
    rng: np.random.Generator,
    num_keys: int,
    num_values: int,
    key_tuple_size: int,
    symbolic_distractor_mode: SymbolicDistractorMode,
    num_motif_types: int,
    motif_style: MotifStyle,
    structural_bridge_fraction: float,
    symbolic_partial_fraction: float,
    structural_channel: StructuralChannel,
    rwse_steps: int,
    avoid_target_overlap: bool,
    add_record_path: bool,
    enable_structural_selector: bool,
    enable_query_key: bool,
    dataset_style: DatasetStyle,
    graph_min_nodes: int,
    graph_max_nodes: int,
    graph_family: GraphFamily,
    graphworld_structural_label: GraphworldStructuralLabel,
    graphworld_distance_classes: int,
) -> GraphExample:
    if dataset_style == "graphworld":
        return make_graphworld_example(
            num_records=num_records,
            rng=rng,
            num_keys=num_keys,
            num_values=num_values,
            key_tuple_size=key_tuple_size,
            symbolic_distractor_mode=symbolic_distractor_mode,
            structural_channel=structural_channel,
            rwse_steps=rwse_steps,
            avoid_target_overlap=avoid_target_overlap,
            enable_structural_selector=enable_structural_selector,
            enable_query_key=enable_query_key,
            graph_min_nodes=graph_min_nodes,
            graph_max_nodes=graph_max_nodes,
            graph_family=graph_family,
            structural_bridge_fraction=structural_bridge_fraction,
            symbolic_partial_fraction=symbolic_partial_fraction,
            graphworld_structural_label=graphworld_structural_label,
            graphworld_distance_classes=graphworld_distance_classes,
        )

    if num_records < 2:
        raise ValueError("Need at least two records")
    if avoid_target_overlap and num_records < 2:
        raise ValueError("Need at least two records when avoiding target overlap")

    structural_record = int(rng.integers(0, num_records))
    if avoid_target_overlap:
        choices = [idx for idx in range(num_records) if idx != structural_record]
        symbolic_record = int(choices[int(rng.integers(0, len(choices)))])
    else:
        symbolic_record = int(rng.integers(0, num_records))
    structural_query_type = (
        int(rng.integers(0, num_motif_types)) if enable_structural_selector else 0
    )
    structural_bridge = bool(
        enable_structural_selector and rng.random() < structural_bridge_fraction
    )
    record_motif_types = (
        make_record_motif_types(
            num_records=num_records,
            structural_record=structural_record,
            query_type=structural_query_type,
            num_motif_types=num_motif_types,
            rng=rng,
        )
        if enable_structural_selector
        else np.zeros(num_records, dtype=np.int64)
    )
    example_key_tuple_size = key_tuple_size
    effective_symbolic_distractor_mode = symbolic_distractor_mode
    if enable_query_key and symbolic_distractor_mode == "mixed":
        use_partial = rng.random() < symbolic_partial_fraction and key_tuple_size >= 2
        effective_symbolic_distractor_mode = "partial" if use_partial else "random"
        example_key_tuple_size = key_tuple_size if use_partial else 1
    query_keys = (
        rng.integers(0, num_keys, size=example_key_tuple_size, dtype=np.int64)
        if enable_query_key
        else np.zeros(example_key_tuple_size, dtype=np.int64)
    )
    query_key = int(query_keys[0]) if query_keys.size else 0
    symbolic_partial_distractors = (
        effective_symbolic_distractor_mode == "partial" and example_key_tuple_size >= 2
    )
    if enable_query_key:
        record_keys = make_record_keys(
            num_records=num_records,
            symbolic_record=symbolic_record,
            query_keys=query_keys,
            num_keys=num_keys,
            key_tuple_size=example_key_tuple_size,
            distractor_mode=effective_symbolic_distractor_mode,
            rng=rng,
        )
    else:
        record_keys = np.zeros((num_records, example_key_tuple_size), dtype=np.int64)
    record_values = rng.integers(0, num_values, size=num_records, dtype=np.int64)

    role: list[int] = [CLS_ROLE]
    symbol: list[int] = [0]
    if enable_query_key:
        symbol[0] = key_symbol(query_key)
    edges: list[tuple[int, int]] = []
    record_nodes: list[int] = []
    value_nodes: list[int] = []

    def add_node(role_id: int, symbol_id: int = 0) -> int:
        node_id = len(role)
        role.append(role_id)
        symbol.append(symbol_id)
        return node_id

    if enable_query_key:
        for query_component in query_keys:
            query_node = add_node(QUERY_ROLE, key_symbol(int(query_component)))
            edges.append((0, query_node))
    if enable_structural_selector:
        add_structural_motif(
            rec=0,
            edges=edges,
            add_node=add_node,
            motif_type=structural_query_type,
            motif_style=motif_style,
        )

    for record_idx in range(num_records):
        rec = add_node(RECORD_ROLE, 0)
        val = add_node(VALUE_ROLE, value_symbol(int(record_values[record_idx]), num_keys))
        record_nodes.append(rec)
        value_nodes.append(val)
        edges.append((0, rec))
        edges.append((rec, val))
        if enable_query_key:
            for key_component in record_keys[record_idx]:
                key_node = add_node(KEY_ROLE, key_symbol(int(key_component)))
                edges.append((rec, key_node))

    if add_record_path:
        order = np.arange(num_records)
        rng.shuffle(order)
        for left, right in zip(order[:-1], order[1:]):
            edges.append((record_nodes[int(left)], record_nodes[int(right)]))

    for record_idx, rec in enumerate(record_nodes):
        add_structural_motif(
            rec=rec,
            edges=edges,
            add_node=add_node,
            motif_type=int(record_motif_types[record_idx]),
            motif_style=motif_style,
        )
    if structural_bridge:
        bridge = add_node(MOTIF_ROLE)
        edges.append((0, bridge))
        edges.append((record_nodes[structural_record], bridge))

    n = len(role)
    adj = adjacency_from_edges(n, edges)
    if structural_channel == "degree":
        degree = adj.sum(axis=1, keepdims=True)
        pe = (degree / max(1.0, float(n - 1))).astype(np.float32)
    elif structural_channel in {"rwse", "rwse_spd_bias"}:
        pe = compute_rwse(adj, steps=rwse_steps).astype(np.float32)
    else:
        pe = np.zeros((n, pe_dim_for_channel(structural_channel, rwse_steps)), dtype=np.float32)

    if uses_spd_bias(structural_channel):
        spd = all_pairs_distances(adj)
    else:
        spd = np.zeros((n, n), dtype=np.int64)

    return GraphExample(
        role=np.asarray(role, dtype=np.int64),
        symbol=np.asarray(symbol, dtype=np.int64),
        adj=adj,
        pe=pe,
        spd=spd,
        n=n,
        record_nodes=np.asarray(record_nodes, dtype=np.int64),
        value_nodes=np.asarray(value_nodes, dtype=np.int64),
        structural_record=structural_record,
        symbolic_record=symbolic_record,
        structural_y=int(record_values[structural_record]),
        symbolic_y=int(record_values[symbolic_record]),
        dataset_style="record",
        graph_family="record",
        anchor_node=-1,
        structural_query_type=structural_query_type,
        structural_bridge=structural_bridge,
        symbolic_partial_distractors=symbolic_partial_distractors,
        record_motif_types=record_motif_types,
        query_key=query_key,
        query_keys=query_keys,
        record_keys=record_keys,
        record_values=record_values,
    )


def generate_examples(
    count: int,
    min_records: int,
    max_records: int,
    seed: int,
    num_keys: int,
    num_values: int,
    key_tuple_size: int,
    symbolic_distractor_mode: SymbolicDistractorMode,
    num_motif_types: int,
    motif_style: MotifStyle,
    structural_bridge_fraction: float,
    symbolic_partial_fraction: float,
    structural_channel: StructuralChannel,
    rwse_steps: int,
    avoid_target_overlap: bool,
    add_record_path: bool,
    enable_structural_selector: bool,
    enable_query_key: bool,
    dataset_style: DatasetStyle,
    graph_min_nodes: int,
    graph_max_nodes: int,
    graph_family: GraphFamily,
    graphworld_structural_label: GraphworldStructuralLabel,
    graphworld_distance_classes: int,
) -> list[GraphExample]:
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(count):
        num_records = int(rng.integers(min_records, max_records + 1))
        graphs.append(
            make_example(
                num_records=num_records,
                rng=rng,
                num_keys=num_keys,
                num_values=num_values,
                key_tuple_size=key_tuple_size,
                symbolic_distractor_mode=symbolic_distractor_mode,
                num_motif_types=num_motif_types,
                motif_style=motif_style,
                structural_bridge_fraction=structural_bridge_fraction,
                symbolic_partial_fraction=symbolic_partial_fraction,
                structural_channel=structural_channel,
                rwse_steps=rwse_steps,
                avoid_target_overlap=avoid_target_overlap,
                add_record_path=add_record_path,
                enable_structural_selector=enable_structural_selector,
                enable_query_key=enable_query_key,
                dataset_style=dataset_style,
                graph_min_nodes=graph_min_nodes,
                graph_max_nodes=graph_max_nodes,
                graph_family=graph_family,
                graphworld_structural_label=graphworld_structural_label,
                graphworld_distance_classes=graphworld_distance_classes,
            )
        )
    return graphs


def iter_generated_batches(
    count: int,
    batch_size: int,
    min_records: int,
    max_records: int,
    seed: int,
    num_keys: int,
    num_values: int,
    key_tuple_size: int,
    symbolic_distractor_mode: SymbolicDistractorMode,
    num_motif_types: int,
    motif_style: MotifStyle,
    structural_bridge_fraction: float,
    symbolic_partial_fraction: float,
    structural_channel: StructuralChannel,
    rwse_steps: int,
    avoid_target_overlap: bool,
    add_record_path: bool,
    enable_structural_selector: bool,
    enable_query_key: bool,
    dataset_style: DatasetStyle,
    graph_min_nodes: int,
    graph_max_nodes: int,
    graph_family: GraphFamily,
    graphworld_structural_label: GraphworldStructuralLabel,
    graphworld_distance_classes: int,
    spd_cap: int,
    device: torch.device,
) -> Iterable[Batch]:
    rng = np.random.default_rng(seed)
    remaining = count
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        examples = []
        for _ in range(this_batch):
            num_records = int(rng.integers(min_records, max_records + 1))
            examples.append(
                make_example(
                    num_records=num_records,
                    rng=rng,
                    num_keys=num_keys,
                    num_values=num_values,
                    key_tuple_size=key_tuple_size,
                    symbolic_distractor_mode=symbolic_distractor_mode,
                    num_motif_types=num_motif_types,
                    motif_style=motif_style,
                    structural_bridge_fraction=structural_bridge_fraction,
                    symbolic_partial_fraction=symbolic_partial_fraction,
                    structural_channel=structural_channel,
                    rwse_steps=rwse_steps,
                    avoid_target_overlap=avoid_target_overlap,
                    add_record_path=add_record_path,
                    enable_structural_selector=enable_structural_selector,
                    enable_query_key=enable_query_key,
                    dataset_style=dataset_style,
                    graph_min_nodes=graph_min_nodes,
                    graph_max_nodes=graph_max_nodes,
                    graph_family=graph_family,
                    graphworld_structural_label=graphworld_structural_label,
                    graphworld_distance_classes=graphworld_distance_classes,
                )
            )
        remaining -= this_batch
        yield collate_examples(examples, spd_cap=spd_cap).to(device)


def iter_static_batches(
    examples: list[GraphExample],
    batch_size: int,
    spd_cap: int,
    device: torch.device,
) -> Iterable[Batch]:
    for start in range(0, len(examples), batch_size):
        yield collate_examples(examples[start : start + batch_size], spd_cap=spd_cap).to(device)


def collate_examples(examples: list[GraphExample], spd_cap: int) -> Batch:
    batch_size = len(examples)
    max_n = max(ex.n for ex in examples)
    max_records = max(len(ex.record_nodes) for ex in examples)
    pe_dim = examples[0].pe.shape[1]

    role = np.full((batch_size, max_n), fill_value=PAD_ROLE, dtype=np.int64)
    symbol = np.zeros((batch_size, max_n), dtype=np.int64)
    mask = np.zeros((batch_size, max_n), dtype=np.bool_)
    adj_norm = np.zeros((batch_size, max_n, max_n), dtype=np.float32)
    pe = np.zeros((batch_size, max_n, pe_dim), dtype=np.float32)
    spd = np.full((batch_size, max_n, max_n), fill_value=spd_cap + 1, dtype=np.int64)
    record_nodes = np.zeros((batch_size, max_records), dtype=np.int64)
    value_nodes = np.zeros((batch_size, max_records), dtype=np.int64)
    record_mask = np.zeros((batch_size, max_records), dtype=np.bool_)
    structural_node = np.zeros(batch_size, dtype=np.int64)
    symbolic_node = np.zeros(batch_size, dtype=np.int64)
    structural_value_node = np.zeros(batch_size, dtype=np.int64)
    symbolic_value_node = np.zeros(batch_size, dtype=np.int64)
    structural_y = np.zeros(batch_size, dtype=np.int64)
    symbolic_y = np.zeros(batch_size, dtype=np.int64)

    for idx, ex in enumerate(examples):
        n = ex.n
        degree = np.maximum(ex.adj.sum(axis=1, keepdims=True), 1.0)
        role[idx, :n] = ex.role
        symbol[idx, :n] = ex.symbol
        mask[idx, :n] = True
        adj_norm[idx, :n, :n] = ex.adj / degree
        pe[idx, :n, :] = ex.pe
        spd[idx, :n, :n] = np.clip(ex.spd, 0, spd_cap)
        records = len(ex.record_nodes)
        record_nodes[idx, :records] = ex.record_nodes
        value_nodes[idx, :records] = ex.value_nodes
        record_mask[idx, :records] = True
        structural_node[idx] = ex.record_nodes[ex.structural_record]
        symbolic_node[idx] = ex.record_nodes[ex.symbolic_record]
        structural_value_node[idx] = ex.value_nodes[ex.structural_record]
        symbolic_value_node[idx] = ex.value_nodes[ex.symbolic_record]
        structural_y[idx] = ex.structural_y
        symbolic_y[idx] = ex.symbolic_y

    return Batch(
        role=torch.from_numpy(role),
        symbol=torch.from_numpy(symbol),
        mask=torch.from_numpy(mask),
        adj_norm=torch.from_numpy(adj_norm),
        pe=torch.from_numpy(pe),
        spd=torch.from_numpy(spd),
        record_nodes=torch.from_numpy(record_nodes),
        record_mask=torch.from_numpy(record_mask),
        value_nodes=torch.from_numpy(value_nodes),
        structural_node=torch.from_numpy(structural_node),
        symbolic_node=torch.from_numpy(symbolic_node),
        structural_value_node=torch.from_numpy(structural_value_node),
        symbolic_value_node=torch.from_numpy(symbolic_value_node),
        structural_y=torch.from_numpy(structural_y),
        symbolic_y=torch.from_numpy(symbolic_y),
    )


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
        attn_for_metrics = attn
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, max_n, self.dim)
        out = self.out(out) * mask.unsqueeze(-1).to(h.dtype)
        metric_logits = logits.masked_fill(~pair_mask, 0.0)
        return out, metric_logits, attn_for_metrics


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


class GraphGPSDecoupledModel(nn.Module):
    def __init__(
        self,
        depth: int,
        hidden_dim: int,
        num_heads: int,
        symbol_vocab: int,
        pe_dim: int,
        num_values: int,
        dropout: float,
        attn_dropout: float,
        use_spd_bias: bool,
        spd_cap: int,
    ) -> None:
        super().__init__()
        self.role_emb = nn.Embedding(NUM_ROLES, hidden_dim)
        self.symbol_emb = nn.Embedding(symbol_vocab, hidden_dim)
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
        self.structural_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_values),
        )
        self.symbolic_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_values),
        )

    def forward(
        self,
        batch: Batch,
        collect_attention: bool = False,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        h = self.role_emb(batch.role) + self.symbol_emb(batch.symbol)
        if self.pe_proj is not None:
            h = h + self.pe_proj(batch.pe)
        h = h * batch.mask.unsqueeze(-1).to(h.dtype)

        layers = []
        for layer in self.layers:
            h, logits, attn = layer(h, batch)
            if collect_attention:
                stored_logits = logits if self.training else logits.detach()
                stored_attn = attn if self.training else attn.detach()
                layers.append(
                    {
                        "logits": stored_logits,
                        "attn": stored_attn,
                        "node_mask": batch.mask.detach(),
                    }
                )

        batch_idx = torch.arange(h.size(0), device=h.device)
        structural_readout = h[batch_idx, batch.structural_node]
        symbolic_readout = h[:, 0, :]
        return {
            "structural": self.structural_head(structural_readout),
            "symbolic": self.symbolic_head(symbolic_readout),
        }, layers


def selector_flags_for_task(task: TaskName, experiment_focus: str) -> tuple[bool, bool]:
    if experiment_focus == "shared" or task == "dual":
        return True, True
    if task == "structural":
        return True, False
    if task == "symbolic":
        return False, True
    raise ValueError(f"Unknown task: {task}")


def batch_loss(
    outputs: dict[str, torch.Tensor],
    batch: Batch,
    task: TaskName,
    dual_symbolic_loss_weight: float = 1.0,
) -> torch.Tensor:
    losses = []
    if task in {"structural", "dual"}:
        losses.append(F.cross_entropy(outputs["structural"], batch.structural_y))
    if task in {"symbolic", "dual"}:
        symbolic_loss = F.cross_entropy(outputs["symbolic"], batch.symbolic_y)
        if task == "dual":
            symbolic_loss = dual_symbolic_loss_weight * symbolic_loss
        losses.append(symbolic_loss)
    return torch.stack(losses).mean()


def attention_specialisation_loss(
    layers: list[dict[str, torch.Tensor]],
    batch: Batch,
    task: TaskName,
) -> torch.Tensor:
    if not layers:
        return torch.zeros((), device=batch.role.device)
    attn = layers[-1]["attn"]
    cls_rows = attn[:, :, 0, :]
    batch_idx = torch.arange(batch.role.size(0), device=batch.role.device)
    eps = 1.0e-8

    losses = []
    structural_mass = cls_rows[batch_idx, :, batch.structural_node]
    symbolic_mass = cls_rows[batch_idx, :, batch.symbolic_node]
    if task in {"structural", "dual"}:
        losses.append(-torch.log(structural_mass.max(dim=1).values.clamp_min(eps)).mean())
    if task in {"symbolic", "dual"}:
        losses.append(-torch.log(symbolic_mass.max(dim=1).values.clamp_min(eps)).mean())

    if task == "dual" and structural_mass.size(1) > 1:
        structural_head_dist = structural_mass / structural_mass.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(eps)
        symbolic_head_dist = symbolic_mass / symbolic_mass.sum(dim=1, keepdim=True).clamp_min(eps)
        losses.append((structural_head_dist * symbolic_head_dist).sum(dim=1).mean())

    return torch.stack(losses).mean()


@dataclass
class EvalStats:
    loss: float
    structural_acc: float
    symbolic_acc: float
    active_acc: float
    both_acc: float


@torch.no_grad()
def evaluate(
    model: GraphGPSDecoupledModel,
    examples: list[GraphExample],
    task: TaskName,
    batch_size: int,
    spd_cap: int,
    device: torch.device,
) -> EvalStats:
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    structural_correct = 0
    symbolic_correct = 0
    both_correct = 0

    for batch in iter_static_batches(
        examples,
        batch_size=batch_size,
        spd_cap=spd_cap,
        device=device,
    ):
        outputs, _ = model(batch, collect_attention=False)
        loss = batch_loss(outputs, batch, task)
        structural_pred = outputs["structural"].argmax(dim=-1)
        symbolic_pred = outputs["symbolic"].argmax(dim=-1)
        structural_ok = structural_pred == batch.structural_y
        symbolic_ok = symbolic_pred == batch.symbolic_y
        graphs = int(batch.role.size(0))

        total_graphs += graphs
        total_loss += float(loss.item()) * graphs
        structural_correct += int(structural_ok.sum().item())
        symbolic_correct += int(symbolic_ok.sum().item())
        both_correct += int((structural_ok & symbolic_ok).sum().item())

    structural_acc = structural_correct / max(1, total_graphs)
    symbolic_acc = symbolic_correct / max(1, total_graphs)
    if task == "structural":
        active_acc = structural_acc
    elif task == "symbolic":
        active_acc = symbolic_acc
    else:
        active_acc = 0.5 * (structural_acc + symbolic_acc)
    return EvalStats(
        loss=total_loss / max(1, total_graphs),
        structural_acc=structural_acc,
        symbolic_acc=symbolic_acc,
        active_acc=active_acc,
        both_acc=both_correct / max(1, total_graphs),
    )


def pair_mask_from_layer(layer: dict[str, torch.Tensor]) -> torch.Tensor:
    node_mask = layer["node_mask"]
    return node_mask[:, None, :, None] & node_mask[:, None, None, :]


def transform_pair_reference(z: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    batch_size, heads, max_n, _ = z.shape
    idx = perm_pos.to(z.device)
    row_idx = idx[:, None, :, None].expand(batch_size, heads, max_n, max_n)
    z_rows = torch.gather(z, dim=2, index=row_idx)
    col_idx = idx[:, None, None, :].expand(batch_size, heads, max_n, max_n)
    return torch.gather(z_rows, dim=3, index=col_idx)


def row_center_tensor(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.size(1) == 1 and z.size(1) != 1:
        mask = mask.expand(-1, z.size(1), -1, -1)
    z0 = torch.where(mask, torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(z.dtype)
    mean = z0.sum(dim=-1, keepdim=True) / denom
    return torch.where(mask, z0 - mean, torch.zeros_like(z0))


def cosine_by_head(
    u: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    *,
    center_rows: bool,
) -> torch.Tensor:
    if mask.size(1) == 1 and u.size(1) != 1:
        mask = mask.expand(-1, u.size(1), -1, -1)
    if center_rows:
        u0 = row_center_tensor(u, mask)
        v0 = row_center_tensor(v, mask)
    else:
        u0 = torch.where(mask, torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        v0 = torch.where(mask, torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    dims = (0, 2, 3)
    num = (u0 * v0).sum(dim=dims)
    u_norm = torch.sqrt((u0 * u0).sum(dim=dims).clamp_min(0.0))
    v_norm = torch.sqrt((v0 * v0).sum(dim=dims).clamp_min(0.0))
    den = u_norm * v_norm
    return (num / den.clamp_min(1.0e-12)).detach().cpu()


def make_symbol_permutation(batch: Batch, generator: torch.Generator) -> tuple[Batch, torch.Tensor]:
    batch_size, max_n = batch.symbol.shape
    perm_pos = torch.zeros((batch_size, max_n), dtype=torch.long, device=batch.symbol.device)
    symbol_perm = batch.symbol.clone()
    for graph_idx in range(batch_size):
        n = int(batch.mask[graph_idx].sum().item())
        perm = torch.randperm(n, generator=generator).to(batch.symbol.device)
        perm_pos[graph_idx, :n] = perm
        if n < max_n:
            perm_pos[graph_idx, n:] = torch.arange(n, max_n, device=batch.symbol.device)
        symbol_perm[graph_idx, :n] = batch.symbol[graph_idx, perm]
    return replace(batch, symbol=symbol_perm), perm_pos


@torch.no_grad()
def compute_symbol_permutation_metrics(
    model: GraphGPSDecoupledModel,
    examples: list[GraphExample],
    split: str,
    depth: int,
    task: TaskName,
    batch_size: int,
    spd_cap: int,
    num_perms: int,
    metric_graphs: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    model.eval()
    selected = examples[: min(metric_graphs, len(examples))]
    rows: list[dict[str, float | int | str]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for batch_idx, batch in enumerate(
        iter_static_batches(selected, batch_size=batch_size, spd_cap=spd_cap, device=device)
    ):
        _, clean_layers = model(batch, collect_attention=True)
        for perm_idx in range(num_perms):
            variant_batch, perm_pos = make_symbol_permutation(batch, generator)
            _, variant_layers = model(variant_batch, collect_attention=True)
            for layer_idx, (clean, variant) in enumerate(zip(clean_layers, variant_layers)):
                a_clean = clean["attn"]
                a_var = variant["attn"]
                mask = pair_mask_from_layer(clean).to(device=a_clean.device)
                a_ref = transform_pair_reference(a_clean, perm_pos)
                structural = cosine_by_head(a_var, a_clean, mask, center_rows=False)
                symbolic = cosine_by_head(a_var, a_ref, mask, center_rows=False)
                centered_structural = cosine_by_head(a_var, a_clean, mask, center_rows=True)
                centered_symbolic = cosine_by_head(a_var, a_ref, mask, center_rows=True)
                for head in range(int(structural.numel())):
                    common = {
                        "depth": depth,
                        "task": task,
                        "split": split,
                        "batch": batch_idx,
                        "perm": perm_idx,
                        "layer": layer_idx,
                        "head": head,
                    }
                    rows.append(
                        {
                            **common,
                            "metric": "structural_score",
                            "score": float(structural[head]),
                            "raw_cos": float(structural[head]),
                        }
                    )
                    rows.append(
                        {
                            **common,
                            "metric": "symbolic_score",
                            "score": float(symbolic[head]),
                            "raw_cos": float(symbolic[head]),
                        }
                    )
                    rows.append(
                        {
                            **common,
                            "metric": "centered_structural_score",
                            "score": float(centered_structural[head]),
                            "raw_cos": float(centered_structural[head]),
                        }
                    )
                    rows.append(
                        {
                            **common,
                            "metric": "centered_symbolic_score",
                            "score": float(centered_symbolic[head]),
                            "raw_cos": float(centered_symbolic[head]),
                        }
                    )
    return rows


@torch.no_grad()
def compute_attention_target_metrics(
    model: GraphGPSDecoupledModel,
    examples: list[GraphExample],
    split: str,
    depth: int,
    task: TaskName,
    batch_size: int,
    spd_cap: int,
    metric_graphs: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    model.eval()
    selected = examples[: min(metric_graphs, len(examples))]
    buckets: dict[tuple[int, int, str], list[float]] = {}

    for batch in iter_static_batches(
        selected,
        batch_size=batch_size,
        spd_cap=spd_cap,
        device=device,
    ):
        _, layers = model(batch, collect_attention=True)
        batch_idx = torch.arange(batch.role.size(0), device=device)
        for layer_idx, layer in enumerate(layers):
            attn = layer["attn"]
            num_heads = int(attn.size(1))
            for head in range(num_heads):
                cls_rows = attn[:, head, 0, :]
                structural_mass = cls_rows[batch_idx, batch.structural_node]
                symbolic_mass = cls_rows[batch_idx, batch.symbolic_node]
                row = cls_rows.masked_fill(~batch.mask, 0.0)
                entropy = -(row * row.clamp_min(1.0e-12).log()).sum(dim=-1)
                normalizer = batch.mask.sum(dim=-1).float().clamp_min(2).log()
                entropy = entropy / normalizer
                structural_value_mass = attn[
                    batch_idx,
                    head,
                    batch.structural_node,
                    batch.structural_value_node,
                ]
                symbolic_value_mass = attn[
                    batch_idx,
                    head,
                    batch.symbolic_node,
                    batch.symbolic_value_node,
                ]
                for metric, values in (
                    ("cls_to_structural_record", structural_mass),
                    ("cls_to_symbolic_record", symbolic_mass),
                    ("cls_attention_entropy", entropy),
                    ("structural_record_to_value", structural_value_mass),
                    ("symbolic_record_to_value", symbolic_value_mass),
                ):
                    buckets.setdefault((layer_idx, head, metric), []).extend(
                        values.detach().cpu().numpy().astype(float).tolist()
                    )

    rows = []
    for (layer, head, metric), values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "depth": depth,
                "task": task,
                "split": split,
                "layer": layer,
                "head": head,
                "metric": metric,
                "score_mean": float(arr.mean()),
                "score_std": float(arr.std(ddof=0)),
                "n": int(arr.size),
            }
        )
    return rows


def summarize_metric_rows(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    buckets: dict[tuple[int, str, str, int, int, str], list[float]] = {}
    for row in rows:
        key = (
            int(row["depth"]),
            str(row["task"]),
            str(row["split"]),
            int(row["layer"]),
            int(row["head"]),
            str(row["metric"]),
        )
        buckets.setdefault(key, []).append(float(row["score"]))

    out = []
    for (depth, task, split, layer, head, metric), values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=np.float64)
        out.append(
            {
                "depth": depth,
                "task": task,
                "split": split,
                "layer": layer,
                "head": head,
                "metric": metric,
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
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except Exception:
        return default


def format_stats(prefix: str, stats: EvalStats, task: TaskName) -> str:
    if task == "structural":
        focus = f"struct_acc={stats.structural_acc:.4f}"
    elif task == "symbolic":
        focus = f"sym_acc={stats.symbolic_acc:.4f}"
    else:
        focus = (
            f"struct_acc={stats.structural_acc:.4f} sym_acc={stats.symbolic_acc:.4f} "
            f"both={stats.both_acc:.4f}"
        )
    return f"{prefix}: loss={stats.loss:.4f} active={stats.active_acc:.4f} {focus}"


def is_solved(stats: EvalStats, task: TaskName, threshold: float) -> bool:
    if task == "structural":
        return stats.structural_acc >= threshold
    if task == "symbolic":
        return stats.symbolic_acc >= threshold
    return stats.structural_acc >= threshold and stats.symbolic_acc >= threshold


def train_depth(
    depth: int,
    args: argparse.Namespace,
    eval_sets: dict[str, list[GraphExample]],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict, list[dict], GraphGPSDecoupledModel]:
    set_seed(args.seed + depth * 997)
    pe_dim = pe_dim_for_channel(args.structural_channel, args.rwse_steps)
    model = GraphGPSDecoupledModel(
        depth=depth,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        symbol_vocab=symbol_vocab_size(args.num_keys, args.num_values),
        pe_dim=pe_dim,
        num_values=args.num_values,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        use_spd_bias=uses_spd_bias(args.structural_channel),
        spd_cap=args.spd_cap,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_dir = output_dir / f"depth_{depth}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict] = []
    best_state = None
    best_val_active = -1.0
    best_epoch = -1
    best_stats: dict[str, EvalStats] | None = None
    solved_hits = 0
    stale_evals = 0
    start_time = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_graphs = 0
        train_seed = args.seed + epoch * 1009 + depth * 9173
        for batch in iter_generated_batches(
            count=args.train_graphs_per_epoch,
            batch_size=args.batch_size,
            min_records=args.train_min_records,
            max_records=args.train_max_records,
            seed=train_seed,
            num_keys=args.num_keys,
            num_values=args.num_values,
            key_tuple_size=args.key_tuple_size,
            symbolic_distractor_mode=args.symbolic_distractor_mode,
            num_motif_types=args.num_motif_types,
            motif_style=args.motif_style,
            structural_bridge_fraction=args.structural_bridge_fraction,
            symbolic_partial_fraction=args.symbolic_partial_fraction,
            structural_channel=args.structural_channel,
            rwse_steps=args.rwse_steps,
            avoid_target_overlap=args.avoid_target_overlap,
            add_record_path=args.add_record_path,
            enable_structural_selector=args.enable_structural_selector,
            enable_query_key=args.enable_query_key,
            dataset_style=args.dataset_style,
            graph_min_nodes=args.graph_min_nodes,
            graph_max_nodes=args.graph_max_nodes,
            graph_family=args.graph_family,
            graphworld_structural_label=args.graphworld_structural_label,
            graphworld_distance_classes=args.graphworld_distance_classes,
            spd_cap=args.spd_cap,
            device=device,
        ):
            optimizer.zero_grad(set_to_none=True)
            use_attention_aux = args.specialisation_loss_weight > 0
            outputs, layers = model(batch, collect_attention=use_attention_aux)
            loss = batch_loss(
                outputs,
                batch,
                args.task,
                dual_symbolic_loss_weight=args.dual_symbolic_loss_weight,
            )
            if use_attention_aux:
                loss = loss + args.specialisation_loss_weight * attention_specialisation_loss(
                    layers,
                    batch,
                    args.task,
                )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            graphs = int(batch.role.size(0))
            epoch_loss += float(loss.item()) * graphs
            epoch_graphs += graphs

        if epoch % args.eval_every != 0 and epoch != args.max_epochs:
            continue

        val = evaluate(
            model,
            eval_sets["val"],
            args.task,
            args.eval_batch_size,
            args.spd_cap,
            device,
        )
        id_test = evaluate(
            model,
            eval_sets["id_test"],
            args.task,
            args.eval_batch_size,
            args.spd_cap,
            device,
        )
        ood_test = evaluate(
            model,
            eval_sets["ood_test"],
            args.task,
            args.eval_batch_size,
            args.spd_cap,
            device,
        )
        train_loss = epoch_loss / max(1, epoch_graphs)

        row = {
            "depth": depth,
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in asdict(val).items()},
            **{f"id_test_{k}": v for k, v in asdict(id_test).items()},
            **{f"ood_test_{k}": v for k, v in asdict(ood_test).items()},
            "elapsed_s": time.time() - start_time,
        }
        log_rows.append(row)
        write_csv(run_dir / "train_log.csv", log_rows)

        improved = val.active_acc > best_val_active
        if improved:
            best_val_active = val.active_acc
            best_epoch = epoch
            best_stats = {"val": val, "id_test": id_test, "ood_test": ood_test}
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_evals = 0
            if not args.no_save_checkpoints:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "depth": depth,
                        "args": vars(args),
                        "best_epoch": best_epoch,
                        "best_stats": {k: asdict(v) for k, v in best_stats.items()},
                    },
                    run_dir / "best.pt",
                )
        else:
            stale_evals += 1

        if is_solved(val, args.task, args.solved_threshold):
            solved_hits += 1
        else:
            solved_hits = 0

        print(
            f"[depth {depth} | epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"{format_stats('val', val, args.task)} | "
            f"{format_stats('ID', id_test, args.task)} | "
            f"{format_stats('OOD', ood_test, args.task)}",
            flush=True,
        )

        if args.stop_on_solved and solved_hits >= args.solved_patience:
            print(
                f"[depth {depth}] early stop: validation solved for {solved_hits} eval(s)",
                flush=True,
            )
            break
        if args.patience > 0 and stale_evals >= args.patience:
            print(f"[depth {depth}] early stop: no validation improvement", flush=True)
            break

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    assert best_stats is not None
    summary = {
        "depth": depth,
        "task": args.task,
        "best_epoch": best_epoch,
        "parameters": sum(p.numel() for p in model.parameters()),
        **{f"best_val_{k}": v for k, v in asdict(best_stats["val"]).items()},
        **{f"best_id_test_{k}": v for k, v in asdict(best_stats["id_test"]).items()},
        **{f"best_ood_test_{k}": v for k, v in asdict(best_stats["ood_test"]).items()},
    }
    return summary, log_rows, model


def import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_graphworld_example(example: GraphExample, path: Path, num_keys: int) -> None:
    plt = import_plotting()
    import matplotlib.patches as mpatches

    backbone_roles = {MOTIF_ROLE, RECORD_ROLE, ANCHOR_ROLE, STRUCT_TARGET_ROLE}
    backbone_nodes = [idx for idx, role_id in enumerate(example.role) if role_id in backbone_roles]
    radius = max(2.4, len(backbone_nodes) / 8.0)
    pos: dict[int, tuple[float, float]] = {0: (0.0, radius + 1.4)}
    for idx, node in enumerate(backbone_nodes):
        theta = 2.0 * math.pi * idx / max(1, len(backbone_nodes))
        pos[node] = (radius * math.cos(theta), radius * math.sin(theta))

    parent_counts: dict[int, int] = {}
    for node in range(example.n):
        if node in pos:
            continue
        parents = [idx for idx in range(example.n) if example.adj[node, idx] > 0 and idx in pos]
        parent = parents[0] if parents else 0
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
        px, py = pos[parent]
        norm = math.hypot(px, py) or 1.0
        ux, uy = px / norm, py / norm
        tx, ty = -uy, ux
        offset = (parent_counts[parent] - 1) * 0.22
        if parent == 0:
            ux, uy, tx, ty = 0.0, 1.0, 1.0, 0.0
        pos[node] = (px + 0.55 * ux + offset * tx, py + 0.55 * uy + offset * ty)

    record_lookup = {int(node): idx for idx, node in enumerate(example.record_nodes)}
    labels: dict[int, str] = {0: "Q"}
    colors = []
    edgecolors = []
    linewidths = []
    node_order = list(range(example.n))
    for node in node_order:
        role_id = int(example.role[node])
        if node == 0:
            labels[node] = "Q"
            colors.append("#f2f2f2")
            edgecolors.append("#222222")
            linewidths.append(1.8)
        elif role_id == ANCHOR_ROLE:
            labels[node] = "A"
            colors.append("#c7e9c0")
            edgecolors.append("#238b45")
            linewidths.append(2.4)
        elif role_id in {RECORD_ROLE, STRUCT_TARGET_ROLE}:
            record_idx = record_lookup[node]
            labels[node] = f"T{record_idx}" if role_id == STRUCT_TARGET_ROLE else f"R{record_idx}"
            is_structural = record_idx == example.structural_record
            is_symbolic = record_idx == example.symbolic_record
            if is_structural and is_symbolic:
                colors.append("#9e9ac8")
                edgecolors.append("#3f007d")
                linewidths.append(3.0)
            elif is_structural:
                colors.append("#9ecae1")
                edgecolors.append("#08519c")
                linewidths.append(3.0)
            elif is_symbolic:
                colors.append("#fcbba1")
                edgecolors.append("#cb181d")
                linewidths.append(3.0)
            else:
                colors.append("#ffffff")
                edgecolors.append("#737373")
                linewidths.append(1.2)
        elif role_id == VALUE_ROLE:
            value = int(example.symbol[node] - 1 - num_keys)
            labels[node] = f"V{value}"
            colors.append(plt.cm.tab20(value % 20))
            edgecolors.append("#525252")
            linewidths.append(1.0)
        elif role_id in {KEY_ROLE, QUERY_ROLE}:
            labels[node] = f"K{int(example.symbol[node] - 1)}"
            colors.append("#fff7bc")
            edgecolors.append("#b58100")
            linewidths.append(1.0)
        else:
            labels[node] = ""
            colors.append("#f7f7f7")
            edgecolors.append("#969696")
            linewidths.append(0.7)

    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    for i in range(example.n):
        for j in range(i + 1, example.n):
            if example.adj[i, j] > 0:
                ax.plot(
                    [pos[i][0], pos[j][0]],
                    [pos[i][1], pos[j][1]],
                    color="#d0d0d0",
                    linewidth=0.9,
                    zorder=1,
                )
    ax.scatter(
        [pos[node][0] for node in node_order],
        [pos[node][1] for node in node_order],
        s=420,
        c=colors,
        edgecolors=edgecolors,
        linewidths=linewidths,
        zorder=3,
    )
    for node in node_order:
        if labels[node]:
            ax.text(pos[node][0], pos[node][1], labels[node], ha="center", va="center", fontsize=8)

    query_title = ":".join(str(int(k)) for k in example.query_keys)
    structural_distance = int(example.structural_query_type)
    ax.set_title(
        "GraphWorld-style random graph task\n"
        f"family={example.graph_family}; structural d={structural_distance} target "
        f"record {example.structural_record} -> y={example.structural_y}; "
        f"symbolic Q{query_title or 'disabled'} record "
        f"{example.symbolic_record} -> V{example.symbolic_y}",
        fontsize=11,
    )
    handles = [
        mpatches.Patch(facecolor="#c7e9c0", edgecolor="#238b45", label="anchor"),
        mpatches.Patch(facecolor="#9ecae1", edgecolor="#08519c", label="structural target"),
        mpatches.Patch(facecolor="#fcbba1", edgecolor="#cb181d", label="symbolic target"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_task_example(example: GraphExample, path: Path, num_keys: int) -> None:
    if example.dataset_style == "graphworld":
        plot_graphworld_example(example, path, num_keys)
        return
    plt = import_plotting()
    import matplotlib.patches as mpatches

    records = len(example.record_nodes)
    record_node_set = {int(node) for node in example.record_nodes}
    query_nodes = [idx for idx, role_id in enumerate(example.role) if role_id == QUERY_ROLE]
    motif_nodes = [idx for idx, role_id in enumerate(example.role) if role_id == MOTIF_ROLE]
    query_motif_nodes = [
        node
        for node in motif_nodes
        if example.adj[0, node] > 0
        and not any(example.adj[node, record_node] > 0 for record_node in record_node_set)
    ]
    query_key_label = ":".join(str(int(k)) for k in example.query_keys) if query_nodes else ""
    query_bits = []
    if query_key_label:
        query_bits.append(f"K{query_key_label}")
    if query_motif_nodes:
        query_bits.append(f"M{example.structural_query_type}")
    query_label = " ".join(query_bits)
    pos: dict[int, tuple[float, float]] = {0: ((records - 1) / 2.0, 4.4)}
    labels: dict[int, str] = {0: f"Q{query_label}" if query_label else "Q"}
    colors: list[str] = []
    edgecolors: list[str] = []
    linewidths: list[float] = []

    node_order = list(range(example.n))
    query_offsets = np.linspace(-0.35, 0.35, max(1, len(query_nodes)))
    for offset, node in zip(query_offsets, query_nodes):
        pos[node] = (pos[0][0] + float(offset), 5.65)
        labels[node] = f"K{int(example.symbol[node] - 1)}"

    seen = {0}
    frontier = [(node, 0) for node in query_motif_nodes]
    by_depth: dict[int, list[int]] = {}
    while frontier:
        node, depth = frontier.pop(0)
        if node in seen:
            continue
        seen.add(node)
        by_depth.setdefault(depth, []).append(node)
        for nxt in motif_nodes:
            if example.adj[node, nxt] > 0 and nxt not in seen:
                frontier.append((nxt, depth + 1))
    for depth, nodes in by_depth.items():
        offsets = np.linspace(-0.34, 0.34, max(1, len(nodes)))
        for offset, node in zip(offsets, nodes):
            pos[node] = (pos[0][0] + float(offset), 3.65 - 0.52 * depth)
            labels[node] = ""

    for record_idx, (rec, val) in enumerate(zip(example.record_nodes, example.value_nodes)):
        x = float(record_idx)
        pos[int(rec)] = (x, 3.2)
        pos[int(val)] = (x, 2.2)
        labels[int(val)] = f"V{int(example.record_values[record_idx])}"
        key_nodes = [
            idx
            for idx, role_id in enumerate(example.role)
            if role_id == KEY_ROLE and example.adj[int(rec), idx] > 0
        ]
        key_tuple = ":".join(str(int(k)) for k in example.record_keys[record_idx])
        motif_type = int(example.record_motif_types[record_idx])
        if key_nodes:
            labels[int(rec)] = f"R{record_idx}\nK{key_tuple} M{motif_type}"
        else:
            labels[int(rec)] = f"R{record_idx}\nM{motif_type}"
        key_offsets = np.linspace(-0.28, 0.28, max(1, len(key_nodes)))
        for offset, key_node in zip(key_offsets, key_nodes):
            pos[key_node] = (x + float(offset), 4.05)
            labels[key_node] = f"K{int(example.symbol[key_node] - 1)}"

    for record_idx, rec in enumerate(example.record_nodes):
        x = float(record_idx)
        seen = {int(rec)}
        frontier = [(node, 0) for node in motif_nodes if example.adj[int(rec), node] > 0]
        by_depth: dict[int, list[int]] = {}
        while frontier:
            node, depth = frontier.pop(0)
            if node in seen:
                continue
            seen.add(node)
            by_depth.setdefault(depth, []).append(node)
            for nxt in motif_nodes:
                if example.adj[node, nxt] > 0 and nxt not in seen:
                    frontier.append((nxt, depth + 1))
        for depth, nodes in by_depth.items():
            offsets = np.linspace(-0.34, 0.34, max(1, len(nodes)))
            for offset, node in zip(offsets, nodes):
                pos[node] = (x + float(offset), 1.15 - 0.58 * depth)
                labels[node] = ""

    for node in node_order:
        role_id = int(example.role[node])
        if node == 0:
            colors.append("#f2f2f2")
            edgecolors.append("#222222")
            linewidths.append(1.8)
        elif role_id == RECORD_ROLE:
            record_idx = int(np.where(example.record_nodes == node)[0][0])
            is_structural = record_idx == example.structural_record
            is_symbolic = record_idx == example.symbolic_record
            if is_structural and is_symbolic:
                colors.append("#9e9ac8")
                edgecolors.append("#3f007d")
                linewidths.append(3.0)
            elif is_structural:
                colors.append("#9ecae1")
                edgecolors.append("#08519c")
                linewidths.append(3.0)
            elif is_symbolic:
                colors.append("#fcbba1")
                edgecolors.append("#cb181d")
                linewidths.append(3.0)
            else:
                colors.append("#ffffff")
                edgecolors.append("#737373")
                linewidths.append(1.2)
        elif role_id == VALUE_ROLE:
            value = int(example.symbol[node] - 1 - num_keys)
            colors.append(plt.cm.tab20(value % 20))
            edgecolors.append("#525252")
            linewidths.append(1.0)
        elif role_id in {KEY_ROLE, QUERY_ROLE}:
            colors.append("#fff7bc")
            edgecolors.append("#b58100")
            linewidths.append(1.0)
        else:
            colors.append("#f7f7f7")
            edgecolors.append("#969696")
            linewidths.append(0.8)

    fig, ax = plt.subplots(figsize=(max(9.0, records * 0.75), 5.4))
    for i in range(example.n):
        for j in range(i + 1, example.n):
            if example.adj[i, j] > 0:
                x0, y0 = pos[i]
                x1, y1 = pos[j]
                ax.plot([x0, x1], [y0, y1], color="#c7c7c7", linewidth=1.1, zorder=1)

    xs = [pos[node][0] for node in node_order]
    ys = [pos[node][1] for node in node_order]
    ax.scatter(
        xs,
        ys,
        s=760,
        c=colors,
        edgecolors=edgecolors,
        linewidths=linewidths,
        zorder=3,
    )
    for node in node_order:
        if labels.get(node):
            ax.text(
                pos[node][0],
                pos[node][1],
                labels[node],
                ha="center",
                va="center",
                fontsize=9,
                color="#111111",
                zorder=4,
            )

    structural_value = int(example.structural_y)
    symbolic_value = int(example.symbolic_y)
    query_title = ":".join(str(int(k)) for k in example.query_keys) if query_nodes else "disabled"
    structural_mode = "bridge" if example.structural_bridge else "motif-only"
    symbolic_mode = "partial" if example.symbolic_partial_distractors else "random"
    ax.set_title(
        "Decoupled graph task: structural motif matching vs symbolic retrieval\n"
        f"structural M{example.structural_query_type} -> record {example.structural_record} "
        f"({structural_mode}) -> V{structural_value}; "
        f"symbolic target = key Q{query_title} ({symbolic_mode}) at record "
        f"{example.symbolic_record} -> V{symbolic_value}",
        fontsize=12,
    )
    handles = [
        mpatches.Patch(facecolor="#9ecae1", edgecolor="#08519c", label="query-motif record"),
        mpatches.Patch(facecolor="#fcbba1", edgecolor="#cb181d", label="query-key record"),
        mpatches.Patch(facecolor="#f7f7f7", edgecolor="#969696", label="unlabeled motifs"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
    )
    ax.set_axis_off()
    ax.set_xlim(-0.75, records - 0.25)
    ax.set_ylim(-1.75, 5.75)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(log_rows: list[dict], path: Path, task: TaskName) -> None:
    if not log_rows:
        return
    plt = import_plotting()
    epochs = np.asarray([row["epoch"] for row in log_rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharex=True)
    axes[0].plot(epochs, [row["train_loss"] for row in log_rows], label="train", color="#252525")
    axes[0].plot(epochs, [row["val_loss"] for row in log_rows], label="val", color="#3182bd")
    axes[0].plot(epochs, [row["id_test_loss"] for row in log_rows], label="ID", color="#31a354")
    axes[0].plot(epochs, [row["ood_test_loss"] for row in log_rows], label="OOD", color="#de2d26")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross entropy")
    axes[0].legend(frameon=False)

    if task == "structural":
        metrics = [("structural_acc", "structural")]
    elif task == "symbolic":
        metrics = [("symbolic_acc", "symbolic")]
    else:
        metrics = [
            ("structural_acc", "structural"),
            ("symbolic_acc", "symbolic"),
            ("both_acc", "both"),
        ]
    colors = {"structural": "#08519c", "symbolic": "#cb181d", "both": "#54278f"}
    styles = {"val": "-", "id_test": "--", "ood_test": ":"}
    for metric, label in metrics:
        for split, style in styles.items():
            axes[1].plot(
                epochs,
                [row[f"{split}_{metric}"] for row in log_rows],
                linestyle=style,
                color=colors[label],
                label=f"{split} {label}",
            )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summary_lookup(
    rows: list[dict[str, float | int | str]],
    split: str,
) -> dict[tuple[int, int], dict[str, float]]:
    out: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        if str(row.get("split")) != split:
            continue
        key = (int(row["layer"]), int(row["head"]))
        out.setdefault(key, {})[str(row["metric"])] = float(row["score_mean"])
    return out


def plot_head_scores(
    perm_summary: list[dict[str, float | int | str]],
    attention_summary: list[dict[str, float | int | str]],
    split: str,
    path: Path,
) -> None:
    if not perm_summary:
        return
    plt = import_plotting()
    scores = summary_lookup(perm_summary, split=split)
    target = summary_lookup(attention_summary, split=split)
    heads = sorted(scores)
    if not heads:
        return

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.3))
    for layer, head in heads:
        structural = scores[(layer, head)].get("structural_score", float("nan"))
        symbolic = scores[(layer, head)].get("symbolic_score", float("nan"))
        centered_structural = scores[(layer, head)].get(
            "centered_structural_score",
            float("nan"),
        )
        centered_symbolic = scores[(layer, head)].get("centered_symbolic_score", float("nan"))
        color = ["#3182bd", "#e6550d", "#756bb1", "#31a354"][layer % 4]
        axes[0].scatter(
            structural,
            symbolic,
            s=110,
            color=color,
            edgecolor="#222222",
            linewidth=0.7,
        )
        axes[0].text(structural + 0.01, symbolic + 0.01, f"L{layer}H{head}", fontsize=9)
        axes[1].scatter(
            centered_structural,
            centered_symbolic,
            s=110,
            color=color,
            edgecolor="#222222",
            linewidth=0.7,
        )
        axes[1].text(
            centered_structural + 0.02,
            centered_symbolic + 0.02,
            f"L{layer}H{head}",
            fontsize=9,
        )
    axes[0].plot([0, 1], [0, 1], color="#bdbdbd", linewidth=1.0)
    axes[0].set_xlim(0.0, 1.02)
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_xlabel("structural score\n(attention invariant)")
    axes[0].set_ylabel("symbolic score\n(attention equivariant)")
    axes[0].set_title("Baseline score plane")

    axes[1].axhline(0.0, color="#bdbdbd", linewidth=1.0)
    axes[1].axvline(0.0, color="#bdbdbd", linewidth=1.0)
    axes[1].plot([-1, 1], [-1, 1], color="#d9d9d9", linewidth=1.0)
    axes[1].set_xlim(-1.02, 1.02)
    axes[1].set_ylim(-1.02, 1.02)
    axes[1].set_xlabel("centered structural score")
    axes[1].set_ylabel("centered symbolic score")
    axes[1].set_title("Centered score plane")

    metric_names = [
        "structural_score",
        "symbolic_score",
        "centered_structural_score",
        "centered_symbolic_score",
    ]
    heat = np.asarray([[scores[h].get(metric, np.nan) for h in heads] for metric in metric_names])
    im = axes[2].imshow(heat, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    axes[2].set_yticks(range(len(metric_names)))
    axes[2].set_yticklabels(["struct", "sym", "ctr struct", "ctr sym"])
    axes[2].set_xticks(range(len(heads)))
    axes[2].set_xticklabels([f"L{l}H{h}" for l, h in heads], rotation=45, ha="right")
    axes[2].set_title("Permutation metrics")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    width = 0.35
    x = np.arange(len(heads))
    struct_mass = [target.get(h, {}).get("cls_to_structural_record", np.nan) for h in heads]
    sym_mass = [target.get(h, {}).get("cls_to_symbolic_record", np.nan) for h in heads]
    axes[3].bar(x - width / 2, struct_mass, width, label="CLS -> structural", color="#9ecae1")
    axes[3].bar(x + width / 2, sym_mass, width, label="CLS -> symbolic", color="#fcbba1")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels([f"L{l}H{h}" for l, h in heads], rotation=45, ha="right")
    axes[3].set_ylim(0.0, max(0.2, np.nanmax([struct_mass, sym_mass]) * 1.25))
    axes[3].set_ylabel("mean attention mass")
    axes[3].set_title("Task-target attention")
    axes[3].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_prediction_panel(
    model: GraphGPSDecoupledModel,
    examples: list[GraphExample],
    task: TaskName,
    num_values: int,
    spd_cap: int,
    device: torch.device,
    path: Path,
    max_examples: int = 8,
) -> None:
    plt = import_plotting()
    selected = examples[: min(max_examples, len(examples))]
    if not selected:
        return
    batch = collate_examples(selected, spd_cap=spd_cap).to(device)
    model.eval()
    outputs, _ = model(batch, collect_attention=False)

    rows = []
    labels = []
    truth_positions = []
    if task in {"structural", "dual"}:
        probs = torch.softmax(outputs["structural"], dim=-1).detach().cpu().numpy()
        for idx, prob in enumerate(probs):
            rows.append(prob)
            labels.append(f"g{idx} structural")
            truth_positions.append(int(selected[idx].structural_y))
    if task in {"symbolic", "dual"}:
        probs = torch.softmax(outputs["symbolic"], dim=-1).detach().cpu().numpy()
        for idx, prob in enumerate(probs):
            rows.append(prob)
            labels.append(f"g{idx} symbolic")
            truth_positions.append(int(selected[idx].symbolic_y))

    data = np.asarray(rows, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(8.0, num_values * 0.45), max(3.0, len(rows) * 0.34)))
    im = ax.imshow(data, vmin=0.0, vmax=max(0.1, float(data.max())), cmap="magma", aspect="auto")
    ax.set_xticks(range(num_values))
    ax.set_xticklabels([f"V{i}" for i in range(num_values)], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for row_idx, truth in enumerate(truth_positions):
        ax.scatter(
            [truth],
            [row_idx],
            marker="s",
            s=90,
            facecolors="none",
            edgecolors="#7fcdbb",
            linewidths=1.8,
        )
    ax.set_title("Prediction distributions; cyan box marks target value")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_suite_comparisons(suite_dir: Path, tasks: list[str], num_values: int) -> None:
    summary_rows = read_csv_rows(suite_dir / "summary_all_tasks.csv")
    metric_rows: list[dict[str, str]] = []
    attention_rows: list[dict[str, str]] = []
    for task in tasks:
        for row in read_csv_rows(suite_dir / task / "symbol_permutation_summary.csv"):
            row = dict(row)
            row["model_task"] = task
            metric_rows.append(row)
        for row in read_csv_rows(suite_dir / task / "attention_target_summary.csv"):
            row = dict(row)
            row["model_task"] = task
            attention_rows.append(row)
    if not summary_rows and not metric_rows:
        return

    plt = import_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0))
    task_order = [task for task in tasks if any(row.get("task") == task for row in summary_rows)]
    if not task_order:
        task_order = tasks

    x = np.arange(len(task_order))
    width = 0.24
    for offset, split, label, color in (
        (-width, "val", "val", "#756bb1"),
        (0.0, "id_test", "ID", "#31a354"),
        (width, "ood_test", "OOD", "#de2d26"),
    ):
        values = []
        for task in task_order:
            row = next((r for r in summary_rows if r.get("task") == task), {})
            values.append(as_float(row, f"best_{split}_active_acc"))
        axes[0, 0].bar(x + offset, values, width, label=label, color=color)
    axes[0, 0].axhline(1.0 / max(1, num_values), color="#d9d9d9", linewidth=1.0)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(task_order, rotation=20, ha="right")
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_ylabel("active accuracy")
    axes[0, 0].set_title("Best accuracy by trained model")
    axes[0, 0].legend(frameon=False)

    colors = {"structural": "#08519c", "symbolic": "#cb181d", "dual": "#54278f"}
    for task in task_order:
        rows = [
            row
            for row in metric_rows
            if row.get("model_task") == task
            and row.get("split") == "id_test"
            and row.get("metric") in {"centered_structural_score", "centered_symbolic_score"}
        ]
        heads = sorted({(row.get("layer"), row.get("head")) for row in rows})
        for layer, head in heads:
            struct = next(
                (
                    as_float(row, "score_mean")
                    for row in rows
                    if row.get("layer") == layer
                    and row.get("head") == head
                    and row.get("metric") == "centered_structural_score"
                ),
                float("nan"),
            )
            sym = next(
                (
                    as_float(row, "score_mean")
                    for row in rows
                    if row.get("layer") == layer
                    and row.get("head") == head
                    and row.get("metric") == "centered_symbolic_score"
                ),
                float("nan"),
            )
            axes[0, 1].scatter(
                struct,
                sym,
                s=95,
                color=colors.get(task, "#636363"),
                edgecolor="#222222",
                linewidth=0.6,
                label=task,
            )
            axes[0, 1].text(struct + 0.015, sym + 0.015, f"{task[0]}L{layer}H{head}", fontsize=8)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    axes[0, 1].axhline(0.0, color="#bdbdbd", linewidth=1.0)
    axes[0, 1].axvline(0.0, color="#bdbdbd", linewidth=1.0)
    axes[0, 1].plot([-1, 1], [-1, 1], color="#d9d9d9", linewidth=1.0)
    axes[0, 1].set_xlim(-1.02, 1.02)
    axes[0, 1].set_ylim(-1.02, 1.02)
    axes[0, 1].set_xlabel("centered structural score")
    axes[0, 1].set_ylabel("centered symbolic score")
    axes[0, 1].set_title("ID centered score plane")
    axes[0, 1].legend(dedup.values(), dedup.keys(), frameon=False)

    metric_names = [
        "structural_score",
        "symbolic_score",
        "centered_structural_score",
        "centered_symbolic_score",
    ]
    head_order = sorted(
        {
            (int(row["layer"]), int(row["head"]))
            for row in metric_rows
            if row.get("split") == "id_test" and row.get("layer") and row.get("head")
        }
    )
    heat_rows = []
    y_labels = []
    for task in task_order:
        task_rows = [
            row
            for row in metric_rows
            if row.get("model_task") == task and row.get("split") == "id_test"
        ]
        for metric in metric_names:
            heat_rows.append(
                [
                    next(
                        (
                            as_float(row, "score_mean")
                            for row in task_rows
                            if int(row["layer"]) == layer
                            and int(row["head"]) == head
                            and row.get("metric") == metric
                        ),
                        float("nan"),
                    )
                    for layer, head in head_order
                ]
            )
            y_labels.append(f"{task} {metric.replace('_score', '').replace('_', ' ')}")
    if heat_rows and head_order:
        heat = np.asarray(heat_rows, dtype=np.float64)
        im = axes[1, 0].imshow(heat, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
        axes[1, 0].set_yticks(range(len(y_labels)))
        axes[1, 0].set_yticklabels(y_labels, fontsize=8)
        axes[1, 0].set_xticks(range(len(head_order)))
        axes[1, 0].set_xticklabels([f"L{l}H{h}" for l, h in head_order], rotation=45, ha="right")
        axes[1, 0].set_title("ID permutation metrics by model")
        fig.colorbar(im, ax=axes[1, 0], fraction=0.035, pad=0.02)
    else:
        axes[1, 0].axis("off")

    target_metric_names = ["cls_to_structural_record", "cls_to_symbolic_record"]
    x_labels = []
    struct_values = []
    sym_values = []
    for task in task_order:
        rows = [
            row
            for row in attention_rows
            if row.get("model_task") == task
            and row.get("split") == "id_test"
            and row.get("layer") == "1"
        ]
        for head in sorted({row.get("head") for row in rows}):
            x_labels.append(f"{task}\nH{head}")
            struct_values.append(
                next(
                    (
                        as_float(row, "score_mean")
                        for row in rows
                        if row.get("head") == head and row.get("metric") == target_metric_names[0]
                    ),
                    float("nan"),
                )
            )
            sym_values.append(
                next(
                    (
                        as_float(row, "score_mean")
                        for row in rows
                        if row.get("head") == head and row.get("metric") == target_metric_names[1]
                    ),
                    float("nan"),
                )
            )
    if x_labels:
        tx = np.arange(len(x_labels))
        axes[1, 1].bar(tx - 0.18, struct_values, 0.36, color="#9ecae1", label="CLS -> structural")
        axes[1, 1].bar(tx + 0.18, sym_values, 0.36, color="#fcbba1", label="CLS -> symbolic")
        axes[1, 1].set_xticks(tx)
        axes[1, 1].set_xticklabels(x_labels, rotation=0, ha="center", fontsize=8)
        axes[1, 1].set_ylabel("mean attention mass")
        axes[1, 1].set_title("Final-layer target attention by model")
        axes[1, 1].legend(frameon=False)
    else:
        axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(suite_dir / "suite_model_metric_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_design_card(path: Path, args: argparse.Namespace) -> None:
    selector_note = (
        f"- Task: `{args.task}` with `{args.experiment_focus}` focus.\n"
        f"- Structural/positional selector enabled: `{args.enable_structural_selector}`.\n"
        f"- Query/key symbolic selector enabled: `{args.enable_query_key}`."
    )
    if args.dataset_style == "graphworld":
        structural_label = (
            f"{args.graphworld_distance_classes}-way anchor-distance shell"
            if args.graphworld_structural_label == "distance_bin"
            else "nearest-candidate value retrieval"
        )
        graph_distribution = f"""- Each graph has one readout/query node, a
  {args.graph_family} random backbone with {args.graph_min_nodes}-{args.graph_max_nodes}
  backbone nodes, and {args.train_min_records}-{args.train_max_records} candidate
  records during training, so graph/input length varies within every split.
- Backbones are generated locally in the spirit of GraphWorld: ER, small-world,
  preferential-attachment, SBM-like, or mixed random graph families.
- Candidate nodes are ordinary backbone nodes marked with a record role. Each
  candidate has one value leaf. In symbolic runs the primary query key is stored
  on the readout node and the primary record key is stored on the candidate,
  with optional extra key leaves for partial-match hard cases.
- In structural-focused runs, an anchor node and a structural target candidate
  are marked by roles. The default structural label is `{structural_label}` with
  RWSE structural features. SPD attention bias is available but usually makes
  this positional task too easy for a 2-layer baseline.
- In symbolic-focused runs, the readout carries query-key leaves and the target
  is the candidate whose key tuple matches exactly. The default `mixed`
  distractor mode combines easier one-key retrieval with harder partial-match
  examples.
- In dual runs, both selectors are enabled and point to different candidates by
  default. Use `--dual-symbolic-loss-weight` if the semantic retrieval target
  needs extra weight during dual training."""
        label_text = f"""- `structural`: classify the {structural_label} target.
- `symbolic`: classify the value attached to the query-key candidate.
- `dual`: predict both labels from the same graph."""
    else:
        graph_distribution = f"""- Each graph has one readout/query node and
  {args.train_min_records}-{args.train_max_records} training records, so
  graph/input length varies within every split.
- Each record has a value leaf, a tuple of key leaves when symbolic input is
  enabled, and a five-node unlabeled structural motif.
- In structural-focused runs, the readout carries an unlabeled query motif and
  the target is the record whose unlabeled motif matches it; query/key symbols
  are disabled. The default `branch` motif style uses branch-count differences,
  while `subtle` keeps the harder tree-shape variants. A fraction of examples
  also include a purely structural bridge from the readout to the target record.
- In symbolic-focused runs, an exact key-tuple match selects the target record
  and the structural query-motif selector is disabled. The default `mixed`
  distractor mode combines easier one-key retrieval with harder partial-match
  examples.
- In dual runs, both selectors are enabled and point to different records by
  default."""
        label_text = """- `structural`: classify the value attached to the query-motif record.
- `symbolic`: classify the value attached to the query-key record.
- `dual`: predict both labels from the same graph."""
    text = f"""# Structural/Symbolic Graph Task

This run uses a decoupled graph task inspired by the positional versus symbolic
attention split from Urrutia et al. (ICLR 2026). The default graph distribution
also follows the GraphWorld benchmarking idea: evaluate models on controlled,
statistically varied synthetic graph populations rather than a single fixed
benchmark shape.

## Graph Distribution

{graph_distribution}

{selector_note}

## Labels

{label_text}

## Intended 2-Layer Mechanism

The structural head reads from the marked target node, while the symbolic head
reads from the readout/query node. Layer 1 can move local key/value evidence
into each candidate node and propagate structural features over the random
backbone. Layer 2 can support either the positional target-node computation or
the semantic candidate-to-readout route. With two attention heads per layer, the
dual task provides pressure for distinct structural and symbolic routing
patterns.

## Metrics

- `structural_score`: attention rows stay invariant when node symbols are
  permuted while graph structure is fixed.
- `symbolic_score`: attention rows transform equivariantly with the symbol
  permutation.
- `centered_structural_score` and `centered_symbolic_score`: row-centered
  attention versions from Appendix C of the project PDF; these remove the
  diffuse uniform-attention baseline before computing cosine similarity.
- `cls_to_structural_record` and `cls_to_symbolic_record`: mean readout-node
  attention mass assigned to each task's target record.

## Optional Specialisation Pressure

Pass `--specialisation-loss-weight 0.01` or similar to add a light auxiliary
attention loss. It encourages some final-layer head to place readout attention
on each active target record; in `dual` mode it also discourages structural and
symbolic target mass from concentrating in the same head distribution.
"""
    path.write_text(text, encoding="utf-8")


def write_visualisations(
    model: GraphGPSDecoupledModel,
    log_rows: list[dict],
    eval_sets: dict[str, list[GraphExample]],
    perm_summary: list[dict[str, float | int | str]],
    attention_summary: list[dict[str, float | int | str]],
    run_dir: Path,
    args: argparse.Namespace,
    depth: int,
    device: torch.device,
) -> None:
    if args.skip_plots:
        return
    try:
        plot_task_example(
            eval_sets["id_test"][0],
            run_dir / "task_example.png",
            num_keys=args.num_keys,
        )
        plot_training_curves(log_rows, run_dir / "training_curves.png", task=args.task)
        plot_head_scores(
            perm_summary,
            attention_summary,
            split="id_test",
            path=run_dir / "head_specialisation_id.png",
        )
        plot_prediction_panel(
            model,
            eval_sets["id_test"],
            task=args.task,
            num_values=args.num_values,
            spd_cap=args.spd_cap,
            device=device,
            path=run_dir / "prediction_panel_id.png",
        )
    except Exception as exc:
        print(f"[plots] skipped for depth {depth}: {exc}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small GraphGPS-style model on decoupled structural/symbolic "
            "graph tasks."
        )
    )
    parser.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default=None,
        help="Run a single task. If omitted, runs structural, symbolic, and dual.",
    )
    parser.add_argument(
        "--tasks",
        choices=TASK_CHOICES,
        nargs="+",
        default=None,
        help="Run an explicit suite of tasks. Defaults to structural symbolic dual.",
    )
    parser.add_argument(
        "--include-dual",
        action="store_true",
        help="Append the mixed dual task when using a custom task suite.",
    )
    parser.add_argument(
        "--experiment-focus",
        choices=["isolated", "shared"],
        default="isolated",
        help=(
            "isolated disables query/key symbols for structural-only runs and disables the "
            "structural query motif for symbolic-only runs. shared keeps both selectors "
            "in all tasks."
        ),
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[2])
    parser.add_argument(
        "--structural-channel",
        choices=["none", "degree", "rwse", "spd_bias", "rwse_spd_bias"],
        default="rwse",
        help=(
            "Structural input to use. rwse is the default GraphWorld setting; "
            "rwse_spd_bias makes anchor-distance labels much easier."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--rwse-steps", type=int, default=12)
    parser.add_argument("--spd-cap", type=int, default=32)

    parser.add_argument(
        "--dataset-style",
        choices=["graphworld", "record"],
        default="graphworld",
        help="graphworld uses random-graph backbones; record keeps the older record motif task.",
    )
    parser.add_argument(
        "--graph-family",
        choices=["er", "ws", "ba", "sbm", "mixed"],
        default="mixed",
        help="Random graph family used by --dataset-style graphworld.",
    )
    parser.add_argument("--graph-min-nodes", type=int, default=24)
    parser.add_argument("--graph-max-nodes", type=int, default=48)
    parser.add_argument(
        "--graphworld-structural-label",
        choices=["distance_bin", "value_at_nearest"],
        default="distance_bin",
        help=(
            "distance_bin classifies the marked target candidate's anchor-distance "
            "shell; value_at_nearest retrieves the uniquely nearest candidate's value."
        ),
    )
    parser.add_argument(
        "--graphworld-distance-classes",
        type=int,
        default=4,
        help="Number of anchor-to-target distance shells used by distance_bin labels.",
    )

    parser.add_argument("--num-keys", type=int, default=24)
    parser.add_argument("--num-values", type=int, default=8)
    parser.add_argument("--key-tuple-size", type=int, default=2)
    parser.add_argument(
        "--symbolic-distractor-mode",
        choices=["random", "partial", "mixed"],
        default="mixed",
        help=(
            "partial makes non-target records share one query-key component when "
            "--key-tuple-size is at least 2; mixed uses partial distractors on "
            "a configurable fraction of graphs."
        ),
    )
    parser.add_argument(
        "--symbolic-partial-fraction",
        type=float,
        default=0.20,
        help="Fraction of symbolic examples using partial-match distractors in mixed mode.",
    )
    parser.add_argument(
        "--num-motif-types",
        type=int,
        default=3,
        help="Number of queryable unlabeled motif templates used by structural runs.",
    )
    parser.add_argument(
        "--motif-style",
        choices=["branch", "subtle"],
        default="branch",
        help="branch uses easier branch-count motifs; subtle keeps the harder tree variants.",
    )
    parser.add_argument(
        "--structural-bridge-fraction",
        type=float,
        default=0.50,
        help=(
            "For GraphWorld value_at_nearest mode, fraction of structural examples "
            "whose nearest candidate is one hop from the anchor. In record mode, "
            "fraction with an explicit structural bridge."
        ),
    )
    parser.add_argument("--train-min-records", type=int, default=8)
    parser.add_argument("--train-max-records", type=int, default=14)
    parser.add_argument("--ood-min-records", type=int, default=16)
    parser.add_argument("--ood-max-records", type=int, default=22)
    parser.add_argument(
        "--avoid-target-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--add-record-path", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--train-graphs-per-epoch", type=int, default=2048)
    parser.add_argument("--val-graphs", type=int, default=512)
    parser.add_argument("--id-test-graphs", type=int, default=1024)
    parser.add_argument("--ood-test-graphs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--solved-threshold", type=float, default=0.999)
    parser.add_argument("--solved-patience", type=int, default=5)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--stop-on-solved", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--specialisation-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Optional auxiliary loss that encourages final-layer CLS attention to put mass on "
            "the active task target record(s). In dual mode it also discourages using the same "
            "head distribution for both targets."
        ),
    )
    parser.add_argument(
        "--dual-symbolic-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Symbolic cross-entropy multiplier used only for dual runs. The "
            "symbolic/semantic target can learn more slowly than the distance-shell "
            "target in some calibrations."
        ),
    )

    parser.add_argument("--metric-perms", type=int, default=4)
    parser.add_argument("--metric-graphs", type=int, default=256)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--skip-permutation-metrics", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output root. Defaults to Google Drive in Colab and to "
            "experiments/synthetic/results/structural_symbolic_graphgps locally."
        ),
    )
    parser.add_argument(
        "--drive-output-root",
        type=Path,
        default=Path("MyDrive/graph_specialisation_metrics/structural_symbolic_graphgps"),
        help="Path under /content/drive used when running in Google Colab.",
    )
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Tiny run for syntax/runtime checks.",
    )
    args = parser.parse_args(notebook_safe_argv(argv))

    if args.task is not None and args.tasks is not None:
        raise ValueError("Use either --task for a single run or --tasks for a suite, not both")
    if args.tasks is None:
        if args.task is None:
            args.tasks = ["structural", "symbolic", "dual"]
        else:
            args.tasks = [args.task]
    else:
        args.tasks = list(args.tasks)
    if args.include_dual and "dual" not in args.tasks:
        args.tasks.append("dual")
    args.task = args.tasks[0]

    if args.num_keys <= args.train_max_records:
        raise ValueError(
            "--num-keys should exceed --train-max-records to keep key matching nontrivial"
        )
    if args.num_values < 3:
        raise ValueError("--num-values should be at least 3")
    if args.key_tuple_size < 1:
        raise ValueError("--key-tuple-size must be at least 1")
    if not 2 <= args.num_motif_types <= 4:
        raise ValueError("--num-motif-types must be between 2 and 4")
    if not 0.0 <= args.structural_bridge_fraction <= 1.0:
        raise ValueError("--structural-bridge-fraction must be in [0, 1]")
    if not 0.0 <= args.symbolic_partial_fraction <= 1.0:
        raise ValueError("--symbolic-partial-fraction must be in [0, 1]")
    if not 2 <= args.graphworld_distance_classes <= args.num_values:
        raise ValueError("--graphworld-distance-classes must be between 2 and --num-values")
    if args.dual_symbolic_loss_weight <= 0:
        raise ValueError("--dual-symbolic-loss-weight must be positive")
    if args.fast_dev_run:
        args.depths = args.depths[:1]
        args.train_graphs_per_epoch = 96
        args.val_graphs = 24
        args.id_test_graphs = 24
        args.ood_test_graphs = 24
        args.batch_size = 16
        args.eval_batch_size = 24
        args.metric_graphs = 16
        args.metric_perms = 1
        args.max_epochs = min(args.max_epochs, 2)
        args.patience = 2
        args.rwse_steps = min(args.rwse_steps, 4)

    return args


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
    return Path("experiments/synthetic/results/structural_symbolic_graphgps")


def build_eval_sets(args: argparse.Namespace) -> dict[str, list[GraphExample]]:
    return {
        "val": generate_examples(
            args.val_graphs,
            args.train_min_records,
            args.train_max_records,
            args.seed + 11,
            args.num_keys,
            args.num_values,
            args.key_tuple_size,
            args.symbolic_distractor_mode,
            args.num_motif_types,
            args.motif_style,
            args.structural_bridge_fraction,
            args.symbolic_partial_fraction,
            args.structural_channel,
            args.rwse_steps,
            args.avoid_target_overlap,
            args.add_record_path,
            args.enable_structural_selector,
            args.enable_query_key,
            args.dataset_style,
            args.graph_min_nodes,
            args.graph_max_nodes,
            args.graph_family,
            args.graphworld_structural_label,
            args.graphworld_distance_classes,
        ),
        "id_test": generate_examples(
            args.id_test_graphs,
            args.train_min_records,
            args.train_max_records,
            args.seed + 23,
            args.num_keys,
            args.num_values,
            args.key_tuple_size,
            args.symbolic_distractor_mode,
            args.num_motif_types,
            args.motif_style,
            args.structural_bridge_fraction,
            args.symbolic_partial_fraction,
            args.structural_channel,
            args.rwse_steps,
            args.avoid_target_overlap,
            args.add_record_path,
            args.enable_structural_selector,
            args.enable_query_key,
            args.dataset_style,
            args.graph_min_nodes,
            args.graph_max_nodes,
            args.graph_family,
            args.graphworld_structural_label,
            args.graphworld_distance_classes,
        ),
        "ood_test": generate_examples(
            args.ood_test_graphs,
            args.ood_min_records,
            args.ood_max_records,
            args.seed + 37,
            args.num_keys,
            args.num_values,
            args.key_tuple_size,
            args.symbolic_distractor_mode,
            args.num_motif_types,
            args.motif_style,
            args.structural_bridge_fraction,
            args.symbolic_partial_fraction,
            args.structural_channel,
            args.rwse_steps,
            args.avoid_target_overlap,
            args.add_record_path,
            args.enable_structural_selector,
            args.enable_query_key,
            args.dataset_style,
            args.graph_min_nodes,
            args.graph_max_nodes,
            args.graph_family,
            args.graphworld_structural_label,
            args.graphworld_distance_classes,
        ),
    }


def format_summary_detail(row: dict) -> str:
    task = str(row["task"])
    if task == "dual":
        return (
            f"val_struct={row['best_val_structural_acc']:.4f} "
            f"val_sym={row['best_val_symbolic_acc']:.4f} "
            f"ID_both={row['best_id_test_both_acc']:.4f} "
            f"OOD_both={row['best_ood_test_both_acc']:.4f}"
        )
    if task == "structural":
        return (
            f"val_struct={row['best_val_structural_acc']:.4f} "
            f"ID_struct={row['best_id_test_structural_acc']:.4f} "
            f"OOD_struct={row['best_ood_test_structural_acc']:.4f}"
        )
    return (
        f"val_sym={row['best_val_symbolic_acc']:.4f} "
        f"ID_sym={row['best_id_test_symbolic_acc']:.4f} "
        f"OOD_sym={row['best_ood_test_symbolic_acc']:.4f}"
    )


def make_task_args(args: argparse.Namespace, task: str) -> argparse.Namespace:
    task_args = argparse.Namespace(**vars(args))
    task_args.task = task
    enable_structural, enable_query = selector_flags_for_task(task, args.experiment_focus)
    task_args.enable_structural_selector = enable_structural
    task_args.enable_query_key = enable_query
    return task_args


def run_task_experiment(
    task_args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(task_args) | {"device_resolved": str(device)}, f, indent=2, default=str)
    write_design_card(output_dir / "task_design.md", task_args)

    print(f"[setup] task={task_args.task} output_dir={output_dir}", flush=True)
    print(
        f"[setup] selectors structural={task_args.enable_structural_selector} "
        f"query_key={task_args.enable_query_key} depth(s)={task_args.depths} "
        f"heads={task_args.num_heads} channel={task_args.structural_channel}",
        flush=True,
    )

    print(f"[data] task={task_args.task} generating fixed val/ID/OOD sets", flush=True)
    eval_sets = build_eval_sets(task_args)
    summaries = []
    all_perm_rows: list[dict[str, float | int | str]] = []
    all_perm_summary: list[dict[str, float | int | str]] = []
    all_attention_summary: list[dict[str, float | int | str]] = []

    for depth in task_args.depths:
        summary, log_rows, model = train_depth(depth, task_args, eval_sets, output_dir, device)
        summaries.append(summary)
        write_csv(output_dir / "summary.csv", summaries)

        run_dir = output_dir / f"depth_{depth}"
        perm_summary: list[dict[str, float | int | str]] = []
        if not task_args.skip_permutation_metrics:
            for split in ("id_test", "ood_test"):
                print(
                    f"[metrics] depth={depth} computing {split} symbol-permutation metrics",
                    flush=True,
                )
                rows = compute_symbol_permutation_metrics(
                    model,
                    eval_sets[split],
                    split=split,
                    depth=depth,
                    task=task_args.task,
                    batch_size=task_args.metric_batch_size,
                    spd_cap=task_args.spd_cap,
                    num_perms=task_args.metric_perms,
                    metric_graphs=task_args.metric_graphs,
                    seed=task_args.seed + 101 * depth + (0 if split == "id_test" else 10000),
                    device=device,
                )
                all_perm_rows.extend(rows)
            perm_summary = summarize_metric_rows(
                [row for row in all_perm_rows if int(row["depth"]) == depth]
            )
            all_perm_summary.extend(perm_summary)
            write_csv(output_dir / "symbol_permutation_metrics.csv", all_perm_rows)
            write_csv(output_dir / "symbol_permutation_summary.csv", all_perm_summary)

        attention_summary: list[dict[str, float | int | str]] = []
        for split in ("id_test", "ood_test"):
            print(f"[metrics] depth={depth} computing {split} target-attention metrics", flush=True)
            attention_summary.extend(
                compute_attention_target_metrics(
                    model,
                    eval_sets[split],
                    split=split,
                    depth=depth,
                    task=task_args.task,
                    batch_size=task_args.metric_batch_size,
                    spd_cap=task_args.spd_cap,
                    metric_graphs=task_args.metric_graphs,
                    device=device,
                )
            )
        all_attention_summary.extend(attention_summary)
        write_csv(output_dir / "attention_target_summary.csv", all_attention_summary)

        write_visualisations(
            model=model,
            log_rows=log_rows,
            eval_sets=eval_sets,
            perm_summary=perm_summary,
            attention_summary=attention_summary,
            run_dir=run_dir,
            args=task_args,
            depth=depth,
            device=device,
        )

    print(f"[done] task={task_args.task} best summary", flush=True)
    for row in summaries:
        print(
            f"depth={row['depth']} params={row['parameters']} best_epoch={row['best_epoch']} "
            f"{format_summary_detail(row)}",
            flush=True,
        )
    print(f"[done] wrote {output_dir}", flush=True)
    return summaries


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = choose_device(args.device)
    output_root = resolve_output_root(args)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_output_dir = output_root / run_name
    suite_output_dir.mkdir(parents=True, exist_ok=True)
    with (suite_output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"device_resolved": str(device)}, f, indent=2, default=str)

    print(f"[setup] suite_output_dir={suite_output_dir}", flush=True)
    print(f"[setup] device={device}", flush=True)
    print(f"[setup] tasks={' '.join(args.tasks)} focus={args.experiment_focus}", flush=True)

    suite_summaries: list[dict] = []
    for task in args.tasks:
        task_args = make_task_args(args, task)
        task_summaries = run_task_experiment(
            task_args=task_args,
            output_dir=suite_output_dir / task,
            device=device,
        )
        suite_summaries.extend(task_summaries)
        write_csv(suite_output_dir / "summary_all_tasks.csv", suite_summaries)

    print("[done] suite summary", flush=True)
    for row in suite_summaries:
        print(
            f"task={row['task']} depth={row['depth']} best_epoch={row['best_epoch']} "
            f"{format_summary_detail(row)}",
            flush=True,
        )
    if not args.skip_plots:
        try:
            plot_suite_comparisons(suite_output_dir, list(args.tasks), args.num_values)
        except Exception as exc:
            print(f"[plots] skipped suite comparison: {exc}", flush=True)
    print(f"[done] wrote suite {suite_output_dir}", flush=True)


if __name__ == "__main__":
    main()
