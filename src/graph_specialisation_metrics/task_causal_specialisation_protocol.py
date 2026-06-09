"""Task-causal specialisation protocol for GraphBench GRIT analyses.

This runner implements the lightweight/full protocol sketched in
``task_causal_specialisation_protocol_A100.md``.  It deliberately reuses the
HPC-facing loader, PE-cache handling, official GRIT hooks, and patching utilities
from ``mechanistic_operator_analysis.py`` and
``mechanistic_pair_patching_scrubbing.py``.

Stages are restartable:

``labels``
    Compute solver-causal unit labels and perturbation metadata.
``alignment``
    Run clean GRIT forwards with gradients and reduce per-unit internal scores.
``field``
    Perturb causal and matched-control units, then measure field response.
``patching``
    Patch perturbed activations into the clean graph at selected sites.
``figures``
    Produce compact CSV summaries, figures, and a claim summary.

The first-class solver label implementations are max-flow edge sensitivity and
bipartite-matching edge/non-edge sensitivity.  Other GraphBench datasets can be
used downstream by supplying an external sensitivity-unit CSV with the columns
written by the ``labels`` stage.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn

from graph_specialisation_metrics import mechanistic_operator_analysis as moa
from graph_specialisation_metrics import mechanistic_pair_patching_scrubbing as mps


EPS = 1.0e-12
STAGES = ("labels", "alignment", "field", "patching", "figures")
ANALYSIS_PRESETS = ("custom", "lightweight", "full")
INTERNAL_SITE_FAMILIES = (
    "routing_influence",
    "pair_value_influence",
    "pair_state_influence",
    "node_endpoint_influence",
)
PATCH_TARGET_BY_SITE = {
    "attention_mass": "attention",
    "routing_influence": "routing_logits",
    "pair_value_influence": "pair_value",
    "pair_state_influence": "pair_state",
    "node_endpoint_influence": "h_endpoint",
}
@dataclass(frozen=True)
class FlowSolveResult:
    value: float
    residual_forward: torch.Tensor
    source_side: torch.Tensor


@dataclass(frozen=True)
class SiteSelection:
    site_family: str
    patch_target: str
    layer: int
    head: int
    selection_metric: str
    selection_value: float
    baseline_value: float
    gain: float


@dataclass
class GRITCausalLayerRecord:
    layer: int
    src: torch.Tensor
    dst: torch.Tensor
    graph: torch.Tensor
    local_src: torch.Tensor
    local_dst: torch.Tensor
    attention: torch.Tensor
    logits: torch.Tensor
    node_message: torch.Tensor
    pair_message: torch.Tensor
    message: torch.Tensor
    pair_state: Optional[torch.Tensor]
    pair_state_ref: Optional[torch.Tensor]
    head_output: torch.Tensor
    heads: int
    message_dim: int


@dataclass
class NodeStateRecord:
    layer: int
    x: torch.Tensor
    x_ref: torch.Tensor
    graph: torch.Tensor
    local_index: torch.Tensor


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_csv(raw: str, *, all_values: Sequence[str]) -> tuple[str, ...]:
    if raw == "all":
        return tuple(all_values)
    values = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    unknown = sorted(set(values) - set(all_values))
    if unknown:
        raise ValueError(f"unknown values {unknown}; expected {list(all_values)} or all")
    return values


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def git_sha(cwd: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def write_table(path: Path, rows: Sequence[Mapping[str, Any]], *, parquet: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path.with_suffix(".csv") if path.suffix == ".parquet" else path
    moa.write_csv(csv_path, rows)
    if parquet:
        parquet_path = path.with_suffix(".parquet")
        try:
            import pandas as pd

            pd.DataFrame(list(rows)).to_parquet(parquet_path, index=False)
            return parquet_path
        except Exception as exc:
            print(f"[write] parquet skipped for {parquet_path}: {exc}; wrote {csv_path}", flush=True)
    return csv_path


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        alt = path.with_suffix(".csv") if path.suffix == ".parquet" else path.with_suffix(".parquet")
        if alt.exists():
            path = alt
    if path.suffix == ".parquet":
        try:
            import pandas as pd

            return pd.read_parquet(path).to_dict("records")
        except Exception as exc:
            raise RuntimeError(f"could not read parquet {path}: {exc}") from exc
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def graph_hash(graph: Any) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(graph.task_type).encode("utf-8"))
    hasher.update(str(int(graph.num_nodes)).encode("utf-8"))
    for tensor in (graph.node_type, graph.edge_index, graph.edge_value, graph.target):
        t = tensor.detach().cpu().contiguous()
        hasher.update(str(tuple(t.shape)).encode("utf-8"))
        hasher.update(str(t.dtype).encode("utf-8"))
        hasher.update(t.numpy().tobytes())
    return hasher.hexdigest()[:16]


def graph_partition(graph_id: str, selection_fraction: float) -> str:
    value = int(hashlib.sha256(graph_id.encode("utf-8")).hexdigest()[:12], 16)
    frac = value / float(16**12)
    return "selection" if frac < float(selection_fraction) else "evaluation"


def bucket_index(value: float, thresholds: Sequence[float]) -> int:
    if not math.isfinite(value):
        return -1
    out = 0
    for threshold in thresholds:
        if value >= threshold:
            out += 1
    return out


def quantile_buckets(values: Sequence[float], buckets: int = 10) -> list[int]:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return [-1 for _ in values]
    cuts = [finite[min(len(finite) - 1, max(0, round(q * (len(finite) - 1) / buckets)))] for q in range(1, buckets)]
    return [bucket_index(v, cuts) for v in values]


def degree_bucket(value: float) -> int:
    return bucket_index(value, (1, 2, 4, 8, 16, 32, 64, 128))


def csv_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def csv_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def csv_float(value: Any, default: float = 0.0) -> float:
    return finite_float(value, default)


def source_sink_from_graph(graph: Any) -> tuple[int, int]:
    node_type = graph.node_type.detach().cpu().long()
    active = node_type[: int(graph.num_nodes)]
    nonzero = torch.unique(active[active != 0])
    if nonzero.numel() < 2:
        raise RuntimeError("flow source/sink roles are not recoverable from node_type")
    source_nodes = torch.nonzero(active == int(nonzero.min()), as_tuple=False).reshape(-1)
    sink_nodes = torch.nonzero(active == int(nonzero.max()), as_tuple=False).reshape(-1)
    if source_nodes.numel() != 1 or sink_nodes.numel() != 1:
        raise RuntimeError("flow sensitivity currently expects exactly one source and one sink")
    return int(source_nodes.item()), int(sink_nodes.item())


def solver_edge_values(graph: Any, args: argparse.Namespace) -> torch.Tensor:
    values = graph.edge_value.detach().cpu().float()
    if args.solver_edge_value_source == "unnormalize":
        if args.solver_edge_value_mean is None or args.solver_edge_value_std is None:
            raise RuntimeError(
                "--solver-edge-value-source unnormalize requires "
                "--solver-edge-value-mean and --solver-edge-value-std"
            )
        values = values * float(args.solver_edge_value_std) + float(args.solver_edge_value_mean)
    if args.solver_capacity_clamp_min is not None:
        values = values.clamp_min(float(args.solver_capacity_clamp_min))
    return values


def model_value_from_solver(value: float, args: argparse.Namespace) -> float:
    if args.solver_edge_value_source == "unnormalize":
        return (float(value) - float(args.solver_edge_value_mean)) / float(args.solver_edge_value_std)
    return float(value)


def capacity_matrix_from_graph(graph: Any, args: argparse.Namespace) -> torch.Tensor:
    n = int(graph.num_nodes)
    capacity = torch.zeros(n, n, dtype=torch.float64)
    edge_index = graph.edge_index.detach().cpu().long()
    values = solver_edge_values(graph, args).double()
    for edge_idx in range(edge_index.size(1)):
        src = int(edge_index[0, edge_idx])
        dst = int(edge_index[1, edge_idx])
        if 0 <= src < n and 0 <= dst < n:
            capacity[dst, src] += float(values[edge_idx])
    return capacity


def directed_mask_from_capacity(capacity: torch.Tensor) -> torch.Tensor:
    return capacity > 0


def edmonds_karp(capacity: torch.Tensor, source: int, sink: int) -> FlowSolveResult:
    n = int(capacity.size(0))
    residual = capacity.detach().cpu().double().clone()
    total = 0.0
    while True:
        parent = [-1] * n
        parent[source] = source
        queue = deque([source])
        while queue and parent[sink] < 0:
            u = queue.popleft()
            for v in range(n):
                if parent[v] < 0 and float(residual[v, u]) > 1.0e-12:
                    parent[v] = u
                    queue.append(v)
                    if v == sink:
                        break
        if parent[sink] < 0:
            break
        aug = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            aug = min(aug, float(residual[v, u]))
            v = u
        v = sink
        while v != source:
            u = parent[v]
            residual[v, u] -= aug
            residual[u, v] += aug
            v = u
        total += aug

    side = torch.zeros(n, dtype=torch.bool)
    side[source] = True
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in range(n):
            if not bool(side[v]) and float(residual[v, u]) > 1.0e-12:
                side[v] = True
                queue.append(v)
    return FlowSolveResult(value=total, residual_forward=residual.float(), source_side=side)


def directed_distances_from_source(directed: torch.Tensor, source: int) -> list[int]:
    n = int(directed.size(0))
    dist = [-1] * n
    dist[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in torch.nonzero(directed[:, u], as_tuple=False).reshape(-1).tolist():
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(int(v))
    return dist


def directed_distances_to_sink(directed: torch.Tensor, sink: int) -> list[int]:
    n = int(directed.size(0))
    dist = [-1] * n
    dist[sink] = 0
    queue = deque([sink])
    while queue:
        v = queue.popleft()
        for u in torch.nonzero(directed[v], as_tuple=False).reshape(-1).tolist():
            if dist[u] < 0:
                dist[u] = dist[v] + 1
                queue.append(int(u))
    return dist


def flow_sensitivity_rows(
    graph: Any,
    *,
    graph_position: int,
    graph_index: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    graph_id = graph_hash(graph)
    partition = graph_partition(graph_id, args.selection_fraction)
    n = int(graph.num_nodes)
    source, sink = source_sink_from_graph(graph)
    capacity = capacity_matrix_from_graph(graph, args)
    directed = directed_mask_from_capacity(capacity)
    base = edmonds_karp(capacity, source, sink)
    dist_source = directed_distances_from_source(directed, source)
    dist_sink = directed_distances_to_sink(directed, sink)
    undirected_degree = (directed | directed.T).float().sum(dim=0).tolist()
    edge_index = graph.edge_index.detach().cpu().long()
    solver_values = solver_edge_values(graph, args).double()
    capacities = [float(max(0.0, solver_values[i].item())) for i in range(edge_index.size(1))]
    capacity_buckets = quantile_buckets(capacities, buckets=10)
    rows: list[dict[str, Any]] = []
    eta = float(args.flow_eta)

    for edge_pos in range(edge_index.size(1)):
        sender = int(edge_index[0, edge_pos])
        receiver = int(edge_index[1, edge_pos])
        current_capacity = float(max(0.0, solver_values[edge_pos].item()))
        plus_capacity = capacity.clone()
        plus_capacity[receiver, sender] = plus_capacity[receiver, sender] + eta
        minus_capacity = capacity.clone()
        minus_capacity[receiver, sender] = max(0.0, float(minus_capacity[receiver, sender]) - eta)
        plus_value = edmonds_karp(plus_capacity, source, sink).value
        minus_value = edmonds_karp(minus_capacity, source, sink).value
        delta_plus = plus_value - base.value
        delta_minus = base.value - minus_value
        if delta_plus >= delta_minus:
            perturb_direction = "increase"
            perturb_solver_value = current_capacity + eta
            target_delta = delta_plus
        else:
            perturb_direction = "decrease"
            perturb_solver_value = max(0.0, current_capacity - eta)
            target_delta = -delta_minus
        perturb_model_value = model_value_from_solver(perturb_solver_value, args)
        source_side = bool(base.source_side[sender])
        receiver_side = bool(base.source_side[receiver])
        unit_id = f"{graph_id}:edge:{sender}->{receiver}:{edge_pos}"
        rows.append(
            {
                "unit_id": unit_id,
                "task_family": "flow",
                "graph_position": graph_position,
                "graph_index": graph_index,
                "graph_id": graph_id,
                "partition": partition,
                "num_nodes": n,
                "unit_kind": "edge",
                "edge_position": edge_pos,
                "sender": sender,
                "receiver": receiver,
                "reverse_sender": receiver,
                "reverse_receiver": sender,
                "edge_exists": 1,
                "critical": int(max(delta_plus, delta_minus) > float(args.critical_threshold)),
                "sensitivity": max(delta_plus, delta_minus),
                "target_delta": target_delta,
                "base_solver_value": base.value,
                "perturbation": perturb_direction,
                "perturb_solver_value": perturb_solver_value,
                "perturb_model_value": perturb_model_value,
                "model_edge_value": float(graph.edge_value.detach().cpu().float()[edge_pos]),
                "capacity_solver": current_capacity,
                "capacity_bucket_10": capacity_buckets[edge_pos],
                "saturated_e": int(float(base.residual_forward[receiver, sender]) <= 1.0e-8),
                "mincut_crossing_e": int(source_side and not receiver_side),
                "source_distance_tail": dist_source[sender],
                "sink_distance_head": dist_sink[receiver],
                "tail_degree_bucket": degree_bucket(float(undirected_degree[sender])),
                "head_degree_bucket": degree_bucket(float(undirected_degree[receiver])),
                "match_signature": (
                    f"sat={int(float(base.residual_forward[receiver, sender]) <= 1.0e-8)};"
                    f"cap={capacity_buckets[edge_pos]};"
                    f"ds={dist_source[sender]};dt={dist_sink[receiver]};"
                    f"du={degree_bucket(float(undirected_degree[sender]))};"
                    f"dv={degree_bucket(float(undirected_degree[receiver]))}"
                ),
            }
        )
    return rows


def infer_bipartition_from_edges(num_nodes: int, edge_index: torch.Tensor) -> list[bool]:
    adj = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.detach().cpu().long().T.tolist():
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            adj[src].append(dst)
            adj[dst].append(src)
    colour = [-1] * num_nodes
    for start in range(num_nodes):
        if colour[start] >= 0:
            continue
        colour[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neigh in adj[node]:
                if colour[neigh] < 0:
                    colour[neigh] = 1 - colour[node]
                    queue.append(neigh)
    return [c == 0 for c in colour]


def canonical_matching_edges(
    num_nodes: int,
    edge_index: torch.Tensor,
    left: Sequence[bool],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], list[int]]]:
    edges: set[tuple[int, int]] = set()
    positions: dict[tuple[int, int], list[int]] = defaultdict(list)
    for pos, (src, dst) in enumerate(edge_index.detach().cpu().long().T.tolist()):
        if src == dst or src < 0 or dst < 0 or src >= num_nodes or dst >= num_nodes:
            continue
        if bool(left[src]) == bool(left[dst]):
            continue
        lnode, rnode = (src, dst) if bool(left[src]) else (dst, src)
        edges.add((lnode, rnode))
        positions[(lnode, rnode)].append(pos)
    return edges, positions


def matching_size(num_nodes: int, edges: set[tuple[int, int]], left: Sequence[bool]) -> tuple[int, set[tuple[int, int]]]:
    try:
        import networkx as nx
    except Exception as exc:
        raise RuntimeError("bipartite matching labels require networkx") from exc
    graph = nx.Graph()
    left_nodes = {idx for idx in range(num_nodes) if bool(left[idx])}
    right_nodes = set(range(num_nodes)) - left_nodes
    graph.add_nodes_from(left_nodes, bipartite=0)
    graph.add_nodes_from(right_nodes, bipartite=1)
    graph.add_edges_from(edges)
    matched = nx.algorithms.bipartite.maximum_matching(graph, top_nodes=left_nodes)
    matched_edges = {
        (u, v)
        for u, v in matched.items()
        if u in left_nodes and v in right_nodes and (u, v) in edges
    }
    return len(matched_edges), matched_edges


def component_sizes(num_nodes: int, edges: set[tuple[int, int]]) -> list[int]:
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    sizes = [1] * num_nodes
    seen = [False] * num_nodes
    for start in range(num_nodes):
        if seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        comp = []
        while queue:
            node = queue.popleft()
            comp.append(node)
            for neigh in adj[node]:
                if not seen[neigh]:
                    seen[neigh] = True
                    queue.append(neigh)
        for node in comp:
            sizes[node] = len(comp)
    return sizes


def sample_matching_nonedges(
    nonedges: list[tuple[int, int]],
    degree_l: Mapping[int, int],
    degree_r: Mapping[int, int],
    *,
    max_count: int,
    seed: int,
) -> list[tuple[int, int]]:
    if len(nonedges) <= max_count:
        return nonedges
    rng = random.Random(seed)
    groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for edge in nonedges:
        groups[(degree_bucket(degree_l[edge[0]]), degree_bucket(degree_r[edge[1]]))].append(edge)
    sampled: list[tuple[int, int]] = []
    group_keys = sorted(groups)
    while len(sampled) < max_count and group_keys:
        next_keys = []
        for key in group_keys:
            pool = groups[key]
            if not pool:
                continue
            idx = rng.randrange(len(pool))
            sampled.append(pool.pop(idx))
            if pool:
                next_keys.append(key)
            if len(sampled) >= max_count:
                break
        group_keys = next_keys
    return sampled


def matching_sensitivity_rows(
    graph: Any,
    *,
    graph_position: int,
    graph_index: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    graph_id = graph_hash(graph)
    partition = graph_partition(graph_id, args.selection_fraction)
    n = int(graph.num_nodes)
    edge_index = graph.edge_index.detach().cpu().long()
    left = infer_bipartition_from_edges(n, edge_index)
    edges, positions = canonical_matching_edges(n, edge_index, left)
    left_nodes = [idx for idx in range(n) if left[idx]]
    right_nodes = [idx for idx in range(n) if not left[idx]]
    base_size, optimum = matching_size(n, edges, left)
    comp = component_sizes(n, edges)
    degree_l = {u: 0 for u in left_nodes}
    degree_r = {v: 0 for v in right_nodes}
    for u, v in edges:
        degree_l[u] += 1
        degree_r[v] += 1
    all_nonedges = [(u, v) for u in left_nodes for v in right_nodes if (u, v) not in edges]
    nonedges = sample_matching_nonedges(
        all_nonedges,
        degree_l,
        degree_r,
        max_count=max(1, int(args.matching_nonedges_per_edge) * max(1, len(edges))),
        seed=int(args.random_seed) + int(graph_index),
    )

    rows: list[dict[str, Any]] = []
    for lnode, rnode in sorted(edges):
        deleted = set(edges)
        deleted.remove((lnode, rnode))
        deleted_size, _deleted_opt = matching_size(n, deleted, left)
        delta = base_size - deleted_size
        edge_positions = positions[(lnode, rnode)]
        edge_position = edge_positions[0] if edge_positions else -1
        unit_id = f"{graph_id}:match_edge:{lnode}-{rnode}"
        rows.append(
            {
                "unit_id": unit_id,
                "task_family": "matching",
                "graph_position": graph_position,
                "graph_index": graph_index,
                "graph_id": graph_id,
                "partition": partition,
                "num_nodes": n,
                "unit_kind": "edge",
                "edge_position": edge_position,
                "all_edge_positions": ",".join(str(pos) for pos in edge_positions),
                "sender": lnode,
                "receiver": rnode,
                "reverse_sender": rnode,
                "reverse_receiver": lnode,
                "edge_exists": 1,
                "critical": int(delta > float(args.critical_threshold)),
                "sensitivity": float(delta),
                "target_delta": float(-delta),
                "base_solver_value": float(base_size),
                "perturbation": "delete",
                "perturb_solver_value": 0.0,
                "perturb_model_value": 0.0,
                "capacity_solver": 1.0,
                "capacity_bucket_10": 0,
                "in_one_deterministic_optimum_matching": int((lnode, rnode) in optimum),
                "left_degree_bucket": degree_bucket(degree_l[lnode]),
                "right_degree_bucket": degree_bucket(degree_r[rnode]),
                "component_size_bucket": degree_bucket(max(comp[lnode], comp[rnode])),
                "match_signature": (
                    f"ld={degree_bucket(degree_l[lnode])};"
                    f"rd={degree_bucket(degree_r[rnode])};"
                    f"cs={degree_bucket(max(comp[lnode], comp[rnode]))};"
                    f"opt={int((lnode, rnode) in optimum)}"
                ),
            }
        )

    for lnode, rnode in sorted(nonedges):
        added = set(edges)
        added.add((lnode, rnode))
        added_size, _added_opt = matching_size(n, added, left)
        delta = added_size - base_size
        unit_id = f"{graph_id}:match_nonedge:{lnode}-{rnode}"
        rows.append(
            {
                "unit_id": unit_id,
                "task_family": "matching",
                "graph_position": graph_position,
                "graph_index": graph_index,
                "graph_id": graph_id,
                "partition": partition,
                "num_nodes": n,
                "unit_kind": "nonedge",
                "edge_position": -1,
                "all_edge_positions": "",
                "sender": lnode,
                "receiver": rnode,
                "reverse_sender": rnode,
                "reverse_receiver": lnode,
                "edge_exists": 0,
                "critical": int(delta > float(args.critical_threshold)),
                "sensitivity": float(delta),
                "target_delta": float(delta),
                "base_solver_value": float(base_size),
                "perturbation": "add",
                "perturb_solver_value": 1.0,
                "perturb_model_value": float(args.matching_added_edge_value),
                "capacity_solver": 0.0,
                "capacity_bucket_10": 0,
                "in_one_deterministic_optimum_matching": 0,
                "left_degree_bucket": degree_bucket(degree_l[lnode]),
                "right_degree_bucket": degree_bucket(degree_r[rnode]),
                "component_size_bucket": degree_bucket(max(comp[lnode], comp[rnode])),
                "match_signature": (
                    f"ld={degree_bucket(degree_l[lnode])};"
                    f"rd={degree_bucket(degree_r[rnode])};"
                    f"cs={degree_bucket(max(comp[lnode], comp[rnode]))}"
                ),
            }
        )
    return rows


def task_family(task: str) -> str:
    lowered = task.lower()
    if "flow" in lowered:
        return "flow"
    if "matching" in lowered or "bipartite" in lowered:
        return "matching"
    return "external"


def run_labels(args: argparse.Namespace, loaded: moa.LoadedExperiment) -> list[dict[str, Any]]:
    if args.sensitivity_units_path is not None and args.sensitivity_units_path.exists() and not args.force_recompute:
        print(f"[labels] using existing sensitivity units: {args.sensitivity_units_path}", flush=True)
        return read_rows(args.sensitivity_units_path)

    family = task_family(args.task)
    if family == "external":
        if args.sensitivity_units_path is None:
            raise RuntimeError(
                f"task {args.task!r} has no built-in solver labels; pass --sensitivity-units-path"
            )
        return read_rows(args.sensitivity_units_path)

    rows: list[dict[str, Any]] = []
    for position, (graph_index, graph) in enumerate(zip(loaded.selected_indices, loaded.graphs)):
        if family == "flow":
            graph_rows = flow_sensitivity_rows(
                graph,
                graph_position=position,
                graph_index=int(graph_index),
                args=args,
            )
        elif family == "matching":
            graph_rows = matching_sensitivity_rows(
                graph,
                graph_position=position,
                graph_index=int(graph_index),
                args=args,
            )
        else:
            raise AssertionError(family)
        rows.extend(graph_rows)
        if (position + 1) % max(1, int(args.progress_every_graphs)) == 0 or position + 1 == len(loaded.graphs):
            positives = sum(csv_int(row["critical"]) for row in rows)
            print(
                f"[labels] graphs={position + 1}/{len(loaded.graphs)} "
                f"units={len(rows)} positives={positives}",
                flush=True,
            )
    write_table(args.output_dir / "sensitivity_units.csv", rows, parquet=args.write_parquet)
    return rows


class GRITCausalCollector:
    """Capture GRIT pair fields and node states with gradients retained."""

    def __init__(self, model: nn.Module) -> None:
        if not hasattr(model, "layers"):
            raise TypeError("task-causal protocol expects an official GRIT/static-GRIT model")
        self.model = model
        self.pair_records: dict[int, GRITCausalLayerRecord] = {}
        self.node_records: dict[int, NodeStateRecord] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> "GRITCausalCollector":
        self.pair_records = {}
        self.node_records = {}
        self.handles = []
        for layer_idx, layer in enumerate(self.model.layers):
            self.handles.append(layer.attention.register_forward_hook(self._make_attention_hook(layer_idx)))
            self.handles.append(layer.register_forward_hook(self._make_layer_hook(layer_idx)))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _make_attention_hook(self, layer_idx: int):
        def hook(module: nn.Module, inputs: tuple[Any, ...], outputs: Any) -> None:
            pyg_batch = inputs[0]
            head_output = outputs[0] if isinstance(outputs, tuple) else outputs
            if not torch.is_tensor(head_output):
                raise RuntimeError("GRIT attention hook did not receive a tensor output")
            if head_output.requires_grad:
                head_output.retain_grad()
            edge_index = pyg_batch.edge_index.long()
            src = edge_index[0]
            dst = edge_index[1]
            attention = pyg_batch.attn.squeeze(-1)
            if attention.dim() != 2:
                raise RuntimeError(f"expected sparse GRIT attention [E,H], got {tuple(attention.shape)}")
            node_msg, pair_msg, logits, _edge_state = moa.grit_attention_components(module, pyg_batch)
            counts = [int(value) for value in pyg_batch.graph_num_nodes.detach().cpu().tolist()]
            graph, local_src, local_dst = moa.edge_local_coordinates(src, dst, counts)
            pair_state_ref = getattr(pyg_batch, "E", None)
            if torch.is_tensor(pair_state_ref) and pair_state_ref.requires_grad:
                pair_state_ref.retain_grad()
            else:
                pair_state_ref = None
            pair_state = None
            if torch.is_tensor(getattr(pyg_batch, "E", None)):
                pair_state = pyg_batch.E.detach().float()
            self.pair_records[layer_idx] = GRITCausalLayerRecord(
                layer=layer_idx,
                src=src.detach(),
                dst=dst.detach(),
                graph=graph.detach(),
                local_src=local_src.detach(),
                local_dst=local_dst.detach(),
                attention=attention.detach().float(),
                logits=logits.detach().float(),
                node_message=node_msg.detach().float(),
                pair_message=pair_msg.detach().float(),
                message=(node_msg + pair_msg).detach().float(),
                pair_state=pair_state,
                pair_state_ref=pair_state_ref,
                head_output=head_output,
                heads=int(attention.size(1)),
                message_dim=int(node_msg.size(-1)),
            )

        return hook

    def _make_layer_hook(self, layer_idx: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], outputs: Any) -> None:
            pyg_batch = outputs[0] if isinstance(outputs, tuple) else outputs
            x = getattr(pyg_batch, "x", None)
            counts = getattr(pyg_batch, "graph_num_nodes", None)
            if not torch.is_tensor(x) or counts is None:
                return
            if x.requires_grad:
                x.retain_grad()
            count_list = [int(value) for value in counts.detach().cpu().tolist()]
            graph_ids: list[int] = []
            local_ids: list[int] = []
            for graph_idx, count in enumerate(count_list):
                graph_ids.extend([graph_idx] * count)
                local_ids.extend(range(count))
            graph = torch.tensor(graph_ids, dtype=torch.long, device=x.device)
            local = torch.tensor(local_ids, dtype=torch.long, device=x.device)
            self.node_records[layer_idx] = NodeStateRecord(
                layer=layer_idx,
                x=x.detach().float(),
                x_ref=x,
                graph=graph,
                local_index=local,
            )

        return hook


def predict_with_causal_capture(
    args: argparse.Namespace,
    loaded: moa.LoadedExperiment,
    batch: Any,
    *,
    backward: bool,
) -> tuple[torch.Tensor, GRITCausalCollector]:
    loaded.model.eval()
    loaded.model.zero_grad(set_to_none=True)
    collector = GRITCausalCollector(loaded.model)
    with collector:
        manager = torch.enable_grad() if backward else torch.no_grad()
        with manager:
            with moa.autocast_context(loaded.device, args.autocast_dtype):
                pred = loaded.model(batch)
            if backward:
                scalar = moa.task_scalar(pred, batch)
                scalar.backward()
    return pred.detach(), collector


def pair_positions(
    record: GRITCausalLayerRecord,
    *,
    graph_idx: int,
    pair_set: Sequence[tuple[int, int]],
) -> list[int]:
    out: list[int] = []
    for receiver, sender in pair_set:
        mask = (
            (record.graph.cpu() == int(graph_idx))
            & (record.local_dst.cpu() == int(receiver))
            & (record.local_src.cpu() == int(sender))
        )
        out.extend(torch.nonzero(mask, as_tuple=False).reshape(-1).tolist())
    return out


def node_positions(
    record: NodeStateRecord,
    *,
    graph_idx: int,
    nodes: Sequence[int],
) -> list[int]:
    out: list[int] = []
    graph_cpu = record.graph.detach().cpu()
    local_cpu = record.local_index.detach().cpu()
    for node in nodes:
        mask = (graph_cpu == int(graph_idx)) & (local_cpu == int(node))
        out.extend(torch.nonzero(mask, as_tuple=False).reshape(-1).tolist())
    return out


def unit_pair_set(unit: Mapping[str, Any]) -> list[tuple[int, int]]:
    receiver = csv_int(unit.get("receiver"))
    sender = csv_int(unit.get("sender"))
    reverse_receiver = csv_int(unit.get("reverse_receiver", sender))
    reverse_sender = csv_int(unit.get("reverse_sender", receiver))
    if unit.get("task_family") == "flow":
        return [(receiver, sender), (reverse_receiver, reverse_sender)]
    return [(receiver, sender), (reverse_receiver, reverse_sender)]


def unit_endpoint_set(unit: Mapping[str, Any]) -> list[int]:
    return sorted({csv_int(unit.get("sender")), csv_int(unit.get("receiver"))})


def head_output_grad(record: GRITCausalLayerRecord) -> torch.Tensor:
    grad = record.head_output.grad
    if grad is None:
        raise RuntimeError("missing retained GRIT head-output gradient")
    if grad.dim() == 2:
        return grad.reshape(grad.size(0), record.heads, -1).float()
    if grad.dim() == 3:
        return grad.float()
    raise RuntimeError(f"unexpected head-output grad shape {tuple(grad.shape)}")


def pair_state_scores_for_unit(
    record: GRITCausalLayerRecord,
    positions: Sequence[int],
) -> list[tuple[int, float]]:
    if record.pair_state is None or record.pair_state_ref is None or record.pair_state_ref.grad is None:
        return []
    if not positions:
        return []
    act = record.pair_state.float()
    grad = record.pair_state_ref.grad.detach().float().cpu()
    pos = torch.tensor(list(positions), dtype=torch.long)
    values = (act[pos].cpu() * grad[pos]).abs()
    if values.dim() == 1:
        return [(-1, float(values.sum()))]
    if values.size(-1) % record.heads == 0:
        view = values.reshape(values.size(0), record.heads, -1)
        return [(head, float(view[:, head].sum())) for head in range(record.heads)]
    return [(-1, float(values.sum()))]


def reduce_alignment_for_batch(
    unit_rows: Sequence[Mapping[str, Any]],
    graph_indices: Sequence[int],
    collector: GRITCausalCollector,
) -> list[dict[str, Any]]:
    by_graph: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in unit_rows:
        by_graph[csv_int(unit.get("graph_index"))].append(unit)
    rows: list[dict[str, Any]] = []
    graph_index_to_local = {int(graph_index): local for local, graph_index in enumerate(graph_indices)}

    for layer, record in sorted(collector.pair_records.items()):
        grad = head_output_grad(record).detach().cpu()
        for graph_index, units in by_graph.items():
            if graph_index not in graph_index_to_local:
                continue
            local_graph = graph_index_to_local[graph_index]
            for unit in units:
                positions = pair_positions(record, graph_idx=local_graph, pair_set=unit_pair_set(unit))
                if not positions:
                    continue
                pos = torch.tensor(positions, dtype=torch.long)
                dst = record.dst.detach().cpu()[pos].long()
                attention = record.attention.cpu()[pos]
                message = record.message.cpu()[pos]
                pair_message = record.pair_message.cpu()[pos]
                logits = record.logits.cpu()[pos]
                for head in range(record.heads):
                    grad_dst = grad[dst, head]
                    attn = attention[:, head]
                    msg = message[:, head]
                    pair_msg = pair_message[:, head]
                    routing = ((grad_dst * msg).sum(dim=-1) * attn).abs().sum()
                    pair_value = ((grad_dst * pair_msg).sum(dim=-1) * attn).abs().sum()
                    rows.append(
                        {
                            "unit_id": unit["unit_id"],
                            "graph_index": graph_index,
                            "graph_id": unit["graph_id"],
                            "partition": unit["partition"],
                            "layer": layer,
                            "head": head,
                            "site_family": "attention_mass",
                            "score": float(attn.abs().sum()),
                        }
                    )
                    rows.append(
                        {
                            "unit_id": unit["unit_id"],
                            "graph_index": graph_index,
                            "graph_id": unit["graph_id"],
                            "partition": unit["partition"],
                            "layer": layer,
                            "head": head,
                            "site_family": "routing_influence",
                            "score": float(routing),
                            "aux_logit_abs": float(logits[:, head].abs().sum()),
                        }
                    )
                    rows.append(
                        {
                            "unit_id": unit["unit_id"],
                            "graph_index": graph_index,
                            "graph_id": unit["graph_id"],
                            "partition": unit["partition"],
                            "layer": layer,
                            "head": head,
                            "site_family": "pair_value_influence",
                            "score": float(pair_value),
                        }
                    )
                for head, score in pair_state_scores_for_unit(record, positions):
                    rows.append(
                        {
                            "unit_id": unit["unit_id"],
                            "graph_index": graph_index,
                            "graph_id": unit["graph_id"],
                            "partition": unit["partition"],
                            "layer": layer,
                            "head": head,
                            "site_family": "pair_state_influence",
                            "score": score,
                        }
                    )

    for layer, record in sorted(collector.node_records.items()):
        grad_ref = record.x_ref.grad
        if grad_ref is None:
            continue
        grad = grad_ref.detach().float().cpu()
        act = record.x.detach().float().cpu()
        for graph_index, units in by_graph.items():
            if graph_index not in graph_index_to_local:
                continue
            local_graph = graph_index_to_local[graph_index]
            for unit in units:
                positions = node_positions(record, graph_idx=local_graph, nodes=unit_endpoint_set(unit))
                if not positions:
                    continue
                pos = torch.tensor(positions, dtype=torch.long)
                score = float((grad[pos] * act[pos]).abs().sum())
                rows.append(
                    {
                        "unit_id": unit["unit_id"],
                        "graph_index": graph_index,
                        "graph_id": unit["graph_id"],
                        "partition": unit["partition"],
                        "layer": layer,
                        "head": -1,
                        "site_family": "node_endpoint_influence",
                        "score": score,
                    }
                )
    return rows


def normalise_score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[tuple[Any, ...], float] = defaultdict(float)
    for row in rows:
        key = (row["graph_index"], row["site_family"], row["layer"], row["head"])
        totals[key] += max(0.0, finite_float(row.get("score"), 0.0))
    for row in rows:
        key = (row["graph_index"], row["site_family"], row["layer"], row["head"])
        row["normalised_score"] = finite_float(row.get("score"), 0.0) / max(EPS, totals[key])
    return rows


def iter_analysis_batches(args: argparse.Namespace, loaded: moa.LoadedExperiment):
    batch_size = max(1, int(args.analysis_cache_batch_size or args.batch_size))
    if loaded.analysis_batches:
        for graph_indices, cached_batch in loaded.analysis_batches:
            yield graph_indices, cached_batch
    else:
        for graph_indices, graphs in moa.graph_batches(loaded.graphs, loaded.selected_indices, batch_size):
            yield graph_indices, loaded.runner.collate_graphs(graphs)


def run_alignment(
    args: argparse.Namespace,
    loaded: moa.LoadedExperiment,
    unit_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[SiteSelection]]:
    score_path = args.output_dir / "clean_internal_scores.csv"
    metric_path = args.output_dir / "alignment_metrics.csv"
    site_path = args.output_dir / "site_selection.json"
    if score_path.exists() and metric_path.exists() and site_path.exists() and not args.force_recompute:
        rows = read_rows(score_path)
        metrics = read_rows(metric_path)
        sites = load_site_selection(site_path)
        print(f"[alignment] using existing clean scores: {score_path}", flush=True)
        return rows, metrics, sites

    if args.model not in {"grit", "static_grit"} or args.model_backend != "official":
        raise RuntimeError("alignment currently supports official GRIT/static-GRIT only")

    score_rows: list[dict[str, Any]] = []
    for batch_idx, (graph_indices, batch_item) in enumerate(iter_analysis_batches(args, loaded)):
        batch = moa.batch_to_device(batch_item, loaded.device)
        _pred, collector = predict_with_causal_capture(args, loaded, batch, backward=True)
        batch_rows = reduce_alignment_for_batch(unit_rows, graph_indices, collector)
        score_rows.extend(batch_rows)
        print(
            f"[alignment] batch={batch_idx + 1} graphs={len(graph_indices)} "
            f"rows={len(score_rows)}",
            flush=True,
        )
        collector.pair_records.clear()
        collector.node_records.clear()
        del batch, collector
        if args.empty_cache_every_batch and loaded.device.type == "cuda":
            torch.cuda.empty_cache()

    score_rows = normalise_score_rows(score_rows)
    metrics = alignment_metrics(unit_rows, score_rows, seed=args.random_seed)
    sites = select_sites(metrics, args)
    write_table(score_path, score_rows, parquet=args.write_parquet)
    write_table(metric_path, metrics, parquet=False)
    write_site_selection(site_path, sites)
    return score_rows, metrics, sites


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(int(y), float(s)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
    positives = sum(y for y, _s in pairs)
    if positives <= 0:
        return float("nan")
    pairs.sort(key=lambda item: item[1], reverse=True)
    hit = 0
    total = 0.0
    for rank, (label, _score) in enumerate(pairs, start=1):
        if label:
            hit += 1
            total += hit / rank
    return total / positives


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    pairs = [(float(s), int(y)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
    pos = sum(y for _s, y in pairs)
    neg = len(pairs) - pos
    if pos <= 0 or neg <= 0:
        return float("nan")
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    idx = 0
    while idx < len(pairs):
        j = idx + 1
        while j < len(pairs) and pairs[j][0] == pairs[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        rank_sum += avg_rank * sum(label for _score, label in pairs[idx:j])
        idx = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def mean_graph_lift_at_k(labels: Sequence[int], scores: Sequence[float], graph_ids: Sequence[Any]) -> float:
    by_graph: dict[Any, list[tuple[int, float]]] = defaultdict(list)
    for y, s, graph_id in zip(labels, scores, graph_ids):
        if math.isfinite(float(s)):
            by_graph[graph_id].append((int(y), float(s)))
    lifts: list[float] = []
    for pairs in by_graph.values():
        positives = sum(y for y, _s in pairs)
        if positives <= 0 or positives >= len(pairs):
            continue
        pairs.sort(key=lambda item: item[1], reverse=True)
        precision = sum(y for y, _s in pairs[:positives]) / positives
        base = positives / len(pairs)
        lifts.append(precision / max(EPS, base))
    return sum(lifts) / len(lifts) if lifts else float("nan")


def spearman(labels: Sequence[float], scores: Sequence[float]) -> float:
    pairs = [(float(a), float(b)) for a, b in zip(labels, scores) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 3:
        return float("nan")

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        idx = 0
        while idx < len(order):
            j = idx + 1
            while j < len(order) and values[order[j]] == values[order[idx]]:
                j += 1
            rank = (idx + 1 + j) / 2.0
            for k in order[idx:j]:
                out[k] = rank
            idx = j
        return out

    xs = ranks([a for a, _b in pairs])
    ys = ranks([b for _a, b in pairs])
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(max(EPS, vx * vy))


def alignment_metrics(
    unit_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    del seed
    unit_by_id = {str(row["unit_id"]): row for row in unit_rows}
    base_fields = [
        "capacity_solver",
        "saturated_e",
        "mincut_crossing_e",
        "source_distance_tail",
        "sink_distance_head",
        "in_one_deterministic_optimum_matching",
        "left_degree_bucket",
        "right_degree_bucket",
        "component_size_bucket",
    ]
    rows: list[dict[str, Any]] = []

    def metric_row(
        *,
        partition: str,
        score_family: str,
        layer: int,
        head: int,
        scorer: str,
        items: list[tuple[Mapping[str, Any], float]],
    ) -> dict[str, Any]:
        labels = [csv_int(unit.get("critical")) for unit, _score in items]
        sensitivities = [csv_float(unit.get("sensitivity")) for unit, _score in items]
        scores = [float(score) for _unit, score in items]
        graph_ids = [unit.get("graph_id") for unit, _score in items]
        return {
            "partition": partition,
            "site_family": score_family,
            "layer": layer,
            "head": head,
            "scorer": scorer,
            "samples": len(items),
            "positives": sum(labels),
            "base_rate": sum(labels) / len(labels) if labels else float("nan"),
            "auprc": average_precision(labels, scores),
            "auroc": auroc(labels, scores),
            "lift_at_k": mean_graph_lift_at_k(labels, scores, graph_ids),
            "spearman_sensitivity": spearman(sensitivities, scores),
        }

    units_by_partition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in unit_rows:
        units_by_partition[str(unit.get("partition", ""))].append(unit)
    for partition, units in sorted(units_by_partition.items()):
        random_values = [
            (
                unit,
                int(hashlib.sha256(str(unit.get("unit_id")).encode("utf-8")).hexdigest()[:12], 16)
                / float(16**12),
            )
            for unit in units
        ]
        if random_values:
            rows.append(
                metric_row(
                    partition=partition,
                    score_family="random",
                    layer=-1,
                    head=-1,
                    scorer="baseline",
                    items=random_values,
                )
            )
        for field in base_fields:
            values = [(unit, csv_float(unit.get(field), float("nan"))) for unit in units if field in unit]
            values = [(unit, score) for unit, score in values if math.isfinite(score)]
            if values:
                rows.append(
                    metric_row(
                        partition=partition,
                        score_family=field,
                        layer=-1,
                        head=-1,
                        scorer="baseline",
                        items=values,
                    )
                )

    grouped: dict[tuple[str, str, int, int], list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for row in score_rows:
        unit = unit_by_id.get(str(row.get("unit_id")))
        if unit is None:
            continue
        key = (
            str(unit.get("partition", "")),
            str(row.get("site_family")),
            csv_int(row.get("layer"), -1),
            csv_int(row.get("head"), -1),
        )
        grouped[key].append((unit, csv_float(row.get("normalised_score", row.get("score")), float("nan"))))
    for (partition, family, layer, head), items in sorted(grouped.items()):
        items = [(unit, score) for unit, score in items if math.isfinite(score)]
        if items:
            rows.append(
                metric_row(
                    partition=partition,
                    score_family=family,
                    layer=layer,
                    head=head,
                    scorer="internal",
                    items=items,
                )
            )

    best_baseline_by_partition: dict[str, float] = {}
    for row in rows:
        if row["scorer"] == "baseline" and row["partition"] not in best_baseline_by_partition:
            best_baseline_by_partition[row["partition"]] = float("-inf")
        if row["scorer"] == "baseline":
            current = finite_float(row.get("auprc"), float("-inf"))
            best_baseline_by_partition[row["partition"]] = max(
                best_baseline_by_partition[row["partition"]],
                current,
            )
    for row in rows:
        baseline = best_baseline_by_partition.get(row["partition"], float("nan"))
        row["best_baseline_auprc"] = baseline
        row["auprc_gain_vs_best_baseline"] = finite_float(row.get("auprc")) - baseline
    return rows


def select_sites(metrics: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> list[SiteSelection]:
    selection_rows = [
        row
        for row in metrics
        if row.get("partition") == "selection"
        and row.get("scorer") == "internal"
        and row.get("site_family") in INTERNAL_SITE_FAMILIES
        and math.isfinite(finite_float(row.get("auprc")))
    ]
    selection_rows.sort(
        key=lambda row: (
            finite_float(row.get("auprc_gain_vs_best_baseline"), -1.0e9),
            finite_float(row.get("auprc"), -1.0e9),
            finite_float(row.get("lift_at_k"), -1.0e9),
        ),
        reverse=True,
    )
    selected: list[SiteSelection] = []
    used_families: dict[str, int] = defaultdict(int)
    used_family_layers: set[tuple[str, int]] = set()
    for row in selection_rows:
        family = str(row["site_family"])
        layer = csv_int(row.get("layer"), -1)
        head = csv_int(row.get("head"), -1)
        if len({site.site_family for site in selected}) >= int(args.top_site_families) and family not in {
            site.site_family for site in selected
        }:
            continue
        if used_families[family] >= int(args.top_layers_per_family) and (family, layer) not in used_family_layers:
            continue
        heads_for_layer = sum(1 for site in selected if site.site_family == family and site.layer == layer)
        if head >= 0 and heads_for_layer >= int(args.top_heads_per_layer):
            continue
        selected.append(
            SiteSelection(
                site_family=family,
                patch_target=PATCH_TARGET_BY_SITE[family],
                layer=layer,
                head=head,
                selection_metric="auprc",
                selection_value=finite_float(row.get("auprc")),
                baseline_value=finite_float(row.get("best_baseline_auprc")),
                gain=finite_float(row.get("auprc_gain_vs_best_baseline")),
            )
        )
        if (family, layer) not in used_family_layers:
            used_families[family] += 1
            used_family_layers.add((family, layer))
        if len(selected) >= int(args.max_selected_sites):
            break
    if selected:
        return selected
    return [
        SiteSelection(
            site_family="pair_value_influence",
            patch_target="pair_value",
            layer=0,
            head=-1,
            selection_metric="fallback",
            selection_value=float("nan"),
            baseline_value=float("nan"),
            gain=float("nan"),
        )
    ]


def write_site_selection(path: Path, sites: Sequence[SiteSelection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sites": [dataclasses.asdict(site) for site in sites]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_site_selection(path: Path) -> list[SiteSelection]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [SiteSelection(**item) for item in payload.get("sites", [])]


def attention_scores_by_unit(score_rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for row in score_rows:
        if row.get("site_family") == "attention_mass":
            out[str(row.get("unit_id"))] += csv_float(row.get("normalised_score", row.get("score")), 0.0)
    return dict(out)


def control_distance(positive: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[int, float]:
    keys = [
        "saturated_e",
        "capacity_bucket_10",
        "source_distance_tail",
        "sink_distance_head",
        "tail_degree_bucket",
        "head_degree_bucket",
        "left_degree_bucket",
        "right_degree_bucket",
        "component_size_bucket",
        "in_one_deterministic_optimum_matching",
    ]
    exact_misses = 0
    numeric = 0.0
    for key in keys:
        if key not in positive or key not in candidate:
            continue
        a = csv_float(positive.get(key), float("nan"))
        b = csv_float(candidate.get(key), float("nan"))
        if not math.isfinite(a) or not math.isfinite(b):
            continue
        if int(a) != int(b):
            exact_misses += 1
        numeric += abs(a - b)
    return exact_misses, numeric


def select_perturbation_specs(
    unit_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    attention = attention_scores_by_unit(score_rows)
    by_graph: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in unit_rows:
        if str(row.get("partition")) == str(args.perturbation_partition):
            by_graph[str(row.get("graph_id"))].append(row)
    specs: list[dict[str, Any]] = []
    for graph_id, units in sorted(by_graph.items()):
        positives = [row for row in units if csv_int(row.get("critical")) > 0]
        positives.sort(key=lambda row: csv_float(row.get("sensitivity")), reverse=True)
        positives = positives[: max(0, int(args.positives_per_graph))]
        noncritical = [row for row in units if csv_int(row.get("critical")) <= 0]
        used_controls: set[str] = set()
        for pos_idx, positive in enumerate(positives):
            group_id = f"{positive['unit_id']}:group"
            specs.append(
                {
                    "group_id": group_id,
                    "source_type": "causal",
                    "control_rank": 0,
                    "match_level": 0,
                    "source_unit_id": positive["unit_id"],
                    "group_unit_id": positive["unit_id"],
                    "graph_id": graph_id,
                    "graph_index": positive["graph_index"],
                    "graph_position": positive["graph_position"],
                    "target_delta_group": positive["target_delta"],
                    **{f"unit_{key}": value for key, value in positive.items()},
                }
            )
            candidates = [
                row
                for row in noncritical
                if str(row.get("unit_kind")) == str(positive.get("unit_kind"))
                and str(row.get("unit_id")) not in used_controls
            ]
            candidates.sort(key=lambda row: control_distance(positive, row))
            for rank, control in enumerate(candidates[: max(0, int(args.matched_controls_per_positive))], start=1):
                used_controls.add(str(control["unit_id"]))
                match_level, _dist = control_distance(positive, control)
                specs.append(
                    {
                        "group_id": group_id,
                        "source_type": "matched_control",
                        "control_rank": rank,
                        "match_level": match_level,
                        "source_unit_id": control["unit_id"],
                        "group_unit_id": positive["unit_id"],
                        "graph_id": graph_id,
                        "graph_index": control["graph_index"],
                        "graph_position": control["graph_position"],
                        "target_delta_group": positive["target_delta"],
                        **{f"unit_{key}": value for key, value in control.items()},
                    }
                )
            high_attention = [
                row
                for row in noncritical
                if str(row.get("unit_id")) not in used_controls
            ]
            high_attention.sort(key=lambda row: attention.get(str(row.get("unit_id")), 0.0), reverse=True)
            for rank, control in enumerate(high_attention[: max(0, int(args.high_attention_controls_per_graph))], start=1):
                used_controls.add(str(control["unit_id"]))
                specs.append(
                    {
                        "group_id": group_id,
                        "source_type": "high_attention_control",
                        "control_rank": rank,
                        "match_level": -1,
                        "source_unit_id": control["unit_id"],
                        "group_unit_id": positive["unit_id"],
                        "graph_id": graph_id,
                        "graph_index": control["graph_index"],
                        "graph_position": control["graph_position"],
                        "target_delta_group": positive["target_delta"],
                        **{f"unit_{key}": value for key, value in control.items()},
                    }
                )
        if (len(specs) and len(specs) % 500 == 0):
            print(f"[perturb] selected specs={len(specs)} through graph={graph_id}", flush=True)
    return specs


def row_to_unit(spec_or_unit: Mapping[str, Any]) -> dict[str, Any]:
    if any(str(key).startswith("unit_") for key in spec_or_unit):
        out = {}
        for key, value in spec_or_unit.items():
            if str(key).startswith("unit_"):
                out[str(key)[5:]] = value
        return out
    return dict(spec_or_unit)


def graph_with_recomputed_pe(
    loaded: moa.LoadedExperiment,
    graph: Any,
    edge_index: torch.Tensor,
    edge_value: torch.Tensor,
    target: torch.Tensor,
    *,
    pe_dtype_name: str,
) -> Any:
    spd = rwse = rrwp = None
    if hasattr(loaded.runner, "compute_graph_pe_from_edge_index"):
        try:
            spd, rwse, rrwp = loaded.runner.compute_graph_pe_from_edge_index(
                int(graph.num_nodes),
                edge_index.detach().cpu().numpy(),
                torch.float16 if pe_dtype_name == "float16" else torch.float32,
            )
        except Exception:
            spd = rwse = rrwp = None
    return loaded.runner.OfficialGraph(
        node_type=graph.node_type.detach().cpu().clone(),
        edge_index=edge_index.detach().cpu().long(),
        edge_value=edge_value.detach().cpu().float(),
        target=target.detach().cpu().float(),
        task_type=graph.task_type,
        num_nodes=int(graph.num_nodes),
        spd=spd,
        rwse=rwse,
        rrwp=rrwp,
    )


def deterministic_matching_target_for_graph(graph: Any) -> torch.Tensor:
    n = int(graph.num_nodes)
    edge_index = graph.edge_index.detach().cpu().long()
    left = infer_bipartition_from_edges(n, edge_index)
    edges, _positions = canonical_matching_edges(n, edge_index, left)
    _size, optimum = matching_size(n, edges, left)
    targets = []
    for src, dst in edge_index.T.tolist():
        if bool(left[src]) == bool(left[dst]):
            targets.append(0.0)
            continue
        lnode, rnode = (src, dst) if bool(left[src]) else (dst, src)
        targets.append(1.0 if (lnode, rnode) in optimum else 0.0)
    return torch.tensor(targets, dtype=torch.float32)


def perturb_graph_for_unit(
    loaded: moa.LoadedExperiment,
    graph: Any,
    unit: Mapping[str, Any],
    args: argparse.Namespace,
) -> Any:
    family = str(unit.get("task_family"))
    edge_index = graph.edge_index.detach().cpu().long().clone()
    edge_value = graph.edge_value.detach().cpu().float().clone()
    target = graph.target.detach().cpu().float().clone()
    if family == "flow":
        edge_pos = csv_int(unit.get("edge_position"), -1)
        if edge_pos < 0 or edge_pos >= edge_value.numel():
            raise RuntimeError(f"invalid flow edge_position={edge_pos}")
        edge_value[edge_pos] = csv_float(unit.get("perturb_model_value"))
        return graph_with_recomputed_pe(
            loaded,
            graph,
            edge_index,
            edge_value,
            target,
            pe_dtype_name=args.pe_cache_dtype,
        )

    if family == "matching":
        perturbation = str(unit.get("perturbation"))
        sender = csv_int(unit.get("sender"))
        receiver = csv_int(unit.get("receiver"))
        if perturbation == "delete":
            positions_raw = str(unit.get("all_edge_positions") or unit.get("edge_position") or "")
            positions = {csv_int(item, -999) for item in positions_raw.split(",") if item.strip()}
            if not positions:
                positions = {csv_int(unit.get("edge_position"), -999)}
            keep = torch.ones(edge_index.size(1), dtype=torch.bool)
            for pos in positions:
                if 0 <= pos < keep.numel():
                    keep[pos] = False
            edge_index = edge_index[:, keep]
            edge_value = edge_value[keep]
        elif perturbation == "add":
            new_edge = torch.tensor([[sender], [receiver]], dtype=torch.long)
            edge_index = torch.cat([edge_index, new_edge], dim=1)
            edge_value = torch.cat(
                [edge_value, torch.tensor([csv_float(unit.get("perturb_model_value"), 1.0)])],
                dim=0,
            )
        else:
            raise RuntimeError(f"unknown matching perturbation {perturbation!r}")
        temp = graph_with_recomputed_pe(
            loaded,
            graph,
            edge_index,
            edge_value,
            target,
            pe_dtype_name=args.pe_cache_dtype,
        )
        if temp.task_type == "edge_binary":
            target = deterministic_matching_target_for_graph(temp)
            temp = graph_with_recomputed_pe(
                loaded,
                graph,
                edge_index,
                edge_value,
                target,
                pe_dtype_name=args.pe_cache_dtype,
            )
        return temp

    raise RuntimeError(f"no perturbation implementation for task_family={family!r}")


def single_graph_batch_from_graph(loaded: moa.LoadedExperiment, graph: Any) -> Any:
    return moa.batch_to_device(loaded.runner.collate_graphs([graph]), loaded.device)


def scalar_score(
    loaded: moa.LoadedExperiment,
    pred: torch.Tensor,
    batch: Any,
) -> float:
    return mps.score_prediction(loaded.runner, pred, batch, loaded.target_stats).scores[0]


def field_value_for_site(
    site: SiteSelection,
    collector: GRITCausalCollector,
    unit: Mapping[str, Any],
) -> Optional[torch.Tensor]:
    if site.site_family == "node_endpoint_influence":
        record = collector.node_records.get(site.layer)
        if record is None:
            return None
        positions = node_positions(record, graph_idx=0, nodes=unit_endpoint_set(unit))
        if not positions:
            return None
        return record.x.detach().cpu()[torch.tensor(positions, dtype=torch.long)].float()

    record = collector.pair_records.get(site.layer)
    if record is None:
        return None
    positions = pair_positions(record, graph_idx=0, pair_set=unit_pair_set(unit))
    if not positions:
        return None
    pos = torch.tensor(positions, dtype=torch.long)
    head = int(site.head)
    if site.site_family == "attention_mass":
        value = record.attention.detach().cpu()[pos]
        return value[:, head] if head >= 0 else value
    if site.site_family == "routing_influence":
        value = record.logits.detach().cpu()[pos]
        return value[:, head] if head >= 0 else value
    if site.site_family == "pair_value_influence":
        value = record.pair_message.detach().cpu()[pos]
        if head >= 0 and value.dim() == 3:
            return value[:, head]
        return value
    if site.site_family == "pair_state_influence":
        if record.pair_state is None:
            return None
        value = record.pair_state.detach().cpu()[pos].float()
        if head >= 0 and value.dim() == 2 and value.size(-1) % record.heads == 0:
            return value.reshape(value.size(0), record.heads, -1)[:, head]
        return value
    return None


def relative_field_change(clean: Optional[torch.Tensor], perturbed: Optional[torch.Tensor]) -> float:
    if clean is None or perturbed is None:
        return float("nan")
    if clean.shape != perturbed.shape:
        return float("nan")
    c = clean.float()
    p = perturbed.float()
    return float(torch.linalg.vector_norm(p - c) / torch.linalg.vector_norm(c).clamp_min(EPS))


def run_field_response(
    args: argparse.Namespace,
    loaded: moa.LoadedExperiment,
    unit_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    sites: Sequence[SiteSelection],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    response_path = args.output_dir / "field_response.csv"
    specs_path = args.output_dir / "perturbation_specs.csv"
    if response_path.exists() and specs_path.exists() and not args.force_recompute:
        print(f"[field] using existing field response: {response_path}", flush=True)
        return read_rows(response_path), read_rows(specs_path)

    specs = select_perturbation_specs(unit_rows, score_rows, args)
    write_table(specs_path, specs, parquet=False)
    rows: list[dict[str, Any]] = []
    clean_cache_pos: Optional[int] = None
    clean_cache_value: Optional[tuple[float, GRITCausalCollector]] = None

    for idx, spec in enumerate(specs):
        unit = row_to_unit(spec)
        pos = csv_int(spec.get("graph_position"))
        graph = loaded.graphs[pos]
        if clean_cache_pos != pos or clean_cache_value is None:
            clean_cache_value = None
            if loaded.device.type == "cuda":
                torch.cuda.empty_cache()
            clean_batch = single_graph_batch_from_graph(loaded, graph)
            clean_pred, clean_collector = predict_with_causal_capture(args, loaded, clean_batch, backward=False)
            clean_cache_pos = pos
            clean_cache_value = (scalar_score(loaded, clean_pred, clean_batch), clean_collector)
        clean_score, clean_collector = clean_cache_value

        pert_graph = perturb_graph_for_unit(loaded, graph, unit, args)
        pert_batch = single_graph_batch_from_graph(loaded, pert_graph)
        pert_pred, pert_collector = predict_with_causal_capture(args, loaded, pert_batch, backward=False)
        pert_score = scalar_score(loaded, pert_pred, pert_batch)
        for site in sites:
            clean_value = field_value_for_site(site, clean_collector, unit)
            pert_value = field_value_for_site(site, pert_collector, unit)
            change = relative_field_change(clean_value, pert_value)
            rows.append(
                {
                    "group_id": spec["group_id"],
                    "source_type": spec["source_type"],
                    "source_unit_id": spec["source_unit_id"],
                    "group_unit_id": spec["group_unit_id"],
                    "graph_index": spec["graph_index"],
                    "graph_id": spec["graph_id"],
                    "site_family": site.site_family,
                    "patch_target": site.patch_target,
                    "layer": site.layer,
                    "head": site.head,
                    "field_change": change,
                    "clean_score": clean_score,
                    "perturbed_score": pert_score,
                    "model_delta": pert_score - clean_score,
                    "target_delta": unit.get("target_delta"),
                    "target_delta_group": spec.get("target_delta_group"),
                    "match_level": spec.get("match_level"),
                }
            )
        if (idx + 1) % max(1, int(args.progress_every_graphs)) == 0 or idx + 1 == len(specs):
            print(f"[field] specs={idx + 1}/{len(specs)} rows={len(rows)}", flush=True)
        del pert_batch, pert_collector
        if args.empty_cache_every_batch and loaded.device.type == "cuda":
            torch.cuda.empty_cache()

    write_table(response_path, rows, parquet=args.write_parquet)
    return rows, specs


class NodeEndpointPatchContext:
    def __init__(
        self,
        model: nn.Module,
        *,
        layer: int,
        unit: Mapping[str, Any],
        source_record: NodeStateRecord,
    ) -> None:
        self.model = model
        self.layer = int(layer)
        self.unit = unit
        self.source_record = source_record
        self.handle: Optional[Any] = None

    def __enter__(self) -> "NodeEndpointPatchContext":
        self.handle = self.model.layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None

    def _hook(self, _module: nn.Module, _inputs: tuple[Any, ...], outputs: Any) -> Any:
        pyg_batch = outputs[0] if isinstance(outputs, tuple) else outputs
        x = getattr(pyg_batch, "x", None)
        if not torch.is_tensor(x):
            return outputs
        nodes = unit_endpoint_set(self.unit)
        source = self.source_record.x.to(device=x.device, dtype=x.dtype)
        out = x.clone()
        for node in nodes:
            if 0 <= node < out.size(0) and node < source.size(0):
                out[node] = source[node]
        pyg_batch.x = out
        return outputs


def dense_mask_for_unit(unit: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    n = csv_int(unit.get("num_nodes"))
    mask = torch.zeros(1, n, n, dtype=torch.float32, device=device)
    for receiver, sender in unit_pair_set(unit):
        if 0 <= receiver < n and 0 <= sender < n:
            mask[0, receiver, sender] = 1.0
    return mask


def capture_pair_and_node_records(
    args: argparse.Namespace,
    loaded: moa.LoadedExperiment,
    batch: Any,
) -> tuple[torch.Tensor, dict[int, mps.LayerActivation], dict[int, NodeStateRecord]]:
    pred, pair_records = mps.capture_prediction(args, loaded, batch)
    _pred2, collector = predict_with_causal_capture(args, loaded, batch, backward=False)
    return pred, pair_records, collector.node_records


def patch_prediction_for_site(
    args: argparse.Namespace,
    loaded: moa.LoadedExperiment,
    clean_batch: Any,
    unit: Mapping[str, Any],
    site: SiteSelection,
    source_pair_records: Mapping[int, mps.LayerActivation],
    source_node_records: Mapping[int, NodeStateRecord],
) -> torch.Tensor:
    if site.patch_target == "h_endpoint":
        source_record = source_node_records.get(site.layer)
        if source_record is None:
            raise RuntimeError(f"missing source node record for layer {site.layer}")
        context = NodeEndpointPatchContext(
            loaded.model,
            layer=site.layer,
            unit=unit,
            source_record=source_record,
        )
        return mps.predict_with_context(args, loaded, clean_batch, context=None if context is None else context)

    source = source_pair_records.get(site.layer)
    if source is None:
        raise RuntimeError(f"missing source pair record for layer {site.layer}")
    dense_mask = dense_mask_for_unit(unit, loaded.device)
    context = mps.GRITInterventionContext(
        loaded.model,
        official_batch=clean_batch,
        layer=site.layer,
        head=site.head,
        target=site.patch_target,
        dense_mask=dense_mask,
        mode="patch",
        source=source,
    )
    return mps.predict_with_context(args, loaded, clean_batch, context)


def target_std_for_units(unit_rows: Sequence[Mapping[str, Any]]) -> float:
    by_graph: dict[str, float] = {}
    for row in unit_rows:
        if "base_solver_value" in row:
            by_graph[str(row.get("graph_id"))] = csv_float(row.get("base_solver_value"))
    values = list(by_graph.values())
    if len(values) < 2:
        return 1.0
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    return math.sqrt(max(var, EPS))


def run_patching(
    args: argparse.Namespace,
    loaded: moa.LoadedExperiment,
    unit_rows: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    sites: Sequence[SiteSelection],
) -> list[dict[str, Any]]:
    result_path = args.output_dir / "patching_results.csv"
    if result_path.exists() and not args.force_recompute:
        print(f"[patching] using existing patching results: {result_path}", flush=True)
        return read_rows(result_path)

    by_unit = {str(row.get("unit_id")): row for row in unit_rows}
    specs_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for spec in specs:
        specs_by_group[str(spec["group_id"])].append(spec)
    target_std = target_std_for_units(unit_rows)
    rows: list[dict[str, Any]] = []
    attempted = 0

    for group_idx, (group_id, group_specs) in enumerate(sorted(specs_by_group.items())):
        causal = next((spec for spec in group_specs if spec.get("source_type") == "causal"), None)
        if causal is None:
            continue
        causal_unit = by_unit[str(causal["group_unit_id"])]
        pos = csv_int(causal.get("graph_position"))
        clean_graph = loaded.graphs[pos]
        clean_batch = single_graph_batch_from_graph(loaded, clean_graph)
        clean_pred = mps.predict_with_context(args, loaded, clean_batch)
        clean_score = scalar_score(loaded, clean_pred, clean_batch)
        causal_graph = perturb_graph_for_unit(loaded, clean_graph, causal_unit, args)
        causal_batch = single_graph_batch_from_graph(loaded, causal_graph)
        causal_pred = mps.predict_with_context(args, loaded, causal_batch)
        causal_score = scalar_score(loaded, causal_pred, causal_batch)
        denom = causal_score - clean_score
        target_delta = csv_float(causal_unit.get("target_delta"))
        pass_filter = (
            math.copysign(1.0, denom) == math.copysign(1.0, target_delta)
            and abs(denom) >= float(args.min_model_delta_std_frac) * target_std
        )

        for spec in group_specs:
            unit = row_to_unit(spec)
            source_graph = perturb_graph_for_unit(loaded, clean_graph, unit, args)
            source_batch = single_graph_batch_from_graph(loaded, source_graph)
            _source_pred, source_pair_records, source_node_records = capture_pair_and_node_records(
                args,
                loaded,
                source_batch,
            )
            for site in sites:
                attempted += 1
                if attempted > int(args.max_patching_interventions):
                    print(
                        f"[patching] reached --max-patching-interventions={args.max_patching_interventions}",
                        flush=True,
                    )
                    write_table(result_path, rows, parquet=args.write_parquet)
                    return rows
                patched_pred = patch_prediction_for_site(
                    args,
                    loaded,
                    clean_batch,
                    unit,
                    site,
                    source_pair_records,
                    source_node_records,
                )
                patched_score = scalar_score(loaded, patched_pred, clean_batch)
                mediation = (patched_score - clean_score) / (denom if abs(denom) > EPS else math.copysign(EPS, denom or 1.0))
                rows.append(
                    {
                        "group_id": group_id,
                        "source_type": spec.get("source_type"),
                        "source_unit_id": spec.get("source_unit_id"),
                        "group_unit_id": spec.get("group_unit_id"),
                        "graph_index": causal.get("graph_index"),
                        "graph_id": causal.get("graph_id"),
                        "site_family": site.site_family,
                        "patch_target": site.patch_target,
                        "layer": site.layer,
                        "head": site.head,
                        "clean_score": clean_score,
                        "causal_perturbed_score": causal_score,
                        "patched_score": patched_score,
                        "causal_model_delta": denom,
                        "target_delta": target_delta,
                        "model_response_pass": int(pass_filter),
                        "mediation": mediation,
                    }
                )
            del source_batch, source_pair_records, source_node_records
        if args.include_wrong_graph_control:
            wrong_pos = next(
                (
                    idx
                    for idx, graph in enumerate(loaded.graphs)
                    if idx != pos and int(graph.num_nodes) == int(clean_graph.num_nodes)
                ),
                None,
            )
            if wrong_pos is not None:
                wrong_batch = single_graph_batch_from_graph(loaded, loaded.graphs[wrong_pos])
                _wrong_pred, wrong_pair_records, wrong_node_records = capture_pair_and_node_records(
                    args,
                    loaded,
                    wrong_batch,
                )
                for site in sites:
                    patched_pred = patch_prediction_for_site(
                        args,
                        loaded,
                        clean_batch,
                        causal_unit,
                        site,
                        wrong_pair_records,
                        wrong_node_records,
                    )
                    patched_score = scalar_score(loaded, patched_pred, clean_batch)
                    rows.append(
                        {
                            "group_id": group_id,
                            "source_type": "wrong_graph_control",
                            "source_unit_id": f"wrong_graph_position:{wrong_pos}",
                            "group_unit_id": causal.get("group_unit_id"),
                            "graph_index": causal.get("graph_index"),
                            "graph_id": causal.get("graph_id"),
                            "site_family": site.site_family,
                            "patch_target": site.patch_target,
                            "layer": site.layer,
                            "head": site.head,
                            "clean_score": clean_score,
                            "causal_perturbed_score": causal_score,
                            "patched_score": patched_score,
                            "causal_model_delta": denom,
                            "target_delta": target_delta,
                            "model_response_pass": int(pass_filter),
                            "mediation": (
                                (patched_score - clean_score)
                                / (denom if abs(denom) > EPS else math.copysign(EPS, denom or 1.0))
                            ),
                        }
                    )
        if (group_idx + 1) % max(1, int(args.progress_every_graphs)) == 0 or group_idx + 1 == len(specs_by_group):
            passed = sum(csv_int(row.get("model_response_pass")) for row in rows)
            print(
                f"[patching] groups={group_idx + 1}/{len(specs_by_group)} "
                f"rows={len(rows)} pass_rows={passed}",
                flush=True,
            )
        if args.empty_cache_every_batch and loaded.device.type == "cuda":
            torch.cuda.empty_cache()
    write_table(result_path, rows, parquet=args.write_parquet)
    return rows


def bootstrap_ci(values: Sequence[float], seed: int, samples: int) -> tuple[float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(max(1, int(samples))):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def summarise_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    group_fields: Sequence[str],
    seed: int,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    pass_counts: dict[tuple[Any, ...], int] = defaultdict(int)
    total_counts: dict[tuple[Any, ...], int] = defaultdict(int)
    for row in rows:
        key = tuple(row.get(field, "") for field in group_fields)
        value = finite_float(row.get(metric))
        if math.isfinite(value):
            grouped[key].append(value)
        pass_counts[key] += csv_int(row.get("model_response_pass"))
        total_counts[key] += 1
    out: list[dict[str, Any]] = []
    for idx, (key, values) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
        mean = sum(values) / len(values)
        lo, hi = bootstrap_ci(values, seed + idx, bootstrap_samples)
        row = {field: value for field, value in zip(group_fields, key)}
        row.update(
            {
                "metric": metric,
                "samples": len(values),
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "positive_fraction": sum(1 for value in values if value > 0) / len(values),
                "model_response_pass_rate": pass_counts[key] / max(1, total_counts[key]),
                "support_status": "strong_support"
                if mean > 0 and lo > 0
                else ("directional_support" if mean > 0 else "not_supported"),
            }
        )
        out.append(row)
    return out


def add_control_gain(
    summary: list[dict[str, Any]],
    *,
    metric_name: str,
    control_type: str,
) -> list[dict[str, Any]]:
    controls: dict[tuple[Any, ...], float] = {}
    for row in summary:
        if row.get("source_type") != control_type:
            continue
        key = (row.get("site_family"), row.get("patch_target"), row.get("layer"), row.get("head"))
        controls[key] = finite_float(row.get("mean"))
    for row in summary:
        key = (row.get("site_family"), row.get("patch_target"), row.get("layer"), row.get("head"))
        control = controls.get(key, float("nan"))
        row[f"{control_type}_{metric_name}_mean"] = control
        row[f"{metric_name}_gain_vs_{control_type}"] = finite_float(row.get("mean")) - control
    return summary


def run_figures_and_summaries(args: argparse.Namespace) -> None:
    out_dir = args.output_dir
    field_rows = read_rows(out_dir / "field_response.csv") if (out_dir / "field_response.csv").exists() else []
    patch_rows = read_rows(out_dir / "patching_results.csv") if (out_dir / "patching_results.csv").exists() else []
    field_summary = summarise_effects(
        field_rows,
        metric="field_change",
        group_fields=("site_family", "patch_target", "layer", "head", "source_type"),
        seed=args.random_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    patch_summary = summarise_effects(
        patch_rows,
        metric="mediation",
        group_fields=("site_family", "patch_target", "layer", "head", "source_type"),
        seed=args.random_seed + 101,
        bootstrap_samples=args.bootstrap_samples,
    )
    field_summary = add_control_gain(field_summary, metric_name="field_change", control_type="matched_control")
    patch_summary = add_control_gain(patch_summary, metric_name="mediation", control_type="matched_control")
    moa.write_csv(out_dir / "field_response_summary.csv", field_summary)
    moa.write_csv(out_dir / "patching_summary.csv", patch_summary)
    write_claim_summary(out_dir, field_summary, patch_summary)
    plot_outputs(out_dir)


def write_claim_summary(
    out_dir: Path,
    field_summary: Sequence[Mapping[str, Any]],
    patch_summary: Sequence[Mapping[str, Any]],
) -> None:
    best_field = max(
        (row for row in field_summary if row.get("source_type") == "causal"),
        key=lambda row: finite_float(row.get("field_change_gain_vs_matched_control")),
        default=None,
    )
    best_patch = max(
        (row for row in patch_summary if row.get("source_type") == "causal"),
        key=lambda row: finite_float(row.get("mediation_gain_vs_matched_control")),
        default=None,
    )
    lines = [
        "# Task-Causal Specialisation Summary",
        "",
        "| Claim | Best evidence | Effect | Status |",
        "|---|---|---:|---|",
    ]
    if best_field is not None:
        lines.append(
            "| H2 field response | "
            f"{best_field.get('site_family')} L{best_field.get('layer')} H{best_field.get('head')} "
            "| "
            f"{finite_float(best_field.get('field_change_gain_vs_matched_control')):.4g} "
            f"| {best_field.get('support_status')} |"
        )
    if best_patch is not None:
        lines.append(
            "| H3 activation mediation | "
            f"{best_patch.get('site_family')} L{best_patch.get('layer')} H{best_patch.get('head')} "
            "| "
            f"{finite_float(best_patch.get('mediation_gain_vs_matched_control')):.4g} "
            f"| {best_patch.get('support_status')} |"
        )
    lines.extend(
        [
            "",
            "Strong support requires a positive paired mean and positive bootstrap lower bound. "
            "Mediation rows also report the model-response pass rate.",
        ]
    )
    (out_dir / "claim_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[plot] skipped figures because plotting imports failed: {exc}", flush=True)
        return

    alignment_path = out_dir / "alignment_metrics.csv"
    if alignment_path.exists() and alignment_path.stat().st_size > 0:
        df = pd.read_csv(alignment_path)
        sub = df[df["partition"].eq("evaluation")].copy()
        if not sub.empty:
            sub = sub.sort_values("auprc", ascending=False).head(20)
            labels = sub["site_family"].astype(str) + "\nL" + sub["layer"].astype(str) + " H" + sub["head"].astype(str)
            fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(sub)), 4.5))
            ax.bar(labels, sub["auprc"])
            ax.set_ylabel("AUPRC")
            ax.set_title("Task-causal alignment")
            ax.tick_params(axis="x", rotation=60)
            fig.tight_layout()
            fig.savefig(out_dir / "alignment_auprc.png", dpi=200)
            plt.close(fig)

    field_path = out_dir / "field_response_summary.csv"
    if field_path.exists() and field_path.stat().st_size > 0:
        df = pd.read_csv(field_path)
        sub = df[df["source_type"].eq("causal")].copy()
        if not sub.empty and "field_change_gain_vs_matched_control" in sub:
            sub = sub.sort_values("field_change_gain_vs_matched_control", ascending=False).head(20)
            labels = sub["site_family"].astype(str) + "\nL" + sub["layer"].astype(str) + " H" + sub["head"].astype(str)
            fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(sub)), 4.5))
            ax.bar(labels, sub["field_change_gain_vs_matched_control"])
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel("field change minus matched control")
            ax.set_title("Task-causal field response")
            ax.tick_params(axis="x", rotation=60)
            fig.tight_layout()
            fig.savefig(out_dir / "field_response_gain.png", dpi=200)
            plt.close(fig)

    patch_path = out_dir / "patching_summary.csv"
    if patch_path.exists() and patch_path.stat().st_size > 0:
        df = pd.read_csv(patch_path)
        sub = df[df["source_type"].eq("causal")].copy()
        if not sub.empty and "mediation_gain_vs_matched_control" in sub:
            sub = sub.sort_values("mediation_gain_vs_matched_control", ascending=False).head(20)
            labels = sub["site_family"].astype(str) + "\nL" + sub["layer"].astype(str) + " H" + sub["head"].astype(str)
            fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(sub)), 4.5))
            ax.bar(labels, sub["mediation_gain_vs_matched_control"])
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel("mediation minus matched control")
            ax.set_title("Activation-mediated restoration")
            ax.tick_params(axis="x", rotation=60)
            fig.tight_layout()
            fig.savefig(out_dir / "patching_mediation_gain.png", dpi=200)
            plt.close(fig)


def metadata(args: argparse.Namespace, loaded: Optional[moa.LoadedExperiment], start: float) -> dict[str, Any]:
    return {
        "task": args.task,
        "split": args.split,
        "model": args.model,
        "model_backend": args.model_backend,
        "checkpoint": str(args.checkpoint),
        "num_graphs": None if loaded is None else len(loaded.graphs),
        "selected_graph_indices": None if loaded is None else loaded.selected_indices,
        "device": None if loaded is None else str(loaded.device),
        "cuda_name": torch.cuda.get_device_name(loaded.device) if loaded is not None and loaded.device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "torch_runtime": None if loaded is None else dict(loaded.runtime_settings),
        "analysis_batch_cache_mode": None if loaded is None else loaded.batch_cache_mode,
        "analysis_batch_cache_estimated_gib": None
        if loaded is None
        else loaded.batch_cache_estimated_bytes / (1024**3),
        "args": jsonable(vars(args)),
        "git_sha": git_sha(Path.cwd()),
        "elapsed_seconds": time.time() - start,
        "protocol": "task_causal_specialisation_protocol_A100",
    }


def apply_analysis_preset(args: argparse.Namespace, argv: Sequence[str]) -> None:
    preset = getattr(args, "analysis_preset", "custom")
    if preset == "custom":
        return
    if preset == "lightweight":
        defaults = {
            "stages": "all",
            "num_graphs": 64,
            "batch_size": 16,
            "analysis_cache_batch_size": 16,
            "bootstrap_samples": 200,
            "positives_per_graph": 2,
            "matched_controls_per_positive": 2,
            "high_attention_controls_per_graph": 2,
            "top_site_families": 2,
            "top_layers_per_family": 2,
            "top_heads_per_layer": 2,
            "max_selected_sites": 12,
            "max_patching_interventions": 2500,
        }
    elif preset == "full":
        defaults = {
            "stages": "all",
            "num_graphs": 512,
            "batch_size": 16,
            "analysis_cache_batch_size": 16,
            "bootstrap_samples": 1000,
            "positives_per_graph": 4,
            "matched_controls_per_positive": 4,
            "high_attention_controls_per_graph": 4,
            "top_site_families": 3,
            "top_layers_per_family": 3,
            "top_heads_per_layer": 4,
            "max_selected_sites": 36,
            "max_patching_interventions": 100_000,
        }
    else:
        raise ValueError(f"unknown analysis preset {preset!r}")
    flag_by_dest = {
        "stages": ("--stages",),
        "num_graphs": ("--num-graphs",),
        "batch_size": ("--batch-size",),
        "analysis_cache_batch_size": ("--analysis-cache-batch-size",),
        "bootstrap_samples": ("--bootstrap-samples",),
        "positives_per_graph": ("--positives-per-graph",),
        "matched_controls_per_positive": ("--matched-controls-per-positive",),
        "high_attention_controls_per_graph": ("--high-attention-controls-per-graph",),
        "top_site_families": ("--top-site-families",),
        "top_layers_per_family": ("--top-layers-per-family",),
        "top_heads_per_layer": ("--top-heads-per-layer",),
        "max_selected_sites": ("--max-selected-sites",),
        "max_patching_interventions": ("--max-patching-interventions",),
    }
    for dest, value in defaults.items():
        if not mps.cli_supplied(argv, *flag_by_dest[dest]):
            setattr(args, dest, value)


def build_parser() -> argparse.ArgumentParser:
    parser = moa.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        experiments="atlas",
        output_dir=Path("outputs/task_causal_specialisation_protocol"),
        num_graphs=64,
        batch_size=16,
        analysis_cache_batch_size=16,
        autocast_dtype="bfloat16",
        cache_batches="cpu",
    )
    parser.add_argument("--analysis-preset", default="lightweight", choices=ANALYSIS_PRESETS)
    parser.add_argument("--stages", default="all", help="Comma list: labels,alignment,field,patching,figures,all")
    parser.add_argument("--sensitivity-units-path", type=Path, default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--selection-fraction", type=float, default=0.5)
    parser.add_argument("--critical-threshold", type=float, default=0.0)
    parser.add_argument("--flow-eta", type=float, default=1.0)
    parser.add_argument("--solver-edge-value-source", choices=("loaded", "unnormalize"), default="loaded")
    parser.add_argument("--solver-edge-value-mean", type=float, default=None)
    parser.add_argument("--solver-edge-value-std", type=float, default=None)
    parser.add_argument("--solver-capacity-clamp-min", type=float, default=0.0)
    parser.add_argument("--matching-nonedges-per-edge", type=int, default=5)
    parser.add_argument("--matching-added-edge-value", type=float, default=1.0)
    parser.add_argument("--perturbation-partition", choices=("selection", "evaluation"), default="evaluation")
    parser.add_argument("--positives-per-graph", type=int, default=2)
    parser.add_argument("--matched-controls-per-positive", type=int, default=2)
    parser.add_argument("--high-attention-controls-per-graph", type=int, default=2)
    parser.add_argument("--top-site-families", type=int, default=2)
    parser.add_argument("--top-layers-per-family", type=int, default=2)
    parser.add_argument("--top-heads-per-layer", type=int, default=2)
    parser.add_argument("--max-selected-sites", type=int, default=12)
    parser.add_argument("--max-patching-interventions", type=int, default=2500)
    parser.add_argument("--min-model-delta-std-frac", type=float, default=0.05)
    parser.add_argument("--include-wrong-graph-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--progress-every-graphs", type=int, default=10)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.model_backend != "official":
        raise RuntimeError("task-causal protocol currently supports --model-backend official only")
    if args.model not in {"grit", "static_grit"}:
        raise RuntimeError("task-causal protocol currently supports --model grit/static_grit")
    if not (0.0 < float(args.selection_fraction) < 1.0):
        raise RuntimeError("--selection-fraction must lie strictly between 0 and 1")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    start = time.time()
    stages = parse_csv(args.stages, all_values=STAGES)
    if args.stages == "all":
        stages = STAGES
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded: Optional[moa.LoadedExperiment] = None
    if any(stage in stages for stage in ("labels", "alignment", "field", "patching")):
        loaded = moa.load_experiment(args)

    unit_rows: list[dict[str, Any]]
    if "labels" in stages:
        assert loaded is not None
        unit_rows = run_labels(args, loaded)
    else:
        unit_path = args.sensitivity_units_path or (args.output_dir / "sensitivity_units.csv")
        unit_rows = read_rows(unit_path)

    score_rows: list[dict[str, Any]] = []
    sites: list[SiteSelection] = []
    if "alignment" in stages:
        assert loaded is not None
        score_rows, _metrics, sites = run_alignment(args, loaded, unit_rows)
    else:
        score_path = args.output_dir / "clean_internal_scores.csv"
        if score_path.exists():
            score_rows = read_rows(score_path)
        site_path = args.output_dir / "site_selection.json"
        if site_path.exists():
            sites = load_site_selection(site_path)

    specs: list[dict[str, Any]] = []
    if "field" in stages:
        assert loaded is not None
        if not score_rows:
            score_rows = read_rows(args.output_dir / "clean_internal_scores.csv")
        if not sites:
            sites = load_site_selection(args.output_dir / "site_selection.json")
        _field_rows, specs = run_field_response(args, loaded, unit_rows, score_rows, sites)
    else:
        specs_path = args.output_dir / "perturbation_specs.csv"
        if specs_path.exists():
            specs = read_rows(specs_path)

    if "patching" in stages:
        assert loaded is not None
        if not specs:
            specs = read_rows(args.output_dir / "perturbation_specs.csv")
        if not sites:
            sites = load_site_selection(args.output_dir / "site_selection.json")
        run_patching(args, loaded, unit_rows, specs, sites)

    if "figures" in stages:
        run_figures_and_summaries(args)

    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata(args, loaded, start), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[done] task-causal protocol outputs written to {args.output_dir}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    apply_analysis_preset(args, raw_argv)
    run(args)


if __name__ == "__main__":
    main()
