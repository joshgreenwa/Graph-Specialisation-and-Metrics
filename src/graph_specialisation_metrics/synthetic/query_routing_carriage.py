"""Finite carriage versus a clean Jacobian in learned categorical graph routing.

The task is a graph-native key-value lookup. One query node carries a categorical key and one
record node exists for every key. The model emits a local query contribution and routes the value
of the matching record through an ordinary softmax. Node contributions are summed by a strictly
linear readout.

A real semantic donor changes the query key. The data-generating process supplies the exact
eventwise carrier field, so a direction-matched clean Jacobian and finite Functional carriage can
both be scored against an independent hard-routing oracle.

The runner is deliberately cache-first. Generated splits, checkpoints, donor-resolved carrier
measurements, derived tables, and every PNG/PDF figure are saved below a fingerprinted output
directory. ``--phase figures`` performs no model inference.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import torch
from torch import nn

from ..carriage.content import ContentAdapter
from ..methodology.bootstrap import Observation, nested_percentile_interval, trimmed_mean
from ..methodology.carriage import (
    beneficial_carriage,
    event_normalise_functional,
    functional_carriage_events,
)
from ..methodology.events import build_channel_events
from ..methodology.protocol import BootstrapPolicy
from ..methodology.sampling import DonorEvent, SemanticDonorPool, manifest_fingerprint


PROTOCOL_VERSION = "query-routing-carriage-v1"
DTYPE = torch.float32
ROLE_CONTEXT = 0
ROLE_QUERY = 1
ROLE_RECORD = 2
ROLE_COUNT = 3
DEFAULT_SEEDS = (0, 1, 2, 3)
DEFAULT_BETA_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_DOSE_ALPHAS = (1.0e-3, 1.0e-2, 0.05, 0.10, 0.25, 0.50, 1.0)
DATA_SHARD_GRAPHS = 64


# --------------------------------------------------------------------------------------
# Configuration and immutable artifact contract
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    beta_multipliers: tuple[float, ...] = DEFAULT_BETA_MULTIPLIERS
    dose_alphas: tuple[float, ...] = DEFAULT_DOSE_ALPHAS
    num_keys: int = 8
    rwse_steps: int = 8
    hidden_dim: int = 96
    layers: int = 3
    heads: int = 4
    dropout: float = 0.10
    attention_dropout: float = 0.10
    train_graphs: int = 12_288
    validation_graphs: int = 1_024
    id_graphs: int = 256
    ood_graphs: int = 256
    donor_graphs: int = 512
    id_min_nodes: int = 32
    id_max_nodes: int = 64
    ood_min_nodes: int = 80
    ood_max_nodes: int = 128
    id_required_distance: int = 6
    ood_required_distance: int = 12
    donors_per_source: int = 16
    batch_size: int = 32
    max_epochs: int = 100
    early_stop_patience: int = 12
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0
    graph_loss_weight: float = 0.25
    data_seed: int = 20_260_730
    event_seed: int = 17_071
    bootstrap_seed: int = 17_071
    effect_floor: float = 1.0e-8
    integrated_atol: float = 1.0e-6
    integrated_rtol: float = 1.0e-5
    integrated_max_intervals: int = 64
    integrated_tolerance: float = 1.0e-5
    compute_beneficial: bool = True
    compute_dose_ladder: bool = True
    production_bootstrap: bool = True
    smoke: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if not self.beta_multipliers or 1.0 not in self.beta_multipliers:
            raise ValueError("beta_multipliers must include the native value 1.0")
        if any(value <= 0 for value in self.beta_multipliers):
            raise ValueError("beta multipliers must be positive")
        if (
            not self.dose_alphas
            or any(value <= 0 or value > 1 for value in self.dose_alphas)
            or tuple(sorted(self.dose_alphas)) != self.dose_alphas
            or 1.0 not in self.dose_alphas
        ):
            raise ValueError(
                "dose_alphas must be increasing, lie in (0,1], and include 1"
            )
        if self.num_keys < 3:
            raise ValueError("num_keys must be at least three")
        if self.hidden_dim < 4 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if self.layers < 1 or self.heads < 1:
            raise ValueError("layers and heads must be positive")
        for name in (
            "train_graphs",
            "validation_graphs",
            "id_graphs",
            "ood_graphs",
            "donor_graphs",
            "donors_per_source",
            "batch_size",
            "max_epochs",
            "early_stop_patience",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if self.id_min_nodes < self.num_keys + 2 or self.ood_min_nodes < self.num_keys + 2:
            raise ValueError("graph minima must leave room for query, records, and context")
        if self.id_max_nodes < self.id_min_nodes or self.ood_max_nodes < self.ood_min_nodes:
            raise ValueError("graph node ranges are invalid")
        if self.id_required_distance < 4 or self.ood_required_distance < 4:
            raise ValueError("required distances must be at least four")
        if self.effect_floor <= 0:
            raise ValueError("effect_floor must be positive")

    @property
    def record(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            **asdict(self),
            "seeds": list(self.seeds),
            "beta_multipliers": list(self.beta_multipliers),
            "dose_alphas": list(self.dose_alphas),
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.record, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def fast_dev(cls, *, seeds: tuple[int, ...] = (0,)) -> "ExperimentConfig":
        return cls(
            seeds=seeds,
            beta_multipliers=(1.0, 4.0),
            dose_alphas=(0.01, 0.10, 1.0),
            num_keys=4,
            rwse_steps=4,
            hidden_dim=32,
            layers=1,
            heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            train_graphs=48,
            validation_graphs=12,
            id_graphs=8,
            ood_graphs=6,
            donor_graphs=12,
            id_min_nodes=12,
            id_max_nodes=18,
            ood_min_nodes=18,
            ood_max_nodes=24,
            id_required_distance=4,
            ood_required_distance=5,
            donors_per_source=2,
            batch_size=8,
            max_epochs=3,
            early_stop_patience=2,
            learning_rate=2.0e-3,
            compute_beneficial=True,
            compute_dose_ladder=True,
            production_bootstrap=False,
            smoke=True,
        )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def ensure_contract(output_dir: Path, config: ExperimentConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment.json"
    expected = {
        "fingerprint": config.fingerprint,
        "config": config.record,
        "smoke": bool(config.smoke),
    }
    if path.exists():
        observed = json.loads(path.read_text())
        if observed != expected:
            raise RuntimeError(
                f"{path} belongs to a different experiment contract. "
                "Choose a new --output-dir rather than mixing caches."
            )
    else:
        _atomic_json(path, expected)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Graph container, generation, and batching
# --------------------------------------------------------------------------------------


@dataclass
class QueryGraph:
    graph_id: int
    x: torch.Tensor
    role: torch.Tensor
    value: torch.Tensor
    pe: torch.Tensor
    edge_index: torch.Tensor
    spd: torch.Tensor
    query_node: int
    record_nodes: torch.Tensor
    y_node: torch.Tensor
    y: torch.Tensor
    family: str

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def keys(self) -> tuple[str, ...]:
        return (
            "x",
            "role",
            "value",
            "pe",
            "edge_index",
            "spd",
            "record_nodes",
            "y_node",
            "y",
        )

    def clone(self) -> "QueryGraph":
        return QueryGraph(
            graph_id=int(self.graph_id),
            x=self.x.clone(),
            role=self.role.clone(),
            value=self.value.clone(),
            pe=self.pe.clone(),
            edge_index=self.edge_index.clone(),
            spd=self.spd.clone(),
            query_node=int(self.query_node),
            record_nodes=self.record_nodes.clone(),
            y_node=self.y_node.clone(),
            y=self.y.clone(),
            family=str(self.family),
        )


def _graph_record(graph: QueryGraph) -> dict[str, Any]:
    return {
        "graph_id": int(graph.graph_id),
        "x": graph.x,
        "role": graph.role,
        "value": graph.value,
        "pe": graph.pe,
        "edge_index": graph.edge_index,
        "spd": graph.spd,
        "query_node": int(graph.query_node),
        "record_nodes": graph.record_nodes,
        "y_node": graph.y_node,
        "y": graph.y,
        "family": str(graph.family),
    }


def _record_graph(record: Mapping[str, Any]) -> QueryGraph:
    return QueryGraph(
        graph_id=int(record["graph_id"]),
        x=record["x"],
        role=record["role"],
        value=record["value"],
        pe=record["pe"],
        edge_index=record["edge_index"],
        spd=record["spd"],
        query_node=int(record["query_node"]),
        record_nodes=record["record_nodes"],
        y_node=record["y_node"],
        y=record["y"],
        family=str(record["family"]),
    )


def local_codes(num_keys: int, *, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    return torch.linspace(-0.75, 0.75, int(num_keys), dtype=dtype)


def teacher_contributions(
    graph: QueryGraph,
    query_onehot: torch.Tensor | None = None,
) -> torch.Tensor:
    query = graph.x[graph.query_node] if query_onehot is None else query_onehot
    query_key = int(torch.argmax(query).item())
    result = torch.zeros(graph.num_nodes, dtype=graph.value.dtype, device=graph.value.device)
    result[graph.query_node] = local_codes(
        graph.x.shape[1], dtype=graph.value.dtype
    ).to(graph.value.device)[query_key]
    record_keys = torch.argmax(graph.x[graph.record_nodes], dim=-1)
    matched = graph.record_nodes[record_keys == query_key]
    if int(matched.numel()) != 1:
        raise RuntimeError(f"query key {query_key} resolves to {int(matched.numel())} records")
    result[int(matched.item())] = graph.value[int(matched.item())]
    return result


def _connect_components(graph: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    graph = nx.Graph(graph)
    components = [list(component) for component in nx.connected_components(graph)]
    for left, right in zip(components, components[1:]):
        graph.add_edge(
            int(left[int(rng.integers(0, len(left)))]),
            int(right[int(rng.integers(0, len(right)))]),
        )
    return graph


def _random_backbone(
    n: int,
    family: str,
    rng: np.random.Generator,
) -> nx.Graph:
    seed = int(rng.integers(0, 2**31 - 1))
    if family == "small_world":
        graph = nx.watts_strogatz_graph(n, 4, 0.12, seed=seed)
    elif family == "preferential":
        graph = nx.barabasi_albert_graph(n, 1, seed=seed)
    elif family == "sbm":
        first = n // 2
        graph = nx.stochastic_block_model(
            [first, n - first],
            [[min(0.18, 6.0 / n), 0.015], [0.015, min(0.18, 6.0 / n)]],
            seed=seed,
        )
    elif family == "sparse_er":
        graph = nx.random_labeled_tree(n, seed=seed)
        extra = max(1, n // 4)
        for _ in range(extra):
            left, right = rng.choice(n, size=2, replace=False)
            graph.add_edge(int(left), int(right))
    else:
        raise ValueError(f"unknown graph family {family!r}")
    return _connect_components(graph, rng)


def _rwse(graph: nx.Graph, steps: int) -> np.ndarray:
    adjacency = nx.to_numpy_array(graph, dtype=np.float64)
    degree = adjacency.sum(axis=1, keepdims=True)
    transition = np.divide(
        adjacency,
        np.maximum(degree, 1.0),
        out=np.zeros_like(adjacency),
        where=degree > 0,
    )
    power = np.eye(adjacency.shape[0], dtype=np.float64)
    values = []
    for _ in range(int(steps)):
        power = power @ transition
        values.append(np.diag(power))
    return np.stack(values, axis=1).astype(np.float32)


def _edge_index(graph: nx.Graph) -> torch.Tensor:
    pairs: list[tuple[int, int]] = []
    for left, right in graph.edges:
        pairs.extend(((int(left), int(right)), (int(right), int(left))))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _all_pairs_spd(graph: nx.Graph) -> np.ndarray:
    n = graph.number_of_nodes()
    result = np.full((n, n), n + 1, dtype=np.int64)
    for source, distances in nx.all_pairs_shortest_path_length(graph):
        for target, distance in distances.items():
            result[int(source), int(target)] = int(distance)
    return result


def _choose_query_and_records(
    graph: nx.Graph,
    *,
    num_keys: int,
    required_distance: int,
    rng: np.random.Generator,
) -> tuple[int, np.ndarray, np.ndarray]:
    initial = int(rng.integers(0, graph.number_of_nodes()))
    first_dist = nx.single_source_shortest_path_length(graph, initial)
    endpoint = max(first_dist, key=first_dist.get)
    distances = nx.single_source_shortest_path_length(graph, endpoint)
    if max(distances.values()) < int(required_distance):
        raise ValueError("graph diameter is too small")
    query = int(endpoint)
    chosen: list[int] = []
    if int(required_distance) <= int(num_keys):
        targets = list(range(1, int(required_distance) + 1))
    else:
        near = list(range(1, min(4, int(num_keys)) + 1))
        far_slots = int(num_keys) - len(near)
        far = (
            np.rint(
                np.linspace(
                    near[-1] + 1,
                    int(required_distance),
                    num=far_slots,
                )
            )
            .astype(np.int64)
            .tolist()
            if far_slots > 0
            else []
        )
        targets = list(dict.fromkeys(near + far))
        if targets[-1] != int(required_distance):
            targets[-1] = int(required_distance)
    for target in targets:
        candidates = [
            node
            for node, distance in distances.items()
            if int(distance) == int(target) and int(node) not in chosen and int(node) != query
        ]
        if not candidates:
            raise ValueError(f"no record candidate at distance {target}")
        chosen.append(int(candidates[int(rng.integers(0, len(candidates)))]))
    remaining = [
        node for node in graph.nodes if int(node) != query and int(node) not in chosen
    ]
    rng.shuffle(remaining)
    chosen.extend(int(node) for node in remaining[: int(num_keys) - len(chosen)])
    if len(chosen) != int(num_keys):
        raise ValueError("insufficient distinct record nodes")
    return query, np.asarray(chosen, dtype=np.int64), np.asarray(
        [int(distances[node]) for node in chosen], dtype=np.int64
    )


def generate_graph(
    config: ExperimentConfig,
    *,
    graph_id: int,
    rng: np.random.Generator,
    ood: bool = False,
) -> QueryGraph:
    min_nodes = config.ood_min_nodes if ood else config.id_min_nodes
    max_nodes = config.ood_max_nodes if ood else config.id_max_nodes
    required = config.ood_required_distance if ood else config.id_required_distance
    families = ("sparse_er", "small_world", "preferential", "sbm")
    last_error: Exception | None = None
    for _ in range(256):
        n = int(rng.integers(min_nodes, max_nodes + 1))
        family = families[int(rng.integers(0, len(families)))]
        graph = _random_backbone(n, family, rng)
        try:
            query, records, _ = _choose_query_and_records(
                graph,
                num_keys=config.num_keys,
                required_distance=required,
                rng=rng,
            )
        except ValueError as exc:
            last_error = exc
            continue
        keys = rng.integers(0, config.num_keys, size=n, dtype=np.int64)
        record_keys = rng.permutation(config.num_keys).astype(np.int64)
        keys[records] = record_keys
        keys[query] = int(rng.integers(0, config.num_keys))
        x = torch.nn.functional.one_hot(
            torch.from_numpy(keys), num_classes=config.num_keys
        ).to(DTYPE)
        roles = torch.full((n,), ROLE_CONTEXT, dtype=torch.long)
        roles[query] = ROLE_QUERY
        roles[torch.from_numpy(records)] = ROLE_RECORD
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=config.num_keys)
        magnitudes = rng.uniform(0.5, 1.5, size=config.num_keys)
        values = torch.zeros(n, dtype=DTYPE)
        values[torch.from_numpy(records)] = torch.tensor(
            signs * magnitudes, dtype=DTYPE
        )
        spd = torch.from_numpy(_all_pairs_spd(graph))
        result = QueryGraph(
            graph_id=int(graph_id),
            x=x,
            role=roles,
            value=values,
            pe=torch.from_numpy(_rwse(graph, config.rwse_steps)),
            edge_index=_edge_index(graph),
            spd=spd,
            query_node=int(query),
            record_nodes=torch.from_numpy(records),
            y_node=torch.zeros(n, dtype=DTYPE),
            y=torch.zeros(1, dtype=DTYPE),
            family=family,
        )
        result.y_node = teacher_contributions(result)
        result.y = result.y_node.sum().reshape(1)
        return result
    raise RuntimeError(f"failed to generate graph after 256 attempts: {last_error}")


def _split_path(output_dir: Path, split: str) -> Path:
    return output_dir / "data" / f"{split}.pt"


def _data_shard_path(
    output_dir: Path,
    split: str,
    start: int,
    stop: int,
) -> Path:
    return (
        output_dir
        / "data"
        / "shards"
        / split
        / f"graphs_{int(start):06d}_{int(stop):06d}.pt"
    )


def _split_spec(config: ExperimentConfig) -> tuple[tuple[str, int, bool, int], ...]:
    return (
        ("train", config.train_graphs, False, 0),
        ("validation", config.validation_graphs, False, 1_000_000),
        ("id", config.id_graphs, False, 2_000_000),
        ("ood", config.ood_graphs, True, 3_000_000),
        ("donor", config.donor_graphs, False, 4_000_000),
    )


def ensure_data(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    progress: bool = True,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    summary: dict[str, Any] = {"fingerprint": config.fingerprint, "splits": {}}
    for split, count, ood, offset in _split_spec(config):
        path = _split_path(output_dir, split)
        paths[split] = str(path)
        if path.exists():
            payload = _load_torch(path)
            if payload.get("fingerprint") != config.fingerprint:
                raise RuntimeError(f"data contract mismatch: {path}")
            graphs = [_record_graph(record) for record in payload["graphs"]]
        else:
            records: list[dict[str, Any]] = []
            for start in range(0, int(count), DATA_SHARD_GRAPHS):
                stop = min(start + DATA_SHARD_GRAPHS, int(count))
                shard_path = _data_shard_path(
                    output_dir,
                    split,
                    start,
                    stop,
                )
                if shard_path.exists():
                    shard = _load_torch(shard_path)
                    if (
                        shard.get("fingerprint") != config.fingerprint
                        or shard.get("split") != split
                        or int(shard.get("start", -1)) != start
                        or int(shard.get("stop", -1)) != stop
                    ):
                        raise RuntimeError(f"data-shard contract mismatch: {shard_path}")
                    shard_records = list(shard["graphs"])
                else:
                    shard_records = []
                    for index in range(start, stop):
                        graph = generate_graph(
                            config,
                            graph_id=int(offset + index),
                            rng=np.random.default_rng(
                                [config.data_seed, offset, index]
                            ),
                            ood=ood,
                        )
                        shard_records.append(_graph_record(graph))
                    _atomic_torch(
                        shard_path,
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "fingerprint": config.fingerprint,
                            "split": split,
                            "start": int(start),
                            "stop": int(stop),
                            "graphs": shard_records,
                        },
                    )
                if len(shard_records) != stop - start:
                    raise RuntimeError(
                        f"data shard has the wrong graph count: {shard_path}"
                    )
                records.extend(shard_records)
                if progress:
                    print(
                        f"[data] split={split} graphs={stop}/{count}",
                        flush=True,
                    )
            graphs = [_record_graph(record) for record in records]
            _atomic_torch(
                path,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "fingerprint": config.fingerprint,
                    "split": split,
                    "graphs": records,
                },
            )
        distances = np.concatenate(
            [
                graph.spd[graph.query_node, graph.record_nodes].numpy()
                for graph in graphs
            ]
        )
        summary["splits"][split] = {
            "graphs": len(graphs),
            "minimum_nodes": min(graph.num_nodes for graph in graphs),
            "maximum_nodes": max(graph.num_nodes for graph in graphs),
            "minimum_record_distance": int(distances.min()),
            "maximum_record_distance": int(distances.max()),
            "families": {
                family: sum(graph.family == family for graph in graphs)
                for family in sorted({graph.family for graph in graphs})
            },
            "path": str(path),
        }
        if progress:
            print(
                f"[data] split={split} graphs={len(graphs)} "
                f"nodes={summary['splits'][split]['minimum_nodes']}-"
                f"{summary['splits'][split]['maximum_nodes']} "
                f"record_d={distances.min()}-{distances.max()}",
                flush=True,
            )
    _atomic_json(output_dir / "data" / "generation_summary.json", summary)
    _atomic_json(
        output_dir / "data" / "split_manifest.json",
        {
            "fingerprint": config.fingerprint,
            "splits": {
                split: {
                    "path": str(_split_path(output_dir, split)),
                    "graph_id_start": int(offset),
                    "graph_id_stop": int(offset + count),
                }
                for split, count, _, offset in _split_spec(config)
            },
        },
    )
    return paths


def load_split(output_dir: Path, config: ExperimentConfig, split: str) -> list[QueryGraph]:
    path = _split_path(output_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"missing generated split {path}; run --phase data first")
    payload = _load_torch(path)
    if payload.get("fingerprint") != config.fingerprint:
        raise RuntimeError(f"data contract mismatch: {path}")
    return [_record_graph(record) for record in payload["graphs"]]


@dataclass
class QueryBatch:
    x: torch.Tensor
    role: torch.Tensor
    value: torch.Tensor
    pe: torch.Tensor
    adj_norm: torch.Tensor
    spd: torch.Tensor
    mask: torch.Tensor
    record_mask: torch.Tensor
    query_node: torch.Tensor
    y_node: torch.Tensor
    y: torch.Tensor
    graph_id: torch.Tensor
    num_nodes: torch.Tensor

    def to(self, device: torch.device) -> "QueryBatch":
        return QueryBatch(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in dataclasses.fields(self)
            }
        )


def collate_graphs(graphs: Sequence[QueryGraph]) -> QueryBatch:
    batch_size = len(graphs)
    max_nodes = max(graph.num_nodes for graph in graphs)
    keys = int(graphs[0].x.shape[1])
    pe_dim = int(graphs[0].pe.shape[1])
    x = torch.zeros(batch_size, max_nodes, keys, dtype=DTYPE)
    role = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    value = torch.zeros(batch_size, max_nodes, dtype=DTYPE)
    pe = torch.zeros(batch_size, max_nodes, pe_dim, dtype=DTYPE)
    adj_norm = torch.zeros(batch_size, max_nodes, max_nodes, dtype=DTYPE)
    spd = torch.full((batch_size, max_nodes, max_nodes), max_nodes + 1, dtype=torch.long)
    mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    record_mask = torch.zeros_like(mask)
    query_node = torch.zeros(batch_size, dtype=torch.long)
    y_node = torch.zeros(batch_size, max_nodes, dtype=DTYPE)
    y = torch.zeros(batch_size, dtype=DTYPE)
    graph_id = torch.zeros(batch_size, dtype=torch.long)
    num_nodes = torch.zeros(batch_size, dtype=torch.long)
    for index, graph in enumerate(graphs):
        n = graph.num_nodes
        x[index, :n] = graph.x
        role[index, :n] = graph.role
        value[index, :n] = graph.value
        pe[index, :n] = graph.pe
        adjacency = torch.zeros(n, n, dtype=DTYPE)
        if graph.edge_index.numel():
            adjacency[graph.edge_index[0], graph.edge_index[1]] = 1.0
        adjacency += torch.eye(n, dtype=DTYPE)
        degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
        adj_norm[index, :n, :n] = adjacency / degree
        spd[index, :n, :n] = graph.spd
        mask[index, :n] = True
        record_mask[index, graph.record_nodes] = True
        query_node[index] = int(graph.query_node)
        y_node[index, :n] = graph.y_node
        y[index] = graph.y.reshape(-1)[0]
        graph_id[index] = int(graph.graph_id)
        num_nodes[index] = n
    return QueryBatch(
        x=x,
        role=role,
        value=value,
        pe=pe,
        adj_norm=adj_norm,
        spd=spd,
        mask=mask,
        record_mask=record_mask,
        query_node=query_node,
        y_node=y_node,
        y=y,
        graph_id=graph_id,
        num_nodes=num_nodes,
    )


# --------------------------------------------------------------------------------------
# GraphGPS-style learned router
# --------------------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class DenseGraphGPSLayer(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, attention_dropout: float) -> None:
        super().__init__()
        self.local_projection = nn.Linear(dim, dim)
        self.attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, batch: QueryBatch) -> torch.Tensor:
        local = torch.bmm(batch.adj_norm, torch.relu(self.local_projection(hidden)))
        global_value, _ = self.attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=~batch.mask,
            need_weights=False,
        )
        hidden = self.norm1(hidden + self.dropout(local + global_value))
        hidden = self.norm2(hidden + self.dropout(self.ffn(hidden)))
        return hidden * batch.mask.unsqueeze(-1).to(hidden.dtype)


@dataclass
class RouterOutput:
    node_contributions: torch.Tensor
    graph_prediction: torch.Tensor
    final_hidden: torch.Tensor
    router_logits: torch.Tensor
    router_probability: torch.Tensor


class GraphGPSQueryRouter(nn.Module):
    def __init__(self, config: ExperimentConfig, *, init_seed: int) -> None:
        super().__init__()
        torch.manual_seed(int(init_seed))
        self.config = config
        dim = int(config.hidden_dim)
        self.role_embedding = nn.Embedding(ROLE_COUNT, dim)
        self.key_projection = nn.Linear(config.num_keys, dim, bias=False)
        self.value_projection = nn.Linear(1, dim, bias=False)
        self.pe_projection = nn.Linear(config.rwse_steps, dim, bias=False)
        self.layers = nn.ModuleList(
            [
                DenseGraphGPSLayer(
                    dim,
                    config.heads,
                    config.dropout,
                    config.attention_dropout,
                )
                for _ in range(config.layers)
            ]
        )
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.record_projection = nn.Linear(dim, dim, bias=False)
        self.local_head = nn.Linear(config.num_keys, 1, bias=False)

    def forward(
        self,
        batch: QueryBatch,
        *,
        beta_multiplier: float = 1.0,
        query_override: torch.Tensor | None = None,
    ) -> RouterOutput:
        batch_size, nodes, _ = batch.x.shape
        batch_index = torch.arange(batch_size, device=batch.x.device)
        if query_override is None:
            effective_x = batch.x
        else:
            if tuple(query_override.shape) != (batch_size, self.config.num_keys):
                raise ValueError("query_override must be [batch,num_keys]")
            source_mask = torch.nn.functional.one_hot(
                batch.query_node, num_classes=nodes
            ).to(batch.x.dtype)
            effective_x = (
                batch.x * (1.0 - source_mask.unsqueeze(-1))
                + query_override.unsqueeze(1) * source_mask.unsqueeze(-1)
            )
        hidden = (
            self.role_embedding(batch.role)
            + self.key_projection(effective_x)
            + self.value_projection(batch.value.unsqueeze(-1))
            + self.pe_projection(batch.pe)
        )
        hidden = hidden * batch.mask.unsqueeze(-1).to(hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, batch)
        query_hidden = hidden[batch_index, batch.query_node]
        query_vector = self.query_projection(query_hidden)
        record_vector = self.record_projection(hidden)
        logits = torch.einsum("bd,bnd->bn", query_vector, record_vector) / math.sqrt(
            hidden.shape[-1]
        )
        logits = float(beta_multiplier) * logits
        masked_logits = logits.masked_fill(~batch.record_mask, -1.0e9)
        probability = torch.softmax(masked_logits, dim=-1).masked_fill(
            ~batch.record_mask, 0.0
        )
        contributions = probability * batch.value
        query_x = effective_x[batch_index, batch.query_node]
        local = self.local_head(query_x).squeeze(-1)
        contributions = contributions.scatter(
            1, batch.query_node[:, None], local[:, None]
        )
        contributions = contributions * batch.mask.to(contributions.dtype)
        return RouterOutput(
            node_contributions=contributions,
            graph_prediction=contributions.sum(dim=-1),
            final_hidden=hidden,
            router_logits=masked_logits,
            router_probability=probability,
        )


def analytic_linear_contributions(
    graph: QueryGraph,
    query_onehot: torch.Tensor,
) -> torch.Tensor:
    contributions = torch.zeros(
        graph.num_nodes, dtype=query_onehot.dtype, device=query_onehot.device
    )
    codes = local_codes(graph.x.shape[1], dtype=query_onehot.dtype).to(query_onehot.device)
    contributions[graph.query_node] = torch.dot(codes, query_onehot)
    record_keys = graph.x[graph.record_nodes].to(
        device=query_onehot.device, dtype=query_onehot.dtype
    )
    values = graph.value[graph.record_nodes].to(
        device=query_onehot.device, dtype=query_onehot.dtype
    )
    contributions[graph.record_nodes] = (record_keys @ query_onehot) * values
    return contributions


# --------------------------------------------------------------------------------------
# Training and checkpoint health
# --------------------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _batch_indices(
    length: int,
    batch_size: int,
    *,
    rng: np.random.Generator,
    shuffle: bool,
) -> Iterable[np.ndarray]:
    indices = np.arange(int(length), dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


def _loss(
    output: RouterOutput,
    batch: QueryBatch,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    supervised = batch.record_mask.clone()
    supervised[
        torch.arange(supervised.shape[0], device=supervised.device),
        batch.query_node,
    ] = True
    squared = (output.node_contributions - batch.y_node).square()
    node_loss = squared.masked_select(supervised).mean()
    graph_loss = (output.graph_prediction - batch.y).square().mean()
    total = node_loss + float(config.graph_loss_weight) * graph_loss
    return total, node_loss, graph_loss


def evaluate_model(
    model: GraphGPSQueryRouter,
    graphs: Sequence[QueryGraph],
    config: ExperimentConfig,
    *,
    device: torch.device,
    batch_size: int | None = None,
) -> dict[str, float]:
    model.eval()
    batch_size = int(batch_size or config.batch_size)
    node_errors: list[torch.Tensor] = []
    graph_errors: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    correct: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    with torch.no_grad():
        rng = np.random.default_rng(0)
        for indices in _batch_indices(
            len(graphs), batch_size, rng=rng, shuffle=False
        ):
            batch = collate_graphs([graphs[int(index)] for index in indices]).to(device)
            output = model(batch)
            supervised = batch.record_mask.clone()
            supervised[
                torch.arange(supervised.shape[0], device=device),
                batch.query_node,
            ] = True
            node_errors.append(
                (output.node_contributions - batch.y_node)
                .abs()
                .masked_select(supervised)
                .detach()
                .cpu()
            )
            graph_errors.append(
                (output.graph_prediction - batch.y).abs().detach().cpu()
            )
            query_key = torch.argmax(
                batch.x[
                    torch.arange(batch.x.shape[0], device=device),
                    batch.query_node,
                ],
                dim=-1,
            )
            record_keys = torch.argmax(batch.x, dim=-1)
            matched = batch.record_mask & record_keys.eq(query_key[:, None])
            match_probability = output.router_probability.masked_select(matched)
            probabilities.append(match_probability.detach().cpu())
            prediction = torch.argmax(output.router_probability, dim=-1)
            target = torch.argmax(matched.to(torch.long), dim=-1)
            correct.append(prediction.eq(target).to(torch.float32).detach().cpu())
            safe = output.router_probability.clamp_min(1.0e-12)
            entropy = -(safe * safe.log()).sum(dim=-1) / math.log(config.num_keys)
            entropies.append(entropy.detach().cpu())
    return {
        "node_mae": float(torch.cat(node_errors).mean()),
        "graph_mae": float(torch.cat(graph_errors).mean()),
        "matched_probability": float(torch.cat(probabilities).mean()),
        "top1_accuracy": float(torch.cat(correct).mean()),
        "normalised_entropy": float(torch.cat(entropies).mean()),
    }


def _checkpoint_path(output_dir: Path, seed: int) -> Path:
    return output_dir / "checkpoints" / f"seed_{int(seed):03d}.pt"


def train_one(
    config: ExperimentConfig,
    *,
    seed: int,
    train_graphs: Sequence[QueryGraph],
    validation_graphs: Sequence[QueryGraph],
    device: torch.device,
    progress_path: Path | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    _seed_everything(seed)
    model = GraphGPSQueryRouter(config, init_seed=seed).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    rng = np.random.default_rng([config.data_seed, seed, 99])
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    best_epoch = 0
    patience = 0
    history: list[dict[str, float | int]] = []
    first_epoch = 1
    previous_seconds = 0.0
    started = time.time()
    if progress_path is not None and progress_path.exists():
        resume = _load_torch(progress_path)
        if resume.get("fingerprint") != config.fingerprint or int(
            resume.get("seed", -1)
        ) != int(seed):
            raise RuntimeError(f"training-progress contract mismatch: {progress_path}")
        model.load_state_dict(resume["model_state"])
        optimiser.load_state_dict(resume["optimiser_state"])
        best_state = resume["best_state"]
        best_validation = float(resume["best_validation"])
        best_epoch = int(resume["best_epoch"])
        patience = int(resume["patience"])
        history = list(resume["history"])
        resumed_epoch = int(resume["last_epoch"])
        first_epoch = resumed_epoch + 1
        if bool(resume.get("complete", False)):
            first_epoch = int(config.max_epochs) + 1
        previous_seconds = float(resume.get("seconds", 0.0))
        rng.bit_generator.state = resume["numpy_rng_state"]
        torch.set_rng_state(resume["torch_rng_state"])
        if device.type == "cuda" and resume.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        if progress:
            print(
                f"[train] seed={seed} resume after epoch={resumed_epoch:03d}",
                flush=True,
            )
    for epoch in range(first_epoch, int(config.max_epochs) + 1):
        model.train()
        total_sum = 0.0
        node_sum = 0.0
        graph_sum = 0.0
        batches = 0
        for indices in _batch_indices(
            len(train_graphs), config.batch_size, rng=rng, shuffle=True
        ):
            batch = collate_graphs(
                [train_graphs[int(index)] for index in indices]
            ).to(device)
            output = model(batch)
            total, node_loss, graph_loss = _loss(output, batch, config)
            optimiser.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.gradient_clip_norm)
            )
            optimiser.step()
            total_sum += float(total.detach())
            node_sum += float(node_loss.detach())
            graph_sum += float(graph_loss.detach())
            batches += 1
        validation = evaluate_model(
            model, validation_graphs, config, device=device
        )
        row = {
            "epoch": epoch,
            "train_total": total_sum / max(1, batches),
            "train_node": node_sum / max(1, batches),
            "train_graph": graph_sum / max(1, batches),
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        if progress:
            print(
                f"[train] seed={seed} epoch={epoch:03d} "
                f"node_mae={validation['node_mae']:.4f} "
                f"graph_mae={validation['graph_mae']:.4f} "
                f"p(match)={validation['matched_probability']:.4f}",
                flush=True,
            )
        if validation["node_mae"] < best_validation - 1.0e-7:
            best_validation = validation["node_mae"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if progress_path is not None:
            _atomic_torch(
                progress_path,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "fingerprint": config.fingerprint,
                    "seed": int(seed),
                    "last_epoch": int(epoch),
                    "model_state": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "optimiser_state": optimiser.state_dict(),
                    "best_state": best_state,
                    "best_validation": float(best_validation),
                    "best_epoch": int(best_epoch),
                    "patience": int(patience),
                    "history": history,
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state_all()
                        if device.type == "cuda"
                        else None
                    ),
                    "seconds": round(
                        previous_seconds + time.time() - started,
                        3,
                    ),
                    "complete": bool(
                        patience >= int(config.early_stop_patience)
                        or epoch >= int(config.max_epochs)
                    ),
                },
            )
        if patience >= int(config.early_stop_patience):
            break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    final_validation = evaluate_model(
        model, validation_graphs, config, device=device
    )
    return {
        "state_dict": best_state,
        "training": {
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "epochs": len(history),
            "best_validation_node_mae": float(best_validation),
            "final_validation": final_validation,
            "seconds": round(
                previous_seconds + time.time() - started,
                3,
            ),
            "history": history,
        },
    }


def ensure_checkpoints(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    device: torch.device,
    progress: bool = True,
) -> list[Path]:
    train_graphs = load_split(output_dir, config, "train")
    validation_graphs = load_split(output_dir, config, "validation")
    paths: list[Path] = []
    for seed in config.seeds:
        path = _checkpoint_path(output_dir, seed)
        paths.append(path)
        if path.exists():
            payload = _load_torch(path)
            if payload.get("fingerprint") != config.fingerprint:
                raise RuntimeError(f"checkpoint contract mismatch: {path}")
            if progress:
                print(f"[train] reuse {path}", flush=True)
            continue
        result = train_one(
            config,
            seed=seed,
            train_graphs=train_graphs,
            validation_graphs=validation_graphs,
            device=device,
            progress_path=(
                output_dir
                / "checkpoints"
                / f"seed_{int(seed):03d}.progress.pt"
            ),
            progress=progress,
        )
        _atomic_torch(
            path,
            {
                "protocol_version": PROTOCOL_VERSION,
                "fingerprint": config.fingerprint,
                **result,
            },
        )
        _atomic_json(
            output_dir / "checkpoints" / f"seed_{int(seed):03d}.training.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "fingerprint": config.fingerprint,
                "training": result["training"],
            },
        )
    return paths


def load_model(
    output_dir: Path,
    config: ExperimentConfig,
    seed: int,
    *,
    device: torch.device,
) -> tuple[GraphGPSQueryRouter, dict[str, Any], Path]:
    path = _checkpoint_path(output_dir, seed)
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint {path}; run --phase train first")
    payload = _load_torch(path)
    if payload.get("fingerprint") != config.fingerprint:
        raise RuntimeError(f"checkpoint contract mismatch: {path}")
    model = GraphGPSQueryRouter(config, init_seed=0)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload, path


# --------------------------------------------------------------------------------------
# Canonical semantic donor task declaration
# --------------------------------------------------------------------------------------


class QueryKeyContentAdapter:
    """Replace the complete one-hot query key and no other field."""

    def rows(self, data: QueryGraph) -> np.ndarray:
        return data.x.detach().cpu().numpy()

    def num_symbols(self, cfg: Any) -> None:
        return None

    def write_donors(self, batch_x, row_idx, donor_rows) -> None:
        values = torch.as_tensor(
            donor_rows, device=batch_x.device, dtype=batch_x.dtype
        )
        batch_x[row_idx] = values.reshape(len(row_idx), -1)


@dataclass(frozen=True)
class QueryRoutingTask:
    content_adapter: ContentAdapter = dataclasses.field(
        default_factory=QueryKeyContentAdapter
    )
    backend_kind: str = "synthetic_query_routing"
    semantic_fields: tuple[str, ...] = ("x",)
    immutable_control_fields: tuple[str, ...] = (
        "role",
        "value",
        "y_node",
        "record_nodes",
        "y",
    )
    node_structural_fields: tuple[str, ...] = ("pe",)
    pair_structural_fields: tuple[tuple[str, str], ...] = ()
    dense_pair_structural_fields: tuple[str, ...] = ()
    fixed_support_fields: tuple[str, ...] = ("edge_index", "spd")
    extra_known_fields: tuple[str, ...] = ()
    name: str = "query_routing_carriage"


TASK = QueryRoutingTask()


def build_graph_events(
    graph: QueryGraph,
    *,
    split: str,
    config: ExperimentConfig,
    donor_pool: SemanticDonorPool,
) -> tuple[list[QueryGraph], list[DonorEvent]]:
    rng = np.random.default_rng(
        [config.event_seed, int(graph.graph_id), 1 if split == "ood" else 0]
    )
    variants, records = build_channel_events(
        graph,
        graph_id=int(graph.graph_id),
        source=int(graph.query_node),
        channel="semantic",
        stage=split,
        donors=int(config.donors_per_source),
        rng=rng,
        task=TASK,
        semantic_pool=donor_pool,
        duplicate_tolerance=1.0e-7,
    )
    expected = math.sqrt(2.0)
    for event, variant in zip(records, variants):
        if not math.isclose(float(event.dose), expected, rel_tol=0.0, abs_tol=1.0e-6):
            raise RuntimeError(
                f"one-hot donor dose is {event.dose}, expected sqrt(2)"
            )
        donor_key = int(torch.argmax(variant.x[graph.query_node]))
        record_keys = torch.argmax(graph.x[graph.record_nodes], dim=-1)
        if int((record_keys == donor_key).sum()) != 1:
            raise RuntimeError("donor query does not resolve to exactly one base record")
    return variants, records


# --------------------------------------------------------------------------------------
# Donor-resolved measurements
# --------------------------------------------------------------------------------------


def _native_beta(value: float) -> bool:
    return math.isclose(float(value), 1.0, rel_tol=0.0, abs_tol=1.0e-12)


def _matched_record(graph: QueryGraph, query: torch.Tensor) -> int:
    key = int(torch.argmax(query).item())
    record_keys = torch.argmax(graph.x[graph.record_nodes], dim=-1)
    matched = graph.record_nodes[record_keys == key]
    if int(matched.numel()) != 1:
        raise RuntimeError(f"query key {key} has {int(matched.numel())} matching records")
    return int(matched.item())


def _directional_jacobian_fields(
    model: GraphGPSQueryRouter,
    batch: QueryBatch,
    clean_query: torch.Tensor,
    donor_queries: torch.Tensor,
    *,
    beta_multiplier: float,
    nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_query = clean_query.detach().clone()

    def carrier_map(query: torch.Tensor) -> torch.Tensor:
        output = model(
            batch,
            beta_multiplier=float(beta_multiplier),
            query_override=query.unsqueeze(0),
        )
        return output.node_contributions[0, :nodes]

    # The input key has K dimensions while the carrier vector has up to O(10^2) entries.
    # Forward-mode therefore needs K tangents instead of an output-sized reverse-mode batch.  This
    # is materially smaller on Colab GPUs and implements the registered direction-matched JVP.
    # Some PyTorch attention kernels have lacked forward-AD support, so retain an audited reverse
    # mode fallback rather than making the scientific run backend-dependent.
    basis = torch.eye(
        clean_query.numel(),
        dtype=clean_query.dtype,
        device=clean_query.device,
    )
    try:
        tangents = [
            torch.func.jvp(
                carrier_map,
                (clean_query,),
                (basis[index],),
            )[1]
            for index in range(clean_query.numel())
        ]
        jacobian = torch.stack(tangents, dim=-1)
    except (RuntimeError, NotImplementedError):
        jacobian = torch.autograd.functional.jacobian(
            carrier_map,
            clean_query.requires_grad_(True),
            create_graph=False,
            vectorize=True,
        )
    direction = clean_query.detach().unsqueeze(0) - donor_queries
    directional = torch.abs(torch.einsum("nk,dk->dn", jacobian, direction))
    entrywise_l1 = torch.abs(jacobian).sum(dim=-1)
    return directional.detach(), entrywise_l1.detach(), jacobian.detach()


def _field_metrics(
    field: np.ndarray,
    oracle: np.ndarray,
    distances: np.ndarray,
    *,
    effect_floor: float,
    far_threshold: int = 3,
) -> dict[str, np.ndarray]:
    field = np.asarray(field, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    field_norm, field_ok, field_total = event_normalise_functional(
        field, effect_floor=effect_floor
    )
    oracle_norm, oracle_ok, oracle_total = event_normalise_functional(
        oracle, effect_floor=effect_floor
    )
    eligible = field_ok & oracle_ok
    tv = np.full(field.shape[0], np.nan, dtype=np.float64)
    range_value = np.full_like(tv, np.nan)
    oracle_range = np.full_like(tv, np.nan)
    far_share = np.full_like(tv, np.nan)
    oracle_far_share = np.full_like(tv, np.nan)
    raw_error = np.full_like(tv, np.nan)
    if np.any(eligible):
        tv[eligible] = 0.5 * np.abs(
            field_norm[eligible] - oracle_norm[eligible]
        ).sum(axis=-1)
        range_value[eligible] = (
            field_norm[eligible] * distances[None, :]
        ).sum(axis=-1)
        oracle_range[eligible] = (
            oracle_norm[eligible] * distances[None, :]
        ).sum(axis=-1)
        far = distances > int(far_threshold)
        far_share[eligible] = field_norm[eligible][:, far].sum(axis=-1)
        oracle_far_share[eligible] = oracle_norm[eligible][:, far].sum(axis=-1)
        raw_error[eligible] = (
            np.abs(field[eligible] - oracle[eligible]).sum(axis=-1)
            / oracle_total[eligible]
        )
    return {
        "normalised": field_norm,
        "eligible": eligible,
        "total": field_total,
        "tv": tv,
        "range": range_value,
        "oracle_range": oracle_range,
        "range_error": np.abs(range_value - oracle_range),
        "far_share": far_share,
        "oracle_far_share": oracle_far_share,
        "far_share_error": np.abs(far_share - oracle_far_share),
        "raw_error": raw_error,
    }


def _clean_routing_diagnostics(
    graph: QueryGraph,
    output: RouterOutput,
) -> dict[str, float]:
    query = graph.x[graph.query_node]
    matched = _matched_record(graph, query)
    probability = output.router_probability[0, : graph.num_nodes]
    record_probability = probability[graph.record_nodes]
    matched_probability = float(probability[matched].detach().cpu())
    safe = record_probability.clamp_min(1.0e-12)
    entropy = float(
        (-(safe * safe.log()).sum() / math.log(len(graph.record_nodes)))
        .detach()
        .cpu()
    )
    logits = output.router_logits[0, graph.record_nodes]
    top = torch.topk(logits, k=min(2, int(logits.numel()))).values
    margin = float((top[0] - top[-1]).detach().cpu())
    return {
        "matched_probability": matched_probability,
        "confidence_stratum": (
            "low"
            if matched_probability < 0.80
            else "moderate"
            if matched_probability < 0.95
            else "high"
        ),
        "normalised_entropy": entropy,
        "logit_margin": margin,
        "derivative_factor": matched_probability * (1.0 - matched_probability),
    }


def _dose_ladder_measurements(
    model: GraphGPSQueryRouter,
    event_batch: QueryBatch,
    clean_query: torch.Tensor,
    donor_queries: torch.Tensor,
    clean_contribution: torch.Tensor,
    directional_jacobian: torch.Tensor,
    config: ExperimentConfig,
    *,
    nodes: int,
) -> dict[str, dict[str, torch.Tensor | float]]:
    """Compare finite interpolation responses with the clean tangent off protocol.

    Only ``alpha=1`` is a valid semantic intervention. Smaller alphas are retained solely as a
    convergence diagnostic showing where the tangent ceases to approximate the finite path.
    """

    result: dict[str, dict[str, torch.Tensor | float]] = {}
    for alpha in config.dose_alphas:
        mixed_queries = clean_query.unsqueeze(0) + float(alpha) * (
            donor_queries - clean_query.unsqueeze(0)
        )
        with torch.no_grad():
            interpolated = model(
                event_batch,
                beta_multiplier=1.0,
                query_override=mixed_queries,
            )
        finite = torch.abs(
            clean_contribution.unsqueeze(0)
            - interpolated.node_contributions[:, :nodes]
        )
        tangent = float(alpha) * directional_jacobian
        finite_np = finite.detach().cpu().numpy()
        tangent_np = tangent.detach().cpu().numpy()
        finite_norm, finite_ok, finite_total = event_normalise_functional(
            finite_np,
            effect_floor=config.effect_floor,
        )
        tangent_norm, tangent_ok, tangent_total = event_normalise_functional(
            tangent_np,
            effect_floor=config.effect_floor,
        )
        eligible = finite_ok & tangent_ok
        tv = np.full(len(donor_queries), np.nan, dtype=np.float64)
        relative_l1 = np.full_like(tv, np.nan)
        if np.any(eligible):
            tv[eligible] = 0.5 * np.abs(
                finite_norm[eligible] - tangent_norm[eligible]
            ).sum(axis=-1)
            relative_l1[eligible] = (
                np.abs(finite_np[eligible] - tangent_np[eligible]).sum(axis=-1)
                / finite_total[eligible]
            )
        result[f"{float(alpha):g}"] = {
            "alpha": float(alpha),
            "finite": finite.detach().cpu(),
            "jacobian_linearised": tangent.detach().cpu(),
            "finite_total": torch.from_numpy(finite_total),
            "jacobian_total": torch.from_numpy(tangent_total),
            "finite_jacobian_tv": torch.from_numpy(tv),
            "finite_jacobian_relative_l1": torch.from_numpy(relative_l1),
        }
    return result


def _measure_graph_with_model(
    model: GraphGPSQueryRouter,
    graph: QueryGraph,
    variants: Sequence[QueryGraph],
    events: Sequence[DonorEvent],
    config: ExperimentConfig,
    *,
    split: str,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nodes = graph.num_nodes
    clean_batch = collate_graphs([graph]).to(device)
    event_batch = collate_graphs(list(variants)).to(device)
    clean_query = graph.x[graph.query_node].to(device)
    donor_queries = torch.stack(
        [variant.x[graph.query_node] for variant in variants]
    ).to(device)
    clean_teacher = teacher_contributions(graph)
    oracle = torch.stack(
        [
            torch.abs(
                clean_teacher
                - teacher_contributions(graph, variant.x[graph.query_node])
            )
            for variant in variants
        ]
    )
    distances = graph.spd[graph.query_node, :nodes].to(torch.float64)
    graph_payload: dict[str, Any] = {
        "graph_id": int(graph.graph_id),
        "source": int(graph.query_node),
        "num_nodes": int(nodes),
        "family": str(graph.family),
        "distance": distances,
        "oracle": oracle,
        "events": [event.record() for event in events],
        "clean_key": int(torch.argmax(graph.x[graph.query_node])),
        "donor_keys": [
            int(torch.argmax(variant.x[graph.query_node])) for variant in variants
        ],
        "old_record": _matched_record(graph, graph.x[graph.query_node]),
        "new_records": [
            _matched_record(graph, variant.x[graph.query_node])
            for variant in variants
        ],
        "beta": {},
        "dose_ladder": {},
        "dose_rows": [],
    }
    rows: list[dict[str, Any]] = []
    for beta in config.beta_multipliers:
        with torch.no_grad():
            clean_output = model(clean_batch, beta_multiplier=beta)
            event_output = model(event_batch, beta_multiplier=beta)
            delta = (
                clean_output.node_contributions[0, :nodes].unsqueeze(0)
                - event_output.node_contributions[:, :nodes]
            )
            gradient = torch.ones((1, nodes, 1), dtype=delta.dtype, device=device)
            functional = functional_carriage_events(
                delta.unsqueeze(0).unsqueeze(-1),
                gradient,
            )[0]
            diagnostics = _clean_routing_diagnostics(graph, clean_output)
        directional, entrywise_l1, jacobian = _directional_jacobian_fields(
            model,
            clean_batch,
            clean_query,
            donor_queries,
            beta_multiplier=beta,
            nodes=nodes,
        )
        if config.compute_dose_ladder and _native_beta(beta):
            dose_ladder = _dose_ladder_measurements(
                model,
                event_batch,
                clean_query,
                donor_queries,
                clean_output.node_contributions[0, :nodes],
                directional,
                config,
                nodes=nodes,
            )
            graph_payload["dose_ladder"] = dose_ladder
            for dose in dose_ladder.values():
                for donor_index, event in enumerate(events):
                    graph_payload["dose_rows"].append(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "seed": int(seed),
                            "split": split,
                            "graph_id": int(graph.graph_id),
                            "source": int(graph.query_node),
                            "donor": int(donor_index),
                            "donor_graph_id": int(event.donor_graph_id),
                            "donor_node": int(event.donor_node),
                            "clean_key": int(graph_payload["clean_key"]),
                            "donor_key": int(
                                graph_payload["donor_keys"][donor_index]
                            ),
                            "semantic_endpoint": bool(
                                math.isclose(
                                    float(dose["alpha"]),
                                    1.0,
                                    rel_tol=0.0,
                                    abs_tol=1.0e-12,
                                )
                            ),
                            "alpha": float(dose["alpha"]),
                            "finite_total": float(
                                dose["finite_total"][donor_index]
                            ),
                            "jacobian_total": float(
                                dose["jacobian_total"][donor_index]
                            ),
                            "finite_jacobian_tv": float(
                                dose["finite_jacobian_tv"][donor_index]
                            ),
                            "finite_jacobian_relative_l1": float(
                                dose["finite_jacobian_relative_l1"][donor_index]
                            ),
                        }
                    )
        beneficial = None
        beneficial_diagnostics = None
        if config.compute_beneficial and _native_beta(beta):
            target = graph.y.to(device).reshape(1)

            def loss_from_states(states: torch.Tensor) -> torch.Tensor:
                prediction = states.sum(dim=(-2, -1))
                return (prediction - target[0]).square()

            result = beneficial_carriage(
                clean_output.node_contributions[0, :nodes].unsqueeze(-1),
                event_output.node_contributions[:, :nodes]
                .unsqueeze(0)
                .unsqueeze(-1),
                loss_from_states=loss_from_states,
                atol=config.integrated_atol,
                rtol=config.integrated_rtol,
                max_intervals=config.integrated_max_intervals,
                tolerance=config.integrated_tolerance,
            )
            beneficial = result.event_field[0].detach().cpu()
            beneficial_diagnostics = {
                "event_loss_increase": result.event_loss_increase[0].detach().cpu(),
                "completeness_residual": result.completeness_residual[0]
                .detach()
                .cpu(),
                "quadrature_error": result.quadrature_error[0].detach().cpu(),
                "intervals": result.intervals[0].detach().cpu(),
                "converged": result.converged[0].detach().cpu(),
            }
        method_fields = {
            "oracle": oracle.numpy(),
            "jacobian": directional.detach().cpu().numpy(),
            "functional": functional.detach().cpu().numpy(),
        }
        metrics = {
            method: _field_metrics(
                field,
                oracle.numpy(),
                distances.numpy(),
                effect_floor=config.effect_floor,
            )
            for method, field in method_fields.items()
        }
        clean_prediction = float(clean_output.graph_prediction[0].detach().cpu())
        clean_loss = (clean_prediction - float(graph.y[0])) ** 2
        event_prediction = event_output.graph_prediction.detach().cpu().numpy()
        event_loss = (event_prediction - float(graph.y[0])) ** 2
        event_target = np.asarray(
            [
                float(
                    teacher_contributions(
                        graph,
                        variant.x[graph.query_node],
                    ).sum()
                )
                for variant in variants
            ],
            dtype=np.float64,
        )
        counterfactual_mae = np.abs(event_prediction - event_target)
        event_probability = event_output.router_probability.detach().cpu()
        event_matched_probability = np.asarray(
            [
                float(event_probability[index, int(new_record)])
                for index, new_record in enumerate(graph_payload["new_records"])
            ]
        )
        event_top1 = torch.argmax(
            event_output.router_probability,
            dim=-1,
        ).detach().cpu().numpy()
        for donor_index, event in enumerate(events):
            common = {
                "protocol_version": PROTOCOL_VERSION,
                "seed": int(seed),
                "split": split,
                "graph_id": int(graph.graph_id),
                "source": int(graph.query_node),
                "donor": int(donor_index),
                "donor_graph_id": int(event.donor_graph_id),
                "donor_node": int(event.donor_node),
                "clean_key": int(graph_payload["clean_key"]),
                "donor_key": int(graph_payload["donor_keys"][donor_index]),
                "old_record": int(graph_payload["old_record"]),
                "new_record": int(graph_payload["new_records"][donor_index]),
                "degree_gap": int(event.degree_gap),
                "donor_dose": float(event.dose),
                "beta_multiplier": float(beta),
                "alpha": 1.0,
                "jacobian_entrywise_l1_total": float(entrywise_l1.sum().cpu()),
                **diagnostics,
                "clean_loss": float(clean_loss),
                "event_loss": float(event_loss[donor_index]),
                "counterfactual_target": float(event_target[donor_index]),
                "counterfactual_mae": float(
                    counterfactual_mae[donor_index]
                ),
                "event_matched_probability": float(
                    event_matched_probability[donor_index]
                ),
                "event_top1_correct": bool(
                    int(event_top1[donor_index])
                    == int(graph_payload["new_records"][donor_index])
                ),
            }
            for method in ("oracle", "jacobian", "functional"):
                values = metrics[method]
                common[f"{method}_total"] = float(
                    values["total"][donor_index]
                )
                common[f"{method}_tv"] = float(values["tv"][donor_index])
                common[f"{method}_range"] = float(values["range"][donor_index])
                common[f"{method}_range_error"] = float(
                    values["range_error"][donor_index]
                )
                common[f"{method}_far_share"] = float(
                    values["far_share"][donor_index]
                )
                common[f"{method}_far_share_error"] = float(
                    values["far_share_error"][donor_index]
                )
                common[f"{method}_raw_error"] = float(
                    values["raw_error"][donor_index]
                )
            common["tv_advantage"] = (
                common["jacobian_tv"] - common["functional_tv"]
            )
            common["range_error_advantage"] = (
                common["jacobian_range_error"]
                - common["functional_range_error"]
            )
            rows.append(common)
        graph_payload["beta"][f"{float(beta):g}"] = {
            "clean_contribution": clean_output.node_contributions[0, :nodes]
            .detach()
            .cpu(),
            "event_contribution": event_output.node_contributions[:, :nodes]
            .detach()
            .cpu(),
            "functional": functional.detach().cpu(),
            "jacobian_directional": directional.detach().cpu(),
            "jacobian_entrywise_l1": entrywise_l1.detach().cpu(),
            "jacobian": jacobian.detach().cpu(),
            "diagnostics": diagnostics,
            "beneficial": beneficial,
            "beneficial_diagnostics": beneficial_diagnostics,
        }
    return graph_payload, rows


def _measure_graph_linear(
    graph: QueryGraph,
    variants: Sequence[QueryGraph],
    events: Sequence[DonorEvent],
    config: ExperimentConfig,
    *,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    clean_query = graph.x[graph.query_node].to(torch.float64)
    donor_queries = torch.stack(
        [variant.x[graph.query_node] for variant in variants]
    ).to(torch.float64)

    def carrier_map(query: torch.Tensor) -> torch.Tensor:
        return analytic_linear_contributions(graph, query)

    clean = carrier_map(clean_query)
    event = torch.stack([carrier_map(query) for query in donor_queries])
    delta = clean.unsqueeze(0) - event
    gradient = torch.ones((1, graph.num_nodes, 1), dtype=torch.float64)
    functional = functional_carriage_events(
        delta.unsqueeze(0).unsqueeze(-1), gradient
    )[0]
    jacobian = torch.autograd.functional.jacobian(
        carrier_map, clean_query, vectorize=True
    )
    direction = clean_query.unsqueeze(0) - donor_queries
    directional = torch.abs(torch.einsum("nk,dk->dn", jacobian, direction))
    clean_teacher = teacher_contributions(graph).to(torch.float64)
    oracle = torch.stack(
        [
            torch.abs(
                clean_teacher
                - teacher_contributions(graph, query.to(DTYPE)).to(torch.float64)
            )
            for query in donor_queries
        ]
    )
    distances = graph.spd[graph.query_node, : graph.num_nodes].to(torch.float64)
    fields = {
        "oracle": oracle.numpy(),
        "jacobian": directional.numpy(),
        "functional": functional.numpy(),
    }
    metrics = {
        method: _field_metrics(
            field,
            oracle.numpy(),
            distances.numpy(),
            effect_floor=config.effect_floor,
        )
        for method, field in fields.items()
    }
    rows: list[dict[str, Any]] = []
    for donor_index, event_record in enumerate(events):
        row: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "seed": -1,
            "split": split,
            "graph_id": int(graph.graph_id),
            "source": int(graph.query_node),
            "donor": int(donor_index),
            "donor_graph_id": int(event_record.donor_graph_id),
            "donor_node": int(event_record.donor_node),
            "clean_key": int(torch.argmax(clean_query)),
            "donor_key": int(torch.argmax(donor_queries[donor_index])),
            "old_record": _matched_record(graph, clean_query.to(DTYPE)),
            "new_record": _matched_record(
                graph, donor_queries[donor_index].to(DTYPE)
            ),
            "degree_gap": int(event_record.degree_gap),
            "donor_dose": float(event_record.dose),
            "beta_multiplier": 1.0,
            "alpha": 1.0,
        }
        for method in ("oracle", "jacobian", "functional"):
            values = metrics[method]
            row[f"{method}_total"] = float(values["total"][donor_index])
            row[f"{method}_tv"] = float(values["tv"][donor_index])
            row[f"{method}_range"] = float(values["range"][donor_index])
            row[f"{method}_range_error"] = float(
                values["range_error"][donor_index]
            )
            row[f"{method}_far_share"] = float(
                values["far_share"][donor_index]
            )
            row[f"{method}_far_share_error"] = float(
                values["far_share_error"][donor_index]
            )
            row[f"{method}_raw_error"] = float(values["raw_error"][donor_index])
        row["tv_advantage"] = row["jacobian_tv"] - row["functional_tv"]
        row["range_error_advantage"] = (
            row["jacobian_range_error"] - row["functional_range_error"]
        )
        rows.append(row)
    return (
        {
            "graph_id": int(graph.graph_id),
            "source": int(graph.query_node),
            "num_nodes": int(graph.num_nodes),
            "distance": distances,
            "oracle": oracle,
            "jacobian_directional": directional,
            "functional": functional,
            "jacobian": jacobian,
            "events": [event.record() for event in events],
        },
        rows,
    )


def _measurement_path(output_dir: Path, seed: int, split: str) -> Path:
    return (
        output_dir
        / "cache"
        / "measurements"
        / f"seed_{int(seed):03d}_{split}.pt"
    )


def _measurement_shard_path(
    output_dir: Path,
    seed: int,
    split: str,
    graph: QueryGraph,
) -> Path:
    return (
        output_dir
        / "cache"
        / "measurement_shards"
        / f"seed_{int(seed):03d}_{split}"
        / f"graph_{int(graph.graph_id):09d}.pt"
    )


def _linear_path(output_dir: Path) -> Path:
    return output_dir / "cache" / "measurements" / "linear_control.pt"


def _donor_pool(
    output_dir: Path,
    config: ExperimentConfig,
) -> SemanticDonorPool:
    donors = load_split(output_dir, config, "donor")
    return SemanticDonorPool(
        [(graph.graph_id, graph) for graph in donors],
        adapter=TASK.content_adapter,
    )


def measure_linear_control(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    path = _linear_path(output_dir)
    if path.exists():
        payload = _load_torch(path)
        if payload.get("fingerprint") != config.fingerprint:
            raise RuntimeError(f"linear-control cache mismatch: {path}")
        return payload
    graphs = load_split(output_dir, config, "id")
    pool = _donor_pool(output_dir, config)
    graph_payloads: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    for index, graph in enumerate(graphs):
        variants, events = build_graph_events(
            graph, split="id", config=config, donor_pool=pool
        )
        payload, graph_rows = _measure_graph_linear(
            graph, variants, events, config, split="id"
        )
        graph_payloads.append(payload)
        rows.extend(graph_rows)
        event_records.extend(event.record() for event in events)
        if progress and (index == 0 or (index + 1) % max(1, len(graphs) // 10) == 0):
            print(
                f"[measure-linear] graph={index + 1}/{len(graphs)}",
                flush=True,
            )
    max_error = max(
        float(torch.max(torch.abs(item["functional"] - item["oracle"])))
        for item in graph_payloads
    )
    max_jacobian_error = max(
        float(
            torch.max(
                torch.abs(item["jacobian_directional"] - item["oracle"])
            )
        )
        for item in graph_payloads
    )
    tolerance = 2.0e-6
    if max(max_error, max_jacobian_error) > tolerance:
        raise RuntimeError(
            "linear exactness failed: "
            f"functional={max_error:.3g}, jacobian={max_jacobian_error:.3g}"
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": config.fingerprint,
        "graph_payloads": graph_payloads,
        "rows": rows,
        "event_manifest_fingerprint": manifest_fingerprint(event_records),
        "max_functional_error": max_error,
        "max_jacobian_error": max_jacobian_error,
    }
    _atomic_torch(path, payload)
    return payload


def measure_seed_split(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    seed: int,
    split: str,
    device: torch.device,
    progress: bool = True,
) -> dict[str, Any]:
    path = _measurement_path(output_dir, seed, split)
    if path.exists():
        payload = _load_torch(path)
        if payload.get("fingerprint") != config.fingerprint:
            raise RuntimeError(f"measurement cache mismatch: {path}")
        return payload
    model, checkpoint, checkpoint_path = load_model(
        output_dir, config, seed, device=device
    )
    checkpoint_digest = _sha256(checkpoint_path)
    graphs = load_split(output_dir, config, split)
    pool = _donor_pool(output_dir, config)
    health = evaluate_model(model, graphs, config, device=device)
    graph_payloads: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    dose_rows: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    started = time.time()
    for index, graph in enumerate(graphs):
        shard_path = _measurement_shard_path(
            output_dir,
            seed,
            split,
            graph,
        )
        if shard_path.exists():
            shard = _load_torch(shard_path)
            if (
                shard.get("fingerprint") != config.fingerprint
                or shard.get("checkpoint_sha256") != checkpoint_digest
                or int(shard.get("graph_id", -1)) != int(graph.graph_id)
            ):
                raise RuntimeError(f"measurement-shard contract mismatch: {shard_path}")
        else:
            variants, events = build_graph_events(
                graph, split=split, config=config, donor_pool=pool
            )
            graph_payload, graph_rows = _measure_graph_with_model(
                model,
                graph,
                variants,
                events,
                config,
                split=split,
                seed=seed,
                device=device,
            )
            shard = {
                "protocol_version": PROTOCOL_VERSION,
                "fingerprint": config.fingerprint,
                "checkpoint_sha256": checkpoint_digest,
                "seed": int(seed),
                "split": split,
                "graph_id": int(graph.graph_id),
                "graph_payload": graph_payload,
                "rows": graph_rows,
                "dose_rows": graph_payload["dose_rows"],
                "event_records": [event.record() for event in events],
            }
            _atomic_torch(shard_path, shard)
        graph_payloads.append(shard["graph_payload"])
        rows.extend(shard["rows"])
        dose_rows.extend(shard["dose_rows"])
        event_records.extend(shard["event_records"])
        if progress and (index == 0 or (index + 1) % max(1, len(graphs) // 10) == 0):
            print(
                f"[measure] seed={seed} split={split} "
                f"graph={index + 1}/{len(graphs)}",
                flush=True,
            )
    native_rows = [
        row for row in rows if _native_beta(row["beta_multiplier"])
    ]
    finite_error = float(
        np.nanmean([row["functional_raw_error"] for row in native_rows])
    )
    health = {
        **health,
        "finite_oracle_relative_error": finite_error,
        "passes_registered_gate": bool(
            health["top1_accuracy"] >= 0.98
            and health["matched_probability"] >= 0.95
            and health["node_mae"] <= 0.03
            and health["graph_mae"] <= 0.05
            and finite_error <= 0.10
        ),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": config.fingerprint,
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_training": checkpoint["training"],
        "seed": int(seed),
        "split": split,
        "health": health,
        "seconds": round(time.time() - started, 3),
        "event_manifest_fingerprint": manifest_fingerprint(event_records),
        "graph_payloads": graph_payloads,
        "rows": rows,
        "dose_rows": dose_rows,
    }
    _atomic_torch(path, payload)
    return payload


# --------------------------------------------------------------------------------------
# Aggregation and machine-readable results
# --------------------------------------------------------------------------------------


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(str(key))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    os.replace(temporary, path)


def _distance_field(
    field: np.ndarray,
    distances: np.ndarray,
    *,
    max_distance: int,
    effect_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(field, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.int64)
    normalised, eligible, _ = event_normalise_functional(
        field, effect_floor=effect_floor
    )
    raw_profile = np.zeros((field.shape[0], max_distance + 1), dtype=np.float64)
    norm_profile = np.full_like(raw_profile, np.nan)
    # An eligible event has exactly zero mass at an unoccupied graph distance.  Treating an
    # unoccupied bin as missing would make population profiles depend on every individual graph
    # supporting every distance and would propagate avoidable NaNs through the bootstrap.
    norm_profile[eligible] = 0.0
    for distance in range(max_distance + 1):
        cells = distances == distance
        if np.any(cells):
            raw_profile[:, distance] = field[:, cells].sum(axis=-1)
            norm_profile[eligible, distance] = normalised[eligible][:, cells].sum(
                axis=-1
            )
    return raw_profile, norm_profile


def _method_field(
    graph_payload: Mapping[str, Any],
    beta: float,
    method: str,
) -> np.ndarray:
    if method == "oracle":
        value = graph_payload["oracle"]
    elif method == "jacobian":
        value = graph_payload["beta"][f"{float(beta):g}"][
            "jacobian_directional"
        ]
    elif method == "functional":
        value = graph_payload["beta"][f"{float(beta):g}"]["functional"]
    elif method == "jacobian_entrywise":
        entrywise = np.asarray(
            graph_payload["beta"][f"{float(beta):g}"][
                "jacobian_entrywise_l1"
            ],
            dtype=np.float64,
        )
        donors = np.asarray(graph_payload["oracle"]).shape[0]
        value = np.broadcast_to(entrywise[None, :], (donors, entrywise.shape[0]))
    else:
        raise ValueError(f"unknown method {method!r}")
    return np.asarray(value, dtype=np.float64)


def _distance_support(
    graph_payloads: Sequence[Mapping[str, Any]],
    *,
    max_distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    graph_count = np.zeros(max_distance + 1, dtype=np.int64)
    carrier_source_pairs = np.zeros_like(graph_count)
    for graph in graph_payloads:
        distances = np.asarray(graph["distance"], dtype=np.int64)
        for distance in range(max_distance + 1):
            carriers = int(np.sum(distances == distance))
            if carriers > 0:
                graph_count[distance] += 1
                carrier_source_pairs[distance] += carriers
    return graph_count, carrier_source_pairs


def _point_estimate(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> np.ndarray:
    by_seed: dict[int, dict[int, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_seed[int(row["seed"])][int(row["graph_id"])].append(row)
    seed_values = []
    for graphs in by_seed.values():
        graph_values = []
        for events in graphs.values():
            graph_values.append(
                np.nanmean(
                    np.asarray(
                        [[float(event[field]) for field in fields] for event in events],
                        dtype=np.float64,
                    ),
                    axis=0,
                )
            )
        seed_values.append(
            trimmed_mean(np.stack(graph_values), proportion=0.20, axis=0)
        )
    return np.nanmean(np.stack(seed_values), axis=0)


def _interval_for_rows(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point = _point_estimate(rows, fields)
    if not config.production_bootstrap:
        return point, np.full_like(point, np.nan), np.full_like(point, np.nan)
    observations = [
        Observation(
            seed=int(row["seed"]),
            graph=int(row["graph_id"]),
            source=int(row["source"]),
            donor=int(row["donor"]),
            value=np.asarray([float(row[field]) for field in fields]),
        )
        for row in rows
    ]
    policy = dataclasses.replace(
        BootstrapPolicy(rng_seed=config.bootstrap_seed),
        resample_source=False,
    )
    interval = nested_percentile_interval(
        observations,
        policy,
        graph_reduce=lambda values: trimmed_mean(values, 0.20, axis=0),
    )
    return (
        np.asarray(interval.estimate),
        np.asarray(interval.low),
        np.asarray(interval.high),
    )


def _paired_grid_intervals(
    rows: Sequence[Mapping[str, Any]],
    *,
    grid_field: str,
    grid_values: Sequence[float],
    fields: Sequence[str],
    config: ExperimentConfig,
) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Bootstrap a complete intervention grid in one paired hierarchy draw."""

    values = tuple(float(value) for value in grid_values)
    by_event: dict[
        tuple[int, int, int, int],
        dict[float, Mapping[str, Any]],
    ] = defaultdict(dict)
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["graph_id"]),
            int(row["source"]),
            int(row["donor"]),
        )
        observed_grid = float(row[grid_field])
        match = next(
            (
                value
                for value in values
                if math.isclose(
                    observed_grid,
                    value,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ),
            None,
        )
        if match is None:
            raise RuntimeError(
                f"unregistered {grid_field}={observed_grid} in paired grid"
            )
        if match in by_event[key]:
            raise RuntimeError(f"duplicate {grid_field}={match} for event {key}")
        by_event[key][match] = row
    wide_fields = [
        f"grid_{grid_index}_{field}"
        for grid_index in range(len(values))
        for field in fields
    ]
    wide_rows: list[dict[str, Any]] = []
    for key, grid_rows in by_event.items():
        missing = [value for value in values if value not in grid_rows]
        if missing:
            raise RuntimeError(f"event {key} is missing {grid_field} values {missing}")
        wide: dict[str, Any] = {
            "seed": key[0],
            "graph_id": key[1],
            "source": key[2],
            "donor": key[3],
        }
        for grid_index, value in enumerate(values):
            for field in fields:
                wide[f"grid_{grid_index}_{field}"] = grid_rows[value][field]
        wide_rows.append(wide)
    estimate, low, high = _interval_for_rows(
        wide_rows,
        wide_fields,
        config,
    )
    width = len(fields)
    return {
        value: (
            estimate[index * width : (index + 1) * width],
            low[index * width : (index + 1) * width],
            high[index * width : (index + 1) * width],
        )
        for index, value in enumerate(values)
    }


def _profile_rows_for_cache(
    cache: Mapping[str, Any],
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    seed = int(cache["seed"])
    split = str(cache["split"])
    max_distance = max(
        int(np.max(np.asarray(graph["distance"])))
        for graph in cache["graph_payloads"]
    )
    support_graphs, support_pairs = _distance_support(
        cache["graph_payloads"],
        max_distance=max_distance,
    )
    rows: list[dict[str, Any]] = []
    for beta in config.beta_multipliers:
        for method in (
            "oracle",
            "jacobian",
            "functional",
            "jacobian_entrywise",
        ):
            graph_raw = []
            graph_norm = []
            for graph in cache["graph_payloads"]:
                raw, norm = _distance_field(
                    _method_field(graph, beta, method),
                    np.asarray(graph["distance"]),
                    max_distance=max_distance,
                    effect_floor=config.effect_floor,
                )
                graph_raw.append(np.nanmean(raw, axis=0))
                graph_norm.append(np.nanmean(norm, axis=0))
            raw_estimate = trimmed_mean(
                np.stack(graph_raw), proportion=0.20, axis=0
            )
            norm_estimate = trimmed_mean(
                np.stack(graph_norm), proportion=0.20, axis=0
            )
            for distance in range(max_distance + 1):
                rows.append(
                    {
                        "level": "seed",
                        "seed": seed,
                        "split": split,
                        "model": "learned",
                        "beta_multiplier": float(beta),
                        "method": method,
                        "distance": distance,
                        "raw": float(raw_estimate[distance]),
                        "normalised": float(norm_estimate[distance]),
                        "raw_low": np.nan,
                        "raw_high": np.nan,
                        "normalised_low": np.nan,
                        "normalised_high": np.nan,
                        "support_graphs": int(support_graphs[distance]),
                        "support_carrier_source_pairs": int(
                            support_pairs[distance]
                        ),
                        "reportable": bool(
                            support_graphs[distance] >= 10
                            and support_pairs[distance] >= 50
                        ),
                    }
                )
    return rows


def _population_profile_rows(
    caches: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    split = str(caches[0]["split"])
    max_distance = max(
        int(np.max(np.asarray(graph["distance"])))
        for cache in caches
        for graph in cache["graph_payloads"]
    )
    # Every seed is evaluated on the same frozen split. Count graph/topology support once rather
    # than inflating it by the number of trained seeds.
    support_graphs, support_pairs = _distance_support(
        caches[0]["graph_payloads"],
        max_distance=max_distance,
    )
    rows: list[dict[str, Any]] = []
    for beta in config.beta_multipliers:
        for method in (
            "oracle",
            "jacobian",
            "functional",
            "jacobian_entrywise",
        ):
            observations: list[Observation] = []
            for cache in caches:
                for graph in cache["graph_payloads"]:
                    raw, norm = _distance_field(
                        _method_field(graph, beta, method),
                        np.asarray(graph["distance"]),
                        max_distance=max_distance,
                        effect_floor=config.effect_floor,
                    )
                    for donor in range(raw.shape[0]):
                        observations.append(
                            Observation(
                                seed=int(cache["seed"]),
                                graph=int(graph["graph_id"]),
                                source=int(graph["source"]),
                                donor=donor,
                                value=np.concatenate((raw[donor], norm[donor])),
                            )
                        )
            if config.production_bootstrap and _native_beta(beta):
                policy = dataclasses.replace(
                    BootstrapPolicy(rng_seed=config.bootstrap_seed),
                    resample_source=False,
                )
                interval = nested_percentile_interval(
                    observations,
                    policy,
                    graph_reduce=lambda values: trimmed_mean(
                        values, 0.20, axis=0
                    ),
                )
                estimate = np.asarray(interval.estimate)
                low = np.asarray(interval.low)
                high = np.asarray(interval.high)
            else:
                per_seed: dict[int, dict[int, list[np.ndarray]]] = defaultdict(
                    lambda: defaultdict(list)
                )
                for observation in observations:
                    per_seed[observation.seed][observation.graph].append(
                        np.asarray(observation.value)
                    )
                seed_estimates = []
                for graphs in per_seed.values():
                    graph_estimates = [
                        np.nanmean(np.stack(events), axis=0)
                        for events in graphs.values()
                    ]
                    seed_estimates.append(
                        trimmed_mean(np.stack(graph_estimates), 0.20, axis=0)
                    )
                estimate = np.nanmean(np.stack(seed_estimates), axis=0)
                low = np.full_like(estimate, np.nan)
                high = np.full_like(estimate, np.nan)
            raw_estimate, norm_estimate = np.split(
                estimate, [max_distance + 1]
            )
            raw_low, norm_low = np.split(low, [max_distance + 1])
            raw_high, norm_high = np.split(high, [max_distance + 1])
            for distance in range(max_distance + 1):
                rows.append(
                    {
                        "level": "population",
                        "seed": "all",
                        "split": split,
                        "model": "learned",
                        "beta_multiplier": float(beta),
                        "method": method,
                        "distance": distance,
                        "raw": float(raw_estimate[distance]),
                        "normalised": float(norm_estimate[distance]),
                        "raw_low": float(raw_low[distance]),
                        "raw_high": float(raw_high[distance]),
                        "normalised_low": float(norm_low[distance]),
                        "normalised_high": float(norm_high[distance]),
                        "support_graphs": int(support_graphs[distance]),
                        "support_carrier_source_pairs": int(
                            support_pairs[distance]
                        ),
                        "reportable": bool(
                            support_graphs[distance] >= 10
                            and support_pairs[distance] >= 50
                        ),
                    }
                )
    return rows


def _linear_profile_rows(
    cache: Mapping[str, Any],
    config: ExperimentConfig,
) -> list[dict[str, Any]]:
    max_distance = max(
        int(np.max(np.asarray(graph["distance"])))
        for graph in cache["graph_payloads"]
    )
    support_graphs, support_pairs = _distance_support(
        cache["graph_payloads"],
        max_distance=max_distance,
    )
    rows: list[dict[str, Any]] = []
    for method in ("oracle", "jacobian", "functional"):
        graph_profiles = []
        for graph in cache["graph_payloads"]:
            field_name = (
                "oracle"
                if method == "oracle"
                else "jacobian_directional"
                if method == "jacobian"
                else "functional"
            )
            _, norm = _distance_field(
                np.asarray(graph[field_name]),
                np.asarray(graph["distance"]),
                max_distance=max_distance,
                effect_floor=config.effect_floor,
            )
            graph_profiles.append(np.nanmean(norm, axis=0))
        estimate = trimmed_mean(np.stack(graph_profiles), 0.20, axis=0)
        for distance, value in enumerate(estimate):
            rows.append(
                {
                    "level": "population",
                    "seed": -1,
                    "split": "id",
                    "model": "linear",
                    "beta_multiplier": 1.0,
                    "method": method,
                    "distance": distance,
                    "raw": np.nan,
                    "normalised": float(value),
                    "raw_low": np.nan,
                    "raw_high": np.nan,
                    "normalised_low": np.nan,
                    "normalised_high": np.nan,
                    "support_graphs": int(support_graphs[distance]),
                    "support_carrier_source_pairs": int(
                        support_pairs[distance]
                    ),
                    "reportable": bool(
                        support_graphs[distance] >= 10
                        and support_pairs[distance] >= 50
                    ),
                }
            )
    return rows


def _beneficial_profile_rows(
    caches: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    split = str(caches[0]["split"])
    max_distance = max(
        int(np.max(np.asarray(graph["distance"])))
        for cache in caches
        for graph in cache["graph_payloads"]
    )
    support_graphs, support_pairs = _distance_support(
        caches[0]["graph_payloads"],
        max_distance=max_distance,
    )
    per_seed: dict[int, list[np.ndarray]] = defaultdict(list)
    for cache in caches:
        seed = int(cache["seed"])
        for graph in cache["graph_payloads"]:
            beneficial = graph["beta"]["1"].get("beneficial")
            if beneficial is None:
                continue
            values = np.asarray(beneficial, dtype=np.float64)
            distances = np.asarray(graph["distance"], dtype=np.int64)
            profile = np.zeros((values.shape[0], max_distance + 1))
            for distance in range(max_distance + 1):
                profile[:, distance] = values[:, distances == distance].sum(axis=-1)
            per_seed[seed].append(np.nanmean(profile, axis=0))
    rows: list[dict[str, Any]] = []
    seed_estimates = []
    for seed, graph_values in sorted(per_seed.items()):
        estimate = trimmed_mean(np.stack(graph_values), 0.20, axis=0)
        seed_estimates.append(estimate)
        for distance, value in enumerate(estimate):
            rows.append(
                {
                    "level": "seed",
                    "seed": seed,
                    "split": split,
                    "distance": distance,
                    "beneficial_mass": float(value),
                    "support_graphs": int(support_graphs[distance]),
                    "support_carrier_source_pairs": int(
                        support_pairs[distance]
                    ),
                    "reportable": bool(
                        support_graphs[distance] >= 10
                        and support_pairs[distance] >= 50
                    ),
                }
            )
    if seed_estimates:
        population = np.nanmean(np.stack(seed_estimates), axis=0)
        for distance, value in enumerate(population):
            rows.append(
                {
                    "level": "population",
                    "seed": "all",
                    "split": split,
                    "distance": distance,
                    "beneficial_mass": float(value),
                    "support_graphs": int(support_graphs[distance]),
                    "support_carrier_source_pairs": int(
                        support_pairs[distance]
                    ),
                    "reportable": bool(
                        support_graphs[distance] >= 10
                        and support_pairs[distance] >= 50
                    ),
                }
            )
    return rows


def aggregate_results(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    progress: bool = True,
) -> dict[str, Any]:
    linear = _load_torch(_linear_path(output_dir))
    caches_by_split: dict[str, list[dict[str, Any]]] = {"id": [], "ood": []}
    all_event_rows: list[dict[str, Any]] = []
    all_dose_rows: list[dict[str, Any]] = []
    health_rows: list[dict[str, Any]] = []
    for split in ("id", "ood"):
        for seed in config.seeds:
            cache = _load_torch(_measurement_path(output_dir, seed, split))
            if cache.get("fingerprint") != config.fingerprint:
                raise RuntimeError("measurement fingerprint mismatch during aggregation")
            caches_by_split[split].append(cache)
            all_event_rows.extend(
                {
                    **row,
                    "experiment_fingerprint": config.fingerprint,
                    "checkpoint_sha256": cache["checkpoint_sha256"],
                }
                for row in cache["rows"]
            )
            all_dose_rows.extend(
                {
                    **row,
                    "experiment_fingerprint": config.fingerprint,
                    "checkpoint_sha256": cache["checkpoint_sha256"],
                }
                for row in cache.get("dose_rows", [])
            )
            health_rows.append(
                {
                    "seed": int(seed),
                    "split": split,
                    **cache["health"],
                    "checkpoint_sha256": cache["checkpoint_sha256"],
                    "event_manifest_fingerprint": cache[
                        "event_manifest_fingerprint"
                    ],
                }
            )
    _write_csv(output_dir / "results" / "event_metrics.csv", all_event_rows)
    _write_csv(
        output_dir / "results" / "linear_event_metrics.csv",
        [
            {
                **row,
                "experiment_fingerprint": config.fingerprint,
                "checkpoint_sha256": "analytic-linear-control",
            }
            for row in linear["rows"]
        ],
    )
    _write_csv(output_dir / "results" / "model_health.csv", health_rows)
    if all_dose_rows:
        _write_csv(output_dir / "results" / "dose_ladder_events.csv", all_dose_rows)

    profile_rows = _linear_profile_rows(linear, config)
    beneficial_rows: list[dict[str, Any]] = []
    for split, caches in caches_by_split.items():
        for cache in caches:
            profile_rows.extend(_profile_rows_for_cache(cache, config))
        profile_rows.extend(_population_profile_rows(caches, config))
        if config.compute_beneficial:
            beneficial_rows.extend(_beneficial_profile_rows(caches))
    _write_csv(output_dir / "results" / "distance_profiles.csv", profile_rows)
    if beneficial_rows:
        _write_csv(
            output_dir / "results" / "beneficial_profiles.csv", beneficial_rows
        )

    summary_records: list[dict[str, Any]] = []
    fields = (
        "jacobian_tv",
        "functional_tv",
        "tv_advantage",
        "jacobian_range_error",
        "functional_range_error",
        "range_error_advantage",
        "jacobian_far_share",
        "functional_far_share",
        "oracle_far_share",
        "matched_probability",
        "normalised_entropy",
        "logit_margin",
        "counterfactual_mae",
        "event_matched_probability",
        "event_top1_correct",
    )
    for split in ("id", "ood"):
        split_rows = [row for row in all_event_rows if row["split"] == split]
        intervals_by_beta = _paired_grid_intervals(
            split_rows,
            grid_field="beta_multiplier",
            grid_values=config.beta_multipliers,
            fields=fields,
            config=config,
        )
        for beta in config.beta_multipliers:
            selected = [
                row
                for row in split_rows
                if math.isclose(
                    float(row["beta_multiplier"]),
                    float(beta),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ]
            estimate, low, high = intervals_by_beta[float(beta)]
            record: dict[str, Any] = {
                "split": split,
                "beta_multiplier": float(beta),
                "events": len(selected),
            }
            for index, field in enumerate(fields):
                record[field] = float(estimate[index])
                record[f"{field}_low"] = float(low[index])
                record[f"{field}_high"] = float(high[index])
            summary_records.append(record)
    _write_csv(output_dir / "results" / "primary_contrasts.csv", summary_records)
    dose_summary_records: list[dict[str, Any]] = []
    if all_dose_rows:
        dose_fields = (
            "finite_jacobian_tv",
            "finite_jacobian_relative_l1",
            "finite_total",
            "jacobian_total",
        )
        for split in ("id", "ood"):
            split_dose_rows = [
                row for row in all_dose_rows if row["split"] == split
            ]
            intervals_by_alpha = _paired_grid_intervals(
                split_dose_rows,
                grid_field="alpha",
                grid_values=config.dose_alphas,
                fields=dose_fields,
                config=config,
            )
            for alpha in config.dose_alphas:
                selected = [
                    row
                    for row in split_dose_rows
                    if math.isclose(
                        float(row["alpha"]),
                        float(alpha),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                ]
                estimate, low, high = intervals_by_alpha[float(alpha)]
                record: dict[str, Any] = {
                    "split": split,
                    "alpha": float(alpha),
                    "semantic_endpoint": bool(_native_beta(alpha)),
                    "events": len(selected),
                }
                for index, field in enumerate(dose_fields):
                    record[field] = float(estimate[index])
                    record[f"{field}_low"] = float(low[index])
                    record[f"{field}_high"] = float(high[index])
                dose_summary_records.append(record)
        _write_csv(
            output_dir / "results" / "dose_ladder_summary.csv",
            dose_summary_records,
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": config.fingerprint,
        "smoke": bool(config.smoke),
        "linear": {
            "max_functional_error": float(linear["max_functional_error"]),
            "max_jacobian_error": float(linear["max_jacobian_error"]),
            "event_manifest_fingerprint": linear[
                "event_manifest_fingerprint"
            ],
        },
        "health": health_rows,
        "contrasts": summary_records,
        "dose_ladder": dose_summary_records,
        "files": {
            "event_metrics": str(output_dir / "results" / "event_metrics.csv"),
            "linear_event_metrics": str(
                output_dir / "results" / "linear_event_metrics.csv"
            ),
            "model_health": str(output_dir / "results" / "model_health.csv"),
            "distance_profiles": str(
                output_dir / "results" / "distance_profiles.csv"
            ),
            "beneficial_profiles": (
                str(output_dir / "results" / "beneficial_profiles.csv")
                if beneficial_rows
                else None
            ),
            "primary_contrasts": str(
                output_dir / "results" / "primary_contrasts.csv"
            ),
            "dose_ladder_events": (
                str(output_dir / "results" / "dose_ladder_events.csv")
                if all_dose_rows
                else None
            ),
            "dose_ladder_summary": (
                str(output_dir / "results" / "dose_ladder_summary.csv")
                if dose_summary_records
                else None
            ),
        },
    }
    _atomic_json(output_dir / "results" / "summary.json", payload)
    if progress:
        native = [
            row
            for row in summary_records
            if row["split"] == "id" and _native_beta(row["beta_multiplier"])
        ][0]
        print(
            f"[aggregate] ID native Delta_TV={native['tv_advantage']:.4f} "
            f"J_TV={native['jacobian_tv']:.4f} "
            f"F_TV={native['functional_tv']:.4f}",
            flush=True,
        )
    return payload


# --------------------------------------------------------------------------------------
# Cache-only publication figures
# --------------------------------------------------------------------------------------


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _plot_schematic(axis, graph: QueryGraph) -> None:
    network = nx.Graph()
    network.add_nodes_from(range(graph.num_nodes))
    network.add_edges_from(
        {
            tuple(sorted((int(left), int(right))))
            for left, right in graph.edge_index.t().tolist()
            if int(left) != int(right)
        }
    )
    query = int(graph.query_node)
    clean_key = int(torch.argmax(graph.x[query]))
    donor_key = int((clean_key + 1) % graph.x.shape[1])
    donor_query = torch.nn.functional.one_hot(
        torch.tensor(donor_key), num_classes=graph.x.shape[1]
    ).to(DTYPE)
    old_record = _matched_record(graph, graph.x[query])
    new_record = _matched_record(graph, donor_query)
    position = nx.spring_layout(network, seed=7)
    colours = []
    sizes = []
    for node in network.nodes:
        if node == query:
            colours.append("#C44E52")
            sizes.append(95)
        elif node == old_record:
            colours.append("#0072B2")
            sizes.append(90)
        elif node == new_record:
            colours.append("#E69F00")
            sizes.append(90)
        elif int(graph.role[node]) == ROLE_RECORD:
            colours.append("#8A8A86")
            sizes.append(48)
        else:
            colours.append("#D1D1CC")
            sizes.append(22)
    nx.draw_networkx_edges(network, position, ax=axis, width=0.45, alpha=0.45)
    nx.draw_networkx_nodes(
        network,
        position,
        ax=axis,
        node_color=colours,
        node_size=sizes,
        linewidths=0.4,
        edgecolors="white",
    )
    labels = {
        query: "query",
        old_record: f"old\nd={int(graph.spd[query, old_record])}",
        new_record: f"new\nd={int(graph.spd[query, new_record])}",
    }
    nx.draw_networkx_labels(network, position, labels=labels, font_size=6, ax=axis)
    axis.set_title("Valid query replacement and exact carriers")
    axis.set_axis_off()


def _profile_series(
    rows: Sequence[Mapping[str, str]],
    *,
    split: str,
    model: str,
    beta: float,
    method: str,
    enforce_reporting_floor: bool,
    quantity: str = "normalised",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if quantity not in {"normalised", "raw"}:
        raise ValueError("quantity must be 'normalised' or 'raw'")
    selected = [
        row
        for row in rows
        if row["level"] == "population"
        and row["split"] == split
        and row["model"] == model
        and row["method"] == method
        and math.isclose(_float(row, "beta_multiplier"), float(beta))
    ]
    selected.sort(key=lambda row: int(row["distance"]))
    reportable = np.asarray(
        [
            str(row.get("reportable", "True")).strip().lower()
            in {"true", "1", "yes"}
            for row in selected
        ],
        dtype=bool,
    )
    value = np.asarray([_float(row, quantity) for row in selected])
    low = np.asarray([_float(row, f"{quantity}_low") for row in selected])
    high = np.asarray([_float(row, f"{quantity}_high") for row in selected])
    if enforce_reporting_floor:
        value = np.where(reportable, value, np.nan)
        low = np.where(reportable, low, np.nan)
        high = np.where(reportable, high, np.nan)
    return (
        np.asarray([int(row["distance"]) for row in selected]),
        value,
        low,
        high,
    )


def _save_figure(
    figure,
    output_dir: Path,
    name: str,
) -> dict[str, str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{name}.png"
    pdf = figure_dir / f"{name}.pdf"
    figure.savefig(png, dpi=350, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    return {"png": str(png), "pdf": str(pdf)}


def render_figures(
    output_dir: Path,
    config: ExperimentConfig,
) -> dict[str, dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_path = output_dir / "results" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"missing {summary_path}; run --phase measure before figures"
        )
    summary = json.loads(summary_path.read_text())
    if summary.get("fingerprint") != config.fingerprint:
        raise RuntimeError("summary contract mismatch")
    health_by_split = {
        split: all(
            bool(row["passes_registered_gate"])
            for row in summary["health"]
            if row["split"] == split
        )
        for split in ("id", "ood")
    }
    profiles = _read_csv(output_dir / "results" / "distance_profiles.csv")
    contrasts = _read_csv(output_dir / "results" / "primary_contrasts.csv")
    id_graphs = load_split(output_dir, config, "id")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "legend.fontsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colours = {
        "oracle": "#7A5195",
        "jacobian": "#0072B2",
        "functional": "#E69F00",
        "jacobian_entrywise": "#009E73",
    }
    labels = {
        "oracle": "Known finite target",
        "jacobian": "Clean Jacobian",
        "functional": "Functional carriage",
        "jacobian_entrywise": "Entrywise-L1 Jacobian",
    }
    figures: dict[str, dict[str, str]] = {}

    figure, axes = plt.subplots(1, 4, figsize=(12.2, 3.0))
    _plot_schematic(axes[0], id_graphs[0])
    for method in ("oracle", "jacobian", "functional"):
        distance, value, _, _ = _profile_series(
            profiles,
            split="id",
            model="linear",
            beta=1.0,
            method=method,
            enforce_reporting_floor=not config.smoke,
        )
        axes[1].plot(
            distance,
            value,
            marker="o",
            ms=2.5,
            color=colours[method],
            label=labels[method],
        )
    axes[1].set_title("Linear agreement control")
    axes[1].set_xlabel("Distance from query")
    axes[1].set_ylabel("Event-normalised mass")
    axes[1].text(
        0.03,
        0.97,
        f"max errors\nJ={summary['linear']['max_jacobian_error']:.1e}\n"
        f"F={summary['linear']['max_functional_error']:.1e}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=7,
    )
    for method in ("oracle", "jacobian", "functional"):
        distance, value, low, high = _profile_series(
            profiles,
            split="id",
            model="learned",
            beta=1.0,
            method=method,
            enforce_reporting_floor=not config.smoke,
        )
        axes[2].plot(
            distance,
            value,
            marker="o",
            ms=2.5,
            color=colours[method],
            label=labels[method],
        )
        if np.isfinite(low).any():
            axes[2].fill_between(
                distance, low, high, color=colours[method], alpha=0.14
            )
    axes[2].set_title("Native learned routing")
    axes[2].set_xlabel("Distance from query")
    axes[2].set_ylabel("Event-normalised mass")
    beta_values = sorted(
        {
            _float(row, "beta_multiplier")
            for row in contrasts
            if row["split"] == "id"
        }
    )
    for method, field in (
        ("jacobian", "jacobian_tv"),
        ("functional", "functional_tv"),
    ):
        values = []
        lows = []
        highs = []
        for beta in beta_values:
            row = next(
                item
                for item in contrasts
                if item["split"] == "id"
                and math.isclose(_float(item, "beta_multiplier"), beta)
            )
            values.append(_float(row, field))
            lows.append(_float(row, f"{field}_low"))
            highs.append(_float(row, f"{field}_high"))
        axes[3].plot(
            beta_values,
            values,
            marker="o",
            color=colours[method],
            label=labels[method],
        )
        if np.isfinite(lows).any():
            axes[3].fill_between(
                beta_values,
                lows,
                highs,
                color=colours[method],
                alpha=0.14,
            )
    axes[3].axvline(1.0, color="#555555", lw=0.8, ls="--")
    axes[3].set_xscale("log", base=2)
    axes[3].set_title("Error across routing confidence")
    axes[3].set_xlabel("Logit multiplier (native = 1)")
    axes[3].set_ylabel("Profile TV error vs target")
    handles, legend_labels = axes[2].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.58, -0.02),
    )
    figure.suptitle(
        "Finite carriage under learned categorical graph routing",
        fontsize=11,
        y=1.02,
    )
    if not health_by_split["id"]:
        figure.text(
            0.5,
            0.01,
            "MODEL HEALTH GATE FAILED ON ID DATA — learned-model panels are descriptive only",
            ha="center",
            va="bottom",
            color="#B22222",
            fontsize=8,
            weight="bold",
        )
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figures["main"] = _save_figure(
        figure, output_dir, "query_routing_carriage"
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for axis, split, title in zip(
        axes, ("id", "ood"), ("In distribution", "Long-distance OOD")
    ):
        for method in ("oracle", "jacobian", "functional"):
            distance, value, low, high = _profile_series(
                profiles,
                split=split,
                model="learned",
                beta=1.0,
                method=method,
                enforce_reporting_floor=not config.smoke,
            )
            axis.plot(
                distance,
                value,
                marker="o",
                ms=2.5,
                color=colours[method],
                label=labels[method],
            )
            if np.isfinite(low).any():
                axis.fill_between(
                    distance, low, high, color=colours[method], alpha=0.14
                )
        axis.set_title(title)
        axis.set_xlabel("Distance from query")
        if not health_by_split[split]:
            axis.text(
                0.5,
                0.98,
                "HEALTH GATE FAILED",
                transform=axis.transAxes,
                ha="center",
                va="top",
                color="#B22222",
                fontsize=7,
                weight="bold",
            )
    axes[0].set_ylabel("Event-normalised mass")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figures["id_ood_profiles"] = _save_figure(
        figure, output_dir, "query_routing_carriage_id_ood"
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for axis, split, title in zip(
        axes, ("id", "ood"), ("In distribution", "Long-distance OOD")
    ):
        for method in ("oracle", "jacobian", "functional"):
            distance, value, low, high = _profile_series(
                profiles,
                split=split,
                model="learned",
                beta=1.0,
                method=method,
                enforce_reporting_floor=not config.smoke,
                quantity="raw",
            )
            axis.plot(
                distance,
                value,
                marker="o",
                ms=2.5,
                color=colours[method],
                label=labels[method],
            )
            if np.isfinite(low).any():
                axis.fill_between(
                    distance, low, high, color=colours[method], alpha=0.14
                )
        axis.set_title(title)
        axis.set_xlabel("Distance from query")
        if not health_by_split[split]:
            axis.text(
                0.5,
                0.98,
                "HEALTH GATE FAILED",
                transform=axis.transAxes,
                ha="center",
                va="top",
                color="#B22222",
                fontsize=7,
                weight="bold",
            )
    axes[0].set_ylabel("Raw carrier mass")
    axes[1].legend(frameon=False)
    figure.suptitle("Raw finite-response scale")
    figure.tight_layout()
    figures["raw_profiles"] = _save_figure(
        figure,
        output_dir,
        "query_routing_carriage_raw_profiles",
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for axis, split, title in zip(
        axes, ("id", "ood"), ("In distribution", "Long-distance OOD")
    ):
        oracle_distance, oracle_value, _, _ = _profile_series(
            profiles,
            split=split,
            model="learned",
            beta=1.0,
            method="oracle",
            enforce_reporting_floor=not config.smoke,
        )
        axis.plot(
            oracle_distance,
            oracle_value,
            color=colours["oracle"],
            lw=2.0,
            label=labels["oracle"],
        )
        seed_values = sorted(
            {
                int(row["seed"])
                for row in profiles
                if row["level"] == "seed"
                and row["split"] == split
                and row["model"] == "learned"
            }
        )
        for method in ("jacobian", "functional"):
            for seed_index, seed in enumerate(seed_values):
                selected = [
                    row
                    for row in profiles
                    if row["level"] == "seed"
                    and row["split"] == split
                    and row["model"] == "learned"
                    and row["method"] == method
                    and int(row["seed"]) == seed
                    and math.isclose(
                        _float(row, "beta_multiplier"),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                ]
                selected.sort(key=lambda row: int(row["distance"]))
                distance = np.asarray(
                    [int(row["distance"]) for row in selected]
                )
                value = np.asarray(
                    [_float(row, "normalised") for row in selected]
                )
                if not config.smoke:
                    reportable = np.asarray(
                        [
                            str(row.get("reportable", "True")).strip().lower()
                            in {"true", "1", "yes"}
                            for row in selected
                        ]
                    )
                    value = np.where(reportable, value, np.nan)
                axis.plot(
                    distance,
                    value,
                    color=colours[method],
                    lw=0.9,
                    alpha=0.45,
                    label=labels[method] if seed_index == 0 else None,
                )
        axis.set_title(title)
        axis.set_xlabel("Distance from query")
        if not health_by_split[split]:
            axis.text(
                0.5,
                0.98,
                "HEALTH GATE FAILED",
                transform=axis.transAxes,
                ha="center",
                va="top",
                color="#B22222",
                fontsize=7,
                weight="bold",
            )
    axes[0].set_ylabel("Event-normalised mass")
    axes[1].legend(frameon=False, fontsize=6.6)
    figure.suptitle("Native-temperature seed profiles (thin lines)")
    figure.tight_layout()
    figures["seed_profiles"] = _save_figure(
        figure,
        output_dir,
        "query_routing_carriage_seed_profiles",
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for axis, split, title in zip(
        axes, ("id", "ood"), ("In distribution", "Long-distance OOD")
    ):
        for method in (
            "oracle",
            "jacobian",
            "jacobian_entrywise",
            "functional",
        ):
            distance, value, low, high = _profile_series(
                profiles,
                split=split,
                model="learned",
                beta=1.0,
                method=method,
                enforce_reporting_floor=not config.smoke,
            )
            axis.plot(
                distance,
                value,
                marker="o",
                ms=2.2,
                lw=1.2,
                ls="--" if method == "jacobian_entrywise" else "-",
                color=colours[method],
                label=labels[method],
            )
            if np.isfinite(low).any():
                axis.fill_between(
                    distance, low, high, color=colours[method], alpha=0.12
                )
        axis.set_title(title)
        axis.set_xlabel("Distance from query")
        if not health_by_split[split]:
            axis.text(
                0.5,
                0.98,
                "HEALTH GATE FAILED",
                transform=axis.transAxes,
                ha="center",
                va="top",
                color="#B22222",
                fontsize=7,
                weight="bold",
            )
    axes[0].set_ylabel("Event-normalised mass")
    axes[1].legend(frameon=False, fontsize=6.6)
    figure.suptitle("Direction-matched and entrywise Jacobian geometries")
    figure.tight_layout()
    figures["entrywise_jacobian"] = _save_figure(
        figure,
        output_dir,
        "query_routing_carriage_entrywise_jacobian",
    )
    plt.close(figure)

    dose_path = output_dir / "results" / "dose_ladder_summary.csv"
    if dose_path.exists():
        dose_rows = _read_csv(dose_path)
        figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
        for axis, split, title in zip(
            axes, ("id", "ood"), ("In distribution", "Long-distance OOD")
        ):
            selected = [row for row in dose_rows if row["split"] == split]
            selected.sort(key=lambda row: _float(row, "alpha"))
            alpha = np.asarray([_float(row, "alpha") for row in selected])
            value = np.asarray(
                [_float(row, "finite_jacobian_tv") for row in selected]
            )
            low = np.asarray(
                [_float(row, "finite_jacobian_tv_low") for row in selected]
            )
            high = np.asarray(
                [_float(row, "finite_jacobian_tv_high") for row in selected]
            )
            axis.plot(alpha, value, marker="o", color="#CC6677")
            if np.isfinite(low).any():
                axis.fill_between(alpha, low, high, color="#CC6677", alpha=0.16)
            axis.axvline(1.0, color="#555555", lw=0.8, ls="--")
            axis.set_xscale("log")
            axis.set_title(title)
            axis.set_xlabel("Interpolation dose alpha")
            if not health_by_split[split]:
                axis.text(
                    0.5,
                    0.98,
                    "HEALTH GATE FAILED",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    color="#B22222",
                    fontsize=7,
                    weight="bold",
                )
            axis.text(
                0.98,
                0.04,
                "only alpha=1 is semantic",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
                color="#555555",
            )
        axes[0].set_ylabel("Profile TV: finite response vs clean tangent")
        figure.suptitle("Off-protocol donor-dose convergence diagnostic")
        figure.tight_layout()
        figures["dose_ladder"] = _save_figure(
            figure,
            output_dir,
            "query_routing_carriage_dose_ladder",
        )
        plt.close(figure)

    beneficial_path = output_dir / "results" / "beneficial_profiles.csv"
    if beneficial_path.exists():
        beneficial_rows = _read_csv(beneficial_path)
        figure, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
        for axis, split, title in zip(
            axes, ("id", "ood"), ("In distribution", "Long-distance OOD")
        ):
            selected = [
                row
                for row in beneficial_rows
                if row["level"] == "population" and row["split"] == split
            ]
            selected.sort(key=lambda row: int(row["distance"]))
            distance = np.asarray([int(row["distance"]) for row in selected])
            mass = np.asarray(
                [_float(row, "beneficial_mass") for row in selected]
            )
            if not config.smoke:
                reportable = np.asarray(
                    [
                        str(row.get("reportable", "True")).strip().lower()
                        in {"true", "1", "yes"}
                        for row in selected
                    ]
                )
                mass = np.where(reportable, mass, np.nan)
            axis.axhline(0.0, color="#444444", lw=0.8)
            axis.bar(distance, mass, color="#55A868")
            axis.set_title(title)
            axis.set_xlabel("Distance from query")
            if not health_by_split[split]:
                axis.text(
                    0.5,
                    0.98,
                    "HEALTH GATE FAILED",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    color="#B22222",
                    fontsize=7,
                    weight="bold",
                )
        axes[0].set_ylabel("Signed Beneficial-carriage mass")
        figure.suptitle("Task-loss allocation at native routing temperature")
        figure.tight_layout()
        figures["beneficial"] = _save_figure(
            figure, output_dir, "query_routing_carriage_beneficial"
        )
        plt.close(figure)

    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": config.fingerprint,
        "smoke": bool(config.smoke),
        "figures": figures,
        "source_files": summary["files"],
        "health_gate_by_split": health_by_split,
        "rendered_from_cache_only": True,
    }
    _atomic_json(
        output_dir / "figures" / "query_routing_carriage.metadata.json",
        metadata,
    )
    return figures


# --------------------------------------------------------------------------------------
# Phase orchestration and CLI
# --------------------------------------------------------------------------------------


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def measure_all(
    output_dir: Path,
    config: ExperimentConfig,
    *,
    device: torch.device,
    progress: bool = True,
) -> dict[str, Any]:
    linear = measure_linear_control(output_dir, config, progress=progress)
    learned: dict[str, Any] = {}
    for seed in config.seeds:
        for split in ("id", "ood"):
            cache = measure_seed_split(
                output_dir,
                config,
                seed=seed,
                split=split,
                device=device,
                progress=progress,
            )
            learned[f"seed_{seed:03d}_{split}"] = {
                "path": str(_measurement_path(output_dir, seed, split)),
                "health": cache["health"],
            }
    aggregate = aggregate_results(output_dir, config, progress=progress)
    return {
        "linear": {
            "path": str(_linear_path(output_dir)),
            "max_functional_error": linear["max_functional_error"],
            "max_jacobian_error": linear["max_jacobian_error"],
        },
        "learned": learned,
        "aggregate": aggregate,
    }


def run(
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
    phase: str = "all",
    device: str = "auto",
    progress: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    ensure_contract(output_dir, config)
    selected_device = choose_device(device)
    result: dict[str, Any] = {
        "output_dir": str(output_dir),
        "fingerprint": config.fingerprint,
        "device": str(selected_device),
        "smoke": bool(config.smoke),
    }
    if phase in {"all", "data"}:
        result["data"] = ensure_data(output_dir, config, progress=progress)
    if phase in {"all", "train"}:
        ensure_data(output_dir, config, progress=progress)
        result["checkpoints"] = [
            str(path)
            for path in ensure_checkpoints(
                output_dir,
                config,
                device=selected_device,
                progress=progress,
            )
        ]
    if phase in {"all", "measure"}:
        ensure_data(output_dir, config, progress=progress)
        ensure_checkpoints(
            output_dir,
            config,
            device=selected_device,
            progress=progress,
        )
        result["measurements"] = measure_all(
            output_dir,
            config,
            device=selected_device,
            progress=progress,
        )
    if phase in {"all", "figures"}:
        result["figures"] = render_figures(output_dir, config)
    return result


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run finite carriage versus Jacobian on learned query routing."
    )
    parser.add_argument(
        "--phase",
        choices=("all", "data", "train", "measure", "figures"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/query_routing_carriage_v1",
        help="Drive-backed directory in Colab; local directory otherwise.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--beta-multipliers", default="0.25,0.5,1,2,4")
    parser.add_argument("--dose-alphas", default="0.001,0.01,0.05,0.1,0.25,0.5,1")
    parser.add_argument("--num-keys", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--train-graphs", type=int, default=12_288)
    parser.add_argument("--validation-graphs", type=int, default=1_024)
    parser.add_argument("--id-graphs", type=int, default=256)
    parser.add_argument("--ood-graphs", type=int, default=256)
    parser.add_argument("--donor-graphs", type=int, default=512)
    parser.add_argument("--donors-per-source", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--skip-beneficial", action="store_true")
    parser.add_argument("--skip-dose-ladder", action="store_true")
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    torch.set_num_threads(max(1, int(args.num_threads)))
    if args.fast_dev_run:
        config = ExperimentConfig.fast_dev(
            seeds=_parse_csv_ints(args.seeds)[:1] or (0,)
        )
    else:
        config = ExperimentConfig(
            seeds=_parse_csv_ints(args.seeds),
            beta_multipliers=_parse_csv_floats(args.beta_multipliers),
            dose_alphas=_parse_csv_floats(args.dose_alphas),
            num_keys=args.num_keys,
            hidden_dim=args.hidden_dim,
            layers=args.layers,
            heads=args.heads,
            train_graphs=args.train_graphs,
            validation_graphs=args.validation_graphs,
            id_graphs=args.id_graphs,
            ood_graphs=args.ood_graphs,
            donor_graphs=args.donor_graphs,
            donors_per_source=args.donors_per_source,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            compute_beneficial=not args.skip_beneficial,
            compute_dose_ladder=not args.skip_dose_ladder,
            production_bootstrap=not args.no_bootstrap,
        )
    result = run(
        config,
        output_dir=args.output_dir,
        phase=args.phase,
        device=args.device,
    )
    if "figures" in result:
        for name, files in result["figures"].items():
            print(f"[figure:{name}] {files['png']}", flush=True)
            print(f"[figure:{name}] {files['pdf']}", flush=True)
    return result


if __name__ == "__main__":
    main()
