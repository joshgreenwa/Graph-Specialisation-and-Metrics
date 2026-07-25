"""Exact donor laws and auditable event manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .protocol import stable_hash


def payload_array(data: Any, adapter: Any | None = None) -> np.ndarray:
    rows = adapter.rows(data) if adapter is not None else data.x.detach().cpu().numpy()
    rows = np.asarray(rows)
    return rows[:, None] if rows.ndim == 1 else rows


def node_degrees(data: Any) -> np.ndarray:
    edge_index = data.edge_index.detach().cpu().numpy()
    degree = np.zeros(int(data.num_nodes), dtype=np.int64)
    if edge_index.size:
        np.add.at(degree, edge_index[0].astype(np.int64), 1)
    return degree


def payload_fingerprint(row: Any) -> str:
    array = np.ascontiguousarray(np.asarray(row))
    return stable_hash(
        {"dtype": str(array.dtype), "shape": array.shape, "bytes": array.tobytes().hex()}
    )


@dataclass(frozen=True)
class DonorNode:
    graph_id: int
    node: int
    degree: int
    payload: tuple[Any, ...]


@dataclass(frozen=True)
class DonorEvent:
    channel: str
    stage: str
    graph_id: int
    source: int
    donor_graph_id: int
    donor_node: int
    source_degree: int
    donor_degree: int
    degree_gap: int
    dose: float
    payload_fingerprint: str
    draw: int

    def record(self) -> dict[str, Any]:
        return asdict(self)


class SemanticDonorPool:
    """Graph-balanced semantic donor index implementing the normative law."""

    def __init__(self, donor_graphs: Sequence[tuple[int, Any]], adapter: Any | None = None):
        self.by_graph: dict[int, tuple[DonorNode, ...]] = {}
        for graph_id, graph in donor_graphs:
            rows = payload_array(graph, adapter)
            degrees = node_degrees(graph)
            self.by_graph[int(graph_id)] = tuple(
                DonorNode(
                    graph_id=int(graph_id),
                    node=int(node),
                    degree=int(degrees[node]),
                    payload=tuple(np.asarray(rows[node]).reshape(-1).tolist()),
                )
                for node in range(len(rows))
            )
        if not self.by_graph:
            raise ValueError("semantic donor pool is empty")

    def eligible(
        self,
        source_payload: Any,
        source_degree: int,
        *,
        base_graph_id: int | None = None,
    ) -> dict[int, tuple[DonorNode, ...]]:
        source = np.asarray(source_payload).reshape(-1)
        candidates: list[DonorNode] = []
        for graph_id, nodes in self.by_graph.items():
            if base_graph_id is not None and int(graph_id) == int(base_graph_id):
                continue
            candidates.extend(
                node
                for node in nodes
                if not np.array_equal(np.asarray(node.payload), source)
            )
        if not candidates:
            return {}
        gap = min(abs(int(node.degree) - int(source_degree)) for node in candidates)
        selected: dict[int, list[DonorNode]] = {}
        for node in candidates:
            if abs(int(node.degree) - int(source_degree)) == gap:
                selected.setdefault(node.graph_id, []).append(node)
        return {graph: tuple(nodes) for graph, nodes in selected.items()}

    def draw(
        self,
        source_payload: Any,
        source_degree: int,
        count: int,
        rng: np.random.Generator,
        *,
        base_graph_id: int | None = None,
    ) -> tuple[DonorNode, ...]:
        """Draw iid with replacement: eligible graph uniform, then node uniform."""

        eligible = self.eligible(
            source_payload, source_degree, base_graph_id=base_graph_id
        )
        graph_ids = np.asarray(sorted(eligible), dtype=np.int64)
        if not graph_ids.size:
            raise ValueError("semantic source has no non-identical eligible donor")
        result: list[DonorNode] = []
        for _ in range(int(count)):
            graph_id = int(rng.choice(graph_ids))
            nodes = eligible[graph_id]
            result.append(nodes[int(rng.integers(0, len(nodes)))])
        return tuple(result)


def structural_eligible_nodes(
    footprints: Sequence[Any],
    degrees: Sequence[int],
    source: int,
    *,
    equal: Any | None = None,
) -> np.ndarray:
    """Return every non-identical node attaining the minimum absolute degree gap."""

    source = int(source)
    compare = equal or (
        lambda left, right: np.array_equal(np.asarray(left), np.asarray(right))
    )
    candidates = np.asarray(
        [
            node
            for node in range(len(footprints))
            if node != source and not compare(footprints[node], footprints[source])
        ],
        dtype=np.int64,
    )
    if not candidates.size:
        return candidates
    degree = np.asarray(degrees, dtype=np.int64)
    gap = np.abs(degree[candidates] - degree[source])
    return candidates[gap == gap.min()]


def draw_structural_donors(
    footprints: Sequence[Any],
    degrees: Sequence[int],
    source: int,
    count: int,
    rng: np.random.Generator,
    *,
    equal: Any | None = None,
) -> np.ndarray:
    eligible = structural_eligible_nodes(footprints, degrees, source, equal=equal)
    if not eligible.size:
        return np.empty(0, dtype=np.int64)
    return rng.choice(eligible, size=int(count), replace=True).astype(np.int64)


def sample_sources(num_nodes: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    count = min(int(num_nodes), int(cap))
    return np.sort(rng.choice(int(num_nodes), size=count, replace=False)).astype(np.int64)


def manifest_fingerprint(events: Iterable[DonorEvent | dict[str, Any]]) -> str:
    records = [
        event.record() if isinstance(event, DonorEvent) else dict(event) for event in events
    ]
    return stable_hash({"events": records})
