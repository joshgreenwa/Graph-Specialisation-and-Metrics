"""Exact shortest-path accounting for scores and carriage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def shortest_path_distances(edge_index: Any, num_nodes: int) -> np.ndarray:
    """All-pairs pristine-graph distance; unreachable entries are ``inf``."""

    edge = (
        edge_index.detach().cpu().numpy()
        if hasattr(edge_index, "detach")
        else np.asarray(edge_index)
    )
    n = int(num_nodes)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for left, right in edge.T:
        left, right = int(left), int(right)
        adjacency[left].append(right)
        adjacency[right].append(left)
    distances = np.full((n, n), np.inf, dtype=np.float64)
    for source in range(n):
        distances[source, source] = 0
        queue = [source]
        for node in queue:
            candidate = distances[source, node] + 1
            for neighbour in adjacency[node]:
                if not np.isfinite(distances[source, neighbour]):
                    distances[source, neighbour] = candidate
                    queue.append(neighbour)
    return distances


@dataclass(frozen=True)
class DistanceAxis:
    labels: tuple[int | str, ...]

    @classmethod
    def from_matrices(
        cls, matrices: Iterable[np.ndarray], *, include_unreachable: bool = True
    ) -> "DistanceAxis":
        maximum = 0
        has_unreachable = False
        for matrix in matrices:
            matrix = np.asarray(matrix)
            finite = matrix[np.isfinite(matrix)]
            if finite.size:
                maximum = max(maximum, int(finite.max()))
            has_unreachable = has_unreachable or bool((~np.isfinite(matrix)).any())
        labels: list[int | str] = list(range(maximum + 1))
        if include_unreachable and has_unreachable:
            labels.append("unreachable")
        return cls(tuple(labels))

    def index(self, value: float | int | str) -> int:
        label: int | str
        if isinstance(value, str):
            label = value
        elif np.isfinite(value):
            label = int(value)
        else:
            label = "unreachable"
        return self.labels.index(label)


def distance_event_contributions(
    q: Any,
    distances: Sequence[float],
    axis: DistanceAxis,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact score mass ``[event,L,H,D]`` and support ``[event,D]``."""

    magnitude = q.square().sum(dim=-1).sqrt().detach().cpu().numpy()  # [E,L,H,N]
    distance = np.asarray(distances)
    if magnitude.shape[-1] != len(distance):
        raise ValueError("carrier distances do not align with q")
    contribution = np.zeros(magnitude.shape[:-1] + (len(axis.labels),), dtype=np.float64)
    support = np.zeros((magnitude.shape[0], len(axis.labels)), dtype=np.float64)
    for carrier, value in enumerate(distance):
        bucket = axis.index(value)
        contribution[..., bucket] += magnitude[..., carrier]
        support[:, bucket] += 1.0
    return contribution, support


def aggregate_distance_events(
    contribution: np.ndarray,
    support: np.ndarray,
    graph_ids: Sequence[int],
    source_ids: Sequence[int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Average donor events within source then sources within graph."""

    contribution = np.asarray(contribution, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64)
    graph_ids = np.asarray(graph_ids)
    source_ids = np.asarray(source_ids)
    graph_c: dict[int, np.ndarray] = {}
    graph_o: dict[int, np.ndarray] = {}
    for graph in np.unique(graph_ids):
        source_c: list[np.ndarray] = []
        source_o: list[np.ndarray] = []
        graph_mask = graph_ids == graph
        for source in np.unique(source_ids[graph_mask]):
            mask = graph_mask & (source_ids == source)
            source_c.append(contribution[mask].mean(axis=0))
            source_o.append(support[mask].mean(axis=0))
        graph_c[int(graph)] = np.stack(source_c).mean(axis=0)
        graph_o[int(graph)] = np.stack(source_o).mean(axis=0)
    return graph_c, graph_o


@dataclass(frozen=True)
class Heatmaps:
    exact: np.ndarray
    per_opportunity: np.ndarray
    exact_graph: Mapping[int, np.ndarray]
    support_graph: Mapping[int, np.ndarray]


def score_heatmaps(
    graph_contribution: Mapping[int, np.ndarray],
    graph_support: Mapping[int, np.ndarray],
    *,
    reconstruction_tolerance: float,
    graph_scores: Mapping[int, np.ndarray] | None = None,
) -> Heatmaps:
    """Compute H and R, dividing by support inside each graph before averaging."""

    keys = sorted(graph_contribution)
    if keys != sorted(graph_support):
        raise ValueError("distance contribution/support graph IDs differ")
    exact_graph = {key: np.asarray(graph_contribution[key]) for key in keys}
    support_graph = {key: np.asarray(graph_support[key]) for key in keys}
    if graph_scores is not None:
        for key in keys:
            reconstructed = exact_graph[key].sum(axis=-1)
            if not np.allclose(
                reconstructed,
                np.asarray(graph_scores[key]),
                atol=float(reconstruction_tolerance),
                rtol=0.0,
            ):
                raise RuntimeError(f"distance buckets do not reconstruct graph score {key}")
    # C is [L,H,D]; sum heads only after equal-graph averaging.
    exact = np.stack([exact_graph[key] for key in keys]).mean(axis=0).sum(axis=1)
    ratios: list[np.ndarray] = []
    for key in keys:
        C = exact_graph[key]
        O = support_graph[key]
        ratio = np.full_like(C, np.nan, dtype=np.float64)
        np.divide(C, O[None, None, :], out=ratio, where=O[None, None, :] > 0)
        ratios.append(ratio)
    stacked = np.stack(ratios)
    mean_ratio = np.nanmean(stacked, axis=0)
    per_opportunity = np.nansum(mean_ratio, axis=1)
    per_opportunity[~np.isfinite(mean_ratio).any(axis=1)] = np.nan
    return Heatmaps(exact, per_opportunity, exact_graph, support_graph)


def adaptive_distance_bins(maximum: int) -> tuple[tuple[int, int], ...]:
    bins = [(0, 0), (1, 1), (2, 2), (3, 3)]
    lower = 4
    while lower <= int(maximum):
        upper = min(2 * lower - 1, int(maximum))
        bins.append((lower, upper))
        lower *= 2
    return tuple(bins)


def aggregate_pair_field_by_distance(
    field: np.ndarray,
    distances: np.ndarray,
    *,
    bins: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Within-graph pair means and additive masses for an ``[carrier,source]`` field."""

    field = np.asarray(field, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    if field.shape != distances.shape:
        raise ValueError("field and carrier-source distance matrix must align")
    means, masses = [], []
    for lower, upper in bins:
        mask = np.isfinite(distances) & (distances >= lower) & (distances <= upper)
        means.append(float(np.mean(field[mask])) if mask.any() else np.nan)
        masses.append(float(np.sum(field[mask])) if mask.any() else np.nan)
    return np.asarray(means), np.asarray(masses)
