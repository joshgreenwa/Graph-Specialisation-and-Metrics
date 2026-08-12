"""Distance-resolved score contributions."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .scores import aggregate_donor_swap_responses, head_output_row_magnitudes


class ReconstructionError(RuntimeError):
    """Distance-resolved score contributions do not reconstruct their score."""


def shortest_path_distances(edge_index: Any, num_nodes: int) -> np.ndarray:
    """Return undirected all-pairs distances, with ``inf`` for unreachable nodes."""

    edges = (
        edge_index.detach().cpu().numpy()
        if hasattr(edge_index, "detach")
        else np.asarray(edge_index)
    )
    if edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edge]")
    node_count = int(num_nodes)
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for left, right in edges.T:
        left = int(left)
        right = int(right)
        if not (0 <= left < node_count and 0 <= right < node_count):
            raise ValueError("edge_index contains an out-of-range node")
        adjacency[left].append(right)
        adjacency[right].append(left)

    result = np.full((node_count, node_count), np.inf, dtype=np.float64)
    for source in range(node_count):
        result[source, source] = 0.0
        queue = [source]
        for node in queue:
            candidate = result[source, node] + 1.0
            for neighbour in adjacency[node]:
                if not np.isfinite(result[source, neighbour]):
                    result[source, neighbour] = candidate
                    queue.append(neighbour)
    return result


def _label(value: Any) -> int | str:
    if isinstance(value, str):
        return value
    numeric = float(value)
    if not np.isfinite(numeric):
        return "unreachable"
    if numeric < 0 or not numeric.is_integer():
        raise ValueError(f"distance must be a non-negative integer, inf, or label; got {value!r}")
    return int(numeric)


def _infer_distance_categories(distances: np.ndarray) -> tuple[int | str, ...]:
    numeric: set[int] = set()
    special: list[str] = []
    for value in distances.flat:
        label = _label(value)
        if isinstance(label, int):
            numeric.add(label)
        elif label not in special:
            special.append(label)
    return tuple(sorted(numeric)) + tuple(special)


def assert_reconstructs_scores(
    score_contributions: Any,
    scores: Any,
    *,
    atol: float = 1e-10,
) -> None:
    """Fail unless summing distance categories recovers every score."""

    contribution_array = np.asarray(score_contributions, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    if contribution_array.shape[:-1] != score_array.shape:
        raise ReconstructionError(
            "distance-resolved score contribution shape does not match scores: "
            f"{contribution_array.shape} versus {score_array.shape}"
        )
    reconstructed = contribution_array.sum(axis=-1)
    if not np.allclose(reconstructed, score_array, atol=atol, rtol=1e-10):
        maximum = float(np.max(np.abs(reconstructed - score_array)))
        raise ReconstructionError(
            "distance-resolved score contributions do not reconstruct scores "
            f"(max error {maximum:.3g})"
        )


@dataclass(frozen=True)
class DonorSwapDistanceContributions:
    """Distance-resolved score contributions for donor-swaps."""

    distance_categories: tuple[int | str, ...]
    score_contributions: np.ndarray
    donor_swap_responses: np.ndarray


def distance_resolved_score_contributions(
    projected_head_responses: Any,
    head_output_row_distances: Any,
    *,
    distance_categories: Sequence[int | str] | None = None,
    atol: float = 1e-10,
) -> DonorSwapDistanceContributions:
    """Partition each donor-swap response by distance category.

    ``head_output_row_distances`` may be ``[head_output_row]`` when every donor-swap
    shares a source or ``[donor_swap, head_output_row]`` otherwise. Graph-token and
    virtual-node rows retain their own distance categories.
    """

    magnitudes = head_output_row_magnitudes(projected_head_responses)
    if hasattr(magnitudes, "detach"):
        magnitudes = magnitudes.detach().cpu().numpy()
    magnitudes = np.asarray(magnitudes, dtype=np.float64)
    if magnitudes.ndim != 4:
        raise ValueError(
            "projected responses must have shape "
            "[donor_swap, layer, head, head_output_row, model_output]"
        )

    distances = np.asarray(head_output_row_distances, dtype=object)
    if distances.ndim == 1:
        distances = np.broadcast_to(distances, (magnitudes.shape[0], distances.shape[0]))
    if distances.shape != (magnitudes.shape[0], magnitudes.shape[-1]):
        raise ValueError("head-output-row distances must align with [donor_swap, head_output_row]")

    axis = (
        tuple(_label(value) for value in distance_categories)
        if distance_categories is not None
        else _infer_distance_categories(distances)
    )
    if not axis or len(set(axis)) != len(axis):
        raise ValueError("distance categories must be non-empty and unique")
    category_index = {category: index for index, category in enumerate(axis)}
    contributions = np.zeros(magnitudes.shape[:-1] + (len(axis),), dtype=np.float64)
    for donor_swap in range(magnitudes.shape[0]):
        for head_output_row in range(magnitudes.shape[-1]):
            category = _label(distances[donor_swap, head_output_row])
            if category not in category_index:
                raise ValueError(f"distance category {category!r} is not in distance_categories")
            contributions[donor_swap, ..., category_index[category]] += magnitudes[
                donor_swap, ..., head_output_row
            ]

    # Reuse the float64 row magnitudes to avoid float32 summation differences.
    responses = magnitudes.sum(axis=-1)
    assert_reconstructs_scores(contributions, responses, atol=atol)
    return DonorSwapDistanceContributions(axis, contributions, responses)


@dataclass(frozen=True)
class DistanceResolvedScoreContributions:
    """Aggregated distance-resolved score contributions."""

    distance_categories: tuple[int | str, ...]
    score_contributions: np.ndarray
    graph_score_contributions: dict[Hashable, np.ndarray]
    source_score_contributions: dict[tuple[Hashable, Hashable], np.ndarray]


def aggregate_distance_resolved_score_contributions(
    contributions: DonorSwapDistanceContributions,
    graph_ids: Sequence[Hashable],
    source_ids: Sequence[Hashable],
    *,
    atol: float = 1e-10,
) -> DistanceResolvedScoreContributions:
    """Aggregate score contributions over donor-swaps, sources, and graphs."""

    score_contributions, graph_contributions, source_contributions = aggregate_donor_swap_responses(
        contributions.score_contributions,
        graph_ids,
        source_ids,
    )
    scores, graph_scores, source_scores = aggregate_donor_swap_responses(
        contributions.donor_swap_responses,
        graph_ids,
        source_ids,
    )
    assert_reconstructs_scores(score_contributions, scores, atol=atol)
    for key in graph_contributions:
        assert_reconstructs_scores(graph_contributions[key], graph_scores[key], atol=atol)
    for key in source_contributions:
        assert_reconstructs_scores(source_contributions[key], source_scores[key], atol=atol)
    return DistanceResolvedScoreContributions(
        contributions.distance_categories,
        score_contributions,
        graph_contributions,
        source_contributions,
    )
