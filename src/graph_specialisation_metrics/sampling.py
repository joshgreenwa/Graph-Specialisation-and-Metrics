"""Semantic and structural donor selection."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SemanticDonor:
    graph_id: Hashable
    node: int
    degree: int
    attributes: Any


class SemanticDonorPool:
    """Semantic donor selection with minimum degree matching."""

    def __init__(
        self,
        attributes: Mapping[Hashable, Any],
        degrees: Mapping[Hashable, Sequence[int]],
    ) -> None:
        if not attributes:
            raise ValueError("semantic donor pool is empty")
        if set(attributes) != set(degrees):
            raise ValueError("attribute and degree mappings must contain the same graphs")
        self._graphs: dict[Hashable, tuple[SemanticDonor, ...]] = {}
        for graph_id, graph_attributes in attributes.items():
            rows = np.asarray(graph_attributes)
            graph_degrees = np.asarray(degrees[graph_id], dtype=np.int64)
            if rows.ndim == 0 or rows.shape[0] != len(graph_degrees):
                raise ValueError(f"attributes and degrees do not align for graph {graph_id!r}")
            self._graphs[graph_id] = tuple(
                SemanticDonor(graph_id, node, int(graph_degrees[node]), rows[node].copy())
                for node in range(rows.shape[0])
            )

    def eligible(
        self,
        source_attributes: Any,
        source_degree: int,
        *,
        source_graph: Hashable | None = None,
    ) -> dict[Hashable, tuple[SemanticDonor, ...]]:
        """Return eligible donors with minimum degree difference."""

        candidates = [
            donor
            for graph_id, graph in self._graphs.items()
            if source_graph is None or graph_id != source_graph
            for donor in graph
            if not np.array_equal(np.asarray(donor.attributes), np.asarray(source_attributes))
        ]
        if not candidates:
            return {}
        minimum_gap = min(abs(donor.degree - int(source_degree)) for donor in candidates)
        result: dict[Hashable, list[SemanticDonor]] = {}
        for donor in candidates:
            if abs(donor.degree - int(source_degree)) == minimum_gap:
                result.setdefault(donor.graph_id, []).append(donor)
        return {graph_id: tuple(graph) for graph_id, graph in result.items()}

    def sample(
        self,
        source_attributes: Any,
        source_degree: int,
        count: int,
        rng: np.random.Generator,
        *,
        source_graph: Hashable | None = None,
    ) -> tuple[SemanticDonor, ...]:
        """Draw independently with replacement: graph uniformly, then node uniformly."""

        if count < 0:
            raise ValueError("count must be non-negative")
        eligible = self.eligible(source_attributes, source_degree, source_graph=source_graph)
        if not eligible:
            if count == 0:
                return ()
            raise ValueError("semantic source has no eligible donor")
        graph_ids = tuple(eligible)
        draws: list[SemanticDonor] = []
        for _ in range(int(count)):
            graph_id = graph_ids[int(rng.integers(len(graph_ids)))]
            graph = eligible[graph_id]
            draws.append(graph[int(rng.integers(len(graph)))])
        return tuple(draws)


def analysis_indices(
    evaluation_length: int,
    evaluation_count: int,
    semantic_pool_length: int,
    semantic_pool_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select evaluated graphs and semantic donor graphs."""

    if (
        min(
            evaluation_length,
            semantic_pool_length,
            evaluation_count,
            semantic_pool_count,
        )
        < 1
    ):
        raise ValueError("dataset lengths and sample counts must be positive")
    if int(evaluation_count) > int(evaluation_length):
        raise ValueError(
            f"requested {evaluation_count} scored graphs from a split containing "
            f"{evaluation_length}"
        )
    evaluation = np.random.default_rng(int(seed)).permutation(int(evaluation_length))[
        : int(evaluation_count)
    ]
    semantic_pool_size = min(int(semantic_pool_count), int(semantic_pool_length))
    semantic_pool = np.random.default_rng(int(seed) + 1).choice(
        int(semantic_pool_length), size=semantic_pool_size, replace=False
    )
    return (
        np.sort(evaluation.astype(np.int64, copy=False)),
        np.sort(semantic_pool.astype(np.int64, copy=False)),
    )


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def eligible_structural_donors(
    structural_profiles: Sequence[Any],
    source: int,
) -> np.ndarray:
    """Return nodes whose structural fields differ from the source."""

    source = int(source)
    if not 0 <= source < len(structural_profiles):
        raise IndexError("source is outside the graph")
    return np.asarray(
        [
            node
            for node, profile in enumerate(structural_profiles)
            if node != source and not _equal(profile, structural_profiles[source])
        ],
        dtype=np.int64,
    )


def sample_structural_donors(
    structural_profiles: Sequence[Any],
    source: int,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample structural donors without replacement."""

    if count < 0:
        raise ValueError("count must be non-negative")
    eligible = eligible_structural_donors(structural_profiles, source)
    selected = min(int(count), len(eligible))
    if selected == len(eligible):
        return eligible.copy()
    return np.asarray(rng.choice(eligible, size=selected, replace=False), dtype=np.int64)
