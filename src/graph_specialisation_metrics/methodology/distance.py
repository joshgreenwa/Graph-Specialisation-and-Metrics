"""Exact shortest-path accounting for scores and carriage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .audit import audit_check


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
    # Preserve mixed numeric/special labels (e.g. ``virtual`` or ``graph_token``).
    # NumPy's default coercion would turn every numeric distance into a string.
    distance = np.asarray(distances, dtype=object)
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


def supported_mean(values: Any, axis: int = 0) -> np.ndarray:
    """Mean over supported entries, ``nan`` where nothing is supported.

    Numerically identical to ``np.nanmean`` but silent: a distance column with no carrier
    opportunity anywhere in the slice is an expected outcome of the frozen distance axis, not a
    numerical defect, so it must not raise ``Mean of empty slice``.
    """

    values = np.asarray(values, dtype=np.float64)
    supported = ~np.isnan(values)
    count = supported.sum(axis=axis)
    total = np.where(supported, values, 0.0).sum(axis=axis)
    result = np.full(np.shape(total), np.nan, dtype=np.float64)
    np.divide(total, count, out=result, where=count > 0)
    return result


def per_opportunity_from_ratio(mean_ratio: np.ndarray) -> np.ndarray:
    """Sum support-normalized cells over heads, keeping unsupported columns non-estimable."""

    mean_ratio = np.asarray(mean_ratio, dtype=np.float64)
    per_opportunity = np.nansum(mean_ratio, axis=1)
    per_opportunity[~np.isfinite(mean_ratio).any(axis=1)] = np.nan
    return per_opportunity


def row_normalised(values: Any) -> np.ndarray:
    """Divide each head's distance profile by that head's own total over the distance axis.

    The denominator is the whole registered axis, so blanking a column at presentation time can
    never inflate the columns that survive; a displayed row then sums to at most one. A row with no
    mass anywhere stays ``nan`` rather than becoming a spurious uniform profile.
    """

    values = np.asarray(values, dtype=np.float64)
    total = np.nansum(values, axis=-1, keepdims=True)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(values, total, out=result, where=np.isfinite(values) & (total > 0))
    return result


@dataclass(frozen=True)
class DisplayAxis:
    """Contiguous grouping of the registered distance axis, for readable figures only.

    Cached measurements always keep unit resolution; this is applied at presentation time, so a
    figure never has more columns than a reader can label. Groups are contiguous and exhaustive,
    which makes mass strictly additive within a group and support-normalized quantities exactly
    recomputable as summed contribution over summed support.
    """

    labels: tuple[str, ...]
    groups: tuple[tuple[int, ...], ...]

    @property
    def source_width(self) -> int:
        return sum(len(group) for group in self.groups)

    @property
    def identity(self) -> bool:
        """True when every registered column is its own group, so binning is a no-op."""

        return all(len(group) == 1 for group in self.groups)

    @property
    def widths(self) -> np.ndarray:
        return np.asarray([len(group) for group in self.groups], dtype=np.float64)

    def group_density(self, values: Any) -> np.ndarray:
        """Group an additive quantity and report it per unit distance.

        Summed mass is width-confounded: a ten-wide tail group accumulates ten columns and draws
        as a resurgence that is not in the data. Dividing by the group width restores exactly the
        reading the ungrouped figure had, and is the identity when every group is one column.
        """

        return self.group_sum(values) / self.widths

    def group_sum(self, values: Any) -> np.ndarray:
        """Sum an additive quantity along its trailing distance axis, within each group."""

        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != self.source_width:
            raise ValueError(
                f"display axis covers {self.source_width} columns, got {values.shape[-1]}"
            )
        return np.stack(
            [values[..., list(group)].sum(axis=-1) for group in self.groups], axis=-1
        )

    def group_ratio(self, contribution: Any, support: Any) -> np.ndarray:
        """Support-normalize after grouping; a group with no opportunity stays non-estimable."""

        grouped_c = self.group_sum(contribution)
        grouped_o = self.group_sum(support)
        result = np.full(grouped_c.shape, np.nan, dtype=np.float64)
        np.divide(grouped_c, grouped_o, out=result, where=grouped_o > 0)
        return result


def _dyadic_groups(start: int, stop: int) -> list[tuple[int, ...]]:
    """Doubling-width contiguous position ranges covering ``[start, stop)``."""

    groups: list[tuple[int, ...]] = []
    lower = max(1, int(start))
    while lower < int(stop):
        upper = min(2 * lower, int(stop))
        groups.append(tuple(range(lower, upper)))
        lower = upper
    return groups


def display_bins(
    labels: Sequence[int | str], *, max_points: int = 14
) -> DisplayAxis:
    """Group a registered distance axis into at most ``max_points`` readable columns.

    Near distances keep unit resolution and the tail widens dyadically, with the unit prefix taken
    as long as the budget allows: the near field is where the response lives, and the tail is where
    per-column support collapses. Explicit non-numeric columns (``unreachable``, ``virtual``) are
    never merged into a numeric range.
    """

    labels = tuple(labels)
    if int(max_points) < 1:
        raise ValueError("max_points must be positive")
    numeric = [index for index, label in enumerate(labels) if not isinstance(label, str)]
    special = [index for index, label in enumerate(labels) if isinstance(label, str)]
    budget = max(1, int(max_points) - len(special))
    positions: list[tuple[int, ...]]
    if len(numeric) <= budget:
        positions = [(index,) for index in numeric]
    else:
        positions = [(index,) for index in numeric]
        for unit in range(budget, 0, -1):
            candidate = [(numeric[offset],) for offset in range(unit)] + [
                tuple(numeric[offset] for offset in group)
                for group in _dyadic_groups(unit, len(numeric))
            ]
            if len(candidate) <= budget:
                positions = candidate
                break
    groups = positions + [(index,) for index in special]
    return DisplayAxis(
        labels=tuple(_group_label(labels, group) for group in groups),
        groups=tuple(groups),
    )


def _group_label(labels: Sequence[int | str], group: Sequence[int]) -> str:
    first, last = labels[group[0]], labels[group[-1]]
    return str(first) if first == last else f"{first}-{last}"


def distance_profile_reduce(rows: np.ndarray) -> np.ndarray:
    """Graph-level reducer for distance-resolved score intervals.

    Takes ``[graph, 2, layer, head, distance]`` holding contribution and broadcast support, and
    returns ``[2, layer + 1, distance]`` whose last row is the layer-summed profile. Support is
    divided within graph before graph averaging, and a column unsupported in every graph of the
    slice stays non-estimable rather than being folded to zero.
    """

    contribution = rows[:, 0]
    support = rows[:, 1]
    ratio = np.full_like(contribution, np.nan)
    np.divide(contribution, support, out=ratio, where=support > 0)
    cells_c = contribution.mean(axis=0).sum(axis=1)
    cells_r = per_opportunity_from_ratio(supported_mean(ratio, axis=0))
    return np.stack(
        (
            np.concatenate((cells_c, cells_c.sum(axis=0, keepdims=True)), axis=0),
            np.concatenate((cells_r, cells_r.sum(axis=0, keepdims=True)), axis=0),
        )
    )


def column_support(
    graph_support: Mapping[int, np.ndarray],
    sources_per_graph: Mapping[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return supporting graph count and eligible (graph, source, carrier) pairs per column.

    ``graph_support`` holds the donor- then source-averaged carrier count per column, so
    multiplying by the graph's estimable source count recovers the exact pair total.
    """

    keys = sorted(graph_support)
    stacked = np.stack([np.asarray(graph_support[key], dtype=np.float64) for key in keys])
    graphs = (stacked > 0).sum(axis=0).astype(np.int64)
    weights = np.asarray(
        [int(sources_per_graph[key]) for key in keys], dtype=np.float64
    ).reshape(-1, 1)
    pairs = np.rint((stacked * weights).sum(axis=0)).astype(np.int64)
    return graphs, pairs


@dataclass(frozen=True)
class Heatmaps:
    # `exact`/`per_opportunity` are the head-summed [L,D] matrices; the `*_head` [L,H,D] arrays
    # they are summed from are what the heatmap figures display, so head diversity survives.
    exact: np.ndarray
    per_opportunity: np.ndarray
    exact_head: np.ndarray
    per_opportunity_head: np.ndarray
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
            residual = float(
                np.max(np.abs(reconstructed - np.asarray(graph_scores[key])))
            )
            audit_check(
                residual <= float(reconstruction_tolerance),
                "distance.bucket_reconstruction",
                f"distance buckets do not reconstruct graph score {key} "
                f"(max residual {residual:.3e} exceeds "
                f"{float(reconstruction_tolerance):.3e})",
                observed=residual,
                tolerance=float(reconstruction_tolerance),
                context={"graph": int(key)},
            )
    # C is [L,H,D]; average graphs equally, and sum heads only for the aggregate view.
    exact_head = np.stack([exact_graph[key] for key in keys]).mean(axis=0)
    exact = exact_head.sum(axis=1)
    ratios: list[np.ndarray] = []
    for key in keys:
        C = exact_graph[key]
        O = support_graph[key]
        ratio = np.full_like(C, np.nan, dtype=np.float64)
        np.divide(C, O[None, None, :], out=ratio, where=O[None, None, :] > 0)
        ratios.append(ratio)
    mean_ratio = supported_mean(np.stack(ratios), axis=0)
    per_opportunity = per_opportunity_from_ratio(mean_ratio)
    return Heatmaps(
        exact=exact,
        per_opportunity=per_opportunity,
        exact_head=exact_head,
        per_opportunity_head=mean_ratio,
        exact_graph=exact_graph,
        support_graph=support_graph,
    )


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
