"""Fixed nested percentile bootstrap and graph-level robust summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .protocol import BOOTSTRAP_REPLICATES, BootstrapPolicy


def trimmed_mean(values: Any, proportion: float = 0.20, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.sort(values, axis=axis)
    count = int(values.shape[axis])
    trim = int(np.floor(float(proportion) * count))
    if trim == 0:
        return np.mean(values, axis=axis)
    indices = np.arange(trim, count - trim)
    return np.mean(np.take(values, indices, axis=axis), axis=axis)


@dataclass(frozen=True)
class Observation:
    seed: int
    graph: int
    source: int
    donor: int
    value: Any


@dataclass(frozen=True)
class Interval:
    estimate: np.ndarray
    low: np.ndarray
    high: np.ndarray
    replicates: int
    rng_seed: int
    resampled_levels: tuple[str, ...]
    # Draws in which each cell was estimable; below `replicates` means some replicate had no
    # support there. None for intervals recorded before this was tracked.
    estimable_draws: np.ndarray | None = None
    # Opt-in transient bootstrap distribution.  Production caches strip this after computing
    # rank/family-stability diagnostics; retaining every draw in every interval would needlessly
    # inflate the cache.
    draws: np.ndarray | None = None


def _choice(keys: Sequence[Any], rng: np.random.Generator, resample: bool) -> list[Any]:
    keys = list(keys)
    if not keys:
        raise ValueError("cannot bootstrap an empty hierarchy")
    if not resample:
        return keys
    return [keys[int(index)] for index in rng.integers(0, len(keys), size=len(keys))]


def _nested_estimate(
    observations: Sequence[Observation],
    rng: np.random.Generator | None,
    policy: BootstrapPolicy,
    graph_reduce: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    by_seed: dict[int, list[Observation]] = {}
    for row in observations:
        by_seed.setdefault(int(row.seed), []).append(row)
    seed_keys = sorted(by_seed)
    resample_seed = policy.resample_seed and len(seed_keys) >= 3 and rng is not None
    selected_seeds = _choice(seed_keys, rng, resample_seed) if rng else seed_keys
    seed_estimates: list[np.ndarray] = []
    for seed in selected_seeds:
        rows_seed = by_seed[seed]
        by_graph: dict[int, list[Observation]] = {}
        for row in rows_seed:
            by_graph.setdefault(int(row.graph), []).append(row)
        graph_keys = sorted(by_graph)
        selected_graphs = (
            _choice(graph_keys, rng, policy.resample_graph) if rng else graph_keys
        )
        graph_estimates: list[np.ndarray] = []
        for graph in selected_graphs:
            rows_graph = by_graph[graph]
            by_source: dict[int, list[Observation]] = {}
            for row in rows_graph:
                by_source.setdefault(int(row.source), []).append(row)
            source_keys = sorted(by_source)
            selected_sources = (
                _choice(source_keys, rng, policy.resample_source) if rng else source_keys
            )
            source_estimates: list[np.ndarray] = []
            for source in selected_sources:
                rows_source = by_source[source]
                donor_indices = list(range(len(rows_source)))
                selected_donors = (
                    _choice(donor_indices, rng, policy.resample_donor)
                    if rng
                    else donor_indices
                )
                source_estimates.append(
                    np.mean(
                        np.stack(
                            [
                                np.asarray(rows_source[index].value, dtype=np.float64)
                                for index in selected_donors
                            ]
                        ),
                        axis=0,
                    )
                )
            graph_estimates.append(np.mean(np.stack(source_estimates), axis=0))
        seed_estimates.append(graph_reduce(np.stack(graph_estimates)))
    return np.mean(np.stack(seed_estimates), axis=0)


def nested_percentile_interval(
    observations: Sequence[Observation],
    policy: BootstrapPolicy,
    *,
    graph_reduce: Callable[[np.ndarray], np.ndarray] | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    retain_draws: bool = False,
) -> Interval:
    """Run the complete seed->graph->source->donor bootstrap hierarchy."""

    policy.validate()
    if int(policy.replicates) != BOOTSTRAP_REPLICATES:
        raise ValueError("canonical intervals require 2,000 replicates")
    if not observations:
        raise ValueError("no observations")
    reduce = graph_reduce or (lambda rows: np.mean(rows, axis=0))
    apply = transform or (lambda value: value)
    estimate = apply(_nested_estimate(observations, None, policy, reduce))
    rng = np.random.default_rng(int(policy.rng_seed))
    draws = np.stack(
        [
            apply(_nested_estimate(observations, rng, policy, reduce))
            for _ in range(policy.replicates)
        ]
    )
    alpha = (1.0 - float(policy.confidence)) / 2.0
    low, high, estimable = _percentiles(draws, alpha)
    levels = [
        name
        for name, enabled in (
            ("seed", policy.resample_seed and len({row.seed for row in observations}) >= 3),
            ("graph", policy.resample_graph),
            ("source", policy.resample_source),
            ("donor", policy.resample_donor),
        )
        if enabled
    ]
    return Interval(
        estimate=estimate,
        low=low,
        high=high,
        replicates=int(policy.replicates),
        rng_seed=int(policy.rng_seed),
        resampled_levels=tuple(levels),
        estimable_draws=estimable,
        draws=draws if retain_draws else None,
    )


def paired_channel_percentile_interval(
    left: Sequence[Observation],
    right: Sequence[Observation],
    policy: BootstrapPolicy,
    *,
    transform: Callable[[np.ndarray], np.ndarray],
    resample_source: tuple[bool, bool] = (True, True),
    retain_draws: bool = False,
    on_draw: Callable[[int, int], None] | None = None,
) -> Interval:
    """Graph-paired bootstrap for channels with different source domains.

    The same graph draw is used for both channels, but each channel independently resamples its
    own sources and donors inside that graph. This is the correct extension for an edge-semantic
    channel paired with a node-structural channel; integer source IDs are never falsely matched.
    """

    policy.validate()
    if not left or not right:
        raise ValueError("both channels require observations")
    by_channel = []
    for observations in (left, right):
        by_graph: dict[int, list[Observation]] = {}
        for row in observations:
            by_graph.setdefault(int(row.graph), []).append(row)
        by_channel.append(by_graph)
    graphs = sorted(set(by_channel[0]) & set(by_channel[1]))
    if not graphs:
        raise ValueError("channels share no graph IDs")

    def channel_graph_estimate(
        rows: Sequence[Observation],
        rng: np.random.Generator | None,
        *,
        resample_sources: bool,
    ) -> np.ndarray:
        by_source: dict[int, list[Observation]] = {}
        for row in rows:
            by_source.setdefault(int(row.source), []).append(row)
        source_keys = sorted(by_source)
        selected_sources = (
            _choice(source_keys, rng, resample_sources)
            if rng is not None
            else source_keys
        )
        values = []
        for source in selected_sources:
            source_rows = by_source[source]
            donor_indices = list(range(len(source_rows)))
            selected_donors = (
                _choice(donor_indices, rng, policy.resample_donor)
                if rng is not None
                else donor_indices
            )
            values.append(
                np.mean(
                    np.stack(
                        [
                            np.asarray(source_rows[index].value, dtype=np.float64)
                            for index in selected_donors
                        ]
                    ),
                    axis=0,
                )
            )
        return np.mean(np.stack(values), axis=0)

    def estimate(rng: np.random.Generator | None) -> np.ndarray:
        selected_graphs = (
            _choice(graphs, rng, policy.resample_graph)
            if rng is not None
            else graphs
        )
        channel_values = []
        for channel in range(2):
            graph_values = [
                channel_graph_estimate(
                    by_channel[channel][graph],
                    rng,
                    resample_sources=(
                        bool(resample_source[channel]) and policy.resample_source
                    ),
                )
                for graph in selected_graphs
            ]
            channel_values.append(np.mean(np.stack(graph_values), axis=0))
        return transform(np.stack(channel_values))

    point = estimate(None)
    rng = np.random.default_rng(int(policy.rng_seed))
    draw_values = []
    for draw in range(int(policy.replicates)):
        draw_values.append(estimate(rng))
        if on_draw is not None:
            on_draw(draw + 1, int(policy.replicates))
    draws = np.stack(draw_values)
    alpha = (1.0 - float(policy.confidence)) / 2.0
    low, high, estimable = _percentiles(draws, alpha)
    levels = ["graph"]
    if any(resample_source) and policy.resample_source:
        levels.append("source(channel-independent)")
    if policy.resample_donor:
        levels.append("donor(channel-independent)")
    return Interval(
        estimate=point,
        low=low,
        high=high,
        replicates=int(policy.replicates),
        rng_seed=int(policy.rng_seed),
        resampled_levels=tuple(levels),
        estimable_draws=estimable,
        draws=draws if retain_draws else None,
    )


def _percentiles(
    draws: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentiles over the estimable draws of each cell.

    A replicate in which a cell has no support is missing, not zero: including it would drag the
    band toward zero and contradict the point estimator, which reports that cell as non-estimable.
    """

    draws = np.asarray(draws, dtype=np.float64)
    estimable = np.isfinite(draws).sum(axis=0)
    empty = estimable == 0
    # Fill fully unsupported cells so np.nanquantile never sees an all-nan slice, then restore nan.
    filled = np.where(np.broadcast_to(empty, draws.shape), 0.0, draws)
    low = np.nanquantile(filled, alpha, axis=0)
    high = np.nanquantile(filled, 1.0 - alpha, axis=0)
    return (
        np.where(empty, np.nan, low),
        np.where(empty, np.nan, high),
        estimable,
    )


def reportable_bin(
    graph_ids: Sequence[int],
    pair_count: int,
    *,
    policy: BootstrapPolicy,
) -> bool:
    return (
        len(set(int(value) for value in graph_ids)) >= int(policy.minimum_graphs)
        and int(pair_count) >= int(policy.minimum_pairs)
    )
