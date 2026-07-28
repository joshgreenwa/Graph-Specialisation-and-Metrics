"""Correctness checks for the range-comparison harness.

Run with ``python3 -m pytest test_rangelib.py -q`` from this directory.
"""

from __future__ import annotations

import numpy as np
import torch

import rangelib as R
from graph_specialisation_metrics.methodology.bootstrap import (
    Observation,
    nested_percentile_interval,
)


def _fixture(nodes: int = 13, channels: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = R.make_graphs(3, nodes=nodes, channels=channels, rng=rng)
    donors = R.make_graphs(4, nodes=nodes, channels=channels, rng=rng)
    pool = R.SemanticDonorPool([(10_000 + i, g) for i, g in enumerate(donors)])
    return base, pool


def test_identity_readout_forms_agree():
    """The compact per-carrier identity readout equals the literal T = n*d readout gradient."""

    nodes, channels = 9, 3
    delta = torch.randn(2, 4, nodes, channels, dtype=R.DTYPE)
    compact = R.functional_carriage_events(delta, R.identity_readout_gradient(channels, nodes))
    full = R.functional_carriage_events(delta, R.full_identity_readout_gradient(channels, nodes))
    assert torch.allclose(compact, full, atol=1e-12)


def test_event_normalised_carriage_recovers_influence_exactly():
    """For a linear operator, every single donor event reproduces |L| -- no Monte Carlo error."""

    base, pool = _fixture()
    graph = base[0]
    distances = R.shortest_path_distances(graph.edge_index, graph.num_nodes)
    for task in ("dirac", "rectangle", "power"):
        matrix = R.operator_matrix(task, 2, distances, graph.edge_index)
        events = R.carriage_events(
            graph,
            R.linear_operator_fn(matrix),
            pool,
            graph_id=0,
            donors=3,
            rng=np.random.default_rng(1),
        )
        assert np.abs(events.normalised_field - np.abs(matrix)).max() < 1e-12
        # Exact per event, not merely on average.
        per_event = events.magnitude / events.dose[:, :, None]
        assert np.abs(per_event - np.abs(matrix).T[:, None, :]).max() < 1e-12


def test_carriage_range_matches_jacobian_range():
    base, pool = _fixture()
    graph = base[0]
    distances = R.shortest_path_distances(graph.edge_index, graph.num_nodes)
    matrix = R.operator_matrix("rectangle", 3, distances, graph.edge_index)
    fn = R.linear_operator_fn(matrix)
    jacobian = R.normalised_range_from_influence(R.influence_matrix_autograd(fn, graph.x), distances)
    events = R.carriage_events(graph, fn, pool, graph_id=0, donors=4, rng=np.random.default_rng(2))
    carriage = R.carriage_range_per_carrier(events.normalised_field, distances, events.sources)
    assert np.nanmax(np.abs(jacobian - carriage)) < 1e-12


def test_autograd_and_analytic_influence_agree():
    base, _ = _fixture()
    graph = base[0]
    distances = R.shortest_path_distances(graph.edge_index, graph.num_nodes)
    matrix = R.operator_matrix("power", 3, distances, graph.edge_index)
    fn = R.linear_operator_fn(matrix)
    analytic = R.influence_matrix_analytic(matrix, int(graph.x.shape[1]))
    assert np.abs(analytic - R.influence_matrix_autograd(fn, graph.x)).max() < 1e-12


def test_dirac_range_on_a_path_is_exactly_k():
    """Analytic ground truth: every node of a path sees its k-hop set at distance exactly k."""

    rng = np.random.default_rng(3)
    graph = R.make_graphs(1, nodes=21, channels=2, rng=rng)[0]
    distances = R.shortest_path_distances(graph.edge_index, graph.num_nodes)
    for k in range(1, 8):
        matrix = R.operator_matrix("dirac", k, distances, graph.edge_index)
        influence = R.influence_matrix_analytic(matrix, 2)
        assert abs(R.safe_nanmean(R.normalised_range_from_influence(influence, distances)) - k) < 1e-12


def test_batched_and_looped_application_agree():
    base, pool = _fixture()
    graph = base[0]
    distances = R.shortest_path_distances(graph.edge_index, graph.num_nodes)
    matrix = R.operator_matrix("rectangle", 2, distances, graph.edge_index)
    tensor = torch.tensor(matrix, dtype=R.DTYPE)
    stacked = torch.randn(5, graph.num_nodes, int(graph.x.shape[1]), dtype=R.DTYPE)
    looped = torch.stack([tensor @ row for row in stacked])
    assert torch.allclose(tensor @ stacked, looped, atol=1e-13)


def test_local_bootstrap_matches_production_estimator():
    """The vectorised replicate loop reproduces the production nested percentile interval."""

    rng = np.random.default_rng(5)
    graphs, sources, donors = 12, 6, 4
    values = rng.standard_normal((graphs, sources, donors))
    policy = R.BootstrapPolicy()

    observations = [
        Observation(seed=0, graph=g, source=s, donor=k, value=np.asarray([values[g, s, k]]))
        for g in range(graphs)
        for s in range(sources)
        for k in range(donors)
    ]
    production = nested_percentile_interval(
        observations,
        policy,
        graph_reduce=lambda rows: R.trimmed_mean(rows, proportion=policy.trim_fraction),
    )

    # Same hierarchy, same RNG seed, same estimator, implemented over dense arrays.
    estimate = R.trimmed_mean(values.mean(axis=2).mean(axis=1), proportion=policy.trim_fraction)
    assert abs(float(np.asarray(production.estimate).ravel()[0]) - float(estimate)) < 1e-12

    local = R.bootstrap_graph_values(values.mean(axis=2).mean(axis=1))
    # The graph-only interval must sit inside the fully nested one (fewer resampled levels).
    assert local.low >= float(np.asarray(production.low).ravel()[0]) - 0.25
    assert local.high <= float(np.asarray(production.high).ravel()[0]) + 0.25


def test_bootstrap_interval_brackets_the_estimate():
    base, pool = _fixture(nodes=11, channels=2, seed=7)
    per_graph = []
    for index, graph in enumerate(base):
        distances = R.shortest_path_distances(graph.edge_index, graph.num_nodes)
        matrix = R.operator_matrix("dirac", 2, distances, graph.edge_index)
        events = R.carriage_events(
            graph,
            R.linear_operator_fn(matrix),
            pool,
            graph_id=index,
            donors=3,
            rng=np.random.default_rng(index),
        )
        per_graph.append(
            {
                "magnitude": events.magnitude,
                "dose": events.dose,
                "distances": distances,
                "sources": events.sources,
            }
        )
    interval = R.bootstrap_carriage_range(per_graph, normalise=True)
    assert interval.replicates == R.BOOTSTRAP_REPLICATES
    assert interval.low <= interval.estimate <= interval.high
    assert abs(interval.estimate - 2.0) < 1e-9
    assert interval.resampled_levels == ("graph", "donor")


def test_source_resampling_is_task_dependent():
    """Why the source level is held fixed, stated at the right strength.

    The methodology's enumeration exception is sufficient on its own.  There is also a practical
    reason, but it is task-dependent, not a general law: on the k-Power family, whose per-carrier
    mass concentrates on a few sources, resampling an enumerated source set shifts the interval
    clear of the point estimate; on k-Dirac and k-Rectangle it does not.
    """

    def per_graph_for(task: str, k: int):
        base, pool = _fixture(nodes=15, channels=2, seed=11)
        rows = []
        for index, graph in enumerate(base):
            distances = R.shortest_path_distances(graph.edge_index, graph.num_nodes)
            matrix = R.operator_matrix(task, k, distances, graph.edge_index)
            events = R.carriage_events(
                graph,
                R.linear_operator_fn(matrix),
                pool,
                graph_id=index,
                donors=3,
                rng=np.random.default_rng(index),
            )
            rows.append(
                {
                    "magnitude": events.magnitude,
                    "dose": events.dose,
                    "distances": distances,
                    "sources": events.sources,
                }
            )
        return rows

    naive_policy = R.BootstrapPolicy(resample_source=True)
    for task in ("dirac", "rectangle", "power"):
        rows = per_graph_for(task, 3)
        registered = R.bootstrap_carriage_range(rows, normalise=True)
        assert registered.low <= registered.estimate <= registered.high
        naive = R.bootstrap_carriage_range(rows, normalise=True, policy=naive_policy)
        if task == "power":
            assert naive.low > naive.estimate  # the shift the exception avoids
        else:
            assert naive.low <= naive.estimate <= naive.high  # no shift here -- not a general law


def test_fast_and_dense_bootstrap_paths_agree():
    """The collapsed [carrier, donor] path must reproduce the dense event-tensor path exactly."""

    base, pool = _fixture(nodes=13, channels=3, seed=13)
    per_graph = []
    for index, graph in enumerate(base):
        distances = R.shortest_path_distances(graph.edge_index, graph.num_nodes)
        matrix = R.operator_matrix("rectangle", 2, distances, graph.edge_index)
        events = R.carriage_events(
            graph,
            R.linear_operator_fn(matrix),
            pool,
            graph_id=index,
            donors=5,
            rng=np.random.default_rng(index),
        )
        per_graph.append(
            {
                "magnitude": events.magnitude,
                "dose": events.dose,
                "distances": distances,
                "sources": events.sources,
            }
        )
    for normalise in (True, False):
        for entry in per_graph:
            sources, donors = entry["magnitude"].shape[:2]
            dense = R._graph_range(
                entry["magnitude"],
                entry["dose"],
                entry["distances"],
                entry["sources"],
                normalise=normalise,
            )
            num, den = R._ratio_terms(entry, normalise=normalise)
            uniform = R._donor_multiplicities(
                sources, donors, np.random.default_rng(0), resample=False
            )
            assert abs(dense - R._range_from_terms(num, den, uniform)) < 1e-12

            # And under a donor resample: the fast path must reproduce the dense path event for
            # event, with donors drawn independently *within* each source.
            for seed in range(4):
                picks = np.random.default_rng(seed).integers(0, donors, size=(sources, donors))
                counts = np.zeros((sources, donors))
                np.add.at(counts, (np.arange(sources)[:, None], picks), 1.0)
                fast = R._range_from_terms(num, den, (counts / donors).reshape(-1))
                slow = R._graph_range(
                    entry["magnitude"],
                    entry["dose"],
                    entry["distances"],
                    entry["sources"],
                    normalise=normalise,
                    donor_index=picks,
                )
                assert abs(fast - slow) < 1e-12


def test_gradient_free_probe_recovers_a_known_linear_range():
    """The probe must weight each distance by its shell total, not its per-node mean.

    On ``F(X)_u = a x_u + b mean_{v at exactly k hops} x_v`` the normalised range is exactly
    ``k b / (a + b)``.  Averaging within a shell instead of summing silently reweights every
    distance by ``1/|shell|`` and under-reports by ~17% here.
    """

    import run_divergence as D

    rng = np.random.default_rng(1)
    base = R.make_graphs(6, nodes=25, channels=1, rng=rng)
    distances = R.shortest_path_distances(base[0].edge_index, base[0].num_nodes)
    local, far, k = 1.0, 2.0, 4
    fn = D.make_task("linear", 0.0, D.far_mask(distances, k), local=local, far=far)

    analytic = k * far / (local + far)
    jacobian = R.trimmed_mean(
        np.asarray(
            [
                R.safe_nanmean(
                    R.normalised_range_from_influence(
                        R.influence_matrix_autograd(fn, graph.x), distances
                    )
                )
                for graph in base
            ]
        )
    )
    assert abs(float(jacobian) - analytic) < 1e-9

    probe = R.trimmed_mean(
        np.asarray(
            [
                D.shell_resample_range(
                    fn, graph.x, distances, repeats=64, rng=np.random.default_rng(index)
                )
                for index, graph in enumerate(base)
            ]
        )
    )
    assert abs(float(probe) - analytic) < 0.10 * analytic
