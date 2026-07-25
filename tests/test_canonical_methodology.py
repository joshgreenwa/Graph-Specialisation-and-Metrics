from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from graph_specialisation_metrics import main as public_main
from graph_specialisation_metrics.methodology.bootstrap import (
    Observation,
    nested_percentile_interval,
    trimmed_mean,
)
from graph_specialisation_metrics.methodology.cache import (
    CacheContract,
    CanonicalCache,
    StaleCacheError,
)
from graph_specialisation_metrics.methodology.carriage import (
    beneficial_carriage,
    functional_carriage,
)
from graph_specialisation_metrics.methodology.causal import (
    donor_necessity,
    mismatch_adjusted_gross,
    patch_response,
)
from graph_specialisation_metrics.methodology.distance import (
    DistanceAxis,
    aggregate_distance_events,
    distance_event_contributions,
    score_heatmaps,
)
from graph_specialisation_metrics.methodology.figures import (
    HeadPlotData,
    attention_distance_profiles,
    causal_family_panels,
    causal_scatter_grid,
    cumulative_prefix_curves,
    joint_selectivity_plane,
    score_plane,
)
from graph_specialisation_metrics.methodology.interventions import (
    StructuralAuditError,
    coalesce_equal_sparse,
    semantic_donor_swap,
    structural_donor_swap,
)
from graph_specialisation_metrics.methodology.protocol import (
    BOOTSTRAP_REPLICATES,
    BootstrapPolicy,
    MethodologyConfig,
    RunSizes,
    deterministic_splits,
)
from graph_specialisation_metrics.methodology.sampling import (
    SemanticDonorPool,
    draw_structural_donors,
)
from graph_specialisation_metrics.methodology.scores import (
    JOINT_AXIS_LABEL,
    SELECTIVITY_AXIS_LABEL,
    SEMANTIC_AXIS_LABEL,
    STRUCTURAL_AXIS_LABEL,
    aggregate_event_scores,
    event_head_scores,
    head_coordinates,
    project_transport,
)
from graph_specialisation_metrics.methodology.tasks import TASKS, OutputGeometry, get_task


class FakeData:
    def __init__(self, **values):
        self.__dict__.update(values)

    def clone(self):
        return deepcopy(self)

    @property
    def keys(self):
        return tuple(self.__dict__)


def graph(rows, edges):
    x = torch.as_tensor(rows, dtype=torch.long)
    edge_index = torch.as_tensor(edges, dtype=torch.long).t().contiguous()
    return FakeData(x=x, edge_index=edge_index, num_nodes=len(rows))


def structural_graph():
    n = 3
    rrwp_index = torch.cartesian_prod(torch.arange(n), torch.arange(n)).t()
    rrwp_val = (
        torch.arange(n * n, dtype=torch.float32).reshape(n * n, 1)
    )
    return FakeData(
        x=torch.tensor([[1], [2], [3]], dtype=torch.long),
        y=torch.tensor([[0.5]]),
        num_nodes=n,
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.ones(4, 1),
        rrwp_local_edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        rrwp=torch.tensor([[0.0], [1.0], [2.0]]),
        deg=torch.tensor([1, 2, 1]),
        rrwp_index=rrwp_index,
        rrwp_val=rrwp_val,
    )


def test_protocol_constants_and_disjoint_splits():
    assert callable(public_main)
    config = MethodologyConfig()
    config.validate()
    assert config.bootstrap.replicates == BOOTSTRAP_REPLICATES == 2_000
    sizes = RunSizes(
        discovery_graphs=3,
        causal_graphs=2,
        clean_ablation_graphs=2,
        semantic_donor_graphs=4,
        sources_per_graph=1,
        donors_per_source=2,
    )
    split = deterministic_splits(20, 20, sizes, 4, same_index_space=True)
    groups = [
        set(split.discovery),
        set(split.causal),
        set(split.clean_ablation),
        set(split.semantic_donor_pool),
    ]
    assert all(not groups[i] & groups[j] for i in range(4) for j in range(i))


def test_registered_grit_geometry_covers_dense_local_khop_and_vnode():
    expected = {
        "zinc",
        "zinc_1hop",
        "zinc_2hop",
        "zinc_1hop_vnode",
        "zinc_2hop_vnode",
    }
    assert expected <= set(TASKS)
    assert not TASKS["zinc"].virtual_node
    assert not TASKS["zinc_2hop"].virtual_node
    assert TASKS["zinc_1hop_vnode"].virtual_node
    assert TASKS["zinc_2hop_vnode"].carrier_policy == (
        "real_nodes_plus_internal_vnode"
    )


def test_output_geometry_uses_fixed_z_space_and_training_only_std():
    regression = OutputGeometry(
        "evaluation_regression", None, "training_target_std"
    )
    targets = np.asarray([[1.0, 10.0], [3.0, 14.0]])
    sigma = regression.resolve(2, training_targets=targets)
    assert np.allclose(sigma, [1.0, 2.0])
    assert np.allclose(
        regression.transform(np.asarray([[2.0, 8.0]]), sigma),
        [[2.0, 4.0]],
    )
    logits = OutputGeometry("logits", None, "unit")
    assert np.array_equal(logits.resolve(3), np.ones(3))


def test_semantic_donor_law_minimum_gap_and_graph_balancing():
    donors = [
        (10, graph([[2], [3], [4], [5]], [])),
        (11, graph([[6]], [])),
    ]
    pool = SemanticDonorPool(donors)
    # Every node has degree 0 or 1. A source degree 0 retains only degree-0 nodes globally.
    eligible = pool.eligible([1], 0)
    assert set(eligible) == {10, 11}
    rng = np.random.default_rng(8)
    draws = pool.draw([1], 0, 10_000, rng)
    fraction_first_graph = np.mean([item.graph_id == 10 for item in draws])
    assert 0.47 < fraction_first_graph < 0.53


def test_semantic_donor_excludes_identical_payload():
    pool = SemanticDonorPool(
        [
            (1, graph([[7], [8]], [[0, 1], [1, 0]])),
            (2, graph([[7]], [])),
        ]
    )
    draws = pool.draw([7], 0, 20, np.random.default_rng(3))
    assert all(item.payload != (7,) for item in draws)


def test_structural_donor_law_minimum_gap_with_replacement():
    footprints = [b"a", b"b", b"c", b"d"]
    degrees = [2, 4, 3, 3]
    result = draw_structural_donors(
        footprints,
        degrees,
        source=0,
        count=30,
        rng=np.random.default_rng(2),
        equal=lambda left, right: left == right,
    )
    assert set(result) <= {2, 3}
    assert len(result) == 30


def test_structural_swap_matches_dense_row_column_self_and_fixed_support():
    task = get_task("zinc")
    base = structural_graph()
    event = structural_donor_swap(
        base, 0, 2, task=task, duplicate_tolerance=1e-7
    )
    dense = torch.zeros(3, 3)
    dense[base.rrwp_index[0], base.rrwp_index[1]] = base.rrwp_val[:, 0]
    changed = torch.zeros(3, 3)
    changed[event.rrwp_index[0], event.rrwp_index[1]] = event.rrwp_val[:, 0]
    expected = dense.clone()
    expected[0, :] = dense[2, :]
    expected[:, 0] = dense[:, 2]
    expected[0, 0] = dense[2, 2]
    assert torch.equal(changed, expected)
    assert torch.equal(event.rrwp[0], base.rrwp[2])
    assert torch.equal(event.x, base.x)
    assert torch.equal(event.edge_index, base.edge_index)
    assert torch.equal(event.edge_attr, base.edge_attr)
    assert torch.equal(event.rrwp_local_edge_index, base.rrwp_local_edge_index)
    assert torch.equal(event.rrwp[2], base.rrwp[2])  # donor is not transposed


def test_sparse_duplicates_agree_or_abort():
    index = torch.tensor([[0, 0, 1], [1, 1, 0]])
    value = torch.tensor([[2.0], [2.0 + 1e-9], [3.0]])
    new_index, new_value = coalesce_equal_sparse(
        index, value, num_nodes=2, tolerance=1e-7
    )
    assert new_index.shape[1] == 2
    with pytest.raises(StructuralAuditError, match="conflicting duplicate"):
        coalesce_equal_sparse(
            index,
            torch.tensor([[2.0], [2.1], [3.0]]),
            num_nodes=2,
            tolerance=1e-7,
        )


def test_unknown_structural_field_is_fatal():
    task = get_task("zinc")
    base = structural_graph()
    base.lap_pe = torch.ones(3, 2)
    with pytest.raises(StructuralAuditError, match="not registered"):
        structural_donor_swap(
            base, 0, 2, task=task, duplicate_tolerance=1e-7
        )


def test_transport_projection_event_norm_and_hierarchical_aggregation():
    delta = torch.tensor(
        [
            [[[[3.0]], [[4.0]]]],
            [[[[0.0]], [[5.0]]]],
            [[[[6.0]], [[8.0]]]],
        ]
    )  # [E=3,L=1,N=2,H=1,D=1]
    gradient = torch.ones(1, 1, 2, 1, 1)
    q = project_transport(delta, gradient)
    event = event_head_scores(q).numpy()
    assert event[:, 0, 0].tolist() == [7.0, 5.0, 14.0]
    total, graphs, sources = aggregate_event_scores(
        event,
        graph_ids=[0, 0, 1],
        source_ids=[0, 0, 0],
    )
    assert total[0, 0] == pytest.approx((6.0 + 14.0) / 2)
    assert sources[(0, 0)][0, 0] == 6.0


def test_head_coordinates_use_within_model_means_and_gate_only_selectivity():
    semantic = np.array([[1.0, 3.0]])
    structural = np.array([[4.0, 2.0]])
    result = head_coordinates(
        semantic,
        structural,
        score_floor=1e-12,
        epsilon=1e-12,
        activity_floor=1.2,
    )
    assert result.semantic_mean == 2.0
    assert result.structural_mean == 3.0
    assert np.allclose(result.joint_sensitivity, 0.5 * (
        semantic / 2.0 + structural / 3.0
    ))
    assert result.joint_sensitivity.shape == result.active.shape


def test_distance_accounting_reconstructs_and_divides_inside_graph():
    q = torch.tensor([[[[[3.0], [4.0]]]]])  # [E,L,H,N,T]
    axis = DistanceAxis((0, 1))
    contribution, support = distance_event_contributions(q, [0, 1], axis)
    graph_c, graph_o = aggregate_distance_events(
        contribution, support, [9], [2]
    )
    score = {9: np.array([[7.0]])}
    result = score_heatmaps(
        graph_c, graph_o, reconstruction_tolerance=1e-8, graph_scores=score
    )
    assert np.allclose(result.exact, [[3.0, 4.0]])
    assert np.allclose(result.per_opportunity, [[3.0, 4.0]])


def test_functional_carriage_takes_event_norm_before_donor_mean():
    delta = torch.tensor([[[[1.0]], [[-1.0]]]])  # [S=1,K=2,N=1,M=1]
    gradient = torch.ones(1, 1, 1)
    result = functional_carriage(delta, gradient)
    assert result.shape == (1, 1)
    assert float(result[0, 0]) == pytest.approx(1.0)


def test_beneficial_carriage_positive_sign_and_completeness():
    h_clean = torch.zeros(2, 1)
    h_event = torch.full((1, 1, 2, 1), 0.5)

    def loss_from_pooled(pooled):
        return pooled[:, 0].square()

    result = beneficial_carriage(
        h_clean,
        h_event,
        loss_from_pooled,
        pooling="add",
        atol=1e-8,
        rtol=1e-8,
        max_intervals=16,
        tolerance=1e-7,
    )
    assert float(result.field.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(result.event_loss_increase[0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert bool((result.field > 0).all())


def test_causal_patch_terms_keep_gross_and_alignment_separate():
    clean = np.array([[2.0, 0.0]])
    event = np.array([[0.0, 0.0]])
    matched = patch_response(
        clean, event, np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]), epsilon=1e-12
    )
    mismatch = patch_response(
        clean, event, np.array([[0.0, 1.0]]), np.array([[2.0, 1.0]]), epsilon=1e-12
    )
    assert matched.restoration_aligned[0] > 0
    assert matched.injection_aligned[0] > 0
    assert mismatch.restoration_gross[0] == 1.0
    assert mismatch.restoration_aligned[0] == 0.0
    assert mismatch_adjusted_gross(matched, mismatch).shape == (1,)


def test_donor_necessity_alignment():
    result = donor_necessity(
        [[2.0, 0.0]],
        [[0.0, 0.0]],
        [[1.0, 0.0]],
        [[0.0, 0.0]],
        epsilon=1e-12,
    )
    assert result["aligned_necessity"][0] == pytest.approx(1.0)
    assert result["gross_necessity"][0] == pytest.approx(1.0)


def test_trimmed_mean_definition_and_fixed_nested_bootstrap():
    assert trimmed_mean([0, 1, 2, 3, 100]) == pytest.approx(2.0)
    policy = BootstrapPolicy(rng_seed=12)
    observations = [
        Observation(0, graph_id, source, donor, graph_id + source + donor)
        for graph_id in range(2)
        for source in range(2)
        for donor in range(2)
    ]
    first = nested_percentile_interval(observations, policy)
    second = nested_percentile_interval(observations, policy)
    assert first.replicates == 2_000
    assert first.estimate == pytest.approx(second.estimate)
    assert first.low == pytest.approx(second.low)
    assert first.high == pytest.approx(second.high)


def contract(**updates):
    values = dict(
        protocol_fingerprint="abc",
        task="zinc",
        task_adapter_version="v1",
        checkpoint_sha256="123",
        train_seed=42,
        model_geometry={"layers": 2},
        output_representation="evaluation_regression",
        sigma=(1.0,),
        split_fingerprint="split",
        event_manifest_hash="events",
        donors_per_source=8,
        source_cap=6,
        bootstrap_seed=17,
    )
    values.update(updates)
    return CacheContract(**values)


def test_cache_rejects_any_contract_change(tmp_path):
    cache = CanonicalCache(tmp_path, contract())
    cache.save("scores", "raw", {"ok": True})
    assert cache.load("scores", "raw") == {"ok": True}
    stale = CanonicalCache(tmp_path, contract(event_manifest_hash="other"))
    assert stale.load("scores", "raw") is None
    with pytest.raises(StaleCacheError):
        stale.load("scores", "raw", strict=True)


def test_figure_axis_strings_are_repository_fixed():
    coordinates = head_coordinates(
        np.array([[1.0, 2.0], [1.5, 2.5]]),
        np.array([[2.0, 1.0], [2.5, 1.5]]),
        score_floor=1e-12,
        epsilon=1e-12,
        activity_floor=0.5,
    )
    fig, ax = score_plane(HeadPlotData(coordinates, seed=42))
    assert ax.get_xlabel() == STRUCTURAL_AXIS_LABEL
    assert ax.get_ylabel() == SEMANTIC_AXIS_LABEL
    fig.clf()
    fig, ax = joint_selectivity_plane(HeadPlotData(coordinates, seed=42))
    assert ax.get_xlabel() == SELECTIVITY_AXIS_LABEL
    assert ax.get_ylabel() == JOINT_AXIS_LABEL
    fig.clf()


def test_modular_causal_and_distance_figure_components_render():
    interval = (np.asarray([0.0, 0.5]), np.asarray([1.0, 1.5]))
    fig, _ = causal_scatter_grid(
        [
            {
                "x": np.asarray([0.2, 0.8]),
                "y": np.asarray([0.3, 1.0]),
                "x_interval": interval,
                "y_interval": interval,
                "layer": np.asarray([0, 1]),
                "xlabel": "x",
                "ylabel": "y",
            }
        ]
    )
    fig.clf()
    keys = (
        "restoration_gross",
        "injection_gross",
        "rescue",
        "induction",
        "necessity",
    )
    values = {key: np.ones((2, 2)) for key in keys}
    intervals = {
        key: (np.full((2, 2), 0.5), np.full((2, 2), 1.5)) for key in keys
    }
    fig, _ = causal_family_panels(
        ("semantic_leaning", "structural_leaning"),
        values,
        intervals=intervals,
    )
    fig.clf()
    curve = {
        "semantic_leaning": {
            "prefix": [1, 2],
            "gross": {"semantic": [0.1, 0.2], "structural": [0.0, 0.1]},
            "gross_interval": {
                "semantic": ([0.0, 0.1], [0.2, 0.3]),
                "structural": ([-0.1, 0.0], [0.1, 0.2]),
            },
            "necessity": {"semantic": [0.1, 0.2], "structural": [0.0, 0.1]},
            "necessity_interval": {
                "semantic": ([0.0, 0.1], [0.2, 0.3]),
                "structural": ([-0.1, 0.0], [0.1, 0.2]),
            },
        }
    }
    fig, _ = cumulative_prefix_curves(curve)
    fig.clf()
    fig, _ = attention_distance_profiles(
        (0, 1), {"semantic_leaning": [0.7, 0.3]}
    )
    fig.clf()
