from __future__ import annotations

import dataclasses
import warnings
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graph_specialisation_metrics import main as public_main
from graph_specialisation_metrics.methodology import audit
from graph_specialisation_metrics.methodology.audit import audit_scope
from graph_specialisation_metrics.methodology.bootstrap import (
    Observation,
    nested_percentile_interval,
    reportable_bin,
    trimmed_mean,
)
from graph_specialisation_metrics.methodology.backend import CanonicalGritBackend
from graph_specialisation_metrics.methodology.cache import (
    CacheContract,
    CanonicalCache,
    StaleCacheError,
)
from graph_specialisation_metrics.methodology.execution import execute_graph_batches
from graph_specialisation_metrics.methodology.carriage import (
    beneficial_carriage,
    beneficial_carriage_ragged,
    functional_carriage,
    functional_carriage_ragged,
)
from graph_specialisation_metrics.methodology.causal import (
    donor_necessity,
    mismatch_adjusted_gross,
    patch_response,
    reference_scale,
)
from graph_specialisation_metrics.methodology.distance import (
    DistanceAxis,
    aggregate_distance_events,
    column_support,
    display_bins,
    distance_event_contributions,
    distance_profile_reduce,
    per_opportunity_from_ratio,
    row_normalised,
    score_heatmaps,
    supported_mean,
)
from graph_specialisation_metrics.methodology.figures import (
    HeadPlotData,
    attention_distance_profiles,
    distance_heatmaps,
    distance_support_profile,
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
    ExecutionPolicy,
    MethodologyConfig,
    RunSizes,
    deterministic_splits,
)
from graph_specialisation_metrics.methodology.progress import (
    ProgressTracker,
    format_duration,
)
from graph_specialisation_metrics.methodology.sampling import (
    DonorEvent,
    SemanticDonorPool,
    draw_structural_donors,
)
from graph_specialisation_metrics.methodology.runner import _carriage_graph_batch
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
from graph_specialisation_metrics.methodology.validation import (
    _clean_ablation_stage,
    _mismatch_indices,
)


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


def test_task_specific_seed_labels_support_mixed_backends():
    config = MethodologyConfig(
        tasks=("zinc", "graphormer_pcqm4mv2"),
        train_seeds=(42, 43),
        task_train_seeds={"graphormer_pcqm4mv2": (0,)},
    )
    config.validate()
    assert config.seeds_for("zinc") == (42, 43)
    assert config.seeds_for("graphormer_pcqm4mv2") == (0,)
    assert config.scientific_record["task_train_seeds"] == {
        "graphormer_pcqm4mv2": [0]
    }


def test_numerical_audits_are_soft_by_default_and_strict_on_request():
    config = MethodologyConfig()
    assert config.strict_audits is False
    # Execution policy must not enter the cache/protocol fingerprint.
    assert config.fingerprint == dataclasses.replace(config, strict_audits=True).fingerprint
    faster = dataclasses.replace(
        config, execution=ExecutionPolicy(graphs_per_batch=16)
    )
    assert config.fingerprint == faster.fingerprint
    assert config.record()["execution"] == {
        "graphs_per_batch": 4,
        "oom_backoff": True,
        "verbose_progress": True,
        "progress_updates": 20,
        "heartbeat_seconds": 60.0,
    }
    quiet = dataclasses.replace(
        config,
        execution=ExecutionPolicy(
            graphs_per_batch=32,
            oom_backoff=False,
            verbose_progress=False,
            progress_updates=7,
            heartbeat_seconds=0,
        ),
    )
    assert config.fingerprint == quiet.fingerprint

    with audit_scope("test") as scope:
        assert audit.audit_check(True, "test.pass", "never recorded")
        assert not audit.within_tolerance(1.0e-5, 1.0e-6, "test.tolerance", "observed")
        assert not audit.within_tolerance(2.0e-5, 1.0e-6, "test.tolerance", "observed")
    findings = scope.records()
    assert [row["name"] for row in findings] == ["test.tolerance"]
    assert findings[0]["count"] == 2
    assert findings[0]["observed"] == pytest.approx(2.0e-5)

    previous = audit.set_strict(True)
    try:
        with pytest.raises(audit.AuditError, match="test.tolerance"):
            audit.within_tolerance(1.0e-5, 1.0e-6, "test.tolerance", "observed")
    finally:
        audit.set_strict(previous)


def test_graph_batch_executor_preserves_order_and_retries_without_double_consumption():
    consumed = []
    attempts = []

    def execute(chunk):
        attempts.append(tuple(chunk))
        if len(chunk) > 2:
            raise RuntimeError("CUDA out of memory")
        return [value * 10 for value in chunk]

    report = execute_graph_batches(
        list(range(7)),
        graphs_per_batch=4,
        execute=execute,
        consume=consumed.extend,
        oom_backoff=True,
    )

    assert consumed == [value * 10 for value in range(7)]
    assert attempts == [
        (0, 1, 2, 3),
        (0, 1),
        (2, 3, 4, 5),
        (2, 3),
        (4, 5, 6),
        (4,),
        (5, 6),
    ]
    assert report.requested_graphs_per_batch == 4
    assert report.minimum_graphs_per_batch == 1
    assert report.maximum_graphs_per_batch == 2
    assert report.successful_batches == 4
    assert report.oom_retries == 3


def test_graph_batch_executor_does_not_hide_non_oom_errors():
    with pytest.raises(RuntimeError, match="scientific failure"):
        execute_graph_batches(
            [1, 2],
            graphs_per_batch=2,
            execute=lambda _chunk: (_ for _ in ()).throw(
                RuntimeError("scientific failure")
            ),
            consume=lambda _result: None,
            oom_backoff=True,
        )


def test_progress_tracker_reports_bounded_progress_and_eta(monkeypatch):
    messages = []
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.progress.log",
        messages.append,
    )
    tracker = ProgressTracker("unit operation", 4, unit="graphs", updates=2)
    for index in range(4):
        tracker.advance(detail=f"graph={index}")
    tracker.finish()

    assert format_duration(65) == "01:05"
    assert format_duration(3661) == "1:01:01"
    assert messages[0].startswith("[progress] START unit operation | 4 graphs")
    updates = [message for message in messages if "elapsed" in message]
    assert len(updates) == 2
    assert "2/4 graphs (50.0%)" in updates[0]
    assert "ETA " in updates[0]
    assert "4/4 graphs (100.0%)" in updates[-1]


def test_grit_grouped_capture_matches_individual_variable_size_groups():
    pyg_data = pytest.importorskip("torch_geometric.data")
    Data = pyg_data.Data

    class Layers(torch.nn.Module):
        def forward(self, batch):
            batch.x = batch.x.float().clone().requires_grad_(True)
            return batch

    class Runtime:
        device = torch.device("cpu")
        L = H = dh = dim_h = 1

        def __init__(self):
            self.model = SimpleNamespace(
                model=SimpleNamespace(layers=Layers())
            )

        def capture(
            self,
            batch,
            *,
            want_grad,
            want_attn,
            include_virtual_transport,
        ):
            del want_grad, want_attn, include_virtual_transport
            output = self.model.model.layers(batch)
            routed = output.x.float().reshape(-1, 1, 1)
            graphs = int(output.batch.max().item()) + 1
            prediction = torch.zeros(graphs, 1)
            prediction.index_add_(0, output.batch, routed.reshape(-1, 1))
            return {
                "pred": prediction,
                "true": output.y.reshape(graphs, -1),
                "wV": [routed],
                "node_graph": output.batch.clone(),
            }

    task = SimpleNamespace(
        virtual_node=False,
        output=OutputGeometry("evaluation_regression", (1.0,), "fixed"),
    )
    backend = CanonicalGritBackend(Runtime(), task, (1.0,))

    def data(values, target):
        values = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
        return Data(
            x=values,
            y=torch.tensor([target], dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=len(values),
        )

    groups = [
        [data([1, 2], 1), data([3, 4], 1)],
        [data([5, 6, 7], 2), data([8, 9, 10], 2)],
    ]
    individual = [
        backend.capture(group, require_grad=False) for group in groups
    ]
    grouped = backend.capture_groups(groups)

    assert len(grouped) == len(individual) == 2
    for observed, expected in zip(grouped, individual):
        assert torch.equal(observed.prediction, expected.prediction)
        assert torch.equal(observed.target, expected.target)
        assert torch.equal(observed.final_state, expected.final_state)
        assert torch.equal(observed.transport[0], expected.transport[0])

    clean_individual = [
        backend.clean_jacobians(groups[0][0]),
        backend.clean_jacobians(groups[1][0]),
    ]
    clean_grouped = backend.clean_jacobians_many(
        [groups[0][0], groups[1][0]]
    )
    for observed, expected in zip(clean_grouped, clean_individual):
        assert torch.equal(observed.capture.prediction, expected.capture.prediction)
        assert torch.equal(observed.capture.final_state, expected.capture.final_state)
        assert torch.equal(observed.transport, expected.transport)
        assert torch.equal(observed.final_state, expected.final_state)


def test_clean_ablation_reuses_clean_captures_and_batches_each_target(monkeypatch):
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.validation._spearman",
        lambda x, y: {"rho": 0.0, "p": 1.0, "n": min(len(x), len(y))},
    )

    class Backend:
        def __init__(self):
            self.clean_batch_sizes = []
            self.ablation_batch_sizes = []

        def ablate(self, data_list, family):
            if not family:
                self.clean_batch_sizes.append(len(data_list))
                prediction = torch.tensor(
                    [[float(data.value)] for data in data_list]
                )
                return prediction, prediction, torch.zeros_like(prediction)
            self.ablation_batch_sizes.append(len(data_list))
            prediction = torch.tensor(
                [[float(data.value) + 1.0] for data in data_list]
            )
            return prediction, prediction, torch.zeros_like(prediction)

        @staticmethod
        def loss_per_graph(prediction, target):
            return (prediction - target).square().sum(dim=-1)

    backend = Backend()
    prepared = SimpleNamespace(
        backend=backend,
        grit=SimpleNamespace(
            eval_ds=[SimpleNamespace(value=value) for value in (1.0, 2.0, 3.0)],
            L=1,
            H=1,
            sc=SimpleNamespace(seed=0),
        ),
        splits=SimpleNamespace(clean_ablation=(0, 1, 2)),
        task=SimpleNamespace(
            metric_fn=lambda prediction, truth: float(
                np.mean(np.abs(prediction - truth))
            )
        ),
    )
    config = SimpleNamespace(
        execution=ExecutionPolicy(graphs_per_batch=2, oom_backoff=True),
        bootstrap=BootstrapPolicy(),
    )
    result = _clean_ablation_stage(
        prepared,
        config,
        {
            "head_L0_H0": ((0, 0),),
            "family_test": ((0, 0),),
        },
        scores=SimpleNamespace(
            coordinates=SimpleNamespace(joint_sensitivity=np.asarray([[1.0]]))
        ).__dict__,
    )

    assert backend.clean_batch_sizes == [2, 1]
    assert backend.ablation_batch_sizes == [2, 1, 2, 1]
    assert result["_execution"]["clean_reused_across_targets"] is True
    assert result["head_L0_H0"]["prediction_movement"] == pytest.approx(1.0)


def test_unsupported_distance_columns_never_warn_and_stay_non_estimable():
    # A column no graph populates is a normal consequence of the frozen distance axis.
    ratio = np.array([[[[1.0, np.nan]]], [[[3.0, np.nan]]]])  # [G=2,L=1,H=1,D=2]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        mean_ratio = supported_mean(ratio, axis=0)
        per_opportunity = per_opportunity_from_ratio(mean_ratio)
        empty = supported_mean(np.full((3, 2), np.nan), axis=0)
    assert mean_ratio[0, 0, 0] == pytest.approx(2.0)
    assert np.isnan(mean_ratio[0, 0, 1])
    assert per_opportunity[0, 0] == pytest.approx(2.0)
    assert np.isnan(per_opportunity[0, 1])
    assert np.isnan(empty).all()
    # Matches np.nanmean exactly wherever np.nanmean is defined.
    supported = np.array([[1.0, 2.0], [3.0, np.nan]])
    assert supported_mean(supported, axis=0) == pytest.approx(
        np.nanmean(supported, axis=0), nan_ok=True
    )


def test_empty_replicates_are_missing_not_zero_in_the_interval():
    policy = BootstrapPolicy(rng_seed=5)
    observations = [
        Observation(0, graph_id, 0, donor, np.array([1.0, np.nan]))
        for graph_id in range(12)
        for donor in range(2)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        interval = nested_percentile_interval(observations, policy)
    # A column that is non-estimable in every draw stays non-estimable, never 0.
    assert interval.low[0] == pytest.approx(1.0)
    assert np.isnan(interval.low[1]) and np.isnan(interval.high[1])
    assert interval.estimable_draws[0] == policy.replicates
    assert interval.estimable_draws[1] == 0


def test_statistic_caption_reports_rho_with_its_interval_and_permutation_p():
    from graph_specialisation_metrics.methodology.figures import statistic_caption

    caption = statistic_caption(
        {"rho": 0.8271, "low": 0.74, "high": 0.89, "p": 0.0004, "n": 80}
    )
    assert caption == "ρ = 0.83 [0.74, 0.89], p < 0.001, n = 80"
    # The permutation floor is 1/(replicates + 1), so an exact p is never claimed below it.
    assert "p < 0.001" in statistic_caption({"rho": 0.5, "p": 0.0})
    assert statistic_caption({"rho": 0.5, "p": 0.0125}).endswith("p = 0.013")
    # A non-estimable coordinate says so rather than printing a misleading number.
    assert statistic_caption({"rho": np.nan}) == "ρ not estimable"
    assert statistic_caption(None) == ""


def test_family_styles_follow_the_channel_colours_used_everywhere_else():
    from graph_specialisation_metrics.methodology.figures import FigureTheme, family_style

    theme = FigureTheme()
    # A semantic-leaning family drawn in the structural channel's blue would contradict every
    # other figure in the set, so the mapping is pinned.
    assert family_style("family_semantic_leaning", theme)["color"] == theme.semantic_color
    assert family_style("structural_leaning", theme)["color"] == theme.structural_color
    assert family_style("central_responsive", theme)["color"] == theme.central_color
    assert family_style("inactive", theme)["color"] == theme.inactive_color
    # Marker and dash also separate the families, so the figure survives greyscale printing.
    styles = [
        family_style(name, theme)
        for name in ("semantic_leaning", "structural_leaning", "central_responsive", "inactive")
    ]
    assert len({style["marker"] for style in styles}) == 4
    assert len({style["linestyle"] for style in styles}) == 4


def test_saved_figures_keep_content_drawn_outside_the_axes(tmp_path):
    import matplotlib.image as mpimg

    from graph_specialisation_metrics.methodology.figures import FigureBuilder, FigureTheme

    names = [
        f"control_{leaning}_leaning_{kind}_control"
        for leaning in ("semantic", "structural")
        for kind in ("central", "inactive", "random")
    ]
    keys = ("restoration_gross", "injection_gross", "rescue", "induction", "necessity")
    values = {key: np.full((2, len(names)), 0.5) for key in keys}
    theme = FigureTheme(dpi=100)
    figure, axes = causal_family_panels(names, values, theme=theme)
    paths = FigureBuilder(tmp_path, theme, common_metadata={}).save(
        "controls", figure, axes, metadata={}
    )
    png = next(path for path in paths if path.suffix == ".png")
    # The legend is drawn below the panels. Saving without a tight bound crops it away silently,
    # leaving an image no taller than the nominal canvas.
    nominal = max(theme.height, 0.42 * len(names) + 1.5) * theme.dpi
    assert mpimg.imread(png).shape[0] > nominal


def test_display_bins_fit_the_budget_and_keep_the_near_field_at_unit_resolution():
    # A peptides-sized axis: 0..116 plus an explicit non-numeric column.
    labels = tuple(range(117)) + ("unreachable",)
    display = display_bins(labels, max_points=14)
    assert len(display.labels) <= 14
    # The explicit column reserves a slot, leaving 13 for the numeric axis.
    assert display.labels[:9] == tuple(str(value) for value in range(9))
    assert display.labels[9:-1] == ("9-17", "18-35", "36-71", "72-116")
    assert display.labels[-1] == "unreachable"
    # Contiguous, exhaustive, and in registered order.
    assert [index for group in display.groups for index in group] == list(range(118))
    assert not display.identity

    # A ZINC-sized axis already fits, so binning is a no-op and figures are unchanged.
    small = display_bins(tuple(range(11)), max_points=14)
    assert small.identity
    assert small.labels == tuple(str(value) for value in range(11))


def test_display_grouping_sums_mass_and_recomputes_ratios_from_support():
    labels = tuple(range(6))
    display = display_bins(labels, max_points=4)
    contribution = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]])  # [L=1,H=1,D=6]
    support = np.array([2.0, 2.0, 1.0, 0.0, 0.0, 0.0])  # nothing supports the widened tail
    grouped = display.group_sum(contribution)
    assert grouped.sum() == pytest.approx(contribution.sum())  # mass is conserved
    ratio = display.group_ratio(contribution, np.broadcast_to(support, contribution.shape))
    # A ratio is summed contribution over summed support, never a mean of per-column ratios.
    expected = [
        contribution[0, 0, list(group)].sum() / support[list(group)].sum()
        if support[list(group)].sum() > 0
        else np.nan
        for group in display.groups
    ]
    assert ratio[0, 0] == pytest.approx(np.asarray(expected), nan_ok=True)
    assert np.isnan(ratio[0, 0, -1])  # a group with no opportunity stays non-estimable


def test_grouped_mass_is_reported_per_unit_distance():
    # A flat profile must stay flat once grouped: a wide group holds more mass only because it is
    # wide, and drawing that sum would invent a resurgence in the tail.
    flat = np.ones(24)
    display = display_bins(tuple(range(24)), max_points=8)
    assert not display.identity
    assert display.group_density(flat) == pytest.approx(np.ones(len(display.labels)))
    assert display.group_sum(flat).sum() == pytest.approx(flat.sum())  # the sum still reconstructs
    # At unit resolution the density is the identity, so short-diameter tasks are untouched.
    identity = display_bins(tuple(range(6)), max_points=14)
    values = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0])
    assert identity.group_density(values) == pytest.approx(values)


def test_column_support_counts_graphs_and_pairs_for_the_reporting_floor():
    support = {
        1: np.array([2.0, 1.0, 0.0]),
        2: np.array([2.0, 0.0, 0.0]),
    }
    graphs, pairs = column_support(support, {1: 3, 2: 6})
    assert graphs.tolist() == [2, 1, 0]
    assert pairs.tolist() == [18, 3, 0]
    policy = BootstrapPolicy()
    assert not reportable_bin([1, 2], 18, policy=policy)
    assert reportable_bin(list(range(10)), 50, policy=policy)


def test_events_without_an_admissible_mismatch_control_are_excluded():
    def event(source, draw, fingerprint, degree_gap, channel="semantic"):
        return FakeData(
            channel=channel,
            source=source,
            draw=draw,
            payload_fingerprint=fingerprint,
            degree_gap=degree_gap,
            dose=float(draw),
        )

    # Distinct payloads inside one degree tier: the registered control is available.
    records = [
        event(0, 0, b"a", 1),
        event(0, 1, b"b", 1),
        event(1, 0, b"c", 1),
    ]
    with audit_scope("mismatch") as scope:
        indices, excluded = _mismatch_indices(records)
    assert indices[0] == 1 and indices[1] == 0
    assert excluded == set()
    assert not scope.records()

    # Distinct payloads but no shared tier: the tier is relaxed, the event is still controlled.
    with audit_scope("mismatch-relaxed") as scope:
        indices, excluded = _mismatch_indices(
            [event(0, 0, b"a", 1), event(1, 0, b"c", 2)]
        )
    assert indices == [1, 0]
    assert excluded == set()
    assert [row["name"] for row in scope.records()] == ["causal.mismatch_control_relaxed"]

    # One payload only: no admissible control exists, so both events leave the causal record.
    with audit_scope("mismatch-none") as scope:
        _, excluded = _mismatch_indices([records[0], event(0, 1, b"a", 1)])
    assert excluded == {0, 1}
    assert [row["name"] for row in scope.records()] == [
        "causal.mismatch_control_unavailable"
    ]

    # Structural controls use the full non-identical donor tier: degree gap is recorded but is
    # not a matching restriction.
    structural = [
        event(0, 0, b"a", 1, "structural"),
        event(0, 1, b"b", 4, "structural"),
    ]
    with audit_scope("structural-mismatch") as scope:
        indices, excluded = _mismatch_indices(structural)
    assert indices == [1, 0]
    assert excluded == set()
    assert not scope.records()


def test_soft_audit_keeps_a_broken_reconstruction_running():
    q = torch.tensor([[[[[3.0], [4.0]]]]])  # [E,L,H,N,T]
    axis = DistanceAxis((0, 1))
    contribution, support = distance_event_contributions(q, [0, 1], axis)
    graph_c, graph_o = aggregate_distance_events(contribution, support, [9], [2])
    with audit_scope("reconstruction") as scope:
        result = score_heatmaps(
            graph_c,
            graph_o,
            reconstruction_tolerance=1e-8,
            graph_scores={9: np.array([[99.0]])},
        )
    assert np.allclose(result.exact, [[3.0, 4.0]])
    assert [row["name"] for row in scope.records()] == ["distance.bucket_reconstruction"]


def test_non_estimable_reference_scale_is_reported_not_raised():
    with audit_scope("reference") as scope:
        assert np.isnan(reference_scale([0.0, 0.0], floor=1e-8))
        assert reference_scale([2.0, 4.0], floor=1e-8) == pytest.approx(3.0)
    assert [row["name"] for row in scope.records()] == ["causal.reference_scale"]


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


def test_structural_donor_law_is_uniform_without_replacement_and_exhaustive():
    footprints = [b"a", b"b", b"c", b"d"]
    result = draw_structural_donors(
        footprints,
        source=0,
        count=30,
        rng=np.random.default_rng(2),
        equal=lambda left, right: left == right,
    )
    # Fewer than K eligible donors means every donor is used exactly once.
    assert result.tolist() == [1, 2, 3]

    rng = np.random.default_rng(9)
    draws = np.concatenate(
        [
            draw_structural_donors(
                footprints,
                source=0,
                count=1,
                rng=rng,
                equal=lambda left, right: left == right,
            )
            for _ in range(6_000)
        ]
    )
    frequencies = np.bincount(draws, minlength=4)[1:] / len(draws)
    assert np.all(np.abs(frequencies - 1.0 / 3.0) < 0.03)

    unique = draw_structural_donors(
        footprints,
        source=0,
        count=2,
        rng=np.random.default_rng(4),
        equal=lambda left, right: left == right,
    )
    assert len(unique) == len(set(unique.tolist())) == 2


def test_structural_donor_law_excludes_source_identical_footprints():
    result = draw_structural_donors(
        [b"a", b"a", b"b"],
        source=0,
        count=8,
        rng=np.random.default_rng(2),
        equal=lambda left, right: left == right,
    )
    assert result.tolist() == [2]


def test_ragged_carriage_averages_each_sources_actual_donors():
    gradient = torch.ones((1, 2, 1), dtype=torch.float64)
    delta = torch.tensor(
        [[[1.0], [0.0]], [[0.0], [2.0]], [[0.0], [4.0]]],
        dtype=torch.float64,
    )
    functional, functional_events = functional_carriage_ragged(
        delta, gradient, (1, 2)
    )
    torch.testing.assert_close(
        functional,
        torch.tensor([[1.0, 0.0], [0.0, 3.0]], dtype=torch.float64),
    )
    assert [int(value.shape[0]) for value in functional_events] == [1, 2]

    h_clean = torch.zeros((2, 1), dtype=torch.float64)
    h_event = torch.tensor(
        [[[-1.0], [0.0]], [[0.0], [-2.0]], [[0.0], [2.0]]],
        dtype=torch.float64,
    )
    beneficial = beneficial_carriage_ragged(
        h_clean,
        h_event,
        (1, 2),
        lambda pooled: pooled[:, 0].square(),
        carrier_weights=torch.ones(2, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-8,
        max_intervals=64,
    )
    torch.testing.assert_close(
        beneficial.field,
        torch.tensor([[1.0, 0.0], [0.0, 4.0]], dtype=torch.float64),
    )
    assert [int(value.shape[0]) for value in beneficial.event_field] == [1, 2]


def test_carriage_runner_preserves_ragged_structural_donor_counts(monkeypatch):
    records = [
        DonorEvent(
            channel="structural",
            stage="carriage",
            graph_id=0,
            source=source,
            donor_graph_id=0,
            donor_node=donor_node,
            source_degree=1,
            donor_degree=donor_node + 1,
            degree_gap=donor_node,
            dose=float(draw + 1),
            payload_fingerprint=f"footprint-{donor_node}",
            draw=draw,
        )
        for source, donor_node, draw in ((0, 1, 0), (1, 0, 0), (1, 2, 1))
    ]
    variants = [FakeData() for _ in records]
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.runner._rebuild_graph_events",
        lambda *_args, **_kwargs: (variants, records),
    )
    final_state = torch.tensor(
        [
            [[0.0], [0.0]],
            [[-1.0], [0.0]],
            [[0.0], [-2.0]],
            [[0.0], [2.0]],
        ],
        dtype=torch.float64,
    )
    prediction = final_state.sum(dim=1)
    captured = FakeData(
        final_state=final_state,
        prediction=prediction,
        target=torch.zeros_like(prediction),
    )
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.runner._capture_event_groups",
        lambda _prepared, _groups: [captured],
    )

    class Backend:
        @staticmethod
        def carriage_weights(_base, _clean):
            return torch.ones(2, dtype=torch.float64)

        @staticmethod
        def loss_from_pooled(_target):
            return lambda pooled: pooled[:, 0].square()

        @staticmethod
        def loss_per_graph(prediction, _target):
            return prediction[:, 0].square()

        @staticmethod
        def carriage_distance_matrix(_base, _sources, _pristine):
            return np.zeros((2, 2), dtype=np.float64)

        @staticmethod
        def carriage_carrier_kind(_base, _carrier):
            return "node"

    base = graph([[1], [2], [3]], [[0, 1], [1, 0]])
    prepared = SimpleNamespace(
        grit=SimpleNamespace(eval_ds=[base], sc=SimpleNamespace(seed=0)),
        backend=Backend(),
    )
    config = SimpleNamespace(
        sizes=SimpleNamespace(donors_per_source=8),
        numerical=SimpleNamespace(
            integrated_atol=1e-10,
            integrated_rtol=1e-8,
            integrated_max_intervals=64,
            reconstruction_tolerance=1e-8,
        ),
    )
    plan = {
        0: {
            "structural": {
                "sources": (0, 1),
                "records": tuple(record.record() for record in records),
            }
        }
    }
    clean = {
        0: SimpleNamespace(
            final_state=torch.ones((1, 2, 1), dtype=torch.float64),
            capture=SimpleNamespace(target=torch.zeros((1, 1), dtype=torch.float64)),
        )
    }
    result = _carriage_graph_batch(
        prepared,
        config,
        plan,
        clean,
        "structural",
        [0],
    )[0]

    assert result["graph_field"]["donor_counts"] == (1, 2)
    assert [len(value) for value in result["graph_field"]["event_B"]] == [1, 2]
    assert len(result["pair_rows"]) == 6
    assert result["paths"] == 3


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
    # The head-resolved arrays the figures display are what the aggregate views sum over heads.
    assert result.exact_head.shape == (1, 1, 2)
    assert np.allclose(result.exact_head.sum(axis=1), result.exact)
    assert np.allclose(result.per_opportunity_head.sum(axis=1), result.per_opportunity)


def test_head_rows_survive_aggregation_with_diverse_peaks():
    # Two heads peaking at opposite ends: the head sum reports a flat layer neither head has.
    graph_c = {0: np.array([[[9.0, 1.0], [1.0, 9.0]]])}  # [L=1,H=2,D=2]
    graph_o = {0: np.array([1.0, 1.0])}
    result = score_heatmaps(graph_c, graph_o, reconstruction_tolerance=1e-8)
    assert np.allclose(result.exact, [[10.0, 10.0]])
    assert np.allclose(result.exact_head, graph_c[0])
    fractions = row_normalised(result.exact_head)
    assert np.allclose(fractions, [[[0.9, 0.1], [0.1, 0.9]]])


def test_row_normalisation_uses_the_full_axis_and_keeps_empty_rows_missing():
    values = np.array([[[3.0, 1.0, np.nan], [0.0, 0.0, 0.0]]])  # [L=1,H=2,D=3]
    result = row_normalised(values)
    # nan columns stay nan and are excluded from the denominator, not treated as zero mass.
    assert result[0, 0, 0] == pytest.approx(0.75)
    assert result[0, 0, 1] == pytest.approx(0.25)
    assert np.isnan(result[0, 0, 2])
    # A row with no mass has no profile to report.
    assert np.isnan(result[0, 1]).all()
    # Blanking a column afterwards leaves the remaining columns untouched.
    assert np.nansum(result[0, 0]) == pytest.approx(1.0)


def test_distance_heatmaps_put_layer_zero_at_the_top_of_head_blocks():
    values = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    fig, axes = distance_heatmaps(
        values, values, [0, 1, 2, 3], channel="semantic", title="task"
    )
    try:
        # One row per head, blocked by layer.
        assert axes[0].get_images()[0].get_array().shape == (6, 4)
        assert [text.get_text() for text in axes[0].get_yticklabels()] == ["L0", "L1"]
        # L0's block centre sits above L1's in display coordinates.
        assert axes[0].get_yticks()[0] < axes[0].get_yticks()[1]
        assert axes[0].yaxis_inverted()
        assert fig._suptitle.get_text() == "task"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    fig, _ = distance_heatmaps(
        values, values, [0, 1, 2, 3], channel="semantic", title="task", normalised=True
    )
    try:
        assert fig._suptitle.get_text() == "task (row-normalised)"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    with pytest.raises(ValueError, match=r"\[layer,head,distance\]"):
        distance_heatmaps(
            values.sum(axis=1), values.sum(axis=1), [0, 1, 2, 3], channel="semantic"
        )


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
    fig, _ = distance_support_profile(
        (0, 1, 2, "unreachable"),
        (48, 48, 12, 1),
        (2_880, 1_450, 60, 3),
        empty_replicate_fraction=(0.0, 0.0, 0.01, 0.37),
        minimum_graphs=10,
        minimum_pairs=50,
        channel="semantic",
    )
    fig.clf()
