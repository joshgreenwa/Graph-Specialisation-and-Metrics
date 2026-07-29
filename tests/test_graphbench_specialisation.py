from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graph_specialisation_metrics.methodology import validation as validation_module
from graph_specialisation_metrics.methodology.bootstrap import (
    Observation,
    paired_channel_percentile_interval,
)
from graph_specialisation_metrics.methodology.carriage import beneficial_carriage
from graph_specialisation_metrics.methodology.graphbench import (
    GraphBenchEdgeDonorPool,
    GraphBenchGritBackend,
    GraphBenchRuntime,
    build_graphbench_channel_events,
    edge_units,
    semantic_edge_swap,
    structural_pe_intervention,
    structural_rrwp_swap,
    verify_structural_pe_intervention,
)
from graph_specialisation_metrics.methodology.scores import event_head_score_systems
from graph_specialisation_metrics.methodology.graphbench_pe_refinement import (
    PERefinementConfig,
    PERefinementSizes,
    PreparedPERefinement,
    _candidate_public_summary,
    _causal_graph,
    _permuted_spearman_values,
    _run_causal_component,
    render_refinement_figures,
    run_common_ablation,
    semantic_event_manifest,
    structural_pair_manifest,
)
from graph_specialisation_metrics.methodology.protocol import BootstrapPolicy
from graph_specialisation_metrics.methodology.protocol import (
    MethodologyConfig,
    RunSizes,
    SplitManifest,
)
from graph_specialisation_metrics.methodology.runner import (
    PreparedTask,
    _stage_plan,
    finalize_cached_run,
    run_carriage,
    run_scores,
    run_worker,
)
from graph_specialisation_metrics.methodology.tasks import get_task


@dataclass
class Graph:
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_value: torch.Tensor
    target: torch.Tensor
    task_type: str
    num_nodes: int
    spd: torch.Tensor
    rwse: torch.Tensor
    rrwp: torch.Tensor


def matching_graph(values=(1.0, 1.0, 2.0, 2.0)) -> Graph:
    return Graph(
        node_type=torch.zeros(3, dtype=torch.long),
        edge_index=torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        ),
        edge_value=torch.tensor(values, dtype=torch.float32),
        target=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        task_type="edge_binary",
        num_nodes=3,
        spd=torch.zeros(3, 3, dtype=torch.long),
        rwse=torch.zeros(3, 16),
        rrwp=torch.arange(3 * 3 * 17, dtype=torch.float32).reshape(3, 3, 17),
    )


def flow_graph() -> Graph:
    graph = matching_graph(values=(1.0, 2.0, 3.0, 4.0))
    graph.task_type = "graph_regression"
    graph.target = torch.tensor([6.0])
    graph.node_type = torch.tensor([1, 0, 2], dtype=torch.long)
    return graph


def six_node_matching_graph() -> Graph:
    undirected = ((0, 3), (0, 4), (1, 4), (1, 5), (2, 5), (2, 3))
    directed = [edge for pair in undirected for edge in (pair, pair[::-1])]
    return Graph(
        node_type=torch.zeros(6, dtype=torch.long),
        edge_index=torch.tensor(directed, dtype=torch.long).t().contiguous(),
        edge_value=torch.tensor(
            [float(1 + index // 2) for index in range(len(directed))],
            dtype=torch.float32,
        ),
        target=torch.tensor(
            [float(index % 4 == 0) for index in range(len(directed))]
        ),
        task_type="edge_binary",
        num_nodes=6,
        spd=torch.zeros(6, 6, dtype=torch.long),
        rwse=torch.zeros(6, 16),
        rrwp=torch.arange(
            6 * 6 * 17, dtype=torch.float32
        ).reshape(6, 6, 17) / 100.0,
    )


def test_graphbench_tasks_are_explicit_protocol_extensions():
    for name in (
        "graphbench_bipartite_matching_hard",
        "graphbench_flow_hard",
    ):
        task = get_task(name)
        assert task.backend_kind == "graphbench_grit"
        assert task.semantic_source_kind == "edge"
        assert task.paired_channel_sources is False
        assert task.protocol_extension == "graphbench-edge-semantic-v1"


def test_matching_edge_semantic_swap_updates_one_reciprocal_unit_only():
    graph = matching_graph()
    assert edge_units(graph) == ((0, 1), (2, 3))

    changed = semantic_edge_swap(graph, 0, 7.0)

    assert changed.edge_value.tolist() == [7.0, 7.0, 2.0, 2.0]
    assert graph.edge_value.tolist() == [1.0, 1.0, 2.0, 2.0]
    assert torch.equal(changed.edge_index, graph.edge_index)
    assert torch.equal(changed.rrwp, graph.rrwp)
    assert torch.equal(changed.node_type, graph.node_type)


def test_graphbench_structural_swap_is_rrwp_row_column_self_on_fixed_support():
    graph = matching_graph()
    changed = structural_rrwp_swap(graph, 0, 2)

    assert torch.equal(changed.rrwp[0, 1:], graph.rrwp[2, 1:])
    assert torch.equal(changed.rrwp[1:, 0], graph.rrwp[1:, 2])
    assert torch.equal(changed.rrwp[0, 0], graph.rrwp[2, 2])
    assert torch.equal(changed.edge_index, graph.edge_index)
    assert torch.equal(changed.edge_value, graph.edge_value)
    assert torch.equal(changed.node_type, graph.node_type)


@pytest.mark.parametrize(
    ("complete_pe", "transpose"),
    ((False, False), (False, True), (True, False), (True, True)),
)
def test_graphbench_pe_refinement_interventions_cover_registered_factorial(
    complete_pe,
    transpose,
):
    graph = matching_graph()
    changed = structural_pe_intervention(
        graph,
        0,
        1,
        complete_pe=complete_pe,
        transpose=transpose,
    )
    verify_structural_pe_intervention(
        graph,
        changed,
        0,
        1,
        complete_pe=complete_pe,
        transpose=transpose,
    )

    if transpose:
        permutation = torch.tensor([1, 0, 2])
        expected_rrwp = graph.rrwp[permutation][:, permutation]
    else:
        expected_rrwp = structural_rrwp_swap(graph, 0, 1).rrwp
    assert torch.equal(changed.rrwp, expected_rrwp)
    assert torch.equal(changed.edge_index, graph.edge_index)
    assert torch.equal(changed.edge_value, graph.edge_value)
    assert torch.equal(changed.target, graph.target)
    if complete_pe:
        assert changed.degree_override.tolist() == (
            [2.0, 1.0, 1.0] if transpose else [2.0, 2.0, 1.0]
        )
    else:
        assert getattr(changed, "degree_override", None) is None


def test_complete_pe_degree_override_is_presented_to_official_grit_adapter():
    graph = matching_graph()
    changed = structural_pe_intervention(
        graph, 0, 1, complete_pe=True, transpose=False
    )
    batch = FakeRunner.collate_graphs([changed])

    assert batch.degree[0, :3].tolist() == [2.0, 2.0, 1.0]


def test_pe_refinement_changes_only_rrwp_channels_consumed_by_official_grit():
    graph = matching_graph()
    changed = structural_pe_intervention(
        graph,
        0,
        1,
        complete_pe=True,
        transpose=True,
        rrwp_steps=16,
    )

    assert torch.equal(changed.rrwp[..., 16], graph.rrwp[..., 16])
    assert not torch.equal(changed.rrwp[..., :16], graph.rrwp[..., :16])


def test_mass_coherent_and_cancellation_score_identities():
    q = torch.tensor(
        [
            [
                [
                    [[1.0, 0.0], [-1.0, 0.0]],
                    [[1.0, 0.0], [1.0, 0.0]],
                ]
            ]
        ]
    )
    systems = event_head_score_systems(q, mass_floor=1.0e-12)

    assert systems["mass"].shape == (1, 1, 2)
    assert systems["mass"][0, 0].tolist() == pytest.approx([2.0, 2.0])
    assert systems["coherent"][0, 0].tolist() == pytest.approx([0.0, 2.0])
    assert systems["carrier_coherence"][0, 0].tolist() == pytest.approx(
        [0.0, 1.0]
    )


def test_edge_donor_events_are_external_graph_and_auditable():
    base = matching_graph()
    donor = matching_graph(values=(4.0, 4.0, 5.0, 5.0))
    pool = GraphBenchEdgeDonorPool([(11, donor)])

    variants, records = build_graphbench_channel_events(
        base,
        graph_id=3,
        source=0,
        channel="semantic",
        stage="scores",
        donors=2,
        rng=np.random.default_rng(7),
        semantic_pool=pool,
    )

    assert len(variants) == len(records) == 2
    assert all(record.donor_graph_id == 11 for record in records)
    assert all(record.source_kind == "edge" for record in records)
    assert all(record.source_endpoints == (0, 1) for record in records)
    assert all(variant.edge_value[0] == variant.edge_value[1] for variant in variants)
    assert all(record.dose > 0 for record in records)


def test_nonlinear_state_readout_beneficial_carriage_is_complete():
    clean = torch.zeros(1, 2, 1)
    event = torch.tensor([[[[2.0], [0.0]]]])

    def loss_from_states(states):
        pooled = torch.cat(
            (states.mean(dim=1), states.max(dim=1).values), dim=-1
        )
        return pooled.square().sum(dim=-1)

    result = beneficial_carriage(
        clean[0],
        event,
        loss_from_states=loss_from_states,
        atol=1.0e-8,
        rtol=1.0e-8,
        max_intervals=32,
        tolerance=1.0e-6,
    )

    assert result.field.shape == (2, 1)
    assert float(result.event_loss_increase[0, 0]) == pytest.approx(5.0, abs=1e-5)
    assert float(result.event_field.sum()) == pytest.approx(5.0, abs=1e-5)
    assert float(result.completeness_residual.abs().max()) < 1.0e-5


def test_channel_independent_sources_are_paired_only_at_graph_level():
    left = [
        Observation(0, graph, source, donor, np.asarray([1.0]))
        for graph in (0, 1)
        for source in (0, 1)
        for donor in (0, 1)
    ]
    right = [
        Observation(0, graph, source, donor, np.asarray([3.0]))
        for graph in (0, 1)
        for source in (0, 1, 2)
        for donor in (0, 1)
    ]
    interval = paired_channel_percentile_interval(
        left,
        right,
        BootstrapPolicy(),
        transform=lambda value: value,
        resample_source=(True, False),
    )

    assert interval.estimate[:, 0].tolist() == [1.0, 3.0]
    assert interval.resampled_levels == (
        "graph",
        "source(channel-independent)",
        "donor(channel-independent)",
    )


def test_causal_summary_uses_graph_paired_independent_channel_sources(monkeypatch):
    endpoints = (
        "G_c",
        "P_gross_matched",
        "P_gross_mismatch",
        "R_gross",
        "I_gross",
        "R_align",
        "I_align",
        "R_align_adjusted",
        "I_align_adjusted",
        "necessity",
        "gross_necessity",
        "M_align",
    )

    def row(source):
        return {
            "graph": 4,
            "source": source,
            "donor": 2,
            **{endpoint: 1.0 for endpoint in endpoints},
        }

    captured = {}

    def fake_interval(left, right, policy, *, transform, resample_source):
        captured["left_sources"] = [item.source for item in left]
        captured["right_sources"] = [item.source for item in right]
        captured["resample_source"] = resample_source
        assert callable(transform)
        return "independent-interval"

    monkeypatch.setattr(
        validation_module,
        "paired_channel_percentile_interval",
        fake_interval,
    )
    result = validation_module._summarize_causal(
        {
            "records": {
                "head_L0_H0": {
                    "semantic": [row(7)],
                    "structural": [row(11)],
                }
            }
        },
        {"head_L0_H0": ((0, 0),)},
        SimpleNamespace(
            numerical=SimpleNamespace(effect_floor=1.0e-12),
            bootstrap=BootstrapPolicy(),
        ),
        {
            "coordinates": SimpleNamespace(
                joint_sensitivity=np.ones((1, 1)),
                selectivity=np.zeros((1, 1)),
                active=np.ones((1, 1), dtype=bool),
            )
        },
        paired_channel_sources=False,
        source_resampling=(True, False),
    )

    assert captured == {
        "left_sources": [7],
        "right_sources": [11],
        "resample_source": (True, False),
    }
    assert result["intervals"]["interval"] == "independent-interval"
    assert result["intervals"]["pairing"] == (
        "graph-paired/channel-source-independent"
    )


class FakeAttention(torch.nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.projection = torch.nn.Linear(width, width, bias=False)

    def forward(self, pyg):
        count = int(pyg.x.shape[0])
        routed = self.projection(pyg.x).reshape(
            count, self.heads, self.head_width
        )
        pyg.attn = torch.ones(
            pyg.edge_index.shape[1],
            self.heads,
            1,
            device=pyg.x.device,
            dtype=pyg.x.dtype,
        )
        receiver = pyg.edge_index[1]
        for node in torch.unique(receiver):
            pyg.attn[receiver == node] /= float((receiver == node).sum())
        return routed, pyg.edge_attr


class FakeLayer(torch.nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attention = FakeAttention(width, heads)

    def forward(self, pyg):
        routed, edge = self.attention(pyg)
        pyg.x = pyg.x + routed.reshape_as(pyg.x)
        pyg.edge_attr = edge
        return pyg


class FakeHeads(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.graph_head = torch.nn.Sequential(
            torch.nn.LayerNorm(2 * width), torch.nn.Linear(2 * width, 1)
        )
        self.edge_head = torch.nn.Sequential(
            torch.nn.LayerNorm(5 * width + 1),
            torch.nn.Linear(5 * width + 1, 1),
        )


class FakeOfficialModel(torch.nn.Module):
    def __init__(self, width: int = 4, heads: int = 2):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, width)
        self.layers = torch.nn.ModuleList(
            [FakeLayer(width, heads), FakeLayer(width, heads)]
        )
        self.heads = FakeHeads(width)
        self.width = width

    def forward(self, batch):
        counts = [int(value) for value in batch.graph_num_nodes.tolist()]
        offsets = torch.tensor(
            [0, *np.cumsum(counts[:-1]).tolist()],
            device=batch.node_type.device,
            dtype=torch.long,
        )
        x = torch.cat(
            [
                self.embedding(batch.node_type[index, :count])
                for index, count in enumerate(counts)
            ]
        )
        full_edges = []
        edge_attr = []
        for index, count in enumerate(counts):
            nodes = torch.arange(count, device=x.device)
            source = nodes.repeat_interleave(count) + offsets[index]
            target = nodes.repeat(count) + offsets[index]
            full_edges.append(torch.stack((source, target)))
            edge_attr.append(
                torch.zeros(count * count, self.width, device=x.device)
            )
        orig_source = offsets[batch.edge_batch] + batch.edge_src
        orig_target = offsets[batch.edge_batch] + batch.edge_dst
        pyg = SimpleNamespace(
            x=x,
            edge_index=torch.cat(full_edges, dim=1),
            edge_attr=torch.cat(edge_attr),
            orig_edge_src=orig_source,
            orig_edge_dst=orig_target,
        )
        for layer in self.layers:
            pyg = layer(pyg)
        if batch.task_type == "graph_regression":
            pooled = []
            offset = 0
            for count in counts:
                states = pyg.x[offset : offset + count]
                pooled.append(
                    torch.cat((states.mean(0), states.max(0).values))
                )
                offset += count
            return self.heads.graph_head(torch.stack(pooled)).squeeze(-1)
        source = pyg.x[orig_source]
        target = pyg.x[orig_target]
        original_edge_state = torch.zeros(
            source.shape[0], self.width, device=x.device
        )
        readout = torch.cat(
            (
                source,
                target,
                torch.abs(source - target),
                source * target,
                original_edge_state,
                batch.edge_value[:, None],
            ),
            dim=-1,
        )
        return self.heads.edge_head(readout).squeeze(-1)


class FakeRunner:
    RRWP_STEPS = 16

    @staticmethod
    def collate_graphs(graphs):
        from graph_specialisation_metrics.methodology.graphbench import _load_module, default_runner_path

        return _load_module(default_runner_path()).collate_graphs(graphs)

    @staticmethod
    def denormalize_graph_target(prediction, stats):
        return prediction * float(stats["std"]) + float(stats["mean"])


def fake_runtime(graph: Graph) -> GraphBenchRuntime:
    model = FakeOfficialModel().eval()
    return GraphBenchRuntime(
        runner=FakeRunner(),
        model=model,
        eval_ds=[graph],
        donor_ds=[graph],
        splits={"val": [graph], "train": [graph]},
        device=torch.device("cpu"),
        seed=0,
        task_name="bipartite_matching_hard",
        target_stats=(
            {"mean": 5.0, "std": 2.0}
            if graph.task_type == "graph_regression"
            else None
        ),
        pos_weight=None,
        checkpoint={},
        cfg=SimpleNamespace(heads=2, hidden_dim=4),
        checks={},
        val_metric=None,
    )


def test_graphbench_backend_captures_chunked_vjps_and_native_patch_site():
    graph = matching_graph()
    runtime = fake_runtime(graph)
    task = get_task("graphbench_bipartite_matching_hard")
    backend = GraphBenchGritBackend(
        runtime, task, sigma=[1.0], jacobian_output_chunk=2
    )

    capture = backend.capture([graph, graph], require_grad=False)
    assert capture.prediction.shape == (2, 4)
    assert capture.transport[0].shape == (2, 3, 2, 2)
    assert capture.final_state.shape == (2, 4, 21)

    clean = backend.clean_jacobians(graph)
    assert clean.transport.shape == (4, 2, 3, 2, 2)
    assert clean.final_state.shape == (4, 21)
    assert torch.isfinite(clean.transport).all()

    ablated, _, target = backend.ablate([graph], ((0, 0),))
    assert ablated.shape == target.shape == (1, 4)
    replacements = backend.replacement_batch(capture, [0])
    patched, _, _ = backend.patch(graph, replacements, ((0, 0),))
    assert torch.allclose(patched, capture.prediction[0:1], atol=1.0e-6)


def test_graphbench_replica_specific_head_batching_matches_serial_patch_and_ablation():
    graph = matching_graph()
    backend = GraphBenchGritBackend(
        fake_runtime(graph),
        get_task("graphbench_bipartite_matching_hard"),
        sigma=[1.0],
        jacobian_output_chunk=2,
    )
    capture = backend.capture([graph, graph], require_grad=False)
    assignments = ((0, 0), (1, 1))
    replacements = backend.replacement_batch(capture, [0, 1])

    _, z_batched_patch, _ = backend.patch_individual_heads(
        [graph, graph], replacements, assignments
    )
    serial_patch = torch.cat(
        [
            backend.patch(
                graph,
                tuple(
                    layer[index]
                    for layer in capture.transport
                ),
                (assignment,),
            )[1]
            for index, assignment in enumerate(assignments)
        ],
        dim=0,
    )
    _, z_batched_ablation, _ = backend.ablate_individual_heads(
        [graph, graph], assignments
    )
    serial_ablation = torch.cat(
        [
            backend.ablate([graph], (assignment,))[1]
            for assignment in assignments
        ],
        dim=0,
    )

    assert torch.allclose(z_batched_patch, serial_patch, atol=1.0e-6)
    assert torch.allclose(z_batched_ablation, serial_ablation, atol=1.0e-6)


@pytest.mark.parametrize(
    ("channel", "arm"),
    (
        ("semantic", None),
        ("structural", "rrwp_copy"),
        ("structural", "rrwp_transpose"),
        ("structural", "complete_pe_copy"),
        ("structural", "complete_pe_transpose"),
    ),
)
def test_pe_refinement_causal_geometry_and_taylor_audit_are_complete(
    tmp_path,
    channel,
    arm,
):
    graph = six_node_matching_graph()
    runtime = fake_runtime(graph)
    # Real GraphBench batches pad each graph to the maximum edge-output width in
    # the validation split. The Taylor Jacobian remains graph-local.
    runtime.eval_ds.append(
        SimpleNamespace(
            edge_index=torch.empty(2, 18, dtype=torch.long),
        )
    )
    task = get_task("graphbench_bipartite_matching_hard")
    backend = GraphBenchGritBackend(
        runtime, task, sigma=[1.0], jacobian_output_chunk=4
    )
    config = PERefinementConfig(
        output_dir=str(tmp_path / "analysis"),
        training_output_root=str(tmp_path / "training"),
        dataset_root=str(tmp_path / "dataset"),
        pe_cache_root=str(tmp_path / "pe"),
        runner_path=str(tmp_path / "runner.py"),
        sizes=PERefinementSizes(
            discovery_graphs=1,
            refinement_graphs=1,
            confirmation_graphs=1,
            clean_ablation_graphs=1,
            semantic_donor_graphs=1,
            semantic_sources_per_graph=1,
            donors_per_source=2,
            taylor_graphs=1,
        ),
        accelerator="cpu",
        head_batch_size=2,
    )
    prepared = SimpleNamespace(
        config=config,
        runtime=runtime,
        backend=backend,
        donor_pool=GraphBenchEdgeDonorPool([(0, graph)]),
        progress=SimpleNamespace(emit=lambda *_args, **_kwargs: None),
    )
    manifest = (
        semantic_event_manifest(prepared, "refinement", (0,))
        if channel == "semantic"
        else structural_pair_manifest(prepared, "refinement", (0,))
    )
    rows = manifest[0]

    if channel == "structural":
        assert len(rows) == 12
        assert all(row["realised_donor_count"] == 2 for row in rows)
        assert all(
            len(
                {
                    row["donor_node"]
                    for row in rows
                    if row["source"] == source
                }
            )
            == 2
            for source in range(6)
        )

    result = _causal_graph(
        prepared,
        graph_id=0,
        channel=channel,
        arm=arm,
        stage="refinement",
        manifest_rows=rows,
        clean_taylor=backend.clean_jacobians(graph),
    )

    assert result["controlled"].all()
    assert result["endpoints"]["G_c"].shape == (4, len(rows))
    assert np.isfinite(result["endpoints"]["G_c"]).all()
    assert result["taylor"]["predicted"].shape == (4, len(rows), 12)
    assert result["taylor"]["exact"].shape == (4, len(rows), 12)
    assert result["taylor"]["padded_output_width"] == 18
    assert result["taylor"]["actual_output_width"] == 12
    assert result["taylor"]["padding_max"] == 0.0
    assert np.isfinite(result["taylor"]["predicted"]).all()
    assert np.isfinite(result["taylor"]["exact"]).all()
    estimable = result["taylor"]["estimable"]
    assert np.isfinite(result["taylor"]["relative_error"][estimable]).all()


def test_pe_refinement_vectorized_permutation_respects_layer_strata():
    x = np.asarray([1.0, 3.0, 2.0, 4.0])
    y = np.asarray([4.0, 2.0, 3.0, 1.0])
    # A singleton stratum cannot be permuted, so every null replicate must equal
    # the observed rank correlation exactly.
    null = _permuted_spearman_values(
        x,
        y,
        np.arange(4),
        replicates=32,
        rng=np.random.default_rng(123),
    )

    from scipy.stats import spearmanr

    assert np.allclose(null, float(spearmanr(x, y).statistic))


def test_pe_refinement_candidate_aggregation_keeps_public_association_names(
    tmp_path,
    monkeypatch,
):
    config = PERefinementConfig(
        output_dir=str(tmp_path / "analysis"),
        training_output_root=str(tmp_path / "training"),
        dataset_root=str(tmp_path / "dataset"),
        pe_cache_root=str(tmp_path / "pe"),
        runner_path=str(tmp_path / "runner.py"),
        sizes=PERefinementSizes(
            discovery_graphs=1,
            refinement_graphs=1,
            confirmation_graphs=1,
            clean_ablation_graphs=1,
            semantic_donor_graphs=1,
            semantic_sources_per_graph=1,
            donors_per_source=1,
            taylor_graphs=1,
            permutation_replicates=8,
        ),
        accelerator="cpu",
    )

    def fake_seed_analysis(_config, *, seed, arm, score_system, split):
        del _config, arm, score_system, split
        rho = 0.1 + 0.1 * int(seed)
        result = {
            "seed": int(seed),
            "associations": {
                "registered_name": {
                    "spearman": {"rho": rho},
                }
            },
        }
        vectors = {
            "registered_name": {
                "x": np.asarray([1.0, 2.0, 3.0, 4.0]),
                "y": np.asarray([1.0, 3.0, 2.0, 4.0]),
                "layer": np.asarray([0, 0, 1, 1]),
            }
        }
        return result, vectors

    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.graphbench_pe_refinement."
        "_seed_candidate_analysis",
        fake_seed_analysis,
    )
    candidate, _ = _candidate_public_summary(
        config,
        arm="rrwp_copy",
        score_system="mass",
        split="refinement",
    )

    assert set(candidate["population_associations"]) == {"registered_name"}
    assert candidate["population_associations"]["registered_name"]["n_seeds"] == 4


def test_confirmation_figure_renderer_accepts_only_the_locked_candidate(tmp_path):
    per_seed = []
    for seed in range(4):
        per_seed.append(
            {
                "seed": seed,
                "scores": {
                    "structural_normalized": np.asarray([[0.8, 1.2], [0.9, 1.1]]),
                    "semantic_normalized": np.asarray([[1.1, 0.9], [1.2, 0.8]]),
                    "joint_sensitivity": np.ones((2, 2)),
                    "selectivity": np.asarray([[0.2, -0.2], [0.1, -0.1]]),
                    "active": np.ones((2, 2), dtype=bool),
                    },
                    "causal": {
                        "semantic": {
                            "G_c": np.asarray([0.5, 0.4, 0.3, 0.2]),
                        },
                        "structural": {
                        "P_gross_matched": np.asarray([0.4, 0.5, 0.6, 0.7]),
                        "P_gross_mismatch": np.asarray([0.1, 0.2, 0.2, 0.3]),
                        "G_c": np.asarray([0.3, 0.3, 0.4, 0.4]),
                    },
                    "gross_total_for_J": np.asarray([0.3, 0.4, 0.5, 0.6]),
                    "gross_contrast_for_D_rel": np.asarray([0.2, -0.2, 0.1, -0.1]),
                },
            }
        )
    population = {
        name: {"fisher_mean_rho": 0.5}
        for name in (
            "S_structural_vs_P_matched",
            "S_structural_vs_P_mismatch",
            "S_structural_vs_G_structural",
            "S_semantic_vs_G_semantic",
            "S_semantic_vs_G_structural_control",
            "S_structural_vs_G_semantic_control",
            "J_vs_gross_total",
            "D_rel_vs_gross_contrast",
        )
    }
    candidate = {
        "candidate_id": "complete_pe_transpose__coherent",
        "arm": "complete_pe_transpose",
        "score_system": "coherent",
        "per_seed": per_seed,
        "population_associations": population,
    }
    selection = {
        "candidate_id": candidate["candidate_id"],
        "registered_rank": 1,
        "structural_score_adjusted_rho": 0.5,
        "D_rel_causal_contrast_rho": 0.5,
        "median_structural_taylor_cosine_across_seeds": 0.8,
        "median_structural_carrier_coherence_across_seeds": 0.7,
    }

    saved = render_refinement_figures(
        {
            "split": "confirmation",
            "seeds": [0, 1, 2, 3],
            "candidates": [candidate],
            "selection_table": [selection],
        },
        tmp_path,
    )

    assert saved
    assert all(Path(path).exists() for paths in saved.values() for path in paths)


def test_pe_refinement_clean_ablation_resumes_its_atomic_graph_shard(
    tmp_path,
    monkeypatch,
):
    graph = matching_graph()
    runtime = fake_runtime(graph)
    task = get_task("graphbench_bipartite_matching_hard")
    backend = GraphBenchGritBackend(
        runtime, task, sigma=[1.0], jacobian_output_chunk=2
    )
    config = PERefinementConfig(
        output_dir=str(tmp_path / "analysis"),
        training_output_root=str(tmp_path / "training"),
        dataset_root=str(tmp_path / "dataset"),
        pe_cache_root=str(tmp_path / "pe"),
        runner_path=str(tmp_path / "runner.py"),
        sizes=PERefinementSizes(
            discovery_graphs=1,
            refinement_graphs=1,
            confirmation_graphs=1,
            clean_ablation_graphs=1,
            semantic_donor_graphs=1,
            semantic_sources_per_graph=1,
            donors_per_source=2,
            taylor_graphs=1,
        ),
        accelerator="cpu",
        head_batch_size=2,
    )
    prepared = PreparedPERefinement(
        config=config,
        runtime=runtime,
        backend=backend,
        task=task,
        donor_pool=GraphBenchEdgeDonorPool([(0, graph)]),
        splits=SimpleNamespace(
            clean_ablation=(0,),
            fingerprint="fake-split",
        ),
        checkpoint=tmp_path / "best.pt",
        checkpoint_sha="fake-checkpoint",
        seed=0,
        progress=SimpleNamespace(emit=lambda *_args, **_kwargs: None),
    )
    first = run_common_ablation(prepared)

    monkeypatch.setattr(
        backend,
        "capture",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("resume should not rerun clean capture")
        ),
    )
    resumed = run_common_ablation(prepared)

    assert np.array_equal(
        resumed["prediction_movement"], first["prediction_movement"]
    )


def test_pe_refinement_causal_component_resumes_with_manifest_validation(tmp_path):
    graph = six_node_matching_graph()
    runtime = fake_runtime(graph)
    task = get_task("graphbench_bipartite_matching_hard")
    backend = GraphBenchGritBackend(
        runtime, task, sigma=[1.0], jacobian_output_chunk=4
    )
    config = PERefinementConfig(
        output_dir=str(tmp_path / "analysis"),
        training_output_root=str(tmp_path / "training"),
        dataset_root=str(tmp_path / "dataset"),
        pe_cache_root=str(tmp_path / "pe"),
        runner_path=str(tmp_path / "runner.py"),
        sizes=PERefinementSizes(
            discovery_graphs=1,
            refinement_graphs=1,
            confirmation_graphs=1,
            clean_ablation_graphs=1,
            semantic_donor_graphs=1,
            semantic_sources_per_graph=1,
            donors_per_source=2,
            taylor_graphs=1,
        ),
        accelerator="cpu",
        head_batch_size=2,
    )
    prepared = PreparedPERefinement(
        config=config,
        runtime=runtime,
        backend=backend,
        task=task,
        donor_pool=GraphBenchEdgeDonorPool([(0, graph)]),
        splits=SimpleNamespace(
            causal=(0, 0),
            fingerprint="fake-split",
        ),
        checkpoint=tmp_path / "best.pt",
        checkpoint_sha="fake-checkpoint",
        seed=0,
        progress=SimpleNamespace(emit=lambda *_args, **_kwargs: None),
    )
    manifest = structural_pair_manifest(prepared, "refinement", (0,))
    clean = backend.clean_jacobians(graph)

    first = _run_causal_component(
        prepared,
        channel="structural",
        arm="complete_pe_copy",
        split="refinement",
        manifest=manifest,
        clean_taylor={0: clean},
    )
    resumed = _run_causal_component(
        prepared,
        channel="structural",
        arm="complete_pe_copy",
        split="refinement",
        manifest=manifest,
        clean_taylor={0: clean},
    )

    assert first["manifest_hash"] == resumed["manifest_hash"]
    assert first["support"] == resumed["support"]


def test_graphbench_flow_backend_replays_exact_mean_max_readout():
    graph = flow_graph()
    runtime = fake_runtime(graph)
    task = get_task("graphbench_flow_hard")
    backend = GraphBenchGritBackend(
        runtime, task, sigma=[2.0], jacobian_output_chunk=2
    )

    capture = backend.capture([graph, graph], require_grad=False)
    assert capture.prediction.shape == (2, 1)
    assert capture.final_state.shape == (2, 3, 4)
    clean = backend.clean_jacobians(graph)
    assert clean.final_state.shape == (1, 3, 4)

    replay = backend.loss_from_states(graph, capture.target[0:1])
    replay_loss = replay(capture.final_state[0:1])
    actual_loss = backend.loss_per_graph(
        capture.prediction[0:1], capture.target[0:1]
    )
    assert torch.allclose(replay_loss, actual_loss, atol=1.0e-6)


@pytest.mark.parametrize(
    ("graph_factory", "task_name", "sigma"),
    (
        (matching_graph, "graphbench_bipartite_matching_hard", [1.0]),
        (flow_graph, "graphbench_flow_hard", [2.0]),
    ),
)
def test_graphbench_clean_jacobians_fall_back_exactly_when_vmap_is_unsupported(
    monkeypatch,
    capsys,
    graph_factory,
    task_name,
    sigma,
):
    graph = graph_factory()
    task = get_task(task_name)
    runtime = fake_runtime(graph)
    expected_backend = GraphBenchGritBackend(
        runtime,
        task,
        sigma=sigma,
        jacobian_output_chunk=2,
    )
    expected = expected_backend.clean_jacobians(graph)

    backend = GraphBenchGritBackend(
        runtime,
        task,
        sigma=sigma,
        jacobian_output_chunk=2,
    )
    original_grad = torch.autograd.grad
    calls = {"batched": 0, "sequential": 0}

    def vmap_incompatible_grad(*args, **kwargs):
        if kwargs.get("is_grads_batched", False):
            calls["batched"] += 1
            raise RuntimeError(
                "vmap: aten::scatter_(self, *extra_args) is not possible because "
                "an official operator has no compatible batching rule"
            )
        calls["sequential"] += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", vmap_incompatible_grad)
    actual = backend.clean_jacobians(graph)
    repeated = backend.clean_jacobians(graph)

    assert backend.jacobian_engine == "sequential_vjp"
    assert calls["batched"] == 1
    assert calls["sequential"] == int(expected.capture.z.numel())
    output = capsys.readouterr().out
    assert "using exact sequential VJPs" in output
    assert "reusing detached clean Jacobians" in output
    assert torch.allclose(actual.transport, expected.transport, atol=1.0e-6)
    assert torch.allclose(actual.final_state, expected.final_state, atol=1.0e-6)
    assert repeated is actual
    assert torch.allclose(repeated.transport, expected.transport, atol=1.0e-6)
    assert torch.allclose(repeated.final_state, expected.final_state, atol=1.0e-6)


def test_graphbench_clean_jacobians_do_not_hide_unrelated_autograd_failures(
    monkeypatch,
):
    graph = matching_graph()
    backend = GraphBenchGritBackend(
        fake_runtime(graph),
        get_task("graphbench_bipartite_matching_hard"),
        sigma=[1.0],
        jacobian_output_chunk=2,
    )

    def unrelated_failure(*_args, **_kwargs):
        raise RuntimeError("CUDA launch failure unrelated to vmap")

    monkeypatch.setattr(torch.autograd, "grad", unrelated_failure)
    with pytest.raises(RuntimeError, match="CUDA launch failure"):
        backend.clean_jacobians(graph)
    assert backend.jacobian_engine == "auto"


def test_graphbench_score_and_carriage_components_resume_from_graph_shards(tmp_path):
    evaluation = [matching_graph() for _ in range(7)]
    donors = [
        matching_graph(values=(3.0 + index, 3.0 + index, 5.0 + index, 5.0 + index))
        for index in range(8)
    ]
    runtime = fake_runtime(evaluation[0])
    runtime.eval_ds = evaluation
    runtime.donor_ds = donors
    task = get_task("graphbench_bipartite_matching_hard")
    backend = GraphBenchGritBackend(
        runtime, task, sigma=[1.0], jacobian_output_chunk=2
    )
    split = SplitManifest(
        discovery=(0, 1, 2),
        causal=(3, 4),
        clean_ablation=(5, 6),
        semantic_donor_pool=tuple(range(8)),
        same_index_space=False,
        seed=31_415,
    )
    prepared = PreparedTask(
        task=task,
        runtime=runtime,
        backend=backend,
        output_dir=tmp_path / task.name / "seed_0",
        checkpoint=tmp_path / "best.pt",
        checkpoint_sha="fake-checkpoint",
        sigma=np.asarray([1.0]),
        splits=split,
        donor_pool=GraphBenchEdgeDonorPool(list(enumerate(donors))),
        progress=None,
    )
    config = MethodologyConfig(
        output_dir=str(tmp_path),
        tasks=(task.name,),
        train_seeds=(0,),
        phases=("scores", "carriage"),
        sizes=RunSizes(
            discovery_graphs=3,
            causal_graphs=2,
            clean_ablation_graphs=2,
            semantic_donor_graphs=8,
            sources_per_graph=3,
            donors_per_source=2,
        ),
        accelerator="cpu",
        compute_beneficial_carriage=False,
    )
    score_plan = _stage_plan(prepared, config, "scores")
    scores = run_scores(prepared, config, plan=score_plan)
    clean_shards = list(
        (prepared.output_dir / "cache" / "clean_jacobians").glob("graph_*.pt")
    )
    assert len(clean_shards) == 3
    backend._clean_jacobian_cache.clear()

    def unexpected_recomputation(_data):
        raise AssertionError("carriage should resume clean Jacobians from graph shards")

    backend.clean_jacobians = unexpected_recomputation
    carriage_plan = _stage_plan(prepared, config, "carriage")
    carriage = run_carriage(prepared, config, plan=carriage_plan)

    score_shards = list(
        (prepared.output_dir / "cache" / "scores" / "semantic").glob("graph_*.pt")
    )
    carriage_shards = list(
        (prepared.output_dir / "cache" / "carriage" / "structural").glob("graph_*.pt")
    )
    assert len(score_shards) == 3
    assert len(carriage_shards) == 3
    assert scores["interval_pairing"] == "graph-paired/channel-source-independent"
    assert carriage["channels"]["structural"]["resample_source"] is False

    resumed_scores = run_scores(prepared, config, plan=score_plan)
    resumed_carriage = run_carriage(prepared, config, plan=carriage_plan)
    assert resumed_scores["manifest_hash"] == scores["manifest_hash"]
    assert resumed_carriage["manifest_hash"] == carriage["manifest_hash"]


def test_model_free_finalizer_owns_shared_four_seed_summaries(
    tmp_path,
    monkeypatch,
):
    tasks = (
        "graphbench_bipartite_matching_hard",
        "graphbench_flow_hard",
    )
    config = MethodologyConfig(
        output_dir=str(tmp_path),
        tasks=tasks,
        train_seeds=(0, 1, 2, 3),
        phases=("figures",),
        accelerator="cpu",
    )
    for task in tasks:
        for seed in config.train_seeds:
            output = tmp_path / task / f"seed_{seed}"
            output.mkdir(parents=True)
            (output / "audits.json").write_text(
                json.dumps({"findings": []}),
                encoding="utf-8",
            )
            if task == tasks[0] and seed == 0:
                figure_dir = output / "figures"
                figure_dir.mkdir()
                figure = figure_dir / "complete.png"
                metadata = figure_dir / "complete.metadata.json"
                figure.write_bytes(b"complete")
                metadata.write_text("{}", encoding="utf-8")
                (output / "figures.json").write_text(
                    json.dumps({"complete": [str(figure)]}),
                    encoding="utf-8",
                )

    def artifact(path):
        path = Path(path)
        seed = int(path.parents[2].name.removeprefix("seed_"))
        task = path.parents[3].name
        stage = path.parent.name
        if stage == "scores":
            value = {
                "channels": {
                    "semantic": {"raw": np.full((1, 2), seed + 1.0)},
                    "structural": {"raw": np.full((1, 2), seed + 2.0)},
                },
                "coordinates": SimpleNamespace(
                    selectivity=np.asarray([[0.1, -0.1]]),
                    active=np.asarray([[True, True]]),
                ),
            }
        elif stage == "causal":
            value = {"associations": {}}
        else:
            value = {}
        return SimpleNamespace(
            path=path,
            metadata={
                "contract": {
                    "task": task,
                    "train_seed": seed,
                    "protocol_fingerprint": config.fingerprint,
                    "repository_commit": "worker-commit",
                },
                "contract_fingerprint": f"{task}:{seed}:{stage}",
            },
            value=value,
        )

    rendered = []
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.runner.load_cache_artifact_file",
        artifact,
    )
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.runner.render_cached_figures",
        lambda _config, task, seed: rendered.append((task, seed)) or {"ok": []},
    )

    results = finalize_cached_run(config)

    assert len(results) == 8
    assert len(rendered) == 7
    assert (tasks[0], 0) not in rendered
    assert results[f"{tasks[0]}:seed0"]["figures"]["complete"]
    assert len(json.loads((tmp_path / "index.json").read_text())["runs"]) == 8
    for task in tasks:
        population = json.loads(
            (tmp_path / task / "population.json").read_text(encoding="utf-8")
        )
        assert [row["seed"] for row in population["seed_estimates"]] == [0, 1, 2, 3]
        assert population["population_interval"]["level"] == "training seed"


def test_seed_worker_writes_no_shared_root_or_task_summaries(tmp_path, monkeypatch):
    task = "graphbench_bipartite_matching_hard"
    config = MethodologyConfig(
        output_dir=str(tmp_path),
        tasks=(task, "graphbench_flow_hard"),
        train_seeds=(0, 1, 2, 3),
        phases=("scores",),
        accelerator="cpu",
    )
    output = tmp_path / task / "seed_2"
    prepared = SimpleNamespace()
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.runner.prepare_task",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        "graph_specialisation_metrics.methodology.runner.run_prepared",
        lambda *_args, **_kwargs: {
            "task": task,
            "seed": 2,
            "output_dir": str(output),
            "scores": {},
            "carriage": None,
            "causal": None,
            "figures": None,
            "audit_findings": [],
            "headline_eligible": True,
        },
    )

    run_worker(config, task, 2)

    assert (output / "protocol.json").exists()
    assert (output / "audits.json").exists()
    assert not (tmp_path / "protocol.json").exists()
    assert not (tmp_path / "index.json").exists()
    assert not (tmp_path / task / "population.json").exists()
