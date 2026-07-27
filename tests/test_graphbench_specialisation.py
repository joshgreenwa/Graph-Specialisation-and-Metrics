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
    structural_rrwp_swap,
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
