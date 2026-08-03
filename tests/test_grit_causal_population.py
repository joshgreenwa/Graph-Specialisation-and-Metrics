from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from graph_specialisation_metrics.methodology.backend import CanonicalGritBackend
from graph_specialisation_metrics.methodology.graphormer_causal_analysis import (
    FocusedExecution,
)
from graph_specialisation_metrics.methodology.graphormer_causal_population import (
    run_population_events,
)
from graph_specialisation_metrics.methodology.grit_causal_population import (
    DENSE_GRIT_TASKS,
    GRIT_POPULATION_CAUSAL_VERSION,
    _population_gate_record,
    _prepare_population_cache,
    production_config,
    run,
)
from graph_specialisation_metrics.methodology.runner import _cache


def test_dense_grit_population_config_uses_disjoint_paper_populations(tmp_path):
    config = production_config(
        output_dir=str(tmp_path),
        accelerator="cpu",
        graphs_per_batch=8,
    )
    config.validate()
    assert config.tasks == DENSE_GRIT_TASKS
    assert config.seeds_for("zinc") == (42,)
    assert config.seeds_for("qm9_gap_dense") == (42,)
    assert config.sizes.discovery_graphs == 256
    assert config.sizes.causal_graphs == 256
    assert config.sizes.clean_ablation_graphs == 256
    assert config.sizes.semantic_donor_graphs == 2_000
    assert config.execution.graphs_per_batch == 8


def test_dense_grit_population_config_rejects_sparse_controls(tmp_path):
    with pytest.raises(ValueError, match="dense causal population tasks"):
        production_config(
            output_dir=str(tmp_path),
            tasks=("zinc_1hop",),
            accelerator="cpu",
        )


def test_dense_grit_population_defaults_use_architecture_adapted_gate():
    parameters = inspect.signature(run).parameters
    assert parameters["population_head_pairs"].default == 8
    assert parameters["population_minimum_pairs"].default == 6


def test_population_gate_cache_lineage_does_not_change_selection_identity():
    gate = {"version": GRIT_POPULATION_CAUSAL_VERSION, "policy": {"head_pairs": 8}}
    resumed = {
        **gate,
        "_cache_lineage": {
            "version": "population-event-retarget-v1",
            "source_event_manifest_hashes": ("old",),
        },
    }
    assert _population_gate_record(resumed) == gate


def test_population_events_retargets_complete_cached_head_rows_without_a_plan():
    targets = ((0, 0), (0, 1), (0, 2), (0, 3))
    gate = {
        "status": "estimable",
        "heads": {
            "semantic": (targets[0],),
            "structural": (targets[1],),
            "j_matched_null": targets[2:],
        },
    }

    class MemoryCache:
        def __init__(self, values=()):
            self.values = dict(values)

        def load(self, stage, name, *, strict=False):
            del strict
            return self.values.get((stage, name))

        def save(self, stage, name, value):
            self.values[(stage, name)] = value

    source_values = []
    for channel in ("semantic", "structural"):
        source_values.append(
            (
                (f"focused_population/events/{channel}", "graph_000007"),
                {
                    "graph": 7,
                    "channel": channel,
                    "rows": tuple(
                        {"head": head, "channel": channel} for head in (*targets, (1, 9))
                    ),
                    "event_count": 48,
                    "controlled_event_count": 40,
                    "uncontrolled_event_count": 8,
                    "clean_same_condition_patch_max": 0.0,
                    "event_same_condition_patch_max": 0.0,
                },
            )
        )
    current = MemoryCache()
    source = MemoryCache(source_values)
    rows, returned_cache, audits = run_population_events(
        SimpleNamespace(progress=None),
        SimpleNamespace(resume=True, force=False),
        {},
        gate,
        execution=FocusedExecution(head_batch_size=2, event_batch_size=2),
        plan={},
        cache=current,
        graph_ids=(7,),
        reuse_caches=(source,),
        reuse_legacy=False,
    )

    assert returned_cache is current
    assert len(rows) == 2 * len(targets)
    assert {_tuple for _tuple in (tuple(row["head"]) for row in rows)} == set(targets)
    assert len(audits) == 2
    assert all(audit["reused_head_count"] == len(targets) for audit in audits)
    assert all(audit["computed_head_count"] == 0 for audit in audits)


def test_population_gate_transition_preserves_old_event_cache_as_reuse_lineage(tmp_path):
    pytest.importorskip("torch")
    config = production_config(
        output_dir=str(tmp_path),
        tasks=("zinc",),
        accelerator="cpu",
    )
    task = SimpleNamespace(
        name="zinc",
        adapter_version="test-adapter-v1",
        output=SimpleNamespace(representation="evaluation_regression"),
        raw_score_system="mass",
        semantic_source_kind="node",
        backend_kind="grit",
    )
    prepared = SimpleNamespace(
        task=task,
        checkpoint_sha="checkpoint",
        grit=SimpleNamespace(sc=SimpleNamespace(seed=42)),
        backend=SimpleNamespace(geometry={"layers": 10, "heads": 8}),
        sigma=(1.0,),
        splits=SimpleNamespace(fingerprint="split", causal=(7,)),
    )
    old_gate = {
        "version": GRIT_POPULATION_CAUSAL_VERSION,
        "status": "estimable",
        "policy": {"head_pairs": 16, "minimum_pairs": 12},
        "specialist_pairs": (),
        "null_pairs": (),
    }
    new_gate = {
        **old_gate,
        "policy": {"head_pairs": 8, "minimum_pairs": 6},
    }
    old_cache = _cache(prepared, config, {}, manifest_hash="old-population")
    gate_path = old_cache.save("focused_population", "gate", old_gate)
    old_cache.save(
        "focused_population/events/semantic",
        "graph_000007",
        {"graph": 7, "rows": ()},
    )

    plan, new_cache, saved_gate, reuse_caches = _prepare_population_cache(
        prepared,
        config,
        new_gate,
        gate_path,
    )

    assert plan == {}
    assert saved_gate["policy"] == new_gate["policy"]
    assert saved_gate["_cache_lineage"]["source_event_manifest_hashes"] == (
        "old-population",
    )
    assert reuse_caches[0].contract.event_manifest_hash == "old-population"
    assert new_cache.contract.event_manifest_hash != "old-population"
    assert new_cache.load("focused_population", "gate") == saved_gate


def test_canonical_grit_backend_assigns_heads_by_graph_with_variable_node_counts():
    torch = pytest.importorskip("torch")
    pyg_data = pytest.importorskip("torch_geometric.data")

    class RoutedLayer(torch.nn.Module):
        def forward(self, routed):
            return routed, None

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList((RoutedLayer(), RoutedLayer()))

        def forward(self, batch):
            routed = batch.x.reshape(-1, 2, 1).float()
            for layer in self.layers:
                routed, _edge = layer(routed)
            prediction = routed.new_zeros((int(batch.num_graphs), 1))
            prediction.index_add_(0, batch.batch, routed.sum(dim=(1, 2), keepdim=False)[:, None])
            return prediction, prediction.new_zeros(prediction.shape)

    class IdentityOutput:
        @staticmethod
        def transform(prediction, sigma):
            del sigma
            return prediction

    model = Model()
    runtime = SimpleNamespace(
        L=2,
        H=2,
        dh=1,
        dim_h=2,
        device=torch.device("cpu"),
        model=model,
        attn_layers=model.layers,
    )
    task = SimpleNamespace(output=IdentityOutput(), virtual_node=False)
    backend = CanonicalGritBackend(runtime, task, sigma=(1.0,))
    graphs = (
        pyg_data.Data(x=torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        pyg_data.Data(x=torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])),
    )
    assignments = ((0, 0), (1, 1))

    _prediction, ablated, _target = backend.ablate_individual_heads(
        graphs,
        assignments,
    )
    assert np.allclose(ablated.detach().numpy().reshape(-1), (6.0, 90.0))

    replacements = tuple(torch.full((5, 2, 1), value) for value in (100.0, -5.0))
    _prediction, patched, _target = backend.patch_individual_heads(
        graphs,
        replacements,
        assignments,
    )
    assert np.allclose(patched.detach().numpy().reshape(-1), (206.0, 75.0))
    assert GRIT_POPULATION_CAUSAL_VERSION.startswith("grit-dense-")
