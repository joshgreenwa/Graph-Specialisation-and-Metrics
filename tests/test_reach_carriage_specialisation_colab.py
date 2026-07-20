from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "synthetic"
        / "training"
        / "reach_carriage_specialisation_colab.py"
    )
    name = "reach_carriage_specialisation_colab_test_module"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config(module):
    return module.Config(
        seeds=(0,),
        carriage_graphs=2,
        carriage_donors=1,
        score_graphs=2,
        score_donors=1,
        score_batch_size=1,
        ablation_graphs=4,
    )


def test_degree_and_cut_matched_topologies():
    module = _load_module()
    cfg = _config(module)
    for topology, expected_cut in (("thin", 2), ("wide", 8)):
        adj, _, distances = module.relabel_graph(cfg, topology, np.random.default_rng(7))
        assert np.all(adj.sum(axis=1) == 4)
        assert int(adj[: cfg.cluster_size, cfg.cluster_size :].sum()) == expected_cut
        assert np.all(distances >= 0)
    thin, _, _ = module.relabel_graph(cfg, "thin", np.random.default_rng(8))
    wide, _, _ = module.relabel_graph(cfg, "wide", np.random.default_rng(8))
    assert int(thin.sum()) == int(wide.sum())


def test_one_generator_enforces_reach_and_load_conditions():
    module = _load_module()
    cfg = _config(module)
    for condition, topology, distance, rank in (
        ("reach", "thin", 5, 1),
        ("load", "thin", 3, 6),
        ("load", "wide", 3, 6),
    ):
        batch = module.make_batch(
            cfg,
            5,
            seed=11,
            condition=condition,
            topology=topology,
            distance=distance,
            rank=rank,
        )
        assert torch.equal(batch.qmask.sum(dim=1), torch.full((5,), rank))
        for graph in range(len(batch)):
            graph_distances = module.all_pairs_distances(np.asarray(batch.adj[graph]))
            queries = batch.qmask[graph].nonzero(as_tuple=False).flatten()
            for query in queries:
                source = int(batch.source_for_query[graph, query])
                assert int(graph_distances[int(query), source]) == distance
                value = int(torch.argmax(batch.x[graph, source, : cfg.classes]))
                assert int(batch.y[graph, query]) == value
                if condition == "load":
                    assert int(query) < cfg.cluster_size <= source


def test_maximum_load_pair_matching_is_robust_across_relabellings():
    module = _load_module()
    cfg = _config(module)
    for seed in range(100):
        for topology in ("thin", "wide"):
            batch = module.make_batch(
                cfg,
                1,
                seed=seed,
                condition="load",
                topology=topology,
                distance=cfg.load_distance,
                rank=max(cfg.ranks),
            )
            assert int(batch.qmask.sum()) == max(cfg.ranks)


def test_khop_support_is_nested_and_parameter_reach_is_declared():
    module = _load_module()
    cfg = _config(module)
    batch = module.make_batch(cfg, 2, seed=21, condition="reach", topology="thin", distance=2, rank=1)
    supports = [module.khop_support(batch.adj, radius) for radius in (0, 1, 2, 3, None)]
    for left, right in zip(supports, supports[1:]):
        assert bool((left & ~right).sum() == 0)
        assert int(left.sum()) < int(right.sum())
    assert module.theoretical_reach(cfg, "1hop") == cfg.layers
    assert module.theoretical_reach(cfg, "2hop") == max(cfg.distances)


def test_checkpoint_gate_uses_requested_split():
    module = _load_module()
    cfg = _config(module)
    row = {
        "condition": "reach",
        "topology": "thin",
        "distance": 2,
        "rank": 1,
        "accuracy": 0.9,
    }
    payload = {
        "best_validation": [row],
        "heldout": [{**row, "accuracy": 0.8}],
    }
    assert module.checkpoint_gate(cfg, "1hop", payload, split="best_validation") == 0.9
    assert module.checkpoint_gate(cfg, "1hop", payload, split="heldout") == 0.8


def test_interventions_freeze_mask_and_change_only_declared_substrate():
    module = _load_module()
    cfg = _config(module)
    clean = module.make_batch(cfg, 2, seed=31, condition="reach", topology="thin", distance=5, rank=1)
    semantic = module.make_all_source_replicas(cfg, clean, factor="semantic", donors=1, seed=32)
    structural = module.make_all_source_replicas(cfg, clean, factor="structural", donors=1, seed=33)
    assert bool((semantic.x[1:] != semantic.x[0]).any())
    assert torch.equal(semantic.adj[1], semantic.adj[0])
    assert torch.equal(semantic.rrwp[1], semantic.rrwp[0])
    assert torch.equal(structural.x[1], structural.x[0])
    assert bool((structural.adj[1:] != structural.adj[0]).any())
    assert bool((structural.rrwp[1:] != structural.rrwp[0]).any())

    # Structural specialisation uses the same RRWP transposition but freezes trained support.
    frozen = module.make_target_replicas(
        cfg,
        clean,
        factor="structural",
        donors=1,
        seed=35,
        structural_support="frozen",
    )
    assert torch.equal(frozen.adj[1], frozen.adj[0])
    assert bool((frozen.rrwp[1] != frozen.rrwp[0]).any())

    for factor in ("semantic", "structural"):
        no_op = module.make_all_source_replicas(
            cfg, clean, factor=factor, donors=1, seed=34, no_op=True
        )
        reps = 1 + cfg.n
        for graph in range(len(clean)):
            start = graph * reps
            assert torch.equal(no_op.x[start : start + reps], no_op.x[start].expand(reps, -1, -1))
            assert torch.equal(no_op.rrwp[start : start + reps], no_op.rrwp[start].expand(reps, -1, -1, -1))


def test_score_and_delta_carriage_hooks_with_mock_model():
    module = _load_module()
    cfg = module.Config(
        classes=4,
        pair_vocab=2,
        ranks=(1, 2),
        distances=(1, 2, 3),
        load_distance=2,
        dim=8,
        heads=2,
        layers=2,
        models=("1hop", "dense"),
        seeds=(0,),
        carriage_graphs=2,
        carriage_donors=1,
        score_graphs=2,
        score_donors=1,
        score_batch_size=1,
        ablation_graphs=4,
        focus_distances=(1, 3),
    )

    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(cfg.dim, cfg.dim)

        def forward(self, states):
            return self.linear(states).reshape(-1, cfg.heads, cfg.dim // cfg.heads), None

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.L, self.H, self.dh = cfg.layers, cfg.heads, cfg.dim // cfg.heads
            self.input = nn.Linear(cfg.feature_dim, cfg.dim)
            self.structure = nn.Linear(1, cfg.dim, bias=False)
            self.attentions = nn.ModuleList([MockAttention() for _ in range(cfg.layers)])
            self.output = nn.Linear(cfg.dim, cfg.classes)

        @property
        def attention_layers(self):
            return list(self.attentions)

        def forward(self, batch):
            relative = batch.rrwp[:, :, 0, :].sum(dim=-1, keepdim=True)
            states = self.input(batch.x) + self.structure(relative)
            for attention in self.attentions:
                heads, _ = attention(states.reshape(-1, cfg.dim))
                states = states + heads.reshape(len(batch), cfg.n, cfg.dim)
            return self.output(states)

    model = MockModel().eval()
    clean = module.make_batch(cfg, 2, seed=41, condition="reach", topology="thin", distance=2, rank=1)
    score = module.score_factor_batch(
        model, cfg, clean, factor="semantic", seed=42, device=torch.device("cpu")
    )
    assert score.shape == (2, cfg.layers, cfg.heads)
    assert torch.isfinite(score).all()

    delta = module.target_carriage_with_head_ablations(
        model, cfg, clean, factor="semantic", seed=43, device=torch.device("cpu")
    )
    assert delta["functional_drop"].shape == (cfg.layers, cfg.heads, len(clean))
    assert delta["benefit_drop"].shape == (cfg.layers, cfg.heads, len(clean))
    assert torch.isfinite(delta["functional_drop"]).all()
