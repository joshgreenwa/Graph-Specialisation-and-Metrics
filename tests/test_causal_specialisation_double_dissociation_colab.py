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
        / "causal_specialisation_double_dissociation_colab.py"
    )
    name = "causal_specialisation_double_dissociation_colab_test_module"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config(module):
    return module.Config(
        n=12,
        classes=6,
        key_vocab=16,
        rrwp_steps=6,
        dim=48,
        heads=4,
        layers=2,
        score_donors=2,
        seeds=(0,),
    )


def test_dual_task_is_balanced_and_labels_are_valid():
    module = _load_module()
    cfg = _config(module)
    cfg.validate()
    batch = module.make_batch(cfg, 20, seed=7, mode=None)
    assert batch.x.shape == (20, cfg.n, cfg.feature_dim)
    assert batch.rrwp.shape == (20, cfg.n, cfg.n, cfg.rrwp_steps)
    assert int((batch.mode == module.MODE_SEMANTIC).sum()) == 10
    assert int((batch.mode == module.MODE_STRUCTURAL).sum()) == 10
    assert int(batch.y.min()) >= 0
    assert int(batch.y.max()) < cfg.classes
    source_flag = 2 * cfg.key_vocab + cfg.classes + 1
    rows = torch.arange(len(batch))
    assert torch.equal(batch.x[rows, batch.target_idx, source_flag], torch.ones(len(batch)))


def test_semantic_and_structural_interventions_change_only_declared_factor():
    module = _load_module()
    cfg = _config(module)
    for mode, factor in (
        (module.MODE_SEMANTIC, "semantic"),
        (module.MODE_STRUCTURAL, "structural"),
    ):
        clean = module.make_batch(cfg, 5, seed=11, mode=mode)
        replicas = module.make_replicas(cfg, clean, factor=factor, donors=2, seed=12)
        assert len(replicas) == 15
        corrupt = module.make_single_corruption(cfg, clean, factor=factor, seed=13)
        if factor == "semantic":
            assert bool((corrupt.x != clean.x).any())
            assert torch.equal(corrupt.rrwp, clean.rrwp)
            # The donor changes exactly the planted source's value field, and always changes
            # the counterfactual answer. Keys, roles, task markers, and every other node stay fixed.
            for graph in range(len(clean)):
                source = int(clean.target_idx[graph])
                expected = clean.x[graph].clone()
                value_slice = slice(cfg.key_vocab, cfg.key_vocab + cfg.classes)
                expected[source, value_slice] = corrupt.x[graph, source, value_slice]
                assert torch.equal(corrupt.x[graph], expected)
                clean_y = int(clean.y[graph])
                corrupt_y = int(torch.argmax(corrupt.x[graph, source, value_slice]))
                assert corrupt_y != clean_y
        else:
            assert torch.equal(corrupt.x, clean.x)
            assert bool((corrupt.rrwp != clean.rrwp).any())
            # The source receives exactly one donor's RRWP footprint without a reciprocal
            # replacement. Every cycle node has degree two, so donor matching is exact.
            for graph in range(len(clean)):
                q = int(clean.q_idx[graph])
                source = int(clean.target_idx[graph])
                donors = module.structural_donors(cfg, q, source)
                matched = [
                    donor
                    for donor in donors
                    if torch.equal(
                        corrupt.rrwp[graph],
                        module.copy_rrwp_footprint(clean.rrwp[graph], source, donor),
                    )
                ]
                assert len(matched) == 1
                assert module.cycle_distance(cfg.n, q, matched[0]) != int(clean.y[graph]) + 1
                donor = matched[0]
                other = [node for node in range(cfg.n) if node != source]
                assert torch.equal(
                    corrupt.rrwp[graph][other][:, other],
                    clean.rrwp[graph][other][:, other],
                )
                assert torch.equal(
                    corrupt.rrwp[graph, donor, donor],
                    clean.rrwp[graph, donor, donor],
                )


def test_no_op_replicas_are_identical_and_head_groups_are_disjoint():
    module = _load_module()
    cfg = _config(module)
    for mode, factor in (
        (module.MODE_SEMANTIC, "semantic"),
        (module.MODE_STRUCTURAL, "structural"),
    ):
        clean = module.make_batch(cfg, 3, seed=21, mode=mode)
        replicas = module.make_replicas(cfg, clean, factor=factor, donors=1, seed=22, no_op=True)
        for graph in range(len(clean)):
            assert torch.equal(replicas.x[2 * graph], replicas.x[2 * graph + 1])
            assert torch.equal(replicas.rrwp[2 * graph], replicas.rrwp[2 * graph + 1])

    semantic = torch.tensor([[8.0, 4.0, 1.0, 1.0], [7.0, 3.0, 1.0, 1.0]]).numpy()
    structural = torch.tensor([[1.0, 1.0, 4.0, 8.0], [1.0, 1.0, 3.0, 7.0]]).numpy()
    groups = module.select_head_groups(semantic, structural, size=2)
    assert len(groups["semantic"]) == 2
    assert len(groups["structural"]) == 2
    assert set(groups["semantic"]).isdisjoint(groups["structural"])
    dj_groups = module.select_dj_groups(semantic, structural, size=2)
    flattened = [head for group in dj_groups.values() for head in group]
    assert set(dj_groups) == {
        "semantic_specialist",
        "structural_specialist",
        "high_J_generalist",
        "low_J_inert",
    }
    assert len(flattened) == len(set(flattened)) == 8


def test_joint_strength_and_selectivity_separate_amplitude_from_role():
    module = _load_module()
    semantic = np.asarray([2.0, 0.0, 1.0, 0.02])
    structural = np.asarray([0.0, 2.0, 1.0, 0.01])
    joint, selectivity = module.joint_selectivity(semantic, structural)
    assert np.allclose(joint[:3], [1.0, 1.0, 1.0])
    assert np.allclose(selectivity[:3], [1.0, -1.0, 0.0])
    # An apparently semantic tiny-score ratio is explicitly classified as unreliable/inert.
    assert selectivity[3] > 0.0
    assert module.dj_class(joint[3], selectivity[3]) == "low-J / inert"
    assert module.dj_class(1.0, 0.3) == "semantic specialist"
    assert module.dj_class(1.0, -0.3) == "structural specialist"
    assert module.dj_class(1.0, 0.0) == "high-J generalist"


def test_score_ablation_and_rescue_hooks_with_mock_head_model():
    module = _load_module()
    cfg = module.Config(
        n=8,
        classes=4,
        key_vocab=8,
        rrwp_steps=4,
        dim=8,
        heads=2,
        layers=2,
        score_graphs=4,
        score_donors=2,
        score_batch_size=2,
        ablation_graphs=4,
        rescue_graphs=4,
        analysis_batch_size=4,
        top_group_size=1,
        seeds=(0,),
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
            self.structural = nn.Linear(1, cfg.dim, bias=False)
            self.attentions = nn.ModuleList([MockAttention() for _ in range(cfg.layers)])
            self.output = nn.Linear(cfg.dim, cfg.classes)

        @property
        def attention_layers(self):
            return list(self.attentions)

        def forward(self, batch):
            # A deliberately simple structural dependence makes both intervention paths active.
            relative = batch.rrwp[:, :, 0, :].sum(dim=-1, keepdim=True)
            states = self.input(batch.x) + self.structural(relative)
            for attention in self.attentions:
                heads, _ = attention(states.reshape(-1, cfg.dim))
                states = states + heads.reshape(len(batch), cfg.n, cfg.dim)
            row = torch.arange(len(batch), device=states.device)
            return self.output(states[row, batch.q_idx])

    model = MockModel().eval()
    clean = module.make_batch(cfg, 3, seed=101, mode=module.MODE_SEMANTIC)
    score = module.score_channel_batch(model, cfg, clean, factor="semantic", seed=102, device=torch.device("cpu"))
    assert score.shape == (3, cfg.layers, cfg.heads)
    assert torch.isfinite(score).all()

    ablation = module.ablation_sweep(
        model,
        cfg,
        mode=module.MODE_SEMANTIC,
        seed=103,
        device=torch.device("cpu"),
    )
    assert ablation["functional"].shape == (cfg.layers, cfg.heads, cfg.ablation_graphs)
    assert torch.isfinite(ablation["functional"]).all()

    rescue = module.rescue_sweep(
        model,
        cfg,
        mode=module.MODE_SEMANTIC,
        factor="semantic",
        seed=104,
        device=torch.device("cpu"),
    )
    assert rescue["mediation"].shape == (cfg.layers, cfg.heads, cfg.rescue_graphs)
    assert torch.isfinite(rescue["mediation"]).all()

    groups = {
        "semantic": [(0, 0), (1, 0)],
        "structural": [(0, 1), (1, 1)],
    }
    family = module.family_ablation_sweep(
        model,
        cfg,
        groups=groups,
        seed=105,
        device=torch.device("cpu"),
    )
    assert family["revision"] == module.FAMILY_ABLATION_REVISION
    for task_name in ("semantic", "structural"):
        for family_name in ("semantic", "structural"):
            values = family["tasks"][task_name]["families"][family_name]
            assert values["loss"].shape == (3, cfg.ablation_graphs)
            assert values["accuracy_drop"].shape == (3, cfg.ablation_graphs)
            assert torch.equal(values["loss"][0], torch.zeros(cfg.ablation_graphs))

    curves = module.iterative_family_ablation_values([{"family_ablation": family}])
    assert curves["semantic"]["semantic"]["loss_by_seed"].shape == (1, 3)
    assert curves["structural"]["structural"]["accuracy_drop_by_seed"].shape == (1, 3)

    dj_groups = {
        "semantic_specialist": [(0, 0)],
        "structural_specialist": [(0, 1)],
        "high_J_generalist": [(1, 0)],
        "low_J_inert": [(1, 1)],
    }
    dj_family = module.family_ablation_sweep(
        model,
        cfg,
        groups=dj_groups,
        seed=106,
        device=torch.device("cpu"),
        revision=module.DJ_FAMILY_ABLATION_REVISION,
    )
    dj_values = module.dj_family_ablation_values([{"dj_family_ablation": dj_family}])
    assert dj_values["semantic"]["high_J_generalist"]["functional_by_seed"].shape == (1, 2)
    assert dj_values["structural"]["low_J_inert"]["accuracy_drop_by_seed"].shape == (1, 2)


def test_validation_performance_summary_reports_seed_mean_and_std():
    module = _load_module()

    def heldout(base):
        return {
            "accuracy": base,
            "semantic_accuracy": base - 0.02,
            "structural_accuracy": base + 0.02,
            "loss": 1.0 - base,
            "semantic_loss": 1.1 - base,
            "structural_loss": 0.9 - base,
        }

    checkpoints = {
        1: {"heldout_validation": heldout(0.90), "best_validation": {"accuracy": 0.91}},
        0: {"heldout_validation": heldout(0.94), "best_validation": {}},
    }
    summary = module.validation_performance_summary(checkpoints)
    assert summary["metric_source"] == "heldout_validation"
    assert summary["n_seeds"] == 2
    assert [row["seed"] for row in summary["per_seed"]] == [0, 1]
    accuracy = summary["aggregate"]["accuracy"]
    assert np.isclose(accuracy["mean"], 0.92)
    assert np.isclose(accuracy["std"], np.std([0.90, 0.94], ddof=1))
    assert np.isclose(accuracy["sem"], accuracy["std"] / np.sqrt(2))
    assert accuracy["n_seeds"] == 2
    # Missing selection metrics degrade to NaN placeholders instead of failing.
    assert np.isnan(summary["per_seed"][0]["selection_loss"])
    assert np.isclose(summary["per_seed"][1]["selection_accuracy"], 0.91)

    single = module.validation_performance_summary({5: {"heldout_validation": heldout(0.88)}})
    assert np.isclose(single["aggregate"]["accuracy"]["mean"], 0.88)
    assert single["aggregate"]["accuracy"]["std"] == 0.0
    assert single["aggregate"]["accuracy"]["sem"] == 0.0

    module.print_validation_performance(summary, module.Config())
