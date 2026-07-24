from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "experiments/methodology/colab_scoring_metric_refinement_dense.py"


class DataLike:
    def __init__(self, torch):
        self.x = torch.tensor([[1], [2], [3]], dtype=torch.long)
        self.y = torch.tensor([0.5])
        self.num_nodes = 3
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )
        self.edge_attr = torch.tensor([[1], [1], [2], [2]], dtype=torch.long)
        self.rrwp = torch.arange(9.0).reshape(3, 3)
        self.deg = torch.tensor([1, 2, 1])
        self.log_deg = torch.log(self.deg.float() + 1)
        row = torch.arange(3).repeat_interleave(3)
        col = torch.arange(3).repeat(3)
        self.rrwp_index = torch.stack([row, col])
        self.rrwp_val = torch.arange(27.0).reshape(9, 3)

    def clone(self):
        return copy.deepcopy(self)


def _capture(torch, *, attention, message, routed=None):
    from graph_specialisation_metrics.scoring_refinement.fields import GraphCapture, HeadFields

    attention = torch.as_tensor(attention, dtype=torch.float32)
    message = torch.as_tensor(message, dtype=torch.float32)
    if routed is None:
        routed = torch.einsum("hij,hijd->ihd", attention, message)
    layer = HeadFields(
        layer=0,
        attention=attention,
        message=message,
        mask=torch.ones_like(attention, dtype=torch.bool),
        routed_output=routed,
    )
    return GraphCapture(
        prediction=torch.tensor([[0.0]]),
        target=torch.tensor([[0.0]]),
        layers=[layer],
    )


def test_semantic_and_pe_transpositions_are_involutions_and_channel_pure():
    torch = pytest.importorskip("torch")
    from graph_specialisation_metrics.scoring_refinement.interventions import (
        audit_preservation,
        pe_transposition,
        semantic_transposition,
    )

    base = DataLike(torch)
    semantic = semantic_transposition(base, 0, 2)
    audit_preservation(
        base,
        semantic,
        {"variant": "semantic_transposition", "source": 0, "partner": 2},
    )
    semantic_twice = semantic_transposition(semantic, 0, 2)
    assert torch.equal(semantic_twice.x, base.x)
    assert torch.equal(semantic.edge_index, base.edge_index)
    assert torch.equal(semantic.rrwp, base.rrwp)

    pe = pe_transposition(base, 0, 2)
    audit_preservation(
        base,
        pe,
        {"variant": "pe_transposition", "source": 0, "partner": 2},
    )
    pe_twice = pe_transposition(pe, 0, 2)
    assert torch.equal(pe_twice.x, base.x)
    assert torch.equal(pe_twice.rrwp, base.rrwp)
    assert torch.equal(pe_twice.rrwp_index, base.rrwp_index)
    assert torch.equal(pe.edge_index, base.edge_index)
    assert torch.equal(pe.edge_attr, base.edge_attr)


def test_self_events_are_exact_noops():
    torch = pytest.importorskip("torch")
    from graph_specialisation_metrics.scoring_refinement.interventions import (
        pe_single_donor,
        pe_transposition,
        semantic_single_donor,
        semantic_transposition,
    )

    base = DataLike(torch)
    own = base.x[1].clone()
    assert torch.equal(semantic_single_donor(base, 1, own).x, base.x)
    assert torch.equal(semantic_transposition(base, 1, 1).x, base.x)
    for variant in (pe_single_donor(base, 1, 1), pe_transposition(base, 1, 1)):
        assert torch.equal(variant.x, base.x)
        assert torch.equal(variant.rrwp, base.rrwp)
        assert torch.equal(variant.rrwp_index, base.rrwp_index)


def test_topology_donor_is_aligned_while_base_content_and_target_stay_fixed():
    torch = pytest.importorskip("torch")
    from graph_specialisation_metrics.scoring_refinement.interventions import (
        topology_donor_variant,
    )

    base = DataLike(torch)
    donor = DataLike(torch)
    donor.x = torch.tensor([[9], [8], [7]])
    donor.y = torch.tensor([99.0])
    donor.rrwp = donor.rrwp + 100.0
    donor.deg = torch.tensor([2, 1, 2])
    donor.log_deg = torch.log(donor.deg.float() + 1)
    alignment = np.asarray([2, 0, 1])
    variant = topology_donor_variant(base, donor, alignment)

    assert torch.equal(variant.x, base.x)
    assert torch.equal(variant.y, base.y)
    torch.testing.assert_close(variant.rrwp, donor.rrwp[torch.tensor(alignment)])
    torch.testing.assert_close(variant.deg, donor.deg[torch.tensor(alignment)])


def test_probability_mass_follow_and_invariant_scores_have_expected_extremes():
    torch = pytest.importorskip("torch")
    from graph_specialisation_metrics.scoring_refinement.scores import (
        follow_invariant_scores,
        transposition_permutation,
    )

    attention = torch.tensor([[[0.8, 0.2], [0.1, 0.9]]])
    message = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]]]
    )
    clean = _capture(torch, attention=attention, message=message)
    permutation = transposition_permutation(2, 0, 1)
    follow = _capture(
        torch,
        attention=attention.index_select(-1, permutation),
        message=message.index_select(-2, permutation),
    )
    result = follow_invariant_scores(clean, follow, permutation)

    assert result["attention_follow"][0, 0] == pytest.approx(1.0)
    expected_invariant = 0.5 * (
        (0.8 * 0.2 + 0.2 * 0.8) / (0.8**2 + 0.2**2)
        + (0.1 * 0.9 + 0.9 * 0.1) / (0.1**2 + 0.9**2)
    )
    assert result["attention_invariant"][0, 0] == pytest.approx(
        expected_invariant
    )
    assert result["transport_follow"][0, 0] == pytest.approx(1.0)
    assert 0.0 <= result["attention_invariant"][0, 0] < 1.0
    assert 0.0 <= result["transport_invariant"][0, 0] < 1.0
    assert result["attention_mass"][0, 0] == pytest.approx(1.0)


def test_appendix_cosine_uses_softmax_mass_movement_weights():
    torch = pytest.importorskip("torch")
    from graph_specialisation_metrics.scoring_refinement.scores import (
        appendix_cosine_group_scores,
        transposition_permutation,
    )

    clean_attention = torch.tensor(
        [[[0.80, 0.15, 0.05]] * 3], dtype=torch.float32
    )
    message = torch.ones(1, 3, 3, 1)
    clean = _capture(torch, attention=clean_attention, message=message)
    event_01 = _capture(
        torch,
        attention=clean_attention[..., [1, 0, 2]],
        message=message[..., [1, 0, 2], :],
    )
    event_02 = _capture(
        torch,
        attention=clean_attention,
        message=message,
    )
    result = appendix_cosine_group_scores(
        clean,
        [event_01, event_02],
        [
            transposition_permutation(3, 0, 1),
            transposition_permutation(3, 0, 2),
        ],
        temperature=0.1,
    )

    weights = np.exp(np.asarray([0.65, 0.75]) / 0.1)
    weights /= weights.sum()

    def swapped_cosine(left, right):
        return 2.0 * left * right / (left**2 + right**2)

    expected_follow = (
        weights[0] + weights[1] * swapped_cosine(0.80, 0.05)
    )
    expected_invariant = (
        weights[0] * swapped_cosine(0.80, 0.15) + weights[1]
    )
    assert result["attention_follow"][0, 0] == pytest.approx(
        expected_follow, rel=1.0e-6
    )
    assert result["attention_invariant"][0, 0] == pytest.approx(
        expected_invariant, rel=1.0e-6
    )


def test_official_grit_field_collector_batches_event_replicas(monkeypatch):
    torch = pytest.importorskip("torch")
    import types

    from graph_specialisation_metrics.scoring_refinement.fields import (
        GritFieldCollector,
        attention_mass_error,
        reconstruction_error,
    )

    class Graph:
        def __init__(self, values):
            self.x = torch.tensor(
                [[value] for value in values],
                dtype=torch.float32,
                requires_grad=True,
            )
            self.y = torch.tensor([0.0])
            self.num_nodes = len(values)

        def clone(self):
            return copy.deepcopy(self)

    class FakeBatch:
        @classmethod
        def from_data_list(cls, items):
            out = cls()
            out.x = torch.cat([item.x for item in items], dim=0)
            out.y = torch.stack([item.y for item in items])
            out.counts = [item.num_nodes for item in items]
            edges = []
            offset = 0
            for count in out.counts:
                for query in range(count):
                    for key in range(count):
                        edges.append((offset + key, offset + query))
                offset += count
            out.edge_index = torch.tensor(edges, dtype=torch.long).T
            return out

        def to(self, _device):
            return self

    class Attention(torch.nn.Module):
        edge_enhance = False

        def forward(self, batch):
            key, query = batch.edge_index
            batch.V_h = batch.x.reshape(-1, 1, 1)
            weights = []
            for count in batch.counts:
                weights.extend([1.0 / count] * (count * count))
            batch.attn = torch.tensor(weights).reshape(-1, 1, 1)
            message = batch.V_h[key] * batch.attn
            routed = torch.zeros(len(batch.x), 1, 1)
            routed.index_add_(0, query, message)
            return routed, torch.zeros(1)

    class Model(torch.nn.Module):
        def __init__(self, attention):
            super().__init__()
            self.attention = attention

        def forward(self, batch):
            routed, _ = self.attention(batch)
            predictions = []
            offset = 0
            for count in batch.counts:
                predictions.append(routed[offset:offset + count].sum().reshape(1))
                offset += count
            return torch.stack(predictions), batch.y

    fake_data_module = types.ModuleType("torch_geometric.data")
    fake_data_module.Batch = FakeBatch
    fake_parent = types.ModuleType("torch_geometric")
    fake_parent.data = fake_data_module
    monkeypatch.setitem(sys.modules, "torch_geometric", fake_parent)
    monkeypatch.setitem(sys.modules, "torch_geometric.data", fake_data_module)

    attention = Attention()
    gm = types.SimpleNamespace(
        device=torch.device("cpu"),
        L=1,
        H=1,
        attn_layers=[attention],
        model=Model(attention),
    )
    captures = GritFieldCollector(gm).collect_many(
        [Graph([1.0, 3.0]), Graph([2.0, 4.0, 6.0])]
    )
    assert len(captures) == 2
    assert captures[0].layers[0].message.shape == (1, 2, 2, 1)
    assert captures[1].layers[0].message.shape == (1, 3, 3, 1)
    assert reconstruction_error(captures[0].layers[0]) == pytest.approx(0.0)
    assert reconstruction_error(captures[1].layers[0]) == pytest.approx(0.0)
    assert attention_mass_error(captures[0].layers[0]) == pytest.approx(0.0)
    assert attention_mass_error(captures[1].layers[0]) == pytest.approx(0.0)
    clean, gradients = GritFieldCollector(gm).clean_with_gradients(
        Graph([1.0, 3.0])
    )
    assert gradients[0].shape == (1, 2, 1, 1)
    assert float(torch.linalg.vector_norm(gradients[0])) > 0.0
    assert clean.layers[0].routed_output is not None


def test_output_projected_eg_and_hierarchical_aggregation():
    torch = pytest.importorskip("torch")
    from graph_specialisation_metrics.scoring_refinement.scores import (
        graph_balanced_mean,
        hierarchical_event_mean,
        projected_event_score,
    )

    attention = torch.ones(1, 2, 2) / 2
    message = torch.ones(1, 2, 2, 1)
    clean = _capture(
        torch,
        attention=attention,
        message=message,
        routed=torch.tensor([[[2.0]], [[4.0]]]),
    )
    variant = _capture(
        torch,
        attention=attention,
        message=message,
        routed=torch.tensor([[[1.0]], [[1.0]]]),
    )
    gradient = [torch.tensor([[[[2.0]], [[1.0]]]])]
    score = projected_event_score(clean, variant, gradient)
    assert score.shape == (1, 1)
    assert score[0, 0] == pytest.approx(5.0)

    aggregated = hierarchical_event_mean(
        [[np.asarray([[1.0]]), np.asarray([[3.0]])], [np.asarray([[10.0]])]]
    )
    # Event mean for source 1 is 2; source 2 is 10; sources are then equal.
    assert aggregated[0, 0] == pytest.approx(6.0)

    graph_values = np.asarray([[[1.0]], [[3.0]], [[8.0]]])
    assert graph_balanced_mean(graph_values)[0, 0] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="zero graphs"):
        graph_balanced_mean(np.empty((0, 1, 1)))


def test_all_method_rows_share_m1_references_and_keep_topology_separate():
    from graph_specialisation_metrics.scoring_refinement.config import METHODS
    from graph_specialisation_metrics.scoring_refinement.scores import build_score_tables
    from graph_specialisation_metrics.scoring_refinement.validation import (
        build_m1_arm_agreement,
        build_m1_cross_method_correlations,
    )

    shape = (2, 2)
    score = {
        "eg_semantic_single": np.full(shape, 2.0),
        "eg_semantic_transposition": np.full(shape, 4.0),
        "eg_pe_single": np.full(shape, 3.0),
        "eg_pe_transposition": np.full(shape, 6.0),
        "semantic_transport_follow": np.full(shape, 0.8),
        "semantic_transport_invariant": np.full(shape, 0.2),
        "pe_transport_follow": np.full(shape, 0.7),
        "pe_transport_invariant": np.full(shape, 0.3),
        "semantic_attention_follow": np.full(shape, 0.9),
        "semantic_attention_invariant": np.full(shape, 0.1),
        "pe_attention_follow": np.full(shape, 0.85),
        "pe_attention_invariant": np.full(shape, 0.15),
        "topology_eg": np.full(shape, 11.0),
    }
    raw, derived, references = build_score_tables(
        "task", "sha", score, graph_count=6, event_count=2
    )
    assert len(raw) == len(METHODS) * 4
    assert len(derived) == len(raw)
    assert references["M1_DD"] == references["M1_TT"] == (2.0, 6.0)
    assert {row["topology_score"] for row in raw} == {11.0}
    assert all("topology" not in row["pe_intervention"] for row in raw)
    arm_agreement = build_m1_arm_agreement(derived, top_k=(3,))
    assert len(arm_agreement) == 6
    assert all(row["comparison_type"] == "m1_arm" for row in arm_agreement)
    cross_method = build_m1_cross_method_correlations(raw)
    assert len(cross_method) == 4 * 2 * 5
    assert {row["axis"] for row in cross_method} == {"semantic", "structural"}


def test_deterministic_splits_are_disjoint_and_fast_sizes_are_small():
    from graph_specialisation_metrics.scoring_refinement.config import (
        RunSizes,
        deterministic_splits,
    )

    sizes = RunSizes.fast()
    split = deterministic_splits(100, sizes, 7)
    assert len(split["score"]) == 6
    assert len(split["causal"]) == 4
    assert len(split["ablation"]) == 8
    assert not (set(split["score"]) & set(split["causal"]))
    assert not (set(split["score"]) & set(split["ablation"]))
    assert not (set(split["causal"]) & set(split["ablation"]))
    assert split == deterministic_splits(100, sizes, 7)


def test_cache_rejects_checkpoint_and_manifest_changes(tmp_path):
    pytest.importorskip("torch")
    from graph_specialisation_metrics.scoring_refinement.cache import CacheStore

    store = CacheStore(
        tmp_path, protocol_fingerprint="protocol", checkpoint_sha="sha-a", task="task"
    )
    manifest = {"event": [1, 2]}
    store.save_torch("scores", "one", {"ok": True}, event_manifest=manifest)
    assert store.load_torch("scores", "one", event_manifest=manifest) == {"ok": True}
    assert store.load_torch("scores", "one", event_manifest={"event": [2, 1]}) is None
    other = CacheStore(
        tmp_path, protocol_fingerprint="protocol", checkpoint_sha="sha-b", task="task"
    )
    assert other.load_torch("scores", "one", event_manifest=manifest) is None


def test_colab_frontend_strips_only_injected_kernel_arguments():
    spec = importlib.util.spec_from_file_location("_scoring_refinement_frontend_test", FRONTEND)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module._strip_colab_kernel_args(
        [
            "--phase",
            "scores",
            "-f",
            "/root/.local/share/jupyter/runtime/kernel-deadbeef.json",
            "--fast-dev-run",
        ]
    ) == ["--phase", "scores", "--fast-dev-run"]
    assert module._strip_colab_kernel_args(["-f", "ordinary.json"]) == [
        "-f",
        "ordinary.json",
    ]
    parsed = module.build_parser().parse_args(
        [
            "--task",
            "zinc",
            "--phase",
            "figures",
            "--cosine-temperature",
            "0.2",
        ]
    )
    assert parsed.task == ["zinc"]
    assert parsed.phase == "figures"
    assert parsed.cosine_temperature == pytest.approx(0.2)


def test_required_figure_atlas_renders_from_cache_only_tables(tmp_path):
    pytest.importorskip("matplotlib")
    from graph_specialisation_metrics.scoring_refinement.figures import make_all_figures
    from graph_specialisation_metrics.scoring_refinement.scores import build_score_tables

    shape = (2, 2)
    score = {
        "eg_semantic_single": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "eg_semantic_transposition": np.asarray([[1.2, 2.2], [3.2, 4.2]]),
        "eg_pe_single": np.asarray([[4.0, 3.0], [2.0, 1.0]]),
        "eg_pe_transposition": np.asarray([[4.2, 3.2], [2.2, 1.2]]),
        "semantic_transport_follow": np.full(shape, 0.8),
        "semantic_transport_invariant": np.full(shape, 0.2),
        "pe_transport_follow": np.full(shape, 0.75),
        "pe_transport_invariant": np.full(shape, 0.25),
        "semantic_attention_follow": np.full(shape, 0.7),
        "semantic_attention_invariant": np.full(shape, 0.3),
        "pe_attention_follow": np.full(shape, 0.65),
        "pe_attention_invariant": np.full(shape, 0.35),
        "topology_eg": np.asarray([[2.0, 1.0], [4.0, 3.0]]),
    }
    raw, derived, _ = build_score_tables(
        "task", "sha", score, graph_count=6, event_count=2
    )
    ablation = [
        {
            "layer": layer,
            "head": head,
            "prediction_movement": float(layer + head + 1),
        }
        for layer in range(2)
        for head in range(2)
    ]
    causal = [
        {
            "channel": channel,
            "layer": layer,
            "head": head,
            "mediation": float((1 if channel == "semantic" else -1) * (layer + head + 1)),
        }
        for channel in ("semantic", "pe")
        for layer in range(2)
        for head in range(2)
    ]
    figures = make_all_figures(
        raw,
        derived,
        out_dir=tmp_path,
        ablation_rows=ablation,
        causal_rows=causal,
    )
    assert set(figures) == {
        "raw_score_atlas",
        "m1_intervention_factorial",
        "donor_vs_transposition",
        "m1_cross_method_correlations",
        "derived_DJ_atlas",
        "topology_companions",
        "significance_validation",
        "role_validation",
    }
    for paths in figures.values():
        assert {Path(path).suffix for path in paths} == {".png", ".pdf"}
        assert all(Path(path).exists() for path in paths)


def test_cached_numpy_payload_continues_through_validation_and_csv(tmp_path):
    from graph_specialisation_metrics.scoring_refinement.cache import read_csv, write_csv
    from graph_specialisation_metrics.scoring_refinement.scores import (
        build_score_tables,
        graph_balanced_mean,
    )
    from graph_specialisation_metrics.scoring_refinement.validation import (
        build_m1_arm_agreement,
        build_m1_cross_method_correlations,
        build_variant_agreement,
        compare_methods,
    )

    graph_count, layers, heads = 5, 2, 3
    graph_offset = np.arange(graph_count, dtype=float)[:, None, None] * 0.01
    head_pattern = np.arange(layers * heads, dtype=float).reshape(1, layers, heads)

    def cached(base, scale):
        return base + scale * head_pattern + graph_offset

    per_graph = {
        "eg_semantic_single": cached(1.0, 0.08),
        "eg_semantic_transposition": cached(1.15, 0.10),
        "eg_pe_single": cached(0.9, 0.07),
        "eg_pe_transposition": cached(1.1, 0.09),
        "semantic_transport_follow": cached(0.55, 0.015),
        "semantic_transport_invariant": cached(0.35, 0.010),
        "pe_transport_follow": cached(0.50, 0.012),
        "pe_transport_invariant": cached(0.40, 0.008),
        "semantic_attention_follow": cached(0.60, 0.010),
        "semantic_attention_invariant": cached(0.30, 0.012),
        "pe_attention_follow": cached(0.58, 0.009),
        "pe_attention_invariant": cached(0.32, 0.011),
        "topology_eg": cached(0.75, 0.06),
        "clean_throughput": cached(1.4, 0.05),
    }
    assert all(value.shape == (graph_count, layers, heads) for value in per_graph.values())

    score = {key: graph_balanced_mean(value) for key, value in per_graph.items()}
    raw, derived, _ = build_score_tables(
        "cached-task",
        "checkpoint-sha",
        score,
        graph_count=graph_count,
        event_count=4,
    )
    agreement = build_variant_agreement(
        score,
        per_graph,
        top_k=(2,),
        bootstrap_samples=10,
        seed=17,
    )
    agreement.extend(build_m1_arm_agreement(derived, top_k=(2,)))
    cross_method = build_m1_cross_method_correlations(raw)

    ablation = [
        {
            "layer": layer,
            "head": head,
            "prediction_movement": float(1 + 2 * layer + head),
        }
        for layer in range(layers)
        for head in range(heads)
    ]
    differential = np.asarray([[0.5, 0.2, -0.1], [0.4, -0.2, -0.6]])
    total = np.asarray([[0.2, 0.3, 0.5], [0.4, 0.6, 0.8]])
    throughput = score["clean_throughput"].copy()
    throughput[0, 0] = np.nan
    ablation_validation, causal_validation, ranking = compare_methods(
        derived,
        ablation,
        {"differential": differential, "total": total},
        throughput,
        top_k=(2,),
        seed=23,
    )

    assert len(raw) == len(derived) == 9 * layers * heads
    assert len(agreement) == 8
    assert len(cross_method) == 40
    assert len(ablation_validation) == len(causal_validation) == len(ranking) == 9
    assert {row["significance_rank"] for row in ranking} == set(range(1, 10))

    for name, rows in (
        ("raw", raw),
        ("derived", derived),
        ("agreement", agreement),
        ("cross_method", cross_method),
        ("ablation_validation", ablation_validation),
        ("causal_validation", causal_validation),
        ("ranking", ranking),
    ):
        path = tmp_path / f"{name}.csv"
        write_csv(path, rows)
        assert len(read_csv(path)) == len(rows)


def test_subset_method_figures_and_m1_agreement_are_graceful(tmp_path):
    pytest.importorskip("matplotlib")
    from graph_specialisation_metrics.scoring_refinement.figures import make_all_figures
    from graph_specialisation_metrics.scoring_refinement.scores import build_score_tables
    from graph_specialisation_metrics.scoring_refinement.validation import (
        build_m1_arm_agreement,
    )

    shape = (2, 2)
    score = {
        "eg_semantic_single": np.full(shape, 1.0),
        "eg_semantic_transposition": np.full(shape, 1.2),
        "eg_pe_single": np.full(shape, 0.9),
        "eg_pe_transposition": np.full(shape, 1.1),
        "semantic_transport_follow": np.asarray([[0.5, 0.6], [0.7, 0.8]]),
        "semantic_transport_invariant": np.asarray([[0.4, 0.3], [0.2, 0.1]]),
        "pe_transport_follow": np.full(shape, 0.6),
        "pe_transport_invariant": np.full(shape, 0.4),
        "semantic_attention_follow": np.full(shape, 0.6),
        "semantic_attention_invariant": np.full(shape, 0.4),
        "pe_attention_follow": np.full(shape, 0.6),
        "pe_attention_invariant": np.full(shape, 0.4),
        "topology_eg": np.full(shape, 0.7),
    }
    raw, derived, _ = build_score_tables(
        "task",
        "sha",
        score,
        graph_count=4,
        event_count=2,
        methods=("M2",),
    )
    assert build_m1_arm_agreement(derived, top_k=(2,)) == []
    figures = make_all_figures(raw, derived, out_dir=tmp_path)
    assert set(figures) == {
        "raw_score_atlas",
        "m1_intervention_factorial",
        "donor_vs_transposition",
        "derived_DJ_atlas",
        "topology_companions",
    }
    assert all(Path(path).exists() for paths in figures.values() for path in paths)
