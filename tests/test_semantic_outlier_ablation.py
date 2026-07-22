"""Selection, cache, and pure-plot tests for the raw semantic-outlier follow-up."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
from types import SimpleNamespace

from graph_specialisation_metrics.comparison import data as D
from graph_specialisation_metrics.comparison import plots as P
from graph_specialisation_metrics.specialisation import semantic_outlier_ablation as S


def test_top_semantic_and_throughput_null_are_exact_layer_and_disjoint():
    sem = np.arange(24, dtype=float).reshape(3, 8)
    scores = {"S_sem": sem, "S_str": np.ones_like(sem)}
    top = S.select_top_semantic(scores, 4)
    assert top == [(2, 7), (2, 6), (2, 5), (2, 4)]
    throughput = np.arange(24, dtype=float).reshape(3, 8)
    null = S.throughput_matched_null(top, throughput)
    assert len(set(null)) == 4
    assert not set(top) & set(null)
    assert [l for l, _ in null] == [l for l, _ in top]


def test_drel_selects_signed_extremes():
    scores = {
        "S_sem": np.asarray([[9., 4., 1., .2, .1, .1]]),
        "S_str": np.asarray([[1., 2., 3., 4., 6., 9.]]),
    }
    chosen = S.select_drel_heads(scores, gsem=1., gstr=1., k=2)
    assert chosen["semantic"] == [(0, 0), (0, 1)]
    assert chosen["structural"] == [(0, 5), (0, 4)]


def test_six_head_control_falls_back_only_after_same_layer_is_exhausted():
    top = [(2, h) for h in range(6)]
    null = S.throughput_matched_null(top, np.ones((4, 8)))
    assert len(null) == 6 and not set(top) & set(null)
    assert sum(l == 2 for l, _ in null) == 2
    assert all(abs(l - 2) <= 1 for l, _ in null)


def _fixture(G=36, K=4, R=6):
    rng = np.random.default_rng(4)
    shape = (K, G)
    return {
        "budgets": np.arange(1, K + 1),
        "top_heads": np.asarray([(9, 7), (8, 6), (7, 5), (6, 4)]),
        "matched_heads": np.asarray([(9, 0), (8, 0), (7, 0), (6, 0)]),
        "top_scores": np.asarray([4.7, 4.2, 3.9, 3.6]),
        "target_func": np.abs(rng.normal(.1, .02, shape)),
        "target_loss": rng.normal(.03, .01, shape),
        "matched_func": np.abs(rng.normal(.03, .01, shape)),
        "matched_loss": rng.normal(.005, .006, shape),
        "reverse_func": np.abs(rng.normal(.08, .02, shape)),
        "reverse_loss": rng.normal(.02, .01, shape),
        "individual_func": np.abs(rng.normal(.05, .02, shape)),
        "individual_loss": rng.normal(.015, .01, shape),
        "matched_individual_func": np.abs(rng.normal(.02, .01, shape)),
        "matched_individual_loss": rng.normal(.003, .006, shape),
        "random_func": np.abs(rng.normal(.02, .01, (K, R, G))),
        "random_loss": rng.normal(.003, .006, (K, R, G)),
        "clean_pred": rng.normal(size=(G, 1)), "clean_loss": np.full(G, .07),
        "y": rng.normal(size=(G, 1)), "graph_ids": np.arange(G),
        "throughput_graph": np.ones((G, 10, 8)),
    }


def test_cache_loader_and_ablation_plot(tmp_path):
    item = _fixture()
    path = D.semantic_outlier_npz_path(tmp_path, "zinc")
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, cache_version=np.asarray(S.CACHE_VERSION),
                        score_fingerprint=np.asarray("abc"), **item)
    loaded = D.load_semantic_outlier(path)
    assert loaded["matched_individual_loss"].shape == item["matched_individual_loss"].shape
    _, out = P.plot_semantic_outlier_ablation(
        {"zinc": loaded}, ["zinc"], tmp_path / "outlier.png", n_boot=60)
    assert (tmp_path / "outlier.png").stat().st_size > 0


def test_attention_cache_and_plot(tmp_path):
    item = _fixture()
    heads = ([tuple(h) for h in item["top_heads"]]
             + [tuple(h) for h in item["matched_heads"]])
    molecules = []
    for graph_id, n in enumerate((6, 8, 10)):
        pos = np.stack([np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False)),
                        np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))], axis=1)
        bonds = np.asarray([(i, (i + 1) % n) for i in range(n)])
        maps = {}
        for hidx, head in enumerate(heads):
            a = np.random.default_rng(100 + hidx + n).uniform(size=(n, n))
            maps[head] = a / a.sum(axis=1, keepdims=True)
        molecules.append({"graph_id": graph_id, "atom_types": np.arange(n) % 5,
                          "bonds": bonds, "pos": pos, "maps": maps})
    selected = {
        heads[0]: {"graph_ids": [2, 0, 1], "scores": [.9, .8, .7],
                   "candidate_ranks": [1, 2, 3], "candidate_count": 32},
    }
    selection_config = {"max_nodes": 18, "candidate_graphs": 32,
                        "score_donors": 32, "examples_per_head": 3,
                        "analysis_seed": 2718}
    attn = {"heads": heads, "molecules": molecules, "selected": selected,
            "selection_config": selection_config, "has_vnode": True,
            "head_channels": ["semantic"] * 4 + ["structural"] * 4}
    path = D.semantic_outlier_attention_path(tmp_path, "zinc")
    S.save_attention(attn, path, score_hash="abc")
    loaded = D.load_semantic_outlier_attention(path)
    assert loaded["heads"] == heads and len(loaded["molecules"]) == 3
    assert loaded["score_fingerprint"] == "abc"
    assert loaded["selection_config"] == selection_config
    assert loaded["selected"][heads[0]]["graph_ids"] == [2, 0, 1]
    assert loaded["has_vnode"] is True
    assert loaded["head_channels"][-1] == "structural"
    _, out = P.plot_semantic_outlier_attention(
        loaded, item, tmp_path / "attention.png")
    assert (tmp_path / "attention.png").stat().st_size > 0
    _, out = P.plot_semantic_outlier_attention_matrices(
        loaded, item, tmp_path / "matrices.png")
    assert (tmp_path / "matrices.png").stat().st_size > 0
    scores = {"S_sem": np.ones((10, 8)), "S_str": np.ones((10, 8))}
    _, out = P.plot_semantic_outlier_head_attention(
        loaded, item, tuple(item["top_heads"][0]), tmp_path / "head.png",
        scores=scores, gsem=1., gstr=1.)
    assert (tmp_path / "head.png").stat().st_size > 0
    _, out = P.plot_specialist_head_attention(
        loaded, heads[0], tmp_path / "specialist.png", channel="semantic",
        model_label="Dense", scores=scores, gsem=1., gstr=1.)
    assert (tmp_path / "specialist.png").stat().st_size > 0


def test_small_candidate_pool_applies_size_constraint_before_sampling():
    gm = SimpleNamespace(eval_ds=[SimpleNamespace(num_nodes=n) for n in (9, 21, 12, 18, 30, 15)])
    selected = S._small_attention_candidates(
        gm, np.arange(6), max_nodes=18, candidate_graphs=4, seed=2)
    assert len(selected) == 4
    assert all(gm.eval_ds[i].num_nodes <= 18 for i in selected)


def test_per_graph_semantic_collector_matches_method_a_sum():
    import torch

    collector = S._PerGraphSemanticCollector()
    phi = torch.tensor([[[[1.], [2.]], [[3.], [4.]]]])  # [T=1,n=2,H=2,dh=1]
    delta = torch.tensor([[[[2.], [1.]], [[1.], [-1.]]]])  # [S=1,n=2,H=2,dh=1]
    collector(channel="semantic", graph_id=7, source_nodes=np.asarray([0]),
              phi_stack=[phi], donor_averaged_delta=[delta])
    np.testing.assert_allclose(collector.scores["semantic"][7], [[5., 6.]])


def test_per_graph_semantic_collector_prefers_eventwise_eg_payload():
    import torch

    collector = S._PerGraphSemanticCollector()
    phi = torch.ones(1, 2, 1, 1)
    coherent_delta = torch.zeros(1, 2, 1, 1)
    eventwise = torch.tensor([[[2.0], [3.0]]])
    collector(
        channel="semantic", graph_id=8, source_nodes=np.asarray([0]),
        phi_stack=[phi], donor_averaged_delta=[coherent_delta],
        eventwise_functional=[eventwise],
    )
    np.testing.assert_allclose(collector.scores["semantic"][8], [[5.0]])


def test_attention_capture_omits_virtual_edges_from_atom_matrix():
    import torch
    import pytest
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    base = Data(x=torch.arange(3).reshape(-1, 1),
                edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                edge_attr=torch.ones(4, 1, dtype=torch.long), y=torch.tensor([0.]))
    captured_edges = torch.tensor([[0, 1, 3, 0], [1, 0, 0, 3]])
    captured_attention = torch.tensor([[.2], [.3], [.4], [.5]])

    class FakeGM:
        eval_ds = [base]
        device = torch.device("cpu")

        @staticmethod
        def capture(_batch, **_):
            return {"edge_index": captured_edges, "attn": [captured_attention]}

    result = S.attention_viz.collect_attention(FakeGM(), [0], [(0, 0)])
    A = result["molecules"][0]["maps"][(0, 0)]
    assert result["has_vnode"] is True and A.shape == (3, 3)
    assert np.isclose(A[1, 0], .2) and np.isclose(A[0, 1], .3)
    assert np.count_nonzero(A) == 2


def test_structural_cache_path_and_plot(tmp_path):
    item = _fixture()
    path = D.structural_outlier_npz_path(tmp_path, "zinc")
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, cache_version=np.asarray(S.CACHE_VERSION),
                        score_fingerprint=np.asarray("str"), **item)
    loaded = D.load_structural_outlier_by_task(tmp_path, ["zinc"])["zinc"]
    _, out = P.plot_semantic_outlier_ablation(
        {"zinc": loaded}, ["zinc"], tmp_path / "structural.png",
        channel="structural", n_boot=60)
    assert (tmp_path / "structural.png").stat().st_size > 0
