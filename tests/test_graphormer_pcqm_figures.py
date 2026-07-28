from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
from matplotlib.container import ErrorbarContainer
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from graph_specialisation_metrics.methodology.graphormer import (
    GraphormerBackend,
    GraphormerRuntime,
    graphormer_graph_from_ogb,
)
from graph_specialisation_metrics.methodology.graphormer_figure_data import (
    CanonicalHeadMetrics,
    GraphormerDiagnosticExtractor,
    SupplementalCache,
    select_specialist_heads,
)
from graph_specialisation_metrics.methodology.graphormer_figure_plots import (
    plot_coordinate_heatmaps,
    plot_score_plane,
    plot_selectivity_joint_plane,
)
from graph_specialisation_metrics.methodology.tasks import get_task


def synthetic_metrics() -> CanonicalHeadMetrics:
    semantic = np.asarray([[0.7, 1.1, 1.3], [0.8, 1.0, 2.0]])
    structural = np.asarray([[1.8, 1.0, 0.9], [1.4, 0.8, 0.7]])
    joint = 0.5 * (semantic + structural)
    selectivity = (semantic - structural) / (semantic + structural)
    coordinates = SimpleNamespace(
        raw_semantic=semantic * 0.2,
        raw_structural=structural * 0.3,
        normalized_semantic=semantic,
        normalized_structural=structural,
        joint_sensitivity=joint,
        selectivity=selectivity,
        active=np.asarray([[True, True, True], [True, False, True]]),
        estimable=True,
    )
    return CanonicalHeadMetrics.from_scores(
        {
            "coordinates": coordinates,
            "channels": {
                "semantic": {"raw": coordinates.raw_semantic.copy()},
                "structural": {"raw": coordinates.raw_structural.copy()},
            },
            "axis": (0, 1, 2, "graph_token"),
            "clean_attention_distance": np.full((2, 3, 4), 0.25),
        }
    )


def tiny_graph(config):
    graph = {
        "num_nodes": 3,
        "node_feat": np.asarray(
            [[5, 0, 4], [6, 1, 3], [7, 2, 2]], dtype=np.int64
        ),
        "edge_index": np.asarray(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64
        ),
        "edge_feat": np.asarray(
            [[0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
            dtype=np.int64,
        ),
    }
    return graphormer_graph_from_ogb(
        graph, [0.25], config=config, smiles="CCO"
    )


def test_canonical_adapter_and_active_drel_selection():
    metrics = synthetic_metrics()
    selected = select_specialist_heads(metrics, semantic_head=(1, 2))
    assert metrics.shape == (2, 3)
    assert selected == {"semantic": (1, 2), "structural": (0, 0)}
    assert metrics.distance_axis[-1] == "graph_token"


def test_supplemental_cache_is_immutable_and_contract_keyed(tmp_path):
    calls = []
    cache = SupplementalCache(tmp_path)
    first, first_path, first_hit = cache.load_or_compute(
        "demo", {"n": 3}, lambda: calls.append(1) or {"value": 7}
    )
    second, second_path, second_hit = cache.load_or_compute(
        "demo", {"n": 3}, lambda: calls.append(2) or {"value": 9}
    )
    third, third_path, third_hit = cache.load_or_compute(
        "demo", {"n": 4}, lambda: calls.append(3) or {"value": 11}
    )
    assert first == second
    assert first_path == second_path
    assert first_path != third_path
    assert not first_hit
    assert second_hit
    assert not third_hit
    assert calls == [1, 3]
    assert third["value"] == 11


def test_requested_scatter_figures_have_no_errorbar_artists():
    metrics = synthetic_metrics()
    selected = {"semantic": (1, 2), "structural": (0, 0)}
    figures = (
        plot_score_plane(metrics, selected),
        plot_selectivity_joint_plane(
            metrics, selected, xlim=(-0.6, 0.6)
        ),
    )
    try:
        for figure in figures:
            containers = [
                container
                for axis in figure.axes
                for container in axis.containers
                if isinstance(container, ErrorbarContainer)
            ]
            assert containers == []
    finally:
        for figure in figures:
            plt.close(figure)


def test_coordinate_heatmaps_mask_inactive_selectivity():
    metrics = synthetic_metrics()
    figure = plot_coordinate_heatmaps(metrics)
    try:
        image_axes = [axis for axis in figure.axes if axis.images]
        assert len(image_axes) == 2
        rendered = image_axes[0].images[0].get_array()
        assert bool(np.ma.getmaskarray(rendered)[1, 1])
    finally:
        plt.close(figure)


def test_graphormer_diagnostic_extractor_matches_exact_attention_sites():
    transformers = pytest.importorskip("transformers")
    config = transformers.GraphormerConfig(
        num_hidden_layers=2,
        embedding_dim=32,
        ffn_embedding_dim=32,
        num_attention_heads=4,
        num_classes=1,
        num_atoms=4608,
        num_edges=1536,
        num_in_degree=512,
        num_out_degree=512,
        num_spatial=512,
        num_edge_dis=128,
        multi_hop_max_dist=5,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    model = transformers.GraphormerForGraphClassification(config).eval()
    data = tiny_graph(config)
    task = get_task("graphormer_pcqm4mv2")
    runtime = GraphormerRuntime(
        model,
        [data],
        [data],
        device=torch.device("cpu"),
        seed=0,
        metric_fn=task.metric_fn,
    )
    backend = GraphormerBackend(runtime, task, sigma=[1.0])
    captured = GraphormerDiagnosticExtractor(backend).extract(data)

    assert len(captured.dot) == config.num_hidden_layers
    assert captured.dot[0].shape == (
        config.num_attention_heads,
        data.num_nodes + 1,
        data.num_nodes + 1,
    )
    assert captured.transport[0].shape == (
        config.num_attention_heads,
        data.num_nodes + 1,
        config.embedding_dim // config.num_attention_heads,
    )
    expected_attention = torch.softmax(captured.dot[0] + captured.bias[0], dim=-1)
    assert torch.allclose(
        captured.attention[0], expected_attention, atol=1e-7, rtol=0.0
    )

    canonical = backend.capture([data], require_grad=False)
    assert torch.allclose(
        captured.transport[0],
        canonical.transport[0][0].permute(1, 0, 2),
        atol=1e-7,
        rtol=0.0,
    )
