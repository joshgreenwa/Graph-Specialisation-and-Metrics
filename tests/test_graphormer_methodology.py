from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graph_specialisation_metrics.methodology.carriage import beneficial_carriage
from graph_specialisation_metrics.methodology.distance import (
    DistanceAxis,
    distance_event_contributions,
)
from graph_specialisation_metrics.methodology.graphormer import (
    GraphormerBackend,
    GraphormerRuntime,
    _official_to_hf_state,
    graphormer_graph_from_ogb,
)
from graph_specialisation_metrics.methodology.interventions import (
    dense_pair_donor_swap,
    semantic_donor_swap,
    structural_donor_swap,
)
from graph_specialisation_metrics.methodology.protocol import MethodologyConfig
from graph_specialisation_metrics.methodology.runner import _carriage_profile
from graph_specialisation_metrics.methodology.tasks import get_task


def tiny_graph(config):
    graph = {
        "num_nodes": 3,
        "node_feat": np.asarray(
            [
                [5, 0, 4],
                [6, 1, 3],
                [7, 2, 2],
            ],
            dtype=np.int64,
        ),
        "edge_index": np.asarray(
            [[0, 1, 1, 2], [1, 0, 2, 1]],
            dtype=np.int64,
        ),
        "edge_feat": np.asarray(
            [[0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]],
            dtype=np.int64,
        ),
    }
    return graphormer_graph_from_ogb(
        graph,
        [0.25],
        config=config,
        smiles="CCO",
    )


def tiny_config():
    transformers = pytest.importorskip("transformers")
    return transformers.GraphormerConfig(
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


def test_graphormer_semantic_and_structural_boundaries():
    config = SimpleNamespace(
        multi_hop_max_dist=5,
        spatial_pos_max=20,
        max_dist=5,
    )
    base = tiny_graph(config)
    task = get_task("graphormer_pcqm4mv2")

    semantic = semantic_donor_swap(
        base,
        0,
        base.x[2].numpy(),
        adapter=task.content_adapter,
    )
    assert torch.equal(semantic.x[0], base.x[2])
    assert torch.equal(semantic.spatial_pos, base.spatial_pos)
    assert torch.equal(semantic.input_edges, base.input_edges)

    structural = structural_donor_swap(
        base,
        0,
        2,
        task=task,
        duplicate_tolerance=0.0,
    )
    assert torch.equal(structural.x, base.x)
    assert torch.equal(structural.attn_bias, base.attn_bias)
    assert torch.equal(structural.edge_index, base.edge_index)
    assert torch.equal(structural.in_degree[0], base.in_degree[2])
    for name in task.dense_pair_structural_fields:
        assert torch.equal(
            getattr(structural, name),
            dense_pair_donor_swap(getattr(base, name), 0, 2),
        )
    # The donor's non-source coordinates remain fixed; this is not a reciprocal swap.
    assert torch.equal(structural.spatial_pos[2, 1:], base.spatial_pos[2, 1:])


def test_graphormer_special_carrier_labels_are_not_coerced_or_hidden():
    q = torch.tensor([[[[[5.0], [3.0], [4.0]]]]])
    axis = DistanceAxis((0, 1, "graph_token"))
    contribution, support = distance_event_contributions(
        q, ["graph_token", 0, 1], axis
    )
    assert contribution.shape == (1, 1, 1, 3)
    assert contribution[0, 0, 0].tolist() == [3.0, 4.0, 5.0]
    assert support[0].tolist() == [1.0, 1.0, 1.0]

    labels, estimate, interval = _carriage_profile(
        [
            {
                "seed": 0,
                "graph_id": 0,
                "source": 0,
                "donor": 0,
                "carrier": 0,
                "distance": np.nan,
                "carrier_kind": "graph_token",
                "B": 1.0,
            }
        ],
        "B",
        MethodologyConfig(),
    )
    assert labels == ["0", "1", "2", "3", "graph token"]
    assert estimate.shape == (5,)
    assert interval[0].shape == (5,)


def test_official_fairseq_readout_keys_map_to_hugging_face_head():
    state = {
        "encoder.layer.weight": torch.ones(2, 2),
        "encoder.embed_out.weight": torch.ones(1, 2),
        "encoder.lm_output_learned_bias": torch.ones(1),
        "encoder.masked_lm_pooler.dense.weight": torch.ones(2, 2),
    }
    converted = _official_to_hf_state(state)
    assert "encoder.layer.weight" in converted
    assert "classifier.classifier.weight" in converted
    assert "classifier.lm_output_learned_bias" in converted
    assert not any("masked_lm_pooler" in key for key in converted)


def test_graph_token_beneficial_carriage_is_the_registered_readout_carrier():
    clean = torch.zeros(3, 1)
    event = torch.zeros(1, 1, 3, 1)
    event[0, 0, 0, 0] = 1.0

    result = beneficial_carriage(
        clean,
        event,
        lambda graph_token: graph_token[:, 0].square(),
        carrier_weights=torch.tensor([1.0, 0.0, 0.0]),
        atol=1e-8,
        rtol=1e-8,
        max_intervals=16,
        tolerance=1e-7,
    )
    assert result.field.shape == (3, 1)
    assert float(result.field[0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert torch.equal(result.field[1:], torch.zeros_like(result.field[1:]))
    assert float(result.completeness_residual.abs().max()) < 1e-6


def test_graphormer_backend_hooks_patch_and_graph_token_replay():
    transformers = pytest.importorskip("transformers")
    config = tiny_config()
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

    captured = backend.capture([data, data], require_grad=False)
    assert captured.prediction.shape == (2, 1)
    assert captured.final_state.shape == (2, data.num_nodes + 1, config.embedding_dim)
    assert len(captured.transport) == config.num_hidden_layers
    assert captured.transport[0].shape == (
        2,
        data.num_nodes + 1,
        config.num_attention_heads,
        config.embedding_dim // config.num_attention_heads,
    )
    assert torch.allclose(captured.z[0], captured.z[1], atol=1e-7, rtol=0.0)

    clean = backend.clean_jacobians(data)
    assert clean.transport.shape == (
        1,
        config.num_hidden_layers,
        data.num_nodes + 1,
        config.num_attention_heads,
        config.embedding_dim // config.num_attention_heads,
    )
    assert clean.final_state.shape == (1, data.num_nodes + 1, config.embedding_dim)
    assert torch.count_nonzero(clean.final_state[:, 1:]) == 0

    replacements = backend.replacement_batch(captured, [0])
    prediction, z, _ = backend.patch(data, replacements, ((0, 0),))
    assert torch.allclose(prediction, captured.prediction[0:1], atol=1e-6, rtol=0.0)
    assert torch.allclose(z, captured.z[0:1], atol=1e-6, rtol=0.0)

    weights = backend.carriage_weights(data, captured.final_state[0])
    assert weights.tolist() == [1.0, 0.0, 0.0, 0.0]
    replay = backend.loss_from_pooled(captured.target[0:1])
    replay_loss = replay(captured.final_state[0:1, 0])
    actual_loss = backend.loss_per_graph(
        captured.prediction[0:1],
        captured.target[0:1],
    )
    assert torch.allclose(replay_loss, actual_loss, atol=1e-7, rtol=0.0)
    assert backend.attention_normalization_error(data) < 1e-6
