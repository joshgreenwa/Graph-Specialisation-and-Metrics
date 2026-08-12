from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from graph_specialisation_metrics.experiments import graphormer
from graph_specialisation_metrics.experiments.graphormer_adapter import preprocess_graph


def _raw_graph(offset: int = 0):
    return {
        "num_nodes": 3,
        "node_feat": np.asarray([[0 + offset, 1], [1 + offset, 0], [2 + offset, 1]]),
        "edge_index": np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_feat": np.asarray([[0], [0], [1], [1]]),
    }


class _Dataset:
    def __init__(self):
        self.graphs = [_raw_graph(index % 2) for index in range(8)]
        self.labels = np.linspace(0.0, 3.5, len(self.graphs))

    def __getitem__(self, index):
        return self.graphs[int(index)], self.labels[int(index)]


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.head_dim = 2
        self.out_proj = nn.Linear(4, 4, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()

    def forward(self, values):
        return torch.tanh(self.self_attn.out_proj(values + values.mean(dim=0, keepdim=True)))


class _GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])

    def forward(
        self,
        input_nodes,
        input_edges,
        attn_bias,
        in_degree,
        out_degree,
        spatial_pos,
        attn_edge_type,
    ):
        del attn_bias, out_degree, attn_edge_type
        base = (
            input_nodes.float().sum(dim=-1) / 2000
            + in_degree.float() / 20
            + spatial_pos.float().sum(dim=-1) / 100
            + input_edges.float().sum(dim=(-1, -2, -3)) / 5000
        )
        nodes = torch.stack((base, base.square(), torch.sin(base), 2 * base), dim=-1)
        token = nodes.mean(dim=1, keepdim=True)
        values = torch.cat((token, nodes), dim=1).transpose(0, 1)
        for layer in self.layers:
            values = layer(values)
        return values.transpose(0, 1)


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_encoder = _GraphEncoder()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.encoder = _Encoder()
        self.classifier = nn.Linear(4, 1, bias=False)
        self.config = SimpleNamespace(multi_hop_max_dist=5, spatial_pos_max=20)

    def forward(self, return_dict=True, **inputs):
        del return_dict
        node_outputs = self.encoder.graph_encoder(**inputs)
        return SimpleNamespace(logits=self.classifier(node_outputs[:, 0]))


def _config():
    return {
        "data": {
            "evaluation_split": "valid",
            "semantic_pool_split": "train",
        },
        "model": {"id": "public/model", "revision": "a" * 40},
        "scoring": {
            "graphs": 1,
            "sources_per_graph": 2,
            "donor_swaps_per_source": 2,
            "semantic_pool_graphs": 4,
            "donor_swap_batch_size": 2,
            "seed": 7,
        },
    }


def test_preprocessing_matches_graphormer_index_shifts():
    config = SimpleNamespace(multi_hop_max_dist=5, spatial_pos_max=20)
    graph = preprocess_graph(
        {
            "num_nodes": 2,
            "node_feat": np.asarray([[0, 1], [2, 0]]),
            "edge_index": np.asarray([[0, 1], [1, 0]]),
            "edge_feat": np.asarray([[0], [0]]),
        },
        config,
    )
    assert graph.x.tolist() == [[3, 516], [5, 515]]
    assert graph.attn_edge_type[0, 1].tolist() == [2]
    assert graph.input_edges[0, 1, 0].tolist() == [3]
    assert graph.spatial_pos.tolist() == [[1, 2], [2, 1]]
    assert graph.in_degree.tolist() == [2, 2]


def test_graphormer_score_uses_donor_swaps_and_saves_scores(monkeypatch, tmp_path):
    dataset = _Dataset()
    assets = graphormer._Assets(
        _Model(),
        dataset,
        {"train": np.arange(6), "valid": np.arange(6, 8)},
        torch.device("cpu"),
    )
    monkeypatch.setattr(graphormer, "_require_dependencies", lambda: None)
    monkeypatch.setattr(graphormer, "_load_assets", lambda _config: assets)

    path = graphormer.score(_config(), checkpoint=None, output_dir=tmp_path, fast=True)
    saved = np.load(path)
    assert saved["semantic_scores"].shape == (4,)
    assert saved["structural_scores"].shape == (4,)
    assert np.all(saved["semantic_scores"] >= 0)
    assert np.all(saved["structural_scores"] >= 0)
    assert saved["semantic_scores"].sum() > 0
    assert saved["structural_scores"].sum() > 0
    assert np.allclose(
        saved["semantic_distance_contributions"].sum(axis=1),
        saved["semantic_scores"],
    )
    assert np.allclose(
        saved["structural_distance_contributions"].sum(axis=1),
        saved["structural_scores"],
    )
    assert saved["layer"].tolist() == [0, 0, 1, 1]
    assert saved["head"].tolist() == [0, 1, 0, 1]


def test_score_rejects_local_checkpoint_override(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="configured public checkpoint"):
        graphormer.score(_config(), checkpoint=tmp_path / "model.pt", output_dir=tmp_path)
