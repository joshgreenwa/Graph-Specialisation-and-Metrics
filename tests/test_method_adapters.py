from pathlib import Path

import torch

from graph_specialisation_metrics.method_adapters import (
    OfficialGRITAdapter,
    make_pyg_adapter,
    make_small_graph_transformer_adapter,
    parameter_count_close,
)
from graph_specialisation_metrics.method_core import GraphBatchView, edge_index_from_edges

try:
    import pytest
except Exception:  # pragma: no cover - allows direct smoke execution without pytest.
    pytest = None


def test_small_graph_transformer_adapter_exposes_attention_and_params():
    adapter = make_small_graph_transformer_adapter(content_dim=8, hidden_dim=16, layers=2, heads=4)
    graph = GraphBatchView(
        x=torch.randn(5, 8),
        edge_index=edge_index_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
        metadata={"selector_mask": torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32)},
    )
    cache = adapter.forward(graph)
    assert cache.prediction.shape == (1,)
    assert cache.final_node_states.shape == (5, 16)
    assert cache.attention is not None
    assert len(cache.attention) == 2
    assert adapter.parameter_count() > 0


def test_official_grit_adapter_counts_checkpoint_parameters_and_matches(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("model: grit\n", encoding="utf-8")
    ckpt_a = tmp_path / "dense.pt"
    ckpt_b = tmp_path / "onehop.pt"
    payload = {"state_dict": {"a": torch.zeros(2, 3), "b": torch.zeros(4)}}
    torch.save(payload, ckpt_a)
    torch.save(payload, ckpt_b)
    dense = OfficialGRITAdapter(tmp_path, config, ckpt_a, variant="official")
    onehop = OfficialGRITAdapter(tmp_path, config, ckpt_b, variant="1hop")
    matched, dense_params, onehop_params = parameter_count_close(dense, onehop)
    assert matched
    assert dense_params == 10
    assert onehop_params == 10


def test_pyg_adapter_reports_no_attention_when_available():
    try:
        adapter = make_pyg_adapter("gin", input_dim=4, hidden_dim=8, layers=2)
    except RuntimeError:
        if pytest is None:
            return
        pytest.skip("torch_geometric is not installed in this environment")
    graph = GraphBatchView(
        x=torch.randn(4, 4),
        edge_index=edge_index_from_edges(4, [(0, 1), (1, 2), (2, 3)]),
    )
    cache = adapter.forward(graph)
    assert cache.prediction.shape == (1,)
    assert cache.attention is None
    assert adapter.attention_maps(graph) is None
