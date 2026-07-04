from types import SimpleNamespace

import torch

from graph_specialisation_metrics.sparse1hop_grit import (
    OFFICIAL_ZINC_GRIT_RRWP_PARAMS,
    Sparse1HopGRIT,
    load_official_grit_state_dict,
    parameter_count,
    translate_official_grit_key,
    zinc_sparse1hop_grit_config,
)


def _toy_batch() -> SimpleNamespace:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 4, 5, 5, 6], [1, 0, 2, 1, 4, 3, 5, 4, 6, 5]],
        dtype=torch.long,
    )
    edge_count = edge_index.size(1)
    return SimpleNamespace(
        x=torch.randint(0, 21, (7, 1)),
        edge_index=edge_index,
        edge_attr=torch.randint(0, 4, (edge_count, 1)),
        edge_rrwp=torch.rand(edge_count, 21),
        rrwp=torch.rand(7, 21),
        log_deg=torch.rand(7),
        batch=torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
    )


def test_sparse1hop_grit_k0_matches_official_parameter_count() -> None:
    model = Sparse1HopGRIT(zinc_sparse1hop_grit_config(num_global_tokens=0))
    assert parameter_count(model) == OFFICIAL_ZINC_GRIT_RRWP_PARAMS


def test_sparse1hop_grit_k4_parameter_matched_global_tokens_match_official_count() -> None:
    model = Sparse1HopGRIT(
        zinc_sparse1hop_grit_config(num_global_tokens=4, parameter_match_global_tokens=True)
    )
    assert parameter_count(model) == OFFICIAL_ZINC_GRIT_RRWP_PARAMS
    assert "global_token" in dict(model.named_buffers())
    assert model.global_token.requires_grad is False
    assert model.virtual_edge_encoder is None


def test_sparse1hop_grit_k1_parameter_matched_global_tokens_match_official_count() -> None:
    model = Sparse1HopGRIT(
        zinc_sparse1hop_grit_config(num_global_tokens=1, parameter_match_global_tokens=True)
    )
    assert parameter_count(model) == OFFICIAL_ZINC_GRIT_RRWP_PARAMS
    assert "global_token" in dict(model.named_buffers())
    assert model.global_token.shape == (1, 64)
    assert model.global_token.requires_grad is False
    assert model.virtual_edge_encoder is None


def test_sparse1hop_grit_forward_backward_with_global_tokens() -> None:
    torch.manual_seed(0)
    model = Sparse1HopGRIT(zinc_sparse1hop_grit_config(num_global_tokens=4))
    pred = model(_toy_batch())
    assert tuple(pred.shape) == (2, 1)
    pred.sum().backward()
    assert model.global_token is not None
    assert model.global_token.grad is not None


def test_sparse1hop_grit_forward_backward_with_parameter_matched_global_tokens() -> None:
    torch.manual_seed(0)
    model = Sparse1HopGRIT(
        zinc_sparse1hop_grit_config(num_global_tokens=4, parameter_match_global_tokens=True)
    )
    pred = model(_toy_batch())
    assert tuple(pred.shape) == (2, 1)
    pred.sum().backward()
    assert model.global_token is not None
    assert model.global_token.grad is None


def test_official_key_translation_covers_shared_components() -> None:
    assert translate_official_grit_key("encoder.node_encoder.encoder.weight") == "node_encoder.weight"
    assert translate_official_grit_key("encoder.edge_encoder.encoder.weight") == "edge_encoder.weight"
    assert translate_official_grit_key("rrwp_abs_encoder.fc.weight") == "rrwp_abs_encoder.weight"
    assert translate_official_grit_key("rrwp_rel_encoder.fc.weight") == "rrwp_rel_encoder.weight"
    assert translate_official_grit_key("layers.0.attention.Q.weight") == "layers.0.attention.Q.weight"
    assert translate_official_grit_key("post_mp.FC_layers.0.weight") == "post_mp.FC_layers.0.weight"


def test_official_state_dict_loader_accepts_translated_shared_weights() -> None:
    torch.manual_seed(1)
    source = Sparse1HopGRIT(zinc_sparse1hop_grit_config(num_global_tokens=0))
    target = Sparse1HopGRIT(zinc_sparse1hop_grit_config(num_global_tokens=4))
    official_like = {}
    for key, value in source.state_dict().items():
        if key == "node_encoder.weight":
            official_like["encoder.node_encoder.encoder.weight"] = value
        elif key == "edge_encoder.weight":
            official_like["encoder.edge_encoder.encoder.weight"] = value
        elif key == "rrwp_abs_encoder.weight":
            official_like["rrwp_abs_encoder.fc.weight"] = value
        elif key == "rrwp_rel_encoder.weight":
            official_like["rrwp_rel_encoder.fc.weight"] = value
        else:
            official_like[key] = value
    report = load_official_grit_state_dict(target, official_like, strict_shared=True)
    assert report["loaded_keys"] == len(source.state_dict())
