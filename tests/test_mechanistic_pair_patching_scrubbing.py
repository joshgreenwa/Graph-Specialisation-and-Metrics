from dataclasses import dataclass

import pytest
import torch

from graph_specialisation_metrics.mechanistic_pair_patching_scrubbing import (
    GraphScore,
    LayerActivation,
    add_control_lift,
    align_source_positions,
    assert_nonzero_sparse_coverage,
    outside_control_mask,
    safe_restoration,
    select_pairs,
    summarise_rows,
)


@dataclass
class FakeBatch:
    node_mask: torch.Tensor
    adj: torch.Tensor
    degree: torch.Tensor
    spd: torch.Tensor
    edge_batch: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor


def make_fake_batch() -> FakeBatch:
    node_mask = torch.ones(1, 4, dtype=torch.bool)
    adj = torch.zeros(1, 4, 4)
    edges = [(0, 1), (1, 2), (2, 3)]
    for src, dst in edges:
        adj[0, src, dst] = 1.0
        adj[0, dst, src] = 1.0
    return FakeBatch(
        node_mask=node_mask,
        adj=adj,
        degree=adj.sum(dim=-1),
        spd=torch.ones(1, 4, 4, dtype=torch.long),
        edge_batch=torch.zeros(len(edges), dtype=torch.long),
        edge_src=torch.tensor([src for src, _dst in edges]),
        edge_dst=torch.tensor([dst for _src, dst in edges]),
    )


def make_activation() -> LayerActivation:
    local_src = torch.tensor([0, 1, 2, 3])
    local_dst = torch.tensor([1, 2, 3, 0])
    zeros_2 = torch.zeros(4, 2)
    zeros_3 = torch.zeros(4, 2, 3)
    return LayerActivation(
        layer=0,
        src=local_src,
        dst=local_dst,
        local_src=local_src,
        local_dst=local_dst,
        attention=zeros_2,
        logits=zeros_2,
        node_message=zeros_3,
        pair_message=zeros_3,
        message=zeros_3,
        pair_e=torch.zeros(4, 12),
        pair_state_input=torch.zeros(4, 6),
        pair_state_output=torch.zeros(4, 6),
        heads=2,
        message_dim=3,
    )


def test_select_pairs_is_deterministic_and_requires_same_node_count():
    scores = [
        GraphScore(0, 10, 4, 0.0, 0.0, 1.0),
        GraphScore(1, 11, 4, 1.0, 1.0, 1.0),
        GraphScore(2, 12, 4, 2.0, 2.0, 1.0),
        GraphScore(3, 13, 4, 3.0, 3.0, 1.0),
        GraphScore(4, 14, 5, 100.0, 100.0, 1.0),
    ]

    pairs_a = select_pairs(
        scores,
        num_pairs=2,
        min_pred_delta=0.1,
        min_target_delta=0.1,
        adaptive_quantile=0.25,
        seed=7,
    )
    pairs_b = select_pairs(
        scores,
        num_pairs=2,
        min_pred_delta=0.1,
        min_target_delta=0.1,
        adaptive_quantile=0.25,
        seed=7,
    )

    assert pairs_a == pairs_b
    assert {pair.num_nodes for pair in pairs_a} == {4}
    assert pairs_a[0].clean_pred_score > pairs_a[0].corrupt_pred_score


def test_select_pairs_raises_when_no_same_size_pair_exists():
    scores = [
        GraphScore(0, 10, 4, 0.0, 0.0, 1.0),
        GraphScore(1, 11, 5, 1.0, 1.0, 1.0),
    ]

    with pytest.raises(RuntimeError, match="same-node-count"):
        select_pairs(
            scores,
            num_pairs=1,
            min_pred_delta=0.0,
            min_target_delta=0.0,
            adaptive_quantile=0.25,
            seed=0,
        )


def test_align_source_positions_uses_local_coordinates_not_tensor_order():
    source = make_activation()
    source.local_src = torch.tensor([1, 0, 2])
    source.local_dst = torch.tensor([0, 1, 2])
    current_src = torch.tensor([2, 1, 0])
    current_dst = torch.tensor([2, 0, 1])

    aligned = align_source_positions(source, current_src, current_dst, n=3)

    assert aligned.tolist() == [2, 0, 1]


def test_restoration_formula_and_degenerate_guard():
    assert safe_restoration(clean=10.0, corrupt=2.0, patched=6.0, min_abs_delta=1.0e-6) == 0.5

    with pytest.raises(RuntimeError, match="degenerate"):
        safe_restoration(clean=1.0, corrupt=1.0 + 1.0e-9, patched=1.0, min_abs_delta=1.0e-6)


def test_outside_control_preserves_operator_positive_count():
    batch = make_fake_batch()
    operator = torch.zeros(1, 4, 4)
    operator[0, 0, 2] = 1.0
    operator[0, 2, 0] = 1.0

    sampled = outside_control_mask(operator, batch, seed=3)

    assert int((sampled > 0).sum()) == int((operator > 0).sum())
    assert not bool(((sampled > 0) & (operator > 0)).any())


def test_zero_sparse_coverage_fails_fast():
    activation = make_activation()
    dense = torch.zeros(1, 4, 4)

    with pytest.raises(RuntimeError, match="zero sparse pair coverage"):
        assert_nonzero_sparse_coverage(activation, dense, "empty")


def test_summary_lift_uses_matched_random_control_mean():
    rows = [
        {
            "operator": "cut",
            "patch_target": "attention",
            "layer": 0,
            "head": -1,
            "control_type": "operator",
            "source_type": "clean",
            "restoration": 0.6,
        },
        {
            "operator": "cut",
            "patch_target": "attention",
            "layer": 0,
            "head": -1,
            "control_type": "matched_random",
            "source_type": "clean",
            "restoration": 0.2,
        },
    ]

    summary = summarise_rows(
        rows,
        metric="restoration",
        keys=("operator", "patch_target", "layer", "head", "control_type", "source_type"),
        seed=0,
    )
    lifted = add_control_lift(summary, target_field="patch_target")
    operator_row = next(row for row in lifted if row["control_type"] == "operator")

    assert operator_row["matched_random_mean"] == 0.2
    assert operator_row["control_normalised_lift"] == pytest.approx(0.4)
