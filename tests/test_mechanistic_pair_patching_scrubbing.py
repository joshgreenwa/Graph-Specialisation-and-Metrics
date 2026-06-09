from dataclasses import dataclass

import pytest
import torch

from graph_specialisation_metrics.mechanistic_pair_patching_scrubbing import (
    GraphScore,
    LayerActivation,
    add_control_lift,
    align_source_positions,
    apply_analysis_preset,
    assert_nonzero_sparse_coverage,
    build_parser,
    estimate_intervention_forwards,
    make_hypothesis_outputs,
    matrix_rows_from_raw,
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


def test_core_fast_preset_sets_task_aware_compact_defaults():
    argv = [
        "--checkpoint",
        "ckpt.pt",
        "--task",
        "flow_hard",
        "--analysis-preset",
        "core_fast",
    ]
    args = build_parser().parse_args(argv)

    apply_analysis_preset(args, argv)

    assert args.num_pairs == 64
    assert args.num_graphs == 768
    assert args.patch_targets == "routing_logits,pair_value,pair_state"
    assert args.operator_masks == "saturated_edge,min_cut_crossing_edge,shortest_st_path_edge"
    assert args.matched_random_controls == 2
    assert args.include_wrong_source_control is False


def test_confirmatory_1h_preset_increases_pairs_and_controls_under_cap():
    argv = [
        "--checkpoint",
        "ckpt.pt",
        "--task",
        "flow_hard",
        "--analysis-preset",
        "confirmatory_1h",
    ]
    args = build_parser().parse_args(argv)

    apply_analysis_preset(args, argv)
    planned = estimate_intervention_forwards(
        pairs=args.num_pairs,
        layers=3,
        heads=8,
        head_mode=args.head_mode,
        targets=3,
        operators=3,
        controls_per_operator=1 + args.matched_random_controls,
        experiments=2,
    )

    assert args.num_pairs == 128
    assert args.num_graphs == 1536
    assert args.matched_random_controls == 4
    assert planned == 34560
    assert planned < args.max_interventions


def test_preset_respects_explicit_overrides():
    argv = [
        "--checkpoint",
        "ckpt.pt",
        "--task",
        "flow_hard",
        "--analysis-preset",
        "core_fast",
        "--num-pairs",
        "12",
        "--operator-masks",
        "saturated_edge",
    ]
    args = build_parser().parse_args(argv)

    apply_analysis_preset(args, argv)

    assert args.num_pairs == 12
    assert args.operator_masks == "saturated_edge"
    assert args.patch_targets == "routing_logits,pair_value,pair_state"


def test_intervention_forward_estimate_scales_with_heads_only_in_per_head_mode():
    layer_level = estimate_intervention_forwards(
        pairs=10,
        layers=3,
        heads=8,
        head_mode="layer",
        targets=3,
        operators=3,
        controls_per_operator=3,
        experiments=2,
    )
    per_head = estimate_intervention_forwards(
        pairs=10,
        layers=3,
        heads=8,
        head_mode="per_head",
        targets=3,
        operators=3,
        controls_per_operator=3,
        experiments=2,
    )

    assert layer_level == 1620
    assert per_head == 8 * layer_level


def test_hypothesis_summary_uses_pair_level_control_normalised_means():
    argv = [
        "--checkpoint",
        "ckpt.pt",
        "--task",
        "flow_hard",
        "--analysis-preset",
        "core_fast",
    ]
    args = build_parser().parse_args(argv)
    apply_analysis_preset(args, argv)
    patch_rows = []
    scrub_rows = []
    for pair_id in [0, 1]:
        for layer in [0, 1]:
            patch_rows.extend(
                [
                    {
                        "pair_id": pair_id,
                        "operator": "saturated_edge",
                        "patch_target": "pair_value",
                        "layer": layer,
                        "head": -1,
                        "control_type": "operator",
                        "restoration": 0.8,
                    },
                    {
                        "pair_id": pair_id,
                        "operator": "saturated_edge",
                        "patch_target": "pair_value",
                        "layer": layer,
                        "head": -1,
                        "control_type": "matched_random",
                        "restoration": 0.3,
                    },
                    {
                        "pair_id": pair_id,
                        "operator": "saturated_edge",
                        "patch_target": "routing_logits",
                        "layer": layer,
                        "head": -1,
                        "control_type": "operator",
                        "restoration": 0.4,
                    },
                    {
                        "pair_id": pair_id,
                        "operator": "saturated_edge",
                        "patch_target": "routing_logits",
                        "layer": layer,
                        "head": -1,
                        "control_type": "matched_random",
                        "restoration": 0.3,
                    },
                ]
            )

    hypothesis_rows, pair_rows = make_hypothesis_outputs(args, patch_rows, scrub_rows)
    h1 = next(row for row in hypothesis_rows if row["hypothesis_id"] == "H1_patch_solver_content_rescue")
    h3 = next(row for row in hypothesis_rows if row["hypothesis_id"] == "H3_patch_content_exceeds_routing")

    assert h1["effect"] == pytest.approx(0.5)
    assert h1["pairs"] == 2
    assert h3["effect"] == pytest.approx(0.4)
    assert {row["pair_id"] for row in pair_rows if row["hypothesis_id"] == h1["hypothesis_id"]} == {0, 1}


def test_operator_target_matrix_is_layer_averaged_not_best_layer():
    rows = [
        {
            "pair_id": 0,
            "operator": "saturated_edge",
            "patch_target": "pair_value",
            "layer": 0,
            "head": -1,
            "control_type": "operator",
            "restoration": 1.0,
        },
        {
            "pair_id": 0,
            "operator": "saturated_edge",
            "patch_target": "pair_value",
            "layer": 1,
            "head": -1,
            "control_type": "operator",
            "restoration": 0.0,
        },
        {
            "pair_id": 0,
            "operator": "saturated_edge",
            "patch_target": "pair_value",
            "layer": 0,
            "head": -1,
            "control_type": "matched_random",
            "restoration": 0.2,
        },
        {
            "pair_id": 0,
            "operator": "saturated_edge",
            "patch_target": "pair_value",
            "layer": 1,
            "head": -1,
            "control_type": "matched_random",
            "restoration": 0.2,
        },
    ]

    matrix = matrix_rows_from_raw(rows, metric="restoration", target_field="patch_target")

    assert matrix[0]["effect"] == pytest.approx(0.3)
