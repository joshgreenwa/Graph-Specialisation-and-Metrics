import torch

from graph_specialisation_metrics.core_interpretability_specialisation_metrics import (
    make_content_swapped_batch,
)
from graph_specialisation_metrics.counterfactual_interchange_mediation import (
    collate_records,
    cv_ridge_summary,
    load_config,
    make_graph_record,
    pathway_deltas,
    response_predictivity_contrast_rows,
    response_feature_sets,
    source_for_intervention,
)


def tiny_cfg(task: str):
    cfg = load_config(None, task=task, fast_dev_run=True)
    cfg["graph_generator"]["n_train_min"] = 8
    cfg["graph_generator"]["n_train_max"] = 8
    cfg["graph_generator"]["chord_edges_per_node"] = 0.5
    cfg["graph_generator"]["planted_rings"]["probability"] = 0.0
    cfg["graph_generator"]["unique_hub"]["probability"] = 0.0
    cfg["teachers"]["voronoi"]["n_anchors"] = 3
    return cfg


def test_ppr_teacher_decomposition_and_payload_pathway():
    cfg = tiny_cfg("ppr_diffusion")
    graph = make_graph_record("ppr_diffusion", 8, 123, cfg)

    k = graph["teacher"]["K"]
    m = graph["teacher"]["M"]
    y = graph["teacher"]["Y"]
    assert torch.allclose(k.sum(dim=1), torch.ones(8), atol=1.0e-6)
    assert torch.allclose(y, k @ m, atol=1.0e-6)

    source = source_for_intervention(graph, "ppr_payload_swap", 0, 1, cfg)
    deltas = pathway_deltas(graph, source)
    assert torch.allclose(source["teacher"]["K"], graph["teacher"]["K"])
    assert torch.linalg.vector_norm(deltas["K"]) == 0
    assert torch.allclose(deltas["M"], deltas["total"], atol=1.0e-6)


def test_voronoi_teacher_kernel_and_marker_pathway():
    cfg = tiny_cfg("nearest_anchor_voronoi")
    graph = make_graph_record("nearest_anchor_voronoi", 10, 456, cfg)
    anchors = torch.nonzero(graph["anchor_indicator"] > 0.5, as_tuple=False).reshape(-1)
    non_anchor = int(torch.nonzero(graph["anchor_indicator"] < 0.5, as_tuple=False)[0])
    anchor = int(anchors[0])

    k = graph["teacher"]["K"]
    assert int(anchors.numel()) == 3
    assert torch.allclose(k.sum(dim=1), torch.ones(graph["n"]))
    assert torch.all((k == 0) | (k == 1))

    source = source_for_intervention(graph, "voronoi_anchor_marker_swap", anchor, non_anchor, cfg)
    deltas = pathway_deltas(graph, source)
    assert torch.allclose(source["teacher"]["M"], graph["teacher"]["M"])
    assert torch.linalg.vector_norm(deltas["M"]) == 0
    assert torch.allclose(deltas["K"], deltas["total"], atol=1.0e-6)


def test_content_swap_moves_continuous_x_payload_fields():
    cfg = tiny_cfg("ppr_diffusion")
    records = [make_graph_record("ppr_diffusion", 8, 100 + idx, cfg) for idx in range(2)]
    batch = collate_records(records)
    perm = torch.arange(batch.max_nodes).repeat(batch.num_graphs, 1)
    perm[:, 0] = 1
    perm[:, 1] = 0

    swapped = make_content_swapped_batch(batch, perm)
    assert torch.allclose(swapped.x[:, 0], batch.x[:, 1])
    assert torch.allclose(swapped.payload[:, 1], batch.payload[:, 0])
    assert torch.allclose(swapped.adj, batch.adj)


def test_response_predictivity_grouped_cv_finds_simple_signal():
    rows = []
    for graph_idx in range(12):
        for item_idx in range(4):
            signal = graph_idx * 0.1 + item_idx * 0.25
            rows.append(
                {
                    "graph_id": f"g{graph_idx}",
                    "L0_routing_follow_H0": signal,
                    "L0_routing_invariant_H0": 1.0 - signal * 0.1,
                    "L0_transport_follow_H0": 0.05 * item_idx,
                    "teacher_pathway_norm": 2.0 * signal + 0.5,
                }
            )

    feature_sets = response_feature_sets(
        [
            "L0_routing_follow_H0",
            "L0_routing_invariant_H0",
            "L0_transport_follow_H0",
        ]
    )
    summary = cv_ridge_summary(
        rows,
        feature_columns=feature_sets["routing_scores"],
        target_column="teacher_pathway_norm",
        folds=4,
        alpha=0.01,
        seed=7,
    )
    assert summary["folds"] == 4
    assert summary["groups"] == 12
    assert summary["r2_mean"] > 0.95


def test_response_predictivity_contrasts_mark_matched_sets():
    rows = [
        {"family": "ppr_payload_swap", "target": "teacher_pathway_norm", "feature_set": "intercept_only", "r2_mean": "0.0"},
        {"family": "ppr_payload_swap", "target": "teacher_pathway_norm", "feature_set": "transport_scores", "r2_mean": "0.8", "pearson_oof": "0.9", "spearman_oof": "0.85"},
        {"family": "ppr_payload_swap", "target": "teacher_pathway_norm", "feature_set": "routing_scores", "r2_mean": "0.3"},
        {"family": "ppr_payload_swap", "target": "teacher_pathway_norm", "feature_set": "all_scores", "r2_mean": "0.95"},
        {"family": "ppr_payload_swap", "target": "student_teacher_pathway_projection", "feature_set": "intercept_only", "r2_mean": "0.0"},
        {"family": "ppr_payload_swap", "target": "student_teacher_pathway_projection", "feature_set": "transport_scores", "r2_mean": "0.7"},
        {"family": "ppr_payload_swap", "target": "student_teacher_pathway_projection", "feature_set": "routing_scores", "r2_mean": "0.2"},
        {"family": "ppr_payload_swap", "target": "student_teacher_pathway_projection", "feature_set": "all_scores", "r2_mean": "0.9"},
    ]
    contrasts = response_predictivity_contrast_rows(rows, "ppr_diffusion")
    teacher = next(row for row in contrasts if row["target"] == "teacher_pathway_norm")
    student = next(row for row in contrasts if row["target"] == "student_teacher_pathway_projection")
    assert teacher["matched_feature_set"] == "transport_scores"
    assert teacher["mismatched_feature_set"] == "routing_scores"
    assert teacher["matched_minus_mismatched_r2"] == 0.5
    assert student["matched_minus_intercept_r2"] == 0.7
