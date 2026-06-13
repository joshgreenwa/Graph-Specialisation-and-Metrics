import tempfile
import csv
import random
from pathlib import Path

import numpy as np
import torch

from graph_specialisation_metrics.core_interpretability_specialisation_metrics import (
    make_content_swapped_batch,
)
from graph_specialisation_metrics.counterfactual_interchange_mediation import (
    cache_data,
    checkpoint_dir,
    cluster_bootstrap_ci,
    collate_records,
    combine_node_graph_stats,
    corr_from_sufficient,
    cv_ridge_summary,
    distance_stratum,
    gnnplus_default_config,
    local_mean_kernel,
    graph_sum_response_delta,
    load_config,
    load_records,
    make_graph_record,
    model_name_for_checkpoint,
    payload_gating_controls,
    pathway_deltas,
    partial_corr,
    response_predictivity_contrast_rows,
    response_feature_sets,
    sample_functional_node_swaps_from_records,
    sample_functional_swaps_from_records,
    node_distance_stratum,
    source_for_intervention,
    teacher_far_response_fraction,
    update_node_stat_accumulator,
    empty_node_stat_accumulator,
    assign_task_quantile_bins,
    functional_node_graph_stats_path,
    functional_node_validity_path,
    functional_q1_gate_features_path,
    functional_q1_graph_coupling_points_path,
    functional_q1_gate_quartile_stats_path,
    functional_response_table_path,
    functional_responses_dir,
    functional_validity_gate_path,
    Q1LayerFields,
    q1_gate_scope_values,
    read_csv_dicts,
    summarise_functional_q1_gates,
    summarise_functional_validity_gate,
)
from graph_specialisation_metrics.long_range_functional_analysis import (
    bucket_for_distance,
    content_relayout_record,
    content_swap_record,
    parse_distance_buckets,
    per_node_linear_readout_deltas,
    sample_far_block_partner_swaps,
    sample_lr_swaps,
    type_compatible_far_relayout,
    weighted_corr,
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


def test_local_mean_gcn_teacher_is_one_hop_local_for_payload_swaps():
    cfg = tiny_cfg("local_mean_gcn")
    graph = make_graph_record("local_mean_gcn", 8, 789, cfg)
    source = source_for_intervention(graph, "local_mean_payload_swap", 0, 1, cfg)
    delta = source["teacher"]["Y"] - graph["teacher"]["Y"]
    dist = torch.minimum(
        graph["struct"]["shortest_path_distance"][:, 0],
        graph["struct"]["shortest_path_distance"][:, 1],
    )
    assert torch.linalg.vector_norm(delta[dist > 1]) < 1.0e-6
    assert torch.allclose(graph["teacher"]["K"], local_mean_kernel(graph["struct"]["adjacency"]))


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


def test_long_range_content_swap_exchanges_all_symbolic_fields_only():
    cfg = tiny_cfg("nearest_anchor_voronoi")
    graph = make_graph_record("nearest_anchor_voronoi", 10, 1234, cfg)
    u, v = 0, 1
    swapped = content_swap_record(graph, u, v, cfg)
    assert torch.allclose(swapped["payload"][u], graph["payload"][v])
    assert torch.allclose(swapped["payload"][v], graph["payload"][u])
    assert swapped["anchor_indicator"][u] == graph["anchor_indicator"][v]
    assert swapped["anchor_indicator"][v] == graph["anchor_indicator"][u]
    assert swapped["anchor_priority"][u] == graph["anchor_priority"][v]
    assert swapped["anchor_priority"][v] == graph["anchor_priority"][u]
    assert torch.allclose(swapped["struct"]["adjacency"], graph["struct"]["adjacency"])
    assert torch.allclose(swapped["struct"]["shortest_path_distance"], graph["struct"]["shortest_path_distance"])


def test_long_range_bucket_parser_and_sampler_are_deterministic():
    cfg = tiny_cfg("ppr_diffusion")
    graph = make_graph_record("ppr_diffusion", 8, 1235, cfg)
    buckets = parse_distance_buckets("1,2,3,4,5-6,7-9,10+")
    assert [bucket_for_distance(d, buckets) for d in [1, 2, 5, 8, 12]] == ["1", "2", "5-6", "7-9", "10+"]
    kwargs = {
        "record": graph,
        "focal_nodes": [0],
        "buckets": buckets,
        "broad_random": 5,
        "targeted_per_bucket": 2,
    }
    first = sample_lr_swaps(**kwargs, rng=random.Random(5))
    second = sample_lr_swaps(**kwargs, rng=random.Random(5))
    assert first == second
    assert len({row["swap_id"] for row in first}) == len(first)
    assert all(row["type_class_u"] == row["type_class_v"] for row in first)


def test_long_range_voronoi_anchor_aware_sampler_targets_assigned_anchor():
    cfg = tiny_cfg("nearest_anchor_voronoi")
    graph = make_graph_record("nearest_anchor_voronoi", 10, 12355, cfg)
    buckets = parse_distance_buckets("1,2,3,4,5-6,7-9,10+")
    focal = 0
    assigned_anchor = int(torch.argmax(graph["teacher"]["K"][focal]).item())
    rows = sample_lr_swaps(
        graph,
        focal_nodes=[focal],
        buckets=buckets,
        broad_random=0,
        targeted_per_bucket=0,
        sampler_mode="voronoi_anchor_aware",
        voronoi_anchor_swaps_per_bucket=4,
        rng=random.Random(19),
    )
    targeted = [row for row in rows if "voronoi_anchor_aware" in row["sample_source"]]
    assert targeted
    assert any(assigned_anchor in {row["u"], row["v"]} for row in targeted)
    assert all(row["target_focal"] == focal for row in targeted)
    assert all(row["target_bucket"] for row in targeted)
    assert {row["distance_basis"] for row in targeted} == {"focal_to_anchor_partner"}


def test_long_range_relayout_is_type_compatible_and_moves_symbolic_content():
    cfg = tiny_cfg("ppr_diffusion")
    graph = make_graph_record("ppr_diffusion", 8, 1236, cfg)
    far_nodes = list(range(graph["n"]))
    old_to_new = type_compatible_far_relayout(graph, far_nodes, random.Random(9))
    for old, new in old_to_new.items():
        assert graph["struct"]["degree"][old] == graph["struct"]["degree"][new]
    relayout = content_relayout_record(graph, old_to_new, cfg, graph_id="relayout")
    moved = [old for old, new in old_to_new.items() if old != new]
    if moved:
        old = moved[0]
        new = old_to_new[old]
        assert torch.allclose(relayout["payload"][new], graph["payload"][old])
    assert torch.allclose(relayout["struct"]["adjacency"], graph["struct"]["adjacency"])


def test_long_range_rq4_partner_swaps_are_type_compatible_and_deterministic():
    cfg = tiny_cfg("ppr_diffusion")
    graph = make_graph_record("ppr_diffusion", 10, 1237, cfg)
    far_nodes = list(range(graph["n"]))
    first = sample_far_block_partner_swaps(graph, far_nodes, partners_per_source=2, rng=random.Random(11))
    second = sample_far_block_partner_swaps(graph, far_nodes, partners_per_source=2, rng=random.Random(11))
    assert first == second
    assert all(row["type_class_u"] == row["type_class_v"] for row in first)
    assert len({row["swap_id"] for row in first}) == len(first)


def test_long_range_linear_readout_p1_p2_identity():
    y_clean = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.5, 0.25]])
    y_source = torch.tensor([[2.0, 1.0], [3.5, 1.0], [-1.0, 0.0]])
    p1, p2 = per_node_linear_readout_deltas(y_clean, y_source)
    assert torch.allclose(p1, y_source - y_clean)
    assert torch.allclose(p2, y_clean - y_source)
    assert torch.allclose(p1.sum(dim=0), y_source.sum(dim=0) - y_clean.sum(dim=0))
    assert torch.allclose(p1, -p2)


def test_long_range_weighted_corr_uses_positive_finite_weights():
    x = np.array([0.0, 1.0, 2.0, np.nan])
    y = np.array([0.0, 2.0, 4.0, 10.0])
    w = np.array([1.0, 0.0, 1.0, 1.0])
    assert weighted_corr(x, y, w) > 0.999


def test_functional_distance_strata_for_two_layer_gnn():
    assert distance_stratum(1, gnn_depth=2) == "d1"
    assert distance_stratum(2, gnn_depth=2) == "d2_to_L"
    assert distance_stratum(3, gnn_depth=2) == "dL1_to_2L"
    assert distance_stratum(4, gnn_depth=2) == "dL1_to_2L"
    assert distance_stratum(5, gnn_depth=2) == "d_gt_2L"


def test_functional_sampler_is_deterministic_and_respects_stratum_caps():
    cfg = tiny_cfg("ppr_diffusion")
    records = [make_graph_record("ppr_diffusion", 8, 500 + idx, cfg) for idx in range(3)]
    kwargs = {
        "family": "ppr_payload_swap",
        "num_graphs": 3,
        "swaps_per_stratum": 2,
        "gnn_depth": 2,
        "seed": 77,
    }
    first = sample_functional_swaps_from_records(records, cfg, "ppr_diffusion", **kwargs)
    second = sample_functional_swaps_from_records(records, cfg, "ppr_diffusion", **kwargs)

    fields = ["graph_id", "u", "v", "d_uv", "stratum", "dy_norm"]
    assert [{field: row[field] for field in fields} for row in first] == [
        {field: row[field] for field in fields} for row in second
    ]
    counts = {}
    for row in first:
        key = (row["graph_id"], row["stratum"])
        counts[key] = counts.get(key, 0) + 1
    assert counts
    assert max(counts.values()) <= 2


def test_functional_graph_sum_response_delta_matches_manual_teacher_delta():
    cfg = tiny_cfg("ppr_diffusion")
    base = make_graph_record("ppr_diffusion", 8, 601, cfg)
    source = source_for_intervention(base, "ppr_payload_swap", 0, 1, cfg)
    delta = graph_sum_response_delta(base, source)
    manual = source["teacher"]["Y"].sum(dim=0) - base["teacher"]["Y"].sum(dim=0)
    assert torch.allclose(delta, manual)


def test_teacher_far_response_fraction_matches_manual_node_mass_ratio():
    cfg = tiny_cfg("ppr_diffusion")
    base = make_graph_record("ppr_diffusion", 8, 602, cfg)
    u, v = 0, 1
    source = source_for_intervention(base, "ppr_payload_swap", u, v, cfg)
    rho_far = teacher_far_response_fraction(base, source, u, v, gnn_depth=2)
    dy_nodes = source["teacher"]["Y"] - base["teacher"]["Y"]
    mass = dy_nodes.square().sum(dim=1)
    dist = torch.minimum(base["struct"]["shortest_path_distance"][:, u], base["struct"]["shortest_path_distance"][:, v])
    manual = float(mass[dist > 2].sum().item() / mass.sum().item()) if float(mass.sum().item()) > 0 else 0.0
    assert abs(rho_far - manual) < 1.0e-8


def test_functional_node_distance_strata_for_two_layer_gnn():
    assert node_distance_stratum(0, 0, 5, 0, gnn_depth=2) == "self"
    assert node_distance_stratum(5, 0, 5, 0, gnn_depth=2) == "self"
    assert node_distance_stratum(3, 0, 5, 1, gnn_depth=2) == "d1"
    assert node_distance_stratum(3, 0, 5, 2, gnn_depth=2) == "d2_to_L"
    assert node_distance_stratum(3, 0, 5, 3, gnn_depth=2) == "d_gt_L"


def test_functional_node_swap_sampler_is_deterministic():
    cfg = tiny_cfg("ppr_diffusion")
    records = [make_graph_record("ppr_diffusion", 8, 700 + idx, cfg) for idx in range(2)]
    first = sample_functional_node_swaps_from_records(
        records,
        "ppr_diffusion",
        family="ppr_payload_swap",
        num_graphs=2,
        swaps_per_graph=5,
        seed=123,
    )
    second = sample_functional_node_swaps_from_records(
        records,
        "ppr_diffusion",
        family="ppr_payload_swap",
        num_graphs=2,
        swaps_per_graph=5,
        seed=123,
    )
    assert first == second
    counts = {}
    for row in first:
        counts[row["graph_id"]] = counts.get(row["graph_id"], 0) + 1
    assert sorted(counts.values()) == [5, 5]


def test_functional_node_sufficient_stats_recover_channel_correlation():
    acc = empty_node_stat_accumulator("ppr_diffusion", "g0", "d_gt_L")
    for value in [0.5, 1.0, 1.5, 2.0]:
        dy = torch.tensor([value, -value])
        grit = 2.0 * dy
        gcn = torch.zeros_like(dy)
        update_node_stat_accumulator(acc, dy=dy, grit_delta=grit, gcn_delta=gcn, epsilon=0.0)
    combined = combine_node_graph_stats([acc])
    corr = corr_from_sufficient(
        combined["channel_n"],
        combined["sum_grit"],
        combined["sum_y"],
        combined["sum_grit2"],
        combined["sum_y2"],
        combined["sum_grit_y"],
    )
    assert corr > 0.999


def test_partial_corr_residualizes_linear_confound():
    z = torch.linspace(-2.0, 2.0, steps=40).numpy()
    signal = torch.sin(torch.linspace(0.0, 6.0, steps=40)).numpy()
    x = 10.0 * z + signal
    y = -4.0 * z + 2.0 * signal
    assert partial_corr(x, y, z) > 0.99


def test_cluster_bootstrap_samples_whole_graph_groups():
    rows = []
    for graph_idx, value in enumerate([1.0, 2.0, 3.0]):
        for _ in range(2):
            rows.append({"graph_id": f"g{graph_idx}", "value": value})

    def stat(sample_rows):
        per_graph_counts = {}
        for row in sample_rows:
            per_graph_counts[row["graph_id"]] = per_graph_counts.get(row["graph_id"], 0) + 1
        assert all(count % 2 == 0 for count in per_graph_counts.values())
        return sum(float(row["value"]) for row in sample_rows) / len(sample_rows)

    point, low, high = cluster_bootstrap_ci(rows, stat, resamples=50, seed=3)
    assert point == 2.0
    assert low <= point <= high


def test_q1_gate_scope_values_match_manual_attention_and_transport():
    attn = torch.tensor(
        [
            [
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.1, 0.2, 0.3, 0.2, 0.2],
            ]
        ]
    )
    msg = torch.ones(1, 5, 5, 1)
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    spd = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
            [3, 2, 1, 0, 1],
            [4, 3, 2, 1, 0],
        ]
    )
    layer = Q1LayerFields(layer=0, attention=attn, message=msg, mask=mask)
    values = q1_gate_scope_values([layer], spd, u=0, v=1, gnn_depth=2)

    assert abs(values["route_swap_far_mass"] - 0.3) < 1.0e-7
    assert abs(values["transport_swap_far_share"] - 0.3) < 1.0e-7
    assert abs(values["route_global_far_share"] - 0.22) < 1.0e-7
    assert abs(values["transport_global_far_share"] - 0.22) < 1.0e-7
    assert values["far_query_count"] == 1.0


def test_q1_quantile_bins_are_task_local_and_deterministic():
    rows = []
    for task in ["a", "b"]:
        for idx, value in enumerate([0.0, 1.0, 2.0, 3.0]):
            rows.append({"task": task, "graph_id": f"{task}{idx}", "gate": value + (10.0 if task == "b" else 0.0)})
    binned = assign_task_quantile_bins(
        rows,
        value_column="gate",
        bin_column="gate_bin",
        label_column="gate_label",
        bin_labels={"q1": "low", "q2": "mid-low", "q3": "mid-high", "q4": "high"},
    )
    by_task = {task: [row["gate_bin"] for row in binned if row["task"] == task] for task in ["a", "b"]}
    assert by_task["a"] == ["q1", "q2", "q3", "q4"]
    assert by_task["b"] == ["q1", "q2", "q3", "q4"]


def write_test_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_q1_summary_preserves_graph_grouping(tmp_path):
    cfg = load_config(None, task="ppr_diffusion", fast_dev_run=True)
    cfg["artifacts"]["root"] = str(tmp_path)
    rows = []
    for graph_idx in range(4):
        for swap_idx in range(4):
            dy = 0.2 * graph_idx + 0.1 * swap_idx
            rows.append(
                {
                    "task": "ppr_diffusion",
                    "family": "ppr_payload_swap",
                    "graph_id": f"g{graph_idx}",
                    "graph_index": graph_idx,
                    "swap_id": f"s{graph_idx}_{swap_idx}",
                    "u": 0,
                    "v": 1,
                    "d_uv": 3,
                    "stratum": "dL1_to_2L",
                    "R_eff_uv": 0.5,
                    "rho_far": 0.25 * swap_idx,
                    "dy_norm": dy,
                    "grit_delta_norm": dy + 0.1,
                    "gcn_plus_delta_norm": dy * 0.5,
                    "grit_teacher_projection": 2.0 * dy + graph_idx,
                    "gcn_plus_teacher_projection": dy + 0.1 * swap_idx,
                    "grit_teacher_cosine": 0.8,
                    "gcn_plus_teacher_cosine": 0.4,
                    "layer": "all",
                    "head": "all",
                    "clean_route_swap_far_mass": dy,
                    "clean_transport_swap_far_share": 1.0 - 0.1 * swap_idx,
                    "clean_route_global_far_share": 0.2,
                    "clean_transport_global_far_share": 0.3,
                    "source_route_swap_far_mass": dy + 0.01,
                    "source_transport_swap_far_share": 1.0 - 0.1 * swap_idx,
                    "source_route_global_far_share": 0.2,
                    "source_transport_global_far_share": 0.3,
                    "delta_route_swap_far_mass": 0.01,
                    "delta_transport_swap_far_share": 0.0,
                    "delta_route_global_far_share": 0.0,
                    "delta_transport_global_far_share": 0.0,
                    "far_query_count": 3,
                    "valid_far_query_head_count": 24,
                    "layer_count": 2,
                    "head_count": 16,
                }
            )
    write_test_csv(functional_q1_gate_features_path(cfg, "ppr_diffusion", 9501), rows)
    summarise_functional_q1_gates(
        cfg,
        tasks=["ppr_diffusion"],
        seed=9501,
        bootstrap_resamples=5,
        min_swaps_per_bin=2,
        min_swaps_per_graph=4,
    )
    points = read_csv_dicts(functional_q1_graph_coupling_points_path(cfg))
    quartiles = read_csv_dicts(functional_q1_gate_quartile_stats_path(cfg))
    assert {row["graph_id"] for row in points} == {"g0", "g1", "g2", "g3"}
    assert any(row["stat"] == "grit_excess_partial_corr" for row in quartiles)


def test_functional_validity_gate_statuses_from_synthetic_artifacts(tmp_path):
    cfg = load_config(None, task="local_mean_gcn", fast_dev_run=True)
    cfg["artifacts"]["root"] = str(tmp_path)
    task = "local_mean_gcn"
    for path in [
        functional_response_table_path(cfg, task, 9101),
        functional_responses_dir(cfg, task) / "functional_clean_context_seed9101.csv",
        functional_node_graph_stats_path(cfg, task, 9301),
        functional_node_validity_path(cfg, task, 9301),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    metrics = tmp_path / "function" / "metrics"
    write_test_csv(metrics / "functional_e1_fingerprint_stats.csv", [{"task": task, "stratum": "d1", "stat": "dy_norm_mean", "mean": 1.0}])
    rho_rows = []
    for idx, bin_name in enumerate(["q1", "q2", "q3", "q4"]):
        rho_rows.append(
            {
                "task": task,
                "rho_far_bin": bin_name,
                "stat": "dy_norm_mean",
                "swaps": 120,
                "rho_far_low": 0.1 * idx,
                "rho_far_high": 0.1 * idx + 0.05,
                "mean": 1.0,
            }
        )
    write_test_csv(metrics / "functional_e1_fingerprint_rhofar_stats.csv", rho_rows)
    write_test_csv(
        metrics / "functional_e1_clean_performance_context.csv",
        [
            {"task": task, "model": "grit", "node_relmse_mean": 0.01, "graph_sum_relmse": 0.01},
            {"task": task, "model": "gcn_plus", "node_relmse_mean": 0.03, "graph_sum_relmse": 0.02},
        ],
    )
    write_test_csv(
        metrics / "functional_e1n_node_fingerprint_stats.csv",
        [
            {"task": task, "stratum": "d_gt_L", "stat": "oracle_demand_mean", "mean": 0.0},
            {"task": task, "stratum": "d_gt_L", "stat": "grit_spurious_silent_norm_mean", "mean": 0.02},
            {"task": task, "stratum": "d_gt_L", "stat": "grit_excess_partial_corr", "mean": 0.5},
        ],
    )
    write_test_csv(
        metrics / "functional_e1n_validity.csv",
        [{"task": task, "beyond_L_gcn_violations": 0, "beyond_L_gcn_max_norm": 0.0}],
    )

    summarise_functional_validity_gate(cfg, tasks=[task])
    rows = read_csv_dicts(functional_validity_gate_path(cfg))
    clean_grit = next(row for row in rows if row["check"] == "Clean performance: grit")
    clean_gcn = next(row for row in rows if row["check"] == "Clean performance: gcn_plus")
    local_signal = next(row for row in rows if row["check"] == "Local control GRIT beyond-L signal")
    assert clean_grit["status"] == "pass"
    assert clean_gcn["status"] == "warn"
    assert local_signal["status"] == "fail"


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


def test_gnnplus_config_uses_separate_checkpoint_namespace():
    tmp_path = Path(tempfile.mkdtemp())
    cfg = gnnplus_default_config("ppr_diffusion", layer_type="gcn")
    cfg["artifacts"]["root"] = str(tmp_path)
    assert cfg["model"]["backend"] == "official_gnnplus"
    assert cfg["model"]["gnnplus"]["layer_type"] == "gcn"
    assert model_name_for_checkpoint(cfg) == "gcn_plus"
    assert checkpoint_dir(cfg, "ppr_diffusion") == tmp_path / "checkpoints" / "gcn_plus" / "ppr_diffusion" / "seed_1001"


def test_cached_train_pool_is_written_for_opt_in_training():
    tmp_path = Path(tempfile.mkdtemp())
    cfg = load_config(None, task="ppr_diffusion", fast_dev_run=True)
    cfg["artifacts"]["root"] = str(tmp_path)
    cfg["training"]["use_cached_train_data"] = True
    cfg["training"]["train_cache_graphs"] = 6
    cache_data(cfg, "ppr_diffusion", force=True)
    records = load_records(tmp_path / "data" / "ppr_diffusion" / "train_pool.pt")
    assert len(records) == 6
    assert records[0]["graph_id"].startswith("ppr_diffusion_train_pool_")


def test_payload_gating_controls_recover_teacher_payload_effect():
    cfg = tiny_cfg("nearest_anchor_voronoi")
    graph = make_graph_record("nearest_anchor_voronoi", 10, 321, cfg)
    anchors = torch.nonzero(graph["anchor_indicator"] > 0.5, as_tuple=False).reshape(-1)
    u = int(anchors[0])
    v = int(torch.nonzero(graph["anchor_indicator"] < 0.5, as_tuple=False)[0])
    source = source_for_intervention(graph, "voronoi_payload_swap", u, v, cfg)
    row = {"base_graph": graph, "source_graph": source, "family": "voronoi_payload_swap", "u": u, "v": v}
    controls = payload_gating_controls(row)
    teacher_norm = torch.linalg.vector_norm(pathway_deltas(graph, source)["M"]).item()
    assert abs(controls["ctrl_km_product_l2"] - teacher_norm) < 1.0e-5
    assert controls["ctrl_num_anchor_swapped"] == 1.0
