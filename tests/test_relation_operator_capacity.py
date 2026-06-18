import math

import torch

from graph_specialisation_metrics.relation_operator_capacity import (
    CAPACITY_CROSSOVER_MODELS,
    CONTROLLED_MODELS,
    ExperimentSpec,
    batch_from_split,
    build_model,
    count_parameters,
    effective_relation_maps,
    graph_tensors,
    make_split,
    parameter_match_grid,
    plot_all,
    realised_rank,
    relation_weights,
    teacher_floor,
    train_one,
    write_json,
)


def test_relation_operator_teacher_matches_manual_sum():
    spec = ExperimentSpec(relation_types=2, input_dim=3, target_dim=3, train_size=4, val_size=2, test_size=2)
    split = make_split(spec, 4, 123)
    weights = relation_weights(spec)
    manual = (
        torch.matmul(weights[0], split["content"][0, 1])
        + torch.matmul(weights[1], split["content"][0, 2])
    ) / math.sqrt(2.0)
    assert torch.allclose(split["y"][0], manual, atol=1.0e-6)


def test_eckart_young_floor_matches_svd_tail_energy():
    spec = ExperimentSpec(relation_types=6, input_dim=4, target_dim=4, heads=2, transport_bases=2)
    floor = teacher_floor(spec)
    weights = relation_weights(spec).reshape(spec.relation_types, -1).double() / math.sqrt(float(spec.relation_types))
    s = torch.linalg.svdvals(weights)
    total = (s**2).sum()
    expected_routing = (s[spec.heads :] ** 2).sum() / total
    expected_full = (s[spec.heads * spec.transport_bases :] ** 2).sum() / total
    assert abs(float(floor["routing_floor"]) - float(expected_routing)) < 1.0e-8
    assert abs(float(floor["full_transport_floor"]) - float(expected_full)) < 1.0e-8


def test_controlled_rank_ceilings_hold_for_realised_relation_maps():
    device = torch.device("cpu")
    routing_spec = ExperimentSpec(relation_types=6, input_dim=3, target_dim=3, heads=2, transport_bases=3)
    routing = build_model(routing_spec, "routing_dense").to(device).eval()
    routing_rank, _ = realised_rank(effective_relation_maps(routing, routing_spec, device))
    assert routing_rank <= routing_spec.heads

    full_spec = ExperimentSpec(relation_types=8, input_dim=3, target_dim=3, heads=2, transport_bases=3)
    full = build_model(full_spec, "full_dense").to(device).eval()
    full_rank, _ = realised_rank(effective_relation_maps(full, full_spec, device))
    assert full_rank <= full_spec.heads * full_spec.transport_bases


def test_capacity_crossover_rank_ceilings_hold_for_global_variants():
    device = torch.device("cpu")
    spec = ExperimentSpec(relation_types=6, input_dim=3, target_dim=3, heads=2, transport_bases=3, task_mode="global")
    routing = build_model(spec, "capacity_routing_only").to(device).eval()
    routing_rank, _ = realised_rank(effective_relation_maps(routing, spec, device))
    assert routing_rank <= spec.heads

    transport = build_model(spec, "capacity_transport_only").to(device).eval()
    transport_rank, _ = realised_rank(effective_relation_maps(transport, spec, device))
    assert transport_rank <= spec.heads * spec.transport_bases

    full = build_model(spec, "capacity_full_relation_transport").to(device).eval()
    full_rank, _ = realised_rank(effective_relation_maps(full, spec, device))
    assert full_rank <= spec.heads * spec.transport_bases


def test_sparse_one_hop_cannot_access_global_non_edge_sources_at_one_layer():
    spec = ExperimentSpec(relation_types=3, input_dim=4, target_dim=4, task_mode="global", layers=1)
    split = make_split(spec, 1, 123)
    model = build_model(spec, "full_1hop").eval()
    with torch.no_grad():
        out = model(batch_from_split(split, torch.tensor([0])))
    assert torch.allclose(out, torch.zeros_like(out), atol=1.0e-7)


def test_local_one_hop_support_contains_all_teacher_sources():
    spec = ExperimentSpec(relation_types=5, input_dim=4, target_dim=4, task_mode="local")
    graph = graph_tensors(spec)
    assert torch.all(graph["support_sparse"][0, 1 : spec.relation_types + 1])


def test_parameter_match_grid_selects_closest_width_and_records_counts():
    spec = ExperimentSpec(relation_types=3, input_dim=4, target_dim=4, heads=2, hidden_dim=8)
    target = count_parameters(build_model(spec, "full_dense"))
    matched, rows = parameter_match_grid(spec, "graphormer_manual", target_params=target, width_grid=[4, 8, 16])
    gaps = [row["abs_parameter_gap"] for row in rows]
    assert matched.hidden_dim in {4, 8, 16}
    assert min(gaps) == next(row["abs_parameter_gap"] for row in rows if row["selected"])
    assert all("candidate_parameters" in row for row in rows)


def test_resume_logic_skips_complete_checkpoint(tmp_path):
    spec = ExperimentSpec(relation_types=2, input_dim=3, target_dim=3, train_size=8, val_size=4, test_size=4)
    first = train_one(
        root=tmp_path,
        experiment="resume",
        spec=spec,
        model_name="routing_dense",
        seed=1001,
        device=torch.device("cpu"),
        batch_size=4,
        eval_batch_size=4,
        max_epochs=1,
        patience=1,
        lr=1.0e-3,
        weight_decay=0.0,
        use_amp=False,
        overwrite=False,
    )
    second = train_one(
        root=tmp_path,
        experiment="resume",
        spec=spec,
        model_name="routing_dense",
        seed=1001,
        device=torch.device("cpu"),
        batch_size=4,
        eval_batch_size=4,
        max_epochs=10,
        patience=10,
        lr=1.0e-3,
        weight_decay=0.0,
        use_amp=False,
        overwrite=False,
    )
    assert second["best_epoch"] == first["best_epoch"]
    assert second["test_rel_mse"] == first["test_rel_mse"]


def test_smoke_plot_generation_writes_required_figures(tmp_path):
    for experiment in ["crossover", "transport_support", "overglobalisation_fixed", "overglobalisation", "depth_escape"]:
        for task_mode in ["local", "global"]:
            for relation_types in [1, 2]:
                for layers in [1, 2]:
                    for model in CONTROLLED_MODELS:
                        if experiment == "depth_escape" and task_mode == "global":
                            continue
                        path = (
                            tmp_path
                            / "checkpoints"
                            / experiment
                            / task_mode
                            / f"R{relation_types}"
                            / f"L{layers}"
                            / "n0_s0"
                            / model
                            / "seed_1001"
                            / "complete.json"
                        )
                        write_json(
                            path,
                            {
                                "experiment": experiment,
                                "task_mode": task_mode,
                                "model": model,
                                "model_label": model,
                                "seed": 1001,
                                "relation_types": relation_types,
                                "input_dim": 4,
                                "target_dim": 4,
                                "hidden_dim": 8,
                                "heads": 2,
                                "transport_bases": 2,
                                "layers": layers,
                                "noise_nodes": 0,
                                "noise_sigma": 0.0,
                                "parameters": 10,
                                "best_epoch": 1,
                                "best_val_rel_mse": 0.1,
                                "test_rel_mse": 0.1 + 0.01 * relation_types + 0.001 * layers,
                                "test_mse": 0.1,
                                "test_mae": 0.1,
                                "realised_rank": min(relation_types, 2),
                                "realised_singular_values": "1;0.5",
                                "routing_floor": 0.1,
                                "full_transport_floor": 0.0,
                                "teacher_rank": relation_types,
                                "teacher_singular_values": "1;0.5",
                            },
                        )
    for relation_types in [2, 4]:
        for model in CAPACITY_CROSSOVER_MODELS:
            path = (
                tmp_path
                / "checkpoints"
                / "capacity_crossover_global"
                / "global"
                / f"R{relation_types}"
                / "L1"
                / "n0_s0"
                / model
                / "seed_1001"
                / "complete.json"
            )
            write_json(
                path,
                {
                    "experiment": "capacity_crossover_global",
                    "task_mode": "global",
                    "model": model,
                    "model_label": model,
                    "seed": 1001,
                    "relation_types": relation_types,
                    "input_dim": 32,
                    "target_dim": 32,
                    "hidden_dim": 32,
                    "heads": 4,
                    "transport_bases": 4,
                    "layers": 1,
                    "noise_nodes": 0,
                    "noise_sigma": 0.0,
                    "parameters": 10,
                    "best_epoch": 1,
                    "best_val_rel_mse": 0.1,
                    "test_rel_mse": 0.1 + 0.01 * relation_types,
                    "test_mse": 0.1,
                    "test_mae": 0.1,
                    "realised_rank": min(relation_types, 4),
                    "realised_singular_values": "1;0.5",
                    "routing_floor": 0.1,
                    "full_transport_floor": 0.0,
                    "teacher_rank": relation_types,
                    "teacher_singular_values": "1;0.5",
                },
            )

    class Args:
        output_root = tmp_path

    plot_all(Args())
    expected = [
        "relation_rank_crossover_local.pdf",
        "relation_rank_crossover_global_capacity_h4.pdf",
        "transport_support_local_relation_operator.pdf",
        "transport_support_global_relation_operator.pdf",
        "overglobalisation_irrelevant_content.pdf",
        "depth_escape_routing_only.pdf",
    ]
    for name in expected:
        path = tmp_path / "figures" / name
        assert path.exists()
        assert path.stat().st_size > 0
