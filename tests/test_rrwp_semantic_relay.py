from pathlib import Path

import numpy as np
import torch

from graph_specialisation_metrics.synthetic.rrwp_semantic_relay import (
    Config,
    RRWPSemanticRelay,
    _batch_inputs,
    _stack_graph_types,
    build_graph_types,
    make_dataset,
    measure_scores_and_carriage,
)


def _config(tmp_path: Path) -> Config:
    return Config(
        output_dir=tmp_path,
        train_examples=64,
        validation_examples=32,
        test_examples=32,
        measurement_examples=8,
        seeds=(0,),
        training_steps=20,
        bootstrap_replicates=100,
    )


def test_local_rrwp_is_noisy_but_higher_order_rrwp_identifies_cycle(tmp_path: Path):
    config = _config(tmp_path)
    graph_types = build_graph_types(config)
    for graph_type in graph_types:
        selector = graph_type.route_nodes[-1]
        cycle = graph_type.candidate_nodes[graph_type.cycle_side]
        chain = graph_type.candidate_nodes[1 - graph_type.cycle_side]
        local_margin = graph_type.rrwp[selector, cycle, 1] - graph_type.rrwp[
            selector, chain, 1
        ]
        global_margin = graph_type.rrwp[selector, cycle, 12] - graph_type.rrwp[
            selector, chain, 12
        ]
        assert bool(local_margin < 0) is graph_type.tag_correct
        assert global_margin > 0
        assert graph_type.distances[cycle, selector] == 1
        assert config.arm_length > config.layers


def test_arms_are_parameter_matched_and_local_channels_are_zeroed(tmp_path: Path):
    config = _config(tmp_path)
    graph_types = build_graph_types(config)
    tensors = _stack_graph_types(graph_types)
    dataset = make_dataset(config, count=8, seed=1)
    indices = np.arange(8)
    local = _batch_inputs(dataset, indices, tensors, arm="local")
    global_values = _batch_inputs(dataset, indices, tensors, arm="global")
    np.testing.assert_allclose(
        local["pair_pe"][..., :2], global_values["pair_pe"][..., :2]
    )
    np.testing.assert_allclose(local["pair_pe"][..., 2:], 0.0)

    models = [
        RRWPSemanticRelay(config, role_count=6, seed=0),
        RRWPSemanticRelay(config, role_count=6, seed=1),
    ]
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert counts[0] == counts[1]


def test_exact_head_distance_reconstructs_raw_scores(tmp_path: Path):
    config = _config(tmp_path)
    graph_types = build_graph_types(config)
    tensors = _stack_graph_types(graph_types)
    dataset = make_dataset(config, count=32, seed=2)
    model = RRWPSemanticRelay(config, role_count=6, seed=0).to(dtype=torch.float32)
    score_rows, carriage_rows, head_rows = measure_scores_and_carriage(
        model,
        dataset,
        graph_types,
        tensors,
        config,
        arm="global",
        seed=0,
    )
    assert score_rows and carriage_rows and head_rows
    for channel in ("semantic", "structural"):
        for graph in range(config.measurement_examples):
            for layer in range(config.layers):
                for head in range(config.heads):
                    distance_total = np.mean(
                        [
                            sum(
                                row["value"]
                                for row in score_rows
                                if row["channel"] == channel
                                and row["graph"] == graph
                                and row["layer"] == layer
                                and row["head"] == head
                                and row["source"] == source
                            )
                            for source in (0, 1)
                        ]
                    )
                    raw = next(
                        row["value"]
                        for row in head_rows
                        if row["channel"] == channel
                        and row["graph"] == graph
                        and row["layer"] == layer
                        and row["head"] == head
                    )
                    np.testing.assert_allclose(distance_total, raw, atol=1.0e-8)
