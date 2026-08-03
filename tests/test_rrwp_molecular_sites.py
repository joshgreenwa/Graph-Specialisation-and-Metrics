import math
from pathlib import Path

import numpy as np
import torch

from graph_specialisation_metrics.synthetic.rrwp_molecular_sites import (
    CHANNELS,
    NODE_COUNT,
    SITE_COUNT,
    Config,
    _batch_inputs,
    _event_inputs,
    _forward,
    _model,
    _stack_graph_types,
    build_graph_types,
    cue_only_bayes_mse,
    make_dataset,
    measure_scores_and_carriage,
    target_scale,
)


def _config(tmp_path: Path) -> Config:
    return Config(
        output_dir=tmp_path,
        train_examples=16,
        validation_examples=8,
        test_examples=8,
        measurement_examples=3,
        seeds=(0,),
        training_steps=1,
        bootstrap_replicates=100,
    )


def _graph_type(graph_types, *, cycle, cue):
    return next(
        graph_type
        for graph_type in graph_types
        if graph_type.cycle_state == tuple(cycle)
        and graph_type.cue_state == tuple(cue)
    )


def test_remote_ring_closure_changes_only_higher_reporter_rrwp(tmp_path: Path):
    config = _config(tmp_path)
    graph_types = build_graph_types(config)
    open_state = (False,) * SITE_COUNT
    base = _graph_type(graph_types, cycle=open_state, cue=open_state)

    assert base.adjacency.shape == (NODE_COUNT, NODE_COUNT) == (37, 37)
    assert len(base.roles) == 37

    nodes = np.arange(NODE_COUNT)
    registered_pairs = base.adjacency.astype(bool) | np.eye(NODE_COUNT, dtype=bool)
    for site, site_nodes in enumerate(base.sites):
        assert 5 > 2 * config.layers
        assert base.distances[site_nodes.reporter, site_nodes.arm_left[-1]] == 5
        assert base.distances[site_nodes.reporter, site_nodes.arm_right[-1]] == 5

        closed_state = list(open_state)
        closed_state[site] = True
        closed = _graph_type(
            graph_types,
            cycle=closed_state,
            cue=open_state,
        )
        reporter = site_nodes.reporter
        reporter_incident = (
            (nodes[:, None] == reporter) | (nodes[None, :] == reporter)
        ) & registered_pairs

        np.testing.assert_allclose(
            base.rrwp[reporter_incident, :2],
            closed.rrwp[reporter_incident, :2],
            atol=0.0,
            rtol=0.0,
        )
        assert np.max(
            np.abs(
                base.rrwp[reporter_incident, 2:]
                - closed.rrwp[reporter_incident, 2:]
            )
        ) > 1.0e-8


def test_bond_order_proxy_is_visible_without_changing_message_support(
    tmp_path: Path,
):
    config = _config(tmp_path)
    graph_types = build_graph_types(config)
    state = (False,) * SITE_COUNT
    single = _graph_type(graph_types, cycle=state, cue=state)

    for site, site_nodes in enumerate(single.sites):
        double_state = list(state)
        double_state[site] = True
        double = _graph_type(graph_types, cycle=state, cue=double_state)
        reporter = site_nodes.reporter

        np.testing.assert_array_equal(
            single.adjacency.astype(bool),
            double.adjacency.astype(bool),
        )
        assert single.adjacency[reporter].sum() == 3
        assert double.adjacency[reporter].sum() == 7
        assert not np.allclose(
            single.rrwp[reporter, :, 1],
            double.rrwp[reporter, :, 1],
        )


def test_arms_are_parameter_matched_and_local_rrwp_is_truncated(tmp_path: Path):
    config = _config(tmp_path)
    graph_types = build_graph_types(config)
    tensors = _stack_graph_types(graph_types)
    dataset = make_dataset(config, count=8, seed=11)
    index = np.arange(8, dtype=np.int64)
    inputs = {
        arm: _batch_inputs(dataset, index, tensors, arm=arm)
        for arm in ("local", "global", "global_shuffled")
    }

    counts = {
        arm: sum(parameter.numel() for parameter in _model(config, seed).parameters())
        for seed, arm in enumerate(inputs)
    }
    assert len(set(counts.values())) == 1
    assert {
        tuple(value["pair_pe"].shape) for value in inputs.values()
    } == {(8, NODE_COUNT, NODE_COUNT, config.rrwp_horizon + 1)}
    torch.testing.assert_close(
        inputs["local"]["pair_pe"][..., :2],
        inputs["global"]["pair_pe"][..., :2],
    )
    torch.testing.assert_close(
        inputs["global_shuffled"]["pair_pe"][..., :2],
        inputs["global"]["pair_pe"][..., :2],
    )
    assert torch.max(
        torch.abs(
            inputs["global_shuffled"]["pair_pe"][..., 2:]
            - inputs["global"]["pair_pe"][..., 2:]
        )
    ) > 0
    assert torch.count_nonzero(inputs["local"]["pair_pe"][..., 2:]) == 0


def test_generated_target_is_sum_of_true_site_contributions(tmp_path: Path):
    config = _config(tmp_path)
    graph_types = build_graph_types(config)
    tensors = _stack_graph_types(graph_types)
    dataset = make_dataset(config, count=8, seed=17)
    index = np.arange(8, dtype=np.int64)
    inputs = _batch_inputs(dataset, index, tensors, arm="global")

    per_site = (
        math.sqrt(2.0 / 3.0)
        * dataset.cycle_state.astype(np.float32)
        * dataset.values
    )
    np.testing.assert_allclose(
        inputs["target"].numpy(),
        per_site.sum(axis=1),
        atol=1.0e-7,
        rtol=1.0e-7,
    )


def test_semantic_event_is_an_in_support_symmetric_sign_flip(tmp_path: Path):
    config = _config(tmp_path)
    graph_types = build_graph_types(config)
    tensors = _stack_graph_types(graph_types)
    dataset = make_dataset(config, count=8, seed=19)
    index = np.arange(8, dtype=np.int64)
    clean = _batch_inputs(dataset, index, tensors, arm="global")
    event, dose = _event_inputs(
        clean,
        tensors,
        arm="global",
        channel="semantic",
        site=1,
    )
    batch = torch.arange(len(index))
    reporter = clean["reporters"][:, 1]
    clean_values = clean["values"][batch, reporter]
    event_values = event["values"][batch, reporter]
    torch.testing.assert_close(event_values, -clean_values)
    torch.testing.assert_close(dose, 2.0 * torch.abs(clean_values))
    torch.testing.assert_close(torch.abs(event_values), torch.abs(clean_values))


def test_structural_event_visibility_matches_registered_channels(tmp_path: Path):
    config = _config(tmp_path)
    graph_types = build_graph_types(config)
    tensors = _stack_graph_types(graph_types)
    dataset = make_dataset(config, count=8, seed=21)
    index = np.arange(8, dtype=np.int64)
    clean_local = _batch_inputs(dataset, index, tensors, arm="local")
    clean_global = _batch_inputs(dataset, index, tensors, arm="global")

    _, local_proxy_dose = _event_inputs(
        clean_local,
        tensors,
        arm="local",
        channel="structural_proxy",
        site=0,
    )
    _, local_remote_dose = _event_inputs(
        clean_local,
        tensors,
        arm="local",
        channel="structural_remote",
        site=0,
    )
    _, global_remote_dose = _event_inputs(
        clean_global,
        tensors,
        arm="global",
        channel="structural_remote",
        site=0,
    )
    assert torch.all(local_proxy_dose > 0)
    assert torch.count_nonzero(local_remote_dose) == 0
    assert torch.all(global_remote_dose > 0)


def test_target_variance_and_cue_only_bayes_mse_are_registered(tmp_path: Path):
    for remote_weight in (1.0, 0.5):
        config = Config(
            **{
                **_config(tmp_path).__dict__,
                "remote_target_weight": remote_weight,
            }
        )
        dataset = make_dataset(config, count=100_000, seed=29)
        target = dataset.site_contribution.sum(axis=1)
        cycle_probability = np.where(
            dataset.cue_state,
            config.local_clue_reliability,
            1.0 - config.local_clue_reliability,
        )
        fitted_weight = (
            (1.0 - remote_weight) * dataset.cue_state
            + remote_weight * cycle_probability
        )
        cue_prediction = target_scale(config) * np.sum(
            fitted_weight * dataset.values, axis=1
        )
        assert abs(float(np.var(target)) - 1.0) < 0.02
        assert (
            abs(
                float(np.mean((target - cue_prediction) ** 2))
                - cue_only_bayes_mse(config)
            )
            < 0.02
        )


def test_untrained_measurement_reconstructs_heads_and_linear_output(
    tmp_path: Path,
):
    config = _config(tmp_path)
    graph_types = build_graph_types(config)
    tensors = _stack_graph_types(graph_types)
    dataset = make_dataset(config, count=config.test_examples, seed=23)
    model = _model(config, seed=0)

    score_rows, carriage_rows, head_rows, dose_rows, fidelity_rows = (
        measure_scores_and_carriage(
            model,
            dataset,
            graph_types,
            tensors,
            config,
            arm="global",
            seed=0,
        )
    )
    assert score_rows and carriage_rows and head_rows and dose_rows and fidelity_rows
    assert {row["channel"] for row in dose_rows} == set(CHANNELS)

    raw_scores = [row for row in score_rows if row["scale"] == "raw"]
    for head_row in (row for row in head_rows if row["scale"] == "raw"):
        distance_total = sum(
            row["value"]
            for row in raw_scores
            if row["seed"] == head_row["seed"]
            and row["graph"] == head_row["graph"]
            and row["arm"] == head_row["arm"]
            and row["channel"] == head_row["channel"]
            and row["site"] == head_row["site"]
            and row["layer"] == head_row["layer"]
            and row["head"] == head_row["head"]
        )
        np.testing.assert_allclose(distance_total, head_row["value"], atol=1.0e-7)

    index = np.arange(config.measurement_examples, dtype=np.int64)
    clean = _batch_inputs(dataset, index, tensors, arm="global")
    prediction, _, clean_final, _, _ = _forward(
        model, clean, return_details=True
    )
    final_gradient = torch.autograd.grad(prediction.sum(), clean_final)[0]
    event, _ = _event_inputs(
        clean,
        tensors,
        arm="global",
        channel="structural_full",
        site=0,
    )
    with torch.no_grad():
        event_prediction, _, event_final, _, _ = _forward(
            model, event, return_details=True
        )
    signed_carriage = torch.sum(
        (clean_final.detach() - event_final) * final_gradient.detach(),
        dim=(-1, -2),
    )
    torch.testing.assert_close(
        signed_carriage,
        prediction.detach() - event_prediction,
        atol=3.0e-5,
        rtol=3.0e-5,
    )
