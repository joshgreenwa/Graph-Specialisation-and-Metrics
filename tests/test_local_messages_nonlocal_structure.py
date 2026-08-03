from pathlib import Path

import numpy as np

from graph_specialisation_metrics.synthetic.local_messages_nonlocal_structure import (
    ExperimentConfig,
    candidate_rrwp_features,
    make_cycle_chain_graph,
    remote_edge_sensitivity,
    rooted_balls_are_isomorphic,
    train_horizon_probes,
)


def test_remote_edge_preserves_short_rrwp_but_changes_local_pair_at_high_order():
    table, metadata = remote_edge_sensitivity(max_horizon=16)

    assert metadata["remote_edge_distance_from_candidate"] == 6
    np.testing.assert_allclose(table.loc[table["horizon"] <= 1, "delta_self"], 0.0)
    np.testing.assert_allclose(table.loc[table["horizon"] <= 1, "delta_bond"], 0.0)
    changed_self = table.loc[table["delta_self"] > 1.0e-14, "horizon"]
    changed_bond = table.loc[table["delta_bond"] > 1.0e-14, "horizon"]
    assert int(changed_self.iloc[0]) == 12
    assert int(changed_bond.iloc[0]) == 11


def test_candidate_local_balls_match_and_rrwp_onset_tracks_cycle_length():
    max_horizon = 12
    for cycle_size in (6, 8, 10):
        graph, cycle_candidate, chain_candidate = make_cycle_chain_graph(
            cycle_size,
            max_horizon=max_horizon,
        )
        assert rooted_balls_are_isomorphic(
            graph,
            cycle_candidate,
            chain_candidate,
            radius=2,
        )
        features, _ = candidate_rrwp_features(
            cycle_size,
            max_horizon=max_horizon,
        )
        difference = np.abs(features[0] - features[1])
        np.testing.assert_allclose(difference[:cycle_size], 0.0, atol=1.0e-14)
        assert difference[cycle_size] > 1.0e-14


def test_parameter_matched_gate_is_at_chance_short_and_accurate_with_rrwp(tmp_path: Path):
    config = ExperimentConfig(
        output_dir=tmp_path,
        cycle_sizes=(6,),
        max_horizon=8,
        seeds=(0, 1, 2),
        training_steps=2_000,
        test_examples=2_048,
    )
    table, _, _ = train_horizon_probes(config)
    short = table[table["horizon"] == 1]
    long = table[table["horizon"] == 8]

    assert short["parameter_count"].nunique() == 1
    assert long["parameter_count"].unique().item() == short["parameter_count"].unique().item()
    np.testing.assert_allclose(short["cycle_gate_mass"], 0.5, atol=1.0e-12)
    assert 0.43 < float(short["test_mse"].mean()) < 0.57
    assert float(long["cycle_gate_mass"].mean()) > 0.95
    assert float(long["test_mse"].mean()) < 0.01
