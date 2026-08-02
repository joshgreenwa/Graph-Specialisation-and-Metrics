from pathlib import Path

import pytest

from graph_specialisation_metrics.synthetic.head_specialisation_mediation import (
    Config,
    measure_seed,
    summarise,
)


def test_specialised_decoy_heads_are_not_output_mediators(tmp_path: Path):
    config = Config(
        output_dir=tmp_path,
        train_samples=1_024,
        test_samples=256,
        seeds=(0,),
        bootstrap_replicates=10,
    )
    rows, health = measure_seed(config, seed=0)
    active = {
        (row["head"], row["donor_channel"]): row
        for row in rows
        if row["head_channel"] == row["donor_channel"]
    }

    assert health["test_mae"] < 1.0e-8
    semantic_decoy = active[("semantic_far_decoy", "semantic")]
    structural_decoy = active[("structural_far_decoy", "structural")]
    assert semantic_decoy["S_norm"] == pytest.approx(0.5)
    assert structural_decoy["S_norm"] == pytest.approx(0.5)
    assert semantic_decoy["M_norm"] < 1.0e-8
    assert structural_decoy["M_norm"] < 1.0e-8

    for name, channel in (
        ("semantic_local_signal", "semantic"),
        ("semantic_far_redundant", "semantic"),
        ("structural_mid_signal", "structural"),
        ("structural_far_redundant", "structural"),
    ):
        assert active[(name, channel)]["M_norm"] == pytest.approx(0.5, abs=1.0e-6)


def test_internal_distance_profiles_are_longer_than_mediated_profiles(tmp_path: Path):
    config = Config(
        output_dir=tmp_path,
        train_samples=1_024,
        test_samples=256,
        seeds=(0, 1, 2, 3),
        bootstrap_replicates=10,
    )
    rows = []
    for seed in config.seeds:
        seed_rows, _ = measure_seed(config, seed=seed)
        rows.extend(seed_rows)
    _, _, channels = summarise(config, rows)

    for channel in channels:
        assert channel["S_decoy_share_mean"] == pytest.approx(0.5)
        assert channel["M_decoy_share_mean"] < 1.0e-8
        assert channel["S_expected_distance_mean"] > channel["M_expected_distance_mean"]
