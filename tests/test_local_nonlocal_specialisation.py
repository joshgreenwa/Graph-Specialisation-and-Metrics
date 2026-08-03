from pathlib import Path

import numpy as np

from graph_specialisation_metrics.synthetic.local_nonlocal_specialisation import (
    Config,
    run_mechanisms,
    run_reliability_sweep,
)


def _config(tmp_path: Path) -> Config:
    return Config(
        output_dir=tmp_path,
        train_examples=2_048,
        test_examples=2_048,
        seeds=(0, 1, 2),
        reliability_sweep=(0.55, 0.75, 0.9, 1.0),
    )


def test_performance_differs_while_normalized_head_coordinates_match(tmp_path: Path):
    results, _ = run_mechanisms(_config(tmp_path))
    short = results[results["mechanism"] == "local_short"]
    multihop = results[results["mechanism"] == "local_multihop"]

    assert float(short["test_mse"].mean()) > 0.30
    assert float(multihop["test_mse"].mean()) < 1.0e-3
    np.testing.assert_allclose(results["semantic_head_D_rel"], 1.0, atol=1.0e-10)
    np.testing.assert_allclose(results["structural_head_D_rel"], -1.0, atol=1.0e-10)
    np.testing.assert_allclose(results["semantic_head_J"], 1.0, atol=1.0e-10)
    np.testing.assert_allclose(results["structural_head_J"], 1.0, atol=1.0e-10)
    np.testing.assert_allclose(results["semantic_only_D_rel"], 1.0, atol=1.0e-10)
    np.testing.assert_allclose(results["semantic_leaning_D_rel"], 0.5, atol=1.0e-10)
    np.testing.assert_allclose(results["structural_leaning_D_rel"], -0.5, atol=1.0e-10)
    np.testing.assert_allclose(results["structural_only_D_rel"], -1.0, atol=1.0e-10)


def test_distance_null_and_transport_positive_control(tmp_path: Path):
    results, distances = run_mechanisms(_config(tmp_path))
    local = results[results["mechanism"].isin(("local_short", "local_multihop"))]
    dense = results[results["mechanism"] == "dense_transport"]

    assert set(local["semantic_carrier_distance"]) == {0}
    assert set(local["structural_carrier_distance"]) == {0}
    assert set(dense["semantic_carrier_distance"]) == {4}
    assert set(dense["structural_carrier_distance"]) == {0}
    structural = distances[distances["channel"] == "structural"]
    assert set(structural["distance"]) == {0}
    assert set(structural["pe_origin_distance"]) == {1, 4}


def test_raw_structural_score_is_not_monotonic_with_clue_usefulness(tmp_path: Path):
    sweep = run_reliability_sweep(_config(tmp_path))
    grouped = sweep.groupby("clue_reliability").mean(numeric_only=True)

    assert grouped.loc[1.0, "test_mse"] < grouped.loc[0.75, "test_mse"]
    assert grouped.loc[1.0, "structural_swap_loss_increase"] > grouped.loc[
        0.75, "structural_swap_loss_increase"
    ]
    assert grouped.loc[1.0, "structural_score_raw"] < grouped.loc[
        0.75, "structural_score_raw"
    ]
    np.testing.assert_allclose(grouped["semantic_head_D_rel"], 1.0, atol=1.0e-10)
    np.testing.assert_allclose(grouped["structural_head_D_rel"], -1.0, atol=1.0e-10)
