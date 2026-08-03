from pathlib import Path

import numpy as np

from graph_specialisation_metrics.synthetic.rrwp_score_distance_comparison import (
    Config,
    run_experiment,
)


def _config(tmp_path: Path) -> Config:
    return Config(
        output_dir=tmp_path,
        train_examples=2_048,
        test_examples=2_048,
        seeds=(0, 1, 2),
        training_steps=2_000,
    )


def test_global_rrwp_outperforms_local_rrwp(tmp_path: Path):
    results, _, health = run_experiment(_config(tmp_path))
    free = results[results["calibration"] == "free"]
    local = float(free.loc[free["scope"] == "local", "test_mse"].mean())
    global_value = float(free.loc[free["scope"] == "global", "test_mse"].mean())

    assert local > 0.30
    assert global_value < 0.12
    assert local - global_value > 0.20
    positive_margins = [
        value > 0
        for item in health["seeds"].values()
        for value in item["global_type_margins"]
    ]
    assert float(np.mean(positive_margins)) > 0.80


def test_confidence_matching_aligns_raw_scores_but_preserves_performance_gap(tmp_path: Path):
    results, _, _ = run_experiment(_config(tmp_path))
    matched = results[results["calibration"] == "matched"].groupby("scope").mean(numeric_only=True)

    assert matched.loc["local", "test_mse"] - matched.loc["global", "test_mse"] > 0.15
    for channel in ("semantic", "structural"):
        left = float(matched.loc["local", f"{channel}_score_raw"])
        right = float(matched.loc["global", f"{channel}_score_raw"])
        assert abs(left - right) / max(left, right) < 0.05


def test_distance_profiles_reconstruct_scores_and_remain_close(tmp_path: Path):
    results, distances, _ = run_experiment(_config(tmp_path))
    for (seed, model, channel), rows in distances.groupby(["seed", "model", "channel"]):
        score = float(
            results.loc[
                (results["seed"] == seed) & (results["model"] == model),
                f"{channel}_score_raw",
            ].iloc[0]
        )
        np.testing.assert_allclose(rows["raw_mass"].sum(), score, atol=1.0e-12)
        np.testing.assert_allclose(rows["normalized_mass"].sum(), 1.0, atol=1.0e-12)

    for calibration in ("free", "matched"):
        for channel in ("semantic", "structural"):
            subset = distances[
                (distances["calibration"] == calibration)
                & (distances["channel"] == channel)
            ]
            profiles = subset.groupby(["scope", "distance"])["normalized_mass"].mean().unstack()
            tv = 0.5 * np.abs(profiles.loc["local"] - profiles.loc["global"]).sum()
            assert float(tv) < 0.03
