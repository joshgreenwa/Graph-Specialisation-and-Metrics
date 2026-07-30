import json
import math

import pytest
import torch

from graph_specialisation_metrics.synthetic.saturation_carriage import (
    ExperimentConfig,
    SaturatedTwoPathStudent,
    measure_model,
    run,
)


def _exact_model(kappa: float) -> SaturatedTwoPathStudent:
    model = SaturatedTwoPathStudent(kappa=kappa, init_seed=0)
    with torch.no_grad():
        model.near_weight.fill_(-1.0)
        model.far_weight.fill_(2.0)
    model.eval()
    return model


def test_saturation_separates_local_jacobian_from_finite_and_beneficial_carriage():
    config = ExperimentConfig(seeds=(0,), kappas=(8.0,), distance=8, gamma=1.0)
    measured = measure_model(_exact_model(8.0), config)

    assert measured["functional_near"] == pytest.approx(2.0)
    assert measured["functional_far"] == pytest.approx(4.0)
    assert measured["functional_range"] == pytest.approx(34.0 / 6.0)
    far_jacobian = 16.0 * (1.0 - math.tanh(8.0) ** 2) / math.tanh(8.0)
    exact_jacobian_range = (1.0 + 8.0 * far_jacobian) / (1.0 + far_jacobian)
    assert measured["jacobian_range"] == pytest.approx(exact_jacobian_range)

    assert measured["beneficial_near"] == pytest.approx(-2.0)
    assert measured["beneficial_far"] == pytest.approx(4.0)
    assert measured["beneficial_sum"] == pytest.approx(2.0)
    assert measured["loss_increase"] == pytest.approx(2.0)
    assert measured["completeness_residual"] < 1e-9
    assert measured["converged"] == pytest.approx(1.0)


def test_cached_measurements_regenerate_figure_without_checkpoints(tmp_path):
    config = ExperimentConfig(
        seeds=(0,),
        kappas=(0.5, 8.0),
        train_points=33,
        train_steps=25,
    )
    first = run(config, output_dir=tmp_path, phase="all", progress=False)

    assert len(first["checkpoints"]) == 2
    assert (tmp_path / "results" / "measurements.csv").is_file()
    assert (tmp_path / "figures" / "finite_intervention_saturation.png").is_file()
    assert (tmp_path / "figures" / "finite_intervention_saturation.pdf").is_file()

    for checkpoint in (tmp_path / "cache" / "checkpoints").rglob("*.pt"):
        checkpoint.unlink()

    second = run(config, output_dir=tmp_path, phase="figures", progress=False)
    assert set(second["figures"]) == {"png", "pdf"}

    metadata = json.loads(
        (tmp_path / "figures" / "finite_intervention_saturation.metadata.json").read_text()
    )
    assert metadata["seeds"] == [0]
    assert metadata["beneficial_sum"] == pytest.approx(metadata["event_loss_increase"])
