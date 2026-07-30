import json

import pytest
import torch

from graph_specialisation_metrics.synthetic.softmax_routing_carriage import (
    ExperimentConfig,
    SoftmaxKeyValueRouter,
    local_codes,
    measure_model,
    run,
)


def _sharp_exact_router(config: ExperimentConfig) -> SoftmaxKeyValueRouter:
    model = SoftmaxKeyValueRouter(
        num_keys=config.num_keys,
        init_seed=0,
        remote_scale=config.remote_scale,
    )
    with torch.no_grad():
        model.compatibility.zero_()
        model.compatibility.diagonal().fill_(6.0)
        model.near_weight.copy_(local_codes(config))
    model.eval()
    return model


def test_finite_carriage_tracks_hard_routing_when_local_jacobian_saturates():
    config = ExperimentConfig(
        seeds=(0,),
        sharpness_multipliers=(0.5, 4.0),
        num_keys=4,
        evaluation_graphs=12,
        train_steps=1,
    )
    records = measure_model(_sharp_exact_router(config), config)
    sharp = records[-1]

    assert sharp["matched_probability"] > 1.0 - 1e-9
    assert sharp["test_mae"] < 1e-9
    assert sharp["functional_range"] == pytest.approx(
        sharp["oracle_range"],
        abs=1e-7,
    )
    assert sharp["jacobian_range"] == pytest.approx(1.0, abs=1e-5)
    assert sharp["functional_far_share"] == pytest.approx(
        sharp["oracle_far_share"],
        abs=1e-7,
    )
    assert sharp["jacobian_far_share"] < 1e-5
    assert sharp["output_completeness_error"] < 1e-12


def test_cached_softmax_measurements_regenerate_figure_without_checkpoints(tmp_path):
    config = ExperimentConfig(
        seeds=(0,),
        sharpness_multipliers=(0.5, 2.0),
        num_keys=3,
        train_steps=30,
        train_batch_size=24,
        evaluation_graphs=6,
    )
    first = run(config, output_dir=tmp_path, phase="all", progress=False)

    assert len(first["checkpoints"]) == 1
    assert (tmp_path / "results" / "measurements.csv").is_file()
    assert (tmp_path / "figures" / "softmax_routing_carriage.png").is_file()
    assert (tmp_path / "figures" / "softmax_routing_carriage.pdf").is_file()

    for checkpoint in (tmp_path / "cache" / "checkpoints").glob("*.pt"):
        checkpoint.unlink()

    second = run(config, output_dir=tmp_path, phase="figures", progress=False)
    assert set(second["figures"]) == {"png", "pdf"}

    metadata = json.loads(
        (tmp_path / "figures" / "softmax_routing_carriage.metadata.json").read_text()
    )
    assert metadata["seeds"] == [0]
    assert metadata["highest_sharpness_multiplier"] == pytest.approx(2.0)
