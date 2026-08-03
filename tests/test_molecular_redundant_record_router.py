from pathlib import Path

import pytest
import torch

from graph_specialisation_metrics.synthetic.molecular_redundant_record_router import (
    Config,
    MolecularSupport,
    RedundantRecordRouter,
    evaluate,
    make_router_dataset,
    train_model,
)


def _datasets():
    supports = [
        MolecularSupport(
            graph_id=index,
            anchor=0,
            record_nodes=(1, 4, 5, 6),
            record_distances=(1, 4, 4, 6),
        )
        for index in range(16)
    ]
    return (
        make_router_dataset(supports[:12], examples_per_graph=8, seed=1),
        make_router_dataset(supports[12:14], examples_per_graph=8, seed=2),
        make_router_dataset(supports[14:], examples_per_graph=8, seed=3),
    )


def test_local_copy_is_exact_while_dense_route_can_be_used(tmp_path: Path):
    train, validation, test = _datasets()
    config = Config(
        output_dir=tmp_path,
        data_root=tmp_path,
        train_graphs=12,
        val_graphs=2,
        test_graphs=2,
        seeds=(0,),
        local_mixes=(0.5,),
        max_steps=4_000,
    )
    model, training = train_model(
        config, train, validation, local_mix=0.5, seed=0
    )
    result = evaluate(model, test)

    assert training["routing_accuracy"] == pytest.approx(1.0)
    assert result["routing_accuracy"] == pytest.approx(1.0)
    assert result["full_mae"] < 1.0e-2
    assert result["local_refit_mae"] == pytest.approx(0.0)
    assert result["frozen_global_removed_mae"] > 0.45
    assert result["interaction_effect_mass"] > 0.5
    assert result["interaction_far_share"] > 0.7


def test_interaction_magnitude_scales_with_redundant_global_mix():
    _, _, test = _datasets()
    low_local = RedundantRecordRouter(0.25, seed=0)
    high_local = RedundantRecordRouter(0.75, seed=0)
    with torch.no_grad():
        for model in (low_local, high_local):
            model.semantic_logits.copy_(torch.tensor(((5.0, -5.0), (-5.0, 5.0))))
            model.structural_logits.copy_(torch.tensor(((5.0, -5.0), (-5.0, 5.0))))
            model.distance_bias.zero_()

    low_result = evaluate(low_local, test)
    high_result = evaluate(high_local, test)
    assert low_result["full_mae"] < 1.0e-3
    assert high_result["full_mae"] < 1.0e-3
    assert low_result["interaction_effect_mass"] == pytest.approx(
        3 * high_result["interaction_effect_mass"], rel=1.0e-6
    )
    assert low_result["interaction_expected_distance"] == pytest.approx(
        high_result["interaction_expected_distance"], rel=1.0e-9
    )
