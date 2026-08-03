import numpy as np
import pytest

from graph_specialisation_metrics.synthetic.softmax_record_bank_routing import (
    ExperimentConfig,
    measure_four_state_field,
    run,
    teacher_contributions,
)


def test_matched_teachers_have_same_marginals_but_different_interaction():
    additive = measure_four_state_field(teacher_contributions(task="additive"))
    gated = measure_four_state_field(teacher_contributions(task="gated"))

    np.testing.assert_allclose(
        additive["semantic_distance_profile"], [0.4, 0.6, 0.0]
    )
    np.testing.assert_allclose(
        additive["structural_distance_profile"], [0.4, 0.6, 0.0]
    )
    np.testing.assert_allclose(
        additive["semantic_distance_profile"], gated["semantic_distance_profile"]
    )
    np.testing.assert_allclose(
        additive["structural_distance_profile"], gated["structural_distance_profile"]
    )
    assert additive["marginal_profile_tv"] == pytest.approx(0.0)
    assert gated["marginal_profile_tv"] == pytest.approx(0.0)
    np.testing.assert_allclose(additive["carrier_interaction"], np.zeros(4))
    assert additive["interaction_estimable"] is False
    np.testing.assert_allclose(
        gated["carrier_interaction"], [1.0, -1.5, -1.5, 4.0]
    )
    assert gated["output_interaction"] == pytest.approx(2.0)
    assert gated["interaction_distance_profile"].tolist() == pytest.approx(
        [0.125, 0.375, 0.5]
    )
    assert gated["far_interaction_share"] == pytest.approx(0.875)
    assert gated["interaction_expected_distance"] == pytest.approx(4.625)


def test_learned_softmax_router_recovers_matched_null_and_gated_effect():
    result = run(
        ExperimentConfig(
            seeds=(0, 1, 2),
            hidden_dim=24,
            heads=4,
            max_steps=8_000,
            early_stop_mse=1.0e-12,
        )
    )

    assert result["teacher_semantic_profile_tv_between_tasks"] == pytest.approx(0.0)
    assert result["teacher_structural_profile_tv_between_tasks"] == pytest.approx(0.0)
    additive = result["tasks"]["additive"]
    gated = result["tasks"]["gated"]
    assert additive["summary"]["max_fit_error"] < 2.0e-5
    assert additive["summary"]["max_abs_output_interaction"] < 5.0e-5
    assert additive["summary"]["estimable_interaction_seeds"] == 0
    assert gated["summary"]["max_fit_error"] < 5.0e-3
    assert gated["summary"]["mean_output_interaction"] == pytest.approx(
        2.0, abs=5.0e-3
    )
    assert gated["summary"]["estimable_interaction_seeds"] == 3
    assert gated["summary"]["mean_far_interaction_share"] == pytest.approx(
        0.875, abs=5.0e-4
    )
    assert gated["summary"]["mean_interaction_expected_distance"] == pytest.approx(
        4.625, abs=5.0e-3
    )
    assert gated["summary"]["max_marginal_profile_tv"] < 2.0e-3
    assert result["learned_profile_match"]["max_semantic_tv_between_tasks"] < 2.0e-3
    assert result["learned_profile_match"]["max_structural_tv_between_tasks"] < 2.0e-3
