import numpy as np
import pytest

from graph_specialisation_metrics.synthetic.structure_gated_routing import (
    STATES,
    ExperimentConfig,
    measure_four_state_field,
    run,
    teacher_contributions,
)


def test_additive_and_gated_teachers_have_identical_marginals_but_different_interaction():
    gamma = 0.75
    additive = measure_four_state_field(
        teacher_contributions(STATES, task="additive", interaction_strength=gamma)
    )
    gated = measure_four_state_field(
        teacher_contributions(STATES, task="gated", interaction_strength=gamma)
    )

    np.testing.assert_allclose(additive["semantic_response"], [2.0, 3.0])
    np.testing.assert_allclose(additive["structural_response"], [2.0, 3.0])
    np.testing.assert_allclose(gated["semantic_response"], [2.0, 3.0])
    np.testing.assert_allclose(gated["structural_response"], [2.0, 3.0])
    np.testing.assert_allclose(additive["semantic_profile"], gated["semantic_profile"])
    assert additive["marginal_profile_tv"] == pytest.approx(0.0)
    assert gated["marginal_profile_tv"] == pytest.approx(0.0)
    np.testing.assert_allclose(additive["interaction"], [0.0, 0.0])
    assert additive["interaction_estimable"] is False
    assert np.isnan(additive["far_interaction_share"])
    np.testing.assert_allclose(gated["interaction"], [0.0, 4.0 * gamma])
    assert gated["interaction_estimable"] is True
    assert gated["far_interaction_share"] == pytest.approx(1.0)
    assert gated["interaction_expected_distance"] == pytest.approx(4.0)


def test_tiny_learned_router_recovers_additive_null_and_gated_far_interaction():
    result = run(
        ExperimentConfig(
            seeds=(0, 1, 2),
            hidden_dim=12,
            max_steps=4_000,
            early_stop_mse=1.0e-12,
        )
    )

    assert result["teacher_marginal_profile_tv_between_tasks"] == pytest.approx(0.0)
    additive = result["tasks"]["additive"]
    gated = result["tasks"]["gated"]
    assert additive["summary"]["max_fit_error"] < 2.0e-5
    assert additive["summary"]["max_abs_output_interaction"] < 5.0e-5
    assert additive["summary"]["estimable_interaction_seeds"] == 0
    assert additive["summary"]["mean_far_interaction_share"] is None
    assert gated["summary"]["max_fit_error"] < 2.0e-5
    assert gated["summary"]["mean_output_interaction"] == pytest.approx(3.0, abs=5.0e-5)
    assert gated["summary"]["estimable_interaction_seeds"] == 3
    assert gated["summary"]["mean_far_interaction_share"] > 0.9999
    assert gated["summary"]["max_marginal_profile_tv"] < 2.0e-5
