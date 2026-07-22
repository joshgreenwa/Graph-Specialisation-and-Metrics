import numpy as np
import pytest
import torch

from graph_specialisation_metrics.carriage import figures
from graph_specialisation_metrics.carriage.core import (
    FUNCTIONAL_CARRIAGE_ESTIMAND,
    FUNCTIONAL_CARRIAGE_VERSION,
    aggregate_carriage_curves,
    functional_magnitude_from_delta,
    functional_magnitudes_from_delta,
)


def test_functional_default_is_eventwise_sensitivity_not_coherent_response():
    # One source, two donors, two carriers and one output. Donor responses cancel at
    # carrier 0 but agree at carrier 1.
    delta = torch.tensor(
        [[[1.0], [2.0]], [[-1.0], [2.0]]], dtype=torch.float64
    )
    g_out = torch.ones((1, 2, 1), dtype=torch.float64)

    F_sens, F_coh = functional_magnitudes_from_delta(delta, g_out, 1, 2)

    np.testing.assert_allclose(F_sens, [[1.0], [2.0]])
    np.testing.assert_allclose(F_coh, [[0.0], [2.0]])
    np.testing.assert_allclose(
        functional_magnitude_from_delta(delta, g_out, 1, 2), F_sens
    )
    assert FUNCTIONAL_CARRIAGE_ESTIMAND == "F_sens"
    assert FUNCTIONAL_CARRIAGE_VERSION == 2


def test_functional_sensitivity_takes_multioutput_norm_per_event():
    # q vectors are (3,4) and (-3,-4): each event has magnitude 5, while their
    # coherent mean vanishes.
    delta = torch.tensor([[[3.0, 4.0]], [[-3.0, -4.0]]], dtype=torch.float64)
    g_out = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.float64
    )

    F_sens, F_coh = functional_magnitudes_from_delta(delta, g_out, 1, 2)

    assert F_sens[0, 0] == pytest.approx(5.0)
    assert F_coh[0, 0] == pytest.approx(0.0)


def test_functional_sensitivity_upper_bounds_coherent_response():
    generator = torch.Generator().manual_seed(11)
    delta = torch.randn((3 * 5, 4, 6), generator=generator, dtype=torch.float64)
    g_out = torch.randn((2, 4, 6), generator=generator, dtype=torch.float64)

    F_sens, F_coh = functional_magnitudes_from_delta(delta, g_out, 3, 5)

    assert np.all(F_sens + 1e-12 >= F_coh)


def test_functional_estimators_validate_event_and_gradient_shapes():
    delta = torch.zeros((2, 3, 4))
    g_out = torch.zeros((1, 3, 4))

    with pytest.raises(ValueError, match="expected S\\*K"):
        functional_magnitudes_from_delta(delta, g_out, 2, 2)
    with pytest.raises(ValueError, match="aligned with delta"):
        functional_magnitudes_from_delta(delta, torch.zeros((1, 2, 4)), 1, 2)


def test_distance_aggregation_retains_coherent_diagnostic_beside_default():
    curves = aggregate_carriage_curves(
        graph_id=np.array([0, 0, 1, 1]),
        distance=np.array([0, 1, 0, 1]),
        F=np.array([4.0, 2.0, 6.0, 4.0]),
        B=np.zeros(4),
        F_coh=np.array([2.0, 1.0, 3.0, 2.0]),
        n_boot=20,
        min_count=1,
        central="mean",
    )

    np.testing.assert_allclose(curves["F_mean"], [5.0, 3.0])
    np.testing.assert_allclose(curves["F_coh_mean"], [2.5, 1.5])
    assert curves["F_per_graph"].shape == curves["F_coh_per_graph"].shape == (2, 2)


def test_saved_artifacts_name_fsens_as_default_and_retain_fcoh(tmp_path):
    results = {
        "graph_id": np.array([0, 0, 1, 1]),
        "carrier_i": np.array([0, 0, 0, 0]),
        "source_j": np.array([0, 1, 0, 1]),
        "distance": np.array([0, 1, 0, 1]),
        "C": np.array([0.2, 0.1, 0.3, 0.2]),
        "B": np.array([-0.2, -0.1, -0.3, -0.2]),
        "F": np.array([4.0, 2.0, 6.0, 4.0]),
        "F_coh": np.array([2.0, 1.0, 3.0, 2.0]),
        "additivity_sumC": np.array([0.3, 0.5]),
        "additivity_dyhat": np.array([0.3, 0.5]),
        "checks": {"additivity_pearson_r": 1.0, "additivity_slope": 1.0},
        "meta": {
            "task": "toy",
            "title": "Toy",
            "checkpoint_epoch": 1,
            "eval_split": "test",
            "donor_split": "test",
            "donors_K": 2,
            "beneficial_denom": "integrated",
            "loss_units": "loss units",
            "functional_estimand": "F_sens",
            "functional_carriage_version": 2,
        },
    }

    out = figures.make_figures_and_save(
        results, str(tmp_path), n_boot=20, min_count=1, central="mean"
    )

    with np.load(out["npz"]) as saved:
        assert str(saved["functional_estimand"].item()) == "F_sens"
        assert int(saved["functional_carriage_version"].item()) == 2
        np.testing.assert_array_equal(saved["F"], saved["F_sens"])
        np.testing.assert_array_equal(saved["F_coh"], results["F_coh"])
        assert "F_coh_mean" in saved
    assert len(out["figures"]) == 3
