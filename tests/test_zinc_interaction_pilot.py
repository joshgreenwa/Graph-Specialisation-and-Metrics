import numpy as np
import pytest

from graph_specialisation_metrics.zinc_interaction_pilot import (
    event_distance_summary,
    four_state_contrast,
    graph_distance_profiles,
    layer_event_summary,
)


def test_four_state_contrast_separates_additive_and_interacting_fields():
    clean = np.asarray([1.0, 2.0])
    semantic = np.asarray([2.0, 4.0])
    structural = np.asarray([3.0, 5.0])
    additive_joint = semantic + structural - clean
    np.testing.assert_allclose(
        four_state_contrast(clean, semantic, structural, additive_joint),
        np.zeros(2),
    )
    np.testing.assert_allclose(
        four_state_contrast(clean, semantic, structural, additive_joint + [0.0, 3.0]),
        [0.0, 3.0],
    )


def test_distance_summary_does_not_normalise_a_null_interaction():
    null = event_distance_summary(
        np.asarray([1.0, 2.0]),
        np.asarray([1.5, 1.5]),
        np.asarray([1.0e-7, 2.0e-7]),
        np.asarray([1.0, 5.0]),
        absolute_floor=1.0e-6,
        relative_floor=1.0e-3,
        far_distance=4,
    )
    assert null["interaction_estimable"] is False
    assert np.isnan(null["interaction_expected_distance"])
    effect = event_distance_summary(
        np.asarray([1.0, 2.0]),
        np.asarray([1.5, 1.5]),
        np.asarray([0.0, 1.0]),
        np.asarray([1.0, 5.0]),
        absolute_floor=1.0e-6,
        relative_floor=1.0e-3,
        far_distance=4,
    )
    assert effect["interaction_estimable"] is True
    assert effect["interaction_expected_distance"] == pytest.approx(5.0)
    assert effect["interaction_far_share"] == pytest.approx(1.0)


def test_layer_summary_keeps_virtual_carriage_out_of_physical_distance():
    summary = layer_event_summary(
        np.asarray([1.0, 1.0, 2.0]),
        np.asarray([2.0, 0.0, 2.0]),
        np.asarray([0.0, 1.0, 3.0]),
        np.asarray([0.0, 4.0]),
        absolute_floor=0.1,
        relative_floor=0.01,
        far_distance=4,
        virtual_index=2,
    )
    assert summary["interaction_estimable"] is True
    assert summary["interaction_distance_estimable"] is True
    assert summary["semantic_expected_distance"] == pytest.approx(2.0)
    assert summary["interaction_expected_distance"] == pytest.approx(4.0)
    assert summary["interaction_far_share"] == pytest.approx(0.25)
    assert summary["interaction_virtual_share"] == pytest.approx(0.75)


def test_graph_profiles_average_pairs_then_sources_and_skip_null_interactions():
    rows = []
    for source, pair, masses, eligible in (
        (0, 0, (1.0, 0.0), True),
        (0, 1, (0.0, 1.0), False),
        (1, 0, (1.0, 0.0), True),
    ):
        for distance, semantic_mass in enumerate(masses):
            rows.append(
                {
                    "task": "zinc_1hop",
                    "graph": 3,
                    "source": source,
                    "pair": pair,
                    "distance": distance,
                    "semantic_mass": semantic_mass,
                    "structural_mass": semantic_mass,
                    "interaction_mass": semantic_mass,
                    "interaction_estimable": eligible,
                }
            )
    profiles = graph_distance_profiles(rows, layerwise=False)
    semantic = {int(row["distance"]): row["share"] for row in profiles if row["term"] == "semantic"}
    interaction = {
        int(row["distance"]): row["share"] for row in profiles if row["term"] == "interaction"
    }
    assert semantic == pytest.approx({0: 0.75, 1: 0.25})
    assert interaction == pytest.approx({0: 1.0, 1: 0.0})
