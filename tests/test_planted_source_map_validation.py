import numpy as np

from graph_specialisation_metrics.planted_source_map_validation import (
    ValidationConfig,
    auroc,
    distance_matrix,
    plant_teacher,
    run_validation,
)


def test_auroc_constant_scores_are_chance():
    labels = np.array([True, False, True, False])
    scores = np.ones(4)
    assert auroc(labels, scores) == 0.5


def test_planted_graph_is_connected_with_finite_far_nodes(tmp_path):
    config = ValidationConfig(n_nodes=24, feature_dim=4, geometric_radius=0.42, output_root=tmp_path)
    teacher = plant_teacher(config)
    distances = distance_matrix(teacher.graph)
    assert np.isfinite(distances).all()
    far_counts = ((distances > config.local_radius) & np.isfinite(distances)).sum(axis=1)
    assert np.all(far_counts > 0)


def test_swap_source_map_recovers_functional_near_support(tmp_path):
    config = ValidationConfig(
        n_nodes=24,
        feature_dim=4,
        geometric_radius=0.42,
        backgrounds=3,
        partners_per_source=12,
        variants="tanh:0",
        output_root=tmp_path,
    )
    rows, _ = run_validation(config)
    row = rows[0]
    assert row["topology_only_near_auroc"] == 0.5
    assert row["used_vs_unused_near_auroc"] > 0.85
    assert row["planted_far_mrr"] > 0.8
