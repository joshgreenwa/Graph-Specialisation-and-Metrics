import numpy as np

from graph_specialisation_metrics.interventions import (
    semantic_donor_swap,
    structural_donor_swap,
)
from graph_specialisation_metrics.sampling import (
    SemanticDonorPool,
    analysis_indices,
    eligible_structural_donors,
    sample_structural_donors,
)


def test_semantic_donor_swap_changes_only_the_source() -> None:
    clean = np.array([[1, 2], [3, 4], [5, 6]])

    donor_swap = semantic_donor_swap(clean, source=1, donor_attributes=[9, 8])

    np.testing.assert_array_equal(donor_swap, [[1, 2], [9, 8], [5, 6]])
    np.testing.assert_array_equal(clean, [[1, 2], [3, 4], [5, 6]])


def test_structural_donor_swap_copies_row_column_self_and_node_fields() -> None:
    node = np.arange(8).reshape(4, 2)
    pair = np.arange(16).reshape(4, 4)

    nodes, pairs = structural_donor_swap(
        {"degree_features": node},
        {"rrwp": pair},
        source=1,
        donor=3,
    )

    np.testing.assert_array_equal(nodes["degree_features"][1], node[3])
    expected = pair.copy()
    expected[1, :] = pair[3, :]
    expected[:, 1] = pair[:, 3]
    expected[1, 1] = pair[3, 3]
    np.testing.assert_array_equal(pairs["rrwp"], expected)
    np.testing.assert_array_equal(pair, np.arange(16).reshape(4, 4))
    assert pairs["rrwp"][0, 2] == pair[0, 2]


def test_structural_donor_swap_preserves_pair_field_torch_type() -> None:
    import torch

    pair = torch.arange(9).reshape(3, 3)
    _, pairs = structural_donor_swap({}, {"rrwp": pair}, source=0, donor=2)
    result = pairs["rrwp"]

    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result[0], torch.tensor([8, 7, 8]))
    torch.testing.assert_close(result[:, 0], torch.tensor([8, 5, 8]))


def test_semantic_sampling_minimises_degree_gap_then_balances_graphs() -> None:
    pool = SemanticDonorPool(
        attributes={
            "source": np.array([[0]]),
            "small": np.array([[1]]),
            "large": np.arange(2, 12).reshape(-1, 1),
            "wrong_degree": np.array([[99]]),
        },
        degrees={
            "source": [2],
            "small": [2],
            "large": [2] * 10,
            "wrong_degree": [3],
        },
    )

    eligible = pool.eligible([0], 2, source_graph="source")
    assert set(eligible) == {"small", "large"}
    draws = pool.sample([0], 2, 4000, np.random.default_rng(7), source_graph="source")
    small_fraction = sum(draw.graph_id == "small" for draw in draws) / len(draws)
    assert 0.47 < small_fraction < 0.53
    assert len({(draw.graph_id, draw.node) for draw in draws}) < len(draws)


def test_structural_sampling_is_unique_and_exhausts_eligible_nodes() -> None:
    structural_profiles = [np.array([0]), np.array([0]), np.array([1]), np.array([2])]
    rng = np.random.default_rng(4)

    np.testing.assert_array_equal(eligible_structural_donors(structural_profiles, 0), [2, 3])
    np.testing.assert_array_equal(
        sample_structural_donors(structural_profiles, 0, 8, rng),
        [2, 3],
    )
    selected = sample_structural_donors(structural_profiles, 0, 1, rng)
    assert len(selected) == len(set(selected.tolist())) == 1


def test_analysis_graphs_and_semantic_pool_use_fixed_independent_seeds() -> None:
    seed = 31_415
    expected_graphs = np.sort(np.random.default_rng(seed).permutation(20)[:5])
    expected_semantic_pool = np.sort(
        np.random.default_rng(seed + 1).choice(30, size=8, replace=False)
    )

    graphs, semantic_pool = analysis_indices(20, 5, 30, 8, seed)

    np.testing.assert_array_equal(graphs, expected_graphs)
    np.testing.assert_array_equal(semantic_pool, expected_semantic_pool)
    assert not np.array_equal(graphs, np.arange(5))
    assert not np.array_equal(semantic_pool, np.arange(8))
    with np.testing.assert_raises_regex(ValueError, "requested 21 scored graphs"):
        analysis_indices(20, 21, 30, 8, seed)
    with np.testing.assert_raises_regex(ValueError, "lengths and sample counts must be positive"):
        analysis_indices(20, 5, 0, 8, seed)
