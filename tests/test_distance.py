import numpy as np
import pytest

from graph_specialisation_metrics.distance import (
    DistanceResolvedScoreContributions,
    DonorSwapDistanceContributions,
    ReconstructionError,
    aggregate_distance_resolved_score_contributions,
    assert_reconstructs_scores,
    distance_resolved_score_contributions,
    shortest_path_distances,
)
from graph_specialisation_metrics.runner import (
    ChannelScore,
    align_distance_contributions,
    mean_channel_scores,
)


def test_shortest_paths_keep_unreachable_nodes_explicit() -> None:
    distances = shortest_path_distances(np.array([[0, 1], [1, 2]]), num_nodes=4)

    np.testing.assert_allclose(distances[:3, :3], [[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    assert np.isinf(distances[0, 3])


def test_distance_categories_reconstruct_each_donor_swap_and_hierarchy() -> None:
    # q[donor_swap, layer, head, head_output_row, model_output]
    projected = np.array(
        [
            [[[[3.0, 4.0], [0.0, 2.0], [1.0, 0.0]]]],
            [[[[0.0, 1.0], [6.0, 8.0], [2.0, 0.0]]]],
            [[[[2.0, 0.0], [0.0, 3.0], [4.0, 0.0]]]],
        ]
    )
    distances = np.array(
        [
            [0, 1, "virtual-node"],
            [0, 1, "virtual-node"],
            [0, 1, "virtual-node"],
        ],
        dtype=object,
    )

    contributions = distance_resolved_score_contributions(projected, distances)
    aggregate = aggregate_distance_resolved_score_contributions(
        contributions,
        graph_ids=[0, 0, 1],
        source_ids=[0, 0, 0],
    )

    assert contributions.distance_categories == (0, 1, "virtual-node")
    np.testing.assert_allclose(
        contributions.score_contributions.sum(axis=-1),
        contributions.donor_swap_responses,
    )
    np.testing.assert_allclose(
        aggregate.score_contributions.sum(axis=-1),
        0.5 * (10.5 + 9.0),
    )


def test_mismatched_distance_contributions_raise() -> None:
    with pytest.raises(ReconstructionError):
        assert_reconstructs_scores(np.array([[1.0, 2.0]]), np.array([4.0]))

    tampered = DonorSwapDistanceContributions(
        distance_categories=(0, 1),
        score_contributions=np.array([[[[1.0, 1.0]]]]),
        donor_swap_responses=np.array([[[3.0]]]),
    )
    with pytest.raises(ReconstructionError):
        aggregate_distance_resolved_score_contributions(tampered, [0], [0])


def test_float32_distance_contributions_sum_exactly() -> None:
    import torch

    generator = torch.Generator().manual_seed(11)
    projected = torch.randn(3, 2, 4, 29, 5, generator=generator)
    distances = np.broadcast_to(np.arange(29) % 6, (3, 29))

    result = distance_resolved_score_contributions(projected, distances, atol=0.0)

    np.testing.assert_array_equal(
        result.score_contributions.sum(axis=-1),
        result.donor_swap_responses,
    )


def test_channel_scores_share_distance_categories() -> None:
    first = ChannelScore(
        np.array([3.0]),
        np.empty(0),
        {},
        {},
        DistanceResolvedScoreContributions(
            (0, "virtual-node"),
            np.array([[1.0, 2.0]]),
            {},
            {},
        ),
    )
    second = ChannelScore(
        np.array([7.0]),
        np.empty(0),
        {},
        {},
        DistanceResolvedScoreContributions((0, 1), np.array([[4.0, 3.0]]), {}, {}),
    )

    mean = mean_channel_scores((first, second))

    contributions = mean.distance_resolved_score_contributions
    assert contributions is not None
    assert contributions.distance_categories == (0, 1, "virtual-node")
    np.testing.assert_allclose(
        align_distance_contributions(mean, contributions.distance_categories),
        [[2.5, 1.5, 1.0]],
    )
    np.testing.assert_allclose(mean.score, contributions.score_contributions.sum(axis=-1))
