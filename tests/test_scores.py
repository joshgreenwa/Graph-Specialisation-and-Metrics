import numpy as np

from graph_specialisation_metrics.runner import clean_output_gradients
from graph_specialisation_metrics.scores import (
    aggregate_donor_swap_responses,
    donor_swap_responses,
    projected_head_responses,
    specialisation_measures,
)


def test_clean_output_gradients_scale_each_model_output() -> None:
    import torch

    first = torch.tensor([[1.0, 2.0]], requires_grad=True)
    second = torch.tensor([[3.0, 4.0]], requires_grad=True)
    outputs = torch.stack(((first + 2 * second).sum(), (4 * first + 6 * second).sum()))

    gradients = clean_output_gradients(
        outputs,
        (first, second),
        torch.tensor([1.0, 2.0]),
    )

    expected = torch.tensor(
        [
            [[[1.0, 1.0]], [[2.0, 2.0]]],
            [[[2.0, 2.0]], [[3.0, 3.0]]],
        ]
    )
    torch.testing.assert_close(gradients, expected)


def test_projected_response_and_head_output_row_magnitudes() -> None:
    delta = np.array(
        [
            [
                [
                    [[1.0, 2.0]],
                    [[3.0, 4.0]],
                ]
            ]
        ]
    )  # [donor_swap=1, layer=1, head_output_row=2, head=1, width=2]
    gradients = np.array(
        [
            [[[[2.0, 0.0]], [[0.0, 1.0]]]],
            [[[[0.0, 1.0]], [[1.0, 0.0]]]],
        ]
    )  # [model_output=2, layer=1, head_output_row=2, head=1, width=2]

    projected = projected_head_responses(delta, gradients)

    np.testing.assert_allclose(projected[0, 0, 0], [[2.0, 2.0], [4.0, 3.0]])
    np.testing.assert_allclose(donor_swap_responses(projected), [[[np.sqrt(8.0) + 5.0]]])


def test_donor_magnitude_is_taken_before_averaging() -> None:
    projected = np.array(
        [
            [[[[1.0]]]],
            [[[[-1.0]]]],
        ]
    )  # two opposing donor-swaps

    scores = donor_swap_responses(projected)

    np.testing.assert_allclose(scores, np.ones((2, 1, 1)))
    np.testing.assert_allclose(scores.mean(axis=0), [[1.0]])
    assert np.linalg.norm(projected.mean(axis=0)) == 0.0


def test_ragged_donor_swaps_follow_donor_source_graph_hierarchy() -> None:
    responses = np.array([2.0, 4.0, 9.0, 1.0, 1.0, 1.0])
    graph_ids = [0, 0, 0, 1, 1, 1]
    source_ids = [0, 0, 1, 0, 0, 0]

    score, graphs, sources = aggregate_donor_swap_responses(
        responses,
        graph_ids,
        source_ids,
    )

    assert sources[(0, 0)] == 3.0
    assert sources[(0, 1)] == 9.0
    assert graphs[0] == 6.0
    assert graphs[1] == 1.0
    assert score == 3.5
    assert score != responses.mean()


def test_specialisation_measures_use_separate_channel_means() -> None:
    measures = specialisation_measures(
        semantic_scores=np.array([[1.0, 3.0]]),
        structural_scores=np.array([[4.0, 4.0]]),
    )

    np.testing.assert_allclose(measures.normalised_semantic_scores, [[0.5, 1.5]])
    np.testing.assert_allclose(measures.normalised_structural_scores, [[1.0, 1.0]])
    np.testing.assert_allclose(measures.joint_sensitivity, [[0.75, 1.25]])
    np.testing.assert_allclose(measures.selectivity, [[-1.0 / 3.0, 0.2]])
    assert measures.joint_sensitivity.mean() == 1.0


def test_zero_channel_cannot_be_normalised() -> None:
    with np.testing.assert_raises_regex(ValueError, "positive mean"):
        specialisation_measures(np.zeros((1, 2)), np.ones((1, 2)))
