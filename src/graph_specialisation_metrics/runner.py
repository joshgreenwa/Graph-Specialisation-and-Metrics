"""Specialisation score computation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .distance import (
    DistanceResolvedScoreContributions,
    aggregate_distance_resolved_score_contributions,
    assert_reconstructs_scores,
    distance_resolved_score_contributions,
)
from .scores import (
    aggregate_donor_swap_responses,
    donor_swap_responses,
    projected_head_responses,
)


@dataclass(frozen=True)
class ChannelScore:
    """A channel score and the quantities from which it was aggregated."""

    score: np.ndarray
    donor_swap_responses: np.ndarray
    graph_scores: dict[Any, np.ndarray]
    source_scores: dict[tuple[Any, Any], np.ndarray]
    distance_resolved_score_contributions: DistanceResolvedScoreContributions | None = None


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def clean_output_gradients(
    model_outputs: Any,
    clean_head_outputs: Sequence[Any],
    scale: Any = 1.0,
) -> Any:
    """Differentiate scaled clean model outputs with respect to head outputs."""

    import torch

    clean_head_outputs = tuple(clean_head_outputs)
    if not clean_head_outputs:
        raise ValueError("at least one head-output tensor is required")
    flat = model_outputs.reshape(-1)
    divisor = torch.as_tensor(scale, device=flat.device, dtype=flat.dtype).reshape(-1)
    if divisor.numel() == 1:
        divisor = divisor.expand_as(flat)
    if divisor.shape != flat.shape or bool((divisor <= 0).any()):
        raise ValueError("output scale must be positive and match the flattened outputs")
    rows = []
    for index, output in enumerate(flat / divisor):
        gradients = torch.autograd.grad(
            output,
            clean_head_outputs,
            retain_graph=index + 1 < flat.numel(),
            allow_unused=False,
        )
        rows.append(torch.stack(gradients, dim=0))
    return torch.stack(rows, dim=0)


def compute_channel_score(
    clean_head_outputs: Any,
    donor_swap_head_outputs: Any,
    clean_output_gradients: Any,
    graph_ids: Sequence[Any],
    source_ids: Sequence[Any],
    *,
    head_output_row_distances: Any | None = None,
) -> ChannelScore:
    """Compute one channel's specialisation score.

    Head outputs are ``[donor_swap, layer, head_output_row, head, width]``. A
    single clean output ``[layer, head_output_row, head, width]`` is broadcast over
    donor-swaps.
    """

    clean = clean_head_outputs
    donor_swaps = donor_swap_head_outputs
    if getattr(clean, "ndim", None) == 4:
        clean = clean[None, ...]
    if getattr(clean, "shape", (0,))[0] == 1 and donor_swaps.shape[0] != 1:
        if hasattr(clean, "expand"):
            clean = clean.expand(donor_swaps.shape[0], *clean.shape[1:])
        else:
            clean = np.broadcast_to(clean, donor_swaps.shape)
    if tuple(clean.shape) != tuple(donor_swaps.shape):
        raise ValueError("clean and donor-swap head outputs must have matching shapes")

    projected_responses = projected_head_responses(
        clean - donor_swaps,
        clean_output_gradients,
    )
    distance_contributions = None
    if head_output_row_distances is not None:
        donor_swap_contributions = distance_resolved_score_contributions(
            projected_responses,
            head_output_row_distances,
        )
        responses = donor_swap_contributions.donor_swap_responses
    else:
        responses = _numpy(donor_swap_responses(projected_responses)).astype(
            np.float64,
            copy=False,
        )
    score, graph_scores, source_scores = aggregate_donor_swap_responses(
        responses,
        graph_ids,
        source_ids,
    )
    if head_output_row_distances is not None:
        distance_contributions = aggregate_distance_resolved_score_contributions(
            donor_swap_contributions,
            graph_ids,
            source_ids,
        )
    return ChannelScore(
        score,
        responses,
        graph_scores,
        source_scores,
        distance_contributions,
    )


def distance_categories(scores: Sequence[ChannelScore]) -> tuple[int | str, ...]:
    """Return the combined distance categories for several channel scores."""

    categories: list[int | str] = []
    for score in scores:
        if score.distance_resolved_score_contributions is None:
            raise ValueError("every channel score needs distance-resolved score contributions")
        for category in score.distance_resolved_score_contributions.distance_categories:
            if category not in categories:
                categories.append(category)
    numeric = sorted(category for category in categories if isinstance(category, int))
    return tuple(numeric + [category for category in categories if not isinstance(category, int)])


def align_distance_contributions(
    channel_score: ChannelScore,
    distance_categories: Sequence[int | str],
) -> np.ndarray:
    """Align a channel's score contributions to a distance-category axis."""

    contributions = channel_score.distance_resolved_score_contributions
    if contributions is None:
        raise ValueError("the channel score has no distance-resolved score contributions")
    axis = tuple(distance_categories)
    if len(set(axis)) != len(axis):
        raise ValueError("distance categories must be unique")
    destination = {category: index for index, category in enumerate(axis)}
    if missing := set(contributions.distance_categories) - set(destination):
        raise ValueError(f"distance_categories is missing {sorted(missing, key=str)}")
    aligned = np.zeros(
        contributions.score_contributions.shape[:-1] + (len(axis),),
        dtype=np.float64,
    )
    for old_index, category in enumerate(contributions.distance_categories):
        aligned[..., destination[category]] = contributions.score_contributions[..., old_index]
    return aligned


def mean_channel_scores(scores: Sequence[ChannelScore]) -> ChannelScore:
    """Average graph-level scores after aligning their distance categories."""

    if not scores:
        raise ValueError("at least one channel score is required")
    score = np.stack([channel.score for channel in scores]).mean(axis=0)
    has_distance = [channel.distance_resolved_score_contributions is not None for channel in scores]
    if any(has_distance) and not all(has_distance):
        raise ValueError("distance-resolved score contributions are missing from some scores")
    distance_contributions = None
    if all(has_distance):
        categories = distance_categories(scores)
        score_contributions = np.stack(
            [align_distance_contributions(channel, categories) for channel in scores]
        ).mean(axis=0)
        assert_reconstructs_scores(score_contributions, score)
        distance_contributions = DistanceResolvedScoreContributions(
            categories,
            score_contributions,
            {},
            {},
        )
    return ChannelScore(score, np.empty((0,)), {}, {}, distance_contributions)


def training_target_scale(targets: Any) -> np.ndarray:
    """Return per-output population standard deviation from the training split."""

    values = _numpy(targets).astype(np.float64, copy=False)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("training targets must have shape [example, output]")
    scale = values.std(axis=0, ddof=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("every regression output must have a finite, positive scale")
    return scale
