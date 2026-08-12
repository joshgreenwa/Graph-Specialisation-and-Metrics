"""Semantic and structural specialisation score calculations."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch")


def _finite(value: Any) -> bool:
    if _is_torch_tensor(value):
        import torch

        return bool(torch.isfinite(value).all())
    return bool(np.isfinite(np.asarray(value)).all())


def projected_head_responses(
    clean_minus_donor_swap_head_outputs: Any,
    clean_output_gradients: Any,
) -> Any:
    """Compute head response ``q`` from donor-swap head-output changes.

    Inputs start with ``donor_swap`` and ``model_output``; the output ends with
    ``[head_output_row, model_output]``. Gradients use the clean forward pass (Eq. 3.13).
    """

    delta = clean_minus_donor_swap_head_outputs
    gradients = clean_output_gradients
    if getattr(delta, "ndim", None) != 5 or getattr(gradients, "ndim", None) != 5:
        raise ValueError("head-output changes and clean output gradients must be rank five")
    if tuple(delta.shape[1:]) != tuple(gradients.shape[1:]):
        raise ValueError(
            "head-output changes and clean output gradients disagree after their "
            "donor-swap/model-output axes: "
            f"{tuple(delta.shape)} versus {tuple(gradients.shape)}"
        )
    if _is_torch_tensor(delta) != _is_torch_tensor(gradients):
        raise TypeError("head-output changes and gradients must use the same array type")
    if not _finite(delta) or not _finite(gradients):
        raise ValueError("head-output changes and clean output gradients must be finite")

    if _is_torch_tensor(delta):
        import torch

        return torch.einsum("klrhw,olrhw->klhro", delta, gradients)
    return np.einsum("klrhw,olrhw->klhro", np.asarray(delta), np.asarray(gradients))


def head_output_row_magnitudes(projected_responses: Any) -> Any:
    """Take the model-output L2 norm at each head-output row."""

    if getattr(projected_responses, "ndim", 0) < 2:
        raise ValueError("projected responses must end in [head_output_row, model_output]")
    if not _finite(projected_responses):
        raise ValueError("projected responses must be finite")
    if _is_torch_tensor(projected_responses):
        import torch

        return torch.linalg.vector_norm(projected_responses, ord=2, dim=-1)
    return np.linalg.norm(np.asarray(projected_responses), ord=2, axis=-1)


def donor_swap_responses(projected_responses: Any) -> Any:
    """Compute donor-swap response ``R`` by summing row-wise response magnitudes."""

    magnitudes = head_output_row_magnitudes(projected_responses)
    if _is_torch_tensor(magnitudes):
        return magnitudes.sum(dim=-1)
    return magnitudes.sum(axis=-1)


def _ordered_unique(values: np.ndarray) -> list[Hashable]:
    result: list[Hashable] = []
    for value in values.tolist():
        if value not in result:
            result.append(value)
    return result


def aggregate_donor_swap_responses(
    responses: Any,
    graph_ids: Sequence[Hashable],
    source_ids: Sequence[Hashable],
) -> tuple[
    np.ndarray,
    dict[Hashable, np.ndarray],
    dict[tuple[Hashable, Hashable], np.ndarray],
]:
    """Average donor-swaps within source, sources within graph, then graphs."""

    values = np.asarray(responses, dtype=np.float64)
    graphs_array = np.asarray(graph_ids, dtype=object)
    sources_array = np.asarray(source_ids, dtype=object)
    if values.ndim < 1 or values.shape[0] == 0:
        raise ValueError("at least one donor-swap response is required")
    if values.shape[0] != len(graphs_array) or len(graphs_array) != len(sources_array):
        raise ValueError("response rows must align with graph_ids and source_ids")
    if not np.isfinite(values).all():
        raise ValueError("donor-swap responses must be finite")

    source_means: dict[tuple[Hashable, Hashable], np.ndarray] = {}
    graph_order = _ordered_unique(graphs_array)
    for graph in graph_order:
        graph_mask = graphs_array == graph
        for source in _ordered_unique(sources_array[graph_mask]):
            mask = graph_mask & (sources_array == source)
            source_means[(graph, source)] = values[mask].mean(axis=0)

    graph_means: dict[Hashable, np.ndarray] = {}
    for graph in graph_order:
        rows = [value for (row_graph, _), value in source_means.items() if row_graph == graph]
        graph_means[graph] = np.stack(rows, axis=0).mean(axis=0)

    score = np.stack([graph_means[graph] for graph in graph_order], axis=0).mean(axis=0)
    return score, graph_means, source_means


@dataclass(frozen=True)
class SpecialisationMeasures:
    """Specialisation scores, joint sensitivity, and selectivity."""

    semantic_scores: np.ndarray
    structural_scores: np.ndarray
    normalised_semantic_scores: np.ndarray
    normalised_structural_scores: np.ndarray
    joint_sensitivity: np.ndarray
    selectivity: np.ndarray


def specialisation_measures(
    semantic_scores: Any,
    structural_scores: Any,
    *,
    epsilon: float = 1e-12,
) -> SpecialisationMeasures:
    """Normalise each channel by its head mean and compute ``J`` and ``D_rel``."""

    semantic_array = np.asarray(semantic_scores, dtype=np.float64)
    structural_array = np.asarray(structural_scores, dtype=np.float64)
    if semantic_array.shape != structural_array.shape or semantic_array.size == 0:
        raise ValueError("semantic and structural scores must have the same non-empty shape")
    if not np.isfinite(semantic_array).all() or not np.isfinite(structural_array).all():
        raise ValueError("specialisation scores must be finite")
    if np.any(semantic_array < 0) or np.any(structural_array < 0):
        raise ValueError("specialisation scores must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    semantic_mean = float(semantic_array.mean())
    structural_mean = float(structural_array.mean())
    if semantic_mean == 0 or structural_mean == 0:
        raise ValueError("each channel must have a positive mean")

    normalised_semantic = semantic_array / semantic_mean
    normalised_structural = structural_array / structural_mean
    joint_sensitivity = 0.5 * (normalised_semantic + normalised_structural)
    selectivity = (normalised_semantic - normalised_structural) / (
        normalised_semantic + normalised_structural + epsilon
    )
    return SpecialisationMeasures(
        semantic_array,
        structural_array,
        normalised_semantic,
        normalised_structural,
        joint_sensitivity,
        selectivity,
    )
