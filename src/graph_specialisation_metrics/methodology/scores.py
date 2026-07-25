"""Pure estimators for canonical raw head scores and derived coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SEMANTIC_AXIS_LABEL = r"Semantic score  $S_{sem}/\overline{S}_{sem}$"
STRUCTURAL_AXIS_LABEL = r"Structural score  $S_{str}/\overline{S}_{str}$"
SELECTIVITY_AXIS_LABEL = (
    r"Selectivity $D_{rel}$  (structural $\leftarrow$ 0 $\rightarrow$ semantic)"
)
JOINT_AXIS_LABEL = r"Joint sensitivity $J$"


def project_transport(delta, clean_gradient):
    """Project clean-minus-event transport through the clean z-space Jacobian.

    Args:
        delta: ``[event, layer, carrier, head, width]``.
        clean_gradient: ``[output, layer, carrier, head, width]``.

    Returns:
        ``q[event, layer, head, carrier, output]``.
    """

    import torch

    if delta.ndim != 5 or clean_gradient.ndim != 5:
        raise ValueError("delta and clean_gradient must both be rank five")
    if tuple(delta.shape[1:]) != tuple(clean_gradient.shape[1:]):
        raise ValueError(
            f"transport/gradient geometry differs: {tuple(delta.shape)} vs "
            f"{tuple(clean_gradient.shape)}"
        )
    if not torch.isfinite(delta).all() or not torch.isfinite(clean_gradient).all():
        raise ValueError("transport projection inputs must be finite")
    return torch.einsum("elnhd,tlnhd->elhnt", delta, clean_gradient)


def event_head_scores(q):
    """Take output L2 per event/carrier, then sum carriers."""

    return q.square().sum(dim=-1).sqrt().sum(dim=-1)


def aggregate_event_scores(
    scores: np.ndarray,
    graph_ids: Sequence[int],
    source_ids: Sequence[int],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[tuple[int, int], np.ndarray]]:
    """Average donors within source, sources within graph, then graphs equally."""

    values = np.asarray(scores, dtype=np.float64)
    graph_ids = np.asarray(graph_ids, dtype=np.int64)
    source_ids = np.asarray(source_ids, dtype=np.int64)
    if values.shape[0] != len(graph_ids) or len(graph_ids) != len(source_ids):
        raise ValueError("score rows must align with graph/source IDs")
    sources: dict[tuple[int, int], np.ndarray] = {}
    for graph in np.unique(graph_ids):
        selected_graph = graph_ids == graph
        for source in np.unique(source_ids[selected_graph]):
            mask = selected_graph & (source_ids == source)
            sources[(int(graph), int(source))] = values[mask].mean(axis=0)
    graphs: dict[int, np.ndarray] = {}
    for graph in np.unique(graph_ids):
        rows = [value for (g, _), value in sources.items() if g == int(graph)]
        graphs[int(graph)] = np.stack(rows).mean(axis=0)
    total = np.stack([graphs[key] for key in sorted(graphs)]).mean(axis=0)
    return total, graphs, sources


@dataclass(frozen=True)
class HeadCoordinates:
    raw_semantic: np.ndarray
    raw_structural: np.ndarray
    semantic_mean: float
    structural_mean: float
    normalized_semantic: np.ndarray
    normalized_structural: np.ndarray
    joint_sensitivity: np.ndarray
    selectivity: np.ndarray
    active: np.ndarray
    estimable: bool


def head_coordinates(
    semantic: Any,
    structural: Any,
    *,
    score_floor: float,
    epsilon: float,
    activity_floor: float,
) -> HeadCoordinates:
    """Normalize within trained model and compute J/D_rel exactly once."""

    semantic = np.asarray(semantic, dtype=np.float64)
    structural = np.asarray(structural, dtype=np.float64)
    if semantic.shape != structural.shape or semantic.ndim != 2:
        raise ValueError("raw semantic and structural scores must share [layer,head] shape")
    if np.any(semantic < 0) or np.any(structural < 0):
        raise ValueError("raw scores must be non-negative")
    semantic_mean = float(np.mean(semantic))
    structural_mean = float(np.mean(structural))
    estimable = bool(
        np.isfinite(semantic_mean)
        and np.isfinite(structural_mean)
        and semantic_mean > float(score_floor)
        and structural_mean > float(score_floor)
    )
    shape = semantic.shape
    if not estimable:
        missing = np.full(shape, np.nan)
        return HeadCoordinates(
            semantic,
            structural,
            semantic_mean,
            structural_mean,
            missing.copy(),
            missing.copy(),
            missing.copy(),
            missing.copy(),
            np.zeros(shape, dtype=bool),
            False,
        )
    normalized_semantic = semantic / semantic_mean
    normalized_structural = structural / structural_mean
    joint = 0.5 * (normalized_semantic + normalized_structural)
    selectivity = (normalized_semantic - normalized_structural) / (
        normalized_semantic + normalized_structural + float(epsilon)
    )
    active = joint >= float(activity_floor)
    return HeadCoordinates(
        semantic,
        structural,
        semantic_mean,
        structural_mean,
        normalized_semantic,
        normalized_structural,
        joint,
        selectivity,
        active,
        True,
    )


def freeze_families(
    coordinates: HeadCoordinates,
    *,
    tail_fraction: float,
    central_fraction: float,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Freeze rank-based discovery families without inspecting causal outcomes."""

    J = coordinates.joint_sensitivity
    D = coordinates.selectivity
    heads = [(l, h) for l in range(J.shape[0]) for h in range(J.shape[1])]
    active = [(l, h) for l, h in heads if coordinates.active[l, h]]
    inactive = [(l, h) for l, h in heads if not coordinates.active[l, h]]
    count = max(1, int(np.floor(float(tail_fraction) * len(active)))) if active else 0
    ordered_d = sorted(active, key=lambda item: (D[item], item))
    structural = ordered_d[:count]
    semantic = list(reversed(ordered_d[-count:])) if count else []
    central_count = (
        max(1, int(np.floor(float(central_fraction) * len(active)))) if active else 0
    )
    median = float(np.median([D[item] for item in active])) if active else np.nan
    central = sorted(
        active, key=lambda item: (abs(D[item] - median), -J[item], item)
    )[:central_count]
    return {
        "semantic_leaning": tuple(semantic),
        "structural_leaning": tuple(structural),
        "central_responsive": tuple(central),
        "inactive": tuple(sorted(inactive, key=lambda item: (J[item], item))[:central_count]),
    }


def freeze_matched_controls(
    coordinates: HeadCoordinates,
    families: Mapping[str, Sequence[tuple[int, int]]],
    throughput: Any,
    *,
    rng_seed: int,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Freeze same-layer size/J/throughput controls for each leaning family."""

    throughput = np.asarray(throughput, dtype=np.float64)
    J = coordinates.joint_sensitivity
    D = coordinates.selectivity
    all_heads = [(l, h) for l in range(J.shape[0]) for h in range(J.shape[1])]
    leaning = set(families.get("semantic_leaning", ())) | set(
        families.get("structural_leaning", ())
    )
    rng = np.random.default_rng(int(rng_seed))
    controls: dict[str, tuple[tuple[int, int], ...]] = {}
    median_d = float(np.nanmedian(D[coordinates.active]))
    for target_name in ("semantic_leaning", "structural_leaning"):
        target = tuple(families.get(target_name, ()))
        if not target:
            continue
        selected: dict[str, list[tuple[int, int]]] = {
            "central": [],
            "inactive": [],
            "random": [],
        }
        used: dict[str, set[tuple[int, int]]] = {key: set(leaning) for key in selected}
        for layer, head in target:
            same_layer = [item for item in all_heads if item[0] == layer]
            active_central = [
                item
                for item in same_layer
                if coordinates.active[item] and item not in used["central"]
            ]
            low_j = [item for item in same_layer if item not in used["inactive"]]
            if active_central:
                scale_j = max(float(np.nanstd(J[layer])), 1e-12)
                scale_t = max(float(np.nanstd(throughput[layer])), 1e-12)
                choice = min(
                    active_central,
                    key=lambda item: (
                        abs(D[item] - median_d)
                        + abs(J[item] - J[layer, head]) / scale_j
                        + abs(throughput[item] - throughput[layer, head]) / scale_t,
                        item,
                    ),
                )
                selected["central"].append(choice)
                used["central"].add(choice)
            if low_j:
                choice = min(
                    low_j,
                    key=lambda item: (
                        J[item],
                        abs(throughput[item] - throughput[layer, head]),
                        item,
                    ),
                )
                selected["inactive"].append(choice)
                used["inactive"].add(choice)
            random_pool = [item for item in same_layer if item not in used["random"]]
            if random_pool:
                choice = random_pool[int(rng.integers(0, len(random_pool)))]
                selected["random"].append(choice)
                used["random"].add(choice)
        for kind, values in selected.items():
            if len(values) == len(target):
                controls[f"{target_name}_{kind}_control"] = tuple(values)
    return controls
