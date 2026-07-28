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


def event_head_score_systems(q, *, mass_floor: float = 0.0) -> dict[str, Any]:
    """Return transport mass, coherent movement, and event-level carrier coherence.

    Args:
        q: ``[event, layer, head, carrier, output]`` projected transport.
        mass_floor: events at or below this mass are non-estimable for the ratio.
    """

    import torch

    if q.ndim != 5:
        raise ValueError("projected transport must be [event,layer,head,carrier,output]")
    if not torch.isfinite(q).all():
        raise ValueError("projected transport must be finite")
    mass = q.square().sum(dim=-1).sqrt().sum(dim=-1)
    coherent_vector = q.sum(dim=-2)
    coherent = coherent_vector.square().sum(dim=-1).sqrt()
    tolerance = 32.0 * torch.finfo(q.dtype).eps * torch.maximum(
        mass, torch.ones_like(mass)
    )
    if bool((coherent > mass + tolerance).any()):
        raise RuntimeError("coherent movement exceeds transport mass beyond numerical tolerance")
    coherence = torch.full_like(mass, float("nan"))
    estimable = mass > float(mass_floor)
    coherence[estimable] = coherent[estimable] / mass[estimable]
    return {
        "mass": mass,
        "coherent": coherent,
        "coherent_vector": coherent_vector,
        "carrier_coherence": coherence,
        "estimable": estimable,
    }


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
    central_pool_fraction: float = 0.50,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Freeze rank-based discovery families without inspecting causal outcomes."""

    return _freeze_family_arrays(
        coordinates.joint_sensitivity,
        coordinates.selectivity,
        coordinates.active,
        tail_fraction=tail_fraction,
        central_fraction=central_fraction,
        central_pool_fraction=central_pool_fraction,
    )


def _freeze_family_arrays(
    joint_sensitivity: Any,
    selectivity: Any,
    active_mask: Any,
    *,
    tail_fraction: float,
    central_fraction: float,
    central_pool_fraction: float = 0.50,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Shared deterministic family rule for the point estimate and every bootstrap draw."""

    J = np.asarray(joint_sensitivity, dtype=np.float64)
    D = np.asarray(selectivity, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=bool)
    if J.shape != D.shape or J.shape != active_mask.shape or J.ndim != 2:
        raise ValueError("J, D_rel, and the activity mask must share [layer,head] shape")
    heads = [(l, h) for l in range(J.shape[0]) for h in range(J.shape[1])]
    active = [
        (l, h)
        for l, h in heads
        if active_mask[l, h] and np.isfinite(J[l, h]) and np.isfinite(D[l, h])
    ]
    active_set = set(active)
    inactive = [(l, h) for l, h in heads if (l, h) not in active_set]
    count = max(1, int(np.floor(float(tail_fraction) * len(active)))) if active else 0
    ordered_d = sorted(active, key=lambda item: (D[item], item))
    structural = ordered_d[:count]
    semantic = list(reversed(ordered_d[-count:])) if count else []
    central_count = (
        max(1, int(np.floor(float(central_fraction) * len(active)))) if active else 0
    )
    median = float(np.median([D[item] for item in active])) if active else np.nan
    pool_count = (
        max(
            central_count,
            int(np.ceil(float(central_pool_fraction) * len(active))),
        )
        if active
        else 0
    )
    central_pool = sorted(
        active, key=lambda item: (abs(D[item] - median), item)
    )[:pool_count]
    central = sorted(
        central_pool, key=lambda item: (-J[item], abs(D[item] - median), item)
    )[:central_count]
    return {
        "semantic_leaning": tuple(semantic),
        "structural_leaning": tuple(structural),
        "central_responsive": tuple(central),
        "inactive": tuple(sorted(inactive, key=lambda item: (J[item], item))[:central_count]),
    }


def specialisation_diagnostics(
    coordinates: HeadCoordinates,
    families: Mapping[str, Sequence[tuple[int, int]]],
    bootstrap_draws: Any,
    *,
    selectivity_interval: tuple[Any, Any],
    activity_floor: float,
    tail_fraction: float,
    central_fraction: float,
    central_pool_fraction: float = 0.50,
    equivalence_half_width: float,
    membership_stability_floor: float,
    generalist_fraction_floor: float,
) -> dict[str, Any]:
    """Describe whether discovery resolves channel-selective families or a generalist regime.

    ``bootstrap_draws`` is the transient output of the same paired nested bootstrap used for the
    coordinate intervals.  Family assignment is rerun in every draw.  Nothing in this diagnostic
    inspects a causal endpoint, so it cannot tune the later confirmation test.
    """

    draws = np.asarray(bootstrap_draws, dtype=np.float64)
    shape = coordinates.joint_sensitivity.shape
    if draws.ndim != 4 or draws.shape[1] < 6 or tuple(draws.shape[2:]) != tuple(shape):
        raise ValueError(
            "coordinate bootstrap draws must be [replicate,coordinate,layer,head]"
        )
    point_active = np.asarray(coordinates.active, dtype=bool)
    D = np.asarray(coordinates.selectivity, dtype=np.float64)
    low, high = (np.asarray(value, dtype=np.float64) for value in selectivity_interval)
    if low.shape != shape or high.shape != shape:
        raise ValueError("selectivity intervals must match the head geometry")
    margin = float(equivalence_half_width)

    def finite_quantile(values: Any, quantile: float) -> float:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        return float(np.quantile(finite, quantile)) if finite.size else np.nan

    def finite_mean(values: Any) -> float:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        return float(np.mean(finite)) if finite.size else np.nan

    equivalent = point_active & (low >= -margin) & (high <= margin)
    semantic_selective = point_active & (low > margin)
    structural_selective = point_active & (high < -margin)
    unresolved = point_active & ~(
        equivalent | semantic_selective | structural_selective
    )
    denominator = max(1, int(point_active.sum()))
    classification = {
        "equivalent": equivalent,
        "semantic_selective": semantic_selective,
        "structural_selective": structural_selective,
        "unresolved": unresolved,
    }
    classification_fraction = {
        name: float(mask.sum() / denominator) for name, mask in classification.items()
    }

    tracked = ("semantic_leaning", "structural_leaning", "central_responsive")
    frozen_sets = {
        name: {tuple(value) for value in families.get(name, ())} for name in tracked
    }
    inclusion = {name: np.zeros(shape, dtype=np.float64) for name in tracked}
    jaccard = {name: [] for name in tracked}
    p90_spans: list[float] = []
    fixed_tail_separation: list[float] = []
    rank_stability: list[float] = []
    point_rank_mask = point_active & np.isfinite(D)

    def ordinal_rank(value: np.ndarray) -> np.ndarray:
        order = np.argsort(value, kind="mergesort")
        result = np.empty(len(value), dtype=np.float64)
        result[order] = np.arange(len(value), dtype=np.float64)
        return result

    def fixed_family_mean(value: np.ndarray, name: str) -> float:
        members = frozen_sets[name]
        if not members:
            return np.nan
        selected = np.asarray([value[item] for item in members], dtype=np.float64)
        return float(np.mean(selected)) if np.isfinite(selected).all() else np.nan

    for draw in draws:
        draw_j = draw[4]
        draw_d = draw[5]
        draw_active = np.isfinite(draw_j) & np.isfinite(draw_d) & (
            draw_j >= float(activity_floor)
        )
        draw_families = _freeze_family_arrays(
            draw_j,
            draw_d,
            draw_active,
            tail_fraction=tail_fraction,
            central_fraction=central_fraction,
            central_pool_fraction=central_pool_fraction,
        )
        for name in tracked:
            selected = {tuple(value) for value in draw_families.get(name, ())}
            for item in selected:
                inclusion[name][item] += 1.0
            union = frozen_sets[name] | selected
            jaccard[name].append(
                float(len(frozen_sets[name] & selected) / len(union))
                if union
                else np.nan
            )
        active_values = draw_d[draw_active]
        p90_spans.append(
            float(np.quantile(active_values, 0.95) - np.quantile(active_values, 0.05))
            if active_values.size
            else np.nan
        )
        fixed_tail_separation.append(
            fixed_family_mean(draw_d, "semantic_leaning")
            - fixed_family_mean(draw_d, "structural_leaning")
        )
        candidate = draw_d[point_rank_mask]
        reference = D[point_rank_mask]
        if len(candidate) >= 3 and np.isfinite(candidate).all():
            rank_stability.append(
                float(np.corrcoef(ordinal_rank(reference), ordinal_rank(candidate))[0, 1])
            )
        else:
            rank_stability.append(np.nan)

    replicates = max(1, draws.shape[0])
    membership: dict[str, Any] = {}
    for name in tracked:
        values = np.asarray(jaccard[name], dtype=np.float64)
        membership[name] = {
            "frozen_members": tuple(sorted(frozen_sets[name])),
            "inclusion_probability": inclusion[name] / float(replicates),
            "mean_jaccard": finite_mean(values),
            "jaccard_low": finite_quantile(values, 0.025),
            "jaccard_high": finite_quantile(values, 0.975),
        }

    active_values = D[point_active & np.isfinite(D)]
    point_span = (
        float(np.quantile(active_values, 0.95) - np.quantile(active_values, 0.05))
        if active_values.size
        else np.nan
    )
    span_draws = np.asarray(p90_spans, dtype=np.float64)
    separation_draws = np.asarray(fixed_tail_separation, dtype=np.float64)
    rank_draws = np.asarray(rank_stability, dtype=np.float64)
    tail_separation = (
        fixed_family_mean(D, "semantic_leaning")
        - fixed_family_mean(D, "structural_leaning")
    )
    unstable = any(
        membership[name]["mean_jaccard"] < float(membership_stability_floor)
        for name in ("semantic_leaning", "structural_leaning")
    )
    narrow = bool(
        np.isfinite(span_draws).any()
        and finite_quantile(span_draws, 0.975) <= 2.0 * margin
    )
    equivalent_fraction = classification_fraction["equivalent"]
    entanglement_compatible = bool(
        narrow
        or unstable
        or equivalent_fraction >= float(generalist_fraction_floor)
    )
    resolved_relative_tails = bool(
        np.isfinite(separation_draws).any()
        and finite_quantile(separation_draws, 0.025) > 2.0 * margin
        and not unstable
    )
    if not coordinates.estimable or not point_active.any():
        status = "not_estimable"
    elif resolved_relative_tails:
        status = "resolved_relative_tails"
    elif entanglement_compatible:
        status = "entanglement_compatible"
    else:
        status = "mixed_or_unresolved"
    return {
        "equivalence_half_width": margin,
        "active_heads": int(point_active.sum()),
        "classification_masks": classification,
        "classification_fraction": classification_fraction,
        "active_selectivity": {
            "median": float(np.median(active_values)) if active_values.size else np.nan,
            "p05": float(np.quantile(active_values, 0.05)) if active_values.size else np.nan,
            "p95": float(np.quantile(active_values, 0.95)) if active_values.size else np.nan,
            "p90_span": point_span,
            "p90_span_low": finite_quantile(span_draws, 0.025),
            "p90_span_high": finite_quantile(span_draws, 0.975),
        },
        "frozen_tail_separation": {
            "estimate": float(tail_separation),
            "low": finite_quantile(separation_draws, 0.025),
            "high": finite_quantile(separation_draws, 0.975),
        },
        "membership": membership,
        "rank_stability": {
            "median": finite_quantile(rank_draws, 0.50),
            "low": finite_quantile(rank_draws, 0.025),
            "high": finite_quantile(rank_draws, 0.975),
        },
        "interpretation": {
            "status": status,
            "narrow_selectivity_distribution": narrow,
            "tail_membership_unstable": unstable,
            "equivalent_head_fraction_reaches_floor": bool(
                equivalent_fraction >= float(generalist_fraction_floor)
            ),
            "entanglement_compatible": entanglement_compatible,
            "resolved_relative_tails": resolved_relative_tails,
            "membership_stability_floor": float(membership_stability_floor),
            "generalist_fraction_floor": float(generalist_fraction_floor),
        },
        "bootstrap_replicates": int(draws.shape[0]),
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
