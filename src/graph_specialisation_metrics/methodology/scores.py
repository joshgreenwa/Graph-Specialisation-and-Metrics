"""Pure estimators for canonical raw head scores and derived coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .audit import audit_check


SEMANTIC_AXIS_LABEL = r"Semantic score  $S_{sem}/\overline{S}_{sem}$"
STRUCTURAL_AXIS_LABEL = r"Structural score  $S_{str}/\overline{S}_{str}$"
SELECTIVITY_AXIS_LABEL = (
    r"Selectivity $D_{rel}$  (structural $\leftarrow$ 0 $\rightarrow$ semantic)"
)
JOINT_AXIS_LABEL = r"Joint sensitivity $J$"
CONFIDENCE_SPECIALIST_VERSION = "bootstrap-confidence-specialists-v1"


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
    finite_delta = torch.isfinite(delta)
    finite_gradient = torch.isfinite(clean_gradient)
    if not bool(finite_delta.all() and finite_gradient.all()):
        audit_check(
            False,
            "scores.non_finite_transport_input",
            "non-finite transport/Jacobian entries were replaced by zero; "
            "the run continues but is not headline eligible",
            context={
                "non_finite_delta": int((~finite_delta).sum().item()),
                "non_finite_gradient": int((~finite_gradient).sum().item()),
            },
            strict=False,
        )
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        clean_gradient = torch.nan_to_num(
            clean_gradient, nan=0.0, posinf=0.0, neginf=0.0
        )
    return torch.einsum("elnhd,tlnhd->elhnt", delta, clean_gradient)


def event_head_scores(q, *, system: str = "mass"):
    """Return one registered event/head score system.

    ``mass`` takes an output norm at each carrier before summing carriers.
    ``coherent`` first sums the carrier output-movement vectors and then takes
    the output norm, so mutually cancelling carrier effects are not counted as
    independent evidence of head engagement.
    """

    systems = event_head_score_systems(q)
    if system not in {"mass", "coherent"}:
        raise ValueError(f"unknown head score system {system!r}")
    return systems[system]


def event_head_score_systems(q, *, mass_floor: float = 0.0) -> dict[str, Any]:
    """Return transport mass, coherent movement, and event-level carrier coherence.

    Args:
        q: ``[event, layer, head, carrier, output]`` projected transport.
        mass_floor: events at or below this mass are non-estimable for the ratio.
    """

    import torch

    if q.ndim != 5:
        raise ValueError("projected transport must be [event,layer,head,carrier,output]")
    finite_q = torch.isfinite(q)
    if not bool(finite_q.all()):
        audit_check(
            False,
            "scores.non_finite_projected_transport",
            "non-finite projected transport entries were replaced by zero; "
            "the run continues but is not headline eligible",
            context={"non_finite_entries": int((~finite_q).sum().item())},
            strict=False,
        )
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    mass = q.square().sum(dim=-1).sqrt().sum(dim=-1)
    coherent_vector = q.sum(dim=-2)
    coherent = coherent_vector.square().sum(dim=-1).sqrt()
    tolerance = 32.0 * torch.finfo(q.dtype).eps * torch.maximum(
        mass, torch.ones_like(mass)
    )
    violation = coherent > mass + tolerance
    if bool(violation.any()):
        maximum_excess = float((coherent - mass)[violation].max().item())
        audit_check(
            False,
            "scores.coherent_exceeds_mass",
            "coherent movement exceeded transport mass numerically and was clipped "
            "to the triangle-inequality bound",
            observed=maximum_excess,
            tolerance=float(tolerance[violation].max().item()),
            context={"affected_entries": int(violation.sum().item())},
            strict=False,
        )
    coherent = torch.minimum(coherent, mass)
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
    negative = (semantic < 0) | (structural < 0)
    if np.any(negative):
        negative_values = np.concatenate(
            (semantic[semantic < 0], structural[structural < 0])
        )
        audit_check(
            False,
            "scores.negative_raw_score",
            "negative raw scores were clipped to zero; the run continues but is "
            "not headline eligible",
            observed=float(abs(np.min(negative_values))),
            tolerance=0.0,
            context={"affected_heads": int(np.sum(negative))},
            strict=False,
        )
        semantic = np.maximum(semantic, 0.0)
        structural = np.maximum(structural, 0.0)
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


def confidence_specialists_from_draws(
    coordinates: HeadCoordinates,
    joint_draws: Any,
    selectivity_draws: Any,
    *,
    activity_floor: float = 0.20,
    preference_threshold: float = 0.10,
    confidence: float = 0.95,
    minimum_pairs: int = 3,
) -> dict[str, Any]:
    """Freeze reliable directional specialists from discovery bootstrap support.

    A specialist must clear the activity and directional-margin rules jointly in
    at least ``confidence`` of complete nested-bootstrap draws. Opposite-sign
    point-active heads form the causal comparison pools but receive no
    specialist label unless they independently clear this confidence gate.
    """

    J = np.asarray(coordinates.joint_sensitivity, dtype=np.float64)
    D = np.asarray(coordinates.selectivity, dtype=np.float64)
    j_draws = np.asarray(joint_draws, dtype=np.float64)
    d_draws = np.asarray(selectivity_draws, dtype=np.float64)
    if J.shape != D.shape or J.ndim != 2:
        raise ValueError("specialist point coordinates must share [layer,head] shape")
    if j_draws.shape != d_draws.shape or j_draws.ndim != 3:
        raise ValueError("J and D_rel draws must share [replicate,layer,head] shape")
    if tuple(j_draws.shape[1:]) != tuple(J.shape):
        raise ValueError("bootstrap head geometry does not match point coordinates")
    if not 0.5 < float(confidence) < 1.0:
        raise ValueError("specialist confidence must lie in (0.5, 1)")
    if float(preference_threshold) <= 0:
        raise ValueError("specialist preference threshold must be positive")
    if float(activity_floor) < 0:
        raise ValueError("specialist activity floor must be non-negative")
    if int(minimum_pairs) < 1:
        raise ValueError("minimum specialist pairs must be positive")

    finite = np.isfinite(j_draws) & np.isfinite(d_draws)
    semantic_support = np.mean(
        finite
        & (j_draws >= float(activity_floor))
        & (d_draws > float(preference_threshold)),
        axis=0,
    )
    structural_support = np.mean(
        finite
        & (j_draws >= float(activity_floor))
        & (d_draws < -float(preference_threshold)),
        axis=0,
    )
    generalist_support = np.mean(
        finite
        & (j_draws >= float(activity_floor))
        & (np.abs(d_draws) <= float(preference_threshold)),
        axis=0,
    )
    activity_support = np.mean(
        np.isfinite(j_draws) & (j_draws >= float(activity_floor)), axis=0
    )
    point_active = (
        np.asarray(coordinates.active, dtype=bool)
        & np.isfinite(J)
        & np.isfinite(D)
    )
    semantic_mask = point_active & (semantic_support >= float(confidence))
    structural_mask = point_active & (structural_support >= float(confidence))
    generalist_mask = (
        point_active
        & ~semantic_mask
        & ~structural_mask
        & (generalist_support >= float(confidence))
    )
    unresolved_mask = point_active & ~semantic_mask & ~structural_mask & ~generalist_mask
    inactive_mask = ~point_active

    def heads(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(layer), int(head))
            for layer, head in np.argwhere(mask).tolist()
        )

    semantic = heads(semantic_mask)
    structural = heads(structural_mask)

    def optimal_match(
        left: Sequence[tuple[int, int]],
        right: Sequence[tuple[int, int]],
        *,
        left_label: str,
        right_label: str,
    ) -> dict[str, Any]:
        pairs: list[dict[str, Any]] = []
        if left and right:
            from scipy.optimize import linear_sum_assignment

            cost = np.asarray(
                [[abs(float(J[a]) - float(J[b])) for b in right] for a in left],
                dtype=np.float64,
            )
            left_positions, right_positions = linear_sum_assignment(cost)
            for left_position, right_position in zip(
                left_positions.tolist(), right_positions.tolist()
            ):
                a = tuple(left[left_position])
                b = tuple(right[right_position])
                pairs.append(
                    {
                        left_label: a,
                        right_label: b,
                        f"{left_label}_J": float(J[a]),
                        f"{right_label}_J": float(J[b]),
                        "absolute_J_gap": float(cost[left_position, right_position]),
                        "absolute_layer_gap": abs(int(a[0]) - int(b[0])),
                    }
                )
        return {
            "method": "minimum-total-absolute-J assignment without replacement",
            "left_label": left_label,
            "right_label": right_label,
            "pairs": tuple(pairs),
            "pair_count": len(pairs),
            "mean_absolute_J_gap": (
                float(np.mean([row["absolute_J_gap"] for row in pairs]))
                if pairs
                else np.nan
            ),
            "mean_absolute_layer_gap": (
                float(np.mean([row["absolute_layer_gap"] for row in pairs]))
                if pairs
                else np.nan
            ),
        }

    specialist_matching = optimal_match(
        semantic,
        structural,
        left_label="semantic",
        right_label="structural",
    )
    negative_pool = heads(point_active & (D < 0))
    positive_pool = heads(point_active & (D > 0))
    semantic_null_matching = optimal_match(
        semantic,
        negative_pool,
        left_label="specialist",
        right_label="null",
    )
    structural_null_matching = optimal_match(
        structural,
        positive_pool,
        left_label="specialist",
        right_label="null",
    )

    return {
        "version": CONFIDENCE_SPECIALIST_VERSION,
        "selection_uses_causal_outcomes": False,
        "activity_floor": float(activity_floor),
        "preference_threshold": float(preference_threshold),
        "bootstrap_confidence": float(confidence),
        "minimum_pairs": int(minimum_pairs),
        "support": {
            "activity": activity_support,
            "semantic": semantic_support,
            "structural": structural_support,
            "generalist": generalist_support,
        },
        "masks": {
            "semantic_specialist": semantic_mask,
            "structural_specialist": structural_mask,
            "persistent_generalist": generalist_mask,
            "unresolved": unresolved_mask,
            "inactive": inactive_mask,
        },
        "heads": {
            "semantic_specialist": semantic,
            "structural_specialist": structural,
            "persistent_generalist": heads(generalist_mask),
            "unresolved": heads(unresolved_mask),
            "inactive": heads(inactive_mask),
            "semantic_null_pool": negative_pool,
            "structural_null_pool": positive_pool,
        },
        "specialist_J_matching": specialist_matching,
        "semantic_null_J_matching": semantic_null_matching,
        "structural_null_J_matching": structural_null_matching,
        "status": (
            "estimable"
            if int(specialist_matching["pair_count"]) >= int(minimum_pairs)
            else "not_estimable"
        ),
    }


def graph_local_fixed_normalization(
    semantic_graph_scores: Mapping[int, Any],
    structural_graph_scores: Mapping[int, Any],
    coordinates: HeadCoordinates,
    *,
    epsilon: float,
    preference_threshold: float = 0.10,
) -> dict[str, Any]:
    """Molecule-level diagnostic under frozen aggregate channel normalization."""

    graph_ids = sorted(set(semantic_graph_scores) & set(structural_graph_scores))
    if not graph_ids:
        raise ValueError("semantic and structural score caches share no graph IDs")
    if not coordinates.estimable:
        raise ValueError("graph-local D_rel is unavailable for non-estimable coordinates")
    semantic = np.stack(
        [np.asarray(semantic_graph_scores[key], dtype=np.float64) for key in graph_ids]
    ) / float(coordinates.semantic_mean)
    structural = np.stack(
        [np.asarray(structural_graph_scores[key], dtype=np.float64) for key in graph_ids]
    ) / float(coordinates.structural_mean)
    joint = 0.5 * (semantic + structural)
    selectivity = (semantic - structural) / (
        semantic + structural + float(epsilon)
    )
    threshold = float(preference_threshold)
    return {
        "graph_ids": tuple(int(value) for value in graph_ids),
        "joint_sensitivity": joint,
        "selectivity": selectivity,
        "semantic_fraction": np.mean(selectivity > threshold, axis=0),
        "structural_fraction": np.mean(selectivity < -threshold, axis=0),
        "central_fraction": np.mean(np.abs(selectivity) <= threshold, axis=0),
        "normalization": "frozen aggregate discovery channel means",
    }


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


def freeze_threshold_specialists(
    coordinates: HeadCoordinates,
    *,
    selectivity_interval: tuple[Any, Any],
    preference_threshold: float,
    activity_threshold: float,
    candidate_limit: int = 6,
    minimum_candidate_pairs: int = 3,
) -> dict[str, Any]:
    """Freeze strongest directional candidates and a 95%-confirmed tier.

    The key categorical analysis uses at most ``candidate_limit`` heads on each
    side whose point estimate clears the fixed preference threshold, ranked by
    absolute ``D_rel`` and optimally matched on ``J``. The stricter confidence-
    interval rule is retained as a separately labelled robustness population.
    Neither selection uses a causal endpoint.
    """

    J = np.asarray(coordinates.joint_sensitivity, dtype=np.float64)
    D = np.asarray(coordinates.selectivity, dtype=np.float64)
    active = np.asarray(coordinates.active, dtype=bool)
    low, high = (
        np.asarray(value, dtype=np.float64) for value in selectivity_interval
    )
    if (
        J.shape != D.shape
        or J.shape != active.shape
        or low.shape != J.shape
        or high.shape != J.shape
    ):
        raise ValueError(
            "specialist coordinates and intervals must share [layer,head] shape"
        )
    threshold = float(preference_threshold)
    if threshold <= 0:
        raise ValueError("specialist preference threshold must be positive")
    activity_floor = float(activity_threshold)
    if activity_floor < 0:
        raise ValueError("specialist activity threshold must be non-negative")
    candidate_limit = int(candidate_limit)
    minimum_candidate_pairs = int(minimum_candidate_pairs)
    if candidate_limit < 1:
        raise ValueError("specialist candidate limit must be positive")
    if not 1 <= minimum_candidate_pairs <= candidate_limit:
        raise ValueError(
            "minimum candidate pairs must lie between 1 and candidate limit"
        )
    active = active & (J >= activity_floor)
    finite = np.isfinite(J) & np.isfinite(D) & np.isfinite(low) & np.isfinite(high)
    semantic_candidate_mask = active & finite & (D > threshold)
    structural_candidate_mask = active & finite & (D < -threshold)
    semantic_confirmed_mask = semantic_candidate_mask & (low > threshold)
    structural_confirmed_mask = structural_candidate_mask & (high < -threshold)
    generalist_mask = active & finite & (D >= -threshold) & (D <= threshold)
    unresolved_mask = active & ~finite
    inactive_mask = ~active

    def heads(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(layer), int(head))
            for layer, head in np.argwhere(mask).tolist()
        )

    def mask_for(selected: Sequence[tuple[int, int]]) -> np.ndarray:
        mask = np.zeros(J.shape, dtype=bool)
        for head in selected:
            mask[head] = True
        return mask

    semantic_candidates = heads(semantic_candidate_mask)
    structural_candidates = heads(structural_candidate_mask)
    semantic_confirmed = heads(semantic_confirmed_mask)
    structural_confirmed = heads(structural_confirmed_mask)
    ranked_semantic = tuple(
        sorted(semantic_candidates, key=lambda item: (-float(D[item]), item))
    )
    ranked_structural = tuple(
        sorted(structural_candidates, key=lambda item: (float(D[item]), item))
    )
    selected_semantic = ranked_semantic[:candidate_limit]
    selected_structural = ranked_structural[:candidate_limit]

    def head_record(head: tuple[int, int]) -> dict[str, Any]:
        direction = "semantic" if float(D[head]) > 0 else "structural"
        conservative_margin = (
            float(low[head] - threshold)
            if direction == "semantic"
            else float(-threshold - high[head])
        )
        return {
            "head": head,
            "J": float(J[head]),
            "D_rel": float(D[head]),
            "absolute_D_rel": abs(float(D[head])),
            "conservative_margin": conservative_margin,
            "confirmed_95": bool(
                semantic_confirmed_mask[head] or structural_confirmed_mask[head]
            ),
        }

    def match(
        semantic_heads: Sequence[tuple[int, int]],
        structural_heads: Sequence[tuple[int, int]],
    ) -> dict[str, Any]:
        pairs: list[dict[str, Any]] = []
        if semantic_heads and structural_heads:
            from scipy.optimize import linear_sum_assignment

            cost = np.asarray(
                [
                    [
                        abs(float(J[sem]) - float(J[struct]))
                        for struct in structural_heads
                    ]
                    for sem in semantic_heads
                ],
                dtype=np.float64,
            )
            semantic_positions, structural_positions = linear_sum_assignment(cost)
            for sem_position, struct_position in zip(
                semantic_positions.tolist(),
                structural_positions.tolist(),
            ):
                sem = tuple(semantic_heads[sem_position])
                struct = tuple(structural_heads[struct_position])
                gap = float(cost[sem_position, struct_position])
                pairs.append(
                    {
                        "semantic": sem,
                        "structural": struct,
                        "semantic_layer": int(sem[0]),
                        "structural_layer": int(struct[0]),
                        "absolute_layer_gap": abs(int(sem[0]) - int(struct[0])),
                        "semantic_J": float(J[sem]),
                        "structural_J": float(J[struct]),
                        "absolute_J_gap": gap,
                        "semantic_D_rel": float(D[sem]),
                        "structural_D_rel": float(D[struct]),
                        "pair_strength": min(abs(float(D[sem])), abs(float(D[struct]))),
                        "semantic_confirmed_95": bool(
                            semantic_confirmed_mask[sem]
                        ),
                        "structural_confirmed_95": bool(
                            structural_confirmed_mask[struct]
                        ),
                        "pair_confirmed_95": bool(
                            semantic_confirmed_mask[sem]
                            and structural_confirmed_mask[struct]
                        ),
                    }
                )
        semantic_j = np.asarray(
            [row["semantic_J"] for row in pairs], dtype=np.float64
        )
        structural_j = np.asarray(
            [row["structural_J"] for row in pairs], dtype=np.float64
        )
        pooled_scale = (
            float(np.sqrt(0.5 * (np.var(semantic_j) + np.var(structural_j))))
            if len(pairs) > 1
            else np.nan
        )
        standardized_difference = (
            float((np.mean(semantic_j) - np.mean(structural_j)) / pooled_scale)
            if np.isfinite(pooled_scale) and pooled_scale > 0
            else np.nan
        )
        return {
            "method": (
                "minimum-total-absolute-J one-to-one assignment within seed, "
                "without replacement; layer is a balance audit, not a gate"
            ),
            "pairs": tuple(pairs),
            "matched_pair_count": len(pairs),
            "mean_semantic_J": (
                float(np.mean(semantic_j)) if semantic_j.size else np.nan
            ),
            "mean_structural_J": (
                float(np.mean(structural_j)) if structural_j.size else np.nan
            ),
            "mean_absolute_J_gap": (
                float(np.mean([row["absolute_J_gap"] for row in pairs]))
                if pairs
                else np.nan
            ),
            "mean_absolute_layer_gap": (
                float(np.mean([row["absolute_layer_gap"] for row in pairs]))
                if pairs
                else np.nan
            ),
            "exact_layer_pair_fraction": (
                float(np.mean([row["absolute_layer_gap"] == 0 for row in pairs]))
                if pairs
                else np.nan
            ),
            "standardized_J_difference": standardized_difference,
        }

    candidate_matching = match(selected_semantic, selected_structural)
    confirmed_matching = match(semantic_confirmed, structural_confirmed)
    candidate_count = int(candidate_matching["matched_pair_count"])
    confirmed_count = int(confirmed_matching["matched_pair_count"])

    return {
        "preference_threshold": threshold,
        "activity_threshold": activity_floor,
        "selection_uses_causal_outcomes": False,
        "rule": (
            "strongest candidates use active point estimates beyond +/- threshold; "
            "95%-confirmed specialists require the complete D_rel interval beyond "
            "the same threshold"
        ),
        "heads": {
            "semantic_candidate_pool": semantic_candidates,
            "structural_candidate_pool": structural_candidates,
            "semantic_selected": selected_semantic,
            "structural_selected": selected_structural,
            "semantic_confirmed_95": semantic_confirmed,
            "structural_confirmed_95": structural_confirmed,
            "generalist": heads(generalist_mask),
            "unresolved": heads(unresolved_mask),
            "inactive": heads(inactive_mask),
        },
        "strength_ranking": {
            "measure": "descending absolute point-estimate D_rel within direction",
            "semantic_candidates": tuple(
                head_record(head) for head in ranked_semantic
            ),
            "structural_candidates": tuple(
                head_record(head) for head in ranked_structural
            ),
        },
        "masks": {
            "semantic_candidate": semantic_candidate_mask,
            "structural_candidate": structural_candidate_mask,
            "semantic_selected": mask_for(selected_semantic),
            "structural_selected": mask_for(selected_structural),
            "semantic_confirmed_95": semantic_confirmed_mask,
            "structural_confirmed_95": structural_confirmed_mask,
            "generalist": generalist_mask,
            "unresolved": unresolved_mask,
            "inactive": inactive_mask,
        },
        "candidate_analysis": {
            "status": (
                "estimable"
                if candidate_count >= minimum_candidate_pairs
                else "not_estimable"
            ),
            "candidate_limit_per_direction": candidate_limit,
            "minimum_pairs_per_seed": minimum_candidate_pairs,
            "semantic_pool_count": len(semantic_candidates),
            "structural_pool_count": len(structural_candidates),
            "selected_semantic_count": len(selected_semantic),
            "selected_structural_count": len(selected_structural),
            "j_matching": candidate_matching,
        },
        "confirmed_95_robustness": {
            "status": "available" if confirmed_count else "not_estimable",
            "population_rule": (
                "headline robustness requires at least 8 total pairs across at "
                "least 3 trained seeds"
            ),
            "semantic_count": len(semantic_confirmed),
            "structural_count": len(structural_confirmed),
            "j_matching": confirmed_matching,
        },
        # Compatibility alias for consumers that only need the key categorical
        # matching record. It now denotes the strongest-candidate analysis.
        "j_matching": candidate_matching,
    }


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
