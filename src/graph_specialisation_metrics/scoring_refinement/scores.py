"""Pure score estimators and M1--M6 table construction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .config import METHODS
from .fields import GraphCapture, HeadFields


EPS = 1.0e-12


def transposition_permutation(n: int, source: int, partner: int, *, device: Any = None) -> Any:
    import torch

    permutation = torch.arange(int(n), dtype=torch.long, device=device)
    source, partner = int(source), int(partner)
    permutation[source] = partner
    permutation[partner] = source
    return permutation


def projected_event_score(
    clean: GraphCapture,
    variant: GraphCapture,
    gradients: Sequence[Any],
) -> np.ndarray:
    """Output-projected eventwise-gross score ``[L,H]``."""

    import torch

    output = []
    for layer, gradient in enumerate(gradients):
        delta = (
            clean.layers[layer].routed_output.detach()
            - variant.layers[layer].routed_output.detach()
        )
        projected = torch.einsum("tnhd,nhd->tnh", gradient, delta)
        magnitude = torch.linalg.vector_norm(projected, dim=0)
        output.append(magnitude.sum(dim=0).detach().cpu().numpy())
    return np.stack(output, axis=0)


def _masked_attention_overlap(
    event: Any,
    reference: Any,
    event_mask: Any,
    reference_mask: Any,
) -> tuple[Any, Any]:
    import torch

    support = event_mask | reference_mask
    left = torch.where(event_mask, event.float(), torch.zeros_like(event.float()))
    right = torch.where(reference_mask, reference.float(), torch.zeros_like(reference.float()))
    overlap = 1.0 - 0.5 * torch.abs(left - right).sum(dim=-1)
    valid = support.any(dim=-1)
    return torch.clamp(overlap, 0.0, 1.0), valid


def _weighted_message_cosine(
    event_message: Any,
    reference_message: Any,
    event_attention: Any,
    reference_attention: Any,
    event_mask: Any,
    reference_mask: Any,
    *,
    centered: bool,
) -> tuple[Any, Any, Any]:
    """Attention-probability-mass-weighted cosine by head/query."""

    import torch

    support = event_mask & reference_mask
    weight = 0.5 * (event_attention.float() + reference_attention.float())
    weight = torch.where(support, weight, torch.zeros_like(weight))
    mass = weight.sum(dim=-1)
    valid = mass > EPS
    left = torch.where(
        support.unsqueeze(-1), event_message.float(), torch.zeros_like(event_message.float())
    )
    right = torch.where(
        support.unsqueeze(-1),
        reference_message.float(),
        torch.zeros_like(reference_message.float()),
    )
    if centered:
        denom = mass.unsqueeze(-1).clamp_min(EPS)
        left_mean = (weight.unsqueeze(-1) * left).sum(dim=-2) / denom
        right_mean = (weight.unsqueeze(-1) * right).sum(dim=-2) / denom
        left = left - left_mean.unsqueeze(-2)
        right = right - right_mean.unsqueeze(-2)
    weighted = weight.unsqueeze(-1)
    numerator = (weighted * left * right).sum(dim=(-2, -1))
    left_norm = (weighted * left.square()).sum(dim=(-2, -1))
    right_norm = (weighted * right.square()).sum(dim=(-2, -1))
    cosine = numerator / torch.sqrt(left_norm * right_norm).clamp_min(EPS)
    cosine = torch.where(valid, cosine, torch.zeros_like(cosine))
    return torch.clamp(cosine, -1.0, 1.0), valid, mass


def _query_mean(value: Any, valid: Any) -> np.ndarray:
    import torch

    count = valid.sum(dim=-1).clamp_min(1).to(value.dtype)
    result = torch.where(valid, value, torch.zeros_like(value)).sum(dim=-1) / count
    return result.detach().cpu().numpy()


def follow_invariant_scores(
    clean: GraphCapture,
    event: GraphCapture,
    permutation: Any,
) -> dict[str, np.ndarray]:
    """Return probability-mass-weighted attention and transport agreements.

    Results are ``[L,H]`` arrays.  Query rows are averaged only after the
    per-key attention-mass weighting has been applied.
    """

    attention_follow = []
    attention_invariant = []
    transport_follow = []
    transport_invariant = []
    transport_follow_raw = []
    transport_invariant_raw = []
    transport_follow_centered = []
    transport_invariant_centered = []
    effective_support = []
    attention_mass = []
    for clean_layer, event_layer in zip(clean.layers, event.layers):
        follow_attention_reference = clean_layer.attention.index_select(-1, permutation)
        follow_message_reference = clean_layer.message.index_select(-2, permutation)
        follow_mask_reference = clean_layer.mask.index_select(-1, permutation)

        attn_inv, valid_attn_inv = _masked_attention_overlap(
            event_layer.attention,
            clean_layer.attention,
            event_layer.mask,
            clean_layer.mask,
        )
        attn_follow, valid_attn_follow = _masked_attention_overlap(
            event_layer.attention,
            follow_attention_reference,
            event_layer.mask,
            follow_mask_reference,
        )
        inv_raw, valid_inv, inv_mass = _weighted_message_cosine(
            event_layer.message,
            clean_layer.message,
            event_layer.attention,
            clean_layer.attention,
            event_layer.mask,
            clean_layer.mask,
            centered=False,
        )
        follow_raw, valid_follow, follow_mass = _weighted_message_cosine(
            event_layer.message,
            follow_message_reference,
            event_layer.attention,
            follow_attention_reference,
            event_layer.mask,
            follow_mask_reference,
            centered=False,
        )
        inv_centered, valid_inv_centered, _ = _weighted_message_cosine(
            event_layer.message,
            clean_layer.message,
            event_layer.attention,
            clean_layer.attention,
            event_layer.mask,
            clean_layer.mask,
            centered=True,
        )
        follow_centered, valid_follow_centered, _ = _weighted_message_cosine(
            event_layer.message,
            follow_message_reference,
            event_layer.attention,
            follow_attention_reference,
            event_layer.mask,
            follow_mask_reference,
            centered=True,
        )
        attention_invariant.append(_query_mean(attn_inv, valid_attn_inv))
        attention_follow.append(_query_mean(attn_follow, valid_attn_follow))
        transport_invariant.append(_query_mean(inv_raw.clamp(0.0, 1.0), valid_inv))
        transport_follow.append(_query_mean(follow_raw.clamp(0.0, 1.0), valid_follow))
        transport_invariant_raw.append(_query_mean(inv_raw, valid_inv))
        transport_follow_raw.append(_query_mean(follow_raw, valid_follow))
        transport_invariant_centered.append(
            _query_mean(inv_centered, valid_inv_centered)
        )
        transport_follow_centered.append(
            _query_mean(follow_centered, valid_follow_centered)
        )
        joint_valid = valid_inv | valid_follow
        effective_support.append(
            _query_mean(
                (event_layer.mask | clean_layer.mask).sum(dim=-1).float(),
                joint_valid,
            )
        )
        attention_mass.append(
            _query_mean(0.5 * (inv_mass + follow_mass), joint_valid)
        )
    return {
        "attention_follow": np.stack(attention_follow),
        "attention_invariant": np.stack(attention_invariant),
        "transport_follow": np.stack(transport_follow),
        "transport_invariant": np.stack(transport_invariant),
        "transport_follow_raw": np.stack(transport_follow_raw),
        "transport_invariant_raw": np.stack(transport_invariant_raw),
        "transport_follow_centered": np.stack(transport_follow_centered),
        "transport_invariant_centered": np.stack(transport_invariant_centered),
        "effective_support": np.stack(effective_support),
        "attention_mass": np.stack(attention_mass),
    }


def hierarchical_event_mean(groups: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """Average events within source, then sources equally."""

    source_values = [
        np.mean(np.stack(list(events), axis=0), axis=0)
        for events in groups
        if len(events)
    ]
    if not source_values:
        raise ValueError("cannot aggregate an empty intervention hierarchy")
    return np.mean(np.stack(source_values, axis=0), axis=0)


def graph_balanced_mean(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        raise ValueError("cannot aggregate zero graphs")
    return np.mean(np.stack(list(values), axis=0), axis=0)


def dj_coordinates(
    semantic: np.ndarray,
    pe: np.ndarray,
    semantic_reference: float,
    pe_reference: float,
) -> tuple[np.ndarray, np.ndarray]:
    semantic_norm = np.asarray(semantic, dtype=float) / max(float(semantic_reference), EPS)
    pe_norm = np.asarray(pe, dtype=float) / max(float(pe_reference), EPS)
    total = semantic_norm + pe_norm
    return (semantic_norm - pe_norm) / (total + EPS), total / 2.0


def _positive_mean(value: np.ndarray) -> float:
    finite = np.asarray(value, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 1.0
    return max(float(np.mean(finite)), EPS)


def _raw_method_axes(score: Mapping[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "M1_DD": (score["eg_semantic_single"], score["eg_pe_single"]),
        "M1_DT": (score["eg_semantic_single"], score["eg_pe_transposition"]),
        "M1_TD": (score["eg_semantic_transposition"], score["eg_pe_single"]),
        "M1_TT": (score["eg_semantic_transposition"], score["eg_pe_transposition"]),
        "M2": (
            score["semantic_transport_follow"],
            score["semantic_transport_invariant"],
        ),
        "M3": (
            score["pe_transport_invariant"],
            score["pe_transport_follow"],
        ),
        "M4": (
            score["semantic_attention_follow"],
            score["semantic_attention_invariant"],
        ),
        "M5": (
            score["pe_attention_invariant"],
            score["pe_attention_follow"],
        ),
        "M6": (
            score["semantic_transport_follow"],
            score["pe_transport_follow"],
        ),
    }


def method_references(
    score: Mapping[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    """Predeclared references.

    All M1 arms share the current-method ``semantic donor / PE transposition``
    references, so changing an intervention variant remains visible rather than
    being normalized away.  Other methods use fixed discovery-split axis means.
    """

    axes = _raw_method_axes(score)
    m1_reference = (
        _positive_mean(score["eg_semantic_single"]),
        _positive_mean(score["eg_pe_transposition"]),
    )
    output = {}
    for method, (semantic, pe) in axes.items():
        output[method] = (
            m1_reference
            if method.startswith("M1_")
            else (_positive_mean(semantic), _positive_mean(pe))
        )
    return output


def build_score_tables(
    task: str,
    checkpoint_sha: str,
    score: Mapping[str, np.ndarray],
    *,
    graph_count: int,
    event_count: int,
    methods: Sequence[str] = METHODS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, tuple[float, float]]]:
    """Build the stable raw and derived per-head output schemas."""

    axes = _raw_method_axes(score)
    references = method_references(score)
    topology = np.asarray(score["topology_eg"], dtype=float)
    raw_rows: list[dict[str, Any]] = []
    derived_rows: list[dict[str, Any]] = []
    m1_interventions = {
        "M1_DD": ("semantic_single_donor", "pe_single_donor"),
        "M1_DT": ("semantic_single_donor", "pe_transposition"),
        "M1_TD": ("semantic_transposition", "pe_single_donor"),
        "M1_TT": ("semantic_transposition", "pe_transposition"),
    }
    field_by_method = {
        "M1_DD": "output_projected_transport",
        "M1_DT": "output_projected_transport",
        "M1_TD": "output_projected_transport",
        "M1_TT": "output_projected_transport",
        "M2": "message_transport",
        "M3": "message_transport",
        "M4": "attention",
        "M5": "attention",
        "M6": "message_transport",
    }
    for method in methods:
        semantic, pe = axes[method]
        sem_ref, pe_ref = references[method]
        d_rel, joint = dj_coordinates(semantic, pe, sem_ref, pe_ref)
        sem_intervention, pe_intervention = m1_interventions.get(
            method,
            (
                "semantic_transposition" if method in {"M2", "M4", "M6"} else "",
                "pe_transposition" if method in {"M3", "M5", "M6"} else "",
            ),
        )
        coordinate_kind = "scientific" if method.startswith("M1_") or method == "M6" \
            else "diagnostic"
        for layer in range(int(semantic.shape[0])):
            for head in range(int(semantic.shape[1])):
                common = {
                    "task": task,
                    "checkpoint_sha": checkpoint_sha,
                    "graph_split": "score",
                    "layer": layer,
                    "head": head,
                    "method": method,
                    "semantic_intervention": sem_intervention,
                    "pe_intervention": pe_intervention,
                    "field": field_by_method[method],
                    "graphs": int(graph_count),
                    "events": int(event_count),
                }
                raw_rows.append(
                    {
                        **common,
                        "semantic_score": float(semantic[layer, head]),
                        "pe_score": float(pe[layer, head]),
                        "topology_score": float(topology[layer, head]),
                        "centered": False,
                        "probability_weighting": (
                            "attention_probability_mass"
                            if method in {"M2", "M3", "M4", "M5", "M6"}
                            else "none"
                        ),
                    }
                )
                derived_rows.append(
                    {
                        **common,
                        "semantic_reference": sem_ref,
                        "pe_reference": pe_ref,
                        "D_rel": float(d_rel[layer, head]),
                        "J": float(joint[layer, head]),
                        "coordinate_kind": coordinate_kind,
                    }
                )
    return raw_rows, derived_rows, references
