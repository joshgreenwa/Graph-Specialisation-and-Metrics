"""Pure score estimators and M1--M6 table construction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .config import METHODS, PROTOCOL_VERSION
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


def _pair_cosine(
    event: Any,
    reference: Any,
    valid: Any,
    *,
    reduce_dims: tuple[int, ...],
) -> tuple[Any, Any]:
    """Cosine similarity with an explicit valid-query mask."""

    import torch

    event = event.float()
    reference = reference.float()
    numerator = (event * reference).sum(dim=reduce_dims)
    event_norm = event.square().sum(dim=reduce_dims)
    reference_norm = reference.square().sum(dim=reduce_dims)
    norm_valid = (event_norm > EPS) & (reference_norm > EPS)
    valid = valid & norm_valid
    cosine = numerator / torch.sqrt(event_norm * reference_norm).clamp_min(EPS)
    cosine = torch.where(valid, cosine, torch.zeros_like(cosine))
    return torch.clamp(cosine, -1.0, 1.0), valid


def _transposition_pair(permutation: Any) -> tuple[int, int]:
    import torch

    permutation = permutation.reshape(-1)
    identity = torch.arange(
        int(permutation.numel()), device=permutation.device, dtype=permutation.dtype
    )
    changed = torch.nonzero(permutation != identity, as_tuple=False).reshape(-1)
    if int(changed.numel()) != 2:
        raise ValueError(
            "appendix cosine scoring requires one non-trivial two-node transposition"
        )
    return int(changed[0].item()), int(changed[1].item())


def _appendix_event_components(
    clean: GraphCapture,
    event: GraphCapture,
    permutation: Any,
) -> dict[str, Any]:
    """Per-query Appendix A.3 cosine terms for one node transposition.

    Graph nodes are singleton blocks. For every receiver/query, the two swapped
    sender locations form ``v_ij``. Attention uses the two scalar masses;
    transport uses the flattened pair of realised contributions ``alpha * m``.
    """

    import torch

    source, partner = _transposition_pair(permutation)
    output: dict[str, list[Any]] = {
        "attention_follow": [],
        "attention_invariant": [],
        "transport_follow": [],
        "transport_invariant": [],
        "attention_valid": [],
        "transport_valid": [],
        "alpha_logit": [],
        "query_mass": [],
    }
    for clean_layer, event_layer in zip(clean.layers, event.layers):
        indices = torch.as_tensor(
            [source, partner],
            dtype=torch.long,
            device=clean_layer.attention.device,
        )
        clean_attention = clean_layer.attention.index_select(-1, indices).float()
        event_attention = event_layer.attention.index_select(-1, indices).float()
        clean_mask = clean_layer.mask.index_select(-1, indices)
        event_mask = event_layer.mask.index_select(-1, indices)
        valid_pair = (clean_mask & event_mask).all(dim=-1)

        attention_invariant, attention_valid = _pair_cosine(
            event_attention,
            clean_attention,
            valid_pair,
            reduce_dims=(-1,),
        )
        attention_follow, attention_follow_valid = _pair_cosine(
            event_attention,
            clean_attention.flip(dims=(-1,)),
            valid_pair,
            reduce_dims=(-1,),
        )
        attention_valid = attention_valid & attention_follow_valid

        clean_message = clean_layer.message.index_select(-2, indices).float()
        event_message = event_layer.message.index_select(-2, indices).float()
        clean_transport = clean_attention.unsqueeze(-1) * clean_message
        event_transport = event_attention.unsqueeze(-1) * event_message
        transport_invariant, transport_valid = _pair_cosine(
            event_transport,
            clean_transport,
            valid_pair,
            reduce_dims=(-2, -1),
        )
        transport_follow, transport_follow_valid = _pair_cosine(
            event_transport,
            clean_transport.flip(dims=(-2,)),
            valid_pair,
            reduce_dims=(-2, -1),
        )
        transport_valid = transport_valid & transport_follow_valid

        output["attention_follow"].append(attention_follow)
        output["attention_invariant"].append(attention_invariant)
        output["transport_follow"].append(transport_follow)
        output["transport_invariant"].append(transport_invariant)
        output["attention_valid"].append(attention_valid)
        output["transport_valid"].append(transport_valid)
        output["alpha_logit"].append(
            torch.abs(clean_attention[..., 0] - clean_attention[..., 1])
        )
        output["query_mass"].append(clean_attention.sum(dim=-1))
    return {key: torch.stack(values, dim=0) for key, values in output.items()}


def appendix_cosine_group_scores(
    clean: GraphCapture,
    events: Sequence[GraphCapture],
    permutations: Sequence[Any],
    *,
    temperature: float,
) -> dict[str, np.ndarray]:
    """Appendix A.3 cosine scores aggregated over sampled transpositions.

    The paper's swap weight is applied as
    ``softmax(|d_i-d_j| / temperature)`` over events for each layer, head, and
    graph query. Graph queries are then averaged using their clean attention
    mass on the swapped node pair.
    """

    import torch

    if len(events) != len(permutations) or not len(events):
        raise ValueError("events and permutations must be aligned and non-empty")
    if float(temperature) <= 0.0:
        raise ValueError("cosine temperature must be positive")
    components = [
        _appendix_event_components(clean, event, permutation)
        for event, permutation in zip(events, permutations)
    ]
    logits = torch.stack([item["alpha_logit"] for item in components], dim=0)
    query_mass = torch.stack([item["query_mass"] for item in components], dim=0)
    output: dict[str, np.ndarray] = {}
    valid_by_family = {
        "attention": torch.stack(
            [item["attention_valid"] for item in components], dim=0
        ),
        "transport": torch.stack(
            [item["transport_valid"] for item in components], dim=0
        ),
    }
    for family in ("attention", "transport"):
        valid = valid_by_family[family]
        any_event = valid.any(dim=0, keepdim=True)
        masked_logits = torch.where(
            valid,
            logits / float(temperature),
            torch.full_like(logits, -torch.inf),
        )
        masked_logits = torch.where(
            any_event, masked_logits, torch.zeros_like(masked_logits)
        )
        alpha = torch.softmax(masked_logits, dim=0)
        alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
        pair_mass = (alpha * query_mass).sum(dim=0)
        valid_query = valid.any(dim=0)
        query_weight = torch.where(
            valid_query, pair_mass, torch.zeros_like(pair_mass)
        )
        mass_total = query_weight.sum(dim=-1, keepdim=True)
        fallback = valid_query.to(query_weight.dtype)
        query_weight = torch.where(
            mass_total > EPS,
            query_weight,
            fallback,
        )
        weight_total = query_weight.sum(dim=-1).clamp_min(EPS)
        for behavior in ("follow", "invariant"):
            key = f"{family}_{behavior}"
            value = torch.stack([item[key] for item in components], dim=0)
            event_average = (
                alpha * torch.where(valid, value, torch.zeros_like(value))
            ).sum(dim=0)
            score = (query_weight * event_average).sum(dim=-1) / weight_total
            output[key] = score.detach().cpu().numpy()
        output[f"{family}_effective_support"] = (
            valid_query.sum(dim=-1).detach().cpu().numpy()
        )
        output[f"{family}_pair_mass"] = (
            torch.where(valid_query, pair_mass, torch.zeros_like(pair_mass))
            .sum(dim=-1)
            .div(valid_query.sum(dim=-1).clamp_min(1))
            .detach()
            .cpu()
            .numpy()
        )
    output["effective_support"] = np.minimum(
        output["attention_effective_support"],
        output["transport_effective_support"],
    )
    output["attention_mass"] = output["attention_pair_mass"]
    return output


def follow_invariant_scores(
    clean: GraphCapture,
    event: GraphCapture,
    permutation: Any,
) -> dict[str, np.ndarray]:
    """Convenience wrapper for one Appendix A.3 transposition event."""

    return appendix_cosine_group_scores(
        clean,
        [event],
        [permutation],
        temperature=1.0,
    )


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
    if len(values) == 0:
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
        "M2": "realised_transport_pair_cosine",
        "M3": "realised_transport_pair_cosine",
        "M4": "attention_pair_cosine",
        "M5": "attention_pair_cosine",
        "M6": "realised_transport_pair_cosine",
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
                    "protocol_version": PROTOCOL_VERSION,
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
                            "appendix_swap_softmax_and_query_pair_mass"
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
