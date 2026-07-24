"""Task-independent semantic, PE, and matched-topology interventions."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from ..carriage import structural


SEMANTIC_VARIANTS = ("semantic_single_donor", "semantic_transposition")
PE_VARIANTS = ("pe_single_donor", "pe_transposition")
TOPOLOGY_VARIANT = "topology_matched_donor"


def semantic_single_donor(data: Any, source: int, donor_row: Any) -> Any:
    """Replace one complete ``x`` row and no other model input."""

    import torch

    out = data.clone()
    value = torch.as_tensor(donor_row, device=out.x.device, dtype=out.x.dtype)
    if out.x.dim() == 1:
        out.x[int(source)] = value.reshape(-1)[0]
    else:
        out.x[int(source)] = value.reshape_as(out.x[int(source)])
    return out


def semantic_transposition(data: Any, source: int, partner: int) -> Any:
    """Exchange two complete semantic rows while holding every PE field fixed."""

    out = data.clone()
    out.x = structural._swap_rows(data.x, int(source), int(partner))
    return out


def pe_intervention(data: Any, source: int, partner: int, *, mode: str) -> Any:
    """Apply a mask-frozen PE copy/transposition.

    ``structural.perturb`` operates on every topology-derived field.  The scoring
    experiment then restores molecular bonds and architectural support, making this
    specifically a PE payload intervention.
    """

    if mode not in {"single_node", "transposition"}:
        raise ValueError("PE mode must be 'single_node' or 'transposition'")
    if int(source) == int(partner):
        return data.clone()
    out = structural.perturb(data, int(source), int(partner), mode)
    out.edge_index = data.edge_index
    if getattr(data, "edge_attr", None) is not None:
        out.edge_attr = data.edge_attr
    if getattr(data, "rrwp_local_edge_index", None) is not None:
        out.rrwp_local_edge_index = data.rrwp_local_edge_index
    return out


def pe_single_donor(data: Any, source: int, partner: int) -> Any:
    return pe_intervention(data, source, partner, mode="single_node")


def pe_transposition(data: Any, source: int, partner: int) -> Any:
    return pe_intervention(data, source, partner, mode="transposition")


def _rows(data: Any) -> np.ndarray:
    value = data.x.detach().cpu().numpy()
    if value.ndim == 1:
        value = value[:, None]
    return value.astype(np.int64, copy=False)


def _degrees(data: Any) -> np.ndarray:
    return structural.node_degrees(data.edge_index.detach().cpu(), int(data.num_nodes))


def _undirected_edges(data: Any) -> np.ndarray:
    edge_index = data.edge_index.detach().cpu().numpy().astype(np.int64)
    if edge_index.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.sort(edge_index.T, axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0)


def graph_descriptor(data: Any, graph_id: int) -> dict[str, Any]:
    """Compact descriptor used for topology matching and event manifests."""

    labels = _rows(data)
    degree = _degrees(data)
    edges = _undirected_edges(data)
    label_counts = Counter(tuple(row.tolist()) for row in labels)
    return {
        "graph_id": int(graph_id),
        "n": int(data.num_nodes),
        "labels": labels.tolist(),
        "degree": degree.tolist(),
        "edges": edges.tolist(),
        "edge_count": int(len(edges)),
        "label_histogram": sorted(
            (list(key), int(value)) for key, value in label_counts.items()
        ),
    }


def _nonisomorphic(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Cheap exact-enough rejection followed by NetworkX isomorphism when available."""

    if int(left["n"]) != int(right["n"]):
        return True
    if int(left["edge_count"]) != int(right["edge_count"]):
        return True
    if sorted(left["degree"]) != sorted(right["degree"]):
        return True
    try:
        import networkx as nx

        g_left = nx.Graph()
        g_left.add_nodes_from(range(int(left["n"])))
        g_left.add_edges_from(left["edges"])
        g_right = nx.Graph()
        g_right.add_nodes_from(range(int(right["n"])))
        g_right.add_edges_from(right["edges"])
        return not nx.is_isomorphic(g_left, g_right)
    except ImportError:
        return sorted(map(tuple, left["edges"])) != sorted(map(tuple, right["edges"]))


def align_donor_nodes(
    base: Mapping[str, Any],
    donor: Mapping[str, Any],
) -> np.ndarray:
    """Return donor node index for every base node using a fixed nuisance cost."""

    from scipy.optimize import linear_sum_assignment

    base_labels = np.asarray(base["labels"], dtype=np.int64)
    donor_labels = np.asarray(donor["labels"], dtype=np.int64)
    base_degree = np.asarray(base["degree"], dtype=float)
    donor_degree = np.asarray(donor["degree"], dtype=float)
    content_mismatch = np.any(
        base_labels[:, None, :] != donor_labels[None, :, :], axis=-1
    ).astype(float)
    degree_gap = np.abs(base_degree[:, None] - donor_degree[None, :])
    # Content is a nuisance matching variable, not copied into the intervention.
    cost = 1_000.0 * content_mismatch + degree_gap
    base_index, donor_index = linear_sum_assignment(cost)
    order = np.empty(len(base_index), dtype=np.int64)
    order[base_index] = donor_index
    return order


def topology_donor_variant(data: Any, donor: Any, donor_for_base: Sequence[int]) -> Any:
    """Copy a donor topology/RRWP into aligned base coordinates, retaining base ``x/y``."""

    import torch

    alignment = np.asarray(donor_for_base, dtype=np.int64)
    n = int(data.num_nodes)
    if int(donor.num_nodes) != n or alignment.shape != (n,):
        raise ValueError("topology donor and base must have the same node count")
    if sorted(alignment.tolist()) != list(range(n)):
        raise ValueError("donor_for_base must be a permutation")
    out = data.clone()
    donor_idx = torch.as_tensor(alignment, dtype=torch.long, device=donor.x.device)
    for name in structural.NODE_STRUCT_ATTRS:
        value = getattr(donor, name, None)
        reference = getattr(data, name, None)
        if value is not None and reference is not None:
            setattr(out, name, value[donor_idx].clone().to(reference.device))

    base_for_donor = np.empty_like(alignment)
    base_for_donor[alignment] = np.arange(n, dtype=np.int64)
    for index_name, value_name in structural.PAIR_TENSORS:
        donor_index = getattr(donor, index_name, None)
        if donor_index is None:
            continue
        mapping = torch.as_tensor(
            base_for_donor, dtype=torch.long, device=donor_index.device
        )
        mapped = mapping[donor_index.long()]
        reference_index = getattr(data, index_name, None)
        if reference_index is not None:
            mapped = mapped.to(reference_index.device)
        setattr(out, index_name, mapped.clone())
        if value_name is not None:
            donor_value = getattr(donor, value_name, None)
            if donor_value is not None:
                reference_value = getattr(data, value_name, None)
                if reference_value is not None:
                    donor_value = donor_value.to(reference_value.device)
                setattr(out, value_name, donor_value.clone())
    out.x = data.x.clone()
    if getattr(data, "y", None) is not None:
        out.y = data.y.clone()
    return out


def pe_payload_dose(base: Any, variant: Any) -> float:
    """RMS change over comparable RRWP node and dense pair payloads."""

    components: list[np.ndarray] = []
    for field in structural.NODE_STRUCT_ATTRS:
        left = getattr(base, field, None)
        right = getattr(variant, field, None)
        if left is not None and right is not None and tuple(left.shape) == tuple(right.shape):
            components.append(
                (left.detach().float() - right.detach().float()).cpu().numpy().reshape(-1)
            )
    left_index = getattr(base, "rrwp_index", None)
    left_value = getattr(base, "rrwp_val", None)
    right_index = getattr(variant, "rrwp_index", None)
    right_value = getattr(variant, "rrwp_val", None)
    if all(item is not None for item in (left_index, left_value, right_index, right_value)):
        n = int(base.num_nodes)
        width = int(left_value.reshape(left_value.shape[0], -1).shape[1])
        left_dense = np.zeros((n, n, width), dtype=np.float32)
        right_dense = np.zeros_like(left_dense)
        li = left_index.detach().cpu().numpy().astype(np.int64)
        ri = right_index.detach().cpu().numpy().astype(np.int64)
        left_dense[li[0], li[1]] = left_value.detach().cpu().numpy().reshape(-1, width)
        right_dense[ri[0], ri[1]] = right_value.detach().cpu().numpy().reshape(-1, width)
        components.append((left_dense - right_dense).reshape(-1))
    if not components:
        return 0.0
    joined = np.concatenate(components).astype(float)
    return float(np.sqrt(np.mean(np.square(joined))))


def intervention_dose(base: Any, variant: Any, channel: str) -> float:
    if channel == "semantic":
        return float(np.linalg.norm(_rows(base).astype(float) - _rows(variant).astype(float)))
    if channel == "pe":
        return pe_payload_dose(base, variant)
    if channel == "topology":
        left = set(map(tuple, _undirected_edges(base)))
        right = set(map(tuple, _undirected_edges(variant)))
        return float(len(left.symmetric_difference(right)))
    raise ValueError(channel)


def _sample_graph_ids(length: int, maximum: int, seed: int) -> np.ndarray:
    count = min(int(length), int(maximum))
    return np.sort(
        np.random.default_rng(int(seed)).choice(int(length), size=count, replace=False)
    ).astype(np.int64)


def build_donor_index(
    donor_ds: Any,
    *,
    semantic_pool: int,
    topology_pool: int,
    seed: int,
) -> dict[str, Any]:
    """Build bounded reusable semantic-row and topology descriptor indices."""

    semantic_graph_ids = _sample_graph_ids(len(donor_ds), semantic_pool, seed + 11)
    semantic_nodes: list[dict[str, Any]] = []
    for graph_id in semantic_graph_ids:
        donor_graph = donor_ds[int(graph_id)]
        rows = _rows(donor_graph)
        degree = _degrees(donor_graph)
        semantic_nodes.extend(
            {
                "graph_id": int(graph_id),
                "node": int(node),
                "row": row.tolist(),
                "degree": int(degree[node]),
            }
            for node, row in enumerate(rows)
        )

    topology_graph_ids = _sample_graph_ids(len(donor_ds), topology_pool, seed + 23)
    descriptors = [
        graph_descriptor(donor_ds[int(graph_id)], int(graph_id))
        for graph_id in topology_graph_ids
    ]
    by_n: dict[int, list[int]] = {}
    for position, descriptor in enumerate(descriptors):
        by_n.setdefault(int(descriptor["n"]), []).append(position)
    return {
        "semantic_nodes": semantic_nodes,
        "topology_descriptors": descriptors,
        "topology_by_n": by_n,
    }


def _sample_partner(
    base: Any,
    source: int,
    rng: np.random.Generator,
    *,
    require_different_content: bool,
    match: str,
) -> int:
    n = int(base.num_nodes)
    if n < 2:
        return int(source)
    degree = _degrees(base)
    rows = _rows(base)
    candidates = np.arange(n)
    candidates = candidates[candidates != int(source)]
    if require_different_content:
        different = np.any(rows[candidates] != rows[int(source)], axis=1)
        if bool(different.any()):
            candidates = candidates[different]
    if match == "degree":
        exact = candidates[degree[candidates] == degree[int(source)]]
        if exact.size:
            candidates = exact
        elif candidates.size:
            gap = np.abs(degree[candidates] - degree[int(source)])
            candidates = candidates[gap == gap.min()]
    if not candidates.size:
        return int(source)
    return int(rng.choice(candidates))


def _topology_events(
    base_descriptor: Mapping[str, Any],
    donor_index: Mapping[str, Any],
    *,
    count: int,
    allow_relaxed: bool,
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, float, int, Mapping[str, Any]]] = []
    for position in donor_index["topology_by_n"].get(int(base_descriptor["n"]), []):
        donor = donor_index["topology_descriptors"][int(position)]
        if not _nonisomorphic(base_descriptor, donor):
            continue
        exact_content = donor["label_histogram"] == base_descriptor["label_histogram"]
        if not exact_content and not allow_relaxed:
            continue
        tier = 0 if exact_content else 1
        cost = abs(int(donor["edge_count"]) - int(base_descriptor["edge_count"]))
        cost += float(
            np.mean(
                np.abs(
                    np.sort(np.asarray(donor["degree"], dtype=float))
                    - np.sort(np.asarray(base_descriptor["degree"], dtype=float))
                )
            )
        )
        candidates.append((tier, cost, int(donor["graph_id"]), donor))
    candidates.sort(key=lambda item: item[:3])
    events: list[dict[str, Any]] = []
    for tier, cost, graph_id, donor in candidates[: int(count)]:
        alignment = align_donor_nodes(base_descriptor, donor)
        events.append(
            {
                "variant": TOPOLOGY_VARIANT,
                "donor_graph_id": graph_id,
                "alignment": alignment.tolist(),
                "tier": int(tier),
                "match_cost": float(cost),
            }
        )
    return events


def build_graph_manifest(
    gm: Any,
    graph_id: int,
    donor_index: Mapping[str, Any],
    *,
    sources: int,
    events_per_source: int,
    topology_events: int,
    seed: int,
    partner_match: str,
    allow_relaxed_topology: bool,
) -> dict[str, Any]:
    """Create one deterministic shared event plan for all scoring methods."""

    base = gm.eval_ds[int(graph_id)]
    rng = np.random.default_rng(int(seed) + 1_000_003 * int(graph_id))
    n = int(base.num_nodes)
    source_count = min(int(sources), n)
    source_nodes = np.sort(rng.choice(n, size=source_count, replace=False)).astype(np.int64)
    base_rows = _rows(base)
    base_degree = _degrees(base)
    semantic_pool = donor_index["semantic_nodes"]
    source_groups = []
    for source in source_nodes:
        different = [
            item for item in semantic_pool
            if item["row"] != base_rows[int(source)].tolist()
        ]
        choices = different or semantic_pool
        if choices:
            exact_degree = [
                item for item in choices
                if int(item["degree"]) == int(base_degree[int(source)])
            ]
            if exact_degree:
                choices = exact_degree
            else:
                degree_gap = np.asarray(
                    [
                        abs(int(item["degree"]) - int(base_degree[int(source)]))
                        for item in choices
                    ],
                    dtype=np.int64,
                )
                minimum_gap = int(np.min(degree_gap))
                choices = [
                    item for item, gap in zip(choices, degree_gap)
                    if int(gap) == minimum_gap
                ]
        if choices:
            chosen_positions = rng.choice(
                len(choices),
                size=int(events_per_source),
                replace=len(choices) < int(events_per_source),
            )
            donor_events = [
                {
                    "variant": "semantic_single_donor",
                    "source": int(source),
                    "donor_graph_id": int(choices[int(pos)]["graph_id"]),
                    "donor_node": int(choices[int(pos)]["node"]),
                    "donor_row": list(choices[int(pos)]["row"]),
                    "donor_degree": int(choices[int(pos)]["degree"]),
                    "degree_gap": abs(
                        int(choices[int(pos)]["degree"])
                        - int(base_degree[int(source)])
                    ),
                }
                for pos in chosen_positions
            ]
        else:
            donor_events = []
        semantic_partners = [
            _sample_partner(
                base,
                int(source),
                rng,
                require_different_content=True,
                match=partner_match,
            )
            for _ in range(int(events_per_source))
        ]
        pe_partners = [
            _sample_partner(
                base,
                int(source),
                rng,
                require_different_content=False,
                match=partner_match,
            )
            for _ in range(int(events_per_source))
        ]
        source_groups.append(
            {
                "source": int(source),
                "semantic_single_donor": donor_events,
                "semantic_transposition": [
                    {
                        "variant": "semantic_transposition",
                        "source": int(source),
                        "partner": int(partner),
                        "degree_gap": abs(
                            int(base_degree[int(source)]) - int(base_degree[int(partner)])
                        ),
                    }
                    for partner in semantic_partners
                ],
                "pe_single_donor": [
                    {
                        "variant": "pe_single_donor",
                        "source": int(source),
                        "partner": int(partner),
                        "degree_gap": abs(
                            int(base_degree[int(source)]) - int(base_degree[int(partner)])
                        ),
                    }
                    for partner in pe_partners
                ],
                "pe_transposition": [
                    {
                        "variant": "pe_transposition",
                        "source": int(source),
                        "partner": int(partner),
                        "degree_gap": abs(
                            int(base_degree[int(source)]) - int(base_degree[int(partner)])
                        ),
                    }
                    for partner in pe_partners
                ],
            }
        )
    descriptor = graph_descriptor(base, int(graph_id))
    return {
        "graph_id": int(graph_id),
        "num_nodes": n,
        "sources": source_groups,
        "topology": _topology_events(
            descriptor,
            donor_index,
            count=topology_events,
            allow_relaxed=allow_relaxed_topology,
        ),
    }


def apply_event(base: Any, gm: Any, event: Mapping[str, Any]) -> Any:
    """Materialize any event from :func:`build_graph_manifest`."""

    variant = str(event["variant"])
    if variant == "semantic_single_donor":
        return semantic_single_donor(base, int(event["source"]), event["donor_row"])
    if variant == "semantic_transposition":
        return semantic_transposition(base, int(event["source"]), int(event["partner"]))
    if variant == "pe_single_donor":
        return pe_single_donor(base, int(event["source"]), int(event["partner"]))
    if variant == "pe_transposition":
        return pe_transposition(base, int(event["source"]), int(event["partner"]))
    if variant == TOPOLOGY_VARIANT:
        donor = gm.donor_ds[int(event["donor_graph_id"])]
        return topology_donor_variant(base, donor, event["alignment"])
    raise KeyError(f"unknown intervention variant {variant!r}")


def audit_preservation(base: Any, variant: Any, event: Mapping[str, Any]) -> None:
    """Fail loudly if an intervention crosses its declared channel boundary."""

    import torch

    name = str(event["variant"])
    if getattr(base, "y", None) is not None:
        assert torch.equal(base.y, variant.y), f"{name} changed target y"
    if name.startswith("semantic_"):
        for field in (*structural.NODE_STRUCT_ATTRS, "edge_index", "edge_attr", "rrwp_index",
                      "rrwp_val", "rrwp_local_edge_index"):
            left, right = getattr(base, field, None), getattr(variant, field, None)
            if left is not None:
                assert right is not None and torch.equal(left, right), \
                    f"{name} changed fixed field {field}"
    elif name.startswith("pe_"):
        assert torch.equal(base.x, variant.x), f"{name} changed semantic x"
        assert torch.equal(base.edge_index, variant.edge_index), f"{name} changed support"
        if getattr(base, "edge_attr", None) is not None:
            assert torch.equal(base.edge_attr, variant.edge_attr), f"{name} changed bonds"
    elif name == TOPOLOGY_VARIANT:
        assert torch.equal(base.x, variant.x), "topology donor changed semantic x"
