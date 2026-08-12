"""Semantic and structural donor-swaps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _copy(value: Any) -> Any:
    return value.clone() if hasattr(value, "clone") else np.array(value, copy=True)


def semantic_donor_swap(values: Any, source: int, donor_attributes: Any) -> Any:
    """Replace the source node's semantic attributes with the donor's."""

    source = int(source)
    if not 0 <= source < len(values):
        raise IndexError("source is outside the graph")
    result = _copy(values)
    if hasattr(result, "new_tensor"):
        replacement = result.new_tensor(donor_attributes).reshape_as(result[source])
    else:
        replacement = np.asarray(donor_attributes, dtype=result.dtype).reshape(result[source].shape)
    result[source] = replacement
    return result


def _structural_donor_swap_pair_field(
    pair_values: Any,
    source: int,
    donor: int,
) -> Any:
    """Copy one pairwise structural field from donor to source."""

    source = int(source)
    donor = int(donor)
    if getattr(pair_values, "ndim", 0) < 2 or pair_values.shape[0] != pair_values.shape[1]:
        raise ValueError("pair values must begin with a square [node, node] matrix")
    node_count = int(pair_values.shape[0])
    if not 0 <= source < node_count or not 0 <= donor < node_count:
        raise IndexError("source or donor is outside the graph")
    clean = _copy(pair_values)
    result = _copy(pair_values)
    result[source, :] = clean[donor, :]
    result[:, source] = clean[:, donor]
    result[source, source] = clean[donor, donor]
    return result


def structural_donor_swap(
    node_fields: Mapping[str, Any],
    pair_fields: Mapping[str, Any],
    source: int,
    donor: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy the donor's complete structural profile onto the source."""

    source = int(source)
    donor = int(donor)
    node_count: int | None = None
    result_nodes: dict[str, Any] = {}
    for name, value in node_fields.items():
        if getattr(value, "ndim", 0) < 1:
            raise ValueError(f"node field {name!r} is not node-indexed")
        if node_count is None:
            node_count = int(value.shape[0])
        elif int(value.shape[0]) != node_count:
            raise ValueError("node structural fields disagree about the number of nodes")
        result = _copy(value)
        result[source] = value[donor]
        result_nodes[name] = result

    result_pairs: dict[str, Any] = {}
    for name, value in pair_fields.items():
        if node_count is None:
            node_count = int(value.shape[0])
        if value.shape[0] != node_count or value.shape[1] != node_count:
            raise ValueError(f"pair field {name!r} does not match the node count")
        result_pairs[name] = _structural_donor_swap_pair_field(value, source, donor)

    if node_count is None:
        raise ValueError("at least one structural field is required")
    if not 0 <= source < node_count or not 0 <= donor < node_count:
        raise IndexError("source or donor is outside the graph")
    return result_nodes, result_pairs
