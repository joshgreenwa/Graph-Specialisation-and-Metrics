"""Canonical semantic and fixed-support structural donor swaps.

The sparse pair implementation is derived from the README's dense entrywise definition.  It does
not copy or relabel support edges and never sums duplicate payloads.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class StructuralAuditError(RuntimeError):
    """A task's declared structural boundary is incomplete or internally inconsistent."""


def semantic_donor_swap(data: Any, source: int, donor_payload: Any, *, adapter=None) -> Any:
    """Replace exactly one task-declared semantic payload."""

    import torch

    source = int(source)
    out = data.clone()
    if adapter is not None:
        adapter.write_donors(
            out.x,
            np.asarray([source], dtype=np.int64),
            np.asarray(donor_payload).reshape(1, -1),
        )
    else:
        value = torch.as_tensor(donor_payload, dtype=out.x.dtype, device=out.x.device)
        if out.x.ndim == 1:
            out.x[source] = value.reshape(-1)[0]
        else:
            out.x[source] = value.reshape_as(out.x[source])
    return out


def _keys(data: Any) -> tuple[str, ...]:
    value = data.keys
    return tuple(value() if callable(value) else value)


def _tensor_equal(left: Any, right: Any, tolerance: float = 0.0) -> bool:
    import torch

    if left is None or right is None:
        return left is right
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return left == right
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if tolerance > 0 and (left.is_floating_point() or left.is_complex()):
        return bool(torch.allclose(left, right, atol=tolerance, rtol=0.0))
    return bool(torch.equal(left, right))


def audit_structural_fields(data: Any, task: Any) -> None:
    """Fail when a PE-like field is present but absent from the task declaration."""

    node = set(task.node_structural_fields)
    pair = {name for pair in task.pair_structural_fields for name in pair}
    fixed = set(task.fixed_support_fields)
    semantic = set(task.semantic_fields)
    controls = set(task.immutable_control_fields)
    extras = set(task.extra_known_fields)
    declared = node | pair | fixed | semantic | controls | extras
    structural_tokens = (
        "rrwp",
        "rwse",
        "pestat",
        "posenc",
        "pos_enc",
        "lap",
        "eigen",
        "eig",
        "abs_pe",
    )
    for name in _keys(data):
        lowered = str(name).lower()
        pe_like = (
            any(token in lowered for token in structural_tokens)
            or lowered in {"deg", "degree", "log_deg", "pe", "pos"}
        )
        if pe_like and name not in declared:
            raise StructuralAuditError(
                f"PE/structural field {name!r} is present but not registered for task "
                f"{task.name!r}"
            )
    for index_name, value_name in task.pair_structural_fields:
        index = getattr(data, index_name, None)
        value = getattr(data, value_name, None)
        if (index is None) != (value is None):
            raise StructuralAuditError(
                f"pair field {index_name!r}/{value_name!r} must be present or absent together"
            )
    if "edge_index" not in fixed:
        raise StructuralAuditError("edge_index must be registered as fixed architectural support")


def coalesce_equal_sparse(
    index: Any,
    value: Any,
    *,
    num_nodes: int,
    tolerance: float,
) -> tuple[Any, Any]:
    """Sort COO entries and retain one duplicate only when payloads agree."""

    import torch

    if index.ndim != 2 or index.shape[0] != 2:
        raise StructuralAuditError(f"sparse pair index must be [2,E], got {tuple(index.shape)}")
    if value.shape[0] != index.shape[1]:
        raise StructuralAuditError("pair index and payload length disagree")
    if index.numel() == 0:
        return index.clone(), value.clone()
    if bool((index < 0).any()) or bool((index >= int(num_nodes)).any()):
        raise StructuralAuditError("sparse pair index is outside the graph's node range")

    linear = index[0].to(torch.long) * int(num_nodes) + index[1].to(torch.long)
    order = torch.argsort(linear, stable=True)
    sorted_linear = linear[order]
    sorted_value = value[order]
    unique_linear, counts = torch.unique_consecutive(sorted_linear, return_counts=True)
    kept: list[int] = []
    cursor = 0
    for count_tensor in counts:
        count = int(count_tensor)
        reference = sorted_value[cursor]
        if count > 1:
            duplicates = sorted_value[cursor : cursor + count]
            if duplicates.is_floating_point() or duplicates.is_complex():
                agreement = torch.allclose(
                    duplicates,
                    reference.expand_as(duplicates),
                    atol=float(tolerance),
                    rtol=0.0,
                )
            else:
                agreement = torch.equal(duplicates, reference.expand_as(duplicates))
            if not agreement:
                coordinate = divmod(int(sorted_linear[cursor]), int(num_nodes))
                raise StructuralAuditError(
                    f"conflicting duplicate pair payloads at coordinate {coordinate}"
                )
        kept.append(cursor)
        cursor += count
    kept_tensor = torch.as_tensor(kept, device=value.device, dtype=torch.long)
    unique_linear = unique_linear.to(index.device)
    new_index = torch.stack(
        (unique_linear // int(num_nodes), unique_linear % int(num_nodes)), dim=0
    ).to(index.dtype)
    return new_index, sorted_value[kept_tensor].clone()


def sparse_to_mapping(index: Any, value: Any) -> dict[tuple[int, int], Any]:
    return {
        (int(index[0, position]), int(index[1, position])): value[position].clone()
        for position in range(int(index.shape[1]))
    }


def _mapping_to_sparse(
    mapping: dict[tuple[int, int], Any],
    *,
    reference_index: Any,
    reference_value: Any,
) -> tuple[Any, Any]:
    import torch

    coordinates = sorted(mapping)
    if not coordinates:
        shape = (0,) + tuple(reference_value.shape[1:])
        return (
            reference_index.new_empty((2, 0)),
            reference_value.new_empty(shape),
        )
    index = torch.as_tensor(
        coordinates, dtype=reference_index.dtype, device=reference_index.device
    ).t().contiguous()
    value = torch.stack([mapping[key] for key in coordinates]).to(reference_value.device)
    return index, value


def copy_sparse_pair_footprint(
    index: Any,
    value: Any,
    source: int,
    donor: int,
    *,
    num_nodes: int,
    tolerance: float,
) -> tuple[Any, Any]:
    """Implement the dense row/column/self rule exactly in sparse COO form."""

    source, donor = int(source), int(donor)
    clean_index, clean_value = coalesce_equal_sparse(
        index, value, num_nodes=num_nodes, tolerance=tolerance
    )
    clean = sparse_to_mapping(clean_index, clean_value)
    result = {
        coordinate: payload.clone()
        for coordinate, payload in clean.items()
        if source not in coordinate
    }

    # r'_{s,j}=r_{v,j} for j != s
    for other in range(int(num_nodes)):
        if other == source:
            continue
        payload = clean.get((donor, other))
        if payload is not None:
            result[(source, other)] = payload.clone()

    # r'_{i,s}=r_{i,v} for i != s
    for other in range(int(num_nodes)):
        if other == source:
            continue
        payload = clean.get((other, donor))
        if payload is not None:
            result[(other, source)] = payload.clone()

    # r'_{s,s}=r_{v,v}; absence represents the same implicit zero as dense storage.
    self_payload = clean.get((donor, donor))
    if self_payload is not None:
        result[(source, source)] = self_payload.clone()

    new_index, new_value = _mapping_to_sparse(
        result, reference_index=index, reference_value=value
    )
    return coalesce_equal_sparse(
        new_index, new_value, num_nodes=num_nodes, tolerance=tolerance
    )


def dense_pair_donor_swap(value: Any, source: int, donor: int) -> Any:
    """Reference implementation of the normative dense pair definition."""

    source, donor = int(source), int(donor)
    out = value.clone()
    clean = value.clone()
    out[source, :] = clean[donor, :]
    out[:, source] = clean[:, donor]
    out[source, source] = clean[donor, donor]
    return out


def structural_donor_swap(
    data: Any,
    source: int,
    donor: int,
    *,
    task: Any,
    duplicate_tolerance: float,
) -> Any:
    """Copy one donor's complete PE/RRWP footprint onto one source on fixed support."""

    source, donor = int(source), int(donor)
    if source < 0 or donor < 0 or source >= int(data.num_nodes) or donor >= int(data.num_nodes):
        raise IndexError("source/donor is outside the graph")
    audit_structural_fields(data, task)
    out = data.clone()
    if source == donor:
        return out

    for name in task.node_structural_fields:
        tensor = getattr(data, name, None)
        if tensor is not None:
            if int(tensor.shape[0]) != int(data.num_nodes):
                raise StructuralAuditError(
                    f"node structural field {name!r} is not node-indexed: {tuple(tensor.shape)}"
                )
            replacement = tensor.clone()
            replacement[source] = tensor[donor]
            setattr(out, name, replacement)

    for index_name, value_name in task.pair_structural_fields:
        index = getattr(data, index_name, None)
        value = getattr(data, value_name, None)
        if index is not None:
            new_index, new_value = copy_sparse_pair_footprint(
                index,
                value,
                source,
                donor,
                num_nodes=int(data.num_nodes),
                tolerance=float(duplicate_tolerance),
            )
            setattr(out, index_name, new_index)
            setattr(out, value_name, new_value)

    verify_structural_swap(
        data,
        out,
        source,
        donor,
        task=task,
        duplicate_tolerance=duplicate_tolerance,
    )
    return out


def _dense_from_sparse(index: Any, value: Any, n: int) -> tuple[Any, Any]:
    import torch

    width = tuple(value.shape[1:])
    dense = value.new_zeros((int(n), int(n)) + width)
    present = torch.zeros((int(n), int(n)), dtype=torch.bool, device=value.device)
    if index.numel():
        dense[index[0].long(), index[1].long()] = value
        present[index[0].long(), index[1].long()] = True
    return dense, present


def verify_structural_swap(
    base: Any,
    event: Any,
    source: int,
    donor: int,
    *,
    task: Any,
    duplicate_tolerance: float,
) -> None:
    """Verify field boundaries plus the dense row/column/self relation."""

    import torch

    source, donor = int(source), int(donor)
    for name in task.semantic_fields + task.immutable_control_fields + task.fixed_support_fields:
        if hasattr(base, name) and not _tensor_equal(getattr(base, name), getattr(event, name)):
            raise StructuralAuditError(f"structural swap changed fixed field {name!r}")

    for name in task.node_structural_fields:
        clean = getattr(base, name, None)
        changed = getattr(event, name, None)
        if clean is None:
            continue
        expected = clean.clone()
        expected[source] = clean[donor]
        if not _tensor_equal(expected, changed, duplicate_tolerance):
            raise StructuralAuditError(f"node structural field {name!r} was copied incorrectly")

    for index_name, value_name in task.pair_structural_fields:
        base_index = getattr(base, index_name, None)
        if base_index is None:
            continue
        base_index, base_value = coalesce_equal_sparse(
            base_index,
            getattr(base, value_name),
            num_nodes=int(base.num_nodes),
            tolerance=duplicate_tolerance,
        )
        event_index, event_value = coalesce_equal_sparse(
            getattr(event, index_name),
            getattr(event, value_name),
            num_nodes=int(base.num_nodes),
            tolerance=duplicate_tolerance,
        )
        clean_dense, clean_present = _dense_from_sparse(
            base_index, base_value, int(base.num_nodes)
        )
        event_dense, event_present = _dense_from_sparse(
            event_index, event_value, int(base.num_nodes)
        )
        expected_dense = dense_pair_donor_swap(clean_dense, source, donor)
        expected_present = dense_pair_donor_swap(clean_present, source, donor)
        if not torch.equal(event_present, expected_present):
            raise StructuralAuditError(f"sparse support for {index_name!r} violates dense copy")
        if not torch.allclose(
            event_dense, expected_dense, atol=float(duplicate_tolerance), rtol=0.0
        ):
            raise StructuralAuditError(f"payload for {value_name!r} violates dense copy")


def verify_semantic_swap(
    base: Any,
    event: Any,
    source: int,
    donor_payload: Any,
    *,
    task: Any,
) -> None:
    """Verify a semantic event changes only the declared source payload."""

    import torch

    source = int(source)
    expected = base.x.clone()
    value = torch.as_tensor(donor_payload, dtype=expected.dtype, device=expected.device)
    if expected.ndim == 1:
        expected[source] = value.reshape(-1)[0]
    else:
        expected[source] = value.reshape_as(expected[source])
    if not torch.equal(event.x, expected):
        raise RuntimeError("semantic donor swap did not replace exactly the source x row")
    for name in task.node_structural_fields + tuple(
        field for pair in task.pair_structural_fields for field in pair
    ) + task.fixed_support_fields + task.immutable_control_fields:
        if hasattr(base, name) and not _tensor_equal(getattr(base, name), getattr(event, name)):
            raise RuntimeError(f"semantic donor swap changed fixed field {name!r}")


def structural_footprints(data: Any, task: Any, *, tolerance: float) -> tuple[bytes, ...]:
    """Canonical fingerprints used to exclude identical structural donors."""

    audit_structural_fields(data, task)
    n = int(data.num_nodes)
    pieces: list[list[bytes]] = [[] for _ in range(n)]
    for name in task.node_structural_fields:
        value = getattr(data, name, None)
        if value is not None:
            array = value.detach().cpu().numpy()
            for node in range(n):
                pieces[node].append(np.ascontiguousarray(array[node]).tobytes())
    for index_name, value_name in task.pair_structural_fields:
        index = getattr(data, index_name, None)
        if index is None:
            continue
        index, value = coalesce_equal_sparse(
            index,
            getattr(data, value_name),
            num_nodes=n,
            tolerance=tolerance,
        )
        dense, present = _dense_from_sparse(index, value, n)
        dense_np = dense.detach().cpu().numpy()
        present_np = present.detach().cpu().numpy()
        for node in range(n):
            # Explicit presence mask distinguishes an absent sparse zero from a stored zero.
            pieces[node].extend(
                (
                    np.ascontiguousarray(dense_np[node, :]).tobytes(),
                    np.ascontiguousarray(dense_np[:, node]).tobytes(),
                    np.ascontiguousarray(present_np[node, :]).tobytes(),
                    np.ascontiguousarray(present_np[:, node]).tobytes(),
                )
            )
    return tuple(b"\x1f".join(part) for part in pieces)


def structural_intervention_dose(
    base: Any,
    event: Any,
    task: Any,
    *,
    tolerance: float,
) -> float:
    """RMS PE/RRWP payload change used only to match causal controls."""

    components: list[np.ndarray] = []
    for name in task.node_structural_fields:
        left = getattr(base, name, None)
        right = getattr(event, name, None)
        if left is not None:
            components.append(
                (left.detach().float() - right.detach().float()).cpu().numpy().reshape(-1)
            )
    for index_name, value_name in task.pair_structural_fields:
        left_index = getattr(base, index_name, None)
        if left_index is None:
            continue
        left_index, left_value = coalesce_equal_sparse(
            left_index,
            getattr(base, value_name),
            num_nodes=int(base.num_nodes),
            tolerance=tolerance,
        )
        right_index, right_value = coalesce_equal_sparse(
            getattr(event, index_name),
            getattr(event, value_name),
            num_nodes=int(base.num_nodes),
            tolerance=tolerance,
        )
        left_dense, left_present = _dense_from_sparse(
            left_index, left_value, int(base.num_nodes)
        )
        right_dense, right_present = _dense_from_sparse(
            right_index, right_value, int(base.num_nodes)
        )
        components.append(
            (left_dense.detach().float() - right_dense.detach().float())
            .cpu()
            .numpy()
            .reshape(-1)
        )
        components.append(
            (left_present.to(dtype=left_dense.dtype) - right_present.to(dtype=left_dense.dtype))
            .cpu()
            .numpy()
            .reshape(-1)
        )
    if not components:
        return 0.0
    joined = np.concatenate(components).astype(np.float64)
    return float(np.sqrt(np.mean(np.square(joined))))
