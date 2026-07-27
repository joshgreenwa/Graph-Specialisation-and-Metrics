"""Canonical Functional carriage and positive-is-beneficial Beneficial carriage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..carriage.core import integrated_loss_carriage
from .audit import within_tolerance


FUNCTIONAL_NAME = "Functional carriage"
FUNCTIONAL_SYMBOL = "F_sens"
BENEFICIAL_NAME = "Beneficial carriage"
BENEFICIAL_SIGN = "positive-is-beneficial"


def functional_carriage(delta, clean_gradient):
    """Compute F_sens with eventwise output magnitude before donor averaging.

    Args:
        delta: ``[source, donor, carrier, width]``, clean minus intervention.
        clean_gradient: ``[output, carrier, width]`` in z-space.
    Returns:
        ``[carrier, source]``.
    """

    import torch

    if delta.ndim != 4 or clean_gradient.ndim != 3:
        raise ValueError("delta must be [S,K,N,M] and gradient [T,N,M]")
    if tuple(delta.shape[2:]) != tuple(clean_gradient.shape[1:]):
        raise ValueError("final-state delta and clean gradient geometry differ")
    return functional_carriage_events(delta, clean_gradient).mean(dim=1).t().contiguous()


def functional_carriage_events(delta, clean_gradient):
    """Return donor-resolved magnitudes ``[source,donor,carrier]``."""

    import torch

    if delta.ndim != 4 or clean_gradient.ndim != 3:
        raise ValueError("delta must be [S,K,N,M] and gradient [T,N,M]")
    if tuple(delta.shape[2:]) != tuple(clean_gradient.shape[1:]):
        raise ValueError("final-state delta and clean gradient geometry differ")
    q = torch.einsum("sknm,tnm->sknt", delta, clean_gradient)
    return q.square().sum(dim=-1).sqrt()


def _validated_donor_counts(donor_counts, total: int) -> tuple[int, ...]:
    counts = tuple(int(value) for value in donor_counts)
    if not counts or any(value < 1 for value in counts):
        raise ValueError("every estimable source must have at least one donor")
    if sum(counts) != int(total):
        raise ValueError(
            f"donor counts sum to {sum(counts)}, but {int(total)} events were supplied"
        )
    return counts


def functional_carriage_ragged(delta, clean_gradient, donor_counts):
    """Compute ``F_sens`` for source-major events with source-specific donor counts.

    Args:
        delta: ``[event, carrier, width]``, clean minus intervention.
        clean_gradient: ``[output, carrier, width]`` in z-space.
        donor_counts: number of contiguous events belonging to each source.
    Returns:
        The graph field ``[carrier, source]`` and donor-resolved tensors, one
        ``[donor, carrier]`` tensor per source.
    """

    import torch

    if delta.ndim != 3 or clean_gradient.ndim != 3:
        raise ValueError("delta must be [E,N,M] and gradient [T,N,M]")
    if tuple(delta.shape[1:]) != tuple(clean_gradient.shape[1:]):
        raise ValueError("final-state delta and clean gradient geometry differ")
    counts = _validated_donor_counts(donor_counts, int(delta.shape[0]))
    q = torch.einsum("enm,tnm->ent", delta, clean_gradient)
    events = q.square().sum(dim=-1).sqrt()
    by_source = tuple(torch.split(events, counts, dim=0))
    field = torch.stack(
        [source_events.mean(dim=0) for source_events in by_source],
        dim=1,
    )
    return field, by_source


@dataclass(frozen=True)
class BeneficialResult:
    field: Any
    event_field: Any
    event_loss_increase: Any
    completeness_residual: Any
    quadrature_error: Any
    intervals: Any
    converged: Any


def beneficial_carriage(
    h_clean,
    h_event,
    loss_from_pooled,
    *,
    pooling: str | None = None,
    carrier_weights=None,
    atol: float,
    rtol: float,
    max_intervals: int,
    tolerance: float = 1e-5,
) -> BeneficialResult:
    """Integrate each donor path, negate the old allocation, then donor-average."""

    import torch

    if h_event.ndim != 4:
        raise ValueError("h_event must be [source,donor,carrier,width]")
    sources, donors, carriers, width = h_event.shape
    if tuple(h_clean.shape) != (carriers, width):
        raise ValueError("h_clean does not align with h_event")
    event = h_event.reshape(sources * donors, carriers, width)
    clean = h_clean.unsqueeze(0).expand_as(event)
    path = integrated_loss_carriage(
        clean,
        event,
        loss_from_pooled,
        pooling=pooling,
        carrier_weights=carrier_weights,
        atol=float(atol),
        rtol=float(rtol),
        max_intervals=int(max_intervals),
    )
    # Legacy numerical engine integrates event->clean, so its carrier sum is
    # loss_clean-loss_event.  Canonical B is the negative: positive means clean reduced loss.
    event_b = -path["carriage"].reshape(sources, donors, carriers)
    event_loss_increase = -path["loss_delta"].reshape(sources, donors)
    residual = event_b.sum(dim=-1) - event_loss_increase
    within_tolerance(
        float(residual.abs().max()),
        float(tolerance),
        "carriage.completeness",
        "Beneficial carriage completeness residual",
        context={"sources": int(sources), "donors": int(donors)},
    )
    field = event_b.mean(dim=1).t().contiguous()
    return BeneficialResult(
        field=field,
        event_field=event_b,
        event_loss_increase=event_loss_increase,
        completeness_residual=residual,
        quadrature_error=path["quadrature_error"].reshape(sources, donors),
        intervals=path["intervals"].reshape(sources, donors),
        converged=path["converged"].reshape(sources, donors),
    )


def beneficial_carriage_ragged(
    h_clean,
    h_event,
    donor_counts,
    loss_from_pooled,
    *,
    pooling: str | None = None,
    carrier_weights=None,
    atol: float,
    rtol: float,
    max_intervals: int,
    tolerance: float = 1e-5,
) -> BeneficialResult:
    """Integrate source-major donor paths with source-specific donor counts."""

    import torch

    if h_event.ndim != 3:
        raise ValueError("h_event must be [event,carrier,width]")
    events, carriers, width = h_event.shape
    if tuple(h_clean.shape) != (carriers, width):
        raise ValueError("h_clean does not align with h_event")
    counts = _validated_donor_counts(donor_counts, int(events))
    clean = h_clean.unsqueeze(0).expand_as(h_event)
    path = integrated_loss_carriage(
        clean,
        h_event,
        loss_from_pooled,
        pooling=pooling,
        carrier_weights=carrier_weights,
        atol=float(atol),
        rtol=float(rtol),
        max_intervals=int(max_intervals),
    )
    # The numerical engine integrates event->clean. Canonical B takes its negative so positive
    # carriage means that the clean signal reduced task loss.
    event_b = -path["carriage"]
    event_loss_increase = -path["loss_delta"]
    residual = event_b.sum(dim=-1) - event_loss_increase
    within_tolerance(
        float(residual.abs().max()),
        float(tolerance),
        "carriage.completeness",
        "Beneficial carriage completeness residual",
        context={"sources": len(counts), "events": int(events)},
    )
    def split_events(value):
        return tuple(torch.split(value, counts, dim=0))

    event_by_source = split_events(event_b)
    field = torch.stack(
        [source_events.mean(dim=0) for source_events in event_by_source],
        dim=1,
    )
    return BeneficialResult(
        field=field,
        event_field=event_by_source,
        event_loss_increase=split_events(event_loss_increase),
        completeness_residual=split_events(residual),
        quadrature_error=split_events(path["quadrature_error"]),
        intervals=split_events(path["intervals"]),
        converged=split_events(path["converged"]),
    )


def additive_beneficial_mass(
    field: Any,
    distances: Any,
    *,
    bins: tuple[tuple[int, int], ...],
    far_thresholds: tuple[int, ...],
) -> dict[str, np.ndarray]:
    field = np.asarray(field, dtype=np.float64)
    distance = np.asarray(distances, dtype=np.float64)
    if field.shape != distance.shape:
        raise ValueError("field and distances must align [carrier,source]")
    bin_mass = np.asarray(
        [
            np.sum(field[np.isfinite(distance) & (distance >= lo) & (distance <= hi)])
            for lo, hi in bins
        ]
    )
    far = np.asarray([
        np.sum(field[np.isfinite(distance) & (distance > threshold)])
        for threshold in far_thresholds
    ])
    return {"S_B": bin_mass, "B_far": far}
