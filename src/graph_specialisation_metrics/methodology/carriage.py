"""Canonical Functional carriage and positive-is-beneficial Beneficial carriage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..carriage.core import integrated_loss_carriage


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
    pooling: str,
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
        atol=float(atol),
        rtol=float(rtol),
        max_intervals=int(max_intervals),
    )
    # Legacy numerical engine integrates event->clean, so its carrier sum is
    # loss_clean-loss_event.  Canonical B is the negative: positive means clean reduced loss.
    event_b = -path["carriage"].reshape(sources, donors, carriers)
    event_loss_increase = -path["loss_delta"].reshape(sources, donors)
    residual = event_b.sum(dim=-1) - event_loss_increase
    if not torch.allclose(
        event_b.sum(dim=-1), event_loss_increase, atol=float(tolerance), rtol=0.0
    ):
        worst = float(residual.abs().max())
        raise RuntimeError(f"Beneficial carriage completeness failed (max residual {worst:.3e})")
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
