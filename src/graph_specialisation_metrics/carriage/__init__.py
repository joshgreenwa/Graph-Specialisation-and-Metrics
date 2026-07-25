"""Internal, checkpoint-compatible GRIT runtime used by the canonical methodology.

This package preserves proven loading, task, metric, and path-integration primitives. It is not
an alternative public methodology; scientific definitions and orchestration live in
``graph_specialisation_metrics.methodology``.
"""

from __future__ import annotations

from .core import (
    aggregate_carriage_curves,
    beneficial_from_carriage,
    carriage_from_states,
    integrated_loss_carriage,
    symlog_linthresh,
)
from .tasks import TASKS, GritTaskSpec, get_task, register

__all__ = [
    "carriage_from_states",
    "beneficial_from_carriage",
    "integrated_loss_carriage",
    "aggregate_carriage_curves",
    "symlog_linthresh",
    "GritTaskSpec",
    "TASKS",
    "get_task",
    "register",
]
