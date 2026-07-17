"""Central, task-agnostic carriage methodology for GRIT-trained models.

Semantic-intervention carriage (dissertation Ch. 3): a single donor-swap primitive yields
functional carriage F(d), beneficial carriage B(d), and B_far(k), for any GRIT checkpoint
described by a ``GritTaskSpec``. Notebooks clone this repo (via the ``dissertation_key``
secret) and call ``carriage.colab.run(task=...)``, so edits here propagate to every notebook.

    from graph_specialisation_metrics.carriage import run, TASKS, GritTaskSpec
    run(task="zinc")

Layout:
    core.py        pure math (carriage, beneficial, aggregation) -- unit-testable, no GRIT
    content.py     ContentAdapter: how to read/write swappable node content per encoder
    tasks.py       GritTaskSpec + TASKS registry (add a GRIT model here)
    env.py         compat patches, deps, GRIT clone, checkpoint discovery
    grit_runner.py the analysis loop + carriage-precondition checks
    figures.py     aggregate curves -> figures + .npz/.json
    colab.py       run(): one-call orchestration; figures collate under one Drive root
"""

from __future__ import annotations

from .core import (
    aggregate_carriage_curves,
    beneficial_from_carriage,
    carriage_from_states,
    symlog_linthresh,
)
from .tasks import TASKS, GritTaskSpec, get_task, register

__all__ = [
    "carriage_from_states",
    "beneficial_from_carriage",
    "aggregate_carriage_curves",
    "symlog_linthresh",
    "GritTaskSpec",
    "TASKS",
    "get_task",
    "register",
    "run",
]


def run(*args, **kwargs):
    """Lazy proxy to carriage.colab.run so `import carriage` stays torch/GRIT-free."""
    from .colab import run as _run
    return _run(*args, **kwargs)
