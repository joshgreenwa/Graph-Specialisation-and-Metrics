"""Central, task-agnostic carriage methodology for GRIT-trained models.

Semantic-intervention carriage (dissertation Ch. 3): a single donor-swap primitive yields
functional carriage F(d), beneficial carriage B(d), and B_far(k), for any GRIT checkpoint
described by a ``GritTaskSpec``. Notebooks clone this repo (via the ``dissertation_key``
secret) and call ``carriage.colab.run(task=...)``, so edits here propagate to every notebook.

    from graph_specialisation_metrics.carriage import run, TASKS, GritTaskSpec
    run(task="zinc")

Supports scalar regression (ZINC), multi-target regression, and multilabel classification
(Peptides-func) uniformly: functional carriage = magnitude of the output movement over the
T outputs; beneficial carriage = the task-loss change attributed to carriers either by a
signed final-state path integral (finite intervention) or by retained comparison estimators.

Layout:
    core.py        pure math (carriage, functional magnitude, beneficial attribution,
                   aggregation) -- unit-testable, no GRIT
    content.py     ContentAdapter: whole-row node-content swap (TypeDictNode and OGB Atom)
    metrics.py     per-graph task loss (l1/mse/BCE) and dataset metrics (MAE, multilabel AP)
    tasks.py       GritTaskSpec + TASKS registry (zinc, peptides_func); add a model here
    env.py         compat patches, deps, GRIT clone, checkpoint discovery
    peptides_env.py env hooks reusing the training code's RDKit + dataset/RRWP patches
    grit_runner.py the analysis loop + carriage-precondition checks
    figures.py     aggregate curves -> figures + .npz/.json
    colab.py       run(): one-call orchestration; figures collate under one Drive root
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
    "run",
]


def run(*args, **kwargs):
    """Lazy proxy to carriage.colab.run so `import carriage` stays torch/GRIT-free."""
    from .colab import run as _run
    return _run(*args, **kwargs)
