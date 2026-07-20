"""Cross-model comparison of ZINC GRIT variants (dense, k-hop, VNode).

Evaluate (val/test), compute functional/beneficial semantic + structural carriage and per-head
specialisation scores for every registered model, CACHE all of it to Drive, and build the
comparison deliverables (scatter grid + standardised and overlaid carriage plots) with
drop/include method selection. Figures rebuild from cache with no re-run.

    from graph_specialisation_metrics.comparison import run_all, build_figures
    run_all()                         # compute (cache-skipping) + all figures
    build_figures(drop_vnode=True)    # restyle/subset from cache, no compute
"""

from __future__ import annotations

from . import data, plots
from .data import DEFAULT_TASKS, METHOD_META, is_vnode, select_methods

__all__ = [
    "run_all",
    "build_figures",
    "performance_table",
    "inventory",
    "data",
    "plots",
    "DEFAULT_TASKS",
    "METHOD_META",
    "is_vnode",
    "select_methods",
]


def run_all(*args, **kwargs):
    """Lazy proxy to comparison.run.run_all (keeps `import comparison` torch/GRIT-free)."""
    from .run import run_all as _run
    return _run(*args, **kwargs)


def inventory(*args, **kwargs):
    """Lazy proxy to comparison.run.inventory (per-model cache/checkpoint diagnostic)."""
    from .run import inventory as _inv
    return _inv(*args, **kwargs)


def build_figures(*args, **kwargs):
    """Lazy proxy to comparison.run.build_figures (pure cache -> figures, no torch needed)."""
    from .run import build_figures as _bf
    return _bf(*args, **kwargs)


def performance_table(*args, **kwargs):
    from .run import performance_table as _pt
    return _pt(*args, **kwargs)
