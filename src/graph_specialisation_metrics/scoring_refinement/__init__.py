"""Reusable semantic/PE head-scoring refinement experiment.

The public entry point is :func:`run`.  Pure intervention and score primitives are
also exported so new task runners can reuse the methodology without importing the
Colab frontend.
"""

from .config import RefinementConfig, RunSizes
from .runner import run

__all__ = ["RefinementConfig", "RunSizes", "run"]
