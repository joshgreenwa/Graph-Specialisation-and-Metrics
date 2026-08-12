"""Experiment entry points."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ..distance import ReconstructionError, assert_reconstructs_scores


class ExperimentSetupError(RuntimeError):
    """Experiment setup failed."""


@contextmanager
def dataset_lock(root: Path, name: str) -> Iterator[None]:
    """Lock dataset setup across Slurm array jobs."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Slurm and Colab are Unix
        yield
        return
    with (root / f".gsm-{name}.lock").open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def save_scores(
    output_dir: Path,
    semantic_scores: Any,
    structural_scores: Any,
    *,
    semantic_distance_contributions: Any | None = None,
    structural_distance_contributions: Any | None = None,
    distance_categories: Any | None = None,
    **extra: Any,
) -> Path:
    """Write ``scores.npz``."""

    semantic_scores = np.asarray(semantic_scores, dtype=np.float64).reshape(-1)
    structural_scores = np.asarray(structural_scores, dtype=np.float64).reshape(-1)
    if semantic_scores.shape != structural_scores.shape or semantic_scores.size == 0:
        raise ValueError("semantic and structural scores must have the same non-empty shape")
    if np.any(~np.isfinite(semantic_scores)) or np.any(~np.isfinite(structural_scores)):
        raise ValueError("scores must be finite")
    if np.any(semantic_scores < 0) or np.any(structural_scores < 0):
        raise ValueError("scores must be non-negative")

    arrays: dict[str, Any] = {
        "semantic_scores": semantic_scores,
        "structural_scores": structural_scores,
    }
    distance_values = (
        semantic_distance_contributions,
        structural_distance_contributions,
        distance_categories,
    )
    if any(value is not None for value in distance_values):
        if any(value is None for value in distance_values):
            raise ValueError(
                "both distance-resolved score contribution arrays and their categories are required"
            )
        semantic_contributions = np.asarray(semantic_distance_contributions, dtype=np.float64)
        structural_contributions = np.asarray(structural_distance_contributions, dtype=np.float64)
        categories = np.asarray(distance_categories, dtype=str)
        if (
            categories.ndim != 1
            or categories.size == 0
            or len(set(categories.tolist())) != categories.size
        ):
            raise ValueError("distance categories must be non-empty and unique")
        expected = (semantic_scores.size, categories.size)
        if semantic_contributions.shape != expected or structural_contributions.shape != expected:
            raise ValueError(f"distance-resolved score contributions must have shape {expected}")
        if (
            not np.isfinite(semantic_contributions).all()
            or not np.isfinite(structural_contributions).all()
            or np.any(semantic_contributions < 0)
            or np.any(structural_contributions < 0)
        ):
            raise ValueError(
                "distance-resolved score contributions must be finite and non-negative"
            )
        try:
            assert_reconstructs_scores(semantic_contributions, semantic_scores)
            assert_reconstructs_scores(structural_contributions, structural_scores)
        except ReconstructionError as exc:
            raise ValueError(str(exc)) from exc
        arrays.update(
            semantic_distance_contributions=semantic_contributions,
            structural_distance_contributions=structural_contributions,
            distance_categories=categories,
        )
    if arrays.keys() & extra.keys():
        raise ValueError("extra metadata cannot replace score arrays")
    arrays.update({name: np.asarray(value) for name, value in extra.items()})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scores.npz"
    np.savez_compressed(path, **arrays)
    return path


from . import graphbench, graphormer, grit, mixed

EXPERIMENTS = {
    "mixed": mixed,
    "graphbench": graphbench,
    "graphormer": graphormer,
    "grit": grit,
}

__all__ = ["EXPERIMENTS", "ExperimentSetupError", "dataset_lock", "save_scores"]
