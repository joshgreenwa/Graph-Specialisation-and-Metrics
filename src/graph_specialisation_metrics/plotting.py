"""Plot specialisation scores and distance-resolved score contributions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .distance import assert_reconstructs_scores
from .scores import specialisation_measures


def load_scores(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate ``scores.npz``."""

    with np.load(Path(path), allow_pickle=False) as archive:
        scores = {name: archive[name] for name in archive.files}

    for name in ("semantic_scores", "structural_scores"):
        if name not in scores:
            raise ValueError(f"score file is missing {name!r}")
        values = np.asarray(scores[name], dtype=np.float64)
        if (
            values.ndim not in (1, 2)
            or values.size == 0
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError(f"{name} must be a finite, non-negative array")
        scores[name] = values

    if scores["semantic_scores"].shape != scores["structural_scores"].shape:
        raise ValueError("semantic and structural scores must have the same shape")

    distance_names = (
        "semantic_distance_contributions",
        "structural_distance_contributions",
        "distance_categories",
    )
    present = [name in scores for name in distance_names]
    if any(present) and not all(present):
        raise ValueError(
            "distance-resolved score contributions and distance categories must be stored together"
        )
    if all(present):
        labels = np.asarray(scores["distance_categories"])
        if labels.ndim != 1 or labels.size == 0 or len(set(labels.tolist())) != labels.size:
            raise ValueError("distance_categories must be non-empty and unique")
        for name in distance_names[:2]:
            values = np.asarray(scores[name], dtype=np.float64)
            if (
                values.ndim < 2
                or values.shape[-1] != labels.size
                or int(np.prod(values.shape[:-1])) != scores["semantic_scores"].size
                or not np.isfinite(values).all()
                or np.any(values < 0)
            ):
                raise ValueError(
                    f"{name} does not align with the specialisation scores and distance categories"
                )
            scores[name] = values.reshape(-1, labels.size)
        scores["distance_categories"] = labels.astype(str)
        assert_reconstructs_scores(
            scores["semantic_distance_contributions"], scores["semantic_scores"].reshape(-1)
        )
        assert_reconstructs_scores(
            scores["structural_distance_contributions"],
            scores["structural_scores"].reshape(-1),
        )
    return scores


def plot_scores(
    scores: Mapping[str, np.ndarray],
    *,
    epsilon: float = 1e-12,
) -> plt.Figure:
    """Plot specialisation scores, ``(D_rel, J)``, and distance-resolved score contributions."""

    semantic_scores = np.asarray(scores["semantic_scores"], dtype=np.float64).reshape(-1)
    structural_scores = np.asarray(scores["structural_scores"], dtype=np.float64).reshape(-1)
    seeds = np.asarray(scores.get("seed", np.zeros(semantic_scores.size, dtype=np.int64))).reshape(
        -1
    )
    if seeds.shape != semantic_scores.shape:
        raise ValueError("seed metadata must align with the flattened specialisation scores")
    joint = np.empty_like(semantic_scores)
    selectivity = np.empty_like(semantic_scores)
    for seed in dict.fromkeys(seeds.tolist()):
        mask = seeds == seed
        measures = specialisation_measures(
            semantic_scores[mask],
            structural_scores[mask],
            epsilon=epsilon,
        )
        joint[mask] = measures.joint_sensitivity
        selectivity[mask] = measures.selectivity
    heads = np.arange(semantic_scores.size)

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    axes[0].plot(heads, semantic_scores, "o-", label=r"$S_{\mathrm{sem}}(t,h)$")
    axes[0].plot(heads, structural_scores, "o-", label=r"$S_{\mathrm{str}}(t,h)$")
    axes[0].set(
        xlabel="Head",
        ylabel=r"Specialisation score $S_c(t,h)$",
        title="Semantic and structural specialisation scores",
    )
    axes[0].legend(frameon=False)

    scatter = axes[1].scatter(selectivity, joint, c=selectivity, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].axvline(0, color="0.75", linewidth=1)
    axes[1].set(
        xlabel=r"Selectivity $D_{\mathrm{rel}}(t,h)$",
        ylabel=r"Joint sensitivity $J(t,h)$",
        title="Joint sensitivity and selectivity",
    )
    figure.colorbar(scatter, ax=axes[1], label=r"Selectivity $D_{\mathrm{rel}}(t,h)$")

    labels = np.asarray(scores["distance_categories"])
    positions = np.arange(labels.size)
    axes[2].plot(
        positions,
        np.asarray(scores["semantic_distance_contributions"]).mean(axis=0),
        "o-",
        label=r"$C_{\mathrm{sem}}(t,h,d)$",
    )
    axes[2].plot(
        positions,
        np.asarray(scores["structural_distance_contributions"]).mean(axis=0),
        "o-",
        label=r"$C_{\mathrm{str}}(t,h,d)$",
    )
    axes[2].set(
        xlabel=r"Distance category $d$",
        ylabel=r"Mean score contribution $C_c(t,h,d)$",
        title="Distance-resolved score contributions",
        xticks=positions,
        xticklabels=labels,
    )
    axes[2].legend(frameon=False)
    return figure
