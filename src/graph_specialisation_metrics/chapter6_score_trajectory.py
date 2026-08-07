"""Cache-only ZINC specialisation landscapes over seed-0 training checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .methodology.cache import load_cache_artifact_file

EPOCHS = (10, 100, 250, 500, 1_000, 1_990)
ARCHITECTURES = ("dense", "1hop")
TASKS = {"dense": "zinc", "1hop": "zinc_1hop"}
LABELS = {"dense": "Dense GRIT", "1hop": "1-hop GRIT"}


def score_cache_path(root: Path, architecture: str, epoch: int) -> Path:
    task = TASKS[str(architecture)]
    return (
        Path(root)
        / "score_trajectory_outputs"
        / str(architecture)
        / f"epoch_{int(epoch):04d}"
        / task
        / "seed_0/cache/scores/raw.pt"
    )


def cache_inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "architecture": architecture,
            "epoch": int(epoch),
            "task": TASKS[architecture],
            "score_path": str(score_cache_path(root, architecture, epoch)),
            "score_exists": score_cache_path(root, architecture, epoch).is_file(),
        }
        for architecture in ARCHITECTURES
        for epoch in EPOCHS
    ]


def missing_architectures(root: Path) -> tuple[str, ...]:
    """Return architectures with at least one absent trajectory score cache."""

    rows = cache_inventory(root)
    return tuple(
        architecture
        for architecture in ARCHITECTURES
        if any(
            row["architecture"] == architecture and not bool(row["score_exists"])
            for row in rows
        )
    )


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def load_rows(root: Path, *, strict: bool = True) -> list[dict[str, Any]]:
    """Load the twelve immutable score caches into one head-level long table."""

    rows: list[dict[str, Any]] = []
    missing = []
    expected_shape: tuple[int, int] | None = None
    for architecture in ARCHITECTURES:
        task = TASKS[architecture]
        for epoch in EPOCHS:
            path = score_cache_path(root, architecture, epoch)
            if not path.is_file():
                missing.append(path)
                continue
            artifact = load_cache_artifact_file(path)
            contract = dict(artifact.metadata["contract"])
            if contract.get("task") != task or int(contract.get("train_seed", -1)) != 0:
                raise RuntimeError(f"trajectory score cache has the wrong task or seed: {path}")
            score = artifact.value
            semantic = np.asarray(score["channels"]["semantic"]["raw"], dtype=np.float64)
            structural = np.asarray(score["channels"]["structural"]["raw"], dtype=np.float64)
            coordinates = score["coordinates"]
            joint = np.asarray(_field(coordinates, "joint_sensitivity"), dtype=np.float64)
            selectivity = np.asarray(_field(coordinates, "selectivity"), dtype=np.float64)
            shape = semantic.shape
            if (
                semantic.ndim != 2
                or structural.shape != shape
                or joint.shape != shape
                or selectivity.shape != shape
                or not all(
                    np.isfinite(values).all()
                    for values in (semantic, structural, joint, selectivity)
                )
            ):
                raise RuntimeError(f"trajectory score arrays are malformed: {path}")
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise RuntimeError("trajectory checkpoints do not share one head geometry")
            for layer, head in np.ndindex(shape):
                rows.append(
                    {
                        "architecture": architecture,
                        "epoch": int(epoch),
                        "task": task,
                        "seed": 0,
                        "layer": int(layer),
                        "head": int(head),
                        "raw_semantic": float(semantic[layer, head]),
                        "raw_structural": float(structural[layer, head]),
                        "joint_sensitivity": float(joint[layer, head]),
                        "selectivity": float(selectivity[layer, head]),
                        "score_path": str(path),
                        "score_cache_sha256": artifact.file_sha256,
                        "checkpoint_sha256": str(contract.get("checkpoint_sha256", "")),
                    }
                )
    if strict and missing:
        detail = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing ZINC trajectory score cache(s): {detail}")
    return rows


def _padded_limits(values: Sequence[float], *, include_zero: bool = False) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return (0.0, 1.0)
    lower = float(np.min(array))
    upper = float(np.max(array))
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = upper - lower
    pad = 0.04 * span if span > 1.0e-12 else max(abs(upper), 1.0) * 0.04
    return lower - pad, upper + pad


def _save_figure(figure: Any, figures_dir: Path, stem: str) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{stem}.png"
    pdf = figures_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def plot(
    rows: Sequence[Mapping[str, Any]],
    figures_dir: Path,
) -> list[Path]:
    """Write one 2x6 figure per architecture using globally shared limits."""

    if not rows:
        return []
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    raw_x_limits = _padded_limits([float(row["raw_semantic"]) for row in rows], include_zero=True)
    raw_y_limits = _padded_limits([float(row["raw_structural"]) for row in rows], include_zero=True)
    joint_limits = _padded_limits(
        [float(row["joint_sensitivity"]) for row in rows], include_zero=True
    )
    maximum_layer = max(int(row["layer"]) for row in rows)
    colour_map = plt.get_cmap("viridis")
    colour_norm = plt.Normalize(0, maximum_layer)
    outputs: list[Path] = []
    for architecture in ARCHITECTURES:
        figure, axes = plt.subplots(
            2,
            len(EPOCHS),
            figsize=(17.4, 6.0),
            sharex="row",
            sharey="row",
            squeeze=False,
            constrained_layout=True,
        )
        scatter = None
        architecture_rows = [row for row in rows if str(row["architecture"]) == architecture]
        for column, epoch in enumerate(EPOCHS):
            selected = [row for row in architecture_rows if int(row["epoch"]) == epoch]
            layers = [int(row["layer"]) for row in selected]
            scatter = axes[0, column].scatter(
                [float(row["raw_semantic"]) for row in selected],
                [float(row["raw_structural"]) for row in selected],
                c=layers,
                cmap=colour_map,
                norm=colour_norm,
                s=23,
                alpha=0.8,
                linewidths=0,
            )
            axes[1, column].scatter(
                [float(row["selectivity"]) for row in selected],
                [float(row["joint_sensitivity"]) for row in selected],
                c=layers,
                cmap=colour_map,
                norm=colour_norm,
                s=23,
                alpha=0.8,
                linewidths=0,
            )
            axes[0, column].set_title(f"epoch {epoch}", fontsize=10)
            axes[0, column].set_xlim(raw_x_limits)
            axes[0, column].set_ylim(raw_y_limits)
            axes[1, column].set_xlim(-1.04, 1.04)
            axes[1, column].set_ylim(joint_limits)
            axes[1, column].axvline(0.0, color="#999999", linestyle="--", linewidth=0.7)
            for row_index in range(2):
                axes[row_index, column].grid(alpha=0.16, linewidth=0.6)
            axes[0, column].set_xlabel("semantic score", fontsize=8)
            axes[1, column].set_xlabel(r"relative selectivity $D_{\mathrm{rel}}$", fontsize=8)
        axes[0, 0].set_ylabel("structural score")
        axes[1, 0].set_ylabel(r"joint sensitivity $J$")
        if scatter is not None:
            figure.colorbar(scatter, ax=axes, label="layer", shrink=0.82, pad=0.01)
        figure.suptitle(f"ZINC {LABELS[architecture]}: specialisation across training")
        stem = f"11{'a' if architecture == 'dense' else 'b'}_{architecture}_score_trajectory"
        outputs.extend(_save_figure(figure, figures_dir, stem))
        plt.close(figure)
    return outputs
