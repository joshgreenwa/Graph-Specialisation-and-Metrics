"""Cache-only multi-seed figures for the Chapter 6 molecular comparison.

The analysis deliberately treats independently trained heads as independent
observations.  Head-level panels retain the seed identity; only layer- and
model-level summaries are averaged over seeds.  Every figure is derived from
the consolidated canonical ``scores/raw.pt`` and ``carriage/fields.pt`` files.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .chapter6_spatial_explorer import (
    SpatialModel,
    head_metrics,
    inventory,
    load_models,
    model_profiles,
)

ANALYSIS_VERSION = "chapter6-molecular-multiseed-v1"
SEEDS = (0, 1, 2)
CHANNELS = ("semantic", "structural")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    tasks: tuple[str, ...]
    labels: Mapping[str, str]


def dataset_spec(dataset: str) -> DatasetSpec:
    """Return the fixed five-architecture comparison for ZINC or QM9."""

    name = str(dataset).strip().lower()
    if name == "zinc":
        tasks = (
            "zinc_1hop",
            "zinc_1hop_vnode",
            "zinc_2hop",
            "zinc_2hop_vnode",
            "zinc",
        )
    elif name == "qm9":
        tasks = (
            "qm9_gap_1hop",
            "qm9_gap_1hop_vnode",
            "qm9_gap_2hop",
            "qm9_gap_2hop_vnode",
            "qm9_gap_dense",
        )
    else:
        raise ValueError("dataset must be 'zinc' or 'qm9'")
    labels = {
        tasks[0]: "1-hop",
        tasks[1]: "1-hop + VNode",
        tasks[2]: "2-hop",
        tasks[3]: "2-hop + VNode",
        tasks[4]: "Dense GRIT",
    }
    return DatasetSpec(name=name, tasks=tasks, labels=labels)


def cache_inventory(
    canonical_root: Path,
    *,
    dataset: str,
    seeds: Sequence[int] = SEEDS,
) -> list[dict[str, Any]]:
    spec = dataset_spec(dataset)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.extend(inventory((Path(canonical_root),), spec.tasks, seed=int(seed)))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(figure: Any, figures_dir: Path, stem: str) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{stem}.png"
    pdf = figures_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=240, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _mean_range(values: Sequence[float]) -> tuple[float, float, float]:
    finite = _finite(values)
    if not finite.size:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(finite)), float(np.min(finite)), float(np.max(finite))


def _normalise_last(values: Any) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    denominator = np.sum(array, axis=-1, keepdims=True)
    output = np.full_like(array, np.nan)
    np.divide(array, denominator, out=output, where=denominator > 1.0e-12)
    return output


def _load_all_models(
    canonical_root: Path,
    spec: DatasetSpec,
    seeds: Sequence[int],
) -> tuple[list[SpatialModel], list[str]]:
    models: list[SpatialModel] = []
    warnings: list[str] = []
    for seed in seeds:
        seed_models, seed_warnings = load_models(
            (Path(canonical_root),), spec.tasks, seed=int(seed)
        )
        models.extend(seed_models)
        warnings.extend(seed_warnings)
    order = {task: index for index, task in enumerate(spec.tasks)}
    models.sort(key=lambda model: (order.get(model.task, len(order)), int(model.seed)))
    return models, warnings


def _head_rows(models: Sequence[SpatialModel]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for row in head_metrics((model,)):
            rows.append({"seed": int(model.seed), **row})
    return rows


def layer_organisation_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate heads within a seed, then seeds within a layer."""

    seed_groups: dict[tuple[str, int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        seed_groups.setdefault((str(row["task"]), int(row["seed"]), int(row["layer"])), []).append(
            row
        )
    per_seed: list[dict[str, Any]] = []
    for (task, seed, layer), group in sorted(seed_groups.items()):
        record: dict[str, Any] = {"task": task, "seed": seed, "layer": layer}
        for source in ("semantic", "structural", "attention"):
            distances = _finite(
                [float(row.get(f"{source}_expected_distance", np.nan)) for row in group]
            )
            record[f"{source}_expected_distance"] = (
                float(np.mean(distances)) if distances.size else float("nan")
            )
        width_gap = _finite(
            [
                float(row.get("structural_spatial_variance", np.nan))
                - float(row.get("semantic_spatial_variance", np.nan))
                for row in group
            ]
        )
        record["structural_excess_width"] = (
            float(np.mean(width_gap)) if width_gap.size else float("nan")
        )
        per_seed.append(record)

    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in per_seed:
        groups.setdefault((str(row["task"]), int(row["layer"])), []).append(row)
    output: list[dict[str, Any]] = []
    fields = (
        "semantic_expected_distance",
        "structural_expected_distance",
        "attention_expected_distance",
        "structural_excess_width",
    )
    for (task, layer), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "task": task,
            "layer": layer,
            "seeds": len({int(row["seed"]) for row in group}),
        }
        for field in fields:
            mean, low, high = _mean_range([float(row.get(field, np.nan)) for row in group])
            record[f"{field}_mean"] = mean
            record[f"{field}_min"] = low
            record[f"{field}_max"] = high
        output.append(record)
    return output


def _display_peak(label: Any) -> tuple[str, int | None]:
    text = str(label).replace("_", " ").lower()
    if text == "virtual":
        return "virtual", None
    try:
        distance = int(float(text))
    except ValueError:
        return text, None
    if distance <= 3:
        return str(distance), distance
    if distance <= 7:
        return "4-7", 4
    return "8+", 5


def alignment_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    activity_quantile: float = 0.25,
) -> list[dict[str, Any]]:
    """Summarise coarse co-variation among reliable heads without matching identities."""

    by_task_seed: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task_seed.setdefault((str(row["task"]), int(row["seed"])), []).append(row)
    reliable: dict[str, list[Mapping[str, Any]]] = {}
    for (task, _seed), group in by_task_seed.items():
        joint = _finite([float(row.get("joint_sensitivity", np.nan)) for row in group])
        floor = float(np.quantile(joint, activity_quantile)) if joint.size else float("inf")
        reliable.setdefault(task, []).extend(
            row
            for row in group
            if np.isfinite(float(row.get("joint_sensitivity", np.nan)))
            and float(row["joint_sensitivity"]) >= floor
        )

    output: list[dict[str, Any]] = []
    for task, group in sorted(reliable.items()):
        semantic = np.asarray(
            [float(row.get("semantic_expected_distance", np.nan)) for row in group]
        )
        structural = np.asarray(
            [float(row.get("structural_expected_distance", np.nan)) for row in group]
        )
        finite = np.isfinite(semantic) & np.isfinite(structural)
        rho = float("nan")
        if int(np.sum(finite)) >= 3:
            from scipy.stats import spearmanr

            rho = float(spearmanr(semantic[finite], structural[finite]).statistic)
        same = []
        adjacent = []
        for row in group:
            semantic_peak, semantic_order = _display_peak(row.get("semantic_peak_distance", ""))
            structural_peak, structural_order = _display_peak(
                row.get("structural_peak_distance", "")
            )
            same.append(semantic_peak == structural_peak)
            if semantic_order is None or structural_order is None:
                adjacent.append(semantic_peak == structural_peak)
            else:
                adjacent.append(abs(semantic_order - structural_order) <= 1)
        output.append(
            {
                "task": task,
                "heads": len(group),
                "activity_quantile": float(activity_quantile),
                "spearman_rho": rho,
                "same_peak_fraction": float(np.mean(same)) if same else float("nan"),
                "same_or_adjacent_peak_fraction": (
                    float(np.mean(adjacent)) if adjacent else float("nan")
                ),
            }
        )
    return output


def vnode_allocation_rows(models: Sequence[SpatialModel]) -> list[dict[str, Any]]:
    """Return virtual-node score and attention shares, first within seed then across seeds."""

    per_seed: list[dict[str, Any]] = []
    for model in models:
        axis = tuple(model.score["axis"])
        virtual = [
            index
            for index, label in enumerate(axis)
            if str(label).replace("_", " ").lower() == "virtual"
        ]
        if len(virtual) != 1:
            continue
        position = virtual[0]
        sources: dict[str, Any] = {
            "semantic_score": model.score["channels"]["semantic"]["heatmap_exact_head"],
            "structural_score": model.score["channels"]["structural"]["heatmap_exact_head"],
            "attention": model.score.get("clean_attention_distance"),
        }
        for source, values in sources.items():
            if values is None:
                continue
            profile = _normalise_last(values)
            if profile.ndim != 3 or profile.shape[-1] != len(axis):
                continue
            for layer in range(profile.shape[0]):
                finite = _finite(profile[layer, :, position].tolist())
                if finite.size:
                    per_seed.append(
                        {
                            "task": model.task,
                            "seed": int(model.seed),
                            "layer": layer,
                            "source": source,
                            "virtual_share": float(np.mean(finite)),
                        }
                    )
    groups: dict[tuple[str, int, str], list[float]] = {}
    for row in per_seed:
        groups.setdefault((str(row["task"]), int(row["layer"]), str(row["source"])), []).append(
            float(row["virtual_share"])
        )
    output: list[dict[str, Any]] = []
    for (task, layer, source), values in sorted(groups.items()):
        mean, low, high = _mean_range(values)
        output.append(
            {
                "task": task,
                "layer": layer,
                "source": source,
                "seeds": len(values),
                "virtual_share_mean": mean,
                "virtual_share_min": low,
                "virtual_share_max": high,
            }
        )
    return output


def population_profile_rows(models: Sequence[SpatialModel]) -> list[dict[str, Any]]:
    per_seed: list[dict[str, Any]] = []
    for model in models:
        for row in model_profiles((model,)):
            per_seed.append({"seed": int(model.seed), **row})
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in per_seed:
        groups.setdefault((str(row["task"]), str(row["source"]), str(row["distance"])), []).append(
            float(row["equal_head_mass"])
        )
    output: list[dict[str, Any]] = []
    for (task, source, distance), values in sorted(groups.items()):
        mean, low, high = _mean_range(values)
        output.append(
            {
                "task": task,
                "source": source,
                "distance": distance,
                "seeds": len(values),
                "mass_mean": mean,
                "mass_min": low,
                "mass_max": high,
            }
        )
    return output


def _plot_spatial_organisation(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2,
        len(spec.tasks),
        figsize=(3.35 * len(spec.tasks), 6.8),
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )
    source_style = {
        "semantic": ("#0072B2", "o", "semantic score"),
        "structural": ("#D55E00", "s", "structural score"),
        "attention": ("#009E73", "^", "attention mass"),
    }
    for column, task in enumerate(spec.tasks):
        selected = sorted(
            [row for row in rows if str(row["task"]) == task],
            key=lambda row: int(row["layer"]),
        )
        layers = np.asarray([int(row["layer"]) for row in selected])
        top = axes[0, column]
        for source, (colour, marker, label) in source_style.items():
            mean = np.asarray([float(row[f"{source}_expected_distance_mean"]) for row in selected])
            low = np.asarray([float(row[f"{source}_expected_distance_min"]) for row in selected])
            high = np.asarray([float(row[f"{source}_expected_distance_max"]) for row in selected])
            if not np.isfinite(mean).any():
                continue
            top.plot(layers, mean, color=colour, marker=marker, label=label)
            top.fill_between(layers, low, high, color=colour, alpha=0.14, linewidth=0)
        top.set_title(spec.labels[task])
        top.set_xlabel("layer")
        if column == 0:
            top.set_ylabel("expected graph distance")
            top.legend(frameon=False, fontsize=8)

        bottom = axes[1, column]
        mean = np.asarray([float(row["structural_excess_width_mean"]) for row in selected])
        low = np.asarray([float(row["structural_excess_width_min"]) for row in selected])
        high = np.asarray([float(row["structural_excess_width_max"]) for row in selected])
        bottom.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
        bottom.plot(layers, mean, color="#7A5195", marker="o")
        bottom.fill_between(layers, low, high, color="#7A5195", alpha=0.18, linewidth=0)
        bottom.set_xlabel("layer")
        if column == 0:
            bottom.set_ylabel("structural $-$ semantic\nspatial variance")
    figure.suptitle(f"{spec.name.upper()}: spatial organisation across architectures")
    return _save_figure(figure, figures_dir, "01_spatial_organisation")


def _plot_specialisation_landscapes(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        2,
        len(spec.tasks),
        figsize=(3.35 * len(spec.tasks), 6.8),
        squeeze=False,
        constrained_layout=True,
    )
    markers = ("o", "s", "^")
    maximum_layer = max(int(row["layer"]) for row in rows)
    colour_map = plt.get_cmap("viridis")
    colour_norm = plt.Normalize(0, maximum_layer)
    scatter = None
    for column, task in enumerate(spec.tasks):
        selected = [row for row in rows if str(row["task"]) == task]
        for row_index, fields in enumerate(
            (
                ("normalized_semantic_score", "normalized_structural_score"),
                ("selectivity", "joint_sensitivity"),
            )
        ):
            axis = axes[row_index, column]
            for seed_index, seed in enumerate(sorted({int(row["seed"]) for row in selected})):
                seed_rows = [row for row in selected if int(row["seed"]) == seed]
                scatter = axis.scatter(
                    [float(row[fields[0]]) for row in seed_rows],
                    [float(row[fields[1]]) for row in seed_rows],
                    c=[int(row["layer"]) for row in seed_rows],
                    cmap=colour_map,
                    norm=colour_norm,
                    marker=markers[seed_index % len(markers)],
                    s=24,
                    alpha=0.78,
                    linewidths=0,
                )
            if row_index == 0:
                limits = np.asarray(axis.get_xlim() + axis.get_ylim())
                lower = min(float(np.min(limits)), 0.0)
                upper = float(np.max(limits))
                axis.plot(
                    [lower, upper],
                    [lower, upper],
                    color="#999999",
                    linestyle="--",
                    linewidth=0.8,
                )
                axis.set_xlim(lower, upper)
                axis.set_ylim(lower, upper)
                axis.set_aspect("equal", adjustable="box")
                axis.set_title(spec.labels[task])
                axis.set_xlabel("normalised semantic score")
                if column == 0:
                    axis.set_ylabel("normalised structural score")
            else:
                axis.axvline(0.0, color="#999999", linewidth=0.8, linestyle="--")
                axis.set_xlabel(r"relative selectivity $D_{\mathrm{rel}}$")
                if column == 0:
                    axis.set_ylabel(r"joint sensitivity $J$")
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, label="layer", shrink=0.82, pad=0.01)
    seed_handles = [
        Line2D([], [], marker=markers[index], linestyle="", color="#555555", label=f"seed {seed}")
        for index, seed in enumerate(sorted({int(row["seed"]) for row in rows}))
    ]
    axes[1, 0].legend(handles=seed_handles, frameon=False, fontsize=8, loc="upper left")
    figure.suptitle(f"{spec.name.upper()}: head specialisation landscapes")
    return _save_figure(figure, figures_dir, "02_specialisation_landscapes")


def _head_matrix(
    rows: Sequence[Mapping[str, Any]], task: str, seed: int, field: str
) -> tuple[np.ndarray, list[int], list[int]]:
    selected = [row for row in rows if str(row["task"]) == task and int(row["seed"]) == int(seed)]
    layers = sorted({int(row["layer"]) for row in selected})
    heads = sorted({int(row["head"]) for row in selected})
    matrix = np.full((len(layers), len(heads)), np.nan)
    layer_index = {value: index for index, value in enumerate(layers)}
    head_index = {value: index for index, value in enumerate(heads)}
    for row in selected:
        matrix[layer_index[int(row["layer"])], head_index[int(row["head"])]] = float(
            row.get(field, np.nan)
        )
    return matrix, layers, heads


def _plot_reach_gap_heatmap(
    rows: Sequence[Mapping[str, Any]],
    spec: DatasetSpec,
    seeds: Sequence[int],
    figures_dir: Path,
    *,
    channel: str,
    stem: str,
) -> list[Path]:
    import matplotlib.pyplot as plt

    field = f"{channel}_attention_reach_gap"
    matrices = [
        _head_matrix(rows, task, int(seed), field)[0] for seed in seeds for task in spec.tasks
    ]
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    limit = float(np.quantile(np.abs(finite), 0.99)) if finite.size else 1.0
    limit = max(limit, 1.0e-6)
    figure, axes = plt.subplots(
        len(seeds),
        len(spec.tasks),
        figsize=(3.25 * len(spec.tasks), 2.45 * len(seeds)),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row_index, seed in enumerate(seeds):
        for column, task in enumerate(spec.tasks):
            matrix, layers, heads = _head_matrix(rows, task, int(seed), field)
            axis = axes[row_index, column]
            image = axis.imshow(
                matrix,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                aspect="auto",
                interpolation="nearest",
            )
            axis.set_xticks(np.arange(len(heads)), heads, fontsize=7)
            axis.set_yticks(np.arange(len(layers)), layers, fontsize=7)
            if row_index == 0:
                axis.set_title(spec.labels[task], fontsize=10)
            if row_index == len(seeds) - 1:
                axis.set_xlabel("head")
            if column == 0:
                axis.set_ylabel(f"seed {seed}\nlayer")
    if image is not None:
        figure.colorbar(image, ax=axes, label="score reach $-$ attention reach (hops)", pad=0.01)
    figure.suptitle(f"{spec.name.upper()}: {channel} attention--score reach gap")
    return _save_figure(figure, figures_dir, stem)


def _plot_distance_alignment(
    rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    spec: DatasetSpec,
    figures_dir: Path,
    *,
    activity_quantile: float,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        1,
        len(spec.tasks),
        figsize=(3.35 * len(spec.tasks), 3.6),
        squeeze=False,
        constrained_layout=True,
    )
    markers = ("o", "s", "^")
    maximum_layer = max(int(row["layer"]) for row in rows)
    norm = plt.Normalize(0, maximum_layer)
    summary_lookup = {str(row["task"]): row for row in summary}
    scatter = None
    for column, task in enumerate(spec.tasks):
        axis = axes[0, column]
        selected = [row for row in rows if str(row["task"]) == task]
        for seed_index, seed in enumerate(sorted({int(row["seed"]) for row in selected})):
            seed_rows = [row for row in selected if int(row["seed"]) == seed]
            joint = _finite([float(row["joint_sensitivity"]) for row in seed_rows])
            floor = float(np.quantile(joint, activity_quantile)) if joint.size else float("inf")
            seed_rows = [
                row
                for row in seed_rows
                if float(row["joint_sensitivity"]) >= floor
                and np.isfinite(float(row["semantic_expected_distance"]))
                and np.isfinite(float(row["structural_expected_distance"]))
            ]
            if not seed_rows:
                continue
            sizes = np.asarray([max(float(row["joint_sensitivity"]), 0.0) for row in seed_rows])
            if float(np.max(sizes)) > 0:
                sizes = 16.0 + 45.0 * np.sqrt(sizes / float(np.max(sizes)))
            else:
                sizes = np.full(len(seed_rows), 16.0)
            scatter = axis.scatter(
                [float(row["semantic_expected_distance"]) for row in seed_rows],
                [float(row["structural_expected_distance"]) for row in seed_rows],
                c=[int(row["layer"]) for row in seed_rows],
                cmap="viridis",
                norm=norm,
                marker=markers[seed_index % len(markers)],
                s=sizes,
                alpha=0.72,
                linewidths=0,
            )
        limits = np.asarray(axis.get_xlim() + axis.get_ylim())
        lower = min(float(np.min(limits)), 0.0)
        upper = float(np.max(limits))
        axis.plot([lower, upper], [lower, upper], color="#888888", linestyle="--", linewidth=0.8)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(spec.labels[task])
        axis.set_xlabel("semantic expected distance")
        if column == 0:
            axis.set_ylabel("structural expected distance")
        record = summary_lookup.get(task)
        if record is not None:
            axis.text(
                0.04,
                0.96,
                (
                    rf"$\rho$={float(record['spearman_rho']):.2f}"
                    + "\n"
                    + f"same peak={100 * float(record['same_peak_fraction']):.0f}%\n"
                    + "within one bin="
                    + f"{100 * float(record['same_or_adjacent_peak_fraction']):.0f}%"
                ),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, label="layer", shrink=0.82, pad=0.01)
    seed_handles = [
        Line2D([], [], marker=markers[index], linestyle="", color="#555555", label=f"seed {seed}")
        for index, seed in enumerate(sorted({int(row["seed"]) for row in rows}))
    ]
    axes[0, 0].legend(handles=seed_handles, frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(
        f"{spec.name.upper()}: semantic and structural distance alignment "
        f"(top {100 * (1 - activity_quantile):.0f}% by $J$)"
    )
    return _save_figure(figure, figures_dir, "05_semantic_structural_distance_alignment")


def _plot_vnode_allocation(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [task for task in spec.tasks if "vnode" in task]
    if not rows or not tasks:
        return []
    figure, axes = plt.subplots(
        1, len(tasks), figsize=(5.2 * len(tasks), 3.8), squeeze=False, constrained_layout=True
    )
    styles = {
        "semantic_score": ("#0072B2", "o", "semantic score"),
        "structural_score": ("#D55E00", "s", "structural score"),
        "attention": ("#009E73", "^", "attention mass"),
    }
    for column, task in enumerate(tasks):
        axis = axes[0, column]
        selected = sorted(
            [row for row in rows if str(row["task"]) == task],
            key=lambda row: (str(row["source"]), int(row["layer"])),
        )
        for source, (colour, marker, label) in styles.items():
            source_rows = [row for row in selected if str(row["source"]) == source]
            if not source_rows:
                continue
            layers = np.asarray([int(row["layer"]) for row in source_rows])
            mean = np.asarray([float(row["virtual_share_mean"]) for row in source_rows])
            low = np.asarray([float(row["virtual_share_min"]) for row in source_rows])
            high = np.asarray([float(row["virtual_share_max"]) for row in source_rows])
            axis.plot(layers, mean, color=colour, marker=marker, label=label)
            axis.fill_between(layers, low, high, color=colour, alpha=0.15, linewidth=0)
        axis.set_title(spec.labels[task])
        axis.set_xlabel("layer")
        if column == 0:
            axis.set_ylabel("fraction assigned to virtual node")
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"{spec.name.upper()}: virtual-node allocation")
    return _save_figure(figure, figures_dir, "06_virtual_node_allocation")


def _distance_order(label: str) -> tuple[int, str]:
    order = {"0": 0, "1": 1, "2": 2, "3": 3, "4-7": 4, "8+": 5, "virtual": 6}
    return order.get(str(label).lower(), 100), str(label)


def _plot_population_profiles(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    if not any(str(row["source"]).endswith("_carriage") for row in rows):
        return []
    figure, axes = plt.subplots(
        2,
        len(spec.tasks),
        figsize=(3.35 * len(spec.tasks), 6.8),
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, channel in enumerate(CHANNELS):
        for column, task in enumerate(spec.tasks):
            axis = axes[row_index, column]
            selected = [row for row in rows if str(row["task"]) == task]
            labels = sorted(
                {
                    str(row["distance"])
                    for row in selected
                    if str(row["source"]) in (f"{channel}_score", f"{channel}_carriage")
                },
                key=_distance_order,
            )
            x = np.arange(len(labels))
            for source, colour, marker, label in (
                (f"{channel}_score", "#4C78A8", "o", "mean head score"),
                (f"{channel}_carriage", "#222222", "s", "final-state response"),
            ):
                lookup = {
                    str(row["distance"]): row for row in selected if str(row["source"]) == source
                }
                if not lookup:
                    continue
                mean = np.asarray(
                    [
                        float(lookup[label]["mass_mean"]) if label in lookup else np.nan
                        for label in labels
                    ]
                )
                low = np.asarray(
                    [
                        float(lookup[label]["mass_min"]) if label in lookup else np.nan
                        for label in labels
                    ]
                )
                high = np.asarray(
                    [
                        float(lookup[label]["mass_max"]) if label in lookup else np.nan
                        for label in labels
                    ]
                )
                axis.plot(x, mean, color=colour, marker=marker, label=label)
                axis.fill_between(x, low, high, color=colour, alpha=0.14, linewidth=0)
            axis.set_xticks(x, labels)
            axis.set_title(spec.labels[task], fontsize=10)
            axis.set_xlabel("distance")
            if column == 0:
                axis.set_ylabel(f"{channel}\nnormalised mass")
            if row_index == 0 and column == 0:
                axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"{spec.name.upper()}: head-score profiles and final-state response")
    return _save_figure(figure, figures_dir, "07_score_and_final_state_response")


def run(
    canonical_root: Path,
    output_dir: Path,
    *,
    dataset: str,
    seeds: Sequence[int] = SEEDS,
    strict_inventory: bool = True,
    activity_quantile: float = 0.25,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate the focused multi-seed Chapter 6 figure suite."""

    spec = dataset_spec(dataset)
    canonical_root = Path(canonical_root)
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows = cache_inventory(canonical_root, dataset=spec.name, seeds=seeds)
    _write_csv(output_dir / "cache_inventory.csv", inventory_rows)
    missing = [
        row
        for row in inventory_rows
        if not bool(row.get("score_exists")) or not bool(row.get("carriage_exists"))
    ]
    if strict_inventory and missing:
        detail = ", ".join(f"{row['task']}/seed_{row['seed']}" for row in missing)
        raise FileNotFoundError(f"missing required consolidated cache(s): {detail}")

    models, warnings = _load_all_models(canonical_root, spec, seeds)
    expected = len(spec.tasks) * len(tuple(seeds))
    if strict_inventory and len(models) != expected:
        raise RuntimeError(f"loaded {len(models)} models; expected {expected}")
    if not models:
        raise FileNotFoundError("no usable score caches were found")
    if verbose:
        for warning in warnings:
            print(f"[chapter6-multiseed:warning] {warning}", flush=True)
        print(
            f"[chapter6-multiseed] {spec.name}: loaded {len(models)} cached runs",
            flush=True,
        )

    head_rows = _head_rows(models)
    organisation_rows = layer_organisation_rows(head_rows)
    alignment_rows = alignment_summary_rows(head_rows, activity_quantile=activity_quantile)
    vnode_rows = vnode_allocation_rows(models)
    profile_rows = population_profile_rows(models)
    tables = {
        "head_metrics.csv": head_rows,
        "layer_spatial_organisation.csv": organisation_rows,
        "semantic_structural_alignment.csv": alignment_rows,
        "vnode_allocation.csv": vnode_rows,
        "population_score_carriage_profiles.csv": profile_rows,
    }
    for filename, rows in tables.items():
        _write_csv(output_dir / filename, rows)

    figures: list[Path] = []
    figures.extend(_plot_spatial_organisation(organisation_rows, spec, figures_dir))
    figures.extend(_plot_specialisation_landscapes(head_rows, spec, figures_dir))
    figures.extend(
        _plot_reach_gap_heatmap(
            head_rows,
            spec,
            seeds,
            figures_dir,
            channel="semantic",
            stem="03_semantic_attention_score_reach_gap",
        )
    )
    figures.extend(
        _plot_reach_gap_heatmap(
            head_rows,
            spec,
            seeds,
            figures_dir,
            channel="structural",
            stem="04_structural_attention_score_reach_gap",
        )
    )
    figures.extend(
        _plot_distance_alignment(
            head_rows,
            alignment_rows,
            spec,
            figures_dir,
            activity_quantile=activity_quantile,
        )
    )
    figures.extend(_plot_vnode_allocation(vnode_rows, spec, figures_dir))
    figures.extend(_plot_population_profiles(profile_rows, spec, figures_dir))

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "dataset": spec.name,
        "canonical_root": str(canonical_root),
        "output_dir": str(output_dir),
        "tasks": list(spec.tasks),
        "seeds": [int(seed) for seed in seeds],
        "strict_inventory": bool(strict_inventory),
        "activity_quantile": float(activity_quantile),
        "runs_loaded": len(models),
        "warnings": warnings,
        "figures": [str(path) for path in figures],
        "tables": [str(output_dir / name) for name in tables],
        "interpretation": {
            "head_identity": "never aligned or averaged across seeds",
            "seed_bands": "minimum-to-maximum range over independently trained seeds",
            "expected_distance": "molecular graph distance; virtual node is separate",
            "reach_gap": "intervention-defined score reach minus clean-attention reach",
            "alignment": (
                "Spearman expected-distance co-variation plus coarse peak-bin agreement "
                f"among the top {100 * (1 - activity_quantile):.0f}% of heads by J "
                "within each seed"
            ),
            "final_state_response": "learned response, not task necessity",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest
