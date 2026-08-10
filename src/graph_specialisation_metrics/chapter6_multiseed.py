"""Cache-only multi-seed figures for the Chapter 6 molecular comparison.

The analysis deliberately treats independently trained heads as independent
observations.  Head-level panels retain the seed identity; only layer- and
model-level summaries are averaged over seeds.  Spatial figures are derived from
the consolidated canonical ``scores/raw.pt`` and ``carriage/fields.pt`` files;
the final validation panel also reads focused clean-head ablation summaries.
"""

from __future__ import annotations

import csv
import ctypes
import gc
import json
import os
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
from .methodology.bootstrap import trimmed_mean
from .zinc_cached_rrwp_comparison import DISPLAY_BINS

ANALYSIS_VERSION = "chapter6-molecular-multiseed-v5"
DISTANCE_ALIGNMENT_VERSION = "all-finite-heads-v1"
SPECIALISATION_LANDSCAPE_VARIANT_VERSION = "j-drel-only-v1"
PRESENTATION_VARIANTS_VERSION = "seed-mean-reach-and-joint-response-v2"
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
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / f"{stem}.png"
    pdf = figures_dir / f"{stem}.pdf"
    metadata = {"Creator": "graph_specialisation_metrics", "Title": stem}
    pdf_only = os.environ.get("CHAPTER6_PDF_ONLY", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    outputs: list[Path] = []
    with mpl.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        if not pdf_only:
            figure.savefig(
                png,
                dpi=400,
                bbox_inches="tight",
                pad_inches=0.04,
                facecolor="white",
            )
            outputs.append(png)
        figure.savefig(
            pdf,
            dpi=1200,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
            metadata=metadata,
        )
        outputs.append(pdf)
    plt.close(figure)
    return outputs


def _scale_figure_text(
    figure: Any,
    *,
    factor: float = 1.5625,
    preserve_tick_axes: Sequence[Any] = (),
) -> None:
    """Scale completed figure text once, optionally preserving selected ticks.

    Most Chapter 6 figures already used a 1.25 publication scale.  The default
    is therefore 1.25 * 1.25: a further 25% increase relative to the figures
    produced before the chapter-wide typography refresh.
    """

    preserved = {id(axis) for axis in preserve_tick_axes}
    seen: set[int] = set()

    def scale(text: Any) -> None:
        if text is None or id(text) in seen:
            return
        seen.add(id(text))
        size = float(text.get_fontsize())
        if np.isfinite(size) and size > 0:
            text.set_fontsize(size * float(factor))

    for text in figure.texts:
        scale(text)
    for legend in figure.legends:
        scale(legend.get_title())
        for text in legend.get_texts():
            scale(text)
    for axis in figure.axes:
        scale(axis.title)
        scale(axis.xaxis.label)
        scale(axis.yaxis.label)
        scale(axis.xaxis.get_offset_text())
        scale(axis.yaxis.get_offset_text())
        if id(axis) not in preserved:
            for text in (*axis.get_xticklabels(), *axis.get_yticklabels()):
                scale(text)
        for text in axis.texts:
            scale(text)
        legend = axis.get_legend()
        if legend is not None:
            scale(legend.get_title())
            for text in legend.get_texts():
                scale(text)


def _add_figure_legend(
    figure: Any,
    handles: Sequence[Any],
    labels: Sequence[str],
    *,
    ncol: int,
) -> None:
    """Place shared legends below the data so enlarged text cannot hide marks."""

    if not handles:
        return
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=max(1, int(ncol)),
        frameon=False,
    )


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _mean_range(values: Sequence[float]) -> tuple[float, float, float]:
    finite = _finite(values)
    if not finite.size:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(finite)), float(np.min(finite)), float(np.max(finite))


def _padded_limits(
    values: Sequence[float], *, include_zero: bool = False
) -> tuple[float, float]:
    finite = _finite(values)
    if not finite.size:
        return (0.0, 1.0)
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = upper - lower
    pad = 0.04 * span if span > 1.0e-12 else max(abs(upper), 1.0) * 0.04
    return lower - pad, upper + pad


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


def _release_cached_models(models: list[SpatialModel]) -> None:
    """Release large score/carriage payloads before loading the next seed."""

    models.clear()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _aggregate_streamed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    value_field: str,
    output_field: str,
) -> list[dict[str, Any]]:
    """Combine single-seed summaries without retaining their source caches."""

    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(
            float(row[value_field])
        )
    output: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        mean, low, high = _mean_range(values)
        output.append(
            {
                **dict(zip(keys, key)),
                "seeds": int(np.sum(np.isfinite(np.asarray(values, dtype=np.float64)))),
                f"{output_field}_mean": mean,
                f"{output_field}_min": low,
                f"{output_field}_max": high,
            }
        )
    return output


def _load_summary_rows_streaming(
    canonical_root: Path,
    spec: DatasetSpec,
    seeds: Sequence[int],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    int,
]:
    """Derive all cache-only summaries while holding only one seed in RAM."""

    head_rows: list[dict[str, Any]] = []
    vnode_seed_rows: list[dict[str, Any]] = []
    profile_seed_rows: list[dict[str, Any]] = []
    matched_seed_rows: list[dict[str, Any]] = []
    response_seed_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    runs_loaded = 0
    for seed in seeds:
        for task in spec.tasks:
            models, model_warnings = load_models(
                (Path(canonical_root),), (task,), seed=int(seed)
            )
            warnings.extend(model_warnings)
            runs_loaded += len(models)
            head_rows.extend(_head_rows(models))
            vnode_seed_rows.extend(vnode_allocation_rows(models))
            profile_seed_rows.extend(population_profile_rows(models))
            matched_seed_rows.extend(matched_strength_profile_rows(models))
            response_seed_rows.extend(carriage_response_variant_rows(models))
            _release_cached_models(models)

    vnode_rows = _aggregate_streamed_rows(
        vnode_seed_rows,
        keys=("task", "layer", "source"),
        value_field="virtual_share_mean",
        output_field="virtual_share",
    )
    profile_rows = _aggregate_streamed_rows(
        profile_seed_rows,
        keys=("task", "source", "distance"),
        value_field="mass_mean",
        output_field="mass",
    )
    matched_rows = _aggregate_streamed_rows(
        matched_seed_rows,
        keys=("task", "source", "distance"),
        value_field="strength_mean",
        output_field="strength",
    )
    response_rows = _aggregate_streamed_rows(
        response_seed_rows,
        keys=("task", "channel", "variant", "distance"),
        value_field="response_mean",
        output_field="response",
    )
    return (
        head_rows,
        vnode_rows,
        profile_rows,
        matched_rows,
        response_rows,
        warnings,
        runs_loaded,
    )


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
) -> list[dict[str, Any]]:
    """Summarise co-variation across every head with defined distance profiles."""

    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if np.isfinite(float(row.get("semantic_expected_distance", np.nan))) and np.isfinite(
            float(row.get("structural_expected_distance", np.nan))
        ):
            by_task.setdefault(str(row["task"]), []).append(row)

    output: list[dict[str, Any]] = []
    for task, group in sorted(by_task.items()):
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
                "eligibility": "finite semantic and structural distance profiles",
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


def _distance_groups(axis: Sequence[Any]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    groups: list[tuple[str, tuple[int, ...]]] = []
    for label, lower, upper in DISPLAY_BINS:
        positions = []
        for index, value in enumerate(axis):
            try:
                distance = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(distance) and distance.is_integer() and lower <= int(distance) <= upper:
                positions.append(index)
        groups.append((label, tuple(positions)))
    specials = sorted(
        {
            str(value).replace("_", " ")
            for value in axis
            if not isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    for special in specials:
        positions = tuple(
            index for index, value in enumerate(axis) if str(value).replace("_", " ") == special
        )
        groups.append((special, positions))
    return tuple(groups)


def per_opportunity_head_profile(
    score: Mapping[str, Any], channel: str
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return the typical head's score strength after controlling for opportunity.

    Display bins are formed from cached graph-level sufficient statistics before
    dividing contribution by support.  This avoids both shell-size confounding and
    the artificial inflation that would result from summing already-normalised
    exact-distance columns into a wide tail bin.
    """

    channel_score = score["channels"][channel]
    axis = tuple(score["axis"])
    groups = _distance_groups(axis)
    labels = tuple(label for label, _ in groups)
    contribution = channel_score.get("graph_distance_contribution")
    support = channel_score.get("graph_distance_support")
    grouped: np.ndarray | None = None
    reportable = np.ones(len(groups), dtype=bool)
    if isinstance(contribution, Mapping) and isinstance(support, Mapping):
        keys = sorted(set(contribution).intersection(support), key=lambda value: str(value))
        graph_profiles = []
        grouped_support: dict[Any, np.ndarray] = {}
        for key in keys:
            graph_contribution = np.asarray(contribution[key], dtype=np.float64)
            graph_support = np.asarray(support[key], dtype=np.float64)
            if graph_contribution.ndim != 3 or graph_contribution.shape[-1] != len(axis):
                continue
            if graph_support.shape != (len(axis),):
                continue
            grouped_contribution = np.stack(
                [
                    (
                        graph_contribution[..., list(positions)].sum(axis=-1)
                        if positions
                        else np.zeros(graph_contribution.shape[:2])
                    )
                    for _label, positions in groups
                ],
                axis=-1,
            )
            grouped_opportunity = np.asarray(
                [
                    (float(np.sum(graph_support[list(positions)])) if positions else 0.0)
                    for _label, positions in groups
                ]
            )
            ratio = np.full_like(grouped_contribution, np.nan)
            np.divide(
                grouped_contribution,
                grouped_opportunity[None, None, :],
                out=ratio,
                where=grouped_opportunity[None, None, :] > 0,
            )
            graph_profiles.append(ratio)
            grouped_support[key] = grouped_opportunity
        if graph_profiles:
            stacked = np.stack(graph_profiles)
            valid = np.sum(np.isfinite(stacked), axis=0)
            grouped = np.full(stacked.shape[1:], np.nan)
            np.divide(
                np.nansum(stacked, axis=0),
                valid,
                out=grouped,
                where=valid > 0,
            )

            support_record = channel_score.get("distance_support", {})
            minimum_graphs = int(
                support_record.get("minimum_graphs", 1)
                if isinstance(support_record, Mapping)
                else 1
            )
            minimum_pairs = int(
                support_record.get("minimum_pairs", 1) if isinstance(support_record, Mapping) else 1
            )
            sources: dict[int, set[int]] = {}
            for row in channel_score.get("events", ()):
                sources.setdefault(int(row["graph_id"]), set()).add(int(row["source"]))
            for position in range(len(groups)):
                supporting = [
                    key for key, values in grouped_support.items() if values[position] > 0
                ]
                pairs = float(
                    np.sum(
                        [
                            grouped_support[key][position] * max(len(sources.get(int(key), ())), 1)
                            for key in supporting
                        ]
                    )
                )
                reportable[position] = len(supporting) >= minimum_graphs and pairs >= minimum_pairs
    if grouped is None:
        exact = channel_score.get("heatmap_per_opportunity_head")
        if exact is None:
            return labels, np.full(len(labels), np.nan)
        exact = np.asarray(exact, dtype=np.float64)
        grouped = np.stack(
            [
                (
                    np.nanmean(exact[..., list(positions)], axis=-1)
                    if positions
                    else np.full(exact.shape[:2], np.nan)
                )
                for _label, positions in groups
            ],
            axis=-1,
        )
    grouped[..., ~reportable] = np.nan
    profile = _normalise_last(np.where(np.isfinite(grouped), grouped, 0.0))
    profile[..., ~reportable] = np.nan
    flattened = profile.reshape(-1, len(labels))
    valid = np.sum(np.isfinite(flattened), axis=0)
    result = np.full(len(labels), np.nan)
    np.divide(np.nansum(flattened, axis=0), valid, out=result, where=valid > 0)
    finite = np.isfinite(result)
    if np.any(finite) and float(np.sum(result[finite])) > 1.0e-12:
        result[finite] /= float(np.sum(result[finite]))
    return labels, result


def raw_carriage_strength_profile(
    carriage: Mapping[str, Any],
    channel: str,
    *,
    minimum_graphs: int = 10,
    minimum_pairs: int = 50,
    normalise: bool = True,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return graph-balanced raw final-state response per eligible carrier.

    Donors are averaged within a source, carrier sums and counts are combined
    across sources within a graph, and graph-level pair means are combined with
    the canonical 20% trimmed mean.  Missing opportunities remain missing.  When
    ``normalise`` is true, only the completed vector is normalised, making it a
    shape comparison without changing the underlying per-carrier estimand.
    """

    rows = list(carriage["channels"][channel]["pairs"])
    # Use the fixed display names directly so empty molecular bins remain visible
    # as missing rather than being silently converted to zero.
    labels = tuple(label for label, _lower, _upper in DISPLAY_BINS) + tuple(
        label.replace("_", " ")
        for label in sorted(
            {
                str(row.get("carrier_kind"))
                for row in rows
                if not np.isfinite(float(row["distance"]))
                and str(row.get("carrier_kind", "molecular_node")) != "molecular_node"
            }
        )
    )

    def bin_label(row: Mapping[str, Any]) -> str | None:
        distance = float(row["distance"])
        if np.isfinite(distance):
            for label, lower, upper in DISPLAY_BINS:
                if lower <= int(distance) <= upper:
                    return label
            return None
        kind = str(row.get("carrier_kind", "molecular_node"))
        return None if kind == "molecular_node" else kind.replace("_", " ")

    estimates = np.full(len(labels), np.nan)
    for position, label in enumerate(labels):
        selected = [
            row for row in rows if bin_label(row) == label and np.isfinite(float(row["F_sens"]))
        ]
        graphs = {int(row["graph_id"]) for row in selected}
        pairs = {
            (int(row["graph_id"]), int(row["carrier"]), int(row["source"])) for row in selected
        }
        if len(graphs) < int(minimum_graphs) or len(pairs) < int(minimum_pairs):
            continue
        events: dict[tuple[int, int, int], list[float]] = {}
        for row in selected:
            events.setdefault(
                (int(row["graph_id"]), int(row["source"]), int(row["donor"])), []
            ).append(float(row["F_sens"]))
        by_source: dict[tuple[int, int], list[tuple[float, int]]] = {}
        for (graph, source, _donor), values in events.items():
            by_source.setdefault((graph, source), []).append((float(np.sum(values)), len(values)))
        by_graph: dict[int, list[tuple[float, float]]] = {}
        for (graph, _source), donor_values in by_source.items():
            by_graph.setdefault(graph, []).append(
                (
                    float(np.mean([value[0] for value in donor_values])),
                    float(np.mean([value[1] for value in donor_values])),
                )
            )
        graph_means = []
        for source_values in by_graph.values():
            total = float(np.sum([value[0] for value in source_values]))
            count = float(np.sum([value[1] for value in source_values]))
            if count > 0:
                graph_means.append(total / count)
        if graph_means:
            estimates[position] = float(trimmed_mean(graph_means, 0.20, axis=0))
    finite = np.isfinite(estimates) & (estimates >= 0.0)
    if normalise and np.any(finite) and float(np.sum(estimates[finite])) > 1.0e-12:
        estimates[finite] /= float(np.sum(estimates[finite]))
    return labels, estimates


def matched_strength_profile_rows(models: Sequence[SpatialModel]) -> list[dict[str, Any]]:
    per_seed: list[dict[str, Any]] = []
    for model in models:
        for channel in CHANNELS:
            labels, score_values = per_opportunity_head_profile(model.score, channel)
            for label, value in zip(labels, score_values):
                per_seed.append(
                    {
                        "task": model.task,
                        "seed": int(model.seed),
                        "source": f"{channel}_score_per_opportunity",
                        "distance": label,
                        "value": float(value),
                    }
                )
            if model.carriage is not None:
                labels, response_values = raw_carriage_strength_profile(model.carriage, channel)
                for label, value in zip(labels, response_values):
                    per_seed.append(
                        {
                            "task": model.task,
                            "seed": int(model.seed),
                            "source": f"{channel}_carriage_per_carrier",
                            "distance": label,
                            "value": float(value),
                        }
                    )
    groups: dict[tuple[str, str, str], list[float]] = {}
    for row in per_seed:
        groups.setdefault((str(row["task"]), str(row["source"]), str(row["distance"])), []).append(
            float(row["value"])
        )
    output: list[dict[str, Any]] = []
    for (task, source, distance), values in sorted(groups.items()):
        mean, low, high = _mean_range(values)
        output.append(
            {
                "task": task,
                "source": source,
                "distance": distance,
                "seeds": int(np.sum(np.isfinite(np.asarray(values, dtype=np.float64)))),
                "strength_mean": mean,
                "strength_min": low,
                "strength_max": high,
            }
        )
    return output


def carriage_response_variant_rows(models: Sequence[SpatialModel]) -> list[dict[str, Any]]:
    """Return raw and jointly normalised final-state response profiles."""

    per_seed: list[dict[str, Any]] = []
    for model in models:
        if model.carriage is None:
            continue
        for channel in CHANNELS:
            labels, raw_values = raw_carriage_strength_profile(
                model.carriage, channel, normalise=False
            )
            for label, value in zip(labels, raw_values):
                per_seed.append(
                    {
                        "task": model.task,
                        "seed": int(model.seed),
                        "channel": channel,
                        "variant": "raw",
                        "distance": label,
                        "value": float(value),
                    }
                )
    groups: dict[tuple[str, str, str, str], list[float]] = {}
    for row in per_seed:
        groups.setdefault(
            (
                str(row["task"]),
                str(row["channel"]),
                str(row["variant"]),
                str(row["distance"]),
            ),
            [],
        ).append(float(row["value"]))
    output: list[dict[str, Any]] = []
    for (task, channel, variant, distance), values in sorted(groups.items()):
        mean, low, high = _mean_range(values)
        output.append(
            {
                "task": task,
                "channel": channel,
                "variant": variant,
                "distance": distance,
                "seeds": int(np.sum(np.isfinite(np.asarray(values, dtype=np.float64)))),
                "response_mean": mean,
                "response_min": low,
                "response_max": high,
            }
        )
    return output + _joint_normalised_response_rows(output)


def _joint_normalised_response_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalise each architecture with one denominator shared by both channels."""

    raw_rows = [row for row in rows if str(row.get("variant")) == "raw"]
    denominators: dict[str, float] = {}
    for task in {str(row["task"]) for row in raw_rows}:
        values = np.asarray(
            [
                float(row.get("response_mean", np.nan))
                for row in raw_rows
                if str(row["task"]) == task
            ],
            dtype=np.float64,
        )
        finite = np.isfinite(values) & (values >= 0.0)
        denominators[task] = float(np.sum(values[finite])) if np.any(finite) else 0.0

    output: list[dict[str, Any]] = []
    for row in raw_rows:
        task = str(row["task"])
        denominator = denominators.get(task, 0.0)
        normalised = dict(row)
        normalised["variant"] = "normalised"
        for field in ("response_mean", "response_min", "response_max"):
            value = float(row.get(field, np.nan))
            normalised[field] = (
                value / denominator
                if denominator > 1.0e-12 and np.isfinite(value)
                else float("nan")
            )
        output.append(normalised)
    return output


def _plot_spatial_organisation(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    figure, axes = plt.subplots(
        2,
        len(spec.tasks),
        figsize=(3.65 * len(spec.tasks), 7.4),
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
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _add_figure_legend(figure, handles, labels, ncol=3)
    _scale_figure_text(figure)
    outputs.extend(_save_figure(figure, figures_dir, "01_spatial_organisation"))
    plt.close(figure)

    distance_figure, distance_axes = plt.subplots(
        1,
        len(spec.tasks),
        figsize=(3.65 * len(spec.tasks), 3.9),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for column, task in enumerate(spec.tasks):
        selected = sorted(
            [row for row in rows if str(row["task"]) == task],
            key=lambda row: int(row["layer"]),
        )
        layers = np.asarray([int(row["layer"]) for row in selected])
        axis = distance_axes[0, column]
        for source, (colour, marker, label) in source_style.items():
            mean = np.asarray(
                [float(row[f"{source}_expected_distance_mean"]) for row in selected]
            )
            low = np.asarray(
                [float(row[f"{source}_expected_distance_min"]) for row in selected]
            )
            high = np.asarray(
                [float(row[f"{source}_expected_distance_max"]) for row in selected]
            )
            if not np.isfinite(mean).any():
                continue
            axis.plot(layers, mean, color=colour, marker=marker, label=label)
            axis.fill_between(layers, low, high, color=colour, alpha=0.14, linewidth=0)
        axis.set_title(spec.labels[task])
        axis.set_xlabel("layer")
        if column == 0:
            axis.set_ylabel("expected graph distance")
    distance_figure.suptitle(
        f"{spec.name.upper()}: expected graph distance across architectures"
    )
    handles, labels = distance_axes[0, 0].get_legend_handles_labels()
    _add_figure_legend(distance_figure, handles, labels, ncol=3)
    _scale_figure_text(distance_figure)
    outputs.extend(
        _save_figure(
            distance_figure,
            figures_dir,
            "01b_expected_graph_distance",
        )
    )
    plt.close(distance_figure)
    return outputs


def _plot_expected_graph_distance(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    """Plot only the expected-distance row of the spatial organisation figure."""

    import matplotlib.pyplot as plt

    source_style = {
        "semantic": ("#0072B2", "o", "semantic score"),
        "structural": ("#D55E00", "s", "structural score"),
        "attention": ("#009E73", "^", "attention mass"),
    }
    figure, axes = plt.subplots(
        1,
        len(spec.tasks),
        figsize=(3.65 * len(spec.tasks), 3.9),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for column, task in enumerate(spec.tasks):
        selected = sorted(
            [row for row in rows if str(row["task"]) == task],
            key=lambda row: int(row["layer"]),
        )
        layers = np.asarray([int(row["layer"]) for row in selected])
        axis = axes[0, column]
        for source, (colour, marker, label) in source_style.items():
            mean = np.asarray(
                [float(row[f"{source}_expected_distance_mean"]) for row in selected]
            )
            low = np.asarray(
                [float(row[f"{source}_expected_distance_min"]) for row in selected]
            )
            high = np.asarray(
                [float(row[f"{source}_expected_distance_max"]) for row in selected]
            )
            if not np.isfinite(mean).any():
                continue
            axis.plot(layers, mean, color=colour, marker=marker, label=label)
            axis.fill_between(layers, low, high, color=colour, alpha=0.14, linewidth=0)
        axis.set_title(spec.labels[task])
        axis.set_xlabel("layer")
        if column == 0:
            axis.set_ylabel("expected graph distance")
    figure.suptitle(f"{spec.name.upper()}: expected graph distance across architectures")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _add_figure_legend(figure, handles, labels, ncol=3)
    _scale_figure_text(figure)
    return _save_figure(figure, figures_dir, "01b_expected_graph_distance")


def _plot_specialisation_landscapes(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        2,
        len(spec.tasks),
        figsize=(3.65 * len(spec.tasks), 7.4),
        squeeze=False,
        constrained_layout=True,
    )
    markers = ("o", "s", "^")
    maximum_layer = max(int(row["layer"]) for row in rows)
    colour_map = plt.get_cmap("viridis")
    colour_norm = plt.Normalize(0, maximum_layer)
    joint_limits = _padded_limits(
        [float(row["joint_sensitivity"]) for row in rows], include_zero=True
    )
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
                axis.set_xlabel("semantic score")
                if column == 0:
                    axis.set_ylabel("structural score")
            else:
                axis.axvline(0.0, color="#999999", linewidth=0.8, linestyle="--")
                axis.set_xlim(-1.04, 1.04)
                axis.set_ylim(joint_limits)
                axis.set_xlabel(r"relative selectivity $D_{\mathrm{rel}}$")
                if column == 0:
                    axis.set_ylabel(r"joint sensitivity $J$")
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, label="layer", shrink=0.82, pad=0.01)
    seed_handles = [
        Line2D([], [], marker=markers[index], linestyle="", color="#555555", label=f"seed {seed}")
        for index, seed in enumerate(sorted({int(row["seed"]) for row in rows}))
    ]
    figure.suptitle(f"{spec.name.upper()}: head specialisation landscapes")
    _add_figure_legend(
        figure,
        seed_handles,
        [handle.get_label() for handle in seed_handles],
        ncol=len(seed_handles),
    )
    _scale_figure_text(figure)
    return _save_figure(figure, figures_dir, "02_specialisation_landscapes")


def _plot_specialisation_selectivity_landscapes(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    """Plot the selectivity--sensitivity row of the specialisation landscapes alone."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        1,
        len(spec.tasks),
        figsize=(3.65 * len(spec.tasks), 4.0),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    markers = ("o", "s", "^")
    maximum_layer = max(int(row["layer"]) for row in rows)
    colour_map = plt.get_cmap("viridis")
    colour_norm = plt.Normalize(0, maximum_layer)
    joint_limits = _padded_limits(
        [float(row["joint_sensitivity"]) for row in rows], include_zero=True
    )
    scatter = None
    for column, task in enumerate(spec.tasks):
        axis = axes[0, column]
        selected = [row for row in rows if str(row["task"]) == task]
        for seed_index, seed in enumerate(sorted({int(row["seed"]) for row in selected})):
            seed_rows = [row for row in selected if int(row["seed"]) == seed]
            scatter = axis.scatter(
                [float(row["selectivity"]) for row in seed_rows],
                [float(row["joint_sensitivity"]) for row in seed_rows],
                c=[int(row["layer"]) for row in seed_rows],
                cmap=colour_map,
                norm=colour_norm,
                marker=markers[seed_index % len(markers)],
                s=24,
                alpha=0.78,
                linewidths=0,
            )
        axis.axvline(0.0, color="#999999", linewidth=0.8, linestyle="--")
        axis.set_xlim(-1.04, 1.04)
        axis.set_ylim(joint_limits)
        axis.set_title(spec.labels[task])
        axis.set_xlabel(r"relative selectivity $D_{\mathrm{rel}}$")
        if column == 0:
            axis.set_ylabel(r"joint sensitivity $J$")
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, label="layer", shrink=0.82, pad=0.01)
    seed_handles = [
        Line2D([], [], marker=markers[index], linestyle="", color="#555555", label=f"seed {seed}")
        for index, seed in enumerate(sorted({int(row["seed"]) for row in rows}))
    ]
    figure.suptitle(f"{spec.name.upper()}: joint sensitivity and relative selectivity")
    _add_figure_legend(
        figure,
        seed_handles,
        [handle.get_label() for handle in seed_handles],
        ncol=len(seed_handles),
    )
    _scale_figure_text(figure)
    return _save_figure(
        figure,
        figures_dir,
        "02b_specialisation_landscapes_selectivity",
    )


def refresh_specialisation_landscape_variant(
    output_dir: Path,
    *,
    dataset: str,
) -> dict[str, Any]:
    """Build the single-row specialisation companion from the saved head table."""

    output_dir = Path(output_dir)
    head_table = output_dir / "head_metrics.csv"
    if not head_table.is_file():
        raise FileNotFoundError(f"missing saved head table: {head_table}")
    with head_table.open("r", encoding="utf-8", newline="") as handle:
        head_rows = list(csv.DictReader(handle))
    if not head_rows:
        raise ValueError(f"saved head table is empty: {head_table}")

    figures = _plot_specialisation_selectivity_landscapes(
        head_rows,
        dataset_spec(dataset),
        output_dir / "figures",
    )
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["specialisation_landscape_variant_version"] = (
        SPECIALISATION_LANDSCAPE_VARIANT_VERSION
    )
    manifest_figures = [str(path) for path in manifest.get("figures", ())]
    for path in figures:
        if str(path) not in manifest_figures:
            manifest_figures.append(str(path))
    manifest["figures"] = manifest_figures
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "specialisation_landscape_variant_version": (
            SPECIALISATION_LANDSCAPE_VARIANT_VERSION
        ),
        "figures": [str(path) for path in figures],
    }


def _plot_dense_one_hop_attention_landscape(
    rows: Sequence[Mapping[str, Any]],
    spec: DatasetSpec,
    figures_dir: Path,
    *,
    threshold: float = 0.75,
) -> list[Path]:
    """Highlight dense-model heads whose attention is concentrated at one hop."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    task = spec.tasks[-1]
    selected = [
        row
        for row in rows
        if str(row["task"]) == task
        and np.isfinite(float(row.get("selectivity", np.nan)))
        and np.isfinite(float(row.get("joint_sensitivity", np.nan)))
    ]
    if not selected or not any(
        np.isfinite(float(row.get("attention_hop1_mass", np.nan))) for row in selected
    ):
        return []

    figure, axis = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    markers = ("o", "s", "^")
    maximum_layer = max(int(row["layer"]) for row in selected)
    colour_map = plt.get_cmap("viridis")
    colour_norm = plt.Normalize(0, maximum_layer)
    scatter = None
    seeds = sorted({int(row["seed"]) for row in selected})
    for seed_index, seed in enumerate(seeds):
        seed_rows = [row for row in selected if int(row["seed"]) == seed]
        for focused, alpha, size in ((False, 0.10, 26), (True, 0.90, 34)):
            subset = [
                row
                for row in seed_rows
                if bool(
                    np.isfinite(float(row.get("attention_hop1_mass", np.nan)))
                    and float(row["attention_hop1_mass"]) >= float(threshold)
                )
                == focused
            ]
            if not subset:
                continue
            scatter = axis.scatter(
                [float(row["selectivity"]) for row in subset],
                [float(row["joint_sensitivity"]) for row in subset],
                c=[int(row["layer"]) for row in subset],
                cmap=colour_map,
                norm=colour_norm,
                marker=markers[seed_index % len(markers)],
                s=size,
                alpha=alpha,
                linewidths=0,
                zorder=3 if focused else 2,
            )
    axis.axvline(0.0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    axis.set_xlim(-1.04, 1.04)
    axis.set_ylim(
        _padded_limits(
            [float(row["joint_sensitivity"]) for row in selected],
            include_zero=True,
        )
    )
    axis.set_xlabel(r"relative selectivity $D_{\mathrm{rel}}$")
    axis.set_ylabel(r"joint sensitivity $J$")
    axis.grid(alpha=0.16, linewidth=0.6)
    focused_count = sum(
        np.isfinite(float(row.get("attention_hop1_mass", np.nan)))
        and float(row["attention_hop1_mass"]) >= float(threshold)
        for row in selected
    )
    axis.set_title(
        f"{focused_count}/{len(selected)} heads at or above {100 * threshold:.0f}% "
        "attention at one hop",
        fontsize=9,
    )
    handles = [
        Line2D(
            [],
            [],
            marker=markers[index],
            linestyle="",
            color="#555555",
            label=f"seed {seed}",
        )
        for index, seed in enumerate(seeds)
    ]
    handles.extend(
        (
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color="#333333",
                alpha=0.90,
                label=rf"$\geq {100 * threshold:.0f}\%$ attention at 1 hop",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color="#999999",
                alpha=0.25,
                label=rf"$< {100 * threshold:.0f}\%$ attention at 1 hop",
            ),
        )
    )
    if scatter is not None:
        figure.colorbar(scatter, ax=axis, label="layer", pad=0.02)
    figure.suptitle(
        f"{spec.name.upper()}: dense GRIT one-hop attention in the specialisation landscape"
    )
    _add_figure_legend(
        figure,
        handles,
        [handle.get_label() for handle in handles],
        ncol=2,
    )
    _scale_figure_text(figure)
    return _save_figure(
        figure,
        figures_dir,
        "02b_dense_one_hop_attention_specialisation",
    )


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


def _seed_mean_head_matrix(
    rows: Sequence[Mapping[str, Any]],
    task: str,
    seeds: Sequence[int],
    field: str,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Average a head field by layer/head coordinate over independent seeds."""

    requested_seeds = {int(seed) for seed in seeds}
    selected = [
        row
        for row in rows
        if str(row["task"]) == task and int(row["seed"]) in requested_seeds
    ]
    layers = sorted({int(row["layer"]) for row in selected})
    heads = sorted({int(row["head"]) for row in selected})
    matrix = np.full((len(layers), len(heads)), np.nan)
    values: dict[tuple[int, int], list[float]] = {}
    for row in selected:
        value = float(row.get(field, np.nan))
        if np.isfinite(value):
            values.setdefault((int(row["layer"]), int(row["head"])), []).append(value)
    layer_index = {value: index for index, value in enumerate(layers)}
    head_index = {value: index for index, value in enumerate(heads)}
    for (layer, head), cell_values in values.items():
        matrix[layer_index[layer], head_index[head]] = float(np.mean(cell_values))
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
        figsize=(3.55 * len(spec.tasks), 2.75 * len(seeds)),
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
        figure.colorbar(
            image,
            ax=axes,
            label="expected score reach $-$ attention reach (hops)",
            pad=0.01,
        )
    figure.suptitle(
        f"{spec.name.upper()}: expected {channel} score reach minus attention reach"
    )
    _scale_figure_text(
        figure,
        preserve_tick_axes=tuple(axis for row in axes for axis in row),
    )
    return _save_figure(figure, figures_dir, stem)


def _plot_seed_mean_reach_gap_heatmap(
    rows: Sequence[Mapping[str, Any]],
    spec: DatasetSpec,
    seeds: Sequence[int],
    figures_dir: Path,
    *,
    channel: str,
    stem: str,
) -> list[Path]:
    """Plot one coordinate-wise seed-mean reach-gap heatmap per architecture."""

    import matplotlib.pyplot as plt

    field = f"{channel}_attention_reach_gap"
    matrices = [
        _seed_mean_head_matrix(rows, task, seeds, field)[0] for task in spec.tasks
    ]
    finite_parts = [matrix[np.isfinite(matrix)] for matrix in matrices]
    finite = (
        np.concatenate([values for values in finite_parts if values.size])
        if any(values.size for values in finite_parts)
        else np.asarray([], dtype=np.float64)
    )
    limit = float(np.quantile(np.abs(finite), 0.99)) if finite.size else 1.0
    limit = max(limit, 1.0e-6)
    figure, axes = plt.subplots(
        1,
        len(spec.tasks),
        figsize=(3.55 * len(spec.tasks), 4.6),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for column, task in enumerate(spec.tasks):
        matrix, layers, heads = _seed_mean_head_matrix(rows, task, seeds, field)
        axis = axes[0, column]
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
        axis.set_title(spec.labels[task], fontsize=10)
        axis.set_xlabel("head")
        if column == 0:
            axis.set_ylabel("layer")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            label="score reach $-$ attention reach (hops)",
            pad=0.01,
        )
    figure.suptitle(
        f"{spec.name.upper()}: seed-mean {channel} score reach minus attention reach"
    )
    _scale_figure_text(
        figure,
        preserve_tick_axes=tuple(axis for row in axes for axis in row),
    )
    return _save_figure(figure, figures_dir, stem)


def _plot_distance_alignment(
    rows: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    spec: DatasetSpec,
    figures_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        1,
        len(spec.tasks),
        figsize=(3.65 * len(spec.tasks), 5.8),
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
            seed_rows = [
                row
                for row in selected
                if int(row["seed"]) == seed
                and np.isfinite(float(row["semantic_expected_distance"]))
                and np.isfinite(float(row["structural_expected_distance"]))
            ]
            if not seed_rows:
                continue
            sizes = np.asarray(
                [
                    max(float(row["joint_sensitivity"]), 0.0)
                    if np.isfinite(float(row["joint_sensitivity"]))
                    else 0.0
                    for row in seed_rows
                ]
            )
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
        axis.set_xlabel("semantic expected distance")
        if column == 0:
            axis.set_ylabel("structural expected distance")
        record = summary_lookup.get(task)
        title = spec.labels[task]
        if record is not None:
            title += (
                "\n"
                + rf"$\rho$={float(record['spearman_rho']):.2f}; peak="
                + f"{100 * float(record['same_peak_fraction']):.0f}%"
            )
        axis.set_title(title)
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, label="layer", shrink=0.82, pad=0.01)
    seed_handles = [
        Line2D([], [], marker=markers[index], linestyle="", color="#555555", label=f"seed {seed}")
        for index, seed in enumerate(sorted({int(row["seed"]) for row in rows}))
    ]
    figure.suptitle(
        f"{spec.name.upper()}: semantic and structural distance alignment"
    )
    _add_figure_legend(
        figure,
        seed_handles,
        [handle.get_label() for handle in seed_handles],
        ncol=len(seed_handles),
    )
    _scale_figure_text(figure)
    return _save_figure(figure, figures_dir, "05_semantic_structural_distance_alignment")


def refresh_distance_alignment(
    output_dir: Path,
    *,
    dataset: str,
) -> dict[str, Any]:
    """Rebuild the all-head distance-alignment outputs from the saved head table."""

    output_dir = Path(output_dir)
    head_table = output_dir / "head_metrics.csv"
    if not head_table.is_file():
        raise FileNotFoundError(f"missing saved head table: {head_table}")
    with head_table.open("r", encoding="utf-8", newline="") as handle:
        head_rows = list(csv.DictReader(handle))
    if not head_rows:
        raise ValueError(f"saved head table is empty: {head_table}")

    spec = dataset_spec(dataset)
    alignment_rows = alignment_summary_rows(head_rows)
    alignment_table = output_dir / "semantic_structural_alignment.csv"
    _write_csv(alignment_table, alignment_rows)
    figures = _plot_distance_alignment(
        head_rows,
        alignment_rows,
        spec,
        output_dir / "figures",
    )

    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["distance_alignment_version"] = DISTANCE_ALIGNMENT_VERSION
    manifest_figures = [str(path) for path in manifest.get("figures", ())]
    for path in figures:
        if str(path) not in manifest_figures:
            manifest_figures.append(str(path))
    manifest["figures"] = manifest_figures
    manifest_tables = [str(path) for path in manifest.get("tables", ())]
    if str(alignment_table) not in manifest_tables:
        manifest_tables.append(str(alignment_table))
    manifest["tables"] = manifest_tables
    interpretation = dict(manifest.get("interpretation", {}))
    interpretation["alignment"] = (
        "Spearman expected-distance co-variation plus coarse peak-bin agreement "
        "across every head with finite semantic and structural distance profiles"
    )
    manifest["interpretation"] = interpretation
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "distance_alignment_version": DISTANCE_ALIGNMENT_VERSION,
        "figures": [str(path) for path in figures],
        "table": str(alignment_table),
        "heads": sum(int(row["heads"]) for row in alignment_rows),
    }


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
    figure.suptitle(f"{spec.name.upper()}: virtual-node allocation")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _add_figure_legend(figure, handles, labels, ncol=3)
    # This figure previously missed the shared publication scaling pass, so it
    # receives the requested 25% uplift directly rather than the cumulative
    # scale used by figures that were already enlarged once.
    _scale_figure_text(figure, factor=1.25)
    return _save_figure(figure, figures_dir, "06_virtual_node_allocation")


def _distance_order(label: str) -> tuple[int, str]:
    text = str(label).strip().lower().replace("–", "-")
    if text == "virtual":
        return 10_000, text
    try:
        return int(float(text)), text
    except ValueError:
        pass
    for separator in ("-", "+"):
        if separator in text:
            try:
                return int(float(text.split(separator, 1)[0])), text
            except ValueError:
                break
    return 20_000, text


def _plot_population_profiles(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    if not any(str(row["source"]).endswith("_carriage") for row in rows):
        return []
    figure, axes = plt.subplots(
        2,
        len(spec.tasks),
        figsize=(3.65 * len(spec.tasks), 7.4),
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
                (
                    f"{channel}_carriage",
                    "#222222",
                    "s",
                    "final-state response allocation",
                ),
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
                axis.set_ylabel(f"{channel}\nprofile mass")
    figure.suptitle(
        f"{spec.name.upper()}: head-score mass and final-state response allocation"
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _add_figure_legend(figure, handles, labels, ncol=2)
    _scale_figure_text(figure)
    return _save_figure(figure, figures_dir, "07_score_and_final_state_response")


def _plot_matched_strength_profiles(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    """Match score and final-state curves on a per-opportunity/per-carrier basis."""

    import matplotlib.pyplot as plt

    if not any(str(row["source"]).endswith("_carriage_per_carrier") for row in rows):
        return []
    figure, axes = plt.subplots(
        2,
        len(spec.tasks),
        figsize=(3.65 * len(spec.tasks), 7.4),
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, channel in enumerate(CHANNELS):
        for column, task in enumerate(spec.tasks):
            axis = axes[row_index, column]
            selected = [row for row in rows if str(row["task"]) == task]
            sources = (
                f"{channel}_score_per_opportunity",
                f"{channel}_carriage_per_carrier",
            )
            labels = sorted(
                {str(row["distance"]) for row in selected if str(row["source"]) in sources},
                key=_distance_order,
            )
            x = np.arange(len(labels))
            for source, colour, marker, label in (
                (sources[0], "#4C78A8", "o", "per-opportunity head score"),
                (sources[1], "#222222", "s", "final-state response"),
            ):
                lookup = {
                    str(row["distance"]): row for row in selected if str(row["source"]) == source
                }
                if not lookup:
                    continue
                mean = np.asarray(
                    [
                        float(lookup[value]["strength_mean"]) if value in lookup else np.nan
                        for value in labels
                    ]
                )
                low = np.asarray(
                    [
                        float(lookup[value]["strength_min"]) if value in lookup else np.nan
                        for value in labels
                    ]
                )
                high = np.asarray(
                    [
                        float(lookup[value]["strength_max"]) if value in lookup else np.nan
                        for value in labels
                    ]
                )
                axis.plot(x, mean, color=colour, marker=marker, label=label)
                axis.fill_between(x, low, high, color=colour, alpha=0.14, linewidth=0)
            axis.set_xticks(x, labels)
            axis.set_title(spec.labels[task], fontsize=10)
            axis.set_xlabel("distance")
            if column == 0:
                axis.set_ylabel(f"{channel}\nprofile strength")
    figure.suptitle(
        f"{spec.name.upper()}: opportunity-matched head scores and final-state response"
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _add_figure_legend(figure, handles, labels, ncol=2)
    _scale_figure_text(figure)
    return _save_figure(figure, figures_dir, "08_matched_score_and_final_state_response")


def _plot_carriage_response_variants(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    """Separate the spatial shape and raw magnitude of per-carrier response."""

    import matplotlib.pyplot as plt

    if not rows:
        return []
    figure, axes = plt.subplots(
        2,
        len(spec.tasks),
        figsize=(3.65 * len(spec.tasks), 7.4),
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )
    channel_styles = {
        "semantic": ("#0072B2", "o", "semantic"),
        "structural": ("#D55E00", "s", "structural"),
    }
    for row_index, variant in enumerate(("normalised", "raw")):
        for column, task in enumerate(spec.tasks):
            axis = axes[row_index, column]
            selected = [
                row for row in rows if str(row["task"]) == task and str(row["variant"]) == variant
            ]
            labels = _observed_response_distance_labels(rows, variant=variant)
            x = np.arange(len(labels))
            for channel, (colour, marker, label) in channel_styles.items():
                lookup = {
                    str(row["distance"]): row for row in selected if str(row["channel"]) == channel
                }
                if not lookup:
                    continue
                mean = np.asarray(
                    [
                        float(lookup[value]["response_mean"]) if value in lookup else np.nan
                        for value in labels
                    ]
                )
                low = np.asarray(
                    [
                        float(lookup[value]["response_min"]) if value in lookup else np.nan
                        for value in labels
                    ]
                )
                high = np.asarray(
                    [
                        float(lookup[value]["response_max"]) if value in lookup else np.nan
                        for value in labels
                    ]
                )
                axis.plot(x, mean, color=colour, marker=marker, label=label)
                axis.fill_between(x, low, high, color=colour, alpha=0.14, linewidth=0)
            axis.set_xticks(x, labels)
            axis.set_title(spec.labels[task], fontsize=10)
            axis.set_xlabel("distance")
            if column == 0:
                axis.set_ylabel(
                    "final-state response profile"
                    if variant == "normalised"
                    else "mean final-state response per carrier"
                )
            if variant == "raw":
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    figure.suptitle(f"{spec.name.upper()}: per-carrier final-state response shape and magnitude")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _add_figure_legend(figure, handles, labels, ncol=2)
    _scale_figure_text(figure)
    return _save_figure(figure, figures_dir, "09_final_state_response_variants")


def _plot_final_state_response(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    """Compare raw per-carrier final-state responses across architectures."""

    import matplotlib.pyplot as plt

    selected_rows = [row for row in rows if str(row["variant"]) == "raw"]
    if not selected_rows:
        return []
    figure, axes = plt.subplots(
        1,
        len(spec.tasks),
        figsize=(3.65 * len(spec.tasks), 4.0),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    styles = {
        "semantic": ("#0072B2", "o", "semantic"),
        "structural": ("#D55E00", "s", "structural"),
    }
    for column, task in enumerate(spec.tasks):
        axis = axes[0, column]
        task_rows = [row for row in selected_rows if str(row["task"]) == task]
        labels = _observed_response_distance_labels(rows, variant="raw")
        x = np.arange(len(labels))
        for channel, (colour, marker, label) in styles.items():
            lookup = {
                str(row["distance"]): row
                for row in task_rows
                if str(row["channel"]) == channel
            }
            if not lookup:
                continue
            mean = np.asarray(
                [float(lookup[value]["response_mean"]) if value in lookup else np.nan for value in labels]
            )
            low = np.asarray(
                [float(lookup[value]["response_min"]) if value in lookup else np.nan for value in labels]
            )
            high = np.asarray(
                [float(lookup[value]["response_max"]) if value in lookup else np.nan for value in labels]
            )
            axis.plot(x, mean, color=colour, marker=marker, label=label)
            axis.fill_between(x, low, high, color=colour, alpha=0.14, linewidth=0)
        axis.set_xticks(x, labels)
        axis.set_title(spec.labels[task])
        axis.set_xlabel("distance")
        axis.grid(axis="y", alpha=0.16, linewidth=0.6)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        if column == 0:
            axis.set_ylabel("Final-state response")
    figure.suptitle(f"{spec.name.upper()}: final-state response across architectures")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _add_figure_legend(figure, handles, labels, ncol=2)
    _scale_figure_text(figure)
    return _save_figure(figure, figures_dir, "09b_final_state_response")


def _observed_response_distance_labels(
    rows: Sequence[Mapping[str, Any]],
    *,
    variant: str,
) -> list[str]:
    """Return only distance categories observed by at least one architecture."""

    selected = [row for row in rows if str(row.get("variant")) == variant]
    labels = sorted({str(row["distance"]) for row in selected}, key=_distance_order)
    observed: list[str] = []
    for label in labels:
        label_rows = [row for row in selected if str(row["distance"]) == label]
        available = False
        for row in label_rows:
            try:
                seed_count = int(float(row.get("seeds", 0)))
            except (TypeError, ValueError):
                seed_count = 0
            value = float(row.get("response_mean", np.nan))
            if np.isfinite(value) and (seed_count > 0 or "seeds" not in row):
                available = True
                break
        if available:
            observed.append(label)
    return observed


def _plot_overlaid_final_state_response(
    rows: Sequence[Mapping[str, Any]],
    spec: DatasetSpec,
    figures_dir: Path,
    *,
    variant: str,
    stem: str,
) -> list[Path]:
    """Overlay architecture curves in separate semantic and structural panels."""

    import matplotlib.pyplot as plt

    selected_rows = [row for row in rows if str(row.get("variant")) == variant]
    if not selected_rows:
        return []
    labels = _observed_response_distance_labels(rows, variant=variant)
    if not labels:
        return []
    x = np.arange(len(labels))
    colours = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(spec.tasks)))
    markers = ("o", "s", "^", "D", "P")
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.4),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for column, channel in enumerate(CHANNELS):
        axis = axes[0, column]
        for task_index, task in enumerate(spec.tasks):
            lookup = {
                str(row["distance"]): row
                for row in selected_rows
                if str(row["task"]) == task and str(row["channel"]) == channel
            }
            if not lookup:
                continue
            mean = np.asarray(
                [
                    float(lookup[label]["response_mean"])
                    if label in lookup
                    else np.nan
                    for label in labels
                ]
            )
            low = np.asarray(
                [
                    float(lookup[label]["response_min"])
                    if label in lookup
                    else np.nan
                    for label in labels
                ]
            )
            high = np.asarray(
                [
                    float(lookup[label]["response_max"])
                    if label in lookup
                    else np.nan
                    for label in labels
                ]
            )
            colour = colours[task_index]
            axis.plot(
                x,
                mean,
                color=colour,
                marker=markers[task_index % len(markers)],
                linewidth=1.7,
                label=spec.labels[task],
            )
            axis.fill_between(x, low, high, color=colour, alpha=0.10, linewidth=0)
        axis.set_xticks(x, labels)
        if len(labels) > 7:
            plt.setp(
                axis.get_xticklabels(),
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
        axis.set_title(channel.capitalize())
        axis.set_xlabel("graph distance")
        axis.grid(axis="y", alpha=0.16, linewidth=0.6)
        if column == 0:
            axis.set_ylabel(
                "Normalised final-state response"
                if variant == "normalised"
                else "Final-state response"
            )
        if variant == "raw":
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    if variant == "normalised":
        figure.suptitle(
            f"{spec.name.upper()}: normalised final-state response by architecture"
        )
    else:
        figure.suptitle(f"{spec.name.upper()}: final-state response by architecture")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    _add_figure_legend(
        figure,
        handles,
        legend_labels,
        ncol=min(len(spec.tasks), 5),
    )
    _scale_figure_text(figure)
    return _save_figure(figure, figures_dir, stem)


def refresh_presentation_variants(
    output_dir: Path,
    *,
    dataset: str,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """Build seed-mean and overlaid companions from the saved analysis tables."""

    output_dir = Path(output_dir)
    head_table = output_dir / "head_metrics.csv"
    response_table = output_dir / "final_state_response_variants.csv"
    missing = [path for path in (head_table, response_table) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing saved Chapter 6 table(s): " + ", ".join(str(path) for path in missing)
        )
    with head_table.open("r", encoding="utf-8", newline="") as handle:
        head_rows = list(csv.DictReader(handle))
    with response_table.open("r", encoding="utf-8", newline="") as handle:
        response_rows = list(csv.DictReader(handle))
    if not head_rows:
        raise ValueError(f"saved head table is empty: {head_table}")
    if not response_rows:
        raise ValueError(f"saved final-state response table is empty: {response_table}")

    # Older saved tables normalised semantic and structural profiles separately.
    # Rebuild the presentation rows from their retained raw summaries so the two
    # channels share one architecture-level denominator; no scientific cache is
    # needed for this migration.
    raw_response_rows = [
        row for row in response_rows if str(row.get("variant")) == "raw"
    ]
    response_rows = raw_response_rows + _joint_normalised_response_rows(
        raw_response_rows
    )
    _write_csv(response_table, response_rows)

    spec = dataset_spec(dataset)
    figures_dir = output_dir / "figures"
    figures: list[Path] = []
    figures.extend(
        _plot_seed_mean_reach_gap_heatmap(
            head_rows,
            spec,
            seeds,
            figures_dir,
            channel="semantic",
            stem="03b_semantic_attention_score_reach_gap_seed_mean",
        )
    )
    figures.extend(
        _plot_seed_mean_reach_gap_heatmap(
            head_rows,
            spec,
            seeds,
            figures_dir,
            channel="structural",
            stem="04b_structural_attention_score_reach_gap_seed_mean",
        )
    )
    figures.extend(
        _plot_overlaid_final_state_response(
            response_rows,
            spec,
            figures_dir,
            variant="raw",
            stem="09c_final_state_response_by_model",
        )
    )
    figures.extend(
        _plot_overlaid_final_state_response(
            response_rows,
            spec,
            figures_dir,
            variant="normalised",
            stem="09d_normalised_final_state_response_by_model",
        )
    )

    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["presentation_variants_version"] = PRESENTATION_VARIANTS_VERSION
    manifest_figures = [str(path) for path in manifest.get("figures", ())]
    for path in figures:
        if str(path) not in manifest_figures:
            manifest_figures.append(str(path))
    manifest["figures"] = manifest_figures
    interpretation = dict(manifest.get("interpretation", {}))
    interpretation["seed_mean_reach_gap"] = (
        "coordinate-wise mean over layer/head slots across independent seeds; "
        "a model-level visual summary, not cross-seed head matching"
    )
    interpretation["overlaid_final_state_response"] = (
        "architecture means with the observed seed range; the normalised variant "
        "uses one denominator shared by semantic and structural channels, and "
        "globally empty distance categories are omitted"
    )
    manifest["interpretation"] = interpretation
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "presentation_variants_version": PRESENTATION_VARIANTS_VERSION,
        "figures": [str(path) for path in figures],
    }


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 3:
        return float("nan")
    result = spearmanr(left[finite], right[finite])
    return float(getattr(result, "statistic", result[0]))


def _plot_joint_sensitivity_ablation(
    rows: Sequence[Mapping[str, Any]], spec: DatasetSpec, figures_dir: Path
) -> list[Path]:
    """Plot clean single-head ablation impact against intervention-defined J."""

    if not rows:
        return []
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        1,
        len(spec.tasks),
        figsize=(18.25, 4.6),
        sharey=True,
        constrained_layout=True,
    )
    markers = ("o", "s", "^")
    maximum_layer = max(int(row["layer"]) for row in rows)
    colour_map = plt.get_cmap("viridis")
    colour_norm = plt.Normalize(0, maximum_layer)
    scatter = None
    for column, task in enumerate(spec.tasks):
        axis = axes[column]
        selected = [row for row in rows if str(row["task"]) == task]
        seed_rhos = []
        for seed_index, seed in enumerate(sorted({int(row["seed"]) for row in selected})):
            seed_rows = [row for row in selected if int(row["seed"]) == seed]
            x = [float(row["joint_sensitivity"]) for row in seed_rows]
            y = [float(row["prediction_movement"]) for row in seed_rows]
            seed_rhos.append((seed, _spearman(x, y)))
            scatter = axis.scatter(
                x,
                y,
                c=[int(row["layer"]) for row in seed_rows],
                cmap=colour_map,
                norm=colour_norm,
                marker=markers[seed_index % len(markers)],
                s=24,
                alpha=0.78,
                linewidths=0,
            )
        finite_rhos = _finite([rho for _, rho in seed_rhos])
        mean_rho = float(np.mean(finite_rhos)) if finite_rhos.size else float("nan")
        axis.set_title(
            spec.labels[task] + "\n" + rf"mean seed $\rho={mean_rho:.2f}$"
        )
        axis.set_xlabel(r"joint sensitivity $J$")
        axis.grid(alpha=0.18, linewidth=0.6)
        if column == 0:
            axis.set_ylabel("output change after head ablation")
    if scatter is not None:
        figure.colorbar(scatter, ax=axes, label="layer", shrink=0.82, pad=0.01)
    seed_handles = [
        Line2D([], [], marker=markers[index], linestyle="", color="#555555", label=f"seed {seed}")
        for index, seed in enumerate(sorted({int(row["seed"]) for row in rows}))
    ]
    figure.suptitle(f"{spec.name.upper()}: joint sensitivity and head-ablation impact")
    _add_figure_legend(
        figure,
        seed_handles,
        [handle.get_label() for handle in seed_handles],
        ncol=len(seed_handles),
    )
    _scale_figure_text(figure)
    return _save_figure(figure, figures_dir, "10_joint_sensitivity_head_ablation")


def run(
    canonical_root: Path,
    output_dir: Path,
    *,
    dataset: str,
    seeds: Sequence[int] = SEEDS,
    strict_inventory: bool = True,
    ablation_root: Path | None = None,
    strict_ablation: bool = False,
    trajectory_root: Path | None = None,
    strict_trajectory: bool = False,
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

    (
        head_rows,
        vnode_rows,
        profile_rows,
        matched_profile_rows,
        response_variant_rows,
        warnings,
        runs_loaded,
    ) = _load_summary_rows_streaming(canonical_root, spec, seeds)
    expected = len(spec.tasks) * len(tuple(seeds))
    if strict_inventory and runs_loaded != expected:
        raise RuntimeError(f"loaded {runs_loaded} models; expected {expected}")
    if not head_rows:
        raise FileNotFoundError("no usable score caches were found")
    if verbose:
        for warning in warnings:
            print(f"[chapter6-multiseed:warning] {warning}", flush=True)
        print(
            f"[chapter6-multiseed] {spec.name}: streamed {runs_loaded} cached runs",
            flush=True,
        )

    organisation_rows = layer_organisation_rows(head_rows)
    alignment_rows = alignment_summary_rows(head_rows)
    ablation_rows: list[dict[str, Any]] = []
    if ablation_root is not None:
        from .chapter6_clean_ablation import load_rows

        ablation_rows = load_rows(
            Path(ablation_root),
            dataset=spec.name,
            seeds=seeds,
            strict=strict_ablation,
        )
    trajectory_rows: list[dict[str, Any]] = []
    if spec.name == "zinc" and trajectory_root is not None:
        from .chapter6_score_trajectory import load_rows as load_trajectory_rows

        trajectory_rows = load_trajectory_rows(
            Path(trajectory_root),
            strict=strict_trajectory,
        )
    tables = {
        "head_metrics.csv": head_rows,
        "layer_spatial_organisation.csv": organisation_rows,
        "semantic_structural_alignment.csv": alignment_rows,
        "vnode_allocation.csv": vnode_rows,
        "population_score_carriage_profiles.csv": profile_rows,
        "matched_score_carriage_profiles.csv": matched_profile_rows,
        "final_state_response_variants.csv": response_variant_rows,
        "joint_sensitivity_head_ablation.csv": ablation_rows,
    }
    if trajectory_rows:
        tables["zinc_checkpoint_trajectory_heads.csv"] = trajectory_rows
    for filename, rows in tables.items():
        _write_csv(output_dir / filename, rows)

    figures: list[Path] = []
    figures.extend(_plot_spatial_organisation(organisation_rows, spec, figures_dir))
    figures.extend(_plot_specialisation_landscapes(head_rows, spec, figures_dir))
    figures.extend(
        _plot_specialisation_selectivity_landscapes(head_rows, spec, figures_dir)
    )
    figures.extend(
        _plot_dense_one_hop_attention_landscape(head_rows, spec, figures_dir)
    )
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
        _plot_seed_mean_reach_gap_heatmap(
            head_rows,
            spec,
            seeds,
            figures_dir,
            channel="semantic",
            stem="03b_semantic_attention_score_reach_gap_seed_mean",
        )
    )
    figures.extend(
        _plot_seed_mean_reach_gap_heatmap(
            head_rows,
            spec,
            seeds,
            figures_dir,
            channel="structural",
            stem="04b_structural_attention_score_reach_gap_seed_mean",
        )
    )
    figures.extend(
        _plot_distance_alignment(
            head_rows,
            alignment_rows,
            spec,
            figures_dir,
        )
    )
    figures.extend(_plot_vnode_allocation(vnode_rows, spec, figures_dir))
    figures.extend(_plot_population_profiles(profile_rows, spec, figures_dir))
    figures.extend(_plot_matched_strength_profiles(matched_profile_rows, spec, figures_dir))
    figures.extend(_plot_carriage_response_variants(response_variant_rows, spec, figures_dir))
    figures.extend(_plot_final_state_response(response_variant_rows, spec, figures_dir))
    figures.extend(
        _plot_overlaid_final_state_response(
            response_variant_rows,
            spec,
            figures_dir,
            variant="raw",
            stem="09c_final_state_response_by_model",
        )
    )
    figures.extend(
        _plot_overlaid_final_state_response(
            response_variant_rows,
            spec,
            figures_dir,
            variant="normalised",
            stem="09d_normalised_final_state_response_by_model",
        )
    )
    figures.extend(_plot_joint_sensitivity_ablation(ablation_rows, spec, figures_dir))
    if trajectory_rows:
        from .chapter6_score_trajectory import plot as plot_score_trajectory

        figures.extend(plot_score_trajectory(trajectory_rows, figures_dir))

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "dataset": spec.name,
        "canonical_root": str(canonical_root),
        "output_dir": str(output_dir),
        "tasks": list(spec.tasks),
        "seeds": [int(seed) for seed in seeds],
        "strict_inventory": bool(strict_inventory),
        "ablation_root": None if ablation_root is None else str(ablation_root),
        "strict_ablation": bool(strict_ablation),
        "trajectory_root": None if trajectory_root is None else str(trajectory_root),
        "strict_trajectory": bool(strict_trajectory),
        "distance_alignment_version": DISTANCE_ALIGNMENT_VERSION,
        "specialisation_landscape_variant_version": (
            SPECIALISATION_LANDSCAPE_VARIANT_VERSION
        ),
        "presentation_variants_version": PRESENTATION_VARIANTS_VERSION,
        "runs_loaded": runs_loaded,
        "warnings": warnings,
        "figures": [str(path) for path in figures],
        "tables": [str(output_dir / name) for name in tables],
        "interpretation": {
            "head_identity": (
                "primary head panels retain seed identity; the seed-mean reach-gap "
                "companion averages layer/head coordinates as a model-level visual summary"
            ),
            "seed_bands": "minimum-to-maximum range over independently trained seeds",
            "expected_distance": "molecular graph distance; virtual node is separate",
            "reach_gap": "intervention-defined score reach minus clean-attention reach",
            "alignment": (
                "Spearman expected-distance co-variation plus coarse peak-bin agreement "
                "across every head with finite semantic and structural distance profiles"
            ),
            "mass_allocation_comparison": (
                "within-head score mass versus final-state response allocation; "
                "both retain distance-shell opportunity"
            ),
            "matched_strength_comparison": (
                "per-opportunity head score versus graph-balanced final-state response per carrier; "
                "only the completed profiles are normalised for plotting"
            ),
            "response_variants": (
                "graph-balanced mean absolute response per eligible carrier within each shell, "
                "shown both in raw output units and after joint semantic--structural "
                "normalisation within each architecture"
            ),
            "head_ablation": (
                "clean single-head ablation prediction movement on 64 held-out graphs; "
                "Spearman correlations are computed and displayed separately within seed"
            ),
            "final_state_response": "learned response, not task necessity",
            "overlaid_final_state_response": (
                "architecture means with the observed seed range; the normalised variant "
                "uses one denominator shared by semantic and structural channels, and "
                "globally empty distance categories are omitted"
            ),
        },
    }
    if trajectory_rows:
        manifest["interpretation"]["checkpoint_trajectory"] = (
            "ZINC seed-0 dense and 1-hop score caches at epochs "
            "10, 100, 250, 500, 1000, and 1990; plot limits are shared "
            "across all twelve checkpoints"
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest
