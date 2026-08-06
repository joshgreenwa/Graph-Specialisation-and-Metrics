"""Permissive, cache-first exploration of spatial head organisation.

The chapter-six explorer deliberately has a lighter compatibility boundary than
the canonical methodology runner.  It reads the scientific arrays already
stored in canonical score and carriage artifacts, records any protocol
differences as warnings, and skips only the unavailable analysis.  It never
silently recomputes a score.

The primary outputs connect three spatial descriptions of a trained model:

* semantic and structural score mass within every head;
* clean attention mass by graph distance, when cached; and
* final-state Functional carriage, when cached.

Spatial width and estimation uncertainty are reported separately.  Width is
the variance of a head's normalized distance profile; uncertainty is the
standard error of its expected distance across cached held-out graphs.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .zinc_cached_rrwp_comparison import (
    TASK_ARTIFACT_ALIASES,
    TASK_LABELS,
    carriage_profile,
    group_distance,
    score_interval_width_profile,
    score_profile,
)

ANALYSIS_VERSION = "chapter6-spatial-explorer-v1"
CHANNELS = ("semantic", "structural")


@dataclass(frozen=True)
class SpatialModel:
    """Arrays selected for one task without enforcing a cache protocol."""

    task: str
    artifact_task: str
    seed: int
    score_path: Path
    score: Mapping[str, Any]
    score_metadata: Mapping[str, Any]
    model_path: Path | None
    model_record: Mapping[str, Any]
    carriage_path: Path | None
    carriage: Mapping[str, Any] | None


def _as_numpy(value: Any, *, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _numeric_distance(value: Any) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
    else:
        try:
            result = float(str(value))
        except ValueError:
            return None
    return result if np.isfinite(result) and result >= 0 and result.is_integer() else None


def _normalise(values: Any) -> np.ndarray:
    values = np.maximum(_as_numpy(values), 0.0)
    total = np.nansum(values, axis=-1, keepdims=True)
    output = np.full_like(values, np.nan, dtype=np.float64)
    np.divide(values, total, out=output, where=total > 1.0e-12)
    return output


def _load_payload(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Load a canonical or plain torch payload without making it an execution gate."""

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and "value" in payload:
        value = payload["value"]
        metadata = payload.get("metadata", {})
    else:
        value = payload
        metadata = {}
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping in {path}, found {type(value).__name__}")
    return value, metadata if isinstance(metadata, Mapping) else {}


def _candidate_records(
    roots: Sequence[Path], task: str, seed: int
) -> list[dict[str, Any]]:
    aliases = TASK_ARTIFACT_ALIASES.get(str(task), (str(task),))
    records: list[dict[str, Any]] = []
    for root_index, root in enumerate(roots):
        for alias_index, artifact_task in enumerate(aliases):
            task_dir = Path(root) / artifact_task / f"seed_{int(seed)}"
            score_path = task_dir / "cache" / "scores" / "raw.pt"
            model_path = task_dir / "model.json"
            carriage_path = task_dir / "cache" / "carriage" / "fields.pt"
            if score_path.is_file() or model_path.is_file() or carriage_path.is_file():
                records.append(
                    {
                        "task": str(task),
                        "artifact_task": artifact_task,
                        "seed": int(seed),
                        "root": str(Path(root)),
                        "root_index": root_index,
                        "alias_index": alias_index,
                        "score_path": score_path,
                        "score_exists": score_path.is_file(),
                        "model_path": model_path,
                        "model_exists": model_path.is_file(),
                        "carriage_path": carriage_path,
                        "carriage_exists": carriage_path.is_file(),
                    }
                )
    return records


def inventory(
    roots: Sequence[Path], tasks: Sequence[str], *, seed: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        candidates = _candidate_records(roots, str(task), int(seed))
        if not candidates:
            rows.append(
                {
                    "task": str(task),
                    "artifact_task": "",
                    "seed": int(seed),
                    "root": "",
                    "score_path": "",
                    "score_exists": False,
                    "model_exists": False,
                    "carriage_exists": False,
                }
            )
            continue
        for candidate in candidates:
            rows.append(
                {
                    key: (
                        str(value)
                        if isinstance(value, Path)
                        else value
                    )
                    for key, value in candidate.items()
                    if key not in {"root_index", "alias_index", "model_path", "carriage_path"}
                }
            )
    return rows


def _protocol_rank(metadata: Mapping[str, Any]) -> int:
    protocol = str(metadata.get("protocol_version", ""))
    if protocol.endswith("v4"):
        return 2
    if protocol.endswith("v3"):
        return 1
    return 0


def _validate_score_shape(score: Mapping[str, Any], path: Path) -> None:
    axis = tuple(score.get("axis", ()))
    channels = score.get("channels")
    if not axis or not isinstance(channels, Mapping):
        raise ValueError(f"{path} has no distance axis or score channels")
    shapes = []
    for channel in CHANNELS:
        values = channels.get(channel)
        if not isinstance(values, Mapping) or "heatmap_exact_head" not in values:
            raise ValueError(f"{path} has no {channel} heatmap_exact_head")
        array = _as_numpy(values["heatmap_exact_head"])
        if array.ndim != 3 or array.shape[-1] != len(axis):
            raise ValueError(
                f"{path} {channel} profile has shape {array.shape}; "
                f"expected [layer,head,{len(axis)}]"
            )
        shapes.append(array.shape)
    if shapes[0] != shapes[1]:
        raise ValueError(f"semantic and structural profile shapes differ in {path}")


def load_models(
    roots: Sequence[Path],
    tasks: Sequence[str],
    *,
    seed: int,
) -> tuple[list[SpatialModel], list[str]]:
    """Load every usable task, returning warnings instead of protocol barriers."""

    models: list[SpatialModel] = []
    warnings: list[str] = []
    for task in tasks:
        candidates = [
            candidate
            for candidate in _candidate_records(roots, str(task), int(seed))
            if candidate["score_exists"]
        ]
        loaded: list[tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
        for candidate in candidates:
            try:
                score, metadata = _load_payload(candidate["score_path"])
                _validate_score_shape(score, candidate["score_path"])
                loaded.append((candidate, score, metadata))
            except (OSError, RuntimeError, EOFError, TypeError, ValueError, KeyError) as error:
                warnings.append(
                    f"{task}: ignored {candidate['score_path']} "
                    f"({type(error).__name__}: {error})"
                )
        if not loaded:
            warnings.append(f"{task}: no usable score cache; task skipped")
            continue
        loaded.sort(
            key=lambda item: (
                -_protocol_rank(item[2]),
                int(item[0]["alias_index"]),
                int(item[0]["root_index"]),
            )
        )
        selected, score, metadata = loaded[0]
        if len(loaded) > 1:
            warnings.append(
                f"{task}: found {len(loaded)} usable score caches; selected "
                f"{selected['score_path']}"
            )
        contract = metadata.get("contract", {})
        stored_task = contract.get("task") if isinstance(contract, Mapping) else None
        if stored_task is not None and str(stored_task) != str(selected["artifact_task"]):
            warnings.append(
                f"{task}: stored task {stored_task!r} differs from path task "
                f"{selected['artifact_task']!r}; arrays retained for exploration"
            )
        model_record: Mapping[str, Any] = {}
        model_path = selected["model_path"] if selected["model_exists"] else None
        if model_path is not None:
            try:
                parsed = json.loads(model_path.read_text(encoding="utf-8"))
                if isinstance(parsed, Mapping):
                    model_record = parsed
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
                warnings.append(
                    f"{task}: could not read model.json ({type(error).__name__}: {error})"
                )
        carriage = None
        carriage_path = selected["carriage_path"] if selected["carriage_exists"] else None
        if carriage_path is not None:
            try:
                carriage, _ = _load_payload(carriage_path)
            except (OSError, RuntimeError, EOFError, TypeError, ValueError, KeyError) as error:
                warnings.append(
                    f"{task}: carriage cache unavailable ({type(error).__name__}: {error})"
                )
                carriage_path = None
        models.append(
            SpatialModel(
                task=str(task),
                artifact_task=str(selected["artifact_task"]),
                seed=int(seed),
                score_path=Path(selected["score_path"]),
                score=score,
                score_metadata=metadata,
                model_path=model_path,
                model_record=model_record,
                carriage_path=carriage_path,
                carriage=carriage,
            )
        )
    return models, warnings


def _reportable(score: Mapping[str, Any], channel: str) -> np.ndarray:
    width = len(score["axis"])
    support = score["channels"][channel].get("distance_support")
    if isinstance(support, Mapping) and "reportable" in support:
        mask = _as_numpy(support["reportable"], dtype=bool)
        if mask.shape == (width,):
            return mask
    return np.ones(width, dtype=bool)


def _head_activity(score: Mapping[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    coordinates = score.get("coordinates", {})
    joint = _field(coordinates, "joint_sensitivity")
    selectivity = _field(coordinates, "selectivity")
    if joint is None:
        semantic = _as_numpy(score["channels"]["semantic"].get("raw"))
        structural = _as_numpy(score["channels"]["structural"].get("raw"))
        joint_array = semantic + structural
    else:
        joint_array = _as_numpy(joint)
    if selectivity is None:
        semantic = _as_numpy(score["channels"]["semantic"].get("raw"))
        structural = _as_numpy(score["channels"]["structural"].get("raw"))
        denominator = semantic + structural
        selectivity_array = np.full(shape, np.nan)
        np.divide(
            semantic - structural,
            denominator,
            out=selectivity_array,
            where=denominator > 1.0e-12,
        )
    else:
        selectivity_array = _as_numpy(selectivity)
    if joint_array.shape != shape or selectivity_array.shape != shape:
        raise ValueError("head coordinate arrays do not match score profile shape")
    return joint_array, selectivity_array


def _molecular_profile(values: Any, axis: Sequence[Any], reportable: Any) -> tuple[np.ndarray, np.ndarray]:
    distances = np.asarray(
        [
            float(value) if (value := _numeric_distance(label)) is not None else np.nan
            for label in axis
        ],
        dtype=np.float64,
    )
    mask = np.isfinite(distances) & _as_numpy(reportable, dtype=bool)
    profile = _normalise(_as_numpy(values)[..., mask])
    return distances[mask], profile


def _profile_statistics(values: Any, axis: Sequence[Any], reportable: Any) -> dict[str, Any]:
    raw = np.maximum(_as_numpy(values), 0.0)
    mask = _as_numpy(reportable, dtype=bool) & np.isfinite(raw)
    if not np.any(mask) or float(np.sum(raw[mask])) <= 1.0e-12:
        return {
            "expected_distance": float("nan"),
            "peak_distance": "",
            "spatial_variance": float("nan"),
            "entropy": float("nan"),
        }
    full_profile = raw[mask] / float(np.sum(raw[mask]))
    labels = np.asarray(tuple(axis), dtype=object)[mask]
    peak = str(labels[int(np.argmax(full_profile))]).replace("_", " ")
    numeric = np.asarray(
        [
            float(value) if (value := _numeric_distance(label)) is not None else np.nan
            for label in labels
        ],
        dtype=np.float64,
    )
    numeric_mask = np.isfinite(numeric)
    if not np.any(numeric_mask) or float(np.sum(raw[mask][numeric_mask])) <= 1.0e-12:
        expected = variance = entropy = float("nan")
    else:
        molecular = raw[mask][numeric_mask]
        molecular = molecular / float(np.sum(molecular))
        distance = numeric[numeric_mask]
        expected = float(np.dot(molecular, distance))
        variance = float(np.dot(molecular, np.square(distance - expected)))
        positive = molecular[molecular > 0.0]
        entropy = float(-np.dot(positive, np.log(positive)))
    return {
        "expected_distance": expected,
        "peak_distance": peak,
        "spatial_variance": variance,
        "entropy": entropy,
    }


def _profile_alignment(
    semantic: Any,
    structural: Any,
    axis: Sequence[Any],
    reportable: Any,
) -> dict[str, float]:
    left = np.maximum(_as_numpy(semantic), 0.0)
    right = np.maximum(_as_numpy(structural), 0.0)
    mask = _as_numpy(reportable, dtype=bool) & np.isfinite(left) & np.isfinite(right)
    if not np.any(mask) or min(float(left[mask].sum()), float(right[mask].sum())) <= 1.0e-12:
        return {"overlap": float("nan"), "cosine": float("nan"), "wasserstein": float("nan"), "wasserstein_similarity": float("nan")}
    left = left[mask] / float(left[mask].sum())
    right = right[mask] / float(right[mask].sum())
    overlap = float(np.minimum(left, right).sum())
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = float(np.dot(left, right) / denominator) if denominator > 1.0e-12 else float("nan")
    labels = np.asarray(tuple(axis), dtype=object)[mask]
    numeric = np.asarray(
        [
            float(value) if (value := _numeric_distance(label)) is not None else np.nan
            for label in labels
        ]
    )
    numeric_mask = np.isfinite(numeric)
    if np.count_nonzero(numeric_mask) < 2:
        wasserstein = 0.0 if np.count_nonzero(numeric_mask) == 1 else float("nan")
        similarity = 1.0 if wasserstein == 0.0 else float("nan")
    else:
        from scipy.stats import wasserstein_distance

        l_mass = left[numeric_mask]
        r_mass = right[numeric_mask]
        l_mass = l_mass / l_mass.sum()
        r_mass = r_mass / r_mass.sum()
        support = numeric[numeric_mask]
        wasserstein = float(
            wasserstein_distance(support, support, u_weights=l_mass, v_weights=r_mass)
        )
        span = max(float(np.max(support) - np.min(support)), 1.0)
        similarity = float(max(0.0, 1.0 - wasserstein / span))
    return {
        "overlap": overlap,
        "cosine": cosine,
        "wasserstein": wasserstein,
        "wasserstein_similarity": similarity,
    }


def _expected_distance_uncertainty(
    channel_score: Mapping[str, Any], axis: Sequence[Any], reportable: Any
) -> tuple[np.ndarray | None, np.ndarray | None]:
    contributions = channel_score.get("graph_distance_contribution")
    if not isinstance(contributions, Mapping) or not contributions:
        return None, None
    values = []
    for graph_id in sorted(contributions, key=lambda value: str(value)):
        array = _as_numpy(contributions[graph_id])
        if array.ndim != 3 or array.shape[-1] != len(axis):
            continue
        distances, profile = _molecular_profile(array, axis, reportable)
        values.append(np.nansum(profile * distances, axis=-1))
    if not values:
        return None, None
    stacked = np.stack(values)
    valid = np.sum(np.isfinite(stacked), axis=0)
    mean = np.nanmean(stacked, axis=0)
    sem = np.full(mean.shape, np.nan, dtype=np.float64)
    enough = valid > 1
    sem[enough] = np.nanstd(stacked, axis=0, ddof=1)[enough] / np.sqrt(valid[enough])
    return mean, sem


def head_metrics(models: Sequence[SpatialModel]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        score = model.score
        axis = tuple(score["axis"])
        semantic = _as_numpy(score["channels"]["semantic"]["heatmap_exact_head"])
        structural = _as_numpy(score["channels"]["structural"]["heatmap_exact_head"])
        layers, heads, _ = semantic.shape
        joint, selectivity = _head_activity(score, (layers, heads))
        reportable = _reportable(score, "semantic") & _reportable(score, "structural")
        uncertainty: dict[str, np.ndarray | None] = {}
        for channel in CHANNELS:
            _, uncertainty[channel] = _expected_distance_uncertainty(
                score["channels"][channel], axis, reportable
            )
        attention = score.get("clean_attention_distance")
        attention_array = None
        if attention is not None:
            candidate = _as_numpy(attention)
            if candidate.shape == semantic.shape:
                attention_array = candidate
        for layer in range(layers):
            for head in range(heads):
                semantic_stats = _profile_statistics(
                    semantic[layer, head], axis, reportable
                )
                structural_stats = _profile_statistics(
                    structural[layer, head], axis, reportable
                )
                alignment = _profile_alignment(
                    semantic[layer, head], structural[layer, head], axis, reportable
                )
                attention_stats = (
                    _profile_statistics(attention_array[layer, head], axis, reportable)
                    if attention_array is not None
                    else {}
                )
                rows.append(
                    {
                        "task": model.task,
                        "layer": layer,
                        "head": head,
                        "joint_sensitivity": float(joint[layer, head]),
                        "selectivity": float(selectivity[layer, head]),
                        **{
                            f"semantic_{key}": value
                            for key, value in semantic_stats.items()
                        },
                        **{
                            f"structural_{key}": value
                            for key, value in structural_stats.items()
                        },
                        **alignment,
                        **{
                            f"attention_{key}": value
                            for key, value in attention_stats.items()
                        },
                        "semantic_expected_distance_sem": (
                            float(uncertainty["semantic"][layer, head])
                            if uncertainty["semantic"] is not None
                            else float("nan")
                        ),
                        "structural_expected_distance_sem": (
                            float(uncertainty["structural"][layer, head])
                            if uncertainty["structural"] is not None
                            else float("nan")
                        ),
                    }
                )
    return rows


def layer_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), int(row["layer"])), []).append(row)
    output: list[dict[str, Any]] = []
    sources = ("semantic", "structural", "attention")
    for (task, layer), group in sorted(groups.items()):
        for source in sources:
            field = f"{source}_expected_distance"
            values = np.asarray(
                [float(row.get(field, np.nan)) for row in group], dtype=np.float64
            )
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            width_field = f"{source}_spatial_variance"
            widths = np.asarray(
                [float(row.get(width_field, np.nan)) for row in group], dtype=np.float64
            )
            widths = widths[np.isfinite(widths)]
            uncertainty_field = f"{source}_expected_distance_sem"
            uncertainties = np.asarray(
                [float(row.get(uncertainty_field, np.nan)) for row in group],
                dtype=np.float64,
            )
            uncertainties = uncertainties[np.isfinite(uncertainties)]
            output.append(
                {
                    "task": task,
                    "layer": layer,
                    "source": source,
                    "heads": int(values.size),
                    "expected_distance_mean": float(np.mean(values)),
                    "expected_distance_median": float(np.median(values)),
                    "expected_distance_q1": float(np.quantile(values, 0.25)),
                    "expected_distance_q3": float(np.quantile(values, 0.75)),
                    "headwise_expected_distance_sem": (
                        float(np.std(values, ddof=1) / math.sqrt(values.size))
                        if values.size > 1
                        else float("nan")
                    ),
                    "spatial_variance_mean": (
                        float(np.mean(widths)) if widths.size else float("nan")
                    ),
                    "estimation_sem_mean": (
                        float(np.mean(uncertainties))
                        if uncertainties.size
                        else float("nan")
                    ),
                }
            )
    return output


def representative_heads(
    rows: Sequence[Mapping[str, Any]], *, activity_quantile: float = 0.25
) -> list[dict[str, Any]]:
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task"]), []).append(row)
    output: list[dict[str, Any]] = []
    for group in by_task.values():
        finite_joint = np.asarray(
            [float(row["joint_sensitivity"]) for row in group if np.isfinite(float(row["joint_sensitivity"]))]
        )
        floor = float(np.quantile(finite_joint, activity_quantile)) if finite_joint.size else -np.inf
        eligible = [
            row
            for row in group
            if float(row["joint_sensitivity"]) >= floor and np.isfinite(float(row["overlap"]))
        ]
        if not eligible:
            continue
        for role, selected in (
            ("high alignment", max(eligible, key=lambda row: float(row["overlap"]))),
            ("low alignment", min(eligible, key=lambda row: float(row["overlap"]))),
        ):
            output.append({"role": role, **dict(selected)})
    return output


def model_profiles(models: Sequence[SpatialModel]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        score = model.score
        for channel in CHANNELS:
            labels, equal_head, activity = score_profile(score, channel)
            for label, equal, weighted in zip(labels, equal_head, activity):
                rows.append(
                    {
                        "task": model.task,
                        "source": f"{channel}_score",
                        "distance": label,
                        "equal_head_mass": float(equal),
                        "activity_weighted_mass": float(weighted),
                    }
                )
            if model.carriage is not None:
                try:
                    carriage_labels, values, _, _ = carriage_profile(model.carriage, channel)
                    for label, value in zip(carriage_labels, values):
                        rows.append(
                            {
                                "task": model.task,
                                "source": f"{channel}_carriage",
                                "distance": label,
                                "equal_head_mass": float(value),
                                "activity_weighted_mass": float(value),
                            }
                        )
                except (KeyError, TypeError, ValueError, RuntimeError):
                    continue
        attention = score.get("clean_attention_distance")
        if attention is not None:
            attention = _as_numpy(attention)
            if attention.ndim == 3 and attention.shape[-1] == len(score["axis"]):
                labels, grouped = group_distance(attention, score["axis"])
                profile = np.nanmean(_normalise(grouped).reshape(-1, len(labels)), axis=0)
                profile = _normalise(profile)
                for label, value in zip(labels, profile):
                    rows.append(
                        {
                            "task": model.task,
                            "source": "attention",
                            "distance": label,
                            "equal_head_mass": float(value),
                            "activity_weighted_mass": float(value),
                        }
                    )
    return rows


def uncertainty_profiles(models: Sequence[SpatialModel]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for channel in CHANNELS:
            labels, widths, reportable = score_interval_width_profile(model.score, channel)
            for label, width, keep in zip(labels, widths, reportable):
                rows.append(
                    {
                        "task": model.task,
                        "channel": channel,
                        "distance": label,
                        "relative_ci_width": float(width),
                        "reportable": bool(keep),
                    }
                )
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
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def _task_label(task: str) -> str:
    return TASK_LABELS.get(task, task.replace("_", " ")).replace("\n", " ")


def _matrix(
    rows: Sequence[Mapping[str, Any]], task: str, field: str, *, text: bool = False
) -> tuple[np.ndarray, list[int], list[int]]:
    selected = [row for row in rows if str(row["task"]) == task]
    layers = sorted({int(row["layer"]) for row in selected})
    heads = sorted({int(row["head"]) for row in selected})
    dtype = object if text else np.float64
    fill = "" if text else np.nan
    matrix = np.full((len(layers), len(heads)), fill, dtype=dtype)
    layer_index = {value: index for index, value in enumerate(layers)}
    head_index = {value: index for index, value in enumerate(heads)}
    for row in selected:
        matrix[layer_index[int(row["layer"])], head_index[int(row["head"])]] = row.get(field, fill)
    return matrix, layers, heads


def _plot_expected_heatmaps(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2, len(tasks), figsize=(3.0 * len(tasks) + 1.0, 7.1), squeeze=False, constrained_layout=True
    )
    finite = np.asarray(
        [
            float(row[f"{channel}_expected_distance"])
            for row in rows
            for channel in CHANNELS
            if np.isfinite(float(row[f"{channel}_expected_distance"]))
        ]
    )
    upper = max(float(np.max(finite)) if finite.size else 1.0, 1.0)
    image = None
    for row_index, channel in enumerate(CHANNELS):
        for column, task in enumerate(tasks):
            axis = axes[row_index, column]
            matrix, layers, heads = _matrix(rows, task, f"{channel}_expected_distance")
            image = axis.imshow(matrix, vmin=0.0, vmax=upper, cmap="viridis", aspect="auto")
            axis.set_xticks(np.arange(len(heads)), heads)
            axis.set_yticks(np.arange(len(layers)), layers)
            axis.set_xlabel("head")
            if column == 0:
                axis.set_ylabel(f"{channel}\nlayer")
            if row_index == 0:
                axis.set_title(_task_label(task), fontsize=10)
    if image is not None:
        figure.colorbar(image, ax=axes, fraction=0.018, pad=0.012, label="expected distance")
    figure.suptitle("Per-head expected score distance")
    paths = _save_figure(figure, figures_dir, "01_expected_distance_heatmaps")
    plt.close(figure)
    return paths


def _plot_peak_heatmaps(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    all_labels = sorted(
        {
            str(row[f"{channel}_peak_distance"])
            for row in rows
            for channel in CHANNELS
            if str(row[f"{channel}_peak_distance"])
        },
        key=lambda value: (
            _numeric_distance(value) is None,
            _numeric_distance(value) if _numeric_distance(value) is not None else value,
        ),
    )
    codes = {label: index for index, label in enumerate(all_labels)}
    figure, axes = plt.subplots(
        2, len(tasks), figsize=(3.0 * len(tasks) + 1.0, 7.1), squeeze=False, constrained_layout=True
    )
    image = None
    for row_index, channel in enumerate(CHANNELS):
        for column, task in enumerate(tasks):
            axis = axes[row_index, column]
            labels, layers, heads = _matrix(rows, task, f"{channel}_peak_distance", text=True)
            matrix = np.full(labels.shape, np.nan)
            for index in np.ndindex(labels.shape):
                if str(labels[index]) in codes:
                    matrix[index] = codes[str(labels[index])]
            image = axis.imshow(
                matrix,
                vmin=-0.5,
                vmax=max(len(codes) - 0.5, 0.5),
                cmap="viridis",
                aspect="auto",
            )
            axis.set_xticks(np.arange(len(heads)), heads)
            axis.set_yticks(np.arange(len(layers)), layers)
            axis.set_xlabel("head")
            if column == 0:
                axis.set_ylabel(f"{channel}\nlayer")
            if row_index == 0:
                axis.set_title(_task_label(task), fontsize=10)
    if image is not None and codes:
        colourbar = figure.colorbar(image, ax=axes, fraction=0.018, pad=0.012)
        colourbar.set_ticks(list(codes.values()))
        colourbar.set_ticklabels(list(codes))
        colourbar.set_label("peak distance")
    figure.suptitle("Per-head peak score distance")
    paths = _save_figure(figure, figures_dir, "02_peak_distance_heatmaps")
    plt.close(figure)
    return paths


def _plot_alignment_heatmaps(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(tasks), figsize=(3.0 * len(tasks) + 1.0, 3.9), squeeze=False, constrained_layout=True
    )
    image = None
    for column, task in enumerate(tasks):
        axis = axes[0, column]
        matrix, layers, heads = _matrix(rows, task, "overlap")
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma", aspect="auto")
        axis.set_xticks(np.arange(len(heads)), heads)
        axis.set_yticks(np.arange(len(layers)), layers)
        axis.set_xlabel("head")
        if column == 0:
            axis.set_ylabel("layer")
        axis.set_title(_task_label(task), fontsize=10)
    if image is not None:
        figure.colorbar(image, ax=axes, fraction=0.024, pad=0.012, label="profile overlap (1 − TV)")
    figure.suptitle("Within-head semantic–structural distance alignment")
    paths = _save_figure(figure, figures_dir, "03_alignment_heatmaps")
    plt.close(figure)
    return paths


def _plot_layerwise_distance(
    summary: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(tasks), figsize=(3.25 * len(tasks), 4.1), squeeze=False, sharey=True, constrained_layout=True
    )
    styles = {
        "semantic": ("#0072B2", "o", "semantic score"),
        "structural": ("#D55E00", "s", "structural score"),
        "attention": ("#009E73", "^", "attention mass"),
    }
    for column, task in enumerate(tasks):
        axis = axes[0, column]
        selected = [row for row in summary if str(row["task"]) == task]
        for source, (colour, marker, label) in styles.items():
            source_rows = sorted(
                [row for row in selected if str(row["source"]) == source],
                key=lambda row: int(row["layer"]),
            )
            if not source_rows:
                continue
            layer = np.asarray([int(row["layer"]) for row in source_rows])
            mean = np.asarray([float(row["expected_distance_mean"]) for row in source_rows])
            q1 = np.asarray([float(row["expected_distance_q1"]) for row in source_rows])
            q3 = np.asarray([float(row["expected_distance_q3"]) for row in source_rows])
            axis.plot(layer, mean, color=colour, marker=marker, linewidth=1.7, label=label)
            axis.fill_between(layer, q1, q3, color=colour, alpha=0.13, linewidth=0)
        axis.set_xlabel("layer")
        axis.set_title(_task_label(task), fontsize=10)
        if column == 0:
            axis.set_ylabel("expected graph distance")
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Layerwise score and attention distance (band: headwise IQR)")
    paths = _save_figure(figure, figures_dir, "04_layerwise_score_attention_distance")
    plt.close(figure)
    return paths


def _plot_width_uncertainty(
    summary: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2, len(tasks), figsize=(3.25 * len(tasks), 7.2), squeeze=False, sharex="col", constrained_layout=True
    )
    styles = {"semantic": ("#0072B2", "o"), "structural": ("#D55E00", "s")}
    any_uncertainty = False
    for column, task in enumerate(tasks):
        selected = [row for row in summary if str(row["task"]) == task]
        for source, (colour, marker) in styles.items():
            source_rows = sorted(
                [row for row in selected if str(row["source"]) == source],
                key=lambda row: int(row["layer"]),
            )
            if not source_rows:
                continue
            layer = np.asarray([int(row["layer"]) for row in source_rows])
            width = np.asarray([float(row["spatial_variance_mean"]) for row in source_rows])
            uncertainty = np.asarray([float(row["estimation_sem_mean"]) for row in source_rows])
            axes[0, column].plot(layer, width, color=colour, marker=marker, label=source)
            if np.isfinite(uncertainty).any():
                any_uncertainty = True
                axes[1, column].plot(layer, uncertainty, color=colour, marker=marker, label=source)
        axes[0, column].set_title(_task_label(task), fontsize=10)
        axes[1, column].set_xlabel("layer")
        if column == 0:
            axes[0, column].set_ylabel("mean spatial variance (hops²)")
            axes[1, column].set_ylabel("mean expected-distance SE")
            axes[0, column].legend(frameon=False, fontsize=8)
    if not any_uncertainty:
        for axis in axes[1]:
            axis.text(0.5, 0.5, "per-graph contributions unavailable", ha="center", va="center", transform=axis.transAxes, fontsize=8)
    figure.suptitle("Spatial width and estimation uncertainty")
    paths = _save_figure(figure, figures_dir, "05_spatial_width_and_uncertainty")
    plt.close(figure)
    return paths


def _profile_for_head(
    model: SpatialModel, layer: int, head: int, channel: str
) -> tuple[tuple[str, ...], np.ndarray]:
    values = model.score["channels"][channel]["heatmap_exact_head"][layer, head]
    labels, grouped = group_distance(values, model.score["axis"])
    return labels, _normalise(grouped)


def _plot_representatives(
    models: Sequence[SpatialModel], representatives: Sequence[Mapping[str, Any]], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    if not representatives:
        return []
    tasks = [model.task for model in models if any(str(row["task"]) == model.task for row in representatives)]
    by_task = {model.task: model for model in models}
    figure, axes = plt.subplots(
        len(tasks), 2, figsize=(9.0, 2.6 * len(tasks)), squeeze=False, constrained_layout=True
    )
    for row_index, task in enumerate(tasks):
        for column, role in enumerate(("high alignment", "low alignment")):
            axis = axes[row_index, column]
            selected = next(
                row for row in representatives if str(row["task"]) == task and str(row["role"]) == role
            )
            model = by_task[task]
            labels, semantic = _profile_for_head(model, int(selected["layer"]), int(selected["head"]), "semantic")
            _, structural = _profile_for_head(model, int(selected["layer"]), int(selected["head"]), "structural")
            x = np.arange(len(labels))
            axis.plot(x, semantic, color="#0072B2", marker="o", label="semantic")
            axis.plot(x, structural, color="#D55E00", marker="s", label="structural")
            axis.set_xticks(x, labels)
            axis.set_ylim(bottom=0.0)
            axis.set_title(
                f"{_task_label(task)} · {role}\n"
                f"(ℓ{int(selected['layer'])}, h{int(selected['head'])}), overlap={float(selected['overlap']):.2f}",
                fontsize=9,
            )
            if column == 0:
                axis.set_ylabel("normalized score mass")
            if row_index == len(tasks) - 1:
                axis.set_xlabel("distance")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("Representative aligned and misaligned head profiles")
    paths = _save_figure(figure, figures_dir, "06_representative_head_profiles")
    plt.close(figure)
    return paths


def _plot_score_carriage(
    profile_rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    if not any(str(row["source"]).endswith("_carriage") for row in profile_rows):
        return []
    figure, axes = plt.subplots(
        2, len(tasks), figsize=(3.25 * len(tasks), 7.0), squeeze=False, sharey="row", constrained_layout=True
    )
    for row_index, channel in enumerate(CHANNELS):
        for column, task in enumerate(tasks):
            axis = axes[row_index, column]
            selected = [row for row in profile_rows if str(row["task"]) == task]
            sources = (f"{channel}_score", f"{channel}_carriage")
            labels = []
            for source in sources:
                labels.extend(str(row["distance"]) for row in selected if str(row["source"]) == source)
            labels = list(dict.fromkeys(labels))
            for source, colour, marker, name in (
                (sources[0], "#4C78A8", "o", "mean head score"),
                (sources[1], "#222222", "s", "final-state carriage"),
            ):
                values_by_label = {
                    str(row["distance"]): float(row["equal_head_mass"])
                    for row in selected
                    if str(row["source"]) == source
                }
                if not values_by_label:
                    continue
                values = np.asarray([values_by_label.get(label, 0.0) for label in labels])
                axis.plot(np.arange(len(labels)), values, color=colour, marker=marker, label=name)
            axis.set_xticks(np.arange(len(labels)), labels)
            axis.set_title(_task_label(task), fontsize=10)
            axis.set_xlabel("distance")
            if column == 0:
                axis.set_ylabel(f"{channel}\nnormalized mass")
            if row_index == 0 and column == 0:
                axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Head score geometry and final-state response")
    paths = _save_figure(figure, figures_dir, "07_score_and_final_state_response")
    plt.close(figure)
    return paths


def run(
    roots: Sequence[Path],
    output_dir: Path,
    *,
    tasks: Sequence[str],
    seed: int = 42,
    activity_quantile: float = 0.25,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build the exploratory tables and figures from every available cache."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    inventory_rows = inventory(roots, tasks, seed=int(seed))
    _write_csv(output_dir / "cache_inventory.csv", inventory_rows)
    models, warnings = load_models(roots, tasks, seed=int(seed))
    if not models:
        raise FileNotFoundError("no usable score caches were found for the requested tasks")
    if verbose:
        for warning in warnings:
            print(f"[chapter6:warning] {warning}", flush=True)
        print(
            "[chapter6:load] " + ", ".join(f"{model.task} ({model.score_path})" for model in models),
            flush=True,
        )
    head_rows = head_metrics(models)
    layer_rows = layer_summary(head_rows)
    representative_rows = representative_heads(head_rows, activity_quantile=activity_quantile)
    profile_rows = model_profiles(models)
    uncertainty_rows = uncertainty_profiles(models)
    tables = {
        "head_spatial_metrics.csv": head_rows,
        "layer_spatial_summary.csv": layer_rows,
        "representative_heads.csv": representative_rows,
        "model_distance_profiles.csv": profile_rows,
        "score_profile_uncertainty.csv": uncertainty_rows,
    }
    for name, rows in tables.items():
        _write_csv(output_dir / name, rows)
    available_tasks = [model.task for model in models]
    figures: list[Path] = []
    figures.extend(_plot_expected_heatmaps(head_rows, available_tasks, figures_dir))
    figures.extend(_plot_peak_heatmaps(head_rows, available_tasks, figures_dir))
    figures.extend(_plot_alignment_heatmaps(head_rows, available_tasks, figures_dir))
    figures.extend(_plot_layerwise_distance(layer_rows, available_tasks, figures_dir))
    figures.extend(_plot_width_uncertainty(layer_rows, available_tasks, figures_dir))
    figures.extend(_plot_representatives(models, representative_rows, figures_dir))
    figures.extend(_plot_score_carriage(profile_rows, available_tasks, figures_dir))
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "tasks_requested": list(tasks),
        "tasks_loaded": available_tasks,
        "seed": int(seed),
        "roots": [str(Path(root)) for root in roots],
        "warnings": warnings,
        "model_artifacts": [
            {
                "task": model.task,
                "artifact_task": model.artifact_task,
                "score": str(model.score_path),
                "protocol": str(model.score_metadata.get("protocol_version", "unknown")),
                "attention_available": model.score.get("clean_attention_distance") is not None,
                "per_graph_contributions_available": all(
                    isinstance(model.score["channels"][channel].get("graph_distance_contribution"), Mapping)
                    for channel in CHANNELS
                ),
                "carriage": None if model.carriage_path is None else str(model.carriage_path),
            }
            for model in models
        ],
        "figures": [str(path) for path in figures],
        "interpretation": {
            "alignment": "overlap of normalized within-head profiles, equal to 1 minus total variation",
            "expected_distance": "molecular-only expected graph distance; virtual carriers remain separate",
            "spatial_width": "variance of normalized molecular score mass over graph distance",
            "uncertainty": "standard error of expected distance across cached held-out graphs",
            "attention": "clean attention mass by graph distance, not a causal score",
            "carriage": "final-state response under the same intervention family, not task necessity",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if verbose:
        print(f"[chapter6:saved] {output_dir}", flush=True)
    return {
        **summary,
        "head_rows": head_rows,
        "layer_rows": layer_rows,
        "representative_rows": representative_rows,
    }


__all__ = [
    "ANALYSIS_VERSION",
    "SpatialModel",
    "head_metrics",
    "inventory",
    "layer_summary",
    "load_models",
    "model_profiles",
    "representative_heads",
    "run",
    "uncertainty_profiles",
]
