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
    head_profile_distance_decomposition_rows,
    score_interval_width_profile,
    score_profile,
    summarise_layerwise_distance_decomposition,
    summarise_layerwise_vnode_profiles,
    vnode_profile_rows,
)

ANALYSIS_VERSION = "chapter6-spatial-explorer-v6"
CHANNELS = ("semantic", "structural")
ORGANISATION_FAMILIES = (
    "semantic_leaning",
    "structural_leaning",
    "central_responsive",
    "other",
    "inactive",
)


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


def _candidate_records(roots: Sequence[Path], task: str, seed: int) -> list[dict[str, Any]]:
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


def inventory(roots: Sequence[Path], tasks: Sequence[str], *, seed: int) -> list[dict[str, Any]]:
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
                    key: (str(value) if isinstance(value, Path) else value)
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
                    f"{task}: ignored {candidate['score_path']} ({type(error).__name__}: {error})"
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
        carriage_path = None
        carriage_candidates = [
            candidate
            for candidate, _candidate_score, _candidate_metadata in loaded
            if candidate["carriage_exists"]
        ]
        for carriage_candidate in carriage_candidates:
            candidate_path = Path(carriage_candidate["carriage_path"])
            try:
                carriage, _ = _load_payload(candidate_path)
                carriage_path = candidate_path
                if carriage_candidate is not selected:
                    warnings.append(
                        f"{task}: selected score cache has no carriage; using equivalent "
                        f"carriage artifact {candidate_path}"
                    )
                break
            except (OSError, RuntimeError, EOFError, TypeError, ValueError, KeyError) as error:
                warnings.append(
                    f"{task}: ignored carriage cache {candidate_path} "
                    f"({type(error).__name__}: {error})"
                )
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


def _head_activity(
    score: Mapping[str, Any], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
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


def _head_family_lookup(score: Mapping[str, Any]) -> dict[tuple[int, int], str]:
    families = score.get("families", {})
    if not isinstance(families, Mapping):
        return {}
    lookup: dict[tuple[int, int], str] = {}
    for family in (
        "semantic_leaning",
        "structural_leaning",
        "central_responsive",
        "inactive",
    ):
        for head in families.get(family, ()):
            if isinstance(head, Sequence) and len(head) == 2:
                lookup[(int(head[0]), int(head[1]))] = family
    return lookup


def _molecular_profile(
    values: Any, axis: Sequence[Any], reportable: Any
) -> tuple[np.ndarray, np.ndarray]:
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
        return {
            "overlap": float("nan"),
            "cosine": float("nan"),
            "wasserstein": float("nan"),
            "wasserstein_similarity": float("nan"),
        }
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
        opportunity: dict[str, np.ndarray | None] = {}
        for channel in CHANNELS:
            candidate = score["channels"][channel].get("heatmap_per_opportunity_head")
            if candidate is None:
                opportunity[channel] = None
                continue
            candidate_array = _as_numpy(candidate)
            opportunity[channel] = (
                candidate_array if candidate_array.shape == semantic.shape else None
            )
        layers, heads, _ = semantic.shape
        joint, selectivity = _head_activity(score, (layers, heads))
        coordinates = score.get("coordinates", {})
        raw_semantic = _as_numpy(score["channels"]["semantic"]["raw"])
        raw_structural = _as_numpy(score["channels"]["structural"]["raw"])
        normalized_semantic_value = _field(coordinates, "normalized_semantic")
        normalized_structural_value = _field(coordinates, "normalized_structural")
        normalized_semantic = (
            _as_numpy(normalized_semantic_value)
            if normalized_semantic_value is not None
            else raw_semantic / max(float(np.nanmean(raw_semantic)), 1.0e-12)
        )
        normalized_structural = (
            _as_numpy(normalized_structural_value)
            if normalized_structural_value is not None
            else raw_structural / max(float(np.nanmean(raw_structural)), 1.0e-12)
        )
        family_lookup = _head_family_lookup(score)
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
                semantic_stats = _profile_statistics(semantic[layer, head], axis, reportable)
                structural_stats = _profile_statistics(structural[layer, head], axis, reportable)
                semantic_opportunity_stats = (
                    _profile_statistics(opportunity["semantic"][layer, head], axis, reportable)
                    if opportunity["semantic"] is not None
                    else {}
                )
                structural_opportunity_stats = (
                    _profile_statistics(opportunity["structural"][layer, head], axis, reportable)
                    if opportunity["structural"] is not None
                    else {}
                )
                alignment = _profile_alignment(
                    semantic[layer, head], structural[layer, head], axis, reportable
                )
                attention_stats = (
                    _profile_statistics(attention_array[layer, head], axis, reportable)
                    if attention_array is not None
                    else {}
                )
                attention_reach = float(
                    attention_stats.get("expected_distance", float("nan"))
                )
                semantic_reach = float(
                    semantic_stats.get("expected_distance", float("nan"))
                )
                structural_reach = float(
                    structural_stats.get("expected_distance", float("nan"))
                )
                semantic_gap = semantic_reach - attention_reach
                structural_gap = structural_reach - attention_reach
                finite_gaps = {
                    "semantic": semantic_gap,
                    "structural": structural_gap,
                }
                finite_gaps = {
                    name: value for name, value in finite_gaps.items() if np.isfinite(value)
                }
                dominant_channel = (
                    max(finite_gaps, key=lambda name: abs(finite_gaps[name]))
                    if finite_gaps
                    else ""
                )
                dominant_gap = (
                    float(finite_gaps[dominant_channel])
                    if dominant_channel
                    else float("nan")
                )
                rows.append(
                    {
                        "task": model.task,
                        "layer": layer,
                        "head": head,
                        "family": family_lookup.get((layer, head), "other"),
                        "raw_semantic_score": float(raw_semantic[layer, head]),
                        "raw_structural_score": float(raw_structural[layer, head]),
                        "normalized_semantic_score": float(normalized_semantic[layer, head]),
                        "normalized_structural_score": float(normalized_structural[layer, head]),
                        "joint_sensitivity": float(joint[layer, head]),
                        "selectivity": float(selectivity[layer, head]),
                        **{f"semantic_{key}": value for key, value in semantic_stats.items()},
                        **{f"structural_{key}": value for key, value in structural_stats.items()},
                        **{
                            f"semantic_opportunity_{key}": value
                            for key, value in semantic_opportunity_stats.items()
                        },
                        **{
                            f"structural_opportunity_{key}": value
                            for key, value in structural_opportunity_stats.items()
                        },
                        **alignment,
                        **{f"attention_{key}": value for key, value in attention_stats.items()},
                        "semantic_attention_reach_gap": semantic_gap,
                        "structural_attention_reach_gap": structural_gap,
                        "max_abs_attention_reach_gap": (
                            abs(dominant_gap) if np.isfinite(dominant_gap) else float("nan")
                        ),
                        "dominant_reach_gap_channel": dominant_channel,
                        "dominant_signed_reach_gap": dominant_gap,
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


def reach_mismatch_summary(
    rows: Sequence[Mapping[str, Any]], *, activity_quantile: float = 0.25
) -> list[dict[str, Any]]:
    """Summarise attention--score reach mismatch across active heads."""

    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task"]), []).append(row)
    output: list[dict[str, Any]] = []
    for task, task_rows in by_task.items():
        active_rows = [
            row for row in task_rows if str(row.get("family")) != "inactive"
        ]
        joint = np.asarray(
            [float(row["joint_sensitivity"]) for row in active_rows], dtype=np.float64
        )
        finite_joint = joint[np.isfinite(joint)]
        floor = (
            float(np.quantile(finite_joint, activity_quantile))
            if finite_joint.size
            else float("inf")
        )
        selected = [
            row
            for row in active_rows
            if np.isfinite(float(row["max_abs_attention_reach_gap"]))
            and np.isfinite(float(row["joint_sensitivity"]))
            and float(row["joint_sensitivity"]) >= floor
        ]
        if not selected:
            continue
        maximum = np.asarray(
            [float(row["max_abs_attention_reach_gap"]) for row in selected]
        )
        weights = np.maximum(
            np.asarray([float(row["joint_sensitivity"]) for row in selected]), 0.0
        )
        semantic = np.asarray(
            [float(row["semantic_attention_reach_gap"]) for row in selected]
        )
        structural = np.asarray(
            [float(row["structural_attention_reach_gap"]) for row in selected]
        )
        denominator = float(np.sum(weights))
        output.append(
            {
                "task": task,
                "activity_quantile": float(activity_quantile),
                "activity_floor": floor,
                "heads": len(selected),
                "median_max_abs_gap": float(np.median(maximum)),
                "p90_max_abs_gap": float(np.quantile(maximum, 0.90)),
                "maximum_abs_gap": float(np.max(maximum)),
                "fraction_abs_gap_ge_0_5": float(np.mean(maximum >= 0.5)),
                "fraction_abs_gap_ge_1": float(np.mean(maximum >= 1.0)),
                "J_weighted_mean_max_abs_gap": (
                    float(np.sum(weights * maximum) / denominator)
                    if denominator > 1.0e-12
                    else float("nan")
                ),
                "semantic_gap_median": float(np.median(semantic)),
                "structural_gap_median": float(np.median(structural)),
            }
        )
    return output


def representative_reach_mismatches(
    rows: Sequence[Mapping[str, Any]], *, activity_quantile: float = 0.25
) -> list[dict[str, Any]]:
    """Select the largest active attention--score reach mismatch per model."""

    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task"]), []).append(row)
    output: list[dict[str, Any]] = []
    for task, task_rows in by_task.items():
        active_rows = [
            row for row in task_rows if str(row.get("family")) != "inactive"
        ]
        finite_joint = np.asarray(
            [
                float(row["joint_sensitivity"])
                for row in active_rows
                if np.isfinite(float(row["joint_sensitivity"]))
            ]
        )
        floor = (
            float(np.quantile(finite_joint, activity_quantile))
            if finite_joint.size
            else float("inf")
        )
        eligible = [
            row
            for row in active_rows
            if np.isfinite(float(row["max_abs_attention_reach_gap"]))
            and float(row["joint_sensitivity"]) >= floor
        ]
        if not eligible:
            continue
        selected = max(
            eligible, key=lambda row: float(row["max_abs_attention_reach_gap"])
        )
        output.append(
            {
                **dict(selected),
                "role": "strongest active reach mismatch",
                "activity_quantile": float(activity_quantile),
                "activity_floor": floor,
            }
        )
    return output


def layer_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), int(row["layer"])), []).append(row)
    output: list[dict[str, Any]] = []
    sources = (
        "semantic",
        "structural",
        "semantic_opportunity",
        "structural_opportunity",
        "attention",
    )
    for (task, layer), group in sorted(groups.items()):
        for source in sources:
            field = f"{source}_expected_distance"
            values = np.asarray([float(row.get(field, np.nan)) for row in group], dtype=np.float64)
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
                    "spatial_variance_q1": (
                        float(np.quantile(widths, 0.25)) if widths.size else float("nan")
                    ),
                    "spatial_variance_q3": (
                        float(np.quantile(widths, 0.75)) if widths.size else float("nan")
                    ),
                    "headwise_spatial_variance_sem": (
                        float(np.std(widths, ddof=1) / math.sqrt(widths.size))
                        if widths.size > 1
                        else float("nan")
                    ),
                    "estimation_sem_mean": (
                        float(np.mean(uncertainties)) if uncertainties.size else float("nan")
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
            [
                float(row["joint_sensitivity"])
                for row in group
                if np.isfinite(float(row["joint_sensitivity"]))
            ]
        )
        floor = (
            float(np.quantile(finite_joint, activity_quantile)) if finite_joint.size else -np.inf
        )
        eligible = [
            row
            for row in group
            if float(row["joint_sensitivity"]) >= floor and np.isfinite(float(row["overlap"]))
        ]
        if not eligible:
            continue
        lower = min(eligible, key=lambda row: float(row["overlap"]))
        upper_floor = float(np.quantile([float(row["overlap"]) for row in eligible], 0.75))
        comparison_pool = [
            row for row in eligible if row is not lower and float(row["overlap"]) >= upper_floor
        ]
        if not comparison_pool:
            comparison_pool = [row for row in eligible if row is not lower]
        if comparison_pool:
            lower_joint = max(float(lower["joint_sensitivity"]), 1.0e-12)
            lower_family = str(lower.get("family"))
            lower_layer = int(lower["layer"])

            def match_cost(
                row: Mapping[str, Any],
                lower_joint: float = lower_joint,
                lower_family: str = lower_family,
                lower_layer: int = lower_layer,
            ) -> float:
                joint = max(float(row["joint_sensitivity"]), 1.0e-12)
                family_penalty = 0.0 if str(row.get("family")) == lower_family else 0.35
                return (
                    abs(math.log(joint / lower_joint))
                    + 0.08 * abs(int(row["layer"]) - lower_layer)
                    + family_penalty
                )

            higher = min(comparison_pool, key=match_cost)
            pair = (
                ("lower alignment", lower),
                ("J-matched higher alignment", higher),
            )
        else:
            pair = (("lower alignment", lower),)
        for role, selected in pair:
            output.append(
                {
                    "role": role,
                    "comparison_overlap_gap": float(selected["overlap"]) - float(lower["overlap"]),
                    "reference_layer": int(lower["layer"]),
                    "reference_head": int(lower["head"]),
                    **dict(selected),
                }
            )
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


def layer_distance_profiles(
    models: Sequence[SpatialModel],
) -> list[dict[str, Any]]:
    """Return equal-head semantic and structural mass in each layer and distance bin."""

    exact_rows = head_profile_distance_decomposition_rows(models)  # type: ignore[arg-type]
    return summarise_layerwise_distance_decomposition(exact_rows)


def vnode_layer_profiles(models: Sequence[SpatialModel]) -> list[dict[str, Any]]:
    """Keep virtual-node allocation separate from molecular graph distance."""

    per_head = vnode_profile_rows(models)  # type: ignore[arg-type]
    return summarise_layerwise_vnode_profiles(per_head)


def _graph_layer_widths(
    channel_score: Mapping[str, Any], axis: Sequence[Any], reportable: np.ndarray
) -> dict[str, np.ndarray]:
    contributions = channel_score.get("graph_distance_contribution")
    if not isinstance(contributions, Mapping):
        return {}
    distances = np.asarray(
        [value if (value := _numeric_distance(label)) is not None else np.nan for label in axis],
        dtype=np.float64,
    )
    mask = np.isfinite(distances) & _as_numpy(reportable, dtype=bool)
    distances = distances[mask]
    output: dict[str, np.ndarray] = {}
    for graph_id, values in contributions.items():
        array = _as_numpy(values)
        if array.ndim != 3 or array.shape[-1] != len(axis):
            continue
        mass = np.maximum(array[..., mask], 0.0)
        total = np.sum(mass, axis=-1, keepdims=True)
        profile = np.full_like(mass, np.nan, dtype=np.float64)
        np.divide(mass, total, out=profile, where=total > 1.0e-12)
        expected = np.sum(profile * distances, axis=-1)
        variance = np.sum(profile * np.square(distances - expected[..., None]), axis=-1)
        output[str(graph_id)] = np.nanmean(variance, axis=1)
    return output


def spatial_width_bootstrap(
    models: Sequence[SpatialModel], *, replicates: int = 2000, seed: int = 2026
) -> list[dict[str, Any]]:
    """Paired graph-bootstrap intervals for semantic and structural profile width."""

    output: list[dict[str, Any]] = []
    for model_index, model in enumerate(models):
        score = model.score
        reportable = _reportable(score, "semantic") & _reportable(score, "structural")
        by_channel = {
            channel: _graph_layer_widths(score["channels"][channel], score["axis"], reportable)
            for channel in CHANNELS
        }
        graph_ids = sorted(set(by_channel["semantic"]) & set(by_channel["structural"]))
        if not graph_ids:
            continue
        semantic = np.stack([by_channel["semantic"][graph_id] for graph_id in graph_ids])
        structural = np.stack([by_channel["structural"][graph_id] for graph_id in graph_ids])
        if semantic.shape != structural.shape:
            continue
        generator = np.random.default_rng(int(seed) + model_index)
        indices = generator.integers(
            0, len(graph_ids), size=(max(int(replicates), 1), len(graph_ids))
        )
        semantic_boot = np.nanmean(semantic[indices], axis=1)
        structural_boot = np.nanmean(structural[indices], axis=1)
        difference_boot = structural_boot - semantic_boot
        for layer in range(semantic.shape[1]):
            record: dict[str, Any] = {
                "task": model.task,
                "layer": layer,
                "graphs": len(graph_ids),
                "bootstrap_replicates": int(replicates),
            }
            for name, values, boot in (
                ("semantic", semantic, semantic_boot),
                ("structural", structural, structural_boot),
                ("structural_minus_semantic", structural - semantic, difference_boot),
            ):
                finite = boot[:, layer][np.isfinite(boot[:, layer])]
                record[f"{name}_mean"] = float(np.nanmean(values[:, layer]))
                record[f"{name}_low"] = (
                    float(np.quantile(finite, 0.025)) if finite.size else float("nan")
                )
                record[f"{name}_high"] = (
                    float(np.quantile(finite, 0.975)) if finite.size else float("nan")
                )
            output.append(record)
    return output


def width_contribution_profiles(
    models: Sequence[SpatialModel],
) -> list[dict[str, Any]]:
    """Locate the distance bins that contribute to semantic and structural width."""

    output: list[dict[str, Any]] = []
    for model in models:
        score = model.score
        axis = tuple(score["axis"])
        reportable = _reportable(score, "semantic") & _reportable(score, "structural")
        profiles: dict[str, np.ndarray] = {}
        distances = None
        for channel in CHANNELS:
            values = score["channels"][channel]["heatmap_exact_head"]
            channel_distances, profiles[channel] = _molecular_profile(values, axis, reportable)
            distances = channel_distances
        if distances is None or not len(distances):
            continue
        joint, _ = _head_activity(score, profiles["semantic"].shape[:2])
        contributions: dict[str, np.ndarray] = {}
        labels: tuple[str, ...] | None = None
        for channel in CHANNELS:
            profile = profiles[channel]
            expected = np.sum(profile * distances, axis=-1)
            exact = profile * np.square(distances - expected[..., None])
            labels, contributions[channel] = group_distance(exact, tuple(distances))
        if labels is None:
            continue
        for layer in range(profiles["semantic"].shape[0]):
            weights = np.maximum(joint[layer], 0.0)
            for weighting in ("equal_head", "J_weighted"):
                if weighting == "equal_head":
                    semantic = np.nanmean(contributions["semantic"][layer], axis=0)
                    structural = np.nanmean(contributions["structural"][layer], axis=0)
                    valid_heads = int(profiles["semantic"].shape[1])
                else:
                    valid = np.isfinite(weights) & (weights > 0.0)
                    valid_heads = int(np.count_nonzero(valid))
                    if not valid_heads:
                        continue
                    denominator = float(np.sum(weights[valid]))
                    semantic = (
                        np.nansum(
                            contributions["semantic"][layer, valid] * weights[valid, None],
                            axis=0,
                        )
                        / denominator
                    )
                    structural = (
                        np.nansum(
                            contributions["structural"][layer, valid] * weights[valid, None],
                            axis=0,
                        )
                        / denominator
                    )
                for label, semantic_value, structural_value in zip(labels, semantic, structural):
                    output.append(
                        {
                            "task": model.task,
                            "layer": layer,
                            "weighting": weighting,
                            "distance_group": label,
                            "valid_heads": valid_heads,
                            "semantic_width_contribution": float(semantic_value),
                            "structural_width_contribution": float(structural_value),
                            "structural_minus_semantic_width_contribution": float(
                                structural_value - semantic_value
                            ),
                        }
                    )
    return output


def width_by_head_role(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize spatial width by established head family and by J weighting."""

    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task"]), []).append(row)
    output: list[dict[str, Any]] = []
    family_order = (
        "all_active",
        "semantic_leaning",
        "structural_leaning",
        "central_responsive",
        "other",
        "inactive",
    )
    for task, task_rows in by_task.items():
        for family in family_order:
            if family == "all_active":
                selected = [
                    row
                    for row in task_rows
                    if str(row.get("family")) != "inactive"
                    and float(row["joint_sensitivity"]) > 0.0
                ]
            else:
                selected = [row for row in task_rows if str(row.get("family")) == family]
            if not selected:
                continue
            semantic = np.asarray([float(row["semantic_spatial_variance"]) for row in selected])
            structural = np.asarray([float(row["structural_spatial_variance"]) for row in selected])
            weights = np.maximum(
                np.asarray([float(row["joint_sensitivity"]) for row in selected]),
                0.0,
            )
            valid = np.isfinite(semantic) & np.isfinite(structural)
            weighted = valid & np.isfinite(weights) & (weights > 0.0)
            if not np.any(valid):
                continue
            record = {
                "task": task,
                "family": family,
                "heads": int(np.count_nonzero(valid)),
                "semantic_width_mean": float(np.mean(semantic[valid])),
                "structural_width_mean": float(np.mean(structural[valid])),
                "structural_excess_mean": float(np.mean(structural[valid] - semantic[valid])),
                "mean_J": float(np.mean(weights[valid])),
                "semantic_width_J_weighted": float("nan"),
                "structural_width_J_weighted": float("nan"),
                "structural_excess_J_weighted": float("nan"),
            }
            if np.any(weighted):
                denominator = float(np.sum(weights[weighted]))
                semantic_weighted = float(
                    np.sum(semantic[weighted] * weights[weighted]) / denominator
                )
                structural_weighted = float(
                    np.sum(structural[weighted] * weights[weighted]) / denominator
                )
                record.update(
                    {
                        "semantic_width_J_weighted": semantic_weighted,
                        "structural_width_J_weighted": structural_weighted,
                        "structural_excess_J_weighted": structural_weighted - semantic_weighted,
                    }
                )
            output.append(record)
    return output


def layer_score_organisation(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Locate total head sensitivity and semantic--structural balance over depth."""

    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task"]), []).append(row)
    output: list[dict[str, Any]] = []
    for task, task_rows in by_task.items():
        total_joint = float(
            np.sum(
                [
                    max(float(row["joint_sensitivity"]), 0.0)
                    for row in task_rows
                    if np.isfinite(float(row["joint_sensitivity"]))
                ]
            )
        )
        for layer in sorted({int(row["layer"]) for row in task_rows}):
            selected = [row for row in task_rows if int(row["layer"]) == layer]
            joint = np.asarray(
                [max(float(row["joint_sensitivity"]), 0.0) for row in selected],
                dtype=np.float64,
            )
            selectivity = np.asarray(
                [float(row["selectivity"]) for row in selected], dtype=np.float64
            )
            valid = np.isfinite(joint) & np.isfinite(selectivity) & (joint > 0.0)
            layer_joint = float(np.sum(joint[np.isfinite(joint)]))
            weighted_selectivity = (
                float(np.sum(joint[valid] * selectivity[valid]) / np.sum(joint[valid]))
                if np.any(valid)
                else float("nan")
            )
            output.append(
                {
                    "task": task,
                    "layer": layer,
                    "heads": len(selected),
                    "active_heads": int(
                        sum(str(row.get("family")) != "inactive" for row in selected)
                    ),
                    "joint_sensitivity_sum": layer_joint,
                    "joint_sensitivity_share": (
                        layer_joint / total_joint if total_joint > 1.0e-12 else float("nan")
                    ),
                    "joint_weighted_selectivity": weighted_selectivity,
                }
            )
    return output


def head_role_score_allocation(
    models: Sequence[SpatialModel],
    rows: Sequence[Mapping[str, Any]],
    *,
    tail_distance: float = 4.0,
) -> list[dict[str, Any]]:
    """Allocate total sensitivity and long-range score mass across head roles."""

    row_lookup = {
        (str(row["task"]), int(row["layer"]), int(row["head"])): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for model in models:
        score = model.score
        axis = tuple(score["axis"])
        numeric = np.asarray(
            [
                value if (value := _numeric_distance(label)) is not None else np.nan
                for label in axis
            ],
            dtype=np.float64,
        )
        molecular_masks = {
            channel: np.isfinite(numeric) & _reportable(score, channel)
            for channel in CHANNELS
        }
        tail_masks = {
            channel: molecular_masks[channel] & (numeric >= float(tail_distance))
            for channel in CHANNELS
        }
        accumulators = {
            family: {
                "heads": 0,
                "J": 0.0,
                "semantic_total": 0.0,
                "semantic_tail": 0.0,
                "structural_total": 0.0,
                "structural_tail": 0.0,
            }
            for family in ORGANISATION_FAMILIES
        }
        semantic = np.maximum(
            _as_numpy(score["channels"]["semantic"]["heatmap_exact_head"]), 0.0
        )
        structural = np.maximum(
            _as_numpy(score["channels"]["structural"]["heatmap_exact_head"]), 0.0
        )
        for layer in range(semantic.shape[0]):
            for head in range(semantic.shape[1]):
                row = row_lookup[(model.task, layer, head)]
                family = str(row.get("family", "other"))
                if family not in accumulators:
                    family = "other"
                target = accumulators[family]
                target["heads"] += 1
                target["J"] += max(float(row["joint_sensitivity"]), 0.0)
                for channel, values in (
                    ("semantic", semantic[layer, head]),
                    ("structural", structural[layer, head]),
                ):
                    target[f"{channel}_total"] += float(
                        np.sum(values[molecular_masks[channel]])
                    )
                    target[f"{channel}_tail"] += float(
                        np.sum(values[tail_masks[channel]])
                    )
        total_joint = sum(float(value["J"]) for value in accumulators.values())
        total_tail = {
            channel: sum(
                float(value[f"{channel}_tail"]) for value in accumulators.values()
            )
            for channel in CHANNELS
        }
        for family in ORGANISATION_FAMILIES:
            values = accumulators[family]
            record: dict[str, Any] = {
                "task": model.task,
                "family": family,
                "heads": int(values["heads"]),
                "tail_distance": float(tail_distance),
                "joint_sensitivity_mass": float(values["J"]),
                "joint_sensitivity_share": (
                    float(values["J"]) / total_joint
                    if total_joint > 1.0e-12
                    else float("nan")
                ),
            }
            for channel in CHANNELS:
                channel_total = float(values[f"{channel}_total"])
                channel_tail = float(values[f"{channel}_tail"])
                record[f"{channel}_tail_mass"] = channel_tail
                record[f"{channel}_tail_share"] = (
                    channel_tail / total_tail[channel]
                    if total_tail[channel] > 1.0e-12
                    else float("nan")
                )
                record[f"{channel}_within_family_tail_fraction"] = (
                    channel_tail / channel_total
                    if channel_total > 1.0e-12
                    else float("nan")
                )
            output.append(record)
    return output


def _jensen_shannon_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.maximum(np.asarray(left, dtype=np.float64), 0.0)
    right = np.maximum(np.asarray(right, dtype=np.float64), 0.0)
    if left.shape != right.shape or left.sum() <= 0.0 or right.sum() <= 0.0:
        return float("nan")
    left = left / left.sum()
    right = right / right.sum()
    midpoint = 0.5 * (left + right)

    def divergence(values: np.ndarray) -> float:
        valid = values > 0.0
        return float(np.sum(values[valid] * np.log2(values[valid] / midpoint[valid])))

    distance = math.sqrt(max(0.0, 0.5 * divergence(left) + 0.5 * divergence(right)))
    return float(np.clip(1.0 - distance, 0.0, 1.0))


def score_organisation_similarity(
    distance_rows: Sequence[Mapping[str, Any]],
    layer_organisation_rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
) -> list[dict[str, Any]]:
    """Compare model-wide score geometry with its layer-resolved placement."""

    tasks = [
        str(task)
        for task in tasks
        if any(str(row["task"]) == str(task) for row in distance_rows)
    ]
    global_keys = sorted(
        {
            (channel, str(row["distance_group"]))
            for row in distance_rows
            if str(row["profile_kind"]) == "score_mass"
            for channel in CHANNELS
        }
    )
    layer_keys = sorted(
        {
            (int(row["layer"]), channel, str(row["distance_group"]))
            for row in distance_rows
            if str(row["profile_kind"]) == "score_mass"
            for channel in CHANNELS
        }
    )
    global_vectors: dict[str, np.ndarray] = {}
    layer_vectors: dict[str, np.ndarray] = {}
    for task in tasks:
        layer_weight = {
            int(row["layer"]): float(row["joint_sensitivity_share"])
            for row in layer_organisation_rows
            if str(row["task"]) == task
        }
        layer_lookup: dict[tuple[int, str, str], float] = {}
        for row in distance_rows:
            if str(row["task"]) != task or str(row["profile_kind"]) != "score_mass":
                continue
            for channel in CHANNELS:
                layer_lookup[
                    (int(row["layer"]), channel, str(row["distance_group"]))
                ] = float(row[f"{channel}_share_mean"]) * layer_weight.get(
                    int(row["layer"]), 0.0
                )
        layer_vector = np.asarray(
            [layer_lookup.get(key, 0.0) for key in layer_keys], dtype=np.float64
        )
        layer_vectors[task] = layer_vector
        global_lookup = {key: 0.0 for key in global_keys}
        for (_layer, channel, distance), value in zip(layer_keys, layer_vector):
            global_lookup[(channel, distance)] += float(value)
        global_vectors[task] = np.asarray(
            [global_lookup[key] for key in global_keys], dtype=np.float64
        )
    output: list[dict[str, Any]] = []
    for left_task in tasks:
        for right_task in tasks:
            global_similarity = _jensen_shannon_similarity(
                global_vectors[left_task], global_vectors[right_task]
            )
            layer_similarity = _jensen_shannon_similarity(
                layer_vectors[left_task], layer_vectors[right_task]
            )
            output.append(
                {
                    "task_a": left_task,
                    "task_b": right_task,
                    "model_wide_similarity": global_similarity,
                    "layer_resolved_similarity": layer_similarity,
                    "placement_gap": global_similarity - layer_similarity,
                    "similarity": "one minus Jensen--Shannon distance",
                }
            )
    return output


def _graph_profile_statistics(
    values: Any, axis: Sequence[Any], reportable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    distances, profile = _molecular_profile(values, axis, reportable)
    expected = np.sum(profile * distances, axis=-1)
    variance = np.sum(profile * np.square(distances - expected[..., None]), axis=-1)
    return np.nanmean(expected, axis=1), np.nanmean(variance, axis=1)


def _graph_virtual_share(values: Any, axis: Sequence[Any], reportable: np.ndarray) -> np.ndarray:
    virtual = np.asarray([str(label).replace("_", " ").lower() == "virtual" for label in axis])
    if not np.any(virtual):
        array = _as_numpy(values)
        return np.full(array.shape[0], np.nan)
    array = np.maximum(_as_numpy(values), 0.0)
    mask = _as_numpy(reportable, dtype=bool)
    total = np.sum(array[..., mask], axis=-1)
    virtual_mass = np.sum(array[..., virtual & mask], axis=-1)
    share = np.full(total.shape, np.nan)
    np.divide(virtual_mass, total, out=share, where=total > 1.0e-12)
    return np.nanmean(share, axis=1)


def graph_spatial_metrics(models: Sequence[SpatialModel]) -> list[dict[str, Any]]:
    """Graph-local spatial metrics with molecule size and diameter from cached support."""

    output: list[dict[str, Any]] = []
    for model in models:
        score = model.score
        axis = tuple(score["axis"])
        numeric = np.asarray(
            [value if (value := _numeric_distance(label)) is not None else np.nan for label in axis]
        )
        numeric_mask = np.isfinite(numeric)
        reportable = _reportable(score, "semantic") & _reportable(score, "structural")
        contributions = {
            channel: score["channels"][channel].get("graph_distance_contribution", {})
            for channel in CHANNELS
        }
        supports = score["channels"]["semantic"].get("graph_distance_support", {})
        if not all(isinstance(value, Mapping) for value in (*contributions.values(), supports)):
            continue
        ids = sorted(
            {str(value) for value in contributions["semantic"]}
            & {str(value) for value in contributions["structural"]}
            & {str(value) for value in supports}
        )
        for graph_id in ids:

            def lookup(mapping: Mapping[Any, Any], graph_id: str = graph_id) -> Any:
                return next(value for key, value in mapping.items() if str(key) == graph_id)

            support = _as_numpy(lookup(supports))
            if support.shape[-1] != len(axis):
                continue
            if support.ndim > 1:
                support = np.nanmean(support.reshape(-1, len(axis)), axis=0)
            molecular_support = support[numeric_mask]
            num_nodes = float(np.sum(molecular_support))
            present = numeric_mask & (support > 1.0e-12)
            diameter = float(np.max(numeric[present])) if np.any(present) else float("nan")
            graph_values = {
                channel: _as_numpy(lookup(contributions[channel])) for channel in CHANNELS
            }
            if any(
                values.ndim != 3 or values.shape[-1] != len(axis)
                for values in graph_values.values()
            ):
                continue
            statistics = {
                channel: _graph_profile_statistics(values, axis, reportable)
                for channel, values in graph_values.items()
            }
            virtual = {
                channel: _graph_virtual_share(values, axis, reportable)
                for channel, values in graph_values.items()
            }
            layers = graph_values["semantic"].shape[0]
            for layer in range(layers):
                output.append(
                    {
                        "task": model.task,
                        "graph_id": graph_id,
                        "layer": layer,
                        "num_nodes": num_nodes,
                        "diameter": diameter,
                        "semantic_expected_distance": float(statistics["semantic"][0][layer]),
                        "structural_expected_distance": float(statistics["structural"][0][layer]),
                        "semantic_width": float(statistics["semantic"][1][layer]),
                        "structural_width": float(statistics["structural"][1][layer]),
                        "structural_excess_width": float(
                            statistics["structural"][1][layer] - statistics["semantic"][1][layer]
                        ),
                        "semantic_vnode_share": float(virtual["semantic"][layer]),
                        "structural_vnode_share": float(virtual["structural"][layer]),
                    }
                )
    return output


def _spearman(left: Sequence[float], right: Sequence[float]) -> tuple[int, float]:
    from scipy.stats import spearmanr

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    if np.count_nonzero(valid) < 3:
        return int(np.count_nonzero(valid)), float("nan")
    if np.std(left_array[valid]) <= 1.0e-12 or np.std(right_array[valid]) <= 1.0e-12:
        return int(np.count_nonzero(valid)), float("nan")
    result = spearmanr(left_array[valid], right_array[valid])
    return int(np.count_nonzero(valid)), float(result.statistic)


def vnode_cross_layer_relationships(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Relate semantic VN allocation to next-layer molecular organisation."""

    lookup = {(str(row["task"]), str(row["graph_id"]), int(row["layer"])): row for row in rows}
    groups: dict[tuple[str, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for (task, graph_id, layer), row in lookup.items():
        following = lookup.get((task, graph_id, layer + 1))
        if following is not None and np.isfinite(float(row["semantic_vnode_share"])):
            groups.setdefault((task, layer), []).append((row, following))
    output: list[dict[str, Any]] = []
    for (task, layer), pairs in sorted(groups.items()):
        source = [float(left["semantic_vnode_share"]) for left, _ in pairs]
        for outcome in (
            "semantic_expected_distance",
            "structural_expected_distance",
            "structural_excess_width",
        ):
            count, rho = _spearman(source, [float(right[outcome]) for _, right in pairs])
            output.append(
                {
                    "task": task,
                    "source_layer": layer,
                    "target_layer": layer + 1,
                    "outcome": outcome,
                    "graphs": count,
                    "spearman_rho": rho,
                }
            )
    return output


def molecular_scale_relationships(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Measure whether graph size or diameter predicts VN use and spatial breadth."""

    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), int(row["layer"])), []).append(row)
    output: list[dict[str, Any]] = []
    for (task, layer), group in sorted(groups.items()):
        for scale in ("num_nodes", "diameter"):
            left = [float(row[scale]) for row in group]
            for outcome in (
                "semantic_vnode_share",
                "structural_vnode_share",
                "semantic_expected_distance",
                "structural_expected_distance",
                "structural_excess_width",
            ):
                count, rho = _spearman(left, [float(row[outcome]) for row in group])
                if count:
                    output.append(
                        {
                            "task": task,
                            "layer": layer,
                            "scale": scale,
                            "outcome": outcome,
                            "graphs": count,
                            "spearman_rho": rho,
                        }
                    )
    return output


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


def _plot_alignment_head_roles(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    """Relate profile alignment directly to the established J/D_rel coordinates."""

    import matplotlib.pyplot as plt

    family_styles = {
        "semantic_leaning": ("#0072B2", "semantic-leaning"),
        "structural_leaning": ("#D55E00", "structural-leaning"),
        "central_responsive": ("#7A5195", "high-J central"),
        "inactive": ("#999999", "inactive"),
        "other": ("#009E73", "other active"),
    }
    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.25 * len(tasks), 4.2),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for column, task in enumerate(tasks):
        axis = axes[0, column]
        selected = [row for row in rows if str(row["task"]) == task]
        finite_joint = np.asarray(
            [float(row["joint_sensitivity"]) for row in selected], dtype=np.float64
        )
        finite_joint = finite_joint[np.isfinite(finite_joint)]
        scale = max(float(np.quantile(finite_joint, 0.90)), 1.0e-12) if finite_joint.size else 1.0
        for family, (colour, label) in family_styles.items():
            family_rows = [row for row in selected if str(row.get("family", "other")) == family]
            if not family_rows:
                continue
            size = np.asarray(
                [float(row["joint_sensitivity"]) for row in family_rows],
                dtype=np.float64,
            )
            size = 18.0 + 72.0 * np.clip(size / scale, 0.0, 1.5)
            axis.scatter(
                [float(row["selectivity"]) for row in family_rows],
                [float(row["overlap"]) for row in family_rows],
                s=size,
                color=colour,
                alpha=0.78,
                edgecolor="white",
                linewidth=0.45,
                label=label,
            )
        eligible = [
            row
            for row in selected
            if np.isfinite(float(row["overlap"])) and np.isfinite(float(row["joint_sensitivity"]))
        ]
        if eligible:
            activity_floor = float(
                np.quantile([float(row["joint_sensitivity"]) for row in eligible], 0.25)
            )
            labelled = sorted(
                [row for row in eligible if float(row["joint_sensitivity"]) >= activity_floor],
                key=lambda row: float(row["overlap"]),
            )[:2]
            for row in labelled:
                axis.annotate(
                    f"ℓ{int(row['layer'])},h{int(row['head'])}",
                    (float(row["selectivity"]), float(row["overlap"])),
                    xytext=(3, -9),
                    textcoords="offset points",
                    fontsize=7,
                    color="#333333",
                )
        axis.axvline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_xlim(-1.02, 1.02)
        axis.set_ylim(0.0, 1.03)
        axis.set_xlabel(r"relative selectivity $D_{\rm rel}$")
        axis.set_title(_task_label(task), fontsize=10)
        if column == 0:
            axis.set_ylabel("semantic–structural profile overlap")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.035),
            ncol=min(len(handles), 5),
            frameon=False,
            fontsize=8,
        )
    figure.suptitle("Does spatial alignment track head specialisation? (point size: J)")
    paths = _save_figure(figure, figures_dir, "06_alignment_and_head_roles")
    plt.close(figure)
    return paths


def _plot_layerwise_distance(
    summary: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.25 * len(tasks), 4.1),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
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
        2,
        len(tasks),
        figsize=(3.25 * len(tasks), 7.2),
        squeeze=False,
        sharex="col",
        constrained_layout=True,
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
            width_q1 = np.asarray([float(row["spatial_variance_q1"]) for row in source_rows])
            width_q3 = np.asarray([float(row["spatial_variance_q3"]) for row in source_rows])
            uncertainty = np.asarray([float(row["estimation_sem_mean"]) for row in source_rows])
            axes[0, column].plot(layer, width, color=colour, marker=marker, label=source)
            axes[0, column].fill_between(
                layer,
                width_q1,
                width_q3,
                color=colour,
                alpha=0.13,
                linewidth=0,
            )
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
            axis.text(
                0.5,
                0.5,
                "per-graph contributions unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=8,
            )
    figure.suptitle("Spatial width and estimation uncertainty")
    paths = _save_figure(figure, figures_dir, "05_spatial_width_and_uncertainty")
    plt.close(figure)
    return paths


def _plot_raw_opportunity_reach(
    summary: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    if not any(str(row["source"]).endswith("_opportunity") for row in summary):
        return []
    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.25 * len(tasks), 4.1),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    for column, task in enumerate(tasks):
        axis = axes[0, column]
        selected = [row for row in summary if str(row["task"]) == task]
        for channel, colour, marker in (
            ("semantic", "#0072B2", "o"),
            ("structural", "#D55E00", "s"),
        ):
            for source, linestyle, suffix in (
                (channel, "-", "raw mass"),
                (f"{channel}_opportunity", "--", "per opportunity"),
            ):
                source_rows = sorted(
                    [row for row in selected if str(row["source"]) == source],
                    key=lambda row: int(row["layer"]),
                )
                if not source_rows:
                    continue
                axis.plot(
                    [int(row["layer"]) for row in source_rows],
                    [float(row["expected_distance_mean"]) for row in source_rows],
                    color=colour,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.6,
                    label=f"{channel}, {suffix}",
                )
        axis.set_xlabel("layer")
        axis.set_title(_task_label(task), fontsize=10)
        if column == 0:
            axis.set_ylabel("expected graph distance")
            axis.legend(frameon=False, fontsize=7)
    figure.suptitle("Raw and opportunity-corrected score reach")
    paths = _save_figure(figure, figures_dir, "08_raw_vs_opportunity_reach")
    plt.close(figure)
    return paths


def _distance_groups(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = ("0", "1", "2", "3", "4-7", "8+")
    observed = {str(row["distance_group"]) for row in rows}
    return [label for label in preferred if label in observed] + sorted(observed - set(preferred))


def _plot_layer_distance_profiles(
    rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
    figures_dir: Path,
    *,
    profile_kind: str,
    stem: str,
) -> list[Path]:
    import matplotlib.pyplot as plt

    selected_rows = [row for row in rows if str(row["profile_kind"]) == profile_kind]
    tasks = [task for task in tasks if any(str(row["task"]) == task for row in selected_rows)]
    if not selected_rows or not tasks:
        return []
    labels = _distance_groups(selected_rows)
    fields = (
        ("semantic_share_mean", "semantic"),
        ("structural_share_mean", "structural"),
        ("structural_minus_semantic_mean", "structural − semantic"),
    )
    positive_values = np.asarray(
        [
            float(row[field])
            for row in selected_rows
            for field in ("semantic_share_mean", "structural_share_mean")
            if np.isfinite(float(row[field]))
        ]
    )
    difference_values = np.asarray(
        [
            abs(float(row["structural_minus_semantic_mean"]))
            for row in selected_rows
            if np.isfinite(float(row["structural_minus_semantic_mean"]))
        ]
    )
    positive_upper = max(float(np.max(positive_values)), 1.0e-12)
    difference_upper = max(float(np.max(difference_values)), 1.0e-12)
    figure, axes = plt.subplots(
        len(tasks),
        3,
        figsize=(10.0, 2.25 * len(tasks) + 0.8),
        squeeze=False,
        constrained_layout=True,
    )
    positive_image = difference_image = None
    for row_index, task in enumerate(tasks):
        task_rows = [row for row in selected_rows if str(row["task"]) == task]
        layers = sorted({int(row["layer"]) for row in task_rows})
        for column, (field, title) in enumerate(fields):
            axis = axes[row_index, column]
            matrix = np.full((len(layers), len(labels)), np.nan)
            lookup = {
                (int(row["layer"]), str(row["distance_group"])): float(row[field])
                for row in task_rows
            }
            for layer_index, layer in enumerate(layers):
                for distance_index, label in enumerate(labels):
                    matrix[layer_index, distance_index] = lookup.get((layer, label), np.nan)
            if column < 2:
                positive_image = axis.imshow(
                    matrix,
                    aspect="auto",
                    interpolation="nearest",
                    cmap="viridis",
                    vmin=0.0,
                    vmax=positive_upper,
                )
            else:
                difference_image = axis.imshow(
                    matrix,
                    aspect="auto",
                    interpolation="nearest",
                    cmap="coolwarm",
                    vmin=-difference_upper,
                    vmax=difference_upper,
                )
            axis.set_xticks(np.arange(len(labels)), labels)
            axis.set_yticks(np.arange(len(layers)), layers)
            if row_index == len(tasks) - 1:
                axis.set_xlabel("distance")
            if column == 0:
                axis.set_ylabel(f"{_task_label(task)}\nlayer")
            if row_index == 0:
                axis.set_title(title, fontsize=10)
    if positive_image is not None:
        figure.colorbar(
            positive_image,
            ax=axes[:, :2],
            fraction=0.016,
            pad=0.012,
            label="mean normalized mass",
        )
    if difference_image is not None:
        figure.colorbar(
            difference_image,
            ax=axes[:, 2],
            fraction=0.032,
            pad=0.012,
            label="mass difference",
        )
    title = (
        "Raw score-mass decomposition"
        if profile_kind == "score_mass"
        else "Opportunity-corrected score decomposition"
    )
    figure.suptitle(title)
    paths = _save_figure(figure, figures_dir, stem)
    plt.close(figure)
    return paths


def _plot_vnode_allocation(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [task for task in tasks if any(str(row["task"]) == task for row in rows)]
    if not rows or not tasks:
        return []
    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.6 * len(tasks), 4.0),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    for column, task in enumerate(tasks):
        axis = axes[0, column]
        selected = [row for row in rows if str(row["task"]) == task]
        for profile_kind, linestyle, suffix in (
            ("score_mass", "-", "raw mass"),
            ("per_opportunity", "--", "per opportunity"),
        ):
            for channel, colour, marker in (
                ("semantic", "#0072B2", "o"),
                ("structural", "#D55E00", "s"),
            ):
                source_rows = sorted(
                    [row for row in selected if str(row["profile_kind"]) == profile_kind],
                    key=lambda row: int(row["layer"]),
                )
                if not source_rows:
                    continue
                axis.plot(
                    [int(row["layer"]) for row in source_rows],
                    [float(row[f"{channel}_virtual_share_mean"]) for row in source_rows],
                    color=colour,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.6,
                    label=f"{channel}, {suffix}",
                )
        axis.set_xlabel("layer")
        axis.set_title(_task_label(task), fontsize=10)
        if column == 0:
            axis.set_ylabel("virtual-node score share")
            axis.legend(frameon=False, fontsize=7)
    figure.suptitle("Virtual-node allocation")
    paths = _save_figure(figure, figures_dir, "11_vnode_allocation")
    plt.close(figure)
    return paths


def _plot_width_difference(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [task for task in tasks if any(str(row["task"]) == task for row in rows)]
    if not rows or not tasks:
        return []
    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.25 * len(tasks), 4.0),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    for column, task in enumerate(tasks):
        axis = axes[0, column]
        selected = sorted(
            [row for row in rows if str(row["task"]) == task],
            key=lambda row: int(row["layer"]),
        )
        layer = np.asarray([int(row["layer"]) for row in selected])
        mean = np.asarray([float(row["structural_minus_semantic_mean"]) for row in selected])
        low = np.asarray([float(row["structural_minus_semantic_low"]) for row in selected])
        high = np.asarray([float(row["structural_minus_semantic_high"]) for row in selected])
        axis.errorbar(
            layer,
            mean,
            yerr=np.vstack((mean - low, high - mean)),
            color="#7A5195",
            marker="o",
            linewidth=1.5,
            capsize=2.5,
        )
        axis.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_xlabel("layer")
        axis.set_title(_task_label(task), fontsize=10)
        if column == 0:
            axis.set_ylabel("structural − semantic variance (hops²)")
    figure.suptitle("Structural excess in spatial width (95% graph-bootstrap CI)")
    paths = _save_figure(figure, figures_dir, "12_structural_minus_semantic_width")
    plt.close(figure)
    return paths


def _plot_width_contributions(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    selected_rows = [row for row in rows if str(row["weighting"]) == "J_weighted"]
    tasks = [task for task in tasks if any(str(row["task"]) == task for row in selected_rows)]
    if not selected_rows or not tasks:
        return []
    labels = _distance_groups(selected_rows)
    values = np.asarray(
        [
            abs(float(row["structural_minus_semantic_width_contribution"]))
            for row in selected_rows
            if np.isfinite(float(row["structural_minus_semantic_width_contribution"]))
        ]
    )
    upper = max(float(np.max(values)), 1.0e-12)
    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.25 * len(tasks), 4.2),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for column, task in enumerate(tasks):
        axis = axes[0, column]
        task_rows = [row for row in selected_rows if str(row["task"]) == task]
        layers = sorted({int(row["layer"]) for row in task_rows})
        lookup = {
            (int(row["layer"]), str(row["distance_group"])): float(
                row["structural_minus_semantic_width_contribution"]
            )
            for row in task_rows
        }
        matrix = np.full((len(layers), len(labels)), np.nan)
        for layer_index, layer in enumerate(layers):
            for distance_index, label in enumerate(labels):
                matrix[layer_index, distance_index] = lookup.get((layer, label), np.nan)
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-upper,
            vmax=upper,
        )
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_yticks(np.arange(len(layers)), layers)
        axis.set_xlabel("distance")
        axis.set_title(_task_label(task), fontsize=10)
        if column == 0:
            axis.set_ylabel("layer")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            fraction=0.018,
            pad=0.012,
            label="structural − semantic width contribution",
        )
    figure.suptitle("Where does structural spatial width arise? (J-weighted heads)")
    paths = _save_figure(figure, figures_dir, "13_width_excess_by_distance")
    plt.close(figure)
    return paths


def _plot_width_by_head_role(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [task for task in tasks if any(str(row["task"]) == task for row in rows)]
    if not rows or not tasks:
        return []
    order = (
        "all_active",
        "semantic_leaning",
        "structural_leaning",
        "central_responsive",
        "other",
        "inactive",
    )
    labels = {
        "all_active": "all active",
        "semantic_leaning": "semantic",
        "structural_leaning": "structural",
        "central_responsive": "generalist",
        "other": "other",
        "inactive": "inactive",
    }
    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.45 * len(tasks), 4.25),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    for column, task in enumerate(tasks):
        axis = axes[0, column]
        task_rows = {str(row["family"]): row for row in rows if str(row["task"]) == task}
        present = [family for family in order if family in task_rows]
        x = np.arange(len(present))
        axis.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.scatter(
            x,
            [float(task_rows[family]["structural_excess_mean"]) for family in present],
            color="#777777",
            marker="o",
            label="equal head",
        )
        axis.scatter(
            x,
            [float(task_rows[family]["structural_excess_J_weighted"]) for family in present],
            color="#7A5195",
            marker="s",
            label="J weighted",
        )
        axis.set_xticks(x, [labels[family] for family in present], rotation=38, ha="right")
        axis.set_title(_task_label(task), fontsize=10)
        if column == 0:
            axis.set_ylabel("structural − semantic variance (hops²)")
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Which head roles produce structural spatial width?")
    paths = _save_figure(figure, figures_dir, "14_width_excess_by_head_role")
    plt.close(figure)
    return paths


def _plot_vnode_cross_layer(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [task for task in tasks if any(str(row["task"]) == task for row in rows)]
    if not rows or not tasks:
        return []
    styles = {
        "semantic_expected_distance": ("#0072B2", "o", "next-layer semantic reach"),
        "structural_expected_distance": ("#D55E00", "s", "next-layer structural reach"),
        "structural_excess_width": ("#7A5195", "^", "next-layer width excess"),
    }
    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(4.1 * len(tasks), 4.0),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    for column, task in enumerate(tasks):
        axis = axes[0, column]
        for outcome, (colour, marker, label) in styles.items():
            selected = sorted(
                [
                    row
                    for row in rows
                    if str(row["task"]) == task and str(row["outcome"]) == outcome
                ],
                key=lambda row: int(row["source_layer"]),
            )
            axis.plot(
                [int(row["source_layer"]) for row in selected],
                [float(row["spearman_rho"]) for row in selected],
                color=colour,
                marker=marker,
                linewidth=1.5,
                label=label,
            )
        axis.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_ylim(-1.02, 1.02)
        axis.set_xlabel("VN source layer")
        axis.set_title(_task_label(task), fontsize=10)
        if column == 0:
            axis.set_ylabel("graphwise Spearman correlation")
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Does semantic VN allocation predict next-layer organisation?")
    paths = _save_figure(figure, figures_dir, "15_vnode_cross_layer_relationships")
    plt.close(figure)
    return paths


def _plot_scale_relationship(
    rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
    figures_dir: Path,
    *,
    outcome: str,
    title: str,
    stem: str,
) -> list[Path]:
    import matplotlib.pyplot as plt

    selected_rows = [row for row in rows if str(row["outcome"]) == outcome]
    tasks = [task for task in tasks if any(str(row["task"]) == task for row in selected_rows)]
    if not selected_rows or not tasks:
        return []
    figure, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(3.25 * len(tasks), 4.0),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    for column, task in enumerate(tasks):
        axis = axes[0, column]
        for scale, colour, marker, label in (
            ("num_nodes", "#0072B2", "o", "number of atoms"),
            ("diameter", "#D55E00", "s", "molecular diameter"),
        ):
            selected = sorted(
                [
                    row
                    for row in selected_rows
                    if str(row["task"]) == task and str(row["scale"]) == scale
                ],
                key=lambda row: int(row["layer"]),
            )
            axis.plot(
                [int(row["layer"]) for row in selected],
                [float(row["spearman_rho"]) for row in selected],
                color=colour,
                marker=marker,
                linewidth=1.5,
                label=label,
            )
        axis.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axis.set_ylim(-1.02, 1.02)
        axis.set_xlabel("layer")
        axis.set_title(_task_label(task), fontsize=10)
        if column == 0:
            axis.set_ylabel("graphwise Spearman correlation")
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(title)
    paths = _save_figure(figure, figures_dir, stem)
    plt.close(figure)
    return paths


def _plot_layer_score_organisation(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [task for task in tasks if any(str(row["task"]) == task for row in rows)]
    if not rows or not tasks:
        return []
    figure, axes = plt.subplots(
        2,
        len(tasks),
        figsize=(3.25 * len(tasks), 6.6),
        squeeze=False,
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    for column, task in enumerate(tasks):
        selected = sorted(
            [row for row in rows if str(row["task"]) == task],
            key=lambda row: int(row["layer"]),
        )
        layers = [int(row["layer"]) for row in selected]
        axes[0, column].plot(
            layers,
            [float(row["joint_sensitivity_share"]) for row in selected],
            color="#5B5F97",
            marker="o",
            linewidth=1.7,
        )
        axes[1, column].plot(
            layers,
            [float(row["joint_weighted_selectivity"]) for row in selected],
            color="#7A5195",
            marker="s",
            linewidth=1.7,
        )
        axes[1, column].axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axes[0, column].set_title(_task_label(task), fontsize=10)
        axes[1, column].set_xlabel("layer")
        if column == 0:
            axes[0, column].set_ylabel("share of total $J$")
            axes[1, column].set_ylabel(r"$J$-weighted $D_{\rm rel}$")
    axes[1, 0].set_ylim(-1.02, 1.02)
    figure.suptitle("Where is semantic–structural computation allocated across depth?")
    paths = _save_figure(figure, figures_dir, "18_layerwise_score_organisation")
    plt.close(figure)
    return paths


def _plot_head_role_score_allocation(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [task for task in tasks if any(str(row["task"]) == task for row in rows)]
    if not rows or not tasks:
        return []
    family_labels = {
        "semantic_leaning": "semantic-leaning",
        "structural_leaning": "structural-leaning",
        "central_responsive": "high-$J$ generalist",
        "other": "other active",
        "inactive": "low-$J$",
    }
    family_colours = {
        "semantic_leaning": "#0072B2",
        "structural_leaning": "#D55E00",
        "central_responsive": "#7A5195",
        "other": "#009E73",
        "inactive": "#AAAAAA",
    }
    panels = (
        ("joint_sensitivity_share", "Total sensitivity $J$"),
        ("semantic_tail_share", r"Semantic score at $d\geq4$"),
        ("structural_tail_share", r"Structural score at $d\geq4$"),
    )
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(15.2, 4.8),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    x = np.arange(len(tasks))
    for column, (field, title) in enumerate(panels):
        axis = axes[0, column]
        bottom = np.zeros(len(tasks), dtype=np.float64)
        for family in ORGANISATION_FAMILIES:
            lookup = {
                str(row["task"]): float(row[field])
                for row in rows
                if str(row["family"]) == family
            }
            values = np.asarray([lookup.get(task, 0.0) for task in tasks])
            values = np.where(np.isfinite(values), values, 0.0)
            axis.bar(
                x,
                values,
                bottom=bottom,
                color=family_colours[family],
                width=0.72,
                label=family_labels[family],
            )
            bottom += values
        axis.set_title(title, fontsize=11)
        axis.set_xticks(
            x,
            [_task_label(task) for task in tasks],
            rotation=28,
            ha="right",
        )
        axis.set_ylim(0.0, 1.02)
        if column == 0:
            axis.set_ylabel("fraction allocated to head family")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=len(ORGANISATION_FAMILIES),
        frameon=False,
        fontsize=8,
    )
    figure.suptitle("How do architectures divide computation across head roles?")
    paths = _save_figure(figure, figures_dir, "19_head_role_score_allocation")
    plt.close(figure)
    return paths


def _plot_score_organisation_similarity(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [task for task in tasks if any(str(row["task_a"]) == task for row in rows)]
    if not rows or not tasks:
        return []
    lookup = {
        (str(row["task_a"]), str(row["task_b"])): row for row in rows
    }
    fields = (
        ("model_wide_similarity", "Model-wide profile similarity"),
        ("layer_resolved_similarity", "Layer-resolved similarity"),
        ("placement_gap", "Profile similarity minus layer similarity"),
    )
    matrices: list[np.ndarray] = []
    for field, _title in fields:
        matrices.append(
            np.asarray(
                [
                    [float(lookup[(left, right)][field]) for right in tasks]
                    for left in tasks
                ],
                dtype=np.float64,
            )
        )
    gap_limit = max(float(np.nanmax(np.abs(matrices[2]))), 0.05)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.0, 5.0),
        squeeze=False,
        constrained_layout=True,
    )
    labels = [_task_label(task) for task in tasks]
    for column, ((field, title), matrix) in enumerate(zip(fields, matrices)):
        del field
        axis = axes[0, column]
        if column < 2:
            image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        else:
            image = axis.imshow(
                matrix, cmap="coolwarm", vmin=-gap_limit, vmax=gap_limit
            )
        axis.set_xticks(np.arange(len(tasks)), labels, rotation=42, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(tasks)), labels, fontsize=8)
        axis.set_title(title, fontsize=11)
        for row_index in range(len(tasks)):
            for task_index in range(len(tasks)):
                value = matrix[row_index, task_index]
                axis.text(
                    task_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=(
                        "white"
                        if abs(value) > (0.55 if column < 2 else 0.55 * gap_limit)
                        else "black"
                    ),
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    figure.suptitle(
        "Do architectures preserve global score geometry while reallocating it across layers?"
    )
    paths = _save_figure(figure, figures_dir, "20_score_organisation_similarity")
    plt.close(figure)
    return paths


def _plot_attention_score_reach(
    rows: Sequence[Mapping[str, Any]],
    mismatch_rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
    figures_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    tasks = [
        task
        for task in tasks
        if any(
            str(row["task"]) == task
            and np.isfinite(float(row.get("attention_expected_distance", np.nan)))
            for row in rows
        )
    ]
    if not tasks:
        return []
    mismatch_lookup = {str(row["task"]): row for row in mismatch_rows}
    finite_reach = np.asarray(
        [
            float(row[field])
            for row in rows
            for field in (
                "attention_expected_distance",
                "semantic_expected_distance",
                "structural_expected_distance",
            )
            if np.isfinite(float(row.get(field, np.nan)))
        ]
    )
    upper = max(float(np.quantile(finite_reach, 0.995)) * 1.06, 1.0)
    exemplar_reach = np.asarray(
        [
            float(row[field])
            for row in mismatch_rows
            for field in (
                "attention_expected_distance",
                "semantic_expected_distance",
                "structural_expected_distance",
            )
            if np.isfinite(float(row.get(field, np.nan)))
        ]
    )
    if exemplar_reach.size:
        upper = max(upper, float(np.max(exemplar_reach)) * 1.06)
    maximum_layer = max(int(row["layer"]) for row in rows)
    norm = Normalize(vmin=0, vmax=max(maximum_layer, 1))
    figure, axes = plt.subplots(
        2,
        len(tasks),
        figsize=(3.25 * len(tasks), 7.1),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    scatter = None
    for column, task in enumerate(tasks):
        task_rows = [row for row in rows if str(row["task"]) == task]
        finite_joint = np.asarray(
            [float(row["joint_sensitivity"]) for row in task_rows], dtype=np.float64
        )
        scale = (
            max(float(np.nanquantile(finite_joint, 0.90)), 1.0e-12)
            if np.isfinite(finite_joint).any()
            else 1.0
        )
        for row_index, (channel, label) in enumerate(
            (("semantic", "semantic score"), ("structural", "structural score"))
        ):
            axis = axes[row_index, column]
            selected = [
                row
                for row in task_rows
                if np.isfinite(float(row.get("attention_expected_distance", np.nan)))
                and np.isfinite(float(row.get(f"{channel}_expected_distance", np.nan)))
            ]
            sizes = 14.0 + 52.0 * np.clip(
                np.asarray([float(row["joint_sensitivity"]) for row in selected]) / scale,
                0.0,
                1.5,
            )
            scatter = axis.scatter(
                [float(row["attention_expected_distance"]) for row in selected],
                [float(row[f"{channel}_expected_distance"]) for row in selected],
                c=[int(row["layer"]) for row in selected],
                s=sizes,
                cmap="viridis",
                norm=norm,
                alpha=0.76,
                edgecolor="white",
                linewidth=0.35,
            )
            exemplar = mismatch_lookup.get(task)
            if exemplar is not None:
                axis.scatter(
                    [float(exemplar["attention_expected_distance"])],
                    [float(exemplar[f"{channel}_expected_distance"])],
                    s=120,
                    marker="*",
                    facecolor="none",
                    edgecolor="#C44E52",
                    linewidth=1.5,
                    zorder=5,
                )
            axis.plot([0.0, upper], [0.0, upper], "--", color="#777777", linewidth=0.8)
            axis.set_xlim(0.0, upper)
            axis.set_ylim(0.0, upper)
            axis.grid(alpha=0.20)
            if row_index == 0:
                severity = (
                    float(exemplar["max_abs_attention_reach_gap"])
                    if exemplar is not None
                    else float("nan")
                )
                suffix = f"\nmax active |Δ|={severity:.2f} hops" if np.isfinite(severity) else ""
                axis.set_title(_task_label(task) + suffix, fontsize=9.5)
            if row_index == 1:
                axis.set_xlabel("attention reach (hops)")
            if column == 0:
                axis.set_ylabel(f"{label} reach (hops)")
    if scatter is not None:
        figure.colorbar(
            scatter,
            ax=axes,
            fraction=0.014,
            pad=0.012,
            label="layer",
        )
    figure.suptitle(
        "Per-head attention reach versus intervention-defined score reach "
        "(size: $J$; star: selected mismatch)"
    )
    paths = _save_figure(figure, figures_dir, "21_attention_vs_score_reach")
    plt.close(figure)
    return paths


def _plot_head_reach_gap(
    rows: Sequence[Mapping[str, Any]],
    mismatch_rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
    figures_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fields = (
        ("semantic_attention_reach_gap", "semantic"),
        ("structural_attention_reach_gap", "structural"),
    )
    values = np.asarray(
        [
            abs(float(row[field]))
            for row in rows
            for field, _label in fields
            if np.isfinite(float(row.get(field, np.nan)))
        ]
    )
    if not values.size:
        return []
    upper = max(float(np.quantile(values, 0.98)), 0.25)
    mismatch_lookup = {str(row["task"]): row for row in mismatch_rows}
    figure, axes = plt.subplots(
        2,
        len(tasks),
        figsize=(3.25 * len(tasks), 7.0),
        squeeze=False,
        sharex="col",
        sharey="row",
        constrained_layout=True,
    )
    image = None
    for column, task in enumerate(tasks):
        for row_index, (field, label) in enumerate(fields):
            axis = axes[row_index, column]
            matrix, layers, heads = _matrix(rows, task, field)
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-upper,
                vmax=upper,
            )
            exemplar = mismatch_lookup.get(task)
            if exemplar is not None:
                layer = int(exemplar["layer"])
                head = int(exemplar["head"])
                if layer in layers and head in heads:
                    axis.add_patch(
                        Rectangle(
                            (heads.index(head) - 0.5, layers.index(layer) - 0.5),
                            1.0,
                            1.0,
                            fill=False,
                            edgecolor="#111111",
                            linewidth=1.5,
                        )
                    )
            axis.set_xticks(np.arange(len(heads)), heads, fontsize=7)
            axis.set_yticks(np.arange(len(layers)), layers, fontsize=8)
            if row_index == 0:
                severity = (
                    float(exemplar["max_abs_attention_reach_gap"])
                    if exemplar is not None
                    else float("nan")
                )
                suffix = f"\nmax active |Δ|={severity:.2f}" if np.isfinite(severity) else ""
                axis.set_title(_task_label(task) + suffix, fontsize=9.5)
            if row_index == 1:
                axis.set_xlabel("head")
            if column == 0:
                axis.set_ylabel(f"{label} gap\nlayer")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            fraction=0.014,
            pad=0.012,
            label="score reach − attention reach (hops)",
        )
    figure.suptitle(
        "Head-level reach gap (outlined cell: strongest active mismatch per model)"
    )
    paths = _save_figure(figure, figures_dir, "22_head_attention_score_reach_gap")
    plt.close(figure)
    return paths


def _plot_head_score_landscapes(
    rows: Sequence[Mapping[str, Any]],
    mismatch_rows: Sequence[Mapping[str, Any]],
    tasks: Sequence[str],
    figures_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if not rows or not tasks:
        return []
    mismatch_lookup = {str(row["task"]): row for row in mismatch_rows}
    score_values = np.asarray(
        [
            float(row[field])
            for row in rows
            for field in ("normalized_semantic_score", "normalized_structural_score")
            if np.isfinite(float(row[field]))
        ]
    )
    score_upper = max(float(np.quantile(score_values, 0.995)) * 1.08, 1.0)
    joint_values = np.asarray(
        [
            float(row["joint_sensitivity"])
            for row in rows
            if np.isfinite(float(row["joint_sensitivity"]))
        ]
    )
    joint_upper = max(float(np.quantile(joint_values, 0.995)) * 1.08, 1.0)
    maximum_layer = max(int(row["layer"]) for row in rows)
    norm = Normalize(vmin=0, vmax=max(maximum_layer, 1))
    figure, axes = plt.subplots(
        2,
        len(tasks),
        figsize=(3.25 * len(tasks), 7.1),
        squeeze=False,
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )
    scatter = None
    for column, task in enumerate(tasks):
        selected = [row for row in rows if str(row["task"]) == task]
        task_joint = np.asarray([float(row["joint_sensitivity"]) for row in selected])
        scale = max(float(np.nanquantile(task_joint, 0.90)), 1.0e-12)
        sizes = 14.0 + 48.0 * np.clip(task_joint / scale, 0.0, 1.5)
        layers = [int(row["layer"]) for row in selected]
        scatter = axes[0, column].scatter(
            [float(row["normalized_semantic_score"]) for row in selected],
            [float(row["normalized_structural_score"]) for row in selected],
            c=layers,
            s=sizes,
            cmap="viridis",
            norm=norm,
            alpha=0.76,
            edgecolor="white",
            linewidth=0.35,
        )
        axes[0, column].plot(
            [0.0, score_upper],
            [0.0, score_upper],
            "--",
            color="#777777",
            linewidth=0.8,
        )
        axes[1, column].scatter(
            [float(row["selectivity"]) for row in selected],
            [float(row["joint_sensitivity"]) for row in selected],
            c=layers,
            s=sizes,
            cmap="viridis",
            norm=norm,
            alpha=0.76,
            edgecolor="white",
            linewidth=0.35,
        )
        exemplar = mismatch_lookup.get(task)
        if exemplar is not None:
            axes[0, column].scatter(
                [float(exemplar["normalized_semantic_score"])],
                [float(exemplar["normalized_structural_score"])],
                s=120,
                marker="*",
                facecolor="none",
                edgecolor="#C44E52",
                linewidth=1.5,
                zorder=5,
            )
            axes[1, column].scatter(
                [float(exemplar["selectivity"])],
                [float(exemplar["joint_sensitivity"])],
                s=120,
                marker="*",
                facecolor="none",
                edgecolor="#C44E52",
                linewidth=1.5,
                zorder=5,
            )
        axes[0, column].set_title(_task_label(task), fontsize=10)
        axes[0, column].set_xlim(0.0, score_upper)
        axes[0, column].set_ylim(0.0, score_upper)
        axes[1, column].set_xlim(-1.02, 1.02)
        axes[1, column].set_ylim(0.0, joint_upper)
        axes[1, column].axvline(0.0, color="#777777", linestyle="--", linewidth=0.8)
        axes[0, column].set_xlabel("normalised semantic score")
        axes[1, column].set_xlabel(r"relative selectivity $D_{\rm rel}$")
        if column == 0:
            axes[0, column].set_ylabel("normalised structural score")
            axes[1, column].set_ylabel(r"joint sensitivity $J$")
    if scatter is not None:
        figure.colorbar(
            scatter,
            ax=axes,
            fraction=0.014,
            pad=0.012,
            label="layer",
        )
    figure.suptitle(
        "Head score landscapes across architectures "
        "(size: $J$; star: selected reach mismatch)"
    )
    paths = _save_figure(figure, figures_dir, "23_head_score_landscapes")
    plt.close(figure)
    return paths


def _plot_score_carriage(
    profile_rows: Sequence[Mapping[str, Any]], tasks: Sequence[str], figures_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    if not any(str(row["source"]).endswith("_carriage") for row in profile_rows):
        return []
    figure, axes = plt.subplots(
        2,
        len(tasks),
        figsize=(3.25 * len(tasks), 7.0),
        squeeze=False,
        sharey="row",
        constrained_layout=True,
    )
    for row_index, channel in enumerate(CHANNELS):
        for column, task in enumerate(tasks):
            axis = axes[row_index, column]
            selected = [row for row in profile_rows if str(row["task"]) == task]
            sources = (f"{channel}_score", f"{channel}_carriage")
            labels = []
            for source in sources:
                labels.extend(
                    str(row["distance"]) for row in selected if str(row["source"]) == source
                )
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
    for stale_path in (
        output_dir / "receptive_field_normalized_reach.csv",
        figures_dir / "12_receptive_field_normalized_reach.png",
        figures_dir / "12_receptive_field_normalized_reach.pdf",
        figures_dir / "13_structural_minus_semantic_width.png",
        figures_dir / "13_structural_minus_semantic_width.pdf",
    ):
        if stale_path.is_file():
            stale_path.unlink()
    inventory_rows = inventory(roots, tasks, seed=int(seed))
    _write_csv(output_dir / "cache_inventory.csv", inventory_rows)
    models, warnings = load_models(roots, tasks, seed=int(seed))
    if not models:
        raise FileNotFoundError("no usable score caches were found for the requested tasks")
    if verbose:
        for warning in warnings:
            print(f"[chapter6:warning] {warning}", flush=True)
        print(
            "[chapter6:load] "
            + ", ".join(f"{model.task} ({model.score_path})" for model in models),
            flush=True,
        )
    head_rows = head_metrics(models)
    mismatch_summary_rows = reach_mismatch_summary(
        head_rows, activity_quantile=activity_quantile
    )
    mismatch_rows = representative_reach_mismatches(
        head_rows, activity_quantile=activity_quantile
    )
    layer_rows = layer_summary(head_rows)
    representative_rows = representative_heads(head_rows, activity_quantile=activity_quantile)
    profile_rows = model_profiles(models)
    uncertainty_rows = uncertainty_profiles(models)
    distance_rows = layer_distance_profiles(models)
    vnode_rows = vnode_layer_profiles(models)
    width_bootstrap_rows = spatial_width_bootstrap(models)
    width_contribution_rows = width_contribution_profiles(models)
    width_role_rows = width_by_head_role(head_rows)
    layer_organisation_rows = layer_score_organisation(head_rows)
    role_allocation_rows = head_role_score_allocation(models, head_rows)
    organisation_similarity_rows = score_organisation_similarity(
        distance_rows, layer_organisation_rows, [model.task for model in models]
    )
    graph_rows = graph_spatial_metrics(models)
    vnode_cross_layer_rows = vnode_cross_layer_relationships(graph_rows)
    scale_rows = molecular_scale_relationships(graph_rows)
    tables = {
        "head_spatial_metrics.csv": head_rows,
        "head_reach_mismatch_summary.csv": mismatch_summary_rows,
        "representative_reach_mismatches.csv": mismatch_rows,
        "layer_spatial_summary.csv": layer_rows,
        "representative_heads.csv": representative_rows,
        "model_distance_profiles.csv": profile_rows,
        "score_profile_uncertainty.csv": uncertainty_rows,
        "layer_distance_profiles.csv": distance_rows,
        "vnode_layer_allocation.csv": vnode_rows,
        "spatial_width_graph_bootstrap.csv": width_bootstrap_rows,
        "width_contributions_by_distance.csv": width_contribution_rows,
        "width_by_head_role.csv": width_role_rows,
        "layer_score_organisation.csv": layer_organisation_rows,
        "head_role_score_allocation.csv": role_allocation_rows,
        "score_organisation_similarity.csv": organisation_similarity_rows,
        "graph_spatial_metrics.csv": graph_rows,
        "vnode_cross_layer_relationships.csv": vnode_cross_layer_rows,
        "molecular_scale_relationships.csv": scale_rows,
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
    figures.extend(_plot_alignment_head_roles(head_rows, available_tasks, figures_dir))
    figures.extend(_plot_score_carriage(profile_rows, available_tasks, figures_dir))
    figures.extend(_plot_raw_opportunity_reach(layer_rows, available_tasks, figures_dir))
    figures.extend(
        _plot_layer_distance_profiles(
            distance_rows,
            available_tasks,
            figures_dir,
            profile_kind="score_mass",
            stem="09_layer_distance_profiles_raw",
        )
    )
    figures.extend(
        _plot_layer_distance_profiles(
            distance_rows,
            available_tasks,
            figures_dir,
            profile_kind="per_opportunity",
            stem="10_layer_distance_profiles_opportunity",
        )
    )
    figures.extend(_plot_vnode_allocation(vnode_rows, available_tasks, figures_dir))
    figures.extend(_plot_width_difference(width_bootstrap_rows, available_tasks, figures_dir))
    figures.extend(_plot_width_contributions(width_contribution_rows, available_tasks, figures_dir))
    figures.extend(_plot_width_by_head_role(width_role_rows, available_tasks, figures_dir))
    figures.extend(_plot_vnode_cross_layer(vnode_cross_layer_rows, available_tasks, figures_dir))
    figures.extend(
        _plot_scale_relationship(
            scale_rows,
            available_tasks,
            figures_dir,
            outcome="structural_excess_width",
            title="Does molecular scale predict structural spatial width?",
            stem="16_scale_and_structural_width",
        )
    )
    figures.extend(
        _plot_scale_relationship(
            scale_rows,
            available_tasks,
            figures_dir,
            outcome="semantic_vnode_share",
            title="Does molecular scale predict semantic VN allocation?",
            stem="17_scale_and_vnode_allocation",
        )
    )
    figures.extend(
        _plot_layer_score_organisation(
            layer_organisation_rows, available_tasks, figures_dir
        )
    )
    figures.extend(
        _plot_head_role_score_allocation(
            role_allocation_rows, available_tasks, figures_dir
        )
    )
    figures.extend(
        _plot_score_organisation_similarity(
            organisation_similarity_rows, available_tasks, figures_dir
        )
    )
    figures.extend(
        _plot_attention_score_reach(
            head_rows, mismatch_rows, available_tasks, figures_dir
        )
    )
    figures.extend(
        _plot_head_reach_gap(head_rows, mismatch_rows, available_tasks, figures_dir)
    )
    figures.extend(
        _plot_head_score_landscapes(
            head_rows, mismatch_rows, available_tasks, figures_dir
        )
    )
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
                    isinstance(
                        model.score["channels"][channel].get("graph_distance_contribution"), Mapping
                    )
                    for channel in CHANNELS
                ),
                "carriage": None if model.carriage_path is None else str(model.carriage_path),
            }
            for model in models
        ],
        "figures": [str(path) for path in figures],
        "interpretation": {
            "alignment": (
                "overlap of normalized within-head profiles, equal to 1 minus "
                "total variation"
            ),
            "expected_distance": (
                "molecular-only expected graph distance; virtual carriers remain "
                "separate"
            ),
            "spatial_width": "variance of normalized molecular score mass over graph distance",
            "uncertainty": "standard error of expected distance across cached held-out graphs",
            "width_interval": "paired graph-bootstrap interval; heads remain fixed",
            "width_contribution": (
                "per-distance contribution to molecular profile variance around each "
                "head's own expected distance"
            ),
            "head_weighting": (
                "equal-head and joint-sensitivity-weighted summaries are reported "
                "separately"
            ),
            "opportunity_correction": (
                "score mass per available source-carrier opportunity in each distance "
                "shell"
            ),
            "virtual_node": "reported separately because it has no molecular graph distance",
            "cross_layer": (
                "graphwise association between semantic VN allocation at layer l and "
                "molecular organisation at layer l+1"
            ),
            "molecular_scale": (
                "graphwise Spearman association with cached carrier count and "
                "molecular diameter"
            ),
            "score_organisation": (
                "total J allocation over layers and established head families; "
                "long-range tails use molecular distances d >= 4"
            ),
            "organisation_similarity": (
                "one minus Jensen--Shannon distance, comparing J-weighted score "
                "profiles before and after retaining layer identity"
            ),
            "reach_gap": (
                "molecular score expected distance minus clean-attention expected "
                "distance; virtual-node mass remains separate"
            ),
            "attention": "clean attention mass by graph distance, not a causal score",
            "carriage": (
                "final-state response under the same intervention family, not task "
                "necessity"
            ),
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
        "mismatch_summary_rows": mismatch_summary_rows,
        "mismatch_rows": mismatch_rows,
        "layer_rows": layer_rows,
        "representative_rows": representative_rows,
        "distance_rows": distance_rows,
        "vnode_rows": vnode_rows,
        "width_bootstrap_rows": width_bootstrap_rows,
        "width_contribution_rows": width_contribution_rows,
        "width_role_rows": width_role_rows,
        "layer_organisation_rows": layer_organisation_rows,
        "role_allocation_rows": role_allocation_rows,
        "organisation_similarity_rows": organisation_similarity_rows,
        "graph_rows": graph_rows,
        "vnode_cross_layer_rows": vnode_cross_layer_rows,
        "scale_rows": scale_rows,
    }


__all__ = [
    "ANALYSIS_VERSION",
    "SpatialModel",
    "graph_spatial_metrics",
    "head_metrics",
    "head_role_score_allocation",
    "inventory",
    "layer_distance_profiles",
    "layer_score_organisation",
    "layer_summary",
    "load_models",
    "model_profiles",
    "molecular_scale_relationships",
    "reach_mismatch_summary",
    "representative_heads",
    "representative_reach_mismatches",
    "run",
    "score_organisation_similarity",
    "spatial_width_bootstrap",
    "uncertainty_profiles",
    "vnode_cross_layer_relationships",
    "vnode_layer_profiles",
    "width_by_head_role",
    "width_contribution_profiles",
]
