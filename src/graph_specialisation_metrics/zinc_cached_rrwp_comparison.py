"""Cache-only comparison of the six trained ZINC GRIT variants.

This module never constructs a model, loads a checkpoint, preprocesses ZINC, or
recomputes a specialisation score.  It validates the immutable canonical
``scores/raw.pt`` and optional ``carriage/fields.pt`` artifacts, then produces a
small cross-architecture comparison of performance, raw learned-head scores,
distance-resolved score mass, profile interval width, and Functional carriage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .methodology.bootstrap import trimmed_mean
from .methodology.cache import (
    ReadOnlyCacheArtifact,
    StaleCacheError,
    checkpoint_sha256,
    load_cache_artifact_file,
)
from .methodology.grit_figure_data import (
    CanonicalHeadMetrics,
    load_canonical_model_record,
    load_canonical_score_artifact,
)
from .methodology.protocol import PROTOCOL_VERSION, stable_hash

ANALYSIS_VERSION = "zinc-cached-rrwp-comparison-v4"
ALIGNMENT_CACHE_VERSION = "zinc-head-profile-colocalization-v3"
SUPPORTED_CACHE_PROTOCOLS = (
    "donor-swap-specialisation-carriage-v3",
    PROTOCOL_VERSION,
)
TASKS = (
    "zinc_1hop_localrrwp",
    "zinc_1hop",
    "zinc_1hop_vnode",
    "zinc_2hop",
    "zinc_2hop_vnode",
    "zinc",
)
TASK_ARTIFACT_ALIASES = {
    # Historical canonical runs used this name. Both registrations reconstruct
    # the same parameter-matched local-RRWP checkpoint, but the stored task
    # identity must still be validated under its original name.
    "zinc_1hop_localrrwp": ("zinc_1hop_localrrwp", "zinc_1hop_local"),
}
TASK_LABELS = {
    "zinc_1hop_localrrwp": "1-hop\nlocal RRWP",
    "zinc_1hop_local": "1-hop\nlocal RRWP",
    "zinc_1hop": "1-hop\nglobal RRWP",
    "zinc_1hop_vnode": "1-hop + VN",
    "zinc_2hop": "2-hop",
    "zinc_2hop_vnode": "2-hop + VN",
    "zinc": "Dense GRIT",
}
TASK_COLOURS = {
    "zinc_1hop_localrrwp": "#E69F00",
    "zinc_1hop_local": "#E69F00",
    "zinc_1hop": "#0072B2",
    "zinc_1hop_vnode": "#CC79A7",
    "zinc_2hop": "#009E73",
    "zinc_2hop_vnode": "#56B4E9",
    "zinc": "#D55E00",
}
TASK_MARKERS = {
    "zinc_1hop_localrrwp": "o",
    "zinc_1hop_local": "o",
    "zinc_1hop": "s",
    "zinc_1hop_vnode": "^",
    "zinc_2hop": "D",
    "zinc_2hop_vnode": "v",
    "zinc": "P",
}
TASK_LINESTYLES = {
    "zinc_1hop_localrrwp": "-",
    "zinc_1hop_local": "-",
    "zinc_1hop": "--",
    "zinc_1hop_vnode": "-.",
    "zinc_2hop": ":",
    "zinc_2hop_vnode": (0, (5, 2)),
    "zinc": (0, (2, 1)),
}
CHANNELS = ("semantic", "structural")
DISPLAY_BINS = (
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4-7", 4, 7),
    ("8+", 8, math.inf),
)


@dataclass(frozen=True)
class CachedModel:
    task: str
    artifact_task: str
    root: Path
    score_artifact: ReadOnlyCacheArtifact
    score: Mapping[str, Any]
    model_record: Mapping[str, Any]
    carriage_artifact: ReadOnlyCacheArtifact | None
    carriage: Mapping[str, Any] | None


def _as_numpy(value: Any, *, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _task_dir(root: Path, task: str, train_seed: int) -> Path:
    return root / task / f"seed_{int(train_seed)}"


def artifact_task_candidates(task: str) -> tuple[str, ...]:
    return TASK_ARTIFACT_ALIASES.get(str(task), (str(task),))


def load_compatible_cache_artifact_file(path: str | Path) -> ReadOnlyCacheArtifact:
    """Load current caches or a fingerprint-valid v3 artifact read-only.

    The canonical loader remains fail-closed. This compatibility boundary is
    intentionally local to the descriptive ZINC comparison and never rewrites,
    upgrades, or relabels an older artifact.
    """

    try:
        return load_cache_artifact_file(path)
    except StaleCacheError:
        import torch

        resolved = Path(path)
        try:
            payload = torch.load(resolved, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError) as error:
            raise StaleCacheError(f"unreadable cache {resolved}") from error
        if not isinstance(payload, Mapping) or not {"metadata", "value"} <= set(payload):
            raise StaleCacheError(f"cache payload is malformed: {resolved}")
        metadata = payload["metadata"]
        protocol = metadata.get("protocol_version")
        if protocol != "donor-swap-specialisation-carriage-v3":
            raise
        contract = metadata.get("contract")
        if not isinstance(contract, Mapping):
            raise StaleCacheError(f"cache contract is malformed: {resolved}")
        complete_contract = dict(contract)
        scientific_contract = dict(complete_contract)
        scientific_contract.pop("repository_commit", None)
        complete_fingerprint = stable_hash(complete_contract)
        scientific_fingerprint = stable_hash(scientific_contract)
        claimed = metadata.get("contract_fingerprint")
        if claimed not in {complete_fingerprint, scientific_fingerprint}:
            raise StaleCacheError(
                f"legacy cache contract fingerprint is inconsistent: {resolved}"
            )
        provenance = metadata.get("provenance_fingerprint")
        if provenance is not None and provenance != complete_fingerprint:
            raise StaleCacheError(
                f"legacy cache provenance fingerprint is inconsistent: {resolved}"
            )
        return ReadOnlyCacheArtifact(
            path=resolved,
            file_sha256=checkpoint_sha256(resolved),
            metadata=metadata,
            value=payload["value"],
        )


def load_compatible_score_artifact(
    path: str | Path, *, expected_task: str
) -> ReadOnlyCacheArtifact:
    try:
        return load_canonical_score_artifact(path, expected_task=expected_task)
    except StaleCacheError:
        artifact = load_compatible_cache_artifact_file(path)
    contract = artifact.metadata.get("contract")
    if not isinstance(contract, Mapping) or contract.get("task") != expected_task:
        raise StaleCacheError(
            f"{artifact.path} is not a valid {expected_task!r} score artifact"
        )
    required = {
        "checkpoint_sha256",
        "model_geometry",
        "sigma",
        "split_fingerprint",
        "task_adapter_version",
        "train_seed",
    }
    missing = sorted(required.difference(contract))
    if missing:
        raise StaleCacheError(
            f"legacy score cache contract is missing {missing}: {artifact.path}"
        )
    return artifact


def cache_inventory(
    roots: Sequence[Path],
    *,
    tasks: Sequence[str] = TASKS,
    train_seed: int = 42,
) -> list[dict[str, Any]]:
    """Return an exact-path inventory without loading any artifact."""

    rows: list[dict[str, Any]] = []
    for task in tasks:
        matches = []
        for root in roots:
            for artifact_task in artifact_task_candidates(str(task)):
                task_dir = _task_dir(Path(root), artifact_task, int(train_seed))
                score = task_dir / "cache" / "scores" / "raw.pt"
                carriage = task_dir / "cache" / "carriage" / "fields.pt"
                model = task_dir / "model.json"
                if score.is_file() or model.is_file() or carriage.is_file():
                    matches.append(
                        {
                            "root": str(Path(root)),
                            "artifact_task": artifact_task,
                            "score": str(score),
                            "score_exists": score.is_file(),
                            "model": str(model),
                            "model_exists": model.is_file(),
                            "carriage": str(carriage),
                            "carriage_exists": carriage.is_file(),
                        }
                    )
        rows.append(
            {
                "task": str(task),
                "train_seed": int(train_seed),
                "complete_score_locations": sum(
                    bool(row["score_exists"] and row["model_exists"]) for row in matches
                ),
                "matches": matches,
            }
        )
    return rows


def _resolve_task_root(
    roots: Sequence[Path], task: str, train_seed: int
) -> tuple[Path, str]:
    for artifact_task in artifact_task_candidates(task):
        candidates: list[tuple[Path, ReadOnlyCacheArtifact]] = []
        for root in roots:
            task_dir = _task_dir(Path(root), artifact_task, train_seed)
            score = task_dir / "cache" / "scores" / "raw.pt"
            model = task_dir / "model.json"
            if score.is_file() and model.is_file():
                candidates.append(
                    (
                        Path(root),
                        load_compatible_score_artifact(
                            score, expected_task=artifact_task
                        ),
                    )
                )
        if not candidates:
            continue
        protocol_rank = {
            protocol: rank for rank, protocol in enumerate(SUPPORTED_CACHE_PROTOCOLS)
        }
        best_rank = max(
            protocol_rank.get(
                str(artifact.metadata.get("protocol_version")), -1
            )
            for _, artifact in candidates
        )
        preferred = [
            (root, artifact)
            for root, artifact in candidates
            if protocol_rank.get(
                str(artifact.metadata.get("protocol_version")), -1
            )
            == best_rank
        ]
        fingerprints = {
            str(artifact.metadata["contract_fingerprint"])
            for _, artifact in preferred
        }
        if len(fingerprints) != 1:
            locations = [str(root) for root, _ in preferred]
            protocol = preferred[0][1].metadata.get("protocol_version")
            raise RuntimeError(
                f"multiple non-identical {protocol} score caches found for "
                f"{artifact_task}: {locations}"
            )
        return preferred[0][0], artifact_task
    expected = "\n".join(
        str(_task_dir(Path(root), candidate, train_seed) / "cache/scores/raw.pt")
        for candidate in artifact_task_candidates(task)
        for root in roots
    )
    raise FileNotFoundError(
        f"no complete canonical score cache for {task!r}, seed {train_seed}; "
        f"checked:\n{expected}"
    )


def _validate_carriage(
    artifact: ReadOnlyCacheArtifact,
    score_artifact: ReadOnlyCacheArtifact,
    *,
    task: str,
    train_seed: int,
) -> None:
    contract = artifact.metadata.get("contract", {})
    score_contract = score_artifact.metadata.get("contract", {})
    expected = {
        "task": task,
        "train_seed": int(train_seed),
        "checkpoint_sha256": score_contract.get("checkpoint_sha256"),
        "model_geometry": score_contract.get("model_geometry"),
        "split_fingerprint": score_contract.get("split_fingerprint"),
    }
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"carriage cache does not match score cache for {task}: {mismatches}"
        )


def load_cached_models(
    roots: Sequence[Path],
    *,
    tasks: Sequence[str] = TASKS,
    train_seed: int = 42,
    require_carriage: bool = False,
) -> list[CachedModel]:
    """Load and contract-validate canonical caches for every requested task."""

    result: list[CachedModel] = []
    for task in tasks:
        root, artifact_task = _resolve_task_root(roots, str(task), int(train_seed))
        task_dir = _task_dir(root, artifact_task, int(train_seed))
        score_artifact = load_compatible_score_artifact(
            task_dir / "cache/scores/raw.pt", expected_task=artifact_task
        )
        model_record = load_canonical_model_record(
            task_dir / "model.json", score_artifact
        )
        carriage_path = task_dir / "cache/carriage/fields.pt"
        carriage_artifact = None
        carriage = None
        if carriage_path.is_file():
            carriage_artifact = load_compatible_cache_artifact_file(carriage_path)
            _validate_carriage(
                carriage_artifact,
                score_artifact,
                task=artifact_task,
                train_seed=int(train_seed),
            )
            carriage = carriage_artifact.value
        elif require_carriage:
            raise FileNotFoundError(f"canonical carriage cache not found: {carriage_path}")
        result.append(
            CachedModel(
                task=str(task),
                artifact_task=artifact_task,
                root=root,
                score_artifact=score_artifact,
                score=score_artifact.value,
                model_record=model_record,
                carriage_artifact=carriage_artifact,
                carriage=carriage,
            )
        )
    return result


def _numeric_distance(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if number.is_integer() and number >= 0 else None


def display_labels(axis: Sequence[Any], *, include_special: bool = True) -> tuple[str, ...]:
    labels = [name for name, _, _ in DISPLAY_BINS]
    if include_special:
        labels.extend(
            sorted(
                {
                    str(value).replace("_", " ")
                    for value in axis
                    if _numeric_distance(value) is None
                }
            )
        )
    return tuple(labels)


def group_distance(values: Any, axis: Sequence[Any]) -> tuple[tuple[str, ...], np.ndarray]:
    """Sum a trailing exact-distance axis into shared cross-model display bins."""

    array = _as_numpy(values)
    if array.shape[-1] != len(axis):
        raise ValueError(
            f"distance array has width {array.shape[-1]} for {len(axis)} labels"
        )
    grouped: list[np.ndarray] = []
    labels: list[str] = []
    for name, lower, upper in DISPLAY_BINS:
        positions = [
            index
            for index, value in enumerate(axis)
            if (distance := _numeric_distance(value)) is not None
            and lower <= distance <= upper
        ]
        labels.append(name)
        grouped.append(
            array[..., positions].sum(axis=-1)
            if positions
            else np.zeros(array.shape[:-1], dtype=np.float64)
        )
    specials = sorted(
        {
            str(value)
            for value in axis
            if _numeric_distance(value) is None
        }
    )
    for special in specials:
        positions = [index for index, value in enumerate(axis) if str(value) == special]
        labels.append(special.replace("_", " "))
        grouped.append(array[..., positions].sum(axis=-1))
    return tuple(labels), np.stack(grouped, axis=-1)


def _normalise_last_axis(values: Any) -> np.ndarray:
    array = _as_numpy(values)
    denominator = np.nansum(array, axis=-1, keepdims=True)
    output = np.full_like(array, np.nan, dtype=np.float64)
    np.divide(array, denominator, out=output, where=denominator > 0)
    return output


def _finite_correlation(left: Any, right: Any) -> float:
    left_array = _as_numpy(left).reshape(-1)
    right_array = _as_numpy(right).reshape(-1)
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    if finite.sum() < 2:
        return float("nan")
    left_array = left_array[finite]
    right_array = right_array[finite]
    if np.std(left_array) <= 0 or np.std(right_array) <= 0:
        return float("nan")
    return float(np.corrcoef(left_array, right_array)[0, 1])


def score_profile(
    score: Mapping[str, Any], channel: str
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return equal-head and activity-weighted normalized distance profiles."""

    axis = tuple(score["axis"])
    channel_score = score["channels"][channel]
    exact_head = _as_numpy(channel_score["heatmap_exact_head"])
    raw = _as_numpy(channel_score["raw"])
    if exact_head.ndim != 3 or exact_head.shape[:2] != raw.shape:
        raise ValueError(
            f"{channel} exact head profile has shape {exact_head.shape}, raw={raw.shape}"
        )
    reconstructed = exact_head.sum(axis=-1)
    if not np.allclose(reconstructed, raw, atol=1.0e-7, rtol=1.0e-5):
        error = float(np.nanmax(np.abs(reconstructed - raw)))
        raise RuntimeError(
            f"{channel} distance profile does not reconstruct raw scores; max error={error}"
        )
    labels, grouped = group_distance(exact_head, axis)
    equal_head = np.nanmean(_normalise_last_axis(grouped).reshape(-1, len(labels)), axis=0)
    activity_weighted = _normalise_last_axis(grouped.sum(axis=(0, 1)))
    return labels, equal_head, activity_weighted


def relative_profile_interval_width(score: Mapping[str, Any], channel: str) -> float:
    """Mean marginal 95% CI width relative to total architecture score mass."""

    channel_score = score["channels"][channel]
    interval = channel_score.get("distance_intervals")
    if interval is None:
        return float("nan")
    estimate = _as_numpy(_field(interval, "estimate"))
    low = _as_numpy(_field(interval, "low"))
    high = _as_numpy(_field(interval, "high"))
    # [estimand=(mass, per-opportunity), layer+sum, distance]
    if estimate.ndim != 3 or estimate.shape[0] < 1:
        raise ValueError(f"unexpected distance interval shape {estimate.shape}")
    point = estimate[0, -1]
    width = high[0, -1] - low[0, -1]
    reportable = np.ones(point.shape, dtype=bool)
    support = channel_score.get("distance_support")
    if support is not None and "reportable" in support:
        reportable &= _as_numpy(support["reportable"], dtype=bool)
    reportable &= np.isfinite(point) & np.isfinite(width)
    total = float(np.nansum(point[reportable]))
    if total <= 0 or not np.any(reportable):
        return float("nan")
    return float(np.mean(width[reportable] / total))


def score_interval_width_profile(
    score: Mapping[str, Any], channel: str
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Return cached exact-distance marginal CI widths on a common score scale.

    Widths are divided by the total reportable architecture-level point mass.
    This preserves the original canonical marginal intervals: unlike the point
    profiles, adjacent interval endpoints are never added after bootstrap.
    """

    axis = tuple(str(label).replace("_", " ") for label in score["axis"])
    channel_score = score["channels"][channel]
    interval = channel_score.get("distance_intervals")
    if interval is None:
        return axis, np.full(len(axis), np.nan), np.zeros(len(axis), dtype=bool)
    estimate = _as_numpy(_field(interval, "estimate"))
    low = _as_numpy(_field(interval, "low"))
    high = _as_numpy(_field(interval, "high"))
    if estimate.ndim != 3 or estimate.shape[0] < 1 or estimate.shape[-1] != len(axis):
        raise ValueError(f"unexpected distance interval shape {estimate.shape}")
    point = estimate[0, -1]
    width = high[0, -1] - low[0, -1]
    reportable = np.ones(point.shape, dtype=bool)
    support = channel_score.get("distance_support")
    if support is not None and "reportable" in support:
        reportable &= _as_numpy(support["reportable"], dtype=bool)
    reportable &= np.isfinite(point) & np.isfinite(width)
    total = float(np.nansum(point[reportable]))
    relative = np.full(point.shape, np.nan, dtype=np.float64)
    if total > 0:
        relative[reportable] = width[reportable] / total
    return axis, relative, reportable


def _carriage_bin(row: Mapping[str, Any]) -> str | None:
    distance = float(row["distance"])
    if np.isfinite(distance):
        integer = int(distance)
        for name, lower, upper in DISPLAY_BINS:
            if lower <= integer <= upper:
                return name
        return None
    kind = str(row.get("carrier_kind", "molecular_node"))
    return None if kind == "molecular_node" else kind.replace("_", " ")


def carriage_profile(
    carriage: Mapping[str, Any], channel: str, *, effect_floor: float = 1.0e-8
) -> tuple[tuple[str, ...], np.ndarray, float, int]:
    """Return canonical event-normalized Functional-carriage geometry.

    Carriers are normalized within each donor event before donor/source/graph
    aggregation.  The final graph population is summarized with the canonical
    20% trimmed mean.  No new uncertainty interval is estimated here.
    """

    rows = list(carriage["channels"][channel]["pairs"])
    if not rows:
        return (
            tuple(name for name, _, _ in DISPLAY_BINS),
            np.full(len(DISPLAY_BINS), np.nan),
            float("nan"),
            0,
        )
    labels = [name for name, _, _ in DISPLAY_BINS]
    labels.extend(
        sorted(
            {
                label
                for row in rows
                if (label := _carriage_bin(row)) is not None
                and label not in labels
            }
        )
    )
    position = {label: index for index, label in enumerate(labels)}
    events: dict[tuple[int, int, int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            int(row["graph_id"]),
            int(row["source"]),
            int(row["donor"]),
        )
        events.setdefault(key, []).append(row)
    event_profile: dict[tuple[int, int, int, int], np.ndarray] = {}
    event_total: dict[tuple[int, int, int, int], float] = {}
    for key, event_rows in events.items():
        values = np.asarray([float(row["F_sens"]) for row in event_rows])
        total = float(np.sum(values))
        if not np.isfinite(values).all() or total <= float(effect_floor):
            continue
        profile = np.zeros(len(labels), dtype=np.float64)
        for row, value in zip(event_rows, values):
            label = _carriage_bin(row)
            if label is not None:
                profile[position[label]] += float(value / total)
        event_profile[key] = profile
        event_total[key] = total
    if not event_profile:
        return tuple(labels), np.full(len(labels), np.nan), float("nan"), 0
    by_source: dict[tuple[int, int, int], list[tuple[np.ndarray, float]]] = {}
    for key, profile in event_profile.items():
        seed, graph, source, _ = key
        by_source.setdefault((seed, graph, source), []).append(
            (profile, event_total[key])
        )
    source_estimates = {
        key: (
            np.mean(np.stack([value[0] for value in rows]), axis=0),
            float(np.mean([value[1] for value in rows])),
        )
        for key, rows in by_source.items()
    }
    by_graph: dict[tuple[int, int], list[tuple[np.ndarray, float]]] = {}
    for (seed, graph, _), value in source_estimates.items():
        by_graph.setdefault((seed, graph), []).append(value)
    graph_estimates = {
        key: (
            np.mean(np.stack([value[0] for value in rows]), axis=0),
            float(np.mean([value[1] for value in rows])),
        )
        for key, rows in by_graph.items()
    }
    by_seed: dict[int, list[tuple[np.ndarray, float]]] = {}
    for (seed, _), value in graph_estimates.items():
        by_seed.setdefault(seed, []).append(value)
    seed_profiles = [
        trimmed_mean(np.stack([value[0] for value in rows]), 0.20, axis=0)
        for rows in by_seed.values()
    ]
    seed_totals = [
        float(trimmed_mean([value[1] for value in rows], 0.20, axis=0))
        for rows in by_seed.values()
    ]
    profile = np.mean(np.stack(seed_profiles), axis=0)
    total = float(np.mean(seed_totals))
    return tuple(labels), _normalise_last_axis(profile), total, len(event_profile)


def summarise_model(model: CachedModel) -> dict[str, Any]:
    metrics = CanonicalHeadMetrics.from_scores(model.score)
    metadata = model.score_artifact.metadata
    contract = metadata.get("contract", {})
    record: dict[str, Any] = {
        "task": model.task,
        "artifact_task": model.artifact_task,
        "cache_protocol": str(metadata.get("protocol_version", "unknown")),
        "raw_score_aggregation": str(
            contract.get("raw_score_aggregation", "unknown")
        ),
        "semantic_donor_law": str(contract.get("semantic_donor_law", "unknown")),
        "structural_donor_law": str(
            contract.get("structural_donor_law", "unknown")
        ),
        "label": TASK_LABELS.get(model.task, model.task),
        "train_seed": int(model.model_record["train_seed"]),
        "test_mae": float(model.model_record["test_metric"]),
        "validation_mae": float(model.model_record["validation_metric"]),
        "parameters": int(model.model_record["parameter_count"]),
        "layers": int(metrics.num_layers),
        "heads": int(metrics.num_heads),
        "score_cache": str(model.score_artifact.path),
        "score_sha256": model.score_artifact.file_sha256,
        "score_contract": str(model.score_artifact.metadata["contract_fingerprint"]),
        "carriage_cache": (
            None if model.carriage_artifact is None else str(model.carriage_artifact.path)
        ),
        "raw_semantic_mean": float(np.nanmean(metrics.raw_semantic)),
        "raw_structural_mean": float(np.nanmean(metrics.raw_structural)),
        "semantic_profile_ci_relative_width": relative_profile_interval_width(
            model.score, "semantic"
        ),
        "structural_profile_ci_relative_width": relative_profile_interval_width(
            model.score, "structural"
        ),
    }
    head_rows = []
    for layer in range(metrics.num_layers):
        for head in range(metrics.num_heads):
            head_rows.append(
                {
                    "task": model.task,
                    "layer": layer,
                    "head": head,
                    "raw_semantic": float(metrics.raw_semantic[layer, head]),
                    "raw_structural": float(metrics.raw_structural[layer, head]),
                    "normalized_semantic": float(
                        metrics.normalized_semantic[layer, head]
                    ),
                    "normalized_structural": float(
                        metrics.normalized_structural[layer, head]
                    ),
                    "joint_sensitivity": float(
                        metrics.joint_sensitivity[layer, head]
                    ),
                    "selectivity": float(metrics.selectivity[layer, head]),
                    "active": bool(metrics.active[layer, head]),
                }
            )
    record["head_rows"] = head_rows
    head_distance_rows = []
    for channel in CHANNELS:
        labels, equal_head, activity_weighted = score_profile(model.score, channel)
        record[f"{channel}_score_distance_labels"] = labels
        record[f"{channel}_score_distance_equal_head"] = equal_head
        record[f"{channel}_score_distance_activity_weighted"] = activity_weighted
        exact_head = _as_numpy(model.score["channels"][channel]["heatmap_exact_head"])
        _, grouped_head = group_distance(exact_head, tuple(model.score["axis"]))
        normalized_head = _normalise_last_axis(grouped_head)
        for layer in range(metrics.num_layers):
            for head in range(metrics.num_heads):
                for index, distance in enumerate(labels):
                    head_distance_rows.append(
                        {
                            "task": model.task,
                            "channel": channel,
                            "layer": layer,
                            "head": head,
                            "distance": distance,
                            "raw_score_mass": float(grouped_head[layer, head, index]),
                            "within_head_score_fraction": float(
                                normalized_head[layer, head, index]
                            ),
                        }
                    )
        interval_labels, interval_width, interval_reportable = (
            score_interval_width_profile(model.score, channel)
        )
        record[f"{channel}_score_ci_labels"] = interval_labels
        record[f"{channel}_score_ci_relative_width"] = interval_width
        record[f"{channel}_score_ci_reportable"] = interval_reportable
        record[f"{channel}_score_ci_reportable_bins"] = int(
            np.sum(interval_reportable)
        )
        record[f"{channel}_score_ci_total_bins"] = len(interval_reportable)
        if model.carriage is not None:
            carriage_labels, carriage_values, carriage_total, events = carriage_profile(
                model.carriage, channel
            )
            record[f"{channel}_carriage_labels"] = carriage_labels
            record[f"{channel}_carriage"] = carriage_values
            record[f"{channel}_carriage_total"] = carriage_total
            record[f"{channel}_carriage_events"] = events
    record["score_semantic_structural_profile_correlation"] = _finite_correlation(
        record["semantic_score_distance_equal_head"],
        record["structural_score_distance_equal_head"],
    )
    if model.carriage is not None:
        record["carriage_semantic_structural_profile_correlation"] = (
            _finite_correlation(record["semantic_carriage"], record["structural_carriage"])
        )
    record["head_distance_rows"] = head_distance_rows
    return record


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # Optional carriage fields make model-summary rows heterogeneous. Preserve
    # first-seen order while including every field before constructing the
    # strict DictWriter.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_value(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _distance_reportable_mask(
    score: Mapping[str, Any], channel: str, width: int
) -> np.ndarray:
    support = score["channels"][channel].get("distance_support")
    if support is None or "reportable" not in support:
        return np.ones(width, dtype=bool)
    reportable = _as_numpy(support["reportable"], dtype=bool).reshape(-1)
    if len(reportable) != width:
        raise ValueError(
            f"{channel} reportable mask has width {len(reportable)}, expected {width}"
        )
    return reportable


def _profile_pair_components(
    semantic: Any,
    structural: Any,
    axis: Sequence[Any],
    reportable: Any,
) -> dict[str, np.ndarray] | None:
    semantic = _as_numpy(semantic).reshape(-1)
    structural = _as_numpy(structural).reshape(-1)
    reportable = _as_numpy(reportable, dtype=bool).reshape(-1)
    if not (len(semantic) == len(structural) == len(axis) == len(reportable)):
        raise ValueError("semantic, structural, axis, and reportable profiles must align")
    valid = reportable & np.isfinite(semantic) & np.isfinite(structural)
    if not np.any(valid):
        return None
    semantic = semantic[valid]
    structural = structural[valid]
    labels = np.asarray(tuple(axis), dtype=object)[valid]
    if np.nanmin(semantic) < -1.0e-10 or np.nanmin(structural) < -1.0e-10:
        raise ValueError("canonical score-distance profiles must be non-negative")
    semantic = np.maximum(semantic, 0.0)
    structural = np.maximum(structural, 0.0)
    semantic_mass = float(np.sum(semantic))
    structural_mass = float(np.sum(structural))
    if semantic_mass <= 1.0e-12 or structural_mass <= 1.0e-12:
        return None
    return {
        "semantic": semantic,
        "structural": structural,
        "semantic_profile": semantic / semantic_mass,
        "structural_profile": structural / structural_mass,
        "labels": labels,
    }


def _profile_pair_metrics(
    semantic: Any,
    structural: Any,
    axis: Sequence[Any],
    reportable: Any,
) -> dict[str, Any] | None:
    components = _profile_pair_components(semantic, structural, axis, reportable)
    if components is None:
        return None
    semantic = components["semantic"]
    structural = components["structural"]
    semantic_profile = components["semantic_profile"]
    structural_profile = components["structural_profile"]
    labels = components["labels"]
    semantic_mass = float(np.sum(semantic))
    structural_mass = float(np.sum(structural))
    denominator = float(
        np.linalg.norm(semantic_profile) * np.linalg.norm(structural_profile)
    )
    cosine = (
        float(np.dot(semantic_profile, structural_profile) / denominator)
        if denominator > 1.0e-12
        else float("nan")
    )
    overlap = float(np.minimum(semantic_profile, structural_profile).sum())
    semantic_peak = int(np.argmax(semantic_profile))
    structural_peak = int(np.argmax(structural_profile))
    numeric = np.asarray(
        [
            float(distance) if (distance := _numeric_distance(label)) is not None else np.nan
            for label in labels
        ],
        dtype=np.float64,
    )
    numeric_mask = np.isfinite(numeric)
    semantic_centroid = float("nan")
    structural_centroid = float("nan")
    if np.any(numeric_mask):
        semantic_numeric_mass = float(semantic[numeric_mask].sum())
        structural_numeric_mass = float(structural[numeric_mask].sum())
        if semantic_numeric_mass > 1.0e-12 and structural_numeric_mass > 1.0e-12:
            semantic_centroid = float(
                np.dot(semantic[numeric_mask], numeric[numeric_mask])
                / semantic_numeric_mass
            )
            structural_centroid = float(
                np.dot(structural[numeric_mask], numeric[numeric_mask])
                / structural_numeric_mass
            )
    centroid_difference = structural_centroid - semantic_centroid
    return {
        "cosine": cosine,
        "overlap": overlap,
        "total_variation": float(1.0 - overlap),
        "peak_match": float(semantic_peak == structural_peak),
        "semantic_peak": str(labels[semantic_peak]).replace("_", " "),
        "structural_peak": str(labels[structural_peak]).replace("_", " "),
        "semantic_centroid": semantic_centroid,
        "structural_centroid": structural_centroid,
        "centroid_difference": centroid_difference,
        "centroid_gap_abs": abs(centroid_difference),
        "semantic_mass": semantic_mass,
        "structural_mass": structural_mass,
        "shared_reportable_bins": len(labels),
    }


def _distance_group_label(value: Any) -> str:
    distance = _numeric_distance(value)
    if distance is None:
        return str(value).replace("_", " ")
    for name, lower, upper in DISPLAY_BINS:
        if lower <= distance <= upper:
            return name
    raise ValueError(f"non-negative distance {distance} has no display bin")


def head_profile_alignment_rows(models: Sequence[CachedModel]) -> list[dict[str, Any]]:
    """Measure semantic--structural distance-profile overlap within learned heads.

    ``score_mass`` asks where total score mass is allocated. ``per_opportunity``
    repeats the comparison after controlling for the number of available
    source--carrier pairs in each distance shell.
    """

    profile_fields = {
        "score_mass": "heatmap_exact_head",
        "per_opportunity": "heatmap_per_opportunity_head",
    }
    rows: list[dict[str, Any]] = []
    for model in models:
        score = model.score
        axis = tuple(score["axis"])
        semantic_channel = score["channels"]["semantic"]
        structural_channel = score["channels"]["structural"]
        joint_reportable = _distance_reportable_mask(
            score, "semantic", len(axis)
        ) & _distance_reportable_mask(score, "structural", len(axis))
        for profile_kind, field in profile_fields.items():
            if field not in semantic_channel or field not in structural_channel:
                continue
            semantic = _as_numpy(semantic_channel[field])
            structural = _as_numpy(structural_channel[field])
            if semantic.ndim != 3 or semantic.shape != structural.shape:
                raise ValueError(
                    f"{model.task} {profile_kind} head profiles do not align: "
                    f"semantic={semantic.shape}, structural={structural.shape}"
                )
            if semantic.shape[-1] != len(axis):
                raise ValueError(
                    f"{model.task} {profile_kind} profile width "
                    f"{semantic.shape[-1]} does not match axis width {len(axis)}"
                )
            layers, heads, _ = semantic.shape
            for layer in range(layers):
                for head in range(heads):
                    metrics = _profile_pair_metrics(
                        semantic[layer, head],
                        structural[layer, head],
                        axis,
                        joint_reportable,
                    )
                    if metrics is None:
                        continue
                    rows.append(
                        {
                            "task": model.task,
                            "cache_protocol": str(
                                model.score_artifact.metadata.get(
                                    "protocol_version", "unknown"
                                )
                            ),
                            "profile_kind": profile_kind,
                            "layer": layer,
                            "head": head,
                            **metrics,
                        }
                    )
    return rows


def head_profile_distance_decomposition_rows(
    models: Sequence[CachedModel],
) -> list[dict[str, Any]]:
    """Decompose each head's profile mismatch into signed exact-distance terms.

    ``structural_minus_semantic`` locates the direction of the mismatch, while
    ``tv_contribution`` is non-negative and sums exactly to ``1 - overlap`` for
    each head.  Display-bin summaries sum exact-bin contributions before any
    aggregation across heads.
    """

    profile_fields = {
        "score_mass": "heatmap_exact_head",
        "per_opportunity": "heatmap_per_opportunity_head",
    }
    rows: list[dict[str, Any]] = []
    for model in models:
        score = model.score
        axis = tuple(score["axis"])
        semantic_channel = score["channels"]["semantic"]
        structural_channel = score["channels"]["structural"]
        joint_reportable = _distance_reportable_mask(
            score, "semantic", len(axis)
        ) & _distance_reportable_mask(score, "structural", len(axis))
        for profile_kind, field in profile_fields.items():
            if field not in semantic_channel or field not in structural_channel:
                continue
            semantic = _as_numpy(semantic_channel[field])
            structural = _as_numpy(structural_channel[field])
            if semantic.ndim != 3 or semantic.shape != structural.shape:
                raise ValueError(
                    f"{model.task} {profile_kind} head profiles do not align: "
                    f"semantic={semantic.shape}, structural={structural.shape}"
                )
            layers, heads, _ = semantic.shape
            for layer in range(layers):
                for head in range(heads):
                    components = _profile_pair_components(
                        semantic[layer, head],
                        structural[layer, head],
                        axis,
                        joint_reportable,
                    )
                    if components is None:
                        continue
                    for index, label in enumerate(components["labels"]):
                        semantic_share = float(components["semantic_profile"][index])
                        structural_share = float(
                            components["structural_profile"][index]
                        )
                        difference = structural_share - semantic_share
                        rows.append(
                            {
                                "task": model.task,
                                "profile_kind": profile_kind,
                                "layer": layer,
                                "head": head,
                                "distance": str(label).replace("_", " "),
                                "distance_order": (
                                    _numeric_distance(label)
                                    if _numeric_distance(label) is not None
                                    else 1_000_000
                                ),
                                "distance_group": _distance_group_label(label),
                                "semantic_share": semantic_share,
                                "structural_share": structural_share,
                                "structural_minus_semantic": difference,
                                "tv_contribution": 0.5 * abs(difference),
                                "semantic_raw_mass": float(
                                    components["semantic"][index]
                                ),
                                "structural_raw_mass": float(
                                    components["structural"][index]
                                ),
                            }
                        )
    return rows


def summarise_layerwise_distance_decomposition(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Sum exact terms into display bins, then summarize heads within layers."""

    head_groups: dict[
        tuple[str, str, int, int, str], dict[str, float]
    ] = {}
    for row in rows:
        key = (
            str(row["task"]),
            str(row["profile_kind"]),
            int(row["layer"]),
            int(row["head"]),
            str(row["distance_group"]),
        )
        target = head_groups.setdefault(
            key,
            {
                "semantic_share": 0.0,
                "structural_share": 0.0,
                "structural_minus_semantic": 0.0,
                "tv_contribution": 0.0,
            },
        )
        for field in target:
            target[field] += float(row[field])
    layer_groups: dict[
        tuple[str, str, int, str], list[dict[str, float]]
    ] = {}
    for (task, profile_kind, layer, _head, distance_group), values in head_groups.items():
        layer_groups.setdefault(
            (task, profile_kind, layer, distance_group), []
        ).append(values)
    output: list[dict[str, Any]] = []
    metrics = (
        "semantic_share",
        "structural_share",
        "structural_minus_semantic",
        "tv_contribution",
    )
    for (task, profile_kind, layer, distance_group), group in sorted(
        layer_groups.items()
    ):
        record: dict[str, Any] = {
            "task": task,
            "profile_kind": profile_kind,
            "layer": layer,
            "distance_group": distance_group,
            "valid_heads": len(group),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_q1"] = float(np.quantile(values, 0.25))
            record[f"{metric}_q3"] = float(np.quantile(values, 0.75))
        output.append(record)
    return output


def vnode_profile_rows(models: Sequence[CachedModel]) -> list[dict[str, Any]]:
    """Isolate virtual-carrier score allocation and molecular-only overlap."""

    profile_fields = {
        "score_mass": "heatmap_exact_head",
        "per_opportunity": "heatmap_per_opportunity_head",
    }
    rows: list[dict[str, Any]] = []
    for model in models:
        score = model.score
        axis = tuple(score["axis"])
        virtual_positions = [
            index
            for index, label in enumerate(axis)
            if str(label).replace("_", " ").lower() == "virtual"
        ]
        if not virtual_positions:
            continue
        if len(virtual_positions) != 1:
            raise ValueError(f"{model.task} has multiple virtual distance bins")
        semantic_channel = score["channels"]["semantic"]
        structural_channel = score["channels"]["structural"]
        joint_reportable = _distance_reportable_mask(
            score, "semantic", len(axis)
        ) & _distance_reportable_mask(score, "structural", len(axis))
        for profile_kind, field in profile_fields.items():
            if field not in semantic_channel or field not in structural_channel:
                continue
            semantic = _as_numpy(semantic_channel[field])
            structural = _as_numpy(structural_channel[field])
            layers, heads, _ = semantic.shape
            for layer in range(layers):
                for head in range(heads):
                    components = _profile_pair_components(
                        semantic[layer, head],
                        structural[layer, head],
                        axis,
                        joint_reportable,
                    )
                    if components is None:
                        continue
                    labels = [str(label).replace("_", " ") for label in components["labels"]]
                    if "virtual" not in [label.lower() for label in labels]:
                        continue
                    virtual = next(
                        index for index, label in enumerate(labels) if label.lower() == "virtual"
                    )
                    semantic_profile = components["semantic_profile"]
                    structural_profile = components["structural_profile"]
                    full_overlap = float(
                        np.minimum(semantic_profile, structural_profile).sum()
                    )
                    molecular = np.arange(len(labels)) != virtual
                    semantic_molecular_mass = float(
                        components["semantic"][molecular].sum()
                    )
                    structural_molecular_mass = float(
                        components["structural"][molecular].sum()
                    )
                    molecular_overlap = float("nan")
                    if (
                        semantic_molecular_mass > 1.0e-12
                        and structural_molecular_mass > 1.0e-12
                    ):
                        semantic_molecular = (
                            components["semantic"][molecular]
                            / semantic_molecular_mass
                        )
                        structural_molecular = (
                            components["structural"][molecular]
                            / structural_molecular_mass
                        )
                        molecular_overlap = float(
                            np.minimum(semantic_molecular, structural_molecular).sum()
                        )
                    semantic_virtual_share = float(semantic_profile[virtual])
                    structural_virtual_share = float(structural_profile[virtual])
                    virtual_difference = structural_virtual_share - semantic_virtual_share
                    rows.append(
                        {
                            "task": model.task,
                            "profile_kind": profile_kind,
                            "layer": layer,
                            "head": head,
                            "full_overlap": full_overlap,
                            "molecular_only_overlap": molecular_overlap,
                            "molecular_minus_full_overlap": molecular_overlap
                            - full_overlap,
                            "semantic_virtual_share": semantic_virtual_share,
                            "structural_virtual_share": structural_virtual_share,
                            "structural_minus_semantic_virtual_share": virtual_difference,
                            "virtual_tv_contribution": 0.5 * abs(virtual_difference),
                            "semantic_virtual_raw_mass": float(
                                components["semantic"][virtual]
                            ),
                            "structural_virtual_raw_mass": float(
                                components["structural"][virtual]
                            ),
                            "semantic_total_raw_mass": float(
                                components["semantic"].sum()
                            ),
                            "structural_total_raw_mass": float(
                                components["structural"].sum()
                            ),
                        }
                    )
    return rows


def summarise_layerwise_vnode_profiles(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["profile_kind"]), int(row["layer"]))
        groups.setdefault(key, []).append(row)
    metrics = (
        "full_overlap",
        "molecular_only_overlap",
        "molecular_minus_full_overlap",
        "semantic_virtual_share",
        "structural_virtual_share",
        "structural_minus_semantic_virtual_share",
        "virtual_tv_contribution",
        "semantic_virtual_raw_mass",
        "structural_virtual_raw_mass",
    )
    output: list[dict[str, Any]] = []
    for (task, profile_kind, layer), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "task": task,
            "profile_kind": profile_kind,
            "layer": layer,
            "valid_heads": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            finite = values[np.isfinite(values)]
            if not len(finite):
                for suffix in ("mean", "median", "q1", "q3"):
                    record[f"{metric}_{suffix}"] = float("nan")
                continue
            record[f"{metric}_mean"] = float(np.mean(finite))
            record[f"{metric}_median"] = float(np.median(finite))
            record[f"{metric}_q1"] = float(np.quantile(finite, 0.25))
            record[f"{metric}_q3"] = float(np.quantile(finite, 0.75))
        output.append(record)
    return output


def summarise_head_profile_alignment(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    metrics = (
        "cosine",
        "overlap",
        "total_variation",
        "peak_match",
        "semantic_centroid",
        "structural_centroid",
        "centroid_difference",
        "centroid_gap_abs",
    )
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["profile_kind"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (task, profile_kind), group in sorted(groups.items()):
        result: dict[str, Any] = {
            "task": task,
            "profile_kind": profile_kind,
            "valid_heads": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            result[f"{metric}_mean"] = (
                float(np.mean(values[np.isfinite(values)]))
                if np.isfinite(values).any()
                else float("nan")
            )
        output.append(result)
    return output


def summarise_layerwise_head_profile_alignment(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarise matched-head overlap and flag unusually low-overlap heads.

    A low-overlap outlier lies below the conventional Tukey lower fence within
    its own model, profile definition, and layer.  This is a descriptive
    checkpoint diagnostic, not a population-level significance test.
    """

    groups: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["profile_kind"]), int(row["layer"]))
        groups.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    outliers: list[dict[str, Any]] = []
    for (task, profile_kind, layer), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row["head"]))
        overlap = np.asarray([float(row["overlap"]) for row in ordered])
        cosine = np.asarray([float(row["cosine"]) for row in ordered])
        centroid_difference = np.asarray(
            [float(row["centroid_difference"]) for row in ordered]
        )
        finite_overlap = overlap[np.isfinite(overlap)]
        finite_cosine = cosine[np.isfinite(cosine)]
        finite_centroid_difference = centroid_difference[
            np.isfinite(centroid_difference)
        ]
        if not len(finite_overlap):
            continue
        overlap_q1, overlap_median, overlap_q3 = np.quantile(
            finite_overlap, (0.25, 0.5, 0.75)
        )
        overlap_iqr = float(overlap_q3 - overlap_q1)
        low_fence = float(overlap_q1 - 1.5 * overlap_iqr)
        minimum_index = int(np.nanargmin(overlap))
        maximum_index = int(np.nanargmax(overlap))
        flagged_heads = [
            int(row["head"])
            for row in ordered
            if np.isfinite(float(row["overlap"]))
            and float(row["overlap"]) < low_fence
        ]
        summaries.append(
            {
                "task": task,
                "profile_kind": profile_kind,
                "layer": layer,
                "valid_heads": len(finite_overlap),
                "overlap_mean": float(np.mean(finite_overlap)),
                "overlap_median": float(overlap_median),
                "overlap_std": (
                    float(np.std(finite_overlap, ddof=1))
                    if len(finite_overlap) > 1
                    else 0.0
                ),
                "overlap_q1": float(overlap_q1),
                "overlap_q3": float(overlap_q3),
                "overlap_iqr": overlap_iqr,
                "overlap_min": float(finite_overlap.min()),
                "overlap_min_head": int(ordered[minimum_index]["head"]),
                "overlap_max": float(finite_overlap.max()),
                "overlap_max_head": int(ordered[maximum_index]["head"]),
                "overlap_low_outlier_fence": low_fence,
                "low_overlap_outlier_count": len(flagged_heads),
                "low_overlap_outlier_heads": ",".join(map(str, flagged_heads)),
                "fraction_overlap_at_least_0_8": float(
                    np.mean(finite_overlap >= 0.8)
                ),
                "fraction_overlap_at_least_0_9": float(
                    np.mean(finite_overlap >= 0.9)
                ),
                "cosine_mean": (
                    float(np.mean(finite_cosine))
                    if len(finite_cosine)
                    else float("nan")
                ),
                "cosine_median": (
                    float(np.median(finite_cosine))
                    if len(finite_cosine)
                    else float("nan")
                ),
                "peak_match_fraction": float(
                    np.mean([float(row["peak_match"]) for row in ordered])
                ),
                "centroid_difference_mean": (
                    float(np.mean(finite_centroid_difference))
                    if len(finite_centroid_difference)
                    else float("nan")
                ),
                "centroid_difference_median": (
                    float(np.median(finite_centroid_difference))
                    if len(finite_centroid_difference)
                    else float("nan")
                ),
                "centroid_difference_iqr": (
                    float(
                        np.quantile(finite_centroid_difference, 0.75)
                        - np.quantile(finite_centroid_difference, 0.25)
                    )
                    if len(finite_centroid_difference)
                    else float("nan")
                ),
                "fraction_structural_centroid_farther": (
                    float(np.mean(finite_centroid_difference > 0.0))
                    if len(finite_centroid_difference)
                    else float("nan")
                ),
            }
        )
        for row in ordered:
            value = float(row["overlap"])
            if np.isfinite(value) and value < low_fence:
                outliers.append(
                    {
                        "task": task,
                        "profile_kind": profile_kind,
                        "layer": layer,
                        "head": int(row["head"]),
                        "overlap": value,
                        "layer_median_overlap": float(overlap_median),
                        "difference_from_layer_median": value
                        - float(overlap_median),
                        "low_outlier_fence": low_fence,
                        "cosine": float(row["cosine"]),
                        "semantic_peak": row["semantic_peak"],
                        "structural_peak": row["structural_peak"],
                        "semantic_centroid": float(row["semantic_centroid"]),
                        "structural_centroid": float(row["structural_centroid"]),
                    }
                )
    return summaries, outliers


def _head_profile_alignment_contract(
    models: Sequence[CachedModel],
) -> dict[str, Any]:
    return {
        "cache_version": ALIGNMENT_CACHE_VERSION,
        "profile_fields": {
            "score_mass": "heatmap_exact_head",
            "per_opportunity": "heatmap_per_opportunity_head",
        },
        "pairing": "semantic and structural profiles from the same learned head",
        "layer_summary": (
            "median, IQR, range, threshold fractions, and Tukey low-overlap outliers"
        ),
        "distance_decomposition": (
            "exact structural-minus-semantic share and additive total-variation terms"
        ),
        "virtual_node": (
            "virtual score share and overlap after molecular-only renormalization"
        ),
        "models": [
            {
                "task": model.task,
                "score_sha256": model.score_artifact.file_sha256,
                "score_contract": model.score_artifact.metadata.get(
                    "contract_fingerprint"
                ),
                "protocol": model.score_artifact.metadata.get("protocol_version"),
            }
            for model in models
        ],
    }


def load_or_compute_head_profile_alignment(
    models: Sequence[CachedModel],
    output_dir: Path,
) -> tuple[dict[str, Any], Path, str]:
    """Load a fingerprinted derived cache or compute it from immutable scores."""

    contract = _head_profile_alignment_contract(models)
    fingerprint = stable_hash(contract)
    path = Path(output_dir) / "cache" / "head_profile_alignment.json"
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload["metadata"]
            value = payload["value"]
            if (
                metadata.get("cache_version") == ALIGNMENT_CACHE_VERSION
                and metadata.get("fingerprint") == fingerprint
                and isinstance(value.get("comparison_rows"), list)
                and isinstance(value.get("summary_rows"), list)
                and isinstance(value.get("layer_summary_rows"), list)
                and isinstance(value.get("outlier_rows"), list)
                and isinstance(value.get("distance_rows"), list)
                and isinstance(value.get("distance_layer_rows"), list)
                and isinstance(value.get("vnode_rows"), list)
                and isinstance(value.get("vnode_layer_rows"), list)
            ):
                return value, path, "hit"
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
    comparison_rows = head_profile_alignment_rows(models)
    layer_summary_rows, outlier_rows = summarise_layerwise_head_profile_alignment(
        comparison_rows
    )
    distance_rows = head_profile_distance_decomposition_rows(models)
    distance_layer_rows = summarise_layerwise_distance_decomposition(distance_rows)
    virtual_rows = vnode_profile_rows(models)
    virtual_layer_rows = summarise_layerwise_vnode_profiles(virtual_rows)
    value = {
        "comparison_rows": comparison_rows,
        "summary_rows": summarise_head_profile_alignment(comparison_rows),
        "layer_summary_rows": layer_summary_rows,
        "outlier_rows": outlier_rows,
        "distance_rows": distance_rows,
        "distance_layer_rows": distance_layer_rows,
        "vnode_rows": virtual_rows,
        "vnode_layer_rows": virtual_layer_rows,
    }
    _write_json(
        path,
        {
            "metadata": {
                "cache_version": ALIGNMENT_CACHE_VERSION,
                "fingerprint": fingerprint,
                "contract": contract,
            },
            "value": value,
        },
    )
    return value, path, "miss"


def _ordered_union(records: Sequence[Mapping[str, Any]], field: str) -> tuple[str, ...]:
    observed = {
        str(label)
        for record in records
        for label in record.get(field, ())
    }
    ordered = [name for name, _, _ in DISPLAY_BINS if name in observed]
    numeric = sorted(
        (
            label
            for label in observed
            if label not in ordered and _numeric_distance(label) is not None
        ),
        key=lambda label: int(_numeric_distance(label)),
    )
    specials = sorted(observed.difference(ordered).difference(numeric))
    return *ordered, *numeric, *specials


def _align_values(
    labels: Sequence[Any],
    values: Any,
    target: Sequence[str],
    *,
    fill: float,
) -> np.ndarray:
    source = {str(label): float(value) for label, value in zip(labels, _as_numpy(values))}
    return np.asarray([source.get(str(label), fill) for label in target], dtype=np.float64)


def _record_label(
    record: Mapping[str, Any], *, multiline: bool, show_protocol: bool
) -> str:
    task = str(record["task"])
    label = TASK_LABELS.get(task, task)
    if not multiline:
        label = label.replace("\n", " ")
    if not show_protocol:
        return label
    protocol = str(record.get("cache_protocol", "unknown")).rsplit("-", 1)[-1]
    separator = "\n" if multiline else " "
    return f"{label}{separator}[{protocol}]"


def _profile_total_variation(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    label_field: str,
    value_field: str,
) -> float:
    labels = _ordered_union((left, right), label_field)
    if not labels or value_field not in left or value_field not in right:
        return float("nan")
    left_values = _align_values(left[label_field], left[value_field], labels, fill=0.0)
    right_values = _align_values(right[label_field], right[value_field], labels, fill=0.0)
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        return float("nan")
    left_values = _normalise_last_axis(left_values)
    right_values = _normalise_last_axis(right_values)
    return float(0.5 * np.sum(np.abs(left_values - right_values)))


def pairwise_comparisons(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Quantify whether two fitted models differ beyond performance alone."""

    from scipy.stats import wasserstein_distance

    rows = []
    for left, right in combinations(records, 2):
        row: dict[str, Any] = {
            "left_task": left["task"],
            "right_task": right["task"],
            "left_cache_protocol": left["cache_protocol"],
            "right_cache_protocol": right["cache_protocol"],
            "same_structural_donor_law": (
                left["structural_donor_law"] == right["structural_donor_law"]
            ),
            "test_mae_right_minus_left": float(right["test_mae"] - left["test_mae"]),
            "parameter_ratio_right_over_left": float(
                right["parameters"] / left["parameters"]
            ),
        }
        for channel in CHANNELS:
            left_raw = np.asarray(
                [item[f"raw_{channel}"] for item in left["head_rows"]],
                dtype=np.float64,
            )
            right_raw = np.asarray(
                [item[f"raw_{channel}"] for item in right["head_rows"]],
                dtype=np.float64,
            )
            row[f"{channel}_raw_score_wasserstein"] = float(
                wasserstein_distance(left_raw, right_raw)
            )
            row[f"{channel}_score_profile_total_variation"] = (
                _profile_total_variation(
                    left,
                    right,
                    label_field=f"{channel}_score_distance_labels",
                    value_field=f"{channel}_score_distance_equal_head",
                )
            )
            left_width = float(left[f"{channel}_profile_ci_relative_width"])
            right_width = float(right[f"{channel}_profile_ci_relative_width"])
            row[f"{channel}_ci_width_ratio_right_over_left"] = (
                float(right_width / left_width)
                if np.isfinite(left_width) and left_width > 0
                else float("nan")
            )
            row[f"{channel}_carriage_profile_total_variation"] = (
                _profile_total_variation(
                    left,
                    right,
                    label_field=f"{channel}_carriage_labels",
                    value_field=f"{channel}_carriage",
                )
            )
        rows.append(row)
    return rows


def _plot_core(records: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    tasks = [str(record["task"]) for record in records]
    show_protocol = len({str(record["cache_protocol"]) for record in records}) > 1
    labels = [
        _record_label(record, multiline=True, show_protocol=show_protocol)
        for record in records
    ]
    colours = [TASK_COLOURS.get(task, "#777777") for task in tasks]
    x = np.arange(len(tasks))

    axis = axes[0, 0]
    mae = [float(record["test_mae"]) for record in records]
    axis.bar(x, mae, color=colours, edgecolor="#333333", linewidth=0.7)
    axis.set_xticks(x, labels)
    axis.set_ylabel("ZINC test MAE")
    axis.set_title("a  Does performance change across communication/PE scope?")
    for position, value, record in zip(x, mae, records):
        parameters = float(record["parameters"]) / 1.0e6
        axis.text(
            position,
            value,
            f"{value:.3f}\n{parameters:.2f}M params",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.margins(y=0.16)

    axis = axes[0, 1]
    offsets = (-0.18, 0.18)
    widths = 0.30
    for channel, offset, hatch in zip(CHANNELS, offsets, ("", "//")):
        data = [
            [float(row[f"raw_{channel}"]) for row in record["head_rows"]]
            for record in records
        ]
        boxes = axis.boxplot(
            data,
            positions=x + offset,
            widths=widths,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222"},
        )
        for patch, colour in zip(boxes["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.55 if channel == "semantic" else 0.28)
            patch.set_hatch(hatch)
        for line in (*boxes["whiskers"], *boxes["caps"]):
            line.set_color("#444444")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Raw canonical head score")
    axis.set_title("b  Do raw semantic/structural score distributions align?")
    axis.plot([], [], color="#555555", linewidth=7, alpha=0.55, label="semantic")
    axis.plot([], [], color="#555555", linewidth=7, alpha=0.28, label="structural")
    axis.legend(frameon=False)

    for column, channel in enumerate(CHANNELS):
        axis = axes[1, column]
        distance_labels = _ordered_union(
            records, f"{channel}_score_distance_labels"
        )
        for record in records:
            task = str(record["task"])
            values = _align_values(
                record[f"{channel}_score_distance_labels"],
                record[f"{channel}_score_distance_equal_head"],
                distance_labels,
                fill=0.0,
            )
            axis.plot(
                np.arange(len(values)),
                values,
                marker=TASK_MARKERS.get(task, "o"),
                linestyle=TASK_LINESTYLES.get(task, "-"),
                linewidth=1.8,
                markersize=5,
                color=TASK_COLOURS.get(task, "#777777"),
                label=_record_label(
                    record, multiline=False, show_protocol=show_protocol
                ),
            )
        axis.set_xticks(np.arange(len(distance_labels)), distance_labels)
        axis.set_xlabel("source-to-carrier graph distance")
        axis.set_ylabel("Mean per-head normalized score mass")
        axis.set_title(
            f"{'c' if channel == 'semantic' else 'd'}  {channel.title()} score distance profile"
        )
        if channel == "semantic":
            axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle(
        "ZINC cached canonical analysis: performance, raw scores, and per-head distance profiles",
        fontsize=15,
    )
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "01_performance_scores_distance.png"
    pdf = figures / "01_performance_scores_distance.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return [png, pdf]


def _plot_carriage_variability(
    records: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[Path] | None:
    import matplotlib.pyplot as plt

    available = [record for record in records if record.get("carriage_cache")]
    show_protocol = len({str(record["cache_protocol"]) for record in records}) > 1
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 9.2),
        sharey="row",
        constrained_layout=True,
    )
    for column, channel in enumerate(CHANNELS):
        axis = axes[0, column]
        labels = _ordered_union(records, f"{channel}_score_ci_labels")
        for record in records:
            task = str(record["task"])
            values = _align_values(
                record[f"{channel}_score_ci_labels"],
                record[f"{channel}_score_ci_relative_width"],
                labels,
                fill=float("nan"),
            )
            axis.plot(
                np.arange(len(values)),
                values,
                marker=TASK_MARKERS.get(task, "o"),
                linestyle=TASK_LINESTYLES.get(task, "-"),
                linewidth=1.8,
                markersize=5,
                color=TASK_COLOURS.get(task, "#777777"),
                label=_record_label(
                    record, multiline=False, show_protocol=show_protocol
                ),
            )
        axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_xlabel("exact source-to-carrier graph distance")
        axis.set_ylabel("Marginal 95% CI width / total score mass")
        axis.set_title(
            f"{'a' if channel == 'semantic' else 'b'}  {channel.title()} score uncertainty"
        )
        if channel == "semantic":
            axis.legend(frameon=False, fontsize=8, ncol=2)

    for axis, channel, panel in zip(axes[1], CHANNELS, ("c", "d")):
        labels = _ordered_union(available, f"{channel}_carriage_labels")
        for record in available:
            task = str(record["task"])
            values = _align_values(
                record[f"{channel}_carriage_labels"],
                record[f"{channel}_carriage"],
                labels,
                fill=0.0,
            )
            axis.plot(
                np.arange(len(values)),
                values,
                marker=TASK_MARKERS.get(task, "o"),
                linestyle=TASK_LINESTYLES.get(task, "-"),
                linewidth=1.8,
                markersize=5,
                color=TASK_COLOURS.get(task, "#777777"),
                label=_record_label(
                    record, multiline=False, show_protocol=show_protocol
                ),
            )
        if labels:
            axis.set_xticks(np.arange(len(labels)), labels)
        axis.set_xlabel("source-to-carrier graph distance")
        axis.set_ylabel("Event-normalized Functional carriage")
        axis.set_title(f"{panel}  {channel.title()} carriage geometry")
    if available:
        axes[1, 0].legend(frameon=False, fontsize=8, ncol=2)
    else:
        for axis in axes[1]:
            axis.text(
                0.5,
                0.5,
                "Canonical carriage cache unavailable",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    figure.suptitle(
        "ZINC cached canonical analysis: profile precision and Functional carriage",
        fontsize=15,
    )
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "02_profile_precision_and_carriage.png"
    pdf = figures / "02_profile_precision_and_carriage.pdf"
    figure.savefig(png, dpi=220)
    figure.savefig(pdf)
    plt.close(figure)
    return [png, pdf]


def _plot_head_profile_alignment(
    records: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
    outlier_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [str(record["task"]) for record in records]
    labels = [
        _record_label(record, multiline=True, show_protocol=False)
        for record in records
    ]
    layers = sorted({int(row["layer"]) for row in layer_rows})
    if not layers:
        return []
    layer_index = {layer: index for index, layer in enumerate(layers)}
    task_index = {task: index for index, task in enumerate(tasks)}
    outlier_counts = {
        (str(row["task"]), str(row["profile_kind"]), int(row["layer"])): 0
        for row in outlier_rows
    }
    for row in outlier_rows:
        key = (str(row["task"]), str(row["profile_kind"]), int(row["layer"]))
        outlier_counts[key] += 1

    figure, axes = plt.subplots(2, 2, figsize=(14.5, 7.8), constrained_layout=True)
    panels = (
        (axes[0, 0], "score_mass", "overlap_median", "a", "Score-mass median overlap"),
        (axes[0, 1], "score_mass", "overlap_iqr", "b", "Score-mass within-layer IQR"),
        (
            axes[1, 0],
            "per_opportunity",
            "overlap_median",
            "c",
            "Opportunity-corrected median overlap",
        ),
        (
            axes[1, 1],
            "per_opportunity",
            "overlap_iqr",
            "d",
            "Opportunity-corrected within-layer IQR",
        ),
    )
    finite_iqr = [
        float(row["overlap_iqr"])
        for row in layer_rows
        if np.isfinite(float(row["overlap_iqr"]))
    ]
    iqr_upper = max(max(finite_iqr, default=0.0), 0.05)
    for axis, profile_kind, field, panel, title in panels:
        matrix = np.full((len(tasks), len(layers)), np.nan, dtype=np.float64)
        for row in layer_rows:
            if str(row["profile_kind"]) != profile_kind:
                continue
            task = str(row["task"])
            if task not in task_index:
                continue
            matrix[task_index[task], layer_index[int(row["layer"])]] = float(row[field])
        is_median = field == "overlap_median"
        image = axis.imshow(
            np.ma.masked_invalid(matrix),
            aspect="auto",
            interpolation="nearest",
            cmap="viridis" if is_median else "magma_r",
            vmin=0.0,
            vmax=1.0 if is_median else iqr_upper,
        )
        for row_index, task in enumerate(tasks):
            for column_index, layer in enumerate(layers):
                value = matrix[row_index, column_index]
                if not np.isfinite(value):
                    continue
                normalized = value if is_median else value / iqr_upper
                text_colour = (
                    ("white" if normalized < 0.42 else "black")
                    if is_median
                    else ("black" if normalized < 0.58 else "white")
                )
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color=text_colour,
                )
                count = outlier_counts.get((task, profile_kind, layer), 0)
                if count:
                    axis.scatter(
                        column_index + 0.37,
                        row_index - 0.34,
                        marker="x",
                        s=28,
                        linewidth=1.5,
                        color="#D62728",
                        zorder=4,
                    )
        axis.set_xticks(np.arange(len(layers)), [str(layer) for layer in layers])
        axis.set_yticks(np.arange(len(tasks)), labels)
        axis.set_xlabel("layer")
        axis.set_title(f"{panel}  {title}")
        colourbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
        colourbar.set_label("median overlap" if is_median else "overlap IQR")

    figure.suptitle(
        "ZINC cached canonical analysis: layerwise semantic–structural profile overlap",
        fontsize=15,
    )
    figure.text(
        0.5,
        -0.01,
        "Overlap = 1 − total variation between each head's normalized semantic and "
        "structural distance profiles. Cells summarize heads; red × marks a layer "
        "containing a Tukey low-overlap outlier. These are one-checkpoint diagnostics.",
        ha="center",
        fontsize=8.5,
    )
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "03_head_profile_colocalization.png"
    pdf = figures / "03_head_profile_colocalization.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _plot_head_profile_distance_decomposition(
    records: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [
        str(record["task"])
        for record in records
        if any(str(row["task"]) == str(record["task"]) for row in layer_rows)
    ]
    if not tasks:
        return []
    observed_groups = {str(row["distance_group"]) for row in layer_rows}
    distance_groups = [
        name for name, _, _ in DISPLAY_BINS if name in observed_groups
    ] + sorted(
        observed_groups.difference({name for name, _, _ in DISPLAY_BINS})
    )
    layers = sorted({int(row["layer"]) for row in layer_rows})
    layer_index = {layer: index for index, layer in enumerate(layers)}
    distance_index = {
        distance_group: index
        for index, distance_group in enumerate(distance_groups)
    }
    values = np.asarray(
        [float(row["structural_minus_semantic_mean"]) for row in layer_rows],
        dtype=np.float64,
    )
    finite = np.abs(values[np.isfinite(values)])
    colour_limit = max(float(np.quantile(finite, 0.98)) if len(finite) else 0.0, 0.05)
    figure, axes = plt.subplots(
        2,
        len(tasks),
        figsize=(3.25 * len(tasks) + 1.2, 7.4),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row_index, (profile_kind, row_label) in enumerate(
        (("score_mass", "score mass"), ("per_opportunity", "per opportunity"))
    ):
        for column_index, task in enumerate(tasks):
            axis = axes[row_index, column_index]
            matrix = np.full(
                (len(layers), len(distance_groups)), np.nan, dtype=np.float64
            )
            for row in layer_rows:
                if (
                    str(row["task"]) != task
                    or str(row["profile_kind"]) != profile_kind
                ):
                    continue
                matrix[
                    layer_index[int(row["layer"])],
                    distance_index[str(row["distance_group"])],
                ] = float(row["structural_minus_semantic_mean"])
            image = axis.imshow(
                np.ma.masked_invalid(matrix),
                aspect="auto",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-colour_limit,
                vmax=colour_limit,
            )
            for layer_position in range(len(layers)):
                for distance_position in range(len(distance_groups)):
                    value = matrix[layer_position, distance_position]
                    if np.isfinite(value) and abs(value) >= 0.04:
                        axis.text(
                            distance_position,
                            layer_position,
                            f"{value:+.2f}",
                            ha="center",
                            va="center",
                            fontsize=6.3,
                            color="black",
                        )
            axis.set_xticks(
                np.arange(len(distance_groups)),
                distance_groups,
                rotation=45,
                ha="right",
            )
            axis.set_yticks(np.arange(len(layers)), [str(layer) for layer in layers])
            axis.set_xlabel("source-to-carrier distance")
            if column_index == 0:
                axis.set_ylabel(f"{row_label}\nlayer")
            else:
                axis.tick_params(axis="y", labelleft=False)
            if row_index == 0:
                axis.set_title(TASK_LABELS.get(task, task).replace("\n", " "))
    if image is not None:
        colourbar = figure.colorbar(
            image,
            ax=axes,
            fraction=0.018,
            pad=0.015,
            extend="both",
        )
        colourbar.set_label("mean structural share − semantic share")
    figure.suptitle(
        "ZINC cached canonical analysis: where do semantic and structural profiles diverge?",
        fontsize=15,
    )
    figure.text(
        0.5,
        -0.01,
        "Red indicates greater structural score allocation; blue indicates greater semantic "
        "allocation. Shares are normalized within each learned head before averaging. "
        "Additive absolute contributions to 1 − overlap are retained in the CSV.",
        ha="center",
        fontsize=8.5,
    )
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "04_head_profile_distance_decomposition.png"
    pdf = figures / "04_head_profile_distance_decomposition.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _plot_vnode_profile_diagnostics(
    records: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    tasks = [
        str(record["task"])
        for record in records
        if any(str(row["task"]) == str(record["task"]) for row in layer_rows)
    ]
    if not tasks:
        return []
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), constrained_layout=True)
    panels = (
        (axes[0, 0], "score_mass", "overlap", "a", "Score-mass overlap"),
        (
            axes[0, 1],
            "per_opportunity",
            "overlap",
            "b",
            "Opportunity-corrected overlap",
        ),
        (axes[1, 0], "score_mass", "share", "c", "Virtual-carrier score share"),
        (
            axes[1, 1],
            "per_opportunity",
            "share",
            "d",
            "Opportunity-corrected virtual share",
        ),
    )
    for axis, profile_kind, panel_kind, panel, title in panels:
        for task in tasks:
            selected = sorted(
                (
                    row
                    for row in layer_rows
                    if str(row["task"]) == task
                    and str(row["profile_kind"]) == profile_kind
                ),
                key=lambda row: int(row["layer"]),
            )
            if not selected:
                continue
            layers = np.asarray([int(row["layer"]) for row in selected])
            task_label = TASK_LABELS.get(task, task).replace("\n", " ")
            if panel_kind == "overlap":
                series = (
                    ("full_overlap", "full profile", "-", "o"),
                    ("molecular_only_overlap", "molecular only", "--", "s"),
                )
            else:
                series = (
                    ("semantic_virtual_share", "semantic", "-", "o"),
                    ("structural_virtual_share", "structural", "--", "s"),
                )
            colours = ("#0072B2", "#D55E00")
            for colour, (field, label, linestyle, marker) in zip(colours, series):
                median = np.asarray(
                    [float(row[f"{field}_median"]) for row in selected]
                )
                q1 = np.asarray([float(row[f"{field}_q1"]) for row in selected])
                q3 = np.asarray([float(row[f"{field}_q3"]) for row in selected])
                axis.plot(
                    layers,
                    median,
                    color=colour,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.8,
                    markersize=4,
                    label=f"{task_label}: {label}",
                )
                axis.fill_between(layers, q1, q3, color=colour, alpha=0.13)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("layer")
        axis.set_ylabel("profile overlap" if panel_kind == "overlap" else "score share")
        axis.set_title(f"{panel}  {title}")
        axis.legend(frameon=False, fontsize=7.5)
    figure.suptitle(
        "ZINC cached canonical analysis: what role does the virtual carrier play?",
        fontsize=15,
    )
    figure.text(
        0.5,
        -0.01,
        "Lines are across-head medians and shading is the within-layer IQR. Molecular-only "
        "overlap removes the virtual bin and renormalizes both profiles; virtual shares are "
        "reported alongside raw virtual score mass in the cached tables.",
        ha="center",
        fontsize=8.5,
    )
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    png = figures / "05_vnode_profile_diagnostics.png"
    pdf = figures / "05_vnode_profile_diagnostics.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def run(
    roots: Sequence[Path],
    output_dir: Path,
    *,
    tasks: Sequence[str] = TASKS,
    train_seed: int = 42,
    require_carriage: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    def log(stage: str, message: str) -> None:
        if verbose:
            print(f"[zinc-cache:{stage}] {message}", flush=True)

    log("load", f"tasks={list(tasks)} seed={int(train_seed)}")
    models = []
    for task in tasks:
        log("load", f"resolving {task}")
        try:
            loaded = load_cached_models(
                roots,
                tasks=(task,),
                train_seed=train_seed,
                require_carriage=require_carriage,
            )
        except Exception as error:
            raise RuntimeError(
                f"ZINC cache analysis failed while loading {task}: "
                f"{type(error).__name__}: {error}"
            ) from error
        models.extend(loaded)
        model = loaded[0]
        log(
            "load",
            f"selected {model.task}: "
            f"protocol={model.score_artifact.metadata.get('protocol_version')} "
            f"score={model.score_artifact.path} carriage={model.carriage is not None}",
        )
    records = []
    for model in models:
        log("summarise", model.task)
        try:
            records.append(summarise_model(model))
        except Exception as error:
            raise RuntimeError(
                f"ZINC cache analysis failed while summarising {model.task}: "
                f"{type(error).__name__}: {error}"
            ) from error
    protocols = sorted({str(record["cache_protocol"]) for record in records})
    semantic_laws = sorted({str(record["semantic_donor_law"]) for record in records})
    structural_laws = sorted(
        {str(record["structural_donor_law"]) for record in records}
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log("alignment-cache", "validating head-profile co-location cache")
    try:
        alignment, alignment_cache, alignment_cache_status = (
            load_or_compute_head_profile_alignment(models, output_dir)
        )
    except Exception as error:
        raise RuntimeError(
            "ZINC cache analysis failed during head-profile co-location analysis: "
            f"{type(error).__name__}: {error}"
        ) from error
    log(
        "alignment-cache",
        f"{alignment_cache_status}: {alignment_cache}",
    )
    alignment_rows = alignment["comparison_rows"]
    alignment_summary_rows = alignment["summary_rows"]
    alignment_layer_rows = alignment["layer_summary_rows"]
    alignment_outlier_rows = alignment["outlier_rows"]
    alignment_distance_rows = alignment["distance_rows"]
    alignment_distance_layer_rows = alignment["distance_layer_rows"]
    vnode_rows = alignment["vnode_rows"]
    vnode_layer_rows = alignment["vnode_layer_rows"]
    summary_rows = [
        {
            key: value
            for key, value in record.items()
            if key
            not in {
                "head_rows",
                "head_distance_rows",
                *(
                    f"{channel}_{suffix}"
                    for channel in CHANNELS
                    for suffix in (
                        "score_distance_labels",
                        "score_distance_equal_head",
                        "score_distance_activity_weighted",
                        "score_ci_labels",
                        "score_ci_relative_width",
                        "score_ci_reportable",
                        "carriage_labels",
                        "carriage",
                    )
                ),
            }
        }
        for record in records
    ]
    head_rows = [row for record in records for row in record["head_rows"]]
    head_distance_rows = [
        row for record in records for row in record["head_distance_rows"]
    ]
    comparison_rows = pairwise_comparisons(records)
    tables = {
        "model_summary.csv": summary_rows,
        "head_scores.csv": head_rows,
        "head_score_distance.csv": head_distance_rows,
        "pairwise_comparisons.csv": comparison_rows,
        "head_profile_alignment.csv": alignment_rows,
        "head_profile_alignment_summary.csv": alignment_summary_rows,
        "head_profile_alignment_layerwise.csv": alignment_layer_rows,
        "head_profile_alignment_outliers.csv": alignment_outlier_rows,
        "head_profile_distance_decomposition.csv": alignment_distance_rows,
        "head_profile_distance_decomposition_layerwise.csv": (
            alignment_distance_layer_rows
        ),
        "vnode_profile_alignment.csv": vnode_rows,
        "vnode_profile_alignment_layerwise.csv": vnode_layer_rows,
    }
    for name, rows in tables.items():
        log("table", f"{name}: {len(rows)} rows")
        try:
            _write_csv(output_dir / name, rows)
        except Exception as error:
            raise RuntimeError(
                f"ZINC cache analysis failed while writing {name}: "
                f"{type(error).__name__}: {error}"
            ) from error
    retired_permutation_table = output_dir / "head_profile_alignment_permutation.csv"
    if retired_permutation_table.is_file():
        retired_permutation_table.unlink()
        log("table", f"retired stale null table: {retired_permutation_table}")
    log("figure", "01_performance_scores_distance")
    try:
        figures = [*_plot_core(records, output_dir)]
    except Exception as error:
        raise RuntimeError(
            "ZINC cache analysis failed while plotting performance/scores/distance: "
            f"{type(error).__name__}: {error}"
        ) from error
    log("figure", "02_profile_precision_and_carriage")
    try:
        carriage_figure = _plot_carriage_variability(records, output_dir)
        if carriage_figure:
            figures.extend(carriage_figure)
    except Exception as error:
        raise RuntimeError(
            "ZINC cache analysis failed while plotting precision/carriage: "
            f"{type(error).__name__}: {error}"
        ) from error
    log("figure", "03_head_profile_colocalization")
    try:
        figures.extend(
            _plot_head_profile_alignment(
                records,
                alignment_layer_rows,
                alignment_outlier_rows,
                output_dir,
            )
        )
    except Exception as error:
        raise RuntimeError(
            "ZINC cache analysis failed while plotting head-profile co-location: "
            f"{type(error).__name__}: {error}"
        ) from error
    log("figure", "04_head_profile_distance_decomposition")
    try:
        figures.extend(
            _plot_head_profile_distance_decomposition(
                records,
                alignment_distance_layer_rows,
                output_dir,
            )
        )
    except Exception as error:
        raise RuntimeError(
            "ZINC cache analysis failed while plotting profile-distance decomposition: "
            f"{type(error).__name__}: {error}"
        ) from error
    log("figure", "05_vnode_profile_diagnostics")
    try:
        figures.extend(
            _plot_vnode_profile_diagnostics(
                records,
                vnode_layer_rows,
                output_dir,
            )
        )
    except Exception as error:
        raise RuntimeError(
            "ZINC cache analysis failed while plotting virtual-node diagnostics: "
            f"{type(error).__name__}: {error}"
        ) from error
    result = {
        "analysis_version": ANALYSIS_VERSION,
        "train_seed": int(train_seed),
        "roots": [str(Path(root)) for root in roots],
        "tasks": list(tasks),
        "models": records,
        "pairwise_comparisons": comparison_rows,
        "head_profile_alignment": {
            "cache": str(alignment_cache),
            "cache_status": alignment_cache_status,
            "summary": alignment_summary_rows,
            "layerwise": alignment_layer_rows,
            "outliers": alignment_outlier_rows,
            "distance_decomposition": alignment_distance_layer_rows,
            "vnode_layerwise": vnode_layer_rows,
        },
        "cache_compatibility": {
            "protocols": protocols,
            "semantic_donor_laws": semantic_laws,
            "structural_donor_laws": structural_laws,
            "same_semantic_donor_law": len(semantic_laws) == 1,
            "same_structural_donor_law": len(structural_laws) == 1,
            "interpretation": (
                "strictly comparable cached score estimands"
                if len(semantic_laws) == 1 and len(structural_laws) == 1
                else "mixed donor laws: semantic and structural channels must be "
                "interpreted according to their per-model stored contracts"
            ),
        },
        "figures": [str(path) for path in figures],
        "interpretation_contract": {
            "scores": "immutable canonical cached donor-swap learned-head scores",
            "legacy_protocols": (
                "v3 is loaded read-only only after contract-fingerprint validation; "
                "the original protocol and donor laws are retained in every table"
            ),
            "score_distance": "exact per-head score mass grouped only for display",
            "head_profile_colocalization": (
                "within-head semantic/structural distance-profile overlap, summarized "
                "per layer for score mass and opportunity-corrected profiles"
            ),
            "head_profile_consistency": (
                "layer medians, IQRs, ranges, threshold fractions, and descriptive Tukey "
                "low-overlap outliers from one cached checkpoint per architecture"
            ),
            "head_profile_distance_decomposition": (
                "within-head normalized structural share minus semantic share at each exact "
                "distance; half its absolute value sums to one minus profile overlap"
            ),
            "virtual_node_profile": (
                "virtual-carrier semantic/structural score share and molecular-only overlap "
                "after removing the virtual bin and renormalizing within each head"
            ),
            "profile_precision": (
                "cached exact-distance marginal 95% interval width divided by total "
                "reportable score mass; one seed is checkpoint-conditional"
            ),
            "carriage": (
                "event-normalized canonical F_sens point geometry; no new carriage "
                "uncertainty interval"
            ),
            "causal_limit": (
                "descriptive fitted-model comparison; ZINC has no atomwise ground-truth "
                "semantic contribution"
            ),
        },
    }
    log("json", "summary.json")
    try:
        _write_json(output_dir / "summary.json", result)
    except Exception as error:
        raise RuntimeError(
            "ZINC cache analysis failed while writing summary.json: "
            f"{type(error).__name__}: {error}"
        ) from error
    log("done", str(output_dir))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare six ZINC GRIT models entirely from canonical caches."
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        action="append",
        required=True,
        help="Canonical methodology root; repeat to search fallback roots.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--require-carriage", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
    if args.inventory_only:
        result = {
            "analysis_version": ANALYSIS_VERSION,
            "inventory": cache_inventory(
                args.canonical_root, tasks=tasks, train_seed=int(args.train_seed)
            ),
        }
        print(json.dumps(_json_value(result), indent=2, sort_keys=True))
        return result
    result = run(
        args.canonical_root,
        args.output_dir,
        tasks=tasks,
        train_seed=int(args.train_seed),
        require_carriage=bool(args.require_carriage),
    )
    print(json.dumps(_json_value(result), indent=2, sort_keys=True))
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
