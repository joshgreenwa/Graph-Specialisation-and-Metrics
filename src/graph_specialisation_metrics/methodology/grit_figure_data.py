"""Read-only canonical score adapters and additive GRIT figure diagnostics.

The canonical methodology owns ``scores/raw.pt`` and the definitions of
``D_rel`` and ``J``.  This module validates those artifacts without modifying
them, reconstructs the matching official GRIT checkpoint, and stores only
supplemental attention/routed-value diagnostics in separate contract-keyed
caches.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .backend import CanonicalGritBackend
from .bootstrap import Observation, nested_percentile_interval
from .cache import (
    ReadOnlyCacheArtifact,
    StaleCacheError,
    checkpoint_sha256,
    load_cache_artifact_file,
)
from .protocol import (
    BootstrapPolicy,
    ExecutionPolicy,
    FamilyPolicy,
    MethodologyConfig,
    NumericalPolicy,
    RunSizes,
    SplitManifest,
    stable_hash,
)
from .runner import PreparedTask, estimate_graph_local_head_coordinates
from .sampling import SemanticDonorPool
from .tasks import CanonicalTask, get_task


Head = tuple[int, int]
SELECTED_HEAD_TRANSPORT_PROFILE_VERSION = (
    "selected-head-transport-response-by-carrier-distance-v1"
)


def _as_numpy(value: Any, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _split_manifest_from_record(record: Mapping[str, Any]) -> SplitManifest:
    split_record = record.get("splits")
    if not isinstance(split_record, Mapping):
        raise StaleCacheError("canonical model record has no valid split manifest")
    try:
        splits = SplitManifest(
            discovery=tuple(int(value) for value in split_record["discovery"]),
            causal=tuple(int(value) for value in split_record["causal"]),
            clean_ablation=tuple(
                int(value) for value in split_record["clean_ablation"]
            ),
            semantic_donor_pool=tuple(
                int(value) for value in split_record["semantic_donor_pool"]
            ),
            same_index_space=bool(split_record["same_index_space"]),
            seed=int(split_record["seed"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise StaleCacheError(
            f"canonical model record has an invalid split manifest: {error}"
        ) from error
    splits.validate()
    return splits


def load_canonical_score_artifact(
    path: str | Path,
    *,
    expected_task: str,
) -> ReadOnlyCacheArtifact:
    """Load a validated score artifact and bind it to one registered task."""

    artifact = load_cache_artifact_file(path)
    contract = artifact.metadata.get("contract")
    if not isinstance(contract, Mapping):
        raise StaleCacheError(
            f"canonical score cache has no valid contract: {artifact.path}"
        )
    if contract.get("task") != str(expected_task):
        raise StaleCacheError(
            f"{artifact.path} is for task {contract.get('task')!r}, "
            f"not {expected_task!r}"
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
            f"canonical score cache contract is missing {missing}: {artifact.path}"
        )
    task = get_task(str(expected_task))
    if task.backend_kind != "grit":
        raise StaleCacheError(
            f"{expected_task!r} uses backend {task.backend_kind!r}, not GRIT"
        )
    return artifact


def load_canonical_model_record(
    path: str | Path,
    artifact: ReadOnlyCacheArtifact,
) -> dict[str, Any]:
    """Load ``model.json`` and verify that it describes ``artifact``."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"canonical model record not found: {resolved}")
    record = json.loads(resolved.read_text(encoding="utf-8"))
    contract = artifact.metadata["contract"]
    expected = {
        "task": contract["task"],
        "task_adapter_version": contract["task_adapter_version"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "train_seed": int(contract["train_seed"]),
    }
    mismatches = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise StaleCacheError(
            f"{resolved} does not describe the score artifact: {mismatches}"
        )
    splits = _split_manifest_from_record(record)
    if splits.fingerprint != contract["split_fingerprint"]:
        raise StaleCacheError(
            f"{resolved} split fingerprint {splits.fingerprint!r} does not match "
            f"the score cache {contract['split_fingerprint']!r}"
        )
    return record


@dataclass(frozen=True)
class CanonicalHeadMetrics:
    """Figure-facing arrays copied from a validated canonical score artifact."""

    raw_semantic: np.ndarray
    raw_structural: np.ndarray
    normalized_semantic: np.ndarray
    normalized_structural: np.ndarray
    joint_sensitivity: np.ndarray
    selectivity: np.ndarray
    active: np.ndarray
    estimable: bool
    distance_axis: tuple[str, ...]
    clean_attention_distance: np.ndarray | None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.selectivity.shape)

    @property
    def num_layers(self) -> int:
        return self.shape[0]

    @property
    def num_heads(self) -> int:
        return self.shape[1]

    @classmethod
    def from_scores(cls, scores: Mapping[str, Any]) -> "CanonicalHeadMetrics":
        if "coordinates" not in scores:
            raise KeyError("canonical score payload has no 'coordinates' entry")
        coordinates = scores["coordinates"]
        arrays = {
            "raw_semantic": _as_numpy(
                _field(coordinates, "raw_semantic"), dtype=np.float64
            ),
            "raw_structural": _as_numpy(
                _field(coordinates, "raw_structural"), dtype=np.float64
            ),
            "normalized_semantic": _as_numpy(
                _field(coordinates, "normalized_semantic"), dtype=np.float64
            ),
            "normalized_structural": _as_numpy(
                _field(coordinates, "normalized_structural"), dtype=np.float64
            ),
            "joint_sensitivity": _as_numpy(
                _field(coordinates, "joint_sensitivity"), dtype=np.float64
            ),
            "selectivity": _as_numpy(
                _field(coordinates, "selectivity"), dtype=np.float64
            ),
            "active": _as_numpy(_field(coordinates, "active"), dtype=bool),
        }
        shape = arrays["selectivity"].shape
        if len(shape) != 2:
            raise ValueError(f"expected canonical head arrays [L,H], got {shape}")
        for name, array in arrays.items():
            if array.shape != shape:
                raise ValueError(
                    f"canonical {name} has shape {array.shape}; expected {shape}"
                )

        channels = scores.get("channels", {})
        for channel, coordinate_name in (
            ("semantic", "raw_semantic"),
            ("structural", "raw_structural"),
        ):
            if channel in channels and "raw" in channels[channel]:
                raw = _as_numpy(channels[channel]["raw"], dtype=np.float64)
                if raw.shape != shape or not np.allclose(
                    raw, arrays[coordinate_name], equal_nan=True
                ):
                    raise ValueError(
                        f"canonical coordinate and channel arrays disagree for {channel}"
                    )

        attention_distance = scores.get("clean_attention_distance")
        if attention_distance is not None:
            attention_distance = _as_numpy(attention_distance, dtype=np.float64)
            if (
                attention_distance.ndim != 3
                or attention_distance.shape[:2] != shape
            ):
                raise ValueError(
                    "clean_attention_distance must be [L,H,D] and match head arrays; "
                    f"got {attention_distance.shape} for {shape}"
                )
        axis = tuple(str(label) for label in scores.get("axis", ()))
        if (
            attention_distance is not None
            and len(axis) != attention_distance.shape[-1]
        ):
            raise ValueError(
                f"distance axis has {len(axis)} labels for "
                f"{attention_distance.shape[-1]} bins"
            )
        return cls(
            **arrays,
            estimable=bool(_field(coordinates, "estimable")),
            distance_axis=axis,
            clean_attention_distance=attention_distance,
        )

    def head_record(self, head: Head) -> dict[str, Any]:
        layer, index = validate_head(head, self.shape)
        return {
            "layer": layer,
            "head": index,
            "semantic": float(self.normalized_semantic[layer, index]),
            "structural": float(self.normalized_structural[layer, index]),
            "joint_sensitivity": float(self.joint_sensitivity[layer, index]),
            "selectivity": float(self.selectivity[layer, index]),
            "active": bool(self.active[layer, index]),
        }


def validate_head(head: Head, shape: tuple[int, int]) -> Head:
    layer, index = int(head[0]), int(head[1])
    if not (0 <= layer < shape[0] and 0 <= index < shape[1]):
        raise IndexError(f"head {(layer, index)} is outside canonical shape {shape}")
    return layer, index


def compute_selected_head_transport_profiles(
    scores: Mapping[str, Any],
    heads: Sequence[Head],
    *,
    bootstrap: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Resolve selected heads' score mass by carrier distance.

    Each curve is the exact additive distance decomposition of the canonical
    transport score. Semantic and structural curves are divided by the same
    fixed within-model channel means used to construct ``D_rel`` and ``J``;
    consequently, a head's curve sums to its corresponding normalised channel
    score. When requested, uncertainty follows the registered
    seed->graph->source->donor bootstrap, using only sufficient statistics
    already stored in the score cache.
    """

    metrics = CanonicalHeadMetrics.from_scores(scores)
    selected = tuple(validate_head(head, metrics.shape) for head in heads)
    if not selected:
        raise ValueError("at least one head is required")
    if len(set(selected)) != len(selected):
        raise ValueError("selected transport-profile heads must be unique")
    axis = tuple(str(label) for label in scores.get("axis", ()))
    if not axis:
        raise ValueError("score cache has no carrier-distance axis")

    channels = scores.get("channels")
    if not isinstance(channels, Mapping):
        raise ValueError("score cache has no channel sufficient statistics")

    result_channels: dict[str, Any] = {}
    channel_graph_ids: dict[str, tuple[int, ...]] = {}
    for channel in ("semantic", "structural"):
        channel_scores = channels.get(channel)
        if not isinstance(channel_scores, Mapping):
            raise ValueError(f"score cache has no {channel!r} channel")
        raw = _as_numpy(channel_scores.get("raw"), dtype=np.float64)
        if raw.shape != metrics.shape:
            raise ValueError(
                f"{channel} raw scores have shape {raw.shape}; "
                f"expected {metrics.shape}"
            )
        scale = float(np.mean(raw))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"{channel} within-model normalisation is not positive"
            )

        graph_contribution = channel_scores.get(
            "graph_distance_contribution"
        )
        if (
            not isinstance(graph_contribution, Mapping)
            or not graph_contribution
        ):
            raise ValueError(
                f"{channel} score cache has no graph-distance contributions"
            )
        graph_ids = tuple(sorted(int(value) for value in graph_contribution))
        channel_graph_ids[channel] = graph_ids
        graph_rows = []
        for graph_id in graph_ids:
            contribution = _as_numpy(
                graph_contribution[graph_id], dtype=np.float64
            )
            expected = metrics.shape + (len(axis),)
            if contribution.shape != expected:
                raise ValueError(
                    f"{channel} graph {graph_id} distance contribution has "
                    f"shape {contribution.shape}; expected {expected}"
                )
            graph_rows.append(
                np.stack(
                    [
                        contribution[layer, head]
                        for layer, head in selected
                    ]
                )
            )
        point = np.stack(graph_rows).mean(axis=0)
        selected_raw = np.asarray(
            [raw[layer, head] for layer, head in selected],
            dtype=np.float64,
        )
        reconstructed = point.sum(axis=-1)
        if not np.allclose(
            reconstructed,
            selected_raw,
            rtol=2e-5,
            atol=1e-8,
            equal_nan=True,
        ):
            residual = float(
                np.nanmax(np.abs(reconstructed - selected_raw))
            )
            raise ValueError(
                f"{channel} carrier-distance profiles do not reconstruct "
                "selected head scores "
                f"(max residual {residual:.3e})"
            )

        support = channel_scores.get("distance_support")
        if not isinstance(support, Mapping) or "reportable" not in support:
            raise ValueError(
                f"{channel} score cache has no registered "
                "distance-reporting mask"
            )
        support_axis = tuple(str(label) for label in support.get("axis", axis))
        if support_axis != axis:
            raise ValueError(
                f"{channel} support axis {support_axis} does not match {axis}"
            )
        reportable = _as_numpy(support["reportable"], dtype=bool)
        if reportable.shape != (len(axis),):
            raise ValueError(
                f"{channel} reporting mask has shape {reportable.shape}; "
                f"expected {(len(axis),)}"
            )

        normalised_point = point / scale
        if bootstrap:
            events = channel_scores.get("events")
            if not isinstance(events, Sequence) or not events:
                raise ValueError(
                    f"{channel} score cache has no event table for uncertainty"
                )
            observations: list[Observation] = []
            for row in events:
                contribution = _as_numpy(
                    row["distance_contribution"], dtype=np.float64
                )
                expected = metrics.shape + (len(axis),)
                if contribution.shape != expected:
                    raise ValueError(
                        f"{channel} event contribution has shape "
                        f"{contribution.shape}; expected {expected}"
                    )
                observations.append(
                    Observation(
                        seed=int(seed),
                        graph=int(row["graph_id"]),
                        source=int(row["source"]),
                        donor=int(row["draw"]),
                        value=np.stack(
                            [
                                contribution[layer, head]
                                for layer, head in selected
                            ]
                        ),
                    )
                )
            policy = BootstrapPolicy(
                resample_source=bool(
                    channel_scores.get("resample_source", True)
                )
            )
            interval = nested_percentile_interval(
                observations,
                policy,
                graph_reduce=lambda rows: np.mean(rows, axis=0),
            )
            if not np.allclose(
                interval.estimate,
                point,
                rtol=2e-5,
                atol=1e-8,
                equal_nan=True,
            ):
                raise ValueError(
                    f"{channel} event table does not reconstruct graph-level "
                    "distance contributions"
                )
            low = np.asarray(interval.low, dtype=np.float64) / scale
            high = np.asarray(interval.high, dtype=np.float64) / scale
            interval_metadata = {
                "replicates": int(interval.replicates),
                "rng_seed": int(interval.rng_seed),
                "resampled_levels": tuple(interval.resampled_levels),
            }
        else:
            low = normalised_point.copy()
            high = normalised_point.copy()
            interval_metadata = {
                "replicates": 0,
                "rng_seed": None,
                "resampled_levels": (),
            }

        result_channels[channel] = {
            "estimate": normalised_point,
            "low": low,
            "high": high,
            "reportable": reportable,
            "normalisation_mean": scale,
            "normalised_head_total": selected_raw / scale,
            "raw_head_total": selected_raw,
            "reconstruction_max_abs": float(
                np.max(np.abs(reconstructed - selected_raw))
            ),
            **interval_metadata,
        }

    if channel_graph_ids["semantic"] != channel_graph_ids["structural"]:
        raise ValueError(
            "semantic and structural transport profiles use different graph IDs"
        )
    return {
        "estimator": SELECTED_HEAD_TRANSPORT_PROFILE_VERSION,
        "heads": selected,
        "axis": axis,
        "graph_ids": channel_graph_ids["semantic"],
        "n_graphs": len(channel_graph_ids["semantic"]),
        "normalisation": "fixed within-model channel mean used by D_rel and J",
        "channels": result_channels,
    }


def _select_extreme_head(
    metrics: CanonicalHeadMetrics,
    *,
    largest: bool,
    active_only: bool = True,
    excluded_heads: Sequence[Head] = (),
) -> Head:
    finite = np.isfinite(metrics.selectivity) & np.isfinite(
        metrics.joint_sensitivity
    )
    eligible = finite & metrics.active if active_only else finite
    eligible = eligible.copy()
    for head in excluded_heads:
        eligible[validate_head(head, metrics.shape)] = False
    layers, heads = np.where(eligible)
    if not len(layers):
        raise ValueError("no eligible head is available for specialist selection")
    selectivity = metrics.selectivity[layers, heads]
    joint = metrics.joint_sensitivity[layers, heads]
    order = (
        np.lexsort((-joint, -selectivity))
        if largest
        else np.lexsort((-joint, selectivity))
    )
    return int(layers[order[0]]), int(heads[order[0]])


def select_specialist_heads(
    metrics: CanonicalHeadMetrics,
    *,
    semantic_head: Head | None = None,
    structural_head: Head | None = None,
    active_only: bool = True,
) -> dict[str, Head]:
    """Select the maximum- and minimum-``D_rel`` active heads."""

    if semantic_head is None:
        semantic_head = _select_extreme_head(
            metrics, largest=True, active_only=active_only
        )
    else:
        semantic_head = validate_head(semantic_head, metrics.shape)
    if structural_head is None:
        structural_head = _select_extreme_head(
            metrics,
            largest=False,
            active_only=active_only,
            excluded_heads=(semantic_head,),
        )
    else:
        structural_head = validate_head(structural_head, metrics.shape)
    return {"semantic": semantic_head, "structural": structural_head}


def select_structural_specialist_head(
    metrics: CanonicalHeadMetrics,
    *,
    active_only: bool = True,
    excluded_heads: Sequence[Head] = (),
) -> Head:
    return _select_extreme_head(
        metrics,
        largest=False,
        active_only=active_only,
        excluded_heads=excluded_heads,
    )


def select_structurally_selective_heads(
    metrics: CanonicalHeadMetrics,
    *,
    maximum_d_rel: float = 0.0,
    active_only: bool = True,
) -> tuple[Head, ...]:
    """Return every structurally selective head, ordered by increasing ``D_rel``."""

    eligible = np.isfinite(metrics.selectivity) & np.isfinite(
        metrics.joint_sensitivity
    )
    if active_only:
        eligible &= metrics.active
    eligible &= metrics.selectivity < float(maximum_d_rel)
    layers, heads = np.where(eligible)
    if not len(layers):
        return ()
    order = np.lexsort(
        (
            heads,
            layers,
            -metrics.joint_sensitivity[layers, heads],
            metrics.selectivity[layers, heads],
        )
    )
    return tuple(
        (int(layers[position]), int(heads[position])) for position in order
    )


def select_attention_grid_indices(
    entries: Sequence[int],
    *,
    num_rows: int,
) -> list[int]:
    num_rows = int(num_rows)
    if num_rows < 1:
        raise ValueError("attention-grid num_rows must be positive")
    indices = [int(value) for value in entries]
    if len(indices) < num_rows:
        raise ValueError(
            f"attention grid needs {num_rows} graph indices, but only "
            f"{len(indices)} were configured"
        )
    return indices[:num_rows]


def select_ranked_heads(
    metrics: CanonicalHeadMetrics,
    *,
    semantic_count: int = 3,
    structural_count: int = 0,
    joint_count: int = 3,
    active_only: bool = True,
    joint_generalist_max_abs_selectivity: float | None = None,
) -> dict[str, Head]:
    """Rank semantic, structural, and high-``J`` generalist heads."""

    semantic_count = int(semantic_count)
    structural_count = int(structural_count)
    joint_count = int(joint_count)
    if semantic_count < 0 or structural_count < 0 or joint_count < 0:
        raise ValueError("rank counts must be non-negative")
    finite = np.isfinite(metrics.selectivity) & np.isfinite(
        metrics.joint_sensitivity
    )
    eligible = finite & metrics.active if active_only else finite
    semantic_layers, semantic_heads = np.where(eligible)
    if len(semantic_layers) < semantic_count:
        raise ValueError(
            f"only {len(semantic_layers)} eligible heads are available for a "
            f"top-{semantic_count} semantic ranking"
        )
    semantic_selectivity = metrics.selectivity[semantic_layers, semantic_heads]
    semantic_joint = metrics.joint_sensitivity[semantic_layers, semantic_heads]
    semantic_order = np.lexsort((-semantic_joint, -semantic_selectivity))
    if len(semantic_layers) < structural_count:
        raise ValueError(
            f"only {len(semantic_layers)} eligible heads are available for a "
            f"top-{structural_count} structural ranking"
        )
    structural_order = np.lexsort(
        (-semantic_joint, semantic_selectivity)
    )

    joint_eligible = eligible.copy()
    if joint_generalist_max_abs_selectivity is not None:
        bound = float(joint_generalist_max_abs_selectivity)
        if bound < 0:
            raise ValueError("joint generalist |D_rel| bound must be non-negative")
        joint_eligible &= np.abs(metrics.selectivity) <= bound
    joint_layers, joint_heads = np.where(joint_eligible)
    if len(joint_layers) < joint_count:
        raise ValueError(
            f"only {len(joint_layers)} eligible heads are available for a "
            f"top-{joint_count} joint-generalist ranking"
        )
    joint_selectivity = metrics.selectivity[joint_layers, joint_heads]
    joint_values = metrics.joint_sensitivity[joint_layers, joint_heads]
    joint_order = np.lexsort((-joint_selectivity, -joint_values))

    ranked: dict[str, Head] = {}
    for rank, position in enumerate(semantic_order[:semantic_count], start=1):
        ranked[f"top_semantic_{rank}"] = (
            int(semantic_layers[position]),
            int(semantic_heads[position]),
        )
    for rank, position in enumerate(
        structural_order[:structural_count], start=1
    ):
        ranked[f"top_structural_{rank}"] = (
            int(semantic_layers[position]),
            int(semantic_heads[position]),
        )
    for rank, position in enumerate(joint_order[:joint_count], start=1):
        ranked[f"top_joint_{rank}"] = (
            int(joint_layers[position]),
            int(joint_heads[position]),
        )
    return ranked


class SupplementalCache:
    """Immutable contract-keyed storage for non-canonical diagnostics."""

    SCHEMA_VERSION = "focused_grit_figure_diagnostics.v1"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str, contract: Mapping[str, Any]) -> Path:
        fingerprint = stable_hash(
            {
                "schema_version": self.SCHEMA_VERSION,
                "name": str(name),
                "contract": dict(contract),
            }
        )
        return self.root / f"{name}-{fingerprint[:16]}.pt"

    def load(
        self,
        name: str,
        contract: Mapping[str, Any],
    ) -> tuple[Any, Path] | None:
        """Read one exact supplemental artifact without invoking a compute path."""

        path = self.path_for(name, contract)
        if not path.exists():
            return None
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") != self.SCHEMA_VERSION
            or payload.get("contract") != dict(contract)
            or "value" not in payload
        ):
            raise StaleCacheError(
                f"supplemental figure cache is malformed: {path}"
            )
        return payload["value"], path

    def load_or_compute(
        self,
        name: str,
        contract: Mapping[str, Any],
        compute: Callable[[], Any],
        *,
        force: bool = False,
    ) -> tuple[Any, Path, bool]:
        path = self.path_for(name, contract)
        if not force:
            cached = self.load(name, contract)
            if cached is not None:
                value, cached_path = cached
                return value, cached_path, True
        value = compute()
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=path.name, suffix=".partial", dir=path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            torch.save(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "contract": dict(contract),
                    "value": value,
                },
                temporary_path,
            )
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return value, path, False


def methodology_config_from_record(
    path_or_record: str | Path | Mapping[str, Any],
    *,
    accelerator: str | None = None,
) -> MethodologyConfig:
    """Reconstruct the canonical public configuration from ``protocol.json``."""

    if isinstance(path_or_record, Mapping):
        record = dict(path_or_record)
    else:
        path = Path(path_or_record).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"canonical protocol record not found: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
    return MethodologyConfig(
        output_dir=str(record["output_dir"]),
        tasks=tuple(str(value) for value in record["tasks"]),
        train_seeds=tuple(int(value) for value in record["train_seeds"]),
        task_train_seeds={
            str(task): tuple(int(seed) for seed in seeds)
            for task, seeds in record.get("task_train_seeds", {}).items()
        },
        phases=(),
        sizes=RunSizes(**record["sizes"]),
        numerical=NumericalPolicy(**record["numerical"]),
        bootstrap=BootstrapPolicy(**record["bootstrap"]),
        families=FamilyPolicy(**record["families"]),
        execution=ExecutionPolicy(**record.get("execution", {})),
        analysis_seed=int(record["analysis_seed"]),
        accelerator=str(accelerator or record.get("accelerator", "cuda:0")),
        num_threads=int(record.get("num_threads", 4)),
        checkpoints=dict(record.get("checkpoints", {})),
        task_overrides={
            str(task): dict(values)
            for task, values in record.get("task_overrides", {}).items()
        },
        figure_overrides=dict(record.get("figure_overrides", {})),
        skip_install=True,
        resume=True,
        force=False,
        strict_audits=bool(record.get("strict_audits", False)),
        compute_beneficial_carriage=bool(
            record.get("beneficial_carriage", True)
        ),
    )


def methodology_config_for_artifact(
    artifact: ReadOnlyCacheArtifact,
    protocol_paths: Sequence[str | Path],
    *,
    accelerator: str | None = None,
) -> tuple[MethodologyConfig, Path, str]:
    """Load the protocol record scientifically bound to a score artifact.

    Task/seed protocol records are immutable run inputs, whereas a shared root
    record may later be rewritten by finalisation or another canonical run.
    Candidate order is therefore significant. A current record is accepted when
    its reconstructed fingerprint matches; a legacy record is accepted when its
    stored fingerprint matches, because later protocol-schema additions can
    change reconstruction without changing the recorded run configuration.
    """

    contract = artifact.metadata.get("contract", {})
    expected = str(contract.get("protocol_fingerprint", ""))
    if not expected:
        raise ValueError("canonical score cache has no protocol fingerprint")

    checked: list[str] = []
    seen: set[Path] = set()
    for value in protocol_paths:
        path = Path(value).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            checked.append(f"{path} (missing)")
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            config = methodology_config_from_record(
                record,
                accelerator=accelerator,
            )
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            checked.append(f"{path} (invalid: {error})")
            continue
        task_name = str(contract.get("task", ""))
        train_seed = int(contract.get("train_seed", -1))
        if (
            record.get("protocol_version") != artifact.metadata.get("protocol_version")
            or task_name not in config.tasks
            or train_seed not in config.seeds_for(task_name)
        ):
            checked.append(f"{path} (task, seed, or protocol version mismatch)")
            continue
        recorded = str(record.get("fingerprint", ""))
        reconstructed = config.fingerprint
        if expected in {recorded, reconstructed}:
            return config, path, expected
        checked.append(
            f"{path} (recorded fingerprint {recorded or 'missing'}; "
            f"reconstructed fingerprint {reconstructed})"
        )

    details = "; ".join(checked) if checked else "no candidates supplied"
    raise ValueError(
        "no protocol.json describes the scientific configuration bound to the "
        f"canonical score cache (expected fingerprint {expected}); checked: {details}"
    )


def _task_with_protocol_overrides(
    task_name: str,
    task_overrides: Mapping[str, Any],
) -> CanonicalTask:
    scientific_names = set(CanonicalTask.__dataclass_fields__) - {
        "name",
        "backend_kind",
        "spec",
    }
    scientific = {
        key: value
        for key, value in task_overrides.items()
        if key in scientific_names
    }
    task = get_task(task_name, scientific)
    if {
        "sigma",
        "output_representation",
        "sigma_policy",
    } & set(task_overrides):
        sigma_override = task_overrides.get("sigma")
        if sigma_override is not None and np.isscalar(sigma_override):
            sigma_override = (float(sigma_override),)
        task = dataclasses.replace(
            task,
            output=dataclasses.replace(
                task.output,
                sigma=(
                    tuple(float(value) for value in sigma_override)
                    if sigma_override is not None
                    else task.output.sigma
                ),
                representation=str(
                    task_overrides.get(
                        "output_representation", task.output.representation
                    )
                ),
                sigma_policy=str(
                    task_overrides.get("sigma_policy", task.output.sigma_policy)
                ),
            ),
        )
    return task


@dataclass(frozen=True)
class GritFigureRuntime:
    """Verified official-GRIT runtime used only for supplemental diagnostics."""

    prepared: PreparedTask
    checkpoint_descriptor: str
    checkpoint_sha256: str
    protocol_config: MethodologyConfig

    @property
    def runtime(self):
        return self.prepared.runtime

    @property
    def backend(self):
        return self.prepared.backend


def build_verified_grit_figure_runtime(
    artifact: ReadOnlyCacheArtifact,
    model_record: Mapping[str, Any],
    protocol_config: MethodologyConfig,
    protocol_record_fingerprint: str,
    *,
    runtime_output_dir: str | Path,
) -> GritFigureRuntime:
    """Reconstruct GRIT and fail if checkpoint, adapter, or geometry differs."""

    from ..carriage import env
    from ..carriage.tasks import resolve_dataset_dir
    from ..specialisation.model import GritHeadModel, SpecConfig

    contract = artifact.metadata["contract"]
    task_name = str(contract["task"])
    if contract.get("protocol_fingerprint") != str(protocol_record_fingerprint):
        raise ValueError(
            "protocol.json does not describe the scientific configuration bound "
            "to the canonical score cache"
        )
    overrides = dict(protocol_config.task_overrides.get(task_name, {}))
    task = _task_with_protocol_overrides(task_name, overrides)
    if task.backend_kind != "grit":
        raise ValueError(
            f"focused GRIT runtime cannot load backend {task.backend_kind!r}"
        )
    if contract.get("task_adapter_version") != task.adapter_version:
        raise ValueError(
            "canonical cache adapter version differs from the current GRIT task"
        )

    runtime_output_dir = Path(runtime_output_dir)
    runtime_output_dir.mkdir(parents=True, exist_ok=True)
    spec = task.spec
    repo_dir = Path(
        str(
            overrides.get(
                "grit_repo_dir",
                spec.grit_repo_dir
                or (f"/content/GRIT_{spec.name}" if spec.env_hooks else "/content/GRIT"),
            )
        )
    )
    env.clone_grit(repo_dir, spec.grit_repo, spec.grit_commit, force_fresh=False)
    for hook in spec.env_hooks:
        hook(repo_dir)
    env.prepare_inprocess_grit(repo_dir)
    config_file = str(
        overrides.get("config_file")
        or env.resolve_config(spec, repo_dir, runtime_output_dir)
    )
    drive_dir = str(overrides.get("drive_dir", spec.drive_dir))
    dataset_dir = str(
        overrides.get("dataset_dir") or resolve_dataset_dir(spec, drive_dir)
    )
    checkpoint = Path(str(model_record["checkpoint"])).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint recorded by canonical model.json is unavailable: {checkpoint}"
        )
    digest = checkpoint_sha256(checkpoint)
    if digest != str(contract["checkpoint_sha256"]):
        raise ValueError(
            "loaded GRIT checkpoint does not match canonical score cache: "
            f"{digest} != {contract['checkpoint_sha256']}"
        )

    seed = int(contract["train_seed"])
    runtime = GritHeadModel(
        spec,
        SpecConfig(
            ckpt=str(checkpoint),
            out_dir=str(runtime_output_dir),
            dataset_dir=dataset_dir,
            config_file=config_file,
            accelerator=str(protocol_config.accelerator),
            seed=seed,
            num_threads=int(protocol_config.num_threads),
            eval_split=str(overrides.get("eval_split", "test")),
            donor_split=str(overrides.get("donor_split", "train")),
            eval_metric=False,
            analysis_seed=int(protocol_config.analysis_seed),
            donors=int(contract["donors_per_source"]),
            content_adapter=spec.content_adapter,
            resume=True,
        ),
    ).load()
    sigma = np.asarray(contract["sigma"], dtype=np.float64)
    backend = CanonicalGritBackend(runtime, task, sigma=sigma)
    expected_geometry = {
        str(key): int(value) for key, value in contract["model_geometry"].items()
    }
    if backend.geometry != expected_geometry:
        raise ValueError(
            f"loaded GRIT geometry {backend.geometry} != canonical {expected_geometry}"
        )
    expected_parameters = model_record.get("parameter_count")
    observed_parameters = runtime.checks.get("num_parameters")
    if (
        expected_parameters is not None
        and int(observed_parameters) != int(expected_parameters)
    ):
        raise ValueError(
            f"loaded GRIT parameter count {observed_parameters} != "
            f"canonical {expected_parameters}"
        )
    splits = _split_manifest_from_record(model_record)
    donor_pool = SemanticDonorPool(
        [
            (graph_id, runtime.donor_ds[graph_id])
            for graph_id in splits.semantic_donor_pool
        ],
        adapter=task.content_adapter,
    )
    prepared = PreparedTask(
        task=task,
        runtime=runtime,
        backend=backend,
        output_dir=runtime_output_dir,
        checkpoint=checkpoint,
        checkpoint_sha=digest,
        sigma=sigma,
        splits=splits,
        donor_pool=donor_pool,
        progress=None,
    )
    return GritFigureRuntime(
        prepared=prepared,
        checkpoint_descriptor=str(checkpoint),
        checkpoint_sha256=digest,
        protocol_config=protocol_config,
    )


@dataclass(frozen=True)
class GritDiagnosticCapture:
    """One-graph tensors at GRIT's exact sparse attention sites."""

    node_only_logits: tuple[Any, ...]
    relation_logits: tuple[Any, ...]
    attention: tuple[Any, ...]
    transport: tuple[Any, ...]


class GritDiagnosticExtractor:
    """Capture attention, routed output, and GRIT-native raw-logit controls."""

    def __init__(self, figure_runtime: GritFigureRuntime):
        self.figure_runtime = figure_runtime
        self.gm = figure_runtime.runtime

    def extract(self, data: Any) -> GritDiagnosticCapture:
        import torch
        from torch_geometric.data import Batch

        layers = self.gm.attn_layers
        node_logits: list[Any | None] = [None] * len(layers)
        relation_logits: list[Any | None] = [None] * len(layers)
        attention: list[Any | None] = [None] * len(layers)
        transport: list[Any | None] = [None] * len(layers)
        handles = []

        def make_hook(layer_index: int):
            def hook(module, inputs, output):
                batch = inputs[0]
                edge_index = batch.edge_index.long()
                source, receiver = edge_index[0], edge_index[1]
                node_score = batch.K_h[source] + batch.Q_h[receiver]
                node_field = module.act(node_score)
                node_raw = torch.einsum(
                    "ehd,dhc->ehc", node_field, module.Aw
                ).squeeze(-1)
                relation_field = (
                    batch.wE.view(-1, module.num_heads, module.out_dim)
                    if batch.get("wE", None) is not None
                    else node_field
                )
                relation_raw = torch.einsum(
                    "ehd,dhc->ehc", relation_field, module.Aw
                ).squeeze(-1)
                if module.clamp is not None:
                    node_raw = torch.clamp(
                        node_raw, min=-module.clamp, max=module.clamp
                    )
                    relation_raw = torch.clamp(
                        relation_raw, min=-module.clamp, max=module.clamp
                    )
                n = int(batch.num_nodes)
                h = int(module.num_heads)
                node_dense = torch.full(
                    (h, n, n),
                    torch.nan,
                    dtype=node_raw.dtype,
                    device=node_raw.device,
                )
                relation_dense = torch.full_like(node_dense, torch.nan)
                attention_dense = torch.zeros_like(node_dense)
                node_dense[:, receiver, source] = node_raw.transpose(0, 1)
                relation_dense[:, receiver, source] = relation_raw.transpose(0, 1)
                attention_dense[:, receiver, source] = (
                    batch.attn.detach().squeeze(-1).transpose(0, 1)
                )
                routed = output[0] if isinstance(output, (tuple, list)) else output
                node_logits[layer_index] = node_dense.detach()
                relation_logits[layer_index] = relation_dense.detach()
                attention[layer_index] = attention_dense.detach()
                transport[layer_index] = routed.detach()

            return hook

        for layer_index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_hook(layer_index)))
        try:
            batch = Batch.from_data_list([data.clone()]).to(self.gm.device)
            with torch.no_grad():
                self.gm.model(batch)
        finally:
            for handle in handles:
                handle.remove()
        groups = (node_logits, relation_logits, attention, transport)
        if any(any(value is None for value in group) for group in groups):
            raise RuntimeError("one or more GRIT diagnostic hooks did not fire")
        n = int(data.num_nodes)
        return GritDiagnosticCapture(
            node_only_logits=tuple(value[:, :n, :n] for value in node_logits),
            relation_logits=tuple(
                value[:, :n, :n] for value in relation_logits
            ),
            attention=tuple(value[:, :n, :n] for value in attention),
            transport=tuple(value[:n] for value in transport),
        )


_ATOMIC_NUMBERS = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

# Exact vocabulary used to create the PyG ZINC subset.  These are augmented
# RDKit atom symbols, not latent clusters: the optional H count and formal
# charge are part of the atom identity.  The source dictionaries are the
# ``molecules/atom_dict.pickle`` and ``bond_dict.pickle`` files distributed
# with the benchmarking-GNNs ZINC data from which PyG ZINC was prepared.
ZINC_ATOM_TYPES = (
    "C",
    "O",
    "N",
    "F",
    "C H1",
    "S",
    "Cl",
    "O -",
    "N H1 +",
    "Br",
    "N H3 +",
    "N H2 +",
    "N +",
    "N -",
    "S -",
    "I",
    "P",
    "O H1 +",
    "N H1 -",
    "O +",
    "S +",
    "P H1",
    "P H2",
    "C H2 -",
    "P +",
    "S H1 +",
    "C H1 -",
    "P H1 +",
)

CHEMISTRY_FOCUS_VERSION = "pcqm_chemistry_focus.v1+explicit_hydrogen"
QM9_VALENCE_TOLERANT_CHEMISTRY_VERSION = (
    f"{CHEMISTRY_FOCUS_VERSION}+qm9_valence_tolerant_v1"
)

_RDKIT_SANITIZED_PROPERTY = "_graph_specialisation_sanitized"
_RDKIT_SANITIZATION_ERROR_PROPERTY = (
    "_graph_specialisation_sanitization_error"
)

ZINC_FIGURE_TASKS = (
    "zinc",
    "zinc_1hop",
    "zinc_1hop_local",
    "zinc_2hop",
    "zinc_1hop_vnode",
    "zinc_2hop_vnode",
)
QM9_FIGURE_TASKS = (
    "qm9_gap_dense",
    "qm9_gap_1hop",
    "qm9_gap_1hop_vnode",
)
SUPPORTED_GRIT_FIGURE_TASKS = ZINC_FIGURE_TASKS + QM9_FIGURE_TASKS

_FIGURE_IDENTITIES = {
    "zinc": {
        "dataset_label": "ZINC",
        "model_label": "GRIT",
    },
    "zinc_1hop": {
        "dataset_label": "ZINC",
        "model_label": "GRIT (1-hop)",
    },
    "zinc_1hop_local": {
        "dataset_label": "ZINC",
        "model_label": "GRIT (1-hop + local RRWP)",
    },
    "zinc_2hop": {
        "dataset_label": "ZINC",
        "model_label": "GRIT (2-hop)",
    },
    "zinc_1hop_vnode": {
        "dataset_label": "ZINC",
        "model_label": "GRIT (1-hop + VN)",
    },
    "zinc_2hop_vnode": {
        "dataset_label": "ZINC",
        "model_label": "GRIT (2-hop + VN)",
    },
    "qm9_gap_dense": {
        "dataset_label": "QM9 HOMO–LUMO gap",
        "model_label": "GRIT",
    },
    "qm9_gap_1hop": {
        "dataset_label": "QM9 HOMO–LUMO gap",
        "model_label": "GRIT (1-hop)",
    },
    "qm9_gap_1hop_vnode": {
        "dataset_label": "QM9 HOMO–LUMO gap",
        "model_label": "GRIT (1-hop + VN)",
    },
}


def molecular_task_family(task_name: str) -> str:
    """Return the shared chemistry/evaluation family for a GRIT task."""

    task_name = str(task_name)
    if task_name in ZINC_FIGURE_TASKS:
        return "zinc"
    if task_name in QM9_FIGURE_TASKS:
        return "qm9"
    raise ValueError(
        f"no molecular figure family is registered for GRIT task {task_name!r}"
    )


def chemistry_focus_version(task_name: str) -> str:
    """Return the task-scoped chemistry/cache contract version."""

    if molecular_task_family(task_name) == "qm9":
        return QM9_VALENCE_TOLERANT_CHEMISTRY_VERSION
    return CHEMISTRY_FOCUS_VERSION


def resolve_canonical_task_root(
    task_name: str,
    train_seed: int,
    canonical_roots: Sequence[str | Path],
    *,
    search_root: str | Path | None = None,
) -> Path:
    """Locate one compatible task/seed canonical cache without guessing.

    Configured roots are checked in order. If none contains the task, an optional
    one-level search finds canonical output roots under the shared metrics folder.
    A candidate must contain both ``scores/raw.pt`` and ``model.json`` and pass
    the current canonical score-artifact validation, including protocol version.
    Structurally complete but stale caches are reported rather than selected.
    """

    task_name = str(task_name)
    seed_name = f"seed_{int(train_seed)}"
    configured = [
        Path(root).expanduser() / task_name / seed_name
        for root in canonical_roots
    ]
    checked: list[Path] = []
    rejected: dict[Path, str] = {}

    def compatible(candidate: Path) -> bool:
        score_path = candidate / "cache/scores/raw.pt"
        if not score_path.is_file() or not (candidate / "model.json").is_file():
            return False
        try:
            load_canonical_score_artifact(score_path, expected_task=task_name)
        except (FileNotFoundError, StaleCacheError) as error:
            rejected[candidate] = str(error)
            return False
        return True

    for candidate in configured:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if compatible(candidate):
            return candidate

    discovered: list[Path] = []
    if search_root is not None:
        root = Path(search_root).expanduser().resolve()
        if root.is_dir():
            for candidate in sorted(root.glob(f"*/{task_name}/{seed_name}")):
                candidate = candidate.resolve()
                if candidate in checked:
                    continue
                checked.append(candidate)
                if compatible(candidate):
                    discovered.append(candidate)
    if len(discovered) == 1:
        return discovered[0]
    if len(discovered) > 1:
        choices = "\n".join(f"  - {path}" for path in discovered)
        raise ValueError(
            f"multiple compatible canonical caches found for {task_name} seed "
            f"{train_seed}; add the intended root to CANONICAL_ROOT_CANDIDATES:\n"
            f"{choices}"
        )
    locations = "\n".join(f"  - {path}" for path in checked)
    rejection_details = "\n".join(
        f"  - {path}: {reason}" for path, reason in rejected.items()
    )
    rejected_message = (
        f"\nRejected incompatible caches:\n{rejection_details}"
        if rejection_details
        else ""
    )
    raise FileNotFoundError(
        f"no compatible canonical score cache was found for {task_name} seed "
        f"{train_seed}. Checked:\n{locations or '  - no candidate roots'}\n"
        f"{rejected_message}\n"
        "Run the canonical methodology scores phase for this registered task, "
        "or add its existing output root to CANONICAL_ROOT_CANDIDATES."
    )


def figure_identity(task_name: str) -> dict[str, str]:
    """Return the invariant reader-facing task/model identity for a figure."""

    task_name = str(task_name)
    identity = dict(
        _FIGURE_IDENTITIES.get(
            task_name,
            {
                "dataset_label": task_name,
                "model_label": "GRIT",
            },
        )
    )
    identity["display_title"] = (
        f"{identity['dataset_label']} — {identity['model_label']}"
    )
    return identity


def graph_node_labels(task_name: str, graph: Any) -> list[str]:
    values = _as_numpy(graph.x).reshape(int(graph.num_nodes), -1)[:, 0]
    if molecular_task_family(task_name) == "zinc":
        labels = []
        for value in values:
            atom_type = int(value)
            if not 0 <= atom_type < len(ZINC_ATOM_TYPES):
                raise ValueError(f"unknown PyG ZINC atom-type ID {atom_type}")
            labels.append(ZINC_ATOM_TYPES[atom_type].split()[0])
        return labels
    return [
        _ATOMIC_NUMBERS.get(int(value), f"Z={int(value)}") for value in values
    ]


def graph_edge_index(graph: Any) -> np.ndarray:
    values = _as_numpy(graph.edge_index, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError(f"expected edge_index [2,E], got {values.shape}")
    return values


def _zinc_atom(token: str):
    from rdkit import Chem

    fields = str(token).split()
    atom = Chem.Atom(fields[0])
    for field in fields[1:]:
        if field.startswith("H") and field[1:].isdigit():
            atom.SetNumExplicitHs(int(field[1:]))
        elif field == "+":
            atom.SetFormalCharge(1)
        elif field == "-":
            atom.SetFormalCharge(-1)
    return atom


def _edge_type_values(graph: Any, edge_count: int) -> np.ndarray:
    if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        raise ValueError("molecular graph has no bond-type edge_attr")
    values = _as_numpy(graph.edge_attr)
    if values.ndim == 2 and values.shape[1] > 1:
        values = np.argmax(values, axis=1)
    else:
        values = values.reshape(-1)
    if len(values) != int(edge_count):
        raise ValueError(
            f"edge_attr has {len(values)} rows for {edge_count} graph edges"
        )
    return values.astype(np.int64, copy=False)


def molecule_from_graph(task_name: str, graph: Any):
    """Reconstruct an RDKit molecule in the graph's exact node-index order.

    Some QM9-derived graph records encode a chemically invalid explicit
    valence.  RDKit sanitization is useful metadata validation, but it must not
    remove a graph or atom from an index-aligned attention visualization.  Such
    records therefore fall back to a property-cache/ring initialization that
    tolerates the invalid valence while preserving every original node.
    """

    from rdkit import Chem, rdBase

    task_name = str(task_name)
    task_family = molecular_task_family(task_name)
    node_values = _as_numpy(graph.x).reshape(int(graph.num_nodes), -1)[:, 0]
    molecule = Chem.RWMol()
    if task_family == "zinc":
        for value in node_values:
            atom_type = int(value)
            if not 0 <= atom_type < len(ZINC_ATOM_TYPES):
                raise ValueError(f"unknown PyG ZINC atom-type ID {atom_type}")
            molecule.AddAtom(_zinc_atom(ZINC_ATOM_TYPES[atom_type]))
        bond_types = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }
    elif task_family == "qm9":
        for value in node_values:
            atomic_number = int(value)
            if atomic_number not in _ATOMIC_NUMBERS:
                raise ValueError(f"unexpected QM9 atomic number {atomic_number}")
            atom = Chem.Atom(atomic_number)
            # QM9 contains all hydrogens explicitly; do not invent additional ones.
            atom.SetNoImplicit(True)
            molecule.AddAtom(atom)
        bond_types = {
            0: Chem.BondType.SINGLE,
            1: Chem.BondType.DOUBLE,
            2: Chem.BondType.TRIPLE,
            3: Chem.BondType.AROMATIC,
        }
    else:
        raise ValueError(
            f"no chemistry decoder is registered for GRIT task {task_name!r}"
        )

    edge_index = graph_edge_index(graph)
    edge_types = _edge_type_values(graph, edge_index.shape[1])
    observed: dict[tuple[int, int], int] = {}
    for edge_position, (source, target) in enumerate(edge_index.T):
        source, target = int(source), int(target)
        if source == target:
            continue
        pair = tuple(sorted((source, target)))
        edge_type = int(edge_types[edge_position])
        if pair in observed and observed[pair] != edge_type:
            raise ValueError(
                f"inconsistent bond types for edge {pair}: "
                f"{observed[pair]} and {edge_type}"
            )
        observed[pair] = edge_type
    for (source, target), edge_type in sorted(observed.items()):
        if edge_type not in bond_types:
            raise ValueError(
                f"unknown {task_name} bond-type ID {edge_type} "
                f"on edge {(source, target)}"
            )
        bond_type = bond_types[edge_type]
        molecule.AddBond(source, target, bond_type)
        if bond_type == Chem.BondType.AROMATIC:
            molecule.GetAtomWithIdx(source).SetIsAromatic(True)
            molecule.GetAtomWithIdx(target).SetIsAromatic(True)
    result = molecule.GetMol()
    try:
        # Sanitization errors are retained as structured cache metadata rather
        # than emitted as noisy RDKit stderr messages during a long Colab run.
        with rdBase.BlockLogs():
            Chem.SanitizeMol(result)
    except Chem.MolSanitizeException as error:
        result = molecule.GetMol()
        result.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(result)
        result.SetBoolProp(_RDKIT_SANITIZED_PROPERTY, False)
        result.SetProp(_RDKIT_SANITIZATION_ERROR_PROPERTY, str(error))
    else:
        result.SetBoolProp(_RDKIT_SANITIZED_PROPERTY, True)
    if result.GetNumAtoms() != int(graph.num_nodes):
        raise RuntimeError("RDKit reconstruction changed the graph atom count")
    return result


def _optional_graph_text(graph: Any, field: str) -> str | None:
    value = getattr(graph, field, None)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    return text or None


def molecule_record(task_name: str, graph: Any) -> dict[str, Any]:
    """Serialisable chemistry record used by cached figure payloads."""

    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    molecule = molecule_from_graph(task_name, graph)
    sanitized = (
        molecule.GetBoolProp(_RDKIT_SANITIZED_PROPERTY)
        if molecule.HasProp(_RDKIT_SANITIZED_PROPERTY)
        else True
    )
    sanitization_error = (
        molecule.GetProp(_RDKIT_SANITIZATION_ERROR_PROPERTY)
        if molecule.HasProp(_RDKIT_SANITIZATION_ERROR_PROPERTY)
        else None
    )
    # RemoveHs can itself request sanitization.  Keep the full, index-aligned
    # graph for tolerant records; this representation is also the one cached
    # in mol_block for the attention depiction.
    display_molecule = (
        Chem.RemoveHs(Chem.Mol(molecule))
        if sanitized
        else Chem.Mol(molecule)
    )
    identity = figure_identity(task_name)
    return {
        "mol_block": Chem.MolToMolBlock(molecule),
        "smiles": Chem.MolToSmiles(display_molecule, canonical=True),
        "formula": rdMolDescriptors.CalcMolFormula(molecule),
        "rdkit_sanitized": bool(sanitized),
        "rdkit_sanitization_error": sanitization_error,
        "molecule_name": _optional_graph_text(graph, "name"),
        "node_labels": [atom.GetSymbol() for atom in molecule.GetAtoms()],
        "chemistry_decoder": (
            "benchmarking-GNNs ZINC atom/bond dictionaries"
            if molecular_task_family(task_name) == "zinc"
            else "PyG QM9 atomic numbers and four bond classes"
        ),
        "chemistry_focus_version": chemistry_focus_version(task_name),
        **identity,
    }


def atom_chemistry_categories(molecule: Any) -> list[str]:
    """PCQM-compatible chemical-group categories in RDKit atom order."""

    from rdkit import Chem

    categories: list[str | None] = [None] * molecule.GetNumAtoms()

    def is_carbonyl_carbon(atom) -> bool:
        return atom.GetSymbol() == "C" and any(
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetOtherAtom(atom).GetSymbol() == "O"
            for bond in atom.GetBonds()
        )

    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        symbol = atom.GetSymbol()
        if symbol == "H":
            categories[index] = "H: hydrogen"
        elif symbol == "O":
            double_carbon = any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(atom).GetSymbol() == "C"
                for bond in atom.GetBonds()
            )
            single_carbonyl = any(
                bond.GetBondType() == Chem.BondType.SINGLE
                and is_carbonyl_carbon(bond.GetOtherAtom(atom))
                for bond in atom.GetBonds()
            )
            if double_carbon:
                categories[index] = "O: carbonyl"
            elif single_carbonyl:
                categories[index] = "O: ester/carboxyl"
            elif atom.GetTotalNumHs() >= 1:
                categories[index] = "O: hydroxyl"
            else:
                categories[index] = "O: other"
        elif symbol == "N":
            if atom.GetIsAromatic():
                categories[index] = "N: aromatic"
            elif any(
                bond.GetBondType() == Chem.BondType.TRIPLE
                and bond.GetOtherAtom(atom).GetSymbol() == "C"
                for bond in atom.GetBonds()
            ):
                categories[index] = "N: nitrile"
            elif atom.GetFormalCharge() > 0 and sum(
                neighbour.GetSymbol() == "O"
                for neighbour in atom.GetNeighbors()
            ) >= 2:
                categories[index] = "N: nitro"
            elif any(
                is_carbonyl_carbon(neighbour)
                for neighbour in atom.GetNeighbors()
            ):
                categories[index] = "N: amide"
            else:
                categories[index] = "N: other"
        elif symbol == "S":
            categories[index] = "S: sulfur"
        elif symbol == "P":
            categories[index] = "P: phosphorus"
        elif symbol in {"F", "Cl", "Br", "I"}:
            categories[index] = "X: halogen"

    ring_info = molecule.GetRingInfo()
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        if categories[index] is not None:
            continue
        rings = ring_info.NumAtomRings(index)
        if rings >= 2:
            categories[index] = "Ring: junction"
        elif rings == 1:
            categories[index] = (
                "Ring: aromatic" if atom.GetIsAromatic() else "Ring: aliphatic"
            )
        elif atom.GetDegree() >= 3:
            categories[index] = "Branch: degree>=3"
        elif atom.GetFormalCharge() > 0:
            categories[index] = "Charge: +"
        elif atom.GetFormalCharge() < 0:
            categories[index] = "Charge: -"
        else:
            categories[index] = "other/diffuse"
    return [str(value) for value in categories]


def collect_attention_examples(
    figure_runtime: GritFigureRuntime,
    *,
    graph_indices: Sequence[int],
    heads: Mapping[str, Head],
) -> dict[str, Any]:
    extractor = GritDiagnosticExtractor(figure_runtime)
    runtime = figure_runtime.runtime
    task_name = figure_runtime.prepared.task.name
    examples = []
    for graph_index in graph_indices:
        graph_index = int(graph_index)
        if not 0 <= graph_index < len(runtime.eval_ds):
            raise IndexError(
                f"{task_name} eval index {graph_index} is outside "
                f"[0, {len(runtime.eval_ds)})"
            )
        graph = runtime.eval_ds[graph_index]
        chemistry = molecule_record(task_name, graph)
        captured = extractor.extract(graph)
        selected = {
            role: captured.attention[layer][head].float().cpu().numpy()
            for role, (layer, head) in heads.items()
        }
        examples.append(
            {
                "dataset_index": graph_index,
                "n_atoms": int(graph.num_nodes),
                "attention": selected,
                **chemistry,
            }
        )
    return {
        "task": task_name,
        "index_space": "GRIT evaluation-split position",
        "heads": dict(heads),
        "examples": examples,
        **figure_identity(task_name),
    }


def label_attention_focus(
    molecule: Any,
    node_mass: np.ndarray,
    *,
    focus_mass: float = 0.75,
    diffuse_threshold: float = 0.35,
) -> str:
    """Label a graph with the PCQM chemistry group receiving most attention."""

    if isinstance(molecule, str):
        from rdkit import Chem

        parsed = Chem.MolFromSmiles(molecule)
        if parsed is None:
            raise ValueError(f"RDKit could not parse SMILES {molecule!r}")
        molecule = parsed
    labels = atom_chemistry_categories(molecule)
    mass = np.asarray(node_mass, dtype=np.float64).reshape(-1)
    if len(labels) != len(mass):
        raise ValueError("molecule atoms and attention mass have different lengths")
    total = float(np.sum(mass))
    if not np.isfinite(total) or total <= 0:
        return "other/diffuse"
    mass = mass / total
    order = np.argsort(-mass)
    selected: list[int] = []
    cumulative = 0.0
    for index in order:
        selected.append(int(index))
        cumulative += float(mass[index])
        if cumulative >= float(focus_mass):
            break
    mass_by_label: dict[str, float] = {}
    for index in selected:
        label = labels[index]
        mass_by_label[label] = mass_by_label.get(label, 0.0) + float(mass[index])
    best, best_mass = max(mass_by_label.items(), key=lambda item: item[1])
    return best if best_mass >= float(diffuse_threshold) else "other/diffuse"


def compute_av_pca_inputs(
    figure_runtime: GritFigureRuntime,
    *,
    head: Head,
    n_graphs: int = 500,
    focus_mass: float = 0.75,
    diffuse_threshold: float = 0.35,
    verbose: bool = True,
) -> dict[str, Any]:
    """Collect pooled native GRIT routed-head outputs for one PCA."""

    key = f"{int(head[0])}:{int(head[1])}"
    return compute_av_pca_inputs_many(
        figure_runtime,
        heads=(head,),
        n_graphs=n_graphs,
        focus_mass=focus_mass,
        diffuse_threshold=diffuse_threshold,
        verbose=verbose,
    )[key]


def compute_av_pca_inputs_many(
    figure_runtime: GritFigureRuntime,
    *,
    heads: Sequence[Head],
    n_graphs: int = 500,
    focus_mass: float = 0.75,
    diffuse_threshold: float = 0.35,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Collect several head PCA inputs in one shared model-forward sweep."""

    shape = (
        int(figure_runtime.runtime.L),
        int(figure_runtime.runtime.H),
    )
    unique_heads = tuple(
        dict.fromkeys(validate_head(head, shape) for head in heads)
    )
    if not unique_heads:
        raise ValueError("at least one head is required for routed-output PCA")
    extractor = GritDiagnosticExtractor(figure_runtime)
    vectors: dict[Head, list[np.ndarray]] = {head: [] for head in unique_heads}
    labels: dict[Head, list[str]] = {head: [] for head in unique_heads}
    positions: dict[Head, list[int]] = {head: [] for head in unique_heads}
    limit = min(int(n_graphs), len(figure_runtime.runtime.eval_ds))
    task_name = figure_runtime.prepared.task.name
    for position in range(limit):
        try:
            graph = figure_runtime.runtime.eval_ds[position]
            captured = extractor.extract(graph)
            molecule = molecule_from_graph(task_name, graph)
        except Exception as error:
            if verbose:
                print(f"  [GRIT routed-output PCA] skipped {position}: {error}")
            continue
        for layer, head_index in unique_heads:
            try:
                vector = (
                    captured.transport[layer][:, head_index, :]
                    .mean(dim=0)
                    .float()
                    .cpu()
                    .numpy()
                )
                matrix = (
                    captured.attention[layer][head_index].float().cpu().numpy()
                )
                denominator = np.clip(
                    matrix.sum(axis=-1, keepdims=True), 1e-12, None
                )
                inbound = (matrix / denominator).mean(axis=0)
                label = label_attention_focus(
                    molecule,
                    inbound,
                    focus_mass=focus_mass,
                    diffuse_threshold=diffuse_threshold,
                )
            except Exception as error:
                if verbose:
                    print(
                        "  [GRIT routed-output PCA] skipped "
                        f"{position} for L{layer} H{head_index}: {error}"
                    )
                continue
            head = (layer, head_index)
            vectors[head].append(vector)
            labels[head].append(label)
            positions[head].append(position)
        if verbose and (position + 1) % 50 == 0:
            print(f"  [GRIT routed-output PCA] {position + 1}/{limit}")
    output: dict[str, dict[str, Any]] = {}
    for head in unique_heads:
        if len(vectors[head]) < 2:
            raise RuntimeError(
                f"fewer than two graphs produced valid vectors for head {head}"
            )
        output[f"{head[0]}:{head[1]}"] = {
            "task": task_name,
            "head": head,
            "vectors": np.stack(vectors[head]),
            "labels": labels[head],
            "positions": np.asarray(positions[head], dtype=np.int64),
            "n_requested": int(n_graphs),
            "n_used": len(vectors[head]),
            "quantity": (
                "mean over receiving nodes of native GRIT routed head output wV"
            ),
            "chemistry_focus_version": chemistry_focus_version(task_name),
            **figure_identity(task_name),
        }
    return output


def compute_layer_av_pca_inputs(
    figure_runtime: GritFigureRuntime,
    *,
    layers: Sequence[int],
    n_graphs: int = 500,
    focus_mass: float = 0.75,
    diffuse_threshold: float = 0.35,
    verbose: bool = True,
) -> dict[str, Any]:
    """Collect routed-output PCA inputs for every head in selected layers.

    GRIT exposes every layer/head transport tensor in one diagnostic forward,
    so all requested layer overviews share one sweep over evaluation molecules.
    A molecule is retained only when every requested layer and head produced a
    valid routed vector and chemistry-focus label.
    """

    runtime = figure_runtime.runtime
    num_layers = int(runtime.L)
    num_heads = int(runtime.H)
    requested_layers = tuple(dict.fromkeys(int(layer) for layer in layers))
    if not requested_layers:
        raise ValueError("at least one layer is required for layer-wide PCA")
    invalid = [
        layer for layer in requested_layers if layer < 0 or layer >= num_layers
    ]
    if invalid:
        raise IndexError(
            f"PCA layers {invalid} are outside [0, {num_layers})"
        )
    if int(n_graphs) < 2:
        raise ValueError("layer-wide PCA requires at least two requested graphs")

    extractor = GritDiagnosticExtractor(figure_runtime)
    vectors: dict[int, list[np.ndarray]] = {
        layer: [] for layer in requested_layers
    }
    labels: dict[int, list[list[str]]] = {
        layer: [] for layer in requested_layers
    }
    positions: list[int] = []
    limit = min(int(n_graphs), len(runtime.eval_ds))
    task_name = figure_runtime.prepared.task.name
    for position in range(limit):
        try:
            graph = runtime.eval_ds[position]
            captured = extractor.extract(graph)
            molecule = molecule_from_graph(task_name, graph)
            graph_vectors: dict[int, np.ndarray] = {}
            graph_labels: dict[int, list[str]] = {}
            for layer in requested_layers:
                layer_vectors = (
                    captured.transport[layer]
                    .mean(dim=0)
                    .float()
                    .cpu()
                    .numpy()
                )
                matrices = (
                    captured.attention[layer].float().cpu().numpy()
                )
                matrices = matrices / np.clip(
                    matrices.sum(axis=-1, keepdims=True),
                    1e-12,
                    None,
                )
                inbound = matrices.mean(axis=1)
                if layer_vectors.shape[0] != num_heads:
                    raise ValueError(
                        f"layer {layer} produced {layer_vectors.shape[0]} head "
                        f"vectors, expected {num_heads}"
                    )
                if inbound.shape[0] != num_heads:
                    raise ValueError(
                        f"layer {layer} produced {inbound.shape[0]} attention "
                        f"profiles, expected {num_heads}"
                    )
                graph_vectors[layer] = layer_vectors
                graph_labels[layer] = [
                    label_attention_focus(
                        molecule,
                        inbound[head],
                        focus_mass=focus_mass,
                        diffuse_threshold=diffuse_threshold,
                    )
                    for head in range(num_heads)
                ]
        except Exception as error:
            if verbose:
                print(
                    "[GRIT layer routed-output PCA] skipped eval position "
                    f"{position}: {error}"
                )
            continue
        for layer in requested_layers:
            vectors[layer].append(graph_vectors[layer])
            labels[layer].append(graph_labels[layer])
        positions.append(position)
        if verbose and (position + 1) % 50 == 0:
            print(f"  [GRIT layer routed-output PCA] {position + 1}/{limit}")

    if len(positions) < 2:
        raise RuntimeError(
            "fewer than two molecules produced valid layer-wide routed outputs"
        )
    return {
        "layers": {
            layer: {
                "task": task_name,
                "layer": int(layer),
                "vectors": np.stack(vectors[layer]),
                "labels": labels[layer],
                "n_used": len(positions),
                **figure_identity(task_name),
            }
            for layer in requested_layers
        },
        "requested_layers": requested_layers,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "positions": np.asarray(positions, dtype=np.int64),
        "n_requested": int(n_graphs),
        "n_used": len(positions),
        "quantity": (
            "mean over receiving nodes of native GRIT routed head output wV"
        ),
        "chemistry_focus_version": chemistry_focus_version(task_name),
        **figure_identity(task_name),
    }


def _mean_keywise_std(matrix: Any) -> Any:
    """Per-head mean query-wise std over finite supported keys."""

    import torch

    finite = torch.isfinite(matrix)
    safe = torch.where(finite, matrix, torch.zeros_like(matrix))
    count = finite.sum(dim=-1)
    mean = safe.sum(dim=-1) / count.clamp_min(1)
    centered = torch.where(
        finite, matrix - mean.unsqueeze(-1), torch.zeros_like(matrix)
    )
    variance = centered.square().sum(dim=-1) / (count - 1).clamp_min(1)
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    valid_query = count > 1
    return (
        torch.where(valid_query, std, torch.zeros_like(std)).sum(dim=-1)
        / valid_query.sum(dim=-1).clamp_min(1)
    )


def _normalised_attention_entropy(matrix: Any) -> Any:
    """Mean query entropy per head, normalised by supported key count."""

    import torch

    attention = torch.clamp(matrix.float(), min=0.0)
    row_mass = attention.sum(dim=-1, keepdim=True)
    valid_query = row_mass.squeeze(-1) > 1e-12
    probabilities = attention / row_mass.clamp_min(1e-12)
    entropy = -(
        probabilities
        * torch.where(
            probabilities > 0,
            probabilities.clamp_min(1e-12).log(),
            torch.zeros_like(probabilities),
        )
    ).sum(dim=-1)
    support = (attention > 0).sum(dim=-1)
    normaliser = support.clamp_min(2).to(entropy.dtype).log()
    normalised = torch.clamp(
        entropy / normaliser.clamp_min(1e-12),
        min=0.0,
        max=1.0,
    )
    return (
        torch.where(valid_query, normalised, torch.zeros_like(normalised)).sum(
            dim=-1
        )
        / valid_query.sum(dim=-1).clamp_min(1)
    )


def aggregate_logit_spread(
    figure_runtime: GritFigureRuntime,
    *,
    n_graphs: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """Aggregate GRIT node-only and relation-conditioned raw-logit spread.

    GRIT does not have Graphormer's additive ``dot + structural bias`` logits.
    The first curve is the exact counterfactual obtained by sending ``K_j+Q_i``
    through GRIT's activation/projection without the edge/RRWP state; the second
    is the actual relation-conditioned pre-softmax logit.
    """

    extractor = GritDiagnosticExtractor(figure_runtime)
    node_rows: list[np.ndarray] = []
    relation_rows: list[np.ndarray] = []
    entropy_rows: list[np.ndarray] = []
    positions: list[int] = []
    limit = min(int(n_graphs), len(figure_runtime.runtime.eval_ds))
    for position in range(limit):
        try:
            graph = figure_runtime.runtime.eval_ds[position]
            captured = extractor.extract(graph)
            node_row = np.stack(
                [
                    _mean_keywise_std(value).float().cpu().numpy()
                    for value in captured.node_only_logits
                ]
            )
            relation_row = np.stack(
                [
                    _mean_keywise_std(value).float().cpu().numpy()
                    for value in captured.relation_logits
                ]
            )
            entropy_row = np.stack(
                [
                    _normalised_attention_entropy(value).float().cpu().numpy()
                    for value in captured.attention
                ]
            )
        except Exception as error:
            if verbose:
                print(f"  [GRIT logit spread] skipped {position}: {error}")
            continue
        node_rows.append(node_row)
        relation_rows.append(relation_row)
        entropy_rows.append(entropy_row)
        positions.append(position)
        if verbose and (position + 1) % 10 == 0:
            print(f"  [GRIT logit spread] {position + 1}/{limit}")
    if not node_rows:
        raise RuntimeError("no graph produced valid GRIT logit diagnostics")

    node_per_graph = np.stack(node_rows)
    relation_per_graph = np.stack(relation_rows)
    entropy_per_graph = np.stack(entropy_rows)
    node_layer_per_graph = node_per_graph.mean(axis=-1)
    relation_layer_per_graph = relation_per_graph.mean(axis=-1)
    ratio_per_graph = np.log10(
        np.clip(
            node_per_graph / np.clip(relation_per_graph, 1e-12, None),
            1e-12,
            None,
        )
    )
    ddof = 1 if len(node_rows) > 1 else 0
    task_name = figure_runtime.prepared.task.name
    return {
        "task": task_name,
        "node_std_mean": node_layer_per_graph.mean(axis=0),
        "node_std_std": node_layer_per_graph.std(axis=0, ddof=ddof),
        "relation_std_mean": relation_layer_per_graph.mean(axis=0),
        "relation_std_std": relation_layer_per_graph.std(axis=0, ddof=ddof),
        "log_r_mean": ratio_per_graph.mean(axis=0),
        "log_r_std": ratio_per_graph.std(axis=0, ddof=ddof),
        "attention_entropy_mean": entropy_per_graph.mean(axis=0),
        "attention_entropy_std": entropy_per_graph.std(axis=0, ddof=ddof),
        "node_per_graph": node_per_graph,
        "relation_per_graph": relation_per_graph,
        "attention_entropy_per_graph": entropy_per_graph,
        "positions": np.asarray(positions, dtype=np.int64),
        "n_requested": int(n_graphs),
        "n_used": len(node_rows),
        "ratio": "log10(std(node-only counterfactual)/std(relation-conditioned actual))",
        "attention_entropy_definition": (
            "mean query entropy normalised by log(supported key count)"
        ),
        **figure_identity(task_name),
    }


class GritPerGraphCoordinateEstimator:
    """Replay canonical scoring for selected evaluation graphs only."""

    ESTIMATOR_VERSION = "canonical-graph-local-head-coordinates-v1"

    def __init__(
        self,
        figure_runtime: GritFigureRuntime,
        artifact: ReadOnlyCacheArtifact,
        model_record: Mapping[str, Any],
        *,
        output_dir: str | Path,
    ):
        self.figure_runtime = figure_runtime
        self.artifact = artifact
        self.model_record = dict(model_record)
        self.output_dir = Path(output_dir)
        self.contract = dict(artifact.metadata["contract"])
        self.original_splits = _split_manifest_from_record(self.model_record)
        if self.original_splits.fingerprint != self.contract["split_fingerprint"]:
            raise StaleCacheError(
                "canonical model record and score cache use different split manifests"
            )

    def estimate(self, graph_indices: Sequence[int]) -> dict[int, dict[str, Any]]:
        indices = tuple(dict.fromkeys(int(value) for value in graph_indices))
        if not indices:
            raise ValueError("at least one evaluation graph index is required")
        runtime = self.figure_runtime.runtime
        invalid = [
            value for value in indices if value < 0 or value >= len(runtime.eval_ds)
        ]
        if invalid:
            raise IndexError(f"GRIT evaluation graph indices are out of range: {invalid}")
        if self.original_splits.same_index_space and set(indices).intersection(
            self.original_splits.semantic_donor_pool
        ):
            raise ValueError(
                "displayed graph indices overlap the canonical semantic donor pool "
                "in a shared eval/donor index space"
            )
        splits = SplitManifest(
            discovery=indices,
            causal=(),
            clean_ablation=(),
            semantic_donor_pool=self.original_splits.semantic_donor_pool,
            same_index_space=self.original_splits.same_index_space,
            seed=int(self.original_splits.seed),
        )
        prepared = dataclasses.replace(
            self.figure_runtime.prepared,
            output_dir=self.output_dir,
            splits=splits,
            progress=None,
        )
        original = self.figure_runtime.protocol_config
        sizes = RunSizes(
            discovery_graphs=len(indices),
            causal_graphs=1,
            clean_ablation_graphs=1,
            semantic_donor_graphs=len(self.original_splits.semantic_donor_pool),
            sources_per_graph=int(self.contract["source_cap"]),
            donors_per_source=int(self.contract["donors_per_source"]),
        )
        config = dataclasses.replace(
            original,
            output_dir=str(self.output_dir),
            tasks=(prepared.task.name,),
            train_seeds=(int(self.contract["train_seed"]),),
            task_train_seeds={},
            phases=("scores",),
            sizes=sizes,
            execution=dataclasses.replace(
                original.execution, graphs_per_batch=1
            ),
            resume=False,
            force=False,
        )
        estimate = estimate_graph_local_head_coordinates(prepared, config)
        output: dict[int, dict[str, Any]] = {}
        for graph_index in indices:
            value = dict(estimate["graphs"][graph_index])
            value["graph_index"] = int(graph_index)
            value["estimation"] = {
                "estimator": self.ESTIMATOR_VERSION,
                "canonical_source_protocol": self.artifact.metadata[
                    "protocol_version"
                ],
                "implementation_protocol": estimate["protocol_version"],
                "manifest_hash": estimate["manifest_hash"],
                "graph_id_seed_space": "GRIT evaluation-split position",
                "analysis_seed": int(estimate["analysis_seed"]),
                "sources_per_graph": int(estimate["sources_per_graph"]),
                "donors_per_source": int(estimate["donors_per_source"]),
                "semantic_donor_pool_size": len(
                    self.original_splits.semantic_donor_pool
                ),
            }
            output[graph_index] = value
        return output


__all__ = [
    "CanonicalHeadMetrics",
    "GritDiagnosticCapture",
    "GritDiagnosticExtractor",
    "GritFigureRuntime",
    "GritPerGraphCoordinateEstimator",
    "Head",
    "QM9_FIGURE_TASKS",
    "SELECTED_HEAD_TRANSPORT_PROFILE_VERSION",
    "SUPPORTED_GRIT_FIGURE_TASKS",
    "SupplementalCache",
    "ZINC_FIGURE_TASKS",
    "aggregate_logit_spread",
    "build_verified_grit_figure_runtime",
    "collect_attention_examples",
    "chemistry_focus_version",
    "compute_av_pca_inputs",
    "compute_av_pca_inputs_many",
    "compute_layer_av_pca_inputs",
    "compute_selected_head_transport_profiles",
    "graph_node_labels",
    "label_attention_focus",
    "load_canonical_model_record",
    "load_canonical_score_artifact",
    "methodology_config_for_artifact",
    "methodology_config_from_record",
    "molecular_task_family",
    "resolve_canonical_task_root",
    "select_attention_grid_indices",
    "select_ranked_heads",
    "select_specialist_heads",
    "select_structural_specialist_head",
]
