"""Canonical score adapters and additive Graphormer figure diagnostics.

The canonical methodology owns the score definitions and their cache contract.  This
module never replaces or relabels the canonical aggregate ``scores/raw.pt`` result. It
adapts that immutable result for reporting and computes explicitly additive diagnostics:
selected attention, pooled ``A@V`` vectors, dot/bias logit spread, and graph-local
``D_rel``/``J`` estimates for displayed molecules. The latter call the same central
event/scoring implementation as canonical runs and are stored only in supplemental,
contract-keyed caches.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .bootstrap import Observation, nested_percentile_interval
from .cache import (
    ReadOnlyCacheArtifact,
    StaleCacheError,
    load_cache_artifact_file,
)
from .graphormer import (
    GraphormerBackend,
    GraphormerGraph,
    GraphormerRuntime,
    PCQMGraphormerDataset,
    build_graphormer_runtime,
)
from .protocol import (
    BootstrapPolicy,
    ExecutionPolicy,
    MethodologyConfig,
    RunSizes,
    SplitManifest,
    stable_hash,
)
from .runner import PreparedTask, estimate_graph_local_head_coordinates
from .sampling import SemanticDonorPool
from .tasks import get_task


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


def load_graphormer_score_artifact(path: str | Path) -> ReadOnlyCacheArtifact:
    """Load a validated Graphormer score cache for additive figures.

    This compatibility boundary is intentionally narrower than the canonical
    analysis loader: it is read-only and requires the stored task and runtime
    contract to describe the official PCQM4Mv2 Graphormer task.
    """

    artifact = load_cache_artifact_file(path)
    contract = artifact.metadata.get("contract")
    if not isinstance(contract, Mapping):
        raise StaleCacheError(
            f"Graphormer score cache has no valid contract: {artifact.path}"
        )
    if contract.get("task") != "graphormer_pcqm4mv2":
        raise StaleCacheError(
            f"{artifact.path} is for task {contract.get('task')!r}, not "
            "'graphormer_pcqm4mv2'"
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
            f"Graphormer score cache contract is missing {missing}: {artifact.path}"
        )
    return artifact


def load_graphormer_model_record(
    path: str | Path,
    artifact: ReadOnlyCacheArtifact,
) -> dict[str, Any]:
    """Load and bind ``model.json`` to the immutable canonical score artifact."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"canonical Graphormer model record not found: {resolved}"
        )
    record = json.loads(resolved.read_text())
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
            f"{resolved} does not describe the canonical score artifact: {mismatches}"
        )
    splits = _split_manifest_from_record(record)
    if splits.fingerprint != contract["split_fingerprint"]:
        raise StaleCacheError(
            f"{resolved} split fingerprint {splits.fingerprint!r} does not match "
            f"the canonical score cache {contract['split_fingerprint']!r}"
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
                f"{channel} raw scores have shape {raw.shape}; expected {metrics.shape}"
            )
        scale = float(np.mean(raw))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{channel} within-model normalisation is not positive")

        graph_contribution = channel_scores.get("graph_distance_contribution")
        if not isinstance(graph_contribution, Mapping) or not graph_contribution:
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
                    f"{channel} graph {graph_id} distance contribution has shape "
                    f"{contribution.shape}; expected {expected}"
                )
            graph_rows.append(
                np.stack([contribution[layer, head] for layer, head in selected])
            )
        point = np.stack(graph_rows).mean(axis=0)
        selected_raw = np.asarray(
            [raw[layer, head] for layer, head in selected], dtype=np.float64
        )
        reconstructed = point.sum(axis=-1)
        if not np.allclose(
            reconstructed,
            selected_raw,
            rtol=2e-5,
            atol=1e-8,
            equal_nan=True,
        ):
            residual = float(np.nanmax(np.abs(reconstructed - selected_raw)))
            raise ValueError(
                f"{channel} carrier-distance profiles do not reconstruct selected "
                f"head scores (max residual {residual:.3e})"
            )

        support = channel_scores.get("distance_support")
        if not isinstance(support, Mapping) or "reportable" not in support:
            raise ValueError(
                f"{channel} score cache has no registered distance-reporting mask"
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


def select_specialist_heads(
    metrics: CanonicalHeadMetrics,
    *,
    semantic_head: Head = (1, 24),
    structural_head: Head | None = None,
    active_only: bool = True,
) -> dict[str, Head]:
    """Keep L1 H24 semantic and select the smallest active-head ``D_rel``.

    The structural tie-break is decreasing ``J`` so an exact selectivity tie keeps
    the more causally engaged head.
    """

    semantic_head = validate_head(semantic_head, metrics.shape)
    if structural_head is not None:
        structural_head = validate_head(structural_head, metrics.shape)
    else:
        structural_head = select_structural_specialist_head(
            metrics,
            active_only=active_only,
            excluded_heads=(semantic_head,),
        )
    return {"semantic": semantic_head, "structural": structural_head}


def select_structural_specialist_head(
    metrics: CanonicalHeadMetrics,
    *,
    active_only: bool = True,
    excluded_heads: Sequence[Head] = (),
    excluded_head_indices: Sequence[int] = (),
) -> Head:
    """Select the smallest-``D_rel`` head after explicit exclusions."""

    finite = np.isfinite(metrics.selectivity) & np.isfinite(
        metrics.joint_sensitivity
    )
    eligible = finite & metrics.active if active_only else finite
    eligible = eligible.copy()
    for head in excluded_heads:
        eligible[validate_head(head, metrics.shape)] = False
    for head_index in excluded_head_indices:
        head_index = int(head_index)
        if not 0 <= head_index < metrics.num_heads:
            raise IndexError(
                f"excluded head index H{head_index} is outside [0, {metrics.num_heads})"
            )
        eligible[:, head_index] = False
    layers, heads = np.where(eligible)
    if not len(layers):
        raise ValueError("no eligible head is available for structural selection")
    selectivity = metrics.selectivity[layers, heads]
    joint = metrics.joint_sensitivity[layers, heads]
    order = np.lexsort((-joint, selectivity))
    return int(layers[order[0]]), int(heads[order[0]])


def select_attention_grid_indices(
    entries: Sequence[int],
    *,
    num_rows: int,
) -> list[int]:
    """Validate a role's configured indices and return its first figure rows."""

    num_rows = int(num_rows)
    if num_rows < 1:
        raise ValueError("attention-grid num_rows must be positive")
    indices = [int(value) for value in entries]
    if len(indices) < num_rows:
        raise ValueError(
            f"attention grid needs {num_rows} graph indices, but only "
            f"{len(indices)} were configured"
        )
    selected = indices[:num_rows]
    if len(set(selected)) != len(selected):
        raise ValueError(
            "attention-grid graph indices must be unique within the displayed rows"
        )
    return selected


def select_attention_examples_for_grid(
    payload: Mapping[str, Any],
    *,
    graph_indices: Sequence[int],
) -> dict[str, Any]:
    """Return examples in the exact configured order and row count.

    The supplemental-cache contract normally prevents a payload for one graph
    selection from being reused for another. This additional boundary check
    ensures a stale or manually copied cache can never silently control the
    rendered rows.
    """

    expected = [int(value) for value in graph_indices]
    if not expected:
        raise ValueError("attention-grid graph indices must not be empty")
    if len(set(expected)) != len(expected):
        raise ValueError("attention-grid graph indices must be unique")
    examples = list(payload.get("examples", ()))
    by_index: dict[int, Mapping[str, Any]] = {}
    for example in examples:
        graph_index = int(example["dataset_index"])
        if graph_index in by_index:
            raise ValueError(
                f"attention payload contains duplicate PCQM index {graph_index}"
            )
        by_index[graph_index] = example
    missing = [index for index in expected if index not in by_index]
    if missing:
        raise ValueError(
            "attention payload does not contain the configured PCQM indices "
            f"{missing}; cached indices are {sorted(by_index)}"
        )
    selected = dict(payload)
    selected["examples"] = [by_index[index] for index in expected]
    return selected


def select_ranked_heads(
    metrics: CanonicalHeadMetrics,
    *,
    semantic_count: int = 2,
    joint_count: int = 2,
    active_only: bool = True,
    joint_generalist_max_abs_selectivity: float | None = None,
) -> dict[str, Head]:
    """Rank heads independently by decreasing ``D_rel`` and decreasing ``J``.

    A head may appear in both rankings. Exact ``D_rel`` ties prefer larger
    ``J``; exact ``J`` ties prefer larger ``D_rel``. When a generalist bound is
    supplied, the ``J`` ranking is restricted to ``|D_rel|`` below that bound.
    """

    semantic_count = int(semantic_count)
    joint_count = int(joint_count)
    if semantic_count < 0 or joint_count < 0:
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
    for rank, position in enumerate(joint_order[:joint_count], start=1):
        ranked[f"top_joint_{rank}"] = (
            int(joint_layers[position]),
            int(joint_heads[position]),
        )
    return ranked


class SupplementalCache:
    """Immutable contract-keyed storage for non-canonical diagnostics."""

    SCHEMA_VERSION = "graphormer_pcqm_figure_diagnostics.v1"

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
        self, name: str, contract: Mapping[str, Any]
    ) -> tuple[Any, Path] | None:
        """Read a matching supplemental result without invoking model code."""

        path = self.path_for(name, contract)
        if not path.is_file():
            return None
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError) as error:
            print(
                f"[supplemental-cache:warning] unreadable {path}; "
                f"treating it as a cache miss ({type(error).__name__}: {error})"
            )
            return None
        if not isinstance(payload, Mapping):
            return None
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            return None
        if payload.get("contract_fingerprint") != stable_hash(dict(contract)):
            return None
        return payload["value"], path

    def load_or_compute(
        self,
        name: str,
        contract: Mapping[str, Any],
        compute: Callable[[], Any],
        *,
        force: bool = False,
    ) -> tuple[Any, Path, bool]:
        """Return ``(value, path, cache_hit)`` for an additive diagnostic."""

        path = self.path_for(name, contract)
        if not force:
            cached = self.load(name, contract)
            if cached is not None:
                return cached[0], cached[1], True

        value = compute()
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "contract": dict(contract),
            "contract_fingerprint": stable_hash(dict(contract)),
            "value": value,
        }
        self._atomic_torch_save(payload, path)
        return value, path, False

    @staticmethod
    def _atomic_torch_save(payload: Any, path: Path) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".partial", dir=path.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class GraphormerFigureRuntime:
    """Official runtime verified against the canonical score cache contract."""

    runtime: Any
    backend: GraphormerBackend
    checkpoint_descriptor: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class GraphormerCausalRuntime:
    """Verified official runtime plus the canonical event-replay context."""

    prepared: PreparedTask
    protocol_config: MethodologyConfig


def build_verified_figure_runtime(
    artifact: ReadOnlyCacheArtifact,
    *,
    dataset_root: str,
    accelerator: str = "cuda:0",
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> GraphormerFigureRuntime:
    """Load PCQM Graphormer and fail if it differs from the score artifact."""

    contract = artifact.metadata["contract"]
    if contract.get("task") != "graphormer_pcqm4mv2":
        raise ValueError(
            "figure runtime requires a graphormer_pcqm4mv2 score artifact; "
            f"got {contract.get('task')!r}"
        )
    task = get_task("graphormer_pcqm4mv2")
    if contract.get("task_adapter_version") != task.adapter_version:
        raise ValueError(
            "canonical cache adapter version differs from the current Graphormer task"
        )
    runtime, descriptor, digest = build_graphormer_runtime(
        task,
        checkpoint=None,
        train_seed=int(contract["train_seed"]),
        accelerator=str(accelerator),
        overrides={
            "dataset_root": str(dataset_root),
            "cache_dir": cache_dir,
            "local_files_only": bool(local_files_only),
        },
    )
    expected_digest = str(contract["checkpoint_sha256"])
    if digest != expected_digest:
        raise ValueError(
            "loaded Graphormer checkpoint does not match canonical score cache: "
            f"{digest} != {expected_digest}"
        )
    backend = GraphormerBackend(runtime, task, sigma=contract["sigma"])
    expected_geometry = {
        str(key): int(value) for key, value in contract["model_geometry"].items()
    }
    if backend.geometry != expected_geometry:
        raise ValueError(
            f"loaded Graphormer geometry {backend.geometry} != "
            f"canonical {expected_geometry}"
        )
    return GraphormerFigureRuntime(runtime, backend, descriptor, digest)


def build_verified_causal_runtime(
    artifact: ReadOnlyCacheArtifact,
    model_record: Mapping[str, Any],
    *,
    dataset_root: str,
    output_dir: str | Path,
    sources_per_graph: int,
    donors_per_source: int,
    accelerator: str = "cuda:0",
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> GraphormerCausalRuntime:
    """Rebuild the score-bound PCQM runtime for held-out causal event replay.

    The score artifact supplies the checkpoint, adapter, geometry, scale, and
    sampling budget. ``model.json`` supplies the immutable split manifest and its
    analysis seed. Only the requested causal source/donor caps may be smaller.
    """

    figure_runtime = build_verified_figure_runtime(
        artifact,
        dataset_root=str(dataset_root),
        accelerator=str(accelerator),
        cache_dir=cache_dir,
        local_files_only=bool(local_files_only),
    )
    contract = artifact.metadata["contract"]
    splits = _split_manifest_from_record(model_record)
    if splits.fingerprint != contract["split_fingerprint"]:
        raise StaleCacheError(
            "canonical Graphormer model record and score cache use different splits"
        )
    requested_sources = int(sources_per_graph)
    requested_donors = int(donors_per_source)
    if requested_sources > int(contract["source_cap"]):
        raise ValueError(
            "causal spatial support cannot exceed the score-bound source cap: "
            f"{requested_sources} > {contract['source_cap']}"
        )
    if requested_donors > int(contract["donors_per_source"]):
        raise ValueError(
            "causal spatial support cannot exceed the score-bound donor cap: "
            f"{requested_donors} > {contract['donors_per_source']}"
        )

    task = get_task("graphormer_pcqm4mv2")
    runtime = figure_runtime.runtime
    donor_pool = SemanticDonorPool(
        [
            (graph_id, runtime.donor_ds[graph_id])
            for graph_id in splits.semantic_donor_pool
        ],
        adapter=task.content_adapter,
    )
    sizes = RunSizes(
        discovery_graphs=max(1, len(splits.discovery)),
        causal_graphs=max(1, len(splits.causal)),
        clean_ablation_graphs=max(1, len(splits.clean_ablation)),
        semantic_donor_graphs=max(1, len(splits.semantic_donor_pool)),
        sources_per_graph=requested_sources,
        donors_per_source=requested_donors,
    )
    protocol_config = MethodologyConfig(
        output_dir=str(output_dir),
        tasks=(task.name,),
        train_seeds=(int(contract["train_seed"]),),
        phases=("causal",),
        sizes=sizes,
        execution=ExecutionPolicy(graphs_per_batch=1),
        analysis_seed=int(splits.seed),
        accelerator=str(accelerator),
        resume=False,
        force=False,
    )
    prepared = PreparedTask(
        task=task,
        runtime=runtime,
        backend=figure_runtime.backend,
        output_dir=Path(output_dir),
        checkpoint=Path("official-graphormer-pcqm4mv2"),
        checkpoint_sha=str(contract["checkpoint_sha256"]),
        sigma=np.asarray(contract["sigma"], dtype=np.float64),
        splits=splits,
        donor_pool=donor_pool,
        progress=None,
    )
    return GraphormerCausalRuntime(prepared, protocol_config)


class _GlobalIndexPCQMDataset:
    """Sparse view whose keys are global PCQM4Mv2 dataset indices."""

    def __init__(self, source: PCQMGraphormerDataset, allowed: Sequence[int]):
        self.dataset = source.dataset
        self.config = source.config
        self.allowed = frozenset(int(value) for value in allowed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, graph_index: int) -> GraphormerGraph:
        graph_index = int(graph_index)
        if graph_index not in self.allowed:
            raise IndexError(
                f"global PCQM graph {graph_index} is outside this sparse estimate"
            )
        return PCQMGraphormerDataset(
            self.dataset,
            (graph_index,),
            self.config,
        )[0]


class GraphormerPerGraphCoordinateEstimator:
    """Estimate graph-local canonical head coordinates for selected molecules.

    The original canonical semantic donor pool and sampling budgets are replayed,
    while the supplied global PCQM index is also the graph ID used by deterministic
    source/donor RNG seeding. No canonical cache file is mutated.
    """

    ESTIMATOR_VERSION = "canonical-graph-local-head-coordinates-v1"

    def __init__(
        self,
        figure_runtime: GraphormerFigureRuntime,
        artifact: ReadOnlyCacheArtifact,
        model_record: Mapping[str, Any],
        *,
        output_dir: str | Path,
    ):
        self.figure_runtime = figure_runtime
        self.artifact = artifact
        self.model_record = dict(model_record)
        self.output_dir = Path(output_dir)
        self.task = get_task("graphormer_pcqm4mv2")
        self.contract = dict(artifact.metadata["contract"])
        self.original_splits = _split_manifest_from_record(self.model_record)
        if self.original_splits.fingerprint != self.contract["split_fingerprint"]:
            raise StaleCacheError(
                "canonical model record and score cache use different split manifests"
            )
        required = {"source_cap", "donors_per_source", "sigma"}
        missing = sorted(required.difference(self.contract))
        if missing:
            raise StaleCacheError(
                f"graph-local estimates require canonical contract fields {missing}"
            )
        self._donor_pool: SemanticDonorPool | None = None

    def _semantic_donor_pool(self) -> SemanticDonorPool:
        if self._donor_pool is None:
            runtime = self.figure_runtime.runtime
            self._donor_pool = SemanticDonorPool(
                [
                    (graph_id, runtime.donor_ds[graph_id])
                    for graph_id in self.original_splits.semantic_donor_pool
                ],
                adapter=self.task.content_adapter,
            )
        return self._donor_pool

    def estimate(self, graph_indices: Sequence[int]) -> dict[int, dict[str, Any]]:
        indices = tuple(dict.fromkeys(int(value) for value in graph_indices))
        if not indices:
            raise ValueError("at least one global PCQM graph index is required")
        source = self.figure_runtime.runtime.eval_ds
        source_dataset = getattr(source, "dataset", None)
        if source_dataset is None or not hasattr(source, "config"):
            raise TypeError(
                "graph-local estimates require a PCQMGraphormerDataset-backed runtime"
            )
        invalid = [
            value for value in indices if value < 0 or value >= len(source_dataset)
        ]
        if invalid:
            raise IndexError(f"global PCQM graph indices are out of range: {invalid}")

        sparse_eval = _GlobalIndexPCQMDataset(source, indices)
        original_runtime = self.figure_runtime.runtime
        runtime = GraphormerRuntime(
            original_runtime.model,
            sparse_eval,
            original_runtime.donor_ds,
            device=original_runtime.device,
            seed=int(self.contract["train_seed"]),
            metric_fn=self.task.metric_fn,
        )
        sigma = np.asarray(self.contract["sigma"], dtype=np.float64)
        backend = GraphormerBackend(runtime, self.task, sigma=sigma)
        splits = SplitManifest(
            discovery=indices,
            causal=(),
            clean_ablation=(),
            semantic_donor_pool=self.original_splits.semantic_donor_pool,
            same_index_space=False,
            seed=int(self.original_splits.seed),
        )
        prepared = PreparedTask(
            task=self.task,
            runtime=runtime,
            backend=backend,
            output_dir=self.output_dir,
            checkpoint=Path("official-graphormer-pcqm4mv2"),
            checkpoint_sha=str(self.contract["checkpoint_sha256"]),
            sigma=sigma,
            splits=splits,
            donor_pool=self._semantic_donor_pool(),
            progress=None,
        )
        sizes = RunSizes(
            discovery_graphs=len(indices),
            causal_graphs=1,
            clean_ablation_graphs=1,
            semantic_donor_graphs=len(self.original_splits.semantic_donor_pool),
            sources_per_graph=int(self.contract["source_cap"]),
            donors_per_source=int(self.contract["donors_per_source"]),
        )
        config = MethodologyConfig(
            output_dir=str(self.output_dir),
            tasks=(self.task.name,),
            train_seeds=(int(self.contract["train_seed"]),),
            phases=("scores",),
            sizes=sizes,
            execution=ExecutionPolicy(graphs_per_batch=1),
            analysis_seed=int(self.original_splits.seed),
            accelerator=str(original_runtime.device),
            resume=False,
            force=False,
        )
        estimate = estimate_graph_local_head_coordinates(prepared, config)
        output: dict[int, dict[str, Any]] = {}
        for graph_index in indices:
            graph_value = dict(estimate["graphs"][graph_index])
            graph_value["graph_index"] = int(graph_index)
            graph_value["estimation"] = {
                "estimator": self.ESTIMATOR_VERSION,
                "canonical_source_protocol": self.artifact.metadata[
                    "protocol_version"
                ],
                "implementation_protocol": estimate["protocol_version"],
                "manifest_hash": estimate["manifest_hash"],
                "graph_id_seed_space": "global PCQM4Mv2 dataset index",
                "analysis_seed": int(estimate["analysis_seed"]),
                "sources_per_graph": int(estimate["sources_per_graph"]),
                "donors_per_source": int(estimate["donors_per_source"]),
                "semantic_donor_pool_size": len(
                    self.original_splits.semantic_donor_pool
                ),
            }
            output[graph_index] = graph_value
        return output


@dataclass(frozen=True)
class GraphormerDiagnosticCapture:
    """One-graph tensors at Graphormer's exact attention computation sites."""

    dot: tuple[Any, ...]
    bias: tuple[Any, ...]
    attention: tuple[Any, ...]
    transport: tuple[Any, ...]


class GraphormerDiagnosticExtractor:
    """Capture ``d``, ``b``, softmax attention, and per-head ``A@V`` in one pass."""

    def __init__(self, backend: GraphormerBackend):
        self.backend = backend
        self.model = backend.model

    def extract(self, data: GraphormerGraph) -> GraphormerDiagnosticCapture:
        import torch

        layers = self.model.encoder.graph_encoder.layers
        dot: list[Any | None] = [None] * len(layers)
        bias: list[Any | None] = [None] * len(layers)
        attention: list[Any | None] = [None] * len(layers)
        transport: list[Any | None] = [None] * len(layers)
        handles = []

        def attention_input_hook(layer_index: int):
            def hook(module, args, kwargs):
                query = kwargs.get("query", args[0] if args else None)
                attention_bias = kwargs.get(
                    "attn_bias", args[3] if len(args) > 3 else None
                )
                if query is None or attention_bias is None:
                    raise RuntimeError("Graphormer diagnostic hook missed query/bias")
                tokens, batch, width = query.shape
                heads = int(module.num_heads)
                head_width = width // heads
                q = module.q_proj(query) * module.scaling
                k = module.k_proj(query)
                q = q.view(tokens, batch, heads, head_width).permute(1, 2, 0, 3)
                k = k.view(tokens, batch, heads, head_width).permute(1, 2, 0, 3)
                dot[layer_index] = torch.matmul(q, k.transpose(-1, -2)).detach()
                bias[layer_index] = attention_bias.view(
                    batch, heads, tokens, tokens
                ).detach()

            return hook

        def attention_output_hook(layer_index: int):
            def hook(module, args, output):
                del module, args
                heads = int(layers[layer_index].self_attn.num_heads)
                batch = int(output.shape[0]) // heads
                attention[layer_index] = output.view(
                    batch, heads, output.shape[-2], output.shape[-1]
                ).detach()

            return hook

        def transport_hook(layer_index: int):
            def hook(module, args):
                del module
                value = args[0]
                tokens, batch, width = value.shape
                heads = int(layers[layer_index].self_attn.num_heads)
                transport[layer_index] = (
                    value.view(tokens, batch, heads, width // heads)
                    .permute(1, 2, 0, 3)
                    .detach()
                )

            return hook

        for layer_index, layer in enumerate(layers):
            handles.append(
                layer.self_attn.register_forward_pre_hook(
                    attention_input_hook(layer_index), with_kwargs=True
                )
            )
            handles.append(
                layer.self_attn.attention_dropout_module.register_forward_hook(
                    attention_output_hook(layer_index)
                )
            )
            handles.append(
                layer.self_attn.out_proj.register_forward_pre_hook(
                    transport_hook(layer_index)
                )
            )
        try:
            inputs, _ = self.backend._batch([data])
            with torch.no_grad():
                self.model(**inputs, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        groups = (dot, bias, attention, transport)
        if any(any(value is None for value in group) for group in groups):
            raise RuntimeError("one or more Graphormer diagnostic hooks did not fire")
        tokens = int(data.num_nodes) + 1
        return GraphormerDiagnosticCapture(
            dot=tuple(value[0, :, :tokens, :tokens] for value in dot),
            bias=tuple(value[0, :, :tokens, :tokens] for value in bias),
            attention=tuple(
                value[0, :, :tokens, :tokens] for value in attention
            ),
            transport=tuple(value[0, :, :tokens, :] for value in transport),
        )


def dataset_index(runtime: Any, position: int) -> int:
    indices = getattr(runtime.eval_ds, "indices", None)
    return int(indices[position]) if indices is not None else int(position)


def graph_at_dataset_index(runtime: Any, index: int) -> GraphormerGraph:
    """Build one graph by its global PCQM4Mv2 dataset index."""

    eval_dataset = runtime.eval_ds
    source_dataset = getattr(eval_dataset, "dataset", None)
    config = getattr(eval_dataset, "config", None)
    if source_dataset is None or config is None:
        raise TypeError(
            "global PCQM graph indices require a PCQMGraphormerDataset-backed runtime"
        )
    direct_view = PCQMGraphormerDataset(source_dataset, (int(index),), config)
    return direct_view[0]


def collect_attention_examples(
    figure_runtime: GraphormerFigureRuntime,
    *,
    graph_indices: Sequence[int],
    heads: Mapping[str, Head],
) -> dict[str, Any]:
    extractor = GraphormerDiagnosticExtractor(figure_runtime.backend)
    examples = []
    for graph_index in graph_indices:
        graph = graph_at_dataset_index(figure_runtime.runtime, int(graph_index))
        captured = extractor.extract(graph)
        selected = {
            role: captured.attention[layer][head].float().cpu().numpy()
            for role, (layer, head) in heads.items()
        }
        examples.append(
            {
                "dataset_index": int(graph_index),
                "smiles": str(graph.smiles),
                "n_atoms": int(graph.num_nodes),
                "attention": selected,
            }
        )
    return {"heads": dict(heads), "examples": examples}


def _atom_categories(molecule) -> list[str]:
    """Assign the chemistry-focus categories used by the historical PCA recipe."""

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
        if symbol == "O":
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
                neighbour.GetSymbol() == "O" for neighbour in atom.GetNeighbors()
            ) >= 2:
                categories[index] = "N: nitro"
            elif any(is_carbonyl_carbon(neighbour) for neighbour in atom.GetNeighbors()):
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


def _attention_focus_categories(smiles: str) -> list[str]:
    """Parse one molecule and return its stable per-atom focus categories."""

    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")
    return _atom_categories(molecule)


def _label_attention_focus_from_categories(
    categories: Sequence[str],
    atom_mass: np.ndarray,
    *,
    focus_mass: float = 0.75,
    diffuse_threshold: float = 0.35,
) -> str:
    atom_mass = np.asarray(atom_mass, dtype=np.float64)
    if atom_mass.ndim != 1 or len(atom_mass) != len(categories):
        raise ValueError(
            "attention mass and atom categories must be aligned one-dimensional "
            f"arrays, got {atom_mass.shape} and {len(categories)} categories"
        )
    order = np.argsort(-atom_mass)
    selected = []
    cumulative = 0.0
    for index in order:
        selected.append(int(index))
        cumulative += float(atom_mass[index])
        if cumulative >= float(focus_mass):
            break
    mass_by_category: dict[str, float] = {}
    for index in selected:
        category = categories[index]
        mass_by_category[category] = mass_by_category.get(category, 0.0) + float(
            atom_mass[index]
        )
    best, mass = max(mass_by_category.items(), key=lambda item: item[1])
    return best if mass >= float(diffuse_threshold) else "other/diffuse"


def label_attention_focus(
    smiles: str,
    atom_mass: np.ndarray,
    *,
    focus_mass: float = 0.75,
    diffuse_threshold: float = 0.35,
) -> str:
    return _label_attention_focus_from_categories(
        _attention_focus_categories(smiles),
        atom_mass,
        focus_mass=focus_mass,
        diffuse_threshold=diffuse_threshold,
    )


def compute_av_pca_inputs(
    figure_runtime: GraphormerFigureRuntime,
    *,
    head: Head = (1, 24),
    n_graphs: int = 500,
    focus_mass: float = 0.75,
    diffuse_threshold: float = 0.35,
    verbose: bool = True,
) -> dict[str, Any]:
    layer, head_index = head
    extractor = GraphormerDiagnosticExtractor(figure_runtime.backend)
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    positions: list[int] = []
    indices: list[int] = []
    limit = min(int(n_graphs), len(figure_runtime.runtime.eval_ds))
    for position in range(limit):
        try:
            graph = figure_runtime.runtime.eval_ds[position]
            captured = extractor.extract(graph)
            vector = (
                captured.transport[layer][head_index, 1:, :]
                .mean(dim=0)
                .float()
                .cpu()
                .numpy()
            )
            matrix = (
                captured.attention[layer][head_index, 1:, 1:]
                .float()
                .cpu()
                .numpy()
            )
            matrix = matrix / np.clip(
                matrix.sum(axis=-1, keepdims=True), 1e-12, None
            )
            inbound = matrix.mean(axis=0)
            label = label_attention_focus(
                graph.smiles,
                inbound,
                focus_mass=focus_mass,
                diffuse_threshold=diffuse_threshold,
            )
        except Exception as error:
            if verbose:
                print(f"  [A@V PCA] skipped eval position {position}: {error}")
            continue
        vectors.append(vector)
        labels.append(label)
        positions.append(position)
        indices.append(dataset_index(figure_runtime.runtime, position))
        if verbose and (position + 1) % 50 == 0:
            print(f"  [A@V PCA] {position + 1}/{limit}")
    if len(vectors) < 2:
        raise RuntimeError("fewer than two molecules produced valid A@V vectors")
    return {
        "head": tuple(head),
        "vectors": np.stack(vectors),
        "labels": labels,
        "positions": np.asarray(positions, dtype=np.int64),
        "indices": np.asarray(indices, dtype=np.int64),
        "n_requested": int(n_graphs),
        "n_used": len(vectors),
    }


def compute_layer_av_pca_inputs(
    figure_runtime: GraphormerFigureRuntime,
    *,
    layers: Sequence[int],
    n_graphs: int = 500,
    focus_mass: float = 0.75,
    diffuse_threshold: float = 0.35,
    verbose: bool = True,
) -> dict[str, Any]:
    """Collect pooled ``A@V`` vectors for every head in selected layers.

    The diagnostic extractor captures all heads and layers in one model forward,
    so the requested layer grids share one sweep over the evaluation molecules.
    """

    model_layers = figure_runtime.backend.model.encoder.graph_encoder.layers
    num_layers = len(model_layers)
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

    num_heads = int(model_layers[requested_layers[0]].self_attn.num_heads)
    for layer in requested_layers:
        layer_heads = int(model_layers[layer].self_attn.num_heads)
        if layer_heads != num_heads:
            raise ValueError(
                "layer-wide PCA requires a constant head count; "
                f"layer {layer} has {layer_heads}, expected {num_heads}"
            )

    extractor = GraphormerDiagnosticExtractor(figure_runtime.backend)
    vectors: dict[int, list[np.ndarray]] = {
        layer: [] for layer in requested_layers
    }
    labels: dict[int, list[list[str]]] = {
        layer: [] for layer in requested_layers
    }
    positions: list[int] = []
    indices: list[int] = []
    limit = min(int(n_graphs), len(figure_runtime.runtime.eval_ds))
    for position in range(limit):
        try:
            graph = figure_runtime.runtime.eval_ds[position]
            captured = extractor.extract(graph)
            atom_categories = _attention_focus_categories(str(graph.smiles))
            graph_vectors: dict[int, np.ndarray] = {}
            graph_labels: dict[int, list[str]] = {}
            for layer in requested_layers:
                layer_vectors = (
                    captured.transport[layer][:, 1:, :]
                    .mean(dim=1)
                    .float()
                    .cpu()
                    .numpy()
                )
                matrices = (
                    captured.attention[layer][:, 1:, 1:]
                    .float()
                    .cpu()
                    .numpy()
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
                graph_vectors[layer] = layer_vectors
                graph_labels[layer] = [
                    _label_attention_focus_from_categories(
                        atom_categories,
                        inbound[head],
                        focus_mass=focus_mass,
                        diffuse_threshold=diffuse_threshold,
                    )
                    for head in range(num_heads)
                ]
        except Exception as error:
            if verbose:
                print(
                    f"  [layer A@V PCA] skipped eval position {position}: {error}"
                )
            continue
        for layer in requested_layers:
            vectors[layer].append(graph_vectors[layer])
            labels[layer].append(graph_labels[layer])
        positions.append(position)
        indices.append(dataset_index(figure_runtime.runtime, position))
        if verbose and (position + 1) % 50 == 0:
            print(f"  [layer A@V PCA] {position + 1}/{limit}")

    if len(positions) < 2:
        raise RuntimeError(
            "fewer than two molecules produced valid layer-wide A@V vectors"
        )
    return {
        "layers": {
            layer: {
                "layer": int(layer),
                "vectors": np.stack(vectors[layer]),
                "labels": labels[layer],
                "n_used": len(positions),
            }
            for layer in requested_layers
        },
        "requested_layers": requested_layers,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "positions": np.asarray(positions, dtype=np.int64),
        "indices": np.asarray(indices, dtype=np.int64),
        "n_requested": int(n_graphs),
        "n_used": len(positions),
    }


def aggregate_logit_spread(
    figure_runtime: GraphormerFigureRuntime,
    *,
    n_graphs: int = 100,
    verbose: bool = True,
) -> dict[str, Any]:
    """Aggregate key-wise ``std(d)`` and ``std(b)`` over validation molecules."""

    import torch

    extractor = GraphormerDiagnosticExtractor(figure_runtime.backend)
    dot_rows: list[np.ndarray] = []
    bias_rows: list[np.ndarray] = []
    positions: list[int] = []
    indices: list[int] = []
    limit = min(int(n_graphs), len(figure_runtime.runtime.eval_ds))
    for position in range(limit):
        try:
            graph = figure_runtime.runtime.eval_ds[position]
            captured = extractor.extract(graph)
            dot_spread = []
            bias_spread = []
            for dot, bias in zip(captured.dot, captured.bias):
                dot_nodes = dot[:, 1:, 1:].float()
                bias_nodes = bias[:, 1:, 1:].float()
                dot_spread.append(
                    dot_nodes.std(dim=-1, correction=1).mean(dim=-1)
                )
                bias_spread.append(
                    bias_nodes.std(dim=-1, correction=1).mean(dim=-1)
                )
            dot_rows.append(torch.stack(dot_spread).cpu().numpy())
            bias_rows.append(torch.stack(bias_spread).cpu().numpy())
        except Exception as error:
            if verbose:
                print(f"  [logit spread] skipped eval position {position}: {error}")
            continue
        positions.append(position)
        indices.append(dataset_index(figure_runtime.runtime, position))
        if verbose and (position + 1) % 10 == 0:
            print(f"  [logit spread] {position + 1}/{limit}")
    if not dot_rows:
        raise RuntimeError("no molecule produced valid Graphormer logit diagnostics")

    dot_per_graph = np.stack(dot_rows)
    bias_per_graph = np.stack(bias_rows)
    dot_layer_per_graph = dot_per_graph.mean(axis=-1)
    bias_layer_per_graph = bias_per_graph.mean(axis=-1)
    ratio_per_graph = np.log10(
        np.clip(
            bias_per_graph / np.clip(dot_per_graph, 1e-12, None),
            1e-12,
            None,
        )
    )
    ddof = 1 if len(dot_rows) > 1 else 0
    return {
        "dot_std_mean": dot_layer_per_graph.mean(axis=0),
        "dot_std_std": dot_layer_per_graph.std(axis=0, ddof=ddof),
        "bias_std_mean": bias_layer_per_graph.mean(axis=0),
        "bias_std_std": bias_layer_per_graph.std(axis=0, ddof=ddof),
        "log_r_mean": ratio_per_graph.mean(axis=0),
        "log_r_std": ratio_per_graph.std(axis=0, ddof=ddof),
        "dot_per_graph": dot_per_graph,
        "bias_per_graph": bias_per_graph,
        "positions": np.asarray(positions, dtype=np.int64),
        "indices": np.asarray(indices, dtype=np.int64),
        "n_requested": int(n_graphs),
        "n_used": len(dot_rows),
    }


__all__ = [
    "CanonicalHeadMetrics",
    "GraphormerCausalRuntime",
    "GraphormerDiagnosticCapture",
    "GraphormerDiagnosticExtractor",
    "GraphormerFigureRuntime",
    "GraphormerPerGraphCoordinateEstimator",
    "Head",
    "SELECTED_HEAD_TRANSPORT_PROFILE_VERSION",
    "SupplementalCache",
    "aggregate_logit_spread",
    "build_verified_causal_runtime",
    "build_verified_figure_runtime",
    "collect_attention_examples",
    "compute_av_pca_inputs",
    "compute_layer_av_pca_inputs",
    "compute_selected_head_transport_profiles",
    "graph_at_dataset_index",
    "label_attention_focus",
    "load_graphormer_model_record",
    "load_graphormer_score_artifact",
    "select_attention_grid_indices",
    "select_attention_examples_for_grid",
    "select_ranked_heads",
    "select_specialist_heads",
    "select_structural_specialist_head",
]
