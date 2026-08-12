"""Cache-only correction for the layer confound in head-ablation figures.

The primary estimand is the equal-weight mean of Spearman correlations computed
separately in every layer of a trained model.  Layer estimates are averaged within
model seed, then model-seed estimates are averaged with equal weight.  Molecular
confidence intervals resample held-out molecules and recompute that complete
estimand; permutation tests shuffle impacts only among heads in the same layer.

This module never imports a model implementation, dataset, checkpoint loader, GRIT,
or Graphormer.  It reads completed cache artifacts and writes derived tables and
figures into a separate correction directory.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .cache import StaleCacheError, atomic_json, load_cache_artifact_file
from .layer_controlled_ablation_figures import (
    MOLECULAR_FIGSIZE,
    PNG_DPI,
    SYNTHETIC_FIGSIZE,
    render_molecular_dissertation_panel,
    render_molecular_within_layer_rank_panel,
    render_synthetic_dissertation_panel,
    render_synthetic_within_layer_rank_panel,
)

CORRECTION_VERSION = "layer-controlled-head-ablation-v1"
DEFAULT_BOOTSTRAP_SEED = 17_071
DEFAULT_PERMUTATION_SEED = 72_019
DEFAULT_TASKS = (
    "synthetic",
    "graphormer_pcqm4mv2",
    "zinc",
    "qm9_gap_dense",
)
MOLECULAR_TASKS = ("graphormer_pcqm4mv2", "zinc", "qm9_gap_dense")
TASK_LABELS = {
    "synthetic": "Mixed synthetic task",
    "graphormer_pcqm4mv2": "PCQM4Mv2 Graphormer",
    "zinc": "ZINC GRIT",
    "qm9_gap_dense": "QM9 GRIT",
}
EXPECTED_GEOMETRY = {
    "synthetic": (3, 8),
    "graphormer_pcqm4mv2": (12, 32),
    "zinc": (10, 8),
    "qm9_gap_dense": (10, 8),
}
EXPECTED_DISSERTATION_GRAPHS = {
    "graphormer_pcqm4mv2": 128,
    "zinc": 64,
    "qm9_gap_dense": 64,
}
EXPECTED_DISSERTATION_POOLED_RHO = {
    "synthetic": 0.77,
    "graphormer_pcqm4mv2": 0.53,
    "zinc": 0.80,
    "qm9_gap_dense": 0.73,
}
EXPECTED_INDEPENDENT_CORRECTION = {
    # Appendix Figure B.2 independently applied within-(seed, layer) ranks to
    # the same mixed-task M1_DD score/ablation endpoint.
    "synthetic": 0.78,
}
SYNTHETIC_EXPERIMENT_VERSION = "causal-specialisation-double-dissociation-v2-shared-source-marker"
SYNTHETIC_ANALYSIS_VERSION = "causal-specialisation-matched-donor-swaps-v1"
SYNTHETIC_ANALYSIS_FINGERPRINT = "5a29d56ac5d21586"
PRESENTATION_PROVENANCE = {
    "synthetic": {
        "source_commit": "b89d9f5",
        "source_file": "figure_redesign/build_publication_colab.py",
        "source_functions": [
            "configure_matplotlib",
            "_dj_scatter",
            "figure_joint_selectivity_validation",
            "save_figure",
        ],
    },
    "molecular": {
        "source_branch_tip": "expansion/graphormer_specialisation@62ea073",
        "paper_adapter_commit": "cf53796",
        "renderer_style_commit": "80749a9",
        "source_files": [
            "src/graph_specialisation_metrics/methodology/paper_causal_figures.py",
            "src/graph_specialisation_metrics/methodology/graphbench_population_figures.py",
            "src/graph_specialisation_metrics/methodology/figures.py",
        ],
        "source_functions": [
            "population_figure_theme",
            "_plot_head_scatter",
            "FigureBuilder.save",
        ],
    },
}


class CacheDiscoveryError(RuntimeError):
    """Raised when an exact dissertation cache cannot be selected unambiguously."""


@dataclass(frozen=True)
class CacheCandidate:
    task: str
    family: str
    root: Path
    primary_paths: tuple[Path, ...]
    exact_dissertation: bool
    expected_graphs: int | None
    priority: int
    origin: str

    def as_dict(self) -> dict[str, Any]:
        shown_paths = list(self.primary_paths)
        omitted = max(0, len(shown_paths) - 8)
        if omitted:
            shown_paths = shown_paths[:8]
        return {
            "task": self.task,
            "family": self.family,
            "root": str(self.root),
            "primary_paths": [str(path) for path in shown_paths],
            "primary_path_count": len(self.primary_paths),
            "omitted_primary_paths": omitted,
            "exact_dissertation": bool(self.exact_dissertation),
            "expected_graphs": self.expected_graphs,
            "priority": int(self.priority),
            "origin": self.origin,
        }


@dataclass
class HeadAblationData:
    """One trained model's complete all-head clean-ablation endpoint."""

    task: str
    seed: int
    joint_sensitivity: np.ndarray
    layers: np.ndarray
    head_ids: tuple[tuple[int, int], ...]
    ablation_impact: np.ndarray
    graph_ids: np.ndarray | None = None
    movement_by_graph: np.ndarray | None = None
    original_pooled_rho: float | None = None
    source_paths: tuple[Path, ...] = ()
    source_sha256: Mapping[str, str] = field(default_factory=dict)
    cache_contract: Mapping[str, Any] = field(default_factory=dict)
    source_family: str = "unknown"
    exact_dissertation: bool = True

    def validate(
        self,
        *,
        expected_geometry: tuple[int, int] | None = None,
        expected_graphs: int | None = None,
        strict: bool = True,
    ) -> HeadAblationData:
        self.joint_sensitivity = np.asarray(self.joint_sensitivity, dtype=np.float64).reshape(-1)
        self.layers = np.asarray(self.layers, dtype=np.int64).reshape(-1)
        self.ablation_impact = np.asarray(self.ablation_impact, dtype=np.float64).reshape(-1)
        count = len(self.joint_sensitivity)
        if not (count == len(self.layers) == len(self.head_ids) == len(self.ablation_impact)):
            raise ValueError(f"{self.task}/seed_{self.seed}: head arrays disagree in length")
        if count < 3 or not (
            np.isfinite(self.joint_sensitivity).all() and np.isfinite(self.ablation_impact).all()
        ):
            raise ValueError(f"{self.task}/seed_{self.seed}: head values must be finite")
        if len(set(self.head_ids)) != count:
            raise ValueError(f"{self.task}/seed_{self.seed}: duplicate head identities")
        expected_layers = sorted({int(value) for value in self.layers})
        if expected_layers != list(range(len(expected_layers))):
            raise ValueError(f"{self.task}/seed_{self.seed}: layers must be consecutive from zero")
        layer_counts = [int(np.sum(self.layers == layer)) for layer in expected_layers]
        if len(set(layer_counts)) != 1:
            raise ValueError(
                f"{self.task}/seed_{self.seed}: layers have unequal head counts {layer_counts}"
            )
        if expected_geometry is not None:
            observed = (len(expected_layers), layer_counts[0])
            if observed != tuple(expected_geometry):
                raise ValueError(
                    f"{self.task}/seed_{self.seed}: observed geometry {observed}; "
                    f"expected {tuple(expected_geometry)}"
                )
        expected_ids = {
            (layer, head) for layer in expected_layers for head in range(layer_counts[0])
        }
        if set(self.head_ids) != expected_ids:
            raise ValueError(f"{self.task}/seed_{self.seed}: incomplete layer/head grid")

        if self.movement_by_graph is not None:
            matrix = np.asarray(self.movement_by_graph, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] != count:
                raise ValueError(
                    f"{self.task}/seed_{self.seed}: movement matrix has shape {matrix.shape}; "
                    f"expected [graphs, {count}]"
                )
            if not np.isfinite(matrix).all():
                raise ValueError(
                    f"{self.task}/seed_{self.seed}: movement matrix contains non-finite values"
                )
            self.movement_by_graph = matrix
            if self.graph_ids is None:
                self.graph_ids = np.arange(matrix.shape[0], dtype=np.int64)
            else:
                self.graph_ids = np.asarray(self.graph_ids, dtype=np.int64).reshape(-1)
            if len(self.graph_ids) != matrix.shape[0] or len(set(self.graph_ids.tolist())) != len(
                self.graph_ids
            ):
                raise ValueError(f"{self.task}/seed_{self.seed}: invalid graph identities")
            if expected_graphs is not None and strict and matrix.shape[0] != int(expected_graphs):
                raise ValueError(
                    f"{self.task}/seed_{self.seed}: observed {matrix.shape[0]} graphs; "
                    f"expected {int(expected_graphs)}"
                )
            reconstructed = np.mean(matrix, axis=0)
            if not np.allclose(
                reconstructed,
                self.ablation_impact,
                rtol=2.0e-5,
                atol=1.0e-9,
            ):
                maximum = float(np.max(np.abs(reconstructed - self.ablation_impact)))
                raise ValueError(
                    f"{self.task}/seed_{self.seed}: graph rows do not reconstruct cached "
                    f"head means (maximum absolute difference {maximum:.3e})"
                )
        elif expected_graphs is not None and strict:
            raise ValueError(
                f"{self.task}/seed_{self.seed}: graph-level rows are required for the "
                "dissertation-matched confidence interval"
            )

        recomputed_pooled = _spearman(self.joint_sensitivity, self.ablation_impact)
        if (
            self.original_pooled_rho is not None
            and np.isfinite(self.original_pooled_rho)
            and not np.isclose(
                recomputed_pooled,
                float(self.original_pooled_rho),
                rtol=1.0e-5,
                atol=5.0e-5,
            )
        ):
            raise ValueError(
                f"{self.task}/seed_{self.seed}: cached pooled rho "
                f"{float(self.original_pooled_rho):.6f} is not reproduced by the "
                f"loaded heads ({recomputed_pooled:.6f})"
            )
        return self

    def head_records(self) -> list[dict[str, Any]]:
        return [
            {
                "task": self.task,
                "seed": int(self.seed),
                "layer": int(self.layers[position]),
                "head": int(self.head_ids[position][1]),
                "joint_sensitivity": float(self.joint_sensitivity[position]),
                "ablation_impact": float(self.ablation_impact[position]),
            }
            for position in range(len(self.head_ids))
        ]


@dataclass(frozen=True)
class LoadedArtifact:
    path: Path
    value: Any
    metadata: Mapping[str, Any]
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pt(
    path: Path,
    *,
    allow_historical_direct: bool = False,
) -> LoadedArtifact:
    """Open an immutable cache and verify its internally stored contract.

    All molecular dissertation caches use the repository's canonical wrapper and
    therefore pass through :func:`load_cache_artifact_file`.  Only the historical
    mixed-synthetic analysis shards predate that wrapper; callers must opt in to
    their direct payload format explicitly.
    """

    resolved = Path(path)
    try:
        artifact = load_cache_artifact_file(resolved)
    except StaleCacheError:
        if not allow_historical_direct:
            raise
        import torch

        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or {"metadata", "value"}.issubset(payload):
            value = payload
            metadata: Mapping[str, Any] = {}
        else:
            # A malformed canonical wrapper must never escape fingerprint checks by
            # taking the historical path.
            raise StaleCacheError(
                f"canonical-looking cache did not pass integrity validation: {resolved}"
            )
        return LoadedArtifact(
            path=resolved,
            value=value,
            metadata=dict(metadata),
            sha256=_sha256(resolved),
        )
    return LoadedArtifact(
        path=artifact.path,
        value=artifact.value,
        metadata=dict(artifact.metadata),
        sha256=artifact.file_sha256,
    )


def _as_array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _field(record: Any, name: str) -> Any:
    return record[name] if isinstance(record, Mapping) else getattr(record, name)


def _contract(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    value = metadata.get("contract", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_scalar(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _json_scalar(dataclasses.asdict(value))
    if hasattr(value, "tolist"):
        return _json_scalar(value.tolist())
    return repr(value)


def _contract_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    contract = _contract(metadata)
    fields = (
        "protocol_fingerprint",
        "task",
        "task_adapter_version",
        "checkpoint_sha256",
        "train_seed",
        "model_geometry",
        "output_representation",
        "sigma",
        "split_fingerprint",
        "score_manifest_hash",
        "event_manifest_hash",
    )
    return {
        "protocol_version": metadata.get("protocol_version"),
        "contract_fingerprint": metadata.get("contract_fingerprint"),
        **{field: _json_scalar(contract.get(field)) for field in fields if field in contract},
    }


def _validate_contract_identity(
    metadata: Mapping[str, Any],
    *,
    task: str,
    seed: int,
    context: str,
) -> None:
    contract = _contract(metadata)
    cached_task = contract.get("task")
    if cached_task is not None and str(cached_task) != str(task):
        raise ValueError(f"{context}: cache task is {cached_task!r}; expected {task!r}")
    cached_seed = contract.get("train_seed")
    if cached_seed is not None and int(cached_seed) != int(seed):
        raise ValueError(f"{context}: cache seed is {cached_seed!r}; expected {int(seed)}")


def _validate_paired_contracts(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    first, second = _contract(left), _contract(right)
    if not first or not second:
        raise ValueError(f"{context}: paired canonical cache contract is missing")
    # Scores and causal validation may have different event manifests because they
    # are stage-specific.  The repository's own multi-stage finalizer deliberately
    # excludes only that field and checkout provenance when binding those stages.
    ignored = {"event_manifest_hash", "repository_commit"}
    comparable_first = {key: value for key, value in first.items() if key not in ignored}
    comparable_second = {key: value for key, value in second.items() if key not in ignored}
    mismatched = sorted(
        key
        for key in set(comparable_first) | set(comparable_second)
        if comparable_first.get(key) != comparable_second.get(key)
    )
    if mismatched:
        raise ValueError(f"{context}: paired cache contracts differ in {mismatched}")
    return {
        "left": _contract_summary(left),
        "right": _contract_summary(right),
        "matched_scientific_fields": sorted(comparable_first),
        "stage_specific_fields_not_compared": sorted(ignored),
    }


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    first = np.asarray(x, dtype=np.float64).reshape(-1)
    second = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(first) & np.isfinite(second)
    first, second = first[finite], second[finite]
    if len(first) < 3 or float(np.std(first)) <= 0.0 or float(np.std(second)) <= 0.0:
        return float("nan")
    return float(spearmanr(first, second).statistic)


def _centred_ranks(values: Sequence[float]) -> np.ndarray:
    from scipy.stats import rankdata

    ranks = rankdata(np.asarray(values, dtype=np.float64), method="average")
    return ranks - np.mean(ranks)


def mean_within_layer_spearman(
    data: HeadAblationData,
    impact: Sequence[float] | None = None,
    *,
    strict: bool = True,
) -> tuple[float, list[dict[str, Any]]]:
    """Return the equal-weight mean of per-layer Spearman correlations."""

    values = (
        data.ablation_impact if impact is None else np.asarray(impact, dtype=np.float64).reshape(-1)
    )
    if len(values) != len(data.joint_sensitivity):
        raise ValueError("replacement impact vector has the wrong head count")
    rows: list[dict[str, Any]] = []
    for layer in sorted(set(data.layers.tolist())):
        mask = data.layers == int(layer)
        rho = _spearman(data.joint_sensitivity[mask], values[mask])
        if strict and not np.isfinite(rho):
            raise ValueError(
                f"{data.task}/seed_{data.seed}/layer_{layer}: Spearman rho is not estimable"
            )
        rows.append(
            {
                "task": data.task,
                "seed": int(data.seed),
                "layer": int(layer),
                "heads": int(np.sum(mask)),
                "rho": float(rho),
            }
        )
    finite = np.asarray([row["rho"] for row in rows if np.isfinite(row["rho"])])
    if not len(finite):
        raise ValueError(f"{data.task}/seed_{data.seed}: no estimable layer correlations")
    return float(np.mean(finite)), rows


def stratified_residual_rank_spearman(
    datasets: Sequence[HeadAblationData],
) -> float:
    """Correlate ranks after centring them inside every seed/layer stratum."""

    x_parts, y_parts = [], []
    for data in datasets:
        for layer in sorted(set(data.layers.tolist())):
            mask = data.layers == int(layer)
            x_parts.append(_centred_ranks(data.joint_sensitivity[mask]))
            y_parts.append(_centred_ranks(data.ablation_impact[mask]))
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else float("nan")


def association_summary(
    datasets: Sequence[HeadAblationData],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarise the corrected estimand while retaining the old pooled diagnostic."""

    if not datasets:
        raise ValueError("association summary needs at least one trained model")
    seed_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for data in sorted(datasets, key=lambda row: int(row.seed)):
        estimate, rows = mean_within_layer_spearman(data)
        layer_rows.extend(rows)
        seed_rows.append(
            {
                "seed": int(data.seed),
                "pooled_raw_rho": _spearman(
                    data.joint_sensitivity,
                    data.ablation_impact,
                ),
                "within_layer_mean_rho": float(estimate),
                "layers": len(rows),
                "heads": len(data.head_ids),
            }
        )
    corrected = float(np.mean([row["within_layer_mean_rho"] for row in seed_rows]))
    common_layers = sorted(
        set.intersection(
            *(
                {int(row["layer"]) for row in layer_rows if int(row["seed"]) == int(seed["seed"])}
                for seed in seed_rows
            )
        )
    )
    leave_one_layer_out = {}
    for omitted in common_layers:
        reduced_seed_estimates = []
        for seed in seed_rows:
            values = [
                float(row["rho"])
                for row in layer_rows
                if int(row["seed"]) == int(seed["seed"])
                and int(row["layer"]) != int(omitted)
                and np.isfinite(row["rho"])
            ]
            if values:
                reduced_seed_estimates.append(float(np.mean(values)))
        if reduced_seed_estimates:
            leave_one_layer_out[str(omitted)] = float(np.mean(reduced_seed_estimates))
    joint = np.concatenate([data.joint_sensitivity for data in datasets])
    impact = np.concatenate([data.ablation_impact for data in datasets])
    finite_layers = np.asarray(
        [float(row["rho"]) for row in layer_rows if np.isfinite(row["rho"])],
        dtype=np.float64,
    )
    finite_seeds = np.asarray(
        [float(row["within_layer_mean_rho"]) for row in seed_rows],
        dtype=np.float64,
    )
    return (
        {
            "estimand": (
                "Spearman rho within each layer; equal-weight layer mean within trained "
                "seed; equal-weight trained-seed mean"
            ),
            "strata": ["trained_seed", "layer"],
            "within_layer_mean_rho": corrected,
            "pooled_raw_rho": _spearman(joint, impact),
            "mean_seed_pooled_raw_rho": float(
                np.mean([row["pooled_raw_rho"] for row in seed_rows])
            ),
            "stratified_residual_rank_rho": stratified_residual_rank_spearman(datasets),
            "seed_estimates": seed_rows,
            "layer_rho_range": [float(np.min(finite_layers)), float(np.max(finite_layers))],
            "seed_rho_range": [float(np.min(finite_seeds)), float(np.max(finite_seeds))],
            "leave_one_layer_out": leave_one_layer_out,
            "trained_seeds": len(seed_rows),
            "layers": int(sum(row["layers"] for row in seed_rows)),
            "heads": int(sum(row["heads"] for row in seed_rows)),
        },
        layer_rows,
    )


def graph_bootstrap_interval(
    datasets: Sequence[HeadAblationData],
    *,
    replicates: int = 2_000,
    rng_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any] | None:
    """Resample molecules and recompute the complete within-layer estimand."""

    if not datasets or any(data.movement_by_graph is None for data in datasets):
        return None
    if int(replicates) < 2:
        raise ValueError("bootstrap needs at least two replicates")
    rng = np.random.default_rng(int(rng_seed))
    draws = np.empty(int(replicates), dtype=np.float64)
    for draw in range(int(replicates)):
        seed_estimates = []
        for data in datasets:
            matrix = np.asarray(data.movement_by_graph, dtype=np.float64)
            indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
            impact = np.mean(matrix[indices], axis=0)
            estimate, _rows = mean_within_layer_spearman(data, impact)
            seed_estimates.append(estimate)
        draws[draw] = float(np.mean(seed_estimates))
    point = float(np.mean([mean_within_layer_spearman(data)[0] for data in datasets]))
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "estimate": point,
        "low": float(low),
        "high": float(high),
        "replicates": int(replicates),
        "rng_seed": int(rng_seed),
        "resampled_unit": "held-out molecule shared across all heads in a trained model",
        "fixed": ["joint sensitivity", "head identities", "layer identities"],
        "draw_mean": float(np.mean(draws)),
        "draw_standard_deviation": float(np.std(draws)),
    }


def within_layer_permutation_test(
    datasets: Sequence[HeadAblationData],
    *,
    replicates: int = 10_000,
    rng_seed: int = DEFAULT_PERMUTATION_SEED,
) -> dict[str, Any]:
    """Shuffle impact ranks only within seed/layer and recompute the effect size."""

    if int(replicates) < 1:
        raise ValueError("permutation test needs at least one replicate")
    strata: list[list[tuple[np.ndarray, np.ndarray]]] = []
    observed_seed = []
    for data in datasets:
        seed_strata = []
        layer_rhos = []
        for layer in sorted(set(data.layers.tolist())):
            mask = data.layers == int(layer)
            x = _centred_ranks(data.joint_sensitivity[mask])
            y = _centred_ranks(data.ablation_impact[mask])
            denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
            if denominator <= 0.0:
                raise ValueError(f"{data.task}/seed_{data.seed}/layer_{layer}: constant ranks")
            seed_strata.append((x, y))
            layer_rhos.append(float(np.dot(x, y) / denominator))
        strata.append(seed_strata)
        observed_seed.append(float(np.mean(layer_rhos)))
    observed = float(np.mean(observed_seed))
    rng = np.random.default_rng(int(rng_seed))
    extreme = 0
    null_sum = 0.0
    null_square_sum = 0.0
    for _draw in range(int(replicates)):
        seed_estimates = []
        for seed_strata in strata:
            layer_values = []
            for x, y in seed_strata:
                shuffled = rng.permutation(y)
                denominator = float(np.linalg.norm(x) * np.linalg.norm(shuffled))
                layer_values.append(float(np.dot(x, shuffled) / denominator))
            seed_estimates.append(float(np.mean(layer_values)))
        value = float(np.mean(seed_estimates))
        null_sum += value
        null_square_sum += value * value
        extreme += int(abs(value) >= abs(observed))
    mean = null_sum / int(replicates)
    variance = max(0.0, null_square_sum / int(replicates) - mean * mean)
    return {
        "observed": observed,
        "p_two_sided": float((extreme + 1) / (int(replicates) + 1)),
        "replicates": int(replicates),
        "rng_seed": int(rng_seed),
        "shuffle_strata": ["trained_seed", "layer"],
        "null_mean": float(mean),
        "null_standard_deviation": float(np.sqrt(variance)),
    }


def _validate_dissertation_regression(task: str, summary: Mapping[str, Any]) -> None:
    expected_pooled = EXPECTED_DISSERTATION_POOLED_RHO[task]
    if round(float(summary["pooled_raw_rho"]), 2) != expected_pooled:
        raise ValueError(
            f"{task}: loaded cache gives pooled rho "
            f"{float(summary['pooled_raw_rho']):.6f}, which does not reproduce "
            f"the dissertation value {expected_pooled:.2f}"
        )
    expected_corrected = EXPECTED_INDEPENDENT_CORRECTION.get(task)
    if expected_corrected is not None and round(
        float(summary["stratified_residual_rank_rho"]), 2
    ) != float(expected_corrected):
        raise ValueError(
            f"{task}: stratified residual-rank rho "
            f"{float(summary['stratified_residual_rank_rho']):.6f} does not "
            f"reproduce the independently reported within-layer value "
            f"{float(expected_corrected):.2f}"
        )


def _candidate(
    task: str,
    family: str,
    root: Path,
    *,
    exact: bool,
    expected_graphs: int | None,
    priority: int,
    origin: str,
) -> CacheCandidate | None:
    root = Path(root)
    if family == "synthetic":
        csv_path = root / "tables/per_head_metrics.csv"
        shards = tuple(sorted((root / "analysis").glob("seed_*__*.pt")))
        primary = ((csv_path,) if csv_path.is_file() else ()) + shards
        if not primary:
            return None
    elif family == "canonical_validation":
        primary = (
            root / "cache/scores/raw.pt",
            root / "cache/causal/validation.pt",
        )
        if not all(path.is_file() for path in primary):
            return None
    elif family == "graphormer_focused":
        core = root / "cache/focused/core_tests.pt"
        shards = tuple(sorted((root / "cache/focused/clean_ablation").glob("graph_*.pt")))
        if not core.is_file():
            return None
        primary = (core, *shards)
    elif family == "population_core":
        core = root / "cache/focused_population/core_tests.pt"
        shard_directories = (
            root / "cache/focused/clean_ablation",
            root / "cache/focused_population/clean_ablation",
        )
        shards = tuple(
            sorted(
                {path for directory in shard_directories for path in directory.glob("graph_*.pt")}
            )
        )
        if not core.is_file():
            return None
        primary = (core, *shards)
    else:
        raise ValueError(f"unknown cache family {family!r}")
    return CacheCandidate(
        task=task,
        family=family,
        root=root,
        primary_paths=tuple(primary),
        exact_dissertation=bool(exact),
        expected_graphs=expected_graphs,
        priority=int(priority),
        origin=origin,
    )


def _normalise_override(task: str, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        if task == "synthetic" and path.name == "per_head_metrics.csv":
            return path.parent.parent
        if task == "synthetic" and path.suffix == ".pt" and path.parent.name == "analysis":
            return path.parent.parent
        if path.name in {"validation.pt", "raw.pt"}:
            current = path.parent
            while current.name != f"seed_{42}" and current != current.parent:
                current = current.parent
            return current
        if path.name == "core_tests.pt":
            current = path.parent
            while not current.name.startswith("seed_") and current != current.parent:
                current = current.parent
            return current
    if task == "synthetic" and (path / "tables/per_head_metrics.csv").is_file():
        return path
    seed = 0 if task == "graphormer_pcqm4mv2" else 42
    descendants = (
        path / task / f"seed_{seed}",
        path / f"seed_{seed}",
        path,
    )
    for descendant in descendants:
        if descendant.is_dir() and (descendant / "cache").is_dir():
            return descendant
    return path


def discover_cache_candidates(
    metrics_root: str | Path,
    *,
    cache_overrides: Mapping[str, str | Path | None] | None = None,
) -> list[CacheCandidate]:
    """Search known dissertation roots first, then shallow Drive alternatives."""

    root = Path(metrics_root)
    overrides = dict(cache_overrides or {})
    candidates: list[CacheCandidate] = []

    for task, value in overrides.items():
        if not value:
            continue
        if task not in DEFAULT_TASKS:
            raise ValueError(f"unknown cache override task {task!r}")
        location = _normalise_override(task, value)
        families = (
            ("synthetic",)
            if task == "synthetic"
            else (
                ("population_core", "graphormer_focused")
                if "paper" in location.as_posix().lower()
                else ("graphormer_focused", "population_core")
            )
            if task == "graphormer_pcqm4mv2"
            else ("canonical_validation", "population_core")
        )
        match = None
        for family in families:
            match = _candidate(
                task,
                family,
                location,
                exact=family != "population_core",
                expected_graphs=(
                    EXPECTED_DISSERTATION_GRAPHS.get(task) if family != "population_core" else 256
                ),
                priority=-100,
                origin="explicit override",
            )
            if match is not None:
                break
        if match is None:
            raise CacheDiscoveryError(
                f"cache override for {task!r} does not contain a supported artifact set: {location}"
            )
        candidates.append(match)

    overridden = {candidate.task for candidate in candidates}
    if "synthetic" not in overridden:
        known = root / "causal_specialisation_double_dissociation/cycle_dual_v2"
        found = _candidate(
            "synthetic",
            "synthetic",
            known,
            exact=True,
            expected_graphs=None,
            priority=0,
            origin="known dissertation root",
        )
        if found is not None:
            candidates.append(found)
        scans = tuple(root.glob("*/cycle_dual_v2")) + tuple(root.glob("*/*/cycle_dual_v2"))
        for location in scans:
            found = _candidate(
                "synthetic",
                "synthetic",
                location,
                exact=True,
                expected_graphs=None,
                priority=20,
                origin="shallow Drive search",
            )
            if found is not None:
                candidates.append(found)

    if "graphormer_pcqm4mv2" not in overridden:
        known = root / "graphormer_pcqm4mv2_causal/graphormer_pcqm4mv2/seed_0"
        found = _candidate(
            "graphormer_pcqm4mv2",
            "graphormer_focused",
            known,
            exact=True,
            expected_graphs=128,
            priority=0,
            origin="known dissertation root",
        )
        if found is not None:
            candidates.append(found)
        scans = tuple(root.glob("*/graphormer_pcqm4mv2/seed_0")) + tuple(
            root.glob("*/*/graphormer_pcqm4mv2/seed_0")
        )
        for location in scans:
            if "paper" in location.as_posix().lower():
                continue
            found = _candidate(
                "graphormer_pcqm4mv2",
                "graphormer_focused",
                location,
                exact=True,
                expected_graphs=128,
                priority=20,
                origin="shallow Drive search",
            )
            if found is not None:
                candidates.append(found)
        population = _candidate(
            "graphormer_pcqm4mv2",
            "population_core",
            root / "graphormer_pcqm4mv2_causal_paper/graphormer_pcqm4mv2/seed_0",
            exact=False,
            expected_graphs=256,
            priority=100,
            origin="paper-population fallback",
        )
        if population is not None:
            candidates.append(population)

    for task in ("zinc", "qm9_gap_dense"):
        if task in overridden:
            continue
        known_roots = (
            (root / "canonical_methodology_v4_zinc_qm9", 0),
            (root / "canonical_methodology", 10),
        )
        for canonical_root, priority in known_roots:
            found = _candidate(
                task,
                "canonical_validation",
                canonical_root / task / "seed_42",
                exact=True,
                expected_graphs=64,
                priority=priority,
                origin="known dissertation root",
            )
            if found is not None:
                candidates.append(found)
        scans = tuple(root.glob(f"*/{task}/seed_42")) + tuple(root.glob(f"*/*/{task}/seed_42"))
        for location in scans:
            found = _candidate(
                task,
                "canonical_validation",
                location,
                exact=True,
                expected_graphs=64,
                priority=20,
                origin="shallow Drive search",
            )
            if found is not None:
                candidates.append(found)
        population = _candidate(
            task,
            "population_core",
            root / "grit_dense_causal_population_paper" / task / "seed_42",
            exact=False,
            expected_graphs=256,
            priority=100,
            origin="paper-population fallback",
        )
        if population is not None:
            candidates.append(population)

    unique: dict[tuple[str, str], CacheCandidate] = {}
    for candidate in candidates:
        key = (candidate.task, str(candidate.root.resolve()))
        current = unique.get(key)
        if current is None or candidate.priority < current.priority:
            unique[key] = candidate
    return sorted(
        unique.values(),
        key=lambda row: (DEFAULT_TASKS.index(row.task), row.priority, str(row.root)),
    )


@cache
def _validated_candidate_data(
    candidate: CacheCandidate,
    *,
    strict: bool,
) -> tuple[HeadAblationData, ...]:
    """Load one immutable candidate once per runtime, including Drive inventory/run reuse."""

    return tuple(load_candidate(candidate, strict=bool(strict)))


@cache
def candidate_preflight(
    candidate: CacheCandidate,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate a discovered candidate before allowing priority to select it.

    This intentionally opens the cached values in read-only mode.  A known-path
    core that is truncated, stale, missing graph rows, or bound to another task is
    therefore reported in the inventory but cannot hide a complete relocated run.
    """

    try:
        datasets = _validated_candidate_data(candidate, strict=bool(strict))
        summary, _rows = association_summary(datasets)
        if strict and candidate.exact_dissertation:
            _validate_dissertation_regression(candidate.task, summary)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, IndexError) as error:
        return {
            "usable": False,
            "error_type": type(error).__name__,
            "reason": str(error),
        }
    return {
        "usable": True,
        "reason": "complete cache contract and analysis schema validated",
        "trained_seeds": len(datasets),
        "heads": int(sum(len(data.head_ids) for data in datasets)),
        "graphs": [
            None if data.movement_by_graph is None else int(data.movement_by_graph.shape[0])
            for data in datasets
        ],
        "pooled_raw_rho": float(summary["pooled_raw_rho"]),
    }


def select_cache_candidate(
    task: str,
    candidates: Sequence[CacheCandidate],
    *,
    allow_non_dissertation_fallbacks: bool = False,
    strict: bool = True,
) -> CacheCandidate:
    matches = [candidate for candidate in candidates if candidate.task == task]
    validated = [
        candidate
        for candidate in matches
        if bool(candidate_preflight(candidate, strict=bool(strict))["usable"])
    ]
    exact = [candidate for candidate in validated if candidate.exact_dissertation]
    eligible = exact or (validated if allow_non_dissertation_fallbacks else [])
    if not eligible:
        alternatives = "\n".join(
            f"  - {candidate.family}: {candidate.root} "
            f"[{candidate_preflight(candidate, strict=bool(strict))['reason']}]"
            for candidate in matches
        )
        detail = f"\nDiscovered candidates:\n{alternatives}" if alternatives else ""
        raise CacheDiscoveryError(
            f"no complete exact dissertation cache was found for {task!r}.{detail}\n"
            "Set CACHE_OVERRIDES for a relocated exact cache. Set "
            "ALLOW_NON_DISSERTATION_FALLBACKS=True only for an explicitly labelled "
            "robustness rerun."
        )
    best_priority = min(candidate.priority for candidate in eligible)
    best = [candidate for candidate in eligible if candidate.priority == best_priority]
    if len(best) != 1:
        listing = "\n".join(f"  - {candidate.root}" for candidate in best)
        raise CacheDiscoveryError(
            f"ambiguous compatible caches for {task!r}:\n{listing}\n"
            "Choose one exact root with CACHE_OVERRIDES."
        )
    return best[0]


def _analysis_seed(path: Path) -> int:
    match = re.match(r"seed_(\d+)__", path.stem)
    if not match:
        raise ValueError(f"cannot infer synthetic seed from {path.name}")
    return int(match.group(1))


def _load_synthetic(candidate: CacheCandidate, *, strict: bool) -> list[HeadAblationData]:
    csv_path = candidate.root / "tables/per_head_metrics.csv"
    source_hashes: dict[str, str] = {}
    table_grouped: dict[int, list[dict[str, Any]]] = {}
    if csv_path.is_file():
        required = {"seed", "layer", "head", "joint_score_J", "ablation_joint_impact"}
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"synthetic table {csv_path} is missing {sorted(required)}")
            for row in reader:
                seed = int(row["seed"])
                table_grouped.setdefault(seed, []).append(
                    {
                        "layer": int(row["layer"]),
                        "head": int(row["head"]),
                        "joint": float(row["joint_score_J"]),
                        "impact": float(row["ablation_joint_impact"]),
                    }
                )
        source_hashes[str(csv_path)] = _sha256(csv_path)

    by_seed: dict[int, list[Path]] = {}
    for path in sorted((candidate.root / "analysis").glob("seed_*__*.pt")):
        by_seed.setdefault(_analysis_seed(path), []).append(path)
    shard_grouped: dict[int, list[dict[str, Any]]] = {}
    shard_contracts: dict[int, dict[str, Any]] = {}
    for seed in (0, 1, 2):
        paths = by_seed.get(seed, [])
        preferred = [path for path in paths if SYNTHETIC_ANALYSIS_FINGERPRINT in path.name]
        selected = preferred if preferred else ([] if table_grouped else paths)
        if not selected:
            continue
        if len(selected) != 1:
            raise CacheDiscoveryError(
                f"synthetic seed {seed} has {len(selected)} candidate analysis shards; "
                "restore per_head_metrics.csv or set an unambiguous cache override"
            )
        artifact = _load_pt(selected[0], allow_historical_direct=True)
        result = artifact.value
        if not isinstance(result, Mapping):
            raise TypeError(f"synthetic analysis shard is malformed: {artifact.path}")
        expected_fields = {
            "version": SYNTHETIC_EXPERIMENT_VERSION,
            "analysis_version": SYNTHETIC_ANALYSIS_VERSION,
            "analysis_fingerprint": SYNTHETIC_ANALYSIS_FINGERPRINT,
            "seed": int(seed),
        }
        mismatched = {
            key: {"observed": result.get(key), "expected": expected}
            for key, expected in expected_fields.items()
            if result.get(key) != expected
        }
        if mismatched:
            raise ValueError(
                f"synthetic analysis identity mismatch in {artifact.path}: {mismatched}"
            )
        semantic = _as_array(result["semantic_score"])
        structural = _as_array(result["structural_score"])
        sem_impact = _as_array(result["ablation_semantic"]["functional"]).mean(axis=-1)
        str_impact = _as_array(result["ablation_structural"]["functional"]).mean(axis=-1)
        if not (
            semantic.shape
            == structural.shape
            == sem_impact.shape
            == str_impact.shape
            == EXPECTED_GEOMETRY["synthetic"]
        ):
            raise ValueError(
                f"synthetic analysis arrays in {artifact.path} have incompatible geometry"
            )
        sem_norm = semantic / np.mean(semantic)
        str_norm = structural / np.mean(structural)
        joint = 0.5 * (sem_norm + str_norm)
        sem_impact = sem_impact / np.mean(sem_impact)
        str_impact = str_impact / np.mean(str_impact)
        impact = 0.5 * (sem_impact + str_impact)
        shard_grouped[seed] = [
            {
                "layer": layer,
                "head": head,
                "joint": float(joint[layer, head]),
                "impact": float(impact[layer, head]),
            }
            for layer in range(joint.shape[0])
            for head in range(joint.shape[1])
        ]
        shard_contracts[seed] = {
            "version": result["version"],
            "analysis_version": result["analysis_version"],
            "training_fingerprint": result.get("fingerprint"),
            "analysis_fingerprint": result["analysis_fingerprint"],
            "seed": int(result["seed"]),
        }
        source_hashes[str(artifact.path)] = artifact.sha256

    if shard_grouped and sorted(shard_grouped) != [0, 1, 2]:
        raise ValueError(
            f"synthetic analysis shards have seeds {sorted(shard_grouped)}; expected [0, 1, 2]"
        )
    training_fingerprints = {
        str(contract["training_fingerprint"])
        for contract in shard_contracts.values()
        if contract.get("training_fingerprint") is not None
    }
    if len(training_fingerprints) > 1:
        raise ValueError(
            f"synthetic analysis shards bind different training fingerprints: "
            f"{sorted(training_fingerprints)}"
        )
    if table_grouped and sorted(table_grouped) != [0, 1, 2]:
        raise ValueError(f"synthetic table has seeds {sorted(table_grouped)}; expected [0, 1, 2]")
    if shard_grouped and table_grouped:
        for seed in (0, 1, 2):
            left = sorted(table_grouped[seed], key=lambda row: (row["layer"], row["head"]))
            right = sorted(shard_grouped[seed], key=lambda row: (row["layer"], row["head"]))
            if [(row["layer"], row["head"]) for row in left] != [
                (row["layer"], row["head"]) for row in right
            ] or not np.allclose(
                [[row["joint"], row["impact"]] for row in left],
                [[row["joint"], row["impact"]] for row in right],
                rtol=2.0e-5,
                atol=1.0e-9,
            ):
                raise ValueError(
                    f"synthetic table does not reproduce the validated analysis shard for seed {seed}"
                )

    grouped = shard_grouped or table_grouped
    if not grouped:
        raise CacheDiscoveryError(
            f"no usable synthetic table or analysis shards under {candidate.root}"
        )

    if sorted(grouped) != [0, 1, 2]:
        raise ValueError(f"synthetic cache has seeds {sorted(grouped)}; expected [0, 1, 2]")
    output = []
    for seed, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: (row["layer"], row["head"]))
        data = HeadAblationData(
            task="synthetic",
            seed=int(seed),
            joint_sensitivity=np.asarray([row["joint"] for row in rows]),
            layers=np.asarray([row["layer"] for row in rows]),
            head_ids=tuple((row["layer"], row["head"]) for row in rows),
            ablation_impact=np.asarray([row["impact"] for row in rows]),
            source_paths=tuple(Path(path) for path in source_hashes),
            source_sha256=source_hashes,
            cache_contract={
                "experiment_version": SYNTHETIC_EXPERIMENT_VERSION,
                "analysis_version": SYNTHETIC_ANALYSIS_VERSION,
                "analysis_fingerprint": SYNTHETIC_ANALYSIS_FINGERPRINT,
                "validated_shard": shard_contracts.get(seed),
                "table_only": not bool(shard_grouped),
            },
            source_family=candidate.family,
            exact_dissertation=candidate.exact_dissertation,
        ).validate(
            expected_geometry=EXPECTED_GEOMETRY["synthetic"],
            strict=strict,
        )
        output.append(data)
    return output


def _head_name(head: tuple[int, int]) -> str:
    return f"head_L{int(head[0])}_H{int(head[1])}"


def _load_canonical_validation(
    candidate: CacheCandidate,
    *,
    strict: bool,
) -> list[HeadAblationData]:
    score = _load_pt(candidate.root / "cache/scores/raw.pt")
    causal = _load_pt(candidate.root / "cache/causal/validation.pt")
    _validate_contract_identity(
        score.metadata,
        task=candidate.task,
        seed=42,
        context=f"{candidate.task} score cache",
    )
    _validate_contract_identity(
        causal.metadata,
        task=candidate.task,
        seed=42,
        context=f"{candidate.task} causal cache",
    )
    contract_audit = _validate_paired_contracts(
        score.metadata,
        causal.metadata,
        context=f"{candidate.task} score/causal cache",
    )
    coordinates = score.value["coordinates"]
    joint_matrix = _as_array(_field(coordinates, "joint_sensitivity"))
    layers_count, heads_count = joint_matrix.shape
    head_ids = tuple((layer, head) for layer in range(layers_count) for head in range(heads_count))
    clean = causal.value["clean_ablation"]
    impacts = []
    by_head_graph: dict[tuple[int, int], dict[int, float]] = {}
    extra_sources: list[LoadedArtifact] = []
    for head in head_ids:
        name = _head_name(head)
        payload = clean.get(name) if isinstance(clean, Mapping) else None
        artifact = None
        if payload is None:
            path = candidate.root / "cache/causal/clean_ablation" / f"{name}.pt"
            artifact = _load_pt(path)
            extra_sources.append(artifact)
            _validate_contract_identity(
                artifact.metadata,
                task=candidate.task,
                seed=42,
                context=f"{candidate.task} {name} shard",
            )
            _validate_paired_contracts(
                causal.metadata,
                artifact.metadata,
                context=f"{candidate.task} causal/{name} shard",
            )
            payload = artifact.value
        impacts.append(float(payload["prediction_movement"]))
        graph_rows = tuple(payload.get("graphs", ()))
        if not graph_rows:
            path = candidate.root / "cache/causal/clean_ablation" / f"{name}.pt"
            if not path.is_file():
                raise ValueError(f"{candidate.task}: no per-graph rows for {name}")
            if artifact is None:
                artifact = _load_pt(path)
                extra_sources.append(artifact)
                _validate_contract_identity(
                    artifact.metadata,
                    task=candidate.task,
                    seed=42,
                    context=f"{candidate.task} {name} shard",
                )
                _validate_paired_contracts(
                    causal.metadata,
                    artifact.metadata,
                    context=f"{candidate.task} causal/{name} shard",
                )
            graph_rows = tuple(artifact.value.get("graphs", ()))
        by_head_graph[head] = {
            int(row["graph"]): float(row["prediction_movement"]) for row in graph_rows
        }
    graph_sets = {tuple(sorted(values)) for values in by_head_graph.values()}
    if len(graph_sets) != 1:
        raise ValueError(f"{candidate.task}: per-head clean-ablation graph IDs disagree")
    graph_ids = np.asarray(next(iter(graph_sets)), dtype=np.int64)
    matrix = np.asarray(
        [[by_head_graph[head][int(graph)] for head in head_ids] for graph in graph_ids],
        dtype=np.float64,
    )
    association = (
        causal.value.get("associations", {})
        .get("J_vs_clean_prediction_movement", {})
        .get("pooled", {})
    )
    original_rho = association.get("rho")
    if original_rho is None:
        original_rho = _spearman(joint_matrix.reshape(-1), impacts)
    sources = (score, causal, *extra_sources)
    data = HeadAblationData(
        task=candidate.task,
        seed=42,
        joint_sensitivity=joint_matrix.reshape(-1),
        layers=np.repeat(np.arange(layers_count), heads_count),
        head_ids=head_ids,
        ablation_impact=np.asarray(impacts),
        graph_ids=graph_ids,
        movement_by_graph=matrix,
        original_pooled_rho=float(original_rho),
        source_paths=tuple(artifact.path for artifact in sources),
        source_sha256={str(artifact.path): artifact.sha256 for artifact in sources},
        cache_contract=contract_audit,
        source_family=candidate.family,
        exact_dissertation=candidate.exact_dissertation,
    ).validate(
        expected_geometry=EXPECTED_GEOMETRY[candidate.task],
        expected_graphs=candidate.expected_graphs,
        strict=strict,
    )
    return [data]


def _load_graph_core(
    candidate: CacheCandidate,
    *,
    strict: bool,
) -> list[HeadAblationData]:
    stage = "focused" if candidate.family == "graphormer_focused" else "focused_population"
    core = _load_pt(candidate.root / f"cache/{stage}/core_tests.pt")
    expected_seed = 0 if candidate.task == "graphormer_pcqm4mv2" else 42
    _validate_contract_identity(
        core.metadata,
        task=candidate.task,
        seed=expected_seed,
        context=f"{candidate.task} core cache",
    )
    clean = core.value["clean_ablation"]
    joint = _as_array(clean["J"]).reshape(-1)
    layers = _as_array(clean["layers"], dtype=np.int64).reshape(-1)
    impact = _as_array(clean["prediction_movement"]).reshape(-1)
    if "head_order" in clean:
        head_ids = tuple(tuple(map(int, head)) for head in clean["head_order"])
    else:
        counts = {int(layer): 0 for layer in sorted(set(layers.tolist()))}
        identifiers = []
        for layer in layers:
            layer = int(layer)
            identifiers.append((layer, counts[layer]))
            counts[layer] += 1
        head_ids = tuple(identifiers)
    position = {head: index for index, head in enumerate(head_ids)}
    shard_directories = (
        candidate.root / "cache/focused/clean_ablation",
        candidate.root / "cache/focused_population/clean_ablation",
    )
    shard_paths = tuple(
        sorted({path for directory in shard_directories for path in directory.glob("graph_*.pt")})
    )
    graph_rows: dict[int, np.ndarray] = {}
    shard_artifacts = []
    for path in shard_paths:
        artifact = _load_pt(path)
        shard_artifacts.append(artifact)
        _validate_contract_identity(
            artifact.metadata,
            task=candidate.task,
            seed=expected_seed,
            context=f"{candidate.task} clean-ablation shard",
        )
        _validate_paired_contracts(
            core.metadata,
            artifact.metadata,
            context=f"{candidate.task} core/clean-ablation shard",
        )
        payload = artifact.value
        graph = int(payload["graph"])
        row = np.full(len(head_ids), np.nan, dtype=np.float64)
        for value in payload.get("rows", ()):
            head = tuple(map(int, value["head"]))
            if head in position:
                row[position[head]] = float(value["prediction_movement"])
        if not np.isfinite(row).all():
            raise ValueError(f"{candidate.task}: incomplete clean-ablation shard {path}")
        graph_rows[graph] = row
    matrix = None
    graph_ids = None
    if graph_rows:
        graph_ids = np.asarray(sorted(graph_rows), dtype=np.int64)
        matrix = np.stack([graph_rows[int(graph)] for graph in graph_ids])
    original_rho = clean.get("spearman_rho", _spearman(joint, impact))
    sources = (core, *shard_artifacts)
    data = HeadAblationData(
        task=candidate.task,
        seed=expected_seed,
        joint_sensitivity=joint,
        layers=layers,
        head_ids=head_ids,
        ablation_impact=impact,
        graph_ids=graph_ids,
        movement_by_graph=matrix,
        original_pooled_rho=float(original_rho),
        source_paths=tuple(artifact.path for artifact in sources),
        source_sha256={str(artifact.path): artifact.sha256 for artifact in sources},
        cache_contract={
            "core": _contract_summary(core.metadata),
            "clean_ablation_shards": {
                "count": len(shard_artifacts),
                "contract_fingerprints": sorted(
                    {
                        str(artifact.metadata.get("contract_fingerprint"))
                        for artifact in shard_artifacts
                    }
                ),
            },
        },
        source_family=candidate.family,
        exact_dissertation=candidate.exact_dissertation,
    ).validate(
        expected_geometry=EXPECTED_GEOMETRY[candidate.task],
        expected_graphs=candidate.expected_graphs,
        strict=strict,
    )
    return [data]


def load_candidate(
    candidate: CacheCandidate,
    *,
    strict: bool = True,
) -> list[HeadAblationData]:
    if candidate.family == "synthetic":
        return _load_synthetic(candidate, strict=strict)
    if candidate.family == "canonical_validation":
        return _load_canonical_validation(candidate, strict=strict)
    if candidate.family in {"graphormer_focused", "population_core"}:
        return _load_graph_core(candidate, strict=strict)
    raise ValueError(f"unsupported cache family {candidate.family!r}")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty table {path}")
    fields = list(rows[0])
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _runtime_versions() -> dict[str, str]:
    packages = ("numpy", "scipy", "matplotlib", "torch", "pypdf", "pillow")
    output = {}
    for package in packages:
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = "not installed"
    return output


def verify_figure_bundle(
    paths: Sequence[str | Path],
    *,
    expected_inches: tuple[float, float],
) -> dict[str, Any]:
    """Reopen a figure bundle and verify its exact dissertation canvas."""

    from PIL import Image
    from pypdf import PdfReader

    by_suffix = {Path(path).suffix: Path(path) for path in paths}
    png, pdf = by_suffix[".png"], by_suffix[".pdf"]
    sidecar = by_suffix.get(".json")
    if sidecar is None or not sidecar.is_file():
        raise RuntimeError("figure provenance sidecar is missing")
    with Image.open(png) as image:
        png_pixels = tuple(map(int, image.size))
    expected_pixels = tuple(round(value * PNG_DPI) for value in expected_inches)
    if png_pixels != expected_pixels:
        raise RuntimeError(f"PNG canvas {png_pixels}; expected {expected_pixels}: {png}")
    reader = PdfReader(str(pdf))
    if len(reader.pages) != 1:
        raise RuntimeError(f"expected a one-page PDF: {pdf}")
    box = reader.pages[0].mediabox
    pdf_points = (float(box.width), float(box.height))
    expected_points = tuple(value * 72.0 for value in expected_inches)
    if not np.allclose(pdf_points, expected_points, atol=0.02):
        raise RuntimeError(f"PDF canvas {pdf_points}; expected {expected_points}: {pdf}")
    pdf_bytes = pdf.read_bytes()
    if b"/Subtype /Type3" in pdf_bytes:
        raise RuntimeError(f"Type-3 fonts are forbidden in publication output: {pdf}")
    if b"/FontFile2" not in pdf_bytes or b"/CIDFontType2" not in pdf_bytes:
        raise RuntimeError(f"TrueType publication fonts are not embedded: {pdf}")
    return {
        "png_pixels": list(png_pixels),
        "pdf_points": list(pdf_points),
        "sha256": {str(path): _sha256(path) for path in (png, pdf, sidecar)},
        "type3_fonts": False,
        "embedded_truetype_fonts": True,
    }


def inventory_rows(
    candidates: Sequence[CacheCandidate],
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        graph_shards = sum(path.name.startswith("graph_") for path in candidate.primary_paths)
        preflight = candidate_preflight(candidate, strict=bool(strict))
        rows.append(
            {
                **candidate.as_dict(),
                "primary_file_count": len(candidate.primary_paths),
                "graph_shard_count": graph_shards,
                "bootstrap_rows": (
                    "embedded; validate on load"
                    if candidate.family == "canonical_validation"
                    else "not required"
                    if candidate.family == "synthetic"
                    else (
                        "complete"
                        if candidate.expected_graphs is not None
                        and graph_shards == int(candidate.expected_graphs)
                        else "incomplete"
                    )
                ),
                "all_primary_files_exist": all(path.is_file() for path in candidate.primary_paths),
                "preflight_usable": bool(preflight["usable"]),
                "preflight_reason": str(preflight["reason"]),
            }
        )
    return rows


def run_layer_controlled_correction(
    *,
    metrics_root: str | Path,
    output_root: str | Path,
    tasks: Sequence[str] = DEFAULT_TASKS,
    cache_overrides: Mapping[str, str | Path | None] | None = None,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    permutation_replicates: int = 10_000,
    permutation_seed: int = DEFAULT_PERMUTATION_SEED,
    strict: bool = True,
    allow_non_dissertation_fallbacks: bool = False,
) -> dict[str, Any]:
    """Run the complete cache-only correction and render dissertation drop-ins."""

    selected_tasks = tuple(str(task) for task in tasks)
    unknown = sorted(set(selected_tasks) - set(DEFAULT_TASKS))
    if unknown:
        raise ValueError(
            f"unsupported tasks {unknown}; GraphBench is deliberately excluded from this Colab"
        )
    if not selected_tasks:
        raise ValueError("at least one correction task is required")
    candidates = discover_cache_candidates(
        metrics_root,
        cache_overrides=cache_overrides,
    )
    selections = {
        task: select_cache_candidate(
            task,
            candidates,
            allow_non_dissertation_fallbacks=allow_non_dissertation_fallbacks,
            strict=bool(strict),
        )
        for task in selected_tasks
    }
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    all_layer_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    task_results: dict[str, Any] = {}

    for task in selected_tasks:
        candidate = selections[task]
        datasets = list(_validated_candidate_data(candidate, strict=bool(strict)))
        summary, layer_rows = association_summary(datasets)
        bootstrap = graph_bootstrap_interval(
            datasets,
            replicates=int(bootstrap_replicates),
            rng_seed=int(bootstrap_seed),
        )
        permutation = within_layer_permutation_test(
            datasets,
            replicates=int(permutation_replicates),
            rng_seed=int(permutation_seed),
        )
        if not np.isclose(
            float(permutation["observed"]),
            float(summary["within_layer_mean_rho"]),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"{task}: permutation and summary effect sizes disagree")
        if bootstrap is not None and not np.isclose(
            float(bootstrap["estimate"]),
            float(summary["within_layer_mean_rho"]),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise RuntimeError(f"{task}: bootstrap and summary effect sizes disagree")
        summary["graph_bootstrap"] = bootstrap
        summary["within_layer_permutation"] = permutation
        summary["task"] = task
        summary["label"] = TASK_LABELS[task]
        summary["cache_family"] = candidate.family
        summary["exact_dissertation_cache"] = bool(candidate.exact_dissertation)
        if strict and candidate.exact_dissertation:
            _validate_dissertation_regression(task, summary)
        all_layer_rows.extend(layer_rows)

        low = None if bootstrap is None else float(bootstrap["low"])
        high = None if bootstrap is None else float(bootstrap["high"])
        common_figure_metadata = {
            "correction_version": CORRECTION_VERSION,
            "task": task,
            "association": {
                "statistic": "equal-weight mean of within-layer Spearman rho",
                "estimate": summary["within_layer_mean_rho"],
                "low": low,
                "high": high,
                "pooled_raw_rho_for_comparison": summary["pooled_raw_rho"],
                "mean_seed_pooled_raw_rho_for_comparison": summary["mean_seed_pooled_raw_rho"],
            },
            "cache_only": True,
            "model_forwards": 0,
            "source_family": candidate.family,
            "presentation_provenance": PRESENTATION_PROVENANCE[
                "synthetic" if task == "synthetic" else "molecular"
            ],
        }
        task_output = destination / task / "figures"
        if task == "synthetic":
            records = [record for data in datasets for record in data.head_records()]
            primary_figure_paths = render_synthetic_within_layer_rank_panel(
                records,
                estimate=float(summary["within_layer_mean_rho"]),
                output_dir=task_output,
                metadata=common_figure_metadata,
            )
            raw_figure_paths = render_synthetic_dissertation_panel(
                records,
                estimate=float(summary["within_layer_mean_rho"]),
                output_dir=task_output,
                metadata=common_figure_metadata,
            )
            primary_verification = verify_figure_bundle(
                primary_figure_paths,
                expected_inches=SYNTHETIC_FIGSIZE,
            )
            raw_verification = verify_figure_bundle(
                raw_figure_paths,
                expected_inches=SYNTHETIC_FIGSIZE,
            )
        else:
            if len(datasets) != 1:
                raise ValueError(f"the dissertation {task} panel expects one trained model")
            data = datasets[0]
            figure_record = {
                "seed": data.seed,
                "joint_sensitivity": data.joint_sensitivity,
                "ablation_impact": data.ablation_impact,
                "layer": data.layers,
            }
            primary_figure_paths = render_molecular_within_layer_rank_panel(
                figure_record,
                estimate=float(summary["within_layer_mean_rho"]),
                low=low,
                high=high,
                output_dir=task_output,
                metadata=common_figure_metadata,
            )
            raw_figure_paths = render_molecular_dissertation_panel(
                figure_record,
                estimate=float(summary["within_layer_mean_rho"]),
                low=low,
                high=high,
                output_dir=task_output,
                metadata=common_figure_metadata,
            )
            primary_verification = verify_figure_bundle(
                primary_figure_paths,
                expected_inches=MOLECULAR_FIGSIZE,
            )
            raw_verification = verify_figure_bundle(
                raw_figure_paths,
                expected_inches=MOLECULAR_FIGSIZE,
            )

        figure_paths = (*primary_figure_paths, *raw_figure_paths)

        source_paths = tuple(dict.fromkeys(path for data in datasets for path in data.source_paths))
        source_hashes = {
            key: value for data in datasets for key, value in data.source_sha256.items()
        }
        task_results[task] = {
            "summary": summary,
            "selected_cache": candidate.as_dict(),
            "source_paths": [str(path) for path in source_paths],
            "source_sha256": source_hashes,
            "cache_contracts": [_json_scalar(data.cache_contract) for data in datasets],
            "figures": [str(path) for path in figure_paths],
            "primary_within_layer_rank_figures": [str(path) for path in primary_figure_paths],
            "raw_coordinate_companion_figures": [str(path) for path in raw_figure_paths],
            "figure_verification": {
                "primary_within_layer_ranks": primary_verification,
                "raw_coordinate_companion": raw_verification,
            },
        }
        leave_one_out_values = np.asarray(
            list(summary["leave_one_layer_out"].values()), dtype=np.float64
        )
        comparison_rows.append(
            {
                "task": task,
                "model": TASK_LABELS[task],
                "pooled_raw_rho": float(summary["pooled_raw_rho"]),
                "mean_seed_pooled_raw_rho": float(summary["mean_seed_pooled_raw_rho"]),
                "within_layer_mean_rho": float(summary["within_layer_mean_rho"]),
                "stratified_residual_rank_rho": float(summary["stratified_residual_rank_rho"]),
                "ci_low": "" if low is None else low,
                "ci_high": "" if high is None else high,
                "layer_rho_min": float(summary["layer_rho_range"][0]),
                "layer_rho_max": float(summary["layer_rho_range"][1]),
                "leave_one_layer_out_min": (
                    "" if not len(leave_one_out_values) else float(np.min(leave_one_out_values))
                ),
                "leave_one_layer_out_max": (
                    "" if not len(leave_one_out_values) else float(np.max(leave_one_out_values))
                ),
                "permutation_p_two_sided": float(permutation["p_two_sided"]),
                "trained_seeds": int(summary["trained_seeds"]),
                "layers": int(summary["layers"]),
                "heads": int(summary["heads"]),
                "cache_family": candidate.family,
                "exact_dissertation_cache": bool(candidate.exact_dissertation),
            }
        )

    tables = destination / "tables"
    layer_table = _atomic_csv(tables / "within_layer_correlations.csv", all_layer_rows)
    comparison_table = _atomic_csv(tables / "pooled_vs_layer_controlled.csv", comparison_rows)
    manifest = {
        "correction_version": CORRECTION_VERSION,
        "scope": {
            "included": list(selected_tasks),
            "excluded": ["graphbench_bipartite_matching_hard"],
            "exclusion_reason": "GraphBench cache is on HPC and will be corrected separately",
        },
        "estimand": {
            "effect_size": (
                "Spearman rho within each layer; equal-weight layer mean within model; "
                "equal-weight model-seed mean"
            ),
            "molecular_interval": (
                "shared held-out-molecule bootstrap, recomputing head means and the complete "
                "layer-controlled statistic"
            ),
            "permutation": "ablation impacts shuffled only within trained-seed/layer strata",
        },
        "figure_views": {
            "primary": (
                "tie-aware percentile ranks computed separately within every trained-seed/layer "
                "stratum; same dissertation visual grammar with linear [0, 1] axes"
            ),
            "raw_coordinate_companion": (
                "original dissertation coordinates and axes; corrected statistic annotation"
            ),
        },
        "cache_only": True,
        "model_forwards": 0,
        "repository_commit": _git_commit(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _runtime_versions(),
        },
        "presentation_provenance": PRESENTATION_PROVENANCE,
        "metrics_root": str(Path(metrics_root)),
        "output_root": str(destination),
        "strict": bool(strict),
        "allow_non_dissertation_fallbacks": bool(allow_non_dissertation_fallbacks),
        "cache_inventory": [
            {
                **row,
                "selection_status": (
                    "task not requested"
                    if row["task"] not in selections
                    else "selected"
                    if selections[row["task"]].root == Path(row["root"])
                    else "not selected"
                ),
                "selection_reason": (
                    "task not requested"
                    if row["task"] not in selections
                    else "selected explicit override or highest-priority exact cache"
                    if selections[row["task"]].root == Path(row["root"])
                    else f"preflight rejected: {row['preflight_reason']}"
                    if not bool(row["preflight_usable"])
                    else "non-dissertation fallback retained for explicit robustness use"
                    if not bool(row["exact_dissertation"])
                    else "lower-priority exact candidate"
                ),
            }
            for row in inventory_rows(candidates, strict=bool(strict))
        ],
        "tasks": task_results,
        "tables": {
            "within_layer_correlations": str(layer_table),
            "pooled_vs_layer_controlled": str(comparison_table),
        },
    }
    manifest_path = destination / "run_manifest.json"
    atomic_json(manifest_path, _json_scalar(manifest))
    manifest["manifest_path"] = str(manifest_path)
    return manifest


__all__ = [
    "CORRECTION_VERSION",
    "DEFAULT_TASKS",
    "CacheCandidate",
    "CacheDiscoveryError",
    "HeadAblationData",
    "association_summary",
    "discover_cache_candidates",
    "graph_bootstrap_interval",
    "inventory_rows",
    "load_candidate",
    "mean_within_layer_spearman",
    "run_layer_controlled_correction",
    "select_cache_candidate",
    "stratified_residual_rank_spearman",
    "verify_figure_bundle",
    "within_layer_permutation_test",
]
