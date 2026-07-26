"""Cache-safe NAR-specific analyses for a methodology paper.

This module is deliberately additive.  It treats the expensive canonical NAR score caches as
immutable input artifacts and writes every derived or newly inferred quantity below a separate,
versioned extension directory.  The three registered analyses are:

1. address-versus-content head specialisation;
2. exact, in-distribution counterfactual mediation by frozen head families; and
3. specialisation changes at retrieval-capacity transitions.

The figures-only phase has no dependency on GRIT and never loads a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..methodology.bootstrap import (
    Interval,
    Observation,
    nested_percentile_interval,
    paired_channel_percentile_interval,
)
from ..methodology.cache import (
    ReadOnlyCacheArtifact,
    StaleCacheError,
    atomic_json,
    checkpoint_sha256,
    load_cache_artifact_file,
)
from ..methodology.protocol import (
    BOOTSTRAP_REPLICATES,
    PROTOCOL_VERSION,
    BootstrapPolicy,
    ExecutionPolicy,
    FamilyPolicy,
    MethodologyConfig,
    NumericalPolicy,
    RunSizes,
    deterministic_splits,
    stable_hash,
)
from ..methodology.scores import (
    HeadCoordinates,
    freeze_families,
    freeze_matched_controls,
    head_coordinates,
)
from . import nar_grit_fixed as training
from .nar_canonical_analysis import (
    MODEL_COLOURS,
    MODEL_LABELS,
    MODEL_MARKERS,
    MODEL_ORDER,
    SEED_MARKERS,
    CanonicalNarBackend,
    NarGraphDataset,
    NarSemanticDonorPool,
    NarTaskSpec,
    _save_figure,
    _style,
    _training_config,
    _write_csv,
    checkpoint_payloads,
    find_checkpoint,
    register_nar_tasks,
    task_name,
)


EXTENSION_VERSION = "nar-role-counterfactual-v1"
EXTENSION_PROTOCOL = "nar-methodology-extension-v1"
DEFAULT_EXTENSION_NAME = "nar_role_counterfactual_v1"
ROLE_FIGURE_STEM = "06_head_role_specialisation_N16"
COUNTERFACTUAL_FIGURE_STEM = "07_counterfactual_head_role_validation"
TRANSITION_FIGURE_STEM = "08_specialisation_performance_transition"


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _atomic_torch_save(path: Path, payload: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=path.name, suffix=".partial", dir=path.parent
    )
    os.close(handle)
    try:
        torch.save(payload, temporary)
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


@dataclass(frozen=True)
class ExtensionContract:
    """Complete scientific identity of one additive extension artifact."""

    stage: str
    task: str
    seed: int
    checkpoint_sha256: str
    source_artifact_sha256: str
    source_contract_fingerprint: str
    scientific_parameters: Mapping[str, Any]
    extension_version: str = EXTENSION_VERSION
    extension_protocol: str = EXTENSION_PROTOCOL

    @property
    def fingerprint(self) -> str:
        return stable_hash(dataclasses.asdict(self))


class ExtensionStore:
    """Immutable, fail-closed cache namespace for the NAR paper extension."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(
        self,
        contract: ExtensionContract,
        name: str,
        *,
        suffix: str = ".pt",
    ) -> Path:
        return (
            self.root
            / "cache"
            / str(contract.stage)
            / str(contract.task)
            / f"seed_{int(contract.seed)}"
            / f"{name}{suffix}"
        )

    def load(self, contract: ExtensionContract, name: str) -> Any | None:
        import torch

        path = self.path(contract, name)
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError) as error:
            raise StaleCacheError(
                f"protected extension cache {path} is unreadable and was left untouched"
            ) from error
        metadata = payload.get("metadata", {}) if isinstance(payload, Mapping) else {}
        if (
            metadata.get("extension_protocol") != EXTENSION_PROTOCOL
            or metadata.get("contract_fingerprint") != contract.fingerprint
        ):
            raise StaleCacheError(
                f"protected extension cache {path} belongs to another scientific contract; "
                "select a new --extension-name"
            )
        return payload.get("value")

    def save(self, contract: ExtensionContract, name: str, value: Any) -> Path:
        path = self.path(contract, name)
        if path.exists():
            self.load(contract, name)
            return path
        _atomic_torch_save(
            path,
            {
                "metadata": {
                    "extension_protocol": EXTENSION_PROTOCOL,
                    "extension_version": EXTENSION_VERSION,
                    "contract": dataclasses.asdict(contract),
                    "contract_fingerprint": contract.fingerprint,
                },
                "value": value,
            },
        )
        return path


@dataclass(frozen=True)
class SourceBinding:
    task: str
    seed: int
    score_artifact: ReadOnlyCacheArtifact
    model_record: Mapping[str, Any]
    score_root: Path

    @property
    def checkpoint_sha256(self) -> str:
        return str(self.model_record["checkpoint_sha256"])

    @property
    def source_contract_fingerprint(self) -> str:
        return str(self.score_artifact.metadata["contract_fingerprint"])


@dataclass(frozen=True)
class RoleCoordinates:
    raw_query: np.ndarray
    raw_value: np.ndarray
    query_mean: float
    value_mean: float
    normalized_query: np.ndarray
    normalized_value: np.ndarray
    joint_sensitivity: np.ndarray
    selectivity: np.ndarray
    active: np.ndarray
    estimable: bool
    role_separation: float


def role_coordinates(
    query: Any,
    value: Any,
    *,
    score_floor: float,
    epsilon: float,
    activity_floor: float,
) -> RoleCoordinates:
    """Compute address/content coordinates and total-variation role separation."""

    coordinates = head_coordinates(
        query,
        value,
        score_floor=float(score_floor),
        epsilon=float(epsilon),
        activity_floor=float(activity_floor),
    )
    query_array = np.asarray(query, dtype=np.float64)
    value_array = np.asarray(value, dtype=np.float64)
    total_query = float(np.sum(query_array))
    total_value = float(np.sum(value_array))
    if (
        not coordinates.estimable
        or total_query <= float(score_floor)
        or total_value <= float(score_floor)
    ):
        role_separation = float("nan")
    else:
        role_separation = float(
            0.5
            * np.sum(
                np.abs(
                    query_array / total_query
                    - value_array / total_value
                )
            )
        )
    return RoleCoordinates(
        raw_query=coordinates.raw_semantic,
        raw_value=coordinates.raw_structural,
        query_mean=coordinates.semantic_mean,
        value_mean=coordinates.structural_mean,
        normalized_query=coordinates.normalized_semantic,
        normalized_value=coordinates.normalized_structural,
        joint_sensitivity=coordinates.joint_sensitivity,
        selectivity=coordinates.selectivity,
        active=coordinates.active,
        estimable=coordinates.estimable,
        role_separation=role_separation,
    )


def _role_interval_transform(
    value: np.ndarray,
    *,
    numerical: NumericalPolicy,
    families: FamilyPolicy,
) -> np.ndarray:
    coordinates = role_coordinates(
        value[0],
        value[1],
        score_floor=numerical.score_floor,
        epsilon=numerical.selectivity_epsilon,
        activity_floor=families.activity_floor,
    )
    separation = np.full(
        coordinates.raw_query.shape,
        coordinates.role_separation,
        dtype=np.float64,
    )
    return np.stack(
        (
            coordinates.raw_query,
            coordinates.raw_value,
            coordinates.normalized_query,
            coordinates.normalized_value,
            coordinates.joint_sensitivity,
            coordinates.selectivity,
            separation,
        )
    )


def _role_observations(
    score_value: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[list[Observation], list[Observation]]:
    semantic = score_value.get("channels", {}).get("semantic", {})
    rows = semantic.get("events", ())
    query: list[Observation] = []
    value: list[Observation] = []
    for row in rows:
        source = int(row["source"])
        observation = Observation(
            seed=int(seed),
            graph=int(row["graph_id"]),
            source=source,
            donor=int(row["draw"]),
            value=np.asarray(row["score"], dtype=np.float64),
        )
        if source == 2:
            query.append(observation)
        elif source >= 3:
            value.append(observation)
        else:
            raise RuntimeError(
                f"unexpected NAR semantic source {source}; expected query=2 or record>=3"
            )
    if not query or not value:
        raise RuntimeError("canonical semantic cache does not contain both NAR source roles")
    return query, value


def _interval_record(interval: Interval) -> dict[str, Any]:
    return {
        "estimate": np.asarray(interval.estimate),
        "low": np.asarray(interval.low),
        "high": np.asarray(interval.high),
        "replicates": int(interval.replicates),
        "rng_seed": int(interval.rng_seed),
        "resampled_levels": tuple(interval.resampled_levels),
        "estimable_draws": (
            None
            if interval.estimable_draws is None
            else np.asarray(interval.estimable_draws)
        ),
    }


def _head_coordinates_for_matching(role: RoleCoordinates) -> HeadCoordinates:
    return HeadCoordinates(
        raw_semantic=role.raw_query,
        raw_structural=role.raw_value,
        semantic_mean=role.query_mean,
        structural_mean=role.value_mean,
        normalized_semantic=role.normalized_query,
        normalized_structural=role.normalized_value,
        joint_sensitivity=role.joint_sensitivity,
        selectivity=role.selectivity,
        active=role.active,
        estimable=role.estimable,
    )


def freeze_role_families(
    role: RoleCoordinates,
    throughput: Any,
    *,
    policy: FamilyPolicy,
    rng_seed: int,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Freeze address/content families and same-layer matched controls from discovery only."""

    coordinates = _head_coordinates_for_matching(role)
    generic = freeze_families(
        coordinates,
        tail_fraction=policy.tail_fraction,
        central_fraction=policy.central_fraction,
    )
    controls = freeze_matched_controls(
        coordinates,
        generic,
        throughput,
        rng_seed=int(rng_seed),
    )
    address = tuple(generic.get("semantic_leaning", ()))
    content = tuple(generic.get("structural_leaning", ()))
    address_control = tuple(
        controls.get("semantic_leaning_central_control", ())
    )
    content_control = tuple(
        controls.get("structural_leaning_central_control", ())
    )
    if (
        (address and len(address_control) != len(address))
        or (content and len(content_control) != len(content))
    ):
        # A small-head model can put several tail heads in one layer and leave no *active*
        # central head there. This is a discovery result, not a reason to discard the cell.
        # Reserve enough non-family heads in every layer for one-for-one controls, retaining the
        # strongest tail members. Then match continuously on J and clean throughput; an arbitrary
        # active/inactive threshold is not imposed on the control pool. No causal outcome enters.
        J = role.joint_sensitivity
        D = role.selectivity
        throughput_array = np.asarray(throughput, dtype=np.float64)
        active = role.active
        median_d = float(np.nanmedian(D[active]))

        address_by_layer = {
            layer: sorted(
                [item for item in address if int(item[0]) == layer],
                key=lambda item: (-float(D[item]), item),
            )
            for layer in range(J.shape[0])
        }
        content_by_layer = {
            layer: sorted(
                [item for item in content if int(item[0]) == layer],
                key=lambda item: (float(D[item]), item),
            )
            for layer in range(J.shape[0])
        }
        for layer in range(J.shape[0]):
            address_layer = address_by_layer[layer]
            content_layer = content_by_layer[layer]
            # Control families may share a matched head with one another (as in the canonical
            # matcher), but heads are unique within each control. Thus the available neutral
            # count must cover the larger retained family in this layer.
            while (
                int(J.shape[1]) - len(address_layer) - len(content_layer)
                < max(len(address_layer), len(content_layer))
            ):
                candidates: list[tuple[float, str]] = []
                if address_layer:
                    # Last is the weakest address-tail member after descending-D sorting.
                    candidates.append((abs(float(D[address_layer[-1]])), "address"))
                if content_layer:
                    # Last is the weakest content-tail member after ascending-D sorting.
                    candidates.append((abs(float(D[content_layer[-1]])), "content"))
                if not candidates:
                    break
                _, remove = min(candidates, key=lambda item: (item[0], item[1]))
                if remove == "address":
                    address_layer.pop()
                else:
                    content_layer.pop()
        address = tuple(
            item
            for layer in range(J.shape[0])
            for item in address_by_layer[layer]
        )
        content = tuple(
            item
            for layer in range(J.shape[0])
            for item in content_by_layer[layer]
        )
        retained_leaning = set(address) | set(content)

        def neutral_pool(layer: int) -> list[tuple[int, int]]:
            return [
                (int(layer), int(head))
                for head in range(J.shape[1])
                if (int(layer), int(head)) not in retained_leaning
            ]

        def matched_control(
            target: Sequence[tuple[int, int]],
        ) -> tuple[tuple[int, int], ...]:
            selected: list[tuple[int, int]] = []
            used: set[tuple[int, int]] = set()
            for layer, head in target:
                candidates = [
                    item for item in neutral_pool(int(layer)) if item not in used
                ]
                if not candidates:
                    raise RuntimeError(
                        "model has no non-family head available for a same-layer control"
                    )
                scale_j = max(float(np.nanstd(J[int(layer)])), 1e-12)
                scale_t = max(
                    float(np.nanstd(throughput_array[int(layer)])), 1e-12
                )
                choice = min(
                    candidates,
                    key=lambda item: (
                        abs(float(D[item]) - median_d)
                        + abs(float(J[item]) - float(J[layer, head])) / scale_j
                        + abs(
                            float(throughput_array[item])
                            - float(throughput_array[layer, head])
                        )
                        / scale_t,
                        item,
                    ),
                )
                selected.append(choice)
                used.add(choice)
            return tuple(selected)

        if not address or not content:
            # This can occur only for a single-head model. Such a model cannot define distinct
            # specialised and control head families, so fail with an explicit estimability
            # boundary rather than an implementation-dependent matching error.
            raise RuntimeError("head-family controls require at least two heads per layer")
        address_control = matched_control(address)
        content_control = matched_control(content)
    return {
        "address": address,
        "content": content,
        "address_control": address_control,
        "content_control": content_control,
    }


def derive_role_result(
    binding: SourceBinding,
    *,
    numerical: NumericalPolicy,
    families: FamilyPolicy,
    bootstrap: BootstrapPolicy,
    analysis_seed: int,
) -> dict[str, Any]:
    """Derive role scores from a canonical score artifact without any model inference."""

    value = binding.score_artifact.value
    query_rows, value_rows = _role_observations(value, seed=binding.seed)
    query_graphs = {row.graph for row in query_rows}
    value_graphs = {row.graph for row in value_rows}
    if query_graphs != value_graphs:
        raise RuntimeError("query and value role scores do not share the discovery graphs")
    role_bootstrap = dataclasses.replace(
        bootstrap,
        rng_seed=int(bootstrap.rng_seed)
        + int(stable_hash({"task": binding.task, "seed": binding.seed}, length=8), 16)
        % 1_000_003,
    )
    interval = paired_channel_percentile_interval(
        query_rows,
        value_rows,
        role_bootstrap,
        transform=lambda rows: _role_interval_transform(
            rows, numerical=numerical, families=families
        ),
        # Each role contributes exactly one task-defined source per graph.  Only donors and
        # graphs are sampled; source resampling would be a no-op.
        resample_source=(False, False),
    )
    query_score = np.asarray(interval.estimate[0], dtype=np.float64)
    value_score = np.asarray(interval.estimate[1], dtype=np.float64)
    coordinates = role_coordinates(
        query_score,
        value_score,
        score_floor=numerical.score_floor,
        epsilon=numerical.selectivity_epsilon,
        activity_floor=families.activity_floor,
    )
    canonical_semantic = np.asarray(
        value["channels"]["semantic"]["raw"], dtype=np.float64
    )
    reconstruction = 0.5 * (query_score + value_score)
    reconstruction_error = float(np.max(np.abs(reconstruction - canonical_semantic)))
    if reconstruction_error > float(numerical.reconstruction_tolerance):
        raise RuntimeError(
            "role decomposition does not reconstruct canonical semantic score: "
            f"{reconstruction_error:.3e} > {numerical.reconstruction_tolerance:.3e}"
        )
    frozen = freeze_role_families(
        coordinates,
        value["clean_throughput"],
        policy=families,
        rng_seed=int(analysis_seed) + int(binding.seed),
    )
    active_heads = int(np.sum(coordinates.active))
    requested_tail = (
        max(1, int(np.floor(float(families.tail_fraction) * active_heads)))
        if active_heads
        else 0
    )
    return {
        "extension_version": EXTENSION_VERSION,
        "task": binding.task,
        "seed": int(binding.seed),
        "checkpoint_sha256": binding.checkpoint_sha256,
        "source_score_artifact": str(binding.score_artifact.path),
        "source_score_sha256": binding.score_artifact.file_sha256,
        "source_contract_fingerprint": binding.source_contract_fingerprint,
        "role_coordinates": coordinates,
        "interval": _interval_record(interval),
        "interval_layout": (
            "raw_query",
            "raw_value",
            "normalized_query",
            "normalized_value",
            "joint_sensitivity",
            "selectivity",
            "role_separation_broadcast",
        ),
        "families": frozen,
        "family_matching": {
            "requested_tail_heads_per_role": requested_tail,
            "retained_address_heads": len(frozen["address"]),
            "retained_content_heads": len(frozen["content"]),
            "feasibility_trimmed": bool(
                len(frozen["address"]) < requested_tail
                or len(frozen["content"]) < requested_tail
            ),
            "fallback": (
                "if a layer has too few active central controls, retain its strongest "
                "tail members while reserving same-layer non-family heads; match controls "
                "continuously on role-neutrality, J, and clean throughput"
            ),
        },
        "clean_throughput": np.asarray(value["clean_throughput"]),
        "canonical_semantic_reconstruction_error": reconstruction_error,
        "graphs": len(query_graphs),
        "query_events": len(query_rows),
        "value_events": len(value_rows),
    }


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _repository_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _score_root_for_n(
    *,
    base_canonical_root: Path,
    extension_root: Path,
    cached_ns: Sequence[int],
    records: int,
) -> Path:
    if int(records) in {int(value) for value in cached_ns}:
        return base_canonical_root
    return extension_root / "transition_scores" / "canonical"


def load_source_binding(
    *,
    base_canonical_root: Path,
    extension_root: Path,
    cached_ns: Sequence[int],
    model_name: str,
    records: int,
    seed: int,
) -> SourceBinding:
    root = _score_root_for_n(
        base_canonical_root=base_canonical_root,
        extension_root=extension_root,
        cached_ns=cached_ns,
        records=records,
    )
    task = task_name(model_name, records)
    cell = root / task / f"seed_{int(seed)}"
    artifact = load_cache_artifact_file(cell / "cache" / "scores" / "raw.pt")
    model_record = _read_json(cell / "model.json")
    contract = artifact.metadata["contract"]
    expected = (task, int(seed), str(model_record["checkpoint_sha256"]))
    actual = (
        str(contract.get("task")),
        int(contract.get("train_seed", -1)),
        str(contract.get("checkpoint_sha256")),
    )
    if actual != expected:
        raise StaleCacheError(
            f"score/model provenance mismatch for {task}:seed{seed}: {actual} != {expected}"
        )
    return SourceBinding(
        task=task,
        seed=int(seed),
        score_artifact=artifact,
        model_record=model_record,
        score_root=root,
    )


def _safe_extension_layout(base_analysis_root: Path, extension_root: Path) -> None:
    base = base_analysis_root.resolve()
    extension = extension_root.resolve()
    canonical = (base / "canonical").resolve()
    if extension == base or extension == canonical or canonical in extension.parents:
        raise ValueError(
            "extension output must be a sibling below <analysis>/extensions, never the "
            "canonical cache tree"
        )
    expected_parent = (base / "extensions").resolve()
    if expected_parent not in extension.parents:
        raise ValueError(
            f"extension root must be below {expected_parent}; received {extension}"
        )


def _methodology_policies(
    base_canonical_root: Path,
) -> tuple[
    RunSizes,
    NumericalPolicy,
    BootstrapPolicy,
    FamilyPolicy,
    int,
]:
    record = _read_json(base_canonical_root / "protocol.json")
    sizes = RunSizes(**dict(record["sizes"]))
    numerical = NumericalPolicy(**dict(record["numerical"]))
    bootstrap = BootstrapPolicy(**dict(record["bootstrap"]))
    families = FamilyPolicy(**dict(record["families"]))
    return sizes, numerical, bootstrap, families, int(record["analysis_seed"])


def _role_contract(
    binding: SourceBinding,
    *,
    numerical: NumericalPolicy,
    families: FamilyPolicy,
    bootstrap: BootstrapPolicy,
    analysis_seed: int,
) -> ExtensionContract:
    return ExtensionContract(
        stage="roles",
        task=binding.task,
        seed=binding.seed,
        checkpoint_sha256=binding.checkpoint_sha256,
        source_artifact_sha256=binding.score_artifact.file_sha256,
        source_contract_fingerprint=binding.source_contract_fingerprint,
        scientific_parameters={
            "score_floor": numerical.score_floor,
            "selectivity_epsilon": numerical.selectivity_epsilon,
            "reconstruction_tolerance": numerical.reconstruction_tolerance,
            "activity_floor": families.activity_floor,
            "tail_fraction": families.tail_fraction,
            "central_fraction": families.central_fraction,
            "bootstrap": dataclasses.asdict(bootstrap),
            "analysis_seed": int(analysis_seed),
            "decomposition": "semantic query-source versus requested-record-source",
        },
    )


def run_role_derivation(
    *,
    store: ExtensionStore,
    base_canonical_root: Path,
    extension_root: Path,
    models: Sequence[str],
    ns: Sequence[int],
    cached_ns: Sequence[int],
    seeds: Sequence[int],
    numerical: NumericalPolicy,
    families: FamilyPolicy,
    bootstrap: BootstrapPolicy,
    analysis_seed: int,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    results: dict[tuple[str, int, int], dict[str, Any]] = {}
    table: list[dict[str, Any]] = []
    for records in ns:
        for model_name in models:
            for seed in seeds:
                binding = load_source_binding(
                    base_canonical_root=base_canonical_root,
                    extension_root=extension_root,
                    cached_ns=cached_ns,
                    model_name=model_name,
                    records=int(records),
                    seed=int(seed),
                )
                contract = _role_contract(
                    binding,
                    numerical=numerical,
                    families=families,
                    bootstrap=bootstrap,
                    analysis_seed=analysis_seed,
                )
                result = store.load(contract, "role_scores")
                if result is None:
                    result = derive_role_result(
                        binding,
                        numerical=numerical,
                        families=families,
                        bootstrap=bootstrap,
                        analysis_seed=analysis_seed,
                    )
                    store.save(contract, "role_scores", result)
                results[(str(model_name), int(records), int(seed))] = result
                role: RoleCoordinates = result["role_coordinates"]
                requested_tail = (
                    max(
                        1,
                        int(
                            np.floor(
                                float(families.tail_fraction)
                                * int(np.sum(role.active))
                            )
                        ),
                    )
                    if np.any(role.active)
                    else 0
                )
                matching = result.get(
                    "family_matching",
                    {
                        "requested_tail_heads_per_role": requested_tail,
                        "feasibility_trimmed": bool(
                            len(result["families"]["address"]) < requested_tail
                            or len(result["families"]["content"]) < requested_tail
                        ),
                    },
                )
                table.append(
                    {
                        "model": model_name,
                        "N": int(records),
                        "seed": int(seed),
                        "R_role": float(role.role_separation),
                        "active_heads": int(np.sum(role.active)),
                        "address_heads": len(result["families"]["address"]),
                        "content_heads": len(result["families"]["content"]),
                        "requested_tail_heads_per_role": int(
                            matching["requested_tail_heads_per_role"]
                        ),
                        "family_feasibility_trimmed": bool(
                            matching["feasibility_trimmed"]
                        ),
                        "reconstruction_error": float(
                            result["canonical_semantic_reconstruction_error"]
                        ),
                        "source_score_sha256": result["source_score_sha256"],
                    }
                )
                print(
                    f"[roles] {binding.task}:seed{seed} "
                    f"R_role={role.role_separation:.3f}"
                    + (
                        " [same-layer control feasibility trim]"
                        if bool(matching["feasibility_trimmed"])
                        else ""
                    ),
                    flush=True,
                )
    summary_path = extension_root / "tables" / "role_summary.csv"
    merged: dict[tuple[str, int, int], dict[str, Any]] = {}
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                merged[
                    (str(row["model"]), int(row["N"]), int(row["seed"]))
                ] = dict(row)
    for row in table:
        merged[(str(row["model"]), int(row["N"]), int(row["seed"]))] = row
    _write_csv(summary_path, [merged[key] for key in sorted(merged)])
    return results


def _canonical_config_for_new_scores(
    *,
    base_canonical_root: Path,
    output_dir: Path,
    tasks: Sequence[str],
    seeds: Sequence[int],
    accelerator: str,
    graphs_per_batch: int,
    num_threads: int,
) -> MethodologyConfig:
    """Replay the stored canonical scientific protocol in an isolated score namespace."""

    record = _read_json(base_canonical_root / "protocol.json")
    return MethodologyConfig(
        output_dir=str(output_dir),
        tasks=tuple(tasks),
        train_seeds=tuple(int(seed) for seed in seeds),
        phases=("scores",),
        sizes=RunSizes(**dict(record["sizes"])),
        numerical=NumericalPolicy(**dict(record["numerical"])),
        bootstrap=BootstrapPolicy(**dict(record["bootstrap"])),
        families=FamilyPolicy(**dict(record["families"])),
        execution=ExecutionPolicy(
            graphs_per_batch=int(graphs_per_batch),
            oom_backoff=True,
            replica_pair_budget=None,
            jacobian_output_chunk=8,
            progress_heartbeat_seconds=30.0,
        ),
        analysis_seed=int(record["analysis_seed"]),
        accelerator=str(accelerator),
        num_threads=int(num_threads),
        figure_overrides=dict(record.get("figure_overrides", {})),
        skip_install=True,
        resume=True,
        force=False,
        strict_audits=False,
        compute_beneficial_carriage=False,
    )


def run_additional_transition_scores(
    *,
    base_canonical_root: Path,
    extension_root: Path,
    training_run_dir: Path,
    models: Sequence[str],
    ns: Sequence[int],
    seeds: Sequence[int],
    width: int,
    accelerator: str,
    graphs_per_batch: int,
    num_threads: int,
) -> None:
    """Run canonical score-only analysis for transition Ns in a new protected namespace."""

    if not ns:
        return
    from ..methodology.runner import run_methodology

    tasks = register_nar_tasks(
        models=models,
        analysis_ns=ns,
        width=int(width),
        training_run_dir=training_run_dir,
    )
    output = extension_root / "transition_scores" / "canonical"
    print(
        "[transition-scores] canonical score-only caches are isolated at "
        f"{output}",
        flush=True,
    )
    # Execute missing cells independently. If a repository update occurs after a partial run,
    # completed cells stay untouched and do not become stale blockers for the remaining cells.
    for task in tasks:
        for seed in seeds:
            score_path = (
                output
                / task
                / f"seed_{int(seed)}"
                / "cache"
                / "scores"
                / "raw.pt"
            )
            if score_path.exists():
                print(f"[transition-scores] protected cache exists: {task}:seed{seed}")
                continue
            config = _canonical_config_for_new_scores(
                base_canonical_root=base_canonical_root,
                output_dir=output,
                tasks=(task,),
                seeds=(int(seed),),
                accelerator=accelerator,
                graphs_per_batch=graphs_per_batch,
                num_threads=num_threads,
            )
            run_methodology(config)


@dataclass
class CounterfactualRuntime:
    task: Any
    spec: NarTaskSpec
    model: Any
    backend: CanonicalNarBackend
    eval_ds: NarGraphDataset
    donor_ds: NarGraphDataset
    donor_pool: NarSemanticDonorPool
    splits: Any
    checkpoint: Path
    checkpoint_sha256: str
    heldout_accuracy: float
    device: Any


def _dataset_seed(analysis_seed: int, model_name: str, records: int) -> int:
    return (
        int(analysis_seed)
        + int(records) * 100_003
        + MODEL_ORDER.index(str(model_name)) * 10_000_019
    )


def load_counterfactual_runtime(
    *,
    training_run_dir: Path,
    model_name: str,
    records: int,
    seed: int,
    width: int,
    sizes: RunSizes,
    analysis_seed: int,
    accelerator: str,
) -> CounterfactualRuntime:
    """Load the native NAR backend without invoking canonical model/Jacobian audits."""

    import torch

    names = register_nar_tasks(
        models=(model_name,),
        analysis_ns=(int(records),),
        width=int(width),
        training_run_dir=training_run_dir,
    )
    from ..methodology.tasks import TASKS

    task = TASKS[names[0]]
    spec: NarTaskSpec = task.spec
    checkpoint = find_checkpoint(spec, int(seed))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = (str(model_name), int(width), int(records), int(seed))
    actual = (
        str(payload.get("model_name")),
        int(payload.get("width", -1)),
        int(payload.get("N", -1)),
        int(payload.get("seed", -1)),
    )
    if actual != expected:
        raise RuntimeError(f"checkpoint metadata {actual} does not match {expected}")
    cfg = _training_config(payload, spec)
    device = torch.device(
        str(accelerator)
        if str(accelerator).startswith("cpu") or torch.cuda.is_available()
        else "cpu"
    )
    model_class = training.build_model_class()
    model = model_class(cfg, str(model_name), int(width), int(records)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    eval_size = (
        int(sizes.discovery_graphs)
        + int(sizes.causal_graphs)
        + int(sizes.clean_ablation_graphs)
    )
    data_seed = _dataset_seed(analysis_seed, model_name, records)
    eval_ds = NarGraphDataset(cfg, int(records), eval_size, data_seed)
    donor_ds = NarGraphDataset(
        cfg,
        int(records),
        int(sizes.semantic_donor_graphs),
        data_seed + 900_000_011,
    )
    splits = deterministic_splits(
        len(eval_ds),
        len(donor_ds),
        sizes,
        int(analysis_seed),
        same_index_space=False,
    )
    donor_pool = NarSemanticDonorPool(
        [(graph_id, donor_ds[graph_id]) for graph_id in splits.semantic_donor_pool],
        records=int(records),
    )
    return CounterfactualRuntime(
        task=task,
        spec=spec,
        model=model,
        backend=CanonicalNarBackend(model, task, device),
        eval_ds=eval_ds,
        donor_ds=donor_ds,
        donor_pool=donor_pool,
        splits=splits,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256(checkpoint),
        heldout_accuracy=float(payload["heldout"]["accuracy"]),
        device=device,
    )


def _counterfactual_rng(
    *,
    analysis_seed: int,
    task: str,
    seed: int,
    graph: int,
    role: str,
) -> np.random.Generator:
    digest = stable_hash(
        {
            "extension": EXTENSION_VERSION,
            "analysis_seed": int(analysis_seed),
            "task": str(task),
            "seed": int(seed),
            "graph": int(graph),
            "role": str(role),
        },
        length=16,
    )
    return np.random.default_rng(int(digest, 16) % (2**63 - 1))


def _counterfactual_variant(
    base: Any,
    donor_graph: Any,
    *,
    donor_node: int,
    role: str,
    records: int,
) -> tuple[Any, int, int]:
    """Return a valid NAR graph, original label, and exactly known counterfactual label."""

    import torch

    changed = base.clone()
    clean_label = int(base.y.reshape(-1)[0])
    if role == "query":
        source = int(base.query_idx)
        donor = donor_graph.x[int(donor_node)].clone()
        new_key = int(donor[0])
        if not (0 <= new_key < int(records)):
            raise RuntimeError("query donor does not contain a valid NAR address")
        target = 3 + new_key
        changed.x[source] = donor
        changed.target_idx = torch.as_tensor(
            target, dtype=base.target_idx.dtype, device=base.target_idx.device
        )
        counterfactual_label = int(base.x[target, 1])
    elif role == "value":
        source = int(base.target_idx)
        donor = donor_graph.x[int(donor_node)].clone()
        if int(donor[0]) != int(base.x[source, 0]):
            raise RuntimeError("value donor changed the requested key")
        changed.x[source] = donor
        counterfactual_label = int(donor[1])
    else:
        raise ValueError(role)
    changed.y = torch.as_tensor(
        [counterfactual_label],
        dtype=base.y.dtype,
        device=base.y.device,
    )
    return changed, clean_label, counterfactual_label


def build_counterfactual_plan(
    runtime: CounterfactualRuntime,
    *,
    seed: int,
    donors_per_role: int,
    analysis_seed: int,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[Any]]]:
    """Freeze disjoint causal events before any patched outcome is observed."""

    from ..methodology.sampling import node_degrees

    plan: dict[int, dict[str, Any]] = {}
    variants: dict[int, list[Any]] = {}
    for graph_id in runtime.splits.causal:
        base = runtime.eval_ds[int(graph_id)]
        degrees = node_degrees(base)
        graph_rows: list[dict[str, Any]] = []
        graph_variants: list[Any] = []
        for role, source in (
            ("query", int(base.query_idx)),
            ("value", int(base.target_idx)),
        ):
            payload = tuple(base.x[source].detach().cpu().numpy().reshape(-1).tolist())
            selected = runtime.donor_pool.draw(
                payload,
                int(degrees[source]),
                int(donors_per_role),
                _counterfactual_rng(
                    analysis_seed=analysis_seed,
                    task=runtime.task.name,
                    seed=seed,
                    graph=int(graph_id),
                    role=role,
                ),
                base_graph_id=None,
            )
            for draw, donor in enumerate(selected):
                donor_graph = runtime.donor_ds[int(donor.graph_id)]
                changed, clean_label, counterfactual_label = _counterfactual_variant(
                    base,
                    donor_graph,
                    donor_node=int(donor.node),
                    role=role,
                    records=int(runtime.spec.records),
                )
                graph_variants.append(changed)
                graph_rows.append(
                    {
                        "graph_id": int(graph_id),
                        "role": role,
                        "source": int(source),
                        "draw": int(draw),
                        "donor_graph_id": int(donor.graph_id),
                        "donor_node": int(donor.node),
                        "clean_label": int(clean_label),
                        "counterfactual_label": int(counterfactual_label),
                        "label_changed": bool(clean_label != counterfactual_label),
                        "payload_fingerprint": stable_hash(
                            {
                                "payload": donor.payload,
                                "role": role,
                                "clean_label": clean_label,
                                "counterfactual_label": counterfactual_label,
                            }
                        ),
                    }
                )
        plan[int(graph_id)] = {
            "records": tuple(graph_rows),
            "manifest_hash": stable_hash(graph_rows),
        }
        variants[int(graph_id)] = graph_variants
    return plan, variants


def _concat_replacements(
    backend: CanonicalNarBackend,
    captures: Sequence[Any],
    indices: Sequence[Sequence[int]],
) -> tuple[Any, ...]:
    import torch

    per_group = [
        backend.replacement_batch(capture, group_indices)
        for capture, group_indices in zip(captures, indices)
    ]
    return tuple(
        torch.cat([rows[layer] for rows in per_group], dim=0)
        for layer in range(len(per_group[0]))
    )


def _margin(logits: Any, clean: np.ndarray, counterfactual: np.ndarray) -> np.ndarray:
    rows = logits.detach().cpu().numpy()
    index = np.arange(len(clean), dtype=np.int64)
    return rows[index, counterfactual] - rows[index, clean]


def counterfactual_estimand(
    *,
    clean_margin: Any,
    counterfactual_margin: Any,
    injected_margin: Any,
    restored_margin: Any,
    label_changed: Any,
    effect_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric targeted mediation and its unnormalised directional numerator."""

    clean = np.asarray(clean_margin, dtype=np.float64)
    counterfactual = np.asarray(counterfactual_margin, dtype=np.float64)
    injected = np.asarray(injected_margin, dtype=np.float64)
    restored = np.asarray(restored_margin, dtype=np.float64)
    changed = np.asarray(label_changed, dtype=bool)
    denominator = counterfactual - clean
    raw = 0.5 * ((injected - clean) + (counterfactual - restored))
    estimable = changed & np.isfinite(denominator) & (denominator > float(effect_floor))
    mediation = np.full_like(raw, np.nan, dtype=np.float64)
    np.divide(raw, denominator, out=mediation, where=estimable)
    return mediation, raw, estimable


def _counterfactual_contract(
    binding: SourceBinding,
    role_result: Mapping[str, Any],
    *,
    donors_per_role: int,
    causal_graphs: Sequence[int],
    numerical: NumericalPolicy,
    accuracy_gate: float,
    analysis_seed: int,
) -> ExtensionContract:
    family_manifest = {
        name: [list(head) for head in values]
        for name, values in role_result["families"].items()
    }
    return ExtensionContract(
        stage="counterfactual",
        task=binding.task,
        seed=binding.seed,
        checkpoint_sha256=binding.checkpoint_sha256,
        source_artifact_sha256=binding.score_artifact.file_sha256,
        source_contract_fingerprint=binding.source_contract_fingerprint,
        scientific_parameters={
            "donors_per_role": int(donors_per_role),
            "causal_graphs": [int(graph) for graph in causal_graphs],
            "effect_floor": float(numerical.effect_floor),
            "accuracy_gate": float(accuracy_gate),
            "analysis_seed": int(analysis_seed),
            "family_manifest": family_manifest,
            "estimand": (
                "0.5*((m(inject)-m(clean))+(m(cf)-m(restore)))"
                "/(m(cf)-m(clean))"
            ),
            "margin": "logit[y_counterfactual]-logit[y_clean]",
        },
    )


def run_counterfactual_cell(
    runtime: CounterfactualRuntime,
    *,
    binding: SourceBinding,
    role_result: Mapping[str, Any],
    store: ExtensionStore,
    numerical: NumericalPolicy,
    donors_per_role: int,
    graphs_per_batch: int,
    accuracy_gate: float,
    analysis_seed: int,
) -> dict[str, Any]:
    """Run batched capture/inject/restore inference for one model/N/seed cell."""

    contract = _counterfactual_contract(
        binding,
        role_result,
        donors_per_role=donors_per_role,
        causal_graphs=runtime.splits.causal,
        numerical=numerical,
        accuracy_gate=accuracy_gate,
        analysis_seed=analysis_seed,
    )
    consolidated = store.load(contract, "results")
    if consolidated is not None:
        return consolidated
    plan, variants = build_counterfactual_plan(
        runtime,
        seed=binding.seed,
        donors_per_role=donors_per_role,
        analysis_seed=analysis_seed,
    )
    rows: list[dict[str, Any]] = []
    graph_ids = sorted(plan)
    missing_graph_ids: list[int] = []
    for graph_id in graph_ids:
        shard = store.load(contract, f"graph_{int(graph_id):06d}")
        if shard is None:
            missing_graph_ids.append(int(graph_id))
        else:
            rows.extend(shard["rows"])

    def execute_chunk(chunk: Sequence[int]) -> dict[int, dict[str, Any]]:
        missing = [int(graph_id) for graph_id in chunk]
        groups = [
            [runtime.eval_ds[graph_id], *variants[graph_id]]
            for graph_id in missing
        ]
        captures = runtime.backend.capture_groups(groups)
        flat_clean: list[Any] = []
        flat_counterfactual: list[Any] = []
        clean_logits: list[Any] = []
        counterfactual_logits: list[Any] = []
        metadata: list[dict[str, Any]] = []
        cf_indices: list[list[int]] = []
        clean_indices: list[list[int]] = []
        for graph_id, capture in zip(missing, captures):
            records = list(plan[graph_id]["records"])
            count = len(records)
            flat_clean.extend([runtime.eval_ds[graph_id]] * count)
            flat_counterfactual.extend(variants[graph_id])
            clean_logits.append(capture.z[0:1].expand(count, -1))
            counterfactual_logits.append(capture.z[1:])
            metadata.extend(records)
            cf_indices.append(list(range(1, count + 1)))
            clean_indices.append([0] * count)
        import torch

        clean_logits_tensor = torch.cat(clean_logits, dim=0)
        counterfactual_logits_tensor = torch.cat(counterfactual_logits, dim=0)
        cf_replacements = _concat_replacements(
            runtime.backend, captures, cf_indices
        )
        clean_replacements = _concat_replacements(
            runtime.backend, captures, clean_indices
        )
        patched: dict[str, tuple[Any, Any]] = {}
        for family_name, family in role_result["families"].items():
            if not family:
                continue
            injected, _, _ = runtime.backend.patch_many(
                flat_clean, cf_replacements, family
            )
            restored, _, _ = runtime.backend.patch_many(
                flat_counterfactual, clean_replacements, family
            )
            patched[family_name] = (injected, restored)
        clean_labels = np.asarray(
            [int(row["clean_label"]) for row in metadata], dtype=np.int64
        )
        counterfactual_labels = np.asarray(
            [int(row["counterfactual_label"]) for row in metadata], dtype=np.int64
        )
        label_changed = clean_labels != counterfactual_labels
        clean_margin = _margin(
            clean_logits_tensor, clean_labels, counterfactual_labels
        )
        cf_margin = _margin(
            counterfactual_logits_tensor, clean_labels, counterfactual_labels
        )
        all_rows: list[dict[str, Any]] = []
        clean_numpy = clean_logits_tensor.detach().cpu().numpy()
        cf_numpy = counterfactual_logits_tensor.detach().cpu().numpy()
        for family_name, (injected_logits, restored_logits) in patched.items():
            injected_margin = _margin(
                injected_logits, clean_labels, counterfactual_labels
            )
            restored_margin = _margin(
                restored_logits, clean_labels, counterfactual_labels
            )
            mediation, raw, estimable = counterfactual_estimand(
                clean_margin=clean_margin,
                counterfactual_margin=cf_margin,
                injected_margin=injected_margin,
                restored_margin=restored_margin,
                label_changed=label_changed,
                effect_floor=numerical.effect_floor,
            )
            injected_numpy = injected_logits.detach().cpu().numpy()
            restored_numpy = restored_logits.detach().cpu().numpy()
            for index, record in enumerate(metadata):
                all_rows.append(
                    {
                        **record,
                        "family": family_name,
                        "clean_margin": float(clean_margin[index]),
                        "counterfactual_margin": float(cf_margin[index]),
                        "injected_margin": float(injected_margin[index]),
                        "restored_margin": float(restored_margin[index]),
                        "full_margin_change": float(
                            cf_margin[index] - clean_margin[index]
                        ),
                        "directional_numerator": float(raw[index]),
                        "targeted_mediation": float(mediation[index]),
                        "estimable": bool(estimable[index]),
                        "clean_correct": bool(
                            int(np.argmax(clean_numpy[index]))
                            == int(clean_labels[index])
                        ),
                        "clean_prediction": int(np.argmax(clean_numpy[index])),
                        "counterfactual_correct": bool(
                            int(np.argmax(cf_numpy[index]))
                            == int(counterfactual_labels[index])
                        ),
                        "counterfactual_prediction": int(
                            np.argmax(cf_numpy[index])
                        ),
                        "injected_counterfactual_prediction": bool(
                            int(np.argmax(injected_numpy[index]))
                            == int(counterfactual_labels[index])
                        ),
                        "injected_prediction": int(
                            np.argmax(injected_numpy[index])
                        ),
                        "restored_clean_prediction": bool(
                            int(np.argmax(restored_numpy[index]))
                            == int(clean_labels[index])
                        ),
                        "restored_prediction": int(
                            np.argmax(restored_numpy[index])
                        ),
                        "invariance_target_logit_drift": (
                            float(
                                0.5
                                * (
                                    abs(
                                        injected_numpy[
                                            index, int(clean_labels[index])
                                        ]
                                        - clean_numpy[index, int(clean_labels[index])]
                                    )
                                    + abs(
                                        cf_numpy[index, int(clean_labels[index])]
                                        - restored_numpy[
                                            index, int(clean_labels[index])
                                        ]
                                    )
                                )
                            )
                            if not label_changed[index]
                            else float("nan")
                        ),
                    }
                )
        result: dict[int, dict[str, Any]] = {}
        for graph_id in missing:
            selected = [
                row for row in all_rows if int(row["graph_id"]) == int(graph_id)
            ]
            result[int(graph_id)] = {
                "manifest": plan[graph_id],
                "rows": selected,
            }
        return result

    def consume_chunk(result: Mapping[int, Mapping[str, Any]]) -> None:
        for graph_id in sorted(result):
            shard = result[graph_id]
            store.save(
                contract,
                f"graph_{int(graph_id):06d}",
                dict(shard),
            )
            rows.extend(shard["rows"])

    def report_batch(cursor: int, total: int, batch_size: int) -> None:
        memory = ""
        try:
            if torch.cuda.is_available():
                memory = (
                    f" | CUDA peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
                    f", reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB"
                )
        except (NameError, RuntimeError):
            pass
        print(
            f"[counterfactual] {binding.task}:seed{binding.seed} "
            f"completed {cursor}/{total} missing graphs (batch={batch_size}){memory}",
            flush=True,
        )
    if missing_graph_ids:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            pass
    from ..methodology.execution import execute_graph_batches

    execution_report = execute_graph_batches(
        missing_graph_ids,
        graphs_per_batch=int(graphs_per_batch),
        execute=execute_chunk,
        consume=consume_chunk,
        oom_backoff=True,
        on_batch=report_batch,
    )
    rows.sort(
        key=lambda row: (
            int(row["graph_id"]),
            str(row["role"]),
            int(row["draw"]),
            str(row["family"]),
        )
    )
    output = {
        "extension_version": EXTENSION_VERSION,
        "task": binding.task,
        "seed": int(binding.seed),
        "checkpoint_sha256": runtime.checkpoint_sha256,
        "heldout_accuracy": float(runtime.heldout_accuracy),
        "primary": bool(runtime.heldout_accuracy >= float(accuracy_gate)),
        "accuracy_gate": float(accuracy_gate),
        "family_manifest": role_result["families"],
        "plan_hash": stable_hash(
            {
                graph: plan[graph]["manifest_hash"]
                for graph in sorted(plan)
            }
        ),
        "rows": rows,
        "graphs": len(graph_ids),
        "donors_per_role": int(donors_per_role),
        "execution": dataclasses.asdict(execution_report),
        "cache_hits": len(graph_ids) - len(missing_graph_ids),
    }
    store.save(contract, "results", output)
    return output


def run_counterfactual_analysis(
    *,
    store: ExtensionStore,
    base_canonical_root: Path,
    extension_root: Path,
    training_run_dir: Path,
    models: Sequence[str],
    ns: Sequence[int],
    cached_ns: Sequence[int],
    seeds: Sequence[int],
    width: int,
    sizes: RunSizes,
    numerical: NumericalPolicy,
    families: FamilyPolicy,
    bootstrap: BootstrapPolicy,
    analysis_seed: int,
    accelerator: str,
    donors_per_role: int,
    graphs_per_batch: int,
    accuracy_gate: float,
    role_results: Mapping[tuple[str, int, int], Mapping[str, Any]] | None = None,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    if role_results is None:
        role_results = run_role_derivation(
            store=store,
            base_canonical_root=base_canonical_root,
            extension_root=extension_root,
            models=models,
            ns=ns,
            cached_ns=cached_ns,
            seeds=seeds,
            numerical=numerical,
            families=families,
            bootstrap=bootstrap,
            analysis_seed=analysis_seed,
        )
    results: dict[tuple[str, int, int], dict[str, Any]] = {}
    table: list[dict[str, Any]] = []
    for records in ns:
        for model_name in models:
            for seed in seeds:
                key = (str(model_name), int(records), int(seed))
                binding = load_source_binding(
                    base_canonical_root=base_canonical_root,
                    extension_root=extension_root,
                    cached_ns=cached_ns,
                    model_name=model_name,
                    records=records,
                    seed=seed,
                )
                runtime = load_counterfactual_runtime(
                    training_run_dir=training_run_dir,
                    model_name=model_name,
                    records=records,
                    seed=seed,
                    width=width,
                    sizes=sizes,
                    analysis_seed=analysis_seed,
                    accelerator=accelerator,
                )
                if runtime.checkpoint_sha256 != binding.checkpoint_sha256:
                    raise StaleCacheError(
                        f"checkpoint changed since discovery for {binding.task}:seed{seed}"
                    )
                result = run_counterfactual_cell(
                    runtime,
                    binding=binding,
                    role_result=role_results[key],
                    store=store,
                    numerical=numerical,
                    donors_per_role=donors_per_role,
                    graphs_per_batch=graphs_per_batch,
                    accuracy_gate=accuracy_gate,
                    analysis_seed=analysis_seed,
                )
                results[key] = result
                for row in result["rows"]:
                    table.append(
                        {
                            "model": model_name,
                            "N": int(records),
                            "seed": int(seed),
                            "heldout_accuracy": result["heldout_accuracy"],
                            "primary": result["primary"],
                            **row,
                        }
                    )
                del runtime
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
    _write_csv(extension_root / "tables" / "counterfactual_events.csv", table)
    return results


def load_cached_role_results(
    *,
    store: ExtensionStore,
    base_canonical_root: Path,
    extension_root: Path,
    models: Sequence[str],
    ns: Sequence[int],
    cached_ns: Sequence[int],
    seeds: Sequence[int],
    numerical: NumericalPolicy,
    families: FamilyPolicy,
    bootstrap: BootstrapPolicy,
    analysis_seed: int,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    results: dict[tuple[str, int, int], dict[str, Any]] = {}
    for records in ns:
        for model_name in models:
            for seed in seeds:
                binding = load_source_binding(
                    base_canonical_root=base_canonical_root,
                    extension_root=extension_root,
                    cached_ns=cached_ns,
                    model_name=model_name,
                    records=records,
                    seed=seed,
                )
                contract = _role_contract(
                    binding,
                    numerical=numerical,
                    families=families,
                    bootstrap=bootstrap,
                    analysis_seed=analysis_seed,
                )
                result = store.load(contract, "role_scores")
                if result is None:
                    raise FileNotFoundError(
                        f"missing role cache for {binding.task}:seed{seed}; run --phase roles"
                    )
                results[(str(model_name), int(records), int(seed))] = result
    return results


def load_cached_counterfactual_results(
    *,
    store: ExtensionStore,
    base_canonical_root: Path,
    extension_root: Path,
    models: Sequence[str],
    ns: Sequence[int],
    cached_ns: Sequence[int],
    seeds: Sequence[int],
    role_results: Mapping[tuple[str, int, int], Mapping[str, Any]],
    sizes: RunSizes,
    numerical: NumericalPolicy,
    donors_per_role: int,
    accuracy_gate: float,
    analysis_seed: int,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    splits = deterministic_splits(
        int(sizes.discovery_graphs)
        + int(sizes.causal_graphs)
        + int(sizes.clean_ablation_graphs),
        int(sizes.semantic_donor_graphs),
        sizes,
        int(analysis_seed),
        same_index_space=False,
    )
    results: dict[tuple[str, int, int], dict[str, Any]] = {}
    for records in ns:
        for model_name in models:
            for seed in seeds:
                key = (str(model_name), int(records), int(seed))
                binding = load_source_binding(
                    base_canonical_root=base_canonical_root,
                    extension_root=extension_root,
                    cached_ns=cached_ns,
                    model_name=model_name,
                    records=records,
                    seed=seed,
                )
                contract = _counterfactual_contract(
                    binding,
                    role_results[key],
                    donors_per_role=donors_per_role,
                    causal_graphs=splits.causal,
                    numerical=numerical,
                    accuracy_gate=accuracy_gate,
                    analysis_seed=analysis_seed,
                )
                result = store.load(contract, "results")
                if result is None:
                    raise FileNotFoundError(
                        f"missing counterfactual cache for {binding.task}:seed{seed}; "
                        "run --phase counterfactual"
                    )
                results[key] = result
    return results


def _t_interval(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, mean, mean
    # The paper design fixes three trained seeds (two-sided 95% Student-t, df=2).
    critical = 4.302652729911275 if len(array) == 3 else 1.96
    half = float(critical * np.std(array, ddof=1) / math.sqrt(len(array)))
    return mean, mean - half, mean + half


def plot_role_specialisation(
    role_results: Mapping[tuple[str, int, int], Mapping[str, Any]],
    *,
    figure_dir: Path,
    table_dir: Path,
    models: Sequence[str],
    seeds: Sequence[int],
    records: int,
    stem: str = ROLE_FIGURE_STEM,
    table_name: str = "figure_06_head_roles.csv",
    headline: bool = True,
) -> None:
    """Figure 1: interpretable address/content roles for every head and seed."""

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows: list[dict[str, Any]] = []
    max_layers = max(
        int(
            role_results[(str(model), int(records), int(seed))][
                "role_coordinates"
            ].raw_query.shape[0]
        )
        for model in models
        for seed in seeds
    )
    cmap = mpl.colormaps["viridis"]
    norm = mpl.colors.Normalize(vmin=0, vmax=max(max_layers - 1, 1))
    with _style():
        fig, axes = plt.subplots(
            1,
            len(models),
            figsize=(3.35 * len(models), 3.45),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for axis, model_name in zip(axes, models):
            axis.axline(
                (0, 0),
                slope=1,
                color="#777777",
                linestyle="--",
                linewidth=0.8,
                alpha=0.75,
                zorder=0,
            )
            for seed_index, seed in enumerate(seeds):
                result = role_results[(str(model_name), int(records), int(seed))]
                role: RoleCoordinates = result["role_coordinates"]
                interval = result["interval"]
                membership = {
                    tuple(head): name
                    for name, heads in result["families"].items()
                    for head in heads
                }
                x = role.normalized_value
                y = role.normalized_query
                x_low = np.asarray(interval["low"])[3]
                x_high = np.asarray(interval["high"])[3]
                y_low = np.asarray(interval["low"])[2]
                y_high = np.asarray(interval["high"])[2]
                for layer in range(x.shape[0]):
                    for head in range(x.shape[1]):
                        axis.errorbar(
                            float(x[layer, head]),
                            float(y[layer, head]),
                            xerr=np.asarray(
                                [
                                    [max(0.0, x[layer, head] - x_low[layer, head])],
                                    [max(0.0, x_high[layer, head] - x[layer, head])],
                                ]
                            ),
                            yerr=np.asarray(
                                [
                                    [max(0.0, y[layer, head] - y_low[layer, head])],
                                    [max(0.0, y_high[layer, head] - y[layer, head])],
                                ]
                            ),
                            fmt=SEED_MARKERS[seed_index % len(SEED_MARKERS)],
                            color=cmap(norm(layer)),
                            markersize=4.4,
                            markeredgecolor="white",
                            markeredgewidth=0.35,
                            elinewidth=0.35,
                            alpha=0.82,
                            capsize=0,
                            zorder=2,
                        )
                        rows.append(
                            {
                                "model": model_name,
                                "N": int(records),
                                "seed": int(seed),
                                "layer": int(layer),
                                "head": int(head),
                                "normalized_content": float(x[layer, head]),
                                "normalized_address": float(y[layer, head]),
                                "joint_sensitivity": float(
                                    role.joint_sensitivity[layer, head]
                                ),
                                "role_selectivity": float(
                                    role.selectivity[layer, head]
                                ),
                                "active": bool(role.active[layer, head]),
                                "frozen_family": membership.get((layer, head), ""),
                                "R_role": float(role.role_separation),
                            }
                        )
            axis.set_title(MODEL_LABELS[str(model_name)])
            axis.set_xlabel(r"Content sensitivity  $S_{\mathrm{value}}/\bar S_{\mathrm{value}}$")
            axis.set_aspect("equal", adjustable="box")
            axis.text(
                0.03,
                0.96,
                "address-leaning",
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="#2F6B9A",
                fontsize=7.5,
            )
            axis.text(
                0.97,
                0.04,
                "content-leaning",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                color="#B33A3A",
                fontsize=7.5,
            )
        axes[0].set_ylabel(
            r"Address sensitivity  $S_{\mathrm{query}}/\bar S_{\mathrm{query}}$"
        )
        seed_handles = [
            Line2D(
                [0],
                [0],
                marker=SEED_MARKERS[index % len(SEED_MARKERS)],
                linestyle="none",
                markerfacecolor="#666666",
                markeredgecolor="white",
                label=f"Seed {seed}",
                markersize=6,
            )
            for index, seed in enumerate(seeds)
        ]
        axes[0].legend(handles=seed_handles, loc="upper left", fontsize=8)
        scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(
            scalar,
            ax=axes,
            location="right",
            shrink=0.82,
            pad=0.02,
            ticks=range(max_layers),
        )
        colorbar.set_label("Layer")
        fig.suptitle(
            (
                "Head specialisation separates address and content processing"
                if headline
                else f"Head-role specialisation at memory size N={int(records)}"
            ),
            fontsize=13,
        )
        _save_figure(
            fig,
            figure_dir,
            stem,
            {
                "extension_version": EXTENSION_VERSION,
                "N": int(records),
                "models": list(models),
                "seeds": list(seeds),
                "point": "one attention head",
                "x": "normalised requested-record value sensitivity",
                "y": "normalised query-key sensitivity",
                "interval": (
                    "95% graph/donor nested percentile interval; source fixed because each "
                    "role has one task-defined source per graph"
                ),
                "dashed_line": "equal address and content sensitivity",
            },
        )
        plt.close(fig)
    _write_csv(table_dir / table_name, rows)


def _counterfactual_summary(
    results: Mapping[tuple[str, int, int], Mapping[str, Any]],
    *,
    models: Sequence[str],
    ns: Sequence[int],
    bootstrap: BootstrapPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    double_rows: list[dict[str, Any]] = []
    rng_offset = 0
    for model_name in models:
        for records in ns:
            cell_rows: list[dict[str, Any]] = []
            primary = False
            for (model, n_value, seed), result in results.items():
                if str(model) != str(model_name) or int(n_value) != int(records):
                    continue
                primary |= bool(result["primary"])
                cell_rows.extend(
                    [
                        {
                            **row,
                            "seed": int(seed),
                            "primary_checkpoint": bool(result["primary"]),
                        }
                        for row in result["rows"]
                    ]
                )
            if not cell_rows:
                continue
            for role in ("query", "value"):
                for family in (
                    "address",
                    "content",
                    "address_control",
                    "content_control",
                    "matched_control",
                ):
                    chosen = [
                        row
                        for row in cell_rows
                        if row["role"] == role
                        and (
                            row["family"] == family
                            if family != "matched_control"
                            else row["family"]
                            in {"address_control", "content_control"}
                        )
                        and bool(row["primary_checkpoint"])
                        and bool(row["estimable"])
                        and np.isfinite(float(row["targeted_mediation"]))
                    ]
                    if not chosen:
                        continue
                    observations = [
                        Observation(
                            seed=int(row["seed"]),
                            graph=int(row["graph_id"]),
                            source=0,
                            donor=int(row["draw"]),
                            value=float(row["targeted_mediation"]),
                        )
                        for row in chosen
                    ]
                    rng_offset += 1
                    policy = dataclasses.replace(
                        bootstrap,
                        rng_seed=int(bootstrap.rng_seed) + rng_offset,
                        resample_source=False,
                    )
                    interval = nested_percentile_interval(observations, policy)
                    summaries.append(
                        {
                            "model": model_name,
                            "N": int(records),
                            "role": role,
                            "family": family,
                            "primary": bool(primary),
                            "events": len(chosen),
                            "graphs": len({(row["seed"], row["graph_id"]) for row in chosen}),
                            "estimate": float(interval.estimate),
                            "ci95_low": float(interval.low),
                            "ci95_high": float(interval.high),
                        }
                    )
            # The double dissociation is evaluated on graph summaries to preserve role pairing
            # without falsely pairing independently sampled query/value donors.
            vectors: list[Observation] = []
            for seed in sorted({int(row["seed"]) for row in cell_rows}):
                for graph in sorted(
                    {
                        int(row["graph_id"])
                        for row in cell_rows
                        if int(row["seed"]) == seed
                    }
                ):
                    vector = []
                    for role, family in (
                        ("query", "address"),
                        ("query", "content"),
                        ("value", "address"),
                        ("value", "content"),
                    ):
                        values = [
                            float(row["targeted_mediation"])
                            for row in cell_rows
                            if int(row["seed"]) == seed
                            and int(row["graph_id"]) == graph
                            and bool(row["primary_checkpoint"])
                            and row["role"] == role
                            and row["family"] == family
                            and bool(row["estimable"])
                            and np.isfinite(float(row["targeted_mediation"]))
                        ]
                        if not values:
                            break
                        vector.append(float(np.mean(values)))
                    if len(vector) == 4:
                        vectors.append(
                            Observation(seed, graph, 0, 0, np.asarray(vector))
                        )
            if vectors:
                rng_offset += 1
                policy = dataclasses.replace(
                    bootstrap,
                    rng_seed=int(bootstrap.rng_seed) + rng_offset,
                    resample_source=False,
                    resample_donor=False,
                )
                interval = nested_percentile_interval(
                    vectors,
                    policy,
                    transform=lambda value: np.asarray(
                        (value[0] - value[1]) - (value[2] - value[3])
                    ),
                )
                double_rows.append(
                    {
                        "model": model_name,
                        "N": int(records),
                        "primary": bool(primary),
                        "graphs": len(vectors),
                        "double_dissociation": float(interval.estimate),
                        "ci95_low": float(interval.low),
                        "ci95_high": float(interval.high),
                    }
                )
    return summaries, double_rows


def _query_invariance_summary(
    results: Mapping[tuple[str, int, int], Mapping[str, Any]],
    *,
    bootstrap: BootstrapPolicy,
) -> list[dict[str, Any]]:
    """Summarise query swaps whose exact answer is unchanged."""

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for (model, records, seed), result in results.items():
        if not bool(result["primary"]):
            continue
        for row in result["rows"]:
            if row["role"] != "query" or bool(row["label_changed"]):
                continue
            grouped.setdefault(
                (str(model), int(records), str(row["family"])), []
            ).append({**row, "seed": int(seed)})
    output: list[dict[str, Any]] = []
    for index, (key, rows) in enumerate(sorted(grouped.items())):
        observations = [
            Observation(
                seed=int(row["seed"]),
                graph=int(row["graph_id"]),
                source=0,
                donor=int(row["draw"]),
                value=float(row["invariance_target_logit_drift"]),
            )
            for row in rows
        ]
        interval = nested_percentile_interval(
            observations,
            dataclasses.replace(
                bootstrap,
                rng_seed=int(bootstrap.rng_seed) + 50_000 + index,
                resample_source=False,
            ),
        )
        output.append(
            {
                "model": key[0],
                "N": key[1],
                "family": key[2],
                "events": len(rows),
                "target_logit_drift": float(interval.estimate),
                "target_logit_drift_ci95_low": float(interval.low),
                "target_logit_drift_ci95_high": float(interval.high),
                "injected_prediction_flip_rate": float(
                    np.mean(
                        [
                            int(row["injected_prediction"])
                            != int(row["clean_prediction"])
                            for row in rows
                        ]
                    )
                ),
                "restored_prediction_flip_rate": float(
                    np.mean(
                        [
                            int(row["restored_prediction"])
                            != int(row["counterfactual_prediction"])
                            for row in rows
                        ]
                    )
                ),
            }
        )
    return output


def plot_counterfactual_validation(
    results: Mapping[tuple[str, int, int], Mapping[str, Any]],
    *,
    figure_dir: Path,
    table_dir: Path,
    models: Sequence[str],
    ns: Sequence[int],
    bootstrap: BootstrapPolicy,
) -> None:
    """Figure 2: exact counterfactual mediation and the address/content double dissociation."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    summaries, double_rows = _counterfactual_summary(
        results, models=models, ns=ns, bootstrap=bootstrap
    )
    invariance_rows = _query_invariance_summary(results, bootstrap=bootstrap)
    primary = [row for row in summaries if bool(row["primary"])]
    if not primary:
        raise RuntimeError("no counterfactual cell passed the predeclared accuracy gate")
    family_positions = {"address": 0.0, "content": 1.0, "control": 2.0}
    marker_by_n = {
        int(records): ("o", "s", "D", "^", "P", "X")[index % 6]
        for index, records in enumerate(ns)
    }
    plotted_rows: list[dict[str, Any]] = []
    with _style():
        fig, axes = plt.subplots(
            1, 2, figsize=(7.15, 3.55), sharey=True, constrained_layout=True
        )
        for axis, role in zip(axes, ("query", "value")):
            axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
            for model_index, model_name in enumerate(models):
                model_offset = (model_index - (len(models) - 1) / 2.0) * 0.11
                for records in ns:
                    rows = [
                        row
                        for row in primary
                        if str(row["model"]) == str(model_name)
                        and int(row["N"]) == int(records)
                        and row["role"] == role
                    ]
                    by_family = {row["family"]: row for row in rows}
                    entries: list[tuple[str, Mapping[str, Any]]] = []
                    for family in ("address", "content", "matched_control"):
                        if family in by_family:
                            display = "control" if family == "matched_control" else family
                            entries.append((display, by_family[family]))
                    n_rank = list(ns).index(records)
                    n_offset = (n_rank - (len(ns) - 1) / 2.0) * 0.025
                    for family, row in entries:
                        x = family_positions[family] + model_offset + n_offset
                        estimate = float(row["estimate"])
                        axis.errorbar(
                            x,
                            estimate,
                            yerr=np.asarray(
                                [
                                    [max(0.0, estimate - float(row["ci95_low"]))],
                                    [max(0.0, float(row["ci95_high"]) - estimate)],
                                ]
                            ),
                            fmt=marker_by_n[int(records)],
                            color=MODEL_COLOURS[str(model_name)],
                            markeredgecolor="white",
                            markeredgewidth=0.5,
                            markersize=5.5,
                            capsize=2,
                            linewidth=1,
                            zorder=3,
                        )
                        plotted_rows.append(
                            {
                                "role": role,
                                "model": model_name,
                                "N": int(records),
                                "display_family": family,
                                **dict(row),
                            }
                        )
            axis.set_xticks([0, 1, 2])
            axis.set_xticklabels(
                ["Address\nfamily", "Content\nfamily", "Matched\ncontrols"]
            )
            axis.set_title(
                "Query-address counterfactual"
                if role == "query"
                else "Target-value counterfactual"
            )
        axes[0].set_ylabel("Targeted counterfactual mediation")
        model_handles = [
            Line2D(
                [0],
                [0],
                color=MODEL_COLOURS[str(model)],
                marker="o",
                linestyle="none",
                label=MODEL_LABELS[str(model)],
            )
            for model in models
        ]
        n_handles = [
            Line2D(
                [0],
                [0],
                color="#555555",
                marker=marker_by_n[int(records)],
                linestyle="none",
                label=f"N={records}",
            )
            for records in ns
        ]
        axes[0].legend(
            handles=model_handles + n_handles,
            ncol=2,
            fontsize=7.5,
            loc="upper left",
        )
        fig.suptitle(
            "Specialised head families mediate exact counterfactual recall",
            fontsize=13,
        )
        _save_figure(
            fig,
            figure_dir,
            COUNTERFACTUAL_FIGURE_STEM,
            {
                "extension_version": EXTENSION_VERSION,
                "primary_filter": "checkpoint held-out accuracy >= configured gate",
                "estimand": (
                    "symmetric injection/restoration mediation of the exact "
                    "counterfactual-vs-clean logit margin"
                ),
                "families": "frozen from disjoint discovery role scores",
                "controls": "same-layer J/throughput-matched central active heads",
                "error_bars": "95% nested seed/graph/donor percentile intervals",
                "double_dissociation_table": "figure_07_double_dissociation.csv",
            },
        )
        plt.close(fig)
    _write_csv(table_dir / "figure_07_counterfactual_summary.csv", summaries)
    _write_csv(table_dir / "figure_07_counterfactual_plotted.csv", plotted_rows)
    _write_csv(table_dir / "figure_07_double_dissociation.csv", double_rows)
    _write_csv(table_dir / "figure_07_query_invariance_control.csv", invariance_rows)


def _excess_accuracy(accuracy: float, records: int) -> float:
    chance = 1.0 / float(records)
    return float((float(accuracy) - chance) / (1.0 - chance))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    left = _rankdata(np.asarray(x, dtype=np.float64))
    right = _rankdata(np.asarray(y, dtype=np.float64))
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _cluster_spearman_interval(
    trajectories: Mapping[tuple[str, int], Sequence[tuple[float, float]]],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float, float]:
    keys = sorted(trajectories)

    def evaluate(selected: Iterable[tuple[str, int]]) -> float:
        points = [point for key in selected for point in trajectories[key]]
        return _spearman([point[0] for point in points], [point[1] for point in points])

    point = evaluate(keys)
    rng = np.random.default_rng(int(seed))
    draws = np.asarray(
        [
            evaluate([keys[int(index)] for index in rng.integers(0, len(keys), len(keys))])
            for _ in range(int(replicates))
        ],
        dtype=np.float64,
    )
    return point, float(np.nanquantile(draws, 0.025)), float(
        np.nanquantile(draws, 0.975)
    )


def plot_performance_transition(
    role_results: Mapping[tuple[str, int, int], Mapping[str, Any]],
    performance_rows: Sequence[Mapping[str, Any]],
    *,
    figure_dir: Path,
    table_dir: Path,
    models: Sequence[str],
    ns: Sequence[int],
    seeds: Sequence[int],
    bootstrap_seed: int,
) -> None:
    """Figure 3: co-transition of recall accuracy and head-role organisation."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    accuracy_by_key = {
        (str(row["model"]), int(row["N"]), int(row["seed"])): float(row["accuracy"])
        for row in performance_rows
    }
    transition_rows: list[dict[str, Any]] = []
    trajectories: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for model_name in models:
        for seed in seeds:
            points: list[tuple[float, float]] = []
            for left_n, right_n in zip(ns[:-1], ns[1:]):
                left_key = (str(model_name), int(left_n), int(seed))
                right_key = (str(model_name), int(right_n), int(seed))
                left_accuracy = _excess_accuracy(accuracy_by_key[left_key], int(left_n))
                right_accuracy = _excess_accuracy(accuracy_by_key[right_key], int(right_n))
                left_role = float(
                    role_results[left_key]["role_coordinates"].role_separation
                )
                right_role = float(
                    role_results[right_key]["role_coordinates"].role_separation
                )
                delta_accuracy = right_accuracy - left_accuracy
                delta_role = right_role - left_role
                points.append((delta_accuracy, delta_role))
                transition_rows.append(
                    {
                        "model": model_name,
                        "seed": int(seed),
                        "transition": f"{left_n}->{right_n}",
                        "N_left": int(left_n),
                        "N_right": int(right_n),
                        "delta_excess_accuracy": delta_accuracy,
                        "delta_R_role": delta_role,
                    }
                )
            trajectories[(str(model_name), int(seed))] = points
    rho, rho_low, rho_high = _cluster_spearman_interval(
        trajectories, seed=int(bootstrap_seed)
    )
    summary_rows: list[dict[str, Any]] = []
    transition_names = [f"{left}->{right}" for left, right in zip(ns[:-1], ns[1:])]
    transition_markers = {
        name: ("o", "s", "D", "^", "P", "X")[index % 6]
        for index, name in enumerate(transition_names)
    }
    with _style():
        fig, axes = plt.subplots(
            1, 3, figsize=(10.2, 3.35), constrained_layout=True
        )
        for model_name in models:
            accuracy_means, accuracy_errors = [], []
            role_means, role_errors = [], []
            for seed in seeds:
                seed_accuracy = [
                    _excess_accuracy(
                        accuracy_by_key[(str(model_name), int(records), int(seed))],
                        int(records),
                    )
                    for records in ns
                ]
                seed_role = [
                    float(
                        role_results[(str(model_name), int(records), int(seed))][
                            "role_coordinates"
                        ].role_separation
                    )
                    for records in ns
                ]
                axes[0].plot(
                    ns,
                    seed_accuracy,
                    color=MODEL_COLOURS[str(model_name)],
                    linewidth=0.55,
                    alpha=0.20,
                )
                axes[1].plot(
                    ns,
                    seed_role,
                    color=MODEL_COLOURS[str(model_name)],
                    linewidth=0.55,
                    alpha=0.20,
                )
            for records in ns:
                accuracy_values = [
                    _excess_accuracy(
                        accuracy_by_key[(str(model_name), int(records), int(seed))],
                        int(records),
                    )
                    for seed in seeds
                ]
                role_values = [
                    float(
                        role_results[(str(model_name), int(records), int(seed))][
                            "role_coordinates"
                        ].role_separation
                    )
                    for seed in seeds
                ]
                accuracy_mean, accuracy_low, accuracy_high = _t_interval(accuracy_values)
                role_mean, role_low, role_high = _t_interval(role_values)
                accuracy_means.append(accuracy_mean)
                accuracy_errors.append(
                    [accuracy_mean - accuracy_low, accuracy_high - accuracy_mean]
                )
                role_means.append(role_mean)
                role_errors.append([role_mean - role_low, role_high - role_mean])
                summary_rows.append(
                    {
                        "model": model_name,
                        "N": int(records),
                        "excess_accuracy": accuracy_mean,
                        "excess_accuracy_ci95_low": accuracy_low,
                        "excess_accuracy_ci95_high": accuracy_high,
                        "R_role": role_mean,
                        "R_role_ci95_low": role_low,
                        "R_role_ci95_high": role_high,
                    }
                )
                axes[0].scatter(
                    [records] * len(seeds),
                    accuracy_values,
                    color=MODEL_COLOURS[str(model_name)],
                    s=10,
                    alpha=0.32,
                    edgecolors="none",
                )
                axes[1].scatter(
                    [records] * len(seeds),
                    role_values,
                    color=MODEL_COLOURS[str(model_name)],
                    s=10,
                    alpha=0.32,
                    edgecolors="none",
                )
            axes[0].errorbar(
                ns,
                accuracy_means,
                yerr=np.asarray(accuracy_errors).T,
                color=MODEL_COLOURS[str(model_name)],
                marker=MODEL_MARKERS[str(model_name)],
                linewidth=1.5,
                capsize=2,
                label=MODEL_LABELS[str(model_name)],
            )
            axes[1].errorbar(
                ns,
                role_means,
                yerr=np.asarray(role_errors).T,
                color=MODEL_COLOURS[str(model_name)],
                marker=MODEL_MARKERS[str(model_name)],
                linewidth=1.5,
                capsize=2,
            )
        for row in transition_rows:
            axes[2].scatter(
                float(row["delta_excess_accuracy"]),
                float(row["delta_R_role"]),
                color=MODEL_COLOURS[str(row["model"])],
                marker=transition_markers[str(row["transition"])],
                s=25,
                alpha=0.82,
                edgecolor="white",
                linewidth=0.35,
            )
        axes[2].axhline(0, color="#888888", linestyle="--", linewidth=0.7)
        axes[2].axvline(0, color="#888888", linestyle="--", linewidth=0.7)
        axes[2].text(
            0.04,
            0.96,
            rf"$\rho_s={rho:.2f}$ [{rho_low:.2f}, {rho_high:.2f}]",
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
        )
        axes[0].set_title("A  Retrieval performance")
        axes[1].set_title("B  Head-role separation")
        axes[2].set_title("C  Adjacent-N transitions")
        axes[0].set_xlabel("Number of records, $N$")
        axes[1].set_xlabel("Number of records, $N$")
        axes[2].set_xlabel(r"$\Delta$ chance-adjusted accuracy")
        axes[0].set_ylabel("Chance-adjusted accuracy")
        axes[1].set_ylabel(r"Role separation  $R_{\mathrm{role}}$")
        axes[2].set_ylabel(r"$\Delta R_{\mathrm{role}}$")
        axes[0].set_xticks(ns)
        axes[1].set_xticks(ns)
        axes[0].legend(fontsize=8, loc="lower left")
        transition_handles = [
            Line2D(
                [0],
                [0],
                color="#555555",
                marker=transition_markers[name],
                linestyle="none",
                label=name,
                markersize=5,
            )
            for name in transition_names
        ]
        axes[2].legend(
            handles=transition_handles,
            title="$N$ transition",
            fontsize=7,
            title_fontsize=7.5,
            loc="lower right",
        )
        fig.suptitle(
            "Head-role organisation changes at retrieval-capacity transitions",
            fontsize=13,
        )
        _save_figure(
            fig,
            figure_dir,
            TRANSITION_FIGURE_STEM,
            {
                "extension_version": EXTENSION_VERSION,
                "role_separation": (
                    "0.5*sum_h |S_query/sum(S_query)-S_value/sum(S_value)|"
                ),
                "chance_adjustment": "(accuracy-1/N)/(1-1/N)",
                "panel_intervals": "95% Student-t intervals over three trained seeds",
                "transition_correlation": {
                    "spearman": rho,
                    "ci95_low": rho_low,
                    "ci95_high": rho_high,
                    "bootstrap": "2,000 trajectory-cluster resamples over model/seed",
                },
            },
        )
        plt.close(fig)
    _write_csv(table_dir / "figure_08_transition_summary.csv", summary_rows)
    _write_csv(table_dir / "figure_08_transition_deltas.csv", transition_rows)
    atomic_json(
        table_dir / "figure_08_transition_correlation.json",
        {
            "spearman": rho,
            "ci95_low": rho_low,
            "ci95_high": rho_high,
            "clusters": len(trajectories),
            "replicates": BOOTSTRAP_REPLICATES,
        },
    )


def _read_performance_rows(
    *,
    base_analysis_root: Path,
    training_run_dir: Path,
    models: Sequence[str],
    ns: Sequence[int],
    seeds: Sequence[int],
    width: int,
) -> list[dict[str, Any]]:
    path = base_analysis_root / "tables" / "checkpoint_performance.csv"
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    return checkpoint_payloads(
        training_run_dir=training_run_dir,
        models=models,
        ns=ns,
        seeds=seeds,
        width=int(width),
    )


def make_extension_figures(
    *,
    store: ExtensionStore,
    base_analysis_root: Path,
    base_canonical_root: Path,
    extension_root: Path,
    training_run_dir: Path,
    models: Sequence[str],
    transition_ns: Sequence[int],
    counterfactual_ns: Sequence[int],
    cached_ns: Sequence[int],
    all_performance_ns: Sequence[int],
    seeds: Sequence[int],
    width: int,
    role_figure_n: int,
    sizes: RunSizes,
    numerical: NumericalPolicy,
    families: FamilyPolicy,
    bootstrap: BootstrapPolicy,
    analysis_seed: int,
    donors_per_role: int,
    accuracy_gate: float,
    role_results: Mapping[tuple[str, int, int], Mapping[str, Any]] | None = None,
    counterfactual: Mapping[tuple[str, int, int], Mapping[str, Any]] | None = None,
) -> None:
    """Regenerate all paper artifacts using only protected cache files."""

    if role_results is None:
        role_results = load_cached_role_results(
            store=store,
            base_canonical_root=base_canonical_root,
            extension_root=extension_root,
            models=models,
            ns=transition_ns,
            cached_ns=cached_ns,
            seeds=seeds,
            numerical=numerical,
            families=families,
            bootstrap=bootstrap,
            analysis_seed=analysis_seed,
        )
    counterfactual_roles = {
        key: value
        for key, value in role_results.items()
        if int(key[1]) in {int(records) for records in counterfactual_ns}
    }
    if counterfactual is None:
        counterfactual = load_cached_counterfactual_results(
            store=store,
            base_canonical_root=base_canonical_root,
            extension_root=extension_root,
            models=models,
            ns=counterfactual_ns,
            cached_ns=cached_ns,
            seeds=seeds,
            role_results=counterfactual_roles,
            sizes=sizes,
            numerical=numerical,
            donors_per_role=donors_per_role,
            accuracy_gate=accuracy_gate,
            analysis_seed=analysis_seed,
        )
    performance = _read_performance_rows(
        base_analysis_root=base_analysis_root,
        training_run_dir=training_run_dir,
        models=models,
        ns=all_performance_ns,
        seeds=seeds,
        width=width,
    )
    figure_dir = extension_root / "figures"
    table_dir = extension_root / "tables"
    plot_role_specialisation(
        role_results,
        figure_dir=figure_dir,
        table_dir=table_dir,
        models=models,
        seeds=seeds,
        records=int(role_figure_n),
    )
    for records in transition_ns:
        if int(records) == int(role_figure_n):
            continue
        plot_role_specialisation(
            role_results,
            figure_dir=figure_dir,
            table_dir=table_dir,
            models=models,
            seeds=seeds,
            records=int(records),
            stem=f"S01_head_role_specialisation_N{int(records)}",
            table_name=f"supplement_head_roles_N{int(records)}.csv",
            headline=False,
        )
    plot_counterfactual_validation(
        counterfactual,
        figure_dir=figure_dir,
        table_dir=table_dir,
        models=models,
        ns=counterfactual_ns,
        bootstrap=bootstrap,
    )
    plot_performance_transition(
        role_results,
        performance,
        figure_dir=figure_dir,
        table_dir=table_dir,
        models=models,
        ns=transition_ns,
        seeds=seeds,
        bootstrap_seed=bootstrap.rng_seed,
    )
    atomic_json(
        extension_root / "nar_extension_figure_index.json",
        {
            "extension_version": EXTENSION_VERSION,
            "figures": sorted(path.name for path in figure_dir.glob("*.png")),
            "vector_figures": sorted(path.name for path in figure_dir.glob("*.pdf")),
            "tables": sorted(path.name for path in table_dir.glob("*.csv")),
            "figure_regeneration": "model-free from protected caches",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache-safe NAR role, counterfactual, and capacity-transition analysis"
    )
    parser.add_argument(
        "--phase",
        choices=("all", "transition_scores", "roles", "counterfactual", "figures"),
        default="all",
    )
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/graph_specialisation_metrics/nar_grit",
    )
    parser.add_argument("--training-run-name", default="nar_grit_fixed_n_v3")
    parser.add_argument("--base-analysis-name", default="canonical_nar_analysis_d128")
    parser.add_argument("--extension-name", default=DEFAULT_EXTENSION_NAME)
    parser.add_argument("--models", default="1hop,2hop,dense")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cached-ns", default="4,16,64")
    parser.add_argument("--additional-score-ns", default="8,32")
    parser.add_argument("--transition-ns", default="4,8,16,32,64")
    parser.add_argument("--counterfactual-ns", default="4,8,16")
    parser.add_argument("--all-performance-ns", default="4,8,16,32,64,80")
    parser.add_argument("--role-figure-n", type=int, default=16)
    parser.add_argument("--analysis-width", type=int, default=128)
    parser.add_argument("--counterfactual-donors-per-role", type=int, default=8)
    parser.add_argument("--accuracy-gate", type=float, default=0.85)
    parser.add_argument(
        "--score-graphs-per-batch",
        type=int,
        default=48,
        help="initial score batch; CUDA OOM automatically halves only the failing batch",
    )
    parser.add_argument(
        "--counterfactual-graphs-per-batch",
        type=int,
        default=48,
        help="initial counterfactual batch; CUDA OOM automatically halves and resumes",
    )
    parser.add_argument(
        "--graphs-per-batch",
        type=int,
        default=None,
        help="deprecated compatibility alias overriding both stage-specific batch sizes",
    )
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--grit-dir", default="/content/GRIT")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser


def _additional_scores_exist(
    *,
    extension_root: Path,
    models: Sequence[str],
    ns: Sequence[int],
    seeds: Sequence[int],
) -> bool:
    root = extension_root / "transition_scores" / "canonical"
    return all(
        (
            root
            / task_name(model, records)
            / f"seed_{int(seed)}"
            / "cache"
            / "scores"
            / "raw.pt"
        ).exists()
        for records in ns
        for model in models
        for seed in seeds
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = _parse_csv_strings(args.models)
    seeds = _parse_csv_ints(args.seeds)
    cached_ns = _parse_csv_ints(args.cached_ns)
    additional_ns = _parse_csv_ints(args.additional_score_ns)
    transition_ns = _parse_csv_ints(args.transition_ns)
    counterfactual_ns = _parse_csv_ints(args.counterfactual_ns)
    performance_ns = _parse_csv_ints(args.all_performance_ns)
    score_graphs_per_batch = int(
        args.graphs_per_batch
        if args.graphs_per_batch is not None
        else args.score_graphs_per_batch
    )
    counterfactual_graphs_per_batch = int(
        args.graphs_per_batch
        if args.graphs_per_batch is not None
        else args.counterfactual_graphs_per_batch
    )
    if score_graphs_per_batch < 1 or counterfactual_graphs_per_batch < 1:
        raise ValueError("stage batch sizes must be positive")
    if any(model not in MODEL_ORDER for model in models):
        raise ValueError(f"models must be drawn from {MODEL_ORDER}")
    if len(seeds) != 3 and not args.fast_dev_run:
        raise ValueError("the registered paper analysis requires exactly three seeds")
    if not set(transition_ns).issubset(set(cached_ns) | set(additional_ns)):
        raise ValueError("every transition N must be cached or listed in additional-score-ns")
    if not set(counterfactual_ns).issubset(set(transition_ns)):
        raise ValueError("counterfactual-ns must be a subset of transition-ns")
    if int(args.role_figure_n) not in set(transition_ns):
        raise ValueError("role-figure-n must be included in transition-ns")
    drive_root = Path(args.drive_root)
    training_run_dir = drive_root / str(args.training_run_name)
    base_analysis_root = training_run_dir / str(args.base_analysis_name)
    base_canonical_root = base_analysis_root / "canonical"
    extension_root = (
        base_analysis_root / "extensions" / str(args.extension_name)
    )
    _safe_extension_layout(base_analysis_root, extension_root)
    extension_root.mkdir(parents=True, exist_ok=True)
    store = ExtensionStore(extension_root)
    sizes, numerical, bootstrap, families, analysis_seed = _methodology_policies(
        base_canonical_root
    )
    if args.fast_dev_run:
        sizes = dataclasses.replace(
            sizes,
            discovery_graphs=min(3, sizes.discovery_graphs),
            causal_graphs=min(2, sizes.causal_graphs),
            clean_ablation_graphs=min(2, sizes.clean_ablation_graphs),
            semantic_donor_graphs=min(8, sizes.semantic_donor_graphs),
            donors_per_source=min(2, sizes.donors_per_source),
        )
    need_grit = args.phase in {"all", "transition_scores", "counterfactual"}
    if need_grit:
        training.setup_official_grit(
            Path(args.grit_dir), install=not bool(args.skip_install)
        )
    if args.phase in {"all", "transition_scores"}:
        if _additional_scores_exist(
            extension_root=extension_root,
            models=models,
            ns=additional_ns,
            seeds=seeds,
        ):
            print("[transition-scores] all protected score caches already exist", flush=True)
        else:
            run_additional_transition_scores(
                base_canonical_root=base_canonical_root,
                extension_root=extension_root,
                training_run_dir=training_run_dir,
                models=models,
                ns=additional_ns,
                seeds=seeds,
                width=int(args.analysis_width),
                accelerator=str(args.accelerator),
                graphs_per_batch=score_graphs_per_batch,
                num_threads=int(args.num_threads),
            )
        if args.phase == "transition_scores":
            return {"extension_root": str(extension_root)}
    role_results: Mapping[tuple[str, int, int], Mapping[str, Any]] | None = None
    if args.phase in {"all", "roles"}:
        role_results = run_role_derivation(
            store=store,
            base_canonical_root=base_canonical_root,
            extension_root=extension_root,
            models=models,
            ns=transition_ns,
            cached_ns=cached_ns,
            seeds=seeds,
            numerical=numerical,
            families=families,
            bootstrap=bootstrap,
            analysis_seed=analysis_seed,
        )
        if args.phase == "roles":
            return {"extension_root": str(extension_root)}
    counterfactual_results: (
        Mapping[tuple[str, int, int], Mapping[str, Any]] | None
    ) = None
    if args.phase in {"all", "counterfactual"}:
        counterfactual_results = run_counterfactual_analysis(
            store=store,
            base_canonical_root=base_canonical_root,
            extension_root=extension_root,
            training_run_dir=training_run_dir,
            models=models,
            ns=counterfactual_ns,
            cached_ns=cached_ns,
            seeds=seeds,
            width=int(args.analysis_width),
            sizes=sizes,
            numerical=numerical,
            families=families,
            bootstrap=bootstrap,
            analysis_seed=analysis_seed,
            accelerator=str(args.accelerator),
            donors_per_role=int(args.counterfactual_donors_per_role),
            graphs_per_batch=counterfactual_graphs_per_batch,
            accuracy_gate=float(args.accuracy_gate),
            role_results=(
                None
                if role_results is None
                else {
                    key: value
                    for key, value in role_results.items()
                    if int(key[1]) in set(counterfactual_ns)
                }
            ),
        )
        if args.phase == "counterfactual":
            return {"extension_root": str(extension_root)}
    make_extension_figures(
        store=store,
        base_analysis_root=base_analysis_root,
        base_canonical_root=base_canonical_root,
        extension_root=extension_root,
        training_run_dir=training_run_dir,
        models=models,
        transition_ns=transition_ns,
        counterfactual_ns=counterfactual_ns,
        cached_ns=cached_ns,
        all_performance_ns=performance_ns,
        seeds=seeds,
        width=int(args.analysis_width),
        role_figure_n=int(args.role_figure_n),
        sizes=sizes,
        numerical=numerical,
        families=families,
        bootstrap=bootstrap,
        analysis_seed=analysis_seed,
        donors_per_role=int(args.counterfactual_donors_per_role),
        accuracy_gate=float(args.accuracy_gate),
        role_results=role_results,
        counterfactual=counterfactual_results,
    )
    atomic_json(
        extension_root / "run_summary.json",
        {
            "extension_version": EXTENSION_VERSION,
            "extension_protocol": EXTENSION_PROTOCOL,
            "repository_commit_recorded_for_provenance_only": _repository_commit(),
            "base_analysis_root": str(base_analysis_root),
            "base_canonical_root": str(base_canonical_root),
            "extension_root": str(extension_root),
            "cache_policy": (
                "canonical caches read-only; new canonical scores isolated under "
                "transition_scores; extension artifacts immutable and fail-closed"
            ),
            "models": list(models),
            "seeds": list(seeds),
            "cached_N_values": list(cached_ns),
            "additional_score_N_values": list(additional_ns),
            "transition_N_values": list(transition_ns),
            "counterfactual_N_values": list(counterfactual_ns),
            "role_figure_N": int(args.role_figure_n),
            "accuracy_gate": float(args.accuracy_gate),
            "counterfactual_donors_per_role": int(
                args.counterfactual_donors_per_role
            ),
            "execution": {
                "score_graphs_per_batch_initial": score_graphs_per_batch,
                "counterfactual_graphs_per_batch_initial": (
                    counterfactual_graphs_per_batch
                ),
                "cuda_oom_backoff": True,
            },
            "figures": [
                ROLE_FIGURE_STEM,
                COUNTERFACTUAL_FIGURE_STEM,
                TRANSITION_FIGURE_STEM,
            ],
        },
    )
    print(f"[done] figures and tables saved under {extension_root}", flush=True)
    return {"extension_root": str(extension_root)}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run(build_parser().parse_args(list(argv) if argv is not None else None))


__all__ = [
    "COUNTERFACTUAL_FIGURE_STEM",
    "DEFAULT_EXTENSION_NAME",
    "EXTENSION_PROTOCOL",
    "EXTENSION_VERSION",
    "ExtensionContract",
    "ExtensionStore",
    "ROLE_FIGURE_STEM",
    "RoleCoordinates",
    "TRANSITION_FIGURE_STEM",
    "build_parser",
    "counterfactual_estimand",
    "derive_role_result",
    "freeze_role_families",
    "main",
    "role_coordinates",
    "run",
]


if __name__ == "__main__":
    main()
