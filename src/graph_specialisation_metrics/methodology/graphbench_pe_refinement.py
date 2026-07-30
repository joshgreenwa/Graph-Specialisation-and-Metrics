"""Focused four-arm PE refinement for official-GRIT GraphBench matching checkpoints.

This experiment is intentionally separate from the canonical carriage runner. It reuses the
official GraphBench/GRIT adapter, its per-head routed-value hook, semantic donor swap, exact
activation patch sites, soft audit ledger, and graph-balanced estimators while narrowing the
scientific scope to the structural-intervention decision.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .audit import audit_check, audit_scope, log_summary, set_strict, within_tolerance
from .cache import StaleCacheError, atomic_json, checkpoint_sha256
from .causal import donor_necessity, patch_response
from .graphbench import (
    GraphBenchGritBackend,
    GraphBenchRuntime,
    GraphBenchEdgeDonorPool,
    _graph_degrees,
    build_graphbench_channel_events,
    build_graphbench_runtime,
    structural_pe_intervention,
    verify_structural_pe_intervention,
)
from .execution import execute_graph_batches, is_cuda_oom
from .progress import ProgressJournal
from .protocol import (
    ExecutionPolicy,
    NumericalPolicy,
    RunSizes,
    deterministic_splits,
    stable_hash,
)
from .scores import event_head_score_systems, head_coordinates, project_transport
from .tasks import get_task


PE_REFINEMENT_VERSION = "graphbench-bipartite-pe-refinement-v1"
TASK_NAME = "graphbench_bipartite_matching_hard"
STRUCTURAL_ARMS = (
    "rrwp_copy",
    "rrwp_transpose",
    "complete_pe_copy",
    "complete_pe_transpose",
)
SCORE_SYSTEMS = ("mass", "coherent")
CAUSAL_SPLITS = ("refinement", "confirmation")


@dataclass(frozen=True)
class PERefinementSizes:
    discovery_graphs: int = 64
    refinement_graphs: int = 24
    confirmation_graphs: int = 24
    clean_ablation_graphs: int = 64
    semantic_donor_graphs: int = 2_000
    semantic_sources_per_graph: int = 12
    donors_per_source: int = 8
    taylor_graphs: int = 12
    permutation_replicates: int = 2_000

    @property
    def causal_graphs(self) -> int:
        return int(self.refinement_graphs) + int(self.confirmation_graphs)

    def validate(self) -> None:
        for field in dataclasses.fields(self):
            if int(getattr(self, field.name)) < 1:
                raise ValueError(f"{field.name} must be positive")
        if int(self.taylor_graphs) > int(self.refinement_graphs):
            raise ValueError("Taylor graphs must be a frozen subset of refinement graphs")


@dataclass(frozen=True)
class PERefinementConfig:
    output_dir: str
    training_output_root: str
    dataset_root: str
    pe_cache_root: str
    runner_path: str
    pe_cache_namespace: str = "base_40k4k4k_n64"
    pe_cache_dtype: str = "float32"
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    sizes: PERefinementSizes = dataclasses.field(default_factory=PERefinementSizes)
    numerical: NumericalPolicy = dataclasses.field(default_factory=NumericalPolicy)
    execution: ExecutionPolicy = dataclasses.field(
        default_factory=lambda: ExecutionPolicy(
            graphs_per_batch=16,
            oom_backoff=True,
            replica_pair_budget=2_000_000,
            jacobian_output_chunk=64,
            progress_heartbeat_seconds=30.0,
        )
    )
    analysis_seed: int = 31_415
    accelerator: str = "cuda:0"
    head_batch_size: int = 24
    strict_audits: bool = False
    resume: bool = True
    force: bool = False

    def validate(self) -> None:
        self.sizes.validate()
        self.numerical.validate()
        self.execution.validate()
        if int(self.head_batch_size) < 1:
            raise ValueError("head_batch_size must be positive")
        if tuple(self.seeds) != (0, 1, 2, 3):
            raise ValueError("the core PE refinement is registered for seeds 0,1,2,3")
        if self.pe_cache_dtype not in {"float32", "float16"}:
            raise ValueError("pe_cache_dtype must be float32 or float16")

    @property
    def root(self) -> Path:
        return Path(self.output_dir)

    @property
    def scientific_record(self) -> dict[str, Any]:
        return {
            "version": PE_REFINEMENT_VERSION,
            "task": TASK_NAME,
            "seeds": list(self.seeds),
            "analysis_population": "GraphBench n=16 validation split",
            "pe_cache_namespace": self.pe_cache_namespace,
            "pe_cache_dtype": self.pe_cache_dtype,
            "official_grit_commit": "6c988ea600a606fbb49a2246c64a2d37396b3ab5",
            "model_visible_rrwp_steps": 16,
            "sizes": asdict(self.sizes),
            "numerical": asdict(self.numerical),
            "analysis_seed": int(self.analysis_seed),
            "structural_arms": list(STRUCTURAL_ARMS),
            "score_systems": list(SCORE_SYSTEMS),
            "semantic_intervention": "reciprocal-edge-unit donor value swap",
            "semantic_donor_law": (
                "external training graph; different edge value; minimum endpoint-degree-signature "
                "gap; graph-uniform then edge-uniform; iid replacement"
            ),
            "structural_pair_law": (
                "same node type and inferred bipartition side; non-identical RRWP role; "
                "near/middle/far RRWP-role strata; unique without replacement; no degree match"
            ),
            "structural_interventions": {
                "rrwp_copy": "RRWP incident row/column/self donor copy",
                "rrwp_transpose": "RRWP two-axis source/donor transposition",
                "complete_pe_copy": "RRWP donor copy plus degree/log-degree role copy",
                "complete_pe_transpose": (
                    "RRWP two-axis transposition plus degree/log-degree transposition"
                ),
            },
            "held_fixed": (
                "edge support, edge values, node types, labels, target/readout coordinates"
            ),
            "causal_control": (
                "same-source alternative donor/partner, distinct payload, nearest full input dose"
            ),
            "score_aggregation": "donor within source; source within graph; graphs equally",
            "confirmation_lockbox": True,
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.scientific_record)


@dataclass
class PreparedPERefinement:
    config: PERefinementConfig
    runtime: GraphBenchRuntime
    backend: GraphBenchGritBackend
    task: Any
    donor_pool: GraphBenchEdgeDonorPool
    splits: Any
    checkpoint: Path
    checkpoint_sha: str
    seed: int
    progress: ProgressJournal

    @property
    def seed_dir(self) -> Path:
        return self.config.root / TASK_NAME / f"seed_{self.seed}"


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cache_scientific_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Return cache-validity fields while retaining the checkout as provenance."""

    record = dict(contract)
    record.pop("repository_commit", None)
    return record


def _cache_scientific_fingerprint(contract: Mapping[str, Any]) -> str:
    return stable_hash(_cache_scientific_contract(contract))


def _contract_difference_fields(
    stored: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> list[str]:
    return [
        str(key)
        for key in sorted(set(stored) | set(expected))
        if stored.get(key) != expected.get(key)
    ]


def _validate_protected_metadata(
    metadata: Any,
    expected_contract: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    """Validate both current and legacy commit-bound PE-refinement caches."""

    if not isinstance(metadata, Mapping):
        raise StaleCacheError(
            f"protected PE-refinement shard has malformed metadata: {path}"
        )
    stored_contract = metadata.get("contract")
    if not isinstance(stored_contract, Mapping):
        raise StaleCacheError(
            f"protected PE-refinement shard has a malformed contract: {path}"
        )
    claimed = metadata.get("fingerprint")
    stored_scientific = _cache_scientific_fingerprint(stored_contract)
    legacy_claim = stable_hash(dict(stored_contract))
    # Earlier v1 shards included repository_commit in the validity fingerprint.
    # Accept that representation when it is internally consistent, then compare
    # only the scientific contract below.
    if claimed not in {stored_scientific, legacy_claim}:
        raise StaleCacheError(
            f"protected PE-refinement shard has an internally inconsistent "
            f"fingerprint: {path}"
        )
    provenance_claim = metadata.get("provenance_fingerprint")
    if provenance_claim is not None and provenance_claim != legacy_claim:
        raise StaleCacheError(
            f"protected PE-refinement shard has an internally inconsistent "
            f"provenance fingerprint: {path}"
        )
    expected_scientific = _cache_scientific_fingerprint(expected_contract)
    if stored_scientific != expected_scientific:
        differing = _contract_difference_fields(
            _cache_scientific_contract(stored_contract),
            _cache_scientific_contract(expected_contract),
        )
        detail = ", ".join(differing[:8]) or "unknown"
        if len(differing) > 8:
            detail += f", +{len(differing) - 8} more"
        raise StaleCacheError(
            f"protected PE-refinement shard has another scientific contract: "
            f"{path}; differing fields: {detail}"
        )


class ProtectedShardStore:
    """Atomic, contract-bound shards shared safely across component jobs."""

    def __init__(
        self,
        prepared: PreparedPERefinement,
        namespace: str,
    ) -> None:
        self.prepared = prepared
        self.namespace = str(namespace)
        self.root = prepared.seed_dir / self.namespace
        self.contract = {
            "version": PE_REFINEMENT_VERSION,
            "scientific_fingerprint": prepared.config.fingerprint,
            "task": TASK_NAME,
            "seed": int(prepared.seed),
            "checkpoint_sha256": prepared.checkpoint_sha,
            "split_fingerprint": prepared.splits.fingerprint,
            "repository_commit": _repository_commit(),
            "namespace": self.namespace,
        }
        # The exact checkout is retained for provenance. Scientific compatibility
        # is governed by the registered methodology/configuration fields, so an
        # audit-only or plotting commit does not invalidate expensive tensors.
        self.fingerprint = _cache_scientific_fingerprint(self.contract)

    def path(self, stage: str, name: str) -> Path:
        return self.root / stage / f"{name}.pt"

    def load(self, stage: str, name: str) -> Any | None:
        import torch

        path = self.path(stage, name)
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError) as error:
            raise StaleCacheError(f"protected PE-refinement shard is unreadable: {path}") from error
        if not isinstance(payload, Mapping):
            raise StaleCacheError(
                f"protected PE-refinement shard is malformed: {path}"
            )
        metadata = payload.get("metadata", {})
        _validate_protected_metadata(metadata, self.contract, path=path)
        return payload.get("value")

    def save(self, stage: str, name: str, value: Any) -> Path:
        import torch

        path = self.path(stage, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self.load(stage, name)
        payload = {
            "metadata": {
                "fingerprint": self.fingerprint,
                "provenance_fingerprint": stable_hash(self.contract),
                "contract": self.contract,
            },
            "value": value,
        }
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(payload, temporary)
        temporary.replace(path)
        return path

    def save_json(self, stage: str, name: str, value: Any) -> Path:
        path = self.root / stage / f"{name}.json"
        atomic_json(
            path,
            {
                "metadata": {
                    "fingerprint": self.fingerprint,
                    "provenance_fingerprint": stable_hash(self.contract),
                    "contract": self.contract,
                },
                "value": value,
            },
        )
        return path


def _checkpoint_path(config: PERefinementConfig, seed: int) -> Path:
    return (
        Path(config.training_output_root)
        / "bipartite_matching_hard"
        / "grit"
        / f"seed{int(seed)}"
        / "best.pt"
    )


@lru_cache(maxsize=16)
def _cached_checkpoint_sha(path: str) -> str:
    return checkpoint_sha256(Path(path))


def _registered_split_sizes(config: PERefinementConfig) -> RunSizes:
    return RunSizes(
        discovery_graphs=int(config.sizes.discovery_graphs),
        causal_graphs=int(config.sizes.causal_graphs),
        clean_ablation_graphs=int(config.sizes.clean_ablation_graphs),
        semantic_donor_graphs=int(config.sizes.semantic_donor_graphs),
        sources_per_graph=16,
        donors_per_source=int(config.sizes.donors_per_source),
    )


def audit_existing_pe_refinement_cache(config: PERefinementConfig) -> int:
    """Fail on incompatible/corrupt existing shards before submitting GPU jobs."""

    import torch

    config.validate()
    # Production preflight already requires the registered 40k-train/4k-validation
    # subset caches. Computing the deterministic split here avoids loading GRIT or
    # the dataset merely to validate existing result metadata.
    splits = deterministic_splits(
        4_000,
        40_000,
        _registered_split_sizes(config),
        int(config.analysis_seed),
        same_index_space=False,
    )
    checked = 0
    for seed in config.seeds:
        seed = int(seed)
        seed_dir = config.root / TASK_NAME / f"seed_{seed}"
        checkpoint = _checkpoint_path(config, seed).expanduser().resolve()
        expected_checkpoint = _cached_checkpoint_sha(str(checkpoint))
        for path in sorted(seed_dir.rglob("*.pt")):
            relative = path.relative_to(seed_dir)
            if "_stale" in relative.parts:
                continue
            if not relative.parts:
                continue
            namespace = relative.parts[0]
            if namespace == "arms":
                if len(relative.parts) < 2:
                    raise StaleCacheError(
                        f"cannot infer arm namespace for protected cache: {path}"
                    )
                namespace = f"arms/{relative.parts[1]}"
            try:
                try:
                    payload = torch.load(
                        path,
                        map_location="cpu",
                        weights_only=False,
                        mmap=True,
                    )
                except (TypeError, RuntimeError):
                    payload = torch.load(
                        path,
                        map_location="cpu",
                        weights_only=False,
                    )
            except (OSError, RuntimeError, EOFError) as error:
                raise StaleCacheError(
                    f"protected PE-refinement shard is unreadable during "
                    f"preflight: {path}"
                ) from error
            if not isinstance(payload, Mapping):
                raise StaleCacheError(
                    f"protected PE-refinement shard is malformed during "
                    f"preflight: {path}"
                )
            expected = {
                "version": PE_REFINEMENT_VERSION,
                "scientific_fingerprint": config.fingerprint,
                "task": TASK_NAME,
                "seed": seed,
                "checkpoint_sha256": expected_checkpoint,
                "split_fingerprint": splits.fingerprint,
                "repository_commit": _repository_commit(),
                "namespace": namespace,
            }
            _validate_protected_metadata(
                payload.get("metadata", {}),
                expected,
                path=path,
            )
            checked += 1
            del payload
    print(
        f"[OK] existing PE-refinement cache contracts: {checked} shard(s) "
        "compatible",
        flush=True,
    )
    return checked


def prepare_pe_refinement(
    config: PERefinementConfig,
    seed: int,
    *,
    component: str,
) -> PreparedPERefinement:
    """Load one official checkpoint and reconstruct the registered n=16 validation split."""

    config.validate()
    set_strict(bool(config.strict_audits))
    seed = int(seed)
    if seed not in config.seeds:
        raise ValueError(f"unregistered seed {seed}; expected one of {config.seeds}")
    checkpoint = _checkpoint_path(config, seed).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing trained GRIT checkpoint: {checkpoint}")
    task = get_task(TASK_NAME)
    overrides = {
        "runner_path": str(config.runner_path),
        "dataset_root": str(config.dataset_root),
        "pe_cache_root": str(config.pe_cache_root),
        "pe_cache_namespace": str(config.pe_cache_namespace),
        "pe_cache_dtype": str(config.pe_cache_dtype),
        "require_subset_cache": True,
        "require_pe_cache": True,
        "build_missing_pe_cache": False,
        "eval_split": "val",
        "donor_split": "train",
    }
    runtime, descriptor = build_graphbench_runtime(
        task,
        checkpoint_path=checkpoint,
        train_seed=seed,
        accelerator=config.accelerator,
        overrides=overrides,
        jacobian_output_chunk=int(config.execution.jacobian_output_chunk),
    )
    backend = GraphBenchGritBackend(
        runtime,
        task,
        np.asarray([1.0], dtype=np.float64),
        jacobian_output_chunk=int(config.execution.jacobian_output_chunk),
    )
    splits = deterministic_splits(
        len(runtime.eval_ds),
        len(runtime.donor_ds),
        _registered_split_sizes(config),
        int(config.analysis_seed),
        same_index_space=False,
    )
    donor_pool = GraphBenchEdgeDonorPool(
        [
            (graph_id, runtime.donor_ds[graph_id])
            for graph_id in splits.semantic_donor_pool
        ]
    )
    checkpoint_sha = checkpoint_sha256(checkpoint)
    seed_dir = config.root / TASK_NAME / f"seed_{seed}"
    progress = ProgressJournal(
        seed_dir / "progress" / f"{component}.jsonl",
        heartbeat_seconds=config.execution.progress_heartbeat_seconds,
    )
    prepared = PreparedPERefinement(
        config=config,
        runtime=runtime,
        backend=backend,
        task=task,
        donor_pool=donor_pool,
        splits=splits,
        checkpoint=Path(descriptor),
        checkpoint_sha=checkpoint_sha,
        seed=seed,
        progress=progress,
    )
    atomic_json(
        seed_dir / "workers" / f"{component}.json",
        {
            "version": PE_REFINEMENT_VERSION,
            "scientific_fingerprint": config.fingerprint,
            "component": component,
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "repository_commit": _repository_commit(),
            "split_fingerprint": splits.fingerprint,
            "splits": asdict(splits),
            "model_geometry": backend.geometry,
            "validation_metric": runtime.val_metric,
            "official_grit_commit": runtime.checks["official_grit_commit"],
        },
    )
    return prepared


def causal_graph_ids(prepared: PreparedPERefinement, split: str) -> tuple[int, ...]:
    causal = tuple(int(value) for value in prepared.splits.causal)
    boundary = int(prepared.config.sizes.refinement_graphs)
    if split == "refinement":
        return causal[:boundary]
    if split == "confirmation":
        return causal[boundary:]
    raise ValueError(f"unknown causal split {split!r}")


def taylor_graph_ids(prepared: PreparedPERefinement) -> tuple[int, ...]:
    return causal_graph_ids(prepared, "refinement")[
        : int(prepared.config.sizes.taylor_graphs)
    ]


def _rng(config: PERefinementConfig, *parts: Any) -> np.random.Generator:
    # Training seed is deliberately absent: graph/event manifests are paired across checkpoints.
    digest = stable_hash(
        {"analysis_seed": int(config.analysis_seed), "parts": parts},
        length=16,
    )
    return np.random.default_rng(int(digest, 16))


def _bipartition_sides(graph: Any) -> np.ndarray:
    """Deterministically two-colour the fixed graph support."""

    n = int(graph.num_nodes)
    neighbours: list[set[int]] = [set() for _ in range(n)]
    for left, right in graph.edge_index.detach().cpu().numpy().T:
        left, right = int(left), int(right)
        neighbours[left].add(right)
        neighbours[right].add(left)
    side = np.full(n, -1, dtype=np.int64)
    for root in range(n):
        if side[root] >= 0:
            continue
        side[root] = 0
        queue = [root]
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            for other in sorted(neighbours[node]):
                if side[other] < 0:
                    side[other] = 1 - side[node]
                    queue.append(other)
                elif side[other] == side[node]:
                    raise RuntimeError("GraphBench matching support is not bipartite")
    return side


def _rrwp_role_distance(
    graph: Any,
    source: int,
    donor: int,
    *,
    rrwp_steps: int,
) -> float:
    rrwp = graph.rrwp.detach().float()[..., : int(rrwp_steps)]
    delta = np.concatenate(
        (
            (rrwp[int(source), :] - rrwp[int(donor), :]).cpu().numpy().reshape(-1),
            (rrwp[:, int(source)] - rrwp[:, int(donor)]).cpu().numpy().reshape(-1),
        )
    ).astype(np.float64)
    return float(np.sqrt(np.mean(np.square(delta))))


def _visible_rrwp_footprints(
    prepared: PreparedPERefinement,
    graph: Any,
) -> tuple[bytes, ...]:
    steps = int(prepared.runtime.runner.RRWP_STEPS)
    rrwp = graph.rrwp.detach().cpu().numpy()[..., :steps]
    return tuple(
        b"\x1f".join(
            (
                np.ascontiguousarray(rrwp[node, :]).tobytes(),
                np.ascontiguousarray(rrwp[:, node]).tobytes(),
            )
        )
        for node in range(int(graph.num_nodes))
    )


def _distance_strata(
    candidates: Sequence[int],
    distances: Mapping[int, float],
) -> dict[int, str]:
    ordered = sorted((int(node) for node in candidates), key=lambda node: (distances[node], node))
    chunks = np.array_split(np.asarray(ordered, dtype=np.int64), 3)
    labels = ("near", "middle", "far")
    return {
        int(node): labels[index]
        for index, chunk in enumerate(chunks)
        for node in chunk.tolist()
    }


def _balanced_stratified_donors(
    candidates: Sequence[int],
    strata: Mapping[int, str],
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    groups: dict[str, list[int]] = {label: [] for label in ("near", "middle", "far")}
    for node in candidates:
        groups[strata[int(node)]].append(int(node))
    for label in groups:
        if groups[label]:
            groups[label] = [
                int(value) for value in rng.permutation(np.asarray(groups[label], dtype=np.int64))
            ]
    selected: list[int] = []
    while len(selected) < min(int(count), len(candidates)):
        progressed = False
        for label in ("near", "middle", "far"):
            if groups[label]:
                selected.append(groups[label].pop())
                progressed = True
                if len(selected) >= min(int(count), len(candidates)):
                    break
        if not progressed:
            break
    return tuple(selected)


def structural_pair_manifest(
    prepared: PreparedPERefinement,
    stage: str,
    graph_ids: Sequence[int],
) -> dict[int, tuple[dict[str, Any], ...]]:
    result: dict[int, tuple[dict[str, Any], ...]] = {}
    for graph_id in graph_ids:
        graph_id = int(graph_id)
        graph = prepared.runtime.eval_ds[graph_id]
        sides = _bipartition_sides(graph)
        node_type = graph.node_type.detach().cpu().numpy().reshape(-1)
        degrees = _graph_degrees(graph)
        footprints = _visible_rrwp_footprints(prepared, graph)
        rrwp_steps = int(prepared.runtime.runner.RRWP_STEPS)
        rows: list[dict[str, Any]] = []
        for source in range(int(graph.num_nodes)):
            candidates = [
                donor
                for donor in range(int(graph.num_nodes))
                if donor != source
                and int(node_type[donor]) == int(node_type[source])
                and int(sides[donor]) == int(sides[source])
                and footprints[donor] != footprints[source]
            ]
            if not candidates:
                audit_check(
                    False,
                    "pe_refinement.no_structural_donor",
                    "structural source has no same-role, non-identical RRWP donor",
                    context={"graph": graph_id, "source": source, "stage": stage},
                )
                continue
            distances = {
                donor: _rrwp_role_distance(
                    graph, source, donor, rrwp_steps=rrwp_steps
                )
                for donor in candidates
            }
            strata = _distance_strata(candidates, distances)
            selected = _balanced_stratified_donors(
                candidates,
                strata,
                count=int(prepared.config.sizes.donors_per_source),
                rng=_rng(prepared.config, "structural_pairs", stage, graph_id, source),
            )
            for draw, donor in enumerate(selected):
                rows.append(
                    {
                        "channel": "structural",
                        "stage": stage,
                        "graph_id": graph_id,
                        "source": int(source),
                        "donor_graph_id": graph_id,
                        "donor_node": int(donor),
                        "draw": int(draw),
                        "source_degree": int(degrees[source]),
                        "donor_degree": int(degrees[donor]),
                        "degree_gap": abs(int(degrees[source]) - int(degrees[donor])),
                        "bipartition_side": int(sides[source]),
                        "source_node_type": int(node_type[source]),
                        "rrwp_role_distance": float(distances[donor]),
                        "distance_stratum": strata[donor],
                        "eligible_pool_size": int(len(candidates)),
                        "realised_donor_count": int(len(selected)),
                        "eligible_pool_exhausted": bool(len(selected) == len(candidates)),
                        "source_footprint": hashlib.sha256(footprints[source]).hexdigest()[:24],
                        "donor_footprint": hashlib.sha256(footprints[donor]).hexdigest()[:24],
                    }
                )
        if not rows:
            raise RuntimeError(f"graph {graph_id} has no estimable structural pairs")
        result[graph_id] = tuple(rows)
    return result


def semantic_event_manifest(
    prepared: PreparedPERefinement,
    stage: str,
    graph_ids: Sequence[int],
) -> dict[int, tuple[dict[str, Any], ...]]:
    result: dict[int, tuple[dict[str, Any], ...]] = {}
    for graph_id in graph_ids:
        graph_id = int(graph_id)
        base = prepared.runtime.eval_ds[graph_id]
        eligible = np.asarray(
            prepared.backend.eligible_sources(base, channel="semantic"),
            dtype=np.int64,
        )
        count = min(
            int(prepared.config.sizes.semantic_sources_per_graph),
            int(eligible.size),
        )
        if count == int(eligible.size):
            sources = np.sort(eligible)
        else:
            sources = np.sort(
                _rng(prepared.config, "semantic_sources", stage, graph_id).choice(
                    eligible, size=count, replace=False
                )
            )
        rows: list[dict[str, Any]] = []
        for source in sources:
            _, records = build_graphbench_channel_events(
                base,
                graph_id=graph_id,
                source=int(source),
                channel="semantic",
                stage=stage,
                donors=int(prepared.config.sizes.donors_per_source),
                rng=_rng(
                    prepared.config,
                    "semantic_events",
                    stage,
                    graph_id,
                    int(source),
                ),
                semantic_pool=prepared.donor_pool,
            )
            rows.extend(record.record() for record in records)
        if not rows:
            raise RuntimeError(f"graph {graph_id} has no estimable semantic events")
        result[graph_id] = tuple(rows)
    return result


def _arm_spec(arm: str) -> tuple[bool, bool]:
    if arm not in STRUCTURAL_ARMS:
        raise ValueError(f"unknown structural arm {arm!r}")
    return arm.startswith("complete_pe"), arm.endswith("transpose")


def _input_dose(
    base: Any,
    event: Any,
    *,
    rrwp_steps: int,
) -> dict[str, float]:
    """Standardised full-input dose plus interpretable component doses."""

    import torch

    rrwp_delta = (
        event.rrwp.detach().float()[..., : int(rrwp_steps)]
        - base.rrwp.detach().float()[..., : int(rrwp_steps)]
    ).reshape(-1)
    rrwp_rms = float(torch.sqrt(torch.mean(rrwp_delta.square())).item())
    rrwp_scale = float(
        base.rrwp.detach().float()[..., : int(rrwp_steps)].std().item()
    )
    degree_base = torch.as_tensor(
        _graph_degrees(base), dtype=torch.float32, device=base.edge_index.device
    )
    degree_event = getattr(event, "degree_override", None)
    if degree_event is None:
        degree_event = degree_base
    degree_event = degree_event.detach().float()
    degree_delta = degree_event - degree_base
    degree_rms = float(torch.sqrt(torch.mean(degree_delta.square())).item())
    log_delta = torch.log1p(degree_event) - torch.log1p(degree_base)
    log_degree_rms = float(torch.sqrt(torch.mean(log_delta.square())).item())
    degree_scale = max(float(degree_base.std().item()), 1.0)
    log_scale = max(float(torch.log1p(degree_base).std().item()), 1.0e-6)
    component_doses = torch.as_tensor(
        (
            rrwp_rms / max(rrwp_scale, 1.0e-6),
            degree_rms / degree_scale,
            log_degree_rms / log_scale,
        ),
        dtype=torch.float64,
    )
    return {
        # Equal component weighting prevents the dense RRWP tensor's entry count from making the
        # explicit degree/log-degree channels irrelevant to mismatch-dose matching.
        "input_dose": float(
            torch.sqrt(torch.mean(component_doses.square())).item()
        ),
        "rrwp_rms_dose": rrwp_rms,
        "degree_rms_dose": degree_rms,
        "log_degree_rms_dose": log_degree_rms,
    }


def rebuild_structural_events(
    prepared: PreparedPERefinement,
    graph_id: int,
    rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    complete_pe, transpose = _arm_spec(arm)
    base = prepared.runtime.eval_ds[int(graph_id)]
    rrwp_steps = int(prepared.runtime.runner.RRWP_STEPS)
    variants, records = [], []
    for row in rows:
        event = structural_pe_intervention(
            base,
            int(row["source"]),
            int(row["donor_node"]),
            complete_pe=complete_pe,
            transpose=transpose,
            rrwp_steps=rrwp_steps,
        )
        verify_structural_pe_intervention(
            base,
            event,
            int(row["source"]),
            int(row["donor_node"]),
            complete_pe=complete_pe,
            transpose=transpose,
            rrwp_steps=rrwp_steps,
        )
        variants.append(event)
        records.append(
            {
                **dict(row),
                "arm": arm,
                "intervention_mode": "transpose" if transpose else "donor_copy",
                "complete_pe": bool(complete_pe),
                **_input_dose(base, event, rrwp_steps=rrwp_steps),
                "payload_fingerprint": stable_hash(
                    {
                        "arm": arm,
                        "source": int(row["source"]),
                        "donor": int(row["donor_node"]),
                        "donor_footprint": row["donor_footprint"],
                    }
                ),
            }
        )
    return variants, records


def rebuild_semantic_events(
    prepared: PreparedPERefinement,
    graph_id: int,
    stage: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    base = prepared.runtime.eval_ds[int(graph_id)]
    sources = []
    for row in rows:
        source = int(row["source"])
        if source not in sources:
            sources.append(source)
    variants, records = [], []
    for source in sources:
        source_variants, source_records = build_graphbench_channel_events(
            base,
            graph_id=int(graph_id),
            source=source,
            channel="semantic",
            stage=stage,
            donors=int(prepared.config.sizes.donors_per_source),
            rng=_rng(
                prepared.config,
                "semantic_events",
                stage,
                int(graph_id),
                source,
            ),
            semantic_pool=prepared.donor_pool,
        )
        variants.extend(source_variants)
        records.extend(record.record() for record in source_records)
    if records != [dict(row) for row in rows]:
        raise RuntimeError(
            f"semantic event replay changed for graph={graph_id} stage={stage}"
        )
    return variants, records


def _clean_to_device(clean: Any, device: Any) -> Any:
    clean.capture.prediction = clean.capture.prediction.to(device)
    clean.capture.z = clean.capture.z.to(device)
    clean.capture.target = clean.capture.target.to(device)
    clean.capture.transport = tuple(value.to(device) for value in clean.capture.transport)
    clean.capture.final_state = clean.capture.final_state.to(device)
    if clean.capture.real_mask is not None:
        clean.capture.real_mask = clean.capture.real_mask.to(device)
    clean.transport = clean.transport.to(device)
    clean.final_state = clean.final_state.to(device)
    return clean


def ensure_clean_jacobians(
    prepared: PreparedPERefinement,
    graph_ids: Sequence[int],
    *,
    population: str,
    allow_compute: bool,
) -> dict[int, Any]:
    """Load or atomically create shared clean Jacobians for score/Taylor jobs."""

    store = ProtectedShardStore(prepared, "common")
    output: dict[int, Any] = {}
    missing: list[int] = []
    for graph_id in graph_ids:
        graph_id = int(graph_id)
        cached = (
            store.load(
                f"clean_jacobians/{population}",
                f"graph_{graph_id:06d}",
            )
            if prepared.config.resume and not prepared.config.force
            else None
        )
        if cached is None:
            missing.append(graph_id)
        else:
            output[graph_id] = _clean_to_device(cached, prepared.runtime.device)
            prepared.backend.remember_clean_jacobians(
                prepared.runtime.eval_ds[graph_id], output[graph_id]
            )
    if missing and not allow_compute:
        raise RuntimeError(
            f"{len(missing)} shared {population} clean-Jacobian shards are missing; "
            "run the corresponding common component first"
        )
    if output:
        print(
            f"[pe-refinement] loaded {len(output)}/{len(graph_ids)} shared "
            f"{population} clean Jacobians",
            flush=True,
        )
    for completed, graph_id in enumerate(missing, start=1):
        graph = prepared.runtime.eval_ds[graph_id]
        clean = prepared.backend.clean_jacobians(graph)
        output[graph_id] = clean
        store.save(
            f"clean_jacobians/{population}",
            f"graph_{graph_id:06d}",
            clean,
        )
        prepared.progress.emit(
            "clean_jacobian_complete",
            population=population,
            graph_id=graph_id,
            completed=completed,
            total=len(missing),
        )
    return output


def _hierarchical_event_mean(
    values: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    *,
    nan: bool = False,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] != len(records):
        raise ValueError("event values and records do not align")
    source_values = []
    for source in sorted({int(row["source"]) for row in records}):
        selected = np.asarray(
            [int(row["source"]) == source for row in records],
            dtype=bool,
        )
        with np.errstate(invalid="ignore"):
            source_values.append(
                np.nanmean(values[selected], axis=0)
                if nan
                else np.mean(values[selected], axis=0)
            )
    if not source_values:
        raise RuntimeError("event aggregation has no estimable source")
    with np.errstate(invalid="ignore"):
        return (
            np.nanmean(np.stack(source_values), axis=0)
            if nan
            else np.mean(np.stack(source_values), axis=0)
        )


def _verify_cached_records(
    cached: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    if len(cached) != len(expected):
        raise StaleCacheError(
            f"cached event count changed for {context}: {len(cached)} != {len(expected)}"
        )
    for position, (observed, registered) in enumerate(zip(cached, expected)):
        for key, value in registered.items():
            candidate = observed.get(key)
            if isinstance(value, float):
                equal = (
                    candidate is not None
                    and bool(
                        np.isclose(
                            float(candidate),
                            float(value),
                            rtol=0.0,
                            atol=1.0e-12,
                        )
                    )
                )
            else:
                equal = candidate == value
            if not equal:
                raise StaleCacheError(
                    f"cached event manifest changed for {context} row={position} "
                    f"field={key}: {candidate!r} != {value!r}"
                )


def _score_result(
    prepared: PreparedPERefinement,
    graph_id: int,
    variants: Sequence[Any],
    records: Sequence[Mapping[str, Any]],
    clean: Any,
    captured: Any,
) -> dict[str, Any]:
    import torch

    transport = torch.stack(captured.transport, dim=1)
    delta = transport[0:1] - transport[1:]
    q = project_transport(delta, clean.transport)
    systems = event_head_score_systems(
        q, mass_floor=prepared.config.numerical.score_floor
    )
    mass = systems["mass"].detach().cpu().numpy()
    coherent = systems["coherent"].detach().cpu().numpy()
    coherence = systems["carrier_coherence"].detach().cpu().numpy()
    if not np.isfinite(mass).all() or not np.isfinite(coherent).all():
        raise RuntimeError("non-finite head score entered the PE-refinement cache")
    return {
        "graph_id": int(graph_id),
        "records": [dict(row) for row in records],
        "event_mass": mass,
        "event_coherent": coherent,
        "event_carrier_coherence": coherence,
        "mass": _hierarchical_event_mean(mass, records),
        "coherent": _hierarchical_event_mean(coherent, records),
        "carrier_coherence": _hierarchical_event_mean(
            coherence, records, nan=True
        ),
        "carrier_coherence_estimable": np.sum(np.isfinite(coherence), axis=0),
        "event_effect": np.linalg.norm(
            (
                captured.z[0:1] - captured.z[1:]
            ).detach().cpu().numpy(),
            axis=-1,
        ),
        "replicas": int(len(variants) + 1),
    }


def _run_score_component(
    prepared: PreparedPERefinement,
    *,
    channel: str,
    arm: str | None,
    graph_ids: Sequence[int],
    manifest: Mapping[int, Sequence[Mapping[str, Any]]],
    clean: Mapping[int, Any],
) -> dict[str, Any]:
    store = ProtectedShardStore(
        prepared, "common" if channel == "semantic" else f"arms/{arm}"
    )
    stage = "scores/semantic" if channel == "semantic" else "scores/structural"
    shards: dict[int, Any] = {}
    missing: list[int] = []
    for graph_id in graph_ids:
        graph_id = int(graph_id)
        cached = (
            store.load(stage, f"graph_{graph_id:06d}")
            if prepared.config.resume and not prepared.config.force
            else None
        )
        if cached is None:
            missing.append(graph_id)
        else:
            _verify_cached_records(
                cached["records"],
                manifest[graph_id],
                context=f"{stage}/graph_{graph_id:06d}",
            )
            shards[graph_id] = cached

    def execute(chunk: Sequence[int]) -> list[dict[str, Any]]:
        contexts = []
        groups = []
        for value in chunk:
            graph_id = int(value)
            if channel == "semantic":
                variants, records = rebuild_semantic_events(
                    prepared, graph_id, "scores", manifest[graph_id]
                )
            else:
                variants, records = rebuild_structural_events(
                    prepared, graph_id, manifest[graph_id], str(arm)
                )
            contexts.append((graph_id, variants, records))
            groups.append([prepared.runtime.eval_ds[graph_id], *variants])
        captures = prepared.backend.capture_groups(groups)
        return [
            _score_result(
                prepared,
                graph_id,
                variants,
                records,
                clean[graph_id],
                captured,
            )
            for (graph_id, variants, records), captured in zip(contexts, captures)
        ]

    def consume(results: Sequence[Mapping[str, Any]]) -> None:
        for result in results:
            graph_id = int(result["graph_id"])
            shards[graph_id] = dict(result)
            store.save(stage, f"graph_{graph_id:06d}", dict(result))

    report = execute_graph_batches(
        missing,
        graphs_per_batch=int(prepared.config.execution.graphs_per_batch),
        execute=execute,
        consume=consume,
        oom_backoff=prepared.config.execution.oom_backoff,
        item_cost=lambda graph_id: prepared.backend.event_pair_cost(
            prepared.runtime.eval_ds[int(graph_id)],
            len({int(row["source"]) for row in manifest[int(graph_id)]}),
            int(prepared.config.sizes.donors_per_source),
        ),
        max_cost=prepared.config.execution.replica_pair_budget,
        on_batch=lambda done, total, size: prepared.progress.emit(
            "score_batch_complete",
            channel=channel,
            arm=arm,
            completed_graphs=int(len(graph_ids) - len(missing) + done),
            total_graphs=len(graph_ids),
            batch_graphs=int(size),
        ),
    )
    ordered = [shards[int(graph_id)] for graph_id in graph_ids]
    summary = {
        "version": PE_REFINEMENT_VERSION,
        "channel": channel,
        "arm": arm,
        "graph_ids": [int(value) for value in graph_ids],
        "manifest_hash": stable_hash(
            {
                str(graph_id): list(manifest[int(graph_id)])
                for graph_id in graph_ids
            }
        ),
        "raw": {
            "mass": np.mean(
                np.stack([np.asarray(row["mass"]) for row in ordered]), axis=0
            ),
            "coherent": np.mean(
                np.stack([np.asarray(row["coherent"]) for row in ordered]), axis=0
            ),
        },
        "carrier_coherence": np.nanmean(
            np.stack(
                [np.asarray(row["carrier_coherence"]) for row in ordered]
            ),
            axis=0,
        ),
        "event_effect": np.concatenate(
            [np.asarray(row["event_effect"]).reshape(-1) for row in ordered]
        ),
        "support": {
            "graphs": len(ordered),
            "events": int(sum(len(row["records"]) for row in ordered)),
            "sources": int(
                sum(
                    len({int(event["source"]) for event in row["records"]})
                    for row in ordered
                )
            ),
            "coherence_estimable_cells": int(
                sum(np.isfinite(row["event_carrier_coherence"]).sum() for row in ordered)
            ),
        },
        "execution": {
            **asdict(report),
            "cache_hits": int(len(graph_ids) - len(missing)),
            "cache_misses": int(len(missing)),
        },
    }
    store.save(stage, "summary", summary)
    return summary


def run_common_scores(prepared: PreparedPERefinement) -> dict[str, Any]:
    graph_ids = tuple(int(value) for value in prepared.splits.discovery)
    store = ProtectedShardStore(prepared, "common")
    manifest = semantic_event_manifest(prepared, "scores", graph_ids)
    store.save_json("manifests", "semantic_scores", manifest)
    clean = ensure_clean_jacobians(
        prepared,
        graph_ids,
        population="discovery",
        allow_compute=True,
    )
    return _run_score_component(
        prepared,
        channel="semantic",
        arm=None,
        graph_ids=graph_ids,
        manifest=manifest,
        clean=clean,
    )


def run_arm_scores(
    prepared: PreparedPERefinement,
    arm: str,
) -> dict[str, Any]:
    graph_ids = tuple(int(value) for value in prepared.splits.discovery)
    store = ProtectedShardStore(prepared, f"arms/{arm}")
    manifest = structural_pair_manifest(prepared, "scores", graph_ids)
    store.save_json("manifests", "structural_scores", manifest)
    clean = ensure_clean_jacobians(
        prepared,
        graph_ids,
        population="discovery",
        allow_compute=False,
    )
    return _run_score_component(
        prepared,
        channel="structural",
        arm=arm,
        graph_ids=graph_ids,
        manifest=manifest,
        clean=clean,
    )


def _head_order(prepared: PreparedPERefinement) -> tuple[tuple[int, int], ...]:
    return tuple(
        (layer, head)
        for layer in range(int(prepared.runtime.L))
        for head in range(int(prepared.runtime.H))
    )


def _mismatch_indices(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Same-source, distinct-payload, dose-matched controls for either intervention mode."""

    mismatch = np.full(len(records), -1, dtype=np.int64)
    controlled = np.zeros(len(records), dtype=bool)
    for position, row in enumerate(records):
        candidates = [
            other
            for other, candidate in enumerate(records)
            if other != position
            and int(candidate["source"]) == int(row["source"])
            and candidate["payload_fingerprint"] != row["payload_fingerprint"]
        ]
        if not candidates:
            audit_check(
                False,
                "pe_refinement.mismatch_control_unavailable",
                "event has no same-source alternative donor and is excluded from adjusted endpoints",
                context={
                    "graph": int(row["graph_id"]),
                    "source": int(row["source"]),
                    "draw": int(row["draw"]),
                },
            )
            continue
        row_dose = float(row.get("input_dose", row.get("dose", 0.0)))
        selected = min(
            candidates,
            key=lambda other: (
                abs(
                    float(
                        records[other].get(
                            "input_dose", records[other].get("dose", 0.0)
                        )
                    )
                    - row_dose
                ),
                abs(
                    int(records[other].get("degree_gap", 0))
                    - int(row.get("degree_gap", 0))
                ),
                other,
            ),
        )
        mismatch[position] = int(selected)
        controlled[position] = True
    return mismatch, controlled


def _repeat_replacements(
    captured: Any,
    *,
    head_count: int,
    event_count: int,
    kind: str,
    mismatch: np.ndarray | None = None,
) -> tuple[Any, ...]:
    if kind == "clean":
        return tuple(
            layer[0:1].expand(
                int(head_count) * int(event_count), *layer.shape[1:]
            )
            for layer in captured.transport
        )
    if kind == "event":
        return tuple(
            layer[1:].repeat(int(head_count), 1, 1, 1)
            for layer in captured.transport
        )
    if kind == "mismatch":
        import torch

        if mismatch is None:
            raise ValueError("mismatch replacements require indices")
        return tuple(
            layer[
                torch.as_tensor(
                    np.asarray(mismatch, dtype=np.int64) + 1,
                    dtype=torch.long,
                    device=layer.device,
                )
            ].repeat(int(head_count), 1, 1, 1)
            for layer in captured.transport
        )
    raise ValueError(kind)


def _taylor_metrics(
    predicted: np.ndarray,
    exact: np.ndarray,
    *,
    effect_floor: float,
) -> dict[str, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float64)
    exact = np.asarray(exact, dtype=np.float64)
    if predicted.shape != exact.shape:
        raise ValueError("Taylor predicted and exact vectors do not align")
    predicted_norm = np.linalg.norm(predicted, axis=-1)
    exact_norm = np.linalg.norm(exact, axis=-1)
    estimable = exact_norm > float(effect_floor)
    cosine = np.full_like(exact_norm, np.nan)
    norm_ratio = np.full_like(exact_norm, np.nan)
    relative_error = np.full_like(exact_norm, np.nan)
    denominator = predicted_norm * exact_norm
    directional = estimable & (denominator > float(effect_floor) ** 2)
    cosine[directional] = (
        np.sum(predicted[directional] * exact[directional], axis=-1)
        / denominator[directional]
    )
    norm_ratio[estimable] = predicted_norm[estimable] / exact_norm[estimable]
    relative_error[estimable] = (
        np.linalg.norm(predicted[estimable] - exact[estimable], axis=-1)
        / exact_norm[estimable]
    )
    return {
        "predicted_norm": predicted_norm,
        "exact_norm": exact_norm,
        "cosine": cosine,
        "norm_ratio": norm_ratio,
        "relative_error": relative_error,
        "estimable": estimable,
    }


def _causal_graph(
    prepared: PreparedPERefinement,
    *,
    graph_id: int,
    channel: str,
    arm: str | None,
    stage: str,
    manifest_rows: Sequence[Mapping[str, Any]],
    clean_taylor: Any | None,
) -> dict[str, Any]:
    """Evaluate all single heads with replica-specific batched patching."""

    graph_id = int(graph_id)
    base = prepared.runtime.eval_ds[graph_id]
    if channel == "semantic":
        variants, records = rebuild_semantic_events(
            prepared, graph_id, stage, manifest_rows
        )
        records = [
            {
                **row,
                "input_dose": float(row["dose"]),
            }
            for row in records
        ]
    else:
        variants, records = rebuild_structural_events(
            prepared, graph_id, manifest_rows, str(arm)
        )
    captured = prepared.backend.capture(
        [base, *variants],
        require_grad=False,
        include_virtual_transport=True,
    )
    z_clean = captured.z[0:1].detach().cpu().numpy()
    z_event = captured.z[1:].detach().cpu().numpy()
    events = len(records)
    heads = _head_order(prepared)
    mismatch, controlled = _mismatch_indices(records)
    safe_mismatch = np.where(controlled, mismatch, np.arange(events))
    endpoint_names = (
        "P_gross_matched",
        "P_gross_mismatch",
        "G_c",
        "R_gross",
        "I_gross",
        "R_align",
        "I_align",
        "M_align",
        "necessity",
        "gross_necessity",
        "event_effect",
    )
    endpoints = {
        name: np.full((len(heads), events), np.nan, dtype=np.float64)
        for name in endpoint_names
    }
    exact_injection = (
        np.full(
            (len(heads), events, int(z_event.shape[-1])),
            np.nan,
            dtype=np.float64,
        )
        if clean_taylor is not None
        else None
    )
    same_condition_max = 0.0
    batch_size = int(prepared.config.head_batch_size)
    for start in range(0, len(heads), batch_size):
        selected_heads = heads[start : start + batch_size]
        b = len(selected_heads)
        event_assignments = [
            head for head in selected_heads for _ in range(events)
        ]
        ablation_assignments = [
            head for head in selected_heads for _ in range(events + 1)
        ]
        event_targets = [
            variant for _head in selected_heads for variant in variants
        ]
        base_targets = [base for _head in selected_heads for _ in range(events)]
        ablation_targets = [
            target
            for _head in selected_heads
            for target in (base, *variants)
        ]
        _, z_ablated, _ = prepared.backend.ablate_individual_heads(
            ablation_targets,
            ablation_assignments,
        )
        z_ablated = (
            z_ablated.detach().cpu().numpy().reshape(b, events + 1, -1)
        )
        _, z_restore, _ = prepared.backend.patch_individual_heads(
            event_targets,
            _repeat_replacements(
                captured,
                head_count=b,
                event_count=events,
                kind="clean",
            ),
            event_assignments,
        )
        _, z_inject, _ = prepared.backend.patch_individual_heads(
            base_targets,
            _repeat_replacements(
                captured,
                head_count=b,
                event_count=events,
                kind="event",
            ),
            event_assignments,
        )
        mismatch_replacements = _repeat_replacements(
            captured,
            head_count=b,
            event_count=events,
            kind="mismatch",
            mismatch=safe_mismatch,
        )
        _, z_restore_mismatch, _ = prepared.backend.patch_individual_heads(
            event_targets,
            mismatch_replacements,
            event_assignments,
        )
        _, z_inject_mismatch, _ = prepared.backend.patch_individual_heads(
            base_targets,
            mismatch_replacements,
            event_assignments,
        )
        clean_repeated = np.broadcast_to(
            z_clean.reshape(1, 1, -1), (b, events, z_clean.shape[-1])
        )
        event_repeated = np.broadcast_to(
            z_event.reshape(1, events, -1), (b, events, z_event.shape[-1])
        )
        matched = patch_response(
            clean_repeated.reshape(b * events, -1),
            event_repeated.reshape(b * events, -1),
            z_restore.detach().cpu().numpy().reshape(b * events, -1),
            z_inject.detach().cpu().numpy().reshape(b * events, -1),
            epsilon=prepared.config.numerical.effect_floor,
        )
        mismatched = patch_response(
            clean_repeated.reshape(b * events, -1),
            event_repeated.reshape(b * events, -1),
            z_restore_mismatch.detach().cpu().numpy().reshape(b * events, -1),
            z_inject_mismatch.detach().cpu().numpy().reshape(b * events, -1),
            epsilon=prepared.config.numerical.effect_floor,
        )
        necessity = donor_necessity(
            clean_repeated.reshape(b * events, -1),
            event_repeated.reshape(b * events, -1),
            np.repeat(z_ablated[:, 0:1], events, axis=1).reshape(b * events, -1),
            z_ablated[:, 1:].reshape(b * events, -1),
            epsilon=prepared.config.numerical.effect_floor,
        )
        target_slice = slice(start, start + b)
        reshape = lambda value: np.asarray(value).reshape(b, events)
        endpoints["P_gross_matched"][target_slice] = reshape(
            matched.bidirectional_gross
        )
        endpoints["P_gross_mismatch"][target_slice] = reshape(
            mismatched.bidirectional_gross
        )
        endpoints["G_c"][target_slice] = (
            endpoints["P_gross_matched"][target_slice]
            - endpoints["P_gross_mismatch"][target_slice]
        )
        endpoints["R_gross"][target_slice] = reshape(matched.restoration_gross)
        endpoints["I_gross"][target_slice] = reshape(matched.injection_gross)
        endpoints["R_align"][target_slice] = reshape(matched.restoration_aligned)
        endpoints["I_align"][target_slice] = reshape(matched.injection_aligned)
        concordant = (
            endpoints["R_align"][target_slice] > 0
        ) & (endpoints["I_align"][target_slice] > 0)
        aligned_adjusted = reshape(
            matched.bidirectional_aligned - mismatched.bidirectional_aligned
        )
        endpoints["M_align"][target_slice] = np.where(
            concordant, aligned_adjusted, np.nan
        )
        endpoints["necessity"][target_slice] = reshape(
            necessity["aligned_necessity"]
        )
        endpoints["gross_necessity"][target_slice] = reshape(
            necessity["gross_necessity"]
        )
        endpoints["event_effect"][target_slice] = reshape(
            necessity["event_effect"]
        )
        if exact_injection is not None:
            exact_injection[target_slice] = (
                z_inject.detach().cpu().numpy().reshape(b, events, -1)
                - clean_repeated
            )
        self_targets = [base] * b
        self_replacements = tuple(
            layer[0:1].expand(b, *layer.shape[1:])
            for layer in captured.transport
        )
        _, self_z, _ = prepared.backend.patch_individual_heads(
            self_targets, self_replacements, selected_heads
        )
        same_condition_max = max(
            same_condition_max,
            float(
                np.max(
                    np.abs(
                        self_z.detach().cpu().numpy()
                        - np.repeat(z_clean, b, axis=0)
                    )
                )
            ),
        )
        prepared.progress.emit(
            "causal_head_batch_complete",
            graph_id=graph_id,
            channel=channel,
            arm=arm,
            split=stage,
            completed_heads=start + b,
            total_heads=len(heads),
            events=events,
        )
    for name in ("P_gross_mismatch", "G_c", "M_align"):
        endpoints[name][:, ~controlled] = np.nan
    within_tolerance(
        same_condition_max,
        prepared.config.numerical.reconstruction_tolerance,
        "pe_refinement.same_condition_patch",
        "same-condition individual-head patch response",
        context={
            "graph": graph_id,
            "channel": channel,
            "arm": arm,
            "split": stage,
        },
    )
    taylor = None
    if clean_taylor is not None:
        import torch

        transport = torch.stack(captured.transport, dim=1)
        q = project_transport(
            transport[0:1] - transport[1:],
            clean_taylor.transport,
        )
        predicted = (
            -q.sum(dim=-2)
            .permute(1, 2, 0, 3)
            .reshape(len(heads), events, -1)
            .detach()
            .cpu()
            .numpy()
        )
        if exact_injection.shape[-1] < predicted.shape[-1]:
            raise RuntimeError(
                "Taylor exact injection has fewer outputs than the graph-local "
                "clean Jacobian"
            )
        padded_output_width = int(exact_injection.shape[-1])
        actual_output_width = int(predicted.shape[-1])
        padding = exact_injection[..., actual_output_width:]
        padding_max = (
            float(np.max(np.abs(padding))) if padding.size else 0.0
        )
        within_tolerance(
            padding_max,
            prepared.config.numerical.reconstruction_tolerance,
            "pe_refinement.taylor_output_padding",
            "padded Taylor output response",
            context={
                "graph": graph_id,
                "channel": channel,
                "arm": arm,
                "split": stage,
                "actual_outputs": actual_output_width,
                "padded_outputs": padded_output_width,
            },
        )
        exact_graph_outputs = exact_injection[..., :actual_output_width]
        taylor = {
            **_taylor_metrics(
                predicted,
                exact_graph_outputs,
                effect_floor=prepared.config.numerical.effect_floor,
            ),
            "predicted": predicted,
            "exact": exact_graph_outputs,
            "actual_output_width": actual_output_width,
            "padded_output_width": padded_output_width,
            "padding_max": padding_max,
        }
    if not all(
        np.isfinite(value[:, controlled]).all()
        for name, value in endpoints.items()
        if name != "M_align"
    ):
        raise RuntimeError("non-finite controlled causal endpoint entered the cache")
    return {
        "graph_id": graph_id,
        "channel": channel,
        "arm": arm,
        "split": stage,
        "head_order": heads,
        "records": records,
        "mismatch_indices": mismatch,
        "controlled": controlled,
        "endpoints": endpoints,
        "same_condition_patch_max": same_condition_max,
        "taylor": taylor,
    }


def _run_causal_component(
    prepared: PreparedPERefinement,
    *,
    channel: str,
    arm: str | None,
    split: str,
    manifest: Mapping[int, Sequence[Mapping[str, Any]]],
    clean_taylor: Mapping[int, Any],
) -> dict[str, Any]:
    graph_ids = causal_graph_ids(prepared, split)
    store = ProtectedShardStore(
        prepared, "common" if channel == "semantic" else f"arms/{arm}"
    )
    stage = f"causal/{split}/{channel}"
    shards: dict[int, Any] = {}
    for completed, graph_id in enumerate(graph_ids, start=1):
        graph_id = int(graph_id)
        cached = (
            store.load(stage, f"graph_{graph_id:06d}")
            if prepared.config.resume and not prepared.config.force
            else None
        )
        cache_hit = cached is not None
        if cached is not None:
            _verify_cached_records(
                cached["records"],
                manifest[graph_id],
                context=f"{stage}/graph_{graph_id:06d}",
            )
        if cached is None:
            trial = prepared
            while True:
                try:
                    cached = _causal_graph(
                        trial,
                        graph_id=graph_id,
                        channel=channel,
                        arm=arm,
                        stage=split,
                        manifest_rows=manifest[graph_id],
                        clean_taylor=clean_taylor.get(graph_id),
                    )
                    break
                except RuntimeError as error:
                    head_batch = int(trial.config.head_batch_size)
                    if (
                        not prepared.config.execution.oom_backoff
                        or head_batch <= 1
                        or not is_cuda_oom(error)
                    ):
                        raise
                    reduced = max(1, head_batch // 2)
                    prepared.progress.emit(
                        "causal_oom_backoff",
                        graph_id=graph_id,
                        channel=channel,
                        arm=arm,
                        split=split,
                        previous_head_batch=head_batch,
                        new_head_batch=reduced,
                    )
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                    trial = dataclasses.replace(
                        prepared,
                        config=dataclasses.replace(
                            prepared.config,
                            head_batch_size=reduced,
                        ),
                    )
            store.save(stage, f"graph_{graph_id:06d}", cached)
        shards[graph_id] = cached
        prepared.progress.emit(
            "causal_graph_complete",
            graph_id=graph_id,
            channel=channel,
            arm=arm,
            split=split,
            completed=completed,
            total=len(graph_ids),
            cache_hit=cache_hit,
        )
    summary = {
        "version": PE_REFINEMENT_VERSION,
        "channel": channel,
        "arm": arm,
        "split": split,
        "graph_ids": list(graph_ids),
        "manifest_hash": stable_hash(
            {str(graph_id): list(manifest[graph_id]) for graph_id in graph_ids}
        ),
        "support": {
            "graphs": len(graph_ids),
            "events": int(sum(len(shards[graph_id]["records"]) for graph_id in graph_ids)),
            "controlled_events": int(
                sum(np.asarray(shards[graph_id]["controlled"]).sum() for graph_id in graph_ids)
            ),
            "taylor_graphs": int(
                sum(shards[graph_id]["taylor"] is not None for graph_id in graph_ids)
            ),
        },
        "same_condition_patch_max": float(
            max(shards[graph_id]["same_condition_patch_max"] for graph_id in graph_ids)
        ),
    }
    store.save(stage, "summary", summary)
    return summary


def run_common_causal(prepared: PreparedPERefinement) -> dict[str, Any]:
    store = ProtectedShardStore(prepared, "common")
    taylor_ids = taylor_graph_ids(prepared)
    clean = ensure_clean_jacobians(
        prepared,
        taylor_ids,
        population="taylor",
        allow_compute=True,
    )
    summaries = {}
    for split in CAUSAL_SPLITS:
        with audit_scope(
            f"{TASK_NAME}:seed{prepared.seed}:common-causal:{split}"
        ) as split_scope:
            graph_ids = causal_graph_ids(prepared, split)
            manifest = semantic_event_manifest(prepared, split, graph_ids)
            store.save_json("manifests", f"semantic_causal_{split}", manifest)
            summaries[split] = _run_causal_component(
                prepared,
                channel="semantic",
                arm=None,
                split=split,
                manifest=manifest,
                clean_taylor=clean,
            )
        findings = log_summary(
            split_scope,
            header=f"common-causal {split} seed={prepared.seed}",
        )
        store.save_json("audits", f"common-causal-{split}", findings)
        store.save("audits", f"common-causal-{split}", findings)
    store.save("causal", "semantic_summary", summaries)
    return summaries


def run_arm_causal(
    prepared: PreparedPERefinement,
    arm: str,
) -> dict[str, Any]:
    store = ProtectedShardStore(prepared, f"arms/{arm}")
    taylor_ids = taylor_graph_ids(prepared)
    clean = ensure_clean_jacobians(
        prepared,
        taylor_ids,
        population="taylor",
        allow_compute=False,
    )
    summaries = {}
    for split in CAUSAL_SPLITS:
        with audit_scope(
            f"{TASK_NAME}:seed{prepared.seed}:arm-{arm}:causal:{split}"
        ) as split_scope:
            graph_ids = causal_graph_ids(prepared, split)
            manifest = structural_pair_manifest(prepared, split, graph_ids)
            store.save_json("manifests", f"structural_causal_{split}", manifest)
            summaries[split] = _run_causal_component(
                prepared,
                channel="structural",
                arm=arm,
                split=split,
                manifest=manifest,
                clean_taylor=clean,
            )
        findings = log_summary(
            split_scope,
            header=f"arm-{arm} causal {split} seed={prepared.seed}",
        )
        store.save_json("audits", f"causal-{split}", findings)
        store.save("audits", f"causal-{split}", findings)
    store.save("causal", "structural_summary", summaries)
    return summaries


def run_common_ablation(prepared: PreparedPERefinement) -> dict[str, Any]:
    """Compute clean single-head necessity once; every arm/score system reuses it."""

    graph_ids = tuple(int(value) for value in prepared.splits.clean_ablation)
    heads = _head_order(prepared)
    store = ProtectedShardStore(prepared, "common")
    stage = "clean_ablation"
    shards: dict[int, Any] = {}
    for completed, graph_id in enumerate(graph_ids, start=1):
        cached = (
            store.load(stage, f"graph_{graph_id:06d}")
            if prepared.config.resume and not prepared.config.force
            else None
        )
        cache_hit = cached is not None
        if cached is None:
            base = prepared.runtime.eval_ds[graph_id]
            clean = prepared.backend.capture(
                [base], require_grad=False, include_virtual_transport=True
            )
            clean_prediction = clean.prediction.detach()
            clean_target = clean.target.detach()
            clean_z = clean.z.detach().cpu().numpy()
            clean_loss = float(
                prepared.backend.loss_per_graph(
                    clean_prediction, clean_target
                )[0].item()
            )
            movement = np.empty(len(heads), dtype=np.float64)
            loss_change = np.empty(len(heads), dtype=np.float64)
            batch_size = int(prepared.config.head_batch_size)
            for start in range(0, len(heads), batch_size):
                selected = heads[start : start + batch_size]
                prediction, z_ablated, target = prepared.backend.ablate_individual_heads(
                    [base] * len(selected), selected
                )
                movement[start : start + len(selected)] = np.linalg.norm(
                    z_ablated.detach().cpu().numpy()
                    - np.repeat(clean_z, len(selected), axis=0),
                    axis=-1,
                )
                loss_change[start : start + len(selected)] = (
                    prepared.backend.loss_per_graph(prediction, target)
                    .detach()
                    .cpu()
                    .numpy()
                    - clean_loss
                )
            if not np.isfinite(movement).all() or not np.isfinite(loss_change).all():
                raise RuntimeError("non-finite clean-ablation endpoint entered the cache")
            cached = {
                "graph_id": graph_id,
                "head_order": heads,
                "prediction_movement": movement,
                "loss_change": loss_change,
            }
            store.save(stage, f"graph_{graph_id:06d}", cached)
        shards[graph_id] = cached
        prepared.progress.emit(
            "clean_ablation_graph_complete",
            graph_id=graph_id,
            completed=completed,
            total=len(graph_ids),
            cache_hit=cache_hit,
        )
    summary = {
        "version": PE_REFINEMENT_VERSION,
        "graph_ids": list(graph_ids),
        "prediction_movement": np.mean(
            np.stack([shards[key]["prediction_movement"] for key in graph_ids]),
            axis=0,
        ).reshape(prepared.runtime.L, prepared.runtime.H),
        "loss_change": np.mean(
            np.stack([shards[key]["loss_change"] for key in graph_ids]),
            axis=0,
        ).reshape(prepared.runtime.L, prepared.runtime.H),
        "support": {"graphs": len(graph_ids), "heads": len(heads)},
    }
    store.save(stage, "summary", summary)
    return summary


def run_common_component(
    config: PERefinementConfig,
    seed: int,
    component: str,
) -> dict[str, Any]:
    if component not in {"common-scores", "common-causal", "common-ablation"}:
        raise ValueError(component)
    prepared = prepare_pe_refinement(config, seed, component=component)
    prepared.progress.start()
    try:
        with audit_scope(f"{TASK_NAME}:seed{seed}:{component}") as scope:
            with prepared.progress.component(component):
                if component == "common-scores":
                    from .runner import _model_audits

                    model_audits = _model_audits(
                        prepared.runtime,
                        prepared.backend,
                        prepared.task,
                        config,
                    )
                    result = run_common_scores(prepared)
                    result["model_audits"] = model_audits
                elif component == "common-causal":
                    result = run_common_causal(prepared)
                else:
                    result = run_common_ablation(prepared)
        findings = log_summary(scope, header=f"{component} seed={seed}")
        store = ProtectedShardStore(prepared, "common")
        store.save_json("audits", component, findings)
        store.save("audits", component, findings)
        if component == "common-scores":
            store.save("audits", "model", result["model_audits"])
        return result
    finally:
        prepared.progress.close()


def run_arm_component(
    config: PERefinementConfig,
    seed: int,
    arm: str,
) -> dict[str, Any]:
    if arm not in STRUCTURAL_ARMS:
        raise ValueError(arm)
    component = f"arm-{arm}"
    prepared = prepare_pe_refinement(config, seed, component=component)
    prepared.progress.start()
    try:
        with audit_scope(f"{TASK_NAME}:seed{seed}:{component}") as scope:
            with audit_scope(
                f"{TASK_NAME}:seed{seed}:{component}:scores"
            ) as score_scope:
                with prepared.progress.component("scores", context={"arm": arm}):
                    scores = run_arm_scores(prepared, arm)
            score_findings = log_summary(
                score_scope,
                header=f"{component} scores seed={seed}",
            )
            score_store = ProtectedShardStore(prepared, f"arms/{arm}")
            score_store.save_json("audits", "scores", score_findings)
            score_store.save("audits", "scores", score_findings)
            with prepared.progress.component("causal", context={"arm": arm}):
                causal = run_arm_causal(prepared, arm)
        findings = log_summary(scope, header=f"{component} seed={seed}")
        store = ProtectedShardStore(prepared, f"arms/{arm}")
        store.save_json("audits", "worker", findings)
        store.save("audits", "worker", findings)
        result = {"scores": scores, "causal": causal}
        store.save("worker", "summary", result)
        return result
    finally:
        prepared.progress.close()


def _read_protected(
    config: PERefinementConfig,
    *,
    seed: int,
    namespace: str,
    stage: str,
    name: str,
) -> Any:
    import torch

    path = (
        config.root
        / TASK_NAME
        / f"seed_{int(seed)}"
        / namespace
        / stage
        / f"{name}.pt"
    )
    if not path.exists():
        raise FileNotFoundError(f"required PE-refinement cache is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise StaleCacheError(f"required PE-refinement cache is malformed: {path}")
    metadata = payload.get("metadata", {})
    expected_checkpoint = _cached_checkpoint_sha(
        str(_checkpoint_path(config, seed).expanduser().resolve())
    )
    expected = {
        "version": PE_REFINEMENT_VERSION,
        "scientific_fingerprint": config.fingerprint,
        "task": TASK_NAME,
        "seed": int(seed),
        "checkpoint_sha256": expected_checkpoint,
        "repository_commit": _repository_commit(),
        "namespace": namespace,
    }
    stored_contract = (
        metadata.get("contract", {}) if isinstance(metadata, Mapping) else {}
    )
    if isinstance(stored_contract, Mapping) and "split_fingerprint" in stored_contract:
        # Finalization does not rebuild the dataset, but the stored split remains a
        # required scientific identity field and is checked for internal consistency.
        expected["split_fingerprint"] = stored_contract["split_fingerprint"]
    _validate_protected_metadata(metadata, expected, path=path)
    return payload["value"]


def _aggregate_causal_shards(
    config: PERefinementConfig,
    *,
    seed: int,
    namespace: str,
    channel: str,
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    summary = _read_protected(
        config,
        seed=seed,
        namespace=namespace,
        stage=f"causal/{split}/{channel}",
        name="summary",
    )
    graph_values: dict[str, list[np.ndarray]] = {}
    taylor_rows: dict[str, list[np.ndarray]] = {}
    dose_rows: list[dict[str, float]] = []
    event_effect_rows: list[np.ndarray] = []
    adjusted_bundles: list[dict[str, Any]] = []
    controlled_events = 0
    total_events = 0
    endpoint_names: tuple[str, ...] = ()
    head_count = 0
    for graph_id in summary["graph_ids"]:
        shard = _read_protected(
            config,
            seed=seed,
            namespace=namespace,
            stage=f"causal/{split}/{channel}",
            name=f"graph_{int(graph_id):06d}",
        )
        controlled = np.asarray(shard["controlled"], dtype=bool)
        endpoint_names = tuple(shard["endpoints"])
        head_count = len(shard["head_order"])
        total_events += int(len(controlled))
        controlled_events += int(controlled.sum())
        for name, values in shard["endpoints"].items():
            endpoint_mask = (
                controlled
                if name in {"P_gross_mismatch", "G_c", "M_align"}
                else np.ones_like(controlled, dtype=bool)
            )
            endpoint_records = [
                row
                for row, keep in zip(shard["records"], endpoint_mask)
                if keep
            ]
            values = np.asarray(values, dtype=np.float64)[:, endpoint_mask].T
            if name == "event_effect" and values.size:
                # Clean-event displacement is head-independent; retain one copy per event.
                event_effect_rows.append(values[:, 0])
            if not endpoint_records:
                continue
            graph_values.setdefault(name, []).append(
                _hierarchical_event_mean(
                    values,
                    endpoint_records,
                    nan=name == "M_align",
                )
            )
        for row in shard["records"]:
            dose_rows.append(
                {
                    "input_dose": float(row.get("input_dose", row.get("dose", np.nan))),
                    "rrwp_rms_dose": float(row.get("rrwp_rms_dose", np.nan)),
                    "degree_gap": float(row.get("degree_gap", np.nan)),
                }
            )
        if controlled.any():
            adjusted_bundles.append(
                {
                    "graph": int(graph_id),
                    "records": [
                        row
                        for row, keep in zip(shard["records"], controlled)
                        if keep
                    ],
                    "values": np.asarray(
                        shard["endpoints"]["G_c"], dtype=np.float64
                    )[:, controlled].T,
                }
            )
        if shard["taylor"] is not None:
            for name in (
                "predicted_norm",
                "exact_norm",
                "cosine",
                "norm_ratio",
                "relative_error",
                "estimable",
            ):
                taylor_rows.setdefault(name, []).append(
                    np.asarray(shard["taylor"][name])
                )
    endpoints = {
        name: (
            np.mean(np.stack(graph_values[name]), axis=0)
            if graph_values.get(name)
            else np.full(head_count, np.nan, dtype=np.float64)
        )
        for name in endpoint_names
    }
    def stratified_adjusted(
        labels: Sequence[str],
        *,
        order: Sequence[str],
    ) -> dict[str, Any]:
        output = {}
        cursor = 0
        labelled_bundles = []
        for bundle in adjusted_bundles:
            count = len(bundle["records"])
            labelled_bundles.append(
                {
                    **bundle,
                    "labels": tuple(labels[cursor : cursor + count]),
                }
            )
            cursor += count
        if cursor != len(labels):
            raise RuntimeError("stratified causal labels do not align with event bundles")
        for label in order:
            graph_rows = []
            events = 0
            for bundle in labelled_bundles:
                mask = np.asarray(
                    [value == label for value in bundle["labels"]],
                    dtype=bool,
                )
                if not mask.any():
                    continue
                records = [
                    row
                    for row, keep in zip(bundle["records"], mask)
                    if keep
                ]
                graph_rows.append(
                    _hierarchical_event_mean(bundle["values"][mask], records)
                )
                events += int(mask.sum())
            output[label] = {
                "G_c": (
                    np.mean(np.stack(graph_rows), axis=0)
                    if graph_rows
                    else np.full(head_count, np.nan, dtype=np.float64)
                ),
                "graphs": len(graph_rows),
                "events": events,
            }
        return output

    flattened_records = [
        row for bundle in adjusted_bundles for row in bundle["records"]
    ]
    input_doses = np.asarray(
        [float(row.get("input_dose", row.get("dose", np.nan))) for row in flattened_records],
        dtype=np.float64,
    )
    if input_doses.size:
        q_low, q_high = np.quantile(input_doses, (1.0 / 3.0, 2.0 / 3.0))
        dose_labels = [
            "near" if value <= q_low else "middle" if value <= q_high else "far"
            for value in input_doses
        ]
    else:
        q_low = q_high = np.nan
        dose_labels = []
    pair_labels = [
        str(row.get("distance_stratum", "not_applicable"))
        for row in flattened_records
    ]
    degree_labels = [
        (
            "gap_0"
            if int(row.get("degree_gap", 0)) == 0
            else "gap_1"
            if int(row.get("degree_gap", 0)) == 1
            else "gap_2_plus"
        )
        for row in flattened_records
    ]
    metadata = {
        "support": {
            "graphs": len(summary["graph_ids"]),
            "events": total_events,
            "controlled_events": controlled_events,
            "controlled_fraction": (
                controlled_events / total_events if total_events else 0.0
            ),
        },
        "doses": dose_rows,
        "event_effects": (
            np.concatenate(event_effect_rows)
            if event_effect_rows
            else np.asarray([], dtype=np.float64)
        ),
        "stratified_adjusted": {
            "input_dose_thresholds": {
                "q33": float(q_low),
                "q67": float(q_high),
            },
            "input_dose": stratified_adjusted(
                dose_labels, order=("near", "middle", "far")
            ),
            "rrwp_pair_distance": stratified_adjusted(
                pair_labels,
                order=("near", "middle", "far", "not_applicable"),
            ),
            "degree_gap": stratified_adjusted(
                degree_labels,
                order=("gap_0", "gap_1", "gap_2_plus"),
            ),
        },
        "taylor": {
            name: (
                np.concatenate([value.reshape(-1) for value in values])
                if name != "estimable"
                else np.concatenate([value.reshape(-1) for value in values]).astype(bool)
            )
            for name, values in taylor_rows.items()
        },
    }
    return endpoints, metadata


def _spearman(x: Any, y: Any) -> dict[str, float]:
    from scipy.stats import spearmanr

    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return {"rho": np.nan, "p": np.nan, "n": int(mask.sum())}
    result = spearmanr(left[mask], right[mask])
    return {
        "rho": float(result.statistic),
        "p": float(result.pvalue),
        "n": int(mask.sum()),
    }


def _within_layer_permutation(
    x: Any,
    y: Any,
    layers: Any,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    layers = np.asarray(layers, dtype=np.int64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, layers = x[mask], y[mask], layers[mask]
    observed = _spearman(x, y)["rho"]
    if len(x) < 3 or not np.isfinite(observed):
        return {"rho": observed, "p": np.nan, "replicates": int(replicates)}
    rng = np.random.default_rng(int(seed))
    null = _permuted_spearman_values(
        x,
        y,
        layers,
        replicates=int(replicates),
        rng=rng,
    )
    exceed = int(np.sum(np.isfinite(null) & (np.abs(null) >= abs(observed))))
    return {
        "rho": float(observed),
        "p": float((exceed + 1) / (int(replicates) + 1)),
        "replicates": int(replicates),
    }


def _permuted_spearman_values(
    x: Any,
    y: Any,
    layers: Any,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorised within-layer Spearman null with exact global tie ranks."""

    from scipy.stats import rankdata

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    layers = np.asarray(layers, dtype=np.int64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, layers = x[mask], y[mask], layers[mask]
    output = np.full(int(replicates), np.nan, dtype=np.float64)
    if len(x) < 3:
        return output
    x_rank = np.asarray(rankdata(x), dtype=np.float64)
    y_rank = np.asarray(rankdata(y), dtype=np.float64)
    x_centered = x_rank - float(np.mean(x_rank))
    y_centered = y_rank - float(np.mean(y_rank))
    denominator = float(
        np.sqrt(np.sum(np.square(x_centered)) * np.sum(np.square(y_centered)))
    )
    if denominator <= 0.0:
        return output
    permuted = np.broadcast_to(y_rank, (int(replicates), len(y_rank))).copy()
    for layer in np.unique(layers):
        indices = np.flatnonzero(layers == layer)
        if len(indices) < 2:
            continue
        orders = np.argsort(
            rng.random((int(replicates), len(indices))),
            axis=1,
        )
        permuted[:, indices] = y_rank[indices][orders]
    output[:] = (
        (permuted - float(np.mean(y_rank))) @ x_centered
    ) / denominator
    return output


def _association(
    x: Any,
    y: Any,
    layers: Any,
    *,
    config: PERefinementConfig,
    seed_offset: int,
) -> dict[str, Any]:
    return {
        "spearman": _spearman(x, y),
        "within_layer_permutation": _within_layer_permutation(
            x,
            y,
            layers,
            replicates=int(config.sizes.permutation_replicates),
            seed=int(config.analysis_seed) + int(seed_offset),
        ),
    }


def _safe_summary(values: Any) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {
            "n": 0,
            "median": np.nan,
            "mean": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def _seed_candidate_analysis(
    config: PERefinementConfig,
    *,
    seed: int,
    arm: str,
    score_system: str,
    split: str,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    if score_system not in SCORE_SYSTEMS:
        raise ValueError(score_system)
    common_score = _read_protected(
        config,
        seed=seed,
        namespace="common",
        stage="scores/semantic",
        name="summary",
    )
    structural_score = _read_protected(
        config,
        seed=seed,
        namespace=f"arms/{arm}",
        stage="scores/structural",
        name="summary",
    )
    ablation = _read_protected(
        config,
        seed=seed,
        namespace="common",
        stage="clean_ablation",
        name="summary",
    )
    semantic_endpoints, semantic_meta = _aggregate_causal_shards(
        config,
        seed=seed,
        namespace="common",
        channel="semantic",
        split=split,
    )
    structural_endpoints, structural_meta = _aggregate_causal_shards(
        config,
        seed=seed,
        namespace=f"arms/{arm}",
        channel="structural",
        split=split,
    )
    semantic_raw = np.asarray(
        common_score["raw"][score_system], dtype=np.float64
    )
    structural_raw = np.asarray(
        structural_score["raw"][score_system], dtype=np.float64
    )
    coordinates = head_coordinates(
        semantic_raw,
        structural_raw,
        score_floor=config.numerical.score_floor,
        epsilon=config.numerical.selectivity_epsilon,
        activity_floor=0.20,
    )
    if not coordinates.estimable:
        raise RuntimeError(
            f"non-estimable coordinates for seed={seed} arm={arm} score={score_system}"
        )
    shape = semantic_raw.shape
    layers = np.repeat(np.arange(shape[0]), shape[1])
    active = coordinates.active.reshape(-1)
    J = coordinates.joint_sensitivity.reshape(-1)
    D = coordinates.selectivity.reshape(-1)
    s_sem = semantic_raw.reshape(-1)
    s_str = structural_raw.reshape(-1)
    n_sem = coordinates.normalized_semantic.reshape(-1)
    n_str = coordinates.normalized_structural.reshape(-1)
    endpoint = {
        "semantic": {
            key: np.asarray(value, dtype=np.float64).reshape(-1)
            for key, value in semantic_endpoints.items()
        },
        "structural": {
            key: np.asarray(value, dtype=np.float64).reshape(-1)
            for key, value in structural_endpoints.items()
        },
    }
    gross_scale = {
        channel: float(np.nanmean(endpoint[channel]["P_gross_matched"]))
        for channel in ("semantic", "structural")
    }
    necessity_scale = {
        channel: float(np.nanmean(endpoint[channel]["gross_necessity"]))
        for channel in ("semantic", "structural")
    }
    for name, values in {
        **{f"gross_{key}": value for key, value in gross_scale.items()},
        **{
            f"necessity_{key}": value
            for key, value in necessity_scale.items()
        },
    }.items():
        if not np.isfinite(values) or values <= config.numerical.effect_floor:
            audit_check(
                False,
                "pe_refinement.reference_scale",
                "causal reference scale is below the registered effect floor",
                observed=float(values),
                tolerance=float(config.numerical.effect_floor),
                context={
                    "seed": seed,
                    "arm": arm,
                    "score_system": score_system,
                    "channel": name,
                },
            )
    g_sem = endpoint["semantic"]["G_c"] / gross_scale["semantic"]
    g_str = endpoint["structural"]["G_c"] / gross_scale["structural"]
    gross_total = 0.5 * (g_sem + g_str)
    gross_contrast = g_sem - g_str
    necessity_sem = endpoint["semantic"]["necessity"] / necessity_scale["semantic"]
    necessity_str = endpoint["structural"]["necessity"] / necessity_scale["structural"]
    necessity_total = 0.5 * (necessity_sem + necessity_str)
    necessity_contrast = necessity_sem - necessity_str
    clean_movement = np.asarray(
        ablation["prediction_movement"], dtype=np.float64
    ).reshape(-1)
    clean_loss = np.asarray(ablation["loss_change"], dtype=np.float64).reshape(-1)
    association_vectors = {
        "S_semantic_vs_G_semantic": (s_sem, endpoint["semantic"]["G_c"]),
        "S_structural_vs_G_structural": (s_str, endpoint["structural"]["G_c"]),
        "S_semantic_vs_G_structural_control": (
            s_sem,
            endpoint["structural"]["G_c"],
        ),
        "S_structural_vs_G_semantic_control": (
            s_str,
            endpoint["semantic"]["G_c"],
        ),
        "S_structural_vs_P_matched": (
            s_str,
            endpoint["structural"]["P_gross_matched"],
        ),
        "S_structural_vs_P_mismatch": (
            s_str,
            endpoint["structural"]["P_gross_mismatch"],
        ),
        "J_vs_gross_total": (J, gross_total),
        "D_rel_vs_gross_contrast": (D[active], gross_contrast[active]),
        "J_vs_necessity_total": (J, necessity_total),
        "D_rel_vs_necessity_contrast": (
            D[active],
            necessity_contrast[active],
        ),
        "J_vs_clean_prediction_movement": (J, clean_movement),
        "J_vs_clean_loss_change": (J, clean_loss),
    }
    for stratification in ("input_dose", "rrwp_pair_distance", "degree_gap"):
        for label, row in structural_meta["stratified_adjusted"][
            stratification
        ].items():
            if int(row["events"]) > 0:
                association_vectors[
                    f"S_structural_vs_G_structural__{stratification}__{label}"
                ] = (
                    s_str,
                    np.asarray(row["G_c"], dtype=np.float64),
                )
    vector_layers = {
        name: (layers[active] if name.startswith("D_rel") else layers)
        for name in association_vectors
    }
    associations = {
        name: _association(
            x,
            y,
            vector_layers[name],
            config=config,
            seed_offset=seed * 100 + index,
        )
        for index, (name, (x, y)) in enumerate(association_vectors.items())
    }
    vectors = {
        name: {
            "x": np.asarray(x, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
            "layer": np.asarray(vector_layers[name], dtype=np.int64),
        }
        for name, (x, y) in association_vectors.items()
    }
    common_taylor = semantic_meta["taylor"]
    structural_taylor = structural_meta["taylor"]
    taylor_source_split = split
    if split == "confirmation" and (
        np.asarray(common_taylor.get("cosine", [])).size == 0
        or np.asarray(structural_taylor.get("cosine", [])).size == 0
    ):
        _, refinement_semantic_meta = _aggregate_causal_shards(
            config,
            seed=seed,
            namespace="common",
            channel="semantic",
            split="refinement",
        )
        _, refinement_structural_meta = _aggregate_causal_shards(
            config,
            seed=seed,
            namespace=f"arms/{arm}",
            channel="structural",
            split="refinement",
        )
        common_taylor = refinement_semantic_meta["taylor"]
        structural_taylor = refinement_structural_meta["taylor"]
        taylor_source_split = "refinement_preselection_audit"
    structural_coherence = np.asarray(
        structural_score["carrier_coherence"], dtype=np.float64
    ).reshape(-1)
    semantic_coherence = np.asarray(
        common_score["carrier_coherence"], dtype=np.float64
    ).reshape(-1)
    result = {
        "seed": int(seed),
        "arm": arm,
        "score_system": score_system,
        "split": split,
        "scores": {
            "semantic_raw": semantic_raw,
            "structural_raw": structural_raw,
            "semantic_normalized": coordinates.normalized_semantic,
            "structural_normalized": coordinates.normalized_structural,
            "joint_sensitivity": coordinates.joint_sensitivity,
            "selectivity": coordinates.selectivity,
            "active": coordinates.active,
            "carrier_coherence_semantic": semantic_coherence.reshape(shape),
            "carrier_coherence_structural": structural_coherence.reshape(shape),
        },
        "range_restriction": {
            "semantic_structural_rho": _spearman(n_sem, n_str),
            "D_rel_sd": float(np.std(D)),
            "D_rel_iqr": float(np.quantile(D, 0.75) - np.quantile(D, 0.25)),
            "active_heads": int(active.sum()),
            "semantic_tail_D_gt_0_20": int(np.sum(active & (D > 0.20))),
            "structural_tail_D_lt_minus_0_20": int(np.sum(active & (D < -0.20))),
            "balanced_abs_D_le_0_20": int(np.sum(active & (np.abs(D) <= 0.20))),
        },
        "causal": {
            "gross_reference_scale": gross_scale,
            "necessity_reference_scale": necessity_scale,
            "semantic": endpoint["semantic"],
            "structural": endpoint["structural"],
            "gross_total_for_J": gross_total,
            "gross_contrast_for_D_rel": gross_contrast,
            "necessity_total_for_J": necessity_total,
            "necessity_contrast_for_D_rel": necessity_contrast,
            "clean_prediction_movement": clean_movement,
            "clean_loss_change": clean_loss,
        },
        "associations": associations,
        "four_way_structural": {
            "matched": associations["S_structural_vs_P_matched"],
            "mismatch": associations["S_structural_vs_P_mismatch"],
            "adjusted": associations["S_structural_vs_G_structural"],
        },
        "four_way_channel_matrix": {
            "semantic_score_semantic_response": associations[
                "S_semantic_vs_G_semantic"
            ],
            "semantic_score_structural_response": associations[
                "S_semantic_vs_G_structural_control"
            ],
            "structural_score_semantic_response": associations[
                "S_structural_vs_G_semantic_control"
            ],
            "structural_score_structural_response": associations[
                "S_structural_vs_G_structural"
            ],
        },
        "support": {
            "semantic": semantic_meta["support"],
            "structural": structural_meta["support"],
            "semantic_score": common_score["support"],
            "structural_score": structural_score["support"],
        },
        "causal_endpoint_distribution": {
            "semantic_matched": _safe_summary(
                endpoint["semantic"]["P_gross_matched"]
            ),
            "semantic_mismatch": _safe_summary(
                endpoint["semantic"]["P_gross_mismatch"]
            ),
            "semantic_adjusted": _safe_summary(endpoint["semantic"]["G_c"]),
            "structural_matched": _safe_summary(
                endpoint["structural"]["P_gross_matched"]
            ),
            "structural_mismatch": _safe_summary(
                endpoint["structural"]["P_gross_mismatch"]
            ),
            "structural_adjusted": _safe_summary(
                endpoint["structural"]["G_c"]
            ),
            "structural_adjusted_negative_head_fraction": float(
                np.mean(endpoint["structural"]["G_c"] < 0)
            ),
            "semantic_event_below_effect_floor_fraction": float(
                np.mean(
                    np.asarray(semantic_meta["event_effects"])
                    <= config.numerical.effect_floor
                )
            ) if np.asarray(semantic_meta["event_effects"]).size else np.nan,
            "structural_event_below_effect_floor_fraction": float(
                np.mean(
                    np.asarray(structural_meta["event_effects"])
                    <= config.numerical.effect_floor
                )
            ) if np.asarray(structural_meta["event_effects"]).size else np.nan,
        },
        "intervention_dose": {
            "structural_input": _safe_summary(
                [row["input_dose"] for row in structural_meta["doses"]]
            ),
            "structural_rrwp": _safe_summary(
                [row["rrwp_rms_dose"] for row in structural_meta["doses"]]
            ),
            "degree_gap": _safe_summary(
                [row["degree_gap"] for row in structural_meta["doses"]]
            ),
            "stratified_adjusted_support": {
                name: {
                    label: {
                        "graphs": int(row["graphs"]),
                        "events": int(row["events"]),
                    }
                    for label, row in values.items()
                }
                for name, values in structural_meta[
                    "stratified_adjusted"
                ].items()
                if name != "input_dose_thresholds"
            },
            "input_dose_thresholds": structural_meta[
                "stratified_adjusted"
            ]["input_dose_thresholds"],
        },
        "cancellation": {
            "semantic_carrier_coherence": _safe_summary(semantic_coherence),
            "structural_carrier_coherence": _safe_summary(structural_coherence),
            "semantic_cancellation_ratio": _safe_summary(1.0 - semantic_coherence),
            "structural_cancellation_ratio": _safe_summary(
                1.0 - structural_coherence
            ),
        },
        "taylor_fidelity": {
            "source_split": taylor_source_split,
            "semantic": {
                name: _safe_summary(common_taylor.get(name, []))
                for name in ("cosine", "norm_ratio", "relative_error")
            },
            "structural": {
                name: _safe_summary(structural_taylor.get(name, []))
                for name in ("cosine", "norm_ratio", "relative_error")
            },
            "semantic_below_effect_floor": int(
                np.sum(~np.asarray(common_taylor.get("estimable", []), dtype=bool))
            ),
            "structural_below_effect_floor": int(
                np.sum(
                    ~np.asarray(
                        structural_taylor.get("estimable", []), dtype=bool
                    )
                )
            ),
        },
    }
    return result, vectors


def _fisher_mean(rhos: Sequence[float]) -> float:
    values = np.asarray(rhos, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    return float(np.tanh(np.mean(np.arctanh(np.clip(values, -0.999999, 0.999999)))))


def _meta_association(
    seed_results: Sequence[Mapping[str, Any]],
    seed_vectors: Sequence[Mapping[str, np.ndarray]],
    *,
    config: PERefinementConfig,
    name: str,
    permutation_key: str | None = None,
) -> dict[str, Any]:
    from scipy.stats import t

    rhos = [
        float(result["associations"][name]["spearman"]["rho"])
        for result in seed_results
    ]
    finite = np.asarray([value for value in rhos if np.isfinite(value)], dtype=np.float64)
    if finite.size:
        z = np.arctanh(np.clip(finite, -0.999999, 0.999999))
        center = float(np.mean(z))
        if len(z) > 1:
            half = float(t.ppf(0.975, len(z) - 1) * np.std(z, ddof=1) / np.sqrt(len(z)))
            interval = [float(np.tanh(center - half)), float(np.tanh(center + half))]
        else:
            interval = [np.nan, np.nan]
    else:
        interval = [np.nan, np.nan]
    leave_one_out = [
        _fisher_mean([rho for index, rho in enumerate(rhos) if index != omitted])
        for omitted in range(len(rhos))
    ]
    observed = _fisher_mean(rhos)
    rng = np.random.default_rng(
        int(config.analysis_seed)
        + int(
            stable_hash(
                {"association": permutation_key or name},
                length=8,
            ),
            16,
        )
    )
    replicates = int(config.sizes.permutation_replicates)
    seed_null = np.stack(
        [
            _permuted_spearman_values(
                vectors["x"],
                vectors["y"],
                vectors["layer"],
                replicates=replicates,
                rng=rng,
            )
            for vectors in seed_vectors
        ],
        axis=0,
    )
    finite_null = np.isfinite(seed_null)
    clipped = np.arctanh(np.clip(seed_null, -0.999999, 0.999999))
    z_sum = np.where(finite_null, clipped, 0.0).sum(axis=0)
    z_count = finite_null.sum(axis=0)
    aggregate_null = np.full(replicates, np.nan, dtype=np.float64)
    estimable_null = z_count > 0
    aggregate_null[estimable_null] = np.tanh(
        z_sum[estimable_null] / z_count[estimable_null]
    )
    exceed = int(
        np.sum(
            np.isfinite(aggregate_null)
            & np.isfinite(observed)
            & (np.abs(aggregate_null) >= abs(observed))
        )
    )
    permutation_p = (
        float((exceed + 1) / (replicates + 1))
        if np.isfinite(observed)
        else np.nan
    )
    return {
        "per_seed_rho": rhos,
        "fisher_mean_rho": observed,
        "seed_t_interval_95": interval,
        "minimum_rho": float(np.nanmin(finite)) if finite.size else np.nan,
        "maximum_rho": float(np.nanmax(finite)) if finite.size else np.nan,
        "same_sign_seeds": int(
            max(np.sum(finite > 0), np.sum(finite < 0))
        ) if finite.size else 0,
        "leave_one_seed_out_rho": leave_one_out,
        "seed_stratified_within_layer_permutation_p": permutation_p,
        "replicates": replicates,
        "n_seeds": int(finite.size),
        "inference_unit": "trained seed; heads remain nested within seed/layer",
    }


def _candidate_id(arm: str, score_system: str) -> str:
    return f"{arm}__{score_system}"


def _validate_paired_manifests(
    config: PERefinementConfig,
    *,
    split: str,
    arms: Sequence[str],
) -> dict[str, str]:
    if split not in CAUSAL_SPLITS:
        raise ValueError(split)
    if not arms or any(arm not in STRUCTURAL_ARMS for arm in arms):
        raise ValueError("paired-manifest validation requires registered structural arms")
    observed: dict[str, list[str]] = {
        "semantic_scores": [],
        "structural_scores": [],
        f"semantic_{split}": [],
        f"structural_{split}": [],
    }
    for seed in config.seeds:
        observed["semantic_scores"].append(
            str(
                _read_protected(
                    config,
                    seed=seed,
                    namespace="common",
                    stage="scores/semantic",
                    name="summary",
                )["manifest_hash"]
            )
        )
        observed[f"semantic_{split}"].append(
            str(
                _read_protected(
                    config,
                    seed=seed,
                    namespace="common",
                    stage=f"causal/{split}/semantic",
                    name="summary",
                )["manifest_hash"]
            )
        )
        for arm in arms:
            observed["structural_scores"].append(
                str(
                    _read_protected(
                        config,
                        seed=seed,
                        namespace=f"arms/{arm}",
                        stage="scores/structural",
                        name="summary",
                    )["manifest_hash"]
                )
            )
            observed[f"structural_{split}"].append(
                str(
                    _read_protected(
                        config,
                        seed=seed,
                        namespace=f"arms/{arm}",
                        stage=f"causal/{split}/structural",
                        name="summary",
                    )["manifest_hash"]
                )
            )
    for name, values in observed.items():
        if len(set(values)) != 1:
            raise RuntimeError(
                f"paired event manifest differs across arms/seeds for {name}: "
                f"{sorted(set(values))}"
            )
    return {name: values[0] for name, values in observed.items()}


def _candidate_public_summary(
    config: PERefinementConfig,
    *,
    arm: str,
    score_system: str,
    split: str,
) -> tuple[dict[str, Any], dict[int, dict[str, dict[str, np.ndarray]]]]:
    per_seed = []
    vectors_by_seed: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for seed in config.seeds:
        result, vectors = _seed_candidate_analysis(
            config,
            seed=seed,
            arm=arm,
            score_system=score_system,
            split=split,
        )
        per_seed.append(result)
        vectors_by_seed[int(seed)] = vectors
    association_names = tuple(per_seed[0]["associations"])
    population = {
        name: _meta_association(
            per_seed,
            [vectors_by_seed[int(seed)][name] for seed in config.seeds],
            config=config,
            name=name,
            permutation_key=f"{arm}:{score_system}:{split}:{name}",
        )
        for name in association_names
    }
    return {
        "candidate_id": _candidate_id(arm, score_system),
        "arm": arm,
        "score_system": score_system,
        "split": split,
        "per_seed": per_seed,
        "population_associations": population,
        "seed_count": len(per_seed),
    }, vectors_by_seed


def _selection_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    population = candidate["population_associations"]
    per_seed = candidate["per_seed"]
    controlled = [
        float(seed["support"]["structural"]["controlled_fraction"])
        for seed in per_seed
    ]
    taylor_n = [
        int(seed["taylor_fidelity"]["structural"]["cosine"]["n"])
        for seed in per_seed
    ]
    finite_primary = all(
        np.isfinite(
            population[name]["fisher_mean_rho"]
        )
        for name in (
            "S_structural_vs_G_structural",
            "D_rel_vs_gross_contrast",
            "J_vs_gross_total",
        )
    )
    integrity_gate = bool(
        min(controlled, default=0.0) >= 0.90
        and min(taylor_n, default=0) > 0
        and finite_primary
    )
    coherence = [
        float(
            seed["cancellation"]["structural_carrier_coherence"]["median"]
        )
        for seed in per_seed
    ]
    taylor_cosine = [
        float(seed["taylor_fidelity"]["structural"]["cosine"]["median"])
        for seed in per_seed
    ]
    return {
        "candidate_id": candidate["candidate_id"],
        "arm": candidate["arm"],
        "score_system": candidate["score_system"],
        "integrity_support_gate": integrity_gate,
        "minimum_controlled_event_fraction": min(controlled, default=np.nan),
        "structural_score_adjusted_rho": population[
            "S_structural_vs_G_structural"
        ]["fisher_mean_rho"],
        "structural_adjusted_same_sign_seeds": population[
            "S_structural_vs_G_structural"
        ]["same_sign_seeds"],
        "D_rel_causal_contrast_rho": population[
            "D_rel_vs_gross_contrast"
        ]["fisher_mean_rho"],
        "J_total_response_rho": population["J_vs_gross_total"][
            "fisher_mean_rho"
        ],
        "J_clean_necessity_rho": population[
            "J_vs_clean_prediction_movement"
        ]["fisher_mean_rho"],
        "median_structural_taylor_cosine_across_seeds": float(
            np.nanmedian(taylor_cosine)
        ),
        "median_structural_carrier_coherence_across_seeds": float(
            np.nanmedian(coherence)
        ),
    }


def _rank_selection_rows(rows: list[dict[str, Any]]) -> None:
    """Registered lexicographic hierarchy; this ranks but never writes the lock."""

    def ranked(value: Any) -> float:
        numeric = float(value)
        return numeric if np.isfinite(numeric) else float("-inf")

    order = sorted(
        range(len(rows)),
        key=lambda index: (
            int(rows[index]["integrity_support_gate"]),
            int(rows[index]["structural_adjusted_same_sign_seeds"]),
            ranked(rows[index]["structural_score_adjusted_rho"]),
            ranked(rows[index]["D_rel_causal_contrast_rho"]),
            ranked(rows[index]["J_total_response_rho"]),
            ranked(rows[index]["J_clean_necessity_rho"]),
            ranked(rows[index]["median_structural_taylor_cosine_across_seeds"]),
            ranked(rows[index]["median_structural_carrier_coherence_across_seeds"]),
        ),
        reverse=True,
    )
    for rank, index in enumerate(order, start=1):
        rows[index]["registered_rank"] = int(rank)


def _save_figure(
    figure: Any,
    output_dir: Path,
    name: str,
    metadata: Mapping[str, Any],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"{name}.{suffix}"
        figure.savefig(
            path,
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
        paths.append(str(path))
    atomic_json(output_dir / f"{name}.json", dict(metadata))
    return paths


def _candidate_lookup(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(candidate["candidate_id"]): candidate
        for candidate in summary["candidates"]
    }


def render_refinement_figures(
    summary: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, list[str]]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    candidates = _candidate_lookup(summary)
    plotted_arms = tuple(
        arm for arm in STRUCTURAL_ARMS
        if any(candidate["arm"] == arm for candidate in summary["candidates"])
    )
    plotted_systems = tuple(
        system for system in SCORE_SYSTEMS
        if any(candidate["score_system"] == system for candidate in summary["candidates"])
    )
    seed_colours = {
        seed: plt.get_cmap("tab10")(index)
        for index, seed in enumerate(summary["seeds"])
    }
    saved: dict[str, list[str]] = {}

    figure, axes = plt.subplots(
        len(plotted_arms),
        len(plotted_systems),
        figsize=(4.5 * len(plotted_systems), 3.5 * len(plotted_arms)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row, arm in enumerate(plotted_arms):
        for column, system in enumerate(plotted_systems):
            candidate_id = _candidate_id(arm, system)
            if candidate_id not in candidates:
                axes[row, column].set_visible(False)
                continue
            candidate = candidates[candidate_id]
            axis = axes[row, column]
            for seed_result in candidate["per_seed"]:
                score = seed_result["scores"]
                axis.scatter(
                    np.asarray(score["structural_normalized"]).reshape(-1),
                    np.asarray(score["semantic_normalized"]).reshape(-1),
                    s=18,
                    alpha=0.70,
                    color=seed_colours[int(seed_result["seed"])],
                    label=f"seed {seed_result['seed']}" if row == 0 and column == 0 else None,
                )
            axis.plot([0, 3.2], [0, 3.2], "--", color="0.45", linewidth=1)
            axis.set_title(f"{arm.replace('_', ' ')} | {system}")
            axis.set_xlim(0, 3.2)
            axis.set_ylim(0, 3.2)
            if row == len(plotted_arms) - 1:
                axis.set_xlabel("Normalized structural score")
            if column == 0:
                axis.set_ylabel("Normalized semantic score")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Bipartite matching raw head scores (four trained seeds; descriptive points)",
        y=1.0,
    )
    figure.tight_layout()
    saved["raw_scores"] = _save_figure(
        figure,
        output_dir,
        "raw_semantic_structural_scores",
        {
            "split": summary["split"],
            "seeds": summary["seeds"],
            "note": "Points are nested within seed and are not treated as 192 independent replicas.",
        },
    )
    plt.close(figure)

    for system in plotted_systems:
        system_arms = tuple(
            arm for arm in plotted_arms
            if _candidate_id(arm, system) in candidates
        )
        figure, axes = plt.subplots(
            len(system_arms),
            4,
            figsize=(15.5, 3.4 * len(system_arms)),
            squeeze=False,
        )
        channel_columns = (
            (
                "semantic_normalized",
                "semantic",
                "S_semantic_vs_G_semantic",
                "S_sem → G_sem",
            ),
            (
                "semantic_normalized",
                "structural",
                "S_semantic_vs_G_structural_control",
                "S_sem → G_str",
            ),
            (
                "structural_normalized",
                "semantic",
                "S_structural_vs_G_semantic_control",
                "S_str → G_sem",
            ),
            (
                "structural_normalized",
                "structural",
                "S_structural_vs_G_structural",
                "S_str → G_str",
            ),
        )
        for row, arm in enumerate(system_arms):
            candidate = candidates[_candidate_id(arm, system)]
            for column, (score_key, channel, association, title) in enumerate(
                channel_columns
            ):
                axis = axes[row, column]
                for seed_result in candidate["per_seed"]:
                    axis.scatter(
                        np.asarray(seed_result["scores"][score_key]).reshape(-1),
                        np.asarray(
                            seed_result["causal"][channel]["G_c"]
                        ).reshape(-1),
                        s=17,
                        alpha=0.70,
                        color=seed_colours[int(seed_result["seed"])],
                    )
                rho = candidate["population_associations"][association][
                    "fisher_mean_rho"
                ]
                axis.set_title(f"{title}\nseed-level Fisher mean ρ={rho:.2f}")
                axis.axhline(0.0, color="0.6", linewidth=0.8)
                if row == len(system_arms) - 1:
                    axis.set_xlabel(
                        "Normalized semantic score"
                        if score_key.startswith("semantic")
                        else "Normalized structural score"
                    )
                if column == 0:
                    axis.set_ylabel(
                        arm.replace("_", " ") + "\nadjusted causal response"
                    )
        figure.suptitle(
            f"Four-way score/channel causal decomposition | {system} | "
            f"{summary['split']}",
            y=1.0,
        )
        figure.tight_layout()
        saved[f"channel_matrix_{system}"] = _save_figure(
            figure,
            output_dir,
            f"four_way_score_channel_matrix_{system}",
            {
                "split": summary["split"],
                "score_system": system,
                "rows": list(system_arms),
                "columns": [column[3] for column in channel_columns],
                "inference": (
                    "Four per-seed estimates; annotations are Fisher means of "
                    "seed estimates."
                ),
            },
        )
        plt.close(figure)

        figure, axes = plt.subplots(
            len(system_arms),
            3,
            figsize=(12.0, 3.4 * len(system_arms)),
            squeeze=False,
        )
        endpoint_columns = (
            ("P_gross_matched", "Matched gross response", "S_structural_vs_P_matched"),
            ("P_gross_mismatch", "Mismatch gross response", "S_structural_vs_P_mismatch"),
            ("G_c", "Mismatch-adjusted response", "S_structural_vs_G_structural"),
        )
        for row, arm in enumerate(system_arms):
            candidate = candidates[_candidate_id(arm, system)]
            for column, (endpoint, title, association) in enumerate(endpoint_columns):
                axis = axes[row, column]
                for seed_result in candidate["per_seed"]:
                    axis.scatter(
                        np.asarray(
                            seed_result["scores"]["structural_normalized"]
                        ).reshape(-1),
                        np.asarray(
                            seed_result["causal"]["structural"][endpoint]
                        ).reshape(-1),
                        s=17,
                        alpha=0.70,
                        color=seed_colours[int(seed_result["seed"])],
                    )
                rho = candidate["population_associations"][association][
                    "fisher_mean_rho"
                ]
                axis.set_title(f"{title}\nseed-level Fisher mean ρ={rho:.2f}")
                axis.axhline(0.0, color="0.6", linewidth=0.8)
                if row == len(system_arms) - 1:
                    axis.set_xlabel("Normalized structural score")
                if column == 0:
                    axis.set_ylabel(
                        arm.replace("_", " ") + "\ncausal endpoint"
                    )
        figure.suptitle(
            f"Four-way structural calibration | {system} score | {summary['split']}",
            y=1.0,
        )
        figure.tight_layout()
        key = f"four_way_{system}"
        saved[key] = _save_figure(
            figure,
            output_dir,
            f"structural_four_way_calibration_{system}",
            {
                "split": summary["split"],
                "score_system": system,
                "inference": "Four per-seed estimates; annotations are Fisher means of seed estimates.",
            },
        )
        plt.close(figure)

        figure, axes = plt.subplots(
            len(system_arms),
            2,
            figsize=(9.5, 3.4 * len(system_arms)),
            squeeze=False,
        )
        for row, arm in enumerate(system_arms):
            candidate = candidates[_candidate_id(arm, system)]
            for seed_result in candidate["per_seed"]:
                seed = int(seed_result["seed"])
                active = np.asarray(seed_result["scores"]["active"]).reshape(-1)
                axes[row, 0].scatter(
                    np.asarray(seed_result["scores"]["joint_sensitivity"]).reshape(-1),
                    np.asarray(seed_result["causal"]["gross_total_for_J"]).reshape(-1),
                    s=17,
                    alpha=0.70,
                    color=seed_colours[seed],
                )
                axes[row, 1].scatter(
                    np.asarray(seed_result["scores"]["selectivity"]).reshape(-1)[active],
                    np.asarray(seed_result["causal"]["gross_contrast_for_D_rel"]).reshape(-1)[active],
                    s=17,
                    alpha=0.70,
                    color=seed_colours[seed],
                )
            j_rho = candidate["population_associations"]["J_vs_gross_total"][
                "fisher_mean_rho"
            ]
            d_rho = candidate["population_associations"][
                "D_rel_vs_gross_contrast"
            ]["fisher_mean_rho"]
            axes[row, 0].set_title(f"J validation | seed-level ρ={j_rho:.2f}")
            axes[row, 1].set_title(f"D_rel validation | seed-level ρ={d_rho:.2f}")
            axes[row, 0].set_ylabel(arm.replace("_", " ") + "\ncausal total")
            axes[row, 1].axhline(0.0, color="0.6", linewidth=0.8)
            axes[row, 1].axvline(0.0, color="0.6", linewidth=0.8)
            if row == len(system_arms) - 1:
                axes[row, 0].set_xlabel("Joint sensitivity J")
                axes[row, 1].set_xlabel("Selectivity D_rel (active heads)")
        figure.suptitle(
            f"Causal coordinate validation | {system} score | {summary['split']}",
            y=1.0,
        )
        figure.tight_layout()
        saved[f"coordinates_{system}"] = _save_figure(
            figure,
            output_dir,
            f"causal_J_Drel_validation_{system}",
            {
                "split": summary["split"],
                "score_system": system,
                "D_rel_reliability_floor": "J >= 0.20",
            },
        )
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.5))
    selection = sorted(summary["selection_table"], key=lambda row: row["registered_rank"])
    labels = [
        row["candidate_id"].replace("complete_pe", "full").replace("__", "\n")
        for row in selection
    ]
    positions = np.arange(len(selection))
    axes[0, 0].bar(
        positions,
        [row["structural_score_adjusted_rho"] for row in selection],
        color="tab:blue",
    )
    axes[0, 0].set_title("Structural score ↔ adjusted causal response")
    axes[0, 1].bar(
        positions,
        [row["D_rel_causal_contrast_rho"] for row in selection],
        color="tab:orange",
    )
    axes[0, 1].set_title("D_rel ↔ causal channel contrast")
    axes[1, 0].bar(
        positions,
        [row["median_structural_taylor_cosine_across_seeds"] for row in selection],
        color="tab:green",
    )
    axes[1, 0].set_title("Median structural Taylor cosine")
    axes[1, 1].bar(
        positions,
        [row["median_structural_carrier_coherence_across_seeds"] for row in selection],
        color="tab:purple",
    )
    axes[1, 1].set_title("Carrier coherence (1 − cancellation)")
    for axis in axes.reshape(-1):
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.set_xticks(positions, labels, rotation=45, ha="right", fontsize=7)
    figure.suptitle("Registered refinement decision diagnostics", y=1.0)
    figure.tight_layout()
    saved["decision"] = _save_figure(
        figure,
        output_dir,
        "registered_decision_diagnostics",
        {
            "split": summary["split"],
            "selection_table": summary["selection_table"],
            "note": "Rank is advisory; confirmation remains inaccessible until an explicit lock is written.",
        },
    )
    plt.close(figure)

    figure, axes = plt.subplots(
        1,
        len(plotted_systems),
        figsize=(5.2 * len(plotted_systems), 4.2),
        squeeze=False,
    )
    stratum_colours = {
        "near": "tab:blue",
        "middle": "tab:orange",
        "far": "tab:green",
    }
    for column, system in enumerate(plotted_systems):
        axis = axes[0, column]
        x = np.arange(len(plotted_arms), dtype=float)
        for offset, label in zip((-0.18, 0.0, 0.18), ("near", "middle", "far")):
            values = []
            for arm in plotted_arms:
                candidate = candidates.get(_candidate_id(arm, system))
                key = (
                    "S_structural_vs_G_structural__input_dose__"
                    f"{label}"
                )
                values.append(
                    candidate["population_associations"][key][
                        "fisher_mean_rho"
                    ]
                    if candidate is not None
                    and key in candidate["population_associations"]
                    else np.nan
                )
            axis.scatter(
                x + offset,
                values,
                s=42,
                color=stratum_colours[label],
                label=label,
            )
        axis.axhline(0.0, color="0.5", linewidth=0.8)
        axis.set_xticks(
            x,
            [arm.replace("complete_pe", "full").replace("_", " ") for arm in plotted_arms],
            rotation=35,
            ha="right",
        )
        axis.set_ylabel("Seed-level Fisher mean ρ")
        axis.set_title(f"{system}: S_str ↔ G_str within input-dose tertile")
        axis.legend(frameon=False)
    figure.tight_layout()
    saved["dose_strata"] = _save_figure(
        figure,
        output_dir,
        "structural_calibration_by_input_dose",
        {
            "split": summary["split"],
            "strata": ["near", "middle", "far"],
            "note": "Dose tertiles are determined from input-space interventions only.",
        },
    )
    plt.close(figure)
    return saved


def finalize_pe_refinement(
    config: PERefinementConfig,
    *,
    split: str,
) -> dict[str, Any]:
    config.validate()
    if split not in CAUSAL_SPLITS:
        raise ValueError(split)
    task_root = config.root / TASK_NAME
    if split == "confirmation":
        lock_path = task_root / "selection_lock.json"
        if not lock_path.exists():
            raise RuntimeError(
                "confirmation is locked; run the refinement finalizer, inspect it, then "
                "write an explicit method lock"
            )
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        refinement = task_root / "analysis" / "refinement" / "summary.json"
        expected_lock = {
            "version": PE_REFINEMENT_VERSION,
            "scientific_fingerprint": config.fingerprint,
            "refinement_summary": str(refinement),
            "refinement_summary_sha256": (
                checkpoint_sha256(refinement) if refinement.exists() else None
            ),
        }
        for key, value in expected_lock.items():
            if lock.get(key) != value:
                raise StaleCacheError(
                    f"selection lock field {key!r} is stale: "
                    f"{lock.get(key)!r} != {value!r}"
                )
        selected = (
            (str(lock["selected_arm"]), str(lock["selected_score_system"])),
        )
        if lock.get("candidate_id") != _candidate_id(*selected[0]):
            raise StaleCacheError("selection lock candidate_id is internally inconsistent")
    else:
        selected = tuple(
            (arm, system)
            for arm in STRUCTURAL_ARMS
            for system in SCORE_SYSTEMS
        )
    selected_arms = tuple(dict.fromkeys(arm for arm, _ in selected))
    paired_manifest_hashes = _validate_paired_manifests(
        config,
        split=split,
        arms=selected_arms,
    )
    candidates = []
    for arm, score_system in selected:
        candidate, _ = _candidate_public_summary(
            config,
            arm=arm,
            score_system=score_system,
            split=split,
        )
        candidates.append(candidate)
    selection_table = [_selection_row(candidate) for candidate in candidates]
    _rank_selection_rows(selection_table)
    audits_by_seed = {}
    model_audits_by_seed = {}
    for seed in config.seeds:
        common = {
            name: _read_protected(
                config,
                seed=seed,
                namespace="common",
                stage="audits",
                name=cache_name,
            )
            for name, cache_name in (
                ("common-scores", "common-scores"),
                (f"common-causal-{split}", f"common-causal-{split}"),
                ("common-ablation", "common-ablation"),
            )
        }
        arms = {
            f"{arm}:{name}": _read_protected(
                config,
                seed=seed,
                namespace=f"arms/{arm}",
                stage="audits",
                name=cache_name,
            )
            for arm in selected_arms
            for name, cache_name in (
                ("scores", "scores"),
                (f"causal-{split}", f"causal-{split}"),
            )
        }
        audits_by_seed[str(seed)] = {"common": common, "arms": arms}
        model_audits_by_seed[str(seed)] = _read_protected(
            config,
            seed=seed,
            namespace="common",
            stage="audits",
            name="model",
        )
    audit_finding_count = int(
        sum(
            len(findings)
            for seed in audits_by_seed.values()
            for group in seed.values()
            for findings in group.values()
        )
    )
    summary = {
        "version": PE_REFINEMENT_VERSION,
        "scientific_fingerprint": config.fingerprint,
        "repository_commit": _repository_commit(),
        "task": TASK_NAME,
        "split": split,
        "seeds": list(config.seeds),
        "seed_count": len(config.seeds),
        "candidates": candidates,
        "selection_table": selection_table,
        "paired_manifest_hashes": paired_manifest_hashes,
        "audits_by_seed": audits_by_seed,
        "model_audits_by_seed": model_audits_by_seed,
        "audit_finding_count": audit_finding_count,
        "headline_eligible": audit_finding_count == 0,
        "statistical_boundary": (
            "per-seed head associations with within-layer permutations; population summaries "
            "combine four seed estimates and never pool 192 heads as independent observations"
        ),
        "confirmation_lock_required": split == "refinement",
    }
    output_dir = task_root / "analysis" / split
    atomic_json(output_dir / "summary.json", summary)
    figures = render_refinement_figures(summary, output_dir / "figures")
    summary["figures"] = figures
    atomic_json(output_dir / "summary.json", summary)
    atomic_json(
        task_root / "protocol.json",
        {
            **config.scientific_record,
            "scientific_fingerprint": config.fingerprint,
            "repository_commit": _repository_commit(),
        },
    )
    return summary


def lock_pe_refinement_selection(
    config: PERefinementConfig,
    *,
    arm: str,
    score_system: str,
) -> Path:
    config.validate()
    if arm not in STRUCTURAL_ARMS or score_system not in SCORE_SYSTEMS:
        raise ValueError("selection must name one registered arm and score system")
    task_root = config.root / TASK_NAME
    refinement = task_root / "analysis" / "refinement" / "summary.json"
    if not refinement.exists():
        raise RuntimeError("refinement summary does not exist; it must be finalized first")
    summary = json.loads(refinement.read_text(encoding="utf-8"))
    expected_summary = {
        "version": PE_REFINEMENT_VERSION,
        "scientific_fingerprint": config.fingerprint,
        "task": TASK_NAME,
        "split": "refinement",
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise StaleCacheError(
                f"refinement summary field {key!r} is stale: "
                f"{summary.get(key)!r} != {value!r}"
            )
    candidate_id = _candidate_id(arm, score_system)
    if candidate_id not in {
        row["candidate_id"] for row in summary["selection_table"]
    }:
        raise RuntimeError(f"candidate {candidate_id!r} is absent from refinement summary")
    digest = checkpoint_sha256(refinement)
    path = task_root / "selection_lock.json"
    payload = {
        "version": PE_REFINEMENT_VERSION,
        "scientific_fingerprint": config.fingerprint,
        "selected_arm": arm,
        "selected_score_system": score_system,
        "candidate_id": candidate_id,
        "refinement_summary": str(refinement),
        "refinement_summary_sha256": digest,
        "repository_commit": _repository_commit(),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _cache_scientific_contract(existing) != _cache_scientific_contract(
            payload
        ):
            raise RuntimeError(
                f"selection lock already exists with another choice: {path}"
            )
        return path
    atomic_json(path, payload)
    return path
