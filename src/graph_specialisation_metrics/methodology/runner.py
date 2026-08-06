"""End-to-end canonical runner shared by every registered model backend."""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import platform
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..carriage import env
from ..carriage.env import log
from ..carriage.tasks import resolve_dataset_dir
from ..specialisation.model import GritHeadModel, SpecConfig
from .audit import audit_check, audit_scope, log_summary, set_strict, within_tolerance
from .backend import CanonicalGritBackend
from .bootstrap import (
    Observation,
    nested_percentile_interval,
    paired_channel_percentile_interval,
    reportable_bin,
    trimmed_mean,
)
from .cache import (
    CacheContract,
    CanonicalCache,
    atomic_json,
    checkpoint_sha256,
    load_cache_artifact_file,
    load_cache_value_file,
)
from .carriage import (
    additive_beneficial_mass,
    beneficial_carriage,
    event_normalise_functional,
    functional_carriage_events,
)
from .distance import (
    DisplayAxis,
    DistanceAxis,
    adaptive_distance_bins,
    aggregate_distance_events,
    column_support,
    display_bins,
    distance_event_contributions,
    distance_profile_reduce,
    row_normalised,
    score_heatmaps,
    shortest_path_distances,
    supported_mean,
)
from .events import build_channel_events
from .execution import execute_graph_batches
from .interventions import semantic_donor_swap, structural_donor_swap
from .figures import (
    FigureBuilder,
    FigureTheme,
    HeadPlotData,
    TASK_FIGURE_MODIFIERS,
    attention_distance_profiles,
    causal_family_panels,
    causal_regime_summary,
    causal_scatter_grid,
    cumulative_prefix_curves,
    carriage_profiles,
    distance_heatmaps,
    distance_support_profile,
    joint_selectivity_plane,
    multi_seed_score_causal_triptych,
    score_distance_profiles,
    score_plane,
    selectivity_regime_diagnostics,
    specialist_causal_panels,
    strong_specialist_map,
)
from .protocol import (
    CHANNELS,
    PROTOCOL_VERSION,
    MethodologyConfig,
    SplitManifest,
    deterministic_splits,
    stable_hash,
)
from .progress import ProgressJournal
from .sampling import SemanticDonorPool, manifest_fingerprint, sample_sources
from .scores import (
    aggregate_event_scores,
    event_head_scores,
    freeze_families,
    freeze_matched_controls,
    freeze_threshold_specialists,
    head_coordinates,
    project_transport,
    specialisation_diagnostics,
)
from .tasks import CanonicalTask, get_task, training_target_matrix


@dataclass
class PreparedTask:
    task: CanonicalTask
    runtime: Any
    backend: Any
    output_dir: Path
    checkpoint: Path
    checkpoint_sha: str
    sigma: np.ndarray
    splits: SplitManifest
    donor_pool: SemanticDonorPool
    progress: ProgressJournal | None = None

    @property
    def grit(self) -> Any:
        """Compatibility alias while historical GRIT callers migrate to ``runtime``."""

        return self.runtime


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


def _model_audits(
    runtime: Any,
    backend: Any,
    task: CanonicalTask,
    config: MethodologyConfig,
) -> dict[str, Any]:
    """Model, gradient, attention, and intervention no-op checks.

    Every check reports through the soft-audit ledger: a breach is measured, logged, and recorded
    in ``model.json`` rather than aborting the run (see ``strict_audits`` for fail-closed runs).
    """

    import torch

    base = runtime.eval_ds[0]
    single = backend.capture([base], require_grad=False, include_virtual_transport=True)
    repeated = backend.capture(
        [base, base], require_grad=False, include_virtual_transport=True
    )
    batch_error = float(
        torch.max(
            torch.abs(
                repeated.z
                - single.z.expand_as(repeated.z)
            )
        ).item()
    )
    within_tolerance(
        batch_error,
        config.numerical.batch_invariance_tolerance,
        "model.batch_invariance",
        "batch invariance error",
        context={"task": task.name, "train_seed": int(runtime.sc.seed)},
    )

    # Attention rows are receiver-normalized on every supported layer.
    attention_error = float(backend.attention_normalization_error(base))
    within_tolerance(
        attention_error,
        config.numerical.attention_tolerance,
        "model.attention_normalization",
        "attention normalization error",
        context={"task": task.name, "train_seed": int(runtime.sc.seed)},
    )

    declared_noops = getattr(backend, "declared_noop_variants", None)
    if callable(declared_noops):
        semantic_noop, structural_noop = declared_noops(base)
    else:
        rows = task.content_adapter.rows(base)
        semantic_noop = semantic_donor_swap(
            base, 0, rows[0], adapter=task.content_adapter
        )
        structural_noop = structural_donor_swap(
            base,
            0,
            0,
            task=task,
            duplicate_tolerance=config.numerical.duplicate_tolerance,
        )
    noops = backend.capture(
        [base, semantic_noop, structural_noop],
        require_grad=False,
        include_virtual_transport=True,
    )
    no_op_error = float(
        torch.max(torch.abs(noops.z[1:] - noops.z[0:1])).item()
    )
    for layer in noops.transport:
        no_op_error = max(
            no_op_error,
            float(torch.max(torch.abs(layer[1:] - layer[0:1])).item()),
        )
    within_tolerance(
        no_op_error,
        config.numerical.no_op_tolerance,
        "model.declared_no_op",
        "declared no-op donor response",
        context={"task": task.name, "train_seed": int(runtime.sc.seed)},
    )

    # This also verifies every layer has a finite, nonzero z-space Jacobian.
    clean = backend.clean_jacobians(base)
    return {
        "batch_invariance_max_error": batch_error,
        "attention_normalization_max_error": attention_error,
        "no_op_max_error": no_op_error,
        "jacobian_engine": getattr(backend, "jacobian_engine", "sequential_vjp"),
        "clean_transport_gradient_norm": float(
            torch.linalg.vector_norm(clean.transport).item()
        ),
        "clean_final_gradient_norm": float(
            torch.linalg.vector_norm(clean.final_state).item()
        ),
    }


def _rng(config: MethodologyConfig, *parts: Any) -> np.random.Generator:
    digest = stable_hash({"seed": int(config.analysis_seed), "parts": parts}, length=16)
    return np.random.default_rng(int(digest, 16))


def _event_rng(
    prepared: PreparedTask,
    config: MethodologyConfig,
    *parts: Any,
) -> np.random.Generator:
    """Return a replay RNG, pairing GraphBench event manifests across model seeds."""

    model_seed = (
        ()
        if getattr(prepared.task, "backend_kind", None) == "graphbench_grit"
        else (int(prepared.grit.sc.seed),)
    )
    return _rng(config, prepared.task.name, *model_seed, *parts)


def _channel_bootstrap_policy(
    prepared: PreparedTask,
    config: MethodologyConfig,
    plan: Mapping[int, Mapping[str, Any]],
    channel: str,
):
    """Disable source resampling only when every eligible source was enumerated."""

    eligible_source_fn = getattr(prepared.backend, "eligible_sources", None)
    if not callable(eligible_source_fn) or prepared.task.paired_channel_sources:
        return config.bootstrap
    exhaustive = all(
        len(plan[int(graph_id)][channel]["sources"])
        == len(
            eligible_source_fn(
                prepared.grit.eval_ds[int(graph_id)], channel=channel
            )
        )
        for graph_id in plan
    )
    return (
        dataclasses.replace(config.bootstrap, resample_source=False)
        if exhaustive
        else config.bootstrap
    )


def _checkpoint_override(config: MethodologyConfig, task: str, seed: int) -> str | None:
    for key in (f"{task}:{seed}", f"{task}_seed{seed}", task):
        if key in config.checkpoints:
            return str(config.checkpoints[key])
    return None


def _prepare_graphormer_task(
    config: MethodologyConfig,
    task: CanonicalTask,
    train_seed: int,
    task_overrides: Mapping[str, Any],
) -> PreparedTask:
    from .graphormer import GraphormerBackend, build_graphormer_runtime

    output_dir = config.root / task.name / f"seed_{int(train_seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime, checkpoint_descriptor, digest = build_graphormer_runtime(
        task,
        checkpoint=_checkpoint_override(config, task.name, int(train_seed)),
        train_seed=int(train_seed),
        accelerator=config.accelerator,
        overrides=task_overrides,
    )
    outputs = int(runtime.model.config.num_classes)
    sigma = task.output.resolve(outputs)
    backend = GraphormerBackend(runtime, task, sigma)
    with audit_scope(f"{task.name}:seed{int(train_seed)}:model") as model_scope:
        audit_checks = _model_audits(runtime, backend, task, config)
    audit_checks["failures"] = log_summary(
        model_scope, header=f"{task.name}:seed{int(train_seed)} model audits"
    )
    splits = deterministic_splits(
        len(runtime.eval_ds),
        len(runtime.donor_ds),
        config.sizes,
        int(config.analysis_seed),
        same_index_space=False,
    )
    donor_pool = SemanticDonorPool(
        [
            (graph_id, runtime.donor_ds[graph_id])
            for graph_id in splits.semantic_donor_pool
        ],
        adapter=task.content_adapter,
    )
    model_record = {
        "protocol_version": PROTOCOL_VERSION,
        "repository_commit": _repository_commit(),
        "task": task.name,
        "backend": task.backend_kind,
        "title": task.title,
        "train_seed": int(train_seed),
        "checkpoint": checkpoint_descriptor,
        "checkpoint_epoch": None,
        "checkpoint_sha256": digest,
        "output_representation": task.output.representation,
        "sigma": sigma.tolist(),
        "task_adapter_version": task.adapter_version,
        "carrier_policy": task.carrier_policy,
        "splits": dataclasses.asdict(splits),
        "test_metric": runtime.test_metric,
        "validation_metric": runtime.val_metric,
        "parameter_count": runtime.checks.get("num_parameters"),
        "canonical_audits": audit_checks,
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
    }
    atomic_json(output_dir / "model.json", model_record)
    checkpoint_label = Path(
        checkpoint_descriptor.replace("/", "__").replace("@", "__at__")
    )
    return PreparedTask(
        task,
        runtime,
        backend,
        output_dir,
        checkpoint_label,
        digest,
        sigma,
        splits,
        donor_pool,
        ProgressJournal(
            output_dir / "progress.jsonl",
            heartbeat_seconds=config.execution.progress_heartbeat_seconds,
        ),
    )


def _prepare_graphbench_task(
    config: MethodologyConfig,
    task: CanonicalTask,
    train_seed: int,
    task_overrides: Mapping[str, Any],
) -> PreparedTask:
    from .graphbench import (
        GraphBenchEdgeDonorPool,
        GraphBenchGritBackend,
        build_graphbench_runtime,
    )

    output_dir = config.root / task.name / f"seed_{int(train_seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_value = _checkpoint_override(config, task.name, int(train_seed))
    if checkpoint_value is None:
        training_root = Path(
            str(
                task_overrides.get(
                    "training_output_root",
                    "/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/"
                    "graphbench_algoreas_hpc_base_v1",
                )
            )
        )
        checkpoint_value = str(
            training_root
            / task.spec.graphbench_task
            / "grit"
            / f"seed{int(train_seed)}"
            / "best.pt"
        )
    runtime, checkpoint_descriptor = build_graphbench_runtime(
        task,
        checkpoint_path=checkpoint_value,
        train_seed=int(train_seed),
        accelerator=config.accelerator,
        overrides=task_overrides,
        jacobian_output_chunk=int(config.execution.jacobian_output_chunk),
    )
    checkpoint = Path(checkpoint_descriptor)
    digest = checkpoint_sha256(checkpoint)
    if task.spec.task_type == "graph_regression":
        if runtime.target_stats is None:
            raise RuntimeError("GraphBench flow checkpoint has no training target scale")
        sigma = np.asarray(
            [max(float(runtime.target_stats["std"]), 1.0e-6)], dtype=np.float64
        )
    else:
        sigma = np.asarray([1.0], dtype=np.float64)
    backend = GraphBenchGritBackend(
        runtime,
        task,
        sigma,
        jacobian_output_chunk=int(config.execution.jacobian_output_chunk),
    )
    with audit_scope(f"{task.name}:seed{int(train_seed)}:model") as model_scope:
        audit_checks = _model_audits(runtime, backend, task, config)
    audit_checks["failures"] = log_summary(
        model_scope, header=f"{task.name}:seed{int(train_seed)} model audits"
    )
    splits = deterministic_splits(
        len(runtime.eval_ds),
        len(runtime.donor_ds),
        config.sizes,
        int(config.analysis_seed),
        same_index_space=False,
    )
    donor_pool = GraphBenchEdgeDonorPool(
        [
            (graph_id, runtime.donor_ds[graph_id])
            for graph_id in splits.semantic_donor_pool
        ]
    )
    model_record = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_extension": task.protocol_extension,
        "repository_commit": _repository_commit(),
        "task": task.name,
        "graphbench_task": task.spec.graphbench_task,
        "backend": task.backend_kind,
        "title": task.title,
        "train_seed": int(train_seed),
        "checkpoint": checkpoint_descriptor,
        "checkpoint_epoch": runtime.checkpoint.get("epoch"),
        "checkpoint_step": runtime.checkpoint.get("step"),
        "checkpoint_sha256": digest,
        "output_representation": task.output.representation,
        "sigma": sigma.tolist(),
        "task_adapter_version": task.adapter_version,
        "raw_score_system": task.raw_score_system,
        "structural_intervention": "complete_pe_copy",
        "structural_degree_matching": False,
        "semantic_source_kind": task.semantic_source_kind,
        "paired_channel_sources": task.paired_channel_sources,
        "carrier_policy": task.carrier_policy,
        "model_geometry": backend.geometry,
        "splits": dataclasses.asdict(splits),
        "validation_metric": runtime.val_metric,
        "test_metric": runtime.test_metric,
        "parameter_count": runtime.checks.get("num_parameters"),
        "fidelity": runtime.checks,
        "canonical_audits": audit_checks,
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
    }
    atomic_json(output_dir / "model.json", model_record)
    return PreparedTask(
        task,
        runtime,
        backend,
        output_dir,
        checkpoint,
        digest,
        sigma,
        splits,
        donor_pool,
        ProgressJournal(
            output_dir / "progress.jsonl",
            heartbeat_seconds=config.execution.progress_heartbeat_seconds,
        ),
    )


def prepare_task(
    config: MethodologyConfig,
    task_name: str,
    train_seed: int,
    *,
    force_fresh_grit: bool = False,
) -> PreparedTask:
    """Rebuild the registered environment and load a cached training checkpoint read-only."""

    set_strict(bool(config.strict_audits))
    task_overrides = config.task_overrides.get(task_name, {})
    runtime_override_names = {
        "grit_repo_dir",
        "config_file",
        "drive_dir",
        "dataset_dir",
        "eval_split",
        "donor_split",
        "sigma",
        "output_representation",
        "sigma_policy",
        "dataset_root",
        "model_id",
        "revision",
        "cache_dir",
        "local_files_only",
        "runner_path",
        "training_output_root",
        "pe_cache_root",
        "pe_cache_namespace",
        "pe_cache_dtype",
        "pe_workers",
        "pe_save_every",
        "require_subset_cache",
        "require_pe_cache",
        "build_missing_pe_cache",
        "force_reload_data",
        "metric_reproduction_tolerance",
        "analysis_split_limits",
        "eval_metric",
        "expected_grit_commit",
        "split_seed",
        "train_size",
        "val_size",
        "test_size",
        "train_node_size",
        "val_node_size",
        "test_node_size",
    }
    canonical_override_names = set(CanonicalTask.__dataclass_fields__) - {
        "name",
        "backend_kind",
        "spec",
    }
    unknown_overrides = sorted(
        set(task_overrides) - runtime_override_names - canonical_override_names
    )
    if unknown_overrides:
        raise ValueError(
            f"unknown task overrides for {task_name!r}: {unknown_overrides}"
        )
    scientific_overrides = {
        key: value
        for key, value in task_overrides.items()
        if key in canonical_override_names
    }
    task = get_task(task_name, scientific_overrides)
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
    if task.backend_kind == "graphormer":
        return _prepare_graphormer_task(
            config,
            task,
            int(train_seed),
            task_overrides,
        )
    if task.backend_kind == "graphbench_grit":
        return _prepare_graphbench_task(
            config,
            task,
            int(train_seed),
            task_overrides,
        )
    if task.backend_kind == "nar_grit":
        from ..synthetic.nar_canonical_analysis import prepare_nar_task

        return prepare_nar_task(
            config,
            task,
            int(train_seed),
            task_overrides,
            prepared_task_class=PreparedTask,
            repository_commit=_repository_commit(),
        )
    if task.backend_kind != "grit":
        raise ValueError(f"unsupported canonical backend {task.backend_kind!r}")

    import torch
    from torch_geometric.data import Batch

    spec = task.spec
    output_dir = config.root / task_name / f"seed_{int(train_seed)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = Path(
        str(
            task_overrides.get(
                "grit_repo_dir",
                spec.grit_repo_dir
                or (f"/content/GRIT_{spec.name}" if spec.env_hooks else "/content/GRIT"),
            )
        )
    )
    env.clone_grit(
        repo_dir,
        spec.grit_repo,
        spec.grit_commit,
        force_fresh=force_fresh_grit,
    )
    for hook in spec.env_hooks:
        hook(repo_dir)
    env.prepare_inprocess_grit(repo_dir)
    config_file = str(
        task_overrides.get("config_file") or env.resolve_config(spec, repo_dir, output_dir)
    )
    drive_dir = str(task_overrides.get("drive_dir", spec.drive_dir))
    dataset_dir = str(
        task_overrides.get("dataset_dir") or resolve_dataset_dir(spec, drive_dir)
    )
    checkpoint, epoch = env.find_checkpoint(
        Path(drive_dir) / "results",
        _checkpoint_override(config, task_name, int(train_seed)),
    )
    digest = checkpoint_sha256(checkpoint)
    model_config = SpecConfig(
        ckpt=str(checkpoint),
        out_dir=str(output_dir),
        dataset_dir=dataset_dir,
        config_file=config_file,
        accelerator=config.accelerator,
        seed=int(train_seed),
        num_threads=int(config.num_threads),
        eval_split=str(task_overrides.get("eval_split", "test")),
        donor_split=str(task_overrides.get("donor_split", "train")),
        eval_metric=bool(task_overrides.get("eval_metric", True)),
        analysis_seed=int(config.analysis_seed),
        donors=int(config.sizes.donors_per_source),
        content_adapter=spec.content_adapter,
        resume=bool(config.resume),
    )
    subset_limits = task_overrides.get("analysis_split_limits")
    subset_environment = "GSM_PEPTIDES_ANALYSIS_SPLIT_LIMITS"
    previous_subset = os.environ.get(subset_environment)
    try:
        if subset_limits is not None:
            os.environ[subset_environment] = json.dumps(
                dict(subset_limits),
                sort_keys=True,
            )
        else:
            os.environ.pop(subset_environment, None)
        grit = GritHeadModel(spec, model_config).load()
    finally:
        if previous_subset is None:
            os.environ.pop(subset_environment, None)
        else:
            os.environ[subset_environment] = previous_subset
    with torch.no_grad():
        first = Batch.from_data_list([grit.eval_ds[0].clone()]).to(grit.device)
        first_prediction, _ = grit.model(first)
    outputs = int(first_prediction.reshape(1, -1).shape[1])
    registered_training_std = getattr(
        grit.loaders[0].dataset,
        "_gsm_full_training_target_std",
        None,
    )
    if task.output.sigma_policy == "training_target_std" and registered_training_std is not None:
        registered_training_std = np.asarray(
            registered_training_std.detach().cpu(),
            dtype=np.float64,
        ).reshape(-1)
        sigma = dataclasses.replace(
            task.output,
            sigma=tuple(float(value) for value in registered_training_std),
        ).resolve(outputs)
    else:
        training_targets = (
            training_target_matrix(grit.loaders[0])
            if task.output.sigma_policy == "training_target_std"
            else None
        )
        sigma = task.output.resolve(outputs, training_targets=training_targets)
    backend = CanonicalGritBackend(grit, task, sigma)
    with audit_scope(f"{task_name}:seed{int(train_seed)}:model") as model_scope:
        audit_checks = _model_audits(grit, backend, task, config)
    audit_checks["failures"] = log_summary(
        model_scope, header=f"{task_name}:seed{int(train_seed)} model audits"
    )
    same_space = (
        grit.eval_ds is grit.donor_ds
        or model_config.eval_split == model_config.donor_split
    )
    splits = deterministic_splits(
        len(grit.eval_ds),
        len(grit.donor_ds),
        config.sizes,
        int(config.analysis_seed),
        same_index_space=same_space,
    )
    donor_pool = SemanticDonorPool(
        [(graph_id, grit.donor_ds[graph_id]) for graph_id in splits.semantic_donor_pool],
        adapter=spec.content_adapter,
    )
    model_record = {
        "protocol_version": PROTOCOL_VERSION,
        "repository_commit": _repository_commit(),
        "task": task_name,
        "backend": task.backend_kind,
        "title": task.title,
        "train_seed": int(train_seed),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(epoch),
        "checkpoint_sha256": digest,
        "output_representation": task.output.representation,
        "sigma": sigma.tolist(),
        "task_adapter_version": task.adapter_version,
        "carrier_policy": task.carrier_policy,
        "splits": dataclasses.asdict(splits),
        "test_metric": grit.test_metric,
        "validation_metric": grit.val_metric,
        "parameter_count": grit.checks.get("num_parameters"),
        "canonical_audits": audit_checks,
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
    }
    atomic_json(output_dir / "model.json", model_record)
    return PreparedTask(
        task,
        grit,
        backend,
        output_dir,
        Path(checkpoint),
        digest,
        sigma,
        splits,
        donor_pool,
        ProgressJournal(
            output_dir / "progress.jsonl",
            heartbeat_seconds=config.execution.progress_heartbeat_seconds,
        ),
    )


def _stage_ids(prepared: PreparedTask, stage: str) -> tuple[int, ...]:
    if stage in {"scores", "carriage"}:
        return prepared.splits.discovery
    if stage == "causal":
        return prepared.splits.causal
    if stage == "clean_ablation":
        return prepared.splits.clean_ablation
    raise ValueError(stage)


def _stage_plan(
    prepared: PreparedTask,
    config: MethodologyConfig,
    stage: str,
) -> dict[int, dict[str, Any]]:
    """Freeze source IDs and both independently drawn donor manifests before inference."""

    plan: dict[int, dict[str, Any]] = {}
    for graph_id in _stage_ids(prepared, stage):
        base = prepared.grit.eval_ds[int(graph_id)]
        eligible_source_fn = getattr(prepared.backend, "eligible_sources", None)
        entry: dict[str, Any] = {}
        shared_sources: np.ndarray | None = None
        if prepared.task.paired_channel_sources:
            source_rng = _event_rng(
                prepared,
                config,
                stage,
                graph_id,
                "sources",
            )
            if callable(eligible_source_fn):
                eligible = np.asarray(eligible_source_fn(base), dtype=np.int64)
                if eligible.ndim != 1 or not eligible.size:
                    raise RuntimeError(
                        f"{prepared.task.name} supplied no eligible analysis sources"
                    )
                count = min(
                    int(config.sizes.sources_per_graph), int(eligible.size)
                )
                shared_sources = np.sort(
                    source_rng.choice(eligible, size=count, replace=False)
                ).astype(np.int64)
            else:
                shared_sources = sample_sources(
                    int(base.num_nodes),
                    int(config.sizes.sources_per_graph),
                    source_rng,
                )
            entry["sources"] = tuple(int(v) for v in shared_sources)
        for channel in CHANNELS:
            if shared_sources is not None:
                sources = shared_sources
            else:
                if not callable(eligible_source_fn):
                    raise RuntimeError(
                        f"{prepared.task.name} uses channel-specific sources but its "
                        "backend does not declare eligible_sources(data, channel)"
                    )
                eligible = np.asarray(
                    eligible_source_fn(base, channel=channel), dtype=np.int64
                )
                if eligible.ndim != 1 or not eligible.size:
                    audit_check(
                        False,
                        "plan.no_eligible_channel_source",
                        f"graph {graph_id} has no eligible {channel} source",
                        context={
                            "stage": stage,
                            "graph": int(graph_id),
                            "channel": channel,
                        },
                    )
                    sources = np.empty(0, dtype=np.int64)
                else:
                    count = min(
                        int(config.sizes.sources_per_graph), int(eligible.size)
                    )
                    if count == int(eligible.size):
                        # Exhaustive source use has no source-sampling uncertainty.
                        sources = np.sort(eligible)
                    else:
                        sources = np.sort(
                            _event_rng(
                                prepared,
                                config,
                                stage,
                                graph_id,
                                channel,
                                "sources",
                            ).choice(eligible, size=count, replace=False)
                        ).astype(np.int64)
            records = []
            estimable_sources = []
            for source in sources:
                _, source_records = build_channel_events(
                    base,
                    graph_id=int(graph_id),
                    source=int(source),
                    channel=channel,
                    stage=stage,
                    donors=int(config.sizes.donors_per_source),
                    rng=_event_rng(
                        prepared,
                        config,
                        stage,
                        graph_id,
                        channel,
                        int(source),
                    ),
                    task=prepared.task,
                    semantic_pool=prepared.donor_pool,
                    duplicate_tolerance=config.numerical.duplicate_tolerance,
                )
                if source_records:
                    estimable_sources.append(int(source))
                    records.extend(source_records)
            entry[channel] = {
                "sources": tuple(estimable_sources),
                "records": tuple(record.record() for record in records),
            }
        if prepared.task.paired_channel_sources:
            common_sources = tuple(
                source
                for source in entry["sources"]
                if source in set(entry["semantic"]["sources"])
                and source in set(entry["structural"]["sources"])
            )
            if not common_sources:
                audit_check(
                    False,
                    "plan.no_estimable_source",
                    f"graph {graph_id} has no source estimable under both donor-swap "
                    f"channels; the graph is dropped from stage {stage!r}",
                    context={"stage": stage, "graph": int(graph_id)},
                )
                continue
            for channel in CHANNELS:
                entry[channel] = {
                    "sources": common_sources,
                    "records": tuple(
                        record
                        for record in entry[channel]["records"]
                        if int(record["source"]) in set(common_sources)
                    ),
                }
        elif any(not entry[channel]["sources"] for channel in CHANNELS):
            audit_check(
                False,
                "plan.no_estimable_channel_source",
                f"graph {graph_id} is not estimable in both independent source domains; "
                f"the graph is dropped from stage {stage!r}",
                context={"stage": stage, "graph": int(graph_id)},
            )
            continue
        plan[int(graph_id)] = entry
    if not plan:
        # Nothing is estimable anywhere, so no stage quantity exists to report on.
        raise RuntimeError(
            f"stage {stage!r} retained no graph with a source estimable under both channels"
        )
    return plan


def _manifest_hash(plan: Mapping[int, Mapping[str, Any]]) -> str:
    rows = []
    for graph_id in sorted(plan):
        for channel in CHANNELS:
            rows.extend(plan[graph_id][channel]["records"])
    return manifest_fingerprint(rows)


def _cache(
    prepared: PreparedTask,
    config: MethodologyConfig,
    plan: Mapping[int, Mapping[str, Any]],
    *,
    manifest_hash: str | None = None,
) -> CanonicalCache:
    return CanonicalCache(
        config.root,
        CacheContract(
            protocol_fingerprint=config.fingerprint,
            task=prepared.task.name,
            task_adapter_version=prepared.task.adapter_version,
            checkpoint_sha256=prepared.checkpoint_sha,
            train_seed=int(prepared.grit.sc.seed),
            model_geometry=prepared.backend.geometry,
            output_representation=prepared.task.output.representation,
            sigma=tuple(float(value) for value in prepared.sigma),
            split_fingerprint=prepared.splits.fingerprint,
            event_manifest_hash=(
                _manifest_hash(plan) if manifest_hash is None else str(manifest_hash)
            ),
            donors_per_source=int(config.sizes.donors_per_source),
            source_cap=int(config.sizes.sources_per_graph),
            bootstrap_seed=int(config.bootstrap.rng_seed),
            repository_commit=_repository_commit(),
            bootstrap_replicates=int(config.bootstrap.replicates),
            raw_score_aggregation=(
                f"{getattr(prepared.task, 'raw_score_system', 'mass')}"
                "->donor->source->graph"
            ),
            semantic_donor_law=(
                "graph-uniform/edge-uniform/min-endpoint-degree-signature-gap/"
                "iid-replacement"
                if prepared.task.semantic_source_kind == "edge"
                else "graph-uniform/node-uniform/min-gap/iid-replacement"
            ),
            structural_donor_law=(
                "same-side/node-type/nonidentical-rrwp/"
                "near-middle-far/no-degree-match/without-replacement/"
                "complete-pe-copy"
                if (
                    prepared.task.backend_kind == "graphbench_grit"
                    and prepared.task.spec.task_type == "edge_binary"
                )
                else (
                    "nonidentical-rrwp/near-middle-far/no-degree-match/"
                    "without-replacement/complete-pe-copy"
                    if prepared.task.backend_kind == "graphbench_grit"
                    else "node-uniform/min-gap/iid-replacement"
                )
            ),
        ),
        stale_policy="archive",
    )


def _rebuild_graph_events(
    prepared: PreparedTask,
    config: MethodologyConfig,
    stage: str,
    graph_id: int,
    channel: str,
    sources: Sequence[int],
) -> tuple[list[Any], list[Any]]:
    base = prepared.grit.eval_ds[int(graph_id)]
    variants: list[Any] = []
    records: list[Any] = []
    for source in sources:
        source_variants, source_records = build_channel_events(
            base,
            graph_id=int(graph_id),
            source=int(source),
            channel=channel,
            stage=stage,
            donors=int(config.sizes.donors_per_source),
            rng=_event_rng(
                prepared,
                config,
                stage,
                graph_id,
                channel,
                int(source),
            ),
            task=prepared.task,
            semantic_pool=prepared.donor_pool,
            duplicate_tolerance=config.numerical.duplicate_tolerance,
        )
        variants.extend(source_variants)
        records.extend(source_records)
    return variants, records


def _distance_axis(prepared: PreparedTask, graph_ids: Sequence[int]) -> DistanceAxis:
    matrices = [
        shortest_path_distances(
            prepared.grit.eval_ds[int(graph_id)].edge_index,
            int(prepared.grit.eval_ds[int(graph_id)].num_nodes),
        )
        for graph_id in graph_ids
    ]
    axis = DistanceAxis.from_matrices(matrices)
    labels = list(axis.labels)
    for label in prepared.backend.special_carrier_labels:
        if label not in labels:
            labels.append(label)
    return DistanceAxis(tuple(labels))


def _clean_attention_distance(
    prepared: PreparedTask,
    base: Any,
    pristine: np.ndarray,
    axis: DistanceAxis,
) -> np.ndarray:
    """Normalize clean attention by graph and head before family averaging."""

    return prepared.backend.clean_attention_distance(base, pristine, axis)


def _capture_event_groups(prepared: PreparedTask, groups: Sequence[Sequence[Any]]):
    capture_groups = getattr(prepared.backend, "capture_groups", None)
    if callable(capture_groups):
        return capture_groups(groups, include_virtual_transport=True)
    return [
        prepared.backend.capture(
            group, require_grad=False, include_virtual_transport=True
        )
        for group in groups
    ]


def _prepare_clean_jacobians(
    prepared: PreparedTask,
    config: MethodologyConfig,
    graph_ids: Sequence[int],
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Resume/persist clean linearisations and return graph-keyed execution provenance."""

    requested_ids = tuple(int(value) for value in graph_ids)
    clean_by_graph: dict[int, Any] = {}
    # Scores and carriage use the same discovery population but different event manifests. Bind
    # clean shards to the complete discovery IDs so either component (and a restarted worker) sees
    # the same protected cache contract even when only a subset of event shards is missing.
    clean_population = tuple(int(graph_id) for graph_id in prepared.splits.discovery)
    cache = _cache(
        prepared,
        config,
        {},
        manifest_hash=stable_hash(
            {
                "stage": "clean_jacobians",
                "graphs": clean_population,
            }
        ),
    )
    cache_hits = 0
    missing: list[int] = []

    def to_device(clean):
        device = getattr(prepared.backend, "device", None)
        if device is None:
            device = getattr(getattr(prepared.backend, "gm", None), "device", None)
        if device is None:
            return clean
        clean.capture.prediction = clean.capture.prediction.to(device)
        clean.capture.z = clean.capture.z.to(device)
        clean.capture.target = clean.capture.target.to(device)
        clean.capture.transport = tuple(
            value.to(device) for value in clean.capture.transport
        )
        clean.capture.final_state = clean.capture.final_state.to(device)
        if clean.capture.real_mask is not None:
            clean.capture.real_mask = clean.capture.real_mask.to(device)
        clean.transport = clean.transport.to(device)
        clean.final_state = clean.final_state.to(device)
        return clean

    for graph_id in requested_ids:
        cached = (
            cache.load(
                "clean_jacobians",
                f"graph_{graph_id:06d}",
                strict=True,
            )
            if config.resume and not config.force
            else None
        )
        if cached is None:
            missing.append(graph_id)
            continue
        clean = to_device(cached)
        clean_by_graph[graph_id] = clean
        remember = getattr(prepared.backend, "remember_clean_jacobians", None)
        if callable(remember):
            remember(prepared.grit.eval_ds[graph_id], clean)
        cache_hits += 1
    if cache_hits:
        log(
            f"[cache] loaded {cache_hits}/{len(requested_ids)} clean Jacobian "
            f"graph shards for {prepared.task.name}"
        )

    def execute(chunk):
        bases = [prepared.grit.eval_ds[int(graph_id)] for graph_id in chunk]
        clean_many = getattr(prepared.backend, "clean_jacobians_many", None)
        values = (
            clean_many(bases)
            if callable(clean_many)
            else [prepared.backend.clean_jacobians(base) for base in bases]
        )
        if len(values) != len(chunk):
            raise RuntimeError("grouped clean Jacobians changed the graph count")
        return list(zip((int(value) for value in chunk), values))

    def consume(rows):
        for graph_id, clean in rows:
            clean_by_graph[int(graph_id)] = clean
            cache.save(
                "clean_jacobians",
                f"graph_{int(graph_id):06d}",
                clean,
            )

    requested_batch = int(config.execution.graphs_per_batch)
    backend_limit = getattr(prepared.backend, "clean_jacobian_batch_size", None)
    clean_batch = (
        requested_batch
        if backend_limit is None
        else min(requested_batch, max(1, int(backend_limit)))
    )

    def on_batch(completed, total, batch_size):
        if prepared.progress is not None:
            prepared.progress.emit(
                "clean_jacobian_progress",
                completed=int(completed),
                total=int(total),
                batch_size=int(batch_size),
                cache_hits=int(cache_hits),
            )

    report = execute_graph_batches(
        missing,
        graphs_per_batch=clean_batch,
        execute=execute,
        consume=consume,
        oom_backoff=config.execution.oom_backoff,
        on_batch=on_batch,
    )
    execution = dataclasses.asdict(report)
    execution.update(
        {
            "cache_hits": int(cache_hits),
            "cache_misses": len(missing),
            "requested_graphs": len(requested_ids),
            "configured_graphs_per_batch": requested_batch,
        }
    )
    return clean_by_graph, execution


def _event_item_cost(
    prepared: PreparedTask,
    config: MethodologyConfig,
    plan: Mapping[int, Mapping[str, Any]],
    graph_id: int,
    channel: str,
) -> int:
    estimator = getattr(prepared.backend, "event_pair_cost", None)
    if not callable(estimator):
        return 1
    data = prepared.grit.eval_ds[int(graph_id)]
    return int(
        estimator(
            data,
            len(plan[int(graph_id)][channel]["sources"]),
            int(config.sizes.donors_per_source),
        )
    )


def _progress_batch_callback(
    prepared: PreparedTask,
    *,
    phase: str,
    channel: str,
    cached: int = 0,
):
    def report(done: int, total: int, size: int) -> None:
        if prepared.progress is not None:
            prepared.progress.emit(
                "batch_complete",
                phase=phase,
                channel=channel,
                completed_graphs=int(cached + done),
                total_graphs=int(cached + total),
                batch_graphs=int(size),
            )

    return report


def _source_population_size(prepared: PreparedTask, data: Any, channel: str) -> int:
    eligible = getattr(prepared.backend, "eligible_sources", None)
    if callable(eligible) and not prepared.task.paired_channel_sources:
        return int(len(eligible(data, channel=channel)))
    return int(data.num_nodes)


def _score_graph_batch(
    prepared: PreparedTask,
    config: MethodologyConfig,
    plan: Mapping[int, Mapping[str, Any]],
    clean_by_graph: Mapping[int, Any],
    axis: DistanceAxis,
    channel: str,
    graph_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Run one multi-graph event forward and return graph-local score sufficient statistics."""

    import torch

    contexts: list[dict[str, Any]] = []
    groups: list[list[Any]] = []
    for graph_id in graph_ids:
        graph_id = int(graph_id)
        base = prepared.grit.eval_ds[graph_id]
        sources = plan[graph_id][channel]["sources"]
        variants, records = _rebuild_graph_events(
            prepared, config, "scores", graph_id, channel, sources
        )
        if [record.record() for record in records] != list(
            plan[graph_id][channel]["records"]
        ):
            raise RuntimeError(
                "deterministic score-event replay changed its manifest for "
                f"graph={graph_id} channel={channel}; refusing misaligned results"
            )
        pristine = shortest_path_distances(base.edge_index, int(base.num_nodes))
        contexts.append(
            {
                "graph_id": graph_id,
                "base": base,
                "clean": clean_by_graph[graph_id],
                "records": records,
                "pristine": pristine,
            }
        )
        groups.append([base, *variants])

    captures = _capture_event_groups(prepared, groups)
    if len(captures) != len(contexts):
        raise RuntimeError("grouped event capture changed the number of base-graph groups")
    results: list[dict[str, Any]] = []
    for context, event_capture in zip(contexts, captures):
        graph_id = context["graph_id"]
        clean = context["clean"]
        records = context["records"]
        base = context["base"]
        pristine = context["pristine"]
        transport = torch.stack(event_capture.transport, dim=1)
        delta = transport[0:1] - transport[1:]
        q = project_transport(delta, clean.transport)
        score_system = getattr(prepared.task, "raw_score_system", "mass")
        scores = event_head_scores(q, system=score_system).detach().cpu().numpy()
        source_ids = [record.source for record in records]
        gids = [graph_id] * len(records)
        _, one_graph, _ = aggregate_event_scores(scores, gids, source_ids)
        event_c, event_o, rows = [], [], []
        score_observations, distance_rows = [], []
        for position, record in enumerate(records):
            distances = prepared.backend.transport_distances(
                base, int(record.source), pristine, channel=channel
            )
            contribution, support = distance_event_contributions(
                q[position : position + 1], distances, axis
            )
            event_c.append(contribution[0])
            event_o.append(support[0])
            rows.append(
                {
                    **record.record(),
                    "score": scores[position],
                    "score_system": score_system,
                    "distance_contribution": contribution[0],
                    "distance_support": support[0],
                }
            )
            score_observations.append(
                Observation(
                    seed=int(prepared.grit.sc.seed),
                    graph=graph_id,
                    source=int(record.source),
                    donor=int(record.draw),
                    value=scores[position],
                )
            )
            distance_rows.append(
                Observation(
                    seed=int(prepared.grit.sc.seed),
                    graph=graph_id,
                    source=int(record.source),
                    donor=int(record.draw),
                    value=np.stack(
                        (
                            contribution[0],
                            np.broadcast_to(
                                support[0][None, None, :], contribution[0].shape
                            ),
                        )
                    ),
                )
            )
        contribution_graph, support_graph = aggregate_distance_events(
            np.stack(event_c), np.stack(event_o), gids, source_ids
        )
        throughput = None
        attention = None
        if channel == "semantic":
            clean_transport = torch.stack(clean.capture.transport, dim=0)
            projected_clean = torch.einsum(
                "lnhd,tlnhd->lhnt", clean_transport, clean.transport
            )
            throughput = (
                projected_clean.square()
                .sum(dim=-1)
                .sqrt()
                .sum(dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
            attention = _clean_attention_distance(prepared, base, pristine, axis)
        results.append(
            {
                "graph_id": graph_id,
                "score": one_graph[graph_id],
                "contribution": contribution_graph[graph_id],
                "support": support_graph[graph_id],
                "event_rows": rows,
                "observations": score_observations,
                "distance_observations": distance_rows,
                "throughput": throughput,
                "attention": attention,
            }
        )
    return results


def estimate_graph_local_head_coordinates(
    prepared: PreparedTask,
    config: MethodologyConfig,
    *,
    plan: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Estimate canonical ``D_rel``/``J`` separately for requested graphs.

    This is an additive reporting diagnostic: it reuses the canonical event
    planner, clean Jacobians, donor-swap scorer, and graph/source aggregation,
    but it never writes or relabels the consolidated ``scores/raw.pt`` cache.
    """

    plan = dict(plan or _stage_plan(prepared, config, "scores"))
    graph_ids = sorted(int(graph_id) for graph_id in plan)
    if not graph_ids:
        raise ValueError("graph-local coordinate estimation needs at least one graph")

    axis = _distance_axis(prepared, graph_ids)
    clean_by_graph: dict[int, Any] = {}
    clean_many = getattr(prepared.backend, "clean_jacobians_many", None)
    configured_batch = int(config.execution.graphs_per_batch)
    backend_limit = getattr(prepared.backend, "clean_jacobian_batch_size", None)
    clean_batch = (
        configured_batch
        if backend_limit is None
        else min(configured_batch, max(1, int(backend_limit)))
    )

    def execute_clean(chunk):
        bases = [prepared.grit.eval_ds[int(graph_id)] for graph_id in chunk]
        values = (
            clean_many(bases)
            if callable(clean_many)
            else [prepared.backend.clean_jacobians(base) for base in bases]
        )
        if len(values) != len(chunk):
            raise RuntimeError("grouped clean Jacobians changed the graph count")
        return list(zip((int(value) for value in chunk), values))

    execute_graph_batches(
        graph_ids,
        graphs_per_batch=clean_batch,
        execute=execute_clean,
        consume=lambda rows: clean_by_graph.update(rows),
        oom_backoff=config.execution.oom_backoff,
    )

    channel_scores: dict[str, dict[int, np.ndarray]] = {
        channel: {} for channel in CHANNELS
    }
    for channel in CHANNELS:

        def consume_score_batch(results):
            for result in results:
                channel_scores[channel][int(result["graph_id"])] = np.asarray(
                    result["score"], dtype=np.float64
                )

        execute_graph_batches(
            graph_ids,
            graphs_per_batch=config.execution.graphs_per_batch,
            execute=lambda chunk, selected_channel=channel: _score_graph_batch(
                prepared,
                config,
                plan,
                clean_by_graph,
                axis,
                selected_channel,
                chunk,
            ),
            consume=consume_score_batch,
            oom_backoff=config.execution.oom_backoff,
            item_cost=lambda graph_id, selected_channel=channel: _event_item_cost(
                prepared,
                config,
                plan,
                int(graph_id),
                selected_channel,
            ),
            max_cost=config.execution.replica_pair_budget,
        )

    graphs: dict[int, dict[str, Any]] = {}
    for graph_id in graph_ids:
        coordinates = head_coordinates(
            channel_scores["semantic"][graph_id],
            channel_scores["structural"][graph_id],
            score_floor=config.numerical.score_floor,
            epsilon=config.numerical.selectivity_epsilon,
            activity_floor=config.families.activity_floor,
        )
        graphs[graph_id] = dataclasses.asdict(coordinates)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "estimator": "canonical-graph-local-head-coordinates-v1",
        "score_system": getattr(prepared.task, "raw_score_system", "mass"),
        "manifest_hash": _manifest_hash(plan),
        "graph_id_seed_space": "caller-supplied graph ID",
        "sources_per_graph": int(config.sizes.sources_per_graph),
        "donors_per_source": int(config.sizes.donors_per_source),
        "analysis_seed": int(config.analysis_seed),
        "graphs": graphs,
    }


def run_scores(
    prepared: PreparedTask,
    config: MethodologyConfig,
    *,
    plan: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute and cache raw S_sem/S_str plus exact distance sufficient statistics."""

    import torch

    plan = dict(plan or _stage_plan(prepared, config, "scores"))
    cache = _cache(prepared, config, plan)
    if config.resume and not config.force:
        cached = cache.load("scores", "raw", strict=True)
        if cached is not None:
            log(f"[cache] loaded canonical scores for {prepared.task.name}")
            if prepared.progress is not None:
                prepared.progress.emit(
                    "cache_hit", phase="scores", cache="consolidated"
                )
            return cached
    graph_ids = sorted(plan)
    axis = _distance_axis(prepared, graph_ids)
    output: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "score_system": getattr(prepared.task, "raw_score_system", "mass"),
        # Carrier-distance contributions remain an additive transport-mass
        # diagnostic. Coherent movement cannot be decomposed additively across
        # carriers without changing its estimand.
        "distance_score_system": "mass",
        "axis": axis.labels,
        "manifest_hash": _manifest_hash(plan),
        "channels": {},
    }
    observations: dict[str, list[Observation]] = {channel: [] for channel in CHANNELS}
    distance_observations: dict[str, list[Observation]] = {
        channel: [] for channel in CHANNELS
    }
    throughput_graph: dict[int, np.ndarray] = {}
    attention_graph: dict[int, np.ndarray] = {}
    cached_graphs: dict[str, dict[int, Any]] = {
        channel: {} for channel in CHANNELS
    }
    missing_graphs: dict[str, list[int]] = {
        channel: [] for channel in CHANNELS
    }
    for channel in CHANNELS:
        for graph_id in graph_ids:
            shard = (
                cache.load(
                    f"scores/{channel}",
                    f"graph_{int(graph_id):06d}",
                    strict=True,
                )
                if config.resume and not config.force
                else None
            )
            if shard is None:
                missing_graphs[channel].append(int(graph_id))
            else:
                cached_graphs[channel][int(graph_id)] = shard
    clean_needed = sorted(
        set(missing_graphs["semantic"]) | set(missing_graphs["structural"])
    )
    clean_by_graph, clean_execution = _prepare_clean_jacobians(
        prepared, config, clean_needed
    )
    execution_reports: dict[str, Any] = {}
    for channel in CHANNELS:
        graph_scores: dict[int, np.ndarray] = {}
        graph_contribution: dict[int, np.ndarray] = {}
        graph_support: dict[int, np.ndarray] = {}
        event_rows: list[dict[str, Any]] = []

        def consume_score_batch(results, *, persist: bool = True):
            for result in results:
                graph_id = int(result["graph_id"])
                graph_scores[graph_id] = result["score"]
                graph_contribution[graph_id] = result["contribution"]
                graph_support[graph_id] = result["support"]
                event_rows.extend(result["event_rows"])
                observations[channel].extend(result["observations"])
                distance_observations[channel].extend(
                    result["distance_observations"]
                )
                if result["throughput"] is not None:
                    throughput_graph[graph_id] = result["throughput"]
                if result["attention"] is not None:
                    attention_graph[graph_id] = result["attention"]
                if persist:
                    cache.save(
                        f"scores/{channel}",
                        f"graph_{graph_id:06d}",
                        result,
                    )

        consume_score_batch(
            [cached_graphs[channel][key] for key in sorted(cached_graphs[channel])],
            persist=False,
        )
        report = execute_graph_batches(
            missing_graphs[channel],
            graphs_per_batch=config.execution.graphs_per_batch,
            execute=lambda chunk: _score_graph_batch(
                prepared,
                config,
                plan,
                clean_by_graph,
                axis,
                channel,
                chunk,
            ),
            consume=consume_score_batch,
            oom_backoff=config.execution.oom_backoff,
            item_cost=lambda graph_id: _event_item_cost(
                prepared, config, plan, int(graph_id), channel
            ),
            max_cost=config.execution.replica_pair_budget,
            on_batch=_progress_batch_callback(
                prepared,
                phase="scores",
                channel=channel,
                cached=len(cached_graphs[channel]),
            ),
        )
        execution_reports[channel] = {
            **dataclasses.asdict(report),
            "cache_hits": len(cached_graphs[channel]),
            "cache_misses": len(missing_graphs[channel]),
        }
        raw = np.stack([graph_scores[key] for key in graph_ids]).mean(axis=0)
        heatmaps = score_heatmaps(
            graph_contribution,
            graph_support,
            reconstruction_tolerance=config.numerical.reconstruction_tolerance,
            graph_scores=(
                graph_scores
                if getattr(prepared.task, "raw_score_system", "mass") == "mass"
                else None
            ),
        )
        output["channels"][channel] = {
            "raw": raw,
            "graph_scores": graph_scores,
            "graph_distance_contribution": graph_contribution,
            "graph_distance_support": graph_support,
            "heatmap_exact": heatmaps.exact,
            "heatmap_per_opportunity": heatmaps.per_opportunity,
            "heatmap_exact_head": heatmaps.exact_head,
            "heatmap_per_opportunity_head": heatmaps.per_opportunity_head,
            "events": event_rows,
            "resample_source": bool(
                _channel_bootstrap_policy(
                    prepared, config, plan, channel
                ).resample_source
            ),
        }
        empty_columns = np.zeros(len(axis.labels), dtype=np.int64)
        replicates_seen = 0

        def graph_distance_reduce(rows):
            nonlocal replicates_seen
            result = distance_profile_reduce(rows)
            replicates_seen += 1
            # cells_r is nan exactly where no resampled graph supported the column.
            empty_columns[~np.isfinite(result[1, :-1]).any(axis=0)] += 1
            return result

        output["channels"][channel]["distance_intervals"] = nested_percentile_interval(
            distance_observations[channel],
            _channel_bootstrap_policy(
                prepared, config, plan, channel
            ),
            graph_reduce=graph_distance_reduce,
        )
        support_graphs, support_pairs = column_support(
            graph_support,
            {
                graph_id: len(plan[graph_id][channel]["sources"])
                for graph_id in graph_ids
            },
        )
        reportable = np.asarray(
            [
                reportable_bin(
                    [key for key in graph_ids if graph_support[key][column] > 0],
                    int(support_pairs[column]),
                    policy=config.bootstrap,
                )
                for column in range(len(axis.labels))
            ]
        )
        output["channels"][channel]["distance_support"] = {
            "axis": axis.labels,
            "graphs": support_graphs,
            "pairs": support_pairs,
            "reportable": reportable,
            "minimum_graphs": int(config.bootstrap.minimum_graphs),
            "minimum_pairs": int(config.bootstrap.minimum_pairs),
            "empty_replicate_fraction": (
                empty_columns / float(replicates_seen) if replicates_seen else empty_columns
            ),
            # Denominator is the point estimate plus every bootstrap replicate.
            "reduce_calls": int(replicates_seen),
        }
        suppressed = [
            str(axis.labels[column])
            for column in range(len(axis.labels))
            if not reportable[column]
        ]
        if suppressed:
            audit_check(
                False,
                "distance.column_below_reporting_floor",
                f"{channel} distance columns {suppressed} fall below the registered "
                f"{int(config.bootstrap.minimum_graphs)}-graph/"
                f"{int(config.bootstrap.minimum_pairs)}-pair reporting floor and are suppressed "
                "in every distance figure",
                context={"channel": channel, "columns": suppressed},
            )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output["execution"] = {
        "clean_jacobian_graphs": len(clean_by_graph),
        "clean_graph_batches": clean_execution,
        "clean_reused_across_channels": True,
        "event_graph_batches": execution_reports,
    }
    semantic = output["channels"]["semantic"]["raw"]
    structural = output["channels"]["structural"]["raw"]
    coordinates = head_coordinates(
        semantic,
        structural,
        score_floor=config.numerical.score_floor,
        epsilon=config.numerical.selectivity_epsilon,
        activity_floor=config.families.activity_floor,
    )
    output["coordinates"] = coordinates
    output["clean_throughput"] = np.stack(
        [throughput_graph[key] for key in sorted(throughput_graph)]
    ).mean(axis=0)
    output["clean_attention_distance_graph"] = attention_graph
    output["clean_attention_distance"] = np.stack(
        [attention_graph[key] for key in sorted(attention_graph)]
    ).mean(axis=0)
    output["families"] = freeze_families(
        coordinates,
        tail_fraction=config.families.tail_fraction,
        central_fraction=config.families.central_fraction,
        central_pool_fraction=config.families.central_pool_fraction,
    )
    output["matched_controls"] = freeze_matched_controls(
        coordinates,
        output["families"],
        output["clean_throughput"],
        rng_seed=int(config.analysis_seed),
    )
    for channel in CHANNELS:
        channel_output = output["channels"][channel]
        family_profiles = {}
        for family_name, family in output["families"].items():
            if not family:
                continue
            exact_graph = {}
            normalized_graph = {}
            for graph_id, contribution in channel_output[
                "graph_distance_contribution"
            ].items():
                support = channel_output["graph_distance_support"][graph_id]
                exact = np.sum(
                    np.stack(
                        [np.asarray(contribution)[layer, head] for layer, head in family]
                    ),
                    axis=0,
                )
                normalized = np.full_like(exact, np.nan, dtype=np.float64)
                np.divide(exact, support, out=normalized, where=np.asarray(support) > 0)
                exact_graph[int(graph_id)] = exact
                normalized_graph[int(graph_id)] = normalized
            family_profiles[family_name] = {
                "exact_graph": exact_graph,
                "per_opportunity_graph": normalized_graph,
                "exact": np.stack(
                    [exact_graph[key] for key in sorted(exact_graph)]
                ).mean(axis=0),
                "per_opportunity": supported_mean(
                    np.stack(
                        [normalized_graph[key] for key in sorted(normalized_graph)]
                    ),
                    axis=0,
                ),
            }
        channel_output["family_distance_profiles"] = family_profiles
    output["family_attention_distance"] = {
        family_name: np.mean(
            np.stack(
                [
                    np.mean(
                        np.stack(
                            [
                                attention_graph[graph_id][layer, head]
                                for layer, head in family
                            ]
                        ),
                        axis=0,
                    )
                    for graph_id in sorted(attention_graph)
                ]
            ),
            axis=0,
        )
        for family_name, family in output["families"].items()
        if family
    }

    # Paired channel draws transform the complete hierarchy through normalization and J/D_rel.
    semantic_by_key = {
        (row.graph, row.source, row.donor): row.value for row in observations["semantic"]
    }
    structural_by_key = {
        (row.graph, row.source, row.donor): row.value for row in observations["structural"]
    }
    def transform(value):
        transformed = head_coordinates(
            value[0],
            value[1],
            score_floor=config.numerical.score_floor,
            epsilon=config.numerical.selectivity_epsilon,
            activity_floor=config.families.activity_floor,
        )
        return np.stack(
            (
                transformed.raw_semantic,
                transformed.raw_structural,
                transformed.normalized_semantic,
                transformed.normalized_structural,
                transformed.joint_sensitivity,
                transformed.selectivity,
            )
        )

    if prepared.task.paired_channel_sources:
        common = sorted(set(semantic_by_key) & set(structural_by_key))
        paired = [
            Observation(
                int(prepared.grit.sc.seed),
                graph,
                source,
                donor,
                np.stack(
                    (
                        semantic_by_key[(graph, source, donor)],
                        structural_by_key[(graph, source, donor)],
                    )
                ),
            )
            for graph, source, donor in common
        ]
        if paired:
            output["intervals"] = nested_percentile_interval(
                paired,
                config.bootstrap,
                transform=transform,
                retain_draws=True,
            )
            output["interval_pairing"] = "source-and-graph-paired"
    else:
        source_resampling = tuple(
            _channel_bootstrap_policy(prepared, config, plan, channel).resample_source
            for channel in CHANNELS
        )
        output["intervals"] = paired_channel_percentile_interval(
            observations["semantic"],
            observations["structural"],
            config.bootstrap,
            transform=transform,
            resample_source=source_resampling,
            retain_draws=True,
        )
        output["interval_pairing"] = "graph-paired/channel-source-independent"
    coordinate_interval = output.get("intervals")
    if coordinate_interval is not None and coordinate_interval.draws is not None:
        output["specialist_classification"] = freeze_threshold_specialists(
            coordinates,
            selectivity_interval=(
                coordinate_interval.low[5],
                coordinate_interval.high[5],
            ),
            preference_threshold=config.families.equivalence_half_width,
            activity_threshold=config.families.activity_floor,
            candidate_limit=config.families.specialist_candidate_limit,
            minimum_candidate_pairs=config.families.specialist_minimum_pairs,
        )
        output["specialisation_diagnostics"] = specialisation_diagnostics(
            coordinates,
            output["families"],
            coordinate_interval.draws,
            selectivity_interval=(
                coordinate_interval.low[5],
                coordinate_interval.high[5],
            ),
            activity_floor=config.families.activity_floor,
            tail_fraction=config.families.tail_fraction,
            central_fraction=config.families.central_fraction,
            central_pool_fraction=config.families.central_pool_fraction,
            equivalence_half_width=config.families.equivalence_half_width,
            membership_stability_floor=config.families.membership_stability_floor,
            generalist_fraction_floor=config.families.generalist_fraction_floor,
        )
        # The stability summaries above are sufficient; the full 2,000-draw tensor is deliberately
        # transient so the score cache stays compact.
        output["intervals"] = dataclasses.replace(coordinate_interval, draws=None)
    cache.save("scores", "raw", output)
    cache.save_audit("scores_manifest", plan)
    return output


def _carriage_graph_batch(
    prepared: PreparedTask,
    config: MethodologyConfig,
    plan: Mapping[int, Mapping[str, Any]],
    clean_by_graph: Mapping[int, Any],
    channel: str,
    graph_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Run grouped endpoint forwards, then compute graph-local carriage fields."""

    import torch

    from ..carriage.core import project_final_states

    contexts: list[dict[str, Any]] = []
    groups: list[list[Any]] = []
    for graph_id in graph_ids:
        graph_id = int(graph_id)
        base = prepared.grit.eval_ds[graph_id]
        sources = plan[graph_id][channel]["sources"]
        if not sources:
            continue
        variants, records = _rebuild_graph_events(
            prepared, config, "carriage", graph_id, channel, sources
        )
        if [record.record() for record in records] != list(
            plan[graph_id][channel]["records"]
        ):
            raise RuntimeError(
                "deterministic carriage-event replay changed its manifest for "
                f"graph={graph_id} channel={channel}; refusing misaligned results"
            )
        contexts.append(
            {
                "graph_id": graph_id,
                "base": base,
                "sources": sources,
                "records": records,
                "clean": clean_by_graph[graph_id],
            }
        )
        groups.append([base, *variants])
    if not groups:
        return []
    captures = _capture_event_groups(prepared, groups)
    if len(captures) != len(contexts):
        raise RuntimeError("grouped carriage capture changed the base-graph group count")

    results: list[dict[str, Any]] = []
    for context, captured in zip(contexts, captures):
        graph_id = context["graph_id"]
        base = context["base"]
        sources = context["sources"]
        records = context["records"]
        clean = context["clean"]
        S = len(sources)
        h_clean = captured.final_state[0]
        carriers, width = int(h_clean.shape[-2]), int(h_clean.shape[-1])
        donor_counts = tuple(
            sum(int(record.source) == int(source) for record in records)
            for source in sources
        )
        if any(count < 1 for count in donor_counts):
            raise RuntimeError(
                f"carriage graph {graph_id} has a source without donor events"
            )
        if sum(donor_counts) != int(captured.final_state.shape[0]) - 1:
            raise RuntimeError(
                f"carriage graph {graph_id} event count does not align with its manifest"
            )
        flat_h_event = captured.final_state[1:]
        h_event_by_source = tuple(
            value.unsqueeze(0)
            for value in torch.split(flat_h_event, donor_counts, dim=0)
        )
        functional_events = getattr(
            prepared.backend, "functional_carriage_events", functional_carriage_events
        )
        event_f_by_source = tuple(
            functional_events(
                h_clean[None, None, :, :] - source_events,
                clean.final_state,
            )[0]
            for source_events in h_event_by_source
        )
        F = (
            torch.stack([value.mean(dim=0) for value in event_f_by_source])
            .t()
            .detach()
            .cpu()
            .numpy()
        )
        max_donors = max(donor_counts)

        def pad_event_values(values, *, fill=float("nan")):
            shape = (S, max_donors) + tuple(values[0].shape[1:])
            padded = values[0].new_full(shape, fill)
            for source_position, value in enumerate(values):
                padded[source_position, : value.shape[0]] = value
            return padded

        event_f = pad_event_values(event_f_by_source)
        integrated_by_source = None
        endpoint_replay_error = 0.0
        B = None
        if config.compute_beneficial_carriage:
            target = clean.capture.target.reshape(1, -1)
            state_loss_factory = getattr(
                prepared.backend, "loss_from_states", None
            )
            if callable(state_loss_factory):
                replay_loss = state_loss_factory(base, target)
                integrated_by_source = tuple(
                    beneficial_carriage(
                        h_clean,
                        source_events,
                        loss_from_states=replay_loss,
                        atol=config.numerical.integrated_atol,
                        rtol=config.numerical.integrated_rtol,
                        max_intervals=config.numerical.integrated_max_intervals,
                        tolerance=max(
                            config.numerical.integrated_atol * 5,
                            config.numerical.reconstruction_tolerance,
                        ),
                    )
                    for source_events in h_event_by_source
                )
                replay_clean_states = h_clean.unsqueeze(0)
                replay_event_states = flat_h_event
                with torch.no_grad():
                    replay_clean_loss = replay_loss(replay_clean_states)
                    replay_event_loss = replay_loss(replay_event_states)
            else:
                carrier_weights = prepared.backend.carriage_weights(base, h_clean)
                replay_loss = prepared.backend.loss_from_pooled(target)
                integrated_by_source = tuple(
                    beneficial_carriage(
                        h_clean,
                        source_events,
                        replay_loss,
                        carrier_weights=carrier_weights,
                        atol=config.numerical.integrated_atol,
                        rtol=config.numerical.integrated_rtol,
                        max_intervals=config.numerical.integrated_max_intervals,
                        tolerance=max(
                            config.numerical.integrated_atol * 5,
                            config.numerical.reconstruction_tolerance,
                        ),
                    )
                    for source_events in h_event_by_source
                )
                pooled_clean = project_final_states(
                    h_clean.unsqueeze(0), carrier_weights
                )
                pooled_event = project_final_states(
                    flat_h_event, carrier_weights
                )
                with torch.no_grad():
                    replay_clean_loss = replay_loss(pooled_clean)
                    replay_event_loss = replay_loss(pooled_event)
            with torch.no_grad():
                actual_clean_loss = prepared.backend.loss_per_graph(
                    captured.prediction[0:1], captured.target[0:1]
                )
                actual_event_loss = prepared.backend.loss_per_graph(
                    captured.prediction[1:], captured.target[1:]
                )
            endpoint_replay_error = float(
                torch.max(
                    torch.abs(
                        torch.cat((replay_clean_loss, replay_event_loss))
                        - torch.cat((actual_clean_loss, actual_event_loss))
                    )
                ).item()
            )
            within_tolerance(
                endpoint_replay_error,
                config.numerical.reconstruction_tolerance,
                "carriage.endpoint_replay",
                "pooling-to-readout replay error",
                context={"graph": graph_id, "channel": channel},
            )
            B = (
                torch.cat(
                    [value.field for value in integrated_by_source],
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )
        pristine = shortest_path_distances(base.edge_index, int(base.num_nodes))
        distance = prepared.backend.carriage_distance_matrix(
            base, sources, pristine, channel=channel
        )
        graph_field = {
            "sources": tuple(sources),
            "donor_counts": donor_counts,
            "F_sens": F,
            "distance": distance,
            "carrier_kinds": tuple(
                prepared.backend.carriage_carrier_kind(
                    base, carrier, channel=channel
                )
                for carrier in range(carriers)
            ),
            "event_F_sens": event_f.detach().cpu().numpy(),
            "endpoint_replay_error": endpoint_replay_error,
        }
        if integrated_by_source is not None:
            event_b_tensor = pad_event_values(
                tuple(value.event_field[0] for value in integrated_by_source)
            )
            event_loss_tensor = pad_event_values(
                tuple(
                    value.event_loss_increase[0, :, None]
                    for value in integrated_by_source
                )
            ).squeeze(-1)
            quadrature_tensor = pad_event_values(
                tuple(
                    value.quadrature_error[0, :, None]
                    for value in integrated_by_source
                )
            ).squeeze(-1)
            residual_tensor = pad_event_values(
                tuple(
                    value.completeness_residual[0, :, None]
                    for value in integrated_by_source
                )
            ).squeeze(-1)
            converged_tensor = pad_event_values(
                tuple(
                    value.converged[0, :, None]
                    for value in integrated_by_source
                ),
                fill=False,
            ).squeeze(-1)
            graph_field.update(
                {
                    "B": B,
                    "event_B": event_b_tensor.detach().cpu().numpy(),
                    "event_loss_increase": event_loss_tensor.detach().cpu().numpy(),
                    "quadrature_error": quadrature_tensor.detach().cpu().numpy(),
                    "completeness_residual": residual_tensor.detach().cpu().numpy(),
                    "converged": converged_tensor.detach().cpu().numpy(),
                }
            )
            event_b = event_b_tensor.detach().cpu().numpy()
        else:
            event_b = None
        event_f_np = event_f.detach().cpu().numpy()
        pair_rows: list[dict[str, Any]] = []
        for source_position, source in enumerate(sources):
            for donor in range(donor_counts[source_position]):
                for carrier in range(carriers):
                    pair_rows.append(
                        {
                            "seed": int(prepared.grit.sc.seed),
                            "graph_id": graph_id,
                            "source": int(source),
                            "donor": donor,
                            "carrier": carrier,
                            "distance": float(distance[carrier, source_position]),
                            "carrier_kind": prepared.backend.carriage_carrier_kind(
                                base, carrier, channel=channel
                            ),
                            "F_sens": float(
                                event_f_np[source_position, donor, carrier]
                            ),
                            "channel": channel,
                        }
                    )
                    if event_b is not None:
                        pair_rows[-1]["B"] = float(
                            event_b[source_position, donor, carrier]
                        )
        results.append(
            {
                "graph_id": graph_id,
                "graph_field": graph_field,
                "pair_rows": pair_rows,
                "paths": (
                    sum(
                        int(value.converged.numel())
                        for value in (integrated_by_source or ())
                    )
                ),
                "capped": (
                    sum(
                        int((~value.converged).sum().item())
                        for value in (integrated_by_source or ())
                    )
                ),
            }
        )
    return results


def run_carriage(
    prepared: PreparedTask,
    config: MethodologyConfig,
    *,
    plan: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute F_sens and donor-wise signed path-integrated B for both channels."""

    import torch

    plan = dict(plan or _stage_plan(prepared, config, "carriage"))
    cache = _cache(prepared, config, plan)
    if config.resume and not config.force:
        cached = cache.load("carriage", "fields", strict=True)
        if cached is not None:
            log(f"[cache] loaded canonical carriage for {prepared.task.name}")
            if prepared.progress is not None:
                prepared.progress.emit(
                    "cache_hit", phase="carriage", cache="consolidated"
                )
            return cached
    output: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_hash": _manifest_hash(plan),
        "channels": {},
    }
    graph_ids = sorted(plan)
    cached_graphs: dict[str, dict[int, Any]] = {
        channel: {} for channel in CHANNELS
    }
    missing_graphs: dict[str, list[int]] = {
        channel: [] for channel in CHANNELS
    }
    for channel in CHANNELS:
        for graph_id in graph_ids:
            shard = (
                cache.load(
                    f"carriage/{channel}",
                    f"graph_{int(graph_id):06d}",
                    strict=True,
                )
                if config.resume and not config.force
                else None
            )
            if shard is None:
                missing_graphs[channel].append(int(graph_id))
            else:
                cached_graphs[channel][int(graph_id)] = shard
    clean_needed = sorted(
        set(missing_graphs["semantic"]) | set(missing_graphs["structural"])
    )
    clean_by_graph, clean_execution = _prepare_clean_jacobians(
        prepared, config, clean_needed
    )
    execution_reports: dict[str, Any] = {}
    for channel in CHANNELS:
        pair_rows: list[dict[str, Any]] = []
        graph_fields: dict[int, dict[str, Any]] = {}
        capped = 0
        paths = 0

        def consume_carriage_batch(results, *, persist: bool = True):
            nonlocal capped, paths
            for result in results:
                graph_id = int(result["graph_id"])
                graph_fields[graph_id] = result["graph_field"]
                pair_rows.extend(result["pair_rows"])
                paths += int(result["paths"])
                capped += int(result["capped"])
                if persist:
                    cache.save(
                        f"carriage/{channel}",
                        f"graph_{graph_id:06d}",
                        result,
                    )

        consume_carriage_batch(
            [cached_graphs[channel][key] for key in sorted(cached_graphs[channel])],
            persist=False,
        )
        report = execute_graph_batches(
            missing_graphs[channel],
            graphs_per_batch=config.execution.graphs_per_batch,
            execute=lambda chunk: _carriage_graph_batch(
                prepared,
                config,
                plan,
                clean_by_graph,
                channel,
                chunk,
            ),
            consume=consume_carriage_batch,
            oom_backoff=config.execution.oom_backoff,
            item_cost=lambda graph_id: _event_item_cost(
                prepared, config, plan, int(graph_id), channel
            ),
            max_cost=config.execution.replica_pair_budget,
            on_batch=_progress_batch_callback(
                prepared,
                phase="carriage",
                channel=channel,
                cached=len(cached_graphs[channel]),
            ),
        )
        execution_reports[channel] = {
            **dataclasses.asdict(report),
            "cache_hits": len(cached_graphs[channel]),
            "cache_misses": len(missing_graphs[channel]),
        }
        capped_fraction = float(capped / paths) if paths else 0.0
        if config.compute_beneficial_carriage:
            within_tolerance(
                capped_fraction,
                config.numerical.integrated_unconverged_fraction,
                "carriage.capped_paths",
                f"{channel} Beneficial carriage capped-path fraction",
                context={"channel": channel, "paths": int(paths), "capped": int(capped)},
            )
        maximum_distance = max(
            (
                int(np.max(field["distance"][np.isfinite(field["distance"])]))
                for field in graph_fields.values()
                if np.isfinite(field["distance"]).any()
            ),
            default=0,
        )
        bins = adaptive_distance_bins(maximum_distance)
        far_thresholds = tuple(range(maximum_distance + 1))
        additive = {
            graph_id: {
                key: value
                * (
                    _source_population_size(
                        prepared,
                        prepared.grit.eval_ds[int(graph_id)],
                        channel,
                    )
                    / len(field["sources"])
                )
                for key, value in additive_beneficial_mass(
                    field["B"],
                    field["distance"],
                    bins=bins,
                    far_thresholds=far_thresholds,
                ).items()
            }
            for graph_id, field in graph_fields.items()
            if config.compute_beneficial_carriage
        }
        additive_observations = []
        for graph_id, field in graph_fields.items():
            if not config.compute_beneficial_carriage:
                continue
            n = _source_population_size(
                prepared,
                prepared.grit.eval_ds[int(graph_id)],
                channel,
            )
            event_b = np.asarray(field["event_B"])
            distance = np.asarray(field["distance"])
            donor_counts = tuple(
                int(value)
                for value in field.get(
                    "donor_counts",
                    (event_b.shape[1],) * len(field["sources"]),
                )
            )
            for source_position, source in enumerate(field["sources"]):
                for donor in range(donor_counts[source_position]):
                    source_values = event_b[source_position, donor]
                    bin_values = [
                        np.sum(
                            source_values[
                                np.isfinite(distance[:, source_position])
                                & (distance[:, source_position] >= lower)
                                & (distance[:, source_position] <= upper)
                            ]
                        )
                        for lower, upper in bins
                    ]
                    far_values = [
                        np.sum(
                            source_values[
                                np.isfinite(distance[:, source_position])
                                & (distance[:, source_position] > threshold)
                            ]
                        )
                        for threshold in far_thresholds
                    ]
                    # Source sampling is uniform without replacement. Multiplying each
                    # source contribution by n makes the later source mean a total-mass
                    # estimator for the graph rather than a per-source mean.
                    additive_observations.append(
                        Observation(
                            seed=int(prepared.grit.sc.seed),
                            graph=int(graph_id),
                            source=int(source),
                            donor=int(donor),
                            value=n
                            * np.asarray(
                                [*bin_values, *far_values], dtype=np.float64
                            ),
                        )
                    )
        additive_interval = (
            nested_percentile_interval(
                additive_observations,
                _channel_bootstrap_policy(prepared, config, plan, channel),
            )
            if additive_observations
            else None
        )
        bin_count = len(bins)
        channel_output = {
            "graph_fields": graph_fields,
            "pairs": pair_rows,
            "capped_paths": capped,
            "total_paths": paths,
            "capped_fraction": capped_fraction,
            "distance_bins": bins,
            "far_thresholds": far_thresholds,
            "resample_source": bool(
                _channel_bootstrap_policy(
                    prepared, config, plan, channel
                ).resample_source
            ),
        }
        if additive_interval is not None:
            channel_output.update(
                {
                    "S_B": additive_interval.estimate[:bin_count],
                    "B_far": additive_interval.estimate[bin_count:],
                    "additive_intervals": {
                        "order": (
                            tuple(f"{lower}-{upper}" for lower, upper in bins)
                            + tuple(f"far>{value}" for value in far_thresholds)
                        ),
                        "interval": additive_interval,
                    },
                    "additive_graph": additive,
                }
            )
        output["channels"][channel] = channel_output
    output["execution"] = {
        "clean_jacobian_graphs": len(clean_by_graph),
        "clean_graph_batches": clean_execution,
        "clean_reused_across_channels": True,
        "event_graph_batches": execution_reports,
    }
    cache.save("carriage", "fields", output)
    cache.save_audit("carriage_manifest", plan)
    return output


def _association_statistic(
    associations: Mapping[str, Any],
    name: str,
    *,
    active: bool = False,
    interval: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Rank correlation, nested interval, and permutation p for one validation panel.

    All three are already estimated by the causal stage; this only selects the active-head variant
    where the coordinate is only defined for active heads, and pairs the value with its interval.
    """

    record = associations.get(name)
    if not record:
        return None
    pooled = record.get("pooled_active" if active else "pooled") or {}
    permutation = (
        record.get("within_layer_permutation_active" if active else "within_layer_permutation")
        or {}
    )
    statistic: dict[str, Any] = {
        "rho": pooled.get("rho"),
        "n": pooled.get("n"),
        "p": permutation.get("p"),
    }
    order = list((interval or associations).get("nested_interval_order", ()) or ())
    low = (interval or associations).get("nested_interval_low")
    high = (interval or associations).get("nested_interval_high")
    # The interval order labels the active variants explicitly; the report keys do not.
    key = f"{name}_active" if active else name
    if low is not None and high is not None and key in order:
        position = order.index(key)
        statistic["low"] = float(np.asarray(low)[position])
        statistic["high"] = float(np.asarray(high)[position])
    return statistic


def _sources_per_graph(channel_scores: Mapping[str, Any]) -> dict[int, int]:
    """Estimable source count per graph, read back from the cached event table."""

    sources: dict[int, set[int]] = {}
    for row in channel_scores.get("events", ()):
        sources.setdefault(int(row["graph_id"]), set()).add(int(row["source"]))
    return {key: len(value) for key, value in sources.items()}


def _display_distance(
    channel_scores: Mapping[str, Any],
    config: MethodologyConfig,
    display: DisplayAxis,
    *,
    seed: int,
) -> dict[str, Any] | None:
    """Every distance quantity a figure needs, on the grouped display axis.

    Grouping is applied to the cached per-graph and per-event sufficient statistics, then the
    registered estimators are rerun on top: mass is summed, support-normalized quantities are
    recomputed as summed contribution over summed support inside each graph, and the interval is a
    fresh nested bootstrap over grouped observations. Nothing here re-derives a grouped value from
    an already-aggregated one, so no figure shows a statistic the estimator would not produce for
    that grouping.
    """

    graph_contribution = channel_scores.get("graph_distance_contribution")
    graph_support = channel_scores.get("graph_distance_support")
    if not graph_contribution or not graph_support:
        return None
    contribution = {
        key: display.group_sum(value) for key, value in graph_contribution.items()
    }
    support = {key: display.group_sum(value) for key, value in graph_support.items()}
    heatmaps = score_heatmaps(
        contribution,
        support,
        reconstruction_tolerance=config.numerical.reconstruction_tolerance,
        graph_scores=channel_scores.get("graph_scores"),
    )
    counts = _sources_per_graph(channel_scores)
    graphs, pairs = column_support(
        support, {key: counts.get(int(key), 0) for key in support}
    )
    keys = sorted(support)
    reportable = np.asarray(
        [
            reportable_bin(
                [key for key in keys if support[key][column] > 0],
                int(pairs[column]),
                policy=config.bootstrap,
            )
            for column in range(len(pairs))
        ]
    )
    if display.identity and channel_scores.get("distance_intervals") is not None:
        interval = channel_scores["distance_intervals"]
    else:
        log(
            f"[figures] rebuilding distance intervals on {len(display.labels)} display columns "
            f"from {len(channel_scores.get('events', ()))} cached events"
        )
        interval = nested_percentile_interval(
            [
                Observation(
                    seed=int(seed),
                    graph=int(row["graph_id"]),
                    source=int(row["source"]),
                    donor=int(row["draw"]),
                    value=_display_observation(row, display),
                )
                for row in channel_scores["events"]
            ],
            dataclasses.replace(
                config.bootstrap,
                resample_source=bool(
                    channel_scores.get("resample_source", True)
                ),
            ),
            graph_reduce=distance_profile_reduce,
        )
    estimable = getattr(interval, "estimable_draws", None)
    # Exact mass is additive, so a widened group would draw as a resurgence; report it per unit
    # distance instead. Support-normalized panels are already width-invariant. Both are identities
    # at unit resolution, and dividing by a width is linear, so the band transforms with the point.
    widths = display.widths
    exact_interval = tuple(
        np.asarray(value)[0] / widths
        for value in (interval.estimate, interval.low, interval.high)
    )
    return {
        "exact_head": heatmaps.exact_head / widths,
        "per_opportunity_head": heatmaps.per_opportunity_head,
        "interval": interval,
        "exact_estimate": exact_interval[0],
        "exact_low": exact_interval[1],
        "exact_high": exact_interval[2],
        "graphs": graphs,
        "pairs": pairs,
        "reportable": reportable,
        "minimum_graphs": int(config.bootstrap.minimum_graphs),
        "minimum_pairs": int(config.bootstrap.minimum_pairs),
        "empty_replicate_fraction": (
            None
            if estimable is None
            else 1.0 - np.asarray(estimable[1, -1], dtype=np.float64)
            / float(interval.replicates)
        ),
    }


def _display_observation(row: Mapping[str, Any], display: DisplayAxis) -> np.ndarray:
    contribution = display.group_sum(np.asarray(row["distance_contribution"]))
    support = display.group_sum(np.asarray(row["distance_support"]))
    return np.stack(
        (contribution, np.broadcast_to(support, contribution.shape))
    )


def _mask_columns(values: Any, reportable: Any) -> np.ndarray:
    """Blank the trailing distance axis of `values` wherever the reporting floor is not met."""

    values = np.asarray(values, dtype=np.float64).copy()
    reportable = np.asarray(reportable, dtype=bool)
    if values.shape[-1] != reportable.shape[-1]:
        raise ValueError("reportable mask does not align with the distance axis")
    values[..., ~reportable] = np.nan
    return values


def _coordinate_plot_data(scores: Mapping[str, Any], seed: int) -> HeadPlotData:
    interval = scores.get("intervals")
    if interval is None:
        return HeadPlotData(scores["coordinates"], int(seed))
    return HeadPlotData(
        scores["coordinates"],
        int(seed),
        semantic_interval=(interval.low[2], interval.high[2]),
        structural_interval=(interval.low[3], interval.high[3]),
        joint_interval=(interval.low[4], interval.high[4]),
        selectivity_interval=(interval.low[5], interval.high[5]),
    )


def _carriage_profile(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    config: MethodologyConfig,
    *,
    sum_within_event: bool = False,
    bootstrap_policy: Any | None = None,
) -> tuple[list[str], np.ndarray, tuple[np.ndarray, np.ndarray]]:
    finite_distances = [int(row["distance"]) for row in rows if np.isfinite(row["distance"])]
    maximum = max(finite_distances, default=0)
    bins = adaptive_distance_bins(maximum)
    labels = [str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in bins]
    special_kinds = sorted(
        {
            str(row.get("carrier_kind", "molecular_node"))
            for row in rows
            if str(row.get("carrier_kind", "molecular_node")) != "molecular_node"
            and not np.isfinite(float(row["distance"]))
        }
    )
    labels.extend(kind.replace("_", " ") for kind in special_kinds)
    estimates, lows, highs = [], [], []
    selectors = [
        lambda row, lower=lo, upper=hi: (
            np.isfinite(row["distance"])
            and lower <= int(row["distance"]) <= upper
        )
        for lo, hi in bins
    ]
    selectors.extend(
        lambda row, kind=kind: str(row.get("carrier_kind")) == kind
        for kind in special_kinds
    )
    for select in selectors:
        selected = [
            row
            for row in rows
            if select(row) and np.isfinite(float(row[field]))
        ]
        graph_ids = {int(row["graph_id"]) for row in selected}
        pairs = {
            (int(row["graph_id"]), int(row["carrier"]), int(row["source"]))
            for row in selected
        }
        if (
            len(graph_ids) < config.bootstrap.minimum_graphs
            or len(pairs) < config.bootstrap.minimum_pairs
        ):
            estimates.append(np.nan)
            lows.append(np.nan)
            highs.append(np.nan)
            continue
        grouped: dict[tuple[int, int, int, int], list[float]] = {}
        for row in selected:
            key = (
                int(row["seed"]),
                int(row["graph_id"]),
                int(row["source"]),
                int(row["donor"]),
            )
            grouped.setdefault(key, []).append(float(row[field]))
        if sum_within_event:
            observations = [
                Observation(
                    seed,
                    graph,
                    source,
                    donor,
                    float(np.sum(values)),
                )
                for (seed, graph, source, donor), values in grouped.items()
            ]

            def graph_reduce(values):
                return trimmed_mean(
                    values, config.bootstrap.trim_fraction, axis=0
                )

        else:
            observations = [
                Observation(
                    seed,
                    graph,
                    source,
                    donor,
                    np.asarray([np.sum(values), len(values)], dtype=np.float64),
                )
                for (seed, graph, source, donor), values in grouped.items()
            ]

            def graph_reduce(values):
                graph_pair_means = values[:, 0] / values[:, 1]
                return trimmed_mean(
                    graph_pair_means, config.bootstrap.trim_fraction, axis=0
                )

        interval = nested_percentile_interval(
            observations,
            bootstrap_policy or config.bootstrap,
            graph_reduce=graph_reduce,
        )
        estimates.append(float(interval.estimate))
        lows.append(float(interval.low))
        highs.append(float(interval.high))
    return labels, np.asarray(estimates), (np.asarray(lows), np.asarray(highs))


def _event_normalised_carriage_rows(
    rows: Sequence[Mapping[str, Any]],
    config: MethodologyConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize each donor event across all registered carriers before binning."""

    result = [dict(row) for row in rows]
    grouped: dict[tuple[int, int, int, int], list[int]] = {}
    for position, row in enumerate(result):
        key = (
            int(row["seed"]),
            int(row["graph_id"]),
            int(row["source"]),
            int(row["donor"]),
        )
        grouped.setdefault(key, []).append(position)

    functional_floor = float(config.numerical.effect_floor)
    beneficial_floor = float(config.numerical.integrated_atol)
    eligible_functional = 0
    eligible_beneficial = 0
    beneficial_present = bool(result and "B" in result[0])
    for positions in grouped.values():
        functional = np.asarray(
            [float(result[position]["F_sens"]) for position in positions],
            dtype=np.float64,
        )
        functional_normalised, functional_eligible, _ = (
            event_normalise_functional(
                functional[None, :],
                effect_floor=functional_floor,
            )
        )
        functional_ok = bool(functional_eligible[0])
        eligible_functional += int(functional_ok)
        for index, position in enumerate(positions):
            result[position]["F_sens_event_normalised"] = float(
                functional_normalised[0, index]
            )

        if not beneficial_present:
            continue
        beneficial = np.asarray(
            [float(result[position]["B"]) for position in positions],
            dtype=np.float64,
        )
        beneficial_denominator = float(np.sum(np.abs(beneficial)))
        beneficial_ok = bool(
            np.isfinite(beneficial).all()
            and np.isfinite(beneficial_denominator)
            and beneficial_denominator > beneficial_floor
        )
        eligible_beneficial += int(beneficial_ok)
        for position, value in zip(positions, beneficial):
            result[position]["B_event_normalised"] = (
                float(value / beneficial_denominator) if beneficial_ok else np.nan
            )

    total = len(grouped)
    metadata = {
        "level": "donor_event",
        "functional": {
            "formula": "F_sens[i] / sum_j F_sens[j]",
            "denominator_floor": functional_floor,
            "eligible_events": eligible_functional,
            "excluded_events": total - eligible_functional,
        },
        "beneficial": (
            {
                "formula": "B[i] / sum_j abs(B[j])",
                "denominator_floor": beneficial_floor,
                "eligible_events": eligible_beneficial,
                "excluded_events": total - eligible_beneficial,
            }
            if beneficial_present
            else {"computed": False}
        ),
        "total_events": total,
    }
    return result, metadata


def make_figures(
    prepared: PreparedTask,
    config: MethodologyConfig,
    scores: Mapping[str, Any],
    carriage: Mapping[str, Any] | None,
    causal: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    figure_values = config.figure_overrides
    if prepared.task.name in figure_values and isinstance(
        figure_values[prepared.task.name], Mapping
    ):
        figure_values = figure_values[prepared.task.name]
    theme = FigureTheme().with_overrides(figure_values)
    builder = FigureBuilder(
        prepared.output_dir / "figures",
        theme,
        modifier=TASK_FIGURE_MODIFIERS.get(prepared.task.name),
        common_metadata={
            "protocol_version": PROTOCOL_VERSION,
            "repository_commit": _repository_commit(),
            "protocol_fingerprint": config.fingerprint,
            "checkpoint_sha256": prepared.checkpoint_sha,
            "output_representation": prepared.task.output.representation,
            "sigma": prepared.sigma.tolist(),
        },
    )
    saved: dict[str, list[str]] = {}
    plot_data = _coordinate_plot_data(scores, int(prepared.grit.sc.seed))
    fig, axes = score_plane(plot_data, title=prepared.task.title, theme=theme)
    paths = builder.save(
        "structural_vs_semantic_scores",
        fig,
        axes,
        metadata={"task": prepared.task.name, "seed": int(prepared.grit.sc.seed)},
    )
    saved["score_plane"] = [str(path) for path in paths]
    fig, axes = joint_selectivity_plane(
        plot_data,
        title=prepared.task.title,
        equivalence_half_width=config.families.equivalence_half_width,
        theme=theme,
    )
    paths = builder.save(
        "selectivity_vs_joint_sensitivity",
        fig,
        axes,
        metadata={"task": prepared.task.name, "seed": int(prepared.grit.sc.seed)},
    )
    saved["coordinate_plane"] = [str(path) for path in paths]
    if scores.get("specialist_classification"):
        fig, axes = strong_specialist_map(
            plot_data,
            scores["specialist_classification"],
            title=prepared.task.title,
            theme=theme,
        )
        paths = builder.save(
            "strongest_D_rel_candidate_map",
            fig,
            axes,
            metadata={
                "task": prepared.task.name,
                "seed": int(prepared.grit.sc.seed),
                "classification": scores["specialist_classification"],
            },
        )
        saved["strongest_candidate_map"] = [str(path) for path in paths]
    if scores.get("specialisation_diagnostics"):
        fig, axes = selectivity_regime_diagnostics(
            plot_data,
            scores["specialisation_diagnostics"],
            scores["families"],
            theme=theme,
        )
        paths = builder.save(
            "selectivity_regime_diagnostics",
            fig,
            axes,
            metadata={
                "task": prepared.task.name,
                "seed": int(prepared.grit.sc.seed),
                "diagnostics": scores["specialisation_diagnostics"],
            },
        )
        saved["selectivity_regime"] = [str(path) for path in paths]
    display = display_bins(scores["axis"], max_points=int(theme.max_distance_points))
    for channel in CHANNELS:
        channel_scores = scores["channels"][channel]
        support = _display_distance(
            channel_scores, config, display, seed=int(prepared.grit.sc.seed)
        )
        labels = display.labels
        # The registered reporting floor is applied at presentation time only: cached measurements
        # keep every column, figures show only groups with adequate graph and pair support.
        reportable = (
            np.asarray(support["reportable"], dtype=bool)
            if support is not None
            else np.ones(len(labels), dtype=bool)
        )
        suppressed_columns = [
            str(label) for label, keep in zip(labels, reportable) if not keep
        ]
        reporting_metadata = {
            "distance_display": {
                "registered_columns": len(scores["axis"]),
                "display_columns": list(labels),
                "grouping": (
                    "unit resolution near the changed node, dyadic widening in the tail"
                    if not display.identity
                    else "none; the registered axis already fits"
                ),
            },
            "reporting_floor": {
                "minimum_graphs": int(config.bootstrap.minimum_graphs),
                "minimum_pairs": int(config.bootstrap.minimum_pairs),
                "suppressed_columns": suppressed_columns,
            },
        }
        if support is None:
            continue
        exact_head = support["exact_head"]
        per_opportunity_head = support["per_opportunity_head"]
        # Both variants share one estimate; row normalisation divides by each head's own total
        # before the reporting floor blanks columns, so suppression never inflates what remains.
        for normalised, suffix in ((False, ""), (True, "_row_normalised")):
            panels = (
                (row_normalised(exact_head), row_normalised(per_opportunity_head))
                if normalised
                else (exact_head, per_opportunity_head)
            )
            fig, axes = distance_heatmaps(
                _mask_columns(panels[0], reportable),
                _mask_columns(panels[1], reportable),
                labels,
                channel=channel,
                title=prepared.task.title,
                normalised=normalised,
                grouped=not display.identity,
                theme=theme,
            )
            paths = builder.save(
                f"{channel}_score_distance_heatmaps{suffix}",
                fig,
                axes,
                metadata={
                    "task": prepared.task.name,
                    "seed": int(prepared.grit.sc.seed),
                    "channel": channel,
                    "head_aggregation": "per-head rows, layer blocks, layer 0 at the top",
                    "normalisation": (
                        "each head divided by its own total over the full distance axis"
                        if normalised
                        else "none"
                    ),
                    **reporting_metadata,
                },
            )
            saved[f"{channel}_distance{suffix}"] = [str(path) for path in paths]
        interval = support["interval"]
        profile_row = int(prepared.grit.L)
        fig, axes = score_distance_profiles(
            labels,
            _mask_columns(support["exact_estimate"][profile_row], reportable),
            _mask_columns(interval.estimate[1, profile_row], reportable),
            exact_interval=(
                _mask_columns(support["exact_low"][profile_row], reportable),
                _mask_columns(support["exact_high"][profile_row], reportable),
            ),
            per_opportunity_interval=(
                _mask_columns(interval.low[1, profile_row], reportable),
                _mask_columns(interval.high[1, profile_row], reportable),
            ),
            channel=channel,
            grouped=not display.identity,
            theme=theme,
        )
        paths = builder.save(
            f"{channel}_score_distance_profiles",
            fig,
            axes,
            metadata={
                "task": prepared.task.name,
                "seed": int(prepared.grit.sc.seed),
                "channel": channel,
                "intervals": "nested percentile bootstrap",
                **reporting_metadata,
            },
        )
        saved[f"{channel}_distance_profile"] = [str(path) for path in paths]
        if support is not None:
            fig, axes = distance_support_profile(
                labels,
                support["graphs"],
                support["pairs"],
                empty_replicate_fraction=support["empty_replicate_fraction"],
                minimum_graphs=int(support["minimum_graphs"]),
                minimum_pairs=int(support["minimum_pairs"]),
                channel=channel,
                theme=theme,
            )
            paths = builder.save(
                f"{channel}_distance_column_support",
                fig,
                axes,
                metadata={
                    "task": prepared.task.name,
                    "seed": int(prepared.grit.sc.seed),
                    "channel": channel,
                    "graphs": np.asarray(support["graphs"]).tolist(),
                    "pairs": np.asarray(support["pairs"]).tolist(),
                    "empty_replicate_fraction": (
                        None
                        if support["empty_replicate_fraction"] is None
                        else np.asarray(support["empty_replicate_fraction"]).tolist()
                    ),
                    **reporting_metadata,
                },
            )
            saved[f"{channel}_distance_support"] = [str(path) for path in paths]
        if carriage is not None:
            carriage_channel = carriage["channels"][channel]
            rows = carriage_channel["pairs"]
            carriage_bootstrap = dataclasses.replace(
                config.bootstrap,
                resample_source=bool(
                    carriage_channel.get("resample_source", True)
                ),
            )
            labels, functional, functional_interval = _carriage_profile(
                rows,
                "F_sens",
                config,
                bootstrap_policy=carriage_bootstrap,
            )
            beneficial = None
            beneficial_interval = None
            if config.compute_beneficial_carriage:
                _, beneficial, beneficial_interval = _carriage_profile(
                    rows,
                    "B",
                    config,
                    bootstrap_policy=carriage_bootstrap,
                )
            fig, axes = carriage_profiles(
                labels,
                functional,
                beneficial,
                functional_interval=functional_interval,
                beneficial_interval=beneficial_interval,
                channel=channel,
                theme=theme,
            )
            paths = builder.save(
                f"{channel}_carriage_profiles",
                fig,
                axes,
                metadata={
                    "task": prepared.task.name,
                    "seed": int(prepared.grit.sc.seed),
                    "channel": channel,
                    "functional_estimand": "F_sens",
                    "beneficial_sign": (
                        "positive-is-beneficial"
                        if config.compute_beneficial_carriage
                        else "not_computed"
                    ),
                },
            )
            saved[f"{channel}_carriage"] = [str(path) for path in paths]

            normalised_rows, normalisation = _event_normalised_carriage_rows(
                rows, config
            )
            (
                normalised_labels,
                normalised_functional,
                normalised_functional_interval,
            ) = _carriage_profile(
                normalised_rows,
                "F_sens_event_normalised",
                config,
                sum_within_event=True,
                bootstrap_policy=carriage_bootstrap,
            )
            normalised_beneficial = None
            normalised_beneficial_interval = None
            if config.compute_beneficial_carriage:
                (
                    _,
                    normalised_beneficial,
                    normalised_beneficial_interval,
                ) = _carriage_profile(
                    normalised_rows,
                    "B_event_normalised",
                    config,
                    sum_within_event=True,
                    bootstrap_policy=carriage_bootstrap,
                )
            fig, axes = carriage_profiles(
                normalised_labels,
                normalised_functional,
                normalised_beneficial,
                functional_interval=normalised_functional_interval,
                beneficial_interval=normalised_beneficial_interval,
                channel=channel,
                event_normalised=True,
                theme=theme,
            )
            paths = builder.save(
                f"{channel}_carriage_profiles_event_normalised",
                fig,
                axes,
                metadata={
                    "task": prepared.task.name,
                    "seed": int(prepared.grit.sc.seed),
                    "channel": channel,
                    "functional_estimand": "F_sens",
                    "beneficial_sign": (
                        "positive-is-beneficial"
                        if config.compute_beneficial_carriage
                        else "not_computed"
                    ),
                    "normalisation": normalisation,
                },
            )
            saved[f"{channel}_carriage_event_normalised"] = [
                str(path) for path in paths
            ]
    if scores.get("family_attention_distance"):
        # Attention mass is a fraction of a fixed total, so grouping is an exact sum.
        fig, axes = attention_distance_profiles(
            display.labels,
            {
                family: display.group_density(np.asarray(profile))
                for family, profile in scores["family_attention_distance"].items()
            },
            grouped=not display.identity,
            theme=theme,
        )
        paths = builder.save(
            "clean_attention_distance_profiles",
            fig,
            axes,
            metadata={
                "task": prepared.task.name,
                "seed": int(prepared.grit.sc.seed),
                "status": "descriptive routing diagnostic",
                "distance_display": {
                    "registered_columns": len(scores["axis"]),
                    "display_columns": list(display.labels),
                },
            },
        )
        saved["attention_distance"] = [str(path) for path in paths]
    if causal is not None:
        focused = causal.get("focused_specialists", {})
        if focused.get("status") in {"estimable", "continuous_only"}:
            continuous = focused.get("continuous_analysis", {})
            if continuous.get("status") == "estimable":
                continuous_metrics = list(continuous["metric_order"])
                statistics = list(continuous["statistic_order"])
                continuous_interval = continuous["interval"]
                contrast_interval = continuous["head_contrast_interval"]
                contrast_metrics = list(
                    continuous["head_contrast_metric_order"]
                )
                coordinates = scores["coordinates"]
                score_interval = scores["intervals"]
                D = coordinates.selectivity.reshape(-1)
                active = coordinates.active.reshape(-1)
                layers = np.repeat(
                    np.arange(coordinates.joint_sensitivity.shape[0]),
                    coordinates.joint_sensitivity.shape[1],
                )
                titles = {
                    "restoration": "Restoration contrast",
                    "injection": "Injection contrast",
                    "necessity_fraction": "Necessity-fraction contrast",
                }
                panels = []
                for metric_position, metric in enumerate(continuous_metrics):
                    contrast_position = contrast_metrics.index(metric)
                    rho_position = statistics.index("spearman_rho")
                    beta_position = statistics.index(
                        "J_and_layer_adjusted_standardized_beta"
                    )
                    beta = float(
                        continuous_interval.estimate[
                            metric_position, beta_position
                        ]
                    )
                    beta_low = float(
                        continuous_interval.low[
                            metric_position, beta_position
                        ]
                    )
                    beta_high = float(
                        continuous_interval.high[
                            metric_position, beta_position
                        ]
                    )
                    panels.append(
                        {
                            "x": D,
                            "y": contrast_interval.estimate[
                                contrast_position
                            ],
                            "x_interval": (
                                score_interval.low[5].reshape(-1),
                                score_interval.high[5].reshape(-1),
                            ),
                            "y_interval": (
                                contrast_interval.low[contrast_position],
                                contrast_interval.high[contrast_position],
                            ),
                            "active": active,
                            "layer": layers,
                            "xlabel": r"Raw selectivity $D_{rel}$",
                            "ylabel": (
                                "Semantic minus structural causal response"
                            ),
                            "title": (
                                f"{titles[metric]}\n"
                                f"adjusted β={beta:.2f} "
                                f"[{beta_low:.2f}, {beta_high:.2f}]"
                            ),
                            "statistic": {
                                "rho": float(
                                    continuous_interval.estimate[
                                        metric_position, rho_position
                                    ]
                                ),
                                "low": float(
                                    continuous_interval.low[
                                        metric_position, rho_position
                                    ]
                                ),
                                "high": float(
                                    continuous_interval.high[
                                        metric_position, rho_position
                                    ]
                                ),
                                "n": int(np.sum(active)),
                            },
                            "zero_x": True,
                            "zero_y": True,
                        }
                    )
                fig, axes = causal_scatter_grid(panels, theme=theme)
                paths = builder.save(
                    "continuous_D_rel_causal_contrasts",
                    fig,
                    axes,
                    metadata={
                        "task": prepared.task.name,
                        "seed": int(prepared.grit.sc.seed),
                        "continuous_analysis": continuous,
                    },
                )
                saved["continuous_D_rel_causal"] = [
                    str(path) for path in paths
                ]
            for pair_set in focused["pair_set_order"]:
                fig, axes = specialist_causal_panels(
                    focused,
                    pair_set=pair_set,
                    metrics=("restoration", "injection"),
                    theme=theme,
                )
                paths = builder.save(
                    f"{pair_set}_specialist_restoration_injection",
                    fig,
                    axes,
                    metadata={
                        "task": prepared.task.name,
                        "seed": int(prepared.grit.sc.seed),
                        "pair_set": pair_set,
                        "focused_specialists": focused,
                    },
                )
                saved[f"{pair_set}_specialist_patching"] = [
                    str(path) for path in paths
                ]
                fig, axes = specialist_causal_panels(
                    focused,
                    pair_set=pair_set,
                    metrics=(
                        "necessity_fraction",
                        "gross_necessity_fraction",
                    ),
                    theme=theme,
                )
                paths = builder.save(
                    f"{pair_set}_specialist_donor_necessity",
                    fig,
                    axes,
                    metadata={
                        "task": prepared.task.name,
                        "seed": int(prepared.grit.sc.seed),
                        "pair_set": pair_set,
                        "focused_specialists": focused,
                    },
                )
                saved[f"{pair_set}_specialist_necessity"] = [
                    str(path) for path in paths
                ]
        if causal.get("regime_evidence"):
            fig, axes = causal_regime_summary(
                causal["regime_evidence"],
                theme=theme,
            )
            paths = builder.save(
                "causal_regime_summary",
                fig,
                axes,
                metadata={
                    "task": prepared.task.name,
                    "seed": int(prepared.grit.sc.seed),
                    "regime_evidence": causal["regime_evidence"],
                },
            )
            saved["causal_regime"] = [str(path) for path in paths]
        coordinates = scores["coordinates"]
        layers = np.repeat(
            np.arange(coordinates.joint_sensitivity.shape[0]),
            coordinates.joint_sensitivity.shape[1],
        )
        head_names = [
            f"head_L{layer}_H{head}"
            for layer in range(int(prepared.grit.L))
            for head in range(int(prepared.grit.H))
        ]
        score_interval = scores["intervals"]
        x_intervals = {
            "J": (score_interval.low[4].reshape(-1), score_interval.high[4].reshape(-1)),
            "D": (score_interval.low[5].reshape(-1), score_interval.high[5].reshape(-1)),
            "S_sem": (
                score_interval.low[0].reshape(-1),
                score_interval.high[0].reshape(-1),
            ),
            "S_str": (
                score_interval.low[1].reshape(-1),
                score_interval.high[1].reshape(-1),
            ),
        }
        causal_intervals = causal["summary"]["intervals"]
        target_order = list(causal_intervals["target_order"])
        target_position = {name: index for index, name in enumerate(target_order)}
        interval_object = causal_intervals["interval"]
        endpoint_order = list(causal_intervals["endpoint_order"])
        raw_size = 2 * len(target_order) * len(endpoint_order)
        point_raw = np.asarray(
            [
                [
                    [
                        causal["summary"]["targets"][name][channel][endpoint]
                        for endpoint in endpoint_order
                    ]
                    for name in target_order
                ]
                for channel in CHANNELS
            ]
        )
        point_calibrated = np.asarray(
            [
                [
                    causal["summary"]["targets"][name]["calibrated"][endpoint]
                    for endpoint in causal_intervals["calibrated_order"]
                ]
                for name in target_order
            ]
        )
        raw_low = interval_object.low[:raw_size].reshape(point_raw.shape)
        raw_high = interval_object.high[:raw_size].reshape(point_raw.shape)
        calibrated_size = point_calibrated.size
        calibrated_low = interval_object.low[
            raw_size : raw_size + calibrated_size
        ].reshape(point_calibrated.shape)
        calibrated_high = interval_object.high[
            raw_size : raw_size + calibrated_size
        ].reshape(point_calibrated.shape)
        head_positions = [target_position[name] for name in head_names]
        clean_interval_meta = causal["clean_ablation"]["_intervals"]
        clean_order = {
            name: index for index, name in enumerate(clean_interval_meta["target_order"])
        }
        clean_positions = [clean_order[name] for name in head_names]
        clean_interval = clean_interval_meta["interval"]
        clean_point = np.asarray(
            [
                causal["clean_ablation"][name]["prediction_movement"]
                for name in head_names
            ]
        )
        J = coordinates.joint_sensitivity.reshape(-1)
        D = coordinates.selectivity.reshape(-1)
        active = coordinates.active.reshape(-1)
        associations = causal["associations"]
        # The clean-ablation association carries its own interval, estimated in that stage.
        clean_association = clean_interval_meta.get("association_interval")
        clean_statistic = _association_statistic(
            associations,
            "J_vs_clean_prediction_movement",
            interval={
                "nested_interval_order": clean_interval_meta.get("association_order", ()),
                "nested_interval_low": (
                    None if clean_association is None else clean_association.low
                ),
                "nested_interval_high": (
                    None if clean_association is None else clean_association.high
                ),
            },
        )
        calibrated_names = list(causal_intervals["calibrated_order"])

        def calibrated_column(name):
            column = calibrated_names.index(name)
            return (
                point_calibrated[head_positions, column],
                (
                    calibrated_low[head_positions, column],
                    calibrated_high[head_positions, column],
                ),
            )

        gross_total, gross_total_ci = calibrated_column("gross_total_for_J")
        gross_contrast, gross_contrast_ci = calibrated_column(
            "gross_contrast_for_D_rel"
        )
        necessity_total, necessity_total_ci = calibrated_column(
            "necessity_total_for_J"
        )
        necessity_contrast, necessity_contrast_ci = calibrated_column(
            "necessity_contrast_for_D_rel"
        )
        panels = [
            {
                "x": J,
                "y": clean_point,
                "x_interval": x_intervals["J"],
                "y_interval": (
                    clean_interval.low[clean_positions, 0],
                    clean_interval.high[clean_positions, 0],
                ),
                "layer": layers,
                "xlabel": r"Joint sensitivity $J$",
                "ylabel": "Clean ablation prediction movement",
                "title": "Clean necessity",
                "statistic": clean_statistic,
            },
            {
                "x": J,
                "y": gross_total,
                "x_interval": x_intervals["J"],
                "y_interval": gross_total_ci,
                "layer": layers,
                "xlabel": r"Joint sensitivity $J$",
                "ylabel": "Calibrated total gross patch response",
                "title": "Gross causal response",
                "statistic": _association_statistic(associations, "J_vs_gross_total"),
            },
            {
                "x": J,
                "y": necessity_total,
                "x_interval": x_intervals["J"],
                "y_interval": necessity_total_ci,
                "layer": layers,
                "xlabel": r"Joint sensitivity $J$",
                "ylabel": "Calibrated total donor-wise necessity",
                "title": "Donor-wise necessity",
                "statistic": _association_statistic(
                    associations, "J_vs_necessity_total"
                ),
            },
            {
                "x": D[active],
                "y": gross_contrast[active],
                "x_interval": (
                    x_intervals["D"][0][active],
                    x_intervals["D"][1][active],
                ),
                "y_interval": (
                    gross_contrast_ci[0][active],
                    gross_contrast_ci[1][active],
                ),
                "layer": layers[active],
                "xlabel": r"Selectivity $D_{rel}$",
                "ylabel": "Calibrated gross channel contrast",
                "title": "Selectivity validation",
                "statistic": _association_statistic(
                    associations, "D_rel_vs_gross_contrast", active=True
                ),
                "zero_x": True,
                "zero_y": True,
            },
            {
                "x": D[active],
                "y": necessity_contrast[active],
                "x_interval": (
                    x_intervals["D"][0][active],
                    x_intervals["D"][1][active],
                ),
                "y_interval": (
                    necessity_contrast_ci[0][active],
                    necessity_contrast_ci[1][active],
                ),
                "layer": layers[active],
                "xlabel": r"Selectivity $D_{rel}$",
                "ylabel": "Calibrated necessity channel contrast",
                "title": "Selectivity and necessity",
                "statistic": _association_statistic(
                    associations, "D_rel_vs_necessity_contrast", active=True
                ),
                "zero_x": True,
                "zero_y": True,
            },
        ]
        fig, axes = causal_scatter_grid([panels[0]], theme=theme)
        paths = builder.save(
            "joint_sensitivity_vs_clean_head_ablation",
            fig,
            axes,
            metadata={
                "task": prepared.task.name,
                "seed": int(prepared.grit.sc.seed),
                "association": clean_statistic,
                "endpoint": "individual-head clean-ablation prediction movement",
            },
        )
        saved["clean_ablation_vs_J"] = [str(path) for path in paths]
        fig, axes = causal_scatter_grid(panels, theme=theme)
        paths = builder.save(
            "causal_validation_coordinates",
            fig,
            axes,
            metadata={"task": prepared.task.name, "seed": int(prepared.grit.sc.seed)},
        )
        saved["causal_coordinates"] = [str(path) for path in paths]

        # Same-channel raw-score calibration; cross-channel relationships remain in tables.
        calibration_panels = []
        for channel_index, channel in enumerate(CHANNELS):
            endpoint = endpoint_order.index("G_c")
            calibration_panels.append(
                {
                    "x": scores["channels"][channel]["raw"].reshape(-1),
                    "y": point_raw[channel_index, head_positions, endpoint],
                    "x_interval": x_intervals[
                        "S_sem" if channel == "semantic" else "S_str"
                    ],
                    "y_interval": (
                        raw_low[channel_index, head_positions, endpoint],
                        raw_high[channel_index, head_positions, endpoint],
                    ),
                    "layer": layers,
                    "xlabel": (
                        r"Raw Semantic score $S_{sem}$"
                        if channel == "semantic"
                        else r"Raw Structural score $S_{str}$"
                    ),
                    "ylabel": rf"Gross patch response $G_{{{channel[:3]}}}$",
                    "title": f"{channel.capitalize()} donor-swap",
                    "zero_y": True,
                }
            )
        fig, axes = causal_scatter_grid(calibration_panels, theme=theme)
        paths = builder.save(
            "raw_score_causal_calibration",
            fig,
            axes,
            metadata={"task": prepared.task.name, "seed": int(prepared.grit.sc.seed)},
        )
        saved["causal_calibration"] = [str(path) for path in paths]

        family_names = [
            name
            for name in (
                "family_semantic_leaning",
                "family_structural_leaning",
                "family_central_responsive",
                "family_inactive",
            )
            if name in target_position
        ]
        if family_names:
            metric_map = {
                "restoration_gross": "R_gross",
                "injection_gross": "I_gross",
                "rescue": "R_align",
                "induction": "I_align",
                "necessity": "necessity",
            }

            def save_family_figure(names, file_name, output_key):
                positions = [target_position[name] for name in names]
                values = {
                    key: point_raw[:, positions, endpoint_order.index(endpoint)]
                    for key, endpoint in metric_map.items()
                }
                intervals = {
                    key: (
                        raw_low[:, positions, endpoint_order.index(endpoint)],
                        raw_high[:, positions, endpoint_order.index(endpoint)],
                    )
                    for key, endpoint in metric_map.items()
                }
                family_figure, family_axes = causal_family_panels(
                    names,
                    values,
                    intervals=intervals,
                    theme=theme,
                )
                family_paths = builder.save(
                    file_name,
                    family_figure,
                    family_axes,
                    metadata={
                        "task": prepared.task.name,
                        "seed": int(prepared.grit.sc.seed),
                    },
                )
                saved[output_key] = [str(path) for path in family_paths]

            save_family_figure(
                family_names, "causal_family_endpoints", "causal_families"
            )
            # Full-size matched controls only. The `control_prefix_*` ladder is the reference for
            # the cumulative prefix curves, and enumerating it here put one bar category per
            # (control, prefix size) pair — a hundred or so on a ten-layer model.
            control_names = [
                name
                for name in target_order
                if name.startswith("control_") and not name.startswith("control_prefix_")
            ]
            if control_names:
                save_family_figure(
                    control_names,
                    "causal_matched_control_endpoints",
                    "causal_controls",
                )
            prefix_curves = {}
            for family in ("semantic_leaning", "structural_leaning"):
                names = sorted(
                    (
                        name
                        for name in target_order
                        if name.startswith(f"prefix_{family}_")
                    ),
                    key=lambda name: int(name.rsplit("_", 1)[1]),
                )
                if not names:
                    continue
                positions = [target_position[name] for name in names]
                # Average the frozen control kinds at each prefix size: the ladder is a reference
                # level for the family curve, not three separate claims.
                control_positions = [
                    [
                        target_position[control]
                        for control in target_order
                        if control.startswith("control_prefix_")
                        and control.endswith(f"_{size}")
                        and f"_{family}_" in control
                    ]
                    for size in (int(name.rsplit("_", 1)[1]) for name in names)
                ]
                endpoint_columns = {
                    "gross_total": "gross_total_for_J",
                    "gross_contrast": "gross_contrast_for_D_rel",
                    "necessity_total": "necessity_total_for_J",
                    "necessity_contrast": "necessity_contrast_for_D_rel",
                }
                record = {
                    "prefix": [int(name.rsplit("_", 1)[1]) for name in names],
                    "equivalence_half_width": float(
                        config.families.causal_equivalence_half_width
                    ),
                }
                for endpoint, calibrated_name in endpoint_columns.items():
                    column = calibrated_names.index(calibrated_name)
                    record[endpoint] = {
                        "estimate": point_calibrated[positions, column],
                        "interval": (
                            calibrated_low[positions, column],
                            calibrated_high[positions, column],
                        ),
                        "control": (
                            [
                                float(np.mean(point_calibrated[group, column]))
                                if group
                                else np.nan
                                for group in control_positions
                            ]
                            if any(control_positions)
                            else None
                        ),
                    }
                prefix_curves[family] = record
            if prefix_curves:
                fig, axes = cumulative_prefix_curves(prefix_curves, theme=theme)
                paths = builder.save(
                    "causal_cumulative_prefix_curves",
                    fig,
                    axes,
                    metadata={
                        "task": prepared.task.name,
                        "seed": int(prepared.grit.sc.seed),
                    },
                )
                saved["causal_prefixes"] = [str(path) for path in paths]
    atomic_json(prepared.output_dir / "figures.json", saved)
    return saved


def run_prepared(prepared: PreparedTask, config: MethodologyConfig) -> dict[str, Any]:
    set_strict(bool(config.strict_audits))
    key = f"{prepared.task.name}:seed{int(prepared.grit.sc.seed)}"
    if prepared.progress is not None:
        prepared.progress.update(
            task=prepared.task.name,
            train_seed=int(prepared.grit.sc.seed),
        )
        prepared.progress.start()
        prepared.progress.emit("run_start", phases=list(config.phases))
    try:
        with audit_scope(key) as scope:
            scores = None
            if {"scores", "causal", "figures"} & set(config.phases):
                context = (
                    prepared.progress.component("scores")
                    if prepared.progress is not None
                    else nullcontext()
                )
                with context:
                    score_plan = _stage_plan(prepared, config, "scores")
                    scores = run_scores(prepared, config, plan=score_plan)
            carriage = None
            if "carriage" in config.phases:
                context = (
                    prepared.progress.component("carriage")
                    if prepared.progress is not None
                    else nullcontext()
                )
                with context:
                    carriage = run_carriage(prepared, config)
            if carriage is None and "figures" in config.phases:
                carriage_plan = _stage_plan(prepared, config, "carriage")
                carriage = _cache(prepared, config, carriage_plan).load(
                    "carriage", "fields", strict=True
                )
            causal = None
            if "causal" in config.phases:
                from .validation import run_causal_validation

                context = (
                    prepared.progress.component("causal")
                    if prepared.progress is not None
                    else nullcontext()
                )
                with context:
                    causal = run_causal_validation(prepared, config, scores)
            elif "figures" in config.phases:
                from .validation import load_cached_causal_validation

                causal = load_cached_causal_validation(prepared, config, scores)
            figures = None
            if "figures" in config.phases:
                context = (
                    prepared.progress.component("figures")
                    if prepared.progress is not None
                    else nullcontext()
                )
                with context:
                    figures = make_figures(
                        prepared, config, scores, carriage, causal
                    )
    finally:
        if prepared.progress is not None:
            prepared.progress.emit("run_stop")
            prepared.progress.close()
    findings = log_summary(scope, header=f"{key} analysis audits")
    atomic_json(
        prepared.output_dir / "audits.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "task": prepared.task.name,
            "train_seed": int(prepared.grit.sc.seed),
            "strict_audits": bool(config.strict_audits),
            "phases": list(config.phases),
            "findings": findings,
            "headline_eligible": not bool(findings),
        },
    )
    return {
        "task": prepared.task.name,
        "seed": int(prepared.grit.sc.seed),
        "output_dir": str(prepared.output_dir),
        "scores": scores,
        "carriage": carriage,
        "causal": causal,
        "figures": figures,
        "audit_findings": findings,
        "headline_eligible": not bool(findings),
    }


def _release_runtime_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_worker(
    config: MethodologyConfig,
    task_name: str,
    train_seed: int,
    *,
    force_fresh_grit: bool = False,
) -> dict[str, Any]:
    """Run one isolated task/seed without writing shared task/root summaries."""

    config.validate()
    if task_name not in config.tasks:
        raise ValueError(f"worker task {task_name!r} is not registered in this run")
    if int(train_seed) not in config.seeds_for(task_name):
        raise ValueError(
            f"worker seed {int(train_seed)} is not registered for {task_name!r}"
        )
    if "figures" in config.phases:
        raise ValueError(
            "worker phases must omit figures; use the dependency-gated figures-only "
            "finalizer after every seed worker completes"
        )
    set_strict(bool(config.strict_audits))
    key = f"{task_name}:seed{int(train_seed)}"
    output_dir = config.root / task_name / f"seed_{int(train_seed)}"
    protocol_record = config.record()
    protocol_record.update(
        {
            "repository_commit": _repository_commit(),
            "execution_mode": "isolated-seed-worker",
            "worker_task": task_name,
            "worker_seed": int(train_seed),
        }
    )
    atomic_json(output_dir / "protocol.json", protocol_record)
    log(f"\n[canonical-worker] {key}")
    with audit_scope(key) as scope:
        prepared = prepare_task(
            config,
            task_name,
            int(train_seed),
            force_fresh_grit=force_fresh_grit,
        )
        result = run_prepared(prepared, config)
    findings = scope.records()
    result["audit_findings"] = findings
    result["headline_eligible"] = not bool(findings)
    atomic_json(
        output_dir / "audits.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "task": task_name,
            "train_seed": int(train_seed),
            "strict_audits": bool(config.strict_audits),
            "phases": list(config.phases),
            "findings": findings,
            "headline_eligible": not bool(findings),
        },
    )
    del prepared
    _release_runtime_memory()
    return result


def _write_run_summaries(
    config: MethodologyConfig,
    results: Mapping[str, Mapping[str, Any]],
    run_findings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    """Write the task-population and root summaries from a complete result set."""

    population: dict[str, Any] = {}
    for task_name in config.tasks:
        task_results = [
            value for value in results.values() if value["task"] == task_name
        ]
        seed_rows = []
        for value in task_results:
            scores = value.get("scores")
            if scores is None:
                continue
            coordinates = scores["coordinates"]
            row = {
                "seed": int(value["seed"]),
                "mean_raw_semantic_score": float(
                    np.mean(scores["channels"]["semantic"]["raw"])
                ),
                "mean_raw_structural_score": float(
                    np.mean(scores["channels"]["structural"]["raw"])
                ),
                "median_active_selectivity": float(
                    np.median(coordinates.selectivity[coordinates.active])
                    if coordinates.active.any()
                    else np.nan
                ),
                "active_head_fraction": float(np.mean(coordinates.active)),
            }
            diagnostics = scores.get("specialisation_diagnostics")
            if diagnostics:
                row.update(
                    {
                        "active_selectivity_p90_span": float(
                            diagnostics["active_selectivity"]["p90_span"]
                        ),
                        "equivalent_active_head_fraction": float(
                            diagnostics["classification_fraction"]["equivalent"]
                        ),
                        "semantic_tail_membership_jaccard": float(
                            diagnostics["membership"]["semantic_leaning"][
                                "mean_jaccard"
                            ]
                        ),
                        "structural_tail_membership_jaccard": float(
                            diagnostics["membership"]["structural_leaning"][
                                "mean_jaccard"
                            ]
                        ),
                    }
                )
            specialists = scores.get("specialist_classification")
            if specialists:
                row.update(
                    {
                        "semantic_candidate_pool_count": len(
                            specialists["heads"]["semantic_candidate_pool"]
                        ),
                        "structural_candidate_pool_count": len(
                            specialists["heads"]["structural_candidate_pool"]
                        ),
                        "semantic_selected_count": len(
                            specialists["heads"]["semantic_selected"]
                        ),
                        "structural_selected_count": len(
                            specialists["heads"]["structural_selected"]
                        ),
                        "semantic_confirmed_95_count": len(
                            specialists["heads"]["semantic_confirmed_95"]
                        ),
                        "structural_confirmed_95_count": len(
                            specialists["heads"]["structural_confirmed_95"]
                        ),
                        "strongest_candidate_matched_pair_count": int(
                            specialists["j_matching"]["matched_pair_count"]
                        ),
                        "strongest_candidate_mean_absolute_J_gap": float(
                            specialists["j_matching"]["mean_absolute_J_gap"]
                        ),
                    }
                )
            causal_value = value.get("causal")
            if causal_value is not None:
                association = causal_value["associations"]
                for name in (
                    "J_vs_clean_prediction_movement",
                    "J_vs_gross_total",
                    "J_vs_necessity_total",
                ):
                    if name in association:
                        row[f"rho_{name}"] = float(
                            association[name]["pooled"]["rho"]
                        )
                for name in (
                    "D_rel_vs_gross_contrast",
                    "D_rel_vs_necessity_contrast",
                ):
                    if name in association:
                        row[f"rho_{name}"] = float(
                            association[name]["pooled_active"]["rho"]
                        )
                for name, record in causal_value.get(
                    "family_interactions", {}
                ).items():
                    if isinstance(record, Mapping) and "estimate" in record:
                        row[f"interaction_{name}"] = float(record["estimate"])
                focused = causal_value.get("focused_specialists", {})
                if focused.get("status") in {"estimable", "continuous_only"}:
                    continuous = focused.get("continuous_analysis", {})
                    if continuous.get("status") == "estimable":
                        continuous_metrics = list(continuous["metric_order"])
                        statistics = list(continuous["statistic_order"])
                        continuous_interval = continuous["interval"]
                        for metric_position, metric in enumerate(
                            continuous_metrics
                        ):
                            for statistic_position, statistic in enumerate(
                                statistics
                            ):
                                row[
                                    f"continuous_{metric}_{statistic}"
                                ] = float(
                                    continuous_interval.estimate[
                                        metric_position,
                                        statistic_position,
                                    ]
                                )
                    pair_sets = list(focused["pair_set_order"])
                    metrics = list(focused["metric_order"])
                    interval = focused["interval"]
                    for pair_set in pair_sets:
                        set_position = pair_sets.index(pair_set)
                        row[f"{pair_set}_specialist_pair_count"] = int(
                            focused["pair_sets"][pair_set]["pair_count"]
                        )
                        for metric in metrics:
                            metric_position = metrics.index(metric)
                            row[
                                f"{pair_set}_{metric}_double_difference"
                            ] = float(
                                interval.estimate[
                                    set_position,
                                    metric_position,
                                    4,
                                ]
                            )
            seed_rows.append(row)
        if not seed_rows:
            if require_complete:
                raise RuntimeError(
                    f"cannot finalize {task_name}: no seed score summaries were loaded"
                )
            continue
        expected_seeds = sorted(config.seeds_for(task_name))
        observed_seeds = sorted(int(row["seed"]) for row in seed_rows)
        if require_complete and observed_seeds != expected_seeds:
            raise RuntimeError(
                f"cannot finalize {task_name}: expected seed summaries "
                f"{expected_seeds}, observed {observed_seeds}"
            )
        numeric_keys = [
            key
            for key in seed_rows[0]
            if key != "seed" and all(key in row for row in seed_rows)
        ]
        task_population: dict[str, Any] = {
            "seed_estimates": seed_rows,
            "head_alignment": "not assumed; summaries are computed within seed",
            "regime_calls": [
                {
                    "seed": int(value["seed"]),
                    "regime": value.get("causal", {})
                    .get("regime_evidence", {})
                    .get("regime", "not_available"),
                }
                for value in task_results
                if value.get("scores") is not None
            ],
        }
        if len(seed_rows) >= 3:
            matrix = np.asarray(
                [[row[key] for key in numeric_keys] for row in seed_rows],
                dtype=np.float64,
            )
            rng = np.random.default_rng(config.bootstrap.rng_seed)
            draws = np.stack(
                [
                    matrix[
                        rng.integers(0, len(matrix), size=len(matrix))
                    ].mean(axis=0)
                    for _ in range(config.bootstrap.replicates)
                ]
            )
            task_population["population_interval"] = {
                "quantity_order": numeric_keys,
                "estimate": np.mean(matrix, axis=0),
                "low": np.quantile(draws, 0.025, axis=0),
                "high": np.quantile(draws, 0.975, axis=0),
                "replicates": int(config.bootstrap.replicates),
                "rng_seed": int(config.bootstrap.rng_seed),
                "level": "training seed",
            }
        else:
            task_population["population_interval"] = None
            task_population["note"] = (
                "Fewer than three trained seeds: report seed estimates and within-seed "
                "graph intervals, not a seed-population confidence interval."
            )
        if any(value.get("causal") is not None for value in task_results):
            scatter_records = []
            for value in task_results:
                scores = value.get("scores")
                causal_value = value.get("causal")
                if scores is None or causal_value is None:
                    continue
                coordinates = scores["coordinates"]
                if not all(
                    hasattr(coordinates, name)
                    for name in (
                        "normalized_semantic",
                        "normalized_structural",
                        "selectivity",
                        "joint_sensitivity",
                    )
                ):
                    continue
                layers, heads = coordinates.joint_sensitivity.shape
                head_names = [
                    f"head_L{layer}_H{head}"
                    for layer in range(int(layers))
                    for head in range(int(heads))
                ]
                clean_ablation = causal_value.get("clean_ablation", {})
                if not all(name in clean_ablation for name in head_names):
                    continue
                scatter_records.append(
                    {
                        "seed": int(value["seed"]),
                        "semantic": coordinates.normalized_semantic.reshape(-1),
                        "structural": coordinates.normalized_structural.reshape(-1),
                        "selectivity": coordinates.selectivity.reshape(-1),
                        "joint": coordinates.joint_sensitivity.reshape(-1),
                        "clean_ablation_impact": np.asarray(
                            [
                                clean_ablation[name]["prediction_movement"]
                                for name in head_names
                            ],
                            dtype=np.float64,
                        ),
                        "layer": np.repeat(np.arange(layers), heads),
                    }
                )
            rho_population = None
            population_interval = task_population.get("population_interval")
            if population_interval is not None:
                quantity_order = list(population_interval["quantity_order"])
                if "rho_J_vs_clean_prediction_movement" in quantity_order:
                    position = quantity_order.index(
                        "rho_J_vs_clean_prediction_movement"
                    )
                    rho_population = {
                        "rho": float(population_interval["estimate"][position]),
                        "low": float(population_interval["low"][position]),
                        "high": float(population_interval["high"][position]),
                    }
            if rho_population is None:
                rho_values = np.asarray(
                    [
                        row["rho_J_vs_clean_prediction_movement"]
                        for row in seed_rows
                        if "rho_J_vs_clean_prediction_movement" in row
                    ],
                    dtype=np.float64,
                )
                if rho_values.size and np.isfinite(rho_values).any():
                    rho_population = {
                        "rho": float(np.nanmean(rho_values)),
                    }
            if scatter_records:
                theme_values = config.figure_overrides
                if task_name in theme_values and isinstance(
                    theme_values[task_name], Mapping
                ):
                    theme_values = theme_values[task_name]
                theme = FigureTheme().with_overrides(theme_values)
                composite_theme = theme.with_overrides(
                    {
                        "font_size": max(float(theme.font_size), 11.0),
                        "label_size": max(float(theme.label_size), 12.4),
                        "title_size": max(float(theme.title_size), 13.5),
                        "tick_size": max(float(theme.tick_size), 10.2),
                        "marker_size": max(float(theme.marker_size), 46.0),
                        "dpi": max(int(theme.dpi), 600),
                    }
                )
                builder = FigureBuilder(
                    config.root / task_name / "figures",
                    composite_theme,
                    common_metadata={
                        "protocol_version": PROTOCOL_VERSION,
                        "repository_commit": _repository_commit(),
                        "protocol_fingerprint": config.fingerprint,
                        "population_unit": "training seed; heads are not aligned across seeds",
                    },
                )
                fig, axes = multi_seed_score_causal_triptych(
                    scatter_records,
                    rho_statistic=rho_population,
                    clean_impact_ylim=(0.0, 16.0),
                    theme=composite_theme,
                )
                paths = builder.save(
                    "score_selectivity_clean_ablation_triptych",
                    fig,
                    axes,
                    metadata={
                        "task": task_name,
                        "seeds": [int(record["seed"]) for record in scatter_records],
                        "rho_statistic": rho_population,
                        "panels": [
                            "semantic and structural head scores",
                            "joint sensitivity and relative selectivity",
                            "joint sensitivity and head-ablation impact",
                        ],
                        "clean_ablation_impact_ylim": [0.0, 16.0],
                    },
                )
                task_population["figures"] = {
                    "score_selectivity_clean_ablation_triptych": [
                        str(path) for path in paths
                    ]
                }
        focused_population: dict[str, Any] = {
            "strongest_candidates": {},
            "confirmed_95": {},
            "population_unit": "training seed; heads are not aligned across seeds",
        }
        for pair_set in ("strongest_candidates", "confirmed_95"):
            pair_seed_rows = []
            for value in task_results:
                focused = (value.get("causal") or {}).get(
                    "focused_specialists", {}
                )
                if pair_set not in focused.get("pair_set_order", ()):
                    continue
                set_position = list(focused["pair_set_order"]).index(pair_set)
                metrics = list(focused["metric_order"])
                interval = focused["interval"]
                pair_seed_rows.append(
                    {
                        "seed": int(value["seed"]),
                        "pair_count": int(
                            focused["pair_sets"][pair_set]["pair_count"]
                        ),
                        "interaction": {
                            metric: {
                                "estimate": float(
                                    interval.estimate[
                                        set_position,
                                        metric_position,
                                        4,
                                    ]
                                ),
                                "low": float(
                                    interval.low[
                                        set_position,
                                        metric_position,
                                        4,
                                    ]
                                ),
                                "high": float(
                                    interval.high[
                                        set_position,
                                        metric_position,
                                        4,
                                    ]
                                ),
                            }
                            for metric_position, metric in enumerate(metrics)
                        },
                    }
                )
            total_pairs = sum(row["pair_count"] for row in pair_seed_rows)
            covered_seeds = len(pair_seed_rows)
            if pair_set == "strongest_candidates":
                eligible = covered_seeds >= 3
                rule = (
                    "at least 3 trained seeds, each already satisfying the "
                    "minimum 3 matched-pair seed rule"
                )
            else:
                eligible = covered_seeds >= 3 and total_pairs >= 8
                rule = "at least 8 total pairs across at least 3 trained seeds"
            pair_record: dict[str, Any] = {
                "status": "estimable" if eligible else "not_estimable",
                "eligibility_rule": rule,
                "covered_seed_count": covered_seeds,
                "total_pair_count": total_pairs,
                "seed_estimates": pair_seed_rows,
            }
            if eligible:
                metric_order = list(
                    next(
                        value["causal"]["focused_specialists"]["metric_order"]
                        for value in task_results
                        if value.get("causal")
                        and pair_set
                        in value["causal"]["focused_specialists"].get(
                            "pair_set_order", ()
                        )
                    )
                )
                matrix = np.asarray(
                    [
                        [
                            row["interaction"][metric]["estimate"]
                            for metric in metric_order
                        ]
                        for row in pair_seed_rows
                    ],
                    dtype=np.float64,
                )
                rng = np.random.default_rng(config.bootstrap.rng_seed + 17)
                draws = np.stack(
                    [
                        matrix[
                            rng.integers(
                                0,
                                len(matrix),
                                size=len(matrix),
                            )
                        ].mean(axis=0)
                        for _ in range(config.bootstrap.replicates)
                    ]
                )
                pair_record["population_interval"] = {
                    "metric_order": metric_order,
                    "estimate": np.mean(matrix, axis=0),
                    "low": np.quantile(draws, 0.025, axis=0),
                    "high": np.quantile(draws, 0.975, axis=0),
                    "replicates": int(config.bootstrap.replicates),
                    "rng_seed": int(config.bootstrap.rng_seed + 17),
                }
            else:
                pair_record["population_interval"] = None
            focused_population[pair_set] = pair_record
        task_population["focused_specialist_validation"] = focused_population
        path = config.root / task_name / "population.json"
        atomic_json(path, task_population)
        population[task_name] = {"path": str(path), **task_population}
    atomic_json(
        config.root / "audits.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "strict_audits": bool(config.strict_audits),
            "runs": run_findings,
        },
    )
    atomic_json(
        config.root / "index.json",
        {
            "runs": {
                key: {
                    "task": value["task"],
                    "seed": value["seed"],
                    "output_dir": value["output_dir"],
                    "figures": value["figures"],
                    "audit_failures": len(run_findings.get(key, ())),
                    "headline_eligible": bool(value["headline_eligible"]),
                }
                for key, value in results.items()
            },
            "population": {
                key: {
                    "path": value["path"],
                    "figures": value.get("figures", {}),
                }
                for key, value in population.items()
            },
            "audits": str(config.root / "audits.json"),
        },
    )
    failed = sorted(key for key, value in run_findings.items() if value)
    if failed:
        log(
            "[audit] soft audit failures were recorded for "
            f"{', '.join(failed)}; see {config.root / 'audits.json'}"
        )
    return population


def run_methodology(
    config: MethodologyConfig,
    *,
    force_fresh_grit: bool = False,
) -> dict[str, Any]:
    """Public non-Colab entry point."""

    config.validate()
    set_strict(bool(config.strict_audits))
    protocol_record = config.record()
    protocol_record["repository_commit"] = _repository_commit()
    atomic_json(config.root / "protocol.json", protocol_record)
    results: dict[str, Any] = {}
    run_findings: dict[str, list[dict[str, Any]]] = {}
    for task_name in config.tasks:
        for train_seed in config.seeds_for(task_name):
            key = f"{task_name}:seed{int(train_seed)}"
            log(f"\n[canonical] {key}")
            with audit_scope(key) as scope:
                prepared = prepare_task(
                    config, task_name, int(train_seed), force_fresh_grit=force_fresh_grit
                )
                results[key] = run_prepared(prepared, config)
            run_findings[key] = scope.records()
            results[key]["audit_findings"] = run_findings[key]
            results[key]["headline_eligible"] = not bool(run_findings[key])
            atomic_json(
                Path(results[key]["output_dir"]) / "audits.json",
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "task": task_name,
                    "train_seed": int(train_seed),
                    "strict_audits": bool(config.strict_audits),
                    "phases": list(config.phases),
                    "findings": run_findings[key],
                    "headline_eligible": not bool(run_findings[key]),
                },
            )
            del prepared
            _release_runtime_memory()
    _write_run_summaries(config, results, run_findings)
    return results


def render_cached_figures(
    config: MethodologyConfig,
    task_name: str,
    train_seed: int,
) -> dict[str, list[str]]:
    """Regenerate figures from CPU caches without loading a checkpoint or dataset."""

    import json
    from types import SimpleNamespace

    output_dir = config.root / task_name / f"seed_{int(train_seed)}"
    model_path = output_dir / "model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"figures-only pass requires {model_path}")
    model_record = json.loads(model_path.read_text(encoding="utf-8"))
    geometry = model_record.get("model_geometry") or {}
    if not geometry:
        raise ValueError(
            f"{model_path} has no model_geometry; rerun model preparation once"
        )

    def consolidated(stage: str, name: str, *, required: bool):
        path = output_dir / "cache" / stage / f"{name}.pt"
        if not path.exists():
            if required:
                raise FileNotFoundError(f"figures-only pass requires {path}")
            return None
        return load_cache_value_file(path)

    scores = consolidated("scores", "raw", required=True)
    carriage = consolidated("carriage", "fields", required=False)
    causal = consolidated("causal", "validation", required=False)
    prepared = PreparedTask(
        task=get_task(task_name),
        runtime=SimpleNamespace(
            sc=SimpleNamespace(seed=int(train_seed)),
            L=int(geometry["layers"]),
            H=int(geometry["heads"]),
        ),
        backend=None,
        output_dir=output_dir,
        checkpoint=Path(str(model_record.get("checkpoint", "checkpoint"))),
        checkpoint_sha=str(model_record["checkpoint_sha256"]),
        sigma=np.asarray(model_record["sigma"], dtype=np.float64),
        splits=None,
        donor_pool=None,
        progress=None,
    )
    return make_figures(prepared, config, scores, carriage, causal)


def _load_complete_figure_manifest(output_dir: Path) -> dict[str, list[str]] | None:
    """Return an atomic per-seed manifest only when every declared artifact is intact."""

    import json

    manifest_path = output_dir / "figures.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or not payload:
        return None
    figure_root = (output_dir / "figures").resolve()
    for paths in payload.values():
        if not isinstance(paths, list) or not paths:
            return None
        for value in paths:
            path = Path(str(value)).resolve()
            if not path.is_relative_to(figure_root):
                return None
            metadata = path.with_suffix(".metadata.json")
            if (
                not path.is_file()
                or path.stat().st_size <= 0
                or not metadata.is_file()
                or metadata.stat().st_size <= 0
            ):
                return None
    return {str(key): [str(value) for value in paths] for key, paths in payload.items()}


def finalize_cached_run(config: MethodologyConfig) -> dict[str, Any]:
    """Render every seed and write shared summaries exactly once from immutable caches."""

    import json

    config.validate()
    if tuple(config.phases) != ("figures",):
        raise ValueError("cached finalization requires phases=('figures',)")
    results: dict[str, Any] = {}
    run_findings: dict[str, list[dict[str, Any]]] = {}
    artifact_commits: set[str] = set()
    artifact_fingerprints: dict[str, dict[str, str]] = {}
    for task_name in config.tasks:
        for train_seed in config.seeds_for(task_name):
            key = f"{task_name}:seed{int(train_seed)}"
            output_dir = config.root / task_name / f"seed_{int(train_seed)}"
            cache_paths = {
                "scores": output_dir / "cache" / "scores" / "raw.pt",
                "carriage": output_dir / "cache" / "carriage" / "fields.pt",
                "causal": output_dir / "cache" / "causal" / "validation.pt",
            }
            artifacts = {
                name: load_cache_artifact_file(path)
                for name, path in cache_paths.items()
                if name != "carriage" or path.exists()
            }
            missing_required = [
                name
                for name in ("scores", "causal")
                if name not in artifacts
            ]
            if missing_required:
                missing_paths = ", ".join(
                    str(cache_paths[name]) for name in missing_required
                )
                raise FileNotFoundError(
                    "cached causal finalization requires score and causal caches; "
                    f"missing {missing_paths}"
                )
            artifact_fingerprints[key] = {}
            for name, artifact in artifacts.items():
                contract = artifact.metadata["contract"]
                if contract.get("task") != task_name or int(
                    contract.get("train_seed", -1)
                ) != int(train_seed):
                    raise RuntimeError(
                        f"{artifact.path} is not the registered {key} {name} cache"
                    )
                if contract.get("protocol_fingerprint") != config.fingerprint:
                    raise RuntimeError(
                        f"{artifact.path} was produced under another scientific "
                        "configuration; use the matching worker/finalizer arguments"
                    )
                artifact_commits.add(str(contract.get("repository_commit", "unknown")))
                artifact_fingerprints[key][name] = str(
                    artifact.metadata["contract_fingerprint"]
                )
            audit_path = output_dir / "audits.json"
            if not audit_path.exists():
                raise FileNotFoundError(
                    f"cached finalization requires worker audit {audit_path}"
                )
            audit_record = json.loads(audit_path.read_text(encoding="utf-8"))
            findings = list(audit_record.get("findings", ()))
            figures = (
                _load_complete_figure_manifest(output_dir)
                if config.resume and not config.force
                else None
            )
            if figures is None:
                log(f"[finalize] rendering {key}")
                figures = render_cached_figures(config, task_name, int(train_seed))
            else:
                log(f"[finalize] reusing complete figures for {key}")
            results[key] = {
                "task": task_name,
                "seed": int(train_seed),
                "output_dir": str(output_dir),
                "scores": artifacts["scores"].value,
                "carriage": (
                    artifacts["carriage"].value
                    if "carriage" in artifacts
                    else None
                ),
                "causal": artifacts["causal"].value,
                "figures": figures,
                "audit_findings": findings,
                "headline_eligible": not bool(findings),
            }
            run_findings[key] = findings
    source_commits = sorted(artifact_commits)
    protocol_record = config.record()
    protocol_record.update(
        {
            "repository_commit": _repository_commit(),
            "execution_mode": "model-free-cache-finalizer",
            # Commits are provenance, not a cache-validity boundary. A resumed run can validly
            # combine artifacts produced by multiple checkouts under one scientific contract.
            "source_repository_commit": (
                source_commits[0] if len(source_commits) == 1 else None
            ),
            "source_repository_commits": source_commits,
            "source_cache_contract_fingerprints": artifact_fingerprints,
        }
    )
    atomic_json(config.root / "protocol.json", protocol_record)
    _write_run_summaries(
        config,
        results,
        run_findings,
        require_complete=True,
    )
    return results
