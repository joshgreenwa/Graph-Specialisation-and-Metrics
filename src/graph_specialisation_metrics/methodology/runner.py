"""End-to-end canonical runner shared by every registered model backend."""

from __future__ import annotations

import dataclasses
import gc
import platform
import subprocess
import sys
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
    reportable_bin,
    trimmed_mean,
)
from .cache import CacheContract, CanonicalCache, atomic_json, checkpoint_sha256
from .carriage import (
    additive_beneficial_mass,
    beneficial_carriage,
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
from .interventions import semantic_donor_swap, structural_donor_swap
from .figures import (
    FigureBuilder,
    FigureTheme,
    HeadPlotData,
    TASK_FIGURE_MODIFIERS,
    attention_distance_profiles,
    causal_family_panels,
    causal_scatter_grid,
    cumulative_prefix_curves,
    carriage_profiles,
    distance_heatmaps,
    distance_support_profile,
    joint_selectivity_plane,
    score_distance_profiles,
    score_plane,
)
from .protocol import (
    CHANNELS,
    PROTOCOL_VERSION,
    MethodologyConfig,
    SplitManifest,
    deterministic_splits,
    stable_hash,
)
from .sampling import SemanticDonorPool, manifest_fingerprint, sample_sources
from .scores import (
    aggregate_event_scores,
    event_head_scores,
    freeze_families,
    freeze_matched_controls,
    head_coordinates,
    project_transport,
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
        eval_metric=True,
        analysis_seed=int(config.analysis_seed),
        donors=int(config.sizes.donors_per_source),
        content_adapter=spec.content_adapter,
        resume=bool(config.resume),
    )
    grit = GritHeadModel(spec, model_config).load()
    with torch.no_grad():
        first = Batch.from_data_list([grit.eval_ds[0].clone()]).to(grit.device)
        first_prediction, _ = grit.model(first)
    outputs = int(first_prediction.reshape(1, -1).shape[1])
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
        sources = sample_sources(
            int(base.num_nodes),
            int(config.sizes.sources_per_graph),
            _rng(config, prepared.task.name, prepared.grit.sc.seed, stage, graph_id, "sources"),
        )
        entry: dict[str, Any] = {"sources": tuple(int(v) for v in sources)}
        for channel in CHANNELS:
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
                    rng=_rng(
                        config,
                        prepared.task.name,
                        prepared.grit.sc.seed,
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
        common_sources = tuple(
            source
            for source in entry["sources"]
            if source in set(entry["semantic"]["sources"])
            and source in set(entry["structural"]["sources"])
        )
        if not common_sources:
            # Drop the graph from the stage rather than abort; the loss of support is recorded.
            audit_check(
                False,
                "plan.no_estimable_source",
                f"graph {graph_id} has no source estimable under both donor-swap channels; "
                f"the graph is dropped from stage {stage!r}",
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
            event_manifest_hash=_manifest_hash(plan),
            donors_per_source=int(config.sizes.donors_per_source),
            source_cap=int(config.sizes.sources_per_graph),
            bootstrap_seed=int(config.bootstrap.rng_seed),
            repository_commit=_repository_commit(),
            bootstrap_replicates=int(config.bootstrap.replicates),
        ),
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
            rng=_rng(
                config,
                prepared.task.name,
                prepared.grit.sc.seed,
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
        cached = cache.load("scores", "raw")
        if cached is not None:
            log(f"[cache] loaded canonical scores for {prepared.task.name}")
            return cached
    graph_ids = sorted(plan)
    axis = _distance_axis(prepared, graph_ids)
    output: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
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
    for channel in CHANNELS:
        graph_scores: dict[int, np.ndarray] = {}
        graph_contribution: dict[int, np.ndarray] = {}
        graph_support: dict[int, np.ndarray] = {}
        event_rows: list[dict[str, Any]] = []
        for graph_id in graph_ids:
            base = prepared.grit.eval_ds[int(graph_id)]
            clean = prepared.backend.clean_jacobians(base)
            if channel == "semantic":
                clean_transport = torch.stack(clean.capture.transport, dim=0)
                projected_clean = torch.einsum(
                    "lnhd,tlnhd->lhnt",
                    clean_transport,
                    clean.transport,
                )
                throughput_graph[graph_id] = (
                    projected_clean.square()
                    .sum(dim=-1)
                    .sqrt()
                    .sum(dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                )
            sources = plan[graph_id][channel]["sources"]
            variants, records = _rebuild_graph_events(
                prepared, config, "scores", graph_id, channel, sources
            )
            audit_check(
                [record.record() for record in records]
                == list(plan[graph_id][channel]["records"]),
                "events.deterministic_replay",
                "deterministic event replay changed its manifest; cached scores are keyed by the "
                "planned manifest hash",
                context={"stage": "scores", "graph": int(graph_id), "channel": channel},
            )
            event_capture = prepared.backend.capture(
                [base, *variants], require_grad=False, include_virtual_transport=True
            )
            transport = torch.stack(event_capture.transport, dim=1)
            delta = transport[0:1] - transport[1:]
            q = project_transport(delta, clean.transport)
            scores = event_head_scores(q).detach().cpu().numpy()
            source_ids = [record.source for record in records]
            gids = [graph_id] * len(records)
            _, one_graph, _ = aggregate_event_scores(scores, gids, source_ids)
            graph_scores[graph_id] = one_graph[graph_id]
            event_c = []
            event_o = []
            pristine = shortest_path_distances(base.edge_index, int(base.num_nodes))
            if channel == "semantic":
                attention_graph[graph_id] = _clean_attention_distance(
                    prepared, base, pristine, axis
                )
            for position, record in enumerate(records):
                distances = prepared.backend.transport_distances(
                    base, int(record.source), pristine
                )
                contribution, support = distance_event_contributions(
                    q[position : position + 1], distances, axis
                )
                event_c.append(contribution[0])
                event_o.append(support[0])
                event_rows.append(
                    {
                        **record.record(),
                        "score": scores[position],
                        "distance_contribution": contribution[0],
                        "distance_support": support[0],
                    }
                )
                observations[channel].append(
                    Observation(
                        seed=int(prepared.grit.sc.seed),
                        graph=graph_id,
                        source=int(record.source),
                        donor=int(record.draw),
                        value=scores[position],
                    )
                )
                distance_observations[channel].append(
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
                np.stack(event_c),
                np.stack(event_o),
                gids,
                source_ids,
            )
            graph_contribution[graph_id] = contribution_graph[graph_id]
            graph_support[graph_id] = support_graph[graph_id]
        raw = np.stack([graph_scores[key] for key in graph_ids]).mean(axis=0)
        heatmaps = score_heatmaps(
            graph_contribution,
            graph_support,
            reconstruction_tolerance=config.numerical.reconstruction_tolerance,
            graph_scores=graph_scores,
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
            config.bootstrap,
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

    if paired:
        output["intervals"] = nested_percentile_interval(
            paired, config.bootstrap, transform=transform
        )
    cache.save("scores", "raw", output)
    cache.save_audit("scores_manifest", plan)
    return output


def run_carriage(
    prepared: PreparedTask,
    config: MethodologyConfig,
    *,
    plan: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute F_sens and donor-wise signed path-integrated B for both channels."""

    import torch

    from ..carriage.core import project_final_states

    plan = dict(plan or _stage_plan(prepared, config, "carriage"))
    cache = _cache(prepared, config, plan)
    if config.resume and not config.force:
        cached = cache.load("carriage", "fields")
        if cached is not None:
            log(f"[cache] loaded canonical carriage for {prepared.task.name}")
            return cached
    output: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "manifest_hash": _manifest_hash(plan),
        "channels": {},
    }
    for channel in CHANNELS:
        pair_rows: list[dict[str, Any]] = []
        graph_fields: dict[int, dict[str, Any]] = {}
        capped = 0
        paths = 0
        for graph_id in sorted(plan):
            base = prepared.grit.eval_ds[int(graph_id)]
            sources = plan[graph_id][channel]["sources"]
            if not sources:
                continue
            clean = prepared.backend.clean_jacobians(base)
            variants, records = _rebuild_graph_events(
                prepared, config, "carriage", graph_id, channel, sources
            )
            audit_check(
                [record.record() for record in records]
                == list(plan[graph_id][channel]["records"]),
                "carriage.deterministic_replay",
                "deterministic carriage event replay changed its manifest; cached fields are "
                "keyed by the planned manifest hash",
                context={"stage": "carriage", "graph": int(graph_id), "channel": channel},
            )
            captured = prepared.backend.capture(
                [base, *variants], require_grad=False, include_virtual_transport=True
            )
            K = int(config.sizes.donors_per_source)
            S = len(sources)
            h_clean = captured.final_state[0]
            carriers, width = int(h_clean.shape[-2]), int(h_clean.shape[-1])
            h_event = captured.final_state[1:].reshape(S, K, carriers, width)
            delta = h_clean[None, None, :, :] - h_event
            event_f = functional_carriage_events(delta, clean.final_state)
            carrier_weights = prepared.backend.carriage_weights(base, h_clean)
            integrated = beneficial_carriage(
                h_clean,
                h_event,
                prepared.backend.loss_from_pooled(clean.capture.target.reshape(1, -1)),
                carrier_weights=carrier_weights,
                atol=config.numerical.integrated_atol,
                rtol=config.numerical.integrated_rtol,
                max_intervals=config.numerical.integrated_max_intervals,
                tolerance=max(
                    config.numerical.integrated_atol * 5,
                    config.numerical.reconstruction_tolerance,
                ),
            )
            target = clean.capture.target.reshape(1, -1)
            replay_loss = prepared.backend.loss_from_pooled(target)
            pooled_clean = project_final_states(
                h_clean.unsqueeze(0), carrier_weights
            )
            pooled_event = project_final_states(
                h_event.reshape(S * K, carriers, width),
                carrier_weights,
            )
            with torch.no_grad():
                replay_clean_loss = replay_loss(pooled_clean)
                replay_event_loss = replay_loss(pooled_event)
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
                context={"graph": int(graph_id), "channel": channel},
            )
            paths += int(integrated.converged.numel())
            capped += int((~integrated.converged).sum().item())
            F = event_f.mean(dim=1).t().detach().cpu().numpy()
            B = integrated.field.detach().cpu().numpy()
            pristine = shortest_path_distances(base.edge_index, int(base.num_nodes))
            distance = prepared.backend.carriage_distance_matrix(
                base, sources, pristine
            )
            graph_fields[graph_id] = {
                "sources": tuple(sources),
                "F_sens": F,
                "B": B,
                "distance": distance,
                "carrier_kinds": tuple(
                    prepared.backend.carriage_carrier_kind(base, carrier)
                    for carrier in range(carriers)
                ),
                "event_F_sens": event_f.detach().cpu().numpy(),
                "event_B": integrated.event_field.detach().cpu().numpy(),
                "event_loss_increase": integrated.event_loss_increase.detach().cpu().numpy(),
                "quadrature_error": integrated.quadrature_error.detach().cpu().numpy(),
                "completeness_residual": (
                    integrated.completeness_residual.detach().cpu().numpy()
                ),
                "endpoint_replay_error": endpoint_replay_error,
                "converged": integrated.converged.detach().cpu().numpy(),
            }
            event_b = integrated.event_field.detach().cpu().numpy()
            event_f_np = event_f.detach().cpu().numpy()
            for source_position, source in enumerate(sources):
                for donor in range(K):
                    for carrier in range(carriers):
                        carrier_kind = prepared.backend.carriage_carrier_kind(
                            base, carrier
                        )
                        pair_rows.append(
                            {
                                "seed": int(prepared.grit.sc.seed),
                                "graph_id": int(graph_id),
                                "source": int(source),
                                "donor": donor,
                                "carrier": carrier,
                                "distance": float(distance[carrier, source_position]),
                                "carrier_kind": carrier_kind,
                                "F_sens": float(event_f_np[source_position, donor, carrier]),
                                "B": float(event_b[source_position, donor, carrier]),
                                "channel": channel,
                            }
                        )
        capped_fraction = float(capped / paths) if paths else 0.0
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
                    int(prepared.grit.eval_ds[int(graph_id)].num_nodes)
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
        }
        additive_observations = []
        for graph_id, field in graph_fields.items():
            n = int(prepared.grit.eval_ds[int(graph_id)].num_nodes)
            event_b = np.asarray(field["event_B"])
            distance = np.asarray(field["distance"])
            for source_position, source in enumerate(field["sources"]):
                for donor in range(event_b.shape[1]):
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
        additive_interval = nested_percentile_interval(
            additive_observations, config.bootstrap
        )
        bin_count = len(bins)
        output["channels"][channel] = {
            "graph_fields": graph_fields,
            "pairs": pair_rows,
            "capped_paths": capped,
            "total_paths": paths,
            "capped_fraction": capped_fraction,
            "distance_bins": bins,
            "far_thresholds": far_thresholds,
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
            config.bootstrap,
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
        selected = [row for row in rows if select(row)]
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

        def pair_graph_reduce(values):
            graph_pair_means = values[:, 0] / values[:, 1]
            return trimmed_mean(
                graph_pair_means, config.bootstrap.trim_fraction, axis=0
            )

        interval = nested_percentile_interval(
            observations,
            config.bootstrap,
            graph_reduce=pair_graph_reduce,
        )
        estimates.append(float(interval.estimate))
        lows.append(float(interval.low))
        highs.append(float(interval.high))
    return labels, np.asarray(estimates), (np.asarray(lows), np.asarray(highs))


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
    fig, axes = joint_selectivity_plane(plot_data, title=prepared.task.title, theme=theme)
    paths = builder.save(
        "selectivity_vs_joint_sensitivity",
        fig,
        axes,
        metadata={"task": prepared.task.name, "seed": int(prepared.grit.sc.seed)},
    )
    saved["coordinate_plane"] = [str(path) for path in paths]
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
            rows = carriage["channels"][channel]["pairs"]
            labels, functional, functional_interval = _carriage_profile(
                rows, "F_sens", config
            )
            _, beneficial, beneficial_interval = _carriage_profile(rows, "B", config)
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
                    "beneficial_sign": "positive-is-beneficial",
                },
            )
            saved[f"{channel}_carriage"] = [str(path) for path in paths]
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
                gross_index = endpoint_order.index("G_c")
                necessity_index = endpoint_order.index("necessity")
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
                control_curve = (
                    {
                        endpoint: {
                            channel: [
                                float(np.mean(point_raw[index, group, metric]))
                                if group
                                else np.nan
                                for group in control_positions
                            ]
                            for channel, index in (("semantic", 0), ("structural", 1))
                        }
                        for endpoint, metric in (
                            ("gross", gross_index),
                            ("necessity", necessity_index),
                        )
                    }
                    if any(control_positions)
                    else None
                )
                prefix_curves[family] = {
                    "control": control_curve,
                    "prefix": [int(name.rsplit("_", 1)[1]) for name in names],
                    "gross": {
                        "semantic": point_raw[0, positions, gross_index],
                        "structural": point_raw[1, positions, gross_index],
                    },
                    "gross_interval": {
                        "semantic": (
                            raw_low[0, positions, gross_index],
                            raw_high[0, positions, gross_index],
                        ),
                        "structural": (
                            raw_low[1, positions, gross_index],
                            raw_high[1, positions, gross_index],
                        ),
                    },
                    "necessity": {
                        "semantic": point_raw[0, positions, necessity_index],
                        "structural": point_raw[1, positions, necessity_index],
                    },
                    "necessity_interval": {
                        "semantic": (
                            raw_low[0, positions, necessity_index],
                            raw_high[0, positions, necessity_index],
                        ),
                        "structural": (
                            raw_low[1, positions, necessity_index],
                            raw_high[1, positions, necessity_index],
                        ),
                    },
                }
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
    with audit_scope(key) as scope:
        score_plan = _stage_plan(prepared, config, "scores")
        scores = run_scores(prepared, config, plan=score_plan) if (
            {"scores", "causal", "figures"} & set(config.phases)
        ) else None
        carriage = (
            run_carriage(prepared, config)
            if "carriage" in config.phases or "figures" in config.phases
            else None
        )
        causal = None
        if "causal" in config.phases:
            from .validation import run_causal_validation

            causal = run_causal_validation(prepared, config, scores)
        elif "figures" in config.phases:
            from .validation import load_cached_causal_validation

            causal = load_cached_causal_validation(prepared, config, scores)
        figures = (
            make_figures(prepared, config, scores, carriage, causal)
            if "figures" in config.phases
            else None
        )
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
    }


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
            del prepared
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
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
                    np.median(
                        coordinates.selectivity[coordinates.active]
                    )
                    if coordinates.active.any()
                    else np.nan
                ),
                "active_head_fraction": float(np.mean(coordinates.active)),
            }
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
            seed_rows.append(row)
        if not seed_rows:
            continue
        numeric_keys = [
            key
            for key in seed_rows[0]
            if key != "seed" and all(key in row for row in seed_rows)
        ]
        task_population: dict[str, Any] = {
            "seed_estimates": seed_rows,
            "head_alignment": "not assumed; summaries are computed within seed",
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
                }
                for key, value in results.items()
            },
            "population": {
                key: {"path": value["path"]} for key, value in population.items()
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
    return results
