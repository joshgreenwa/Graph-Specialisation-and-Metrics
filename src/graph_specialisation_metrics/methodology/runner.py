"""End-to-end canonical runner for all registered GRIT variants."""

from __future__ import annotations

import dataclasses
import gc
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..carriage import env
from ..carriage.env import log
from ..carriage.tasks import resolve_dataset_dir
from ..specialisation.model import GritHeadModel, SpecConfig
from .backend import CanonicalGritBackend
from .bootstrap import Observation, nested_percentile_interval, trimmed_mean
from .cache import CacheContract, CanonicalCache, atomic_json, checkpoint_sha256
from .carriage import (
    additive_beneficial_mass,
    beneficial_carriage,
    functional_carriage_events,
)
from .distance import (
    DistanceAxis,
    adaptive_distance_bins,
    aggregate_distance_events,
    distance_event_contributions,
    score_heatmaps,
    shortest_path_distances,
)
from .events import build_channel_events
from .interventions import semantic_donor_swap, structural_donor_swap
from .figures import (
    FigureBuilder,
    FigureTheme,
    HeadPlotData,
    TASK_FIGURE_MODIFIERS,
    causal_family_panels,
    causal_scatter_grid,
    carriage_profiles,
    distance_heatmaps,
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
    grit: GritHeadModel
    backend: CanonicalGritBackend
    output_dir: Path
    checkpoint: Path
    checkpoint_sha: str
    sigma: np.ndarray
    splits: SplitManifest
    donor_pool: SemanticDonorPool


def _model_audits(
    grit: GritHeadModel,
    backend: CanonicalGritBackend,
    task: CanonicalTask,
    config: MethodologyConfig,
) -> dict[str, Any]:
    """Mandatory model, gradient, attention, and intervention no-op checks."""

    import torch
    from torch_geometric.data import Batch

    base = grit.eval_ds[0]
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
    if batch_error > config.numerical.batch_invariance_tolerance:
        raise RuntimeError(
            f"batch invariance error {batch_error:.3e} exceeds "
            f"{config.numerical.batch_invariance_tolerance:.3e}"
        )

    # Attention rows are receiver-normalized on every supported layer.
    batch = Batch.from_data_list([base.clone()]).to(grit.device)
    attention_capture = grit.capture(
        batch,
        want_grad=False,
        want_attn=True,
        include_virtual_transport=True,
    )
    attention_error = 0.0
    edge_index = attention_capture["edge_index"].long()
    receiver = edge_index[1]
    for attention in attention_capture["attn"]:
        mass = torch.zeros(
            int(attention_capture["wV"][0].shape[0]),
            int(grit.H),
            dtype=attention.dtype,
            device=attention.device,
        )
        mass.index_add_(0, receiver, attention)
        valid = torch.unique(receiver)
        attention_error = max(
            attention_error,
            float(torch.max(torch.abs(mass[valid] - 1.0)).item()),
        )
    if attention_error > config.numerical.attention_tolerance:
        raise RuntimeError(
            f"attention normalization error {attention_error:.3e} exceeds "
            f"{config.numerical.attention_tolerance:.3e}"
        )

    rows = task.grit.content_adapter.rows(base)
    semantic_noop = semantic_donor_swap(
        base, 0, rows[0], adapter=task.grit.content_adapter
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
    if no_op_error > config.numerical.no_op_tolerance:
        raise RuntimeError(
            f"declared no-op donor response {no_op_error:.3e} exceeds "
            f"{config.numerical.no_op_tolerance:.3e}"
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


def prepare_task(
    config: MethodologyConfig,
    task_name: str,
    train_seed: int,
    *,
    force_fresh_grit: bool = False,
) -> PreparedTask:
    """Rebuild the registered environment and load a cached training checkpoint read-only."""

    import torch
    from torch_geometric.data import Batch

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
    }
    canonical_override_names = set(CanonicalTask.__dataclass_fields__) - {"name", "grit"}
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
    spec = task.grit
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
    audit_checks = _model_audits(grit, backend, task, config)
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
        "task": task_name,
        "title": task.title,
        "train_seed": int(train_seed),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(epoch),
        "checkpoint_sha256": digest,
        "output_representation": task.output.representation,
        "sigma": sigma.tolist(),
        "task_adapter_version": task.adapter_version,
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
            raise RuntimeError(
                f"graph {graph_id} has no source estimable under both donor-swap channels"
            )
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
    if prepared.task.virtual_node and "virtual" not in labels:
        labels.append("virtual")
    return DistanceAxis(tuple(labels))


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
    for channel in CHANNELS:
        graph_scores: dict[int, np.ndarray] = {}
        graph_contribution: dict[int, np.ndarray] = {}
        graph_support: dict[int, np.ndarray] = {}
        event_rows: list[dict[str, Any]] = []
        for graph_id in graph_ids:
            base = prepared.grit.eval_ds[int(graph_id)]
            clean = prepared.backend.clean_jacobians(base)
            if channel == "semantic":
                throughput_graph[graph_id] = torch.stack(
                    [
                        torch.linalg.vector_norm(layer.detach(), dim=-1).sum(dim=0)
                        for layer in clean.capture.transport
                    ]
                ).cpu().numpy()
            sources = plan[graph_id][channel]["sources"]
            variants, records = _rebuild_graph_events(
                prepared, config, "scores", graph_id, channel, sources
            )
            if [record.record() for record in records] != list(
                plan[graph_id][channel]["records"]
            ):
                raise RuntimeError("deterministic event replay changed its manifest")
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
            for position, record in enumerate(records):
                distances: list[Any] = list(pristine[int(record.source), :])
                if q.shape[3] == int(base.num_nodes) + 1:
                    distances.append("virtual")
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
            "events": event_rows,
        }
        def graph_distance_reduce(rows):
            # rows [G,2,L,H,D]. Divide support within graph, then average graphs.
            C = rows[:, 0]
            O = rows[:, 1]
            ratio = np.full_like(C, np.nan)
            np.divide(C, O, out=ratio, where=O > 0)
            cells_c = C.mean(axis=0).sum(axis=1)
            cells_r = np.nansum(np.nanmean(ratio, axis=0), axis=1)
            profile_c = cells_c.sum(axis=0, keepdims=True)
            profile_r = cells_r.sum(axis=0, keepdims=True)
            return np.stack(
                (
                    np.concatenate((cells_c, profile_c), axis=0),
                    np.concatenate((cells_r, profile_r), axis=0),
                )
            )

        output["channels"][channel]["distance_intervals"] = nested_percentile_interval(
            distance_observations[channel],
            config.bootstrap,
            graph_reduce=graph_distance_reduce,
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
                "per_opportunity": np.nanmean(
                    np.stack(
                        [normalized_graph[key] for key in sorted(normalized_graph)]
                    ),
                    axis=0,
                ),
            }
        channel_output["family_distance_profiles"] = family_profiles

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

    from ..carriage.core import pool_final_states

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
            if [record.record() for record in records] != list(
                plan[graph_id][channel]["records"]
            ):
                raise RuntimeError("deterministic carriage event replay changed its manifest")
            captured = prepared.backend.capture(
                [base, *variants], require_grad=False, include_virtual_transport=True
            )
            K = int(config.sizes.donors_per_source)
            S = len(sources)
            h_clean = captured.final_state[0]
            h_event = captured.final_state[1:].reshape(
                S, K, int(base.num_nodes), prepared.grit.dim_h
            )
            delta = h_clean[None, None, :, :] - h_event
            event_f = functional_carriage_events(delta, clean.final_state)
            integrated = beneficial_carriage(
                h_clean,
                h_event,
                prepared.backend.loss_from_pooled(clean.capture.target.reshape(1, -1)),
                pooling=str(prepared.grit.cfg.model.graph_pooling),
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
            pooled_clean = pool_final_states(h_clean.unsqueeze(0), str(
                prepared.grit.cfg.model.graph_pooling
            ))
            pooled_event = pool_final_states(
                h_event.reshape(S * K, int(base.num_nodes), prepared.grit.dim_h),
                str(prepared.grit.cfg.model.graph_pooling),
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
            if endpoint_replay_error > config.numerical.reconstruction_tolerance:
                raise RuntimeError(
                    f"pooling-to-readout replay error {endpoint_replay_error:.3e} exceeds "
                    f"{config.numerical.reconstruction_tolerance:.3e}"
                )
            paths += int(integrated.converged.numel())
            capped += int((~integrated.converged).sum().item())
            F = event_f.mean(dim=1).t().detach().cpu().numpy()
            B = integrated.field.detach().cpu().numpy()
            pristine = shortest_path_distances(base.edge_index, int(base.num_nodes))
            distance = pristine[np.asarray(sources), :].T
            graph_fields[graph_id] = {
                "sources": tuple(sources),
                "F_sens": F,
                "B": B,
                "distance": distance,
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
                    for carrier in range(int(base.num_nodes)):
                        pair_rows.append(
                            {
                                "seed": int(prepared.grit.sc.seed),
                                "graph_id": int(graph_id),
                                "source": int(source),
                                "donor": donor,
                                "carrier": carrier,
                                "distance": float(distance[carrier, source_position]),
                                "F_sens": float(event_f_np[source_position, donor, carrier]),
                                "B": float(event_b[source_position, donor, carrier]),
                                "channel": channel,
                            }
                        )
        capped_fraction = float(capped / paths) if paths else 0.0
        if capped_fraction > config.numerical.integrated_unconverged_fraction:
            raise RuntimeError(
                f"{channel} Beneficial carriage capped-path fraction {capped_fraction:.3%} "
                f"exceeds {config.numerical.integrated_unconverged_fraction:.3%}"
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
            graph_id: additive_beneficial_mass(
                field["B"],
                field["distance"],
                bins=bins,
                far_thresholds=far_thresholds,
            )
            for graph_id, field in graph_fields.items()
        }
        output["channels"][channel] = {
            "graph_fields": graph_fields,
            "pairs": pair_rows,
            "capped_paths": capped,
            "total_paths": paths,
            "capped_fraction": capped_fraction,
            "distance_bins": bins,
            "far_thresholds": far_thresholds,
            "S_B": np.stack(
                [additive[key]["S_B"] for key in sorted(additive)]
            ).mean(axis=0),
            "B_far": np.stack(
                [additive[key]["B_far"] for key in sorted(additive)]
            ).mean(axis=0),
            "additive_graph": additive,
        }
    cache.save("carriage", "fields", output)
    cache.save_audit("carriage_manifest", plan)
    return output


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
    estimates, lows, highs = [], [], []
    for lo, hi in bins:
        selected = [
            row
            for row in rows
            if np.isfinite(row["distance"]) and lo <= int(row["distance"]) <= hi
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
    for channel in CHANNELS:
        channel_scores = scores["channels"][channel]
        fig, axes = distance_heatmaps(
            channel_scores["heatmap_exact"],
            channel_scores["heatmap_per_opportunity"],
            scores["axis"],
            channel=channel,
            title=prepared.task.title,
            theme=theme,
        )
        paths = builder.save(
            f"{channel}_score_distance_heatmaps",
            fig,
            axes,
            metadata={
                "task": prepared.task.name,
                "seed": int(prepared.grit.sc.seed),
                "channel": channel,
            },
        )
        saved[f"{channel}_distance"] = [str(path) for path in paths]
        interval = channel_scores["distance_intervals"]
        profile_row = int(prepared.grit.L)
        fig, axes = score_distance_profiles(
            scores["axis"],
            interval.estimate[0, profile_row],
            interval.estimate[1, profile_row],
            exact_interval=(
                interval.low[0, profile_row],
                interval.high[0, profile_row],
            ),
            per_opportunity_interval=(
                interval.low[1, profile_row],
                interval.high[1, profile_row],
            ),
            channel=channel,
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
            },
        )
        saved[f"{channel}_distance_profile"] = [str(path) for path in paths]
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
        calibrated_low = interval_object.low[raw_size:].reshape(point_calibrated.shape)
        calibrated_high = interval_object.high[raw_size:].reshape(point_calibrated.shape)
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
            family_positions = [target_position[name] for name in family_names]
            metric_map = {
                "restoration_gross": "R_gross",
                "injection_gross": "I_gross",
                "rescue": "R_align",
                "induction": "I_align",
                "necessity": "necessity",
            }
            family_values = {
                key: point_raw[:, family_positions, endpoint_order.index(endpoint)]
                for key, endpoint in metric_map.items()
            }
            family_intervals = {
                key: (
                    raw_low[:, family_positions, endpoint_order.index(endpoint)],
                    raw_high[:, family_positions, endpoint_order.index(endpoint)],
                )
                for key, endpoint in metric_map.items()
            }
            fig, axes = causal_family_panels(
                family_names,
                family_values,
                intervals=family_intervals,
                theme=theme,
            )
            paths = builder.save(
                "causal_family_endpoints",
                fig,
                axes,
                metadata={
                    "task": prepared.task.name,
                    "seed": int(prepared.grit.sc.seed),
                    "controls": list(scores.get("matched_controls", {})),
                },
            )
            saved["causal_families"] = [str(path) for path in paths]
    atomic_json(prepared.output_dir / "figures.json", saved)
    return saved


def run_prepared(prepared: PreparedTask, config: MethodologyConfig) -> dict[str, Any]:
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
    figures = (
        make_figures(prepared, config, scores, carriage, causal)
        if "figures" in config.phases
        else None
    )
    return {
        "task": prepared.task.name,
        "seed": int(prepared.grit.sc.seed),
        "output_dir": str(prepared.output_dir),
        "scores": scores,
        "carriage": carriage,
        "causal": causal,
        "figures": figures,
    }


def run_methodology(
    config: MethodologyConfig,
    *,
    force_fresh_grit: bool = False,
) -> dict[str, Any]:
    """Public non-Colab entry point."""

    config.validate()
    atomic_json(config.root / "protocol.json", config.record())
    results: dict[str, Any] = {}
    for task_name in config.tasks:
        for train_seed in config.train_seeds:
            key = f"{task_name}:seed{int(train_seed)}"
            log(f"\n[canonical] {key}")
            prepared = prepare_task(
                config, task_name, int(train_seed), force_fresh_grit=force_fresh_grit
            )
            results[key] = run_prepared(prepared, config)
            del prepared
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    atomic_json(
        config.root / "index.json",
        {
            key: {
                "task": value["task"],
                "seed": value["seed"],
                "output_dir": value["output_dir"],
                "figures": value["figures"],
            }
            for key, value in results.items()
        },
    )
    return results
