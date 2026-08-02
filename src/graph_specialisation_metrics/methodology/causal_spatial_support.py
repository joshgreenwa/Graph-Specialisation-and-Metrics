"""Head- and distance-resolved finite mediation for dense molecular GRIT models.

The experiment asks where, inside a causally relevant specialist head, a finite
semantic or structural intervention becomes output-relevant.  It keeps three
objects separate:

``A(h,d)``
    clean, source-conditioned direct attention access from the intervened node;
``S_x(h,d)``
    canonical discovery-split internal response mass from the immutable score cache;
``M_x(h,d)``
    held-out symmetric injection/restoration mediation when only carrier shell
    ``d`` of head ``h`` (or a frozen family) is patched.

``M`` describes the realised mechanism of the trained model.  It is not a
minimal task-requirement or retraining estimand.  Individual and joint family
patches additionally diagnose overlap among realised head pathways.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .distance import shortest_path_distances
from .grit_figure_data import (
    CanonicalHeadMetrics,
    SupplementalCache,
    build_verified_grit_figure_runtime,
    load_canonical_model_record,
    load_canonical_score_artifact,
    methodology_config_from_record,
)
PROTOCOL_VERSION = "causal-spatial-support-v1"
CHANNELS = ("semantic", "structural")
PRIMARY_ROLES = ("semantic", "structural", "generalist")
CONTROL_ROLES = ("semantic_control", "structural_control")
ROLE_LABELS = {
    "semantic": "Semantic specialists",
    "structural": "Structural specialists",
    "generalist": "High-J generalists",
    "semantic_control": "Semantic matched controls",
    "structural_control": "Structural matched controls",
}
TASK_LABELS = {"zinc": "ZINC", "qm9_gap_dense": "QM9"}
MEASURE_COLOURS = {"A": "#8C8C8C", "S": "#0072B2", "M": "#CC79A7"}
ROLE_COLOURS = {
    "semantic": "#0072B2",
    "structural": "#009E73",
    "generalist": "#CC79A7",
    "semantic_control": "#7A7A7A",
    "structural_control": "#B0B0B0",
}


@dataclasses.dataclass(frozen=True)
class Config:
    canonical_root: Path
    output_dir: Path
    tasks: tuple[str, ...] = ("zinc", "qm9_gap_dense")
    train_seed: int = 42
    phase: str = "all"
    graphs: int = 8
    sources_per_graph: int = 2
    donors_per_source: int = 1
    heads_per_family: int = 3
    long_range_radius: int = 2
    bootstrap_replicates: int = 2_000
    analysis_seed: int = 72_019
    accelerator: str = "cuda:0"
    force: bool = False

    def validate(self) -> None:
        if self.phase not in {"all", "measure", "figures"}:
            raise ValueError("phase must be 'all', 'measure', or 'figures'")
        if not self.tasks:
            raise ValueError("at least one task is required")
        for value, name in (
            (self.graphs, "graphs"),
            (self.sources_per_graph, "sources_per_graph"),
            (self.donors_per_source, "donors_per_source"),
            (self.heads_per_family, "heads_per_family"),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")


def _as_array(value: Any, *, dtype=float) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _normalise_profile(values: Any) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    total = float(values.sum())
    return values / total if total > 0 else np.full_like(values, np.nan)


def _head_label(head: tuple[int, int]) -> str:
    return f"L{int(head[0]) + 1} H{int(head[1]) + 1}"


def _head_target(head: tuple[int, int]) -> str:
    return f"head:{int(head[0])}:{int(head[1])}"


def select_role_heads(
    scores: Mapping[str, Any],
    metrics: CanonicalHeadMetrics,
    *,
    count: int,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Freeze strongest tails, high-J centre, and registered matched controls."""

    families = scores.get("families", {})
    controls = scores.get("matched_controls", {})

    def take(name: str) -> tuple[tuple[int, int], ...]:
        values = tuple(tuple(int(v) for v in head) for head in families.get(name, ()))
        return values[: min(int(count), len(values))]

    semantic = take("semantic_leaning")
    structural = take("structural_leaning")
    generalist = take("central_responsive")
    if len(generalist) < int(count):
        eligible = [
            (layer, head)
            for layer in range(metrics.num_layers)
            for head in range(metrics.num_heads)
            if metrics.active[layer, head]
        ]
        eligible.sort(
            key=lambda item: (
                abs(float(metrics.selectivity[item])),
                -float(metrics.joint_sensitivity[item]),
                item,
            )
        )
        generalist = tuple(eligible[: int(count)])

    def control(name: str, target_size: int) -> tuple[tuple[int, int], ...]:
        values = tuple(tuple(int(v) for v in head) for head in controls.get(name, ()))
        return values[: min(target_size, len(values))]

    role_heads = {
        "semantic": semantic,
        "structural": structural,
        "generalist": generalist,
        "semantic_control": control(
            "semantic_leaning_central_control", len(semantic)
        ),
        "structural_control": control(
            "structural_leaning_central_control", len(structural)
        ),
    }
    missing = [name for name in PRIMARY_ROLES if not role_heads[name]]
    if missing:
        raise ValueError(f"canonical score cache cannot freeze role families: {missing}")
    return {name: heads for name, heads in role_heads.items() if heads}


def _measurement_contract(
    config: Config,
    task: str,
    artifact: Any,
    role_heads: Mapping[str, Sequence[tuple[int, int]]],
) -> dict[str, Any]:
    canonical = artifact.metadata["contract"]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task": task,
        "train_seed": int(config.train_seed),
        "canonical_score_sha256": artifact.file_sha256,
        "canonical_contract_fingerprint": artifact.metadata["contract_fingerprint"],
        "checkpoint_sha256": canonical["checkpoint_sha256"],
        "split_fingerprint": canonical["split_fingerprint"],
        "model_geometry": canonical["model_geometry"],
        "graphs": int(config.graphs),
        "sources_per_graph": int(config.sources_per_graph),
        "donors_per_source": int(config.donors_per_source),
        "heads_per_family": int(config.heads_per_family),
        "long_range_radius": int(config.long_range_radius),
        "role_heads": {
            name: [list(head) for head in heads] for name, heads in role_heads.items()
        },
        "estimand": "matched-symmetric-shell-specific-injection-restoration",
        "carrier_distance": "pristine-source-to-carrier-SPD",
        "attention_access": "clean-direct-source-to-receiver-attention",
    }


def _source_attention_rows(
    prepared: Any,
    base: Any,
    graph_id: int,
    sources: Sequence[int],
    role_heads: Mapping[str, Sequence[tuple[int, int]]],
    pristine: np.ndarray,
) -> list[dict[str, Any]]:
    """Direct clean attention from each source to receivers, resolved by SPD."""

    from torch_geometric.data import Batch

    batch = Batch.from_data_list([base.clone()]).to(prepared.runtime.device)
    captured = prepared.runtime.capture(
        batch,
        want_grad=False,
        want_attn=True,
        include_virtual_transport=False,
    )
    edge = captured["edge_index"].detach().cpu().numpy()
    sender, receiver = edge[0], edge[1]
    rows: list[dict[str, Any]] = []
    for source in sources:
        source = int(source)
        selected_edges = np.flatnonzero(sender == source)
        if not len(selected_edges):
            continue
        edge_distances = pristine[source, receiver[selected_edges]].astype(int)
        for role, heads in role_heads.items():
            for distance in sorted(set(int(value) for value in edge_distances)):
                mask = selected_edges[edge_distances == distance]
                values = []
                for layer, head in heads:
                    attention = captured["attn"][int(layer)].detach().cpu().numpy()
                    values.append(float(attention[mask, int(head)].sum()))
                rows.append(
                    {
                        "graph": int(graph_id),
                        "source": source,
                        "role": role,
                        "distance": int(distance),
                        "A": float(np.mean(values)),
                    }
                )
    return rows


def _selected_events(
    prepared: Any,
    protocol_config: Any,
    plan: Mapping[int, Mapping[str, Any]],
    graph_id: int,
    channel: str,
    config: Config,
) -> tuple[list[Any], list[Any]]:
    from .runner import _rebuild_graph_events

    sources = tuple(
        int(value)
        for value in plan[int(graph_id)][channel]["sources"][: int(config.sources_per_graph)]
    )
    variants, records = _rebuild_graph_events(
        prepared,
        protocol_config,
        "causal",
        int(graph_id),
        channel,
        sources,
    )
    keep = [
        index
        for index, record in enumerate(records)
        if int(record.draw) < int(config.donors_per_source)
    ]
    return [variants[index] for index in keep], [records[index] for index in keep]


def _patch_targets(
    role_heads: Mapping[str, Sequence[tuple[int, int]]]
) -> dict[str, tuple[tuple[int, int], ...]]:
    unique = sorted({tuple(head) for heads in role_heads.values() for head in heads})
    targets = {_head_target(head): (head,) for head in unique}
    targets.update({f"family:{role}": tuple(heads) for role, heads in role_heads.items()})
    return targets


def _event_conditions(
    targets: Mapping[str, Sequence[tuple[int, int]]],
    pristine: np.ndarray,
    source: int,
) -> list[dict[str, Any]]:
    distances = pristine[int(source)]
    conditions: list[dict[str, Any]] = []
    for distance in sorted(set(int(value) for value in distances[np.isfinite(distances)])):
        nodes = tuple(int(value) for value in np.flatnonzero(distances == distance))
        for target, family in targets.items():
            conditions.append(
                {
                    "target": target,
                    "family": tuple(family),
                    "distance": int(distance),
                    "nodes": nodes,
                }
            )
    all_nodes = tuple(range(int(pristine.shape[0])))
    for target, family in targets.items():
        conditions.append(
            {
                "target": target,
                "family": tuple(family),
                "distance": "all",
                "nodes": all_nodes,
            }
        )
    return conditions


def _measure_graph_channel(
    prepared: Any,
    protocol_config: Any,
    plan: Mapping[int, Mapping[str, Any]],
    graph_id: int,
    channel: str,
    role_heads: Mapping[str, Sequence[tuple[int, int]]],
    config: Config,
) -> dict[str, Any]:
    base = prepared.runtime.eval_ds[int(graph_id)]
    pristine = shortest_path_distances(base.edge_index, int(base.num_nodes))
    variants, records = _selected_events(
        prepared,
        protocol_config,
        plan,
        int(graph_id),
        channel,
        config,
    )
    if not records:
        return {
            "status": "not_estimable",
            "warning": f"graph {graph_id} has no selected {channel} events",
            "attention_rows": [],
            "mediation_rows": [],
        }
    sources = sorted({int(record.source) for record in records})
    attention_rows = _source_attention_rows(
        prepared,
        base,
        int(graph_id),
        sources,
        role_heads,
        pristine,
    )
    capture = prepared.backend.capture(
        [base, *variants], require_grad=False, include_virtual_transport=False
    )
    z = capture.z.detach().cpu().numpy().reshape(len(variants) + 1, -1)
    if z.shape[1] != 1:
        return {
            "status": "not_estimable",
            "warning": f"{prepared.task.name} has {z.shape[1]} transformed outputs; expected one",
            "attention_rows": attention_rows,
            "mediation_rows": [],
        }
    targets = _patch_targets(role_heads)
    mediation_rows: list[dict[str, Any]] = []
    for event_index, (variant, record) in enumerate(zip(variants, records, strict=True)):
        conditions = _event_conditions(targets, pristine, int(record.source))
        count = len(conditions)
        donor_replacements = prepared.backend.replacement_batch(
            capture, [event_index + 1] * count
        )
        clean_replacements = prepared.backend.replacement_batch(capture, [0] * count)
        _, z_inject, _ = prepared.backend.patch_masked_many(
            [base] * count,
            donor_replacements,
            conditions,
        )
        _, z_restore, _ = prepared.backend.patch_masked_many(
            [variant] * count,
            clean_replacements,
            conditions,
        )
        injection = np.abs(z_inject.detach().cpu().numpy().reshape(-1) - z[0, 0])
        restoration = np.abs(z[event_index + 1, 0] - z_restore.detach().cpu().numpy().reshape(-1))
        mediated = 0.5 * (injection + restoration)
        event_effect = float(abs(z[event_index + 1, 0] - z[0, 0]))
        for condition, value, inject, restore in zip(
            conditions, mediated, injection, restoration, strict=True
        ):
            mediation_rows.append(
                {
                    "graph": int(graph_id),
                    "source": int(record.source),
                    "donor": int(record.draw),
                    "channel": channel,
                    "target": str(condition["target"]),
                    "distance": condition["distance"],
                    "M": float(value),
                    "injection": float(inject),
                    "restoration": float(restore),
                    "event_effect": event_effect,
                }
            )
    return {
        "status": "complete",
        "graph": int(graph_id),
        "channel": channel,
        "sources": sources,
        "events": len(records),
        "attention_rows": attention_rows,
        "mediation_rows": mediation_rows,
    }


def _safe_measure_graph_channel(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Turn a failed shard into an auditable soft result, preserving other shards."""

    graph_id = int(kwargs.get("graph_id", args[3] if len(args) > 3 else -1))
    channel = str(kwargs.get("channel", args[4] if len(args) > 4 else "unknown"))
    try:
        return _measure_graph_channel(*args, **kwargs)
    except Exception as error:
        warning = (
            f"graph {graph_id} {channel}: {type(error).__name__}: {error}"
        )
        print(f"[causal-spatial-support:warning] {warning}")
        return {
            "status": "not_estimable",
            "graph": graph_id,
            "channel": channel,
            "warning": warning,
            "attention_rows": [],
            "mediation_rows": [],
        }


def _sufficient_profile_rows(
    scores: Mapping[str, Any],
    role_heads: Mapping[str, Sequence[tuple[int, int]]],
) -> list[dict[str, Any]]:
    labels = tuple(scores.get("axis", ()))
    rows: list[dict[str, Any]] = []
    for channel in CHANNELS:
        contributions = scores["channels"][channel]["graph_distance_contribution"]
        for graph_id, contribution in contributions.items():
            value = _as_array(contribution)
            for role, heads in role_heads.items():
                profile = np.sum(
                    np.stack([value[int(layer), int(head)] for layer, head in heads]),
                    axis=0,
                )
                for position, label in enumerate(labels):
                    if isinstance(label, (int, np.integer)) or str(label).isdigit():
                        rows.append(
                            {
                                "graph": int(graph_id),
                                "channel": channel,
                                "role": role,
                                "distance": int(label),
                                "S": float(profile[position]),
                            }
                        )
    return rows


def _graph_profile_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    distances: Sequence[int],
    target: str | None = None,
    role: str | None = None,
    channel: str | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    selected = [
        row
        for row in rows
        if (target is None or row.get("target") == target)
        and (role is None or row.get("role") == role)
        and (channel is None or row.get("channel") == channel)
        and isinstance(row.get("distance"), (int, np.integer))
    ]
    graphs = sorted({int(row["graph"]) for row in selected})
    matrix = []
    for graph in graphs:
        graph_rows = [row for row in selected if int(row["graph"]) == graph]
        sources = sorted({int(row.get("source", -1)) for row in graph_rows})
        source_profiles = []
        for source in sources:
            source_rows = [row for row in graph_rows if int(row.get("source", -1)) == source]
            profile = []
            for distance in distances:
                values = [
                    float(row[key])
                    for row in source_rows
                    if int(row["distance"]) == int(distance)
                ]
                profile.append(float(np.mean(values)) if values else 0.0)
            source_profiles.append(profile)
        if source_profiles:
            matrix.append(np.mean(np.asarray(source_profiles, dtype=float), axis=0))
    return np.asarray(matrix, dtype=float), tuple(graphs)


def _profile_interval(
    matrix: np.ndarray,
    *,
    width: int,
    distances: Sequence[int] | None = None,
    long_range_radius: int = 2,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    distance_axis = np.asarray(
        tuple(range(int(width))) if distances is None else tuple(distances), dtype=float
    )
    long_mask = distance_axis > int(long_range_radius)
    if matrix.size == 0:
        return {
            "mean": [np.nan] * int(width),
            "low": [np.nan] * int(width),
            "high": [np.nan] * int(width),
            "graphs": 0,
            "expected_distance": np.nan,
            "expected_distance_low": np.nan,
            "expected_distance_high": np.nan,
            "long_range_share": np.nan,
            "long_range_share_low": np.nan,
            "long_range_share_high": np.nan,
        }
    point = _normalise_profile(np.mean(matrix, axis=0))
    if matrix.shape[0] < 2 or int(replicates) < 2:
        low = high = point.copy()
        draws = point[None, :]
    else:
        rng = np.random.default_rng(int(seed))
        draws = np.empty((int(replicates), matrix.shape[1]), dtype=float)
        for draw in range(int(replicates)):
            selected = rng.integers(0, matrix.shape[0], matrix.shape[0])
            draws[draw] = _normalise_profile(np.mean(matrix[selected], axis=0))
        low = np.nanquantile(draws, 0.025, axis=0)
        high = np.nanquantile(draws, 0.975, axis=0)
    expected = draws @ distance_axis
    long_share = np.sum(draws[:, long_mask], axis=1)
    return {
        "mean": point.tolist(),
        "low": low.tolist(),
        "high": high.tolist(),
        "graphs": int(matrix.shape[0]),
        "expected_distance": float(np.sum(point * distance_axis)),
        "expected_distance_low": float(np.nanquantile(expected, 0.025)),
        "expected_distance_high": float(np.nanquantile(expected, 0.975)),
        "long_range_share": float(np.sum(point[long_mask])),
        "long_range_share_low": float(np.nanquantile(long_share, 0.025)),
        "long_range_share_high": float(np.nanquantile(long_share, 0.975)),
    }


def _summarise_measurement(
    config: Config,
    task: str,
    scores: Mapping[str, Any],
    role_heads: Mapping[str, Sequence[tuple[int, int]]],
    shards: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attention_rows = [row for shard in shards for row in shard.get("attention_rows", ())]
    mediation_rows = [row for shard in shards for row in shard.get("mediation_rows", ())]
    s_rows = _sufficient_profile_rows(scores, role_heads)
    maximum = max(
        [int(row["distance"]) for row in attention_rows + mediation_rows + s_rows if isinstance(row.get("distance"), (int, np.integer))],
        default=0,
    )
    distances = tuple(range(maximum + 1))
    profiles: dict[str, Any] = {}
    statistics: list[dict[str, Any]] = []
    seed_offset = 0
    for channel in CHANNELS:
        profiles[channel] = {}
        for role, heads in role_heads.items():
            profiles[channel][role] = {}
            matrices = {
                "A": _graph_profile_matrix(
                    attention_rows, key="A", distances=distances, role=role
                )[0],
                "S": _graph_profile_matrix(
                    s_rows, key="S", distances=distances, role=role, channel=channel
                )[0],
                "M": _graph_profile_matrix(
                    mediation_rows,
                    key="M",
                    distances=distances,
                    target=f"family:{role}",
                    channel=channel,
                )[0],
            }
            for measure, matrix in matrices.items():
                record = _profile_interval(
                    matrix,
                    width=len(distances),
                    distances=distances,
                    long_range_radius=config.long_range_radius,
                    replicates=config.bootstrap_replicates,
                    seed=config.analysis_seed + seed_offset,
                )
                seed_offset += 1
                profiles[channel][role][measure] = record

            s_expected = profiles[channel][role]["S"]["expected_distance"]
            m_expected = profiles[channel][role]["M"]["expected_distance"]
            individual_long = []
            for head in heads:
                matrix, _ = _graph_profile_matrix(
                    mediation_rows,
                    key="M",
                    distances=distances,
                    target=_head_target(tuple(head)),
                    channel=channel,
                )
                point = (
                    np.mean(matrix, axis=0)
                    if matrix.size
                    else np.full(len(distances), np.nan)
                )
                individual_long.append(
                    float(np.nansum(point[np.asarray(distances) > int(config.long_range_radius)]))
                )
            long_mass = np.asarray(individual_long, dtype=float)
            long_allocation = _normalise_profile(long_mass)
            effective = (
                float(1.0 / np.nansum(long_allocation**2))
                if np.isfinite(long_allocation).any() and np.nansum(long_allocation**2) > 0
                else np.nan
            )

            family_raw, _ = _graph_profile_matrix(
                mediation_rows,
                key="M",
                distances=distances,
                target=f"family:{role}",
                channel=channel,
            )
            individual_raw = []
            for head in heads:
                matrix, _ = _graph_profile_matrix(
                    mediation_rows,
                    key="M",
                    distances=distances,
                    target=_head_target(tuple(head)),
                    channel=channel,
                )
                if matrix.size:
                    individual_raw.append(np.mean(matrix, axis=0))
            joint = np.mean(family_raw, axis=0) if family_raw.size else np.full(len(distances), np.nan)
            summed = np.sum(np.stack(individual_raw), axis=0) if individual_raw else np.full(len(distances), np.nan)
            ratio = np.full(len(distances), np.nan)
            np.divide(joint, summed, out=ratio, where=np.asarray(summed) > 1.0e-12)
            profiles[channel][role]["joint_to_sum"] = ratio.tolist()
            statistics.append(
                {
                    "task": task,
                    "channel": channel,
                    "role": role,
                    "heads": len(heads),
                    "S_expected_distance": float(s_expected),
                    "M_expected_distance": float(m_expected),
                    "realisation_contraction": float(s_expected - m_expected),
                    "S_long_range_share": float(profiles[channel][role]["S"]["long_range_share"]),
                    "M_long_range_share": float(profiles[channel][role]["M"]["long_range_share"]),
                    "effective_long_range_heads": effective,
                    "mean_joint_to_sum": float(np.nanmean(ratio)) if np.isfinite(ratio).any() else np.nan,
                }
            )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "task": task,
        "distances": list(distances),
        "role_heads": {
            role: [list(head) for head in heads] for role, heads in role_heads.items()
        },
        "profiles": profiles,
        "statistics": statistics,
        "attention_rows": attention_rows,
        "mediation_rows": mediation_rows,
        "score_rows": s_rows,
        "warnings": [
            str(shard["warning"])
            for shard in shards
            if shard.get("status") != "complete" and shard.get("warning")
        ],
    }


def _load_task_context(config: Config, task: str) -> dict[str, Any]:
    task_root = config.canonical_root / task / f"seed_{int(config.train_seed)}"
    artifact = load_canonical_score_artifact(
        task_root / "cache" / "scores" / "raw.pt", expected_task=task
    )
    model_record = load_canonical_model_record(task_root / "model.json", artifact)
    metrics = CanonicalHeadMetrics.from_scores(artifact.value)
    roles = select_role_heads(artifact.value, metrics, count=config.heads_per_family)
    contract = _measurement_contract(config, task, artifact, roles)
    cache = SupplementalCache(config.output_dir / task / "cache")
    return {
        "task_root": task_root,
        "artifact": artifact,
        "model_record": model_record,
        "metrics": metrics,
        "role_heads": roles,
        "contract": contract,
        "cache": cache,
    }


def measure_task(config: Config, task: str, context: Mapping[str, Any]) -> dict[str, Any]:
    cached = None if config.force else context["cache"].load("causal-spatial-support", context["contract"])
    if cached is not None:
        value, path = cached
        print(f"[cache] loaded {task} causal spatial support: {path}")
        return value
    protocol_config = methodology_config_from_record(
        config.canonical_root / "protocol.json", accelerator=config.accelerator
    )
    runtime = build_verified_grit_figure_runtime(
        context["artifact"],
        context["model_record"],
        protocol_config,
        runtime_output_dir=config.output_dir / task / "runtime",
    )
    if runtime.prepared.task.virtual_node:
        raise ValueError(f"{task} uses a VNode; this experiment is registered for dense models")
    from .runner import _stage_plan

    plan = _stage_plan(runtime.prepared, protocol_config, "causal")
    graph_ids = tuple(sorted(plan)[: int(config.graphs)])
    shards = []
    for graph_id in graph_ids:
        for channel in CHANNELS:
            shard_contract = {
                **context["contract"],
                "graph": int(graph_id),
                "channel": channel,
            }
            value, path, hit = context["cache"].load_or_compute(
                f"shell-{channel}-graph-{int(graph_id):06d}",
                shard_contract,
                lambda graph_id=graph_id, channel=channel: _safe_measure_graph_channel(
                    runtime.prepared,
                    protocol_config,
                    plan,
                    int(graph_id),
                    channel,
                    context["role_heads"],
                    config,
                ),
                force=config.force,
            )
            print(
                f"[{task}] graph={graph_id} channel={channel} "
                f"{'cache' if hit else 'measured'}"
            )
            shards.append(value)
    summary = _summarise_measurement(
        config,
        task,
        context["artifact"].value,
        context["role_heads"],
        shards,
    )
    value, path, _ = context["cache"].load_or_compute(
        "causal-spatial-support",
        context["contract"],
        lambda: summary,
        force=config.force,
    )
    print(f"[cache] wrote {task} causal spatial support: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_headline(config: Config, results: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    tasks = [task for task in config.tasks if task in results]
    fig, axes = plt.subplots(len(tasks), 3, figsize=(14.0, 4.25 * len(tasks)), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.09, top=0.84, hspace=0.48, wspace=0.28)
    for row, task in enumerate(tasks):
        result = results[task]
        distances = np.asarray(result["distances"], dtype=int)
        for column, (channel, role) in enumerate(
            (("semantic", "semantic"), ("structural", "structural"))
        ):
            axis = axes[row, column]
            for measure, label, linestyle, marker in (
                ("A", "Direct attention access", ":", None),
                ("S", "Internal response", "--", "^"),
                ("M", "Finite output mediation", "-", "o"),
            ):
                record = result["profiles"][channel][role][measure]
                mean = np.asarray(record["mean"], dtype=float)
                low = np.asarray(record["low"], dtype=float)
                high = np.asarray(record["high"], dtype=float)
                axis.plot(
                    distances,
                    mean,
                    color=MEASURE_COLOURS[measure],
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=2.0,
                    markersize=5,
                    label=label,
                )
                if measure in {"S", "M"} and low.size:
                    axis.fill_between(distances, low, high, color=MEASURE_COLOURS[measure], alpha=0.10)
            axis.set_xlim(left=-0.1)
            axis.set_ylim(bottom=0)
            axis.set_xlabel("Distance from intervened node")
            axis.set_ylabel("Normalised radial mass")
            axis.set_title(f"{TASK_LABELS.get(task, task)}: {ROLE_LABELS[role]}")
            axis.spines[["top", "right"]].set_visible(False)
            if row == 0 and column == 0:
                axis.legend(frameon=False, fontsize=8.5)

        axis = axes[row, 2]
        groups = [
            ("semantic", "semantic", "Semantic"),
            ("structural", "structural", "Structural"),
            ("semantic", "semantic_control", "Sem. control"),
            ("structural", "structural_control", "Str. control"),
            ("semantic", "generalist", "Generalist\n(semantic)"),
            ("structural", "generalist", "Generalist\n(structural)"),
        ]
        groups = [entry for entry in groups if entry[1] in result["profiles"][entry[0]]]
        positions = np.arange(len(groups))
        width = 0.34
        records = {
            measure: [result["profiles"][channel][role][measure] for channel, role, _ in groups]
            for measure in ("S", "M")
        }
        for offset, measure, label in (
            (-width / 2, "S", "Internal"),
            (width / 2, "M", "Mediated"),
        ):
            values = np.asarray(
                [record["long_range_share"] for record in records[measure]], dtype=float
            )
            low = np.asarray(
                [record["long_range_share_low"] for record in records[measure]], dtype=float
            )
            high = np.asarray(
                [record["long_range_share_high"] for record in records[measure]], dtype=float
            )
            errors = np.vstack((np.maximum(values - low, 0), np.maximum(high - values, 0)))
            errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
            axis.barh(
                positions + offset,
                values,
                width,
                xerr=errors,
                capsize=2,
                color=MEASURE_COLOURS[measure],
                alpha=0.85,
                label=label,
            )
        axis.set_yticks(positions, [label for _, _, label in groups])
        axis.invert_yaxis()
        axis.set_xlabel(f"Normalised mass beyond $d>{int(config.long_range_radius)}$")
        axis.set_xlim(left=0)
        axis.set_title(f"{TASK_LABELS.get(task, task)}: realised long-range share")
        axis.spines[["top", "right"]].set_visible(False)
        if row == 0:
            axis.legend(frameon=False, fontsize=8.5)
    fig.suptitle("From specialised head activity to realised spatial computation", fontsize=16, y=0.97)
    fig.text(
        0.5,
        0.91,
        "Discovery-split internal response; held-out source-conditioned attention and symmetric shell mediation",
        ha="center",
        color="#666666",
        fontsize=9.5,
    )
    figure_dir = config.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "causal_spatial_support_headline"
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"headline_png": str(png), "headline_pdf": str(pdf)}


def _plot_overlap(config: Config, results: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    tasks = [task for task in config.tasks if task in results]
    fig, axes = plt.subplots(len(tasks), 2, figsize=(11.8, 3.9 * len(tasks)), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.84, hspace=0.46, wspace=0.30)
    for row, task in enumerate(tasks):
        result = results[task]
        distances = np.asarray(result["distances"], dtype=int)
        axis = axes[row, 0]
        for role in PRIMARY_ROLES:
            if role not in result["profiles"]["semantic"]:
                continue
            # Specialists use their matching channel; generalists are shown for both.
            channel = "structural" if role == "structural" else "semantic"
            ratio = np.asarray(result["profiles"][channel][role]["joint_to_sum"], dtype=float)
            axis.plot(distances, ratio, marker="o", linewidth=1.8, color=ROLE_COLOURS[role], label=ROLE_LABELS[role])
        axis.axhline(1.0, color="#999999", linestyle="--", linewidth=1)
        axis.set_xlabel("Distance from intervened node")
        axis.set_ylabel("Joint / summed individual mediation")
        axis.set_title(f"{TASK_LABELS.get(task, task)}: pathway overlap")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8)

        axis = axes[row, 1]
        rows = [
            record
            for record in result["statistics"]
            if (record["role"], record["channel"])
            in {("semantic", "semantic"), ("structural", "structural"), ("generalist", "semantic")}
        ]
        labels = [ROLE_LABELS[record["role"]] for record in rows]
        values = [record["effective_long_range_heads"] for record in rows]
        axis.barh(np.arange(len(rows)), values, color=[ROLE_COLOURS[record["role"]] for record in rows])
        axis.set_yticks(np.arange(len(rows)), labels)
        axis.set_xlabel("Effective heads carrying long-range mediation")
        axis.set_title(f"{TASK_LABELS.get(task, task)}: concentration")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Concentration and overlap of realised head pathways", fontsize=15.5, y=0.97)
    fig.text(
        0.5,
        0.91,
        "Joint/sum below one indicates overlapping finite pathways; it is not a retrained task-necessity estimate",
        ha="center",
        color="#666666",
        fontsize=9,
    )
    figure_dir = config.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / "causal_spatial_support_overlap"
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    fig.savefig(png, dpi=260, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"overlap_png": str(png), "overlap_pdf": str(pdf)}


def render(config: Config, results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    figures = {}
    figures.update(_plot_headline(config, results))
    figures.update(_plot_overlap(config, results))
    result_dir = config.output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    statistics = [row for result in results.values() for row in result["statistics"]]
    _write_csv(result_dir / "causal_spatial_support_statistics.csv", statistics)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in dataclasses.asdict(config).items()
        },
        "tasks": list(results),
        "figures": figures,
        "warnings": [warning for result in results.values() for warning in result.get("warnings", ())],
        "statistics": statistics,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def run(config: Config) -> dict[str, Any]:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    warnings: list[str] = []
    for task in config.tasks:
        try:
            context = _load_task_context(config, task)
            cached = None if config.force else context["cache"].load(
                "causal-spatial-support", context["contract"]
            )
            if cached is not None:
                results[task] = cached[0]
                print(f"[cache] loaded {task}: {cached[1]}")
            elif config.phase == "figures":
                warnings.append(f"{task}: complete measurement cache is unavailable")
            else:
                results[task] = measure_task(config, task, context)
        except Exception as error:
            warning = f"{task}: {type(error).__name__}: {error}"
            warnings.append(warning)
            print(f"[causal-spatial-support:warning] {warning}")
    if results:
        summary = render(config, results)
        summary["warnings"] = warnings + list(summary.get("warnings", ()))
        summary_path = config.output_dir / "results" / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    else:
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "status": "not_estimable",
            "warnings": warnings or ["no task produced an estimable result"],
            "figures": {},
        }
    audit = config.output_dir / "results" / "reported_error_audit.json"
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", default="zinc,qm9_gap_dense")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--phase", choices=("all", "measure", "figures"), default="all")
    parser.add_argument("--graphs", type=int, default=8)
    parser.add_argument("--sources-per-graph", type=int, default=2)
    parser.add_argument("--donors-per-source", type=int, default=1)
    parser.add_argument("--heads-per-family", type=int, default=3)
    parser.add_argument("--long-range-radius", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--analysis-seed", type=int, default=72_019)
    parser.add_argument("--accelerator", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    values = vars(args)
    values["tasks"] = tuple(value.strip() for value in values["tasks"].split(",") if value.strip())
    return Config(**values)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
