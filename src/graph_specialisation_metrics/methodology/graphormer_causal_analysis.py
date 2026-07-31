"""Focused, cache-resumable causal tests for the official PCQM4Mv2 Graphormer."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..carriage.env import log
from .bootstrap import (
    Observation,
    nested_percentile_interval,
    paired_channel_percentile_interval,
)
from .cache import CanonicalCache, atomic_json, load_cache_artifact_file
from .causal import donor_necessity, patch_response
from .protocol import (
    CHANNELS,
    BootstrapPolicy,
    ExecutionPolicy,
    FamilyPolicy,
    MethodologyConfig,
    RunSizes,
    stable_hash,
)
from .scores import (
    CONFIDENCE_SPECIALIST_VERSION,
    confidence_specialists_from_draws,
    graph_local_fixed_normalization,
    head_coordinates,
)


FOCUSED_CAUSAL_VERSION = "graphormer-pcqm-focused-causal-v4"
PATCH_METRICS = (
    "R_gross_matched",
    "R_gross_null",
    "R_gross_adj",
    "R_align_matched",
    "R_align_null",
    "R_align_adj",
    "I_gross_matched",
    "I_gross_null",
    "I_gross_adj",
    "I_align_matched",
    "I_align_null",
    "I_align_adj",
)
NECESSITY_METRICS = (
    "N_gross",
    "N_align",
    "N_gross_fraction",
    "N_fraction",
)


@dataclass(frozen=True)
class FocusedExecution:
    head_batch_size: int = 4
    event_batch_size: int = 8

    def validate(self) -> None:
        if int(self.head_batch_size) < 1:
            raise ValueError("focused head_batch_size must be positive")
        if int(self.event_batch_size) < 1:
            raise ValueError("focused event_batch_size must be positive")


def production_config(
    *,
    output_dir: str,
    dataset_root: str,
    cache_dir: str,
    accelerator: str = "cuda:0",
    force: bool = False,
    graphs_per_batch: int = 2,
) -> MethodologyConfig:
    """Return the fixed 128/128/128 PCQM causal configuration."""

    return MethodologyConfig(
        output_dir=str(output_dir),
        tasks=("graphormer_pcqm4mv2",),
        train_seeds=(0,),
        task_train_seeds={"graphormer_pcqm4mv2": (0,)},
        phases=("scores",),
        sizes=RunSizes(
            discovery_graphs=128,
            causal_graphs=128,
            clean_ablation_graphs=128,
            semantic_donor_graphs=2_000,
            sources_per_graph=6,
            donors_per_source=8,
            bootstrap_replicates=2_000,
        ),
        families=FamilyPolicy(
            activity_floor=0.20,
            specialist_minimum_pairs=3,
            equivalence_half_width=0.10,
        ),
        bootstrap=BootstrapPolicy(replicates=2_000, rng_seed=17_071),
        execution=ExecutionPolicy(
            graphs_per_batch=int(graphs_per_batch),
            oom_backoff=True,
        ),
        accelerator=str(accelerator),
        task_overrides={
            "graphormer_pcqm4mv2": {
                "dataset_root": str(dataset_root),
                "cache_dir": str(cache_dir),
            }
        },
        resume=True,
        force=bool(force),
        compute_beneficial_carriage=False,
    )


def _head_tuple(value: Sequence[int]) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def _head_name(head: Sequence[int]) -> str:
    layer, index = _head_tuple(head)
    return f"L{layer}H{index}"


def _score_gate_cache(prepared: Any, config: MethodologyConfig, scores: Mapping[str, Any]):
    from .runner import _cache, _stage_plan

    plan = _stage_plan(prepared, config, "scores")
    base = _cache(prepared, config, plan)
    fingerprint = stable_hash(
        {
            "base_event_manifest": base.contract.event_manifest_hash,
            "source_score_manifest": str(scores["manifest_hash"]),
            "analysis": FOCUSED_CAUSAL_VERSION,
            "classifier": CONFIDENCE_SPECIALIST_VERSION,
            "activity_floor": float(config.families.activity_floor),
            "preference_threshold": float(config.families.equivalence_half_width),
            "bootstrap_confidence": 0.95,
            "minimum_pairs": int(config.families.specialist_minimum_pairs),
            "coordinate_shape": tuple(scores["coordinates"].joint_sensitivity.shape),
        }
    )
    return CanonicalCache(
        config.root,
        dataclasses.replace(base.contract, event_manifest_hash=fingerprint),
        stale_policy="archive",
    )


def _score_observations(
    scores: Mapping[str, Any], seed: int
) -> dict[str, list[Observation]]:
    """Rebuild each discovery channel's hierarchy from its cached event rows."""

    observations: dict[str, list[Observation]] = {}
    for channel in CHANNELS:
        observations[channel] = [
            Observation(
                seed=int(seed),
                graph=int(row["graph_id"]),
                source=int(row["source"]),
                donor=int(row["draw"]),
                value=np.asarray(row["score"], dtype=np.float64),
            )
            for row in scores["channels"][channel]["events"]
        ]
        if not observations[channel]:
            raise ValueError(f"score cache has no {channel} discovery events")
    return observations


def run_confidence_gate(
    prepared: Any,
    config: MethodologyConfig,
    scores: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify specialists from complete discovery bootstrap support."""

    cache = _score_gate_cache(prepared, config, scores)
    if config.resume and not config.force:
        cached = cache.load("focused", "gate", strict=True)
        if cached is not None:
            log("[cache] loaded focused bootstrap-confidence specialist gate")
            return cached

    coordinates = scores["coordinates"]

    def transform(value: np.ndarray) -> np.ndarray:
        derived = head_coordinates(
            value[0],
            value[1],
            score_floor=config.numerical.score_floor,
            epsilon=config.numerical.selectivity_epsilon,
            activity_floor=config.families.activity_floor,
        )
        return np.stack((derived.joint_sensitivity, derived.selectivity))

    observations = _score_observations(scores, int(prepared.grit.sc.seed))
    log(
        "[focused] bootstrapping the specialist gate: 2,000 CPU draws over "
        "graph/source/donor score summaries"
    )

    def report_gate_draw(completed: int, total: int) -> None:
        if completed == 1 or completed % 100 == 0 or completed == total:
            log(
                "[focused] specialist-gate bootstrap "
                f"draws={completed}/{total}"
            )

    interval = paired_channel_percentile_interval(
        observations["semantic"],
        observations["structural"],
        config.bootstrap,
        transform=transform,
        resample_source=tuple(
            bool(scores["channels"][channel].get("resample_source", True))
            for channel in CHANNELS
        ),
        retain_draws=True,
        on_draw=report_gate_draw,
    )
    if interval.draws is None:
        raise RuntimeError("specialist gate bootstrap did not retain its transient draws")
    gate = confidence_specialists_from_draws(
        coordinates,
        interval.draws[:, 0],
        interval.draws[:, 1],
        activity_floor=config.families.activity_floor,
        preference_threshold=config.families.equivalence_half_width,
        confidence=0.95,
        minimum_pairs=config.families.specialist_minimum_pairs,
    )
    gate["coordinate_interval"] = {
        "order": ("J", "D_rel"),
        "estimate": interval.estimate,
        "low": interval.low,
        "high": interval.high,
        "replicates": int(interval.replicates),
        "rng_seed": int(interval.rng_seed),
        "resampled_levels": interval.resampled_levels,
        "estimable_draws": interval.estimable_draws,
    }
    gate["molecule_diagnostic"] = graph_local_fixed_normalization(
        scores["channels"]["semantic"]["graph_scores"],
        scores["channels"]["structural"]["graph_scores"],
        coordinates,
        epsilon=config.numerical.selectivity_epsilon,
        preference_threshold=config.families.equivalence_half_width,
    )
    gate["discovery_graph_count"] = int(
        len(gate["molecule_diagnostic"]["graph_ids"])
    )
    gate["score_manifest_hash"] = str(scores["manifest_hash"])
    cache.save("focused", "gate", gate)
    log(
        "[focused] specialist gate complete: "
        f"{len(gate['heads']['semantic_specialist'])} semantic, "
        f"{len(gate['heads']['structural_specialist'])} structural, "
        f"status={gate['status']}"
    )
    return gate


def _target_heads(gate: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    values: set[tuple[int, int]] = set()
    for key in ("semantic_specialist", "structural_specialist"):
        values.update(_head_tuple(head) for head in gate["heads"][key])
    for key in ("semantic_null_J_matching", "structural_null_J_matching"):
        for pair in gate[key]["pairs"]:
            values.add(_head_tuple(pair["specialist"]))
            values.add(_head_tuple(pair["null"]))
    return tuple(sorted(values))


def _focused_causal_cache(
    prepared: Any,
    config: MethodologyConfig,
    gate: Mapping[str, Any],
) -> tuple[Mapping[int, Mapping[str, Any]], CanonicalCache]:
    from .runner import _cache, _stage_plan

    plan = _stage_plan(prepared, config, "causal")
    base = _cache(prepared, config, plan)
    target_manifest = {
        "targets": _target_heads(gate),
        "specialist_matching": gate["specialist_J_matching"]["pairs"],
        "semantic_null_matching": gate["semantic_null_J_matching"]["pairs"],
        "structural_null_matching": gate["structural_null_J_matching"]["pairs"],
    }
    fingerprint = stable_hash(
        {
            "base_event_manifest": base.contract.event_manifest_hash,
            "analysis": FOCUSED_CAUSAL_VERSION,
            "gate_version": gate["version"],
            "target_manifest": target_manifest,
            "patch_metrics": PATCH_METRICS,
            "necessity_metrics": NECESSITY_METRICS,
            "mismatch": "same-source/same-degree-tier/distinct-payload/nearest-dose/no-caliper",
        }
    )
    return plan, CanonicalCache(
        config.root,
        dataclasses.replace(base.contract, event_manifest_hash=fingerprint),
        stale_policy="archive",
    )


def _mismatch_indices(records: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Same-source, same-tier, distinct-payload nearest-dose activation controls."""

    mismatch = np.full(len(records), -1, dtype=np.int64)
    controlled = np.zeros(len(records), dtype=bool)
    for position, row in enumerate(records):
        candidates = [
            other
            for other, candidate in enumerate(records)
            if other != position
            and int(candidate.source) == int(row.source)
            and int(candidate.degree_gap) == int(row.degree_gap)
            and candidate.payload_fingerprint != row.payload_fingerprint
        ]
        if not candidates:
            continue
        selected = min(
            candidates,
            key=lambda other: (
                abs(float(records[other].dose) - float(row.dose)),
                int(records[other].draw),
                other,
            ),
        )
        mismatch[position] = int(selected)
        controlled[position] = True
    return mismatch, controlled


def _split_head_chunk(heads: Sequence[tuple[int, int]]) -> tuple[tuple[Any, ...], ...]:
    midpoint = max(1, len(heads) // 2)
    return tuple(filter(None, (tuple(heads[:midpoint]), tuple(heads[midpoint:]))))


def _causal_graph_channel(
    prepared: Any,
    config: MethodologyConfig,
    graph_id: int,
    channel: str,
    plan: Mapping[int, Mapping[str, Any]],
    heads: Sequence[tuple[int, int]],
    execution: FocusedExecution,
) -> dict[str, Any]:
    import torch

    from .runner import _rebuild_graph_events

    base = prepared.grit.eval_ds[int(graph_id)]
    sources = plan[int(graph_id)][channel]["sources"]
    variants, records = _rebuild_graph_events(
        prepared, config, "causal", int(graph_id), channel, sources
    )
    captured = prepared.backend.capture(
        [base, *variants], require_grad=False, include_virtual_transport=True
    )
    z_clean = captured.z[0:1].detach().cpu().numpy()
    z_event_all = captured.z[1:].detach().cpu().numpy()
    mismatch, controlled = _mismatch_indices(records)
    safe_mismatch = np.where(controlled, mismatch, np.arange(len(records)))
    output_rows: list[dict[str, Any]] = []
    clean_self_patch_max = 0.0
    event_self_patch_max = 0.0

    def process_head_chunk(selected_heads: Sequence[tuple[int, int]]):
        nonlocal clean_self_patch_max, event_self_patch_max
        try:
            b = len(selected_heads)
            self_replacements = prepared.backend.replacement_batch(
                captured, [0] * b
            )
            _, self_z, _ = prepared.backend.patch_individual_heads(
                [base] * b, self_replacements, selected_heads
            )
            clean_self_patch_max = max(
                clean_self_patch_max,
                float(
                    np.max(
                        np.abs(
                            self_z.detach().cpu().numpy()
                            - np.repeat(z_clean, b, axis=0)
                        )
                    )
                ),
            )
            rows: list[dict[str, Any]] = []
            for event_start in range(0, len(records), int(execution.event_batch_size)):
                positions = list(
                    range(
                        event_start,
                        min(len(records), event_start + int(execution.event_batch_size)),
                    )
                )
                event_count = len(positions)
                assignments = [head for head in selected_heads for _ in positions]
                event_targets = [
                    variants[position]
                    for _head in selected_heads
                    for position in positions
                ]
                base_targets = [base] * (b * event_count)
                ablation_targets = [
                    target
                    for _head in selected_heads
                    for target in (base, *(variants[position] for position in positions))
                ]
                ablation_assignments = [
                    head for head in selected_heads for _ in range(event_count + 1)
                ]
                _, z_ablated, _ = prepared.backend.ablate_individual_heads(
                    ablation_targets, ablation_assignments
                )
                z_ablated_np = z_ablated.detach().cpu().numpy().reshape(
                    b, event_count + 1, -1
                )

                clean_replacements = prepared.backend.replacement_batch(
                    captured, [0] * (b * event_count)
                )
                event_capture_indices = [
                    position + 1 for _head in selected_heads for position in positions
                ]
                event_replacements = prepared.backend.replacement_batch(
                    captured, event_capture_indices
                )
                mismatch_capture_indices = [
                    int(safe_mismatch[position]) + 1
                    for _head in selected_heads
                    for position in positions
                ]
                mismatch_replacements = prepared.backend.replacement_batch(
                    captured, mismatch_capture_indices
                )
                _, z_restore, _ = prepared.backend.patch_individual_heads(
                    event_targets, clean_replacements, assignments
                )
                _, z_inject, _ = prepared.backend.patch_individual_heads(
                    base_targets, event_replacements, assignments
                )
                _, z_restore_null, _ = prepared.backend.patch_individual_heads(
                    event_targets, mismatch_replacements, assignments
                )
                _, z_inject_null, _ = prepared.backend.patch_individual_heads(
                    base_targets, mismatch_replacements, assignments
                )
                _, z_event_self, _ = prepared.backend.patch_individual_heads(
                    event_targets, event_replacements, assignments
                )

                clean_repeated = np.broadcast_to(
                    z_clean.reshape(1, 1, -1), (b, event_count, z_clean.shape[-1])
                )
                event_values = z_event_all[positions]
                event_repeated = np.broadcast_to(
                    event_values.reshape(1, event_count, -1),
                    (b, event_count, event_values.shape[-1]),
                )
                event_self_patch_max = max(
                    event_self_patch_max,
                    float(
                        np.max(
                            np.abs(
                                z_event_self.detach().cpu().numpy()
                                - event_repeated.reshape(b * event_count, -1)
                            )
                        )
                    ),
                )
                matched = patch_response(
                    clean_repeated.reshape(b * event_count, -1),
                    event_repeated.reshape(b * event_count, -1),
                    z_restore.detach().cpu().numpy(),
                    z_inject.detach().cpu().numpy(),
                    epsilon=config.numerical.effect_floor,
                )
                null = patch_response(
                    clean_repeated.reshape(b * event_count, -1),
                    event_repeated.reshape(b * event_count, -1),
                    z_restore_null.detach().cpu().numpy(),
                    z_inject_null.detach().cpu().numpy(),
                    epsilon=config.numerical.effect_floor,
                )
                necessity = donor_necessity(
                    clean_repeated.reshape(b * event_count, -1),
                    event_repeated.reshape(b * event_count, -1),
                    np.repeat(z_ablated_np[:, 0:1], event_count, axis=1).reshape(
                        b * event_count, -1
                    ),
                    z_ablated_np[:, 1:].reshape(b * event_count, -1),
                    epsilon=config.numerical.effect_floor,
                )

                reshape = lambda value: np.asarray(value).reshape(b, event_count)
                arrays = {
                    "R_gross_matched": reshape(matched.restoration_gross),
                    "R_gross_null": reshape(null.restoration_gross),
                    "R_align_matched": reshape(matched.restoration_aligned),
                    "R_align_null": reshape(null.restoration_aligned),
                    "I_gross_matched": reshape(matched.injection_gross),
                    "I_gross_null": reshape(null.injection_gross),
                    "I_align_matched": reshape(matched.injection_aligned),
                    "I_align_null": reshape(null.injection_aligned),
                    "N_gross": reshape(necessity["gross_necessity"]),
                    "N_align": reshape(necessity["aligned_necessity"]),
                    "event_effect": reshape(necessity["event_effect"]),
                }
                arrays["R_gross_adj"] = arrays["R_gross_matched"] - arrays["R_gross_null"]
                arrays["R_align_adj"] = arrays["R_align_matched"] - arrays["R_align_null"]
                arrays["I_gross_adj"] = arrays["I_gross_matched"] - arrays["I_gross_null"]
                arrays["I_align_adj"] = arrays["I_align_matched"] - arrays["I_align_null"]
                effect = arrays["event_effect"]
                above = effect > float(config.numerical.effect_floor)
                arrays["N_gross_fraction"] = np.where(
                    above, arrays["N_gross"] / effect, np.nan
                )
                arrays["N_fraction"] = np.where(
                    above, arrays["N_align"] / effect, np.nan
                )

                for head_position, head in enumerate(selected_heads):
                    for local_position, event_position in enumerate(positions):
                        record = records[event_position]
                        row = {
                            "graph": int(graph_id),
                            "source": int(record.source),
                            "donor": int(record.draw),
                            "channel": channel,
                            "head": tuple(head),
                            "head_name": _head_name(head),
                            "matched_payload_fingerprint": record.payload_fingerprint,
                            "matched_dose": float(record.dose),
                            "controlled": bool(controlled[event_position]),
                            "event_effect": float(
                                arrays["event_effect"][head_position, local_position]
                            ),
                        }
                        if controlled[event_position]:
                            null_record = records[int(mismatch[event_position])]
                            row.update(
                                {
                                    "null_donor": int(null_record.draw),
                                    "null_payload_fingerprint": null_record.payload_fingerprint,
                                    "null_dose": float(null_record.dose),
                                    "absolute_dose_gap": abs(
                                        float(null_record.dose) - float(record.dose)
                                    ),
                                }
                            )
                        else:
                            row.update(
                                {
                                    "null_donor": -1,
                                    "null_payload_fingerprint": None,
                                    "null_dose": np.nan,
                                    "absolute_dose_gap": np.nan,
                                }
                            )
                        for metric in PATCH_METRICS:
                            row[metric] = (
                                float(arrays[metric][head_position, local_position])
                                if controlled[event_position]
                                else np.nan
                            )
                        for metric in NECESSITY_METRICS:
                            row[metric] = float(
                                arrays[metric][head_position, local_position]
                            )
                        rows.append(row)
            return rows
        except torch.cuda.OutOfMemoryError:
            if len(selected_heads) <= 1:
                raise
            torch.cuda.empty_cache()
            split_rows: list[dict[str, Any]] = []
            for smaller in _split_head_chunk(selected_heads):
                split_rows.extend(process_head_chunk(smaller))
            return split_rows

    for start in range(0, len(heads), int(execution.head_batch_size)):
        output_rows.extend(
            process_head_chunk(heads[start : start + int(execution.head_batch_size)])
        )
    return {
        "graph": int(graph_id),
        "channel": channel,
        "rows": output_rows,
        "event_count": len(records),
        "controlled_event_count": int(controlled.sum()),
        "uncontrolled_event_count": int((~controlled).sum()),
        "clean_same_condition_patch_max": float(clean_self_patch_max),
        "event_same_condition_patch_max": float(event_self_patch_max),
    }


def run_causal_events(
    prepared: Any,
    config: MethodologyConfig,
    gate: Mapping[str, Any],
    *,
    execution: FocusedExecution,
) -> tuple[list[dict[str, Any]], CanonicalCache, list[dict[str, Any]]]:
    execution.validate()
    plan, cache = _focused_causal_cache(prepared, config, gate)
    heads = _target_heads(gate)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    if gate["status"] != "estimable" or not heads:
        return rows, cache, audits
    for graph_id in sorted(plan):
        for channel in CHANNELS:
            stage = f"focused/events/{channel}"
            name = f"graph_{int(graph_id):06d}"
            cached = (
                cache.load(stage, name, strict=True)
                if config.resume and not config.force
                else None
            )
            if cached is None:
                cached = _causal_graph_channel(
                    prepared,
                    config,
                    int(graph_id),
                    channel,
                    plan,
                    heads,
                    execution,
                )
                cache.save(stage, name, cached)
            rows.extend(cached["rows"])
            audits.append(
                {
                    key: cached[key]
                    for key in (
                        "graph",
                        "channel",
                        "event_count",
                        "controlled_event_count",
                        "uncontrolled_event_count",
                        "clean_same_condition_patch_max",
                        "event_same_condition_patch_max",
                    )
                }
            )
    return rows, cache, audits


def _all_heads(prepared: Any) -> tuple[tuple[int, int], ...]:
    return tuple(
        (layer, head)
        for layer in range(int(prepared.grit.L))
        for head in range(int(prepared.grit.H))
    )


def _clean_ablation_graphs(
    prepared: Any,
    config: MethodologyConfig,
    cache: CanonicalCache,
    execution: FocusedExecution,
) -> list[dict[str, Any]]:
    heads = _all_heads(prepared)
    rows: list[dict[str, Any]] = []
    graph_ids = list(prepared.splits.clean_ablation)
    missing: list[int] = []
    cached_by_graph: dict[int, Any] = {}
    for graph_id in graph_ids:
        name = f"graph_{int(graph_id):06d}"
        cached = (
            cache.load("focused/clean_ablation", name, strict=True)
            if config.resume and not config.force
            else None
        )
        if cached is None:
            missing.append(int(graph_id))
        else:
            cached_by_graph[int(graph_id)] = cached
    for graph_id in sorted(cached_by_graph):
        rows.extend(cached_by_graph[graph_id]["rows"])

    graph_batch = max(1, int(config.execution.graphs_per_batch))
    for graph_start in range(0, len(missing), graph_batch):
        selected_ids = missing[graph_start : graph_start + graph_batch]
        bases = [prepared.grit.eval_ds[int(graph_id)] for graph_id in selected_ids]
        _, z_clean, _ = prepared.backend.ablate(bases, ())
        clean_np = z_clean.detach().cpu().numpy()
        matrix = np.full((len(selected_ids), len(heads)), np.nan, dtype=np.float64)

        def process_head_chunk(selected_heads: Sequence[tuple[int, int]], offset: int):
            import torch

            try:
                targets = [base for _head in selected_heads for base in bases]
                assignments = [head for head in selected_heads for _base in bases]
                _, z_ablated, _ = prepared.backend.ablate_individual_heads(
                    targets, assignments
                )
                values = z_ablated.detach().cpu().numpy().reshape(
                    len(selected_heads), len(bases), -1
                )
                movement = np.linalg.norm(values - clean_np[None, :, :], axis=-1)
                matrix[:, offset : offset + len(selected_heads)] = movement.T
            except torch.cuda.OutOfMemoryError:
                if len(selected_heads) <= 1:
                    raise
                torch.cuda.empty_cache()
                first, second = _split_head_chunk(selected_heads)
                process_head_chunk(first, offset)
                if second:
                    process_head_chunk(second, offset + len(first))

        for head_start in range(0, len(heads), int(execution.head_batch_size)):
            process_head_chunk(
                heads[head_start : head_start + int(execution.head_batch_size)],
                head_start,
            )
        for graph_position, graph_id in enumerate(selected_ids):
            graph_rows = [
                {
                    "graph": int(graph_id),
                    "head": head,
                    "head_name": _head_name(head),
                    "prediction_movement": float(matrix[graph_position, position]),
                }
                for position, head in enumerate(heads)
            ]
            payload = {"graph": int(graph_id), "rows": graph_rows}
            cache.save(
                "focused/clean_ablation", f"graph_{int(graph_id):06d}", payload
            )
            rows.extend(graph_rows)
    return rows


def _matching_positions(
    matching: Mapping[str, Any],
    head_position: Mapping[tuple[int, int], int],
    left_key: str,
    right_key: str,
) -> tuple[list[int], list[int]]:
    left, right = [], []
    for pair in matching["pairs"]:
        a, b = _head_tuple(pair[left_key]), _head_tuple(pair[right_key])
        if a in head_position and b in head_position:
            left.append(head_position[a])
            right.append(head_position[b])
    return left, right


def _causal_interval(
    rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    config: MethodologyConfig,
    *,
    metric_order: Sequence[str],
    require_controlled: bool,
) -> dict[str, Any] | None:
    if gate["status"] != "estimable":
        return None
    head_order = _target_heads(gate)
    if not head_order:
        return None
    head_position = {head: position for position, head in enumerate(head_order)}
    by_channel: dict[str, dict[tuple[int, int, int], np.ndarray]] = {
        channel: {} for channel in CHANNELS
    }
    for row in rows:
        if require_controlled and not bool(row["controlled"]):
            continue
        values = np.asarray([float(row[name]) for name in metric_order])
        if not np.isfinite(values).all():
            continue
        key = (int(row["graph"]), int(row["source"]), int(row["donor"]))
        channel = str(row["channel"])
        matrix = by_channel[channel].setdefault(
            key,
            np.full((len(head_order), len(metric_order)), np.nan, dtype=np.float64),
        )
        matrix[head_position[_head_tuple(row["head"])]] = values
    observations: dict[str, list[Observation]] = {}
    for channel in CHANNELS:
        observations[channel] = [
            Observation(0, key[0], key[1], key[2], matrix)
            for key, matrix in sorted(by_channel[channel].items())
            if np.isfinite(matrix).all()
        ]
    if any(not observations[channel] for channel in CHANNELS):
        return None

    sem_positions, str_positions = _matching_positions(
        gate["specialist_J_matching"],
        head_position,
        "semantic",
        "structural",
    )
    sem_null_left, sem_null_right = _matching_positions(
        gate["semantic_null_J_matching"],
        head_position,
        "specialist",
        "null",
    )
    str_null_left, str_null_right = _matching_positions(
        gate["structural_null_J_matching"],
        head_position,
        "specialist",
        "null",
    )

    def mean_at(value: np.ndarray, channel: int, positions: Sequence[int], metric: int):
        return float(np.mean(value[channel, list(positions), metric])) if positions else np.nan

    def transform(value: np.ndarray) -> np.ndarray:
        cells = np.full((len(metric_order), 5), np.nan, dtype=np.float64)
        nulls = np.full((len(metric_order), 2), np.nan, dtype=np.float64)
        for metric in range(len(metric_order)):
            sem_sem = mean_at(value, 0, sem_positions, metric)
            sem_str = mean_at(value, 1, sem_positions, metric)
            str_sem = mean_at(value, 0, str_positions, metric)
            str_str = mean_at(value, 1, str_positions, metric)
            cells[metric] = (
                sem_sem,
                sem_str,
                str_sem,
                str_str,
                (sem_sem - sem_str) - (str_sem - str_str),
            )
            if sem_null_left:
                nulls[metric, 0] = float(
                    np.mean(
                        value[0, sem_null_left, metric]
                        - value[0, sem_null_right, metric]
                    )
                )
            if str_null_left:
                nulls[metric, 1] = float(
                    np.mean(
                        value[1, str_null_left, metric]
                        - value[1, str_null_right, metric]
                    )
                )
        return np.concatenate((value.reshape(-1), cells.reshape(-1), nulls.reshape(-1)))

    interval = paired_channel_percentile_interval(
        observations["semantic"],
        observations["structural"],
        config.bootstrap,
        transform=transform,
        resample_source=(True, True),
    )
    head_count = len(head_order) * len(metric_order) * len(CHANNELS)
    cell_count = len(metric_order) * 5

    def split(value: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        array = np.asarray(value, dtype=np.float64)
        heads = array[:head_count].reshape(len(CHANNELS), len(head_order), len(metric_order))
        cells = array[head_count : head_count + cell_count].reshape(len(metric_order), 5)
        nulls = array[head_count + cell_count :].reshape(len(metric_order), 2)
        return heads, cells, nulls

    estimate = split(interval.estimate)
    low = split(interval.low)
    high = split(interval.high)
    return {
        "metric_order": tuple(metric_order),
        "channel_order": CHANNELS,
        "head_order": head_order,
        "cell_order": (
            "semantic_heads_on_semantic",
            "semantic_heads_on_structural",
            "structural_heads_on_semantic",
            "structural_heads_on_structural",
            "double_difference",
        ),
        "null_order": (
            "semantic_specialist_minus_opposite_sign_null_on_semantic",
            "structural_specialist_minus_opposite_sign_null_on_structural",
        ),
        "head_estimate": estimate[0],
        "head_low": low[0],
        "head_high": high[0],
        "cell_estimate": estimate[1],
        "cell_low": low[1],
        "cell_high": high[1],
        "null_estimate": estimate[2],
        "null_low": low[2],
        "null_high": high[2],
        "replicates": int(interval.replicates),
        "rng_seed": int(interval.rng_seed),
        "resampled_levels": interval.resampled_levels,
        "event_counts": {channel: len(observations[channel]) for channel in CHANNELS},
    }


def _standardized_layer_coefficient(x: np.ndarray, y: np.ndarray, layers: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, layers = x[mask], y[mask], layers[mask]
    if len(x) < 3 or float(np.std(x)) <= 0 or float(np.std(y)) <= 0:
        return np.nan
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    unique = sorted(set(int(value) for value in layers))
    indicators = (
        np.column_stack([(layers == value).astype(float) for value in unique[1:]])
        if len(unique) > 1
        else np.empty((len(x), 0))
    )
    design = np.column_stack((np.ones(len(x)), x, indicators))
    return float(np.linalg.lstsq(design, y, rcond=None)[0][1])


def _clean_ablation_interval(
    rows: Sequence[Mapping[str, Any]],
    scores: Mapping[str, Any],
    config: MethodologyConfig,
) -> dict[str, Any]:
    from scipy.stats import spearmanr

    coordinates = scores["coordinates"]
    J = np.asarray(coordinates.joint_sensitivity, dtype=np.float64).reshape(-1)
    shape = coordinates.joint_sensitivity.shape
    heads = tuple((layer, head) for layer in range(shape[0]) for head in range(shape[1]))
    position = {head: index for index, head in enumerate(heads)}
    layers = np.repeat(np.arange(shape[0]), shape[1])
    by_graph: dict[int, np.ndarray] = {}
    for row in rows:
        matrix = by_graph.setdefault(
            int(row["graph"]), np.full(len(heads), np.nan, dtype=np.float64)
        )
        matrix[position[_head_tuple(row["head"])]] = float(row["prediction_movement"])
    observations = [
        Observation(0, graph, 0, 0, value)
        for graph, value in sorted(by_graph.items())
        if np.isfinite(value).all()
    ]
    if not observations:
        raise ValueError("clean ablation cache has no complete graph/head matrix")

    def transform(value: np.ndarray) -> np.ndarray:
        rho = float(spearmanr(J, value).statistic)
        beta = _standardized_layer_coefficient(J, value, layers)
        return np.concatenate((value, np.asarray((rho, beta))))

    interval = nested_percentile_interval(
        observations, config.bootstrap, transform=transform
    )
    count = len(heads)
    return {
        "head_order": heads,
        "J": J,
        "layers": layers,
        "prediction_movement": interval.estimate[:count],
        "prediction_movement_low": interval.low[:count],
        "prediction_movement_high": interval.high[:count],
        "spearman_rho": float(interval.estimate[count]),
        "spearman_low": float(interval.low[count]),
        "spearman_high": float(interval.high[count]),
        "layer_adjusted_standardized_beta": float(interval.estimate[count + 1]),
        "layer_adjusted_low": float(interval.low[count + 1]),
        "layer_adjusted_high": float(interval.high[count + 1]),
        "graph_count": len(observations),
        "head_count": count,
        "replicates": int(interval.replicates),
        "rng_seed": int(interval.rng_seed),
    }


def aggregate_core_tests(
    causal_rows: Sequence[Mapping[str, Any]],
    clean_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    scores: Mapping[str, Any],
    config: MethodologyConfig,
    causal_audits: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    controlled = [row for row in causal_rows if bool(row["controlled"])]
    dose_gaps = np.asarray(
        [float(row["absolute_dose_gap"]) for row in controlled], dtype=np.float64
    )
    unique_events = {
        (row["channel"], int(row["graph"]), int(row["source"]), int(row["donor"]))
        for row in causal_rows
    }
    controlled_events = {
        (row["channel"], int(row["graph"]), int(row["source"]), int(row["donor"]))
        for row in controlled
    }
    return {
        "version": FOCUSED_CAUSAL_VERSION,
        "sample_sizes": {
            "discovery_molecules": int(config.sizes.discovery_graphs),
            "causal_molecules": int(config.sizes.causal_graphs),
            "clean_ablation_molecules": int(config.sizes.clean_ablation_graphs),
            "sources_per_molecule": int(config.sizes.sources_per_graph),
            "donors_per_source": int(config.sizes.donors_per_source),
        },
        "gate": gate,
        "target_heads": _target_heads(gate),
        "patch": _causal_interval(
            causal_rows,
            gate,
            config,
            metric_order=PATCH_METRICS,
            require_controlled=True,
        ),
        "necessity": _causal_interval(
            causal_rows,
            gate,
            config,
            metric_order=NECESSITY_METRICS,
            require_controlled=False,
        ),
        "clean_ablation": _clean_ablation_interval(clean_rows, scores, config),
        "control_audit": {
            "event_count": len(unique_events),
            "controlled_event_count": len(controlled_events),
            "uncontrolled_event_count": len(unique_events - controlled_events),
            "controlled_fraction": (
                len(controlled_events) / len(unique_events) if unique_events else np.nan
            ),
            "dose_gap_mean": float(np.mean(dose_gaps)) if dose_gaps.size else np.nan,
            "dose_gap_median": float(np.median(dose_gaps)) if dose_gaps.size else np.nan,
            "dose_gap_max": float(np.max(dose_gaps)) if dose_gaps.size else np.nan,
            "dose_caliper": None,
            "clean_same_condition_patch_max": (
                max(
                    float(row["clean_same_condition_patch_max"])
                    for row in causal_audits
                )
                if causal_audits
                else np.nan
            ),
            "event_same_condition_patch_max": (
                max(
                    float(row["event_same_condition_patch_max"])
                    for row in causal_audits
                )
                if causal_audits
                else np.nan
            ),
            "per_graph_channel": tuple(causal_audits),
        },
    }


def run_focused_analysis(
    prepared: Any,
    config: MethodologyConfig,
    scores: Mapping[str, Any],
    gate: Mapping[str, Any],
    *,
    execution: FocusedExecution = FocusedExecution(),
) -> dict[str, Any]:
    """Run/resume selected-head causal events and every-head clean ablation."""

    execution.validate()
    plan, cache = _focused_causal_cache(prepared, config, gate)
    del plan
    if config.resume and not config.force:
        cached = cache.load("focused", "core_tests", strict=True)
        if cached is not None:
            return cached
    causal_rows, cache, causal_audits = run_causal_events(
        prepared, config, gate, execution=execution
    )
    clean_rows = _clean_ablation_graphs(
        prepared, config, cache, execution
    )
    core = aggregate_core_tests(
        causal_rows,
        clean_rows,
        gate,
        scores,
        config,
        causal_audits,
    )
    core["execution"] = dataclasses.asdict(execution)
    cache.save("focused", "core_tests", core)
    return core


def _artifact_paths(root: Path) -> dict[str, Path]:
    base = root / "graphormer_pcqm4mv2" / "seed_0"
    return {
        "model": base / "model.json",
        "scores": base / "cache" / "focused" / "scores" / "raw_inputs_v1.pt",
        "gate": base / "cache" / "focused" / "gate.pt",
        "core": base / "cache" / "focused" / "core_tests.pt",
        "figures": base / "figures" / "focused_causal",
        "manifest": base / "focused_causal_manifest.json",
    }


def render_cached_focused_figures(output_dir: str | Path) -> dict[str, Any]:
    """Render every focused figure from CPU caches without loading Graphormer."""

    from .graphormer_causal_plots import render_focused_figure_suite

    paths = _artifact_paths(Path(output_dir))
    missing = [name for name in ("scores", "gate", "core") if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(
            "figures-only focused run is missing "
            + ", ".join(str(paths[name]) for name in missing)
        )
    scores_artifact = load_cache_artifact_file(paths["scores"])
    gate_artifact = load_cache_artifact_file(paths["gate"])
    core_artifact = load_cache_artifact_file(paths["core"])
    scores = scores_artifact.value
    gate = gate_artifact.value
    core = core_artifact.value
    if gate.get("score_manifest_hash") != scores.get("manifest_hash"):
        raise RuntimeError(
            "focused gate and canonical score cache have different event manifests"
        )
    if core.get("version") != FOCUSED_CAUSAL_VERSION:
        raise RuntimeError(
            f"focused core cache uses {core.get('version')!r}; "
            f"expected {FOCUSED_CAUSAL_VERSION!r}"
        )
    if tuple(map(tuple, core.get("target_heads", ()))) != _target_heads(gate):
        raise RuntimeError("focused core cache and gate have different target heads")
    model_record = (
        json.loads(paths["model"].read_text(encoding="utf-8"))
        if paths["model"].is_file()
        else {}
    )
    figures = render_focused_figure_suite(
        scores,
        gate,
        core,
        output_dir=paths["figures"],
        common_metadata={
            "analysis_version": FOCUSED_CAUSAL_VERSION,
            "score_cache": str(paths["scores"]),
            "score_cache_sha256": scores_artifact.file_sha256,
            "score_contract_fingerprint": scores_artifact.metadata[
                "contract_fingerprint"
            ],
            "gate_cache": str(paths["gate"]),
            "gate_cache_sha256": gate_artifact.file_sha256,
            "core_cache": str(paths["core"]),
            "core_cache_sha256": core_artifact.file_sha256,
            "checkpoint_sha256": model_record.get("checkpoint_sha256"),
        },
    )
    manifest = {
        "analysis_version": FOCUSED_CAUSAL_VERSION,
        "model_record": str(paths["model"]),
        "score_cache": str(paths["scores"]),
        "gate_cache": str(paths["gate"]),
        "core_cache": str(paths["core"]),
        "sample_sizes": core["sample_sizes"],
        "gate_status": gate["status"],
        "target_heads": [list(head) for head in core["target_heads"]],
        "control_audit": core["control_audit"],
        "figures": figures,
    }
    atomic_json(paths["manifest"], manifest)
    return manifest


def run(
    *,
    phase: str,
    output_dir: str,
    dataset_root: str,
    cache_dir: str,
    accelerator: str = "cuda:0",
    force: bool = False,
    graphs_per_batch: int = 2,
    head_batch_size: int = 4,
    event_batch_size: int = 8,
) -> dict[str, Any]:
    """Colab-facing run/all/figures dispatcher with Drive-backed caches."""

    phase = str(phase).lower()
    if phase not in {"run", "figures", "all"}:
        raise ValueError("focused Graphormer phase must be 'run', 'figures', or 'all'")
    root = Path(output_dir)
    if phase == "figures":
        return render_cached_focused_figures(root)

    from .runner import _stage_plan, prepare_task, run_scores

    config = production_config(
        output_dir=str(root),
        dataset_root=dataset_root,
        cache_dir=cache_dir,
        accelerator=accelerator,
        force=force,
        graphs_per_batch=graphs_per_batch,
    )
    protocol = config.record()
    protocol.update(
        {
            "focused_analysis_version": FOCUSED_CAUSAL_VERSION,
            "focused_phase": phase,
            "focused_execution": {
                "head_batch_size": int(head_batch_size),
                "event_batch_size": int(event_batch_size),
            },
        }
    )
    atomic_json(root / "focused_causal_protocol.json", protocol)
    prepared = prepare_task(config, "graphormer_pcqm4mv2", 0)
    score_plan = _stage_plan(prepared, config, "scores")
    scores = run_scores(prepared, config, plan=score_plan, focused_only=True)
    gate = run_confidence_gate(prepared, config, scores)
    core = run_focused_analysis(
        prepared,
        config,
        scores,
        gate,
        execution=FocusedExecution(
            head_batch_size=int(head_batch_size),
            event_batch_size=int(event_batch_size),
        ),
    )
    result = {
        "analysis_version": FOCUSED_CAUSAL_VERSION,
        "output_dir": str(root),
        "gate_status": gate["status"],
        "target_heads": core["target_heads"],
        "sample_sizes": core["sample_sizes"],
        "control_audit": core["control_audit"],
    }
    if phase == "all":
        result["manifest"] = render_cached_focused_figures(root)
    return result


__all__ = [
    "FOCUSED_CAUSAL_VERSION",
    "FocusedExecution",
    "aggregate_core_tests",
    "production_config",
    "render_cached_focused_figures",
    "run",
    "run_confidence_gate",
    "run_focused_analysis",
]
