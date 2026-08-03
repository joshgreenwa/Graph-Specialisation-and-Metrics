"""Population-level causal tests for the official PCQM4Mv2 Graphormer.

The focused causal runner remains the source of the finite patching and clean-
ablation measurements.  This additive analysis freezes a larger, explicitly
J-matched head population from discovery scores, reuses compatible per-graph
focused shards, computes only missing heads, and produces the three primary
causal tests used in the paper figure.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..carriage.env import log
from .bootstrap import Observation, paired_channel_percentile_interval
from .cache import (
    CanonicalCache,
    StaleCacheError,
    atomic_json,
    load_cache_artifact_file,
)
from .graphormer_causal_analysis import (
    NECESSITY_METRICS,
    FocusedExecution,
    _causal_graph_channel,
    _clean_ablation_graphs,
    _clean_ablation_interval,
    _focused_causal_cache,
    _head_tuple,
    production_config,
    run_confidence_gate,
)
from .protocol import CHANNELS, MethodologyConfig, stable_hash

POPULATION_CAUSAL_VERSION = "graphormer-pcqm-causal-population-v1"


@dataclass(frozen=True)
class PopulationPolicy:
    """Discovery-only selection policy for the causal head populations."""

    head_pairs: int = 12
    minimum_pairs: int = 8
    candidate_pool_multiplier: int = 3
    layer_cost: float = 0.08
    null_layer_cost: float = 0.20
    null_selectivity_cost: float = 0.02

    def validate(self) -> None:
        if int(self.head_pairs) < 1:
            raise ValueError("population head_pairs must be positive")
        if not 1 <= int(self.minimum_pairs) <= int(self.head_pairs):
            raise ValueError("minimum_pairs must lie in [1, head_pairs]")
        if int(self.candidate_pool_multiplier) < 1:
            raise ValueError("candidate_pool_multiplier must be positive")
        if (
            float(self.layer_cost) < 0
            or float(self.null_layer_cost) < 0
            or float(self.null_selectivity_cost) < 0
        ):
            raise ValueError("matching costs must be non-negative")


DEFAULT_POPULATION_POLICY = PopulationPolicy()


def _standardized_difference(left: Sequence[float], right: Sequence[float]) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if not left.size or not right.size:
        return np.nan
    scale = float(np.sqrt(0.5 * (np.var(left) + np.var(right))))
    return (
        float((np.mean(left) - np.mean(right)) / scale)
        if np.isfinite(scale) and scale > 0
        else np.nan
    )


def _matching_diagnostics(
    pairs: Sequence[Mapping[str, Any]],
    *,
    left_j: str,
    right_j: str,
    reference_scale: float | None = None,
) -> dict[str, Any]:
    left = np.asarray([float(row[left_j]) for row in pairs], dtype=np.float64)
    right = np.asarray([float(row[right_j]) for row in pairs], dtype=np.float64)
    gaps = np.abs(left - right)
    layer_gaps = np.asarray([float(row["absolute_layer_gap"]) for row in pairs], dtype=np.float64)
    scale = float(reference_scale) if reference_scale is not None else np.nan
    standardized = (
        float((np.mean(left) - np.mean(right)) / scale)
        if left.size and right.size and np.isfinite(scale) and scale > 0
        else _standardized_difference(left, right)
    )
    return {
        "pair_count": len(pairs),
        "mean_left_J": float(np.mean(left)) if left.size else np.nan,
        "mean_right_J": float(np.mean(right)) if right.size else np.nan,
        "standardized_J_difference": standardized,
        "standardization_scale": (
            "pre-match eligible-head J SD"
            if reference_scale is not None
            else "post-match pooled within-family J SD"
        ),
        "mean_absolute_J_gap": float(np.mean(gaps)) if gaps.size else np.nan,
        "maximum_absolute_J_gap": float(np.max(gaps)) if gaps.size else np.nan,
        "mean_absolute_layer_gap": (float(np.mean(layer_gaps)) if layer_gaps.size else np.nan),
        "exact_layer_fraction": (float(np.mean(layer_gaps == 0)) if layer_gaps.size else np.nan),
    }


def build_population_gate(
    scores: Mapping[str, Any],
    confidence_gate: Mapping[str, Any],
    policy: PopulationPolicy = DEFAULT_POPULATION_POLICY,
    *,
    analysis_version: str = POPULATION_CAUSAL_VERSION,
) -> dict[str, Any]:
    """Freeze adequately sized directional and neutral populations without outcomes.

    Candidate pools contain the strongest ``3K`` point-active heads on each side
    of the registered D_rel margin.  Hungarian assignment is performed on J with
    a small layer-distance tie-break, and the best-balanced K pairs are retained.
    Every retained specialist is then matched to a distinct non-selected head.
    """

    from scipy.optimize import linear_sum_assignment

    policy.validate()
    coordinates = scores["coordinates"]
    J = np.asarray(coordinates.joint_sensitivity, dtype=np.float64)
    D = np.asarray(coordinates.selectivity, dtype=np.float64)
    active = np.asarray(coordinates.active, dtype=bool)
    threshold = float(confidence_gate["preference_threshold"])
    activity_floor = float(confidence_gate["activity_floor"])
    finite = np.isfinite(J) & np.isfinite(D)
    eligible = active & finite & (J >= activity_floor)

    def heads(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
        return tuple((int(layer), int(head)) for layer, head in np.argwhere(mask).tolist())

    semantic_all = sorted(
        heads(eligible & (D > threshold)), key=lambda item: (-float(D[item]), item)
    )
    structural_all = sorted(
        heads(eligible & (D < -threshold)), key=lambda item: (float(D[item]), item)
    )
    pool_size = int(policy.head_pairs) * int(policy.candidate_pool_multiplier)
    semantic_pool = tuple(semantic_all[:pool_size])
    structural_pool = tuple(structural_all[:pool_size])
    joint_scale = float(np.std([float(J[head]) for head in semantic_pool + structural_pool]))
    joint_scale = joint_scale if np.isfinite(joint_scale) and joint_scale > 0 else 1.0
    layer_scale = max(1, J.shape[0] - 1)
    candidate_pairs: list[dict[str, Any]] = []
    if semantic_pool and structural_pool:
        j_cost = np.asarray(
            [[abs(float(J[a]) - float(J[b])) for b in structural_pool] for a in semantic_pool],
            dtype=np.float64,
        )
        layer_gap = np.asarray(
            [[abs(int(a[0]) - int(b[0])) for b in structural_pool] for a in semantic_pool],
            dtype=np.float64,
        )
        cost = j_cost / joint_scale + float(policy.layer_cost) * layer_gap / layer_scale
        sem_positions, str_positions = linear_sum_assignment(cost)
        for sem_position, str_position in zip(sem_positions.tolist(), str_positions.tolist()):
            semantic = semantic_pool[sem_position]
            structural = structural_pool[str_position]
            candidate_pairs.append(
                {
                    "semantic": semantic,
                    "structural": structural,
                    "semantic_J": float(J[semantic]),
                    "structural_J": float(J[structural]),
                    "semantic_D_rel": float(D[semantic]),
                    "structural_D_rel": float(D[structural]),
                    "absolute_J_gap": float(j_cost[sem_position, str_position]),
                    "absolute_layer_gap": abs(int(semantic[0]) - int(structural[0])),
                    "matching_cost": float(cost[sem_position, str_position]),
                    "pair_strength": min(abs(float(D[semantic])), abs(float(D[structural]))),
                }
            )
    candidate_targets = tuple(
        _head_tuple(row[key]) for row in candidate_pairs for key in ("semantic", "structural")
    )
    candidate_target_set = set(candidate_targets)
    null_pool = tuple(
        sorted(
            (head for head in heads(eligible) if head not in candidate_target_set),
            key=lambda item: (abs(float(D[item])), item),
        )
    )

    def match_nulls(
        targets: Sequence[tuple[int, int]],
        pool: Sequence[tuple[int, int]],
    ) -> dict[tuple[int, int], dict[str, Any]]:
        if not targets or not pool:
            return {}
        null_j_scale = float(np.std([float(J[head]) for head in tuple(targets) + tuple(pool)]))
        null_j_scale = null_j_scale if np.isfinite(null_j_scale) and null_j_scale > 0 else 1.0
        null_d_scale = max(
            threshold,
            float(np.nanmax(np.abs(D[eligible]))) if bool(np.any(eligible)) else threshold,
        )
        null_j_cost = np.asarray(
            [[abs(float(J[target]) - float(J[null])) for null in pool] for target in targets],
            dtype=np.float64,
        )
        null_layer_gap = np.asarray(
            [[abs(int(target[0]) - int(null[0])) for null in pool] for target in targets],
            dtype=np.float64,
        )
        null_direction_cost = np.asarray(
            [[abs(float(D[null])) for null in pool] for _target in targets],
            dtype=np.float64,
        )
        null_cost = (
            null_j_cost / null_j_scale
            + float(policy.null_layer_cost) * null_layer_gap / layer_scale
            + float(policy.null_selectivity_cost) * null_direction_cost / null_d_scale
        )
        target_positions, null_positions = linear_sum_assignment(null_cost)
        result: dict[tuple[int, int], dict[str, Any]] = {}
        for target_position, null_position in zip(
            target_positions.tolist(), null_positions.tolist()
        ):
            target = targets[target_position]
            result[target] = {
                "target": target,
                "null": pool[null_position],
                "target_J": float(J[target]),
                "null_J": float(J[pool[null_position]]),
                "target_D_rel": float(D[target]),
                "null_D_rel": float(D[pool[null_position]]),
                "absolute_J_gap": float(null_j_cost[target_position, null_position]),
                "absolute_layer_gap": int(null_layer_gap[target_position, null_position]),
                "matching_cost": float(null_cost[target_position, null_position]),
            }
        return result

    null_by_target = match_nulls(candidate_targets, null_pool)
    complete_candidates = []
    for row in candidate_pairs:
        semantic_head = _head_tuple(row["semantic"])
        structural_head = _head_tuple(row["structural"])
        if semantic_head not in null_by_target or structural_head not in null_by_target:
            continue
        completed = dict(row)
        completed["null_matching_cost"] = float(
            null_by_target[semantic_head]["matching_cost"]
            + null_by_target[structural_head]["matching_cost"]
        )
        completed["total_balance_cost"] = float(
            completed["matching_cost"] + 0.5 * completed["null_matching_cost"]
        )
        complete_candidates.append(completed)
    complete_candidates.sort(
        key=lambda row: (
            float(row["total_balance_cost"]),
            -float(row["pair_strength"]),
            tuple(row["semantic"]),
            tuple(row["structural"]),
        )
    )
    specialist_pairs = tuple(complete_candidates[: int(policy.head_pairs)])
    semantic = tuple(_head_tuple(row["semantic"]) for row in specialist_pairs)
    structural = tuple(_head_tuple(row["structural"]) for row in specialist_pairs)
    selected = semantic + structural
    selected_set = set(selected)
    final_null_pool = tuple(
        sorted(
            (head for head in heads(eligible) if head not in selected_set),
            key=lambda item: (abs(float(D[item])), item),
        )
    )
    final_null_by_target = match_nulls(selected, final_null_pool)
    null_pairs = tuple(
        final_null_by_target[head] for head in selected if head in final_null_by_target
    )
    null_heads = tuple(_head_tuple(row["null"]) for row in null_pairs)

    confidence_semantic = {
        _head_tuple(head) for head in confidence_gate["heads"]["semantic_specialist"]
    }
    confidence_structural = {
        _head_tuple(head) for head in confidence_gate["heads"]["structural_specialist"]
    }
    pair_count = len(specialist_pairs)
    complete_nulls = len(null_pairs) == 2 * pair_count
    eligible_j_scale = float(np.std(J[eligible]))
    return {
        "version": str(analysis_version),
        "selection_uses_causal_outcomes": False,
        "policy": dataclasses.asdict(policy),
        "activity_floor": activity_floor,
        "preference_threshold": threshold,
        "candidate_rule": (
            "top 3K directional heads beyond the registered D_rel margin; "
            "minimum-cost J/layer specialist and null assignments; retain K "
            "complete triples with the best combined balance"
        ),
        "null_rule": (
            "distinct non-selected active heads; minimum-cost J/layer assignment "
            "without replacement, with a small near-zero-D_rel tie-break"
        ),
        "candidate_counts": {
            "semantic": len(semantic_all),
            "structural": len(structural_all),
            "semantic_pool": len(semantic_pool),
            "structural_pool": len(structural_pool),
            "null_pool": len(final_null_pool),
        },
        "specialist_pairs": specialist_pairs,
        "null_pairs": null_pairs,
        "heads": {
            "semantic": semantic,
            "structural": structural,
            "j_matched_null": null_heads,
        },
        "confidence_supported_selected": {
            "semantic": int(sum(head in confidence_semantic for head in semantic)),
            "structural": int(sum(head in confidence_structural for head in structural)),
        },
        "matching_balance": {
            "semantic_vs_structural": _matching_diagnostics(
                specialist_pairs,
                left_j="semantic_J",
                right_j="structural_J",
                reference_scale=eligible_j_scale,
            ),
            "specialists_vs_null": _matching_diagnostics(
                null_pairs,
                left_j="target_J",
                right_j="null_J",
                reference_scale=eligible_j_scale,
            ),
        },
        "status": (
            "estimable"
            if pair_count >= int(policy.minimum_pairs) and complete_nulls
            else "not_estimable"
        ),
        "status_reason": (
            None
            if pair_count >= int(policy.minimum_pairs) and complete_nulls
            else (
                f"requires at least {int(policy.minimum_pairs)} specialist pairs "
                "and one distinct J-match per selected specialist"
            )
        ),
    }


def _target_heads(gate: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    values = {
        _head_tuple(head)
        for family in ("semantic", "structural", "j_matched_null")
        for head in gate["heads"][family]
    }
    return tuple(sorted(values))


def _population_cache(
    prepared: Any,
    config: MethodologyConfig,
    gate: Mapping[str, Any],
    *,
    analysis_version: str = POPULATION_CAUSAL_VERSION,
) -> tuple[Mapping[int, Mapping[str, Any]], CanonicalCache]:
    from .runner import _cache, _stage_plan

    plan = _stage_plan(prepared, config, "causal")
    base = _cache(prepared, config, plan)
    manifest = {
        "version": str(analysis_version),
        "policy": gate["policy"],
        "specialist_pairs": gate["specialist_pairs"],
        "null_pairs": gate["null_pairs"],
        "metrics": ("R_align_adj", "I_align_adj", "N_fraction"),
    }
    fingerprint = stable_hash(
        {
            "base_event_manifest": base.contract.event_manifest_hash,
            "population_manifest": manifest,
        }
    )
    return plan, CanonicalCache(
        config.root,
        dataclasses.replace(base.contract, event_manifest_hash=fingerprint),
        stale_policy="archive",
    )


def _legacy_rows(
    path: Path,
    targets: set[tuple[int, int]],
    *,
    expected_contract_fingerprint: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        artifact = load_cache_artifact_file(path)
    except (StaleCacheError, OSError, RuntimeError) as error:
        log(f"[population] legacy shard ignored: {path} ({type(error).__name__})")
        return []
    stored_contract = dict(artifact.metadata.get("contract", {}))
    stored_contract.pop("repository_commit", None)
    if stable_hash(stored_contract) != str(expected_contract_fingerprint):
        log(
            "[population] legacy shard has a different scientific contract and "
            f"will not be reused: {path}"
        )
        return []
    return [
        dict(row) for row in artifact.value.get("rows", ()) if _head_tuple(row["head"]) in targets
    ]


def run_population_events(
    prepared: Any,
    config: MethodologyConfig,
    confidence_gate: Mapping[str, Any],
    population_gate: Mapping[str, Any],
    *,
    execution: FocusedExecution,
    analysis_version: str = POPULATION_CAUSAL_VERSION,
    plan: Mapping[int, Mapping[str, Any]] | None = None,
    cache: CanonicalCache | None = None,
    graph_ids: Sequence[int] | None = None,
    reuse_caches: Sequence[CanonicalCache] = (),
    reuse_legacy: bool = True,
) -> tuple[list[dict[str, Any]], CanonicalCache, list[dict[str, Any]]]:
    """Reuse complete focused rows and compute only missing population heads."""

    execution.validate()
    if cache is None:
        if plan is not None:
            raise ValueError("a supplied population plan requires its matching cache")
        plan, cache = _population_cache(
            prepared,
            config,
            population_gate,
            analysis_version=analysis_version,
        )
    else:
        plan = dict(plan or {})
    selected_graph_ids = tuple(
        sorted(int(graph_id) for graph_id in (graph_ids if graph_ids is not None else plan))
    )
    legacy_cache = None
    if reuse_legacy:
        _, legacy_cache = _focused_causal_cache(prepared, config, confidence_gate)
    targets = set(_target_heads(population_gate))
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    if population_gate["status"] != "estimable":
        return rows, cache, audits
    total_items = len(selected_graph_ids) * len(CHANNELS)
    completed_items = 0
    cache_hits = 0
    computed_items = 0
    started = time.monotonic()
    if prepared.progress is not None:
        prepared.progress.emit(
            "population_causal_plan",
            total_graphs=len(selected_graph_ids),
            total_graph_channels=total_items,
            total_heads=len(targets),
            head_batch_size=int(execution.head_batch_size),
            event_batch_size=int(execution.event_batch_size),
        )
    for graph_id in selected_graph_ids:
        for channel in CHANNELS:
            stage = f"focused_population/events/{channel}"
            name = f"graph_{int(graph_id):06d}"
            reusable_payloads = []
            for source_cache in reuse_caches:
                try:
                    reusable = source_cache.load(stage, name, strict=True)
                except StaleCacheError:
                    reusable = None
                if reusable is not None:
                    reusable_payloads.append(reusable)
            cached = None
            if not reusable_payloads and config.resume and not config.force:
                cached = cache.load(stage, name, strict=True)
            was_cached = cached is not None
            did_compute = False
            if cached is None:
                reused: list[dict[str, Any]] = []
                present: set[tuple[int, int]] = set()
                audit_source: Mapping[str, Any] = {}
                for reusable in reusable_payloads:
                    available = {
                        _head_tuple(row["head"])
                        for row in reusable.get("rows", ())
                    }
                    accepted = (targets - present) & available
                    if not accepted:
                        continue
                    reused.extend(
                        dict(row)
                        for row in reusable.get("rows", ())
                        if _head_tuple(row["head"]) in accepted
                    )
                    present.update(accepted)
                    if not audit_source:
                        audit_source = reusable
                if legacy_cache is not None and present != targets:
                    legacy_rows = _legacy_rows(
                        legacy_cache.path(f"focused/events/{channel}", name),
                        targets - present,
                        expected_contract_fingerprint=legacy_cache.contract.fingerprint,
                    )
                    legacy_heads = {_head_tuple(row["head"]) for row in legacy_rows}
                    reused.extend(legacy_rows)
                    present.update(legacy_heads)
                missing = tuple(sorted(targets - present))
                if missing and int(graph_id) not in plan:
                    from .runner import _stage_plan

                    plan.update(
                        _stage_plan(
                            prepared,
                            config,
                            "causal",
                            graph_ids=(int(graph_id),),
                            allow_empty=True,
                        )
                    )
                if missing and int(graph_id) not in plan:
                    log(
                        "[population] skipped a causal graph with no source "
                        f"estimable under both channels: {int(graph_id)}"
                    )
                    continue
                measured = (
                    _causal_graph_channel(
                        prepared,
                        config,
                        int(graph_id),
                        channel,
                        plan,
                        missing,
                        execution,
                    )
                    if missing
                    else {
                        "rows": [],
                        "event_count": int(audit_source.get("event_count", 0)),
                        "controlled_event_count": int(
                            audit_source.get("controlled_event_count", 0)
                        ),
                        "uncontrolled_event_count": int(
                            audit_source.get("uncontrolled_event_count", 0)
                        ),
                        "clean_same_condition_patch_max": float(
                            audit_source.get("clean_same_condition_patch_max", 0.0)
                        ),
                        "event_same_condition_patch_max": float(
                            audit_source.get("event_same_condition_patch_max", 0.0)
                        ),
                    }
                )
                did_compute = bool(missing)
                merged = reused + list(measured["rows"])
                cached = {
                    "graph": int(graph_id),
                    "channel": channel,
                    "rows": merged,
                    "target_head_count": len(targets),
                    "reused_head_count": len(present),
                    "computed_head_count": len(missing),
                    "event_count": int(measured.get("event_count", 0)),
                    "controlled_event_count": int(measured.get("controlled_event_count", 0)),
                    "uncontrolled_event_count": int(measured.get("uncontrolled_event_count", 0)),
                    "clean_same_condition_patch_max": float(
                        measured.get("clean_same_condition_patch_max", 0.0)
                    ),
                    "event_same_condition_patch_max": float(
                        measured.get("event_same_condition_patch_max", 0.0)
                    ),
                }
                cache.save(stage, name, cached)
                computed_items += int(did_compute)
            else:
                cache_hits += 1
            rows.extend(cached["rows"])
            audits.append(
                {
                    key: cached.get(key)
                    for key in (
                        "graph",
                        "channel",
                        "target_head_count",
                        "reused_head_count",
                        "computed_head_count",
                        "event_count",
                        "controlled_event_count",
                        "uncontrolled_event_count",
                        "clean_same_condition_patch_max",
                        "event_same_condition_patch_max",
                    )
                }
            )
            completed_items += 1
            elapsed = max(time.monotonic() - started, 1e-9)
            computed_rate = computed_items / elapsed if computed_items else 0.0
            overall_rate = completed_items / elapsed
            should_report = (
                not was_cached
                or completed_items == total_items
                or completed_items % 16 == 0
            )
            if should_report and prepared.progress is not None:
                remaining_items = max(0, total_items - completed_items)
                prepared.progress.emit(
                    "population_causal_progress",
                    graph=int(graph_id),
                    channel=str(channel),
                    completed_graph_channels=completed_items,
                    total_graph_channels=total_items,
                    cache_hits=cache_hits,
                    computed_graph_channels=computed_items,
                    reused_heads=int(cached.get("reused_head_count") or 0),
                    computed_heads=int(cached.get("computed_head_count") or 0),
                    computed_graph_channels_per_second=computed_rate,
                    graph_channels_per_second=overall_rate,
                    eta_seconds=remaining_items / overall_rate,
                )
    return rows, cache, audits


def _event_interval(
    rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    config: MethodologyConfig,
    *,
    metric_order: Sequence[str],
    require_controlled: bool,
    seed_offset: int,
    progress: Any | None = None,
) -> dict[str, Any]:
    head_order = _target_heads(gate)
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
        matrix = by_channel[str(row["channel"])].setdefault(
            key,
            np.full((len(head_order), len(metric_order)), np.nan, dtype=np.float64),
        )
        matrix[head_position[_head_tuple(row["head"])]] = values
    observations = {
        channel: [
            Observation(0, key[0], key[1], key[2], value)
            for key, value in sorted(by_channel[channel].items())
            if np.isfinite(value).all()
        ]
        for channel in CHANNELS
    }
    if any(not observations[channel] for channel in CHANNELS):
        raise ValueError("population causal cache has no complete channel event matrix")
    def report_draw(completed: int, total: int) -> None:
        if progress is not None and (
            completed == 1 or completed % 100 == 0 or completed == total
        ):
            progress.emit(
                "population_causal_bootstrap_progress",
                metrics=tuple(metric_order),
                completed_draws=int(completed),
                total_draws=int(total),
            )

    interval = paired_channel_percentile_interval(
        observations["semantic"],
        observations["structural"],
        config.bootstrap,
        transform=lambda value: value,
        resample_source=(True, True),
        retain_draws=True,
        on_draw=report_draw,
    )
    if interval.draws is None:
        raise RuntimeError("population event bootstrap did not retain draws")

    specialist_pairs = tuple(gate["specialist_pairs"])
    semantic_positions = np.asarray(
        [head_position[_head_tuple(row["semantic"])] for row in specialist_pairs],
        dtype=np.int64,
    )
    structural_positions = np.asarray(
        [head_position[_head_tuple(row["structural"])] for row in specialist_pairs],
        dtype=np.int64,
    )
    null_positions = np.asarray(
        [head_position[_head_tuple(row["null"])] for row in gate["null_pairs"]],
        dtype=np.int64,
    )

    def reduce(value: np.ndarray, family_positions: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([np.mean(value[:, positions, :], axis=1) for positions in family_positions])

    include_null = any(metric in NECESSITY_METRICS for metric in metric_order)
    families = [semantic_positions, structural_positions]
    family_order = ["semantic", "structural"]
    if include_null:
        families.append(null_positions)
        family_order.append("j_matched_null")
    estimate = reduce(np.asarray(interval.estimate), families)
    rng = np.random.default_rng(int(config.bootstrap.rng_seed) + int(seed_offset))
    head_draws = np.asarray(interval.draws, dtype=np.float64)
    family_draws = []
    for draw in head_draws:
        pair_sample = rng.integers(0, len(specialist_pairs), size=len(specialist_pairs))
        selected = [
            semantic_positions[pair_sample],
            structural_positions[pair_sample],
        ]
        if include_null:
            null_sample = rng.integers(0, len(null_positions), size=len(null_positions))
            selected.append(null_positions[null_sample])
        family_draws.append(reduce(draw, selected))
    family_draws = np.asarray(family_draws, dtype=np.float64)
    alpha = (1.0 - float(config.bootstrap.confidence)) / 2.0
    low = np.nanquantile(family_draws, alpha, axis=0)
    high = np.nanquantile(family_draws, 1.0 - alpha, axis=0)
    pairing_estimate = (estimate[0, 0] - estimate[0, 1]) - (
        estimate[1, 0] - estimate[1, 1]
    )
    pairing_draws = (family_draws[:, 0, 0] - family_draws[:, 0, 1]) - (
        family_draws[:, 1, 0] - family_draws[:, 1, 1]
    )
    pairing_low = np.nanquantile(pairing_draws, alpha, axis=0)
    pairing_high = np.nanquantile(pairing_draws, 1.0 - alpha, axis=0)
    return {
        "metric_order": tuple(metric_order),
        "family_order": tuple(family_order),
        "channel_order": CHANNELS,
        "estimate": estimate,
        "low": low,
        "high": high,
        "correct_pairing_advantage": pairing_estimate,
        "correct_pairing_low": pairing_low,
        "correct_pairing_high": pairing_high,
        "replicates": int(interval.replicates),
        "resampled_levels": interval.resampled_levels + ("matched head pair",),
        "event_counts": {channel: len(observations[channel]) for channel in CHANNELS},
        "head_pair_count": len(specialist_pairs),
        "null_head_count": len(null_positions),
        "head_order": head_order,
        "head_estimate": np.asarray(interval.estimate, dtype=np.float64),
        "head_draws": head_draws,
    }


def _endpoint(summary: Mapping[str, Any], metric: str) -> dict[str, Any]:
    position = tuple(summary["metric_order"]).index(metric)
    estimate = np.asarray(summary["estimate"])[:, :, position]
    low = np.asarray(summary["low"])[:, :, position]
    high = np.asarray(summary["high"])[:, :, position]
    interaction = (estimate[0, 0] - estimate[0, 1]) - (estimate[1, 0] - estimate[1, 1])
    return {
        "family_order": summary["family_order"],
        "channel_order": summary["channel_order"],
        "estimate": estimate,
        "low": low,
        "high": high,
        "double_difference": float(interaction),
        "correct_pairing_advantage": float(
            np.asarray(summary["correct_pairing_advantage"])[position]
        ),
        "correct_pairing_low": float(
            np.asarray(summary["correct_pairing_low"])[position]
        ),
        "correct_pairing_high": float(
            np.asarray(summary["correct_pairing_high"])[position]
        ),
        "replicates": summary["replicates"],
        "resampled_levels": summary["resampled_levels"],
        "event_counts": summary["event_counts"],
    }


def response_adjusted_components(
    selectivity: Sequence[float],
    preference: Sequence[float],
    mean_response: Sequence[float],
    layers: Sequence[int],
) -> dict[str, Any]:
    """Adjust causal preference for response magnitude and layer.

    The reported coefficient is the coefficient of globally standardized
    ``D_rel`` in a regression of standardized causal preference on ``D_rel``,
    mean absolute causal response, and layer fixed effects.  Treating response
    magnitude as a covariate avoids the instability of dividing by an effect
    that can be close to zero.
    """

    selectivity = np.asarray(selectivity, dtype=np.float64)
    preference = np.asarray(preference, dtype=np.float64)
    mean_response = np.asarray(mean_response, dtype=np.float64)
    layers = np.asarray(layers, dtype=np.int64)
    finite = (
        np.isfinite(selectivity)
        & np.isfinite(preference)
        & np.isfinite(mean_response)
    )
    selectivity = selectivity[finite]
    preference = preference[finite]
    mean_response = mean_response[finite]
    layers = layers[finite]
    if len(selectivity) < 4:
        return {"beta": np.nan, "head_count": len(selectivity)}

    def standardize(values: np.ndarray) -> np.ndarray | None:
        scale = float(np.std(values))
        if not np.isfinite(scale) or scale <= 0:
            return None
        return (values - np.mean(values)) / scale

    x = standardize(selectivity)
    y = standardize(preference)
    response = standardize(mean_response)
    if x is None or y is None or response is None:
        return {"beta": np.nan, "head_count": len(selectivity)}
    unique_layers = sorted({int(value) for value in layers})
    indicators = (
        np.column_stack(
            [(layers == value).astype(np.float64) for value in unique_layers[1:]]
        )
        if len(unique_layers) > 1
        else np.empty((len(selectivity), 0), dtype=np.float64)
    )
    nuisance = np.column_stack((np.ones(len(selectivity)), response, indicators))
    x_residual = x - nuisance @ np.linalg.lstsq(nuisance, x, rcond=None)[0]
    y_residual = y - nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    denominator = float(np.dot(x_residual, x_residual))
    beta = (
        float(np.dot(x_residual, y_residual) / denominator)
        if denominator > 0
        else np.nan
    )
    return {
        "beta": beta,
        "head_count": len(selectivity),
        "selectivity_residual": x_residual,
        "preference_residual": y_residual,
        "layers": layers,
        "definition": (
            "standardized D_rel coefficient after controlling for globally "
            "standardized mean absolute causal response and layer fixed effects"
        ),
    }


def _causal_preference_analysis(
    raw_summary: Mapping[str, Any],
    scores: Mapping[str, Any],
    population_gate: Mapping[str, Any],
    config: MethodologyConfig,
    *,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Relate discovery D_rel to semantic-minus-structural causal effects."""

    from scipy.stats import spearmanr

    head_order = tuple(_head_tuple(head) for head in raw_summary["head_order"])
    coordinates = scores["coordinates"]
    selectivity = np.asarray(
        [float(coordinates.selectivity[head]) for head in head_order],
        dtype=np.float64,
    )
    joint_sensitivity = np.asarray(
        [float(coordinates.joint_sensitivity[head]) for head in head_order],
        dtype=np.float64,
    )
    layers = np.asarray([head[0] for head in head_order], dtype=np.int64)
    head_position = {head: position for position, head in enumerate(head_order)}
    null_by_target = {
        _head_tuple(row["target"]): _head_tuple(row["null"])
        for row in population_gate["null_pairs"]
    }
    blocks = []
    for pair in population_gate["specialist_pairs"]:
        semantic = _head_tuple(pair["semantic"])
        structural = _head_tuple(pair["structural"])
        blocks.append(
            np.asarray(
                [
                    head_position[semantic],
                    head_position[structural],
                    head_position[null_by_target[semantic]],
                    head_position[null_by_target[structural]],
                ],
                dtype=np.int64,
            )
        )
    if not blocks:
        raise ValueError("causal preference analysis requires matched head blocks")
    blocks = np.stack(blocks)

    point = np.asarray(raw_summary["head_estimate"], dtype=np.float64)
    event_draws = np.asarray(raw_summary["head_draws"], dtype=np.float64)
    metric_order = tuple(raw_summary["metric_order"])
    alpha = (1.0 - float(config.bootstrap.confidence)) / 2.0
    rng = np.random.default_rng(int(config.bootstrap.rng_seed) + 421)
    endpoints: dict[str, Any] = {}
    for endpoint_name, metric in (
        ("restoration", "R_align_matched"),
        ("injection", "I_align_matched"),
    ):
        metric_position = metric_order.index(metric)
        head_effect = point[:, :, metric_position]
        preference = head_effect[0] - head_effect[1]
        mean_response = 0.5 * (np.abs(head_effect[0]) + np.abs(head_effect[1]))
        preference_draws = (
            event_draws[:, 0, :, metric_position]
            - event_draws[:, 1, :, metric_position]
        )
        response_draws = 0.5 * (
            np.abs(event_draws[:, 0, :, metric_position])
            + np.abs(event_draws[:, 1, :, metric_position])
        )
        raw_rho = float(spearmanr(selectivity, preference).statistic)
        adjusted = response_adjusted_components(
            selectivity,
            preference,
            mean_response,
            layers,
        )
        rho_draws = []
        beta_draws = []
        for draw_position in range(event_draws.shape[0]):
            sampled_blocks = blocks[
                rng.integers(0, len(blocks), size=len(blocks))
            ].reshape(-1)
            draw_preference = preference_draws[draw_position, sampled_blocks]
            draw_response = response_draws[draw_position, sampled_blocks]
            draw_selectivity = selectivity[sampled_blocks]
            draw_layers = layers[sampled_blocks]
            rho_draws.append(
                float(spearmanr(draw_selectivity, draw_preference).statistic)
            )
            beta_draws.append(
                float(
                    response_adjusted_components(
                        draw_selectivity,
                        draw_preference,
                        draw_response,
                        draw_layers,
                    )["beta"]
                )
            )
            completed = draw_position + 1
            if progress is not None and (
                completed == 1
                or completed % 100 == 0
                or completed == event_draws.shape[0]
            ):
                progress.emit(
                    "causal_preference_bootstrap_progress",
                    endpoint=endpoint_name,
                    completed_draws=completed,
                    total_draws=int(event_draws.shape[0]),
                )
        rho_draws = np.asarray(rho_draws, dtype=np.float64)
        beta_draws = np.asarray(beta_draws, dtype=np.float64)
        endpoints[endpoint_name] = {
            "causal_preference": preference,
            "preference_low": np.nanquantile(preference_draws, alpha, axis=0),
            "preference_high": np.nanquantile(
                preference_draws, 1.0 - alpha, axis=0
            ),
            "average_response": mean_response,
            "semantic_effect": head_effect[0],
            "structural_effect": head_effect[1],
            "spearman_rho": raw_rho,
            "spearman_low": float(np.nanquantile(rho_draws, alpha)),
            "spearman_high": float(np.nanquantile(rho_draws, 1.0 - alpha)),
            "response_layer_adjusted_beta": float(adjusted["beta"]),
            "response_layer_adjusted_low": float(
                np.nanquantile(beta_draws, alpha)
            ),
            "response_layer_adjusted_high": float(
                np.nanquantile(beta_draws, 1.0 - alpha)
            ),
            "response_adjusted_components": adjusted,
        }
    return {
        "version": "causal-preference-v1",
        "head_order": head_order,
        "selectivity": selectivity,
        "joint_sensitivity": joint_sensitivity,
        "layers": layers,
        "head_count": len(head_order),
        "graph_count": int(config.sizes.causal_graphs),
        "matched_block_count": len(blocks),
        "replicates": int(event_draws.shape[0]),
        "resampled_levels": tuple(
            level
            for level in raw_summary["resampled_levels"]
            if level != "matched head pair"
        )
        + ("matched four-head block",),
        "response_definition": (
            "mean absolute direction-aligned response across semantic and "
            "structural intervention channels"
        ),
        "preference_definition": (
            "semantic-event direction-aligned effect minus structural-event "
            "direction-aligned effect"
        ),
        "endpoints": endpoints,
    }


def layer_adjusted_components(
    x: Sequence[float], y: Sequence[float], layers: Sequence[int]
) -> dict[str, Any]:
    """Return the Frisch-Waugh-Lovell view of the reported layer-fixed-effect beta."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    layers = np.asarray(layers, dtype=np.int64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, layers = x[finite], y[finite], layers[finite]

    def partial(values_x: np.ndarray, values_y: np.ndarray, group: np.ndarray):
        x_standardized = (values_x - np.mean(values_x)) / np.std(values_x)
        y_standardized = (values_y - np.mean(values_y)) / np.std(values_y)
        unique_group = sorted({int(value) for value in group})
        indicators = (
            np.column_stack([(group == value).astype(float) for value in unique_group[1:]])
            if len(unique_group) > 1
            else np.empty((len(values_x), 0), dtype=np.float64)
        )
        nuisance = np.column_stack((np.ones(len(values_x)), indicators))
        x_residual = (
            x_standardized - nuisance @ np.linalg.lstsq(nuisance, x_standardized, rcond=None)[0]
        )
        y_residual = (
            y_standardized - nuisance @ np.linalg.lstsq(nuisance, y_standardized, rcond=None)[0]
        )
        denominator = float(np.dot(x_residual, x_residual))
        beta = float(np.dot(x_residual, y_residual) / denominator) if denominator > 0 else np.nan
        return x_residual, y_residual, beta

    unique = sorted({int(value) for value in layers})
    x_residual, y_residual, beta = partial(x, y, layers)
    leave_one_layer_out = {}
    for layer in unique:
        keep = layers != layer
        if int(np.sum(keep)) < 3 or len(set(layers[keep].tolist())) < 1:
            continue
        leave_one_layer_out[str(layer)] = float(partial(x[keep], y[keep], layers[keep])[2])
    return {
        "x_residual": x_residual,
        "y_residual": y_residual,
        "layers": layers,
        "beta": beta,
        "leave_one_layer_out": leave_one_layer_out,
        "definition": (
            "OLS coefficient of globally standardized J after layer fixed effects; "
            "equivalently the slope between layer-residualized standardized variables"
        ),
    }


def _add_clean_diagnostics(clean: Mapping[str, Any]) -> dict[str, Any]:
    from scipy.stats import spearmanr

    result = dict(clean)
    partial = layer_adjusted_components(clean["J"], clean["prediction_movement"], clean["layers"])
    # The cached beta and the residualized slope are algebraically identical.
    if not np.isclose(
        partial["beta"],
        float(clean["layer_adjusted_standardized_beta"]),
        rtol=1.0e-8,
        atol=1.0e-10,
    ):
        raise RuntimeError("layer-adjusted partial slope disagrees with cached beta")
    J = np.asarray(clean["J"], dtype=np.float64)
    movement = np.asarray(clean["prediction_movement"], dtype=np.float64)
    layers = np.asarray(clean["layers"], dtype=np.int64)
    partial["within_layer_spearman"] = {
        str(layer): float(spearmanr(J[layers == layer], movement[layers == layer]).statistic)
        for layer in sorted(set(layers.tolist()))
    }
    result["layer_adjusted_partial"] = partial
    return result


def aggregate_population_tests(
    causal_rows: Sequence[Mapping[str, Any]],
    clean_rows: Sequence[Mapping[str, Any]],
    population_gate: Mapping[str, Any],
    scores: Mapping[str, Any],
    config: MethodologyConfig,
    audits: Sequence[Mapping[str, Any]] = (),
    *,
    progress: Any | None = None,
    analysis_version: str = POPULATION_CAUSAL_VERSION,
) -> dict[str, Any]:
    raw_patch = _event_interval(
        causal_rows,
        population_gate,
        config,
        metric_order=("R_align_matched", "I_align_matched"),
        require_controlled=False,
        seed_offset=397,
        progress=progress,
    )
    mismatch_adjusted_patch = _event_interval(
        causal_rows,
        population_gate,
        config,
        metric_order=("R_align_adj", "I_align_adj"),
        require_controlled=True,
        seed_offset=401,
        progress=progress,
    )
    necessity = _event_interval(
        causal_rows,
        population_gate,
        config,
        metric_order=("N_fraction",),
        require_controlled=False,
        seed_offset=409,
        progress=progress,
    )
    causal_preference = _causal_preference_analysis(
        raw_patch,
        scores,
        population_gate,
        config,
        progress=progress,
    )
    return {
        "version": str(analysis_version),
        "sample_sizes": {
            "discovery_molecules": int(config.sizes.discovery_graphs),
            "causal_molecules": int(config.sizes.causal_graphs),
            "clean_ablation_molecules": int(config.sizes.clean_ablation_graphs),
            "sources_per_molecule": int(config.sizes.sources_per_graph),
            "donors_per_source": int(config.sizes.donors_per_source),
        },
        "population_gate": population_gate,
        "raw_primary_version": "donor-averaged-direction-aligned-v1",
        "raw_restoration": _endpoint(raw_patch, "R_align_matched"),
        "raw_injection": _endpoint(raw_patch, "I_align_matched"),
        "causal_preference": causal_preference,
        "restoration": _endpoint(mismatch_adjusted_patch, "R_align_adj"),
        "injection": _endpoint(mismatch_adjusted_patch, "I_align_adj"),
        "necessity": _endpoint(necessity, "N_fraction"),
        "clean_ablation": _add_clean_diagnostics(
            _clean_ablation_interval(clean_rows, scores, config, progress=progress)
        ),
        "execution_audit": {
            "shards": tuple(audits),
            "reused_head_shard_total": int(
                sum(int(row.get("reused_head_count") or 0) for row in audits)
            ),
            "computed_head_shard_total": int(
                sum(int(row.get("computed_head_count") or 0) for row in audits)
            ),
        },
    }


def run_population_analysis(
    prepared: Any,
    config: MethodologyConfig,
    scores: Mapping[str, Any],
    confidence_gate: Mapping[str, Any],
    population_gate: Mapping[str, Any],
    *,
    execution: FocusedExecution,
    analysis_version: str = POPULATION_CAUSAL_VERSION,
    plan: Mapping[int, Mapping[str, Any]] | None = None,
    cache: CanonicalCache | None = None,
    graph_ids: Sequence[int] | None = None,
    reuse_caches: Sequence[CanonicalCache] = (),
    reuse_legacy: bool = True,
    clean_rows: Sequence[Mapping[str, Any]] | None = None,
    clean_cache: CanonicalCache | None = None,
    clean_cache_stage: str = "focused/clean_ablation",
) -> dict[str, Any]:
    execution.validate()
    if cache is None:
        if plan is not None:
            raise ValueError("a supplied population plan requires its matching cache")
        plan, cache = _population_cache(
            prepared,
            config,
            population_gate,
            analysis_version=analysis_version,
        )
    if config.resume and not config.force:
        cached = cache.load("focused_population", "core_tests", strict=True)
        if (
            cached is not None
            and "raw_restoration" in cached
            and "correct_pairing_advantage" in cached.get("necessity", {})
            and "causal_preference" in cached
        ):
            log("[cache] loaded complete causal population analysis")
            if prepared.progress is not None:
                prepared.progress.emit(
                    "cache_hit",
                    phase="population_causal",
                    cache="core_tests",
                )
            return cached
        if cached is not None:
            log(
                "[cache] deriving donor-averaged primary endpoints from existing "
                "causal event shards"
            )
            causal_rows, cache, _audits = run_population_events(
                prepared,
                config,
                confidence_gate,
                population_gate,
                execution=execution,
                analysis_version=analysis_version,
                plan=plan,
                cache=cache,
                graph_ids=graph_ids,
                reuse_caches=reuse_caches,
                reuse_legacy=reuse_legacy,
            )
            raw_patch = _event_interval(
                causal_rows,
                population_gate,
                config,
                metric_order=("R_align_matched", "I_align_matched"),
                require_controlled=False,
                seed_offset=397,
                progress=prepared.progress,
            )
            cached_necessity = cached.get("necessity", {})
            if "correct_pairing_advantage" in cached_necessity:
                necessity_endpoint = cached_necessity
            else:
                necessity = _event_interval(
                    causal_rows,
                    population_gate,
                    config,
                    metric_order=("N_fraction",),
                    require_controlled=False,
                    seed_offset=409,
                    progress=prepared.progress,
                )
                necessity_endpoint = _endpoint(necessity, "N_fraction")
            causal_preference = _causal_preference_analysis(
                raw_patch,
                scores,
                population_gate,
                config,
                progress=prepared.progress,
            )
            upgraded = dict(cached)
            upgraded.update(
                {
                    "raw_primary_version": "donor-averaged-direction-aligned-v1",
                    "raw_restoration": _endpoint(raw_patch, "R_align_matched"),
                    "raw_injection": _endpoint(raw_patch, "I_align_matched"),
                    "causal_preference": causal_preference,
                    "necessity": necessity_endpoint,
                }
            )
            cache.save("focused_population", "core_tests", upgraded)
            if prepared.progress is not None:
                prepared.progress.emit(
                    "cache_upgrade_complete",
                    cache="core_tests",
                    added=(
                        "raw_restoration",
                        "raw_injection",
                        "causal_preference",
                        "necessity.correct_pairing_interval",
                    ),
                    model_forwards=0,
                )
            return upgraded
    if population_gate["status"] != "estimable":
        raise RuntimeError(population_gate["status_reason"])
    causal_rows, cache, audits = run_population_events(
        prepared,
        config,
        confidence_gate,
        population_gate,
        execution=execution,
        analysis_version=analysis_version,
        plan=plan,
        cache=cache,
        graph_ids=graph_ids,
        reuse_caches=reuse_caches,
        reuse_legacy=reuse_legacy,
    )
    if clean_rows is None:
        if clean_cache is None:
            _, clean_cache = _focused_causal_cache(prepared, config, confidence_gate)
        clean_rows = _clean_ablation_graphs(
            prepared,
            config,
            clean_cache,
            execution,
            cache_stage=clean_cache_stage,
        )
    core = aggregate_population_tests(
        causal_rows,
        clean_rows,
        population_gate,
        scores,
        config,
        audits,
        progress=prepared.progress,
        analysis_version=analysis_version,
    )
    core["execution"] = dataclasses.asdict(execution)
    cache.save("focused_population", "core_tests", core)
    return core


def _artifact_paths(root: Path) -> dict[str, Path]:
    base = root / "graphormer_pcqm4mv2" / "seed_0"
    return {
        "model": base / "model.json",
        "scores": base / "cache" / "focused" / "scores" / "raw_inputs_v1.pt",
        "confidence_gate": base / "cache" / "focused" / "gate.pt",
        "population_gate": base / "cache" / "focused_population" / "gate.pt",
        "core": base / "cache" / "focused_population" / "core_tests.pt",
        "figures": base / "figures" / "focused_causal_population",
        "manifest": base / "focused_causal_population_manifest.json",
    }


def render_cached_population_figures(output_dir: str | Path) -> dict[str, Any]:
    from .graphormer_causal_population_plots import render_population_figure_suite

    paths = _artifact_paths(Path(output_dir))
    required = ("scores", "population_gate", "core")
    missing = [name for name in required if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError(
            "population figures require a completed run; missing "
            + ", ".join(str(paths[name]) for name in missing)
        )
    artifacts = {name: load_cache_artifact_file(paths[name]) for name in required}
    scores = artifacts["scores"].value
    gate = artifacts["population_gate"].value
    core = artifacts["core"].value
    if gate.get("version") != POPULATION_CAUSAL_VERSION:
        raise RuntimeError("population gate uses a different analysis version")
    if core.get("version") != POPULATION_CAUSAL_VERSION:
        raise RuntimeError("population core uses a different analysis version")
    model_record = (
        json.loads(paths["model"].read_text(encoding="utf-8")) if paths["model"].is_file() else {}
    )
    figures = render_population_figure_suite(
        scores,
        gate,
        core,
        output_dir=paths["figures"],
        common_metadata={
            "analysis_version": POPULATION_CAUSAL_VERSION,
            "checkpoint_sha256": model_record.get("checkpoint_sha256"),
            "score_cache": str(paths["scores"]),
            "score_cache_sha256": artifacts["scores"].file_sha256,
            "population_gate_cache": str(paths["population_gate"]),
            "population_core_cache": str(paths["core"]),
            "population_core_sha256": artifacts["core"].file_sha256,
        },
    )
    manifest = {
        "analysis_version": POPULATION_CAUSAL_VERSION,
        "sample_sizes": core["sample_sizes"],
        "matching_balance": gate["matching_balance"],
        "selected_heads": gate["heads"],
        "execution_audit": core["execution_audit"],
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
    population_head_pairs: int = 12,
    population_minimum_pairs: int = 8,
    discovery_graphs: int = 128,
    causal_graphs: int = 128,
    clean_ablation_graphs: int = 128,
    semantic_donor_graphs: int = 2_000,
    sources_per_graph: int = 6,
    donors_per_source: int = 8,
) -> dict[str, Any]:
    """Colab-facing run/all/figures dispatcher for the population analysis."""

    phase = str(phase).lower()
    if phase not in {"run", "figures", "all"}:
        raise ValueError("phase must be 'run', 'figures', or 'all'")
    root = Path(output_dir)
    if phase == "figures":
        return render_cached_population_figures(root)

    from .runner import _stage_plan, prepare_task, run_scores

    config = production_config(
        output_dir=str(root),
        dataset_root=dataset_root,
        cache_dir=cache_dir,
        accelerator=accelerator,
        force=force,
        graphs_per_batch=graphs_per_batch,
        discovery_graphs=discovery_graphs,
        causal_graphs=causal_graphs,
        clean_ablation_graphs=clean_ablation_graphs,
        semantic_donor_graphs=semantic_donor_graphs,
        sources_per_graph=sources_per_graph,
        donors_per_source=donors_per_source,
    )
    policy = PopulationPolicy(
        head_pairs=int(population_head_pairs),
        minimum_pairs=int(population_minimum_pairs),
    )
    execution = FocusedExecution(
        head_batch_size=int(head_batch_size),
        event_batch_size=int(event_batch_size),
    )
    atomic_json(
        root / "focused_causal_population_protocol.json",
        {
            **config.record(),
            "analysis_version": POPULATION_CAUSAL_VERSION,
            "phase": phase,
            "population_policy": dataclasses.asdict(policy),
            "execution": dataclasses.asdict(execution),
        },
    )
    prepared = prepare_task(config, "graphormer_pcqm4mv2", 0)
    progress = prepared.progress
    if progress is not None:
        progress.update(task="graphormer_pcqm4mv2", train_seed=0)
        progress.start()
        progress.emit(
            "population_run_start",
            message=(
                "paper causal run "
                f"discovery={config.sizes.discovery_graphs} "
                f"causal={config.sizes.causal_graphs} "
                f"clean_ablation={config.sizes.clean_ablation_graphs} "
                f"graph_batch={config.execution.graphs_per_batch} "
                f"head_batch={execution.head_batch_size} "
                f"event_batch={execution.event_batch_size}"
            ),
            phase=phase,
            sample_sizes=dataclasses.asdict(config.sizes),
            population_policy=dataclasses.asdict(policy),
            execution=dataclasses.asdict(execution),
            graphs_per_batch=int(config.execution.graphs_per_batch),
        )
    try:
        context = progress.component("population_scores") if progress else nullcontext()
        with context:
            scores = run_scores(
                prepared,
                config,
                plan=_stage_plan(prepared, config, "scores"),
                focused_only=True,
            )
        context = progress.component("population_selection") if progress else nullcontext()
        with context:
            confidence_gate = run_confidence_gate(prepared, config, scores)
            population_gate = build_population_gate(scores, confidence_gate, policy)
            _, cache = _population_cache(prepared, config, population_gate)
            existing_gate = (
                cache.load("focused_population", "gate", strict=True)
                if config.resume and not config.force
                else None
            )
            if existing_gate is None:
                cache.save("focused_population", "gate", population_gate)
            else:
                population_gate = existing_gate
            if progress is not None:
                progress.emit(
                    "population_selection_complete",
                    status=population_gate["status"],
                    specialist_pairs=len(population_gate["specialist_pairs"]),
                    null_heads=len(population_gate["null_pairs"]),
                    candidate_counts=population_gate["candidate_counts"],
                    matching_balance=population_gate["matching_balance"],
                )
        context = progress.component("population_causal") if progress else nullcontext()
        with context:
            core = run_population_analysis(
                prepared,
                config,
                scores,
                confidence_gate,
                population_gate,
                execution=execution,
            )
        result = {
            "analysis_version": POPULATION_CAUSAL_VERSION,
            "output_dir": str(root),
            "population_status": population_gate["status"],
            "head_pair_count": len(population_gate["specialist_pairs"]),
            "null_head_count": len(population_gate["null_pairs"]),
            "sample_sizes": core["sample_sizes"],
            "matching_balance": population_gate["matching_balance"],
            "execution_audit": core["execution_audit"],
        }
        if phase == "all":
            context = progress.component("population_figures") if progress else nullcontext()
            with context:
                result["manifest"] = render_cached_population_figures(root)
        return result
    finally:
        if progress is not None:
            progress.emit("population_run_stop")
            progress.close()


__all__ = [
    "POPULATION_CAUSAL_VERSION",
    "PopulationPolicy",
    "aggregate_population_tests",
    "build_population_gate",
    "layer_adjusted_components",
    "render_cached_population_figures",
    "response_adjusted_components",
    "run",
]
