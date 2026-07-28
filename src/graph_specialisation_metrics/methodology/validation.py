"""Held-out clean ablation, donor-wise necessity, and bidirectional patch validation."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import numpy as np

from .audit import audit_check, within_tolerance
from .bootstrap import (
    Observation,
    nested_percentile_interval,
    paired_channel_percentile_interval,
)
from .causal import (
    calibrated_targets,
    clean_ablation,
    donor_necessity,
    mismatch_adjusted_aligned,
    mismatch_adjusted_gross,
    patch_response,
    reference_scale,
)
from .execution import execute_graph_batches
from .protocol import CHANNELS, PROTOCOL_VERSION
from .scores import HeadCoordinates


def _targets(prepared: Any, scores: Mapping[str, Any]) -> dict[str, tuple[tuple[int, int], ...]]:
    result = {
        f"head_L{layer}_H{head}": ((layer, head),)
        for layer in range(int(prepared.grit.L))
        for head in range(int(prepared.grit.H))
    }
    for name, family in scores["families"].items():
        if family:
            result[f"family_{name}"] = tuple(tuple(value) for value in family)
            for prefix in range(1, len(family) + 1):
                result[f"prefix_{name}_{prefix}"] = tuple(
                    tuple(value) for value in family[:prefix]
                )
    for name, family in scores.get("matched_controls", {}).items():
        if family:
            result[f"control_{name}"] = tuple(tuple(value) for value in family)
            for prefix in range(1, len(family) + 1):
                result[f"control_prefix_{name}_{prefix}"] = tuple(
                    tuple(value) for value in family[:prefix]
                )
    return result


def _reference_target_names(
    target: str,
    target_order: Sequence[str],
) -> list[str]:
    """Matched reference population for a head, family, or cumulative prefix."""

    if target.startswith("head_"):
        return [name for name in target_order if name.startswith("head_")]
    for leaning in ("semantic_leaning", "structural_leaning"):
        if target == f"family_{leaning}":
            return [
                name
                for name in target_order
                if name.startswith(f"control_{leaning}_")
                and not name.startswith("control_prefix_")
            ]
        prefix = f"prefix_{leaning}_"
        if target.startswith(prefix):
            size = target.removeprefix(prefix)
            return [
                name
                for name in target_order
                if name.startswith(f"control_prefix_{leaning}_")
                and name.endswith(f"_{size}")
            ]
    return [name for name in target_order if name.startswith("head_")]


def _requires_matched_reference(target: str) -> bool:
    return target in {
        "family_semantic_leaning",
        "family_structural_leaning",
    } or target.startswith(("prefix_semantic_leaning_", "prefix_structural_leaning_"))


def _aggregate(records: Sequence[Mapping[str, Any]], key: str) -> float:
    """Donors within source, sources within graph, graphs equally."""

    graph_values: list[float] = []
    for graph in sorted({int(row["graph"]) for row in records}):
        rows_graph = [row for row in records if int(row["graph"]) == graph]
        source_values = []
        for source in sorted({int(row["source"]) for row in rows_graph}):
            values = [float(row[key]) for row in rows_graph if int(row["source"]) == source]
            source_values.append(float(np.mean(values)))
        graph_values.append(float(np.mean(source_values)))
    return float(np.mean(graph_values))


def equivalence_decision(
    estimate: float,
    low: float,
    high: float,
    *,
    half_width: float,
) -> str:
    """Classify an interval against a preregistered practical-equivalence region."""

    values = np.asarray((estimate, low, high), dtype=np.float64)
    if not np.isfinite(values).all():
        return "not_estimable"
    margin = float(half_width)
    if float(low) >= -margin and float(high) <= margin:
        return "equivalent"
    if float(low) > margin:
        return "specialised"
    if float(high) < -margin:
        return "reversed"
    return "unresolved"


def _mismatch_indices(records: Sequence[Any]) -> tuple[list[int], set[int]]:
    """Prefer another donor event for the same source.

    Returns the control index chosen for each event and the positions that have no admissible
    control at all. A self-control would silently null that event's mismatch adjustment, so those
    events are excluded from the causal record instead of being reported with a zeroed control.
    """

    result: list[int] = []
    excluded: set[int] = set()
    for position, record in enumerate(records):
        candidates = [
            other
            for other, candidate in enumerate(records)
            if candidate.source == record.source
            and candidate.payload_fingerprint != record.payload_fingerprint
            and candidate.degree_gap == record.degree_gap
            and other != position
        ]
        if not candidates:
            candidates = [
                other
                for other, candidate in enumerate(records)
                if candidate.payload_fingerprint != record.payload_fingerprint
                and candidate.degree_gap == record.degree_gap
                and other != position
            ]
        if not candidates:
            # Relax the degree tier before giving up on the same-geometry control.
            relaxed = [
                other
                for other, candidate in enumerate(records)
                if candidate.payload_fingerprint != record.payload_fingerprint
                and other != position
            ]
            audit_check(
                False,
                "causal.mismatch_control_relaxed"
                if relaxed
                else "causal.mismatch_control_unavailable",
                "no distinct-payload donor shares this event's degree tier; "
                + (
                    "the mismatch control was drawn from another degree tier"
                    if relaxed
                    else "the event has no admissible control and is excluded from the "
                    "causal record"
                ),
                context={"source": int(record.source), "relaxed_degree_tier": bool(relaxed)},
            )
            if not relaxed:
                excluded.add(position)
            candidates = relaxed or [position]
        result.append(
            min(
                candidates,
                key=lambda other: (
                    abs(float(records[other].dose) - float(record.dose)),
                    abs(int(records[other].degree_gap) - int(record.degree_gap)),
                    other,
                ),
            )
        )
    return result, excluded


def _clean_ablation_stage(
    prepared: Any,
    config: Any,
    targets: Mapping[str, Sequence[tuple[int, int]]],
    scores: Mapping[str, Any],
    *,
    cache: Any | None = None,
) -> dict[str, Any]:
    import torch

    from .runner import _stage_ids

    output: dict[str, Any] = {}
    graph_ids = list(_stage_ids(prepared, "clean_ablation"))
    cached_targets = {
        name: (
            cache.load("causal/clean_ablation", name, strict=True)
            if cache is not None and config.resume and not config.force
            else None
        )
        for name in targets
    }
    missing_targets = [
        name for name, value in cached_targets.items() if value is None
    ]
    clean_by_graph: dict[int, Any] = {}

    def execute_clean(chunk):
        bases = [
            prepared.grit.eval_ds[int(graph_id)]
            for graph_id in chunk
        ]
        prediction, z, target = prepared.backend.ablate(bases, ())
        if int(prediction.shape[0]) != len(chunk):
            raise RuntimeError("grouped clean-ablation capture changed the graph count")
        return [
            (
                int(graph_id),
                {
                    "prediction": prediction[position : position + 1].detach(),
                    "z": z[position : position + 1].detach(),
                    "target": target[position : position + 1].detach(),
                },
            )
            for position, graph_id in enumerate(chunk)
        ]

    clean_report = execute_graph_batches(
        graph_ids if missing_targets else [],
        graphs_per_batch=config.execution.graphs_per_batch,
        execute=execute_clean,
        consume=lambda rows: clean_by_graph.update(rows),
        oom_backoff=config.execution.oom_backoff,
    )
    target_reports: dict[str, Any] = {}
    for name, family in targets.items():
        if cached_targets[name] is not None:
            output[name] = cached_targets[name]
            target_reports[name] = {
                "cache_hit": True,
                "cache_miss": False,
            }
            if getattr(prepared, "progress", None) is not None:
                prepared.progress.emit(
                    "causal_target_cache_hit",
                    causal_component="clean_ablation",
                    target=name,
                )
            continue
        rows = []
        clean_predictions = []
        ablated_predictions = []
        truths = []

        def execute_target(chunk):
            bases = [
                prepared.grit.eval_ds[int(graph_id)]
                for graph_id in chunk
            ]
            prediction_a, z_a, target_a = prepared.backend.ablate(bases, family)
            if int(prediction_a.shape[0]) != len(chunk):
                raise RuntimeError("grouped clean ablation changed the graph count")
            loss_clean = prepared.backend.loss_per_graph(
                torch.cat(
                    [clean_by_graph[int(graph_id)]["prediction"] for graph_id in chunk],
                    dim=0,
                ),
                torch.cat(
                    [clean_by_graph[int(graph_id)]["target"] for graph_id in chunk],
                    dim=0,
                ),
            ).detach().cpu().numpy()
            loss_ablated = prepared.backend.loss_per_graph(
                prediction_a, target_a
            ).detach().cpu().numpy()
            result = []
            for position, graph_id in enumerate(chunk):
                graph_id = int(graph_id)
                clean = clean_by_graph[graph_id]
                endpoint = clean_ablation(
                    clean["z"].detach().cpu().numpy(),
                    z_a[position : position + 1].detach().cpu().numpy(),
                    loss_clean[position : position + 1],
                    loss_ablated[position : position + 1],
                )
                result.append(
                    {
                        "graph": graph_id,
                        "clean_prediction": clean["prediction"]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(1, -1),
                        "ablated_prediction": prediction_a[
                            position : position + 1
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(1, -1),
                        "truth": clean["target"]
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(1, -1),
                        "prediction_movement": float(
                            endpoint["prediction_movement"][0]
                        ),
                        "loss_change": float(endpoint["loss_change"][0]),
                    }
                )
            return result

        def consume_target(values):
            for value in values:
                clean_predictions.append(value.pop("clean_prediction"))
                ablated_predictions.append(value.pop("ablated_prediction"))
                truths.append(value.pop("truth"))
                rows.append(value)

        target_report = execute_graph_batches(
            graph_ids,
            graphs_per_batch=config.execution.graphs_per_batch,
            execute=execute_target,
            consume=consume_target,
            oom_backoff=config.execution.oom_backoff,
        )
        target_reports[name] = dataclasses.asdict(target_report)
        metric_clean = float(
            prepared.task.metric_fn(
                np.concatenate(clean_predictions), np.concatenate(truths)
            )
        )
        metric_ablated = float(
            prepared.task.metric_fn(
                np.concatenate(ablated_predictions), np.concatenate(truths)
            )
        )
        output[name] = {
            "prediction_movement": float(
                np.mean([row["prediction_movement"] for row in rows])
            ),
            "loss_change": float(np.mean([row["loss_change"] for row in rows])),
            "registered_metric_clean": metric_clean,
            "registered_metric_ablated": metric_ablated,
            "registered_metric_change": metric_ablated - metric_clean,
            "graphs": rows,
        }
        if cache is not None:
            cache.save("causal/clean_ablation", name, output[name])
        if getattr(prepared, "progress", None) is not None:
            prepared.progress.emit(
                "causal_target_complete",
                causal_component="clean_ablation",
                target=name,
            )
    names = list(targets)
    vector_observations = []
    graph_ids = sorted(
        {int(row["graph"]) for name in names for row in output[name]["graphs"]}
    )
    for graph_id in graph_ids:
        vector_observations.append(
            Observation(
                seed=int(prepared.grit.sc.seed),
                graph=graph_id,
                source=0,
                donor=0,
                value=np.asarray(
                    [
                        [
                            next(
                                row["prediction_movement"]
                                for row in output[name]["graphs"]
                                if int(row["graph"]) == graph_id
                            ),
                            next(
                                row["loss_change"]
                                for row in output[name]["graphs"]
                                if int(row["graph"]) == graph_id
                            ),
                        ]
                        for name in names
                    ]
                ),
            )
        )
    endpoint_interval = nested_percentile_interval(
        vector_observations, config.bootstrap
    )
    head_positions = [
        position for position, name in enumerate(names) if name.startswith("head_")
    ]
    J = scores["coordinates"].joint_sensitivity.reshape(-1)

    def association_transform(value):
        head_values = value[head_positions]
        return np.asarray(
            (
                _spearman(J, head_values[:, 0])["rho"],
                _spearman(J, head_values[:, 1])["rho"],
            )
        )

    association_interval = nested_percentile_interval(
        vector_observations,
        config.bootstrap,
        transform=association_transform,
    )
    output["_intervals"] = {
        "target_order": names,
        "endpoint_order": ("prediction_movement", "loss_change"),
        "interval": endpoint_interval,
        "association_order": (
            "J_vs_clean_prediction_movement",
            "J_vs_clean_loss_change",
        ),
        "association_interval": association_interval,
    }
    output["_execution"] = {
        "clean_reused_across_targets": True,
        "clean_graph_batches": dataclasses.asdict(clean_report),
        "target_graph_batches": target_reports,
        "target_cache_hits": int(len(targets) - len(missing_targets)),
        "target_cache_misses": int(len(missing_targets)),
    }
    return output


def _causal_events(
    prepared: Any,
    config: Any,
    scores: Mapping[str, Any],
    targets: Mapping[str, Sequence[tuple[int, int]]],
    plan: Mapping[int, Mapping[str, Any]],
    *,
    cache: Any | None = None,
) -> dict[str, Any]:
    from .runner import _rebuild_graph_events

    records_by_target: dict[str, dict[str, list[dict[str, Any]]]] = {
        target: {channel: [] for channel in CHANNELS} for target in targets
    }
    self_patch_max = 0.0
    excluded_events = 0
    for graph_id in sorted(plan):
        base = prepared.grit.eval_ds[int(graph_id)]
        for channel in CHANNELS:
            shard_stage = f"causal/events/{channel}/graph_{int(graph_id):06d}"
            cached_targets = {
                target_name: (
                    cache.load(shard_stage, target_name, strict=True)
                    if cache is not None and config.resume and not config.force
                    else None
                )
                for target_name in targets
            }
            if all(value is not None for value in cached_targets.values()):
                first = next(iter(cached_targets.values()))
                excluded_events += int(first.get("uncontrolled_events_excluded", 0))
                for target_name, cached in cached_targets.items():
                    records_by_target[target_name][channel].extend(cached["rows"])
                    self_patch_max = max(
                        self_patch_max, float(cached["same_condition_patch_max"])
                    )
                continue
            sources = plan[graph_id][channel]["sources"]
            variants, records = _rebuild_graph_events(
                prepared, config, "causal", graph_id, channel, sources
            )
            if not records:
                continue
            captured = prepared.backend.capture(
                [base, *variants], require_grad=False, include_virtual_transport=True
            )
            z_clean = captured.z[0:1].detach().cpu().numpy()
            z_event = captured.z[1:].detach().cpu().numpy()
            mismatch, uncontrolled = _mismatch_indices(records)
            excluded_events += len(uncontrolled)
            event_indices = list(range(1, len(records) + 1))
            mismatch_capture_indices = [value + 1 for value in mismatch]
            clean_replacements = prepared.backend.replacement_batch(
                captured, [0] * len(records)
            )
            event_replacements = prepared.backend.replacement_batch(
                captured, event_indices
            )
            mismatch_replacements = prepared.backend.replacement_batch(
                captured, mismatch_capture_indices
            )
            for target_name, family in targets.items():
                cached = cached_targets[target_name]
                if cached is not None:
                    records_by_target[target_name][channel].extend(cached["rows"])
                    self_patch_max = max(
                        self_patch_max, float(cached["same_condition_patch_max"])
                    )
                    if getattr(prepared, "progress", None) is not None:
                        prepared.progress.emit(
                            "causal_target_cache_hit",
                            causal_component="events",
                            graph_id=int(graph_id),
                            channel=channel,
                            target=target_name,
                        )
                    continue
                # Same-condition patch is a numerical audit, not a control endpoint.
                self_replacement = prepared.backend.replacement_batch(captured, [0])
                _, self_z, _ = prepared.backend.patch(base, self_replacement, family)
                target_self_patch = float(
                    np.max(np.abs(self_z.detach().cpu().numpy() - z_clean))
                )
                self_patch_max = max(self_patch_max, target_self_patch)

                # Donor-wise necessity is evaluated in one aligned [clean,*events] batch.
                _, z_ablated, _ = prepared.backend.ablate([base, *variants], family)
                necessity = donor_necessity(
                    np.repeat(z_clean, len(records), axis=0),
                    z_event,
                    np.repeat(
                        z_ablated[0:1].detach().cpu().numpy(), len(records), axis=0
                    ),
                    z_ablated[1:].detach().cpu().numpy(),
                    epsilon=config.numerical.effect_floor,
                )

                # Matched restoration/injection.
                _, z_restore, _ = prepared.backend.patch_many(
                    variants, clean_replacements, family
                )
                _, z_inject, _ = prepared.backend.patch_many(
                    [base] * len(records), event_replacements, family
                )
                matched = patch_response(
                    np.repeat(z_clean, len(records), axis=0),
                    z_event,
                    z_restore.detach().cpu().numpy(),
                    z_inject.detach().cpu().numpy(),
                    epsilon=config.numerical.effect_floor,
                )

                # Same-geometry nonspecific control.
                _, z_restore_mismatch, _ = prepared.backend.patch_many(
                    variants, mismatch_replacements, family
                )
                _, z_inject_mismatch, _ = prepared.backend.patch_many(
                    [base] * len(records), mismatch_replacements, family
                )
                mismatched = patch_response(
                    np.repeat(z_clean, len(records), axis=0),
                    z_event,
                    z_restore_mismatch.detach().cpu().numpy(),
                    z_inject_mismatch.detach().cpu().numpy(),
                    epsilon=config.numerical.effect_floor,
                )
                gross_adjusted = mismatch_adjusted_gross(matched, mismatched)
                aligned_adjusted = mismatch_adjusted_aligned(matched, mismatched)
                restoration_aligned_adjusted = (
                    matched.restoration_aligned - mismatched.restoration_aligned
                )
                injection_aligned_adjusted = (
                    matched.injection_aligned - mismatched.injection_aligned
                )
                target_rows: list[dict[str, Any]] = []
                for position, record in enumerate(records):
                    if position in uncontrolled:
                        # No admissible mismatch control: excluded rather than self-controlled.
                        continue
                    mismatch_record = records[mismatch[position]]
                    target_rows.append(
                        {
                            "graph": int(graph_id),
                            "source": int(record.source),
                            "donor": int(record.draw),
                            "mismatch_source": int(mismatch_record.source),
                            "mismatch_donor": int(mismatch_record.draw),
                            "mismatch_payload_fingerprint": (
                                mismatch_record.payload_fingerprint
                            ),
                            "mismatch_dose": float(mismatch_record.dose),
                            "matched_dose": float(record.dose),
                            "G_c": float(gross_adjusted[position]),
                            "P_gross_matched": float(
                                matched.bidirectional_gross[position]
                            ),
                            "P_gross_mismatch": float(
                                mismatched.bidirectional_gross[position]
                            ),
                            "R_gross": float(matched.restoration_gross[position]),
                            "I_gross": float(matched.injection_gross[position]),
                            "R_align": float(matched.restoration_aligned[position]),
                            "I_align": float(matched.injection_aligned[position]),
                            "R_align_adjusted": float(
                                restoration_aligned_adjusted[position]
                            ),
                            "I_align_adjusted": float(
                                injection_aligned_adjusted[position]
                            ),
                            "M_align": float(aligned_adjusted[position]),
                            "necessity": float(
                                necessity["aligned_necessity"][position]
                            ),
                            "gross_necessity": float(
                                necessity["gross_necessity"][position]
                            ),
                            "event_effect": float(necessity["event_effect"][position]),
                        }
                    )
                records_by_target[target_name][channel].extend(target_rows)
                if cache is not None:
                    cache.save(
                        shard_stage,
                        target_name,
                        {
                            "rows": target_rows,
                            "same_condition_patch_max": target_self_patch,
                            "uncontrolled_events_excluded": len(uncontrolled),
                        },
                    )
                if getattr(prepared, "progress", None) is not None:
                    prepared.progress.emit(
                        "causal_target_complete",
                        causal_component="events",
                        graph_id=int(graph_id),
                        channel=channel,
                        target=target_name,
                    )
    within_tolerance(
        self_patch_max,
        config.numerical.reconstruction_tolerance,
        "causal.same_condition_patch",
        "same-condition activation patch response",
    )
    return {
        "records": records_by_target,
        "uncontrolled_events_excluded": int(excluded_events),
        "same_condition_patch_max": self_patch_max,
    }


def _summarize_causal(
    event_output: Mapping[str, Any],
    targets: Mapping[str, Sequence[tuple[int, int]]],
    config: Any,
    scores: Mapping[str, Any],
    *,
    paired_channel_sources: bool = True,
    source_resampling: tuple[bool, bool] = (True, True),
) -> dict[str, Any]:
    records = event_output["records"]
    summary: dict[str, dict[str, Any]] = {}
    for target in targets:
        summary[target] = {}
        for channel in CHANNELS:
            rows = records[target][channel]
            summary[target][channel] = {
                key: _aggregate(rows, key)
                for key in (
                    "G_c",
                    "P_gross_matched",
                    "P_gross_mismatch",
                    "R_gross",
                    "I_gross",
                    "R_align",
                    "I_align",
                    "R_align_adjusted",
                    "I_align_adjusted",
                    "necessity",
                    "gross_necessity",
                )
            }
            finite_aligned = [row for row in rows if np.isfinite(row["M_align"])]
            summary[target][channel]["M_align"] = (
                _aggregate(finite_aligned, "M_align") if finite_aligned else np.nan
            )

    head_targets = [name for name in targets if name.startswith("head_")]
    gross_scales = {}
    necessity_scales = {}
    for channel in CHANNELS:
        gross_scales[channel] = reference_scale(
            [summary[name][channel]["P_gross_matched"] for name in head_targets],
            floor=config.numerical.effect_floor,
        )
        necessity_scales[channel] = reference_scale(
            [summary[name][channel]["gross_necessity"] for name in head_targets],
            floor=config.numerical.effect_floor,
        )
    family_reference_scales = {}
    target_order = list(targets)
    for target in targets:
        reference_names = _reference_target_names(target, target_order)
        if _requires_matched_reference(target) and not reference_names:
            audit_check(
                False,
                "causal.matched_reference",
                f"{target!r} has no frozen same-composition matched reference family; "
                "its calibration falls back to the all-head reference",
                context={"target": target},
            )
        if not reference_names:
            reference_names = head_targets
        target_gross_scales = {
            channel: reference_scale(
                [
                    summary[name][channel]["P_gross_matched"]
                    for name in reference_names
                ],
                floor=config.numerical.effect_floor,
            )
            for channel in CHANNELS
        }
        target_necessity_scales = {
            channel: reference_scale(
                [
                    summary[name][channel]["gross_necessity"]
                    for name in reference_names
                ],
                floor=config.numerical.effect_floor,
            )
            for channel in CHANNELS
        }
        calibrated = calibrated_targets(
            summary[target]["semantic"]["G_c"],
            summary[target]["structural"]["G_c"],
            summary[target]["semantic"]["necessity"],
            summary[target]["structural"]["necessity"],
            gross_scales=target_gross_scales,
            necessity_scales=target_necessity_scales,
        )
        calibrated.update(
            {
                "rescue_semantic": (
                    summary[target]["semantic"]["R_align_adjusted"]
                    / target_gross_scales["semantic"]
                ),
                "rescue_structural": (
                    summary[target]["structural"]["R_align_adjusted"]
                    / target_gross_scales["structural"]
                ),
                "induction_semantic": (
                    summary[target]["semantic"]["I_align_adjusted"]
                    / target_gross_scales["semantic"]
                ),
                "induction_structural": (
                    summary[target]["structural"]["I_align_adjusted"]
                    / target_gross_scales["structural"]
                ),
            }
        )
        summary[target]["calibrated"] = {
            key: float(value) for key, value in calibrated.items()
        }
        summary[target]["calibration_reference"] = {
            "targets": reference_names,
            "gross_scales": target_gross_scales,
            "necessity_scales": target_necessity_scales,
        }
        if not target.startswith("head_"):
            family_reference_scales[target] = summary[target][
                "calibration_reference"
            ]
    endpoint_order = (
        "G_c",
        "P_gross_matched",
        "P_gross_mismatch",
        "R_gross",
        "I_gross",
        "R_align",
        "I_align",
        "R_align_adjusted",
        "I_align_adjusted",
        "necessity",
        "gross_necessity",
    )
    calibrated_order = (
        "gross_total_for_J",
        "gross_contrast_for_D_rel",
        "necessity_total_for_J",
        "necessity_contrast_for_D_rel",
        "g_semantic",
        "g_structural",
        "n_semantic",
        "n_structural",
        "rescue_semantic",
        "rescue_structural",
        "induction_semantic",
        "induction_structural",
    )
    head_positions = [
        position for position, name in enumerate(target_order) if name.startswith("head_")
    ]
    by_channel_key = {}
    for channel_index, channel in enumerate(CHANNELS):
        for target_index, target in enumerate(target_order):
            for row in records[target][channel]:
                key = (int(row["graph"]), int(row["source"]), int(row["donor"]))
                by_channel_key.setdefault(key, np.full(
                    (len(CHANNELS), len(target_order), len(endpoint_order)), np.nan
                ))
                by_channel_key[key][channel_index, target_index] = [
                    float(row[name]) for name in endpoint_order
                ]
    complete = (
        [
            (key, value)
            for key, value in sorted(by_channel_key.items())
            if np.isfinite(value).all()
        ]
        if paired_channel_sources
        else []
    )
    semantic_family = "family_semantic_leaning"
    structural_family = "family_structural_leaning"
    interaction_order = (
        (
            "gross_family_by_channel",
            "necessity_family_by_channel",
            "rescue_family_by_channel",
            "induction_family_by_channel",
        )
        if semantic_family in target_order and structural_family in target_order
        else ()
    )

    def family_interaction_values(calibrated: np.ndarray) -> np.ndarray:
        if not interaction_order:
            return np.empty(0, dtype=np.float64)
        semantic_position = target_order.index(semantic_family)
        structural_position = target_order.index(structural_family)

        def difference(column: str) -> float:
            index = calibrated_order.index(column)
            return float(
                calibrated[semantic_position, index]
                - calibrated[structural_position, index]
            )

        return np.asarray(
            (
                difference("gross_contrast_for_D_rel"),
                difference("necessity_contrast_for_D_rel"),
                (
                    difference("rescue_semantic")
                    - difference("rescue_structural")
                ),
                (
                    difference("induction_semantic")
                    - difference("induction_structural")
                ),
            ),
            dtype=np.float64,
        )

    def transform(value):
        # Recompute positive unadjusted reference scales in every bootstrap draw.
        G = value[:, :, endpoint_order.index("G_c")]
        N = value[:, :, endpoint_order.index("necessity")]
        calibrated_rows = []
        for target_index, target in enumerate(target_order):
            reference_names = _reference_target_names(target, target_order)
            if _requires_matched_reference(target) and not reference_names:
                audit_check(
                    False,
                    "causal.bootstrap_matched_reference",
                    f"{target!r} has no matched reference in a causal bootstrap draw; "
                    "the draw falls back to the all-head reference",
                    context={"target": target},
                )
            reference_positions = [
                target_order.index(name) for name in reference_names
            ] or head_positions
            a_g = np.asarray(
                [
                    np.mean(
                        value[
                            channel,
                            reference_positions,
                            endpoint_order.index("P_gross_matched"),
                        ]
                    )
                    for channel in range(len(CHANNELS))
                ]
            )
            a_n = np.asarray(
                [
                    np.mean(
                        value[
                            channel,
                            reference_positions,
                            endpoint_order.index("gross_necessity"),
                        ]
                    )
                    for channel in range(len(CHANNELS))
                ]
            )
            if np.any(a_g <= config.numerical.effect_floor) or np.any(
                a_n <= config.numerical.effect_floor
            ):
                audit_check(
                    False,
                    "causal.bootstrap_reference_scale",
                    "a causal bootstrap reference scale fell below the registered floor; "
                    "the affected draw is reported as non-estimable",
                    observed=float(min(np.min(a_g), np.min(a_n))),
                    tolerance=float(config.numerical.effect_floor),
                    context={"target": target},
                )
                # Non-estimable scales become nan rather than exploding the ratio.
                a_g = np.where(a_g > config.numerical.effect_floor, a_g, np.nan)
                a_n = np.where(a_n > config.numerical.effect_floor, a_n, np.nan)
            g_sem = G[0, target_index] / a_g[0]
            g_str = G[1, target_index] / a_g[1]
            n_sem = N[0, target_index] / a_n[0]
            n_str = N[1, target_index] / a_n[1]
            rescue_sem = (
                value[
                    0,
                    target_index,
                    endpoint_order.index("R_align_adjusted"),
                ]
                / a_g[0]
            )
            rescue_str = (
                value[
                    1,
                    target_index,
                    endpoint_order.index("R_align_adjusted"),
                ]
                / a_g[1]
            )
            induction_sem = (
                value[
                    0,
                    target_index,
                    endpoint_order.index("I_align_adjusted"),
                ]
                / a_g[0]
            )
            induction_str = (
                value[
                    1,
                    target_index,
                    endpoint_order.index("I_align_adjusted"),
                ]
                / a_g[1]
            )
            calibrated_rows.append(
                (
                    0.5 * (g_sem + g_str),
                    g_sem - g_str,
                    0.5 * (n_sem + n_str),
                    n_sem - n_str,
                    g_sem,
                    g_str,
                    n_sem,
                    n_str,
                    rescue_sem,
                    rescue_str,
                    induction_sem,
                    induction_str,
                )
            )
        calibrated = np.asarray(calibrated_rows)
        coordinates = scores["coordinates"]
        J = coordinates.joint_sensitivity.reshape(-1)
        D = coordinates.selectivity.reshape(-1)
        active = coordinates.active.reshape(-1)
        head_calibrated = calibrated[head_positions]
        associations = np.asarray(
            (
                _spearman(J, head_calibrated[:, 0])["rho"],
                _spearman(D[active], head_calibrated[active, 1])["rho"],
                _spearman(J, head_calibrated[:, 2])["rho"],
                _spearman(D[active], head_calibrated[active, 3])["rho"],
            )
        )
        return np.concatenate(
            (
                value.reshape(-1),
                calibrated.reshape(-1),
                family_interaction_values(calibrated),
                associations,
            )
        )

    if complete:
        causal_observations = [
            Observation(
                seed=0,
                graph=key[0],
                source=key[1],
                donor=key[2],
                value=value,
            )
            for key, value in complete
        ]
        causal_interval = nested_percentile_interval(
            causal_observations, config.bootstrap, transform=transform
        )
        interval_pairing = "source-and-graph-paired"
    elif not paired_channel_sources:
        channel_observations: dict[str, list[Observation]] = {}
        for channel in CHANNELS:
            by_key: dict[tuple[int, int, int], np.ndarray] = {}
            for target_index, target in enumerate(target_order):
                for row in records[target][channel]:
                    key = (
                        int(row["graph"]),
                        int(row["source"]),
                        int(row["donor"]),
                    )
                    by_key.setdefault(
                        key,
                        np.full(
                            (len(target_order), len(endpoint_order)),
                            np.nan,
                        ),
                    )
                    by_key[key][target_index] = [
                        float(row[name]) for name in endpoint_order
                    ]
            channel_observations[channel] = [
                Observation(0, key[0], key[1], key[2], value)
                for key, value in sorted(by_key.items())
                if np.isfinite(value).all()
            ]
        causal_interval = paired_channel_percentile_interval(
            channel_observations["semantic"],
            channel_observations["structural"],
            config.bootstrap,
            transform=transform,
            resample_source=source_resampling,
        )
        interval_pairing = "graph-paired/channel-source-independent"
    else:
        causal_interval = None
        interval_pairing = None
    return {
        "targets": summary,
        "gross_reference_scales": gross_scales,
        "necessity_reference_scales": necessity_scales,
        "family_reference_scales": family_reference_scales,
        "intervals": {
            "target_order": target_order,
            "channel_order": CHANNELS,
            "endpoint_order": endpoint_order,
            "calibrated_order": calibrated_order,
            "interaction_order": interaction_order,
            "association_order": (
                "J_vs_gross_total",
                "D_rel_vs_gross_contrast_active",
                "J_vs_necessity_total",
                "D_rel_vs_necessity_contrast_active",
            ),
            "interval": causal_interval,
            "pairing": interval_pairing,
        },
    }


def _spearman(x: Any, y: Any) -> dict[str, float]:
    from scipy.stats import spearmanr

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return {"rho": np.nan, "p": np.nan, "n": int(mask.sum())}
    result = spearmanr(x[mask], y[mask])
    return {"rho": float(result.statistic), "p": float(result.pvalue), "n": int(mask.sum())}


def _within_layer_permutation(
    x: Any,
    y: Any,
    layers: Any,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    layers = np.asarray(layers, int)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, layers = x[mask], y[mask], layers[mask]
    observed = _spearman(x, y)["rho"]
    if len(x) < 3 or not np.isfinite(observed):
        return {"rho": observed, "p": np.nan, "replicates": int(replicates)}
    rng = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(int(replicates)):
        permuted = y.copy()
        for layer in np.unique(layers):
            indices = np.flatnonzero(layers == layer)
            permuted[indices] = permuted[rng.permutation(indices)]
        candidate = _spearman(x, permuted)["rho"]
        exceed += int(np.isfinite(candidate) and abs(candidate) >= abs(observed))
    return {
        "rho": float(observed),
        "p": float((exceed + 1) / (int(replicates) + 1)),
        "replicates": int(replicates),
    }


def _layer_adjusted(x: Any, y: Any, layers: Any) -> dict[str, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    layers = np.asarray(layers, int)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y, layers = x[mask], y[mask], layers[mask]
    if len(x) < 3:
        return {"coefficient": np.nan, "n": int(len(x))}
    indicators = np.column_stack(
        [(layers == value).astype(float) for value in sorted(set(layers))[1:]]
    ) if len(set(layers)) > 1 else np.empty((len(layers), 0))
    design = np.column_stack((np.ones(len(x)), x, indicators))
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0][1]
    return {"coefficient": float(coefficient), "n": int(len(x))}


def _association_report(
    prepared: Any,
    scores: Mapping[str, Any],
    summary: Mapping[str, Any],
    clean: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    coordinates: HeadCoordinates = scores["coordinates"]
    layers, names = [], []
    for layer in range(int(prepared.grit.L)):
        for head in range(int(prepared.grit.H)):
            layers.append(layer)
            names.append(f"head_L{layer}_H{head}")
    J = coordinates.joint_sensitivity.reshape(-1)
    D = coordinates.selectivity.reshape(-1)
    active = coordinates.active.reshape(-1)
    target_rows = summary["targets"]
    endpoints = {
        "clean_prediction_movement": np.asarray(
            [clean[name]["prediction_movement"] for name in names]
        ),
        "clean_loss_change": np.asarray([clean[name]["loss_change"] for name in names]),
        "gross_total": np.asarray(
            [target_rows[name]["calibrated"]["gross_total_for_J"] for name in names]
        ),
        "gross_contrast": np.asarray(
            [target_rows[name]["calibrated"]["gross_contrast_for_D_rel"] for name in names]
        ),
        "necessity_total": np.asarray(
            [target_rows[name]["calibrated"]["necessity_total_for_J"] for name in names]
        ),
        "necessity_contrast": np.asarray(
            [
                target_rows[name]["calibrated"]["necessity_contrast_for_D_rel"]
                for name in names
            ]
        ),
        "G_semantic": np.asarray(
            [target_rows[name]["semantic"]["G_c"] for name in names]
        ),
        "G_structural": np.asarray(
            [target_rows[name]["structural"]["G_c"] for name in names]
        ),
    }
    report: dict[str, Any] = {}
    for endpoint in (
        "clean_prediction_movement",
        "clean_loss_change",
        "gross_total",
        "necessity_total",
    ):
        report[f"J_vs_{endpoint}"] = {
            "pooled": _spearman(J, endpoints[endpoint]),
            "layer_adjusted": _layer_adjusted(J, endpoints[endpoint], layers),
            "within_layer": {
                str(layer): _spearman(
                    J[np.asarray(layers) == layer],
                    endpoints[endpoint][np.asarray(layers) == layer],
                )
                for layer in sorted(set(layers))
            },
            "within_layer_permutation": _within_layer_permutation(
                J,
                endpoints[endpoint],
                layers,
                replicates=config.bootstrap.replicates,
                seed=config.bootstrap.rng_seed,
            ),
        }
    for endpoint in ("gross_contrast", "necessity_contrast"):
        report[f"D_rel_vs_{endpoint}"] = {
            "pooled_active": _spearman(D[active], endpoints[endpoint][active]),
            "layer_adjusted_active": _layer_adjusted(
                D[active], endpoints[endpoint][active], np.asarray(layers)[active]
            ),
            "n_active": int(active.sum()),
            "within_layer_permutation_active": _within_layer_permutation(
                D[active],
                endpoints[endpoint][active],
                np.asarray(layers)[active],
                replicates=config.bootstrap.replicates,
                seed=config.bootstrap.rng_seed + 1,
            ),
        }
    semantic_raw = scores["channels"]["semantic"]["raw"].reshape(-1)
    structural_raw = scores["channels"]["structural"]["raw"].reshape(-1)
    report["raw_score_validation"] = {
        "S_semantic_vs_G_semantic": _spearman(
            semantic_raw, endpoints["G_semantic"]
        ),
        "S_structural_vs_G_structural": _spearman(
            structural_raw, endpoints["G_structural"]
        ),
        "S_semantic_vs_G_structural_control": _spearman(
            semantic_raw, endpoints["G_structural"]
        ),
        "S_structural_vs_G_semantic_control": _spearman(
            structural_raw, endpoints["G_semantic"]
        ),
    }
    return report


def _causal_cache(prepared: Any, config: Any, scores: Mapping[str, Any]):
    from .cache import CanonicalCache
    from .protocol import stable_hash
    from .runner import _cache, _stage_plan

    plan = _stage_plan(prepared, config, "causal")
    base_cache = _cache(prepared, config, plan)
    causal_manifest = stable_hash(
        {
            "events": base_cache.contract.event_manifest_hash,
            "families": scores["families"],
            "matched_controls": scores.get("matched_controls", {}),
        }
    )
    return plan, CanonicalCache(
        config.root,
        dataclasses.replace(
            base_cache.contract, event_manifest_hash=causal_manifest
        ),
    )


def load_cached_causal_validation(
    prepared: Any,
    config: Any,
    scores: Mapping[str, Any],
) -> dict[str, Any] | None:
    if scores is None:
        return None
    _, cache = _causal_cache(prepared, config, scores)
    return cache.load("causal", "validation", strict=True)


def run_causal_validation(
    prepared: Any,
    config: Any,
    scores: Mapping[str, Any],
) -> dict[str, Any]:
    """Run all-head validation and frozen-family causal confirmation on disjoint graphs."""

    if scores is None:
        raise ValueError("causal validation requires discovery scores")
    from .runner import _channel_bootstrap_policy

    plan, cache = _causal_cache(prepared, config, scores)
    if config.resume and not config.force:
        cached = cache.load("causal", "validation", strict=True)
        if cached is not None:
            return cached
    targets = _targets(prepared, scores)
    clean = _clean_ablation_stage(
        prepared, config, targets, scores, cache=cache
    )
    events = _causal_events(
        prepared, config, scores, targets, plan, cache=cache
    )
    summary = _summarize_causal(
        events,
        targets,
        config,
        scores,
        paired_channel_sources=prepared.task.paired_channel_sources,
        source_resampling=tuple(
            _channel_bootstrap_policy(prepared, config, plan, channel).resample_source
            for channel in CHANNELS
        ),
    )
    associations = _association_report(prepared, scores, summary, clean, config)
    causal_interval = summary["intervals"]["interval"]
    if causal_interval is not None:
        association_count = len(summary["intervals"]["association_order"])
        associations["nested_interval_order"] = summary["intervals"][
            "association_order"
        ]
        associations["nested_interval_low"] = causal_interval.low[
            -association_count:
        ]
        associations["nested_interval_high"] = causal_interval.high[
            -association_count:
        ]
    target_summary = summary["targets"]
    interval_metadata = summary["intervals"]
    target_order = list(interval_metadata["target_order"])
    calibrated_order = list(interval_metadata["calibrated_order"])
    endpoint_order = list(interval_metadata["endpoint_order"])
    raw_size = len(CHANNELS) * len(target_order) * len(endpoint_order)
    calibrated_size = len(target_order) * len(calibrated_order)
    interaction_start = raw_size + calibrated_size
    interaction_order = list(interval_metadata.get("interaction_order", ()))
    margin = float(config.families.causal_equivalence_half_width)
    family_interactions: dict[str, Any] = {}
    if causal_interval is not None:
        for position, name in enumerate(interaction_order):
            index = interaction_start + position
            estimate = float(causal_interval.estimate[index])
            low = float(causal_interval.low[index])
            high = float(causal_interval.high[index])
            family_interactions[name] = {
                "estimate": estimate,
                "low": low,
                "high": high,
                "equivalence_half_width": margin,
                "decision": equivalence_decision(
                    estimate, low, high, half_width=margin
                ),
            }

    core_name = "family_central_responsive"
    core_profile: dict[str, Any] = {}
    response_floor = float(config.families.causal_response_floor)
    if core_name in target_summary:
        target_position = target_order.index(core_name)
        profile_columns = {
            "gross_response": ("g_semantic", "g_structural"),
            "donor_wise_necessity": ("n_semantic", "n_structural"),
            "causal_rescue": ("rescue_semantic", "rescue_structural"),
            "causal_induction": ("induction_semantic", "induction_structural"),
        }
        for endpoint, channel_columns in profile_columns.items():
            channel_records = {}
            for channel, column in zip(CHANNELS, channel_columns):
                value = float(target_summary[core_name]["calibrated"][column])
                column_position = calibrated_order.index(column)
                flat_position = (
                    raw_size
                    + target_position * len(calibrated_order)
                    + column_position
                )
                channel_records[channel] = {
                    "estimate": value,
                    "low": (
                        float(causal_interval.low[flat_position])
                        if causal_interval is not None
                        else np.nan
                    ),
                    "high": (
                        float(causal_interval.high[flat_position])
                        if causal_interval is not None
                        else np.nan
                    ),
                }
            core_profile[endpoint] = {
                "channels": channel_records,
                "response_floor": response_floor,
                "dual_channel": bool(
                    all(
                        np.isfinite(record["low"])
                        and record["low"] > response_floor
                        for record in channel_records.values()
                    )
                ),
            }

    importance_floor = float(config.families.importance_correlation_floor)
    activity_validation: dict[str, Any] = {}
    clean_interval_metadata = clean.get("_intervals", {})
    clean_association_interval = clean_interval_metadata.get("association_interval")
    causal_association_order = list(
        associations.get("nested_interval_order", ()) or ()
    )
    causal_association_low = associations.get("nested_interval_low")
    causal_association_high = associations.get("nested_interval_high")
    for name in (
        "J_vs_clean_prediction_movement",
        "J_vs_gross_total",
        "J_vs_necessity_total",
    ):
        pooled = associations.get(name, {}).get("pooled", {})
        estimate = float(pooled.get("rho", np.nan))
        if name.startswith("J_vs_clean"):
            clean_order = list(clean_interval_metadata.get("association_order", ()))
            if clean_association_interval is not None and name in clean_order:
                position = clean_order.index(name)
                low = float(clean_association_interval.low[position])
                high = float(clean_association_interval.high[position])
            else:
                low = high = np.nan
        elif (
            causal_association_low is not None
            and causal_association_high is not None
            and name in causal_association_order
        ):
            position = causal_association_order.index(name)
            low = float(np.asarray(causal_association_low)[position])
            high = float(np.asarray(causal_association_high)[position])
        else:
            low = high = np.nan
        decision = (
            "positive"
            if np.isfinite(low) and low > importance_floor
            else "negative"
            if np.isfinite(high) and high < -importance_floor
            else "unresolved"
        )
        activity_validation[name] = {
            "estimate": estimate,
            "low": low,
            "high": high,
            "importance_correlation_floor": importance_floor,
            "decision": decision,
        }
    importance_validated = bool(
        activity_validation
        and all(
            record["decision"] == "positive"
            for record in activity_validation.values()
        )
    )

    score_diagnostics = scores.get("specialisation_diagnostics", {})
    discovery_interpretation = score_diagnostics.get("interpretation", {})
    decisions = [
        record["decision"] for record in family_interactions.values()
    ]
    all_interactions_equivalent = bool(
        decisions and all(value == "equivalent" for value in decisions)
    )
    directional_specialisation = bool(
        family_interactions.get("gross_family_by_channel", {}).get("decision")
        == "specialised"
        and any(
            family_interactions.get(name, {}).get("decision") == "specialised"
            for name in (
                "rescue_family_by_channel",
                "induction_family_by_channel",
                "necessity_family_by_channel",
            )
        )
    )
    central_dual_response = bool(
        core_profile.get("gross_response", {}).get("dual_channel")
        and core_profile.get("causal_rescue", {}).get("dual_channel")
    )
    entangled_generalist = bool(
        discovery_interpretation.get("entanglement_compatible")
        and importance_validated
        and all_interactions_equivalent
        and central_dual_response
    )
    if directional_specialisation:
        regime = "confirmed_causal_specialisation"
    elif entangled_generalist:
        regime = "confirmed_entangled_generalist"
    else:
        regime = "mixed_or_unresolved"
    regime_evidence = {
        "regime": regime,
        "discovery_status": discovery_interpretation.get(
            "status", "not_available"
        ),
        "checks": {
            "discovery_entanglement_compatible": bool(
                discovery_interpretation.get("entanglement_compatible", False)
            ),
            "J_predicts_clean_and_causal_importance": importance_validated,
            "all_family_interactions_equivalent": all_interactions_equivalent,
            "central_family_dual_channel_response_and_rescue": central_dual_response,
            "directional_specialisation": directional_specialisation,
        },
        "family_interactions": family_interactions,
        "activity_validation": activity_validation,
        "central_generalist_core": core_profile,
        "decision_rule": (
            "specialisation requires a practically positive gross family-by-channel interval "
            "and at least one practically positive directional/necessity interaction; "
            "entangled-generalist requires discovery entanglement evidence, practically positive "
            "J-to-clean/gross/necessity rank correlations, every registered family interaction "
            "wholly inside its equivalence region, and reference-scaled dual-channel response plus "
            "rescue by the high-J central family"
        ),
    }
    output = {
        "protocol_version": PROTOCOL_VERSION,
        "clean_ablation": clean,
        "event_records": events["records"],
        "same_condition_patch_max": events["same_condition_patch_max"],
        "uncontrolled_events_excluded": events["uncontrolled_events_excluded"],
        "summary": summary,
        "associations": associations,
        "families": scores["families"],
        "matched_controls": scores.get("matched_controls", {}),
        "family_interactions": family_interactions,
        "regime_evidence": regime_evidence,
    }
    cache.save("causal", "validation", output)
    cache.save_audit("causal_manifest", plan)
    return output
