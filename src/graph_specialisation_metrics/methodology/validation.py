"""Held-out clean ablation, donor-wise necessity, and bidirectional patch validation."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import numpy as np

from .bootstrap import Observation, nested_percentile_interval
from .causal import (
    calibrated_targets,
    clean_ablation,
    donor_necessity,
    mismatch_adjusted_aligned,
    mismatch_adjusted_gross,
    patch_response,
    reference_scale,
)
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
    return result


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


def _mismatch_indices(records: Sequence[Any]) -> list[int]:
    """Prefer another donor event for the same source."""

    result: list[int] = []
    for position, record in enumerate(records):
        candidates = [
            other
            for other, candidate in enumerate(records)
            if candidate.source == record.source
            and candidate.payload_fingerprint != record.payload_fingerprint
            and other != position
        ]
        if not candidates:
            candidates = [
                other
                for other, candidate in enumerate(records)
                if candidate.payload_fingerprint != record.payload_fingerprint
                and other != position
            ]
        if not candidates:
            raise RuntimeError(
                "causal mismatch controls require at least two distinct donor payloads/footprints"
            )
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
    return result


def _clean_ablation_stage(
    prepared: Any,
    config: Any,
    targets: Mapping[str, Sequence[tuple[int, int]]],
) -> dict[str, Any]:
    from .runner import _stage_ids

    output: dict[str, Any] = {}
    for name, family in targets.items():
        rows = []
        clean_predictions = []
        ablated_predictions = []
        truths = []
        for graph_id in _stage_ids(prepared, "clean_ablation"):
            base = prepared.grit.eval_ds[int(graph_id)]
            clean = prepared.backend.capture([base], require_grad=False)
            prediction_a, z_a, target_a = prepared.backend.ablate([base], family)
            loss_clean = prepared.backend.loss_per_graph(
                clean.prediction, clean.target
            ).detach().cpu().numpy()
            loss_ablated = prepared.backend.loss_per_graph(
                prediction_a, target_a
            ).detach().cpu().numpy()
            endpoint = clean_ablation(
                clean.z.detach().cpu().numpy(),
                z_a.detach().cpu().numpy(),
                loss_clean,
                loss_ablated,
            )
            clean_predictions.append(clean.prediction.detach().cpu().numpy().reshape(1, -1))
            ablated_predictions.append(
                prediction_a.detach().cpu().numpy().reshape(1, -1)
            )
            truths.append(clean.target.detach().cpu().numpy().reshape(1, -1))
            rows.append(
                {
                    "graph": int(graph_id),
                    "prediction_movement": float(endpoint["prediction_movement"][0]),
                    "loss_change": float(endpoint["loss_change"][0]),
                }
            )
        metric_clean = float(
            prepared.task.grit.metric_fn(
                np.concatenate(clean_predictions), np.concatenate(truths)
            )
        )
        metric_ablated = float(
            prepared.task.grit.metric_fn(
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
    output["_intervals"] = {
        "target_order": names,
        "endpoint_order": ("prediction_movement", "loss_change"),
        "interval": nested_percentile_interval(vector_observations, config.bootstrap),
    }
    return output


def _causal_events(
    prepared: Any,
    config: Any,
    scores: Mapping[str, Any],
    targets: Mapping[str, Sequence[tuple[int, int]]],
    plan: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    from .runner import _rebuild_graph_events

    records_by_target: dict[str, dict[str, list[dict[str, Any]]]] = {
        target: {channel: [] for channel in CHANNELS} for target in targets
    }
    self_patch_max = 0.0
    for graph_id in sorted(plan):
        base = prepared.grit.eval_ds[int(graph_id)]
        for channel in CHANNELS:
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
            mismatch = _mismatch_indices(records)
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
                # Same-condition patch is a numerical audit, not a control endpoint.
                self_replacement = prepared.backend.replacement_batch(captured, [0])
                _, self_z, _ = prepared.backend.patch(base, self_replacement, family)
                self_patch_max = max(
                    self_patch_max,
                    float(
                        np.max(
                            np.abs(self_z.detach().cpu().numpy() - z_clean)
                        )
                    ),
                )

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
                rows = records_by_target[target_name][channel]
                for position, record in enumerate(records):
                    rows.append(
                        {
                            "graph": int(graph_id),
                            "source": int(record.source),
                            "donor": int(record.draw),
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
    if self_patch_max > config.numerical.reconstruction_tolerance:
        raise RuntimeError(
            f"same-condition activation patch was not zero (max {self_patch_max:.3e})"
        )
    return {
        "records": records_by_target,
        "same_condition_patch_max": self_patch_max,
    }


def _summarize_causal(
    event_output: Mapping[str, Any],
    targets: Mapping[str, Sequence[tuple[int, int]]],
    config: Any,
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
    for target in targets:
        summary[target]["calibrated"] = {
            key: float(value)
            for key, value in calibrated_targets(
                summary[target]["semantic"]["G_c"],
                summary[target]["structural"]["G_c"],
                summary[target]["semantic"]["necessity"],
                summary[target]["structural"]["necessity"],
                gross_scales=gross_scales,
                necessity_scales=necessity_scales,
            ).items()
        }
    endpoint_order = (
        "G_c",
        "P_gross_matched",
        "P_gross_mismatch",
        "R_gross",
        "I_gross",
        "R_align",
        "I_align",
        "necessity",
        "gross_necessity",
    )
    target_order = list(targets)
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
    complete = [
        (key, value) for key, value in sorted(by_channel_key.items()) if np.isfinite(value).all()
    ]
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

        def transform(value):
            # Recompute positive unadjusted reference scales in every bootstrap draw.
            a_g = np.asarray(
                [
                    np.mean(value[channel, head_positions, endpoint_order.index("P_gross_matched")])
                    for channel in range(len(CHANNELS))
                ]
            )
            a_n = np.asarray(
                [
                    np.mean(value[channel, head_positions, endpoint_order.index("gross_necessity")])
                    for channel in range(len(CHANNELS))
                ]
            )
            G = value[:, :, endpoint_order.index("G_c")]
            N = value[:, :, endpoint_order.index("necessity")]
            calibrated = np.stack(
                (
                    0.5 * (G[0] / a_g[0] + G[1] / a_g[1]),
                    G[0] / a_g[0] - G[1] / a_g[1],
                    0.5 * (N[0] / a_n[0] + N[1] / a_n[1]),
                    N[0] / a_n[0] - N[1] / a_n[1],
                ),
                axis=-1,
            )
            return np.concatenate((value.reshape(-1), calibrated.reshape(-1)))

        causal_interval = nested_percentile_interval(
            causal_observations, config.bootstrap, transform=transform
        )
    else:
        causal_interval = None
    return {
        "targets": summary,
        "gross_reference_scales": gross_scales,
        "necessity_reference_scales": necessity_scales,
        "intervals": {
            "target_order": target_order,
            "channel_order": CHANNELS,
            "endpoint_order": endpoint_order,
            "calibrated_order": (
                "gross_total_for_J",
                "gross_contrast_for_D_rel",
                "necessity_total_for_J",
                "necessity_contrast_for_D_rel",
            ),
            "interval": causal_interval,
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


def run_causal_validation(
    prepared: Any,
    config: Any,
    scores: Mapping[str, Any],
) -> dict[str, Any]:
    """Run all-head validation and frozen-family causal confirmation on disjoint graphs."""

    from .cache import CanonicalCache
    from .protocol import stable_hash
    from .runner import _cache, _stage_plan

    if scores is None:
        raise ValueError("causal validation requires discovery scores")
    plan = _stage_plan(prepared, config, "causal")
    base_cache = _cache(prepared, config, plan)
    causal_manifest = stable_hash(
        {
            "events": base_cache.contract.event_manifest_hash,
            "families": scores["families"],
            "matched_controls": scores.get("matched_controls", {}),
        }
    )
    cache = CanonicalCache(
        config.root,
        dataclasses.replace(
            base_cache.contract, event_manifest_hash=causal_manifest
        ),
    )
    if config.resume and not config.force:
        cached = cache.load("causal", "validation")
        if cached is not None:
            return cached
    targets = _targets(prepared, scores)
    clean = _clean_ablation_stage(prepared, config, targets)
    events = _causal_events(prepared, config, scores, targets, plan)
    summary = _summarize_causal(events, targets, config)
    associations = _association_report(prepared, scores, summary, clean, config)
    target_summary = summary["targets"]
    sem_name = "family_semantic_leaning"
    str_name = "family_structural_leaning"
    family_interactions = {}
    if sem_name in target_summary and str_name in target_summary:
        semantic_focus = (
            target_summary[sem_name]["calibrated"]["g_semantic"]
            - target_summary[sem_name]["calibrated"]["g_structural"]
        )
        structural_focus = (
            target_summary[str_name]["calibrated"]["g_semantic"]
            - target_summary[str_name]["calibrated"]["g_structural"]
        )
        family_interactions["gross_score_validation"] = float(
            semantic_focus - structural_focus
        )
        a_g = summary["gross_reference_scales"]
        sem_aligned = (
            target_summary[sem_name]["semantic"]["M_align"] / a_g["semantic"]
            - target_summary[sem_name]["structural"]["M_align"] / a_g["structural"]
        )
        str_aligned = (
            target_summary[str_name]["semantic"]["M_align"] / a_g["semantic"]
            - target_summary[str_name]["structural"]["M_align"] / a_g["structural"]
        )
        family_interactions["aligned_rescue_induction"] = float(
            sem_aligned - str_aligned
        )
    output = {
        "protocol_version": PROTOCOL_VERSION,
        "clean_ablation": clean,
        "event_records": events["records"],
        "same_condition_patch_max": events["same_condition_patch_max"],
        "summary": summary,
        "associations": associations,
        "families": scores["families"],
        "matched_controls": scores.get("matched_controls", {}),
        "family_interactions": family_interactions,
    }
    cache.save("causal", "validation", output)
    cache.save_audit("causal_manifest", plan)
    return output
