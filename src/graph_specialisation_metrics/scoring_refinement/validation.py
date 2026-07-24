"""Held-out ablation, causal patching, and method-comparison utilities."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Mapping, Sequence

import numpy as np


EPS = 1.0e-12


def _spearman(left: Any, right: Any) -> float:
    from scipy.stats import spearmanr

    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= EPS or np.std(right[valid]) <= EPS:
        return float("nan")
    return float(spearmanr(left[valid], right[valid]).statistic)


def _pearson(left: Any, right: Any) -> float:
    from scipy.stats import pearsonr

    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or np.std(left[valid]) <= EPS or np.std(right[valid]) <= EPS:
        return float("nan")
    return float(pearsonr(left[valid], right[valid]).statistic)


def _layer_center(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    return value - np.nanmean(value, axis=1, keepdims=True)


def _within_layer_spearman(left: np.ndarray, right: np.ndarray) -> float:
    values = [
        _spearman(left[layer], right[layer])
        for layer in range(int(left.shape[0]))
    ]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _topk_overlap(left: np.ndarray, right: np.ndarray, k: int) -> float:
    left_flat = np.asarray(left, dtype=float).reshape(-1)
    right_flat = np.asarray(right, dtype=float).reshape(-1)
    if left_flat.shape != right_flat.shape:
        raise ValueError(
            f"top-k arrays must align, got {left_flat.shape} and {right_flat.shape}"
        )
    valid = np.isfinite(left_flat) & np.isfinite(right_flat)
    left_flat = left_flat[valid]
    right_flat = right_flat[valid]
    if not len(left_flat):
        return float("nan")
    count = min(int(k), len(left_flat))
    a = set(np.argsort(left_flat)[-count:].tolist())
    b = set(np.argsort(right_flat)[-count:].tolist())
    return float(len(a & b) / max(count, 1))


def _bootstrap_correlation(
    left_graph: np.ndarray,
    right_graph: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    count = min(int(left_graph.shape[0]), int(right_graph.shape[0]))
    if count < 2:
        return float("nan"), float("nan")
    values = []
    for _ in range(int(samples)):
        indices = rng.choice(count, size=count, replace=True)
        values.append(
            _spearman(
                np.mean(left_graph[indices], axis=0),
                np.mean(right_graph[indices], axis=0),
            )
        )
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))


def _bootstrap_topk_overlap(
    left_graph: np.ndarray,
    right_graph: np.ndarray,
    *,
    k: int,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(int(seed))
    count = min(int(left_graph.shape[0]), int(right_graph.shape[0]))
    if count < 2:
        return float("nan"), float("nan"), float("nan")
    values = []
    for _ in range(int(samples)):
        indices = rng.choice(count, size=count, replace=True)
        values.append(
            _topk_overlap(
                np.mean(left_graph[indices], axis=0),
                np.mean(right_graph[indices], axis=0),
                int(k),
            )
        )
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    low, high = np.quantile(values, [0.025, 0.975])
    return float(np.mean(values)), float(low), float(high)


def build_variant_agreement(
    score: Mapping[str, np.ndarray],
    per_graph: Mapping[str, np.ndarray],
    *,
    top_k: Sequence[int],
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    comparisons = (
        (
            "semantic",
            "semantic_single_donor",
            "semantic_transposition",
            "eg_semantic_single",
            "eg_semantic_transposition",
        ),
        (
            "pe",
            "pe_single_donor",
            "pe_transposition",
            "eg_pe_single",
            "eg_pe_transposition",
        ),
    )
    rows = []
    for channel, donor_name, transposition_name, left_key, right_key in comparisons:
        left = np.asarray(score[left_key], dtype=float)
        right = np.asarray(score[right_key], dtype=float)
        low, high = _bootstrap_correlation(
            np.asarray(per_graph[left_key], dtype=float),
            np.asarray(per_graph[right_key], dtype=float),
            samples=bootstrap_samples,
            seed=seed + len(rows),
        )
        base = {
            "comparison_type": "intervention_variant",
            "channel": channel,
            "donor_variant": donor_name,
            "transposition_variant": transposition_name,
            "pearson": _pearson(left, right),
            "spearman": _spearman(left, right),
            "within_layer_spearman": _within_layer_spearman(left, right),
            "layer_centered_spearman": _spearman(_layer_center(left), _layer_center(right)),
            "bootstrap_spearman_low": low,
            "bootstrap_spearman_high": high,
            "mean_absolute_difference": float(np.mean(np.abs(left - right))),
            "log1p_spearman": _spearman(np.log1p(left), np.log1p(right)),
        }
        for k in top_k:
            base[f"top_{int(k)}_overlap"] = _topk_overlap(left, right, int(k))
            mean, overlap_low, overlap_high = _bootstrap_topk_overlap(
                np.asarray(per_graph[left_key], dtype=float),
                np.asarray(per_graph[right_key], dtype=float),
                k=int(k),
                samples=bootstrap_samples,
                seed=seed + 101 * int(k) + len(rows),
            )
            base[f"top_{int(k)}_bootstrap_mean"] = mean
            base[f"top_{int(k)}_bootstrap_low"] = overlap_low
            base[f"top_{int(k)}_bootstrap_high"] = overlap_high
        rows.append(base)
    return rows


def build_m1_arm_agreement(
    derived_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: Sequence[int],
) -> list[dict[str, Any]]:
    """Pairwise D/J rank agreement across the four M1 intervention arms."""

    arm_order = ("M1_DD", "M1_DT", "M1_TD", "M1_TT")
    present = {str(row["method"]) for row in derived_rows}
    arms = tuple(arm for arm in arm_order if arm in present)
    values: dict[str, dict[str, np.ndarray]] = {}
    for arm in arms:
        rows = [row for row in derived_rows if str(row["method"]) == arm]
        rows.sort(key=lambda row: (int(row["layer"]), int(row["head"])))
        values[arm] = {
            "D_rel": np.asarray([float(row["D_rel"]) for row in rows]),
            "J": np.asarray([float(row["J"]) for row in rows]),
        }
    output = []
    for left_index, left in enumerate(arms):
        for right in arms[left_index + 1:]:
            row = {
                "comparison_type": "m1_arm",
                "left_arm": left,
                "right_arm": right,
                "D_rel_spearman": _spearman(
                    values[left]["D_rel"], values[right]["D_rel"]
                ),
                "J_spearman": _spearman(values[left]["J"], values[right]["J"]),
            }
            for k in top_k:
                row[f"D_rel_top_{int(k)}_overlap"] = _topk_overlap(
                    np.abs(values[left]["D_rel"]),
                    np.abs(values[right]["D_rel"]),
                    int(k),
                )
                row[f"J_top_{int(k)}_overlap"] = _topk_overlap(
                    values[left]["J"], values[right]["J"], int(k)
                )
            output.append(row)
    return output


def build_m1_cross_method_correlations(
    raw_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Correlate every M1 arm with the role-aligned raw axes of M2--M6."""

    grouped: dict[str, dict[tuple[int, int], Mapping[str, Any]]] = defaultdict(dict)
    for row in raw_rows:
        grouped[str(row["method"])][
            (int(row["layer"]), int(row["head"]))
        ] = row
    m1_arms = [
        arm for arm in ("M1_DD", "M1_DT", "M1_TD", "M1_TT") if arm in grouped
    ]
    comparisons = [
        method for method in ("M2", "M3", "M4", "M5", "M6") if method in grouped
    ]
    output = []
    for arm in m1_arms:
        for axis, field in (
            ("semantic", "semantic_score"),
            ("structural", "pe_score"),
        ):
            for method in comparisons:
                keys = sorted(set(grouped[arm]) & set(grouped[method]))
                left = np.asarray(
                    [float(grouped[arm][key][field]) for key in keys], dtype=float
                )
                right = np.asarray(
                    [float(grouped[method][key][field]) for key in keys], dtype=float
                )
                layers = np.asarray([key[0] for key in keys], dtype=int)
                within = [
                    _spearman(left[layers == layer], right[layers == layer])
                    for layer in sorted(set(layers.tolist()))
                ]
                within = [value for value in within if np.isfinite(value)]
                left_centered = left.copy()
                right_centered = right.copy()
                for layer in sorted(set(layers.tolist())):
                    mask = layers == layer
                    left_centered[mask] -= np.nanmean(left[mask])
                    right_centered[mask] -= np.nanmean(right[mask])
                output.append(
                    {
                        "comparison_type": "m1_cross_method",
                        "m1_arm": arm,
                        "axis": axis,
                        "m1_field": field,
                        "comparison_method": method,
                        "comparison_field": field,
                        "heads": len(keys),
                        "pearson": _pearson(left, right),
                        "spearman": _spearman(left, right),
                        "within_layer_spearman": (
                            float(np.mean(within)) if within else float("nan")
                        ),
                        "layer_centered_spearman": _spearman(
                            left_centered, right_centered
                        ),
                    }
                )
    return output


def _batch_groups(items: Sequence[Any], size: int = 32) -> list[list[Any]]:
    return [list(items[start:start + size]) for start in range(0, len(items), size)]


def _targets(items: Sequence[Any]) -> np.ndarray:
    return np.concatenate(
        [item.y.detach().cpu().numpy().reshape(1, -1) for item in items],
        axis=0,
    )


def _per_graph_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    loss_fun: str,
) -> np.ndarray:
    """Delegate task loss semantics to the shared specialisation adapter."""

    from ..specialisation.channel_ablation import per_graph_loss_np

    return per_graph_loss_np(prediction, target, loss_fun)


def _bootstrap_interval(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    means = [
        float(np.mean(values[rng.choice(len(values), size=len(values), replace=True)]))
        for _ in range(int(samples))
    ]
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def _graph_means(values: Sequence[Mapping[str, float]], key: str) -> np.ndarray:
    grouped: dict[int, list[float]] = defaultdict(list)
    for item in values:
        grouped[int(item["graph_id"])].append(float(item[key]))
    output = []
    for graph_id in sorted(grouped):
        group = np.asarray(grouped[graph_id], dtype=float)
        finite = group[np.isfinite(group)]
        output.append(float(np.mean(finite)) if len(finite) else float("nan"))
    return np.asarray(output, dtype=float)


def run_single_head_ablation(
    gm: Any,
    graph_ids: Sequence[int],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Ablate each full routed head independently on held-out clean graphs."""

    items = [gm.eval_ds[int(graph_id)] for graph_id in graph_ids]
    groups = _batch_groups(items)
    target = _targets(items)
    clean = gm.collect_preds_ablated(groups)
    clean_loss = _per_graph_loss(clean, target, gm.loss_fun)
    movement = np.zeros((len(items), gm.L, gm.H), dtype=np.float64)
    loss_increase = np.zeros_like(movement)
    rows = []
    for layer in range(gm.L):
        for head in range(gm.H):
            ablated = gm.collect_preds_ablated(groups, [(layer, head)])
            graph_movement = np.mean(np.abs(ablated - clean), axis=1)
            graph_loss = _per_graph_loss(ablated, target, gm.loss_fun) - clean_loss
            movement[:, layer, head] = graph_movement
            loss_increase[:, layer, head] = graph_loss
            low, high = _bootstrap_interval(
                graph_movement, bootstrap_samples, seed + 1009 * layer + head
            )
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "graphs": len(items),
                    "prediction_movement": float(np.mean(graph_movement)),
                    "loss_increase": float(np.mean(graph_loss)),
                    "prediction_movement_ci_low": low,
                    "prediction_movement_ci_high": high,
                }
            )
    return rows, {
        "prediction_movement": movement,
        "loss_increase": loss_increase,
        "clean_prediction": clean,
        "target": target,
    }


@contextmanager
def _patched_head(gm: Any, layer: int, head: int, source: Any):
    """Replace one head's full carrier-aligned routed output at one layer."""

    import torch

    replacement = source.to(gm.device)

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, (tuple, list)):
            routed, edge = output
            patched = routed.clone()
            patched[:, int(head), :] = replacement[:, int(head), :].to(patched)
            return type(output)((patched, edge)) if isinstance(output, list) else (patched, edge)
        patched = output.clone()
        patched[:, int(head), :] = replacement[:, int(head), :].to(patched)
        return patched

    handle = gm.attn_layers[int(layer)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _forward_prediction(
    gm: Any,
    data: Any,
    patch: tuple[int, int, Any] | None = None,
) -> np.ndarray:
    import torch
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([data.clone()]).to(gm.device)
    context = (
        _patched_head(gm, patch[0], patch[1], patch[2])
        if patch is not None
        else _nullcontext()
    )
    with context, torch.no_grad():
        prediction, _ = gm.model(batch)
    return prediction.detach().cpu().numpy().reshape(-1)


@contextmanager
def _nullcontext():
    yield


def _capture_transport(gm: Any, data: Any) -> tuple[np.ndarray, list[Any]]:
    import torch
    from torch_geometric.data import Batch

    batch = Batch.from_data_list([data.clone()]).to(gm.device)
    with torch.no_grad():
        capture = gm.capture(batch, want_grad=False, want_attn=False)
    prediction = capture["pred"].detach().cpu().numpy().reshape(-1)
    return prediction, [value.detach() for value in capture["wV"]]


def run_causal_patching(
    gm: Any,
    events: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Restore/inject every head on held-out semantic, PE, and topology events."""

    observations: dict[tuple[str, int, int], list[dict[str, float]]] = defaultdict(list)
    sham_max = 0.0
    captured_events = []
    for event in events:
        base = event["base"]
        variant = event["variant_data"]
        clean_prediction, clean_wv = _capture_transport(gm, base)
        corrupt_prediction, corrupt_wv = _capture_transport(gm, variant)
        captured_events.append(
            {
                **event,
                "clean_prediction": clean_prediction,
                "clean_wv": clean_wv,
                "corrupt_prediction": corrupt_prediction,
                "corrupt_wv": corrupt_wv,
            }
        )
    for event_index, event in enumerate(captured_events):
        base = event["base"]
        variant = event["variant_data"]
        channel = str(event["channel"])
        clean_prediction = event["clean_prediction"]
        clean_wv = event["clean_wv"]
        corrupt_prediction = event["corrupt_prediction"]
        corrupt_wv = event["corrupt_wv"]
        total_effect = float(np.mean(np.abs(corrupt_prediction - clean_prediction)))
        wrong = next(
            (
                candidate
                for candidate in captured_events
                if int(candidate.get("graph_id", -1)) != int(event.get("graph_id", -1))
                and int(candidate["base"].num_nodes) == int(base.num_nodes)
            ),
            None,
        )
        for layer in range(gm.L):
            for head in range(gm.H):
                restore = _forward_prediction(
                    gm, variant, (layer, head, clean_wv[layer])
                )
                inject = _forward_prediction(
                    gm, base, (layer, head, corrupt_wv[layer])
                )
                sham = _forward_prediction(
                    gm, variant, (layer, head, corrupt_wv[layer])
                )
                zero = corrupt_wv[layer].new_zeros(corrupt_wv[layer].shape)
                clean_zero = _forward_prediction(gm, base, (layer, head, zero))
                corrupt_zero = _forward_prediction(gm, variant, (layer, head, zero))
                zero_effect = float(np.mean(np.abs(corrupt_zero - clean_zero)))
                necessity = total_effect - zero_effect
                if wrong is not None:
                    wrong_restore = _forward_prediction(
                        gm, variant, (layer, head, wrong["clean_wv"][layer])
                    )
                    wrong_reduction = total_effect - float(
                        np.mean(np.abs(wrong_restore - clean_prediction))
                    )
                else:
                    wrong_reduction = float("nan")
                residual = float(np.mean(np.abs(restore - clean_prediction)))
                injection = float(np.mean(np.abs(inject - clean_prediction)))
                restore_reduction = total_effect - residual
                mediation = 0.5 * (restore_reduction + injection)
                sham_error = float(np.max(np.abs(sham - corrupt_prediction)))
                sham_max = max(sham_max, sham_error)
                observations[(channel, layer, head)].append(
                    {
                        "mediation": mediation,
                        "restore_reduction": restore_reduction,
                        "injection": injection,
                        "necessity": necessity,
                        "wrong_graph_reduction": wrong_reduction,
                        "total_effect": total_effect,
                        "sham_error": sham_error,
                        "event": float(event_index),
                        "graph_id": float(event.get("graph_id", event_index)),
                    }
                )

    channels = sorted({key[0] for key in observations})
    arrays = {
        channel: np.full((gm.L, gm.H), np.nan, dtype=float) for channel in channels
    }
    rows = []
    for channel in channels:
        for layer in range(gm.L):
            for head in range(gm.H):
                values = observations.get((channel, layer, head), [])
                mediation = _graph_means(values, "mediation")
                if not len(mediation):
                    continue
                arrays[channel][layer, head] = float(np.mean(mediation))
                low, high = _bootstrap_interval(
                    mediation,
                    bootstrap_samples,
                    seed + 100_003 * (channels.index(channel) + 1) + 101 * layer + head,
                )
                rows.append(
                    {
                        "channel": channel,
                        "layer": layer,
                        "head": head,
                        "events": len(values),
                        "graphs": len(mediation),
                        "mediation": float(np.mean(mediation)),
                        "restore_reduction": float(
                            np.mean(_graph_means(values, "restore_reduction"))
                        ),
                        "injection": float(np.mean(_graph_means(values, "injection"))),
                        "necessity": float(np.mean(_graph_means(values, "necessity"))),
                        "wrong_graph_reduction": float(
                            np.nanmean(_graph_means(values, "wrong_graph_reduction"))
                        )
                        if np.any(
                            np.isfinite(
                                _graph_means(values, "wrong_graph_reduction")
                            )
                        )
                        else float("nan"),
                        "total_event_effect": float(
                            np.mean(_graph_means(values, "total_effect"))
                        ),
                        "mediation_ci_low": low,
                        "mediation_ci_high": high,
                        "sham_error_max": float(
                            np.max([item["sham_error"] for item in values])
                        ),
                    }
                )
    semantic = arrays.get("semantic", np.zeros((gm.L, gm.H)))
    pe = arrays.get("pe", np.zeros((gm.L, gm.H)))
    arrays["differential"] = semantic - pe
    arrays["total"] = 0.5 * (semantic + pe)
    arrays["sham_error_max"] = np.asarray(sham_max)
    return rows, arrays


def _matrix_from_rows(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    shape: tuple[int, int],
) -> np.ndarray:
    output = np.full(shape, np.nan, dtype=float)
    for row in rows:
        output[int(row["layer"]), int(row["head"])] = float(row[key])
    return output


def _partial_spearman_layer(
    left: np.ndarray,
    right: np.ndarray,
    throughput: np.ndarray,
) -> float:
    from scipy.stats import rankdata

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    throughput = np.asarray(throughput, dtype=float)
    if left.shape != right.shape or left.shape != throughput.shape:
        raise ValueError(
            "partial-correlation arrays must share a [layer, head] shape; "
            f"got {left.shape}, {right.shape}, and {throughput.shape}"
        )
    layer_count = int(left.shape[0])
    layer = np.repeat(np.arange(layer_count), left.shape[1])
    left = left.reshape(-1)
    right = right.reshape(-1)
    throughput = throughput.reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right) & np.isfinite(throughput)
    if int(valid.sum()) < 3:
        return float("nan")
    layer = layer[valid]
    x = rankdata(left[valid])
    y = rankdata(right[valid])
    activity = rankdata(throughput[valid])
    design = np.column_stack(
        [
            np.ones(len(layer)),
            *[(layer == value).astype(float) for value in range(1, layer_count)],
            activity,
        ]
    )
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return _pearson(x_residual, y_residual)


def _within_layer_permutation_p(
    score: np.ndarray,
    target: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> float:
    observed = abs(_spearman(score, target))
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(int(permutations)):
        shuffled = np.stack([rng.permutation(row) for row in score], axis=0)
        exceed += abs(_spearman(shuffled, target)) >= observed
    return float((exceed + 1) / (int(permutations) + 1))


def compare_methods(
    derived_rows: Sequence[Mapping[str, Any]],
    ablation_rows: Sequence[Mapping[str, Any]],
    causal_arrays: Mapping[str, np.ndarray],
    throughput: np.ndarray,
    *,
    top_k: Sequence[int],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply identical held-out significance and role criteria to every method."""

    shape = tuple(int(value) for value in throughput.shape)
    ablation = _matrix_from_rows(ablation_rows, "prediction_movement", shape)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in derived_rows:
        grouped[str(row["method"])].append(row)
    ablation_validation = []
    causal_validation = []
    ranking = []
    for method, rows in sorted(grouped.items()):
        d_rel = _matrix_from_rows(rows, "D_rel", shape)
        joint = _matrix_from_rows(rows, "J", shape)
        significance = {
            "method": method,
            "J_ablation_spearman": _spearman(joint, ablation),
            "J_ablation_within_layer_spearman": _within_layer_spearman(joint, ablation),
            "J_ablation_partial_spearman": _partial_spearman_layer(
                joint, ablation, throughput
            ),
        }
        for k in top_k:
            significance[f"top_{int(k)}_ablation_overlap"] = _topk_overlap(
                joint, ablation, int(k)
            )
        ablation_validation.append(significance)
        role = {
            "method": method,
            "D_differential_mediation_spearman": _spearman(
                d_rel, causal_arrays["differential"]
            ),
            "D_differential_within_layer_spearman": _within_layer_spearman(
                d_rel, causal_arrays["differential"]
            ),
            "D_differential_permutation_p": _within_layer_permutation_p(
                d_rel,
                causal_arrays["differential"],
                permutations=999,
                seed=seed + len(causal_validation),
            ),
            "J_total_mediation_spearman": _spearman(joint, causal_arrays["total"]),
        }
        causal_validation.append(role)
        ranking.append(
            {
                "method": method,
                "significance_score": significance["J_ablation_spearman"],
                "role_score": role["D_differential_mediation_spearman"],
                "significance_rank": 0,
                "role_rank": 0,
            }
        )
    for field, rank_field in (
        ("significance_score", "significance_rank"),
        ("role_score", "role_rank"),
    ):
        order = sorted(
            range(len(ranking)),
            key=lambda index: (
                -np.nan_to_num(float(ranking[index][field]), nan=-np.inf),
                ranking[index]["method"],
            ),
        )
        for rank, index in enumerate(order, start=1):
            ranking[index][rank_field] = rank
    return ablation_validation, causal_validation, ranking
