"""Held-out, channel-specific causal mediation for per-head GRIT scores.

This module is deliberately an *extension* of the existing specialisation analysis.  It does
not change the score estimator or the legacy ablation path.  Given the dictionary returned by
``scores.score_model``, it

* calibrates semantic and RRWP-role scores into an importance coordinate ``I`` and a bounded
  channel-preference coordinate ``q``;
* constructs a frozen confirmation bank disjoint from the graphs used to discover the scores;
* performs bidirectional interchange interventions at the same per-head ``wV`` transport site
  used by the score (counterfactual -> clean noising and clean -> counterfactual denoising);
* evaluates every single head, cumulative top-k specialised groups, and layer/importance-matched
  control groups; and
* reports graph-cluster-bootstrap mediation fractions, signed task-loss effects, double
  dissociations, and group-minus-singles interactions.

The primary mediation estimator is a stable ratio of sums.  For total model response
``delta = pred_cf - pred_clean`` and a patched displacement ``m`` it is

    beta = sum_e <m_e, delta_e> / sum_e ||delta_e||^2.

For noising, ``m = pred_noise - pred_clean``.  For denoising,
``m = pred_cf - pred_denoise``.  The estimator is not clipped: overshoot and sign reversal are
real diagnostics.  Bootstrap resampling is over whole molecular graphs, so all channels,
directions, interventions, heads, and groups for a molecule remain paired.

The returned object contains only numpy arrays, Python scalars, lists, and dictionaries.  Live
PyG ``Data`` objects and torch tensors are retained only while the runner is executing, making
the result straightforward to persist as NPZ/JSON artefacts by the Colab orchestration layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..carriage import metrics, structural
from ..carriage.env import log


SCHEMA_VERSION = 1
CHANNELS = ("semantic", "rrwp_role")
DIRECTIONS = ("noising", "denoising")
_EPS = 1.0e-12
_DEFAULT_BATCH_SIZE = 32


@dataclass
class _InterventionEvent:
    """Runtime-only clean/counterfactual pair plus persistence-safe metadata."""

    metadata: dict[str, Any]
    clean: Any
    counterfactual: Any


def _head_order(L: int, H: int) -> list[tuple[int, int]]:
    return [(layer, head) for layer in range(int(L)) for head in range(int(H))]


def calibrate_head_scores(
    S_sem: np.ndarray,
    S_rrwp: np.ndarray,
    *,
    importance_quantile: float = 0.5,
    min_eligible: int = 1,
) -> dict[str, Any]:
    """Separate overall transport importance from semantic/RRWP channel preference.

    Each positive channel is divided by its own across-head mean.  This makes the preference
    invariant to a global rescaling of either intervention family while retaining within-channel
    head ordering.  ``I = s + r`` measures overall score magnitude and
    ``q = (s-r)/(s+r+eps)`` lies in [-1, 1].  Preference rankings are restricted to heads above an
    importance floor so a nearly inert head cannot look specialised because of a noisy ratio.
    """

    sem = np.asarray(S_sem, dtype=np.float64)
    rrwp = np.asarray(S_rrwp, dtype=np.float64)
    if sem.shape != rrwp.shape or sem.ndim != 2:
        raise ValueError(
            f"score matrices must have equal [L,H] shape, got {sem.shape} and {rrwp.shape}"
        )
    if not np.isfinite(sem).all() or not np.isfinite(rrwp).all():
        raise ValueError("specialisation scores must be finite")
    if (sem < 0).any() or (rrwp < 0).any():
        raise ValueError("specialisation scores are non-negative magnitudes")
    if not 0.0 <= float(importance_quantile) <= 1.0:
        raise ValueError("importance_quantile must lie in [0,1]")

    sem_scale = float(sem.mean())
    rrwp_scale = float(rrwp.mean())
    if sem_scale <= _EPS or rrwp_scale <= _EPS:
        raise ValueError(
            "both channels need non-zero mean score for calibrated selectivity "
            f"(semantic={sem_scale:.3e}, rrwp_role={rrwp_scale:.3e})"
        )
    sem_cal = sem / sem_scale
    rrwp_cal = rrwp / rrwp_scale
    importance = sem_cal + rrwp_cal
    preference = (sem_cal - rrwp_cal) / (importance + _EPS)

    flat_importance = importance.reshape(-1)
    floor = float(np.quantile(flat_importance, float(importance_quantile)))
    eligible = flat_importance >= floor
    required = min(max(int(min_eligible), 1), flat_importance.size)
    if int(eligible.sum()) < required:
        kth = np.sort(flat_importance)[-required]
        floor = float(kth)
        eligible = flat_importance >= floor

    L, H = sem.shape
    heads = _head_order(L, H)
    flat_q = preference.reshape(-1)

    def sem_key(idx: int):
        layer, head = heads[idx]
        return (-flat_q[idx], -flat_importance[idx], layer, head)

    def rrwp_key(idx: int):
        layer, head = heads[idx]
        return (flat_q[idx], -flat_importance[idx], layer, head)

    eligible_idx = [idx for idx in range(len(heads)) if bool(eligible[idx])]
    sem_rank = [heads[idx] for idx in sorted(eligible_idx, key=sem_key)]
    rrwp_rank = [heads[idx] for idx in sorted(eligible_idx, key=rrwp_key)]
    return {
        "semantic_scale": sem_scale,
        "rrwp_role_scale": rrwp_scale,
        "semantic_calibrated": sem_cal,
        "rrwp_role_calibrated": rrwp_cal,
        "importance": importance,
        "preference": preference,
        "importance_floor": floor,
        "importance_quantile": float(importance_quantile),
        "eligible": eligible.reshape(L, H),
        "rankings": {
            "semantic": [[int(l), int(h)] for l, h in sem_rank],
            "rrwp_role": [[int(l), int(h)] for l, h in rrwp_rank],
        },
    }


def select_confirmation_graph_ids(
    num_graphs_total: int,
    discovery_graph_ids: Sequence[int],
    n_graphs: int,
    *,
    seed: int,
) -> np.ndarray:
    """Choose a deterministic confirmation sample from the complement of discovery graphs."""

    total = int(num_graphs_total)
    discovery = np.asarray(discovery_graph_ids, dtype=np.int64).reshape(-1)
    if total <= 0:
        raise ValueError("num_graphs_total must be positive")
    if ((discovery < 0) | (discovery >= total)).any():
        raise ValueError("discovery graph id outside the evaluation dataset")
    candidates = np.setdiff1d(np.arange(total, dtype=np.int64), np.unique(discovery))
    take = min(max(int(n_graphs), 0), int(candidates.size))
    if take <= 0:
        raise ValueError("no confirmation graphs remain after excluding discovery graphs")
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(candidates, size=take, replace=False)
    return np.sort(chosen.astype(np.int64))


def _head_flat_index(head: Sequence[int], H: int) -> int:
    return int(head[0]) * int(H) + int(head[1])


def _matched_control_group(
    target_heads: Sequence[tuple[int, int]],
    importance: np.ndarray,
    *,
    rng: np.random.Generator,
    excluded_heads: Sequence[tuple[int, int]] = (),
) -> tuple[list[tuple[int, int]], dict[str, float]]:
    """Greedily match a group on layer and log-importance, with seeded local randomisation.

    Exact-layer candidates are always preferred.  When a target group exhausts a layer, the
    fallback cost first minimizes layer distance and then log-importance distance.  Sampling from
    the three nearest candidates produces a genuine control distribution without abandoning the
    matching objective.
    """

    imp = np.asarray(importance, dtype=np.float64)
    if imp.ndim != 2:
        raise ValueError("importance must have shape [L,H]")
    L, H = imp.shape
    all_heads = _head_order(L, H)
    target = [tuple(map(int, head)) for head in target_heads]
    # Controls must not quietly recycle either prespecified specialised extreme.
    # Otherwise a nominal null for one channel can contain the strongest heads
    # from the opposite channel and attenuate the double dissociation by design.
    forbidden = set(target) | {tuple(map(int, head)) for head in excluded_heads}
    chosen: list[tuple[int, int]] = []
    exact_layers = 0
    log_diffs: list[float] = []

    # Harder-to-match heads first: layers with more target members have fewer remaining controls.
    layer_counts = {layer: sum(int(h[0]) == layer for h in target) for layer in range(L)}
    ordered_targets = sorted(target, key=lambda hd: (-layer_counts[hd[0]], hd[0], hd[1]))
    for target_head in ordered_targets:
        tl, th = target_head
        candidates = [hd for hd in all_heads if hd not in forbidden and hd not in chosen]
        if not candidates:
            raise ValueError("not enough non-target heads to construct a matched control group")
        same_layer = [hd for hd in candidates if hd[0] == tl]
        pool = same_layer if same_layer else candidates
        target_log_i = float(np.log(imp[tl, th] + _EPS))

        def cost(hd: tuple[int, int]):
            layer_gap = abs(int(hd[0]) - tl)
            log_gap = abs(float(np.log(imp[hd] + _EPS)) - target_log_i)
            return (layer_gap, log_gap, hd[0], hd[1])

        pool = sorted(pool, key=cost)
        local = pool[: min(3, len(pool))]
        pick = local[int(rng.integers(len(local)))]
        chosen.append(pick)
        exact_layers += int(pick[0] == tl)
        log_diffs.append(abs(float(np.log(imp[pick] + _EPS)) - target_log_i))

    # Restore target-layer order so group manifests are easy to compare.
    chosen.sort(key=lambda hd: (hd[0], hd[1]))
    quality = {
        "exact_layer_fraction": float(exact_layers / max(len(target), 1)),
        "mean_abs_log_importance_difference": float(np.mean(log_diffs)) if log_diffs else 0.0,
    }
    return chosen, quality


def make_head_group_specs(
    calibration: Mapping[str, Any],
    *,
    topk: Sequence[int],
    matched_control_draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Create cumulative specialised groups and matched controls for both channels."""

    importance = np.asarray(calibration["importance"], dtype=np.float64)
    ks = tuple(sorted({int(k) for k in topk}))
    if not ks or min(ks) <= 0:
        raise ValueError("topk must contain positive integers")
    draws = max(int(matched_control_draws), 0)
    rng = np.random.default_rng(int(seed))
    specs: list[dict[str, Any]] = []
    seen_control_ids: set[tuple[tuple[int, int], ...]] = set()

    # Reserve the semantic extreme before constructing the opposite extreme.  Disjoint sets make
    # the 2x2 mediation matrix identifiable: overlapping diagonal groups would mechanically wash
    # out the double dissociation even when the underlying preference ordering is informative.
    max_k = max(ks)
    semantic_ranking = [
        tuple(map(int, hd)) for hd in calibration["rankings"]["semantic"]
    ]
    semantic_reserved = set(semantic_ranking[:max_k])
    rrwp_ranking = [
        tuple(map(int, hd))
        for hd in calibration["rankings"]["rrwp_role"]
        if tuple(map(int, hd)) not in semantic_reserved
    ]
    rankings = {"semantic": semantic_ranking, "rrwp_role": rrwp_ranking}
    if len(semantic_ranking) < max_k or len(rrwp_ranking) < max_k:
        raise ValueError(
            f"need at least {2 * max_k} usable heads for disjoint semantic/RRWP top-{max_k} groups"
        )
    specialised_extremes = semantic_reserved | set(rrwp_ranking[:max_k])

    for selector in CHANNELS:
        ranking = rankings[selector]
        if max(ks) > len(ranking):
            raise ValueError(
                f"top-k={max(ks)} exceeds {len(ranking)} importance-eligible {selector} heads"
            )
        for k in ks:
            target = ranking[:k]
            if selector == "rrwp_role" and set(target) & semantic_reserved:
                raise RuntimeError("semantic and RRWP-role specialised groups must be disjoint")
            specs.append(
                {
                    "group_id": f"{selector}_top{k}",
                    "group_type": "specialised",
                    "selector": selector,
                    "k": int(k),
                    "control_draw": None,
                    "heads": [[int(l), int(h)] for l, h in target],
                    "match_quality": None,
                }
            )
            for draw in range(draws):
                # Try a few times to avoid duplicate control groups while retaining determinism.
                control = None
                quality = None
                for _attempt in range(24):
                    candidate, q = _matched_control_group(
                        target,
                        importance,
                        rng=rng,
                        excluded_heads=specialised_extremes,
                    )
                    key = tuple(candidate)
                    if key not in seen_control_ids or len(seen_control_ids) >= 10_000:
                        control, quality = candidate, q
                        seen_control_ids.add(key)
                        break
                if control is None:
                    control, quality = _matched_control_group(
                        target,
                        importance,
                        rng=rng,
                        excluded_heads=specialised_extremes,
                    )
                specs.append(
                    {
                        "group_id": f"control_for_{selector}_top{k}_draw{draw}",
                        "group_type": "matched_control",
                        "selector": selector,
                        "k": int(k),
                        "control_draw": int(draw),
                        "heads": [[int(l), int(h)] for l, h in control],
                        "match_quality": quality,
                    }
                )
    return specs


def _build_intervention_bank(
    gm: Any,
    sc: Any,
    graph_ids: Sequence[int],
    *,
    interventions_per_graph: int,
    seed: int,
) -> list[_InterventionEvent]:
    """Construct paired semantic and mask-frozen RRWP-role ZINC counterfactuals."""

    import torch

    from .scores import _build_donor_pool, _perturb_mask_frozen

    repeats = int(interventions_per_graph)
    if repeats <= 0:
        raise ValueError("interventions_per_graph must be positive")
    rng = np.random.default_rng(int(seed))
    donor_rows, donor_gids = _build_donor_pool(gm)
    donor_rows = np.asarray(donor_rows)
    donor_gids = np.asarray(donor_gids, dtype=np.int64)
    events: list[_InterventionEvent] = []

    for graph_id in np.asarray(graph_ids, dtype=np.int64):
        base = gm.eval_ds[int(graph_id)]
        n = int(base.num_nodes)
        if n < 2:
            raise ValueError(
                f"ZINC channel mediation requires at least two nodes; graph {int(graph_id)} has {n}"
            )
        own_rows = np.asarray(gm.adapter.rows(base))
        degrees = structural.node_degrees(base.edge_index, n)
        if getattr(sc, "donor_split", "test") == getattr(sc, "eval_split", "test"):
            external = donor_gids != int(graph_id)
        else:
            external = np.ones(donor_gids.shape[0], dtype=bool)

        for replicate in range(repeats):
            anchor = int(rng.integers(n))
            different = np.any(donor_rows != own_rows[anchor][None, :], axis=1)
            donor_candidates = np.flatnonzero(external & different)
            if donor_candidates.size == 0:
                raise ValueError(
                    f"no external non-noop semantic donor for graph={int(graph_id)}, node={anchor}"
                )
            donor_idx = int(rng.choice(donor_candidates))
            semantic_cf = base.clone()
            gm.adapter.write_donors(semantic_cf.x, [anchor], donor_rows[donor_idx : donor_idx + 1])
            if torch.equal(semantic_cf.x, base.x):
                raise RuntimeError("semantic intervention unexpectedly produced a no-op")
            for structural_name in (
                "edge_index",
                "edge_attr",
                "rrwp",
                "rrwp_index",
                "rrwp_val",
                "deg",
                "log_deg",
            ):
                clean_value = getattr(base, structural_name, None)
                cf_value = getattr(semantic_cf, structural_name, None)
                if clean_value is not None and not torch.equal(clean_value, cf_value):
                    raise RuntimeError(
                        f"semantic intervention changed structural field {structural_name!r}"
                    )

            pair_id = f"g{int(graph_id)}_r{replicate}"
            semantic_meta = {
                "event_id": f"{pair_id}_semantic",
                "pair_id": pair_id,
                "graph_id": int(graph_id),
                "replicate": int(replicate),
                "channel": "semantic",
                "anchor": anchor,
                "partner": None,
                "donor_pool_index": donor_idx,
                "donor_graph_id": int(donor_gids[donor_idx]),
                "donor_content": donor_rows[donor_idx].astype(np.int64).tolist(),
                "num_nodes": n,
            }
            events.append(_InterventionEvent(semantic_meta, base.clone(), semantic_cf))

            partner = int(
                structural.sample_partners(
                    degrees,
                    anchor,
                    1,
                    rng,
                    getattr(sc, "partner_match", "degree"),
                )[0]
            )
            if partner == anchor:
                raise RuntimeError(
                    "RRWP-role confirmation intervention must not be a self-transposition"
                )
            rrwp_cf = _perturb_mask_frozen(base, anchor, partner)
            if not torch.equal(rrwp_cf.x, base.x):
                raise RuntimeError("RRWP-role intervention changed semantic content x")
            if not torch.equal(rrwp_cf.edge_index, base.edge_index):
                raise RuntimeError("RRWP-role intervention changed the frozen attention support")
            if getattr(base, "edge_attr", None) is not None and not torch.equal(
                rrwp_cf.edge_attr, base.edge_attr
            ):
                raise RuntimeError("RRWP-role intervention changed frozen bond attributes")
            rrwp_meta = {
                "event_id": f"{pair_id}_rrwp_role",
                "pair_id": pair_id,
                "graph_id": int(graph_id),
                "replicate": int(replicate),
                "channel": "rrwp_role",
                "anchor": anchor,
                "partner": partner,
                "donor_pool_index": None,
                "donor_graph_id": None,
                "donor_content": None,
                "num_nodes": n,
            }
            events.append(_InterventionEvent(rrwp_meta, base.clone(), rrwp_cf))
    return events


def _fresh_batch(data_list: Sequence[Any], device: Any):
    """Build a new Batch every call; GRIT forward mutates encoded batch fields in place."""

    from torch_geometric.data import Batch

    return Batch.from_data_list(list(data_list)).to(device)


def _capture_predictions_and_wv(gm: Any, data_list: Sequence[Any]):
    """Capture detached transport tensors from a fresh batch."""

    batch = _fresh_batch(data_list, gm.device)
    cap = gm.capture(batch, want_grad=False, want_attn=False)
    pred = cap["pred"].detach().cpu().numpy().reshape(len(data_list), -1)
    wv = [tensor.detach().clone() for tensor in cap["wV"]]
    return pred.astype(np.float64), wv


def _prediction_from_model_output(output: Any):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _predict_with_transport_patch(
    gm: Any,
    data_list: Sequence[Any],
    source_wv: Sequence[Any],
    heads: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Patch selected wV head slices and return predictions from a fresh target batch."""

    import torch

    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        layer = int(layer)
        head = int(head)
        if not 0 <= layer < gm.L or not 0 <= head < gm.H:
            raise IndexError(f"invalid head L{layer}H{head} for geometry {gm.L}x{gm.H}")
        by_layer.setdefault(layer, []).append(head)

    # Construct before registering hooks: if batching fails there is nothing to clean up.
    batch = _fresh_batch(data_list, gm.device)
    handles = []
    for layer, layer_heads in by_layer.items():
        index = torch.as_tensor(sorted(set(layer_heads)), dtype=torch.long, device=gm.device)
        source = source_wv[layer]

        def make_hook(head_index, source_tensor):
            def hook(_module, _inputs, output):
                if isinstance(output, (tuple, list)):
                    h_out, e_out = output
                else:
                    h_out, e_out = output, None
                if tuple(h_out.shape) != tuple(source_tensor.shape):
                    raise RuntimeError(
                        "source/target wV shapes differ during patch: "
                        f"{tuple(source_tensor.shape)} vs {tuple(h_out.shape)}"
                    )
                patched = h_out.clone()
                src = source_tensor.to(device=h_out.device, dtype=h_out.dtype)
                patched[:, head_index, :] = src[:, head_index, :]
                if isinstance(output, tuple):
                    return (patched, e_out)
                if isinstance(output, list):
                    return [patched, e_out]
                return patched

            return hook

        handles.append(gm.attn_layers[layer].register_forward_hook(make_hook(index, source)))

    try:
        with torch.no_grad():
            output = gm.model(batch)
            pred = _prediction_from_model_output(output)
    finally:
        for handle in handles:
            handle.remove()
    return pred.detach().cpu().numpy().reshape(len(data_list), -1).astype(np.float64)


def directional_mediation_components(
    pred_clean: np.ndarray,
    pred_cf: np.ndarray,
    pred_patch: np.ndarray,
    *,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-event numerator, denominator, and mediated displacement."""

    clean = np.asarray(pred_clean, dtype=np.float64)
    cf = np.asarray(pred_cf, dtype=np.float64)
    patch = np.asarray(pred_patch, dtype=np.float64)
    if clean.ndim != 2 or cf.ndim != 2 or clean.shape != cf.shape:
        raise ValueError(
            f"clean/counterfactual predictions must have equal [N,T] shape, got "
            f"{clean.shape} and {cf.shape}"
        )
    if patch.ndim == 2:
        if patch.shape != clean.shape:
            raise ValueError(
                f"2D patched predictions must have shape {clean.shape}, got {patch.shape}"
            )
    elif patch.ndim == 3:
        if patch.shape[0] != clean.shape[0] or patch.shape[2] != clean.shape[1]:
            raise ValueError(
                "3D patched predictions must have shape [N,K,T] matching clean [N,T], "
                f"got patch={patch.shape}, clean={clean.shape}"
            )
    else:
        raise ValueError(f"patched predictions must have shape [N,T] or [N,K,T], got {patch.shape}")
    delta = cf - clean
    if direction == "noising":
        mediated = patch - clean[:, None, :] if patch.ndim == 3 else patch - clean
    elif direction == "denoising":
        mediated = cf[:, None, :] - patch if patch.ndim == 3 else cf - patch
    else:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    denominator = np.sum(delta * delta, axis=-1)
    if mediated.ndim == 3:
        numerator = np.einsum("nkt,nt->nk", mediated, delta)
    else:
        numerator = np.einsum("nt,nt->n", mediated, delta)
    return numerator, denominator, mediated


def ratio_of_sums(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Stable directional mediation: sum numerator divided by sum total-effect energy."""

    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64).reshape(-1)
    if num.shape[0] != den.shape[0]:
        raise ValueError("numerator and denominator must share their event dimension")
    total_den = float(den.sum())
    out_shape = num.shape[1:]
    if total_den <= _EPS:
        return np.full(out_shape or (), np.nan, dtype=np.float64)
    return np.asarray(num.sum(axis=0) / total_den, dtype=np.float64)


def _loss_numpy(pred: np.ndarray, targets: np.ndarray, loss_fun: str) -> np.ndarray:
    import torch

    p = torch.as_tensor(np.asarray(pred), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(targets), dtype=torch.float64)
    return metrics.per_graph_loss(p, y, loss_fun).detach().cpu().numpy().astype(np.float64)


def _loss_effect_matrix(
    pred_clean: np.ndarray,
    pred_cf: np.ndarray,
    pred_patch: np.ndarray,
    targets: np.ndarray,
    loss_fun: str,
    *,
    direction: str,
) -> np.ndarray:
    patch = np.asarray(pred_patch, dtype=np.float64)
    if patch.ndim == 2:
        patch = patch[:, None, :]
    n, k, t = patch.shape
    tiled_y = np.repeat(np.asarray(targets)[:, None, :], k, axis=1).reshape(n * k, t)
    loss_patch = _loss_numpy(patch.reshape(n * k, t), tiled_y, loss_fun).reshape(n, k)
    if direction == "noising":
        base = _loss_numpy(pred_clean, targets, loss_fun)[:, None]
        return loss_patch - base
    if direction == "denoising":
        base = _loss_numpy(pred_cf, targets, loss_fun)[:, None]
        return base - loss_patch
    raise ValueError(direction)


def _cluster_weights(num_clusters: int, replicates: int, *, seed: int) -> np.ndarray:
    """Multinomial cluster counts; one shared matrix keeps all contrasts paired."""

    g = int(num_clusters)
    b = max(int(replicates), 0)
    if g <= 0:
        raise ValueError("at least one graph cluster is required")
    if b == 0:
        return np.empty((0, g), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    samples = rng.integers(0, g, size=(b, g))
    weights = np.zeros((b, g), dtype=np.float64)
    for row in range(b):
        weights[row] = np.bincount(samples[row], minlength=g)
    return weights


def _sum_by_graph(
    values: np.ndarray,
    graph_ids: np.ndarray,
    unique_graphs: np.ndarray,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    one_dimensional = vals.ndim == 1
    if one_dimensional:
        vals = vals[:, None]
    out = np.zeros((len(unique_graphs), vals.shape[1]), dtype=np.float64)
    position = {int(graph_id): idx for idx, graph_id in enumerate(unique_graphs)}
    rows = np.array([position[int(graph_id)] for graph_id in graph_ids], dtype=np.int64)
    np.add.at(out, rows, vals)
    return out[:, 0] if one_dimensional else out


def _nan_quantiles(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(draws, dtype=np.float64)
    if arr.shape[0] == 0:
        shape = arr.shape[1:]
        return np.full(shape, np.nan), np.full(shape, np.nan)
    return np.nanquantile(arr, 0.025, axis=0), np.nanquantile(arr, 0.975, axis=0)


def _summarise_matrix(
    numerator: np.ndarray,
    denominator: np.ndarray,
    loss_effect: np.ndarray,
    graph_ids: np.ndarray,
    unique_graphs: np.ndarray,
    bootstrap_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Summarise K columns jointly so every bootstrap draw is paired."""

    num = np.asarray(numerator, dtype=np.float64)
    if num.ndim == 1:
        num = num[:, None]
    loss = np.asarray(loss_effect, dtype=np.float64)
    if loss.ndim == 1:
        loss = loss[:, None]
    den = np.asarray(denominator, dtype=np.float64).reshape(-1)
    point_beta = np.asarray(ratio_of_sums(num, den)).reshape(-1)
    point_loss = loss.mean(axis=0)

    graph_num = _sum_by_graph(num, graph_ids, unique_graphs)
    graph_den = _sum_by_graph(den, graph_ids, unique_graphs)
    graph_loss = _sum_by_graph(loss, graph_ids, unique_graphs)
    graph_count = _sum_by_graph(np.ones(len(graph_ids)), graph_ids, unique_graphs)
    if bootstrap_weights.shape[0]:
        draw_num = bootstrap_weights @ graph_num
        draw_den = bootstrap_weights @ graph_den
        beta_draws = draw_num / np.maximum(draw_den[:, None], _EPS)
        beta_draws[draw_den <= _EPS] = np.nan
        loss_draws = (bootstrap_weights @ graph_loss) / np.maximum(
            (bootstrap_weights @ graph_count)[:, None], 1.0
        )
    else:
        beta_draws = np.empty((0, num.shape[1]))
        loss_draws = np.empty((0, num.shape[1]))
    beta_low, beta_high = _nan_quantiles(beta_draws)
    loss_low, loss_high = _nan_quantiles(loss_draws)
    return {
        "beta": point_beta,
        "beta_ci_low": beta_low,
        "beta_ci_high": beta_high,
        "loss_effect_mean": point_loss,
        "loss_effect_ci_low": loss_low,
        "loss_effect_ci_high": loss_high,
        "beta_draws": beta_draws,
        "loss_draws": loss_draws,
        "total_effect_energy": np.full(num.shape[1], float(den.sum())),
    }


def _targets_from_events(events: Sequence[_InterventionEvent]) -> np.ndarray:
    rows = []
    for event in events:
        y = (
            event.clean.y.detach().cpu().numpy()
            if hasattr(event.clean.y, "detach")
            else event.clean.y
        )
        rows.append(np.asarray(y, dtype=np.float64).reshape(-1))
    return np.stack(rows)


def _ols_preference_slope(
    preference: np.ndarray,
    importance: np.ndarray,
    layers: np.ndarray,
    response: np.ndarray,
    eligible: np.ndarray,
) -> float:
    keep = np.asarray(eligible, dtype=bool) & np.isfinite(response)
    if int(keep.sum()) < 4:
        return float("nan")
    q = np.asarray(preference, dtype=float)[keep]
    log_i = np.log(np.asarray(importance, dtype=float)[keep] + _EPS)
    layer = np.asarray(layers, dtype=int)[keep]
    columns = [np.ones(len(q)), q, log_i - log_i.mean()]
    for value in sorted(np.unique(layer))[1:]:
        columns.append((layer == value).astype(float))
    design = np.column_stack(columns)
    coef = np.linalg.lstsq(design, np.asarray(response, dtype=float)[keep], rcond=None)[0]
    return float(coef[1])


def _within_layer_preference_permutation_p(
    preference: np.ndarray,
    importance: np.ndarray,
    layers: np.ndarray,
    response: np.ndarray,
    eligible: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> tuple[float, int]:
    """Two-sided within-layer randomisation test for the preference-slope statistic."""

    observed = _ols_preference_slope(preference, importance, layers, response, eligible)
    if not np.isfinite(observed) or int(permutations) <= 0:
        return float("nan"), 0
    q = np.asarray(preference, dtype=np.float64)
    layer = np.asarray(layers, dtype=np.int64)
    keep = np.asarray(eligible, dtype=bool) & np.isfinite(response)
    rng = np.random.default_rng(int(seed))
    null = []
    for _ in range(int(permutations)):
        permuted = q.copy()
        for layer_value in np.unique(layer[keep]):
            idx = np.flatnonzero(keep & (layer == int(layer_value)))
            if idx.size > 1:
                permuted[idx] = q[rng.permutation(idx)]
        value = _ols_preference_slope(permuted, importance, layer, response, keep)
        if np.isfinite(value):
            null.append(float(value))
    if not null:
        return float("nan"), 0
    exceed = sum(abs(value) >= abs(observed) for value in null)
    return float((exceed + 1) / (len(null) + 1)), int(len(null))


def run_channel_causal_mediation(
    result: Mapping[str, Any],
    sc: Any,
    *,
    n_graphs: int = 64,
    interventions_per_graph: int = 2,
    topk: Sequence[int] = (1, 2, 4, 8),
    primary_k: int = 4,
    matched_control_draws: int = 8,
    bootstrap_replicates: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run held-out semantic/RRWP-role head mediation on a loaded ZINC GRIT model.

    ``result`` must be the live result returned by ``scores.score_model`` (including ``gm``), and
    ``result['graph_ids']`` is treated as the immutable discovery set.  Confirmation molecules are
    selected only from its complement.
    """

    gm = result.get("gm")
    if gm is None:
        raise ValueError("run_channel_causal_mediation requires live result['gm']")
    if int(primary_k) not in {int(k) for k in topk}:
        raise ValueError("primary_k must be one of topk")
    L, H = int(gm.L), int(gm.H)
    head_order = _head_order(L, H)
    P = len(head_order)
    discovery_ids = np.asarray(result["graph_ids"], dtype=np.int64)
    confirmation_ids = select_confirmation_graph_ids(
        len(gm.eval_ds), discovery_ids, n_graphs, seed=int(seed) + 101
    )
    if np.intersect1d(discovery_ids, confirmation_ids).size:
        raise RuntimeError("discovery and confirmation graph sets overlap")

    calibration = calibrate_head_scores(
        np.asarray(result["S_sem"]),
        np.asarray(result["S_str"]),
        importance_quantile=0.5,
        min_eligible=2 * max(int(k) for k in topk),
    )
    group_specs = make_head_group_specs(
        calibration,
        topk=topk,
        matched_control_draws=matched_control_draws,
        seed=int(seed) + 211,
    )
    events = _build_intervention_bank(
        gm,
        sc,
        confirmation_ids,
        interventions_per_graph=interventions_per_graph,
        seed=int(seed) + 307,
    )
    if not events:
        raise RuntimeError("frozen confirmation bank is empty")

    N = len(events)
    Q = len(group_specs)
    event_graph_ids = np.array([row.metadata["graph_id"] for row in events], dtype=np.int64)
    event_channels = np.array([row.metadata["channel"] for row in events], dtype=object)
    targets = _targets_from_events(events)
    T = int(targets.shape[1])
    pred_clean = np.full((N, T), np.nan, dtype=np.float64)
    pred_cf = np.full((N, T), np.nan, dtype=np.float64)
    single_patch = {
        direction: np.full((N, P, T), np.nan, dtype=np.float64) for direction in DIRECTIONS
    }
    group_patch = {
        direction: np.full((N, Q, T), np.nan, dtype=np.float64) for direction in DIRECTIONS
    }
    noop_clean_max = 0.0
    noop_cf_max = 0.0
    all_heads = head_order

    log(
        f"[causal-mediation] confirmation graphs={len(confirmation_ids)}, events={N} "
        f"({interventions_per_graph}/graph/channel), heads={P}, groups={Q}"
    )
    for start in range(0, N, _DEFAULT_BATCH_SIZE):
        stop = min(start + _DEFAULT_BATCH_SIZE, N)
        chunk = events[start:stop]
        clean_data = [row.clean for row in chunk]
        cf_data = [row.counterfactual for row in chunk]
        clean_pred_chunk, clean_wv = _capture_predictions_and_wv(gm, clean_data)
        cf_pred_chunk, cf_wv = _capture_predictions_and_wv(gm, cf_data)
        if clean_pred_chunk.shape[1] != T or cf_pred_chunk.shape[1] != T:
            raise RuntimeError("model output width does not match event target width")
        pred_clean[start:stop] = clean_pred_chunk
        pred_cf[start:stop] = cf_pred_chunk

        # Explicit same-source clamps validate the hook and fresh-batch discipline.
        clean_noop = _predict_with_transport_patch(gm, clean_data, clean_wv, all_heads)
        cf_noop = _predict_with_transport_patch(gm, cf_data, cf_wv, all_heads)
        noop_clean_max = max(noop_clean_max, float(np.max(np.abs(clean_noop - clean_pred_chunk))))
        noop_cf_max = max(noop_cf_max, float(np.max(np.abs(cf_noop - cf_pred_chunk))))

        for head_idx, head in enumerate(head_order):
            single_patch["noising"][start:stop, head_idx] = _predict_with_transport_patch(
                gm, clean_data, cf_wv, [head]
            )
            single_patch["denoising"][start:stop, head_idx] = _predict_with_transport_patch(
                gm, cf_data, clean_wv, [head]
            )
        for group_idx, spec in enumerate(group_specs):
            heads = [tuple(map(int, head)) for head in spec["heads"]]
            group_patch["noising"][start:stop, group_idx] = _predict_with_transport_patch(
                gm, clean_data, cf_wv, heads
            )
            group_patch["denoising"][start:stop, group_idx] = _predict_with_transport_patch(
                gm, cf_data, clean_wv, heads
            )
        log(f"[causal-mediation] events {stop}/{N}")

    if not np.isfinite(pred_clean).all() or not np.isfinite(pred_cf).all():
        raise RuntimeError("non-finite clean/counterfactual predictions")
    for collection in (single_patch, group_patch):
        if any(not np.isfinite(values).all() for values in collection.values()):
            raise RuntimeError("non-finite patched predictions")
    noop_tol = max(float(getattr(sc, "tol", 1.0e-4)), float(getattr(sc, "float_noise_tol", 5.0e-3)))
    if max(noop_clean_max, noop_cf_max) > noop_tol:
        raise RuntimeError(
            f"same-source all-head patch drift {max(noop_clean_max, noop_cf_max):.3e} "
            f"exceeds tolerance {noop_tol:.3e}"
        )

    unique_graphs = np.unique(event_graph_ids)
    bootstrap_weights = _cluster_weights(
        len(unique_graphs), int(bootstrap_replicates), seed=int(seed) + 401
    )
    single_components: dict[tuple[str, str], dict[str, Any]] = {}
    group_components: dict[tuple[str, str], dict[str, Any]] = {}
    single_summary: list[dict[str, Any]] = []
    group_summary: list[dict[str, Any]] = []
    single_loss_effects: dict[str, np.ndarray] = {}
    group_loss_effects: dict[str, np.ndarray] = {}

    for direction in DIRECTIONS:
        snum, denominator, _smed = directional_mediation_components(
            pred_clean, pred_cf, single_patch[direction], direction=direction
        )
        gnum, _gden, _gmed = directional_mediation_components(
            pred_clean, pred_cf, group_patch[direction], direction=direction
        )
        sloss = _loss_effect_matrix(
            pred_clean, pred_cf, single_patch[direction], targets, gm.loss_fun, direction=direction
        )
        gloss = _loss_effect_matrix(
            pred_clean, pred_cf, group_patch[direction], targets, gm.loss_fun, direction=direction
        )
        single_loss_effects[direction] = sloss
        group_loss_effects[direction] = gloss

        for channel in CHANNELS:
            mask = event_channels == channel
            sg = event_graph_ids[mask]
            single_stats = _summarise_matrix(
                snum[mask], denominator[mask], sloss[mask], sg, unique_graphs, bootstrap_weights
            )
            group_stats = _summarise_matrix(
                gnum[mask], denominator[mask], gloss[mask], sg, unique_graphs, bootstrap_weights
            )
            single_components[(direction, channel)] = {
                "numerator": snum,
                "denominator": denominator,
                "mediated": _smed,
                "stats": single_stats,
                "mask": mask,
            }
            group_components[(direction, channel)] = {
                "numerator": gnum,
                "denominator": denominator,
                "mediated": _gmed,
                "stats": group_stats,
                "mask": mask,
            }
            for head_idx, (layer, head) in enumerate(head_order):
                single_summary.append(
                    {
                        "layer": int(layer),
                        "head": int(head),
                        "channel": channel,
                        "direction": direction,
                        "beta": float(single_stats["beta"][head_idx]),
                        "beta_ci_low": float(single_stats["beta_ci_low"][head_idx]),
                        "beta_ci_high": float(single_stats["beta_ci_high"][head_idx]),
                        "loss_effect_mean": float(single_stats["loss_effect_mean"][head_idx]),
                        "loss_effect_ci_low": float(single_stats["loss_effect_ci_low"][head_idx]),
                        "loss_effect_ci_high": float(single_stats["loss_effect_ci_high"][head_idx]),
                        "total_effect_energy": float(single_stats["total_effect_energy"][head_idx]),
                        "n_events": int(mask.sum()),
                        "n_graphs": int(len(np.unique(sg))),
                    }
                )
            for group_idx, spec in enumerate(group_specs):
                group_summary.append(
                    {
                        "group_id": spec["group_id"],
                        "group_type": spec["group_type"],
                        "selector": spec["selector"],
                        "k": int(spec["k"]),
                        "control_draw": spec["control_draw"],
                        "channel": channel,
                        "direction": direction,
                        "beta": float(group_stats["beta"][group_idx]),
                        "beta_ci_low": float(group_stats["beta_ci_low"][group_idx]),
                        "beta_ci_high": float(group_stats["beta_ci_high"][group_idx]),
                        "loss_effect_mean": float(group_stats["loss_effect_mean"][group_idx]),
                        "loss_effect_ci_low": float(group_stats["loss_effect_ci_low"][group_idx]),
                        "loss_effect_ci_high": float(group_stats["loss_effect_ci_high"][group_idx]),
                        "total_effect_energy": float(group_stats["total_effect_energy"][group_idx]),
                        "n_events": int(mask.sum()),
                        "n_graphs": int(len(np.unique(sg))),
                    }
                )

    group_index = {spec["group_id"]: idx for idx, spec in enumerate(group_specs)}
    topk_rows: list[dict[str, Any]] = []
    for k in sorted({int(value) for value in topk}):
        sem_idx = group_index[f"semantic_top{k}"]
        rrwp_idx = group_index[f"rrwp_role_top{k}"]
        for direction in DIRECTIONS:
            sem_sem = group_components[(direction, "semantic")]["stats"]
            rrwp_channel = group_components[(direction, "rrwp_role")]["stats"]
            a = float(sem_sem["beta"][sem_idx])
            b = float(rrwp_channel["beta"][sem_idx])
            c = float(sem_sem["beta"][rrwp_idx])
            d = float(rrwp_channel["beta"][rrwp_idx])
            dd = a - b - c + d
            if bootstrap_weights.shape[0]:
                dd_draws = (
                    sem_sem["beta_draws"][:, sem_idx]
                    - rrwp_channel["beta_draws"][:, sem_idx]
                    - sem_sem["beta_draws"][:, rrwp_idx]
                    + rrwp_channel["beta_draws"][:, rrwp_idx]
                )
                dd_low, dd_high = np.nanquantile(dd_draws, [0.025, 0.975])
            else:
                dd_draws = np.empty(0)
                dd_low = dd_high = float("nan")

            control_values = []
            control_draw_matrix = []
            for draw in range(max(int(matched_control_draws), 0)):
                sem_control = group_index[f"control_for_semantic_top{k}_draw{draw}"]
                rrwp_control = group_index[f"control_for_rrwp_role_top{k}_draw{draw}"]
                control_values.append(
                    float(sem_sem["beta"][sem_control])
                    - float(rrwp_channel["beta"][sem_control])
                    - float(sem_sem["beta"][rrwp_control])
                    + float(rrwp_channel["beta"][rrwp_control])
                )
                if bootstrap_weights.shape[0]:
                    control_draw_matrix.append(
                        sem_sem["beta_draws"][:, sem_control]
                        - rrwp_channel["beta_draws"][:, sem_control]
                        - sem_sem["beta_draws"][:, rrwp_control]
                        + rrwp_channel["beta_draws"][:, rrwp_control]
                    )
            if control_values:
                control_mean = float(np.mean(control_values))
                if control_draw_matrix:
                    paired_control_draws = np.mean(np.column_stack(control_draw_matrix), axis=1)
                    control_low, control_high = np.nanquantile(
                        paired_control_draws, [0.025, 0.975]
                    )
                else:
                    control_low, control_high = np.nanquantile(control_values, [0.025, 0.975])
            else:
                control_mean = control_low = control_high = float("nan")
            topk_rows.append(
                {
                    "k": int(k),
                    "direction": direction,
                    "semantic_on_semantic": a,
                    "semantic_on_rrwp_role": b,
                    "rrwp_role_on_semantic": c,
                    "rrwp_role_on_rrwp_role": d,
                    "double_dissociation": float(dd),
                    "ci_low": float(dd_low),
                    "ci_high": float(dd_high),
                    "matched_control_mean": control_mean,
                    "matched_control_ci_low": float(control_low),
                    "matched_control_ci_high": float(control_high),
                }
            )

    interaction_rows: list[dict[str, Any]] = []
    for spec in group_specs:
        if spec["group_type"] != "specialised":
            continue
        group_idx = group_index[spec["group_id"]]
        selected_idx = np.array([_head_flat_index(hd, H) for hd in spec["heads"]], dtype=np.int64)
        for direction in DIRECTIONS:
            single_num_all = single_components[(direction, "semantic")]["numerator"]
            group_num_all = group_components[(direction, "semantic")]["numerator"]
            denominator_all = group_components[(direction, "semantic")]["denominator"]
            for channel in CHANNELS:
                component = group_components[(direction, channel)]
                mask = component["mask"]
                # Numerators are linear in mediated prediction displacement.  Subtracting them
                # therefore measures the exact group-minus-summed-singles interaction projected
                # onto the total counterfactual response.
                group_num = group_num_all[mask, group_idx]
                summed_num = single_num_all[mask][:, selected_idx].sum(axis=1)
                interaction_num = group_num - summed_num
                group_loss = group_loss_effects[direction][mask, group_idx]
                summed_loss = single_loss_effects[direction][mask][:, selected_idx].sum(axis=1)
                interaction_loss = group_loss - summed_loss
                stacked = np.column_stack([group_num, summed_num, interaction_num])
                stacked_loss = np.column_stack([group_loss, summed_loss, interaction_loss])
                summary = _summarise_matrix(
                    stacked,
                    denominator_all[mask],
                    stacked_loss,
                    event_graph_ids[mask],
                    unique_graphs,
                    bootstrap_weights,
                )
                interaction_rows.append(
                    {
                        "group_id": spec["group_id"],
                        "selector": spec["selector"],
                        "k": int(spec["k"]),
                        "channel": channel,
                        "direction": direction,
                        "group_beta": float(summary["beta"][0]),
                        "summed_single_beta": float(summary["beta"][1]),
                        "interaction_beta": float(summary["beta"][2]),
                        "interaction_ci_low": float(summary["beta_ci_low"][2]),
                        "interaction_ci_high": float(summary["beta_ci_high"][2]),
                        "group_loss_effect_mean": float(summary["loss_effect_mean"][0]),
                        "summed_single_loss_effect_mean": float(summary["loss_effect_mean"][1]),
                        "interaction_loss_effect_mean": float(summary["loss_effect_mean"][2]),
                    }
                )

    # Continuous held-out check: does q predict semantic-minus-RRWP mediation after controlling
    # for importance and layer?  It complements, rather than replaces, the prespecified top-k test.
    continuous_rows: list[dict[str, Any]] = []
    q_flat = np.asarray(calibration["preference"]).reshape(-1)
    i_flat = np.asarray(calibration["importance"]).reshape(-1)
    eligible_flat = np.asarray(calibration["eligible"]).reshape(-1)
    layer_flat = np.repeat(np.arange(L), H)
    for direction in DIRECTIONS:
        sem_stats = single_components[(direction, "semantic")]["stats"]
        rrwp_stats = single_components[(direction, "rrwp_role")]["stats"]
        contrast = sem_stats["beta"] - rrwp_stats["beta"]
        slope = _ols_preference_slope(q_flat, i_flat, layer_flat, contrast, eligible_flat)
        slopes = []
        for draw in range(bootstrap_weights.shape[0]):
            draw_contrast = (
                sem_stats["beta_draws"][draw] - rrwp_stats["beta_draws"][draw]
            )
            slopes.append(
                _ols_preference_slope(q_flat, i_flat, layer_flat, draw_contrast, eligible_flat)
            )
        if slopes:
            low, high = np.nanquantile(slopes, [0.025, 0.975])
        else:
            low = high = float("nan")
        permutation_p, permutations_effective = _within_layer_preference_permutation_p(
            q_flat,
            i_flat,
            layer_flat,
            contrast,
            eligible_flat,
            permutations=min(max(int(bootstrap_replicates), 0), 1000),
            seed=int(seed) + 503 + DIRECTIONS.index(direction),
        )
        continuous_rows.append(
            {
                "direction": direction,
                "preference_slope": float(slope),
                "ci_low": float(low),
                "ci_high": float(high),
                "permutation_p_two_sided": float(permutation_p),
                "permutations": int(permutations_effective),
                "n_heads": int(eligible_flat.sum()),
                "controls": ["log_importance", "layer_fixed_effects"],
            }
        )

    primary_rows = [row for row in topk_rows if int(row["k"]) == int(primary_k)]
    total_delta = pred_cf - pred_clean
    channel_energy = {
        channel: float(np.sum(total_delta[event_channels == channel] ** 2)) for channel in CHANNELS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "task": str(result.get("task", "zinc")),
        "title": str(result.get("title", "")),
        "config": {
            "n_graphs_requested": int(n_graphs),
            "n_graphs_effective": int(len(confirmation_ids)),
            "interventions_per_graph": int(interventions_per_graph),
            "topk": [int(k) for k in sorted({int(value) for value in topk})],
            "primary_k": int(primary_k),
            "matched_control_draws": int(matched_control_draws),
            "bootstrap_replicates": int(bootstrap_replicates),
            "seed": int(seed),
            "batch_size": _DEFAULT_BATCH_SIZE,
        },
        "channels": list(CHANNELS),
        "directions": list(DIRECTIONS),
        "discovery_graph_ids": discovery_ids.copy(),
        "confirmation_graph_ids": confirmation_ids.copy(),
        "head_order": np.asarray(head_order, dtype=np.int64),
        "calibration": calibration,
        "intervention_bank": [dict(event.metadata) for event in events],
        "predictions": {
            "clean": pred_clean,
            "counterfactual": pred_cf,
            "targets": targets,
        },
        "single_head": {
            "patched_predictions": single_patch,
            "loss_effects": single_loss_effects,
            "summary": single_summary,
        },
        "groups": {
            "specs": group_specs,
            "patched_predictions": group_patch,
            "loss_effects": group_loss_effects,
            "summary": group_summary,
        },
        "contrasts": {
            "topk": topk_rows,
            "primary": primary_rows,
            "interactions": interaction_rows,
            "continuous": continuous_rows,
        },
        "checks": {
            "discovery_confirmation_overlap": 0,
            "same_source_clean_patch_max_abs": float(noop_clean_max),
            "same_source_counterfactual_patch_max_abs": float(noop_cf_max),
            "noop_tolerance": float(noop_tol),
            "semantic_total_effect_energy": channel_energy["semantic"],
            "rrwp_role_total_effect_energy": channel_energy["rrwp_role"],
            "all_finite": True,
        },
    }


__all__ = [
    "CHANNELS",
    "DIRECTIONS",
    "SCHEMA_VERSION",
    "calibrate_head_scores",
    "directional_mediation_components",
    "make_head_group_specs",
    "ratio_of_sums",
    "run_channel_causal_mediation",
    "select_confirmation_graph_ids",
]
