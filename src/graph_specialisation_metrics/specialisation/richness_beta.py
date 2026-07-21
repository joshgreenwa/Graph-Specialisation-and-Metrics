"""BETA: causal validation and compensation analysis of ZINC head specialisation.

This standalone experiment retains the established semantic donor swap and mask-frozen
structural transposition.  It deliberately removes the earlier effective-response-rank beta.
The score is tested by pre-head ablation, then used to test a specific global-vs-local RRWP
account: high-order coordinate novelty absent from the local model induces compensatory semantic
specialisation, and that compensation accompanies the paired local-model error penalty.  This is
a graphwise pathway analysis, not a claim of causal mediation; causal identification still
requires matched training-time RRWP-horizon interventions.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..carriage import env, metrics
from ..carriage.env import log
from ..carriage.grit_runner import _spd
from ..carriage.tasks import GritTaskSpec, get_task, resolve_dataset_dir
from .ablation import _build_groups
from .model import SpecConfig
from .scores import score_model


BETA_TASKS = ("zinc", "zinc_1hop", "zinc_1hop_local")
DEFAULT_OUT_DIR = "/content/drive/MyDrive/graph_specialisation_metrics/beta_zinc_causal_specialisation"
BIN_LABELS = ("0", "1", "2", "3", "4-7", "8+")
CHANNELS = ("semantic", "structural")


def _stable_rank(x: np.ndarray) -> float:
    """Participation-ratio rank of the role-by-coordinate matrix."""
    x = np.asarray(x, float)
    if x.ndim != 2 or min(x.shape) == 0:
        return 0.0
    singular = np.linalg.svd(x, compute_uv=False)
    energy = np.square(singular)
    denom = float(np.square(energy).sum())
    return float(energy.sum() ** 2 / denom) if denom > 0 else 0.0


def _conditional_coordinate_novelty(local: np.ndarray, high: np.ndarray) -> dict:
    """High-order variation left within roles sharing the same local ``(I,P)`` code."""
    local, high = np.asarray(local, float), np.asarray(high, float)
    if local.ndim != 2 or high.ndim != 2 or len(local) != len(high) or high.shape[1] == 0:
        return {"rms": 0.0, "stable_rank": 0.0, "roles": int(len(local))}
    keys = np.round(local, decimals=7)
    residual = np.zeros_like(high)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    for group in np.unique(inverse):
        selected = inverse == group
        residual[selected] = high[selected] - high[selected].mean(axis=0, keepdims=True)
    return {
        "rms": float(np.sqrt(np.mean(np.square(residual)))) if residual.size else 0.0,
        "stable_rank": _stable_rank(residual),
        "roles": int(len(local)),
    }


def rrwp_coordinate_novelty(base) -> dict:
    """Novel high-order RRWP coordinates beyond local ``(I,P)``, split node/pair.

    Node roles use the diagonal RRWP code. Pair roles use the directed molecular edges consumed
    by the 1-hop model.  Novelty is the residual high-order variation within roles with identical
    local codes, so it measures structural distinctions unavailable to the local-only substrate.
    """
    n = int(base.num_nodes)
    node = getattr(base, "rrwp", None)
    pair_index = getattr(base, "rrwp_index", None)
    pair_value = getattr(base, "rrwp_val", None)
    if pair_index is None or pair_value is None:
        raise RuntimeError("RRWP novelty audit requires rrwp_index and rrwp_val")
    pair_value = pair_value.detach().cpu().numpy()
    pair_index = pair_index.detach().cpu().numpy().astype(np.int64)
    k = int(pair_value.shape[-1])
    dense = np.zeros((n, n, k), dtype=np.float64)
    dense[pair_index[0], pair_index[1]] = pair_value
    if node is None:
        idx = np.arange(n)
        node_code = dense[idx, idx]
    else:
        node_code = node.detach().cpu().numpy().reshape(n, -1)
    support = base.edge_index.detach().cpu().numpy().astype(np.int64)
    pair_code = dense[support[0], support[1]]
    keep = min(2, k)
    node_stats = _conditional_coordinate_novelty(node_code[:, :keep], node_code[:, keep:])
    pair_stats = _conditional_coordinate_novelty(pair_code[:, :keep], pair_code[:, keep:])
    return {f"node_{key}": value for key, value in node_stats.items()} | {
        f"pair_{key}": value for key, value in pair_stats.items()
    }


def _distance_bin_masks(distances: np.ndarray) -> list[np.ndarray]:
    d = np.asarray(distances, dtype=float)
    return [d == 0, d == 1, d == 2, d == 3, (d >= 4) & (d <= 7),
            (d >= 8) & np.isfinite(d)]


@dataclass
class ResponseCollector:
    """Retain graphwise score contributions and clean head throughput from the scorer."""

    task: str
    records: dict[str, list[dict]] = field(
        default_factory=lambda: {"semantic": [], "structural": []}
    )
    abs_error: dict[int, float] = field(default_factory=dict)
    throughput: dict[int, np.ndarray] = field(default_factory=dict)
    rrwp_novelty: dict[int, dict] = field(default_factory=dict)
    _spd_cache: dict[int, np.ndarray] = field(default_factory=dict)

    def __call__(
        self, *, channel, graph_id, base, source_nodes, phi_stack,
        donor_averaged_delta, clean_prediction, clean_head_output=None,
    ) -> None:
        import torch

        source_nodes = np.asarray(source_nodes, dtype=np.int64)
        n = int(base.num_nodes)
        if int(graph_id) not in self.rrwp_novelty:
            self.rrwp_novelty[int(graph_id)] = rrwp_coordinate_novelty(base)
        if graph_id not in self._spd_cache:
            self._spd_cache[graph_id] = _spd(base, n)
        distances = self._spd_cache[graph_id][:, source_nodes].T
        masks = [torch.as_tensor(m, device=phi_stack[0].device,
                                 dtype=phi_stack[0].dtype)
                 for m in _distance_bin_masks(distances)]
        S, L, H = len(source_nodes), len(phi_stack), int(phi_stack[0].shape[2])
        response = torch.zeros(S, len(BIN_LABELS), L, H,
                               device=phi_stack[0].device, dtype=phi_stack[0].dtype)
        for layer, (phi, delta) in enumerate(zip(phi_stack, donor_averaged_delta)):
            projected = torch.einsum("tnhd,snhd->tsnh", phi, delta)
            functional = projected.square().sum(dim=0).sqrt()
            for b, mask in enumerate(masks):
                response[:, b, layer, :] = torch.einsum("snh,sn->sh", functional, mask)
        self.records[channel].append({
            "graph_id": int(graph_id),
            "source_nodes": source_nodes.copy(),
            "response": response.detach().cpu().numpy().reshape(
                S, len(BIN_LABELS), L * H
            ).astype(np.float32),
        })
        if channel == "semantic":
            pred = clean_prediction.detach().cpu().reshape(-1).numpy().astype(float)
            target = base.y.detach().cpu().reshape(-1).numpy().astype(float)
            self.abs_error[int(graph_id)] = float(np.mean(np.abs(pred - target)))
            if clean_head_output is not None:
                self.throughput[int(graph_id)] = np.stack([
                    w.detach().norm(dim=-1).mean(dim=0).cpu().numpy()
                    for w in clean_head_output
                ]).astype(np.float32)

    def score_reconstruction(self, channel: str) -> np.ndarray:
        records = self.records[channel]
        total = sum(r["response"].sum(axis=(0, 1)) for r in records)
        count = sum(int(r["response"].shape[0]) for r in records)
        return total / max(count, 1)

    def graph_score(self, channel: str, graph_id: int, bins=None) -> np.ndarray:
        record = next(r for r in self.records[channel] if int(r["graph_id"]) == int(graph_id))
        z = np.asarray(record["response"], float)
        if bins is not None:
            z = z[:, np.asarray(bins, dtype=int), :]
        return z.sum(axis=(0, 1)) / max(int(z.shape[0]), 1)


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def _rank_columns(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    x = np.asarray(x, float)
    if x.ndim == 1:
        return rankdata(x)
    return np.column_stack([rankdata(x[:, j]) for j in range(x.shape[1])])


def _residualise(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    z = np.asarray(controls, float)
    if z.ndim == 1:
        z = z[:, None]
    design = np.column_stack([np.ones(len(y)), z])
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]


def partial_spearman(x, y, controls) -> float:
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(controls, float)
    if z.ndim == 1:
        z = z[:, None]
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z).all(axis=1)
    if ok.sum() < z.shape[1] + 4:
        return float("nan")
    rx, ry, rz = _rank_columns(x[ok]), _rank_columns(y[ok]), _rank_columns(z[ok])
    xres, yres = _residualise(rx, rz), _residualise(ry, rz)
    if np.std(xres) < 1e-12 or np.std(yres) < 1e-12:
        return float("nan")
    return float(np.corrcoef(xres, yres)[0, 1])


def _bootstrap_stat(n: int, fn, *, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray([fn(rng.integers(0, n, n)) for _ in range(int(n_boot))], float)
    draws = draws[np.isfinite(draws)]
    return ((float(np.quantile(draws, .025)), float(np.quantile(draws, .975)))
            if draws.size else (float("nan"), float("nan")))


def run_head_ablation(gm, graph_ids, *, batch_size: int = 64) -> dict:
    """Exact all-head pre-head-output ablation on the graphs used to estimate the score."""
    import torch

    graph_ids = np.asarray(graph_ids, dtype=np.int64)
    groups, ys, feats, order = _build_groups(gm, graph_ids, batch_size)
    G = len(order)
    clean = gm.collect_preds_ablated(groups, None).reshape(G, -1)
    clean_loss = metrics.per_graph_loss(torch.as_tensor(clean), torch.as_tensor(ys),
                                        gm.loss_fun).cpu().numpy()
    func = np.zeros((gm.L, gm.H, G), dtype=np.float32)
    loss = np.zeros_like(func)
    for layer in range(gm.L):
        for head in range(gm.H):
            pred = gm.collect_preds_ablated(groups, [(layer, head)]).reshape(G, -1)
            func[layer, head] = np.linalg.norm(pred - clean, axis=1)
            pl = metrics.per_graph_loss(torch.as_tensor(pred), torch.as_tensor(ys),
                                        gm.loss_fun).cpu().numpy()
            loss[layer, head] = pl - clean_loss
        log(f"[beta-ablation] layer {layer + 1}/{gm.L}")
    names = list(feats[0])
    return {
        "graph_ids": np.asarray(order), "functional": func, "loss": loss,
        "clean_pred": clean, "y": ys,
        "feature_names": names,
        "features": np.column_stack([[f[k] for f in feats] for k in names]),
    }


def analyse_head_causality(score: dict, ablation: dict, throughput: np.ndarray,
                           *, per_graph_scores: Optional[dict] = None,
                           graph_weights: Optional[np.ndarray] = None,
                           n_boot: int, seed: int) -> dict:
    """Raw and multivariate partial score-impact associations with graph-bootstrap CIs."""
    L, H = int(score["L"]), int(score["H"])
    scores = {"semantic": np.asarray(score["S_sem"], float).reshape(-1),
              "structural": np.asarray(score["S_str"], float).reshape(-1)}
    layer = np.repeat(np.arange(L), H)
    throughput_array = np.asarray(throughput, float)
    throughput_mean = (throughput_array.mean(axis=0) if throughput_array.ndim == 3
                       else throughput_array).reshape(-1)
    graph_weights = (np.ones(ablation["functional"].shape[2], float)
                     if graph_weights is None else np.asarray(graph_weights, float))
    result = {}
    for outcome, cube in (("functional", ablation["functional"]),
                          ("loss", ablation["loss"])):
        mean_impact = np.asarray(cube, float).mean(axis=2).reshape(-1)
        result[outcome] = {}
        for channel, other in (("semantic", "structural"),
                               ("structural", "semantic")):
            controls = np.column_stack([scores[other], layer,
                                        np.log1p(np.maximum(throughput_mean, 0))])
            point_raw = _spearman(scores[channel], mean_impact)
            point_partial = partial_spearman(scores[channel], mean_impact, controls)

            def stat(idx, partial=False):
                imp = np.asarray(cube, float)[:, :, idx].mean(axis=2).reshape(-1)
                draw_scores = scores
                draw_throughput = throughput_mean
                if per_graph_scores is not None:
                    w = graph_weights[idx]
                    draw_scores = {
                        c: np.average(np.asarray(per_graph_scores[c], float)[idx],
                                      axis=0, weights=w).reshape(-1)
                        for c in CHANNELS
                    }
                    if throughput_array.ndim == 3:
                        draw_throughput = throughput_array[idx].mean(axis=0).reshape(-1)
                draw_controls = np.column_stack([
                    draw_scores[other], layer,
                    np.log1p(np.maximum(draw_throughput, 0)),
                ])
                return (partial_spearman(draw_scores[channel], imp, draw_controls)
                        if partial else _spearman(draw_scores[channel], imp))

            raw_ci = _bootstrap_stat(cube.shape[2], lambda i: stat(i),
                                     n_boot=n_boot, seed=seed + (0 if channel == "semantic" else 1))
            partial_ci = _bootstrap_stat(cube.shape[2], lambda i: stat(i, True),
                                         n_boot=n_boot, seed=seed + 20 + (0 if channel == "semantic" else 1))
            result[outcome][channel] = {
                "rho": point_raw, "ci": raw_ci,
                "partial_rho": point_partial, "partial_ci": partial_ci,
            }
    return result


def _common_graphs(task_results: dict) -> list[int]:
    sets = [set(map(int, task_results[t]["graph_ids"])) for t in BETA_TASKS]
    common = sorted(set.intersection(*sets))
    if not common or any(set(common) != s for s in sets):
        raise RuntimeError("the three checkpoints must use exactly the same ZINC graph IDs")
    return common


def _standardise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (x - np.mean(x, axis=0)) / scale


def _rank_standardise(x: np.ndarray) -> np.ndarray:
    return _standardise(_rank_columns(np.asarray(x, float)))


def _drop_constant_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    keep = np.isfinite(x).all(axis=0) & (np.std(x, axis=0) > 1e-12)
    return x[:, keep]


def compensation_path_coefficients(x, mediator, outcome, controls) -> dict[str, float]:
    """Rank-standardised graphwise X→M→Y pathway coefficients.

    These are associative path coefficients. They are deliberately not labelled causal
    mediation because the checkpoints were not generated by a randomised training intervention.
    """
    x = _rank_standardise(np.asarray(x, float)).reshape(-1)
    m = _rank_standardise(np.asarray(mediator, float)).reshape(-1)
    y = _rank_standardise(np.asarray(outcome, float)).reshape(-1)
    z = _drop_constant_columns(_rank_standardise(np.asarray(controls, float)))
    intercept = np.ones((len(x), 1))
    a_design = np.column_stack([intercept, x, z])
    a = float(np.linalg.lstsq(a_design, m, rcond=None)[0][1])
    c = float(np.linalg.lstsq(a_design, y, rcond=None)[0][1])
    b_design = np.column_stack([intercept, x, m, z])
    b_fit = np.linalg.lstsq(b_design, y, rcond=None)[0]
    return {"a": a, "b": float(b_fit[2]), "c": c,
            "c_prime": float(b_fit[1]), "indirect": a * float(b_fit[2])}


def _path_analysis(x, mediator, outcome, controls, *, n_boot: int, seed: int) -> dict:
    x = np.asarray(x, float); mediator = np.asarray(mediator, float)
    outcome = np.asarray(outcome, float); controls = np.asarray(controls, float)
    point = compensation_path_coefficients(x, mediator, outcome, controls)
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in point}
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(x), len(x))
        fitted = compensation_path_coefficients(
            x[idx], mediator[idx], outcome[idx], controls[idx])
        for key, value in fitted.items():
            if np.isfinite(value):
                draws[key].append(value)
    coefficients = {}
    for key, estimate in point.items():
        values = np.asarray(draws[key], float)
        ci = ((float(np.quantile(values, .025)), float(np.quantile(values, .975)))
              if values.size else (float("nan"), float("nan")))
        coefficients[key] = {"estimate": estimate, "ci": ci}

    rz = _drop_constant_columns(_rank_standardise(controls))
    rx, rm, ry = (_rank_standardise(v).reshape(-1) for v in (x, mediator, outcome))
    x_a = _residualise(rx, rz); m_a = _residualise(rm, rz)
    xz = np.column_stack([rx, rz])
    m_b = _residualise(rm, xz); y_b = _residualise(ry, xz)
    y_c = _residualise(ry, rz)
    return {
        "coefficients": coefficients,
        "residuals": {"a_x": x_a, "a_m": m_a, "b_m": m_b,
                      "b_y": y_b, "c_x": x_a, "c_y": y_c},
    }


def _causal_weighted_graph_scores(item: dict, graph_ids: Sequence[int],
                                  sem_scale: float, str_scale: float,
                                  *, weighted: bool) -> dict:
    collector: ResponseCollector = item["collector"]
    impact = np.asarray(item["ablation"]["functional"], float)
    ablation_ids = list(map(int, item["ablation"]["graph_ids"]))
    semantic, structural, throughput, error = [], [], [], []
    for gid in graph_ids:
        graph_impact = impact[:, :, ablation_ids.index(int(gid))].reshape(-1)
        weights = (graph_impact / (graph_impact.mean() + 1e-12)
                   if weighted else np.ones_like(graph_impact))
        if not np.isfinite(weights).all() or weights.sum() <= 1e-12:
            weights = np.ones_like(graph_impact)
        sg = collector.graph_score("semantic", gid) / sem_scale
        tg = collector.graph_score("structural", gid) / str_scale
        semantic.append(float(np.average(sg, weights=weights)))
        structural.append(float(np.average(tg, weights=weights)))
        throughput.append(float(np.asarray(collector.throughput[gid], float).mean()))
        error.append(float(collector.abs_error[gid]))
    semantic, structural = np.asarray(semantic), np.asarray(structural)
    return {
        "semantic": semantic, "structural": structural,
        "preference": np.log(semantic + 1e-12) - np.log(structural + 1e-12),
        "total": semantic + structural,
        "throughput": np.asarray(throughput), "error": np.asarray(error),
    }


def analyse_semantic_compensation(task_results: dict, *, n_boot: int, seed: int) -> dict:
    """Test RRWP coordinate novelty → semantic reweighting → local-model error penalty."""
    graph_ids = _common_graphs(task_results)
    global_item, local_item = task_results["zinc_1hop"], task_results["zinc_1hop_local"]
    sem_scale = float(np.mean([
        np.asarray(item["score"]["S_sem"], float).mean()
        for item in (global_item, local_item)
    ])) + 1e-12
    str_scale = float(np.mean([
        np.asarray(item["score"]["S_str"], float).mean()
        for item in (global_item, local_item)
    ])) + 1e-12

    weighted = {
        task: _causal_weighted_graph_scores(
            task_results[task], graph_ids, sem_scale, str_scale, weighted=True)
        for task in ("zinc_1hop", "zinc_1hop_local")
    }
    unweighted = {
        task: _causal_weighted_graph_scores(
            task_results[task], graph_ids, sem_scale, str_scale, weighted=False)
        for task in ("zinc_1hop", "zinc_1hop_local")
    }

    novelty = global_item["collector"].rrwp_novelty
    node = np.asarray([novelty[int(g)]["node_rms"] for g in graph_ids], float)
    pair = np.asarray([novelty[int(g)]["pair_rms"] for g in graph_ids], float)
    # Equal-weight, predeclared node/pair composite. Log compression prevents one large graph
    # from defining the result; molecule controls below remove ordinary size/topology effects.
    coordinate_novelty = np.mean(np.column_stack([
        _standardise(np.log(node + 1e-12)),
        _standardise(np.log(pair + 1e-12)),
    ]), axis=1)
    mediator = (weighted["zinc_1hop_local"]["preference"] -
                weighted["zinc_1hop"]["preference"])
    mediator_unweighted = (unweighted["zinc_1hop_local"]["preference"] -
                           unweighted["zinc_1hop"]["preference"])
    error_penalty = (weighted["zinc_1hop_local"]["error"] -
                     weighted["zinc_1hop"]["error"])

    ablation = global_item["ablation"]
    feature_ids = list(map(int, ablation["graph_ids"]))
    take = [feature_ids.index(int(g)) for g in graph_ids]
    all_features = np.asarray(ablation["features"], float)[take]
    names = list(ablation.get("feature_names", []))
    wanted = ("n_nodes", "n_rings", "diameter", "atom_entropy")
    selected_names = [name for name in wanted if name in names]
    columns = [names.index(name) for name in selected_names]
    molecule_controls = all_features[:, columns] if columns else all_features
    molecule_control_names = (selected_names if columns else
                              (names if names else [f"feature_{i}" for i in range(all_features.shape[1])]))
    total_shift = (np.log(weighted["zinc_1hop_local"]["total"] + 1e-12) -
                   np.log(weighted["zinc_1hop"]["total"] + 1e-12))
    throughput_shift = (np.log1p(weighted["zinc_1hop_local"]["throughput"]) -
                        np.log1p(weighted["zinc_1hop"]["throughput"]))
    controls = np.column_stack([molecule_controls, total_shift, throughput_shift])
    primary = _path_analysis(coordinate_novelty, mediator, error_penalty, controls,
                             n_boot=n_boot, seed=seed)
    robustness = _path_analysis(coordinate_novelty, mediator_unweighted, error_penalty,
                                controls, n_boot=n_boot, seed=seed + 1)
    primary_ci = primary["coefficients"]
    directional_support = all(primary_ci[key]["ci"][0] > 0
                              for key in ("a", "b", "indirect"))
    robust_support = robustness["coefficients"]["indirect"]["ci"][0] > 0
    contradicted = any(primary_ci[key]["ci"][1] < 0 for key in ("a", "b"))
    verdict = ("SUPPORTED ASSOCIATION" if directional_support and robust_support else
               "WEIGHT-SENSITIVE SIGNAL" if directional_support else
               "DIRECTION CONTRADICTED" if contradicted else "NO CLEAR PATHWAY")

    def novelty_matrix(task: str) -> np.ndarray:
        values = task_results[task]["collector"].rrwp_novelty
        return np.asarray([[values[int(g)]["node_rms"], values[int(g)]["pair_rms"]]
                           for g in graph_ids], float)

    global_novelty = novelty_matrix("zinc_1hop")
    dense_novelty = novelty_matrix("zinc")
    local_novelty = novelty_matrix("zinc_1hop_local")
    audit = {
        "local_high_order_rms_max": float(np.max(np.abs(local_novelty))),
        "dense_global_rms_max_abs_difference": float(np.max(
            np.abs(dense_novelty - global_novelty))),
        "node_rms_median": float(np.median(node)),
        "pair_rms_median": float(np.median(pair)),
    }
    return {
        "graph_ids": np.asarray(graph_ids, dtype=np.int64),
        "coordinate_novelty": coordinate_novelty,
        "node_novelty": node, "pair_novelty": pair,
        "semantic_compensation": mediator,
        "semantic_compensation_unweighted": mediator_unweighted,
        "local_error_penalty": error_penalty,
        "controls": controls, "control_names": [
            *molecule_control_names,
            "total_specialisation_shift", "clean_throughput_shift",
        ],
        "primary": primary, "unweighted_robustness": robustness,
        "substrate_audit": audit, "verdict": verdict,
    }


def _task_label(task: str) -> str:
    return {
        "zinc": "Dense GRIT\n(global RRWP)",
        "zinc_1hop": "1-hop GRIT\n(global RRWP)",
        "zinc_1hop_local": "1-hop GRIT\n(local RRWP)",
    }[task]


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2,
                         "axes.axisbelow": True, "figure.dpi": 160})
    return plt


def make_score_scatter(task_results: dict, out: Path) -> str:
    plt = _setup_matplotlib()
    gsem = np.mean([task_results[t]["score"]["S_sem"].mean() for t in BETA_TASKS]) + 1e-12
    gstr = np.mean([task_results[t]["score"]["S_str"].mean() for t in BETA_TASKS]) + 1e-12
    limit = 1.06 * max(max(np.max(task_results[t]["score"]["S_sem"] / gsem),
                           np.max(task_results[t]["score"]["S_str"] / gstr))
                       for t in BETA_TASKS)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=True, sharey=True,
                             constrained_layout=True)
    scatter = None
    for ax, task in zip(axes, BETA_TASKS):
        score = task_results[task]["score"]
        L, H = int(score["L"]), int(score["H"])
        x = (score["S_str"] / gstr).reshape(-1)
        y = (score["S_sem"] / gsem).reshape(-1)
        layer = np.repeat(np.arange(L), H)
        ax.plot([0, limit], [0, limit], "k:", lw=1)
        scatter = ax.scatter(x, y, c=layer, cmap="viridis", s=42, alpha=.88,
                             edgecolors="black", linewidths=.35, vmin=0, vmax=L - 1)
        ax.set_title(f"{_task_label(task)}\nMAE={score['test_metric']:.4f}", fontweight="bold")
        ax.set_xlim(0, limit); ax.set_ylim(0, limit)
        ax.set_xlabel("structural score (global-channel normalised)")
    axes[0].set_ylabel("semantic score (global-channel normalised)")
    fig.colorbar(scatter, ax=axes, label="layer index (0 = input)", shrink=.82)
    fig.suptitle("BETA · Separate-intervention per-head specialisation on ZINC",
                 fontsize=14, fontweight="bold")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return str(out)


def make_ablation_figure(task_results: dict, out: Path) -> str:
    from scipy.stats import rankdata
    plt = _setup_matplotlib()
    colors = {"semantic": "#1976b9", "structural": "#d55e00"}
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.0), sharex=True, sharey=True,
                             constrained_layout=True)
    for col, task in enumerate(BETA_TASKS):
        item = task_results[task]
        impact = item["ablation"]["functional"].mean(axis=2).reshape(-1)
        iy = rankdata(impact) / len(impact)
        for row, channel in enumerate(CHANNELS):
            score = item["score"]["S_sem" if channel == "semantic" else "S_str"].reshape(-1)
            sx = rankdata(score) / len(score)
            ax = axes[row, col]
            ax.scatter(sx, iy, s=30, alpha=.7, color=colors[channel], edgecolors="white", lw=.3)
            ax.plot([0, 1], [0, 1], color="#888", ls=":", lw=1)
            stats = item["causality"]["functional"][channel]
            ci = stats["partial_ci"]
            ax.set_title((f"{_task_label(task)}\n" if row == 0 else "") +
                         f"ρ={stats['rho']:+.2f}; partial ρ={stats['partial_rho']:+.2f} "
                         f"[{ci[0]:+.2f},{ci[1]:+.2f}]", fontsize=10, fontweight="bold")
            ax.set_xlabel(f"{channel} score rank")
            if col == 0:
                ax.set_ylabel("functional ablation-impact rank")
    fig.suptitle("Do intervention scores predict causal pre-head ablation impact?\n"
                 "Partial rank correlation controls the other channel, layer, and clean head throughput",
                 fontsize=13, fontweight="bold")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return str(out)


def make_compensation_figure(analysis: dict, out: Path) -> str:
    plt = _setup_matplotlib()
    path = analysis["primary"]
    residuals = path["residuals"]
    panels = (
        ("a", "a_x", "a_m", "High-order RRWP coordinate novelty",
         "Local − global causal-weighted semantic preference",
         "a · Does unavailable structure track compensation?"),
        ("b", "b_m", "b_y", "Causal-weighted semantic compensation",
         "Local − global absolute error",
         "b · Does compensation track the error penalty?"),
        ("c", "c_x", "c_y", "High-order RRWP coordinate novelty",
         "Local − global absolute error",
         "c · Does coordinate novelty track the error penalty?"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    colors = ("#1976b9", "#d55e00", "#4b7f52")
    for ax, (coefficient, xkey, ykey, xlabel, ylabel, title), color in zip(
            axes, panels, colors):
        x, y = np.asarray(residuals[xkey]), np.asarray(residuals[ykey])
        ax.axhline(0, color="#888", lw=.8); ax.axvline(0, color="#888", lw=.8)
        ax.scatter(x, y, s=30, alpha=.68, color=color, edgecolors="white", lw=.25)
        if np.std(x) > 0:
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, np.polyval(np.polyfit(x, y, 1), xx), color="#222", lw=1.5)
        stat = path["coefficients"][coefficient]
        ci = stat["ci"]
        ax.set_title(f"{title}\nβ={stat['estimate']:+.2f} "
                     f"[{ci[0]:+.2f},{ci[1]:+.2f}]", fontweight="bold", fontsize=10)
        ax.set_xlabel(xlabel + "\n(rank residual after controls)")
        ax.set_ylabel(ylabel + "\n(rank residual after controls)")
    indirect = path["coefficients"]["indirect"]
    robust = analysis["unweighted_robustness"]["coefficients"]["indirect"]
    fig.suptitle(
        f"BETA VERDICT: {analysis['verdict']}\n"
        "Structural-coordinate loss, semantic compensation, and the local-RRWP error penalty\n"
        f"Paired graph pathway (associative, not causal mediation): indirect a×b="
        f"{indirect['estimate']:+.2f} [{indirect['ci'][0]:+.2f},{indirect['ci'][1]:+.2f}]; "
        f"unweighted robustness={robust['estimate']:+.2f}",
        fontsize=12.5, fontweight="bold")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return str(out)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()
                if k not in {"x_residual", "y_residual", "x", "y", "matched_null"}}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _save_raw(task_results: dict, compensation: dict, out_dir: Path) -> tuple[str, str]:
    payload = {}
    for task, item in task_results.items():
        score, collector = item["score"], item["collector"]
        payload[f"{task}__S_sem"] = score["S_sem"]
        payload[f"{task}__S_str"] = score["S_str"]
        for channel in CHANNELS:
            records = collector.records[channel]
            payload[f"{task}__{channel}__response"] = np.concatenate(
                [r["response"] for r in records], axis=0)
            payload[f"{task}__{channel}__graph_id"] = np.concatenate([
                np.full(len(r["source_nodes"]), r["graph_id"], dtype=np.int64)
                for r in records
            ])
        payload[f"{task}__ablation_functional"] = item["ablation"]["functional"]
        payload[f"{task}__ablation_loss"] = item["ablation"]["loss"]
        gids = np.asarray(sorted(collector.abs_error), dtype=np.int64)
        payload[f"{task}__graph_ids"] = gids
        payload[f"{task}__absolute_error"] = np.asarray([collector.abs_error[int(g)] for g in gids])
        payload[f"{task}__throughput"] = np.stack([collector.throughput[int(g)] for g in gids])
        payload[f"{task}__rrwp_node_novelty"] = np.asarray([
            collector.rrwp_novelty[int(g)]["node_rms"] for g in gids])
        payload[f"{task}__rrwp_pair_novelty"] = np.asarray([
            collector.rrwp_novelty[int(g)]["pair_rms"] for g in gids])
    for key in ("graph_ids", "coordinate_novelty", "node_novelty", "pair_novelty",
                "semantic_compensation", "semantic_compensation_unweighted",
                "local_error_penalty", "controls"):
        payload[f"compensation__{key}"] = np.asarray(compensation[key])
    npz_path = out_dir / "beta_zinc_causal_specialisation.npz"
    np.savez_compressed(npz_path, **payload)
    summary = {
        "status": "BETA",
        "removed_analysis": [
            "effective head-response rank", "response-rank verdict",
            "single-head rescue double dissociation", "channel-response model-gap regression",
        ],
        "method": (
            "Existing separate semantic/structural interventions, validated by exact-graph "
            "pre-head ablation, followed by a paired graphwise test of the predeclared "
            "RRWP novelty -> semantic compensation -> local-model error pathway."
        ),
        "models": {
            task: {
                "title": item["score"]["title"],
                "test_metric": item["score"]["test_metric"],
                "score_reconstruction_max_abs": item["score_reconstruction_max_abs"],
                "causality": _jsonable(item["causality"]),
            } for task, item in task_results.items()
        },
        "semantic_compensation": _jsonable(compensation),
    }
    json_path = out_dir / "beta_zinc_causal_specialisation_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(npz_path), str(json_path)


def _mount_drive(mount_point: str = "/content/drive") -> None:
    try:
        from google.colab import drive
        log(f"[drive] Mounting Google Drive at {mount_point} ...")
        drive.mount(mount_point, force_remount=False)
    except Exception:  # noqa: BLE001
        log("[drive] google.colab unavailable; using paths as supplied.")


def run(
    *, tasks: Sequence[str] = BETA_TASKS, out_dir: str = DEFAULT_OUT_DIR,
    ckpt: Optional[dict] = None, num_graphs: int = 128, donors: int = 8,
    max_sources: Optional[int] = None,
    n_boot: int = 1000, analysis_seed: int = 0, bootstrap_seed: int = 1729,
    partner_match: str = "degree", seed: int = 42, accelerator: str = "cuda:0",
    num_threads: int = 4, mount: bool = True, skip_install: bool = False,
    pyg_version: str = "2.2.0", force_fresh_grit: bool = False,
) -> dict:
    """Run the isolated three-checkpoint causal-specialisation beta end to end."""
    tasks = tuple(tasks)
    if tasks != BETA_TASKS:
        raise ValueError(f"tasks must be exactly {BETA_TASKS}")
    if min(int(num_graphs), int(donors), int(n_boot)) < 1:
        raise ValueError("graph, donor, and bootstrap counts must be positive")
    if mount:
        _mount_drive()
    output = Path(out_dir); output.mkdir(parents=True, exist_ok=True)
    if not skip_install:
        env.install_dependencies(pyg_version=pyg_version)
    else:
        log("[deps] Skipping dependency installation (skip_install=True).")
    # The retired null analyses should not survive from an earlier run in the same Drive folder.
    for retired in ("fig_beta_zinc_response_rank_by_distance.png",
                    "fig_beta_zinc_response_rank_verdict.png",
                    "fig_beta_zinc_rescue_double_dissociation.png",
                    "fig_beta_zinc_graphwise_residual_gaps.png"):
        path = output / retired
        if path.exists():
            path.unlink()

    task_results = {}
    for task_index, task_name in enumerate(tasks):
        spec: GritTaskSpec = get_task(task_name)
        task_out = output / task_name; task_out.mkdir(parents=True, exist_ok=True)
        log("\n" + "#" * 88 + f"\n# BETA TASK: {spec.name} ({spec.title})\n" + "#" * 88)
        default_dir = f"/content/GRIT_{spec.name}" if spec.env_hooks else "/content/GRIT"
        repo_dir = Path(spec.grit_repo_dir or default_dir)
        env.clone_grit(repo_dir, spec.grit_repo, spec.grit_commit,
                       force_fresh=force_fresh_grit)
        for hook in spec.env_hooks:
            hook(repo_dir)
        env.prepare_inprocess_grit(repo_dir)
        config_file = env.resolve_config(spec, repo_dir, task_out)
        chosen_ckpt, _ = env.find_checkpoint(Path(spec.drive_dir) / "results",
                                             (ckpt or {}).get(task_name))
        log(f"[env] {platform.platform()} | python {sys.version.split()[0]}")
        sc = SpecConfig(
            ckpt=str(chosen_ckpt), out_dir=str(task_out),
            dataset_dir=resolve_dataset_dir(spec), config_file=config_file,
            accelerator=accelerator, seed=seed, num_threads=num_threads,
            num_graphs=num_graphs, donors=donors, ablation_graphs=num_graphs,
            analysis_seed=analysis_seed, partner_match=partner_match,
        )
        collector = ResponseCollector(task_name)
        score = score_model(spec, sc, with_attn_routing=False, max_sources=max_sources,
                            seed=analysis_seed, response_observer=collector)
        gm = score["gm"]
        sem_rebuilt = collector.score_reconstruction("semantic").reshape(score["L"], score["H"])
        str_rebuilt = collector.score_reconstruction("structural").reshape(score["L"], score["H"])
        reconstruction_error = max(float(np.max(np.abs(sem_rebuilt - score["S_sem"]))),
                                   float(np.max(np.abs(str_rebuilt - score["S_str"]))))
        reconstruction_tol = max(1e-8, 1e-4 * float(max(score["S_sem"].max(),
                                                        score["S_str"].max(), 1e-12)))
        if reconstruction_error > reconstruction_tol:
            raise RuntimeError(f"graphwise responses do not reconstruct scores: "
                               f"{reconstruction_error:.3e} > {reconstruction_tol:.3e}")
        graph_ids = np.asarray(score["graph_ids"], dtype=np.int64)
        if not collector.throughput:
            raise RuntimeError("clean head throughput was not captured by the scorer")
        throughput_graph = np.stack([collector.throughput[int(g)] for g in graph_ids])
        per_graph_scores = {
            channel: np.stack([collector.graph_score(channel, int(g)).reshape(
                score["L"], score["H"]) for g in graph_ids])
            for channel in CHANNELS
        }
        source_counts = np.asarray([
            len(next(r for r in collector.records["semantic"]
                     if int(r["graph_id"]) == int(g))["source_nodes"])
            for g in graph_ids
        ], dtype=float)
        ablation = run_head_ablation(gm, graph_ids)
        if not np.array_equal(ablation["graph_ids"], graph_ids):
            raise RuntimeError("ablation graphs are not aligned with score graphs")
        causality = analyse_head_causality(
            score, ablation, throughput_graph, per_graph_scores=per_graph_scores,
            graph_weights=source_counts, n_boot=n_boot,
            seed=bootstrap_seed + task_index * 100,
        )
        log(f"[beta-causal] functional partial rho: semantic="
            f"{causality['functional']['semantic']['partial_rho']:+.2f}, structural="
            f"{causality['functional']['structural']['partial_rho']:+.2f}")
        score.pop("gm", None)
        task_results[task_name] = {
            "score": score, "collector": collector, "graph_ids": graph_ids,
            "score_reconstruction_max_abs": reconstruction_error,
            "ablation": ablation, "causality": causality,
        }
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    compensation = analyse_semantic_compensation(
        task_results, n_boot=n_boot, seed=bootstrap_seed + 1000)
    coefficients = compensation["primary"]["coefficients"]
    log(f"[beta-compensation] verdict: {compensation['verdict']}")
    log("[beta-compensation] RRWP novelty -> semantic compensation -> local error penalty:")
    for name in ("a", "b", "indirect", "c", "c_prime"):
        stat = coefficients[name]
        log(f"  {name}: {stat['estimate']:+.3f} "
            f"[{stat['ci'][0]:+.3f}, {stat['ci'][1]:+.3f}]")
    audit = compensation["substrate_audit"]
    log(f"[beta-compensation] substrate audit: local high-order max="
        f"{audit['local_high_order_rms_max']:.3e}; dense/global max difference="
        f"{audit['dense_global_rms_max_abs_difference']:.3e}")
    figures = {
        "score_scatter": make_score_scatter(
            task_results, output / "fig_beta_zinc_specialisation_scatter.png"),
        "score_ablation": make_ablation_figure(
            task_results, output / "fig_beta_zinc_score_ablation_causality.png"),
        "rrwp_semantic_compensation": make_compensation_figure(
            compensation, output / "fig_beta_zinc_rrwp_semantic_compensation.png"),
    }
    raw_npz, summary_json = _save_raw(task_results, compensation, output)
    log("\n" + "=" * 88)
    log("BETA complete. Retired rank, rescue, and generic response-gap outputs were not generated.")
    for name, path in figures.items():
        log(f"  {name}: {path}")
    return {
        "status": "BETA", "figures": figures, "raw_npz": raw_npz,
        "summary_json": summary_json, "semantic_compensation": compensation,
        "task_results": task_results, "out_dir": str(output),
    }
