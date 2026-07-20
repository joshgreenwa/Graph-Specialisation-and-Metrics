"""BETA: causal validation and graph-gap analysis of ZINC head specialisation.

This standalone experiment retains the established semantic donor swap and mask-frozen
structural transposition.  It deliberately removes the earlier effective-response-rank beta.
The score is tested in three direct ways: prediction under pre-head ablation, clean-head rescue
of matched semantic/structural corruptions, and paired per-graph associations with the accuracy
gaps between dense, global-RRWP 1-hop, and local-RRWP 1-hop GRIT.
"""

from __future__ import annotations

import contextlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..carriage import env, metrics, structural
from ..carriage.env import log
from ..carriage.grit_runner import _spd
from ..carriage.tasks import GritTaskSpec, get_task
from .ablation import _build_groups
from .model import SpecConfig
from .scores import _perturb_mask_frozen, score_model


BETA_TASKS = ("zinc", "zinc_1hop", "zinc_1hop_local")
DEFAULT_OUT_DIR = "/content/drive/MyDrive/graph_specialisation_metrics/beta_zinc_causal_specialisation"
BIN_LABELS = ("0", "1", "2", "3", "4-7", "8+")
CHANNELS = ("semantic", "structural")


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
    _spd_cache: dict[int, np.ndarray] = field(default_factory=dict)

    def __call__(
        self, *, channel, graph_id, base, source_nodes, phi_stack,
        donor_averaged_delta, clean_prediction, clean_head_output=None,
    ) -> None:
        import torch

        source_nodes = np.asarray(source_nodes, dtype=np.int64)
        n = int(base.num_nodes)
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


def select_score_families(score: dict, *, family_size: int = 3) -> dict[str, list[tuple[int, int]]]:
    """Score-only semantic/structural families, matching the synthetic protocol."""
    sem, st = np.asarray(score["S_sem"], float), np.asarray(score["S_str"], float)
    sn, tn = sem / (sem.mean() + 1e-12), st / (st.mean() + 1e-12)
    eligible = (sn + tn).reshape(-1) >= np.quantile((sn + tn).reshape(-1), .4)
    selectivity = (np.log(sn + 1e-12) - np.log(tn + 1e-12)).reshape(-1)
    sem_order = [int(i) for i in np.argsort(selectivity)[::-1] if eligible[i]]
    str_order = [int(i) for i in np.argsort(selectivity) if eligible[i]]
    sem_idx = sem_order[:family_size]
    str_idx = [i for i in str_order if i not in sem_idx][:family_size]
    if len(sem_idx) < family_size or len(str_idx) < family_size:
        raise RuntimeError("not enough distinct eligible heads for the requested families")
    return {
        "semantic": [tuple(map(int, np.unravel_index(i, sem.shape))) for i in sem_idx],
        "structural": [tuple(map(int, np.unravel_index(i, st.shape))) for i in str_idx],
    }


@contextlib.contextmanager
def _patch_head_output(gm, layer: int, head: int, clean_wv):
    def hook(_module, _inputs, output):
        h_out, e_out = output
        if tuple(h_out.shape) != tuple(clean_wv.shape):
            raise RuntimeError("clean/corrupt routed-output alignment failed")
        changed = h_out.clone()
        changed[:, int(head), :] = clean_wv[:, int(head), :].to(changed)
        return changed, e_out

    handle = gm.attn_layers[int(layer)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _donor_pool(gm) -> tuple[np.ndarray, np.ndarray]:
    rows, gids = [], []
    for gi in range(len(gm.donor_ds)):
        r = gm.adapter.rows(gm.donor_ds[gi])
        rows.append(r)
        gids.append(np.full(len(r), gi, dtype=np.int64))
    return np.concatenate(rows), np.concatenate(gids)


def _make_corruptions(gm, graph_ids, channel: str, *, seed: int):
    rng = np.random.default_rng(seed)
    donor_rows = donor_gids = None
    if channel == "semantic":
        donor_rows, donor_gids = _donor_pool(gm)
    clean, corrupt, meta = [], [], []
    for gi in map(int, graph_ids):
        base = gm.eval_ds[gi]
        n = int(base.num_nodes)
        source = int(rng.integers(n))
        if channel == "semantic":
            own = gm.adapter.rows(base)[source]
            candidates = np.flatnonzero((donor_gids != gi) & (donor_rows != own).any(axis=1))
            donor = int(rng.choice(candidates))
            pert = base.clone()
            gm.adapter.write_donors(pert.x, np.asarray([source]), donor_rows[[donor]])
            meta.append((source, donor))
        elif channel == "structural":
            deg = structural.node_degrees(base.edge_index, n)
            candidates = np.flatnonzero((deg == deg[source]) & (np.arange(n) != source))
            if not len(candidates):
                distance = np.abs(deg - deg[source]).astype(float)
                distance[source] = np.inf
                candidates = np.flatnonzero(distance == distance.min())
            partner = int(rng.choice(candidates))
            pert = _perturb_mask_frozen(base, source, partner)
            meta.append((source, partner))
        else:
            raise ValueError(channel)
        clean.append(base)
        corrupt.append(pert)
    return clean, corrupt, np.asarray(meta, dtype=np.int64)


def run_rescue_sweep(gm, graph_ids, channel: str, *, seed: int,
                     batch_size: int = 64, effect_floor: float = 1e-5) -> dict:
    """Patch one clean routed head into a matched corrupt run; no head is selected here."""
    import torch
    from torch_geometric.data import Batch

    graph_ids = np.asarray(graph_ids, dtype=np.int64)
    clean, corrupt, meta = _make_corruptions(gm, graph_ids, channel, seed=seed)
    G = len(clean)
    mediation = np.full((gm.L, gm.H, G), np.nan, dtype=np.float32)
    recovery = np.full_like(mediation, np.nan)
    effect = np.zeros(G, dtype=np.float32)
    clean_pred, corrupt_pred = [], []
    for start in range(0, G, batch_size):
        stop = min(G, start + batch_size)
        cb = Batch.from_data_list(clean[start:stop]).to(gm.device)
        cap = gm.capture(cb, want_grad=False, want_attn=False)
        cp = cap["pred"].reshape(stop - start, -1).detach()
        clean_wv = [x.detach() for x in cap["wV"]]
        with torch.no_grad():
            xb = Batch.from_data_list(corrupt[start:stop]).to(gm.device)
            xp, _ = gm.model(xb)
            xp = xp.reshape(stop - start, -1).detach()
        total = cp - xp
        denom = total.square().sum(dim=-1)
        effect[start:stop] = denom.sqrt().cpu().numpy()
        y = cb.y.reshape(cp.shape).to(cp.dtype)
        clean_loss = metrics.per_graph_loss(cp, y, gm.loss_fun)
        corrupt_loss = metrics.per_graph_loss(xp, y, gm.loss_fun)
        loss_denom = corrupt_loss - clean_loss
        clean_pred.append(cp.cpu().numpy()); corrupt_pred.append(xp.cpu().numpy())
        for layer in range(gm.L):
            for head in range(gm.H):
                # GRIT's encoders mutate a Batch in-place, so every forward gets a fresh Batch.
                xb = Batch.from_data_list(corrupt[start:stop]).to(gm.device)
                with _patch_head_output(gm, layer, head, clean_wv[layer]), torch.no_grad():
                    pp, _ = gm.model(xb)
                    pp = pp.reshape(stop - start, -1).detach()
                mem = ((pp - xp) * total).sum(dim=-1) / (denom + 1e-12)
                mem = torch.where(denom.sqrt() >= effect_floor, mem,
                                  torch.full_like(mem, float("nan")))
                patched_loss = metrics.per_graph_loss(pp, y, gm.loss_fun)
                lr = (corrupt_loss - patched_loss) / (loss_denom + 1e-12)
                lr = torch.where(loss_denom > 1e-5, lr,
                                 torch.full_like(lr, float("nan")))
                mediation[layer, head, start:stop] = mem.cpu().numpy()
                recovery[layer, head, start:stop] = lr.cpu().numpy()
        log(f"[beta-rescue:{channel}] {stop}/{G}")
    return {
        "graph_ids": graph_ids, "mediation": mediation, "loss_recovery": recovery,
        "effect_norm": effect, "clean_pred": np.concatenate(clean_pred),
        "corrupt_pred": np.concatenate(corrupt_pred), "corruption_meta": meta,
        "valid_effect_fraction": float(np.mean(effect >= effect_floor)),
    }


def _matched_families(selected, throughput, *, n: int, seed: int):
    """Layer-exact, throughput-near random family controls."""
    rng = np.random.default_rng(seed)
    throughput = np.asarray(throughput, float)
    L, H = throughput.shape
    blocked = set(selected["semantic"]) | set(selected["structural"])
    output = []
    for _ in range(int(n)):
        pair, used = {}, set()
        for name in CHANNELS:
            heads = []
            for target in selected[name]:
                candidates = [(target[0], h) for h in range(H)
                              if (target[0], h) not in blocked | used]
                if not candidates:
                    candidates = [(target[0], h) for h in range(H)
                                  if (target[0], h) not in used]
                d = np.asarray([abs(np.log1p(throughput[c]) -
                                       np.log1p(throughput[target])) for c in candidates])
                scale = max(float(np.median(d[d > 0])) if np.any(d > 0) else 1.0, 1e-8)
                p = np.exp(-d / scale); p /= p.sum()
                choice = candidates[int(rng.choice(len(candidates), p=p))]
                heads.append(choice); used.add(choice)
            pair[name] = heads
        output.append(pair)
    return output


def _family_graph_values(cube, heads):
    return np.nanmean(np.stack([np.asarray(cube[h], float) for h in heads]), axis=0)


def analyse_double_dissociation(score, rescue, throughput, *, family_size: int,
                                n_null: int, n_boot: int, seed: int) -> dict:
    selected = select_score_families(score, family_size=family_size)
    graph_matrix = np.empty((2, 2, len(rescue["semantic"]["graph_ids"])), float)
    for r, family in enumerate(CHANNELS):
        for c, corruption in enumerate(CHANNELS):
            graph_matrix[r, c] = _family_graph_values(
                rescue[corruption]["mediation"], selected[family]
            )
    matrix = np.nanmean(graph_matrix, axis=2)
    interaction_graph = ((graph_matrix[0, 0] - graph_matrix[0, 1]) +
                         (graph_matrix[1, 1] - graph_matrix[1, 0]))
    interaction = float(np.nanmean(interaction_graph))
    ci = _bootstrap_stat(len(interaction_graph),
                         lambda i: float(np.nanmean(interaction_graph[i])),
                         n_boot=n_boot, seed=seed)
    null_sets = _matched_families(selected, throughput, n=n_null, seed=seed + 1)
    null = []
    for controls in null_sets:
        m = np.empty((2, 2), float)
        for r, family in enumerate(CHANNELS):
            for c, corruption in enumerate(CHANNELS):
                m[r, c] = np.nanmean(_family_graph_values(
                    rescue[corruption]["mediation"], controls[family]
                ))
        null.append((m[0, 0] - m[0, 1]) + (m[1, 1] - m[1, 0]))
    null = np.asarray(null, float)
    finite_null = null[np.isfinite(null)]
    p = (float((1 + np.sum(finite_null >= interaction)) / (len(finite_null) + 1))
         if np.isfinite(interaction) and finite_null.size else float("nan"))
    return {
        "selected_heads": {k: [list(h) for h in v] for k, v in selected.items()},
        "mediation_matrix": matrix, "interaction": interaction,
        "interaction_ci": ci, "matched_null": null, "p_ge": p,
        "valid_effect_fraction": {c: rescue[c]["valid_effect_fraction"] for c in CHANNELS},
    }


def _common_graphs(task_results: dict) -> list[int]:
    sets = [set(map(int, task_results[t]["graph_ids"])) for t in BETA_TASKS]
    common = sorted(set.intersection(*sets))
    if not common or any(set(common) != s for s in sets):
        raise RuntimeError("the three checkpoints must use exactly the same ZINC graph IDs")
    return common


def _graph_arrays(item: dict, graph_ids: Sequence[int]) -> dict:
    collector: ResponseCollector = item["collector"]
    family = {k: [tuple(h) for h in v]
              for k, v in item["double_dissociation"]["selected_heads"].items()}
    shape = np.asarray(item["score"]["S_sem"]).shape
    sem_idx = [np.ravel_multi_index(h, shape) for h in family["semantic"]]
    str_idx = [np.ravel_multi_index(h, shape) for h in family["structural"]]
    sem, st, far_sem, selected_mass, throughput, error = [], [], [], [], [], []
    for gid in graph_ids:
        sg = collector.graph_score("semantic", gid)
        tg = collector.graph_score("structural", gid)
        fg = collector.graph_score("semantic", gid, bins=(4, 5))
        sem.append(sg.sum()); st.append(tg.sum())
        far_sem.append(fg[sem_idx].sum())
        selected_mass.append(sg[sem_idx].sum() + tg[str_idx].sum())
        throughput.append(np.asarray(collector.throughput[gid], float).mean())
        error.append(collector.abs_error[gid])
    return {k: np.asarray(v, float) for k, v in {
        "semantic_mass": sem, "structural_mass": st, "far_semantic_family": far_sem,
        "selected_mass": selected_mass, "throughput": throughput, "error": error,
    }.items()}


def _controlled_association(x, y, controls, *, n_boot: int, seed: int) -> dict:
    x, y, controls = np.asarray(x, float), np.asarray(y, float), np.asarray(controls, float)
    rx, ry, rz = _rank_columns(x), _rank_columns(y), _rank_columns(controls)
    xres, yres = _residualise(rx, rz), _residualise(ry, rz)
    rho = (float(np.corrcoef(xres, yres)[0, 1])
           if np.std(xres) >= 1e-12 and np.std(yres) >= 1e-12 else float("nan"))
    ci = _bootstrap_stat(len(x),
                         lambda i: partial_spearman(x[i], y[i], controls[i]),
                         n_boot=n_boot, seed=seed)
    return {"rho": rho, "ci": ci, "x_residual": xres, "y_residual": yres,
            "x": x, "y": y}


def analyse_graphwise_gaps(task_results: dict, *, n_boot: int, seed: int) -> dict:
    """Paired graph residual gaps tested against channel-specific score contributions."""
    graph_ids = _common_graphs(task_results)
    arr = {t: _graph_arrays(task_results[t], graph_ids) for t in BETA_TASKS}
    ablation = task_results["zinc"]["ablation"]
    all_features = np.asarray(ablation["features"], float)
    names = list(ablation.get("feature_names", []))
    wanted = ("n_nodes", "n_rings", "diameter", "atom_entropy")
    columns = [names.index(k) for k in wanted if k in names]
    features = all_features[:, columns] if columns else all_features
    feature_ids = list(map(int, ablation["graph_ids"]))
    take = [feature_ids.index(g) for g in graph_ids]
    features = features[take]

    def compare(first, second, predictor, nuisance, label, offset):
        # Positive y means the first model has lower absolute error than the second.
        y = arr[second]["error"] - arr[first]["error"]
        x = arr[first][predictor] - arr[second][predictor]
        controls = np.column_stack([
            arr[first][nuisance] - arr[second][nuisance],
            arr[first]["throughput"] - arr[second]["throughput"],
            features,
        ])
        out = _controlled_association(x, y, controls, n_boot=n_boot, seed=seed + offset)
        out.update({"first": first, "second": second, "predictor": predictor,
                    "label": label, "nuisance": nuisance})
        return out

    comparisons = {
        "global_vs_local": compare(
            "zinc_1hop", "zinc_1hop_local", "structural_mass", "semantic_mass",
            "Global-RRWP structural mass", 0,
        ),
        "dense_vs_global": compare(
            "zinc", "zinc_1hop", "far_semantic_family", "structural_mass",
            "Dense excess long-range semantic-head carriage", 1,
        ),
        "dense_vs_local": compare(
            "zinc", "zinc_1hop_local", "far_semantic_family", "structural_mass",
            "Dense excess long-range semantic-head carriage", 2,
        ),
    }
    # Closely related score-only alternative: total mass carried by selected channel families.
    for key, item in comparisons.items():
        first, second = item["first"], item["second"]
        x = arr[first]["selected_mass"] - arr[second]["selected_mass"]
        y = arr[second]["error"] - arr[first]["error"]
        controls = np.column_stack([
            arr[first]["throughput"] - arr[second]["throughput"], features
        ])
        item["selected_mass_association"] = _controlled_association(
            x, y, controls, n_boot=n_boot, seed=seed + 10 + list(comparisons).index(key)
        )
    return {"graph_ids": graph_ids, "comparisons": comparisons}


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


def make_rescue_figure(task_results: dict, out: Path) -> str:
    plt = _setup_matplotlib()
    matrices = [np.asarray(task_results[t]["double_dissociation"]["mediation_matrix"], float)
                for t in BETA_TASKS]
    finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
    vmax = (float(np.max(np.abs(finite))) if finite.size else 1.0) + 1e-9
    fig = plt.figure(figsize=(15.2, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25])
    image = None
    for i, (task, matrix) in enumerate(zip(BETA_TASKS, matrices)):
        ax = fig.add_subplot(gs[0, i])
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for r in range(2):
            for c in range(2):
                text = f"{matrix[r,c]:+.2f}" if np.isfinite(matrix[r, c]) else "NA"
                ax.text(c, r, text, ha="center", va="center",
                        fontweight="bold")
        dd = task_results[task]["double_dissociation"]
        ci = dd["interaction_ci"]
        ax.set_xticks([0, 1], ["semantic", "structural"], rotation=15)
        ax.set_yticks([0, 1], ["semantic heads", "structural heads"])
        ax.set_xlabel("corruption rescued")
        ax.set_title(f"{_task_label(task)}\nDD={dd['interaction']:+.2f} "
                     f"[{ci[0]:+.2f},{ci[1]:+.2f}], pnull={dd['p_ge']:.3f}",
                     fontsize=10, fontweight="bold")
    ax = fig.add_subplot(gs[0, 3])
    vals = np.asarray([task_results[t]["double_dissociation"]["interaction"] for t in BETA_TASKS])
    lo = np.asarray([task_results[t]["double_dissociation"]["interaction_ci"][0] for t in BETA_TASKS])
    hi = np.asarray([task_results[t]["double_dissociation"]["interaction_ci"][1] for t in BETA_TASKS])
    ax.axhline(0, color="black", lw=1)
    ax.errorbar(np.arange(3), vals, yerr=np.vstack([vals - lo, hi - vals]), fmt="o",
                capsize=4, color="#333")
    ax.set_xticks(np.arange(3), ["dense", "1-hop\nglobal", "1-hop\nlocal"])
    ax.set_ylabel("semantic/structural interaction")
    ax.set_title("Score-selected family\ndouble dissociation", fontweight="bold")
    fig.colorbar(image, ax=fig.axes[:3], shrink=.7, label="fraction of corruption effect mediated")
    fig.suptitle("Causal rescue at the established routed head output\n"
                 "Clean one-head patches; families selected from scores only; nulls match layer and throughput",
                 fontsize=13, fontweight="bold")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return str(out)


def make_graph_gap_figure(analysis: dict, out: Path) -> str:
    plt = _setup_matplotlib()
    keys = ("global_vs_local", "dense_vs_global", "dense_vs_local")
    titles = ("1-hop global RRWP vs local RRWP", "Dense vs 1-hop global RRWP",
              "Dense vs 1-hop local RRWP")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)
    for ax, key, title in zip(axes, keys, titles):
        item = analysis["comparisons"][key]
        x, y = np.asarray(item["x_residual"]), np.asarray(item["y_residual"])
        ax.axhline(0, color="#888", lw=.8); ax.axvline(0, color="#888", lw=.8)
        ax.scatter(x, y, s=28, alpha=.65, color="#3c6e9f", edgecolors="white", lw=.25)
        if np.std(x) > 0:
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, np.polyval(np.polyfit(x, y, 1), xx), color="#c33", lw=1.5)
        ci = item["ci"]
        alt = item["selected_mass_association"]
        ax.set_title(f"{title}\npartial ρ={item['rho']:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}]\n"
                     f"selected-mass partial ρ={alt['rho']:+.2f}", fontweight="bold", fontsize=10)
        ax.set_xlabel(item["label"] + "\n(rank residual after controls)")
        ax.set_ylabel("per-graph absolute-error advantage\n(rank residual after controls)")
    fig.suptitle("Do channel-specific responses explain paired model performance gaps?\n"
                 "Controls: nuisance channel, clean throughput, and molecule size/structure/content features",
                 fontsize=13, fontweight="bold")
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


def _save_raw(task_results: dict, graphwise: dict, out_dir: Path) -> tuple[str, str]:
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
            payload[f"{task}__rescue_{channel}__mediation"] = item["rescue"][channel]["mediation"]
            payload[f"{task}__rescue_{channel}__loss_recovery"] = item["rescue"][channel]["loss_recovery"]
        payload[f"{task}__ablation_functional"] = item["ablation"]["functional"]
        payload[f"{task}__ablation_loss"] = item["ablation"]["loss"]
        gids = np.asarray(sorted(collector.abs_error), dtype=np.int64)
        payload[f"{task}__graph_ids"] = gids
        payload[f"{task}__absolute_error"] = np.asarray([collector.abs_error[int(g)] for g in gids])
        payload[f"{task}__throughput"] = np.stack([collector.throughput[int(g)] for g in gids])
    npz_path = out_dir / "beta_zinc_causal_specialisation.npz"
    np.savez_compressed(npz_path, **payload)
    summary = {
        "status": "BETA",
        "removed_analysis": ["effective head-response rank", "response-rank verdict"],
        "method": (
            "Existing separate semantic/structural interventions, validated by exact-graph "
            "pre-head ablation, clean routed-head rescue, and paired graphwise model-gap tests."
        ),
        "models": {
            task: {
                "title": item["score"]["title"],
                "test_metric": item["score"]["test_metric"],
                "score_reconstruction_max_abs": item["score_reconstruction_max_abs"],
                "causality": _jsonable(item["causality"]),
                "double_dissociation": _jsonable(item["double_dissociation"]),
            } for task, item in task_results.items()
        },
        "graphwise_gaps": _jsonable(graphwise),
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
    max_sources: Optional[int] = None, rescue_graphs: Optional[int] = None,
    rescue_batch_size: int = 64, family_size: int = 3, n_null: int = 1000,
    n_boot: int = 1000, analysis_seed: int = 0, bootstrap_seed: int = 1729,
    partner_match: str = "degree", seed: int = 42, accelerator: str = "cuda:0",
    num_threads: int = 4, mount: bool = True, skip_install: bool = False,
    pyg_version: str = "2.2.0", force_fresh_grit: bool = False,
) -> dict:
    """Run the isolated three-checkpoint causal-specialisation beta end to end."""
    tasks = tuple(tasks)
    if tasks != BETA_TASKS:
        raise ValueError(f"tasks must be exactly {BETA_TASKS}")
    if min(int(num_graphs), int(donors), int(n_boot), int(n_null), int(family_size)) < 1:
        raise ValueError("graph, donor, bootstrap, null, and family counts must be positive")
    if mount:
        _mount_drive()
    output = Path(out_dir); output.mkdir(parents=True, exist_ok=True)
    if not skip_install:
        env.install_dependencies(pyg_version=pyg_version)
    else:
        log("[deps] Skipping dependency installation (skip_install=True).")

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
            dataset_dir=str(Path(spec.drive_dir) / "datasets"), config_file=config_file,
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
        throughput = throughput_graph.mean(axis=0)
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
        available_rescue = np.setdiff1d(np.arange(len(gm.eval_ds), dtype=np.int64), graph_ids)
        rg = min(int(rescue_graphs or num_graphs), len(available_rescue))
        rescue_rng = np.random.default_rng(analysis_seed + 77_777)
        rescue_ids = np.sort(rescue_rng.choice(available_rescue, size=rg, replace=False))
        if rg < int(rescue_graphs or num_graphs):
            log(f"[beta-rescue] only {rg} score-held-out eval graphs are available")
        rescue = {
            channel: run_rescue_sweep(
                gm, rescue_ids, channel,
                seed=analysis_seed + 10_000 + (0 if channel == "semantic" else 1),
                batch_size=rescue_batch_size,
            ) for channel in CHANNELS
        }
        dissociation = analyse_double_dissociation(
            score, rescue, throughput, family_size=family_size, n_null=n_null,
            n_boot=n_boot, seed=bootstrap_seed + task_index * 100 + 50,
        )
        log(f"[beta-causal] functional partial rho: semantic="
            f"{causality['functional']['semantic']['partial_rho']:+.2f}, structural="
            f"{causality['functional']['structural']['partial_rho']:+.2f}; "
            f"rescue DD={dissociation['interaction']:+.2f}, pnull={dissociation['p_ge']:.3f}")
        score.pop("gm", None)
        task_results[task_name] = {
            "score": score, "collector": collector, "graph_ids": graph_ids,
            "score_reconstruction_max_abs": reconstruction_error,
            "ablation": ablation, "causality": causality, "rescue": rescue,
            "double_dissociation": dissociation,
        }
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    rescue_reference = task_results[BETA_TASKS[0]]["rescue"]["semantic"]["graph_ids"]
    for task in BETA_TASKS:
        for channel in CHANNELS:
            if not np.array_equal(task_results[task]["rescue"][channel]["graph_ids"],
                                  rescue_reference):
                raise RuntimeError("rescue graphs are not paired across model/channel runs")
    graphwise = analyse_graphwise_gaps(task_results, n_boot=n_boot, seed=bootstrap_seed + 1000)
    log("[beta-graphwise] paired model-gap associations (partial Spearman):")
    for name, item in graphwise["comparisons"].items():
        log(f"  {name}: rho={item['rho']:+.2f} "
            f"[{item['ci'][0]:+.2f}, {item['ci'][1]:+.2f}]")
    figures = {
        "score_scatter": make_score_scatter(
            task_results, output / "fig_beta_zinc_specialisation_scatter.png"),
        "score_ablation": make_ablation_figure(
            task_results, output / "fig_beta_zinc_score_ablation_causality.png"),
        "rescue_double_dissociation": make_rescue_figure(
            task_results, output / "fig_beta_zinc_rescue_double_dissociation.png"),
        "graphwise_gaps": make_graph_gap_figure(
            graphwise, output / "fig_beta_zinc_graphwise_residual_gaps.png"),
    }
    raw_npz, summary_json = _save_raw(task_results, graphwise, output)
    log("\n" + "=" * 88)
    log("BETA complete. Effective-rank outputs were not generated.")
    for name, path in figures.items():
        log(f"  {name}: {path}")
    return {
        "status": "BETA", "figures": figures, "raw_npz": raw_npz,
        "summary_json": summary_json, "graphwise_gaps": graphwise,
        "task_results": task_results, "out_dir": str(output),
    }
