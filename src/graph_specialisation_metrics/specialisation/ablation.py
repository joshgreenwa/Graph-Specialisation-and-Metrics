"""Causal head-ablation analysis: do the score-selected heads matter more than random heads?

Ablation = zero a head's routed value ``wV[:, head, :]`` (``model.collect_preds_ablated``). For
the well-fit dense ZINC model (and its 1-hop control) we ask, on the eval split:

  (1) Does ablating the TOP semantic / TOP structural / TOP-JOINT head raise the error more than
      ablating a random head? -> each target's rank / percentile / z-score against the null
      distribution of all single-head impacts, plus a random-pair null for joint ablation.
  (2) Is a head's ablation impact predicted by its specialisation score? -> Spearman(score, impact)
      over all heads (a validity check: the transport score should track causal importance).
  (3) Does the top STRUCTURAL head matter more on more-structural inputs, and the top SEMANTIC head
      on more content-diverse inputs? -> per-graph impact vs per-graph graph features (rings,
      diameter, degree; atom-type diversity), reported as Spearman correlations.

Impact is reported two ways per graph: functional |pred_ablate - pred_clean| (label-free) and the
loss change |pred_ablate - y| - |pred_clean - y| (MAE units, signed; >0 = ablation hurt).
"""

from __future__ import annotations

import numpy as np

from ..carriage.env import log
from ..carriage.grit_runner import _spd


# --------------------------------------------------------------------------------------- #
# Per-graph structural / content features (the "how structural is this molecule" axis).
# --------------------------------------------------------------------------------------- #
def graph_features(data) -> dict:
    import torch

    n = int(data.num_nodes)
    ei = data.edge_index
    e_undir = int(ei.shape[1] // 2)                       # ZINC stores each bond both ways
    D = _spd(data, n)
    finite = D[np.isfinite(D)]
    diameter = float(finite.max()) if finite.size else 0.0
    avg_deg = 2.0 * e_undir / max(n, 1)
    n_rings = e_undir - n + 1                             # cyclomatic number (connected molecule)
    # Content diversity, task-general: use the first content column (ZINC atom type; OGB's first
    # atom feature = atomic number) so multi-field node encoders (peptides x is [n,9]) don't flatten.
    xc = data.x
    xc = (xc[:, 0] if xc.dim() > 1 else xc).cpu().numpy()
    _, counts = np.unique(xc, return_counts=True)
    p = counts / counts.sum()
    atom_entropy = float(-(p * np.log(p + 1e-12)).sum())
    return {
        "n_nodes": float(n), "n_edges": float(e_undir), "n_rings": float(n_rings),
        "diameter": diameter, "avg_deg": avg_deg,
        "n_atom_types": float(len(counts)), "atom_entropy": atom_entropy,
    }


def _spearman(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])); ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------------------- #
# The ablation sweep.
# --------------------------------------------------------------------------------------- #
def _build_groups(gm, graph_ids, batch_size):
    """Fixed groups of Data objects (rebuilt into fresh Batches on every ablation forward).

    Returns the groups plus per-graph y / features / ids in the SAME order the predictions come
    back, so impacts align to graphs. Data objects are not mutated by the forward (only the Batch
    is), so caching them once is safe.
    """
    groups, ys, feats, order = [], [], [], []
    buf = []
    for gi in graph_ids:
        d = gm.eval_ds[int(gi)]
        buf.append(d)
        ys.append(d.y.reshape(-1).cpu().numpy().astype(np.float64))   # [T] (T=1 for ZINC)
        feats.append(graph_features(d))
        order.append(int(gi))
        if len(buf) == batch_size:
            groups.append(buf); buf = []
    if buf:
        groups.append(buf)
    return groups, np.stack(ys), feats, order   # ys: [G, T]


def run_ablation(result, sc, *, seed: int = 0, n_random_pairs: int = 300) -> dict:
    """Full ablation analysis for one loaded model (reuses ``result['gm']`` from score_model)."""
    import torch

    gm = result["gm"]
    L, H = gm.L, gm.H
    S_sem, S_str = result["S_sem"], result["S_str"]
    rng = np.random.default_rng(seed)

    n_abl = min(sc.ablation_graphs, len(gm.eval_ds))
    graph_ids = np.sort(rng.choice(len(gm.eval_ds), size=n_abl, replace=False))
    groups, ys, feats, order = _build_groups(gm, graph_ids, batch_size=64)
    G = len(order)
    log(f"[ablate] {G} eval graphs; sweeping all {L*H} single heads + random pairs.")

    import torch
    from ..carriage import metrics

    def _loss_np(pred, y):    # per-graph task loss [G] (l1 / mse / BCE), task-general over T
        return metrics.per_graph_loss(torch.as_tensor(pred), torch.as_tensor(y),
                                      gm.loss_fun).cpu().numpy()

    clean = gm.collect_preds_ablated(groups, None).reshape(G, -1)            # [G, T]
    Lc = _loss_np(clean, ys)                                                 # [G] clean per-graph loss

    # Per-head impact over all L*H heads (this IS the random-head null distribution).
    #   functional = ||pred_ablate - pred_clean||_2 over the T outputs (label-free; |.| for T=1)
    #   loss       = task-loss(pred_ablate, y) - task-loss(pred_clean, y)  (signed; >0 = ablation hurt)
    func_impact = np.zeros((L, H, G))
    loss_impact = np.zeros((L, H, G))
    for l in range(L):
        for h in range(H):
            pa = gm.collect_preds_ablated(groups, [(l, h)]).reshape(G, -1)   # [G, T]
            func_impact[l, h] = np.linalg.norm(pa - clean, axis=1)
            loss_impact[l, h] = _loss_np(pa, ys) - Lc
    func_mean = func_impact.mean(axis=2)    # [L,H] mean functional impact per head
    loss_mean = loss_impact.mean(axis=2)    # [L,H] mean loss impact per head

    # ---- pick the interesting heads (shared selection) ----
    from .scores import select_heads
    heads_of_interest = select_heads(S_sem, S_str)
    top_sem = heads_of_interest["top_semantic"]
    top_str = heads_of_interest["top_structural"]
    top_joint = heads_of_interest["top_joint"]

    # ---- significance of each target vs the single-head null ----
    flat_func = func_mean.reshape(-1)
    def _rank_stats(head):
        val = func_mean[head]
        others = np.array([func_mean[l, h] for l in range(L) for h in range(H) if (l, h) != head])
        rank = 1 + int((others > val).sum())              # 1 = highest-impact head
        pctile = float((others < val).mean() * 100.0)
        z = float((val - others.mean()) / (others.std() + 1e-12))
        ratio = float(val / (others.mean() + 1e-12))
        # per-graph paired sign test vs the MEDIAN head's per-graph impact
        med_head = np.median(func_impact.reshape(L * H, G), axis=0)
        wins = int((func_impact[head] > med_head).sum())
        return {"head": [int(head[0]), int(head[1])], "func_mean": float(val),
                "loss_mean": float(loss_mean[head]), "rank": rank, "n_heads": L * H,
                "percentile": pctile, "z": z, "ratio_vs_random": ratio,
                "per_graph_wins_vs_median": wins, "per_graph_n": G}
    target_stats = {name: _rank_stats(hd) for name, hd in heads_of_interest.items()}

    # ---- joint (pair) ablation: top_sem + top_str together vs random pairs ----
    def _distinct_pair(a, b):
        """Two DISTINCT heads (a, b). If a==b (top-sem and top-str coincide), substitute the
        highest-scoring structural head != a, so the target stays a 2-head ablation comparable
        to the 2-head random-pair null (else the percentile is understated)."""
        if a != b:
            return [a, b]
        for idx in np.argsort(S_str, axis=None)[::-1]:
            h = tuple(int(x) for x in np.unravel_index(idx, S_str.shape))
            if h != a:
                return [a, h]
        return [a]
    pair_targets = {
        "sem+str": _distinct_pair(top_sem, top_str),
        "joint+layermate": _distinct_pair(top_joint, (top_joint[0], (top_joint[1] + 1) % H)),
    }
    def _pair_impact(pair):
        pa = gm.collect_preds_ablated(groups, list(pair)).reshape(G, -1)     # [G, T]
        return float(np.linalg.norm(pa - clean, axis=1).mean())
    all_pairs = [(l, h) for l in range(L) for h in range(H)]
    rand_pair_impacts = []
    for _ in range(n_random_pairs):
        i, j = rng.choice(len(all_pairs), size=2, replace=False)
        rand_pair_impacts.append(_pair_impact([all_pairs[i], all_pairs[j]]))
    rand_pair_impacts = np.array(rand_pair_impacts)
    pair_stats = {}
    for name, pair in pair_targets.items():
        val = _pair_impact(pair)
        pair_stats[name] = {"heads": [list(p) for p in pair], "func_mean": val,
                            "rand_pair_mean": float(rand_pair_impacts.mean()),
                            "percentile": float((rand_pair_impacts < val).mean() * 100.0),
                            "p_ge": float((rand_pair_impacts >= val).mean())}

    # ---- score-predicts-impact validity check ----
    score_impact_corr = {
        "sem_score_vs_impact": _spearman(S_sem.reshape(-1), flat_func),
        "str_score_vs_impact": _spearman(S_str.reshape(-1), flat_func),
    }

    # ---- per-graph impact vs graph features (the "structural head on structural inputs" test) ----
    feat_names = list(feats[0].keys())
    feat_mat = {k: np.array([f[k] for f in feats]) for k in feat_names}
    feature_corr = {}
    for name, hd in heads_of_interest.items():
        feature_corr[name] = {k: _spearman(func_impact[hd], feat_mat[k]) for k in feat_names}

    log("[ablate] top heads (functional impact, rank / L*H, ratio vs random):")
    for name, st in target_stats.items():
        log(f"   {name:14s} L{st['head'][0]}H{st['head'][1]}  impact={st['func_mean']:.3e}  "
            f"rank={st['rank']}/{st['n_heads']}  z={st['z']:+.2f}  x{st['ratio_vs_random']:.1f}")
    log(f"[ablate] score->impact Spearman: sem={score_impact_corr['sem_score_vs_impact']:.2f} "
        f"str={score_impact_corr['str_score_vs_impact']:.2f}")

    return {
        "func_mean": func_mean, "loss_mean": loss_mean,
        "func_impact_per_graph": func_impact, "loss_impact_per_graph": loss_impact,
        "clean_pred": clean, "y": ys, "graph_ids": np.array(order),
        "features": feat_mat, "feat_names": feat_names,
        "heads_of_interest": {k: [int(v[0]), int(v[1])] for k, v in heads_of_interest.items()},
        "target_stats": target_stats, "pair_stats": pair_stats,
        "rand_pair_impacts": rand_pair_impacts,
        "score_impact_corr": score_impact_corr, "feature_corr": feature_corr,
    }
