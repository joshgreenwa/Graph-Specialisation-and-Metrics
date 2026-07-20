"""Per-head, channel-split *causal* ablation on the real GRIT model (swap x ablate).

The per-head scores ``S_sem`` / ``S_str`` (``scores.py``) are head-LOCAL first-order transport
magnitudes. Their causal counterpart -- and the coordinate the D/J plane is validated against --
is how much *ablating* a head changes the model's OUTPUT response to each intervention. The
methodology doc (SPECIALISATION_SCORES.md) is explicit that this causal side must be
ABLATION-based (the exact ``L(ablate) - L(clean)`` / swap x ablate 2x2), NOT the first-order loss
projection, which fails at the L1 kink on a well-fit model.

The synthetic double-dissociation gets its semantic/structural split from TWO SEPARATE TASKS
(semantic-task vs structural-task graphs). Real ZINC is single-task, so there is no such split;
instead we define the split via the two INTERVENTIONS, as the exact per-head interaction in a
2x2 {clean, intervened} x {intact, ablated} design at the pooled output:

    Delta_sem      = yhat(clean)      - mean_k yhat(semantic-swap_k)          # intact response
    Delta_sem^abl  = yhat_abl(clean)  - mean_k yhat_abl(semantic-swap_k)      # response w/ (l,h) zeroed
    I_sem^func(l,h)= mean_{graph,source} || Delta_sem - Delta_sem^abl ||_2     # >=0 functional
    I_sem^loss(l,h)= mean_{graph,source} [ (L_swap-L_clean) - (L_swap^abl-L_clean^abl) ]  # signed loss

and identically for the structural transposition (partner-marginalised) to give ``I_str``. The
interaction cancels the head's swap-independent baseline shift, isolating its causal role in
propagating THAT channel to the output. ``overall_func``/``overall_loss`` are the ordinary
single-channel impacts (``||yhat_clean - yhat_abl||`` / ``L_abl - L_clean``) for the influence panel.

Reuses the proven primitives -- ``model.collect_preds_ablated`` (arbitrary head ablation over
Data groups), ``scores._perturb_mask_frozen`` (mask-frozen structural replica), the content
adapter (semantic replica), and ``carriage.structural`` (degree-matched partners). The heavy loop
is exactly ``ablation.run_ablation``'s L*H passes, but over intervention replicas instead of
task-mode graphs. The aggregation is pure numpy (``_impacts_for_head``) so it is unit-testable
without a GPU; the model half is validated in Colab by the no-op self-check below.
"""

from __future__ import annotations

import time

import numpy as np

from ..carriage.env import log


# --------------------------------------------------------------------------------------
# pure-numpy pieces (unit-testable without torch/GRIT)
# --------------------------------------------------------------------------------------

def per_graph_loss_np(pred, y, loss_fun: str) -> np.ndarray:
    """Numpy mirror of ``carriage.metrics.per_graph_loss`` (mean over the T targets)."""
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    lf = str(loss_fun).lower()
    if lf in ("l1", "mae"):
        return np.abs(pred - y).mean(axis=-1)
    if lf in ("mse", "l2"):
        return ((pred - y) ** 2).mean(axis=-1)
    if lf in ("cross_entropy", "bce", "binary_cross_entropy"):
        z = pred  # BCEWithLogits, numerically stable: max(z,0) - z*y + log1p(exp(-|z|))
        return (np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))).mean(axis=-1)
    raise ValueError(f"unsupported loss_fun for channel ablation: {loss_fun!r}")


def _impacts_for_head(preds: np.ndarray, abl: np.ndarray, *, clean_g: np.ndarray,
                      y_g: np.ndarray, unit_g: np.ndarray, sem_idx: np.ndarray,
                      str_idx: np.ndarray, loss_fun: str) -> dict:
    """Six scalar impacts for ONE ablated head, from clean + ablated replica predictions.

    Args are pure numpy so this is fully testable:
      preds, abl : [N, T] pooled predictions of every replica, intact and with the head ablated.
      clean_g    : [G]      flat index into preds of each graph's clean replica.
      y_g        : [G, T]   per-graph target.
      unit_g     : [U]      graph index of each (graph, source) unit.
      sem_idx    : [U, K]   flat indices of the K semantic-swap replicas per unit.
      str_idx    : [U, K]   flat indices of the K structural-transposition replicas per unit.
    """
    pc, pca = preds[clean_g], abl[clean_g]                       # [G, T] clean intact / ablated
    overall_func = float(np.linalg.norm(pca - pc, axis=1).mean())
    lc = per_graph_loss_np(pc, y_g, loss_fun)                    # [G]
    lca = per_graph_loss_np(pca, y_g, loss_fun)
    overall_loss = float((lca - lc).mean())

    pc_u, pca_u = preds[clean_g[unit_g]], abl[clean_g[unit_g]]   # [U, T]
    y_u = y_g[unit_g]                                            # [U, T]
    lc_u = per_graph_loss_np(pc_u, y_u, loss_fun)
    lca_u = per_graph_loss_np(pca_u, y_u, loss_fun)

    def _channel(idx):
        ps = preds[idx].mean(axis=1)                            # [U, T] donor/partner-averaged intact
        psa = abl[idx].mean(axis=1)                             # ablated
        d = pc_u - ps                                           # intact response  [U, T]
        da = pca_u - psa                                        # ablated response [U, T]
        func = float(np.linalg.norm(d - da, axis=1).mean())    # >=0 interaction magnitude
        ls = per_graph_loss_np(ps, y_u, loss_fun) - lc_u       # L_swap - L_clean, intact
        lsa = per_graph_loss_np(psa, y_u, loss_fun) - lca_u    # ablated
        loss = float((ls - lsa).mean())                        # signed interaction
        return func, loss

    sf, sl = _channel(sem_idx)
    tf, tl = _channel(str_idx)
    return {"I_sem_func": sf, "I_str_func": tf, "I_sem_loss": sl, "I_str_loss": tl,
            "overall_func": overall_func, "overall_loss": overall_loss}


# --------------------------------------------------------------------------------------
# model half (torch/GRIT; validated in Colab by the no-op self-check)
# --------------------------------------------------------------------------------------

def _semantic_swap_data(base, source: int, donor_row: np.ndarray, adapter):
    """A copy of ``base`` with node ``source``'s content overwritten by a real donor row."""
    import torch
    d = base.clone()
    adapter.write_donors(d.x, torch.as_tensor([int(source)], device=d.x.device),
                         np.asarray(donor_row)[None, :])
    return d


def run_channel_ablation(gm, sc, *, num_graphs: int = 48, max_sources: int = 6,
                         donors: int = 3, seed: int = 0, batch_graphs: int = 64,
                         verify: bool = True) -> dict:
    """Per-head channel-split ablation for one loaded ``GritHeadModel`` (reuses ``result['gm']``).

    Returns [L, H] arrays ``I_sem_func``/``I_str_func`` (>=0), ``I_sem_loss``/``I_str_loss``
    (signed), and ``overall_func``/``overall_loss`` (single-channel), plus meta + checks.
    """
    from ..carriage import structural
    from .scores import _build_donor_pool, _perturb_mask_frozen

    L, H = gm.L, gm.H
    adapter = gm.adapter
    rng = np.random.default_rng(seed)
    donor_rows, donor_gids = _build_donor_pool(gm)
    n_graphs = min(int(num_graphs), len(gm.eval_ds))
    graph_ids = np.sort(rng.choice(len(gm.eval_ds), size=n_graphs, replace=False))
    K = int(donors)
    same_split = (sc.donor_split == sc.eval_split)
    log(f"[chan-abl] {n_graphs} graphs, <= {max_sources} sources/graph, K={K}; "
        f"sweeping all {L * H} heads under semantic + structural interventions.")

    datas: list = []                       # flat replica catalogue
    clean_g_idx: list = []                 # [G] flat index of each graph's clean replica
    y_g: list = []                         # [G, T] targets
    unit_g: list = []                      # [U] graph index per (graph, source) unit
    sem_rows: list = []                    # [U, K]
    str_rows: list = []                    # [U, K]
    verify_pairs: list = []                # (replica_idx, clean_idx) expected ~equal (no-op)

    def _add(d) -> int:
        datas.append(d)
        return len(datas) - 1

    for g, gi in enumerate(graph_ids):
        base = gm.eval_ds[int(gi)]
        n = int(base.num_nodes)
        if n < 2:
            continue
        ci = _add(base.clone())
        gpos = len(clean_g_idx)
        clean_g_idx.append(ci)
        y_g.append(base.y.reshape(-1).cpu().numpy().astype(np.float64))
        cap_S = min(int(max_sources), n)
        sources = (list(range(n)) if cap_S >= n
                   else sorted(rng.choice(n, size=cap_S, replace=False).tolist()))
        own = adapter.rows(base)                                  # [n, F]
        deg = structural.node_degrees(base.edge_index, n)
        pool = (np.flatnonzero(donor_gids != int(gi)) if same_split
                else np.arange(donor_rows.shape[0]))
        for s in sources:
            donor_ids = rng.choice(pool, size=K, replace=True)
            sem_is = [_add(_semantic_swap_data(base, s, donor_rows[donor_ids[k]], adapter))
                      for k in range(K)]
            partners = structural.sample_partners(deg, s, K, rng, sc.partner_match)
            str_is = [_add(_perturb_mask_frozen(base, int(s), int(partners[k])))
                      for k in range(K)]
            unit_g.append(gpos)
            sem_rows.append(sem_is)
            str_rows.append(str_is)
        if verify and g < 2:                                     # no-op replicas: must match clean
            verify_pairs.append((_add(_semantic_swap_data(base, sources[0],
                                                          own[sources[0]], adapter)), ci))
            verify_pairs.append((_add(_perturb_mask_frozen(base, int(sources[0]),
                                                          int(sources[0]))), ci))

    if not unit_g:
        raise RuntimeError("channel ablation: no eligible graphs (all had < 2 nodes).")

    groups = [datas[i:i + batch_graphs] for i in range(0, len(datas), batch_graphs)]
    clean_g = np.asarray(clean_g_idx)
    y_g = np.stack(y_g)                                          # [G, T]
    unit_g = np.asarray(unit_g)
    sem_idx = np.asarray(sem_rows)                               # [U, K]
    str_idx = np.asarray(str_rows)                               # [U, K]

    t0 = time.perf_counter()
    clean_preds = gm.collect_preds_ablated(groups, None)         # [N, T]
    log(f"[chan-abl] clean pass over {len(datas)} replicas [{time.perf_counter()-t0:.1f}s]")

    # no-op self-check: a same-content swap / self-transposition must not move the output.
    noop_max = 0.0
    for ri, ci in verify_pairs:
        noop_max = max(noop_max, float(np.abs(clean_preds[ri] - clean_preds[ci]).max()))
    if verify_pairs:
        log(f"[chan-abl] [no-op] max|yhat(noop) - yhat(clean)| = {noop_max:.3e} (must be ~0)")
        tol = max(getattr(sc, "tol", 1e-4), getattr(sc, "float_noise_tol", 5e-3))
        if noop_max > tol:
            raise RuntimeError(
                f"channel ablation no-op check failed: |dyhat|={noop_max:.3e} > {tol:.1e}. A "
                f"same-content swap / self-transposition must leave the output invariant.")

    out = {k: np.zeros((L, H)) for k in
           ("I_sem_func", "I_str_func", "I_sem_loss", "I_str_loss", "overall_func", "overall_loss")}
    for l in range(L):
        for h in range(H):
            abl = gm.collect_preds_ablated(groups, [(l, h)])     # [N, T]
            v = _impacts_for_head(clean_preds, abl, clean_g=clean_g, y_g=y_g, unit_g=unit_g,
                                  sem_idx=sem_idx, str_idx=str_idx, loss_fun=gm.loss_fun)
            for k in out:
                out[k][l, h] = v[k]
        log(f"[chan-abl] layer {l + 1}/{L} swept [{time.perf_counter()-t0:.1f}s]")

    out.update({
        "graph_ids": graph_ids, "num_graphs": int(len(clean_g)), "n_units": int(len(unit_g)),
        "donors_K": K, "max_sources": int(max_sources), "loss_fun": str(gm.loss_fun),
        "noop_max": float(noop_max),
    })
    log(f"[chan-abl] done: I_sem_func mean/max = {out['I_sem_func'].mean():.3e}/"
        f"{out['I_sem_func'].max():.3e} | I_str_func mean/max = {out['I_str_func'].mean():.3e}/"
        f"{out['I_str_func'].max():.3e}")
    return out
