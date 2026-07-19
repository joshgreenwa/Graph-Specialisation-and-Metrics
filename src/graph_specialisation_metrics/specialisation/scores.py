"""Per-head semantic / structural specialisation scores on the real GRIT model.

Lifts ``spec_head_scores.score_model`` (the spec-lite ``Net``) onto GRIT, reading at the
per-head TRANSPORT site  o^{lh}_i = ``batch.wV``  [n, H, dh]  (``model.GritHeadModel``):

  METHOD A -- SEPARATE INTERVENTION, read as per-head carriage (SPECIALISATION_SCORES.md):
    phi^{lh}_i = d yhat / d o^{lh}_i                          (per-head readout grad, clean input)
    F^{lh}[i,s] = | phi^{lh}_i . Dbar-o^{lh}_i(s) |           (donor/partner-averaged transport delta)
    S_sem(l,h)  = mean_{graph, j}  sum_i F   under the SEMANTIC donor swap (carriage.content)
    S_str(l,h)  = mean_{graph, u}  sum_i F   under the STRUCTURAL transposition (RRWP-feature channel,
                  ``_perturb_mask_frozen``): conjugate the RRWP payload (rrwp/rrwp_index/val/deg/
                  log_deg) but FREEZE the attention mask (edge_index/edge_attr). SPECIALISATION_SCORES.md
                  requires the k-hop mask be held fixed; conjugating it (as a full relabel would) would
                  confound a sparse head's wiring reliance with its structural payload reliance and
                  inflate S_str for the 1-hop control. For the dense model the mask is all-pairs, so
                  freezing is a no-op on the support.

  Complementary SELECTION-site (attention-routing) score, SEMANTIC only (a content swap keeps the
  edge set fixed, so the per-edge attention slots stay aligned across replicas; a structural
  transposition relabels edges, so the selection score is not slot-comparable and is omitted):
    S_attn_sem(l,h) = mean_{graph, j}  sum_e | Dbar-a^{lh}_e(j) |    (donor-avg the attention delta,
                      then abs -- Jensen at the softmax, mirroring the transport recipe).

Both interventions feed the IDENTICAL estimator; the transport delta uses a WITHIN-BATCH clean
baseline (replica 0 of every chunk) exactly as ``carriage.grit_runner``, so a no-op donor / self
transposition gives ~0 and the batch-context float32 offset cancels.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..carriage import structural
from ..carriage.env import log
from .model import GritHeadModel, SpecConfig


def _perturb_mask_frozen(base, u: int, v: int):
    """Spec-faithful structural transposition: conjugate the RRWP-derived structural PAYLOAD
    (rrwp, rrwp_index/rrwp_val, deg, log_deg) by P_(u v) while FREEZING the attention mask / wiring.

    SPECIALISATION_SCORES.md is explicit: the k-hop attention MASK is architecture and must be
    held FIXED under the structural intervention -- "conjugate only the RRWP features". In GRIT the
    masked (1-hop) edge encoder derives its support from ``edge_index``, so ``carriage.structural``'s
    full transposition (which relabels ``edge_index``) would CONJUGATE the mask and confound a
    sparse head's *wiring* reliance with its structural *payload* reliance, inflating S_str for the
    1-hop control task-independently. We therefore relabel the RRWP payload but RESTORE
    ``edge_index``/``edge_attr`` to the original (frozen wiring). For the dense model the support is
    the complete graph, so freezing is a no-op on the mask; the difference is only that the bond
    channel is held fixed, matching the spec (bonds are wiring, not RRWP payload).
    """
    pert = structural.perturb(base, u, v, "transposition")   # conjugates rrwp/rrwp_index/val/deg/log_deg + edge_index
    pert.edge_index = base.edge_index                         # freeze the mask/wiring
    if getattr(base, "edge_attr", None) is not None:
        pert.edge_attr = base.edge_attr                      # bonds unchanged (perturb left values intact); realign
    return pert


def _funcmag_contrib(phi_stack_l, dob):
    """Per-head functional-carriage contribution summed over sources & carriers.

    phi_stack_l: [T, n, H, dh] readout gradients (one per output t) at layer l.
    dob:         [S, n, H, dh] donor/partner-averaged transport delta (per source-row).
    Returns [H]: sum_{source, carrier} sqrt( sum_t (phi_t^{lh}_i . Dbar-o^{lh}_i)^2 ). For T=1
    this is sum_{source, carrier} |phi . Dbar-o| (the ZINC scalar case), matching Method A.
    """
    import torch

    c = torch.einsum("tnhd,snhd->tsnh", phi_stack_l, dob)   # [T, S, n, H]
    F = (c * c).sum(0).sqrt()                               # [S, n, H] magnitude over outputs
    return F.sum(dim=(0, 1))                                # [H]


def select_heads(S_sem, S_str) -> dict:
    """The interesting heads, shared by ablation / attention-viz / figures.

    top_semantic / top_structural = argmax of that channel; top_joint = high in BOTH (max of the
    summed per-channel z-scores); low_both = inert on both (a natural ablation control).
    """
    def _amax(M):
        return tuple(int(x) for x in np.unravel_index(np.argmax(np.asarray(M)), np.asarray(M).shape))

    def _z(M):
        M = np.asarray(M, float)
        return (M - M.mean()) / (M.std() + 1e-12)

    joint = _z(S_sem) + _z(S_str)
    return {
        "top_semantic": _amax(S_sem),
        "top_structural": _amax(S_str),
        "top_joint": _amax(joint),
        "low_both": _amax(-joint),
    }


def _plan_chunk(n: int, max_replicas: int = 2048, max_pair_edges: int = 8_000_000) -> int:
    """Replicas per forward: bounded by a total edge budget (dense support ~ n^2 per replica)."""
    per = max(n * n, 1)
    return max(1, min(max_replicas, max_pair_edges // per))


def _build_donor_pool(model: GritHeadModel):
    """Real content rows from the donor split (Def 3.2.2) + the graph id each row came from."""
    rows, gids = [], []
    for gi in range(len(model.donor_ds)):
        r = model.adapter.rows(model.donor_ds[gi])            # [n, F]
        rows.append(r)
        gids.append(np.full(r.shape[0], gi, dtype=np.int64))
    donor_rows = np.concatenate(rows).astype(np.int64)         # [Npool, F]
    donor_gids = np.concatenate(gids)
    return donor_rows, donor_gids


def score_model(task, sc: SpecConfig, *, with_attn_routing: bool = True,
                max_sources: int | None = None, seed: int = 0) -> dict:
    """Return per-head {S_sem, S_str, S_attn_sem} [L,H] for one loaded GRIT checkpoint + diagnostics."""
    import torch
    from torch_geometric.data import Batch

    gm = GritHeadModel(task, sc).load()
    model, device = gm.model, gm.device
    L, H, dh = gm.L, gm.H, gm.dh
    adapter = gm.adapter
    rng = np.random.default_rng(seed)

    donor_rows, donor_gids = _build_donor_pool(gm)
    F_feat = donor_rows.shape[1]
    log(f"[donors] pool={donor_rows.shape[0]} rows over {len(gm.donor_ds)} graphs (F={F_feat}).")

    n_graphs = min(sc.num_graphs, len(gm.eval_ds))
    graph_ids = np.sort(rng.choice(len(gm.eval_ds), size=n_graphs, replace=False))
    K = int(sc.donors)
    log(f"[select] {n_graphs} {sc.eval_split} graphs, K={K} donors/partners, "
        f"max_sources={max_sources}.")

    S_sem = np.zeros((L, H)); S_str = np.zeros((L, H)); S_attn = np.zeros((L, H))
    tot_sem_sources = tot_str_anchors = tot_attn_sources = 0
    noop_max = relabel_inv_max = softmax_err = 0.0
    mask_frozen_ok = False        # did the mask-freeze actually neutralise an edge_index relabel?
    T = 1                         # #outputs (set from the first graph; 1 for ZINC scalar regression)
    t0 = time.perf_counter()

    for gi_pos, gi in enumerate(graph_ids):
        base = gm.eval_ds[int(gi)]
        n = int(base.num_nodes)
        if n < 2:
            continue

        # ---- clean forward (batch-of-1, grad): phi_t^{lh}_i = d yhat_t / d wV_l ---------
        # Task-general: for T>1 outputs (multi-target regression / multilabel classification)
        # we keep one readout gradient per output and combine as the functional MAGNITUDE
        # F = sqrt(sum_t (phi_t . Dbar-o)^2), exactly carriage.core.functional_magnitude. For a
        # scalar target (T=1, ZINC) this reduces to |phi . Dbar-o|.
        cb = Batch.from_data_list([base]).to(device)
        cap = gm.capture(cb, want_grad=True, want_attn=with_attn_routing)
        pred_c = cap["pred"].reshape(-1)                            # [T]
        T = int(pred_c.numel())
        phi_stack = []                                             # [L] of [T, n, H, dh]
        grads_t = [torch.autograd.grad(pred_c[t], cap["wV"], retain_graph=(t < T - 1))
                   for t in range(T)]                              # grads_t[t] = [L] of [n,H,dh]
        for l in range(L):
            phi_stack.append(torch.stack([grads_t[t][l].detach() for t in range(T)]))  # [T,n,H,dh]
        del grads_t

        # Effective per-graph source cap. The score is a POOLED mean over measured (graph, source)
        # pairs (denominator = total sources), so subsampling sources on large-graph tasks (e.g.
        # peptides, n up to ~450) is an unbiased estimate of that pooled mean. NB: when capping is
        # active, large graphs contribute cap_S (not n) sources, so their per-source weight is the
        # same as a small graph's -- a mild reweighting vs the uncapped pooled mean, never a bias
        # in the estimator. Bounds the transport accumulator dObar[L] = [S,n,H,dh]*4B to ~2 GiB;
        # an explicit max_sources always wins. Logged, never silent.
        if max_sources is not None:
            cap_S = min(int(max_sources), n)
        else:
            per_source_bytes = n * H * dh * L * 4 * 3           # dObar + working slack
            cap_S = min(n, max(4, int(2e9 // max(per_source_bytes, 1))))
            if cap_S < n and gi_pos == 0:
                log(f"[scale] n={n}: capping sources/graph to {cap_S} (memory budget); "
                    f"pass max_sources to override.")
        sources = list(range(n)) if cap_S >= n else \
            sorted(rng.choice(n, size=cap_S, replace=False).tolist())
        S = len(sources)

        # Per-graph attention-routing gate: dAbar = [S, E, H] with E = attention edges/replica
        # (n^2 for the dense full-attention support), which blows up on large graphs. The
        # transport score is unaffected; only the (secondary) selection-site score is skipped.
        graph_attn = with_attn_routing
        E = 0
        if with_attn_routing:
            E = int(cap["attn"][0].shape[0])
            if E * S * H * L * 4 * 3 > 2e9:
                graph_attn = False
                if gi_pos == 0:
                    log(f"[scale] n={n}: attention-routing disabled this graph (E={E} too large); "
                        f"transport scores unaffected.")

        # softmax sanity: attention into each dest node sums to 1 (first graphs only).
        if graph_attn and gi_pos < 2 and cap["edge_index"] is not None:
            dest = cap["edge_index"][1]
            for l in range(L):
                s = torch.zeros(n, H, device=device)
                s.index_add_(0, dest, cap["attn"][l])
                softmax_err = max(softmax_err, float((s - 1.0).abs().max().item()))

        # ============================ SEMANTIC donor swap ============================
        pool = np.flatnonzero(donor_gids != int(gi)) if sc.donor_split == sc.eval_split \
            else np.arange(donor_rows.shape[0])
        donor_ids = rng.choice(pool, size=(S, K), replace=True)   # [S, K]
        donor_content = donor_rows[donor_ids]                     # [S, K, F]
        own_rows = adapter.rows(base)                             # [n, F]
        noop = (donor_content == own_rows[sources][:, None, :]).all(axis=-1)   # [S, K]

        R = S * K
        src_of_rep = np.repeat(np.arange(S), K)                  # replica r -> source-row (0..S-1)
        node_of_rep = np.array(sources, dtype=np.int64)[src_of_rep]   # replica r -> node j
        flat_donor = donor_content.reshape(R, F_feat)

        dObar = [torch.zeros(S, n, H, dh, device=device) for _ in range(L)]     # per source-row
        dAbar = ([torch.zeros(S, E, H, device=device) for _ in range(L)]
                 if graph_attn else None)

        chunk = _plan_chunk(n)
        r0 = 0
        while r0 < R:
            m = min(chunk, R - r0)
            while True:
                try:
                    with torch.no_grad():
                        b = Batch.from_data_list([base] * (m + 1)).to(device)
                        rows = (torch.arange(1, m + 1, device=device) * n
                                + torch.as_tensor(node_of_rep[r0:r0 + m], device=device))
                        adapter.write_donors(b.x, rows, flat_donor[r0:r0 + m])
                        cc = gm.capture(b, want_grad=False, want_attn=graph_attn)
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or m == 1:
                        raise
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    m = max(1, m // 2); chunk = m
                    log(f"[mem] OOM -> reducing replicas/forward to {m}")
            assert cc["wV"][0].shape[0] == (m + 1) * n, "semantic replica reshape mismatch"
            reps_src = torch.as_tensor(src_of_rep[r0:r0 + m], device=device)
            for l in range(L):
                wv = cc["wV"][l].view(m + 1, n, H, dh)
                delta = wv[0:1] - wv[1:]                          # [m, n, H, dh]
                dObar[l].index_add_(0, reps_src, delta)
                if graph_attn:
                    assert cc["attn"][l].shape[0] == (m + 1) * E, "attention replica reshape mismatch"
                    at = cc["attn"][l].view(m + 1, E, H)
                    dAbar[l].index_add_(0, reps_src, at[0:1] - at[1:])
            # no-op donor check: same-content swap -> ~0 transport under within-batch baseline.
            nm = noop.reshape(-1)[r0:r0 + m]
            if nm.any() and gi_pos < 3:
                sel = torch.as_tensor(np.flatnonzero(nm), device=device)
                for l in range(L):
                    wv = cc["wV"][l].view(m + 1, n, H, dh)
                    noop_max = max(noop_max, float((wv[0:1] - wv[1 + sel]).abs().max().item()))
            r0 += m

        for l in range(L):
            S_sem[l] += _funcmag_contrib(phi_stack[l], dObar[l] / K).cpu().numpy()
            if graph_attn:
                da = (dAbar[l] / K).abs().sum(dim=1)             # [S, H]  sum_e |Dbar-a|
                S_attn[l] += da.sum(0).cpu().numpy()
        tot_sem_sources += S
        if graph_attn:
            tot_attn_sources += S

        # ============================ STRUCTURAL transposition ============================
        # completeness check: a FULL relabel (structure+content) is an isomorphism -> pred invariant
        # (validates that carriage.structural.perturb captures every structure-derived channel).
        if gi_pos < 2:
            with torch.no_grad():
                rb = Batch.from_data_list([structural.full_relabel(base, 0, 1)]).to(device)
                pr, _ = model(rb)
                relabel_inv_max = max(relabel_inv_max,
                                      float((pr.view(-1) - pred_c.detach()).abs().max().item()))
            # mask-frozen invariant: the structural intervention must leave edge_index (the k-hop
            # mask / wiring) UNCHANGED. The raw transposition WOULD relabel it (that is exactly the
            # mask-conjugation we neutralise); confirm both here on the first graphs.
            mf = _perturb_mask_frozen(base, 0, 1)
            raw = structural.perturb(base, 0, 1, "transposition")
            assert torch.equal(mf.edge_index, base.edge_index), \
                "structural intervention changed the attention mask (edge_index must be frozen)"
            if not torch.equal(raw.edge_index, base.edge_index):
                mask_frozen_ok = True   # the freeze is doing real work on this graph

        deg = structural.node_degrees(base.edge_index, n)
        partners = np.stack([structural.sample_partners(deg, u, K, rng, sc.partner_match)
                             for u in sources])                  # [S, K]
        R = S * K
        anchor_of_rep = np.repeat(np.arange(S), K)               # replica -> anchor-row
        partners_flat = partners.reshape(-1)                     # replica -> partner node v
        anchor_nodes_flat = np.array(sources, dtype=np.int64)[anchor_of_rep]   # replica -> anchor node u
        dObar = [torch.zeros(S, n, H, dh, device=device) for _ in range(L)]

        chunk = _plan_chunk(n)
        r0 = 0
        while r0 < R:
            m = min(chunk, R - r0)
            while True:
                try:
                    perts = [_perturb_mask_frozen(base, int(anchor_nodes_flat[r]),
                                                  int(partners_flat[r]))
                             for r in range(r0, r0 + m)]
                    with torch.no_grad():
                        b = Batch.from_data_list([base] + perts).to(device)
                        cc = gm.capture(b, want_grad=False, want_attn=False)
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or m == 1:
                        raise
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    m = max(1, m // 2); chunk = m
                    log(f"[mem] OOM (structural) -> reducing replicas/forward to {m}")
            assert cc["wV"][0].shape[0] == (m + 1) * n, "structural replica reshape mismatch"
            reps_anc = torch.as_tensor(anchor_of_rep[r0:r0 + m], device=device)
            for l in range(L):
                wv = cc["wV"][l].view(m + 1, n, H, dh)
                dObar[l].index_add_(0, reps_anc, wv[0:1] - wv[1:])
            # no-op partner (v==u) check: a self-transposition must move ~0 transport.
            nm = partners_flat[r0:r0 + m] == anchor_nodes_flat[r0:r0 + m]
            if nm.any() and gi_pos < 3:
                sel = torch.as_tensor(np.flatnonzero(nm), device=device)
                for l in range(L):
                    wv = cc["wV"][l].view(m + 1, n, H, dh)
                    noop_max = max(noop_max, float((wv[0:1] - wv[1 + sel]).abs().max().item()))
            r0 += m

        for l in range(L):
            S_str[l] += _funcmag_contrib(phi_stack[l], dObar[l] / K).cpu().numpy()
        tot_str_anchors += S

        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (gi_pos + 1) % max(1, n_graphs // 10) == 0 or gi_pos == 0:
            log(f"[run] graph {gi_pos+1}/{n_graphs} (id={int(gi)}, n={n}) | "
                f"{time.perf_counter()-t0:.1f}s")

    S_sem /= max(tot_sem_sources, 1)
    S_str /= max(tot_str_anchors, 1)
    attn_measured = with_attn_routing and tot_attn_sources > 0
    if attn_measured:
        S_attn /= tot_attn_sources

    log("\n" + "=" * 72 + "\nSPECIALISATION-SCORE VERIFICATION\n" + "=" * 72)
    log(f"  test {gm.checks['test_metric_name']:>4} from ckpt : {gm.test_metric}")
    log(f"  [softmax] max|sum_j a_ij - 1|   : {softmax_err:.3e}  (must be ~0)")
    log(f"  [no-op]   max|dwV| same-content : {noop_max:.3e}  (within-batch => ~0; "
        f"floor < {sc.float_noise_tol:.0e})")
    log(f"  [relabel] full-relabel |dpred|  : {relabel_inv_max:.3e}  (isomorphism => ~0)")
    log(f"  [mask]    frozen under structural: edge_index unchanged; raw relabel neutralised="
        f"{mask_frozen_ok} (spec: k-hop mask is architecture, held FIXED)")
    log(f"  S_sem mean/max = {S_sem.mean():.3e}/{S_sem.max():.3e} | "
        f"S_str mean/max = {S_str.mean():.3e}/{S_str.max():.3e}")
    noise_tol = max(sc.tol, sc.float_noise_tol)
    if noop_max > noise_tol:
        raise RuntimeError(f"No-op transport |dwV|={noop_max:.3e} > {noise_tol:.1e}: a same-content "
                           f"swap / self-transposition must be ~0 under the within-batch baseline. "
                           f"Wrong row written or model not in eval().")
    if relabel_inv_max > noise_tol:
        raise RuntimeError(f"Full-relabel |dpred|={relabel_inv_max:.3e} > {noise_tol:.1e}: a "
                           f"structure+content relabel is an isomorphism, so pred must be invariant. "
                           f"The transposition is missing a structure-derived channel.")

    return {
        "task": task.name, "title": task.title,
        "S_sem": S_sem, "S_str": S_str, "S_attn_sem": S_attn if attn_measured else None,
        "L": L, "H": H, "dh": dh, "n_heads": H, "n_layers": L, "T": T,
        "test_metric": gm.test_metric, "test_metric_name": gm.checks["test_metric_name"],
        "num_graphs": int(len(graph_ids)), "donors_K": K,
        "checks": {"softmax_err": softmax_err, "noop_max": noop_max,
                   "relabel_inv_max": relabel_inv_max, **gm.checks},
        "graph_ids": graph_ids,
        "gm": gm,     # the loaded model, reused by ablation / attention-viz (no reload)
    }
