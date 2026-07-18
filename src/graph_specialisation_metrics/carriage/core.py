"""Task-agnostic carriage estimators and distance aggregation.

Pure NumPy/PyTorch, no GRIT/GraphGym dependency, so it is unit-testable without a GPU
or a trained model. Everything model- or task-specific lives in ``grit_runner`` and
``tasks``; this module only knows about captured states, gradients, and per-pair arrays.

Implements the dissertation methodology (Ch. 3, Sections 3.2-3.3):

  semantic carriage      C_swap[i,j] = (1/K) sum_k g_i^T [h^L_i(clean) - h^L_i(swap_k)]   (3.4/3.5)
  functional carriage    F[i,j] = |C[i,j]|,  F(d) = mean_{d(i,j)=d} |C[i,j]|             (3.6/3.7)
  beneficial carriage    exact per-source loss change, carrier-attributed (see grit_runner)
  B_far(k)               sum over {(i,j): d(i,j) > k} of B[i,j]                            [MAE units]
"""

from __future__ import annotations

import numpy as np


def carriage_from_states(h_clean, h_swap, g, num_sources: int, num_donors: int):
    """Donor-swap semantic carriage estimator, dissertation Eq. 3.4 / 3.5.

        C_swap[i, j] = (1/K) sum_k  g_i^T [ h^L_i(X, S) - h^L_i(X_{j->xt^(k)}, S) ]

    Args:
        h_clean: [n, m]       h^L_i(X, S), the clean final-layer states.
        h_swap:  [S*K, n, m]  h^L_i under each intervention. Replica r = j*K + k,
                              matching Batch.from_data_list's contiguous stacking.
        g:       [n, m]       g_i = d yhat / d h^L_i, evaluated at the CLEAN input.
                              Eq. 3.5 does not index g by k, and Remark 3.3.1 takes the
                              benefit direction from the clean, correctly-labeled input.
        num_sources: S (one source per node, so S == n).
        num_donors:  K, the number of donors averaged over.

    Returns:
        [n, n] tensor C[i, j]: carrier i (rows) x source j (columns).

    Sign convention: dh is CLEAN minus SWAPPED, exactly as written in Eq. 3.4. The
    donor average is taken BEFORE any abs() (Eq. 3.6) or sign(), because both are
    defined as functions of the estimator C, not of the per-donor terms.
    """
    S, K = int(num_sources), int(num_donors)
    n, m = h_clean.shape
    if h_swap.shape != (S * K, n, m):
        raise ValueError(f"h_swap {tuple(h_swap.shape)} != {(S * K, n, m)}")
    if g.shape != (n, m):
        raise ValueError(f"g {tuple(g.shape)} != {(n, m)}")

    delta = h_clean.unsqueeze(0) - h_swap        # [R, n, m]  dh_i(j,k), Eq. 3.4
    return carriage_from_delta(delta, g, S, K)


def carriage_from_delta(delta, g, num_sources: int, num_donors: int):
    """Carriage from a precomputed transport delta dh_i(j,k) = h^L_i(clean) - h^L_i(swap).

    Same as carriage_from_states but takes the delta directly, so the caller can compute it
    with a WITHIN-BATCH clean baseline (clean and swap in the same forward pass), which
    cancels the batch-context float32 offset that a separate batch-of-1 clean would carry.

    Args:
        delta: [S*K, n, m] transport delta, replica r = j*K + k.
        g:     [n, m] readout gradient at the clean input.
    Returns:
        [n, n] tensor C[i, j] (carrier i x source j), donor-averaged (Eq. 3.5).
    """
    import torch

    S, K = int(num_sources), int(num_donors)
    c = torch.einsum("rnm,nm->rn", delta, g)     # [R, n]  g_i . dh_i(j,k)
    c = c.view(S, K, delta.shape[1])             # [source j, donor k, carrier i]
    C_js = c.mean(dim=1)                         # [j, i]  average over donors
    return C_js.t().contiguous()                 # [i, j]


def functional_magnitude(h_clean, h_swap, g_out, num_sources, num_donors):
    """Functional carriage magnitude F[i,j] = || C_out[i,j] ||_2 over the T outputs.

    Label-free (Def 3.3.2): how much does moving content from j change node i's effect on
    the model's OUTPUT? For a scalar output (T=1) this is |C[i,j]| exactly; for a vector
    output it is the L2 norm of the per-output carriage.

    Args:
        g_out: [T, n, m] Jacobian of the output w.r.t. h^L (one [n,m] slice per output t),
               evaluated at the clean input.

    Returns:
        [n, n] numpy: F[i, j] (carrier x source), non-negative.
    """
    delta = h_clean.unsqueeze(0) - h_swap
    return functional_magnitude_from_delta(delta, g_out, num_sources, num_donors)


def functional_magnitude_from_delta(delta, g_out, num_sources, num_donors):
    """Functional carriage magnitude from a precomputed transport delta (see above)."""
    T = int(g_out.shape[0])
    acc = None
    for t in range(T):
        Ct = carriage_from_delta(delta, g_out[t], num_sources, num_donors)  # [i,j]
        acc = Ct.pow(2) if acc is None else acc + Ct.pow(2)
    return acc.sqrt().cpu().numpy()  # [i, j], carrier x source


def beneficial_attribute(C_basis, dL_j, eps=1e-9):
    """Attribute the exact per-source loss change dL_j to carriers by their carriage share.

        B[i,j] = dL_j . ( C_basis[i,j] / sum_i C_basis[i,j] )   =>  sum_i B[i,j] = dL_j exactly.

    ``C_basis`` is the signed loss-carriage C_loss[i,j] = (dL/dh_i).dh_i(j): its column sum
    is the first-order loss change from source j, so it is the natural basis for splitting
    the (exact) dL_j across carriers. Sources that move nothing (sum_i C_basis ~ 0, hence
    dL_j ~ 0) get a zeroed column via the eps guard.

    Args:
        C_basis: [n, n] numpy, the signed loss-carriage (carrier x source).
        dL_j:    [n] numpy, exact per-source loss change L_clean - mean_k L_swap(j,k).
    """
    C_basis = np.asarray(C_basis)
    sumC_j = C_basis.sum(axis=0)                                       # [n]
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(np.abs(sumC_j) > eps, C_basis / sumC_j, 0.0)
    B = np.asarray(dL_j)[None, :] * share                             # sum_i B[i,j] = dL_j
    return B, sumC_j


def beneficial_from_carriage(C, yhat_clean, yhat_swap, y, num_sources, num_donors, eps=1e-9):
    """Exact per-source loss change, attributed to carriers by their carriage share.

    Replaces the linearized B = sign(yhat_clean - y) * C (Eq. 3.8) with a form that is
    exact across the |.| kink and uses the ACTUAL post-swap loss:

        L_clean = |yhat_clean - y|
        dL_j    = L_clean - mean_k |yhat_swap(j,k) - y|          per source j  [MAE units]
        B[i,j]  = dL_j . ( C[i,j] / sum_i C[i,j] )               carrier share
        => sum_i B[i,j] = dL_j   EXACTLY.

    Donor-average the LOSS, not the prediction: the kink makes E|.| != |E.| exactly when
    donor variance is large, so averaging yhat over donors and then taking |.| would
    misstate benefit near a residual sign crossing. Sources that move nothing
    (sum_i C[i,j] ~ 0, hence dL_j ~ 0) get a zeroed column via the eps guard.

    Args:
        C:          [n, n] numpy, C[i, j] (carrier x source), from carriage_from_states.
        yhat_clean: float, clean scalar prediction.
        yhat_swap:  torch tensor [S*K], per-replica scalar predictions (r = j*K + k).
        y:          float, clean target.
        num_sources / num_donors: S, K.

    Returns:
        (B, dL_j, sumC_j): B is [n, n] numpy (B[i,j], MAE units, B<0 beneficial);
        dL_j and sumC_j are [S] / [n] numpy for the exactness check.
    """
    S, K = int(num_sources), int(num_donors)
    L_clean = abs(float(yhat_clean) - float(y))
    L_swap = (yhat_swap.view(S, K) - float(y)).abs().mean(dim=1).cpu().numpy()  # [S]
    dL_j = L_clean - L_swap                                                     # [S]  <0 = beneficial
    sumC_j = np.asarray(C).sum(axis=0)                                          # [n]  sum_i C[i,j]
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(np.abs(sumC_j) > eps, np.asarray(C) / sumC_j, 0.0)     # C[i,j] / sum_i C[i,j]
    B = dL_j[None, :] * share                                                   # sum_i B[i,j] = dL_j
    return B, dL_j, sumC_j


def aggregate_carriage_curves(graph_id, distance, F, B, n_boot: int = 2000,
                              boot_seed: int = 1234) -> dict:
    """Distance profiles F(d), B(d), the per-distance MAE mass S(d), and B_far(k).

        F(d)     = mean over {(i,j) : d(i,j) = d} of |C[i,j]|            (Eq. 3.7)
        B(d)     = mean over {(i,j) : d(i,j) = d} of B[i,j]              (analogue of 3.7)
        S(d)     = per-graph SUM of B[i,j] at distance d, averaged over graphs [MAE units]
        B_far(k) = per-graph SUM of B[i,j] over d(i,j) > k, averaged over graphs [MAE units]

    F(d) and B(d) are means over PAIRS, exactly as the definitions state. S(d) and
    B_far(k) are SUMS taken per graph first, which keeps them in MAE units and makes them
    telescope: sum over d>k of S(d) is B_far(k).

    All CIs are 95% bootstrap intervals CLUSTERED ON GRAPHS: pairs inside one molecule are
    strongly dependent, so resampling pairs would understate the interval. For the pooled
    means the bootstrap resamples graphs and recomputes sum(sums)/sum(counts).

    Args:
        graph_id: [P] graph id per pair.
        distance: [P] integer hop distance per pair (finite only; d=inf dropped by caller).
        F:        [P] |C[i,j]|.
        B:        [P] beneficial carriage per pair.
    """
    graph_id = np.asarray(graph_id)
    distance = np.asarray(distance).astype(np.int64)
    F = np.asarray(F, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    dmax = int(distance.max())
    ds = np.arange(0, dmax + 1)
    uniq_g = np.unique(graph_id)
    n_g = int(uniq_g.size)
    gpos = {int(v): p for p, v in enumerate(uniq_g)}
    gidx = np.array([gpos[int(v)] for v in graph_id])

    def _by_graph(vals: np.ndarray, mask: np.ndarray):
        s = np.zeros(n_g)
        c = np.zeros(n_g)
        np.add.at(s, gidx[mask], vals[mask])
        np.add.at(c, gidx[mask], 1.0)
        return s, c

    def _boot_pooled(s, c):
        if c.sum() == 0:
            return float("nan"), float("nan"), float("nan")
        point = s.sum() / c.sum()
        br = np.random.default_rng(boot_seed)
        draws = br.integers(0, n_g, size=(n_boot, n_g))
        bs, bc = s[draws].sum(axis=1), c[draws].sum(axis=1)
        good = bc > 0
        if not np.any(good):
            return float(point), float("nan"), float("nan")
        vals = bs[good] / bc[good]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(point), float(lo), float(hi)

    def _boot_graph(per_graph):
        br = np.random.default_rng(boot_seed)
        draws = br.integers(0, per_graph.size, size=(n_boot, per_graph.size))
        vals = per_graph[draws].mean(axis=1)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(per_graph.mean()), float(lo), float(hi)

    counts = np.zeros(ds.size, dtype=np.int64)
    F_mean, F_lo, F_hi = (np.full(ds.size, np.nan) for _ in range(3))
    B_mean, B_lo, B_hi = (np.full(ds.size, np.nan) for _ in range(3))
    S_mean, S_lo, S_hi = (np.full(ds.size, np.nan) for _ in range(3))
    F_per_graph = np.full((n_g, ds.size), np.nan)

    for d in ds:
        m = distance == d
        counts[d] = int(m.sum())
        sF, cF = _by_graph(F, m)
        F_mean[d], F_lo[d], F_hi[d] = _boot_pooled(sF, cF)
        with np.errstate(invalid="ignore", divide="ignore"):
            F_per_graph[:, d] = np.where(cF > 0, sF / np.maximum(cF, 1), np.nan)
        sB, cB = _by_graph(B, m)
        B_mean[d], B_lo[d], B_hi[d] = _boot_pooled(sB, cB)
        S_mean[d], S_lo[d], S_hi[d] = _boot_graph(sB)

    ks = np.arange(0, dmax + 1)
    Bf_mean, Bf_lo, Bf_hi = (np.full(ks.size, np.nan) for _ in range(3))
    for k in ks:
        sB, _ = _by_graph(B, distance > k)   # per-graph SUM over d(i,j) > k
        Bf_mean[k], Bf_lo[k], Bf_hi[k] = _boot_graph(sB)

    return {
        "distances": ds, "pair_counts": counts, "n_graphs": n_g,
        "F_mean": F_mean, "F_lo": F_lo, "F_hi": F_hi,
        "B_mean": B_mean, "B_lo": B_lo, "B_hi": B_hi,
        "S_mean": S_mean, "S_lo": S_lo, "S_hi": S_hi,
        "k": ks, "B_far_mean": Bf_mean, "B_far_lo": Bf_lo, "B_far_hi": Bf_hi,
        "F_per_graph": F_per_graph,
    }


def symlog_linthresh(values, override: float = 0.0, floor: float = 1e-9) -> float:
    """Pick the symlog linear-region width for a signed distance curve.

    The d=0 self-pair term (a node's own content acting on its own state) is ~100x the
    cross-node transport terms, so a linear axis flattens every d>=1 point onto zero. A
    symlog axis with a small linear region shows the d=0 spike AND the tail together while
    preserving sign. We set the threshold to the median magnitude of the curve, which puts
    the bulk of the tail into the (spread-out) log region. ``override`` (>0) wins.
    """
    if override and override > 0:
        return float(override)
    a = np.abs(np.asarray(values, dtype=float))
    a = a[np.isfinite(a) & (a > 0)]
    if a.size == 0:
        return floor
    return float(max(floor, np.median(a)))
