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


def beneficial_attribute(C_basis, dL_j, eps=1e-8, denom="slope", slope_clip=1.0):
    """Turn the signed loss-carriage into per-pair beneficial carriage B[i,j].

    ``C_basis`` is the signed first-order loss-carriage C_loss[i,j] = (dL/dh_i).dh_i(j); summed
    over carriers it is the FIRST-ORDER loss change  sum_i C_loss[i,j] = (dL/dyhat)|_clean . dyhat_j.
    ``dL_j`` is the EXACT per-source loss change  L_clean - mean_k L_swap(j,k)  (<0 = beneficial).

    Modes (default "slope"):

      denom="slope" (DEFAULT -- signed AND stable):
          s_j    = dL_j / sum_i C_loss[i,j]                          exact / first-order loss change
          B[i,j] = clip(s_j, -slope_clip, +slope_clip) . C_loss[i,j]
        Keeps the per-carrier SIGN of C_loss, so adverse (B>0) stays measurable; and is bounded
        -- for a 1-Lipschitz loss (L1/BCE) |s_j| <= 1 by the reverse triangle inequality, so
        |B[i,j]| <= |C_loss[i,j]|: beneficial can never exceed functional and the signed-share
        far-tail blow-up is impossible. The clip only activates on estimation noise (first-order
        change ~ 0). "No delta => no carriage": C_loss[i,j]=0 -> B=0, and a source that moves
        nothing (sum_i C_loss ~ 0, hence dL_j ~ 0) gets s_j=0 -> a zeroed column. slope_clip is
        the loss's max |dL/dyhat| (1 for L1/BCE); it is NOT valid for MSE (unbounded slope).

      denom="magnitude" (bounded but sign-collapsing):
          B[i,j] = dL_j * |C_loss[i,j]| / sum_i |C_loss[i,j]|
        Convex weights, |B| <= |dL_j|, sum_i B = dL_j exactly -- but every carrier inherits
        sign(dL_j), so per-carrier adverse structure is erased.

      denom="signed" (legacy, unstable):
          B[i,j] = dL_j * C_loss[i,j] / sum_i C_loss[i,j]
        Signed and sums to dL_j, but the shares are not convex and blow up when the signed
        denominator cancels to ~0 (the |B| >> |C| far-tail spikes).

    Returns:
        (B, denom_used, clamped): denom_used is the [n] per-source denominator (sum_i C_loss for
        slope/signed, sum_i |C_loss| for magnitude); a source "moved" when |denom_used| > eps.
        clamped is a [n] bool, True where the slope clip was active (slope mode only) -- for those
        sources sum_i B != dL_j by construction (they are exactly the estimation-noise sources, so
        exclude them from the sum_i B == dL_j exactness check).
    """
    C_basis = np.asarray(C_basis, dtype=np.float64)
    dL_j = np.asarray(dL_j, dtype=np.float64)
    clamped = np.zeros(C_basis.shape[1], dtype=bool)
    if denom == "slope":
        s = C_basis.sum(axis=0)                                       # [n] first-order loss change
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(np.abs(s) > eps, dL_j / s, 0.0)          # exact/first-order, |.|<=1 (1-Lipschitz)
        clamped = np.abs(slope) > slope_clip
        slope = np.clip(slope, -slope_clip, slope_clip)              # guard estimation noise near s~0
        B = slope[None, :] * C_basis                                 # B[i,j] = slope_j . C_loss[i,j]
        return B, s, clamped
    if denom == "magnitude":
        Z = np.abs(C_basis).sum(axis=0)                              # [n] sum_i |C[i,j]|
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(Z > eps, np.abs(C_basis) / Z, 0.0)      # convex, in [0,1]
        B = dL_j[None, :] * share                                    # sum_i B[i,j] = dL_j
        return B, Z, clamped
    if denom == "signed":
        s = C_basis.sum(axis=0)                                      # [n] sum_i C[i,j]
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(np.abs(s) > eps, C_basis / s, 0.0)
        B = dL_j[None, :] * share                                    # sum_i B[i,j] = dL_j
        return B, s, clamped
    raise ValueError(f"denom must be 'slope', 'magnitude' or 'signed', got {denom!r}")


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


def make_distance_bins(dmax: int, distance=None, strategy: str = "log", n_equal: int = 8):
    """Adaptive shortest-path-distance bins as a list of inclusive (lo, hi) ranges.

    Per-hop means are dominated by a few pairs at far distances (the graph-diameter tail is
    heavy), so their CIs are huge and, on large-graph tasks, the x-axis is unreadable.
    Pooling distances into bins tightens the CIs and keeps the axis legible while preserving
    near/mid/far resolution.

      "hop"          one bin per hop (fine for small-diameter tasks like ZINC).
      "log"          singletons {0},{1},{2},{3} then dyadic {4-7},{8-15},{16-31},... to dmax.
                     Keeps near-hop resolution; pools the far tail. Universal default.
      "equal_count"  singletons {0..3} then quantile bins on d>=4 with ~equal pair counts.
    """
    dmax = int(dmax)
    if dmax < 0:
        return []
    if strategy == "hop":
        return [(d, d) for d in range(dmax + 1)]
    if strategy == "log":
        bins = [(d, d) for d in range(0, min(4, dmax + 1))]
        lo = 4
        while lo <= dmax:
            hi = min(2 * lo - 1, dmax)
            bins.append((lo, hi))
            lo = hi + 1
        return bins
    if strategy == "equal_count":
        if distance is None:
            raise ValueError("equal_count binning needs the distance array")
        d = np.asarray(distance)
        bins = [(k, k) for k in range(0, min(4, dmax + 1))]
        rest = d[d >= 4]
        if rest.size:
            edges = np.unique(np.quantile(rest, np.linspace(0, 1, n_equal + 1)).round().astype(int))
            edges = np.clip(edges, 4, dmax)
            edges = np.unique(np.concatenate([edges, [dmax + 1]]))
            lo = 4
            for e in edges:
                hi = int(e) - 1 if int(e) > lo else lo
                if hi >= lo:
                    bins.append((lo, min(hi, dmax)))
                    lo = min(hi, dmax) + 1
                if lo > dmax:
                    break
        return bins
    raise ValueError(f"unknown bin strategy {strategy!r}")


def _bin_index(distance, bins):
    """Map each distance to its bin index (-1 if in no bin; bins cover 0..dmax so rare)."""
    idx = np.full(distance.shape, -1, dtype=np.int64)
    for b, (lo, hi) in enumerate(bins):
        idx[(distance >= lo) & (distance <= hi)] = b
    return idx


def _central(x, how: str):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if how == "median":
        return float(np.median(x))
    if how == "mean":
        return float(np.mean(x))
    if how == "trimmed":  # 20% trimmed mean
        xs = np.sort(x)
        k = int(np.floor(0.2 * xs.size))
        return float(xs[k:xs.size - k].mean()) if 2 * k < xs.size else float(np.median(xs))
    raise ValueError(f"unknown central tendency {how!r}")


def aggregate_carriage_curves(graph_id, distance, F, B, n_boot: int = 2000,
                              boot_seed: int = 1234, bin_strategy: str = "log",
                              central: str = "trimmed", min_count: int = 50) -> dict:
    """Binned, robust, graph-clustered distance profiles F, B, the loss mass S, and B_far.

    For each SPD bin:
      * F, B: two-stage estimator over graphs -- per-graph MEAN within the bin, then a robust
        central tendency (``central``: median / 20%-trimmed mean / mean) ACROSS graphs, with a
        graph-clustered bootstrap CI. This weights each graph equally (not by pair count) and
        resists the few large-diameter graphs that dominate far bins. Bins with < ``min_count``
        pairs are dropped (NaN).
      * S: per-graph SUM within the bin, MEAN over graphs (loss units; additive, so the tail
        bins telescope to B_far).
      * B_far(edge): per-graph SUM over d > edge, MEAN over graphs, evaluated at each bin's
        upper edge.

    Returns bins as (bin_lo, bin_hi, bin_label) with an integer bin_center (the bin index) for
    even, readable x-spacing.
    """
    graph_id = np.asarray(graph_id)
    distance = np.asarray(distance).astype(np.int64)
    F = np.asarray(F, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    dmax = int(distance.max()) if distance.size else -1
    bins = make_distance_bins(dmax, distance, bin_strategy)
    nb = len(bins)
    bin_lo = np.array([lo for lo, _ in bins], dtype=np.int64)
    bin_hi = np.array([hi for _, hi in bins], dtype=np.int64)
    bin_label = [str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in bins]
    bin_center = np.arange(nb)

    uniq_g = np.unique(graph_id)
    n_g = int(uniq_g.size)
    gpos = {int(v): p for p, v in enumerate(uniq_g)}
    gidx = np.array([gpos[int(v)] for v in graph_id])
    bidx = _bin_index(distance, bins)

    def _per_graph_mean(vals, mask):
        s = np.zeros(n_g); c = np.zeros(n_g)
        np.add.at(s, gidx[mask], vals[mask])
        np.add.at(c, gidx[mask], 1.0)
        present = c > 0
        return np.where(present, s / np.maximum(c, 1), np.nan), present

    def _per_graph_sum(vals, mask):
        s = np.zeros(n_g)
        np.add.at(s, gidx[mask], vals[mask])
        return s

    def _robust_over_graphs(per_graph_vals):
        vg = per_graph_vals[np.isfinite(per_graph_vals)]
        if vg.size == 0:
            return float("nan"), float("nan"), float("nan")
        point = _central(vg, central)
        br = np.random.default_rng(boot_seed)
        draws = br.integers(0, vg.size, size=(n_boot, vg.size))
        boot = np.array([_central(vg[d], central) for d in draws])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return float(point), float(lo), float(hi)

    def _mean_over_graphs(per_graph):
        br = np.random.default_rng(boot_seed)
        draws = br.integers(0, per_graph.size, size=(n_boot, per_graph.size))
        vals = per_graph[draws].mean(axis=1)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(per_graph.mean()), float(lo), float(hi)

    counts = np.zeros(nb, dtype=np.int64)
    F_mean, F_lo, F_hi = (np.full(nb, np.nan) for _ in range(3))
    B_mean, B_lo, B_hi = (np.full(nb, np.nan) for _ in range(3))
    S_mean, S_lo, S_hi = (np.full(nb, np.nan) for _ in range(3))
    F_per_graph = np.full((n_g, nb), np.nan)

    for b in range(nb):
        m = bidx == b
        counts[b] = int(m.sum())
        Fg, present = _per_graph_mean(F, m)
        F_per_graph[:, b] = Fg
        if counts[b] >= min_count:
            F_mean[b], F_lo[b], F_hi[b] = _robust_over_graphs(Fg)
            Bg, _ = _per_graph_mean(B, m)
            B_mean[b], B_lo[b], B_hi[b] = _robust_over_graphs(Bg)
        S_mean[b], S_lo[b], S_hi[b] = _mean_over_graphs(_per_graph_sum(B, m))

    # B_far at each bin's upper edge: per-graph sum over d > edge, mean over graphs.
    edges = bin_hi.copy()
    Bf_mean, Bf_lo, Bf_hi = (np.full(nb, np.nan) for _ in range(3))
    for b, k in enumerate(edges):
        Bf_mean[b], Bf_lo[b], Bf_hi[b] = _mean_over_graphs(_per_graph_sum(B, distance > k))

    return {
        "bin_lo": bin_lo, "bin_hi": bin_hi, "bin_label": bin_label, "bin_center": bin_center,
        "distances": bin_center, "n_bins": nb, "bin_strategy": bin_strategy, "central": central,
        "min_count": int(min_count), "pair_counts": counts, "n_graphs": n_g,
        "F_mean": F_mean, "F_lo": F_lo, "F_hi": F_hi,
        "B_mean": B_mean, "B_lo": B_lo, "B_hi": B_hi,
        "S_mean": S_mean, "S_lo": S_lo, "S_hi": S_hi,
        "k": edges, "B_far_mean": Bf_mean, "B_far_lo": Bf_lo, "B_far_hi": Bf_hi,
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
