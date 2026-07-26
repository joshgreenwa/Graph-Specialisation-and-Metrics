"""Task-agnostic carriage estimators and distance aggregation.

Pure NumPy/PyTorch, with no model-backend dependency, so it is unit-testable without a
GPU or a trained model. Model-specific code registers captured states, gradients, loss
replay, and a linear carrier-to-readout projection.

Implements the dissertation methodology (Ch. 3, Sections 3.2-3.3):

  semantic carriage      C_swap[i,j] = (1/K) sum_k g_i^T [h^L_i(clean) - h^L_i(swap_k)]   (3.4/3.5)
  functional carriage    F_sens[i,j] = (1/K) sum_k ||q[k,i,j]||_2                         (3.6/3.7)
  beneficial carriage    exact loss change via a signed final-state path integral
  B_far(k)               sum over {(i,j): d(i,j) > k} of B[i,j]                            [MAE units]
"""

from __future__ import annotations

import numpy as np


# Included in graph-wise progress fingerprints and persisted output metadata.  Version 2 makes
# eventwise sensitivity (F_sens) the production functional-carriage estimand; version 1 used the
# norm of the donor-mean response (now retained as the F_coh diagnostic).
FUNCTIONAL_CARRIAGE_VERSION = 2
FUNCTIONAL_CARRIAGE_ESTIMAND = "F_sens"


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

    Sign convention: dh is CLEAN minus SWAPPED, exactly as written in Eq. 3.4. This
    function returns the signed donor-mean estimator ``C`` used by loss carriage.
    Production functional carriage ``F_sens`` is computed separately from eventwise terms.
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
    """Default functional carriage F_sens: mean eventwise output-response magnitude.

    Label-free (Def 3.3.2): how much does moving content from j change node i's effect on
    the model's OUTPUT? Magnitude is taken for each donor/partner event before nuisance
    averaging, so valid responses in opposite directions cannot cancel.

    Args:
        g_out: [T, n, m] Jacobian of the output w.r.t. h^L (one [n,m] slice per output t),
               evaluated at the clean input.

    Returns:
        [n, n] numpy: F_sens[i, j] (carrier x source), non-negative.
    """
    delta = h_clean.unsqueeze(0) - h_swap
    return functional_magnitude_from_delta(delta, g_out, num_sources, num_donors)


def functional_magnitude_from_delta(delta, g_out, num_sources, num_donors):
    """Return production ``F_sens`` from a precomputed transport delta.

    This public name is retained for API compatibility, but now resolves to the declared
    default estimand.  Use :func:`functional_magnitudes_from_delta` when the companion
    coherent-response diagnostic is also required.
    """
    F_sens, _F_coh = functional_magnitudes_from_delta(
        delta, g_out, num_sources, num_donors
    )
    return F_sens


def functional_magnitudes_from_delta(delta, g_out, num_sources, num_donors):
    """Compute eventwise sensitivity and coherent-response functional carriage together.

    For ``q[k,i,s] = (g_out[t,i] . delta[k,i,s])_t``:

    ``F_sens[i,s] = mean_k ||q[k,i,s]||_2`` (production default), while
    ``F_coh[i,s] = ||mean_k q[k,i,s]||_2`` is a cancellation/coherence diagnostic.

    Returns two ``[carrier i, source s]`` NumPy arrays ``(F_sens, F_coh)``.  Computing both
    in one pass avoids retaining the full ``[event, output, carrier]`` projection tensor.
    """
    import torch

    S, K = int(num_sources), int(num_donors)
    if delta.ndim != 3:
        raise ValueError(f"delta must be [S*K,n,m], got {tuple(delta.shape)}")
    if int(delta.shape[0]) != S * K:
        raise ValueError(f"delta has {delta.shape[0]} events, expected S*K={S*K}")
    if g_out.ndim != 3 or tuple(g_out.shape[1:]) != tuple(delta.shape[1:]):
        raise ValueError(
            f"g_out must be [T,n,m] aligned with delta; got {tuple(g_out.shape)} "
            f"for delta {tuple(delta.shape)}"
        )
    if int(g_out.shape[0]) < 1:
        raise ValueError("g_out must contain at least one output direction")

    event_sq = torch.zeros(
        (S, K, int(delta.shape[1])), device=delta.device, dtype=delta.dtype
    )
    coherent_sq = torch.zeros(
        (S, int(delta.shape[1])), device=delta.device, dtype=delta.dtype
    )
    for t in range(int(g_out.shape[0])):
        q_t = torch.einsum("rnm,nm->rn", delta, g_out[t]).view(S, K, -1)
        event_sq.add_(q_t.square())
        coherent_sq.add_(q_t.mean(dim=1).square())

    F_sens = event_sq.sqrt().mean(dim=1).t().contiguous()
    F_coh = coherent_sq.sqrt().t().contiguous()
    return F_sens.cpu().numpy(), F_coh.cpu().numpy()


def pool_final_states(h, pooling: str):
    """Apply legacy add/mean graph pooling.

    New backends should register explicit carrier weights with
    :func:`project_final_states`.
    """
    if pooling == "add":
        return h.sum(dim=-2)
    if pooling == "mean":
        return h.mean(dim=-2)
    raise ValueError(f"integrated carriage requires add/mean pooling, got {pooling!r}")


def project_final_states(h, carrier_weights):
    """Apply a registered linear carrier-to-readout projection.

    ``carrier_weights`` is one scalar per carrier. Add/mean pooling and Graphormer's
    graph-token readout are all instances of this operation.
    """

    import torch

    weights = torch.as_tensor(carrier_weights, device=h.device, dtype=h.dtype).reshape(-1)
    if h.ndim not in (2, 3) or int(h.shape[-2]) != int(weights.numel()):
        raise ValueError(
            f"final states {tuple(h.shape)} do not align with "
            f"{int(weights.numel())} carrier weights"
        )
    if not torch.isfinite(weights).all():
        raise ValueError("carrier weights must be finite")
    return torch.einsum("...nm,n->...m", h, weights)


# Gauss--Kronrod (7, 15) nodes and weights on [-1, 1].  The embedded Gauss
# estimate supplies a local carrier-level error estimate; intervals containing
# an L1/ReLU kink are split rather than hidden by a global rescale or clipping.
_GK15_NODES = (
    -0.9914553711208126, -0.9491079123427585, -0.8648644233597691,
    -0.7415311855993945, -0.5860872354676911, -0.4058451513773972,
    -0.2077849550078985, 0.0, 0.2077849550078985, 0.4058451513773972,
    0.5860872354676911, 0.7415311855993945, 0.8648644233597691,
    0.9491079123427585, 0.9914553711208126,
)
_GK15_WEIGHTS = (
    0.0229353220105292, 0.0630920926299786, 0.1047900103222502,
    0.140653259715526, 0.169004726639268, 0.190350578064785,
    0.204432940075299, 0.209482141084728, 0.204432940075299,
    0.190350578064785, 0.169004726639268, 0.140653259715526,
    0.1047900103222502, 0.0630920926299786, 0.0229353220105292,
)
_G7_WEIGHTS = (
    0.0, 0.129484966168870, 0.0, 0.279705391489277, 0.0,
    0.381830050505119, 0.0, 0.417959183673469, 0.0,
    0.381830050505119, 0.0, 0.279705391489277, 0.0,
    0.129484966168870, 0.0,
)


def integrated_loss_carriage(
    h_clean,
    h_swap,
    loss_from_pooled=None,
    *,
    loss_from_states=None,
    pooling: str | None = None,
    carrier_weights=None,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    max_intervals: int = 64,
):
    """Signed finite-loss carriage along each swapped-to-clean final-state path.

    For each replica ``r=(source j, donor k)`` this computes

    ``b[r,i] = integral_0^1 <d loss(H(alpha))/d h_i, h_clean_i-h_swap_i> d alpha``

    with ``H(alpha)=h_swap+alpha*(h_clean-h_swap)``.  The caller donor-averages
    these *per-donor* paths afterwards. A backend may register either a linear
    carrier projection followed by ``loss_from_pooled`` or an exact nonlinear
    ``loss_from_states`` readout.

    Adaptive embedded Gauss--Kronrod quadrature localises L1 and ReLU kinks.  A
    path is converged only when both (a) the L1 carrier refinement estimate and
    (b) endpoint completeness are within tolerance.  Nothing is clipped,
    rescaled, or completeness-corrected: the raw numerical residual is returned.

    Args:
        h_clean / h_swap: ``[R,n,m]`` tensors at the input to the graph head.
        loss_from_pooled: differentiable callable mapping pooled states ``[Q,m]``
            to one scalar task loss per row, ``[Q]``.
        loss_from_states: optional differentiable callable mapping complete states
            ``[Q,n,m]`` to one scalar task loss per row. Pass exactly one loss callable.
        pooling: legacy ``"add"`` or ``"mean"`` shortcut.
        carrier_weights: optional registered linear readout projection. For
            Graphormer this is one at the graph token and zero at molecular nodes.
        atol / rtol: convergence tolerances in task-loss/carrier units.
        max_intervals: maximum locally-adapted intervals per donor path.

    Returns a dict of tensors, all in source-major replica order:
        ``carriage [R,n]``, ``loss_delta [R]`` (clean minus swap),
        ``completeness_residual [R]``, ``quadrature_error [R]``,
        ``intervals [R]``, and ``converged [R]``.
    """
    import torch

    if h_clean.ndim != 3 or h_swap.ndim != 3 or h_clean.shape != h_swap.shape:
        raise ValueError(
            f"h_clean and h_swap must have identical [R,n,m] shape; got "
            f"{tuple(h_clean.shape)} and {tuple(h_swap.shape)}"
        )
    if int(h_clean.shape[0]) < 1:
        raise ValueError("integrated carriage needs at least one donor path")
    if atol < 0 or rtol < 0:
        raise ValueError("integrated carriage tolerances must be non-negative")
    if int(max_intervals) < 1:
        raise ValueError("max_intervals must be >= 1")
    clean = h_clean.detach()
    swap = h_swap.detach()
    R, n, _m = clean.shape
    delta_h = clean - swap
    direct = loss_from_states is not None
    if direct and loss_from_pooled is not None:
        raise ValueError("pass loss_from_states or loss_from_pooled, not both")
    if not direct and loss_from_pooled is None:
        raise ValueError("one differentiable readout loss callable is required")
    if direct:
        weights = None
        endpoint_clean = clean
        endpoint_swap = swap
        endpoint_delta = delta_h
    else:
        if carrier_weights is None:
            if pooling not in ("add", "mean"):
                raise ValueError(
                    "integrated carriage requires add/mean pooling or explicit carrier_weights"
                )
            weights = clean.new_ones(n)
            if pooling == "mean":
                weights /= float(n)
        else:
            weights = torch.as_tensor(
                carrier_weights, device=clean.device, dtype=clean.dtype
            ).reshape(-1)
            if tuple(weights.shape) != (int(n),):
                raise ValueError(
                    f"carrier_weights has shape {tuple(weights.shape)}; expected {(int(n),)}"
                )
            if not torch.isfinite(weights).all():
                raise ValueError("carrier_weights must be finite")
        endpoint_clean = project_final_states(clean, weights)
        endpoint_swap = project_final_states(swap, weights)
        endpoint_delta = endpoint_clean - endpoint_swap

    def _loss(value):
        out = loss_from_states(value) if direct else loss_from_pooled(value)
        if out.ndim != 1 or int(out.shape[0]) != int(value.shape[0]):
            raise ValueError(
                "readout loss must return one scalar per row: "
                f"input {tuple(value.shape)} -> output {tuple(out.shape)}"
            )
        return out

    with torch.no_grad():
        loss_delta = (_loss(endpoint_clean) - _loss(endpoint_swap)).detach()

    nodes = torch.as_tensor(_GK15_NODES, device=clean.device, dtype=clean.dtype)
    wk = torch.as_tensor(_GK15_WEIGHTS, device=clean.device, dtype=clean.dtype)
    wg = torch.as_tensor(_G7_WEIGHTS, device=clean.device, dtype=clean.dtype)

    def _eval_intervals(replica, left, right):
        """Embedded estimates for a batch of (replica, [left,right]) intervals."""
        replica = torch.as_tensor(replica, device=clean.device, dtype=torch.long)
        left_t = torch.as_tensor(left, device=clean.device, dtype=clean.dtype)
        right_t = torch.as_tensor(right, device=clean.device, dtype=clean.dtype)
        centre = (left_t + right_t) * 0.5
        half = (right_t - left_t) * 0.5
        alpha = centre[:, None] + half[:, None] * nodes[None, :]
        with torch.enable_grad():
            if direct:
                path_value = (
                    endpoint_swap[replica, None, :, :]
                    + alpha[:, :, None, None]
                    * endpoint_delta[replica, None, :, :]
                ).reshape(-1, n, clean.shape[-1]).detach().requires_grad_(True)
            else:
                path_value = (
                    endpoint_swap[replica, None, :]
                    + alpha[:, :, None] * endpoint_delta[replica, None, :]
                ).reshape(-1, endpoint_clean.shape[-1]).detach().requires_grad_(True)
            losses = _loss(path_value)
            grad = torch.autograd.grad(
                losses.sum(), path_value, create_graph=False
            )[0]
        if direct:
            grad = grad.reshape(
                replica.numel(), nodes.numel(), n, clean.shape[-1]
            )
            a_k = half[:, None, None] * torch.einsum("q,rqnm->rnm", wk, grad)
            a_g = half[:, None, None] * torch.einsum("q,rqnm->rnm", wg, grad)
            b_k = torch.einsum("rnm,rnm->rn", delta_h[replica], a_k)
            b_g = torch.einsum("rnm,rnm->rn", delta_h[replica], a_g)
        else:
            grad = grad.reshape(
                replica.numel(), nodes.numel(), endpoint_clean.shape[-1]
            )
            a_k = half[:, None] * torch.einsum("q,rqm->rm", wk, grad)
            a_g = half[:, None] * torch.einsum("q,rqm->rm", wg, grad)
            b_k = weights[None, :] * torch.einsum(
                "rnm,rm->rn", delta_h[replica], a_k
            )
            b_g = weights[None, :] * torch.einsum(
                "rnm,rm->rn", delta_h[replica], a_g
            )
        error = (b_k - b_g).abs().sum(dim=-1)
        return b_k.detach(), error.detach()

    replica0 = torch.arange(R, device=clean.device)
    initial_b, initial_error = _eval_intervals(replica0, [0.0] * R, [1.0] * R)
    total_b = initial_b.clone()
    error_sum = initial_error.detach().cpu().double().numpy()

    # Per-path interval records: [left, right, Kronrod carrier vector, embedded L1 error].
    # Chunks in the runners keep R moderate; Python records make the adaptive choice explicit
    # and avoid padding every donor to the worst path's refinement depth.
    records = [
        [[0.0, 1.0, initial_b[r].clone(), float(error_sum[r])]]
        for r in range(R)
    ]

    def _convergence_mask():
        residual = (total_b.sum(dim=-1) - loss_delta).abs()
        carrier_scale = total_b.abs().sum(dim=-1)
        quad_tol = float(atol) + float(rtol) * carrier_scale
        complete_tol = float(atol) + float(rtol) * loss_delta.abs()
        qerr = torch.as_tensor(error_sum, device=clean.device, dtype=clean.dtype)
        return (qerr <= quad_tol) & (residual <= complete_tol)

    converged = _convergence_mask()
    while True:
        unresolved = [
            r for r in range(R)
            if not bool(converged[r].item()) and len(records[r]) < int(max_intervals)
        ]
        if not unresolved:
            break

        parents = []
        child_replica, child_left, child_right = [], [], []
        for r in unresolved:
            # Split the locally least-certain interval.  Width breaks ties so a rare
            # zero embedded-error / nonzero-completeness case still makes progress.
            q = max(
                range(len(records[r])),
                key=lambda z: (records[r][z][3], records[r][z][1] - records[r][z][0]),
            )
            left, right, value, error = records[r].pop(q)
            mid = (left + right) * 0.5
            parents.append((r, left, right, value, error))
            child_replica.extend((r, r))
            child_left.extend((left, mid))
            child_right.extend((mid, right))

        child_b, child_error = _eval_intervals(
            child_replica, child_left, child_right
        )
        child_error_cpu = child_error.detach().cpu().double().numpy()
        for z, (r, left, right, old_value, old_error) in enumerate(parents):
            mid = (left + right) * 0.5
            b_left, b_right = child_b[2 * z], child_b[2 * z + 1]
            e_left = float(child_error_cpu[2 * z])
            e_right = float(child_error_cpu[2 * z + 1])
            total_b[r] += b_left + b_right - old_value
            error_sum[r] = max(0.0, error_sum[r] + e_left + e_right - old_error)
            records[r].append([left, mid, b_left.clone(), e_left])
            records[r].append([mid, right, b_right.clone(), e_right])
        converged = _convergence_mask()

    completeness = total_b.sum(dim=-1) - loss_delta
    intervals = torch.as_tensor(
        [len(x) for x in records], device=clean.device, dtype=torch.long
    )
    return {
        "carriage": total_b,
        "loss_delta": loss_delta,
        "completeness_residual": completeness,
        "quadrature_error": torch.as_tensor(
            error_sum, device=clean.device, dtype=clean.dtype
        ),
        "intervals": intervals,
        "converged": converged,
    }


def beneficial_attribute(C_basis, dL_j, eps=1e-8, denom="slope", slope_clip=1.0):
    """Turn the signed loss-carriage into per-pair beneficial carriage B[i,j].

    ``C_basis`` is the signed first-order loss-carriage C_loss[i,j] = (dL/dh_i).dh_i(j); summed
    over carriers it is the FIRST-ORDER loss change  sum_i C_loss[i,j] = (dL/dyhat)|_clean . dyhat_j.
    ``dL_j`` is the EXACT per-source loss change  L_clean - mean_k L_swap(j,k)  (<0 = beneficial).

    Modes (default "slope"):

      denom="slope" (DEFAULT for backwards compatibility; finite-tangent approximation):
          s_j    = dL_j / sum_i C_loss[i,j]                          exact / first-order loss change
          B[i,j] = clip(s_j, -slope_clip, +slope_clip) . C_loss[i,j]
        Keeps the per-carrier SIGN of C_loss and bounds the first-order estimate.  However,
        C_loss uses the gradient frozen at the clean endpoint, so a finite intervention can
        cross readout/loss curvature or an L1/ReLU kink and legitimately give |s_j| > 1.
        Clip activation therefore diagnoses tangent failure; it is not evidence that only
        numerical noise was removed. Use the runner's ``integrated`` option for signed
        finite-loss attribution without a ratio or clipping.

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
        sources sum_i B != dL_j by construction, so they are excluded from that estimator's
        sum_i B == dL_j exactness check.
    """
    C_basis = np.asarray(C_basis, dtype=np.float64)
    dL_j = np.asarray(dL_j, dtype=np.float64)
    clamped = np.zeros(C_basis.shape[1], dtype=bool)
    if denom == "slope":
        s = C_basis.sum(axis=0)                                       # [n] first-order loss change
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(np.abs(s) > eps, dL_j / s, 0.0)          # finite / clean-tangent change
        clamped = np.abs(slope) > slope_clip
        slope = np.clip(slope, -slope_clip, slope_clip)              # bound tangent mismatch / s~0
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
                              central: str = "trimmed", min_count: int = 50,
                              F_coh=None) -> dict:
    """Binned, robust, graph-clustered profiles F_sens, B, loss mass S, and B_far.

    For each SPD bin:
      * F_sens, B: two-stage estimator over graphs -- per-graph MEAN within the bin, then a robust
        central tendency (``central``: median / 20%-trimmed mean / mean) ACROSS graphs, with a
        graph-clustered bootstrap CI. This weights each graph equally (not by pair count) and
        resists the few large-diameter graphs that dominate far bins. Bins with < ``min_count``
        pairs are dropped (NaN). If supplied, F_coh is aggregated identically as a diagnostic.
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
    F_coh = None if F_coh is None else np.asarray(F_coh, dtype=np.float64)
    if F_coh is not None and F_coh.shape != F.shape:
        raise ValueError(f"F_coh shape {F_coh.shape} != F shape {F.shape}")

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
    F_coh_mean, F_coh_lo, F_coh_hi = (np.full(nb, np.nan) for _ in range(3))
    B_mean, B_lo, B_hi = (np.full(nb, np.nan) for _ in range(3))
    S_mean, S_lo, S_hi = (np.full(nb, np.nan) for _ in range(3))
    F_per_graph = np.full((n_g, nb), np.nan)
    F_coh_per_graph = np.full((n_g, nb), np.nan)

    for b in range(nb):
        m = bidx == b
        counts[b] = int(m.sum())
        Fg, present = _per_graph_mean(F, m)
        F_per_graph[:, b] = Fg
        Fcg = None
        if F_coh is not None:
            Fcg, _ = _per_graph_mean(F_coh, m)
            F_coh_per_graph[:, b] = Fcg
        if counts[b] >= min_count:
            F_mean[b], F_lo[b], F_hi[b] = _robust_over_graphs(Fg)
            if Fcg is not None:
                F_coh_mean[b], F_coh_lo[b], F_coh_hi[b] = _robust_over_graphs(Fcg)
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
        "F_coh_mean": F_coh_mean, "F_coh_lo": F_coh_lo, "F_coh_hi": F_coh_hi,
        "B_mean": B_mean, "B_lo": B_lo, "B_hi": B_hi,
        "S_mean": S_mean, "S_lo": S_lo, "S_hi": S_hi,
        "k": edges, "B_far_mean": Bf_mean, "B_far_lo": Bf_lo, "B_far_hi": Bf_hi,
        "F_per_graph": F_per_graph,
        "F_coh_per_graph": F_coh_per_graph,
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
