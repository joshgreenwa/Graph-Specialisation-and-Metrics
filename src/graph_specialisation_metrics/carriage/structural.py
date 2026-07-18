"""Structural intervention: perturb the topology-derived encodings, hold content X fixed.

This is the structural complement of the semantic donor swap (``content.py``). Where the
semantic swap overwrites a node's *content* row and leaves structure S fixed, a structural
intervention perturbs the *structure-derived* tensors (node-RRWP, pair-RRWP, and -- for the
k-hop variants -- the attention-support mask, which those encoders derive at forward time)
and leaves ``data.x`` fixed. It feeds the IDENTICAL core estimator: the runner still computes
``delta = h_clean - h_swap`` and hands it to ``core`` unchanged.

Two modes, both anchored at a single source node ``u`` so the distance profile ``d(i,u)`` is
defined exactly as on the semantic side:

  * ``transposition`` (primary, on-manifold): conjugate every structure-derived tensor by the
    node transposition ``P_(u v)`` -- swap node-wise rows ``u<->v`` and relabel ``u<->v`` in
    every pairwise index (``r -> Pπ r Pπᵀ``). The perturbed state is the structure of a REAL
    isomorphic copy of the graph paired with the original (unmoved) content, so it is on the
    "physical structure, mismatched content" manifold, exactly like the semantic donor swap.
    Both ``u`` and ``v`` move, so the runner anchors at ``u`` and MARGINALISES the partner
    ``v`` over K degree-matched draws (the donor-average recipe). Because the k-hop mask is
    derived from the (conjugated) pairwise RRWP, it is conjugated too -- so dense and masked
    models are both perturbed through the one operation.

  * ``single_node`` (secondary, off-manifold): copy a donor ``v``'s structural footprint onto
    ``u`` only, leaving ``v`` untouched. Only ``u`` moves, so ``d(i,u)`` is perfectly clean,
    but ``u`` now duplicates ``v``'s role (no real graph has this) -- off-manifold. A cheap
    functional-sensitivity baseline; do not read beneficial B off it as task-intrinsic.

Everything here is pure tensor manipulation on a PyG ``Data`` object. The helpers that do the
index/row surgery take plain tensors, so they are unit-testable without PyG or a model.

BETA: this module is deliberately self-contained. Delete it, ``structural_runner.py``, and the
``intervention="structural"`` branch in ``colab.run`` to remove the structural path entirely.
"""

from __future__ import annotations

import numpy as np

# Node-indexed, topology-derived tensors that a transposition swaps by row (and single_node
# copies by row). Content (``x``) and the graph label (``y``) are NEVER in this list. Applied
# only if present on the Data object, so it is safe across GRIT configs.
NODE_STRUCT_ATTRS = ("rrwp", "deg", "log_deg", "abs_pe", "pestat_RRWP")

# Pairwise (edge-indexed) tensors: an index [2, E] plus an aligned value [E, ...]. Under a
# transposition the index is relabelled and the value rides along unchanged.
PAIR_TENSORS = (("rrwp_index", "rrwp_val"), ("edge_index", "edge_attr"))


# --------------------------------------------------------------------------------------- #
# Pure tensor helpers (torch only; unit-testable without PyG).
# --------------------------------------------------------------------------------------- #
def _swap_rows(t, u: int, v: int):
    """Return a copy of ``t`` with rows ``u`` and ``v`` exchanged (node-wise conjugation)."""
    t = t.clone()
    tmp = t[u].clone()
    t[u] = t[v]
    t[v] = tmp
    return t


def _relabel_uv(index, u: int, v: int):
    """Relabel node ids ``u<->v`` everywhere in an index tensor (the pairwise conjugation).

    For a symmetric structural relation stored in COO, swapping the label ``u<->v`` in both
    rows of the index IS the conjugation ``P R Pᵀ`` (the aligned value tensor is unchanged).
    """
    idx = index.clone()
    mu = idx == u
    mv = idx == v
    idx[mu] = v
    idx[mv] = u
    return idx


def _copy_incidence(index, u: int, v: int, val=None):
    """Copy node ``v``'s incidences onto ``u`` (single-node footprint copy); ``v`` untouched.

    Drops every entry currently touching ``u``, then re-adds ``v``'s incident entries with the
    ``v`` endpoint relabelled to ``u`` (``(v,w)->(u,w)`` and ``(w,v)->(w,u)``). ``u`` thereby
    adopts ``v``'s connectivity/relations. This is off-manifold by construction; ``v``'s own
    self-entry contributes a negligible identity-channel artefact at ``(u,v)``/``(v,u)``.
    """
    import torch

    row, col = index[0], index[1]
    keep = (row != u) & (col != u)
    src = row == v            # v's outgoing  (v, w) -> (u, w)
    dst = col == v            # v's incoming  (w, v) -> (w, u)

    idx_parts = [index[:, keep]]
    val_parts = [val[keep]] if val is not None else None
    if bool(src.any()):
        oi = index[:, src].clone()
        oi[0] = u
        idx_parts.append(oi)
        if val is not None:
            val_parts.append(val[src])
    if bool(dst.any()):
        ii = index[:, dst].clone()
        ii[1] = u
        idx_parts.append(ii)
        if val is not None:
            val_parts.append(val[dst])

    new_idx = torch.cat(idx_parts, dim=1)
    new_val = torch.cat(val_parts, dim=0) if val is not None else None
    return new_idx, new_val


# --------------------------------------------------------------------------------------- #
# The intervention: perturb one Data object.
# --------------------------------------------------------------------------------------- #
def perturb(data, u: int, v: int, mode: str):
    """Return a new Data with ``u``'s (and, for transposition, ``v``'s) structure perturbed.

    Content ``data.x`` and the label ``data.y`` are never touched. ``u == v`` is a genuine
    no-op for both modes (used as the structural analog of the no-op donor check).
    """
    d = data.clone()
    if mode == "transposition":
        for name in NODE_STRUCT_ATTRS:
            t = getattr(data, name, None)
            if t is not None:
                setattr(d, name, _swap_rows(t, u, v))
        for idx_name, val_name in PAIR_TENSORS:
            idx = getattr(data, idx_name, None)
            if idx is not None:
                setattr(d, idx_name, _relabel_uv(idx, u, v))
                # value tensor rides along unchanged (already carried by d = data.clone())
    elif mode == "single_node":
        for name in NODE_STRUCT_ATTRS:
            t = getattr(data, name, None)
            if t is not None:
                tt = t.clone()
                tt[u] = t[v]
                setattr(d, name, tt)
        for idx_name, val_name in PAIR_TENSORS:
            idx = getattr(data, idx_name, None)
            if idx is not None:
                val = getattr(data, val_name, None)
                new_idx, new_val = _copy_incidence(idx, u, v, val)
                setattr(d, idx_name, new_idx)
                if val is not None:
                    setattr(d, val_name, new_val)
    else:
        raise ValueError(f"structural mode must be 'transposition' or 'single_node', got {mode!r}")
    return d


def full_relabel(data, u: int, v: int):
    """A FULL node relabelling of the graph: transpose structure AND content ``x`` rows u<->v.

    This is a pure isomorphism, so a permutation-invariant GRIT must give the identical pooled
    prediction. Used as a completeness check: if ``perturb(..., 'transposition')`` misses a
    structural channel, this relabelling's prediction will drift from the clean one.
    """
    d = perturb(data, u, v, "transposition")
    if getattr(data, "x", None) is not None:
        d.x = _swap_rows(data.x, u, v)
    return d


# --------------------------------------------------------------------------------------- #
# Partner sampling (the marginalised nuisance v for the transposition anchor u).
# --------------------------------------------------------------------------------------- #
def node_degrees(edge_index, n: int) -> np.ndarray:
    """Undirected-style out-degree per node from ``edge_index`` (for degree-matched partners)."""
    import torch

    deg = torch.zeros(n, dtype=torch.long)
    if edge_index is not None and edge_index.numel():
        deg.scatter_add_(0, edge_index[0].to(torch.long),
                         torch.ones(edge_index.shape[1], dtype=torch.long))
    return deg.cpu().numpy()


def sample_partners(deg: np.ndarray, u: int, K: int, rng, match: str = "degree") -> np.ndarray:
    """Draw ``K`` partner nodes ``v`` for anchor ``u`` (with replacement), the on-manifold ladder.

    ``match='degree'`` samples from the same-degree bucket (minimal, near-manifold role swap);
    falls back to the nearest-degree nodes, then to any other node, on tiny/degenerate graphs.
    ``match='any'`` samples uniformly from the other nodes (a looser, less on-manifold null).
    """
    n = int(len(deg))
    others = np.arange(n)
    others = others[others != u]
    if others.size == 0:                      # single-node graph: only the no-op partner exists
        return np.full(K, u, dtype=np.int64)
    if match == "degree":
        cand = others[deg[others] == deg[u]]
        if cand.size == 0:                    # nearest degree band
            gap = np.abs(deg[others] - deg[u])
            cand = others[gap == gap.min()]
    elif match == "any":
        cand = others
    else:
        raise ValueError(f"partner_match must be 'degree' or 'any', got {match!r}")
    return rng.choice(cand, size=K, replace=True).astype(np.int64)


# --------------------------------------------------------------------------------------- #
# Verification (mirrors the semantic checks, structural side).
# --------------------------------------------------------------------------------------- #
def verify_perturbation(base, pert, u: int, v: int, mode: str) -> None:
    """Assert content-invariance and (for transposition) exact structural equivariance."""
    import torch

    assert torch.equal(pert.x, base.x), "structural intervention changed content x (must hold X fixed)"
    if mode != "transposition":
        return
    assert torch.equal(pert.edge_index, _relabel_uv(base.edge_index, u, v)), \
        "edge_index not conjugated by P_(u v) under a structural transposition"
    for name in NODE_STRUCT_ATTRS:
        t = getattr(base, name, None)
        if t is not None:
            assert torch.equal(getattr(pert, name), _swap_rows(t, u, v)), \
                f"node-structural tensor {name!r} not row-swapped under a transposition"
    for idx_name, val_name in PAIR_TENSORS:
        idx = getattr(base, idx_name, None)
        if idx is not None:
            assert torch.equal(getattr(pert, idx_name), _relabel_uv(idx, u, v)), \
                f"pairwise index {idx_name!r} not relabelled under a transposition"
            val = getattr(base, val_name, None)
            if val is not None:
                assert torch.equal(getattr(pert, val_name), val), \
                    f"pairwise value {val_name!r} changed under a transposition (must ride along)"
