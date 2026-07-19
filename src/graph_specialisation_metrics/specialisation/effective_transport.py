"""Static routing versus effective relational transport in trained GRIT heads.

This module is an optional, self-contained command extension of the specialisation
analysis.  It does not alter the model or score implementation.  It reconstructs
the exact pair messages produced by official GRIT, including the ``VeRow``
edge-enhancement term, and separates two questions that attention maps conflate:

``routing``
    Are receiver-specific attention weights needed, beyond the fixed support and
    each source's total attention mass?

``effective transport``
    After value transport, edge enhancement, degree scaling, and the per-head
    block of ``O_h``, does a head deliver a receiver-specific signal that matters
    for the ZINC prediction?

The public :func:`run_effective_transport` function consumes a loaded
``GritHeadModel`` through ``result['gm']`` and *only* the explicitly supplied
confirmation graph ids and mediation-selected head rankings.  Every forward
builds a fresh PyG Batch and every temporary hook is removed by a context manager.

The algebra mirrors ``grit/layer/grit_layer.py`` at the pinned GRIT commit:

    P^V_{e,h} = a_{e,h} V_h(x_src)
    P^E_{e,h} = a_{e,h} (e^t_{e,h} VeRow_h)
    wV_{i,h}  = sum_{e:dst(e)=i} (P^V_{e,h} + P^E_{e,h})

The effective, head-separable site is the input to ``layer.O_h``.  It is after
the optional degree scaler and before heads are mixed by the output projection.
For a graph g, ``U = B + R`` with ``B`` the node mean and ``R`` centred.  This is
an exact orthogonal decomposition; its projected counterpart ``T`` is exact too.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Mapping, Optional, Sequence

import numpy as np


SCHEMA_VERSION = 1
CAUSAL_TREATMENTS = (
    "static_routing",
    "broadcast_only",
    "residual_permuted",
    "full_head_zero",
    "remove_content_residual",
    "remove_edge_residual",
)


def _batch_get(batch, key: str, default=None):
    """Read a PyG Batch field without depending on its exact Mapping API."""

    try:
        return batch.get(key, default)
    except (AttributeError, TypeError):
        return getattr(batch, key, default)


def _scatter_sum(values, index, size: int):
    """Torch ``index_add`` scatter on dimension zero."""

    out = values.new_zeros((int(size),) + tuple(values.shape[1:]))
    if values.numel():
        out.index_add_(0, index.long(), values)
    return out


def _max_abs(x) -> float:
    return float(x.detach().abs().max().cpu()) if x.numel() else 0.0


def _relative_error(lhs, rhs, eps: float = 1.0e-12) -> float:
    return _max_abs(lhs - rhs) / max(_max_abs(rhs), eps)


def _record_close(
    checks: dict[str, Any],
    key: str,
    lhs,
    rhs,
    *,
    atol: float = 2.0e-5,
    rtol: float = 2.0e-4,
) -> None:
    """Record a maximum error and fail immediately if an exact invariant breaks."""

    abs_err = _max_abs(lhs - rhs)
    rel_err = _relative_error(lhs, rhs)
    checks[key] = max(float(checks.get(key, 0.0)), abs_err)
    checks[f"{key}_relative"] = max(float(checks.get(f"{key}_relative", 0.0)), rel_err)
    if abs_err > atol + rtol * _max_abs(rhs):
        raise RuntimeError(
            f"effective-transport invariant {key!r} failed: max_abs={abs_err:.3e}, "
            f"relative={rel_err:.3e}, tolerance={atol:.1e}+{rtol:.1e}*scale"
        )


def grit_attention_components(module, batch, output) -> dict[str, Any]:
    """Reconstruct official GRIT's exact content and edge-enhanced pair messages.

    Parameters
    ----------
    module:
        A GRIT ``MultiHeadAttentionLayerGritSparse`` instance.
    batch:
        The (mutated) Batch passed to the module.  After ``forward`` it exposes
        ``V_h`` and post-softmax ``attn``.
    output:
        ``(h_out, e_out)`` from the module.  In official GRIT, ``e_out`` is the
        activated pair state ``e_t`` flattened over heads.

    Returns
    -------
    A tensor dictionary.  ``pair_value`` is the unweighted value including
    ``VeRow``; ``pair_message`` is attention weighted; ``wv_reconstructed`` must
    equal ``h_out``.  All tensors remain on the input device.
    """

    import torch

    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise TypeError("GRIT attention output must be a (h_out, e_out) pair")
    h_out, e_out = output[0], output[1]
    edge_index = batch.edge_index
    src, dst = edge_index[0].long(), edge_index[1].long()
    attn = _batch_get(batch, "attn")
    v_h = _batch_get(batch, "V_h")
    if attn is None or v_h is None:
        raise RuntimeError("GRIT attention hook requires batch.attn and batch.V_h")
    if attn.dim() == 3 and attn.size(-1) == 1:
        attn = attn.squeeze(-1)
    if attn.dim() != 2:
        raise RuntimeError(f"expected attention [E,H], got {tuple(attn.shape)}")
    if h_out.dim() != 3:
        raise RuntimeError(f"expected wV [N,H,dh], got {tuple(h_out.shape)}")
    e_count, heads = attn.shape
    if e_count != int(edge_index.size(1)) or heads != int(h_out.size(1)):
        raise RuntimeError("attention, support, and wV head geometry do not align")

    value_content = v_h[src]
    if tuple(value_content.shape) != (e_count, heads, int(h_out.size(2))):
        raise RuntimeError("batch.V_h geometry does not match the attention output")

    value_edge = torch.zeros_like(value_content)
    edge_enhance = bool(getattr(module, "edge_enhance", False))
    if edge_enhance and e_out is not None:
        if not hasattr(module, "VeRow"):
            raise RuntimeError("edge_enhance=True but the attention module has no VeRow")
        edge_state = e_out
        if edge_state.dim() == 2:
            edge_state = edge_state.reshape(e_count, heads, int(h_out.size(2)))
        elif edge_state.dim() != 3:
            raise RuntimeError(f"unexpected edge state shape {tuple(edge_state.shape)}")
        value_edge = torch.einsum("ehd,dhc->ehc", edge_state, module.VeRow)

    pair_value = value_content + value_edge
    pair_message_content = attn.unsqueeze(-1) * value_content
    pair_message_edge = attn.unsqueeze(-1) * value_edge
    pair_message = pair_message_content + pair_message_edge
    wv_content = _scatter_sum(pair_message_content, dst, int(h_out.size(0)))
    wv_edge = _scatter_sum(pair_message_edge, dst, int(h_out.size(0)))
    wv_reconstructed = wv_content + wv_edge

    return {
        "attention": attn,
        "edge_index": edge_index,
        "src": src,
        "dst": dst,
        "value_content": value_content,
        "value_edge": value_edge,
        "pair_value": pair_value,
        "pair_message_content": pair_message_content,
        "pair_message_edge": pair_message_edge,
        "pair_message": pair_message,
        "wv_content": wv_content,
        "wv_edge": wv_edge,
        "wv_reconstructed": wv_reconstructed,
        "wv": h_out,
    }


def support_marginal_static_attention(
    attention,
    edge_index,
    num_nodes: int,
    *,
    max_iter: int = 500,
    tol: float = 1.0e-9,
    return_diagnostics: bool = False,
):
    """Maximum-entropy routing null on the exact clean support and marginals.

    The returned edge-slot weights preserve, independently for every head:

    * each receiver's incoming attention mass; and
    * each source's total outgoing attention mass.

    Starting IPF from ones returns the maximum-entropy matrix on that support.
    On complete support with row sums one this is exactly the repeated clean
    column mean, ``X[i,j] = mean_i A[i,j]``.  On sparse support it removes learned
    pair interactions without inventing non-edges or changing source popularity.
    """

    import torch

    if attention.dim() == 3 and attention.size(-1) == 1:
        attention = attention.squeeze(-1)
    if attention.dim() != 2:
        raise ValueError("attention must have shape [E,H]")
    src, dst = edge_index[0].long(), edge_index[1].long()
    if int(attention.size(0)) != int(src.numel()):
        raise ValueError("attention and edge_index have different edge counts")
    if attention.numel() == 0:
        empty = attention.clone()
        diagnostics = {
            "iterations": 0,
            "max_destination_mass_error": 0.0,
            "max_source_mass_error": 0.0,
        }
        return (empty, diagnostics) if return_diagnostics else empty
    if not bool(torch.isfinite(attention).all()) or bool((attention < 0).any()):
        raise ValueError("attention must be finite and non-negative")

    # Float64 makes the hard marginal checks stable even when the model is fp32.
    work = attention.detach().to(dtype=torch.float64)
    target_dst = _scatter_sum(work, dst, num_nodes)
    target_src = _scatter_sum(work, src, num_nodes)
    x = torch.ones_like(work)
    eps = torch.finfo(work.dtype).tiny
    iterations = 0
    dst_err = src_err = float("inf")
    for iteration in range(int(max_iter)):
        dst_sum = _scatter_sum(x, dst, num_nodes)
        dst_scale = torch.where(
            target_dst > 0,
            target_dst / dst_sum.clamp_min(eps),
            torch.zeros_like(target_dst),
        )
        x = x * dst_scale[dst]

        src_sum = _scatter_sum(x, src, num_nodes)
        src_scale = torch.where(
            target_src > 0,
            target_src / src_sum.clamp_min(eps),
            torch.zeros_like(target_src),
        )
        x = x * src_scale[src]
        iterations = iteration + 1

        if iteration < 4 or iteration % 10 == 9:
            got_dst = _scatter_sum(x, dst, num_nodes)
            got_src = _scatter_sum(x, src, num_nodes)
            dst_err = _max_abs(got_dst - target_dst)
            src_err = _max_abs(got_src - target_src)
            if max(dst_err, src_err) <= tol:
                break

    got_dst = _scatter_sum(x, dst, num_nodes)
    got_src = _scatter_sum(x, src, num_nodes)
    dst_err = _max_abs(got_dst - target_dst)
    src_err = _max_abs(got_src - target_src)
    if max(dst_err, src_err) > max(10.0 * tol, 1.0e-7):
        raise RuntimeError(
            "support/marginal static-routing IPF did not converge: "
            f"dst_error={dst_err:.3e}, src_error={src_err:.3e}, iterations={iterations}"
        )

    result = x.to(dtype=attention.dtype)
    # Diagnostics describe the tensor actually returned to the caller, including
    # the final fp64 -> model-dtype cast.
    returned_dst_error = _max_abs(
        _scatter_sum(result, dst, num_nodes) - target_dst.to(result.dtype)
    )
    returned_src_error = _max_abs(
        _scatter_sum(result, src, num_nodes) - target_src.to(result.dtype)
    )
    diagnostics = {
        "iterations": int(iterations),
        "max_destination_mass_error": float(returned_dst_error),
        "max_source_mass_error": float(returned_src_error),
    }
    return (result, diagnostics) if return_diagnostics else result


def _dense_static_mean_error(
    attention, static_attention, edge_index, node_batch
) -> tuple[float, int]:
    """Return max dense-null error and number of complete graphs checked."""

    import torch

    src, dst = edge_index[0].long(), edge_index[1].long()
    graph_count = int(node_batch.max().item()) + 1 if node_batch.numel() else 0
    max_error, checked = 0.0, 0
    for graph in range(graph_count):
        nodes = torch.nonzero(node_batch == graph, as_tuple=False).flatten()
        n = int(nodes.numel())
        if n == 0:
            continue
        edge_mask = node_batch[dst] == graph
        eg_src, eg_dst = src[edge_mask], dst[edge_mask]
        if int(eg_src.numel()) != n * n:
            continue
        offset = int(nodes.min().item())
        keys = (eg_dst - offset) * n + (eg_src - offset)
        if int(torch.unique(keys).numel()) != n * n:
            continue
        clean = attention[edge_mask]
        source_mass = _scatter_sum(clean, eg_src - offset, n)
        expected = source_mass[eg_src - offset] / float(n)
        max_error = max(max_error, _max_abs(static_attention[edge_mask] - expected))
        checked += 1
    return float(max_error), int(checked)


def _routing_metrics(attention, static_attention, edge_index, node_batch, eps: float = 1.0e-12):
    """Per-graph/head Jensen-Shannon staticity and relative L2 displacement."""

    import torch

    dst = edge_index[1].long()
    graph_count = int(node_batch.max().item()) + 1 if node_batch.numel() else 0
    midpoint = 0.5 * (attention + static_attention)
    a = attention.clamp_min(eps)
    x = static_attention.clamp_min(eps)
    m = midpoint.clamp_min(eps)
    js_edge = 0.5 * attention * torch.log(a / m) + 0.5 * static_attention * torch.log(x / m)
    js_node = _scatter_sum(js_edge, dst, int(node_batch.numel())) / float(np.log(2.0))
    js_graph = _scatter_sum(js_node, node_batch, graph_count)
    counts = _scatter_sum(
        torch.ones_like(node_batch, dtype=attention.dtype).unsqueeze(-1),
        node_batch,
        graph_count,
    )
    js_graph = js_graph / counts.clamp_min(1.0)

    edge_graph = node_batch[dst]
    l2_num = _scatter_sum((attention - static_attention).pow(2), edge_graph, graph_count)
    l2_den = _scatter_sum(attention.pow(2), edge_graph, graph_count)
    l2_relative = l2_num / l2_den.clamp_min(eps)
    return js_graph, l2_relative


def graphwise_broadcast_residual(values, node_batch, num_graphs: Optional[int] = None):
    """Exact graphwise node-mean broadcast and centred relational residual."""

    if values.dim() != 3:
        raise ValueError("values must have shape [N,H,D]")
    if num_graphs is None:
        num_graphs = int(node_batch.max().item()) + 1 if node_batch.numel() else 0
    counts = _scatter_sum(
        values.new_ones((values.size(0), 1, 1)), node_batch.long(), int(num_graphs)
    )
    means = _scatter_sum(values, node_batch.long(), int(num_graphs)) / counts.clamp_min(1.0)
    broadcast = means[node_batch.long()]
    residual = values - broadcast
    return broadcast, residual, means, counts.reshape(int(num_graphs))


def _per_graph_energy(values, node_batch, num_graphs: int):
    energy_node = values.pow(2).sum(dim=-1)
    energy = _scatter_sum(energy_node, node_batch.long(), num_graphs)
    counts = _scatter_sum(
        values.new_ones((values.size(0), 1)), node_batch.long(), num_graphs
    )
    return energy / counts.clamp_min(1.0)


def _project_heads(u, output_weight, output_scale=None):
    """Project each head with its exact block of a Linear layer's weight."""

    if u.dim() != 3:
        raise ValueError("u must be [N,H,dh]")
    n, heads, head_dim = u.shape
    if int(output_weight.size(1)) != heads * head_dim:
        raise ValueError("O_h input width does not match [H,dh]")
    blocks = output_weight.reshape(output_weight.size(0), heads, head_dim).permute(1, 2, 0)
    import torch
    projected = torch.einsum("nhd,hdo->nho", u, blocks)
    if output_scale is not None:
        projected = projected * output_scale
    return projected


def effective_transport_decomposition(
    u,
    node_batch,
    output_weight,
    *,
    output_bias=None,
    u_content=None,
    u_edge=None,
    u_static=None,
    output_scale=None,
) -> dict[str, Any]:
    """Decompose effective pre-``O_h`` transport and its per-head projection.

    ``u_content`` and ``u_edge`` are required for value/edge attribution and must
    sum to ``u``.  ``u_static`` is the effective transport recomputed under the
    support/marginal static-routing null.
    """

    import torch

    graph_count = int(node_batch.max().item()) + 1 if node_batch.numel() else 0
    b_u, r_u, _, _ = graphwise_broadcast_residual(u, node_batch, graph_count)
    t = _project_heads(u, output_weight, output_scale)
    b_t, r_t, _, _ = graphwise_broadcast_residual(t, node_batch, graph_count)

    result: dict[str, Any] = {
        "u": u,
        "u_broadcast": b_u,
        "u_residual": r_u,
        "t": t,
        "t_broadcast": b_t,
        "t_residual": r_t,
        "total_energy": _per_graph_energy(t, node_batch, graph_count),
        "relational_energy": _per_graph_energy(r_t, node_batch, graph_count),
    }
    if output_bias is not None:
        result["oh_reconstructed"] = t.sum(dim=1) + output_bias
    else:
        result["oh_reconstructed"] = t.sum(dim=1)

    if u_content is not None or u_edge is not None:
        if u_content is None or u_edge is None:
            raise ValueError("u_content and u_edge must be supplied together")
        t_content = _project_heads(u_content, output_weight, output_scale)
        t_edge = _project_heads(u_edge, output_weight, output_scale)
        _, r_content, _, _ = graphwise_broadcast_residual(t_content, node_batch, graph_count)
        _, r_edge, _, _ = graphwise_broadcast_residual(t_edge, node_batch, graph_count)
        cross_node = 2.0 * (r_content * r_edge).sum(dim=-1)
        counts = _scatter_sum(t.new_ones((t.size(0), 1)), node_batch.long(), graph_count)
        cross = _scatter_sum(cross_node, node_batch.long(), graph_count) / counts.clamp_min(1.0)
        result.update({
            "t_content": t_content,
            "t_edge": t_edge,
            "t_content_residual": r_content,
            "t_edge_residual": r_edge,
            "content_relational_energy": _per_graph_energy(r_content, node_batch, graph_count),
            "edge_relational_energy": _per_graph_energy(r_edge, node_batch, graph_count),
            "content_edge_cross_term": cross,
        })

    if u_static is not None:
        t_static = _project_heads(u_static, output_weight, output_scale)
        _, r_static, _, _ = graphwise_broadcast_residual(t_static, node_batch, graph_count)
        result.update({
            "t_static": t_static,
            "t_static_residual": r_static,
            "static_relational_energy": _per_graph_energy(r_static, node_batch, graph_count),
        })
    return result


def _degree_scaled(layer, batch, wv):
    """Apply the exact eval-time transformation between ``wV`` and ``O_h``."""

    import torch

    if bool(getattr(layer, "training", False)):
        raise RuntimeError("effective transport requires model.eval(); dropout must be disabled")
    flat = wv.reshape(wv.size(0), -1)
    if not bool(getattr(layer, "deg_scaler", False)):
        return flat.reshape_as(wv)
    log_deg = _batch_get(batch, "log_deg")
    if log_deg is None:
        deg = _batch_get(batch, "deg")
        if deg is not None:
            log_deg = torch.log(deg.to(flat.dtype) + 1.0).reshape(-1, 1)
        else:
            dst = batch.edge_index[1].long()
            deg = flat.new_zeros((flat.size(0),))
            deg.index_add_(0, dst, flat.new_ones((dst.numel(),)))
            log_deg = torch.log(deg + 1.0).reshape(-1, 1)
    else:
        log_deg = log_deg.to(device=flat.device, dtype=flat.dtype).reshape(-1, 1)
    coef = layer.deg_coef.to(device=flat.device, dtype=flat.dtype)
    scaled = flat * (coef[..., 0] + log_deg * coef[..., 1])
    return scaled.reshape_as(wv)


def apply_effective_treatment(
    u,
    node_batch,
    heads: Sequence[int],
    treatment: str,
    *,
    u_content=None,
    u_edge=None,
    graph_ids: Optional[Sequence[int]] = None,
    layer_index: int = 0,
    permutation_index: int = 0,
    seed: int = 0,
):
    """Apply an exact causal treatment to selected head slices at the effective site."""

    import torch

    if treatment not in {
        "broadcast_only", "residual_permuted", "full_head_zero",
        "remove_content_residual", "remove_edge_residual", "remove_both_residual",
    }:
        raise ValueError(f"unsupported effective-site treatment {treatment!r}")
    selected = sorted({int(head) for head in heads})
    if not selected:
        return u
    if min(selected) < 0 or max(selected) >= int(u.size(1)):
        raise IndexError("head index outside U's head dimension")

    graph_count = int(node_batch.max().item()) + 1 if node_batch.numel() else 0
    b_u, r_u, _, _ = graphwise_broadcast_residual(u, node_batch, graph_count)
    out = u.clone()
    if treatment == "full_head_zero":
        out[:, selected, :] = 0.0
    elif treatment == "broadcast_only":
        out[:, selected, :] = b_u[:, selected, :]
    elif treatment == "residual_permuted":
        ids = list(range(graph_count)) if graph_ids is None else [int(x) for x in graph_ids]
        if len(ids) != graph_count:
            raise ValueError("graph_ids must align with the graphs in node_batch")
        for graph, graph_id in enumerate(ids):
            nodes = torch.nonzero(node_batch == graph, as_tuple=False).flatten()
            n = int(nodes.numel())
            if n <= 1:
                out[nodes[:, None], torch.as_tensor(selected, device=u.device)[None, :], :] = \
                    b_u[nodes[:, None], torch.as_tensor(selected, device=u.device)[None, :], :]
                continue
            for head in selected:
                # Stable non-zero cyclic shifts have no fixed points and preserve the
                # exact residual multiset, mean, and Frobenius norm.
                key = (
                    int(seed) * 1_000_003
                    + int(permutation_index) * 9_176
                    + int(graph_id) * 1_315_423_911
                    + int(layer_index) * 2_654_435_761
                    + int(head) * 97_531
                )
                shift = 1 + (key % (n - 1))
                out[nodes, head, :] = b_u[nodes, head, :] + torch.roll(
                    r_u[nodes, head, :], shifts=int(shift), dims=0
                )
    else:
        if u_content is None or u_edge is None:
            raise ValueError(f"{treatment} requires u_content and u_edge")
        _, r_content, _, _ = graphwise_broadcast_residual(u_content, node_batch, graph_count)
        _, r_edge, _, _ = graphwise_broadcast_residual(u_edge, node_batch, graph_count)
        if treatment == "remove_content_residual":
            out[:, selected, :] = u[:, selected, :] - r_content[:, selected, :]
        elif treatment == "remove_edge_residual":
            out[:, selected, :] = u[:, selected, :] - r_edge[:, selected, :]
        else:
            out[:, selected, :] = (
                u[:, selected, :] - r_content[:, selected, :] - r_edge[:, selected, :]
            )
    return out


def paired_bootstrap_summary(values, *, replicates: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Equal-graph paired bootstrap interval for a per-graph causal contrast."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_graphs": 0, "bootstrap_replicates": int(replicates)}
    point = float(values.mean())
    if int(replicates) <= 0:
        low = high = float("nan")
    else:
        rng = np.random.default_rng(int(seed))
        means = np.empty(int(replicates), dtype=np.float64)
        for draw in range(int(replicates)):
            means[draw] = values[rng.integers(0, values.size, size=values.size)].mean()
        low, high = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    return {"mean": point, "ci_low": low, "ci_high": high, "n_graphs": int(values.size),
            "bootstrap_replicates": int(replicates)}


class _TransportHookSession(AbstractContextManager):
    """One-forward hook scope.  Hooks are always removed by ``__exit__``."""

    def __init__(
        self,
        gm,
        batch,
        graph_ids: Sequence[int],
        *,
        treatment: Optional[str],
        heads_by_layer: Mapping[int, Sequence[int]],
        capture_metrics: bool,
        permutation_index: int,
        seed: int,
        checks: dict[str, Any],
    ) -> None:
        self.gm = gm
        self.batch = batch
        self.graph_ids = [int(x) for x in graph_ids]
        self.treatment = treatment
        self.heads_by_layer = {int(k): [int(h) for h in v] for k, v in heads_by_layer.items()}
        self.capture_metrics = bool(capture_metrics)
        self.permutation_index = int(permutation_index)
        self.seed = int(seed)
        self.checks = checks
        self.handles = []
        self.state: dict[int, dict[str, Any]] = {}
        self.metrics: dict[int, dict[str, Any]] = {}
        self.layers = list(gm.model.model.layers)
        if len(self.layers) != int(gm.L):
            raise RuntimeError("GRIT layer list does not match gm.L")
        self.node_batch = _batch_get(batch, "batch")
        if self.node_batch is None:
            import torch
            self.node_batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.x.device)

    def __enter__(self):
        needs_pairs = self.capture_metrics or self.treatment in {
            "static_routing", "remove_content_residual", "remove_edge_residual"
        }
        for layer_index, layer in enumerate(self.layers):
            if needs_pairs:
                self.handles.append(
                    layer.attention.register_forward_hook(
                        self._attention_hook(layer_index, layer)
                    )
                )
            self.handles.append(
                layer.O_h.register_forward_pre_hook(self._oh_pre_hook(layer_index, layer))
            )
            self.handles.append(layer.O_h.register_forward_hook(self._oh_post_hook(layer_index)))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False

    def _attention_hook(self, layer_index, layer):
        def hook(module, inputs, output):
            batch = inputs[0]
            components = grit_attention_components(module, batch, output)
            _record_close(
                self.checks, "max_wv_reconstruction_error",
                components["wv_reconstructed"], components["wv"],
            )
            incoming_mass = _scatter_sum(
                components["attention"], components["dst"], int(components["wv"].size(0))
            )
            attention_norm_error = _max_abs(
                incoming_mass - incoming_mass.new_ones(incoming_mass.shape)
            )
            self.checks["max_attention_destination_normalization_error"] = max(
                float(self.checks.get("max_attention_destination_normalization_error", 0.0)),
                attention_norm_error,
            )
            if attention_norm_error > 5.0e-5:
                raise RuntimeError(
                    "captured eval-time attention is not receiver-normalised; "
                    f"max error={attention_norm_error:.3e}"
                )
            state = {"components": components, "batch": batch}
            state["u_content"] = _degree_scaled(layer, batch, components["wv_content"])
            state["u_edge"] = _degree_scaled(layer, batch, components["wv_edge"])

            need_static = self.capture_metrics or self.treatment == "static_routing"
            if need_static:
                static, diagnostics = support_marginal_static_attention(
                    components["attention"],
                    components["edge_index"],
                    int(components["wv"].size(0)),
                    return_diagnostics=True,
                )
                static_content_pair = static.unsqueeze(-1) * components["value_content"]
                static_edge_pair = static.unsqueeze(-1) * components["value_edge"]
                wv_static_content = _scatter_sum(
                    static_content_pair, components["dst"], int(components["wv"].size(0))
                )
                wv_static_edge = _scatter_sum(
                    static_edge_pair, components["dst"], int(components["wv"].size(0))
                )
                state.update({
                    "static_attention": static,
                    "wv_static_content": wv_static_content,
                    "wv_static_edge": wv_static_edge,
                    "wv_static": wv_static_content + wv_static_edge,
                    "u_static": _degree_scaled(layer, batch, wv_static_content + wv_static_edge),
                })
                self.checks["max_static_destination_mass_error"] = max(
                    float(self.checks.get("max_static_destination_mass_error", 0.0)),
                    float(diagnostics["max_destination_mass_error"]),
                )
                self.checks["max_static_source_mass_error"] = max(
                    float(self.checks.get("max_static_source_mass_error", 0.0)),
                    float(diagnostics["max_source_mass_error"]),
                )
                if max(
                    float(diagnostics["max_destination_mass_error"]),
                    float(diagnostics["max_source_mass_error"]),
                ) > 5.0e-5:
                    raise RuntimeError(
                        "returned static-routing weights do not preserve clean marginals"
                    )
                dense_error, dense_checked = _dense_static_mean_error(
                    components["attention"], static, components["edge_index"], self.node_batch
                )
                self.checks["max_dense_static_mean_error"] = max(
                    float(self.checks.get("max_dense_static_mean_error", 0.0)), dense_error
                )
                self.checks["dense_graphs_checked"] = (
                    int(self.checks.get("dense_graphs_checked", 0)) + dense_checked
                )
                if dense_error > 5.0e-5:
                    raise RuntimeError(
                        "dense static-routing null != repeated column mean: "
                        f"{dense_error:.3e}"
                    )
                if self.capture_metrics:
                    js, l2 = _routing_metrics(
                        components["attention"], static, components["edge_index"], self.node_batch
                    )
                    state["routing_js_static"] = js
                    state["routing_l2_relative"] = l2

            self.state[layer_index] = state
            if self.treatment == "static_routing":
                selected = self.heads_by_layer.get(layer_index, [])
                if selected:
                    h_out, e_out = output
                    patched = h_out.clone()
                    patched[:, selected, :] = state["wv_static"][:, selected, :]
                    # e_out is deliberately untouched: this intervention changes node
                    # routing, not pair-state evolution.
                    return patched, e_out
            return None
        return hook

    def _oh_pre_hook(self, layer_index, layer):
        def hook(module, inputs):
            import torch

            u_flat = inputs[0]
            u = u_flat.reshape(u_flat.size(0), int(self.gm.H), int(self.gm.dh))
            selected = self.heads_by_layer.get(layer_index, [])
            state = self.state.setdefault(layer_index, {})

            if "u_content" in state:
                expected = state["u_content"] + state["u_edge"]
                if self.treatment == "static_routing" and selected:
                    expected = expected.clone()
                    expected[:, selected, :] = state["u_static"][:, selected, :]
                _record_close(self.checks, "max_u_reconstruction_error", expected, u)

            if self.capture_metrics:
                if self.treatment is not None:
                    raise RuntimeError("mechanism metrics must be captured on a clean forward")
                scale = (
                    getattr(layer, "alpha1_h", None)
                    if bool(getattr(layer, "rezero", False))
                    else None
                )
                decomposition = effective_transport_decomposition(
                    u, self.node_batch, module.weight,
                    output_bias=module.bias,
                    u_content=state["u_content"], u_edge=state["u_edge"],
                    u_static=state["u_static"], output_scale=scale,
                )
                _record_close(
                    self.checks, "max_broadcast_reconstruction_error",
                    decomposition["u_broadcast"] + decomposition["u_residual"], u,
                )
                residual_sum = _scatter_sum(
                    decomposition["u_residual"], self.node_batch,
                    len(self.graph_ids),
                )
                self.checks["max_residual_sum_error"] = max(
                    float(self.checks.get("max_residual_sum_error", 0.0)), _max_abs(residual_sum)
                )
                if _max_abs(residual_sum) > 5.0e-5:
                    raise RuntimeError("graphwise relational residual is not centred")
                _record_close(
                    self.checks, "max_component_identity_error",
                    state["u_content"] + state["u_edge"], u,
                )
                removed_both = apply_effective_treatment(
                    u, self.node_batch, list(range(int(self.gm.H))), "remove_both_residual",
                    u_content=state["u_content"], u_edge=state["u_edge"],
                )
                _record_close(
                    self.checks, "max_component_removal_broadcast_parity_error",
                    removed_both, decomposition["u_broadcast"],
                )
                rel_identity = (
                    decomposition["content_relational_energy"]
                    + decomposition["edge_relational_energy"]
                    + decomposition["content_edge_cross_term"]
                )
                _record_close(
                    self.checks, "max_energy_identity_error",
                    rel_identity, decomposition["relational_energy"], atol=5.0e-5, rtol=5.0e-4,
                )
                # Orthogonality of broadcast and residual after projection.
                total_identity = (
                    _per_graph_energy(
                        decomposition["t_broadcast"], self.node_batch, len(self.graph_ids)
                    )
                    + decomposition["relational_energy"]
                )
                _record_close(
                    self.checks, "max_projected_energy_identity_error",
                    total_identity, decomposition["total_energy"], atol=5.0e-5, rtol=5.0e-4,
                )
                self.metrics[layer_index] = {
                    "routing_js_static": state["routing_js_static"],
                    "routing_l2_relative": state["routing_l2_relative"],
                    "effective_total_energy": decomposition["total_energy"],
                    "effective_relational_energy": decomposition["relational_energy"],
                    "static_effective_relational_energy": decomposition["static_relational_energy"],
                    "content_relational_energy": decomposition["content_relational_energy"],
                    "edge_relational_energy": decomposition["edge_relational_energy"],
                    "content_edge_cross_term": decomposition["content_edge_cross_term"],
                }

            patched = u
            if self.treatment in {
                "broadcast_only", "residual_permuted", "full_head_zero",
                "remove_content_residual", "remove_edge_residual",
            } and selected:
                patched = apply_effective_treatment(
                    u, self.node_batch, selected, self.treatment,
                    u_content=state.get("u_content"), u_edge=state.get("u_edge"),
                    graph_ids=self.graph_ids, layer_index=layer_index,
                    permutation_index=self.permutation_index, seed=self.seed,
                )
                if self.treatment == "residual_permuted":
                    before_b, before_r, _, _ = graphwise_broadcast_residual(
                        u, self.node_batch, len(self.graph_ids)
                    )
                    after_b, after_r, _, _ = graphwise_broadcast_residual(
                        patched, self.node_batch, len(self.graph_ids)
                    )
                    selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=u.device)
                    _record_close(
                        self.checks, "max_residual_permutation_mean_error",
                        after_b[:, selected_tensor, :], before_b[:, selected_tensor, :],
                    )
                    before_energy = _per_graph_energy(
                        before_r[:, selected_tensor, :], self.node_batch, len(self.graph_ids)
                    )
                    after_energy = _per_graph_energy(
                        after_r[:, selected_tensor, :], self.node_batch, len(self.graph_ids)
                    )
                    _record_close(
                        self.checks, "max_residual_permutation_energy_error",
                        after_energy, before_energy,
                    )
            state["u_after_intervention"] = patched
            scale = (
                getattr(layer, "alpha1_h", None)
                if bool(getattr(layer, "rezero", False))
                else None
            )
            state["oh_reconstructed"] = _project_heads(patched, module.weight, scale).sum(dim=1)
            if module.bias is not None:
                # Bias is inside O_h, before optional ReZero.  ReZero scales the whole
                # O_h output, so only use the unscaled exact O_h check here.
                state["oh_reconstructed_unscaled"] = (
                    _project_heads(patched, module.weight).sum(dim=1) + module.bias
                )
            else:
                state["oh_reconstructed_unscaled"] = _project_heads(
                    patched, module.weight
                ).sum(dim=1)
            if patched is not u:
                return (patched.reshape_as(u_flat),)
            return None
        return hook

    def _oh_post_hook(self, layer_index):
        def hook(module, inputs, output):
            state = self.state[layer_index]
            _record_close(
                self.checks, "max_oh_reconstruction_error",
                state["oh_reconstructed_unscaled"], output,
            )
            return None
        return hook

    def exported_metrics(self) -> dict[str, np.ndarray]:
        if not self.capture_metrics:
            return {}
        if len(self.metrics) != int(self.gm.L):
            raise RuntimeError("not every GRIT layer produced effective-transport metrics")
        keys = tuple(self.metrics[0].keys())
        return {
            key: np.stack([
                self.metrics[layer][key].detach().cpu().numpy()
                for layer in range(int(self.gm.L))
            ], axis=1)  # [graphs, layers, heads]
            for key in keys
        }


def _heads_by_layer(heads: Sequence[Sequence[int]], layers: int, heads_per_layer: int):
    result: dict[int, list[int]] = {}
    for pair in heads:
        if len(pair) != 2:
            raise ValueError(f"head must be (layer, head), got {pair!r}")
        layer, head = int(pair[0]), int(pair[1])
        if not (0 <= layer < layers and 0 <= head < heads_per_layer):
            raise IndexError(
                f"head {(layer, head)} outside model geometry {(layers, heads_per_layer)}"
            )
        if head not in result.setdefault(layer, []):
            result[layer].append(head)
    return result


def _extract_ranking(value) -> list[tuple[int, int]]:
    if isinstance(value, Mapping):
        for key in ("ranking", "ordered_heads", "heads", "selected_heads"):
            if key in value:
                return _extract_ranking(value[key])
        # A mapping from k to an explicit nested top-k set: the largest set is a
        # valid order when it was constructed as a prefix (the mediation API does so).
        numeric = []
        for key, heads in value.items():
            try:
                numeric.append((int(key), heads))
            except (TypeError, ValueError):
                continue
        if numeric:
            return _extract_ranking(max(numeric, key=lambda item: item[0])[1])
        raise ValueError("head-selection mapping has no recognised ranking field")
    ranking = []
    for head in value:
        if len(head) != 2:
            raise ValueError(f"invalid head entry {head!r}")
        pair = (int(head[0]), int(head[1]))
        if pair not in ranking:
            ranking.append(pair)
    return ranking


def _normalise_head_selection(head_selection: Mapping[str, Any], layers: int, heads: int):
    if not isinstance(head_selection, Mapping) or not head_selection:
        raise ValueError("head_selection must be a non-empty mapping of named ordered rankings")
    rankings: dict[str, list[tuple[int, int]]] = {}
    for name, value in head_selection.items():
        ranking = _extract_ranking(value)
        _heads_by_layer(ranking, layers, heads)
        if not ranking:
            raise ValueError(f"head-selection group {name!r} is empty")
        rankings[str(name)] = ranking
    return rankings


def _build_confirmation_groups(gm, graph_ids, batch_size: int):
    ids = np.asarray(graph_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("graph_ids must be a non-empty sequence of unique confirmation ids")
    if ids.min() < 0 or ids.max() >= len(gm.eval_ds):
        raise IndexError("confirmation graph id outside gm.eval_ds")
    groups, id_groups, ys = [], [], []
    for start in range(0, len(ids), int(batch_size)):
        chunk = ids[start:start + int(batch_size)]
        data = [gm.eval_ds[int(graph_id)] for graph_id in chunk]
        groups.append(data)
        id_groups.append([int(x) for x in chunk])
        ys.extend([d.y.reshape(-1).detach().cpu().numpy().astype(np.float64) for d in data])
    return groups, id_groups, np.stack(ys), ids


def _predict_with_treatment(
    gm,
    groups,
    id_groups,
    *,
    treatment: Optional[str],
    heads: Sequence[Sequence[int]],
    capture_metrics: bool,
    permutation_index: int,
    seed: int,
    checks: dict[str, Any],
):
    import torch
    from torch_geometric.data import Batch

    if bool(gm.model.training):
        raise RuntimeError("run_effective_transport requires a loaded model in eval mode")
    by_layer = _heads_by_layer(heads, int(gm.L), int(gm.H))
    predictions, metric_chunks = [], []
    for data_group, graph_id_group in zip(groups, id_groups):
        # Load-bearing: GRIT mutates Batch.x/edge_attr/edge_index in place.
        batch = Batch.from_data_list(list(data_group)).to(gm.device)
        session = _TransportHookSession(
            gm, batch, graph_id_group,
            treatment=treatment, heads_by_layer=by_layer,
            capture_metrics=capture_metrics, permutation_index=permutation_index,
            seed=seed, checks=checks,
        )
        with session:
            with torch.no_grad():
                pred, _ = gm.model(batch)
        predictions.append(pred.detach().cpu().numpy().reshape(len(graph_id_group), -1))
        if capture_metrics:
            metric_chunks.append(session.exported_metrics())
    pred_array = np.concatenate(predictions, axis=0)
    metrics = {}
    if capture_metrics:
        for key in metric_chunks[0]:
            metrics[key] = np.concatenate([chunk[key] for chunk in metric_chunks], axis=0)
    return pred_array, metrics


def _causal_effect(
    predictions,
    clean_predictions,
    targets,
    *,
    bootstrap_replicates: int,
    seed: int,
    loss_delta_override=None,
    functional_override=None,
) -> dict[str, Any]:
    pred = np.asarray(predictions, dtype=np.float64)
    clean = np.asarray(clean_predictions, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    clean_mae = np.abs(clean - y).mean(axis=1)
    delta_mae = np.abs(pred - y).mean(axis=1) - clean_mae
    functional = np.linalg.norm(pred - clean, axis=1)
    if loss_delta_override is not None:
        delta_mae = np.asarray(loss_delta_override, dtype=np.float64)
    if functional_override is not None:
        functional = np.asarray(functional_override, dtype=np.float64)
    return {
        "pred": pred,
        "delta_mae_per_graph": delta_mae,
        "abs_delta_pred_per_graph": functional,
        "delta_mae": paired_bootstrap_summary(
            delta_mae, replicates=bootstrap_replicates, seed=seed
        ),
        "abs_delta_pred": paired_bootstrap_summary(
            functional, replicates=bootstrap_replicates, seed=seed + 1
        ),
    }


def _head_metrics(metric_arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    mean = {key: np.nanmean(np.asarray(value, dtype=np.float64), axis=0)
            for key, value in metric_arrays.items()}
    eps = 1.0e-12
    mean["effective_relational_fraction"] = np.divide(
        mean["effective_relational_energy"], mean["effective_total_energy"],
        out=np.full_like(mean["effective_total_energy"], np.nan),
        where=mean["effective_total_energy"] > eps,
    )
    mean["static_relational_retention"] = np.divide(
        mean["static_effective_relational_energy"], mean["effective_relational_energy"],
        out=np.full_like(mean["effective_relational_energy"], np.nan),
        where=mean["effective_relational_energy"] > eps,
    )
    return mean


def run_effective_transport(
    result,
    sc,
    *,
    graph_ids,
    head_selection,
    topk=(1, 2, 4, 8),
    primary_k=4,
    residual_permutations=4,
    bootstrap_replicates=1000,
    seed=0,
):
    """Run held-out static-routing/effective-transport analysis on ZINC.

    ``graph_ids`` are used exactly as supplied and are never resampled.  Head
    rankings are supplied by the mediation discovery analysis; this function
    performs no score-based reselection on the confirmation graphs.
    """

    from ..carriage.env import log

    gm = result.get("gm")
    if gm is None:
        raise ValueError("result must retain the loaded model as result['gm']")
    rankings = _normalise_head_selection(head_selection, int(gm.L), int(gm.H))
    topk = tuple(sorted({int(k) for k in topk if int(k) > 0}))
    if not topk:
        raise ValueError("topk must contain at least one positive integer")
    if int(primary_k) <= 0:
        raise ValueError("primary_k must be positive")
    if int(residual_permutations) <= 0:
        raise ValueError("residual_permutations must be positive")

    batch_size = int(getattr(sc, "effective_transport_batch_size", 32))
    groups, id_groups, targets, graph_ids_array = _build_confirmation_groups(
        gm, graph_ids, batch_size
    )
    checks: dict[str, Any] = {
        "passed": False,
        "confirmation_graphs_only": True,
        "fresh_batch_per_forward": True,
        "hooks_scoped_try_finally": True,
    }
    log(f"[transport] {len(graph_ids_array)} confirmation graphs; "
        f"groups={list(rankings)}; top-k={list(topk)}")

    clean_pred, per_graph_metrics = _predict_with_treatment(
        gm, groups, id_groups, treatment=None, heads=[], capture_metrics=True,
        permutation_index=0, seed=seed, checks=checks,
    )
    clean_mae = np.abs(clean_pred - targets).mean(axis=1)
    head_metrics = _head_metrics(per_graph_metrics)

    # Eval-mode GRIT should be graph-batch invariant (BatchNorm uses running
    # statistics).  Re-run one molecule alone to catch accidental train mode,
    # cross-graph support, or a hook that groups nodes incorrectly.
    solo_pred, solo_metrics = _predict_with_treatment(
        gm, [[groups[0][0]]], [[id_groups[0][0]]], treatment=None, heads=[],
        capture_metrics=True, permutation_index=0, seed=seed, checks=checks,
    )
    batch_pred_error = float(np.max(np.abs(solo_pred[0] - clean_pred[0])))
    batch_metric_error = max(
        float(np.nanmax(np.abs(solo_metrics[key][0] - per_graph_metrics[key][0])))
        for key in per_graph_metrics
    )
    checks["max_batch_context_prediction_error"] = batch_pred_error
    checks["max_batch_context_metric_error"] = batch_metric_error
    if batch_pred_error > 5.0e-5 or batch_metric_error > 5.0e-5:
        raise RuntimeError(
            "effective transport changed when a confirmation graph was evaluated alone: "
            f"prediction={batch_pred_error:.3e}, metrics={batch_metric_error:.3e}"
        )

    causal: dict[str, Any] = {}
    resolved_selection: dict[str, Any] = {}
    parity_max = 0.0
    for group_index, (group_name, ranking) in enumerate(rankings.items()):
        causal[group_name] = {}
        resolved_selection[group_name] = {}
        for k_index, k in enumerate(topk):
            selected = ranking[:min(int(k), len(ranking))]
            if not selected:
                continue
            key = str(int(k))
            selected_json = [[int(layer), int(head)] for layer, head in selected]
            resolved_selection[group_name][key] = selected_json
            causal[group_name][key] = {}
            base_seed = int(seed) + 100_003 * group_index + 1_009 * k_index

            for treatment_index, treatment in enumerate(CAUSAL_TREATMENTS):
                if treatment == "residual_permuted":
                    permutation_predictions, permutation_losses, permutation_functional = [], [], []
                    for permutation in range(int(residual_permutations)):
                        pred, _ = _predict_with_treatment(
                            gm, groups, id_groups, treatment=treatment, heads=selected,
                            capture_metrics=False, permutation_index=permutation,
                            seed=seed, checks=checks,
                        )
                        permutation_predictions.append(pred)
                        permutation_losses.append(np.abs(pred - targets).mean(axis=1) - clean_mae)
                        permutation_functional.append(np.linalg.norm(pred - clean_pred, axis=1))
                    pred = np.mean(permutation_predictions, axis=0)
                    loss_override = np.mean(permutation_losses, axis=0)
                    functional_override = np.mean(permutation_functional, axis=0)
                    effect = _causal_effect(
                        pred, clean_pred, targets,
                        bootstrap_replicates=bootstrap_replicates,
                        seed=base_seed + treatment_index * 17,
                        loss_delta_override=loss_override,
                        functional_override=functional_override,
                    )
                    effect["prediction_sd_across_permutations"] = np.std(
                        permutation_predictions, axis=0
                    )
                    effect["residual_permutations"] = int(residual_permutations)
                else:
                    pred, _ = _predict_with_treatment(
                        gm, groups, id_groups, treatment=treatment, heads=selected,
                        capture_metrics=False, permutation_index=0,
                        seed=seed, checks=checks,
                    )
                    effect = _causal_effect(
                        pred, clean_pred, targets,
                        bootstrap_replicates=bootstrap_replicates,
                        seed=base_seed + treatment_index * 17,
                    )
                effect["heads"] = selected_json
                effect["requested_k"] = int(k)
                effect["actual_k"] = int(len(selected))
                causal[group_name][key][treatment] = effect

            # Hard parity: zeroing at post-degree-scaler U must equal the existing
            # wV-head ablation path exactly.
            reference_zero = gm.collect_preds_ablated(groups, selected).reshape(clean_pred.shape)
            ours_zero = causal[group_name][key]["full_head_zero"]["pred"]
            parity = float(np.max(np.abs(reference_zero - ours_zero)))
            parity_max = max(parity_max, parity)
            if parity > 5.0e-5:
                raise RuntimeError(
                    "effective-site full-head zero does not match existing wV ablation: "
                    f"{parity:.3e}"
                )

    checks["max_full_zero_parity_error"] = float(parity_max)
    checks["passed"] = True

    primary_available = {}
    for name in rankings:
        keys = [int(key) for key in resolved_selection[name]]
        primary_available[name] = str(
            int(primary_k) if int(primary_k) in keys
            else min(keys, key=lambda value: (abs(value - int(primary_k)), value))
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "graph_ids": graph_ids_array,
        "topk": np.asarray(topk, dtype=np.int64),
        "primary_k": int(primary_k),
        "primary_available_key": primary_available,
        "head_selection": resolved_selection,
        "head_metrics": head_metrics,
        "head_metrics_per_graph": per_graph_metrics,
        "causal": causal,
        "clean": {
            "pred": clean_pred,
            "y": targets,
            "mae_per_graph": clean_mae,
            "mae": float(clean_mae.mean()),
        },
        "checks": checks,
        "config": {
            "confirmation_only": True,
            "num_confirmation_graphs": int(len(graph_ids_array)),
            "batch_size": int(batch_size),
            "topk": [int(k) for k in topk],
            "primary_k": int(primary_k),
            "residual_permutations": int(residual_permutations),
            "bootstrap_replicates": int(bootstrap_replicates),
            "seed": int(seed),
            "treatments": list(CAUSAL_TREATMENTS),
            "loss_endpoint": "paired signed delta MAE (ZINC units)",
            "functional_endpoint": "L2 prediction displacement",
        },
    }


__all__ = [
    "CAUSAL_TREATMENTS",
    "SCHEMA_VERSION",
    "apply_effective_treatment",
    "effective_transport_decomposition",
    "graphwise_broadcast_residual",
    "grit_attention_components",
    "paired_bootstrap_summary",
    "run_effective_transport",
    "support_marginal_static_attention",
]
