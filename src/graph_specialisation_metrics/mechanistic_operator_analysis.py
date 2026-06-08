"""Mechanistic operator-transport analysis for trained GraphBench models.

This is intentionally separate from ``core_interpretability_specialisation_metrics``.
The core metric script estimates model-agnostic equivariance/invariance scores under
swaps. This script asks a complementary mechanistic question: where does a trained
head send task-relevant realised transport?

The implemented first-class adapter is official GRIT/static-GRIT. It records, for
each layer/head, the post-softmax routing matrix ``A``, transported pair message
``m``, realised contribution ``C = A*m``, and a scalar task-gradient influence

    I_ij = | < d s / d o_i, C_ij > |,

where ``o_i = sum_j C_ij`` is the head output at receiver ``i``. The reductions are
streamed over graph batches and written to CSV plus paper-oriented figures.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn


EPS = 1.0e-12
PRIMARY_OPERATOR_EXCLUDE_PREFIXES = ("random__",)


def import_module_from_path(path: Path, module_name: str):
    path = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def find_default_runner_path() -> Path:
    candidates: list[Path] = []
    cwd = Path.cwd()
    candidates.append(cwd / "graphbench-algoreas-hpc" / "bin" / "algoreas_hpc.py")
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "graphbench-algoreas-hpc" / "bin" / "algoreas_hpc.py")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("could not find graphbench AlgoReas HPC runner; pass --runner-path")


def load_runner(path: Optional[Path]):
    runner_path = path if path is not None else find_default_runner_path()
    runner_path = runner_path.expanduser().resolve()
    os.environ.setdefault("PROJECT_ROOT", str(runner_path.parents[1]))
    return import_module_from_path(runner_path, "_graphbench_algoreas_hpc_runner")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = torch.device(device)
    if out.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def checkpoint_config_kwargs(runner: Any, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    signature = checkpoint.get("run_signature", {})
    cfg = signature.get("config", {}) if isinstance(signature, Mapping) else {}
    if not isinstance(cfg, Mapping):
        return {}
    allowed = {field.name for field in dataclasses.fields(runner.ScreenConfig)}
    return {key: value for key, value in cfg.items() if key in allowed}


def build_screen_config(
    runner: Any,
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
) -> Any:
    values = checkpoint_config_kwargs(runner, checkpoint)
    explicit = {
        "split_seed": args.split_seed,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "train_node_size": args.train_node_size,
        "val_node_size": args.val_node_size,
        "test_node_size": args.test_node_size,
        "seed": args.model_seed,
    }
    values.update({key: value for key, value in explicit.items() if value is not None})
    return runner.ScreenConfig(**values)


def select_graph_indices(dataset: Any, num_graphs: int, seed: int) -> list[int]:
    if num_graphs <= 0 or num_graphs >= len(dataset):
        return list(range(len(dataset)))
    rng = random.Random(seed)
    return sorted(rng.sample(range(len(dataset)), k=num_graphs))


def to_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def parse_csv_list(raw: str, default: Sequence[str], all_values: Sequence[str]) -> tuple[str, ...]:
    if raw == "all":
        return tuple(all_values)
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        return tuple(default)
    unknown = sorted(set(values) - set(all_values))
    if unknown:
        raise ValueError(f"unknown values {unknown}; expected one of {list(all_values)} or all")
    return values


@dataclass
class SparseLayerCapture:
    layer: int
    src: torch.Tensor
    dst: torch.Tensor
    graph: torch.Tensor
    local_src: torch.Tensor
    local_dst: torch.Tensor
    attention: torch.Tensor
    logits: torch.Tensor
    node_message: torch.Tensor
    pair_message: torch.Tensor
    message: torch.Tensor
    head_output: torch.Tensor
    heads: int
    message_dim: int


class OfficialGRITMechanisticCollector:
    """Forward-hook collector for official GRIT/static-GRIT wrappers.

    The hook stores sparse per-pair routing and message tensors after the official
    GRIT attention module has computed them. Gradients are retained on the per-head
    aggregated transport output, giving exact gradients for each pair contribution
    because ``o_i = sum_j A_ij m_ij``.
    """

    def __init__(self, model: nn.Module) -> None:
        if not hasattr(model, "layers"):
            raise TypeError("official GRIT mechanistic collector expects model.layers")
        self.model = model
        self.records: list[SparseLayerCapture] = []
        self.handles: list[Any] = []

    def __enter__(self) -> "OfficialGRITMechanisticCollector":
        self.records = []
        self.handles = []

        def make_hook(layer_idx: int):
            def hook(module: nn.Module, inputs: tuple[Any, ...], outputs: Any) -> None:
                pyg_batch = inputs[0]
                head_output = outputs[0] if isinstance(outputs, tuple) else outputs
                if not torch.is_tensor(head_output):
                    raise RuntimeError("GRIT attention hook did not receive tensor output")
                head_output.retain_grad()
                self.records.append(self._capture(layer_idx, module, pyg_batch, head_output))

            return hook

        for layer_idx, layer in enumerate(self.model.layers):
            self.handles.append(layer.attention.register_forward_hook(make_hook(layer_idx)))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _capture(
        self,
        layer_idx: int,
        attention_module: nn.Module,
        pyg_batch: Any,
        head_output: torch.Tensor,
    ) -> SparseLayerCapture:
        edge_index = pyg_batch.edge_index.long()
        src = edge_index[0]
        dst = edge_index[1]
        attn_e = pyg_batch.attn.squeeze(-1)
        if attn_e.dim() != 2:
            raise RuntimeError(f"expected GRIT attention [E,H], got {tuple(attn_e.shape)}")
        heads = int(attn_e.size(1))
        node_msg, pair_msg, logits = grit_attention_components(attention_module, pyg_batch)
        total_msg = node_msg + pair_msg
        if total_msg.dim() != 3:
            raise RuntimeError(f"expected GRIT message [E,H,D], got {tuple(total_msg.shape)}")
        if int(total_msg.size(1)) != heads:
            raise RuntimeError("attention head count and message head count differ")

        if head_output.dim() == 2:
            head_view = head_output.reshape(head_output.size(0), heads, -1)
        elif head_output.dim() == 3:
            head_view = head_output
        else:
            raise RuntimeError(f"unexpected GRIT head output shape {tuple(head_output.shape)}")
        if int(head_view.size(1)) != heads:
            raise RuntimeError("head output cannot be reshaped into attention heads")

        counts = [int(value) for value in pyg_batch.graph_num_nodes.detach().cpu().tolist()]
        graph, local_src, local_dst = edge_local_coordinates(src, dst, counts)
        return SparseLayerCapture(
            layer=layer_idx,
            src=src.detach(),
            dst=dst.detach(),
            graph=graph.detach(),
            local_src=local_src.detach(),
            local_dst=local_dst.detach(),
            attention=attn_e.detach().float(),
            logits=logits.detach().float(),
            node_message=node_msg.detach().float(),
            pair_message=pair_msg.detach().float(),
            message=total_msg.detach().float(),
            head_output=head_output,
            heads=heads,
            message_dim=int(total_msg.size(-1)),
        )


def grit_attention_components(
    attention_module: nn.Module,
    pyg_batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct official GRIT node-value, pair-value and logit tensors.

    This mirrors ``MultiHeadAttentionLayerGritSparse.propagate_attention`` from
    the official implementation. It is used only for logging/reductions; the
    model forward path remains untouched.
    """

    edge_index = pyg_batch.edge_index.long()
    src = edge_index[0]
    dst = edge_index[1]
    node_msg = pyg_batch.V_h[src]
    heads = int(node_msg.size(1))
    dim = int(node_msg.size(-1))
    score = pyg_batch.K_h[src] + pyg_batch.Q_h[dst]
    edge_state_for_value = torch.zeros_like(score)
    if getattr(pyg_batch, "E", None) is not None:
        edge_proj = pyg_batch.E.view(-1, heads, dim * 2)
        edge_weight = edge_proj[:, :, :dim]
        edge_bias = edge_proj[:, :, dim:]
        score = score * edge_weight
        score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
        score = score + edge_bias
        score = attention_module.act(score)
        edge_state_for_value = score

    logits = torch.einsum("ehd,dhc->ehc", score, attention_module.Aw)
    clamp = getattr(attention_module, "clamp", None)
    if clamp is not None:
        logits = torch.clamp(logits, min=-float(clamp), max=float(clamp))
    logits = logits.squeeze(-1)

    pair_msg = torch.zeros_like(node_msg)
    if getattr(attention_module, "edge_enhance", False) and getattr(pyg_batch, "E", None) is not None:
        pair_msg = torch.einsum("ehd,dhc->ehc", edge_state_for_value, attention_module.VeRow)
    return node_msg, pair_msg, logits


def edge_local_coordinates(
    src: torch.Tensor,
    dst: torch.Tensor,
    counts: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    graph = torch.empty_like(src)
    local_src = torch.empty_like(src)
    local_dst = torch.empty_like(dst)
    offset = 0
    for graph_idx, count in enumerate(counts):
        in_graph = (src >= offset) & (src < offset + count)
        graph[in_graph] = graph_idx
        local_src[in_graph] = src[in_graph] - offset
        local_dst[in_graph] = dst[in_graph] - offset
        offset += count
    return graph, local_src, local_dst


def task_scalar(pred: torch.Tensor, batch: Any) -> torch.Tensor:
    """Scalarise model output for task-gradient attribution.

    For graph regression this is the sum of predicted scalars. For binary node/edge
    tasks this is the signed logit margin on the supervised labels. Scaling does not
    affect lift fractions, but using margins gives task-directional influence.
    """

    if batch.task_type == "graph_regression":
        return pred.float().reshape(-1).sum()
    if batch.task_type == "edge_binary":
        labels = batch.edge_target.float().reshape_as(pred.float())
        return ((2.0 * labels - 1.0) * pred.float()).sum()
    if batch.task_type == "node_binary":
        node_pred = pred.float()[batch.node_mask]
        labels = batch.node_target.float()[batch.node_mask].reshape_as(node_pred)
        return ((2.0 * labels - 1.0) * node_pred).sum()
    if batch.task_type == "node_regression":
        return pred.float()[batch.node_mask].sum()
    raise ValueError(batch.task_type)


def view_head_grad(record: SparseLayerCapture) -> torch.Tensor:
    grad = record.head_output.grad
    if grad is None:
        raise RuntimeError(
            "missing retained GRIT head-output gradient; task scalar may not depend on layer output"
        )
    if grad.dim() == 2:
        return grad.reshape(grad.size(0), record.heads, -1).float()
    if grad.dim() == 3:
        return grad.float()
    raise RuntimeError(f"unexpected head-output gradient shape {tuple(grad.shape)}")


def valid_pair_mask(batch: Any) -> torch.Tensor:
    mask = batch.node_mask[:, :, None] & batch.node_mask[:, None, :]
    eye = torch.eye(mask.size(1), dtype=torch.bool, device=mask.device)[None, :, :]
    return mask & ~eye


def sparse_edge_mask(batch: Any) -> torch.Tensor:
    out = torch.zeros_like(batch.adj, dtype=torch.bool)
    if batch.edge_batch.numel():
        out[batch.edge_batch, batch.edge_dst, batch.edge_src] = True
    return out & valid_pair_mask(batch)


def infer_bipartition(adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    n = int(node_mask.sum().item())
    color = torch.full((adj.size(0),), -1, dtype=torch.long, device=adj.device)
    neighbors = (adj[:n, :n] > 0).detach().cpu()
    for start in range(n):
        if int(color[start]) >= 0:
            continue
        color[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            next_color = 1 - int(color[node])
            for neigh in torch.nonzero(neighbors[node], as_tuple=False).reshape(-1).tolist():
                if int(color[neigh]) < 0:
                    color[neigh] = next_color
                    queue.append(neigh)
    color[color < 0] = torch.arange(color.numel(), device=adj.device)[color < 0] % 2
    return color == 0


def matching_target_matrix(batch: Any) -> torch.Tensor:
    out = torch.zeros_like(batch.adj, dtype=torch.bool)
    if batch.task_type != "edge_binary" or batch.edge_target.numel() == 0:
        return out
    selected = batch.edge_target.float() > 0.5
    if bool(selected.any()):
        out[
            batch.edge_batch[selected],
            batch.edge_dst[selected],
            batch.edge_src[selected],
        ] = True
    return out & valid_pair_mask(batch)


def bipartite_operator_masks(batch: Any, random_controls: bool, seed: int) -> dict[str, torch.Tensor]:
    device = batch.node_mask.device
    bsz, nmax = batch.node_mask.shape
    valid = valid_pair_mask(batch)
    feasible = sparse_edge_mask(batch)
    optimum = matching_target_matrix(batch)
    left_nodes = torch.zeros(bsz, nmax, dtype=torch.bool, device=device)
    right_nodes = torch.zeros_like(left_nodes)
    cross_partition = torch.zeros_like(valid)
    alternating = torch.zeros_like(valid)

    for graph_idx in range(bsz):
        left = infer_bipartition(batch.adj[graph_idx], batch.node_mask[graph_idx])
        left_nodes[graph_idx] = left
        right_nodes[graph_idx] = batch.node_mask[graph_idx] & ~left
        cross_partition[graph_idx] = (
            left[:, None] != left[None, :]
        ) & valid[graph_idx]
        alternating[graph_idx] = alternating_forest_mask(
            feasible[graph_idx],
            optimum[graph_idx],
            left,
            batch.node_mask[graph_idx],
        )

    matched_node = (optimum.any(dim=-1) | optimum.any(dim=-2)) & batch.node_mask
    unmatched_left = left_nodes & ~matched_node
    unmatched_right = right_nodes & ~matched_node
    masks = {
        "valid_pair": valid.float(),
        "feasible_edge": feasible.float(),
        "cross_partition": cross_partition.float(),
        "optimum_matching_edge": optimum.float(),
        "unmatched_feasible_edge": (feasible & ~optimum).float(),
        "alternating_forest_edge": alternating.float(),
        "unmatched_left_incidence": incidence_mask(unmatched_left, valid).float(),
        "unmatched_right_incidence": incidence_mask(unmatched_right, valid).float(),
        "global_nonedge": (valid & ~feasible).float(),
    }
    if random_controls:
        masks.update(rate_matched_random_masks(masks, batch, valid, seed))
    return masks


def alternating_forest_mask(
    feasible: torch.Tensor,
    optimum: torch.Tensor,
    left: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    n = int(node_mask.sum().item())
    reached = torch.zeros_like(node_mask)
    traversed = torch.zeros_like(feasible)
    matched = optimum | optimum.T
    matched_node = matched[:n, :n].any(dim=0) | matched[:n, :n].any(dim=1)
    queue = [idx for idx in range(n) if bool(left[idx]) and not bool(matched_node[idx])]
    for idx in queue:
        reached[idx] = True
    while queue:
        node = queue.pop(0)
        if bool(left[node]):
            candidates = torch.nonzero(feasible[:, node] & ~matched[:, node], as_tuple=False)
        else:
            candidates = torch.nonzero(matched[:, node], as_tuple=False)
        for item in candidates.reshape(-1).tolist():
            if item >= n:
                continue
            traversed[item, node] = True
            if not bool(reached[item]):
                reached[item] = True
                queue.append(item)
    return traversed


def incidence_mask(nodes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return valid & (nodes[:, :, None] | nodes[:, None, :])


def flow_source_sink(batch: Any, graph_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    node_type = batch.node_type[graph_idx]
    node_mask = batch.node_mask[graph_idx]
    active = node_type[node_mask]
    nonzero = torch.unique(active[active != 0])
    source = torch.zeros_like(node_mask)
    sink = torch.zeros_like(node_mask)
    if nonzero.numel() >= 2:
        source[node_type == nonzero.min()] = True
        sink[node_type == nonzero.max()] = True
    else:
        raise RuntimeError(
            "flow source/sink roles were not recoverable from node_type; "
            "memo-faithful flow masks require explicit source and sink roles"
        )
    return source & node_mask, sink & node_mask


def flow_operator_masks(batch: Any, random_controls: bool, seed: int) -> dict[str, torch.Tensor]:
    device = batch.node_mask.device
    bsz, nmax = batch.node_mask.shape
    valid = valid_pair_mask(batch)
    directed = sparse_edge_mask(batch)
    capacity = torch.zeros(bsz, nmax, nmax, dtype=torch.float32, device=device)
    source_inc = torch.zeros_like(valid)
    sink_inc = torch.zeros_like(valid)
    residual_side_inc = torch.zeros_like(valid)
    residual_side_internal = torch.zeros_like(valid)
    min_cut_crossing = torch.zeros_like(valid)
    saturated = torch.zeros_like(valid)
    st_path = torch.zeros_like(valid)

    for graph_idx in range(bsz):
        source, sink = flow_source_sink(batch, graph_idx)
        capacity[graph_idx] = capacity_matrix_for_graph(batch, graph_idx)
        maxflow = deterministic_max_flow(capacity[graph_idx], source, sink, batch.node_mask[graph_idx])
        source_inc[graph_idx] = valid[graph_idx] & (source[:, None] | source[None, :])
        sink_inc[graph_idx] = valid[graph_idx] & (sink[:, None] | sink[None, :])
        side = maxflow.source_side.to(device=device)
        residual_side_inc[graph_idx] = valid[graph_idx] & (side[:, None] | side[None, :])
        residual_side_internal[graph_idx] = valid[graph_idx] & side[:, None] & side[None, :]
        min_cut_crossing[graph_idx] = (
            directed[graph_idx]
            & side[None, :]
            & ~side[:, None]
        )
        saturated[graph_idx] = (
            directed[graph_idx]
            & (capacity[graph_idx] > 0)
            & (maxflow.residual_forward.to(device=device) <= 1.0e-8)
        )
        st_path[graph_idx] = shortest_st_path_mask(
            directed[graph_idx],
            source,
            sink,
            batch.node_mask[graph_idx],
        )

    masks = {
        "valid_pair": valid.float(),
        "directed_edge": directed.float(),
        "source_incidence": source_inc.float(),
        "sink_incidence": sink_inc.float(),
        "residual_source_side_incidence": residual_side_inc.float(),
        "residual_source_side_internal": residual_side_internal.float(),
        "min_cut_crossing_edge": min_cut_crossing.float(),
        "saturated_edge": saturated.float(),
        "shortest_st_path_edge": st_path.float(),
        "capacity_weighted_edge": capacity * directed.float(),
        "global_nonedge": (valid & ~directed).float(),
    }
    if random_controls:
        masks.update(rate_matched_random_masks(masks, batch, valid, seed))
    return masks


@dataclass
class MaxFlowResult:
    residual_forward: torch.Tensor
    source_side: torch.Tensor


def capacity_matrix_for_graph(batch: Any, graph_idx: int) -> torch.Tensor:
    nmax = batch.node_mask.size(1)
    out = torch.zeros(nmax, nmax, dtype=torch.float32, device=batch.node_mask.device)
    edge_mask = batch.edge_batch == graph_idx
    if not bool(edge_mask.any()):
        return out
    src = batch.edge_src[edge_mask]
    dst = batch.edge_dst[edge_mask]
    cap = batch.edge_value[edge_mask].float().clamp_min(0.0)
    out.index_put_((dst, src), cap, accumulate=True)
    return out


def deterministic_max_flow(
    capacity: torch.Tensor,
    source: torch.Tensor,
    sink: torch.Tensor,
    node_mask: torch.Tensor,
) -> MaxFlowResult:
    n = int(node_mask.sum().item())
    source_idx = torch.nonzero(source[:n], as_tuple=False).reshape(-1)
    sink_idx = torch.nonzero(sink[:n], as_tuple=False).reshape(-1)
    if source_idx.numel() != 1 or sink_idx.numel() != 1:
        raise RuntimeError("flow masks currently require exactly one source and one sink")
    s = int(source_idx.item())
    t = int(sink_idx.item())
    residual = capacity[:n, :n].detach().cpu().double().clone()
    while True:
        parent = [-1] * n
        parent[s] = s
        queue = [s]
        while queue and parent[t] < 0:
            u = queue.pop(0)
            for v in range(n):
                if parent[v] < 0 and float(residual[v, u]) > 1.0e-12:
                    parent[v] = u
                    queue.append(v)
                    if v == t:
                        break
        if parent[t] < 0:
            break
        aug = float("inf")
        v = t
        while v != s:
            u = parent[v]
            aug = min(aug, float(residual[v, u]))
            v = u
        v = t
        while v != s:
            u = parent[v]
            residual[v, u] -= aug
            residual[u, v] += aug
            v = u

    source_side_cpu = torch.zeros(n, dtype=torch.bool)
    source_side_cpu[s] = True
    queue = [s]
    while queue:
        u = queue.pop(0)
        for v in range(n):
            if not bool(source_side_cpu[v]) and float(residual[v, u]) > 1.0e-12:
                source_side_cpu[v] = True
                queue.append(v)

    full_residual = torch.zeros_like(capacity)
    full_side = torch.zeros_like(node_mask)
    full_residual[:n, :n] = residual.to(device=capacity.device, dtype=capacity.dtype)
    full_side[:n] = source_side_cpu.to(device=node_mask.device)
    return MaxFlowResult(residual_forward=full_residual, source_side=full_side)


def shortest_st_path_mask(
    directed: torch.Tensor,
    source: torch.Tensor,
    sink: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    n = int(node_mask.sum().item())
    srcs = torch.nonzero(source[:n], as_tuple=False).reshape(-1).tolist()
    sinks = set(torch.nonzero(sink[:n], as_tuple=False).reshape(-1).tolist())
    parent = [-1] * n
    queue = list(srcs)
    seen = set(srcs)
    hit = None
    while queue and hit is None:
        node = queue.pop(0)
        if node in sinks:
            hit = node
            break
        for nxt in torch.nonzero(directed[:, node], as_tuple=False).reshape(-1).tolist():
            if nxt < n and nxt not in seen:
                seen.add(nxt)
                parent[nxt] = node
                queue.append(nxt)
    out = torch.zeros_like(directed)
    if hit is None:
        return out
    cur = hit
    while parent[cur] >= 0:
        prev = parent[cur]
        out[cur, prev] = True
        cur = prev
    return out


def generic_operator_masks(batch: Any, random_controls: bool, seed: int) -> dict[str, torch.Tensor]:
    valid = valid_pair_mask(batch)
    edge = sparse_edge_mask(batch)
    masks = {
        "valid_pair": valid.float(),
        "directed_edge": edge.float(),
        "global_nonedge": (valid & ~edge).float(),
    }
    if random_controls:
        masks.update(rate_matched_random_masks(masks, batch, valid, seed))
    return masks


def make_operator_masks(
    task: str,
    batch: Any,
    random_controls: bool,
    seed: int,
) -> dict[str, torch.Tensor]:
    if "bipartite" in task or "matching" in task:
        return bipartite_operator_masks(batch, random_controls, seed)
    if "flow" in task:
        return flow_operator_masks(batch, random_controls, seed)
    return generic_operator_masks(batch, random_controls, seed)


def rate_matched_random_masks(
    masks: Mapping[str, torch.Tensor],
    batch: Any,
    valid: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    out: dict[str, torch.Tensor] = {}
    strata = random_control_strata(batch, valid)
    for name, mask in masks.items():
        if name == "valid_pair" or name.startswith("random__"):
            continue
        sampled = torch.zeros_like(mask)
        for graph_idx in range(mask.size(0)):
            sampled[graph_idx] = sample_stratified_random_mask(
                mask[graph_idx],
                valid[graph_idx],
                strata[graph_idx],
                generator,
            )
        out[f"random__{name}"] = sampled
    return out


def random_control_strata(batch: Any, valid: torch.Tensor) -> torch.Tensor:
    """Strata named in the memo: SPD, endpoint degree bins, local/global, edge/non-edge."""

    device = valid.device
    degree_bins = degree_bin(batch.degree.float())
    spd = batch.spd.long().clamp_min(0).clamp_max(1024)
    local = (batch.adj > 0).long()
    directed = sparse_edge_mask(batch).long()
    recv_bins = degree_bins[:, :, None].expand_as(valid).long()
    send_bins = degree_bins[:, None, :].expand_as(valid).long()
    code = directed
    code = code * 2 + local
    code = code * 2048 + spd
    code = code * 16 + recv_bins
    code = code * 16 + send_bins
    return code.to(device=device)


def degree_bin(degree: torch.Tensor) -> torch.Tensor:
    bins = torch.zeros_like(degree, dtype=torch.long)
    thresholds = [1, 2, 4, 8, 16, 32, 64, 128]
    for idx, threshold in enumerate(thresholds, start=1):
        bins = torch.where(degree >= threshold, torch.full_like(bins, idx), bins)
    return bins


def sample_stratified_random_mask(
    mask: torch.Tensor,
    valid: torch.Tensor,
    strata: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    device = mask.device
    sampled = torch.zeros_like(mask)
    positive = torch.nonzero((mask > 0) & valid, as_tuple=False).detach().cpu()
    if positive.numel() == 0:
        return sampled
    valid_positions = torch.nonzero(valid, as_tuple=False).detach().cpu()
    strata_cpu = strata.detach().cpu()
    candidates: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, col in valid_positions.tolist():
        candidates[int(strata_cpu[row, col])].append((row, col))

    positive_by_stratum: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, col in positive.tolist():
        positive_by_stratum[int(strata_cpu[row, col])].append((row, col))

    for code, positives in positive_by_stratum.items():
        pool = candidates.get(code, [])
        if not pool:
            continue
        count = len(positives)
        order = torch.randperm(len(pool), generator=generator)
        if count <= len(pool):
            chosen = [pool[int(i)] for i in order[:count]]
        else:
            chosen = [pool[int(i)] for i in order.tolist()]
            extra = torch.randint(len(pool), (count - len(pool),), generator=generator)
            chosen.extend(pool[int(i)] for i in extra.tolist())
        weights = mask[
            torch.tensor([row for row, _col in positives], device=device),
            torch.tensor([col for _row, col in positives], device=device),
        ].detach().cpu()
        weight_order = torch.randperm(weights.numel(), generator=generator)
        weights = weights[weight_order].to(device=device, dtype=mask.dtype)
        rows = torch.tensor([row for row, _col in chosen], device=device)
        cols = torch.tensor([col for _row, col in chosen], device=device)
        sampled[rows, cols] = weights
    return sampled


class OperatorTransportAccumulator:
    def __init__(self, task_transport_responsibility: Mapping[tuple[int, int], float]) -> None:
        self.rows: dict[tuple[int, int, str, str], dict[str, float]] = defaultdict(float_dict)
        self.base: dict[str, dict[str, float]] = defaultdict(float_dict)
        self.graph_rows: list[dict[str, Any]] = []
        self.graph_count = 0
        self.task_transport_responsibility = dict(task_transport_responsibility)

    def add_batch(
        self,
        captures: Sequence[SparseLayerCapture],
        masks: Mapping[str, torch.Tensor],
        valid: torch.Tensor,
        global_mask: torch.Tensor,
        graph_indices: Sequence[int],
    ) -> None:
        self.graph_count += len(graph_indices)
        self._add_base_rates(masks, valid)
        for record in captures:
            self._add_layer(record, masks, global_mask)

    def _add_base_rates(self, masks: Mapping[str, torch.Tensor], valid: torch.Tensor) -> None:
        valid_den = valid.float().sum().item()
        for name, mask in masks.items():
            self.base[name]["positive"] += float((mask * valid.float()).sum().item())
            self.base[name]["denominator"] += float(valid_den)

    def _add_layer(
        self,
        record: SparseLayerCapture,
        masks: Mapping[str, torch.Tensor],
        global_mask: torch.Tensor,
    ) -> None:
        grad = view_head_grad(record)
        grad_e = grad[record.dst]
        attention = record.attention.clamp_min(0.0)
        global_values = global_mask[record.graph, record.local_dst, record.local_src].float()

        components = {
            "total": record.message,
            "node_value": record.node_message,
            "pair_value": record.pair_message,
        }
        for component_name, message in components.items():
            contribution = message * record.attention[:, :, None]
            influence = (grad_e * contribution).sum(dim=-1).abs()
            contribution_norm = torch.linalg.vector_norm(contribution, dim=-1)
            influence_total = influence.sum(dim=0)
            contribution_total = contribution_norm.sum(dim=0)
            attention_total = attention.sum(dim=0)
            global_influence = (influence * global_values[:, None]).sum(dim=0)
            global_contribution = (contribution_norm * global_values[:, None]).sum(dim=0)

            for name, mask in masks.items():
                values = mask[record.graph, record.local_dst, record.local_src].float()
                influence_on = (influence * values[:, None]).sum(dim=0)
                contribution_on = (contribution_norm * values[:, None]).sum(dim=0)
                attention_on = (attention * values[:, None]).sum(dim=0)
                for head in range(record.heads):
                    row = self.rows[(record.layer, head, name, component_name)]
                    row["influence_on_operator"] += to_float(influence_on[head])
                    row["influence_total"] += to_float(influence_total[head])
                    row["abs_transport_on_operator"] += to_float(contribution_on[head])
                    row["abs_transport_total"] += to_float(contribution_total[head])
                    row["attention_on_operator"] += to_float(attention_on[head])
                    row["attention_total"] += to_float(attention_total[head])
                    row["global_influence"] += to_float(global_influence[head])
                    row["global_abs_transport"] += to_float(global_contribution[head])

    def per_head_rows(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for (layer, head, operator, component), values in sorted(self.rows.items()):
            base_rate = safe_div(
                self.base[operator]["positive"],
                self.base[operator]["denominator"],
            )
            influence_fraction = safe_div(
                values["influence_on_operator"],
                values["influence_total"],
            )
            abs_transport_fraction = safe_div(
                values["abs_transport_on_operator"],
                values["abs_transport_total"],
            )
            attention_fraction = safe_div(
                values["attention_on_operator"],
                values["attention_total"],
            )
            influence_lift = safe_div(influence_fraction, base_rate)
            abs_transport_lift = safe_div(abs_transport_fraction, base_rate)
            attention_lift = safe_div(attention_fraction, base_rate)
            global_task_transport = safe_div(
                values["global_influence"],
                values["influence_total"],
            )
            global_abs_transport = safe_div(
                values["global_abs_transport"],
                values["abs_transport_total"],
            )
            lift_gate = (
                influence_lift * global_task_transport
                if math.isfinite(influence_lift) and math.isfinite(global_task_transport)
                else float("nan")
            )
            responsibility = self.task_transport_responsibility.get((layer, head), float("nan"))
            ots = (
                lift_gate * responsibility
                if math.isfinite(lift_gate) and math.isfinite(responsibility)
                else float("nan")
            )
            out.append(
                {
                    "layer": layer,
                    "head": head,
                    "operator": operator,
                    "transport_component": component,
                    "graphs": self.graph_count,
                    "base_rate": base_rate,
                    "influence_fraction": influence_fraction,
                    "influence_lift": influence_lift,
                    "abs_transport_fraction": abs_transport_fraction,
                    "abs_transport_lift": abs_transport_lift,
                    "attention_fraction": attention_fraction,
                    "attention_lift": attention_lift,
                    "global_task_transport": global_task_transport,
                    "global_abs_transport": global_abs_transport,
                    "transport_lift_global_gate": lift_gate,
                    "task_transport_responsibility": responsibility,
                    "operator_transport_score": ots,
                    **values,
                }
            )
        return out

    def layer_rows(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[int, str, str], dict[str, float]] = defaultdict(float_dict)
        for (layer, _head, operator, component), values in self.rows.items():
            group = grouped[(layer, operator, component)]
            for key, value in values.items():
                group[key] += value
        out: list[dict[str, Any]] = []
        for (layer, operator, component), values in sorted(grouped.items()):
            base_rate = safe_div(
                self.base[operator]["positive"],
                self.base[operator]["denominator"],
            )
            influence_fraction = safe_div(
                values["influence_on_operator"],
                values["influence_total"],
            )
            abs_transport_fraction = safe_div(
                values["abs_transport_on_operator"],
                values["abs_transport_total"],
            )
            attention_fraction = safe_div(
                values["attention_on_operator"],
                values["attention_total"],
            )
            influence_lift = safe_div(influence_fraction, base_rate)
            global_task_transport = safe_div(
                values["global_influence"],
                values["influence_total"],
            )
            out.append(
                {
                    "layer": layer,
                    "operator": operator,
                    "transport_component": component,
                    "graphs": self.graph_count,
                    "base_rate": base_rate,
                    "influence_fraction": influence_fraction,
                    "influence_lift": influence_lift,
                    "abs_transport_fraction": abs_transport_fraction,
                    "abs_transport_lift": safe_div(abs_transport_fraction, base_rate),
                    "attention_fraction": attention_fraction,
                    "attention_lift": safe_div(attention_fraction, base_rate),
                    "global_task_transport": global_task_transport,
                    "global_abs_transport": safe_div(
                        values["global_abs_transport"],
                        values["abs_transport_total"],
                    ),
                    "transport_lift_global_gate": (
                        influence_lift * global_task_transport
                        if math.isfinite(influence_lift)
                        and math.isfinite(global_task_transport)
                        else float("nan")
                    ),
                    **values,
                }
            )
        return out

    def head_summary_rows(self) -> list[dict[str, Any]]:
        by_head: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in self.per_head_rows():
            if row["transport_component"] == "total":
                by_head[(int(row["layer"]), int(row["head"]))].append(row)
        out: list[dict[str, Any]] = []
        for (layer, head), rows in sorted(by_head.items()):
            primary = [
                row
                for row in rows
                if not str(row["operator"]).startswith(PRIMARY_OPERATOR_EXCLUDE_PREFIXES)
                and str(row["operator"]) != "valid_pair"
            ]
            best = max(
                primary,
                key=lambda row: nan_to_neg_inf(float(row["transport_lift_global_gate"])),
                default=None,
            )
            first = rows[0]
            out.append(
                {
                    "layer": layer,
                    "head": head,
                    "top_operator": "" if best is None else best["operator"],
                    "top_transport_lift_global_gate": float("nan")
                    if best is None
                    else best["transport_lift_global_gate"],
                    "top_operator_transport_score": float("nan")
                    if best is None
                    else best["operator_transport_score"],
                    "top_influence_lift": float("nan") if best is None else best["influence_lift"],
                    "task_transport_responsibility": first["task_transport_responsibility"],
                    "global_task_transport": first["global_task_transport"],
                    "global_abs_transport": first["global_abs_transport"],
                    "total_influence": first["influence_total"],
                    "total_abs_transport": first["abs_transport_total"],
                }
            )
        return out


def float_dict() -> dict[str, float]:
    return defaultdict(float)


def safe_div(num: float, den: float) -> float:
    if abs(den) <= EPS:
        return float("nan")
    return float(num) / float(den)


def nan_to_neg_inf(value: float) -> float:
    return value if math.isfinite(value) else -float("inf")


def make_collector(model: nn.Module, model_name: str, adapter: str) -> OfficialGRITMechanisticCollector:
    selected = adapter
    if selected == "auto":
        selected = "official_grit" if model_name in {"grit", "static_grit"} else "unsupported"
    if selected == "official_grit":
        return OfficialGRITMechanisticCollector(model)
    raise NotImplementedError(
        f"No mechanistic collector is available for model={model_name!r}, adapter={adapter!r}. "
        "The memo's A/m/C/r operator analysis requires model-specific hooks. "
        "This file currently implements official GRIT/static-GRIT."
    )


def load_task_transport_responsibility(
    path: Optional[Path],
    *,
    intervention: str,
    block: str,
    centered: str,
) -> dict[tuple[int, int], float]:
    """Load the memo's ``task_transport_responsibility_lh`` term.

    This comes from the separate swap-specialisation metric summary. If it is not
    supplied, formal OTS is intentionally left undefined rather than approximated.
    """

    if path is None:
        return {}
    out: dict[tuple[int, int], float] = {}
    with path.expanduser().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("metric") != "output_transport_responsibility":
                continue
            if row.get("intervention") != intervention:
                continue
            if row.get("block") != block:
                continue
            row_centered = str(row.get("centered", ""))
            if centered != "any" and row_centered != centered:
                continue
            out[(int(row["layer"]), int(row["head"]))] = float(row["mean"])
    if not out:
        raise RuntimeError(
            f"no task transport responsibility rows matched {path} "
            f"(metric=output_transport_responsibility, intervention={intervention}, "
            f"block={block}, centered={centered})"
        )
    return out


def plot_outputs(out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"[plot] skipped plots because plotting imports failed: {exc}", flush=True)
        return

    layer_path = out_dir / "operator_transport_by_layer.csv"
    head_path = out_dir / "operator_transport_per_head.csv"
    if not layer_path.exists() or not head_path.exists():
        print("[plot] skipped plots because CSV outputs are missing", flush=True)
        return
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    layer_df = pd.read_csv(layer_path)
    head_df = pd.read_csv(head_path)
    layer_total = layer_df[layer_df["transport_component"] == "total"]
    head_total = head_df[head_df["transport_component"] == "total"]
    layer_primary = layer_total[
        ~layer_total["operator"].astype(str).str.startswith(PRIMARY_OPERATOR_EXCLUDE_PREFIXES)
    ]

    heatmap(
        layer_primary,
        "influence_lift",
        figures / "operator_influence_lift_by_layer.png",
        "Operator influence lift",
    )
    heatmap(
        layer_primary,
        "transport_lift_global_gate",
        figures / "transport_lift_global_gate_by_layer.png",
        "Influence lift x global task-transport gate",
    )
    heatmap(
        layer_primary,
        "abs_transport_lift",
        figures / "operator_abs_transport_lift_by_layer.png",
        "Realised transport lift",
    )

    for value, title, name in [
        ("global_task_transport", "Global task-weighted transport", "global_task_transport"),
        ("global_abs_transport", "Global realised transport", "global_abs_transport"),
    ]:
        pivot = head_total.pivot_table(index="layer", columns="head", values=value, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=180)
        image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), labels=[str(i) for i in pivot.index])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figures / f"{name}_per_head.png")
        plt.close(fig)

    controls = layer_total[layer_total["operator"].astype(str).str.startswith("random__")]
    if not controls.empty:
        fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)
        primary_mean = layer_primary.groupby("operator")["influence_lift"].mean()
        control_mean = controls.assign(
            matched=controls["operator"].astype(str).str.replace("random__", "", regex=False)
        ).groupby("matched")["influence_lift"].mean()
        ops = sorted(set(primary_mean.index) & set(control_mean.index))
        x = torch.arange(len(ops)).float().numpy()
        ax.bar(x - 0.18, [primary_mean[o] for o in ops], width=0.36, label="operator")
        ax.bar(x + 0.18, [control_mean[o] for o in ops], width=0.36, label="rate-matched random")
        ax.axhline(1.0, color="0.3", linewidth=0.8)
        ax.set_xticks(x, labels=ops, rotation=35, ha="right")
        ax.set_ylabel("mean influence lift")
        ax.set_title("Operator lift versus rate-matched controls")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures / "operator_lift_random_controls.png")
        plt.close(fig)


def heatmap(df: Any, value: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(index="layer", columns="operator", values=value, aggfunc="mean")
    fig_width = max(8.0, 0.72 * max(1, len(pivot.columns)))
    fig, ax = plt.subplots(figsize=(fig_width, 4.6), dpi=180)
    image = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("operator")
    ax.set_ylabel("layer")
    ax.set_xticks(range(len(pivot.columns)), labels=[str(c) for c in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), labels=[str(i) for i in pivot.index])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def analysis_protocol_audit() -> dict[str, Any]:
    return {
        "primary_question": (
            "whether realised, task-weighted head transport concentrates on "
            "task-relevant graph operators beyond their base rate"
        ),
        "recorded_fields": {
            "A": "post-softmax per-head routing A[l,h,i,j]",
            "logits": "reconstructed official GRIT pre-softmax attention logits",
            "m_node": "node-value message W_v h_j before aggregation",
            "m_pair": "pair-value relation message u(r_ij) before aggregation",
            "m_total": "m_node + m_pair",
            "C": "realised pair contribution C[l,h,i,j,d] = A_ij m_ij,d",
            "o": "head output o[l,h,i,d] = sum_j C_ij,d",
            "I": "task influence I_ij = abs(<grad_s(o_i), C_ij>)",
        },
        "scalarisation": {
            "graph_regression": "sum of predicted graph scalars",
            "binary_edge_or_node": "sum of signed logit margins (2y-1)*logit",
        },
        "scores": {
            "influence_fraction": "sum I on operator / sum I over valid pairs",
            "influence_lift": "influence_fraction divided by operator base rate",
            "abs_transport_fraction": "sum ||C|| on operator / sum ||C|| over valid pairs",
            "attention_fraction": "sum A on operator / sum A over valid pairs",
            "global_task_transport": "sum I on global/non-edge pairs / sum I",
            "transport_lift_global_gate": "influence_lift * global_task_transport",
            "operator_transport_score": (
                "influence_lift * global_task_transport * task_transport_responsibility; "
                "left NaN unless the swap-metric responsibility CSV is supplied"
            ),
        },
        "implemented_controls": (
            "matched random masks preserving SPD bucket, endpoint degree bins, "
            "local/global adjacency, and directed edge/non-edge status"
        ),
        "first_hypotheses": [
            "Task-useful GRIT heads should show influence lift on solver-relevant operator masks.",
            "Influence lift should separate true masks from rate-matched random controls.",
            "Attention-only lift can disagree with realised/influence transport lift.",
            "Global task transport should be concentrated in a subset of layers/heads.",
        ],
        "known_scope": [
            "Official GRIT/static-GRIT are implemented first because their pair transport is explicit.",
            "Flow masks require explicit source/sink roles in node_type and use deterministic max-flow.",
            "M1 distillation, M2 mechanism knockouts, M4 ranked causal ablation, and pair-set patching "
            "are not silently approximated by this atlas script.",
        ],
    }


def run(args: argparse.Namespace) -> None:
    runner = load_runner(args.runner_path)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must be a mapping: {args.checkpoint}")
    cfg = build_screen_config(runner, args, checkpoint)
    print(
        f"[setup] task={args.task} model={args.model} split={args.split} "
        f"graphs={args.num_graphs} device={device}",
        flush=True,
    )
    splits = runner.load_official_graphbench_task(
        args.dataset_root,
        args.task,
        cfg,
        force_reload=args.force_reload_data,
        log=print,
    )
    splits = runner.attach_or_build_pe_cache(
        splits,
        args.pe_cache_root,
        cfg,
        namespace=args.pe_cache_namespace,
        dtype_name=args.pe_cache_dtype,
        pe_workers=args.pe_workers,
        pe_save_every=args.pe_save_every,
        force_recompute=False,
        build_missing=args.build_missing_pe_cache,
        require_present=not args.build_missing_pe_cache,
        log=print,
    )
    dataset = splits[args.split]
    selected_indices = select_graph_indices(dataset, args.num_graphs, args.graph_seed)
    graphs = [dataset[i] for i in selected_indices]
    if not graphs:
        raise RuntimeError(f"no graphs selected from {args.task}/{args.split}")

    model = runner.build_model(args.model, cfg, backend=args.model_backend)
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint state did not match model exactly: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.to(device).eval()
    collector = make_collector(model, args.model, args.adapter)
    responsibility = load_task_transport_responsibility(
        args.specialisation_summary_csv,
        intervention=args.responsibility_intervention,
        block=args.responsibility_block,
        centered=args.responsibility_centered,
    )
    accumulator = OperatorTransportAccumulator(responsibility)
    start_time = time.time()
    batch_size = max(1, int(args.batch_size))

    for start_idx in range(0, len(graphs), batch_size):
        end_idx = min(len(graphs), start_idx + batch_size)
        batch_graphs = graphs[start_idx:end_idx]
        graph_indices = selected_indices[start_idx:end_idx]
        batch = runner.collate_graphs(batch_graphs).to(device)
        model.zero_grad(set_to_none=True)
        with collector as active_collector:
            with torch.enable_grad():
                pred = model(batch)
                scalar = task_scalar(pred, batch)
                scalar.backward()
        masks = make_operator_masks(
            args.task,
            batch,
            random_controls=args.random_controls,
            seed=args.random_seed + start_idx,
        )
        valid = valid_pair_mask(batch)
        global_mask = masks.get("global_nonedge", (valid & ~sparse_edge_mask(batch)).float())
        accumulator.add_batch(
            active_collector.records,
            masks,
            valid.float(),
            global_mask.float(),
            graph_indices,
        )
        print(
            f"[analysis] batch={start_idx // batch_size + 1} "
            f"graphs={start_idx}:{end_idx} scalar={to_float(scalar):.5g}",
            flush=True,
        )
        del batch, pred, scalar, masks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_head = accumulator.per_head_rows()
    by_layer = accumulator.layer_rows()
    head_summary = accumulator.head_summary_rows()
    write_csv(out_dir / "operator_transport_per_head.csv", per_head)
    write_csv(out_dir / "operator_transport_by_layer.csv", by_layer)
    write_csv(out_dir / "head_operator_summary.csv", head_summary)
    metadata = {
        "task": args.task,
        "split": args.split,
        "model": args.model,
        "model_backend": args.model_backend,
        "adapter": args.adapter,
        "checkpoint": str(args.checkpoint),
        "num_graphs": len(graphs),
        "batch_size": batch_size,
        "selected_graph_indices": selected_indices,
        "device": str(device),
        "elapsed_seconds": time.time() - start_time,
        "random_controls": args.random_controls,
        "task_transport_responsibility_source": None
        if args.specialisation_summary_csv is None
        else str(args.specialisation_summary_csv),
        "task_transport_responsibility_rows": len(responsibility),
        "analysis_protocol_audit": analysis_protocol_audit(),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plot_outputs(out_dir)
    print(
        f"[done] wrote {len(per_head)} per-head rows and {len(by_layer)} layer rows to {out_dir}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="grit")
    parser.add_argument("--model-backend", default="official", choices=("official", "local"))
    parser.add_argument("--adapter", default="auto", choices=("auto", "official_grit"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--num-graphs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--graph-seed", type=int, default=0)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--runner-path", type=Path, default=None)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            os.environ.get(
                "GRAPHBENCH_DATASET_ROOT",
                "/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets",
            )
        ),
    )
    parser.add_argument(
        "--pe-cache-root",
        type=Path,
        default=Path(
            os.environ.get(
                "GRAPHBENCH_PE_CACHE_ROOT",
                "/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache",
            )
        ),
    )
    parser.add_argument("--pe-cache-namespace", default="base_40k4k4k_n64")
    parser.add_argument("--pe-cache-dtype", default="float32", choices=("float32", "float16"))
    parser.add_argument("--build-missing-pe-cache", action="store_true")
    parser.add_argument("--pe-workers", type=int, default=1)
    parser.add_argument("--pe-save-every", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mechanistic_operator_analysis"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-reload-data", action="store_true")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--train-node-size", type=int, default=None)
    parser.add_argument("--val-node-size", type=int, default=None)
    parser.add_argument("--test-node-size", type=int, default=None)
    parser.add_argument("--random-controls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--specialisation-summary-csv",
        type=Path,
        default=None,
        help=(
            "Optional per_head_summary.csv from core swap metrics. Required only "
            "for formal OTS; otherwise operator_transport_score is NaN."
        ),
    )
    parser.add_argument("--responsibility-intervention", default="content")
    parser.add_argument("--responsibility-block", default="all")
    parser.add_argument(
        "--responsibility-centered",
        default="",
        help="Filter for centred column in the summary CSV; use 'any' to ignore it.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
