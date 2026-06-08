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
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn


EPS = 1.0e-12
PRIMARY_OPERATOR_EXCLUDE_PREFIXES = ("random__",)


def is_primary_operator(operator: str) -> bool:
    return operator != "valid_pair" and not operator.startswith(PRIMARY_OPERATOR_EXCLUDE_PREFIXES)


MECHANISM_ABLATION_METADATA: dict[str, dict[str, str]] = {
    "clean": {
        "mechanism_axis": "reference",
        "display_label": "Clean",
        "claim_tested": "Unablated model performance.",
    },
    "local_only_support": {
        "mechanism_axis": "support",
        "display_label": "Local-only support",
        "claim_tested": "Does complete graph support carry useful beyond-GNN computation?",
    },
    "global_only_support": {
        "mechanism_axis": "support_diagnostic",
        "display_label": "Global-only support",
        "claim_tested": "Can dense global attention operate without the local graph channel?",
    },
    "no_structural_routing": {
        "mechanism_axis": "routing",
        "display_label": "No structural routing",
        "claim_tested": "Does relation information matter as a scalar sender-selection kernel?",
    },
    "no_pair_value_transport": {
        "mechanism_axis": "transport",
        "display_label": "No pair-value transport",
        "claim_tested": "Does relation information change the content being transported?",
    },
    "frozen_pair_state": {
        "mechanism_axis": "pair_state",
        "display_label": "Frozen pair state",
        "claim_tested": "Does pair-state evolution refine useful graph relations?",
    },
    "permuted_pair_state": {
        "mechanism_axis": "pair_state",
        "display_label": "Permuted pair state",
        "claim_tested": "Does pair-state content matter beyond matched marginal structure?",
    },
}

RANKING_DISPLAY_LABELS = {
    "random": "Random",
    "norm": "Output norm",
    "attention_mass": "Attention mass",
    "attention_kernel_alignment": "Attention/operator",
    "contribution_alignment": "Realised transport/operator",
    "influence_lift": "Task-weighted operator lift",
    "global_task_transport": "Global task transport",
    "transport_lift_global_gate": "Lift x global gate",
    "ots": "OTS",
}

RANKING_FAMILIES = {
    "random": "control",
    "norm": "control",
    "attention_mass": "attention_control",
    "attention_kernel_alignment": "attention_control",
    "contribution_alignment": "transport",
    "influence_lift": "task_weighted_transport",
    "global_task_transport": "transport",
    "transport_lift_global_gate": "task_weighted_transport",
    "ots": "task_weighted_transport",
}


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


def configure_torch_runtime(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Set GPU runtime options that matter for repeated A100 analysis passes."""

    settings: dict[str, Any] = {
        "device": str(device),
        "allow_tf32": False,
        "float32_matmul_precision": None,
        "autocast_dtype": args.autocast_dtype,
    }
    if device.type != "cuda":
        return settings

    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    settings["allow_tf32"] = bool(args.allow_tf32)
    if args.float32_matmul_precision != "default":
        torch.set_float32_matmul_precision(args.float32_matmul_precision)
        settings["float32_matmul_precision"] = args.float32_matmul_precision
    if args.cuda_benchmark:
        torch.backends.cudnn.benchmark = True
        settings["cudnn_benchmark"] = True
    return settings


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda" or dtype_name == "none":
        return nullcontext()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    return torch.autocast(device_type="cuda", dtype=dtype)


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


def graph_batches(
    graphs: Sequence[Any],
    graph_indices: Sequence[int],
    batch_size: int,
) -> list[tuple[list[int], list[Any]]]:
    out: list[tuple[list[int], list[Any]]] = []
    for start_idx in range(0, len(graphs), batch_size):
        end_idx = min(len(graphs), start_idx + batch_size)
        out.append((list(graph_indices[start_idx:end_idx]), list(graphs[start_idx:end_idx])))
    return out


def batch_tensor_bytes(batch: Any) -> int:
    total = 0
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        if torch.is_tensor(value):
            total += int(value.numel() * value.element_size())
    return total


def batch_device(batch: Any) -> Optional[torch.device]:
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        if torch.is_tensor(value):
            return value.device
    return None


def batch_to_device(batch: Any, device: torch.device) -> Any:
    current = batch_device(batch)
    if current is not None and current == device:
        return batch
    return batch.to(device)


def collate_analysis_batches(
    runner: Any,
    graphs: Sequence[Any],
    graph_indices: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
    cache_mode: str,
    gpu_cache_limit_gb: float,
    log=print,
) -> tuple[list[tuple[list[int], Any]], str, int]:
    """Pre-collate selected analysis graphs once.

    Knockouts and ranked-ablation curves rerun the same selected graphs many times.
    Caching avoids repeatedly rebuilding pair tensors and, when feasible, avoids repeated
    CPU-to-GPU transfers.
    """

    if cache_mode == "none":
        log("[cache] analysis batch cache disabled; batches will be collated on demand")
        return [], "none", 0

    grouped = graph_batches(graphs, graph_indices, batch_size)
    cpu_batches: list[tuple[list[int], Any]] = []
    total_bytes = 0
    start = time.time()
    for indices, batch_graphs in grouped:
        batch = runner.collate_graphs(batch_graphs)
        total_bytes += batch_tensor_bytes(batch)
        cpu_batches.append((indices, batch))

    selected_mode = cache_mode
    if selected_mode == "auto":
        if device.type == "cuda" and (total_bytes / (1024**3)) <= gpu_cache_limit_gb:
            selected_mode = "gpu"
        else:
            selected_mode = "cpu"
    if selected_mode not in {"cpu", "gpu"}:
        raise ValueError(f"unknown batch cache mode {cache_mode!r}")
    if selected_mode == "gpu" and device.type != "cuda":
        log("[cache] requested GPU batch cache on non-CUDA device; using CPU cache")
        selected_mode = "cpu"

    if selected_mode == "gpu":
        cached = [(indices, batch.to(device)) for indices, batch in cpu_batches]
        if device.type == "cuda":
            torch.cuda.synchronize()
    else:
        cached = cpu_batches

    log(
        f"[cache] prepared {len(cached)} analysis batches in {time.time() - start:.1f}s; "
        f"mode={selected_mode} approx_cpu_bytes={total_bytes / (1024**3):.3f} GiB"
    )
    return cached, selected_mode, total_bytes


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
    pair_state_input: Optional[torch.Tensor]
    pair_state_output: Optional[torch.Tensor]
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
        node_msg, pair_msg, logits, _edge_state = grit_attention_components(
            attention_module,
            pyg_batch,
        )
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
            pair_state_input=None
            if getattr(pyg_batch, "edge_attr", None) is None
            else pyg_batch.edge_attr.detach().float(),
            pair_state_output=None
            if getattr(pyg_batch, "wE", None) is None
            else pyg_batch.wE.detach().float(),
            head_output=head_output,
            heads=heads,
            message_dim=int(total_msg.size(-1)),
        )


def grit_attention_components(
    attention_module: nn.Module,
    pyg_batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return node_msg, pair_msg, logits, edge_state_for_value


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


def load_teacher_kernels(path: Optional[Path]) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    payload = torch.load(path.expanduser(), map_location="cpu", weights_only=False)
    if torch.is_tensor(payload):
        return {"teacher_kernel": payload.float()}
    if isinstance(payload, Mapping):
        out = {}
        for key, value in payload.items():
            if not torch.is_tensor(value):
                raise ValueError(f"teacher kernel {key!r} is not a tensor")
            out[str(key)] = value.float()
        return out
    raise ValueError("--teacher-kernel-pt must contain a tensor or a mapping of name -> tensor")


def teacher_operator_masks(
    kernels: Mapping[str, torch.Tensor],
    graph_indices: Sequence[int],
    batch: Any,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if not kernels:
        return out
    bsz, nmax = batch.node_mask.shape
    for name, kernel in kernels.items():
        if kernel.dim() == 2:
            selected = kernel[None, :, :].expand(bsz, -1, -1)
        elif kernel.dim() == 3:
            if max(graph_indices, default=-1) < kernel.size(0):
                selected = kernel[list(graph_indices)]
            elif kernel.size(0) == bsz:
                selected = kernel
            else:
                raise ValueError(
                    f"teacher kernel {name!r} first dimension {kernel.size(0)} "
                    f"does not cover graph indices {graph_indices[:3]}..."
                )
        else:
            raise ValueError(f"teacher kernel {name!r} must have shape [N,N] or [G,N,N]")
        dense = torch.zeros(bsz, nmax, nmax, dtype=torch.float32, device=batch.node_mask.device)
        h = min(nmax, selected.size(-2))
        w = min(nmax, selected.size(-1))
        dense[:, :h, :w] = selected[:, :h, :w].to(device=batch.node_mask.device, dtype=torch.float32)
        dense = dense * valid_pair_mask(batch).float()
        out[f"teacher__{name}"] = dense
    return out


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
            self._add_layer(record, masks, global_mask, graph_indices, valid)

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
        graph_indices: Sequence[int],
        valid: torch.Tensor,
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

                self._add_graph_rows(
                    record,
                    graph_indices,
                    valid,
                    name,
                    mask,
                    component_name,
                    influence,
                    contribution_norm,
                    attention,
                    global_values,
                )

    def _add_graph_rows(
        self,
        record: SparseLayerCapture,
        graph_indices: Sequence[int],
        valid: torch.Tensor,
        operator: str,
        mask: torch.Tensor,
        component: str,
        influence: torch.Tensor,
        contribution_norm: torch.Tensor,
        attention: torch.Tensor,
        global_values: torch.Tensor,
    ) -> None:
        for local_graph, graph_index in enumerate(graph_indices):
            edge_mask = record.graph == local_graph
            if not bool(edge_mask.any()):
                continue
            op_values = mask[
                record.graph[edge_mask],
                record.local_dst[edge_mask],
                record.local_src[edge_mask],
            ].float()
            influence_g = influence[edge_mask]
            contribution_g = contribution_norm[edge_mask]
            attention_g = attention[edge_mask]
            global_g = global_values[edge_mask]
            base_rate = safe_div(
                float((mask[local_graph] * valid[local_graph]).sum().item()),
                float(valid[local_graph].sum().item()),
            )
            influence_total = to_float(influence_g.sum())
            influence_on = to_float((influence_g * op_values[:, None]).sum())
            abs_transport_total = to_float(contribution_g.sum())
            abs_transport_on = to_float((contribution_g * op_values[:, None]).sum())
            attention_total = to_float(attention_g.sum())
            attention_on = to_float((attention_g * op_values[:, None]).sum())
            global_influence = to_float((influence_g * global_g[:, None]).sum())
            global_abs_transport = to_float((contribution_g * global_g[:, None]).sum())
            influence_fraction = safe_div(influence_on, influence_total)
            abs_transport_fraction = safe_div(abs_transport_on, abs_transport_total)
            attention_fraction = safe_div(attention_on, attention_total)
            influence_lift = safe_div(influence_fraction, base_rate)
            global_task_transport = safe_div(global_influence, influence_total)
            self.graph_rows.append(
                {
                    "graph_index": int(graph_index),
                    "layer": record.layer,
                    "head": "all",
                    "heads_aggregated": record.heads,
                    "aggregation_unit": "graph_layer_all_heads",
                    "operator": operator,
                    "transport_component": component,
                    "base_rate": base_rate,
                    "influence_fraction": influence_fraction,
                    "influence_lift": influence_lift,
                    "abs_transport_fraction": abs_transport_fraction,
                    "abs_transport_lift": safe_div(abs_transport_fraction, base_rate),
                    "attention_fraction": attention_fraction,
                    "attention_lift": safe_div(attention_fraction, base_rate),
                    "global_task_transport": global_task_transport,
                    "global_abs_transport": safe_div(global_abs_transport, abs_transport_total),
                    "transport_lift_global_gate": (
                        influence_lift * global_task_transport
                        if math.isfinite(influence_lift)
                        and math.isfinite(global_task_transport)
                        else float("nan")
                    ),
                    "influence_on_operator": influence_on,
                    "influence_total": influence_total,
                    "abs_transport_on_operator": abs_transport_on,
                    "abs_transport_total": abs_transport_total,
                    "attention_on_operator": attention_on,
                    "attention_total": attention_total,
                }
            )

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
                if is_primary_operator(str(row["operator"]))
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

    def base_rate_rows(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for operator, values in sorted(self.base.items()):
            matched_operator = (
                operator.replace("random__", "", 1)
                if operator.startswith("random__")
                else ""
            )
            out.append(
                {
                    "operator": operator,
                    "matched_operator": matched_operator,
                    "is_primary_operator": is_primary_operator(operator),
                    "graphs": self.graph_count,
                    "positive_mass": values["positive"],
                    "valid_pair_denominator": values["denominator"],
                    "base_rate": safe_div(values["positive"], values["denominator"]),
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


def pyg_sparse_softmax(logits: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    try:
        from torch_geometric.utils import softmax

        return softmax(logits, index, num_nodes=num_nodes)
    except Exception:
        out = torch.zeros_like(logits)
        for node in torch.unique(index.detach().cpu()).tolist():
            mask = index == int(node)
            out[mask] = torch.softmax(logits[mask], dim=0)
        return out


def sparse_pair_strata(pyg_batch: Any, official_batch: Any) -> torch.Tensor:
    src = pyg_batch.edge_index[0].long()
    dst = pyg_batch.edge_index[1].long()
    graph, local_src, local_dst = edge_local_coordinates(
        src,
        dst,
        [int(v) for v in pyg_batch.graph_num_nodes.detach().cpu().tolist()],
    )
    degree_bins = degree_bin(official_batch.degree.float())
    directed = sparse_edge_mask(official_batch).long()
    local = (official_batch.adj > 0).long()
    spd = official_batch.spd.long().clamp_min(0).clamp_max(1024)
    code = directed[graph, local_dst, local_src]
    code = code * 2 + local[graph, local_dst, local_src]
    code = code * 2048 + spd[graph, local_dst, local_src]
    code = code * 16 + degree_bins[graph, local_dst].long()
    code = code * 16 + degree_bins[graph, local_src].long()
    return code.to(device=src.device)


def sparse_support_masks(pyg_batch: Any, official_batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    src = pyg_batch.edge_index[0].long()
    dst = pyg_batch.edge_index[1].long()
    graph, local_src, local_dst = edge_local_coordinates(
        src,
        dst,
        [int(v) for v in pyg_batch.graph_num_nodes.detach().cpu().tolist()],
    )
    is_self = local_src == local_dst
    is_local = official_batch.adj[graph, local_dst, local_src] > 0
    local_or_self = is_local | is_self
    global_pair = (~is_local) & (~is_self)
    return local_or_self, global_pair


def ensure_receiver_support(
    keep: torch.Tensor,
    dst: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Avoid all-masked softmax groups by falling back per receiver when needed."""

    keep = keep.bool().clone()
    fallback = fallback.bool()
    for receiver in torch.unique(dst.detach().cpu()).tolist():
        receiver_mask = dst == int(receiver)
        if bool((keep & receiver_mask).any()):
            continue
        receiver_fallback = fallback & receiver_mask
        if bool(receiver_fallback.any()):
            keep[receiver_fallback] = True
        else:
            keep[receiver_mask] = True
    return keep


def matched_mean_sparse(values: torch.Tensor, strata: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(values)
    flat_strata = strata.detach().cpu()
    for code in torch.unique(flat_strata).tolist():
        mask = strata == int(code)
        if bool(mask.any()):
            out[mask] = values[mask].mean(dim=0, keepdim=True)
    return out


def permute_sparse_within_strata(
    values: torch.Tensor,
    strata: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    out = values.clone()
    for code in torch.unique(strata.detach().cpu()).tolist():
        mask = strata == int(code)
        idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if idx.numel() <= 1:
            continue
        order = torch.randperm(idx.numel(), generator=generator).to(device=idx.device)
        out[idx] = values[idx[order]]
    return out


class GRITAblationContext:
    """Inference-time GRIT ablations from the memo.

    The implementation patches only the official GRIT attention modules during
    the context lifetime. It leaves checkpoint weights unchanged and restores all
    methods/hooks afterwards.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        mode: str,
        selected_heads: Optional[Mapping[int, set[int]]] = None,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.mode = mode
        self.selected_heads = {int(k): set(v) for k, v in (selected_heads or {}).items()}
        self.seed = int(seed)
        self.current_batch: Any = None
        self.initial_edge_attr: Optional[torch.Tensor] = None
        self.original_methods: list[tuple[Any, Any]] = []
        self.handles: list[Any] = []
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def __enter__(self) -> "GRITAblationContext":
        self.handles.append(self.model.register_forward_pre_hook(self._model_pre_hook))
        for layer_idx, layer in enumerate(self.model.layers):
            attention = layer.attention
            original = attention.propagate_attention
            self.original_methods.append((attention, original))
            attention.propagate_attention = self._make_propagate(attention, layer_idx)
            if self.mode in {"frozen_pair_state", "permuted_pair_state"}:
                self.handles.append(attention.register_forward_pre_hook(self._pair_state_pre_hook))
        return self

    def __exit__(self, *_exc: object) -> None:
        for attention, original in self.original_methods:
            attention.propagate_attention = original
        for handle in self.handles:
            handle.remove()
        self.original_methods = []
        self.handles = []
        self.current_batch = None
        self.initial_edge_attr = None

    def _model_pre_hook(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        self.current_batch = inputs[0]
        self.initial_edge_attr = None

    def _pair_state_pre_hook(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        pyg_batch = inputs[0]
        if getattr(pyg_batch, "edge_attr", None) is None:
            return
        if self.initial_edge_attr is None:
            self.initial_edge_attr = pyg_batch.edge_attr.detach().clone()
        if self.mode == "frozen_pair_state":
            pyg_batch.edge_attr = self.initial_edge_attr.to(device=pyg_batch.edge_attr.device)
        elif self.mode == "permuted_pair_state":
            if self.current_batch is None:
                raise RuntimeError("missing OfficialBatch for pair-state permutation")
            strata = sparse_pair_strata(pyg_batch, self.current_batch)
            pyg_batch.edge_attr = permute_sparse_within_strata(
                pyg_batch.edge_attr,
                strata,
                self.generator,
            )

    def _make_propagate(self, attention_module: nn.Module, layer_idx: int):
        def propagate(pyg_batch: Any) -> None:
            from torch_scatter import scatter

            if self.current_batch is None:
                raise RuntimeError("missing OfficialBatch during GRIT ablation")
            node_msg, pair_msg, logits, edge_state = grit_attention_components(
                attention_module,
                pyg_batch,
            )
            logits = self._ablate_logits(attention_module, pyg_batch, logits)
            if self.mode == "no_pair_value_transport":
                strata = sparse_pair_strata(pyg_batch, self.current_batch)
                pair_msg = matched_mean_sparse(pair_msg, strata)

            score = pyg_sparse_softmax(
                logits.unsqueeze(-1),
                pyg_batch.edge_index[1],
                pyg_batch.num_nodes,
            )
            score = attention_module.dropout(score)
            pyg_batch.attn = score
            if getattr(pyg_batch, "E", None) is not None:
                pyg_batch.wE = edge_state.flatten(1)
            message = node_msg + pair_msg
            weighted = message * score
            pyg_batch.wV = torch.zeros_like(pyg_batch.V_h)
            scatter(weighted, pyg_batch.edge_index[1], dim=0, out=pyg_batch.wV, reduce="add")
            for head in self.selected_heads.get(layer_idx, set()):
                if 0 <= int(head) < pyg_batch.wV.size(1):
                    for graph_idx in torch.unique(pyg_batch.batch.detach().cpu()).tolist():
                        node_mask = pyg_batch.batch == int(graph_idx)
                        pyg_batch.wV[node_mask, int(head), :] = pyg_batch.wV[
                            node_mask,
                            int(head),
                            :,
                        ].mean(dim=0, keepdim=True)

        return propagate

    def _ablate_logits(
        self,
        attention_module: nn.Module,
        pyg_batch: Any,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "clean" or self.mode == "no_pair_value_transport":
            return logits
        if self.mode in {"local_only_support", "global_only_support"}:
            local_or_self, global_pair = sparse_support_masks(pyg_batch, self.current_batch)
            keep = local_or_self if self.mode == "local_only_support" else global_pair
            fallback = (
                local_or_self
                if self.mode == "global_only_support"
                else torch.ones_like(keep)
            )
            keep = ensure_receiver_support(keep, pyg_batch.edge_index[1].long(), fallback)
            return logits.masked_fill(~keep[:, None], -1.0e9)
        if self.mode == "no_structural_routing":
            content = (
                pyg_batch.K_h[pyg_batch.edge_index[0]]
                + pyg_batch.Q_h[pyg_batch.edge_index[1]]
            )
            content = attention_module.act(content)
            content_logits = torch.einsum("ehd,dhc->ehc", content, attention_module.Aw).squeeze(-1)
            if attention_module.clamp is not None:
                content_logits = torch.clamp(
                    content_logits,
                    min=-float(attention_module.clamp),
                    max=float(attention_module.clamp),
                )
            structural_bias = logits - content_logits
            strata = sparse_pair_strata(pyg_batch, self.current_batch)
            return content_logits + matched_mean_sparse(
                structural_bias.unsqueeze(-1),
                strata,
            ).squeeze(-1)
        if self.mode in {"frozen_pair_state", "permuted_pair_state"}:
            return logits
        if self.mode == "head_ablation":
            return logits
        raise ValueError(f"unknown GRIT ablation mode {self.mode!r}")


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


@torch.no_grad()
def evaluate_primary_series(
    runner: Any,
    model: nn.Module,
    dataset: Any,
    *,
    batch_size: int,
    device: torch.device,
    target_stats: Optional[Mapping[str, float]],
    cached_batches: Optional[Sequence[tuple[Sequence[int], Any]]] = None,
    autocast_dtype: str = "none",
) -> tuple[dict[str, float], list[float]]:
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    per_graph: list[float] = []
    start = time.time()
    model.eval()
    if cached_batches:
        iterator = (batch for _indices, batch in cached_batches)
    else:
        iterator = runner.make_loader(dataset, batch_size, shuffle=False, seed=0)
    for batch in iterator:
        batch = batch_to_device(batch, device)
        with autocast_context(device, autocast_dtype):
            pred = model(batch)
        if batch.task_type == "edge_binary":
            preds.append(pred.detach().cpu())
            targets.append(batch.edge_target.detach().cpu())
            for graph_idx in range(batch.num_graphs):
                mask = batch.edge_batch == graph_idx
                if bool(mask.any()):
                    metrics = runner.binary_metrics(
                        pred[mask].detach().cpu(),
                        batch.edge_target[mask].detach().cpu(),
                    )
                    per_graph.append(float(metrics["f1"]))
        elif batch.task_type == "node_binary":
            node_pred = runner.node_predictions_for_loss(pred, batch)
            preds.append(node_pred.detach().cpu())
            targets.append(runner.flatten_node_target(batch).detach().cpu())
            offset = 0
            flat_target = runner.flatten_node_target(batch)
            for graph_idx, count in enumerate(batch.graph_num_nodes.detach().cpu().tolist()):
                count = int(count)
                metrics = runner.binary_metrics(
                    node_pred[offset : offset + count].detach().cpu(),
                    flat_target[offset : offset + count].detach().cpu(),
                )
                per_graph.append(float(metrics["f1"]))
                offset += count
        elif batch.task_type == "graph_regression":
            preds.append(pred.detach().cpu())
            targets.append(batch.graph_target.detach().cpu())
            pred_raw = runner.denormalize_graph_target(pred.detach().cpu(), target_stats)
            err = (pred_raw - batch.graph_target.detach().cpu()).abs()
            per_graph.extend(float(v) for v in err.reshape(-1))
        else:
            preds.append(pred.detach().cpu())
            targets.append(batch.graph_target.detach().cpu())
    seconds = time.time() - start
    pred_all = torch.cat(preds) if preds else torch.empty(0)
    target_all = torch.cat(targets) if targets else torch.empty(0)
    task_type = dataset[0].task_type
    if task_type in {"edge_binary", "node_binary"}:
        metrics = runner.binary_metrics(pred_all, target_all)
        metrics["primary"] = metrics["f1"]
        metrics["higher_is_better"] = 1.0
    elif task_type == "graph_regression":
        metrics = runner.flow_metrics(pred_all, target_all, target_stats)
        metrics["primary"] = metrics["raw_mae"]
        metrics["higher_is_better"] = 0.0
    else:
        metrics = runner.regression_metrics(pred_all, target_all)
        metrics["primary"] = metrics["mae"]
        metrics["higher_is_better"] = 0.0
    metrics["primary_graph_mean"] = sum(per_graph) / max(1, len(per_graph))
    metrics["eval_seconds"] = seconds
    return metrics, per_graph


def bootstrap_ci(values: Sequence[float], seed: int, samples: int = 1000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def primary_drop(clean: Mapping[str, float], ablated: Mapping[str, float]) -> float:
    if float(clean.get("higher_is_better", 0.0)) > 0.5:
        return float(clean["primary"]) - float(ablated["primary"])
    return float(ablated["primary"]) - float(clean["primary"])


def finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def preferred_drop_value(row: Mapping[str, Any]) -> float:
    paired = finite_or_nan(row.get("paired_drop_graph_mean"))
    if math.isfinite(paired):
        return paired
    return finite_or_nan(row.get("raw_drop"))


def paired_drop_series(
    clean_metrics: Mapping[str, float],
    clean_series: Sequence[float],
    ablated_series: Sequence[float],
) -> list[float]:
    higher = float(clean_metrics.get("higher_is_better", 0.0)) > 0.5
    count = min(len(clean_series), len(ablated_series))
    if higher:
        return [float(clean_series[i]) - float(ablated_series[i]) for i in range(count)]
    return [float(ablated_series[i]) - float(clean_series[i]) for i in range(count)]


def advantage_removed(
    clean: Mapping[str, float],
    ablated: Mapping[str, float],
    baseline_primary: Optional[float],
) -> float:
    if baseline_primary is None or not math.isfinite(float(baseline_primary)):
        return float("nan")
    higher = float(clean.get("higher_is_better", 0.0)) > 0.5
    if higher:
        denom = float(clean["primary"]) - float(baseline_primary)
        return primary_drop(clean, ablated) / (denom + EPS)
    denom = float(baseline_primary) - float(clean["primary"])
    return primary_drop(clean, ablated) / (denom + EPS)


def evaluate_with_ablation(
    loaded: LoadedExperiment,
    args: argparse.Namespace,
    *,
    mode: str,
    selected_heads: Optional[Mapping[int, set[int]]] = None,
) -> tuple[dict[str, float], list[float]]:
    context_mode = "clean" if mode == "clean" else mode
    if context_mode == "clean":
        return evaluate_primary_series(
            loaded.runner,
            loaded.model,
            loaded.subset_dataset,
            batch_size=max(1, int(args.eval_batch_size or args.batch_size)),
            device=loaded.device,
            target_stats=loaded.target_stats,
            cached_batches=loaded.analysis_batches,
            autocast_dtype=args.autocast_dtype,
        )
    with GRITAblationContext(
        loaded.model,
        mode=context_mode,
        selected_heads=selected_heads,
        seed=args.random_seed,
    ):
        return evaluate_primary_series(
            loaded.runner,
            loaded.model,
            loaded.subset_dataset,
            batch_size=max(1, int(args.eval_batch_size or args.batch_size)),
            device=loaded.device,
            target_stats=loaded.target_stats,
            cached_batches=loaded.analysis_batches,
            autocast_dtype=args.autocast_dtype,
        )


def run_knockouts(args: argparse.Namespace, loaded: Optional[LoadedExperiment] = None) -> None:
    loaded = loaded or load_experiment(args)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ablations = parse_csv_list(
        args.knockout_ablations,
        default=(
            "local_only_support",
            "global_only_support",
            "no_structural_routing",
            "no_pair_value_transport",
            "frozen_pair_state",
            "permuted_pair_state",
        ),
        all_values=(
            "local_only_support",
            "global_only_support",
            "no_structural_routing",
            "no_pair_value_transport",
            "frozen_pair_state",
            "permuted_pair_state",
        ),
    )
    clean_metrics, clean_series = evaluate_with_ablation(loaded, args, mode="clean")
    clean_lo, clean_hi = bootstrap_ci(clean_series, args.random_seed)
    rows: list[dict[str, Any]] = [
        {
            "ablation": "clean",
            **MECHANISM_ABLATION_METADATA["clean"],
            "primary": clean_metrics["primary"],
            "primary_graph_mean": clean_metrics["primary_graph_mean"],
            "primary_ci_low": clean_lo,
            "primary_ci_high": clean_hi,
            "raw_drop": 0.0,
            "paired_drop_graph_mean": 0.0,
            "raw_drop_ci_low": 0.0,
            "raw_drop_ci_high": 0.0,
            "advantage_removed": 0.0,
        }
    ]
    for ablation in ablations:
        print(f"[knockout] evaluating {ablation}", flush=True)
        metrics, series = evaluate_with_ablation(loaded, args, mode=ablation)
        drop_series = paired_drop_series(clean_metrics, clean_series, series)
        drop_lo, drop_hi = bootstrap_ci(drop_series, args.random_seed + len(rows))
        lo, hi = bootstrap_ci(series, args.random_seed + len(rows))
        raw_drop = primary_drop(clean_metrics, metrics)
        paired_drop = mean_or_nan(drop_series)
        rows.append(
            {
                "ablation": ablation,
                **MECHANISM_ABLATION_METADATA[ablation],
                "primary": metrics["primary"],
                "primary_graph_mean": metrics["primary_graph_mean"],
                "primary_ci_low": lo,
                "primary_ci_high": hi,
                "raw_drop": raw_drop,
                "paired_drop_graph_mean": paired_drop,
                "raw_drop_ci_low": drop_lo,
                "raw_drop_ci_high": drop_hi,
                "advantage_removed": advantage_removed(
                    clean_metrics,
                    metrics,
                    args.baseline_primary,
                ),
            }
        )
    write_csv(out_dir / "mechanism_knockouts.csv", rows)
    plot_knockouts(out_dir / "mechanism_knockouts.csv", out_dir / "figures")
    print(f"[done] wrote knockout results to {out_dir / 'mechanism_knockouts.csv'}", flush=True)


def plot_knockouts(csv_path: Path, figures: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[plot] skipped knockout plot: {exc}", flush=True)
        return
    df = pd.read_csv(csv_path)
    df = df[df["ablation"] != "clean"]
    if df.empty:
        return
    if "display_label" not in df.columns:
        df["display_label"] = df["ablation"]
    if "mechanism_axis" not in df.columns:
        df["mechanism_axis"] = "mechanism"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.2, 5.2), dpi=180)
    x = list(range(len(df)))
    y_col = "paired_drop_graph_mean" if "paired_drop_graph_mean" in df.columns else "raw_drop"
    y = df[y_col].to_numpy(dtype=float)
    low = df["raw_drop_ci_low"].to_numpy(dtype=float)
    high = df["raw_drop_ci_high"].to_numpy(dtype=float)
    yerr = [
        torch.as_tensor(y - low).clamp_min(0.0).numpy(),
        torch.as_tensor(high - y).clamp_min(0.0).numpy(),
    ]
    axis_colors = {
        "support": "#4c78a8",
        "support_diagnostic": "#9ecae9",
        "routing": "#f58518",
        "transport": "#59a14f",
        "pair_state": "#b279a2",
    }
    colors = [axis_colors.get(axis, "#777777") for axis in df["mechanism_axis"].astype(str)]
    ax.bar(x, y, color=colors)
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="none",
        ecolor="0.15",
        elinewidth=1.0,
        capsize=3,
    )
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xticks(x, labels=df["display_label"], rotation=30, ha="right")
    ax.set_ylabel("paired graph-mean performance drop")
    ax.set_title("M2: support-routing-transport knockouts")
    ax.text(
        0.01,
        0.98,
        (
            "Hypothesis: pair-value / pair-state ablations should hurt more "
            "than routing/support controls"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=axis)
        for axis, color in axis_colors.items()
        if axis in set(df["mechanism_axis"].astype(str))
    ]
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "mechanism_knockout_drops.png")
    plt.close(fig)


def load_atlas_rows(path: Path) -> list[dict[str, Any]]:
    with path.expanduser().open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def component_scores_from_atlas(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int], dict[str, float]]:
    scores: dict[tuple[int, int], dict[str, float]] = defaultdict(float_dict)
    for row in rows:
        if row.get("transport_component") != "total":
            continue
        operator = str(row.get("operator", ""))
        if not is_primary_operator(operator):
            continue
        key = (int(row["layer"]), int(row["head"]))
        scores[key]["attention_mass"] += float(row.get("attention_total", 0.0) or 0.0)
        scores[key]["output_norm"] += float(row.get("abs_transport_total", 0.0) or 0.0)
        for metric in [
            "attention_lift",
            "abs_transport_lift",
            "influence_lift",
            "transport_lift_global_gate",
            "operator_transport_score",
            "global_task_transport",
        ]:
            value = float(row.get(metric, "nan") or "nan")
            if math.isfinite(value):
                scores[key][metric] = max(scores[key].get(metric, -float("inf")), value)
    return scores


def ranked_components(
    scores: Mapping[tuple[int, int], Mapping[str, float]],
    ranking: str,
    seed: int,
) -> list[tuple[int, int, float]]:
    items = [(layer, head, dict(values)) for (layer, head), values in scores.items()]
    if ranking == "random":
        rng = random.Random(seed)
        rng.shuffle(items)
        return [(layer, head, float("nan")) for layer, head, _values in items]
    metric_for_ranking = {
        "norm": "output_norm",
        "attention_mass": "attention_mass",
        "attention_kernel_alignment": "attention_lift",
        "contribution_alignment": "abs_transport_lift",
        "influence_lift": "influence_lift",
        "global_task_transport": "global_task_transport",
        "transport_lift_global_gate": "transport_lift_global_gate",
        "ots": "operator_transport_score",
    }[ranking]
    ranked = [
        (layer, head, float(values.get(metric_for_ranking, float("nan"))))
        for layer, head, values in items
    ]
    ranked.sort(key=lambda item: nan_to_neg_inf(item[2]), reverse=True)
    return ranked


def selected_head_map(
    ranked: Sequence[tuple[int, int, float]],
    percent: float,
) -> dict[int, set[int]]:
    count = max(1, int(math.ceil(len(ranked) * float(percent) / 100.0)))
    selected: dict[int, set[int]] = defaultdict(set)
    for layer, head, _score in ranked[:count]:
        selected[int(layer)].add(int(head))
    return selected


def run_ranked_ablation(
    args: argparse.Namespace,
    loaded: Optional[LoadedExperiment] = None,
) -> None:
    loaded = loaded or load_experiment(args)
    atlas_path = args.atlas_per_head_csv or (args.output_dir / "operator_transport_per_head.csv")
    if not atlas_path.exists():
        raise RuntimeError(
            f"missing atlas per-head CSV: {atlas_path}. Run --experiments atlas first "
            "or pass --atlas-per-head-csv."
        )
    scores = component_scores_from_atlas(load_atlas_rows(atlas_path))
    if not scores:
        raise RuntimeError(f"no rankable layer/head components found in {atlas_path}")
    rankings = parse_csv_list(
        args.rankings,
        default=(
            "random",
            "norm",
            "attention_mass",
            "attention_kernel_alignment",
            "contribution_alignment",
            "influence_lift",
            "transport_lift_global_gate",
            "ots",
        ),
        all_values=(
            "random",
            "norm",
            "attention_mass",
            "attention_kernel_alignment",
            "contribution_alignment",
            "influence_lift",
            "global_task_transport",
            "transport_lift_global_gate",
            "ots",
        ),
    )
    percents = [float(value) for value in args.topk_percents.split(",") if value.strip()]
    clean_metrics, _clean_series = evaluate_with_ablation(loaded, args, mode="clean")
    rows: list[dict[str, Any]] = []
    for ranking in rankings:
        ranked = ranked_components(scores, ranking, args.random_seed)
        if ranking != "random" and not any(math.isfinite(score) for _l, _h, score in ranked):
            print(f"[ranked] skipping undefined ranking={ranking}", flush=True)
            continue
        for percent in percents:
            selected = selected_head_map(ranked, percent)
            print(
                f"[ranked] ranking={ranking} top={percent:g}% "
                f"heads={sum(len(v) for v in selected.values())}",
                flush=True,
            )
            metrics, series = evaluate_with_ablation(
                loaded,
                args,
                mode="head_ablation",
                selected_heads=selected,
            )
            drop_series = paired_drop_series(clean_metrics, _clean_series, series)
            drop_lo, drop_hi = bootstrap_ci(
                drop_series,
                args.random_seed + int(percent * 17) + len(rows),
            )
            lo, hi = bootstrap_ci(series, args.random_seed + int(percent * 17) + len(rows))
            raw_drop = primary_drop(clean_metrics, metrics)
            paired_drop = mean_or_nan(drop_series)
            rows.append(
                {
                    "ranking": ranking,
                    "ranking_label": RANKING_DISPLAY_LABELS.get(ranking, ranking),
                    "ranking_family": RANKING_FAMILIES.get(ranking, "other"),
                    "percent_ablated": percent,
                    "heads_ablated": sum(len(v) for v in selected.values()),
                    "primary": metrics["primary"],
                    "primary_graph_mean": metrics["primary_graph_mean"],
                    "primary_ci_low": lo,
                    "primary_ci_high": hi,
                    "raw_drop": raw_drop,
                    "paired_drop_graph_mean": paired_drop,
                    "raw_drop_ci_low": drop_lo,
                    "raw_drop_ci_high": drop_hi,
                    "advantage_removed": advantage_removed(
                        clean_metrics,
                        metrics,
                        args.baseline_primary,
                    ),
                }
            )
    out_path = args.output_dir / "metric_ranked_ablation_curves.csv"
    auc_path = args.output_dir / "metric_ranked_ablation_auc.csv"
    write_csv(out_path, rows)
    write_csv(auc_path, ranked_ablation_auc_rows(rows))
    plot_ranked_ablation(out_path, args.output_dir / "figures")
    plot_ranked_ablation_auc(auc_path, args.output_dir / "figures")
    print(f"[done] wrote ranked ablation curves to {out_path}", flush=True)


def ranked_ablation_auc_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    metadata: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        ranking = str(row["ranking"])
        metadata.setdefault(ranking, row)
        grouped[ranking].append((float(row["percent_ablated"]), preferred_drop_value(row)))
    out = []
    for ranking, points in sorted(grouped.items()):
        points = sorted((x, y) for x, y in points if math.isfinite(y))
        auc = 0.0
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            auc += 0.5 * (y0 + y1) * (x1 - x0)
        norm = max(1.0, points[-1][0] - points[0][0]) if points else 1.0
        row_meta = metadata.get(ranking, {})
        out.append(
            {
                "ranking": ranking,
                "ranking_label": row_meta.get(
                    "ranking_label",
                    RANKING_DISPLAY_LABELS.get(ranking, ranking),
                ),
                "ranking_family": row_meta.get(
                    "ranking_family",
                    RANKING_FAMILIES.get(ranking, "other"),
                ),
                "drop_metric": "paired_drop_graph_mean",
                "ablation_auc": auc,
                "normalised_auc": auc / norm,
            }
        )
    return out


def plot_ranked_ablation(csv_path: Path, figures: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[plot] skipped ranked-ablation plot: {exc}", flush=True)
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=180)
    for ranking, group in df.groupby("ranking"):
        group = group.sort_values("percent_ablated")
        y_col = (
            "paired_drop_graph_mean"
            if "paired_drop_graph_mean" in group.columns
            else "raw_drop"
        )
        label = group.get("ranking_label", group["ranking"]).iloc[0]
        ax.plot(group["percent_ablated"], group[y_col], marker="o", label=label)
        if {"raw_drop_ci_low", "raw_drop_ci_high"}.issubset(group.columns):
            y = group[y_col].to_numpy(dtype=float)
            low = group["raw_drop_ci_low"].to_numpy(dtype=float)
            high = group["raw_drop_ci_high"].to_numpy(dtype=float)
            ax.fill_between(
                group["percent_ablated"].to_numpy(dtype=float),
                y - torch.as_tensor(y - low).clamp_min(0.0).numpy(),
                y + torch.as_tensor(high - y).clamp_min(0.0).numpy(),
                alpha=0.12,
            )
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xlabel("percent layer-head components ablated")
    ax.set_ylabel("paired graph-mean performance drop")
    ax.set_title("M4: metric-ranked causal ablation")
    ax.text(
        0.01,
        0.98,
        "Hypothesis: operator/transport rankings degrade performance fastest",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "metric_ranked_ablation_curves.png")
    plt.close(fig)


def plot_ranked_ablation_auc(csv_path: Path, figures: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[plot] skipped ranked-ablation AUC plot: {exc}", flush=True)
        return
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    if "ranking_label" not in df.columns:
        df["ranking_label"] = df["ranking"].map(
            lambda ranking: RANKING_DISPLAY_LABELS.get(str(ranking), str(ranking))
        )
    if "ranking_family" not in df.columns:
        df["ranking_family"] = df["ranking"].map(
            lambda ranking: RANKING_FAMILIES.get(str(ranking), "other")
        )
    df = df.sort_values("normalised_auc", ascending=False)
    figures.mkdir(parents=True, exist_ok=True)
    family_colors = {
        "control": "#9d9d9d",
        "attention_control": "#f58518",
        "transport": "#59a14f",
        "task_weighted_transport": "#4c78a8",
    }
    colors = [family_colors.get(family, "#777777") for family in df["ranking_family"]]
    labels = df["ranking_label"]
    x = list(range(len(df)))
    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=180)
    ax.bar(x, df["normalised_auc"].to_numpy(dtype=float), color=colors)
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xticks(x, labels=labels, rotation=30, ha="right")
    ax.set_ylabel("ablation AUC, paired graph-mean drop")
    ax.set_title("M4 summary: causal ranking quality")
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=family)
        for family, color in family_colors.items()
        if family in set(df["ranking_family"].astype(str))
    ]
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "metric_ranked_ablation_auc.png")
    plt.close(fig)


def run_distillation_calibration(
    args: argparse.Namespace,
    loaded: Optional[LoadedExperiment] = None,
) -> None:
    if args.teacher_kernel_pt is None:
        raise RuntimeError(
            "distillation_calibration requires --teacher-kernel-pt. The memo's M1 "
            "experiment depends on a teacher/operator kernel K_t; this script will "
            "not fabricate one from GraphBench labels."
        )
    loaded = loaded or load_experiment(args)
    kernels = load_teacher_kernels(args.teacher_kernel_pt)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_atlas(args, loaded)
    run_ranked_ablation(args, loaded)
    metadata = {
        "teacher_kernel_pt": str(args.teacher_kernel_pt),
        "status": (
            "teacher kernels used as known O_k masks for M1 calibration; "
            "atlas alignment and metric-ranked ablation data were generated"
        ),
        "teacher_kernels": sorted(kernels.keys()),
    }
    (out_dir / "distillation_calibration_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

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
        layer_total["operator"].astype(str).map(is_primary_operator)
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
    plot_operator_specificity_evidence(out_dir)


def heatmap(df: Any, value: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    pivot = df.pivot_table(index="layer", columns="operator", values=value, aggfunc="mean")
    if pivot.empty:
        return
    fig_width = max(8.0, 0.72 * max(1, len(pivot.columns)))
    fig, ax = plt.subplots(figsize=(fig_width, 4.6), dpi=180)
    image = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("operator")
    ax.set_ylabel("layer")
    ax.set_xticks(
        range(len(pivot.columns)),
        labels=[str(c) for c in pivot.columns],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(range(len(pivot.index)), labels=[str(i) for i in pivot.index])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def finite_float_values(values: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def mean_or_nan(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def median_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def max_finite(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else float("nan")


def safe_log2_ratio(real: Any, random_value: Any) -> float:
    try:
        numerator = max(float(real), EPS)
        denominator = max(float(random_value), EPS)
    except (TypeError, ValueError):
        return float("nan")
    ratio = numerator / denominator
    return math.log2(ratio) if math.isfinite(ratio) else float("nan")


def operator_specificity_summary_rows(merged: Any, metrics: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for operator, group in merged.groupby("operator_real"):
        for metric in metrics:
            column = f"log2_real_over_random_{metric}"
            row_values = finite_float_values(group[column].tolist())
            graph_values = []
            for _graph_index, graph_group in group.groupby("graph_index"):
                values = finite_float_values(graph_group[column].tolist())
                if values:
                    graph_values.append(mean_or_nan(values))
            lo, hi = bootstrap_ci(graph_values, seed=1337 + len(rows), samples=1000)
            rows.append(
                {
                    "operator": operator,
                    "metric": metric,
                    "ci_unit": "graph_mean_over_layers",
                    "samples": len(graph_values),
                    "graphs": len(graph_values),
                    "graph_layer_rows": len(row_values),
                    "mean_log2_real_over_random": mean_or_nan(graph_values),
                    "median_log2_real_over_random": median_or_nan(graph_values),
                    "row_mean_log2_real_over_random": mean_or_nan(row_values),
                    "positive_graph_fraction": safe_div(
                        sum(1.0 for value in graph_values if value > 0.0),
                        len(graph_values),
                    ),
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return rows


def plot_operator_specificity_evidence(out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[plot] skipped specificity evidence plots: {exc}", flush=True)
        return
    path = out_dir / "operator_transport_per_graph.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["transport_component"] == "total"].copy()
    if df.empty:
        return
    real = df[df["operator"].astype(str).map(is_primary_operator)].copy()
    random_df = df[df["operator"].astype(str).str.startswith("random__")].copy()
    if real.empty or random_df.empty:
        return
    random_df["matched_operator"] = random_df["operator"].astype(str).str.replace(
        "random__",
        "",
        regex=False,
    )
    keys = ["graph_index", "layer"]
    merged = real.merge(
        random_df,
        left_on=keys + ["operator"],
        right_on=keys + ["matched_operator"],
        suffixes=("_real", "_random"),
    )
    if merged.empty:
        return
    metrics = [
        "attention_lift",
        "abs_transport_lift",
        "influence_lift",
        "transport_lift_global_gate",
    ]
    for metric in metrics:
        merged[f"log2_real_over_random_{metric}"] = [
            safe_log2_ratio(real, random_value)
            for real, random_value in zip(
                merged[f"{metric}_real"].tolist(),
                merged[f"{metric}_random"].tolist(),
            )
        ]
    rows = operator_specificity_summary_rows(merged, metrics)
    write_csv(out_dir / "operator_specificity_effects.csv", rows)

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    influence = summary[summary["metric"] == "influence_lift"].sort_values(
        "mean_log2_real_over_random",
        ascending=False,
    )
    influence = influence[
        influence["mean_log2_real_over_random"].map(lambda value: math.isfinite(float(value)))
    ]
    if not influence.empty:
        fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=180)
        x = list(range(len(influence)))
        y = influence["mean_log2_real_over_random"].to_numpy(dtype=float)
        yerr = [
            y - influence["ci_low"].to_numpy(dtype=float),
            influence["ci_high"].to_numpy(dtype=float) - y,
        ]
        ax.bar(x, y, color="#4c78a8")
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.15", capsize=3, linewidth=1.0)
        ax.axhline(0.0, color="0.25", linewidth=0.8)
        ax.set_xticks(x, labels=influence["operator"], rotation=30, ha="right")
        ax.set_ylabel("log2(real operator lift / matched random lift)")
        ax.set_title("M3: operator specificity against matched null")
        ax.text(
            0.01,
            0.98,
            "Hypothesis: solver-derived operators exceed matched random masks",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(figures / "M3_operator_specificity_effects.png")
        plt.close(fig)

    heat = merged.pivot_table(
        index="layer",
        columns="operator_real",
        values="log2_real_over_random_influence_lift",
        aggfunc="mean",
    )
    if not heat.empty:
        fig_width = max(8.0, 0.72 * max(1, len(heat.columns)))
        fig, ax = plt.subplots(figsize=(fig_width, 4.8), dpi=180)
        heat_values = torch.as_tensor(heat.to_numpy(dtype=float))
        finite_abs = heat_values[torch.isfinite(heat_values)].abs()
        vmax = max(1.0, float(finite_abs.max())) if finite_abs.numel() else 1.0
        image = ax.imshow(heat.values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.axhline(-0.5, color="none")
        ax.set_title("M3: layer-wise operator specificity")
        ax.set_xlabel("operator")
        ax.set_ylabel("layer")
        ax.set_xticks(range(len(heat.columns)), labels=heat.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(heat.index)), labels=[str(i) for i in heat.index])
        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
            label="log2 real/random influence lift",
        )
        fig.tight_layout()
        fig.savefig(figures / "M3_operator_specificity_by_layer.png")
        plt.close(fig)

    comparison_metrics = ["attention_lift", "abs_transport_lift", "influence_lift"]
    comp = summary[summary["metric"].isin(comparison_metrics)]
    if not comp.empty:
        operators = (
            list(influence["operator"])
            if not influence.empty
            else sorted(comp["operator"].unique())
        )
        labels = {
            "attention_lift": "attention",
            "abs_transport_lift": "realised transport",
            "influence_lift": "task-weighted influence",
        }
        fig, ax = plt.subplots(figsize=(10.2, 5.4), dpi=180)
        width = 0.24
        base = torch.arange(len(operators)).float().numpy()
        for idx, metric in enumerate(comparison_metrics):
            sub = comp[comp["metric"] == metric].set_index("operator").reindex(operators)
            x = base + (idx - 1) * width
            ax.bar(x, sub["mean_log2_real_over_random"], width=width, label=labels[metric])
        ax.axhline(0.0, color="0.25", linewidth=0.8)
        ax.set_xticks(base, labels=operators, rotation=30, ha="right")
        ax.set_ylabel("log2(real / matched random)")
        ax.set_title("M3 control: attention versus realised/influence transport")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures / "M3_attention_vs_transport_specificity.png")
        plt.close(fig)
    plot_graph_count_convergence(merged, figures)


def plot_graph_count_convergence(merged: Any, figures: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped graph-count convergence plot: {exc}", flush=True)
        return
    if "log2_real_over_random_influence_lift" not in merged.columns:
        return
    graph_ids = sorted(merged["graph_index"].unique().tolist())
    if len(graph_ids) < 4:
        return
    final = (
        merged.groupby("operator_real")["log2_real_over_random_influence_lift"]
        .mean()
        .sort_values(ascending=False)
    )
    operators = final.head(5).index.tolist()
    checkpoints = sorted(
        {
            min(len(graph_ids), k)
            for k in [4, 8, 16, 32, 64, 128, len(graph_ids)]
            if k <= len(graph_ids)
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    for operator in operators:
        ys = []
        for count in checkpoints:
            keep = set(graph_ids[:count])
            sub = merged[
                (merged["operator_real"] == operator)
                & (merged["graph_index"].isin(keep))
            ]
            ys.append(float(sub["log2_real_over_random_influence_lift"].mean()))
        ax.plot(checkpoints, ys, marker="o", label=operator)
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("graphs included")
    ax.set_ylabel("mean log2 real/random influence lift")
    ax.set_title("A1: graph-count stability of operator specificity")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "A1_graph_count_convergence.png")
    plt.close(fig)


def plot_main_crux_dashboard(out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[plot] skipped main crux dashboard: {exc}", flush=True)
        return
    knockout_path = out_dir / "mechanism_knockouts.csv"
    specificity_path = out_dir / "operator_specificity_effects.csv"
    ranked_path = out_dir / "metric_ranked_ablation_curves.csv"
    if not (knockout_path.exists() and specificity_path.exists() and ranked_path.exists()):
        return
    knock = pd.read_csv(knockout_path)
    spec = pd.read_csv(specificity_path)
    ranked = pd.read_csv(ranked_path)
    knock = knock[knock["ablation"] != "clean"].copy()
    spec = spec[spec["metric"] == "influence_lift"].sort_values(
        "mean_log2_real_over_random",
        ascending=False,
    ).head(8)

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), dpi=180)

    ax = axes[0]
    x = list(range(len(knock)))
    knock_y_col = (
        "paired_drop_graph_mean"
        if "paired_drop_graph_mean" in knock.columns
        else "raw_drop"
    )
    y = knock[knock_y_col].to_numpy(dtype=float)
    low = knock["raw_drop_ci_low"].to_numpy(dtype=float)
    high = knock["raw_drop_ci_high"].to_numpy(dtype=float)
    yerr = [
        torch.as_tensor(y - low).clamp_min(0.0).numpy(),
        torch.as_tensor(high - y).clamp_min(0.0).numpy(),
    ]
    ax.bar(x, y, color="#4c78a8")
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.15", capsize=3, linewidth=1)
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_xticks(x, labels=knock["ablation"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("paired graph-mean performance drop")
    ax.set_title("M2. Mechanism knockouts")

    ax = axes[1]
    x = list(range(len(spec)))
    y = spec["mean_log2_real_over_random"].to_numpy(dtype=float)
    yerr = [
        y - spec["ci_low"].to_numpy(dtype=float),
        spec["ci_high"].to_numpy(dtype=float) - y,
    ]
    ax.bar(x, y, color="#59a14f")
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="0.15", capsize=3, linewidth=1)
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_xticks(x, labels=spec["operator"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("log2 real / matched random")
    ax.set_title("M3. Operator specificity")

    ax = axes[2]
    for ranking, group in ranked.groupby("ranking"):
        group = group.sort_values("percent_ablated")
        ranked_y_col = (
            "paired_drop_graph_mean"
            if "paired_drop_graph_mean" in group.columns
            else "raw_drop"
        )
        ax.plot(group["percent_ablated"], group[ranked_y_col], marker="o", label=ranking)
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("percent heads ablated")
    ax.set_ylabel("paired graph-mean performance drop")
    ax.set_title("M4. Ranked causal validation")
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Mechanistic crux tests: causal necessity, operator specificity, metric adequacy")
    fig.tight_layout()
    fig.savefig(figures / "M_main_mechanistic_crux_dashboard.png")
    plt.close(fig)


def write_mechanistic_narrative_outputs(out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[plot] skipped narrative scorecard: {exc}", flush=True)
        return

    rows: list[dict[str, Any]] = []
    knockout_path = out_dir / "mechanism_knockouts.csv"
    if knockout_path.exists():
        knock = pd.read_csv(knockout_path)
        if not knock.empty and "ablation" in knock.columns:
            knock = knock[knock["ablation"] != "clean"].copy()
            knock["effect"] = [preferred_drop_value(row) for row in knock.to_dict("records")]
            drops = {
                str(row["ablation"]): finite_or_nan(row["effect"])
                for row in knock.to_dict("records")
            }
            lows = {
                str(row["ablation"]): finite_or_nan(row.get("raw_drop_ci_low"))
                for row in knock.to_dict("records")
            }

            local_drop = drops.get("local_only_support", float("nan"))
            rows.append(
                narrative_claim_row(
                    "complete_support_necessity",
                    "Complete support carries useful beyond-GNN computation.",
                    "mechanism_knockouts.csv",
                    "paired graph-mean drop, local-only ablation",
                    local_drop,
                    supports=math.isfinite(local_drop) and local_drop > 0.0,
                    strong=math.isfinite(lows.get("local_only_support", float("nan")))
                    and lows["local_only_support"] > 0.0,
                )
            )

            routing_drop = drops.get("no_structural_routing", float("nan"))
            transport_drop = drops.get("no_pair_value_transport", float("nan"))
            rows.append(
                narrative_claim_row(
                    "transport_beyond_routing",
                    "Relation-conditioned value transport matters beyond scalar routing.",
                    "mechanism_knockouts.csv",
                    "transport ablation drop minus routing ablation drop",
                    transport_drop - routing_drop,
                    supports=(
                        math.isfinite(transport_drop)
                        and math.isfinite(routing_drop)
                        and transport_drop > routing_drop
                    ),
                    strong=False,
                )
            )

            pair_state_drop = max_finite(
                [
                    drops.get("frozen_pair_state", float("nan")),
                    drops.get("permuted_pair_state", float("nan")),
                ]
            )
            rows.append(
                narrative_claim_row(
                    "pair_state_content_or_evolution",
                    "Dense pair-state content or evolution is functionally used.",
                    "mechanism_knockouts.csv",
                    "max pair-state ablation drop minus routing ablation drop",
                    pair_state_drop - routing_drop,
                    supports=(
                        math.isfinite(pair_state_drop)
                        and math.isfinite(routing_drop)
                        and pair_state_drop > routing_drop
                    ),
                    strong=False,
                )
            )

    specificity_path = out_dir / "operator_specificity_effects.csv"
    if specificity_path.exists():
        spec = pd.read_csv(specificity_path)
        if not spec.empty and "metric" in spec.columns:
            influence = spec[spec["metric"] == "influence_lift"].copy()
            attention = spec[spec["metric"] == "attention_lift"].copy()
            if not influence.empty:
                best = influence.sort_values(
                    "mean_log2_real_over_random",
                    ascending=False,
                ).iloc[0]
                effect = finite_or_nan(best["mean_log2_real_over_random"])
                ci_low = finite_or_nan(best.get("ci_low"))
                rows.append(
                    narrative_claim_row(
                        "operator_specific_transport",
                        "Task-weighted realised transport lands on solver-derived operators.",
                        "operator_specificity_effects.csv",
                        f"best operator log2 real/random influence lift: {best['operator']}",
                        effect,
                        supports=math.isfinite(effect) and effect > 0.0,
                        strong=math.isfinite(ci_low) and ci_low > 0.0,
                    )
                )
            if not influence.empty and not attention.empty:
                influence_mean = finite_or_nan(influence["mean_log2_real_over_random"].mean())
                attention_mean = finite_or_nan(attention["mean_log2_real_over_random"].mean())
                rows.append(
                    narrative_claim_row(
                        "influence_beats_attention_control",
                        "Task-weighted transport is more specific than attention-only alignment.",
                        "operator_specificity_effects.csv",
                        "mean influence specificity minus mean attention specificity",
                        influence_mean - attention_mean,
                        supports=(
                            math.isfinite(influence_mean)
                            and math.isfinite(attention_mean)
                            and influence_mean > attention_mean
                        ),
                        strong=False,
                    )
                )

    auc_path = out_dir / "metric_ranked_ablation_auc.csv"
    if auc_path.exists():
        auc = pd.read_csv(auc_path)
        if not auc.empty and "normalised_auc" in auc.columns:
            if "ranking_family" not in auc.columns:
                auc["ranking_family"] = auc["ranking"].map(
                    lambda ranking: RANKING_FAMILIES.get(str(ranking), "other")
                )
            controls = auc[auc["ranking_family"].isin(["control", "attention_control"])]
            transport = auc[
                auc["ranking_family"].isin(["transport", "task_weighted_transport"])
            ]
            if not controls.empty and not transport.empty:
                best_transport = finite_or_nan(transport["normalised_auc"].max())
                best_control = finite_or_nan(controls["normalised_auc"].max())
                rows.append(
                    narrative_claim_row(
                        "metrics_select_causal_components",
                        "Operator-transport rankings find functional components faster.",
                        "metric_ranked_ablation_auc.csv",
                        "best transport-family AUC minus best control-family AUC",
                        best_transport - best_control,
                        supports=(
                            math.isfinite(best_transport)
                            and math.isfinite(best_control)
                            and best_transport > best_control
                        ),
                        strong=False,
                    )
                )

    if not rows:
        return

    write_csv(out_dir / "mechanistic_narrative_claims.csv", rows)
    write_narrative_markdown(out_dir / "mechanistic_narrative_summary.md", rows)
    plot_narrative_scorecard(rows, out_dir / "figures", plt)


def narrative_claim_row(
    claim_id: str,
    claim: str,
    source: str,
    evidence_metric: str,
    effect_value: float,
    *,
    supports: bool,
    strong: bool,
) -> dict[str, Any]:
    if strong:
        status = "strong_support"
    elif supports:
        status = "directional_support"
    elif math.isfinite(effect_value):
        status = "not_supported"
    else:
        status = "missing"
    return {
        "claim_id": claim_id,
        "claim": claim,
        "source": source,
        "evidence_metric": evidence_metric,
        "effect_value": effect_value,
        "supports_claim": supports,
        "evidence_status": status,
    }


def write_narrative_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mechanistic Narrative Summary",
        "",
        "Generated from mechanistic operator analysis CSV outputs.",
        "",
        "| Claim | Evidence | Effect | Status |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        effect = finite_or_nan(row.get("effect_value"))
        effect_text = "nan" if not math.isfinite(effect) else f"{effect:.4g}"
        lines.append(
            "| "
            + str(row["claim"])
            + " | "
            + str(row["evidence_metric"])
            + " | "
            + effect_text
            + " | "
            + str(row["evidence_status"])
            + " |"
        )
    lines.append("")
    lines.append(
        "Interpretation rule: strong support means the effect is positive with an available "
        "positive lower bootstrap bound; directional support means the sign is aligned but "
        "the scorecard has not established a bootstrap-sign claim."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_narrative_scorecard(rows: Sequence[Mapping[str, Any]], figures: Path, plt: Any) -> None:
    finite_rows = [row for row in rows if math.isfinite(finite_or_nan(row.get("effect_value")))]
    if not finite_rows:
        return
    figures.mkdir(parents=True, exist_ok=True)
    labels = [str(row["claim_id"]).replace("_", "\n") for row in finite_rows]
    values = [finite_or_nan(row["effect_value"]) for row in finite_rows]
    colors = [
        {
            "strong_support": "#4c78a8",
            "directional_support": "#59a14f",
            "not_supported": "#d62728",
            "missing": "#9d9d9d",
        }.get(str(row["evidence_status"]), "#777777")
        for row in finite_rows
    ]
    x = list(range(len(finite_rows)))
    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    ax.bar(x, values, color=colors)
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xticks(x, labels=labels, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("claim-specific effect")
    ax.set_title("Mechanistic narrative scorecard")
    fig.tight_layout()
    fig.savefig(figures / "M_narrative_scorecard.png")
    plt.close(fig)


def analysis_protocol_audit() -> dict[str, Any]:
    return {
        "primary_question": (
            "whether realised, task-weighted head transport concentrates on "
            "task-relevant graph operators beyond their base rate"
        ),
        "dissertation_narrative_axes": {
            "support": "which node pairs may communicate",
            "routing": "which supported senders receive scalar mass",
            "transport": "what node or pair-conditioned value is sent",
            "pair_state": "whether dense pair relations are evolved and used",
        },
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
        "output_tables": {
            "operator_transport_per_head.csv": "layer/head/operator reductions for ranking and OTS",
            "operator_transport_by_layer.csv": "head-pooled layer/operator atlas values",
            "operator_transport_per_graph.csv": (
                "graph-layer rows aggregated over heads for matched-null specificity and stability"
            ),
            "operator_mask_base_rates.csv": (
                "documented operator and matched-random mask base rates"
            ),
            "operator_specificity_effects.csv": (
                "graph-level real-vs-matched-random log2 lift effects with bootstrap CIs"
            ),
            "mechanistic_narrative_claims.csv": (
                "claim-level scorecard aligned to support, routing, transport, and pair-state axes"
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
            (
                "Official GRIT/static-GRIT are implemented first because their "
                "pair transport is explicit."
            ),
            (
                "Flow masks require explicit source/sink roles in node_type and "
                "use deterministic max-flow."
            ),
            (
                "M1 distillation requires supplied teacher kernels; this script "
                "will not fabricate them."
            ),
            "M2 knockouts and M4 ranked ablations are implemented as inference-time GRIT patches.",
            "Pair-set scrubbing and clean/corrupt/patch remain second-phase methods from the memo.",
        ],
    }


@dataclass
class LoadedExperiment:
    runner: Any
    device: torch.device
    runtime_settings: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    cfg: Any
    splits: Mapping[str, Any]
    dataset: Any
    selected_indices: list[int]
    graphs: list[Any]
    subset_dataset: Any
    analysis_batches: list[tuple[list[int], Any]]
    batch_cache_mode: str
    batch_cache_estimated_bytes: int
    model: nn.Module
    target_stats: Optional[Mapping[str, float]]
    pos_weight: Optional[torch.Tensor]


def load_experiment(args: argparse.Namespace) -> LoadedExperiment:
    runner = load_runner(args.runner_path)
    device = resolve_device(args.device)
    runtime_settings = configure_torch_runtime(args, device)
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
        require_present=args.require_pe_cache and not args.build_missing_pe_cache,
        log=print,
    )
    dataset = splits[args.split]
    selected_indices = select_graph_indices(dataset, args.num_graphs, args.graph_seed)
    graphs = [dataset[i] for i in selected_indices]
    if not graphs:
        raise RuntimeError(f"no graphs selected from {args.task}/{args.split}")
    subset_dataset = runner.OfficialGraphDataset(
        graphs,
        task_name=args.task,
        split=f"{args.split}_selected",
    )

    model = runner.build_model(args.model, cfg, backend=args.model_backend)
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint state did not match model exactly: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.to(device).eval()
    analysis_batch_size = max(1, int(args.analysis_cache_batch_size or args.batch_size))
    analysis_batches, batch_cache_mode, batch_cache_estimated_bytes = collate_analysis_batches(
        runner,
        graphs,
        selected_indices,
        batch_size=analysis_batch_size,
        device=device,
        cache_mode=args.cache_batches,
        gpu_cache_limit_gb=args.gpu_cache_limit_gb,
        log=print,
    )
    target_stats = checkpoint.get("target_stats")
    if target_stats is None:
        target_stats = runner.compute_target_stats(splits["train"])
    pos_weight = runner.compute_pos_weight(splits["train"])
    return LoadedExperiment(
        runner=runner,
        device=device,
        runtime_settings=runtime_settings,
        checkpoint=checkpoint,
        cfg=cfg,
        splits=splits,
        dataset=dataset,
        selected_indices=selected_indices,
        graphs=graphs,
        subset_dataset=subset_dataset,
        analysis_batches=analysis_batches,
        batch_cache_mode=batch_cache_mode,
        batch_cache_estimated_bytes=batch_cache_estimated_bytes,
        model=model,
        target_stats=target_stats,
        pos_weight=pos_weight,
    )


def run_atlas(args: argparse.Namespace, loaded: Optional[LoadedExperiment] = None) -> None:
    loaded = loaded or load_experiment(args)
    runner = loaded.runner
    device = loaded.device
    graphs = loaded.graphs
    selected_indices = loaded.selected_indices
    model = loaded.model
    collector = make_collector(model, args.model, args.adapter)
    responsibility = load_task_transport_responsibility(
        args.specialisation_summary_csv,
        intervention=args.responsibility_intervention,
        block=args.responsibility_block,
        centered=args.responsibility_centered,
    )
    accumulator = OperatorTransportAccumulator(responsibility)
    start_time = time.time()
    batch_size = max(1, int(args.analysis_cache_batch_size or args.batch_size))
    teacher_kernels = load_teacher_kernels(args.teacher_kernel_pt)
    if loaded.analysis_batches:
        analysis_items = loaded.analysis_batches
        batches_are_cached = True
    else:
        analysis_items = graph_batches(graphs, selected_indices, batch_size)
        batches_are_cached = False

    for batch_idx, (graph_indices, batch_item) in enumerate(analysis_items):
        cached_batch = batch_item if batches_are_cached else runner.collate_graphs(batch_item)
        batch = batch_to_device(cached_batch, device)
        model.zero_grad(set_to_none=True)
        with collector as active_collector:
            with torch.enable_grad():
                with autocast_context(device, args.autocast_dtype):
                    pred = model(batch)
                scalar = task_scalar(pred, batch)
                scalar.backward()
        valid = valid_pair_mask(batch)
        masks = make_operator_masks(
            args.task,
            batch,
            random_controls=args.random_controls,
            seed=args.random_seed + start_idx,
        )
        teacher_masks = teacher_operator_masks(teacher_kernels, graph_indices, batch)
        masks.update(teacher_masks)
        if args.random_controls and teacher_masks:
            masks.update(
                rate_matched_random_masks(
                    teacher_masks,
                    batch,
                    valid,
                    args.random_seed + start_idx,
                )
            )
        global_mask = masks.get("global_nonedge", (valid & ~sparse_edge_mask(batch)).float())
        accumulator.add_batch(
            active_collector.records,
            masks,
            valid.float(),
            global_mask.float(),
            graph_indices,
        )
        print(
            f"[analysis] batch={batch_idx + 1}/{len(analysis_items)} "
            f"graphs={graph_indices[0]}:{graph_indices[-1] + 1} scalar={to_float(scalar):.5g}",
            flush=True,
        )
        active_collector.records.clear()
        del batch, pred, scalar, masks
        if args.empty_cache_every_batch and device.type == "cuda":
            torch.cuda.empty_cache()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_head = accumulator.per_head_rows()
    by_layer = accumulator.layer_rows()
    head_summary = accumulator.head_summary_rows()
    base_rates = accumulator.base_rate_rows()
    write_csv(out_dir / "operator_transport_per_head.csv", per_head)
    write_csv(out_dir / "operator_transport_by_layer.csv", by_layer)
    write_csv(out_dir / "operator_transport_per_graph.csv", accumulator.graph_rows)
    write_csv(out_dir / "operator_mask_base_rates.csv", base_rates)
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
        "analysis_batch_cache_mode": loaded.batch_cache_mode,
        "analysis_batch_cache_estimated_gib": loaded.batch_cache_estimated_bytes / (1024**3),
        "selected_graph_indices": selected_indices,
        "device": str(device),
        "torch_runtime": dict(loaded.runtime_settings),
        "elapsed_seconds": time.time() - start_time,
        "random_controls": args.random_controls,
        "task_transport_responsibility_source": None
        if args.specialisation_summary_csv is None
        else str(args.specialisation_summary_csv),
        "task_transport_responsibility_rows": len(responsibility),
        "teacher_kernel_pt": None
        if args.teacher_kernel_pt is None
        else str(args.teacher_kernel_pt),
        "teacher_kernels": sorted(teacher_kernels.keys()),
        "operator_mask_base_rate_rows": len(base_rates),
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


def run(args: argparse.Namespace) -> None:
    loaded = load_experiment(args)
    if args.experiments == "all":
        experiments = ("atlas", "knockouts", "ranked_ablation")
    else:
        experiments = parse_csv_list(
            args.experiments,
            default=("atlas",),
            all_values=("atlas", "knockouts", "ranked_ablation", "distillation_calibration"),
        )
    if "distillation_calibration" in experiments:
        run_distillation_calibration(args, loaded)
        experiments = tuple(
            item
            for item in experiments
            if item not in {"distillation_calibration", "atlas", "ranked_ablation"}
        )
    if "atlas" in experiments:
        run_atlas(args, loaded)
    if "knockouts" in experiments:
        run_knockouts(args, loaded)
    if "ranked_ablation" in experiments:
        run_ranked_ablation(args, loaded)
    if "distillation_calibration" in experiments:
        run_distillation_calibration(args, loaded)
    plot_main_crux_dashboard(args.output_dir)
    write_mechanistic_narrative_outputs(args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--experiments",
        default="atlas",
        help="Comma list: atlas,knockouts,ranked_ablation,distillation_calibration,all",
    )
    parser.add_argument("--model", default="grit")
    parser.add_argument("--model-backend", default="official", choices=("official", "local"))
    parser.add_argument("--adapter", default="auto", choices=("auto", "official_grit"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--num-graphs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=0)
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
    parser.add_argument(
        "--require-pe-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the official GraphBench PE cache by default. Use "
            "--no-require-pe-cache only for debugging or with --build-missing-pe-cache."
        ),
    )
    parser.add_argument("--build-missing-pe-cache", action="store_true")
    parser.add_argument("--pe-workers", type=int, default=1)
    parser.add_argument("--pe-save-every", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mechanistic_operator_analysis"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cache-batches",
        default="auto",
        choices=("auto", "gpu", "cpu", "none"),
        help=(
            "Pre-collate selected analysis batches once. auto stores them on GPU when "
            "they fit under --gpu-cache-limit-gb, otherwise on CPU."
        ),
    )
    parser.add_argument(
        "--analysis-cache-batch-size",
        type=int,
        default=0,
        help="Batch size for shared cached analysis batches; 0 uses --batch-size.",
    )
    parser.add_argument(
        "--gpu-cache-limit-gb",
        type=float,
        default=48.0,
        help="Maximum approximate cached batch tensor size to keep resident on GPU in auto mode.",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF32 matmul/convolution on CUDA GPUs. Recommended for A100 analysis runs.",
    )
    parser.add_argument(
        "--float32-matmul-precision",
        default="high",
        choices=("default", "highest", "high", "medium"),
        help="torch.set_float32_matmul_precision setting used on CUDA.",
    )
    parser.add_argument(
        "--autocast-dtype",
        default="none",
        choices=("none", "bfloat16", "float16"),
        help=(
            "Optional CUDA autocast dtype. bfloat16 can speed A100 sweeps, while none "
            "keeps attribution numerics closest to checkpoint dtype."
        ),
    )
    parser.add_argument(
        "--cuda-benchmark",
        action="store_true",
        help="Enable cudnn.benchmark for repeated fixed-shape CUDA workloads.",
    )
    parser.add_argument(
        "--empty-cache-every-batch",
        action="store_true",
        help=(
            "Call torch.cuda.empty_cache() after every atlas batch. Slower; "
            "use only for OOM triage."
        ),
    )
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
        "--knockout-ablations",
        default="all",
        help=(
            "Comma list: local_only_support,global_only_support,no_structural_routing,"
            "no_pair_value_transport,frozen_pair_state,permuted_pair_state,all"
        ),
    )
    parser.add_argument(
        "--baseline-primary",
        type=float,
        default=None,
        help="Optional matched GNN+/baseline primary score for advantage_removed normalisation.",
    )
    parser.add_argument(
        "--atlas-per-head-csv",
        type=Path,
        default=None,
        help="Existing operator_transport_per_head.csv for ranked ablation.",
    )
    parser.add_argument(
        "--rankings",
        default="all",
        help=(
            "Comma list: random,norm,attention_mass,attention_kernel_alignment,"
            "contribution_alignment,influence_lift,global_task_transport,"
            "transport_lift_global_gate,ots,all"
        ),
    )
    parser.add_argument("--topk-percents", default="5,10,20,40")
    parser.add_argument(
        "--teacher-kernel-pt",
        type=Path,
        default=None,
        help="Optional tensor or dict[str,tensor] of known teacher/operator kernels.",
    )
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
