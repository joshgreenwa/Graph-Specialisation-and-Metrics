#!/usr/bin/env python3
"""Compute core symbolic/structural specialisation metrics for graph models.

The metric engine is model-agnostic: it consumes, for each layer/head, a routing
field ``A[i, j]`` and a transported message field ``m[i, j]`` from an adapter.
The CLI included here wires that engine to the GraphBench AlgoReas HPC runner and
the official GRIT/static-GRIT wrappers used in this repository.

Implemented metrics follow Section 3 of the dissertation draft:

1. routing scores over attention rows;
2. transport scores over per-key message fields;
3. realised-output routing/transport responsibility and sensitivity.

For metrics 1 and 2, permutations are weighted with the field-matched clean
moved-mass softmax. Centered variants subtract the masked key-wise mean before
cosine scoring. Content interventions swap symbolic node features while holding
structure fixed. Structural interventions swap topology-derived structural
inputs while holding symbolic node features fixed.
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
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn as nn


EPS = 1.0e-12
DEFAULT_METRICS = ("routing", "transport", "output")
DEFAULT_INTERVENTIONS = ("content", "structure")
DEFAULT_BLOCKS = ("all",)


@dataclass
class LayerFields:
    """Per-layer/head fields needed by the metric engine.

    Shapes:
      attention: [B, H, N, N], query-by-key post-softmax routing mass.
      message:   [B, H, N, N, D], pre-attention message transported from key to query.
      mask:      [B, H, N, N], valid query-key support for the fields.
      node_mask: [B, N], real-node mask.
    """

    layer: int
    attention: torch.Tensor
    message: torch.Tensor
    mask: torch.Tensor
    node_mask: torch.Tensor


@dataclass
class MetricOptions:
    metrics: tuple[str, ...] = DEFAULT_METRICS
    interventions: tuple[str, ...] = DEFAULT_INTERVENTIONS
    blocks: tuple[str, ...] = DEFAULT_BLOCKS
    centered: tuple[bool, ...] = (False, True)
    num_permutations: int = 96
    alpha_tau: float = 0.1
    seed: int = 0


@dataclass
class ResultBundle:
    per_graph_rows: list[dict[str, Any]]
    query_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]


def parse_csv_set(
    text: str,
    allowed: Iterable[str],
    *,
    all_values: Sequence[str],
) -> tuple[str, ...]:
    allowed_set = set(allowed)
    out: list[str] = []
    for item in text.split(","):
        value = item.strip().lower()
        if not value:
            continue
        if value == "all":
            out.extend(all_values)
            continue
        if value not in allowed_set:
            allowed_values = sorted(allowed_set | {"all"})
            raise ValueError(f"unsupported value {value!r}; allowed={allowed_values}")
        out.append(value)
    deduped = tuple(dict.fromkeys(out))
    if not deduped:
        raise ValueError("empty option set")
    return deduped


def parse_centered(text: str) -> tuple[bool, ...]:
    value = text.strip().lower()
    if value in {"both", "all"}:
        return (False, True)
    if value in {"uncentered", "false", "0", "no"}:
        return (False,)
    if value in {"centered", "true", "1", "yes"}:
        return (True,)
    raise ValueError("--centered must be one of both, centered, uncentered")


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
    candidates = []
    cwd = Path.cwd()
    candidates.append(cwd / "graphbench-algoreas-hpc" / "bin" / "algoreas_hpc.py")
    candidates.append(cwd / "experiments" / "graphbench" / "hpc" / "bin" / "algoreas_hpc.py")
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "graphbench-algoreas-hpc" / "bin" / "algoreas_hpc.py")
        candidates.append(parent / "experiments" / "graphbench" / "hpc" / "bin" / "algoreas_hpc.py")
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


def masked_mean(values: torch.Tensor, valid: torch.Tensor, dim: int) -> torch.Tensor:
    weights = valid.to(values.dtype)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def gather_node_axis(tensor: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    if tensor.dim() < 2:
        raise ValueError(f"expected [B,N,...] tensor, got {tuple(tensor.shape)}")
    bsz, n = tensor.shape[:2]
    idx = perm_pos.to(device=tensor.device)
    view = (bsz, n) + (1,) * (tensor.dim() - 2)
    return torch.gather(tensor, dim=1, index=idx.view(view).expand_as(tensor))


def gather_pair_axes(tensor: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    if tensor.dim() < 3:
        raise ValueError(f"expected [B,N,N,...] tensor, got {tuple(tensor.shape)}")
    bsz, n = tensor.shape[:2]
    idx = perm_pos.to(device=tensor.device)
    extra = (1,) * (tensor.dim() - 3)
    row_idx = idx.view(bsz, n, 1, *extra).expand_as(tensor)
    rows = torch.gather(tensor, dim=1, index=row_idx)
    col_idx = idx.view(bsz, 1, n, *extra).expand_as(rows)
    return torch.gather(rows, dim=2, index=col_idx)


def key_swap_field(field: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    """Apply the key-side reference transform ``Q_i -> Q_i P_pi``."""

    if field.dim() < 4:
        raise ValueError(f"expected [B,H,N,N,...] field, got {tuple(field.shape)}")
    bsz, heads, n = field.shape[:3]
    idx = perm_pos.to(device=field.device)
    extra = (1,) * (field.dim() - 4)
    key_idx = idx.view(bsz, 1, 1, n, *extra).expand_as(field)
    return torch.gather(field, dim=3, index=key_idx)


def inverse_perm_pos(perm_pos: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm_pos)
    arange = torch.arange(perm_pos.size(1), dtype=perm_pos.dtype, device=perm_pos.device)
    inv.scatter_(1, perm_pos, arange[None, :].expand_as(perm_pos))
    return inv


def clone_batch_with(batch: Any, **kwargs: Any) -> Any:
    return dataclasses.replace(batch, **kwargs)


def make_content_swapped_batch(batch: Any, perm_pos: torch.Tensor) -> Any:
    updates: dict[str, torch.Tensor] = {}
    if hasattr(batch, "node_type"):
        updates["node_type"] = gather_node_axis(batch.node_type, perm_pos)
    for name in ("x", "payload", "anchor_indicator", "anchor_priority"):
        if hasattr(batch, name):
            value = getattr(batch, name)
            if isinstance(value, torch.Tensor):
                updates[name] = gather_node_axis(value, perm_pos)
    if not updates:
        raise TypeError("content swap requires at least one node-level content tensor")
    return clone_batch_with(batch, **updates)


def permute_sparse_edges(batch: Any, perm_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv = inverse_perm_pos(perm_pos).to(device=batch.edge_src.device)
    new_src = batch.edge_src.clone()
    new_dst = batch.edge_dst.clone()
    for graph_idx in range(batch.num_graphs):
        edge_mask = batch.edge_batch == graph_idx
        if bool(edge_mask.any()):
            new_src[edge_mask] = inv[graph_idx, batch.edge_src[edge_mask]]
            new_dst[edge_mask] = inv[graph_idx, batch.edge_dst[edge_mask]]
    return new_src, new_dst


def make_structure_swapped_batch(batch: Any, perm_pos: torch.Tensor) -> Any:
    new_src, new_dst = permute_sparse_edges(batch, perm_pos)
    return clone_batch_with(
        batch,
        adj=gather_pair_axes(batch.adj, perm_pos),
        edge_value_mat=gather_pair_axes(batch.edge_value_mat, perm_pos),
        degree=gather_node_axis(batch.degree, perm_pos),
        spd=gather_pair_axes(batch.spd, perm_pos),
        rwse=gather_node_axis(batch.rwse, perm_pos),
        rrwp=gather_pair_axes(batch.rrwp, perm_pos),
        pair_xi=gather_pair_axes(batch.pair_xi, perm_pos),
        edge_src=new_src,
        edge_dst=new_dst,
    )


def sample_transposition_permutation(
    node_mask: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one within-graph node transposition for every graph in the batch."""

    bsz, n = node_mask.shape
    perm_pos = torch.arange(n, dtype=torch.long).repeat(bsz, 1)
    swap_u = torch.zeros(bsz, dtype=torch.long)
    swap_v = torch.zeros(bsz, dtype=torch.long)
    counts = node_mask.long().sum(dim=1).tolist()
    for b, count_raw in enumerate(counts):
        count = int(count_raw)
        if count < 2:
            continue
        pair = torch.randperm(count, generator=generator)[:2]
        u = int(pair[0])
        v = int(pair[1])
        perm_pos[b, u] = v
        perm_pos[b, v] = u
        swap_u[b] = u
        swap_v[b] = v
    return perm_pos, swap_u, swap_v


def block_key_membership(batch: Any, block: str) -> torch.Tensor:
    """Return [B,N,N] mask saying whether key j is in query i's scoring block."""

    node = batch.node_mask.bool()
    bsz, n = node.shape
    real_pair = node[:, :, None] & node[:, None, :]
    eye = torch.eye(n, dtype=torch.bool, device=node.device).unsqueeze(0)
    if block == "all":
        return real_pair
    if block == "local":
        return real_pair & (batch.adj > 0) & ~eye
    if block == "global":
        return real_pair & ~(batch.adj > 0) & ~eye
    raise ValueError(f"unsupported block {block!r}")


def transposition_valid_for_block(
    block_membership: torch.Tensor,
    swap_u: torch.Tensor,
    swap_v: torch.Tensor,
) -> torch.Tensor:
    """Return [B,1,N] query-valid mask for a sampled graph-level swap."""

    bsz, n, _ = block_membership.shape
    u = swap_u.to(device=block_membership.device)
    v = swap_v.to(device=block_membership.device)
    graph_idx = torch.arange(bsz, device=block_membership.device)
    u_ok = block_membership[graph_idx, :, u]
    v_ok = block_membership[graph_idx, :, v]
    return (u_ok & v_ok).unsqueeze(1)


def expand_mask(mask: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    out = mask.to(device=field.device, dtype=torch.bool)
    if out.size(1) == 1 and field.size(1) != 1:
        out = out.expand(-1, field.size(1), -1, -1)
    return out


def center_field(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = expand_mask(mask, field)
    while m.dim() < field.dim():
        m = m.unsqueeze(-1)
    clean = torch.where(m, torch.nan_to_num(field.float()), torch.zeros_like(field.float()))
    denom = m.sum(dim=3, keepdim=True).clamp_min(1).to(clean.dtype)
    mean = clean.sum(dim=3, keepdim=True) / denom
    return torch.where(m, clean - mean, torch.zeros_like(clean))


def cosine_by_query(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    mask: torch.Tensor,
    *,
    centered: bool,
) -> torch.Tensor:
    m = expand_mask(mask, lhs)
    if centered:
        lhs0 = center_field(lhs, m)
        rhs0 = center_field(rhs, m)
    else:
        me = m
        while me.dim() < lhs.dim():
            me = me.unsqueeze(-1)
        lhs0 = torch.where(me, torch.nan_to_num(lhs.float()), torch.zeros_like(lhs.float()))
        rhs0 = torch.where(me, torch.nan_to_num(rhs.float()), torch.zeros_like(rhs.float()))
    reduce_dims = tuple(range(3, lhs0.dim()))
    num = (lhs0 * rhs0).sum(dim=reduce_dims)
    lhs_norm = torch.sqrt((lhs0 * lhs0).sum(dim=reduce_dims).clamp_min(EPS))
    rhs_norm = torch.sqrt((rhs0 * rhs0).sum(dim=reduce_dims).clamp_min(EPS))
    out = num / (lhs_norm * rhs_norm).clamp_min(EPS)
    if centered:
        return torch.clamp(out, -1.0, 1.0)
    return torch.clamp(out, 0.0, 1.0)


def moved_mass_by_query(
    clean: torch.Tensor,
    clean_mask: torch.Tensor,
    perm_pos: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    field: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ref = key_swap_field(clean, perm_pos)
    ref_mask = key_swap_field(expand_mask(clean_mask, clean).long(), perm_pos).bool()
    base_mask = expand_mask(clean_mask, clean)
    block = block_mask[:, None, :, :].to(device=clean.device)
    if block.size(1) == 1 and clean.size(1) != 1:
        block = block.expand(-1, clean.size(1), -1, -1)
    support = (base_mask | ref_mask) & block
    support_e = support
    while support_e.dim() < clean.dim():
        support_e = support_e.unsqueeze(-1)
    lhs = torch.where(support_e, clean.float(), torch.zeros_like(clean.float()))
    rhs = torch.where(support_e, ref.float(), torch.zeros_like(ref.float()))
    diff = lhs - rhs
    if field == "routing":
        moved = 0.5 * diff.abs().sum(dim=3)
    elif field == "transport":
        moved = 0.5 * torch.sqrt((diff * diff).sum(dim=tuple(range(3, diff.dim()))).clamp_min(0.0))
    else:
        raise ValueError(field)
    valid = support.any(dim=3)
    return moved, valid


def masked_softmax_permutation_weights(
    moved: torch.Tensor,
    valid: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("alpha temperature must be positive")
    logits = moved / float(tau)
    logits = logits.masked_fill(~valid, -torch.inf)
    any_valid = valid.any(dim=0, keepdim=True)
    logits = torch.where(any_valid, logits, torch.zeros_like(logits))
    alpha = torch.softmax(logits, dim=0)
    alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
    normaliser = alpha.sum(dim=0, keepdim=True).clamp_min(EPS)
    return alpha / normaliser


def masked_key_field(field: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = expand_mask(mask, field)
    while m.dim() < field.dim():
        m = m.unsqueeze(-1)
    return torch.where(m, field.float(), torch.zeros_like(field.float()))


def output_from_fields(attn: torch.Tensor, msg: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    a = masked_key_field(attn, mask)
    m = masked_key_field(msg, mask)
    return (a.unsqueeze(-1) * m).sum(dim=3)


def normalized_attention_entropy(
    attn: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = expand_mask(mask, attn)
    probs = masked_key_field(attn, m)
    row_mass = probs.sum(dim=3, keepdim=True).clamp_min(EPS)
    probs = probs / row_mass
    log_probs = torch.where(probs > 0, torch.log(probs.clamp_min(EPS)), torch.zeros_like(probs))
    entropy = -(probs * log_probs).sum(dim=3)
    support = m.sum(dim=3).to(entropy.dtype)
    normalizer = torch.log(support.clamp_min(2.0))
    normalized = entropy / normalizer.clamp_min(EPS)
    valid = support > 1
    return normalized, valid


class OnlineAlphaAccumulator:
    """Online softmax-weighted accumulator over sampled permutations.

    For each graph/head/query cell, this maintains a numerically stable
    log-sum-exp normalizer and weighted score sums. Memory is independent of the
    number of sampled permutations.
    """

    def __init__(self, centered_options: Sequence[bool]) -> None:
        self.centered_options = tuple(centered_options)
        self.initialized = False
        self.max_logit: torch.Tensor
        self.weight_sum: torch.Tensor
        self.weight_sq_sum: torch.Tensor
        self.valid_any: torch.Tensor
        self.moved_sum: torch.Tensor
        self.moved_count: torch.Tensor
        self.invariant_sum: dict[bool, torch.Tensor] = {}
        self.follow_sum: dict[bool, torch.Tensor] = {}

    def _init_like(self, template: torch.Tensor) -> None:
        self.max_logit = torch.full_like(template, -torch.inf)
        self.weight_sum = torch.zeros_like(template)
        self.weight_sq_sum = torch.zeros_like(template)
        self.valid_any = torch.zeros_like(template, dtype=torch.bool)
        self.moved_sum = torch.zeros_like(template)
        self.moved_count = torch.zeros_like(template)
        self.invariant_sum = {
            centered: torch.zeros_like(template) for centered in self.centered_options
        }
        self.follow_sum = {
            centered: torch.zeros_like(template) for centered in self.centered_options
        }
        self.initialized = True

    def update(
        self,
        moved: torch.Tensor,
        valid_alpha: torch.Tensor,
        valid_score: torch.Tensor,
        invariant_scores: Mapping[bool, torch.Tensor],
        follow_scores: Mapping[bool, torch.Tensor],
        alpha_tau: float,
    ) -> None:
        moved = moved.detach()
        valid = (valid_alpha & valid_score).detach()
        if not self.initialized:
            self._init_like(moved.float())
        logits = moved.float() / float(alpha_tau)
        old_max = self.max_logit
        candidate_max = torch.maximum(old_max, logits)
        new_max = torch.where(valid, candidate_max, old_max)
        old_scale = torch.where(
            torch.isfinite(old_max),
            torch.exp(old_max - new_max),
            torch.zeros_like(old_max),
        )
        new_scale = torch.where(valid, torch.exp(logits - new_max), torch.zeros_like(logits))
        self.max_logit = new_max
        self.weight_sum = self.weight_sum * old_scale + new_scale
        self.weight_sq_sum = self.weight_sq_sum * torch.square(old_scale) + torch.square(new_scale)
        self.valid_any |= valid
        self.moved_sum += torch.where(valid, moved.float(), torch.zeros_like(moved.float()))
        self.moved_count += valid.to(self.moved_count.dtype)
        for centered in self.centered_options:
            self.invariant_sum[centered] = (
                self.invariant_sum[centered] * old_scale
                + new_scale * invariant_scores[centered].detach().float()
            )
            self.follow_sum[centered] = (
                self.follow_sum[centered] * old_scale
                + new_scale * follow_scores[centered].detach().float()
            )

    def finalize(self) -> dict[str, Any]:
        if not self.initialized:
            raise RuntimeError("cannot finalize an empty OnlineAlphaAccumulator")
        denom = self.weight_sum.clamp_min(EPS)
        valid = self.valid_any
        moved_mean = self.moved_sum / self.moved_count.clamp_min(1.0)
        alpha_max = torch.where(valid, 1.0 / denom, torch.zeros_like(denom))
        effective_perms = torch.where(
            valid,
            torch.square(self.weight_sum) / self.weight_sq_sum.clamp_min(EPS),
            torch.zeros_like(denom),
        )
        return {
            "valid_any": valid,
            "moved_mean": moved_mean,
            "alpha_max": alpha_max,
            "effective_perms": effective_perms,
            "invariant": {
                centered: self.invariant_sum[centered] / denom
                for centered in self.centered_options
            },
            "follow": {
                centered: self.follow_sum[centered] / denom
                for centered in self.centered_options
            },
        }


class MetricAccumulator:
    def __init__(self, *, write_query_scores: bool = False) -> None:
        self.write_query_scores = write_query_scores
        self.per_graph_rows: list[dict[str, Any]] = []
        self.query_rows: list[dict[str, Any]] = []
        self._summary_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        self._summary_within: dict[tuple[Any, ...], list[float]] = defaultdict(list)

    def add_query_scores(
        self,
        scores: torch.Tensor,
        valid: torch.Tensor,
        *,
        graph_indices: Sequence[int],
        layer: int,
        metric: str,
        field: str,
        intervention: str,
        block: str,
        centered: Optional[bool],
        alpha_tau: Optional[float],
        extra: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> None:
        scores = scores.detach().float().cpu()
        valid = valid.detach().bool().cpu()
        bsz, heads, n = scores.shape
        key = (layer, metric, field, intervention, block, centered)
        extra = extra or {}
        extra_cpu = {name: value.detach().float().cpu() for name, value in extra.items()}
        for b in range(bsz):
            graph_index = int(graph_indices[b])
            for h in range(heads):
                q_valid = valid[b, h]
                if bool(q_valid.any()):
                    q_scores = scores[b, h, q_valid]
                    graph_score = float(q_scores.mean())
                    within_var = (
                        float(q_scores.var(unbiased=False)) if q_scores.numel() > 1 else 0.0
                    )
                else:
                    graph_score = float("nan")
                    within_var = float("nan")
                row: dict[str, Any] = {
                    "graph_index": graph_index,
                    "layer": int(layer),
                    "head": int(h),
                    "metric": metric,
                    "field": field,
                    "intervention": intervention,
                    "block": block,
                    "centered": "" if centered is None else bool(centered),
                    "alpha_tau": "" if alpha_tau is None else float(alpha_tau),
                    "score": graph_score,
                    "within_graph_query_variance": within_var,
                    "valid_queries": int(q_valid.sum()),
                }
                for name, tensor in extra_cpu.items():
                    row[name] = float(tensor[b, h]) if tensor.dim() == 2 else float("nan")
                self.per_graph_rows.append(row)
                if math.isfinite(graph_score):
                    self._summary_values[key + (h,)].append(graph_score)
                if math.isfinite(within_var):
                    self._summary_within[key + (h,)].append(within_var)
                if self.write_query_scores:
                    for q in range(n):
                        if not bool(q_valid[q]):
                            continue
                        self.query_rows.append(
                            {
                                "graph_index": graph_index,
                                "query": q,
                                "layer": int(layer),
                                "head": int(h),
                                "metric": metric,
                                "field": field,
                                "intervention": intervention,
                                "block": block,
                                "centered": "" if centered is None else bool(centered),
                                "alpha_tau": "" if alpha_tau is None else float(alpha_tau),
                                "score": float(scores[b, h, q]),
                            }
                        )

    def summaries(self) -> list[dict[str, Any]]:
        rows = []
        sort_key = lambda item: tuple(str(x) for x in item[0])
        for key, values in sorted(self._summary_values.items(), key=sort_key):
            layer, metric, field, intervention, block, centered, head = key
            if not values:
                continue
            mean = sum(values) / len(values)
            between_var = (
                sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                if len(values) > 1
                else 0.0
            )
            within_values = self._summary_within.get(key, [])
            within_mean = sum(within_values) / len(within_values) if within_values else float("nan")
            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "metric": metric,
                    "field": field,
                    "intervention": intervention,
                    "block": block,
                    "centered": "" if centered is None else bool(centered),
                    "graphs": len(values),
                    "mean": mean,
                    "std_between_graphs": math.sqrt(max(0.0, between_var)),
                    "between_graph_variance": between_var,
                    "within_graph_query_variance_mean": within_mean,
                    "total_variance_estimate": between_var + within_mean
                    if math.isfinite(within_mean)
                    else float("nan"),
                }
            )
        return rows


class OfficialGRITFieldCollector:
    """Adapter for the official GRIT/static-GRIT wrappers in ``algoreas_hpc.py``."""

    def __init__(self, model: nn.Module, max_nodes: int) -> None:
        if not hasattr(model, "layers"):
            raise TypeError("official GRIT collector expects model.layers")
        self.model = model
        self.max_nodes = int(max_nodes)

    @torch.no_grad()
    def collect(self, batch: Any) -> list[LayerFields]:
        self.model.eval()
        records: list[LayerFields] = []
        handles = []

        def make_hook(layer_idx: int):
            def hook(module: nn.Module, inputs: tuple[Any, ...], _outputs: Any) -> None:
                pyg_batch = inputs[0]
                records.append(self._densify(layer_idx, module, pyg_batch, batch.node_mask))

            return hook

        for layer_idx, layer in enumerate(self.model.layers):
            handles.append(layer.attention.register_forward_hook(make_hook(layer_idx)))
        try:
            _ = self.model(batch)
        finally:
            for handle in handles:
                handle.remove()
        records.sort(key=lambda item: item.layer)
        if not records:
            raise RuntimeError(
                "no attention/message fields were collected from official GRIT model"
            )
        return records

    def _densify(
        self,
        layer_idx: int,
        attention_module: nn.Module,
        pyg_batch: Any,
        node_mask: torch.Tensor,
    ) -> LayerFields:
        edge_index = pyg_batch.edge_index.long()
        src = edge_index[0]
        dst = edge_index[1]
        attn_e = pyg_batch.attn.squeeze(-1).float()
        heads = int(attn_e.size(1))
        msg_e = pyg_batch.V_h[src].float()
        has_edge_values = getattr(pyg_batch, "wE", None) is not None
        if getattr(attention_module, "edge_enhance", False) and has_edge_values:
            edge_state = pyg_batch.wE.view(-1, heads, msg_e.size(-1)).float()
            pair_msg = torch.einsum("ehd,dhc->ehc", edge_state, attention_module.VeRow.float())
            msg_e = msg_e + pair_msg

        counts = [int(value) for value in pyg_batch.graph_num_nodes.detach().cpu().tolist()]
        bsz = len(counts)
        nmax = self.max_nodes
        dim = int(msg_e.size(-1))
        device = attn_e.device
        attention = torch.zeros(bsz, heads, nmax, nmax, dtype=torch.float32, device=device)
        message = torch.zeros(bsz, heads, nmax, nmax, dim, dtype=torch.float32, device=device)
        mask = torch.zeros(bsz, heads, nmax, nmax, dtype=torch.bool, device=device)

        offset = 0
        for graph_idx, count in enumerate(counts):
            in_graph = (
                (src >= offset)
                & (src < offset + count)
                & (dst >= offset)
                & (dst < offset + count)
            )
            if bool(in_graph.any()):
                local_src = src[in_graph] - offset
                local_dst = dst[in_graph] - offset
                attention[graph_idx, :, local_dst, local_src] = attn_e[in_graph].transpose(0, 1)
                message[graph_idx, :, local_dst, local_src] = msg_e[in_graph].permute(1, 0, 2)
                mask[graph_idx, :, local_dst, local_src] = True
            offset += count
        return LayerFields(
            layer=layer_idx,
            attention=attention,
            message=message,
            mask=mask,
            node_mask=node_mask.to(device=device, dtype=torch.bool),
        )


class SpecialisationMetricEngine:
    def __init__(
        self,
        collector: OfficialGRITFieldCollector,
        options: MetricOptions,
        *,
        write_query_scores: bool = False,
    ) -> None:
        self.collector = collector
        self.options = options
        self.acc = MetricAccumulator(write_query_scores=write_query_scores)
        self.generator = torch.Generator(device="cpu").manual_seed(int(options.seed))

    @torch.no_grad()
    def compute_batch(self, batch: Any, *, graph_indices: Sequence[int]) -> None:
        clean_layers = self.collector.collect(batch)
        block_membership = {
            block: block_key_membership(batch, block) for block in self.options.blocks
        }
        field_cache: dict[tuple[int, str, str, str], OnlineAlphaAccumulator] = {}
        output_cache: dict[tuple[int, str, str], dict[str, torch.Tensor]] = {}

        for _perm_idx in range(self.options.num_permutations):
            perm_cpu, swap_u, swap_v = sample_transposition_permutation(
                batch.node_mask.detach().cpu(),
                self.generator,
            )
            perm_pos = perm_cpu.to(device=batch.node_mask.device)
            if "content" in self.options.interventions:
                variant = make_content_swapped_batch(batch, perm_pos)
                variant_layers = self.collector.collect(variant)
                for clean, variant_layer in zip(clean_layers, variant_layers):
                    for block, block_mask in block_membership.items():
                        if "routing" in self.options.metrics:
                            self._append_field_cache(
                                field_cache,
                                clean,
                                variant_layer,
                                perm_pos,
                                swap_u,
                                swap_v,
                                block,
                                block_mask,
                                intervention="content",
                                field_name="routing",
                                field_getter=lambda layer: layer.attention,
                            )
                        if "transport" in self.options.metrics:
                            self._append_field_cache(
                                field_cache,
                                clean,
                                variant_layer,
                                perm_pos,
                                swap_u,
                                swap_v,
                                block,
                                block_mask,
                                intervention="content",
                                field_name="transport",
                                field_getter=lambda layer: layer.message,
                            )
                        if "output" in self.options.metrics:
                            self._append_output_cache(
                                output_cache,
                                clean,
                                variant_layer,
                                perm_pos,
                                swap_u,
                                swap_v,
                                block,
                                block_mask,
                                intervention="content",
                            )
                del variant_layers
            if "structure" in self.options.interventions:
                variant = make_structure_swapped_batch(batch, perm_pos)
                variant_layers = self.collector.collect(variant)
                for clean, variant_layer in zip(clean_layers, variant_layers):
                    for block, block_mask in block_membership.items():
                        if "routing" in self.options.metrics:
                            self._append_field_cache(
                                field_cache,
                                clean,
                                variant_layer,
                                perm_pos,
                                swap_u,
                                swap_v,
                                block,
                                block_mask,
                                intervention="structure",
                                field_name="routing",
                                field_getter=lambda layer: layer.attention,
                            )
                        if "transport" in self.options.metrics:
                            self._append_field_cache(
                                field_cache,
                                clean,
                                variant_layer,
                                perm_pos,
                                swap_u,
                                swap_v,
                                block,
                                block_mask,
                                intervention="structure",
                                field_name="transport",
                                field_getter=lambda layer: layer.message,
                            )
                        if "output" in self.options.metrics:
                            self._append_output_cache(
                                output_cache,
                                clean,
                                variant_layer,
                                perm_pos,
                                swap_u,
                                swap_v,
                                block,
                                block_mask,
                                intervention="structure",
                            )
                del variant_layers

        for clean in clean_layers:
            self._add_attention_entropy(clean, graph_indices)
            for block, block_mask in block_membership.items():
                self._add_gates(clean, batch, block, block_mask, graph_indices)
        self._finalize_field_cache(field_cache, graph_indices)
        self._finalize_output_cache(output_cache, graph_indices)

    def _append_field_cache(
        self,
        cache: dict[tuple[int, str, str, str], OnlineAlphaAccumulator],
        clean: LayerFields,
        variant: LayerFields,
        perm_pos: torch.Tensor,
        swap_u: torch.Tensor,
        swap_v: torch.Tensor,
        block: str,
        block_mask: torch.Tensor,
        *,
        intervention: str,
        field_name: str,
        field_getter,
    ) -> None:
        clean_field = field_getter(clean)
        variant_field = field_getter(variant)
        ref = key_swap_field(clean_field, perm_pos)
        clean_mask = expand_mask(clean.mask, clean_field)
        variant_mask = expand_mask(variant.mask, variant_field)
        ref_mask = key_swap_field(clean_mask.long(), perm_pos).bool()
        block_m = block_mask[:, None, :, :].to(device=clean_field.device)
        if block_m.size(1) == 1 and clean_field.size(1) != 1:
            block_m = block_m.expand(-1, clean_field.size(1), -1, -1)
        query_valid = transposition_valid_for_block(block_mask, swap_u, swap_v).to(
            device=clean_field.device
        )
        if query_valid.size(1) == 1 and clean_field.size(1) != 1:
            query_valid = query_valid.expand(-1, clean_field.size(1), -1)
        stable_mask = clean_mask & variant_mask & block_m
        follow_mask = ref_mask & variant_mask & block_m
        valid_score = (stable_mask.any(dim=3) | follow_mask.any(dim=3)) & query_valid
        moved, valid_alpha = moved_mass_by_query(
            clean_field,
            clean.mask,
            perm_pos,
            block_mask,
            field=field_name,
        )
        item = cache.setdefault(
            (clean.layer, block, intervention, field_name),
            OnlineAlphaAccumulator(self.options.centered),
        )
        invariant_scores = {}
        follow_scores = {}
        for centered in self.options.centered:
            invariant_scores[centered] = cosine_by_query(
                variant_field,
                clean_field,
                stable_mask,
                centered=centered,
            )
            follow_scores[centered] = cosine_by_query(
                variant_field,
                ref,
                follow_mask,
                centered=centered,
            )
        item.update(
            moved,
            valid_alpha & query_valid,
            valid_score,
            invariant_scores,
            follow_scores,
            self.options.alpha_tau,
        )

    def _finalize_field_cache(
        self,
        cache: Mapping[tuple[int, str, str, str], OnlineAlphaAccumulator],
        graph_indices: Sequence[int],
    ) -> None:
        for (layer, block, intervention, field_name), item in cache.items():
            final = item.finalize()
            valid_any = final["valid_any"]
            extra = {
                "moved_mass_mean": masked_mean(final["moved_mean"], valid_any, dim=2),
                "alpha_max_mean": masked_mean(final["alpha_max"], valid_any, dim=2),
                "effective_perms_mean": masked_mean(final["effective_perms"], valid_any, dim=2),
            }
            for centered in self.options.centered:
                self.acc.add_query_scores(
                    final["invariant"][centered],
                    valid_any,
                    graph_indices=graph_indices,
                    layer=layer,
                    metric=f"{field_name}_invariant",
                    field=field_name,
                    intervention=intervention,
                    block=block,
                    centered=centered,
                    alpha_tau=self.options.alpha_tau,
                    extra=extra,
                )
                self.acc.add_query_scores(
                    final["follow"][centered],
                    valid_any,
                    graph_indices=graph_indices,
                    layer=layer,
                    metric=f"{field_name}_follow",
                    field=field_name,
                    intervention=intervention,
                    block=block,
                    centered=centered,
                    alpha_tau=self.options.alpha_tau,
                    extra=extra,
                )

    def _append_output_cache(
        self,
        cache: dict[tuple[int, str, str], dict[str, torch.Tensor]],
        clean: LayerFields,
        variant: LayerFields,
        perm_pos: torch.Tensor,
        swap_u: torch.Tensor,
        swap_v: torch.Tensor,
        block: str,
        block_mask: torch.Tensor,
        *,
        intervention: str,
    ) -> None:
        clean_mask = expand_mask(clean.mask, clean.attention)
        block_m = block_mask[:, None, :, :].to(device=clean.attention.device)
        if block_m.size(1) == 1 and clean.attention.size(1) != 1:
            block_m = block_m.expand(-1, clean.attention.size(1), -1, -1)
        clean_block_mask = clean_mask & block_m
        query_valid = transposition_valid_for_block(block_mask, swap_u, swap_v).to(
            device=clean.attention.device
        )
        if query_valid.size(1) == 1 and clean.attention.size(1) != 1:
            query_valid = query_valid.expand(-1, clean.attention.size(1), -1)
        var_mask = expand_mask(variant.mask, variant.attention) & block_m
        score_mask = clean_block_mask & var_mask
        valid = score_mask.any(dim=3) & query_valid
        a0 = masked_key_field(clean.attention, score_mask)
        a1 = masked_key_field(variant.attention, score_mask)
        m0 = masked_key_field(clean.message, score_mask)
        m1 = masked_key_field(variant.message, score_mask)
        a_bar = 0.5 * (a0 + a1)
        m_bar = 0.5 * (m0 + m1)
        delta_a = ((a1 - a0).unsqueeze(-1) * m_bar).sum(dim=3)
        delta_m = (a_bar.unsqueeze(-1) * (m1 - m0)).sum(dim=3)
        delta = output_from_fields(
            variant.attention,
            variant.message,
            score_mask,
        ) - output_from_fields(clean.attention, clean.message, score_mask)
        da_norm = torch.sqrt((delta_a * delta_a).sum(dim=-1).clamp_min(0.0))
        dm_norm = torch.sqrt((delta_m * delta_m).sum(dim=-1).clamp_min(0.0))
        delta_norm = torch.sqrt((delta * delta).sum(dim=-1).clamp_min(0.0))
        key = (clean.layer, block, intervention)
        if key not in cache:
            o_clean = output_from_fields(clean.attention, clean.message, clean_block_mask)
            base = torch.zeros_like(o_clean[..., 0])
            cache[key] = {
                "da_sum": base.clone(),
                "dm_sum": base.clone(),
                "delta_sum": base.clone(),
                "count": base.clone(),
                "valid_any": torch.zeros_like(base, dtype=torch.bool),
                "output_norm": torch.sqrt((o_clean * o_clean).sum(dim=-1).clamp_min(EPS)),
            }
        item = cache[key]
        item["da_sum"] += torch.where(valid, da_norm, torch.zeros_like(da_norm)).detach()
        item["dm_sum"] += torch.where(valid, dm_norm, torch.zeros_like(dm_norm)).detach()
        item["delta_sum"] += torch.where(valid, delta_norm, torch.zeros_like(delta_norm)).detach()
        item["count"] += valid.to(item["count"].dtype).detach()
        item["valid_any"] |= valid.detach()

    def _finalize_output_cache(
        self,
        cache: Mapping[tuple[int, str, str], Mapping[str, torch.Tensor]],
        graph_indices: Sequence[int],
    ) -> None:
        for (layer, block, intervention), item in cache.items():
            denom = (item["da_sum"] + item["dm_sum"]).clamp_min(EPS)
            routing_resp = item["da_sum"] / denom
            transport_resp = item["dm_sum"] / denom
            sensitivity = (
                item["delta_sum"] / item["count"].clamp_min(1.0)
            ) / item["output_norm"].clamp_min(EPS)
            valid_any = item["valid_any"]
            self.acc.add_query_scores(
                routing_resp,
                valid_any,
                graph_indices=graph_indices,
                layer=layer,
                metric="output_routing_responsibility",
                field="output",
                intervention=intervention,
                block=block,
                centered=None,
                alpha_tau=None,
            )
            self.acc.add_query_scores(
                transport_resp,
                valid_any,
                graph_indices=graph_indices,
                layer=layer,
                metric="output_transport_responsibility",
                field="output",
                intervention=intervention,
                block=block,
                centered=None,
                alpha_tau=None,
            )
            self.acc.add_query_scores(
                sensitivity,
                valid_any,
                graph_indices=graph_indices,
                layer=layer,
                metric="output_sensitivity",
                field="output",
                intervention=intervention,
                block=block,
                centered=None,
                alpha_tau=None,
            )

    def _add_gates(
        self,
        clean: LayerFields,
        batch: Any,
        block: str,
        block_mask: torch.Tensor,
        graph_indices: Sequence[int],
    ) -> None:
        if block != "global":
            return
        mask = expand_mask(clean.mask, clean.attention) & block_mask[:, None].to(
            device=clean.attention.device
        )
        route_gate = torch.where(mask, clean.attention, torch.zeros_like(clean.attention)).sum(
            dim=3
        )
        msg = masked_key_field(clean.message, mask)
        attn = masked_key_field(clean.attention, mask).unsqueeze(-1)
        trans_global = (attn * msg).sum(dim=3)
        full_out = output_from_fields(clean.attention, clean.message, clean.mask)
        trans_gate = torch.sqrt(
            (trans_global * trans_global).sum(dim=-1).clamp_min(0.0)
        ) / torch.sqrt((full_out * full_out).sum(dim=-1).clamp_min(EPS))
        valid = clean.node_mask[:, None, :].expand(-1, clean.attention.size(1), -1)
        self.acc.add_query_scores(
            route_gate,
            valid,
            graph_indices=graph_indices,
            layer=clean.layer,
            metric="global_routing_gate",
            field="routing",
            intervention="none",
            block=block,
            centered=None,
            alpha_tau=None,
        )
        self.acc.add_query_scores(
            trans_gate,
            valid,
            graph_indices=graph_indices,
            layer=clean.layer,
            metric="global_transport_gate",
            field="output",
            intervention="none",
            block=block,
            centered=None,
            alpha_tau=None,
        )

    def _add_attention_entropy(
        self,
        clean: LayerFields,
        graph_indices: Sequence[int],
    ) -> None:
        entropy, valid = normalized_attention_entropy(clean.attention, clean.mask)
        node_valid = clean.node_mask[:, None, :].expand(-1, clean.attention.size(1), -1)
        self.acc.add_query_scores(
            entropy,
            valid & node_valid,
            graph_indices=graph_indices,
            layer=clean.layer,
            metric="attention_entropy",
            field="routing",
            intervention="none",
            block="all",
            centered=None,
            alpha_tau=None,
        )

    def results(self) -> ResultBundle:
        return ResultBundle(
            per_graph_rows=self.acc.per_graph_rows,
            query_rows=self.acc.query_rows,
            summary_rows=self.acc.summaries(),
        )


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


def metric_protocol_audit() -> dict[str, Any]:
    """Machine-readable notes tying the implementation to the Section 3 metrics."""

    return {
        "routing_metric": {
            "field": "A_i,:",
            "follow_reference": "A_i,: P_pi",
            "invariant_reference": "A_i,:",
            "similarity": "masked cosine over keys",
            "centered_variant": "subtract masked key mean before cosine",
        },
        "transport_metric": {
            "field": "m_i,: = M(h_j, r_ij)",
            "follow_reference": "P_pi m_i,: implemented as key-axis gather",
            "invariant_reference": "m_i,:",
            "similarity": "masked Frobenius cosine over key-message field",
            "centered_variant": "subtract masked key mean per message channel before cosine",
        },
        "alpha_weighting": {
            "normalization_axis": "sampled transpositions per graph/head/query",
            "routing_logit": "0.5 * ||A_i,: - A_i,: P_pi||_1 / tau",
            "transport_logit": "0.5 * ||m_i,: - P_pi m_i,:||_F / tau",
            "computed_from": "clean unintervened field",
            "shared_across": "content and structural variants for the same field",
            "implementation": "streaming log-sum-exp; no per-permutation activation cache",
        },
        "permutation_family": {
            "family": "within-graph node transpositions",
            "block_policy": (
                "all samples a graph-level transposition; local/global scores use the same "
                "sample and keep only queries for which both swapped nodes lie in the query block"
            ),
            "local_block": "N(i), excluding self",
            "global_block": "V \\ (N(i) union {i})",
        },
        "interventions": {
            "content": "permute symbolic node_type rows; keep all topology-derived tensors fixed",
            "structure": (
                "permute topology-derived node and pair tensors jointly; keep symbolic node_type "
                "fixed. For official GRIT this includes degree, RRWP, sparse edge endpoints, and "
                "edge values attached to their structural edge identities"
            ),
            "coordinate_convention": (
                "queries remain in fixed output coordinates; equivariant references are key-side "
                "fields Q_i,: P_pi, matching the draft's fixed-query view"
            ),
        },
        "realised_output_metric": {
            "decomposition": "Delta o_i = (A'_i,: - A_i,:) m_bar_i,: + A_bar_i,: (m'_i,: - m_i,:)",
            "routing_responsibility": "||delta_A|| / (||delta_A|| + ||delta_m||)",
            "transport_responsibility": "||delta_m|| / (||delta_A|| + ||delta_m||)",
            "sensitivity": "mean_pi ||Delta o_i|| / (||o_i|| + eps)",
            "alpha_weighted": False,
            "centered_variant": "not computed by default, matching Section 3",
        },
        "grit_adapter": {
            "attention": "post-softmax per-head sparse GRIT attention densified to [B,H,N,N]",
            "transport": (
                "pre-attention message V_h[src] plus GRIT edge_enhance relation term when present; "
                "attention is stored separately so output is sum_j A_ij m_ij"
            ),
            "source": "official GRIT GritTransformerLayer forward hooks",
        },
        "attention_entropy": {
            "definition": "mean query entropy normalised by log(valid key count)",
            "range": "0=delta-like routing, 1=uniform over valid keys",
            "intervention": "none",
            "block": "all",
        },
        "known_ambiguities_resolved": [
            {
                "ambiguity": "whether structural swaps should also reindex query rows",
                "resolution": (
                    "no; all scores are from a fixed-query coordinate system and only compare "
                    "key fields against Q_i,: and Q_i,: P_pi"
                ),
            },
            {
                "ambiguity": "whether output metric should reuse alpha weights",
                "resolution": (
                    "no; Section 3 states Metric 3 is norm-pooled across swaps, "
                    "not alpha-weighted"
                ),
            },
            {
                "ambiguity": "whether local/global alpha is over all swaps or within-block swaps",
                "resolution": (
                    "within-block per query; implemented as rejection over the sampled graph-level "
                    "transpositions and normalised over valid sampled swaps"
                ),
            },
        ],
    }


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


def select_graphs(dataset: Any, num_graphs: int, seed: int) -> list[Any]:
    return [dataset[i] for i in select_graph_indices(dataset, num_graphs, seed)]


def make_collector(
    model: nn.Module,
    model_name: str,
    adapter: str,
    max_nodes: int,
) -> OfficialGRITFieldCollector:
    selected = adapter
    if selected == "auto":
        selected = "official_grit" if model_name in {"grit", "static_grit"} else "unsupported"
    if selected == "official_grit":
        return OfficialGRITFieldCollector(model, max_nodes=max_nodes)
    raise NotImplementedError(
        f"No field collector is available for model={model_name!r}, adapter={adapter!r}. "
        "Metric computation requires an adapter exposing per-layer/head attention and "
        "message fields. "
        "The current CLI supports official GRIT/static-GRIT checkpoints."
    )


def run(args: argparse.Namespace) -> ResultBundle:
    runner = load_runner(args.runner_path)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"checkpoint must be a mapping: {args.checkpoint}")
    cfg = build_screen_config(runner, args, checkpoint)

    log = print
    splits = runner.load_official_graphbench_task(
        args.dataset_root,
        args.task,
        cfg,
        force_reload=args.force_reload_data,
        log=log,
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
        log=log,
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
    collector = make_collector(
        model,
        args.model,
        args.adapter,
        max_nodes=max(graph.num_nodes for graph in graphs),
    )
    options = MetricOptions(
        metrics=parse_csv_set(args.metrics, DEFAULT_METRICS, all_values=DEFAULT_METRICS),
        interventions=parse_csv_set(
            args.interventions,
            DEFAULT_INTERVENTIONS,
            all_values=DEFAULT_INTERVENTIONS,
        ),
        blocks=parse_csv_set(
            args.blocks,
            ("all", "local", "global"),
            all_values=("all", "local", "global"),
        ),
        centered=parse_centered(args.centered),
        num_permutations=args.num_permutations,
        alpha_tau=args.alpha_tau,
        seed=args.permutation_seed,
    )
    engine = SpecialisationMetricEngine(
        collector,
        options,
        write_query_scores=args.write_query_scores,
    )

    start = time.time()
    batch_size = max(1, int(args.batch_size))
    for start_idx in range(0, len(graphs), batch_size):
        end_idx = min(len(graphs), start_idx + batch_size)
        batch_graphs = graphs[start_idx:end_idx]
        graph_indices = selected_indices[start_idx:end_idx]
        batch = runner.collate_graphs(batch_graphs).to(device)
        print(
            f"[metrics] batch={start_idx // batch_size + 1} "
            f"graphs={start_idx}:{end_idx} perms={options.num_permutations}",
            flush=True,
        )
        engine.compute_batch(batch, graph_indices=graph_indices)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = engine.results()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_graph_head_scores.csv", results.per_graph_rows)
    write_csv(out_dir / "per_head_summary.csv", results.summary_rows)
    if args.write_query_scores:
        write_csv(out_dir / "per_query_scores.csv", results.query_rows)
    metadata = {
        "task": args.task,
        "split": args.split,
        "model": args.model,
        "model_backend": args.model_backend,
        "adapter": args.adapter,
        "checkpoint": str(args.checkpoint),
        "num_graphs": len(graphs),
        "num_permutations": options.num_permutations,
        "alpha_tau": options.alpha_tau,
        "metrics": list(options.metrics),
        "interventions": list(options.interventions),
        "blocks": list(options.blocks),
        "centered": list(options.centered),
        "selected_graph_indices": selected_indices,
        "metric_protocol_audit": metric_protocol_audit(),
        "elapsed_seconds": time.time() - start,
        "device": str(device),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "theory_alignment.json").write_text(
        json.dumps(metric_protocol_audit(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[done] wrote {len(results.per_graph_rows)} per-graph rows and "
        f"{len(results.summary_rows)} summary rows to {out_dir}",
        flush=True,
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-backend", default="official", choices=("official", "local"))
    parser.add_argument("--adapter", default="auto", choices=("auto", "official_grit"))
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--num-graphs", type=int, default=32)
    parser.add_argument("--num-permutations", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--metrics", default="all", help="Comma list: routing,transport,output,all")
    parser.add_argument(
        "--interventions",
        default="content,structure",
        help="Comma list: content,structure,all",
    )
    parser.add_argument("--blocks", default="all", help="Comma list: all,local,global")
    parser.add_argument("--centered", default="both", help="both, centered, or uncentered")
    parser.add_argument("--alpha-tau", type=float, default=0.1)
    parser.add_argument("--permutation-seed", type=int, default=0)
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
        default=Path("outputs/core_interpretability_specialisation_metrics"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--write-query-scores", action="store_true")
    parser.add_argument("--force-reload-data", action="store_true")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--train-node-size", type=int, default=None)
    parser.add_argument("--val-node-size", type=int, default=None)
    parser.add_argument("--test-node-size", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
