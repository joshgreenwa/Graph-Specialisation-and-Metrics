#!/usr/bin/env python3
"""Attention-teacher calibration tasks for symbolic/structural metrics.

This runner is intentionally simpler than the graph-level teacher-student
benchmark. The teacher is a hand-coded attention operator over all query rows:

* symbolic_equality: each query key matches exactly one source key;
* path_predecessor: each path position attends to its anchored predecessor;
* tree_parent: each node attends to its rooted-tree parent;
* mirror_node: each node attends to its structurally mirrored counterpart;
* hop_then_key: structural hop routing plus symbolic selection inside the routed set;
* parent_then_key: tree parent/mirror structural routing plus symbolic selection;
* ring_quad_key: four-way ring structural routing plus symbolic selection;
* previous_same_key: nearest previous node with the same key on an anchored path.

The student is a tiny attention copy model. By default it is trained
with both value-label supervision and direct attention supervision, so this is a
positive control for the metric machinery rather than a natural task.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CELL_ARGS: list[str] | str | None = None

TASKS = (
    "symbolic_equality",
    "path_predecessor",
    "tree_parent",
    "mirror_node",
    "hop_then_key",
    "parent_then_key",
    "ring_quad_key",
    "previous_same_key",
)
TASK_ALIASES = {
    "symbolic": "symbolic_equality",
    "structural": "path_predecessor",
    "structural_predecessor": "path_predecessor",
    "mixed": "hop_then_key",
    "hop_key": "hop_then_key",
}
TASK_CHOICES = TASKS + tuple(TASK_ALIASES)

ARCHITECTURES = ("copy_attn",)
SUITE_TASKS = {
    "full_zoo": list(TASKS),
    "stability_three": ["symbolic_equality", "path_predecessor", "hop_then_key"],
}

M_POSITIONAL = "positional_score"
M_SYMBOLIC = "symbolic_score"
M_PE_INVARIANT = "pe_invariance"
M_PE_EQUIVARIANT = "pe_equivariance"
M_POSITIONAL_CENTERED = "positional_score_centered"
M_SYMBOLIC_CENTERED = "symbolic_score_centered"
M_PE_INVARIANT_CENTERED = "pe_invariance_centered"
M_PE_EQUIVARIANT_CENTERED = "pe_equivariance_centered"
M_INTERACTION_RESIDUAL_CENTERED = "interaction_residual_norm_centered"
M_JOINT_EQUIVARIANCE_EXCESS_CENTERED = "joint_equivariance_excess_centered"
M_ENTROPY = "entropy_norm"
M_RESIDUAL_NORM = "attention_residual_norm"
M_RELABEL_EQUIVARIANT = "relabel_equivariance"


@dataclass(frozen=True)
class OracleSpec:
    name: str
    base_task: str
    components: tuple[tuple[str, float], ...]


def running_in_notebook() -> bool:
    launcher = Path(sys.argv[0]).name
    if launcher in {"ipykernel_launcher.py", "colab_kernel_launcher.py"}:
        return True
    if "google.colab" in sys.modules or "ipykernel" in sys.modules:
        return True
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except Exception:
        return False
    return shell in {"ZMQInteractiveShell", "Shell"}


def notebook_safe_argv(argv: list[str] | None) -> list[str] | None:
    if argv is not None:
        return argv
    if not running_in_notebook():
        return sys.argv[1:]
    if CELL_ARGS is None:
        return []
    if isinstance(CELL_ARGS, str):
        return shlex.split(CELL_ARGS)
    return list(CELL_ARGS)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Batch:
    q_key: torch.Tensor
    k_key: torch.Tensor
    value_id: torch.Tensor
    pe: torch.Tensor
    mask: torch.Tensor
    source: torch.Tensor
    target_y: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            q_key=self.q_key.to(device),
            k_key=self.k_key.to(device),
            value_id=self.value_id.to(device),
            pe=self.pe.to(device),
            mask=self.mask.to(device),
            source=self.source.to(device),
            target_y=self.target_y.to(device),
        )


@dataclass
class MetricOutputs:
    metric_rows: list[dict]
    query_rows: list[dict]
    sanity_rows: list[dict]
    partition_rows: list[dict]
    attention_rows: list[dict]


def value_symbol(value: int) -> int:
    return int(value) + 1


def canonical_task(task: str) -> str:
    return TASK_ALIASES.get(task, task)


def structural_pe(num_nodes: int, hop: int = 2) -> np.ndarray:
    denom = max(1, num_nodes - 1)
    idx = np.arange(num_nodes, dtype=np.float32)
    t = idx / float(denom)
    pred = np.maximum(np.arange(num_nodes) - 1, 0).astype(np.float32) / float(denom)
    parent = np.asarray(
        [0 if node == 0 else (node - 1) // 2 for node in range(num_nodes)],
        dtype=np.float32,
    ) / float(denom)
    mirror = (num_nodes - 1 - idx) / float(denom)
    plus = ((np.arange(num_nodes) + hop) % num_nodes).astype(np.float32) / float(denom)
    minus = ((np.arange(num_nodes) - hop) % num_nodes).astype(np.float32) / float(denom)
    depth = np.asarray(
        [0 if node == 0 else int(math.floor(math.log2(node + 1))) for node in range(num_nodes)],
        dtype=np.float32,
    )
    depth = depth / max(1.0, float(depth.max()))
    parity = (np.arange(num_nodes) % 2).astype(np.float32)
    angle = 2.0 * math.pi * t
    return np.stack(
        [
            t,
            1.0 - t,
            pred,
            parent,
            mirror,
            plus,
            minus,
            depth,
            parity,
            np.sin(angle),
            np.cos(angle),
            np.ones_like(t),
        ],
        axis=1,
    ).astype(np.float32)


def repeated_keys(
    rng: np.random.Generator,
    num_nodes: int,
    key_vocab_size: int,
    period: int | None = None,
) -> np.ndarray:
    if period is None:
        period = max(2, min(key_vocab_size, max(2, num_nodes // 4)))
    period = max(2, min(period, key_vocab_size))
    offset = int(rng.integers(0, key_vocab_size))
    return ((np.arange(num_nodes, dtype=np.int64) + offset) % period).astype(np.int64)


def ring_quad_offsets(num_nodes: int) -> list[int]:
    candidates = [1, max(2, num_nodes // 4), max(3, num_nodes // 2 - 1), num_nodes - 1]
    out = []
    for offset in candidates + list(range(1, num_nodes)):
        offset = int(offset % num_nodes)
        if offset != 0 and offset not in out:
            out.append(offset)
        if len(out) == 4:
            break
    return out


def structural_candidates(task: str, query: int, num_nodes: int, hop: int | None = None) -> list[int]:
    task = canonical_task(task)
    if hop is None:
        hop = max(2, num_nodes // 4)
    if task == "path_predecessor":
        return [0 if query == 0 else query - 1]
    if task == "tree_parent":
        return [0 if query == 0 else (query - 1) // 2]
    if task == "mirror_node":
        return [num_nodes - 1 - query]
    if task == "hop_then_key":
        return [int((query - hop) % num_nodes), int((query + hop) % num_nodes)]
    if task == "parent_then_key":
        parent = 0 if query == 0 else (query - 1) // 2
        mirror = num_nodes - 1 - query
        candidates = [int(parent), int(mirror)]
        if candidates[0] == candidates[1]:
            candidates.append(int((query + hop) % num_nodes))
        return list(dict.fromkeys(candidates))
    if task == "ring_quad_key":
        return [int((query + offset) % num_nodes) for offset in ring_quad_offsets(num_nodes)]
    if task == "previous_same_key":
        return list(range(0, query + 1))
    return list(range(num_nodes))


def generate_batch(
    task: str,
    batch_size: int,
    num_nodes: int,
    key_vocab_size: int,
    value_vocab_size: int,
    seed: int,
) -> Batch:
    if key_vocab_size < num_nodes:
        raise ValueError("--key-vocab-size must be >= --num-nodes")
    if value_vocab_size < num_nodes:
        raise ValueError("--value-vocab-size must be >= --num-nodes")

    task = canonical_task(task)
    hop = max(2, num_nodes // 4)
    rng = np.random.default_rng(seed)
    q_key = np.zeros((batch_size, num_nodes), dtype=np.int64)
    k_key = np.zeros((batch_size, num_nodes), dtype=np.int64)
    value_id = np.zeros((batch_size, num_nodes), dtype=np.int64)
    source = np.zeros((batch_size, num_nodes), dtype=np.int64)
    target_y = np.zeros((batch_size, num_nodes), dtype=np.int64)
    pe_one = structural_pe(num_nodes, hop=hop)
    pe = np.repeat(pe_one[None, :, :], batch_size, axis=0)

    for graph_idx in range(batch_size):
        values = rng.choice(value_vocab_size, size=num_nodes, replace=False).astype(np.int64)
        value_id[graph_idx] = values + 1
        if task == "symbolic_equality":
            keys = rng.choice(key_vocab_size, size=num_nodes, replace=False).astype(np.int64)
            query_to_source = rng.permutation(num_nodes).astype(np.int64)
            k_key[graph_idx] = keys
            q_key[graph_idx] = keys[query_to_source]
            source[graph_idx] = query_to_source
        elif task == "path_predecessor":
            k_key[graph_idx] = 0
            q_key[graph_idx] = 0
            source[graph_idx, 0] = 0
            source[graph_idx, 1:] = np.arange(num_nodes - 1, dtype=np.int64)
        elif task == "tree_parent":
            k_key[graph_idx] = 0
            q_key[graph_idx] = 0
            source[graph_idx, 0] = 0
            source[graph_idx, 1:] = ((np.arange(1, num_nodes) - 1) // 2).astype(np.int64)
        elif task == "mirror_node":
            k_key[graph_idx] = 0
            q_key[graph_idx] = 0
            source[graph_idx] = (num_nodes - 1 - np.arange(num_nodes)).astype(np.int64)
        elif task == "hop_then_key":
            keys = repeated_keys(rng, num_nodes, key_vocab_size)
            k_key[graph_idx] = keys
            for query in range(num_nodes):
                left, right = structural_candidates(task, query, num_nodes, hop)
                chosen, other = (left, right) if rng.random() < 0.5 else (right, left)
                source[graph_idx, query] = chosen
                if keys[chosen] == keys[other]:
                    keys[other] = int((keys[chosen] + 1) % max(2, min(key_vocab_size, num_nodes)))
            q_key[graph_idx] = keys[source[graph_idx]]
            k_key[graph_idx] = keys
        elif task in {"parent_then_key", "ring_quad_key"}:
            keys = repeated_keys(rng, num_nodes, key_vocab_size, period=max(4, num_nodes // 3))
            for query in range(num_nodes):
                candidates = structural_candidates(task, query, num_nodes, hop)
                chosen = int(candidates[int(rng.integers(0, len(candidates)))])
                source[graph_idx, query] = chosen
                used = {int(keys[chosen])}
                for candidate in candidates:
                    if candidate == chosen:
                        continue
                    if int(keys[candidate]) in used:
                        keys[candidate] = int((max(used) + 1) % key_vocab_size)
                    used.add(int(keys[candidate]))
            q_key[graph_idx] = keys[source[graph_idx]]
            k_key[graph_idx] = keys
        elif task == "previous_same_key":
            keys = repeated_keys(rng, num_nodes, key_vocab_size)
            k_key[graph_idx] = keys
            q_key[graph_idx] = keys
            for query in range(num_nodes):
                previous = [
                    node for node in range(query - 1, -1, -1)
                    if int(keys[node]) == int(keys[query])
                ]
                source[graph_idx, query] = previous[0] if previous else query
        else:
            raise ValueError(f"Unknown task: {task}")
        target_y[graph_idx] = values[source[graph_idx]]

    return Batch(
        q_key=torch.from_numpy(q_key),
        k_key=torch.from_numpy(k_key),
        value_id=torch.from_numpy(value_id),
        pe=torch.from_numpy(pe),
        mask=torch.ones((batch_size, num_nodes), dtype=torch.bool),
        source=torch.from_numpy(source),
        target_y=torch.from_numpy(target_y),
    )


class CopyStudent(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        key_vocab_size: int,
        value_vocab_size: int,
        pe_dim: int,
        fixed_value_embeddings: bool,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("--hidden-dim must be divisible by --num-heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.value_vocab_size = value_vocab_size
        self.q_key_emb = nn.Embedding(key_vocab_size, hidden_dim)
        self.k_key_emb = nn.Embedding(key_vocab_size, hidden_dim)
        self.value_emb = nn.Embedding(value_vocab_size + 1, hidden_dim)
        if fixed_value_embeddings:
            self._init_fixed_value_embeddings(hidden_dim, value_vocab_size)
        self.pe_proj = nn.Linear(pe_dim, hidden_dim, bias=False)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_bias = nn.Parameter(torch.zeros(value_vocab_size))

    def _init_fixed_value_embeddings(self, hidden_dim: int, value_vocab_size: int) -> None:
        with torch.no_grad():
            self.value_emb.weight.zero_()
            eye_dim = min(hidden_dim, value_vocab_size)
            self.value_emb.weight[1 : eye_dim + 1, :eye_dim] = (
                torch.eye(eye_dim) * math.sqrt(float(hidden_dim))
            )
            if value_vocab_size > eye_dim:
                extra = torch.randn(value_vocab_size - eye_dim, hidden_dim)
                extra = F.normalize(extra, dim=-1) * math.sqrt(float(hidden_dim))
                self.value_emb.weight[eye_dim + 1 :] = extra
        self.value_emb.weight.requires_grad_(False)

    def forward(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        selector_h = (
            self.q_key_emb(batch.q_key)
            + self.k_key_emb(batch.k_key)
            + self.pe_proj(batch.pe)
        )
        batch_size, num_nodes, _ = selector_h.shape
        q = self.q_proj(selector_h).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(selector_h).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        pair_mask = batch.mask[:, None, :, None] & batch.mask[:, None, None, :]
        masked_logits = logits.masked_fill(~pair_mask, -1.0e9)
        attn = torch.softmax(masked_logits, dim=-1).masked_fill(~pair_mask, 0.0)
        value_codes = self.value_emb(batch.value_id).view(
            batch_size, num_nodes, self.num_heads, self.head_dim
        ).transpose(1, 2)
        context = torch.matmul(attn, value_codes)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        z = self.out_proj(self.out_norm(context))
        decoder_codes = self.value_emb.weight[1 : self.value_vocab_size + 1]
        pred_logits = (z @ decoder_codes.t()) / math.sqrt(float(z.size(-1))) + self.out_bias
        layer = {
            "logits": logits,
            "attn": attn,
            "node_mask": batch.mask,
        }
        return pred_logits, layer


OneHeadCopyStudent = CopyStudent


def batch_loss(
    pred_logits: torch.Tensor,
    layer: dict[str, torch.Tensor],
    batch: Batch,
    attention_loss_weight: float,
    attention_loss_mode: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value_loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        batch.target_y.reshape(-1),
    )
    attn = layer["attn"]
    source_idx = batch.source[:, None, :, None].expand(-1, attn.size(1), -1, 1)
    source_mass = torch.gather(attn, dim=-1, index=source_idx).squeeze(-1)
    if attention_loss_mode == "best_head":
        source_mass_for_loss = source_mass.max(dim=1).values
    elif attention_loss_mode == "mean":
        source_mass_for_loss = source_mass
    else:
        raise ValueError(f"Unknown attention loss mode: {attention_loss_mode}")
    attention_loss = -torch.log(source_mass_for_loss.clamp_min(1.0e-8)).mean()
    loss = value_loss + attention_loss_weight * attention_loss
    return loss, value_loss, attention_loss


@torch.no_grad()
def evaluate(
    model: CopyStudent,
    task: str,
    num_graphs: int,
    batch_size: int,
    num_nodes: int,
    key_vocab_size: int,
    value_vocab_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    losses = []
    source_mrr = []
    source_mass = []
    for offset in range(0, num_graphs, batch_size):
        current = min(batch_size, num_graphs - offset)
        batch = generate_batch(
            task, current, num_nodes, key_vocab_size, value_vocab_size, seed + offset
        ).to(device)
        pred_logits, layer = model(batch)
        loss, _, _ = batch_loss(pred_logits, layer, batch, attention_loss_weight=0.0)
        pred = pred_logits.argmax(dim=-1)
        correct += int((pred == batch.target_y).sum().item())
        total += int(batch.target_y.numel())
        losses.append(float(loss.item()))
        attn_rows = layer["attn"]
        source_idx = batch.source[:, None, :, None].expand(-1, attn_rows.size(1), -1, 1)
        mass = torch.gather(attn_rows, dim=-1, index=source_idx).squeeze(-1)
        rank = (attn_rows > mass[:, :, :, None]).sum(dim=-1).float() + 1.0
        source_mass.extend(mass.detach().cpu().reshape(-1).tolist())
        source_mrr.extend((1.0 / rank).detach().cpu().reshape(-1).tolist())
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": correct / max(1, total),
        "teacher_source_mass": float(np.mean(source_mass)) if source_mass else 0.0,
        "teacher_source_mrr": float(np.mean(source_mrr)) if source_mrr else 0.0,
    }


def gather_dense_node_axis(t: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    batch_size, max_n = perm_pos.shape
    extra = t.dim() - 2
    idx = perm_pos.to(t.device).view(batch_size, max_n, *([1] * extra)).expand_as(t)
    return torch.gather(t, dim=1, index=idx)


def transform_pair_reference(z: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    batch_size, heads, max_n, _ = z.shape
    idx = perm_pos.to(z.device)
    row_idx = idx[:, None, :, None].expand(batch_size, heads, max_n, max_n)
    z_rows = torch.gather(z, dim=2, index=row_idx)
    col_idx = idx[:, None, None, :].expand(batch_size, heads, max_n, max_n)
    return torch.gather(z_rows, dim=3, index=col_idx)


def row_center_logits(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=z.device, dtype=torch.bool)
    if m.size(1) == 1 and z.size(1) != 1:
        m = m.expand(-1, z.size(1), -1, -1)
    z0 = torch.where(m, torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    denom = m.sum(dim=-1, keepdim=True).clamp_min(1).to(z.dtype)
    mean = z0.sum(dim=-1, keepdim=True) / denom
    return torch.where(m, z0 - mean, torch.zeros_like(z0))


def cosine_by_head_logits(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=u.device, dtype=torch.bool)
    if m.size(1) == 1 and u.size(1) != 1:
        m = m.expand(-1, u.size(1), -1, -1)
    u0 = row_center_logits(u, m)
    v0 = row_center_logits(v, m)
    dims = (0, 2, 3)
    num = (u0 * v0).sum(dim=dims)
    den = (
        torch.sqrt((u0 * u0).sum(dim=dims).clamp_min(1.0e-12))
        * torch.sqrt((v0 * v0).sum(dim=dims).clamp_min(1.0e-12))
    )
    return (num / den.clamp_min(1.0e-12)).detach().cpu()


def cosine_by_query_logits(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=u.device, dtype=torch.bool)
    if m.size(1) == 1 and u.size(1) != 1:
        m = m.expand(-1, u.size(1), -1, -1)
    u0 = row_center_logits(u, m)
    v0 = row_center_logits(v, m)
    num = (u0 * v0).sum(dim=-1)
    den = (
        torch.sqrt((u0 * u0).sum(dim=-1).clamp_min(1.0e-12))
        * torch.sqrt((v0 * v0).sum(dim=-1).clamp_min(1.0e-12))
    )
    return (num / den.clamp_min(1.0e-12)).detach().cpu()


def attention_cosine_by_query(
    u_attn: torch.Tensor,
    v_attn: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.to(device=u_attn.device, dtype=torch.bool)
    if m.size(1) == 1 and u_attn.size(1) != 1:
        m = m.expand(-1, u_attn.size(1), -1, -1)
    u0 = torch.where(m, u_attn, torch.zeros_like(u_attn))
    v0 = torch.where(m, v_attn, torch.zeros_like(v_attn))
    num = (u0 * v0).sum(dim=-1)
    den = (
        torch.sqrt((u0 * u0).sum(dim=-1).clamp_min(1.0e-12))
        * torch.sqrt((v0 * v0).sum(dim=-1).clamp_min(1.0e-12))
    )
    return torch.clamp(num / den.clamp_min(1.0e-12), 0.0, 1.0)


def centered_attention_tensor_from_layer(
    layer: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = pair_mask(layer).to(device=layer["attn"].device, dtype=torch.bool)
    if mask.size(1) == 1 and layer["attn"].size(1) != 1:
        mask = mask.expand(-1, layer["attn"].size(1), -1, -1)
    return row_center_logits(layer["attn"], mask), mask


def metric_tensor_from_layer(
    layer: dict[str, torch.Tensor],
    metric_space: str,
    centered: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if centered:
        raise ValueError(
            "Centred scores must use centered_attention_tensor_from_layer() "
            "and raw_cosine_by_query_tensor(), not shifted signed cosine."
        )
    mask = pair_mask(layer).to(device=layer["logits"].device, dtype=torch.bool)
    if metric_space == "attention":
        tensor = layer["attn"]
        signed = False
    elif metric_space == "logits":
        tensor = torch.nan_to_num(layer["logits"], nan=0.0, posinf=0.0, neginf=0.0)
        signed = True
    else:
        raise ValueError(f"Unknown metric space: {metric_space}")
    if mask.size(1) == 1 and tensor.size(1) != 1:
        mask = mask.expand(-1, tensor.size(1), -1, -1)
    return tensor, mask, signed


def cosine_by_query_tensor(
    u: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    signed: bool,
) -> torch.Tensor:
    m = mask.to(device=u.device, dtype=torch.bool)
    if m.size(1) == 1 and u.size(1) != 1:
        m = m.expand(-1, u.size(1), -1, -1)
    u0 = torch.where(m, torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(u))
    v0 = torch.where(m, torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(v))
    num = (u0 * v0).sum(dim=-1)
    den = (
        torch.sqrt((u0 * u0).sum(dim=-1).clamp_min(1.0e-12))
        * torch.sqrt((v0 * v0).sum(dim=-1).clamp_min(1.0e-12))
    )
    cos = num / den.clamp_min(1.0e-12)
    if signed:
        return torch.clamp(0.5 * (cos + 1.0), 0.0, 1.0)
    return torch.clamp(cos, 0.0, 1.0)


def raw_cosine_by_query_tensor(
    u: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.to(device=u.device, dtype=torch.bool)
    if m.size(1) == 1 and u.size(1) != 1:
        m = m.expand(-1, u.size(1), -1, -1)
    u0 = torch.where(m, torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(u))
    v0 = torch.where(m, torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(v))
    num = (u0 * v0).sum(dim=-1)
    den = (
        torch.sqrt((u0 * u0).sum(dim=-1).clamp_min(1.0e-12))
        * torch.sqrt((v0 * v0).sum(dim=-1).clamp_min(1.0e-12))
    )
    return torch.clamp(num / den.clamp_min(1.0e-12), -1.0, 1.0)


def interaction_residual_norm_by_query_tensor(
    clean_c: torch.Tensor,
    var_x_c: torch.Tensor,
    var_pe_c: torch.Tensor,
    var_both_c: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    m = mask.to(device=clean_c.device, dtype=torch.bool)
    if m.size(1) == 1 and clean_c.size(1) != 1:
        m = m.expand(-1, clean_c.size(1), -1, -1)
    clean0 = torch.where(m, clean_c, torch.zeros_like(clean_c))
    x0 = torch.where(m, var_x_c, torch.zeros_like(var_x_c))
    pe0 = torch.where(m, var_pe_c, torch.zeros_like(var_pe_c))
    both0 = torch.where(m, var_both_c, torch.zeros_like(var_both_c))
    delta_x = x0 - clean0
    delta_pe = pe0 - clean0
    delta_both = both0 - clean0
    interaction = both0 - x0 - pe0 + clean0
    numerator = torch.sqrt((interaction * interaction).sum(dim=-1).clamp_min(0.0))
    denominator = (
        torch.sqrt((delta_both * delta_both).sum(dim=-1).clamp_min(0.0))
        + torch.sqrt((delta_x * delta_x).sum(dim=-1).clamp_min(0.0))
        + torch.sqrt((delta_pe * delta_pe).sum(dim=-1).clamp_min(0.0))
    )
    return torch.clamp(numerator / denominator.clamp_min(1.0e-12), 0.0, 1.0)


def attention_moved_mass_by_query(
    clean_attn: torch.Tensor,
    perm_pos: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ref = transform_pair_reference(clean_attn, perm_pos)
    ref_mask = transform_pair_reference(mask, perm_pos).to(dtype=torch.bool)
    m = (mask.to(device=clean_attn.device, dtype=torch.bool) | ref_mask)
    if m.size(1) == 1 and clean_attn.size(1) != 1:
        m = m.expand(-1, clean_attn.size(1), -1, -1)
    clean0 = torch.where(m, clean_attn, torch.zeros_like(clean_attn))
    ref0 = torch.where(m, ref, torch.zeros_like(ref))
    return torch.clamp(0.5 * torch.abs(clean0 - ref0).sum(dim=-1), 0.0, 1.0)


def pair_mask(layer: dict[str, torch.Tensor]) -> torch.Tensor:
    node_mask = layer["node_mask"]
    return node_mask[:, None, :, None] & node_mask[:, None, None, :]


def pair_mask_from_batch(batch: Batch) -> torch.Tensor:
    return batch.mask[:, None, :, None] & batch.mask[:, None, None, :]


def _normalize_attention_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=scores.device, dtype=torch.bool)
    scores = torch.where(m, scores.clamp_min(0.0), torch.zeros_like(scores))
    denom = scores.sum(dim=-1, keepdim=True)
    key_count = m.sum(dim=-1, keepdim=True).clamp_min(1).to(scores.dtype)
    uniform = torch.where(m, torch.ones_like(scores) / key_count, torch.zeros_like(scores))
    return torch.where(denom > 1.0e-12, scores / denom.clamp_min(1.0e-12), uniform)


def uniform_oracle_attention(batch: Batch) -> torch.Tensor:
    mask = pair_mask_from_batch(batch).squeeze(1)
    return _normalize_attention_scores(mask.to(torch.float32), mask)


def symbolic_oracle_attention(batch: Batch) -> torch.Tensor:
    mask = pair_mask_from_batch(batch).squeeze(1)
    scores = (batch.q_key[:, :, None] == batch.k_key[:, None, :]).to(torch.float32)
    return _normalize_attention_scores(scores, mask)


def path_predecessor_oracle_attention(batch: Batch) -> torch.Tensor:
    mask = pair_mask_from_batch(batch).squeeze(1)
    key_pos = batch.pe[:, :, 0]
    query_pred_pos = batch.pe[:, :, 2]
    dist = torch.abs(query_pred_pos[:, :, None] - key_pos[:, None, :])
    dist = torch.where(mask, dist, torch.full_like(dist, 1.0e9))
    source = dist.argmin(dim=-1)
    scores = torch.zeros_like(dist)
    scores.scatter_(2, source[:, :, None], 1.0)
    return _normalize_attention_scores(scores, mask)


def hop_support_oracle_attention(batch: Batch) -> torch.Tensor:
    mask = pair_mask_from_batch(batch).squeeze(1)
    key_pos = batch.pe[:, :, 0]
    query_plus = batch.pe[:, :, 5]
    query_minus = batch.pe[:, :, 6]
    dist_plus = torch.abs(query_plus[:, :, None] - key_pos[:, None, :])
    dist_minus = torch.abs(query_minus[:, :, None] - key_pos[:, None, :])
    dist_plus = torch.where(mask, dist_plus, torch.full_like(dist_plus, 1.0e9))
    dist_minus = torch.where(mask, dist_minus, torch.full_like(dist_minus, 1.0e9))
    src_plus = dist_plus.argmin(dim=-1)
    src_minus = dist_minus.argmin(dim=-1)
    scores = torch.zeros_like(dist_plus)
    scores.scatter_add_(2, src_plus[:, :, None], torch.ones_like(src_plus[:, :, None], dtype=scores.dtype))
    scores.scatter_add_(2, src_minus[:, :, None], torch.ones_like(src_minus[:, :, None], dtype=scores.dtype))
    return _normalize_attention_scores(scores, mask)


def mixed_hop_key_oracle_attention(batch: Batch) -> torch.Tensor:
    mask = pair_mask_from_batch(batch).squeeze(1)
    structural = hop_support_oracle_attention(batch) > 0
    symbolic = batch.q_key[:, :, None] == batch.k_key[:, None, :]
    scores = (structural & symbolic & mask).to(torch.float32)
    if bool((scores.sum(dim=-1) == 0).any()):
        fallback = structural.to(torch.float32)
        scores = torch.where(scores.sum(dim=-1, keepdim=True) > 0, scores, fallback)
    return _normalize_attention_scores(scores, mask)


def oracle_component_attention(batch: Batch, component: str) -> torch.Tensor:
    if component == "uniform":
        return uniform_oracle_attention(batch)
    if component == "symbolic":
        return symbolic_oracle_attention(batch)
    if component == "path_predecessor":
        return path_predecessor_oracle_attention(batch)
    if component == "hop_support":
        return hop_support_oracle_attention(batch)
    if component == "mixed_hop_key":
        return mixed_hop_key_oracle_attention(batch)
    raise ValueError(f"Unknown oracle component: {component}")


def oracle_attention(batch: Batch, components: Iterable[tuple[str, float]]) -> torch.Tensor:
    attn = None
    total = 0.0
    for component, weight in components:
        if weight <= 0:
            continue
        component_attn = oracle_component_attention(batch, component)
        attn = component_attn * float(weight) if attn is None else attn + component_attn * float(weight)
        total += float(weight)
    if attn is None or total <= 0:
        attn = uniform_oracle_attention(batch)
    else:
        attn = attn / total
    return _normalize_attention_scores(attn, pair_mask_from_batch(batch).squeeze(1))


class OracleAttentionModel(nn.Module):
    def __init__(
        self,
        components: Iterable[tuple[str, float]],
        value_vocab_size: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.components = tuple((str(name), float(weight)) for name, weight in components)
        self.value_vocab_size = value_vocab_size
        self.num_heads = num_heads

    def forward(self, batch: Batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn_one = oracle_attention(batch, self.components).to(batch.pe.device)
        attn = attn_one[:, None, :, :].expand(-1, self.num_heads, -1, -1).contiguous()
        mask = pair_mask_from_batch(batch).to(device=attn.device, dtype=torch.bool)
        logits = torch.log(attn.clamp_min(1.0e-12)).masked_fill(~mask, -1.0e9)
        pred_logits = torch.zeros(
            batch.q_key.size(0),
            batch.q_key.size(1),
            self.value_vocab_size,
            device=attn.device,
            dtype=attn.dtype,
        )
        return pred_logits, {"logits": logits, "attn": attn, "node_mask": batch.mask}


def build_oracle_specs(args: argparse.Namespace) -> list[OracleSpec]:
    specs = [
        OracleSpec("oracle_uniform", "symbolic_equality", (("uniform", 1.0),)),
        OracleSpec("oracle_symbolic", "symbolic_equality", (("symbolic", 1.0),)),
        OracleSpec("oracle_structural_path", "path_predecessor", (("path_predecessor", 1.0),)),
        OracleSpec("oracle_mixed_hop_key", "hop_then_key", (("mixed_hop_key", 1.0),)),
    ]
    if args.oracle_suite_preset in {"mix_sweep", "full"}:
        for lam in args.oracle_lambdas:
            tag = str(lam).replace(".", "p")
            specs.extend(
                [
                    OracleSpec(
                        f"oracle_symbolic_lam{tag}",
                        "symbolic_equality",
                        (("uniform", 1.0 - float(lam)), ("symbolic", float(lam))),
                    ),
                    OracleSpec(
                        f"oracle_structural_lam{tag}",
                        "path_predecessor",
                        (("uniform", 1.0 - float(lam)), ("path_predecessor", float(lam))),
                    ),
                    OracleSpec(
                        f"oracle_mixed_lam{tag}",
                        "hop_then_key",
                        (("uniform", 1.0 - float(lam)), ("mixed_hop_key", float(lam))),
                    ),
                ]
            )
    if args.oracle_specs:
        wanted = set(args.oracle_specs)
        specs = [spec for spec in specs if spec.name in wanted]
        missing = wanted - {spec.name for spec in specs}
        if missing:
            raise ValueError(f"Unknown --oracle-specs entries: {sorted(missing)}")
    return specs


def partition_masks(batch: Batch, task: str) -> dict[str, torch.Tensor]:
    task = canonical_task(task)
    base = pair_mask_from_batch(batch)
    batch_size, num_nodes = batch.mask.shape
    device = batch.mask.device
    source_mask = torch.zeros((batch_size, 1, num_nodes, num_nodes), dtype=torch.bool, device=device)
    source_mask.scatter_(3, batch.source[:, None, :, None], True)
    source_mask = source_mask & base

    support_mask = torch.zeros_like(source_mask)
    hop = max(2, num_nodes // 4)
    for query in range(num_nodes):
        candidates = structural_candidates(task, query, num_nodes, hop)
        support_mask[:, 0, query, candidates] = True
    support_mask = support_mask & base

    symbolic_match = (
        batch.q_key[:, None, :, None] == batch.k_key[:, None, None, :]
    ) & base

    masks = {
        "all": base,
        "teacher_source": source_mask,
        "non_source": base & ~source_mask,
        "symbolic_match_set": symbolic_match,
        "symbolic_nonmatch": base & ~symbolic_match,
    }
    if task in {
        "path_predecessor",
        "tree_parent",
        "mirror_node",
        "hop_then_key",
        "parent_then_key",
        "ring_quad_key",
        "previous_same_key",
    }:
        masks["structural_support"] = support_mask
        masks["outside_structural_support"] = base & ~support_mask
        masks["structural_distractor"] = support_mask & ~source_mask
    return masks


def make_perm(batch: Batch, generator: torch.Generator) -> torch.Tensor:
    batch_size, max_n = batch.mask.shape
    perm_pos = torch.zeros((batch_size, max_n), dtype=torch.long, device=batch.mask.device)
    for graph_idx in range(batch_size):
        n = int(batch.mask[graph_idx].sum().item())
        perm_pos[graph_idx, :n] = torch.randperm(n, generator=generator).to(batch.mask.device)
    return perm_pos


def x_permuted(batch: Batch, perm_pos: torch.Tensor) -> Batch:
    return replace(
        batch,
        q_key=gather_dense_node_axis(batch.q_key.unsqueeze(-1), perm_pos).squeeze(-1),
        k_key=gather_dense_node_axis(batch.k_key.unsqueeze(-1), perm_pos).squeeze(-1),
        value_id=gather_dense_node_axis(batch.value_id.unsqueeze(-1), perm_pos).squeeze(-1),
    )


def pe_permuted(batch: Batch, perm_pos: torch.Tensor) -> Batch:
    return replace(batch, pe=gather_dense_node_axis(batch.pe, perm_pos))


def x_pe_permuted(batch: Batch, perm_pos: torch.Tensor) -> Batch:
    return replace(x_permuted(batch, perm_pos), pe=gather_dense_node_axis(batch.pe, perm_pos))


def relabel_permuted(batch: Batch, perm_pos: torch.Tensor) -> Batch:
    return replace(x_pe_permuted(batch, perm_pos), mask=gather_dense_node_axis(batch.mask.unsqueeze(-1), perm_pos).squeeze(-1))


def entropy_rows(layer: dict[str, torch.Tensor], meta: dict, batch_idx: int) -> list[dict]:
    z = layer["logits"]
    mask = pair_mask(layer).to(device=z.device, dtype=torch.bool)
    z_masked = torch.where(mask, z, torch.full_like(z, -1.0e9))
    attn = torch.softmax(z_masked, dim=-1).masked_fill(~mask, 0.0)
    key_count = mask.sum(dim=-1).to(attn.dtype)
    ent_raw = -(attn * torch.log(attn.clamp_min(1.0e-12))).sum(dim=-1)
    ent_norm = ent_raw / torch.log(key_count.clamp_min(2.0))
    entropy = ent_norm.mean(dim=(0, 2))
    return [
        {
            **meta,
            "batch": batch_idx,
            "perm": -1,
            "layer": 0,
            "head": head,
            "metric": M_ENTROPY,
            "score": float(entropy[head].detach().cpu()),
        }
        for head in range(int(z.size(1)))
    ]


def per_query_entropy_rows(layer: dict[str, torch.Tensor], meta: dict, batch_idx: int) -> list[dict]:
    z = layer["logits"]
    mask = pair_mask(layer).to(device=z.device, dtype=torch.bool)
    z_masked = torch.where(mask, z, torch.full_like(z, -1.0e9))
    attn = torch.softmax(z_masked, dim=-1).masked_fill(~mask, 0.0)
    key_count = mask.sum(dim=-1).to(attn.dtype)
    ent_raw = -(attn * torch.log(attn.clamp_min(1.0e-12))).sum(dim=-1)
    ent_norm = ent_raw / torch.log(key_count.clamp_min(2.0))
    rows = []
    batch_size, heads, num_nodes = ent_norm.shape
    for graph_idx in range(batch_size):
        for head in range(heads):
            for query in range(num_nodes):
                rows.append(
                    {
                        **meta,
                        "batch": batch_idx,
                        "graph_in_batch": graph_idx,
                        "perm": -1,
                        "layer": 0,
                        "head": head,
                        "query_index": query,
                        "metric": M_ENTROPY,
                        "score": float(ent_norm[graph_idx, head, query].detach().cpu()),
                    }
                )
    return rows


def residual_norm_rows(layer: dict[str, torch.Tensor], meta: dict, batch_idx: int) -> list[dict]:
    attn = layer["attn"]
    mask = pair_mask(layer).to(device=attn.device, dtype=torch.bool)
    if mask.size(1) == 1 and attn.size(1) != 1:
        mask = mask.expand(-1, attn.size(1), -1, -1)
    centered = row_center_logits(attn, mask)
    key_count = mask.sum(dim=-1).to(attn.dtype)
    raw_norm = torch.sqrt((centered * centered).sum(dim=-1).clamp_min(0.0))
    max_norm = torch.sqrt((1.0 - 1.0 / key_count.clamp_min(2.0)).clamp_min(1.0e-12))
    norm = torch.where(key_count > 1, raw_norm / max_norm.clamp_min(1.0e-12), torch.zeros_like(raw_norm))
    valid = key_count > 1
    denom = valid.sum(dim=(0, 2)).clamp_min(1).to(norm.dtype)
    head_norm = (norm * valid.to(norm.dtype)).sum(dim=(0, 2)) / denom
    rows = []
    for head in range(int(attn.size(1))):
        rows.append(
            {
                **meta,
                "batch": batch_idx,
                "perm": -1,
                "layer": 0,
                "head": head,
                "metric": M_RESIDUAL_NORM,
                "score": float(head_norm[head].detach().cpu()),
            }
        )
    return rows


def relabel_rows(
    clean: dict[str, torch.Tensor],
    variant: dict[str, torch.Tensor],
    perm_pos: torch.Tensor,
    meta: dict,
    batch_idx: int,
    perm_idx: int,
) -> list[dict]:
    a_ref = transform_pair_reference(clean["attn"], perm_pos)
    ref_mask = transform_pair_reference(pair_mask(clean), perm_pos).to(dtype=torch.bool)
    var_mask = pair_mask(variant).to(device=variant["attn"].device, dtype=torch.bool)
    mask = ref_mask.to(device=variant["attn"].device) & var_mask
    score_q = cosine_by_query_tensor(variant["attn"], a_ref, mask, signed=False)
    valid = mask.any(dim=-1)
    denom = valid.sum(dim=(0, 2)).clamp_min(1).to(score_q.dtype)
    score = (score_q * valid.to(score_q.dtype)).sum(dim=(0, 2)) / denom
    return [
        {
            **meta,
            "batch": batch_idx,
            "perm": perm_idx,
            "layer": 0,
            "head": head,
            "metric": M_RELABEL_EQUIVARIANT,
            "score": float(score[head].detach().cpu()),
        }
        for head in range(int(score.numel()))
    ]


def partition_attention_mass_rows(
    clean: dict[str, torch.Tensor],
    partitions: dict[str, torch.Tensor],
    meta: dict,
    batch_idx: int,
) -> list[dict]:
    attn = clean["attn"]
    rows = []
    for name, mask in partitions.items():
        m = mask.to(device=attn.device, dtype=torch.bool)
        if m.size(1) == 1 and attn.size(1) != 1:
            m = m.expand(-1, attn.size(1), -1, -1)
        active_rows = m.any(dim=-1)
        if not bool(active_rows.any()):
            continue
        mass = (attn * m.to(attn.dtype)).sum(dim=-1)
        denom = active_rows.sum(dim=(0, 2)).clamp_min(1).to(attn.dtype)
        mean_mass = (mass * active_rows.to(attn.dtype)).sum(dim=(0, 2)) / denom
        active_count = m.sum(dim=-1).to(attn.dtype)
        mean_candidates = (
            active_count.sum(dim=(0, 2)) / active_rows.sum(dim=(0, 2)).clamp_min(1).to(attn.dtype)
        )
        for head in range(int(attn.size(1))):
            rows.append(
                {
                    **meta,
                    "batch": batch_idx,
                    "perm": -1,
                    "layer": 0,
                    "head": head,
                    "partition": name,
                    "metric": "attention_mass",
                    "score": float(mean_mass[head].detach().cpu()),
                    "mean_candidates_per_query": float(mean_candidates[head].detach().cpu()),
                }
            )
    return rows


def aggregate_alpha_metric_rows(
    score_lists: dict[str, list[torch.Tensor]],
    moved_masses: list[torch.Tensor],
    node_mask: torch.Tensor,
    meta: dict,
    batch_idx: int,
    alpha_tau: float,
) -> tuple[list[dict], list[dict]]:
    if not moved_masses:
        return [], []
    if alpha_tau <= 0:
        raise ValueError("--metric-alpha-tau must be positive")

    moved = torch.stack(moved_masses, dim=0)
    alpha = torch.softmax(moved / alpha_tau, dim=0)
    alpha_max = alpha.max(dim=0).values
    effective_perms = 1.0 / torch.square(alpha).sum(dim=0).clamp_min(1.0e-12)
    valid_queries = node_mask[:, None, :].to(device=moved.device, dtype=torch.bool)
    valid_queries = valid_queries.expand(-1, moved.size(2), -1)

    rows = []
    query_rows = []
    for metric, values in score_lists.items():
        if not values:
            continue
        scores = torch.stack(values, dim=0)
        weighted_query = (alpha * scores).sum(dim=0)
        denom = valid_queries.sum(dim=-1).clamp_min(1).to(weighted_query.dtype)
        head_score = (weighted_query * valid_queries.to(weighted_query.dtype)).sum(dim=-1) / denom
        moved_mean = (moved.mean(dim=0) * valid_queries.to(moved.dtype)).sum(dim=-1) / denom
        alpha_max_mean = (alpha_max * valid_queries.to(alpha_max.dtype)).sum(dim=-1) / denom
        effective_mean = (effective_perms * valid_queries.to(effective_perms.dtype)).sum(dim=-1) / denom
        batch_size, heads, num_nodes = weighted_query.shape
        for graph_idx in range(batch_size):
            for head in range(heads):
                rows.append(
                    {
                        **meta,
                        "batch": batch_idx,
                        "graph_in_batch": graph_idx,
                        "perm": -1,
                        "layer": 0,
                        "head": head,
                        "metric": metric,
                        "centered": metric in BACKGROUND_REMOVED_METRICS,
                        "score": float(head_score[graph_idx, head].detach().cpu()),
                        "moved_mass_mean": float(moved_mean[graph_idx, head].detach().cpu()),
                        "alpha_max_mean": float(alpha_max_mean[graph_idx, head].detach().cpu()),
                        "effective_perms_mean": float(effective_mean[graph_idx, head].detach().cpu()),
                    }
                )
                for query in range(num_nodes):
                    if not bool(valid_queries[graph_idx, head, query].detach().cpu()):
                        continue
                    query_rows.append(
                        {
                            **meta,
                            "batch": batch_idx,
                            "graph_in_batch": graph_idx,
                            "perm": -1,
                            "layer": 0,
                            "head": head,
                            "query_index": query,
                            "metric": metric,
                            "centered": metric in BACKGROUND_REMOVED_METRICS,
                            "score": float(weighted_query[graph_idx, head, query].detach().cpu()),
                            "moved_mass_mean": float(moved[:, graph_idx, head, query].mean().detach().cpu()),
                            "alpha_max": float(alpha_max[graph_idx, head, query].detach().cpu()),
                            "effective_perms": float(effective_perms[graph_idx, head, query].detach().cpu()),
                        }
                    )
    return rows, query_rows


@torch.no_grad()
def attention_snapshot(
    model: CopyStudent,
    task: str,
    num_graphs: int,
    num_nodes: int,
    key_vocab_size: int,
    value_vocab_size: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    batch = generate_batch(
        task,
        num_graphs,
        num_nodes,
        key_vocab_size,
        value_vocab_size,
        seed,
    ).to(device)
    _, layer = model(batch)
    mask = pair_mask(layer).to(device=device, dtype=torch.bool)
    centered = row_center_logits(layer["attn"], mask)
    return centered.detach().cpu().reshape(centered.size(1), -1).numpy()


def _cosine_np(u: np.ndarray, v: np.ndarray) -> float:
    num = float(np.dot(u, v))
    den = float(np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))
    if den <= 1.0e-12:
        return 0.0
    return num / den


def compute_attention_stability(snapshots: list[dict]) -> list[dict]:
    rows = []
    groups: dict[tuple, list[dict]] = {}
    for item in snapshots:
        meta = item["meta"]
        key = (meta.get("architecture", "copy_attn"), meta.get("num_heads", 1), meta["task"])
        groups.setdefault(key, []).append(item)
    for (_architecture, _num_heads, _task), items in sorted(groups.items()):
        if len(items) < 2:
            continue
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = items[i]
                b = items[j]
                avec = a["vectors"]
                bvec = b["vectors"]
                heads = min(avec.shape[0], bvec.shape[0])
                cos_matrix = np.zeros((heads, heads), dtype=np.float64)
                for ha in range(heads):
                    for hb in range(heads):
                        cos_matrix[ha, hb] = _cosine_np(avec[ha], bvec[hb])
                for head in range(heads):
                    cos = float(cos_matrix[head, head])
                    rows.append(
                        {
                            **a["meta"],
                            "seed_a": a["meta"]["seed"],
                            "seed_b": b["meta"]["seed"],
                            "head": head,
                            "stability_type": "same_head",
                            "metric": "attention_pattern_stability",
                            "score": float(np.clip(0.5 * (cos + 1.0), 0.0, 1.0)),
                            "raw_cos": cos,
                        }
                    )
                best_cos = float(np.mean(np.max(cos_matrix, axis=1)))
                rows.append(
                    {
                        **a["meta"],
                        "seed_a": a["meta"]["seed"],
                        "seed_b": b["meta"]["seed"],
                        "head": -1,
                        "stability_type": "best_head_match",
                        "metric": "attention_pattern_stability",
                        "score": float(np.clip(0.5 * (best_cos + 1.0), 0.0, 1.0)),
                        "raw_cos": best_cos,
                    }
                )
    return rows


@torch.no_grad()
def compute_metrics(
    model: CopyStudent,
    task: str,
    meta: dict,
    num_graphs: int,
    batch_size: int,
    num_nodes: int,
    key_vocab_size: int,
    value_vocab_size: int,
    num_perms: int,
    metric_alpha_tau: float,
    metric_space: str,
    seed: int,
    device: torch.device,
    generation_task: str | None = None,
) -> MetricOutputs:
    model.eval()
    generation_task = canonical_task(generation_task or task)
    rows = []
    query_rows = []
    sanity_rows = []
    partition_rows = []
    attention_rows = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 100_000)
    for batch_idx, offset in enumerate(range(0, num_graphs, batch_size)):
        current = min(batch_size, num_graphs - offset)
        batch = generate_batch(
            generation_task, current, num_nodes, key_vocab_size, value_vocab_size, seed + offset
        ).to(device)
        _, clean = model(batch)
        sanity_rows.extend(entropy_rows(clean, meta, batch_idx))
        sanity_rows.extend(residual_norm_rows(clean, meta, batch_idx))
        sanity_rows.extend(per_query_entropy_rows(clean, meta, batch_idx))
        partitions = partition_masks(batch, generation_task)
        partition_rows.extend(partition_attention_mass_rows(clean, partitions, meta, batch_idx))
        attn = clean["attn"]
        source_idx = batch.source[:, None, :, None].expand(-1, attn.size(1), -1, 1)
        mass = torch.gather(attn, dim=-1, index=source_idx).squeeze(-1)
        rank = (attn > mass[:, :, :, None]).sum(dim=-1).float() + 1.0
        for head in range(int(attn.size(1))):
            attention_rows.append(
                {
                    **meta,
                    "layer": 0,
                    "head": head,
                    "teacher_source_mass": float(mass[:, head].mean().detach().cpu()),
                    "teacher_source_mrr": float((1.0 / rank[:, head]).mean().detach().cpu()),
                }
            )
        score_lists = {
            M_POSITIONAL: [],
            M_SYMBOLIC: [],
            M_PE_INVARIANT: [],
            M_PE_EQUIVARIANT: [],
            M_POSITIONAL_CENTERED: [],
            M_SYMBOLIC_CENTERED: [],
            M_PE_INVARIANT_CENTERED: [],
            M_PE_EQUIVARIANT_CENTERED: [],
            M_INTERACTION_RESIDUAL_CENTERED: [],
            M_JOINT_EQUIVARIANCE_EXCESS_CENTERED: [],
        }
        moved_masses = []
        for perm_idx in range(num_perms):
            perm_pos = make_perm(batch, generator)
            _, var_x = model(x_permuted(batch, perm_pos))
            _, var_pe = model(pe_permuted(batch, perm_pos))
            _, var_both = model(x_pe_permuted(batch, perm_pos))
            _, var_relabel = model(relabel_permuted(batch, perm_pos))
            attn_clean = clean["attn"]
            attn_mask = pair_mask(clean).to(device=attn_clean.device, dtype=torch.bool)
            moved_masses.append(attention_moved_mass_by_query(attn_clean, perm_pos, attn_mask))

            clean_t, clean_mask, signed = metric_tensor_from_layer(clean, metric_space, centered=False)
            clean_ref = transform_pair_reference(clean_t, perm_pos)
            clean_ref_mask = transform_pair_reference(clean_mask, perm_pos).to(dtype=torch.bool)
            var_x_t, _, _ = metric_tensor_from_layer(var_x, metric_space, centered=False)
            var_pe_t, _, _ = metric_tensor_from_layer(var_pe, metric_space, centered=False)
            score_lists[M_POSITIONAL].append(
                cosine_by_query_tensor(var_x_t, clean_t, clean_mask, signed=signed)
            )
            score_lists[M_SYMBOLIC].append(
                cosine_by_query_tensor(var_x_t, clean_ref, clean_ref_mask, signed=signed)
            )
            score_lists[M_PE_INVARIANT].append(
                cosine_by_query_tensor(var_pe_t, clean_t, clean_mask, signed=signed)
            )
            score_lists[M_PE_EQUIVARIANT].append(
                cosine_by_query_tensor(var_pe_t, clean_ref, clean_ref_mask, signed=signed)
            )

            clean_c, clean_c_mask = centered_attention_tensor_from_layer(clean)
            clean_c_ref = transform_pair_reference(clean_c, perm_pos)
            clean_c_ref_mask = transform_pair_reference(clean_c_mask, perm_pos).to(dtype=torch.bool)
            var_x_c, _ = centered_attention_tensor_from_layer(var_x)
            var_pe_c, _ = centered_attention_tensor_from_layer(var_pe)
            var_both_c, _ = centered_attention_tensor_from_layer(var_both)
            positional_c = raw_cosine_by_query_tensor(var_x_c, clean_c, clean_c_mask)
            symbolic_c = raw_cosine_by_query_tensor(var_x_c, clean_c_ref, clean_c_ref_mask)
            pe_invariance_c = raw_cosine_by_query_tensor(var_pe_c, clean_c, clean_c_mask)
            pe_equivariance_c = raw_cosine_by_query_tensor(var_pe_c, clean_c_ref, clean_c_ref_mask)
            score_lists[M_POSITIONAL_CENTERED].append(
                positional_c
            )
            score_lists[M_SYMBOLIC_CENTERED].append(
                symbolic_c
            )
            score_lists[M_PE_INVARIANT_CENTERED].append(
                pe_invariance_c
            )
            score_lists[M_PE_EQUIVARIANT_CENTERED].append(
                pe_equivariance_c
            )
            interaction_mask = clean_c_mask
            score_lists[M_INTERACTION_RESIDUAL_CENTERED].append(
                interaction_residual_norm_by_query_tensor(
                    clean_c, var_x_c, var_pe_c, var_both_c, interaction_mask
                )
            )
            joint_equivariance_c = raw_cosine_by_query_tensor(
                var_both_c, clean_c_ref, clean_c_ref_mask
            )
            best_single_c = torch.stack(
                [positional_c, symbolic_c, pe_invariance_c, pe_equivariance_c],
                dim=0,
            ).max(dim=0).values
            score_lists[M_JOINT_EQUIVARIANCE_EXCESS_CENTERED].append(
                joint_equivariance_c - best_single_c
            )

            sanity_rows.extend(relabel_rows(clean, var_relabel, perm_pos, meta, batch_idx, perm_idx))
        batch_rows, batch_query_rows = aggregate_alpha_metric_rows(
            score_lists,
            moved_masses,
            clean["node_mask"],
            meta,
            batch_idx,
            metric_alpha_tau,
        )
        rows.extend(batch_rows)
        query_rows.extend(batch_query_rows)
    return MetricOutputs(
        metric_rows=rows,
        query_rows=query_rows,
        sanity_rows=sanity_rows,
        partition_rows=partition_rows,
        attention_rows=attention_rows,
    )


DEFAULT_GROUP_FIELDS = (
    "architecture",
    "num_heads",
    "seed",
    "task",
    "layer",
    "head",
    "metric",
)


def summarize_rows(rows: list[dict], group_fields: Iterable[str] = DEFAULT_GROUP_FIELDS) -> list[dict]:
    buckets: dict[tuple, list[float]] = {}
    meta_by_key: dict[tuple, dict] = {}
    for row in rows:
        fields = tuple(field for field in group_fields if field in row)
        key = tuple(row.get(field) for field in fields)
        buckets.setdefault(key, []).append(float(row["score"]))
        meta_by_key[key] = {field: row.get(field) for field in fields}
    out = []
    for key, values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=np.float64)
        out.append(
            {
                **meta_by_key[key],
                "score_mean": float(arr.mean()),
                "score_std": float(arr.std(ddof=0)),
                "n": int(arr.size),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─── visualisation constants ──────────────────────────────────────────────────

TASK_COLORS: dict[str, str] = {
    "symbolic_equality":  "#4C72B0",
    "path_predecessor":   "#DD8452",
    "tree_parent":        "#55A868",
    "mirror_node":        "#C44E52",
    "hop_then_key":       "#8172B3",
    "parent_then_key":    "#64B5CD",
    "ring_quad_key":      "#DA8BC3",
    "previous_same_key":  "#937860",
}

METRIC_DISPLAY: dict[str, str] = {
    M_POSITIONAL:          "Positional",
    M_SYMBOLIC:            "Symbolic",
    M_PE_INVARIANT:        "PE Invariance",
    M_PE_EQUIVARIANT:      "PE Equivariance",
    M_POSITIONAL_CENTERED: "Positional (ctr)",
    M_SYMBOLIC_CENTERED:   "Symbolic (ctr)",
    M_PE_INVARIANT_CENTERED: "PE Invariance (ctr)",
    M_PE_EQUIVARIANT_CENTERED: "PE Equivariance (ctr)",
    M_INTERACTION_RESIDUAL_CENTERED: "Interaction Residual (ctr)",
    M_JOINT_EQUIVARIANCE_EXCESS_CENTERED: "Joint Eq. Excess (ctr)",
    M_ENTROPY:             "Entropy (norm)",
    M_RESIDUAL_NORM:       "Residual Norm",
    M_RELABEL_EQUIVARIANT: "Relabel Equivariance",
    "attention_mass":      "Attention Mass",
}

TASK_SHORT: dict[str, str] = {
    "symbolic_equality":  "sym_eq",
    "path_predecessor":   "path_pred",
    "tree_parent":        "tree_par",
    "mirror_node":        "mirror",
    "hop_then_key":       "hop_key",
    "parent_then_key":    "par_key",
    "ring_quad_key":      "quad_key",
    "previous_same_key":  "prev_key",
}

ALL_METRICS_ORDERED = [
    M_POSITIONAL, M_SYMBOLIC,
    M_PE_INVARIANT, M_PE_EQUIVARIANT,
]

CENTERED_METRICS_ORDERED = [
    M_POSITIONAL_CENTERED, M_SYMBOLIC_CENTERED,
    M_PE_INVARIANT_CENTERED, M_PE_EQUIVARIANT_CENTERED,
]

MIXED_METRICS_ORDERED = [
    M_INTERACTION_RESIDUAL_CENTERED,
    M_JOINT_EQUIVARIANCE_EXCESS_CENTERED,
]

BACKGROUND_REMOVED_METRICS = set(CENTERED_METRICS_ORDERED) | set(MIXED_METRICS_ORDERED)


def _vis_style() -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.6,
    })


def _task_color(task: str) -> str:
    task = str(task)
    if task in TASK_COLORS:
        return TASK_COLORS[task]
    for full_name, short_name in TASK_SHORT.items():
        if task == short_name or task.startswith(f"{short_name} "):
            return TASK_COLORS.get(full_name, "#888888")
    return "#888888"


def _run_label(row: dict) -> str:
    task = str(row.get("task", "task"))
    label = TASK_SHORT.get(task, task)
    parts = [label]
    num_heads = row.get("num_heads")
    if num_heads is not None:
        try:
            parts.append(f"h{int(num_heads)}")
        except (TypeError, ValueError):
            parts.append(f"h{num_heads}")
    seed = row.get("seed")
    if seed is not None:
        try:
            parts.append(f"s{int(seed)}")
        except (TypeError, ValueError):
            parts.append(f"s{seed}")
    arch = str(row.get("architecture", ""))
    if arch and arch not in {"copy_attn", "oracle_attention"}:
        parts.append(arch.replace("_oracle_student", "+student"))
    return " ".join(parts)


def _run_sort_key(label: str) -> tuple:
    for idx, task in enumerate(TASKS):
        short = TASK_SHORT.get(task, task)
        if label == short or label.startswith(f"{short} "):
            return (idx, label)
    return (len(TASKS), label)


def _cleanup_plot_outputs(out_dir: Path, patterns: Iterable[str]) -> None:
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def _pivot_metric_means(rows: list[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        buckets.setdefault((_run_label(row), str(row["metric"])), []).append(float(row["score_mean"]))
    out: dict[str, dict[str, float]] = {}
    for (run, metric), values in buckets.items():
        out.setdefault(run, {})[metric] = float(np.mean(values))
    return out


def _pivot_metric_stds(rows: list[dict]) -> dict[str, dict[str, float]]:
    buckets: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        buckets.setdefault((_run_label(row), str(row["metric"])), []).append(float(row.get("score_std", 0.0)))
    out: dict[str, dict[str, float]] = {}
    for (run, metric), values in buckets.items():
        out.setdefault(run, {})[metric] = float(np.sqrt(np.mean(np.square(values))))
    return out


def plot_metric_heatmap(
    summary_rows: list[dict],
    out_dir: Path,
    filename: str = "heatmap_all_metrics.png",
    title: str = "Alpha-Weighted Global Permutation Scores per Run",
    metrics: list[str] | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name: str = "RdYlGn",
) -> None:
    if not summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    by_task = _pivot_metric_means(summary_rows)
    tasks = sorted(by_task, key=_run_sort_key)
    if metrics is None:
        metrics = list(ALL_METRICS_ORDERED)
    n_tasks, n_metrics = len(tasks), len(metrics)

    data = np.full((n_tasks, n_metrics), np.nan)
    for i, task in enumerate(tasks):
        for j, metric in enumerate(metrics):
            if metric in by_task[task]:
                data[i, j] = by_task[task][metric]

    fig, ax = plt.subplots(figsize=(n_metrics * 1.35 + 1.2, max(2.5, n_tasks * 0.65 + 1.8)))
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#f0f0f0")
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(
        [METRIC_DISPLAY.get(m, m) for m in metrics],
        rotation=35, ha="right", fontsize=8.5,
    )
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(tasks, fontsize=9)

    for i in range(n_tasks):
        for j in range(n_metrics):
            v = data[i, j]
            if not np.isnan(v):
                text_col = "white" if v < 0.28 or v > 0.82 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=text_col)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="score")
    if n_metrics > 2:
        for sep in range(2, n_metrics, 2):
            ax.axvline(sep - 0.5, color="white", linewidth=2.2)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_planes(summary_rows: list[dict], out_dir: Path) -> None:
    if not summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    by_task = _pivot_metric_means(summary_rows)

    plane_specs = [
        (M_POSITIONAL,  M_SYMBOLIC,        "Positional",    "Symbolic",        "Symbolic / Positional"),
        (M_PE_INVARIANT, M_PE_EQUIVARIANT,  "PE Invariance", "PE Equivariance", "PE Invariance / Equivariance"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, (xm, ym, xl, yl, title) in zip(axes, plane_specs):
        ax.plot([0, 1], [0, 1], color="#dddddd", linewidth=0.9, zorder=0)
        for task, metrics in sorted(by_task.items(), key=lambda kv: _run_sort_key(kv[0])):
            if xm not in metrics or ym not in metrics:
                continue
            xv, yv = metrics[xm], metrics[ym]
            ax.scatter(xv, yv, s=130, color=_task_color(task), zorder=3,
                       edgecolors="white", linewidths=0.8, label=task)
            ax.annotate(task, (xv, yv),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=7.5, color=_task_color(task))
        ax.set_xlim(-0.04, 1.06)
        ax.set_ylim(-0.04, 1.06)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    fig.tight_layout(pad=2.0)
    fig.savefig(out_dir / "scatter_planes.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_centered_scatter_planes(summary_rows: list[dict], out_dir: Path) -> None:
    if not summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    by_task = _pivot_metric_means(summary_rows)
    plane_specs = [
        (
            M_POSITIONAL_CENTERED,
            M_SYMBOLIC_CENTERED,
            "Positional (centered)",
            "Symbolic (centered)",
            "Centered Symbolic / Positional",
        ),
        (
            M_PE_INVARIANT_CENTERED,
            M_PE_EQUIVARIANT_CENTERED,
            "PE Invariance (centered)",
            "PE Equivariance (centered)",
            "Centered PE Invariance / Equivariance",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (xm, ym, xl, yl, title) in zip(axes, plane_specs):
        ax.plot([-1, 1], [-1, 1], color="#dddddd", linewidth=0.9, zorder=0)
        ax.axhline(0.0, color="#eeeeee", linewidth=0.8, zorder=0)
        ax.axvline(0.0, color="#eeeeee", linewidth=0.8, zorder=0)
        for task, metrics in sorted(by_task.items(), key=lambda kv: _run_sort_key(kv[0])):
            if xm not in metrics or ym not in metrics:
                continue
            xv, yv = metrics[xm], metrics[ym]
            ax.scatter(xv, yv, s=130, color=_task_color(task), zorder=3,
                       edgecolors="white", linewidths=0.8, label=task)
            ax.annotate(task, (xv, yv),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=7.5, color=_task_color(task))
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(xl, fontsize=9)
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(pad=2.0)
    fig.savefig(out_dir / "scatter_planes_centered.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_mixed_metric_summary(summary_rows: list[dict], out_dir: Path) -> None:
    if not summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    _cleanup_plot_outputs(
        out_dir,
        [
            "heatmap_interaction_residual_centered.png",
            "heatmap_joint_equivariance_excess_centered.png",
        ],
    )
    by_run = _pivot_metric_means(summary_rows)
    runs = sorted(by_run, key=_run_sort_key)
    specs = [
        (
            M_INTERACTION_RESIDUAL_CENTERED,
            "Centered Interaction Residual Norm",
            0.0,
            1.0,
            "magma",
        ),
        (
            M_JOINT_EQUIVARIANCE_EXCESS_CENTERED,
            "Centered Joint Equivariance Excess",
            -1.0,
            1.0,
            "coolwarm",
        ),
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(8.5, len(runs) * 0.45 + 4.0), 3.8),
        squeeze=False,
    )
    for ax, (metric, title, vmin, vmax, cmap_name) in zip(axes[0], specs):
        data = np.asarray([[by_run.get(run, {}).get(metric, np.nan)] for run in runs])
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#f0f0f0")
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks([0])
        ax.set_xticklabels([METRIC_DISPLAY.get(metric, metric)], rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels(runs, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for i, value in enumerate(data[:, 0]):
            if not np.isnan(value):
                ax.text(0, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, label="score")
    fig.suptitle("Mixed Metric Comparison Across Runs", fontsize=11, fontweight="bold")
    fig.tight_layout(pad=1.5)
    fig.savefig(out_dir / "mixed_centered_metric_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(summary_rows: list[dict], out_dir: Path) -> None:
    if not summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    by_task = _pivot_metric_means(summary_rows)
    stds    = _pivot_metric_stds(summary_rows)
    tasks   = sorted(by_task, key=_run_sort_key)

    ncols = 4
    nrows = math.ceil(len(ALL_METRICS_ORDERED) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.8), squeeze=False)

    for idx, metric in enumerate(ALL_METRICS_ORDERED):
        ax = axes[idx // ncols][idx % ncols]
        vals = [by_task.get(t, {}).get(metric, np.nan) for t in tasks]
        errs = [stds.get(t, {}).get(metric, 0.0) for t in tasks]
        x    = np.arange(len(tasks))
        ax.bar(x, [v if not np.isnan(v) else 0.0 for v in vals],
               color=[_task_color(t) for t in tasks], width=0.6, zorder=2)
        ax.errorbar(x, [v if not np.isnan(v) else 0.0 for v in vals],
                    yerr=errs, fmt="none", color="#333", linewidth=1.0, capsize=3, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=40, ha="right", fontsize=7)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(METRIC_DISPLAY.get(metric, metric), fontsize=9, fontweight="bold")

    for idx in range(len(ALL_METRICS_ORDERED), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Per-Metric Scores Across Runs", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout(pad=1.5)
    fig.savefig(out_dir / "metric_bars_all.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_radar(summary_rows: list[dict], out_dir: Path) -> None:
    if not summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    by_task = _pivot_metric_means(summary_rows)
    tasks   = sorted(by_task, key=_run_sort_key)
    N = len(ALL_METRICS_ORDERED)
    angles = [n / N * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_DISPLAY.get(m, m) for m in ALL_METRICS_ORDERED], size=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=6.5, color="#888")

    for task in tasks:
        vals = [by_task.get(task, {}).get(m, 0.0) for m in ALL_METRICS_ORDERED]
        vals += vals[:1]
        c = _task_color(task)
        ax.plot(angles, vals, linewidth=1.8, color=c, label=task)
        ax.fill(angles, vals, color=c, alpha=0.07)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), frameon=False, fontsize=8)
    ax.set_title("Metric Profiles per Run", fontsize=11, fontweight="bold", pad=18)
    fig.tight_layout()
    _cleanup_plot_outputs(out_dir, ["radar_all_tasks.png"])
    fig.savefig(out_dir / "radar_all_runs.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_comparison(summaries: list[dict], out_dir: Path) -> None:
    if not summaries:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    tasks   = [_run_label(s) for s in summaries]
    val_acc = [float(s.get("best_val_acc", 0.0)) for s in summaries]
    id_acc  = [float(s.get("id_acc",  0.0)) for s in summaries]
    ood_acc = [float(s.get("ood_acc", 0.0)) for s in summaries]
    id_mrr  = [float(s.get("id_teacher_source_mrr",  0.0)) for s in summaries]
    id_mass = [float(s.get("id_teacher_source_mass", 0.0)) for s in summaries]

    x = np.arange(len(tasks))
    w = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(max(10, len(tasks) * 1.6), 4.0))

    ax = axes[0]
    ax.bar(x - w, val_acc, w, label="Val",      color="#4C72B0", alpha=0.85, zorder=2)
    ax.bar(x,     id_acc,  w, label="ID Test",  color="#55A868", alpha=0.85, zorder=2)
    ax.bar(x + w, ood_acc, w, label="OOD Test", color="#C44E52", alpha=0.85, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_title("Accuracy  (Val / ID / OOD)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(1.0, color="#aaa", linewidth=0.7, linestyle="--")

    ax = axes[1]
    ax.bar(x, id_mrr, color=[_task_color(t) for t in tasks], width=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_title("ID Teacher Source MRR", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Reciprocal Rank")
    ax.axhline(1.0, color="#aaa", linewidth=0.7, linestyle="--")

    ax = axes[2]
    ax.bar(x, id_mass, color=[_task_color(t) for t in tasks], width=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_title("ID Teacher Source Mass", fontsize=10, fontweight="bold")
    ax.set_ylabel("Attention Mass on Teacher Target")
    ax.axhline(1.0, color="#aaa", linewidth=0.7, linestyle="--")

    fig.suptitle("Run Performance Summary", fontsize=11, fontweight="bold")
    fig.tight_layout(pad=2.0)
    fig.savefig(out_dir / "accuracy_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(all_log_rows: list[dict], out_dir: Path) -> None:
    if not all_log_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    by_task: dict[str, list[dict]] = {}
    for row in all_log_rows:
        by_task.setdefault(_run_label(row), []).append(row)

    tasks   = sorted(by_task, key=_run_sort_key)
    n_tasks = len(tasks)
    if n_tasks == 0:
        return

    fig, axes = plt.subplots(n_tasks, 2, figsize=(9.0, n_tasks * 2.5 + 0.8), squeeze=False)

    for i, task in enumerate(tasks):
        log        = sorted(by_task[task], key=lambda r: int(r["epoch"]))
        epochs     = [int(r["epoch"])        for r in log]
        train_loss = [float(r["train_loss"]) for r in log]
        val_loss   = [float(r["val_loss"])   for r in log]
        train_acc  = [float(r["train_acc"])  for r in log]
        val_acc    = [float(r["val_acc"])    for r in log]
        c = _task_color(task)

        ax_l = axes[i, 0]
        ax_l.plot(epochs, train_loss, color=c, linewidth=1.5, label="train")
        ax_l.plot(epochs, val_loss,   color=c, linewidth=1.5, linestyle="--", label="val")
        ax_l.set_ylabel("Loss", fontsize=8)
        ax_l.set_title(f"{task} - Loss", fontsize=8.5, fontweight="bold")
        ax_l.legend(frameon=False, fontsize=7)

        ax_a = axes[i, 1]
        ax_a.plot(epochs, train_acc, color=c, linewidth=1.5, label="train")
        ax_a.plot(epochs, val_acc,   color=c, linewidth=1.5, linestyle="--", label="val")
        ax_a.set_ylim(0, 1.06)
        ax_a.set_ylabel("Accuracy", fontsize=8)
        ax_a.set_title(f"{task} - Accuracy", fontsize=8.5, fontweight="bold")
        ax_a.legend(frameon=False, fontsize=7)

        if i == n_tasks - 1:
            ax_l.set_xlabel("Epoch", fontsize=8)
            ax_a.set_xlabel("Epoch", fontsize=8)

    fig.suptitle("Training Curves", fontsize=11, fontweight="bold")
    fig.tight_layout(pad=1.5)
    fig.savefig(out_dir / "training_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _aggregate_summary(
    rows: list[dict],
    fields: tuple[str, ...],
    value_key: str = "score_mean",
) -> list[dict]:
    buckets: dict[tuple, list[float]] = {}
    meta: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        buckets.setdefault(key, []).append(float(row.get(value_key, row.get("score", 0.0))))
        meta[key] = {field: row.get(field) for field in fields}
    out = []
    for key, values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=np.float64)
        out.append({**meta[key], "score_mean": float(arr.mean()), "score_std": float(arr.std(ddof=0)), "n": int(arr.size)})
    return out


def plot_per_query_metrics(
    query_summary_rows: list[dict],
    out_dir: Path,
    metrics: list[str] | None = None,
    filename_prefix: str = "per_query_metrics",
    title_prefix: str = "Per-Query Metrics",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap_name: str = "RdYlGn",
) -> None:
    if not query_summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    rows = _aggregate_summary(
        query_summary_rows,
        ("architecture", "num_heads", "seed", "task", "query_index", "metric"),
    )
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(_run_label(row), []).append(row)
    if metrics is None:
        metrics = list(ALL_METRICS_ORDERED)
    runs = sorted(by_task, key=_run_sort_key)
    if not runs:
        return
    _cleanup_plot_outputs(out_dir, [f"{filename_prefix}_*.png"])
    ncols = min(3, len(runs))
    nrows = math.ceil(len(runs) / ncols)
    max_queries = max(
        len({int(row["query_index"]) for row in by_task[run] if row.get("query_index") is not None})
        for run in runs
    )
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(4.4 * ncols, max_queries * 0.36 * ncols), max(3.0 * nrows, len(metrics) * 0.24 * nrows + 1.0)),
        squeeze=False,
    )
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#f0f0f0")
    im = None
    for idx, task in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        task_rows = by_task[task]
        queries = sorted({int(row["query_index"]) for row in task_rows if row.get("query_index") is not None})
        if not queries:
            continue
        data = np.full((len(metrics), len(queries)), np.nan)
        row_map = {
            (int(row["query_index"]), str(row["metric"])): float(row["score_mean"])
            for row in task_rows
            if row.get("query_index") is not None
        }
        for i, metric in enumerate(metrics):
            for j, query in enumerate(queries):
                if (query, metric) in row_map:
                    data[i, j] = row_map[(query, metric)]
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(queries)))
        ax.set_xticklabels([str(q) for q in queries], fontsize=6.5)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([METRIC_DISPLAY.get(m, m) for m in metrics], fontsize=7)
        ax.set_xlabel("Query index")
        ax.set_title(task, fontsize=9, fontweight="bold", color=_task_color(task))
    for idx in range(len(runs), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="score")
    fig.suptitle(title_prefix, fontsize=11, fontweight="bold")
    fig.savefig(out_dir / f"{filename_prefix}_all_runs.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_partitioned_metrics(partition_summary_rows: list[dict], out_dir: Path) -> None:
    if not partition_summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    rows = _aggregate_summary(
        partition_summary_rows,
        ("architecture", "num_heads", "seed", "task", "partition", "metric"),
    )
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("partition") is None:
            continue
        by_task.setdefault(_run_label(row), []).append(row)
    metrics = ["attention_mass"]
    metric_colors = {
        M_POSITIONAL: "#4C72B0",
        M_SYMBOLIC: "#55A868",
        M_PE_INVARIANT: "#DD8452",
        M_PE_EQUIVARIANT: "#8172B3",
        "attention_mass": "#333333",
    }
    runs = sorted(by_task, key=_run_sort_key)
    if not runs:
        return
    _cleanup_plot_outputs(out_dir, ["partitioned_metrics_*.png"])
    ncols = min(3, len(runs))
    nrows = math.ceil(len(runs) / ncols)
    max_partitions = max(len({str(row["partition"]) for row in by_task[run]}) for run in runs)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(4.2 * ncols, max_partitions * 0.8 * ncols), 3.3 * nrows),
        squeeze=False,
    )
    for idx, task in enumerate(runs):
        ax = axes[idx // ncols][idx % ncols]
        task_rows = by_task[task]
        partitions = sorted({str(row["partition"]) for row in task_rows})
        if not partitions:
            continue
        x = np.arange(len(partitions))
        width = 0.6
        for idx, metric in enumerate(metrics):
            vals = []
            for partition in partitions:
                match = [
                    float(row["score_mean"])
                    for row in task_rows
                    if str(row["partition"]) == partition and str(row["metric"]) == metric
                ]
                vals.append(float(np.mean(match)) if match else np.nan)
            xpos = x - 0.4 + width / 2 + idx * width
            ax.bar(
                xpos,
                [0.0 if np.isnan(v) else v for v in vals],
                width=width,
                color=metric_colors.get(metric, "#888"),
                label=METRIC_DISPLAY.get(metric, metric),
                zorder=2,
            )
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels(partitions, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("attention mass")
        ax.set_title(task, fontsize=9, fontweight="bold", color=_task_color(task))
        ax.legend(frameon=False, fontsize=7.5)
    for idx in range(len(runs), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Partitioned Attention Mass Across Runs", fontsize=11, fontweight="bold")
    fig.tight_layout(pad=1.5)
    fig.savefig(out_dir / "partitioned_metrics_all_runs.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_multihead_metrics(summary_rows: list[dict], out_dir: Path) -> None:
    if not summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    rows = _aggregate_summary(
        summary_rows,
        ("architecture", "seed", "task", "num_heads", "head", "metric"),
    )
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(_run_label(row), []).append(row)
    metrics = [M_POSITIONAL, M_SYMBOLIC, M_PE_INVARIANT, M_PE_EQUIVARIANT]
    runs = sorted(by_task, key=_run_sort_key)
    if not runs:
        return
    _cleanup_plot_outputs(out_dir, ["multihead_metrics_*.png"])
    plottable = []
    for task in runs:
        task_rows = by_task[task]
        head_labels = sorted({
            (int(row.get("num_heads", 1)), int(row.get("head", 0)))
            for row in task_rows
            if row.get("head") is not None
        })
        if len(head_labels) <= 1:
            continue
        plottable.append((task, task_rows, head_labels))
    if not plottable:
        return
    ncols = min(3, len(plottable))
    nrows = math.ceil(len(plottable) / ncols)
    max_heads = max(len(head_labels) for _task, _rows, head_labels in plottable)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(4.2 * ncols, max_heads * 0.55 * ncols), 3.0 * nrows),
        squeeze=False,
    )
    im = None
    for idx, (task, task_rows, head_labels) in enumerate(plottable):
        ax = axes[idx // ncols][idx % ncols]
        data = np.full((len(metrics), len(head_labels)), np.nan)
        for i, metric in enumerate(metrics):
            for j, (num_heads, head) in enumerate(head_labels):
                vals = [
                    float(row["score_mean"])
                    for row in task_rows
                    if str(row["metric"]) == metric
                    and int(row.get("num_heads", 1)) == num_heads
                    and int(row.get("head", 0)) == head
                ]
                if vals:
                    data[i, j] = float(np.mean(vals))
        im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(head_labels)))
        ax.set_xticklabels([f"H{h}/{nh}" for nh, h in head_labels], rotation=35, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([METRIC_DISPLAY.get(m, m) for m in metrics], fontsize=8)
        ax.set_title(task, fontsize=9, fontweight="bold", color=_task_color(task))
    for idx in range(len(plottable), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="score")
    fig.suptitle("Multi-Head Metric Profiles Across Runs", fontsize=11, fontweight="bold")
    fig.savefig(out_dir / "multihead_metrics_all_runs.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_architecture_metrics(summary_rows: list[dict], out_dir: Path) -> None:
    if not summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    rows = _aggregate_summary(summary_rows, ("architecture", "task", "metric"))
    architectures = sorted({str(row.get("architecture", "copy_attn")) for row in rows})
    if len(architectures) <= 1:
        return
    tasks = sorted({str(row["task"]) for row in rows})
    metrics = [M_POSITIONAL, M_SYMBOLIC, M_PE_INVARIANT, M_PE_EQUIVARIANT]
    _cleanup_plot_outputs(out_dir, ["architecture_*.png"])
    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(8.0, len(architectures) * 1.2 * ncols), max(3.0, len(tasks) * 0.45 * nrows + 2.0)),
        squeeze=False,
    )
    im = None
    for idx, metric in enumerate(metrics):
        ax = axes[idx // ncols][idx % ncols]
        data = np.full((len(tasks), len(architectures)), np.nan)
        for i, task in enumerate(tasks):
            for j, arch in enumerate(architectures):
                vals = [
                    float(row["score_mean"])
                    for row in rows
                    if str(row.get("architecture", "copy_attn")) == arch
                    and str(row["task"]) == task
                    and str(row["metric"]) == metric
                ]
                if vals:
                    data[i, j] = float(np.mean(vals))
        im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(architectures)))
        ax.set_xticklabels(architectures, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels([TASK_SHORT.get(t, t) for t in tasks], fontsize=8)
        ax.set_title(f"Architecture Dependence: {METRIC_DISPLAY.get(metric, metric)}", fontsize=10, fontweight="bold")
    for idx in range(len(metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="score")
    fig.suptitle("Architecture Metric Comparison Across Tasks", fontsize=11, fontweight="bold")
    fig.savefig(out_dir / "architecture_metrics_all.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_stability_metrics(stability_summary_rows: list[dict], out_dir: Path) -> None:
    if not stability_summary_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _vis_style()
    rows = _aggregate_summary(stability_summary_rows, ("task", "num_heads", "stability_type"))
    tasks = sorted({str(row["task"]) for row in rows})
    if not tasks:
        return
    types = ["same_head", "best_head_match"]
    head_counts = sorted({int(row.get("num_heads", 1)) for row in rows})
    fig, axes = plt.subplots(1, len(head_counts), figsize=(max(5.5, 3.4 * len(head_counts)), 3.8), squeeze=False)
    for ax, num_heads in zip(axes[0], head_counts):
        x = np.arange(len(tasks))
        width = 0.34
        for idx, stability_type in enumerate(types):
            vals = []
            for task in tasks:
                match = [
                    float(row["score_mean"])
                    for row in rows
                    if str(row["task"]) == task
                    and int(row.get("num_heads", 1)) == num_heads
                    and str(row.get("stability_type")) == stability_type
                ]
                vals.append(float(np.mean(match)) if match else np.nan)
            ax.bar(
                x + (idx - 0.5) * width,
                [0.0 if np.isnan(v) else v for v in vals],
                width=width,
                label=stability_type.replace("_", " "),
                zorder=2,
            )
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_SHORT.get(t, t) for t in tasks], rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{num_heads} head{'s' if num_heads != 1 else ''}", fontsize=9, fontweight="bold")
        ax.axhline(0.5, color="#cccccc", linewidth=0.7, linestyle="--", zorder=1)
        ax.set_ylabel("stability score")
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Attention Pattern Stability Across Seeds", fontsize=11, fontweight="bold")
    fig.tight_layout(pad=1.5)
    fig.savefig(out_dir / "attention_pattern_stability.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_all_metrics(
    summary_rows: list[dict],
    summaries: list[dict],
    all_log_rows: list[dict],
    query_summary_rows: list[dict],
    partition_summary_rows: list[dict],
    stability_summary_rows: list[dict],
    out_dir: Path,
) -> None:
    plot_metric_heatmap(summary_rows, out_dir)
    plot_metric_heatmap(
        summary_rows,
        out_dir,
        filename="heatmap_centered_metrics.png",
        title="Alpha-Weighted Centered Attention Scores per Run",
        metrics=list(CENTERED_METRICS_ORDERED),
        vmin=-1.0,
        vmax=1.0,
        cmap_name="coolwarm",
    )
    plot_scatter_planes(summary_rows, out_dir)
    plot_centered_scatter_planes(summary_rows, out_dir)
    plot_mixed_metric_summary(summary_rows, out_dir)
    plot_metric_bars(summary_rows, out_dir)
    plot_radar(summary_rows, out_dir)
    plot_accuracy_comparison(summaries, out_dir)
    plot_training_curves(all_log_rows, out_dir)
    plot_per_query_metrics(query_summary_rows, out_dir)
    plot_per_query_metrics(
        query_summary_rows,
        out_dir,
        metrics=list(CENTERED_METRICS_ORDERED),
        filename_prefix="per_query_centered_metrics",
        title_prefix="Per-Query Centered Metrics",
        vmin=-1.0,
        vmax=1.0,
        cmap_name="coolwarm",
    )
    plot_per_query_metrics(
        query_summary_rows,
        out_dir,
        metrics=[M_INTERACTION_RESIDUAL_CENTERED],
        filename_prefix="per_query_interaction_residual_centered",
        title_prefix="Per-Query Centered Interaction Residual",
        vmin=0.0,
        vmax=1.0,
        cmap_name="magma",
    )
    plot_per_query_metrics(
        query_summary_rows,
        out_dir,
        metrics=[M_JOINT_EQUIVARIANCE_EXCESS_CENTERED],
        filename_prefix="per_query_joint_equivariance_excess_centered",
        title_prefix="Per-Query Centered Joint Equivariance Excess",
        vmin=-1.0,
        vmax=1.0,
        cmap_name="coolwarm",
    )
    plot_partitioned_metrics(partition_summary_rows, out_dir)
    plot_multihead_metrics(summary_rows, out_dir)
    plot_architecture_metrics(summary_rows, out_dir)
    plot_stability_metrics(stability_summary_rows, out_dir)


def plot_planes(summary_rows: list[dict], out_dir: Path) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_task = _pivot_metric_means(summary_rows)
    specs = [
        ("x", M_POSITIONAL, M_SYMBOLIC, "positional_score", "symbolic_score", 0.0, 1.0),
        ("pe", M_PE_INVARIANT, M_PE_EQUIVARIANT, "pe_invariance", "pe_equivariance", 0.0, 1.0),
        (
            "x_centered",
            M_POSITIONAL_CENTERED,
            M_SYMBOLIC_CENTERED,
            "positional_score_centered",
            "symbolic_score_centered",
            -1.0,
            1.0,
        ),
        (
            "pe_centered",
            M_PE_INVARIANT_CENTERED,
            M_PE_EQUIVARIANT_CENTERED,
            "pe_invariance_centered",
            "pe_equivariance_centered",
            -1.0,
            1.0,
        ),
        (
            "mixed_centered",
            M_INTERACTION_RESIDUAL_CENTERED,
            M_JOINT_EQUIVARIANCE_EXCESS_CENTERED,
            "interaction_residual_norm_centered",
            "joint_equivariance_excess_centered",
            -1.0,
            1.0,
        ),
    ]
    _cleanup_plot_outputs(out_dir, ["suite_permutation_plane_*.png"])
    ncols = 3
    nrows = math.ceil(len(specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.4 * nrows), squeeze=False)
    for idx, (suffix, x_metric, y_metric, x_label, y_label, lo, hi) in enumerate(specs):
        ax = axes[idx // ncols][idx % ncols]
        for task, metrics in sorted(by_task.items(), key=lambda kv: _run_sort_key(kv[0])):
            if x_metric not in metrics or y_metric not in metrics:
                continue
            ax.scatter(metrics[x_metric], metrics[y_metric], s=80, label=task, color=_task_color(task))
            ax.text(metrics[x_metric] + 0.01, metrics[y_metric] + 0.01, task, fontsize=8)
        ax.plot([lo, hi], [lo, hi], color="#dddddd", linewidth=0.8)
        if lo < 0:
            ax.axhline(0.0, color="#eeeeee", linewidth=0.8)
            ax.axvline(0.0, color="#eeeeee", linewidth=0.8)
        pad = 0.02 if lo >= 0 else 0.05
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Calibration {suffix}-permutation plane")
        ax.grid(color="#f2f2f2", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8)
    for idx in range(len(specs), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    fig.suptitle("Suite Permutation Plane Comparison Across Runs", fontsize=11, fontweight="bold")
    fig.tight_layout(pad=1.5)
    fig.savefig(out_dir / "suite_permutation_planes.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def resolve_output_root(args: argparse.Namespace) -> Path:
    in_colab = "google.colab" in sys.modules
    if in_colab and not args.no_mount_drive:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount("/content/drive", force_remount=False)
    if args.output_dir is not None:
        return Path(args.output_dir)
    if in_colab:
        root = Path(args.drive_output_root)
        return root if root.is_absolute() else Path("/content/drive") / root
    return Path("experiments/synthetic/results/teacher_head_calibration_graphgps")


def train_one_task(
    task: str,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    run_seed: int,
    num_heads: int,
    architecture: str,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    task = canonical_task(task)
    if architecture != "copy_attn":
        raise NotImplementedError(
            f"Synthetic calibration architecture '{architecture}' is not implemented in this runner yet."
        )
    task_dir = out_dir / task / f"{architecture}_h{num_heads}_seed{run_seed}"
    task_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "architecture": architecture,
        "num_heads": num_heads,
        "seed": run_seed,
        "task": task,
    }
    model = CopyStudent(
        hidden_dim=args.hidden_dim,
        num_heads=num_heads,
        key_vocab_size=args.key_vocab_size,
        value_vocab_size=args.value_vocab_size,
        pe_dim=structural_pe(args.num_nodes).shape[1],
        fixed_value_embeddings=args.fixed_value_embeddings,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = -1.0
    bad_epochs = 0
    log_rows = []
    steps_per_epoch = max(1, math.ceil(args.train_graphs_per_epoch / args.batch_size))
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc_total = 0
        train_n = 0
        for step in range(steps_per_epoch):
            current = min(args.batch_size, args.train_graphs_per_epoch - step * args.batch_size)
            batch = generate_batch(
                task,
                current,
                args.num_nodes,
                args.key_vocab_size,
                args.value_vocab_size,
                run_seed + epoch * 10_000 + step,
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_logits, layer = model(batch)
            loss, value_loss, attention_loss = batch_loss(
                pred_logits, layer, batch, args.attention_loss_weight, args.attention_loss_mode
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * current
            train_acc_total += int((pred_logits.argmax(dim=-1) == batch.target_y).sum().item())
            train_n += int(batch.target_y.numel())
        if epoch % args.eval_every != 0 and epoch != args.max_epochs:
            continue
        val = evaluate(
            model,
            task,
            args.val_graphs,
            args.eval_batch_size,
            args.num_nodes,
            args.key_vocab_size,
            args.value_vocab_size,
            run_seed + 20_000,
            device,
        )
        row = {
            **meta,
            "task": task,
            "epoch": epoch,
            "train_loss": train_loss / max(1, args.train_graphs_per_epoch),
            "train_acc": train_acc_total / max(1, train_n),
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_teacher_source_mass": val["teacher_source_mass"],
            "val_teacher_source_mrr": val["teacher_source_mrr"],
        }
        log_rows.append(row)
        print(
            f"[{task} | {architecture} h={num_heads} seed={run_seed} | epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.4f} train={row['train_acc']:.3f} "
            f"val={row['val_acc']:.3f} source_mrr={row['val_teacher_source_mrr']:.3f}",
            flush=True,
        )
        if val["acc"] > best_val:
            best_val = val["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += args.eval_every
        if args.stop_on_solved and val["acc"] >= args.solved_threshold:
            break
        if bad_epochs >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    id_stats = evaluate(
        model,
        task,
        args.id_test_graphs,
        args.eval_batch_size,
        args.num_nodes,
        args.key_vocab_size,
        args.value_vocab_size,
        run_seed + 30_000,
        device,
    )
    ood_stats = evaluate(
        model,
        task,
        args.ood_test_graphs,
        args.eval_batch_size,
        args.ood_num_nodes,
        args.key_vocab_size,
        args.value_vocab_size,
        run_seed + 40_000,
        device,
    )
    summary = {
        **meta,
        "task": task,
        "best_val_acc": best_val,
        "id_acc": id_stats["acc"],
        "ood_acc": ood_stats["acc"],
        "id_teacher_source_mass": id_stats["teacher_source_mass"],
        "id_teacher_source_mrr": id_stats["teacher_source_mrr"],
    }
    metric_outputs = compute_metrics(
        model,
        task,
        meta,
        args.metric_graphs,
        args.metric_batch_size,
        args.num_nodes,
        args.key_vocab_size,
        args.value_vocab_size,
        args.metric_perms,
        args.metric_alpha_tau,
        args.metric_space,
        run_seed + 50_000,
        device,
    )
    metric_summary = summarize_rows(metric_outputs.metric_rows)
    query_summary = summarize_rows(
        metric_outputs.query_rows,
        group_fields=DEFAULT_GROUP_FIELDS + ("query_index",),
    )
    sanity_summary = summarize_rows(metric_outputs.sanity_rows)
    partition_summary = summarize_rows(
        metric_outputs.partition_rows,
        group_fields=DEFAULT_GROUP_FIELDS + ("partition",),
    )
    snapshot_vectors = attention_snapshot(
        model,
        task,
        args.stability_graphs,
        args.num_nodes,
        args.key_vocab_size,
        args.value_vocab_size,
        run_seed + 60_000,
        device,
    )
    snapshot = {"meta": meta, "vectors": snapshot_vectors}
    write_csv(task_dir / "train_log.csv", log_rows)
    write_csv(task_dir / "summary.csv", [summary])
    write_csv(task_dir / "metric_rows.csv", metric_outputs.metric_rows)
    write_csv(task_dir / "metric_summary.csv", metric_summary)
    write_csv(task_dir / "metric_query_rows.csv", metric_outputs.query_rows)
    write_csv(task_dir / "metric_query_summary.csv", query_summary)
    write_csv(task_dir / "metric_sanity_rows.csv", metric_outputs.sanity_rows)
    write_csv(task_dir / "metric_sanity_summary.csv", sanity_summary)
    write_csv(task_dir / "metric_partition_rows.csv", metric_outputs.partition_rows)
    write_csv(task_dir / "metric_partition_summary.csv", partition_summary)
    write_csv(task_dir / "teacher_attention_summary.csv", metric_outputs.attention_rows)
    np.save(task_dir / "attention_snapshot.npy", snapshot_vectors)
    if not args.no_save_checkpoints:
        torch.save(
            {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "summary": summary,
                "args": vars(args),
            },
            task_dir / "checkpoint.pt",
        )
    return (
        summary,
        metric_summary,
        query_summary,
        sanity_summary,
        partition_summary,
        metric_outputs.attention_rows,
        log_rows,
        snapshot,
    )


def teacher_value_distribution(
    batch: Batch,
    teacher_attn: torch.Tensor,
    value_vocab_size: int,
) -> torch.Tensor:
    attn = teacher_attn.mean(dim=1)
    batch_size, num_nodes, _ = attn.shape
    target = torch.zeros(
        batch_size,
        num_nodes,
        value_vocab_size,
        device=attn.device,
        dtype=attn.dtype,
    )
    value_index = (batch.value_id - 1).clamp(0, value_vocab_size - 1)
    scatter_index = value_index[:, None, :].expand(batch_size, num_nodes, num_nodes)
    target.scatter_add_(2, scatter_index, attn)
    return target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def oracle_student_loss(
    pred_logits: torch.Tensor,
    layer: dict[str, torch.Tensor],
    batch: Batch,
    teacher_attn: torch.Tensor,
    value_vocab_size: int,
    output_loss_weight: float,
    attention_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target = teacher_value_distribution(batch, teacher_attn, value_vocab_size)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    output_loss = -(target * log_probs).sum(dim=-1).mean()
    teacher = teacher_attn.expand(-1, layer["attn"].size(1), -1, -1)
    mask = pair_mask(layer).to(device=layer["attn"].device, dtype=layer["attn"].dtype)
    if mask.size(1) == 1 and layer["attn"].size(1) != 1:
        mask = mask.expand(-1, layer["attn"].size(1), -1, -1)
    attention_mse = torch.square(layer["attn"] - teacher) * mask
    attention_loss = attention_mse.sum() / mask.sum().clamp_min(1.0)
    loss = output_loss_weight * output_loss + attention_loss_weight * attention_loss
    with torch.no_grad():
        cos_q = attention_cosine_by_query(layer["attn"], teacher, mask.to(dtype=torch.bool))
        valid = mask.to(dtype=torch.bool).any(dim=-1)
        attn_cos = (cos_q * valid.to(cos_q.dtype)).sum() / valid.sum().clamp_min(1).to(cos_q.dtype)
    return loss, output_loss, attention_loss, attn_cos


@torch.no_grad()
def evaluate_oracle_student(
    model: CopyStudent,
    teacher: OracleAttentionModel,
    spec: OracleSpec,
    num_graphs: int,
    batch_size: int,
    num_nodes: int,
    key_vocab_size: int,
    value_vocab_size: int,
    seed: int,
    device: torch.device,
    output_loss_weight: float,
    attention_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    losses = []
    output_losses = []
    attention_losses = []
    attention_cosines = []
    for offset in range(0, num_graphs, batch_size):
        current = min(batch_size, num_graphs - offset)
        batch = generate_batch(
            spec.base_task, current, num_nodes, key_vocab_size, value_vocab_size, seed + offset
        ).to(device)
        _, teacher_layer = teacher(batch)
        pred_logits, layer = model(batch)
        loss, output_loss, attention_loss, attn_cos = oracle_student_loss(
            pred_logits,
            layer,
            batch,
            teacher_layer["attn"],
            value_vocab_size,
            output_loss_weight,
            attention_loss_weight,
        )
        losses.append(float(loss.detach().cpu()))
        output_losses.append(float(output_loss.detach().cpu()))
        attention_losses.append(float(attention_loss.detach().cpu()))
        attention_cosines.append(float(attn_cos.detach().cpu()))
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "output_loss": float(np.mean(output_losses)) if output_losses else 0.0,
        "attention_loss": float(np.mean(attention_losses)) if attention_losses else 0.0,
        "teacher_attention_cosine": float(np.mean(attention_cosines)) if attention_cosines else 0.0,
    }


def train_one_oracle_student(
    spec: OracleSpec,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
    run_seed: int,
    num_heads: int,
    architecture: str,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict], list[dict]]:
    if architecture != "copy_attn":
        raise NotImplementedError(
            f"Synthetic calibration architecture '{architecture}' is not implemented in this runner yet."
        )
    task_dir = out_dir / "oracle_students" / spec.name / f"{architecture}_h{num_heads}_seed{run_seed}"
    task_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "architecture": f"{architecture}_oracle_student",
        "num_heads": num_heads,
        "seed": run_seed,
        "task": spec.name,
        "base_task": spec.base_task,
        "oracle_components": json.dumps(spec.components),
    }
    teacher = OracleAttentionModel(spec.components, args.value_vocab_size, num_heads=1).to(device)
    model = CopyStudent(
        hidden_dim=args.hidden_dim,
        num_heads=num_heads,
        key_vocab_size=args.key_vocab_size,
        value_vocab_size=args.value_vocab_size,
        pe_dim=structural_pe(args.num_nodes).shape[1],
        fixed_value_embeddings=args.fixed_value_embeddings,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    log_rows = []
    steps_per_epoch = max(1, math.ceil(args.train_graphs_per_epoch / args.batch_size))
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        train_output_losses = []
        train_attention_losses = []
        train_attention_cosines = []
        for step in range(steps_per_epoch):
            current = min(args.batch_size, args.train_graphs_per_epoch - step * args.batch_size)
            batch = generate_batch(
                spec.base_task,
                current,
                args.num_nodes,
                args.key_vocab_size,
                args.value_vocab_size,
                run_seed + epoch * 10_000 + step,
            ).to(device)
            with torch.no_grad():
                _, teacher_layer = teacher(batch)
            optimizer.zero_grad(set_to_none=True)
            pred_logits, layer = model(batch)
            loss, output_loss, attention_loss, attn_cos = oracle_student_loss(
                pred_logits,
                layer,
                batch,
                teacher_layer["attn"],
                args.value_vocab_size,
                args.oracle_output_loss_weight,
                args.oracle_student_attention_loss_weight,
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_output_losses.append(float(output_loss.detach().cpu()))
            train_attention_losses.append(float(attention_loss.detach().cpu()))
            train_attention_cosines.append(float(attn_cos.detach().cpu()))
        if epoch % args.eval_every != 0 and epoch != args.max_epochs:
            continue
        val = evaluate_oracle_student(
            model,
            teacher,
            spec,
            args.val_graphs,
            args.eval_batch_size,
            args.num_nodes,
            args.key_vocab_size,
            args.value_vocab_size,
            run_seed + 20_000,
            device,
            args.oracle_output_loss_weight,
            args.oracle_student_attention_loss_weight,
        )
        row = {
            **meta,
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_output_loss": float(np.mean(train_output_losses)),
            "train_attention_loss": float(np.mean(train_attention_losses)),
            "train_teacher_attention_cosine": float(np.mean(train_attention_cosines)),
            "val_loss": val["loss"],
            "val_output_loss": val["output_loss"],
            "val_attention_loss": val["attention_loss"],
            "val_teacher_attention_cosine": val["teacher_attention_cosine"],
        }
        log_rows.append(row)
        print(
            f"[oracle-student:{spec.name} | {architecture} h={num_heads} seed={run_seed} | epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"attn_cos={row['val_teacher_attention_cosine']:.3f}",
            flush=True,
        )
        if val["loss"] < best_val:
            best_val = val["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += args.eval_every
        if bad_epochs >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    id_stats = evaluate_oracle_student(
        model,
        teacher,
        spec,
        args.id_test_graphs,
        args.eval_batch_size,
        args.num_nodes,
        args.key_vocab_size,
        args.value_vocab_size,
        run_seed + 30_000,
        device,
        args.oracle_output_loss_weight,
        args.oracle_student_attention_loss_weight,
    )
    summary = {
        **meta,
        "best_val_loss": best_val,
        "id_loss": id_stats["loss"],
        "id_output_loss": id_stats["output_loss"],
        "id_attention_loss": id_stats["attention_loss"],
        "id_teacher_attention_cosine": id_stats["teacher_attention_cosine"],
    }
    metric_outputs = compute_metrics(
        model,
        spec.name,
        meta,
        args.metric_graphs,
        args.metric_batch_size,
        args.num_nodes,
        args.key_vocab_size,
        args.value_vocab_size,
        args.metric_perms,
        args.metric_alpha_tau,
        args.metric_space,
        run_seed + 50_000,
        device,
        generation_task=spec.base_task,
    )
    metric_summary = summarize_rows(metric_outputs.metric_rows)
    query_summary = summarize_rows(
        metric_outputs.query_rows,
        group_fields=DEFAULT_GROUP_FIELDS + ("query_index",),
    )
    sanity_summary = summarize_rows(metric_outputs.sanity_rows)
    write_csv(task_dir / "train_log.csv", log_rows)
    write_csv(task_dir / "summary.csv", [summary])
    write_csv(task_dir / "metric_rows.csv", metric_outputs.metric_rows)
    write_csv(task_dir / "metric_summary.csv", metric_summary)
    write_csv(task_dir / "metric_query_rows.csv", metric_outputs.query_rows)
    write_csv(task_dir / "metric_query_summary.csv", query_summary)
    write_csv(task_dir / "metric_sanity_rows.csv", metric_outputs.sanity_rows)
    write_csv(task_dir / "metric_sanity_summary.csv", sanity_summary)
    if not args.no_save_checkpoints:
        torch.save(
            {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "summary": summary,
                "args": vars(args),
                "oracle_spec": spec.__dict__,
            },
            task_dir / "checkpoint.pt",
        )
    return summary, metric_summary, query_summary, sanity_summary, metric_outputs.partition_rows, log_rows


def run_oracle_metric_suite(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    oracle_dir = out_dir / "oracle_direct"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_query_rows = []
    all_sanity_rows = []
    all_partition_rows = []
    for spec in build_oracle_specs(args):
        print(f"[oracle-direct:{spec.name}] base_task={spec.base_task}", flush=True)
        model = OracleAttentionModel(spec.components, args.value_vocab_size, num_heads=1).to(device)
        meta = {
            "architecture": "oracle_attention",
            "num_heads": 1,
            "seed": args.seed,
            "task": spec.name,
            "base_task": spec.base_task,
            "oracle_components": json.dumps(spec.components),
        }
        outputs = compute_metrics(
            model,
            spec.name,
            meta,
            args.metric_graphs,
            args.metric_batch_size,
            args.num_nodes,
            args.key_vocab_size,
            args.value_vocab_size,
            args.metric_perms,
            args.metric_alpha_tau,
            args.metric_space,
            args.seed + 50_000,
            device,
            generation_task=spec.base_task,
        )
        all_rows.extend(outputs.metric_rows)
        all_query_rows.extend(outputs.query_rows)
        all_sanity_rows.extend(outputs.sanity_rows)
        all_partition_rows.extend(outputs.partition_rows)
    metric_summary = summarize_rows(all_rows)
    query_summary = summarize_rows(all_query_rows, group_fields=DEFAULT_GROUP_FIELDS + ("query_index",))
    sanity_summary = summarize_rows(all_sanity_rows)
    partition_summary = summarize_rows(all_partition_rows, group_fields=DEFAULT_GROUP_FIELDS + ("partition",))
    write_csv(oracle_dir / "metric_rows.csv", all_rows)
    write_csv(oracle_dir / "metric_summary.csv", metric_summary)
    write_csv(oracle_dir / "metric_query_rows.csv", all_query_rows)
    write_csv(oracle_dir / "metric_query_summary.csv", query_summary)
    write_csv(oracle_dir / "metric_sanity_rows.csv", all_sanity_rows)
    write_csv(oracle_dir / "metric_sanity_summary.csv", sanity_summary)
    write_csv(oracle_dir / "metric_partition_rows.csv", all_partition_rows)
    write_csv(oracle_dir / "metric_partition_summary.csv", partition_summary)
    plot_all_metrics(metric_summary, [], [], query_summary, partition_summary, [], oracle_dir)
    plot_planes(metric_summary, oracle_dir)
    return metric_summary, query_summary, sanity_summary, partition_summary


def run_oracle_student_suite(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> tuple[list[dict], list[dict], list[dict]]:
    student_dir = out_dir / "oracle_students"
    student_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_metric_summary = []
    all_query_summary = []
    all_sanity_summary = []
    all_partition_rows = []
    all_log_rows = []
    for architecture in args.architectures:
        for num_heads in args.head_counts:
            for run_seed in args.seeds:
                set_seed(run_seed)
                for spec in build_oracle_specs(args):
                    summary, metric_summary, query_summary, sanity_summary, partition_rows, log_rows = train_one_oracle_student(
                        spec, args, out_dir, device, run_seed, num_heads, architecture
                    )
                    summaries.append(summary)
                    all_metric_summary.extend(metric_summary)
                    all_query_summary.extend(query_summary)
                    all_sanity_summary.extend(sanity_summary)
                    all_partition_rows.extend(partition_rows)
                    all_log_rows.extend(log_rows)
                    write_csv(student_dir / "summary_all_runs.csv", summaries)
                    write_csv(student_dir / "metric_summary_all_runs.csv", all_metric_summary)
                    write_csv(student_dir / "metric_query_summary_all_runs.csv", all_query_summary)
                    write_csv(student_dir / "metric_sanity_summary_all_runs.csv", all_sanity_summary)
                    write_csv(student_dir / "metric_partition_rows_all_runs.csv", all_partition_rows)
                    write_csv(student_dir / "train_log_all_runs.csv", all_log_rows)
    partition_summary = summarize_rows(
        all_partition_rows,
        group_fields=DEFAULT_GROUP_FIELDS + ("partition",),
    )
    plot_all_metrics(all_metric_summary, [], [], all_query_summary, partition_summary, [], student_dir)
    plot_planes(all_metric_summary, student_dir)
    return summaries, all_metric_summary, all_query_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-head metric calibration runner.")
    parser.add_argument(
        "--mode",
        choices=["task", "oracle_direct", "oracle_student", "oracle_full"],
        default="task",
        help="task = existing learned task suite; oracle_direct = score deterministic oracle heads; "
             "oracle_student = train students on oracle value-transport targets; oracle_full = both oracle modes.",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASK_CHOICES, default=list(TASKS))
    parser.add_argument(
        "--suite-preset",
        choices=["custom", "full_zoo", "stability_three"],
        default="custom",
        help="Optional task preset. stability_three = symbolic, structural, mixed.",
    )
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=["copy_attn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--head-counts", nargs="+", type=int, default=[1])
    parser.add_argument("--num-nodes", type=int, default=16)
    parser.add_argument("--ood-num-nodes", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--key-vocab-size", type=int, default=64)
    parser.add_argument("--value-vocab-size", type=int, default=64)
    parser.add_argument("--fixed-value-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attention-loss-weight", type=float, default=1.0)
    parser.add_argument("--attention-loss-mode", choices=["mean", "best_head"], default="mean")
    parser.add_argument(
        "--oracle-suite-preset",
        choices=["core", "mix_sweep", "full"],
        default="core",
        help="Oracle calibration suite. core has uniform/symbolic/structural/mixed; mix_sweep adds uniform mixtures.",
    )
    parser.add_argument(
        "--oracle-specs",
        nargs="+",
        default=None,
        help="Optional explicit subset of oracle spec names after applying --oracle-suite-preset.",
    )
    parser.add_argument("--oracle-lambdas", nargs="+", type=float, default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument(
        "--oracle-output-loss-weight",
        type=float,
        default=1.0,
        help="Weight for soft value-transport supervision in oracle_student mode.",
    )
    parser.add_argument(
        "--oracle-student-attention-loss-weight",
        type=float,
        default=0.0,
        help="Optional direct attention MSE weight for oracle_student positive controls.",
    )
    parser.add_argument("--train-graphs-per-epoch", type=int, default=1024)
    parser.add_argument("--val-graphs", type=int, default=256)
    parser.add_argument("--id-test-graphs", type=int, default=512)
    parser.add_argument("--ood-test-graphs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--stop-on-solved", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--solved-threshold", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--metric-graphs", type=int, default=128)
    parser.add_argument("--metric-perms", type=int, default=96)
    parser.add_argument("--metric-batch-size", type=int, default=128)
    parser.add_argument(
        "--metric-alpha-tau",
        "--paper-alpha-tau",
        dest="metric_alpha_tau",
        type=float,
        default=0.1,
        help="Softmax temperature for permutation alpha weights.",
    )
    parser.add_argument(
        "--metric-space",
        choices=["logits", "attention"],
        default="attention",
        help="Pairwise tensor used for cosine scoring. Attention is the paper-aligned default.",
    )
    parser.add_argument("--stability-graphs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--drive-output-root",
        type=Path,
        default=Path("MyDrive/graph_specialisation_metrics/teacher_head_calibration"),
    )
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args(notebook_safe_argv(argv))
    if args.suite_preset != "custom":
        args.tasks = list(SUITE_TASKS[args.suite_preset])
    else:
        args.tasks = [canonical_task(task) for task in args.tasks]
    if args.seeds is None:
        args.seeds = [args.seed]
    args.head_counts = sorted(set(int(heads) for heads in args.head_counts))
    if args.fast_dev_run:
        args.num_nodes = 8
        args.ood_num_nodes = 10
        args.train_graphs_per_epoch = 64
        args.val_graphs = 32
        args.id_test_graphs = 32
        args.ood_test_graphs = 32
        args.batch_size = 16
        args.eval_batch_size = 32
        args.max_epochs = 2
        args.metric_graphs = 16
        args.metric_perms = 1
        args.stability_graphs = 8
        args.no_save_checkpoints = True
    if args.key_vocab_size < max(args.num_nodes, args.ood_num_nodes):
        raise ValueError("--key-vocab-size must be >= max(num_nodes, ood_num_nodes)")
    if args.value_vocab_size < max(args.num_nodes, args.ood_num_nodes):
        raise ValueError("--value-vocab-size must be >= max(num_nodes, ood_num_nodes)")
    if args.metric_alpha_tau <= 0:
        raise ValueError("--metric-alpha-tau must be positive")
    if args.oracle_output_loss_weight < 0 or args.oracle_student_attention_loss_weight < 0:
        raise ValueError("Oracle student loss weights must be non-negative")
    for lam in args.oracle_lambdas:
        if lam < 0 or lam > 1:
            raise ValueError("--oracle-lambdas values must lie in [0, 1]")
    for heads in args.head_counts:
        if args.hidden_dim % heads != 0:
            raise ValueError("--hidden-dim must be divisible by every value in --head-counts")
    return args


def print_plan(args: argparse.Namespace) -> None:
    print(f"mode={args.mode}")
    print(f"tasks={args.tasks}")
    if args.mode.startswith("oracle"):
        print(f"oracle_specs={[spec.name for spec in build_oracle_specs(args)]}")
        print(f"oracle_student_attention_loss_weight={args.oracle_student_attention_loss_weight}")
    print(f"architectures={args.architectures}")
    print(f"seeds={args.seeds}")
    print(f"head_counts={args.head_counts}")
    print(f"num_nodes={args.num_nodes}, ood_num_nodes={args.ood_num_nodes}")
    print(f"hidden_dim={args.hidden_dim}")
    print(f"attention_loss_weight={args.attention_loss_weight}")
    print(f"metric_alpha_tau={args.metric_alpha_tau}")
    print(f"metric_space={args.metric_space}")
    print(f"train_graphs_per_epoch={args.train_graphs_per_epoch}, max_epochs={args.max_epochs}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        print_plan(args)
        return
    device = choose_device(args.device)
    output_root = resolve_output_root(args)
    run_name = args.run_name or datetime.now().strftime("teacher_head_calibration_%Y%m%d_%H%M%S")
    out_dir = output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"device_resolved": str(device)}, f, indent=2, default=str)
    print(f"[setup] output_dir={out_dir}", flush=True)
    print(f"[setup] device={device}", flush=True)
    start = time.time()
    if args.mode in {"oracle_direct", "oracle_student", "oracle_full"}:
        if args.mode in {"oracle_direct", "oracle_full"}:
            run_oracle_metric_suite(args, out_dir, device)
        if args.mode in {"oracle_student", "oracle_full"}:
            run_oracle_student_suite(args, out_dir, device)
        print(f"[done] oracle calibration outputs at {out_dir}", flush=True)
        print(f"[done] elapsed_minutes={(time.time() - start) / 60.0:.2f}", flush=True)
        return
    summaries = []
    all_metric_summary = []
    all_query_summary = []
    all_sanity_summary = []
    all_partition_summary = []
    all_attention = []
    all_log_rows: list[dict] = []
    snapshots: list[dict] = []
    for architecture in args.architectures:
        for num_heads in args.head_counts:
            for run_seed in args.seeds:
                set_seed(run_seed)
                for task in args.tasks:
                    (
                        summary,
                        metric_summary,
                        query_summary,
                        sanity_summary,
                        partition_summary,
                        attention_rows,
                        task_log_rows,
                        snapshot,
                    ) = train_one_task(task, args, out_dir, device, run_seed, num_heads, architecture)
                    summaries.append(summary)
                    all_metric_summary.extend(metric_summary)
                    all_query_summary.extend(query_summary)
                    all_sanity_summary.extend(sanity_summary)
                    all_partition_summary.extend(partition_summary)
                    all_attention.extend(attention_rows)
                    all_log_rows.extend(task_log_rows)
                    snapshots.append(snapshot)
                    write_csv(out_dir / "summary_all_runs.csv", summaries)
                    write_csv(out_dir / "metric_summary_all_runs.csv", all_metric_summary)
                    write_csv(out_dir / "metric_query_summary_all_runs.csv", all_query_summary)
                    write_csv(out_dir / "metric_sanity_summary_all_runs.csv", all_sanity_summary)
                    write_csv(out_dir / "metric_partition_summary_all_runs.csv", all_partition_summary)
                    write_csv(out_dir / "teacher_attention_all_runs.csv", all_attention)
    stability_rows = compute_attention_stability(snapshots)
    stability_summary = summarize_rows(
        stability_rows,
        group_fields=("architecture", "num_heads", "task", "stability_type", "head", "metric"),
    )
    write_csv(out_dir / "attention_stability_rows.csv", stability_rows)
    write_csv(out_dir / "attention_stability_summary.csv", stability_summary)
    plot_all_metrics(
        all_metric_summary,
        summaries,
        all_log_rows,
        all_query_summary,
        all_partition_summary,
        stability_summary,
        out_dir,
    )
    plot_planes(all_metric_summary, out_dir)
    print("[done] calibration summary", flush=True)
    for row in summaries:
        print(
            f"{row['task']} h={row['num_heads']} seed={row['seed']} val={row['best_val_acc']:.3f} "
            f"ID={row['id_acc']:.3f} OOD={row['ood_acc']:.3f} "
            f"source_mrr={row['id_teacher_source_mrr']:.3f}",
            flush=True,
        )
    print(f"[done] wrote {out_dir}", flush=True)
    print(f"[done] elapsed_minutes={(time.time() - start) / 60.0:.2f}", flush=True)


if __name__ == "__main__":
    main()
