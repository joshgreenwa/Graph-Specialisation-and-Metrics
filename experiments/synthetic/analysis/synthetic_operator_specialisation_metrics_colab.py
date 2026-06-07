#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Head-specialisation metrics for synthetic operator-distillation students.

This is a standalone, Colab-friendly analysis script for checkpoints produced
by the synthetic graph-operator distillation runner. It loads the best seed for
each `(teacher, student)` pair, where "best" means the lowest `id_relative_mse`
in that run's `summary.json`, then computes ZINC-style alpha-weighted
permutation metrics over the trained graph-transformer attention heads.

Default expected checkpoint layout:

    /content/drive/MyDrive/graph_operator_distillation/paper_lite_4seed/
      seed0/runs/<teacher>/<student>/checkpoint.pt
      seed0/runs/<teacher>/<student>/summary.json
      seed1/runs/<teacher>/<student>/...

Core metrics:

* positional_score: x-only permutation invariance. High means the head ignores
  symbolic/content node features.
* symbolic_score: x-only permutation equivariance. High means the head follows
  symbolic/content node features.
* pe_invariance: structural-channel permutation invariance. High means the
  head ignores exposed structural graph channels.
* pe_equivariance: structural-channel permutation equivariance. High means the
  head follows exposed structural graph channels.
* centered variants of the same scores, using mean-subtracted attention rows.
* entropy_norm and relabel_equivariance sanity metrics.

The scoring follows the current ZINC metric pipeline's alpha-weighted global
permutation recipe: for every graph, head, and query, permutation scores are
weighted by the clean attention mass moved by that permutation.

`local_gnn` is excluded from head-specialisation metrics because it has no
attention heads or learned dense pairwise routing distribution. It is still
reported in selected-run metadata as not applicable.

Colab usage:

    !python synthetic_operator_specialisation_metrics_colab.py \\
        --input-root /content/drive/MyDrive/graph_operator_distillation/paper_lite_4seed \\
        --num-perms 128 \\
        --num-graphs 32

If pasted into a Colab cell, edit CELL_ARGS at the bottom and run:

    main(CELL_ARGS)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Synthetic runner compatibility: constants, graph generation, and students.
# These are the inference-compatible pieces from the operator-distillation
# training file. The training loop and teachers are intentionally omitted.
# -----------------------------------------------------------------------------


TEACHERS = (
    "local_mean_gcn",
    "local_gin_sum",
    "local_edge_gated",
    "ppr_diffusion",
    "global_anchor",
    "structural_hub",
    "ring_global",
)

STUDENTS = (
    "vanilla_gt",
    "graphormer",
    "csa",
    "csa_ring",
    "grit",
    "graphgps",
    "local_gnn",
)

ATTENTION_STUDENTS = tuple(s for s in STUDENTS if s != "local_gnn")

EDGE_TYPES = 4
SPD_CAP = 8
SPD_BUCKETS = SPD_CAP + 2
RRWP_STEPS = 6
RING_EXTRA_DIM = 3

PAIR_GRAPHORMER_DIM = 2 + EDGE_TYPES + SPD_BUCKETS
PAIR_BASE_DIM = 3 + EDGE_TYPES + SPD_BUCKETS + RRWP_STEPS + 2
PAIR_RING_DIM = PAIR_BASE_DIM + RING_EXTRA_DIM


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    train_nodes: int = 24
    train_nodes_min: int = 24
    train_nodes_max: int = 24
    ood_nodes: int = 40
    input_dim: int = 16
    hidden_dim: int = 64
    target_dim: int = 16
    pair_dim: int = 32
    layers: int = 4
    heads: int = 4
    batch_size: int = 32
    eval_batch_size: int = 32
    max_steps: int = 2200
    eval_every: int = 100
    patience_evals: int = 8
    train_graphs_per_eval: int = 3200
    eval_graphs: int = 384
    example_nodes: int = 24
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-5
    warmup_steps: int = 200
    grad_clip: float = 1.0
    dropout: float = 0.0
    attn_dropout: float = 0.05
    amp: bool = True
    operator_loss_weight: float = 0.0
    p_extra: float = 0.07
    ring_count: int = 1
    ring_min: int = 5
    ring_max: int = 7


@dataclass
class GraphBatch:
    x: torch.Tensor
    node_mask: torch.Tensor
    adj: torch.Tensor
    edge_type: torch.Tensor
    degree: torch.Tensor
    spd: torch.Tensor
    rrwp: torch.Tensor
    rwse: torch.Tensor
    pair_graphormer: torch.Tensor
    pair_base: torch.Tensor
    pair_ring: torch.Tensor
    ring_pair: torch.Tensor
    anchor_index: torch.Tensor

    def to(self, device: torch.device) -> "GraphBatch":
        return GraphBatch(
            x=self.x.to(device),
            node_mask=self.node_mask.to(device),
            adj=self.adj.to(device),
            edge_type=self.edge_type.to(device),
            degree=self.degree.to(device),
            spd=self.spd.to(device),
            rrwp=self.rrwp.to(device),
            rwse=self.rwse.to(device),
            pair_graphormer=self.pair_graphormer.to(device),
            pair_base=self.pair_base.to(device),
            pair_ring=self.pair_ring.to(device),
            ring_pair=self.ring_pair.to(device),
            anchor_index=self.anchor_index.to(device),
        )

    def clone(self) -> "GraphBatch":
        values = {f.name: getattr(self, f.name).clone() for f in fields(GraphBatch)}
        return GraphBatch(**values)

    @property
    def pair_mask(self) -> torch.Tensor:
        return self.node_mask[:, :, None] & self.node_mask[:, None, :]


def stable_hash_int(*items: object, modulo: int = 1_000_000_000) -> int:
    text = "|".join(str(item) for item in items)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def shortest_path_buckets_np(adj: np.ndarray, cap: int = SPD_CAP) -> np.ndarray:
    n = adj.shape[0]
    dist = np.full((n, n), cap + 1, dtype=np.int64)
    for src in range(n):
        dist[src, src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            if dist[src, cur] >= cap:
                continue
            for nxt in np.flatnonzero(adj[cur] > 0):
                if dist[src, nxt] > dist[src, cur] + 1:
                    dist[src, nxt] = dist[src, cur] + 1
                    queue.append(int(nxt))
    return dist


def rrwp_np(adj: np.ndarray, steps: int = RRWP_STEPS) -> np.ndarray:
    n = adj.shape[0]
    deg = adj.sum(axis=1, keepdims=True).astype(np.float32)
    m = np.divide(
        adj.astype(np.float32),
        np.maximum(deg, 1.0),
        out=np.zeros_like(adj, dtype=np.float32),
        where=deg > 0,
    )
    powers = [np.eye(n, dtype=np.float32)]
    cur = np.eye(n, dtype=np.float32)
    for _ in range(1, steps):
        cur = cur @ m
        powers.append(cur.astype(np.float32))
    return np.stack(powers, axis=-1)


def one_hot_np(values: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros(values.shape + (num_classes,), dtype=np.float32)
    safe = np.clip(values, 0, num_classes - 1)
    np.put_along_axis(out, safe[..., None], 1.0, axis=-1)
    return out


def build_pair_features_np(
    adj: np.ndarray,
    edge_type: np.ndarray,
    spd: np.ndarray,
    rrwp: np.ndarray,
    degree: np.ndarray,
    ring_pair: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = adj.shape[0]
    eye = np.eye(n, dtype=np.float32)
    adj_f = adj.astype(np.float32)
    nonedge = ((1.0 - adj_f) * (1.0 - eye)).astype(np.float32)
    spd_oh = one_hot_np(np.clip(spd, 0, SPD_BUCKETS - 1), SPD_BUCKETS)
    edge_oh = one_hot_np(edge_type.astype(np.int64), EDGE_TYPES)
    deg_log = np.log1p(degree.astype(np.float32)) / math.log1p(max(1.0, float(n - 1)))
    deg_i = np.repeat(deg_log[:, None], n, axis=1)[..., None]
    deg_j = np.repeat(deg_log[None, :], n, axis=0)[..., None]

    pair_graphormer = np.concatenate(
        [eye[..., None], adj_f[..., None], edge_oh, spd_oh],
        axis=-1,
    ).astype(np.float32)
    pair_base = np.concatenate(
        [
            eye[..., None],
            adj_f[..., None],
            nonedge[..., None],
            edge_oh,
            spd_oh,
            rrwp.astype(np.float32),
            deg_i.astype(np.float32),
            deg_j.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    ring_edge = (edge_type == 3).astype(np.float32)
    ring_member = (ring_pair.sum(axis=-1) > 0).astype(np.float32)
    both_ring_member = (ring_member[:, None] * ring_member[None, :]).astype(np.float32)
    ring_extra = np.stack(
        [ring_pair.astype(np.float32), ring_edge, both_ring_member],
        axis=-1,
    )
    pair_ring = np.concatenate([pair_base, ring_extra], axis=-1).astype(np.float32)
    rwse = np.stack(
        [np.diag(rrwp[..., k]) for k in range(rrwp.shape[-1])],
        axis=-1,
    ).astype(np.float32)
    return pair_graphormer, pair_base, pair_ring, rwse


def add_undirected_edge(adj: np.ndarray, edge_type: np.ndarray, i: int, j: int, typ: int) -> None:
    if i == j:
        return
    adj[i, j] = 1
    adj[j, i] = 1
    edge_type[i, j] = max(edge_type[i, j], typ)
    edge_type[j, i] = max(edge_type[j, i], typ)


def sample_graph_np(n: int, cfg: ExperimentConfig, rng: np.random.Generator) -> dict[str, np.ndarray | int]:
    adj = np.zeros((n, n), dtype=np.int64)
    edge_type = np.zeros((n, n), dtype=np.int64)
    ring_pair = np.zeros((n, n), dtype=np.int64)

    for node in range(1, n):
        parent = int(rng.integers(0, node))
        add_undirected_edge(adj, edge_type, node, parent, 1)

    p = min(0.35, max(0.0, cfg.p_extra))
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0 and float(rng.random()) < p:
                add_undirected_edge(adj, edge_type, i, j, 2)

    for _ in range(max(0, cfg.ring_count)):
        if n < cfg.ring_min:
            break
        length = int(rng.integers(cfg.ring_min, min(cfg.ring_max, n) + 1))
        nodes = rng.choice(n, size=length, replace=False).astype(int).tolist()
        for idx, u in enumerate(nodes):
            v = nodes[(idx + 1) % length]
            add_undirected_edge(adj, edge_type, int(u), int(v), 3)
        for u in nodes:
            for v in nodes:
                ring_pair[int(u), int(v)] = 1

    degree = adj.sum(axis=1).astype(np.float32)
    spd = shortest_path_buckets_np(adj, SPD_CAP)
    rrwp = rrwp_np(adj, RRWP_STEPS)
    pair_graphormer, pair_base, pair_ring, rwse = build_pair_features_np(
        adj, edge_type, spd, rrwp, degree, ring_pair
    )

    x = rng.normal(0.0, 1.0, size=(n, cfg.input_dim)).astype(np.float32)
    anchor_index = int(rng.integers(0, n))
    x[:, 0] = rng.normal(0.0, 0.2, size=n).astype(np.float32)
    x[anchor_index, 0] = 3.0 + float(rng.random())
    x[:, 1] = rng.normal(0.0, 1.0, size=n).astype(np.float32)

    return {
        "x": x,
        "adj": adj.astype(np.float32),
        "edge_type": edge_type.astype(np.int64),
        "degree": degree.astype(np.float32),
        "spd": spd.astype(np.int64),
        "rrwp": rrwp.astype(np.float32),
        "rwse": rwse.astype(np.float32),
        "pair_graphormer": pair_graphormer,
        "pair_base": pair_base,
        "pair_ring": pair_ring,
        "ring_pair": ring_pair.astype(np.float32),
        "anchor_index": anchor_index,
    }


def generate_batch(
    cfg: ExperimentConfig,
    batch_size: int,
    num_nodes: int,
    seed: int,
    device: torch.device | None = None,
) -> GraphBatch:
    rng = np.random.default_rng(seed)
    records = [sample_graph_np(num_nodes, cfg, rng) for _ in range(batch_size)]
    tensors = {
        "x": torch.tensor(np.stack([r["x"] for r in records]), dtype=torch.float32),
        "adj": torch.tensor(np.stack([r["adj"] for r in records]), dtype=torch.float32),
        "edge_type": torch.tensor(np.stack([r["edge_type"] for r in records]), dtype=torch.long),
        "degree": torch.tensor(np.stack([r["degree"] for r in records]), dtype=torch.float32),
        "spd": torch.tensor(np.stack([r["spd"] for r in records]), dtype=torch.long),
        "rrwp": torch.tensor(np.stack([r["rrwp"] for r in records]), dtype=torch.float32),
        "rwse": torch.tensor(np.stack([r["rwse"] for r in records]), dtype=torch.float32),
        "pair_graphormer": torch.tensor(np.stack([r["pair_graphormer"] for r in records]), dtype=torch.float32),
        "pair_base": torch.tensor(np.stack([r["pair_base"] for r in records]), dtype=torch.float32),
        "pair_ring": torch.tensor(np.stack([r["pair_ring"] for r in records]), dtype=torch.float32),
        "ring_pair": torch.tensor(np.stack([r["ring_pair"] for r in records]), dtype=torch.float32),
        "anchor_index": torch.tensor([int(r["anchor_index"]) for r in records], dtype=torch.long),
    }
    node_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
    batch = GraphBatch(node_mask=node_mask, **tensors)
    return batch.to(device) if device is not None else batch


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    sentinel = torch.finfo(logits.dtype).min if logits.dtype.is_floating_point else -1.0e9
    logits = logits.masked_fill(~mask, sentinel)
    out = torch.softmax(logits, dim=dim)
    return out.masked_fill(~mask, 0.0)


class NodeEncoder(nn.Module):
    def __init__(self, cfg: ExperimentConfig, structural: str) -> None:
        super().__init__()
        self.structural = structural
        extra = 0
        if structural in {"degree", "rwse"}:
            extra += 1
        if structural == "rwse":
            extra += RRWP_STEPS
        self.proj = nn.Linear(cfg.input_dim + extra, cfg.hidden_dim)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        parts = [batch.x]
        if self.structural in {"degree", "rwse"}:
            denom = math.log1p(max(1, batch.x.size(1) - 1))
            parts.append(torch.log1p(batch.degree).unsqueeze(-1) / denom)
        if self.structural == "rwse":
            parts.append(batch.rwse)
        return self.proj(torch.cat(parts, dim=-1))


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class StandardAttentionLayer(nn.Module):
    def __init__(self, cfg: ExperimentConfig, pair_dim: int = 0, value_pair: bool = False) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.cfg = cfg
        self.head_dim = cfg.hidden_dim // cfg.heads
        self.value_pair = value_pair
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim, bias=False)
        self.pair_bias = nn.Linear(pair_dim, cfg.heads, bias=False) if pair_dim > 0 else None
        self.pair_value = nn.Linear(pair_dim, cfg.hidden_dim, bias=False) if pair_dim > 0 and value_pair else None
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.attn_dropout)
        self.ffn = FeedForward(cfg.hidden_dim, cfg.dropout)

    def forward(
        self,
        h: torch.Tensor,
        pair: torch.Tensor | None,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n, dim = h.shape
        x = self.norm(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if pair is not None and self.pair_bias is not None:
            logits = logits + self.pair_bias(pair).permute(0, 3, 1, 2)
        mask = pair_mask[:, None, :, :]
        attn = masked_softmax(logits, mask)
        attn_used = self.drop(attn)
        if pair is not None and self.pair_value is not None:
            pair_v = self.pair_value(pair).view(bsz, n, n, self.cfg.heads, self.head_dim).permute(0, 3, 1, 2, 4)
            msg = v[:, :, None, :, :] + pair_v
            ctx = torch.einsum("bhij,bhijd->bhid", attn_used, msg)
        else:
            ctx = torch.matmul(attn_used, v)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, n, dim)
        h = h + self.out(ctx)
        h = self.ffn(h)
        return h, attn.detach()


class CSAChannelLayer(nn.Module):
    def __init__(self, cfg: ExperimentConfig, pair_dim: int) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.cfg = cfg
        self.head_dim = cfg.hidden_dim // cfg.heads
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim, bias=False)
        self.filter = nn.Linear(pair_dim, cfg.hidden_dim, bias=False)
        self.pair_value = nn.Linear(pair_dim, cfg.hidden_dim, bias=False)
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.attn_dropout)
        self.ffn = FeedForward(cfg.hidden_dim, cfg.dropout)

    def forward(self, h: torch.Tensor, pair: torch.Tensor, pair_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n, dim = h.shape
        x = self.norm(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        filt = self.filter(pair).view(bsz, n, n, self.cfg.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        pair_v = self.pair_value(pair).view(bsz, n, n, self.cfg.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        logits = qk[..., None] + filt
        mask = pair_mask[:, None, :, :, None]
        alpha = masked_softmax(logits, mask, dim=3)
        alpha_used = self.drop(alpha)
        msg = v[:, :, None, :, :] + pair_v
        ctx = (alpha_used * msg).sum(dim=3)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, n, dim)
        h = h + self.out(ctx)
        h = self.ffn(h)
        return h, alpha.detach().mean(dim=-1)


class GRITLayer(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.cfg = cfg
        self.head_dim = cfg.hidden_dim // cfg.heads
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.pair_norm = nn.LayerNorm(cfg.pair_dim)
        self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim, bias=False)
        self.pair_bias = nn.Linear(cfg.pair_dim, cfg.heads, bias=False)
        self.pair_value = nn.Linear(cfg.pair_dim, cfg.hidden_dim, bias=False)
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.theta1 = nn.Parameter(torch.ones(cfg.hidden_dim))
        self.theta2 = nn.Parameter(torch.zeros(cfg.hidden_dim))
        self.ffn = FeedForward(cfg.hidden_dim, cfg.dropout)
        self.drop = nn.Dropout(cfg.attn_dropout)
        self.pair_update = nn.Sequential(
            nn.Linear(cfg.pair_dim + 2 * cfg.hidden_dim, 2 * cfg.pair_dim),
            nn.GELU(),
            nn.Linear(2 * cfg.pair_dim, cfg.pair_dim),
        )
        self.pair_gate = nn.Linear(cfg.pair_dim, cfg.pair_dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor, batch: GraphBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n, dim = h.shape
        x = self.norm(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        z_read = self.pair_norm(z)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits + self.pair_bias(z_read).permute(0, 3, 1, 2)
        mask = batch.pair_mask[:, None, :, :]
        attn = masked_softmax(logits, mask)
        attn_used = self.drop(attn)
        pair_v = self.pair_value(z_read).view(bsz, n, n, self.cfg.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        msg = v[:, :, None, :, :] + pair_v
        ctx = torch.einsum("bhij,bhijd->bhid", attn_used, msg)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, n, dim)
        deg = torch.log1p(batch.degree).unsqueeze(-1)
        ctx = ctx * (self.theta1 + deg * self.theta2)
        h_new = h + self.out(ctx)
        h_new = self.ffn(h_new)

        hi = h_new[:, :, None, :].expand(-1, -1, n, -1)
        hj = h_new[:, None, :, :].expand(-1, n, -1, -1)
        proposal = self.pair_update(torch.cat([z_read, hi, hj], dim=-1))
        gate = torch.sigmoid(self.pair_gate(z_read))
        z_new = (z + 0.25 * gate * proposal) * batch.pair_mask.unsqueeze(-1)
        return h_new, z_new, attn.detach()


class DenseGINEBranch(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.edge_emb = nn.Embedding(EDGE_TYPES, cfg.hidden_dim)
        self.eps = nn.Parameter(torch.zeros(()))
        self.msg = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, h: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
        edge = self.edge_emb(batch.edge_type.clamp(0, EDGE_TYPES - 1))
        msg = self.msg(F.relu(h[:, None, :, :] + edge))
        agg = (msg * batch.adj.unsqueeze(-1)).sum(dim=2)
        return self.norm((1.0 + self.eps) * h + agg)


class GPSLayer(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.local = DenseGINEBranch(cfg)
        self.global_attn = StandardAttentionLayer(cfg, pair_dim=0, value_pair=False)
        self.mix = nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim)
        self.ffn = FeedForward(cfg.hidden_dim, cfg.dropout)

    def forward(self, h: torch.Tensor, batch: GraphBatch) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.local(h, batch)
        global_h, attn = self.global_attn(h, None, batch.pair_mask)
        h = h + self.mix(torch.cat([local, global_h], dim=-1))
        h = self.ffn(h)
        return h, attn


class LocalGNNLayer(nn.Module):
    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.branch = DenseGINEBranch(cfg)
        self.ffn = FeedForward(cfg.hidden_dim, cfg.dropout)

    def forward(self, h: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
        return self.ffn(h + self.branch(h, batch))


class StudentModel(nn.Module):
    def __init__(self, cfg: ExperimentConfig, name: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        if name == "vanilla_gt":
            self.encoder = NodeEncoder(cfg, "none")
            self.layers = nn.ModuleList([StandardAttentionLayer(cfg) for _ in range(cfg.layers)])
        elif name == "graphormer":
            self.encoder = NodeEncoder(cfg, "degree")
            self.layers = nn.ModuleList(
                [StandardAttentionLayer(cfg, pair_dim=PAIR_GRAPHORMER_DIM, value_pair=False) for _ in range(cfg.layers)]
            )
        elif name == "csa":
            self.encoder = NodeEncoder(cfg, "degree")
            self.layers = nn.ModuleList([CSAChannelLayer(cfg, PAIR_BASE_DIM) for _ in range(cfg.layers)])
        elif name == "csa_ring":
            self.encoder = NodeEncoder(cfg, "degree")
            self.layers = nn.ModuleList([CSAChannelLayer(cfg, PAIR_RING_DIM) for _ in range(cfg.layers)])
        elif name == "grit":
            self.encoder = NodeEncoder(cfg, "rwse")
            self.pair_encoder = nn.Sequential(
                nn.Linear(PAIR_BASE_DIM, cfg.pair_dim),
                nn.GELU(),
                nn.Linear(cfg.pair_dim, cfg.pair_dim),
            )
            self.layers = nn.ModuleList([GRITLayer(cfg) for _ in range(cfg.layers)])
        elif name == "graphgps":
            self.encoder = NodeEncoder(cfg, "rwse")
            self.layers = nn.ModuleList([GPSLayer(cfg) for _ in range(cfg.layers)])
        elif name == "local_gnn":
            self.encoder = NodeEncoder(cfg, "rwse")
            self.layers = nn.ModuleList([LocalGNNLayer(cfg) for _ in range(cfg.layers)])
        else:
            raise ValueError(f"unknown student {name}")
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)
        self.out = nn.Linear(cfg.hidden_dim, cfg.target_dim)

    def forward(self, batch: GraphBatch, collect_attention: bool = False) -> tuple[torch.Tensor, dict[str, object]]:
        h = self.encoder(batch)
        attentions: list[torch.Tensor] = []
        if self.name == "grit":
            z = self.pair_encoder(batch.pair_base) * batch.pair_mask.unsqueeze(-1)
            for layer in self.layers:
                h, z, attn = layer(h, z, batch)
                attentions.append(attn)
        elif self.name in {"csa", "csa_ring"}:
            pair = batch.pair_ring if self.name == "csa_ring" else batch.pair_base
            for layer in self.layers:
                h, attn = layer(h, pair, batch.pair_mask)
                attentions.append(attn)
        elif self.name == "graphormer":
            for layer in self.layers:
                h, attn = layer(h, batch.pair_graphormer, batch.pair_mask)
                attentions.append(attn)
        elif self.name == "vanilla_gt":
            for layer in self.layers:
                h, attn = layer(h, None, batch.pair_mask)
                attentions.append(attn)
        elif self.name == "graphgps":
            for layer in self.layers:
                h, attn = layer(h, batch)
                attentions.append(attn)
        elif self.name == "local_gnn":
            for layer in self.layers:
                h = layer(h, batch)
        y = self.out(self.out_norm(h))
        info: dict[str, object] = {"attentions": attentions, "pair_mask": batch.pair_mask}
        return y, info


# -----------------------------------------------------------------------------
# Metric names and utilities.
# -----------------------------------------------------------------------------


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
M_RELABEL_EQUIVARIANT = "relabel_equivariance"

CORE_METRICS = [
    M_POSITIONAL,
    M_SYMBOLIC,
    M_PE_INVARIANT,
    M_PE_EQUIVARIANT,
    M_POSITIONAL_CENTERED,
    M_SYMBOLIC_CENTERED,
    M_PE_INVARIANT_CENTERED,
    M_PE_EQUIVARIANT_CENTERED,
    M_INTERACTION_RESIDUAL_CENTERED,
    M_JOINT_EQUIVARIANCE_EXCESS_CENTERED,
    M_ENTROPY,
    M_RELABEL_EQUIVARIANT,
]


def safe_mkdir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    safe_mkdir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def cfg_from_dict(data: Mapping[str, Any]) -> ExperimentConfig:
    allowed = {f.name for f in fields(ExperimentConfig)}
    clean = {k: data[k] for k in allowed if k in data}
    return ExperimentConfig(**clean)


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def is_number_like(value: Any) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return math.isfinite(x)


def gather_node_axis(t: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    bsz, n = perm_pos.shape
    if t.dim() < 2 or t.size(0) != bsz or t.size(1) != n:
        raise ValueError(f"expected tensor with leading shape [B,N], got {tuple(t.shape)}")
    idx = perm_pos.to(device=t.device)
    extra = t.dim() - 2
    idx = idx.view(bsz, n, *([1] * extra)).expand_as(t)
    return torch.gather(t, dim=1, index=idx)


def gather_pair_axes(t: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    bsz, n = perm_pos.shape
    if t.dim() < 3 or t.size(0) != bsz or t.size(1) != n or t.size(2) != n:
        raise ValueError(f"expected tensor with leading shape [B,N,N], got {tuple(t.shape)}")
    idx = perm_pos.to(device=t.device)
    extra = t.dim() - 3
    row_idx = idx.view(bsz, n, 1, *([1] * extra)).expand_as(t)
    out = torch.gather(t, dim=1, index=row_idx)
    col_idx = idx.view(bsz, 1, n, *([1] * extra)).expand_as(out)
    return torch.gather(out, dim=2, index=col_idx)


def inverse_perm_pos(perm_pos: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm_pos)
    arange = torch.arange(perm_pos.size(1), dtype=perm_pos.dtype, device=perm_pos.device)
    inv.scatter_(1, perm_pos, arange[None, :].expand_as(perm_pos))
    return inv


def make_node_permutation(batch_size: int, num_nodes: int, generator: torch.Generator) -> torch.Tensor:
    perm_pos = torch.empty(batch_size, num_nodes, dtype=torch.long)
    for b in range(batch_size):
        perm_pos[b] = torch.randperm(num_nodes, generator=generator)
    return perm_pos


def make_x_permuted_batch(batch: GraphBatch, perm_pos: torch.Tensor) -> GraphBatch:
    out = batch.clone()
    out.x = gather_node_axis(batch.x, perm_pos)
    return out


def make_structure_permuted_batch(batch: GraphBatch, perm_pos: torch.Tensor) -> GraphBatch:
    out = batch.clone()
    out.degree = gather_node_axis(batch.degree, perm_pos)
    out.rwse = gather_node_axis(batch.rwse, perm_pos)
    out.adj = gather_pair_axes(batch.adj, perm_pos)
    out.edge_type = gather_pair_axes(batch.edge_type, perm_pos)
    out.spd = gather_pair_axes(batch.spd, perm_pos)
    out.rrwp = gather_pair_axes(batch.rrwp, perm_pos)
    out.pair_graphormer = gather_pair_axes(batch.pair_graphormer, perm_pos)
    out.pair_base = gather_pair_axes(batch.pair_base, perm_pos)
    out.pair_ring = gather_pair_axes(batch.pair_ring, perm_pos)
    out.ring_pair = gather_pair_axes(batch.ring_pair, perm_pos)
    return out


def make_pe_permuted_batch(batch: GraphBatch, perm_pos: torch.Tensor, student_name: str) -> GraphBatch | None:
    if student_name == "vanilla_gt":
        return None
    return make_structure_permuted_batch(batch, perm_pos)


def make_x_and_pe_permuted_batch(batch: GraphBatch, perm_pos: torch.Tensor, student_name: str) -> GraphBatch | None:
    out = make_pe_permuted_batch(batch, perm_pos, student_name)
    if out is None:
        return None
    out.x = gather_node_axis(batch.x, perm_pos)
    return out


def make_relabel_batch(batch: GraphBatch, perm_pos: torch.Tensor) -> GraphBatch:
    out = make_structure_permuted_batch(batch, perm_pos)
    out.x = gather_node_axis(batch.x, perm_pos)
    inv = inverse_perm_pos(perm_pos)
    out.anchor_index = torch.gather(inv, 1, batch.anchor_index[:, None]).squeeze(1)
    return out


@torch.no_grad()
def collect_attention(model: StudentModel, batch_cpu: GraphBatch, device: torch.device) -> dict[str, Any]:
    model.eval()
    batch = batch_cpu.to(device)
    _, info = model(batch, collect_attention=True)
    attentions = info.get("attentions", [])
    if not isinstance(attentions, list):
        attentions = []
    pair_mask = batch_cpu.pair_mask[:, None, :, :].detach().cpu()
    node_mask = batch_cpu.node_mask.detach().cpu()
    layers = []
    for layer_idx, attn in enumerate(attentions):
        if not isinstance(attn, torch.Tensor):
            continue
        layers.append(
            {
                "layer": layer_idx,
                "attn": attn.detach().float().cpu(),
                "pair_mask": pair_mask,
                "node_mask": node_mask,
            }
        )
    return {"layers": layers}


def pair_mask_from_layer(layer: Mapping[str, Any]) -> torch.Tensor:
    attn = layer["attn"]
    mask = layer["pair_mask"].to(dtype=torch.bool, device=attn.device)
    if mask.size(1) == 1 and attn.size(1) != 1:
        mask = mask.expand(-1, attn.size(1), -1, -1)
    return mask


def transform_pair_reference(z: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    bsz, heads, n, _ = z.shape
    idx = perm_pos.to(device=z.device)
    row_idx = idx[:, None, :, None].expand(bsz, heads, n, n)
    rows = torch.gather(z, dim=2, index=row_idx)
    col_idx = idx[:, None, None, :].expand(bsz, heads, n, n)
    return torch.gather(rows, dim=3, index=col_idx)


def transform_pair_mask(mask: torch.Tensor, perm_pos: torch.Tensor, heads: int) -> torch.Tensor:
    m = mask.to(dtype=torch.long)
    if m.size(1) == 1 and heads != 1:
        m = m.expand(-1, heads, -1, -1)
    return transform_pair_reference(m, perm_pos).to(dtype=torch.bool)


def row_center(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(dtype=torch.bool, device=t.device)
    if m.size(1) == 1 and t.size(1) != 1:
        m = m.expand(-1, t.size(1), -1, -1)
    t0 = torch.where(m, torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(t))
    denom = m.sum(dim=-1, keepdim=True).clamp_min(1).to(t.dtype)
    mean = t0.sum(dim=-1, keepdim=True) / denom
    return torch.where(m, t0 - mean, torch.zeros_like(t0))


def cosine_by_query(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, centered: bool) -> torch.Tensor:
    m = mask.to(dtype=torch.bool, device=u.device)
    if m.size(1) == 1 and u.size(1) != 1:
        m = m.expand(-1, u.size(1), -1, -1)
    if centered:
        u0 = row_center(u, m)
        v0 = row_center(v, m)
    else:
        u0 = torch.where(m, torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(u))
        v0 = torch.where(m, torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(v))
    num = (u0 * v0).sum(dim=-1)
    den = torch.sqrt((u0 * u0).sum(dim=-1).clamp_min(1e-12)) * torch.sqrt((v0 * v0).sum(dim=-1).clamp_min(1e-12))
    cos = num / den.clamp_min(1e-12)
    if centered:
        return torch.clamp(cos, -1.0, 1.0)
    return torch.clamp(cos, 0.0, 1.0)


def moved_attention_mass_by_query(clean_layer: Mapping[str, Any], perm_pos: torch.Tensor) -> torch.Tensor:
    attn = clean_layer["attn"]
    mask = pair_mask_from_layer(clean_layer)
    ref = transform_pair_reference(attn, perm_pos)
    ref_mask = transform_pair_mask(mask, perm_pos, attn.size(1)).to(device=attn.device)
    union = mask | ref_mask
    clean0 = torch.where(union, attn, torch.zeros_like(attn))
    ref0 = torch.where(union, ref, torch.zeros_like(ref))
    return torch.clamp(0.5 * torch.abs(clean0 - ref0).sum(dim=-1), 0.0, 1.0)


def comparison_masks(clean_layer: Mapping[str, Any], variant_layer: Mapping[str, Any], perm_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    attn = variant_layer["attn"]
    heads = int(attn.size(1))
    clean_mask = pair_mask_from_layer(clean_layer).to(device=attn.device)
    variant_mask = pair_mask_from_layer(variant_layer).to(device=attn.device)
    if clean_mask.size(1) == 1 and heads != 1:
        clean_mask = clean_mask.expand(-1, heads, -1, -1)
    if variant_mask.size(1) == 1 and heads != 1:
        variant_mask = variant_mask.expand(-1, heads, -1, -1)
    clean_t = transform_pair_mask(clean_mask, perm_pos, heads).to(device=attn.device)
    return clean_mask & variant_mask, clean_t & variant_mask


def head_entropy_rows(model_meta: Mapping[str, Any], clean: Mapping[str, Any], batch_idx: int, graph_offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_idx, layer in enumerate(clean["layers"]):
        attn = layer["attn"].clamp_min(1e-12)
        mask = pair_mask_from_layer(layer)
        valid_rows = mask.any(dim=-1)
        key_count = mask.sum(dim=-1).to(attn.dtype).clamp_min(2.0)
        ent = -(torch.where(mask, attn * attn.log(), torch.zeros_like(attn))).sum(dim=-1) / torch.log(key_count)
        ent = torch.where(valid_rows, ent, torch.zeros_like(ent))
        denom = valid_rows.sum(dim=-1).clamp_min(1).to(attn.dtype)
        by_graph_head = (ent * valid_rows.to(attn.dtype)).sum(dim=-1) / denom
        bsz, heads = by_graph_head.shape
        for b in range(bsz):
            for h in range(heads):
                rows.append(
                    {
                        **model_meta,
                        "batch": batch_idx,
                        "graph_in_batch": b,
                        "graph_index": graph_offset + b,
                        "perm": -1,
                        "layer": layer_idx,
                        "head": h,
                        "metric": M_ENTROPY,
                        "score": float(by_graph_head[b, h]),
                    }
                )
    return rows


def alpha_weighted_plane_rows(
    model_meta: Mapping[str, Any],
    clean: Mapping[str, Any],
    variant_records: Sequence[tuple[int, torch.Tensor, Mapping[str, Any]]],
    batch_idx: int,
    graph_offset: int,
    inv_metric: str,
    equi_metric: str,
    alpha_tau: float,
    centered: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not variant_records:
        return rows
    if alpha_tau <= 0:
        raise ValueError("alpha_tau must be positive")

    for layer_idx, clean_layer in enumerate(clean["layers"]):
        clean_attn = clean_layer["attn"]
        inv_scores = []
        equi_scores = []
        moved_masses = []
        valid_queries = None
        for _perm_idx, perm_pos, variant in variant_records:
            variant_layer = variant["layers"][layer_idx]
            variant_attn = variant_layer["attn"]
            ref_attn = transform_pair_reference(clean_attn, perm_pos)
            stable_mask, follow_mask = comparison_masks(clean_layer, variant_layer, perm_pos)
            inv_scores.append(cosine_by_query(variant_attn, clean_attn, stable_mask, centered=centered))
            equi_scores.append(cosine_by_query(variant_attn, ref_attn, follow_mask, centered=centered))
            moved_masses.append(moved_attention_mass_by_query(clean_layer, perm_pos))
            row_valid = stable_mask.any(dim=-1) | follow_mask.any(dim=-1)
            valid_queries = row_valid if valid_queries is None else (valid_queries | row_valid)

        inv_stack = torch.stack(inv_scores, dim=0)
        equi_stack = torch.stack(equi_scores, dim=0)
        moved = torch.stack(moved_masses, dim=0).to(device=inv_stack.device)
        alpha = torch.softmax(moved / float(alpha_tau), dim=0)
        inv_query = (alpha * inv_stack).sum(dim=0)
        equi_query = (alpha * equi_stack).sum(dim=0)
        alpha_max = alpha.max(dim=0).values
        effective_perms = 1.0 / torch.square(alpha).sum(dim=0).clamp_min(1e-12)
        if valid_queries is None:
            continue
        valid = valid_queries.to(device=inv_query.device, dtype=torch.bool)
        denom = valid.sum(dim=-1).clamp_min(1).to(inv_query.dtype)
        inv_head = (inv_query * valid.to(inv_query.dtype)).sum(dim=-1) / denom
        equi_head = (equi_query * valid.to(equi_query.dtype)).sum(dim=-1) / denom
        moved_mean = (moved.mean(dim=0) * valid.to(moved.dtype)).sum(dim=-1) / denom
        alpha_max_mean = (alpha_max * valid.to(alpha_max.dtype)).sum(dim=-1) / denom
        effective_mean = (effective_perms * valid.to(effective_perms.dtype)).sum(dim=-1) / denom

        bsz, heads = inv_head.shape
        for b in range(bsz):
            for h in range(heads):
                common = {
                    **model_meta,
                    "batch": batch_idx,
                    "graph_in_batch": b,
                    "graph_index": graph_offset + b,
                    "perm": -1,
                    "layer": layer_idx,
                    "head": h,
                    "metric_space": "attention",
                    "centered": bool(centered),
                    "alpha_tau": float(alpha_tau),
                    "moved_mass_mean": float(moved_mean[b, h]),
                    "alpha_max_mean": float(alpha_max_mean[b, h]),
                    "effective_perms_mean": float(effective_mean[b, h]),
                }
                rows.append({**common, "metric": inv_metric, "score": float(inv_head[b, h])})
                rows.append({**common, "metric": equi_metric, "score": float(equi_head[b, h])})
    return rows


def interaction_residual_norm_by_query(clean_c: torch.Tensor, x_c: torch.Tensor, pe_c: torch.Tensor, both_c: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=clean_c.device, dtype=torch.bool)
    if m.size(1) == 1 and clean_c.size(1) != 1:
        m = m.expand(-1, clean_c.size(1), -1, -1)
    clean0 = torch.where(m, clean_c, torch.zeros_like(clean_c))
    x0 = torch.where(m, x_c, torch.zeros_like(x_c))
    pe0 = torch.where(m, pe_c, torch.zeros_like(pe_c))
    both0 = torch.where(m, both_c, torch.zeros_like(both_c))
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
    return torch.clamp(numerator / denominator.clamp_min(1e-12), 0.0, 1.0)


def alpha_weighted_mixed_rows(
    model_meta: Mapping[str, Any],
    clean: Mapping[str, Any],
    mixed_records: Sequence[tuple[int, torch.Tensor, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]],
    batch_idx: int,
    graph_offset: int,
    alpha_tau: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not mixed_records:
        return rows
    for layer_idx, clean_layer in enumerate(clean["layers"]):
        clean_mask = pair_mask_from_layer(clean_layer)
        clean_c = row_center(clean_layer["attn"], clean_mask)
        interaction_scores = []
        joint_excess_scores = []
        moved_masses = []
        valid_queries = None
        for _perm_idx, perm_pos, var_x, var_pe, var_both in mixed_records:
            x_layer = var_x["layers"][layer_idx]
            pe_layer = var_pe["layers"][layer_idx]
            both_layer = var_both["layers"][layer_idx]
            x_mask = pair_mask_from_layer(x_layer)
            pe_mask = pair_mask_from_layer(pe_layer)
            both_mask = pair_mask_from_layer(both_layer)
            x_c = row_center(x_layer["attn"], x_mask)
            pe_c = row_center(pe_layer["attn"], pe_mask)
            both_c = row_center(both_layer["attn"], both_mask)
            heads = int(clean_c.size(1))
            clean_ref = transform_pair_reference(clean_c, perm_pos)
            clean_ref_mask = transform_pair_mask(clean_mask, perm_pos, heads).to(device=clean_c.device)
            clean_mask_h = clean_mask.to(device=clean_c.device)
            if clean_mask_h.size(1) == 1 and heads != 1:
                clean_mask_h = clean_mask_h.expand(-1, heads, -1, -1)
            if x_mask.size(1) == 1 and heads != 1:
                x_mask = x_mask.expand(-1, heads, -1, -1)
            if pe_mask.size(1) == 1 and heads != 1:
                pe_mask = pe_mask.expand(-1, heads, -1, -1)
            if both_mask.size(1) == 1 and heads != 1:
                both_mask = both_mask.expand(-1, heads, -1, -1)

            x_stable = clean_mask_h & x_mask
            x_follow = clean_ref_mask & x_mask
            pe_stable = clean_mask_h & pe_mask
            pe_follow = clean_ref_mask & pe_mask
            both_follow = clean_ref_mask & both_mask
            interaction_mask = clean_mask_h & x_mask & pe_mask & both_mask

            positional_c = cosine_by_query(x_c, clean_c, x_stable, centered=True)
            symbolic_c = cosine_by_query(x_c, clean_ref, x_follow, centered=True)
            pe_invariance_c = cosine_by_query(pe_c, clean_c, pe_stable, centered=True)
            pe_equivariance_c = cosine_by_query(pe_c, clean_ref, pe_follow, centered=True)
            joint_c = cosine_by_query(both_c, clean_ref, both_follow, centered=True)
            best_single_c = torch.stack(
                [positional_c, symbolic_c, pe_invariance_c, pe_equivariance_c],
                dim=0,
            ).max(dim=0).values

            interaction_scores.append(
                interaction_residual_norm_by_query(clean_c, x_c, pe_c, both_c, interaction_mask)
            )
            joint_excess_scores.append(joint_c - best_single_c)
            moved_masses.append(moved_attention_mass_by_query(clean_layer, perm_pos))
            row_valid = interaction_mask.any(dim=-1) | both_follow.any(dim=-1)
            valid_queries = row_valid if valid_queries is None else (valid_queries | row_valid)

        if valid_queries is None:
            continue
        interaction_stack = torch.stack(interaction_scores, dim=0)
        joint_excess_stack = torch.stack(joint_excess_scores, dim=0)
        moved = torch.stack(moved_masses, dim=0).to(device=interaction_stack.device)
        alpha = torch.softmax(moved / float(alpha_tau), dim=0)
        interaction_query = (alpha * interaction_stack).sum(dim=0)
        joint_excess_query = (alpha * joint_excess_stack).sum(dim=0)
        valid = valid_queries.to(device=interaction_query.device, dtype=torch.bool)
        denom = valid.sum(dim=-1).clamp_min(1).to(interaction_query.dtype)
        interaction_head = (interaction_query * valid.to(interaction_query.dtype)).sum(dim=-1) / denom
        joint_excess_head = (joint_excess_query * valid.to(joint_excess_query.dtype)).sum(dim=-1) / denom
        bsz, heads = interaction_head.shape
        for b in range(bsz):
            for h in range(heads):
                common = {
                    **model_meta,
                    "batch": batch_idx,
                    "graph_in_batch": b,
                    "graph_index": graph_offset + b,
                    "perm": -1,
                    "layer": layer_idx,
                    "head": h,
                    "metric_space": "attention",
                    "centered": True,
                    "alpha_tau": float(alpha_tau),
                }
                rows.append(
                    {
                        **common,
                        "metric": M_INTERACTION_RESIDUAL_CENTERED,
                        "score": float(interaction_head[b, h]),
                    }
                )
                rows.append(
                    {
                        **common,
                        "metric": M_JOINT_EQUIVARIANCE_EXCESS_CENTERED,
                        "score": float(joint_excess_head[b, h]),
                    }
                )
    return rows


def relabel_equivariance_rows(
    model_meta: Mapping[str, Any],
    clean: Mapping[str, Any],
    variant: Mapping[str, Any],
    perm_pos: torch.Tensor,
    batch_idx: int,
    graph_offset: int,
    perm_idx: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_idx, (clean_layer, variant_layer) in enumerate(zip(clean["layers"], variant["layers"])):
        clean_attn = clean_layer["attn"]
        variant_attn = variant_layer["attn"]
        ref_attn = transform_pair_reference(clean_attn, perm_pos)
        _stable_mask, follow_mask = comparison_masks(clean_layer, variant_layer, perm_pos)
        score_q = cosine_by_query(variant_attn, ref_attn, follow_mask, centered=False)
        valid = follow_mask.any(dim=-1)
        denom = valid.sum(dim=-1).clamp_min(1).to(score_q.dtype)
        by_graph_head = (score_q * valid.to(score_q.dtype)).sum(dim=-1) / denom
        bsz, heads = by_graph_head.shape
        for b in range(bsz):
            for h in range(heads):
                rows.append(
                    {
                        **model_meta,
                        "batch": batch_idx,
                        "graph_in_batch": b,
                        "graph_index": graph_offset + b,
                        "perm": perm_idx,
                        "layer": layer_idx,
                        "head": h,
                        "metric": M_RELABEL_EQUIVARIANT,
                        "score": float(by_graph_head[b, h]),
                    }
                )
    return rows


def summarise_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, int, str], list[float]] = {}
    meta_by_key: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row in rows:
        score = row.get("score")
        if not is_number_like(score):
            continue
        key = (str(row["model"]), int(row["layer"]), int(row["head"]), str(row["metric"]))
        buckets.setdefault(key, []).append(float(score))
        if key not in meta_by_key:
            meta_by_key[key] = {
                "model": row.get("model"),
                "teacher": row.get("teacher"),
                "student": row.get("student"),
                "selected_seed": row.get("selected_seed"),
                "layer": row.get("layer"),
                "head": row.get("head"),
                "metric": row.get("metric"),
            }
    out: list[dict[str, Any]] = []
    for key, vals in sorted(buckets.items()):
        arr = np.asarray(vals, dtype=np.float64)
        rec = dict(meta_by_key[key])
        rec.update(
            {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                "count": int(arr.size),
                "se": float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0,
            }
        )
        out.append(rec)
    return out


# -----------------------------------------------------------------------------
# Run discovery and checkpoint loading.
# -----------------------------------------------------------------------------


def load_summary_index(input_root: Path) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    index: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    csv_paths = [
        input_root / "summary_all_seeds.csv",
        input_root / "summary_all_seeds_partial.csv",
        *sorted(input_root.glob("seed*/summary.csv")),
        *sorted(input_root.glob("seed*/summary_partial.csv")),
    ]
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        try:
            with csv_path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for csv_row in reader:
                    teacher = str(csv_row.get("teacher", ""))
                    student = str(csv_row.get("student", ""))
                    if not teacher or not student:
                        continue
                    seed_value = csv_row.get("seed")
                    seed_key = str(seed_value) if seed_value not in (None, "") else None
                    index[(teacher, student, seed_key)] = dict(csv_row)
        except Exception as exc:
            print(f"[discover][warning] failed to read summary CSV {csv_path}: {exc}", flush=True)
    return index


def infer_seed_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"seed(\d+)", part)
        if match:
            return int(match.group(1))
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else None


def discover_runs(input_root: Path, teachers: Sequence[str] | None, students: Sequence[str] | None) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    ckpt_paths: list[Path] = []
    for pattern in (
        "seed*/runs/*/*/checkpoint.pt",
        "runs/*/*/checkpoint.pt",
        "**/runs/*/*/checkpoint.pt",
    ):
        for path in input_root.glob(pattern):
            if path not in seen:
                seen.add(path)
                ckpt_paths.append(path)
    if not ckpt_paths:
        # Robust fallback for copied Drive folders or renamed checkpoints. We
        # still only accept files under .../runs/<teacher>/<student>/...
        # so unrelated PyTorch files do not become synthetic runs.
        for path in input_root.rglob("*.pt"):
            parts = path.parts
            if "runs" not in parts:
                continue
            run_idx = max(i for i, part in enumerate(parts) if part == "runs")
            if len(parts) <= run_idx + 3:
                continue
            if path not in seen:
                seen.add(path)
                ckpt_paths.append(path)

    summary_index = load_summary_index(input_root)
    rows: list[dict[str, Any]] = []
    teacher_filter = set(teachers or TEACHERS)
    student_filter = set(students or STUDENTS)
    for ckpt_path in sorted(ckpt_paths):
        parts = ckpt_path.parts
        if "runs" in parts:
            run_idx = max(i for i, part in enumerate(parts) if part == "runs")
            teacher = parts[run_idx + 1] if len(parts) > run_idx + 1 else ckpt_path.parent.parent.name
            student = parts[run_idx + 2] if len(parts) > run_idx + 2 else ckpt_path.parent.name
            run_dir = Path(*parts[: run_idx + 3])
        else:
            run_dir = ckpt_path.parent
            student = run_dir.name
            teacher = run_dir.parent.name
        if teacher not in teacher_filter or student not in student_filter:
            continue
        seed = infer_seed_from_path(ckpt_path)
        summary_path = run_dir / "summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = read_json(summary_path)
            except Exception as exc:
                print(f"[discover][warning] failed to read {summary_path}: {exc}", flush=True)
                summary = {}
        seed_from_summary = summary.get("seed")
        if seed_from_summary is not None and is_number_like(seed_from_summary):
            seed = int(float(seed_from_summary))
        csv_summary = summary_index.get((teacher, student, str(seed) if seed is not None else None))
        if csv_summary is None:
            csv_summary = summary_index.get((teacher, student, None))
        if csv_summary:
            for key, value in csv_summary.items():
                summary.setdefault(key, value)
        rank_value = summary.get("id_relative_mse", summary.get("best_val_relative_mse", float("inf")))
        if not is_number_like(rank_value):
            rank_value = float("inf")
        rows.append(
            {
                "teacher": teacher,
                "student": student,
                "seed": seed,
                "run_dir": str(run_dir),
                "summary_path": str(summary_path) if summary_path.exists() else "",
                "checkpoint_path": str(ckpt_path),
                "id_relative_mse": float(rank_value),
                "summary": summary,
                "attention_applicable": student in ATTENTION_STUDENTS,
                "not_applicable_reason": "" if student in ATTENTION_STUDENTS else "local_gnn has no attention heads",
            }
        )
    return rows


def discovery_diagnostics(input_root: Path) -> dict[str, Any]:
    exists = input_root.exists()
    diag: dict[str, Any] = {
        "input_root": str(input_root),
        "exists": exists,
        "is_dir": input_root.is_dir() if exists else False,
        "children": [],
        "runs_dirs": [],
        "checkpoint_pt_examples": [],
        "pt_examples": [],
    }
    if not exists:
        parent = input_root.parent
        diag["parent"] = str(parent)
        diag["parent_exists"] = parent.exists()
        if parent.exists():
            diag["parent_children"] = [p.name for p in sorted(parent.iterdir())[:30]]
        return diag
    try:
        diag["children"] = [p.name for p in sorted(input_root.iterdir())[:30]]
    except Exception as exc:
        diag["children_error"] = str(exc)
    try:
        diag["runs_dirs"] = [str(p) for p in list(input_root.rglob("runs"))[:20]]
    except Exception as exc:
        diag["runs_dirs_error"] = str(exc)
    try:
        diag["checkpoint_pt_examples"] = [str(p) for p in list(input_root.rglob("checkpoint.pt"))[:20]]
    except Exception as exc:
        diag["checkpoint_pt_error"] = str(exc)
    try:
        diag["pt_examples"] = [str(p) for p in list(input_root.rglob("*.pt"))[:20]]
    except Exception as exc:
        diag["pt_error"] = str(exc)
    return diag


def select_best_runs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((str(row["teacher"]), str(row["student"])), []).append(row)
    selected: list[dict[str, Any]] = []
    for (_teacher, _student), group in sorted(buckets.items()):
        best = min(group, key=lambda r: float(r.get("id_relative_mse", float("inf"))))
        selected.append(dict(best))
    return selected


def build_model_from_checkpoint(row: Mapping[str, Any], device: torch.device) -> tuple[ExperimentConfig, StudentModel, dict[str, Any]]:
    ckpt_path = Path(str(row["checkpoint_path"]))
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    cfg_data = ckpt.get("cfg", {}) if isinstance(ckpt, Mapping) else {}
    if not cfg_data:
        cfg_data = row.get("summary", {})
    cfg = cfg_from_dict(cfg_data)
    student = str(row["student"])
    model = StudentModel(cfg, student).to(device)
    state = ckpt["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return cfg, model, ckpt


def eval_nodes_for_run(cfg: ExperimentConfig, summary: Mapping[str, Any], eval_nodes_arg: str) -> int:
    if eval_nodes_arg == "id":
        return int(summary.get("id_eval_nodes", cfg.train_nodes))
    if eval_nodes_arg == "ood":
        return int(summary.get("ood_eval_nodes", cfg.ood_nodes))
    try:
        return int(eval_nodes_arg)
    except Exception as exc:
        raise ValueError("--eval-nodes must be 'id', 'ood', or an integer") from exc


def run_metrics_for_model(row: Mapping[str, Any], args: argparse.Namespace, device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    teacher = str(row["teacher"])
    student = str(row["student"])
    seed = row.get("seed")
    model_label = f"{teacher}/{student}"
    model_meta = {
        "model": model_label,
        "teacher": teacher,
        "student": student,
        "selected_seed": seed,
    }
    cfg, model, ckpt = build_model_from_checkpoint(row, device)
    summary = row.get("summary", {})
    num_nodes = eval_nodes_for_run(cfg, summary if isinstance(summary, Mapping) else {}, args.eval_nodes)
    rows: list[dict[str, Any]] = []
    rng = torch.Generator(device="cpu").manual_seed(
        int(args.seed) + stable_hash_int(teacher, student, "perms", modulo=10_000_000)
    )
    start = time.time()

    for batch_idx, graph_offset in enumerate(range(0, args.num_graphs, args.batch_size)):
        current = min(args.batch_size, args.num_graphs - graph_offset)
        batch_seed = int(args.seed) + stable_hash_int(teacher, student, "graphs", graph_offset, modulo=100_000_000)
        batch = generate_batch(cfg, current, num_nodes, batch_seed, device=None)
        clean = collect_attention(model, batch, device)
        if not clean["layers"]:
            break
        rows.extend(head_entropy_rows(model_meta, clean, batch_idx, graph_offset))

        x_records: list[tuple[int, torch.Tensor, Mapping[str, Any]]] = []
        pe_records: list[tuple[int, torch.Tensor, Mapping[str, Any]]] = []
        mixed_records: list[tuple[int, torch.Tensor, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []

        for perm_idx in range(args.num_perms):
            perm_pos = make_node_permutation(current, num_nodes, rng)
            x_variant = collect_attention(model, make_x_permuted_batch(batch, perm_pos), device)
            x_records.append((perm_idx, perm_pos, x_variant))

            pe_batch = make_pe_permuted_batch(batch, perm_pos, student)
            if pe_batch is not None:
                pe_variant = collect_attention(model, pe_batch, device)
                pe_records.append((perm_idx, perm_pos, pe_variant))
                both_batch = make_x_and_pe_permuted_batch(batch, perm_pos, student)
                assert both_batch is not None
                both_variant = collect_attention(model, both_batch, device)
                mixed_records.append((perm_idx, perm_pos, x_variant, pe_variant, both_variant))

            if not args.no_relabel_sanity:
                relabel_variant = collect_attention(model, make_relabel_batch(batch, perm_pos), device)
                rows.extend(
                    relabel_equivariance_rows(
                        model_meta,
                        clean,
                        relabel_variant,
                        perm_pos,
                        batch_idx,
                        graph_offset,
                        perm_idx,
                    )
                )

        rows.extend(
            alpha_weighted_plane_rows(
                model_meta,
                clean,
                x_records,
                batch_idx,
                graph_offset,
                M_POSITIONAL,
                M_SYMBOLIC,
                args.metric_alpha_tau,
                centered=False,
            )
        )
        rows.extend(
            alpha_weighted_plane_rows(
                model_meta,
                clean,
                x_records,
                batch_idx,
                graph_offset,
                M_POSITIONAL_CENTERED,
                M_SYMBOLIC_CENTERED,
                args.metric_alpha_tau,
                centered=True,
            )
        )
        if pe_records:
            rows.extend(
                alpha_weighted_plane_rows(
                    model_meta,
                    clean,
                    pe_records,
                    batch_idx,
                    graph_offset,
                    M_PE_INVARIANT,
                    M_PE_EQUIVARIANT,
                    args.metric_alpha_tau,
                    centered=False,
                )
            )
            rows.extend(
                alpha_weighted_plane_rows(
                    model_meta,
                    clean,
                    pe_records,
                    batch_idx,
                    graph_offset,
                    M_PE_INVARIANT_CENTERED,
                    M_PE_EQUIVARIANT_CENTERED,
                    args.metric_alpha_tau,
                    centered=True,
                )
            )
        if mixed_records:
            rows.extend(
                alpha_weighted_mixed_rows(
                    model_meta,
                    clean,
                    mixed_records,
                    batch_idx,
                    graph_offset,
                    args.metric_alpha_tau,
                )
            )

    meta = {
        "model": model_label,
        "teacher": teacher,
        "student": student,
        "selected_seed": seed,
        "checkpoint_path": row.get("checkpoint_path"),
        "run_dir": row.get("run_dir"),
        "eval_nodes": num_nodes,
        "num_graphs": args.num_graphs,
        "num_perms": args.num_perms,
        "metric_alpha_tau": args.metric_alpha_tau,
        "elapsed_sec": time.time() - start,
        "layers": cfg.layers,
        "heads": cfg.heads,
        "checkpoint_step": ckpt.get("step") if isinstance(ckpt, Mapping) else None,
    }
    return rows, meta


# -----------------------------------------------------------------------------
# Plotting.
# -----------------------------------------------------------------------------


def summary_lookup(summary_rows: Sequence[Mapping[str, Any]], metric: str) -> list[Mapping[str, Any]]:
    return [r for r in summary_rows if r.get("metric") == metric and is_number_like(r.get("mean"))]


def make_plane_records(summary_rows: Sequence[Mapping[str, Any]], x_metric: str, y_metric: str) -> list[dict[str, Any]]:
    left: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    right: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in summary_rows:
        key = (str(row.get("model")), int(row.get("layer", 0)), int(row.get("head", 0)))
        if row.get("metric") == x_metric and is_number_like(row.get("mean")):
            left[key] = row
        elif row.get("metric") == y_metric and is_number_like(row.get("mean")):
            right[key] = row
    records: list[dict[str, Any]] = []
    for key, x_row in left.items():
        y_row = right.get(key)
        if y_row is None:
            continue
        records.append(
            {
                "model": key[0],
                "teacher": x_row.get("teacher"),
                "student": x_row.get("student"),
                "layer": key[1],
                "head": key[2],
                "x": float(x_row["mean"]),
                "y": float(y_row["mean"]),
                "x_metric": x_metric,
                "y_metric": y_metric,
            }
        )
    return records


def scatter_plane(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    color_key: str = "layer",
    lo: float = 0.0,
    hi: float = 1.0,
) -> None:
    if not records:
        return
    plt = import_plotting()
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    if color_key == "layer":
        colors = [float(r["layer"]) for r in records]
        sc = ax.scatter(
            [float(r["x"]) for r in records],
            [float(r["y"]) for r in records],
            c=colors,
            cmap="viridis",
            s=64,
            edgecolors="black",
            linewidths=0.4,
        )
        fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label="Layer")
    else:
        values = sorted({str(r[color_key]) for r in records})
        cmap = plt.get_cmap("tab10" if len(values) <= 10 else "tab20")
        for idx, value in enumerate(values):
            sub = [r for r in records if str(r[color_key]) == value]
            ax.scatter(
                [float(r["x"]) for r in sub],
                [float(r["y"]) for r in sub],
                color=cmap(idx % cmap.N),
                s=52,
                alpha=0.82,
                edgecolors="black",
                linewidths=0.25,
                label=value,
            )
        ax.legend(loc="best", fontsize=7, frameon=True)
    ref_y = [lo, hi] if lo < 0 else [hi, lo]
    ax.plot([lo, hi], ref_y, color="grey", linestyle="--", linewidth=0.9)
    if lo < 0:
        ax.axhline(0.0, color="#dddddd", linewidth=0.8)
        ax.axvline(0.0, color="#dddddd", linewidth=0.8)
    pad = 0.03 if lo >= 0 else 0.06
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    safe_mkdir(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_planes(summary_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    plots_dir = safe_mkdir(out_dir / "plots" / "planes")
    plane_specs = [
        (
            M_POSITIONAL,
            M_SYMBOLIC,
            "positional_score",
            "symbolic_score",
            "Symbolic vs positional head scores",
            "x_plane",
            0.0,
            1.0,
        ),
        (
            M_PE_INVARIANT,
            M_PE_EQUIVARIANT,
            "pe_invariance",
            "pe_equivariance",
            "PE equivariance vs invariance head scores",
            "pe_plane",
            0.0,
            1.0,
        ),
        (
            M_POSITIONAL_CENTERED,
            M_SYMBOLIC_CENTERED,
            "positional_score_centered",
            "symbolic_score_centered",
            "Centered symbolic vs positional head scores",
            "x_plane_centered",
            -1.0,
            1.0,
        ),
        (
            M_PE_INVARIANT_CENTERED,
            M_PE_EQUIVARIANT_CENTERED,
            "pe_invariance_centered",
            "pe_equivariance_centered",
            "Centered PE equivariance vs invariance head scores",
            "pe_plane_centered",
            -1.0,
            1.0,
        ),
    ]
    for x_metric, y_metric, x_label, y_label, title, slug, lo, hi in plane_specs:
        records = make_plane_records(summary_rows, x_metric, y_metric)
        if not records:
            continue
        for model in sorted({str(r["model"]) for r in records}):
            sub = [r for r in records if str(r["model"]) == model]
            scatter_plane(
                sub,
                plots_dir / "per_model" / f"{slug}__{slugify(model)}.png",
                f"{model}: {title}",
                x_label,
                y_label,
                color_key="layer",
                lo=lo,
                hi=hi,
            )
        scatter_plane(
            records,
            plots_dir / f"combined__{slug}__by_student.png",
            f"All selected models: {title}",
            x_label,
            y_label,
            color_key="student",
            lo=lo,
            hi=hi,
        )
        scatter_plane(
            records,
            plots_dir / f"combined__{slug}__by_teacher.png",
            f"All selected models: {title}",
            x_label,
            y_label,
            color_key="teacher",
            lo=lo,
            hi=hi,
        )
        plot_teacher_facets(records, plots_dir / f"facets_by_teacher__{slug}.png", title, x_label, y_label, lo, hi)


def plot_teacher_facets(
    records: Sequence[Mapping[str, Any]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    lo: float,
    hi: float,
) -> None:
    if not records:
        return
    plt = import_plotting()
    teachers = [t for t in TEACHERS if any(str(r.get("teacher")) == t for r in records)]
    if not teachers:
        return
    students = [s for s in ATTENTION_STUDENTS if any(str(r.get("student")) == s for r in records)]
    ncols = min(4, len(teachers))
    nrows = int(math.ceil(len(teachers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 4.0 * nrows), squeeze=False)
    cmap = plt.get_cmap("tab10")
    color = {s: cmap(i % cmap.N) for i, s in enumerate(students)}
    for ax in axes.flat:
        ax.axis("off")
    for idx, teacher in enumerate(teachers):
        ax = axes.flat[idx]
        ax.axis("on")
        sub_t = [r for r in records if str(r.get("teacher")) == teacher]
        for student in students:
            sub = [r for r in sub_t if str(r.get("student")) == student]
            if not sub:
                continue
            ax.scatter(
                [float(r["x"]) for r in sub],
                [float(r["y"]) for r in sub],
                color=color[student],
                s=38,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.25,
                label=student,
            )
        ref_y = [lo, hi] if lo < 0 else [hi, lo]
        ax.plot([lo, hi], ref_y, color="grey", linestyle="--", linewidth=0.8)
        if lo < 0:
            ax.axhline(0.0, color="#dddddd", linewidth=0.7)
            ax.axvline(0.0, color="#dddddd", linewidth=0.7)
        pad = 0.03 if lo >= 0 else 0.06
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_title(teacher, fontsize=10)
        ax.grid(alpha=0.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), fontsize=8)
    fig.suptitle(title, y=0.995)
    fig.supxlabel(x_label)
    fig.supylabel(y_label)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    safe_mkdir(output_path.parent)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_model_metric_heatmaps(summary_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    plt = import_plotting()
    plots_dir = safe_mkdir(out_dir / "plots" / "per_model_heatmaps")
    metric_specs = [
        (M_POSITIONAL, "Positional score", 0.0, 1.0, "Reds"),
        (M_SYMBOLIC, "Symbolic score", 0.0, 1.0, "Greens"),
        (M_PE_INVARIANT, "PE invariance", 0.0, 1.0, "Purples"),
        (M_PE_EQUIVARIANT, "PE equivariance", 0.0, 1.0, "Oranges"),
        (M_POSITIONAL_CENTERED, "Centered positional score", -1.0, 1.0, "coolwarm"),
        (M_SYMBOLIC_CENTERED, "Centered symbolic score", -1.0, 1.0, "coolwarm"),
        (M_ENTROPY, "Attention entropy", 0.0, 1.0, "Blues"),
        (M_RELABEL_EQUIVARIANT, "Relabel equivariance", 0.0, 1.0, "Greys"),
    ]
    models = sorted({str(r.get("model")) for r in summary_rows})
    for model in models:
        model_rows = [r for r in summary_rows if str(r.get("model")) == model]
        for metric, title, vmin, vmax, cmap in metric_specs:
            sub = [r for r in model_rows if r.get("metric") == metric and is_number_like(r.get("mean"))]
            if not sub:
                continue
            layers = sorted({int(r["layer"]) for r in sub})
            heads = sorted({int(r["head"]) for r in sub})
            arr = np.full((len(layers), len(heads)), np.nan, dtype=np.float32)
            li = {v: i for i, v in enumerate(layers)}
            hi = {v: i for i, v in enumerate(heads)}
            for r in sub:
                arr[li[int(r["layer"])], hi[int(r["head"])]] = float(r["mean"])
            fig, ax = plt.subplots(figsize=(max(4.8, 0.75 * len(heads) + 2), max(3.8, 0.45 * len(layers) + 2)))
            im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(heads)))
            ax.set_xticklabels(heads)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels(layers)
            ax.set_xlabel("Head")
            ax.set_ylabel("Layer")
            ax.set_title(f"{model}: {title}")
            for i in range(len(layers)):
                for j in range(len(heads)):
                    if np.isfinite(arr[i, j]):
                        ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(plots_dir / f"{slugify(model)}__{metric}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)


def plot_teacher_student_summary_heatmaps(summary_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    plt = import_plotting()
    plots_dir = safe_mkdir(out_dir / "plots" / "teacher_student_summary")
    metric_specs = [
        (M_POSITIONAL, "Best positional score by teacher/student", 0.0, 1.0, "Reds", "max"),
        (M_SYMBOLIC, "Best symbolic score by teacher/student", 0.0, 1.0, "Greens", "max"),
        (M_PE_INVARIANT, "Best PE-invariance by teacher/student", 0.0, 1.0, "Purples", "max"),
        (M_PE_EQUIVARIANT, "Best PE-equivariance by teacher/student", 0.0, 1.0, "Oranges", "max"),
        (M_ENTROPY, "Mean entropy by teacher/student", 0.0, 1.0, "Blues", "mean"),
        (M_RELABEL_EQUIVARIANT, "Mean relabel equivariance by teacher/student", 0.0, 1.0, "Greys", "mean"),
    ]
    teachers = [t for t in TEACHERS if any(str(r.get("teacher")) == t for r in summary_rows)]
    students = [s for s in ATTENTION_STUDENTS if any(str(r.get("student")) == s for r in summary_rows)]
    if not teachers or not students:
        return
    for metric, title, vmin, vmax, cmap, reduce in metric_specs:
        arr = np.full((len(teachers), len(students)), np.nan, dtype=np.float32)
        for i, teacher in enumerate(teachers):
            for j, student in enumerate(students):
                vals = [
                    float(r["mean"])
                    for r in summary_rows
                    if r.get("metric") == metric
                    and str(r.get("teacher")) == teacher
                    and str(r.get("student")) == student
                    and is_number_like(r.get("mean"))
                ]
                if vals:
                    arr[i, j] = float(np.max(vals) if reduce == "max" else np.mean(vals))
        if not np.isfinite(arr).any():
            continue
        fig, ax = plt.subplots(figsize=(max(8.0, len(students) * 1.25), max(4.8, len(teachers) * 0.55 + 1.5)))
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(students)))
        ax.set_xticklabels(students, rotation=35, ha="right")
        ax.set_yticks(range(len(teachers)))
        ax.set_yticklabels(teachers)
        ax.set_title(title)
        for i in range(len(teachers)):
            for j in range(len(students)):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.82)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric}__teacher_student_heatmap.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def plot_layerwise_curves(summary_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    plt = import_plotting()
    plots_dir = safe_mkdir(out_dir / "plots")
    metric_specs = [
        (M_POSITIONAL, "Positional score", 0.0, 1.0),
        (M_SYMBOLIC, "Symbolic score", 0.0, 1.0),
        (M_PE_INVARIANT, "PE invariance", 0.0, 1.0),
        (M_PE_EQUIVARIANT, "PE equivariance", 0.0, 1.0),
        (M_POSITIONAL_CENTERED, "Centered positional", -1.0, 1.0),
        (M_SYMBOLIC_CENTERED, "Centered symbolic", -1.0, 1.0),
        (M_ENTROPY, "Attention entropy", 0.0, 1.0),
        (M_RELABEL_EQUIVARIANT, "Relabel equivariance", 0.0, 1.0),
    ]
    students = [s for s in ATTENTION_STUDENTS if any(str(r.get("student")) == s for r in summary_rows)]
    if not students:
        return
    ncols = 4
    nrows = int(math.ceil(len(metric_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.7 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (metric, title, ymin, ymax) in zip(axes.flat, metric_specs):
        ax.axis("on")
        for student in students:
            sub = [
                r
                for r in summary_rows
                if r.get("metric") == metric
                and str(r.get("student")) == student
                and is_number_like(r.get("mean"))
            ]
            if not sub:
                continue
            layers = sorted({int(r["layer"]) for r in sub})
            means = []
            stds = []
            for layer in layers:
                vals = np.asarray([float(r["mean"]) for r in sub if int(r["layer"]) == layer], dtype=np.float64)
                means.append(float(vals.mean()))
                stds.append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)
            x = np.asarray(layers, dtype=np.float64)
            y = np.asarray(means, dtype=np.float64)
            s = np.asarray(stds, dtype=np.float64)
            ax.plot(x, y, marker="o", linewidth=1.5, label=student)
            ax.fill_between(x, y - s, y + s, alpha=0.13)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), fontsize=8)
    fig.suptitle("Layer-wise metric trends by student architecture", y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(plots_dir / "layerwise_curves_by_student.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_plots(summary_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    if not summary_rows:
        return
    plot_planes(summary_rows, out_dir)
    plot_model_metric_heatmaps(summary_rows, out_dir)
    plot_teacher_student_summary_heatmaps(summary_rows, out_dir)
    plot_layerwise_curves(summary_rows, out_dir)


# -----------------------------------------------------------------------------
# CLI.
# -----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/content/drive/MyDrive/graph_operator_distillation/paper_lite_4seed"),
        help="Root containing seed*/runs/<teacher>/<student> outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input-root>/synthetic_specialisation_metrics.",
    )
    parser.add_argument("--teachers", nargs="+", choices=TEACHERS, default=None)
    parser.add_argument("--students", nargs="+", choices=STUDENTS, default=None)
    parser.add_argument("--num-perms", type=int, default=128)
    parser.add_argument("--num-graphs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--eval-nodes",
        type=str,
        default="id",
        help="'id', 'ood', or an integer graph size. Default uses selected run's ID eval size.",
    )
    parser.add_argument("--metric-alpha-tau", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-models", type=int, default=None, help="Debug limit after best-run selection.")
    parser.add_argument("--no-relabel-sanity", action="store_true", help="Skip relabel equivariance sanity metric.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV outputs.")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[args] ignoring unknown args: {unknown}", flush=True)
    if args.out_dir is None:
        args.out_dir = args.input_root / "synthetic_specialisation_metrics"
    return args


def main(argv: Sequence[str] | None = None) -> None:
    warnings.filterwarnings("ignore", message=r".*TypedStorage is deprecated.*")
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = safe_mkdir(args.out_dir)
    print(f"[run] input_root={args.input_root}", flush=True)
    print(f"[run] out_dir={out_dir}", flush=True)
    print(f"[run] device={device}", flush=True)
    print(f"[run] num_graphs={args.num_graphs} num_perms={args.num_perms} eval_nodes={args.eval_nodes}", flush=True)

    discovered = discover_runs(args.input_root, args.teachers, args.students)
    print(f"[discover] checkpoint-backed candidate runs={len(discovered)}", flush=True)
    selected = select_best_runs(discovered)
    print(f"[discover] selected best runs={len(selected)}", flush=True)
    if args.max_models is not None:
        selected = selected[: int(args.max_models)]
    if not selected:
        diag = discovery_diagnostics(args.input_root)
        write_json(out_dir / "discovery_diagnostics.json", diag)
        print("[discover] no runs selected. Diagnostics:", flush=True)
        print(json.dumps(diag, indent=2), flush=True)
        raise FileNotFoundError(
            f"No completed synthetic runs found under {args.input_root}. "
            f"See {out_dir / 'discovery_diagnostics.json'} for path examples."
        )

    selected_csv_rows = []
    for row in selected:
        slim = {k: v for k, v in row.items() if k != "summary"}
        selected_csv_rows.append(slim)
    write_csv_rows(out_dir / "selected_runs.csv", selected_csv_rows)
    write_json(
        out_dir / "analysis_config.json",
        {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "attention_students": ATTENTION_STUDENTS,
            "local_gnn_note": "Skipped for head-specialisation metrics because it has no attention heads.",
        },
    )

    metric_rows_path = out_dir / "metric_rows.csv"
    summary_path = out_dir / "metric_summary.csv"
    if metric_rows_path.exists() and summary_path.exists() and not args.force:
        print(f"[skip] Existing outputs found at {out_dir}; pass --force to recompute.", flush=True)
        return

    all_rows: list[dict[str, Any]] = []
    model_metas: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        teacher = str(row["teacher"])
        student = str(row["student"])
        print(f"\n[{idx}/{len(selected)}] {teacher}/{student}", flush=True)
        if student not in ATTENTION_STUDENTS:
            reason = str(row.get("not_applicable_reason") or "not attention-bearing")
            print(f"  [skip] {reason}", flush=True)
            skipped.append(
                {
                    "teacher": teacher,
                    "student": student,
                    "selected_seed": row.get("seed"),
                    "reason": reason,
                    "checkpoint_path": row.get("checkpoint_path"),
                }
            )
            continue
        try:
            rows, meta = run_metrics_for_model(row, args, device)
        except Exception as exc:
            print(f"  [error] {teacher}/{student}: {exc}", flush=True)
            skipped.append(
                {
                    "teacher": teacher,
                    "student": student,
                    "selected_seed": row.get("seed"),
                    "reason": f"metric computation failed: {exc}",
                    "checkpoint_path": row.get("checkpoint_path"),
                }
            )
            continue
        all_rows.extend(rows)
        model_metas.append(meta)
        print(f"  [done] rows={len(rows)} elapsed={meta['elapsed_sec']:.1f}s", flush=True)
        write_csv_rows(out_dir / "metric_rows_partial.csv", all_rows)
        write_csv_rows(out_dir / "skipped_models.csv", skipped)

    summary_rows = summarise_metric_rows(all_rows)
    write_csv_rows(metric_rows_path, all_rows)
    write_csv_rows(summary_path, summary_rows)
    write_csv_rows(out_dir / "skipped_models.csv", skipped)
    write_json(out_dir / "model_metadata.json", {"models": model_metas})
    print("\n[plots] writing figures", flush=True)
    make_plots(summary_rows, out_dir)
    print(f"[done] wrote outputs to {out_dir}", flush=True)


CELL_ARGS: list[str] = []


if __name__ == "__main__":
    main(CELL_ARGS if CELL_ARGS else None)
