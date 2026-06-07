#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast relation-rank tests for Graphormer-style structural routing.

This standalone runner is designed for the cleanest first experiment behind the
claim:

    Graphormer can approximate relation-conditioned message passing only through
    a finite head basis, unless deeper layers build latent structural node
    states.

The teacher is a one-hop relation-conditioned operator

    y_i = (1 / sqrt(R)) sum_{r=1}^R W_r x_{j(i,r)}

where each query node i has exactly one incoming neighbour j(i,r) for every
relation type r. This removes degree/count/support confounds: the only hard
part is applying a different value transform per relation.

Main students:

* graphormer_struct_support:
    Oracle local support, scalar pair-bias per head, no content qk. This is the
    clean head-basis approximation test.
* graphormer_support:
    Same support and pair-bias, but with normal content qk terms.
* csa_bias_support:
    Oracle local support with channel-wise structural filters, but no
    pair-conditioned value transform.
* rel_value_support:
    Oracle local support with explicit relation-conditioned value transforms.
* edge_gnn:
    Local relation-conditioned message-passing baseline.
* graphormer_dense:
    Dense support Graphormer-style variant. Useful as a confounded realism check,
    not as the clean capacity test.

Outputs:

* summary.csv with output and attention diagnostics.
* one JSON summary per run.
* relation_mass.csv per run: average attention mass by head and relation.
* optional checkpoints.
* paper_figures/: compact figures intended for the main dissertation text.
* appendix_figures/: heatmaps and per-run head/relation diagnostics.
* findings_summary.md: automatically generated quantitative interpretation.

Colab examples:

    !python graphormer_relation_rank_sweep.py --preset smoke
    !python graphormer_relation_rank_sweep.py --preset fast --mount-drive
    !python graphormer_relation_rank_sweep.py --preset full --layers-grid 1 2 4

The one-layer fast preset should run quickly on a Colab GPU. Use the depth grid
only after the one-layer diagnostic behaves as expected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


STUDENTS = (
    "graphormer_struct_support",
    "graphormer_support",
    "csa_bias_support",
    "rel_value_support",
    "edge_gnn",
    "vanilla_support",
    "graphormer_dense",
)


DEFAULT_FAST_STUDENTS = (
    "graphormer_struct_support",
    "graphormer_support",
    "csa_bias_support",
    "rel_value_support",
    "edge_gnn",
)


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    num_nodes: int = 32
    input_dim: int = 16
    hidden_dim: int = 64
    target_dim: int = 16
    layers: int = 1
    heads: int = 4
    relation_types: int = 8
    batch_size: int = 64
    eval_batch_size: int = 64
    steps: int = 350
    eval_graphs: int = 128
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-5
    grad_clip: float = 1.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    warmup_steps: int = 50


@dataclass
class Batch:
    x: torch.Tensor
    edge_type: torch.Tensor
    pair: torch.Tensor
    support_mask: torch.Tensor
    node_mask: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            x=self.x.to(device),
            edge_type=self.edge_type.to(device),
            pair=self.pair.to(device),
            support_mask=self.support_mask.to(device),
            node_mask=self.node_mask.to(device),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def maybe_mount_drive(enabled: bool) -> None:
    if not enabled:
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception as exc:
        print(f"[mount] skipped: {exc}", flush=True)


def default_output_root() -> Path:
    if Path("/content/drive/MyDrive").exists():
        return Path("/content/drive/MyDrive/graph_operator_distillation/relation_rank_sweep")
    return Path("relation_rank_sweep_results")


def one_hot(values: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(values.clamp(0, num_classes - 1), num_classes=num_classes).to(torch.float32)


def make_batch(
    cfg: RunConfig,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> Batch:
    """Generate directed local relation graphs with one source per relation."""
    n = cfg.num_nodes
    r = cfg.relation_types
    if r >= n:
        raise ValueError(f"relation_types ({r}) must be < num_nodes ({n})")

    x = torch.randn(batch_size, n, cfg.input_dim, generator=generator)
    scores = torch.rand(batch_size, n, n, generator=generator)
    idx = torch.arange(n)
    scores[:, idx, idx] = -1.0
    src = scores.topk(r, dim=-1).indices  # [B,N,R], one key per relation.

    edge_type = torch.zeros(batch_size, n, n, dtype=torch.long)
    rel_values = torch.arange(1, r + 1, dtype=torch.long).view(1, 1, r).expand(batch_size, n, r)
    edge_type.scatter_(2, src, rel_values)

    pair = one_hot(edge_type, r + 1)
    support_mask = edge_type > 0
    node_mask = torch.ones(batch_size, n, dtype=torch.bool)
    return Batch(x=x, edge_type=edge_type, pair=pair, support_mask=support_mask, node_mask=node_mask).to(device)


class RelationTeacher(nn.Module):
    def __init__(self, cfg: RunConfig, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(1_000_003 + 97 * seed + 17 * cfg.relation_types)
        weights = torch.randn(cfg.relation_types, cfg.input_dim, cfg.target_dim, generator=gen)
        if cfg.input_dim == cfg.target_dim:
            orthogonal = []
            for rel in range(cfg.relation_types):
                q, _ = torch.linalg.qr(weights[rel])
                orthogonal.append(q)
            weights = torch.stack(orthogonal, dim=0)
        else:
            weights = weights / math.sqrt(cfg.input_dim)
        self.register_buffer("weights", weights)

    def forward(self, batch: Batch) -> torch.Tensor:
        r = self.weights.size(0)
        edge_rel = one_hot(batch.edge_type, r + 1)[..., 1:]  # [B,N,N,R]
        xw = torch.einsum("bjf,rfo->bjro", batch.x, self.weights)  # [B,N,R,Dout]
        return torch.einsum("bijr,bjro->bio", edge_rel, xw) / math.sqrt(float(r))

    def diagnostics(self) -> dict[str, float]:
        flat = self.weights.flatten(1)
        s = torch.linalg.svdvals(flat)
        rank = int((s > 1.0e-5 * s.max().clamp_min(1.0e-8)).sum().item())
        norm = F.normalize(flat, dim=-1)
        if flat.size(0) > 1:
            cos = norm @ norm.T
            off_diag = cos[~torch.eye(flat.size(0), dtype=torch.bool, device=cos.device)]
            mean_abs_cos = float(off_diag.abs().mean().detach().cpu())
        else:
            mean_abs_cos = 0.0
        return {
            "teacher_relation_rank": float(rank),
            "teacher_mean_abs_transform_cosine": mean_abs_cos,
            "teacher_top_singular_value": float(s.max().detach().cpu()),
            "teacher_min_singular_value": float(s.min().detach().cpu()),
        }


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits.masked_fill(~mask, -1.0e9)
    return torch.softmax(logits, dim=dim)


class GraphormerBlock(nn.Module):
    def __init__(
        self,
        cfg: RunConfig,
        structural_only: bool,
        hard_support: bool,
        no_pair_bias: bool = False,
    ) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.cfg = cfg
        self.structural_only = structural_only
        self.hard_support = hard_support
        self.no_pair_bias = no_pair_bias
        self.head_dim = cfg.hidden_dim // cfg.heads
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim, bias=False)
        self.pair_bias = nn.Linear(cfg.relation_types + 1, cfg.heads, bias=False)
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.attn_dropout)
        self.ffn = FeedForward(cfg.hidden_dim, cfg.dropout)

    def _mask(self, batch: Batch) -> torch.Tensor:
        if self.hard_support:
            return batch.support_mask[:, None, :, :]
        bsz, n = batch.edge_type.shape[:2]
        eye = torch.eye(n, dtype=torch.bool, device=batch.edge_type.device)[None]
        return ((~eye).expand(bsz, -1, -1) & batch.node_mask[:, :, None] & batch.node_mask[:, None, :])[:, None]

    def forward(self, h: torch.Tensor, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n, dim = h.shape
        x = self.norm(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        if self.structural_only:
            logits = torch.zeros(bsz, self.cfg.heads, n, n, dtype=h.dtype, device=h.device)
        else:
            logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if not self.no_pair_bias:
            logits = logits + self.pair_bias(batch.pair).permute(0, 3, 1, 2)
        attn = masked_softmax(logits, self._mask(batch))
        ctx = torch.matmul(self.drop(attn), v)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, n, dim)
        h = h + self.out(ctx)
        h = self.ffn(h)
        return h, attn.detach()


class CSABiasBlock(nn.Module):
    """Channel-wise structural routing without pair-conditioned values."""

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.cfg = cfg
        self.head_dim = cfg.hidden_dim // cfg.heads
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim, bias=False)
        self.filter = nn.Linear(cfg.relation_types + 1, cfg.hidden_dim, bias=False)
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.attn_dropout)
        self.ffn = FeedForward(cfg.hidden_dim, cfg.dropout)

    def forward(self, h: torch.Tensor, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n, dim = h.shape
        x = self.norm(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        qk = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        filt = self.filter(batch.pair).view(bsz, n, n, self.cfg.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        logits = qk[..., None] + filt
        mask = batch.support_mask[:, None, :, :, None]
        alpha = masked_softmax(logits, mask, dim=3)
        msg = v[:, :, None, :, :].expand(-1, -1, n, -1, -1)
        ctx = (self.drop(alpha) * msg).sum(dim=3)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, n, dim)
        h = h + self.out(ctx)
        h = self.ffn(h)
        return h, alpha.detach().mean(dim=-1)


class RelationValueBlock(nn.Module):
    """Explicit relation-conditioned value transform under local support."""

    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        if cfg.hidden_dim % cfg.heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.cfg = cfg
        self.head_dim = cfg.hidden_dim // cfg.heads
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.qkv = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim, bias=False)
        self.pair_bias = nn.Linear(cfg.relation_types + 1, cfg.heads, bias=False)
        kernel = torch.eye(self.head_dim).view(1, 1, self.head_dim, self.head_dim)
        kernel = kernel.repeat(cfg.relation_types + 1, cfg.heads, 1, 1)
        kernel = kernel + 0.02 * torch.randn_like(kernel)
        self.rel_kernel = nn.Parameter(kernel)
        self.out = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.drop = nn.Dropout(cfg.attn_dropout)
        self.ffn = FeedForward(cfg.hidden_dim, cfg.dropout)

    def forward(self, h: torch.Tensor, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n, dim = h.shape
        x = self.norm(h)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, n, self.cfg.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits + self.pair_bias(batch.pair).permute(0, 3, 1, 2)
        attn = masked_softmax(logits, batch.support_mask[:, None, :, :])
        kernels = self.rel_kernel[batch.edge_type]  # [B,N,N,H,D,D]
        msg = torch.einsum("bhjd,bijhde->bhije", v, kernels)
        ctx = torch.einsum("bhij,bhijd->bhid", self.drop(attn), msg)
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, n, dim)
        h = h + self.out(ctx)
        h = self.ffn(h)
        return h, attn.detach()


class EdgeGNNLayer(nn.Module):
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.rel = nn.Parameter(
            torch.randn(cfg.relation_types + 1, cfg.hidden_dim, cfg.hidden_dim) / math.sqrt(cfg.hidden_dim)
        )
        self.self_lin = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.ffn = FeedForward(cfg.hidden_dim, cfg.dropout)

    def forward(self, h: torch.Tensor, batch: Batch) -> torch.Tensor:
        x = self.norm(h)
        kernels = self.rel[batch.edge_type]  # [B,N,N,D,D]
        msg = torch.einsum("bjd,bijde->bije", x, kernels)
        msg = (msg * batch.support_mask.unsqueeze(-1)).sum(dim=2) / math.sqrt(float(self.cfg.relation_types))
        h = h + self.self_lin(x) + msg
        return self.ffn(h)


class StudentModel(nn.Module):
    def __init__(self, cfg: RunConfig, name: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.encoder = nn.Sequential(nn.Linear(cfg.input_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU())
        layers: list[nn.Module] = []
        for _ in range(cfg.layers):
            if name == "graphormer_struct_support":
                layers.append(GraphormerBlock(cfg, structural_only=True, hard_support=True))
            elif name == "graphormer_support":
                layers.append(GraphormerBlock(cfg, structural_only=False, hard_support=True))
            elif name == "graphormer_dense":
                layers.append(GraphormerBlock(cfg, structural_only=False, hard_support=False))
            elif name == "vanilla_support":
                layers.append(GraphormerBlock(cfg, structural_only=False, hard_support=True, no_pair_bias=True))
            elif name == "csa_bias_support":
                layers.append(CSABiasBlock(cfg))
            elif name == "rel_value_support":
                layers.append(RelationValueBlock(cfg))
            elif name == "edge_gnn":
                layers.append(EdgeGNNLayer(cfg))
            else:
                raise ValueError(f"unknown student {name}")
        self.layers = nn.ModuleList(layers)
        self.out = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, cfg.target_dim))

    def forward(self, batch: Batch, collect_attention: bool = False) -> tuple[torch.Tensor, dict[str, object]]:
        h = self.encoder(batch.x)
        attentions: list[torch.Tensor] = []
        for layer in self.layers:
            if isinstance(layer, EdgeGNNLayer):
                h = layer(h, batch)
            else:
                h, attn = layer(h, batch)  # type: ignore[misc]
                if collect_attention:
                    attentions.append(attn)
        return self.out(h), {"attentions": attentions}


def build_student(cfg: RunConfig, name: str, device: torch.device) -> StudentModel:
    return StudentModel(cfg, name).to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def last_attention(info: Mapping[str, object]) -> torch.Tensor | None:
    attns = info.get("attentions")
    if not isinstance(attns, list) or not attns:
        return None
    attn = attns[-1]
    if isinstance(attn, torch.Tensor):
        return attn
    return None


def relation_attention_diagnostics(
    attn: torch.Tensor | None,
    edge_type: torch.Tensor,
    relation_types: int,
) -> tuple[dict[str, float | str], list[dict[str, object]]]:
    if attn is None:
        return (
            {
                "attn_real_edge_mass": float("nan"),
                "attn_nonedge_mass": float("nan"),
                "head_relation_purity": float("nan"),
                "head_relation_specialization": float("nan"),
                "relation_coverage": float("nan"),
                "relation_mass_rank": float("nan"),
                "relation_mass_singular_values": "",
            },
            [],
        )

    with torch.no_grad():
        bsz, heads, n, _ = attn.shape
        rel_rows: list[dict[str, object]] = []
        mass = []
        for rel in range(1, relation_types + 1):
            mask = (edge_type == rel).to(attn.dtype)[:, None, :, :]
            by_head = (attn * mask).sum(dim=-1).mean(dim=(0, 2))
            mass.append(by_head)
        mass_t = torch.stack(mass, dim=1)  # [H,R]
        nonedge = (edge_type == 0).to(attn.dtype)[:, None, :, :]
        real = (edge_type > 0).to(attn.dtype)[:, None, :, :]
        nonedge_mass = (attn * nonedge).sum(dim=-1).mean()
        real_mass = (attn * real).sum(dim=-1).mean()

        p = mass_t / mass_t.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
        purity = p.max(dim=1).values.mean()
        if relation_types > 1:
            entropy = -(p.clamp_min(1.0e-12) * p.clamp_min(1.0e-12).log()).sum(dim=1) / math.log(float(relation_types))
            specialization = 1.0 - entropy.mean()
        else:
            specialization = torch.tensor(1.0, device=attn.device)
        coverage = float(torch.unique(p.argmax(dim=1)).numel())
        s = torch.linalg.svdvals(mass_t.detach().cpu())
        rank = int((s > 0.05 * s.max().clamp_min(1.0e-12)).sum().item())

        for h in range(heads):
            for rel in range(1, relation_types + 1):
                rel_rows.append(
                    {
                        "head": h,
                        "relation": rel,
                        "attention_mass": float(mass_t[h, rel - 1].detach().cpu()),
                        "normalized_attention_mass": float(p[h, rel - 1].detach().cpu()),
                    }
                )

        return (
            {
                "attn_real_edge_mass": float(real_mass.detach().cpu()),
                "attn_nonedge_mass": float(nonedge_mass.detach().cpu()),
                "head_relation_purity": float(purity.detach().cpu()),
                "head_relation_specialization": float(specialization.detach().cpu()),
                "relation_coverage": coverage,
                "relation_mass_rank": float(rank),
                "relation_mass_singular_values": ";".join(f"{float(v):.6g}" for v in s),
            },
            rel_rows,
        )


@torch.no_grad()
def evaluate(
    cfg: RunConfig,
    teacher: RelationTeacher,
    model: StudentModel,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, float | str], list[dict[str, object]]]:
    model.eval()
    total = 0
    mse_sum = 0.0
    mae_sum = 0.0
    rel_sum = 0.0
    cos_sum = 0.0
    diag_accum: list[dict[str, float | str]] = []
    rel_rows: list[dict[str, object]] = []
    gen = torch.Generator().manual_seed(seed)
    for offset in range(0, cfg.eval_graphs, cfg.eval_batch_size):
        current = min(cfg.eval_batch_size, cfg.eval_graphs - offset)
        batch = make_batch(cfg, current, gen, device)
        target = teacher(batch)
        pred, info = model(batch, collect_attention=True)
        mse_per = ((pred - target) ** 2).mean(dim=(1, 2))
        mae_per = (pred - target).abs().mean(dim=(1, 2))
        var_per = target.var(dim=(1, 2), unbiased=False).clamp_min(1.0e-8)
        cos_per = F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=-1)
        mse_sum += float(mse_per.sum().detach().cpu())
        mae_sum += float(mae_per.sum().detach().cpu())
        rel_sum += float((mse_per / var_per).sum().detach().cpu())
        cos_sum += float(cos_per.sum().detach().cpu())
        total += current
        if offset == 0:
            diag, rows = relation_attention_diagnostics(last_attention(info), batch.edge_type, cfg.relation_types)
            diag_accum.append(diag)
            rel_rows = rows

    metrics: dict[str, float | str] = {
        "output_mse": mse_sum / max(1, total),
        "output_mae": mae_sum / max(1, total),
        "relative_mse": rel_sum / max(1, total),
        "output_cosine": cos_sum / max(1, total),
    }
    if diag_accum:
        metrics.update(diag_accum[0])
    return metrics, rel_rows


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: RunConfig):
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return max(1.0e-4, float(step + 1) / max(1, cfg.warmup_steps))
        progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one(
    cfg: RunConfig,
    student_name: str,
    output_root: Path,
    device: torch.device,
    force: bool,
    save_model: bool,
) -> dict[str, object]:
    run_name = f"{student_name}_R{cfg.relation_types}_H{cfg.heads}_L{cfg.layers}_seed{cfg.seed}"
    run_dir = output_root / "runs" / run_name
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        print(f"[skip] {run_name}", flush=True)
        return read_json(summary_path)

    set_seed(cfg.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    teacher = RelationTeacher(cfg, cfg.seed).to(device)
    model = build_student(cfg, student_name, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg)
    gen = torch.Generator().manual_seed(7_000_001 + cfg.seed)
    log_rows: list[dict[str, object]] = []
    start = time.time()

    for step in range(1, cfg.steps + 1):
        model.train()
        batch = make_batch(cfg, cfg.batch_size, gen, device)
        target = teacher(batch)
        pred, _ = model(batch, collect_attention=False)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        if step == 1 or step == cfg.steps or step % max(1, cfg.steps // 5) == 0:
            log_rows.append(
                {
                    "step": step,
                    "train_mse": float(loss.detach().cpu()),
                    "lr": float(scheduler.get_last_lr()[0]),
                }
            )

    eval_metrics, rel_rows = evaluate(cfg, teacher, model, device, seed=900_000 + cfg.seed)
    elapsed = time.time() - start
    write_csv(run_dir / "train_log.csv", log_rows)
    write_csv(run_dir / "relation_mass.csv", rel_rows)
    if save_model:
        torch.save(
            {
                "cfg": asdict(cfg),
                "student": student_name,
                "model_state_dict": model.state_dict(),
                "teacher_weights": teacher.weights.detach().cpu(),
            },
            run_dir / "checkpoint.pt",
        )
    summary: dict[str, object] = {
        "student": student_name,
        "seed": cfg.seed,
        "num_nodes": cfg.num_nodes,
        "relation_types": cfg.relation_types,
        "heads": cfg.heads,
        "head_dim": cfg.hidden_dim // cfg.heads,
        "head_dim_at_least_io_rank": (cfg.hidden_dim // cfg.heads) >= min(cfg.input_dim, cfg.target_dim),
        "layers": cfg.layers,
        "basis_pressure_R_over_H": float(cfg.relation_types) / float(cfg.heads),
        "hidden_dim": cfg.hidden_dim,
        "parameters": count_parameters(model),
        "steps": cfg.steps,
        "elapsed_sec": elapsed,
        "run_dir": str(run_dir),
        **teacher.diagnostics(),
        **eval_metrics,
    }
    write_json(summary_path, summary)
    print(
        f"[done] {run_name} rel_mse={float(summary['relative_mse']):.4g} "
        f"purity={summary.get('head_relation_purity')} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


def import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def is_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


STUDENT_LABELS = {
    "graphormer_struct_support": "Graphormer\nstructural routing",
    "graphormer_support": "Graphormer\nrouting + QK",
    "csa_bias_support": "CSA bias-only\nrouting",
    "rel_value_support": "Pair-value\ntransport",
    "edge_gnn": "Edge-GNN\ntransport",
    "vanilla_support": "Vanilla\nlocal support",
    "graphormer_dense": "Dense\nGraphormer",
}


STUDENT_COLORS = {
    "graphormer_struct_support": "#4063A3",
    "graphormer_support": "#5F83C2",
    "csa_bias_support": "#B56B45",
    "rel_value_support": "#2F8A5B",
    "edge_gnn": "#277C8E",
    "vanilla_support": "#7B7B7B",
    "graphormer_dense": "#7A4FA3",
}


STUDENT_MARKERS = {
    "graphormer_struct_support": "o",
    "graphormer_support": "s",
    "csa_bias_support": "^",
    "rel_value_support": "D",
    "edge_gnn": "P",
    "vanilla_support": "X",
    "graphormer_dense": "v",
}


def fnum(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def mean_sem(values: Sequence[float]) -> tuple[float, float]:
    vals = np.array([v for v in values if math.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals.mean()), 0.0
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(vals.size))


def group_metric(
    rows: Sequence[Mapping[str, object]],
    student: str,
    x_key: str,
    metric: str,
    filters: Mapping[str, object] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    grouped: dict[float, list[float]] = {}
    for row in rows:
        if row.get("student") != student or not is_number(row.get(x_key)) or not is_number(row.get(metric)):
            continue
        if filters is not None:
            if any(str(row.get(key)) != str(value) for key, value in filters.items()):
                continue
        grouped.setdefault(float(row[x_key]), []).append(float(row[metric]))
    xs = sorted(grouped)
    means: list[float] = []
    sems: list[float] = []
    for x in xs:
        mean, sem = mean_sem(grouped[x])
        means.append(mean)
        sems.append(sem)
    return xs, means, sems


def save_figure(fig, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")


def add_legend_if_any(ax, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(**kwargs)


def sorted_unique_numbers(rows: Sequence[Mapping[str, object]], key: str) -> list[float]:
    vals = sorted({fnum(row.get(key)) for row in rows if is_number(row.get(key))})
    return [v for v in vals if math.isfinite(v)]


def choose_main_layer(rows: Sequence[Mapping[str, object]]) -> int:
    layers = sorted_unique_numbers(rows, "layers")
    return int(layers[0]) if layers else 1


def plot_main_paper_figure(rows: Sequence[Mapping[str, object]], output_root: Path) -> None:
    plt = import_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.35), gridspec_kw={"width_ratios": [1.55, 1.0, 1.0]})
    paper_dir = output_root / "paper_figures"
    layer = choose_main_layer(rows)
    filters = {"layers": layer}
    main_students = [s for s in DEFAULT_FAST_STUDENTS if any(r.get("student") == s for r in rows)]

    ax = axes[0]
    for student in main_students:
        xs, ys, sems = group_metric(rows, student, "basis_pressure_R_over_H", "relative_mse", filters)
        if not xs:
            continue
        ax.errorbar(
            xs,
            ys,
            yerr=sems,
            marker=STUDENT_MARKERS.get(student, "o"),
            color=STUDENT_COLORS.get(student),
            linewidth=1.7,
            markersize=4.5,
            capsize=2,
            label=STUDENT_LABELS.get(student, student),
        )
    ax.set_xlabel("relation pressure $R/H$")
    ax.set_ylabel("relative MSE")
    ax.set_yscale("log")
    ax.set_title("A. Relation-conditioned transport")
    ax.grid(alpha=0.22)
    add_legend_if_any(ax, ncol=1, frameon=False, loc="best", handlelength=1.6)

    ax = axes[1]
    for metric, label, marker, color in [
        ("head_relation_purity", "head purity", "o", "#4063A3"),
        ("relation_coverage", "coverage / min(R,H)", "s", "#B56B45"),
        ("relation_mass_rank", "mass rank / min(R,H)", "^", "#2F8A5B"),
    ]:
        xs, ys, sems = group_metric(rows, "graphormer_struct_support", "basis_pressure_R_over_H", metric, filters)
        if metric in {"relation_coverage", "relation_mass_rank"}:
            norm_ys: list[float] = []
            norm_sems: list[float] = []
            for x, y, sem in zip(xs, ys, sems):
                denom = [
                    min(fnum(row.get("heads")), fnum(row.get("relation_types")))
                    for row in rows
                    if row.get("student") == "graphormer_struct_support"
                    and str(row.get("layers")) == str(layer)
                    and is_number(row.get("basis_pressure_R_over_H"))
                    and abs(float(row["basis_pressure_R_over_H"]) - x) < 1.0e-9
                ]
                d_mean = np.mean([d for d in denom if d > 0]) if denom else float("nan")
                norm_ys.append(y / d_mean if math.isfinite(d_mean) and d_mean > 0 else float("nan"))
                norm_sems.append(sem / d_mean if math.isfinite(d_mean) and d_mean > 0 else float("nan"))
            ys, sems = norm_ys, norm_sems
        if xs:
            ax.errorbar(xs, ys, yerr=sems, marker=marker, color=color, linewidth=1.6, markersize=4.2, capsize=2, label=label)
    ax.set_xlabel("relation pressure $R/H$")
    ax.set_ylabel("head-basis diagnostic")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("B. Graphormer head diversity")
    ax.grid(alpha=0.22)
    add_legend_if_any(ax, frameon=False, loc="best")

    ax = axes[2]
    graph_x, graph_y, _ = group_metric(rows, "graphormer_struct_support", "basis_pressure_R_over_H", "relative_mse", filters)
    rel_x, rel_y, _ = group_metric(rows, "rel_value_support", "basis_pressure_R_over_H", "relative_mse", filters)
    edge_x, edge_y, _ = group_metric(rows, "edge_gnn", "basis_pressure_R_over_H", "relative_mse", filters)
    xs_common = sorted(set(graph_x) & set(rel_x))
    if xs_common:
        rel_lookup = dict(zip(rel_x, rel_y))
        graph_lookup = dict(zip(graph_x, graph_y))
        gap = [graph_lookup[x] / max(rel_lookup[x], 1.0e-12) for x in xs_common]
        ax.plot(xs_common, gap, marker="o", color="#4063A3", linewidth=1.7, markersize=4.5, label="vs pair-value")
    xs_common = sorted(set(graph_x) & set(edge_x))
    if xs_common:
        edge_lookup = dict(zip(edge_x, edge_y))
        graph_lookup = dict(zip(graph_x, graph_y))
        gap = [graph_lookup[x] / max(edge_lookup[x], 1.0e-12) for x in xs_common]
        ax.plot(xs_common, gap, marker="P", color="#277C8E", linewidth=1.7, markersize=4.5, label="vs Edge-GNN")
    ax.axhline(1.0, color="#303030", linewidth=1.0, linestyle="--")
    ax.set_xlabel("relation pressure $R/H$")
    ax.set_ylabel("Graphormer error ratio")
    ax.set_yscale("log")
    ax.set_title("C. Cost of routing-only structure")
    ax.grid(alpha=0.22)
    add_legend_if_any(ax, frameon=False, loc="best")

    fig.suptitle("One-layer relation-rank test of Graphormer structural routing", y=1.03, fontsize=11)
    save_figure(fig, paper_dir / "main_relation_rank")
    plt.close(fig)


def plot_depth_paper_figure(rows: Sequence[Mapping[str, object]], output_root: Path) -> None:
    layers = sorted_unique_numbers(rows, "layers")
    if len(layers) < 2:
        return
    plt = import_plotting()
    paper_dir = output_root / "paper_figures"
    relation_values = sorted_unique_numbers(rows, "relation_types")
    head_values = sorted_unique_numbers(rows, "heads")
    if not relation_values or not head_values:
        return
    relation = int(max(relation_values))
    head = int(min(head_values, key=lambda h: abs(h - np.median(head_values))))
    filters = {"relation_types": relation, "heads": head}
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for student in ["graphormer_struct_support", "graphormer_support", "rel_value_support", "edge_gnn"]:
        if not any(r.get("student") == student for r in rows):
            continue
        xs, ys, sems = group_metric(rows, student, "layers", "relative_mse", filters)
        if not xs:
            continue
        ax.errorbar(
            xs,
            ys,
            yerr=sems,
            marker=STUDENT_MARKERS.get(student, "o"),
            color=STUDENT_COLORS.get(student),
            linewidth=1.7,
            markersize=4.5,
            capsize=2,
            label=STUDENT_LABELS.get(student, student).replace("\n", " "),
        )
    ax.set_xlabel("layers")
    ax.set_ylabel("relative MSE")
    ax.set_yscale("log")
    ax.set_title(f"Depth rescue test (R={relation}, H={head})")
    ax.grid(alpha=0.22)
    add_legend_if_any(ax, frameon=False)
    save_figure(fig, paper_dir / "depth_rescue")
    plt.close(fig)


def pivot_matrix(
    rows: Sequence[Mapping[str, object]],
    student: str,
    metric: str,
    layer: int,
) -> tuple[list[int], list[int], np.ndarray]:
    rels = sorted({int(fnum(r.get("relation_types"))) for r in rows if r.get("student") == student and is_number(r.get("relation_types"))})
    heads = sorted({int(fnum(r.get("heads"))) for r in rows if r.get("student") == student and is_number(r.get("heads"))})
    mat = np.full((len(rels), len(heads)), np.nan, dtype=np.float64)
    for i, rel in enumerate(rels):
        for j, head in enumerate(heads):
            vals = [
                fnum(r.get(metric))
                for r in rows
                if r.get("student") == student
                and str(r.get("layers")) == str(layer)
                and int(fnum(r.get("relation_types"))) == rel
                and int(fnum(r.get("heads"))) == head
                and is_number(r.get(metric))
            ]
            mat[i, j] = mean_sem(vals)[0]
    return rels, heads, mat


def plot_appendix_heatmaps(rows: Sequence[Mapping[str, object]], output_root: Path) -> None:
    plt = import_plotting()
    appendix_dir = output_root / "appendix_figures"
    layer = choose_main_layer(rows)
    available = [s for s in STUDENTS if any(r.get("student") == s and str(r.get("layers")) == str(layer) for r in rows)]
    if not available:
        return
    cols = min(3, len(available))
    rows_n = int(math.ceil(len(available) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.4 * cols, 2.75 * rows_n), squeeze=False)
    finite_vals: list[float] = []
    matrices: dict[str, tuple[list[int], list[int], np.ndarray]] = {}
    for student in available:
        rels, heads, mat = pivot_matrix(rows, student, "relative_mse", layer)
        matrices[student] = (rels, heads, mat)
        finite_vals.extend(np.log10(mat[np.isfinite(mat)]).tolist())
    vmin = np.percentile(finite_vals, 5) if finite_vals else -2.0
    vmax = np.percentile(finite_vals, 95) if finite_vals else 1.0
    im = None
    for idx, student in enumerate(available):
        ax = axes[idx // cols][idx % cols]
        rels, heads, mat = matrices[student]
        arr = np.log10(mat)
        im = ax.imshow(arr, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(STUDENT_LABELS.get(student, student).replace("\n", " "))
        ax.set_xticks(range(len(heads)), [str(h) for h in heads])
        ax.set_yticks(range(len(rels)), [str(r) for r in rels])
        ax.set_xlabel("heads H")
        ax.set_ylabel("relations R")
        for i in range(len(rels)):
            for j in range(len(heads)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2g}", ha="center", va="center", fontsize=6.5, color="white")
    for idx in range(len(available), rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82)
        cbar.set_label("log10(relative MSE)")
    fig.suptitle(f"Appendix: relation-rank performance heatmaps, L={layer}", y=0.99)
    save_figure(fig, appendix_dir / "relative_mse_heatmaps")
    plt.close(fig)

    for metric, title, filename in [
        ("head_relation_purity", "Head relation purity", "graphormer_head_purity_heatmap"),
        ("relation_mass_rank", "Relation mass rank", "graphormer_relation_mass_rank_heatmap"),
        ("attn_nonedge_mass", "Dense Graphormer nonedge mass", "dense_graphormer_nonedge_mass_heatmap"),
    ]:
        student = "graphormer_dense" if metric == "attn_nonedge_mass" else "graphormer_struct_support"
        if not any(r.get("student") == student and is_number(r.get(metric)) for r in rows):
            continue
        rels, heads, mat = pivot_matrix(rows, student, metric, layer)
        fig, ax = plt.subplots(figsize=(4.1, 3.1))
        im2 = ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set_title(f"{title}, L={layer}")
        ax.set_xticks(range(len(heads)), [str(h) for h in heads])
        ax.set_yticks(range(len(rels)), [str(r) for r in rels])
        ax.set_xlabel("heads H")
        ax.set_ylabel("relations R")
        for i in range(len(rels)):
            for j in range(len(heads)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2g}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(im2, ax=ax, shrink=0.85)
        save_figure(fig, appendix_dir / filename)
        plt.close(fig)


def find_representative_runs(rows: Sequence[Mapping[str, object]], output_root: Path, limit: int = 4) -> list[Mapping[str, object]]:
    layer = choose_main_layer(rows)
    candidates = [
        r
        for r in rows
        if r.get("student") in {"graphormer_struct_support", "graphormer_support"}
        and str(r.get("layers")) == str(layer)
        and is_number(r.get("relation_types"))
        and is_number(r.get("heads"))
        and r.get("run_dir")
    ]
    scored: list[tuple[float, float, float, Mapping[str, object]]] = []
    for row in candidates:
        pressure = fnum(row.get("basis_pressure_R_over_H"))
        target = abs(math.log2(max(pressure, 1.0e-9)))
        scored.append((target, -fnum(row.get("relation_types")), fnum(row.get("heads")), row))
    scored.sort()
    picked: list[Mapping[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    for _, _, _, row in scored:
        key = (str(row["student"]), int(fnum(row["relation_types"])), int(fnum(row["heads"])))
        if key in seen:
            continue
        path = Path(str(row["run_dir"])) / "relation_mass.csv"
        if path.exists():
            picked.append(row)
            seen.add(key)
        if len(picked) >= limit:
            break
    return picked


def plot_relation_mass_examples(rows: Sequence[Mapping[str, object]], output_root: Path) -> None:
    examples = find_representative_runs(rows, output_root)
    if not examples:
        return
    plt = import_plotting()
    appendix_dir = output_root / "appendix_figures"
    fig, axes = plt.subplots(1, len(examples), figsize=(3.2 * len(examples), 2.8), squeeze=False)
    im = None
    for ax, row in zip(axes[0], examples):
        path = Path(str(row["run_dir"])) / "relation_mass.csv"
        rel_rows = read_csv(path)
        heads = sorted({int(fnum(r.get("head"))) for r in rel_rows})
        rels = sorted({int(fnum(r.get("relation"))) for r in rel_rows})
        mat = np.full((len(heads), len(rels)), np.nan)
        for rel_row in rel_rows:
            h = heads.index(int(fnum(rel_row.get("head"))))
            rel = rels.index(int(fnum(rel_row.get("relation"))))
            mat[h, rel] = fnum(rel_row.get("normalized_attention_mass"))
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(
            f"{str(row['student']).replace('_', ' ')}\nR={int(fnum(row['relation_types']))}, H={int(fnum(row['heads']))}",
            fontsize=8,
        )
        ax.set_xlabel("relation")
        ax.set_ylabel("head")
        ax.set_xticks(range(len(rels)), [str(r) for r in rels])
        ax.set_yticks(range(len(heads)), [str(h) for h in heads])
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, label="normalised mass")
    fig.suptitle("Appendix: head-by-relation attention mass", y=1.02)
    save_figure(fig, appendix_dir / "relation_mass_examples")
    plt.close(fig)


def plot_summary(rows: Sequence[Mapping[str, object]], output_root: Path) -> None:
    try:
        plot_main_paper_figure(rows, output_root)
        plot_depth_paper_figure(rows, output_root)
        plot_appendix_heatmaps(rows, output_root)
        plot_relation_mass_examples(rows, output_root)
    except Exception as exc:
        print(f"[plot] skipped: {exc}", flush=True)


def best_row(
    rows: Sequence[Mapping[str, object]],
    student: str,
    relation_types: int,
    heads: int,
    layers: int,
    metric: str = "relative_mse",
) -> float:
    vals = [
        fnum(r.get(metric))
        for r in rows
        if r.get("student") == student
        and int(fnum(r.get("relation_types"))) == relation_types
        and int(fnum(r.get("heads"))) == heads
        and int(fnum(r.get("layers"))) == layers
        and is_number(r.get(metric))
    ]
    return mean_sem(vals)[0]


def format_ratio(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if value >= 10:
        return f"{value:.1f}x"
    return f"{value:.2f}x"


def write_findings_summary(rows: Sequence[Mapping[str, object]], output_root: Path) -> None:
    layer = choose_main_layer(rows)
    relation_values = [int(v) for v in sorted_unique_numbers(rows, "relation_types")]
    head_values = [int(v) for v in sorted_unique_numbers(rows, "heads")]
    lines: list[str] = [
        "# Relation-Rank Sweep: Auto-Generated Findings",
        "",
        "This file is generated from `summary.csv`. Treat it as a quantitative draft: inspect figures and rerun with more seeds before quoting exact numbers.",
        "",
        "## Intended Claim",
        "",
        "Graphormer-style structural routing can approximate relation-conditioned message passing only through a finite head basis. Explicit pair-value or edge-conditioned transport should be less sensitive to relation pressure.",
        "",
        "## Main Figure Files",
        "",
        "- `paper_figures/main_relation_rank.pdf`: main three-panel result.",
        "- `paper_figures/depth_rescue.pdf`: generated when multiple layer counts are run.",
        "- `appendix_figures/relative_mse_heatmaps.pdf`: per-student R-by-H heatmaps.",
        "- `appendix_figures/relation_mass_examples.pdf`: representative head-by-relation attention matrices.",
        "",
        "## Quantitative Checks",
        "",
    ]
    if relation_values and head_values:
        max_r = max(relation_values)
        min_h = min(head_values)
        max_h = max(head_values)
        graph_high = best_row(rows, "graphormer_struct_support", max_r, min_h, layer)
        graph_low = best_row(rows, "graphormer_struct_support", max_r, max_h, layer)
        rel_high = best_row(rows, "rel_value_support", max_r, min_h, layer)
        edge_high = best_row(rows, "edge_gnn", max_r, min_h, layer)
        purity_high = best_row(rows, "graphormer_struct_support", max_r, min_h, layer, "head_relation_purity")
        rank_high = best_row(rows, "graphormer_struct_support", max_r, min_h, layer, "relation_mass_rank")
        lines.extend(
            [
                f"- Hardest one-layer setting found: `R={max_r}`, `H={min_h}`, `L={layer}`.",
                f"- Graphormer structural-routing relative MSE there: `{graph_high:.4g}`.",
                f"- Pair-value transport relative MSE there: `{rel_high:.4g}`; Graphormer / pair-value error ratio: `{format_ratio(graph_high / max(rel_high, 1.0e-12))}`.",
                f"- Edge-GNN transport relative MSE there: `{edge_high:.4g}`; Graphormer / Edge-GNN error ratio: `{format_ratio(graph_high / max(edge_high, 1.0e-12))}`.",
                f"- Graphormer head-relation purity there: `{purity_high:.3g}`; relation-mass rank: `{rank_high:.3g}`.",
            ]
        )
        if max_h != min_h:
            lines.append(
                f"- At the same `R={max_r}`, increasing heads from `H={min_h}` to `H={max_h}` changes Graphormer relative MSE from `{graph_high:.4g}` to `{graph_low:.4g}`."
            )
        dense_nonedge_vals = [
            fnum(r.get("attn_nonedge_mass"))
            for r in rows
            if r.get("student") == "graphormer_dense" and int(fnum(r.get("layers"))) == layer and is_number(r.get("attn_nonedge_mass"))
        ]
        if dense_nonedge_vals:
            lines.append(
                f"- Dense Graphormer mean nonedge attention mass: `{np.mean(dense_nonedge_vals):.3g}`. Use this as a support-discovery appendix/control, not the main capacity test."
            )
    lines.extend(
        [
            "",
            "## Interpretation Template",
            "",
            "- If Graphormer error rises with `R/H` while pair-value and Edge-GNN baselines stay lower, the result supports the head-basis bottleneck prediction.",
            "- If Graphormer heads become relation-specialised but performance still degrades, the result supports the claim that relation routing is not equivalent to relation-conditioned transport.",
            "- If additional layers rescue Graphormer, the clean interpretation is not direct pair-value transport but a latent structural node-state path; report this separately from the one-layer result.",
            "- If `graphormer_dense` performs worse or assigns mass to nonedges, keep it as an appendix confound showing why the oracle-support one-layer setting is the clean test.",
            "",
        ]
    )
    (output_root / "findings_summary.md").write_text("\n".join(lines), encoding="utf-8")


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset == "smoke":
        args.relation_types = args.relation_types or [2, 4]
        args.heads_grid = args.heads_grid or [1, 2]
        args.layers_grid = args.layers_grid or [1]
        args.students = args.students or ["graphormer_struct_support", "rel_value_support", "edge_gnn"]
        args.steps = args.steps or 80
        args.eval_graphs = args.eval_graphs or 32
        args.batch_size = args.batch_size or 32
        args.eval_batch_size = args.eval_batch_size or 32
    elif args.preset == "fast":
        args.relation_types = args.relation_types or [2, 4, 8]
        args.heads_grid = args.heads_grid or [1, 2, 4]
        args.layers_grid = args.layers_grid or [1]
        args.students = args.students or list(DEFAULT_FAST_STUDENTS)
        args.steps = args.steps or 350
        args.eval_graphs = args.eval_graphs or 128
        args.batch_size = args.batch_size or 64
        args.eval_batch_size = args.eval_batch_size or 64
    elif args.preset == "full":
        args.relation_types = args.relation_types or [2, 4, 8, 16]
        args.heads_grid = args.heads_grid or [1, 2, 4, 8]
        args.layers_grid = args.layers_grid or [1]
        args.students = args.students or list(DEFAULT_FAST_STUDENTS) + ["vanilla_support", "graphormer_dense"]
        args.steps = args.steps or 800
        args.eval_graphs = args.eval_graphs or 256
        args.batch_size = args.batch_size or 64
        args.eval_batch_size = args.eval_batch_size or 64
    return args


def run_all(args: argparse.Namespace) -> None:
    args = apply_preset(args)
    maybe_mount_drive(args.mount_drive)
    output_root = args.output_root or default_output_root()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[device] {device}", flush=True)
    print(f"[output] {output_root.resolve()}", flush=True)
    print(f"[students] {' '.join(args.students)}", flush=True)

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for layers in args.layers_grid:
            for relation_types in args.relation_types:
                if relation_types >= args.num_nodes:
                    print(f"[skip] R={relation_types} because num_nodes={args.num_nodes}", flush=True)
                    continue
                for heads in args.heads_grid:
                    if args.hidden_dim % heads != 0:
                        print(f"[skip] H={heads} because hidden_dim={args.hidden_dim} is not divisible", flush=True)
                        continue
                    for student in args.students:
                        cfg = RunConfig(
                            seed=int(seed),
                            num_nodes=args.num_nodes,
                            input_dim=args.input_dim,
                            hidden_dim=args.hidden_dim,
                            target_dim=args.target_dim,
                            layers=int(layers),
                            heads=int(heads),
                            relation_types=int(relation_types),
                            batch_size=args.batch_size,
                            eval_batch_size=args.eval_batch_size,
                            steps=args.steps,
                            eval_graphs=args.eval_graphs,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            dropout=args.dropout,
                            attn_dropout=args.attn_dropout,
                            warmup_steps=args.warmup_steps,
                        )
                        rows.append(
                            train_one(
                                cfg,
                                student,
                                output_root,
                                device,
                                force=args.force,
                                save_model=not args.no_save_model,
                            )
                        )
                        write_csv(output_root / "summary_partial.csv", rows)

    write_csv(output_root / "summary.csv", rows)
    write_json(
        output_root / "experiment_notes.json",
        {
            "purpose": "Clean test of Graphormer head-basis approximation for relation-conditioned value transforms.",
            "interpretation": {
                "graphormer_struct_support": "Primary non-confounded test: local support is given, qk is disabled, only scalar head-wise structural routing remains.",
                "graphormer_support": "Checks whether normal content qk changes the head-basis picture.",
                "csa_bias_support": "Routing-only but channel-wise rather than scalar per head.",
                "rel_value_support": "Explicit relation-conditioned value transform upper bound.",
                "edge_gnn": "Local relation-conditioned message passing baseline.",
                "graphormer_dense": "Confounded realism check: model must learn both local support and relation basis.",
            },
            "configs": {
                "seeds": args.seeds,
                "relation_types": args.relation_types,
                "heads_grid": args.heads_grid,
                "layers_grid": args.layers_grid,
                "students": args.students,
            },
        },
    )
    if not args.no_plots:
        plot_summary(rows, output_root)
    write_findings_summary(rows, output_root)
    print(f"[summary] {output_root / 'summary.csv'}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "fast", "full"], default="fast")
    parser.add_argument("--students", nargs="+", choices=STUDENTS, default=None)
    parser.add_argument("--relation-types", nargs="+", type=int, default=None)
    parser.add_argument("--heads-grid", nargs="+", type=int, default=None)
    parser.add_argument("--layers-grid", nargs="+", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--num-nodes", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--target-dim", type=int, default=16)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-graphs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--mount-drive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    run_all(parse_args(argv))


if __name__ == "__main__":
    main()
