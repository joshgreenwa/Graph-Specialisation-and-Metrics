#!/usr/bin/env python3
"""GraphGPS-style baseline for the MarkedTreePath synthetic task.

The task is deliberately small but interaction-heavy. Each graph is a random
tree with two marked endpoints S and T. The node target is the unique S-to-T
path mask:

    y[v] = 1 iff d(S, v) + d(v, T) == d(S, T)

This file trains one small GPS-style model per requested depth, evaluates on
fixed ID and OOD graph-size splits, stops early when validation performance is
perfect, and computes the x-only permutation attention-logit metrics used by
the ZINC analysis script:

    positional_score = invariance under node-feature permutation
    symbolic_score   = equivariance under node-feature permutation

The model is self-contained so it can run without the full GraphGPS/GraphGym
stack. Its block follows the GraphGPS shape: a local GINE-like message-passing
branch, a global multi-head self-attention branch, residual fusion, and an FFN.
Structural inputs are configurable. The default uses shortest-path-distance
attention bias because this task is designed to test symbolic/structural
interaction; use --structural-channel rwse for a closer GPS+RWSE baseline.
"""

import argparse
import csv
import heapq
import json
import math
import random
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Notebook/Colab hook:
# If this whole file is pasted into a notebook cell, kernel argv contains
# IPython flags that would normally break argparse. In a notebook cell we ignore
# those flags and parse CELL_ARGS instead. Edit this value in the pasted cell to
# override defaults, for example:
# CELL_ARGS = "--depths 1 2 3 --structural-channel rwse --max-epochs 100"
CELL_ARGS: list[str] | str | None = None


StructuralChannel = Literal["none", "degree", "rwse", "spd_bias", "rwse_spd_bias"]


@dataclass
class GraphExample:
    x: np.ndarray
    y: np.ndarray
    adj: np.ndarray
    pe: np.ndarray
    spd: np.ndarray
    n: int
    path_len: int


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    mask: torch.Tensor
    adj_norm: torch.Tensor
    pe: torch.Tensor
    spd: torch.Tensor
    path_len: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            x=self.x.to(device),
            y=self.y.to(device),
            mask=self.mask.to(device),
            adj_norm=self.adj_norm.to(device),
            pe=self.pe.to(device),
            spd=self.spd.to(device),
            path_len=self.path_len.to(device),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
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


def prufer_tree_edges(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if n < 2:
        raise ValueError("MarkedTreePath graphs need at least two nodes")
    if n == 2:
        return [(0, 1)]

    prufer = rng.integers(0, n, size=n - 2, dtype=np.int64)
    degree = np.ones(n, dtype=np.int64)
    for node in prufer:
        degree[int(node)] += 1

    leaves = [i for i, deg in enumerate(degree) if deg == 1]
    heapq.heapify(leaves)
    edges: list[tuple[int, int]] = []
    for node in prufer:
        leaf = heapq.heappop(leaves)
        parent = int(node)
        edges.append((leaf, parent))
        degree[leaf] -= 1
        degree[parent] -= 1
        if degree[parent] == 1:
            heapq.heappush(leaves, parent)
    a = heapq.heappop(leaves)
    b = heapq.heappop(leaves)
    edges.append((a, b))
    return edges


def adjacency_from_edges(
    n: int,
    edges: list[tuple[int, int]],
) -> tuple[np.ndarray, list[list[int]]]:
    adj = np.zeros((n, n), dtype=np.float32)
    neighbours = [[] for _ in range(n)]
    for a, b in edges:
        adj[a, b] = 1.0
        adj[b, a] = 1.0
        neighbours[a].append(b)
        neighbours[b].append(a)
    return adj, neighbours


def bfs_path(neighbours: list[list[int]], start: int, end: int) -> list[int]:
    parent = [-1] * len(neighbours)
    parent[start] = start
    queue = [start]
    for node in queue:
        if node == end:
            break
        for nxt in neighbours[node]:
            if parent[nxt] < 0:
                parent[nxt] = node
                queue.append(nxt)

    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def all_pairs_tree_distances(neighbours: list[list[int]]) -> np.ndarray:
    n = len(neighbours)
    out = np.zeros((n, n), dtype=np.int64)
    for source in range(n):
        dist = np.full(n, -1, dtype=np.int64)
        dist[source] = 0
        queue = [source]
        for node in queue:
            for nxt in neighbours[node]:
                if dist[nxt] < 0:
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
        out[source] = dist
    return out


def sample_marked_endpoint_path(
    neighbours: list[list[int]],
    rng: np.random.Generator,
    min_path_frac: float,
    num_candidates: int,
) -> list[int]:
    n = len(neighbours)
    min_dist = max(2, int(round(min_path_frac * math.sqrt(n))))
    candidates: list[list[int]] = []
    best: list[int] | None = None
    for _ in range(num_candidates):
        s, t = rng.choice(n, size=2, replace=False)
        path = bfs_path(neighbours, int(s), int(t))
        if best is None or len(path) > len(best):
            best = path
        if len(path) - 1 >= min_dist:
            candidates.append(path)
    if candidates:
        return candidates[int(rng.integers(0, len(candidates)))]
    assert best is not None
    return best


def compute_rwse(adj: np.ndarray, steps: int) -> np.ndarray:
    n = int(adj.shape[0])
    if steps <= 0:
        return np.zeros((n, 0), dtype=np.float32)
    deg = np.maximum(adj.sum(axis=1), 1.0).astype(np.float32)
    row_scaled = adj / deg[:, None]

    try:
        import scipy.sparse as sp

        p_mat = sp.csr_matrix(row_scaled)
        cur = sp.identity(n, dtype=np.float32, format="csr")
        feats = []
        for _ in range(steps):
            cur = cur @ p_mat
            feats.append(cur.diagonal().astype(np.float32))
        return np.stack(feats, axis=1)
    except Exception:
        cur_dense = np.eye(n, dtype=np.float32)
        feats = []
        for _ in range(steps):
            cur_dense = cur_dense @ row_scaled
            feats.append(np.diag(cur_dense).astype(np.float32))
        return np.stack(feats, axis=1)


def pe_dim_for_channel(channel: StructuralChannel, rwse_steps: int) -> int:
    if channel == "degree":
        return 1
    if channel in {"rwse", "rwse_spd_bias"}:
        return rwse_steps
    return 0


def uses_spd_bias(channel: StructuralChannel) -> bool:
    return channel in {"spd_bias", "rwse_spd_bias"}


def make_example(
    n: int,
    rng: np.random.Generator,
    structural_channel: StructuralChannel,
    rwse_steps: int,
    min_path_frac: float,
    endpoint_candidates: int,
) -> GraphExample:
    edges = prufer_tree_edges(n, rng)
    adj, neighbours = adjacency_from_edges(n, edges)
    path = sample_marked_endpoint_path(
        neighbours, rng, min_path_frac=min_path_frac, num_candidates=endpoint_candidates
    )
    x = np.zeros(n, dtype=np.int64)
    x[path[0]] = 1
    x[path[-1]] = 2

    y = np.zeros(n, dtype=np.float32)
    y[np.asarray(path, dtype=np.int64)] = 1.0

    pe_dim = pe_dim_for_channel(structural_channel, rwse_steps)
    if structural_channel == "degree":
        degree = adj.sum(axis=1, keepdims=True)
        pe = (degree / max(1.0, float(n - 1))).astype(np.float32)
    elif structural_channel in {"rwse", "rwse_spd_bias"}:
        pe = compute_rwse(adj, steps=rwse_steps).astype(np.float32)
    else:
        pe = np.zeros((n, pe_dim), dtype=np.float32)

    if uses_spd_bias(structural_channel):
        spd = all_pairs_tree_distances(neighbours)
    else:
        spd = np.zeros((n, n), dtype=np.int64)

    return GraphExample(x=x, y=y, adj=adj, pe=pe, spd=spd, n=n, path_len=len(path))


def generate_examples(
    count: int,
    min_n: int,
    max_n: int,
    seed: int,
    structural_channel: StructuralChannel,
    rwse_steps: int,
    min_path_frac: float,
    endpoint_candidates: int,
) -> list[GraphExample]:
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(count):
        n = int(rng.integers(min_n, max_n + 1))
        graphs.append(
            make_example(
                n=n,
                rng=rng,
                structural_channel=structural_channel,
                rwse_steps=rwse_steps,
                min_path_frac=min_path_frac,
                endpoint_candidates=endpoint_candidates,
            )
        )
    return graphs


def iter_generated_batches(
    count: int,
    batch_size: int,
    min_n: int,
    max_n: int,
    seed: int,
    structural_channel: StructuralChannel,
    rwse_steps: int,
    min_path_frac: float,
    endpoint_candidates: int,
    spd_cap: int,
    device: torch.device,
) -> Iterable[Batch]:
    rng = np.random.default_rng(seed)
    remaining = count
    while remaining > 0:
        this_batch = min(batch_size, remaining)
        examples = []
        for _ in range(this_batch):
            n = int(rng.integers(min_n, max_n + 1))
            examples.append(
                make_example(
                    n=n,
                    rng=rng,
                    structural_channel=structural_channel,
                    rwse_steps=rwse_steps,
                    min_path_frac=min_path_frac,
                    endpoint_candidates=endpoint_candidates,
                )
            )
        remaining -= this_batch
        yield collate_examples(examples, spd_cap=spd_cap).to(device)


def iter_static_batches(
    examples: list[GraphExample],
    batch_size: int,
    spd_cap: int,
    device: torch.device,
) -> Iterable[Batch]:
    for start in range(0, len(examples), batch_size):
        yield collate_examples(examples[start : start + batch_size], spd_cap=spd_cap).to(device)


def collate_examples(examples: list[GraphExample], spd_cap: int) -> Batch:
    batch_size = len(examples)
    max_n = max(ex.n for ex in examples)
    pe_dim = examples[0].pe.shape[1]

    x = np.zeros((batch_size, max_n), dtype=np.int64)
    y = np.zeros((batch_size, max_n), dtype=np.float32)
    mask = np.zeros((batch_size, max_n), dtype=np.bool_)
    adj_norm = np.zeros((batch_size, max_n, max_n), dtype=np.float32)
    pe = np.zeros((batch_size, max_n, pe_dim), dtype=np.float32)
    spd = np.full((batch_size, max_n, max_n), fill_value=spd_cap + 1, dtype=np.int64)
    path_len = np.zeros(batch_size, dtype=np.int64)

    for idx, ex in enumerate(examples):
        n = ex.n
        degree = np.maximum(ex.adj.sum(axis=1, keepdims=True), 1.0)
        x[idx, :n] = ex.x
        y[idx, :n] = ex.y
        mask[idx, :n] = True
        adj_norm[idx, :n, :n] = ex.adj / degree
        pe[idx, :n, :] = ex.pe
        spd[idx, :n, :n] = np.clip(ex.spd, 0, spd_cap)
        path_len[idx] = ex.path_len

    return Batch(
        x=torch.from_numpy(x),
        y=torch.from_numpy(y),
        mask=torch.from_numpy(mask),
        adj_norm=torch.from_numpy(adj_norm),
        pe=torch.from_numpy(pe),
        spd=torch.from_numpy(spd),
        path_len=torch.from_numpy(path_len),
    )


class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenseGINEBranch(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_emb = nn.Parameter(torch.zeros(dim))
        self.eps = nn.Parameter(torch.zeros(()))
        self.mlp = MLP(dim, dim * 2, dim, dropout)

    def forward(self, h: torch.Tensor, adj_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        msg = F.relu(h + self.edge_emb)
        agg = torch.bmm(adj_norm, msg)
        out = self.mlp((1.0 + self.eps) * h + agg)
        return out * mask.unsqueeze(-1).to(out.dtype)


class BiasedMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        use_spd_bias: bool,
        spd_cap: int,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_spd_bias = use_spd_bias
        self.spd_cap = spd_cap
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        if use_spd_bias:
            self.spd_bias = nn.Embedding(spd_cap + 2, num_heads)
        else:
            self.spd_bias = None

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        spd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_n, _ = h.shape
        qkv = self.qkv(h).view(batch_size, max_n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.spd_bias is not None:
            bias = self.spd_bias(spd.clamp(0, self.spd_cap + 1)).permute(0, 3, 1, 2)
            logits = logits + bias

        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        key_mask = mask[:, None, None, :]
        masked_logits = logits.masked_fill(~key_mask, -1.0e9)
        attn = torch.softmax(masked_logits, dim=-1)
        attn = attn.masked_fill(~pair_mask, 0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, max_n, self.dim)
        out = self.out(out)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        metric_logits = logits.masked_fill(~pair_mask, 0.0)
        return out, metric_logits


class GraphGPSLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        attn_dropout: float,
        use_spd_bias: bool,
        spd_cap: int,
    ) -> None:
        super().__init__()
        self.local = DenseGINEBranch(dim, dropout=dropout)
        self.attn = BiasedMultiheadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            use_spd_bias=use_spd_bias,
            spd_cap=spd_cap,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, h: torch.Tensor, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        local = self.local(h, batch.adj_norm, batch.mask)
        attn, logits = self.attn(h, batch.mask, batch.spd)
        h = self.norm1(h + self.dropout(local + attn))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        h = h * batch.mask.unsqueeze(-1).to(h.dtype)
        return h, logits


class GraphGPSPathModel(nn.Module):
    def __init__(
        self,
        depth: int,
        hidden_dim: int,
        num_heads: int,
        pe_dim: int,
        dropout: float,
        attn_dropout: float,
        use_spd_bias: bool,
        spd_cap: int,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(3, hidden_dim)
        self.pe_proj = nn.Linear(pe_dim, hidden_dim, bias=False) if pe_dim > 0 else None
        self.layers = nn.ModuleList(
            [
                GraphGPSLayer(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    use_spd_bias=use_spd_bias,
                    spd_cap=spd_cap,
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        batch: Batch,
        collect_attention: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        h = self.token_emb(batch.x)
        if self.pe_proj is not None:
            h = h + self.pe_proj(batch.pe)
        h = h * batch.mask.unsqueeze(-1).to(h.dtype)

        layers = []
        for layer in self.layers:
            h, logits = layer(h, batch)
            if collect_attention:
                layers.append(
                    {
                        "logits": logits.detach(),
                        "node_mask": batch.mask.detach(),
                    }
                )
        node_logits = self.head(h).squeeze(-1)
        node_logits = node_logits.masked_fill(~batch.mask, 0.0)
        return node_logits, layers


def batch_loss(logits: torch.Tensor, batch: Batch, max_pos_weight: float) -> torch.Tensor:
    valid_logits = logits[batch.mask]
    valid_y = batch.y[batch.mask]
    pos = valid_y.sum()
    neg = valid_y.numel() - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(max=max_pos_weight)
    return F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_y,
        pos_weight=pos_weight.detach(),
    )


@dataclass
class EvalStats:
    loss: float
    node_acc: float
    graph_exact: float
    precision: float
    recall: float
    f1: float
    pos_rate: float
    pred_pos_rate: float
    avg_path_len: float


@torch.no_grad()
def evaluate(
    model: GraphGPSPathModel,
    examples: list[GraphExample],
    batch_size: int,
    spd_cap: int,
    device: torch.device,
) -> EvalStats:
    model.eval()
    total_loss = 0.0
    total_nodes = 0
    total_graphs = 0
    exact_graphs = 0
    correct = 0
    tp = fp = fn = 0
    pos_total = 0
    pred_pos_total = 0
    path_len_total = 0

    for batch in iter_static_batches(
        examples,
        batch_size=batch_size,
        spd_cap=spd_cap,
        device=device,
    ):
        logits, _ = model(batch, collect_attention=False)
        loss = batch_loss(logits, batch, max_pos_weight=1.0)
        pred = torch.sigmoid(logits) >= 0.5
        truth = batch.y.bool()
        valid = batch.mask
        correct_mask = (pred == truth) & valid

        nodes = int(valid.sum().item())
        total_nodes += nodes
        correct += int(correct_mask.sum().item())
        total_loss += float(loss.item()) * nodes

        graph_correct = ((pred == truth) | ~valid).all(dim=1)
        exact_graphs += int(graph_correct.sum().item())
        total_graphs += int(batch.x.size(0))

        tp += int((pred & truth & valid).sum().item())
        fp += int((pred & ~truth & valid).sum().item())
        fn += int((~pred & truth & valid).sum().item())
        pos_total += int((truth & valid).sum().item())
        pred_pos_total += int((pred & valid).sum().item())
        path_len_total += int(batch.path_len.sum().item())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
    return EvalStats(
        loss=total_loss / max(1, total_nodes),
        node_acc=correct / max(1, total_nodes),
        graph_exact=exact_graphs / max(1, total_graphs),
        precision=precision,
        recall=recall,
        f1=f1,
        pos_rate=pos_total / max(1, total_nodes),
        pred_pos_rate=pred_pos_total / max(1, total_nodes),
        avg_path_len=path_len_total / max(1, total_graphs),
    )


def pair_mask_from_layer(layer: dict[str, torch.Tensor]) -> torch.Tensor:
    node_mask = layer["node_mask"]
    return node_mask[:, None, :, None] & node_mask[:, None, None, :]


def transform_pair_reference(z: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    batch_size, heads, max_n, _ = z.shape
    idx = perm_pos.to(z.device)
    row_idx = idx[:, None, :, None].expand(batch_size, heads, max_n, max_n)
    z_rows = torch.gather(z, dim=2, index=row_idx)
    col_idx = idx[:, None, None, :].expand(batch_size, heads, max_n, max_n)
    return torch.gather(z_rows, dim=3, index=col_idx)


def row_center_logits(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.size(1) == 1 and z.size(1) != 1:
        mask = mask.expand(-1, z.size(1), -1, -1)
    z0 = torch.where(mask, torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(z.dtype)
    mean = z0.sum(dim=-1, keepdim=True) / denom
    return torch.where(mask, z0 - mean, torch.zeros_like(z0))


def cosine_by_head_logits(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.size(1) == 1 and u.size(1) != 1:
        mask = mask.expand(-1, u.size(1), -1, -1)
    u0 = row_center_logits(u, mask)
    v0 = row_center_logits(v, mask)
    dims = (0, 2, 3)
    num = (u0 * v0).sum(dim=dims)
    den = torch.sqrt((u0 * u0).sum(dim=dims).clamp_min(1.0e-12)) * torch.sqrt(
        (v0 * v0).sum(dim=dims).clamp_min(1.0e-12)
    )
    return (num / den.clamp_min(1.0e-12)).detach().cpu()


def cosine_to_score01(cos: torch.Tensor) -> torch.Tensor:
    return torch.clamp(0.5 * (cos + 1.0), 0.0, 1.0)


def make_x_permutation(batch: Batch, generator: torch.Generator) -> tuple[Batch, torch.Tensor]:
    batch_size, max_n = batch.x.shape
    perm_pos = torch.zeros((batch_size, max_n), dtype=torch.long, device=batch.x.device)
    x_perm = batch.x.clone()
    for graph_idx in range(batch_size):
        n = int(batch.mask[graph_idx].sum().item())
        perm = torch.randperm(n, generator=generator).to(batch.x.device)
        perm_pos[graph_idx, :n] = perm
        if n < max_n:
            perm_pos[graph_idx, n:] = torch.arange(n, max_n, device=batch.x.device)
        x_perm[graph_idx, :n] = batch.x[graph_idx, perm]

    variant = Batch(
        x=x_perm,
        y=batch.y,
        mask=batch.mask,
        adj_norm=batch.adj_norm,
        pe=batch.pe,
        spd=batch.spd,
        path_len=batch.path_len,
    )
    return variant, perm_pos


@torch.no_grad()
def compute_xperm_metrics(
    model: GraphGPSPathModel,
    examples: list[GraphExample],
    split: str,
    depth: int,
    batch_size: int,
    spd_cap: int,
    num_perms: int,
    metric_graphs: int,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    model.eval()
    selected = examples[: min(metric_graphs, len(examples))]
    rows: list[dict[str, float | int | str]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for batch_idx, batch in enumerate(
        iter_static_batches(selected, batch_size=batch_size, spd_cap=spd_cap, device=device)
    ):
        _, clean_layers = model(batch, collect_attention=True)
        for perm_idx in range(num_perms):
            variant_batch, perm_pos = make_x_permutation(batch, generator)
            _, variant_layers = model(variant_batch, collect_attention=True)
            for layer_idx, (clean, variant) in enumerate(zip(clean_layers, variant_layers)):
                z_clean = clean["logits"]
                z_var = variant["logits"]
                mask = pair_mask_from_layer(clean).to(device=z_clean.device)
                z_ref = transform_pair_reference(z_clean, perm_pos)
                inv_cos = cosine_by_head_logits(z_var, z_clean, mask)
                equi_cos = cosine_by_head_logits(z_var, z_ref, mask)
                inv = cosine_to_score01(inv_cos)
                equi = cosine_to_score01(equi_cos)
                for head in range(int(inv.numel())):
                    common = {
                        "depth": depth,
                        "split": split,
                        "batch": batch_idx,
                        "perm": perm_idx,
                        "layer": layer_idx,
                        "head": head,
                    }
                    rows.append(
                        {
                            **common,
                            "metric": "positional_score",
                            "score": float(inv[head]),
                            "raw_cos": float(inv_cos[head]),
                        }
                    )
                    rows.append(
                        {
                            **common,
                            "metric": "symbolic_score",
                            "score": float(equi[head]),
                            "raw_cos": float(equi_cos[head]),
                        }
                    )
    return rows


def summarize_xperm_rows(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    buckets: dict[tuple[int, str, int, int, str], list[float]] = {}
    for row in rows:
        key = (
            int(row["depth"]),
            str(row["split"]),
            int(row["layer"]),
            int(row["head"]),
            str(row["metric"]),
        )
        buckets.setdefault(key, []).append(float(row["score"]))

    out = []
    for (depth, split, layer, head, metric), values in sorted(buckets.items()):
        arr = np.asarray(values, dtype=np.float64)
        out.append(
            {
                "depth": depth,
                "split": split,
                "layer": layer,
                "head": head,
                "metric": metric,
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
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_stats(prefix: str, stats: EvalStats) -> str:
    return (
        f"{prefix}: loss={stats.loss:.4f} node_acc={stats.node_acc:.4f} "
        f"exact={stats.graph_exact:.4f} f1={stats.f1:.4f} "
        f"pos={stats.pos_rate:.3f} pred_pos={stats.pred_pos_rate:.3f}"
    )


def is_perfect(stats: EvalStats) -> bool:
    return stats.node_acc >= 1.0 and stats.graph_exact >= 1.0 and stats.f1 >= 1.0


def train_depth(
    depth: int,
    args: argparse.Namespace,
    eval_sets: dict[str, list[GraphExample]],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict, list[dict], GraphGPSPathModel]:
    set_seed(args.seed + depth * 997)
    pe_dim = pe_dim_for_channel(args.structural_channel, args.rwse_steps)
    model = GraphGPSPathModel(
        depth=depth,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        pe_dim=pe_dim,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        use_spd_bias=uses_spd_bias(args.structural_channel),
        spd_cap=args.spd_cap,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    run_dir = output_dir / f"depth_{depth}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_rows: list[dict] = []
    best_state = None
    best_val_exact = -1.0
    best_val_node_acc = -1.0
    best_epoch = -1
    best_stats: dict[str, EvalStats] | None = None
    perfect_hits = 0
    stale_evals = 0
    start_time = time.time()

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_nodes = 0
        train_seed = args.seed + epoch * 1009
        for batch in iter_generated_batches(
            count=args.train_graphs_per_epoch,
            batch_size=args.batch_size,
            min_n=args.train_min_n,
            max_n=args.train_max_n,
            seed=train_seed,
            structural_channel=args.structural_channel,
            rwse_steps=args.rwse_steps,
            min_path_frac=args.min_path_frac,
            endpoint_candidates=args.endpoint_candidates,
            spd_cap=args.spd_cap,
            device=device,
        ):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch, collect_attention=False)
            loss = batch_loss(logits, batch, max_pos_weight=args.max_pos_weight)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            nodes = int(batch.mask.sum().item())
            epoch_loss += float(loss.item()) * nodes
            epoch_nodes += nodes

        if epoch % args.eval_every != 0 and epoch != args.max_epochs:
            continue

        val = evaluate(model, eval_sets["val"], args.eval_batch_size, args.spd_cap, device)
        id_test = evaluate(model, eval_sets["id_test"], args.eval_batch_size, args.spd_cap, device)
        ood_test = evaluate(
            model,
            eval_sets["ood_test"],
            args.eval_batch_size,
            args.spd_cap,
            device,
        )
        train_loss = epoch_loss / max(1, epoch_nodes)

        row = {
            "depth": depth,
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in asdict(val).items()},
            **{f"id_test_{k}": v for k, v in asdict(id_test).items()},
            **{f"ood_test_{k}": v for k, v in asdict(ood_test).items()},
            "elapsed_s": time.time() - start_time,
        }
        log_rows.append(row)
        write_csv(run_dir / "train_log.csv", log_rows)

        improved = (val.graph_exact, val.node_acc) > (best_val_exact, best_val_node_acc)
        if improved:
            best_val_exact = val.graph_exact
            best_val_node_acc = val.node_acc
            best_epoch = epoch
            best_stats = {"val": val, "id_test": id_test, "ood_test": ood_test}
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            stale_evals = 0
            if not args.no_save_checkpoints:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "depth": depth,
                        "args": vars(args),
                        "best_epoch": best_epoch,
                        "best_stats": {k: asdict(v) for k, v in best_stats.items()},
                    },
                    run_dir / "best.pt",
                )
        else:
            stale_evals += 1

        if is_perfect(val):
            perfect_hits += 1
        else:
            perfect_hits = 0

        print(
            f"[depth {depth} | epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"{format_stats('val', val)} | {format_stats('ID', id_test)} | "
            f"{format_stats('OOD', ood_test)}",
            flush=True,
        )

        if args.stop_on_perfect and perfect_hits >= args.perfect_patience:
            print(
                f"[depth {depth}] early stop: validation perfect for "
                f"{perfect_hits} eval(s)",
                flush=True,
            )
            break
        if args.patience > 0 and stale_evals >= args.patience:
            print(f"[depth {depth}] early stop: no validation improvement", flush=True)
            break

    if best_state is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    assert best_stats is not None
    summary = {
        "depth": depth,
        "best_epoch": best_epoch,
        "parameters": sum(p.numel() for p in model.parameters()),
        **{f"best_val_{k}": v for k, v in asdict(best_stats["val"]).items()},
        **{f"best_id_test_{k}": v for k, v in asdict(best_stats["id_test"]).items()},
        **{f"best_ood_test_{k}": v for k, v in asdict(best_stats["ood_test"]).items()},
    }
    return summary, log_rows, model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train small GraphGPS-style baselines on MarkedTreePath."
    )
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--structural-channel",
        choices=["none", "degree", "rwse", "spd_bias", "rwse_spd_bias"],
        default="spd_bias",
        help="Structural input to use. spd_bias is the strongest/default path-task baseline.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--rwse-steps", type=int, default=16)
    parser.add_argument("--spd-cap", type=int, default=32)

    parser.add_argument("--train-min-n", type=int, default=12)
    parser.add_argument("--train-max-n", type=int, default=32)
    parser.add_argument("--ood-min-n", type=int, default=64)
    parser.add_argument("--ood-max-n", type=int, default=128)
    parser.add_argument("--min-path-frac", type=float, default=0.75)
    parser.add_argument("--endpoint-candidates", type=int, default=32)

    parser.add_argument("--train-graphs-per-epoch", type=int, default=4096)
    parser.add_argument("--val-graphs", type=int, default=512)
    parser.add_argument("--id-test-graphs", type=int, default=1024)
    parser.add_argument("--ood-test-graphs", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--perfect-patience", type=int, default=2)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--stop-on-perfect", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)

    parser.add_argument("--metric-perms", type=int, default=3)
    parser.add_argument("--metric-graphs", type=int, default=256)
    parser.add_argument("--metric-batch-size", type=int, default=64)
    parser.add_argument("--skip-xperm-metrics", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/synthetic/results/marked_tree_path_graphgps"),
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Tiny run for syntax/runtime checks.",
    )
    args = parser.parse_args(notebook_safe_argv(argv))

    if args.fast_dev_run:
        args.depths = args.depths[:1]
        args.train_graphs_per_epoch = 128
        args.val_graphs = 32
        args.id_test_graphs = 32
        args.ood_test_graphs = 32
        args.batch_size = 16
        args.eval_batch_size = 32
        args.metric_graphs = 16
        args.metric_perms = 1
        args.max_epochs = min(args.max_epochs, 2)
        args.patience = 2

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = choose_device(args.device)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"device_resolved": str(device)}, f, indent=2, default=str)

    print(f"[setup] output_dir={output_dir}", flush=True)
    print(f"[setup] device={device}", flush=True)
    print(f"[setup] structural_channel={args.structural_channel}", flush=True)

    print("[data] generating fixed validation, ID-test, and OOD-test sets", flush=True)
    eval_sets = {
        "val": generate_examples(
            args.val_graphs,
            args.train_min_n,
            args.train_max_n,
            args.seed + 11,
            args.structural_channel,
            args.rwse_steps,
            args.min_path_frac,
            args.endpoint_candidates,
        ),
        "id_test": generate_examples(
            args.id_test_graphs,
            args.train_min_n,
            args.train_max_n,
            args.seed + 23,
            args.structural_channel,
            args.rwse_steps,
            args.min_path_frac,
            args.endpoint_candidates,
        ),
        "ood_test": generate_examples(
            args.ood_test_graphs,
            args.ood_min_n,
            args.ood_max_n,
            args.seed + 37,
            args.structural_channel,
            args.rwse_steps,
            args.min_path_frac,
            args.endpoint_candidates,
        ),
    }

    summaries = []
    all_xperm_rows: list[dict[str, float | int | str]] = []
    for depth in args.depths:
        summary, _, model = train_depth(depth, args, eval_sets, output_dir, device)
        summaries.append(summary)
        write_csv(output_dir / "summary.csv", summaries)

        if not args.skip_xperm_metrics:
            print(f"[xperm] depth={depth} computing ID metrics", flush=True)
            all_xperm_rows.extend(
                compute_xperm_metrics(
                    model,
                    eval_sets["id_test"],
                    split="id_test",
                    depth=depth,
                    batch_size=args.metric_batch_size,
                    spd_cap=args.spd_cap,
                    num_perms=args.metric_perms,
                    metric_graphs=args.metric_graphs,
                    seed=args.seed + 101 * depth,
                    device=device,
                )
            )
            print(f"[xperm] depth={depth} computing OOD metrics", flush=True)
            all_xperm_rows.extend(
                compute_xperm_metrics(
                    model,
                    eval_sets["ood_test"],
                    split="ood_test",
                    depth=depth,
                    batch_size=args.metric_batch_size,
                    spd_cap=args.spd_cap,
                    num_perms=args.metric_perms,
                    metric_graphs=args.metric_graphs,
                    seed=args.seed + 202 * depth,
                    device=device,
                )
            )
            write_csv(output_dir / "xperm_metrics.csv", all_xperm_rows)
            write_csv(output_dir / "xperm_summary.csv", summarize_xperm_rows(all_xperm_rows))

    print("[done] best summary", flush=True)
    for row in summaries:
        print(
            f"depth={row['depth']} params={row['parameters']} best_epoch={row['best_epoch']} "
            f"val_exact={row['best_val_graph_exact']:.4f} "
            f"ID_exact={row['best_id_test_graph_exact']:.4f} "
            f"OOD_exact={row['best_ood_test_graph_exact']:.4f}",
            flush=True,
        )
    print(f"[done] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
