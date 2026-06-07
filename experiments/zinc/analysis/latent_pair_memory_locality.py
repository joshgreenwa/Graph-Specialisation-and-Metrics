#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone locality analysis for the trained LatentPairMemory.ValueActive ZINC
checkpoint.

This file intentionally imports no project-local training module. It contains the
model, cache loading, batching, and optional ZINC preprocessing code needed to
load a checkpoint saved by the training script and inspect the latent pair memory
z_ij after each layer.

Reports:
  1. RMS of z_ij binned by shortest-path distance.
  2. Locality ratios per layer.
  3. Effective dynamic-bias logit contribution binned by shortest-path distance.
  4. Channel utilisation of z per layer.
  5. Optional test-set MAE ablations that zero the dynamic latent pair memory
     z_ij by SPD/non-edge category while leaving the fixed structural xi^0
     features intact.

Typical Colab usage:

    !python latent_pair_memory_locality.py

or:

    !python latent_pair_memory_locality.py --num-batches 32

Run compact test-set ablations:

    !python latent_pair_memory_locality.py --run-ablations
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# =============================================================================
# Constants matching the trained variant.
# =============================================================================

NUM_ATOM_TYPES = 28
NUM_BOND_TYPES = 3

D_MODEL = 64
N_HEADS = 8
N_LAYERS = 10
D_HEAD = D_MODEL // N_HEADS

K_WALK = 12
D_MAX_SPD = 5

RING_MIN_SIZE = 3
RING_MAX_SIZE = 8
P_RING_SIZES = RING_MAX_SIZE - RING_MIN_SIZE + 1
P_RING_PAIR = 2 * P_RING_SIZES
P_SYM = (D_MAX_SPD + 1) + K_WALK + NUM_BOND_TYPES
P_PAIR = P_RING_PAIR + P_SYM
P_LATENT_PAIR = 32

BIAS_INPUT_DIM = P_PAIR + P_LATENT_PAIR
VALUE_INPUT_DIM = P_PAIR + P_LATENT_PAIR
DYNAMIC_GATE_INIT = 5e-2

PAIR_UPDATE_RANK = 16
PAIR_UPDATE_SCALE_INIT = 5e-2
PAIR_OUT_INIT_STD = 1e-3
PAIR_WRITE_GATE_INIT = 0.7
RMS_NORM_EPS = 1e-6

FFN_MULT = 2
ATTN_DROPOUT = 0.2
RESID_DROPOUT = 0.0


# =============================================================================
# Utility.
# =============================================================================

def log(msg: str = "") -> None:
    print(msg, flush=True)


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def mount_drive_if_available(mount_point: str, no_drive: bool) -> None:
    if no_drive:
        log("[drive] --no-drive set; not mounting Google Drive.")
        return
    if not in_colab():
        log("[drive] google.colab unavailable; skipping Drive mount.")
        return
    from google.colab import drive  # type: ignore

    log(f"[drive] Mounting Google Drive at {mount_point} ...")
    drive.mount(mount_point, force_remount=False)


def run_cmd(cmd: Sequence[str]) -> None:
    printable = " ".join(map(str, cmd))
    log(f"\n[cmd] {printable}")
    proc = subprocess.run(list(map(str, cmd)), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {printable}")


def install_deps_if_requested(skip_install: bool, install_deps: bool) -> None:
    if skip_install or not install_deps:
        return

    log("[deps] Installing torch_geometric for cache recomputation / first-time data load.")
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    torch_version = torch.__version__.split("+")[0]
    major_minor = ".".join(torch_version.split(".")[:2])
    cuda_version = torch.version.cuda
    cuda_tag = "cu" + cuda_version.replace(".", "") if cuda_version else "cpu"
    wheel_url = f"https://data.pyg.org/whl/torch-{major_minor}.0+{cuda_tag}.html"
    log(f"[deps] Python: {platform.python_version()} | torch: {torch.__version__} | CUDA: {cuda_version}")
    log(f"[deps] PyG wheel index: {wheel_url}")

    run_cmd([sys.executable, "-m", "pip", "install", "torch_geometric"])
    optional = ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]
    try:
        run_cmd([sys.executable, "-m", "pip", "install", *optional, "-f", wheel_url])
    except Exception as exc:
        log(f"[deps-warning] Optional PyG acceleration wheels unavailable; continuing. Reason: {exc}")


def safe_torch_load(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class TorchLoadCompat:
    """Temporarily force torch.load(..., weights_only=False) for PyG caches."""

    def __enter__(self):
        self.orig_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return self.orig_load(*args, **kwargs)

        torch.load = patched_load
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.load = self.orig_load
        return False


def logit_from_prob(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def softplus_inverse(x: float) -> float:
    return math.log(math.expm1(float(x)))


# =============================================================================
# Dataset preprocessing and collation.
# =============================================================================

def normalize_edge_attr(edge_attr):
    ea = edge_attr.view(-1).long()
    if ea.numel() == 0:
        return ea
    mn, mx = int(ea.min().item()), int(ea.max().item())
    if mn >= 1 and mx <= NUM_BOND_TYPES:
        ea = ea - 1
    if int(ea.min().item()) < 0 or int(ea.max().item()) >= NUM_BOND_TYPES:
        raise ValueError(
            f"Expected {NUM_BOND_TYPES} bond types encoded as 0..{NUM_BOND_TYPES - 1} "
            f"or 1..{NUM_BOND_TYPES}; got min={int(ea.min())}, max={int(ea.max())}."
        )
    return ea


def bfs_spd_classes(adj_bool, dmax: int):
    n = int(adj_bool.size(0))
    dist = torch.full((n, n), dmax, dtype=torch.long)
    neighbors = [torch.nonzero(adj_bool[i], as_tuple=False).view(-1).tolist() for i in range(n)]
    for s in range(n):
        dist[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = int(dist[s, u].item())
            if du >= dmax:
                continue
            for v in neighbors[u]:
                if int(dist[s, v].item()) == dmax and v != s:
                    dist[s, v] = min(du + 1, dmax)
                    q.append(v)
    return dist


def compute_ring_pair_features(adj_bool, min_size: int = RING_MIN_SIZE, max_size: int = RING_MAX_SIZE):
    n = int(adj_bool.size(0))
    num_sizes = max_size - min_size + 1
    same_ring = torch.zeros((n, n, num_sizes), dtype=torch.float32)
    edge_in_ring = torch.zeros((n, n, num_sizes), dtype=torch.float32)
    neighbors = [torch.nonzero(adj_bool[i], as_tuple=False).view(-1).tolist() for i in range(n)]
    seen_cycles = set()

    def is_chordless(path):
        m = len(path)
        path_set = set(path)
        if len(path_set) != m:
            return False
        for a in range(m):
            u = path[a]
            for b in range(a + 1, m):
                if b == a + 1 or (a == 0 and b == m - 1):
                    continue
                if bool(adj_bool[u, path[b]]):
                    return False
        return True

    def mark_cycle(path):
        m = len(path)
        if m < min_size or m > max_size or not is_chordless(path):
            return
        edge_key = []
        for i in range(m):
            u, v = path[i], path[(i + 1) % m]
            edge_key.append((u, v) if u < v else (v, u))
        key = tuple(sorted(edge_key))
        if key in seen_cycles:
            return
        seen_cycles.add(key)

        c = m - min_size
        for u in path:
            for v in path:
                same_ring[u, v, c] = 1.0
        for i in range(m):
            u, v = path[i], path[(i + 1) % m]
            edge_in_ring[u, v, c] = 1.0
            edge_in_ring[v, u, c] = 1.0

    def dfs(start, current, path, visited):
        if len(path) > max_size:
            return
        for nxt in neighbors[current]:
            if nxt == start:
                mark_cycle(path)
            elif nxt > start and nxt not in visited and len(path) < max_size:
                visited.add(nxt)
                path.append(nxt)
                dfs(start, nxt, path, visited)
                path.pop()
                visited.remove(nxt)

    for start in range(n):
        dfs(start, start, [start], {start})

    return torch.cat([same_ring, edge_in_ring], dim=-1)


def compute_variant_a_features(data) -> Dict[str, torch.Tensor]:
    import torch.nn.functional as F

    x = data.x.view(-1).long()
    if int(x.min()) < 0 or int(x.max()) >= NUM_ATOM_TYPES:
        raise ValueError(f"Atom type out of expected 0..{NUM_ATOM_TYPES - 1}: max={int(x.max())}")

    edge_index = data.edge_index.long()
    edge_attr = normalize_edge_attr(data.edge_attr)

    n = int(data.num_nodes)
    A = torch.zeros((n, n), dtype=torch.float32)
    if edge_index.numel() > 0:
        src, dst = edge_index
        A[src, dst] = 1.0
    A = torch.maximum(A, A.t())
    A.fill_diagonal_(0.0)

    degree = A.sum(dim=1)
    W = torch.zeros_like(A)
    nz = degree > 0
    W[nz] = A[nz] / degree[nz].unsqueeze(1)

    powers = []
    P = W.clone()
    for _ in range(K_WALK):
        powers.append(P.clone())
        P = P @ W

    rwse = torch.stack([Pk.diagonal() for Pk in powers], dim=-1).float()
    walk_sum = torch.stack([Pk + Pk.t() for Pk in powers], dim=-1).float()
    ring_pair = compute_ring_pair_features(A > 0)

    spd_cls = bfs_spd_classes(A > 0, D_MAX_SPD)
    spd_oh = F.one_hot(spd_cls.clamp(max=D_MAX_SPD), num_classes=D_MAX_SPD + 1).float()

    bond_oh = torch.zeros((n, n, NUM_BOND_TYPES), dtype=torch.float32)
    if edge_index.numel() > 0:
        src, dst = edge_index
        bond_oh[src, dst, edge_attr] = 1.0
        bond_oh[dst, src, edge_attr] = 1.0

    sym = torch.cat([spd_oh, walk_sum, bond_oh], dim=-1)
    pair_xi = torch.cat([ring_pair, sym], dim=-1).float()
    assert pair_xi.shape == (n, n, P_PAIR)

    return {
        "x": x.cpu(),
        "edge_index": edge_index.cpu(),
        "edge_attr": edge_attr.cpu(),
        "y": data.y.view(1).float().cpu(),
        "rwse": rwse.cpu(),
        "pair_xi": pair_xi.cpu(),
        "degree": degree.float().cpu(),
    }


def cache_path(args: argparse.Namespace) -> str:
    data_root = args.data_dir or os.path.join(args.drive_dir if not args.no_drive else "/content", "datasets")
    return os.path.join(
        data_root,
        "variant_a_cache",
        f"zinc_subset_ringpair_K{K_WALK}_D{D_MAX_SPD}_R{RING_MIN_SIZE}-{RING_MAX_SIZE}_v1.pt",
    )


def load_or_precompute_dataset(args: argparse.Namespace):
    path = cache_path(args)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.isfile(path) and not args.force_recompute_cache:
        log(f"[data] Loading cached ZINC pair features: {path}")
        bundle = safe_torch_load(path, map_location="cpu")
        return bundle["splits"], bundle["meta"]

    try:
        from torch_geometric.datasets import ZINC
    except Exception as exc:
        raise RuntimeError(
            "The precomputed cache was not found and torch_geometric is not importable. "
            "Re-run with --install-deps, or point --data-dir/--drive-dir at an existing "
            f"cache. Missing cache path: {path}"
        ) from exc

    data_root = args.data_dir or os.path.join(args.drive_dir if not args.no_drive else "/content", "datasets")
    log("[data] Loading PyG ZINC subset and precomputing pair features.")
    log(f"[data] Cache will be saved to: {path}")

    with TorchLoadCompat():
        raw = {
            "train": ZINC(root=data_root, subset=True, split="train"),
            "val": ZINC(root=data_root, subset=True, split="val"),
            "test": ZINC(root=data_root, subset=True, split="test"),
        }

    splits = {}
    t0 = time.time()
    for split_name, ds in raw.items():
        out = []
        n_graphs = len(ds)
        log(f"[data] Precomputing {split_name}: {n_graphs} graphs")
        for i, graph in enumerate(ds, start=1):
            out.append(compute_variant_a_features(graph))
            if i % 500 == 0 or i == n_graphs:
                log(f"[data]   {split_name}: {i:>5d}/{n_graphs} ({time.time() - t0:.1f}s)")
        splits[split_name] = out

    meta = {
        "dataset": "PyG-ZINC/subset",
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "num_atom_types": NUM_ATOM_TYPES,
        "num_bond_types": NUM_BOND_TYPES,
        "K_walk": K_WALK,
        "D_max_spd": D_MAX_SPD,
        "ring_min_size": RING_MIN_SIZE,
        "ring_max_size": RING_MAX_SIZE,
        "p_ring_pair": P_RING_PAIR,
        "p_sym": P_SYM,
        "p_pair": P_PAIR,
        "created_time": time.asctime(),
    }
    torch.save({"splits": splits, "meta": meta}, path)
    log(f"[data] Saved cache: {path}")
    return splits, meta


class TensorGraphDataset:
    def __init__(self, graphs: List[Dict[str, torch.Tensor]]):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


@dataclass
class VariantABatch:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor
    rwse: torch.Tensor
    pair_xi: torch.Tensor
    degree: torch.Tensor
    degree_log: torch.Tensor
    batch_index: torch.Tensor
    node_pos: torch.Tensor
    node_mask: torch.Tensor
    pair_mask: torch.Tensor
    num_graphs: int
    max_nodes: int

    def to(self, device):
        fields = {}
        for name, value in self.__dict__.items():
            fields[name] = value.to(device) if hasattr(value, "to") else value
        return VariantABatch(**fields)


def collate_variant_a(graphs: List[Dict[str, torch.Tensor]]) -> VariantABatch:
    batch_size = len(graphs)
    counts = [int(g["x"].numel()) for g in graphs]
    max_nodes = max(counts)
    total_nodes = sum(counts)

    xs, edge_indices, edge_attrs, ys, rwse_list, degree_list = [], [], [], [], [], []
    batch_index = torch.empty(total_nodes, dtype=torch.long)
    node_pos = torch.empty(total_nodes, dtype=torch.long)

    pair_xi = torch.zeros((batch_size, max_nodes, max_nodes, P_PAIR), dtype=torch.float32)
    node_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
    pair_mask = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)

    node_offset = 0
    for b, g in enumerate(graphs):
        n = counts[b]
        xs.append(g["x"])
        ys.append(g["y"])
        rwse_list.append(g["rwse"])
        degree_list.append(g["degree"])

        edge_indices.append(g["edge_index"] + node_offset)
        edge_attrs.append(g["edge_attr"])

        batch_index[node_offset:node_offset + n] = b
        node_pos[node_offset:node_offset + n] = torch.arange(n, dtype=torch.long)

        pair_xi[b, :n, :n] = g["pair_xi"]
        node_mask[b, :n] = True
        pair_mask[b, :n, :n] = True
        node_offset += n

    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.cat(edge_attrs, dim=0) if edge_attrs else torch.empty((0,), dtype=torch.long)
    degree = torch.cat(degree_list, dim=0).float()

    return VariantABatch(
        x=torch.cat(xs, dim=0).long(),
        edge_index=edge_index.long(),
        edge_attr=edge_attr.long(),
        y=torch.cat(ys, dim=0).float(),
        rwse=torch.cat(rwse_list, dim=0).float(),
        pair_xi=pair_xi.float(),
        degree=degree,
        degree_log=torch.log1p(degree),
        batch_index=batch_index,
        node_pos=node_pos,
        node_mask=node_mask,
        pair_mask=pair_mask,
        num_graphs=batch_size,
        max_nodes=max_nodes,
    )


# =============================================================================
# Model definition needed for checkpoint loading and inference.
# =============================================================================

def flat_to_dense(x, batch: VariantABatch):
    out = x.new_zeros((batch.num_graphs, batch.max_nodes, x.size(-1)))
    out[batch.batch_index, batch.node_pos] = x
    return out


def dense_to_flat(x_dense, batch: VariantABatch):
    return x_dense[batch.node_mask]


def mask_pair_state(pair_state, pair_mask):
    return pair_state.masked_fill(~pair_mask.unsqueeze(-1), 0.0)


class LastDimRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = RMS_NORM_EPS):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


class VariantALayer(torch.nn.Module):
    def __init__(self, use_value_add: bool = True, pair_update_rank: int = PAIR_UPDATE_RANK):
        super().__init__()
        import torch.nn as nn

        self.projection_mode = "normalized_latent_pair_memory"
        self.use_value_add = use_value_add
        self.pair_update_rank = int(pair_update_rank)
        if self.pair_update_rank <= 0:
            raise ValueError(f"pair_update_rank must be positive, got {self.pair_update_rank}")

        self.pre_attn_bn = nn.BatchNorm1d(D_MODEL)
        self.pre_ffn_bn = nn.BatchNorm1d(D_MODEL)

        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL)

        self.bias_weight = nn.Parameter(torch.empty(N_HEADS, BIAS_INPUT_DIM))
        self.value_weight = nn.Parameter(torch.empty(N_HEADS, D_HEAD, VALUE_INPUT_DIM))
        self.dynamic_bias_logit = nn.Parameter(torch.full((1,), logit_from_prob(DYNAMIC_GATE_INIT)))
        self.dynamic_value_logit = nn.Parameter(torch.full((1,), logit_from_prob(DYNAMIC_GATE_INIT)))
        nn.init.normal_(self.bias_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.value_weight, mean=0.0, std=0.02)

        self.bias_gate = nn.Parameter(torch.ones(N_HEADS))
        self.theta1 = nn.Parameter(torch.ones(D_MODEL))
        self.theta2 = nn.Parameter(torch.zeros(D_MODEL))

        self.attn_dropout = nn.Dropout(ATTN_DROPOUT)
        self.resid_dropout = nn.Dropout(RESID_DROPOUT)

        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, FFN_MULT * D_MODEL),
            nn.GELU(),
            nn.Dropout(RESID_DROPOUT),
            nn.Linear(FFN_MULT * D_MODEL, D_MODEL),
            nn.Dropout(RESID_DROPOUT),
        )

        r = self.pair_update_rank
        self.pair_read_norm = LastDimRMSNorm(P_LATENT_PAIR)
        self.pair_write_state_norm = LastDimRMSNorm(P_LATENT_PAIR)
        self.pair_proposal_norm = LastDimRMSNorm(P_LATENT_PAIR)
        self.h_pair_norm = LastDimRMSNorm(D_MODEL)
        self.pair_u_proj = nn.Linear(D_MODEL, r, bias=False)
        self.pair_v_proj = nn.Linear(D_MODEL, r, bias=False)
        self.pair_u_norm = LastDimRMSNorm(r)
        self.pair_v_norm = LastDimRMSNorm(r)
        self.pair_out = nn.Linear(r * r, P_LATENT_PAIR)
        self.pair_write_gate = nn.Linear(P_LATENT_PAIR + P_PAIR, 1)
        self.pair_update_log_scale = nn.Parameter(torch.full((1,), softplus_inverse(PAIR_UPDATE_SCALE_INIT)))

        nn.init.normal_(self.pair_out.weight, mean=0.0, std=PAIR_OUT_INIT_STD)
        nn.init.zeros_(self.pair_out.bias)
        nn.init.zeros_(self.pair_write_gate.weight)
        nn.init.constant_(self.pair_write_gate.bias, logit_from_prob(PAIR_WRITE_GATE_INIT))

    def dynamic_bias_gate(self):
        return torch.sigmoid(self.dynamic_bias_logit)

    def dynamic_value_gate(self):
        return torch.sigmoid(self.dynamic_value_logit)

    def pair_update_scale(self):
        import torch.nn.functional as F
        return F.softplus(self.pair_update_log_scale)

    def latent_pair_read(self, z_pair, xi_pair):
        z_read = self.pair_read_norm(z_pair)
        bias_dynamic = self.dynamic_bias_gate() * z_read
        value_dynamic = self.dynamic_value_gate() * z_read
        return xi_pair, z_read, bias_dynamic, value_dynamic

    def forward(self, h_flat, z_pair, batch: VariantABatch, ablation_keep_mask: Optional[torch.Tensor] = None):
        if ablation_keep_mask is not None:
            keep = (ablation_keep_mask & batch.pair_mask).unsqueeze(-1)
            z_pair = z_pair.masked_fill(~keep, 0.0)
        else:
            keep = None

        h_in_flat = h_flat
        h_norm_flat = self.pre_attn_bn(h_flat)
        h_norm = flat_to_dense(h_norm_flat, batch)

        B, N, _ = h_norm.shape
        q = self.q_proj(h_norm).view(B, N, N_HEADS, D_HEAD).transpose(1, 2)
        k = self.k_proj(h_norm).view(B, N, N_HEADS, D_HEAD).transpose(1, 2)
        v = self.v_proj(h_norm).view(B, N, N_HEADS, D_HEAD).transpose(1, 2)

        xi_pair, _z_read, bias_dynamic_pair, value_dynamic_pair = self.latent_pair_read(z_pair, batch.pair_xi)

        logits_base = torch.matmul(q, k.transpose(-2, -1)) * (D_HEAD ** -0.5)
        bias_in = torch.cat([xi_pair, bias_dynamic_pair], dim=-1)
        bias = torch.einsum("bijp,hp->bhij", bias_in, self.bias_weight)
        bias = bias * self.bias_gate.view(1, N_HEADS, 1, 1)
        logits = logits_base + bias
        logits = logits.masked_fill(~batch.node_mask[:, None, None, :], float("-inf"))

        alpha = torch.softmax(logits, dim=-1)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))

        out = torch.matmul(alpha, v)
        if self.use_value_add:
            value_in = torch.cat([xi_pair, value_dynamic_pair], dim=-1)
            value_avg = torch.einsum("bhij,bijp->bhip", alpha, value_in)
            value_contrib = torch.einsum("bhip,hdp->bhid", value_avg, self.value_weight)
            out = out + value_contrib

        out = out.transpose(1, 2).contiguous().view(B, N, D_MODEL)
        out = self.o_proj(out)
        out = out.masked_fill(~batch.node_mask.unsqueeze(-1), 0.0)
        out_flat = dense_to_flat(out, batch)

        degree_log = batch.degree_log.view(-1, 1)
        scaled = out_flat * self.theta1.view(1, -1) + degree_log * out_flat * self.theta2.view(1, -1)
        h_half = h_in_flat + self.resid_dropout(scaled)

        h_ffn = self.ffn(self.pre_ffn_bn(h_half))
        h_out = h_half + h_ffn

        h_post = flat_to_dense(h_half, batch)
        h_post = self.h_pair_norm(h_post).masked_fill(~batch.node_mask.unsqueeze(-1), 0.0)
        u = self.pair_u_norm(self.pair_u_proj(h_post))
        vv = self.pair_v_norm(self.pair_v_proj(h_post))
        outer = (u[:, :, None, :, None] * vv[:, None, :, None, :]).flatten(start_dim=-2)
        proposal = self.pair_out(outer) * (self.pair_update_rank ** -0.5)
        proposal = self.pair_proposal_norm(proposal)
        proposal = proposal.masked_fill(~batch.pair_mask.unsqueeze(-1), 0.0)

        z_gate_state = self.pair_write_state_norm(z_pair.masked_fill(~batch.pair_mask.unsqueeze(-1), 0.0))
        z_gate_state = z_gate_state.masked_fill(~batch.pair_mask.unsqueeze(-1), 0.0)
        write_gate = torch.sigmoid(self.pair_write_gate(torch.cat([z_gate_state, xi_pair], dim=-1)))
        write_gate = write_gate.masked_fill(~batch.pair_mask.unsqueeze(-1), 0.0)
        z_next = z_pair + self.pair_update_scale() * write_gate * proposal
        z_next = mask_pair_state(z_next, batch.pair_mask)
        if keep is not None:
            z_next = z_next.masked_fill(~keep, 0.0)

        return h_out, z_next


class VariantAZincModel(torch.nn.Module):
    def __init__(self, use_value_add: bool = True, pair_update_rank: int = PAIR_UPDATE_RANK):
        super().__init__()
        import torch.nn as nn

        self.projection_mode = "normalized_latent_pair_memory"
        self.use_value_add = use_value_add
        self.pair_update_rank = int(pair_update_rank)

        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES, D_MODEL)
        self.bond_emb = nn.Embedding(NUM_BOND_TYPES, D_MODEL)
        self.rwse_proj = nn.Linear(K_WALK, D_MODEL)
        self.layers = nn.ModuleList([
            VariantALayer(use_value_add=use_value_add, pair_update_rank=self.pair_update_rank)
            for _ in range(N_LAYERS)
        ])
        self.final_bn = nn.BatchNorm1d(D_MODEL)
        self.readout = nn.Sequential(
            nn.Linear(D_MODEL, 2 * D_MODEL),
            nn.GELU(),
            nn.Linear(2 * D_MODEL, 1),
        )

    def forward(self, batch: VariantABatch):
        pred, _, _, _ = forward_capture_z(self, batch)
        return pred


def forward_capture_z(
    model: VariantAZincModel,
    batch: VariantABatch,
    ablation_keep_mask: Optional[torch.Tensor] = None,
):
    h = model.atom_emb(batch.x)
    edge_msg = model.bond_emb(batch.edge_attr)
    bond_sum = h.new_zeros(h.shape)
    if batch.edge_index.numel() > 0:
        bond_sum.index_add_(0, batch.edge_index[1], edge_msg)

    h = h + bond_sum + model.rwse_proj(batch.rwse)
    z = batch.pair_xi.new_zeros(batch.pair_xi.shape[:-1] + (P_LATENT_PAIR,))
    z_by_layer = []

    for layer in model.layers:
        h, z = layer(h, z, batch, ablation_keep_mask=ablation_keep_mask)
        z_by_layer.append(z.detach())

    h = model.final_bn(h)
    pooled = h.new_zeros((batch.num_graphs, D_MODEL))
    pooled.index_add_(0, batch.batch_index, h)
    pred = model.readout(pooled).view(-1)
    return pred, z_by_layer, batch.pair_xi.detach(), batch.pair_mask.detach()


# =============================================================================
# Analysis.
# =============================================================================

def resolve_checkpoint(args: argparse.Namespace) -> str:
    if args.ckpt:
        if not os.path.isfile(args.ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
        return args.ckpt

    candidates = [
        os.path.join(args.checkpoint_dir, "checkpoint_best.pt"),
        os.path.join(args.checkpoint_dir, "checkpoint_latest.pt"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "No checkpoint found. Pass --ckpt explicitly, or ensure --checkpoint-dir "
        f"contains checkpoint_best.pt or checkpoint_latest.pt. Tried: {candidates}"
    )


def build_model_from_checkpoint(ckpt: Dict[str, object], device: torch.device):
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    model = VariantAZincModel(
        use_value_add=not bool(ckpt_args.get("disable_value_add", False)),
        pair_update_rank=int(ckpt_args.get("pair_update_rank", PAIR_UPDATE_RANK)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def analyse(model: VariantAZincModel, loader, device: torch.device, num_batches: int):
    spd_off = P_RING_PAIR
    spd_c = D_MAX_SPD + 1

    z_sq = torch.zeros(N_LAYERS, spd_c, dtype=torch.float64)
    z_cnt = torch.zeros(N_LAYERS, spd_c, dtype=torch.long)
    dyn_sq = torch.zeros(N_LAYERS, spd_c, dtype=torch.float64)
    chan_sq = torch.zeros(N_LAYERS, P_LATENT_PAIR, dtype=torch.float64)
    chan_cnt = torch.zeros(N_LAYERS, dtype=torch.long)
    total_graphs = 0

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= num_batches:
                break
            batch = batch.to(device)
            _pred, z_by_layer, xi, mask = forward_capture_z(model, batch)
            total_graphs += batch.num_graphs

            spd = xi[..., spd_off:spd_off + spd_c].argmax(dim=-1)

            for layer_idx, layer in enumerate(model.layers):
                z = z_by_layer[layer_idx]
                zsq_pair = (z ** 2).sum(dim=-1)

                z_read = layer.pair_read_norm(z)
                dyn_in = layer.dynamic_bias_gate() * z_read
                w_dyn = layer.bias_weight[:, P_PAIR:]
                head_g = layer.bias_gate.view(1, -1, 1, 1)
                dyn_log = torch.einsum("bijp,hp->bhij", dyn_in, w_dyn) * head_g
                dyn_sq_pair = (dyn_log ** 2).mean(dim=1)

                for c in range(spd_c):
                    sel = mask & (spd == c)
                    n = int(sel.sum().item())
                    if n:
                        z_sq[layer_idx, c] += zsq_pair[sel].double().sum().cpu()
                        z_cnt[layer_idx, c] += n
                        dyn_sq[layer_idx, c] += dyn_sq_pair[sel].double().sum().cpu()

                selected_z = z[mask]
                if selected_z.numel() > 0:
                    chan_sq[layer_idx] += (selected_z ** 2).double().sum(dim=0).cpu()
                    chan_cnt[layer_idx] += int(selected_z.size(0))

    return {
        "z_sq": z_sq,
        "z_cnt": z_cnt,
        "dyn_sq": dyn_sq,
        "chan_sq": chan_sq,
        "chan_cnt": chan_cnt,
        "total_graphs": total_graphs,
    }


def spd_classes_from_batch(batch: VariantABatch) -> torch.Tensor:
    spd_off = P_RING_PAIR
    spd_c = D_MAX_SPD + 1
    return batch.pair_xi[..., spd_off:spd_off + spd_c].argmax(dim=-1)


def ablation_names(suite: str) -> List[str]:
    compact = [
        "baseline",
        "no_latent_all",
        "drop_self",
        "drop_edge",
        "drop_non_edge_spd_ge_2",
        "drop_far_spd_ge_5",
        "keep_self_edge",
        "keep_non_edge_spd_ge_2",
        "keep_edge",
        "keep_far_spd_ge_5",
    ]
    if suite == "compact":
        return compact
    if suite != "full":
        raise ValueError(f"Unknown ablation suite: {suite}")
    return compact + [
        "drop_spd2",
        "drop_spd3",
        "drop_spd4",
        "keep_self",
        "keep_spd2",
        "keep_spd3",
        "keep_spd4",
    ]


def make_ablation_keep_mask(batch: VariantABatch, name: str) -> Optional[torch.Tensor]:
    """Return True for pairs whose dynamic latent z path remains active.

    The fixed xi^0 structural path is never ablated here. This isolates the
    learned latent-memory contribution by distance category.
    """
    if name == "baseline":
        return None

    spd = spd_classes_from_batch(batch)
    valid = batch.pair_mask

    if name == "no_latent_all":
        return torch.zeros_like(valid)
    if name == "drop_self":
        return valid & (spd != 0)
    if name == "drop_edge":
        return valid & (spd != 1)
    if name == "drop_spd2":
        return valid & (spd != 2)
    if name == "drop_spd3":
        return valid & (spd != 3)
    if name == "drop_spd4":
        return valid & (spd != 4)
    if name == "drop_far_spd_ge_5":
        return valid & (spd != 5)
    if name == "drop_non_edge_spd_ge_2":
        return valid & (spd < 2)
    if name == "drop_self_edge":
        return valid & (spd >= 2)

    if name == "keep_self":
        return valid & (spd == 0)
    if name == "keep_edge":
        return valid & (spd == 1)
    if name == "keep_spd2":
        return valid & (spd == 2)
    if name == "keep_spd3":
        return valid & (spd == 3)
    if name == "keep_spd4":
        return valid & (spd == 4)
    if name == "keep_far_spd_ge_5":
        return valid & (spd == 5)
    if name == "keep_non_edge_spd_ge_2":
        return valid & (spd >= 2)
    if name == "keep_self_edge":
        return valid & (spd <= 1)

    raise ValueError(f"Unknown ablation name: {name}")


@torch.no_grad()
def eval_mae_with_ablation(
    model: VariantAZincModel,
    loader,
    device: torch.device,
    ablation_name: str,
    max_batches: int = 0,
) -> Tuple[float, int, int]:
    model.eval()
    total_abs = 0.0
    total_graphs = 0
    batches_seen = 0

    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        batch = batch.to(device)
        keep_mask = make_ablation_keep_mask(batch, ablation_name)
        pred, _z_by_layer, _xi, _mask = forward_capture_z(model, batch, ablation_keep_mask=keep_mask)
        total_abs += float((pred - batch.y).abs().sum().detach().cpu())
        total_graphs += batch.num_graphs
        batches_seen += 1

    if total_graphs == 0:
        return float("nan"), 0, batches_seen
    return total_abs / total_graphs, total_graphs, batches_seen


def run_test_ablations(
    model: VariantAZincModel,
    loader,
    device: torch.device,
    suite: str,
    max_batches: int,
) -> List[Dict[str, object]]:
    names = ablation_names(suite)
    results = []
    baseline_mae = None

    print()
    print("== (5) Dynamic latent-memory MAE ablations ==")
    print("Ablates z_ij only; fixed xi^0 SPD/ring/bond features remain active.")
    print("ablation                     graphs  batches   mae        delta_vs_baseline")

    for name in names:
        mae, graphs, batches = eval_mae_with_ablation(
            model,
            loader,
            device,
            ablation_name=name,
            max_batches=max_batches,
        )
        if name == "baseline":
            baseline_mae = mae
        delta = mae - baseline_mae if baseline_mae is not None and math.isfinite(mae) else float("nan")
        results.append({
            "ablation": name,
            "graphs": graphs,
            "batches": batches,
            "mae": mae,
            "delta_vs_baseline": delta,
        })
        delta_text = f"{delta:+.6f}" if math.isfinite(delta) else "nan"
        print(f"{name:<28s}  {graphs:>6d}  {batches:>7d}  {mae:.6f}   {delta_text:>16s}")

    return results


def fmt(v, width: int = 10):
    value = float(v)
    if math.isnan(value):
        return f"{'nan':>{width}}"
    return f"{value:{width}.3e}"


def print_report(stats: Dict[str, torch.Tensor]) -> Dict[str, object]:
    labels = ["self", "edge", "spd=2", "spd=3", "spd=4", "spd>=5"]
    z_sq = stats["z_sq"]
    z_cnt = stats["z_cnt"]
    dyn_sq = stats["dyn_sq"]
    chan_sq = stats["chan_sq"]
    chan_cnt = stats["chan_cnt"]

    z_rms = torch.zeros(N_LAYERS, D_MAX_SPD + 1)
    dyn_rms = torch.zeros(N_LAYERS, D_MAX_SPD + 1)
    locality = []
    channel_util = []

    print()
    print("== (1) RMS of z_ij by SPD bin per layer ==")
    print("layer  " + "  ".join(f"{label:>10}" for label in labels))
    for layer_idx in range(N_LAYERS):
        row = []
        for c in range(D_MAX_SPD + 1):
            n = int(z_cnt[layer_idx, c].item())
            r = math.sqrt(z_sq[layer_idx, c].item() / (n * P_LATENT_PAIR)) if n else float("nan")
            z_rms[layer_idx, c] = r
            row.append(r)
        print(f"  L{layer_idx:02d}  " + "  ".join(fmt(v) for v in row))

    print()
    print("== (2) Locality ratios per layer (>1 = concentrated locally) ==")
    print("layer   edge/far     edge/avg(spd>=3)   self/edge")
    for layer_idx in range(N_LAYERS):
        edge = float(z_rms[layer_idx, 1].item())
        far = float(z_rms[layer_idx, 5].item())
        mid = float(z_rms[layer_idx, 3:6].mean().item())
        self_r = float(z_rms[layer_idx, 0].item())
        edge_far = edge / far if far > 0 else float("nan")
        edge_mid = edge / mid if mid > 0 else float("nan")
        self_edge = self_r / edge if edge > 0 else float("nan")
        locality.append({
            "layer": layer_idx,
            "edge_over_far": edge_far,
            "edge_over_avg_spd_ge_3": edge_mid,
            "self_over_edge": self_edge,
        })
        print(f"  L{layer_idx:02d}    {edge_far:8.2f}    {edge_mid:14.2f}    {self_edge:8.2f}")

    print()
    print("== (3) Effective dynamic-bias logit RMS by SPD (what attention actually sees) ==")
    print("layer  " + "  ".join(f"{label:>10}" for label in labels))
    for layer_idx in range(N_LAYERS):
        row = []
        for c in range(D_MAX_SPD + 1):
            n = int(z_cnt[layer_idx, c].item())
            r = math.sqrt(dyn_sq[layer_idx, c].item() / n) if n else float("nan")
            dyn_rms[layer_idx, c] = r
            row.append(r)
        print(f"  L{layer_idx:02d}  " + "  ".join(fmt(v) for v in row))

    print()
    print("== (4) Channel utilisation of z per layer ==")
    print("layer  max_rms     active(>5% max)    effective#chans (exp-entropy)")
    for layer_idx in range(N_LAYERS):
        n = int(chan_cnt[layer_idx].item())
        if not n:
            continue
        per_c = (chan_sq[layer_idx] / n).sqrt()
        pmax = float(per_c.max().item())
        n_act = int((per_c > 0.05 * pmax).sum().item())
        p = per_c / per_c.sum().clamp_min(1e-12)
        eff_n = math.exp(-(p * p.clamp_min(1e-12).log()).sum().item())
        channel_util.append({
            "layer": layer_idx,
            "max_rms": pmax,
            "active_gt_5pct_max": n_act,
            "effective_channels": eff_n,
        })
        print(f"  L{layer_idx:02d}  {pmax:.3e}    {n_act:>2}/{P_LATENT_PAIR}                  {eff_n:5.1f}")

    return {
        "labels": labels,
        "z_rms": z_rms.tolist(),
        "dynamic_bias_logit_rms": dyn_rms.tolist(),
        "locality": locality,
        "channel_utilisation": channel_util,
        "total_graphs": int(stats["total_graphs"]),
    }


def strip_colab_kernel_args(argv: Optional[Sequence[str]]) -> Optional[List[str]]:
    if argv is None:
        return None
    cleaned = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "-f" and i + 1 < len(argv) and "kernel-" in argv[i + 1]:
            i += 2
            continue
        if tok.startswith("/root/.local/share/jupyter/runtime/kernel-"):
            i += 1
            continue
        cleaned.append(tok)
        i += 1
    return cleaned


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(
        description="Standalone latent-pair-memory locality analysis for the ZINC checkpoint."
    )
    parser.add_argument("--drive-mount", default="/content/drive")
    parser.add_argument("--drive-dir", default="/content/drive/MyDrive/variant_a_zinc")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--checkpoint-dir", default="/content/drive/MyDrive/variant_a_zinc/results/LatentPairMemory.ValueActive/checkpoints")
    parser.add_argument("--ckpt", default=None, help="Explicit checkpoint path. Overrides --checkpoint-dir.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-ablations", action="store_true", help="Evaluate dynamic z ablations on --ablation-split.")
    parser.add_argument("--ablation-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--ablation-suite", choices=["compact", "full"], default="compact")
    parser.add_argument(
        "--ablation-num-batches",
        type=int,
        default=0,
        help="Number of ablation batches to evaluate. 0 means the full split.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-drive", action="store_true")
    parser.add_argument("--force-recompute-cache", action="store_true")
    parser.add_argument("--install-deps", action="store_true", help="Install torch_geometric if the cache needs recomputing.")
    parser.add_argument("--skip-install", action="store_true", help="Compatibility flag; disables --install-deps if both are passed.")
    args, unknown = parser.parse_known_args(strip_colab_kernel_args(argv))
    if unknown:
        raise SystemExit(f"Unrecognized arguments: {' '.join(unknown)}")
    if args.num_batches <= 0:
        raise SystemExit("--num-batches must be positive.")
    if args.ablation_num_batches < 0:
        raise SystemExit("--ablation-num-batches must be >= 0.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    mount_drive_if_available(args.drive_mount, args.no_drive)
    install_deps_if_requested(args.skip_install, args.install_deps)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    log(f"[env] device={device}")
    if device.type == "cuda":
        log(f"[env] gpu={torch.cuda.get_device_name(0)}")

    ckpt_path = resolve_checkpoint(args)
    log(f"[ckpt] {ckpt_path}")
    ckpt = safe_torch_load(ckpt_path, map_location=device)
    best = ckpt.get("best", {})
    if isinstance(best, dict):
        log(
            f"[ckpt] epoch={ckpt.get('epoch', '?')}  "
            f"best_val={float(best.get('val_mae', float('nan'))):.4f}  "
            f"test@best={float(best.get('test_mae_at_best', float('nan'))):.4f}"
        )
    else:
        log(f"[ckpt] epoch={ckpt.get('epoch', '?')}")

    model = build_model_from_checkpoint(ckpt, device)

    splits, meta = load_or_precompute_dataset(args)
    dataset = TensorGraphDataset(splits[args.split])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_variant_a,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    log(f"[data] split={args.split} graphs={len(dataset)} meta={meta.get('dataset', 'unknown')}")
    log(f"[analysis] batches={min(args.num_batches, len(loader))} batch_size={args.batch_size}")

    stats = analyse(model, loader, device, args.num_batches)
    report = print_report(stats)

    ablation_results = None
    if args.run_ablations:
        ablation_dataset = TensorGraphDataset(splits[args.ablation_split])
        ablation_loader = torch.utils.data.DataLoader(
            ablation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_variant_a,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        planned_batches = len(ablation_loader)
        if args.ablation_num_batches > 0:
            planned_batches = min(planned_batches, args.ablation_num_batches)
        log(
            f"[ablations] split={args.ablation_split} graphs={len(ablation_dataset)} "
            f"batches={planned_batches} suite={args.ablation_suite}"
        )
        ablation_results = run_test_ablations(
            model,
            ablation_loader,
            device,
            suite=args.ablation_suite,
            max_batches=args.ablation_num_batches,
        )

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        payload = {
            "checkpoint": ckpt_path,
            "split": args.split,
            "num_batches": args.num_batches,
            "batch_size": args.batch_size,
            "report": report,
            "ablation_split": args.ablation_split if args.run_ablations else None,
            "ablation_suite": args.ablation_suite if args.run_ablations else None,
            "ablation_num_batches": args.ablation_num_batches if args.run_ablations else None,
            "ablations": ablation_results,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        log(f"\n[json] wrote {args.output_json}")

    print()
    print("[done]")


if __name__ == "__main__":
    main()
