# Extracted from CSA_ZINC_core.ipynb
# Keep the notebook and script in sync when changing experiment logic.

# -*- coding: utf-8 -*-
"""static_anchor_symmetric_zinc.py

Controlled ablation derived from Variant A1.5 (anchored-residual P2 typed
structural-action graph transformer). Strips back to a fixed symmetric
structural anchor with the same additive-bias + symmetric-value-addition
mechanics as A1.5.

Motivation
----------
A1.5 diagnostics on the trained model showed:
  - rotation channel ~vestigial (KL(full||no_rot) ≈ 0.16, rot gate ≈ 0.04)
  - pair-MLP evolution wasted (drift ≈ 10x, gates ~0.04/0.09/0.001)
  - bias dominates content in logits, value/base ≈ 0.68 from xi^0 path
  - attention behaviour and structural read essentially explained by
    xi^0 + content attention + value addition from xi^0.

This variant tests the prediction that A1.5 performance is reproduced by:
  fixed symmetric xi^0 -> additive logit bias + symmetric value addition,
  with no evolution and no antisymmetric / rotation channel.

All other machinery is held identical to A1.5 to preserve a fair comparison.

Held identical to A1.5
----------------------
  - Dataset: PyG ZINC subset, MAE/L1.
  - L = 10, d = 64, H = 8, d_h = 8.
  - K = 12 random-walk truncation.
  - Node features: atom_emb(x_i) + sum_j edge_emb(f_ji) + Linear(rwse_i),
    where rwse_i = [(W^k)_{ii}]_{k=1..K} (RWSE diagonal injection).
  - BatchNorm pre-attention and pre-FFN.
  - Degree scaler after MSA:
        o'_i = o_i * theta_1 + log(1+d_i) * o_i * theta_2,
    with theta_1 init 1, theta_2 init 0.
  - Symmetric additive logit bias:
        z_ij^h = q_i^h . k_j^h / sqrt(d_h) + bias_gate^h * b_h(xi^0_ij).
  - Symmetric value addition (default on):
        o_i^h = sum_j alpha_ij^h * (v_j^h + W_Ev^h . xi^0_ij).
  - Sum pool -> MLP [d -> 2d -> 1] with GELU.
  - AdamW lr 1e-3, wd 1e-5, warmup 50, cosine decay, 2000 epochs, batch 32.

Differences from A1.5
---------------------
  - Antisymmetric channel removed entirely:
        no e_a, no (W^k)_ij - (W^k)_ji features, no SO(2) rotation pathway,
        no rot_weight, no rotation gates.
  - Pair state is fixed: e_ij = xi^0_ij in every layer.
        no pair_mlp, no PAIR_UPDATE_ETA, no pair-state update.
  - No anchored-residual read: bias and value maps read xi^0_ij directly
        (no [xi^0, gate * Pi(e - xi^0)] doubling).
  - No typed-symmetry projection (xi^0_sym is symmetric by construction:
        SPD is symmetric, walk-sum W^k + W^k.T is symmetric by definition,
        bond_oh is symmetric on undirected ZINC).
  - No dynamic_{rot,bias,value}_gate parameters.

Symmetric feature set (xi^0_ij)
-------------------------------
  - SPD one-hot clipped at D_max=5, dim 6.
  - Symmetric walk-sum [(W^k)_{ij} + (W^k)_{ji}]_{k=1..K}, dim 12.
  - Bond-type one-hot (0 for non-edges), dim 3.
  Total p_sym = 21. p_pair = p_sym (no asymmetric component).

Run in Colab:
    from static_anchor_symmetric_zinc import main
    main([])

Resume without reinstall:
    main(["--skip-install"])

Ablate symmetric value addition (-> pure structural bias only):
    main(["--disable-symmetric-value-add"])
"""

import argparse
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


# =============================================================================
# Constants — held identical to A1.5 except where the antisymmetric channel
# and pair-evolution machinery are removed.
# =============================================================================

NUM_ATOM_TYPES = 28
NUM_BOND_TYPES = 3

D_MODEL = 64
N_HEADS = 8
N_LAYERS = 10
D_HEAD = D_MODEL // N_HEADS

K_WALK = 12
D_MAX_SPD = 5

# Symmetric-only pair feature set: SPD + walk-sum + bond.
P_SYM = (D_MAX_SPD + 1) + K_WALK + NUM_BOND_TYPES
P_PAIR = P_SYM  # no antisymmetric component in this variant

# Read paths use xi^0 directly (no anchored-residual doubling).
BIAS_INPUT_DIM = P_SYM
VALUE_INPUT_DIM = P_SYM

USE_SYMMETRIC_VALUE_ADD = True

FFN_MULT = 2
ATTN_DROPOUT = 0.2
RESID_DROPOUT = 0.0

LR = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 50
MAX_EPOCHS = 2000
BATCH_SIZE = 32
GRAD_CLIP = 1.0
MIN_LR = 0.0

EXPECTED_PARAM_CEILING = 550_000


# =============================================================================
# Utility
# =============================================================================

def log(msg: str = "") -> None:
    print(msg, flush=True)


def run_cmd(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> None:
    log(f"[cmd] {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=cwd, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


# =============================================================================
# Args
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train static-anchor symmetric-only graph transformer on ZINC subset (controlled ablation of Variant A1.5)."
    )
    parser.add_argument("--drive-mount", default="/content/drive")
    parser.add_argument("--drive-dir", default="/content/drive/MyDrive/static_anchor_symmetric_zinc")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--ckpt-period", type=int, default=100)
    parser.add_argument("--console-epoch-period", type=int, default=1)
    parser.add_argument("--diagnostic-period", type=int, default=25)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--force-recompute-cache", action="store_true")
    parser.add_argument("--auto-resume", dest="auto_resume", action="store_true", default=True)
    parser.add_argument("--no-auto-resume", dest="auto_resume", action="store_false")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--disable-symmetric-value-add",
        action="store_true",
        help="Ablation: disable symmetric value-level additive contribution (pure structural-bias-only variant).",
    )
    parser.add_argument("--allow-over-500k", action="store_true")
    parser.add_argument("--no-drive", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--name-tag", default="StaticAnchor.SymBiasValue")
    parser.add_argument("--quiet-setup", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    filtered_unknown = []
    skip_next = False
    for i, tok in enumerate(unknown):
        if skip_next:
            skip_next = False
            continue
        if tok == "-f" and i + 1 < len(unknown) and "kernel-" in unknown[i + 1]:
            log(f"[args] Ignoring Colab/Jupyter kernel argument: {tok} {unknown[i + 1]}")
            skip_next = True
        elif tok.startswith("/root/.local/share/jupyter/runtime/kernel-"):
            log(f"[args] Ignoring Colab/Jupyter kernel argument: {tok}")
        else:
            filtered_unknown.append(tok)
    if filtered_unknown:
        raise SystemExit(f"Unrecognized arguments: {' '.join(filtered_unknown)}")
    return args


# =============================================================================
# Environment / install (kept identical to A1.5 for consistency)
# =============================================================================

def mount_drive_if_needed(args: argparse.Namespace) -> None:
    if args.no_drive:
        log("[drive] --no-drive set; using local runtime storage.")
        return
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        log("[drive] Not in Colab; skipping mount.")
        return
    if not os.path.isdir(args.drive_mount):
        os.makedirs(args.drive_mount, exist_ok=True)
    if not os.path.ismount(args.drive_mount):
        log(f"[drive] Mounting Google Drive at {args.drive_mount}")
        drive.mount(args.drive_mount, force_remount=False)
    os.makedirs(args.drive_dir, exist_ok=True)


def install_deps(args: argparse.Namespace) -> None:
    if args.skip_install:
        log("[install] --skip-install set; assuming PyG is already installed.")
        return
    try:
        import torch_geometric  # noqa: F401
        log("[install] torch_geometric already importable; skipping pip install.")
        return
    except Exception:
        pass
    log("[install] Installing PyG (torch-geometric) ...")
    cmd = [sys.executable, "-m", "pip", "install", "-q", "torch-geometric"]
    if args.quiet_setup:
        with open(os.devnull, "w") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    else:
        subprocess.run(cmd, check=True)


# =============================================================================
# torch.load compatibility shim (kept identical to A1.5)
# =============================================================================

class TorchLoadCompat:
    """Make torch.load default to weights_only=False on torch versions that flipped the default."""

    def __enter__(self):
        import torch as _torch
        self._orig = _torch.load

        def patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return self._orig(*args, **kwargs)

        _torch.load = patched_load
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import torch as _torch
        _torch.load = self._orig


def safe_torch_load(path: str, map_location=None):
    import torch
    with TorchLoadCompat():
        return torch.load(path, map_location=map_location)


# =============================================================================
# Feature precomputation
# Symmetric-only pair features: SPD + walk-sum + bond. No antisymmetric channel.
# RWSE diagonal kept identical to A1.5.
# =============================================================================

def normalize_edge_attr(edge_attr):
    import torch
    ea = edge_attr.view(-1).long()
    if ea.numel() == 0:
        return ea
    mn, mx = int(ea.min().item()), int(ea.max().item())
    if mn >= 1 and mx <= NUM_BOND_TYPES:
        ea = ea - 1
    if int(ea.min().item()) < 0 or int(ea.max().item()) >= NUM_BOND_TYPES:
        raise ValueError(
            f"Expected {NUM_BOND_TYPES} bond types encoded as 0..{NUM_BOND_TYPES-1} "
            f"or 1..{NUM_BOND_TYPES}; got min={int(ea.min())}, max={int(ea.max())}."
        )
    return ea


def bfs_spd_classes(adj_bool, dmax: int):
    """All-pairs shortest-path classes clipped at dmax."""
    import torch
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
                if dist[s, v] == dmax and v != s:
                    dist[s, v] = min(du + 1, dmax)
                    q.append(v)
    return dist


def compute_pair_features(data) -> Dict[str, "torch.Tensor"]:
    """Return a plain tensor dict with x, edge_index, edge_attr, y, rwse, pair_xi, degree.

    pair_xi has only the symmetric channel: SPD one-hot + walk-sum + bond one-hot.
    """
    import torch
    import torch.nn.functional as F

    x = data.x.view(-1).long()
    if int(x.min()) < 0 or int(x.max()) >= NUM_ATOM_TYPES:
        raise ValueError(f"Atom type out of expected 0..{NUM_ATOM_TYPES-1}: max={int(x.max())}")

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

    # Node-level RWSE diagonal injection (unchanged from A1.5).
    rwse = torch.stack([Pk.diagonal() for Pk in powers], dim=-1).float()

    # Pair-level symmetric walk-sum only (no antisymmetric).
    walk_sum = torch.stack([Pk + Pk.t() for Pk in powers], dim=-1).float()

    spd_cls = bfs_spd_classes(A > 0, D_MAX_SPD)
    spd_oh = F.one_hot(spd_cls.clamp(max=D_MAX_SPD), num_classes=D_MAX_SPD + 1).float()

    bond_oh = torch.zeros((n, n, NUM_BOND_TYPES), dtype=torch.float32)
    if edge_index.numel() > 0:
        src, dst = edge_index
        bond_oh[src, dst, edge_attr] = 1.0

    # Sanity check: bond is symmetric on ZINC because of reciprocal directed edges.
    # SPD is symmetric by construction. walk_sum is symmetric by construction.
    pair_xi = torch.cat([spd_oh, walk_sum, bond_oh], dim=-1).float()
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
    data_dir = args.data_dir or os.path.join(args.drive_dir if not args.no_drive else "/content", "datasets")
    return os.path.join(
        data_dir, "static_anchor_cache",
        f"zinc_subset_static_anchor_K{K_WALK}_D{D_MAX_SPD}_v1.pt"
    )


def load_or_precompute_dataset(args: argparse.Namespace):
    import torch
    from torch_geometric.datasets import ZINC

    path = cache_path(args)
    if os.path.isfile(path) and not args.force_recompute_cache:
        log(f"[data] Loading cached precomputed dataset: {path}")
        cached = safe_torch_load(path, map_location="cpu")
        return cached["splits"], cached["meta"]

    log("[data] Precomputing pair features from PyG ZINC subset ...")
    root = os.path.join(args.drive_dir if not args.no_drive else "/content", "datasets", "ZINC")
    os.makedirs(root, exist_ok=True)
    train = ZINC(root=root, subset=True, split="train")
    val = ZINC(root=root, subset=True, split="val")
    test = ZINC(root=root, subset=True, split="test")
    splits = {
        "train": [compute_pair_features(d) for d in train],
        "val": [compute_pair_features(d) for d in val],
        "test": [compute_pair_features(d) for d in test],
    }
    meta = {
        "num_atom_types": NUM_ATOM_TYPES,
        "num_bond_types": NUM_BOND_TYPES,
        "k_walk": K_WALK,
        "d_max_spd": D_MAX_SPD,
        "p_sym": P_SYM,
        "p_pair": P_PAIR,
        "sizes": {k: len(v) for k, v in splits.items()},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"splits": splits, "meta": meta}, path)
    log(f"[data] Cached precomputed dataset to: {path}")
    log(f"[data] Sizes: {meta['sizes']}")
    return splits, meta


# =============================================================================
# Batching
# =============================================================================

class TensorGraphDataset:
    def __init__(self, graphs: List[Dict[str, "torch.Tensor"]]):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


@dataclass
class StaticAnchorBatch:
    x: "torch.Tensor"
    edge_index: "torch.Tensor"
    edge_attr: "torch.Tensor"
    y: "torch.Tensor"
    rwse: "torch.Tensor"
    pair_xi: "torch.Tensor"
    degree: "torch.Tensor"
    batch_index: "torch.Tensor"
    node_pos: "torch.Tensor"
    node_mask: "torch.Tensor"
    pair_mask: "torch.Tensor"
    num_graphs: int
    max_nodes: int

    def to(self, device):
        fields = {}
        for name, value in self.__dict__.items():
            if hasattr(value, "to"):
                fields[name] = value.to(device)
            else:
                fields[name] = value
        return StaticAnchorBatch(**fields)


def collate_static_anchor(graphs: List[Dict[str, "torch.Tensor"]]) -> StaticAnchorBatch:
    import torch

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

        ei = g["edge_index"] + node_offset
        edge_indices.append(ei)
        edge_attrs.append(g["edge_attr"])

        batch_index[node_offset:node_offset + n] = b
        node_pos[node_offset:node_offset + n] = torch.arange(n, dtype=torch.long)

        pair_xi[b, :n, :n] = g["pair_xi"]
        node_mask[b, :n] = True
        pair_mask[b, :n, :n] = True
        node_offset += n

    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.cat(edge_attrs, dim=0) if edge_attrs else torch.empty((0,), dtype=torch.long)

    return StaticAnchorBatch(
        x=torch.cat(xs, dim=0).long(),
        edge_index=edge_index.long(),
        edge_attr=edge_attr.long(),
        y=torch.cat(ys, dim=0).float(),
        rwse=torch.cat(rwse_list, dim=0).float(),
        pair_xi=pair_xi.float(),
        degree=torch.cat(degree_list, dim=0).float(),
        batch_index=batch_index,
        node_pos=node_pos,
        node_mask=node_mask,
        pair_mask=pair_mask,
        num_graphs=batch_size,
        max_nodes=max_nodes,
    )


# =============================================================================
# Model
# =============================================================================

def flat_to_dense(x, batch: StaticAnchorBatch):
    import torch
    out = x.new_zeros((batch.num_graphs, batch.max_nodes, x.size(-1)))
    out[batch.batch_index, batch.node_pos] = x
    return out


def dense_to_flat(x_dense, batch: StaticAnchorBatch):
    return x_dense[batch.node_mask]


class StaticAnchorLayer(__import__("torch").nn.Module):
    """One transformer layer with fixed symmetric-anchor structural injection.

    The structural pair tensor is xi^0 (fixed), entering the attention layer as:
      - an additive scalar logit bias per head: bias_gate^h * b_h(xi^0_ij);
      - (optionally) a symmetric value addition per head: W_Ev^h . xi^0_ij,
        added to v_j^h inside the alpha_ij^h-weighted sum.
    All other components (BN, Q/K/V/O, FFN, degree scaler, dropouts) are
    identical to the Variant A1.5 layer with rotation and pair-evolution removed.
    """

    def __init__(self, use_symmetric_value_add: bool = True):
        import torch
        import torch.nn as nn

        super().__init__()
        self.use_symmetric_value_add = use_symmetric_value_add
        self.pre_attn_bn = nn.BatchNorm1d(D_MODEL)
        self.pre_ffn_bn = nn.BatchNorm1d(D_MODEL)

        self.q_proj = nn.Linear(D_MODEL, D_MODEL)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL)

        # Structural read: bias and value maps applied to xi^0_sym directly.
        # Shapes match the xi^0-reading slice of the A1.5 anchored read.
        self.bias_weight = nn.Parameter(torch.empty(N_HEADS, BIAS_INPUT_DIM))
        nn.init.normal_(self.bias_weight, mean=0.0, std=0.02)
        # value_weight is only registered when the symmetric value addition is enabled,
        # so the bias-only ablation truly removes the pathway (and its parameters).
        if use_symmetric_value_add:
            self.value_weight = nn.Parameter(torch.empty(N_HEADS, D_HEAD, VALUE_INPUT_DIM))
            nn.init.normal_(self.value_weight, mean=0.0, std=0.02)
        else:
            self.register_parameter("value_weight", None)

        # Per-head learnable gate on the additive bias, kept for parity with A1.5.
        self.bias_gate = nn.Parameter(torch.ones(N_HEADS))

        # Degree scaler (kept identical to A1.5).
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

        # Diagnostic slots.
        self.last_entropy = None

    def forward(self, h_flat, batch: StaticAnchorBatch, collect_diag: bool = False):
        import torch

        h_in_flat = h_flat
        h_norm_flat = self.pre_attn_bn(h_flat)
        h_norm = flat_to_dense(h_norm_flat, batch)

        B, N, _ = h_norm.shape
        q = self.q_proj(h_norm).view(B, N, N_HEADS, D_HEAD).transpose(1, 2)
        k = self.k_proj(h_norm).view(B, N, N_HEADS, D_HEAD).transpose(1, 2)
        v = self.v_proj(h_norm).view(B, N, N_HEADS, D_HEAD).transpose(1, 2)

        # Content logits.
        logits = torch.einsum("bhid,bhjd->bhij", q, k) * (D_HEAD ** -0.5)

        # Structural read: xi^0 directly (no anchored-residual doubling, no projection).
        xi_sym = batch.pair_xi.masked_fill(~batch.pair_mask.unsqueeze(-1), 0.0)

        bias = torch.einsum("bijp,hp->bhij", xi_sym, self.bias_weight)
        bias = bias * self.bias_gate.view(1, N_HEADS, 1, 1)
        logits = logits + bias

        key_mask = batch.node_mask[:, None, None, :]
        logits = logits.masked_fill(~key_mask, float("-inf"))

        alpha = torch.softmax(logits, dim=-1)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
        alpha = self.attn_dropout(alpha)

        out = torch.matmul(alpha, v)
        if self.use_symmetric_value_add:
            value_pair = torch.einsum("bijp,hdp->bhijd", xi_sym, self.value_weight)
            out = out + torch.einsum("bhij,bhijd->bhid", alpha, value_pair)

        out = out.transpose(1, 2).contiguous().view(B, N, D_MODEL)
        out = self.o_proj(out)
        out = out.masked_fill(~batch.node_mask.unsqueeze(-1), 0.0)
        out_flat = dense_to_flat(out, batch)

        degree_log = torch.log1p(batch.degree).view(-1, 1)
        scaled = out_flat * self.theta1.view(1, -1) + degree_log * out_flat * self.theta2.view(1, -1)
        h_half = h_in_flat + self.resid_dropout(scaled)

        h_ffn = self.ffn(self.pre_ffn_bn(h_half))
        h_out = h_half + h_ffn

        if collect_diag:
            with torch.no_grad():
                a = alpha.clamp_min(1e-12)
                entropy = -(a * a.log()).sum(dim=-1)
                qmask = batch.node_mask[:, None, :]
                denom = qmask.sum().clamp_min(1)
                self.last_entropy = (entropy * qmask).sum(dim=(0, 2)) / denom

        return h_out


class StaticAnchorZincModel(__import__("torch").nn.Module):
    def __init__(self, use_symmetric_value_add: bool = True):
        import torch.nn as nn
        super().__init__()
        self.use_symmetric_value_add = use_symmetric_value_add
        self.atom_emb = nn.Embedding(NUM_ATOM_TYPES, D_MODEL)
        self.bond_emb = nn.Embedding(NUM_BOND_TYPES, D_MODEL)
        self.rwse_proj = nn.Linear(K_WALK, D_MODEL)

        self.layers = nn.ModuleList([
            StaticAnchorLayer(use_symmetric_value_add=use_symmetric_value_add)
            for _ in range(N_LAYERS)
        ])
        self.final_bn = nn.BatchNorm1d(D_MODEL)
        self.readout = nn.Sequential(
            nn.Linear(D_MODEL, 2 * D_MODEL),
            nn.GELU(),
            nn.Linear(2 * D_MODEL, 1),
        )

    def forward(self, batch: StaticAnchorBatch, collect_diag: bool = False):
        import torch

        h = self.atom_emb(batch.x)
        edge_msg = self.bond_emb(batch.edge_attr)
        bond_sum = h.new_zeros(h.shape)
        if batch.edge_index.numel() > 0:
            dst = batch.edge_index[1]
            bond_sum.index_add_(0, dst, edge_msg)
        h = h + bond_sum + self.rwse_proj(batch.rwse)

        for layer in self.layers:
            h = layer(h, batch, collect_diag=collect_diag)

        h = self.final_bn(h)
        pooled = h.new_zeros((batch.num_graphs, D_MODEL))
        pooled.index_add_(0, batch.batch_index, h)
        return self.readout(pooled).view(-1)

    def diagnostic_snapshot(self) -> Dict[str, object]:
        out = {}
        bias_gates = []
        bias_weight_norms = []
        value_weight_norms = []
        theta1_mean = []
        theta2_mean = []
        entropy = []
        for layer in self.layers:
            bias_gates.append(layer.bias_gate.detach().cpu().tolist())
            bias_weight_norms.append(layer.bias_weight.detach().norm(dim=-1).cpu().tolist())
            if layer.value_weight is not None:
                value_weight_norms.append(layer.value_weight.detach().norm(dim=-1).cpu().tolist())
            theta1_mean.append(float(layer.theta1.detach().mean().cpu()))
            theta2_mean.append(float(layer.theta2.detach().mean().cpu()))
            if layer.last_entropy is not None:
                entropy.append(layer.last_entropy.detach().cpu().tolist())
        out["bias_gate_per_layer_head"] = bias_gates
        out["bias_weight_norms_per_layer_head"] = bias_weight_norms
        out["symmetric_value_weight_norms_per_layer_head_dh"] = value_weight_norms
        out["degree_theta1_mean_per_layer"] = theta1_mean
        out["degree_theta2_mean_per_layer"] = theta2_mean
        out["attention_entropy_per_layer_head_last_diag_batch"] = entropy
        return out


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Train / eval (held identical to A1.5)
# =============================================================================

def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int):
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_static_anchor,
    )


def epoch_lr(epoch: int, max_epochs: int) -> float:
    if epoch <= WARMUP_EPOCHS:
        return LR * (epoch / max(1, WARMUP_EPOCHS))
    progress = (epoch - WARMUP_EPOCHS) / max(1, max_epochs - WARMUP_EPOCHS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr_value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr_value


def train_one_epoch(model, loader, optimizer, device) -> Tuple[float, float]:
    import torch
    import torch.nn.functional as F

    model.train()
    total_abs = 0.0
    total_loss = 0.0
    total_graphs = 0
    last_grad_norm = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = F.l1_loss(pred, batch.y)
        loss.backward()
        last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).detach().cpu())
        optimizer.step()

        n = batch.num_graphs
        total_loss += float(loss.detach().cpu()) * n
        total_abs += float((pred.detach() - batch.y).abs().sum().cpu())
        total_graphs += n

    return total_loss / total_graphs, last_grad_norm


@torch.no_grad()
def eval_mae(model, loader, device, collect_diag: bool = False) -> float:
    import torch

    model.eval()
    total_abs = 0.0
    total_graphs = 0
    first = True
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch, collect_diag=(collect_diag and first))
        first = False
        total_abs += float((pred - batch.y).abs().sum().cpu())
        total_graphs += batch.num_graphs
    return total_abs / total_graphs


def save_checkpoint(path: str, epoch: int, model, optimizer, best: Dict[str, float], args: argparse.Namespace) -> None:
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best": best,
            "args": vars(args),
            "config": variant_config_dict(args),
        },
        path,
    )


def variant_config_dict(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "method": "Static-anchor symmetric-only graph transformer (controlled ablation of Variant A1.5)",
        "dataset": "PyG-ZINC subset",
        "structural_state": "fixed xi^0 (no evolution, no antisymmetric channel)",
        "num_atom_types": NUM_ATOM_TYPES,
        "num_bond_types": NUM_BOND_TYPES,
        "layers": N_LAYERS,
        "hidden_dim": D_MODEL,
        "heads": N_HEADS,
        "d_head": D_HEAD,
        "K_walk": K_WALK,
        "D_max_spd": D_MAX_SPD,
        "p_sym": P_SYM,
        "p_pair": P_PAIR,
        "bias_input_dim": BIAS_INPUT_DIM,
        "value_input_dim": VALUE_INPUT_DIM,
        "antisymmetric_channel": False,
        "rotation_pathway": False,
        "pair_evolution": False,
        "feature_set": {
            "spd_one_hot_dim": D_MAX_SPD + 1,
            "symmetric_walk_sum_dim": K_WALK,
            "bond_one_hot_dim": NUM_BOND_TYPES,
        },
        "node_features": "atom_emb + sum_edge_emb + Linear(RWSE_diag)",
        "ffn_hidden_dim": FFN_MULT * D_MODEL,
        "symmetric_value_add": not args.disable_symmetric_value_add,
        "symmetric_value_add_params_per_layer": (
            0 if args.disable_symmetric_value_add else N_HEADS * D_HEAD * VALUE_INPUT_DIM
        ),
        "attn_dropout": ATTN_DROPOUT,
        "dropout": RESID_DROPOUT,
        "pooling": "sum",
        "optimizer": "AdamW",
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "loss": "L1/MAE",
        "grad_clip": GRAD_CLIP,
    }


def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    latest = os.path.join(out_dir, "checkpoints", "checkpoint_latest.pt")
    return latest if os.path.isfile(latest) else None


def run_training(args: argparse.Namespace, splits, meta) -> None:
    import torch

    out_dir = args.out_dir or os.path.join(args.drive_dir if not args.no_drive else "/content", "results", args.name_tag)
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, "logs")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    log(f"[env] device: {device}")
    if device.type == "cuda":
        log(f"[env] gpu: {torch.cuda.get_device_name(0)}")
    log(f"[env] torch: {torch.__version__}")

    train_ds = TensorGraphDataset(splits["train"])
    val_ds = TensorGraphDataset(splits["val"])
    test_ds = TensorGraphDataset(splits["test"])

    train_loader = make_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = StaticAnchorZincModel(
        use_symmetric_value_add=not args.disable_symmetric_value_add
    ).to(device)
    n_params = count_parameters(model)

    cfg = variant_config_dict(args)
    cfg.update({"param_count": n_params, "out_dir": out_dir, "data_meta": meta})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    log("\n[config] ---- Static-Anchor ZINC ----")
    for key in [
        "dataset", "structural_state", "layers", "hidden_dim", "heads", "d_head",
        "K_walk", "D_max_spd", "p_sym", "p_pair",
        "bias_input_dim", "value_input_dim",
        "antisymmetric_channel", "rotation_pathway", "pair_evolution",
        "node_features", "ffn_hidden_dim", "symmetric_value_add",
        "symmetric_value_add_params_per_layer", "attn_dropout", "pooling",
        "lr", "weight_decay", "warmup_epochs", "max_epochs", "batch_size",
    ]:
        log(f"[config] {key:>30s}: {cfg[key]}")
    log(f"[config] {'param_count':>30s}: {n_params:,}")
    log(f"[config] {'out_dir':>30s}: {out_dir}")
    if n_params > EXPECTED_PARAM_CEILING and not args.allow_over_500k:
        raise RuntimeError(
            f"Parameter count {n_params:,} exceeds ceiling {EXPECTED_PARAM_CEILING:,}. "
            "Pass --allow-over-500k to run anyway."
        )
    log("[config] ------------------------\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY,
    )

    best = {
        "val_mae": float("inf"),
        "test_mae_at_best": float("nan"),
        "train_mae_at_best": float("nan"),
        "epoch": 0,
    }
    start_epoch = 1

    if args.auto_resume:
        ckpt_path = find_latest_checkpoint(out_dir)
        if ckpt_path:
            log(f"[resume] Loading checkpoint: {ckpt_path}")
            ckpt = safe_torch_load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            best.update(ckpt.get("best", {}))
            start_epoch = int(ckpt["epoch"]) + 1
            log(f"[resume] Resuming at epoch {start_epoch}; best val={best['val_mae']:.6f}")

    max_epochs = 5 if args.smoke_test else args.max_epochs
    metrics_path = os.path.join(log_dir, "metrics.jsonl")
    diag_path = os.path.join(log_dir, "diagnostics.jsonl")
    if start_epoch == 1 and os.path.isfile(metrics_path):
        os.remove(metrics_path)

    log("[train] Starting training")
    log("[train] Compact output shows current train/val/test MAE each epoch and best-so-far validation.")
    t_run = time.time()
    epoch_times = []

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        lr_now = epoch_lr(epoch, max_epochs)
        set_lr(optimizer, lr_now)

        train_loss, grad_norm = train_one_epoch(model, train_loader, optimizer, device)
        collect_diag = args.diagnostic_period > 0 and (epoch % args.diagnostic_period == 0 or epoch == 1)
        val_mae = eval_mae(model, val_loader, device, collect_diag=collect_diag)
        test_mae = eval_mae(model, test_loader, device, collect_diag=False)

        is_best = val_mae < best["val_mae"]
        if is_best:
            best.update(
                {
                    "val_mae": val_mae,
                    "test_mae_at_best": test_mae,
                    "train_mae_at_best": train_loss,
                    "epoch": epoch,
                }
            )
            save_checkpoint(os.path.join(ckpt_dir, "checkpoint_best.pt"), epoch, model, optimizer, best, args)

        save_checkpoint(os.path.join(ckpt_dir, "checkpoint_latest.pt"), epoch, model, optimizer, best, args)

        if epoch % args.ckpt_period == 0 or epoch == max_epochs:
            periodic = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch:05d}.pt")
            save_checkpoint(periodic, epoch, model, optimizer, best, args)

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        avg_time = sum(epoch_times) / len(epoch_times)

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_mae": train_loss,
            "val_mae": val_mae,
            "test_mae": test_mae,
            "best_val_mae": best["val_mae"],
            "best_test_mae": best["test_mae_at_best"],
            "best_epoch": best["epoch"],
            "grad_norm": grad_norm,
            "epoch_time_s": epoch_time,
            "avg_epoch_time_s": avg_time,
            "is_best": is_best,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if collect_diag:
            diag = model.diagnostic_snapshot()
            diag["epoch"] = epoch
            with open(diag_path, "a") as f:
                f.write(json.dumps(diag) + "\n")

        if epoch % args.console_epoch_period == 0 or epoch == 1 or is_best or epoch == max_epochs:
            flag = " <-- BEST" if is_best else ""
            log(
                f"[epoch {epoch:04d}/{max_epochs}] "
                f"train_loss={train_loss:.6f} train_mae={train_loss:.6f} | "
                f"val_mae={val_mae:.6f} | test_mae={test_mae:.6f} | "
                f"best@{best['epoch']:04d} val={best['val_mae']:.6f} "
                f"test@best={best['test_mae_at_best']:.6f} | "
                f"lr={lr_now:.3e} grad={grad_norm:.3f} | "
                f"time={epoch_time:.1f}s avg={avg_time:.1f}s{flag}"
            )

    total = time.time() - t_run
    log("\n[done] Training complete.")
    log(f"[done] Best val MAE: {best['val_mae']:.6f} at epoch {best['epoch']}")
    log(f"[done] Test MAE @ best val: {best['test_mae_at_best']:.6f}")
    log(f"[done] Total wall time: {total/60:.1f} min")
    log(f"[done] Checkpoints: {ckpt_dir}")
    log(f"[done] Metrics: {metrics_path}")


# =============================================================================
# Main
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    mount_drive_if_needed(args)
    install_deps(args)

    import torch  # noqa: F401 - imported after optional install

    splits, meta = load_or_precompute_dataset(args)
    run_training(args, splits, meta)


if __name__ == "__main__":
    main()