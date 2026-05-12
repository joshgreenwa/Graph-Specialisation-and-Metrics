# Extracted from graphormer_ZINC_core.ipynb
# Keep the notebook and script in sync when changing experiment logic.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone GraphormerSLIM runner for ZINC(subset), designed for Colab.

This file intentionally does NOT import Microsoft Graphormer, fairseq, Hydra, or any
other official-repo training stack. It implements the Graphormer ingredients directly
in PyTorch/PyG:

  - Graph token
  - Atom/node categorical embedding
  - Degree/centrality encoding
  - Shortest-path spatial attention bias
  - Multi-hop shortest-path edge attention bias
  - Transformer encoder with per-head graph attention bias
  - GraphormerSLIM model size for ZINC: L=12, d=80, ffn=80, heads=8
  - Official-style output head: transform + GELU + LayerNorm + linear(no bias) + learned scalar bias

Training controls are aligned with the user's successful GraphGPS ZINC runner:

  - ZINC subset=True
  - batch_size=32
  - epochs=2000
  - lr=1e-3
  - warmup_epochs=50
  - AdamW, weight_decay=1e-5
  - cosine decay after warmup
  - gradient clipping
  - eval every epoch
  - checkpoint every 100 epochs to Google Drive
  - default graph readout: official graph-token readout; optional sum pooling for controlled ablations

Suggested Colab usage:

    !python train_graphormer_zinc_standalone_colab.py --no-train
    !python train_graphormer_zinc_standalone_colab.py

Notes on fidelity:
- The model-size/dropout settings follow GraphormerSLIM for ZINC from the paper.
- The training controls deliberately deviate from the paper to match GraphGPS controls.
- This is a faithful standalone implementation of the Graphormer design, not a bitwise
  reproduction of the fairseq training stack. It mirrors the official model components
  relevant for ZINC graph regression, including the graph-token output path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def log(msg: str) -> None:
    print(msg, flush=True)


class CommandError(RuntimeError):
    pass


def run_cmd(cmd: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    printable = " ".join(map(str, cmd))
    log(f"\n[cmd] {printable}")
    proc = subprocess.run(list(map(str, cmd)), text=True, check=False)
    if check and proc.returncode != 0:
        raise CommandError(f"Command failed with exit code {proc.returncode}: {printable}")
    return proc


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def mount_drive(mount_point: Path) -> None:
    if in_colab():
        from google.colab import drive  # type: ignore
        log(f"[drive] Mounting Google Drive at {mount_point} ...")
        drive.mount(str(mount_point), force_remount=False)
    else:
        log("[drive] google.colab unavailable; using local paths.")
        mount_point.mkdir(parents=True, exist_ok=True)


def strip_colab_kernel_args(argv: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-f" and i + 1 < len(argv) and "kernel-" in argv[i + 1] and argv[i + 1].endswith(".json"):
            log(f"[args] Ignoring Colab/Jupyter kernel argument: {a} {argv[i + 1]}")
            i += 2
            continue
        if a.startswith("-f=") and "kernel-" in a and a.endswith(".json"):
            log(f"[args] Ignoring Colab/Jupyter kernel argument: {a}")
            i += 1
            continue
        cleaned.append(a)
        i += 1
    return cleaned


def install_dependencies(skip_install: bool = False, pyg_version: str = "2.2.0") -> None:
    if skip_install:
        log("[deps] Skipping dependency installation (--skip-install).")
        return

    log("[deps] Installing current-Colab compatible PyG stack. No fairseq/Hydra/Graphormer repo is used.")
    run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    import importlib
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is not importable in this runtime. Select a Colab GPU runtime first.") from exc

    torch_version = str(torch.__version__).split("+")[0]
    cuda_version = getattr(torch.version, "cuda", None)
    cuda_tag = "cu" + cuda_version.replace(".", "") if cuda_version else "cpu"
    pyg_wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    log(f"[deps] Python: {sys.version.split()[0]} | torch: {torch.__version__} | CUDA: {cuda_version}")
    log(f"[deps] PyG wheel index: {pyg_wheel_url}")

    run_cmd([
        sys.executable, "-m", "pip", "install",
        "pyg-lib", "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv",
        "-f", pyg_wheel_url,
    ])
    run_cmd([
        sys.executable, "-m", "pip", "install",
        f"torch-geometric=={pyg_version}",
        "numpy", "scipy", "pandas", "tqdm", "tensorboardX>=2.6,<2.7",
    ])


@dataclass
class GraphormerConfig:
    # GraphormerSLIM/ZINC model-size config from the Graphormer paper.
    num_layers: int = 12
    hidden_dim: int = 80
    ffn_dim: int = 80
    num_heads: int = 8
    # Official zinc.sh uses --dropout 0.0 and --act-dropout 0.1; the paper table calls this FFN dropout.
    dropout: float = 0.0              # residual dropout in fairseq terminology.
    attention_dropout: float = 0.1
    embedding_dropout: float = 0.0
    activation_dropout: float = 0.1

    # ZINC/PyG categorical ranges chosen to reproduce the paper's ~489K GraphormerSLIM
    # parameter budget. These are *post-collator shifted-index maxima*, not raw PyG
    # vocabulary sizes. Padding stays at 0; raw atom id a is encoded as a+2, raw edge
    # id e on a shortest path is encoded as e+3, and raw degree d is encoded as d+1.
    num_atom_types: int = 29           # PyG ZINC raw atom ids 0..27 -> shifted ids up to 29.
    num_edge_types: int = 6            # PyG ZINC raw bond ids 0..3 -> shifted ids up to 6.
    num_in_degree: int = 6             # molecular graph degree 0..4 -> shifted ids up to 5.
    num_out_degree: int = 6
    num_spatial: int = 64              # comfortably above shifted ZINC shortest-path diameters.
    num_edge_dis: int = 128            # official default used by multi-hop edge distance encoder.
    multi_hop_max_dist: int = 5
    spatial_pos_max: int = 1024
    encoder_normalize_before: bool = True

    # Exact Graphormer graph regression uses the graph token. Sum/mean are explicit controlled deviations.
    graph_pooling: str = "graph_token" # choices: graph_token, sum, mean.

    # Task.
    num_classes: int = 1

    @property
    def head_dim(self) -> int:
        assert self.hidden_dim % self.num_heads == 0
        return self.hidden_dim // self.num_heads


@dataclass
class TrainConfig:
    # GraphGPS-control training knobs requested by the user.
    epochs: int = 2000
    batch_size: int = 32
    lr: float = 1e-3
    warmup_epochs: int = 50
    weight_decay: float = 1e-5
    clip_grad_norm: float = 5.0
    optimizer: str = "adamw"
    scheduler: str = "cosine_with_warmup"
    eval_period: int = 1
    ckpt_period: int = 100
    seed: int = 0
    num_workers: int = 0
    pin_memory: bool = True
    amp: bool = False


class GraphRecordDataset:
    """Tiny wrapper around a list of precomputed graph dictionaries."""

    def __init__(self, records: List[Dict[str, object]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.records[idx]


def patch_torch_load_for_pyg_processed_data() -> None:
    """Restore legacy torch.load default for trusted local PyG processed files.

    PyTorch >= 2.6 defaults torch.load(..., weights_only=True). PyG's
    InMemoryDataset/ZINC processed files contain torch_geometric.data.Data
    objects, so the new safe default rejects them. These files are generated
    locally from the ZINC benchmark during this run, so loading them with the
    old behavior is appropriate here.
    """
    import torch

    if getattr(torch.load, "_graphormer_pyg_compat", False):
        return

    _orig_torch_load = torch.load

    def _compat_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    _compat_torch_load._graphormer_pyg_compat = True  # type: ignore[attr-defined]
    torch.load = _compat_torch_load



def safe_torch_load(path: Path):
    import torch
    patch_torch_load_for_pyg_processed_data()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def edge_lookup_from_pyg(edge_index, edge_attr) -> Dict[Tuple[int, int], int]:
    # PyG ZINC edge_attr is usually shape [E, 1] with categorical bond type.
    lookup: Dict[Tuple[int, int], int] = {}
    ei = edge_index.cpu().tolist()
    ea = edge_attr.view(edge_attr.size(0), -1)[:, 0].cpu().tolist()
    for u, v, e in zip(ei[0], ei[1], ea):
        lookup[(int(u), int(v))] = int(e)
    return lookup


def convert_to_single_emb_1d(x, offset: int = 512):
    """Official Graphormer feature-indexing convention for one categorical feature."""
    import torch
    return x.long() + 1  # same as convert_to_single_emb for feature_num=1.


def preprocess_one_graph(data, multi_hop_max_dist: int = 5) -> Dict[str, object]:
    """Precompute Graphormer structural tensors for one PyG ZINC graph.

    This mirrors the official preprocessing/collator conventions:
      - raw categorical node features are converted by +1 here and +1 again in collate;
      - raw edge features are converted by +2 here and +1 again in collate for edge_input;
      - spatial_pos is raw shortest-path distance here and shifted by +1 in collate;
      - in/out degree are raw here and shifted by +1 in collate;
      - attn_bias starts as zeros and receives -inf masking for too-distant pairs in collate.
    """
    import torch
    from collections import deque

    n = int(data.num_nodes)
    x_raw = data.x.view(n, -1)[:, 0].long().cpu()
    x = convert_to_single_emb_1d(x_raw)  # official preprocess_item: convert_to_single_emb(x).
    y = data.y.view(-1).float().cpu()
    edge_index = data.edge_index.long().cpu()
    edge_attr = data.edge_attr.long().cpu() if getattr(data, "edge_attr", None) is not None else torch.zeros(edge_index.size(1), 1, dtype=torch.long)

    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    edge_lookup = edge_lookup_from_pyg(edge_index, edge_attr)
    for (u, v), e in edge_lookup.items():
        # official attn_edge_type uses convert_to_single_emb(edge_attr)+1; for one feature this is raw+2.
        adj[u].append((v, int(e) + 2))

    degree = torch.tensor([len(adj[i]) for i in range(n)], dtype=torch.long)
    spatial_pos = torch.zeros(n, n, dtype=torch.long)
    edge_input = torch.zeros(n, n, multi_hop_max_dist, 1, dtype=torch.long)
    attn_bias = torch.zeros(n + 1, n + 1, dtype=torch.float)

    for src in range(n):
        dist = [-1] * n
        parent = [-1] * n
        parent_edge = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v, e in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    parent_edge[v] = e
                    q.append(v)

        for dst in range(n):
            d = dist[dst]
            if d < 0:
                continue
            spatial_pos[src, dst] = d
            if 0 < d <= multi_hop_max_dist:
                rev_edges: List[int] = []
                cur = dst
                while cur != src and cur != -1:
                    rev_edges.append(int(parent_edge[cur]))
                    cur = parent[cur]
                path_edges = list(reversed(rev_edges))
                for k, e in enumerate(path_edges[:multi_hop_max_dist]):
                    edge_input[src, dst, k, 0] = e

    return {
        "x": x.view(n, 1),
        "y": y,
        "degree": degree,
        "spatial_pos": spatial_pos,
        "edge_input": edge_input,
        "attn_bias": attn_bias,
        "num_nodes": n,
    }

def precompute_split(split: str, root: Path, cache_dir: Path, cfg: GraphormerConfig, force: bool = False) -> GraphRecordDataset:
    import torch
    patch_torch_load_for_pyg_processed_data()
    from torch_geometric.datasets import ZINC
    from tqdm import tqdm

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"zinc_subset_graphormer_{split}_mh{cfg.multi_hop_max_dist}.pt"
    if cache_file.exists() and not force:
        log(f"[data] Loading cached {split} records: {cache_file}")
        return GraphRecordDataset(safe_torch_load(cache_file))

    log(f"[data] Building PyG ZINC(subset=True, split={split}) and precomputing Graphormer features.")
    ds = ZINC(root=str(root), subset=True, split=split)
    records: List[Dict[str, object]] = []
    max_x = max_e = max_deg = max_sp = 0
    for data in tqdm(ds, desc=f"precompute {split}"):
        rec = preprocess_one_graph(data, multi_hop_max_dist=cfg.multi_hop_max_dist)
        records.append(rec)
        max_x = max(max_x, int(rec["x"].max().item()) if int(rec["num_nodes"]) else 0)
        max_deg = max(max_deg, int(rec["degree"].max().item()) if int(rec["num_nodes"]) else 0)
        max_sp = max(max_sp, int(rec["spatial_pos"].max().item()) if int(rec["num_nodes"]) else 0)
        ep = rec["edge_input"]
        max_e = max(max_e, int(ep.max().item()) if ep.numel() else 0)

    log(f"[data:{split}] graphs={len(records)} max_preprocessed_atom_id={max_x} max_preprocessed_edge_id={max_e} max_degree={max_deg} max_spatial={max_sp}")
    torch.save(records, cache_file)
    log(f"[data] Saved cache: {cache_file}")
    return GraphRecordDataset(records)


def collate_graphormer(records: Sequence[Dict[str, object]], cfg: GraphormerConfig) -> Dict[str, object]:
    import torch

    bsz = len(records)
    max_n = max(int(r["num_nodes"]) for r in records)
    max_dist = cfg.multi_hop_max_dist

    x = torch.zeros(bsz, max_n, 1, dtype=torch.long)
    in_degree = torch.zeros(bsz, max_n, dtype=torch.long)
    out_degree = torch.zeros(bsz, max_n, dtype=torch.long)
    spatial_pos = torch.zeros(bsz, max_n, max_n, dtype=torch.long)
    edge_input = torch.zeros(bsz, max_n, max_n, max_dist, 1, dtype=torch.long)
    attn_bias = torch.full((bsz, max_n + 1, max_n + 1), float("-inf"), dtype=torch.float)
    node_mask = torch.zeros(bsz, max_n, dtype=torch.bool)
    y = torch.zeros(bsz, cfg.num_classes, dtype=torch.float)

    for i, r in enumerate(records):
        n = int(r["num_nodes"])
        # Official collator shifts by +1 again so padding remains 0. Do not clamp:
        # if these assertions fail, the configured embedding vocabulary is wrong.
        xi = torch.as_tensor(r["x"], dtype=torch.long) + 1
        if xi.numel() and int(xi.max()) > cfg.num_atom_types:
            raise ValueError(f"node feature id {int(xi.max())} exceeds num_atom_types={cfg.num_atom_types}; do not silently clamp")
        x[i, :n, :] = xi

        deg = torch.as_tensor(r["degree"], dtype=torch.long)
        deg_shifted = deg + 1
        if deg_shifted.numel() and int(deg_shifted.max()) >= cfg.num_in_degree:
            raise ValueError(f"degree id {int(deg_shifted.max())} exceeds degree embedding size {cfg.num_in_degree}; increase num_in_degree/out_degree")
        in_degree[i, :n] = deg_shifted
        out_degree[i, :n] = deg_shifted

        sp_raw = torch.as_tensor(r["spatial_pos"], dtype=torch.long)
        sp = sp_raw + 1
        if sp.numel() and int(sp.max()) >= cfg.num_spatial:
            raise ValueError(f"spatial id {int(sp.max())} exceeds num_spatial={cfg.num_spatial}; increase num_spatial")
        spatial_pos[i, :n, :n] = sp

        ei = torch.as_tensor(r["edge_input"], dtype=torch.long)
        ei_shifted = ei + 1
        if ei_shifted.numel() and int(ei_shifted.max()) > cfg.num_edge_types:
            raise ValueError(f"edge feature id {int(ei_shifted.max())} exceeds num_edge_types={cfg.num_edge_types}; do not silently clamp")
        edge_input[i, :n, :n, :, :] = ei_shifted
        ab = torch.as_tensor(r["attn_bias"], dtype=torch.float).clone()
        ab[1:, 1:][sp_raw >= cfg.spatial_pos_max] = float("-inf")
        attn_bias[i, : n + 1, : n + 1] = ab
        # Official pad_attn_bias_unsqueeze sets padded query rows to zero for real keys.
        if n + 1 < max_n + 1:
            attn_bias[i, n + 1 :, : n + 1] = 0.0
        node_mask[i, :n] = True
        yy = torch.as_tensor(r["y"], dtype=torch.float).view(-1)
        y[i, : min(cfg.num_classes, yy.numel())] = yy[: cfg.num_classes]

    return {
        "x": x,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "spatial_pos": spatial_pos,
        "edge_input": edge_input,
        "attn_bias": attn_bias,
        "node_mask": node_mask,
        "y": y,
    }


def _define_torch_modules():
    """Define modules lazily after dependency installation/import."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MultiHeadSelfAttentionWithBias(nn.Module):
        def __init__(self, dim: int, num_heads: int, attn_dropout: float, out_dropout: float) -> None:
            super().__init__()
            assert dim % num_heads == 0
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scaling = self.head_dim ** -0.5
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.attn_dropout = nn.Dropout(attn_dropout)
            self.out_dropout = nn.Dropout(out_dropout)

        def forward(self, x, attn_bias):
            bsz, seq_len, dim = x.shape
            q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
            scores = scores + attn_bias
            attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, dim)
            out = self.out_proj(out)
            return self.out_dropout(out)

    class GraphormerEncoderLayer(nn.Module):
        def __init__(self, cfg: GraphormerConfig) -> None:
            super().__init__()
            self.self_attn = MultiHeadSelfAttentionWithBias(cfg.hidden_dim, cfg.num_heads, cfg.attention_dropout, cfg.dropout)
            self.self_attn_layer_norm = nn.LayerNorm(cfg.hidden_dim)
            self.fc1 = nn.Linear(cfg.hidden_dim, cfg.ffn_dim)
            self.fc2 = nn.Linear(cfg.ffn_dim, cfg.hidden_dim)
            self.final_layer_norm = nn.LayerNorm(cfg.hidden_dim)
            self.dropout = nn.Dropout(cfg.dropout)
            self.activation_dropout = nn.Dropout(cfg.activation_dropout)
            self.pre_layernorm = False

        def forward(self, x, attn_bias):
            residual = x
            x = self.self_attn(x, attn_bias)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.activation_dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = residual + x
            x = self.final_layer_norm(x)
            return x

    class StandaloneGraphormer(nn.Module):
        def __init__(self, cfg: GraphormerConfig) -> None:
            super().__init__()
            self.cfg = cfg
            h = cfg.num_heads
            d = cfg.hidden_dim
            self.atom_encoder = nn.Embedding(cfg.num_atom_types + 1, d, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(cfg.num_in_degree, d, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(cfg.num_out_degree, d, padding_idx=0)
            self.graph_token = nn.Embedding(1, d)
            self.spatial_pos_encoder = nn.Embedding(cfg.num_spatial, h, padding_idx=0)
            self.edge_encoder = nn.Embedding(cfg.num_edge_types + 1, h, padding_idx=0)
            self.edge_dis_encoder = nn.Embedding(cfg.num_edge_dis * h * h, 1)
            self.graph_token_virtual_distance = nn.Embedding(1, h)
            self.emb_layer_norm = nn.LayerNorm(d) if cfg.encoder_normalize_before else None
            self.emb_dropout = nn.Dropout(cfg.embedding_dropout)
            self.layers = nn.ModuleList([GraphormerEncoderLayer(cfg) for _ in range(cfg.num_layers)])
            # Official graph-level head: transform + GELU + LayerNorm + output projection(no bias) + scalar bias.
            self.lm_head_transform_weight = nn.Linear(d, d)
            self.layer_norm = nn.LayerNorm(d)
            self.embed_out = nn.Linear(d, cfg.num_classes, bias=False)
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
            self.apply(self._graphormer_init)

        @staticmethod
        def _graphormer_init(module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)

        def build_attn_bias(self, batch):
            spatial_pos = batch["spatial_pos"]
            edge_input = batch["edge_input"]
            attn_bias = batch["attn_bias"]
            x = batch["x"]
            bsz, n = x.size()[:2]
            h = self.cfg.num_heads

            graph_attn_bias = attn_bias.clone().unsqueeze(1).repeat(1, h, 1, 1)
            spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

            t = self.graph_token_virtual_distance.weight.view(1, h, 1)
            graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
            graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.cfg.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.cfg.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.cfg.multi_hop_max_dist, :]

            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, h)
            edge_dis_weight = self.edge_dis_encoder.weight.reshape(-1, h, h)[:max_dist, :, :]
            edge_input_flat = torch.bmm(edge_input_flat, edge_dis_weight)
            edge_input = edge_input_flat.reshape(max_dist, bsz, n, n, h).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) / spatial_pos_.float().unsqueeze(-1)).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input

            # Official code adds the original attn_bias again after structural terms.
            graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)
            return graph_attn_bias

        def graph_node_feature(self, batch):
            x = batch["x"]
            in_degree = batch["in_degree"]
            out_degree = batch["out_degree"]
            bsz = x.size(0)
            node_feature = self.atom_encoder(x).sum(dim=-2)
            node_feature = node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
            graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(bsz, 1, 1)
            return torch.cat([graph_token_feature, node_feature], dim=1)

        def output_projection(self, h):
            z = self.layer_norm(F.gelu(self.lm_head_transform_weight(h)))
            z = self.embed_out(z)
            z = z + self.lm_output_learned_bias
            return z

        def forward(self, batch):
            h = self.graph_node_feature(batch)
            if self.emb_layer_norm is not None:
                h = self.emb_layer_norm(h)
            h = self.emb_dropout(h)
            attn_bias = self.build_attn_bias(batch)
            for layer in self.layers:
                h = layer(h, attn_bias)

            if self.cfg.graph_pooling == "graph_token":
                logits = self.output_projection(h)
                return logits[:, 0, :]

            # Controlled non-official readouts use the same official head after pooling.
            node_mask = batch["node_mask"]
            node_h = h[:, 1:, :] * node_mask.unsqueeze(-1).to(h.dtype)
            if self.cfg.graph_pooling == "sum":
                graph_repr = node_h.sum(dim=1)
            elif self.cfg.graph_pooling == "mean":
                denom = node_mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
                graph_repr = node_h.sum(dim=1) / denom
            else:
                raise ValueError(f"Unknown graph_pooling={self.cfg.graph_pooling!r}")
            return self.output_projection(graph_repr)

    return StandaloneGraphormer


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: Mapping[str, object], device) -> Dict[str, object]:
    import torch
    out: Dict[str, object] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def mae_loss(pred, target):
    import torch.nn.functional as F
    return F.l1_loss(pred.view_as(target), target)


def evaluate(model, loader, device, amp: bool = False) -> Dict[str, float]:
    import torch
    model.eval()
    total_abs = 0.0
    total_count = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
                pred = model(batch)
                target = batch["y"]
                loss = mae_loss(pred, target)
            total_loss += float(loss.item()) * int(target.numel())
            total_abs += float((pred.view_as(target) - target).abs().sum().item())
            total_count += int(target.numel())
    mae = total_abs / max(1, total_count)
    return {"loss": total_loss / max(1, total_count), "mae": mae}


def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    import torch
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path: Path, *, epoch: int, model, optimizer, scheduler, best: Dict[str, float], cfg: GraphormerConfig, train_cfg: TrainConfig) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best": best,
        "graphormer_config": asdict(cfg),
        "train_config": asdict(train_cfg),
    }, tmp)
    tmp.replace(path)


def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    paths = sorted(ckpt_dir.glob("epoch_*.pt"))
    return paths[-1] if paths else None


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, device="cpu") -> Tuple[int, Dict[str, float]]:
    ckpt = safe_torch_load(path)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt.get("epoch", 0)), dict(ckpt.get("best", {}))


def train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    cfg = GraphormerConfig(
        graph_pooling=args.graph_pooling,
        multi_hop_max_dist=args.multi_hop_max_dist,
        num_atom_types=args.num_atom_types,
        num_edge_types=args.num_edge_types,
        num_spatial=args.num_spatial,
        num_edge_dis=args.num_edge_dis,
        spatial_pos_max=args.spatial_pos_max,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
        ckpt_period=args.ckpt_period,
        seed=args.seed,
        num_workers=args.num_workers,
        amp=args.amp,
    )

    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    args.drive_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.drive_dir / "results" / f"seed{args.seed}_{args.graph_pooling}pool"
    ckpt_dir = results_dir / "checkpoints"
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.drive_dir / "datasets" / "zinc_pyg"
    cache_dir = args.drive_dir / "datasets" / "zinc_graphormer_cache"

    log("[config] Standalone GraphormerSLIM ZINC(sub-set), no official repo/fairseq/Hydra.")
    log(f"[config] Model: layers={cfg.num_layers}, d={cfg.hidden_dim}, ffn={cfg.ffn_dim}, heads={cfg.num_heads}, head_dim={cfg.head_dim}, dropout={cfg.dropout}, attn_dropout={cfg.attention_dropout}, emb_dropout={cfg.embedding_dropout}")
    log(f"[config] Encodings: official-style centrality, shifted shortest-path spatial bias, multi-hop edge-distance bias with max_dist={cfg.multi_hop_max_dist}, edge_dis={cfg.num_edge_dis}")
    log(f"[config] ZINC vocab budget: num_atom_types={cfg.num_atom_types}, num_edge_types={cfg.num_edge_types}, degree_vocab={cfg.num_in_degree}, num_spatial={cfg.num_spatial}; expected_params=489,489")
    log(f"[config] Training controls: epochs={train_cfg.epochs}, batch={train_cfg.batch_size}, lr={train_cfg.lr}, warmup_epochs={train_cfg.warmup_epochs}, wd={train_cfg.weight_decay}, clip={train_cfg.clip_grad_norm}, pooling={cfg.graph_pooling}")
    log(f"[env] Python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} device={device}")

    (logs_dir / "config.json").write_text(json.dumps({
        "graphormer_config": asdict(cfg),
        "train_config": asdict(train_cfg),
        "device": str(device),
    }, indent=2), encoding="utf-8")

    train_ds = precompute_split("train", dataset_root, cache_dir, cfg, force=args.force_reprocess)
    val_ds = precompute_split("val", dataset_root, cache_dir, cfg, force=args.force_reprocess)
    test_ds = precompute_split("test", dataset_root, cache_dir, cfg, force=args.force_reprocess)

    collate = lambda recs: collate_graphormer(recs, cfg)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate,
                              num_workers=train_cfg.num_workers, pin_memory=train_cfg.pin_memory and device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate,
                            num_workers=train_cfg.num_workers, pin_memory=train_cfg.pin_memory and device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate,
                             num_workers=train_cfg.num_workers, pin_memory=train_cfg.pin_memory and device.type == "cuda")

    ModelCls = _define_torch_modules()
    model = ModelCls(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_params = 489_489
    log(f"[model] Trainable parameters: {n_params:,}")
    if n_params != expected_params:
        msg = (
            f"[model-error] Expected {expected_params:,} trainable parameters for this standalone "
            f"GraphormerSLIM/ZINC parameter-budget match, got {n_params:,}."
        )
        if not args.allow_param_mismatch:
            raise RuntimeError(msg + " Pass --allow-param-mismatch only for deliberate ablations.")
        log(msg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=train_cfg.weight_decay)
    total_steps = train_cfg.epochs * len(train_loader)
    warmup_steps = train_cfg.warmup_epochs * len(train_loader)
    scheduler = make_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.amp and device.type == "cuda")

    start_epoch = 1
    best = {"val_mae": float("inf"), "val_loss": float("inf"), "test_mae": float("inf"), "epoch": -1}
    if args.resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            prev_epoch, prev_best = load_checkpoint(ckpt, model, optimizer, scheduler, device=device)
            start_epoch = prev_epoch + 1
            best.update(prev_best)
            log(f"[resume] Loaded {ckpt}; resuming at epoch {start_epoch}")

    if args.no_train:
        log("[no-train] Preflight complete: data, model, optimizer, scheduler instantiated successfully.")
        return

    metrics_path = logs_dir / "metrics.csv"
    write_header = not metrics_path.exists() or start_epoch == 1
    with metrics_path.open("a", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=[
            "epoch", "lr", "train_loss", "train_mae", "val_loss", "val_mae", "test_loss", "test_mae",
            "best_epoch", "best_val_mae", "best_test_mae", "seconds",
        ])
        if write_header:
            writer.writeheader()

        for epoch in range(start_epoch, train_cfg.epochs + 1):
            t0 = time.perf_counter()
            model.train()
            total_loss = 0.0
            total_abs = 0.0
            total_count = 0
            for batch in train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=train_cfg.amp and device.type == "cuda"):
                    pred = model(batch)
                    target = batch["y"]
                    loss = mae_loss(pred, target)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if train_cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += float(loss.item()) * int(target.numel())
                total_abs += float((pred.detach().view_as(target) - target).abs().sum().item())
                total_count += int(target.numel())

            train_metrics = {"loss": total_loss / max(1, total_count), "mae": total_abs / max(1, total_count)}
            if epoch % train_cfg.eval_period == 0:
                val_metrics = evaluate(model, val_loader, device, amp=train_cfg.amp)
                test_metrics = evaluate(model, test_loader, device, amp=train_cfg.amp)
            else:
                val_metrics = {"loss": float("nan"), "mae": float("nan")}
                test_metrics = {"loss": float("nan"), "mae": float("nan")}

            if val_metrics["mae"] < best["val_mae"]:
                best.update({
                    "epoch": epoch,
                    "val_mae": val_metrics["mae"],
                    "val_loss": val_metrics["loss"],
                    "test_mae": test_metrics["mae"],
                    "test_loss": test_metrics["loss"],
                })
                save_checkpoint(ckpt_dir / "best.pt", epoch=epoch, model=model, optimizer=optimizer,
                                scheduler=scheduler, best=best, cfg=cfg, train_cfg=train_cfg)

            if epoch % train_cfg.ckpt_period == 0 or epoch == train_cfg.epochs:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", epoch=epoch, model=model, optimizer=optimizer,
                                scheduler=scheduler, best=best, cfg=cfg, train_cfg=train_cfg)

            seconds = time.perf_counter() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "test_loss": test_metrics["loss"],
                "test_mae": test_metrics["mae"],
                "best_epoch": best["epoch"],
                "best_val_mae": best["val_mae"],
                "best_test_mae": best["test_mae"],
                "seconds": seconds,
            }
            writer.writerow(row)
            f_csv.flush()

            log(
                f"[epoch {epoch:04d}] "
                f"train_loss={train_metrics['loss']:.5f} train_mae={train_metrics['mae']:.5f} | "
                f"val_loss={val_metrics['loss']:.5f} val_mae={val_metrics['mae']:.5f} | "
                f"test_loss={test_metrics['loss']:.5f} test_mae={test_metrics['mae']:.5f} | "
                f"best@{int(best['epoch']):04d} val_mae={best['val_mae']:.5f} test_mae={best['test_mae']:.5f} | "
                f"lr={lr_now:.3e} time={seconds:.1f}s"
            )

    log(f"[done] Results: {results_dir}")
    log(f"[done] Best checkpoint: {ckpt_dir / 'best.pt'}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone GraphormerSLIM ZINC(subset) Colab runner.")
    p.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    p.add_argument("--drive-dir", type=Path, default=Path("/content/drive/MyDrive/graphormer_zinc_standalone"))
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--pyg-version", type=str, default="2.2.0")
    p.add_argument("--force-reprocess", action="store_true", help="Rebuild precomputed shortest-path caches.")
    p.add_argument("--no-train", action="store_true", help="Install/import/data/model preflight only; do not train.")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--cpu", action="store_true")

    # Controlled training knobs.
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--clip-grad-norm", type=float, default=5.0)
    p.add_argument("--ckpt-period", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")

    # Model/preprocessing knobs.
    p.add_argument("--graph-pooling", choices=["graph_token", "sum", "mean"], default="graph_token", help="graph_token is exact Graphormer; sum/mean are controlled deviations.")
    p.add_argument("--multi-hop-max-dist", type=int, default=5)
    p.add_argument("--num-atom-types", type=int, default=29)
    p.add_argument("--num-edge-types", type=int, default=6)
    p.add_argument("--num-spatial", type=int, default=64)
    p.add_argument("--num-edge-dis", type=int, default=128)
    p.add_argument("--spatial-pos-max", type=int, default=1024)
    p.add_argument("--allow-param-mismatch", action="store_true")

    argv = strip_colab_kernel_args(list(sys.argv[1:] if argv is None else argv))
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    mount_drive(args.drive_mount)
    install_dependencies(skip_install=args.skip_install, pyg_version=args.pyg_version)
    train(args)


if __name__ == "__main__":
    main()
