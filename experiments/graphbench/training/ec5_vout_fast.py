#!/usr/bin/env python3
"""Fast EC5-vout pilot for Graphormer, GraphGPS, and GRIT-style models.

The runner targets GraphBench ``electronic_circuits_5_vout`` and keeps the
ZINC model identities while using much smaller pilot configs. It can load
through ``graphbench-lib`` when installed, or directly from the public EC5 JSON
archive used by GraphBench.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


DATASET_NAME = "electronic_circuits_5_vout"
EC5_URL = (
    "https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/"
    "resolve/main/ec_5.zip"
)
NODE_FEATURE_DIM = 9
DEFAULT_SIZE_BANDS = {
    "graphormer": (170_000, 200_000),
    "graphgps": (180_000, 240_000),
    "grit": (210_000, 280_000),
}


@dataclass(frozen=True)
class EC5FastConfig:
    preset: str = "ec5_vout_fast"
    dataset_name: str = DATASET_NAME
    seed: int = 0
    train_size: int = 50_000
    val_size: int = 5_000
    batch_size: int = 512
    max_epochs: int = 80
    patience: int = 10
    min_delta: float = 1e-4
    final_eval_size: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    grad_clip_norm: float = 5.0
    amp: bool = True
    graphormer_layers: int = 4
    graphormer_hidden_dim: int = 80
    graphormer_ffn_dim: int = 80
    graphormer_heads: int = 8
    graphormer_num_spatial: int = 16
    graphormer_multi_hop_max_dist: int = 10
    graphgps_layers: int = 4
    graphgps_hidden_dim: int = 64
    graphgps_heads: int = 4
    graphgps_rwse_steps: int = 10
    graphgps_pe_dim: int = 16
    grit_layers: int = 4
    grit_hidden_dim: int = 64
    grit_heads: int = 8
    grit_rrwp_steps: int = 10
    dropout: float = 0.0
    attn_dropout: float = 0.1


@dataclass
class ECGraph:
    node_type: torch.Tensor
    edge_index: torch.Tensor
    duty: torch.Tensor
    y: torch.Tensor
    cache: Optional[dict[str, torch.Tensor]] = None

    @property
    def num_nodes(self) -> int:
        return int(self.node_type.numel())


class ECGraphDataset(Dataset[ECGraph]):
    def __init__(self, graphs: Sequence[ECGraph]) -> None:
        self.graphs = list(graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> ECGraph:
        return self.graphs[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_2d_one_hot(x: object) -> torch.Tensor:
    x_tensor = torch.as_tensor(x, dtype=torch.float32)
    if x_tensor.dim() == 3 and x_tensor.size(1) == 1:
        x_tensor = x_tensor.squeeze(1)
    if x_tensor.dim() != 2 or x_tensor.size(1) != NODE_FEATURE_DIM:
        raise ValueError(f"expected node features [num_nodes, 9], got {tuple(x_tensor.shape)}")
    row_sums = x_tensor.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
        raise ValueError("EC5 node features must be one-hot rows")
    if float(x_tensor.min()) < -1e-5 or float(x_tensor.max()) > 1.0 + 1e-5:
        raise ValueError("EC5 node features must be one-hot values in [0, 1]")
    return x_tensor


def normalize_edge_index(edge_index: object) -> torch.Tensor:
    edge_tensor = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_tensor.dim() != 2:
        raise ValueError(f"expected edge_index with rank 2, got {tuple(edge_tensor.shape)}")
    if edge_tensor.size(0) == 2:
        return edge_tensor.contiguous()
    if edge_tensor.size(1) == 2:
        return edge_tensor.t().contiguous()
    raise ValueError(f"expected edge_index [2, E] or [E, 2], got {tuple(edge_tensor.shape)}")


def normalize_vout(raw_vout: float) -> float:
    return float(max(0.0, min(1.0, (float(raw_vout) + 300.0) / 600.0)))


def graph_from_json_record(record: Mapping[str, object]) -> ECGraph:
    x = as_2d_one_hot(record["node_features"])
    edge_index = normalize_edge_index(record["edge_index"])
    duty = torch.tensor(float(record["duty"]), dtype=torch.float32)
    y = torch.tensor(normalize_vout(float(record["vout"])), dtype=torch.float32)
    return ECGraph(node_type=x.argmax(dim=-1).long(), edge_index=edge_index, duty=duty, y=y)


def graph_from_graphbench_data(data: object) -> ECGraph:
    x = as_2d_one_hot(getattr(data, "x"))
    edge_index = normalize_edge_index(getattr(data, "edge_index"))
    duty = torch.as_tensor(getattr(data, "duty"), dtype=torch.float32).reshape(())
    y = torch.as_tensor(getattr(data, "y"), dtype=torch.float32).reshape(())
    y = torch.clamp(y, 0.0, 1.0)
    return ECGraph(node_type=x.argmax(dim=-1).long(), edge_index=edge_index, duty=duty, y=y)


def direct_json_dir(root: Path) -> Path:
    return root / "electroniccircuits" / DATASET_NAME / "raw"


def ensure_ec5_json(root: Path, log=print) -> Path:
    raw_dir = direct_json_dir(root)
    train_json = raw_dir / "dataset_5_train.json"
    valid_json = raw_dir / "dataset_5_valid.json"
    test_json = raw_dir / "dataset_5_test.json"
    if train_json.exists() and valid_json.exists() and test_json.exists():
        return raw_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ec_5.zip"
    if not zip_path.exists():
        log(f"[data] Downloading EC5 archive to {zip_path}")
        with urllib.request.urlopen(EC5_URL) as response, zip_path.open("wb") as out:
            shutil.copyfileobj(response, out)

    log(f"[data] Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    return raw_dir


def load_json_split(raw_dir: Path, split: Literal["train", "val", "test"]) -> list[ECGraph]:
    file_split = "valid" if split == "val" else split
    path = raw_dir / f"dataset_5_{file_split}.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return [graph_from_json_record(record) for record in records]


def load_direct_json_splits(root: Path, log=print) -> dict[str, ECGraphDataset]:
    raw_dir = ensure_ec5_json(root, log=log)
    return {
        "train": ECGraphDataset(load_json_split(raw_dir, "train")),
        "val": ECGraphDataset(load_json_split(raw_dir, "val")),
        "test": ECGraphDataset(load_json_split(raw_dir, "test")),
    }


def load_graphbench_splits(root: Path) -> dict[str, ECGraphDataset]:
    from graphbench import Loader  # type: ignore

    loaded = Loader(root=root, dataset_names=DATASET_NAME).load()
    if len(loaded) != 1:
        raise RuntimeError(f"expected one GraphBench dataset, got {len(loaded)}")
    split_map = loaded[0]
    return {
        split: ECGraphDataset(
            [graph_from_graphbench_data(split_map[split][i]) for i in range(len(split_map[split]))]
        )
        for split in ("train", "val", "test")
    }


def load_splits(source: str, root: Path, log=print) -> dict[str, ECGraphDataset]:
    if source in {"auto", "graphbench"}:
        try:
            log("[data] Loading EC5-vout through graphbench-lib")
            return load_graphbench_splits(root)
        except Exception as exc:
            if source == "graphbench":
                raise
            log(
                f"[data] graphbench-lib load unavailable ({type(exc).__name__}: {exc}); "
                "using JSON fallback"
            )
    return load_direct_json_splits(root, log=log)


def subset_indices(length: int, size: int, seed: int) -> list[int]:
    if size <= 0 or size >= length:
        return list(range(length))
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(length, generator=generator)[:size].sort().values.tolist()


def dataset_stats(dataset: Dataset[ECGraph]) -> dict[str, float]:
    n_graphs = len(dataset)
    node_counts = [dataset[i].num_nodes for i in range(n_graphs)]
    ys = torch.stack([dataset[i].y for i in range(n_graphs)])
    duties = torch.stack([dataset[i].duty for i in range(n_graphs)])
    return {
        "graphs": float(n_graphs),
        "min_nodes": float(min(node_counts)),
        "max_nodes": float(max(node_counts)),
        "mean_nodes": float(sum(node_counts) / max(1, n_graphs)),
        "target_min": float(ys.min()),
        "target_max": float(ys.max()),
        "target_mean": float(ys.mean()),
        "duty_min": float(duties.min()),
        "duty_max": float(duties.max()),
    }


def dense_adjacency(graph: ECGraph) -> torch.Tensor:
    n = graph.num_nodes
    adj = torch.zeros(n, n, dtype=torch.float32)
    if graph.edge_index.numel() > 0:
        src, dst = graph.edge_index
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
    adj.fill_diagonal_(0.0)
    return adj


def shortest_path_distances(adj: torch.Tensor) -> torch.Tensor:
    n = adj.size(0)
    dist = torch.full((n, n), 10_000, dtype=torch.long)
    for src in range(n):
        dist[src, src] = 0
        queue = [src]
        for u in queue:
            neighbors = torch.nonzero(adj[u] > 0, as_tuple=False).flatten().tolist()
            for v in neighbors:
                if dist[src, v] > dist[src, u] + 1:
                    dist[src, v] = dist[src, u] + 1
                    queue.append(v)
    return dist


def random_walk_features(adj: torch.Tensor, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = adj.size(0)
    degree = adj.sum(dim=-1, keepdim=True)
    transition = torch.where(degree > 0, adj / degree.clamp_min(1.0), torch.zeros_like(adj))
    power = torch.eye(n, dtype=torch.float32)
    rwse = []
    rrwp = [torch.eye(n, dtype=torch.float32)]
    for _ in range(steps):
        power = power @ transition
        rwse.append(torch.diagonal(power, dim1=-2, dim2=-1))
        rrwp.append(power.clone())
    return torch.stack(rwse, dim=-1), torch.stack(rrwp, dim=-1)


def cached_features(graph: ECGraph) -> dict[str, torch.Tensor]:
    if graph.cache is not None:
        return graph.cache

    max_hop = 10
    rw_steps = 10
    rrwp_steps = 10
    n = graph.num_nodes
    adj = dense_adjacency(graph)
    degree = adj.sum(dim=-1).long().clamp(max=15) + 1

    dist = shortest_path_distances(adj)
    shifted = torch.where(dist < 10_000, dist + 1, torch.zeros_like(dist))
    spatial_pos = shifted.clamp(max=15)
    edge_input = torch.zeros(n, n, max_hop, dtype=torch.long)
    for i in range(n):
        for j in range(n):
            d = int(dist[i, j])
            if 0 < d < 10_000:
                edge_input[i, j, : min(d, max_hop)] = 1

    rwse, rrwp = random_walk_features(adj, steps=rw_steps)
    graph.cache = {
        "adj": adj,
        "degree": degree,
        "spatial_pos": spatial_pos,
        "edge_input": edge_input,
        "rwse": rwse,
        "rrwp": rrwp[:, :, : rrwp_steps + 1],
    }
    return graph.cache


def collate_ec_graphs(graphs: Sequence[ECGraph]) -> dict[str, torch.Tensor]:
    batch_size = len(graphs)
    max_nodes = max(graph.num_nodes for graph in graphs)
    max_hop = 10
    rw_steps = 10
    rrwp_steps = 10

    node_type = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    adj = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.float32)
    degree = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    spatial_pos = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.long)
    edge_input = torch.zeros(batch_size, max_nodes, max_nodes, max_hop, dtype=torch.long)
    rwse = torch.zeros(batch_size, max_nodes, rw_steps, dtype=torch.float32)
    rrwp = torch.zeros(batch_size, max_nodes, max_nodes, rrwp_steps + 1, dtype=torch.float32)
    duty = torch.zeros(batch_size, 1, dtype=torch.float32)
    y = torch.zeros(batch_size, dtype=torch.float32)

    for bidx, graph in enumerate(graphs):
        n = graph.num_nodes
        features = cached_features(graph)
        node_type[bidx, :n] = graph.node_type + 1
        node_mask[bidx, :n] = True
        duty[bidx, 0] = graph.duty
        y[bidx] = graph.y
        adj[bidx, :n, :n] = features["adj"]
        degree[bidx, :n] = features["degree"]
        spatial_pos[bidx, :n, :n] = features["spatial_pos"]
        edge_input[bidx, :n, :n, :] = features["edge_input"]
        rwse[bidx, :n, :] = features["rwse"]
        rrwp[bidx, :n, :n, :] = features["rrwp"]

    return {
        "node_type": node_type,
        "node_mask": node_mask,
        "adj": adj,
        "degree": degree,
        "spatial_pos": spatial_pos,
        "edge_input": edge_input,
        "rwse": rwse,
        "rrwp": rrwp,
        "duty": duty,
        "y": y,
    }


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


class RegressionHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.duty_encoder = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, graph_repr: torch.Tensor, duty: torch.Tensor) -> torch.Tensor:
        return self.out(graph_repr + self.duty_encoder(duty)).squeeze(-1)


class MultiHeadAttentionWithBias(nn.Module):
    def __init__(self, dim: int, heads: int, attn_dropout: float, out_dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("hidden dim must be divisible by attention heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_bias
        scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, seq_len, dim)
        return self.out_dropout(self.out_proj(out))


class TransformerBlockWithBias(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        heads: int,
        attn_dropout: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithBias(dim, heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_bias, key_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class ECGraphormer(nn.Module):
    def __init__(self, cfg: EC5FastConfig) -> None:
        super().__init__()
        dim = cfg.graphormer_hidden_dim
        heads = cfg.graphormer_heads
        self.cfg = cfg
        self.node_encoder = nn.Embedding(NODE_FEATURE_DIM + 1, dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(17, dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(17, dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, dim)
        self.spatial_encoder = nn.Embedding(cfg.graphormer_num_spatial, heads, padding_idx=0)
        self.edge_encoder = nn.Embedding(2, heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, heads)
        self.emb_norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            TransformerBlockWithBias(
                dim,
                cfg.graphormer_ffn_dim,
                heads,
                cfg.attn_dropout,
                cfg.dropout,
            )
            for _ in range(cfg.graphormer_layers)
        )
        self.head = RegressionHead(dim)

    def build_bias(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        node_type = batch["node_type"]
        spatial_pos = batch["spatial_pos"].clamp(max=self.cfg.graphormer_num_spatial - 1)
        edge_input = batch["edge_input"][:, :, :, : self.cfg.graphormer_multi_hop_max_dist]
        bsz, num_nodes = node_type.shape
        heads = self.cfg.graphormer_heads
        bias = torch.zeros(
            bsz,
            heads,
            num_nodes + 1,
            num_nodes + 1,
            dtype=torch.float32,
            device=node_type.device,
        )
        spatial_bias = self.spatial_encoder(spatial_pos).permute(0, 3, 1, 2)
        edge_bias = self.edge_encoder(edge_input).mean(dim=-2).permute(0, 3, 1, 2)
        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + spatial_bias + edge_bias
        token_bias = self.graph_token_virtual_distance.weight.view(1, heads, 1)
        bias[:, :, 1:, 0] = bias[:, :, 1:, 0] + token_bias
        bias[:, :, 0, 1:] = bias[:, :, 0, 1:] + token_bias
        return bias

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        node_type = batch["node_type"]
        mask = batch["node_mask"]
        degree = batch["degree"]
        bsz = node_type.size(0)
        h = self.node_encoder(node_type)
        h = h + self.in_degree_encoder(degree) + self.out_degree_encoder(degree)
        graph_token = self.graph_token.weight.unsqueeze(0).expand(bsz, -1, -1)
        h = torch.cat([graph_token, h], dim=1)
        h = self.emb_norm(h)
        key_mask = torch.cat(
            [torch.ones(bsz, 1, dtype=torch.bool, device=mask.device), mask],
            dim=1,
        )
        attn_bias = self.build_bias(batch).to(h.dtype)
        for layer in self.layers:
            h = layer(h, attn_bias, key_mask)
        return self.head(h[:, 0], batch["duty"])


class DenseGINE(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_encoder = nn.Embedding(2, dim, padding_idx=0)
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm = nn.BatchNorm1d(dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        edge_emb = self.edge_encoder(torch.ones((), dtype=torch.long, device=h.device))
        messages = torch.relu(h + edge_emb.view(1, 1, -1))
        agg = torch.matmul(adj, messages)
        out = self.mlp((1.0 + self.eps) * h + agg)
        out = self.norm(out.reshape(-1, out.size(-1))).view_as(out)
        return out * mask.unsqueeze(-1)


class GPSLayer(nn.Module):
    def __init__(self, dim: int, heads: int, attn_dropout: float, dropout: float) -> None:
        super().__init__()
        self.local = DenseGINE(dim, dropout)
        self.local_norm = nn.LayerNorm(dim)
        self.global_attn = nn.MultiheadAttention(
            dim,
            heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.global_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = h + self.local(self.local_norm(h), adj, mask)
        global_out, _ = self.global_attn(
            self.global_norm(h),
            self.global_norm(h),
            self.global_norm(h),
            key_padding_mask=~mask,
            need_weights=False,
        )
        h = h + global_out * mask.unsqueeze(-1)
        h = h + self.ffn(self.ffn_norm(h)) * mask.unsqueeze(-1)
        return h * mask.unsqueeze(-1)


class ECGraphGPS(nn.Module):
    def __init__(self, cfg: EC5FastConfig) -> None:
        super().__init__()
        dim = cfg.graphgps_hidden_dim
        self.type_encoder = nn.Embedding(NODE_FEATURE_DIM + 1, dim, padding_idx=0)
        self.pe_bn = nn.BatchNorm1d(cfg.graphgps_rwse_steps)
        self.pe_encoder = nn.Sequential(
            nn.Linear(cfg.graphgps_rwse_steps, cfg.graphgps_pe_dim),
            nn.ReLU(),
            nn.Linear(cfg.graphgps_pe_dim, dim),
        )
        self.layers = nn.ModuleList(
            GPSLayer(dim, cfg.graphgps_heads, cfg.attn_dropout, cfg.dropout)
            for _ in range(cfg.graphgps_layers)
        )
        self.pool_norm = nn.LayerNorm(dim)
        self.head = RegressionHead(dim)

    def encode_pe(self, rwse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, num_nodes, steps = rwse.shape
        pe = self.pe_bn(rwse.reshape(-1, steps)).view(bsz, num_nodes, steps)
        return self.pe_encoder(pe) * mask.unsqueeze(-1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["node_mask"]
        h = self.type_encoder(batch["node_type"]) + self.encode_pe(batch["rwse"], mask)
        h = h * mask.unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, batch["adj"], mask)
        graph_repr = (h * mask.unsqueeze(-1)).sum(dim=1)
        return self.head(self.pool_norm(graph_repr), batch["duty"])


class GritLayer(nn.Module):
    def __init__(self, dim: int, heads: int, attn_dropout: float, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("hidden dim must be divisible by attention heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.node_norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.edge_to_bias = nn.Linear(dim, heads)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.node_norm2 = nn.LayerNorm(dim)
        self.node_ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.edge_norm = nn.LayerNorm(dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_repr: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, num_nodes, dim = h.shape
        hn = self.node_norm1(h)
        q = self.q_proj(hn).view(bsz, num_nodes, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hn).view(bsz, num_nodes, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hn).view(bsz, num_nodes, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        edge_bias = self.edge_to_bias(edge_repr).permute(0, 3, 1, 2)
        scores = scores + edge_bias
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        node_delta = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, num_nodes, dim)
        h = h + self.out_dropout(self.out_proj(node_delta)) * mask.unsqueeze(-1)
        h = h + self.node_ffn(self.node_norm2(h)) * mask.unsqueeze(-1)

        src = h.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        dst = h.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        edge_delta = self.edge_mlp(torch.cat([edge_repr, src, dst], dim=-1))
        pair_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
        edge_repr = self.edge_norm(edge_repr + edge_delta) * pair_mask
        return h * mask.unsqueeze(-1), edge_repr


class ECGrit(nn.Module):
    def __init__(self, cfg: EC5FastConfig) -> None:
        super().__init__()
        dim = cfg.grit_hidden_dim
        self.type_encoder = nn.Embedding(NODE_FEATURE_DIM + 1, dim, padding_idx=0)
        self.rrwp_encoder = nn.Linear(cfg.grit_rrwp_steps + 1, dim)
        self.layers = nn.ModuleList(
            GritLayer(dim, cfg.grit_heads, cfg.attn_dropout, cfg.dropout)
            for _ in range(cfg.grit_layers)
        )
        self.pool_norm = nn.LayerNorm(dim)
        self.head = RegressionHead(dim)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["node_mask"]
        h = self.type_encoder(batch["node_type"]) * mask.unsqueeze(-1)
        edge_repr = self.rrwp_encoder(batch["rrwp"])
        pair_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
        edge_repr = edge_repr * pair_mask
        for layer in self.layers:
            h, edge_repr = layer(h, edge_repr, mask)
        graph_repr = (h * mask.unsqueeze(-1)).sum(dim=1)
        return self.head(self.pool_norm(graph_repr), batch["duty"])


def build_model(model_name: str, cfg: EC5FastConfig) -> nn.Module:
    if model_name == "graphormer":
        return ECGraphormer(cfg)
    if model_name == "graphgps":
        return ECGraphGPS(cfg)
    if model_name == "grit":
        return ECGrit(cfg)
    raise ValueError(f"unknown model {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def assert_size_band(model_name: str, n_params: int, skip: bool = False) -> None:
    if skip:
        return
    lo, hi = DEFAULT_SIZE_BANDS[model_name]
    if not (lo <= n_params <= hi):
        raise RuntimeError(
            f"{model_name} has {n_params:,} trainable parameters; expected {lo:,}-{hi:,}. "
            "Use --skip-size-check only for deliberate ablations."
        )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, warmup_steps)
    total_steps = max(total_steps, warmup_steps + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def metric_values(
    pred: torch.Tensor,
    target: torch.Tensor,
    baseline_mean: float,
) -> dict[str, float]:
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    denom = torch.mean((target - target.mean()) ** 2).clamp_min(1e-12)
    rse = float(torch.mean((pred - target) ** 2) / denom)
    baseline = torch.full_like(target, float(baseline_mean))
    baseline_rse = float(torch.mean((baseline - target) ** 2) / denom)
    return {"mse": mse, "mae": mae, "rse": rse, "train_mean_baseline_rse": baseline_rse}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    baseline_mean: float,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            pred = model(batch)
        preds.append(pred.detach().cpu())
        targets.append(batch["y"].detach().cpu())
    return metric_values(torch.cat(preds), torch.cat(targets), baseline_mean=baseline_mean)


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def train_model(
    model_name: str,
    splits: Mapping[str, ECGraphDataset],
    cfg: EC5FastConfig,
    output_root: Path,
    device: torch.device,
    skip_size_check: bool = False,
) -> dict[str, object]:
    run_dir = output_root / cfg.preset / model_name / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(run_dir / "run.log")
    log(f"[run] model={model_name} preset={cfg.preset} seed={cfg.seed} device={device}")

    train_indices = subset_indices(len(splits["train"]), cfg.train_size, cfg.seed)
    val_indices = subset_indices(len(splits["val"]), cfg.val_size, cfg.seed + 17)
    train_set = Subset(splits["train"], train_indices)
    sampled_val_set = Subset(splits["val"], val_indices)

    train_mean = float(torch.stack([train_set[i].y for i in range(len(train_set))]).mean())
    data_summary = {
        "full_train": dataset_stats(splits["train"]),
        "sampled_train": dataset_stats(train_set),
        "full_val": dataset_stats(splits["val"]),
        "sampled_val": dataset_stats(sampled_val_set),
        "full_test": dataset_stats(splits["test"]),
    }
    write_json(run_dir / "dataset_stats.json", data_summary)
    write_json(run_dir / "config.json", asdict(cfg) | {"model": model_name})

    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_ec_graphs,
        num_workers=0,
    )
    sampled_val_loader = DataLoader(
        sampled_val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_ec_graphs,
        num_workers=0,
    )
    full_val_loader = DataLoader(
        Subset(
            splits["val"],
            subset_indices(len(splits["val"]), cfg.final_eval_size, cfg.seed + 31),
        )
        if cfg.final_eval_size > 0
        else splits["val"],
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_ec_graphs,
        num_workers=0,
    )
    full_test_loader = DataLoader(
        Subset(
            splits["test"],
            subset_indices(len(splits["test"]), cfg.final_eval_size, cfg.seed + 47),
        )
        if cfg.final_eval_size > 0
        else splits["test"],
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_ec_graphs,
        num_workers=0,
    )

    model = build_model(model_name, cfg).to(device)
    n_params = count_parameters(model)
    assert_size_band(model_name, n_params, skip=skip_size_check)
    log(f"[model] trainable_parameters={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.max_epochs * max(1, len(train_loader))
    warmup_steps = cfg.warmup_epochs * max(1, len(train_loader))
    scheduler = make_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    use_amp = cfg.amp and device.type == "cuda"

    metrics_path = run_dir / "metrics.csv"
    best_rse = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_path = run_dir / "best.pt"
    fieldnames = [
        "epoch",
        "lr",
        "train_mse",
        "train_mae",
        "train_rse",
        "sampled_val_mse",
        "sampled_val_mae",
        "sampled_val_rse",
        "sampled_val_train_mean_baseline_rse",
        "seconds",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, cfg.max_epochs + 1):
            started = time.time()
            model.train()
            train_preds = []
            train_targets = []
            for batch in train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    pred = model(batch)
                    loss = F.mse_loss(pred, batch["y"])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_preds.append(pred.detach().cpu())
                train_targets.append(batch["y"].detach().cpu())

            train_metrics = metric_values(
                torch.cat(train_preds),
                torch.cat(train_targets),
                baseline_mean=train_mean,
            )
            val_metrics = evaluate(model, sampled_val_loader, device, train_mean, use_amp=use_amp)
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "train_rse": train_metrics["rse"],
                "sampled_val_mse": val_metrics["mse"],
                "sampled_val_mae": val_metrics["mae"],
                "sampled_val_rse": val_metrics["rse"],
                "sampled_val_train_mean_baseline_rse": val_metrics["train_mean_baseline_rse"],
                "seconds": time.time() - started,
            }
            writer.writerow(row)
            handle.flush()
            log(
                f"[epoch {epoch:03d}] train_mse={row['train_mse']:.6f} "
                f"sampled_val_rse={row['sampled_val_rse']:.6f} "
                f"baseline_rse={row['sampled_val_train_mean_baseline_rse']:.6f} "
                f"time={row['seconds']:.1f}s"
            )

            if val_metrics["rse"] < best_rse - cfg.min_delta:
                best_rse = val_metrics["rse"]
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(cfg),
                        "model_name": model_name,
                        "epoch": epoch,
                        "sampled_val_metrics": val_metrics,
                        "trainable_parameters": n_params,
                    },
                    best_path,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    log(f"[early-stop] patience={cfg.patience} best_epoch={best_epoch}")
                    break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    full_val_metrics = evaluate(model, full_val_loader, device, train_mean, use_amp=use_amp)
    full_test_metrics = evaluate(model, full_test_loader, device, train_mean, use_amp=use_amp)
    accepted = full_test_metrics["rse"] < full_test_metrics["train_mean_baseline_rse"]
    summary = {
        "model": model_name,
        "preset": cfg.preset,
        "seed": cfg.seed,
        "trainable_parameters": n_params,
        "best_epoch": best_epoch,
        "best_sampled_val_rse": best_rse,
        "full_val_metrics": full_val_metrics,
        "full_test_metrics": full_test_metrics,
        "pilot_acceptance": {
            "beats_train_mean_baseline": accepted,
            "status": "passed" if accepted else "failed",
        },
        "best_checkpoint": str(best_path),
    }
    write_json(run_dir / "summary.json", summary)
    log(
        f"[done] full_test_rse={full_test_metrics['rse']:.6f} "
        f"baseline={full_test_metrics['train_mean_baseline_rse']:.6f} "
        f"accepted={accepted}"
    )
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EC5-vout fast pilot graph transformer configs."
    )
    parser.add_argument("--model", choices=["graphormer", "graphgps", "grit", "all"], default="all")
    parser.add_argument("--preset", choices=["ec5_vout_fast"], default="ec5_vout_fast")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--val-size", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--final-eval-size",
        type=int,
        default=0,
        help="Limit final val/test evaluation for smoke runs. Default 0 evaluates full splits.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/graphbench"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/graphbench/results"))
    parser.add_argument("--source", choices=["auto", "graphbench", "json"], default="auto")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-size-check", action="store_true")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Override to a 1-epoch 1024/256 sample run for integration checks.",
    )
    return parser.parse_args(argv)


def strip_colab_kernel_args(argv: Sequence[str]) -> list[str]:
    """Drop IPython/Colab launcher args injected into sys.argv."""
    cleaned: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in {"-f", "--f", "--file"}:
            skip_next = True
            continue
        if arg.startswith("--f=") or arg.startswith("--file="):
            continue
        if "jupyter/runtime/kernel-" in arg or arg.endswith(".json") and "kernel-" in arg:
            continue
        cleaned.append(arg)
    return cleaned


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def config_from_args(args: argparse.Namespace) -> EC5FastConfig:
    train_size = args.train_size
    val_size = args.val_size
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    patience = args.patience
    final_eval_size = args.final_eval_size
    if args.fast_dev_run:
        train_size = min(train_size, 1024)
        val_size = min(val_size, 256)
        batch_size = min(batch_size, 128)
        max_epochs = 1
        patience = 1
        final_eval_size = final_eval_size or 512
    return EC5FastConfig(
        preset=args.preset,
        seed=args.seed,
        train_size=train_size,
        val_size=val_size,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        min_delta=args.min_delta,
        final_eval_size=final_eval_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        amp=not args.no_amp,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = strip_colab_kernel_args(list(sys.argv[1:] if argv is None else argv))
    args = parse_args(argv)
    cfg = config_from_args(args)
    set_seed(cfg.seed)
    device = resolve_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    top_log = RunLogger(args.output_root / cfg.preset / "run.log")
    top_log(f"[setup] source={args.source} dataset_root={args.dataset_root}")
    splits = load_splits(args.source, args.dataset_root, log=top_log)
    models = ["graphormer", "graphgps", "grit"] if args.model == "all" else [args.model]
    summaries = []
    for model_name in models:
        summaries.append(
            train_model(
                model_name,
                splits,
                cfg,
                args.output_root,
                device,
                skip_size_check=args.skip_size_check,
            )
        )
    write_json(args.output_root / cfg.preset / "summary.json", {"runs": summaries})


if __name__ == "__main__":
    main(sys.argv[1:])
