#!/usr/bin/env python3
"""Colab standalone EC5-vout pilot for Graphormer, GraphGPS, and GRIT.

The runner targets GraphBench ``electronic_circuits_5_vout`` and keeps the
ZINC model identities while using much smaller pilot configs. It can load
through ``graphbench-lib`` when installed, or directly from the public EC JSON
archives used by GraphBench. After EC5 training it can zero-shot evaluate EC7
and EC10, and it writes alpha-weighted positional/symbolic attention metrics.

Default Colab command:

    python ec5_vout_fast_colab.py --model all

This file is intentionally self-contained: it only requires PyTorch plus the
Python standard library. ``graphbench-lib`` is optional; if unavailable, the
runner downloads and parses the same public EC5 JSON archive directly.
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


TRAIN_COMPONENTS = 5
DATASET_NAME = "electronic_circuits_5_vout"
EC_URL_TEMPLATE = (
    "https://huggingface.co/datasets/log-rwth-aachen/Graphbench_ElectronicCircuits/"
    "resolve/main/ec_{components}.zip"
)
NODE_FEATURE_DIM = 9
M_POSITIONAL = "positional_score"
M_SYMBOLIC = "symbolic_score"
M_PE_INVARIANT = "pe_invariance"
M_PE_EQUIVARIANT = "pe_equivariance"
M_JOINT_EQUIVARIANT = "joint_equivariance"
M_POSITIONAL_CENTERED = "positional_score_centered"
M_SYMBOLIC_CENTERED = "symbolic_score_centered"
M_PE_INVARIANT_CENTERED = "pe_invariance_centered"
M_PE_EQUIVARIANT_CENTERED = "pe_equivariance_centered"
M_JOINT_EQUIVARIANT_CENTERED = "joint_equivariance_centered"
M_INTERACTION_RESIDUAL_CENTERED = "interaction_residual_norm_centered"
M_JOINT_EQUIVARIANCE_EXCESS_CENTERED = "joint_equivariance_excess_centered"
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
    metric_graphs: int = 512
    metric_batch_size: int = 128
    num_metric_perms: int = 16
    metric_alpha_tau: float = 0.1
    zero_shot_components: tuple[int, ...] = (7, 10)
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


def mount_drive(mount_point: Path, enabled: bool, log=print) -> None:
    if not enabled:
        log("[drive] Drive mount disabled.")
        return
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        log("[drive] google.colab unavailable; continuing without mounting Drive.")
        return
    log(f"[drive] Mounting Google Drive at {mount_point}")
    drive.mount(str(mount_point), force_remount=False)


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


def ec_dataset_name(components: int) -> str:
    return f"electronic_circuits_{components}_vout"


def direct_json_dir(root: Path, components: int) -> Path:
    return root / "electroniccircuits" / ec_dataset_name(components) / "raw"


def expected_ec_json_paths(raw_dir: Path, components: int) -> dict[str, Path]:
    return {
        "train": raw_dir / f"dataset_{components}_train.json",
        "valid": raw_dir / f"dataset_{components}_valid.json",
        "test": raw_dir / f"dataset_{components}_test.json",
    }


def normalize_extracted_ec_jsons(raw_dir: Path, components: int, log=print) -> None:
    expected = expected_ec_json_paths(raw_dir, components)
    if all(path.exists() for path in expected.values()):
        return

    for split, dest in expected.items():
        if dest.exists():
            continue
        pattern = f"dataset_{components}_{split}.json"
        candidates = [path for path in raw_dir.rglob(pattern) if path != dest]
        if not candidates:
            available = sorted(str(path.relative_to(raw_dir)) for path in raw_dir.rglob("*.json"))
            raise FileNotFoundError(
                f"Could not find {pattern} under {raw_dir}. Available JSON files: {available}"
            )
        src = candidates[0]
        shutil.copyfile(src, dest)
        log(f"[data] Normalized {src.relative_to(raw_dir)} -> {dest.name}")


def ensure_ec_json(root: Path, components: int, log=print) -> Path:
    raw_dir = direct_json_dir(root, components)
    expected = expected_ec_json_paths(raw_dir, components)
    if all(path.exists() for path in expected.values()):
        return raw_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"ec_{components}.zip"
    if not zip_path.exists():
        url = EC_URL_TEMPLATE.format(components=components)
        log(f"[data] Downloading EC{components} archive to {zip_path}")
        with urllib.request.urlopen(url) as response, zip_path.open("wb") as out:
            shutil.copyfileobj(response, out)

    log(f"[data] Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    normalize_extracted_ec_jsons(raw_dir, components, log=log)
    return raw_dir


def load_json_split(
    raw_dir: Path,
    components: int,
    split: Literal["train", "val", "test"],
) -> list[ECGraph]:
    file_split = "valid" if split == "val" else split
    path = raw_dir / f"dataset_{components}_{file_split}.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    return [graph_from_json_record(record) for record in records]


def load_direct_json_splits(
    root: Path,
    components: int = TRAIN_COMPONENTS,
    log=print,
) -> dict[str, ECGraphDataset]:
    raw_dir = ensure_ec_json(root, components, log=log)
    return {
        "train": ECGraphDataset(load_json_split(raw_dir, components, "train")),
        "val": ECGraphDataset(load_json_split(raw_dir, components, "val")),
        "test": ECGraphDataset(load_json_split(raw_dir, components, "test")),
    }


def load_graphbench_splits(
    root: Path,
    dataset_name: str = DATASET_NAME,
) -> dict[str, ECGraphDataset]:
    from graphbench import Loader  # type: ignore

    loaded = Loader(root=root, dataset_names=dataset_name).load()
    if len(loaded) != 1:
        raise RuntimeError(f"expected one GraphBench dataset, got {len(loaded)}")
    split_map = loaded[0]
    return {
        split: ECGraphDataset(
            [graph_from_graphbench_data(split_map[split][i]) for i in range(len(split_map[split]))]
        )
        for split in ("train", "val", "test")
    }


def load_splits(
    source: str,
    root: Path,
    components: int = TRAIN_COMPONENTS,
    log=print,
) -> dict[str, ECGraphDataset]:
    dataset_name = ec_dataset_name(components)
    if source in {"auto", "graphbench"}:
        try:
            log(f"[data] Loading {dataset_name} through graphbench-lib")
            return load_graphbench_splits(root, dataset_name=dataset_name)
        except Exception as exc:
            if source == "graphbench":
                raise
            log(
                f"[data] graphbench-lib load unavailable ({type(exc).__name__}: {exc}); "
                "using JSON fallback"
            )
    return load_direct_json_splits(root, components=components, log=log)


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
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        out = self.out_dropout(self.out_proj(out))
        if collect_attention:
            return out, attn
        return out


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
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.attn(self.norm1(x), attn_bias, key_mask, collect_attention)
        if collect_attention:
            attn_delta, attn = attn_out
        else:
            attn_delta = attn_out
            attn = None
        x = x + attn_delta
        x = x + self.ffn(self.norm2(x))
        if collect_attention:
            return x, attn
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

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
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
        layers = []
        for layer_idx, layer in enumerate(self.layers):
            if collect_attention:
                h, attn = layer(h, attn_bias, key_mask, collect_attention=True)
                layers.append(
                    {
                        "layer": layer_idx,
                        "attn": attn.detach(),
                        "node_mask": key_mask.detach(),
                    }
                )
            else:
                h = layer(h, attn_bias, key_mask)
        pred = self.head(h[:, 0], batch["duty"])
        if collect_attention:
            return pred, layers
        return pred


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

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = h + self.local(self.local_norm(h), adj, mask)
        global_out, attn = self.global_attn(
            self.global_norm(h),
            self.global_norm(h),
            self.global_norm(h),
            key_padding_mask=~mask,
            need_weights=collect_attention,
            average_attn_weights=False,
        )
        h = h + global_out * mask.unsqueeze(-1)
        h = h + self.ffn(self.ffn_norm(h)) * mask.unsqueeze(-1)
        if collect_attention:
            return h * mask.unsqueeze(-1), attn
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

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        mask = batch["node_mask"]
        h = self.type_encoder(batch["node_type"]) + self.encode_pe(batch["rwse"], mask)
        h = h * mask.unsqueeze(-1)
        layers = []
        for layer_idx, layer in enumerate(self.layers):
            if collect_attention:
                h, attn = layer(h, batch["adj"], mask, collect_attention=True)
                layers.append(
                    {
                        "layer": layer_idx,
                        "attn": attn.detach(),
                        "node_mask": mask.detach(),
                    }
                )
            else:
                h = layer(h, batch["adj"], mask)
        graph_repr = (h * mask.unsqueeze(-1)).sum(dim=1)
        pred = self.head(self.pool_norm(graph_repr), batch["duty"])
        if collect_attention:
            return pred, layers
        return pred


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
        collect_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if collect_attention:
            return h * mask.unsqueeze(-1), edge_repr, attn
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

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        collect_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        mask = batch["node_mask"]
        h = self.type_encoder(batch["node_type"]) * mask.unsqueeze(-1)
        edge_repr = self.rrwp_encoder(batch["rrwp"])
        pair_mask = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(-1)
        edge_repr = edge_repr * pair_mask
        layers = []
        for layer_idx, layer in enumerate(self.layers):
            if collect_attention:
                h, edge_repr, attn = layer(h, edge_repr, mask, collect_attention=True)
                layers.append(
                    {
                        "layer": layer_idx,
                        "attn": attn.detach(),
                        "node_mask": mask.detach(),
                    }
                )
            else:
                h, edge_repr = layer(h, edge_repr, mask)
        graph_repr = (h * mask.unsqueeze(-1)).sum(dim=1)
        pred = self.head(self.pool_norm(graph_repr), batch["duty"])
        if collect_attention:
            return pred, layers
        return pred


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


@torch.no_grad()
def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            pred = model(batch)
        preds.append(pred.detach().cpu())
        targets.append(batch["y"].detach().cpu())
    return torch.cat(preds), torch.cat(targets)


def write_json(path: Path, data: Mapping[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({field for row in rows for field in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pair_mask_from_layer(layer: Mapping[str, torch.Tensor]) -> torch.Tensor:
    node_mask = layer["node_mask"].to(dtype=torch.bool)
    return node_mask[:, None, :, None] & node_mask[:, None, None, :]


@dataclass
class PermutationMetricOutputs:
    metric_rows: list[dict[str, object]]
    query_rows: list[dict[str, object]]


def gather_dense_node_axis(t: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    batch_size, max_n = perm_pos.shape
    extra = t.dim() - 2
    idx = perm_pos.to(t.device).view(batch_size, max_n, *([1] * extra)).expand_as(t)
    return torch.gather(t, dim=1, index=idx)


def gather_dense_pair_axes(t: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    batch_size, max_n = perm_pos.shape
    extra = t.dim() - 3
    row_idx = (
        perm_pos.to(t.device)
        .view(batch_size, max_n, 1, *([1] * extra))
        .expand_as(t)
    )
    rows = torch.gather(t, dim=1, index=row_idx)
    col_idx = (
        perm_pos.to(t.device)
        .view(batch_size, 1, max_n, *([1] * extra))
        .expand_as(rows)
    )
    return torch.gather(rows, dim=2, index=col_idx)


def transform_pair_reference(z: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    batch_size, heads, max_n, _ = z.shape
    idx = perm_pos.to(z.device)
    row_idx = idx[:, None, :, None].expand(batch_size, heads, max_n, max_n)
    z_rows = torch.gather(z, dim=2, index=row_idx)
    col_idx = idx[:, None, None, :].expand(batch_size, heads, max_n, max_n)
    return torch.gather(z_rows, dim=3, index=col_idx)


def expand_head_mask(mask: torch.Tensor, heads: int) -> torch.Tensor:
    if mask.size(1) == 1 and heads != 1:
        return mask.expand(-1, heads, -1, -1)
    return mask


def row_center_tensor(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.size(1) == 1 and z.size(1) != 1:
        mask = mask.expand(-1, z.size(1), -1, -1)
    z0 = torch.where(mask, torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(z.dtype)
    mean = z0.sum(dim=-1, keepdim=True) / denom
    return torch.where(mask, z0 - mean, torch.zeros_like(z0))


def cosine_by_query_tensor(
    u: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = expand_head_mask(mask.to(device=u.device, dtype=torch.bool), u.size(1))
    u0 = torch.where(mask, torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    v0 = torch.where(mask, torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    num = (u0 * v0).sum(dim=-1)
    den = (
        torch.sqrt((u0 * u0).sum(dim=-1).clamp_min(1.0e-12))
        * torch.sqrt((v0 * v0).sum(dim=-1).clamp_min(1.0e-12))
    )
    return torch.clamp(num / den.clamp_min(1.0e-12), 0.0, 1.0)


def raw_cosine_by_query_tensor(
    u: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = expand_head_mask(mask.to(device=u.device, dtype=torch.bool), u.size(1))
    u0 = torch.where(mask, torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    v0 = torch.where(mask, torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    num = (u0 * v0).sum(dim=-1)
    den = (
        torch.sqrt((u0 * u0).sum(dim=-1).clamp_min(1.0e-12))
        * torch.sqrt((v0 * v0).sum(dim=-1).clamp_min(1.0e-12))
    )
    return torch.clamp(num / den.clamp_min(1.0e-12), -1.0, 1.0)


def attention_moved_mass_by_query(
    clean_attn: torch.Tensor,
    perm_pos: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ref = transform_pair_reference(clean_attn, perm_pos)
    ref_mask = transform_pair_reference(mask, perm_pos).to(dtype=torch.bool)
    moved_mask = mask.to(device=clean_attn.device, dtype=torch.bool) | ref_mask
    moved_mask = expand_head_mask(moved_mask, clean_attn.size(1))
    clean0 = torch.where(moved_mask, clean_attn, torch.zeros_like(clean_attn))
    ref0 = torch.where(moved_mask, ref, torch.zeros_like(ref))
    return torch.clamp(0.5 * torch.abs(clean0 - ref0).sum(dim=-1), 0.0, 1.0)


def interaction_residual_norm_by_query_tensor(
    clean: torch.Tensor,
    var_x: torch.Tensor,
    var_pe: torch.Tensor,
    var_both: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = expand_head_mask(mask.to(device=clean.device, dtype=torch.bool), clean.size(1))
    clean0 = torch.where(mask, clean, torch.zeros_like(clean))
    x0 = torch.where(mask, var_x, torch.zeros_like(var_x))
    pe0 = torch.where(mask, var_pe, torch.zeros_like(var_pe))
    both0 = torch.where(mask, var_both, torch.zeros_like(var_both))
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


def make_node_permutation(
    batch: Mapping[str, torch.Tensor],
    generator: torch.Generator,
) -> torch.Tensor:
    node_type = batch["node_type"]
    mask = batch["node_mask"]
    batch_size, max_n = node_type.shape
    perm_pos = torch.zeros((batch_size, max_n), dtype=torch.long, device=node_type.device)
    for graph_idx in range(batch_size):
        n = int(mask[graph_idx].sum().item())
        perm = torch.randperm(n, generator=generator).to(node_type.device)
        perm_pos[graph_idx, :n] = perm
        if n < max_n:
            perm_pos[graph_idx, n:] = torch.arange(n, max_n, device=node_type.device)
    return perm_pos


def make_node_type_permutation(
    batch: Mapping[str, torch.Tensor],
    perm_pos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    variant = {key: value for key, value in batch.items()}
    variant["node_type"] = gather_dense_node_axis(batch["node_type"], perm_pos)
    return variant


def make_structural_permutation(
    batch: Mapping[str, torch.Tensor],
    perm_pos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    variant = {key: value for key, value in batch.items()}
    for key in ("degree", "rwse"):
        variant[key] = gather_dense_node_axis(batch[key], perm_pos)
    for key in ("adj", "spatial_pos", "edge_input", "rrwp"):
        variant[key] = gather_dense_pair_axes(batch[key], perm_pos)
    return variant


def make_joint_permutation(
    batch: Mapping[str, torch.Tensor],
    perm_pos: torch.Tensor,
) -> dict[str, torch.Tensor]:
    variant = make_structural_permutation(batch, perm_pos)
    variant["node_type"] = gather_dense_node_axis(batch["node_type"], perm_pos)
    return variant


def perm_for_attention(attn: torch.Tensor, perm_pos: torch.Tensor) -> torch.Tensor:
    if attn.size(-1) == perm_pos.size(1) + 1:
        batch_size, max_n = perm_pos.shape
        out = torch.zeros((batch_size, max_n + 1), dtype=torch.long, device=perm_pos.device)
        out[:, 0] = 0
        out[:, 1:] = perm_pos + 1
        return out
    return perm_pos


def attention_basic_rows(
    clean: Mapping[str, torch.Tensor],
    meta: Mapping[str, object],
    batch_idx: int,
) -> list[dict[str, object]]:
    attn = clean["attn"].float()
    mask = expand_head_mask(pair_mask_from_layer(clean).to(device=attn.device), attn.size(1))
    active_rows = mask.any(dim=-1)

    attn0 = torch.where(mask, attn, torch.zeros_like(attn))
    key_count = mask.sum(dim=-1).to(attn.dtype)
    entropy = -(attn0 * torch.log(attn0.clamp_min(1.0e-12))).sum(dim=-1)
    entropy = entropy / torch.log(key_count.clamp_min(2.0))
    entropy_den = active_rows.sum(dim=(0, 2)).clamp_min(1).to(attn.dtype)
    entropy_head = (entropy * active_rows.to(attn.dtype)).sum(dim=(0, 2)) / entropy_den

    centered = row_center_tensor(attn, mask)
    raw_norm = torch.sqrt((centered * centered).sum(dim=-1).clamp_min(0.0))
    max_norm = torch.sqrt((1.0 - 1.0 / key_count.clamp_min(2.0)).clamp_min(1.0e-12))
    residual = torch.where(key_count > 1, raw_norm / max_norm.clamp_min(1.0e-12), 0.0)
    residual_den = (key_count > 1).sum(dim=(0, 2)).clamp_min(1).to(attn.dtype)
    residual_head = (residual * (key_count > 1).to(attn.dtype)).sum(dim=(0, 2)) / residual_den

    rows = []
    for head in range(int(attn.size(1))):
        common = {
            **meta,
            "batch": batch_idx,
            "graph_in_batch": -1,
            "perm": -1,
            "layer": int(clean["layer"]),
            "head": head,
            "metric_alpha_tau": "",
            "num_perms": 0,
            "moved_mass_mean": "",
            "alpha_max_mean": "",
            "effective_perms_mean": "",
        }
        rows.append({**common, "metric": "entropy_norm", "score": float(entropy_head[head])})
        rows.append(
            {
                **common,
                "metric": "attention_residual_norm",
                "score": float(residual_head[head]),
            }
        )
    return rows


def aggregate_alpha_metric_rows(
    score_lists: Mapping[str, list[torch.Tensor]],
    moved_masses: Sequence[torch.Tensor],
    node_mask: torch.Tensor,
    meta: Mapping[str, object],
    batch_idx: int,
    layer: int,
    alpha_tau: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not moved_masses:
        return [], []
    if alpha_tau <= 0:
        raise ValueError("--metric-alpha-tau must be positive")

    moved = torch.stack(list(moved_masses), dim=0)
    alpha = torch.softmax(moved / alpha_tau, dim=0)
    alpha_max = alpha.max(dim=0).values
    effective_perms = 1.0 / torch.square(alpha).sum(dim=0).clamp_min(1.0e-12)
    valid_queries = node_mask[:, None, :].to(device=moved.device, dtype=torch.bool)
    valid_queries = valid_queries.expand(-1, moved.size(2), -1)

    rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    for metric, values in score_lists.items():
        if not values:
            continue
        scores = torch.stack(values, dim=0)
        weighted_query = (alpha * scores).sum(dim=0)
        denom = valid_queries.sum(dim=-1).clamp_min(1).to(weighted_query.dtype)
        head_score = (weighted_query * valid_queries.to(weighted_query.dtype)).sum(dim=-1) / denom
        moved_mean = (moved.mean(dim=0) * valid_queries.to(moved.dtype)).sum(dim=-1) / denom
        alpha_max_mean = (alpha_max * valid_queries.to(alpha_max.dtype)).sum(dim=-1) / denom
        effective_mean = (
            effective_perms * valid_queries.to(effective_perms.dtype)
        ).sum(dim=-1) / denom

        batch_size, heads, num_queries = weighted_query.shape
        for graph_idx in range(batch_size):
            for head in range(heads):
                common = {
                    **meta,
                    "batch": batch_idx,
                    "graph_in_batch": graph_idx,
                    "perm": -1,
                    "layer": layer,
                    "head": head,
                    "metric": metric,
                    "metric_alpha_tau": alpha_tau,
                    "num_perms": len(moved_masses),
                    "moved_mass_mean": float(moved_mean[graph_idx, head].detach().cpu()),
                    "alpha_max_mean": float(alpha_max_mean[graph_idx, head].detach().cpu()),
                    "effective_perms_mean": float(effective_mean[graph_idx, head].detach().cpu()),
                }
                rows.append(
                    {
                        **common,
                        "score": float(head_score[graph_idx, head].detach().cpu()),
                    }
                )
                for query in range(num_queries):
                    if not bool(valid_queries[graph_idx, head, query].detach().cpu()):
                        continue
                    query_rows.append(
                        {
                            **common,
                            "query_index": query,
                            "score": float(
                                weighted_query[graph_idx, head, query].detach().cpu()
                            ),
                            "moved_mass_mean": float(
                                moved[:, graph_idx, head, query].mean().detach().cpu()
                            ),
                            "alpha_max": float(
                                alpha_max[graph_idx, head, query].detach().cpu()
                            ),
                            "effective_perms": float(
                                effective_perms[graph_idx, head, query].detach().cpu()
                            ),
                        }
                    )
    return rows, query_rows


@torch.no_grad()
def compute_permutation_metrics(
    model: nn.Module,
    dataset: Dataset[ECGraph],
    model_name: str,
    phase: str,
    split: str,
    cfg: EC5FastConfig,
    device: torch.device,
    use_amp: bool,
) -> PermutationMetricOutputs:
    selected = Subset(
        dataset,
        subset_indices(len(dataset), cfg.metric_graphs, cfg.seed + 10_003),
    )
    loader = DataLoader(
        selected,
        batch_size=cfg.metric_batch_size,
        shuffle=False,
        collate_fn=collate_ec_graphs,
        num_workers=0,
    )
    model.eval()
    metric_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 20_003)
    base_meta = {
        "model": model_name,
        "phase": phase,
        "split": split,
        "permutation_family": "global_node",
    }

    for batch_idx, batch in enumerate(loader):
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            _pred, clean_layers = model(batch, collect_attention=True)
        by_layer = {
            int(clean["layer"]): {
                "clean": clean,
                "score_lists": {
                    M_POSITIONAL: [],
                    M_SYMBOLIC: [],
                    M_PE_INVARIANT: [],
                    M_PE_EQUIVARIANT: [],
                    M_JOINT_EQUIVARIANT: [],
                    M_POSITIONAL_CENTERED: [],
                    M_SYMBOLIC_CENTERED: [],
                    M_PE_INVARIANT_CENTERED: [],
                    M_PE_EQUIVARIANT_CENTERED: [],
                    M_JOINT_EQUIVARIANT_CENTERED: [],
                    M_INTERACTION_RESIDUAL_CENTERED: [],
                    M_JOINT_EQUIVARIANCE_EXCESS_CENTERED: [],
                },
                "moved_masses": [],
            }
            for clean in clean_layers
        }
        for clean in clean_layers:
            metric_rows.extend(attention_basic_rows(clean, base_meta, batch_idx))

        for _perm_idx in range(cfg.num_metric_perms):
            perm_pos = make_node_permutation(batch, generator)
            x_batch = make_node_type_permutation(batch, perm_pos)
            pe_batch = make_structural_permutation(batch, perm_pos)
            both_batch = make_joint_permutation(batch, perm_pos)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                _pred_x, x_layers = model(x_batch, collect_attention=True)
                _pred_pe, pe_layers = model(pe_batch, collect_attention=True)
                _pred_both, both_layers = model(both_batch, collect_attention=True)
            for clean, x_layer, pe_layer, both_layer in zip(
                clean_layers,
                x_layers,
                pe_layers,
                both_layers,
            ):
                layer_idx = int(clean["layer"])
                layer_state = by_layer[layer_idx]
                a_clean = clean["attn"].float()
                a_x = x_layer["attn"].float()
                a_pe = pe_layer["attn"].float()
                a_both = both_layer["attn"].float()
                clean_mask = pair_mask_from_layer(clean).to(device=a_clean.device)
                attn_perm = perm_for_attention(a_clean, perm_pos)
                a_ref = transform_pair_reference(a_clean, attn_perm)
                ref_mask = transform_pair_reference(clean_mask, attn_perm).to(dtype=torch.bool)
                layer_state["moved_masses"].append(
                    attention_moved_mass_by_query(a_clean, attn_perm, clean_mask)
                )

                score_lists = layer_state["score_lists"]
                score_lists[M_POSITIONAL].append(
                    cosine_by_query_tensor(a_x, a_clean, clean_mask)
                )
                score_lists[M_SYMBOLIC].append(cosine_by_query_tensor(a_x, a_ref, ref_mask))
                score_lists[M_PE_INVARIANT].append(
                    cosine_by_query_tensor(a_pe, a_clean, clean_mask)
                )
                score_lists[M_PE_EQUIVARIANT].append(
                    cosine_by_query_tensor(a_pe, a_ref, ref_mask)
                )
                score_lists[M_JOINT_EQUIVARIANT].append(
                    cosine_by_query_tensor(a_both, a_ref, ref_mask)
                )

                clean_c = row_center_tensor(a_clean, clean_mask)
                ref_c = transform_pair_reference(clean_c, attn_perm)
                ref_c_mask = transform_pair_reference(clean_mask, attn_perm).to(dtype=torch.bool)
                x_c = row_center_tensor(a_x, clean_mask)
                pe_c = row_center_tensor(a_pe, clean_mask)
                both_c = row_center_tensor(a_both, clean_mask)
                positional_c = raw_cosine_by_query_tensor(x_c, clean_c, clean_mask)
                symbolic_c = raw_cosine_by_query_tensor(x_c, ref_c, ref_c_mask)
                pe_invariant_c = raw_cosine_by_query_tensor(pe_c, clean_c, clean_mask)
                pe_equivariant_c = raw_cosine_by_query_tensor(pe_c, ref_c, ref_c_mask)
                joint_equivariant_c = raw_cosine_by_query_tensor(both_c, ref_c, ref_c_mask)
                score_lists[M_POSITIONAL_CENTERED].append(positional_c)
                score_lists[M_SYMBOLIC_CENTERED].append(symbolic_c)
                score_lists[M_PE_INVARIANT_CENTERED].append(pe_invariant_c)
                score_lists[M_PE_EQUIVARIANT_CENTERED].append(pe_equivariant_c)
                score_lists[M_JOINT_EQUIVARIANT_CENTERED].append(joint_equivariant_c)
                score_lists[M_INTERACTION_RESIDUAL_CENTERED].append(
                    interaction_residual_norm_by_query_tensor(
                        clean_c,
                        x_c,
                        pe_c,
                        both_c,
                        clean_mask,
                    )
                )
                best_single_c = torch.stack(
                    [positional_c, symbolic_c, pe_invariant_c, pe_equivariant_c],
                    dim=0,
                ).max(dim=0).values
                score_lists[M_JOINT_EQUIVARIANCE_EXCESS_CENTERED].append(
                    joint_equivariant_c - best_single_c
                )

        for layer_idx, layer_state in sorted(by_layer.items()):
            layer_rows, layer_query_rows = aggregate_alpha_metric_rows(
                layer_state["score_lists"],
                layer_state["moved_masses"],
                layer_state["clean"]["node_mask"],
                base_meta,
                batch_idx,
                layer_idx,
                cfg.metric_alpha_tau,
            )
            metric_rows.extend(layer_rows)
            query_rows.extend(layer_query_rows)
    return PermutationMetricOutputs(metric_rows=metric_rows, query_rows=query_rows)


def summarize_metric_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, str, str, int, int], list[float]] = {}
    for row in rows:
        key = (
            str(row["model"]),
            str(row["phase"]),
            str(row["split"]),
            str(row["metric"]),
            int(row["layer"]),
            int(row["head"]),
        )
        buckets.setdefault(key, []).append(float(row["score"]))
    summary = []
    for (model, phase, split, metric, layer, head), values in sorted(buckets.items()):
        mean = sum(values) / max(1, len(values))
        var = sum((value - mean) ** 2 for value in values) / max(1, len(values))
        summary.append(
            {
                "model": model,
                "phase": phase,
                "split": split,
                "metric": metric,
                "layer": layer,
                "head": head,
                "score_mean": mean,
                "score_std": math.sqrt(var),
                "n": len(values),
            }
        )
    return summary


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_curves(metrics_path: Path, output_path: Path, log=print) -> None:
    rows = read_csv_rows(metrics_path)
    if not rows:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping training curves ({type(exc).__name__}: {exc})")
        return
    epochs = [int(row["epoch"]) for row in rows]
    train_mse = [float(row["train_mse"]) for row in rows]
    val_mse = [float(row["sampled_val_mse"]) for row in rows]
    val_rse = [float(row["sampled_val_rse"]) for row in rows]
    baseline_rse = [float(row["sampled_val_train_mean_baseline_rse"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    axes[0].plot(epochs, train_mse, label="train MSE", color="#386cb0")
    axes[0].plot(epochs, val_mse, label="sampled val MSE", color="#fdb462")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Loss")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, val_rse, label="sampled val RSE", color="#7fc97f")
    axes[1].plot(epochs, baseline_rse, label="train-mean baseline", color="#666666")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("RSE")
    axes[1].set_title("GraphBench metric")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_final_metrics(summary: Mapping[str, object], output_path: Path, log=print) -> None:
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping final metrics ({type(exc).__name__}: {exc})")
        return
    val = summary["full_val_metrics"]
    test = summary["full_test_metrics"]
    metric_names = ["rse", "mse", "mae"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    for ax, metric in zip(axes, metric_names):
        values = [float(val[metric]), float(test[metric])]
        ax.bar(["full val", "full test"], values, color=["#80b1d3", "#fb8072"])
        if metric == "rse":
            ax.axhline(
                float(test["train_mean_baseline_rse"]),
                color="#333333",
                linestyle="--",
                linewidth=1.0,
                label="test baseline",
            )
            ax.legend(frameon=False)
        ax.set_title(metric.upper())
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_diagnostics(
    pred: torch.Tensor,
    target: torch.Tensor,
    output_path: Path,
    log=print,
) -> None:
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping predictions ({type(exc).__name__}: {exc})")
        return
    pred = pred.float().flatten()
    target = target.float().flatten()
    if pred.numel() > 5_000:
        idx = torch.linspace(0, pred.numel() - 1, 5_000).long()
        pred_plot = pred[idx]
        target_plot = target[idx]
    else:
        pred_plot = pred
        target_plot = target
    residual = pred - target

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].scatter(target_plot.numpy(), pred_plot.numpy(), s=8, alpha=0.35, color="#4daf4a")
    lo = float(min(target_plot.min(), pred_plot.min()))
    hi = float(max(target_plot.max(), pred_plot.max()))
    axes[0].plot([lo, hi], [lo, hi], color="#333333", linewidth=1.0)
    axes[0].set_xlabel("target")
    axes[0].set_ylabel("prediction")
    axes[0].set_title("Full-test predictions")
    axes[0].grid(alpha=0.25)

    axes[1].hist(residual.numpy(), bins=60, color="#984ea3", alpha=0.8)
    axes[1].axvline(0.0, color="#333333", linewidth=1.0)
    axes[1].set_xlabel("prediction - target")
    axes[1].set_ylabel("count")
    axes[1].set_title("Full-test residuals")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_plane(
    summary_rows: Sequence[Mapping[str, object]],
    output_path: Path,
    phase: str,
    log=print,
) -> None:
    rows = [row for row in summary_rows if str(row["phase"]) == phase]
    if not rows:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping permutation plane ({type(exc).__name__}: {exc})")
        return

    grouped: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        key = (int(row["layer"]), int(row["head"]))
        grouped.setdefault(key, {})[str(row["metric"])] = float(row["score_mean"])

    fig, axes_grid = plt.subplots(2, 2, figsize=(11.2, 8.8))
    axes = list(axes_grid.flatten())
    planes = [
        (M_POSITIONAL, M_SYMBOLIC, "node-feature permutation"),
        (M_PE_INVARIANT, M_PE_EQUIVARIANT, "structural-feature permutation"),
        (M_POSITIONAL_CENTERED, M_SYMBOLIC_CENTERED, "node-feature centered"),
        (M_PE_INVARIANT_CENTERED, M_PE_EQUIVARIANT_CENTERED, "structural-feature centered"),
    ]
    colors = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
    for ax, (x_metric, y_metric, title) in zip(axes, planes):
        for (layer, head), metrics in sorted(grouped.items()):
            if x_metric not in metrics or y_metric not in metrics:
                continue
            ax.scatter(
                metrics[x_metric],
                metrics[y_metric],
                color=colors[layer % len(colors)],
                s=42,
                alpha=0.85,
            )
            ax.text(
                metrics[x_metric] + 0.01,
                metrics[y_metric] + 0.01,
                f"L{layer}H{head}",
                fontsize=7,
            )
        ax.axhline(0.0, color="#dddddd", linewidth=0.8)
        ax.axvline(0.0, color="#dddddd", linewidth=0.8)
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(f"{phase}: {title}")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_suite_summary(
    suite_dir: Path,
    summaries: Sequence[Mapping[str, object]],
    log=print,
) -> None:
    if not summaries:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping suite summary ({type(exc).__name__}: {exc})")
        return
    models = [str(summary["model"]) for summary in summaries]
    test_rse = [float(summary["full_test_metrics"]["rse"]) for summary in summaries]
    baseline = [
        float(summary["full_test_metrics"]["train_mean_baseline_rse"]) for summary in summaries
    ]
    params = [float(summary["trainable_parameters"]) / 1_000.0 for summary in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].bar(models, test_rse, color="#8dd3c7", label="model")
    axes[0].plot(models, baseline, color="#333333", marker="o", linestyle="--", label="baseline")
    axes[0].set_ylabel("full-test RSE")
    axes[0].set_title("EC5-vout performance")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(models, params, color="#bebada")
    axes[1].set_ylabel("trainable params (K)")
    axes[1].set_title("Model size")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(suite_dir / "suite_model_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_zero_shot_metrics(summary: Mapping[str, object], output_path: Path, log=print) -> None:
    zero_shot = summary.get("zero_shot_metrics", {})
    if not isinstance(zero_shot, Mapping) or not zero_shot:
        return
    try:
        plt = import_plotting()
    except Exception as exc:
        log(f"[plot] Skipping zero-shot metrics ({type(exc).__name__}: {exc})")
        return
    labels = ["EC5 test"]
    rse = [float(summary["full_test_metrics"]["rse"])]
    baseline = [float(summary["full_test_metrics"]["train_mean_baseline_rse"])]
    for name, metrics in sorted(zero_shot.items()):
        labels.append(str(name).replace("electronic_circuits_", "EC").replace("_vout", ""))
        test = metrics["test_metrics"]
        rse.append(float(test["rse"]))
        baseline.append(float(test["train_mean_baseline_rse"]))

    fig, ax = plt.subplots(figsize=(max(6.5, 1.5 * len(labels)), 4.0))
    ax.bar(labels, rse, color="#8dd3c7", label="model")
    ax.plot(
        labels,
        baseline,
        color="#333333",
        marker="o",
        linestyle="--",
        label="train-mean baseline",
    )
    ax.set_ylabel("RSE")
    ax.set_title("EC size generalization")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_eval_loader(
    dataset: Dataset[ECGraph],
    cfg: EC5FastConfig,
    seed: int,
) -> DataLoader:
    eval_set: Dataset[ECGraph]
    if cfg.final_eval_size > 0:
        eval_set = Subset(dataset, subset_indices(len(dataset), cfg.final_eval_size, seed))
    else:
        eval_set = dataset
    return DataLoader(
        eval_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_ec_graphs,
        num_workers=0,
    )


def train_model(
    model_name: str,
    splits: Mapping[str, ECGraphDataset],
    cfg: EC5FastConfig,
    output_root: Path,
    device: torch.device,
    zero_shot_splits: Optional[Mapping[str, Mapping[str, ECGraphDataset]]] = None,
    skip_size_check: bool = False,
    force_retrain: bool = False,
    force_metrics: bool = False,
    skip_permutation_metrics: bool = False,
) -> dict[str, object]:
    set_seed(cfg.seed)
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
    full_val_loader = make_eval_loader(splits["val"], cfg, cfg.seed + 31)
    full_test_loader = make_eval_loader(splits["test"], cfg, cfg.seed + 47)

    model = build_model(model_name, cfg).to(device)
    n_params = count_parameters(model)
    assert_size_band(model_name, n_params, skip=skip_size_check)
    log(f"[model] trainable_parameters={n_params:,}")
    best_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"

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
    existing_complete = best_path.exists() and summary_path.exists()
    skip_training = existing_complete and not force_retrain

    if not skip_permutation_metrics:
        initial_summary_path = run_dir / "alpha_permutation_summary_initial.csv"
        if force_metrics or force_retrain or not initial_summary_path.exists():
            log("[metrics] computing initial alpha-weighted permutation metrics")
            metric_outputs = compute_permutation_metrics(
                model,
                sampled_val_set,
                model_name=model_name,
                phase="initial",
                split="sampled_val",
                cfg=cfg,
                device=device,
                use_amp=use_amp,
            )
            summary_rows = summarize_metric_rows(metric_outputs.metric_rows)
            write_csv_rows(
                run_dir / "alpha_permutation_metrics_initial.csv",
                metric_outputs.metric_rows,
            )
            write_csv_rows(
                run_dir / "alpha_permutation_query_metrics_initial.csv",
                metric_outputs.query_rows,
            )
            write_csv_rows(initial_summary_path, summary_rows)
            plot_permutation_plane(
                summary_rows,
                run_dir / "alpha_permutation_plane_initial.png",
                phase="initial",
                log=log,
            )
        else:
            log("[metrics] initial permutation metrics already exist; skipping")

    if skip_training:
        log(f"[resume] Found existing checkpoint and summary at {run_dir}; skipping training")
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        best_epoch = int(checkpoint.get("epoch", 0))
        best_rse = float(checkpoint.get("sampled_val_metrics", {}).get("rse", float("nan")))
    else:
        if force_retrain and (best_path.exists() or summary_path.exists()):
            log("[resume] --force-retrain set; overwriting existing training outputs")
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
                val_metrics = evaluate(
                    model,
                    sampled_val_loader,
                    device,
                    train_mean,
                    use_amp=use_amp,
                )
                row = {
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_mse": train_metrics["mse"],
                    "train_mae": train_metrics["mae"],
                    "train_rse": train_metrics["rse"],
                    "sampled_val_mse": val_metrics["mse"],
                    "sampled_val_mae": val_metrics["mae"],
                    "sampled_val_rse": val_metrics["rse"],
                    "sampled_val_train_mean_baseline_rse": val_metrics[
                        "train_mean_baseline_rse"
                    ],
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
    full_val_pred, full_val_target = predict_all(model, full_val_loader, device, use_amp=use_amp)
    full_test_pred, full_test_target = predict_all(model, full_test_loader, device, use_amp=use_amp)
    full_val_metrics = metric_values(full_val_pred, full_val_target, baseline_mean=train_mean)
    full_test_metrics = metric_values(full_test_pred, full_test_target, baseline_mean=train_mean)
    zero_shot_metrics: dict[str, object] = {}
    if zero_shot_splits:
        for dataset_name, split_map in zero_shot_splits.items():
            z_val_loader = make_eval_loader(split_map["val"], cfg, cfg.seed + 503)
            z_test_loader = make_eval_loader(split_map["test"], cfg, cfg.seed + 509)
            z_val_pred, z_val_target = predict_all(model, z_val_loader, device, use_amp=use_amp)
            z_test_pred, z_test_target = predict_all(model, z_test_loader, device, use_amp=use_amp)
            z_val_metrics = metric_values(z_val_pred, z_val_target, baseline_mean=train_mean)
            z_test_metrics = metric_values(z_test_pred, z_test_target, baseline_mean=train_mean)
            zero_shot_metrics[dataset_name] = {
                "val_metrics": z_val_metrics,
                "test_metrics": z_test_metrics,
                "full_val": dataset_stats(split_map["val"]),
                "full_test": dataset_stats(split_map["test"]),
            }
            safe_name = dataset_name.replace("/", "_")
            plot_prediction_diagnostics(
                z_test_pred,
                z_test_target,
                run_dir / f"zero_shot_{safe_name}_prediction_diagnostics.png",
                log=log,
            )
            log(
                f"[zero-shot] {dataset_name} test_rse={z_test_metrics['rse']:.6f} "
                f"baseline={z_test_metrics['train_mean_baseline_rse']:.6f}"
            )
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
        "zero_shot_metrics": zero_shot_metrics,
        "pilot_acceptance": {
            "beats_train_mean_baseline": accepted,
            "status": "passed" if accepted else "failed",
        },
        "best_checkpoint": str(best_path),
    }
    write_json(run_dir / "summary.json", summary)
    plot_training_curves(metrics_path, run_dir / "training_curves.png", log=log)
    plot_final_metrics(summary, run_dir / "final_metrics.png", log=log)
    plot_prediction_diagnostics(
        full_test_pred,
        full_test_target,
        run_dir / "prediction_diagnostics.png",
        log=log,
    )
    if not skip_permutation_metrics:
        best_summary_path = run_dir / "alpha_permutation_summary_best.csv"
        if force_metrics or force_retrain or not best_summary_path.exists():
            log("[metrics] computing best-checkpoint alpha-weighted permutation metrics")
            metric_outputs = compute_permutation_metrics(
                model,
                sampled_val_set,
                model_name=model_name,
                phase="best",
                split="sampled_val",
                cfg=cfg,
                device=device,
                use_amp=use_amp,
            )
            summary_rows = summarize_metric_rows(metric_outputs.metric_rows)
            write_csv_rows(
                run_dir / "alpha_permutation_metrics_best.csv",
                metric_outputs.metric_rows,
            )
            write_csv_rows(
                run_dir / "alpha_permutation_query_metrics_best.csv",
                metric_outputs.query_rows,
            )
            write_csv_rows(best_summary_path, summary_rows)
            plot_permutation_plane(
                summary_rows,
                run_dir / "alpha_permutation_plane_best.png",
                phase="best",
                log=log,
            )
        else:
            log("[metrics] best permutation metrics already exist; skipping")
    log(
        f"[done] full_test_rse={full_test_metrics['rse']:.6f} "
        f"baseline={full_test_metrics['train_mean_baseline_rse']:.6f} "
        f"accepted={accepted}"
    )
    plot_zero_shot_metrics(summary, run_dir / "zero_shot_metrics.png", log=log)
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
    parser.add_argument("--metric-graphs", type=int, default=512)
    parser.add_argument("--metric-batch-size", type=int, default=128)
    parser.add_argument("--num-metric-perms", type=int, default=16)
    parser.add_argument(
        "--metric-alpha-tau",
        type=float,
        default=0.1,
        help="Softmax temperature for alpha-weighting sampled permutations.",
    )
    parser.add_argument(
        "--zero-shot-components",
        type=str,
        default="7,10",
        help="Comma-separated EC component counts for zero-shot eval after EC5 training.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument(
        "--drive-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/graph_specialisation_metrics/graphbench_ec5"),
        help="Colab Drive root for datasets, logs, checkpoints, and metrics.",
    )
    parser.add_argument("--drive-mount", type=Path, default=Path("/content/drive"))
    parser.add_argument("--no-mount-drive", action="store_true")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override dataset cache root. Defaults to <drive-dir>/datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output root. Defaults to <drive-dir>/results.",
    )
    parser.add_argument("--source", choices=["auto", "graphbench", "json"], default="auto")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "Allow CPU/MPS fallback for local smoke tests. "
            "Colab training requires CUDA by default."
        ),
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-size-check", action="store_true")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if best.pt and summary.json already exist.",
    )
    parser.add_argument(
        "--force-metrics",
        action="store_true",
        help="Recompute permutation metrics and diagnostic plots even if present.",
    )
    parser.add_argument("--skip-permutation-metrics", action="store_true")
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Override to a 1-epoch 1024/256 sample run for integration checks.",
    )
    return parser.parse_args(argv)


def strip_colab_kernel_args(argv: Sequence[str]) -> list[str]:
    """Drop IPython/Colab launcher args injected into sys.argv.

    Running a script through `%run` or by executing a pasted cell can leave
    arguments like `-f /root/.local/share/jupyter/runtime/kernel-....json` in
    `sys.argv`. Those belong to the notebook kernel, not this training CLI.
    """
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


def resolve_device(requested: str, allow_cpu: bool = False) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu:
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        raise RuntimeError(
            "CUDA GPU is not available. In Colab, select Runtime > Change runtime type "
            "> GPU before running this script. For local smoke tests only, pass --allow-cpu."
        )

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but CUDA is not available. In Colab, select "
                "Runtime > Change runtime type > GPU before running this script."
            )
        return device

    if not allow_cpu:
        raise RuntimeError(
            f"Refusing to train on {device}. This Colab runner requires CUDA by default; "
            "pass --allow-cpu only for local smoke tests."
        )
    return device


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return str(device)
    index = 0 if device.index is None else device.index
    name = torch.cuda.get_device_name(index)
    props = torch.cuda.get_device_properties(index)
    total_gb = props.total_memory / (1024**3)
    return f"cuda:{index} ({name}, {total_gb:.1f} GiB)"


def parse_component_list(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    components: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        value = int(item)
        if value == TRAIN_COMPONENTS:
            continue
        if value not in {7, 10}:
            raise ValueError("--zero-shot-components currently supports only 7 and 10")
        if value not in components:
            components.append(value)
    return tuple(components)


def config_from_args(args: argparse.Namespace) -> EC5FastConfig:
    train_size = args.train_size
    val_size = args.val_size
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    patience = args.patience
    final_eval_size = args.final_eval_size
    metric_graphs = args.metric_graphs
    metric_batch_size = args.metric_batch_size
    num_metric_perms = args.num_metric_perms
    zero_shot_components = parse_component_list(args.zero_shot_components)
    if args.metric_alpha_tau <= 0:
        raise ValueError("--metric-alpha-tau must be positive")
    if args.fast_dev_run:
        train_size = min(train_size, 1024)
        val_size = min(val_size, 256)
        batch_size = min(batch_size, 128)
        max_epochs = 1
        patience = 1
        final_eval_size = final_eval_size or 512
        metric_graphs = min(metric_graphs, 128)
        metric_batch_size = min(metric_batch_size, 64)
        num_metric_perms = min(num_metric_perms, 1)
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
        metric_graphs=metric_graphs,
        metric_batch_size=metric_batch_size,
        num_metric_perms=num_metric_perms,
        metric_alpha_tau=args.metric_alpha_tau,
        zero_shot_components=zero_shot_components,
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
    device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    mount_drive(args.drive_mount, enabled=not args.no_mount_drive)
    dataset_root = args.dataset_root or args.drive_dir / "datasets"
    output_root = args.output_root or args.drive_dir / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    top_log = RunLogger(output_root / cfg.preset / "run.log")
    top_log(f"[setup] device={describe_device(device)} amp={cfg.amp and device.type == 'cuda'}")
    top_log(f"[setup] source={args.source} dataset_root={dataset_root}")
    top_log(f"[setup] output_root={output_root}")
    splits = load_splits(args.source, dataset_root, components=TRAIN_COMPONENTS, log=top_log)
    zero_shot_splits: dict[str, dict[str, ECGraphDataset]] = {}
    for components in cfg.zero_shot_components:
        dataset_name = ec_dataset_name(components)
        top_log(f"[setup] loading zero-shot evaluation dataset {dataset_name}")
        zero_shot_splits[dataset_name] = load_splits(
            args.source,
            dataset_root,
            components=components,
            log=top_log,
        )
    models = ["graphormer", "graphgps", "grit"] if args.model == "all" else [args.model]
    summaries = []
    for model_name in models:
        summaries.append(
            train_model(
                model_name,
                splits,
                cfg,
                output_root,
                device,
                zero_shot_splits=zero_shot_splits,
                skip_size_check=args.skip_size_check,
                force_retrain=args.force_retrain,
                force_metrics=args.force_metrics,
                skip_permutation_metrics=args.skip_permutation_metrics,
            )
        )
    write_json(output_root / cfg.preset / "summary.json", {"runs": summaries})
    plot_suite_summary(output_root / cfg.preset, summaries, log=top_log)


if __name__ == "__main__":
    main(sys.argv[1:])
