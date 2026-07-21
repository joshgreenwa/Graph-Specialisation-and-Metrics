"""Paper-aligned Neighbor Associative Recall with trained-support official GRIT.

The task follows Section 3.2 of Blayney et al. (2026): a fixed-N experiment has
N key-value neighbours, one central node, one intermediate node, and one query
node.  Every key appears exactly once, values are sampled with replacement from
an N-value vocabulary, and the central node predicts the value named by the
query after exactly two layers.  Each checkpoint is trained for one N only.

The sole intended architectural change is the model family: parameter-matched
official GRITs are trained with 1-hop, 2-hop, or dense attention support.  The
core task contains no structural gadgets, auxiliary target, task marker, or
cross-N curriculum.  Mechanistic analysis is correspondingly semantic-only:
value transport scores, clean head ablations, target attention, semantic
carriage, and score-selected family ablations.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPERIMENT_VERSION = "nar-grit-fixed-n-v3"
OFFICIAL_GRIT_URL = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
DEFAULT_GRIT_DIR = "/content/GRIT"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/graph_specialisation_metrics/nar_grit"
# Batched sparse/scatter reductions on CUDA are not bitwise deterministic across two otherwise
# identical graph replicas. Values below this scale are numerical round-off, not routed signal.
NOOP_TRANSPORT_ATOL = 2.0e-5
MODEL_RADII: dict[str, int | None] = {"1hop": 1, "2hop": 2, "dense": None}
MODEL_ORDER = ("1hop", "2hop", "dense")
MODEL_COLOURS = {"1hop": "#6550a4", "2hop": "#2b8cbe", "dense": "#d7301f"}
MODEL_MARKERS = {"1hop": "o", "2hop": "s", "dense": "D"}
EPS = 1.0e-12

TASK_ALIGNMENT = {
    "graph": "N key-value neighbours plus central, intermediate, and query nodes (N+3 total)",
    "keys": "fixed N-key vocabulary; every key occurs exactly once in every graph",
    "values": "fixed N-value vocabulary; neighbour values sampled independently with replacement",
    "features": "learned key and value embeddings; central/intermediate inputs are zero",
    "query": "queried key embedding concatenated with a zero value embedding",
    "target": "N-way value classification at the central node",
    "training_unit": "one independently trained checkpoint for each fixed N",
    "depth": "exactly two layers",
}

INTENTIONAL_DIFFERENCE = {
    "models": "official GRIT with trained 1-hop, 2-hop, or dense support instead of GCN/gLSTM",
    "positional_encoding": "official GRIT RRWP node/pair encodings are retained",
    "analysis": "semantic intervention, ablation, attention, and carriage diagnostics are added",
    "lightweight_protocol": (
        "fresh online training graphs, 192 validation graphs, 512 held-out graphs, "
        "N up to 64, and widths up to 128; the authors use fixed 8000/1000/1000 splits, "
        "also sweep N=80,96 and width 256"
    ),
}


def _run(command: Sequence[str], *, check: bool = True) -> int:
    print(f"[cmd] {' '.join(map(str, command))}", flush=True)
    return subprocess.run(list(map(str, command)), check=check).returncode


def install_pyg_stack() -> None:
    import torch

    torch_version = str(torch.__version__).split("+")[0]
    cuda = getattr(torch.version, "cuda", None)
    cuda_tag = ("cu" + cuda.replace(".", "")) if cuda else "cpu"
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    print(f"[deps] torch={torch.__version__} cuda={cuda} | {wheel_url}", flush=True)
    _run(
        [sys.executable, "-m", "pip", "install", "-q", "torch_geometric", "yacs", "ogb", "einops", "opt_einsum"],
        check=False,
    )
    for package in ("pyg-lib", "torch-spline-conv", "torch-cluster"):
        _run([sys.executable, "-m", "pip", "install", "-q", package, "-f", wheel_url], check=False)
    for package in ("torch-scatter", "torch-sparse"):
        if _run([sys.executable, "-m", "pip", "install", "-q", package, "-f", wheel_url], check=False):
            raise SystemExit(f"Required PyG extension {package!r} has no wheel at {wheel_url}")


def setup_official_grit(grit_dir: Path, *, install: bool) -> None:
    if not (grit_dir / ".git").exists():
        if grit_dir.exists():
            shutil.rmtree(grit_dir)
        _run(["git", "clone", OFFICIAL_GRIT_URL, str(grit_dir)])
    _run(["git", "-C", str(grit_dir), "checkout", OFFICIAL_GRIT_COMMIT])
    if install:
        install_pyg_stack()
        _run([sys.executable, "-m", "pip", "install", "-q", "-e", str(grit_dir), "--no-deps"], check=False)
    os.environ["GRIT_ROOT"] = str(grit_dir)
    if str(grit_dir) not in sys.path:
        sys.path.insert(0, str(grit_dir))
    print(f"[grit] official GRIT @ {OFFICIAL_GRIT_COMMIT}", flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    raise TypeError(f"cannot JSON-encode {type(value)!r}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class Config:
    run_name: str = "nar_grit_fixed_n_v3"
    drive_root: str = DEFAULT_DRIVE_ROOT
    converged_legacy_run: str = "nar_grit_fixed_n_v2"
    ns: tuple[int, ...] = (4, 8, 16, 32, 64)
    mechanistic_ns: tuple[int, ...] = (4, 16, 64)
    widths: tuple[int, ...] = (64, 128)
    analysis_width: int = 128
    heads: int = 8
    layers: int = 2
    models: tuple[str, ...] = MODEL_ORDER
    rrwp_steps: int = 6
    dropout: float = 0.0
    attention_dropout: float = 0.0
    batch_size: int = 64
    max_batch_nodes: int = 4500
    max_dense_pairs: int = 300_000
    steps: int = 10_000
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    # The reference NAR setup has 8,000 training examples and batch size 64,
    # hence 125 optimiser steps per epoch.  Evaluate at the same cadence.
    eval_every: int = 125
    validation_graphs: int = 192
    heldout_graphs: int = 512
    early_stopping_loss_threshold: float = 0.001
    low_n_accuracy_gate: float = 0.85
    score_graphs: int = 8
    score_donors: int = 3
    analysis_graphs: int = 12
    ablation_graphs: int = 128
    family_size: int = 2
    random_families: int = 8
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "cuda"

    def nodes_for_n(self, records: int) -> int:
        return int(records) + 3

    def feature_dim_for_n(self, records: int) -> int:
        del records
        return 2

    def validate(self) -> None:
        if self.layers != 2:
            raise ValueError("paper-aligned NAR requires exactly two layers")
        if not self.ns or any(value <= 1 for value in self.ns):
            raise ValueError("N values must be integers greater than one")
        if tuple(sorted(set(self.ns))) != tuple(self.ns):
            raise ValueError("ns must be sorted and unique")
        if any(value not in self.ns for value in self.mechanistic_ns):
            raise ValueError("mechanistic_ns must be a subset of ns")
        if self.analysis_width not in self.widths:
            raise ValueError("analysis_width must be one of widths")
        if any(width % 2 or width % self.heads for width in self.widths):
            raise ValueError("every width must be even and divisible by heads")
        if any(model not in MODEL_RADII for model in self.models):
            raise ValueError(f"models must be drawn from {sorted(MODEL_RADII)}")
        if "1hop" not in self.models or "dense" not in self.models:
            raise ValueError("the core comparison requires 1hop and dense")
        if not 0 < 2 * self.family_size <= self.layers * self.heads:
            raise ValueError("family_size must leave an equal-size non-top control pool")


def config_fingerprint(cfg: Config) -> str:
    payload = {"version": EXPERIMENT_VERSION, **asdict(cfg)}
    payload.pop("drive_root", None)
    payload.pop("run_name", None)
    payload.pop("converged_legacy_run", None)
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class NarBatch:
    x: Any
    adj: Any
    rrwp: Any
    central_idx: Any
    intermediate_idx: Any
    query_idx: Any
    target_idx: Any
    record_mask: Any
    y: Any
    n_records: Any

    def __len__(self) -> int:
        return int(self.x.size(0))

    def to(self, device: Any) -> "NarBatch":
        return NarBatch(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})

    def slice(self, start: int, stop: int) -> "NarBatch":
        return NarBatch(**{name: getattr(self, name)[start:stop] for name in self.__dataclass_fields__})


def concat_batches(batches: Sequence[NarBatch]) -> NarBatch:
    import torch

    if not batches:
        raise ValueError("cannot concatenate an empty batch list")
    if len({int(batch.x.size(1)) for batch in batches}) != 1:
        raise ValueError("NarBatch concatenation requires a common fixed N")
    return NarBatch(**{
        name: torch.cat([getattr(batch, name) for batch in batches], dim=0)
        for name in NarBatch.__dataclass_fields__
    })


def clone_batch(batch: NarBatch) -> NarBatch:
    return NarBatch(**{name: getattr(batch, name).clone() for name in NarBatch.__dataclass_fields__})


def add_undirected(adj: np.ndarray, left: int, right: int) -> None:
    adj[int(left), int(right)] = 1.0
    adj[int(right), int(left)] = 1.0


def rrwp_from_adj(adj: np.ndarray, steps: int) -> np.ndarray:
    from scipy.sparse import csr_matrix

    degree = adj.sum(axis=1, keepdims=True)
    transition = adj / np.maximum(degree, 1.0)
    sparse = csr_matrix(transition)
    nodes = int(adj.shape[0])
    output = np.zeros((nodes, nodes, int(steps)), dtype=np.float32)
    power = np.eye(nodes, dtype=np.float32)
    for step in range(int(steps)):
        if step:
            power = np.asarray(sparse.T.dot(power.T).T, dtype=np.float32)
        output[:, :, step] = power
    return output


@lru_cache(maxsize=None)
def fixed_n_topology(records: int, rrwp_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the immutable NAR adjacency and RRWP tensors shared by every fixed-N graph."""
    records = int(records)
    nodes = records + 3
    adjacency = np.zeros((nodes, nodes), dtype=np.float32)
    add_undirected(adjacency, 0, 1)
    add_undirected(adjacency, 1, 2)
    for slot in range(records):
        add_undirected(adjacency, 0, 3 + slot)
    adjacency.setflags(write=False)
    rrwp = rrwp_from_adj(adjacency, int(rrwp_steps))
    rrwp.setflags(write=False)
    return adjacency, rrwp


def make_batch(cfg: Config, size: int, records: int, seed: int) -> NarBatch:
    """Generate the fixed-N classification task from the NAR paper."""
    import torch

    records = int(records)
    if records <= 1:
        raise ValueError("records must be greater than one")
    rng = np.random.default_rng(int(seed))
    nodes = cfg.nodes_for_n(records)
    # The paper's encoder reserves token N as a null/padding value for missing fields.
    xs = np.full((int(size), nodes, 2), records, dtype=np.int64)
    adjs = np.zeros((int(size), nodes, nodes), dtype=np.float32)
    rrwps = np.zeros((int(size), nodes, nodes, cfg.rrwp_steps), dtype=np.float32)
    central_indices = np.zeros(int(size), dtype=np.int64)
    intermediate_indices = np.zeros(int(size), dtype=np.int64)
    query_indices = np.zeros(int(size), dtype=np.int64)
    target_indices = np.zeros(int(size), dtype=np.int64)
    record_masks = np.zeros((int(size), nodes), dtype=bool)
    labels = np.zeros(int(size), dtype=np.int64)
    n_records = np.full(int(size), records, dtype=np.int64)

    # NAR has one fixed topology for each N. Building it once (as the reference dataset does)
    # removes repeated sparse random-walk construction from the training loop and makes longer,
    # convergence-safe optimization budgets cheap enough for Colab.
    central, intermediate, query = 0, 1, 2
    base_adj, base_rrwp = fixed_n_topology(records, cfg.rrwp_steps)
    base_record_mask = np.zeros(nodes, dtype=bool)
    base_record_mask[3:] = True
    adjs[:] = base_adj
    rrwps[:] = base_rrwp
    central_indices[:] = central
    intermediate_indices[:] = intermediate
    query_indices[:] = query
    record_masks[:] = base_record_mask

    for graph in range(int(size)):
        x = np.full((nodes, 2), records, dtype=np.int64)
        values = rng.integers(0, records, size=records)
        target_slot = int(rng.integers(0, records))
        for slot in range(records):
            node = 3 + slot
            x[node, 0] = slot
            x[node, 1] = int(values[slot])
        x[query, 0] = target_slot
        labels[graph] = int(values[target_slot])
        xs[graph] = x
        target_indices[graph] = 3 + target_slot

    return NarBatch(
        x=torch.from_numpy(xs),
        adj=torch.from_numpy(adjs),
        rrwp=torch.from_numpy(rrwps),
        central_idx=torch.from_numpy(central_indices),
        intermediate_idx=torch.from_numpy(intermediate_indices),
        query_idx=torch.from_numpy(query_indices),
        target_idx=torch.from_numpy(target_indices),
        record_mask=torch.from_numpy(record_masks),
        y=torch.from_numpy(labels),
        n_records=torch.from_numpy(n_records),
    )


def khop_support(adj: Any, radius: int | None) -> Any:
    import torch

    batch, nodes, _ = adj.shape
    if radius is None:
        return torch.ones(batch, nodes, nodes, dtype=torch.bool, device=adj.device)
    identity = torch.eye(nodes, dtype=torch.bool, device=adj.device).expand(batch, -1, -1)
    reach = identity.clone()
    frontier = identity.float()
    adjacency = (adj > 0).float()
    for _ in range(int(radius)):
        frontier = (torch.bmm(frontier, adjacency) > 0).float()
        reach |= frontier.bool()
    return reach


def grit_layer_cfg(update_e: bool) -> Any:
    from yacs.config import CfgNode

    cfg = CfgNode()
    cfg.update_e = bool(update_e)
    cfg.bn_momentum = 0.1
    cfg.bn_no_runner = False
    cfg.rezero = False
    cfg.attn = CfgNode()
    cfg.attn.use = True
    cfg.attn.deg_scaler = True
    cfg.attn.use_bias = False
    cfg.attn.clamp = 5.0
    cfg.attn.act = "relu"
    cfg.attn.edge_enhance = True
    cfg.attn.sqrt_relu = False
    cfg.attn.signed_sqrt = True
    cfg.attn.scaled_attn = False
    cfg.attn.no_qk = False
    cfg.attn.graphormer_attn = False
    cfg.attn.norm_e = True
    cfg.attn.O_e = True
    return cfg


class OfficialFixedNNARGRIT:
    """Late-bound placeholder so generator tests do not need GRIT installed."""


def build_model_class() -> type:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from grit.layer.grit_layer import GritTransformerLayer

    class _OfficialFixedNNARGRIT(nn.Module):
        def __init__(self, cfg: Config, model_name: str, width: int, records: int) -> None:
            super().__init__()
            self.cfg = cfg
            self.model_name = str(model_name)
            self.radius = MODEL_RADII[model_name]
            self.width = int(width)
            self.records = int(records)
            self.L, self.H = cfg.layers, cfg.heads
            self.dh = self.width // self.H
            half = self.width // 2
            self.key_encoder = nn.Embedding(self.records + 1, half, padding_idx=self.records)
            self.value_encoder = nn.Embedding(self.records + 1, half, padding_idx=self.records)
            self.node_rrwp_encoder = nn.Linear(cfg.rrwp_steps, self.width, bias=False)
            self.pair_rrwp_encoder = nn.Linear(cfg.rrwp_steps, self.width, bias=False)
            self.edge_type_encoder = nn.Embedding(3, self.width)
            layer_cfg = grit_layer_cfg(update_e=True)
            self.layers = nn.ModuleList([
                GritTransformerLayer(
                    self.width,
                    self.width,
                    cfg.heads,
                    dropout=cfg.dropout,
                    attn_dropout=cfg.attention_dropout,
                    layer_norm=False,
                    batch_norm=True,
                    residual=True,
                    act="relu",
                    norm_e=True,
                    O_e=True,
                    cfg=layer_cfg,
                )
                for _ in range(cfg.layers)
            ])
            self.output_head = nn.Linear(self.width, self.records)

        @property
        def attention_layers(self) -> list[Any]:
            return [layer.attention for layer in self.layers]

        def _pyg_batch(self, batch: NarBatch) -> Any:
            batch_size, nodes = int(batch.x.size(0)), int(batch.x.size(1))
            if int(batch.x.size(-1)) != 2:
                raise ValueError("checkpoint N and batch N differ")
            if int(batch.n_records[0]) != self.records:
                raise ValueError("checkpoint N and batch N differ")
            encoded = torch.cat(
                [self.key_encoder(batch.x[..., 0]), self.value_encoder(batch.x[..., 1])],
                dim=-1,
            )
            support = khop_support(batch.adj, self.radius)
            graph_idx, source_local, destination_local = support.nonzero(as_tuple=True)
            source = graph_idx * nodes + source_local
            destination = graph_idx * nodes + destination_local
            is_self = source_local == destination_local
            is_graph_edge = batch.adj[graph_idx, source_local, destination_local] > 0
            edge_type = torch.where(is_self, 0, torch.where(is_graph_edge, 1, 2)).long()
            edge_rrwp = batch.rrwp[graph_idx, source_local, destination_local]
            diagonal = torch.arange(nodes, device=batch.x.device)
            node_rrwp = batch.rrwp[:, diagonal, diagonal].reshape(batch_size * nodes, self.cfg.rrwp_steps)
            data = Data(num_nodes=batch_size * nodes)
            data.x = encoded.reshape(batch_size * nodes, self.width) + self.node_rrwp_encoder(node_rrwp)
            data.edge_index = torch.stack([source, destination], dim=0)
            data.edge_attr = self.edge_type_encoder(edge_type) + self.pair_rrwp_encoder(edge_rrwp)
            data.batch = torch.arange(batch_size, device=batch.x.device).repeat_interleave(nodes)
            degree = torch.zeros(batch_size * nodes, device=batch.x.device)
            degree.index_add_(0, destination, torch.ones_like(destination, dtype=torch.float32))
            data.deg = degree
            data.log_deg = torch.log(degree + 1.0)
            data.graph_num_nodes = torch.full((batch_size,), nodes, dtype=torch.long, device=batch.x.device)
            self.last_support_graph = graph_idx
            self.last_support_source_local = source_local
            self.last_support_destination_local = destination_local
            return data

        def forward(self, batch: NarBatch) -> Any:
            data = self._pyg_batch(batch)
            for layer in self.layers:
                data = layer(data)
            states = data.x.reshape(len(batch), batch.x.size(1), self.width)
            rows = torch.arange(len(batch), device=states.device)
            return self.output_head(states[rows, batch.central_idx])

    return _OfficialFixedNNARGRIT


@contextlib.contextmanager
def ablate_heads(model: Any, heads: Sequence[tuple[int, int]]):
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    handles = []

    def make_hook(indices: Sequence[int]):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            node_heads, edge_output = output
            changed = node_heads.clone()
            changed[:, list(indices), :] = 0.0
            return changed, edge_output
        return hook

    try:
        for layer, indices in by_layer.items():
            handles.append(model.attention_layers[layer].register_forward_hook(make_hook(indices)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def capture_forward(model: Any, batch: NarBatch, *, want_grad: bool, want_attention: bool = False) -> dict[str, Any]:
    import torch

    routed: list[Any] = [None] * model.L
    attention: list[Any] = [None] * model.L
    index = {id(module): layer for layer, module in enumerate(model.attention_layers)}

    def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        layer = index[id(module)]
        routed[layer] = output[0]
        if want_attention:
            attention[layer] = inputs[0].attn.detach().squeeze(-1)

    handles = [module.register_forward_hook(hook) for module in model.attention_layers]
    try:
        with (torch.enable_grad() if want_grad else torch.no_grad()):
            logits = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in routed):
        raise RuntimeError("an official GRIT attention hook did not fire")
    return {"logits": logits, "wV": routed, "attention": attention}


def set_seed(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def batch_graphs(cfg: Config, records: int) -> int:
    nodes = cfg.nodes_for_n(records)
    return max(1, min(cfg.batch_size, cfg.max_batch_nodes // nodes, cfg.max_dense_pairs // (nodes * nodes)))


def predict_chunks(
    model: Any,
    batch: NarBatch,
    *,
    device: Any,
    chunk_size: int,
    heads: Sequence[tuple[int, int]] = (),
) -> Any:
    import torch

    outputs = []
    context = ablate_heads(model, heads) if heads else contextlib.nullcontext()
    with context, torch.no_grad():
        for start in range(0, len(batch), int(chunk_size)):
            part = batch.slice(start, min(start + int(chunk_size), len(batch))).to(device)
            outputs.append(model(part).detach().cpu())
    return torch.cat(outputs, dim=0)


def evaluate(
    model: Any,
    cfg: Config,
    *,
    records: int,
    graphs: int,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    import torch.nn.functional as F

    batch = make_batch(cfg, int(graphs), int(records), int(seed))
    logits = predict_chunks(model, batch, device=device, chunk_size=batch_graphs(cfg, records))
    return {
        "N": int(records),
        "graphs": int(graphs),
        "loss": float(F.cross_entropy(logits, batch.y.long())),
        "accuracy": float((logits.argmax(dim=-1) == batch.y).float().mean()),
        "chance": 1.0 / float(records),
    }


def checkpoint_path(
    run_dir: Path,
    cfg: Config,
    model_name: str,
    width: int,
    records: int,
    seed: int,
) -> Path:
    return run_dir / "checkpoints" / (
        f"{model_name}__d{width}__N{records}__seed_{seed}__{config_fingerprint(cfg)}.pt"
    )


def converged_legacy_checkpoint(
    cfg: Config,
    model_name: str,
    width: int,
    records: int,
    seed: int,
) -> Path | None:
    """Find a v2 checkpoint only as a candidate for exact solved-model promotion."""
    if not cfg.converged_legacy_run or cfg.converged_legacy_run == cfg.run_name:
        return None
    directory = Path(cfg.drive_root) / cfg.converged_legacy_run / "checkpoints"
    matches = sorted(
        directory.glob(f"{model_name}__d{width}__N{records}__seed_{seed}__*.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def train_model(
    cfg: Config,
    *,
    model_name: str,
    width: int,
    records: int,
    seed: int,
    run_dir: Path,
    device: Any,
    force: bool,
    load_only: bool,
    optimization_attempt: int = 0,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    global OfficialFixedNNARGRIT
    if OfficialFixedNNARGRIT.__name__ == "OfficialFixedNNARGRIT":
        OfficialFixedNNARGRIT = build_model_class()
    path = checkpoint_path(run_dir, cfg, model_name, width, records, seed)
    # A targeted audit retry is an independently randomized optimization attempt for the same
    # experimental cell. Evaluation sets remain fixed, and the attempt is stored in the payload.
    optimization_seed = int(seed) + int(optimization_attempt) * 1_000_003
    model_seed = (
        optimization_seed
        + width * 1009
        + records * 10_007
        + MODEL_ORDER.index(model_name) * 100_003
    )
    set_seed(model_seed)
    model = OfficialFixedNNARGRIT(cfg, model_name, width, records).to(device)
    if path.exists() and not force:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("fingerprint") != config_fingerprint(cfg):
            raise RuntimeError(f"checkpoint fingerprint mismatch at {path}")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        print(f"[train {model_name} d={width} N={records} seed={seed}] loaded {path}", flush=True)
        return model, payload
    if load_only:
        raise FileNotFoundError(f"analysis requested but checkpoint is missing: {path}")

    # The v2 run already produced several exact 100%-held-out solutions. Reuse only those
    # checkpoints after they independently pass the v3 validation and held-out sets; all
    # non-converged v2 runs are discarded and trained afresh under the longer common budget.
    legacy_path = (
        converged_legacy_checkpoint(cfg, model_name, width, records, seed)
        if optimization_attempt == 0
        else None
    )
    if legacy_path is not None and not force:
        fresh_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        legacy = torch.load(legacy_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(legacy["state_dict"])
            model.eval()
            legacy_validation = evaluate(
                model,
                cfg,
                records=records,
                graphs=cfg.validation_graphs,
                seed=50_000_003 + seed * 10_007 + records,
                device=device,
            )
            legacy_heldout = evaluate(
                model,
                cfg,
                records=records,
                graphs=cfg.heldout_graphs,
                seed=800_000_011 + seed * 10_007 + records,
                device=device,
            )
        except (KeyError, RuntimeError, ValueError):
            legacy_validation = {"accuracy": 0.0}
            legacy_heldout = {"accuracy": 0.0}
        if min(legacy_validation["accuracy"], legacy_heldout["accuracy"]) >= 0.995:
            promoted_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            history = [{
                "step": 0,
                "train_loss": float("nan"),
                "validation_loss": legacy_validation["loss"],
                "validation_accuracy": legacy_validation["accuracy"],
                "elapsed_s": 0.0,
                "promoted_converged_legacy": True,
            }]
            payload = {
                "version": EXPERIMENT_VERSION,
                "fingerprint": config_fingerprint(cfg),
                "official_grit_commit": OFFICIAL_GRIT_COMMIT,
                "model_name": model_name,
                "width": int(width),
                "N": int(records),
                "seed": int(seed),
                "optimization_attempt": int(optimization_attempt),
                "optimization_seed": int(optimization_seed),
                "config": asdict(cfg),
                "state_dict": promoted_state,
                "best_validation": legacy_validation,
                "heldout": legacy_heldout,
                "history": history,
                "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
                "promoted_from": str(legacy_path),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
            write_csv(
                run_dir / "tables" / f"training_{model_name}_d{width}_N{records}_seed_{seed}.csv",
                history,
            )
            print(
                f"[train {model_name} d={width} N={records} seed={seed}] "
                f"promoted converged v2 checkpoint -> {path}",
                flush=True,
            )
            return model, payload
        model.load_state_dict(fresh_state)
        print(
            f"[train {model_name} d={width} N={records} seed={seed}] "
            "discarded non-converged v2 checkpoint; training v3 from initialization",
            flush=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_loss = float("inf")
    best_state = None
    best_validation: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    started = time.time()
    graphs = batch_graphs(cfg, records)
    for step in range(1, cfg.steps + 1):
        batch = make_batch(
            cfg,
            graphs,
            records,
            seed=optimization_seed * 10_000_019 + records * 100_003 + step * 101,
        ).to(device)
        model.train()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y.long())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 1 or step % cfg.eval_every == 0 or step == cfg.steps:
            model.eval()
            validation = evaluate(
                model,
                cfg,
                records=records,
                graphs=cfg.validation_graphs,
                seed=50_000_003 + seed * 10_007 + records,
                device=device,
            )
            history.append({
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "elapsed_s": time.time() - started,
            })
            print(
                f"[train {model_name} d={width} N={records} seed={seed}] "
                f"{step:4d}/{cfg.steps} loss={float(loss.detach().cpu()):.4f} "
                f"val={validation['accuracy']:.3f}",
                flush=True,
            )
            if validation["loss"] < best_loss:
                best_loss = float(validation["loss"])
                best_validation = validation
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            if validation["loss"] < cfg.early_stopping_loss_threshold:
                print(
                    f"[train {model_name} d={width} N={records} seed={seed}] "
                    f"early stop: validation loss {validation['loss']:.6f} "
                    f"< {cfg.early_stopping_loss_threshold:.6f}",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint candidate")
    model.load_state_dict(best_state)
    model.eval()
    heldout = evaluate(
        model,
        cfg,
        records=records,
        graphs=cfg.heldout_graphs,
        seed=800_000_011 + seed * 10_007 + records,
        device=device,
    )
    payload = {
        "version": EXPERIMENT_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "official_grit_commit": OFFICIAL_GRIT_COMMIT,
        "model_name": model_name,
        "width": int(width),
        "N": int(records),
        "seed": int(seed),
        "optimization_attempt": int(optimization_attempt),
        "optimization_seed": int(optimization_seed),
        "config": asdict(cfg),
        "state_dict": best_state,
        "best_validation": best_validation,
        "heldout": heldout,
        "history": history,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    write_csv(
        run_dir / "tables" / f"training_{model_name}_d{width}_N{records}_seed_{seed}.csv",
        history,
    )
    print(f"[train {model_name} d={width} N={records} seed={seed}] cached {path}", flush=True)
    return model, payload


def semantic_replica(batch: NarBatch, records: int, donor_index: int) -> NarBatch:
    """Change only the queried record's value while preserving key and graph."""
    changed = clone_batch(batch)
    for graph in range(len(changed)):
        target = int(changed.target_idx[graph])
        original = int(changed.y[graph])
        donor = (original + 1 + int(donor_index)) % int(records)
        changed.x[graph, target, 1] = donor
    return changed


def intervention_replicas(
    batch: NarBatch,
    records: int,
    donors: int,
) -> list[NarBatch]:
    return [clone_batch(batch)] + [
        semantic_replica(batch, records, donor) for donor in range(int(donors))
    ]


def verify_intervention(batch: NarBatch, records: int) -> dict[str, float]:
    import torch

    changed = semantic_replica(batch, records, 0)
    checks = {
        "adj_max": float((changed.adj - batch.adj).abs().max()),
        "rrwp_max": float((changed.rrwp - batch.rrwp).abs().max()),
        "key_max": float((changed.x[..., 0] - batch.x[..., 0]).abs().max()),
        "changed_value_entries": float(
            torch.count_nonzero((changed.x[..., 1] - batch.x[..., 1]).abs()).item()
        ),
    }
    if max(checks["adj_max"], checks["rrwp_max"], checks["key_max"]) > 0:
        raise RuntimeError(f"semantic intervention changed a nuisance factor: {checks}")
    if checks["changed_value_entries"] != len(batch):
        raise RuntimeError(f"semantic intervention did not change exactly one value token: {checks}")
    return checks


def transport_head_scores(
    model: Any,
    batch: NarBatch,
    cfg: Config,
    records: int,
    *,
    device: Any,
) -> Any:
    """Method-A functional magnitude at official GRIT's routed wV site."""
    import torch

    replicas = intervention_replicas(batch, records, cfg.score_donors)
    combined = concat_batches(replicas).to(device)
    captured = capture_forward(model, combined, want_grad=True)
    batch_size, nodes = len(batch), int(batch.x.size(1))
    scores = torch.zeros(model.L, model.H, device=device)
    for layer, routed in enumerate(captured["wV"]):
        view = routed.view(len(replicas), batch_size, nodes, model.H, model.dh)
        delta = view[0] - view[1:].mean(dim=0)
        projections = []
        for output in range(records):
            gradient = torch.autograd.grad(
                captured["logits"][:batch_size, output].sum(),
                routed,
                retain_graph=True,
            )[0]
            phi = gradient.view(len(replicas), batch_size, nodes, model.H, model.dh)[0]
            projections.append(torch.einsum("bnhd,bnhd->bnh", phi, delta))
        functional = torch.stack(projections).square().sum(dim=0).sqrt()
        scores[layer] = functional.sum(dim=1).mean(dim=0)
    return scores.detach().cpu()


def exact_intervention_effect(
    model: Any,
    batch: NarBatch,
    cfg: Config,
    records: int,
    *,
    device: Any,
    heads: Sequence[tuple[int, int]] = (),
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    replicas = intervention_replicas(batch, records, cfg.score_donors)
    combined = concat_batches(replicas).to(device)
    context = ablate_heads(model, heads) if heads else contextlib.nullcontext()
    with context, torch.no_grad():
        logits = model(combined).view(len(replicas), len(batch), records)
    clean, corrupt = logits[0], logits[1:]
    labels = batch.y.to(device).long()
    clean_loss = F.cross_entropy(clean, labels, reduction="none")
    corrupt_loss = torch.stack([
        F.cross_entropy(item, labels, reduction="none") for item in corrupt
    ])
    return {
        "functional": torch.linalg.vector_norm(clean - corrupt.mean(dim=0), dim=-1).cpu(),
        "beneficial": (corrupt_loss.mean(dim=0) - clean_loss).cpu(),
    }


def clean_head_ablation(model: Any, batch: NarBatch, *, device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    clean = batch.to(device)
    with torch.no_grad():
        baseline = model(clean)
    labels = clean.y.long()
    base_loss = F.cross_entropy(baseline, labels, reduction="none")
    functional = torch.zeros(model.L, model.H, len(batch))
    loss_delta = torch.zeros_like(functional)
    accuracy_drop = torch.zeros_like(functional)
    for layer in range(model.L):
        for head in range(model.H):
            with ablate_heads(model, [(layer, head)]), torch.no_grad():
                changed = model(clean)
            functional[layer, head] = torch.linalg.vector_norm(changed - baseline, dim=-1).cpu()
            loss_delta[layer, head] = (
                F.cross_entropy(changed, labels, reduction="none") - base_loss
            ).cpu()
            accuracy_drop[layer, head] = (
                (baseline.argmax(-1) == labels).float() - (changed.argmax(-1) == labels).float()
            ).cpu()
    return {"functional": functional, "loss_delta": loss_delta, "accuracy_drop": accuracy_drop}


def head_resolved_carriage(
    model: Any,
    batch: NarBatch,
    cfg: Config,
    records: int,
    *,
    device: Any,
) -> dict[str, Any]:
    import torch

    intact = exact_intervention_effect(model, batch, cfg, records, device=device)
    functional_loss = torch.zeros(model.L, model.H, len(batch))
    beneficial_loss = torch.zeros_like(functional_loss)
    for layer in range(model.L):
        for head in range(model.H):
            changed = exact_intervention_effect(
                model,
                batch,
                cfg,
                records,
                device=device,
                heads=[(layer, head)],
            )
            functional_loss[layer, head] = intact["functional"] - changed["functional"]
            beneficial_loss[layer, head] = intact["beneficial"] - changed["beneficial"]
    return {
        "functional": intact["functional"],
        "beneficial": intact["beneficial"],
        "functional_head_loss": functional_loss,
        "beneficial_head_loss": beneficial_loss,
    }


def attention_target_selection(model: Any, batch: NarBatch, *, device: Any) -> dict[str, Any]:
    import torch

    clean = batch.to(device)
    captured = capture_forward(model, clean, want_grad=False, want_attention=True)
    graph_idx = model.last_support_graph
    source = model.last_support_source_local
    destination = model.last_support_destination_local
    advantage = torch.full((model.L, model.H, len(batch)), float("nan"), device=device)
    ratio = torch.full_like(advantage, float("nan"))
    for graph in range(len(batch)):
        eligible = (graph_idx == graph) & (destination == clean.central_idx[graph])
        target_edge = eligible & (source == clean.target_idx[graph])
        background = eligible & clean.record_mask[graph, source] & ~target_edge
        if not target_edge.any() or not background.any():
            continue
        for layer, attention in enumerate(captured["attention"]):
            target_weight = attention[target_edge].mean(dim=0)
            background_weight = attention[background].mean(dim=0)
            advantage[layer, :, graph] = target_weight - background_weight
            ratio[layer, :, graph] = target_weight / background_weight.clamp_min(1e-9)
    return {"advantage": advantage.cpu(), "ratio": ratio.cpu()}


def noop_transport_max(model: Any, batch: NarBatch, *, device: Any) -> float:
    combined = concat_batches([clone_batch(batch), clone_batch(batch)]).to(device)
    captured = capture_forward(model, combined, want_grad=False)
    batch_size, nodes = len(batch), int(batch.x.size(1))
    maximum = 0.0
    for routed in captured["wV"]:
        view = routed.view(2, batch_size, nodes, model.H, model.dh)
        maximum = max(maximum, float((view[0] - view[1]).abs().max().cpu()))
    return maximum


def decode_heads(indices: Sequence[int], heads: int) -> list[tuple[int, int]]:
    return [(int(index // heads), int(index % heads)) for index in indices]


def select_families(scores: Any, cfg: Config, seed: int) -> dict[str, list[list[tuple[int, int]]]]:
    flat = scores.flatten()
    order = np.argsort(-flat.numpy()).tolist()
    top = decode_heads(order[: cfg.family_size], cfg.heads)
    pool = np.asarray(order[cfg.family_size :], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    controls = [
        decode_heads(rng.choice(pool, size=cfg.family_size, replace=False).tolist(), cfg.heads)
        for _ in range(cfg.random_families)
    ]
    return {"top": [top], "control": controls}


def family_ablation(
    model: Any,
    batch: NarBatch,
    families: Mapping[str, Sequence[Sequence[tuple[int, int]]]],
    *,
    device: Any,
    chunk_size: int,
) -> dict[str, list[dict[str, Any]]]:
    import torch
    import torch.nn.functional as F

    result: dict[str, list[dict[str, Any]]] = {name: [] for name in families}
    for name, sets in families.items():
        for heads in sets:
            loss_delta, accuracy_drop = [], []
            for start in range(0, len(batch), int(chunk_size)):
                clean = batch.slice(start, min(start + int(chunk_size), len(batch))).to(device)
                with torch.no_grad():
                    baseline = model(clean)
                with ablate_heads(model, heads), torch.no_grad():
                    changed = model(clean)
                labels = clean.y.long()
                loss_delta.append((
                    F.cross_entropy(changed, labels, reduction="none")
                    - F.cross_entropy(baseline, labels, reduction="none")
                ).cpu())
                accuracy_drop.append((
                    (baseline.argmax(-1) == labels).float()
                    - (changed.argmax(-1) == labels).float()
                ).cpu())
            result[name].append({
                "heads": list(heads),
                "loss_delta": torch.cat(loss_delta),
                "accuracy_drop": torch.cat(accuracy_drop),
            })
    return result


def analysis_path(
    run_dir: Path,
    cfg: Config,
    model_name: str,
    records: int,
    seed: int,
) -> Path:
    return run_dir / "analysis" / (
        f"{model_name}__d{cfg.analysis_width}__N{records}__seed_{seed}__{config_fingerprint(cfg)}.pt"
    )


def analyze_model(
    model: Any,
    payload: Mapping[str, Any],
    cfg: Config,
    model_name: str,
    records: int,
    seed: int,
    *,
    run_dir: Path,
    device: Any,
    force: bool,
) -> dict[str, Any]:
    import torch

    path = analysis_path(run_dir, cfg, model_name, records, seed)
    if path.exists() and not force:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("fingerprint") != config_fingerprint(cfg):
            raise RuntimeError(f"analysis fingerprint mismatch at {path}")
        print(f"[analysis cache] {path}", flush=True)
        return cached
    print(f"[analysis {model_name} N={records} seed={seed}]", flush=True)
    score_batch = make_batch(
        cfg,
        cfg.score_graphs,
        records,
        1_100_000_007 + seed * 10_007 + records,
    )
    checks = verify_intervention(score_batch, records)
    scores = transport_head_scores(model, score_batch, cfg, records, device=device)
    causal_batch = make_batch(
        cfg,
        cfg.analysis_graphs,
        records,
        1_200_000_017 + seed * 10_007 + records,
    )
    ablations = clean_head_ablation(model, causal_batch, device=device)
    carriage = head_resolved_carriage(model, causal_batch, cfg, records, device=device)
    attention = attention_target_selection(model, causal_batch, device=device)
    families = select_families(scores, cfg, seed=1_250_000 + seed * 101 + records)
    family_batch = make_batch(
        cfg,
        cfg.ablation_graphs,
        records,
        1_300_000_019 + seed * 10_007 + records,
    )
    family = family_ablation(
        model,
        family_batch,
        families,
        device=device,
        chunk_size=batch_graphs(cfg, records),
    )
    noop_max = noop_transport_max(
        model,
        score_batch.slice(0, min(2, len(score_batch))),
        device=device,
    )
    if noop_max > NOOP_TRANSPORT_ATOL:
        raise RuntimeError(
            "identical replicas produced routed transport "
            f"{noop_max:.3e} (tolerance {NOOP_TRANSPORT_ATOL:.1e})"
        )
    result = {
        "version": EXPERIMENT_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "model_name": model_name,
        "width": cfg.analysis_width,
        "N": int(records),
        "seed": int(seed),
        "heldout": payload["heldout"],
        "scores": scores,
        "ablations": ablations,
        "carriage": carriage,
        "attention": attention,
        "families": families,
        "family_ablation": family,
        "intervention_checks": checks,
        "noop_transport_max": noop_max,
        "noop_transport_tolerance": NOOP_TRANSPORT_ATOL,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    print(f"[analysis] cached {path}", flush=True)
    return result


def configure_plots() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "legend.frameon": False,
    })


def save_figure(fig: Any, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")


def mean_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    mean = float(array.mean())
    error = 0.0 if len(array) == 1 else float(1.96 * array.std(ddof=1) / math.sqrt(len(array)))
    return mean, error


def rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x_array, y_array = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x_array) & np.isfinite(y_array)
    if keep.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x_array[keep]), rankdata(y_array[keep])
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def performance_rows(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "model": payload["model_name"],
        "width": int(payload["width"]),
        "N": int(payload["N"]),
        "seed": int(payload["seed"]),
        "optimization_attempt": int(payload.get("optimization_attempt", 0)),
        "outlier_retrained": bool(payload.get("outlier_retraining")),
        "parameters": int(payload["parameters"]),
        **dict(payload["heldout"]),
    } for payload in payloads]


def head_rows(analyses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analysis in analyses:
        scores = analysis["scores"]
        ablation = analysis["ablations"]
        carriage = analysis["carriage"]
        attention = analysis["attention"]
        score_scale = float(scores.mean().clamp_min(EPS))
        ablation_scale = float(ablation["functional"].mean().clamp_min(EPS))
        carriage_scale = float(carriage["functional_head_loss"].abs().mean().clamp_min(EPS))
        for layer in range(int(scores.size(0))):
            for head in range(int(scores.size(1))):
                rows.append({
                    "model": analysis["model_name"],
                    "width": int(analysis["width"]),
                    "N": int(analysis["N"]),
                    "seed": int(analysis["seed"]),
                    "layer": layer,
                    "head": head,
                    "score": float(scores[layer, head]),
                    "score_normalized": float(scores[layer, head]) / score_scale,
                    "ablation_functional": float(ablation["functional"][layer, head].mean()),
                    "ablation_functional_normalized": (
                        float(ablation["functional"][layer, head].mean()) / ablation_scale
                    ),
                    "ablation_loss": float(ablation["loss_delta"][layer, head].mean()),
                    "ablation_accuracy_drop": float(ablation["accuracy_drop"][layer, head].mean()),
                    "functional_carriage_loss": float(carriage["functional_head_loss"][layer, head].mean()),
                    "functional_carriage_loss_normalized": (
                        float(carriage["functional_head_loss"][layer, head].mean()) / carriage_scale
                    ),
                    "beneficial_carriage_loss": float(carriage["beneficial_head_loss"][layer, head].mean()),
                    "attention_advantage": float(torch_nanmean(attention["advantage"][layer, head])),
                    "attention_ratio": float(torch_nanmean(attention["ratio"][layer, head])),
                    "heldout_accuracy": float(analysis["heldout"]["accuracy"]),
                })
    return rows


def torch_nanmean(value: Any) -> float:
    import torch

    tensor = torch.as_tensor(value, dtype=torch.float32)
    finite = torch.isfinite(tensor)
    return float(tensor[finite].mean()) if finite.any() else float("nan")


def family_rows(analyses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analysis in analyses:
        for kind, entries in analysis["family_ablation"].items():
            for replicate, entry in enumerate(entries):
                rows.append({
                    "model": analysis["model_name"],
                    "N": int(analysis["N"]),
                    "seed": int(analysis["seed"]),
                    "family": kind,
                    "replicate": replicate,
                    "heads": str(entry["heads"]),
                    "loss_delta": float(entry["loss_delta"].mean()),
                    "accuracy_drop": float(entry["accuracy_drop"].mean()),
                    "heldout_accuracy": float(analysis["heldout"]["accuracy"]),
                })
    return rows


def carriage_rows(analyses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "model": analysis["model_name"],
        "N": int(analysis["N"]),
        "seed": int(analysis["seed"]),
        "functional": float(analysis["carriage"]["functional"].mean()),
        "beneficial": float(analysis["carriage"]["beneficial"].mean()),
        "attention_ratio": float(torch_nanmean(analysis["attention"]["ratio"][-1])),
        "accuracy": float(analysis["heldout"]["accuracy"]),
        "noop_transport_max": float(analysis["noop_transport_max"]),
    } for analysis in analyses]


def _panel(axis: Any, letter: str, title: str) -> None:
    axis.set_title(f"{letter}  {title}", loc="left")


def _suptitle(fig: Any, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.06, y=0.985, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.06, 0.945, subtitle, ha="left", color="#555555", fontsize=10.5)


def plot_capacity(rows: Sequence[Mapping[str, Any]], cfg: Config, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cfg.widths), figsize=(6.4 * len(cfg.widths), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for column, width in enumerate(cfg.widths):
        axis = axes[column]
        for model in cfg.models:
            means, errors = [], []
            for records in cfg.ns:
                values = [
                    row["accuracy"] for row in rows
                    if row["width"] == width and row["model"] == model and row["N"] == records
                ]
                mean, error = mean_ci(values)
                means.append(mean)
                errors.append(error)
            axis.errorbar(
                cfg.ns,
                means,
                yerr=errors,
                color=MODEL_COLOURS[model],
                marker=MODEL_MARKERS[model],
                lw=2.2,
                capsize=2.5,
                label=model.replace("hop", "-hop"),
            )
        axis.plot(cfg.ns, [1.0 / value for value in cfg.ns], color="#777777", ls=":", label="chance")
        axis.set_xscale("log", base=2)
        axis.set_xticks(cfg.ns, [str(value) for value in cfg.ns])
        axis.set_ylim(-0.02, 1.04)
        axis.set_xlabel("Fixed neighbourhood size N")
        if column == 0:
            axis.set_ylabel("Held-out recall accuracy")
        _panel(axis, chr(ord("A") + column), f"Width {width}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    _suptitle(
        fig,
        "Trained attention support determines Neighbor Associative Recall capacity",
        "Each point is a separately trained fixed-N official GRIT; bars are seed-level 95% intervals. "
        "Audited retries are disclosed in the performance table.",
    )
    fig.subplots_adjust(top=0.84, bottom=0.20, wspace=0.14)
    save_figure(fig, figure_dir, "01_fixed_n_capacity")
    plt.close(fig)


def plot_causality(
    heads: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
    cfg: Config,
    figure_dir: Path,
) -> dict[str, float]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    solved_heads = [row for row in heads if row["heldout_accuracy"] >= 0.85]

    def scatter(axis: Any, x_key: str, y_key: str, title: str, letter: str) -> float:
        for model in cfg.models:
            selected = [row for row in solved_heads if row["model"] == model]
            axis.scatter(
                [row[x_key] for row in selected],
                [row[y_key] for row in selected],
                s=24,
                alpha=0.62,
                color=MODEL_COLOURS[model],
                marker=MODEL_MARKERS[model],
                label=model.replace("hop", "-hop"),
            )
        x = [row[x_key] for row in solved_heads]
        y = [row[y_key] for row in solved_heads]
        rho = spearman(x, y)
        axis.text(0.04, 0.95, rf"pooled $\rho={rho:.2f}$", transform=axis.transAxes, va="top")
        axis.axhline(0, color="#777777", lw=1)
        axis.set_xlabel("Semantic transport score (cell-normalized)")
        axis.set_ylabel(y_key.replace("_", " ").capitalize())
        _panel(axis, letter, title)
        return rho

    rho_ablation = scatter(
        axes[0, 0], "score_normalized", "ablation_functional_normalized", "Scores predict clean-input head impact", "A"
    )
    rho_carriage = scatter(
        axes[0, 1], "score_normalized", "functional_carriage_loss_normalized", "Scores identify heads sustaining carriage", "B"
    )

    labels = ["Score-selected", "Random controls"]
    for index, measure in enumerate(("loss_delta", "accuracy_drop")):
        axis = axes[1, index]
        solved_families = [row for row in families if row["heldout_accuracy"] >= 0.85]
        top = [row[measure] for row in solved_families if row["family"] == "top"]
        grouped_controls: dict[tuple[str, int, int], list[float]] = {}
        for row in solved_families:
            if row["family"] == "control":
                grouped_controls.setdefault((row["model"], row["N"], row["seed"]), []).append(row[measure])
        control = [float(np.mean(values)) for values in grouped_controls.values()]
        top_mean, top_error = mean_ci(top)
        control_mean, control_error = mean_ci(control)
        means = (top_mean, control_mean)
        errors = (top_error, control_error)
        axis.bar(
            [0, 1],
            means,
            yerr=errors,
            color=["#d7301f", "#bdbdbd"],
            capsize=4,
            width=0.66,
        )
        axis.set_xticks([0, 1], labels)
        axis.axhline(0, color="#777777", lw=1)
        axis.set_ylabel("Cross-entropy increase" if measure == "loss_delta" else "Accuracy drop")
        _panel(
            axis,
            chr(ord("C") + index),
            "Score-selected family necessity" if index == 0 else "Task-level family effect",
        )
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=len(legend_labels), frameon=False)
    _suptitle(
        fig,
        "Semantic transport scores identify the GRIT heads that causally support recall",
        "Independent graph sets and solved cells only (accuracy >= 0.85); controls are equal-size random families.",
    )
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.24, hspace=0.30)
    save_figure(fig, figure_dir, "02_score_causality")
    plt.close(fig)
    return {"score_ablation_rho": rho_ablation, "score_carriage_rho": rho_carriage}


def plot_retrieval_mechanism(
    rows: Sequence[Mapping[str, Any]],
    cfg: Config,
    figure_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    measures = (
        ("functional", "Output displacement", "Functional semantic carriage"),
        ("beneficial", "Corruption loss increase", "Beneficial semantic carriage"),
        ("attention_ratio", "Target / background attention", "Queried-record selection"),
    )
    for column, (measure, ylabel, title) in enumerate(measures):
        axis = axes[column]
        for model in cfg.models:
            means, errors = [], []
            for records in cfg.mechanistic_ns:
                values = [row[measure] for row in rows if row["model"] == model and row["N"] == records]
                mean, error = mean_ci(values)
                means.append(mean)
                errors.append(error)
            axis.errorbar(
                cfg.mechanistic_ns,
                means,
                yerr=errors,
                color=MODEL_COLOURS[model],
                marker=MODEL_MARKERS[model],
                lw=2.1,
                capsize=2.5,
                label=model.replace("hop", "-hop"),
            )
        axis.axhline(0 if measure != "attention_ratio" else 1, color="#777777", lw=1, ls=":")
        axis.set_xscale("log", base=2)
        axis.set_xticks(cfg.mechanistic_ns, [str(value) for value in cfg.mechanistic_ns])
        axis.set_xlabel("Fixed neighbourhood size N")
        axis.set_ylabel(ylabel)
        _panel(axis, chr(ord("A") + column), title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    _suptitle(
        fig,
        "Carriage and attention expose how retrieval changes as memory load grows",
        "Target-value interventions and final-layer target selection are measured on held-out graphs.",
    )
    fig.subplots_adjust(top=0.82, bottom=0.22, wspace=0.27)
    save_figure(fig, figure_dir, "03_retrieval_mechanism")
    plt.close(fig)


def make_all_figures(
    payloads: Sequence[Mapping[str, Any]],
    analyses: Sequence[Mapping[str, Any]],
    cfg: Config,
    run_dir: Path,
) -> dict[str, Any]:
    configure_plots()
    figure_dir, table_dir = run_dir / "figures", run_dir / "tables"
    performance = performance_rows(payloads)
    heads = head_rows(analyses)
    families = family_rows(analyses)
    carriage = carriage_rows(analyses)
    write_csv(table_dir / "heldout_performance.csv", performance)
    write_csv(table_dir / "head_mechanisms.csv", heads)
    write_csv(table_dir / "family_ablations.csv", families)
    write_csv(table_dir / "aggregate_carriage.csv", carriage)
    plot_capacity(performance, cfg, figure_dir)
    headline = plot_causality(heads, families, cfg, figure_dir)
    plot_retrieval_mechanism(carriage, cfg, figure_dir)
    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "paper_task_alignment": TASK_ALIGNMENT,
        "intentional_difference": INTENTIONAL_DIFFERENCE,
        "outlier_retrained_cells": sum(bool(row["outlier_retrained"]) for row in performance),
        "headline_statistics": headline,
        "max_noop_transport": max((row["noop_transport_max"] for row in carriage), default=float("nan")),
        "artifacts": {
            "figures": sorted(path.name for path in figure_dir.glob("*.png")),
            "tables": sorted(path.name for path in table_dir.glob("*.csv")),
        },
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def parse_int_tuple(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(map(int, value))


def parse_str_tuple(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(map(str, value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="nar_grit_fixed_n_v3")
    parser.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--phase", choices=("all", "train", "analyze", "figures"), default="all")
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--widths", default="64,128")
    parser.add_argument("--analysis-width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ns", default="4,8,16,32,64")
    parser.add_argument("--mechanistic-ns", default="4,16,64")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--early-stopping-loss-threshold", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--score-graphs", type=int, default=8)
    parser.add_argument("--score-donors", type=int, default=3)
    parser.add_argument("--analysis-graphs", type=int, default=12)
    parser.add_argument("--ablation-graphs", type=int, default=128)
    parser.add_argument("--family-size", type=int, default=2)
    parser.add_argument("--random-families", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grit-dir", default=DEFAULT_GRIT_DIR)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--force-training", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument(
        "--retrain-outliers",
        action="store_true",
        help="audit isolated seed outliers and replace each with one unconditional retry",
    )
    parser.add_argument(
        "--outlier-min-gap",
        type=float,
        default=0.15,
        help="minimum accuracy separation from the agreeing peer seeds",
    )
    parser.add_argument(
        "--outlier-peer-range",
        type=float,
        default=0.05,
        help="maximum held-out accuracy range among the peer seeds",
    )
    parser.add_argument("--allow-low-accuracy", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    values: dict[str, Any] = {
        "run_name": args.run_name,
        "drive_root": args.drive_root,
        "models": parse_str_tuple(args.models),
        "widths": parse_int_tuple(args.widths),
        "analysis_width": args.analysis_width,
        "heads": args.heads,
        "layers": args.layers,
        "ns": parse_int_tuple(args.ns),
        "mechanistic_ns": parse_int_tuple(args.mechanistic_ns),
        "seeds": parse_int_tuple(args.seeds),
        "steps": args.steps,
        "early_stopping_loss_threshold": args.early_stopping_loss_threshold,
        "batch_size": args.batch_size,
        "score_graphs": args.score_graphs,
        "score_donors": args.score_donors,
        "analysis_graphs": args.analysis_graphs,
        "ablation_graphs": args.ablation_graphs,
        "family_size": args.family_size,
        "random_families": args.random_families,
        "device": args.device,
    }
    if args.fast_dev_run:
        values.update({
            "models": ("1hop", "dense"),
            "widths": (32,),
            "analysis_width": 32,
            "heads": 4,
            "ns": (4, 8),
            "mechanistic_ns": (4, 8),
            "seeds": (0,),
            "steps": 8,
            "eval_every": 4,
            "validation_graphs": 4,
            "heldout_graphs": 8,
            "batch_size": 2,
            "max_batch_nodes": 128,
            "max_dense_pairs": 20_000,
            "score_graphs": 2,
            "score_donors": 1,
            "analysis_graphs": 2,
            "ablation_graphs": 4,
            "family_size": 1,
            "random_families": 2,
            "early_stopping_loss_threshold": -1.0,
            "low_n_accuracy_gate": 0.0,
        })
    cfg = Config(**values)
    cfg.validate()
    return cfg


def resolve_device(requested: str) -> Any:
    import torch

    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def load_payloads(cfg: Config, run_dir: Path) -> list[dict[str, Any]]:
    import torch

    payloads = []
    for width in cfg.widths:
        for records in cfg.ns:
            for model_name in cfg.models:
                for seed in cfg.seeds:
                    path = checkpoint_path(run_dir, cfg, model_name, width, records, seed)
                    if not path.exists():
                        raise FileNotFoundError(f"missing checkpoint: {path}")
                    payloads.append(torch.load(path, map_location="cpu", weights_only=False))
    return payloads


def load_analyses(cfg: Config, run_dir: Path) -> list[dict[str, Any]]:
    import torch

    analyses = []
    for records in cfg.mechanistic_ns:
        for model_name in cfg.models:
            for seed in cfg.seeds:
                path = analysis_path(run_dir, cfg, model_name, records, seed)
                if not path.exists():
                    raise FileNotFoundError(f"missing analysis cache: {path}")
                cached = torch.load(path, map_location="cpu", weights_only=False)
                if cached.get("fingerprint") != config_fingerprint(cfg):
                    raise RuntimeError(f"analysis fingerprint mismatch at {path}")
                analyses.append(cached)
    return analyses


def payload_cell(payload: Mapping[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(payload["width"]),
        int(payload["N"]),
        str(payload["model_name"]),
        int(payload["seed"]),
    )


def detect_accuracy_outliers(
    payloads: Sequence[Mapping[str, Any]],
    *,
    min_gap: float,
    peer_range: float,
) -> list[dict[str, Any]]:
    """Identify isolated optimization outcomes using held-out and validation agreement.

    A run is eligible only when at least two *other* seeds agree within ``peer_range``, its
    held-out accuracy differs from their median by ``min_gap``, and the independently measured
    validation difference points the same way and is at least half as large. A checkpoint that
    has already received its one audit retry is never selected again.
    """
    grouped: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {}
    for payload in payloads:
        key = (int(payload["width"]), int(payload["N"]), str(payload["model_name"]))
        grouped.setdefault(key, []).append(payload)

    detected: list[dict[str, Any]] = []
    for (width, records, model_name), group in grouped.items():
        if len(group) < 3:
            continue
        for candidate in group:
            if candidate.get("outlier_retraining"):
                continue
            peers = [item for item in group if int(item["seed"]) != int(candidate["seed"])]
            if len(peers) < 2:
                continue
            peer_heldout = np.asarray(
                [float(item["heldout"]["accuracy"]) for item in peers], dtype=float
            )
            if float(np.ptp(peer_heldout)) > float(peer_range):
                continue
            candidate_heldout = float(candidate["heldout"]["accuracy"])
            peer_heldout_median = float(np.median(peer_heldout))
            heldout_gap = candidate_heldout - peer_heldout_median
            if abs(heldout_gap) < float(min_gap):
                continue

            candidate_validation = float(candidate["best_validation"]["accuracy"])
            peer_validation_median = float(np.median([
                float(item["best_validation"]["accuracy"]) for item in peers
            ]))
            validation_gap = candidate_validation - peer_validation_median
            if heldout_gap * validation_gap <= 0 or abs(validation_gap) < 0.5 * float(min_gap):
                continue
            detected.append({
                "width": width,
                "N": records,
                "model": model_name,
                "seed": int(candidate["seed"]),
                "direction": "high" if heldout_gap > 0 else "low",
                "heldout_accuracy": candidate_heldout,
                "peer_heldout_median": peer_heldout_median,
                "heldout_gap": heldout_gap,
                "validation_accuracy": candidate_validation,
                "peer_validation_median": peer_validation_median,
                "validation_gap": validation_gap,
                "previous_optimization_attempt": int(candidate.get("optimization_attempt", 0)),
            })
    return detected


def archive_active_cache(source: Path, run_dir: Path, archive_root: Path) -> str:
    """Move an active cache artifact into the recoverable outlier-audit archive."""
    if not source.exists():
        return ""
    relative = source.relative_to(run_dir)
    destination = archive_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return str(destination)


def retrain_accuracy_outliers(
    payloads: Sequence[dict[str, Any]],
    cfg: Config,
    run_dir: Path,
    *,
    device: Any,
    min_gap: float,
    peer_range: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Archive and unconditionally replace each detected seed with one independent retry."""
    import torch

    flagged = detect_accuracy_outliers(
        payloads,
        min_gap=float(min_gap),
        peer_range=float(peer_range),
    )
    if not flagged:
        print("[outlier audit] no isolated seed outliers detected", flush=True)
        return list(payloads), []

    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"
    archive_root = run_dir / "outlier_archive" / stamp
    current = {payload_cell(payload): payload for payload in payloads}
    audit_rows: list[dict[str, Any]] = []
    for finding in flagged:
        key = (
            int(finding["width"]),
            int(finding["N"]),
            str(finding["model"]),
            int(finding["seed"]),
        )
        original = current[key]
        width, records, model_name, seed = key
        checkpoint = checkpoint_path(run_dir, cfg, model_name, width, records, seed)
        training_table = (
            run_dir / "tables" / f"training_{model_name}_d{width}_N{records}_seed_{seed}.csv"
        )
        archived_checkpoint = archive_active_cache(checkpoint, run_dir, archive_root)
        archived_training = archive_active_cache(training_table, run_dir, archive_root)
        archived_analysis = ""
        if width == cfg.analysis_width and records in cfg.mechanistic_ns:
            archived_analysis = archive_active_cache(
                analysis_path(run_dir, cfg, model_name, records, seed),
                run_dir,
                archive_root,
            )

        next_attempt = int(original.get("optimization_attempt", 0)) + 1
        print(
            f"[outlier audit] retraining {model_name} d={width} N={records} seed={seed} "
            f"({finding['direction']} gap={finding['heldout_gap']:+.3f}, attempt={next_attempt})",
            flush=True,
        )
        model, replacement = train_model(
            cfg,
            model_name=model_name,
            width=width,
            records=records,
            seed=seed,
            run_dir=run_dir,
            device=device,
            force=True,
            load_only=False,
            optimization_attempt=next_attempt,
        )
        retry_metadata = {
            "completed": True,
            "detector": "agreeing-peers plus independent-validation v1",
            "accepted_unconditionally": True,
            "min_gap": float(min_gap),
            "peer_range": float(peer_range),
            "original_heldout_accuracy": float(original["heldout"]["accuracy"]),
            "replacement_heldout_accuracy": float(replacement["heldout"]["accuracy"]),
            "archive_root": str(archive_root),
        }
        replacement["outlier_retraining"] = retry_metadata
        torch.save(replacement, checkpoint)
        current[key] = replacement
        audit_rows.append({
            **finding,
            "new_optimization_attempt": next_attempt,
            "replacement_validation_accuracy": float(replacement["best_validation"]["accuracy"]),
            "replacement_heldout_accuracy": float(replacement["heldout"]["accuracy"]),
            "accepted_unconditionally": True,
            "archived_checkpoint": archived_checkpoint,
            "archived_training_table": archived_training,
            "archived_analysis": archived_analysis,
        })
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    audit_path = run_dir / "tables" / f"outlier_retraining_{stamp}.csv"
    write_csv(audit_path, audit_rows)
    print(f"[outlier audit] wrote {audit_path}", flush=True)
    ordered = [current[payload_cell(payload)] for payload in payloads]
    return ordered, audit_rows


def assert_parameter_matching(
    payloads: Sequence[Mapping[str, Any]],
    cfg: Config,
    run_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for width in cfg.widths:
        for records in cfg.ns:
            selected = [
                payload for payload in payloads
                if int(payload["width"]) == width and int(payload["N"]) == records
            ]
            values = {int(payload["parameters"]) for payload in selected}
            if len(values) != 1:
                raise RuntimeError(
                    f"support variants are not parameter matched at width={width}, N={records}: {values}"
                )
            rows.append({
                "width": width,
                "N": records,
                "heads": cfg.heads,
                "layers": cfg.layers,
                "parameters": values.pop(),
            })
    write_csv(run_dir / "tables" / "parameter_budget.csv", rows)


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    parser_argv = list(argv) if argv is not None else None
    if parser_argv is None and ("google.colab" in sys.modules or "ipykernel" in sys.modules):
        parser_argv = []
    args = build_parser().parse_args(parser_argv)
    if args.retrain_outliers and args.phase == "figures":
        parser.error("--retrain-outliers requires --phase train, analyze, or all")
    if args.retrain_outliers and args.force_training:
        parser.error("use either --retrain-outliers or --force-training, not both")
    if not 0 < args.outlier_min_gap <= 1:
        parser.error("--outlier-min-gap must lie in (0, 1]")
    if not 0 <= args.outlier_peer_range < args.outlier_min_gap:
        parser.error("--outlier-peer-range must be non-negative and smaller than the gap")
    cfg = config_from_args(args)
    run_dir = Path(cfg.drive_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "experiment_config.json", {
        "version": EXPERIMENT_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "config": asdict(cfg),
        "paper_task_alignment": TASK_ALIGNMENT,
        "intentional_difference": INTENTIONAL_DIFFERENCE,
        "outlier_audit": {
            "enabled": bool(args.retrain_outliers),
            "minimum_accuracy_gap": float(args.outlier_min_gap),
            "maximum_peer_range": float(args.outlier_peer_range),
            "maximum_retries_per_checkpoint": 1,
            "replacement_policy": "accept the single retry unconditionally",
        },
    })

    if args.phase == "figures":
        payloads = load_payloads(cfg, run_dir)
        analyses = load_analyses(cfg, run_dir)
        assert_parameter_matching(payloads, cfg, run_dir)
        return make_all_figures(payloads, analyses, cfg, run_dir)

    setup_official_grit(Path(args.grit_dir), install=not args.skip_install)
    device = resolve_device(cfg.device)
    write_json(run_dir / "environment.json", {
        "python": sys.version,
        "device": str(device),
        "official_grit_commit": OFFICIAL_GRIT_COMMIT,
        "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    })

    payloads: list[dict[str, Any]] | None = None
    if args.phase in ("all", "train"):
        payloads = []
        for width in cfg.widths:
            for records in cfg.ns:
                for model_name in cfg.models:
                    for seed in cfg.seeds:
                        model, payload = train_model(
                            cfg,
                            model_name=model_name,
                            width=width,
                            records=records,
                            seed=seed,
                            run_dir=run_dir,
                            device=device,
                            force=args.force_training,
                            load_only=False,
                        )
                        accuracy = float(payload["heldout"]["accuracy"])
                        print(
                            f"[gate {model_name} d={width} N={records} seed={seed}] accuracy={accuracy:.3f}",
                            flush=True,
                        )
                        if records == min(cfg.ns) and accuracy < cfg.low_n_accuracy_gate and not (
                            args.allow_low_accuracy or args.fast_dev_run
                        ):
                            raise RuntimeError(
                                f"{model_name} width {width} seed {seed} failed the N={records} "
                                f"matched-performance gate ({accuracy:.3f} < {cfg.low_n_accuracy_gate:.3f})"
                            )
                        payloads.append(payload)
                        del model
    elif args.retrain_outliers:
        payloads = load_payloads(cfg, run_dir)

    if args.retrain_outliers:
        if payloads is None:
            raise RuntimeError("outlier audit requires loaded training payloads")
        payloads, _ = retrain_accuracy_outliers(
            payloads,
            cfg,
            run_dir,
            device=device,
            min_gap=args.outlier_min_gap,
            peer_range=args.outlier_peer_range,
        )

    if args.phase in ("all", "train"):
        if payloads is None:
            raise RuntimeError("training phase produced no payloads")
        assert_parameter_matching(payloads, cfg, run_dir)
        if args.phase == "train":
            print("[done] training caches are complete; run --phase analyze next", flush=True)
            return None

    if args.phase in ("all", "analyze"):
        for records in cfg.mechanistic_ns:
            for model_name in cfg.models:
                for seed in cfg.seeds:
                    model, payload = train_model(
                        cfg,
                        model_name=model_name,
                        width=cfg.analysis_width,
                        records=records,
                        seed=seed,
                        run_dir=run_dir,
                        device=device,
                        force=False,
                        load_only=True,
                    )
                    analyze_model(
                        model,
                        payload,
                        cfg,
                        model_name,
                        records,
                        seed,
                        run_dir=run_dir,
                        device=device,
                        force=args.force_analysis,
                    )
                    del model
                    if str(device).startswith("cuda"):
                        import torch

                        torch.cuda.empty_cache()
        if args.phase == "analyze":
            print("[done] analysis caches are complete; run --phase figures next", flush=True)
            return None

    payloads = load_payloads(cfg, run_dir)
    analyses = load_analyses(cfg, run_dir)
    assert_parameter_matching(payloads, cfg, run_dir)
    summary = make_all_figures(payloads, analyses, cfg, run_dir)
    print(f"[done] all artifacts saved under {run_dir}", flush=True)
    return summary


if __name__ == "__main__":
    main()
