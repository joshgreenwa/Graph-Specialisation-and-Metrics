"""Ch4 relation-operator capacity experiments.

This standalone runner implements fast controlled experiments for the additive
relation-operator teacher

    y_t = (1 / sqrt(R)) sum_r W_r x_{j(t,r)}.

The core controls separate support, scalar structural routing, and explicit
relation-conditioned value transport.  Practical model adapters are guarded by
official-backend imports so paper runs do not silently fall back to local
approximations.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib
import importlib.util
import json
import math
import os
import random
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


R_SWEEP = (1, 2, 4, 8, 12, 16)
DEFAULT_SEEDS = (1001, 1002, 1003)
CONTROLLED_MODELS = ("routing_dense", "full_dense", "routing_1hop", "full_1hop")
PRACTICAL_MODELS = ("graphormer_manual", "graphgps_official", "grit_official", "gatedgcn_plus_official")
OFFICIAL_GRIT_MODELS = ("grit_official", "grit_1hop_official")
CAPACITY_CROSSOVER_MODELS = (
    "capacity_routing_only",
    "capacity_additive_value_bias_routed",
    "capacity_multiplicative_value_gate",
    "capacity_multiplicative_value_gate_routed",
    "capacity_full_relation_transport",
)
LEGACY_CAPACITY_MODELS = ("capacity_transport_only", "capacity_additive_value_bias")
CAPACITY_MODEL_NAMES = tuple(dict.fromkeys(CAPACITY_CROSSOVER_MODELS + LEGACY_CAPACITY_MODELS))
ALL_MODELS = tuple(dict.fromkeys(CONTROLLED_MODELS + PRACTICAL_MODELS + OFFICIAL_GRIT_MODELS + CAPACITY_MODEL_NAMES))
TASK_MODES = ("local", "global")
PARAMETER_MATCH_WIDTHS = (32, 48, 64, 96, 128, 192)
EPS = 1.0e-12

MODEL_LABELS = {
    "routing_dense": "Routing-only dense",
    "full_dense": "Full-GT dense",
    "routing_1hop": "Routing-only 1-hop",
    "full_1hop": "Full-GT 1-hop",
    "graphormer_manual": "Graphormer-manual",
    "graphgps_official": "Official GraphGPS",
    "grit_official": "Official GRIT dense",
    "grit_1hop_official": "Official GRIT 1-hop",
    "gatedgcn_plus_official": "Official GatedGCN+",
    "capacity_routing_only": "Routing-only",
    "capacity_transport_only": "Transport-only",
    "capacity_additive_value_bias": "Additive value bias",
    "capacity_multiplicative_value_gate": "Multiplicative value gate only",
    "capacity_additive_value_bias_routed": "Routing + additive value bias",
    "capacity_multiplicative_value_gate_routed": "Routing + multiplicative value gate",
    "capacity_full_relation_transport": "Full relation transport",
}

MODEL_COLORS = {
    "routing_dense": "#4969a8",
    "routing_1hop": "#86a6d9",
    "full_dense": "#c2473f",
    "full_1hop": "#e38a7f",
    "graphormer_manual": "#5f6b7a",
    "graphgps_official": "#4f8f5b",
    "grit_official": "#8b5fbf",
    "grit_1hop_official": "#b89ad9",
    "gatedgcn_plus_official": "#c4862f",
    "capacity_routing_only": "#4969a8",
    "capacity_transport_only": "#4f8f5b",
    "capacity_additive_value_bias": "#7f7f7f",
    "capacity_multiplicative_value_gate": "#c4862f",
    "capacity_additive_value_bias_routed": "#4d4d4d",
    "capacity_multiplicative_value_gate_routed": "#9c6b1f",
    "capacity_full_relation_transport": "#c2473f",
}

CAPACITY_MODEL_DESCRIPTIONS = {
    "capacity_routing_only": "Relation labels affect only scalar attention scores; values use shared linear maps.",
    "capacity_transport_only": "Attention is unstructured/uniform; relation labels select low-rank value-transport maps.",
    "capacity_additive_value_bias": "Attention is unstructured/uniform; relation labels add a value bias independent of content.",
    "capacity_multiplicative_value_gate": "Attention is unstructured/uniform; relation labels gate transported value channels multiplicatively.",
    "capacity_additive_value_bias_routed": "Relation labels affect scalar attention and add a value bias independent of content.",
    "capacity_multiplicative_value_gate_routed": "Relation labels affect scalar attention and gate transported value channels multiplicatively.",
    "capacity_full_relation_transport": "Relation labels affect scalar attention and low-rank value-transport maps.",
}


@dataclass(frozen=True)
class ExperimentSpec:
    relation_types: int = 16
    input_dim: int = 32
    target_dim: int = 32
    hidden_dim: int = 64
    heads: int = 4
    transport_bases: int = 4
    layers: int = 1
    task_mode: str = "local"
    noise_nodes: int = 0
    noise_sigma: float = 0.0
    train_size: int = 8192
    val_size: int = 2048
    test_size: int = 2048
    data_seed: int = 7101

    @property
    def feature_dim(self) -> int:
        # content, target marker, content-node marker
        return self.input_dim + 2

    @property
    def num_nodes(self) -> int:
        return 1 + self.relation_types + max(0, self.noise_nodes)


@dataclass
class RelationBatch:
    x: torch.Tensor
    y: torch.Tensor
    content: torch.Tensor
    pair_rel: torch.Tensor
    support_dense: torch.Tensor
    support_sparse: torch.Tensor
    node_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_rel: torch.Tensor

    def to(self, device: torch.device) -> "RelationBatch":
        return RelationBatch(**{field.name: getattr(self, field.name).to(device) for field in dataclasses.fields(self)})


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_root() -> Path:
    return Path.cwd() / "artifacts" / "ch4_relation_operator_capacity"


def parse_csv_list(text: str, allowed: Sequence[str] | None = None) -> list[str]:
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    if allowed is not None:
        bad = [value for value in values if value not in allowed]
        if bad:
            raise argparse.ArgumentTypeError(f"unknown values {bad}; expected one of {list(allowed)}")
    return values


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    ensure_dir(path.parent)
    keys: list[str] = []
    if fieldnames is None:
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relation_weights(spec: ExperimentSpec) -> torch.Tensor:
    gen = torch.Generator().manual_seed(int(spec.data_seed) + 19 * int(spec.relation_types))
    weights = []
    for _ in range(spec.relation_types):
        mat = torch.randn(spec.target_dim, spec.input_dim, generator=gen)
        u, _, vh = torch.linalg.svd(mat, full_matrices=False)
        weights.append(u @ vh)
    return torch.stack(weights, dim=0)


def graph_tensors(spec: ExperimentSpec) -> dict[str, torch.Tensor]:
    n = spec.num_nodes
    r = spec.relation_types
    pair_rel = torch.zeros(n, n, dtype=torch.long)
    support_dense = torch.zeros(n, n, dtype=torch.bool)
    support_sparse = torch.zeros(n, n, dtype=torch.bool)
    # Receiver row 0 attends to relation sources 1..R.  Dense support also sees
    # irrelevant content/noise nodes with null relation.
    support_dense[0, 1:n] = True
    for rel in range(1, r + 1):
        node = rel
        pair_rel[0, node] = rel
        if spec.task_mode == "local":
            support_sparse[0, node] = True
    if spec.task_mode not in TASK_MODES:
        raise ValueError(f"unknown task_mode {spec.task_mode!r}")
    src, dst = torch.where(support_sparse)
    edge_index = torch.stack([dst, src], dim=0) if src.numel() else torch.empty(2, 0, dtype=torch.long)
    edge_rel = pair_rel[src, dst] if src.numel() else torch.empty(0, dtype=torch.long)
    return {
        "pair_rel": pair_rel,
        "support_dense": support_dense,
        "support_sparse": support_sparse,
        "edge_index": edge_index,
        "edge_rel": edge_rel,
        "node_mask": torch.ones(n, dtype=torch.bool),
    }


def make_split(spec: ExperimentSpec, split_size: int, split_seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(int(split_seed))
    graph = graph_tensors(spec)
    weights = relation_weights(spec)
    n = spec.num_nodes
    content = torch.zeros(split_size, n, spec.input_dim, dtype=torch.float32)
    content[:, 1:, :] = torch.randn(split_size, n - 1, spec.input_dim, generator=gen)
    if spec.noise_nodes and spec.noise_sigma != 1.0:
        noise_start = 1 + spec.relation_types
        content[:, noise_start:, :] *= float(spec.noise_sigma)
    x = torch.zeros(split_size, n, spec.feature_dim, dtype=torch.float32)
    x[..., : spec.input_dim] = content
    x[:, 0, spec.input_dim] = 1.0
    x[:, 1:, spec.input_dim + 1] = 1.0
    source_content = content[:, 1 : spec.relation_types + 1, :]
    y = torch.einsum("rtd,brd->bt", weights, source_content) / math.sqrt(float(spec.relation_types))
    return {
        "x": x,
        "content": content,
        "y": y,
        "teacher_weights": weights,
        **graph,
    }


def dataset_path(root: Path, spec: ExperimentSpec) -> Path:
    noise = f"noise{spec.noise_nodes}_sig{float(spec.noise_sigma):g}".replace(".", "p")
    return (
        root
        / "data"
        / f"{spec.task_mode}_R{spec.relation_types}_d{spec.input_dim}_{noise}_seed{spec.data_seed}.pt"
    )


def save_dataset(root: Path, spec: ExperimentSpec, overwrite: bool = False) -> Path:
    path = dataset_path(root, spec)
    if path.exists() and not overwrite:
        print(f"[data] using cache {path}")
        return path
    payload = {
        "spec": asdict(spec),
        "splits": {
            "train": make_split(spec, spec.train_size, spec.data_seed + 101),
            "val": make_split(spec, spec.val_size, spec.data_seed + 211),
            "test": make_split(spec, spec.test_size, spec.data_seed + 307),
        },
    }
    ensure_dir(path.parent)
    torch.save(payload, path)
    print(f"[data] wrote {path}")
    return path


def load_dataset(path: Path) -> tuple[ExperimentSpec, dict[str, dict[str, torch.Tensor]]]:
    payload = torch.load(path, map_location="cpu")
    return ExperimentSpec(**payload["spec"]), payload["splits"]


def batch_from_split(split: Mapping[str, torch.Tensor], indices: torch.Tensor) -> RelationBatch:
    bsz = int(indices.numel())
    pair_rel = split["pair_rel"].unsqueeze(0).expand(bsz, -1, -1)
    support_dense = split["support_dense"].unsqueeze(0).expand(bsz, -1, -1)
    support_sparse = split["support_sparse"].unsqueeze(0).expand(bsz, -1, -1)
    node_mask = split["node_mask"].unsqueeze(0).expand(bsz, -1)
    return RelationBatch(
        x=split["x"][indices],
        y=split["y"][indices],
        content=split["content"][indices],
        pair_rel=pair_rel,
        support_dense=support_dense,
        support_sparse=support_sparse,
        node_mask=node_mask,
        edge_index=split["edge_index"],
        edge_rel=split["edge_rel"],
    )


def iter_batches(split: Mapping[str, torch.Tensor], batch_size: int, generator: torch.Generator) -> Iterable[RelationBatch]:
    n = int(split["x"].shape[0])
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        yield batch_from_split(split, order[start : start + batch_size])


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits.masked_fill(~mask, -1.0e9)
    out = torch.softmax(logits, dim=dim)
    out = out * mask.to(out.dtype)
    return out / out.sum(dim=dim, keepdim=True).clamp_min(EPS)


class ControlledRelationModel(nn.Module):
    def __init__(self, spec: ExperimentSpec, *, full_transport: bool, dense_support: bool) -> None:
        super().__init__()
        self.spec = spec
        self.full_transport = bool(full_transport)
        self.dense_support = bool(dense_support)
        h = int(spec.heads)
        a = int(spec.transport_bases)
        d = int(spec.input_dim)
        self.routing_bias = nn.Parameter(torch.zeros(h, spec.relation_types + 1))
        self.routing_basis = nn.Parameter(torch.randn(h, d, d) / math.sqrt(d))
        self.transport_basis = nn.Parameter(torch.randn(h, a, d, d) / math.sqrt(d))
        self.transport_coeff = nn.Parameter(torch.randn(spec.relation_types + 1, h, a) * 0.02)
        self.output_head = nn.Linear(d, int(spec.target_dim))
        if int(spec.target_dim) == d:
            with torch.no_grad():
                self.output_head.weight.copy_(torch.eye(d))
                self.output_head.bias.zero_()

    def layer(self, state: torch.Tensor, pair_rel: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        bsz, n, d = state.shape
        h = int(self.spec.heads)
        logits = self.routing_bias[:, pair_rel].permute(1, 0, 2, 3)
        attn = masked_softmax(logits, support[:, None, :, :])
        if self.full_transport:
            coeff = self.transport_coeff[pair_rel]  # [B,N,N,H,A]
            ctx = state.new_zeros((bsz, h, n, d))
            # Contract over the small transport-basis axis without materialising
            # a [B,N,N,H,D,D] relation kernel for every query/source pair.
            for basis_idx in range(int(self.spec.transport_bases)):
                basis_msg = torch.einsum("bjd,hdo->bhjo", state, self.transport_basis[:, basis_idx])
                pair_weight = attn * coeff[..., basis_idx].permute(0, 3, 1, 2)
                ctx = ctx + torch.einsum("bhij,bhjo->bhio", pair_weight, basis_msg)
            return state + ctx.sum(dim=1)
        else:
            msg_per_head = torch.einsum("bjd,hdo->bhjo", state, self.routing_basis)
            msg = torch.einsum("bhij,bhjo->bhio", attn, msg_per_head)
            return state + msg.sum(dim=1)

    def forward(self, batch: RelationBatch) -> torch.Tensor:
        state = batch.content
        support = batch.support_dense if self.dense_support else batch.support_sparse
        for _ in range(int(self.spec.layers)):
            state = self.layer(state, batch.pair_rel, support)
        return self.output_head(state[:, 0, :])


class CapacityRelationModel(nn.Module):
    """Dense global capacity variants for the clean relation-rank crossover.

    These variants intentionally differ only in where the relation label enters:
    scalar routing, value transport, additive value bias, diagonal value gating,
    or scalar routing plus value transport.  The task uses dense support so the
    experiment isolates relation-conditioned capacity rather than reach.
    """

    def __init__(self, spec: ExperimentSpec, *, variant: str) -> None:
        super().__init__()
        if variant not in CAPACITY_MODEL_NAMES:
            raise ValueError(f"unknown capacity variant {variant!r}")
        self.spec = spec
        self.variant = variant
        h = int(spec.heads)
        a = int(spec.transport_bases)
        d = int(spec.input_dim)
        r = int(spec.relation_types)
        self.routing_bias = nn.Parameter(torch.zeros(h, r + 1))
        self.routing_basis = nn.Parameter(torch.randn(h, d, d) / math.sqrt(d))
        self.transport_basis = nn.Parameter(torch.randn(h, a, d, d) / math.sqrt(d))
        self.transport_coeff = nn.Parameter(torch.randn(r + 1, h, a) * 0.02)
        self.additive_value_bias = nn.Parameter(torch.zeros(r + 1, h, d))
        self.value_gate = nn.Parameter(torch.ones(r + 1, h, d))
        self.output_head = nn.Linear(d, int(spec.target_dim))
        if int(spec.target_dim) == d:
            with torch.no_grad():
                self.output_head.weight.copy_(torch.eye(d))
                self.output_head.bias.zero_()

    def relation_attention(self, pair_rel: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        bsz, n, _ = pair_rel.shape
        routed_variants = {
            "capacity_routing_only",
            "capacity_additive_value_bias_routed",
            "capacity_multiplicative_value_gate_routed",
            "capacity_full_relation_transport",
        }
        if self.variant in routed_variants:
            logits = self.routing_bias[:, pair_rel].permute(1, 0, 2, 3)
        else:
            logits = torch.zeros((bsz, int(self.spec.heads), n, n), dtype=torch.float32, device=pair_rel.device)
        return masked_softmax(logits, support[:, None, :, :])

    def shared_value_message(self, state: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        msg_per_head = torch.einsum("bjd,hdo->bhjo", state, self.routing_basis)
        return torch.einsum("bhij,bhjo->bhio", attn, msg_per_head)

    def low_rank_transport_message(self, state: torch.Tensor, pair_rel: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        bsz, _, d = state.shape
        h = int(self.spec.heads)
        ctx = state.new_zeros((bsz, h, pair_rel.shape[1], d))
        coeff = self.transport_coeff[pair_rel]
        for basis_idx in range(int(self.spec.transport_bases)):
            basis_msg = torch.einsum("bjd,hdo->bhjo", state, self.transport_basis[:, basis_idx])
            pair_weight = attn * coeff[..., basis_idx].permute(0, 3, 1, 2)
            ctx = ctx + torch.einsum("bhij,bhjo->bhio", pair_weight, basis_msg)
        return ctx

    def layer(self, state: torch.Tensor, pair_rel: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        attn = self.relation_attention(pair_rel, support)
        if self.variant == "capacity_routing_only":
            ctx = self.shared_value_message(state, attn)
        elif self.variant == "capacity_transport_only":
            ctx = self.low_rank_transport_message(state, pair_rel, attn)
        elif self.variant in {"capacity_additive_value_bias", "capacity_additive_value_bias_routed"}:
            shared = torch.einsum("bjd,hdo->bhjo", state, self.routing_basis)
            bias = self.additive_value_bias[pair_rel].permute(0, 3, 1, 2, 4)
            msg = shared[:, :, None, :, :] + bias
            ctx = torch.einsum("bhij,bhijd->bhid", attn, msg)
        elif self.variant in {"capacity_multiplicative_value_gate", "capacity_multiplicative_value_gate_routed"}:
            shared = torch.einsum("bjd,hdo->bhjo", state, self.routing_basis)
            gate = self.value_gate[pair_rel].permute(0, 3, 1, 2, 4)
            msg = shared[:, :, None, :, :] * gate
            ctx = torch.einsum("bhij,bhijd->bhid", attn, msg)
        elif self.variant == "capacity_full_relation_transport":
            ctx = self.low_rank_transport_message(state, pair_rel, attn)
        else:
            raise ValueError(f"unknown capacity variant {self.variant!r}")
        return state + ctx.sum(dim=1)

    def forward(self, batch: RelationBatch) -> torch.Tensor:
        state = batch.content
        for _ in range(int(self.spec.layers)):
            state = self.layer(state, batch.pair_rel, batch.support_dense)
        return self.output_head(state[:, 0, :])


class GraphormerManualModel(nn.Module):
    def __init__(self, spec: ExperimentSpec) -> None:
        super().__init__()
        if spec.hidden_dim % spec.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.spec = spec
        self.head_dim = spec.hidden_dim // spec.heads
        self.input = nn.Linear(spec.feature_dim, spec.hidden_dim)
        self.qkv = nn.Linear(spec.hidden_dim, 3 * spec.hidden_dim, bias=False)
        self.pair_bias = nn.Embedding(spec.relation_types + 1, spec.heads)
        self.out = nn.Linear(spec.hidden_dim, spec.hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(spec.hidden_dim),
            nn.Linear(spec.hidden_dim, 2 * spec.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * spec.hidden_dim, spec.hidden_dim),
        )
        self.head = nn.Sequential(nn.LayerNorm(spec.hidden_dim), nn.Linear(spec.hidden_dim, spec.target_dim))

    def forward(self, batch: RelationBatch) -> torch.Tensor:
        h = self.input(batch.x)
        bsz, n, _ = h.shape
        support = batch.support_dense
        for _ in range(int(self.spec.layers)):
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            q = q.view(bsz, n, self.spec.heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, n, self.spec.heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, n, self.spec.heads, self.head_dim).transpose(1, 2)
            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            logits = logits + self.pair_bias(batch.pair_rel).permute(0, 3, 1, 2)
            attn = masked_softmax(logits, support[:, None, :, :])
            ctx = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, n, self.spec.hidden_dim)
            h = h + self.out(ctx)
            h = h + self.ffn(h)
        return self.head(h[:, 0, :])


def external_repo_paths(env_name: str, candidates: Sequence[str]) -> list[Path]:
    paths = []
    if os.environ.get(env_name):
        paths.append(Path(os.environ[env_name]))
    cwd = Path.cwd().resolve()
    for name in candidates:
        paths.append(cwd / "external" / name)
        paths.append(cwd / "graphbench-algoreas-hpc" / "external" / name)
    return paths


def add_external_repo_path(env_name: str, candidates: Sequence[str]) -> None:
    for path in external_repo_paths(env_name, candidates):
        if path.exists():
            text = str(path.resolve())
            if text not in sys.path:
                sys.path.insert(0, text)
            return


def resolve_external_repo_path(env_name: str, candidates: Sequence[str]) -> Path:
    for path in external_repo_paths(env_name, candidates):
        if path.exists():
            return path.resolve()
    checked = ", ".join(str(path) for path in external_repo_paths(env_name, candidates))
    raise RuntimeError(f"official backend path for {env_name} was not found; checked {checked}")


def allow_graphgym_duplicate_registration() -> None:
    try:
        graphgym_register = importlib.import_module("torch_geometric.graphgym.register")
    except Exception:
        return
    if getattr(graphgym_register, "_ch4_duplicate_registration_ok", False):
        return
    original_register_base = graphgym_register.register_base

    def register_base_idempotent(mapping: dict[str, Any], key: str, module: Any) -> None:
        if key in mapping:
            return
        return original_register_base(mapping, key, module)

    graphgym_register.register_base = register_base_idempotent
    graphgym_register._ch4_duplicate_registration_ok = True


def require_official_import(module_name: str, env_name: str, candidates: Sequence[str]):
    add_external_repo_path(env_name, candidates)
    allow_graphgym_duplicate_registration()
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        checked = ", ".join(str(path) for path in external_repo_paths(env_name, candidates))
        raise RuntimeError(f"official backend import failed for {module_name!r}; checked {checked}") from exc


def require_official_file_module(module_name: str, path: Path):
    path = path.resolve()
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not path.exists():
        raise RuntimeError(f"official backend file is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load official backend file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    allow_graphgym_duplicate_registration()
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def require_official_file_exists(path: Path) -> Path:
    path = path.resolve()
    if not path.exists():
        raise RuntimeError(f"official backend file is missing: {path}")
    return path


def setup_graphgym_activation(act: str = "relu") -> None:
    try:
        graphgym_config = importlib.import_module("torch_geometric.graphgym.config")
        graphgym_register = importlib.import_module("torch_geometric.graphgym.register")
        yacs_config = importlib.import_module("yacs.config")
    except Exception:
        return
    allow_graphgym_duplicate_registration()
    cfg = graphgym_config.cfg
    if hasattr(cfg, "defrost"):
        cfg.defrost()
    if hasattr(cfg, "set_new_allowed"):
        cfg.set_new_allowed(True)
    if not hasattr(cfg, "gnn"):
        cfg.gnn = yacs_config.CfgNode(new_allowed=True)
    elif hasattr(cfg.gnn, "set_new_allowed"):
        cfg.gnn.set_new_allowed(True)
    cfg.gnn.act = str(act)
    if str(act) not in graphgym_register.act_dict:
        graphgym_register.act_dict[str(act)] = getattr(torch.nn, "ReLU")


def support_edge_triples(
    batch: RelationBatch,
    *,
    dense_support: bool,
    include_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, n, _ = batch.x.shape
    device = batch.x.device
    support = batch.support_dense if dense_support else batch.support_sparse
    support0 = support[0].clone()
    if include_self_loops:
        diag = torch.arange(n, device=device)
        support0[diag, diag] = True
    receiver, sender = torch.where(support0)
    offsets = torch.arange(bsz, device=device, dtype=torch.long) * n
    src = (sender.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
    dst = (receiver.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
    rel = batch.pair_rel[0, receiver, sender].repeat(bsz)
    return src, dst, rel.long()


def relation_attn_bias(batch: RelationBatch, embedding: nn.Embedding, heads: int) -> torch.Tensor:
    bsz, n, _ = batch.x.shape
    bias = embedding(batch.pair_rel.long()).permute(0, 3, 1, 2).contiguous()
    return bias.view(bsz * int(heads), n, n)


def load_official_grit_layer_module():
    grit_root = resolve_external_repo_path("GRIT_ROOT", ("GRIT",))
    grit_pkg = grit_root / "grit" if (grit_root / "grit").exists() else grit_root
    require_official_file_exists(grit_pkg / "utils.py")
    require_official_file_exists(grit_pkg / "layer" / "grit_layer.py")

    # Avoid GRIT's package-level __init__.py because it imports GraphGym config
    # decorators that are not robust in this environment.  We still execute the
    # official utility and layer source files.
    grit_stub = types.ModuleType("grit")
    grit_stub.__path__ = [str(grit_pkg)]
    sys.modules["grit"] = grit_stub
    require_official_file_module("grit.utils", grit_pkg / "utils.py")
    return require_official_file_module("_ch4_official_grit_layer", grit_pkg / "layer" / "grit_layer.py")


def official_grit_layer_cfg():
    yacs_config = require_official_import("yacs.config", "YACS", ())
    cn = yacs_config.CfgNode
    cfg = cn(new_allowed=True)
    cfg.update_e = True
    cfg.bn_momentum = 0.1
    cfg.bn_no_runner = False
    cfg.rezero = False
    cfg.attn = cn(new_allowed=True)
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
    return cfg


class OfficialGRITRelationModel(nn.Module):
    """Actual official GRIT layer body for the Ch4 relation-operator task."""

    def __init__(self, spec: ExperimentSpec, *, dense_support: bool) -> None:
        super().__init__()
        if int(spec.hidden_dim) % int(spec.heads):
            raise ValueError("hidden_dim must be divisible by heads for official GRIT")
        grit_layer_mod = load_official_grit_layer_module()
        self.spec = spec
        self.dense_support = bool(dense_support)
        dim = int(spec.hidden_dim)
        self.input_encoder = nn.Linear(spec.feature_dim, dim)
        self.edge_encoder = nn.Embedding(spec.relation_types + 1, dim)
        cfg = official_grit_layer_cfg()
        self.layers = nn.ModuleList(
            grit_layer_mod.GritTransformerLayer(
                dim,
                dim,
                int(spec.heads),
                dropout=0.0,
                attn_dropout=0.0,
                layer_norm=True,
                batch_norm=False,
                residual=True,
                act="relu",
                norm_e=True,
                O_e=True,
                cfg=cfg,
            )
            for _ in range(int(spec.layers))
        )
        self.output_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, int(spec.target_dim)))

    def _support_for_batch(self, batch: RelationBatch) -> torch.Tensor:
        bsz, n, _ = batch.content.shape
        if self.dense_support:
            support = torch.ones((bsz, n, n), dtype=torch.bool, device=batch.content.device)
        else:
            support = batch.support_sparse.clone()
        diag = torch.arange(n, device=batch.content.device)
        support[:, diag, diag] = True
        return support

    def _pyg_batch(self, batch: RelationBatch):
        data_mod = require_official_import("torch_geometric.data", "torch_geometric", ())
        bsz, n, _ = batch.x.shape
        device = batch.x.device
        src, dst, rel = support_edge_triples(batch, dense_support=self.dense_support, include_self_loops=True)

        data = data_mod.Data(num_nodes=bsz * n)
        data.x = self.input_encoder(batch.x.reshape(bsz * n, -1).float())
        data.edge_index = torch.stack([src, dst], dim=0)
        data.edge_attr = self.edge_encoder(rel.long())
        data.batch = torch.arange(bsz, device=device).repeat_interleave(n)
        data.deg = torch.bincount(dst, minlength=bsz * n).float()
        data.log_deg = torch.log(data.deg + 1.0).view(-1, 1)
        return data

    def forward(self, batch: RelationBatch) -> torch.Tensor:
        pyg_batch = self._pyg_batch(batch)
        for layer in self.layers:
            pyg_batch = layer(pyg_batch)
        states = pyg_batch.x.view(batch.x.shape[0], batch.x.shape[1], -1)
        return self.output_head(states[:, 0, :])


class OfficialGraphGPSRelationModel(nn.Module):
    """Actual official GPSLayer for relation-only Ch4 comparisons."""

    def __init__(self, spec: ExperimentSpec) -> None:
        super().__init__()
        if int(spec.hidden_dim) % int(spec.heads):
            raise ValueError("hidden_dim must be divisible by heads for official GraphGPS")
        setup_graphgym_activation("relu")
        gps_layer_mod = require_official_import("graphgps.layer.gps_layer", "GRAPHGPS_ROOT", ("GraphGPS",))
        self.spec = spec
        dim = int(spec.hidden_dim)
        self.input_encoder = nn.Linear(spec.feature_dim, dim)
        self.edge_encoder = nn.Embedding(spec.relation_types + 1, dim)
        self.attn_bias_encoder = nn.Embedding(spec.relation_types + 1, int(spec.heads))
        self.layers = nn.ModuleList(
            gps_layer_mod.GPSLayer(
                dim_h=dim,
                local_gnn_type="GINE",
                global_model_type="BiasedTransformer",
                num_heads=int(spec.heads),
                act="relu",
                pna_degrees=None,
                equivstable_pe=False,
                dropout=0.0,
                attn_dropout=0.0,
                layer_norm=True,
                batch_norm=False,
                bigbird_cfg=None,
                log_attn_weights=False,
            )
            for _ in range(int(spec.layers))
        )
        self.output_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, int(spec.target_dim)))

    def forward(self, batch: RelationBatch) -> torch.Tensor:
        data_mod = require_official_import("torch_geometric.data", "torch_geometric", ())
        bsz, n, _ = batch.x.shape
        src, dst, rel = support_edge_triples(batch, dense_support=False, include_self_loops=True)
        data = data_mod.Data(num_nodes=bsz * n)
        data.x = self.input_encoder(batch.x.reshape(bsz * n, -1).float())
        data.edge_index = torch.stack([src, dst], dim=0)
        data.edge_attr = self.edge_encoder(rel)
        data.batch = torch.arange(bsz, device=batch.x.device).repeat_interleave(n)
        data.attn_bias = relation_attn_bias(batch, self.attn_bias_encoder, int(self.spec.heads)).to(data.x.dtype)
        for layer in self.layers:
            data = layer(data)
        states = data.x.view(bsz, n, -1)
        return self.output_head(states[:, 0, :])


def load_official_gnnplus_gatedgcn_module():
    setup_graphgym_activation("relu")
    gnnplus_root = resolve_external_repo_path("GNNPLUS_ROOT", ("GNNPlus",))
    gnnplus_pkg = gnnplus_root / "GNNPlus" if (gnnplus_root / "GNNPlus").exists() else gnnplus_root
    return require_official_file_module(
        "_ch4_official_gnnplus_gatedgcn_layer",
        gnnplus_pkg / "layer" / "gatedgcn_layer.py",
    )


class OfficialGNNPlusGatedGCNRelationModel(nn.Module):
    """Actual official GNN+ GatedGCN layer for sparse relation-only Ch4 comparisons."""

    def __init__(self, spec: ExperimentSpec) -> None:
        super().__init__()
        gated_mod = load_official_gnnplus_gatedgcn_module()
        self.spec = spec
        dim = int(spec.hidden_dim)
        self.input_encoder = nn.Linear(spec.feature_dim, dim)
        self.edge_encoder = nn.Embedding(spec.relation_types + 1, dim)
        self.layers = nn.ModuleList(
            gated_mod.GatedGCNLayer(
                dim,
                dim,
                dropout=0.0,
                residual=True,
                ffn=True,
                act="relu",
                equivstable_pe=False,
            )
            for _ in range(int(spec.layers))
        )
        self.output_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, int(spec.target_dim)))

    def forward(self, batch: RelationBatch) -> torch.Tensor:
        data_mod = require_official_import("torch_geometric.data", "torch_geometric", ())
        bsz, n, _ = batch.x.shape
        src, dst, rel = support_edge_triples(batch, dense_support=False, include_self_loops=True)
        data = data_mod.Data(num_nodes=bsz * n)
        data.x = self.input_encoder(batch.x.reshape(bsz * n, -1).float())
        data.edge_index = torch.stack([src, dst], dim=0)
        data.edge_attr = self.edge_encoder(rel)
        data.batch = torch.arange(bsz, device=batch.x.device).repeat_interleave(n)
        for layer in self.layers:
            data = layer(data)
        states = data.x.view(bsz, n, -1)
        return self.output_head(states[:, 0, :])


def build_model(spec: ExperimentSpec, model_name: str) -> nn.Module:
    if model_name in CAPACITY_MODEL_NAMES:
        return CapacityRelationModel(spec, variant=model_name)
    if model_name == "routing_dense":
        return ControlledRelationModel(spec, full_transport=False, dense_support=True)
    if model_name == "full_dense":
        return ControlledRelationModel(spec, full_transport=True, dense_support=True)
    if model_name == "routing_1hop":
        return ControlledRelationModel(spec, full_transport=False, dense_support=False)
    if model_name == "full_1hop":
        return ControlledRelationModel(spec, full_transport=True, dense_support=False)
    if model_name == "graphormer_manual":
        return GraphormerManualModel(spec)
    if model_name == "grit_official":
        return OfficialGRITRelationModel(spec, dense_support=True)
    if model_name == "grit_1hop_official":
        return OfficialGRITRelationModel(spec, dense_support=False)
    if model_name == "graphgps_official":
        return OfficialGraphGPSRelationModel(spec)
    if model_name == "gatedgcn_plus_official":
        return OfficialGNNPlusGatedGCNRelationModel(spec)
    raise ValueError(f"unknown model {model_name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parameter_match_grid(
    spec: ExperimentSpec,
    model_name: str,
    *,
    target_params: int,
    width_grid: Sequence[int],
) -> tuple[ExperimentSpec, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for width in width_grid:
        candidate = dataclasses.replace(spec, hidden_dim=int(width))
        params = int(count_parameters(build_model(candidate, model_name)))
        row = {
            "model": model_name,
            "relation_types": int(spec.relation_types),
            "task_mode": spec.task_mode,
            "layers": int(spec.layers),
            "candidate_hidden_dim": int(width),
            "target_parameters": int(target_params),
            "candidate_parameters": params,
            "abs_parameter_gap": abs(params - int(target_params)),
            "selected": False,
        }
        rows.append(row)
        if best is None or row["abs_parameter_gap"] < best["abs_parameter_gap"]:
            best = row
    assert best is not None
    for row in rows:
        row["selected"] = bool(row is best)
    return dataclasses.replace(spec, hidden_dim=int(best["candidate_hidden_dim"])), rows


def maybe_parameter_matched_spec(
    root: Path,
    spec: ExperimentSpec,
    model_name: str,
    *,
    width_grid: Sequence[int],
) -> ExperimentSpec:
    if model_name not in {"graphormer_manual", "graphgps_official", "gatedgcn_plus_official"}:
        return spec
    target_params = int(count_parameters(build_model(spec, "full_dense")))
    matched, rows = parameter_match_grid(spec, model_name, target_params=target_params, width_grid=width_grid)
    path = ensure_dir(root / "configs") / "parameter_matching_grid.csv"
    existing = read_csv(path) if path.exists() else []
    keyed: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in [*existing, *rows]:
        key = (
            str(row["model"]),
            str(row["relation_types"]),
            str(row["task_mode"]),
            str(row["layers"]),
            str(row["candidate_hidden_dim"]),
        )
        keyed[key] = row
    write_csv(path, list(keyed.values()))
    print(
        f"[match] {model_name} target={target_params:,} "
        f"hidden_dim={matched.hidden_dim} for R={spec.relation_types} {spec.task_mode}"
    )
    return matched


@torch.no_grad()
def evaluate_model(model: nn.Module, split: Mapping[str, torch.Tensor], device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    n = int(split["x"].shape[0])
    mse_sum = 0.0
    denom_sum = 0.0
    mae_sum = 0.0
    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n))
        batch = batch_from_split(split, idx).to(device)
        pred = model(batch)
        target = batch.y
        mse_sum += float(((pred - target) ** 2).sum().detach().cpu())
        denom_sum += float((target**2).sum().detach().cpu())
        mae_sum += float((pred - target).abs().sum().detach().cpu())
    return {
        "rel_mse": mse_sum / max(denom_sum, EPS),
        "mse": mse_sum / max(n, 1),
        "mae": mae_sum / max(n, 1),
    }


@torch.no_grad()
def effective_relation_maps(model: nn.Module, spec: ExperimentSpec, device: torch.device) -> torch.Tensor:
    model.eval()
    maps = torch.zeros(spec.relation_types, spec.target_dim, spec.input_dim, device=device)
    base = make_split(spec, 1, spec.data_seed + 9999)
    base["content"].zero_()
    base["x"].zero_()
    base["x"][:, 0, spec.input_dim] = 1.0
    base["x"][:, 1:, spec.input_dim + 1] = 1.0
    baseline = model(batch_from_split(base, torch.tensor([0])).to(device))[0]
    for rel in range(1, spec.relation_types + 1):
        for dim in range(spec.input_dim):
            split = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in base.items()}
            split["content"][0, rel, dim] = 1.0
            split["x"][0, rel, dim] = 1.0
            batch = batch_from_split(split, torch.tensor([0])).to(device)
            maps[rel - 1, :, dim] = model(batch)[0] - baseline
    return maps.detach().cpu()


def realised_rank(maps: torch.Tensor, tol: float = 1.0e-4) -> tuple[int, list[float]]:
    mat = maps.reshape(maps.shape[0], -1).double()
    s = torch.linalg.svdvals(mat)
    threshold = max(float(s.max()) * tol, tol)
    return int((s > threshold).sum().item()), [float(v) for v in s]


def singular_values_from_text(text: Any) -> list[float]:
    return [float(item) for item in str(text or "").split(";") if item]


def numerical_rank_from_singular_values(values: Sequence[float], tol: float = 1.0e-4) -> float:
    if not values:
        return float("nan")
    smax = max(abs(float(value)) for value in values)
    threshold = max(smax * float(tol), float(tol))
    return float(sum(abs(float(value)) > threshold for value in values))


def teacher_floor(spec: ExperimentSpec) -> dict[str, Any]:
    weights = relation_weights(spec) / math.sqrt(float(spec.relation_types))
    mat = weights.reshape(spec.relation_types, -1).double()
    s = torch.linalg.svdvals(mat)
    total = float((s**2).sum().item())
    h_floor = float((s[int(spec.heads) :] ** 2).sum().item() / max(total, EPS))
    ht_floor = float((s[int(spec.heads * spec.transport_bases) :] ** 2).sum().item() / max(total, EPS))
    return {
        "routing_floor": h_floor,
        "full_transport_floor": ht_floor,
        "teacher_rank": int((s > 1.0e-8).sum().item()),
        "teacher_singular_values": ";".join(f"{float(v):.8g}" for v in s),
    }


def run_id(experiment: str, spec: ExperimentSpec, model_name: str, seed: int) -> str:
    noise = f"n{spec.noise_nodes}_s{float(spec.noise_sigma):g}".replace(".", "p")
    return (
        f"{experiment}/{spec.task_mode}/R{spec.relation_types}/L{spec.layers}/"
        f"{noise}/{model_name}/seed_{int(seed)}"
    )


def checkpoint_dir(root: Path, experiment: str, spec: ExperimentSpec, model_name: str, seed: int) -> Path:
    return root / "checkpoints" / run_id(experiment, spec, model_name, seed)


def train_one(
    *,
    root: Path,
    experiment: str,
    spec: ExperimentSpec,
    model_name: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    eval_batch_size: int,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
    overwrite: bool,
) -> dict[str, Any]:
    ckpt_dir = checkpoint_dir(root, experiment, spec, model_name, seed)
    complete_path = ckpt_dir / "complete.json"
    if complete_path.exists() and not overwrite:
        print(f"[skip] {run_id(experiment, spec, model_name, seed)}")
        return read_json(complete_path)
    data_file = save_dataset(root, spec, overwrite=False)
    loaded_spec, splits = load_dataset(data_file)
    set_seed(seed)
    model = build_model(loaded_spec, model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    amp_enabled = bool(use_amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    rng = torch.Generator().manual_seed(int(seed) + 17)
    best_val = float("inf")
    best_epoch = -1
    train_rows: list[dict[str, Any]] = []
    ensure_dir(ckpt_dir)
    print(f"[train] {run_id(experiment, spec, model_name, seed)} params={count_parameters(model):,} device={device}")
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_cpu in iter_batches(splits["train"], batch_size, rng):
            batch = batch_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.autocast(
                device_type=autocast_device,
                dtype=torch.bfloat16,
                enabled=bool(use_amp and device.type == "cuda"),
            ):
                pred = model(batch)
                loss = F.mse_loss(pred.float(), batch.y.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu()) * int(batch.x.shape[0])
            count += int(batch.x.shape[0])
        val = evaluate_model(model, splits["val"], device, eval_batch_size)
        train_loss = total / max(count, 1)
        improved = val["rel_mse"] < best_val - 1.0e-8
        if improved:
            best_val = val["rel_mse"]
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "spec": asdict(loaded_spec), "model": model_name}, ckpt_dir / "best.pt")
        train_rows.append({"epoch": epoch, "train_mse": train_loss, "val_rel_mse": val["rel_mse"], "best_epoch": best_epoch})
        if epoch == 1 or improved or epoch % 25 == 0:
            print(
                f"[train] epoch={epoch:03d} train_mse={train_loss:.5g} "
                f"val_rel_mse={val['rel_mse']:.5g} best={best_val:.5g}@{best_epoch}"
            )
        if epoch - best_epoch >= patience:
            print(f"[train] early stop epoch={epoch} best_epoch={best_epoch}")
            break
    torch.save({"state_dict": model.state_dict(), "spec": asdict(loaded_spec), "model": model_name}, ckpt_dir / "final.pt")
    if (ckpt_dir / "best.pt").exists():
        payload = torch.load(ckpt_dir / "best.pt", map_location=device)
        model.load_state_dict(payload["state_dict"])
    test = evaluate_model(model, splits["test"], device, eval_batch_size)
    if isinstance(model, OfficialGRITRelationModel) and experiment == "grit_r_local_diagnostic":
        maps = effective_relation_maps(model, loaded_spec, device)
        rank, singular = realised_rank(maps)
    elif isinstance(model, (OfficialGRITRelationModel, OfficialGraphGPSRelationModel, OfficialGNNPlusGatedGCNRelationModel)):
        rank, singular = -1, []
    else:
        maps = effective_relation_maps(model, loaded_spec, device)
        rank, singular = realised_rank(maps)
    floor = teacher_floor(loaded_spec)
    summary = {
        "experiment": experiment,
        "task_mode": loaded_spec.task_mode,
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "seed": int(seed),
        "relation_types": int(loaded_spec.relation_types),
        "input_dim": int(loaded_spec.input_dim),
        "target_dim": int(loaded_spec.target_dim),
        "hidden_dim": int(loaded_spec.hidden_dim),
        "heads": int(loaded_spec.heads),
        "transport_bases": int(loaded_spec.transport_bases),
        "layers": int(loaded_spec.layers),
        "noise_nodes": int(loaded_spec.noise_nodes),
        "noise_sigma": float(loaded_spec.noise_sigma),
        "parameters": int(count_parameters(model)),
        "best_epoch": int(best_epoch),
        "best_val_rel_mse": float(best_val),
        "test_rel_mse": float(test["rel_mse"]),
        "test_mse": float(test["mse"]),
        "test_mae": float(test["mae"]),
        "realised_rank": int(rank),
        "realised_singular_values": ";".join(f"{v:.8g}" for v in singular),
        **floor,
    }
    write_csv(ckpt_dir / "train_log.csv", train_rows)
    write_json(complete_path, summary)
    print(f"[done] {run_id(experiment, spec, model_name, seed)} rel_mse={test['rel_mse']:.5g} rank={rank}")
    return summary


def scan_summaries(root: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted((root / "checkpoints").glob("**/complete.json"))]


def write_summary_tables(root: Path) -> None:
    rows = scan_summaries(root)
    metrics = root / "metrics"
    write_csv(metrics / "clean_metrics.csv", rows)
    write_csv(
        metrics / "relation_rank_metrics.csv",
        [
            {
                key: row[key]
                for key in [
                    "experiment",
                    "task_mode",
                    "model",
                    "seed",
                    "relation_types",
                    "layers",
                    "noise_nodes",
                    "noise_sigma",
                    "realised_rank",
                    "realised_singular_values",
                ]
                if key in row
            }
            for row in rows
        ],
    )
    write_csv(
        metrics / "eckart_young_floors.csv",
        [
            {
                key: row[key]
                for key in [
                    "experiment",
                    "task_mode",
                    "relation_types",
                    "routing_floor",
                    "full_transport_floor",
                    "teacher_rank",
                    "teacher_singular_values",
                ]
                if key in row
            }
            for row in rows
        ],
    )
    write_csv(
        metrics / "parameter_counts.csv",
        [
            {
                "experiment": row["experiment"],
                "task_mode": row["task_mode"],
                "model": row["model"],
                "relation_types": row["relation_types"],
                "layers": row["layers"],
                "parameters": row["parameters"],
            }
            for row in rows
        ],
    )
    write_csv(metrics / "noise_sweep_metrics.csv", [row for row in rows if row["experiment"] == "overglobalisation"])
    write_csv(metrics / "depth_escape_metrics.csv", [row for row in rows if row["experiment"] == "depth_escape"])
    print(f"[summary] wrote metrics under {metrics}")


def train_grid(
    args: argparse.Namespace,
    *,
    experiment: str,
    specs: Sequence[ExperimentSpec],
    models: Sequence[str],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    root = Path(args.output_root)
    device = resolve_device(args.device)
    rows = []
    for spec in specs:
        save_dataset(root, spec, overwrite=bool(args.overwrite_data))
        for model_name in models:
            matched_spec = maybe_parameter_matched_spec(
                root,
                spec,
                model_name,
                width_grid=parse_int_list(
                    getattr(args, "parameter_match_widths", ",".join(str(v) for v in PARAMETER_MATCH_WIDTHS))
                ),
            )
            for seed in seeds:
                rows.append(
                    train_one(
                        root=root,
                        experiment=experiment,
                        spec=matched_spec,
                        model_name=model_name,
                        seed=seed,
                        device=device,
                        batch_size=int(args.batch_size),
                        eval_batch_size=int(args.eval_batch_size),
                        max_epochs=int(args.max_epochs),
                        patience=int(args.patience),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        use_amp=bool(args.amp),
                        overwrite=bool(args.overwrite_checkpoints),
                    )
                )
    write_summary_tables(root)
    return rows


def base_spec_from_args(args: argparse.Namespace, **overrides: Any) -> ExperimentSpec:
    values = {
        "relation_types": int(args.relation_types),
        "input_dim": int(args.input_dim),
        "target_dim": int(args.target_dim),
        "hidden_dim": int(args.hidden_dim),
        "heads": int(args.heads),
        "transport_bases": int(args.transport_bases),
        "layers": int(args.layers),
        "task_mode": str(args.task_mode),
        "noise_nodes": int(args.noise_nodes),
        "noise_sigma": float(args.noise_sigma),
        "train_size": int(args.train_size),
        "val_size": int(args.val_size),
        "test_size": int(args.test_size),
        "data_seed": int(args.data_seed),
    }
    values.update(overrides)
    return ExperimentSpec(**values)


def run_crossover(args: argparse.Namespace) -> None:
    specs = [
        base_spec_from_args(args, relation_types=r, task_mode="local", noise_nodes=0, noise_sigma=0.0, layers=1)
        for r in parse_int_list(args.r_sweep)
    ]
    train_grid(args, experiment="crossover", specs=specs, models=CONTROLLED_MODELS, seeds=parse_int_list(args.seeds))


def capacity_crossover_specs_from_args(args: argparse.Namespace) -> list[ExperimentSpec]:
    relation_counts = parse_int_list(args.r_sweep)
    total_nodes = int(args.n_nodes)
    min_nodes = 1 + max(relation_counts)
    if total_nodes < min_nodes:
        raise ValueError(
            f"--n-nodes={total_nodes} is too small for max R={max(relation_counts)}; "
            f"need at least {min_nodes} nodes"
        )
    specs = []
    for r in relation_counts:
        # Keep graph size and dense-support size fixed across R.  The padded
        # nodes are zero-valued null distractors with relation label 0, so R
        # changes relation demand rather than graph size.
        null_distractors = total_nodes - 1 - int(r)
        specs.append(
            base_spec_from_args(
                args,
                relation_types=int(r),
                task_mode="global",
                noise_nodes=null_distractors,
                noise_sigma=0.0,
                layers=1,
            )
        )
    return specs


def run_capacity_crossover(args: argparse.Namespace) -> None:
    specs = capacity_crossover_specs_from_args(args)
    default_models_arg = ",".join(CONTROLLED_MODELS)
    models = (
        list(CAPACITY_CROSSOVER_MODELS)
        if str(args.models) == default_models_arg
        else parse_csv_list(args.models, allowed=CAPACITY_MODEL_NAMES)
    )
    train_grid(
        args,
        experiment="capacity_crossover_global",
        specs=specs,
        models=models,
        seeds=parse_int_list(args.seeds),
    )


def run_transport_support(args: argparse.Namespace) -> None:
    models = list(CONTROLLED_MODELS)
    if args.include_practical:
        models.extend(PRACTICAL_MODELS)
    specs = [
        base_spec_from_args(args, relation_types=int(args.relation_types), task_mode=mode, noise_nodes=0, noise_sigma=0.0, layers=1)
        for mode in ("local", "global")
    ]
    train_grid(args, experiment="transport_support", specs=specs, models=models, seeds=parse_int_list(args.seeds))


def run_transport_support_official(args: argparse.Namespace) -> None:
    specs = [
        base_spec_from_args(args, relation_types=int(args.relation_types), task_mode=mode, noise_nodes=0, noise_sigma=0.0, layers=1)
        for mode in ("local", "global")
    ]
    train_grid(
        args,
        experiment="transport_support_official",
        specs=specs,
        models=("graphgps_official", "grit_official", "gatedgcn_plus_official"),
        seeds=parse_int_list(args.seeds),
    )


def run_overglobalisation(args: argparse.Namespace) -> None:
    seeds = parse_int_list(args.seeds)
    fixed_models = list(CONTROLLED_MODELS)
    if args.include_practical:
        fixed_models.extend(PRACTICAL_MODELS)
    fixed_specs = [
        base_spec_from_args(
            args,
            relation_types=int(args.relation_types),
            task_mode=mode,
            noise_nodes=int(args.noise_nodes or 3 * int(args.relation_types)),
            noise_sigma=1.0,
            layers=1,
        )
        for mode in ("local", "global")
    ]
    train_grid(args, experiment="overglobalisation_fixed", specs=fixed_specs, models=fixed_models, seeds=seeds)
    sweep_specs = [
        base_spec_from_args(
            args,
            relation_types=int(args.relation_types),
            task_mode=mode,
            noise_nodes=int(args.noise_nodes or 3 * int(args.relation_types)),
            noise_sigma=sigma,
            layers=1,
        )
        for mode in ("local", "global")
        for sigma in parse_float_list(args.noise_sweep)
    ]
    train_grid(args, experiment="overglobalisation", specs=sweep_specs, models=("full_dense", "full_1hop"), seeds=seeds)


def run_overglobalisation_grit(args: argparse.Namespace) -> None:
    seeds = parse_int_list(args.seeds)
    specs = [
        base_spec_from_args(
            args,
            relation_types=int(args.relation_types),
            task_mode=mode,
            noise_nodes=int(args.noise_nodes or 3 * int(args.relation_types)),
            noise_sigma=sigma,
            layers=1,
        )
        for mode in ("local", "global")
        for sigma in parse_float_list(args.noise_sweep)
    ]
    train_grid(
        args,
        experiment="overglobalisation_grit_official",
        specs=specs,
        models=OFFICIAL_GRIT_MODELS,
        seeds=seeds,
    )


def run_depth_escape(args: argparse.Namespace) -> None:
    specs = [
        base_spec_from_args(args, relation_types=int(args.relation_types), task_mode="local", noise_nodes=0, noise_sigma=0.0, layers=layers)
        for layers in parse_int_list(args.depth_sweep)
    ]
    train_grid(args, experiment="depth_escape", specs=specs, models=("routing_dense", "routing_1hop"), seeds=parse_int_list(args.seeds))
    ref_spec = base_spec_from_args(args, relation_types=int(args.relation_types), task_mode="local", noise_nodes=0, noise_sigma=0.0, layers=1)
    train_grid(args, experiment="depth_escape", specs=[ref_spec], models=("full_dense", "full_1hop"), seeds=parse_int_list(args.seeds))


def build_data(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    for r in parse_int_list(args.r_sweep):
        save_dataset(root, base_spec_from_args(args, relation_types=r, task_mode="local"), overwrite=bool(args.overwrite_data))
    for mode in ("local", "global"):
        save_dataset(root, base_spec_from_args(args, task_mode=mode), overwrite=bool(args.overwrite_data))
        save_dataset(
            root,
            base_spec_from_args(args, task_mode=mode, noise_nodes=int(args.noise_nodes or 3 * int(args.relation_types)), noise_sigma=1.0),
            overwrite=bool(args.overwrite_data),
        )


def run_train(args: argparse.Namespace) -> None:
    models = parse_csv_list(args.models, allowed=ALL_MODELS)
    spec = base_spec_from_args(args)
    train_grid(args, experiment=str(args.experiment), specs=[spec], models=models, seeds=parse_int_list(args.seeds))


def run_evaluate(args: argparse.Namespace) -> None:
    write_summary_tables(Path(args.output_root))


def import_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
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


def aggregate(rows: Sequence[Mapping[str, Any]], keys: Sequence[str], value: str) -> dict[tuple[Any, ...], tuple[float, float]]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        if value not in row or row[value] in {"", None}:
            continue
        key = tuple(row[k] for k in keys)
        grouped.setdefault(key, []).append(float(row[value]))
    out = {}
    for key, vals in grouped.items():
        arr = np.asarray(vals, dtype=float)
        sem = float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        out[key] = (float(arr.mean()), sem)
    return out


def metric_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "metrics" / "clean_metrics.csv"
    if not path.exists():
        write_summary_tables(root)
    return read_csv(path) if path.exists() else []


def plot_crossover(root: Path) -> Path | None:
    rows = [row for row in metric_rows(root) if row["experiment"] == "crossover"]
    if not rows:
        return None
    plt = import_plotting()
    agg_mse = aggregate(rows, ("model", "relation_types"), "test_rel_mse")
    agg_rank = aggregate(rows, ("model", "relation_types"), "realised_rank")
    models = list(CONTROLLED_MODELS)
    rs = sorted({int(row["relation_types"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for model in models:
        xs = rs
        ys = [agg_mse.get((model, str(r)), (np.nan, 0.0))[0] for r in xs]
        es = [agg_mse.get((model, str(r)), (np.nan, 0.0))[1] for r in xs]
        axes[0].errorbar(xs, ys, yerr=es, marker="o", linewidth=1.8, capsize=3, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
        ranks = [agg_rank.get((model, str(r)), (np.nan, 0.0))[0] for r in xs]
        axes[1].plot(xs, ranks, marker="o", linewidth=1.8, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
    floor_by_r = {int(row["relation_types"]): row for row in rows}
    axes[0].plot(rs, [float(floor_by_r[r]["routing_floor"]) for r in rs], "--", color="#333333", label="Eckart-Young routing floor")
    axes[0].plot(rs, [float(floor_by_r[r]["full_transport_floor"]) for r in rs], ":", color="#333333", label="Eckart-Young transport floor")
    axes[0].set_xlabel("number of relations R")
    axes[0].set_ylabel("relative MSE")
    axes[0].set_title("Error follows the relation-rank floor")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[1].axhline(4, linestyle="--", color="#555555", linewidth=0.9, label="H")
    axes[1].axhline(16, linestyle=":", color="#555555", linewidth=0.9, label="H x k_tr")
    axes[1].set_xlabel("number of relations R")
    axes[1].set_ylabel("realised relation rank")
    axes[1].set_title("Transport increases realised relation rank")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Relation Rank Crossover Under Local Support", y=1.13, fontsize=13)
    fig.tight_layout()
    path = ensure_dir(root / "figures") / "relation_rank_crossover_local.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def write_capacity_model_table(root: Path) -> Path:
    rows = [
        {
            "model": model,
            "display_name": MODEL_LABELS[model],
            "description": CAPACITY_MODEL_DESCRIPTIONS[model],
        }
        for model in CAPACITY_CROSSOVER_MODELS
    ]
    path = ensure_dir(root / "metrics") / "capacity_crossover_model_table.csv"
    write_csv(path, rows)
    return path


def plot_capacity_crossover(root: Path, target_nodes: int | None = None) -> Path | None:
    rows = [row for row in metric_rows(root) if row["experiment"] == "capacity_crossover_global"]
    if not rows:
        return None
    by_total_nodes: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        total_nodes = 1 + int(row["relation_types"]) + int(row.get("noise_nodes", 0))
        by_total_nodes.setdefault(total_nodes, []).append(row)
    if target_nodes is not None and target_nodes in by_total_nodes:
        if len(by_total_nodes) > 1:
            print(f"[plot] capacity crossover found multiple graph sizes; using requested N={target_nodes}")
        rows = by_total_nodes[int(target_nodes)]
    elif target_nodes is not None and len(by_total_nodes) > 1:
        available = ", ".join(str(n) for n in sorted(by_total_nodes))
        print(f"[plot] capacity crossover requested N={target_nodes}, but available graph sizes are {available}; skipping")
        return None
    elif len(by_total_nodes) > 1:
        complete_counts = {
            total_nodes: len({(row["model"], row["relation_types"], row["seed"]) for row in grouped})
            for total_nodes, grouped in by_total_nodes.items()
        }
        selected_nodes = max(complete_counts, key=lambda key: (complete_counts[key], key))
        print(f"[plot] capacity crossover found multiple graph sizes; using N={selected_nodes}")
        rows = by_total_nodes[selected_nodes]
    write_capacity_model_table(root)
    plt = import_plotting()
    agg_mse = aggregate(rows, ("model", "relation_types"), "test_rel_mse")
    capacity_rank_rows = [
        {
            **row,
            "capacity_plot_rank": numerical_rank_from_singular_values(
                singular_values_from_text(row.get("realised_singular_values")),
                tol=1.0e-4,
            ),
        }
        for row in rows
    ]
    agg_rank = aggregate(capacity_rank_rows, ("model", "relation_types"), "capacity_plot_rank")
    rs = sorted({int(row["relation_types"]) for row in rows})
    first = rows[0]
    heads = int(first.get("heads", 4))
    input_dim = int(first.get("input_dim", 32))
    total_nodes = 1 + int(first["relation_types"]) + int(first.get("noise_nodes", 0))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    completed_models = [
        model
        for model in CAPACITY_CROSSOVER_MODELS
        if any((model, str(r)) in agg_mse for r in rs)
    ]
    for model in completed_models:
        xs = rs
        ys = [agg_mse.get((model, str(r)), (np.nan, 0.0))[0] for r in xs]
        es = [agg_mse.get((model, str(r)), (np.nan, 0.0))[1] for r in xs]
        axes[0].errorbar(
            xs,
            ys,
            yerr=es,
            marker="o",
            linewidth=1.9,
            capsize=3,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    floor_by_r: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        floor_by_r.setdefault(int(row["relation_types"]), row)
    axes[0].plot(
        rs,
        [float(floor_by_r[r]["routing_floor"]) for r in rs],
        "--",
        color="#222222",
        linewidth=1.4,
        label=f"Rank-{heads} approximation floor",
    )
    axes[0].set_xlabel("number of relations R")
    axes[0].set_ylabel("relative MSE")
    axes[0].set_title("Error versus relation count")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.6)
    rank_values = np.full((len(completed_models), len(rs)), np.nan, dtype=float)
    rank_fraction = np.full_like(rank_values, np.nan)
    for row_idx, model in enumerate(completed_models):
        for col_idx, r in enumerate(rs):
            rank = agg_rank.get((model, str(r)), (np.nan, 0.0))[0]
            rank_values[row_idx, col_idx] = rank
            rank_fraction[row_idx, col_idx] = rank / max(float(r), EPS)
    heat = np.ma.masked_invalid(rank_fraction)
    image = axes[1].imshow(heat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    axes[1].set_xticks(np.arange(len(rs)), [str(r) for r in rs])
    axes[1].set_yticks(np.arange(len(completed_models)), [MODEL_LABELS[model] for model in completed_models])
    axes[1].set_xlabel("number of relations R")
    axes[1].set_title("Realised rank as fraction of demand")
    for row_idx in range(len(completed_models)):
        for col_idx in range(len(rs)):
            rank = rank_values[row_idx, col_idx]
            if np.isfinite(rank):
                frac = rank_fraction[row_idx, col_idx]
                text_color = "white" if frac < 0.55 else "black"
                axes[1].text(col_idx, row_idx, f"{rank:.0f}", ha="center", va="center", color=text_color, fontsize=8)
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    axes[1].tick_params(axis="both", length=0)
    cbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03)
    cbar.set_label("realised rank / R")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"Dense Global Relation-Operator Capacity (N={total_nodes}, H={heads}, d={input_dim})", y=1.15, fontsize=13)
    fig.tight_layout()
    path = ensure_dir(root / "figures") / "relation_rank_crossover_global_capacity_h4.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_crossover_with_grit(root: Path) -> Path | None:
    rows = [row for row in metric_rows(root) if row["experiment"] == "crossover"]
    grit_rows = [row for row in metric_rows(root) if row["experiment"] == "grit_r_local_diagnostic"]
    if not rows or not grit_rows:
        return None
    plt = import_plotting()
    agg_mse = aggregate(rows, ("model", "relation_types"), "test_rel_mse")
    agg_rank = aggregate(rows, ("model", "relation_types"), "realised_rank")
    agg_grit = aggregate(grit_rows, ("model", "relation_types"), "test_rel_mse")
    agg_grit_rank = aggregate(grit_rows, ("model", "relation_types"), "realised_rank")
    models = list(CONTROLLED_MODELS)
    rs = sorted({int(row["relation_types"]) for row in rows})
    grit_rs = sorted({int(row["relation_types"]) for row in grit_rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    for model in models:
        ys = [agg_mse.get((model, str(r)), (np.nan, 0.0))[0] for r in rs]
        es = [agg_mse.get((model, str(r)), (np.nan, 0.0))[1] for r in rs]
        axes[0].errorbar(rs, ys, yerr=es, marker="o", linewidth=1.8, capsize=3, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
        axes[1].plot(rs, [agg_rank.get((model, str(r)), (np.nan, 0.0))[0] for r in rs], marker="o", linewidth=1.8, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
    grit_model = "grit_official"
    axes[0].plot(
        grit_rs,
        [agg_grit.get((grit_model, str(r)), (np.nan, 0.0))[0] for r in grit_rs],
        marker="D",
        linewidth=2.3,
        color=MODEL_COLORS[grit_model],
        label="Official GRIT (1 seed)",
    )
    grit_rank_x = [r for r in grit_rs if agg_grit_rank.get((grit_model, str(r)), (-1.0, 0.0))[0] >= 0]
    if grit_rank_x:
        axes[1].plot(
            grit_rank_x,
            [agg_grit_rank[(grit_model, str(r))][0] for r in grit_rank_x],
            marker="D",
            linewidth=2.3,
            color=MODEL_COLORS[grit_model],
            label="Official GRIT empirical rank",
        )
    floor_by_r = {int(row["relation_types"]): row for row in rows}
    axes[0].plot(rs, [float(floor_by_r[r]["routing_floor"]) for r in rs], "--", color="#333333", label="Eckart-Young routing floor")
    axes[0].plot(rs, [float(floor_by_r[r]["full_transport_floor"]) for r in rs], ":", color="#333333", label="Eckart-Young transport floor")
    axes[0].set_xlabel("number of relations R")
    axes[0].set_ylabel("relative MSE")
    axes[0].set_title("Official GRIT overlaid on controlled capacity curves")
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[1].axhline(4, linestyle="--", color="#555555", linewidth=0.9, label="H")
    axes[1].axhline(16, linestyle=":", color="#555555", linewidth=0.9, label="H x k_tr")
    axes[1].set_xlabel("number of relations R")
    axes[1].set_ylabel("realised relation rank")
    axes[1].set_title("Realised/effective relation rank")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Relation Rank Crossover Under Local Support", y=1.13, fontsize=13)
    fig.tight_layout()
    path = ensure_dir(root / "figures") / "relation_rank_crossover_local_with_official_grit.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_transport_support(root: Path, task_mode: str) -> Path | None:
    rows = [row for row in metric_rows(root) if row["experiment"] == "transport_support" and row["task_mode"] == task_mode]
    if not rows:
        return None
    plt = import_plotting()
    agg_mse = aggregate(rows, ("model",), "test_rel_mse")
    controlled = [m for m in CONTROLLED_MODELS if (m,) in agg_mse]
    practical = [m for m in PRACTICAL_MODELS if (m,) in agg_mse]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), width_ratios=[1.15, 1.0])
    for ax, models, title in [(axes[0], controlled, "Controlled capacity models"), (axes[1], practical, "Matched practical models")]:
        xs = np.arange(len(models))
        ys = [agg_mse[(m,)][0] for m in models]
        es = [agg_mse[(m,)][1] for m in models]
        ax.bar(xs, ys, yerr=es, capsize=3, color=[MODEL_COLORS[m] for m in models], edgecolor="black", linewidth=0.5)
        ax.set_xticks(xs, [MODEL_LABELS[m] for m in models], rotation=25, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("relative MSE")
        ax.set_title(title)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    headline = (
        "Local Relation Operator: Transport Matters, Dense Support Is Not Required"
        if task_mode == "local"
        else "Long-Range Relation Operator: Dense Support and Transport Are Both Required"
    )
    fig.suptitle(headline, y=1.03, fontsize=13)
    fig.tight_layout()
    name = "transport_support_local_relation_operator.pdf" if task_mode == "local" else "transport_support_global_relation_operator.pdf"
    path = ensure_dir(root / "figures") / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_transport_support_official(root: Path) -> Path | None:
    rows = [row for row in metric_rows(root) if row["experiment"] == "transport_support_official"]
    if not rows:
        return None
    plt = import_plotting()
    models = ["graphgps_official", "grit_official", "gatedgcn_plus_official"]
    agg_mse = aggregate(rows, ("task_mode", "model"), "test_rel_mse")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), sharey=True)
    for ax, mode in zip(axes, ("local", "global"), strict=True):
        xs = np.arange(len(models))
        ys = [agg_mse.get((mode, model), (np.nan, 0.0))[0] for model in models]
        es = [agg_mse.get((mode, model), (np.nan, 0.0))[1] for model in models]
        ax.bar(
            xs,
            ys,
            yerr=es,
            capsize=3,
            color=[MODEL_COLORS[model] for model in models],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(xs, [MODEL_LABELS[model] for model in models], rotation=25, ha="right")
        ax.set_yscale("log")
        ax.set_title("Local relation operator" if mode == "local" else "Long-range relation operator")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("relative MSE")
    fig.suptitle("Official Practical Layers on the Relation Operator", y=1.03, fontsize=13)
    fig.tight_layout()
    path = ensure_dir(root / "figures") / "transport_support_official_practical_relation_operator.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_overglobalisation(root: Path) -> Path | None:
    rows = [row for row in metric_rows(root) if row["experiment"] in {"overglobalisation", "overglobalisation_fixed"}]
    if not rows:
        return None
    plt = import_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    sweep = [row for row in rows if row["experiment"] == "overglobalisation"]
    for mode, style in [("local", "-"), ("global", "--")]:
        for model in ("full_dense", "full_1hop"):
            sub = [row for row in sweep if row["task_mode"] == mode and row["model"] == model]
            if not sub:
                continue
            sigmas = sorted({float(row["noise_sigma"]) for row in sub})
            agg_mse = aggregate(sub, ("model", "task_mode", "noise_sigma"), "test_rel_mse")
            axes[0].errorbar(
                sigmas,
                [agg_mse[(model, mode, str(sigma))][0] for sigma in sigmas],
                yerr=[agg_mse[(model, mode, str(sigma))][1] for sigma in sigmas],
                marker="o",
                linestyle=style,
                color=MODEL_COLORS[model],
                label=f"{MODEL_LABELS[model]} / {mode}",
            )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("irrelevant content noise magnitude")
    axes[0].set_ylabel("relative MSE")
    axes[0].set_title("Noise sweep for full-transport models")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.6)
    fixed = [row for row in rows if row["experiment"] == "overglobalisation_fixed"]
    agg_fixed = aggregate(fixed, ("model", "task_mode"), "test_rel_mse")
    labels = [m for m in ALL_MODELS if (m, "local") in agg_fixed or (m, "global") in agg_fixed]
    x = np.arange(len(labels))
    width = 0.36
    for idx, mode in enumerate(("local", "global")):
        ys = [agg_fixed.get((m, mode), (np.nan, 0.0))[0] for m in labels]
        axes[1].bar(x + (idx - 0.5) * width, ys, width=width, label=mode, color=("#9ecae1" if mode == "local" else "#fdae6b"), edgecolor="black", linewidth=0.5)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, [MODEL_LABELS[m] for m in labels], rotation=25, ha="right")
    axes[1].set_title("Fixed irrelevant-content stress test")
    axes[1].set_ylabel("relative MSE")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.6)
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, fontsize=7)
    fig.suptitle("Irrelevant Content Exposes the Cost of Dense Support", y=1.03, fontsize=13)
    fig.tight_layout()
    path = ensure_dir(root / "figures") / "overglobalisation_irrelevant_content.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_overglobalisation_grit(root: Path) -> Path | None:
    rows = [row for row in metric_rows(root) if row["experiment"] == "overglobalisation_grit_official"]
    if not rows:
        return None
    plt = import_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, mode in zip(axes, ("local", "global"), strict=True):
        for model in OFFICIAL_GRIT_MODELS:
            sub = [row for row in rows if row["task_mode"] == mode and row["model"] == model]
            if not sub:
                continue
            sigmas = sorted({float(row["noise_sigma"]) for row in sub})
            agg_mse = aggregate(sub, ("model", "task_mode", "noise_sigma"), "test_rel_mse")
            ax.errorbar(
                sigmas,
                [agg_mse[(model, mode, str(sigma))][0] for sigma in sigmas],
                yerr=[agg_mse[(model, mode, str(sigma))][1] for sigma in sigmas],
                marker="o",
                linewidth=2.0,
                capsize=3,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        ax.set_title("Local teacher" if mode == "local" else "Long-range teacher")
        ax.set_xlabel("irrelevant content noise magnitude")
        ax.set_yscale("log")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("relative MSE")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Official GRIT: Dense Support Increases Irrelevant-Content Sensitivity", y=1.04, fontsize=13)
    fig.tight_layout()
    path = ensure_dir(root / "figures") / "overglobalisation_official_grit_dense_vs_1hop.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_depth_escape(root: Path) -> Path | None:
    rows = [row for row in metric_rows(root) if row["experiment"] == "depth_escape"]
    if not rows:
        return None
    plt = import_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for model in ("routing_dense", "routing_1hop", "full_dense", "full_1hop"):
        sub = [row for row in rows if row["model"] == model]
        if not sub:
            continue
        layers = sorted({int(row["layers"]) for row in sub})
        agg_mse = aggregate(sub, ("model", "layers"), "test_rel_mse")
        agg_rank = aggregate(sub, ("model", "layers"), "realised_rank")
        axes[0].errorbar(layers, [agg_mse[(model, str(l))][0] for l in layers], yerr=[agg_mse[(model, str(l))][1] for l in layers], marker="o", label=MODEL_LABELS[model], color=MODEL_COLORS[model])
        axes[1].plot(layers, [agg_rank[(model, str(l))][0] for l in layers], marker="o", label=MODEL_LABELS[model], color=MODEL_COLORS[model])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of layers")
    axes[0].set_ylabel("relative MSE")
    axes[0].set_title("Routing-only improves with composition")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[1].set_xlabel("number of layers")
    axes[1].set_ylabel("realised relation rank")
    axes[1].set_title("Depth changes realised relation rank")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Depth Partially Escapes the Routing Rank Limit", y=1.13, fontsize=13)
    fig.tight_layout()
    path = ensure_dir(root / "figures") / "depth_escape_routing_only.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")
    return path


def plot_all(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    write_summary_tables(root)
    plot_crossover(root)
    plot_capacity_crossover(root, target_nodes=getattr(args, "n_nodes", None))
    plot_crossover_with_grit(root)
    plot_transport_support(root, "local")
    plot_transport_support(root, "global")
    plot_transport_support_official(root)
    plot_overglobalisation(root)
    plot_overglobalisation_grit(root)
    plot_depth_escape(root)


def run_all(args: argparse.Namespace) -> None:
    build_data(args)
    run_crossover(args)
    run_transport_support(args)
    run_overglobalisation(args)
    run_depth_escape(args)
    plot_all(args)


def print_hpc_commands(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    common = (
        "source /usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh; "
        "conda activate graphbench-algoreas; "
        "export PYTHONPATH=$PWD/src:$PYTHONPATH; "
        "export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}; "
        "export GRAPHGPS_ROOT=${GRAPHGPS_ROOT:-$PWD/graphbench-algoreas-hpc/external/GraphGPS}; "
        "export GRIT_ROOT=${GRIT_ROOT:-$PWD/graphbench-algoreas-hpc/external/GRIT}; "
        "export GNNPLUS_ROOT=${GNNPLUS_ROOT:-$PWD/graphbench-algoreas-hpc/external/GNNPlus}; "
    )
    module = "python -u -m graph_specialisation_metrics.relation_operator_capacity"
    base_flags = (
        f"--output-root {root} --seeds {args.seeds} --r-sweep {args.r_sweep} "
        f"--relation-types {args.relation_types} --input-dim {args.input_dim} "
        f"--target-dim {args.target_dim} --hidden-dim {args.hidden_dim} "
        f"--heads {args.heads} --transport-bases {args.transport_bases} "
        f"--train-size {args.train_size} --val-size {args.val_size} --test-size {args.test_size} "
        f"--batch-size {args.batch_size} --eval-batch-size {args.eval_batch_size} "
        f"--max-epochs {args.max_epochs} --patience {args.patience} --lr {args.lr} "
        f"--weight-decay {args.weight_decay} --noise-sweep {args.noise_sweep} "
        f"--depth-sweep {args.depth_sweep} --parameter-match-widths {args.parameter_match_widths} "
        f"--device cuda --amp"
    )
    print("mkdir -p logs")
    for name, command, extra in [
        ("ch4-ctrl", "run-crossover", ""),
        ("ch4-capacity", "run-capacity-crossover", " --seeds 1001 --r-sweep 2,4,5,6,8 --n-nodes 10 --hidden-dim 32"),
        ("ch4-ts", "run-transport-support", " --include-practical"),
        ("ch4-ts-official", "run-transport-support-official", ""),
        ("ch4-noise", "run-overglobalisation", " --include-practical"),
        ("ch4-grit-noise", "run-overglobalisation-grit", " --seeds 1001 --noise-sweep 0,0.5,1,2"),
        ("ch4-depth", "run-depth-escape", ""),
        ("ch4-plot", "plot", ""),
    ]:
        wrap = common + module + f" {command} {base_flags}{extra}"
        print(
            "sbatch -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 "
            "--gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G --time=01:00:00 "
            f"-J {name} -o logs/{name}-%j.out -e logs/{name}-%j.err --wrap {json.dumps(wrap)}"
        )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--seeds", type=str, default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--r-sweep", type=str, default=",".join(str(r) for r in R_SWEEP))
    parser.add_argument("--relation-types", type=int, default=16)
    parser.add_argument("--task-mode", type=str, default="local", choices=TASK_MODES)
    parser.add_argument("--models", type=str, default=",".join(CONTROLLED_MODELS))
    parser.add_argument("--experiment", type=str, default="manual")
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--target-dim", type=int, default=32)
    parser.add_argument(
        "--n-nodes",
        type=int,
        default=16,
        help="Total nodes for capacity-crossover padding; other Ch4 stages use R plus irrelevant-content nodes.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--transport-bases", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--noise-nodes", type=int, default=0)
    parser.add_argument("--noise-sigma", type=float, default=0.0)
    parser.add_argument("--noise-sweep", type=str, default="0,0.25,0.5,1,2,4")
    parser.add_argument("--depth-sweep", type=str, default="1,2,4,8")
    parser.add_argument("--parameter-match-widths", type=str, default=",".join(str(width) for width in PARAMETER_MATCH_WIDTHS))
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--val-size", type=int, default=2048)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--data-seed", type=int, default=7101)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=0, help="Reserved for HPC config logging; data is tensor-cached in memory.")
    parser.add_argument("--amp", "--use-amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-practical", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite-checkpoints", action=argparse.BooleanOptionalAction, default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text, fn in [
        ("build-data", "Cache Ch4 relation-operator datasets.", build_data),
        ("train", "Train selected model(s) for one explicit config.", run_train),
        ("evaluate", "Rebuild metrics tables from completed run summaries.", run_evaluate),
        ("run-crossover", "Run the local relation-rank crossover experiment.", run_crossover),
        ("run-capacity-crossover", "Run the clean dense global capacity crossover experiment.", run_capacity_crossover),
        ("run-transport-support", "Run local/global transport-support experiments.", run_transport_support),
        ("run-transport-support-official", "Run official GraphGPS/GRIT/GNN+ transport-support experiments.", run_transport_support_official),
        ("run-overglobalisation", "Run irrelevant-content over-globalisation experiments.", run_overglobalisation),
        ("run-overglobalisation-grit", "Run official GRIT dense-vs-1-hop over-globalisation experiment.", run_overglobalisation_grit),
        ("run-depth-escape", "Run routing-only depth escape experiments.", run_depth_escape),
        ("plot", "Generate all Ch4 figures from cached metrics.", plot_all),
        ("run-all", "Run all Ch4 stages.", run_all),
        ("print-hpc-commands", "Print HPC sbatch commands.", print_hpc_commands),
    ]:
        sub = subparsers.add_parser(name, help=help_text)
        add_common_args(sub)
        sub.set_defaults(func=fn)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
