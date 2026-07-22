"""Single-query Neighbor Associative Recall with dense and trained-mask official GRIT.

This is the centrally editable implementation used by the standalone Colab launcher at
``experiments/synthetic/training/nar_grit_colab.py``.  It deliberately keeps the two-layer
Neighbor Associative Recall (NAR) computation: record nodes reach the output centre in the first
round, while the query is two graph hops from the centre and only arrives after that initial
aggregation.  A 1-hop model must therefore store the complete key--payload map before it knows
which record will be requested.  A 2-hop or dense model can place the query in the centre state
after layer one and content-select records in layer two.

Every selected record carries two independent payloads: a semantic value and a topology-derived
binary role (triangle-tail versus square gadget).  Both are predicted on every graph.  This makes
the repository's calibrated joint influence ``J`` and semantic--structural selectivity ``D_rel``
meaningful without introducing a task-switch marker.

Methodology alignment
---------------------
* official GRIT ``GritTransformerLayer`` pinned to the repository commit;
* trained 1-hop/2-hop/dense support, never a post-training mask;
* per-head scores at official GRIT's routed ``wV`` transport site;
* within-forward clean/corrupt replicas and nuisance averaging before magnitude;
* semantic value donors with structure fixed;
* mask-frozen RRWP transposition for the structural head score;
* support-conjugated structural transposition for structural carriage; and
* pre-output head ablation at the same routed head site.

Intentional synthetic differences are written to every summary.  The readout is the central node,
not graph pooling; the semantic donor changes the task-defined value field rather than a complete
molecular feature row; and beneficial carriage is the exact source-level cross-entropy change at
that sole readout carrier, rather than the production multi-carrier integrated allocation.
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
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPERIMENT_VERSION = "nar-grit-capacity-v1"
OFFICIAL_GRIT_URL = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
DEFAULT_GRIT_DIR = "/content/GRIT"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/graph_specialisation_metrics/nar_grit"
MODEL_RADII: dict[str, int | None] = {"1hop": 1, "2hop": 2, "dense": None}
MODEL_ORDER = ("1hop", "2hop", "dense")
CHANNELS = ("semantic", "structural")
EPS = 1.0e-12
DJ_RELIABILITY_FLOOR = 0.50
DJ_SELECTIVITY_THRESHOLD = 0.20

METHODOLOGY_ALIGNMENT = {
    "official_grit_commit": OFFICIAL_GRIT_COMMIT,
    "score_site": "official GRIT routed per-head wV",
    "replica_baseline": "within-forward clean replica",
    "semantic_score_intervention": "target value donor; topology fixed",
    "structural_score_intervention": "degree-matched RRWP transposition; support frozen",
    "semantic_carriage_intervention": "target value donor; topology fixed",
    "structural_carriage_intervention": "degree-matched RRWP and support transposition",
    "head_ablation": "zero routed head output before head concatenation",
}

METHODOLOGY_DIFFERENCES = {
    "readout": "central-node dual classifier, not graph pooling",
    "score_sources": "the task-defined requested record only, not an all-node source sweep",
    "semantic_content": "task-defined value field, not a complete molecular feature row",
    "beneficial_carriage": (
        "exact source-level cross-entropy change at the sole central carrier; "
        "not production multi-carrier path-integrated allocation"
    ),
    "structural_factor": "controlled binary local motif role on degree-matched record nodes",
}


# ======================================================================================
# Bootstrap, serialisation, and configuration
# ======================================================================================


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
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "torch_geometric",
            "yacs",
            "ogb",
            "einops",
            "opt_einsum",
        ],
        check=False,
    )
    for package in ("pyg-lib", "torch-spline-conv", "torch-cluster"):
        _run([sys.executable, "-m", "pip", "install", "-q", package, "-f", wheel_url], check=False)
    for package in ("torch-scatter", "torch-sparse"):
        code = _run(
            [sys.executable, "-m", "pip", "install", "-q", package, "-f", wheel_url],
            check=False,
        )
        if code:
            raise SystemExit(
                f"Required PyG extension {package!r} has no wheel at {wheel_url}. "
                "Use a Colab runtime with a PyG-supported torch/CUDA build."
            )


def setup_official_grit(grit_dir: Path, *, install: bool) -> None:
    if not (grit_dir / ".git").exists():
        if grit_dir.exists():
            shutil.rmtree(grit_dir)
        _run(["git", "clone", OFFICIAL_GRIT_URL, str(grit_dir)])
    _run(["git", "-C", str(grit_dir), "checkout", OFFICIAL_GRIT_COMMIT])
    if install:
        install_pyg_stack()
        _run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(grit_dir), "--no-deps"],
            check=False,
        )
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
    run_name: str = "nar_grit_v1"
    drive_root: str = DEFAULT_DRIVE_ROOT
    key_vocab: int = 128
    semantic_classes: int = 8
    structural_classes: int = 2
    rrwp_steps: int = 6
    widths: tuple[int, ...] = (64, 128)
    analysis_width: int = 128
    heads: int = 8
    layers: int = 2
    models: tuple[str, ...] = MODEL_ORDER
    train_ns: tuple[int, ...] = (4, 8, 16, 32, 64)
    eval_ns: tuple[int, ...] = (4, 8, 12, 16, 24, 32, 48, 64)
    mechanistic_ns: tuple[int, ...] = (4, 16, 32, 64)
    dropout: float = 0.0
    attention_dropout: float = 0.0
    batch_size: int = 16
    max_batch_nodes: int = 1100
    max_dense_pairs: int = 300_000
    steps: int = 2000
    min_steps: int = 1500
    warmup_steps: int = 400
    train_n_sampling_exponent: float = -0.5
    lr: float = 1.5e-3
    weight_decay: float = 1.0e-5
    eval_every: int = 100
    validation_graphs: int = 48
    heldout_graphs: int = 256
    patience_checks: int = 7
    low_n_accuracy_gate: float = 0.85
    score_graphs: int = 8
    score_donors: int = 3
    analysis_batch_size: int = 8
    ablation_graphs: int = 96
    family_size: int = 2
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "cuda"

    @property
    def feature_dim(self) -> int:
        # key | semantic value | query | record | centre | intermediate | gadget
        return self.key_vocab + self.semantic_classes + 5

    def nodes_for_n(self, records: int) -> int:
        # centre, intermediate, query + (record and three private gadget nodes) per memory.
        return 3 + 4 * int(records)

    def validate(self) -> None:
        if self.layers != 2:
            raise ValueError("NAR capacity isolation requires exactly two GRIT layers")
        if tuple(self.models) != tuple(dict.fromkeys(self.models)):
            raise ValueError("models must be unique")
        if any(model not in MODEL_RADII for model in self.models):
            raise ValueError(f"models must be drawn from {sorted(MODEL_RADII)}")
        if "1hop" not in self.models or "dense" not in self.models:
            raise ValueError("the core comparison requires 1hop and dense")
        if self.analysis_width not in self.widths:
            raise ValueError("analysis_width must be one of widths")
        if any(width % self.heads for width in self.widths):
            raise ValueError("every width must be divisible by heads")
        if any(value <= 0 or value % 2 for value in self.train_ns + self.eval_ns):
            raise ValueError("all N values must be positive and even for role balance")
        if any(value not in self.eval_ns for value in self.mechanistic_ns):
            raise ValueError("mechanistic_ns must be a subset of eval_ns")
        if max(self.train_ns + self.eval_ns) > self.key_vocab:
            raise ValueError("key_vocab must cover the maximum record count")
        if self.structural_classes != 2:
            raise ValueError("the controlled motif generator currently defines two roles")
        if not 0 < self.family_size <= self.layers * self.heads // 3:
            raise ValueError("family_size must leave room for disjoint controls")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be non-negative and smaller than steps")


def config_fingerprint(cfg: Config) -> str:
    payload = {"version": EXPERIMENT_VERSION, **asdict(cfg)}
    payload.pop("drive_root", None)
    payload.pop("run_name", None)
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ======================================================================================
# Controlled single-query dual-payload NAR generator
# ======================================================================================


@dataclass
class NarBatch:
    x: Any
    adj: Any
    rrwp: Any
    central_idx: Any
    query_idx: Any
    target_idx: Any
    record_mask: Any
    record_role: Any
    y_semantic: Any
    y_structural: Any
    n_records: Any

    def __len__(self) -> int:
        return int(self.x.size(0))

    def to(self, device: Any) -> "NarBatch":
        return NarBatch(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})

    def cpu(self) -> "NarBatch":
        return self.to("cpu")

    def slice(self, start: int, stop: int) -> "NarBatch":
        return NarBatch(**{name: getattr(self, name)[start:stop] for name in self.__dataclass_fields__})

    def select(self, indices: Any) -> "NarBatch":
        return NarBatch(**{name: getattr(self, name)[indices] for name in self.__dataclass_fields__})


def concat_batches(batches: Sequence[NarBatch]) -> NarBatch:
    import torch

    if not batches:
        raise ValueError("cannot concatenate an empty batch list")
    node_counts = {int(batch.x.size(1)) for batch in batches}
    if len(node_counts) != 1:
        raise ValueError("NarBatch concatenation requires a common N")
    return NarBatch(
        **{
            name: torch.cat([getattr(batch, name) for batch in batches], dim=0)
            for name in NarBatch.__dataclass_fields__
        }
    )


def add_undirected(adj: np.ndarray, left: int, right: int) -> None:
    adj[int(left), int(right)] = 1.0
    adj[int(right), int(left)] = 1.0


def rrwp_from_adj(adj: np.ndarray, steps: int) -> np.ndarray:
    from scipy.sparse import csr_matrix

    degree = adj.sum(axis=1, keepdims=True)
    transition = adj / np.maximum(degree, 1.0)
    transition_sparse = csr_matrix(transition)
    nodes = int(adj.shape[0])
    output = np.zeros((nodes, nodes, int(steps)), dtype=np.float32)
    power = np.eye(nodes, dtype=np.float32)
    for step in range(int(steps)):
        if step:
            # Dense@sparse via its transposed sparse@dense form: O(n|E|), not O(n^3).
            power = np.asarray(transition_sparse.T.dot(power.T).T, dtype=np.float32)
        output[:, :, step] = power
    return output


def make_batch(cfg: Config, size: int, records: int, seed: int) -> NarBatch:
    """Generate a uniform-N batch with fresh keys, values, roles, and node relabellings."""
    import torch

    records = int(records)
    if records <= 0 or records % 2:
        raise ValueError("records must be positive and even")
    rng = np.random.default_rng(int(seed))
    nodes = cfg.nodes_for_n(records)
    xs = np.zeros((size, nodes, cfg.feature_dim), dtype=np.float32)
    adjs = np.zeros((size, nodes, nodes), dtype=np.float32)
    rrwps = np.zeros((size, nodes, nodes, cfg.rrwp_steps), dtype=np.float32)
    central_indices = np.zeros(size, dtype=np.int64)
    query_indices = np.zeros(size, dtype=np.int64)
    target_indices = np.zeros(size, dtype=np.int64)
    record_masks = np.zeros((size, nodes), dtype=bool)
    record_roles = np.full((size, nodes), -1, dtype=np.int64)
    semantic_labels = np.zeros(size, dtype=np.int64)
    structural_labels = np.zeros(size, dtype=np.int64)
    n_records = np.full(size, records, dtype=np.int64)

    value_start = cfg.key_vocab
    query_flag = value_start + cfg.semantic_classes
    record_flag = query_flag + 1
    centre_flag = query_flag + 2
    intermediate_flag = query_flag + 3
    gadget_flag = query_flag + 4

    for graph in range(int(size)):
        central, intermediate, query = 0, 1, 2
        adj = np.zeros((nodes, nodes), dtype=np.float32)
        x = np.zeros((nodes, cfg.feature_dim), dtype=np.float32)
        role_by_node = np.full(nodes, -1, dtype=np.int64)
        record_mask = np.zeros(nodes, dtype=bool)
        add_undirected(adj, query, intermediate)
        add_undirected(adj, intermediate, central)

        roles = np.asarray([0] * (records // 2) + [1] * (records // 2), dtype=np.int64)
        rng.shuffle(roles)
        keys = rng.choice(cfg.key_vocab, size=records, replace=False)
        values = rng.integers(0, cfg.semantic_classes, size=records)
        target_slot = int(rng.integers(0, records))
        target_old = -1

        for slot in range(records):
            record = 3 + 4 * slot
            first, second, third = record + 1, record + 2, record + 3
            add_undirected(adj, central, record)
            add_undirected(adj, record, first)
            add_undirected(adj, record, second)
            if int(roles[slot]) == 0:
                # Triangle at the record plus a tail: an odd-return structural signature.
                add_undirected(adj, first, second)
                add_undirected(adj, second, third)
            else:
                # Degree-matched four-cycle at the record: no three-step return.
                add_undirected(adj, first, third)
                add_undirected(adj, third, second)
            x[record, int(keys[slot])] = 1.0
            x[record, value_start + int(values[slot])] = 1.0
            x[record, record_flag] = 1.0
            x[[first, second, third], gadget_flag] = 1.0
            record_mask[record] = True
            role_by_node[record] = int(roles[slot])
            if slot == target_slot:
                target_old = record
                semantic_labels[graph] = int(values[slot])
                structural_labels[graph] = int(roles[slot])

        x[query, int(keys[target_slot])] = 1.0
        x[query, query_flag] = 1.0
        x[central, centre_flag] = 1.0
        x[intermediate, intermediate_flag] = 1.0
        if target_old < 0:
            raise RuntimeError("target record was not assigned")

        # Every graph is independently relabelled; no array index can carry task information.
        order = rng.permutation(nodes)
        inverse = np.empty(nodes, dtype=np.int64)
        inverse[order] = np.arange(nodes)
        x = x[order]
        adj = adj[order][:, order]
        role_by_node = role_by_node[order]
        record_mask = record_mask[order]
        xs[graph] = x
        adjs[graph] = adj
        rrwps[graph] = rrwp_from_adj(adj, cfg.rrwp_steps)
        central_indices[graph] = int(inverse[central])
        query_indices[graph] = int(inverse[query])
        target_indices[graph] = int(inverse[target_old])
        record_masks[graph] = record_mask
        record_roles[graph] = role_by_node

    return NarBatch(
        x=torch.from_numpy(xs),
        adj=torch.from_numpy(adjs),
        rrwp=torch.from_numpy(rrwps),
        central_idx=torch.from_numpy(central_indices),
        query_idx=torch.from_numpy(query_indices),
        target_idx=torch.from_numpy(target_indices),
        record_mask=torch.from_numpy(record_masks),
        record_role=torch.from_numpy(record_roles),
        y_semantic=torch.from_numpy(semantic_labels),
        y_structural=torch.from_numpy(structural_labels),
        n_records=torch.from_numpy(n_records),
    )


def transpose_pair_tensor(tensor: Any, left: int, right: int) -> Any:
    if int(left) == int(right):
        return tensor.clone()
    order = list(range(int(tensor.size(0))))
    order[int(left)], order[int(right)] = order[int(right)], order[int(left)]
    return tensor[order][:, order].clone()


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


# ======================================================================================
# Official GRIT with trained support masks
# ======================================================================================


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


class OfficialNARGRIT:
    """Late-bound placeholder so data/tests do not require a local GRIT installation."""


def build_model_class() -> type:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from grit.layer.grit_layer import GritTransformerLayer

    class _OfficialNARGRIT(nn.Module):
        def __init__(self, cfg: Config, model_name: str, width: int) -> None:
            super().__init__()
            if model_name not in MODEL_RADII:
                raise ValueError(model_name)
            self.cfg = cfg
            self.model_name = str(model_name)
            self.radius = MODEL_RADII[model_name]
            self.width = int(width)
            self.L, self.H = cfg.layers, cfg.heads
            self.dh = self.width // self.H
            self.input_encoder = nn.Linear(cfg.feature_dim, self.width)
            self.node_rrwp_encoder = nn.Linear(cfg.rrwp_steps, self.width, bias=False)
            self.pair_rrwp_encoder = nn.Linear(cfg.rrwp_steps, self.width, bias=False)
            self.edge_type_encoder = nn.Embedding(3, self.width)
            nn.init.xavier_uniform_(self.node_rrwp_encoder.weight)
            nn.init.xavier_uniform_(self.pair_rrwp_encoder.weight)
            layer_cfg = grit_layer_cfg(update_e=True)
            self.layers = nn.ModuleList(
                [
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
                ]
            )
            self.semantic_head = nn.Linear(self.width, cfg.semantic_classes)
            self.structural_head = nn.Linear(self.width, cfg.structural_classes)

        @property
        def attention_layers(self) -> list[Any]:
            return [layer.attention for layer in self.layers]

        def _pyg_batch(self, batch: NarBatch) -> Any:
            batch_size, nodes = int(batch.x.size(0)), int(batch.x.size(1))
            support = khop_support(batch.adj, self.radius)
            graph_idx, source_local, destination_local = support.nonzero(as_tuple=True)
            source = graph_idx * nodes + source_local
            destination = graph_idx * nodes + destination_local
            is_self = source_local == destination_local
            is_graph_edge = batch.adj[graph_idx, source_local, destination_local] > 0
            edge_type = torch.where(is_self, 0, torch.where(is_graph_edge, 1, 2)).long()
            edge_rrwp = batch.rrwp[graph_idx, source_local, destination_local]
            diagonal = torch.arange(nodes, device=batch.x.device)
            node_rrwp = batch.rrwp[:, diagonal, diagonal].reshape(
                batch_size * nodes, self.cfg.rrwp_steps
            )
            data = Data(num_nodes=batch_size * nodes)
            data.x = self.input_encoder(batch.x.reshape(batch_size * nodes, -1))
            data.x = data.x + self.node_rrwp_encoder(node_rrwp)
            data.edge_index = torch.stack([source, destination], dim=0)
            self.last_edge_index = data.edge_index
            self.last_support_graph = graph_idx
            self.last_support_source_local = source_local
            self.last_support_destination_local = destination_local
            data.edge_attr = self.edge_type_encoder(edge_type) + self.pair_rrwp_encoder(edge_rrwp)
            data.batch = torch.arange(batch_size, device=batch.x.device).repeat_interleave(nodes)
            degree = torch.zeros(batch_size * nodes, device=batch.x.device)
            degree.index_add_(0, destination, torch.ones_like(destination, dtype=torch.float32))
            data.deg = degree
            data.log_deg = torch.log(degree + 1.0)
            data.graph_num_nodes = torch.full(
                (batch_size,), nodes, dtype=torch.long, device=batch.x.device
            )
            return data

        def node_states(self, batch: NarBatch) -> Any:
            data = self._pyg_batch(batch)
            for layer in self.layers:
                data = layer(data)
            return data.x.reshape(len(batch), batch.x.size(1), self.width)

        def forward(self, batch: NarBatch) -> dict[str, Any]:
            states = self.node_states(batch)
            rows = torch.arange(len(batch), device=states.device)
            central = states[rows, batch.central_idx]
            return {
                "semantic": self.semantic_head(central),
                "structural": self.structural_head(central),
            }

    return _OfficialNARGRIT


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


def capture_forward(
    model: Any,
    batch: NarBatch,
    *,
    want_grad: bool,
    want_attention: bool = False,
) -> dict[str, Any]:
    import torch

    captures: list[Any] = [None] * model.L
    attentions: list[Any] = [None] * model.L
    index = {id(module): layer for layer, module in enumerate(model.attention_layers)}

    def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        layer = index[id(module)]
        captures[layer] = output[0]
        if want_attention:
            attentions[layer] = inputs[0].attn.detach().squeeze(-1)

    handles = [module.register_forward_hook(hook) for module in model.attention_layers]
    try:
        context = torch.enable_grad() if want_grad else torch.no_grad()
        with context:
            logits = model(batch)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captures):
        raise RuntimeError("an official GRIT attention hook did not fire")
    return {"logits": logits, "wV": captures, "attention": attentions}


def set_seed(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def dual_loss(logits: Mapping[str, Any], batch: NarBatch, *, reduction: str = "mean") -> Any:
    import torch.nn.functional as F

    semantic = F.cross_entropy(logits["semantic"], batch.y_semantic.long(), reduction=reduction)
    structural = F.cross_entropy(logits["structural"], batch.y_structural.long(), reduction=reduction)
    return semantic + structural


def batch_graphs(cfg: Config, records: int) -> int:
    nodes = cfg.nodes_for_n(records)
    by_nodes = max(1, cfg.max_batch_nodes // nodes)
    by_pairs = max(1, cfg.max_dense_pairs // (nodes * nodes))
    return max(1, min(cfg.batch_size, by_nodes, by_pairs))


def predict_chunks(
    model: Any,
    batch: NarBatch,
    *,
    device: Any,
    chunk_size: int,
    heads: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    import torch

    outputs: dict[str, list[Any]] = {channel: [] for channel in CHANNELS}
    context = ablate_heads(model, heads or ()) if heads else contextlib.nullcontext()
    with context, torch.no_grad():
        for start in range(0, len(batch), int(chunk_size)):
            part = batch.slice(start, min(start + int(chunk_size), len(batch))).to(device)
            logits = model(part)
            for channel in CHANNELS:
                outputs[channel].append(logits[channel].detach().cpu())
    return {channel: torch.cat(values, dim=0) for channel, values in outputs.items()}


def evaluate_n(
    model: Any,
    cfg: Config,
    *,
    records: int,
    graphs: int,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    import torch.nn.functional as F

    batch = make_batch(cfg, graphs, records, seed)
    logits = predict_chunks(
        model,
        batch,
        device=device,
        chunk_size=batch_graphs(cfg, records),
    )
    output: dict[str, Any] = {"N": int(records), "graphs": int(graphs)}
    for channel, labels in (
        ("semantic", batch.y_semantic),
        ("structural", batch.y_structural),
    ):
        output[f"{channel}_loss"] = float(
            F.cross_entropy(logits[channel], labels.long(), reduction="mean")
        )
        output[f"{channel}_accuracy"] = float(
            (logits[channel].argmax(dim=-1) == labels).float().mean()
        )
    output["loss"] = output["semantic_loss"] + output["structural_loss"]
    output["mean_accuracy"] = 0.5 * (
        output["semantic_accuracy"] + output["structural_accuracy"]
    )
    return output


def evaluate_grid(
    model: Any,
    cfg: Config,
    *,
    ns: Sequence[int],
    graphs: int,
    seed: int,
    device: Any,
) -> list[dict[str, Any]]:
    return [
        evaluate_n(
            model,
            cfg,
            records=int(records),
            graphs=int(graphs),
            seed=int(seed) + int(records) * 10_007,
            device=device,
        )
        for records in ns
    ]


def checkpoint_path(run_dir: Path, cfg: Config, model_name: str, width: int, seed: int) -> Path:
    return (
        run_dir
        / "checkpoints"
        / f"{model_name}__d{width}__seed_{seed}__{config_fingerprint(cfg)}.pt"
    )


def train_model(
    cfg: Config,
    *,
    model_name: str,
    width: int,
    seed: int,
    run_dir: Path,
    device: Any,
    force: bool,
    load_only: bool,
) -> tuple[Any, dict[str, Any]]:
    import torch

    global OfficialNARGRIT
    if OfficialNARGRIT.__name__ == "OfficialNARGRIT":
        OfficialNARGRIT = build_model_class()
    path = checkpoint_path(run_dir, cfg, model_name, width, seed)
    set_seed(seed + width * 1009 + MODEL_ORDER.index(model_name) * 100_003)
    model = OfficialNARGRIT(cfg, model_name, width).to(device)
    if path.exists() and not force:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("fingerprint") != config_fingerprint(cfg):
            raise RuntimeError(f"checkpoint fingerprint mismatch at {path}")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        print(f"[train {model_name} d={width} seed={seed}] loaded {path}", flush=True)
        return model, payload
    if load_only:
        raise FileNotFoundError(f"analysis requested but checkpoint is missing: {path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_objective = float("inf")
    best_state = None
    best_grid: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    good_checks = 0
    rng = np.random.default_rng(seed * 1_000_003 + width)
    train_ns_array = np.asarray(cfg.train_ns, dtype=np.int64)
    train_weights = train_ns_array.astype(np.float64) ** cfg.train_n_sampling_exponent
    train_probabilities = train_weights / train_weights.sum()
    started = time.time()
    for step in range(1, cfg.steps + 1):
        # First learn the small-map algorithm shared by every support.  Thereafter all models see
        # the identical load distribution, with enough low-load examples to avoid impossible
        # high-load gradients erasing the required matched-performance baseline.
        records = (
            int(min(cfg.train_ns))
            if step <= cfg.warmup_steps
            else int(rng.choice(train_ns_array, p=train_probabilities))
        )
        graphs = batch_graphs(cfg, records)
        batch = make_batch(
            cfg,
            graphs,
            records,
            seed=seed * 10_000_019 + step * 101 + records,
        ).to(device)
        model.train()
        logits = model(batch)
        loss = dual_loss(logits, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 1 or step % cfg.eval_every == 0 or step == cfg.steps:
            model.eval()
            grid = evaluate_grid(
                model,
                cfg,
                ns=cfg.train_ns,
                graphs=cfg.validation_graphs,
                seed=50_000 + seed,
                device=device,
            )
            loss_by_n = {int(row["N"]): float(row["loss"]) for row in grid}
            objective = float(
                sum(
                    float(probability) * loss_by_n[int(records_value)]
                    for probability, records_value in zip(train_probabilities, train_ns_array)
                )
            )
            low_n = next(row for row in grid if int(row["N"]) == min(cfg.train_ns))
            history.append(
                {
                    "step": step,
                    "train_N": records,
                    "train_graphs": graphs,
                    "train_loss": float(loss.detach().cpu()),
                    "validation_loss": objective,
                    "warmup": bool(step <= cfg.warmup_steps),
                    "low_N_min_accuracy": float(
                        min(low_n["semantic_accuracy"], low_n["structural_accuracy"])
                    ),
                    "elapsed_s": time.time() - started,
                }
            )
            print(
                f"[train {model_name} d={width} seed={seed}] {step:4d}/{cfg.steps} "
                f"loss={float(loss.detach().cpu()):.4f} "
                f"N{min(cfg.train_ns)}="
                f"{min(low_n['semantic_accuracy'], low_n['structural_accuracy']):.3f}",
                flush=True,
            )
            if objective < best_objective:
                best_objective = objective
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                best_grid = grid
            good_checks = (
                good_checks + 1
                if min(low_n["semantic_accuracy"], low_n["structural_accuracy"]) >= 0.995
                and step >= cfg.min_steps
                else 0
            )
            if good_checks >= cfg.patience_checks:
                print(f"[train {model_name} d={width} seed={seed}] early stop", flush=True)
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint candidate")
    model.load_state_dict(best_state)
    model.eval()
    heldout = evaluate_grid(
        model,
        cfg,
        ns=cfg.eval_ns,
        graphs=cfg.heldout_graphs,
        seed=800_000 + seed,
        device=device,
    )
    payload = {
        "version": EXPERIMENT_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "official_grit_commit": OFFICIAL_GRIT_COMMIT,
        "model_name": model_name,
        "width": int(width),
        "seed": int(seed),
        "config": asdict(cfg),
        "state_dict": best_state,
        "best_validation": best_grid,
        "heldout": heldout,
        "history": history,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    write_csv(run_dir / "tables" / f"training_{model_name}_d{width}_seed_{seed}.csv", history)
    print(f"[train {model_name} d={width} seed={seed}] cached {path}", flush=True)
    return model, payload


def checkpoint_gate(cfg: Config, payload: Mapping[str, Any]) -> float:
    selected = [
        row
        for row in payload.get("heldout", [])
        if int(row["N"]) == min(cfg.train_ns)
    ]
    return (
        float(min(selected[0]["semantic_accuracy"], selected[0]["structural_accuracy"]))
        if selected
        else 0.0
    )


# ======================================================================================
# Factor-isolating interventions and head-resolved causal measurements
# ======================================================================================


def clone_batch(batch: NarBatch) -> NarBatch:
    return NarBatch(
        **{name: getattr(batch, name).clone() for name in NarBatch.__dataclass_fields__}
    )


def _semantic_replica(batch: NarBatch, cfg: Config, donor_index: int) -> NarBatch:
    """Change only the selected record's value, never its key, role, or graph position."""
    changed = clone_batch(batch)
    value = changed.x[:, :, cfg.key_vocab : cfg.key_vocab + cfg.semantic_classes]
    rows = range(len(changed))
    for graph in rows:
        target = int(changed.target_idx[graph])
        original = int(changed.y_semantic[graph])
        donor = (original + 1 + int(donor_index)) % cfg.semantic_classes
        value[graph, target].zero_()
        value[graph, target, donor] = 1.0
    return changed


def _opposite_role_partner(batch: NarBatch, graph: int, donor_index: int) -> int:
    import torch

    target = int(batch.target_idx[graph])
    role = int(batch.record_role[graph, target])
    candidates = torch.nonzero(
        batch.record_mask[graph] & (batch.record_role[graph] == 1 - role), as_tuple=False
    ).flatten()
    if not len(candidates):
        raise RuntimeError("a graph has no degree-matched opposite-role record")
    return int(candidates[int(donor_index) % len(candidates)])


def _structural_replica(
    batch: NarBatch,
    donor_index: int,
    *,
    conjugate_support: bool,
) -> NarBatch:
    """Transpose target/opposite-role RRWP; optionally transpose graph support as carriage does."""
    changed = clone_batch(batch)
    for graph in range(len(changed)):
        target = int(changed.target_idx[graph])
        partner = _opposite_role_partner(batch, graph, donor_index)
        changed.rrwp[graph] = transpose_pair_tensor(changed.rrwp[graph], target, partner)
        if conjugate_support:
            changed.adj[graph] = transpose_pair_tensor(changed.adj[graph], target, partner)
    return changed


def intervention_replicas(
    batch: NarBatch,
    cfg: Config,
    channel: str,
    donors: int,
    *,
    structural_support: str,
) -> list[NarBatch]:
    replicas = [clone_batch(batch)]
    for donor in range(int(donors)):
        if channel == "semantic":
            replicas.append(_semantic_replica(batch, cfg, donor))
        elif channel == "structural":
            replicas.append(
                _structural_replica(
                    batch,
                    donor,
                    conjugate_support=structural_support == "conjugated",
                )
            )
        else:
            raise ValueError(channel)
    return replicas


def verify_interventions(batch: NarBatch, cfg: Config) -> dict[str, float]:
    import torch

    semantic = _semantic_replica(batch, cfg, 0)
    structural = _structural_replica(batch, 0, conjugate_support=False)
    structural_full = _structural_replica(batch, 0, conjugate_support=True)
    value = slice(cfg.key_vocab, cfg.key_vocab + cfg.semantic_classes)
    checks = {
        "semantic_adj_max": float((semantic.adj - batch.adj).abs().max()),
        "semantic_rrwp_max": float((semantic.rrwp - batch.rrwp).abs().max()),
        "structural_x_max": float((structural.x - batch.x).abs().max()),
        "structural_frozen_adj_max": float((structural.adj - batch.adj).abs().max()),
        "structural_full_x_max": float((structural_full.x - batch.x).abs().max()),
        "semantic_changed_values": float(
            torch.count_nonzero((semantic.x[:, :, value] - batch.x[:, :, value]).abs()).item()
        ),
    }
    if max(
        checks["semantic_adj_max"],
        checks["semantic_rrwp_max"],
        checks["structural_x_max"],
        checks["structural_frozen_adj_max"],
        checks["structural_full_x_max"],
    ) > 0:
        raise RuntimeError(f"factor-isolation check failed: {checks}")
    return checks


def transport_head_scores(
    model: Any,
    batch: NarBatch,
    cfg: Config,
    channel: str,
    *,
    device: Any,
) -> Any:
    """Repository Method A on the sole intervened source, returning ``[L,H]`` scores.

    The clean replica and all nuisance donors share a forward. Readout gradients are taken at
    the clean routed ``wV``. This isolated synthetic control retains the historical coherent-gross
    aggregation; central production specialisation now uses eventwise-gross.
    """
    import torch

    replicas = intervention_replicas(
        batch,
        cfg,
        channel,
        cfg.score_donors,
        structural_support="frozen",
    )
    combined = concat_batches(replicas).to(device)
    captured = capture_forward(model, combined, want_grad=True)
    batch_size, nodes = len(batch), int(batch.x.size(1))
    output = captured["logits"][channel]
    scores = torch.zeros(model.L, model.H, device=device)
    for layer, routed in enumerate(captured["wV"]):
        routed_view = routed.view(len(replicas), batch_size, nodes, model.H, model.dh)
        delta = routed_view[0] - routed_view[1:].mean(dim=0)  # [B,n,H,dh]
        projections = []
        for target_output in range(int(output.size(1))):
            gradient = torch.autograd.grad(
                output[:batch_size, target_output].sum(),
                routed,
                retain_graph=True,
            )[0]
            phi = gradient.view(len(replicas), batch_size, nodes, model.H, model.dh)[0]
            projections.append(torch.einsum("bnhd,bnhd->bnh", phi, delta))
        functional = torch.stack(projections).square().sum(dim=0).sqrt()  # [B,n,H]
        scores[layer] = functional.sum(dim=1).mean(dim=0)
    return scores.detach().cpu()


def _exact_intervention_effects(
    model: Any,
    batch: NarBatch,
    cfg: Config,
    channel: str,
    *,
    device: Any,
    heads: Sequence[tuple[int, int]] = (),
) -> dict[str, Any]:
    """Exact output and loss change for target-source intervention, averaged over donors."""
    import torch
    import torch.nn.functional as F

    replicas = intervention_replicas(
        batch,
        cfg,
        channel,
        cfg.score_donors,
        structural_support="conjugated" if channel == "structural" else "frozen",
    )
    combined = concat_batches(replicas).to(device)
    context = ablate_heads(model, heads) if heads else contextlib.nullcontext()
    with context, torch.no_grad():
        logits = model(combined)[channel]
    replica_logits = logits.view(len(replicas), len(batch), -1)
    clean, corrupt = replica_logits[0], replica_logits[1:]
    labels = getattr(batch.to(device), f"y_{channel}").long()
    clean_loss = F.cross_entropy(clean, labels, reduction="none")
    corrupt_loss = torch.stack(
        [F.cross_entropy(item, labels, reduction="none") for item in corrupt], dim=0
    )
    functional = torch.linalg.vector_norm(clean - corrupt.mean(dim=0), dim=-1)
    beneficial = corrupt_loss.mean(dim=0) - clean_loss
    return {
        "functional": functional.cpu(),
        "beneficial": beneficial.cpu(),
        "clean_loss": clean_loss.cpu(),
    }


def cross_intervention_effects(
    model: Any,
    batch: NarBatch,
    cfg: Config,
    *,
    device: Any,
) -> dict[str, Any]:
    """Full intervention-factor × output matrix, retained as a specificity control."""
    import torch
    import torch.nn.functional as F

    result: dict[str, Any] = {}
    for factor in CHANNELS:
        replicas = intervention_replicas(
            batch,
            cfg,
            factor,
            cfg.score_donors,
            structural_support="conjugated" if factor == "structural" else "frozen",
        )
        combined = concat_batches(replicas).to(device)
        with torch.no_grad():
            logits = model(combined)
        result[factor] = {}
        for output_channel in CHANNELS:
            values = logits[output_channel].view(len(replicas), len(batch), -1)
            clean, corrupt = values[0], values[1:]
            labels = getattr(combined, f"y_{output_channel}")[: len(batch)].long()
            clean_loss = F.cross_entropy(clean, labels, reduction="none")
            corrupt_loss = torch.stack(
                [F.cross_entropy(item, labels, reduction="none") for item in corrupt], dim=0
            )
            result[factor][output_channel] = {
                "functional": torch.linalg.vector_norm(
                    clean - corrupt.mean(dim=0), dim=-1
                ).cpu(),
                "loss_increase": (corrupt_loss.mean(dim=0) - clean_loss).cpu(),
            }
    return result


def noop_transport_max(model: Any, batch: NarBatch, *, device: Any) -> float:
    """Identical replicas must produce zero within-forward routed transport."""
    combined = concat_batches([clone_batch(batch), clone_batch(batch)]).to(device)
    captured = capture_forward(model, combined, want_grad=False)
    batch_size, nodes = len(batch), int(batch.x.size(1))
    maximum = 0.0
    for routed in captured["wV"]:
        view = routed.view(2, batch_size, nodes, model.H, model.dh)
        maximum = max(maximum, float((view[0] - view[1]).abs().max().cpu()))
    return maximum


def clean_head_ablation(
    model: Any,
    batch: NarBatch,
    *,
    device: Any,
) -> dict[str, Any]:
    """Pre-head clean-input ablation impact for every head and both output channels."""
    import torch
    import torch.nn.functional as F

    clean = batch.to(device)
    with torch.no_grad():
        base = model(clean)
    result: dict[str, Any] = {}
    for channel in CHANNELS:
        labels = getattr(clean, f"y_{channel}").long()
        base_loss = F.cross_entropy(base[channel], labels, reduction="none")
        functional = torch.zeros(model.L, model.H, len(batch))
        loss_delta = torch.zeros_like(functional)
        for layer in range(model.L):
            for head in range(model.H):
                with ablate_heads(model, [(layer, head)]), torch.no_grad():
                    changed = model(clean)[channel]
                functional[layer, head] = torch.linalg.vector_norm(
                    changed - base[channel], dim=-1
                ).cpu()
                loss_delta[layer, head] = (
                    F.cross_entropy(changed, labels, reduction="none") - base_loss
                ).cpu()
        result[channel] = {"functional": functional, "loss_delta": loss_delta}
    return result


def head_resolved_carriage(
    model: Any,
    batch: NarBatch,
    cfg: Config,
    *,
    device: Any,
) -> dict[str, Any]:
    """Aggregate carriage and its decrease after ablating each routed head."""
    import torch

    result: dict[str, Any] = {}
    for channel in CHANNELS:
        intact = _exact_intervention_effects(model, batch, cfg, channel, device=device)
        functional_loss = torch.zeros(model.L, model.H, len(batch))
        beneficial_loss = torch.zeros_like(functional_loss)
        for layer in range(model.L):
            for head in range(model.H):
                changed = _exact_intervention_effects(
                    model, batch, cfg, channel, device=device, heads=[(layer, head)]
                )
                functional_loss[layer, head] = (
                    intact["functional"] - changed["functional"]
                )
                beneficial_loss[layer, head] = (
                    intact["beneficial"] - changed["beneficial"]
                )
        result[channel] = {
            "functional": intact["functional"],
            "beneficial": intact["beneficial"],
            "functional_head_loss": functional_loss,
            "beneficial_head_loss": beneficial_loss,
        }
    return result


def attention_target_selection(model: Any, batch: NarBatch, *, device: Any) -> dict[str, Any]:
    """Attention from the output centre to the requested record versus all other records."""
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


def calibrated_jd(semantic: Any, structural: Any) -> tuple[Any, Any, Any, Any]:
    """Channel-mean calibration followed by joint influence J and relative selectivity D."""
    import torch

    semantic = torch.as_tensor(semantic, dtype=torch.float32)
    structural = torch.as_tensor(structural, dtype=torch.float32)
    sem_rel = semantic / semantic.mean().clamp_min(EPS)
    str_rel = structural / structural.mean().clamp_min(EPS)
    joint = 0.5 * (sem_rel + str_rel)
    selectivity = (sem_rel - str_rel) / (sem_rel + str_rel).clamp_min(EPS)
    return sem_rel, str_rel, joint, selectivity


def select_head_families(scores: Mapping[str, Any], size: int) -> dict[str, list[tuple[int, int]]]:
    import torch

    _, _, joint, selectivity = calibrated_jd(scores["semantic"], scores["structural"])
    flat_j, flat_d = joint.flatten(), selectivity.flatten()
    reliable = flat_j >= DJ_RELIABILITY_FLOOR
    indices = list(range(int(flat_j.numel())))
    semantic = sorted(indices, key=lambda i: (bool(reliable[i]), float(flat_d[i])), reverse=True)
    sem = semantic[: int(size)]
    structural = sorted(
        [index for index in indices if index not in sem],
        key=lambda i: (bool(reliable[i]), -float(flat_d[i])),
        reverse=True,
    )[: int(size)]

    def decode(values: Sequence[int]) -> list[tuple[int, int]]:
        heads = int(joint.size(1))
        return [(int(value // heads), int(value % heads)) for value in values]

    return {"semantic": decode(sem), "structural": decode(structural)}


def family_ablation(
    model: Any,
    batch: NarBatch,
    families: Mapping[str, Sequence[tuple[int, int]]],
    *,
    device: Any,
    chunk_size: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    result: dict[str, Any] = {
        family: {
            channel: {"loss_delta": [], "accuracy_drop": []} for channel in CHANNELS
        }
        for family in families
    }
    for start in range(0, len(batch), int(chunk_size)):
        clean = batch.slice(start, min(start + int(chunk_size), len(batch))).to(device)
        with torch.no_grad():
            base = model(clean)
        for family, heads in families.items():
            with ablate_heads(model, heads), torch.no_grad():
                changed = model(clean)
            for channel in CHANNELS:
                labels = getattr(clean, f"y_{channel}").long()
                result[family][channel]["loss_delta"].append(
                    (
                        F.cross_entropy(changed[channel], labels, reduction="none")
                        - F.cross_entropy(base[channel], labels, reduction="none")
                    ).cpu()
                )
                result[family][channel]["accuracy_drop"].append(
                    (
                        (base[channel].argmax(-1) == labels).float()
                        - (changed[channel].argmax(-1) == labels).float()
                    ).cpu()
                )
    for family in result:
        for channel in CHANNELS:
            for measure in ("loss_delta", "accuracy_drop"):
                result[family][channel][measure] = torch.cat(
                    result[family][channel][measure], dim=0
                )
    return result


# ======================================================================================
# Mechanistic cache
# ======================================================================================


def analysis_path(run_dir: Path, cfg: Config, model_name: str, seed: int) -> Path:
    return run_dir / "analysis" / f"{model_name}__d{cfg.analysis_width}__seed_{seed}.pt"


def analyze_model(
    model: Any,
    payload: Mapping[str, Any],
    cfg: Config,
    model_name: str,
    seed: int,
    *,
    run_dir: Path,
    device: Any,
    force: bool,
) -> dict[str, Any]:
    import torch

    path = analysis_path(run_dir, cfg, model_name, seed)
    if path.exists() and not force:
        print(f"[analysis cache] {path}", flush=True)
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("fingerprint") != config_fingerprint(cfg):
            raise RuntimeError(
                f"analysis fingerprint mismatch at {path}; rerun with --force-analysis "
                "or choose a fresh run name"
            )
        return cached
    result: dict[str, Any] = {
        "version": EXPERIMENT_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "model_name": model_name,
        "width": cfg.analysis_width,
        "seed": seed,
        "heldout": payload["heldout"],
        "by_n": {},
    }
    for records in cfg.mechanistic_ns:
        print(f"[analysis {model_name} seed={seed}] N={records}", flush=True)
        score_batch = make_batch(
            cfg, cfg.score_graphs, records, 1_100_000 + 10_000 * seed + records
        )
        checks = verify_interventions(score_batch, cfg)
        scores = {
            channel: transport_head_scores(
                model, score_batch, cfg, channel, device=device
            )
            for channel in CHANNELS
        }
        causal_batch = make_batch(
            cfg,
            cfg.analysis_batch_size,
            records,
            1_200_000 + 10_000 * seed + records,
        )
        ablations = clean_head_ablation(model, causal_batch, device=device)
        carriage = head_resolved_carriage(model, causal_batch, cfg, device=device)
        cross_effects = cross_intervention_effects(
            model, causal_batch, cfg, device=device
        )
        attention = attention_target_selection(model, causal_batch, device=device)
        families = select_head_families(scores, cfg.family_size)
        family_batch = make_batch(
            cfg, cfg.ablation_graphs, records, 1_300_000 + 10_000 * seed + records
        )
        family = family_ablation(
            model,
            family_batch,
            families,
            device=device,
            chunk_size=batch_graphs(cfg, records),
        )
        noop_max = noop_transport_max(
            model, score_batch.slice(0, min(2, len(score_batch))), device=device
        )
        if noop_max > 1.0e-5:
            raise RuntimeError(
                f"identical replicas produced routed transport {noop_max:.3e}; "
                "within-forward score baseline is not numerically reliable"
            )
        result["by_n"][int(records)] = {
            "scores": scores,
            "ablations": ablations,
            "carriage": carriage,
            "cross_intervention": cross_effects,
            "attention": attention,
            "families": families,
            "family_ablation": family,
            "intervention_checks": checks,
            "noop_transport_max": noop_max,
        }
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    print(f"[analysis] cached {path}", flush=True)
    return result


# ======================================================================================
# Paper figures and tabular exports
# ======================================================================================


MODEL_COLOURS = {"1hop": "#6550a4", "2hop": "#2b8cbe", "dense": "#d7301f"}
MODEL_MARKERS = {"1hop": "o", "2hop": "s", "dense": "D"}
LAYER_COLOURS = ("#443983", "#21918c")


def configure_plots() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: Any, figure_dir: Path, stem: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = figure_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"[figure] {path}", flush=True)


def mean_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return float("nan"), float("nan")
    mean = float(finite.mean())
    error = 1.96 * float(finite.std(ddof=1)) / math.sqrt(len(finite)) if len(finite) > 1 else 0.0
    return mean, error


def rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for index in range(len(unique)):
            selected = inverse == index
            ranks[selected] = ranks[selected].mean()
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    xr, yr = rankdata(x[keep]), rankdata(y[keep])
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _annotate_panel(axis: Any, letter: str, title: str) -> None:
    axis.set_title(f"{letter}  {title}", loc="left", pad=8)


def _suptitle(fig: Any, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.055, y=1.02, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.055, 0.982, subtitle, ha="left", va="top", color="#5c5c5c", fontsize=10.5)


def performance_rows(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for heldout in payload["heldout"]:
            rows.append(
                {
                    "model": payload["model_name"],
                    "width": int(payload["width"]),
                    "seed": int(payload["seed"]),
                    "parameters": int(payload["parameters"]),
                    **{key: value for key, value in heldout.items()},
                }
            )
    return rows


def analysis_head_rows(analyses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analysis in analyses:
        for records_key, cell in analysis["by_n"].items():
            records = int(records_key)
            scores = cell["scores"]
            sem_rel, str_rel, joint, selectivity = calibrated_jd(
                scores["semantic"], scores["structural"]
            )
            sem_ab = cell["ablations"]["semantic"]["functional"].mean(dim=-1)
            str_ab = cell["ablations"]["structural"]["functional"].mean(dim=-1)
            sem_ab_rel, str_ab_rel, ab_joint, ab_role = calibrated_jd(sem_ab, str_ab)
            sem_carriage = cell["carriage"]["semantic"]["functional_head_loss"].mean(-1)
            str_carriage = cell["carriage"]["structural"]["functional_head_loss"].mean(-1)
            attention = cell["attention"]["advantage"].mean(-1)
            ratio = cell["attention"]["ratio"].mean(-1)
            for layer in range(int(joint.size(0))):
                for head in range(int(joint.size(1))):
                    rows.append(
                        {
                            "model": analysis["model_name"],
                            "seed": int(analysis["seed"]),
                            "N": records,
                            "layer": layer,
                            "head": head,
                            "semantic_score": float(scores["semantic"][layer, head]),
                            "structural_score": float(scores["structural"][layer, head]),
                            "semantic_score_relative": float(sem_rel[layer, head]),
                            "structural_score_relative": float(str_rel[layer, head]),
                            "J": float(joint[layer, head]),
                            "D_rel": float(selectivity[layer, head]),
                            "reliable": bool(joint[layer, head] >= DJ_RELIABILITY_FLOOR),
                            "semantic_ablation": float(sem_ab[layer, head]),
                            "structural_ablation": float(str_ab[layer, head]),
                            "semantic_ablation_relative": float(sem_ab_rel[layer, head]),
                            "structural_ablation_relative": float(str_ab_rel[layer, head]),
                            "ablation_joint": float(ab_joint[layer, head]),
                            "ablation_role": float(ab_role[layer, head]),
                            "semantic_carriage_loss": float(sem_carriage[layer, head]),
                            "structural_carriage_loss": float(str_carriage[layer, head]),
                            "attention_advantage": float(attention[layer, head]),
                            "attention_ratio": float(ratio[layer, head]),
                        }
                    )
    return rows


def family_rows(analyses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analysis in analyses:
        for records_key, cell in analysis["by_n"].items():
            for family in CHANNELS:
                for channel in CHANNELS:
                    value = cell["family_ablation"][family][channel]
                    rows.append(
                        {
                            "model": analysis["model_name"],
                            "seed": int(analysis["seed"]),
                            "N": int(records_key),
                            "family": family,
                            "task": channel,
                            "loss_delta": float(value["loss_delta"].mean()),
                            "accuracy_drop": float(value["accuracy_drop"].mean()),
                            "heads": str(cell["families"][family]),
                        }
                    )
    return rows


def carriage_rows(analyses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analysis in analyses:
        for records_key, cell in analysis["by_n"].items():
            for channel in CHANNELS:
                carriage = cell["carriage"][channel]
                rows.append(
                    {
                        "model": analysis["model_name"],
                        "seed": int(analysis["seed"]),
                        "N": int(records_key),
                        "channel": channel,
                        "functional": float(carriage["functional"].mean()),
                        "beneficial": float(carriage["beneficial"].mean()),
                    }
                )
    return rows


def cross_intervention_rows(analyses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for analysis in analyses:
        for records_key, cell in analysis["by_n"].items():
            for factor in CHANNELS:
                for output_channel in CHANNELS:
                    effect = cell["cross_intervention"][factor][output_channel]
                    rows.append(
                        {
                            "model": analysis["model_name"],
                            "seed": int(analysis["seed"]),
                            "N": int(records_key),
                            "intervention": factor,
                            "output": output_channel,
                            "functional": float(effect["functional"].mean()),
                            "loss_increase": float(effect["loss_increase"].mean()),
                            "noop_transport_max": float(cell["noop_transport_max"]),
                        }
                    )
    return rows


def _group_values(
    rows: Sequence[Mapping[str, Any]],
    filters: Mapping[str, Any],
    value: str,
) -> list[float]:
    return [
        float(row[value])
        for row in rows
        if all(row.get(key) == expected for key, expected in filters.items())
    ]


def plot_performance(rows: Sequence[Mapping[str, Any]], cfg: Config, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    widths = list(cfg.widths)
    fig, axes = plt.subplots(2, len(widths), figsize=(6.3 * len(widths), 8.0), sharex=True)
    axes = np.asarray(axes).reshape(2, len(widths))
    for column, width in enumerate(widths):
        for row_index, channel in enumerate(CHANNELS):
            axis = axes[row_index, column]
            for model in cfg.models:
                means, errors = [], []
                for records in cfg.eval_ns:
                    values = _group_values(
                        rows,
                        {"model": model, "width": width, "N": records},
                        f"{channel}_accuracy",
                    )
                    mean, error = mean_ci(values)
                    means.append(mean)
                    errors.append(error)
                colour = MODEL_COLOURS[model]
                axis.errorbar(
                    cfg.eval_ns,
                    means,
                    yerr=errors,
                    color=colour,
                    marker=MODEL_MARKERS[model],
                    lw=2.2,
                    ms=5.5,
                    capsize=2.5,
                    label=model.replace("hop", "-hop"),
                )
            chance = 1 / (cfg.semantic_classes if channel == "semantic" else 2)
            axis.axhline(chance, color="#777777", ls=":", lw=1.2, label="chance")
            axis.set_xscale("log", base=2)
            axis.set_xticks(cfg.eval_ns, [str(value) for value in cfg.eval_ns])
            axis.set_ylim(max(0, chance - 0.08), 1.035)
            _annotate_panel(
                axis,
                chr(ord("A") + row_index * len(widths) + column),
                f"{channel.capitalize()} recall · width {width}",
            )
            axis.set_ylabel("Held-out accuracy")
            if row_index == 1:
                axis.set_xlabel("Number of key–value records, N")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    _suptitle(
        fig,
        "Trained attention support controls associative-recall capacity",
        "Two-layer, parameter-matched official GRITs; points are seed means and bars are 95% intervals.",
    )
    fig.subplots_adjust(top=0.88, bottom=0.13, wspace=0.20, hspace=0.27)
    save_figure(fig, figure_dir, "01_nar_capacity_phase_diagram")
    plt.close(fig)


def _scatter_jd(axis: Any, rows: Sequence[Mapping[str, Any]], records: int) -> None:
    selected = [row for row in rows if int(row["N"]) == int(records)]
    for model in MODEL_ORDER:
        subset = [row for row in selected if row["model"] == model]
        axis.scatter(
            [row["D_rel"] for row in subset],
            [row["J"] for row in subset],
            s=29,
            alpha=0.72,
            c=MODEL_COLOURS[model],
            marker=MODEL_MARKERS[model],
            edgecolors="white",
            linewidths=0.35,
            label=model.replace("hop", "-hop"),
        )
    axis.axvline(0, color="#777777", lw=1)
    axis.axhline(DJ_RELIABILITY_FLOOR, color="#777777", lw=1, ls="--")
    axis.axvspan(-DJ_SELECTIVITY_THRESHOLD, DJ_SELECTIVITY_THRESHOLD, color="#bbbbbb", alpha=0.12)
    axis.set_xlim(-1.04, 1.04)
    axis.set_yscale("log")
    axis.set_xlabel(r"Selectivity $D_{rel}$  (structural $\leftarrow 0 \rightarrow$ semantic)")
    axis.set_ylabel(r"Joint influence $J$")


def plot_jd(rows: Sequence[Mapping[str, Any]], cfg: Config, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    low, high = min(cfg.mechanistic_ns), max(cfg.mechanistic_ns)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    _scatter_jd(axes[0, 0], rows, low)
    _annotate_panel(axes[0, 0], "A", f"Head roles at low load (N={low})")
    _scatter_jd(axes[0, 1], rows, high)
    _annotate_panel(axes[0, 1], "B", f"Head roles at high load (N={high})")
    for axis, measure, title, letter in (
        (axes[1, 0], "high_influence", "Concentration of influence", "C"),
        (axes[1, 1], "absolute_selectivity", "Specialisation among reliable heads", "D"),
    ):
        for model in cfg.models:
            means, errors = [], []
            for records in cfg.mechanistic_ns:
                seed_values = []
                for seed in cfg.seeds:
                    cell = [
                        row
                        for row in rows
                        if row["model"] == model and row["seed"] == seed and row["N"] == records
                    ]
                    if measure == "high_influence":
                        seed_values.append(float(np.mean([row["J"] >= 1.5 for row in cell])))
                    else:
                        reliable = [abs(row["D_rel"]) for row in cell if row["reliable"]]
                        seed_values.append(float(np.mean(reliable)) if reliable else float("nan"))
                mean, error = mean_ci(seed_values)
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
        axis.set_xscale("log", base=2)
        axis.set_xticks(cfg.mechanistic_ns, [str(value) for value in cfg.mechanistic_ns])
        axis.set_xlabel("Number of records, N")
        axis.set_ylabel("Fraction with J ≥ 1.5" if measure == "high_influence" else r"Mean $|D_{rel}|$")
        axis.set_ylim(bottom=0)
        _annotate_panel(axis, letter, title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    _suptitle(
        fig,
        "Joint influence and factor preference expose how head organisation changes with load",
        "J measures calibrated response strength; D_rel measures semantic versus structural preference only when J is reliable.",
    )
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.24, hspace=0.31)
    save_figure(fig, figure_dir, "02_joint_influence_and_selectivity")
    plt.close(fig)


def plot_causal(
    rows: Sequence[Mapping[str, Any]],
    families: Sequence[Mapping[str, Any]],
    cfg: Config,
    figure_dir: Path,
) -> dict[str, float]:
    import matplotlib.pyplot as plt

    reliable = [row for row in rows if row["reliable"]]
    rho_j = spearman([row["J"] for row in rows], [row["ablation_joint"] for row in rows])
    rho_d = spearman(
        [row["D_rel"] for row in reliable], [row["ablation_role"] for row in reliable]
    )
    carriage_joint = [
        abs(row["semantic_carriage_loss"]) + abs(row["structural_carriage_loss"])
        for row in rows
    ]
    rho_carriage = spearman([row["J"] for row in rows], carriage_joint)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0))
    plots = (
        (axes[0, 0], rows, "J", "ablation_joint", "A", "Does J predict how much a head matters?", rho_j),
        (axes[0, 1], reliable, "D_rel", "ablation_role", "B", "Does D predict which output a head serves?", rho_d),
        (axes[1, 0], rows, "J", None, "C", "Does J identify carriage-critical heads?", rho_carriage),
    )
    for axis, selected, x_key, y_key, letter, title, rho in plots:
        for model in cfg.models:
            subset = [row for row in selected if row["model"] == model]
            y = (
                [row[y_key] for row in subset]
                if y_key
                else [abs(row["semantic_carriage_loss"]) + abs(row["structural_carriage_loss"]) for row in subset]
            )
            axis.scatter(
                [row[x_key] for row in subset],
                y,
                s=27,
                alpha=0.65,
                color=MODEL_COLOURS[model],
                marker=MODEL_MARKERS[model],
                edgecolors="none",
                label=model.replace("hop", "-hop"),
            )
        axis.text(0.04, 0.94, rf"pooled $\rho={rho:.2f}$", transform=axis.transAxes, va="top")
        axis.set_xlabel(r"Joint influence $J$" if x_key == "J" else r"Score selectivity $D_{rel}$")
        axis.set_ylabel(
            "Mean cross-output ablation impact"
            if y_key == "ablation_joint"
            else ("Ablation role contrast" if y_key else "Absolute head-resolved carriage loss")
        )
        if x_key == "J":
            axis.set_xscale("log")
        else:
            axis.axvline(0, color="#777777", lw=1)
            axis.axhline(0, color="#777777", lw=1)
        _annotate_panel(axis, letter, title)

    axis = axes[1, 1]
    for model in cfg.models:
        means, errors = [], []
        for records in cfg.mechanistic_ns:
            per_seed = []
            for seed in cfg.seeds:
                chosen = [
                    row
                    for row in families
                    if row["model"] == model and row["seed"] == seed and row["N"] == records
                ]
                lookup = {(row["family"], row["task"]): row["loss_delta"] for row in chosen}
                if len(lookup) == 4:
                    interaction = 0.5 * (
                        lookup[("semantic", "semantic")]
                        - lookup[("semantic", "structural")]
                        + lookup[("structural", "structural")]
                        - lookup[("structural", "semantic")]
                    )
                    per_seed.append(float(interaction))
            mean, error = mean_ci(per_seed)
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
    axis.axhline(0, color="#777777", lw=1)
    axis.set_xscale("log", base=2)
    axis.set_xticks(cfg.mechanistic_ns, [str(value) for value in cfg.mechanistic_ns])
    axis.set_xlabel("Number of records, N")
    axis.set_ylabel("Family × output loss interaction")
    _annotate_panel(axis, "D", "Do score-selected families causally dissociate?")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    _suptitle(
        fig,
        "Intervention scores predict causal strength and causal role",
        "Score graphs and ablation graphs are independent; D_rel analyses exclude heads with J < 0.5.",
    )
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.24, hspace=0.31)
    save_figure(fig, figure_dir, "03_causal_score_validation")
    plt.close(fig)
    return {"rho_J_ablation": rho_j, "rho_D_role": rho_d, "rho_J_carriage": rho_carriage}


def _heldout_accuracy(
    performance: Sequence[Mapping[str, Any]],
    model: str,
    seed: int,
    records: int,
    width: int,
) -> float:
    values = [
        float(row["semantic_accuracy"])
        for row in performance
        if row["model"] == model
        and row["seed"] == seed
        and row["N"] == records
        and row["width"] == int(width)
    ]
    return values[0] if values else float("nan")


def plot_attention(
    rows: Sequence[Mapping[str, Any]],
    performance: Sequence[Mapping[str, Any]],
    cfg: Config,
    figure_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    for model in cfg.models:
        means, errors = [], []
        for records in cfg.mechanistic_ns:
            seed_values = []
            for seed in cfg.seeds:
                values = [
                    row["attention_ratio"]
                    for row in rows
                    if row["model"] == model
                    and row["seed"] == seed
                    and row["N"] == records
                    and row["layer"] == cfg.layers - 1
                ]
                seed_values.append(float(np.nanmean(values)))
            mean, error = mean_ci(seed_values)
            means.append(mean)
            errors.append(error)
        axes[0, 0].errorbar(
            cfg.mechanistic_ns,
            means,
            yerr=errors,
            color=MODEL_COLOURS[model],
            marker=MODEL_MARKERS[model],
            lw=2.1,
            capsize=2.5,
            label=model.replace("hop", "-hop"),
        )
    axes[0, 0].axhline(1, color="#777777", ls="--", lw=1)
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xticks(cfg.mechanistic_ns, [str(value) for value in cfg.mechanistic_ns])
    axes[0, 0].set_xlabel("Number of records, N")
    axes[0, 0].set_ylabel("Target / background attention")
    _annotate_panel(axes[0, 0], "A", "Does the final layer select the queried record?")

    selection_points = []
    for model in cfg.models:
        for seed in cfg.seeds:
            for records in cfg.mechanistic_ns:
                selected = [
                    row["attention_ratio"]
                    for row in rows
                    if row["model"] == model
                    and row["seed"] == seed
                    and row["N"] == records
                    and row["layer"] == cfg.layers - 1
                ]
                ratio = float(np.nanmean(selected))
                accuracy = _heldout_accuracy(
                    performance, model, seed, records, cfg.analysis_width
                )
                selection_points.append((model, ratio, accuracy))
    for model in cfg.models:
        points = [point for point in selection_points if point[0] == model]
        axes[0, 1].scatter(
            [point[1] for point in points],
            [point[2] for point in points],
            color=MODEL_COLOURS[model],
            marker=MODEL_MARKERS[model],
            s=42,
            alpha=0.8,
            label=model.replace("hop", "-hop"),
        )
    rho = spearman([point[1] for point in selection_points], [point[2] for point in selection_points])
    axes[0, 1].text(0.04, 0.94, rf"pooled $\rho={rho:.2f}$", transform=axes[0, 1].transAxes, va="top")
    axes[0, 1].set_xlabel("Mean target / background attention")
    axes[0, 1].set_ylabel("Semantic recall accuracy")
    _annotate_panel(axes[0, 1], "B", "Selection predicts recall")

    reliable = [row for row in rows if row["reliable"]]
    for model in cfg.models:
        selected = [row for row in reliable if row["model"] == model]
        axes[1, 0].scatter(
            [row["D_rel"] for row in selected],
            [row["attention_advantage"] for row in selected],
            color=MODEL_COLOURS[model],
            marker=MODEL_MARKERS[model],
            s=28,
            alpha=0.65,
        )
    axes[1, 0].axvline(0, color="#777777", lw=1)
    axes[1, 0].axhline(0, color="#777777", lw=1)
    axes[1, 0].set_xlabel(r"Score selectivity $D_{rel}$")
    axes[1, 0].set_ylabel("Target attention advantage")
    rho_d = spearman(
        [row["D_rel"] for row in reliable], [row["attention_advantage"] for row in reliable]
    )
    axes[1, 0].text(0.04, 0.94, rf"pooled $\rho={rho_d:.2f}$", transform=axes[1, 0].transAxes, va="top")
    _annotate_panel(axes[1, 0], "C", "Does semantic preference reflect content selection?")

    high = max(cfg.mechanistic_ns)
    matrix = np.full((len(cfg.models) * cfg.layers, cfg.heads), np.nan)
    labels = []
    for model_index, model in enumerate(cfg.models):
        for layer in range(cfg.layers):
            labels.append(f"{model.replace('hop', '-hop')} · L{layer}")
            for head in range(cfg.heads):
                values = [
                    row["attention_ratio"]
                    for row in rows
                    if row["model"] == model
                    and row["N"] == high
                    and row["layer"] == layer
                    and row["head"] == head
                ]
                matrix[model_index * cfg.layers + layer, head] = float(np.nanmean(values))
    image = axes[1, 1].imshow(matrix, aspect="auto", cmap="magma", vmin=0)
    axes[1, 1].set_xticks(range(cfg.heads), [str(head) for head in range(cfg.heads)])
    axes[1, 1].set_yticks(range(len(labels)), labels)
    axes[1, 1].set_xlabel("Head")
    fig.colorbar(image, ax=axes[1, 1], label="Target / background attention", shrink=0.88)
    _annotate_panel(axes[1, 1], "D", f"Selection map at N={high}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    _suptitle(
        fig,
        "Attention selection reveals the retrieval mechanism behind the capacity gap",
        "Ratios compare attention into the central readout from the queried record and other records.",
    )
    fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.25, hspace=0.31)
    save_figure(fig, figure_dir, "04_attention_selection_mechanism")
    plt.close(fig)


def plot_carriage(rows: Sequence[Mapping[str, Any]], cfg: Config, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), sharex=True)
    for column, channel in enumerate(CHANNELS):
        for row_index, measure in enumerate(("functional", "beneficial")):
            axis = axes[row_index, column]
            for model in cfg.models:
                means, errors = [], []
                for records in cfg.mechanistic_ns:
                    values = _group_values(
                        rows, {"model": model, "N": records, "channel": channel}, measure
                    )
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
            axis.axhline(0, color="#777777", lw=1)
            axis.set_xscale("log", base=2)
            axis.set_xticks(cfg.mechanistic_ns, [str(value) for value in cfg.mechanistic_ns])
            if row_index == 1:
                axis.set_xlabel("Number of records, N")
            axis.set_ylabel(
                "Output displacement" if measure == "functional" else "Corruption loss increase"
            )
            _annotate_panel(
                axis,
                chr(ord("A") + row_index * 2 + column),
                f"{channel.capitalize()} {measure} carriage",
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    _suptitle(
        fig,
        "Semantic and structural carriage track the factors actually used for retrieval",
        "Exact target-source effects at the central readout; bands are seed-level 95% intervals.",
    )
    fig.subplots_adjust(top=0.87, bottom=0.13, wspace=0.22, hspace=0.29)
    save_figure(fig, figure_dir, "05_aggregate_carriage")
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
    heads = analysis_head_rows(analyses)
    families = family_rows(analyses)
    carriage = carriage_rows(analyses)
    cross_effects = cross_intervention_rows(analyses)
    write_csv(table_dir / "heldout_performance.csv", performance)
    write_csv(table_dir / "head_mechanisms.csv", heads)
    write_csv(table_dir / "family_ablations.csv", families)
    write_csv(table_dir / "aggregate_carriage.csv", carriage)
    write_csv(table_dir / "cross_intervention_specificity.csv", cross_effects)
    plot_performance(performance, cfg, figure_dir)
    plot_jd(heads, cfg, figure_dir)
    causal = plot_causal(heads, families, cfg, figure_dir)
    plot_attention(heads, performance, cfg, figure_dir)
    plot_carriage(carriage, cfg, figure_dir)
    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "methodology_alignment": METHODOLOGY_ALIGNMENT,
        "intentional_synthetic_differences": METHODOLOGY_DIFFERENCES,
        "headline_statistics": causal,
        "max_noop_transport": max(
            (float(row["noop_transport_max"]) for row in cross_effects), default=float("nan")
        ),
        "reliability_rule": f"D_rel interpreted only for J >= {DJ_RELIABILITY_FLOOR}",
        "artifacts": {
            "figures": sorted(path.name for path in figure_dir.glob("*.png")),
            "tables": sorted(path.name for path in table_dir.glob("*.csv")),
        },
    }
    write_json(run_dir / "summary.json", summary)
    return summary


# ======================================================================================
# Colab/CLI orchestration
# ======================================================================================


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
    parser.add_argument("--run-name", default="nar_grit_v1")
    parser.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--phase", choices=("all", "train", "analyze", "figures"), default="all")
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--widths", default="64,128")
    parser.add_argument("--analysis-width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--train-ns", default="4,8,16,32,64")
    parser.add_argument("--eval-ns", default="4,8,12,16,24,32,48,64")
    parser.add_argument("--mechanistic-ns", default="4,16,32,64")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--min-steps", type=int, default=1500)
    parser.add_argument("--warmup-steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--score-graphs", type=int, default=8)
    parser.add_argument("--score-donors", type=int, default=3)
    parser.add_argument("--analysis-graphs", type=int, default=8)
    parser.add_argument("--ablation-graphs", type=int, default=96)
    parser.add_argument("--family-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grit-dir", default=DEFAULT_GRIT_DIR)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--force-training", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
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
        "train_ns": parse_int_tuple(args.train_ns),
        "eval_ns": parse_int_tuple(args.eval_ns),
        "mechanistic_ns": parse_int_tuple(args.mechanistic_ns),
        "seeds": parse_int_tuple(args.seeds),
        "steps": args.steps,
        "min_steps": args.min_steps,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "score_graphs": args.score_graphs,
        "score_donors": args.score_donors,
        "analysis_batch_size": args.analysis_graphs,
        "ablation_graphs": args.ablation_graphs,
        "family_size": args.family_size,
        "device": args.device,
    }
    if args.fast_dev_run:
        values.update(
            {
                "models": ("1hop", "dense"),
                "widths": (32,),
                "analysis_width": 32,
                "heads": 4,
                "train_ns": (4, 8),
                "eval_ns": (4, 8),
                "mechanistic_ns": (4, 8),
                "seeds": (0,),
                "steps": 12,
                "min_steps": 12,
                "warmup_steps": 4,
                "eval_every": 4,
                "validation_graphs": 4,
                "heldout_graphs": 8,
                "batch_size": 2,
                "max_batch_nodes": 128,
                "max_dense_pairs": 20_000,
                "score_graphs": 2,
                "score_donors": 1,
                "analysis_batch_size": 2,
                "ablation_graphs": 4,
                "family_size": 1,
                "patience_checks": 99,
                "low_n_accuracy_gate": 0.0,
            }
        )
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
        for model_name in cfg.models:
            for seed in cfg.seeds:
                path = checkpoint_path(run_dir, cfg, model_name, width, seed)
                if not path.exists():
                    raise FileNotFoundError(f"missing checkpoint: {path}")
                payloads.append(torch.load(path, map_location="cpu", weights_only=False))
    return payloads


def load_analyses(cfg: Config, run_dir: Path) -> list[dict[str, Any]]:
    import torch

    analyses = []
    for model_name in cfg.models:
        for seed in cfg.seeds:
            path = analysis_path(run_dir, cfg, model_name, seed)
            if not path.exists():
                raise FileNotFoundError(f"missing analysis cache: {path}")
            cached = torch.load(path, map_location="cpu", weights_only=False)
            if cached.get("fingerprint") != config_fingerprint(cfg):
                raise RuntimeError(f"analysis fingerprint mismatch at {path}")
            analyses.append(cached)
    return analyses


def assert_parameter_matching(payloads: Sequence[Mapping[str, Any]], cfg: Config) -> None:
    table: list[dict[str, Any]] = []
    for width in cfg.widths:
        values = {
            int(payload["parameters"])
            for payload in payloads
            if int(payload["width"]) == int(width)
        }
        if len(values) != 1:
            raise RuntimeError(f"support variants are not parameter matched at width {width}: {values}")
        table.append({"width": width, "heads": cfg.heads, "layers": cfg.layers, "parameters": values.pop()})
    write_csv(Path(cfg.drive_root) / cfg.run_name / "tables" / "parameter_budget.csv", table)


def main(argv: Sequence[str] | None = None) -> dict[str, Any] | None:
    # A source file pasted directly into Colab has ``__name__ == '__main__'`` and inherits
    # ipykernel's private ``-f <connection.json>`` arguments.  Treat that case as the documented
    # default run; normal command-line execution still consumes sys.argv.
    parser_argv = list(argv) if argv is not None else None
    if parser_argv is None and ("google.colab" in sys.modules or "ipykernel" in sys.modules):
        parser_argv = []
    args = build_parser().parse_args(parser_argv)
    cfg = config_from_args(args)
    run_dir = Path(cfg.drive_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "experiment_config.json",
        {
            "version": EXPERIMENT_VERSION,
            "fingerprint": config_fingerprint(cfg),
            "config": asdict(cfg),
            "methodology_alignment": METHODOLOGY_ALIGNMENT,
            "intentional_synthetic_differences": METHODOLOGY_DIFFERENCES,
        },
    )
    if args.phase == "figures":
        payloads = load_payloads(cfg, run_dir)
        analyses = load_analyses(cfg, run_dir)
        assert_parameter_matching(payloads, cfg)
        return make_all_figures(payloads, analyses, cfg, run_dir)

    setup_official_grit(Path(args.grit_dir), install=not args.skip_install)
    device = resolve_device(cfg.device)
    write_json(
        run_dir / "environment.json",
        {
            "python": sys.version,
            "device": str(device),
            "official_grit_commit": OFFICIAL_GRIT_COMMIT,
            "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        },
    )

    if args.phase in ("all", "train"):
        payloads = []
        for width in cfg.widths:
            for model_name in cfg.models:
                for seed in cfg.seeds:
                    model, payload = train_model(
                        cfg,
                        model_name=model_name,
                        width=width,
                        seed=seed,
                        run_dir=run_dir,
                        device=device,
                        force=args.force_training,
                        load_only=False,
                    )
                    gate = checkpoint_gate(cfg, payload)
                    print(
                        f"[gate {model_name} d={width} seed={seed}] "
                        f"low-N minimum channel accuracy={gate:.3f}",
                        flush=True,
                    )
                    if gate < cfg.low_n_accuracy_gate and not (
                        args.allow_low_accuracy or args.fast_dev_run
                    ):
                        raise RuntimeError(
                            f"{model_name} width {width} seed {seed} failed the low-load gate "
                            f"({gate:.3f} < {cfg.low_n_accuracy_gate:.3f}); refusing causal analysis"
                        )
                    payloads.append(payload)
                    del model
        assert_parameter_matching(payloads, cfg)
        if args.phase == "train":
            print("[done] training caches are complete; run --phase analyze next", flush=True)
            return None

    if args.phase in ("all", "analyze"):
        for model_name in cfg.models:
            for seed in cfg.seeds:
                model, payload = train_model(
                    cfg,
                    model_name=model_name,
                    width=cfg.analysis_width,
                    seed=seed,
                    run_dir=run_dir,
                    device=device,
                    force=False,
                    load_only=True,
                )
                gate = checkpoint_gate(cfg, payload)
                if gate < cfg.low_n_accuracy_gate and not (
                    args.allow_low_accuracy or args.fast_dev_run
                ):
                    raise RuntimeError(
                        f"cached {model_name} seed {seed} fails the low-load accuracy gate"
                    )
                analyze_model(
                    model,
                    payload,
                    cfg,
                    model_name,
                    seed,
                    run_dir=run_dir,
                    device=device,
                    force=args.force_analysis,
                )
                del model
        if args.phase == "analyze":
            print("[done] analysis caches are complete; run --phase figures next", flush=True)
            return None

    payloads = load_payloads(cfg, run_dir)
    analyses = load_analyses(cfg, run_dir)
    assert_parameter_matching(payloads, cfg)
    summary = make_all_figures(payloads, analyses, cfg, run_dir)
    print(f"[done] all artifacts saved under {run_dir}", flush=True)
    return summary


if __name__ == "__main__":
    main()
