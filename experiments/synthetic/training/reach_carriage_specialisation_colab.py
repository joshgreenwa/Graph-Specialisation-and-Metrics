"""Standalone Colab: trained-mask reach, oversquashing, carriage, and head specialisation.

Paste this complete file into one Colab cell.  It mounts Drive, installs the pinned official
LiamMa/GRIT implementation, trains parameter-matched dense/1-hop/2-hop/3-hop models, caches all
checkpoints and analyses, and writes paper-facing PNG/PDF figures.

Scientific design
-----------------
One generator produces 24-node, degree-4 two-block graphs.  Degree-preserving edge switches give
either a thin cut (2 cross-cut edges) or a wide cut (8); node count, degree, edge count, features,
and local cluster construction are otherwise identical.  Queries retrieve the random value of a
colour-matched marked source.

* Reachability: rank-1 retrieval on the thin graph with source distance d=1,...,6.  With three
  layers, a k-hop mask has theoretical content reach 3k; dense attention has direct reach.
* Oversquashing: distance is fixed at d=3, reachable by every trained mask, while the number of
  simultaneous cross-cut query/source pairs is swept over 1,2,4,6 on thin versus wide cuts.

The analysis computes semantic donor-swap and mask-frozen structural-transposition carriage.
Functional carriage is the norm of the donor-averaged logit movement.  Beneficial carriage is the
exact finite loss improvement, L(corrupt)-L(clean), so positive values mean that the clean factor
helps the task.  Per-head intervention scores use official GRIT's routed ``wV`` transport site.
Every head is then ablated to measure task impact and the resulting loss of matching carriage;
top-score clean-head rescue, direct query-source edge lesions, and cross-cut lesions provide causal
controls.

Primary outputs
---------------
``fig1_reachability_accuracy``
``fig2_oversquashing_and_lesions``
``fig3_semantic_structural_carriage``
``fig4_specialisation_carriage_causality``
``fig5_dense_masked_similarity``

Use ``--fast-dev-run`` only for plumbing.  Paper claims require the default multi-seed run.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import functools
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPERIMENT_VERSION = "reach-carriage-specialisation-v1"
OFFICIAL_GRIT_URL = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
DEFAULT_GRIT_DIR = "/content/GRIT"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/graph_specialisation_metrics/reach_carriage_specialisation"
MODEL_RADII: dict[str, int | None] = {"1hop": 1, "2hop": 2, "3hop": 3, "dense": None}
MODEL_ORDER = ("1hop", "2hop", "3hop", "dense")
TOPOLOGY_SWITCHES = {"thin": 1, "wide": 4}
EPS = 1.0e-12


# ======================================================================================
# Colab/bootstrap and serialisation
# ======================================================================================


def _run(cmd: Sequence[str], *, check: bool = True) -> int:
    print(f"[cmd] {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run(list(map(str, cmd)), check=check).returncode


def mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
    except Exception as exc:  # pragma: no cover
        print(f"[drive] mount unavailable ({exc}); using the configured path", flush=True)


def install_pyg_stack() -> None:
    import torch

    torch_version = str(torch.__version__).split("+")[0]
    cuda = getattr(torch.version, "cuda", None)
    cuda_tag = ("cu" + cuda.replace(".", "")) if cuda else "cpu"
    wheel_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"
    print(f"[deps] torch={torch.__version__} cuda={cuda} | {wheel_url}", flush=True)
    _run([
        sys.executable, "-m", "pip", "install", "-q",
        "torch_geometric", "yacs", "ogb", "einops", "opt_einsum",
    ], check=False)
    for pkg in ("pyg-lib", "torch-spline-conv", "torch-cluster"):
        _run([sys.executable, "-m", "pip", "install", "-q", pkg, "-f", wheel_url], check=False)
    for pkg in ("torch-scatter", "torch-sparse"):
        rc = _run([sys.executable, "-m", "pip", "install", "-q", pkg, "-f", wheel_url], check=False)
        if rc:
            raise SystemExit(
                f"Required PyG extension {pkg!r} has no wheel at {wheel_url}. "
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


# ======================================================================================
# Configuration and controlled graph generator
# ======================================================================================


@dataclass(frozen=True)
class Config:
    run_name: str = "reach_carriage_v1"
    drive_root: str = DEFAULT_DRIVE_ROOT
    n: int = 24
    cluster_size: int = 12
    classes: int = 8
    pair_vocab: int = 6
    rrwp_steps: int = 10
    dim: int = 64
    heads: int = 4
    layers: int = 3
    dropout: float = 0.0
    attention_dropout: float = 0.05
    models: tuple[str, ...] = MODEL_ORDER
    distances: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    ranks: tuple[int, ...] = (1, 2, 4, 6)
    load_distance: int = 3
    batch_size: int = 64
    steps: int = 1500
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    eval_every: int = 100
    validation_graphs: int = 48
    heldout_graphs: int = 256
    patience_checks: int = 5
    accuracy_gate: float = 0.85
    carriage_graphs: int = 8
    carriage_donors: int = 3
    carriage_batch_size: int = 96
    score_graphs: int = 24
    score_donors: int = 4
    score_batch_size: int = 8
    ablation_graphs: int = 128
    focus_distances: tuple[int, ...] = (2, 5)
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "cuda"

    @property
    def feature_dim(self) -> int:
        # random value | pair colour | query marker | source marker
        return self.classes + self.pair_vocab + 2

    def validate(self) -> None:
        if self.n != 2 * self.cluster_size:
            raise ValueError("n must equal 2*cluster_size")
        if self.cluster_size < 10 or self.cluster_size % 2:
            raise ValueError("cluster_size must be an even integer >=10")
        if self.pair_vocab < max(self.ranks):
            raise ValueError("pair_vocab must cover the largest simultaneous rank")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if any(model not in MODEL_RADII for model in self.models):
            raise ValueError(f"models must be drawn from {sorted(MODEL_RADII)}")
        if self.load_distance > self.layers:
            raise ValueError("load_distance must be reachable by the 1-hop model")


def config_fingerprint(cfg: Config) -> str:
    payload = {"version": EXPERIMENT_VERSION, **asdict(cfg)}
    payload.pop("drive_root", None)
    payload.pop("run_name", None)
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class RetrievalBatch:
    x: Any
    adj: Any
    rrwp: Any
    qmask: Any
    y: Any
    source_for_query: Any
    blocked: Any
    condition: Any
    topology: Any
    target_distance: Any
    rank: Any

    def __len__(self) -> int:
        return int(self.x.size(0))

    def to(self, device: Any) -> "RetrievalBatch":
        return RetrievalBatch(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})

    def cpu(self) -> "RetrievalBatch":
        return self.to("cpu")

    def slice(self, start: int, stop: int) -> "RetrievalBatch":
        return RetrievalBatch(**{name: getattr(self, name)[start:stop] for name in self.__dataclass_fields__})

    def select(self, indices: Any) -> "RetrievalBatch":
        return RetrievalBatch(**{name: getattr(self, name)[indices] for name in self.__dataclass_fields__})


def concat_batches(batches: Sequence[RetrievalBatch]) -> RetrievalBatch:
    import torch

    return RetrievalBatch(**{
        name: torch.cat([getattr(batch, name) for batch in batches], dim=0)
        for name in RetrievalBatch.__dataclass_fields__
    })


def bfs_distances(adj: np.ndarray, source: int) -> np.ndarray:
    n = int(adj.shape[0])
    distances = np.full(n, -1, dtype=np.int64)
    distances[int(source)] = 0
    frontier = [int(source)]
    while frontier:
        nxt: list[int] = []
        for node in frontier:
            for neighbour in np.flatnonzero(adj[node]):
                if distances[neighbour] < 0:
                    distances[neighbour] = distances[node] + 1
                    nxt.append(int(neighbour))
        frontier = nxt
    return distances


def all_pairs_distances(adj: np.ndarray) -> np.ndarray:
    return np.stack([bfs_distances(adj, node) for node in range(adj.shape[0])])


def rrwp_from_adj(adj: np.ndarray, steps: int) -> np.ndarray:
    degree = adj.sum(axis=1, keepdims=True)
    transition = adj / np.maximum(degree, 1.0)
    n = int(adj.shape[0])
    output = np.zeros((n, n, steps), dtype=np.float32)
    power = np.eye(n, dtype=np.float32)
    for step in range(steps):
        if step:
            power = power @ transition
        output[:, :, step] = power
    return output


@functools.lru_cache(maxsize=16)
def base_two_block_graph(cluster_size: int, switches: int, rrwp_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Degree-4 two-block graph; each 2-switch preserves every node degree and edge count."""
    m = int(cluster_size)
    n = 2 * m
    adj = np.zeros((n, n), dtype=np.float32)
    for base in (0, m):
        for offset in (1, 2):
            for node in range(m):
                u, v = base + node, base + (node + offset) % m
                adj[u, v] = adj[v, u] = 1.0
    for switch in range(int(switches)):
        left_a, left_b = 2 * switch, 2 * switch + 1
        right_a, right_b = m + 2 * switch, m + 2 * switch + 1
        adj[left_a, left_b] = adj[left_b, left_a] = 0.0
        adj[right_a, right_b] = adj[right_b, right_a] = 0.0
        adj[left_a, right_a] = adj[right_a, left_a] = 1.0
        adj[left_b, right_b] = adj[right_b, left_b] = 1.0
    if not np.all(adj.sum(axis=1) == 4):
        raise RuntimeError("degree-preserving construction failed")
    return adj, rrwp_from_adj(adj, rrwp_steps), all_pairs_distances(adj)


def relabel_graph(cfg: Config, topology: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_adj, base_rrwp, base_dist = base_two_block_graph(
        cfg.cluster_size, TOPOLOGY_SWITCHES[topology], cfg.rrwp_steps
    )
    left = rng.permutation(cfg.cluster_size)
    right = cfg.cluster_size + rng.permutation(cfg.cluster_size)
    order = np.concatenate([left, right])
    return (
        base_adj[order][:, order].copy(),
        base_rrwp[order][:, order].copy(),
        base_dist[order][:, order].copy(),
    )


def sample_pairs(
    cfg: Config,
    distances: np.ndarray,
    *,
    rank: int,
    distance: int,
    cross_cut: bool,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if cross_cut:
        candidates = [
            (q, source)
            for q in range(cfg.cluster_size)
            for source in range(cfg.cluster_size, cfg.n)
            if int(distances[q, source]) == int(distance)
        ]
    else:
        candidates = [
            (q, source)
            for q in range(cfg.n)
            for source in range(cfg.n)
            if q != source and int(distances[q, source]) == int(distance)
        ]
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    used: set[int] = set()
    for q, source in candidates:
        if q in used or source in used:
            continue
        chosen.append((int(q), int(source)))
        used.update((int(q), int(source)))
        if len(chosen) == int(rank):
            return chosen
    raise RuntimeError(
        f"could not place rank={rank} disjoint pairs at distance={distance}, cross_cut={cross_cut}"
    )


def make_batch(
    cfg: Config,
    size: int,
    seed: int,
    *,
    condition: str | None = None,
    topology: str | None = None,
    distance: int | None = None,
    rank: int | None = None,
) -> RetrievalBatch:
    import torch

    rng = np.random.default_rng(int(seed))
    xs = np.zeros((size, cfg.n, cfg.feature_dim), dtype=np.float32)
    adjs = np.zeros((size, cfg.n, cfg.n), dtype=np.float32)
    rrwps = np.zeros((size, cfg.n, cfg.n, cfg.rrwp_steps), dtype=np.float32)
    qmasks = np.zeros((size, cfg.n), dtype=bool)
    labels = np.full((size, cfg.n), -1, dtype=np.int64)
    sources = np.full((size, cfg.n), -1, dtype=np.int64)
    blocked = np.zeros((size, cfg.n, cfg.n), dtype=bool)
    conditions = np.zeros(size, dtype=np.int64)
    topologies = np.zeros(size, dtype=np.int64)
    target_distances = np.zeros(size, dtype=np.int64)
    ranks = np.zeros(size, dtype=np.int64)
    pair_start = cfg.classes
    query_flag = cfg.classes + cfg.pair_vocab
    source_flag = query_flag + 1

    for graph in range(size):
        chosen_condition = condition or ("reach" if rng.random() < 0.5 else "load")
        if chosen_condition == "reach":
            chosen_topology = topology or "thin"
            chosen_distance = int(distance if distance is not None else rng.choice(cfg.distances))
            chosen_rank = int(rank if rank is not None else 1)
            cross_cut = False
            conditions[graph] = 0
        elif chosen_condition == "load":
            chosen_topology = topology or str(rng.choice(("thin", "wide")))
            chosen_distance = int(distance if distance is not None else cfg.load_distance)
            chosen_rank = int(rank if rank is not None else rng.choice(cfg.ranks))
            cross_cut = True
            conditions[graph] = 1
        else:
            raise ValueError(f"unknown condition {chosen_condition!r}")
        if chosen_topology not in TOPOLOGY_SWITCHES:
            raise ValueError(f"unknown topology {chosen_topology!r}")
        adj, rrwp, graph_distances = relabel_graph(cfg, chosen_topology, rng)
        pairs = sample_pairs(
            cfg,
            graph_distances,
            rank=chosen_rank,
            distance=chosen_distance,
            cross_cut=cross_cut,
            rng=rng,
        )
        values = rng.integers(0, cfg.classes, size=cfg.n)
        xs[graph, np.arange(cfg.n), values] = 1.0
        pair_colours = rng.permutation(cfg.pair_vocab)[:chosen_rank]
        for pair_index, (q, source) in enumerate(pairs):
            colour = int(pair_colours[pair_index])
            xs[graph, q, pair_start + colour] = 1.0
            xs[graph, source, pair_start + colour] = 1.0
            xs[graph, q, query_flag] = 1.0
            xs[graph, source, source_flag] = 1.0
            qmasks[graph, q] = True
            labels[graph, q] = int(values[source])
            sources[graph, q] = source
        adjs[graph] = adj
        rrwps[graph] = rrwp
        topologies[graph] = 0 if chosen_topology == "thin" else 1
        target_distances[graph] = chosen_distance
        ranks[graph] = chosen_rank

    return RetrievalBatch(
        x=torch.from_numpy(xs),
        adj=torch.from_numpy(adjs),
        rrwp=torch.from_numpy(rrwps),
        qmask=torch.from_numpy(qmasks),
        y=torch.from_numpy(labels),
        source_for_query=torch.from_numpy(sources),
        blocked=torch.from_numpy(blocked),
        condition=torch.from_numpy(conditions),
        topology=torch.from_numpy(topologies),
        target_distance=torch.from_numpy(target_distances),
        rank=torch.from_numpy(ranks),
    )


def khop_support(adj: Any, radius: int | None) -> Any:
    import torch

    batch, n, _ = adj.shape
    if radius is None:
        return torch.ones(batch, n, n, dtype=torch.bool, device=adj.device)
    identity = torch.eye(n, dtype=torch.bool, device=adj.device).expand(batch, -1, -1)
    reach = identity.clone()
    frontier = identity.float()
    adjacency = (adj > 0).float()
    for _ in range(int(radius)):
        frontier = torch.bmm(frontier, adjacency)
        frontier = (frontier > 0).float()
        reach |= frontier.bool()
    return reach


# ======================================================================================
# Official GRIT model: the trained mask is the only architectural difference
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


class OfficialMaskedGRIT:  # dynamically replaced after Colab bootstrap
    pass


def build_model_class() -> type:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from grit.layer.grit_layer import GritTransformerLayer

    class _OfficialMaskedGRIT(nn.Module):
        def __init__(self, cfg: Config, model_name: str) -> None:
            super().__init__()
            if model_name not in MODEL_RADII:
                raise ValueError(model_name)
            self.cfg = cfg
            self.model_name = model_name
            self.radius = MODEL_RADII[model_name]
            self.L, self.H = cfg.layers, cfg.heads
            self.dh = cfg.dim // cfg.heads
            self.input_encoder = nn.Linear(cfg.feature_dim, cfg.dim)
            self.node_rrwp_encoder = nn.Linear(cfg.rrwp_steps, cfg.dim, bias=False)
            self.pair_rrwp_encoder = nn.Linear(cfg.rrwp_steps, cfg.dim, bias=False)
            # Identical three-way edge vocabulary for every mask: self, graph edge, non-edge pair.
            self.edge_type_encoder = nn.Embedding(3, cfg.dim)
            nn.init.xavier_uniform_(self.node_rrwp_encoder.weight)
            nn.init.xavier_uniform_(self.pair_rrwp_encoder.weight)
            layer_cfg = grit_layer_cfg(update_e=True)
            self.layers = nn.ModuleList([
                GritTransformerLayer(
                    cfg.dim,
                    cfg.dim,
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
            self.output_head = nn.Linear(cfg.dim, cfg.classes)

        @property
        def attention_layers(self) -> list[Any]:
            return [layer.attention for layer in self.layers]

        def _pyg_batch(self, batch: RetrievalBatch) -> Any:
            bsz, n = int(batch.x.size(0)), int(batch.x.size(1))
            support = khop_support(batch.adj, self.radius) & ~batch.blocked.bool()
            graph_idx, src_local, dst_local = support.nonzero(as_tuple=True)
            src = graph_idx * n + src_local
            dst = graph_idx * n + dst_local
            is_self = src_local == dst_local
            is_graph_edge = batch.adj[graph_idx, src_local, dst_local] > 0
            edge_type = torch.where(is_self, 0, torch.where(is_graph_edge, 1, 2)).long()
            edge_rrwp = batch.rrwp[graph_idx, src_local, dst_local]
            diagonal = torch.arange(n, device=batch.x.device)
            node_rrwp = batch.rrwp[:, diagonal, diagonal].reshape(bsz * n, self.cfg.rrwp_steps)
            data = Data(num_nodes=bsz * n)
            data.x = self.input_encoder(batch.x.reshape(bsz * n, -1)) + self.node_rrwp_encoder(node_rrwp)
            data.edge_index = torch.stack([src, dst], dim=0)
            data.edge_attr = self.edge_type_encoder(edge_type) + self.pair_rrwp_encoder(edge_rrwp)
            data.batch = torch.arange(bsz, device=batch.x.device).repeat_interleave(n)
            degree = torch.zeros(bsz * n, device=batch.x.device)
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
            data.deg = degree
            data.log_deg = torch.log(degree + 1.0)
            data.graph_num_nodes = torch.full((bsz,), n, dtype=torch.long, device=batch.x.device)
            return data

        def node_states(self, batch: RetrievalBatch) -> Any:
            data = self._pyg_batch(batch)
            for layer in self.layers:
                data = layer(data)
            return data.x.reshape(len(batch), self.cfg.n, self.cfg.dim)

        def forward(self, batch: RetrievalBatch) -> Any:
            return self.output_head(self.node_states(batch))

    return _OfficialMaskedGRIT


@contextlib.contextmanager
def ablate_heads(model: Any, heads: Sequence[tuple[int, int]]):
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    handles = []

    def make_hook(indices: Sequence[int]):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            h_out, e_out = output
            changed = h_out.clone()
            changed[:, list(indices), :] = 0.0
            return changed, e_out

        return hook

    try:
        for layer, indices in by_layer.items():
            handles.append(model.attention_layers[layer].register_forward_hook(make_hook(indices)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def patch_head_output(model: Any, layer: int, head: int, clean_wv: Any):
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        h_out, e_out = output
        if tuple(h_out.shape) != tuple(clean_wv.shape):
            raise RuntimeError(f"patch alignment failed: {tuple(h_out.shape)} != {tuple(clean_wv.shape)}")
        changed = h_out.clone()
        changed[:, int(head), :] = clean_wv.to(changed.device, changed.dtype)[:, int(head), :]
        return changed, e_out

    handle = model.attention_layers[int(layer)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def capture_forward(model: Any, batch: RetrievalBatch, *, want_grad: bool, want_attention: bool = False) -> dict[str, Any]:
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


# ======================================================================================
# Training, evaluation, and checkpoint caches
# ======================================================================================


def set_seed(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def graph_balanced_loss(logits: Any, batch: RetrievalBatch, *, reduction: str = "mean") -> Any:
    import torch
    import torch.nn.functional as F

    losses = F.cross_entropy(logits[batch.qmask], batch.y[batch.qmask].long(), reduction="none")
    graph_ids = torch.arange(len(batch), device=logits.device).repeat_interleave(batch.qmask.sum(dim=1))
    totals = torch.zeros(len(batch), device=logits.device)
    counts = torch.zeros(len(batch), device=logits.device)
    totals.index_add_(0, graph_ids, losses)
    counts.index_add_(0, graph_ids, torch.ones_like(losses))
    per_graph = totals / counts.clamp_min(1.0)
    return per_graph.mean() if reduction == "mean" else per_graph


def batch_metrics(logits: Any, batch: RetrievalBatch) -> dict[str, float]:
    import torch.nn.functional as F

    selected = logits[batch.qmask]
    labels = batch.y[batch.qmask].long()
    return {
        "loss": float(F.cross_entropy(selected, labels).detach().cpu()),
        "accuracy": float((selected.argmax(-1) == labels).float().mean().detach().cpu()),
    }


def predict_in_chunks(
    model: Any,
    batch: RetrievalBatch,
    *,
    device: Any,
    chunk_size: int,
    ablations: Sequence[tuple[int, int]] | None = None,
) -> Any:
    import torch

    outputs = []
    context = ablate_heads(model, ablations) if ablations else contextlib.nullcontext()
    with context, torch.no_grad():
        for start in range(0, len(batch), int(chunk_size)):
            stop = min(start + int(chunk_size), len(batch))
            outputs.append(model(batch.slice(start, stop).to(device)).detach().cpu())
    return torch.cat(outputs, dim=0)


def evaluate_cell(
    model: Any,
    cfg: Config,
    *,
    seed: int,
    device: Any,
    condition: str,
    topology: str,
    distance: int,
    rank: int,
    graphs: int,
    lesion: str = "clean",
) -> dict[str, Any]:
    batch = make_batch(
        cfg,
        graphs,
        seed,
        condition=condition,
        topology=topology,
        distance=distance,
        rank=rank,
    )
    if lesion != "clean":
        blocked = batch.blocked.clone()
        for graph in range(len(batch)):
            queries = batch.qmask[graph].nonzero(as_tuple=False).flatten()
            if lesion == "direct":
                for q in queries:
                    source = int(batch.source_for_query[graph, q])
                    blocked[graph, source, int(q)] = True
            elif lesion == "crosscut":
                m = cfg.cluster_size
                blocked[graph, :m, m:] = True
                blocked[graph, m:, :m] = True
            else:
                raise ValueError(lesion)
        batch.blocked = blocked
    logits = predict_in_chunks(model, batch, device=device, chunk_size=cfg.carriage_batch_size)
    return {
        "condition": condition,
        "topology": topology,
        "distance": int(distance),
        "rank": int(rank),
        "lesion": lesion,
        **batch_metrics(logits, batch),
    }


def evaluate_grid(model: Any, cfg: Config, *, seed: int, device: Any, graphs: int) -> list[dict[str, Any]]:
    rows = []
    for distance in cfg.distances:
        rows.append(evaluate_cell(
            model,
            cfg,
            seed=seed + 1000 * distance,
            device=device,
            condition="reach",
            topology="thin",
            distance=distance,
            rank=1,
            graphs=graphs,
        ))
    for topology_index, topology in enumerate(("thin", "wide")):
        for rank in cfg.ranks:
            rows.append(evaluate_cell(
                model,
                cfg,
                seed=seed + 100_000 + topology_index * 10_000 + rank * 101,
                device=device,
                condition="load",
                topology=topology,
                distance=cfg.load_distance,
                rank=rank,
                graphs=graphs,
            ))
    return rows


def theoretical_reach(cfg: Config, model_name: str) -> int:
    radius = MODEL_RADII[model_name]
    return max(cfg.distances) if radius is None else cfg.layers * int(radius)


def checkpoint_path(run_dir: Path, cfg: Config, model_name: str, seed: int) -> Path:
    return run_dir / "checkpoints" / f"{model_name}__seed_{seed}__{config_fingerprint(cfg)}.pt"


def train_model(
    cfg: Config,
    *,
    model_name: str,
    seed: int,
    run_dir: Path,
    device: Any,
    force: bool,
    load_only: bool,
) -> tuple[Any, dict[str, Any]]:
    import torch

    global OfficialMaskedGRIT
    if OfficialMaskedGRIT.__name__ == "OfficialMaskedGRIT":
        OfficialMaskedGRIT = build_model_class()
    path = checkpoint_path(run_dir, cfg, model_name, seed)
    set_seed(seed)
    model = OfficialMaskedGRIT(cfg, model_name).to(device)
    if path.exists() and not force:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("fingerprint") != config_fingerprint(cfg):
            raise RuntimeError(f"checkpoint fingerprint mismatch at {path}")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        print(f"[train {model_name} seed={seed}] loaded {path}", flush=True)
        return model, payload
    if load_only:
        raise FileNotFoundError(f"analysis requested but checkpoint is missing: {path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_objective = float("inf")
    best_state = None
    best_grid: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    good_checks = 0
    started = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        batch = make_batch(cfg, cfg.batch_size, seed * 1_000_003 + step).to(device)
        logits = model(batch)
        loss = graph_balanced_loss(logits, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 1 or step % cfg.eval_every == 0 or step == cfg.steps:
            model.eval()
            grid = evaluate_grid(
                model,
                cfg,
                seed=50_000 + seed,
                device=device,
                graphs=cfg.validation_graphs,
            )
            reachable = [
                row for row in grid
                if (row["condition"] == "reach" and row["distance"] <= theoretical_reach(cfg, model_name))
                or (row["condition"] == "load" and row["rank"] == 1)
            ]
            objective = float(np.mean([row["loss"] for row in reachable]))
            mean_accuracy = float(np.mean([row["accuracy"] for row in reachable]))
            history.append({
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "reachable_validation_loss": objective,
                "reachable_validation_accuracy": mean_accuracy,
                "elapsed_s": time.time() - started,
            })
            print(
                f"[train {model_name} seed={seed}] {step:4d}/{cfg.steps} "
                f"loss={float(loss.detach().cpu()):.4f} reachable_acc={mean_accuracy:.3f}",
                flush=True,
            )
            if objective < best_objective:
                best_objective = objective
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_grid = grid
            good_checks = good_checks + 1 if mean_accuracy >= 0.985 else 0
            model.train()
            if good_checks >= cfg.patience_checks:
                print(f"[train {model_name} seed={seed}] early stop", flush=True)
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint candidate")
    model.load_state_dict(best_state)
    model.eval()
    heldout = evaluate_grid(
        model,
        cfg,
        seed=800_000 + seed,
        device=device,
        graphs=cfg.heldout_graphs,
    )
    payload = {
        "version": EXPERIMENT_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "official_grit_commit": OFFICIAL_GRIT_COMMIT,
        "model_name": model_name,
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
    write_csv(run_dir / "tables" / f"training_{model_name}_seed_{seed}.csv", history)
    print(f"[train {model_name} seed={seed}] cached {path}", flush=True)
    return model, payload


def checkpoint_gate(cfg: Config, model_name: str, payload: Mapping[str, Any]) -> float:
    reachable = [
        row for row in payload.get("heldout", [])
        if (row["condition"] == "reach" and row["distance"] <= theoretical_reach(cfg, model_name))
        or (row["condition"] == "load" and row["rank"] == 1)
    ]
    return float(np.mean([row["accuracy"] for row in reachable])) if reachable else 0.0

