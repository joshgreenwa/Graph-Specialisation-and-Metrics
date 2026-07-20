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

The analysis computes semantic donor-swap carriage and full structural carriage, for which RRWP
and sparse support are transposed together. Functional carriage is the norm of the donor-averaged
logit movement. Beneficial carriage is the exact finite loss improvement,
L(corrupt)-L(clean), so positive values mean that the clean factor helps the task. Per-head
intervention scores use official GRIT's routed ``wV`` transport site; their structural intervention
freezes support, as required by the specialisation methodology. Every head is then ablated to
measure task impact and the resulting loss of full matching carriage; top-score clean-head rescue,
direct query-source edge lesions, and cross-cut lesions provide causal controls.

The node-retrieval head reads only the query state, so the query is the sole carrier: its linear
readout makes the functional logit movement equal to aggregate functional carriage, and the exact
finite cross-entropy change equals aggregate beneficial carriage without a multi-carrier allocation.

Primary outputs
---------------
``fig1_reachability_accuracy``
``fig2_oversquashing_and_lesions``
``fig3_semantic_structural_carriage``
``fig4_specialisation_carriage_causality``
``fig5_dense_masked_similarity``

Use ``--fast-dev-run`` only for plumbing. Paper claims require the default multi-seed run. A
failed convergence gate receives at most two predetermined alternate initialisations, selected on
validation accuracy only; the task/data seed is unchanged and the accepted attempt is recorded.
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
        if "dense" not in self.models or len(self.models) < 2:
            raise ValueError("the comparison and similarity figures require dense plus at least one masked model")
        if self.load_distance > self.layers:
            raise ValueError("load_distance must be reachable by the 1-hop model")
        if any(distance not in self.distances for distance in self.focus_distances):
            raise ValueError("focus_distances must be included in distances")


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
    # A greedy matching occasionally gets stuck at the largest load even when a
    # valid matching exists.  Rank is deliberately tiny (<= 6), so an exact,
    # randomised backtracking matcher is both cheap and deterministic by seed.
    by_query: dict[int, list[int]] = {}
    rng.shuffle(candidates)
    for q, source in candidates:
        by_query.setdefault(int(q), []).append(int(source))
    query_order = list(by_query)
    rng.shuffle(query_order)
    query_order.sort(key=lambda q: len(by_query[q]))

    def find_matching(
        position: int,
        used: frozenset[int],
        chosen: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...] | None:
        if len(chosen) == int(rank):
            return chosen
        needed = int(rank) - len(chosen)
        available_queries = sum(q not in used for q in query_order[position:])
        if available_queries < needed:
            return None
        for index in range(position, len(query_order)):
            q = query_order[index]
            if q in used:
                continue
            for source in by_query[q]:
                if source in used:
                    continue
                result = find_matching(
                    index + 1,
                    used | frozenset((q, source)),
                    chosen + ((q, source),),
                )
                if result is not None:
                    return result
        return None

    matching = find_matching(0, frozenset(), ())
    if matching is not None:
        return list(matching)
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
            self.last_edge_index = data.edge_index
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
    initialisation_attempt: int = 0,
) -> tuple[Any, dict[str, Any]]:
    import torch

    global OfficialMaskedGRIT
    if OfficialMaskedGRIT.__name__ == "OfficialMaskedGRIT":
        OfficialMaskedGRIT = build_model_class()
    path = checkpoint_path(run_dir, cfg, model_name, seed)
    initialisation_seed = int(seed) + int(initialisation_attempt) * 100_003
    set_seed(initialisation_seed)
    model = OfficialMaskedGRIT(cfg, model_name).to(device)
    if path.exists() and not force:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("fingerprint") != config_fingerprint(cfg):
            raise RuntimeError(f"checkpoint fingerprint mismatch at {path}")
        model.load_state_dict(payload["state_dict"])
        model.initialisation_attempt = int(payload.get("initialisation_attempt", 0))
        model.initialisation_seed = int(payload.get("initialisation_seed", seed))
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
        "initialisation_attempt": int(initialisation_attempt),
        "initialisation_seed": int(initialisation_seed),
        "config": asdict(cfg),
        "state_dict": best_state,
        "best_validation": best_grid,
        "heldout": heldout,
        "history": history,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    write_csv(
        run_dir / "tables" / f"training_{model_name}_seed_{seed}_attempt_{initialisation_attempt}.csv",
        history,
    )
    model.initialisation_attempt = int(initialisation_attempt)
    model.initialisation_seed = int(initialisation_seed)
    print(f"[train {model_name} seed={seed}] cached {path}", flush=True)
    return model, payload


def checkpoint_gate(
    cfg: Config,
    model_name: str,
    payload: Mapping[str, Any],
    *,
    split: str = "heldout",
) -> float:
    if split not in {"best_validation", "heldout"}:
        raise ValueError(split)
    reachable = [
        row for row in payload.get(split, [])
        if (row["condition"] == "reach" and row["distance"] <= theoretical_reach(cfg, model_name))
        or (row["condition"] == "load" and row["rank"] == 1)
    ]
    return float(np.mean([row["accuracy"] for row in reachable])) if reachable else 0.0


# ======================================================================================
# Semantic/structural interventions and aggregate carriage
# ======================================================================================


def rank1_query_source(batch: RetrievalBatch, graph: int) -> tuple[int, int]:
    queries = batch.qmask[int(graph)].nonzero(as_tuple=False).flatten()
    if len(queries) != 1:
        raise ValueError("this analysis requires rank-1 graphs")
    query = int(queries[0])
    return query, int(batch.source_for_query[int(graph), query])


def replace_value(cfg: Config, row: Any, value: int) -> Any:
    output = row.clone()
    output[: cfg.classes] = 0.0
    output[int(value)] = 1.0
    return output


def draw_other_class(rng: np.random.Generator, classes: int, current: int) -> int:
    value = int(rng.integers(0, classes - 1))
    return value + (1 if value >= int(current) else 0)


def transpose_node_axes(tensor: Any, u: int, v: int) -> Any:
    if int(u) == int(v):
        return tensor.clone()
    order = list(range(int(tensor.size(0))))
    order[int(u)], order[int(v)] = order[int(v)], order[int(u)]
    return tensor[order][:, order].clone()


def structural_partners(adj: Any, source: int) -> list[int]:
    adj_np = np.asarray(adj.detach().cpu(), dtype=np.float32)
    degree = adj_np.sum(axis=1)
    return [
        node for node in range(adj_np.shape[0])
        if node != int(source)
        and degree[node] == degree[int(source)]
    ]


def _replica(
    batch: RetrievalBatch,
    graph: int,
    *,
    x: Any | None = None,
    adj: Any | None = None,
    rrwp: Any | None = None,
    blocked: Any | None = None,
) -> RetrievalBatch:
    one = batch.slice(graph, graph + 1)
    if x is not None:
        one.x = x
    if adj is not None:
        one.adj = adj
    if rrwp is not None:
        one.rrwp = rrwp
    if blocked is not None:
        one.blocked = blocked
    return one


def make_all_source_replicas(
    cfg: Config,
    clean: RetrievalBatch,
    *,
    factor: str,
    donors: int,
    seed: int,
    no_op: bool = False,
    structural_support: str = "conjugated",
) -> RetrievalBatch:
    """All-source carriage replicas; structural carriage conjugates support by default."""
    import torch

    rng = np.random.default_rng(int(seed))
    replicas: list[RetrievalBatch] = []
    for graph in range(len(clean)):
        one = clean.slice(graph, graph + 1)
        replicas.append(one)
        rank1_query_source(clean, graph)
        for source in range(cfg.n):
            for _ in range(int(donors)):
                x = one.x.clone()
                adj = one.adj.clone()
                rrwp = one.rrwp.clone()
                blocked = one.blocked.clone()
                if factor == "semantic":
                    current = int(torch.argmax(x[0, source, : cfg.classes]))
                    value = current if no_op else draw_other_class(rng, cfg.classes, current)
                    x[0, source] = replace_value(cfg, x[0, source], value)
                elif factor == "structural":
                    if no_op:
                        partner = source
                    else:
                        candidates = structural_partners(one.adj[0], source)
                        partner = int(rng.choice(candidates))
                    rrwp[0] = transpose_node_axes(rrwp[0], source, partner)
                    if structural_support == "conjugated":
                        adj[0] = transpose_node_axes(adj[0], source, partner)
                        blocked[0] = transpose_node_axes(blocked[0], source, partner)
                    elif structural_support != "frozen":
                        raise ValueError(structural_support)
                else:
                    raise ValueError(factor)
                replicas.append(_replica(one, 0, x=x, adj=adj, rrwp=rrwp, blocked=blocked))
    return concat_batches(replicas)


def make_target_replicas(
    cfg: Config,
    clean: RetrievalBatch,
    *,
    factor: str,
    donors: int,
    seed: int,
    structural_support: str = "frozen",
) -> RetrievalBatch:
    """Graph-major target corruptions; head scores default to mask-frozen structure."""
    import torch

    rng = np.random.default_rng(int(seed))
    replicas: list[RetrievalBatch] = []
    for graph in range(len(clean)):
        one = clean.slice(graph, graph + 1)
        replicas.append(one)
        _, source = rank1_query_source(clean, graph)
        for _ in range(int(donors)):
            x = one.x.clone()
            adj = one.adj.clone()
            rrwp = one.rrwp.clone()
            blocked = one.blocked.clone()
            if factor == "semantic":
                current = int(torch.argmax(x[0, source, : cfg.classes]))
                x[0, source] = replace_value(
                    cfg, x[0, source], draw_other_class(rng, cfg.classes, current)
                )
            elif factor == "structural":
                partner = int(rng.choice(structural_partners(one.adj[0], source)))
                rrwp[0] = transpose_node_axes(rrwp[0], source, partner)
                if structural_support == "conjugated":
                    adj[0] = transpose_node_axes(adj[0], source, partner)
                    blocked[0] = transpose_node_axes(blocked[0], source, partner)
                elif structural_support != "frozen":
                    raise ValueError(structural_support)
            else:
                raise ValueError(factor)
            replicas.append(_replica(one, 0, x=x, adj=adj, rrwp=rrwp, blocked=blocked))
    return concat_batches(replicas)


def rank1_logits(node_logits: Any, batch: RetrievalBatch) -> Any:
    import torch

    rows = torch.arange(len(batch), device=node_logits.device)
    queries = batch.qmask.to(node_logits.device).float().argmax(dim=1).long()
    return node_logits[rows, queries]


def rank1_labels(batch: RetrievalBatch) -> Any:
    queries = batch.qmask.float().argmax(dim=1).long()
    rows = np.arange(len(batch))
    return batch.y[rows, queries]


def carriage_from_replica_logits(
    query_logits: Any,
    labels: Any,
    *,
    graphs: int,
    sources: int,
    donors: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    reps = 1 + int(sources) * int(donors)
    shaped = query_logits.reshape(graphs, reps, -1)
    clean = shaped[:, 0]
    corrupt = shaped[:, 1:].reshape(graphs, sources, donors, -1)
    mean_corrupt = corrupt.mean(dim=2)
    functional = torch.linalg.vector_norm(clean[:, None, :] - mean_corrupt, dim=-1)
    clean_loss = F.cross_entropy(clean, labels.long(), reduction="none")
    corrupt_labels = labels[:, None, None].expand(-1, sources, donors).reshape(-1)
    corrupt_loss = F.cross_entropy(
        corrupt.reshape(graphs * sources * donors, -1),
        corrupt_labels,
        reduction="none",
    ).reshape(graphs, sources, donors)
    # Production sign B = L(clean)-E L(corrupt): negative means the clean factor was beneficial.
    beneficial_signed = clean_loss[:, None] - corrupt_loss.mean(dim=2)
    return {
        "functional": functional,
        "beneficial_signed": beneficial_signed,
        "benefit": -beneficial_signed,
        "clean_logits": clean,
        "mean_corrupt_logits": mean_corrupt,
    }


def predict_rank1_replicas(
    model: Any,
    replicas: RetrievalBatch,
    *,
    device: Any,
    chunk_size: int,
    ablations: Sequence[tuple[int, int]] | None = None,
) -> Any:
    node_logits = predict_in_chunks(
        model,
        replicas,
        device=device,
        chunk_size=chunk_size,
        ablations=ablations,
    )
    return rank1_logits(node_logits, replicas)


def carriage_profile(
    model: Any,
    cfg: Config,
    *,
    distance: int,
    factor: str,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    clean = make_batch(
        cfg,
        cfg.carriage_graphs,
        seed,
        condition="reach",
        topology="thin",
        distance=distance,
        rank=1,
    )
    replicas = make_all_source_replicas(
        cfg,
        clean,
        factor=factor,
        donors=cfg.carriage_donors,
        seed=seed + 1,
    )
    logits = predict_rank1_replicas(
        model,
        replicas,
        device=device,
        chunk_size=cfg.carriage_batch_size,
    )
    labels = rank1_labels(clean)
    carriage = carriage_from_replica_logits(
        logits,
        labels,
        graphs=len(clean),
        sources=cfg.n,
        donors=cfg.carriage_donors,
    )
    distances = np.zeros((len(clean), cfg.n), dtype=np.int64)
    is_target = np.zeros((len(clean), cfg.n), dtype=bool)
    for graph in range(len(clean)):
        query, source = rank1_query_source(clean, graph)
        distances[graph] = bfs_distances(np.asarray(clean.adj[graph]), query)
        is_target[graph, source] = True
    return {
        "distance": int(distance),
        "factor": factor,
        "functional": carriage["functional"].cpu(),
        "beneficial_signed": carriage["beneficial_signed"].cpu(),
        "benefit": carriage["benefit"].cpu(),
        "source_distance": distances,
        "is_target": is_target,
    }


def no_op_carriage_check(model: Any, cfg: Config, *, factor: str, seed: int, device: Any) -> float:
    clean = make_batch(cfg, 2, seed, condition="reach", topology="thin", distance=2, rank=1)
    replicas = make_all_source_replicas(cfg, clean, factor=factor, donors=1, seed=seed + 1, no_op=True)
    logits = predict_rank1_replicas(model, replicas, device=device, chunk_size=cfg.carriage_batch_size)
    carriage = carriage_from_replica_logits(
        logits,
        rank1_labels(clean),
        graphs=len(clean),
        sources=cfg.n,
        donors=1,
    )
    return float(carriage["functional"].abs().max())


def relabel_invariance_check(model: Any, cfg: Config, *, seed: int, device: Any) -> float:
    import torch

    clean = make_batch(cfg, 3, seed, condition="reach", topology="thin", distance=2, rank=1)
    rng = np.random.default_rng(seed + 1)
    permuted = []
    for graph in range(len(clean)):
        one = clean.slice(graph, graph + 1)
        order = rng.permutation(cfg.n)
        inverse = np.empty(cfg.n, dtype=np.int64)
        inverse[order] = np.arange(cfg.n)
        source_map = one.source_for_query[:, order].clone()
        active = source_map >= 0
        if bool(active.any()):
            source_map[active] = torch.as_tensor(inverse[source_map[active].numpy()], dtype=torch.long)
        permuted.append(RetrievalBatch(
            x=one.x[:, order],
            adj=one.adj[:, order][:, :, order],
            rrwp=one.rrwp[:, order][:, :, order],
            qmask=one.qmask[:, order],
            y=one.y[:, order],
            source_for_query=source_map,
            blocked=one.blocked[:, order][:, :, order],
            condition=one.condition.clone(),
            topology=one.topology.clone(),
            target_distance=one.target_distance.clone(),
            rank=one.rank.clone(),
        ))
    relabelled = concat_batches(permuted)
    with torch.no_grad():
        clean_logits = rank1_logits(model(clean.to(device)), clean.to(device))
        relabelled_logits = rank1_logits(model(relabelled.to(device)), relabelled.to(device))
    return float((clean_logits - relabelled_logits).abs().max().cpu())


def softmax_check(model: Any, cfg: Config, *, seed: int, device: Any) -> float:
    import torch

    batch = make_batch(cfg, 2, seed, condition="load", topology="thin", distance=cfg.load_distance, rank=2).to(device)
    result = capture_forward(model, batch, want_grad=False, want_attention=True)
    edge_index = model.last_edge_index
    errors = []
    for attention in result["attention"]:
        sums = torch.zeros(len(batch) * cfg.n, cfg.heads, device=device)
        sums.index_add_(0, edge_index[1], attention)
        errors.append(float((sums - 1.0).abs().max().cpu()))
    return max(errors)


# ======================================================================================
# Head-resolved scores, carriage loss under ablation, task ablation, and rescue
# ======================================================================================


def score_factor_batch(
    model: Any,
    cfg: Config,
    clean: RetrievalBatch,
    *,
    factor: str,
    seed: int,
    device: Any,
) -> Any:
    import torch

    replicas = make_target_replicas(
        cfg,
        clean.cpu(),
        factor=factor,
        donors=cfg.score_donors,
        seed=seed,
    ).to(device)
    result = capture_forward(model, replicas, want_grad=True)
    query_logits = rank1_logits(result["logits"], replicas)
    graphs = len(clean)
    reps = cfg.score_donors + 1
    clean_rows = torch.arange(graphs, device=device) * reps
    accum = [torch.zeros(graphs, cfg.n, cfg.heads, device=device) for _ in range(cfg.layers)]
    for output_index in range(cfg.classes):
        gradients = torch.autograd.grad(
            query_logits[clean_rows, output_index].sum(),
            result["wV"],
            retain_graph=output_index < cfg.classes - 1,
            allow_unused=False,
        )
        for layer in range(cfg.layers):
            states = result["wV"][layer].reshape(graphs, reps, cfg.n, cfg.heads, -1)
            grads = gradients[layer].reshape(graphs, reps, cfg.n, cfg.heads, -1)[:, 0]
            delta = states[:, 0] - states[:, 1:].mean(dim=1)
            accum[layer] += (grads * delta).sum(dim=-1).square()
    return torch.stack([value.sqrt().sum(dim=1) for value in accum], dim=1).detach().cpu()


def score_factor(
    model: Any,
    cfg: Config,
    clean: RetrievalBatch,
    *,
    factor: str,
    seed: int,
    device: Any,
) -> Any:
    import torch

    pieces = []
    for start in range(0, len(clean), cfg.score_batch_size):
        stop = min(start + cfg.score_batch_size, len(clean))
        pieces.append(score_factor_batch(
            model,
            cfg,
            clean.slice(start, stop),
            factor=factor,
            seed=seed + start * 97,
            device=device,
        ))
    return torch.cat(pieces, dim=0)


def target_carriage_with_head_ablations(
    model: Any,
    cfg: Config,
    clean: RetrievalBatch,
    *,
    factor: str,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    replicas = make_target_replicas(
        cfg,
        clean,
        factor=factor,
        donors=cfg.score_donors,
        seed=seed,
        structural_support="conjugated",
    )
    labels = rank1_labels(clean)
    baseline_logits = predict_rank1_replicas(
        model, replicas, device=device, chunk_size=cfg.carriage_batch_size
    )
    baseline = carriage_from_replica_logits(
        baseline_logits,
        labels,
        graphs=len(clean),
        sources=1,
        donors=cfg.score_donors,
    )
    base_functional = baseline["functional"][:, 0]
    base_benefit = baseline["benefit"][:, 0]
    functional_drop = torch.zeros(cfg.layers, cfg.heads, len(clean))
    benefit_drop = torch.zeros_like(functional_drop)
    for layer in range(cfg.layers):
        for head in range(cfg.heads):
            ablated_logits = predict_rank1_replicas(
                model,
                replicas,
                device=device,
                chunk_size=cfg.carriage_batch_size,
                ablations=[(layer, head)],
            )
            ablated = carriage_from_replica_logits(
                ablated_logits,
                labels,
                graphs=len(clean),
                sources=1,
                donors=cfg.score_donors,
            )
            functional_drop[layer, head] = base_functional - ablated["functional"][:, 0]
            benefit_drop[layer, head] = base_benefit - ablated["benefit"][:, 0]
    return {
        "base_functional": base_functional,
        "base_benefit": base_benefit,
        "functional_drop": functional_drop,
        "benefit_drop": benefit_drop,
    }


def task_ablation_sweep(
    model: Any,
    cfg: Config,
    *,
    distance: int,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    clean = make_batch(
        cfg,
        cfg.ablation_graphs,
        seed,
        condition="reach",
        topology="thin",
        distance=distance,
        rank=1,
    )
    node_logits = predict_in_chunks(model, clean, device=device, chunk_size=cfg.carriage_batch_size)
    logits = rank1_logits(node_logits, clean)
    labels = rank1_labels(clean)
    clean_loss = F.cross_entropy(logits, labels.long(), reduction="none")
    functional = torch.zeros(cfg.layers, cfg.heads, len(clean))
    loss = torch.zeros_like(functional)
    for layer in range(cfg.layers):
        for head in range(cfg.heads):
            ablated_nodes = predict_in_chunks(
                model,
                clean,
                device=device,
                chunk_size=cfg.carriage_batch_size,
                ablations=[(layer, head)],
            )
            ablated = rank1_logits(ablated_nodes, clean)
            functional[layer, head] = torch.linalg.vector_norm(ablated - logits, dim=-1)
            loss[layer, head] = F.cross_entropy(ablated, labels.long(), reduction="none") - clean_loss
    return {
        "functional": functional,
        "loss": loss,
        "clean_accuracy": float((logits.argmax(-1) == labels).float().mean()),
    }


def _paired_clean_corrupt(replicas: RetrievalBatch, donors: int) -> tuple[RetrievalBatch, RetrievalBatch]:
    clean_indices = np.arange(0, len(replicas), donors + 1, dtype=np.int64)
    corrupt_indices = clean_indices + 1
    return replicas.select(clean_indices), replicas.select(corrupt_indices)


def rescue_controls(
    model: Any,
    cfg: Config,
    clean: RetrievalBatch,
    *,
    factor: str,
    score: np.ndarray,
    joint_score: np.ndarray,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    import torch

    replicas = make_target_replicas(
        cfg, clean, factor=factor, donors=cfg.score_donors, seed=seed
    )
    clean_batch, corrupt_batch = _paired_clean_corrupt(replicas, cfg.score_donors)
    clean_device = clean_batch.to(device)
    corrupt_device = corrupt_batch.to(device)
    clean_result = capture_forward(model, clean_device, want_grad=False)
    with torch.no_grad():
        corrupt_nodes = model(corrupt_device).detach()
    clean_logits = rank1_logits(clean_result["logits"], clean_device)
    corrupt_logits = rank1_logits(corrupt_nodes, corrupt_device)
    total = clean_logits - corrupt_logits
    denominator = total.square().sum(dim=-1) + EPS
    top_flat = int(np.nanargmax(score))
    top = (top_flat // cfg.heads, top_flat % cfg.heads)
    same_layer = [(top[0], head) for head in range(cfg.heads) if head != top[1]]
    layer_median = float(np.median([score[item] for item in same_layer]))
    low_factor = [item for item in same_layer if float(score[item]) <= layer_median]
    candidates = low_factor or same_layer
    # Match the selected head's aggregate semantic+structural throughput as closely as the small
    # head set permits, while requiring below-median matching-factor score within the same layer.
    control = min(
        candidates,
        key=lambda item: abs(math.log(float(joint_score[item]) + EPS) - math.log(float(joint_score[top]) + EPS)),
    )
    output: dict[str, Any] = {"top_head": list(top), "control_head": list(control)}
    for name, (layer, head) in (("top", top), ("layer_matched_control", control)):
        clean_wv = clean_result["wV"][layer].detach()
        with patch_head_output(model, layer, head, clean_wv), torch.no_grad():
            patched_nodes = model(corrupt_device).detach()
        patched = rank1_logits(patched_nodes, corrupt_device)
        mem = ((patched - corrupt_logits) * total).sum(dim=-1) / denominator
        output[name] = mem.detach().cpu()
    return output


def head_analysis_condition(
    model: Any,
    cfg: Config,
    *,
    distance: int,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    clean = make_batch(
        cfg,
        cfg.score_graphs,
        seed,
        condition="reach",
        topology="thin",
        distance=distance,
        rank=1,
    )
    output: dict[str, Any] = {"distance": int(distance)}
    factor_seeds: dict[str, int] = {}
    for factor_index, factor in enumerate(("semantic", "structural")):
        factor_seed = seed + 10_000 * (factor_index + 1)
        factor_seeds[factor] = factor_seed
        score_per_graph = score_factor(
            model,
            cfg,
            clean,
            factor=factor,
            seed=factor_seed,
            device=device,
        )
        carriage_delta = target_carriage_with_head_ablations(
            model,
            cfg,
            clean,
            factor=factor,
            seed=factor_seed,
            device=device,
        )
        score_mean = score_per_graph.mean(dim=0).numpy()
        output[factor] = {
            "score_per_graph": score_per_graph,
            "score": score_mean,
            "carriage": carriage_delta,
        }
    joint_score = np.asarray(output["semantic"]["score"]) + np.asarray(output["structural"]["score"])
    for factor in ("semantic", "structural"):
        output[factor]["rescue"] = rescue_controls(
            model,
            cfg,
            clean,
            factor=factor,
            score=np.asarray(output[factor]["score"]),
            joint_score=joint_score,
            seed=factor_seeds[factor],
            device=device,
        )
    output["task_ablation"] = task_ablation_sweep(
        model,
        cfg,
        distance=distance,
        seed=seed + 90_000,
        device=device,
    )
    return output


# ======================================================================================
# Cached analysis orchestration
# ======================================================================================


def analysis_path(run_dir: Path, cfg: Config, model_name: str, seed: int) -> Path:
    return run_dir / "analysis" / f"{model_name}__seed_{seed}__{config_fingerprint(cfg)}.pt"


def analyze_model(
    model: Any,
    cfg: Config,
    *,
    model_name: str,
    seed: int,
    run_dir: Path,
    device: Any,
    force: bool,
) -> dict[str, Any]:
    import torch

    path = analysis_path(run_dir, cfg, model_name, seed)
    if path.exists() and not force:
        print(f"[analysis {model_name} seed={seed}] loaded {path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    checks = {
        "semantic_no_op_max": no_op_carriage_check(
            model, cfg, factor="semantic", seed=1_100_000 + seed, device=device
        ),
        "structural_no_op_max": no_op_carriage_check(
            model, cfg, factor="structural", seed=1_200_000 + seed, device=device
        ),
        "full_relabel_max": relabel_invariance_check(
            model, cfg, seed=1_300_000 + seed, device=device
        ),
        "softmax_max_error": softmax_check(
            model, cfg, seed=1_400_000 + seed, device=device
        ),
    }
    if max(checks["semantic_no_op_max"], checks["structural_no_op_max"]) > 5.0e-5:
        raise RuntimeError(f"no-op verification failed: {checks}")
    if max(checks["full_relabel_max"], checks["softmax_max_error"]) > 5.0e-4:
        raise RuntimeError(f"GRIT invariance/normalisation verification failed: {checks}")
    profiles = []
    for distance in cfg.distances:
        for factor_index, factor in enumerate(("semantic", "structural")):
            profiles.append(carriage_profile(
                model,
                cfg,
                distance=distance,
                factor=factor,
                seed=2_000_000 + seed * 10_000 + distance * 101 + factor_index,
                device=device,
            ))
        print(f"[carriage {model_name} seed={seed}] distance {distance}", flush=True)
    head_conditions = []
    for distance in cfg.focus_distances:
        head_conditions.append(head_analysis_condition(
            model,
            cfg,
            distance=distance,
            seed=3_000_000 + seed * 10_000 + distance * 101,
            device=device,
        ))
        print(f"[heads {model_name} seed={seed}] distance {distance}", flush=True)
    lesions = []
    for lesion_index, lesion in enumerate(("clean", "direct", "crosscut")):
        lesions.append(evaluate_cell(
            model,
            cfg,
            seed=4_000_000 + seed * 10_000 + lesion_index,
            device=device,
            condition="load",
            topology="thin",
            distance=cfg.load_distance,
            rank=max(cfg.ranks),
            graphs=cfg.heldout_graphs,
            lesion=lesion,
        ))
    result = {
        "version": EXPERIMENT_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "model_name": model_name,
        "seed": int(seed),
        "initialisation_attempt": int(getattr(model, "initialisation_attempt", 0)),
        "initialisation_seed": int(getattr(model, "initialisation_seed", seed)),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "checks": checks,
        "evaluation": evaluate_grid(
            model,
            cfg,
            seed=5_000_000 + seed,
            device=device,
            graphs=cfg.heldout_graphs,
        ),
        "lesions": lesions,
        "carriage_profiles": profiles,
        "head_conditions": head_conditions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    print(f"[analysis {model_name} seed={seed}] cached {path}", flush=True)
    return result


def rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for index in np.where(counts > 1)[0]:
        positions = np.where(inverse == index)[0]
        ranks[positions] = ranks[positions].mean()
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 3 or np.std(x_arr[valid]) <= 0 or np.std(y_arr[valid]) <= 0:
        return float("nan")
    return float(np.corrcoef(rank_values(x_arr[valid]), rank_values(y_arr[valid]))[0, 1])


def mean_ci(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.nanmean(array))
    error = float(np.nanstd(array, ddof=1) / math.sqrt(len(array)) * 1.96) if len(array) > 1 else 0.0
    return mean, error


def configure_matplotlib() -> Any:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def save_figure(fig: Any, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(".png"), base.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, facecolor="white")
        print(f"[figure] {path}", flush=True)
    return [str(path) for path in paths]


MODEL_STYLE = {
    "1hop": {"color": "#6a51a3", "marker": "o", "label": "1-hop"},
    "2hop": {"color": "#2b8cbe", "marker": "s", "label": "2-hop"},
    "3hop": {"color": "#41ab5d", "marker": "^", "label": "3-hop"},
    "dense": {"color": "#d7301f", "marker": "D", "label": "Dense"},
}


def evaluation_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for row in result["evaluation"]:
            rows.append({"model": result["model_name"], "seed": result["seed"], **row})
    return rows


def _cell_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    condition: str,
    topology: str,
    distance: int | None = None,
    rank: int | None = None,
    key: str = "accuracy",
) -> list[float]:
    return [
        float(row[key]) for row in rows
        if row["model"] == model
        and row["condition"] == condition
        and row["topology"] == topology
        and (distance is None or int(row["distance"]) == int(distance))
        and (rank is None or int(row["rank"]) == int(rank))
    ]


def figure_reachability(
    results: Sequence[dict[str, Any]], cfg: Config, figures_dir: Path
) -> tuple[list[str], dict[str, Any]]:
    plt = configure_matplotlib()
    rows = evaluation_rows(results)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), gridspec_kw={"width_ratios": [1.35, 1.0]})
    summary: dict[str, Any] = {}
    for model_name in cfg.models:
        means, errors = [], []
        for distance in cfg.distances:
            values = _cell_values(
                rows,
                model=model_name,
                condition="reach",
                topology="thin",
                distance=distance,
            )
            mean, error = mean_ci(values)
            means.append(mean)
            errors.append(error)
        style = MODEL_STYLE[model_name]
        axes[0].errorbar(
            cfg.distances,
            means,
            yerr=errors,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            capsize=3,
            label=style["label"],
        )
        summary[model_name] = {"accuracy_mean": means, "accuracy_ci95": errors}
    axes[0].axhline(1.0 / cfg.classes, color="#777777", linestyle=":", linewidth=1.0, label="Chance")
    axes[0].axvspan(0.5, cfg.layers + 0.5, color="#eeeeee", alpha=0.7, zorder=0)
    axes[0].text(1.05, 0.82, "1-hop reachable", color="#555555", fontsize=9)
    axes[0].set_xticks(cfg.distances)
    axes[0].set_ylim(0.05, 1.04)
    axes[0].set_xlabel("Query–source distance $d$")
    axes[0].set_ylabel("Retrieval accuracy")
    axes[0].set_title("A  Rank-1 reachability", loc="left", fontweight="bold")
    axes[0].grid(True, linewidth=0.5, alpha=0.22)
    axes[0].legend(frameon=False, ncol=2, loc="lower left")

    # Difference is paired by seed and distance; construct explicitly to avoid pseudo-replication.
    for model_name in [name for name in cfg.models if name != "dense"]:
        gap_means, gap_errors = [], []
        for distance in cfg.distances:
            gaps = []
            for seed in cfg.seeds:
                dense = next(
                    float(row["accuracy"]) for row in rows
                    if row["model"] == "dense" and int(row["seed"]) == seed
                    and row["condition"] == "reach" and int(row["distance"]) == distance
                )
                masked = next(
                    float(row["accuracy"]) for row in rows
                    if row["model"] == model_name and int(row["seed"]) == seed
                    and row["condition"] == "reach" and int(row["distance"]) == distance
                )
                gaps.append(dense - masked)
            mean, error = mean_ci(gaps)
            gap_means.append(mean)
            gap_errors.append(error)
        style = MODEL_STYLE[model_name]
        axes[1].errorbar(
            cfg.distances,
            gap_means,
            yerr=gap_errors,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            capsize=3,
            label=f"Dense − {style['label']}",
        )
    axes[1].axhline(0.0, color="#777777", linewidth=0.9)
    axes[1].set_xticks(cfg.distances)
    axes[1].set_xlabel("Query–source distance $d$")
    axes[1].set_ylabel("Paired accuracy gap")
    axes[1].set_title("B  Dense advantage emerges beyond mask reach", loc="left", fontweight="bold")
    axes[1].grid(True, linewidth=0.5, alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Trained attention radius determines non-redundant retrieval reach",
        x=0.055,
        y=1.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.945,
        "All models are parameter-matched official GRITs; only the attention support used during training differs.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.4)
    return save_figure(fig, figures_dir / "fig1_reachability_accuracy"), summary


def figure_oversquashing(
    results: Sequence[dict[str, Any]], cfg: Config, figures_dir: Path
) -> tuple[list[str], dict[str, Any]]:
    plt = configure_matplotlib()
    rows = evaluation_rows(results)
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 4.9), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    summary: dict[str, Any] = {"accuracy": {}, "lesions": {}}
    for ax, topology, panel in zip(axes[:2], ("thin", "wide"), ("A", "B")):
        for model_name in cfg.models:
            means, errors = [], []
            for rank in cfg.ranks:
                mean, error = mean_ci(_cell_values(
                    rows,
                    model=model_name,
                    condition="load",
                    topology=topology,
                    rank=rank,
                ))
                means.append(mean)
                errors.append(error)
            style = MODEL_STYLE[model_name]
            ax.errorbar(
                cfg.ranks,
                means,
                yerr=errors,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                capsize=3,
                label=style["label"],
            )
            summary["accuracy"][f"{topology}_{model_name}"] = means
        ax.axhline(1.0 / cfg.classes, color="#777777", linestyle=":", linewidth=1.0)
        ax.set_xticks(cfg.ranks)
        ax.set_ylim(0.05, 1.04)
        ax.set_xlabel("Simultaneous cross-cut retrievals $r$")
        ax.set_title(
            f"{panel}  {'Thin cut (2 edges)' if topology == 'thin' else 'Wide cut (8 edges)'}",
            loc="left",
            fontweight="bold",
        )
        ax.grid(True, linewidth=0.5, alpha=0.22)
    axes[0].set_ylabel("Retrieval accuracy")
    axes[1].legend(frameon=False, ncol=2, loc="lower left")

    lesion_names = ("clean", "direct", "crosscut")
    lesion_labels = ("Clean", "Direct pair\nblocked", "All cross-cut\nattention blocked")
    x = np.arange(len(lesion_names))
    width = 0.19
    for model_index, model_name in enumerate(cfg.models):
        values_by_lesion = []
        for lesion in lesion_names:
            values = [
                float(row["accuracy"])
                for result in results
                if result["model_name"] == model_name
                for row in result["lesions"]
                if row["lesion"] == lesion
            ]
            values_by_lesion.append(values)
        means = [mean_ci(values)[0] for values in values_by_lesion]
        errors = [mean_ci(values)[1] for values in values_by_lesion]
        style = MODEL_STYLE[model_name]
        axes[2].bar(
            x + (model_index - 1.5) * width,
            means,
            yerr=errors,
            width=width,
            color=style["color"],
            alpha=0.85,
            capsize=2,
            label=style["label"],
        )
        summary["lesions"][model_name] = {name: mean for name, mean in zip(lesion_names, means)}
    axes[2].set_xticks(x, lesion_labels)
    axes[2].set_ylim(0.0, 1.04)
    axes[2].set_ylabel("Accuracy at thin-cut $r=6$")
    axes[2].set_title("C  Routing lesions", loc="left", fontweight="bold")
    axes[2].grid(True, axis="y", linewidth=0.5, alpha=0.22)
    fig.suptitle(
        "Bottleneck load at fixed reachable distance",
        x=0.045,
        y=1.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.945,
        f"Every query/source pair is at d={cfg.load_distance}; thin and wide graphs are both 24-node, 4-regular, and edge-count matched.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.0)
    return save_figure(fig, figures_dir / "fig2_oversquashing_and_lesions"), summary


def profile_target_values(
    result: Mapping[str, Any], *, factor: str, metric: str
) -> dict[int, float]:
    values: dict[int, float] = {}
    for profile in result["carriage_profiles"]:
        if profile["factor"] != factor:
            continue
        target = np.asarray(profile["is_target"], dtype=bool)
        array = np.asarray(profile[metric], dtype=float)
        values[int(profile["distance"])] = float(array[target].mean())
    return values


def figure_carriage(
    results: Sequence[dict[str, Any]], cfg: Config, figures_dir: Path
) -> tuple[list[str], dict[str, Any]]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 8.5), sharex=True)
    specs = [
        ("semantic", "functional", "A  Semantic functional carriage", r"$F_{sem}$"),
        ("structural", "functional", "B  Structural functional carriage", r"$F_{str}$"),
        ("semantic", "benefit", "C  Semantic beneficial carriage", r"$-B_{sem}$ (loss improvement)"),
        ("structural", "benefit", "D  Structural beneficial carriage", r"$-B_{str}$ (loss improvement)"),
    ]
    summary: dict[str, Any] = {}
    for ax, (factor, metric, title, ylabel) in zip(axes.ravel(), specs):
        for model_name in cfg.models:
            curves = []
            for result in results:
                if result["model_name"] != model_name:
                    continue
                by_distance = profile_target_values(result, factor=factor, metric=metric)
                curves.append([by_distance[distance] for distance in cfg.distances])
            array = np.asarray(curves, dtype=float)
            means = np.nanmean(array, axis=0)
            errors = (
                np.nanstd(array, axis=0, ddof=1) / math.sqrt(array.shape[0]) * 1.96
                if array.shape[0] > 1 else np.zeros(len(cfg.distances))
            )
            style = MODEL_STYLE[model_name]
            ax.plot(
                cfg.distances,
                means,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                label=style["label"],
            )
            ax.fill_between(cfg.distances, means - errors, means + errors, color=style["color"], alpha=0.12)
            summary[f"{factor}_{metric}_{model_name}"] = means
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.5, alpha=0.22)
    for ax in axes[1]:
        ax.set_xlabel("Planted query–source distance $d$")
        ax.set_xticks(cfg.distances)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle(
        "Aggregate semantic and structural carriage tracks learned retrieval reach",
        x=0.055,
        y=0.99,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.948,
        "Target-source donor swaps and full RRWP-plus-support transpositions; bands are seed-level 95% intervals.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.0, w_pad=2.0)
    return save_figure(fig, figures_dir / "fig3_semantic_structural_carriage"), summary


def build_head_rows(results: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rescues: list[dict[str, Any]] = []
    for result in results:
        model_name = str(result["model_name"])
        seed = int(result["seed"])
        for condition in result["head_conditions"]:
            distance = int(condition["distance"])
            clean_accuracy = float(condition["task_ablation"]["clean_accuracy"])
            task_functional = np.asarray(condition["task_ablation"]["functional"], dtype=float).mean(axis=-1)
            task_loss = np.asarray(condition["task_ablation"]["loss"], dtype=float).mean(axis=-1)
            factor_arrays: dict[str, dict[str, np.ndarray]] = {}
            for factor in ("semantic", "structural"):
                payload = condition[factor]
                factor_arrays[factor] = {
                    "score": np.asarray(payload["score"], dtype=float),
                    "functional_drop": np.asarray(payload["carriage"]["functional_drop"], dtype=float).mean(axis=-1),
                    "benefit_drop": np.asarray(payload["carriage"]["benefit_drop"], dtype=float).mean(axis=-1),
                }
                rescues.append({
                    "model": model_name,
                    "seed": seed,
                    "distance": distance,
                    "factor": factor,
                    "clean_accuracy": clean_accuracy,
                    "top": float(np.asarray(payload["rescue"]["top"], dtype=float).mean()),
                    "control": float(np.asarray(payload["rescue"]["layer_matched_control"], dtype=float).mean()),
                })
            for layer in range(task_functional.shape[0]):
                for head in range(task_functional.shape[1]):
                    rows.append({
                        "model": model_name,
                        "seed": seed,
                        "distance": distance,
                        "layer": layer,
                        "head": head,
                        "clean_accuracy": clean_accuracy,
                        "semantic_score": factor_arrays["semantic"]["score"][layer, head],
                        "structural_score": factor_arrays["structural"]["score"][layer, head],
                        "semantic_carriage_drop": factor_arrays["semantic"]["functional_drop"][layer, head],
                        "structural_carriage_drop": factor_arrays["structural"]["functional_drop"][layer, head],
                        "semantic_benefit_drop": factor_arrays["semantic"]["benefit_drop"][layer, head],
                        "structural_benefit_drop": factor_arrays["structural"]["benefit_drop"][layer, head],
                        "task_ablation_functional": task_functional[layer, head],
                        "task_ablation_loss": task_loss[layer, head],
                    })
    # Normalise only within a model/seed/distance cell; ranks and within-cell causal contrasts remain.
    for key in sorted({(row["model"], row["seed"], row["distance"]) for row in rows}):
        selected = [row for row in rows if (row["model"], row["seed"], row["distance"]) == key]
        for field in ("semantic_score", "structural_score", "task_ablation_functional"):
            scale = np.mean([abs(float(row[field])) for row in selected]) + EPS
            for row in selected:
                row[field + "_norm"] = float(row[field]) / scale
        for field in ("semantic_carriage_drop", "structural_carriage_drop"):
            scale = np.mean([abs(float(row[field])) for row in selected]) + EPS
            for row in selected:
                row[field + "_norm"] = float(row[field]) / scale
    return rows, rescues


def _scatter_by_model_distance(ax: Any, rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str) -> None:
    positive_y = y_key.endswith("_score_norm") or y_key == "task_ablation_functional_norm"
    for model_name in MODEL_ORDER:
        for distance in sorted({int(row["distance"]) for row in rows}):
            selected = [row for row in rows if row["model"] == model_name and int(row["distance"]) == distance]
            if not selected:
                continue
            style = MODEL_STYLE[model_name]
            face = style["color"] if distance == min(int(row["distance"]) for row in rows) else "none"
            ax.scatter(
                [max(float(row[x_key]), 1.0e-10) for row in selected],
                [max(float(row[y_key]), 1.0e-10) if positive_y else float(row[y_key]) for row in selected],
                s=34,
                marker=style["marker"],
                facecolor=face,
                edgecolor=style["color"],
                linewidth=0.9,
                alpha=0.72,
            )


def cell_correlation_summary(
    rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str
) -> tuple[float, float, list[float]]:
    pooled = spearman([row[x_key] for row in rows], [row[y_key] for row in rows])
    correlations = []
    cells = sorted({(row["model"], int(row["seed"]), int(row["distance"])) for row in rows})
    for model_name, seed, distance in cells:
        selected = [
            row for row in rows
            if row["model"] == model_name and int(row["seed"]) == seed and int(row["distance"]) == distance
        ]
        correlations.append(spearman([row[x_key] for row in selected], [row[y_key] for row in selected]))
    finite = np.asarray([value for value in correlations if np.isfinite(value)], dtype=float)
    median = float(np.median(finite)) if len(finite) else float("nan")
    return pooled, median, correlations


def figure_head_causality(
    results: Sequence[dict[str, Any]], cfg: Config, figures_dir: Path
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    plt = configure_matplotlib()
    rows, rescues = build_head_rows(results)
    # Do not turn receptive-field failure into an apparent mechanistic null: causal correlations
    # and rescue summaries are estimated only where the independently sampled clean task is solved.
    plot_rows = [row for row in rows if float(row["clean_accuracy"]) >= cfg.accuracy_gate]
    plot_rescues = [row for row in rescues if float(row["clean_accuracy"]) >= cfg.accuracy_gate]
    if not plot_rows or not plot_rescues:
        raise RuntimeError("no accuracy-qualified cells remain for the head-causality figure")
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.0))
    all_cells = {(row["model"], row["seed"], row["distance"]) for row in rows}
    qualified_cells = {(row["model"], row["seed"], row["distance"]) for row in plot_rows}
    summary: dict[str, Any] = {
        "accuracy_qualification": float(cfg.accuracy_gate),
        "qualified_cells": len(qualified_cells),
        "excluded_unsolved_cells": len(all_cells - qualified_cells),
    }

    x = np.asarray([float(row["structural_score_norm"]) for row in plot_rows])
    y = np.asarray([float(row["semantic_score_norm"]) for row in plot_rows])
    _scatter_by_model_distance(axes[0, 0], plot_rows, "structural_score_norm", "semantic_score_norm")
    lo = max(min(np.min(x), np.min(y)) * 0.7, 1.0e-6)
    hi = max(np.max(x), np.max(y)) * 1.35
    axes[0, 0].plot([lo, hi], [lo, hi], color="#777777", linestyle="--", linewidth=1.0)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlim(lo, hi)
    axes[0, 0].set_ylim(lo, hi)
    axes[0, 0].set_xlabel("Structural score (cell-normalised)")
    axes[0, 0].set_ylabel("Semantic score (cell-normalised)")
    axes[0, 0].set_title("A  Intervention-defined head plane", loc="left", fontweight="bold")

    bridge_specs = [
        ("semantic", axes[0, 1], "B  Semantic score predicts semantic-carriage loss"),
        ("structural", axes[0, 2], "C  Structural score predicts structural-carriage loss"),
    ]
    for factor, ax, title in bridge_specs:
        x_key = f"{factor}_score_norm"
        y_key = f"{factor}_carriage_drop_norm"
        _scatter_by_model_distance(ax, plot_rows, x_key, y_key)
        rho, median_rho, cell_rhos = cell_correlation_summary(plot_rows, x_key, y_key)
        summary[f"{factor}_score_carriage_drop_spearman"] = {
            "pooled": rho, "cell_median": median_rho, "by_cell": cell_rhos,
        }
        ax.set_xscale("log")
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.set_xlabel(f"{factor.capitalize()} score (cell-normalised)")
        ax.set_ylabel("Matching functional-carriage loss under ablation")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(0.04, 0.95, f"pooled $\\rho={rho:.2f}$\ncell median $\\rho={median_rho:.2f}$", transform=ax.transAxes, va="top")

    ablation_specs = [
        ("semantic", axes[1, 0], "D  Semantic score–task ablation"),
        ("structural", axes[1, 1], "E  Structural score–task ablation"),
    ]
    for factor, ax, title in ablation_specs:
        x_key = f"{factor}_score_norm"
        y_key = "task_ablation_functional_norm"
        _scatter_by_model_distance(ax, plot_rows, x_key, y_key)
        rho, median_rho, cell_rhos = cell_correlation_summary(plot_rows, x_key, y_key)
        summary[f"{factor}_score_task_ablation_spearman"] = {
            "pooled": rho, "cell_median": median_rho, "by_cell": cell_rhos,
        }
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{factor.capitalize()} score (cell-normalised)")
        ax.set_ylabel("Clean-input logit impact (cell-normalised)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(0.04, 0.95, f"pooled $\\rho={rho:.2f}$\ncell median $\\rho={median_rho:.2f}$", transform=ax.transAxes, va="top")

    rescue_ax = axes[1, 2]
    categories = [("semantic", "top"), ("semantic", "control"), ("structural", "top"), ("structural", "control")]
    values = []
    errors = []
    for factor, kind in categories:
        cell = [float(row[kind]) for row in plot_rescues if row["factor"] == factor]
        mean, error = mean_ci(cell)
        values.append(mean)
        errors.append(error)
    rescue_ax.bar(
        np.arange(4),
        values,
        yerr=errors,
        color=["#cb181d", "#fcae91", "#2171b5", "#9ecae1"],
        capsize=4,
        width=0.72,
    )
    rescue_ax.axhline(0.0, color="#777777", linewidth=0.8)
    rescue_ax.set_xticks(np.arange(4), ["Sem top", "Sem control", "Str top", "Str control"], rotation=16)
    rescue_ax.set_ylabel("Clean→corrupt mediated-effect fraction")
    rescue_ax.set_title("F  Score-selected rescue control", loc="left", fontweight="bold")
    summary["rescue_means"] = {f"{factor}_{kind}": value for (factor, kind), value in zip(categories, values)}

    for ax in axes.ravel():
        ax.grid(True, which="both", linewidth=0.5, alpha=0.20)
    from matplotlib.lines import Line2D

    model_handles = [
        Line2D([0], [0], marker=MODEL_STYLE[name]["marker"], color=MODEL_STYLE[name]["color"], linestyle="none", label=MODEL_STYLE[name]["label"])
        for name in cfg.models
    ]
    distance_handles = [
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor=("#555555" if distance == min(cfg.focus_distances) else "none"), linestyle="none", label=f"d={distance}")
        for distance in cfg.focus_distances
    ]
    fig.legend(model_handles + distance_handles, [handle.get_label() for handle in model_handles + distance_handles], frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "Specialisation scores identify heads that causally sustain matching carriage",
        x=0.045,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.045,
        0.958,
        f"Filled: redundant d=2; open: non-redundant d=5. Scores freeze support; carriage transposes it. Solved cells only ($\\geq$ {cfg.accuracy_gate:.2f}).",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.92), h_pad=2.2, w_pad=2.0)
    return save_figure(fig, figures_dir / "fig4_specialisation_carriage_causality"), summary, rows


def carriage_signature(result: Mapping[str, Any], distance: int) -> np.ndarray:
    channels = []
    for factor in ("semantic", "structural"):
        profile = next(
            value for value in result["carriage_profiles"]
            if value["factor"] == factor and int(value["distance"]) == int(distance)
        )
        source_distance = np.asarray(profile["source_distance"], dtype=int)
        for metric in ("functional", "benefit"):
            values = np.asarray(profile[metric], dtype=float)
            curve = np.asarray([
                float(values[source_distance == hop].mean()) if np.any(source_distance == hop) else 0.0
                for hop in range(8)
            ])
            curve = curve / (np.linalg.norm(curve) + EPS)
            channels.append(curve)
    return np.concatenate(channels)


def reach_accuracy(result: Mapping[str, Any], distance: int) -> float:
    return next(
        float(row["accuracy"]) for row in result["evaluation"]
        if row["condition"] == "reach" and int(row["distance"]) == int(distance)
    )


def figure_similarity(
    results: Sequence[dict[str, Any]], cfg: Config, figures_dir: Path
) -> tuple[list[str], dict[str, Any]]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    cells = []
    for seed in cfg.seeds:
        dense = next(result for result in results if result["model_name"] == "dense" and int(result["seed"]) == seed)
        for model_name in [name for name in cfg.models if name != "dense"]:
            masked = next(result for result in results if result["model_name"] == model_name and int(result["seed"]) == seed)
            for distance in cfg.distances:
                dense_signature = carriage_signature(dense, distance)
                masked_signature = carriage_signature(masked, distance)
                similarity = float(
                    np.dot(dense_signature, masked_signature)
                    / ((np.linalg.norm(dense_signature) * np.linalg.norm(masked_signature)) + EPS)
                )
                cells.append({
                    "seed": seed,
                    "model": model_name,
                    "distance": distance,
                    "carriage_similarity": similarity,
                    "accuracy_gap": abs(reach_accuracy(dense, distance) - reach_accuracy(masked, distance)),
                })
    for model_name in [name for name in cfg.models if name != "dense"]:
        means, errors = [], []
        for distance in cfg.distances:
            values = [
                row["carriage_similarity"] for row in cells
                if row["model"] == model_name and row["distance"] == distance
            ]
            mean, error = mean_ci(values)
            means.append(mean)
            errors.append(error)
        style = MODEL_STYLE[model_name]
        axes[0].errorbar(
            cfg.distances,
            means,
            yerr=errors,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            capsize=3,
            label=f"Dense vs {style['label']}",
        )
        selected = [row for row in cells if row["model"] == model_name]
        axes[1].scatter(
            [row["accuracy_gap"] for row in selected],
            [row["carriage_similarity"] for row in selected],
            color=style["color"],
            marker=style["marker"],
            s=46,
            alpha=0.75,
            label=style["label"],
        )
    axes[0].set_xticks(cfg.distances)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("Query–source distance $d$")
    axes[0].set_ylabel("Dense–masked carriage-signature cosine")
    axes[0].set_title("A  Mechanistic similarity across reach", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    rho = spearman([row["accuracy_gap"] for row in cells], [row["carriage_similarity"] for row in cells])
    axes[1].set_xlabel("Absolute dense–masked accuracy gap")
    axes[1].set_ylabel("Carriage-signature cosine")
    axes[1].set_title("B  Functional parity predicts carriage similarity", loc="left", fontweight="bold")
    axes[1].text(0.96, 0.95, f"Spearman $\\rho={rho:.2f}$", transform=axes[1].transAxes, ha="right", va="top")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, linewidth=0.5, alpha=0.22)
    fig.suptitle(
        "Aggregate carriage characterises when dense and masked GRIT learn similar solutions",
        x=0.055,
        y=1.01,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.2)
    return save_figure(fig, figures_dir / "fig5_dense_masked_similarity"), {"cells": cells, "gap_similarity_spearman": rho}


def create_outputs(results: Sequence[dict[str, Any]], cfg: Config, run_dir: Path) -> dict[str, Any]:
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    parameter_counts = {
        str(result["model_name"]): int(result["parameters"])
        for result in results
    }
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"parameter-matching control failed: {parameter_counts}")
    eval_rows = evaluation_rows(results)
    write_csv(tables_dir / "evaluation_grid.csv", eval_rows)
    fig1, reach_summary = figure_reachability(results, cfg, figures_dir)
    fig2, load_summary = figure_oversquashing(results, cfg, figures_dir)
    fig3, carriage_summary = figure_carriage(results, cfg, figures_dir)
    fig4, causal_summary, head_rows = figure_head_causality(results, cfg, figures_dir)
    fig5, similarity_summary = figure_similarity(results, cfg, figures_dir)
    write_csv(tables_dir / "per_head_metrics.csv", head_rows)
    summary = {
        "version": EXPERIMENT_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "official_grit_commit": OFFICIAL_GRIT_COMMIT,
        "config": asdict(cfg),
        "parameter_counts": parameter_counts,
        "reachability": reach_summary,
        "oversquashing_and_lesions": load_summary,
        "carriage": carriage_summary,
        "head_causality": causal_summary,
        "dense_masked_similarity": similarity_summary,
        "verification": [
            {
                "model": result["model_name"],
                "seed": result["seed"],
                "initialisation_attempt": result.get("initialisation_attempt", 0),
                "initialisation_seed": result.get("initialisation_seed", result["seed"]),
                **result["checks"],
            }
            for result in results
        ],
        "figures": fig1 + fig2 + fig3 + fig4 + fig5,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def environment_record(device: Any) -> dict[str, Any]:
    import torch

    return {
        "experiment_version": EXPERIMENT_VERSION,
        "official_grit_commit": OFFICIAL_GRIT_COMMIT,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() and str(device).startswith("cuda") else None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official-GRIT trained-mask reach, carriage, and specialisation")
    parser.add_argument("--run-name", default="reach_carriage_v1")
    parser.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--grit-dir", default=DEFAULT_GRIT_DIR)
    parser.add_argument("--phase", choices=("all", "train", "analyze", "figures"), default="all")
    parser.add_argument("--models", nargs="+", default=list(MODEL_ORDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--distances", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 6])
    parser.add_argument("--focus-distances", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--cluster-size", type=int, default=12)
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--pair-vocab", type=int, default=6)
    parser.add_argument("--rrwp-steps", type=int, default=10)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--load-distance", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-graphs", type=int, default=48)
    parser.add_argument("--heldout-graphs", type=int, default=256)
    parser.add_argument("--patience-checks", type=int, default=5)
    parser.add_argument("--accuracy-gate", type=float, default=0.85)
    parser.add_argument("--carriage-graphs", type=int, default=8)
    parser.add_argument("--carriage-donors", type=int, default=3)
    parser.add_argument("--carriage-batch-size", type=int, default=96)
    parser.add_argument("--score-graphs", type=int, default=24)
    parser.add_argument("--score-donors", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--ablation-graphs", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument("--allow-low-accuracy", action="store_true")
    parser.add_argument(
        "--failed-init-retries",
        type=int,
        default=2,
        help="validation-gated alternate initialisations after a convergence failure",
    )
    parser.add_argument("--skip-drive-mount", action="store_true")
    parser.add_argument("--skip-grit-install", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args(argv)


def make_config(args: argparse.Namespace) -> Config:
    values = {
        "run_name": args.run_name,
        "drive_root": args.drive_root,
        "n": args.n,
        "cluster_size": args.cluster_size,
        "classes": args.classes,
        "pair_vocab": args.pair_vocab,
        "rrwp_steps": args.rrwp_steps,
        "dim": args.dim,
        "heads": args.heads,
        "layers": args.layers,
        "dropout": args.dropout,
        "attention_dropout": args.attention_dropout,
        "models": tuple(args.models),
        "distances": tuple(args.distances),
        "ranks": tuple(args.ranks),
        "load_distance": args.load_distance,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "eval_every": args.eval_every,
        "validation_graphs": args.validation_graphs,
        "heldout_graphs": args.heldout_graphs,
        "patience_checks": args.patience_checks,
        "accuracy_gate": args.accuracy_gate,
        "carriage_graphs": args.carriage_graphs,
        "carriage_donors": args.carriage_donors,
        "carriage_batch_size": args.carriage_batch_size,
        "score_graphs": args.score_graphs,
        "score_donors": args.score_donors,
        "score_batch_size": args.score_batch_size,
        "ablation_graphs": args.ablation_graphs,
        "focus_distances": tuple(args.focus_distances),
        "seeds": tuple(args.seeds),
        "device": args.device,
    }
    if args.fast_dev_run:
        values.update({
            "run_name": args.run_name + "_fast_dev",
            "models": ("1hop", "dense"),
            "seeds": (int(args.seeds[0]),),
            "steps": 20,
            "batch_size": 8,
            "eval_every": 5,
            "validation_graphs": 4,
            "heldout_graphs": 8,
            "patience_checks": 100,
            "accuracy_gate": 0.0,
            "carriage_graphs": 2,
            "carriage_donors": 1,
            "carriage_batch_size": 32,
            "score_graphs": 2,
            "score_donors": 1,
            "score_batch_size": 1,
            "ablation_graphs": 4,
        })
    cfg = Config(**values)
    cfg.validate()
    return cfg


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    import torch

    args = parse_args(argv)
    if not args.skip_drive_mount:
        mount_drive()
    cfg = make_config(args)
    if args.phase != "figures":
        setup_official_grit(Path(args.grit_dir), install=not args.skip_grit_install)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA unavailable; official GRIT execution will be slow", flush=True)
    run_dir = Path(cfg.drive_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {
        "config": asdict(cfg),
        "fingerprint": config_fingerprint(cfg),
        "failed_initialisation_retries": int(args.failed_init_retries),
    })
    write_json(run_dir / "environment.json", environment_record(device))

    models: dict[tuple[str, int], Any] = {}
    if args.phase in {"all", "train", "analyze"}:
        for model_name in cfg.models:
            for seed in cfg.seeds:
                model, payload = train_model(
                    cfg,
                    model_name=model_name,
                    seed=seed,
                    run_dir=run_dir,
                    device=device,
                    force=args.force_retrain,
                    load_only=args.phase == "analyze",
                )
                validation_gate = checkpoint_gate(
                    cfg, model_name, payload, split="best_validation"
                )
                attempt = int(payload.get("initialisation_attempt", 0))
                while (
                    validation_gate < cfg.accuracy_gate
                    and attempt < int(args.failed_init_retries)
                    and args.phase != "analyze"
                    and not args.allow_low_accuracy
                ):
                    attempt += 1
                    print(
                        f"[retry {model_name} seed={seed}] validation accuracy "
                        f"{validation_gate:.3f} < {cfg.accuracy_gate:.3f}; "
                        f"trying initialisation {attempt}/{args.failed_init_retries}",
                        flush=True,
                    )
                    model, payload = train_model(
                        cfg,
                        model_name=model_name,
                        seed=seed,
                        run_dir=run_dir,
                        device=device,
                        force=True,
                        load_only=False,
                        initialisation_attempt=attempt,
                    )
                    validation_gate = checkpoint_gate(
                        cfg, model_name, payload, split="best_validation"
                    )
                heldout_gate = checkpoint_gate(cfg, model_name, payload, split="heldout")
                print(
                    f"[gate {model_name} seed={seed}] validation={validation_gate:.3f} "
                    f"heldout={heldout_gate:.3f} init_attempt={attempt}",
                    flush=True,
                )
                gate = min(validation_gate, heldout_gate)
                if gate < cfg.accuracy_gate and not args.allow_low_accuracy:
                    raise RuntimeError(
                        f"{model_name} seed {seed} failed reachable-task accuracy gate "
                        f"(validation={validation_gate:.3f}, heldout={heldout_gate:.3f}, "
                        f"required={cfg.accuracy_gate:.3f}); refusing causal analysis"
                    )
                models[(model_name, seed)] = model
    if args.phase == "train":
        return {"run_dir": str(run_dir), "phase": "train"}

    results = []
    if args.phase in {"all", "analyze"}:
        for model_name in cfg.models:
            for seed in cfg.seeds:
                results.append(analyze_model(
                    models[(model_name, seed)],
                    cfg,
                    model_name=model_name,
                    seed=seed,
                    run_dir=run_dir,
                    device=device,
                    force=args.force_analysis,
                ))
    else:
        for model_name in cfg.models:
            for seed in cfg.seeds:
                path = analysis_path(run_dir, cfg, model_name, seed)
                if not path.exists():
                    raise FileNotFoundError(f"figures phase requires {path}")
                results.append(torch.load(path, map_location="cpu", weights_only=False))

    summary = create_outputs(results, cfg, run_dir)
    print("\n[done]", flush=True)
    print(f"  run_dir: {run_dir}", flush=True)
    print(f"  checkpoints: {run_dir / 'checkpoints'}", flush=True)
    print(f"  analyses: {run_dir / 'analysis'}", flush=True)
    print(f"  figures: {run_dir / 'figures'}", flush=True)
    return summary


if __name__ == "__main__":
    main([
        "--run-name", "reach_carriage_v1",
        "--phase", "all",
        "--models", "1hop", "2hop", "3hop", "dense",
        "--seeds", "0", "1", "2",
        "--steps", "1500",
        "--distances", "1", "2", "3", "4", "5", "6",
        "--ranks", "1", "2", "4", "6",
        "--focus-distances", "2", "5",
    ])
