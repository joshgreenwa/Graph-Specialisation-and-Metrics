"""Standalone Colab: causal validation of semantic/structural GRIT head scores.

Paste this whole file into one Google Colab cell and run it.  The experiment uses the
official LiamMa/GRIT ``GritTransformerLayer`` at the repository-pinned commit.  It:

1. mounts Google Drive and installs official GRIT plus its PyG dependencies;
2. trains a dense GRIT jointly on two balanced, controlled tasks over the same cycle graphs;
3. caches checkpoints and every expensive analysis tensor to Drive;
4. computes planted-source semantic and structural specialisation scores;
5. sweeps every head's clean-input ablation impact; and
6. performs clean-to-corrupt head-output interchange patching for necessity-plus-rescue
   double dissociation; and
7. cumulatively ablates the score-selected semantic and structural head families to test
   distributed necessity beyond single-head redundancy.

Tasks
-----
``semantic``
    A marked query and source identify one other node and the model must return that
    node's random value class.  The source position is random and independent of graph
    structure.  Its key is also placed at the query as a redundant associative cue.

``structural``
    A query and anchor are marked and the model must classify their shortest-path distance
    on an even cycle.  Node keys/values are independent distractors.  This task depends on
    the RRWP payload but not on semantic payload values.

The source/anchor marker is deliberately shared across tasks.  It controls away the difficulty
of discovering an address: the causal comparison is whether a head transports the marked node's
semantic value or its structural relation to the query, not whether GRIT can first learn a large
categorical equality circuit.

Interventions
-------------
Semantic score/corruption
    Replace the planted semantic target's complete role-conditioned content row with an
    on-manifold donor having the same key and role but a different value.  In this dataset,
    that is equivalent to replacing only the value one-hot while task-control fields stay
    fixed.  Donor deltas are averaged before the functional magnitude.

Structural score/corruption
    Replace the anchor's complete RRWP footprint with that of a degree-matched structural
    donor while content and the dense all-pairs mask remain fixed.  There is no reciprocal
    replacement: every RRWP entry not incident to the anchor is unchanged.  Every cycle
    node has degree two, so donor matching is exact.  Donor deltas are averaged before the
    functional magnitude.

The score at head (l,h) is the production transport-site estimator

    sum_i sqrt(sum_t (phi_t[i,l,h] dot mean_k Delta-o[i,l,h,k])**2),

where ``o`` is the official GRIT attention module's routed per-head value ``wV`` and ``phi``
is the clean output-logit gradient.  Scores are label-free over all output logits.

Primary outputs (PNG + vector PDF)
----------------------------------
``fig1_specialisation_plane``
    Structural score (x) versus semantic score (y), amplitude-normalised per seed.
``fig2_score_ablation``
    Score versus same-channel single-head functional ablation impact.
``fig3_necessity_rescue_double_dissociation``
    Simultaneous semantic/structural head-family necessity and causal rescue matrices plus
    seed-level interaction contrasts.
``fig4_iterative_family_ablation``
    Cumulative fixed-order family ablation curves for loss and accuracy on both tasks.
``fig5_joint_influence_selectivity``
    The calibrated D--J head plane plus tests that J predicts overall ablation impact and
    D predicts semantic-versus-structural ablation and rescue role.
``fig6_joint_selectivity_family_ablation``
    Fixed score-selected cumulative ablations of semantic specialists, structural specialists,
    high-J generalists, and low-J/inert heads on both tasks.
``tables/validation_performance.csv`` and the printed ``[validation]`` block
    Per-seed best-checkpoint held-out validation accuracy and loss (overall and per task)
    with the across-seed mean ± sample standard deviation.

Paper claims must use the default (or larger) run, never ``--fast-dev-run``.  The fast run
exists only to verify installation, official-layer execution, caching, and plotting.
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


EXPERIMENT_VERSION = "causal-specialisation-double-dissociation-v2-shared-source-marker"
ANALYSIS_VERSION = "causal-specialisation-matched-donor-swaps-v1"
FAMILY_ABLATION_REVISION = "score-selected-prefix-family-ablation-v1"
DJ_FAMILY_ABLATION_REVISION = "joint-selectivity-prefix-family-ablation-v1"
OFFICIAL_GRIT_URL = "https://github.com/LiamMa/GRIT.git"
OFFICIAL_GRIT_COMMIT = "6c988ea600a606fbb49a2246c64a2d37396b3ab5"
DEFAULT_GRIT_DIR = "/content/GRIT"
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/graph_specialisation_metrics/causal_specialisation_double_dissociation"
MODE_SEMANTIC = 0
MODE_STRUCTURAL = 1
MODE_NAMES = {MODE_SEMANTIC: "semantic", MODE_STRUCTURAL: "structural"}
EPS = 1.0e-12
DJ_RELIABILITY_FLOOR = 0.50  # combined sensitivity relative to the within-seed head mean
DJ_SELECTIVITY_THRESHOLD = 0.20  # equivalent to a 1.5:1 calibrated channel ratio


# ======================================================================================
# Colab/bootstrap helpers
# ======================================================================================


def _run(cmd: Sequence[str], *, check: bool = True) -> int:
    print(f"[cmd] {' '.join(map(str, cmd))}", flush=True)
    return subprocess.run(list(map(str, cmd)), check=check).returncode


def mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
    except Exception as exc:  # pragma: no cover - only outside/in broken Colab.
        print(f"[drive] mount unavailable ({exc}); local output path will be used", flush=True)


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
    for pkg in ("pyg-lib", "torch-spline-conv", "torch-cluster"):
        _run([sys.executable, "-m", "pip", "install", "-q", pkg, "-f", wheel_url], check=False)
    for pkg in ("torch-scatter", "torch-sparse"):
        rc = _run([sys.executable, "-m", "pip", "install", "-q", pkg, "-f", wheel_url], check=False)
        if rc:
            raise SystemExit(
                f"Required PyG extension {pkg!r} has no matching wheel at {wheel_url}. "
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"cannot JSON-encode {type(value)!r}")


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
# Configuration and deterministic synthetic data
# ======================================================================================


@dataclass(frozen=True)
class Config:
    run_name: str = "cycle_dual_v2"
    drive_root: str = DEFAULT_DRIVE_ROOT
    n: int = 16
    key_vocab: int = 32
    classes: int = 8
    rrwp_steps: int = 10
    dim: int = 96
    heads: int = 8
    layers: int = 3
    dropout: float = 0.0
    attention_dropout: float = 0.05
    batch_size: int = 64
    steps: int = 3000
    lr: float = 2.0e-3
    weight_decay: float = 1.0e-5
    eval_every: int = 100
    validation_graphs: int = 512
    patience_checks: int = 10
    accuracy_gate: float = 0.90
    score_graphs: int = 96
    score_donors: int = 6
    score_batch_size: int = 12
    ablation_graphs: int = 256
    rescue_graphs: int = 128
    analysis_batch_size: int = 64
    top_group_size: int = 3
    seeds: tuple[int, ...] = (0, 1, 2)
    device: str = "cuda"

    @property
    def feature_dim(self) -> int:
        # key | value | semantic query-key | query | shared source/anchor | semantic-mode | structural-mode
        return 2 * self.key_vocab + self.classes + 4

    @property
    def max_cycle_distance(self) -> int:
        return self.n // 2

    def validate(self) -> None:
        if self.n % 2:
            raise ValueError("n must be even so the maximum cycle distance is unambiguous")
        if self.classes != self.max_cycle_distance:
            raise ValueError("classes must equal n//2: the two tasks share one balanced output head")
        if self.key_vocab < self.n:
            raise ValueError("key_vocab must be >= n so every node key can be unique")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.score_donors < 1:
            raise ValueError("score_donors must be positive")


def config_fingerprint(cfg: Config) -> str:
    payload = {"version": EXPERIMENT_VERSION, **asdict(cfg)}
    payload.pop("drive_root", None)
    payload.pop("run_name", None)
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def analysis_fingerprint(cfg: Config) -> str:
    """Fingerprint intervention-dependent work without invalidating training caches."""

    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "training_fingerprint": config_fingerprint(cfg),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class SynthBatch:
    x: Any
    rrwp: Any
    q_idx: Any
    target_idx: Any
    anchor_idx: Any
    y: Any
    mode: Any

    def to(self, device: Any) -> "SynthBatch":
        return SynthBatch(**{name: getattr(self, name).to(device) for name in self.__dataclass_fields__})

    def cpu(self) -> "SynthBatch":
        return self.to("cpu")

    def __len__(self) -> int:
        return int(self.x.size(0))

    def slice(self, start: int, stop: int) -> "SynthBatch":
        return SynthBatch(**{name: getattr(self, name)[start:stop] for name in self.__dataclass_fields__})

    def select(self, indices: Any) -> "SynthBatch":
        return SynthBatch(**{name: getattr(self, name)[indices] for name in self.__dataclass_fields__})


def cycle_distance(n: int, a: int, b: int) -> int:
    delta = abs(int(a) - int(b))
    return min(delta, int(n) - delta)


@functools.lru_cache(maxsize=8)
def cycle_rrwp(n: int, steps: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.float32)
    idx = np.arange(n)
    adj[idx, (idx + 1) % n] = 1.0
    adj[idx, (idx - 1) % n] = 1.0
    p = adj / adj.sum(axis=1, keepdims=True)
    out = np.zeros((n, n, steps), dtype=np.float32)
    power = np.eye(n, dtype=np.float32)
    for s in range(steps):
        if s:
            power = power @ p
        out[:, :, s] = power
    return out


def make_batch(cfg: Config, size: int, seed: int, mode: int | None = None) -> SynthBatch:
    import torch

    rng = np.random.default_rng(int(seed))
    rrwp0 = cycle_rrwp(cfg.n, cfg.rrwp_steps)
    xs = np.zeros((size, cfg.n, cfg.feature_dim), dtype=np.float32)
    rrwps = np.repeat(rrwp0[None, ...], size, axis=0)
    q_idx = np.zeros(size, dtype=np.int64)
    target_idx = np.zeros(size, dtype=np.int64)
    anchor_idx = np.full(size, -1, dtype=np.int64)
    ys = np.zeros(size, dtype=np.int64)
    modes = np.zeros(size, dtype=np.int64)

    key_start = 0
    value_start = cfg.key_vocab
    query_key_start = cfg.key_vocab + cfg.classes
    query_flag = 2 * cfg.key_vocab + cfg.classes
    source_flag = query_flag + 1
    sem_mode_flag = query_flag + 2
    str_mode_flag = query_flag + 3

    if mode is None:
        requested_modes = np.arange(size, dtype=np.int64) % 2
        rng.shuffle(requested_modes)
    else:
        requested_modes = np.full(size, int(mode), dtype=np.int64)

    for b in range(size):
        keys = rng.permutation(cfg.key_vocab)[: cfg.n]
        values = rng.integers(0, cfg.classes, size=cfg.n)
        xs[b, np.arange(cfg.n), key_start + keys] = 1.0
        xs[b, np.arange(cfg.n), value_start + values] = 1.0
        q = int(rng.integers(0, cfg.n))
        q_idx[b] = q
        xs[b, q, query_flag] = 1.0
        m = int(requested_modes[b])
        modes[b] = m
        if m == MODE_SEMANTIC:
            candidates = [i for i in range(cfg.n) if i != q]
            target = int(rng.choice(candidates))
            target_idx[b] = target
            xs[b, q, query_key_start + keys[target]] = 1.0
            # Use the same role marker as the structural anchor.  This makes addressing
            # difficulty identical across modes while the required payload remains orthogonal.
            xs[b, target, source_flag] = 1.0
            xs[b, q, sem_mode_flag] = 1.0
            ys[b] = int(values[target])
        else:
            d = int(rng.integers(1, cfg.max_cycle_distance + 1))
            direction = int(rng.choice((-1, 1)))
            anchor = int((q + direction * d) % cfg.n)
            target_idx[b] = anchor
            anchor_idx[b] = anchor
            xs[b, q, str_mode_flag] = 1.0
            xs[b, anchor, source_flag] = 1.0
            ys[b] = d - 1

    return SynthBatch(
        x=torch.from_numpy(xs),
        rrwp=torch.from_numpy(rrwps),
        q_idx=torch.from_numpy(q_idx),
        target_idx=torch.from_numpy(target_idx),
        anchor_idx=torch.from_numpy(anchor_idx),
        y=torch.from_numpy(ys),
        mode=torch.from_numpy(modes),
    )


def concat_batches(batches: Sequence[SynthBatch]) -> SynthBatch:
    import torch

    return SynthBatch(**{
        name: torch.cat([getattr(batch, name) for batch in batches], dim=0)
        for name in SynthBatch.__dataclass_fields__
    })


def transpose_rrwp(rrwp: Any, u: int, v: int) -> Any:
    """Relabel RRWP for the independent full-isomorphism verification check."""

    if int(u) == int(v):
        return rrwp.clone()
    order = list(range(int(rrwp.size(0))))
    order[int(u)], order[int(v)] = order[int(v)], order[int(u)]
    return rrwp[order][:, order].clone()


def copy_rrwp_footprint(rrwp: Any, source: int, donor: int) -> Any:
    """Replace one source node's dense RRWP row/column with a donor footprint.

    There is no reciprocal replacement: every entry not incident to ``source`` remains
    unchanged.  This is the mask-frozen, single-node structural donor-swap used by the
    final methodology.
    """

    source, donor = int(source), int(donor)
    if source == donor:
        return rrwp.clone()
    output = rrwp.clone()
    donor_row = rrwp[donor].clone()
    donor_column = rrwp[:, donor].clone()
    output[source] = donor_row
    output[:, source] = donor_column
    output[source, source] = rrwp[donor, donor]
    return output


def replace_value(cfg: Config, row: Any, value: int) -> Any:
    out = row.clone()
    start = cfg.key_vocab
    out[start : start + cfg.classes] = 0.0
    out[start + int(value)] = 1.0
    return out


def draw_other_class(rng: np.random.Generator, classes: int, current: int) -> int:
    value = int(rng.integers(0, classes - 1))
    return value + (1 if value >= int(current) else 0)


def structural_donors(cfg: Config, q: int, anchor: int) -> list[int]:
    clean_distance = cycle_distance(cfg.n, q, anchor)
    candidates = [
        node for node in range(cfg.n)
        if node not in {q, anchor} and cycle_distance(cfg.n, q, node) != clean_distance
    ]
    if not candidates:
        raise RuntimeError("no degree-matched structural donor changes the planted distance")
    return candidates


def make_replicas(
    cfg: Config,
    clean: SynthBatch,
    *,
    factor: str,
    donors: int,
    seed: int,
    no_op: bool = False,
) -> SynthBatch:
    """Return graph-major replicas: [g0 clean, g0 k1..K, g1 clean, ...]."""
    import torch

    rng = np.random.default_rng(int(seed))
    replicas: list[SynthBatch] = []
    for b in range(len(clean)):
        one = clean.slice(b, b + 1)
        replicas.append(one)
        source = int(one.target_idx[0])
        q = int(one.q_idx[0])
        for _ in range(int(donors)):
            x = one.x.clone()
            rrwp = one.rrwp.clone()
            if factor == "semantic":
                current = int(torch.argmax(x[0, source, cfg.key_vocab : cfg.key_vocab + cfg.classes]).item())
                donor_value = current if no_op else draw_other_class(rng, cfg.classes, current)
                x[0, source] = replace_value(cfg, x[0, source], donor_value)
            elif factor == "structural":
                if no_op:
                    donor = source
                else:
                    candidates = structural_donors(cfg, q, source)
                    donor = int(rng.choice(candidates))
                rrwp[0] = copy_rrwp_footprint(rrwp[0], source, donor)
            else:
                raise ValueError(f"unknown factor {factor!r}")
            replicas.append(SynthBatch(
                x=x,
                rrwp=rrwp,
                q_idx=one.q_idx.clone(),
                target_idx=one.target_idx.clone(),
                anchor_idx=one.anchor_idx.clone(),
                y=one.y.clone(),
                mode=one.mode.clone(),
            ))
    return concat_batches(replicas)


def make_single_corruption(cfg: Config, clean: SynthBatch, *, factor: str, seed: int) -> SynthBatch:
    reps = make_replicas(cfg, clean, factor=factor, donors=1, seed=seed)
    # Graph-major [clean, corrupt]; retain the odd entries.
    indices = np.arange(1, 2 * len(clean), 2, dtype=np.int64)
    return reps.select(indices)


# ======================================================================================
# Official GRIT model and head intervention hooks
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


class OfficialGRITDualModel:  # replaced with nn.Module dynamically to keep bootstrap imports late.
    pass


def build_model_class() -> type:
    import torch
    import torch.nn as nn
    from torch_geometric.data import Data
    from grit.layer.grit_layer import GritTransformerLayer

    class _OfficialGRITDualModel(nn.Module):
        """Transparent synthetic adapter around the official GRIT transformer layers.

        Only the small input/RRWP/output linear maps are local.  Every attention, edge-enhance,
        residual, normalisation, and FFN operation is the pinned official implementation.  This
        mirrors ``synthetic_bottleneck_retrieval_official.py`` in the dissertation repository.
        """

        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.cfg = cfg
            self.dim = cfg.dim
            self.H = cfg.heads
            self.L = cfg.layers
            self.dh = cfg.dim // cfg.heads
            self.input_encoder = nn.Linear(cfg.feature_dim, cfg.dim)
            self.node_rrwp_encoder = nn.Linear(cfg.rrwp_steps, cfg.dim, bias=False)
            self.pair_rrwp_encoder = nn.Linear(cfg.rrwp_steps, cfg.dim, bias=False)
            self.edge_type_encoder = nn.Embedding(2, cfg.dim)
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

        def _pyg_batch(self, batch: SynthBatch) -> Any:
            bsz, n = int(batch.x.size(0)), int(batch.x.size(1))
            device = batch.x.device
            local = torch.arange(n, device=device)
            src_local = local.repeat_interleave(n)
            dst_local = local.repeat(n)
            src = torch.cat([src_local + graph * n for graph in range(bsz)], dim=0)
            dst = torch.cat([dst_local + graph * n for graph in range(bsz)], dim=0)
            edge_type_local = (src_local != dst_local).long()
            edge_type = edge_type_local.repeat(bsz)
            edge_rrwp = batch.rrwp.reshape(bsz * n * n, self.cfg.rrwp_steps)
            data = Data(num_nodes=bsz * n)
            diag = torch.arange(n, device=device)
            node_rrwp = batch.rrwp[:, diag, diag].reshape(bsz * n, self.cfg.rrwp_steps)
            data.x = self.input_encoder(batch.x.reshape(bsz * n, -1)) + self.node_rrwp_encoder(node_rrwp)
            data.edge_index = torch.stack([src, dst], dim=0)
            data.edge_attr = self.edge_type_encoder(edge_type) + self.pair_rrwp_encoder(edge_rrwp)
            data.batch = torch.arange(bsz, device=device).repeat_interleave(n)
            data.deg = torch.full((bsz * n,), float(n), device=device)
            data.log_deg = torch.log(data.deg + 1.0)
            data.graph_num_nodes = torch.full((bsz,), n, dtype=torch.long, device=device)
            return data

        def node_states(self, batch: SynthBatch) -> Any:
            data = self._pyg_batch(batch)
            for layer in self.layers:
                data = layer(data)
            return data.x.reshape(len(batch), self.cfg.n, self.cfg.dim)

        def forward(self, batch: SynthBatch) -> Any:
            states = self.node_states(batch)
            row = torch.arange(len(batch), device=states.device)
            query_states = states[row, batch.q_idx.long()]
            return self.output_head(query_states)

    return _OfficialGRITDualModel


@contextlib.contextmanager
def ablate_head(model: Any, layer: int, head: int):
    with ablate_heads(model, [(int(layer), int(head))]):
        yield


@contextlib.contextmanager
def ablate_heads(model: Any, heads: Sequence[tuple[int, int]]):
    """Zero several routed heads in one pass, grouping hooks by layer."""

    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(int(layer), []).append(int(head))
    handles = []

    def make_hook(head_indices: Sequence[int]):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            h_out, e_out = output
            changed = h_out.clone()
            changed[:, list(head_indices), :] = 0.0
            return changed, e_out

        return hook

    try:
        for layer, head_indices in by_layer.items():
            handles.append(model.attention_layers[layer].register_forward_hook(make_hook(head_indices)))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def patch_head_output(model: Any, layer: int, head: int, source_wv: Any):
    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        h_out, e_out = output
        if tuple(h_out.shape) != tuple(source_wv.shape):
            raise RuntimeError(f"patch alignment failed: current {tuple(h_out.shape)} vs source {tuple(source_wv.shape)}")
        changed = h_out.clone()
        changed[:, int(head), :] = source_wv.to(changed.device, changed.dtype)[:, int(head), :]
        return changed, e_out

    handle = model.attention_layers[int(layer)].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def capture_forward(model: Any, batch: SynthBatch, *, want_grad: bool, want_attention: bool = False) -> dict[str, Any]:
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
# Training and cache management
# ======================================================================================


def set_seed(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def split_metrics(logits: Any, batch: SynthBatch) -> dict[str, float]:
    import torch.nn.functional as F

    loss = F.cross_entropy(logits, batch.y.long(), reduction="none")
    pred = logits.argmax(dim=-1)
    out = {"loss": float(loss.mean().detach().cpu()), "accuracy": float((pred == batch.y).float().mean().detach().cpu())}
    for mode, name in MODE_NAMES.items():
        mask = batch.mode == int(mode)
        if bool(mask.any()):
            out[f"{name}_loss"] = float(loss[mask].mean().detach().cpu())
            out[f"{name}_accuracy"] = float((pred[mask] == batch.y[mask]).float().mean().detach().cpu())
    return out


def predict_in_chunks(
    model: Any,
    batch: SynthBatch,
    *,
    device: Any,
    chunk_size: int,
    ablation: tuple[int, int] | None = None,
    ablations: Sequence[tuple[int, int]] | None = None,
) -> Any:
    import torch

    if ablation is not None and ablations is not None:
        raise ValueError("pass ablation or ablations, not both")
    outputs = []
    context = contextlib.nullcontext()
    if ablation is not None:
        context = ablate_head(model, ablation[0], ablation[1])
    elif ablations is not None:
        context = ablate_heads(model, ablations)
    with context, torch.no_grad():
        for start in range(0, len(batch), int(chunk_size)):
            outputs.append(model(batch.slice(start, min(start + chunk_size, len(batch))).to(device)).detach().cpu())
    return torch.cat(outputs, dim=0)


def evaluate_model(model: Any, cfg: Config, *, seed: int, device: Any) -> dict[str, float]:
    batch = make_batch(cfg, cfg.validation_graphs, seed, mode=None)
    logits = predict_in_chunks(model, batch, device=device, chunk_size=cfg.analysis_batch_size)
    return split_metrics(logits, batch)


def checkpoint_path(run_dir: Path, cfg: Config, seed: int) -> Path:
    return run_dir / "checkpoints" / f"seed_{int(seed)}__{config_fingerprint(cfg)}.pt"


def train_seed(
    cfg: Config,
    *,
    seed: int,
    run_dir: Path,
    device: Any,
    force: bool,
    load_only: bool,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    global OfficialGRITDualModel
    if OfficialGRITDualModel.__name__ == "OfficialGRITDualModel":
        OfficialGRITDualModel = build_model_class()
    path = checkpoint_path(run_dir, cfg, seed)
    # Seed before construction: cached and fresh runs must instantiate identical weights.
    set_seed(seed)
    model = OfficialGRITDualModel(cfg).to(device)
    if path.exists() and not force:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("fingerprint") != config_fingerprint(cfg):
            raise RuntimeError(f"checkpoint fingerprint mismatch at {path}")
        model.load_state_dict(payload["state_dict"])
        model.eval()
        print(f"[train seed={seed}] loaded {path}", flush=True)
        return model, payload
    if load_only:
        raise FileNotFoundError(f"analysis requested but checkpoint is missing: {path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.steps, 1), eta_min=cfg.lr * 0.05)
    fixed_val = make_batch(cfg, cfg.validation_graphs, 50_000 + seed, mode=None)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, float] = {}
    history: list[dict[str, Any]] = []
    good_checks = 0
    started = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        batch = make_batch(cfg, cfg.batch_size, seed * 1_000_003 + step, mode=None).to(device)
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y.long())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % cfg.eval_every == 0 or step == cfg.steps:
            model.eval()
            val_logits = predict_in_chunks(model, fixed_val, device=device, chunk_size=cfg.analysis_batch_size)
            metrics = split_metrics(val_logits, fixed_val)
            row = {
                "step": step,
                "train_loss": float(loss.detach().cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": time.time() - started,
                **metrics,
            }
            history.append(row)
            print(
                f"[train seed={seed}] {step:4d}/{cfg.steps} loss={row['loss']:.4f} "
                f"sem={row['semantic_accuracy']:.3f} str={row['structural_accuracy']:.3f}",
                flush=True,
            )
            if row["loss"] < best_loss:
                best_loss = float(row["loss"])
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_metrics = dict(metrics)
            if min(row["semantic_accuracy"], row["structural_accuracy"]) >= 0.995:
                good_checks += 1
            else:
                good_checks = 0
            model.train()
            if good_checks >= cfg.patience_checks:
                print(f"[train seed={seed}] early stop after {good_checks} near-perfect checks", flush=True)
                break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint candidate")
    model.load_state_dict(best_state)
    model.eval()
    final_metrics = evaluate_model(model, cfg, seed=80_000 + seed, device=device)
    payload = {
        "version": EXPERIMENT_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "seed": int(seed),
        "config": asdict(cfg),
        "official_grit_commit": OFFICIAL_GRIT_COMMIT,
        "state_dict": best_state,
        "best_validation": best_metrics,
        "heldout_validation": final_metrics,
        "history": history,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    write_csv(run_dir / "tables" / f"training_seed_{seed}.csv", history)
    print(f"[train seed={seed}] cached {path}", flush=True)
    return model, payload


# ======================================================================================
# Exact transport-site score, ablation sweep, and causal rescue
# ======================================================================================


def score_channel_batch(model: Any, cfg: Config, clean: SynthBatch, *, factor: str, seed: int, device: Any) -> Any:
    """Per-graph [G,L,H] functional score; donor average occurs before magnitude."""
    import torch

    replicas = make_replicas(cfg, clean.cpu(), factor=factor, donors=cfg.score_donors, seed=seed).to(device)
    result = capture_forward(model, replicas, want_grad=True)
    logits = result["logits"]
    wv = result["wV"]
    graphs = len(clean)
    reps = cfg.score_donors + 1
    clean_rows = torch.arange(graphs, device=device) * reps
    accum = [torch.zeros(graphs, cfg.n, cfg.heads, device=device) for _ in range(cfg.layers)]
    for output_idx in range(cfg.classes):
        gradients = torch.autograd.grad(
            logits[clean_rows, output_idx].sum(),
            wv,
            retain_graph=output_idx < cfg.classes - 1,
            allow_unused=False,
        )
        for layer in range(cfg.layers):
            states = wv[layer].reshape(graphs, reps, cfg.n, cfg.heads, -1)
            grads = gradients[layer].reshape(graphs, reps, cfg.n, cfg.heads, -1)[:, 0]
            delta = states[:, 0] - states[:, 1:].mean(dim=1)
            projected = (grads * delta).sum(dim=-1)
            accum[layer] += projected.square()
    score = torch.stack([value.sqrt().sum(dim=1) for value in accum], dim=1)  # [G,L,H]
    return score.detach().cpu()


def score_channel(model: Any, cfg: Config, *, mode: int, factor: str, seed: int, device: Any) -> Any:
    import torch

    clean = make_batch(cfg, cfg.score_graphs, seed, mode=mode)
    rows = []
    for start in range(0, len(clean), cfg.score_batch_size):
        stop = min(start + cfg.score_batch_size, len(clean))
        rows.append(score_channel_batch(
            model,
            cfg,
            clean.slice(start, stop),
            factor=factor,
            seed=seed + start * 97,
            device=device,
        ))
        print(f"[score {MODE_NAMES[mode]}/{factor}] {stop}/{len(clean)}", flush=True)
    return torch.cat(rows, dim=0)


def ablation_sweep(model: Any, cfg: Config, *, mode: int, seed: int, device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    clean_batch = make_batch(cfg, cfg.ablation_graphs, seed, mode=mode)
    clean = predict_in_chunks(model, clean_batch, device=device, chunk_size=cfg.analysis_batch_size)
    clean_loss = F.cross_entropy(clean, clean_batch.y.long(), reduction="none")
    func = torch.zeros(cfg.layers, cfg.heads, len(clean_batch))
    loss = torch.zeros_like(func)
    for layer in range(cfg.layers):
        for head in range(cfg.heads):
            ablated = predict_in_chunks(
                model,
                clean_batch,
                device=device,
                chunk_size=cfg.analysis_batch_size,
                ablation=(layer, head),
            )
            func[layer, head] = torch.linalg.vector_norm(ablated - clean, dim=-1)
            loss[layer, head] = F.cross_entropy(ablated, clean_batch.y.long(), reduction="none") - clean_loss
        print(f"[ablation {MODE_NAMES[mode]}] layer {layer + 1}/{cfg.layers}", flush=True)
    return {
        "functional": func,
        "loss": loss,
        "clean_logits": clean,
        "y": clean_batch.y,
        "clean_accuracy": float((clean.argmax(-1) == clean_batch.y).float().mean()),
    }


def family_ablation_sweep(
    model: Any,
    cfg: Config,
    *,
    groups: Mapping[str, Sequence[tuple[int, int]]],
    seed: int,
    device: Any,
    revision: str = FAMILY_ABLATION_REVISION,
) -> dict[str, Any]:
    """Cumulatively zero score-selected head families on independent clean graphs.

    Orders are frozen by the intervention scores.  No ablation outcome is used to choose or
    reorder heads, so every prefix remains an out-of-sample causal test of the score ranking.
    """
    import torch
    import torch.nn.functional as F

    output: dict[str, Any] = {"revision": str(revision), "tasks": {}}
    for mode, task_name in MODE_NAMES.items():
        batch = make_batch(cfg, cfg.ablation_graphs, seed + mode * 10_000, mode=mode)
        clean = predict_in_chunks(model, batch, device=device, chunk_size=cfg.analysis_batch_size)
        clean_loss = F.cross_entropy(clean, batch.y.long(), reduction="none")
        clean_correct = clean.argmax(-1) == batch.y
        task: dict[str, Any] = {
            "clean_logits": clean,
            "y": batch.y,
            "clean_accuracy": float(clean_correct.float().mean()),
            "families": {},
        }
        for family_name, group in groups.items():
            order = [tuple(map(int, head)) for head in group]
            functional = torch.zeros(len(order) + 1, len(batch))
            loss = torch.zeros_like(functional)
            accuracy_drop = torch.zeros_like(functional)
            for prefix in range(1, len(order) + 1):
                ablated = predict_in_chunks(
                    model,
                    batch,
                    device=device,
                    chunk_size=cfg.analysis_batch_size,
                    ablations=order[:prefix],
                )
                functional[prefix] = torch.linalg.vector_norm(ablated - clean, dim=-1)
                loss[prefix] = F.cross_entropy(ablated, batch.y.long(), reduction="none") - clean_loss
                accuracy_drop[prefix] = clean_correct.float() - (ablated.argmax(-1) == batch.y).float()
            task["families"][family_name] = {
                "head_order": [list(head) for head in order],
                "prefix_size": torch.arange(len(order) + 1),
                "functional": functional,
                "loss": loss,
                "accuracy_drop": accuracy_drop,
            }
            print(
                f"[family ablation {task_name}/{family_name}] "
                f"0..{len(order)} score-ranked heads",
                flush=True,
            )
        output["tasks"][task_name] = task
    return output


def rescue_sweep(model: Any, cfg: Config, *, mode: int, factor: str, seed: int, device: Any) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    clean_all = make_batch(cfg, cfg.rescue_graphs, seed, mode=mode)
    corrupt_all = make_single_corruption(cfg, clean_all, factor=factor, seed=seed + 1)
    mediation = torch.full((cfg.layers, cfg.heads, len(clean_all)), float("nan"))
    loss_recovery = torch.full_like(mediation, float("nan"))
    effect_norm = torch.zeros(len(clean_all))
    clean_logits_all = []
    corrupt_logits_all = []

    for start in range(0, len(clean_all), cfg.analysis_batch_size):
        stop = min(start + cfg.analysis_batch_size, len(clean_all))
        clean = clean_all.slice(start, stop).to(device)
        corrupt = corrupt_all.slice(start, stop).to(device)
        clean_result = capture_forward(model, clean, want_grad=False)
        clean_logits = clean_result["logits"].detach()
        clean_wv = [value.detach() for value in clean_result["wV"]]
        with torch.no_grad():
            corrupt_logits = model(corrupt).detach()
        total = clean_logits - corrupt_logits
        denom = total.square().sum(dim=-1)
        effect_norm[start:stop] = denom.sqrt().cpu()
        clean_loss = F.cross_entropy(clean_logits, clean.y.long(), reduction="none")
        corrupt_loss = F.cross_entropy(corrupt_logits, clean.y.long(), reduction="none")
        loss_denom = corrupt_loss - clean_loss
        clean_logits_all.append(clean_logits.cpu())
        corrupt_logits_all.append(corrupt_logits.cpu())

        for layer in range(cfg.layers):
            for head in range(cfg.heads):
                with patch_head_output(model, layer, head, clean_wv[layer]), torch.no_grad():
                    patched = model(corrupt).detach()
                delta_patch = patched - corrupt_logits
                mem = (delta_patch * total).sum(dim=-1) / (denom + EPS)
                patched_loss = F.cross_entropy(patched, clean.y.long(), reduction="none")
                lr = (corrupt_loss - patched_loss) / (loss_denom + EPS)
                valid_loss = loss_denom > 1.0e-4
                lr = torch.where(valid_loss, lr, torch.full_like(lr, float("nan")))
                mediation[layer, head, start:stop] = mem.cpu()
                loss_recovery[layer, head, start:stop] = lr.cpu()
        print(f"[rescue {MODE_NAMES[mode]}/{factor}] {stop}/{len(clean_all)}", flush=True)

    return {
        "mediation": mediation,
        "loss_recovery": loss_recovery,
        "effect_norm": effect_norm,
        "clean_logits": torch.cat(clean_logits_all),
        "corrupt_logits": torch.cat(corrupt_logits_all),
        "y": clean_all.y,
        "clean_accuracy": float((torch.cat(clean_logits_all).argmax(-1) == clean_all.y).float().mean()),
        "corrupt_clean_label_accuracy": float((torch.cat(corrupt_logits_all).argmax(-1) == clean_all.y).float().mean()),
    }


def no_op_delta(model: Any, cfg: Config, *, mode: int, factor: str, seed: int, device: Any) -> float:
    clean = make_batch(cfg, 2, seed, mode=mode)
    replicas = make_replicas(cfg, clean, factor=factor, donors=1, seed=seed + 1, no_op=True).to(device)
    result = capture_forward(model, replicas, want_grad=False)
    maxima = []
    for value in result["wV"]:
        reshaped = value.reshape(2, 2, cfg.n, cfg.heads, -1)
        maxima.append(float((reshaped[:, 0] - reshaped[:, 1]).abs().max().cpu()))
    return max(maxima)


def relabel_invariance(model: Any, cfg: Config, *, seed: int, device: Any) -> float:
    import torch

    clean = make_batch(cfg, 4, seed, mode=None)
    rng = np.random.default_rng(seed + 1)
    permuted = []
    for b in range(len(clean)):
        one = clean.slice(b, b + 1)
        order = rng.permutation(cfg.n)
        inverse = np.empty(cfg.n, dtype=np.int64)
        inverse[order] = np.arange(cfg.n)
        permuted.append(SynthBatch(
            x=one.x[:, order],
            rrwp=one.rrwp[:, order][:, :, order],
            q_idx=torch.tensor([inverse[int(one.q_idx[0])]], dtype=torch.long),
            target_idx=torch.tensor([inverse[int(one.target_idx[0])]], dtype=torch.long),
            anchor_idx=torch.tensor([
                -1 if int(one.anchor_idx[0]) < 0 else inverse[int(one.anchor_idx[0])]
            ], dtype=torch.long),
            y=one.y.clone(),
            mode=one.mode.clone(),
        ))
    relabelled = concat_batches(permuted)
    with torch.no_grad():
        a = model(clean.to(device)).detach()
        b = model(relabelled.to(device)).detach()
    return float((a - b).abs().max().cpu())


def softmax_error(model: Any, cfg: Config, *, seed: int, device: Any) -> float:
    import torch

    batch = make_batch(cfg, 2, seed, mode=None).to(device)
    result = capture_forward(model, batch, want_grad=False, want_attention=True)
    errors = []
    for attention in result["attention"]:
        # Fixed all-pairs ordering: [graph, src, dst, head]. Sum incoming source mass per dst.
        reshaped = attention.reshape(2, cfg.n, cfg.n, cfg.heads)
        incoming = reshaped.sum(dim=1)
        errors.append(float((incoming - 1.0).abs().max().cpu()))
    return max(errors)


def select_head_groups(semantic: np.ndarray, structural: np.ndarray, size: int) -> dict[str, list[tuple[int, int]]]:
    sem = np.asarray(semantic, dtype=float)
    st = np.asarray(structural, dtype=float)
    sem_n = sem / (np.mean(sem) + EPS)
    str_n = st / (np.mean(st) + EPS)
    total = sem_n + str_n
    eligible = total >= np.quantile(total, 0.40)
    selectivity = np.log(sem_n + 1.0e-10) - np.log(str_n + 1.0e-10)
    all_heads = [(layer, head) for layer in range(sem.shape[0]) for head in range(sem.shape[1])]
    sem_order = sorted(all_heads, key=lambda item: selectivity[item], reverse=True)
    str_order = sorted(all_heads, key=lambda item: selectivity[item])

    sem_group = [item for item in sem_order if eligible[item]][: int(size)]
    str_group = [item for item in str_order if eligible[item] and item not in sem_group][: int(size)]
    if len(str_group) < int(size):
        str_group += [item for item in str_order if item not in sem_group and item not in str_group][: int(size) - len(str_group)]
    return {"semantic": sem_group, "structural": str_group}


def select_dj_groups(
    semantic: np.ndarray, structural: np.ndarray, size: int
) -> dict[str, list[tuple[int, int]]]:
    """Four disjoint, score-only families for specialist/generalist/inert causal tests."""
    sem = np.asarray(semantic, dtype=float)
    st = np.asarray(structural, dtype=float)
    sem_n = sem / (np.mean(sem) + EPS)
    str_n = st / (np.mean(st) + EPS)
    joint, selectivity = joint_selectivity(sem_n, str_n)
    heads = [(layer, head) for layer in range(sem.shape[0]) for head in range(sem.shape[1])]
    reliable = {item for item in heads if joint[item] >= DJ_RELIABILITY_FLOOR}
    used: set[tuple[int, int]] = set()

    def take(primary: Sequence[tuple[int, int]], fallback: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
        chosen = [item for item in primary if item not in used][: int(size)]
        if len(chosen) < int(size):
            chosen += [
                item for item in fallback
                if item not in used and item not in chosen
            ][: int(size) - len(chosen)]
        used.update(chosen)
        return chosen

    sem_order = sorted(reliable, key=lambda item: (-selectivity[item], item))
    str_order = sorted(reliable, key=lambda item: (selectivity[item], item))
    general_order = sorted(reliable, key=lambda item: (abs(selectivity[item]), item))
    inert_order = sorted(heads, key=lambda item: (joint[item], item))
    all_sem = sorted(heads, key=lambda item: (-selectivity[item], item))
    all_str = sorted(heads, key=lambda item: (selectivity[item], item))
    all_general = sorted(heads, key=lambda item: (abs(selectivity[item]), item))
    return {
        "semantic_specialist": take(sem_order, all_sem),
        "structural_specialist": take(str_order, all_str),
        "high_J_generalist": take(general_order, all_general),
        "low_J_inert": take(inert_order, inert_order),
    }


def analysis_path(run_dir: Path, cfg: Config, seed: int) -> Path:
    return run_dir / "analysis" / f"seed_{seed}__{analysis_fingerprint(cfg)}.pt"


def analyze_seed(
    model: Any,
    cfg: Config,
    *,
    seed: int,
    run_dir: Path,
    device: Any,
    force: bool,
) -> dict[str, Any]:
    import torch

    path = analysis_path(run_dir, cfg, seed)
    if path.exists() and not force:
        result = torch.load(path, map_location="cpu", weights_only=False)
        family = result.get("family_ablation", {})
        dj_family = result.get("dj_family_ablation", {})
        if (
            family.get("revision") == FAMILY_ABLATION_REVISION
            and dj_family.get("revision") == DJ_FAMILY_ABLATION_REVISION
        ):
            print(f"[analysis seed={seed}] loaded {path}", flush=True)
            return result
        # Backward-compatible cache enrichment: never repeat scores, single-head ablations,
        # or rescues when only a newly added score-selected family analysis is absent.
        model.eval()
        if family.get("revision") != FAMILY_ABLATION_REVISION:
            print(f"[analysis seed={seed}] adding original family ablations", flush=True)
            result["family_ablation"] = family_ablation_sweep(
                model,
                cfg,
                groups=result["selected_groups"],
                seed=1_000_000 + seed,
                device=device,
            )
        if dj_family.get("revision") != DJ_FAMILY_ABLATION_REVISION:
            print(f"[analysis seed={seed}] adding J/D quadrant family ablations", flush=True)
            dj_groups = select_dj_groups(
                np.asarray(result["semantic_score"]),
                np.asarray(result["structural_score"]),
                cfg.top_group_size,
            )
            result["dj_selected_groups"] = dj_groups
            result["dj_family_ablation"] = family_ablation_sweep(
                model,
                cfg,
                groups=dj_groups,
                seed=1_100_000 + seed,
                device=device,
                revision=DJ_FAMILY_ABLATION_REVISION,
            )
        torch.save(result, path)
        print(f"[analysis seed={seed}] updated {path}", flush=True)
        return result

    model.eval()
    checks = {
        "semantic_no_op_max": no_op_delta(model, cfg, mode=MODE_SEMANTIC, factor="semantic", seed=110_000 + seed, device=device),
        "structural_no_op_max": no_op_delta(model, cfg, mode=MODE_STRUCTURAL, factor="structural", seed=120_000 + seed, device=device),
        "full_relabel_max": relabel_invariance(model, cfg, seed=130_000 + seed, device=device),
        "softmax_max_error": softmax_error(model, cfg, seed=140_000 + seed, device=device),
    }
    if max(checks.values()) > 5.0e-4:
        raise RuntimeError(f"verification failed for seed {seed}: {checks}")
    print(f"[verify seed={seed}] {checks}", flush=True)

    semantic_pg = score_channel(
        model, cfg, mode=MODE_SEMANTIC, factor="semantic", seed=200_000 + seed, device=device
    )
    structural_pg = score_channel(
        model, cfg, mode=MODE_STRUCTURAL, factor="structural", seed=300_000 + seed, device=device
    )
    # Negative-control channels: value donor-swaps should not solve the structural task;
    # structural donor-swaps should not matter to marked-source value retrieval.
    semantic_on_structural_pg = score_channel(
        model, cfg, mode=MODE_STRUCTURAL, factor="semantic", seed=400_000 + seed, device=device
    )
    structural_on_semantic_pg = score_channel(
        model, cfg, mode=MODE_SEMANTIC, factor="structural", seed=500_000 + seed, device=device
    )
    semantic_score = semantic_pg.mean(dim=0).numpy()
    structural_score = structural_pg.mean(dim=0).numpy()
    groups = select_head_groups(semantic_score, structural_score, cfg.top_group_size)

    ablation_sem = ablation_sweep(model, cfg, mode=MODE_SEMANTIC, seed=600_000 + seed, device=device)
    ablation_str = ablation_sweep(model, cfg, mode=MODE_STRUCTURAL, seed=700_000 + seed, device=device)
    rescue_sem = rescue_sweep(
        model, cfg, mode=MODE_SEMANTIC, factor="semantic", seed=800_000 + seed, device=device
    )
    rescue_str = rescue_sweep(
        model, cfg, mode=MODE_STRUCTURAL, factor="structural", seed=900_000 + seed, device=device
    )
    family_ablation = family_ablation_sweep(
        model, cfg, groups=groups, seed=1_000_000 + seed, device=device
    )
    dj_groups = select_dj_groups(semantic_score, structural_score, cfg.top_group_size)
    dj_family_ablation = family_ablation_sweep(
        model,
        cfg,
        groups=dj_groups,
        seed=1_100_000 + seed,
        device=device,
        revision=DJ_FAMILY_ABLATION_REVISION,
    )
    result = {
        "version": EXPERIMENT_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "analysis_fingerprint": analysis_fingerprint(cfg),
        "seed": int(seed),
        "checks": checks,
        "semantic_score_per_graph": semantic_pg,
        "structural_score_per_graph": structural_pg,
        "semantic_on_structural_per_graph": semantic_on_structural_pg,
        "structural_on_semantic_per_graph": structural_on_semantic_pg,
        "semantic_score": semantic_score,
        "structural_score": structural_score,
        "selected_groups": groups,
        "dj_selected_groups": dj_groups,
        "ablation_semantic": ablation_sem,
        "ablation_structural": ablation_str,
        "rescue_semantic": rescue_sem,
        "rescue_structural": rescue_str,
        "family_ablation": family_ablation,
        "dj_family_ablation": dj_family_ablation,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    print(f"[analysis seed={seed}] cached {path}", flush=True)
    return result


# ======================================================================================
# Statistics, tables, and paper-facing figures
# ======================================================================================


def rank_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # Average exact ties.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for idx in range(len(unique)):
            positions = np.where(inverse == idx)[0]
            ranks[positions] = ranks[positions].mean()
    return ranks


def correlation(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) <= 0 or np.std(y[ok]) <= 0:
        return float("nan")
    return float(np.corrcoef(rank_values(x[ok]), rank_values(y[ok]))[0, 1])


def partial_rank_correlation(x: np.ndarray, y: np.ndarray, controls: Sequence[np.ndarray]) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    controls = [np.asarray(value, dtype=float) for value in controls]
    ok = np.isfinite(x) & np.isfinite(y)
    for control in controls:
        ok &= np.isfinite(control)
    if ok.sum() < 5:
        return float("nan")
    rx = rank_values(x[ok])
    ry = rank_values(y[ok])
    design = np.column_stack([np.ones(ok.sum())] + [rank_values(control[ok]) for control in controls])
    resid_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    resid_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if np.std(resid_x) <= 0 or np.std(resid_y) <= 0:
        return float("nan")
    return float(np.corrcoef(resid_x, resid_y)[0, 1])


def finite_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    ok = np.isfinite(arr)
    return float(arr[ok].mean()) if ok.any() else float("nan")


def group_metric(matrix: Any, heads: Sequence[tuple[int, int]]) -> float:
    values = [np.asarray(matrix[layer, head], dtype=float) for layer, head in heads]
    return finite_mean(np.concatenate(values))


def joint_selectivity(semantic: Any, structural: Any) -> tuple[np.ndarray, np.ndarray]:
    """Rotate two comparable non-negative channels into strength and bounded preference."""
    semantic = np.asarray(semantic, dtype=float)
    structural = np.asarray(structural, dtype=float)
    joint = 0.5 * (semantic + structural)
    selectivity = (semantic - structural) / (semantic + structural + EPS)
    return joint, selectivity


def dj_class(joint: float, selectivity: float) -> str:
    if float(joint) < DJ_RELIABILITY_FLOOR:
        return "low-J / inert"
    if float(selectivity) >= DJ_SELECTIVITY_THRESHOLD:
        return "semantic specialist"
    if float(selectivity) <= -DJ_SELECTIVITY_THRESHOLD:
        return "structural specialist"
    return "high-J generalist"


def build_head_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        seed = int(result["seed"])
        sem = np.asarray(result["semantic_score"])
        st = np.asarray(result["structural_score"])
        sem_func = np.asarray(result["ablation_semantic"]["functional"]).mean(axis=-1)
        str_func = np.asarray(result["ablation_structural"]["functional"]).mean(axis=-1)
        sem_loss = np.asarray(result["ablation_semantic"]["loss"]).mean(axis=-1)
        str_loss = np.asarray(result["ablation_structural"]["loss"]).mean(axis=-1)
        sem_rescue = np.nanmean(np.asarray(result["rescue_semantic"]["mediation"]), axis=-1)
        str_rescue = np.nanmean(np.asarray(result["rescue_structural"]["mediation"]), axis=-1)
        sem_cross = np.asarray(result["semantic_on_structural_per_graph"], dtype=float).mean(axis=0)
        str_cross = np.asarray(result["structural_on_semantic_per_graph"], dtype=float).mean(axis=0)
        sem_norm = sem / (sem.mean() + EPS)
        str_norm = st / (st.mean() + EPS)
        joint, selectivity = joint_selectivity(sem_norm, str_norm)
        sem_func_norm = sem_func / (sem_func.mean() + EPS)
        str_func_norm = str_func / (str_func.mean() + EPS)
        impact_joint, impact_selectivity = joint_selectivity(sem_func_norm, str_func_norm)
        sem_set = set(map(tuple, result["selected_groups"]["semantic"]))
        str_set = set(map(tuple, result["selected_groups"]["structural"]))
        for layer in range(sem.shape[0]):
            for head in range(sem.shape[1]):
                group = "semantic" if (layer, head) in sem_set else "structural" if (layer, head) in str_set else "other"
                rows.append({
                    "seed": seed,
                    "layer": layer,
                    "head": head,
                    "selected_group": group,
                    "semantic_score": float(sem[layer, head]),
                    "structural_score": float(st[layer, head]),
                    "semantic_score_norm": float(sem_norm[layer, head]),
                    "structural_score_norm": float(str_norm[layer, head]),
                    "joint_score_J": float(joint[layer, head]),
                    "selectivity_D": float(selectivity[layer, head]),
                    "absolute_selectivity": float(abs(selectivity[layer, head])),
                    "selectivity_reliable": bool(joint[layer, head] >= DJ_RELIABILITY_FLOOR),
                    "DJ_class": dj_class(joint[layer, head], selectivity[layer, head]),
                    "semantic_cross_task_score": float(sem_cross[layer, head]),
                    "structural_cross_task_score": float(str_cross[layer, head]),
                    "semantic_matched_enrichment": float(
                        (sem[layer, head] - sem_cross[layer, head])
                        / (sem[layer, head] + sem_cross[layer, head] + EPS)
                    ),
                    "structural_matched_enrichment": float(
                        (st[layer, head] - str_cross[layer, head])
                        / (st[layer, head] + str_cross[layer, head] + EPS)
                    ),
                    "semantic_ablation_functional": float(sem_func[layer, head]),
                    "structural_ablation_functional": float(str_func[layer, head]),
                    "semantic_ablation_functional_norm": float(sem_func_norm[layer, head]),
                    "structural_ablation_functional_norm": float(str_func_norm[layer, head]),
                    "ablation_joint_impact": float(impact_joint[layer, head]),
                    "ablation_role_selectivity": float(impact_selectivity[layer, head]),
                    "semantic_ablation_loss": float(sem_loss[layer, head]),
                    "structural_ablation_loss": float(str_loss[layer, head]),
                    "semantic_rescue_mem": float(sem_rescue[layer, head]),
                    "structural_rescue_mem": float(str_rescue[layer, head]),
                    "rescue_joint_mem": float(0.5 * (sem_rescue[layer, head] + str_rescue[layer, head])),
                    "rescue_role_contrast": float(sem_rescue[layer, head] - str_rescue[layer, head]),
                })
    return rows


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
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    print(f"[figure] {png}\n[figure] {pdf}", flush=True)
    return [str(png), str(pdf)]


def layer_palette(layers: int) -> list[Any]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    return [cmap((layer + 0.5) / max(layers, 1)) for layer in range(layers)]


def figure_specialisation_plane(results: Sequence[dict[str, Any]], cfg: Config, figures_dir: Path) -> list[str]:
    plt = configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    colors = layer_palette(cfg.layers)
    markers = ["o", "s", "^", "D", "P"]
    all_x, all_y = [], []
    for result_idx, result in enumerate(results):
        sem = np.asarray(result["semantic_score"], dtype=float)
        st = np.asarray(result["structural_score"], dtype=float)
        x = st / (st.mean() + EPS)
        y = sem / (sem.mean() + EPS)
        all_x.extend(x.ravel())
        all_y.extend(y.ravel())
        selected_sem = set(map(tuple, result["selected_groups"]["semantic"]))
        selected_str = set(map(tuple, result["selected_groups"]["structural"]))
        for layer in range(cfg.layers):
            for head in range(cfg.heads):
                edge = "#c43c39" if (layer, head) in selected_sem else "#2878b5" if (layer, head) in selected_str else "white"
                width = 1.7 if (layer, head) in selected_sem | selected_str else 0.55
                ax.scatter(
                    x[layer, head],
                    y[layer, head],
                    s=64,
                    marker=markers[result_idx % len(markers)],
                    color=colors[layer],
                    edgecolor=edge,
                    linewidth=width,
                    alpha=0.90,
                    zorder=3,
                )
    lo = max(min(all_x + all_y) * 0.75, 1.0e-3)
    hi = max(all_x + all_y) * 1.30
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#777777", linewidth=1.1, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Structural score  $S_{str}/\overline{S}_{str}$")
    ax.set_ylabel(r"Semantic score  $S_{sem}/\overline{S}_{sem}$")
    fig.suptitle(
        "Intervention-defined GRIT head specialisation",
        x=0.075,
        y=0.99,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.935,
        "Planted-source functional transport; red/blue rims mark preregistered head families",
        fontsize=9,
        color="#555555",
    )
    ax.grid(True, which="both", linewidth=0.5, alpha=0.22)
    from matplotlib.lines import Line2D

    layer_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[layer], markeredgecolor="none", label=f"Layer {layer}") for layer in range(cfg.layers)]
    seed_handles = [Line2D([0], [0], marker=markers[i % len(markers)], color="#555555", linestyle="none", markerfacecolor="none", label=f"Seed {result['seed']}") for i, result in enumerate(results)]
    first = ax.legend(handles=layer_handles, title="Depth", frameon=False, loc="upper left")
    ax.add_artist(first)
    ax.legend(handles=seed_handles, title="Training seed", frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save_figure(fig, figures_dir / "fig1_specialisation_plane")


def figure_score_ablation(rows: Sequence[Mapping[str, Any]], cfg: Config, figures_dir: Path) -> tuple[list[str], dict[str, Any]]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=False)
    colors = layer_palette(cfg.layers)
    markers = ["o", "s", "^", "D", "P"]
    specs = [
        ("semantic", "semantic_score_norm", "semantic_ablation_functional", "Semantic donor-swap score", "Semantic-task ablation impact"),
        ("structural", "structural_score_norm", "structural_ablation_functional", "Structural donor-swap score", "Structural-task ablation impact"),
    ]
    summary: dict[str, Any] = {}
    for ax, (name, x_key, y_key, x_label, y_label) in zip(axes, specs):
        x = np.asarray([float(row[x_key]) for row in rows])
        y_raw = np.asarray([float(row[y_key]) for row in rows])
        seeds = np.asarray([int(row["seed"]) for row in rows])
        layers = np.asarray([int(row["layer"]) for row in rows])
        # Within-seed impact normalisation removes seed-level calibration without changing ranks.
        y = y_raw.copy()
        for seed in np.unique(seeds):
            mask = seeds == seed
            y[mask] /= y_raw[mask].mean() + EPS
        other = np.asarray([
            float(row["structural_score_norm"] if name == "semantic" else row["semantic_score_norm"])
            for row in rows
        ])
        rho = correlation(x, y)
        partial = partial_rank_correlation(x, y, [other, layers.astype(float)])
        per_seed: dict[str, dict[str, float]] = {}
        for seed in sorted(np.unique(seeds)):
            mask = seeds == seed
            per_seed[str(int(seed))] = {
                "spearman": correlation(x[mask], y[mask]),
                "partial_controlling_other_score_and_layer": partial_rank_correlation(
                    x[mask], y[mask], [other[mask], layers[mask].astype(float)]
                ),
            }
        seed_rho = np.asarray([value["spearman"] for value in per_seed.values()], dtype=float)
        finite_seed_rho = seed_rho[np.isfinite(seed_rho)]
        seed_median = float(np.median(finite_seed_rho)) if len(finite_seed_rho) else float("nan")
        seed_range = (
            [float(finite_seed_rho.min()), float(finite_seed_rho.max())]
            if len(finite_seed_rho)
            else [float("nan"), float("nan")]
        )
        summary[name] = {
            "pooled_spearman": rho,
            "pooled_partial_controlling_other_score_and_layer": partial,
            "per_seed": per_seed,
            "seed_median_spearman": seed_median,
            "seed_range_spearman": seed_range,
        }
        for layer in range(cfg.layers):
            for seed_idx, seed in enumerate(sorted(np.unique(seeds))):
                mask = (layers == layer) & (seeds == seed)
                ax.scatter(
                    np.maximum(x[mask], 1.0e-10),
                    np.maximum(y[mask], 1.0e-10),
                    s=48,
                    marker=markers[seed_idx % len(markers)],
                    color=colors[layer],
                    alpha=0.82,
                    edgecolor="white",
                    linewidth=0.55,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_label + " (within-seed normalised)")
        ax.set_ylabel(y_label + " (within-seed normalised)")
        ax.set_title(name.capitalize(), loc="left", fontweight="bold")
        ax.text(
            0.04,
            0.96,
            f"pooled ρ = {rho:.2f}\n"
            f"seed median ρ = {seed_median:.2f}  [{seed_range[0]:.2f}, {seed_range[1]:.2f}]\n"
            f"pooled partial ρ = {partial:.2f}",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
        )
        ax.grid(True, which="both", linewidth=0.5, alpha=0.22)
    fig.suptitle(
        "Do intervention scores predict clean-input head necessity?",
        x=0.06,
        y=0.99,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.91,
        "Independent score/ablation graph sets; partial controls the other score and depth",
        fontsize=9,
        color="#555555",
    )
    from matplotlib.lines import Line2D

    layer_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[layer], markeredgecolor="none", label=f"Layer {layer}")
        for layer in range(cfg.layers)
    ]
    unique_seeds = sorted({int(row["seed"]) for row in rows})
    seed_handles = [
        Line2D([0], [0], marker=markers[idx % len(markers)], color="#555555", linestyle="none", markerfacecolor="none", label=f"Seed {seed}")
        for idx, seed in enumerate(unique_seeds)
    ]
    fig.legend(
        handles=layer_handles + seed_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=max(1, len(layer_handles) + len(seed_handles)),
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.84))
    return save_figure(fig, figures_dir / "fig2_score_ablation"), summary


def correlation_by_seed(
    rows: Sequence[Mapping[str, Any]],
    x_key: str,
    y_key: str,
    *,
    reliable_only: bool = False,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if not reliable_only or bool(row["selectivity_reliable"])
    ]
    per_seed: dict[str, float] = {}
    for seed in sorted({int(row["seed"]) for row in selected}):
        cell = [row for row in selected if int(row["seed"]) == seed]
        per_seed[str(seed)] = correlation(
            [float(row[x_key]) for row in cell],
            [float(row[y_key]) for row in cell],
        )
    finite = np.asarray([value for value in per_seed.values() if np.isfinite(value)], dtype=float)
    return {
        "pooled_spearman": correlation(
            [float(row[x_key]) for row in selected],
            [float(row[y_key]) for row in selected],
        ),
        "seed_median_spearman": float(np.median(finite)) if len(finite) else float("nan"),
        "seed_range_spearman": (
            [float(finite.min()), float(finite.max())]
            if len(finite) else [float("nan"), float("nan")]
        ),
        "per_seed": per_seed,
        "heads": len(selected),
        "reliable_only": bool(reliable_only),
    }


def dj_quadrant_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    fields = (
        "joint_score_J",
        "selectivity_D",
        "ablation_joint_impact",
        "ablation_role_selectivity",
        "semantic_ablation_functional_norm",
        "structural_ablation_functional_norm",
        "semantic_rescue_mem",
        "structural_rescue_mem",
        "rescue_role_contrast",
    )
    order = ("semantic specialist", "high-J generalist", "structural specialist", "low-J / inert")
    for seed in sorted({int(row["seed"]) for row in rows}):
        for category in order:
            selected = [
                row for row in rows
                if int(row["seed"]) == seed and row["DJ_class"] == category
            ]
            record: dict[str, Any] = {
                "seed": seed,
                "DJ_class": category,
                "heads": len(selected),
            }
            for field in fields:
                record[field] = (
                    finite_mean([float(row[field]) for row in selected])
                    if selected else float("nan")
                )
            output.append(record)
    return output


def _dj_scatter(
    ax: Any,
    rows: Sequence[Mapping[str, Any]],
    cfg: Config,
    x_key: str,
    y_key: str,
    *,
    reliable_only: bool = False,
) -> None:
    colors = layer_palette(cfg.layers)
    markers = ["o", "s", "^", "D", "P"]
    seeds = sorted({int(row["seed"]) for row in rows})
    for row in rows:
        reliable = bool(row["selectivity_reliable"])
        if reliable_only and not reliable:
            continue
        seed_index = seeds.index(int(row["seed"]))
        ax.scatter(
            float(row[x_key]),
            float(row[y_key]),
            s=48,
            marker=markers[seed_index % len(markers)],
            color=colors[int(row["layer"])],
            alpha=0.82 if reliable else 0.16,
            edgecolor="white" if reliable else "none",
            linewidth=0.5,
            zorder=3,
        )


def figure_joint_selectivity(
    rows: Sequence[Mapping[str, Any]], cfg: Config, figures_dir: Path
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    plt = configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
    seeds = sorted({int(row["seed"]) for row in rows})
    summary: dict[str, Any] = {
        "score_calibration": "each channel divided by its within-seed head mean",
        "J_formula": "0.5 * (semantic_score_norm + structural_score_norm)",
        "D_formula": "(semantic_score_norm - structural_score_norm) / (semantic_score_norm + structural_score_norm)",
        "J_reliability_floor": DJ_RELIABILITY_FLOOR,
        "D_specialist_threshold": DJ_SELECTIVITY_THRESHOLD,
    }

    # A: interpretable rotation of the original score plane.
    ax = axes[0, 0]
    _dj_scatter(ax, rows, cfg, "selectivity_D", "joint_score_J")
    ax.axvspan(-DJ_SELECTIVITY_THRESHOLD, DJ_SELECTIVITY_THRESHOLD, color="#eeeeee", alpha=0.65, zorder=0)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.axhline(DJ_RELIABILITY_FLOOR, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_xlim(-1.04, 1.04)
    ax.set_xlabel(r"Selectivity $D_{rel}$  (structural $\leftarrow$ 0 $\rightarrow$ semantic)")
    ax.set_ylabel(r"Joint sensitivity $J$")
    ax.set_title("A  Strength and factor preference", loc="left", fontweight="bold")

    # B: J should explain task-general causal influence.
    ax = axes[0, 1]
    _dj_scatter(ax, rows, cfg, "joint_score_J", "ablation_joint_impact")
    joint_stats = correlation_by_seed(rows, "joint_score_J", "ablation_joint_impact")
    summary["J_vs_joint_ablation"] = joint_stats
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Joint sensitivity $J$")
    ax.set_ylabel("Mean cross-task functional ablation impact")
    ax.set_title("B  Does $J$ predict how much a head matters?", loc="left", fontweight="bold")
    ax.text(
        0.04, 0.96,
        f"pooled $\\rho={joint_stats['pooled_spearman']:.2f}$\n"
        f"seed median $\\rho={joint_stats['seed_median_spearman']:.2f}$",
        transform=ax.transAxes, va="top",
    )

    # C: D is useful only if it predicts which task is affected, not merely score geometry.
    ax = axes[1, 0]
    _dj_scatter(ax, rows, cfg, "selectivity_D", "ablation_role_selectivity", reliable_only=True)
    role_stats = correlation_by_seed(
        rows, "selectivity_D", "ablation_role_selectivity", reliable_only=True
    )
    summary["D_vs_ablation_role"] = role_stats
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.set_xlabel(r"Score selectivity $D_{rel}$")
    ax.set_ylabel("Ablation role contrast")
    ax.set_title("C  Does $D$ predict which task a head serves?", loc="left", fontweight="bold")
    ax.text(
        0.04, 0.96,
        f"reliable heads only\npooled $\\rho={role_stats['pooled_spearman']:.2f}$\n"
        f"seed median $\\rho={role_stats['seed_median_spearman']:.2f}$",
        transform=ax.transAxes, va="top",
    )

    # D: an independent intervention-patching check of the same signed role prediction.
    ax = axes[1, 1]
    _dj_scatter(ax, rows, cfg, "selectivity_D", "rescue_role_contrast", reliable_only=True)
    rescue_stats = correlation_by_seed(
        rows, "selectivity_D", "rescue_role_contrast", reliable_only=True
    )
    summary["D_vs_rescue_role"] = rescue_stats
    ax.axhline(0.0, color="#777777", linewidth=0.8)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlim(-1.04, 1.04)
    ax.set_xlabel(r"Score selectivity $D_{rel}$")
    ax.set_ylabel("Semantic − structural rescue mediation")
    ax.set_title("D  Does $D$ predict causal rescue role?", loc="left", fontweight="bold")
    ax.text(
        0.04, 0.96,
        f"reliable heads only\npooled $\\rho={rescue_stats['pooled_spearman']:.2f}$\n"
        f"seed median $\\rho={rescue_stats['seed_median_spearman']:.2f}$",
        transform=ax.transAxes, va="top",
    )

    for index, ax in enumerate(axes.ravel()):
        ax.grid(True, which="both", linewidth=0.5, alpha=0.20)
    from matplotlib.lines import Line2D

    colors = layer_palette(cfg.layers)
    markers = ["o", "s", "^", "D", "P"]
    layer_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[layer], markeredgecolor="none", label=f"Layer {layer}")
        for layer in range(cfg.layers)
    ]
    seed_handles = [
        Line2D([0], [0], marker=markers[index % len(markers)], color="#555555", linestyle="none", markerfacecolor="none", label=f"Seed {seed}")
        for index, seed in enumerate(seeds)
    ]
    fig.legend(
        layer_handles + seed_handles,
        [handle.get_label() for handle in layer_handles + seed_handles],
        frameon=False, loc="lower center", ncol=len(layer_handles) + len(seed_handles),
        bbox_to_anchor=(0.5, 0.004),
    )
    fig.suptitle(
        "Joint influence and semantic–structural role separate head strength from preference",
        x=0.045, y=0.995, ha="left", fontsize=15, fontweight="bold",
    )
    fig.text(
        0.045, 0.958,
        "Scores and functional ablations are channel-mean calibrated within seed; faded heads have J < 0.5 and do not support selectivity claims.",
        fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.925), h_pad=2.2, w_pad=2.0)
    quadrant_rows = dj_quadrant_rows(rows)
    summary["quadrants_by_seed"] = quadrant_rows
    return save_figure(fig, figures_dir / "fig5_joint_influence_selectivity"), summary, quadrant_rows


def draw_heatmap(ax: Any, matrix: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], title: str, cmap: str, center: float | None = None) -> None:
    import matplotlib.colors as colors

    if center is None:
        image = ax.imshow(matrix, cmap=cmap, aspect="auto")
    else:
        vmax = max(abs(float(np.nanmin(matrix)) - center), abs(float(np.nanmax(matrix)) - center), 1.0e-6)
        norm = colors.TwoSlopeNorm(vmin=center - vmax, vcenter=center, vmax=center + vmax)
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title, loc="left", fontweight="bold")
    threshold = float(np.nanmedian(matrix))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center", color="white" if value > threshold else "black", fontweight="bold")
    ax.figure.colorbar(image, ax=ax, fraction=0.047, pad=0.04)


def double_dissociation_values(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    necessity_seeds = []
    rescue_seeds = []
    selected = []
    for result in results:
        groups = result["selected_groups"]
        sem_heads = [tuple(item) for item in groups["semantic"]]
        str_heads = [tuple(item) for item in groups["structural"]]
        if result.get("family_ablation", {}).get("revision") != FAMILY_ABLATION_REVISION:
            raise RuntimeError(
                "family-ablation cache is missing; run --phase analyze once before regenerating figures"
            )
        family_tasks = result["family_ablation"]["tasks"]
        necessity = np.asarray([
            [
                finite_mean(family_tasks["semantic"]["families"]["semantic"]["loss"][-1]),
                finite_mean(family_tasks["structural"]["families"]["semantic"]["loss"][-1]),
            ],
            [
                finite_mean(family_tasks["semantic"]["families"]["structural"]["loss"][-1]),
                finite_mean(family_tasks["structural"]["families"]["structural"]["loss"][-1]),
            ],
        ])
        rescue = np.asarray([
            [
                group_metric(result["rescue_semantic"]["mediation"], sem_heads),
                group_metric(result["rescue_structural"]["mediation"], sem_heads),
            ],
            [
                group_metric(result["rescue_semantic"]["mediation"], str_heads),
                group_metric(result["rescue_structural"]["mediation"], str_heads),
            ],
        ])
        necessity_seeds.append(necessity)
        rescue_seeds.append(rescue)
        selected.append({"seed": int(result["seed"]), "semantic": sem_heads, "structural": str_heads})
    necessity_seeds = np.stack(necessity_seeds)
    rescue_seeds = np.stack(rescue_seeds)
    necessity_interaction = (necessity_seeds[:, 0, 0] - necessity_seeds[:, 0, 1]) + (necessity_seeds[:, 1, 1] - necessity_seeds[:, 1, 0])
    rescue_interaction = (rescue_seeds[:, 0, 0] - rescue_seeds[:, 0, 1]) + (rescue_seeds[:, 1, 1] - rescue_seeds[:, 1, 0])
    return {
        "necessity_by_seed": necessity_seeds,
        "rescue_by_seed": rescue_seeds,
        "necessity_mean": np.nanmean(necessity_seeds, axis=0),
        "rescue_mean": np.nanmean(rescue_seeds, axis=0),
        "necessity_interaction": necessity_interaction,
        "rescue_interaction": rescue_interaction,
        "selected_heads": selected,
    }


def interaction_panel(ax: Any, necessity: np.ndarray, rescue: np.ndarray, seeds: Sequence[int]) -> None:
    means = [float(np.nanmean(necessity)), float(np.nanmean(rescue))]
    if len(seeds) > 1:
        errors = [float(np.nanstd(necessity, ddof=1) / math.sqrt(len(seeds))), float(np.nanstd(rescue, ddof=1) / math.sqrt(len(seeds)))]
    else:
        errors = [0.0, 0.0]
    x = np.arange(2)
    ax.bar(x, means, yerr=np.asarray(errors) * 1.96, color=["#6a51a3", "#238b45"], alpha=0.84, capsize=5, width=0.64)
    ax.axhline(0.0, color="#777777", linewidth=0.9)
    ax.set_xticks(x, ["Necessity", "Rescue"])
    ax.set_ylabel("Double-dissociation interaction")
    ax.set_title("Channel-specific causal interaction", loc="left", fontweight="bold")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.22)


def negative_control_panel(ax: Any, results: Sequence[dict[str, Any]]) -> None:
    matched_sem, cross_sem, matched_str, cross_str = [], [], [], []
    for result in results:
        matched_sem.append(float(np.asarray(result["semantic_score_per_graph"]).mean()))
        cross_sem.append(float(np.asarray(result["semantic_on_structural_per_graph"]).mean()))
        matched_str.append(float(np.asarray(result["structural_score_per_graph"]).mean()))
        cross_str.append(float(np.asarray(result["structural_on_semantic_per_graph"]).mean()))
    values = [matched_sem, cross_sem, matched_str, cross_str]
    means = [np.mean(value) for value in values]
    means_norm = [means[0] / (means[0] + EPS), means[1] / (means[0] + EPS), means[2] / (means[2] + EPS), means[3] / (means[2] + EPS)]
    x = np.arange(4)
    ax.bar(x, means_norm, color=["#cb181d", "#fcae91", "#2171b5", "#9ecae1"], width=0.70)
    ax.set_xticks(x, ["Sem→sem", "Sem→str", "Str→str", "Str→sem"], rotation=18)
    ax.set_ylabel("Mean score / matched-channel mean")
    ax.set_title("Intervention negative controls", loc="left", fontweight="bold")
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.22)


def figure_double_dissociation(results: Sequence[dict[str, Any]], cfg: Config, figures_dir: Path) -> tuple[list[str], dict[str, Any]]:
    plt = configure_matplotlib()
    values = double_dissociation_values(results)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0))
    rows = ["Semantic-score heads", "Structural-score heads"]
    draw_heatmap(
        axes[0, 0],
        values["necessity_mean"],
        rows,
        ["Semantic task", "Structural task"],
        "A  Necessity: ΔCE under simultaneous family ablation",
        "Purples",
        center=0.0,
    )
    draw_heatmap(
        axes[0, 1],
        values["rescue_mean"],
        rows,
        ["Semantic corruption", "Structural corruption"],
        "B  Rescue: clean→corrupt mediated-effect fraction",
        "Greens",
        center=0.0,
    )
    interaction_panel(
        axes[1, 0],
        np.asarray(values["necessity_interaction"]),
        np.asarray(values["rescue_interaction"]),
        [int(result["seed"]) for result in results],
    )
    negative_control_panel(axes[1, 1], results)
    fig.suptitle("Necessity plus rescue: a causal semantic/structural double dissociation", x=0.055, ha="left", fontsize=15, fontweight="bold")
    fig.text(
        0.055,
        0.945,
        "Necessity jointly ablates each selected family; rescue patches one clean routed head at a time into the corrupted run.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.5, w_pad=2.0)
    return save_figure(fig, figures_dir / "fig3_necessity_rescue_double_dissociation"), values


def iterative_family_ablation_values(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for task_name in ("semantic", "structural"):
        values[task_name] = {}
        for family_name in ("semantic", "structural"):
            loss_curves = []
            accuracy_curves = []
            functional_curves = []
            head_orders = []
            for result in results:
                family = result["family_ablation"]["tasks"][task_name]["families"][family_name]
                loss_curves.append(np.asarray(family["loss"], dtype=float).mean(axis=-1))
                accuracy_curves.append(np.asarray(family["accuracy_drop"], dtype=float).mean(axis=-1))
                functional_curves.append(np.asarray(family["functional"], dtype=float).mean(axis=-1))
                head_orders.append(family["head_order"])
            values[task_name][family_name] = {
                "loss_by_seed": np.stack(loss_curves),
                "accuracy_drop_by_seed": np.stack(accuracy_curves),
                "functional_by_seed": np.stack(functional_curves),
                "head_orders": head_orders,
            }
    return values


def figure_iterative_family_ablation(
    results: Sequence[dict[str, Any]], cfg: Config, figures_dir: Path
) -> tuple[list[str], dict[str, Any]]:
    plt = configure_matplotlib()
    values = iterative_family_ablation_values(results)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8), sharex=True)
    colors = {"semantic": "#cb181d", "structural": "#2171b5"}
    labels = {"semantic": "Semantic-score family", "structural": "Structural-score family"}
    metrics = (("loss_by_seed", "Δ cross-entropy"), ("accuracy_drop_by_seed", "Accuracy drop (percentage points)"))
    seeds = [int(result["seed"]) for result in results]

    for col, task_name in enumerate(("semantic", "structural")):
        for row, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            for family_name in ("semantic", "structural"):
                curves = np.asarray(values[task_name][family_name][metric], dtype=float)
                if metric == "accuracy_drop_by_seed":
                    curves = curves * 100.0
                x = np.arange(curves.shape[1])
                for seed_idx, curve in enumerate(curves):
                    ax.plot(x, curve, color=colors[family_name], alpha=0.20, linewidth=1.0)
                    ax.scatter(x[-1], curve[-1], color=colors[family_name], alpha=0.35, s=18)
                mean = np.nanmean(curves, axis=0)
                if curves.shape[0] > 1:
                    error = np.nanstd(curves, axis=0, ddof=1) / math.sqrt(curves.shape[0]) * 1.96
                else:
                    error = np.zeros_like(mean)
                ax.plot(x, mean, color=colors[family_name], linewidth=2.6, marker="o", markersize=5, label=labels[family_name])
                ax.fill_between(x, mean - error, mean + error, color=colors[family_name], alpha=0.12, linewidth=0)
            ax.axhline(0.0, color="#777777", linewidth=0.8)
            ax.set_xticks(np.arange(cfg.top_group_size + 1))
            ax.grid(True, linewidth=0.5, alpha=0.22)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f"{chr(65 + col)}  {task_name.capitalize()} task", loc="left", fontweight="bold")
            if row == 1:
                ax.set_xlabel("Number of score-ranked family heads jointly ablated")

    fig.suptitle(
        "Does necessity emerge when specialised heads are removed cumulatively?",
        x=0.06,
        y=0.99,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.945,
        "Thin lines are training seeds; thick lines are means and bands are 95% normal-approximation intervals.",
        fontsize=9,
        color="#555555",
    )
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.06, 1, 0.91), h_pad=2.0, w_pad=2.0)
    summary = {
        task: {
            family: {
                "loss_by_seed": values[task][family]["loss_by_seed"],
                "accuracy_drop_by_seed": values[task][family]["accuracy_drop_by_seed"],
                "functional_by_seed": values[task][family]["functional_by_seed"],
                "head_orders": values[task][family]["head_orders"],
            }
            for family in ("semantic", "structural")
        }
        for task in ("semantic", "structural")
    }
    return save_figure(fig, figures_dir / "fig4_iterative_family_ablation"), summary


def dj_family_ablation_values(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    family_names = (
        "semantic_specialist",
        "structural_specialist",
        "high_J_generalist",
        "low_J_inert",
    )
    values: dict[str, Any] = {}
    for result in results:
        if result.get("dj_family_ablation", {}).get("revision") != DJ_FAMILY_ABLATION_REVISION:
            raise RuntimeError(
                "J/D family-ablation cache is missing; run --phase analyze once before figures"
            )
    for task_name in ("semantic", "structural"):
        values[task_name] = {}
        for family_name in family_names:
            functional, loss, accuracy, orders = [], [], [], []
            for result in results:
                family = result["dj_family_ablation"]["tasks"][task_name]["families"][family_name]
                functional.append(np.asarray(family["functional"], dtype=float).mean(axis=-1))
                loss.append(np.asarray(family["loss"], dtype=float).mean(axis=-1))
                accuracy.append(np.asarray(family["accuracy_drop"], dtype=float).mean(axis=-1))
                orders.append(family["head_order"])
            values[task_name][family_name] = {
                "functional_by_seed": np.stack(functional),
                "loss_by_seed": np.stack(loss),
                "accuracy_drop_by_seed": np.stack(accuracy),
                "head_orders": orders,
            }
    return values


def figure_dj_family_ablation(
    results: Sequence[dict[str, Any]], cfg: Config, figures_dir: Path
) -> tuple[list[str], dict[str, Any]]:
    plt = configure_matplotlib()
    values = dj_family_ablation_values(results)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True)
    styles = {
        "semantic_specialist": ("#cb181d", "Semantic specialists"),
        "structural_specialist": ("#2171b5", "Structural specialists"),
        "high_J_generalist": ("#6a51a3", "High-J generalists"),
        "low_J_inert": ("#969696", "Low-J / inert"),
    }
    metrics = (
        ("functional_by_seed", "Functional logit impact", 1.0),
        ("accuracy_drop_by_seed", "Accuracy drop (percentage points)", 100.0),
    )
    max_prefix = 0
    for col, task_name in enumerate(("semantic", "structural")):
        for row_index, (metric, ylabel, scale) in enumerate(metrics):
            ax = axes[row_index, col]
            for family_name, (color, label) in styles.items():
                curves = np.asarray(values[task_name][family_name][metric], dtype=float) * scale
                x = np.arange(curves.shape[1])
                max_prefix = max(max_prefix, int(x[-1]))
                for curve in curves:
                    ax.plot(x, curve, color=color, alpha=0.18, linewidth=1.0)
                mean = np.nanmean(curves, axis=0)
                error = (
                    np.nanstd(curves, axis=0, ddof=1) / math.sqrt(curves.shape[0]) * 1.96
                    if curves.shape[0] > 1 else np.zeros_like(mean)
                )
                ax.plot(x, mean, color=color, marker="o", linewidth=2.3, markersize=4.5, label=label)
                ax.fill_between(x, mean - error, mean + error, color=color, alpha=0.10, linewidth=0)
            ax.axhline(0.0, color="#777777", linewidth=0.8)
            ax.grid(True, linewidth=0.5, alpha=0.20)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title(
                    f"{chr(65 + col)}  {task_name.capitalize()} task",
                    loc="left", fontweight="bold",
                )
            if row_index == 1:
                ax.set_xlabel("Number of fixed score-selected heads jointly ablated")
    for ax in axes.ravel():
        ax.set_xticks(np.arange(max_prefix + 1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "Joint ablation tests the causal roles of specialists, generalists, and inert heads",
        x=0.055, y=0.99, ha="left", fontsize=15, fontweight="bold",
    )
    fig.text(
        0.055, 0.945,
        "Families are selected from J and D only; thin curves are training seeds and bands are seed-level 95% intervals.",
        fontsize=9, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.91), h_pad=2.0, w_pad=2.0)
    return save_figure(fig, figures_dir / "fig6_joint_selectivity_family_ablation"), values


VALIDATION_METRIC_LABELS = (
    ("accuracy", "overall accuracy"),
    ("semantic_accuracy", "semantic accuracy"),
    ("structural_accuracy", "structural accuracy"),
    ("loss", "overall loss"),
    ("semantic_loss", "semantic loss"),
    ("structural_loss", "structural loss"),
)


def validation_performance_summary(checkpoints: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Per-seed checkpoint validation metrics plus the across-seed mean ± sample std.

    Aggregates use ``heldout_validation``: it is evaluated once on fresh graphs after the
    best checkpoint is restored, so unlike the selection-set metrics it is untouched by
    checkpoint selection.  Selection-set metrics are kept per seed for reference only.
    """

    rows: list[dict[str, Any]] = []
    for seed in sorted(checkpoints):
        payload = checkpoints[seed]
        heldout = payload.get("heldout_validation") or {}
        selection = payload.get("best_validation") or {}
        row: dict[str, Any] = {"seed": int(seed)}
        for key, _ in VALIDATION_METRIC_LABELS:
            row[f"heldout_{key}"] = float(heldout.get(key, float("nan")))
            row[f"selection_{key}"] = float(selection.get(key, float("nan")))
        rows.append(row)
    aggregate: dict[str, dict[str, float]] = {}
    for key, _ in VALIDATION_METRIC_LABELS:
        values = np.asarray([row[f"heldout_{key}"] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        n = int(finite.size)
        std = float(finite.std(ddof=1)) if n > 1 else 0.0
        aggregate[key] = {
            "mean": float(finite.mean()) if n else float("nan"),
            "std": std,
            "sem": std / math.sqrt(n) if n > 1 else 0.0,
            "n_seeds": n,
        }
    return {
        "metric_source": "heldout_validation",
        "n_seeds": len(rows),
        "per_seed": rows,
        "aggregate": aggregate,
    }


def print_validation_performance(summary: Mapping[str, Any], cfg: Config) -> None:
    print(
        f"\n[validation] best-checkpoint held-out validation over {summary['n_seeds']} seeds "
        f"({cfg.validation_graphs} fresh graphs per seed; ± is the across-seed sample std)",
        flush=True,
    )
    for row in summary["per_seed"]:
        print(
            f"  seed {int(row['seed'])}: overall={row['heldout_accuracy']:.4f} "
            f"sem={row['heldout_semantic_accuracy']:.4f} "
            f"str={row['heldout_structural_accuracy']:.4f} "
            f"loss={row['heldout_loss']:.4f}",
            flush=True,
        )
    for key, label in VALIDATION_METRIC_LABELS:
        stats = summary["aggregate"][key]
        print(f"  {label:<19} = {stats['mean']:.4f} ± {stats['std']:.4f}", flush=True)


def create_outputs(
    results: Sequence[dict[str, Any]],
    cfg: Config,
    run_dir: Path,
    validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    head_rows = build_head_rows(results)
    write_csv(tables_dir / "per_head_metrics.csv", head_rows)
    fig1 = figure_specialisation_plane(results, cfg, figures_dir)
    fig2, correlations = figure_score_ablation(head_rows, cfg, figures_dir)
    fig3, dissociation = figure_double_dissociation(results, cfg, figures_dir)
    fig4, iterative_ablation = figure_iterative_family_ablation(results, cfg, figures_dir)
    fig5, joint_selectivity_summary, quadrant_rows = figure_joint_selectivity(
        head_rows, cfg, figures_dir
    )
    write_csv(tables_dir / "joint_selectivity_quadrants.csv", quadrant_rows)
    fig6, dj_family_ablation = figure_dj_family_ablation(results, cfg, figures_dir)

    seed_summaries = []
    for result in results:
        seed_summaries.append({
            "seed": int(result["seed"]),
            "selected_groups": result["selected_groups"],
            "checks": result["checks"],
            "semantic_clean_accuracy": result["ablation_semantic"]["clean_accuracy"],
            "structural_clean_accuracy": result["ablation_structural"]["clean_accuracy"],
            "semantic_corrupt_clean_label_accuracy": result["rescue_semantic"]["corrupt_clean_label_accuracy"],
            "structural_corrupt_clean_label_accuracy": result["rescue_structural"]["corrupt_clean_label_accuracy"],
            "semantic_corruption_effect_norm": finite_mean(result["rescue_semantic"]["effect_norm"]),
            "structural_corruption_effect_norm": finite_mean(result["rescue_structural"]["effect_norm"]),
        })
    summary = {
        "version": EXPERIMENT_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "fingerprint": config_fingerprint(cfg),
        "analysis_fingerprint": analysis_fingerprint(cfg),
        "official_grit_commit": OFFICIAL_GRIT_COMMIT,
        "config": asdict(cfg),
        "score_ablation_correlations": correlations,
        "necessity_mean": dissociation["necessity_mean"],
        "rescue_mean": dissociation["rescue_mean"],
        "necessity_interaction_by_seed": dissociation["necessity_interaction"],
        "rescue_interaction_by_seed": dissociation["rescue_interaction"],
        "selected_heads": dissociation["selected_heads"],
        "iterative_family_ablation": iterative_ablation,
        "joint_influence_selectivity": joint_selectivity_summary,
        "joint_selectivity_family_ablation": dj_family_ablation,
        "validation_performance": validation,
        "seeds": seed_summaries,
        "figures": fig1 + fig2 + fig3 + fig4 + fig5 + fig6,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(tables_dir / "selected_heads.json", {"selected_heads": dissociation["selected_heads"]})
    return summary


# ======================================================================================
# Orchestration / single-cell entry point
# ======================================================================================


def environment_record(device: Any) -> dict[str, Any]:
    import torch

    return {
        "experiment_version": EXPERIMENT_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "official_grit_commit": OFFICIAL_GRIT_COMMIT,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() and str(device).startswith("cuda") else None,
    }


def make_config(args: argparse.Namespace) -> Config:
    seeds = tuple(int(value) for value in args.seeds)
    values = dict(
        run_name=args.run_name,
        drive_root=args.drive_root,
        n=args.n,
        key_vocab=args.key_vocab,
        classes=args.classes,
        rrwp_steps=args.rrwp_steps,
        dim=args.dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        validation_graphs=args.validation_graphs,
        patience_checks=args.patience_checks,
        accuracy_gate=args.accuracy_gate,
        score_graphs=args.score_graphs,
        score_donors=args.score_donors,
        score_batch_size=args.score_batch_size,
        ablation_graphs=args.ablation_graphs,
        rescue_graphs=args.rescue_graphs,
        analysis_batch_size=args.analysis_batch_size,
        top_group_size=args.top_group_size,
        seeds=seeds,
        device=args.device,
    )
    if args.fast_dev_run:
        values.update({
            "run_name": args.run_name + "_fast_dev",
            "n": 12,
            "classes": 6,
            "key_vocab": 16,
            "rrwp_steps": 6,
            "dim": 48,
            "heads": 4,
            "layers": 2,
            "batch_size": 12,
            "steps": 20,
            "eval_every": 5,
            "validation_graphs": 24,
            "patience_checks": 100,
            "accuracy_gate": 0.0,
            "score_graphs": 4,
            "score_donors": 2,
            "score_batch_size": 2,
            "ablation_graphs": 8,
            "rescue_graphs": 6,
            "analysis_batch_size": 6,
            "top_group_size": 1,
            "seeds": (seeds[0],),
        })
    cfg = Config(**values)
    cfg.validate()
    return cfg


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Causal semantic/structural GRIT head validation in one Colab cell")
    parser.add_argument("--run-name", default="cycle_dual_v2")
    parser.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--grit-dir", default=DEFAULT_GRIT_DIR)
    parser.add_argument("--phase", choices=("all", "train", "analyze", "figures"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--key-vocab", type=int, default=32)
    parser.add_argument("--classes", type=int, default=8)
    parser.add_argument("--rrwp-steps", type=int, default=10)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-graphs", type=int, default=512)
    parser.add_argument("--patience-checks", type=int, default=10)
    parser.add_argument("--accuracy-gate", type=float, default=0.90)
    parser.add_argument("--score-graphs", type=int, default=96)
    parser.add_argument("--score-donors", type=int, default=6)
    parser.add_argument("--score-batch-size", type=int, default=12)
    parser.add_argument("--ablation-graphs", type=int, default=256)
    parser.add_argument("--rescue-graphs", type=int, default=128)
    parser.add_argument("--analysis-batch-size", type=int, default=64)
    parser.add_argument("--top-group-size", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument("--allow-low-accuracy", action="store_true")
    parser.add_argument("--skip-drive-mount", action="store_true")
    parser.add_argument("--skip-grit-install", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    import torch

    args = parse_args(argv)
    if not args.skip_drive_mount:
        mount_drive()
    cfg = make_config(args)
    # Figure-only reruns consume tensor caches and should not pay the GRIT/PyG install cost.
    if args.phase != "figures":
        setup_official_grit(Path(args.grit_dir), install=not args.skip_grit_install)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA unavailable; official GRIT analysis will be slow", flush=True)
    run_dir = Path(cfg.drive_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "config.json",
        {
            "config": asdict(cfg),
            "fingerprint": config_fingerprint(cfg),
            "analysis_version": ANALYSIS_VERSION,
            "analysis_fingerprint": analysis_fingerprint(cfg),
        },
    )
    write_json(run_dir / "environment.json", environment_record(device))

    models: dict[int, Any] = {}
    checkpoints: dict[int, dict[str, Any]] = {}
    if args.phase in {"all", "train", "analyze"}:
        for seed in cfg.seeds:
            model, payload = train_seed(
                cfg,
                seed=seed,
                run_dir=run_dir,
                device=device,
                force=args.force_retrain,
                load_only=args.phase == "analyze",
            )
            heldout = payload.get("heldout_validation", {})
            minimum = min(
                float(heldout.get("semantic_accuracy", 0.0)),
                float(heldout.get("structural_accuracy", 0.0)),
            )
            if minimum < cfg.accuracy_gate and not args.allow_low_accuracy:
                raise RuntimeError(
                    f"seed {seed} failed the clean accuracy gate: {heldout}; "
                    "refusing causal analysis (pass --allow-low-accuracy only for debugging)"
                )
            models[int(seed)] = model
            checkpoints[int(seed)] = payload
    if args.phase == "train":
        validation_performance = validation_performance_summary(checkpoints) if checkpoints else None
        if validation_performance is not None:
            write_csv(run_dir / "tables" / "validation_performance.csv", validation_performance["per_seed"])
            print_validation_performance(validation_performance, cfg)
        return {
            "run_dir": str(run_dir),
            "checkpoints": [str(checkpoint_path(run_dir, cfg, seed)) for seed in cfg.seeds],
            "validation_performance": validation_performance,
        }

    results = []
    if args.phase in {"all", "analyze"}:
        for seed in cfg.seeds:
            results.append(analyze_seed(
                models[int(seed)],
                cfg,
                seed=seed,
                run_dir=run_dir,
                device=device,
                force=args.force_analysis,
            ))
    else:
        for seed in cfg.seeds:
            # Figures reruns skip training, so recover the cached validation metrics directly.
            ckpt = checkpoint_path(run_dir, cfg, seed)
            if not ckpt.exists():
                print(f"[validation] checkpoint missing for seed {seed}; it will be absent from the report", flush=True)
            else:
                payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                if payload.get("fingerprint") == config_fingerprint(cfg):
                    checkpoints[int(seed)] = payload
                else:
                    print(f"[validation] fingerprint mismatch; skipping {ckpt}", flush=True)
            path = analysis_path(run_dir, cfg, seed)
            if not path.exists():
                raise FileNotFoundError(f"figures phase requires cached analysis: {path}")
            results.append(torch.load(path, map_location="cpu", weights_only=False))

    validation_performance = validation_performance_summary(checkpoints) if checkpoints else None
    if validation_performance is not None:
        write_csv(run_dir / "tables" / "validation_performance.csv", validation_performance["per_seed"])
    summary = create_outputs(results, cfg, run_dir, validation=validation_performance)
    print("\n[done]", flush=True)
    print(f"  run_dir: {run_dir}", flush=True)
    print(f"  checkpoints: {run_dir / 'checkpoints'}", flush=True)
    print(f"  analysis cache: {run_dir / 'analysis'}", flush=True)
    print(f"  figures: {run_dir / 'figures'}", flush=True)
    print(f"  per-head table: {run_dir / 'tables' / 'per_head_metrics.csv'}", flush=True)
    if validation_performance is None:
        print("[validation] no matching checkpoints found; validation performance unavailable", flush=True)
    else:
        print(f"  validation table: {run_dir / 'tables' / 'validation_performance.csv'}", flush=True)
        print_validation_performance(validation_performance, cfg)
    return summary


if __name__ == "__main__":
    main([
        "--run-name", "cycle_dual_v2",
        "--phase", "analyze",
        "--seeds", "0", "1", "2",
        "--steps", "3000",
        "--score-graphs", "96",
        "--score-donors", "6",
        "--ablation-graphs", "256",
        "--rescue-graphs", "128",
    ])
