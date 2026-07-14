"""Synthetic bottleneck content-retrieval: does dense attention break away from 1-hop?

First-run experiment for the dissertation's "when does dense attention help?" question.

Task -- cross-bottleneck content-addressed retrieval:
    A graph of two clusters joined by a thin bridge (a "dumbbell"). Every node carries a
    distinct KEY and a VALUE class. A set of QUERY nodes in cluster A each hold the key of a
    target node in cluster B, and must output that target's value. The match is by CONTENT
    (key), uncorrelated with structure, so global positional encoding cannot locate it. Graph
    diameter is small, so REACH is not the limiter -- the only thing dense attention has is
    direct cross-bridge routing vs. the bridge's over-squashed bandwidth.

Models (parameter-matched; only the attention mask differs -- the dense vs 1-hop contrast):
    * dense  : attention over all node pairs (a stand-in for dense GRIT);
    * 1-hop  : attention masked to graph edges, but with the SAME global RRWP features
               (a stand-in for 1-hop GRIT + global RRWP).

Controllable axes -> the breakaway plot:
    * rank r  = number of simultaneous cross-bridge retrievals (bandwidth stress);
    * graph   = "dumbbell" (bottleneck) vs "wellconnected" (control, no bottleneck).
Prediction: on the dumbbell, dense stays flat while 1-hop falls as r grows; on the
wellconnected control they stay together (the gap is the bottleneck, not the task).

This module is pure torch/numpy (no torch_geometric) so it runs and smoke-tests locally; the
same task + harness swaps in official dense/1-hop GRIT on Colab for the carriage analysis.
Run from ``main([...])``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# ======================================================================================
# Task generation (pure numpy/torch; fixed node count n so graphs batch cleanly)
# ======================================================================================
def _add_cluster_edges(nodes: np.ndarray, degree: int, adj: np.ndarray, rng: np.random.Generator) -> None:
    """Wire ``nodes`` into a connected ~``degree``-regular block (in place).

    A random spanning path guarantees connectivity; random stub-pairing adds the rest of the
    target degree. Using the SAME routine for both graph types means dumbbell and wellconnected
    have matched local density and differ only in the bottleneck.
    """
    nodes = np.asarray(nodes)
    m = len(nodes)
    if m <= 1:
        return
    degree = int(min(degree, m - 1))
    perm = rng.permutation(nodes)                      # spanning path -> connected
    for i in range(m - 1):
        u, v = int(perm[i]), int(perm[i + 1])
        adj[u, v] = adj[v, u] = 1.0
    extra = max(0, degree - 2)                          # path already gives ~degree 2
    if extra > 0:
        stubs = np.repeat(nodes, extra)
        rng.shuffle(stubs)
        for i in range(0, len(stubs) - 1, 2):
            u, v = int(stubs[i]), int(stubs[i + 1])
            if u != v:
                adj[u, v] = adj[v, u] = 1.0


def _dumbbell_adj(n: int, cluster_degree: int, bridge_edges: int, rng: np.random.Generator) -> np.ndarray:
    """Two equal ~cluster_degree-regular blocks joined by ``bridge_edges`` bridge edges."""
    half = n // 2
    adj = np.zeros((n, n), dtype=np.float32)
    _add_cluster_edges(np.arange(half), cluster_degree, adj, rng)
    _add_cluster_edges(np.arange(half, n), cluster_degree, adj, rng)
    a_nodes = rng.permutation(half)[:bridge_edges]
    b_nodes = half + rng.permutation(n - half)[:bridge_edges]
    for a, b in zip(a_nodes, b_nodes):
        adj[a, b] = adj[b, a] = 1.0
    return adj


def _wellconnected_adj(n: int, degree: int, rng: np.random.Generator) -> np.ndarray:
    """One ~degree-regular block over all nodes: same local density as a dumbbell cluster, no bottleneck."""
    adj = np.zeros((n, n), dtype=np.float32)
    _add_cluster_edges(np.arange(n), degree, adj, rng)
    return adj


def _expander_adj(n: int, degree: int, rng: np.random.Generator) -> np.ndarray:
    """Random ~degree-regular expander: same sparsity/degree as a dumbbell but no bottleneck.

    The sparsification-story control. An expander is a SPARSE graph (edge count ~ n*degree/2, like
    the dumbbell) whose spectral gap stays large, so k random rewired shortcuts relieve the
    over-squashed bandwidth WITHOUT going dense. If 1-hop attention on an expander recovers the
    dense-retrieval accuracy the dumbbell loses, the bottleneck -- not the density -- was the whole
    story, and a good sparsifier only needs to add expander-like shortcuts (not all n^2 edges)."""
    adj = np.zeros((n, n), dtype=np.float32)
    perm = rng.permutation(n)                              # Hamiltonian cycle -> connected, 2-regular
    for i in range(n):
        u, v = int(perm[i]), int(perm[(i + 1) % n])
        adj[u, v] = adj[v, u] = 1.0
    extra = max(0, int(degree) - 2)                        # add random perfect matchings to reach degree
    for _ in range(extra):
        pm = rng.permutation(n)
        for i in range(0, n - 1, 2):
            u, v = int(pm[i]), int(pm[i + 1])
            if u != v:
                adj[u, v] = adj[v, u] = 1.0
    np.fill_diagonal(adj, 0.0)
    return adj


def _bfs_distances(adj: np.ndarray, src: int) -> np.ndarray:
    """Shortest-path hop distances from ``src`` (unreachable -> -1)."""
    n = adj.shape[0]
    dist = np.full(n, -1, dtype=np.int64)
    dist[src] = 0
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.nonzero(adj[u])[0]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    nxt.append(int(v))
        frontier = nxt
    return dist


def _rrwp(adj: np.ndarray, steps: int) -> np.ndarray:
    """Random-walk return/transition features: rrwp[i, j, s] = (P^s)[i, j], P = D^-1 A."""
    n = adj.shape[0]
    deg = adj.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    p = adj / deg
    out = np.zeros((n, n, steps), dtype=np.float32)
    power = np.eye(n, dtype=np.float32)
    for s in range(steps):
        power = power @ p if s > 0 else np.eye(n, dtype=np.float32)
        out[:, :, s] = power
    return out


@dataclass
class Batch:
    x: Tensor           # [B, n, Fin] node features
    adj: Tensor         # [B, n, n]   0/1 adjacency (attention mask for the 1-hop model)
    rrwp: Tensor        # [B, n, n, steps]
    qmask: Tensor       # [B, n] bool  (query nodes)
    label: Tensor       # [B, n] long  (target value class at query nodes, -1 elsewhere)
    target_idx: Tensor  # [B, n] long  (planted target node per query, -1 elsewhere)


def make_batch(
    batch_size: int,
    *,
    graph: str,
    n: int,
    rank: int,
    key_vocab: int,
    value_vocab: int,
    rrwp_steps: int,
    bridge_edges: int,
    degree: int,
    seed: int,
    device: torch.device,
    addressing: str = "content",
    target_distance: int | None = None,
) -> Batch:
    """Feature layout: [key(K) | value(V) | query-key(K) | is_query(1) | anchor_flag(1)].

    addressing="content"  : query token = target's KEY (target in the opposite cluster); the
        match is by content and structure-independent -> global RRWP cannot locate it.
    addressing="structural": one node is marked as an ANCHOR; every query must output the
        anchor's value. Selection is trivial (the anchor is marked) -- this isolates ROUTING
        without content-selection, the task 1-hop + RRWP should be able to do.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    feat = 2 * key_vocab + value_vocab + 2
    q_dim = 2 * key_vocab + value_vocab       # is_query flag index
    anchor_dim = q_dim + 1                     # anchor flag index
    xs, adjs, rrwps, qmasks, labels, tidx = [], [], [], [], [], []
    for _ in range(batch_size):
        if graph == "dumbbell":
            adj = _dumbbell_adj(n, degree, bridge_edges, rng)
        elif graph == "wellconnected":
            adj = _wellconnected_adj(n, degree, rng)
        elif graph == "expander":
            adj = _expander_adj(n, degree, rng)
        else:
            raise ValueError(f"unknown graph type {graph!r}")
        keys = rng.permutation(key_vocab)[:n]          # distinct keys per node
        values = rng.integers(0, value_vocab, size=n)  # value class per node
        x = np.zeros((n, feat), dtype=np.float32)
        x[np.arange(n), keys] = 1.0                                   # key onehot
        x[np.arange(n), key_vocab + values] = 1.0                    # value onehot
        qmask = np.zeros(n, dtype=bool)
        label = np.full(n, -1, dtype=np.int64)
        tgt = np.full(n, -1, dtype=np.int64)

        if addressing == "content":
            if target_distance is None:
                # cross-cluster (dumbbell) pairing: a fixed non-local retrieval
                r = min(rank, half)
                pairs = list(zip([int(v) for v in rng.permutation(half)[:r]],
                                 [int(half + v) for v in rng.permutation(n - half)[:r]]))
            else:
                # plant each content target at BFS distance `target_distance` from its query
                pairs = []
                used_q: set[int] = set()
                for _try in range(80):
                    if len(pairs) >= rank:
                        break
                    q = int(rng.integers(0, n))
                    if q in used_q:
                        continue
                    dist = _bfs_distances(adj, q)
                    cand = [int(c) for c in np.nonzero(dist == target_distance)[0] if c != q]
                    if cand:
                        pairs.append((q, int(rng.choice(cand))))
                        used_q.add(q)
            for q, t in pairs:
                qmask[q] = True
                x[q, q_dim] = 1.0                                    # is_query flag
                x[q, key_vocab + value_vocab + keys[t]] = 1.0        # query = target's key
                label[q] = values[t]
                tgt[q] = t
        elif addressing == "structural":
            # one marked anchor; queries route the anchor's value (no content selection)
            anchor = int(rng.integers(0, n))
            x[anchor, anchor_dim] = 1.0
            r = min(rank, n - 1)
            q_nodes = [i for i in rng.permutation(n) if i != anchor][:r]
            for q in q_nodes:
                qmask[q] = True
                x[q, q_dim] = 1.0
                label[q] = values[anchor]
                tgt[q] = anchor
        else:
            raise ValueError(f"unknown addressing {addressing!r}")

        xs.append(x)
        adjs.append(adj)
        rrwps.append(_rrwp(adj, rrwp_steps))
        qmasks.append(qmask)
        labels.append(label)
        tidx.append(tgt)

    to = lambda a, dt: torch.as_tensor(np.stack(a), dtype=dt, device=device)
    return Batch(
        x=to(xs, torch.float32),
        adj=to(adjs, torch.float32),
        rrwp=to(rrwps, torch.float32),
        qmask=to(qmasks, torch.bool),
        label=to(labels, torch.long),
        target_idx=to(tidx, torch.long),
    )


# ======================================================================================
# Model: RRWP-biased multi-head attention, dense or edge-masked (the only difference)
# ======================================================================================
class RRWPAttentionLayer(nn.Module):
    def __init__(self, dim: int, heads: int, rrwp_steps: int, dropout: float) -> None:
        super().__init__()
        self.h = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.pair_bias = nn.Linear(rrwp_steps, heads)  # RRWP pairwise attention bias
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, h: Tensor, rrwp: Tensor, mask: Tensor | None) -> Tensor:
        b, n, _ = h.shape
        q, k, v = self.qkv(self.norm1(h)).chunk(3, dim=-1)
        resh = lambda t: t.view(b, n, self.h, self.dk).transpose(1, 2)  # [B,H,n,dk]
        q, k, v = resh(q), resh(k), resh(v)
        logits = (q @ k.transpose(-1, -2)) / (self.dk ** 0.5)           # [B,H,n,n]
        logits = logits + self.pair_bias(rrwp).permute(0, 3, 1, 2)      # RRWP bias
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        attn = self.drop(F.softmax(logits, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        h = h + self.drop(self.proj(out))
        h = h + self.drop(self.ff(self.norm2(h)))
        return h


class BottleneckRetriever(nn.Module):
    """Dense vs 1-hop is a single flag: `dense` toggles whether attention sees all pairs or
    only graph edges. Everything else (params, depth, width, RRWP) is identical."""

    def __init__(
        self,
        *,
        in_dim: int,
        value_vocab: int,
        dim: int = 64,
        heads: int = 4,
        layers: int = 4,
        rrwp_steps: int = 8,
        dropout: float = 0.0,
        dense: bool = True,
        vnode: bool = False,
    ) -> None:
        super().__init__()
        self.dense = dense
        self.use_vnode = vnode
        self.encoder = nn.Linear(in_dim, dim)
        self.rrwp_node = nn.Linear(rrwp_steps, dim)  # node RRWP = diagonal of the pair tensor
        self.vnode_emb = nn.Parameter(torch.zeros(1, 1, dim)) if vnode else None
        self.layers = nn.ModuleList(
            RRWPAttentionLayer(dim, heads, rrwp_steps, dropout) for _ in range(layers)
        )
        self.head = nn.Linear(dim, value_vocab)

    def encode(self, batch: Batch, rrwp: Tensor | None = None) -> Tensor:
        """Post-encoder node state h0 = content embedding + node-RRWP (the IG resample unit).

        ``rrwp`` overrides ``batch.rrwp`` (used by the structural-carriage probes to perturb the raw
        node-RRWP diagonal before it reaches the node encoder)."""
        b, n = batch.x.shape[:2]
        r = batch.rrwp if rrwp is None else rrwp
        node_rrwp = r[torch.arange(b)[:, None], torch.arange(n)[None], torch.arange(n)[None]]
        return self.encoder(batch.x) + self.rrwp_node(node_rrwp)

    def propagate(self, batch: Batch, h0: Tensor, rrwp: Tensor | None = None) -> Tensor:
        """Run the attention stack from a given h0 -- lets IG integrate over the encoded content.

        ``rrwp`` overrides ``batch.rrwp`` for the pairwise attention bias (used by the pair-RRWP
        structural-carriage probe)."""
        b, n = h0.shape[:2]
        rrwp = batch.rrwp if rrwp is None else rrwp
        if self.dense:
            mask = None
        else:
            eye = torch.eye(n, device=batch.adj.device).unsqueeze(0)
            mask = ((batch.adj + eye) > 0).float()
        h = h0
        if self.use_vnode:
            h = torch.cat([h, self.vnode_emb.expand(b, 1, -1)], dim=1)
            rrwp = F.pad(rrwp, (0, 0, 0, 1, 0, 1))
            base = mask if mask is not None else torch.ones(b, n, n, device=h.device)
            big = torch.zeros(b, n + 1, n + 1, device=h.device, dtype=base.dtype)
            big[:, :n, :n] = base
            big[:, :n, n] = 1.0
            big[:, n, :] = 1.0
            mask = big
        for layer in self.layers:
            h = layer(h, rrwp, mask)
        return h[:, :n] if self.use_vnode else h

    def node_states(self, batch: Batch, rrwp: Tensor | None = None) -> Tensor:
        return self.propagate(batch, self.encode(batch, rrwp), rrwp)

    def forward(self, batch: Batch) -> Tensor:
        return self.head(self.node_states(batch))


# ======================================================================================
# Train / eval / sweep
# ======================================================================================
def _loss_and_acc(logits: Tensor, batch: Batch) -> tuple[Tensor, float, int]:
    qm = batch.qmask
    if int(qm.sum()) == 0:
        return logits.new_zeros(()), float("nan"), 0
    sel = logits[qm]
    tgt = batch.label[qm]
    loss = F.cross_entropy(sel, tgt)
    acc = float((sel.argmax(-1) == tgt).float().mean().item())
    return loss, acc, int(qm.sum())


MODEL_SPECS = {
    "dense": dict(dense=True, vnode=False),
    "1hop": dict(dense=False, vnode=False),
    "1hop_vnode": dict(dense=False, vnode=True),
}


CKPT_KEYS = ("n", "layers", "dim", "heads", "rrwp_steps", "dropout", "key_vocab",
             "value_vocab", "bridge_edges", "degree", "steps", "lr", "batch_size")


def _ckpt_fingerprint(model_name: str, graph: str, cfg: dict, seed: int) -> str:
    """Stable id from architecture + data-generating config, so a stale checkpoint is never reused."""
    import hashlib

    payload = {
        "model": model_name, "graph": graph, "seed": int(seed),
        "addressing": cfg.get("addressing"), "rank": cfg.get("rank"),
        "target_distance": cfg.get("target_distance"),
        **{k: cfg[k] for k in CKPT_KEYS if k in cfg},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _ckpt_stem(prefix: str, model_name: str, graph: str, cfg: dict, seed: int) -> str:
    """Semantic checkpoint key: everything except the trailing config-hash."""
    return (f"{prefix}{model_name}__{graph}__{cfg.get('addressing')}__r{cfg.get('rank')}"
            f"__d{cfg.get('target_distance')}__s{seed}")


def _ckpt_filename(prefix: str, model_name: str, graph: str, cfg: dict, seed: int) -> str:
    return f"{_ckpt_stem(prefix, model_name, graph, cfg, seed)}__{_ckpt_fingerprint(model_name, graph, cfg, seed)}.pt"


def _find_checkpoint_by_stem(ckpt_dir: Path, stem: str, fp: str):
    """Locate a checkpoint by its SEMANTIC key ``stem`` (model/graph/addressing/rank/distance/seed),
    preferring the exact config-hash ``fp`` but falling back to ANY hash -- so a changed config default
    never silently blocks a load (a drift warning is printed)."""
    exact = ckpt_dir / f"{stem}__{fp}.pt"
    if exact.exists():
        return exact
    matches = sorted(ckpt_dir.glob(f"{stem}__*.pt")) if ckpt_dir.exists() else []
    if matches:
        print(f"[cache] config-fingerprint drift for {stem}: loading {matches[0].name} (config differs from "
              "this run's defaults, but model/graph/addressing/rank/distance/seed match).", flush=True)
        return matches[0]
    return None


def _missing_ckpt_message(ckpt_dir: Path | None, stem: str) -> str:
    present = sorted(p.name for p in ckpt_dir.glob("*.pt")) if (ckpt_dir is not None and ckpt_dir.exists()) else []
    is_official = stem.startswith("official_")
    mismatched = [nm for nm in present if nm.startswith("official_") != is_official]
    hint = ""
    if mismatched:
        if is_official:
            hint = (" NOTE: this directory contains non-'official_' (pure-torch) checkpoints -- run the analysis "
                    "from the PURE-TORCH file synthetic_bottleneck_retrieval.py instead.")
        else:
            hint = (" NOTE: this directory contains 'official_*' checkpoints from the OFFICIAL-GRIT variant -- those "
                    "are a different model; run synthetic_bottleneck_retrieval_official.py instead (this pure-torch "
                    "file cannot load them).")
    more = f"  (+{len(present) - 12} more)" if len(present) > 12 else ""
    return (f"[analyze] no checkpoint matching '{stem}__*.pt' in {ckpt_dir}.\n"
            f"  present .pt files: {present[:12]}{more}.{hint}\n"
            "  If nothing is present, run --phase train first; if only the config differs, it is now loaded anyway.")


def train_one(
    *,
    model_name: str,
    graph: str,
    cfg: dict,
    device: torch.device,
    seed: int,
    ckpt_dir: Path | None = None,
    force_retrain: bool = False,
    load_only: bool = False,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    in_dim = 2 * cfg["key_vocab"] + cfg["value_vocab"] + 2
    model = BottleneckRetriever(
        in_dim=in_dim,
        value_vocab=cfg["value_vocab"],
        dim=cfg["dim"],
        heads=cfg["heads"],
        layers=cfg["layers"],
        rrwp_steps=cfg["rrwp_steps"],
        dropout=cfg["dropout"],
        **MODEL_SPECS[model_name],
    ).to(device)
    ckpt_dir = Path(ckpt_dir) if ckpt_dir is not None else None
    stem = _ckpt_stem("", model_name, graph, cfg, seed)
    fp = _ckpt_fingerprint(model_name, graph, cfg, seed)
    if ckpt_dir is not None:
        chosen = _find_checkpoint_by_stem(ckpt_dir, stem, fp)
        if chosen is not None and not force_retrain:
            try:
                payload = torch.load(chosen, map_location=device, weights_only=False)
                model.load_state_dict(payload["state_dict"])
            except Exception as exc:  # noqa: BLE001 -- architecture mismatch etc.
                if load_only:
                    raise RuntimeError(
                        f"[analyze] found '{chosen.name}' but its weights do not fit the current model "
                        f"(architecture/config differs): {exc}. Pass the SAME architecture args used at "
                        "training (--n --layers --dim --heads --rrwp-steps)."
                    ) from exc
                chosen = None  # train phase: fall through and retrain
            else:
                model.eval()
                meta = dict(payload.get("meta", {}))
                meta["loaded_from_cache"] = True
                return model, meta
    if load_only:  # analyze phase: never train; the model must already be on Drive
        raise RuntimeError(_missing_ckpt_message(ckpt_dir, stem))

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sample = lambda s: make_batch(
        cfg["batch_size"], graph=graph, n=cfg["n"], rank=cfg["rank"],
        key_vocab=cfg["key_vocab"], value_vocab=cfg["value_vocab"], rrwp_steps=cfg["rrwp_steps"],
        bridge_edges=cfg["bridge_edges"], degree=cfg["degree"], seed=s, device=device,
        addressing=cfg["addressing"], target_distance=cfg.get("target_distance"),
    )
    model.train()
    log = []
    _t0 = time.time()
    _every = max(1, cfg["steps"] // 10)
    for step in range(cfg["steps"]):
        batch = sample(seed * 100003 + step)
        loss, acc, nq = _loss_and_acc(model(batch), batch)
        if nq == 0:  # no query placeable at this distance in the whole batch -> zero has no grad_fn
            continue
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % _every == 0 or step == cfg["steps"] - 1:
            log.append({"step": step, "loss": float(loss.item()), "train_acc": acc})
            rate = (step + 1) / max(time.time() - _t0, 1e-6)
            print(f"    [{model_name}/{graph} s{seed}] step {step + 1}/{cfg['steps']} "
                  f"loss={loss.item():.4f} acc={acc:.3f} ({rate:.1f} it/s)", flush=True)
    # eval on fresh graphs; also record final TRAIN accuracy (the routing-vs-memorisation guardrail)
    model.eval()
    accs, train_accs = [], []
    with torch.no_grad():
        for e in range(cfg["eval_batches"]):
            batch = sample(10_000_000 + seed * 991 + e)
            _, acc, nq = _loss_and_acc(model(batch), batch)
            if nq:
                accs.append(acc)
            tb = sample(seed * 100003 + e)  # graphs the model trained on
            _, tacc, tnq = _loss_and_acc(model(tb), tb)
            if tnq:
                train_accs.append(tacc)
    meta = {
        "val_acc": float(np.mean(accs)) if accs else float("nan"),
        "train_acc": float(np.mean(train_accs)) if train_accs else float("nan"),
        "train_log": log,
        "params": int(sum(p.numel() for p in model.parameters())),
        "loaded_from_cache": False,
    }
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / _ckpt_filename("", model_name, graph, cfg, seed)
        torch.save({"state_dict": model.state_dict(), "meta": meta,
                    "cfg": {k: cfg[k] for k in CKPT_KEYS if k in cfg},
                    "model_name": model_name, "graph": graph, "seed": seed}, ckpt_path)
    return model, meta


def run_sweep(cfg: dict, device: torch.device, *, ckpt_dir: Path | None = None,
              force_retrain: bool = False) -> list[dict]:
    rows = []
    distances = cfg.get("distances", [None])
    for addressing in cfg["addressings"]:
        for graph in cfg["graphs"]:
            for dist in distances:
                for rank in cfg["ranks"]:
                    for model_name in cfg["models"]:
                        run_cfg = {**cfg, "rank": rank, "addressing": addressing, "target_distance": dist}
                        accs, tr_accs, params, n_cached = [], [], None, 0
                        for seed in range(cfg["seeds"]):
                            _, res = train_one(model_name=model_name, graph=graph, cfg=run_cfg,
                                               device=device, seed=seed, ckpt_dir=ckpt_dir,
                                               force_retrain=force_retrain)
                            accs.append(res["val_acc"])
                            tr_accs.append(res.get("train_acc", float("nan")))
                            params = res["params"]
                            n_cached += int(bool(res.get("loaded_from_cache")))
                        rows.append({
                            "addressing": addressing, "graph": graph,
                            "distance": dist, "rank": rank, "model": model_name,
                            "val_acc_mean": float(np.nanmean(accs)), "val_acc_std": float(np.nanstd(accs)),
                            "train_acc_mean": float(np.nanmean(tr_accs)),
                            "seeds": cfg["seeds"], "params": params,
                        })
                        dstr = f"d={dist}" if dist is not None else "d=far"
                        cflag = f" [cache {n_cached}/{cfg['seeds']}]" if n_cached else ""
                        print(f"  [{addressing:10s} {graph:13s} {dstr:6s} rank={rank:2d} {model_name:11s}] "
                              f"val={rows[-1]['val_acc_mean']:.3f}+/-{rows[-1]['val_acc_std']:.3f} "
                              f"train={rows[-1]['train_acc_mean']:.3f}{cflag}", flush=True)
    return rows


def plot_breakaway(rows: list[dict], out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_dist = any(r.get("distance") is not None for r in rows)
    xkey = "distance" if has_dist else "rank"
    xlabel = "query->target distance (hops)" if has_dist else "rank  (# simultaneous retrievals)"
    model_styles = {
        "dense": dict(marker="o", color="#1f77b4"),
        "1hop": dict(marker="s", color="#d62728"),
        "1hop_vnode": dict(marker="^", color="#2ca02c"),
    }
    addressings = sorted({r.get("addressing", "content") for r in rows})
    graphs = sorted({r["graph"] for r in rows})
    fig, axes = plt.subplots(
        len(addressings), len(graphs),
        figsize=(5.2 * len(graphs), 3.8 * len(addressings)), squeeze=False,
    )
    for ai, addressing in enumerate(addressings):
        for gi, graph in enumerate(graphs):
            ax = axes[ai][gi]
            for model, style in model_styles.items():
                pts = sorted(
                    [r for r in rows if r.get("addressing", "content") == addressing
                     and r["graph"] == graph and r["model"] == model and r.get(xkey) is not None],
                    key=lambda r: r[xkey],
                )
                if not pts:
                    continue
                ax.errorbar([p[xkey] for p in pts], [p["val_acc_mean"] for p in pts],
                            yerr=[p["val_acc_std"] for p in pts], label=model, capsize=3, **style)
            ax.set_title(f"{addressing} / {graph}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("retrieval accuracy")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle(f"dense vs 1-hop vs 1-hop+VNode: retrieval accuracy vs {xkey}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ======================================================================================
# Entry point
# ======================================================================================
def default_config() -> dict:
    return {
        "n": 16, "layers": 4, "dim": 64, "heads": 4, "rrwp_steps": 8, "dropout": 0.0,
        "key_vocab": 16, "value_vocab": 8, "bridge_edges": 1, "degree": 6,
        "addressings": ["content", "structural"],
        "graphs": ["dumbbell", "wellconnected"], "ranks": [1, 2, 4, 8],
        "models": ["dense", "1hop", "1hop_vnode"],
        "distances": [None],  # set e.g. [1,2,3,4] to sweep query->target distance (the reach axis)
        "batch_size": 64, "steps": 400, "eval_batches": 8, "lr": 1e-3, "seeds": 2,
        # --- Step-7 carriage suite ---
        "carriage_graphs": ["dumbbell", "expander", "wellconnected"],
        "carriage_target_distance": 3, "carriage_graphs_count": 12, "carriage_channel_start": 2,
        "carriage_ig_steps": 24, "carriage_rrwp_replacement": "donor", "carriage_donor_samples": 4,
        "carriage_min_distance": 1,
    }


# ======================================================================================
# Step-7 carriage suite (mirrors grit_intervention_procedure.symbolic_structural_carriage_rows):
# symbolic (content) vs structural (node-RRWP / pair-RRWP) carriage, each via BOTH the finite-swap
# AND the integrated-gradients estimator, in FUNCTIONAL and BENEFICIAL modes -- so the standalone
# synthetic uses the same definitions as Step 7 of the core methodology. Methodology is beta: we
# emit all six factors x two modes so the IG/swap agreement can be checked and the better estimator
# picked per substrate.
#
#   Carriage C[query carrier, source j] = g . (h_clean - h_corrupt), the readout-projected node
#   delta. g = dS/dh^L is the readout gradient of a scalar S at the query node(s):
#     functional : S = sum_q logit[q, label_q]        (readout mass on the correct value)
#     beneficial : S = sum_q CE(logit_q, label_q)     (query loss; signed<0 = loss-reducing = good)
#   swap = discrete on-manifold perturbation of source j's substrate (multi-donor averaged, frozen g);
#   IG   = integrate the substrate from its graph-mean baseline to clean and attribute per source.
#   Structural channels < ``channel_start`` (self/one-hop identity) are preserved, so the RRWP probes
#   perturb GLOBAL structural information only.
# ======================================================================================
IG_FACTORS = ("content_ig", "node_rrwp_ig", "pair_rrwp_ig")
SWAP_FACTORS = ("content", "node_rrwp", "pair_rrwp")
# RRWP carriage is a 2D object over (source node j, walk-length channel r). Beyond the by-DISTANCE
# marginal (factors above), we also emit the by-WALK-LENGTH (scale) marginal -- which structural
# scales of the per-node/pair RRWP the focal node collectively uses (high-r mass = the global
# structure local-RRWP truncation removes) -- and the (distance x scale) JOINT. Both estimators.
STRUCTURAL_KINDS = (("node", "node_rrwp"), ("pair", "pair_rrwp"))
SCALE_IG_FACTORS = {"node": "node_rrwp_scale_ig", "pair": "pair_rrwp_scale_ig"}
JOINT_IG_FACTORS = {"node": "node_rrwp_joint_ig", "pair": "pair_rrwp_joint_ig"}
SCALE_SWAP_FACTORS = {"node": "node_rrwp_scale", "pair": "pair_rrwp_scale"}
CARRIAGE_FACTORS = SWAP_FACTORS + IG_FACTORS
CARRIAGE_MODES = ("functional", "beneficial")


def _slice_graph(batch: Batch, gi: int) -> Batch:
    return Batch(
        x=batch.x[gi:gi + 1], adj=batch.adj[gi:gi + 1], rrwp=batch.rrwp[gi:gi + 1],
        qmask=batch.qmask[gi:gi + 1], label=batch.label[gi:gi + 1], target_idx=batch.target_idx[gi:gi + 1],
    )


def _readout_scalar(logits: Tensor, b1: Batch, mode: str) -> Tensor:
    qm = b1.qmask
    if mode == "functional":
        return logits[qm].gather(1, b1.label[qm].clamp_min(0)[:, None]).sum()
    return F.cross_entropy(logits[qm], b1.label[qm], reduction="sum")  # beneficial: loss (<0 carriage = good)


def _readout_gradient(model: nn.Module, b1: Batch, mode: str) -> tuple[Tensor, Tensor]:
    """Clean final states h [1,n,dim] and frozen readout gradient g = dS/dh (nonzero at query carriers)."""
    h = model.node_states(b1).detach().requires_grad_(True)
    (g,) = torch.autograd.grad(_readout_scalar(model.head(h), b1, mode), h)
    return h.detach(), g.detach()


def _draw_donor(rng: np.random.Generator, n: int, j: int) -> int:
    d = int(rng.integers(0, n - 1))
    return d + (1 if d >= j else 0)


def _content_replaced(b1: Batch, source: int, donor: int) -> Batch:
    x = b1.x.clone()
    x[0, source] = b1.x[0, donor]
    return Batch(x=x, adj=b1.adj, rrwp=b1.rrwp, qmask=b1.qmask, label=b1.label, target_idx=b1.target_idx)


def _rrwp_perturbed(rrwp: Tensor, source: int, *, kind: str, start: int, replacement: str, donor: int) -> Tensor:
    """Replace long-range (channels>=start) RRWP at ``source``: node = its diagonal; pair = entries
    incident to it (row+col), preserving the diagonal node-RRWP. donor/mean/zero replacements."""
    r = rrwp.clone()
    n = r.size(1)
    idx = torch.arange(n, device=r.device)
    off = idx != source
    if kind == "node":
        if replacement == "donor":
            r[0, source, source, start:] = rrwp[0, donor, donor, start:]
        elif replacement == "mean":
            diag = rrwp[0, idx, idx]
            r[0, source, source, start:] = diag[:, start:].mean(0)
        else:
            r[0, source, source, start:] = 0.0
    else:  # pair: all off-diagonal entries incident to source
        if replacement == "donor":
            r[0, source, off, start:] = rrwp[0, donor, off, start:]
            r[0, off, source, start:] = rrwp[0, off, donor, start:]
        elif replacement == "mean":
            eye = torch.eye(n, dtype=torch.bool, device=r.device)
            m = rrwp[0][~eye][:, start:].mean(0)
            r[0, source, off, start:] = m
            r[0, off, source, start:] = m
        else:
            r[0, source, off, start:] = 0.0
            r[0, off, source, start:] = 0.0
    return r


def _swap_carriage(model: nn.Module, b1: Batch, factor: str, mode: str, *, start: int,
                   replacement: str, donors: int, rng: np.random.Generator) -> Tensor:
    """Per-source carriage c[j] via the finite on-manifold swap (frozen readout gradient)."""
    h_clean, g = _readout_gradient(model, b1, mode)
    n = b1.x.size(1)
    k = donors if replacement == "donor" else 1
    c = torch.zeros(n)
    for j in range(n):
        acc, got = 0.0, 0
        for _ in range(max(1, k)):
            donor = _draw_donor(rng, n, j)
            if factor == "content":
                h = model.node_states(_content_replaced(b1, j, donor)).detach()
            else:
                kind = "node" if factor == "node_rrwp" else "pair"
                r = _rrwp_perturbed(b1.rrwp, j, kind=kind, start=start, replacement=replacement, donor=donor)
                h = model.node_states(b1, r).detach()
            acc += float((g * (h_clean - h)).sum().item())
            got += 1
        c[j] = acc / max(got, 1)
    return c


def _ig_carriage(model: nn.Module, b1: Batch, factor: str, mode: str, *, start: int,
                 steps: int) -> tuple[Tensor, dict[str, float]]:
    """Per-source carriage c[j] via IG over the substrate; completeness = sum c vs S(1)-S(0)."""
    n = b1.x.size(1)
    dev = b1.x.device
    if factor == "content_ig":
        h0 = model.encode(b1).detach()
        base = h0.mean(dim=1, keepdim=True).expand_as(h0).contiguous()
        delta = h0 - base
        c = torch.zeros(n, device=dev)
        for a in range(1, steps + 1):
            pt = (base + (a / steps) * delta).detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(_readout_scalar(model.head(model.propagate(b1, pt)), b1, mode), pt)
            c += (grad.detach() * delta).sum(dim=-1)[0] / steps
        with torch.no_grad():
            s0 = float(_readout_scalar(model.head(model.propagate(b1, base)), b1, mode).item())
            s1 = float(_readout_scalar(model.head(model.propagate(b1, (base + delta))), b1, mode).item())
        return c.cpu(), {"recon": float(c.sum().item()), "target": s1 - s0}

    K = b1.rrwp.size(-1)
    flat_clean = b1.rrwp.detach().reshape(n * n, K)
    a_idx = torch.arange(n, device=dev).repeat_interleave(n)
    b_idx = torch.arange(n, device=dev).repeat(n)
    diag = a_idx == b_idx
    if factor == "node_rrwp_ig":
        active, src = diag, a_idx
    else:  # pair: attribute entry (a,b) to source b (the key node whose structure reaches carriers)
        active, src = ~diag, b_idx
    base_flat = flat_clean.clone()
    if bool(active.any()) and K > start:
        base_flat[active, start:] = flat_clean[active, start:].mean(0)
    delta_flat = flat_clean - base_flat
    c = torch.zeros(n, device=dev)
    for a in range(1, steps + 1):
        pt = (base_flat + (a / steps) * delta_flat).detach().requires_grad_(True)
        logits = model.head(model.node_states(b1, pt.reshape(1, n, n, K)))
        (grad,) = torch.autograd.grad(_readout_scalar(logits, b1, mode), pt)
        contrib = (grad.detach() * delta_flat).sum(dim=-1)
        c.index_add_(0, src[active], contrib[active] / steps)
    with torch.no_grad():
        s0 = float(_readout_scalar(model.head(model.node_states(b1, base_flat.reshape(1, n, n, K))), b1, mode).item())
        s1 = float(_readout_scalar(model.head(model.node_states(b1, flat_clean.reshape(1, n, n, K))), b1, mode).item())
    return c.cpu(), {"recon": float(c.sum().item()), "target": s1 - s0}


def _structural_ig_joint(model: nn.Module, b1: Batch, target_kind: str, mode: str, *,
                         steps: int) -> tuple[Tensor, dict[str, float]]:
    """Channel-resolved structural IG: integrate the RRWP (node diagonal or pair off-diagonal) over
    ALL walk-length channels from the per-channel graph-mean baseline, attributing per (source, r).
    Returns joint[source, channel] (sum_{j,r} telescopes to S(clean)-S(base)). The by-distance and
    by-scale marginals are joint.sum(dim=1) (per source) and joint.sum(dim=0) (per channel)."""
    n = int(b1.x.size(1))
    dev = b1.x.device
    K = int(b1.rrwp.size(-1))
    flat_clean = b1.rrwp.detach().reshape(n * n, K)
    a_idx = torch.arange(n, device=dev).repeat_interleave(n)
    b_idx = torch.arange(n, device=dev).repeat(n)
    diag = a_idx == b_idx
    active, src = (diag, a_idx) if target_kind == "node" else (~diag, b_idx)  # pair -> source = key node b
    base_flat = flat_clean.clone()
    if bool(active.any()):
        base_flat[active] = flat_clean[active].mean(dim=0)  # per-channel mean over active slots, ALL channels
    delta_flat = flat_clean - base_flat
    joint = torch.zeros(n, K, device=dev)
    for a in range(1, steps + 1):
        pt = (base_flat + (a / steps) * delta_flat).detach().requires_grad_(True)
        logits = model.head(model.node_states(b1, pt.reshape(1, n, n, K)))
        (grad,) = torch.autograd.grad(_readout_scalar(logits, b1, mode), pt)
        contrib = grad.detach() * delta_flat  # [n*n, K]
        joint.index_add_(0, src[active], contrib[active] / steps)
    with torch.no_grad():
        s0 = float(_readout_scalar(model.head(model.node_states(b1, base_flat.reshape(1, n, n, K))), b1, mode).item())
        s1 = float(_readout_scalar(model.head(model.node_states(b1, flat_clean.reshape(1, n, n, K))), b1, mode).item())
    return joint.cpu(), {"recon": float(joint.sum().item()), "target": s1 - s0}


def _swap_scale(model: nn.Module, b1: Batch, target_kind: str, mode: str) -> Tensor:
    """Finite-swap scale marginal: replace channel r (across ALL node/pair slots) with its graph-mean
    and measure the frozen-readout carriage. Returns c[K] over walk-lengths -- the swap cross-check on
    the IG scale marginal."""
    h_clean, g = _readout_gradient(model, b1, mode)
    n = int(b1.x.size(1))
    K = int(b1.rrwp.size(-1))
    dev = b1.x.device
    eye = torch.eye(n, dtype=torch.bool, device=dev)
    active = (eye if target_kind == "node" else ~eye).reshape(-1)
    mean_ch = b1.rrwp.detach().reshape(n * n, K)[active].mean(dim=0)  # [K]
    c = torch.zeros(K)
    for r_ch in range(K):
        r = b1.rrwp.clone()
        r.reshape(1, n * n, K)[0, active, r_ch] = mean_ch[r_ch]
        h = model.node_states(b1, r).detach()
        c[r_ch] = float((g * (h_clean - h)).sum().item())
    return c


def carriage_rows_for_graph(model: nn.Module, model_name: str, b1: Batch, gid: str, *,
                            start: int, ig_steps: int, replacement: str, donors: int,
                            min_distance: int, rng: np.random.Generator,
                            completeness_out: list[dict] | None = None) -> list[dict]:
    """All six factors x two modes for one rank-1 graph (carrier = the single query node)."""
    qm = b1.qmask[0].cpu().numpy()
    qs = np.nonzero(qm)[0]
    if len(qs) != 1:
        return []
    q = int(qs[0])
    adj = b1.adj[0].cpu().numpy()
    d_from_q = _bfs_distances(adj, q)
    tgt = int(b1.target_idx[0, q].item())
    rows: list[dict] = []

    def emit(factor: str, mode: str, c: Tensor) -> None:
        for j in range(int(c.numel())):
            if j == q:
                continue
            d = int(d_from_q[j])
            if d < min_distance or d < 0:
                continue
            v = float(c[j].item())
            rows.append({
                "model": model_name, "graph_id": gid, "carrier": q, "source": j, "distance": d,
                "scale": None, "is_target": int(j == tgt), "factor": factor, "mode": mode,
                "effect_signed": v, "effect_abs": abs(v),
            })

    def emit_scale(factor: str, mode: str, c: Tensor) -> None:
        # by walk-length marginal: one row per channel r (aggregated over all sources).
        for r_ch in range(int(c.numel())):
            v = float(c[r_ch].item())
            rows.append({
                "model": model_name, "graph_id": gid, "carrier": q, "source": None, "distance": None,
                "scale": int(r_ch), "is_target": 0, "factor": factor, "mode": mode,
                "effect_signed": v, "effect_abs": abs(v),
            })

    def emit_joint(factor: str, mode: str, joint: Tensor) -> None:
        # (distance x scale): one row per (source j -> distance d, channel r).
        for j in range(int(joint.size(0))):
            if j == q:
                continue
            d = int(d_from_q[j])
            if d < min_distance or d < 0:
                continue
            for r_ch in range(int(joint.size(1))):
                v = float(joint[j, r_ch].item())
                rows.append({
                    "model": model_name, "graph_id": gid, "carrier": q, "source": j, "distance": d,
                    "scale": int(r_ch), "is_target": int(j == tgt), "factor": factor, "mode": mode,
                    "effect_signed": v, "effect_abs": abs(v),
                })

    for mode in CARRIAGE_MODES:
        for factor in SWAP_FACTORS:
            emit(factor, mode, _swap_carriage(model, b1, factor, mode, start=start,
                                              replacement=replacement, donors=donors, rng=rng))
        for factor in IG_FACTORS:
            c, comp = _ig_carriage(model, b1, factor, mode, start=start, steps=ig_steps)
            emit(factor, mode, c)
            if completeness_out is not None and math.isfinite(comp["target"]):
                completeness_out.append({"model": model_name, "graph_id": gid, "factor": factor,
                                         "mode": mode, **comp, "abs_error": abs(comp["recon"] - comp["target"])})
        # structural RRWP by walk-length (scale) + (distance x scale) joint -- node and pair
        for kind, _base in STRUCTURAL_KINDS:
            joint, comp = _structural_ig_joint(model, b1, kind, mode, steps=ig_steps)  # [n, K]
            emit_scale(SCALE_IG_FACTORS[kind], mode, joint.sum(dim=0))                  # by walk-length (IG)
            emit_joint(JOINT_IG_FACTORS[kind], mode, joint)                             # distance x scale (IG)
            emit_scale(SCALE_SWAP_FACTORS[kind], mode, _swap_scale(model, b1, kind, mode))  # by walk-length (swap)
            if completeness_out is not None and math.isfinite(comp["target"]):
                completeness_out.append({"model": model_name, "graph_id": gid, "factor": JOINT_IG_FACTORS[kind],
                                         "mode": mode, **comp, "abs_error": abs(comp["recon"] - comp["target"])})
    return rows


def ensure_colab_drive_out_dir(out_dir: Path, *, subdir: str) -> Path:
    """In Colab: mount Drive and redirect a local/ephemeral out-dir onto Drive so EVERYTHING (models,
    carriage caches, figures) survives runtime restarts. No-op outside Colab or if already on Drive."""
    try:
        import google.colab  # type: ignore  # noqa: F401
    except Exception:
        return out_dir
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[drive] mount failed ({exc}); using {out_dir} (NOT persistent!)", flush=True)
        return out_dir
    if str(out_dir).startswith("/content/drive"):
        return out_dir
    new = Path("/content/drive/MyDrive/graph_specialisation_metrics") / subdir
    print(f"[drive] redirecting out-dir to Drive for persistence: {out_dir} -> {new}", flush=True)
    return new


def _carriage_cell_fingerprint(cfg: dict, graph: str, model_name: str, td: int) -> str:
    import hashlib

    keys = ("n", "key_vocab", "value_vocab", "rrwp_steps", "degree", "bridge_edges",
            "carriage_graphs_count", "carriage_channel_start", "carriage_ig_steps",
            "carriage_rrwp_replacement", "carriage_donor_samples", "carriage_min_distance")
    payload = {"graph": graph, "model": model_name, "td": int(td), **{k: cfg.get(k) for k in keys}}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def compute_carriage_rows(model: nn.Module, model_name: str, graph: str, cfg: dict, td: int,
                          device: torch.device, *, cache_dir: Path | None = None,
                          force: bool = False) -> tuple[list[dict], list[dict], bool]:
    """Per-(graph, model) carriage rows, cached to ``cache_dir`` (Drive). Resumable at PROBE-GRAPH
    granularity: each probe graph's rows are written the moment it finishes (under a per-cell
    ``parts/`` dir), so an interrupt/restart mid-cell reloads finished graphs and only computes the
    remaining ones. Returns (rows, completeness, fully_from_cache)."""
    path = parts_dir = None
    if cache_dir is not None:
        fp = _carriage_cell_fingerprint(cfg, graph, model_name, td)
        path = Path(cache_dir) / f"{graph}__{model_name}__{fp}.json"
        parts_dir = Path(cache_dir) / f"{graph}__{model_name}__{fp}__parts"
        if path.exists() and not force:  # fast path: whole cell already finished
            d = json.loads(path.read_text())
            return d["rows"], d["completeness"], True
    start = int(cfg.get("carriage_channel_start", 2))
    ig_steps = int(cfg.get("carriage_ig_steps", 24))
    replacement = str(cfg.get("carriage_rrwp_replacement", "donor"))
    donors = int(cfg.get("carriage_donor_samples", 4))
    min_distance = int(cfg.get("carriage_min_distance", 1))
    n_graphs = int(cfg.get("carriage_graphs_count", 12))
    probe = make_batch(n_graphs, graph=graph, n=cfg["n"], rank=1, key_vocab=cfg["key_vocab"],
                       value_vocab=cfg["value_vocab"], rrwp_steps=cfg["rrwp_steps"],
                       bridge_edges=cfg["bridge_edges"], degree=cfg["degree"], seed=90210,
                       device=device, addressing="content", target_distance=td)
    rows: list[dict] = []
    completeness: list[dict] = []
    computed_any = False
    t0 = time.time()
    print(f"    [compute {graph}/{model_name}] carriage over {n_graphs} probe graphs "
          f"(content/node/pair x swap+IG + scale/joint, functional+beneficial, ig_steps={ig_steps}, "
          f"donors={donors})...", flush=True)
    for gi in range(n_graphs):
        gt = time.time()
        gpart = (parts_dir / f"graph_{gi:04d}.json") if parts_dir is not None else None
        if gpart is not None and gpart.exists() and not force:  # resume: this probe graph is done
            d = json.loads(gpart.read_text())
            cell, gcomp = d["rows"], d["completeness"]
            src = "cache"
        else:
            gcomp: list[dict] = []
            # per-graph RNG seed -> each probe graph is deterministic regardless of resume order
            cell = carriage_rows_for_graph(model, model_name, _slice_graph(probe, gi), f"{graph}:{gi}",
                                           start=start, ig_steps=ig_steps, replacement=replacement,
                                           donors=donors, min_distance=min_distance,
                                           rng=np.random.default_rng(1234 + gi), completeness_out=gcomp)
            for r in cell:
                r["graph"] = graph
            if gpart is not None:
                gpart.parent.mkdir(parents=True, exist_ok=True)
                gpart.write_text(json.dumps({"rows": cell, "completeness": gcomp}))
            computed_any = True
            src = "computed"
        rows.extend(cell)
        completeness.extend(gcomp)
        done = gi + 1
        elapsed = time.time() - t0
        eta = elapsed / max(done, 1) * (n_graphs - done)
        print(f"      graph {done}/{n_graphs} ({src}, {time.time()-gt:.1f}s, {len(cell)} rows) | "
              f"total {len(rows)} rows, {elapsed:.0f}s elapsed, ~{eta:.0f}s left", flush=True)
    print(f"    [compute {graph}/{model_name}] done: {len(rows)} rows in {time.time()-t0:.0f}s", flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"rows": rows, "completeness": completeness}))
        if parts_dir is not None and parts_dir.exists():  # cell finished -> parts are redundant
            try:
                for p in parts_dir.glob("graph_*.json"):
                    p.unlink()
                parts_dir.rmdir()
            except OSError:
                pass
    return rows, completeness, not computed_any


def collect_carriage(cfg: dict, device: torch.device, out_dir: Path, *, graphs: Sequence[str],
                     models: Sequence[str], td: int, train_fn, ckpt_dir: Path | None,
                     force_retrain: bool, force_carriage: bool,
                     load_only: bool = False) -> tuple[list[dict], list[dict], list[dict]]:
    """Shared model-agnostic carriage driver: per cell, load-or-train the model (ckpt cache) then
    load-or-compute its carriage rows (carriage cache). ``train_fn`` supplies the model backend.
    ``load_only`` (analyze phase) requires the model to already be cached on Drive."""
    cache_dir = Path(out_dir) / "carriage_cache"
    rows: list[dict] = []
    completeness: list[dict] = []
    accs: list[dict] = []
    for graph in graphs:
        run_cfg = {**cfg, "rank": 1, "addressing": "content", "target_distance": td}
        for model_name in models:
            model, res = train_fn(model_name=model_name, graph=graph, cfg=run_cfg, device=device,
                                  seed=0, ckpt_dir=ckpt_dir, force_retrain=force_retrain,
                                  load_only=load_only)
            model.eval()
            accs.append({"graph": graph, "model": model_name, "val_acc": res.get("val_acc"),
                         "train_acc": res.get("train_acc"),
                         "source": "cache" if res.get("loaded_from_cache") else "trained"})
            cell_rows, cell_comp, cached = compute_carriage_rows(
                model, model_name, graph, cfg, td, device, cache_dir=cache_dir, force=force_carriage)
            rows.extend(cell_rows)
            completeness.extend(cell_comp)
            print(f"  [carriage {graph:13s} {model_name:11s}] val={res.get('val_acc'):.3f} "
                  f"train={res.get('train_acc', float('nan')):.3f} "
                  f"model={'cache' if res.get('loaded_from_cache') else 'trained'} "
                  f"carriage={'cache' if cached else 'computed'} ({len(cell_rows)} rows)", flush=True)
    return rows, completeness, accs


def train_carriage_models(cfg: dict, device: torch.device, ckpt_dir: Path, *, train_fn,
                          force_retrain: bool = False) -> int:
    """Train + cache (to Drive) every model the carriage suite will need (seed 0, rank 1, content,
    at the planted target distance). Used by the ``--phase train`` GPU pass."""
    td = int(cfg.get("carriage_target_distance") or cfg.get("target_distance") or 3)
    graphs = list(cfg.get("carriage_graphs") or cfg.get("graphs") or ["dumbbell"])
    n = 0
    for graph in graphs:
        run_cfg = {**cfg, "rank": 1, "addressing": "content", "target_distance": td}
        for model_name in cfg["models"]:
            _, res = train_fn(model_name=model_name, graph=graph, cfg=run_cfg, device=device,
                              seed=0, ckpt_dir=ckpt_dir, force_retrain=force_retrain)
            n += 1
            print(f"  [train carriage {graph:13s} {model_name:11s}] val={res.get('val_acc'):.3f} "
                  f"({'cache' if res.get('loaded_from_cache') else 'trained'})", flush=True)
    return n


def run_carriage_suite(cfg: dict, device: torch.device, out_dir: Path, *,
                       ckpt_dir: Path | None = None, force_retrain: bool = False,
                       force_carriage: bool = False, load_only: bool = False) -> dict:
    """Train (cached) the models on each carriage graph at the planted target distance, then compute
    the full symbolic/structural x swap/IG x functional/beneficial carriage suite and render figures.
    Both the trained models and the per-cell carriage rows are cached under ``out_dir`` (Drive), so a
    restarted runtime reloads everything and only regenerates the (cheap) figures. ``load_only``
    (analyze phase) requires the models to already be trained on Drive."""
    td = int(cfg.get("carriage_target_distance") or cfg.get("target_distance") or 3)
    graphs = list(cfg.get("carriage_graphs") or cfg.get("graphs") or ["dumbbell"])
    models = list(cfg.get("models") or ["dense", "1hop"])
    start = int(cfg.get("carriage_channel_start", 2))
    ig_steps = int(cfg.get("carriage_ig_steps", 24))
    replacement = str(cfg.get("carriage_rrwp_replacement", "donor"))
    donors = int(cfg.get("carriage_donor_samples", 4))
    n_graphs = int(cfg.get("carriage_graphs_count", 12))

    rows, completeness, accs = collect_carriage(
        cfg, device, out_dir, graphs=graphs, models=models, td=td, train_fn=train_one,
        ckpt_dir=ckpt_dir, force_retrain=force_retrain, force_carriage=force_carriage,
        load_only=load_only,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": {"target_distance": td, "graphs": graphs, "models": models,
                          "channel_start": start, "ig_steps": ig_steps, "replacement": replacement,
                          "donor_samples": donors, "n_graphs": n_graphs},
               "rows": rows, "completeness": completeness, "accuracy": accs}
    (out_dir / "carriage_rows.json").write_text(json.dumps(payload, indent=2))
    figs = plot_carriage_suite(rows, accs, td, out_dir)
    print(f"[carriage] wrote {out_dir/'carriage_rows.json'} ({len(rows)} rows); figures: {len(figs)}", flush=True)
    return {"rows": rows, "completeness": completeness, "accuracy": accs, "figures": figs,
            "out_dir": str(out_dir), "target_distance": td}


def run_positive_control(cfg: dict, device: torch.device, out_dir: Path,
                         ckpt_dir: Path | None = None, force_retrain: bool = False) -> dict:
    """Back-compat entry: the beneficial-carriage-at-target headline is now part of the full suite."""
    return run_carriage_suite(cfg, device, out_dir, ckpt_dir=ckpt_dir, force_retrain=force_retrain)


_MODEL_STYLE = {
    "dense": dict(marker="o", color="#1f77b4"),
    "1hop": dict(marker="s", color="#d62728"),
    "1hop_vnode": dict(marker="^", color="#2ca02c"),
}


def _profile(rows: Sequence[dict], *, factor: str, mode: str, model: str, graph: str,
             value: str = "effect_abs", normalise: bool = False) -> tuple[list[int], list[float]]:
    """Mean carriage by distance for one (factor, mode, model, graph); optionally per-graph-normalised."""
    sub = [r for r in rows if r["factor"] == factor and r["mode"] == mode
           and r["model"] == model and r["graph"] == graph]
    if not sub:
        return [], []
    if normalise:
        totals: dict[str, float] = {}
        for r in sub:
            totals[r["graph_id"]] = totals.get(r["graph_id"], 0.0) + abs(r["effect_abs"])
        by_d: dict[int, list[float]] = {}
        for r in sub:
            denom = max(totals.get(r["graph_id"], 0.0), 1e-12)
            by_d.setdefault(int(r["distance"]), []).append(abs(r["effect_abs"]) / denom)
    else:
        by_d = {}
        for r in sub:
            by_d.setdefault(int(r["distance"]), []).append(float(r[value]))
    ds = sorted(by_d)
    return ds, [float(np.mean(by_d[d])) for d in ds]


def _profile_spread(rows: Sequence[dict], *, factor: str, mode: str, model: str, graph: str,
                    value: str) -> tuple[list[int], list[float], list[float]]:
    """Mean carriage by distance + standard error, over the distance-marginal rows only (excludes the
    scale/joint rows which also carry a distance)."""
    by_d: dict[int, list[float]] = {}
    for r in rows:
        if (r["factor"] == factor and r["mode"] == mode and r["model"] == model and r["graph"] == graph
                and r.get("scale") is None and r.get("distance") is not None):
            by_d.setdefault(int(r["distance"]), []).append(float(r[value]))
    ds = sorted(by_d)
    means = [float(np.mean(by_d[d])) for d in ds]
    sems = [float(np.std(by_d[d]) / max(len(by_d[d]) ** 0.5, 1.0)) for d in ds]
    return ds, means, sems


def _scale_profile(rows: Sequence[dict], *, factor: str, mode: str, model: str, graph: str,
                   value: str = "effect_abs") -> tuple[list[int], list[float]]:
    """Mean carriage by walk-length r for one (factor, mode, model, graph)."""
    sub = [r for r in rows if r["factor"] == factor and r["mode"] == mode and r["model"] == model
           and r["graph"] == graph and r.get("scale") is not None]
    by_r: dict[int, list[float]] = {}
    for r in sub:
        by_r.setdefault(int(r["scale"]), []).append(float(r[value]))
    rs = sorted(by_r)
    return rs, [float(np.mean(by_r[x])) for x in rs]


def _joint_grid(rows: Sequence[dict], *, factor: str, mode: str, model: str, graph: str,
                value: str = "effect_abs") -> tuple[Any, list[int], list[int]]:
    """(distance x walk-length) mean-carriage grid for one (factor, mode, model, graph)."""
    sub = [r for r in rows if r["factor"] == factor and r["mode"] == mode and r["model"] == model
           and r["graph"] == graph and r.get("scale") is not None and r.get("distance") is not None]
    if not sub:
        return None, [], []
    ds = sorted({int(r["distance"]) for r in sub})
    rs = sorted({int(r["scale"]) for r in sub})
    di = {d: i for i, d in enumerate(ds)}
    ri = {x: i for i, x in enumerate(rs)}
    acc = np.zeros((len(ds), len(rs)))
    cnt = np.zeros((len(ds), len(rs)))
    for r in sub:
        i, j = di[int(r["distance"])], ri[int(r["scale"])]
        acc[i, j] += float(r[value])
        cnt[i, j] += 1.0
    return acc / np.maximum(cnt, 1.0), ds, rs


def plot_carriage_suite(rows: Sequence[dict], accs: Sequence[dict], target_distance: int,
                        out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return []
    graphs = sorted({r["graph"] for r in rows})
    models = [m for m in ("dense", "1hop", "1hop_vnode") if any(r["model"] == m for r in rows)]
    primary = "dumbbell" if "dumbbell" in graphs else graphs[0]
    figs: list[str] = []

    def save(fig, name: str) -> None:
        p = out_dir / name
        fig.savefig(p, dpi=150, bbox_inches="tight")
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        figs.append(str(p))

    panels = [("content_ig", "Content (IG)"), ("node_rrwp_ig", "Node RRWP (IG)"),
              ("pair_rrwp_ig", "Pair RRWP (IG)"), ("content", "Content (swap)"),
              ("node_rrwp", "Node RRWP (swap)"), ("pair_rrwp", "Pair RRWP (swap)")]

    # (0) HEADLINE per task (graph): functional vs beneficial carriage across distance, for the three
    #     substrates (symbolic content / node-RRWP / pair-RRWP), lines per model. One figure per task.
    substrates = [("content_ig", "Symbolic (content)"), ("node_rrwp_ig", "Node RRWP"),
                  ("pair_rrwp_ig", "Pair RRWP")]
    modes = [("functional", "|carriage|  (functional)", "effect_abs"),
             ("beneficial", "beneficial carriage  (<0 = helps task)", "effect_signed")]
    for graph in graphs:
        fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.0), constrained_layout=True)
        acc_txt = "  ".join(f"{a['model'].split('_')[0]}={a['val_acc']:.2f}" for a in accs if a["graph"] == graph)
        for ri_, (factor, sub_title) in enumerate(substrates):
            for ci_, (mode, ylabel, value) in enumerate(modes):
                ax = axes[ri_][ci_]
                for model in models:
                    style = _MODEL_STYLE.get(model, {})
                    ds, ys, es = _profile_spread(rows, factor=factor, mode=mode, model=model, graph=graph, value=value)
                    if not ds:
                        continue
                    ax.plot(ds, ys, label=model, markersize=6, linewidth=1.9, **style)
                    lo = [y - e for y, e in zip(ys, es)]
                    hi = [y + e for y, e in zip(ys, es)]
                    ax.fill_between(ds, lo, hi, color=style.get("color", "gray"), alpha=0.15, linewidth=0)
                ax.axvline(target_distance, color="k", ls=":", lw=0.9)
                if mode == "beneficial":
                    ax.axhline(0, color="gray", ls=":", lw=0.9)
                ax.set_title(f"{sub_title} — {mode}", fontsize=11)
                ax.set_xlabel("query→source distance (hops)")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.3)
                ax.legend(frameon=False, fontsize=9)
        fig.suptitle(f"Functional vs beneficial carriage by distance — task: {graph}\n"
                     f"(content retrieval; dotted = planted target d={target_distance}; val acc  {acc_txt})",
                     fontsize=13)
        save(fig, f"carriage_func_vs_benef__{graph}.png")

    # (1) Functional carriage by distance on the bottleneck graph -- the substrate decomposition.
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    for ax, (factor, title) in zip(axes.reshape(-1), panels):
        for model in models:
            ds, ys = _profile(rows, factor=factor, mode="functional", model=model, graph=primary)
            if ds:
                ax.plot(ds, ys, label=model, **_MODEL_STYLE.get(model, {}))
        ax.axvline(target_distance, color="k", ls=":", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("query->source distance (hops)")
        ax.set_ylabel("mean |carriage|")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Functional carriage by distance ({primary}); dotted = planted target d={target_distance}")
    save(fig, "carriage_functional_by_distance.png")

    # (2) HEADLINE: beneficial carriage at the planted target -- dense transports, 1-hop cannot.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, (factor, title) in zip(axes, [("content_ig", "Content beneficial (IG)"),
                                          ("pair_rrwp_ig", "Pair-RRWP beneficial (IG)")]):
        for model in models:
            ds, ys = _profile(rows, factor=factor, mode="beneficial", model=model, graph=primary,
                              value="effect_signed")
            if ds:
                ax.plot(ds, ys, label=model, **_MODEL_STYLE.get(model, {}))
        ax.axvline(target_distance, color="k", ls=":", lw=0.9, label=f"target d={target_distance}")
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("query->source distance (hops)")
        ax.set_ylabel("beneficial carriage (signed<0 = loss-reducing)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Beneficial far carriage where the target lives ({primary})")
    save(fig, "carriage_beneficial_at_target.png")

    # (3) IG vs swap estimator agreement (beta consistency check), per substrate, pooled over models.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (ig_f, sw_f, title) in zip(axes, [("content_ig", "content", "Content"),
                                              ("node_rrwp_ig", "node_rrwp", "Node RRWP"),
                                              ("pair_rrwp_ig", "pair_rrwp", "Pair RRWP")]):
        xs, ys, cs = [], [], []
        for model in models:
            ig_map = {(r["graph_id"], r["source"]): r["effect_signed"] for r in rows
                      if r["factor"] == ig_f and r["mode"] == "functional" and r["model"] == model and r["graph"] == primary}
            sw_map = {(r["graph_id"], r["source"]): r["effect_signed"] for r in rows
                      if r["factor"] == sw_f and r["mode"] == "functional" and r["model"] == model and r["graph"] == primary}
            for key in ig_map.keys() & sw_map.keys():
                xs.append(sw_map[key]); ys.append(ig_map[key]); cs.append(_MODEL_STYLE.get(model, {}).get("color", "gray"))
        if xs:
            ax.scatter(xs, ys, s=10, c=cs, alpha=0.5)
            lim = max(abs(min(xs + ys)), abs(max(xs + ys)), 1e-6)
            ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.7)
            r = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")
            ax.set_title(f"{title}  (r={r:.2f}, n={len(xs)})")
        ax.set_xlabel("swap carriage"); ax.set_ylabel("IG carriage")
        ax.grid(alpha=0.3)
    fig.suptitle(f"IG vs swap agreement ({primary}, functional) -- estimator consistency [beta]")
    save(fig, "carriage_ig_vs_swap_agreement.png")

    # (4) SPARSIFICATION: far beneficial carriage by substrate across graph types (bottleneck vs
    #     expander vs wellconnected) + accuracy -- does an expander recover what the dumbbell loses?
    far_by = {}  # (graph, model, substrate) -> mean beneficial carriage at d >= target
    for r in rows:
        if r["mode"] != "beneficial" or r.get("distance") is None or int(r["distance"]) < target_distance:
            continue
        sub = {"content_ig": "content", "pair_rrwp_ig": "pair_rrwp", "node_rrwp_ig": "node_rrwp"}.get(r["factor"])
        if sub is None:
            continue
        far_by.setdefault((r["graph"], r["model"], sub), []).append(r["effect_signed"])
    substrates = ["content", "node_rrwp", "pair_rrwp"]
    fig, axes = plt.subplots(1, len(graphs), figsize=(4.6 * len(graphs), 4.6), squeeze=False, constrained_layout=True)
    for gi, graph in enumerate(graphs):
        ax = axes[0][gi]
        xpos = np.arange(len(substrates))
        w = 0.8 / max(len(models), 1)
        for mi, model in enumerate(models):
            vals = [float(np.mean(far_by.get((graph, model, s), [0.0]))) for s in substrates]
            ax.bar(xpos + mi * w, vals, w, label=model, color=_MODEL_STYLE.get(model, {}).get("color"))
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xticks(xpos + w * (len(models) - 1) / 2)
        ax.set_xticklabels(["content", "node RRWP", "pair RRWP"], fontsize=8)
        acc_txt = "  ".join(f"{a['model'].split('_')[0]}:{a['val_acc']:.2f}" for a in accs if a["graph"] == graph)
        ax.set_title(f"{graph}\nval acc {acc_txt}", fontsize=9)
        ax.set_ylabel(f"beneficial carriage (d>={target_distance}, <0=good)")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Far beneficial carriage by substrate across graph types: bottleneck vs expander control")
    save(fig, "carriage_sparsification_by_graph.png")

    # (5) Accuracy guardrail: train vs val (a low val with high train = routing failure, not undertraining).
    fig, ax = plt.subplots(figsize=(1.9 * max(len(accs), 3), 4.2), constrained_layout=True)
    labels = [f"{a['graph'][:4]}/{a['model'].split('_')[0]}" for a in accs]
    xpos = np.arange(len(accs))
    ax.bar(xpos - 0.2, [a.get("train_acc", float("nan")) for a in accs], 0.4, label="train", color="#999999")
    ax.bar(xpos + 0.2, [a.get("val_acc", float("nan")) for a in accs], 0.4, label="val", color="#1f77b4")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("retrieval accuracy"); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.set_title("Train vs val accuracy (train>>val = routing failure, not undertraining)")
    save(fig, "carriage_accuracy_guardrail.png")

    # (6) RRWP carriage by WALK-LENGTH r (scale marginal, aggregated over sources): which structural
    #     scales the focal node collectively uses. High-r mass = global structure local-RRWP truncates.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    for col, (kind, ktitle) in enumerate([("node", "Node RRWP"), ("pair", "Pair RRWP")]):
        for ri_, (mode, val) in enumerate([("functional", "effect_abs"), ("beneficial", "effect_signed")]):
            ax = axes[ri_][col]
            for model in models:
                color = _MODEL_STYLE.get(model, {}).get("color")
                rs, ys = _scale_profile(rows, factor=SCALE_IG_FACTORS[kind], mode=mode, model=model, graph=primary, value=val)
                if rs:
                    ax.plot(rs, ys, marker="o", label=f"{model} (IG)", color=color)
                rs2, ys2 = _scale_profile(rows, factor=SCALE_SWAP_FACTORS[kind], mode=mode, model=model, graph=primary, value=val)
                if rs2:
                    ax.plot(rs2, ys2, marker="x", ls="--", alpha=0.55, color=color, label=f"{model} (swap)")
            ax.axhline(0, color="gray", ls=":", lw=0.7)
            ax.set_title(f"{ktitle} -- {mode}")
            ax.set_xlabel("RRWP walk-length r (structural scale)")
            ax.set_ylabel("|carriage|" if mode == "functional" else "beneficial (signed<0=good)")
            ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"RRWP carriage by walk-length ({primary}); solid=IG, dashed=swap cross-check")
    save(fig, "carriage_rrwp_by_scale.png")

    # (7) Node-RRWP (distance x walk-length) JOINT heatmap, functional, per model on the primary graph.
    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * max(len(models), 1), 4.4), squeeze=False, constrained_layout=True)
    for mi, model in enumerate(models):
        ax = axes[0][mi]
        grid, ds, rs = _joint_grid(rows, factor=JOINT_IG_FACTORS["node"], mode="functional", model=model, graph=primary)
        if grid is not None:
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(rs))); ax.set_xticklabels(rs)
            ax.set_yticks(range(len(ds))); ax.set_yticklabels(ds)
            fig.colorbar(im, ax=ax, shrink=0.85, label="mean |carriage|")
        ax.set_title(model)
        ax.set_xlabel("walk-length r")
        ax.set_ylabel("source distance d")
    fig.suptitle(f"Node-RRWP carriage joint: distance x walk-length, functional ({primary})")
    save(fig, "carriage_node_rrwp_joint.png")
    return figs


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Dense vs 1-hop attention breakaway on bottleneck retrieval.")
    ap.add_argument("--out-dir", default="experiments/synthetic/results/bottleneck_retrieval")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--positive-control", action="store_true", help="(alias for --carriage) run the carriage suite only.")
    ap.add_argument("--carriage", action="store_true", help="Run the full Step-7 symbolic/structural carriage suite (swap+IG, functional+beneficial).")
    ap.add_argument("--skip-sweep", action="store_true", help="Skip the accuracy breakaway sweep (carriage only).")
    ap.add_argument("--force-retrain", action="store_true", help="Ignore cached checkpoints and retrain.")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--ranks", type=int, nargs="+", default=None)
    ap.add_argument("--graphs", nargs="+", default=None)
    ap.add_argument("--addressings", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None, help="subset of: dense 1hop 1hop_vnode")
    ap.add_argument("--distances", type=int, nargs="+", default=None,
                    help="query->target hop distances to sweep (the reach axis); omit for the default far placement")
    ap.add_argument("--degree", type=int, default=None, help="shared cluster degree (local density)")
    ap.add_argument("--carriage-graphs", nargs="+", default=None, help="graph types for the carriage suite (e.g. dumbbell expander wellconnected)")
    ap.add_argument("--carriage-target-distance", type=int, default=None)
    ap.add_argument("--carriage-graphs-count", type=int, default=None)
    ap.add_argument("--channel-start", type=int, default=None, help="first RRWP channel perturbed (preserve self/1-hop identity below it)")
    ap.add_argument("--ig-steps", type=int, default=None)
    ap.add_argument("--rrwp-replacement", default=None, choices=["donor", "mean", "zero"])
    ap.add_argument("--donor-samples", type=int, default=None)
    ap.add_argument("--force-carriage", action="store_true", help="recompute carriage even if cached cells exist")
    ap.add_argument("--no-drive", action="store_true", help="do not auto-redirect the out-dir onto Google Drive in Colab")
    ap.add_argument("--phase", choices=["all", "train", "analyze"], default="all",
                    help="train = GPU pass, train+cache all models to Drive; analyze = load pretrained models "
                         "from Drive and compute carriage+figures (never trains); all = both in one run")
    ap.add_argument("--fast-dev-run", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    cfg = default_config()
    for key in ("n", "layers", "steps", "seeds", "ranks", "graphs", "addressings", "models", "distances", "degree"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    for arg_key, cfg_key in (("carriage_graphs", "carriage_graphs"), ("carriage_target_distance", "carriage_target_distance"),
                             ("carriage_graphs_count", "carriage_graphs_count"), ("channel_start", "carriage_channel_start"),
                             ("ig_steps", "carriage_ig_steps"), ("rrwp_replacement", "carriage_rrwp_replacement"),
                             ("donor_samples", "carriage_donor_samples")):
        if getattr(args, arg_key) is not None:
            cfg[cfg_key] = getattr(args, arg_key)
    if args.fast_dev_run:
        cfg.update({"steps": 30, "eval_batches": 2, "seeds": 1, "ranks": [1, 8], "batch_size": 16,
                    "carriage_graphs": ["dumbbell", "expander"], "carriage_graphs_count": 3,
                    "carriage_ig_steps": 6, "carriage_donor_samples": 2})

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) if args.no_drive else ensure_colab_drive_out_dir(
        Path(args.out_dir), subdir=Path(args.out_dir).name)
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    print(f"[persist] all models/carriage/figures under: {out_dir}", flush=True)

    phase = getattr(args, "phase", "all")
    cache_path = out_dir / "results.json"

    # PHASE=train: the GPU-heavy pass. Train + cache EVERY model (sweep + carriage) to Drive; no
    # analysis. Safe to interrupt/resume -- each finished model is checkpointed as it completes.
    if phase == "train":
        n_car = len(cfg.get("carriage_graphs") or cfg["graphs"]) * len(cfg["models"])
        if args.skip_sweep:
            print(f"[train] --skip-sweep: training only the {n_car} CARRIAGE models -> {ckpt_dir}", flush=True)
        else:
            n_sweep = (len(cfg["addressings"]) * len(cfg["graphs"]) * len(cfg.get("distances", [None]))
                       * len(cfg["ranks"]) * len(cfg["models"]) * cfg["seeds"])
            print(f"[train] device={device}: training {n_sweep} sweep + {n_car} carriage models -> {ckpt_dir}", flush=True)
            rows = run_sweep(cfg, device, ckpt_dir=ckpt_dir, force_retrain=args.force_retrain)
            cache_path.write_text(json.dumps({"config": cfg, "rows": rows, "device": str(device)}, indent=2))
        train_carriage_models(cfg, device, ckpt_dir, train_fn=train_one, force_retrain=args.force_retrain)
        print(f"[train] done. Run --phase analyze later to compute carriage + figures from these.", flush=True)
        return {"out_dir": str(out_dir), "phase": "train", "cache": str(cache_path)}

    load_only = phase == "analyze"  # never train in analyze; models must already be on Drive
    run_carriage = bool(getattr(args, "carriage", False) or getattr(args, "positive_control", False) or load_only)
    carriage_result: dict | None = None
    if run_carriage:
        print(f"[carriage] device={device} (load_only={load_only})", flush=True)
        carriage_result = run_carriage_suite(cfg, device, out_dir / "carriage", ckpt_dir=ckpt_dir,
                                             force_retrain=args.force_retrain,
                                             force_carriage=args.force_carriage, load_only=load_only)
    if args.skip_sweep:
        return {"out_dir": str(out_dir), "carriage": carriage_result,
                "figures": (carriage_result or {}).get("figures", [])}

    if cache_path.exists() and not args.force_retrain:
        print(f"[cache] loading existing results: {cache_path}", flush=True)
        rows = json.loads(cache_path.read_text())["rows"]
    elif load_only:
        print(f"[analyze] no results.json at {cache_path}; skipping breakaway figure "
              "(run --phase train first for the sweep).", flush=True)
        return {"out_dir": str(out_dir), "carriage": carriage_result,
                "figures": (carriage_result or {}).get("figures", [])}
    else:
        print(f"[run] device={device}  cfg={ {k: cfg[k] for k in ('n','layers','steps','seeds','ranks','graphs')} }", flush=True)
        rows = run_sweep(cfg, device, ckpt_dir=ckpt_dir, force_retrain=args.force_retrain)
        cache_path.write_text(json.dumps({"config": cfg, "rows": rows, "device": str(device)}, indent=2))
        print(f"[cache] wrote {cache_path}", flush=True)

    fig_path = plot_breakaway(rows, out_dir / "breakaway.png")
    print(f"[figure] wrote {fig_path}", flush=True)

    # headline: model accuracies across the swept axis for each (addressing, graph)
    has_dist = any(r.get("distance") is not None for r in rows)
    xkey = "distance" if has_dist else "rank"
    xs = sorted({r[xkey] for r in rows if r.get(xkey) is not None})

    def acc(addressing: str, graph: str, model: str, xval) -> float:
        r = next((r for r in rows if r.get("addressing", "content") == addressing and r["graph"] == graph
                  and r["model"] == model and r.get(xkey) == xval), None)
        return r["val_acc_mean"] if r else float("nan")

    print(f"\n[headline] accuracy by model (x-axis = {xkey}):")
    for addressing in cfg["addressings"]:
        for graph in cfg["graphs"]:
            cells = [f"{xkey}={xval} [" + " ".join(f"{m}={acc(addressing, graph, m, xval):.2f}" for m in cfg["models"]) + "]"
                     for xval in xs]
            print(f"  {addressing:10s}/{graph:13s}  " + "   ".join(cells))

    return {"out_dir": str(out_dir), "cache": str(cache_path), "figure": str(fig_path),
            "rows": rows, "carriage": carriage_result}


if __name__ == "__main__":
    main()
