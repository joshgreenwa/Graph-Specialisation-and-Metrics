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

    def encode(self, batch: Batch) -> Tensor:
        """Post-encoder node state h0 = content embedding + node-RRWP (the IG resample unit)."""
        b, n = batch.x.shape[:2]
        node_rrwp = batch.rrwp[torch.arange(b)[:, None], torch.arange(n)[None], torch.arange(n)[None]]
        return self.encoder(batch.x) + self.rrwp_node(node_rrwp)

    def propagate(self, batch: Batch, h0: Tensor) -> Tensor:
        """Run the attention stack from a given h0 -- lets IG integrate over the encoded content."""
        b, n = h0.shape[:2]
        rrwp = batch.rrwp
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

    def node_states(self, batch: Batch) -> Tensor:
        return self.propagate(batch, self.encode(batch))

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


def train_one(
    *,
    model_name: str,
    graph: str,
    cfg: dict,
    device: torch.device,
    seed: int,
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
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sample = lambda s: make_batch(
        cfg["batch_size"], graph=graph, n=cfg["n"], rank=cfg["rank"],
        key_vocab=cfg["key_vocab"], value_vocab=cfg["value_vocab"], rrwp_steps=cfg["rrwp_steps"],
        bridge_edges=cfg["bridge_edges"], degree=cfg["degree"], seed=s, device=device,
        addressing=cfg["addressing"], target_distance=cfg.get("target_distance"),
    )
    model.train()
    log = []
    for step in range(cfg["steps"]):
        batch = sample(seed * 100003 + step)
        loss, acc, _ = _loss_and_acc(model(batch), batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, cfg["steps"] // 10) == 0 or step == cfg["steps"] - 1:
            log.append({"step": step, "loss": float(loss.item()), "train_acc": acc})
    # eval on fresh graphs
    model.eval()
    accs = []
    with torch.no_grad():
        for e in range(cfg["eval_batches"]):
            batch = sample(10_000_000 + seed * 991 + e)
            _, acc, nq = _loss_and_acc(model(batch), batch)
            if nq:
                accs.append(acc)
    return model, {"val_acc": float(np.mean(accs)) if accs else float("nan"), "train_log": log,
                   "params": int(sum(p.numel() for p in model.parameters()))}


def run_sweep(cfg: dict, device: torch.device) -> list[dict]:
    rows = []
    distances = cfg.get("distances", [None])
    for addressing in cfg["addressings"]:
        for graph in cfg["graphs"]:
            for dist in distances:
                for rank in cfg["ranks"]:
                    for model_name in cfg["models"]:
                        run_cfg = {**cfg, "rank": rank, "addressing": addressing, "target_distance": dist}
                        accs, params = [], None
                        for seed in range(cfg["seeds"]):
                            _, res = train_one(model_name=model_name, graph=graph, cfg=run_cfg, device=device, seed=seed)
                            accs.append(res["val_acc"])
                            params = res["params"]
                        rows.append({
                            "addressing": addressing, "graph": graph,
                            "distance": dist, "rank": rank, "model": model_name,
                            "val_acc_mean": float(np.nanmean(accs)), "val_acc_std": float(np.nanstd(accs)),
                            "seeds": cfg["seeds"], "params": params,
                        })
                        dstr = f"d={dist}" if dist is not None else "d=far"
                        print(f"  [{addressing:10s} {graph:13s} {dstr:6s} rank={rank:2d} {model_name:11s}] "
                              f"acc={rows[-1]['val_acc_mean']:.3f} +/- {rows[-1]['val_acc_std']:.3f}", flush=True)
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
    }


def ig_loss_carriage_decay(model: nn.Module, batch: Batch, *, steps: int = 32, max_d: int = 6) -> dict[int, float]:
    """B_IG positive control: IG attribution of the query loss to each source's encoded content,
    banded by query->source transport distance. Negative = loss-reducing (beneficial). On the
    bottleneck task the planted target's content (far) should be strongly beneficial for a model
    that can transport it (dense) and ~0 for one that cannot (1-hop)."""
    device = batch.x.device
    model.eval()
    h0 = model.encode(batch).detach()
    base = h0.mean(dim=1, keepdim=True).expand_as(h0)   # in-distribution-ish baseline: mean over nodes
    b, n, _ = h0.shape
    ig = torch.zeros(b, n, device=device)
    for a in range(1, steps + 1):
        pt = (base + (a / steps) * (h0 - base)).detach().requires_grad_(True)
        logits = model.head(model.propagate(batch, pt))
        loss = F.cross_entropy(logits[batch.qmask], batch.label[batch.qmask], reduction="sum")
        (g,) = torch.autograd.grad(loss, pt)
        ig += (g.detach() * (h0 - base)).sum(dim=-1) / steps
    adj = batch.adj.cpu().numpy()
    qm = batch.qmask.cpu().numpy()
    ig_np = ig.cpu().numpy()
    bins: dict[int, list[float]] = {}
    for gi in range(b):
        qs = np.nonzero(qm[gi])[0]
        if len(qs) != 1:  # rank-1 only, so query->source distance is unambiguous
            continue
        d_from_q = _bfs_distances(adj[gi], int(qs[0]))
        for j in range(n):
            dd = int(d_from_q[j])
            if 0 <= dd <= max_d:
                bins.setdefault(dd, []).append(float(ig_np[gi, j]))
    return {d: float(np.mean(v)) for d, v in sorted(bins.items())}


def run_positive_control(cfg: dict, device: torch.device, out_dir: Path) -> dict:
    """Train dense + 1-hop on content retrieval with the target planted at a fixed distance, then
    check B_IG(d) spikes (negative) at that distance for dense and stays ~0 for 1-hop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    td = int(cfg.get("target_distance") or 3)
    max_d = max(int(cfg.get("max_distance", 6)), td + 1)
    results: dict[str, dict[int, float]] = {}
    for model_name in ("dense", "1hop"):
        run_cfg = {**cfg, "rank": 1, "addressing": "content", "target_distance": td}
        model, res = train_one(model_name=model_name, graph="dumbbell", cfg=run_cfg, device=device, seed=0)
        batch = make_batch(cfg["batch_size"], graph="dumbbell", n=cfg["n"], rank=1,
                           key_vocab=cfg["key_vocab"], value_vocab=cfg["value_vocab"], rrwp_steps=cfg["rrwp_steps"],
                           bridge_edges=cfg["bridge_edges"], degree=cfg["degree"], seed=777, device=device,
                           addressing="content", target_distance=td)
        results[model_name] = ig_loss_carriage_decay(model, batch, steps=32, max_d=max_d)
        print(f"  [positive-control {model_name}] val_acc={res['val_acc']:.3f}  B_IG(target d={td})="
              f"{results[model_name].get(td, float('nan')):+.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for model_name, style in (("dense", dict(marker="o", color="#1f77b4")), ("1hop", dict(marker="s", color="#d62728"))):
        dec = results[model_name]
        ds = sorted(dec)
        ax.plot(ds, [dec[d] for d in ds], label=model_name, **style)
    ax.axvline(td, color="k", ls=":", lw=0.8, label=f"planted target d={td}")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("query->source transport distance (hops)")
    ax.set_ylabel("B_IG loss-carriage  (<0 = beneficial)")
    ax.set_title("B_IG positive control: beneficial far carriage where the target lives (dense only)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "positive_control_bIG.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    (out_dir / "positive_control_bIG.json").write_text(json.dumps({"target_distance": td, "results": results}, indent=2))
    print(f"[positive-control] wrote {fig_path}", flush=True)
    return {"target_distance": td, "results": results, "figure": str(fig_path)}


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Dense vs 1-hop attention breakaway on bottleneck retrieval.")
    ap.add_argument("--out-dir", default="experiments/synthetic/results/bottleneck_retrieval")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--positive-control", action="store_true", help="Run the B_IG loss-carriage positive control instead of the sweep.")
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
    ap.add_argument("--fast-dev-run", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    cfg = default_config()
    for key in ("n", "layers", "steps", "seeds", "ranks", "graphs", "addressings", "models", "distances", "degree"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.fast_dev_run:
        cfg.update({"steps": 30, "eval_batches": 2, "seeds": 1, "ranks": [1, 8], "batch_size": 16})

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "positive_control", False):
        print(f"[positive-control] device={device}", flush=True)
        return run_positive_control(cfg, device, out_dir)

    cache_path = out_dir / "results.json"
    if cache_path.exists():
        print(f"[cache] loading existing results: {cache_path}", flush=True)
        payload = json.loads(cache_path.read_text())
        rows = payload["rows"]
    else:
        print(f"[run] device={device}  cfg={ {k: cfg[k] for k in ('n','layers','steps','seeds','ranks','graphs')} }", flush=True)
        rows = run_sweep(cfg, device)
        payload = {"config": cfg, "rows": rows, "device": str(device)}
        cache_path.write_text(json.dumps(payload, indent=2))
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

    return {"out_dir": str(out_dir), "cache": str(cache_path), "figure": str(fig_path), "rows": rows}


if __name__ == "__main__":
    main()
