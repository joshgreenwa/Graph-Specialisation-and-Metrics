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
def _dumbbell_adj(n: int, bridge_edges: int, rng: np.random.Generator) -> np.ndarray:
    """Two equal cliques joined by ``bridge_edges`` bridge edges. The bridge is the bottleneck."""
    half = n // 2
    adj = np.zeros((n, n), dtype=np.float32)
    for lo, hi in ((0, half), (half, n)):
        adj[lo:hi, lo:hi] = 1.0
    np.fill_diagonal(adj, 0.0)
    a_nodes = rng.permutation(half)[:bridge_edges]
    b_nodes = half + rng.permutation(n - half)[:bridge_edges]
    for a, b in zip(a_nodes, b_nodes):
        adj[a, b] = adj[b, a] = 1.0
    return adj


def _wellconnected_adj(n: int, degree: int, rng: np.random.Generator) -> np.ndarray:
    """Random near-regular graph: low diameter, no bottleneck (the alignment control)."""
    degree = min(degree, n - 1)
    for _ in range(200):
        adj = np.zeros((n, n), dtype=np.float32)
        stubs = np.repeat(np.arange(n), degree)
        rng.shuffle(stubs)
        ok = True
        for i in range(0, len(stubs) - 1, 2):
            u, v = int(stubs[i]), int(stubs[i + 1])
            if u == v or adj[u, v]:
                ok = False
                break
            adj[u, v] = adj[v, u] = 1.0
        if ok:
            return adj
    # fallback: ring + a few chords (always connected)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1.0
    return adj


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
            adj = _dumbbell_adj(n, bridge_edges, rng)
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
            # queries in cluster A retrieve distinct targets in cluster B (content match)
            r = min(rank, half)
            q_nodes = rng.permutation(half)[:r]
            t_nodes = half + rng.permutation(n - half)[:r]
            for q, t in zip(q_nodes, t_nodes):
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
    ) -> None:
        super().__init__()
        self.dense = dense
        self.encoder = nn.Linear(in_dim, dim)
        self.rrwp_node = nn.Linear(rrwp_steps, dim)  # node RRWP = diagonal of the pair tensor
        self.layers = nn.ModuleList(
            RRWPAttentionLayer(dim, heads, rrwp_steps, dropout) for _ in range(layers)
        )
        self.head = nn.Linear(dim, value_vocab)

    def node_states(self, batch: Batch) -> Tensor:
        b, n = batch.x.shape[:2]
        node_rrwp = batch.rrwp[torch.arange(b)[:, None], torch.arange(n)[None], torch.arange(n)[None]]
        h = self.encoder(batch.x) + self.rrwp_node(node_rrwp)
        # 1-hop attends to edges + self; dense attends everywhere.
        mask = None
        if not self.dense:
            eye = torch.eye(n, device=batch.adj.device).unsqueeze(0)
            mask = ((batch.adj + eye) > 0).float()
        for layer in self.layers:
            h = layer(h, batch.rrwp, mask)
        return h

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


def train_one(
    *,
    dense: bool,
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
        dense=dense,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sample = lambda s: make_batch(
        cfg["batch_size"], graph=graph, n=cfg["n"], rank=cfg["rank"],
        key_vocab=cfg["key_vocab"], value_vocab=cfg["value_vocab"], rrwp_steps=cfg["rrwp_steps"],
        bridge_edges=cfg["bridge_edges"], degree=cfg["degree"], seed=s, device=device,
        addressing=cfg["addressing"],
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
    for addressing in cfg["addressings"]:
        for graph in cfg["graphs"]:
            for rank in cfg["ranks"]:
                for dense in (True, False):
                    run_cfg = {**cfg, "rank": rank, "addressing": addressing}
                    accs, params = [], None
                    for seed in range(cfg["seeds"]):
                        _, res = train_one(dense=dense, graph=graph, cfg=run_cfg, device=device, seed=seed)
                        accs.append(res["val_acc"])
                        params = res["params"]
                    rows.append({
                        "addressing": addressing, "graph": graph, "rank": rank,
                        "model": "dense" if dense else "1hop",
                        "val_acc_mean": float(np.nanmean(accs)), "val_acc_std": float(np.nanstd(accs)),
                        "seeds": cfg["seeds"], "params": params,
                    })
                    print(f"  [{addressing:10s} {graph:13s} rank={rank:2d} {'dense' if dense else '1hop ':5s}] "
                          f"acc={rows[-1]['val_acc_mean']:.3f} +/- {rows[-1]['val_acc_std']:.3f}", flush=True)
    return rows


def plot_breakaway(rows: list[dict], out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    addressings = sorted({r.get("addressing", "content") for r in rows})
    graphs = sorted({r["graph"] for r in rows})
    fig, axes = plt.subplots(
        len(addressings), len(graphs),
        figsize=(5.2 * len(graphs), 3.8 * len(addressings)), squeeze=False,
    )
    for ai, addressing in enumerate(addressings):
        for gi, graph in enumerate(graphs):
            ax = axes[ai][gi]
            for model, style in (("dense", dict(marker="o", color="#1f77b4")),
                                 ("1hop", dict(marker="s", color="#d62728"))):
                pts = sorted(
                    [r for r in rows if r.get("addressing", "content") == addressing
                     and r["graph"] == graph and r["model"] == model],
                    key=lambda r: r["rank"],
                )
                if not pts:
                    continue
                ax.errorbar([p["rank"] for p in pts], [p["val_acc_mean"] for p in pts],
                            yerr=[p["val_acc_std"] for p in pts], label=model, capsize=3, **style)
            expect = "deviate" if addressing == "content" else "align"
            ax.set_title(f"{addressing} / {graph}  (expect {expect})")
            ax.set_xlabel("rank  (# simultaneous retrievals)")
            ax.set_ylabel("retrieval accuracy")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle("Dense vs 1-hop attention: content addressing (deviate) vs structural (align)")
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
        "batch_size": 64, "steps": 400, "eval_batches": 8, "lr": 1e-3, "seeds": 2,
    }


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Dense vs 1-hop attention breakaway on bottleneck retrieval.")
    ap.add_argument("--out-dir", default="experiments/synthetic/results/bottleneck_retrieval")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--ranks", type=int, nargs="+", default=None)
    ap.add_argument("--graphs", nargs="+", default=None)
    ap.add_argument("--addressings", nargs="+", default=None)
    ap.add_argument("--fast-dev-run", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    cfg = default_config()
    for key in ("n", "layers", "steps", "seeds", "ranks", "graphs", "addressings"):
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

    # headline: dense-minus-1hop gap at max rank for each (addressing, graph)
    def gap(addressing: str, graph: str, rank: int) -> float:
        pick = lambda m: next((r for r in rows if r.get("addressing", "content") == addressing
                               and r["graph"] == graph and r["model"] == m and r["rank"] == rank), None)
        d, s = pick("dense"), pick("1hop")
        return (d["val_acc_mean"] - s["val_acc_mean"]) if d and s else float("nan")

    rmax = max(cfg["ranks"])
    print(f"\n[headline] dense - 1hop accuracy gap at rank={rmax}:")
    for addressing in cfg["addressings"]:
        expect = "expect a LARGE gap (deviate)" if addressing == "content" else "expect ~0 (align)"
        for graph in cfg["graphs"]:
            print(f"  {addressing:10s} / {graph:13s}: {gap(addressing, graph, rmax):+.3f}   ({expect})")

    return {"out_dir": str(out_dir), "cache": str(cache_path), "figure": str(fig_path), "rows": rows}


if __name__ == "__main__":
    main()
