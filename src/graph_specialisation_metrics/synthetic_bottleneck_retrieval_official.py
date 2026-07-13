"""Official-GRIT variant of the bottleneck-retrieval synthetic (dense GRIT + masked 1-hop GRIT).

Parallel to ``synthetic_bottleneck_retrieval.py`` (the pure-torch stand-in). This file keeps the
task, the Step-7 carriage suite, the caching, and the figures IDENTICAL (imported from that module)
and swaps ONLY the model for the OFFICIAL LiamMa/GRIT ``GritTransformerLayer`` used by the
peptides-struct / ZINC training files:

  * ``dense`` : official GRIT attention over all node pairs (full RRWP support).
  * ``1hop``  : the parameter-matched masked control -- attention restricted to graph edges + self
                (the ``full_attn=False, sparsity=one_hop`` / ``pad_to_full_graph=False`` intervention),
                with global RRWP values retained on those edges and as node-RRWP features.

Only the ATTENTION LAYERS come from official GRIT; the node/edge RRWP linear encoders are the same
Linear(rrwp_steps, dim) maps the official GRIT/Sparse1HopGRIT models use, kept local so the carriage
probes can integrate/perturb the RRWP substrate cleanly (mirrors ``Sparse1HopGRIT`` construction but
substitutes the official layer class and adds the dense/1-hop edge-set toggle).

REQUIRES the official GRIT repository on ``GRIT_ROOT`` (or ``external/GRIT``) plus ``torch_geometric``.
It therefore runs on Colab/HPC only, NOT in a bare local environment -- model construction raises a
clear ImportError otherwise. Everything except the model instantiation is exercised by the local
structure smoke test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_specialisation_metrics import synthetic_bottleneck_retrieval as base
from graph_specialisation_metrics.synthetic_bottleneck_retrieval import (
    Batch,
    _bfs_distances,
    _ckpt_fingerprint,
    _loss_and_acc,
    _slice_graph,
    carriage_rows_for_graph,
    collect_carriage,
    default_config,
    ensure_colab_drive_out_dir,
    make_batch,
    plot_breakaway,
    plot_carriage_suite,
)
from graph_specialisation_metrics.counterfactual_interchange_mediation import (
    add_external_repo_path,
    grit_layer_cfg,
    require_import,
)

Tensor = torch.Tensor

# dense vs the masked 1-hop control -- the single scientific intervention (matches the peptides-struct
# ``full_attn=False, sparsity=one_hop`` / ``pad_to_full_graph=False`` masked RRWP edge encoder).
MODEL_SPECS = {"dense": dict(dense=True), "1hop": dict(dense=False)}


# ======================================================================================
# Official-GRIT retrieval model. Owns encode/propagate (so the carriage suite can integrate over the
# content and RRWP substrates) and uses the official GritTransformerLayer for attention.
# ======================================================================================
def _edge_set(b1: Batch, dense: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Edge support for one graph: self-loops + (all pairs if dense else graph edges).

    Returns local source/dest indices, an edge-type id (0=self, 1=pair) and a matching count.
    Attention runs over exactly these edges -- so ``dense`` vs ``1hop`` IS the support intervention.
    """
    n = int(b1.adj.size(1))
    dev = b1.adj.device
    idx = torch.arange(n, device=dev)
    src = [idx]
    dst = [idx]
    etype = [torch.zeros(n, dtype=torch.long, device=dev)]
    if dense:
        a = idx.repeat_interleave(n)
        c = idx.repeat(n)
        keep = a != c
        a, c = a[keep], c[keep]
    else:
        ei = (b1.adj[0] > 0).nonzero(as_tuple=False)
        a, c = ei[:, 0], ei[:, 1]
    src.append(a)
    dst.append(c)
    etype.append(torch.ones(a.numel(), dtype=torch.long, device=dev))
    return torch.cat(src), torch.cat(dst), torch.cat(etype), torch.tensor([n], device=dev)


class OfficialGRITRetriever(nn.Module):
    """Single-graph (B=1) official-GRIT retriever exposing the carriage-suite hook interface."""

    def __init__(self, *, in_dim: int, value_vocab: int, dim: int = 96, heads: int = 8,
                 layers: int = 4, rrwp_steps: int = 8, dropout: float = 0.0,
                 attn_dropout: float = 0.2, dense: bool = True) -> None:
        super().__init__()
        add_external_repo_path("GRIT_ROOT", ("GRIT", "external/GRIT"))
        grit_layer_mod = require_import("grit.layer.grit_layer", "official GRIT repository")
        self.data_mod = require_import("torch_geometric.data", "torch_geometric")
        self.dense = bool(dense)
        self.dim = int(dim)
        self.rrwp_steps = int(rrwp_steps)
        self.encoder = nn.Linear(int(in_dim), dim)
        self.rrwp_abs_encoder = nn.Linear(rrwp_steps, dim, bias=False)   # node RRWP (diagonal)
        self.rrwp_rel_encoder = nn.Linear(rrwp_steps, dim, bias=False)   # pair RRWP (off-diagonal, on edges)
        self.edge_type_encoder = nn.Embedding(2, dim)                     # 0=self-loop, 1=pair edge
        nn.init.xavier_uniform_(self.rrwp_abs_encoder.weight)
        nn.init.xavier_uniform_(self.rrwp_rel_encoder.weight)
        lcfg = grit_layer_cfg(update_e=True)
        self.layers = nn.ModuleList(
            grit_layer_mod.GritTransformerLayer(
                dim, dim, heads, dropout=dropout, attn_dropout=attn_dropout,
                layer_norm=False, batch_norm=True, residual=True, act="relu",
                norm_e=True, O_e=True, cfg=lcfg,
            )
            for _ in range(int(layers))
        )
        self.head = nn.Linear(dim, int(value_vocab))

    def encode(self, b1: Batch, rrwp: Tensor | None = None) -> Tensor:
        """h0 = content embedding + node-RRWP (diagonal). Shape [1, n, dim]. IG resample unit."""
        r = b1.rrwp if rrwp is None else rrwp
        n = int(b1.x.size(1))
        node_rrwp = r[0, torch.arange(n), torch.arange(n)]           # [n, rrwp_steps]
        return (self.encoder(b1.x[0]) + self.rrwp_abs_encoder(node_rrwp)).unsqueeze(0)

    def propagate(self, b1: Batch, h0: Tensor, rrwp: Tensor | None = None) -> Tensor:
        """Run the official GRIT attention stack from a given h0 over the (dense or 1-hop) edge set."""
        r = b1.rrwp if rrwp is None else rrwp
        n = int(b1.x.size(1))
        src, dst, etype, _ = _edge_set(b1, self.dense)
        edge_index = torch.stack([src, dst], dim=0)
        edge_rrwp = r[0, src, dst]                                    # [E, rrwp_steps]
        edge_attr = self.edge_type_encoder(etype) + self.rrwp_rel_encoder(edge_rrwp)
        deg = torch.zeros(n, device=h0.device)
        deg.index_add_(0, dst, torch.ones(dst.numel(), device=h0.device))
        data = self.data_mod.Data(num_nodes=n)
        data.x = h0[0]
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.log_deg = torch.log(deg + 1.0)
        data.deg = deg
        data.batch = torch.zeros(n, dtype=torch.long, device=h0.device)
        for layer in self.layers:
            data = layer(data)
        return data.x.unsqueeze(0)

    def node_states(self, b1: Batch, rrwp: Tensor | None = None) -> Tensor:
        return self.propagate(b1, self.encode(b1, rrwp), rrwp)

    def forward(self, batch: Batch) -> Tensor:
        # Training path: iterate graphs in the batch (GRIT layers run per single graph here).
        outs = [self.head(self.node_states(_slice_graph(batch, gi))[0]) for gi in range(batch.x.size(0))]
        return torch.stack(outs, dim=0)


def build_model(model_name: str, cfg: dict, device: torch.device) -> OfficialGRITRetriever:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"official-GRIT variant supports {sorted(MODEL_SPECS)}, got {model_name!r} "
                         "(1hop_vnode global-token control is not ported here yet)")
    in_dim = 2 * cfg["key_vocab"] + cfg["value_vocab"] + 2
    return OfficialGRITRetriever(
        in_dim=in_dim, value_vocab=cfg["value_vocab"], dim=cfg["dim"], heads=cfg["heads"],
        layers=cfg["layers"], rrwp_steps=cfg["rrwp_steps"], dropout=cfg["dropout"],
        **MODEL_SPECS[model_name],
    ).to(device)


# ======================================================================================
# Training / caching (copied from the pure-torch runner; only model construction differs, so the
# official variant is a self-contained, revertible file).
# ======================================================================================
def train_one(*, model_name: str, graph: str, cfg: dict, device: torch.device, seed: int,
              ckpt_dir: Path | None = None, force_retrain: bool = False) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    model = build_model(model_name, cfg, device)
    ckpt_path = None
    if ckpt_dir is not None:
        fp = _ckpt_fingerprint(f"official_{model_name}", graph, cfg, seed)
        ckpt_path = Path(ckpt_dir) / (
            f"official_{model_name}__{graph}__{cfg.get('addressing')}__r{cfg.get('rank')}"
            f"__d{cfg.get('target_distance')}__s{seed}__{fp}.pt"
        )
        if ckpt_path.exists() and not force_retrain:
            payload = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            meta = dict(payload.get("meta", {}))
            meta["loaded_from_cache"] = True
            return model, meta

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sample = lambda s: make_batch(
        cfg["batch_size"], graph=graph, n=cfg["n"], rank=cfg["rank"], key_vocab=cfg["key_vocab"],
        value_vocab=cfg["value_vocab"], rrwp_steps=cfg["rrwp_steps"], bridge_edges=cfg["bridge_edges"],
        degree=cfg["degree"], seed=s, device=device, addressing=cfg["addressing"],
        target_distance=cfg.get("target_distance"),
    )
    model.train()
    log = []
    for step in range(cfg["steps"]):
        batch = sample(seed * 100003 + step)
        loss, acc, nq = _loss_and_acc(model(batch), batch)
        if nq == 0:  # no query placeable at this distance in the whole batch -> zero has no grad_fn
            continue
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, cfg["steps"] // 10) == 0 or step == cfg["steps"] - 1:
            log.append({"step": step, "loss": float(loss.item()), "train_acc": acc})
    model.eval()
    accs, train_accs = [], []
    with torch.no_grad():
        for e in range(cfg["eval_batches"]):
            batch = sample(10_000_000 + seed * 991 + e)
            _, acc, nq = _loss_and_acc(model(batch), batch)
            if nq:
                accs.append(acc)
            tb = sample(seed * 100003 + e)
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
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "meta": meta, "model_name": f"official_{model_name}",
                    "graph": graph, "seed": seed}, ckpt_path)
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
                            "addressing": addressing, "graph": graph, "distance": dist, "rank": rank,
                            "model": model_name, "val_acc_mean": float(np.nanmean(accs)),
                            "val_acc_std": float(np.nanstd(accs)), "train_acc_mean": float(np.nanmean(tr_accs)),
                            "seeds": cfg["seeds"], "params": params,
                        })
                        dstr = f"d={dist}" if dist is not None else "d=far"
                        cflag = f" [cache {n_cached}/{cfg['seeds']}]" if n_cached else ""
                        print(f"  [official {addressing:10s} {graph:13s} {dstr:6s} rank={rank:2d} {model_name:6s}] "
                              f"val={rows[-1]['val_acc_mean']:.3f}+/-{rows[-1]['val_acc_std']:.3f} "
                              f"train={rows[-1]['train_acc_mean']:.3f}{cflag}", flush=True)
    return rows


def run_carriage_suite(cfg: dict, device: torch.device, out_dir: Path, *,
                       ckpt_dir: Path | None = None, force_retrain: bool = False,
                       force_carriage: bool = False) -> dict:
    td = int(cfg.get("carriage_target_distance") or cfg.get("target_distance") or 3)
    graphs = list(cfg.get("carriage_graphs") or cfg.get("graphs") or ["dumbbell"])
    models = list(cfg.get("models") or ["dense", "1hop"])
    rows, completeness, accs = collect_carriage(
        cfg, device, out_dir, graphs=graphs, models=models, td=td, train_fn=train_one,
        ckpt_dir=ckpt_dir, force_retrain=force_retrain, force_carriage=force_carriage,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"backend": "official_grit", "config": {"target_distance": td, "graphs": graphs, "models": models,
               "channel_start": int(cfg.get("carriage_channel_start", 2)),
               "ig_steps": int(cfg.get("carriage_ig_steps", 24)),
               "replacement": str(cfg.get("carriage_rrwp_replacement", "donor")),
               "donor_samples": int(cfg.get("carriage_donor_samples", 4)),
               "n_graphs": int(cfg.get("carriage_graphs_count", 12))}, "rows": rows,
               "completeness": completeness, "accuracy": accs}
    (out_dir / "carriage_rows.json").write_text(json.dumps(payload, indent=2))
    figs = plot_carriage_suite(rows, accs, td, out_dir)
    print(f"[official carriage] wrote {out_dir/'carriage_rows.json'} ({len(rows)} rows); figures: {len(figs)}", flush=True)
    return {"rows": rows, "completeness": completeness, "accuracy": accs, "figures": figs,
            "out_dir": str(out_dir), "target_distance": td}


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Official dense/1-hop GRIT on bottleneck retrieval + carriage suite.")
    ap.add_argument("--out-dir", default="experiments/synthetic/results/bottleneck_retrieval_official")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--carriage", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--rrwp-steps", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--ranks", type=int, nargs="+", default=None)
    ap.add_argument("--graphs", nargs="+", default=None)
    ap.add_argument("--addressings", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=["dense", "1hop"], help="subset of: dense 1hop")
    ap.add_argument("--distances", type=int, nargs="+", default=None)
    ap.add_argument("--degree", type=int, default=None)
    ap.add_argument("--carriage-graphs", nargs="+", default=None)
    ap.add_argument("--carriage-target-distance", type=int, default=None)
    ap.add_argument("--carriage-graphs-count", type=int, default=None)
    ap.add_argument("--channel-start", type=int, default=None)
    ap.add_argument("--ig-steps", type=int, default=None)
    ap.add_argument("--rrwp-replacement", default=None, choices=["donor", "mean", "zero"])
    ap.add_argument("--donor-samples", type=int, default=None)
    ap.add_argument("--force-carriage", action="store_true", help="recompute carriage even if cached cells exist")
    ap.add_argument("--no-drive", action="store_true", help="do not auto-redirect the out-dir onto Google Drive in Colab")
    ap.add_argument("--fast-dev-run", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    cfg = default_config()
    cfg["dim"] = 96  # official GRIT peptides/ZINC-style width by default
    cfg["models"] = ["dense", "1hop"]
    for key in ("n", "layers", "dim", "heads", "rrwp_steps", "steps", "seeds", "ranks", "graphs",
                "addressings", "models", "distances", "degree"):
        v = getattr(args, key, None)
        if v is not None:
            cfg[key] = v
    for arg_key, cfg_key in (("carriage_graphs", "carriage_graphs"), ("carriage_target_distance", "carriage_target_distance"),
                             ("carriage_graphs_count", "carriage_graphs_count"), ("channel_start", "carriage_channel_start"),
                             ("ig_steps", "carriage_ig_steps"), ("rrwp_replacement", "carriage_rrwp_replacement"),
                             ("donor_samples", "carriage_donor_samples")):
        if getattr(args, arg_key) is not None:
            cfg[cfg_key] = getattr(args, arg_key)
    if args.fast_dev_run:
        cfg.update({"steps": 20, "eval_batches": 2, "seeds": 1, "ranks": [1], "batch_size": 8,
                    "carriage_graphs": ["dumbbell", "expander"], "carriage_graphs_count": 2,
                    "carriage_ig_steps": 4, "carriage_donor_samples": 2})

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    run_name = args.run_name or time.strftime("official_run_%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) if args.no_drive else ensure_colab_drive_out_dir(
        Path(args.out_dir), subdir=Path(args.out_dir).name)
    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    print(f"[persist] all models/carriage/figures under: {out_dir}", flush=True)

    carriage_result = None
    if args.carriage:
        print(f"[official carriage] device={device}", flush=True)
        carriage_result = run_carriage_suite(cfg, device, out_dir / "carriage", ckpt_dir=ckpt_dir,
                                             force_retrain=args.force_retrain,
                                             force_carriage=args.force_carriage)
    if args.skip_sweep:
        return {"out_dir": str(out_dir), "carriage": carriage_result,
                "figures": (carriage_result or {}).get("figures", [])}

    cache_path = out_dir / "results.json"
    if cache_path.exists() and not args.force_retrain:
        print(f"[cache] loading existing results: {cache_path}", flush=True)
        rows = json.loads(cache_path.read_text())["rows"]
    else:
        print(f"[official run] device={device}", flush=True)
        rows = run_sweep(cfg, device, ckpt_dir=ckpt_dir, force_retrain=args.force_retrain)
        cache_path.write_text(json.dumps({"config": cfg, "rows": rows, "device": str(device)}, indent=2))
        print(f"[cache] wrote {cache_path}", flush=True)

    fig_path = plot_breakaway(rows, out_dir / "breakaway.png")
    print(f"[figure] wrote {fig_path}", flush=True)
    return {"out_dir": str(out_dir), "cache": str(cache_path), "figure": str(fig_path),
            "rows": rows, "carriage": carriage_result}


if __name__ == "__main__":
    main([
        "--run-name", "official_bottleneck_carriage_v1",
        "--steps", "1500", "--seeds", "3", "--ranks", "1",
        "--distances", "1", "2", "3", "4",
        "--addressings", "content", "--graphs", "dumbbell", "wellconnected",
        "--models", "dense", "1hop",
        "--carriage", "--carriage-graphs", "dumbbell", "expander", "wellconnected",
        "--carriage-target-distance", "3", "--carriage-graphs-count", "12", "--ig-steps", "24",
    ])
