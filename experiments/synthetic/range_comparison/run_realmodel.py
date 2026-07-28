"""Follow-up: does the tangent/secant gap transfer to a real trained model?

The synthetic divergences (``run_divergence.py``) were demonstrated on tasks built to break a
first-order expansion.  This runs the same comparison on a GraphGPS-style model trained on
MarkedTreePath -- a node-level task authored two months before this study, so neither the task nor
the architecture was chosen to make the point.

Both measures are based at the post-encoder state ``h0 = token_emb(x) + pe_proj(pe)``, because the
raw node input is a discrete mark through an embedding and has no Jacobian.  ``h0`` is the first
continuous object every downstream computation sees, and everything the model knows about a node
flows through it.

Three measurements:

``dose``       the carriage range as the donor payload is interpolated from the clean value
               (alpha -> 0, the Jacobian limit) to a full real-donor replacement (alpha = 1).
               A FLAT curve falsifies the transfer claim: it would mean the model is effectively
               linear over the scale of realistic input changes and the tangent loses nothing.
``mechanism``  the per-node gap between the two ranges against local gate inactivity -- the share
               of ReLU pre-activations at or below zero, and attention effective support.  Tests
               the proposed mechanism, not merely the phenomenon.
``reference``  each measure against the task's OWN required range.  A node's label depends on the
               two marked endpoints, so ``(d(v,S) + d(v,T)) / 2`` is derivable from the task
               definition and is visible to neither measure.  Compared by rank across graphs,
               because the reference counts only two sources while the measures integrate over
               all of them (including the dominant ``d = 0`` self term).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import rangelib as R

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "experiments" / "synthetic" / "training"))

import marked_tree_path_graphgps as MTP  # noqa: E402


# --------------------------------------------------------------------------------------
# Model surgery: expose the map h0 -> node logits
# --------------------------------------------------------------------------------------


def load_model(checkpoint: Path) -> tuple[MTP.GraphGPSPathModel, argparse.Namespace]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = state["args"]
    if isinstance(args, dict):  # saved as a plain dict rather than a Namespace
        args = argparse.Namespace(**args)
    model = MTP.GraphGPSPathModel(
        depth=int(state["depth"]),
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        pe_dim=MTP.pe_dim_for_channel(args.structural_channel, args.rwse_steps),
        dropout=0.0,
        attn_dropout=0.0,
        use_spd_bias=MTP.uses_spd_bias(args.structural_channel),
        spd_cap=int(args.spd_cap),
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, args


def encode(model: MTP.GraphGPSPathModel, batch: MTP.Batch) -> torch.Tensor:
    """The clean ``h0``, exactly as ``GraphGPSPathModel.forward`` builds it."""

    h = model.token_emb(batch.x)
    if model.pe_proj is not None:
        h = h + model.pe_proj(batch.pe)
    return h * batch.mask.unsqueeze(-1).to(h.dtype)


def forward_from_h0(model: MTP.GraphGPSPathModel, h0: torch.Tensor, batch: MTP.Batch) -> torch.Tensor:
    """Resume ``GraphGPSPathModel.forward`` from a supplied ``h0``."""

    h = h0 * batch.mask.unsqueeze(-1).to(h0.dtype)
    for layer in model.layers:
        h, _ = layer(h, batch)
    logits = model.head(h).squeeze(-1)
    return logits.masked_fill(~batch.mask, 0.0)


def repeat_batch(batch: MTP.Batch, count: int) -> MTP.Batch:
    return MTP.Batch(
        x=batch.x.repeat(count, 1),
        y=batch.y.repeat(count, 1),
        mask=batch.mask.repeat(count, 1),
        adj_norm=batch.adj_norm.repeat(count, 1, 1),
        pe=batch.pe.repeat(count, 1, 1),
        spd=batch.spd.repeat(count, 1, 1),
        path_len=batch.path_len.repeat(count),
    )


# --------------------------------------------------------------------------------------
# Gate statistics: where is the tangent structurally blind?
# --------------------------------------------------------------------------------------


def gate_statistics(model: MTP.GraphGPSPathModel, h0: torch.Tensor, batch: MTP.Batch) -> dict:
    """Per-node share of dead ReLU units, and attention effective support.

    The message-passing gate ``relu(h + edge_emb)`` in ``DenseGINEBranch`` is the site that most
    directly controls what a node forwards to its neighbours, so it is measured separately from
    the pointwise FFN and head gates.
    """

    dead_message, dead_pointwise, support = [], [], []

    def message_hook(module, inputs):
        h = inputs[0]
        dead_message.append((h + module.edge_emb <= 0).to(torch.float64).mean(dim=-1))

    def relu_hook(module, inputs, output):
        dead_pointwise.append((inputs[0] <= 0).to(torch.float64).mean(dim=-1))

    def attention_hook(module, inputs, output):
        # output = (out, metric_logits); recompute the realised attention from the logits.
        mask = inputs[1]
        logits = output[1]
        attn = torch.softmax(logits.masked_fill(~mask[:, None, None, :], -1.0e9), dim=-1)
        attn = attn.masked_fill(~(mask[:, None, :, None] & mask[:, None, None, :]), 0.0)
        # Effective number of sources each (head, destination) actually reads.
        support.append(1.0 / attn.square().sum(dim=-1).clamp_min(1e-30))

    handles = [layer.local.register_forward_pre_hook(message_hook) for layer in model.layers]
    handles += [
        module.register_forward_hook(relu_hook)
        for module in model.modules()
        if isinstance(module, torch.nn.ReLU)
    ]
    handles += [layer.attn.register_forward_hook(attention_hook) for layer in model.layers]
    try:
        with torch.no_grad():
            forward_from_h0(model, h0, batch)
    finally:
        for handle in handles:
            handle.remove()

    valid = batch.mask[0].numpy()
    return {
        "dead_message": torch.stack(dead_message).mean(dim=0)[0].numpy()[valid],
        "dead_pointwise": torch.stack(dead_pointwise).mean(dim=0)[0].numpy()[valid],
        # Mean over heads and layers, per destination node.
        "attention_support": torch.stack(support).mean(dim=(0, 2))[0].numpy()[valid],
    }


# --------------------------------------------------------------------------------------
# The two measures, both based at h0
# --------------------------------------------------------------------------------------


def jacobian_range(model, h0, batch, distances) -> np.ndarray:
    """Bamberger normalised range: ``I_u(v) = sum_b |d logit_u / d h0_{v,b}|``."""

    def fn(state: torch.Tensor) -> torch.Tensor:
        return forward_from_h0(model, state, batch)

    jac = torch.autograd.functional.jacobian(fn, h0, vectorize=True)  # [1,N,1,N,H]
    influence = jac.abs().sum(dim=-1)[0, :, 0, :].detach().numpy()
    valid = batch.mask[0].numpy()
    influence = influence[np.ix_(valid, valid)]
    return R.normalised_range_from_influence(influence, distances), influence


def carriage_field(
    model,
    h0,
    batch,
    pool: R.SemanticDonorPool,
    *,
    graph_id: int,
    donors: int,
    rng: np.random.Generator,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Event magnitudes ``[source, donor, carrier]`` and doses, on the real model."""

    valid = batch.mask[0].numpy()
    nodes = int(valid.sum())
    repeated = repeat_batch(batch, donors)
    with torch.no_grad():
        clean = forward_from_h0(model, h0, batch)[0][valid]

    payload = h0[0].detach().numpy()
    degrees = (batch.adj_norm[0] > 0).sum(dim=1).numpy()
    magnitudes, doses = [], []
    for source in range(nodes):
        drawn = pool.draw(payload[source], int(degrees[source]), donors, rng, base_graph_id=graph_id)
        rows = np.asarray([node.payload for node in drawn], dtype=np.float64)
        delta = rows - payload[source][None, :]
        variants = h0.repeat(donors, 1, 1).clone()
        variants[:, source, :] = h0[0, source] + float(alpha) * torch.tensor(
            delta, dtype=h0.dtype
        )
        with torch.no_grad():
            moved = forward_from_h0(model, variants, repeated)[:, valid]
        # Identity readout on node logits: ||q_i||_2 = |clean_i - event_i|.
        magnitudes.append((clean[None, :] - moved).abs().numpy())
        doses.append(float(alpha) * np.linalg.norm(delta, axis=1))
    return np.stack(magnitudes), np.stack(doses)


# --------------------------------------------------------------------------------------


def build_graphs(args, seed: int, count: int) -> list:
    return MTP.generate_examples(
        count=count,
        min_n=args.min_n,
        max_n=args.max_n,
        seed=seed,
        structural_channel=args.structural_channel,
        rwse_steps=args.rwse_steps,
        min_path_frac=args.min_path_frac,
        endpoint_candidates=args.endpoint_candidates,
    )


def edge_index_from_adjacency(adj: np.ndarray) -> torch.Tensor:
    src, dst = np.nonzero(adj)
    return torch.tensor(np.stack([src, dst]), dtype=torch.long)


def prepare(model, example, spd_cap: int):
    batch = MTP.collate_examples([example], spd_cap=spd_cap)
    with torch.no_grad():
        h0 = encode(model, batch)
    distances = R.shortest_path_distances(edge_index_from_adjacency(example.adj), example.n)
    return batch, h0, distances


def task_reference(example, distances: np.ndarray) -> float:
    """Mean over nodes of ``(d(v,S) + d(v,T)) / 2`` -- derivable from the task definition."""

    source = int(np.flatnonzero(example.x == 1)[0])
    target = int(np.flatnonzero(example.x == 2)[0])
    return float(0.5 * (distances[:, source] + distances[:, target]).mean())


def run(args) -> dict:
    model, train_args = load_model(Path(args.checkpoint))
    for name in ("structural_channel", "rwse_steps", "min_path_frac", "endpoint_candidates", "spd_cap"):
        setattr(args, name, getattr(train_args, name))
    print(f"[setup] depth={len(model.layers)} channel={args.structural_channel}", flush=True)

    base = build_graphs(args, args.data_seed, args.graphs)
    donor_examples = build_graphs(args, args.data_seed + 5_000, args.donor_graphs)

    donor_entries = []
    for index, example in enumerate(donor_examples):
        batch, h0, _ = prepare(model, example, args.spd_cap)
        donor_entries.append(
            (10_000 + index, R.TinyData(h0[0].detach().double(), edge_index_from_adjacency(example.adj)))
        )
    pool = R.SemanticDonorPool(donor_entries)

    prepared = [prepare(model, example, args.spd_cap) for example in base]

    # --- the two ranges, plus the task's own reference, per graph -----------------------
    jac_per_graph, reference, influences = [], [], []
    for example, (batch, h0, distances) in zip(base, prepared):
        rho, influence = jacobian_range(model, h0, batch, distances)
        jac_per_graph.append(R.safe_nanmean(rho))
        influences.append((rho, influence))
        reference.append(task_reference(example, distances))
    jac = R.bootstrap_graph_values(jac_per_graph)
    print(f"[jacobian] range = {jac.estimate:.4f}  [{jac.low:.4f}, {jac.high:.4f}]", flush=True)

    # --- dose ladder --------------------------------------------------------------------
    ladder, carriage_by_alpha = [], {}
    for alpha in args.alphas:
        started = time.time()
        per_graph = []
        for index, (example, (batch, h0, distances)) in enumerate(zip(base, prepared)):
            magnitude, dose = carriage_field(
                model,
                h0,
                batch,
                pool,
                graph_id=index,
                donors=args.donors,
                rng=np.random.default_rng(args.event_seed + 977 * index),
                alpha=alpha,
            )
            per_graph.append(
                {
                    "magnitude": magnitude,
                    "dose": dose,
                    "distances": distances,
                    "sources": np.arange(magnitude.shape[0]),
                }
            )
        interval = R.bootstrap_carriage_range(per_graph, normalise=True)
        raw = R.bootstrap_carriage_range(per_graph, normalise=False)
        carriage_by_alpha[alpha] = per_graph
        ladder.append({"alpha": alpha, "carriage": interval.__dict__, "carriage_raw": raw.__dict__})
        print(
            f"[dose] alpha={alpha:<6} carriage={interval.estimate:.4f} "
            f"[{interval.low:.4f}, {interval.high:.4f}]  ({time.time() - started:.0f}s)",
            flush=True,
        )

    # --- why the direction? attenuation of the influence profile by distance -------------
    # If a full-dose swap attenuates FAR responses more than near ones, the measured range
    # shortens relative to the tangent -- which is the over-report direction, and the opposite
    # of the gate mechanism.  A flat ratio would mean the shortening comes from somewhere else.
    def distance_profile(entries):
        totals, counts = {}, {}
        for entry in entries:
            field = (entry["magnitude"] / entry["dose"][:, :, None]).mean(axis=1).T
            d = entry["distances"][:, entry["sources"]]
            for step in range(int(np.nanmax(d[np.isfinite(d)])) + 1):
                cells = d == step
                if cells.any():
                    totals[step] = totals.get(step, 0.0) + float(field[cells].sum())
                    counts[step] = counts.get(step, 0) + int(cells.sum())
        return {k: totals[k] / counts[k] for k in sorted(totals)}

    tangent_profile = distance_profile(carriage_by_alpha[min(args.alphas)])
    finite_profile = distance_profile(carriage_by_alpha[max(args.alphas)])
    steps = sorted(set(tangent_profile) & set(finite_profile))
    attenuation = {
        int(step): float(finite_profile[step] / tangent_profile[step])
        for step in steps
        if tangent_profile[step] > 0
    }
    print("[attenuation] finite/tangent influence by hop distance:")
    print("             " + "  ".join(f"d={k}:{v:.3f}" for k, v in list(attenuation.items())[:9]))

    # --- mechanism: per-node gap against local gate inactivity ---------------------------
    rows = []
    full = carriage_by_alpha[max(args.alphas)]
    for index, ((rho_jac, _), entry) in enumerate(zip(influences, full)):
        batch, h0, distances = prepared[index]
        field = (entry["magnitude"] / entry["dose"][:, :, None]).mean(axis=1).T
        rho_car = R.carriage_range_per_carrier(field, distances, entry["sources"])
        gates = gate_statistics(model, h0, batch)
        for node in range(len(rho_jac)):
            if np.isfinite(rho_jac[node]) and np.isfinite(rho_car[node]):
                rows.append(
                    {
                        "graph": index,
                        "node": node,
                        "jacobian": float(rho_jac[node]),
                        "carriage": float(rho_car[node]),
                        "gap": float(rho_car[node] - rho_jac[node]),
                        "dead_message": float(gates["dead_message"][node]),
                        "dead_pointwise": float(gates["dead_pointwise"][node]),
                        "attention_support": float(gates["attention_support"][node]),
                    }
                )

    def spearman(a, b) -> float:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
            return float("nan")
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        return float(np.corrcoef(ra, rb)[0, 1])

    gap = [r["gap"] for r in rows]
    mechanism = {
        "n_nodes": len(rows),
        "mean_gap": float(np.mean(gap)),
        "spearman_gap_dead_message": spearman(gap, [r["dead_message"] for r in rows]),
        "spearman_gap_dead_pointwise": spearman(gap, [r["dead_pointwise"] for r in rows]),
        "spearman_gap_attention_support": spearman(gap, [r["attention_support"] for r in rows]),
    }
    print(f"[mechanism] mean per-node gap = {mechanism['mean_gap']:+.4f} over {len(rows)} nodes")
    for key in ("dead_message", "dead_pointwise", "attention_support"):
        print(f"            spearman(gap, {key}) = {mechanism['spearman_gap_' + key]:+.3f}")

    # --- reference: which measure tracks the task's required range across graphs ---------
    car_per_graph = [
        R.safe_nanmean(
            R.carriage_range_per_carrier(
                (e["magnitude"] / e["dose"][:, :, None]).mean(axis=1).T, e["distances"], e["sources"]
            )
        )
        for e in full
    ]
    agreement = {
        "task_reference_mean": float(np.mean(reference)),
        "spearman_jacobian_vs_reference": spearman(jac_per_graph, reference),
        "spearman_carriage_vs_reference": spearman(car_per_graph, reference),
    }
    print(
        f"[reference] spearman vs task-required range: "
        f"jacobian {agreement['spearman_jacobian_vs_reference']:+.3f}  "
        f"carriage {agreement['spearman_carriage_vs_reference']:+.3f}"
    )

    return {
        "config": {k: str(v) for k, v in vars(args).items()},
        "fit": {k: v for k, v in torch.load(
            Path(args.checkpoint), map_location="cpu", weights_only=False
        )["best_stats"]["id_test"].items() if k in ("node_acc", "graph_exact", "f1", "avg_path_len")},
        "jacobian": jac.__dict__,
        "jacobian_per_graph": jac_per_graph,
        "carriage_per_graph": car_per_graph,
        "task_reference_per_graph": reference,
        "dose_ladder": ladder,
        "attenuation_by_distance": attenuation,
        "tangent_profile": {int(k): v for k, v in tangent_profile.items()},
        "finite_profile": {int(k): v for k, v in finite_profile.items()},
        "mechanism": mechanism,
        "mechanism_rows": rows,
        "agreement": agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="models/range_followup/depth_3/best.pt"
    )
    parser.add_argument("--graphs", type=int, default=16)
    parser.add_argument("--donor-graphs", type=int, default=8)
    parser.add_argument("--donors", type=int, default=8)
    parser.add_argument("--min-n", type=int, default=16)
    parser.add_argument("--max-n", type=int, default=28)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.001, 0.01, 0.05, 0.2, 0.5, 1.0]
    )
    parser.add_argument("--data-seed", type=int, default=20260728)
    parser.add_argument("--event-seed", type=int, default=17_071)
    parser.add_argument("--out", default="results/realmodel.json")
    args = parser.parse_args()
    args.checkpoint = str(Path(__file__).resolve().parent / args.checkpoint)

    payload = run(args)
    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
