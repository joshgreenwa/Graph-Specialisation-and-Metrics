"""Beneficial carriage on MarkedTreePath, and a ground-truth faithfulness test.

Everything measured so far -- the Jacobian range, ``F_sens``, the shell probe -- is **label-free**.
All three answer "did the output move".  Beneficial carriage (methodology section 6) answers "did
the answer get worse": it integrates the signed task loss along a straight final-state path from
each intervened endpoint back to clean, so its carrier sum is exactly the donor event's loss
change.

``B`` is signed, so it cannot be put through the expected-distance formula -- the denominator
passes through zero.  Section 7 gives the right accumulations instead: ``S_B(b)``, signed loss mass
per distance bin, and ``B_far(r)``, mass carried beyond radius ``r``.

THE GROUND-TRUTH TEST, AND WHY ITS NAIVE FORM FAILS.  Labels are determined by the two marked
endpoints, and corrupting a mark costs far more than corrupting an ordinary node (+1.92 against
+0.14 in distribution).  That much is solidly established.  But ranking sources by any measure is
NOT a faithfulness test, because a mark's ``h0`` row sits 6x further from a typical donor row than
an ordinary node's: **donor dose alone scores AUROC 1.000**, above every measure it would be used
to score.  The dose null is therefore computed and reported alongside, and the ranking table must
be read as a dose-monotonicity ordering, not a faithfulness ordering.

Two mechanism notes, both measured rather than assumed: an unmarked donor's token embedding *is*
``e_0``, so a token-channel swap at an ordinary node is an exact no-op, and essentially all of an
ordinary node's damage comes through the RWSE channel -- which is load-bearing here, not
compensated (zeroing it drops ID F1 from 0.946 to 0.560).  This checkpoint uses
``structural_channel='rwse'``, so ``batch.spd`` is identically zero and RWSE is the only per-node
positional input.

Run on both splits: in-distribution, where the model is at 0.98 node accuracy and the loss sits
near its floor, and out-of-distribution, where it collapses and there is real loss headroom.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import rangelib as R
from run_realmodel import (
    build_graphs,
    edge_index_from_adjacency,
    encode,
    forward_from_h0,
    load_model,
    repeat_batch,
)

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "experiments" / "synthetic" / "training"))

import marked_tree_path_graphgps as MTP  # noqa: E402
from graph_specialisation_metrics.carriage.core import integrated_loss_carriage  # noqa: E402


def forward_to_states(model, h0: torch.Tensor, batch: MTP.Batch) -> torch.Tensor:
    """Pre-head node states ``h^L`` -- the carriers for Beneficial carriage."""

    h = h0 * batch.mask.unsqueeze(-1).to(h0.dtype)
    for layer in model.layers:
        h, _ = layer(h, batch)
    return h


def make_loss_from_states(model, y_valid: torch.Tensor, pos_weight: torch.Tensor):
    """``[Q, n_valid, width] -> [Q]``.  The head is pointwise over nodes, so slicing is safe."""

    def loss_from_states(states: torch.Tensor) -> torch.Tensor:
        logits = model.head(states).squeeze(-1)
        target = y_valid.unsqueeze(0).expand_as(logits)
        return F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pos_weight, reduction="none"
        ).mean(dim=-1)

    return loss_from_states


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Probability a random positive outranks a random negative; ties count a half."""

    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    pos, neg = scores[positive], scores[~positive]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    comparisons = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(comparisons / (pos.size * neg.size))


def measure_graph(model, example, args, pool, *, graph_id: int, rng) -> dict:
    batch = MTP.collate_examples([example], spd_cap=args.spd_cap)
    with torch.no_grad():
        h0 = encode(model, batch)
    valid = batch.mask[0].numpy()
    nodes = int(valid.sum())
    distances = R.shortest_path_distances(edge_index_from_adjacency(example.adj), example.n)

    marks = np.asarray(
        [int(np.flatnonzero(example.x == 1)[0]), int(np.flatnonzero(example.x == 2)[0])]
    )
    sources = np.arange(nodes)
    if args.source_cap and nodes > args.source_cap:
        # Stratified: both marks always in, the rest sampled uniformly.  AUROC compares marks
        # against ordinary nodes, so sampling only the negatives leaves it unbiased -- but the
        # source set is no longer the plain uniform cap, and that is a declared deviation.
        others = np.setdiff1d(np.arange(nodes), marks)
        keep = rng.choice(others, size=max(0, args.source_cap - marks.size), replace=False)
        sources = np.sort(np.concatenate([marks, keep]))

    y_valid = batch.y[0][valid]
    positives = y_valid.sum()
    pos_weight = torch.clamp(
        (y_valid.numel() - positives) / positives.clamp_min(1.0), max=args.max_pos_weight
    ).detach()
    loss_from_states = make_loss_from_states(model, y_valid, pos_weight)

    with torch.no_grad():
        clean_states = forward_to_states(model, h0, batch)[0][valid]
        clean_logits = forward_from_h0(model, h0, batch)[0][valid]

    repeated = repeat_batch(batch, args.donors)
    payload = h0[0].detach().numpy()
    degrees = (batch.adj_norm[0] > 0).sum(dim=1).numpy()

    event_states, magnitudes, doses = [], [], []
    for source in sources:
        drawn = pool.draw(
            payload[source], int(degrees[source]), args.donors, rng, base_graph_id=graph_id
        )
        rows = np.asarray([node.payload for node in drawn], dtype=np.float64)
        delta = rows - payload[source][None, :]
        variants = h0.repeat(args.donors, 1, 1).clone()
        variants[:, source, :] = h0[0, source] + torch.tensor(delta, dtype=h0.dtype)
        with torch.no_grad():
            event_states.append(forward_to_states(model, variants, repeated)[:, valid])
            moved = forward_from_h0(model, variants, repeated)[:, valid]
        magnitudes.append((clean_logits[None, :] - moved).abs().numpy())
        doses.append(np.linalg.norm(delta, axis=1))

    # --- Beneficial carriage: one integrated path per (source, donor) --------------------
    swap = torch.cat(event_states, dim=0).to(torch.float64)
    clean = clean_states.to(torch.float64).unsqueeze(0).expand_as(swap)
    path = integrated_loss_carriage(
        clean,
        swap,
        loss_from_states=make_loss_from_states(
            model.to(torch.float64), y_valid.to(torch.float64), pos_weight.to(torch.float64)
        ),
        atol=args.atol,
        rtol=args.rtol,
        max_intervals=args.max_intervals,
    )
    model.to(torch.float32)
    # Section 6: B is the NEGATIVE of the loop's allocation; positive means clean reduced loss.
    beneficial = -path["carriage"].reshape(len(sources), args.donors, nodes).mean(dim=1).T.numpy()
    residual = path["completeness_residual"].abs().max().item()
    converged = float(path["converged"].to(torch.float64).mean())
    event_loss_increase = (-path["loss_delta"]).reshape(len(sources), args.donors).mean(dim=1)

    magnitude = np.stack(magnitudes)
    dose = np.stack(doses)
    functional = (magnitude / dose[:, :, None]).mean(axis=1).T
    functional_raw = magnitude.mean(axis=1).T

    # --- Jacobian influence at the same base point --------------------------------------
    def fn(state: torch.Tensor) -> torch.Tensor:
        return forward_from_h0(model, state, batch)

    jac = torch.autograd.functional.jacobian(fn, h0, vectorize=True)
    influence = jac.abs().sum(dim=-1)[0, :, 0, :].detach().numpy()[np.ix_(valid, valid)]

    # --- ground truth: the two marks -----------------------------------------------------
    is_mark = np.zeros(nodes, dtype=bool)
    is_mark[marks] = True
    mark_in_sources = is_mark[sources]

    # Inverse-probability weights.  Marks are force-included and carry ~8x the per-source signed
    # mass of an ordinary node, so an unweighted sum over the stratified set would inflate every
    # section 7 accumulation -- and inflate it far more on the OOD split, where every graph is
    # capped, than on ID, where almost none are.  AUROC is unaffected (only negatives are
    # sampled); the SUMS are not, so they are weighted back to the full node set.
    weights = np.ones(len(sources), dtype=np.float64)
    if len(sources) < nodes:
        kept_others = int((~mark_in_sources).sum())
        if kept_others:
            weights[~mark_in_sources] = (nodes - marks.size) / kept_others

    return {
        "nodes": nodes,
        "sources": sources,
        "source_weights": weights,
        "capped": bool(len(sources) < nodes),
        "distances": distances,
        "beneficial": beneficial,
        "functional": functional,
        "functional_raw": functional_raw,
        "influence": influence,
        "is_mark": is_mark,
        "completeness_residual": residual,
        "converged_fraction": converged,
        "mean_event_loss_increase": float(event_loss_increase.mean()),
        "auroc": {
            # The model-free null this table has to beat, and does not.
            "dose_null": auroc(dose.mean(axis=1), mark_in_sources),
            "beneficial": auroc(beneficial.sum(axis=0), mark_in_sources),
            "functional": auroc(functional.sum(axis=0), mark_in_sources),
            "functional_raw": auroc(functional_raw.sum(axis=0), mark_in_sources),
            "jacobian": auroc(influence.sum(axis=0)[sources], mark_in_sources),
        },
    }


def accumulate_by_distance(records: list[dict], field: str, edges: list[int]) -> dict:
    """``S_B(b)``: additive signed mass per distance bin, graph-averaged."""

    per_graph = {edge: [] for edge in edges}
    for record in records:
        d = record["distances"][:, record["sources"]]
        values = record[field]
        for low, high in zip(edges, edges[1:] + [10**9]):
            cells = (d >= low) & (d < high) & np.isfinite(d)
            weighted = values * record["source_weights"][None, :]
            per_graph[low].append(float(weighted[cells].sum()) if cells.any() else 0.0)
    return {int(k): float(np.mean(v)) for k, v in per_graph.items()}


def far_mass(records: list[dict], field: str, radii: list[int]) -> dict:
    out = {}
    for radius in radii:
        values = []
        for record in records:
            d = record["distances"][:, record["sources"]]
            cells = (d > radius) & np.isfinite(d)
            weighted = record[field] * record["source_weights"][None, :]
            values.append(float(weighted[cells].sum()) if cells.any() else 0.0)
        out[int(radius)] = float(np.mean(values))
    return out


def run_split(model, args, split: str, min_n: int, max_n: int, pool) -> dict:
    graph_args = argparse.Namespace(
        min_n=min_n,
        max_n=max_n,
        structural_channel=args.structural_channel,
        rwse_steps=args.rwse_steps,
        min_path_frac=args.min_path_frac,
        endpoint_candidates=args.endpoint_candidates,
    )
    examples = build_graphs(graph_args, args.data_seed + (0 if split == "id" else 991), args.graphs)
    started = time.time()
    records = []
    for index, example in enumerate(examples):
        records.append(
            measure_graph(
                model,
                example,
                args,
                pool,
                graph_id=index,
                rng=np.random.default_rng(args.event_seed + 977 * index),
            )
        )
    edges = [0, 1, 2, 3, 4, 8]
    radii = [0, 1, 2, 3, 4]
    summary = {
        "split": split,
        "graphs": len(records),
        "mean_nodes": float(np.mean([r["nodes"] for r in records])),
        "seconds": round(time.time() - started, 1),
        "completeness_residual_max": float(max(r["completeness_residual"] for r in records)),
        "converged_fraction": float(np.mean([r["converged_fraction"] for r in records])),
        "mean_event_loss_increase": float(np.mean([r["mean_event_loss_increase"] for r in records])),
        "S_B": accumulate_by_distance(records, "beneficial", edges),
        "S_F": accumulate_by_distance(records, "functional", edges),
        "B_far": far_mass(records, "beneficial", radii),
        "F_far": far_mass(records, "functional", radii),
        "auroc": {
            key: float(np.mean([r["auroc"][key] for r in records]))
            for key in ("dose_null", "beneficial", "functional", "functional_raw", "jacobian")
        },
        "capped_graphs": int(sum(r["capped"] for r in records)),
        "auroc_per_graph": {
            key: [r["auroc"][key] for r in records]
            for key in ("dose_null", "beneficial", "functional", "functional_raw", "jacobian")
        },
    }
    print(
        f"[{split}] graphs={summary['graphs']} nodes~{summary['mean_nodes']:.0f} "
        f"({summary['seconds']}s)  completeness_resid_max={summary['completeness_residual_max']:.2e} "
        f"converged={summary['converged_fraction']:.3f}",
        flush=True,
    )
    print(f"      mean event loss increase = {summary['mean_event_loss_increase']:+.4f}")
    print(f"      capped graphs: {summary['capped_graphs']}/{summary['graphs']} "
          "(section 7 sums are inverse-probability weighted)")
    print("      AUROC for ranking the two marks as sources (dose_null is model-free):")
    for key, value in summary["auroc"].items():
        print(f"        {key:>15}: {value:.3f}")
    print("      S_B by distance bin:", {k: round(v, 4) for k, v in summary["S_B"].items()})
    print("      B_far(r):           ", {k: round(v, 4) for k, v in summary["B_far"].items()})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/range_followup/depth_3/best.pt")
    parser.add_argument("--graphs", type=int, default=12)
    parser.add_argument("--donor-graphs", type=int, default=8)
    parser.add_argument("--donors", type=int, default=6)
    parser.add_argument("--source-cap", type=int, default=24)
    parser.add_argument("--id-min-n", type=int, default=16)
    parser.add_argument("--id-max-n", type=int, default=28)
    parser.add_argument("--ood-min-n", type=int, default=64)
    parser.add_argument("--ood-max-n", type=int, default=96)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--max-intervals", type=int, default=512)
    parser.add_argument("--data-seed", type=int, default=20260728)
    parser.add_argument("--event-seed", type=int, default=17_071)
    parser.add_argument("--out", default="results/beneficial.json")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    args.checkpoint = str(here / args.checkpoint)

    model, train_args = load_model(Path(args.checkpoint))
    for name in (
        "structural_channel",
        "rwse_steps",
        "min_path_frac",
        "endpoint_candidates",
        "spd_cap",
    ):
        setattr(args, name, getattr(train_args, name))

    donor_args = argparse.Namespace(
        min_n=args.id_min_n,
        max_n=args.id_max_n,
        structural_channel=args.structural_channel,
        rwse_steps=args.rwse_steps,
        min_path_frac=args.min_path_frac,
        endpoint_candidates=args.endpoint_candidates,
    )
    donor_entries = []
    for index, example in enumerate(build_graphs(donor_args, args.data_seed + 5_000, args.donor_graphs)):
        batch = MTP.collate_examples([example], spd_cap=args.spd_cap)
        with torch.no_grad():
            h0 = encode(model, batch)
        donor_entries.append(
            (10_000 + index, R.TinyData(h0[0].detach().double(), edge_index_from_adjacency(example.adj)))
        )
    pool = R.SemanticDonorPool(donor_entries)

    payload = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "id": run_split(model, args, "id", args.id_min_n, args.id_max_n, pool),
        "ood": run_split(model, args, "ood", args.ood_min_n, args.ood_max_n, pool),
    }
    out = here / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
