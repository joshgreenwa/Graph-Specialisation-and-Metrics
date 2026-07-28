"""Experiment 4: a graph-level task, where their measure needs the Hessian and ours does not.

For a pooled scalar output the Jacobian is a vector and carries no pairwise structure, so the paper
switches to the Hessian: ``eta_hat_u = sum_v |d^2 y / dx_u dx_v| d(u,v) / sum_v |...|``.

A donor swap already supplies one index (the source ``s``) as a finite intervention and the other
(the carrier ``i``) as a pre-pooling node state, so Functional carriage reads a pairwise field from
**first-order information alone** -- forward evaluations plus one readout gradient, no second
derivative anywhere.  The claim is about derivative order, not wall clock: at this size autograd's
batched Hessian is the faster of the two, and the measured seconds are recorded per run so that
stays visible.

The task is the paper's own pairwise family (Section 5.1, Example 2):

    y(X) = sum_u h_u(X),   h_u(X) = (1/|N_<=k(u)|) sum_{v in N_<=k(u)} (x_u - x_v)^2

whose Hessian is analytic: ``d^2 y / dx_u dx_v = -2(1/m_u + 1/m_v)`` for ``u != v`` in the ball.
The two measures are *not* the same object -- theirs pairs two inputs, ours pairs an input with a
carrier state -- so this is a tracking comparison, not an identity.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import rangelib as R


def ball_mask(distances: np.ndarray, k: int) -> torch.Tensor:
    mask = (distances <= k).astype(np.float64)
    counts = mask.sum(axis=1, keepdims=True)
    return torch.tensor(mask / np.maximum(counts, 1.0), dtype=R.DTYPE)


def node_summands(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``h_u = sum_v w_uv (x_u - x_v)^2`` summed over channels, shape ``[..., n, 1]``."""

    squared = (x * x).sum(dim=-1, keepdim=True)
    cross = torch.matmul(weights, x)
    mean_squared = torch.matmul(weights, squared)
    return squared - 2.0 * (x * cross).sum(dim=-1, keepdim=True) + mean_squared


def hessian_influence(weights: torch.Tensor, x: torch.Tensor) -> np.ndarray:
    """``|d^2 y / dx_u dx_v|`` by autograd, for the single-channel pooled scalar."""

    flat = x.detach().clone().reshape(-1).requires_grad_(True)
    n = x.shape[0]

    def scalar(vector: torch.Tensor) -> torch.Tensor:
        return node_summands(vector.reshape(n, 1), weights).sum()

    hessian = torch.autograd.functional.hessian(scalar, flat, vectorize=True)
    return hessian.abs().detach().numpy()


def run(args) -> dict:
    rng = np.random.default_rng(args.data_seed)
    base = R.make_graphs(args.graphs, nodes=args.nodes, channels=1, rng=rng)
    donor_graphs = R.make_graphs(args.donor_graphs, nodes=args.nodes, channels=1, rng=rng)
    pool = R.SemanticDonorPool([(10_000 + i, g) for i, g in enumerate(donor_graphs)])
    distances = R.shortest_path_distances(base[0].edge_index, base[0].num_nodes)

    results = []
    profiles: dict[str, list] = {}
    for k in range(1, args.max_k + 1):
        weights = ball_mask(distances, k)
        # Sum pooling: z = sum_i h_i, so g_out[t, i, m] = 1 for the single output.
        readout = torch.ones(1, args.nodes, 1, dtype=R.DTYPE)

        started = time.time()
        hess_per_graph = []
        for graph in base:
            influence = hessian_influence(weights, graph.x)
            hess_per_graph.append(
                R.safe_nanmean(R.normalised_range_from_influence(influence, distances))
            )
        hessian_seconds = time.time() - started

        started = time.time()
        per_graph = []
        for index, graph in enumerate(base):
            events = R.carriage_events(
                graph,
                lambda x, w=weights: node_summands(x, w),
                pool,
                graph_id=index,
                donors=args.donors,
                rng=np.random.default_rng(args.event_seed + 977 * index),
                readout_gradient=readout,
            )
            per_graph.append(
                {
                    "magnitude": events.magnitude,
                    "dose": events.dose,
                    "distances": distances,
                    "sources": events.sources,
                }
            )
        carriage_seconds = time.time() - started

        hess = R.bootstrap_graph_values(hess_per_graph)
        carriage = R.bootstrap_carriage_range(per_graph, normalise=True)
        carriage_raw = R.bootstrap_carriage_range(per_graph, normalise=False)

        if k == args.profile_k:
            influence = hessian_influence(weights, base[0].x)
            field = (
                per_graph[0]["magnitude"] / per_graph[0]["dose"][:, :, None]
            ).mean(axis=1).T
            steps = np.arange(0, int(np.nanmax(distances[np.isfinite(distances)])) + 1)
            hess_profile, car_profile = [], []
            for d in steps:
                cells = distances == d
                hess_profile.append(float(influence[cells].mean()) if cells.any() else 0.0)
                car_profile.append(float(field[cells].mean()) if cells.any() else 0.0)
            hess_profile = np.asarray(hess_profile)
            car_profile = np.asarray(car_profile)
            profiles = {
                "k": int(k),
                "distance": steps.tolist(),
                "hessian": (hess_profile / max(hess_profile.sum(), 1e-300)).tolist(),
                "carriage": (car_profile / max(car_profile.sum(), 1e-300)).tolist(),
            }
        results.append(
            {
                "k": k,
                "hessian": hess.__dict__,
                "carriage": carriage.__dict__,
                    "carriage_raw": carriage_raw.__dict__,
                "hessian_seconds": round(hessian_seconds, 3),
                "carriage_seconds": round(carriage_seconds, 3),
                "backward_passes": args.graphs * args.nodes,
                "forward_passes": args.graphs * args.nodes * args.donors,
            }
        )
        print(
            f"k={k}  hessian={hess.estimate:6.3f} ({hessian_seconds:5.2f}s)  "
            f"carriage={carriage.estimate:6.3f} ({carriage_seconds:5.2f}s)",
            flush=True,
        )
    return {"config": vars(args), "results": results, "profiles": profiles}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=12)
    parser.add_argument("--donor-graphs", type=int, default=6)
    parser.add_argument("--nodes", type=int, default=41)
    parser.add_argument("--donors", type=int, default=16)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--profile-k", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=20260728)
    parser.add_argument("--event-seed", type=int, default=17_071)
    parser.add_argument("--out", default="results/graphlevel.json")
    args = parser.parse_args()

    payload = run(args)
    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
