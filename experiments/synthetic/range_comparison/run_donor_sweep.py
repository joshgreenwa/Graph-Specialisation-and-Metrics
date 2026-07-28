"""How many donor events does the finite measure need?

Event-normalised carriage is exact for a linear operator at every single event, so its estimate is
independent of ``K``.  The un-normalised production field ``F_sens`` carries the per-source donor
scale and therefore converges only as ``K`` grows.  This sweep separates the two.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import rangelib as R


def run(args, scale_profile=None) -> dict:
    rng = np.random.default_rng(args.data_seed)
    shape = tuple(args.grid_shape)
    base = R.make_graphs(
        args.graphs, nodes=args.nodes, channels=args.channels, rng=rng,
        topology=args.topology, grid_shape=shape, scale_profile=scale_profile,
    )
    donor_graphs = R.make_graphs(
        args.donor_graphs, nodes=args.nodes, channels=args.channels, rng=rng,
        topology=args.topology, grid_shape=shape, scale_profile=scale_profile,
    )
    pool = R.SemanticDonorPool([(10_000 + i, g) for i, g in enumerate(donor_graphs)])
    distances = R.shortest_path_distances(base[0].edge_index, base[0].num_nodes)
    operator = R.figure3_operator if args.convention == "figure3" else R.operator_matrix

    jac = []
    for graph in base:
        matrix = operator(args.task, args.k, distances, graph.edge_index)
        influence = R.influence_matrix_analytic(matrix, args.channels)
        jac.append(R.safe_nanmean(R.normalised_range_from_influence(influence, distances)))
    jacobian = float(R.trimmed_mean(np.asarray(jac)))

    results = []
    for donors in args.donor_counts:
        per_graph = []
        for index, graph in enumerate(base):
            matrix = operator(args.task, args.k, distances, graph.edge_index)
            events = R.carriage_events(
                graph,
                R.linear_operator_fn(matrix),
                pool,
                graph_id=index,
                donors=int(donors),
                rng=np.random.default_rng(args.event_seed + 977 * index),
            )
            per_graph.append(
                {
                    "magnitude": events.magnitude,
                    "dose": events.dose,
                    "distances": distances,
                    "sources": events.sources,
                }
            )
        normalised = R.bootstrap_carriage_range(per_graph, normalise=True)
        raw = R.bootstrap_carriage_range(per_graph, normalise=False)
        results.append(
            {
                "donors": int(donors),
                "carriage_normalised": normalised.__dict__,
                "carriage_raw": raw.__dict__,
            }
        )
        print(
            f"K={donors:<4} jacobian={jacobian:7.4f}  normalised={normalised.estimate:7.4f}  "
            f"raw={raw.estimate:7.4f}",
            flush=True,
        )
    return {"config": vars(args), "task": args.task, "k": args.k, "jacobian": jacobian, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=12)
    parser.add_argument("--donor-graphs", type=int, default=6)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--topology", default="grid", choices=("path", "grid"))
    parser.add_argument("--grid-shape", type=int, nargs=2, default=[16, 16])
    parser.add_argument("--convention", default="figure3", choices=("figure3", "plain"))
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--task", default="rectangle")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--donor-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--data-seed", type=int, default=20260728)
    parser.add_argument("--event-seed", type=int, default=17_071)
    parser.add_argument("--out", default="results/donor_sweep.json")
    args = parser.parse_args()

    payload = run(args)
    print("\n--- heteroscedastic payloads (node-dependent scale) ---")
    payload["heteroscedastic"] = run(args, scale_profile="linear")["results"]
    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
