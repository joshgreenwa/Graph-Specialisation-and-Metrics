"""Experiment 1: replicate Bamberger et al. Figure 3 with donor-swap Functional carriage.

For each of the three synthetic tasks (k-Dirac, k-Rectangle, k-Power) on a fixed graph
distribution, the dataset range is measured twice:

* ``jacobian``            -- the paper's normalised range from the exact autograd Jacobian;
* ``carriage_normalised`` -- the expected distance of event-normalised Functional carriage;
* ``carriage_raw``        -- the same reduction applied to the production ``F_sens`` field, which
                             still carries the per-source donor scale ``mean_k ||delta_{s,k}||_2``.

Both are reduced with the registered graph estimator (20% trimmed mean over graphs) and carry the
registered 2,000-replicate percentile bootstrap.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import rangelib as R

TASKS = ("dirac", "rectangle", "power_loops", "power")
TASK_LABELS = {
    "dirac": "k-Dirac",
    "rectangle": "k-Rectangle",
    "power_loops": "k-Power",
    "power": "k-Power (no self-loops)",
}
# The paper's own plotted values, shipped in its repository as
# data/plotting/grid_task_range_{dirac,rectangle,power}.csv, column Val/TaskRangeSPDNorm.
# k-Power there is the self-loop variant, contradicting the Section 5.2 task definition.
PUBLISHED_GRID = {
    "dirac": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    "rectangle": [
        0.785794, 1.499524, 2.167486, 2.810191, 3.434266, 4.041735, 4.632934, 5.211021
    ],
    "power_loops": [
        0.786299, 1.240359, 1.570750, 1.836234, 2.063603, 2.264430, 2.445888, 2.612149
    ],
}


def build_dataset(args) -> tuple[list[R.TinyData], R.SemanticDonorPool]:
    rng = np.random.default_rng(args.data_seed)
    base = R.make_graphs(
        args.graphs,
        nodes=args.nodes,
        channels=args.channels,
        rng=rng,
        topology=args.topology,
        grid_shape=tuple(args.grid_shape) if args.grid_shape else None,
    )
    donor_graphs = R.make_graphs(
        args.donor_graphs,
        nodes=args.nodes,
        channels=args.channels,
        rng=rng,
        topology=args.topology,
        grid_shape=tuple(args.grid_shape) if args.grid_shape else None,
    )
    # Donor graph IDs are disjoint from every base graph ID (methodology section 9.1).
    pool = R.SemanticDonorPool([(10_000 + i, g) for i, g in enumerate(donor_graphs)])
    return base, pool


def run(args) -> dict:
    base, pool = build_dataset(args)
    distances = [R.shortest_path_distances(g.edge_index, g.num_nodes) for g in base]
    operator = R.figure3_operator if args.convention == "figure3" else R.operator_matrix
    results: list[dict] = []
    influence_panels: dict[str, dict] = {}

    for task in TASKS:
        for k in range(1, args.max_k + 1):
            started = time.time()
            jac_per_graph: list[float] = []
            per_graph_events: list[dict] = []
            for index, (graph, dist) in enumerate(zip(base, distances)):
                matrix = operator(task, k, dist, graph.edge_index)
                fn = R.linear_operator_fn(matrix)
                influence = R.influence_matrix_autograd(fn, graph.x)
                jac_per_graph.append(
                    R.safe_nanmean(R.normalised_range_from_influence(influence, dist))
                )
                events = R.carriage_events(
                    graph,
                    fn,
                    pool,
                    graph_id=index,
                    donors=args.donors,
                    rng=np.random.default_rng(args.event_seed + 977 * index),
                )
                per_graph_events.append(
                    {
                        "magnitude": events.magnitude,
                        "dose": events.dose,
                        "distances": dist,
                        "sources": events.sources,
                    }
                )
                if k == args.influence_k and index == 0:
                    if args.topology == "grid":
                        rows_, cols_ = tuple(args.grid_shape)
                        centre = (rows_ // 2) * cols_ + cols_ // 2
                    else:
                        centre = graph.num_nodes // 2
                    row_influence = influence[centre]
                    row_carriage = events.normalised_field[centre]
                    influence_panels[task] = {
                        "centre": int(centre),
                        "nodes": int(graph.num_nodes),
                        "distance": dist[centre].tolist(),
                        "jacobian": (row_influence / max(row_influence.sum(), 1e-300)).tolist(),
                        "carriage": (row_carriage / max(row_carriage.sum(), 1e-300)).tolist(),
                    }

            jac = R.bootstrap_graph_values(jac_per_graph)
            carriage = R.bootstrap_carriage_range(per_graph_events, normalise=True)
            raw = R.bootstrap_carriage_range(per_graph_events, normalise=False)
            # Exactness of the recovered influence profile, worst case over graphs.
            worst = 0.0
            for index, (graph, dist) in enumerate(zip(base, distances)):
                matrix = operator(task, k, dist, graph.edge_index)
                field = (
                    per_graph_events[index]["magnitude"]
                    / per_graph_events[index]["dose"][:, :, None]
                ).mean(axis=1).T
                worst = max(worst, float(np.abs(field - np.abs(matrix)).max()))
            published = (
                PUBLISHED_GRID[task][k - 1]
                if args.topology == "grid" and args.convention == "figure3" and task in PUBLISHED_GRID
                else None
            )
            results.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    # k-Power exists in two variants; only the self-loop one is what the paper
                    # plots, so name it in the record rather than leaving the bare key ambiguous.
                    "self_loops": task == "power_loops",
                    "is_paper_figure3_variant": task in PUBLISHED_GRID,
                    "k": k,
                    "jacobian": jac.__dict__,
                    "carriage_normalised": carriage.__dict__,
                    "carriage_raw": raw.__dict__,
                    "jacobian_per_graph": jac_per_graph,
                    "influence_max_abs_error": worst,
                    "published": published,
                    "seconds": round(time.time() - started, 2),
                }
            )
            against = (
                f"  published={published:7.4f} (d={abs(published - carriage.estimate):.1e})"
                if published is not None
                else ""
            )
            print(
                f"{TASK_LABELS[task]:>22}  k={k}  "
                f"jacobian={jac.estimate:7.4f}  carriage={carriage.estimate:7.4f}  "
                f"raw={raw.estimate:7.4f}  |F-|L||max={worst:.2e}{against}"
                f"  ({results[-1]['seconds']}s)",
                flush=True,
            )

    return {
        "config": vars(args),
        "protocol": "donor-swap-specialisation-carriage-v4 / semantic channel / identity readout",
        "results": results,
        "influence_panels": influence_panels,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=12)
    parser.add_argument("--donor-graphs", type=int, default=6)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--donors", type=int, default=8)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--influence-k", type=int, default=2)
    parser.add_argument("--topology", default="grid", choices=("path", "grid"))
    parser.add_argument("--grid-shape", type=int, nargs=2, default=[16, 16])
    parser.add_argument(
        "--convention",
        default="figure3",
        choices=("figure3", "plain"),
        help="figure3 reproduces the paper's implemented map F(X) = M^T X",
    )
    parser.add_argument("--data-seed", type=int, default=20260728)
    parser.add_argument("--event-seed", type=int, default=17_071)
    parser.add_argument("--out", default="results/replication.json")
    args = parser.parse_args()

    payload = run(args)
    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
