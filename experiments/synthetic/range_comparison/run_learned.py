"""Experiment 2: the same comparison on *learned* approximators of the three tasks.

A small message-passing network is trained to regress each ground-truth operator, then both range
measures are applied to the trained map.  The operator's own range is the reference: a model that
has learned the task should report the task's range under either measure.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import rangelib as R

TASKS = ("dirac", "rectangle", "power_loops")


class SimpleMPNN(nn.Module):
    """``h <- relu(W_self h + W_neighbour A_hat h)`` -- one hop of receptive field per layer."""

    def __init__(self, channels: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.encode = nn.Linear(channels, hidden)
        self.self_weight = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.neighbour_weight = nn.ModuleList(
            nn.Linear(hidden, hidden, bias=False) for _ in range(layers)
        )
        self.decode = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        for self_weight, neighbour_weight in zip(self.self_weight, self.neighbour_weight):
            h = torch.relu(self_weight(h) + neighbour_weight(adjacency @ h))
        return self.decode(h)


def train_model(matrix: np.ndarray, adjacency: np.ndarray, args, *, seed: int) -> tuple[SimpleMPNN, dict]:
    torch.manual_seed(seed)
    nodes = matrix.shape[0]
    model = SimpleMPNN(args.channels, args.hidden, args.layers).to(R.DTYPE)
    target = torch.tensor(matrix, dtype=R.DTYPE)
    adj = torch.tensor(adjacency, dtype=R.DTYPE)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(seed + 1)
    history: list[float] = []
    for step in range(args.steps):
        x = torch.randn(args.batch, nodes, args.channels, dtype=R.DTYPE, generator=generator)
        loss = torch.nn.functional.mse_loss(model(x, adj), target @ x)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % max(1, args.steps // 20) == 0:
            history.append(float(loss.detach()))
    with torch.no_grad():
        x = torch.randn(64, nodes, args.channels, dtype=R.DTYPE, generator=generator)
        truth = target @ x
        prediction = model(x, adj)
        mse = float(torch.nn.functional.mse_loss(prediction, truth))
        variance = float(truth.var())
    return model, {"train_history": history, "test_mse": mse, "test_r2": 1.0 - mse / variance}


def train_best(matrix, adjacency, args, *, seed: int) -> tuple[SimpleMPNN, dict]:
    """Take the best of a few restarts.

    A deep ReLU stack on a pure ``k``-hop shell target collapses to a constant for some seeds; the
    restart keeps a non-degenerate map so the range stays estimable.  The achieved ``R^2`` is
    reported either way -- a poorly fit model genuinely has a shorter range than its target, and
    that is the point of the panel rather than something to hide.
    """

    best, best_fit = None, None
    for restart in range(max(1, args.restarts)):
        model, fit = train_model(matrix, adjacency, args, seed=seed + 1_000 * restart)
        if best_fit is None or fit["test_r2"] > best_fit["test_r2"]:
            best, best_fit = model, fit
    best_fit = dict(best_fit, restarts=int(max(1, args.restarts)))
    return best, best_fit


def run(args) -> dict:
    rng = np.random.default_rng(args.data_seed)
    base = R.make_graphs(args.graphs, nodes=args.nodes, channels=args.channels, rng=rng)
    donor_graphs = R.make_graphs(args.donor_graphs, nodes=args.nodes, channels=args.channels, rng=rng)
    pool = R.SemanticDonorPool([(10_000 + i, g) for i, g in enumerate(donor_graphs)])
    distances = R.shortest_path_distances(base[0].edge_index, base[0].num_nodes)
    adjacency = R.normalised_adjacency(base[0].edge_index, base[0].num_nodes)

    results: list[dict] = []
    for task in TASKS:
        for k in args.k_values:
            started = time.time()
            matrix = R.operator_matrix(task, k, distances, base[0].edge_index)
            args.layers = k + args.extra_layers
            model, fit = train_best(matrix, adjacency, args, seed=args.model_seed + 31 * k)
            adj = torch.tensor(adjacency, dtype=R.DTYPE)

            def fn(x: torch.Tensor, model=model, adj=adj) -> torch.Tensor:
                return model(x, adj)

            operator_range = R.safe_nanmean(
                R.normalised_range_from_influence(
                    R.influence_matrix_analytic(matrix, args.channels), distances
                )
            )
            jac_per_graph: list[float] = []
            per_graph_events: list[dict] = []
            for index, graph in enumerate(base):
                influence = R.influence_matrix_autograd(fn, graph.x)
                jac_per_graph.append(
                    R.safe_nanmean(R.normalised_range_from_influence(influence, distances))
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
                        "distances": distances,
                        "sources": events.sources,
                    }
                )
            jac = R.bootstrap_graph_values(jac_per_graph)
            carriage = R.bootstrap_carriage_range(per_graph_events, normalise=True)
            carriage_raw = R.bootstrap_carriage_range(per_graph_events, normalise=False)
            results.append(
                {
                    "task": task,
                    "k": k,
                    "layers": args.layers,
                    "operator_range": operator_range,
                    "fit": fit,
                    "jacobian": jac.__dict__,
                    "carriage_normalised": carriage.__dict__,
                    "carriage_raw": carriage_raw.__dict__,
                    "seconds": round(time.time() - started, 2),
                }
            )
            print(
                f"{task:>10} k={k} L={args.layers}  R2={fit['test_r2']:.4f}  "
                f"operator={operator_range:6.3f}  jacobian={jac.estimate:6.3f}  "
                f"carriage={carriage.estimate:6.3f}  ({results[-1]['seconds']}s)",
                flush=True,
            )
    return {"config": {k: v for k, v in vars(args).items()}, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=12)
    parser.add_argument("--donor-graphs", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=25)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--donors", type=int, default=8)
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--extra-layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--data-seed", type=int, default=20260728)
    parser.add_argument("--model-seed", type=int, default=99)
    parser.add_argument("--event-seed", type=int, default=17_071)
    parser.add_argument("--out", default="results/learned.json")
    args = parser.parse_args()
    args.layers = 0

    payload = run(args)
    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
