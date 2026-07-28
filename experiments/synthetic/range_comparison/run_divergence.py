"""Experiment 3: where the tangent measure and the finite donor-swap measure must disagree.

Three sweeps on a path graph, all with a single feature channel so that the paper's L1
channel aggregation and our L2-over-outputs aggregation coincide exactly:

``dose``      interpolate the donor payload from the clean value (alpha -> 0) to the registered
              full donor swap (alpha = 1).  Carriage range converges to the Jacobian range as
              alpha -> 0, so the paper's measure is the infinitesimal limit of ours.
``step``      ``F(X)_u = a x_u + b mean_{v in N_k(u)} sigmoid(x_v/tau)``.  As ``tau -> 0``
              the long-range term is flat at almost every clean input, so its tangent vanishes,
              while a realistic node replacement still flips it: the Jacobian range collapses to
              zero and the finite range does not.  Jacobian *under*-reports.
``oscillate`` ``F(X)_u = a x_u + b mean_{v in N_k(u)} sin(w x_v)/w``.  The derivative stays O(1)
              for every ``w`` while the finite response decays like ``1/w``: the Jacobian range
              stays long while the realised dependence vanishes.  Jacobian *over*-reports.

Each sweep also reports a gradient-free probe: the expected distance of the response to
resampling one node of a distance shell around the read node.  It is a robustness check on the dose
normalisation and the per-source aggregation, not an independent arbiter -- see the docstring of
``shell_resample_range``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import rangelib as R
from graph_specialisation_metrics.methodology.carriage import functional_carriage_events
from graph_specialisation_metrics.methodology.events import build_channel_events


# --------------------------------------------------------------------------------------
# Task family
# --------------------------------------------------------------------------------------


def far_mask(distances: np.ndarray, k: int) -> torch.Tensor:
    mask = (distances == k).astype(np.float64)
    counts = mask.sum(axis=1, keepdims=True)
    return torch.tensor(np.where(counts > 0, mask / np.maximum(counts, 1.0), 0.0), dtype=R.DTYPE)


def make_task(kind: str, parameter: float, mask: torch.Tensor, *, local: float, far: float):
    if kind == "step":
        nonlinearity = lambda value: torch.sigmoid(value / parameter)  # noqa: E731
    elif kind == "gate":
        nonlinearity = lambda value: torch.relu(value - parameter)  # noqa: E731
    elif kind == "oscillate":
        nonlinearity = lambda value: torch.sin(parameter * value) / parameter  # noqa: E731
    elif kind == "linear":
        nonlinearity = lambda value: value  # noqa: E731
    else:
        raise ValueError(f"unknown nonlinearity {kind!r}")

    def fn(x: torch.Tensor) -> torch.Tensor:
        return local * x + far * (mask @ nonlinearity(x))

    return fn


# --------------------------------------------------------------------------------------
# Carriage with a dose-scaled donor payload (alpha = 1 is the registered protocol)
# --------------------------------------------------------------------------------------


def dose_scaled_carriage(
    graph: R.TinyData,
    fn,
    pool: R.SemanticDonorPool,
    *,
    graph_id: int,
    donors: int,
    rng: np.random.Generator,
    alpha: float,
) -> R.CarriageEvents:
    n = graph.num_nodes
    with torch.no_grad():
        clean = fn(graph.x)
    gradient = R.identity_readout_gradient(int(clean.shape[-1]), n)
    magnitudes, doses, gaps = [], [], []
    for source in range(n):
        variants, records = build_channel_events(
            graph,
            graph_id=int(graph_id),
            source=int(source),
            channel="semantic",
            stage="dose-diagnostic",
            donors=int(donors),
            rng=rng,
            task=R.TASK,
            semantic_pool=pool,
            duplicate_tolerance=0.0,
        )
        with torch.no_grad():
            stacked = torch.stack([variant.x for variant in variants])
            # Straight line from the clean payload to the registered donor payload.
            stacked = graph.x[None] + float(alpha) * (stacked - graph.x[None])
            delta = (clean[None] - fn(stacked))[None]
            events = functional_carriage_events(delta, gradient)[0]
        magnitudes.append(events.numpy())
        doses.append(float(alpha) * np.asarray([r.dose for r in records], dtype=np.float64))
        gaps.append(np.asarray([r.degree_gap for r in records], dtype=np.float64))
    return R.CarriageEvents(np.arange(n), np.stack(magnitudes), np.stack(doses), np.stack(gaps))


# --------------------------------------------------------------------------------------
# Model-free behavioural reference: resample a whole distance shell
# --------------------------------------------------------------------------------------


def shell_resample_range(fn, x: torch.Tensor, distances: np.ndarray, *, repeats: int, rng) -> float:
    """Gradient-free probe: randomise one node at distance ``d`` and watch the read node move.

    ``response(d) = E |F(X)_u - F(X with one node of the d-shell resampled)_u|`` and the reported
    range is its expected distance.

    Gradient-free, but NOT independent of carriage: same finite payload replacement, same
    output-norm readout, same distance weighting (shell totals, matching how both ranges weight a
    distance), and here the replacement is drawn from the same standard-normal law as the donor
    payloads.  It differs by dropping the dose normalisation -- so it is the sibling of the
    un-normalised ``F_sens``, not of the normalised field -- and by perturbing one node per shell
    rather than every source.  Treat it as a robustness check on those two choices; it is not an
    arbiter of tangent versus secant, since it lives entirely on the secant side.
    """

    n = x.shape[0]
    with torch.no_grad():
        clean = fn(x)
    ranges = []
    for node in range(n):
        row = distances[node]
        response, weights = [], []
        for d in np.unique(row[np.isfinite(row)]).astype(int):
            shell = np.flatnonzero(row == d)
            batch = x[None].repeat(repeats, 1, 1).clone()
            picked = rng.integers(0, shell.size, size=repeats)
            batch[np.arange(repeats), shell[picked], :] = torch.tensor(
                rng.standard_normal((repeats, x.shape[1])), dtype=R.DTYPE
            )
            with torch.no_grad():
                moved = (clean[None, node] - fn(batch)[:, node]).norm(dim=-1).mean()
            # Both ranges weight a distance by the TOTAL influence mass over its shell, so the
            # per-node mean response has to be scaled back up by the shell size to be comparable.
            # Without this the probe silently reweights every distance by 1/|shell| and
            # under-reports (17% on a linear control whose analytic range is known).
            response.append(float(moved) * shell.size)
            weights.append(float(d))
        response = np.asarray(response)
        total = response.sum()
        ranges.append(float((response * np.asarray(weights)).sum() / total) if total > 0 else np.nan)
    return R.safe_nanmean(np.asarray(ranges))


# --------------------------------------------------------------------------------------


def measure(fn, base, pool, distances, *, donors, alpha, event_seed):
    jac, per_graph = [], []
    for index, graph in enumerate(base):
        influence = R.influence_matrix_autograd(fn, graph.x)
        jac.append(R.safe_nanmean(R.normalised_range_from_influence(influence, distances)))
        events = dose_scaled_carriage(
            graph,
            fn,
            pool,
            graph_id=index,
            donors=donors,
            rng=np.random.default_rng(event_seed + 977 * index),
            alpha=alpha,
        )
        per_graph.append(
            {
                "magnitude": events.magnitude,
                "dose": events.dose,
                "distances": distances,
                "sources": events.sources,
            }
        )
    return (
        R.bootstrap_graph_values(jac),
        R.bootstrap_carriage_range(per_graph, normalise=True),
        R.bootstrap_carriage_range(per_graph, normalise=False),
    )


def channel_mixing_sweep(args) -> list[dict]:
    """L1-over-channel-pairs versus L2-over-outputs, on a task with a *diffuse* far block.

    Both measures see the same Frobenius mass at every distance.  The paper's influence sums
    ``|dF^a_u/dx^b_v|`` over channel pairs, so a block whose mass is spread entrywise counts
    ``sqrt(d)`` times more than a conformal block of equal energy; carriage contracts the donor
    displacement through the block first, so it sees only the singular-value profile.  The two
    ranges must therefore separate as the channel count grows.
    """

    rows = []
    for channels in args.channel_counts:
        rng = np.random.default_rng(args.data_seed)
        base = R.make_graphs(args.graphs, nodes=args.nodes, channels=channels, rng=rng)
        donor_graphs = R.make_graphs(args.donor_graphs, nodes=args.nodes, channels=channels, rng=rng)
        pool = R.SemanticDonorPool([(10_000 + i, g) for i, g in enumerate(donor_graphs)])
        distances = R.shortest_path_distances(base[0].edge_index, base[0].num_nodes)
        near = torch.eye(channels, dtype=R.DTYPE)
        # Rank-one, entrywise-spread block of the same Frobenius norm as the identity.
        diffuse = torch.ones(channels, channels, dtype=R.DTYPE) / float(channels) ** 0.5
        mask = far_mask(distances, args.k)

        def fn(x: torch.Tensor, near=near, diffuse=diffuse, mask=mask) -> torch.Tensor:
            return args.local * (x @ near.T) + args.far * ((mask @ x) @ diffuse.T)

        jac, carriage, raw = measure(
            fn, base, pool, distances, donors=args.donors, alpha=1.0, event_seed=args.event_seed
        )
        rows.append(
            {
                "channels": int(channels),
                "jacobian": jac.__dict__,
                "carriage": carriage.__dict__,
                "carriage_raw": raw.__dict__,
            }
        )
        print(
            f"channels   d={channels:<6} jacobian={jac.estimate:6.3f} "
            f"carriage={carriage.estimate:6.3f} ratio={jac.estimate / carriage.estimate:5.2f}",
            flush=True,
        )
    return rows


def run(args) -> dict:
    rng = np.random.default_rng(args.data_seed)
    base = R.make_graphs(args.graphs, nodes=args.nodes, channels=1, rng=rng)
    donor_graphs = R.make_graphs(args.donor_graphs, nodes=args.nodes, channels=1, rng=rng)
    pool = R.SemanticDonorPool([(10_000 + i, g) for i, g in enumerate(donor_graphs)])
    distances = R.shortest_path_distances(base[0].edge_index, base[0].num_nodes)
    mask = far_mask(distances, args.k)

    payload: dict = {"config": vars(args), "sweeps": {}}

    # 1. Dose interpolation on a nonlinear task: alpha -> 0 recovers the Jacobian range.
    fn = make_task("step", args.dose_tau, mask, local=args.local, far=args.far)
    rows = []
    for alpha in args.alphas:
        jac, carriage, raw = measure(
            fn, base, pool, distances, donors=args.donors, alpha=alpha, event_seed=args.event_seed
        )
        rows.append(
            {
                "alpha": alpha,
                "jacobian": jac.__dict__,
                "carriage": carriage.__dict__,
                "carriage_raw": raw.__dict__,
            }
        )
        print(f"dose  alpha={alpha:<6} jacobian={jac.estimate:6.3f} carriage={carriage.estimate:6.3f}", flush=True)
    payload["sweeps"]["dose"] = rows

    # 2 & 3. Nonlinearity sweeps at the registered full donor dose.
    for kind, values in (("step", args.sharpness), ("oscillate", args.frequencies)):
        rows = []
        for value in values:
            fn = make_task(kind, value, mask, local=args.local, far=args.far)
            jac, carriage, raw = measure(
                fn, base, pool, distances, donors=args.donors, alpha=1.0, event_seed=args.event_seed
            )
            shell = R.trimmed_mean(
                np.asarray(
                    [
                        shell_resample_range(
                            fn,
                            graph.x,
                            distances,
                            repeats=args.shell_repeats,
                            rng=np.random.default_rng(args.event_seed + index),
                        )
                        for index, graph in enumerate(base)
                    ]
                )
            )
            rows.append(
                {
                    "parameter": value,
                    "jacobian": jac.__dict__,
                    "carriage": carriage.__dict__,
                    "carriage_raw": raw.__dict__,
                    "shell_reference": float(shell),
                }
            )
            print(
                f"{kind:<10} p={value:<6} jacobian={jac.estimate:6.3f} "
                f"carriage={carriage.estimate:6.3f} raw={raw.estimate:6.3f} "
                f"shell={float(shell):6.3f}",
                flush=True,
            )
        payload["sweeps"][kind] = rows

    # 4. Channel aggregation: L1 over channel pairs versus L2 over outputs.
    payload["sweeps"]["channels"] = channel_mixing_sweep(args)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=int, default=12)
    parser.add_argument("--donor-graphs", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=25)
    parser.add_argument("--donors", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--local", type=float, default=1.0)
    parser.add_argument("--far", type=float, default=2.0)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.001, 0.01, 0.05, 0.2, 0.5, 1.0])
    parser.add_argument("--dose-tau", type=float, default=0.1)
    parser.add_argument(
        "--sharpness", type=float, nargs="+", default=[1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    )
    parser.add_argument("--frequencies", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    parser.add_argument("--shell-repeats", type=int, default=48)
    parser.add_argument("--channel-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--data-seed", type=int, default=20260728)
    parser.add_argument("--event-seed", type=int, default=17_071)
    parser.add_argument("--out", default="results/divergence.json")
    args = parser.parse_args()

    result = run(args)
    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
