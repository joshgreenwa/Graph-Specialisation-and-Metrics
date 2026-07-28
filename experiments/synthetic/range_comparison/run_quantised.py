"""A checkable head-to-head: does a tangent measure survive a quantised dependence?

DESIGN.  One operator family on a cycle graph (homogeneous: no boundary corrections, every node
identical), scalar features (so L1-over-channels and L2-over-outputs coincide), and exactly two
contributing sources per output node:

    F(X)_v = a * x_v  +  b * g_tau(x_{v+k}),      g_tau(z) = tanh(z / tau)

Mass sits at distance 0 and at distance k and nowhere else.  ``tau`` is the only knob: large tau is
linear, small tau is a sign step whose derivative is zero almost everywhere while the *functional*
dependence on ``x_{v+k}`` is undiminished.

THE REFERENCE, AND ITS ANCHOR.  A first-order spread decomposition of the operator, by Monte Carlo
from the operator definition alone.  Each source's weight is the spread of the contribution it
makes: ``w_0 = |a| spread(x)``, ``w_k = |b| spread(g_tau(x))``, ``range = k w_k / (w_0 + w_k)``.

Any spread functional that is positively homogeneous of degree 1 and translation invariant reduces
this **exactly** to the paper's ``rho_hat`` when ``g`` is linear -- proven, and checked here at
``a != b`` so the anchor is not degenerate.  THREE such conventions are reported, because each one
is some measure's own sufficient statistic and picking one would decide the winner by definition:

    sd   standard deviation           -- algebraically IDENTICAL to sqrt(S_B) here
    mad  mean absolute deviation      -- E|t - E t|
    gmd  Gini mean difference E|t-t'| -- the statistic un-normalised F_sens actually estimates

AGGREGATION IS A SEPARATE AXIS FROM TANGENT-VERSUS-FINITE, and conflating them was the first
version's mistake.  The paper prescribes a mean of per-node ratios (Table 1); carriage averages
donor events *before* taking the ratio, and that inner average is what suppresses a Jensen
collapse.  Four arms disentangle the two effects:

    jacobian_meanratio    paper-faithful: one clean point, ratio per node, then mean
    jacobian_ratiomeans   same derivative, pooled numerator/denominator (no per-node ratio)
    jacobian_smooth       derivative averaged over the SAME donor-perturbed inputs, then ratio
    carriage_perevent     carriage with the ratio taken per event, then averaged

Only ``jacobian_meanratio`` versus ``carriage_normalised`` mixes the two axes; the other pairs
isolate one at a time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import rangelib as R
from graph_specialisation_metrics.carriage.core import integrated_loss_carriage
from graph_specialisation_metrics.methodology.events import build_channel_events

DTYPE = torch.float64


def cycle_edge_index(n: int) -> torch.Tensor:
    left = np.arange(n)
    right = (left + 1) % n
    return torch.tensor(
        np.stack([np.concatenate([left, right]), np.concatenate([right, left])]), dtype=torch.long
    )


def nonlinearity(name: str, tau: float):
    if name == "linear":
        return lambda z: z
    if name == "tanh":
        return lambda z: torch.tanh(z / tau)
    raise ValueError(name)


def make_operator(n: int, k: int, a: float, b: float, g):
    index = torch.tensor((np.arange(n) + k) % n, dtype=torch.long)

    def apply(x: torch.Tensor) -> torch.Tensor:
        return a * x + b * g(x.index_select(-2, index))

    return apply


# --------------------------------------------------------------------------------------
# Reference
# --------------------------------------------------------------------------------------


def ground_truth(g, *, a: float, b: float, k: int, samples: int, rng) -> dict:
    z = torch.tensor(rng.standard_normal(samples), dtype=DTYPE)
    z2 = torch.tensor(rng.standard_normal(samples), dtype=DTYPE)
    near, far = a * z, b * g(z)
    near2, far2 = a * z2, b * g(z2)
    conventions = {
        "sd": lambda t, t2: float(t.std(unbiased=True)),
        "mad": lambda t, t2: float((t - t.mean()).abs().mean()),
        "gmd": lambda t, t2: float((t - t2).abs().mean()),
    }
    out = {}
    for label, spread in conventions.items():
        w0, wk = spread(near, near2), spread(far, far2)
        fraction = wk / (w0 + wk)
        out[label] = {"w_near": w0, "w_far": wk, "far_fraction": fraction, "range": k * fraction}
    return out


# --------------------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------------------


def _range_from_weights(near: np.ndarray, far: np.ndarray, k: int, *, pooled: bool) -> float:
    """Mean of per-node ratios (the paper's Table 1 rule), or the pooled ratio of means."""

    if pooled:
        total = near.sum() + far.sum()
        return float(k * far.sum() / total) if total > 0 else float("nan")
    total = near + far
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(total > 0, k * far / np.where(total > 0, total, 1.0), np.nan)
    return R.safe_nanmean(ratio)


def jacobian_weights(fn, x: torch.Tensor, distances: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    jac = torch.autograd.functional.jacobian(fn, x, vectorize=True)
    influence = jac.abs().sum(dim=(1, 3)).detach().numpy()
    n = influence.shape[0]
    near = np.asarray([influence[u, u] for u in range(n)])
    far = np.asarray([influence[u][distances[u] == k].sum() for u in range(n)])
    return near, far


def carriage_weights(field: np.ndarray, distances: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = field.shape[0]
    near = np.asarray([field[i, i] for i in range(n)])
    far = np.asarray([field[i][distances[i] == k].sum() for i in range(n)])
    return near, far


def measure_graph(graph, fn, pool, *, graph_id, donors, k, rng, noise_rng, args) -> dict:
    n = graph.num_nodes
    distances = R.shortest_path_distances(graph.edge_index, n)
    out = {}

    # --- tangent arms ------------------------------------------------------------------
    near, far = jacobian_weights(fn, graph.x, distances, k)
    out["jacobian_meanratio"] = _range_from_weights(near, far, k, pooled=False)
    out["jacobian_ratiomeans"] = _range_from_weights(near, far, k, pooled=True)

    # --- donor events (shared by every finite arm) --------------------------------------
    with torch.no_grad():
        clean = fn(graph.x)
    magnitudes, doses, perturbed = [], [], []
    for source in range(n):
        variants, records = build_channel_events(
            graph,
            graph_id=graph_id,
            source=int(source),
            channel="semantic",
            stage="quantised",
            donors=donors,
            rng=rng,
            task=R.TASK,
            semantic_pool=pool,
            duplicate_tolerance=0.0,
        )
        stacked = torch.stack([v.x for v in variants])
        perturbed.append(stacked)
        with torch.no_grad():
            magnitudes.append((clean.unsqueeze(0) - fn(stacked)).abs().squeeze(-1).numpy())
        doses.append(np.asarray([r.dose for r in records], dtype=np.float64))
    magnitude = np.stack(magnitudes)  # [source, donor, carrier]
    dose = np.stack(doses)

    # Smoothed tangent: same inner average as carriage, still a derivative.
    smooth_near, smooth_far = np.zeros(n), np.zeros(n)
    for source in range(n):
        for j in range(donors):
            nr, fr = jacobian_weights(fn, perturbed[source][j], distances, k)
            smooth_near += nr
            smooth_far += fr
    smooth_near /= n * donors
    smooth_far /= n * donors
    out["jacobian_smooth"] = _range_from_weights(smooth_near, smooth_far, k, pooled=False)

    # --- functional carriage ------------------------------------------------------------
    normalised = (magnitude / dose[:, :, None]).mean(axis=1).T
    raw = magnitude.mean(axis=1).T
    for label, field in (("carriage_normalised", normalised), ("carriage_raw", raw)):
        nr, fr = carriage_weights(field, distances, k)
        out[label] = _range_from_weights(nr, fr, k, pooled=False)

    # Ratio taken per event, then averaged -- matches the tangent's single-sample ratio.
    per_event = []
    for j in range(donors):
        field = (magnitude[:, j, :] / dose[:, j, None]).T
        nr, fr = carriage_weights(field, distances, k)
        per_event.append(_range_from_weights(nr, fr, k, pooled=False))
    out["carriage_perevent"] = float(np.nanmean(per_event))

    # --- beneficial carriage ------------------------------------------------------------
    targets = clean + torch.tensor(noise_rng.standard_normal(clean.shape) * args.noise, dtype=DTYPE)

    def loss_from_states(states: torch.Tensor) -> torch.Tensor:
        return ((states - targets.unsqueeze(0)) ** 2).mean(dim=(-2, -1))

    with torch.no_grad():
        swap = torch.cat([fn(block) for block in perturbed], dim=0)
    path = integrated_loss_carriage(
        clean.unsqueeze(0).expand_as(swap),
        swap,
        loss_from_states=loss_from_states,
        atol=args.atol,
        rtol=args.rtol,
        max_intervals=args.max_intervals,
    )
    field = -path["carriage"].reshape(n, donors, n).mean(dim=1).T.numpy()
    nr, fr = carriage_weights(field, distances, k)
    out["beneficial_raw"] = _range_from_weights(nr, fr, k, pooled=False)
    out["beneficial_sqrt"] = _range_from_weights(
        np.sqrt(np.clip(nr, 0, None)), np.sqrt(np.clip(fr, 0, None)), k, pooled=False
    )
    out["_intervals_max"] = float(path["intervals"].max())
    out["_residual"] = float(path["completeness_residual"].abs().max())
    return out


ARMS = (
    "jacobian_meanratio",
    "jacobian_ratiomeans",
    "jacobian_smooth",
    "carriage_perevent",
    "carriage_normalised",
    "carriage_raw",
    "beneficial_sqrt",
    "beneficial_raw",
)


def run(args) -> dict:
    rng = np.random.default_rng(args.data_seed)
    edge_index = cycle_edge_index(args.nodes)
    base = [
        R.TinyData(torch.tensor(rng.standard_normal((args.nodes, 1)), dtype=DTYPE), edge_index.clone())
        for _ in range(args.graphs)
    ]
    donor_graphs = [
        R.TinyData(torch.tensor(rng.standard_normal((args.nodes, 1)), dtype=DTYPE), edge_index.clone())
        for _ in range(args.donor_graphs)
    ]
    pool = R.SemanticDonorPool([(10_000 + i, g) for i, g in enumerate(donor_graphs)])

    settings = [("linear", float("inf"))] + [("tanh", t) for t in args.taus]
    rows = []
    for name, tau in settings:
        started = time.time()
        g = nonlinearity(name, tau)
        fn = make_operator(args.nodes, args.k, args.a, args.b, g)
        truth = ground_truth(
            g, a=args.a, b=args.b, k=args.k, samples=args.mc_samples,
            rng=np.random.default_rng(args.data_seed + 7),
        )
        per_graph = [
            measure_graph(
                graph, fn, pool, graph_id=index, donors=args.donors, k=args.k,
                rng=np.random.default_rng(args.event_seed + 977 * index),
                # A dedicated stream: the first version aliased this with the data seed, so
                # graph 0's "noisy target" was a multiple of its own features.
                noise_rng=np.random.default_rng([args.noise_seed, index]),
                args=args,
            )
            for index, graph in enumerate(base)
        ]
        row = {"nonlinearity": name, "tau": None if np.isinf(tau) else tau, "truth": truth}
        for arm in ARMS:
            values = [entry[arm] for entry in per_graph]
            interval = R.bootstrap_graph_values(values)
            row[arm] = {
                "range": interval.estimate,
                "low": interval.low,
                "high": interval.high,
            }
        row["quadrature_intervals_max"] = float(max(e["_intervals_max"] for e in per_graph))
        row["completeness_residual_max"] = float(max(e["_residual"] for e in per_graph))
        row["seconds"] = round(time.time() - started, 1)
        rows.append(row)
        label = "linear" if name == "linear" else f"tau={tau:g}"
        print(
            f"{label:>10}  truth[sd/mad/gmd]="
            f"{truth['sd']['range']:.3f}/{truth['mad']['range']:.3f}/{truth['gmd']['range']:.3f}  "
            f"jac={row['jacobian_meanratio']['range']:.3f}  "
            f"jac_pool={row['jacobian_ratiomeans']['range']:.3f}  "
            f"jac_smooth={row['jacobian_smooth']['range']:.3f}  "
            f"F~={row['carriage_normalised']['range']:.3f}  "
            f"F~_event={row['carriage_perevent']['range']:.3f}  "
            f"sqrtB={row['beneficial_sqrt']['range']:.3f}  ({row['seconds']}s)",
            flush=True,
        )
    return {"config": {k: str(v) for k, v in vars(args).items()}, "arms": list(ARMS), "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--graphs", type=int, default=60)
    parser.add_argument("--donor-graphs", type=int, default=8)
    parser.add_argument("--donors", type=int, default=8)
    parser.add_argument("--taus", type=float, nargs="+", default=[2.0, 1.0, 0.5, 0.25, 0.1, 0.05])
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--mc-samples", type=int, default=400_000)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--max-intervals", type=int, default=512)
    parser.add_argument("--data-seed", type=int, default=20260728)
    parser.add_argument("--event-seed", type=int, default=17_071)
    parser.add_argument("--noise-seed", type=int, default=90_210)
    parser.add_argument("--out", default="results/quantised.json")
    args = parser.parse_args()

    payload = run(args)
    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
