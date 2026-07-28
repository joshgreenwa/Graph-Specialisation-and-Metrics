"""A far gate controlling a near value: can a first-order range measure see an interaction?

THE TASK.  On a cycle of ``n`` nodes with scalar i.i.d. N(0,1) features,

    F(X)_v = a * x_v  +  b * sigma(x_{v+j} / tau) * x_{v+k}

The node at distance ``j`` is a GATE: it decides whether the value at distance ``k`` is used.  This
is multiplicative gating -- the mechanism behind attention and gated message passing -- so the
nonlinearity is an interaction, which is a structural blind spot for a first-order method rather
than a constructed cliff.  ``tau -> inf`` drives the gate to the constant 1/2 and the whole map
becomes linear, giving an anchor where every method must agree.

We put the gate FAR (``j = 8``) and the value it gates NEAR (``k = 3``), because that is the case
where the two ground truths diverge most: a far node deciding whether near information is used is
exactly the "is this task long-range?" question.

THE GROUND TRUTH.  Standard Sobol sensitivity indices for the three inputs of one output node,
estimated by Monte Carlo from the operator definition alone (Saltelli/Jansen estimators):

    first-order S_1(d)  variance explained by input d ALONE
    total       S_T(d)  variance that vanishes when input d is fixed (main + all interactions)

The gate's first-order index is exactly zero -- its mean effect vanishes because the value it gates
is mean-zero -- while its total index is large.  Weights are taken as sqrt(S), the degree-1
homogeneous choice that reduces the reference exactly to the paper's rho_hat in the linear case;
the anchor is checked numerically rather than assumed.

THE MEASURES, each as its own specification requires:

    jacobian_pooled   paper-faithful influence, aggregated as a ratio of means -- the STRONGEST
                      form, since the per-node ratio of Table 1 was shown separately to contribute
                      about half the error of the naive estimator.
    F_sens            functional carriage per the repository README: production donor law,
                      production eventwise magnitude before donor averaging, identity readout;
                      reported event-normalised and raw.
    S_B               Beneficial carriage per the README, through the REGISTERED
                      ``beneficial_carriage`` wrapper (so its completeness audit actually fires),
                      reported as the section 7 signed mass per distance -- never as an expected
                      distance, which is not defined for a signed field.

Because all mass sits at exactly three distances, the comparable object is the normalised
mass-by-distance PROFILE, which every method produces and both ground truths define.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import rangelib as R
from graph_specialisation_metrics.methodology.carriage import beneficial_carriage
from graph_specialisation_metrics.methodology.events import build_channel_events

DTYPE = torch.float64


def cycle_edge_index(n: int) -> torch.Tensor:
    left = np.arange(n)
    right = (left + 1) % n
    return torch.tensor(
        np.stack([np.concatenate([left, right]), np.concatenate([right, left])]), dtype=torch.long
    )


def gate_fn(tau: float):
    if np.isinf(tau):
        return lambda z: torch.full_like(z, 0.5)
    return lambda z: torch.sigmoid(z / tau)


def make_operator(n: int, j: int, k: int, a: float, b: float, gate):
    gate_index = torch.tensor((np.arange(n) + j) % n, dtype=torch.long)
    value_index = torch.tensor((np.arange(n) + k) % n, dtype=torch.long)

    def apply(x: torch.Tensor) -> torch.Tensor:
        return a * x + b * gate(x.index_select(-2, gate_index)) * x.index_select(-2, value_index)

    return apply


# --------------------------------------------------------------------------------------
# Ground truth: Sobol indices of one output node, by Monte Carlo
# --------------------------------------------------------------------------------------


def sobol(gate, *, a: float, b: float, samples: int, rng) -> dict:
    """First-order and total indices for the three inputs (near, gate, value).

    ``y = a*x0 + b*g(x1)*x2`` with independent standard normal inputs.  Jansen estimators:
    S_1(i) from resampling everything EXCEPT i, S_T(i) from resampling i alone.
    """

    def evaluate(m: np.ndarray) -> np.ndarray:
        t = torch.tensor(m, dtype=DTYPE)
        return (a * t[:, 0] + b * gate(t[:, 1]) * t[:, 2]).numpy()

    A = rng.standard_normal((samples, 3))
    B = rng.standard_normal((samples, 3))
    fa, fb = evaluate(A), evaluate(B)
    variance = float(np.var(np.concatenate([fa, fb]), ddof=1))

    first, total, total_abs = {}, {}, {}
    for i in range(3):
        AB = A.copy()
        AB[:, i] = B[:, i]          # input i replaced -> everything else held
        f_ab = evaluate(AB)
        # Jansen: S_T(i) uses the variance created by moving i alone.
        total[i] = float(np.mean((fa - f_ab) ** 2) / (2.0 * variance))
        # Mean-absolute total effect.  The variance convention is what sqrt(S_B) estimates by
        # construction, so a second, non-quadratic convention is required or the comparison is
        # circular -- this one is what an absolute-response measure estimates instead.
        total_abs[i] = float(np.mean(np.abs(fa - f_ab)))
        BA = B.copy()
        BA[:, i] = A[:, i]          # input i held, everything else resampled
        f_ba = evaluate(BA)
        first[i] = float(1.0 - np.mean((fa - f_ba) ** 2) / (2.0 * variance))
    return {"first": first, "total": total, "total_abs": total_abs, "variance": variance}


def truth_profile(indices: dict, key: str, *, j: int, k: int) -> dict:
    """Degree-1 homogeneous weights at distances 0, j, k, so the linear anchor is exact.

    Variance-flavoured indices need a square root to be degree 1; the mean-absolute total effect
    already is.
    """

    root = key != "total_abs"
    def weight(i):
        v = max(indices[key][i], 0.0)
        return np.sqrt(v) if root else v

    weights = {0: weight(0), j: weight(1), k: weight(2)}
    total = sum(weights.values())
    share = {d: (w / total if total > 0 else float("nan")) for d, w in weights.items()}
    return {
        "share": share,
        "range": float(sum(d * s for d, s in share.items())),
        "gate_share": share[j],
    }


# --------------------------------------------------------------------------------------
# Measures
# --------------------------------------------------------------------------------------


def profile_from_weights(near_j_k: dict, *, j: int, k: int) -> dict:
    total = sum(near_j_k.values())
    share = {d: (w / total if total > 0 else float("nan")) for d, w in near_j_k.items()}
    return {
        "share": share,
        "range": float(sum(d * s for d, s in share.items())),
        "gate_share": share[j],
    }


def measure_graph(graph, fn, pool, *, graph_id, donors, j, k, rng, noise_rng, args) -> dict:
    n = graph.num_nodes
    distances = R.shortest_path_distances(graph.edge_index, n)
    out = {}

    # --- tangent, pooled (ratio of means: the strongest form) ---------------------------
    jac = torch.autograd.functional.jacobian(fn, graph.x, vectorize=True)
    influence = jac.abs().sum(dim=(1, 3)).detach().numpy()
    weights = {
        d: float(np.mean([influence[u][distances[u] == d].sum() for u in range(n)]))
        for d in (0, j, k)
    }
    out["jacobian_pooled"] = profile_from_weights(weights, j=j, k=k)

    # --- donor events -------------------------------------------------------------------
    with torch.no_grad():
        clean = fn(graph.x)
    magnitudes, doses, perturbed = [], [], []
    for source in range(n):
        variants, records = build_channel_events(
            graph, graph_id=graph_id, source=int(source), channel="semantic", stage="gate",
            donors=donors, rng=rng, task=R.TASK, semantic_pool=pool, duplicate_tolerance=0.0,
        )
        stacked = torch.stack([v.x for v in variants])
        perturbed.append(stacked)
        with torch.no_grad():
            magnitudes.append((clean.unsqueeze(0) - fn(stacked)).abs().squeeze(-1).numpy())
        doses.append(np.asarray([r.dose for r in records], dtype=np.float64))
    magnitude = np.stack(magnitudes)
    dose = np.stack(doses)

    for label, field in (
        ("carriage_normalised", (magnitude / dose[:, :, None]).mean(axis=1).T),
        ("carriage_raw", magnitude.mean(axis=1).T),
    ):
        weights = {
            d: float(np.mean([field[i][distances[i] == d].sum() for i in range(n)]))
            for d in (0, j, k)
        }
        out[label] = profile_from_weights(weights, j=j, k=k)

    # --- Beneficial carriage through the registered wrapper ------------------------------
    targets = clean + torch.tensor(noise_rng.standard_normal(clean.shape) * args.noise, dtype=DTYPE)

    def loss_from_states(states: torch.Tensor) -> torch.Tensor:
        return ((states - targets.unsqueeze(0)) ** 2).mean(dim=(-2, -1))

    with torch.no_grad():
        events = torch.stack([fn(block) for block in perturbed])  # [source, donor, carrier, 1]
    result = beneficial_carriage(
        clean,
        events,
        loss_from_states=loss_from_states,
        atol=args.atol,
        rtol=args.rtol,
        max_intervals=args.max_intervals,
        tolerance=args.tolerance,
    )
    field = result.field.numpy()  # [carrier, source]
    # Section 7: signed mass per distance bin.  Never an expected distance.
    signed = {
        d: float(np.mean([field[i][distances[i] == d].sum() for i in range(n)])) for d in (0, j, k)
    }
    out["beneficial_S_B"] = signed
    # For a profile comparison the degree-1 comparable is sqrt of a squared-loss allocation.
    out["beneficial_sqrt"] = profile_from_weights(
        {d: float(np.sqrt(max(v, 0.0))) for d, v in signed.items()}, j=j, k=k
    )
    out["_residual"] = float(result.completeness_residual.abs().max())
    out["_intervals"] = float(result.intervals.max())
    return out


ARMS = ("jacobian_pooled", "carriage_normalised", "carriage_raw", "beneficial_sqrt")


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

    rows = []
    for tau in [float("inf")] + list(args.taus):
        started = time.time()
        gate = gate_fn(tau)
        fn = make_operator(args.nodes, args.j, args.k, args.a, args.b, gate)
        indices = sobol(
            gate, a=args.a, b=args.b, samples=args.mc_samples,
            rng=np.random.default_rng(args.data_seed + 7),
        )
        truth = {
            "first": truth_profile(indices, "first", j=args.j, k=args.k),
            "total": truth_profile(indices, "total", j=args.j, k=args.k),
            "total_abs": truth_profile(indices, "total_abs", j=args.j, k=args.k),
            "indices": indices,
        }
        per_graph = [
            measure_graph(
                graph, fn, pool, graph_id=index, donors=args.donors, j=args.j, k=args.k,
                rng=np.random.default_rng(args.event_seed + 977 * index),
                noise_rng=np.random.default_rng([args.noise_seed, index]),
                args=args,
            )
            for index, graph in enumerate(base)
        ]
        row = {"tau": None if np.isinf(tau) else tau, "truth": truth}
        for arm in ARMS:
            row[arm] = {
                "range": R.bootstrap_graph_values([e[arm]["range"] for e in per_graph]).__dict__,
                "gate_share": R.bootstrap_graph_values(
                    [e[arm]["gate_share"] for e in per_graph]
                ).__dict__,
            }
        row["S_B"] = {
            str(d): float(np.mean([e["beneficial_S_B"][d] for e in per_graph]))
            for d in (0, args.j, args.k)
        }
        row["completeness_residual_max"] = float(max(e["_residual"] for e in per_graph))
        row["quadrature_intervals_max"] = float(max(e["_intervals"] for e in per_graph))
        row["seconds"] = round(time.time() - started, 1)
        rows.append(row)
        label = "linear" if np.isinf(tau) else f"tau={tau:g}"
        print(
            f"{label:>9}  truth[1st/tot-var/tot-abs]="
            f"{truth['first']['range']:.2f}/{truth['total']['range']:.2f}/{truth['total_abs']['range']:.2f}"
            f"  ||  jac={row['jacobian_pooled']['range']['estimate']:.2f}"
            f"(g={row['jacobian_pooled']['gate_share']['estimate']:.3f})"
            f"  F~={row['carriage_normalised']['range']['estimate']:.2f}"
            f"(g={row['carriage_normalised']['gate_share']['estimate']:.3f})"
            f"  sqrtB={row['beneficial_sqrt']['range']['estimate']:.2f}"
            f"(g={row['beneficial_sqrt']['gate_share']['estimate']:.3f})"
            f"  [{row['seconds']}s]",
            flush=True,
        )
    return {"config": {k: str(v) for k, v in vars(args).items()}, "arms": list(ARMS), "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--j", type=int, default=8, help="distance to the GATE")
    parser.add_argument("--k", type=int, default=3, help="distance to the gated VALUE")
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=2.0)
    parser.add_argument("--graphs", type=int, default=40)
    parser.add_argument("--donor-graphs", type=int, default=8)
    parser.add_argument("--donors", type=int, default=8)
    parser.add_argument("--taus", type=float, nargs="+", default=[2.0, 1.0, 0.5, 0.2, 0.05])
    parser.add_argument("--noise", type=float, default=0.3)
    parser.add_argument("--mc-samples", type=int, default=400_000)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--max-intervals", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--data-seed", type=int, default=20260728)
    parser.add_argument("--event-seed", type=int, default=17_071)
    parser.add_argument("--noise-seed", type=int, default=90_210)
    parser.add_argument("--out", default="results/gate.json")
    args = parser.parse_args()

    payload = run(args)
    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
