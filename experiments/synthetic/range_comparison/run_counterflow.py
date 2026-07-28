"""A saturated long-range pathway with local counterflow.

CONSTRUCTION (path graph, one source ``s``, a near carrier at ``d=1``, a far carrier at ``d=D``):

    h_1 = -gamma * b                     local pathway, linear in the source feature
    h_D = (1 + gamma) * phi_kappa(b)     far pathway, saturating
    phi_kappa(b) = tanh(kappa b) / tanh(kappa)
    y_hat = h_1 + h_D                    sum pooling

The source feature is ``b in {-1, +1}``, and ``phi_kappa(+-1) = +-1`` EXACTLY for every kappa.  So
the prediction is ``y_hat = b`` and a donor swap that flips the sign gives ``y_hat = -b``: the
finite response is completely invariant to ``kappa``, while the derivative of the far pathway,
``(1+gamma) phi'_kappa(b)``, vanishes as the pathway saturates.  That is the whole point -- the
tangent and the finite response are driven apart by a parameter that changes neither the function's
values on the data nor the task.

WHAT EACH MEASURE SHOULD SAY

    Jacobian range   rho_J = [gamma + (1+gamma) phi' D] / [gamma + (1+gamma) phi']  ->  1 as
                     kappa grows: it reports a purely local computation.
    F_sens           F(1) = 2 gamma, F(D) = 2 (1+gamma), independent of kappa: both pathways are
                     active and the far one dominates.
    B                with target y = b and an L1 loss, B(1) = -2 gamma and B(D) = 2 (1+gamma):
                     the far pathway is beneficial, the LOCAL pathway actively counteracts it, and
                     B(1) + B(D) = 2 is exactly the donor-swap loss increase.

The control sweeps the target to ``y = tau b``.  The Jacobian range and F_sens cannot move -- they
never see the target -- while B scales with ``tau`` and vanishes at ``tau = 0``.  That separates
label-dependent information from mere response magnitude.  It also puts the L1 kink strictly inside
the integration path (it sits at ``alpha = (1+tau)/2``), which finally exercises the adaptive
Gauss-Kronrod refinement that a quadratic loss on an affine path leaves untouched.

ORIENTATION -- BOTH ARE REPORTED, because they behave completely differently and reporting only one
would be unfair.

``rho_src``      source-anchored: normalise the influence over CARRIERS at the fixed source.  This
                 is the orientation ``F_sens`` and ``B`` are natively defined in, so it is the only
                 one in which the three measures are comparable.  It collapses to 1 as the far
                 pathway saturates.
``rho_carrier``  the paper's own carrier-anchored ``rho_hat_u``, normalised over INPUTS at fixed
                 output, averaged over estimable outputs.  It is NOT fooled by saturation -- but
                 only because it is uninformative here: every estimable output node has exactly one
                 input, so its per-node ratio collapses to that distance and the graph mean is
                 ``(1 + D) / 2`` for every gamma and every kappa.  It therefore also fails to move
                 when gamma changes the true balance between the pathways, which the true range
                 tracks and it does not.

``true_range`` is the expected distance of the finite response.  With a binary input the donor swap
is the ONLY possible change to it, so that response profile IS the complete functional dependence
and its expected distance is exact -- there is nothing left for a derivative to add.

Production entry points only: ``functional_carriage`` for F_sens and the registered
``beneficial_carriage`` wrapper (with ``pooling='add'``, the README's registered linear carrier
projection) for B, so its completeness audit fires.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

import rangelib as R
from graph_specialisation_metrics.methodology.carriage import (
    beneficial_carriage,
    functional_carriage,
)
from graph_specialisation_metrics.methodology.events import build_channel_events

DTYPE = torch.float64


# --------------------------------------------------------------------------------------
# Geometry and the two pathways
# --------------------------------------------------------------------------------------


def layout(distance: int) -> tuple[int, int, int, int]:
    """Path graph indices: source interior (degree 2), near carrier at 1, far carrier at D."""

    source = 1
    near = source + 1
    far = source + distance
    nodes = far + 2
    return nodes, source, near, far


def phi(b: torch.Tensor, kappa: float) -> torch.Tensor:
    return torch.tanh(kappa * b) / np.tanh(kappa)


def make_model(nodes: int, source: int, near: int, far: int, *, gamma: float, kappa: float):
    """``X -> H``: only the two carriers are non-zero, and both read the single source feature."""

    def carriers(x: torch.Tensor) -> torch.Tensor:
        b = x[..., source, :]
        h = torch.zeros_like(x)
        local = -gamma * b
        distant = (1.0 + gamma) * phi(b, kappa)
        h = h.index_copy(-2, torch.tensor([near]), local.unsqueeze(-2))
        h = h.index_copy(-2, torch.tensor([far]), distant.unsqueeze(-2))
        return h

    return carriers


def pooled(h: torch.Tensor) -> torch.Tensor:
    return h.sum(dim=-2)


# --------------------------------------------------------------------------------------


def measure(graph, model, pool, *, source, near, far, distance, donors, tau, rng, args) -> dict:
    n = graph.num_nodes
    distances = R.shortest_path_distances(graph.edge_index, n)
    b = float(graph.x[source, 0])
    target = torch.tensor([tau * b], dtype=DTYPE)

    with torch.no_grad():
        h_clean = model(graph.x)

    # --- Jacobian influence from the source to each carrier -----------------------------
    jac = torch.autograd.functional.jacobian(model, graph.x, vectorize=True)  # [N,1,N,1]
    influence = jac.abs().sum(dim=(1, 3)).detach().numpy()[:, source]         # [carrier]
    mass = influence.sum()
    jac_range = float((influence * distances[:, source]).sum() / mass) if mass > 0 else float("nan")

    # The paper's OWN carrier-anchored orientation, for fairness: rho_hat_u normalises over inputs
    # at fixed output u, then averages over estimable u.  Reported alongside because it behaves
    # completely differently here -- see the note in the module docstring.
    full = jac.abs().sum(dim=(1, 3)).detach().numpy()          # [out, in]
    per_node = R.normalised_range_from_influence(full, distances)
    jac_carrier = R.safe_nanmean(per_node)

    # --- the registered semantic donor swap ---------------------------------------------
    variants, records = build_channel_events(
        graph, graph_id=0, source=int(source), channel="semantic", stage="counterflow",
        donors=donors, rng=rng, task=R.TASK, semantic_pool=pool, duplicate_tolerance=0.0,
    )
    stacked = torch.stack([v.x for v in variants])
    with torch.no_grad():
        h_event = model(stacked)
    flipped = bool(np.allclose([float(v.x[source, 0]) for v in variants], -b))

    # --- Functional carriage: production entry point ------------------------------------
    # z = y_hat (sum pooling), so g_out[t, i, m] = d y_hat / d h_{i,m}; taken by autograd on the
    # model's own graph rather than assumed to be ones.
    states = h_clean.detach().clone().requires_grad_(True)
    readout = torch.autograd.functional.jacobian(pooled, states, vectorize=True)  # [1,N,1]
    delta = (h_clean.unsqueeze(0) - h_event).unsqueeze(0)                          # [1,K,N,1]
    F = functional_carriage(delta, readout)[:, 0].numpy()                          # [carrier]

    # The complete answer.  The input is binary, so the donor swap is the ONLY possible change to
    # it: the finite response profile is the entire functional dependence, and its expected
    # distance is the exact range with nothing left for a derivative to add.
    finite_mass = F.sum()
    true_range = float((F * distances[:, source]).sum() / finite_mass) if finite_mass > 0 else float("nan")

    # --- Beneficial carriage: registered wrapper, add pooling, L1 loss -------------------
    def loss_from_pooled(values: torch.Tensor) -> torch.Tensor:
        return (values[:, 0] - target[0]).abs()

    result = beneficial_carriage(
        h_clean,
        h_event.unsqueeze(0),
        loss_from_pooled,
        pooling="add",
        atol=args.atol,
        rtol=args.rtol,
        max_intervals=args.max_intervals,
        tolerance=args.tolerance,
    )
    B = result.field[:, 0].numpy()

    return {
        "b": b,
        "donor_flipped": flipped,
        "jacobian_range": jac_range,
        "jacobian_carrier_anchored": float(jac_carrier),
        "jacobian_weight_near": float(influence[near]),
        "jacobian_weight_far": float(influence[far]),
        "true_range": true_range,
        "F_near": float(F[near]),
        "F_far": float(F[far]),
        "B_near": float(B[near]),
        "B_far": float(B[far]),
        "B_sum": float(B.sum()),
        "loss_increase": float(result.event_loss_increase.mean()),
        "residual": float(result.completeness_residual.abs().max()),
        "intervals": float(result.intervals.max()),
        "converged": float(result.converged.to(torch.float64).mean()),
    }


def analytic(*, gamma: float, kappa: float, distance: int, tau: float) -> dict:
    """Closed forms from the construction, at b = +-1 (phi(+-1) = +-1 exactly)."""

    derivative = kappa / (np.cosh(kappa) ** 2) / np.tanh(kappa)
    near_w, far_w = gamma, (1.0 + gamma) * derivative
    return {
        "jacobian_range": (near_w * 1.0 + far_w * distance) / (near_w + far_w),
        # Carrier-anchored: each estimable output node has exactly ONE input, so its per-node ratio
        # collapses to that distance and the graph mean is (1 + D) / 2 -- free of gamma and kappa.
        "jacobian_carrier_anchored": (1.0 + distance) / 2.0,
        "true_range": (gamma + distance * (1.0 + gamma)) / (1.0 + 2.0 * gamma),
        "F_near": 2.0 * gamma,
        "F_far": 2.0 * (1.0 + gamma),
        "B_near": -2.0 * gamma * tau,
        "B_far": 2.0 * (1.0 + gamma) * tau,
        "B_sum": 2.0 * tau,
    }


def run(args) -> dict:
    rows = []
    for gamma, kappa, distance, tau in itertools.product(
        args.gammas, args.kappas, args.distances, args.taus
    ):
        started = time.time()
        nodes, source, near, far = layout(distance)
        edge_index = R.path_edge_index(nodes)
        model = make_model(nodes, source, near, far, gamma=gamma, kappa=kappa)

        rng = np.random.default_rng(args.data_seed)
        # Binary payloads: donor eligibility rule 2 (payload must differ) then guarantees the
        # flip b -> -b, which is exactly the intervention the construction asks for.
        donor_graphs = [
            R.TinyData(
                torch.tensor(rng.choice([-1.0, 1.0], size=(nodes, 1)), dtype=DTYPE),
                edge_index.clone(),
            )
            for _ in range(args.donor_graphs)
        ]
        pool = R.SemanticDonorPool([(10_000 + i, g) for i, g in enumerate(donor_graphs)])

        per_graph = []
        for index in range(args.graphs):
            features = rng.choice([-1.0, 1.0], size=(nodes, 1))
            graph = R.TinyData(torch.tensor(features, dtype=DTYPE), edge_index.clone())
            per_graph.append(
                measure(
                    graph, model, pool, source=source, near=near, far=far, distance=distance,
                    donors=args.donors, tau=tau,
                    rng=np.random.default_rng(args.event_seed + 977 * index), args=args,
                )
            )

        expected = analytic(gamma=gamma, kappa=kappa, distance=distance, tau=tau)
        row = {
            "gamma": gamma, "kappa": kappa, "distance": distance, "tau": tau,
            "expected": expected,
            "measured": {
                key: float(np.mean([e[key] for e in per_graph]))
                for key in ("jacobian_range", "jacobian_carrier_anchored", "true_range",
                            "F_near", "F_far", "B_near", "B_far", "B_sum", "loss_increase")
            },
            "donor_always_flipped": bool(all(e["donor_flipped"] for e in per_graph)),
            "residual_max": float(max(e["residual"] for e in per_graph)),
            "intervals_max": float(max(e["intervals"] for e in per_graph)),
            "converged": float(np.mean([e["converged"] for e in per_graph])),
            "seconds": round(time.time() - started, 2),
        }
        rows.append(row)
        m, x = row["measured"], row["expected"]
        print(
            f"g={gamma:<4} k={kappa:<5} D={distance:<3} t={tau:<4} | "
            f"true {m['true_range']:6.3f}  rho_src {m['jacobian_range']:6.3f}"
            f"  rho_carrier {m['jacobian_carrier_anchored']:5.2f}  "
            f"F {m['F_near']:5.2f}/{m['F_far']:5.2f} (exp {x['F_near']:.2f}/{x['F_far']:.2f})  "
            f"B {m['B_near']:+5.2f}/{m['B_far']:+5.2f} (exp {x['B_near']:+.2f}/{x['B_far']:+.2f})  "
            f"sum {m['B_sum']:+5.2f}  int={row['intervals_max']:.0f}  res={row['residual_max']:.1e}",
            flush=True,
        )
    return {"config": {k: str(v) for k, v in vars(args).items()}, "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.25, 1.0, 3.0])
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--distances", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument(
        "--taus", type=float, nargs="+",
        # Off-dyadic values are deliberate: a kink landing on a Gauss-Kronrod panel
        # centre is integrated exactly by odd symmetry WITHOUT refining, so a dyadic
        # grid alone cannot demonstrate that the adaptive quadrature does any work.
        default=[1.0, 0.7, 0.5, 0.3, 0.0, -0.3, -0.5],
    )
    parser.add_argument("--graphs", type=int, default=8)
    parser.add_argument("--donor-graphs", type=int, default=6)
    parser.add_argument("--donors", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--max-intervals", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--data-seed", type=int, default=20260728)
    parser.add_argument("--event-seed", type=int, default=17_071)
    parser.add_argument("--out", default="results/counterflow.json")
    args = parser.parse_args()

    payload = run(args)
    out = Path(__file__).resolve().parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
