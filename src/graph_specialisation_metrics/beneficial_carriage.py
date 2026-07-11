"""Beneficial carriage B(d): does distance-d content *help the task*, not just move the output?

Pairs with functional carriage. For a focal readout node i* per graph and a source j in the band
d = {j : dist(i*, j) = d}, replace only j's encoded content with an on-manifold resample
x'_j ~ q(. | context) and forward-pass. From the SAME passes:

    F(d) = E[(y_hat' - y_hat)^2]                     functional (label-free: the output moved)
    B(d) = E[ |y_hat' - y| - |y_hat - y| ]           beneficial (both scored vs the CLEAN label y)

B(d) = 0 under (i) on-manifold q and (ii) y _|_ x_j | x_{-j}. A bad q fabricates benefit, so the
resampler ladder controls (i): ``marginal`` (naive baseline, measures the artifact), ``matched``
(donors sharing j's neighbourhood signature -- the practical proxy), and, where the true
conditional is known (synthetic), ``oracle``. The label-free ``var_ratio`` = Var(y_hat')/Var(y_hat)
audits q without labels (<1 => shrinkage toward the prior => bad q).

single-source B(d) is a *necessity* measure (conditions on x_{-j}); the ``whole_band`` variant
resamples the whole shell jointly to capture *redundantly*-carried benefit -- the gap between them
is the redundancy. B_train - B_test exposes memorisation of long-range content.

The backend is a small protocol so this runs on the official GRIT adapter (real models/tasks) and
on a toy model (tests) unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Protocol, Sequence

import numpy as np
import torch

Tensor = torch.Tensor


class FBCBackend(Protocol):
    """Everything B(d)/F(d) needs from a model, so the operator is model/task agnostic."""

    def encoded(self, graph: Any) -> Tensor: ...            # [n, dim] encoded content (the resample unit)
    def predict(self, graph: Any, encoded: Tensor) -> float: ...  # model scalar prediction from encoded content
    def label(self, graph: Any) -> float: ...               # true target
    def distances(self, graph: Any) -> Tensor: ...          # [n, n] shortest-path hop distances
    def degree(self, graph: Any) -> Tensor: ...             # [n]
    def signature(self, graph: Any, node: int) -> Hashable: ...  # donor match key (neighbourhood signature)


def focal_node(backend: FBCBackend, graph: Any) -> int:
    """Focal readout node = highest degree (matches the validated synthetic B(d) convention)."""
    return int(torch.as_tensor(backend.degree(graph)).argmax().item())


def distance_bands(dist_row: Tensor, focal: int, max_d: int) -> dict[int, list[int]]:
    d = torch.as_tensor(dist_row).long().tolist()
    bands: dict[int, list[int]] = {}
    for j, dj in enumerate(d):
        if j == focal or dj < 1 or dj > max_d:
            continue
        bands.setdefault(int(dj), []).append(j)
    return bands


@dataclass
class DonorBank:
    by_sig: dict[Hashable, list[Tensor]] = field(default_factory=dict)
    allv: list[Tensor] = field(default_factory=list)


def build_donor_bank(backend: FBCBackend, graphs: Sequence[Any]) -> DonorBank:
    bank = DonorBank()
    for g in graphs:
        enc = backend.encoded(g).detach()
        for node in range(int(enc.size(0))):
            v = enc[node].clone()
            bank.allv.append(v)
            bank.by_sig.setdefault(backend.signature(g, node), []).append(v)
    return bank


def _draw_donor(backend: FBCBackend, graph: Any, source: int, enc: Tensor, bank: DonorBank,
                mode: str, rng: np.random.Generator) -> Tensor:
    if mode == "marginal":
        pool = bank.allv
    else:  # matched
        pool = bank.by_sig.get(backend.signature(graph, source), []) or bank.allv
    src = enc[source]
    for _ in range(8):  # avoid the trivial no-op draw
        cand = pool[int(rng.integers(0, len(pool)))].to(enc)
        if not torch.allclose(cand, src):
            return cand
    return pool[int(rng.integers(0, len(pool)))].to(enc)


def fbc_for_graph(
    backend: FBCBackend,
    graph: Any,
    *,
    mode: str,
    bank: DonorBank,
    donors: int,
    max_d: int,
    whole_band: bool,
    rng: np.random.Generator,
) -> dict[int, dict[str, Any]]:
    """Per-band F/B accumulators for one graph (single-source, or whole-band joint resample)."""
    enc = backend.encoded(graph).detach()
    y = float(backend.label(graph))
    yhat = float(backend.predict(graph, enc))
    base_loss = abs(yhat - y)
    focal = focal_node(backend, graph)
    dist = backend.distances(graph)[focal]
    bands = distance_bands(dist, focal, max_d)

    out: dict[int, dict[str, Any]] = {}
    for d, sources in bands.items():
        f_vals, b_vals, yprimes = [], [], []
        if whole_band:
            for _ in range(donors):
                pert = enc.clone()
                for j in sources:
                    pert[j] = _draw_donor(backend, graph, j, enc, bank, mode, rng)
                yp = float(backend.predict(graph, pert))
                f_vals.append((yp - yhat) ** 2)
                b_vals.append(abs(yp - y) - base_loss)
                yprimes.append(yp)
        else:
            for j in sources:
                for _ in range(donors):
                    pert = enc.clone()
                    pert[j] = _draw_donor(backend, graph, j, enc, bank, mode, rng)
                    yp = float(backend.predict(graph, pert))
                    f_vals.append((yp - yhat) ** 2)
                    b_vals.append(abs(yp - y) - base_loss)
                    yprimes.append(yp)
        if f_vals:
            out[d] = {
                "F": float(np.mean(f_vals)),   # within-graph mean (sources not independent -> collapse first)
                "B": float(np.mean(b_vals)),
                "yhat": yhat,
                "yprimes": yprimes,            # kept raw for the label-free var-ratio
                "n": len(f_vals),
            }
    return out


def _bootstrap_ci(values: Sequence[float], *, iters: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(iters, v.size))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_fbc(per_graph: Sequence[dict[int, dict[str, Any]]], *, model: str, split: str,
                  mode: str, whole_band: bool) -> list[dict[str, Any]]:
    """Aggregate per-graph band results -> one row per band with F, B (+95% CI over graphs), var-ratio."""
    bands = sorted({d for g in per_graph for d in g})
    rows: list[dict[str, Any]] = []
    for d in bands:
        f_g = [g[d]["F"] for g in per_graph if d in g]
        b_g = [g[d]["B"] for g in per_graph if d in g]
        yprime_all = [yp for g in per_graph if d in g for yp in g[d]["yprimes"]]
        yhat_all = [g[d]["yhat"] for g in per_graph if d in g]
        var_ratio = float(np.var(yprime_all) / (np.var(yhat_all) + 1e-12)) if len(yhat_all) > 1 else float("nan")
        b_mean, b_lo, b_hi = _bootstrap_ci(b_g)
        f_mean, _, _ = _bootstrap_ci(f_g)
        verdict = "beneficial" if b_lo > 0 else ("harmful" if b_hi < 0 else "null")
        rows.append({
            "model": model, "split": split, "resampler": mode,
            "scope": "whole_band" if whole_band else "single_source",
            "distance": int(d), "n_graphs": len(b_g),
            "F": f_mean, "B": b_mean, "B_lo": b_lo, "B_hi": b_hi,
            "var_ratio": var_ratio, "verdict": verdict,
        })
    return rows


def beneficial_carriage_rows(
    backend: FBCBackend,
    graphs: Sequence[Any],
    *,
    model: str,
    split: str,
    modes: Sequence[str] = ("matched", "marginal"),
    donors: int = 4,
    max_d: int = 6,
    whole_band: bool = False,
    donor_graphs: Sequence[Any] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Compute B(d)/F(d) summary rows for one model+split across the resampler ladder."""
    bank = build_donor_bank(backend, list(donor_graphs) if donor_graphs is not None else list(graphs))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for mode in modes:
        per_graph = [
            fbc_for_graph(backend, g, mode=mode, bank=bank, donors=donors, max_d=max_d,
                          whole_band=whole_band, rng=rng)
            for g in graphs
        ]
        rows.extend(summarize_fbc(per_graph, model=model, split=split, mode=mode, whole_band=whole_band))
    return rows
