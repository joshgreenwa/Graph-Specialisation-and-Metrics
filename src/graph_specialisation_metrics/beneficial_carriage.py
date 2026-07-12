"""Beneficial carriage B(d): does distance-d content *help the task*, not just move the output?

Pairs with functional carriage. For a focal readout node i* per graph and a source j in the band
d = {j : dist(i*, j) = d}, resample j's content on-manifold and forward-pass. From the SAME passes:

    F(d) = E[(y_hat' - y_hat)^2]                     functional (label-free: the output moved)
    B(d) = E[ |y_hat' - y| - |y_hat - y| ]           beneficial (both scored vs the CLEAN label y)

B(d) = 0 under (i) on-manifold q and (ii) y _|_ x_j | x_{-j}. A bad q fabricates benefit, so the
resampler ladder controls (i): ``marginal`` (naive, measures the artifact) and ``matched`` (donors
sharing j's *environment* signature -- the practical proxy). ``var_ratio`` = Var(y_hat')/Var(y_hat)
audits q without labels; the ``perturbation_ratio`` = F_matched/F_marginal gates against a *no-op*
resampler (matched donors so similar to the source that nothing is actually resampled -- var_ratio
alone cannot catch this, F(d)>0 is the missing check).

Resampling is delegated to the backend so it can be done cleanly per model: the GRIT backend swaps
the *symbolic* content (atom) for a matched-environment donor's and re-encodes, keeping structure
(node-RRWP) fixed. single-source B(d) is a *necessity* measure; the ``whole_band`` variant resamples
the shell jointly to capture *redundantly*-carried benefit (the gap = redundancy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Protocol, Sequence

import numpy as np
import torch

Tensor = torch.Tensor


class FBCBackend(Protocol):
    """Model/task interface. ``donor_token``/``apply_donor`` are optional -- without them the
    operator falls back to encoded-content injection (used by the toy tests)."""

    def encoded(self, graph: Any) -> Tensor: ...            # [n, dim] encoded content
    def predict(self, graph: Any, encoded: Tensor) -> float: ...  # scalar prediction from encoded content
    def label(self, graph: Any) -> float: ...               # true target
    def distances(self, graph: Any) -> Tensor: ...          # [n, n] shortest-path hop distances
    def degree(self, graph: Any) -> Tensor: ...             # [n]
    def signature(self, graph: Any, node: int) -> Hashable: ...  # ENVIRONMENT signature for donors


def _donor_token(backend: FBCBackend, graph: Any, node: int) -> Any:
    fn = getattr(backend, "donor_token", None)
    return fn(graph, node) if fn is not None else backend.encoded(graph).detach()[node].clone()


def _apply_donor(backend: FBCBackend, graph: Any, encoded: Tensor, source: int, token: Any) -> Tensor:
    fn = getattr(backend, "apply_donor", None)
    if fn is not None:
        return fn(graph, encoded, source, token)
    pert = encoded.clone()
    pert[source] = torch.as_tensor(token).to(encoded)
    return pert


def _same_token(a: Any, b: Any) -> bool:
    if torch.is_tensor(a) or torch.is_tensor(b):
        try:
            return bool(torch.allclose(torch.as_tensor(a), torch.as_tensor(b)))
        except Exception:  # noqa: BLE001
            return a is b
    return a == b


def focal_node(backend: FBCBackend, graph: Any) -> int:
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
    by_sig: dict[Hashable, list[Any]] = field(default_factory=dict)


def build_donor_bank(backend: FBCBackend, graphs: Sequence[Any]) -> DonorBank:
    bank = DonorBank()
    for g in graphs:
        n = int(backend.encoded(g).size(0))
        for node in range(n):
            bank.by_sig.setdefault(backend.signature(g, node), []).append(_donor_token(backend, g, node))
    return bank


def _draw_donor(backend: FBCBackend, graph: Any, source: int, bank: DonorBank, mode: str,
                source_token: Any, rng: np.random.Generator) -> Any:
    if mode == "marginal":
        pool = [tok for toks in bank.by_sig.values() for tok in toks]
    else:  # matched: same ENVIRONMENT, content free to vary
        pool = bank.by_sig.get(backend.signature(graph, source), [])
        pool = pool or [tok for toks in bank.by_sig.values() for tok in toks]
    for _ in range(12):  # avoid the trivial no-op draw (same content)
        cand = pool[int(rng.integers(0, len(pool)))]
        if not _same_token(cand, source_token):
            return cand
    return pool[int(rng.integers(0, len(pool)))]


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
        draws = donors if whole_band else 1
        if whole_band:
            for _ in range(donors):
                pert = enc.clone()
                for j in sources:
                    tok = _draw_donor(backend, graph, j, bank, mode, _donor_token(backend, graph, j), rng)
                    pert = _apply_donor(backend, graph, pert, j, tok)
                yp = float(backend.predict(graph, pert))
                f_vals.append((yp - yhat) ** 2)
                b_vals.append(abs(yp - y) - base_loss)
                yprimes.append(yp)
        else:
            for j in sources:
                src_tok = _donor_token(backend, graph, j)
                for _ in range(donors):
                    tok = _draw_donor(backend, graph, j, bank, mode, src_tok, rng)
                    pert = _apply_donor(backend, graph, enc, j, tok)
                    yp = float(backend.predict(graph, pert))
                    f_vals.append((yp - yhat) ** 2)
                    b_vals.append(abs(yp - y) - base_loss)
                    yprimes.append(yp)
        if f_vals:
            out[d] = {"F": float(np.mean(f_vals)), "B": float(np.mean(b_vals)),
                      "yhat": yhat, "yprimes": yprimes, "n": len(f_vals)}
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


def _apply_health_gate(rows: list[dict[str, Any]], *, min_perturbation_ratio: float = 0.05,
                       marginal_floor: float = 1e-12) -> None:
    """Flag a *no-op* matched resampler (donors ~= source), distinguishing it from a true null.

    The gate only fires where the marginal resampler proves the content actually drives the
    output (F_marginal > floor). There, F_matched should be a real fraction of F_marginal; if not,
    matched is a no-op (var_ratio ~= 1 alone cannot catch this). Where F_marginal ~= 0 the content
    simply does not affect the output -- a genuine null, resampler health is not applicable.
    """
    by_key: dict[tuple, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_key.setdefault((r["model"], r["split"], r["scope"], r["distance"]), {})[r["resampler"]] = r
    for modes in by_key.values():
        matched, marginal = modes.get("matched"), modes.get("marginal")
        if matched is None:
            continue
        ref = marginal["F"] if marginal else float("nan")
        if not (np.isfinite(ref) and ref > marginal_floor):
            matched["perturbation_ratio"] = float("nan")  # content doesn't drive output -> true null; N/A
            matched["resampler_ok"] = None
            continue
        ratio = matched["F"] / ref
        matched["perturbation_ratio"] = ratio
        ok = bool(ratio >= min_perturbation_ratio)
        matched["resampler_ok"] = ok
        if not ok:
            matched["verdict"] = "resampler_noop"


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
    _apply_health_gate(rows)
    return rows
