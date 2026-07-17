"""Figures + persisted artifacts for a carriage run, from aggregated curves + meta.

Task-agnostic: given the per-pair arrays and provenance ``meta`` from grit_runner, it
aggregates the distance curves, writes the three figures, the per-pair .npz, and the
summary .json into ``out_dir``. The d=0 self-pair term is ~100x the transport terms, so
B(d)/S(d) use symlog and the self term is marked; F(d) uses log.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import core
from .env import log

C_FUNC, C_BEN, C_ADV = "#2b6cb0", "#2f855a", "#c53030"


def _caption(meta: dict, n_g: int, K: int) -> str:
    tm = meta.get("test_metric")
    mn = meta.get("test_metric_name", "metric")
    ep = meta.get("checkpoint_epoch")
    return (f"{meta.get('title', meta.get('task', 'GRIT'))} | epoch {ep}"
            + (f" | test {mn} {tm:.4f}" if tm is not None else "")
            + f"\n{n_g} {meta.get('eval_split')} graphs, K={K} donor swaps/source, "
              f"donors from '{meta.get('donor_split')}'")


def make_figures_and_save(results: dict, out_dir: str, n_boot: int = 2000,
                          boot_seed: int = 1234, bd_linthresh: float = 0.0) -> dict:
    """Aggregate curves, render the 3 figures, and persist .npz + .json. Returns paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    gid = results["graph_id"]
    pd_ = results["distance"]
    pC, pB, F = results["C"], results["B"], results["F"]
    a_sumC, a_dyhat = results["additivity_sumC"], results["additivity_dyhat"]
    checks, meta = results["checks"], results["meta"]
    K = int(meta["donors_K"])

    agg = core.aggregate_carriage_curves(gid, pd_, F, pB, n_boot=n_boot, boot_seed=boot_seed)
    ds, counts, n_g = agg["distances"], agg["pair_counts"], agg["n_graphs"]
    F_mean, F_lo, F_hi = agg["F_mean"], agg["F_lo"], agg["F_hi"]
    B_mean, B_lo, B_hi = agg["B_mean"], agg["B_lo"], agg["B_hi"]
    S_mean, S_lo, S_hi = agg["S_mean"], agg["S_lo"], agg["S_hi"]
    ks = agg["k"]
    Bf_mean, Bf_lo, Bf_hi = agg["B_far_mean"], agg["B_far_lo"], agg["B_far_hi"]
    F_per_graph = agg["F_per_graph"]
    r = checks.get("additivity_pearson_r", float("nan"))
    slope = checks.get("additivity_slope", float("nan"))

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    tag = _caption(meta, n_g, K)
    self_col = C_ADV
    units = meta.get("loss_units", "MAE units")
    lin_B = core.symlog_linthresh(B_mean[1:] if ds.size > 1 else B_mean, bd_linthresh)

    # ---- Fig 1: F(d) and B(d) ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.fill_between(ds, F_lo, F_hi, color=C_FUNC, alpha=0.18, lw=0)
    ax.plot(ds, F_mean, "o-", color=C_FUNC, lw=1.8, ms=4.5, zorder=3)
    if ds.size > 0:
        ax.plot(ds[0], F_mean[0], "o", color=self_col, ms=8, zorder=4)
        ax.annotate("self ($i{=}j$)", (ds[0], F_mean[0]), textcoords="offset points",
                    xytext=(8, -2), fontsize=8, color=self_col, va="center")
    ax.set_xlabel("shortest-path distance $d(i,j)$ [hops]")
    ax.set_ylabel(r"$F(d)=\mathrm{mean}_{d(i,j)=d}\,\|C_{\mathrm{out}}[i,j]\|$")
    ax.set_title("Functional carriage $F(d)$\n"
                 r"(label-free: magnitude of the output movement from $j$ at $i$)")
    if np.nanmin(F_mean) > 0:
        ax.set_yscale("log")
    ax.set_xticks(ds)

    ax = axes[1]
    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.fill_between(ds, np.minimum(B_mean, 0), 0, color=C_BEN, alpha=0.30, lw=0)
    ax.fill_between(ds, np.maximum(B_mean, 0), 0, color=C_ADV, alpha=0.30, lw=0)
    ax.errorbar(ds, B_mean, yerr=[B_mean - B_lo, B_hi - B_mean], fmt="none",
                ecolor="0.45", elinewidth=0.9, capsize=2, zorder=2)
    ax.plot(ds, B_mean, "o-", color="0.15", lw=1.8, ms=4.5, zorder=3)
    if ds.size > 0:
        ax.plot(ds[0], B_mean[0], "o", color=self_col, ms=8, zorder=4)
        ax.annotate("self ($i{=}j$)", (ds[0], B_mean[0]), textcoords="offset points",
                    xytext=(8, 0), fontsize=8, color=self_col, va="center")
    ax.set_yscale("symlog", linthresh=lin_B)
    ax.set_xlabel("shortest-path distance $d(i,j)$ [hops]")
    ax.set_ylabel(rf"$B(d)=\mathrm{{mean}}_{{d(i,j)=d}}\,B[i,j]$   [{units}, symlog]")
    ax.set_title("Beneficial carriage $B(d)$\n"
                 r"$B<0$ beneficial $\cdot$ $B>0$ adverse $\cdot$ $B\approx0$ dispensable")
    ax.set_xticks(ds)
    ax.text(0.985, 0.06, "beneficial", transform=ax.transAxes, ha="right", va="bottom",
            color=C_BEN, fontsize=8.5, fontweight="bold")
    ax.text(0.985, 0.94, "adverse", transform=ax.transAxes, ha="right", va="top",
            color=C_ADV, fontsize=8.5, fontweight="bold")
    fig.suptitle("Semantic donor-swap interventions on GRIT   " + tag, fontsize=9.5, y=1.12)
    p1 = fig_dir / "fig_semantic_carriage_Fd_Bd.png"
    fig.savefig(p1); plt.close(fig)
    log(f"\n[fig] {p1}")

    # ---- Fig 2: B_far(k) --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.fill_between(ks, Bf_lo, Bf_hi, color="0.5", alpha=0.20, lw=0)
    ax.plot(ks, Bf_mean, "o-", color="0.15", lw=2.0, ms=5, zorder=3)
    ax.fill_between(ks, np.minimum(Bf_mean, 0), 0, color=C_BEN, alpha=0.30, lw=0)
    ax.fill_between(ks, np.maximum(Bf_mean, 0), 0, color=C_ADV, alpha=0.30, lw=0)
    ax.set_xlabel("hop threshold $k$")
    ax.set_ylabel(rf"$B_{{\mathrm{{far}}}}(k)=\sum_{{\{{(i,j):\,d(i,j)>k\}}}} B[i,j]$   [{units}]")
    ax.set_title(r"Beneficial carriage beyond $k$ hops"
                 "\n" r"per graph, mean over graphs (95% CI, bootstrap over graphs)")
    ax.set_xticks(ks)
    ax.text(0.985, 0.05, "net beneficial", transform=ax.transAxes, ha="right", va="bottom",
            color=C_BEN, fontsize=9, fontweight="bold")
    ax.text(0.985, 0.95, "net adverse", transform=ax.transAxes, ha="right", va="top",
            color=C_ADV, fontsize=9, fontweight="bold")
    fig.suptitle(tag, fontsize=8.5, y=1.02)
    p2 = fig_dir / "fig_semantic_carriage_Bfar.png"
    fig.savefig(p2); plt.close(fig)
    log(f"[fig] {p2}")

    # ---- Fig 3: diagnostics -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    ax = axes[0, 0]
    ax.bar(ds, counts, color=C_FUNC, alpha=0.75)
    ax.set_yscale("log")
    ax.set_xlabel("$d(i,j)$ [hops]"); ax.set_ylabel("# pairs")
    ax.set_title("Pair support per distance")
    ax.set_xticks(ds)

    ax = axes[0, 1]
    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.fill_between(ds, np.minimum(S_mean, 0), 0, color=C_BEN, alpha=0.25, lw=0)
    ax.fill_between(ds, np.maximum(S_mean, 0), 0, color=C_ADV, alpha=0.25, lw=0)
    ax.plot(ds, S_mean, "o-", color="0.15", lw=1.8, ms=4.5, zorder=3)
    if ds.size > 0:
        ax.plot(ds[0], S_mean[0], "o", color=self_col, ms=8, zorder=4)
        ax.annotate("self ($i{=}j$)", (ds[0], S_mean[0]), textcoords="offset points",
                    xytext=(8, 0), fontsize=8, color=self_col, va="center")
    ax.set_yscale("symlog", linthresh=core.symlog_linthresh(S_mean[1:] if ds.size > 1 else S_mean))
    ax.set_xlabel("$d(i,j)$ [hops]")
    ax.set_ylabel(rf"$\sum_{{d(i,j)=d}} B[i,j]$ per graph  [{units}, symlog]")
    ax.set_title("Error mass carried at each distance\n"
                 r"(tail sums give $B_{\mathrm{far}}(k)$; per source $\sum_i B[i,j]=dL_j$ exactly)")
    ax.set_xticks(ds)

    ax = axes[1, 0]
    ax.scatter(a_sumC, a_dyhat, s=6, alpha=0.28, color=C_FUNC, edgecolors="none")
    lim = max(float(np.nanmax(np.abs(np.concatenate([a_sumC, a_dyhat])))) * 1.05, 1e-12)
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.4", lw=1.0, label="$y=x$")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\sum_i C_{\mathrm{loss}}[i,j]$  (first-order $\Delta L$)")
    ax.set_ylabel(r"$dL_j = L_{\mathrm{clean}} - \mathrm{mean}_k L_{\mathrm{swap}}(j,k)$")
    ax.set_title(f"Loss additivity audit (first order)\n$r$={r:.3f}, slope={slope:.3f}")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for gi_ in range(n_g):
        ax.plot(ds, F_per_graph[gi_], "-", color=C_FUNC, alpha=0.13, lw=0.9)
    ax.plot(ds, F_mean, "o-", color="0.1", lw=2.0, ms=4.5, label="pooled $F(d)$")
    if np.nanmin(F_mean) > 0:
        ax.set_yscale("log")
    ax.set_xlabel("$d(i,j)$ [hops]"); ax.set_ylabel("$F(d)$")
    ax.set_title("Per-graph $F(d)$ (spaghetti) vs pooled")
    ax.set_xticks(ds)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Diagnostics   " + tag, fontsize=9.5, y=1.03)
    p3 = fig_dir / "fig_semantic_carriage_diagnostics.png"
    fig.savefig(p3); plt.close(fig)
    log(f"[fig] {p3}")

    # ---- persist ----------------------------------------------------------------------
    npz_path = out_dir / "carriage_pairs.npz"
    np.savez_compressed(
        npz_path,
        graph_id=gid, carrier_i=results["carrier_i"], source_j=results["source_j"],
        distance=pd_, C=pC, B=pB, F=F,
        distances=ds, F_mean=F_mean, F_lo=F_lo, F_hi=F_hi,
        B_mean=B_mean, B_lo=B_lo, B_hi=B_hi,
        B_sum_per_graph_mean=S_mean, B_sum_per_graph_lo=S_lo, B_sum_per_graph_hi=S_hi,
        pair_counts=counts, k=ks, B_far_mean=Bf_mean, B_far_lo=Bf_lo, B_far_hi=Bf_hi,
        additivity_sumC=a_sumC, additivity_dyhat=a_dyhat,
    )
    log(f"[data] {npz_path}")

    summary = {
        "meta": meta, "checks": checks,
        "curves": {
            "distance": ds.tolist(), "pair_counts": counts.tolist(),
            "F_mean": F_mean.tolist(), "F_lo": F_lo.tolist(), "F_hi": F_hi.tolist(),
            "B_mean": B_mean.tolist(), "B_lo": B_lo.tolist(), "B_hi": B_hi.tolist(),
            "B_sum_per_graph_mean": S_mean.tolist(),
            "k": ks.tolist(), "B_far_mean": Bf_mean.tolist(),
            "B_far_lo": Bf_lo.tolist(), "B_far_hi": Bf_hi.tolist(),
        },
        "figures": [str(p1), str(p2), str(p3)],
    }
    json_path = out_dir / "carriage_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[data] {json_path}")

    # ---- console table ----------------------------------------------------------------
    log("\n" + "=" * 84)
    log(f"RESULTS  (F=||dy_hat||; B<0 = beneficial, {units})")
    log("=" * 84)
    log(f"{'d':>3} {'#pairs':>8} {'F(d)':>12} {'B(d)':>13} {'sum_d B/graph':>15}")
    for d in ds:
        log(f"{d:>3} {counts[d]:>8} {F_mean[d]:>12.4e} {B_mean[d]:>13.3e} {S_mean[d]:>15.3e}")
    log("")
    log(f"{'k':>3} {'B_far(k) [MAE]':>16} {'95% CI':>28}")
    for k in ks:
        log(f"{k:>3} {Bf_mean[k]:>16.4e}   [{Bf_lo[k]:>+.3e}, {Bf_hi[k]:>+.3e}]")

    return {"figures": [str(p1), str(p2), str(p3)], "npz": str(npz_path), "json": str(json_path),
            "curves": summary["curves"]}
