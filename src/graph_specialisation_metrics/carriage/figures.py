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
    swap_word = meta.get("swap_word", "donor swaps")   # structural sets "partner-swaps"
    return (f"{meta.get('title', meta.get('task', 'GRIT'))} | epoch {ep}"
            + (f" | test {mn} {tm:.4f}" if tm is not None else "")
            + f"\n{n_g} {meta.get('eval_split')} graphs, K={K} {swap_word}/source, "
              f"from '{meta.get('donor_split')}'")


def make_figures_and_save(results: dict, out_dir: str, n_boot: int = 2000,
                          boot_seed: int = 1234, bd_linthresh: float = 0.0,
                          bin_strategy: str = "log", central: str = "trimmed",
                          min_count: int = 50) -> dict:
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
    # Output naming: semantic keeps its historical names; structural runs prefix theirs so the
    # two coexist in one task folder without clobbering (default = semantic behaviour).
    fig_tag = meta.get("fig_tag", "semantic")
    data_prefix = "" if fig_tag == "semantic" else f"{fig_tag}_"

    agg = core.aggregate_carriage_curves(gid, pd_, F, pB, n_boot=n_boot, boot_seed=boot_seed,
                                         bin_strategy=bin_strategy, central=central,
                                         min_count=min_count)
    ds, counts, n_g = agg["bin_center"], agg["pair_counts"], agg["n_graphs"]
    labels = agg["bin_label"]
    F_mean, F_lo, F_hi = agg["F_mean"], agg["F_lo"], agg["F_hi"]
    B_mean, B_lo, B_hi = agg["B_mean"], agg["B_lo"], agg["B_hi"]
    S_mean, S_lo, S_hi = agg["S_mean"], agg["S_lo"], agg["S_hi"]
    Bf_mean, Bf_lo, Bf_hi = agg["B_far_mean"], agg["B_far_lo"], agg["B_far_hi"]
    edges = agg["k"]                              # bin upper edges (B_far thresholds)
    far_labels = [f">{int(e)}" for e in edges]
    F_per_graph = agg["F_per_graph"]
    r = checks.get("additivity_pearson_r", float("nan"))
    slope = checks.get("additivity_slope", float("nan"))

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    ctag = {"median": "median", "trimmed": "20%-trimmed mean", "mean": "mean"}.get(central, central)
    tag = (_caption(meta, n_g, K)
           + f"\nSPD bins ({bin_strategy}), {ctag} over graphs; denom={meta.get('beneficial_denom','magnitude')}")
    self_col = C_ADV
    units = meta.get("loss_units", "MAE units")
    xt = list(ds)
    lin_B = core.symlog_linthresh(B_mean[1:] if ds.size > 1 else B_mean, bd_linthresh)

    def _xaxis(ax):
        ax.set_xticks(xt)
        ax.set_xticklabels(labels, rotation=0 if len(labels) <= 10 else 45, fontsize=8)

    # ---- Fig 1: F(d) and B(d) ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.fill_between(ds, F_lo, F_hi, color=C_FUNC, alpha=0.18, lw=0)
    ax.plot(ds, F_mean, "o-", color=C_FUNC, lw=1.8, ms=4.5, zorder=3)
    if ds.size > 0:
        ax.plot(ds[0], F_mean[0], "o", color=self_col, ms=8, zorder=4)
        ax.annotate("self ($i{=}j$)", (ds[0], F_mean[0]), textcoords="offset points",
                    xytext=(8, -2), fontsize=8, color=self_col, va="center")
    ax.set_xlabel("shortest-path distance bin [hops]")
    ax.set_ylabel(r"$F$(bin) $=$ central $\|C_{\mathrm{out}}[i,j]\|$")
    ax.set_title("Functional carriage $F$\n"
                 r"(label-free: magnitude of the output movement from $j$ at $i$)")
    if np.nanmin(F_mean[np.isfinite(F_mean)]) > 0 if np.any(np.isfinite(F_mean)) else False:
        ax.set_yscale("log")
    _xaxis(ax)

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
    ax.set_xlabel("shortest-path distance bin [hops]")
    ax.set_ylabel(rf"$B$(bin) central $B[i,j]$   [{units}, symlog]")
    ax.set_title("Beneficial carriage $B$\n"
                 r"$B<0$ beneficial $\cdot$ $B>0$ adverse $\cdot$ $B\approx0$ dispensable")
    _xaxis(ax)
    ax.text(0.985, 0.06, "beneficial", transform=ax.transAxes, ha="right", va="bottom",
            color=C_BEN, fontsize=8.5, fontweight="bold")
    ax.text(0.985, 0.94, "adverse", transform=ax.transAxes, ha="right", va="top",
            color=C_ADV, fontsize=8.5, fontweight="bold")
    fig.suptitle(meta.get("fig_suptitle", "Semantic donor-swap interventions on GRIT")
                 + "   " + tag, fontsize=9.5, y=1.12)
    p1 = fig_dir / f"fig_{fig_tag}_carriage_Fd_Bd.png"
    fig.savefig(p1); plt.close(fig)
    log(f"\n[fig] {p1}")

    # ---- Fig 2: B_far(k) at bin upper edges -------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.axhline(0.0, color="0.35", lw=1.0)
    ax.fill_between(ds, Bf_lo, Bf_hi, color="0.5", alpha=0.20, lw=0)
    ax.plot(ds, Bf_mean, "o-", color="0.15", lw=2.0, ms=5, zorder=3)
    ax.fill_between(ds, np.minimum(Bf_mean, 0), 0, color=C_BEN, alpha=0.30, lw=0)
    ax.fill_between(ds, np.maximum(Bf_mean, 0), 0, color=C_ADV, alpha=0.30, lw=0)
    ax.set_xlabel("hop threshold $k$ (bin upper edge)")
    ax.set_ylabel(rf"$B_{{\mathrm{{far}}}}(k)=\sum_{{\{{(i,j):\,d(i,j)>k\}}}} B[i,j]$   [{units}]")
    ax.set_title(r"Beneficial carriage beyond $k$ hops"
                 "\n" r"per graph, mean over graphs (95% CI, bootstrap over graphs)")
    ax.set_xticks(list(ds))
    ax.set_xticklabels(far_labels, rotation=0 if len(far_labels) <= 10 else 45, fontsize=8)
    ax.text(0.985, 0.05, "net beneficial", transform=ax.transAxes, ha="right", va="bottom",
            color=C_BEN, fontsize=9, fontweight="bold")
    ax.text(0.985, 0.95, "net adverse", transform=ax.transAxes, ha="right", va="top",
            color=C_ADV, fontsize=9, fontweight="bold")
    fig.suptitle(tag, fontsize=8.5, y=1.02)
    p2 = fig_dir / f"fig_{fig_tag}_carriage_Bfar.png"
    fig.savefig(p2); plt.close(fig)
    log(f"[fig] {p2}")

    # ---- Fig 3: diagnostics -----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    ax = axes[0, 0]
    ax.bar(ds, counts, color=C_FUNC, alpha=0.75)
    ax.axhline(min_count, color=C_ADV, lw=1.0, ls="--", label=f"min_count={min_count}")
    ax.set_yscale("log")
    ax.set_xlabel("SPD bin [hops]"); ax.set_ylabel("# pairs")
    ax.set_title("Pair support per bin")
    ax.legend(frameon=False, fontsize=8)
    _xaxis(ax)

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
    ax.set_xlabel("SPD bin [hops]")
    ax.set_ylabel(rf"$\sum_{{\mathrm{{bin}}}} B[i,j]$ per graph  [{units}, symlog]")
    ax.set_title("Error mass carried per bin\n"
                 r"(mean over graphs; tail bins telescope to $B_{\mathrm{far}}$)")
    _xaxis(ax)

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
    ax.plot(ds, F_mean, "o-", color="0.1", lw=2.0, ms=4.5, label=f"{ctag} over graphs")
    if np.any(np.isfinite(F_mean)) and np.nanmin(F_mean[np.isfinite(F_mean)]) > 0:
        ax.set_yscale("log")
    ax.set_xlabel("SPD bin [hops]"); ax.set_ylabel("$F$(bin)")
    ax.set_title("Per-graph $F$ (spaghetti) vs central")
    _xaxis(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Diagnostics   " + tag, fontsize=9.5, y=1.03)
    p3 = fig_dir / f"fig_{fig_tag}_carriage_diagnostics.png"
    fig.savefig(p3); plt.close(fig)
    log(f"[fig] {p3}")

    # ---- persist ----------------------------------------------------------------------
    npz_path = out_dir / f"{data_prefix}carriage_pairs.npz"
    np.savez_compressed(
        npz_path,
        graph_id=gid, carrier_i=results["carrier_i"], source_j=results["source_j"],
        distance=pd_, C=pC, B=pB, F=F,
        bin_lo=agg["bin_lo"], bin_hi=agg["bin_hi"], bin_center=ds,
        F_mean=F_mean, F_lo=F_lo, F_hi=F_hi,
        B_mean=B_mean, B_lo=B_lo, B_hi=B_hi,
        B_sum_per_graph_mean=S_mean, B_sum_per_graph_lo=S_lo, B_sum_per_graph_hi=S_hi,
        pair_counts=counts, k=edges, B_far_mean=Bf_mean, B_far_lo=Bf_lo, B_far_hi=Bf_hi,
        additivity_sumC=a_sumC, additivity_dyhat=a_dyhat,
    )
    log(f"[data] {npz_path}")

    summary = {
        "meta": meta, "checks": checks,
        "curves": {
            "bin_lo": agg["bin_lo"].tolist(), "bin_hi": agg["bin_hi"].tolist(),
            "bin_label": labels, "bin_strategy": bin_strategy, "central": central,
            "min_count": int(min_count), "pair_counts": counts.tolist(),
            "F_mean": F_mean.tolist(), "F_lo": F_lo.tolist(), "F_hi": F_hi.tolist(),
            "B_mean": B_mean.tolist(), "B_lo": B_lo.tolist(), "B_hi": B_hi.tolist(),
            "B_sum_per_graph_mean": S_mean.tolist(),
            "k": edges.tolist(), "B_far_mean": Bf_mean.tolist(),
            "B_far_lo": Bf_lo.tolist(), "B_far_hi": Bf_hi.tolist(),
        },
        "figures": [str(p1), str(p2), str(p3)],
    }
    json_path = out_dir / f"{data_prefix}carriage_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[data] {json_path}")

    # ---- console table ----------------------------------------------------------------
    log("\n" + "=" * 88)
    log(f"RESULTS  (bins={bin_strategy}, {ctag} over graphs; F=||dy_hat||; B<0 = beneficial, {units})")
    log("=" * 88)
    log(f"{'bin':>7} {'#pairs':>9} {'F':>12} {'B':>13} {'sumB/graph':>13} {'B_far(>hi)':>13}")
    for b in range(len(labels)):
        drop = "" if counts[b] >= min_count else " (<min)"
        log(f"{labels[b]:>7} {counts[b]:>9} {F_mean[b]:>12.4e} {B_mean[b]:>13.3e} "
            f"{S_mean[b]:>13.3e} {Bf_mean[b]:>13.3e}{drop}")

    return {"figures": [str(p1), str(p2), str(p3)], "npz": str(npz_path), "json": str(json_path),
            "curves": summary["curves"]}
