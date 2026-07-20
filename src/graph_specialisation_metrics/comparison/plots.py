"""Cross-model comparison figures (deliverables ii-iv), rebuilt purely from cached artefacts.

Pure numpy/matplotlib -- no torch, no GRIT -- so figures can be re-styled or re-subsetted
(drop/include methods) without any re-run. Every function reads the cached carriage summaries
(``curves`` blocks) or specialisation score matrices loaded via ``comparison.data``.

Deliverables:
* ``plot_spec_scatter_grid``      -- (ii)  side-by-side per-model scatter of the specialisation
                                     scores: structural (x) vs semantic (y), one panel per model,
                                     axes divided by the global per-channel mean, layer-coloured.
* ``plot_spec_DJ_grid``           -- (ii-b) the same scores rotated to selectivity D (x) vs joint
                                     strength J (y): D = (S~sem - S~str)/2, J = (S~sem + S~str)/2,
                                     separating head strength (J) from channel preference (D).
* ``plot_carriage_small_multiples`` -- (iii) side-by-side functional (top row) and beneficial
                                     (bottom row) carriage, one column per model, with a
                                     STANDARDISED (reference-fixed) y-axis per row so the scale
                                     does not move when methods are dropped/included.
* ``plot_carriage_overlay``       -- (iv)  two panels (functional | beneficial) overlaying every
                                     method's curve on shared axes for direct comparison.

The y-axis "standardisation" is a reference-fixed limit: pass ``ref_curves_by_task`` (all cached
models) and the limits are computed from it, so showing a subset never rescales the axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import data as _data


# --------------------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------------------

def _finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return a[np.isfinite(a)]


def _pos_finite(a: np.ndarray) -> np.ndarray:
    a = _finite(a)
    return a[a > 0]


def _channel(curve: dict, which: str):
    """(mean, lo, hi) arrays for which in {'F','B'} from a curves dict."""
    return (np.asarray(curve[f"{which}_mean"], float),
            np.asarray(curve[f"{which}_lo"], float),
            np.asarray(curve[f"{which}_hi"], float))


def _pool_channel(curves_by_task: dict, tasks: Sequence[str], which: str,
                  master: Sequence[str], drop_labels: Sequence[str] = ()) -> np.ndarray:
    """All methods' (mean, lo, hi) values for a channel pooled onto the master axis."""
    drop = set(drop_labels)
    keep = [i for i, lab in enumerate(master) if lab not in drop]
    vals = []
    for t in tasks:
        c = curves_by_task.get(t)
        if not c:
            continue
        cur = c["curves"]
        for key in (f"{which}_mean", f"{which}_lo", f"{which}_hi"):
            arr = _data.curve_on_master(cur, key, master)
            vals.append(arr[keep])
    return np.concatenate(vals) if vals else np.array([])


def _b_linthresh(values: np.ndarray, floor: float = 1e-9) -> float:
    a = np.abs(_finite(values))
    a = a[a > 0]
    return float(max(floor, np.median(a))) if a.size else floor


def _symmetric_ylim(values: np.ndarray, pad: float = 1.25, floor: float = 1e-9):
    a = _finite(values)
    if a.size == 0:
        return (-floor, floor)
    m = max(floor, float(np.max(np.abs(a))) * pad)
    return (-m, m)


def _log_ylim(values: np.ndarray, lo_pad: float = 0.6, hi_pad: float = 1.6):
    a = _pos_finite(values)
    if a.size == 0:
        return None
    return (float(a.min()) * lo_pad, float(a.max()) * hi_pad)


def _metric_label(metrics_by_task: Optional[dict], task: str) -> str:
    if not metrics_by_task or task not in metrics_by_task:
        return ""
    m = metrics_by_task[task]
    name = m.get("name", "MAE")
    bits = []
    if m.get("val") is not None:
        bits.append(f"val {name} {float(m['val']):.3f}")
    if m.get("test") is not None:
        bits.append(f"test {name} {float(m['test']):.3f}")
    return "\n".join(bits)


# --------------------------------------------------------------------------------------
# (ii) specialisation-score scatter grid
# --------------------------------------------------------------------------------------

def global_norms(scores_by_task: dict, tasks: Sequence[str]):
    """(gsem, gstr): mean over the given models of each channel's per-model mean (+eps)."""
    sem = [scores_by_task[t]["S_sem"].mean() for t in tasks if t in scores_by_task]
    strv = [scores_by_task[t]["S_str"].mean() for t in tasks if t in scores_by_task]
    gsem = float(np.mean(sem)) + 1e-12 if sem else 1e-12
    gstr = float(np.mean(strv)) + 1e-12 if strv else 1e-12
    return gsem, gstr


def plot_spec_scatter_grid(scores_by_task: dict, tasks: Sequence[str], out_path,
                           *, gsem: Optional[float] = None, gstr: Optional[float] = None,
                           ncols: Optional[int] = None, metrics_by_task: Optional[dict] = None,
                           suptitle: str = "Per-head specialisation: structural (x) vs semantic (y)"):
    """(ii) Side-by-side per-model scatter of S_str (x) vs S_sem (y), layer-coloured.

    ``gsem``/``gstr`` are the per-channel global means the axes are divided by; pass the reference
    (all-model) values so panels stay comparable across drop/include selections. Returns
    (fig, path).
    """
    import matplotlib.pyplot as plt

    tasks = [t for t in tasks if t in scores_by_task]
    if not tasks:
        raise ValueError("plot_spec_scatter_grid: no cached score matrices for the given tasks.")
    if gsem is None or gstr is None:
        gsem, gstr = global_norms(scores_by_task, tasks)

    # shared square limit + layer range across shown panels
    mx = 1e-9
    Lmax = 1
    for t in tasks:
        S_sem, S_str = scores_by_task[t]["S_sem"], scores_by_task[t]["S_str"]
        L = S_sem.shape[0]
        Lmax = max(Lmax, L)
        mx = max(mx, float((S_str / gstr).max()), float((S_sem / gsem).max()))
    mx *= 1.08

    n = len(tasks)
    ncols = ncols or n
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.7 * nrows),
                             squeeze=False, constrained_layout=True)
    sc = None
    for idx, t in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        S_sem, S_str = scores_by_task[t]["S_sem"], scores_by_task[t]["S_str"]
        L, H = S_sem.shape
        x = (S_str / gstr).reshape(-1)
        y = (S_sem / gsem).reshape(-1)
        layer = np.repeat(np.arange(L), H)
        ax.plot([0, mx], [0, mx], "k:", lw=0.8, zorder=0)
        sc = ax.scatter(x, y, c=layer, cmap="viridis", s=48, edgecolors="k", linewidths=0.4,
                        alpha=0.9, vmin=0, vmax=Lmax - 1)
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)
        ax.set_aspect("equal", adjustable="box")
        meta = _data.method_meta(t, idx)
        ml = _metric_label(metrics_by_task, t)
        ax.set_title(meta["label"] + (f"\n{ml}" if ml else ""), fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel("semantic  S_sem / mean")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("structural  S_str / mean")
    # blank any unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
        cb.set_label("layer")
    fig.suptitle(suptitle, fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return fig, str(out_path)


# --------------------------------------------------------------------------------------
# (ii-b) selectivity D vs joint-strength J scatter grid (rotation of the S_str/S_sem plane)
# --------------------------------------------------------------------------------------

def plot_spec_DJ_grid(scores_by_task: dict, tasks: Sequence[str], out_path,
                      *, gsem: Optional[float] = None, gstr: Optional[float] = None,
                      ncols: Optional[int] = None, metrics_by_task: Optional[dict] = None,
                      ref_tasks: Optional[Sequence[str]] = None,
                      suptitle: str = ("Per-head specialisation: selectivity D (x) vs "
                                       "joint strength J (y)")):
    """(ii-b) Side-by-side per-model scatter of D (x) vs J (y), layer-coloured.

    Using the amplitude-normalised scores S~ = raw / channel-mean (same ``gsem``/``gstr`` as the
    S_str-vs-S_sem grid), each head maps to::

        J = (S~_sem + S~_str) / 2     joint strength (how influential the head is overall)
        D = (S~_sem - S~_str) / 2     selectivity  (>0 semantic-leaning, <0 structural-leaning)

    This is a 45 degrees rotation of the (S~_str, S~_sem) plane that separates *strength* from
    *preference*: high J = influential, sign of D = which channel it prefers, low J = inert.
    The D-J plane limits are fixed from ``ref_tasks`` (default: all cached models) so dropping or
    including methods does not rescale the plane. Returns (fig, path).
    """
    import matplotlib.pyplot as plt

    tasks = [t for t in tasks if t in scores_by_task]
    if not tasks:
        raise ValueError("plot_spec_DJ_grid: no cached score matrices for the given tasks.")
    if gsem is None or gstr is None:
        gsem, gstr = global_norms(scores_by_task, tasks)

    def _DJ(t):
        ssem = scores_by_task[t]["S_sem"] / gsem
        sstr = scores_by_task[t]["S_str"] / gstr
        return 0.5 * (ssem - sstr), 0.5 * (ssem + sstr)   # (D, J)

    # Fixed plane from the reference set so the axes are stable across drop/include selections.
    lim_tasks = [t for t in (ref_tasks or list(scores_by_task.keys())) if t in scores_by_task]
    dmax, jmax, Lmax = 1e-9, 1e-9, 1
    for t in lim_tasks:
        D, J = _DJ(t)
        dmax = max(dmax, float(np.abs(D).max()))
        jmax = max(jmax, float(J.max()))
        Lmax = max(Lmax, scores_by_task[t]["S_sem"].shape[0])
    dmax *= 1.08
    jmax *= 1.08

    n = len(tasks)
    ncols = ncols or n
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.7 * nrows),
                             squeeze=False, constrained_layout=True)
    sc = None
    for idx, t in enumerate(tasks):
        ax = axes[idx // ncols][idx % ncols]
        D, J = _DJ(t)
        L, H = scores_by_task[t]["S_sem"].shape
        layer = np.repeat(np.arange(L), H)
        ax.axvline(0.0, color="k", ls=":", lw=0.9, zorder=0)   # no-preference axis (D=0)
        sc = ax.scatter(D.reshape(-1), J.reshape(-1), c=layer, cmap="viridis", s=48,
                        edgecolors="k", linewidths=0.4, alpha=0.9, vmin=0, vmax=Lmax - 1)
        ax.set_xlim(-dmax, dmax)
        ax.set_ylim(0, jmax)
        meta = _data.method_meta(t, idx)
        ml = _metric_label(metrics_by_task, t)
        ax.set_title(meta["label"] + (f"\n{ml}" if ml else ""), fontsize=9)
        if idx % ncols == 0:
            ax.set_ylabel(r"joint strength  $J=(\tilde S_{\rm sem}+\tilde S_{\rm str})/2$")
        if idx // ncols == nrows - 1:
            ax.set_xlabel(r"selectivity  $D=(\tilde S_{\rm sem}-\tilde S_{\rm str})/2$"
                          "\n($<0$ structural  ·  $>0$ semantic)")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
        cb.set_label("layer")
    fig.suptitle(suptitle, fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return fig, str(out_path)


# --------------------------------------------------------------------------------------
# (iii) functional / beneficial small multiples with a standardised y-axis
# --------------------------------------------------------------------------------------

def plot_carriage_small_multiples(curves_by_task: dict, tasks: Sequence[str], out_path,
                                  *, intervention: str = "semantic",
                                  ref_curves_by_task: Optional[dict] = None,
                                  include_self_B: bool = False,
                                  metrics_by_task: Optional[dict] = None,
                                  suptitle: Optional[str] = None):
    """(iii) Rows = {functional, beneficial}; columns = models; y-axis standardised per row.

    The per-row y-limits (F: log, B: symlog) are computed from ``ref_curves_by_task`` (default:
    all cached models) so they DO NOT move when the shown ``tasks`` change -- the requested
    stable-scale behaviour. The beneficial row drops the self-pair (d=0) bin by default because
    it is ~100x the transport terms and would dominate the shared scale. Returns (fig, path).
    """
    import matplotlib.pyplot as plt

    tasks = [t for t in tasks if t in curves_by_task]
    if not tasks:
        raise ValueError("plot_carriage_small_multiples: no cached carriage curves for tasks.")
    ref = ref_curves_by_task or curves_by_task
    ref_tasks = list(ref.keys())

    # Both rows share ONE bin axis (`master`) so the functional (top) and beneficial (bottom)
    # panels of a column stay column-aligned under sharex. The self-pair (d=0) is NaN-ed out of
    # the beneficial row (not dropped from the axis) so it is omitted visually while the x-axis
    # stays identical to the functional row.
    master = _data.master_bins(ref, ref_tasks)
    drop_B = () if include_self_B else ("0",)
    drop_B_set = set(drop_B)

    def _mask_self(arr):
        return np.array([np.nan if lab in drop_B_set else v
                         for lab, v in zip(master, arr)], dtype=float)

    # standardised per-row scales from the reference set
    F_pool = _pool_channel(ref, ref_tasks, "F", master)
    B_pool = _pool_channel(ref, ref_tasks, "B", master, drop_labels=drop_B)
    f_ylim = _log_ylim(F_pool)
    b_ylim = _symmetric_ylim(B_pool)
    b_lin = _b_linthresh(B_pool)

    x = np.arange(len(master))

    n = len(tasks)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 6.2), squeeze=False,
                             constrained_layout=True, sharex="col")
    for j, t in enumerate(tasks):
        cur = curves_by_task[t]["curves"]
        meta = _data.method_meta(t, j)
        col = meta["color"]
        # --- functional (top) ---
        axF = axes[0][j]
        Fm = _data.curve_on_master(cur, "F_mean", master)
        Flo = _data.curve_on_master(cur, "F_lo", master)
        Fhi = _data.curve_on_master(cur, "F_hi", master)
        axF.fill_between(x, Flo, Fhi, color=col, alpha=0.18, linewidth=0)
        axF.plot(x, Fm, "-o", color=col, ms=4, lw=1.6)
        if f_ylim:
            axF.set_yscale("log")
            axF.set_ylim(*f_ylim)
        ml = _metric_label(metrics_by_task, t)
        axF.set_title(meta["label"] + (f"\n{ml}" if ml else ""), fontsize=9)
        axF.grid(True, which="both", alpha=0.25)
        if j == 0:
            axF.set_ylabel("functional carriage  F(d)")
        # --- beneficial (bottom); self-pair NaN-ed so the axis matches the functional row ---
        axB = axes[1][j]
        Bm = _mask_self(_data.curve_on_master(cur, "B_mean", master))
        Blo = _mask_self(_data.curve_on_master(cur, "B_lo", master))
        Bhi = _mask_self(_data.curve_on_master(cur, "B_hi", master))
        axB.axhline(0.0, color="k", lw=0.6, alpha=0.5)
        axB.fill_between(x, Blo, Bhi, color=col, alpha=0.18, linewidth=0)
        axB.plot(x, Bm, "-o", color=col, ms=4, lw=1.6)
        axB.set_yscale("symlog", linthresh=b_lin)
        axB.set_ylim(*b_ylim)
        axB.grid(True, which="both", alpha=0.25)
        axB.set_xticks(x)
        axB.set_xticklabels(master, rotation=0, fontsize=8)
        axB.set_xlabel("shortest-path distance d")
        if j == 0:
            axB.set_ylabel("beneficial carriage  B(d)\n(<0 helps, >0 hurts)")
    sup = suptitle or (f"Functional & beneficial {intervention} carriage "
                       f"(standardised y-axis; B self-pair omitted"
                       f"{' ' if include_self_B else ''})")
    fig.suptitle(sup, fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return fig, str(out_path)


# --------------------------------------------------------------------------------------
# (iv) functional / beneficial overlay across methods
# --------------------------------------------------------------------------------------

def plot_carriage_overlay(curves_by_task: dict, tasks: Sequence[str], out_path,
                          *, intervention: str = "semantic",
                          ref_curves_by_task: Optional[dict] = None,
                          include_self_B: bool = False, standardise: bool = True,
                          show_ci: bool = True, suptitle: Optional[str] = None):
    """(iv) Two panels -- functional (left) and beneficial (right) -- overlaying every method.

    With ``standardise=True`` the y-limits are fixed from ``ref_curves_by_task`` (all cached
    models) so the axis does not rescale when methods are dropped. Returns (fig, path).
    """
    import matplotlib.pyplot as plt

    tasks = [t for t in tasks if t in curves_by_task]
    if not tasks:
        raise ValueError("plot_carriage_overlay: no cached carriage curves for tasks.")
    ref = ref_curves_by_task or curves_by_task
    ref_tasks = list(ref.keys())

    master = _data.master_bins(ref, ref_tasks)
    drop_B = () if include_self_B else ("0",)
    master_B = [lab for lab in master if lab not in set(drop_B)]
    x = np.arange(len(master))
    xB = np.arange(len(master_B))

    F_pool = _pool_channel(ref, ref_tasks, "F", master)
    B_pool = _pool_channel(ref, ref_tasks, "B", master, drop_labels=drop_B)
    f_ylim = _log_ylim(F_pool)
    b_ylim = _symmetric_ylim(B_pool)
    b_lin = _b_linthresh(B_pool)

    fig, (axF, axB) = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    for j, t in enumerate(tasks):
        cur = curves_by_task[t]["curves"]
        meta = _data.method_meta(t, j)
        col, mk, lab = meta["color"], meta["marker"], meta["label"]
        Fm = _data.curve_on_master(cur, "F_mean", master)
        axF.plot(x, Fm, "-", marker=mk, color=col, ms=5, lw=1.7, label=lab)
        if show_ci:
            Flo = _data.curve_on_master(cur, "F_lo", master)
            Fhi = _data.curve_on_master(cur, "F_hi", master)
            axF.fill_between(x, Flo, Fhi, color=col, alpha=0.10, linewidth=0)
        Bm = _data.curve_on_master(cur, "B_mean", master_B)
        axB.plot(xB, Bm, "-", marker=mk, color=col, ms=5, lw=1.7, label=lab)
        if show_ci:
            Blo = _data.curve_on_master(cur, "B_lo", master_B)
            Bhi = _data.curve_on_master(cur, "B_hi", master_B)
            axB.fill_between(xB, Blo, Bhi, color=col, alpha=0.10, linewidth=0)

    if f_ylim:
        axF.set_yscale("log")
        if standardise:
            axF.set_ylim(*f_ylim)
    axF.set_xticks(x)
    axF.set_xticklabels(master, fontsize=9)
    axF.set_xlabel("shortest-path distance d")
    axF.set_ylabel("functional carriage  F(d)")
    axF.set_title("Functional carriage")
    axF.grid(True, which="both", alpha=0.25)
    axF.legend(fontsize=8, framealpha=0.9)

    axB.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    axB.set_yscale("symlog", linthresh=b_lin)
    if standardise:
        axB.set_ylim(*b_ylim)
    axB.set_xticks(xB)
    axB.set_xticklabels(master_B, fontsize=9)
    axB.set_xlabel("shortest-path distance d")
    axB.set_ylabel("beneficial carriage  B(d)   (<0 helps, >0 hurts)")
    axB.set_title("Beneficial carriage" + ("" if include_self_B else "  (self-pair d=0 omitted)"))
    axB.grid(True, which="both", alpha=0.25)
    axB.legend(fontsize=8, framealpha=0.9)

    sup = suptitle or f"{intervention.capitalize()} carriage across methods"
    fig.suptitle(sup, fontsize=13)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return fig, str(out_path)


# --------------------------------------------------------------------------------------
# (1) val/test performance bar (a convenience for the evaluation deliverable)
# --------------------------------------------------------------------------------------

def plot_performance(metrics_by_task: dict, tasks: Sequence[str], out_path,
                     *, metric_name: str = "MAE",
                     suptitle: str = "ZINC val / test performance (recomputed from checkpoint)"):
    """Grouped val/test bar chart per model. Returns (fig, path)."""
    import matplotlib.pyplot as plt

    tasks = [t for t in tasks if t in metrics_by_task]
    if not tasks:
        raise ValueError("plot_performance: no metrics for the given tasks.")
    labels = [_data.method_meta(t, i)["label"] for i, t in enumerate(tasks)]

    def _f(x):  # None (metric not computed / stale cache) -> NaN so matplotlib skips the bar
        return float(x) if x is not None else np.nan
    val = [_f(metrics_by_task[t].get("val")) for t in tasks]
    test = [_f(metrics_by_task[t].get("test")) for t in tasks]
    xpos = np.arange(len(tasks))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.7 * len(tasks) + 2, 4.6), constrained_layout=True)
    b1 = ax.bar(xpos - w / 2, val, w, label="val", color="#1f77b4")
    b2 = ax.bar(xpos + w / 2, test, w, label="test", color="#d62728")
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.3f}", (r.get_x() + r.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=8, xytext=(0, 1),
                            textcoords="offset points")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(suptitle, fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return fig, str(out_path)
