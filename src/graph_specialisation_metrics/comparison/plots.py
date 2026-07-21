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


def _spearman(x, y) -> float:
    """Spearman rank correlation over finite pairs (NaN if < 3 valid or a channel is constant)."""
    x = np.asarray(x, float).reshape(-1)
    y = np.asarray(y, float).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def _norm_scores(scores_by_task: dict, task: str, gsem: float, gstr: float):
    """(S~_sem, S~_str) amplitude-normalised score matrices [L,H] for a task."""
    return scores_by_task[task]["S_sem"] / gsem, scores_by_task[task]["S_str"] / gstr


def d_rel(ssem, sstr, eps: float = 1e-9):
    """Relative selectivity in ~[-1,1]: (S~_sem - S~_str)/(S~_sem + S~_str + eps)."""
    return (ssem - sstr) / (ssem + sstr + eps)


def joint_strength(ssem, sstr):
    """Joint strength J = (S~_sem + S~_str)/2 (>=0)."""
    return 0.5 * (ssem + sstr)


def signed_contrast(a, b, eps: float = 1e-9):
    """(a-b)/(|a|+|b|+eps) in [-1,1]; for a,b>=0 this is (a-b)/(a+b+eps)."""
    return (a - b) / (np.abs(a) + np.abs(b) + eps)


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
                           suptitle: str = ("Per-head specialisation: structural (x) vs semantic (y)\n"
                                            "shared reference normalisation; absolute transport "
                                            "amplitude retained")):
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
            ax.set_ylabel("semantic  S_sem / shared mean")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("structural  S_str / shared mean")
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
                      selectivity: str = "relative", eps: float = 1e-9,
                      suptitle: Optional[str] = None):
    """(ii-b) Side-by-side per-model scatter of selectivity D (x) vs joint strength J (y).

    Using the amplitude-normalised scores S~ = raw / channel-mean (same ``gsem``/``gstr`` as the
    S_str-vs-S_sem grid), each head maps to::

        J = (S~_sem + S~_str) / 2                          joint strength (overall influence)
        D_raw = (S~_sem - S~_str) / 2                       raw selectivity
        D_rel = (S~_sem - S~_str) / (S~_sem + S~_str + eps) relative selectivity, bounded ~[-1,1]

    ``selectivity="relative"`` (default) plots D_rel, which -- unlike raw D -- does not mechanically
    grow with J, so it is comparable across heads of different strength; ``"raw"`` plots D_raw.
    Either way >0 = semantic-leaning, <0 = structural-leaning; high J = influential, low J = inert.
    The plane limits are fixed from ``ref_tasks`` (default: all cached models) so dropping or
    including methods does not rescale the plane. Returns (fig, path).
    """
    import matplotlib.pyplot as plt

    if selectivity not in ("relative", "raw"):
        raise ValueError(f"selectivity must be 'relative' or 'raw', got {selectivity!r}")
    tasks = [t for t in tasks if t in scores_by_task]
    if not tasks:
        raise ValueError("plot_spec_DJ_grid: no cached score matrices for the given tasks.")
    if gsem is None or gstr is None:
        gsem, gstr = global_norms(scores_by_task, tasks)

    def _DJ(t):
        ssem = scores_by_task[t]["S_sem"] / gsem
        sstr = scores_by_task[t]["S_str"] / gstr
        J = 0.5 * (ssem + sstr)
        D = ((ssem - sstr) / (ssem + sstr + eps) if selectivity == "relative"
             else 0.5 * (ssem - sstr))
        return D, J

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
            xlab = (r"relative selectivity  $D_{\rm rel}=\frac{\tilde S_{\rm sem}-\tilde S_{\rm str}}"
                    r"{\tilde S_{\rm sem}+\tilde S_{\rm str}}$" if selectivity == "relative"
                    else r"selectivity  $D=(\tilde S_{\rm sem}-\tilde S_{\rm str})/2$")
            ax.set_xlabel(xlab + "\n($<0$ structural  ·  $>0$ semantic)")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, pad=0.01)
        cb.set_label("layer")
    sup = suptitle or (f"Per-head specialisation (shared reference normalisation): "
                       f"{'relative ' if selectivity == 'relative' else ''}selectivity "
                       f"{'D_rel' if selectivity == 'relative' else 'D'} (x) vs joint strength J (y)")
    fig.suptitle(sup, fontsize=12)
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


# --------------------------------------------------------------------------------------
# (v) D/J causal validation against the channel-split ablation (swap x ablate)
# --------------------------------------------------------------------------------------

def _Lmax(scores_by_task, tasks):
    return max((scores_by_task[t]["S_sem"].shape[0] for t in tasks if t in scores_by_task),
              default=1)


def plot_DJ_ablation_validation(scores_by_task: dict, chan_by_task: dict, tasks: Sequence[str],
                                out_path, *, gsem: Optional[float] = None,
                                gstr: Optional[float] = None, eps: float = 1e-9,
                                metrics_by_task: Optional[dict] = None,
                                suptitle: Optional[str] = None):
    """(v, central) Does score selectivity D_rel predict the causal ablation contrast?

    Rows = {functional, loss} swap x ablate contrast; columns = models. Per head, x is the
    score-derived relative selectivity D_rel, y is the ablation contrast
    (I_sem - I_str)/(|I_sem| + |I_str|). A positive rank correlation means a head that the SCORES
    call semantic-leaning is causally more necessary for the semantic channel (and vice versa) --
    the independent causal validation of the D/J coordinates. Layer-coloured; Spearman rho per
    panel. Returns (fig, path).
    """
    import matplotlib.pyplot as plt

    tasks = [t for t in tasks if t in scores_by_task and t in chan_by_task]
    if not tasks:
        raise ValueError("plot_DJ_ablation_validation: need both scores and channel-ablation cache.")
    if gsem is None or gstr is None:
        gsem, gstr = global_norms(scores_by_task, tasks)
    Lmax = _Lmax(scores_by_task, tasks)
    rows = [("functional", "I_sem_func", "I_str_func"), ("loss", "I_sem_loss", "I_str_loss")]

    n = len(tasks)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7.0), squeeze=False, constrained_layout=True)
    sc = None
    for r, (rname, ksem, kstr) in enumerate(rows):
        for j, t in enumerate(tasks):
            ax = axes[r][j]
            ssem, sstr = _norm_scores(scores_by_task, t, gsem, gstr)
            x = d_rel(ssem, sstr, eps).reshape(-1)
            ch = chan_by_task[t]
            if ksem not in ch or kstr not in ch:
                ax.axis("off"); continue
            y = signed_contrast(ch[ksem], ch[kstr], eps).reshape(-1)
            L, H = ssem.shape
            layer = np.repeat(np.arange(L), H)
            ax.plot([-1, 1], [-1, 1], "k:", lw=0.8, zorder=0)
            ax.axhline(0, color="k", lw=0.5, alpha=0.4); ax.axvline(0, color="k", lw=0.5, alpha=0.4)
            sc = ax.scatter(x, y, c=layer, cmap="viridis", s=40, edgecolors="k",
                            linewidths=0.35, alpha=0.9, vmin=0, vmax=Lmax - 1)
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
            rho = _spearman(x, y)
            meta = _data.method_meta(t, j)
            if r == 0:
                ax.set_title(f"{meta['label']}\n{rname} contrast  ρ={rho:.2f}", fontsize=9)
            else:
                ax.set_title(f"{rname} contrast  ρ={rho:.2f}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{rname} ablation contrast\n(I_sem−I_str)/(|I_sem|+|I_str|)")
            if r == 1:
                ax.set_xlabel(r"score selectivity  $D_{\rm rel}$")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.01)
        cb.set_label("layer")
    fig.suptitle(suptitle or ("Central validation: score selectivity D_rel (x) vs causal "
                              "ablation contrast (y) — functional (top) & loss (bottom)"),
                 fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return fig, str(out_path)


def plot_DJ_quadrants(scores_by_task: dict, chan_by_task: dict, tasks: Sequence[str], out_path,
                      *, gsem: Optional[float] = None, gstr: Optional[float] = None,
                      eps: float = 1e-9, ref_tasks: Optional[Sequence[str]] = None,
                      metrics_by_task: Optional[dict] = None, suptitle: Optional[str] = None):
    """(v) Quadrant taxonomy on the D_rel-J plane, one panel per model, layer-coloured.

    A vertical line at D_rel=0 splits semantic- vs structural-leaning; a horizontal line at the
    reference median J splits influential vs inert. The four regions are influential semantic
    specialists / structural specialists (top-left/right), and inert heads (bottom). Panel titles
    carry the influence correlation rho(J, overall functional ablation impact). Returns (fig, path).
    """
    import matplotlib.pyplot as plt

    tasks = [t for t in tasks if t in scores_by_task]
    if not tasks:
        raise ValueError("plot_DJ_quadrants: no cached score matrices for the given tasks.")
    if gsem is None or gstr is None:
        gsem, gstr = global_norms(scores_by_task, tasks)
    lim_tasks = [t for t in (ref_tasks or list(scores_by_task.keys())) if t in scores_by_task]
    Lmax = _Lmax(scores_by_task, lim_tasks)
    # reference median J (inert threshold) + J axis top, fixed from the reference set
    Jall, jmax = [], 1e-9
    for t in lim_tasks:
        ss, st = _norm_scores(scores_by_task, t, gsem, gstr)
        J = joint_strength(ss, st)
        Jall.append(J.reshape(-1)); jmax = max(jmax, float(J.max()))
    j_thresh = float(np.median(np.concatenate(Jall))) if Jall else 0.0
    jmax *= 1.08

    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.1), squeeze=False, constrained_layout=True)
    sc = None
    for j, t in enumerate(tasks):
        ax = axes[0][j]
        ss, st = _norm_scores(scores_by_task, t, gsem, gstr)
        D = d_rel(ss, st, eps).reshape(-1)
        J = joint_strength(ss, st).reshape(-1)
        L, H = ss.shape
        layer = np.repeat(np.arange(L), H)
        ax.axvline(0.0, color="k", ls=":", lw=0.9)
        ax.axhline(j_thresh, color="k", ls="--", lw=0.8, alpha=0.6)
        sc = ax.scatter(D, J, c=layer, cmap="viridis", s=40, edgecolors="k", linewidths=0.35,
                        alpha=0.9, vmin=0, vmax=Lmax - 1)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(0, jmax)
        # quadrant counts
        infl = J >= j_thresh
        n_sem = int(np.sum(infl & (D > 0))); n_str = int(np.sum(infl & (D < 0)))
        n_inert = int(np.sum(~infl))
        ax.text(0.97, 0.97, f"sem {n_sem}", transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="#b30000")
        ax.text(0.03, 0.97, f"str {n_str}", transform=ax.transAxes, ha="left", va="top", fontsize=8,
                color="#00429d")
        ax.text(0.5, 0.03, f"inert {n_inert}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8, color="#555555")
        meta = _data.method_meta(t, j)
        title = meta["label"]
        ch = chan_by_task.get(t)
        if ch is not None and "overall_func" in ch:
            title += f"\ninfluence ρ(J,I)={_spearman(J, ch['overall_func'].reshape(-1)):.2f}"
        ax.set_title(title, fontsize=9)
        if j == 0:
            ax.set_ylabel(r"joint strength  $J$")
        ax.set_xlabel(r"selectivity  $D_{\rm rel}$")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.85, pad=0.01)
        cb.set_label("layer")
    fig.suptitle(suptitle or "Quadrant taxonomy: semantic/structural specialists, generalists, inert",
                 fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return fig, str(out_path)


def plot_DJ_influence_strength(scores_by_task: dict, chan_by_task: dict, tasks: Sequence[str],
                               out_path, *, gsem: Optional[float] = None,
                               gstr: Optional[float] = None, eps: float = 1e-9,
                               suptitle: Optional[str] = None):
    """(v) Two pooled panels: influence (J vs overall functional ablation impact) and strength-vs-
    specialisation (J vs |D_rel|). Heads pooled across the shown models, coloured by model, with
    Spearman rho. Answers 'does J predict causal importance?' and 'are influential heads specialists
    or generalists?'. Returns (fig, path)."""
    import matplotlib.pyplot as plt

    tasks = [t for t in tasks if t in scores_by_task]
    have_chan = [t for t in tasks if t in chan_by_task and "overall_func" in chan_by_task[t]]
    if gsem is None or gstr is None:
        gsem, gstr = global_norms(scores_by_task, tasks)

    fig, (axI, axS) = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    Jx_all, Iy_all, Jx2_all, Dy_all = [], [], [], []
    for j, t in enumerate(tasks):
        ss, st = _norm_scores(scores_by_task, t, gsem, gstr)
        J = joint_strength(ss, st).reshape(-1)
        absD = np.abs(d_rel(ss, st, eps)).reshape(-1)
        meta = _data.method_meta(t, j)
        col, mk, lab = meta["color"], meta["marker"], meta["label"]
        axS.scatter(J, absD, s=28, color=col, marker=mk, alpha=0.7, edgecolors="none", label=lab)
        Jx2_all.append(J); Dy_all.append(absD)
        if t in have_chan:
            I = chan_by_task[t]["overall_func"].reshape(-1)
            axI.scatter(J, I, s=28, color=col, marker=mk, alpha=0.7, edgecolors="none", label=lab)
            Jx_all.append(J); Iy_all.append(I)

    if Jx_all:
        rho = _spearman(np.concatenate(Jx_all), np.concatenate(Iy_all))
        axI.set_title(f"Influence: J vs overall functional ablation impact  (pooled ρ={rho:.2f})",
                      fontsize=10)
    else:
        axI.set_title("Influence: needs channel-ablation cache", fontsize=10)
    axI.set_xlabel(r"joint strength  $J$"); axI.set_ylabel("overall functional ablation impact")
    axI.grid(True, alpha=0.25)
    if Jx_all:  # only when the influence panel actually has (labelled) points
        axI.legend(fontsize=8, framealpha=0.9)

    rhoS = _spearman(np.concatenate(Jx2_all), np.concatenate(Dy_all)) if Jx2_all else float("nan")
    axS.set_title(f"Strength vs specialisation: J vs |D_rel|  (pooled ρ={rhoS:.2f})", fontsize=10)
    axS.set_xlabel(r"joint strength  $J$"); axS.set_ylabel(r"|selectivity|  $|D_{\rm rel}|$")
    axS.grid(True, alpha=0.25); axS.legend(fontsize=8, framealpha=0.9)

    fig.suptitle(suptitle or "Head influence and strength-vs-specialisation", fontsize=12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    return fig, str(out_path)


# --------------------------------------------------------------------------------------
# (vi) cached-score D x J factorial family ablation
# --------------------------------------------------------------------------------------

_FAMILY_COLOURS = {"semantic": "#0072B2", "structural": "#D55E00", "generalist": "#666666"}
_FAMILY_MARKERS = {"semantic": "o", "structural": "s", "generalist": "^"}


def _mean_ci_graph(values, *, n_boot: int = 2000, seed: int = 0):
    """Mean and paired graph-bootstrap CI for one per-graph statistic."""
    x = np.asarray(values, float).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    # Chunking avoids allocating n_boot x G for the larger five-model figures.
    draws = []
    for start in range(0, int(n_boot), 256):
        m = min(256, int(n_boot) - start)
        idx = rng.integers(0, len(x), size=(m, len(x)))
        draws.append(x[idx].mean(axis=1))
    boot = np.concatenate(draws)
    return float(x.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def plot_factorial_family_ablation_curves(family_by_task: dict, tasks: Sequence[str], out_path,
                                          *, n_boot: int = 2000,
                                          suptitle: Optional[str] = None):
    """Cumulative held-out loss impact for the six D x J families, one column per model.

    Rows are high/low J. Within each row semantic and structural specialists are compared to the
    layer/J/clean-throughput-matched generalist family. The grey band is a secondary layer-matched
    random-set reference. All panels share raw task-loss units because these are checkpoints on the
    same ZINC target. Returns ``(fig, path)``.
    """
    import matplotlib.pyplot as plt

    tasks = [t for t in tasks if t in family_by_task]
    if not tasks:
        raise ValueError("plot_factorial_family_ablation_curves: no family-ablation cache")
    fig, axes = plt.subplots(2, len(tasks), figsize=(3.45 * len(tasks), 6.6), squeeze=False,
                             constrained_layout=True, sharey=True)
    strengths = (("highJ", "high J: active families"), ("lowJ", "low J: weak/inactive control"))
    for row, (strength, row_label) in enumerate(strengths):
        for col, task in enumerate(tasks):
            ax = axes[row][col]
            item = family_by_task[task]
            names = list(item["family_names"])
            budgets = np.asarray(item["budgets"], int)
            loss = np.asarray(item["loss"], float)
            for pidx, pref in enumerate(("semantic", "structural", "generalist")):
                fi = names.index(f"{pref}_{strength}")
                means, los, his = [], [], []
                for b in range(len(budgets)):
                    m, lo, hi = _mean_ci_graph(
                        loss[fi, b], n_boot=n_boot,
                        seed=7919 + row * 1000 + col * 100 + pidx * 10 + b)
                    means.append(m); los.append(lo); his.append(hi)
                colour = _FAMILY_COLOURS[pref]
                ax.plot(budgets, means, color=colour, marker=_FAMILY_MARKERS[pref], lw=1.7,
                        ms=4.5, label=pref)
                ax.fill_between(budgets, los, his, color=colour, alpha=0.12, linewidth=0)

            rnd = np.asarray(item.get("random_loss", []), float)
            if rnd.ndim == 4 and rnd.shape[2] > 0:
                # Distribution of graph-mean impacts over random head sets: not the scientific
                # null, just a visual reference after exact layer-count matching.
                set_means = rnd[row].mean(axis=-1)  # [budget, random set]
                rlo, rhi = np.quantile(set_means, [.025, .975], axis=1)
                ax.fill_between(budgets, rlo, rhi, color="#999999", alpha=0.12,
                                label="layer-matched random")
            ax.axhline(0, color="k", lw=0.7, alpha=0.45)
            ax.grid(True, alpha=0.22)
            ax.set_xticks(budgets)
            if row == 1:
                ax.set_xlabel("cumulatively ablated heads k")
            if col == 0:
                ax.set_ylabel(f"{row_label}\nΔ validation loss")
            if row == 0:
                meta = _data.method_meta(task, col)
                K = int(np.asarray(item["heads"]).shape[1])
                clean = float(np.asarray(item["clean_loss"], float).mean())
                ax.set_title(f"{meta['label']}\nK={K}; clean={clean:.3f}", fontsize=9)
            if row == 0 and col == len(tasks) - 1:
                ax.legend(fontsize=7.5, framealpha=.9, loc="best")
    fig.suptitle(suptitle or
                 ("Matched specialisation-family ablation on held-out ZINC graphs\n"
                  "specialists vs active generalists; ribbons are paired graph-bootstrap 95% CIs"),
                 fontsize=12)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    return fig, str(out_path)


def plot_factorial_family_ablation_contrasts(family_by_task: dict, tasks: Sequence[str], out_path,
                                             *, n_boot: int = 4000,
                                             suptitle: Optional[str] = None):
    """Full-family specialist-minus-matched-generalist contrasts across models.

    Positive functional contrast means the specialist family moves the output more; positive loss
    contrast means it damages held-out performance more. CIs are paired over the same graphs.
    """
    import matplotlib.pyplot as plt

    tasks = [t for t in tasks if t in family_by_task]
    if not tasks:
        raise ValueError("plot_factorial_family_ablation_contrasts: no family-ablation cache")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    contrasts = [
        ("semantic", "highJ", "#0072B2", "o", "semantic, high J"),
        ("structural", "highJ", "#D55E00", "s", "structural, high J"),
        ("semantic", "lowJ", "#56B4E9", "o", "semantic, low J"),
        ("structural", "lowJ", "#E69F00", "s", "structural, low J"),
    ]
    x = np.arange(len(tasks), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(contrasts))
    for ax, (outcome, ylabel) in zip(
            axes, (("functional", "specialist − generalist\nfunctional output movement"),
                   ("loss", "specialist − generalist\nΔ validation loss"))):
        for c, (pref, strength, colour, marker, label) in enumerate(contrasts):
            means, los, his = [], [], []
            for ti, task in enumerate(tasks):
                item = family_by_task[task]
                names = list(item["family_names"])
                arr = np.asarray(item[outcome], float)
                target = arr[names.index(f"{pref}_{strength}"), -1]
                control = arr[names.index(f"generalist_{strength}"), -1]
                m, lo, hi = _mean_ci_graph(target - control, n_boot=n_boot,
                                            seed=1543 + c * 100 + ti)
                means.append(m); los.append(lo); his.append(hi)
            means, los, his = map(np.asarray, (means, los, his))
            ax.errorbar(x + offsets[c], means, yerr=[means - los, his - means], fmt=marker,
                        color=colour, ms=6, capsize=3, lw=1.4, label=label)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([_data.method_meta(t, i)["label"] for i, t in enumerate(tasks)],
                           rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=.25)
    axes[0].legend(fontsize=8, framealpha=.9)
    fig.suptitle(suptitle or
                 ("Does score-targeted family ablation exceed a strength-matched active null?\n"
                  "full matched families; paired graph-bootstrap 95% CIs"), fontsize=12)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    return fig, str(out_path)


# --------------------------------------------------------------------------------------
# (vii) raw semantic-score outliers: necessity + descriptive attention
# --------------------------------------------------------------------------------------

def _curve_with_graph_ci(values, *, n_boot: int, seed: int):
    values = np.asarray(values, float)
    means, lows, highs = [], [], []
    for k in range(values.shape[0]):
        m, lo, hi = _mean_ci_graph(values[k], n_boot=n_boot, seed=seed + k)
        means.append(m); lows.append(lo); highs.append(hi)
    return np.asarray(means), np.asarray(lows), np.asarray(highs)


def plot_semantic_outlier_ablation(outlier_by_task: dict, tasks: Sequence[str], out_path, *,
                                   n_boot: int = 3000,
                                   suptitle: Optional[str] = None):
    """Held-out loss test for the largest raw-S_sem heads in every model.

    The cumulative panel compares score order with an exact-layer, nearest-throughput control.
    Reverse score order is a diagnostic for the dense model's non-monotone family curve, not a
    null. The second row exposes the individual-head effects so cancellation/redundancy is visible.
    """
    import matplotlib.pyplot as plt

    tasks = [task for task in tasks if task in outlier_by_task]
    if not tasks:
        raise ValueError("plot_semantic_outlier_ablation: no semantic-outlier cache")
    fig, axes = plt.subplots(2, len(tasks), figsize=(3.55 * len(tasks), 6.8), squeeze=False,
                             sharey="row", constrained_layout=True)
    target_c, matched_c, reverse_c = "#0072B2", "#666666", "#CC79A7"
    for col, task in enumerate(tasks):
        item = outlier_by_task[task]
        budgets = np.asarray(item["budgets"], int)
        top_heads = np.asarray(item["top_heads"], int)
        matched_heads = np.asarray(item["matched_heads"], int)
        top_scores = np.asarray(item["top_scores"], float)

        ax = axes[0, col]
        for key, colour, marker, label, ls in (
                ("target_loss", target_c, "o", "top raw semantic score", "-"),
                ("matched_loss", matched_c, "s", "layer + throughput control", "-"),
                ("reverse_loss", reverse_c, "^", "same targets, reverse order", "--")):
            mean, lo, hi = _curve_with_graph_ci(
                item[key], n_boot=n_boot, seed=7100 + 100 * col + 10 * len(label))
            x = np.r_[0, budgets]; mean = np.r_[0., mean]
            lo = np.r_[0., lo]; hi = np.r_[0., hi]
            ax.plot(x, mean, color=colour, marker=marker, ms=4, lw=1.7, ls=ls, label=label)
            ax.fill_between(x, lo, hi, color=colour, alpha=.12, linewidth=0)
        rnd = np.asarray(item.get("random_loss", []), float)
        if rnd.ndim == 3 and rnd.shape[1] > 0:
            set_means = rnd.mean(axis=-1)  # [budget, random set]
            rlo, rhi = np.quantile(set_means, [.025, .975], axis=1)
            ax.fill_between(np.r_[0, budgets], np.r_[0., rlo], np.r_[0., rhi],
                            color="#AAAAAA", alpha=.14, label="exact-layer random")
        ax.axhline(0, color="k", lw=.7, alpha=.5)
        ax.set_xticks(np.r_[0, budgets])
        ax.grid(True, alpha=.22)
        if col == 0:
            ax.set_ylabel("cumulative ablation\nΔ validation loss")
        meta = _data.method_meta(task, col)
        clean = float(np.asarray(item["clean_loss"], float).mean())
        ax.set_title(f"{meta['label']}\nclean={clean:.3f}", fontsize=9.5)
        if col == len(tasks) - 1:
            ax.legend(fontsize=7.1, framealpha=.92, loc="best")

        ax = axes[1, col]
        x = np.arange(1, len(top_heads) + 1)
        for key, colour, marker, label, offset in (
                ("individual_loss", target_c, "o", "raw-semantic target", -.08),
                ("matched_individual_loss", matched_c, "s", "matched control", .08)):
            values = np.asarray(item[key], float)
            means, los, his = [], [], []
            for rank in range(len(values)):
                m, lo, hi = _mean_ci_graph(
                    values[rank], n_boot=n_boot, seed=9100 + 100 * col + 10 * rank
                    + (1 if key.startswith("matched") else 0))
                means.append(m); los.append(lo); his.append(hi)
            means, los, his = map(np.asarray, (means, los, his))
            ax.errorbar(x + offset, means, yerr=[means - los, his - means], fmt=marker,
                        color=colour, ms=5, capsize=2.5, lw=1.25, label=label)
        labels = [f"L{l}H{h}\n{score:.2g}" for (l, h), score in zip(top_heads, top_scores)]
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_xlabel("semantic-score rank\n(head; raw S_sem)")
        ax.axhline(0, color="k", lw=.7, alpha=.5)
        ax.grid(True, axis="y", alpha=.22)
        if col == 0:
            ax.set_ylabel("single-head ablation\nΔ validation loss")
        if col == len(tasks) - 1:
            ax.legend(fontsize=7.2, framealpha=.92, loc="best")

    fig.suptitle(suptitle or
                 ("Are the raw semantic-score outliers uniquely necessary?\n"
                  "score-selected on test interventions; ablated on held-out validation graphs; "
                  "J deliberately not matched"), fontsize=12)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    return fig, str(out_path)


def plot_semantic_outlier_attention(attention: dict, outlier: dict, out_path, *,
                                    suptitle: Optional[str] = None):
    """Fixed-molecule attention examples for dense raw-semantic heads and their controls."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    heads = [tuple(map(int, h)) for h in attention["heads"]]
    molecules = list(attention["molecules"])
    if not heads or not molecules:
        raise ValueError("plot_semantic_outlier_attention: empty attention cache")
    target_set = {tuple(map(int, h)) for h in np.asarray(outlier["top_heads"], int)}
    nrows, ncols = len(heads), len(molecules)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.55 * nrows),
                             squeeze=False, constrained_layout=True)
    for row, head in enumerate(heads):
        is_target = head in target_set
        row_colour = "#0072B2" if is_target else "#666666"
        for col, mol in enumerate(molecules):
            ax = axes[row, col]
            pos = np.asarray(mol["pos"], float)
            bonds = np.asarray(mol["bonds"], int)
            A = np.asarray(mol["maps"][head], float)
            for a, b in bonds:
                ax.plot(pos[[a, b], 0], pos[[a, b], 1], color="#B8B8B8", lw=2.0,
                        alpha=.65, zorder=0)
            nonself = A.copy(); np.fill_diagonal(nonself, 0.)
            flat = np.argsort(nonself.reshape(-1), kind="stable")[::-1]
            chosen = [int(i) for i in flat if nonself.reshape(-1)[i] > 0][:12]
            vmax = max([nonself.reshape(-1)[i] for i in chosen], default=1.)
            for idx in reversed(chosen):
                dest, src = np.unravel_index(idx, nonself.shape)
                w = float(nonself[dest, src]) / max(float(vmax), 1e-12)
                arrow = FancyArrowPatch(
                    pos[src], pos[dest], arrowstyle="-|>", mutation_scale=6 + 3 * w,
                    linewidth=.45 + 2.1 * w, color=row_colour, alpha=.18 + .62 * w,
                    connectionstyle="arc3,rad=0.08", shrinkA=7, shrinkB=7, zorder=1)
                ax.add_patch(arrow)
            atom = np.asarray(mol["atom_types"], int)
            self_w = np.diag(A)
            sizes = 70 + 170 * self_w / max(float(self_w.max()), 1e-12)
            ax.scatter(pos[:, 0], pos[:, 1], c=atom, cmap="tab20", s=sizes,
                       edgecolor="white", linewidth=.65, zorder=3)
            for node, (xx, yy) in enumerate(pos):
                ax.text(xx, yy, str(int(atom[node])), ha="center", va="center", fontsize=6,
                        color="black", zorder=4)
            ax.set_aspect("equal"); ax.axis("off")
            if row == 0:
                ax.set_title(f"fixed validation graph {mol['graph_id']}\nn={len(atom)}", fontsize=9)
            if col == 0:
                kind = "raw-semantic target" if is_target else "throughput control"
                ax.text(-.08, .5, f"{kind}\nL{head[0]}H{head[1]}", transform=ax.transAxes,
                        ha="right", va="center", rotation=90, color=row_colour,
                        fontsize=9, fontweight="bold")
    fig.suptitle(suptitle or
                 ("Dense GRIT: static attention of raw semantic-score outliers\n"
                  "arrows = 12 strongest non-self weights; node size = self weight; "
                  "descriptive, not causal"), fontsize=12)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    return fig, str(out_path)
