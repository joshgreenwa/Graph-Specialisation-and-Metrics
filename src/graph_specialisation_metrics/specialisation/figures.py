"""Figures for the per-head GRIT specialisation analysis (the four deliverables + extras).

Given the score / ablation / attention results for one or more models (dense GRIT and its 1-hop
control), writes:

  (i)   fig_scatter_<task>.png     structural (x) vs semantic (y) per head, coloured by layer.
  (ii)  fig_heatmaps_<task>.png    per-head [L x H] heatmaps of S_sem, S_str, S_attn + layer profile.
  (iii) fig_attention_<task>.png   attention maps of the interesting heads across molecules,
        fig_attention_graph_<task>.png  the same heads as attention-weighted molecular graphs.
  (iv)  fig_ablation_<task>.png    per-head impact vs random-head null, per-graph impact
        distributions for the interesting heads, and impact-vs-graph-feature correlations.
  extra fig_scatter_combined.png   dense vs 1-hop on one amplitude-normalised scatter.

Method-A normalisation (SPECIALISATION_SCORES.md): each axis is divided by that channel's GLOBAL
mean across ALL scored models, so the dotted diagonal is amplitude-normalised-equal (content swaps
move ~unit, RRWP transpositions ~0.1). Above diagonal = semantic-leaning, below = structural.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402


_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]


def _task_style(task: str, i: int, result: dict | None = None) -> dict:
    """Task-agnostic marker/colour: '^' for a sparse/1-hop variant, 'o' otherwise; colour by index.

    Works for any registered task (zinc, zinc_1hop, peptides_func, peptides_struct, future k-hop
    variants); the label is the task's human title when available, else the task name.
    """
    sparse = any(k in task.lower() for k in ("1hop", "1-hop", "khop", "k-hop", "sparse", "masked"))
    label = (result or {}).get("title", task)
    return dict(marker="^" if sparse else "o", color=_PALETTE[i % len(_PALETTE)], label=label)


def _rc():
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.axisbelow": True, "figure.dpi": 140})


def _spearman_np(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])); ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def _global_norms(results: dict):
    gsem = np.mean([r["S_sem"].mean() for r in results.values()]) + 1e-12
    gstr = np.mean([r["S_str"].mean() for r in results.values()]) + 1e-12
    return gsem, gstr


# --------------------------------------------------------------------------------------- #
# (i) scatter, per model, coloured by layer index.
# --------------------------------------------------------------------------------------- #
def fig_scatter(result, gsem, gstr, out) -> str:
    _rc()
    L, H = result["L"], result["H"]
    x = (result["S_str"] / gstr).reshape(-1)
    y = (result["S_sem"] / gsem).reshape(-1)
    layer = np.repeat(np.arange(L), H)
    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    mx = 1.08 * max(x.max(), y.max(), 1e-9)
    ax.plot([0, mx], [0, mx], "k:", lw=0.8, zorder=0)
    sc = ax.scatter(x, y, c=layer, cmap="viridis", s=70, edgecolors="k", linewidths=0.5,
                    alpha=0.9, vmin=0, vmax=L - 1)
    cb = fig.colorbar(sc, ax=ax, label="layer index (0 = input)")
    cb.set_ticks(range(0, L, max(1, L // 8)))
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("structural score  (S_str / global mean)")
    ax.set_ylabel("semantic score  (S_sem / global mean)")
    ax.set_title(f"{result['title']}\nper-head specialisation  ·  each point = one attention head",
                 fontsize=11, fontweight="bold")
    ax.annotate("semantic-leaning", xy=(0.05 * mx, 0.92 * mx), color="#1f77b4", fontsize=9)
    ax.annotate("structural-leaning", xy=(0.55 * mx, 0.06 * mx), color="#d62728", fontsize=9)
    tm = result.get("test_metric")
    if tm is not None:
        ax.text(0.98, 0.02, f"test {result['test_metric_name']}={tm:.4f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#444")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def fig_scatter_combined(results: dict, gsem, gstr, out) -> str:
    _rc()
    fig, ax = plt.subplots(figsize=(6.4, 5.8), constrained_layout=True)
    mx = 0.0
    styles = {t: _task_style(t, i, r) for i, (t, r) in enumerate(results.items())}
    for task, r in results.items():
        x = (r["S_str"] / gstr).reshape(-1); y = (r["S_sem"] / gsem).reshape(-1)
        st = styles[task]
        ax.scatter(x, y, s=55, edgecolors="k", linewidths=0.4, alpha=0.8,
                   marker=st["marker"], color=st["color"])
        mx = max(mx, x.max(), y.max())
    mx = (mx or 1e-9) * 1.08
    ax.plot([0, mx], [0, mx], "k:", lw=0.8)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("structural score  (/ global mean)")
    ax.set_ylabel("semantic score  (/ global mean)")
    ax.set_title("Per-head specialisation across models  (amplitude-normalised)",
                 fontsize=11, fontweight="bold")
    handles = [Line2D([0], [0], linestyle="", markeredgecolor="k", markersize=8,
                      marker=styles[t]["marker"], color=styles[t]["color"], label=styles[t]["label"])
               for t in results]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------------------- #
# (ii) per-head heatmaps + layer profile.
# --------------------------------------------------------------------------------------- #
def fig_heatmaps(result, out) -> str:
    _rc()
    L, H = result["L"], result["H"]
    panels = [("S_sem", result["S_sem"], "semantic score"),
              ("S_str", result["S_str"], "structural score")]
    if result.get("S_attn_sem") is not None:
        panels.append(("S_attn_sem", result["S_attn_sem"], "attention-routing (semantic)"))
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.4 * (len(panels) + 1), 4.6),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, (key, M, name) in zip(axes, panels):
        im = ax.imshow(M, aspect="auto", cmap="magma", origin="lower")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("head"); ax.set_ylabel("layer")
        ax.set_xticks(range(H)); ax.set_yticks(range(0, L, max(1, L // 10)))
        ax.grid(False)
    # layer profile: mean over heads per layer.
    axp = axes[-1]
    axp.plot(result["S_sem"].mean(1), range(L), "-o", color="#1f77b4", ms=3, label="semantic")
    axp.plot(result["S_str"].mean(1), range(L), "-^", color="#d62728", ms=3, label="structural")
    axp.set_xlabel("mean score over heads"); axp.set_ylabel("layer")
    axp.set_title("layer profile", fontsize=10, fontweight="bold")
    axp.legend(frameon=False, fontsize=8); axp.grid(True, alpha=0.25)
    fig.suptitle(f"{result['title']}  ·  per-head scores  (raw magnitudes)", fontsize=12, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------------------- #
# (iii) attention across molecules.
# --------------------------------------------------------------------------------------- #
def fig_attention(attn_data, heads_named, out) -> str:
    _rc()
    mols = attn_data["molecules"]
    named = list(heads_named.items())                 # [(name, (l,h)), ...]
    nrow, ncol = len(named), len(mols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.7 * nrow),
                             constrained_layout=True, squeeze=False)
    for ri, (name, hd) in enumerate(named):
        hd = tuple(int(x) for x in hd)
        for ci, mol in enumerate(mols):
            ax = axes[ri][ci]
            A = mol["maps"][hd]
            im = ax.imshow(A, cmap="magma", vmin=0, vmax=max(A.max(), 1e-6), aspect="equal")
            ax.grid(False)
            if ri == 0:
                ax.set_title(f"mol {mol['graph_id']} (n={mol['n']}, y={mol['y']:.2f})", fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{name}\nL{hd[0]}H{hd[1]}\n(receiver i)", fontsize=8)
            ax.set_xlabel("sender j", fontsize=7)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"{attn_data.get('title','')}  ·  attention A[i,j] = a_{{i<-j}} of interesting heads "
                 f"across molecules", fontsize=11, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def fig_attention_graph(attn_data, heads_named, out, max_mols=3) -> str:
    """Attention as an edge-weighted molecular graph for the interesting heads (a few molecules)."""
    _rc()
    mols = attn_data["molecules"][:max_mols]
    named = list(heads_named.items())
    nrow, ncol = len(named), len(mols)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.0 * nrow),
                             constrained_layout=True, squeeze=False)
    for ri, (name, hd) in enumerate(named):
        hd = tuple(int(x) for x in hd)
        for ci, mol in enumerate(mols):
            ax = axes[ri][ci]
            pos, A = mol["pos"], mol["maps"][hd]
            n = mol["n"]
            # draw bonds faintly
            for a, b in mol["bonds"]:
                ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], "-", color="#cccccc",
                        lw=1.0, zorder=1)
            # draw the strongest attention edges (off-diagonal), width/alpha ~ weight
            Ao = A.copy(); np.fill_diagonal(Ao, 0.0)
            thr = np.quantile(Ao[Ao > 0], 0.9) if (Ao > 0).any() else 0.0
            mxw = Ao.max() + 1e-9
            for i in range(n):
                for j in range(n):
                    w = Ao[i, j]
                    if w >= thr and w > 0:
                        ax.annotate("", xy=pos[i], xytext=pos[j],
                                    arrowprops=dict(arrowstyle="-|>", color="#d62728",
                                                    alpha=float(min(1.0, w / mxw)),
                                                    lw=0.5 + 2.5 * w / mxw), zorder=2)
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=mol["atom_types"], cmap="tab20", s=120,
                            edgecolors="k", linewidths=0.6, zorder=3)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.set_aspect("equal")
            if ri == 0:
                ax.set_title(f"mol {mol['graph_id']}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{name}\nL{hd[0]}H{hd[1]}", fontsize=9)
    fig.suptitle(f"{attn_data.get('title','')}  ·  top-decile attention edges on the molecule "
                 f"(node colour = atom type)", fontsize=11, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------------------- #
# (iv) ablation.
# --------------------------------------------------------------------------------------- #
def fig_ablation(result, abl, out) -> str:
    _rc()
    L, H = result["L"], result["H"]
    func_mean = abl["func_mean"].reshape(-1)
    hoi = abl["heads_of_interest"]
    tstats = abl["target_stats"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), constrained_layout=True)

    # (a) null histogram of per-head impact with targets marked.
    ax = axes[0, 0]
    ax.hist(func_mean, bins=24, color="#bbbbbb", edgecolor="k", alpha=0.8)
    colors = {"top_semantic": "#1f77b4", "top_structural": "#d62728",
              "top_joint": "#9467bd", "low_both": "#2ca02c"}
    for name, hd in hoi.items():
        v = abl["func_mean"][hd[0], hd[1]]
        ax.axvline(v, color=colors.get(name, "k"), lw=2,
                   label=f"{name} L{hd[0]}H{hd[1]} (rank {tstats[name]['rank']}/{L*H})")
    ax.set_xlabel("mean functional impact  |Δpred|  of ablating a single head")
    ax.set_ylabel("# heads"); ax.set_title("(a) single-head impact vs random-head null", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5)

    # (b) per-graph impact distributions for the interesting heads + a random head.
    ax = axes[0, 1]
    rng = np.random.default_rng(0)
    rl, rh = rng.integers(L), rng.integers(H)
    dist_items = list(hoi.items()) + [("random", [int(rl), int(rh)])]
    data = [abl["func_impact_per_graph"][hd[0], hd[1]] for _, hd in dist_items]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for pc, (name, _) in zip(parts["bodies"], dist_items):
        pc.set_facecolor(colors.get(name, "#888888")); pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(dist_items) + 1))
    ax.set_xticklabels([f"{n}\nL{hd[0]}H{hd[1]}" for n, hd in dist_items], fontsize=7)
    ax.set_ylabel("per-graph functional impact |Δpred|")
    ax.set_title("(b) per-graph impact distribution", fontweight="bold")

    # (c) impact-vs-feature correlation heatmap (interesting heads x features).
    ax = axes[1, 0]
    fnames = abl["feat_names"]
    names = list(hoi.keys())
    M = np.array([[abl["feature_corr"][nm][f] for f in fnames] for nm in names])
    im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(fnames))); ax.set_xticklabels(fnames, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(fnames)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="k" if abs(M[i, j]) < 0.6 else "w")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    ax.grid(False)
    ax.set_title("(c) per-graph impact vs graph feature\n(does the structural head fire on rings?)",
                 fontweight="bold")

    # (d) top-structural head impact vs a structural feature (rings), with fit.
    ax = axes[1, 1]
    ts = tuple(hoi["top_structural"]); tsem = tuple(hoi["top_semantic"])
    feat = abl["features"]
    ax.scatter(feat["n_rings"], abl["func_impact_per_graph"][ts[0], ts[1]], s=16, alpha=0.55,
               color="#d62728", label=f"top_structural L{ts[0]}H{ts[1]} (ρ={abl['feature_corr']['top_structural']['n_rings']:.2f})")
    ax.scatter(feat["n_rings"], abl["func_impact_per_graph"][tsem[0], tsem[1]], s=16, alpha=0.55,
               color="#1f77b4", label=f"top_semantic L{tsem[0]}H{tsem[1]} (ρ={abl['feature_corr']['top_semantic']['n_rings']:.2f})")
    ax.set_xlabel("# rings (cyclomatic number) of the molecule")
    ax.set_ylabel("per-graph functional impact |Δpred|")
    ax.set_title("(d) impact vs molecular ring count", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    sic = abl["score_impact_corr"]
    fig.suptitle(f"{result['title']}  ·  head ablation  (score→impact Spearman: "
                 f"sem={sic['sem_score_vs_impact']:.2f}, str={sic['str_score_vs_impact']:.2f})",
                 fontsize=12, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------------------- #
# Score vs ablation impact: does the specialisation score predict causal importance?
# --------------------------------------------------------------------------------------- #
def fig_score_impact(result, abl, out) -> str:
    """Per-head specialisation score vs ablation impact, with raw + partial correlations.

    Both the score and the impact scale with a head's overall output-reach/throughput, so a raw
    score->impact correlation is expected. The bar panel shows what survives controlling for the
    OTHER channel and for DEPTH (layer) -- i.e. whether the semantic/structural distinction carries
    causal signal beyond that shared amplitude factor.
    """
    _rc()
    L, H = result["L"], result["H"]
    layer = np.repeat(np.arange(L), H)
    impact = abl["func_mean"].reshape(-1)
    sic = abl["score_impact_corr"]
    channels = [("semantic", result["S_sem"].reshape(-1), sic["func"]["sem"], "#1f77b4"),
                ("structural", result["S_str"].reshape(-1), sic["func"]["str"], "#d62728")]
    if result.get("S_attn_sem") is not None and "attn" in sic["func"]:
        channels.append(("attention", result["S_attn_sem"].reshape(-1), sic["func"]["attn"], "#9467bd"))

    ncol = len(channels) + 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.6), constrained_layout=True, squeeze=False)
    axes = axes[0]
    sc = None
    for ax, (name, score, rho, _c) in zip(axes, channels):
        sc = ax.scatter(score, impact, c=layer, cmap="viridis", s=55, edgecolors="k",
                        linewidths=0.4, alpha=0.9, vmin=0, vmax=L - 1)
        ax.set_xlabel(f"{name} score  (raw magnitude)")
        ax.set_ylabel("mean ablation impact  |Δpred|")
        ax.set_title(f"{name}: Spearman ρ = {rho:.2f}", fontsize=10, fontweight="bold")
    if sc is not None:
        fig.colorbar(sc, ax=axes[len(channels) - 1], label="layer", fraction=0.046, pad=0.04)

    # bar panel: raw vs partial correlations (functional impact), + loss-impact raw.
    axb = axes[-1]
    cats = ["raw", "| other\nchannel", "| layer\n(depth)"]
    sem_vals = [sic["func"]["sem"], sic["func"]["sem_ctrl_str"], sic["func"]["sem_ctrl_layer"]]
    str_vals = [sic["func"]["str"], sic["func"]["str_ctrl_sem"], sic["func"]["str_ctrl_layer"]]
    x = np.arange(len(cats)); w = 0.38
    axb.bar(x - w / 2, sem_vals, w, color="#1f77b4", edgecolor="k", label="semantic")
    axb.bar(x + w / 2, str_vals, w, color="#d62728", edgecolor="k", label="structural")
    axb.axhline(0, color="k", lw=0.6)
    axb.set_xticks(x); axb.set_xticklabels(cats, fontsize=8)
    axb.set_ylabel("Spearman ρ  (score vs impact)")
    axb.set_ylim(min(-0.1, min(sem_vals + str_vals) - 0.1), 1.02)
    axb.set_title("(does it survive controls?)", fontsize=10, fontweight="bold")
    axb.legend(frameon=False, fontsize=8)
    axb.text(0.5, -0.28, f"loss-impact raw ρ: sem={sic['loss']['sem']:.2f}, str={sic['loss']['str']:.2f}",
             transform=axb.transAxes, ha="center", fontsize=8, color="#444")
    for xi, (sv, tv) in enumerate(zip(sem_vals, str_vals)):
        axb.text(xi - w / 2, sv + 0.02, f"{sv:.2f}", ha="center", fontsize=7)
        axb.text(xi + w / 2, tv + 0.02, f"{tv:.2f}", ha="center", fontsize=7)

    fig.suptitle(f"{result['title']}  ·  specialisation score vs head-ablation impact  "
                 f"(raw ρ shares the head-importance factor; partials isolate the rest)",
                 fontsize=12, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------------------- #
# Skill-drop vs score, three depth-handling variants.
# --------------------------------------------------------------------------------------- #
def fig_skill_drop_vs_score(result, abl, out) -> str:
    """Ablation skill-drop I(l,h) vs raw per-head score S(l,h), under three depth treatments.

    I(l,h) = mean_graph [ loss(ablate) - loss(clean) ]  (Δ task loss; >0 = skill dropped).
    L̄_S(l), L̄_I(l) = per-layer means over the H heads. Columns:
      * raw          : S(l,h)                vs I(l,h)                 (layers mixed; amplitude in)
      * layer-norm   : S(l,h) / L̄_S(l)       vs I(l,h)                (score rescaled to layer-mean 1)
      * within-layer : S(l,h) - L̄_S(l)       vs I(l,h) - L̄_I(l)       (depth removed from BOTH -> the
                       head-to-head signal inside each layer)
    Rows = semantic / structural score. Points coloured by layer; Spearman ρ per panel.
    """
    _rc()
    L, H = result["L"], result["H"]
    layer_1d = np.repeat(np.arange(L), H)
    I = np.asarray(abl["loss_mean"], float)                 # [L,H] skill drop (Δ loss)
    Im = I.mean(axis=1, keepdims=True)                      # [L,1] per-layer mean impact
    channels = [("semantic", np.asarray(result["S_sem"], float), "#1f77b4"),
                ("structural", np.asarray(result["S_str"], float), "#d62728")]
    variants = ["raw", "layer-norm", "within-layer"]

    fig, axes = plt.subplots(len(channels), 3, figsize=(15.0, 4.7 * len(channels)),
                             constrained_layout=True, squeeze=False)
    sc = None
    for ri, (name, S, _c) in enumerate(channels):
        Sm = S.mean(axis=1, keepdims=True)                 # [L,1] per-layer mean score
        panels = {
            "raw": (S.reshape(-1), I.reshape(-1), f"{name} score  S(l,h)", "skill drop  I(l,h)"),
            "layer-norm": ((S / (Sm + 1e-12)).reshape(-1), I.reshape(-1),
                           f"{name} score / layer-mean", "skill drop  I(l,h)"),
            "within-layer": ((S - Sm).reshape(-1), (I - Im).reshape(-1),
                             f"{name} score - layer-mean", "skill drop  I - layer-mean"),
        }
        for ci, v in enumerate(variants):
            ax = axes[ri][ci]
            x, y, xl, yl = panels[v]
            rho = _spearman_np(x, y)
            sc = ax.scatter(x, y, c=layer_1d, cmap="viridis", s=52, edgecolors="k",
                            linewidths=0.4, alpha=0.9, vmin=0, vmax=L - 1)
            if v == "within-layer":
                ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
            ax.set_xlabel(xl); ax.set_ylabel(yl)
            ax.set_title(f"{v}:  ρ = {rho:.2f}", fontsize=10, fontweight="bold")
    if sc is not None:
        fig.colorbar(sc, ax=axes[:, -1], label="layer index (0 = input)", fraction=0.046, pad=0.02)
    fig.suptitle(f"{result['title']}  ·  skill-drop vs score  (raw | layer-normalised | within-layer)  "
                 f"— each point = one head", fontsize=12, fontweight="bold")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# --------------------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------------------- #
def make_all_figures(results: dict, ablations: dict, attn: dict, out_dir) -> dict:
    """Write every figure for the given models. results/ablations/attn keyed by task name."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gsem, gstr = _global_norms(results)
    figs: dict[str, str] = {}
    for task, r in results.items():
        figs[f"scatter_{task}"] = fig_scatter(r, gsem, gstr, out_dir / f"fig_scatter_{task}.png")
        figs[f"heatmaps_{task}"] = fig_heatmaps(r, out_dir / f"fig_heatmaps_{task}.png")
        if task in ablations:
            figs[f"ablation_{task}"] = fig_ablation(r, ablations[task], out_dir / f"fig_ablation_{task}.png")
            figs[f"score_impact_{task}"] = fig_score_impact(
                r, ablations[task], out_dir / f"fig_score_impact_{task}.png")
            figs[f"skill_drop_{task}"] = fig_skill_drop_vs_score(
                r, ablations[task], out_dir / f"fig_skill_drop_{task}.png")
        if task in attn:
            ad = dict(attn[task]); ad["title"] = r["title"]
            hoi = ablations[task]["heads_of_interest"] if task in ablations \
                else {f"L{l}H{h}": (l, h) for (l, h) in attn[task]["heads"]}
            figs[f"attention_{task}"] = fig_attention(ad, hoi, out_dir / f"fig_attention_{task}.png")
            figs[f"attention_graph_{task}"] = fig_attention_graph(
                ad, hoi, out_dir / f"fig_attention_graph_{task}.png")
    if len(results) > 1:
        figs["scatter_combined"] = fig_scatter_combined(results, gsem, gstr,
                                                         out_dir / "fig_scatter_combined.png")
    return figs
