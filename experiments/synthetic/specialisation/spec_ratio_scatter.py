"""Ratio-based semantic-vs-structural scatter (log-log, calibrated) for the two pure retrieval
tasks and the two-target (mixed) task.

Each channel is divided by its GLOBAL cross-head mean (effect-size calibration), then plotted on
LOG-LOG axes. In log space the shared per-head gain g_h is a translation ALONG the y=x diagonal,
while the semantic/structural RATIO is the perpendicular OFFSET from it: above the diagonal =
semantic-leaning (sem/str ratio > calibrated balance), below = structural-leaning. Gain no longer
confounds the read (it slides points along, not across, the balance line).
"""
import numpy as np, torch
import spec_tasks_train as S
import spec_head_scores as HS
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLS = [("sem_retrieval", "semantic retrieval"), ("str_retrieval", "structural retrieval"),
        ("mixed2", "two-target (mixed)")]
STY = {"dense": dict(color="#1f77b4", marker="o"), "1-hop": dict(color="#d62728", marker="^")}

E = {}
for col, _ in COLS:
    E[col] = []
    if col == "mixed2":
        for seed in range(8):
            r = HS.score_model(f"mix2seed{seed}", "dense", S.CKPT, 200, 6, 15, seed)
            E[col].append(("dense", r["S_sem"].ravel(), r["S_str"].ravel()))
    else:
        for v in ["dense", "1-hop"]:
            r = HS.score_model(col, v, S.CKPT, 200, 6, 15, 0)
            E[col].append((v, r["S_sem"].ravel(), r["S_str"].ravel()))

allsem = np.concatenate([e[1] for col, _ in COLS for e in E[col]])
allstr = np.concatenate([e[2] for col, _ in COLS for e in E[col]])
gsem, gstr = allsem.mean() + 1e-12, allstr.mean() + 1e-12
lo, hi = -1.6, 1.4                                   # log10 axis range (normalised scores)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 140})
fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
for ci, (col, title) in enumerate(COLS):
    a = ax[ci]
    for grp in E[col]:
        vkey, ssem, sstr = grp
        x = np.log10(sstr / gstr + 1e-9); y = np.log10(ssem / gsem + 1e-9)
        a.scatter(x, y, s=40, alpha=0.6, edgecolors="k", linewidths=0.35, **STY[vkey])
    a.plot([lo, hi], [lo, hi], "k:", lw=1.0)         # calibrated balance (sem/str ratio = 1)
    a.fill_between([lo, hi], [lo, hi], hi, color="#1f77b4", alpha=0.05)   # semantic region (above)
    a.fill_between([lo, hi], lo, [lo, hi], color="#d62728", alpha=0.05)   # structural region (below)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect("equal")
    a.set_title(title, fontsize=12, fontweight="bold")
    a.set_xlabel("structural score  $\\log_{10}(\\tilde{str})$")
    if ci == 0:
        a.set_ylabel("semantic score  $\\log_{10}(\\tilde{sem})$")
    a.text(hi - 0.1, hi - 0.15, "semantic\nleaning", ha="right", va="top", fontsize=9, color="#1f77b4")
    a.text(hi - 0.1, lo + 0.25, "structural\nleaning", ha="right", va="bottom", fontsize=9, color="#d62728")
handles = [Line2D([0], [0], linestyle="", markeredgecolor="k", ms=8, **STY[v], label=v) for v in ["dense", "1-hop"]]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=10)
fig.suptitle("Ratio-based semantic vs structural scores (log-log, calibrated):  offset from the dotted "
             "balance line = the sem/str ratio (gain slides along it, not across)", fontsize=11.5, y=1.02)
out = S.CKPT.parent / "fig_ratio_scatter.png"
fig.savefig(out, bbox_inches="tight"); print(f"saved {out}")
