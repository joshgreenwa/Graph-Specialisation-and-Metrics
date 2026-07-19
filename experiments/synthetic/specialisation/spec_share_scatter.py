"""Semantic-share vs structural-share scatter. Each head's score is expressed as its SHARE of that
channel's total across the model's heads:  sem_share_h = S_sem_h / sum_h S_sem_h  (sums to 1),
str_share_h = S_str_h / sum_h S_str_h.  These are INDEPENDENT axes (they do NOT sum to 1 with each
other -- unlike sem/(sem+str) which would collapse onto a line), each self-normalised to its own
channel so the intervention-amplitude difference cancels. Above the y=x diagonal = the head carries a
larger share of the semantic total than the structural total (semantic-specialised); below =
structural-specialised. Shares computed PER MODEL (per seed) then pooled.
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

def shares(ssem, sstr):
    return ssem / (ssem.sum() + 1e-12), sstr / (sstr.sum() + 1e-12)

E = {}
for col, _ in COLS:
    E[col] = []
    if col == "mixed2":
        for seed in range(8):
            r = HS.score_model(f"mix2seed{seed}", "dense", S.CKPT, 200, 6, 15, seed)
            E[col].append(("dense",) + shares(r["S_sem"].ravel(), r["S_str"].ravel()))
    else:
        for v in ["dense", "1-hop"]:
            r = HS.score_model(col, v, S.CKPT, 200, 6, 15, 0)
            E[col].append((v,) + shares(r["S_sem"].ravel(), r["S_str"].ravel()))

mx = 1.05 * max(max(e[1].max(), e[2].max()) for col, _ in COLS for e in E[col])
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 140})
fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), constrained_layout=True)
for ci, (col, title) in enumerate(COLS):
    a = ax[ci]
    for vkey, semsh, strsh in E[col]:
        a.scatter(strsh, semsh, s=42, alpha=0.6, edgecolors="k", linewidths=0.35, **STY[vkey])
    a.plot([0, mx], [0, mx], "k:", lw=1.0)
    a.fill_between([0, mx], [0, mx], mx, color="#1f77b4", alpha=0.05)
    a.fill_between([0, mx], 0, [0, mx], color="#d62728", alpha=0.05)
    a.axhline(1 / (S.L * S.HEADS), color="0.7", lw=0.6, ls="-")           # uniform share 1/12
    a.axvline(1 / (S.L * S.HEADS), color="0.7", lw=0.6, ls="-")
    a.set_xlim(0, mx); a.set_ylim(0, mx); a.set_aspect("equal")
    a.set_title(title, fontsize=12, fontweight="bold")
    a.set_xlabel("structural share  $S_{str,h}/\\sum_h S_{str}$")
    if ci == 0:
        a.set_ylabel("semantic share  $S_{sem,h}/\\sum_h S_{sem}$")
    a.text(0.03 * mx, 0.97 * mx, "semantic-specialised", fontsize=9, color="#1f77b4", va="top")
    a.text(0.97 * mx, 0.03 * mx, "structural-specialised", fontsize=9, color="#d62728", ha="right", va="bottom")
handles = [Line2D([0], [0], linestyle="", markeredgecolor="k", ms=8, **STY[v], label=v) for v in ["dense", "1-hop"]]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=10)
fig.suptitle("Per-head semantic-share vs structural-share (each channel self-normalised to its total):  "
             "offset from the diagonal = specialisation, distance from origin = importance", fontsize=11.5, y=1.02)
out = S.CKPT.parent / "fig_share_scatter.png"
fig.savefig(out, bbox_inches="tight"); print(f"saved {out}")
