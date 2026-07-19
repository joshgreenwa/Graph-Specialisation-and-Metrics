"""Per-head semantic-vs-structural score scatter for the two-target (mixed2) task, pooled over the
8 trained dense seeds. Method-A (separate-intervention) scores; global per-channel normalisation so
the diagonal is amplitude-normalised equal. Stars/triangles = the per-seed top-semantic / top-
structural heads (the ones ablated in the dissociation analysis).
"""
import numpy as np, torch
import spec_tasks_train as S
import spec_head_scores as HS
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
sem_all, str_all, tops, topt = [], [], [], []
for seed in SEEDS:
    r = HS.score_model(f"mix2seed{seed}", "dense", S.CKPT, 250, 6, 20, seed)
    Ssem, Sstr = r["S_sem"].ravel(), r["S_str"].ravel()
    sem_all.append(Ssem); str_all.append(Sstr)
    tops.append(int(Ssem.argmax())); topt.append(int(Sstr.argmax()))
sem_all = np.concatenate(sem_all); str_all = np.concatenate(str_all)
gsem, gstr = sem_all.mean() + 1e-12, str_all.mean() + 1e-12
xs, ys = str_all / gstr, sem_all / gsem
H = S.L * S.HEADS
# per-seed top-head indices into the pooled array
ts_idx = [i * H + tops[i] for i in range(len(SEEDS))]
tt_idx = [i * H + topt[i] for i in range(len(SEEDS))]

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 140})
fig, ax = plt.subplots(figsize=(6.8, 6.4), constrained_layout=True)
ax.scatter(xs, ys, s=34, color="#4c78a8", edgecolors="k", linewidths=0.3, alpha=0.55, label="all heads (8 seeds)")
ax.scatter(xs[ts_idx], ys[ts_idx], s=150, marker="*", color="#1f77b4", edgecolors="k", linewidths=0.6, label="top-semantic head (ablated)")
ax.scatter(xs[tt_idx], ys[tt_idx], s=90, marker="^", color="#d62728", edgecolors="k", linewidths=0.6, label="top-structural head (ablated)")
mx = 1.08 * max(xs.max(), ys.max())
ax.plot([0, mx], [0, mx], "k:", lw=0.8); ax.set_xlim(0, mx); ax.set_ylim(0, mx)
ax.set_xlabel("structural score  (norm.)"); ax.set_ylabel("semantic score  (norm.)")
ax.set_title("Per-head semantic vs structural scores\ntwo-target task (both channels), 8 dense seeds", fontsize=12, fontweight="bold")
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.text(0.03, 0.97, "above diagonal = semantic-leaning", transform=ax.transAxes, fontsize=8.5, color="#555", va="top")
out = S.CKPT.parent / "fig_scatter_mixed2.png"
fig.savefig(out, bbox_inches="tight"); print(f"saved {out}  ({len(xs)} heads)")
