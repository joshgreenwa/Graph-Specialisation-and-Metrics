"""Raw (magnitude) vs CALIBRATED (ratio-based) specialisation scores, for the two pure retrieval
tasks and the two-target (mixed) task.

Calibration: pool all heads, divide each channel by its GLOBAL cross-head mean (matches the two
interventions' effect sizes), then the structural fraction  f = str~/(sem~+str~)  is a monotone
function of the RATIO str/sem, so the shared per-head gain g_h cancels. f<0.5 semantic-leaning,
f>0.5 structural-leaning. Row 0 shows the raw magnitudes (collinear = the confound); row 1 shows f.
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

# ---- collect per-head scores ----
E = {}   # col -> list of (label, variant_style_key, S_sem[12], S_str[12])
for col, _ in COLS:
    E[col] = []
    if col == "mixed2":
        for seed in range(8):
            r = HS.score_model(f"mix2seed{seed}", "dense", S.CKPT, 200, 6, 15, seed)
            E[col].append((f"seed{seed}", "dense", r["S_sem"].ravel(), r["S_str"].ravel()))
    else:
        for v in ["dense", "1-hop"]:
            r = HS.score_model(col, v, S.CKPT, 200, 6, 15, 0)
            E[col].append((v, v, r["S_sem"].ravel(), r["S_str"].ravel()))

allsem = np.concatenate([e[2] for col, _ in COLS for e in E[col]])
allstr = np.concatenate([e[3] for col, _ in COLS for e in E[col]])
gsem, gstr = allsem.mean() + 1e-12, allstr.mean() + 1e-12
frac = lambda ssem, sstr: (sstr / gstr) / ((ssem / gsem) + (sstr / gstr) + 1e-12)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 140})
fig, ax = plt.subplots(2, 3, figsize=(14, 8.4), constrained_layout=True)
mxr = 1.08 * max((allstr / gstr).max(), (allsem / gsem).max())

print(f"{'task':22s}{'variant':8s}{'mean struct-frac f':>20s}")
for ci, (col, title) in enumerate(COLS):
    a0, a1 = ax[0, ci], ax[1, ci]
    for lab, vkey, ssem, sstr in E[col]:
        st = STY[vkey]
        a0.scatter(sstr / gstr, ssem / gsem, s=34, alpha=0.6, edgecolors="k", linewidths=0.3, **st)
    a0.plot([0, mxr], [0, mxr], "k:", lw=0.8); a0.set_xlim(0, mxr); a0.set_ylim(0, mxr)
    a0.set_title(title, fontsize=12, fontweight="bold")
    a0.set_xlabel("structural score (norm.)"); a0.set_ylabel("semantic score (norm.)")
    # row 1: calibrated structural fraction, strip by variant group
    groups = {}
    for lab, vkey, ssem, sstr in E[col]:
        groups.setdefault(vkey, []).append(frac(ssem, sstr))
    for gi, (vkey, fs) in enumerate(groups.items()):
        fs = np.concatenate(fs); st = STY[vkey]
        yj = gi + 0.12 * np.random.default_rng(0).standard_normal(fs.size)
        a1.scatter(fs, yj, s=30, alpha=0.55, edgecolors="k", linewidths=0.3, **st)
        a1.errorbar(fs.mean(), gi, xerr=fs.std() / np.sqrt(fs.size), fmt="D", color=st["color"],
                    ms=9, capsize=4, markeredgecolor="k")
        print(f"{title:22s}{vkey:8s}{fs.mean():>20.2f}")
    a1.axvline(0.5, color="k", lw=0.9, ls="--")
    a1.set_xlim(0, 1); a1.set_yticks(range(len(groups))); a1.set_yticklabels(list(groups))
    a1.set_ylim(-0.6, len(groups) - 0.4)
    a1.set_xlabel("calibrated structural fraction  $f=\\tilde{str}/(\\tilde{sem}+\\tilde{str})$")
ax[0, 0].annotate("RAW scores\n(magnitude)", xy=(-0.34, 0.5), xycoords="axes fraction",
                  ha="center", va="center", rotation=90, fontsize=11, fontweight="bold")
ax[1, 0].annotate("CALIBRATED\nratio score", xy=(-0.34, 0.5), xycoords="axes fraction",
                  ha="center", va="center", rotation=90, fontsize=11, fontweight="bold")
fig.suptitle("Raw magnitude scores are collinear (shared per-head gain);  the calibrated ratio "
             "fraction removes it and tracks the task  (<0.5 semantic, >0.5 structural)", fontsize=11.5)
handles = [Line2D([0], [0], linestyle="", markeredgecolor="k", ms=8, **STY[v], label=v) for v in ["dense", "1-hop"]]
fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.99, 1.04), fontsize=10)
out = S.CKPT.parent / "fig_calibrated_scores.png"
fig.savefig(out, bbox_inches="tight"); print(f"\nsaved {out}")
