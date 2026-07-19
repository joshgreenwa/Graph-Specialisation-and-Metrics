"""Does the score<->ablation-impact correlation survive removing the layer/readout-proximity factor?

The raw per-head score ||phi.Do|| includes phi=dŷ/do^{lh}, whose scale tends to grow for readout-
proximal (late) layers. Ablation impact does too, so their correlation may be partly a shared depth
factor. Test, per task (dense), the head-score vs skill-drop correlation under:
  raw            corr(S, I)
  layer-norm     corr(S / layer_mean_h S, I)          # remove cross-layer SCALE from the score
  within-layer   corr(S - layer_mean, I - layer_mean) # partial corr controlling for layer (the strict test)
plus the per-layer mean score & mean impact (to see the depth factor directly).
Score = task-relevant functional score (sem_retrieval:S_sem, str_retrieval:S_str, mixed2:S_sem+S_str).
"""
import math, numpy as np, torch
import spec_tasks_train as S
import spec_head_scores as HS
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(8)
N, Kw, CV, H, HEADS, L, dh = S.N, S.Kw, S.CV, S.H, S.HEADS, S.L, S.dh
TASKS = [("sem_retrieval", ["dense"], "sem"), ("str_retrieval", ["dense"], "str"),
         ("mixed2", [f"seed{s}" for s in range(8)], "tot")]


def forward_ablate(m, feat, nd, pr, mask, abl=None):
    x = m.enc(torch.cat([feat, nd], -1))
    for li, blk in enumerate(m.blocks):
        B = x.size(0); xn = blk.n1(x)
        q = blk.q(xn).view(B, N, HEADS, dh).transpose(1, 2)
        k = blk.k(xn).view(B, N, HEADS, dh).transpose(1, 2)
        v = blk.v(xn).view(B, N, HEADS, dh).transpose(1, 2)
        a = ((q @ k.transpose(-2, -1)) / math.sqrt(dh) + blk.bb(pr).permute(0, 3, 1, 2)).masked_fill(~mask, float("-inf")).softmax(-1)
        ho = a @ v
        if abl is not None and abl[0] == li:
            ho = ho.clone(); ho[:, abl[1]] = 0.0
        x = x + blk.o(ho.transpose(1, 2).reshape(B, N, H))
        x = x + blk.mlp(blk.n2(x))
    return m.head(x[:, 0]).squeeze(-1)


def pear(a, b): return float(np.corrcoef(a, b)[0, 1])
def spear(a, b): return pear(np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float))


def per_model(tag, variant, chan, seed):
    r = HS.score_model(tag, variant, S.CKPT, 300, 8, 30, seed)
    Sc = {"sem": r["S_sem"], "str": r["S_str"], "tot": r["S_sem"] + r["S_str"]}[chan]   # [L,HEADS]
    m = HS.load_model(tag, variant, S.CKPT)
    d = torch.load(S.CKPT / f"eval_{tag}.pt", map_location="cpu", weights_only=False)
    mask = S.mask_for({"y": d["y"], "mask1": d["mask1"]}, True)
    with torch.no_grad():
        base = S.skill(forward_ablate(m, d["feat"], d["nd"], d["pr"], mask), d["y"])
        I = np.array([[base - S.skill(forward_ablate(m, d["feat"], d["nd"], d["pr"], mask, (l, h)), d["y"])
                       for h in range(HEADS)] for l in range(L)])
    return Sc, I


print(f"{'task':16s}{'metric':6s}{'r_raw':>8s}{'r_layernorm':>12s}{'r_within':>10s}   per-layer  mean-score / mean-impact")
fig, ax = plt.subplots(2, 3, figsize=(14, 8.6), constrained_layout=True)
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, L))
for ti, (base_task, mkeys, chan) in enumerate(TASKS):
    Sr, Ir, Sln, Sw, Iw, lay = [], [], [], [], [], []
    lyr_S = np.zeros(L); lyr_I = np.zeros(L); nm = 0
    for mk in mkeys:
        tag = base_task if base_task != "mixed2" else f"mix2seed{mk[4:]}"
        seed = 0 if base_task != "mixed2" else int(mk[4:])
        variant = mk if base_task != "mixed2" else "dense"
        Sc, I = per_model(tag, variant, chan, seed)
        lm_S = Sc.mean(1, keepdims=True); lm_I = I.mean(1, keepdims=True)
        Sr.append(Sc.ravel()); Ir.append(I.ravel())
        Sln.append((Sc / (lm_S + 1e-12)).ravel())
        Sw.append((Sc - lm_S).ravel()); Iw.append((I - lm_I).ravel())
        lay.append(np.repeat(np.arange(L), HEADS))
        lyr_S += Sc.mean(1); lyr_I += I.mean(1); nm += 1
    Sr, Ir, Sln, Sw, Iw, lay = map(np.concatenate, (Sr, Ir, Sln, Sw, Iw, lay))
    lyr_S /= nm; lyr_I /= nm
    r_raw, r_ln, r_w = pear(Sr, Ir), pear(Sln, Ir), pear(Sw, Iw)
    s_raw, s_ln, s_w = spear(Sr, Ir), spear(Sln, Ir), spear(Sw, Iw)
    ls = " ".join(f"L{l}:{lyr_S[l]:.2e}/{lyr_I[l]:+.2f}" for l in range(L))
    print(f"{base_task:16s}{'pear':6s}{r_raw:>8.2f}{r_ln:>12.2f}{r_w:>10.2f}   {ls}")
    print(f"{'':16s}{'spear':6s}{s_raw:>8.2f}{s_ln:>12.2f}{s_w:>10.2f}")
    # figure: row0 raw score vs impact, row1 layer-normed score vs impact; colour = layer
    for l in range(L):
        msk = lay == l
        ax[0, ti].scatter(Sr[msk], Ir[msk], s=34, color=cmap[l], edgecolors="k", linewidths=0.3, label=f"layer {l}")
        ax[1, ti].scatter(Sln[msk], Ir[msk], s=34, color=cmap[l], edgecolors="k", linewidths=0.3)
    ax[0, ti].set_title(f"{base_task}\nraw: r={r_raw:.2f} (ρ={s_raw:.2f})", fontsize=11, fontweight="bold")
    ax[1, ti].set_title(f"layer-norm: r={r_ln:.2f} (ρ={s_ln:.2f}) | within-layer r={r_w:.2f}", fontsize=10)
    ax[0, ti].set_xlabel("raw score"); ax[1, ti].set_xlabel("layer-normed score")
    for rr in range(2):
        ax[rr, ti].set_ylabel("ablation skill-drop")
ax[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle("Score vs ablation impact — raw (top) vs layer-normalised (bottom).  Colour = layer; "
             "does the correlation survive removing the depth/readout-proximity factor?", fontsize=11.5)
out = S.CKPT.parent / "fig_layernorm_ablation.png"
fig.savefig(out, bbox_inches="tight"); print(f"\nsaved {out}")
