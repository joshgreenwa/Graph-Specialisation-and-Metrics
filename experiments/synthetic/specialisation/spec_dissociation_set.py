"""Iteration: redundancy-aware SET ablation + dose-response for the double-dissociation.

Reuses the trained mix2seed{0..7}__dense models (two-target task). Single-head ablation localised the
SEMANTIC channel but not the STRUCTURAL one -> hypothesis: structural processing is redundant/
distributed. Test it by ablating the top-q head SET of each type jointly (q=1,2,3) and reading the
dose-response of each channel's functional-carriage collapse. If the structural channel only collapses
once a SET is removed (dM/dq rising), that quantifies redundancy; if it never collapses, structure is
genuinely non-localised.
"""
import math, numpy as np, torch
import spec_tasks_train as S
import spec_head_scores as HS
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(8)
N, Kw, CV, H, HEADS, L, dh = S.N, S.Kw, S.CV, S.H, S.HEADS, S.L, S.dh
CKPT = S.CKPT
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
QS = [1, 2, 3]
GC, KC = 150, 6


def forward_ablate(m, feat, nd, pr, mask, abl_set=()):
    absmap = {}
    for (l, h) in abl_set:
        absmap.setdefault(l, []).append(h)
    x = m.enc(torch.cat([feat, nd], -1))
    for li, blk in enumerate(m.blocks):
        B = x.size(0); xn = blk.n1(x)
        q = blk.q(xn).view(B, N, HEADS, dh).transpose(1, 2)
        k = blk.k(xn).view(B, N, HEADS, dh).transpose(1, 2)
        v = blk.v(xn).view(B, N, HEADS, dh).transpose(1, 2)
        bias = blk.bb(pr).permute(0, 3, 1, 2)
        a = ((q @ k.transpose(-2, -1)) / math.sqrt(dh) + bias).masked_fill(~mask, float("-inf")).softmax(-1)
        ho = a @ v
        if li in absmap:
            ho = ho.clone()
            for h in absmap[li]:
                ho[:, h] = 0.0
        x = x + blk.o(ho.transpose(1, 2).reshape(B, N, H))
        x = x + blk.mlp(blk.n2(x))
    return m.head(x[:, 0]).squeeze(-1)


def chan_carriage(m, d, abl_set, rng):
    G = min(GC, len(d["y"])); ar = torch.arange(G)
    feat, nd, pr = d["feat"][:G], d["nd"][:G], d["pr"][:G]
    mask = S.mask_for({"y": d["y"][:G], "mask1": d["mask1"][:G]}, True)
    content_all = feat[..., 2:2 + CV]; deg = d["adj"][:G].cpu().numpy().sum(-1).astype(int)
    with torch.no_grad():
        yc = forward_ablate(m, feat, nd, pr, mask, abl_set)
        csem = 0.0
        for j in range(N):
            dy = torch.zeros(G)
            for _ in range(KC):
                bp = (np.arange(G) + rng.integers(1, G, size=G)) % G
                nn = rng.integers(0, N, size=G)
                f2 = feat.clone(); f2[ar, j, 2:2 + CV] = content_all[torch.as_tensor(bp), torch.as_tensor(nn)]
                dy += (yc - forward_ablate(m, f2, nd, pr, mask, abl_set)).abs()
            csem += float((dy / KC).mean())
        cstr = 0.0
        for u in range(N):
            vv = HS.sample_partners(deg, u, KC, rng); dy = torch.zeros(G)
            for kk in range(KC):
                nd2, pr2 = HS.conjugate_rrwp(nd, pr, u, torch.as_tensor(vv[:, kk]))
                dy += (yc - forward_ablate(m, feat, nd2, pr2, mask, abl_set)).abs()
            cstr += float((dy / KC).mean())
    return csem / N, cstr / N


def top_set(score, q):
    order = np.argsort(score.ravel())[::-1][:q]
    return [(int(i // HEADS), int(i % HEADS)) for i in order]


# M[seed, q, ablated_type, channel] ; ablated_type 0=sem-set 1=str-set ; channel 0=sem 1=str
M = np.full((len(SEEDS), len(QS), 2, 2), np.nan)
for si, seed in enumerate(SEEDS):
    tag = f"mix2seed{seed}"
    r = HS.score_model(tag, "dense", CKPT, 300, 8, 30, seed)
    Ssem, Sstr = r["S_sem"], r["S_str"]
    m = HS.load_model(tag, "dense", CKPT)
    d = torch.load(CKPT / f"eval_{tag}.pt", map_location="cpu", weights_only=False)
    rng = np.random.default_rng(200 + seed)
    c0 = chan_carriage(m, d, (), rng)
    for qi, q in enumerate(QS):
        cs = chan_carriage(m, d, top_set(Ssem, q), rng)
        ct = chan_carriage(m, d, top_set(Sstr, q), rng)
        M[si, qi, 0] = [1 - cs[0] / (c0[0] + 1e-12), 1 - cs[1] / (c0[1] + 1e-12)]
        M[si, qi, 1] = [1 - ct[0] / (c0[0] + 1e-12), 1 - ct[1] / (c0[1] + 1e-12)]
    print(f"seed{seed}: q=3 sem-set->(sem {M[si,2,0,0]:+.2f}, str {M[si,2,0,1]:+.2f})  "
          f"str-set->(sem {M[si,2,1,0]:+.2f}, str {M[si,2,1,1]:+.2f})", flush=True)

nS = len(SEEDS)
Mm = np.nanmean(M, 0); Me = np.nanstd(M, 0) / math.sqrt(nS)   # [q,type,chan]
print(f"\nSET-ablation dose-response (mean±SEM over {nS} seeds), fractional channel collapse:")
for qi, q in enumerate(QS):
    print(f"  q={q}:  sem-set-> sem {Mm[qi,0,0]:+.2f}±{Me[qi,0,0]:.2f}, str {Mm[qi,0,1]:+.2f}±{Me[qi,0,1]:.2f}"
          f"   |  str-set-> sem {Mm[qi,1,0]:+.2f}±{Me[qi,1,0]:.2f}, str {Mm[qi,1,1]:+.2f}±{Me[qi,1,1]:.2f}")
dstr = M[:, :, 1, 1] - M[:, :, 1, 0]                          # str-set: str-collapse minus sem-collapse per q
print("structural dissociation contrast (str-set: str minus sem collapse), want >0 & rising:")
for qi, q in enumerate(QS):
    print(f"  q={q}: {dstr[:,qi].mean():+.2f}±{dstr[:,qi].std()/math.sqrt(nS):.2f} ({int((dstr[:,qi]>0).sum())}/{nS} seeds)")

# ---- figure: dose-response, diagonal (matched) vs off-diagonal collapse per channel ----
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 140})
fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True, sharey=True)
qx = np.array(QS)
ax[0].errorbar(qx, Mm[:, 0, 0], Me[:, 0, 0], marker="o", color="#1f77b4", capsize=3, label="semantic channel (matched)")
ax[0].errorbar(qx, Mm[:, 0, 1], Me[:, 0, 1], marker="s", color="#d62728", capsize=3, label="structural channel (off)")
ax[0].set_title("ablate top-q SEMANTIC heads"); ax[0].axhline(0, color="k", lw=0.7)
ax[1].errorbar(qx, Mm[:, 1, 1], Me[:, 1, 1], marker="o", color="#d62728", capsize=3, label="structural channel (matched)")
ax[1].errorbar(qx, Mm[:, 1, 0], Me[:, 1, 0], marker="s", color="#1f77b4", capsize=3, label="semantic channel (off)")
ax[1].set_title("ablate top-q STRUCTURAL heads"); ax[1].axhline(0, color="k", lw=0.7)
for a in ax:
    a.set_xlabel("q  (size of ablated head set)"); a.set_xticks(qx); a.legend(frameon=False, fontsize=9)
ax[0].set_ylabel("fractional carriage collapse  $1-C_c^{ablate}/C_c^{clean}$")
fig.suptitle(f"Set-ablation dose-response: does the matched channel collapse as more heads are removed?  "
             f"({nS} seeds, two-target task)", fontsize=11.5)
out = S.CKPT.parent / "fig_dissociation_doseresponse.png"
fig.savefig(out, bbox_inches="tight"); print(f"\nsaved {out}")
