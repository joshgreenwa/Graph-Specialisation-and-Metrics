"""Causal double-dissociation, multi-seed. On a TWO-CHANNEL (mixed) task y=0.5 g_sem(content[B]) +
0.5 g_str(nodeRRWP[B]), does ablating the top-SEMANTIC head (Method-A) collapse the SEMANTIC carriage
channel specifically, and the top-STRUCTURAL head the STRUCTURAL channel?

Per seed (independent init + data): train dense model -> Method-A scores S_sem,S_str -> top-sem &
top-str heads -> ablate each -> recompute each channel's FUNCTIONAL carriage C_c = mean_{graph,source}
|ŷ_clean - ŷ_swap| under the donor swap (c=sem) and the node transposition (c=str). Mediation
M[a,c] = 1 - C_c(ablate a)/C_c(clean) = fractional collapse. Prediction: diagonal dominance
M[sem,sem]>M[sem,str] and M[str,str]>M[str,sem], across seeds with CIs.
"""
import math, copy, numpy as np, torch
import spec_tasks_train as S
import spec_head_scores as HS
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.set_num_threads(8)
N, Kw, CV, H, HEADS, L, dh = S.N, S.Kw, S.CV, S.H, S.HEADS, S.L, S.dh
CKPT = S.CKPT
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
GC, KC = 150, 6                                  # graphs / donors-partners for the channel carriage


def forward_ablate(m, feat, nd, pr, mask, abl=None):
    x = m.enc(torch.cat([feat, nd], -1))
    for li, blk in enumerate(m.blocks):
        B = x.size(0); xn = blk.n1(x)
        q = blk.q(xn).view(B, N, HEADS, dh).transpose(1, 2)
        k = blk.k(xn).view(B, N, HEADS, dh).transpose(1, 2)
        v = blk.v(xn).view(B, N, HEADS, dh).transpose(1, 2)
        bias = blk.bb(pr).permute(0, 3, 1, 2)
        a = ((q @ k.transpose(-2, -1)) / math.sqrt(dh) + bias).masked_fill(~mask, float("-inf")).softmax(-1)
        ho = a @ v
        if abl is not None and abl[0] == li:
            ho = ho.clone(); ho[:, abl[1]] = 0.0
        x = x + blk.o(ho.transpose(1, 2).reshape(B, N, H))
        x = x + blk.mlp(blk.n2(x))
    return m.head(x[:, 0]).squeeze(-1)


def train_mixed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, st = S.gen("mixed2", S.G_TR, seed=10 + seed)
    va, _ = S.gen("mixed2", S.G_VA, ystat=st, seed=20 + seed)
    me, _ = S.gen("mixed2", S.G_ME, ystat=st, seed=30 + seed)
    m = S.Net(); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4); lf = torch.nn.MSELoss()
    Mtr, Mva = S.mask_for(tr, True), S.mask_for(va, True); best, bs = -9.0, None
    for ep in range(S.EPOCHS):
        m.train(); perm = torch.randperm(len(tr["y"]))
        for i in range(0, len(perm), S.BS):
            idx = perm[i:i + S.BS]; opt.zero_grad()
            lf(m(tr["feat"][idx], tr["nd"][idx], tr["pr"][idx], Mtr[idx]), tr["y"][idx]).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            vs = S.skill(m(va["feat"], va["nd"], va["pr"], Mva), va["y"])
        if vs > best: best, bs = vs, copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); m.eval()
    tag = f"mix2seed{seed}"
    torch.save({"state": bs, "cfg": dict(N=N, Kw=Kw, CV=CV, H=H, HEADS=HEADS, L=L), "val_skill": best}, CKPT / f"{tag}__dense.pt")
    torch.save({k: me[k] for k in me}, CKPT / f"eval_{tag}.pt")
    return tag, best


def chan_carriage(m, d, abl, rng):
    """(C_sem, C_str): the ablated model's functional carriage per channel = mean |ŷ_clean-ŷ_swap|."""
    G = min(GC, len(d["y"])); ar = torch.arange(G)
    feat, nd, pr = d["feat"][:G], d["nd"][:G], d["pr"][:G]
    mask = S.mask_for({"y": d["y"][:G], "mask1": d["mask1"][:G]}, True)
    content_all = feat[..., 2:2 + CV]; deg = d["adj"][:G].cpu().numpy().sum(-1).astype(int)
    with torch.no_grad():
        yc = forward_ablate(m, feat, nd, pr, mask, abl)
        csem = 0.0
        for j in range(N):                                    # semantic channel: donor swap of node j
            dy = torch.zeros(G)
            for _ in range(KC):
                bp = (np.arange(G) + rng.integers(1, G, size=G)) % G
                nn = rng.integers(0, N, size=G)
                f2 = feat.clone(); f2[ar, j, 2:2 + CV] = content_all[torch.as_tensor(bp), torch.as_tensor(nn)]
                dy += (yc - forward_ablate(m, f2, nd, pr, mask, abl)).abs()
            csem += float((dy / KC).mean())
        cstr = 0.0
        for u in range(N):                                    # structural channel: transposition at u
            vv = HS.sample_partners(deg, u, KC, rng); dy = torch.zeros(G)
            for kk in range(KC):
                nd2, pr2 = HS.conjugate_rrwp(nd, pr, u, torch.as_tensor(vv[:, kk]))
                dy += (yc - forward_ablate(m, feat, nd2, pr2, mask, abl)).abs()
            cstr += float((dy / KC).mean())
    return csem / N, cstr / N


Ms, tops, skills = [], [], []
for seed in SEEDS:
    tag, vsk = train_mixed(seed)
    r = HS.score_model(tag, "dense", CKPT, 300, 8, 30, seed)
    Ssem, Sstr = r["S_sem"], r["S_str"]
    tsem = tuple(int(x) for x in np.unravel_index(Ssem.argmax(), Ssem.shape))
    tstr = tuple(int(x) for x in np.unravel_index(Sstr.argmax(), Sstr.shape))
    m = HS.load_model(tag, "dense", CKPT)
    d = torch.load(CKPT / f"eval_{tag}.pt", map_location="cpu", weights_only=False)
    rng = np.random.default_rng(100 + seed)
    c0 = chan_carriage(m, d, None, rng)
    cs = chan_carriage(m, d, tsem, rng)
    ct = chan_carriage(m, d, tstr, rng)
    M = np.array([[1 - cs[0] / (c0[0] + 1e-12), 1 - cs[1] / (c0[1] + 1e-12)],     # ablate SEM head -> (sem chan, str chan)
                  [1 - ct[0] / (c0[0] + 1e-12), 1 - ct[1] / (c0[1] + 1e-12)]])    # ablate STR head
    Ms.append(M); tops.append((tsem, tstr)); skills.append(vsk)
    same = "  (SAME head!)" if tsem == tstr else ""
    print(f"seed{seed} skill={vsk:.3f} topsem=L{tsem[0]}H{tsem[1]} topstr=L{tstr[0]}H{tstr[1]}{same} "
          f"| M[sem->sem,str]={M[0,0]:+.2f},{M[0,1]:+.2f}  M[str->sem,str]={M[1,0]:+.2f},{M[1,1]:+.2f}", flush=True)

Ms = np.stack(Ms); nS = len(SEEDS)
Mmean, Msem = Ms.mean(0), Ms.std(0) / math.sqrt(nS)
d_sem = Ms[:, 0, 0] - Ms[:, 0, 1]        # sem head: sem-collapse minus str-collapse (want > 0)
d_str = Ms[:, 1, 1] - Ms[:, 1, 0]        # str head: str-collapse minus sem-collapse (want > 0)
print(f"\nMEAN mediation matrix (rows=ablated head, cols=channel collapse), n={nS} seeds:")
print(f"            sem-chan     str-chan")
print(f"  sem head  {Mmean[0,0]:+.2f}±{Msem[0,0]:.2f}  {Mmean[0,1]:+.2f}±{Msem[0,1]:.2f}")
print(f"  str head  {Mmean[1,0]:+.2f}±{Msem[1,0]:.2f}  {Mmean[1,1]:+.2f}±{Msem[1,1]:.2f}")
print(f"dissociation contrasts (want >0): sem-head {d_sem.mean():+.2f}±{d_sem.std()/math.sqrt(nS):.2f} "
      f"({int((d_sem>0).sum())}/{nS} seeds) | str-head {d_str.mean():+.2f}±{d_str.std()/math.sqrt(nS):.2f} "
      f"({int((d_str>0).sum())}/{nS} seeds)")

# ---- figure: grouped bars, per ablated head the two channel collapses (mean +/- SEM over seeds) ----
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 140})
fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
x = np.arange(2); w = 0.36
b1 = ax.bar(x - w / 2, Mmean[:, 0], w, yerr=Msem[:, 0], capsize=4, color="#1f77b4", label="semantic channel collapse")
b2 = ax.bar(x + w / 2, Mmean[:, 1], w, yerr=Msem[:, 1], capsize=4, color="#d62728", label="structural channel collapse")
ax.set_xticks(x); ax.set_xticklabels(["ablate top-SEMANTIC head", "ablate top-STRUCTURAL head"])
ax.set_ylabel("fractional carriage collapse   $M[a,c]=1-C_c^{ablate}/C_c^{clean}$")
ax.axhline(0, color="k", lw=0.7)
ax.set_title(f"Causal double-dissociation on a two-channel task  (mean±SEM, {nS} seeds)\n"
             "diagonal (matched-channel) bars should dominate", fontsize=11)
ax.legend(frameon=False, fontsize=10)
for rect in list(b1) + list(b2):
    h = rect.get_height(); ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                                       ha="center", va="bottom" if h >= 0 else "top", fontsize=9)
out = S.CKPT.parent / "fig_dissociation.png"
fig.savefig(out, bbox_inches="tight"); print(f"\nsaved {out}")
