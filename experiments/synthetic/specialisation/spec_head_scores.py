"""Per-head semantic/structural specialisation scores + the 2-row scatter figure.

Reusable methodology (see SPECIALISATION_SCORES.md for the formal spec). Two scoring methods,
both read at the per-head TRANSPORT site  o^{lh}_i = (attn.V)_{h,i}:

METHOD A -- SEPARATE INTERVENTION (refined) read as per-head carriage.
  semantic  = donor swap (content.py): overwrite node j's CONTENT with a REAL donor row from
              ANOTHER graph (on-manifold); K donors, delta donor-averaged BEFORE abs.
  structural= node transposition at a FIXED anchor u (structural.py): conjugate node/pair-RRWP by
              P_(u v) with a degree-matched partner v (K draws); content held fixed. THE ATTENTION
              MASK IS ARCHITECTURE -> HELD FIXED (frozen), never conjugated: it defines the model's
              receptive field, not input structure. Conjugating it confounds wiring with payload.
  score:  phi^{lh}_i = d yhat / d o^{lh}_i (per-head readout grad at the CLEAN input)
          F_{lh}[i,j] = | phi^{lh}_i . Dbar-o^{lh}_i(j) |   (Dbar-o = donor/partner-averaged delta)
          S_sem/S_str(l,h) = mean_{graph, source} sum_i F  under the semantic / structural swap.
  Alpha-weighted by construction (o is the attention-routed value); phi. weights by output reach.

METHOD B -- SEMANTIC NODE TRANSPOSITION (original; head_scores.py value channel).
  Swap the CONTENT of a random node pair (a,b); per head read whether ho equivariates (semantic)
  or is invariant (structural), each in [0,1], alpha-weighted by value mass ||ho_a||+||ho_b||.

Figure: 2 rows (Method A, Method B) x len(tasks) cols. x=structural score, y=semantic score;
dense vs 1-hop labelled. Method A axes divided by each channel's GLOBAL mean (de-bias the
content-vs-RRWP amplitude) so the diagonal is amplitude-normalised-equal.

CLI:  python spec_head_scores.py [--tasks semantic structural mixed] [--variants dense 1-hop]
                                 [--ckpt-dir ckpts] [--out fig.png] [--gsub 400] [--donors 8]
                                 [--pairs 40] [--seed 0]
Needs, per (task,variant): a checkpoint ckpts/<task>__<variant>.pt (state+cfg) and eval data
ckpts/eval_<task>.pt (feat,nd,pr,mask1,adj,y), as written by spec_tasks_train.py -- so it runs
unchanged on ANY synthetic task trained with that architecture (add the task there, then pass it).
"""
import argparse, math, numpy as np, torch
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import spec_tasks_train as S               # the architecture: Net, dims, mask_for

torch.set_num_threads(8)
N, Kw, CV, H, HEADS, L, dh = S.N, S.Kw, S.CV, S.H, S.HEADS, S.L, S.dh


# ---- forward exposing per-head attention alpha and transport ho (ho kept in-graph for grad) ----
def forward_capture(m, feat, nd, pr, mask):
    x = m.enc(torch.cat([feat, nd], -1)); alphas, hos = [], []
    for blk in m.blocks:
        B = x.size(0); xn = blk.n1(x)
        q = blk.q(xn).view(B, N, HEADS, dh).transpose(1, 2)
        k = blk.k(xn).view(B, N, HEADS, dh).transpose(1, 2)
        v = blk.v(xn).view(B, N, HEADS, dh).transpose(1, 2)
        bias = blk.bb(pr).permute(0, 3, 1, 2)
        a = ((q @ k.transpose(-2, -1)) / math.sqrt(dh) + bias).masked_fill(~mask, float("-inf")).softmax(-1)
        ho = a @ v                                   # [B,HEADS,N,dh]  transport site
        x = x + blk.o(ho.transpose(1, 2).reshape(B, N, H))
        x = x + blk.mlp(blk.n2(x))
        alphas.append(a); hos.append(ho)
    pred = m.head(x[:, 0]).squeeze(-1)
    return pred, alphas, hos


def conjugate_rrwp(nd, pr, u, v):
    """Node transposition P_(u v) on the RRWP FEATURES only (mask is architecture, kept fixed).
    u scalar anchor; v [B] per-graph partner. node-RRWP rows u<->v; pair-RRWP rows AND cols u<->v."""
    B = v.shape[0]; ar = torch.arange(B)
    nd2 = nd.clone(); nd2[ar, u] = nd[ar, v]; nd2[ar, v] = nd[ar, u]
    pr2 = pr.clone(); pr2[ar, u] = pr[ar, v]; pr2[ar, v] = pr[ar, u]                 # rows u<->v
    pr3 = pr2.clone(); pr3[ar, :, u] = pr2[ar, :, v]; pr3[ar, :, v] = pr2[ar, :, u]  # cols u<->v
    return nd2, pr3


def sample_partners(deg, u, k, rng):
    """Per-graph degree-matched partner v for anchor u (structural.py sample_partners)."""
    B, n = deg.shape; out = np.empty((B, k), dtype=np.int64); others = np.arange(n)
    for b in range(B):
        oth = others[others != u]
        cand = oth[deg[b, oth] == deg[b, u]]
        if cand.size == 0:
            gap = np.abs(deg[b, oth] - deg[b, u]); cand = oth[gap == gap.min()]
        out[b] = rng.choice(cand, size=k, replace=True)
    return out                                       # [B,k]


def load_model(task, variant, ckpt_dir):
    blob = torch.load(ckpt_dir / f"{task}__{variant.replace('-', '')}.pt", map_location="cpu", weights_only=False)
    cfg = blob.get("cfg", {})
    for kdim, val in dict(N=N, Kw=Kw, CV=CV, H=H, HEADS=HEADS, L=L).items():
        if cfg.get(kdim, val) != val:
            raise ValueError(f"checkpoint {task}/{variant} cfg.{kdim}={cfg[kdim]} != architecture {val}; "
                             f"re-import the matching spec_tasks_train architecture.")
    m = S.Net(); m.load_state_dict(blob["state"]); m.eval()
    return m


def score_model(task, variant, ckpt_dir, gsub, donors, npairs, seed):
    """Return per-head {S_sem,S_str (Method A), B_sem,B_str (Method B), noop_dh} for one model."""
    rng = np.random.default_rng(seed)
    m = load_model(task, variant, ckpt_dir)
    d = torch.load(ckpt_dir / f"eval_{task}.pt", map_location="cpu", weights_only=False)
    idx = torch.arange(min(gsub, len(d["y"])))
    feat, nd, pr = d["feat"][idx], d["nd"][idx], d["pr"][idx]
    mask = S.mask_for({"y": d["y"][idx], "mask1": d["mask1"][idx]}, variant == "dense")
    deg = d["adj"][idx].sum(-1).cpu().numpy().astype(int)
    B = len(idx); ar = torch.arange(B); K = donors

    # clean forward WITH grad -> per-head readout gradient phi = d yhat / d o^{lh} (clean input)
    pred, _, hos0 = forward_capture(m, feat, nd, pr, mask)
    with torch.no_grad():
        assert torch.allclose(pred, m(feat, nd, pr, mask), atol=1e-5), "forward_capture != Net.forward"
    phis = [p.detach() for p in torch.autograd.grad(pred.sum(), hos0)]
    hos0 = [h.detach() for h in hos0]

    content_all = feat[..., 2:2 + CV]                # [B,N,CV] real content rows
    sources = list(range(1, N))                      # exclude query node 0

    S_sem = np.zeros((L, HEADS)); S_str = np.zeros((L, HEADS)); noop_dh = 0.0
    with torch.no_grad():
        # ---- Method A semantic: donor swap of node j, K donors, average delta BEFORE abs ----
        for j in sources:
            dObar = [torch.zeros_like(h) for h in hos0]
            for _ in range(K):
                bprime = (np.arange(B) + rng.integers(1, B, size=B)) % B          # a DIFFERENT graph
                nodes = rng.integers(0, N, size=B)
                donor = content_all[torch.as_tensor(bprime), torch.as_tensor(nodes)]
                f2 = feat.clone(); f2[ar, j, 2:2 + CV] = donor
                _, _, hp = forward_capture(m, f2, nd, pr, mask)
                for l in range(L):
                    dObar[l] += (hos0[l] - hp[l])
                noop = (donor == content_all[:, j]).all(-1)
                if noop.any():
                    noop_dh = max(noop_dh, max(float((hos0[l][noop] - hp[l][noop]).abs().max()) for l in range(L)))
            for l in range(L):
                c = (phis[l] * (dObar[l] / K)).sum(-1)                            # [B,HEADS,N] phi.Do per carrier
                S_sem[l] += c.abs().sum(dim=(0, 2)).cpu().numpy()
        # ---- Method A structural: transposition at fixed anchor u, mask FROZEN (architecture) ----
        for u in sources:
            vv = sample_partners(deg, u, K, rng)
            dObar = [torch.zeros_like(h) for h in hos0]
            for kk in range(K):
                nd2, pr2 = conjugate_rrwp(nd, pr, u, torch.as_tensor(vv[:, kk]))
                _, _, hp = forward_capture(m, feat, nd2, pr2, mask)              # original mask kept
                for l in range(L):
                    dObar[l] += (hos0[l] - hp[l])
            for l in range(L):
                c = (phis[l] * (dObar[l] / K)).sum(-1)
                S_str[l] += c.abs().sum(dim=(0, 2)).cpu().numpy()
    S_sem /= (len(sources) * B); S_str /= (len(sources) * B)

    # ---- Method B: semantic node transposition, transport-channel equivariance/invariance ----
    veq = np.zeros((L, HEADS)); viv = np.zeros((L, HEADS)); vw = np.zeros((L, HEADS))
    with torch.no_grad():
        for _ in range(npairs):
            aa = rng.integers(1, N, size=B)
            bb = np.array([rng.choice([x for x in range(1, N) if x != aa[b]]) for b in range(B)])
            at, bt = torch.as_tensor(aa), torch.as_tensor(bb)
            f2 = feat.clone()
            ca = f2[ar, at, 2:2 + CV].clone()
            f2[ar, at, 2:2 + CV] = f2[ar, bt, 2:2 + CV]; f2[ar, bt, 2:2 + CV] = ca
            _, _, hp = forward_capture(m, f2, nd, pr, mask)
            for l in range(L):
                hi, hj = hos0[l][ar, :, at], hos0[l][ar, :, bt]
                hi2, hj2 = hp[l][ar, :, at], hp[l][ar, :, bt]
                ev = torch.sqrt(((hj - hi) ** 2).sum(-1) + ((hi - hj) ** 2).sum(-1))
                deq = torch.sqrt(((hi2 - hj) ** 2).sum(-1) + ((hj2 - hi) ** 2).sum(-1))
                din = torch.sqrt(((hi2 - hi) ** 2).sum(-1) + ((hj2 - hj) ** 2).sum(-1))
                f = 0.1 * ev.mean().clamp(min=1e-6)
                eqV = (1 - deq / (ev + f)).clamp(0, 1); ivV = (1 - din / (ev + f)).clamp(0, 1)
                w = hi.norm(dim=-1) + hj.norm(dim=-1)
                veq[l] += (w * eqV).sum(0).cpu().numpy(); viv[l] += (w * ivV).sum(0).cpu().numpy()
                vw[l] += w.sum(0).cpu().numpy()
    B_sem = veq / (vw + 1e-9); B_str = viv / (vw + 1e-9)
    return dict(S_sem=S_sem, S_str=S_str, B_sem=B_sem, B_str=B_str, noop_dh=noop_dh)


def make_figure(R, tasks, variants, out):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.axisbelow": True, "figure.dpi": 140})
    STY = {"dense": dict(marker="o", color="#1f77b4"), "1-hop": dict(marker="^", color="#d62728")}
    dflt = dict(marker="s", color="#2ca02c")
    T = len(tasks)
    fig, axes = plt.subplots(2, T, figsize=(4.5 * T, 9.0), constrained_layout=True, squeeze=False)

    # Method A: GLOBAL per-channel normalisation (one common scale) -> diagonal = amplitude-norm equal.
    gsem = np.mean([R[(t, v)]["S_sem"].mean() for t in tasks for v in variants]) + 1e-12
    gstr = np.mean([R[(t, v)]["S_str"].mean() for t in tasks for v in variants]) + 1e-12
    mxA = 1.08 * max(max((R[(t, v)]["S_str"] / gstr).max(), (R[(t, v)]["S_sem"] / gsem).max())
                     for t in tasks for v in variants)
    for ci, t in enumerate(tasks):
        axA, axB = axes[0, ci], axes[1, ci]
        for v in variants:
            st = STY.get(v, dflt)
            axA.scatter((R[(t, v)]["S_str"] / gstr).ravel(), (R[(t, v)]["S_sem"] / gsem).ravel(),
                        s=55, edgecolors="k", linewidths=0.5, alpha=0.85, **st)
            axB.scatter(R[(t, v)]["B_str"].ravel(), R[(t, v)]["B_sem"].ravel(),
                        s=55, edgecolors="k", linewidths=0.5, alpha=0.85, **st)
        axA.plot([0, mxA], [0, mxA], "k:", lw=0.8); axA.set_xlim(0, mxA); axA.set_ylim(0, mxA)
        axA.set_title(f"{t}", fontsize=12, fontweight="bold")
        axB.plot([0, 1], [0, 1], "k:", lw=0.8); axB.set_xlim(-0.02, 1.02); axB.set_ylim(-0.02, 1.02)
    for ax in axes.ravel():
        ax.set_xlabel("structural score"); ax.set_ylabel("semantic score")
    axes[0, 0].annotate("Method A\nseparate intervention\ndonor swap + node transposition\n(RRWP-feature channel, mask frozen)\nper-head carriage, transport site",
                        xy=(-0.46, 0.5), xycoords="axes fraction", ha="center", va="center", rotation=90, fontsize=10.5, fontweight="bold")
    axes[1, 0].annotate("Method B\nsemantic node transposition\nequivariance / invariance\ntransport site",
                        xy=(-0.46, 0.5), xycoords="axes fraction", ha="center", va="center", rotation=90, fontsize=10.5, fontweight="bold")
    handles = [Line2D([0], [0], linestyle="", markeredgecolor="k", markersize=8,
                      **STY.get(v, dflt), label=v) for v in variants]
    fig.legend(handles=handles, loc="upper center", ncol=len(variants), frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=11)
    fig.suptitle("Per-head semantic vs structural scores  ·  each point = one attention head", fontsize=13, y=1.10)
    fig.text(0.5, 1.055, "Method A: separate donor/transposition interventions read as per-head carriage;  "
             "Method B: single content-transposition equivariance/invariance", ha="center", fontsize=10.5, style="italic")
    fig.text(0.5, -0.055, "Method A axes are divided by each channel's global mean (content swaps move ~unit, RRWP "
             "transpositions ~0.1), so the dotted line is amplitude-normalised equal.  The 1-hop attention mask is "
             "architecture and is\nheld FIXED under the structural intervention (only node/pair-RRWP conjugated).  "
             "Method B scores are raw agreements in [0,1].", ha="center", fontsize=8.6, color="#333333")
    fig.savefig(out, bbox_inches="tight")
    return out


def run(tasks, variants, ckpt_dir, out, gsub=400, donors=8, npairs=40, seed=0):
    ckpt_dir = Path(ckpt_dir)
    R = {}
    for t in tasks:
        for v in variants:
            R[(t, v)] = score_model(t, v, ckpt_dir, gsub, donors, npairs, seed)
            r = R[(t, v)]
            print(f"[{t}/{v}] A sem/str mean={r['S_sem'].mean():.3e}/{r['S_str'].mean():.3e}  "
                  f"B sem/str mean={r['B_sem'].mean():.2f}/{r['B_str'].mean():.2f}  noop|dh|={r['noop_dh']:.1e}", flush=True)
    path = make_figure(R, tasks, variants, Path(out))
    print(f"\nsaved {path}")
    return R


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Per-head semantic/structural specialisation scores + figure.")
    ap.add_argument("--tasks", nargs="+", default=["semantic", "structural", "mixed"])
    ap.add_argument("--variants", nargs="+", default=["dense", "1-hop"])
    ap.add_argument("--ckpt-dir", default=str(HERE / "ckpts"))
    ap.add_argument("--out", default=str(HERE / "fig_head_scores_2x3.png"))
    ap.add_argument("--gsub", type=int, default=400, help="graphs scored per model")
    ap.add_argument("--donors", type=int, default=8, help="K donors/partners per source")
    ap.add_argument("--pairs", type=int, default=40, help="Method-B transposition samples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    run(a.tasks, a.variants, a.ckpt_dir, a.out, a.gsub, a.donors, a.pairs, a.seed)
