"""Official /carriage/ functional & beneficial DISTANCE CURVES F(d), B(d) on the synthetic models.

Uses the precise repo methodology UNCHANGED -- imports the estimators straight from
`graph_specialisation_metrics.carriage.core` (pure numpy/torch, no GraphGym dependency):
  * functional carriage  F_sens[i,j] = mean_k ||q[k,i,j]||
                          (core.functional_magnitude_from_delta, g=∂ŷ/∂h^L)
  * loss carriage        C_loss = core.carriage_from_delta(delta, g_loss),  g_loss = ∂|ŷ−y|/∂h^L
  * beneficial carriage  B      = core.beneficial_attribute(C_loss, dL_j, denom='slope')
                                  dL_j = L_clean − mean_k L_swap(j,k)   (exact, L1 loss)
  * distance curves      core.aggregate_carriage_curves(graph_id, distance, F, B)   [log SPD bins]

TWO intervention channels, exactly the two repo interventions, delta = h^L_clean − h^L_swap:
  * SEMANTIC channel  = donor swap (content.py): node j's content <- a REAL donor row from another
    graph; K donors, held structure fixed.  Distance d(i,j).
  * STRUCTURAL channel = node transposition at fixed anchor u (structural.py 'transposition'):
    conjugate node-RRWP, pair-RRWP AND the mask by P_(u v) (the FULL official transposition -- the
    k-hop mask is topology-derived, so structural.py conjugates it via the edge relabel), degree-
    matched partner v over K draws, content fixed.  Distance d(i,u).

Readout note: this model reads a single query node (node 0), so g_i = ∂ŷ/∂h^L_i is nonzero ONLY at
the readout carrier i=0 (no add/mean pooling => g not shared). The carriage C[i,j] = g_i·Δh_i is
therefore supported at i=0; the curves are over the source distance d(0, source) -- the standard
single-readout carriage estimand -- with the estimators applied verbatim (other carriers are exact
structural zeros, like unreachable pairs). Beneficial at i=0 then equals the exact per-source dL_j.

Figure (fig_carriage_curves.png): 4 rows x 2 cols. rows = {F semantic-ch, F structural-ch,
B semantic-ch, B structural-ch}; cols = {semantic task, structural task}; lines = dense vs 1-hop
with graph-clustered 95% CIs. Mixed task ignored.
"""
import sys, math, numpy as np, torch
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src" / "graph_specialisation_metrics" / "carriage"))
import core as cc                                   # the OFFICIAL estimators, imported unchanged
import spec_tasks_train as S                        # architecture + bfs_dist

torch.set_num_threads(8)
N, Kw, CV, H, HEADS, L, dh = S.N, S.Kw, S.CV, S.H, S.HEADS, S.L, S.dh
CKPT = HERE / "ckpts"
TASKS, VARIANTS, CHANNELS = ["sem_retrieval", "str_retrieval"], ["dense", "1-hop"], ["semantic", "structural"]
TASK_LABELS = {"sem_retrieval": "semantic retrieval", "str_retrieval": "structural retrieval",
               "semantic": "semantic", "structural": "structural", "mixed": "mixed"}
GSUB, K = 200, 16                                    # graphs scored, donors/partners per source


def forward_states(m, feat, nd, pr, mask):
    """Return (pred, h^L) where h^L = node states after the L blocks (the tensor read by the head)."""
    x = m.enc(torch.cat([feat, nd], -1))
    for blk in m.blocks:
        B = x.size(0); xn = blk.n1(x)
        q = blk.q(xn).view(B, N, HEADS, dh).transpose(1, 2)
        k = blk.k(xn).view(B, N, HEADS, dh).transpose(1, 2)
        v = blk.v(xn).view(B, N, HEADS, dh).transpose(1, 2)
        bias = blk.bb(pr).permute(0, 3, 1, 2)
        a = ((q @ k.transpose(-2, -1)) / math.sqrt(dh) + bias).masked_fill(~mask, float("-inf")).softmax(-1)
        ho = a @ v
        x = x + blk.o(ho.transpose(1, 2).reshape(B, N, H))
        x = x + blk.mlp(blk.n2(x))
    pred = m.head(x[:, 0]).squeeze(-1)
    return pred, x                                   # x = h^L [B,N,H]


def load_model(task, variant):
    blob = torch.load(CKPT / f"{task}__{variant.replace('-', '')}.pt", map_location="cpu", weights_only=False)
    m = S.Net(); m.load_state_dict(blob["state"]); m.eval()
    return m


def partners_for_graph(deg1d, k, rng):
    """[N,k] degree-matched partner per anchor u (structural.py sample_partners, one graph)."""
    out = np.empty((N, k), dtype=np.int64); nodes = np.arange(N)
    for u in range(N):
        oth = nodes[nodes != u]
        cand = oth[deg1d[oth] == deg1d[u]]
        if cand.size == 0:
            gap = np.abs(deg1d[oth] - deg1d[u]); cand = oth[gap == gap.min()]
        out[u] = rng.choice(cand, size=k, replace=True)
    return out


def conjugate_batch(nd_r, pr_r, mask_r, u, v):
    """Per-replica transposition P_(u_r v_r): node-RRWP rows, pair-RRWP + mask rows AND cols. FULL."""
    R = u.shape[0]; ar = torch.arange(R)
    nd2 = nd_r.clone(); nd2[ar, u] = nd_r[ar, v]; nd2[ar, v] = nd_r[ar, u]
    pr2 = pr_r.clone(); pr2[ar, u] = pr_r[ar, v]; pr2[ar, v] = pr_r[ar, u]              # pair rows
    pr3 = pr2.clone(); pr3[ar, :, u] = pr2[ar, :, v]; pr3[ar, :, v] = pr2[ar, :, u]     # pair cols
    mr = mask_r.clone(); mr[ar, :, u] = mask_r[ar, :, v]; mr[ar, :, v] = mask_r[ar, :, u]  # mask rows
    mk = mr.clone();     mk[ar, :, :, u] = mr[ar, :, :, v]; mk[ar, :, :, v] = mr[ar, :, :, u]  # mask cols
    return nd2, pr3, mk


def curves_for_model(task, variant, rng):
    """Return {'semantic': curves, 'structural': curves} for one model (both intervention channels)."""
    m = load_model(task, variant)
    d = torch.load(CKPT / f"eval_{task}.pt", map_location="cpu", weights_only=False)
    G = min(GSUB, len(d["y"]))
    feat_all, nd_all, pr_all = d["feat"][:G], d["nd"][:G], d["pr"][:G]
    y_all = d["y"][:G].numpy(); adj_all = d["adj"][:G].cpu().numpy()
    content_all = feat_all[..., 2:2 + CV]            # [G,N,CV] real content rows (donor pool)
    dense = (variant == "dense")

    acc = {ch: dict(gid=[], dist=[], F=[], B=[]) for ch in CHANNELS}
    R = N * K
    src_rep = torch.arange(N).repeat_interleave(K)   # [R] source/anchor per replica

    for g in range(G):
        feat = feat_all[g:g + 1]; nd = nd_all[g:g + 1]; pr = pr_all[g:g + 1]
        mask1 = torch.as_tensor(adj_all[g] + np.eye(N)) > 0
        mask = torch.ones(1, 1, N, N, dtype=torch.bool) if dense else mask1[None, None]
        y = float(y_all[g])
        D0 = S.bfs_dist(adj_all[g], 0)               # d(0, ·)  [N]
        deg = adj_all[g].sum(-1).astype(int)

        # clean forward WITH grad -> g_out = ∂ŷ/∂h^L (T=1),  g_loss = ∂|ŷ−y|/∂h^L (L1)
        pred_c, hL = forward_states(m, feat, nd, pr, mask)
        g_out = torch.autograd.grad(pred_c.sum(), hL, retain_graph=True)[0][0]          # [N,H]
        Lc = (pred_c - y).abs().sum()
        g_loss = torch.autograd.grad(Lc, hL)[0][0]                                       # [N,H]
        hL_clean = hL.detach()[0]                                                         # [N,H]
        L_clean = abs(float(pred_c.detach()[0]) - y)
        g_out_T = g_out.detach()[None]                                                    # [1,N,H]
        g_loss_d = g_loss.detach()                                                        # [N,H]

        for ch in CHANNELS:
            with torch.no_grad():
                if ch == "semantic":                 # donor swap: node j's content <- other-graph donor
                    feat_r = feat.repeat(R, 1, 1).clone()
                    gp = (g + 1 + rng.integers(0, G - 1, size=R)) % G     # a DIFFERENT graph per replica
                    nn = rng.integers(0, N, size=R)
                    donor = content_all[torch.as_tensor(gp), torch.as_tensor(nn)]         # [R,CV]
                    feat_r[torch.arange(R), src_rep, 2:2 + CV] = donor
                    pred_s, hL_s = forward_states(m, feat_r, nd.repeat(R, 1, 1),
                                                  pr.repeat(R, 1, 1, 1), mask.repeat(R, 1, 1, 1))
                else:                                # structural: transposition at anchor u=src, mask too
                    part = partners_for_graph(deg, K, rng)                # [N,K]
                    v_rep = torch.as_tensor(part.reshape(R))              # partner per replica
                    nd2, pr2, mk2 = conjugate_batch(nd.repeat(R, 1, 1), pr.repeat(R, 1, 1, 1),
                                                    mask.repeat(R, 1, 1, 1), src_rep, v_rep)
                    pred_s, hL_s = forward_states(m, feat.repeat(R, 1, 1), nd2, pr2, mk2)

                delta = hL_clean.unsqueeze(0) - hL_s                      # [R,N,H]  = h_clean − h_swap
                F_ij = cc.functional_magnitude_from_delta(delta, g_out_T, N, K)           # [N,N]
                C_loss = cc.carriage_from_delta(delta, g_loss_d, N, K).cpu().numpy()       # [N,N]
                L_swap = (pred_s - y).abs().view(N, K).mean(1).cpu().numpy()               # [N]
                dL_j = L_clean - L_swap                                                    # [N] <0=beneficial
                B_ij, _, _ = cc.beneficial_attribute(C_loss, dL_j, denom="slope")          # [N,N]

            # carrier = readout node 0 (only live carrier); source s over reachable nodes, dist d(0,s)
            for s in range(N):
                if not np.isfinite(D0[s]):
                    continue
                acc[ch]["gid"].append(g); acc[ch]["dist"].append(int(D0[s]))
                acc[ch]["F"].append(float(F_ij[0, s])); acc[ch]["B"].append(float(B_ij[0, s]))

    out = {}
    for ch in CHANNELS:
        a = acc[ch]
        out[ch] = cc.aggregate_carriage_curves(np.array(a["gid"]), np.array(a["dist"]),
                                                np.array(a["F"]), np.array(a["B"]),
                                                bin_strategy="log", central="trimmed")
    return out


def main(out_path=None):
    # ---- run 2 tasks x 2 variants ----
    R = {}
    for t in TASKS:
        for v in VARIANTS:
            R[(t, v)] = curves_for_model(t, v, np.random.default_rng(0))
            print(f"[{t}/{v}] curves computed "
                  f"(sem F@d0={R[(t,v)]['semantic']['F_mean'][0]:.2e}, str F@d0={R[(t,v)]['structural']['F_mean'][0]:.2e})",
                  flush=True)

    # ---- figure: 4 rows (F-sem, F-str, B-sem, B-str) x 2 cols (tasks) ----
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "figure.dpi": 140})
    STY = {"dense": dict(color="#1f77b4", marker="o"), "1-hop": dict(color="#d62728", marker="^")}
    ROWS = [("F", "semantic", "Functional $F_{sens}(d)$ · semantic channel (donor swap)"),
            ("F", "structural", "Functional $F_{sens}(d)$ · structural channel (transposition)"),
            ("B", "semantic", "Beneficial $B(d)$ · semantic channel (donor swap)"),
            ("B", "structural", "Beneficial $B(d)$ · structural channel (transposition)")]
    fig, axes = plt.subplots(4, 2, figsize=(11.5, 15.0), constrained_layout=True)

    for ri, (metric, ch, rlab) in enumerate(ROWS):
        for cj, t in enumerate(TASKS):
            ax = axes[ri, cj]
            xr = R[(t, VARIANTS[0])][ch]
            for v in VARIANTS:
                cur = R[(t, v)][ch]
                mk, lk, hk = (("F_mean", "F_lo", "F_hi") if metric == "F" else ("B_mean", "B_lo", "B_hi"))
                ax.plot(cur["bin_center"], cur[mk], **STY[v], lw=1.8, ms=6, label=v)
                ax.fill_between(cur["bin_center"], cur[lk], cur[hk], color=STY[v]["color"], alpha=0.18, linewidth=0)
            if metric == "B":
                ax.axhline(0, color="k", lw=0.7, ls="--")
            ax.set_xticks(xr["bin_center"]); ax.set_xticklabels(xr["bin_label"])
            ax.set_xlabel("shortest-path distance  d(readout, source)")
            ax.set_ylabel("functional $F_{sens}$" if metric == "F" else "beneficial $B$  ($<0$ helps)")
            if ri == 0:
                ax.set_title(f"{TASK_LABELS.get(t, t)} task", fontsize=12, fontweight="bold")
            ax.annotate(rlab, xy=(0.97, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=9,
                        color="#333", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9))

    handles = [Line2D([0], [0], **STY[v], lw=1.8, ms=6, label=v) for v in VARIANTS]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.015), fontsize=11)
    fig.suptitle("Official carriage distance curves — functional $F_{sens}(d)$ and beneficial $B(d)$  "
                 "(semantic & structural channels, dense vs 1-hop)", fontsize=12.5, y=1.035)
    fig.text(0.5, -0.04, "Repo /carriage/ estimators (core.py) applied verbatim: donor swap (semantic channel) and full node "
             "transposition incl. mask (structural channel);  g=∂ŷ/∂h$^L$ functional, g=∂|ŷ−y|/∂h$^L$ + exact dL beneficial;  "
             "log SPD bins, graph-clustered 95% CI.\nCarrier = the single readout node;  B<0 = the source reduces the error.",
             ha="center", fontsize=8.4, color="#333")
    out = Path(out_path) if out_path else (HERE / "fig_carriage_curves.png")
    fig.savefig(out, bbox_inches="tight"); print(f"\nsaved {out}")
    np.savez(HERE / "carriage_curves.npz", **{f"{t}__{v}__{ch}__{key}": R[(t, v)][ch][key]
             for t in TASKS for v in VARIANTS for ch in CHANNELS
             for key in ("bin_center", "bin_label", "F_mean", "F_lo", "F_hi", "B_mean", "B_lo", "B_hi", "pair_counts")})
    return R


if __name__ == "__main__":
    main()
