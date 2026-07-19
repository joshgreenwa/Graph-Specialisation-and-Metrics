"""Causal check: does ablating the top Method-A (separate-intervention) head for a task's channel
hurt more than ablating a random other head? Dense (well-fit) models.

For sem_retrieval -> top SEMANTIC head (argmax S_sem); for str_retrieval -> top STRUCTURAL head
(argmax S_str). Ablate = zero that head's routed output o^{lh}=attn.V. Impact = skill (R^2) drop on
the eval set. Baseline = ablate every OTHER head individually -> random-head impact distribution.
"""
import math, numpy as np, torch
import spec_tasks_train as S
import spec_head_scores as HS

torch.set_num_threads(8)
N, Kw, CV, H, HEADS, L, dh = S.N, S.Kw, S.CV, S.H, S.HEADS, S.L, S.dh


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
            ho = ho.clone(); ho[:, abl[1]] = 0.0                    # ablate head abl[1] in layer li
        x = x + blk.o(ho.transpose(1, 2).reshape(B, N, H))
        x = x + blk.mlp(blk.n2(x))
    return m.head(x[:, 0]).squeeze(-1)


def skill_of(m, d, mask, abl=None):
    with torch.no_grad():
        pred = forward_ablate(m, d["feat"], d["nd"], d["pr"], mask, abl)
    return S.skill(pred, d["y"])


print(f"{'task':16s}{'channel':8s}{'top head':10s}{'score':>10s}{'drop_top':>10s}"
      f"{'rand_mean':>10s}{'rand_max':>10s}{'rank/12':>9s}{'top/mean':>9s}")
for task, chan in [("sem_retrieval", "sem"), ("str_retrieval", "str")]:
    r = HS.score_model(task, "dense", S.CKPT, 400, 8, 40, 0)
    score = r["S_sem"] if chan == "sem" else r["S_str"]            # [L,HEADS]
    m = HS.load_model(task, "dense", S.CKPT)
    d = torch.load(S.CKPT / f"eval_{task}.pt", map_location="cpu", weights_only=False)
    mask = S.mask_for({"y": d["y"], "mask1": d["mask1"]}, True)    # dense mask
    base = skill_of(m, d, mask, None)
    imp = np.array([[base - skill_of(m, d, mask, (l, h)) for h in range(HEADS)] for l in range(L)])
    top = tuple(int(x) for x in np.unravel_index(np.argmax(score), score.shape))
    top_imp = float(imp[top])
    others = np.array([imp[l, h] for l in range(L) for h in range(HEADS) if (l, h) != top])
    rank = 1 + int((others > top_imp).sum())                      # 1 = highest-impact head
    print(f"{task:16s}{chan:8s}L{top[0]}H{top[1]:<8d}{score[top]:>10.2e}{top_imp:>10.3f}"
          f"{others.mean():>10.3f}{others.max():>10.3f}{rank:>6d}/12{top_imp/(others.mean()+1e-9):>9.1f}x")
