"""Train dense-GT and 1-hop GT variants on pure-semantic, pure-structural and mixed tasks.

Architecture (deliberately simple, exactly the spec):
  node feats = [query-flag, target-flag, content(CV), node-RRWP(Kw)] -> Linear -> H
  L blocks of: standard multi-head attention with a STATIC additive RRWP bias
      logit_{h,ij} = (q_i . k_j)/sqrt(dh) + Linear(pair-RRWP_{ij})[h]
    softmax over j (masked: dense=all nodes, 1-hop = adjacency+self), value = attn . V.
    NO structural value term (pv) -- attention bias is the only structural routing channel;
    structure still enters as node-RRWP node features (a routable payload).
  readout = Linear(H->1) on the query node's (node 0) final state.  MSE regression.

Three tasks -- SAME retrieval skeleton (route target node B's payload to node 0), different
payload, so "semantic vs structural" is a clean payload contrast:
  (i)  semantic   y = g_sem(content[B])          (content payload; structure is a distractor)
  (ii) structural y = g_str(nodeRRWP[B])         (structural payload; content is a distractor)
  (iii)mixed      y = 0.5 g_sem(content[B]) + 0.5 g_str(nodeRRWP[B])   (needs both)
g_sem, g_str are FROZEN random 2-layer tanh MLPs; y standardised per task.

Dense reaches B directly (1 layer suffices); 1-hop needs d(0,B) <= L (reachability wall),
so the dense-vs-1hop gap is a function of the target-distance distribution -- reported.

Saves per (task,variant): model state_dict + config, and per task the eval graphs + frozen
task MLPs, under ckpts/, so the specialisation/carriage step reloads identical models & data.
"""
import os, math, time, json, copy, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import deque

def bfs_dist(A, src):
    """Shortest-path hop distance from src to every node (-1 if unreachable)."""
    n = A.shape[0]; d = np.full(n, -1, dtype=np.int64); d[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for w in np.nonzero(A[u])[0]:
            if d[w] < 0: d[w] = d[u] + 1; q.append(int(w))
    return d

torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
SMOKE = os.environ.get("SMOKE", "0") == "1"
HERE = Path(__file__).resolve().parent
CKPT = HERE / "ckpts"; CKPT.mkdir(parents=True, exist_ok=True)

# ---- dims / training ---------------------------------------------------------------------
N, Kw, CV, H, HEADS, L = 12, 6, 8, 32, 4, 3
KK, VV = 4, 4                 # content split for retrieval: key dims + value dims (KK+VV = CV)
dh = H // HEADS
G_TR, G_VA, G_ME = (1500, 400, 300) if SMOKE else (5000, 1200, 800)
EPOCHS = 6 if SMOKE else 60
BS, LR, WD = 256, 1e-3, 1e-4
TASKS = ["sem_retrieval", "str_retrieval"]
VARIANTS = ["dense", "1-hop"]
EXTRA_EDGES_MAX = 3            # keep some diameter so 1-hop reachability matters

# ---- frozen task read-out MLPs -----------------------------------------------------------
def mk_mlp(d, seed):
    r = np.random.RandomState(seed)
    return (r.randn(d, 8).astype(np.float32) / np.sqrt(d),
            r.randn(8).astype(np.float32),
            r.randn(8, 1).astype(np.float32) / np.sqrt(8))
def apply_g(x, g):
    W1, b1, W2 = g
    return (np.tanh(x @ W1 + b1) @ W2)[..., 0]
G_SEM = mk_mlp(CV, 101)       # legacy flag-retrieval tasks (semantic/structural/mixed)
G_STR = mk_mlp(Kw, 202)
G_SEMRET = mk_mlp(VV, 303)    # sem_retrieval payload MLP (on the value of the matched node)
G_STRRET = mk_mlp(Kw, 404)    # str_retrieval payload MLP (on the node-RRWP of the matched node)
W_STR = np.random.RandomState(505).randn(Kw).astype(np.float32)  # fixed structural query direction

# ---- graph + RRWP ------------------------------------------------------------------------
def rand_graph_adj(n):
    A = np.zeros((n, n), np.float32)
    for i in range(1, n):                       # random spanning tree (keeps some diameter -> reach contrast)
        p = np.random.randint(0, i); A[i, p] = A[p, i] = 1.0
    for _ in range(np.random.randint(0, EXTRA_EDGES_MAX + 1)):
        a, b = np.random.randint(0, n), np.random.randint(0, n)
        if a != b: A[a, b] = A[b, a] = 1.0
    return A
def rrwp(A):
    G = A.shape[0]; I = torch.eye(N).expand(G, N, N); Asl = A + I
    M = Asl / Asl.sum(-1, keepdim=True)
    nd = [torch.ones(G, N)]; pr = [I.clone()]; cur = I.clone()
    for _ in range(1, Kw):
        cur = cur @ M
        nd.append(torch.diagonal(cur, dim1=1, dim2=2)); pr.append(cur.clone())
    return torch.stack(nd, -1), torch.stack(pr, -1)      # [G,N,Kw], [G,N,N,Kw]

def gen(task, G, ystat=None, seed=0):
    rs = np.random.RandomState(seed)
    np.random.seed(seed)
    A = np.stack([rand_graph_adj(N) for _ in range(G)]); At = torch.tensor(A)
    nd, pr = rrwp(At)
    cont = rs.randn(G, N, CV).astype(np.float32)
    ar = np.arange(G)
    ndn = nd.numpy()
    flags = np.zeros((G, N, 2), np.float32); flags[:, 0, 0] = 1.0   # query flag @ readout node 0
    if task in ("semantic", "structural", "mixed"):
        # flag-marked retrieval: target B is flagged; payload routed to node 0.
        B = rs.randint(1, N, size=G); flags[ar, B, 1] = 1.0        # target flag @ B
        y_sem = apply_g(cont[ar, B], G_SEM); y_str = apply_g(ndn[ar, B], G_STR)
        y = y_sem if task == "semantic" else y_str if task == "structural" else 0.5 * (y_sem + y_str)
    elif task == "sem_retrieval":
        # content-addressed with a PLANTED clear key-match: one node j* has key ≈ node 0's query key
        # (others i.i.d.), so the argmax is unambiguous & the read generalises. Read j*'s value. Pure semantic.
        B = rs.randint(1, N, size=G)                              # planted match location j*
        cont[ar, B, :KK] = cont[ar, 0, :KK] + 0.15 * rs.randn(G, KK).astype(np.float32)  # key_{j*} ≈ key_0
        y = apply_g(cont[ar, B, KK:KK + VV], G_SEMRET)            # read the matched node's value
    elif task == "str_retrieval":
        # structural POINTER retrieval: a flagged random target B (unconstrained structure) -> read its
        # node-RRWP. Pure structural. A content-key-query can't select structurally here (structural
        # keys are low-variance and selecting-by-structure constrains the read to ~0 variance), so the
        # target is addressed by a content-free pointer flag; the payload nodeRRWP[B] is the structural read.
        B = rs.randint(1, N, size=G); flags[ar, B, 1] = 1.0        # pointer flag @ target B
        y = apply_g(ndn[ar, B], G_STRRET)
    elif task == "mixed2":
        # TWO-CHANNEL, TWO-TARGET: content of B1 (flag +1) AND node-RRWP of a DISTINCT B2 (flag -1).
        # Distinct sources force distinct routing -> separable semantic vs structural heads (for the
        # causal double-dissociation). Stays in the 2-flag layout (content at cols 2:2+CV).
        B1 = rs.randint(1, N, size=G)
        B2 = np.array([rs.choice([x for x in range(1, N) if x != B1[g]]) for g in range(G)])
        flags[ar, B1, 1] = 1.0; flags[ar, B2, 1] = -1.0
        y = 0.5 * apply_g(cont[ar, B1], G_SEM) + 0.5 * apply_g(ndn[ar, B2], G_STR)
        B = B1
    else:
        raise ValueError(f"unknown task {task!r}")
    if ystat is None: ystat = (float(y.mean()), float(y.std()) + 1e-6)
    y = (y - ystat[0]) / ystat[1]
    feat = torch.tensor(np.concatenate([flags, cont], -1))          # [G,N,2+CV]
    mask1 = torch.tensor((A + np.eye(N)) > 0)                        # [G,N,N] 1-hop+self
    # shortest path d(0,B) for stratified reporting
    d0B = np.zeros(G, dtype=np.int64)
    for g in range(G):
        d0B[g] = int(bfs_dist(A[g], 0)[B[g]])
    data = dict(feat=feat, nd=nd, pr=pr, y=torch.tensor(y.astype(np.float32)),
                B=B, adj=At, mask1=mask1, d0B=d0B)
    return data, ystat

# ---- model ------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.q = nn.Linear(H, H); s.k = nn.Linear(H, H); s.v = nn.Linear(H, H); s.o = nn.Linear(H, H)
        s.bb = nn.Linear(Kw, HEADS)                    # static RRWP additive attention bias
        s.n1 = nn.LayerNorm(H); s.n2 = nn.LayerNorm(H)
        s.mlp = nn.Sequential(nn.Linear(H, 2 * H), nn.GELU(), nn.Linear(2 * H, H))
    def forward(s, x, pr, mask, cap=False):
        B = x.size(0); xn = s.n1(x)
        q = s.q(xn).view(B, N, HEADS, dh).transpose(1, 2)
        k = s.k(xn).view(B, N, HEADS, dh).transpose(1, 2)
        v = s.v(xn).view(B, N, HEADS, dh).transpose(1, 2)
        bias = s.bb(pr).permute(0, 3, 1, 2)            # [B,HEADS,N,N]
        a = ((q @ k.transpose(-2, -1)) / math.sqrt(dh) + bias).masked_fill(~mask, float("-inf")).softmax(-1)
        ho = a @ v                                     # [B,HEADS,N,dh]  routed value (transport site)
        x = x + s.o(ho.transpose(1, 2).reshape(B, N, H))
        x = x + s.mlp(s.n2(x))
        if cap: return x, (a.detach(), ho.detach())
        return x
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc = nn.Linear(2 + CV + Kw, H)
        s.blocks = nn.ModuleList([Block() for _ in range(L)])
        s.head = nn.Linear(H, 1)
    def forward(s, feat, nd, pr, mask, cap=False):
        x = s.enc(torch.cat([feat, nd], -1)); caps = []
        for b in s.blocks:
            if cap: x, c = b(x, pr, mask, cap=True); caps.append(c)
            else:   x = b(x, pr, mask)
        pred = s.head(x[:, 0]).squeeze(-1)
        return (pred, caps) if cap else pred

def mask_for(data, dense):
    G = len(data["y"])
    if dense: return torch.ones(G, 1, N, N, dtype=torch.bool)
    return data["mask1"][:, None]

def skill(pred, y):
    return 1.0 - ((pred - y) ** 2).mean().item() / (y.var().item() + 1e-12)

# ---- train one (task, variant) -----------------------------------------------------------
def train(task, variant, tr, va, me):
    torch.manual_seed(0); np.random.seed(0)
    dense = (variant == "dense")
    m = Net(); opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD); lf = nn.MSELoss()
    Mtr, Mva, Mme = mask_for(tr, dense), mask_for(va, dense), mask_for(me, dense)
    best, bs = -1e9, None
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(len(tr["y"]))
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS]; opt.zero_grad()
            p = m(tr["feat"][idx], tr["nd"][idx], tr["pr"][idx], Mtr[idx])
            lf(p, tr["y"][idx]).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            vs = skill(m(va["feat"], va["nd"], va["pr"], Mva), va["y"])
        if vs > best: best, bs = vs, copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        pme = m(me["feat"], me["nd"], me["pr"], Mme)
    sk = skill(pme, me["y"])
    # 1-hop reachability stratification: skill for d(0,B) <= L vs > L
    d = me["d0B"]; y = me["y"].numpy(); pr_ = pme.numpy()
    def sk_sub(mask):
        if mask.sum() < 5: return float("nan"), int(mask.sum())
        yy, pp = y[mask], pr_[mask]
        return 1.0 - ((pp - yy) ** 2).mean() / (yy.var() + 1e-12), int(mask.sum())
    near, n_near = sk_sub(d <= L); far, n_far = sk_sub(d > L)
    torch.save({"state": bs, "task": task, "variant": variant, "val_skill": best,
                "cfg": dict(N=N, Kw=Kw, CV=CV, H=H, HEADS=HEADS, L=L)},
               CKPT / f"{task}__{variant.replace('-', '')}.pt")
    return dict(task=task, variant=variant, skill=sk, val=best,
                sk_near=near, n_near=n_near, sk_far=far, n_far=n_far)

# ---- run --------------------------------------------------------------------------------
if __name__ == "__main__":
    print(("SMOKE " if SMOKE else "") + f"train dense/1-hop GT on 3 tasks | N={N} L={L} H={H} heads={HEADS} Kw={Kw}")
    data = {}
    for t in TASKS:
        tr, st = gen(t, G_TR, seed=1)
        va, _ = gen(t, G_VA, ystat=st, seed=2)
        me, _ = gen(t, G_ME, ystat=st, seed=3)
        data[t] = (tr, va, me)
        torch.save({k: me[k] for k in me}, CKPT / f"eval_{t}.pt")
        dd = me["d0B"]
        print(f"[{t}] d(0,B) dist: min={dd.min()} med={int(np.median(dd))} max={dd.max()} "
              f"| frac d<=L({L})={np.mean(dd<=L):.2f} frac d>L={np.mean(dd>L):.2f}")
    np.savez(CKPT / "task_mlps.npz",
             **{f"g_sem_{i}": a for i, a in enumerate(G_SEM)}, **{f"g_str_{i}": a for i, a in enumerate(G_STR)},
             **{f"g_semret_{i}": a for i, a in enumerate(G_SEMRET)}, **{f"g_strret_{i}": a for i, a in enumerate(G_STRRET)},
             w_str=W_STR)
    rows = []
    t0 = time.time()
    for t in TASKS:
        for v in VARIANTS:
            r = train(t, v, *data[t]); rows.append(r)
            print(f"  [{t:10s}/{v:5s}] skill={r['skill']:.3f} (val {r['val']:.3f}) "
                  f"| near(d<=L) {r['sk_near']:.3f} n={r['n_near']} | far(d>L) {r['sk_far']:.3f} n={r['n_far']}",
                  flush=True)
    rows_ser = [{k: (float(v) if isinstance(v, (np.floating, float)) else
                     int(v) if isinstance(v, (np.integer,)) else v) for k, v in r.items()} for r in rows]
    json.dump(rows_ser, open(CKPT / "train_summary.json", "w"), indent=2)
    print(f"\ndone in {time.time()-t0:.0f}s. checkpoints in {CKPT}")
    print("\nSUMMARY  skill (R^2), higher=better")
    print(f"  {'task':12s}{'dense':>9s}{'1-hop':>9s}{'gap':>8s}")
    by = {(r['task'], r['variant']): r for r in rows}
    for t in TASKS:
        de, oh = by[(t, 'dense')]['skill'], by[(t, '1-hop')]['skill']
        print(f"  {t:12s}{de:9.3f}{oh:9.3f}{de-oh:8.3f}")
