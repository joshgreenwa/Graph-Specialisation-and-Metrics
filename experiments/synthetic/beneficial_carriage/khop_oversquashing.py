"""k-hop oversquashing: does long-range *functional* carriage survive multi-hop compression?

Setup that isolates COMPRESSION from REACHABILITY:
  * Graph = path of N nodes, node 0 is the readout/query node.
  * Every other node j holds a (key_j, value_j). Node 0 holds a query = one node's key.
  * Label = the value of the node whose key matches the query  (associative recall).
  * Model = masked graph transformer. Attention radius per layer = k (the knob).
    A GLOBAL model (k=N-1) retrieves any item in ONE dense hop. A radius-k model must
    RELAY a matched value across ceil(d/k) hops -> capacity/compression bottleneck.

We use a shallow, realistic depth L=4. Reach = k*L, so for each k there are TWO regimes:
  * d <= kL : reachable, but a matched value must survive ceil(d/k) compression rounds
              -> skill decays with over-squashing.
  * d >  kL : unreachable at all -> skill collapses to chance (the hard wall).
Prediction of the RQ: below the wall, skill tracks compression rounds ceil(d/k); GT's
niche is the band k < d <= kL where a single dense hop beats L squeezed ones (and beyond
kL, where the k-hop model cannot reach at all).
"""
import math, time, sys, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0)
dev = "cpu"; torch.set_num_threads(4)

# ---------------- config ----------------
N = 14                 # path length (distances 1..13)
CK = 3                 # key dim (needs precision to match)
H = 48; HEADS = 4; L = 4   # shallow (realistic). reach = k*L: BELOW it = compression, ABOVE = unreachable
KS = [1, 2, 3, 5, N-1]     # attention radius per layer; N-1 = global (GT). walls at kL = 4,8,12,20,inf
G_TR, G_VA, G_TE = 10000, 2000, 8000
EPOCHS, BS, LR, WD = 30, 256, 1e-3, 1e-4
SEEDS = [0, 1]             # average over inits to denoise the skill curves

dist = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])          # path distance
def radius_mask(k):
    m = torch.tensor((dist <= k), dtype=torch.bool)
    return m                                                          # [N,N] allowed

def gen(G):
    keys = np.random.randn(G, N, CK).astype(np.float32)
    vals = np.random.randn(G, N, 1).astype(np.float32)
    qpos = np.random.randint(1, N, size=G)                            # queried node (=distance)
    feat = np.zeros((G, N, 1 + CK + CK + 1), np.float32)              # [flag,query,key,value]
    feat[:, :, 0] = 0.0
    feat[np.arange(G), 0, 0] = 1.0                                    # readout flag on node 0
    feat[np.arange(G), 0, 1:1+CK] = keys[np.arange(G), qpos]          # query = matched key
    feat[:, 1:, 1+CK:1+2*CK] = keys[:, 1:]                            # keys on non-readout nodes
    feat[:, 1:, 1+2*CK] = vals[:, 1:, 0]                              # values
    y = vals[np.arange(G), qpos, 0].astype(np.float32)
    return torch.tensor(feat), torch.tensor(y), torch.tensor(qpos)

Xtr, ytr, qtr = gen(G_TR); Xva, yva, qva = gen(G_VA); Xte, yte, qte = gen(G_TE)
FIN = Xtr.shape[-1]
pe = torch.zeros(N, H)                                                # sinusoidal positional enc
pos = torch.arange(N).float()[:, None]
div = torch.exp(torch.arange(0, H, 2).float() * (-math.log(10000.0) / H))
pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(H, H); self.k = nn.Linear(H, H); self.v = nn.Linear(H, H)
        self.o = nn.Linear(H, H); self.n1 = nn.LayerNorm(H); self.n2 = nn.LayerNorm(H)
        self.mlp = nn.Sequential(nn.Linear(H, 2*H), nn.GELU(), nn.Linear(2*H, H))
    def forward(self, x, mask):
        B = x.size(0); dh = H // HEADS
        xn = self.n1(x)
        q = self.q(xn).view(B, N, HEADS, dh).transpose(1, 2)
        k = self.k(xn).view(B, N, HEADS, dh).transpose(1, 2)
        v = self.v(xn).view(B, N, HEADS, dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        att = att.masked_fill(~mask, float("-inf"))
        att = att.softmax(-1)
        out = (att @ v).transpose(1, 2).reshape(B, N, H)
        x = x + self.o(out)
        x = x + self.mlp(self.n2(x))
        return x

class Model(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.enc = nn.Linear(FIN, H)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.head = nn.Linear(H, 1)
        self.register_buffer("mask", radius_mask(k)[None, None])       # [1,1,N,N]
    def forward(self, X):
        h = self.enc(X) + pe[None]
        for b in self.blocks:
            h = b(h, self.mask)
        return self.head(h[:, 0, 0:H]).squeeze(-1)                     # readout at node 0

def val_skill(m):
    m.eval()
    with torch.no_grad():
        p = m(Xva)
    return 1 - ((p - yva)**2).mean().item() / yva.var().item()

def train(k):
    import copy
    t0 = time.time()
    m = Model(k).to(dev); opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.MSELoss()
    best_vs, best_state = -1e9, None
    for ep in range(EPOCHS):
        m.train(); perm = torch.randperm(G_TR)
        for i in range(0, G_TR, BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            loss = lossf(m(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
        vs = val_skill(m)                                    # early stop on val skill
        if vs > best_vs:
            best_vs = vs; best_state = copy.deepcopy(m.state_dict())
        if ep % 10 == 0 or ep == EPOCHS-1:
            print(f"   k={k} ep{ep:>2} val_skill={vs:.3f} best={best_vs:.3f} ({time.time()-t0:.0f}s)", flush=True)
    m.load_state_dict(best_state)
    return m

def evaluate(m):
    m.eval()
    with torch.no_grad():
        pred = m(Xte)
    vary = yte.var().item()
    skill_by_d = {}
    for d in range(1, N):
        sel = (qte == d)
        if sel.sum() > 0:
            mse = ((pred[sel]-yte[sel])**2).mean().item()
            skill_by_d[d] = 1 - mse/vary
    overall = 1 - ((pred-yte)**2).mean().item()/vary
    # functional carriage by distance: E || d yhat / d feat_j ||
    Xg = Xte[:2000].clone().requires_grad_(True)
    out = m(Xg).sum(); out.backward()
    g = Xg.grad.detach().norm(dim=-1)                                  # [B,N]
    carr = g.mean(0).numpy()                                          # by node position = distance
    return overall, skill_by_d, carr

ds = list(range(1, N))
# ---- run over seeds, average skill/carriage per k ----
agg = {k: {"skill": [], "carr": [], "overall": []} for k in KS}
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    for k in KS:
        m = train(k); overall, sbd, carr = evaluate(m)
        agg[k]["skill"].append(np.array([sbd.get(d, np.nan) for d in ds]))
        agg[k]["carr"].append(carr[1:])                               # skip readout node
        agg[k]["overall"].append(overall)
        print(f"[seed {seed}] k={k:>2} (reach kL={k*L:>3}) overall={overall:.3f}", flush=True)

skill = {k: np.nanmean(np.stack(agg[k]["skill"]), 0) for k in KS}     # [len(ds)]
carrm = {k: np.mean(np.stack(agg[k]["carr"]), 0) for k in KS}
overallm = {k: float(np.mean(agg[k]["overall"])) for k in KS}
print("\nOVERALL skill by k:", {("global" if k==N-1 else k): round(overallm[k],3) for k in KS})

# carriage 'reach' = largest d with carriage >= 25% of its d=1 value
reach = {}
for k in KS:
    c = carrm[k] / (carrm[k][0] + 1e-9)
    idx = np.where(c >= 0.25)[0]
    reach[k] = ds[idx.max()] if len(idx) else 1
print("carriage reach (25% of near) by k:", {("global" if k==N-1 else k): reach[k] for k in KS})

# ---------------- figures ----------------
colors = plt.cm.viridis(np.linspace(0, 0.85, len(KS)))
fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# (a) functional carriage by distance  (HEADLINE: the robust object)
for k, col in zip(KS, colors):
    c = carrm[k] / (carrm[k].max() + 1e-9)
    lbl = "global (GT)" if k == N-1 else f"k={k} (reach kL={k*L})"
    ax[0,0].plot(ds, c, "o-", color=col, label=lbl)
ax[0,0].set_xlabel("source distance d"); ax[0,0].set_ylabel("functional carriage (norm.)")
ax[0,0].set_title("(a) Functional semantic carriage by distance, per k\n"
                  "global stays flat to the horizon; k-hop decays -> squashed by ceil(d/k) relays")
ax[0,0].legend(fontsize=8)

# (b) carriage reach grows ~linearly with k; global = full horizon
kk = [k for k in KS if k != N-1]
ax[0,1].plot(kk, [reach[k] for k in kk], "o-", color="#1f77b4", label="carriage reach (25%)")
ax[0,1].plot(kk, [min(k*L, N-1) for k in kk], "s--", color="gray", label="reachability wall kL")
ax[0,1].axhline(reach[N-1], color="#2ca02c", ls="-", label="global reach (full horizon)")
ax[0,1].set_xlabel("attention radius k"); ax[0,1].set_ylabel("distance")
ax[0,1].set_title("(b) How far content actually carries vs k\n"
                  "carriage reach < reachability wall kL -> compression eats the tail")
ax[0,1].legend(fontsize=8)

# (c) skill vs distance (seed-averaged), reachability walls marked
for k, col in zip(KS, colors):
    lbl = "global (GT)" if k == N-1 else f"k={k}"
    ax[1,0].plot(ds, skill[k], "o-", color=col, label=lbl)
    if k*L < N: ax[1,0].axvline(k*L, color=col, ls=":", lw=1)
ax[1,0].set_xlabel("queried distance d"); ax[1,0].set_ylabel("test skill (1=perfect, 0=mean)")
ax[1,0].set_title(f"(c) Retrieval skill vs distance (L={L}, {len(SEEDS)}-seed mean)\n"
                  "global solves every distance; k-hop only near, then collapses")
ax[1,0].legend(fontsize=8); ax[1,0].axhline(0, color="k", lw=0.5); ax[1,0].set_ylim(-0.3, 1.05)

# (d) GT niche: skill advantage of global over each k, by distance
for k, col in zip(KS, colors):
    if k == N-1: continue
    adv = skill[N-1] - skill[k]
    ax[1,1].plot(ds, adv, "o-", color=col, label=f"global - k={k}")
    if k*L < N: ax[1,1].axvline(k*L, color=col, ls=":", lw=1)
ax[1,1].set_xlabel("queried distance d"); ax[1,1].set_ylabel("skill advantage of GLOBAL")
ax[1,1].set_title("(d) GT niche: single dense hop beats L squeezed ones\n"
                  "advantage grows with d; dotted = where k-hop can't even reach")
ax[1,1].legend(fontsize=8); ax[1,1].axhline(0, color="k", lw=0.5)

fig.tight_layout()
fig.savefig("fig_khop.png", dpi=130)
print("saved fig_khop.png")
