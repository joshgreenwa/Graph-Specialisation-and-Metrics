"""Test measures of BENEFICIAL USAGE on the trained k-hop models.

The recall task gives an exact per-graph ground truth: the MATCHED node (whose value
IS the label) is the only beneficial node; all others are distractors. Node contents
are i.i.d., so an on-manifold resample is just a fresh draw -- the donor-matching
problem is absent. So we can fairly ask, per trained model:

  Which per-node beneficial-usage measure best separates the matched node from
  distractors, and does its by-distance profile track held-out task skill (i.e. does
  it collapse for the over-squashed local models exactly where skill collapses)?

Measures (per node j, value channel = the transported payload):
  functional : |d yhat / d value_j|                         (label-free sensitivity)
  B_direct   : E_resample |yhat' - y| - |yhat - y|          (on-manifold loss diff; +=beneficial)
  B_firstord : sign(yhat - y) * (E_resample yhat' - yhat)   (the flawed first-order shortcut)
"""
import math, time, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)

N, CK, H, HEADS, L = 14, 3, 48, 4, 4
KS = [1, 2, 3, 5, N-1]
G_TR, G_VA, G_ME = 10000, 2000, 1500
EPOCHS, BS, LR, WD = 30, 256, 1e-3, 1e-4
VAL_IDX = 1 + 2*CK                                            # value channel index
dist = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :])

def gen(G):
    keys = np.random.randn(G, N, CK).astype(np.float32)
    vals = np.random.randn(G, N, 1).astype(np.float32)
    qpos = np.random.randint(1, N, size=G)
    feat = np.zeros((G, N, 1 + CK + CK + 1), np.float32)
    feat[np.arange(G), 0, 0] = 1.0
    feat[np.arange(G), 0, 1:1+CK] = keys[np.arange(G), qpos]
    feat[:, 1:, 1+CK:1+2*CK] = keys[:, 1:]
    feat[:, 1:, VAL_IDX] = vals[:, 1:, 0]
    y = vals[np.arange(G), qpos, 0].astype(np.float32)
    return torch.tensor(feat), torch.tensor(y), torch.tensor(qpos)

Xtr, ytr, _ = gen(G_TR); Xva, yva, _ = gen(G_VA); Xme, yme, qme = gen(G_ME)
FIN = Xtr.shape[-1]
pe = torch.zeros(N, H); pos = torch.arange(N).float()[:, None]
div = torch.exp(torch.arange(0, H, 2).float() * (-math.log(10000.0)/H))
pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.q=nn.Linear(H,H); s.k=nn.Linear(H,H); s.v=nn.Linear(H,H); s.o=nn.Linear(H,H)
        s.n1=nn.LayerNorm(H); s.n2=nn.LayerNorm(H)
        s.mlp=nn.Sequential(nn.Linear(H,2*H), nn.GELU(), nn.Linear(2*H,H))
    def forward(s,x,mask):
        B=x.size(0); dh=H//HEADS; xn=s.n1(x)
        q=s.q(xn).view(B,N,HEADS,dh).transpose(1,2); k=s.k(xn).view(B,N,HEADS,dh).transpose(1,2)
        v=s.v(xn).view(B,N,HEADS,dh).transpose(1,2)
        a=(q@k.transpose(-2,-1))/math.sqrt(dh)
        a=a.masked_fill(~mask, float("-inf")).softmax(-1)
        x=x+s.o((a@v).transpose(1,2).reshape(B,N,H))
        return x+s.mlp(s.n2(x))

class Model(nn.Module):
    def __init__(s,k):
        super().__init__()
        s.enc=nn.Linear(FIN,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
        s.register_buffer("mask", torch.tensor(dist<=k)[None,None])
    def forward(s,X):
        h=s.enc(X)+pe[None]
        for b in s.blocks: h=b(h,s.mask)
        return s.head(h[:,0]).squeeze(-1)

def val_skill(m):
    m.eval()
    with torch.no_grad(): p=m(Xva)
    return 1-((p-yva)**2).mean().item()/yva.var().item()

def train(k):
    t0=time.time(); m=Model(k); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss()
    best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(G_TR)
        for i in range(0,G_TR,BS):
            idx=perm[i:i+BS]; opt.zero_grad(); lf(m(Xtr[idx]),ytr[idx]).backward(); opt.step()
        vs=val_skill(m)
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); print(f"  k={k:>2} trained best_val={best:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return m

def auroc(scores, labels):
    order=np.argsort(scores); ranks=np.empty(len(scores)); ranks[order]=np.arange(1,len(scores)+1)
    npos=labels.sum(); nneg=len(labels)-npos
    if npos==0 or nneg==0: return float("nan")
    return (ranks[labels==1].sum()-npos*(npos+1)/2)/(npos*nneg)

def measures(m):
    m.eval()
    with torch.no_grad(): yhat=m(Xme)
    r=(yhat-yme)
    # functional via gradient wrt value channel
    Xg=Xme.clone().requires_grad_(True); out=m(Xg).sum(); out.backward()
    func=Xg.grad.detach()[:,:,VAL_IDX].abs().numpy()                 # [G,N]
    # swap-based direct & first-order (resample value channel, K draws)
    K=8; base=Xme.clone(); absr=r.abs()
    Bdir=np.zeros((G_ME,N)); Bfo=np.zeros((G_ME,N))
    for j in range(1,N):
        keep=base[:,j,VAL_IDX].clone(); accL=torch.zeros(G_ME); accY=torch.zeros(G_ME)
        for _ in range(K):
            base[:,j,VAL_IDX]=torch.randn(G_ME)
            with torch.no_grad(): yp=m(base)
            accL+=(yp-yme).abs(); accY+=yp
        base[:,j,VAL_IDX]=keep
        Bdir[:,j]=(accL/K-absr).numpy()
        Bfo[:,j]=(torch.sign(r)*(accY/K-yhat)).numpy()
    return dict(functional=func, B_direct=Bdir, B_firstord=Bfo), yhat.detach()

# ---- run ----
print("training + measuring...")
models={}; meas={}; skill_by_d={}; overall={}
vary=yme.var().item()
for k in KS:
    m=train(k); models[k]=m
    ms,yhat=measures(m); meas[k]=ms
    pred=yhat.numpy()
    overall[k]=1-((pred-yme.numpy())**2).mean()/vary
    sbd={}
    for d in range(1,N):
        sel=(qme.numpy()==d)
        if sel.sum()>0: sbd[d]=1-((pred[sel]-yme.numpy()[sel])**2).mean()/vary
    skill_by_d[k]=sbd

# matched-vs-distractor labels over (graph, node) with j>=1
lab=np.zeros((G_ME,N)); lab[np.arange(G_ME),qme.numpy()]=1.0
mask_j=np.zeros((G_ME,N),bool); mask_j[:,1:]=True
labels=lab[mask_j]

print("\n=== AUROC: does the measure rank the MATCHED node above distractors? ===")
print(f"{'k':>7}{'overall_skill':>15}{'functional':>13}{'B_direct':>12}{'B_firstord':>13}")
auc={}
for k in KS:
    row={}
    for name in ["functional","B_direct","B_firstord"]:
        row[name]=auroc(meas[k][name][mask_j], labels)
    auc[k]=row
    kn="global" if k==N-1 else str(k)
    print(f"{kn:>7}{overall[k]:>15.3f}{row['functional']:>13.3f}{row['B_direct']:>12.3f}{row['B_firstord']:>13.3f}")

# ---- figures ----
ds=list(range(1,N)); colors=plt.cm.viridis(np.linspace(0,0.85,len(KS)))
fig,ax=plt.subplots(2,2,figsize=(13,9))

# (a) AUROC bars per measure per k
names=["functional","B_direct","B_firstord"]; x=np.arange(len(KS)); w=0.26
for i,name in enumerate(names):
    ax[0,0].bar(x+(i-1)*w,[auc[k][name] for k in KS],w,label=name)
ax[0,0].axhline(0.5,color="k",ls=":",lw=1); ax[0,0].set_ylim(0.4,1.0)
ax[0,0].set_xticks(x); ax[0,0].set_xticklabels(["global" if k==N-1 else f"k={k}" for k in KS])
ax[0,0].set_ylabel("AUROC (matched node vs distractors)")
ax[0,0].set_title("(a) Which measure identifies the truly-beneficial node?\n(0.5=chance; higher=better)")
ax[0,0].legend(fontsize=8)

# (b) B_direct at matched node vs distance, per k, with skill overlay
for k,col in zip(KS,colors):
    bd=meas[k]["B_direct"]
    prof=[np.nanmean([bd[g,qme[g]] for g in range(G_ME) if qme[g].item()==d]) for d in ds]
    ax[0,1].plot(ds,prof,"o-",color=col,label=("global" if k==N-1 else f"k={k}"))
ax[0,1].set_xlabel("distance of matched node d"); ax[0,1].set_ylabel("B_direct at matched node")
ax[0,1].set_title("(b) Beneficial usage of the matched node vs distance\n(collapses for local models = over-squashed benefit)")
ax[0,1].legend(fontsize=8); ax[0,1].axhline(0,color="k",lw=0.5)

# (c) skill by distance (for comparison to b)
for k,col in zip(KS,colors):
    sbd=skill_by_d[k]
    ax[1,0].plot(ds,[sbd.get(d,np.nan) for d in ds],"o-",color=col,label=("global" if k==N-1 else f"k={k}"))
ax[1,0].set_xlabel("queried distance d"); ax[1,0].set_ylabel("held-out skill")
ax[1,0].set_title("(c) Held-out task skill vs distance (ground-truth benefit)\nB_direct in (b) should mirror this")
ax[1,0].legend(fontsize=8); ax[1,0].axhline(0,color="k",lw=0.5); ax[1,0].set_ylim(-0.2,1.05)

# (d) spurious usage: functional vs B_direct on DISTRACTOR nodes (per k, mean over near distractors d<=3)
labk=["global" if k==N-1 else f"k={k}" for k in KS]
distr=~lab.astype(bool) & mask_j
near=(dist[0][None,:]<=3)&mask_j[0][None,:]  # near distractors
fmean=[meas[k]["functional"][distr].mean() for k in KS]
bmean=[meas[k]["B_direct"][distr].mean() for k in KS]
xx=np.arange(len(KS))
ax[1,1].bar(xx-0.2,fmean,0.4,label="functional @ distractors")
ax[1,1].bar(xx+0.2,bmean,0.4,label="B_direct @ distractors")
ax[1,1].axhline(0,color="k",lw=0.5); ax[1,1].set_xticks(xx); ax[1,1].set_xticklabels(labk)
ax[1,1].set_title("(d) Spurious usage: functional flags distractors,\nB_direct correctly ~0 on them")
ax[1,1].set_ylabel("mean measure on distractor nodes"); ax[1,1].legend(fontsize=8)

fig.tight_layout(); fig.savefig("fig_beneficial_trained.png",dpi=130); print("\nsaved fig_beneficial_trained.png")
