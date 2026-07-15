"""Tunable off-manifold risk: how do beneficial-carriage estimators degrade?

Reuses the trained dense GT on associative recall (matched node = ground-truth
beneficial). Each node carries a CHECKSUM channel s_j = v_j; the data manifold is
{s_j = v_j}. We wrap the model:  yhat_kappa(X) = GT(X) + kappa * sum_j (v_j - s_j)^2.
On-manifold the penalty is 0 (GT accurate). A swap that changes v_j but not s_j is
OFF-manifold -> penalty fires -> yhat' distorted. kappa is the off-manifold-risk DIAL.

Crucially this makes on/off-manifold a property of the SWAP POLICY (as in reality):
  - marginal / perturb / ig  : change v_j only        -> break checksum -> off-manifold
  - matched-donor            : copy a real (v,s) pair -> consistent     -> on-manifold
GT ignores the checksum channel, so GT(X') depends only on the value -> we forward
once and sweep kappa analytically (penalty added in closed form).
"""
import os, math, time, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE = os.environ.get("SMOKE","0")=="1"

N,CK,H,HEADS,L = 14,3,48,4,4
G_TR,G_VA=(2000,500) if SMOKE else (10000,2000)
G_ME=200 if SMOKE else 800
EPOCHS=4 if SMOKE else 30
BS,LR,WD=256,1e-3,1e-4
K=8 if SMOKE else 32                       # swap samples per estimator
IGSTEPS=4 if SMOKE else 8
KAPPAS=[0.0,0.25,0.5,1.0,2.0,4.0,8.0]
VAL_IDX=1+2*CK; CHK_IDX=1+2*CK+1; FIN_GT=1+2*CK+1   # GT sees [flag,query,key,value]; checksum extra
dist=np.abs(np.arange(N)[:,None]-np.arange(N)[None,:])

def gen(G):
    keys=np.random.randn(G,N,CK).astype(np.float32); vals=np.random.randn(G,N,1).astype(np.float32)
    q=np.random.randint(1,N,size=G); feat=np.zeros((G,N,FIN_GT+1),np.float32)
    feat[np.arange(G),0,0]=1.0; feat[np.arange(G),0,1:1+CK]=keys[np.arange(G),q]
    feat[:,1:,1+CK:1+2*CK]=keys[:,1:]; feat[:,1:,VAL_IDX]=vals[:,1:,0]
    feat[:,:,CHK_IDX]=feat[:,:,VAL_IDX]                 # checksum = value (consistent)
    y=vals[np.arange(G),q,0].astype(np.float32)
    return torch.tensor(feat),torch.tensor(y),torch.tensor(q)

Xtr,ytr,_=gen(G_TR); Xva,yva,_=gen(G_VA); Xme,yme,qme=gen(G_ME)
pe=torch.zeros(N,H); pos=torch.arange(N).float()[:,None]
div=torch.exp(torch.arange(0,H,2).float()*(-math.log(10000.0)/H))
pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Linear(H,H); s.k=nn.Linear(H,H); s.v=nn.Linear(H,H); s.o=nn.Linear(H,H)
        s.n1=nn.LayerNorm(H); s.n2=nn.LayerNorm(H); s.mlp=nn.Sequential(nn.Linear(H,2*H),nn.GELU(),nn.Linear(2*H,H))
    def forward(s,x,mask):
        B=x.size(0); dh=H//HEADS; xn=s.n1(x)
        q=s.q(xn).view(B,N,HEADS,dh).transpose(1,2); k=s.k(xn).view(B,N,HEADS,dh).transpose(1,2); v=s.v(xn).view(B,N,HEADS,dh).transpose(1,2)
        a=((q@k.transpose(-2,-1))/math.sqrt(dh)).masked_fill(~mask,float("-inf")).softmax(-1)
        x=x+s.o((a@v).transpose(1,2).reshape(B,N,H)); return x+s.mlp(s.n2(x))
class GTnet(nn.Module):
    def __init__(s,k):
        super().__init__(); s.enc=nn.Linear(FIN_GT,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
        s.register_buffer("mask",torch.tensor(dist<=k)[None,None])
    def forward(s,X):                                   # X: full features; GT uses first FIN_GT
        h=s.enc(X[...,:FIN_GT])+pe[None]
        for b in s.blocks: h=b(h,s.mask)
        return s.head(h[:,0]).squeeze(-1)

def train():
    t0=time.time(); m=GTnet(N-1); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(G_TR)
        for i in range(0,G_TR,BS):
            idx=perm[i:i+BS]; opt.zero_grad(); lf(m(Xtr[idx]),ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(Xva)-yva)**2).mean().item()/yva.var().item()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); print(f"GT best_val={best:.3f} ({time.time()-t0:.0f}s)",flush=True); return m

def auroc(scores,labels):
    order=np.argsort(scores); ranks=np.empty(len(scores)); ranks[order]=np.arange(1,len(scores)+1)
    npos=labels.sum(); nneg=len(labels)-npos
    return float((ranks[labels==1].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else float("nan")

print(("SMOKE " if SMOKE else "")+"training GT ...")
m=train()
lab=np.zeros((G_ME,N)); lab[np.arange(G_ME),qme.numpy()]=1.0
mask_j=np.zeros((G_ME,N),bool); mask_j[:,1:]=True; labels=lab[mask_j]
vclean=Xme[:,:,VAL_IDX].numpy()                          # [G,N] clean values (=checksum)
with torch.no_grad(): GTclean=m(Xme).numpy()
absr=np.abs(GTclean-yme.numpy())                         # clean penalty=0 so absr kappa-independent
yv=yme.numpy()

def gt_pool(newval_fn):
    """GT(X') and the swapped value, per (G,N,K). newval_fn(j,k)->[G]."""
    GTo=np.zeros((G_ME,N,K),np.float32); vpr=np.zeros((G_ME,N,K),np.float32)
    with torch.no_grad():
        for j in range(1,N):
            for kk in range(K):
                nv=newval_fn(j,kk); X=Xme.clone(); X[:,j,VAL_IDX]=nv
                GTo[:,j,kk]=m(X).numpy(); vpr[:,j,kk]=nv.numpy()
    return GTo,vpr

zr=np.random.randn(G_ME,N,K).astype(np.float32)          # resample draws (marginal & donor share GT_out)
dp=(0.3*np.random.randn(G_ME,N,K)).astype(np.float32)    # small perturbation deltas
GTo_res,vpr_res=gt_pool(lambda j,kk: torch.tensor(zr[:,j,kk]))
GTo_prt,vpr_prt=gt_pool(lambda j,kk: Xme[:,j,VAL_IDX]+torch.tensor(dp[:,j,kk]))
pen_res=(vpr_res-vclean[:,:,None])**2                    # marginal breaks checksum
pen_prt=(vpr_prt-vclean[:,:,None])**2                    # = dp^2
# donor: checksum updated with value -> penalty 0 (GT_out identical to resample pool)

def ig_kappa(kappa):
    ben=np.zeros((G_ME,N),np.float32)
    for j in range(1,N):
        v=Xme[:,j,VAL_IDX]; chk=Xme[:,j,CHK_IDX]; acc=torch.zeros(G_ME)
        for s in range(IGSTEPS):
            al=(s+0.5)/IGSTEPS; val=(al*v).detach().requires_grad_(True)   # baseline 0 -> alpha*v
            X=Xme.clone(); X[:,j,VAL_IDX]=val
            yk=m(X)+kappa*((val-chk)**2)                                    # only node j perturbed
            loss=(yk-yme).abs().sum(); g,=torch.autograd.grad(loss,val); acc+=g.detach()
        ben[:,j]=(-(v*acc/IGSTEPS)).numpy()
    return ben

names=["firstord","direct_marginal","direct_donor","perturb_small","ig"]
res={n:[] for n in names}; departure={}
for kap in KAPPAS:
    ym_res=GTo_res+kap*pen_res                            # marginal predictions
    ym_don=GTo_res                                        # donor: no penalty
    ym_prt=GTo_prt+kap*pen_prt
    Bm=np.abs(ym_res-yv[:,None,None]).mean(2)-absr[:,None]
    Bd=np.abs(ym_don-yv[:,None,None]).mean(2)-absr[:,None]
    Bp=np.abs(ym_prt-yv[:,None,None]).mean(2)-absr[:,None]
    Fo=np.sign(GTclean-yv)[:,None]*(ym_res.mean(2)-GTclean[:,None])
    Bi=ig_kappa(kap)
    res["firstord"].append(auroc(Fo[mask_j],labels))
    res["direct_marginal"].append(auroc(Bm[mask_j],labels))
    res["direct_donor"].append(auroc(Bd[mask_j],labels))
    res["perturb_small"].append(auroc(Bp[mask_j],labels))
    res["ig"].append(auroc(Bi[mask_j],labels))
    print(f"  kappa={kap:>4}  "+"  ".join(f"{n}:{res[n][-1]:.3f}" for n in names),flush=True)

# manifold-departure diagnostic: mean added |penalty shift| at kappa=1
departure={"direct_marginal":float((1.0*pen_res).mean()),
           "direct_donor":0.0,
           "perturb_small":float((1.0*pen_prt).mean()),
           "ig":float((1.0*((0.5*Xme[:,1:,VAL_IDX]-Xme[:,1:,CHK_IDX])**2)).mean())}

# ---- figure ----
cmap={"firstord":"#d62728","direct_marginal":"#1f77b4","direct_donor":"#2ca02c","perturb_small":"#9467bd","ig":"#ff7f0e"}
fig,ax=plt.subplots(1,2,figsize=(13,5))
for n in names:
    ax[0].plot(KAPPAS,res[n],"o-",color=cmap[n],label=n)
ax[0].axhline(0.5,color="k",ls="--",lw=0.7); ax[0].set_xlabel("kappa  (off-manifold risk dial)")
ax[0].set_ylabel("AUROC (matched vs distractors)"); ax[0].set_ylim(0.45,1.02); ax[0].grid(alpha=0.3)
ax[0].set_title("Beneficial-carriage estimators vs off-manifold risk\n(donor stays on-manifold; the rest break the checksum)")
ax[0].legend(fontsize=9)
labs=["direct_marginal","perturb_small","ig","direct_donor"]
ax[1].bar(range(len(labs)),[departure[l] for l in labs],color=[cmap[l] for l in labs])
ax[1].set_xticks(range(len(labs))); ax[1].set_xticklabels(labs,rotation=20)
ax[1].set_ylabel("mean off-manifold penalty triggered (@kappa=1)")
ax[1].set_title("How far each estimator pushes off-manifold\n(donor=0; small-perturb << marginal)")
fig.tight_layout(); fig.savefig("fig_offmanifold.png",dpi=130); print("saved fig_offmanifold.png")
