"""Improved practical estimators of beneficial carriage, tested on trained models.

Models: global (dense GT, well-fit) and k=5 (natural error). Ground truth per graph:
the matched node is the only beneficial node -> AUROC(matched vs distractors) is the
score. direct_iid @ K=256 is the on-manifold reference (contents are i.i.d.).

Estimators (positive=beneficial, per (graph,node), value channel), swept over sample size:
  firstord   sign(r)*(E ynew - yhat)                     first-order (current shortcut)
  direct_iid E|ynew - y| - |yhat - y|, iid resamples     MC direct loss-diff
  direct_crn same, common random numbers across nodes    variance-reduced ranking
  perturb    E|y(v+delta) - y| - |yhat - y|, delta~N(0,1) finite-perturbation family
  ig         single-feature IG of loss, baseline v->0, S steps
  eg         expected gradients: IG averaged over M data baselines
"""
import os, math, time, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE = os.environ.get("SMOKE", "0") == "1"

N, CK, H, HEADS, L = 14, 3, 48, 4, 4
KMODELS = [13, 5]                                            # global(GT) and k=5
G_TR, G_VA = (2000, 500) if SMOKE else (10000, 2000)
G_ME = 200 if SMOKE else 800
EPOCHS, BS, LR, WD = (4, 256, 1e-3, 1e-4) if SMOKE else (30, 256, 1e-3, 1e-4)
KMAX = 8 if SMOKE else 256
S_SIZES = [1,2,4,8] if SMOKE else [1,2,4,8,16,32,64,128,256]  # swap sample sizes
IG_STEPS = [1,2,4] if SMOKE else [1,2,4,8,16,32]             # IG path steps
EG_BASE = [1,2,4] if SMOKE else [1,2,4,8,16,32]              # EG #baselines (S_eg fixed)
S_EG = 4
VAL_IDX = 1 + 2*CK
dist = np.abs(np.arange(N)[:,None]-np.arange(N)[None,:])

def gen(G):
    keys=np.random.randn(G,N,CK).astype(np.float32); vals=np.random.randn(G,N,1).astype(np.float32)
    q=np.random.randint(1,N,size=G); feat=np.zeros((G,N,1+2*CK+1),np.float32)
    feat[np.arange(G),0,0]=1.0; feat[np.arange(G),0,1:1+CK]=keys[np.arange(G),q]
    feat[:,1:,1+CK:1+2*CK]=keys[:,1:]; feat[:,1:,VAL_IDX]=vals[:,1:,0]
    y=vals[np.arange(G),q,0].astype(np.float32)
    return torch.tensor(feat),torch.tensor(y),torch.tensor(q)

Xtr,ytr,_=gen(G_TR); Xva,yva,_=gen(G_VA); Xme,yme,qme=gen(G_ME)
FIN=Xtr.shape[-1]
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
class Model(nn.Module):
    def __init__(s,k):
        super().__init__(); s.enc=nn.Linear(FIN,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
        s.register_buffer("mask",torch.tensor(dist<=k)[None,None])
    def forward(s,X):
        h=s.enc(X)+pe[None]
        for b in s.blocks: h=b(h,s.mask)
        return s.head(h[:,0]).squeeze(-1)

def train(k):
    t0=time.time(); m=Model(k); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(G_TR)
        for i in range(0,G_TR,BS):
            idx=perm[i:i+BS]; opt.zero_grad(); lf(m(Xtr[idx]),ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(Xva)-yva)**2).mean().item()/yva.var().item()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); print(f"  k={k} best_val={best:.3f} ({time.time()-t0:.0f}s)",flush=True); return m

def auroc(scores,labels):
    order=np.argsort(scores); ranks=np.empty(len(scores)); ranks[order]=np.arange(1,len(scores)+1)
    npos=labels.sum(); nneg=len(labels)-npos
    return float((ranks[labels==1].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else float("nan")

# ground-truth labels over (graph, node>=1)
lab=np.zeros((G_ME,N)); lab[np.arange(G_ME),qme.numpy()]=1.0
mask_j=np.zeros((G_ME,N),bool); mask_j[:,1:]=True; labels=lab[mask_j]

def swap_pool(m,newvals_fn):
    """newvals_fn(j,k)->[G] value for node j draw k. Returns yp pool [G,N,KMAX]."""
    yp=np.zeros((G_ME,N,KMAX),np.float32)
    with torch.no_grad():
        for j in range(1,N):
            for kk in range(KMAX):
                X=Xme.clone(); X[:,j,VAL_IDX]=newvals_fn(j,kk); yp[:,j,kk]=m(X).numpy()
    return yp

def ig_attr(m,baseline_fn,steps):
    """single-feature IG benefit per (G,N): benefit = L(baseline)-L(clean). steps>=1.
    baseline_fn(j)->[G] baseline value for node j (0 for ig, random for eg)."""
    ben=np.zeros((G_ME,N),np.float32); yv=yme
    for j in range(1,N):
        base=baseline_fn(j); v=Xme[:,j,VAL_IDX]
        acc=torch.zeros(G_ME)
        for s in range(steps):
            alpha=(s+0.5)/steps
            val=(base+alpha*(v-base)).detach().requires_grad_(True)
            X=Xme.clone(); X[:,j,VAL_IDX]=val
            yhat=m(X); loss=(yhat-yv).abs().sum()
            g,=torch.autograd.grad(loss,val)
            acc+=g.detach()
        ig=(v-base)*acc/steps                       # = L(clean)-L(baseline)
        ben[:,j]=(-ig).numpy()                       # benefit = L(baseline)-L(clean)
    return ben

# ---- run ----
print(("SMOKE " if SMOKE else "")+"training GT + k=5 ...")
res={}   # model -> dict(estimator -> {size: auroc}), plus overall skill
for k in KMODELS:
    m=train(k)
    with torch.no_grad(): yhat=m(Xme); r=(yhat-yme); absr=r.abs()
    yhat=yhat.numpy(); rn=r.numpy(); absr_n=absr.numpy(); yv=yme.numpy()
    skill=1-((yhat-yv)**2).mean()/yme.var().item()

    z_iid=np.random.randn(G_ME,N,KMAX).astype(np.float32)       # per-node draws
    z_shared=np.random.randn(G_ME,KMAX).astype(np.float32)      # shared across nodes (CRN)
    d_pert=np.random.randn(G_ME,N,KMAX).astype(np.float32)      # perturbation deltas
    yp_iid=swap_pool(m,lambda j,kk: torch.tensor(z_iid[:,j,kk]))
    yp_crn=swap_pool(m,lambda j,kk: torch.tensor(z_shared[:,kk]))
    yp_pert=swap_pool(m,lambda j,kk: Xme[:,j,VAL_IDX]+torch.tensor(d_pert[:,j,kk]))

    est={n:{} for n in ["firstord","direct_iid","direct_crn","perturb","ig","eg"]}
    for K in S_SIZES:
        Li=np.abs(yp_iid[:,:,:K]-yv[:,None,None]).mean(2)-absr_n[:,None]
        Lc=np.abs(yp_crn[:,:,:K]-yv[:,None,None]).mean(2)-absr_n[:,None]
        Lp=np.abs(yp_pert[:,:,:K]-yv[:,None,None]).mean(2)-absr_n[:,None]
        Fo=np.sign(rn)[:,None]*(yp_iid[:,:,:K].mean(2)-yhat[:,None])
        est["direct_iid"][K]=auroc(Li[mask_j],labels)
        est["direct_crn"][K]=auroc(Lc[mask_j],labels)
        est["perturb"][K]=auroc(Lp[mask_j],labels)
        est["firstord"][K]=auroc(Fo[mask_j],labels)
    for S in IG_STEPS:
        b=ig_attr(m,lambda j: torch.zeros(G_ME),S); est["ig"][S]=auroc(b[mask_j],labels)
    for M in EG_BASE:
        acc=np.zeros((G_ME,N),np.float32)
        for _ in range(M):
            bv=torch.tensor(np.random.randn(G_ME).astype(np.float32))
            acc+=ig_attr(m,lambda j: bv,S_EG)
        est["eg"][M]=auroc((acc/M)[mask_j],labels)
    ref=est["direct_iid"][S_SIZES[-1]]
    res[k]=dict(est=est,skill=skill,ref=ref)
    print(f"  k={k} skill={skill:.3f} ref(direct@{S_SIZES[-1]})={ref:.3f} firstord@{S_SIZES[-1]}={est['firstord'][S_SIZES[-1]]:.3f}",flush=True)

# ---- figures ----
names=["firstord","direct_iid","direct_crn","perturb","ig","eg"]
cmap={"firstord":"#d62728","direct_iid":"#1f77b4","direct_crn":"#17becf","perturb":"#9467bd","ig":"#ff7f0e","eg":"#2ca02c"}
fig,ax=plt.subplots(1,2,figsize=(13,5.2),sharey=True)
for ai,k in enumerate(KMODELS):
    est=res[k]["est"]
    for n in names:
        xs=sorted(est[n]); ys=[est[n][x] for x in xs]
        ax[ai].plot(xs,ys,"o-",color=cmap[n],label=n)
    ax[ai].axhline(res[k]["ref"],color="gray",ls=":",lw=1)
    ax[ai].set_xscale("log",base=2); ax[ai].set_xlabel("sample size (K swaps / S steps / M baselines)")
    ttl="global (GT, well-fit)" if k==13 else f"k={k} (natural error)"
    ax[ai].set_title(f"{ttl}  skill={res[k]['skill']:.2f}")
    ax[ai].axhline(0.5,color="k",ls="--",lw=0.7); ax[ai].grid(alpha=0.3)
ax[0].set_ylabel("AUROC (matched node vs distractors)"); ax[0].legend(fontsize=8,loc="lower right")
fig.suptitle("Beneficial-carriage estimators vs sample size. Many-swap expectation (blue/cyan/green) recovers the\n"
             "beneficial node; first-order (red) plateaus low — worst on the well-fit GT. Dotted=direct reference.")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig("fig_estimators.png",dpi=130); print("saved fig_estimators.png")

print("\n=== AUROC summary ===")
for k in KMODELS:
    print(f"\n[{'global GT' if k==13 else f'k={k}'}] skill={res[k]['skill']:.3f}")
    for n in names:
        est=res[k]["est"][n]; xs=sorted(est)
        print(f"  {n:<11} " + "  ".join(f"{x}:{est[x]:.2f}" for x in xs))
