"""Off-manifold risk: full estimator set (policy-not-form) + matched-donor QUALITY sweep.

Extends off_manifold_estimators.py: (a) adds direct_crn, perturb_large, eg to the kappa
sweep to show off-manifold behaviour is set by the intervention POLICY not the estimator
form; (b) sweeps matched-donor QUALITY q in [0,1] (residual checksum mismatch = (1-q) x
full), the middle ground between an immune perfect donor (q=1) and off-manifold marginal
(q=0). Paper-quality figures.
"""
import os, math, time, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style; from paper_style import PALETTE
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE = os.environ.get("SMOKE","0")=="1"

N,CK,H,HEADS,L = 14,3,48,4,4
G_TR,G_VA=(2000,500) if SMOKE else (10000,2000)
G_ME=200 if SMOKE else 800
EPOCHS=4 if SMOKE else 30
BS,LR,WD=256,1e-3,1e-4
K=8 if SMOKE else 32; IGSTEPS=4 if SMOKE else 8; EG_M=4 if SMOKE else 8; EG_S=4
KAPPAS=[0.0,0.25,0.5,1.0,2.0,4.0,8.0]
QS=[0.0,0.2,0.4,0.6,0.8,0.9,1.0]; Q_KAPPAS=[0.5,1.0,2.0,4.0]
VAL_IDX=1+2*CK; CHK_IDX=1+2*CK+1; FIN_GT=1+2*CK+1
dist=np.abs(np.arange(N)[:,None]-np.arange(N)[None,:])

def gen(G):
    keys=np.random.randn(G,N,CK).astype(np.float32); vals=np.random.randn(G,N,1).astype(np.float32)
    q=np.random.randint(1,N,size=G); feat=np.zeros((G,N,FIN_GT+1),np.float32)
    feat[np.arange(G),0,0]=1.0; feat[np.arange(G),0,1:1+CK]=keys[np.arange(G),q]
    feat[:,1:,1+CK:1+2*CK]=keys[:,1:]; feat[:,1:,VAL_IDX]=vals[:,1:,0]
    feat[:,:,CHK_IDX]=feat[:,:,VAL_IDX]
    y=vals[np.arange(G),q,0].astype(np.float32)
    return torch.tensor(feat),torch.tensor(y),torch.tensor(q)

Xtr,ytr,_=gen(G_TR); Xva,yva,_=gen(G_VA); Xme,yme,qme=gen(G_ME)
pe=torch.zeros(N,H); pos=torch.arange(N).float()[:,None]
div=torch.exp(torch.arange(0,H,2).float()*(-math.log(10000.0)/H)); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
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
    def forward(s,X):
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
def auroc(sc,lb):
    o=np.argsort(sc); r=np.empty(len(sc)); r[o]=np.arange(1,len(sc)+1); p=lb.sum(); n=len(lb)-p
    return float((r[lb==1].sum()-p*(p+1)/2)/(p*n)) if p and n else float("nan")

print(("SMOKE " if SMOKE else "")+"training GT ..."); m=train()
lab=np.zeros((G_ME,N)); lab[np.arange(G_ME),qme.numpy()]=1.0
mask_j=np.zeros((G_ME,N),bool); mask_j[:,1:]=True; labels=lab[mask_j]
vclean=Xme[:,:,VAL_IDX].numpy()
with torch.no_grad(): GTclean=m(Xme).numpy()
absr=np.abs(GTclean-yme.numpy()); yv=yme.numpy()

def gt_pool(fn):
    GTo=np.zeros((G_ME,N,K),np.float32); vpr=np.zeros((G_ME,N,K),np.float32)
    with torch.no_grad():
        for j in range(1,N):
            for kk in range(K):
                nv=fn(j,kk); X=Xme.clone(); X[:,j,VAL_IDX]=nv; GTo[:,j,kk]=m(X).numpy(); vpr[:,j,kk]=nv.numpy()
    return GTo,vpr
zr=np.random.randn(G_ME,N,K).astype(np.float32); zc=np.random.randn(G_ME,K).astype(np.float32)
dS=(0.3*np.random.randn(G_ME,N,K)).astype(np.float32); dL=(1.0*np.random.randn(G_ME,N,K)).astype(np.float32)
GTo_res,vpr_res=gt_pool(lambda j,kk: torch.tensor(zr[:,j,kk]))
GTo_crn,vpr_crn=gt_pool(lambda j,kk: torch.tensor(zc[:,kk]))
GTo_pS,vpr_pS=gt_pool(lambda j,kk: Xme[:,j,VAL_IDX]+torch.tensor(dS[:,j,kk]))
GTo_pL,vpr_pL=gt_pool(lambda j,kk: Xme[:,j,VAL_IDX]+torch.tensor(dL[:,j,kk]))
pen_res=(vpr_res-vclean[:,:,None])**2; pen_crn=(vpr_crn-vclean[:,:,None])**2
pen_pS=(vpr_pS-vclean[:,:,None])**2; pen_pL=(vpr_pL-vclean[:,:,None])**2
def Bswap(GTo,pen,kap): return np.abs(GTo+kap*pen-yv[:,None,None]).mean(2)-absr[:,None]
def eg_kappa(kappa):
    ben=np.zeros((G_ME,N),np.float32)
    for j in range(1,N):
        v=Xme[:,j,VAL_IDX]; chk=Xme[:,j,CHK_IDX]; tot=torch.zeros(G_ME)
        for _ in range(EG_M):
            b=torch.randn(G_ME); acc=torch.zeros(G_ME)
            for s in range(EG_S):
                al=(s+0.5)/EG_S; val=(b+al*(v-b)).detach().requires_grad_(True)
                X=Xme.clone(); X[:,j,VAL_IDX]=val; yk=m(X)+kappa*((val-chk)**2)
                g,=torch.autograd.grad((yk-yme).abs().sum(),val); acc+=g.detach()
            tot+=(v-b)*acc/EG_S
        ben[:,j]=(-(tot/EG_M)).numpy()
    return ben
def ig_kappa(kappa):
    ben=np.zeros((G_ME,N),np.float32)
    for j in range(1,N):
        v=Xme[:,j,VAL_IDX]; chk=Xme[:,j,CHK_IDX]; acc=torch.zeros(G_ME)
        for s in range(IGSTEPS):
            al=(s+0.5)/IGSTEPS; val=(al*v).detach().requires_grad_(True)
            X=Xme.clone(); X[:,j,VAL_IDX]=val; yk=m(X)+kappa*((val-chk)**2)
            g,=torch.autograd.grad((yk-yme).abs().sum(),val); acc+=g.detach()
        ben[:,j]=(-(v*acc/IGSTEPS)).numpy()
    return ben

names=["firstord","direct_marginal","direct_crn","direct_donor","perturb_small","perturb_large","ig","eg"]
res={n:[] for n in names}
for kap in KAPPAS:
    ymr=GTo_res+kap*pen_res
    res["firstord"].append(auroc((np.sign(GTclean-yv)[:,None]*(ymr.mean(2)-GTclean[:,None]))[mask_j],labels))
    res["direct_marginal"].append(auroc(Bswap(GTo_res,pen_res,kap)[mask_j],labels))
    res["direct_crn"].append(auroc(Bswap(GTo_crn,pen_crn,kap)[mask_j],labels))
    res["direct_donor"].append(auroc(Bswap(GTo_res,pen_res*0,kap)[mask_j],labels))
    res["perturb_small"].append(auroc(Bswap(GTo_pS,pen_pS,kap)[mask_j],labels))
    res["perturb_large"].append(auroc(Bswap(GTo_pL,pen_pL,kap)[mask_j],labels))
    res["ig"].append(auroc(ig_kappa(kap)[mask_j],labels))
    res["eg"].append(auroc(eg_kappa(kap)[mask_j],labels))
    print(f"  kappa={kap:>4}  "+"  ".join(f"{n}:{res[n][-1]:.2f}" for n in names),flush=True)

# donor-quality sweep: penalty kappa*(1-q)^2*(v'-v)^2
donorq={kap:[] for kap in Q_KAPPAS}
for kap in Q_KAPPAS:
    for q in QS:
        B=Bswap(GTo_res,pen_res*((1-q)**2),kap); donorq[kap].append(auroc(B[mask_j],labels))
print("donor-quality sweep done",flush=True)

# ---------- figures ----------
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
for n in names:
    ls="-" if n.startswith(("direct","firstord")) else "--"
    ax[0].plot(KAPPAS,res[n],marker="o",ls=ls,color=PALETTE[n],label=n.replace("_"," "))
ax[0].axhline(0.5,color="k",ls=":",lw=0.8); ax[0].set_ylim(0.45,1.02)
ax[0].set_xlabel(r"off-manifold risk  $\kappa$"); ax[0].set_ylabel("AUROC (matched vs distractors)")
ax[0].set_title("(a) Robustness is set by intervention policy, not estimator form")
ax[0].legend(ncol=2,fontsize=7.5)
for kap in Q_KAPPAS:
    ax[1].plot(QS,donorq[kap],marker="o",label=rf"$\kappa={kap}$")
ax[1].axhline(0.9,color="gray",ls=":",lw=0.8); ax[1].set_ylim(0.45,1.02)
ax[1].set_xlabel(r"matched-donor quality  $q$  (1 = perfect, 0 = marginal)"); ax[1].set_ylabel("AUROC")
ax[1].set_title("(b) How good must donors be? Threshold rises with risk")
ax[1].legend(title="")
fig.tight_layout(); fig.savefig("fig_offmanifold.png"); print("saved fig_offmanifold.png (a) + donor quality (b)")
