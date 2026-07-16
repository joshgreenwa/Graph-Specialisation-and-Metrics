"""Structural carriage v2: functional<->beneficial agreement + NEUTRAL discrimination.

Node-RRWP payload task with a planted spurious source. Each graph has causal node B and
spurious node Z. Label:  y_train = g_B(rrwp[B]) + lam * g_Z(rrwp[Z]);  y_test = g_B(rrwp[B]).
So the trained model READS Z (functional>0) but Z is task-irrelevant at test (beneficial~0).
Classes: B (causal), Z (neutral/spurious), others (irrelevant).

Two localisation regimes: 'flagged' (B,Z pointed at) and 'recall' (B,Z found by key-match).
Estimators over node_rrwp[j]: functional (grad), first_order (cheap loss-projected),
direct_mean (K swaps), matched_donor (K swaps, degree-matched real blocks).

Evidence sought:
  (1) agreement: functional predicts beneficial on B/others, DIVERGES on Z (compute-sharing
      works for localisation; a few swaps needed to prune neutral).
  (2) neutral: correct beneficial -> AUROC(B vs others) high, AUROC(Z vs others)~0.5 (Z looks
      irrelevant); functional/first_order wrongly flag Z.
  (3) holds in the non-flagged 'recall' regime too.
"""
import os, math, time, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"

N,Kw,CK,H,HEADS,L=10,6,3,40,4,3
G_TR,G_VA=(2000,500) if SMOKE else (8000,1500)
G_ME=200 if SMOKE else 800
EPOCHS=4 if SMOKE else 45
BS,LR,WD=256,1e-3,1e-4
LAM=0.5; dh=H//HEADS

def rand_graph_adj(n):
    A=np.zeros((n,n),np.float32)
    for i in range(1,n): p=np.random.randint(0,i); A[i,p]=A[p,i]=1.0
    for _ in range(np.random.randint(0,n)):
        a,b=np.random.randint(0,n),np.random.randint(0,n)
        if a!=b: A[a,b]=A[b,a]=1.0
    return A
def node_rrwp(A):
    G=A.shape[0]; I=torch.eye(N).expand(G,N,N); Asl=A+I; M=Asl/Asl.sum(-1,keepdim=True)
    outs=[torch.ones(G,N)]; cur=I.clone()
    for _ in range(1,Kw): cur=cur@M; outs.append(torch.diagonal(cur,dim1=1,dim2=2))
    return torch.stack(outs,-1)  # [G,N,K]
def mk_mlp():
    return (np.random.randn(Kw,8).astype(np.float32)/np.sqrt(Kw), np.random.randn(8).astype(np.float32),
            np.random.randn(8,1).astype(np.float32)/np.sqrt(8))
def apply_g(blk,g): W1,b1,W2=g; return (np.tanh(blk@W1+b1)@W2)[...,0]
gB=mk_mlp(); gZ=mk_mlp()

def gen(G,regime,split,ystat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); At=torch.tensor(A); nd=node_rrwp(At)
    B=np.random.randint(1,N,size=G); Z=np.array([np.random.choice([k for k in range(1,N) if k!=B[g]]) for g in range(G)])
    yB=apply_g(nd.numpy()[np.arange(G),B],gB); yZ=apply_g(nd.numpy()[np.arange(G),Z],gZ)
    y=yB+(LAM*yZ if split=="train" else 0.0)
    if ystat is None: ystat=(y.mean(),y.std()+1e-6)
    y=(y-ystat[0])/ystat[1]
    deg=A.sum(-1).astype(int)
    if regime=="flagged":
        feat=np.zeros((G,N,3),np.float32); feat[:,0,0]=1.0
        feat[np.arange(G),B,1]=1.0; feat[np.arange(G),Z,2]=1.0
    else:  # recall: node keys + queries for B and Z at readout
        keys=np.random.randn(G,N,CK).astype(np.float32)
        feat=np.zeros((G,N,1+3*CK),np.float32); feat[:,0,0]=1.0
        feat[np.arange(G),0,1:1+CK]=keys[np.arange(G),B]; feat[np.arange(G),0,1+CK:1+2*CK]=keys[np.arange(G),Z]
        feat[:,1:,1+2*CK:1+3*CK]=keys[:,1:]
    return dict(nd=nd,feat=torch.tensor(feat),y=torch.tensor(y.astype(np.float32)),B=B,Z=Z,deg=deg),ystat

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Linear(H,H); s.k=nn.Linear(H,H); s.v=nn.Linear(H,H); s.o=nn.Linear(H,H)
        s.n1=nn.LayerNorm(H); s.n2=nn.LayerNorm(H); s.mlp=nn.Sequential(nn.Linear(H,2*H),nn.GELU(),nn.Linear(2*H,H))
    def forward(s,x):
        Bsz=x.size(0); xn=s.n1(x)
        q=s.q(xn).view(Bsz,N,HEADS,dh).transpose(1,2); k=s.k(xn).view(Bsz,N,HEADS,dh).transpose(1,2); v=s.v(xn).view(Bsz,N,HEADS,dh).transpose(1,2)
        a=((q@k.transpose(-2,-1))/math.sqrt(dh)).softmax(-1)
        x=x+s.o((a@v).transpose(1,2).reshape(Bsz,N,H)); return x+s.mlp(s.n2(x))
class Net(nn.Module):
    def __init__(s,fin):
        super().__init__(); s.enc=nn.Linear(fin+Kw,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
    def forward(s,feat,nd):
        x=s.enc(torch.cat([feat,nd],-1))
        for b in s.blocks: x=b(x)
        return s.head(x[:,0]).squeeze(-1)

def train(tr,va,fin):
    t0=time.time(); m=Net(fin); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(tr["y"]))
        for i in range(0,len(perm),BS):
            idx=perm[i:i+BS]; opt.zero_grad(); lf(m(tr["feat"][idx],tr["nd"][idx]),tr["y"][idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(va["feat"],va["nd"])-va["y"])**2).mean().item()/va["y"].var().item()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); return m,best,time.time()-t0
def auroc(sc,lb):
    o=np.argsort(sc); r=np.empty(len(sc)); r[o]=np.arange(1,len(sc)+1); p=lb.sum(); n=len(lb)-p
    return float((r[lb==1].sum()-p*(p+1)/2)/(p*n)) if p and n else float("nan")

def carriage(regime):
    fin=3 if regime=="flagged" else 1+3*CK
    tr,st=gen(G_TR,regime,"train"); va,_=gen(G_VA,regime,"train",st); me,_=gen(G_ME,regime,"test",st)
    m,vskill,dt=train(tr,va,fin); m.eval()
    with torch.no_grad(): yh=m(me["feat"],me["nd"]).numpy()
    yv=me["y"].numpy(); r=yh-yv; absr=np.abs(r); G=len(yv)
    tskill=1-((yh-yv)**2).mean()/me["y"].var().item()
    mean_blk=tr["nd"].mean((0,1))
    npool={}
    for g in range(min(2500,G_TR)):
        for j in range(1,N): npool.setdefault(int(tr["deg"][g,j]),[]).append(tr["nd"][g,j].numpy())
    def fwd(nd2):
        with torch.no_grad(): return m(me["feat"],nd2).numpy()
    def set_blk(j,vec): nd2=me["nd"].clone(); nd2[:,j,:]=vec; return nd2
    # functional (grad)
    ndg=me["nd"].clone().requires_grad_(True); m(me["feat"],ndg).sum().backward(); gn=ndg.grad
    func=np.stack([gn[:,j,:].norm(dim=-1).numpy() for j in range(N)],1)  # [G,N]
    # first-order (loss-projected, one mean-swap)
    fo=np.zeros((G,N))
    for j in range(1,N): fo[:,j]=np.sign(r)*(fwd(set_blk(j,mean_blk.expand(G,Kw)))-yh)
    # direct-mean & matched-donor (K swaps)
    def loss_diff(newfn,K):
        out=np.zeros((G,N))
        for j in range(1,N):
            acc=np.zeros(G)
            for kk in range(K): acc+=np.abs(fwd(set_blk(j,newfn(j,kk)))-yv)
            out[:,j]=acc/K-absr
        return out
    def donor(j,kk):
        keys=me["deg"][:,j]; out=np.zeros((G,Kw),np.float32)
        for key in np.unique(keys):
            cand=npool.get(int(key)) or mean_blk.numpy()[None]; idx=np.where(keys==key)[0]
            out[idx]=np.stack([cand[p] for p in np.random.randint(0,len(cand),size=len(idx))])
        return torch.tensor(out)
    Ksw=12 if not SMOKE else 4
    direct=loss_diff(lambda j,kk: mean_blk.expand(G,Kw),Ksw)
    matched=loss_diff(donor,Ksw)
    # swap-count sweep: nulling Z needs the right (on-manifold) BASELINE, not more swaps
    Ks=[1,2,4,8,16] if not SMOKE else [1,2]
    Bcol=lambda M: M[np.arange(G),me["B"]]; Zcol=lambda M: M[np.arange(G),me["Z"]]
    others_mask=np.ones((G,N),bool); others_mask[:,0]=False; others_mask[np.arange(G),me["B"]]=False; others_mask[np.arange(G),me["Z"]]=False
    def au_vs(col_fn,M): oth=M[others_mask]; return auroc(np.r_[col_fn(M),oth],np.r_[np.ones(G),np.zeros(len(oth))])
    sweep={"mean_Z":[], "matched_Z":[], "matched_B":[]}
    for K in Ks:
        dm=loss_diff(lambda j,kk: mean_blk.expand(G,Kw),K); md=loss_diff(donor,K)
        sweep["mean_Z"].append(au_vs(Zcol,dm)); sweep["matched_Z"].append(au_vs(Zcol,md)); sweep["matched_B"].append(au_vs(Bcol,md))
    # per-estimator AUROC(B vs others) and AUROC(Z vs others)
    def AU(M,which):   # signed score: B/Z should outrank irrelevant 'others' if flagged as used/beneficial
        col=Bcol(M) if which=="B" else Zcol(M); oth=M[others_mask]
        return auroc(np.r_[col,oth], np.r_[np.ones(G),np.zeros(len(oth))])
    ests={"functional":func,"first_order":fo,"direct_mean":direct,"matched_donor":matched}
    res={n:{"B":AU(M,"B"),"Z":AU(M,"Z")} for n,M in ests.items()}
    return dict(regime=regime,tskill=tskill,vskill=vskill,dt=dt,res=res,ests=ests,
                Bidx=me["B"],Zidx=me["Z"],others_mask=others_mask,sweep=(Ks,sweep))

print(("SMOKE " if SMOKE else "")+"running structural carriage v2 ...")
R={}
for reg in ["flagged","recall"]:
    R[reg]=carriage(reg)
    print(f"[{reg}] test_skill={R[reg]['tskill']:.3f} (val {R[reg]['vskill']:.3f}, {R[reg]['dt']:.0f}s)")
    for n,d in R[reg]["res"].items(): print(f"   {n:<13} AUROC(B vs oth)={d['B']:.3f}  AUROC(Z vs oth)={d['Z']:.3f}")

# ---------------- figures ----------------
ests=["functional","first_order","direct_mean","matched_donor"]
ccls={"B":"#2ca02c","Z":"#d62728","other":"#999999"}
fig,ax=plt.subplots(2,2,figsize=(11,8))
# (a) AUROC(B vs oth) and AUROC(Z vs oth) per estimator, flagged
for ci,reg in enumerate(["flagged","recall"]):
    x=np.arange(len(ests)); w=0.38
    ax[0,ci].bar(x-w/2,[R[reg]["res"][e]["B"] for e in ests],w,label="B vs others (want HIGH)",color="#2ca02c")
    ax[0,ci].bar(x+w/2,[R[reg]["res"][e]["Z"] for e in ests],w,label="Z(neutral) vs others (want ~0.5)",color="#d62728")
    ax[0,ci].axhline(0.5,color="k",ls=":",lw=0.8); ax[0,ci].set_ylim(0.4,1.02)
    ax[0,ci].set_xticks(x); ax[0,ci].set_xticklabels(ests,rotation=18,ha="right")
    ax[0,ci].set_title(f"({'a' if ci==0 else 'b'}) {reg} regime")
    if ci==0: ax[0,ci].set_ylabel("AUROC"); ax[0,ci].legend(fontsize=8)
# (c) agreement scatter (flagged): direct beneficial vs functional & first_order, colored by class
reg="flagged"; Rr=R[reg]; G=len(Rr["Bidx"]); om=Rr["others_mask"]
def classvals(M):
    b=M[np.arange(G),Rr["Bidx"]]; z=M[np.arange(G),Rr["Zidx"]]; o=M[om]
    return b,z,o
db,dz,do=classvals(Rr["ests"]["matched_donor"]); fb,fz,fo_=classvals(Rr["ests"]["functional"])
oidx=np.random.choice(len(do),size=min(1500,len(do)),replace=False)
ax[1,0].scatter(fo_[oidx],do[oidx],s=6,c=ccls["other"],alpha=0.4,label="other")
ax[1,0].scatter(fz,dz,s=10,c=ccls["Z"],alpha=0.6,label="Z (neutral)")
ax[1,0].scatter(fb,db,s=10,c=ccls["B"],alpha=0.6,label="B (causal)")
ax[1,0].set_xlabel("functional carriage (grad norm)"); ax[1,0].set_ylabel("beneficial (matched-donor)")
ax[1,0].set_title("(c) Functional vs beneficial carriage, per source (flagged)")
ax[1,0].legend(fontsize=8)
# (d) sweep: nulling Z needs the right (on-manifold) baseline, not more swaps
Ks,sw=Rr["sweep"]
ax[1,1].plot(Ks,sw["matched_B"],"o-",color="#2ca02c",label="matched-donor: B vs others")
ax[1,1].plot(Ks,sw["matched_Z"],"o-",color="#9467bd",label="matched-donor: Z vs others (->0.5 good)")
ax[1,1].plot(Ks,sw["mean_Z"],"s--",color="#d62728",label="mean-baseline: Z vs others (biased)")
ax[1,1].axhline(0.5,color="k",ls=":",lw=0.8); ax[1,1].set_xscale("log",base=2)
ax[1,1].set_xlabel("# swaps K"); ax[1,1].set_ylabel("AUROC"); ax[1,1].set_ylim(0.2,1.02)
ax[1,1].set_title("(d) AUROC vs number of swaps K, by baseline")
ax[1,1].legend(fontsize=7.5)
fig.suptitle("Structural carriage estimators: AUROC(B vs others) and AUROC(Z vs others) for causal B and spurious Z")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig("fig_structural_v2.png"); print("\nsaved fig_structural_v2.png")
