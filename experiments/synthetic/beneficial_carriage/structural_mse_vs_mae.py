"""MSE vs MAE for structural beneficial carriage (same swaps, two loss definitions).

Flagged node-RRWP task, causal B + spurious Z (lambda sweep). For each estimator we compute
benefit under BOTH  L=|.|  and  L=(.)^2  from the SAME cached swap predictions:
  direct_mean/matched_donor:  E[L(yhat',y)] - L(yhat,y)
  first_order:  MAE-> sign(r)*E[dy];  MSE-> 2r*E[dy]   (loss-gradient x mean effect)
  functional: grad norm (loss-independent).
Report AUROC(B vs others) [want high] and AUROC(Z vs others) [want ~0.5].
Expectation: MSE's E[dy^2] term de-couples the causal signal from the residual offset ->
STRONGER B; on-manifold matched-donor still nulls Z under both; off-manifold direct_mean
mislabels Z under both.
"""
import os, math, time, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,H,HEADS,L=10,6,40,4,3
G_TR,G_VA=(2000,500) if SMOKE else (8000,1500)
G_ME=200 if SMOKE else 900
EPOCHS=4 if SMOKE else 45
BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; Ksw=4 if SMOKE else 16
LAMS=[0.5] if SMOKE else [0.3,0.5]

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
    return torch.stack(outs,-1)
def mk_mlp(): return (np.random.randn(Kw,8).astype(np.float32)/np.sqrt(Kw),np.random.randn(8).astype(np.float32),np.random.randn(8,1).astype(np.float32)/np.sqrt(8))
def apply_g(blk,g): W1,b1,W2=g; return (np.tanh(blk@W1+b1)@W2)[...,0]
gB=mk_mlp(); gZ=mk_mlp()
def gen(G,lam,split,ystat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); nd=node_rrwp(torch.tensor(A))
    B=np.random.randint(1,N,size=G); Z=np.array([np.random.choice([k for k in range(1,N) if k!=B[g]]) for g in range(G)])
    y=apply_g(nd.numpy()[np.arange(G),B],gB)+(lam*apply_g(nd.numpy()[np.arange(G),Z],gZ) if split=="train" else 0.0)
    if ystat is None: ystat=(y.mean(),y.std()+1e-6)
    y=(y-ystat[0])/ystat[1]
    feat=np.zeros((G,N,3),np.float32); feat[:,0,0]=1.0; feat[np.arange(G),B,1]=1.0; feat[np.arange(G),Z,2]=1.0
    return dict(nd=nd,feat=torch.tensor(feat),y=torch.tensor(y.astype(np.float32)),B=B,Z=Z,deg=A.sum(-1).astype(int)),ystat
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Linear(H,H);s.k=nn.Linear(H,H);s.v=nn.Linear(H,H);s.o=nn.Linear(H,H)
        s.n1=nn.LayerNorm(H);s.n2=nn.LayerNorm(H);s.mlp=nn.Sequential(nn.Linear(H,2*H),nn.GELU(),nn.Linear(2*H,H))
    def forward(s,x):
        Bsz=x.size(0);xn=s.n1(x)
        q=s.q(xn).view(Bsz,N,HEADS,dh).transpose(1,2);k=s.k(xn).view(Bsz,N,HEADS,dh).transpose(1,2);v=s.v(xn).view(Bsz,N,HEADS,dh).transpose(1,2)
        a=((q@k.transpose(-2,-1))/math.sqrt(dh)).softmax(-1)
        x=x+s.o((a@v).transpose(1,2).reshape(Bsz,N,H)); return x+s.mlp(s.n2(x))
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=nn.Linear(3+Kw,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
    def forward(s,feat,nd):
        x=s.enc(torch.cat([feat,nd],-1))
        for b in s.blocks: x=b(x)
        return s.head(x[:,0]).squeeze(-1)
def train(tr,va):
    m=Net(); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(tr["y"]))
        for i in range(0,len(perm),BS):
            idx=perm[i:i+BS]; opt.zero_grad(); lf(m(tr["feat"][idx],tr["nd"][idx]),tr["y"][idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(va["feat"],va["nd"])-va["y"])**2).mean().item()/va["y"].var().item()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); return m,best
def auroc(sc,lb):
    o=np.argsort(sc); r=np.empty(len(sc)); r[o]=np.arange(1,len(sc)+1); p=lb.sum(); n=len(lb)-p
    return float((r[lb==1].sum()-p*(p+1)/2)/(p*n)) if p and n else float("nan")

def analyse(lam):
    tr,st=gen(G_TR,lam,"train"); va,_=gen(G_VA,lam,"train",st); me,_=gen(G_ME,lam,"test",st)
    m,vs=train(tr,va); m.eval()
    with torch.no_grad(): yh=m(me["feat"],me["nd"]).numpy()
    yv=me["y"].numpy(); r=yh-yv; abs_r=np.abs(r); sq_r=r**2; G=len(yv); mean_blk=tr["nd"].mean((0,1))
    npool={}
    for g in range(min(2500,G_TR)):
        for j in range(1,N): npool.setdefault(int(tr["deg"][g,j]),[]).append(tr["nd"][g,j].numpy())
    def fwd(nd2):
        with torch.no_grad(): return m(me["feat"],nd2).numpy()
    def setb(j,vec): nd2=me["nd"].clone(); nd2[:,j,:]=vec; return nd2
    # functional (loss-independent)
    ndg=me["nd"].clone().requires_grad_(True); m(me["feat"],ndg).sum().backward(); gn=ndg.grad
    func=np.stack([gn[:,j,:].norm(dim=-1).numpy() for j in range(N)],1)
    # both-loss direct-swap benefit
    def loss_diff2(fn,K):
        mae=np.zeros((G,N)); mse=np.zeros((G,N))
        for j in range(1,N):
            aa=np.zeros(G); ss=np.zeros(G)
            for kk in range(K): yp=fwd(setb(j,fn(j,kk))); aa+=np.abs(yp-yv); ss+=(yp-yv)**2
            mae[:,j]=aa/K-abs_r; mse[:,j]=ss/K-sq_r
        return mae,mse
    def donor(j,kk):
        keys=me["deg"][:,j]; out=np.zeros((G,Kw),np.float32)
        for key in np.unique(keys):
            cand=npool.get(int(key)) or mean_blk.numpy()[None]; idx=np.where(keys==key)[0]
            out[idx]=np.stack([cand[p] for p in np.random.randint(0,len(cand),size=len(idx))])
        return torch.tensor(out)
    dm_mae,dm_mse=loss_diff2(lambda j,kk: mean_blk.expand(G,Kw),Ksw)
    md_mae,md_mse=loss_diff2(donor,Ksw)
    # first-order (loss-projected mean effect): MAE-> sign(r)*dy ; MSE-> 2r*dy
    fo_mae=np.zeros((G,N)); fo_mse=np.zeros((G,N))
    for j in range(1,N):
        dy=fwd(setb(j,mean_blk.expand(G,Kw)))-yh; fo_mae[:,j]=np.sign(r)*dy; fo_mse[:,j]=2*r*dy
    om=np.ones((G,N),bool); om[:,0]=False; om[np.arange(G),me["B"]]=False; om[np.arange(G),me["Z"]]=False
    def AU(M,which):
        col=M[np.arange(G),me["B"] if which=="B" else me["Z"]]; oth=M[om]
        return auroc(np.r_[col,oth],np.r_[np.ones(G),np.zeros(len(oth))])
    ts=1-((yh-yv)**2).mean()/me["y"].var().item()
    rows={"functional":(func,func),"first_order":(fo_mae,fo_mse),"direct_mean":(dm_mae,dm_mse),"matched_donor":(md_mae,md_mse)}
    out={}
    for name,(Mm,Ms) in rows.items():
        out[name]={"MAE":{"B":AU(Mm,"B"),"Z":AU(Mm,"Z")},"MSE":{"B":AU(Ms,"B"),"Z":AU(Ms,"Z")}}
    return ts,out

print(("SMOKE " if SMOKE else "")+"MSE vs MAE structural ...")
R={}
for lam in LAMS:
    ts,out=analyse(lam); R[lam]=out
    print(f"\n lam={lam} (skill {ts:.2f}):")
    for n,d in out.items():
        print(f"   {n:<13} B: MAE={d['MAE']['B']:.2f} MSE={d['MSE']['B']:.2f}  |  Z: MAE={d['MAE']['Z']:.2f} MSE={d['MSE']['Z']:.2f}")

# ---------- figure (use largest lambda = hardest contamination) ----------
lam=LAMS[-1]; out=R[lam]; ests=["functional","first_order","direct_mean","matched_donor"]
fig,ax=plt.subplots(1,2,figsize=(11,4.5)); x=np.arange(len(ests)); w=0.38
for pi,which in enumerate(["B","Z"]):
    ax[pi].bar(x-w/2,[out[e]["MAE"][which] for e in ests],w,label="MAE  |y'-y|",color="#1f77b4")
    ax[pi].bar(x+w/2,[out[e]["MSE"][which] for e in ests],w,label="MSE  (y'-y)^2",color="#ff7f0e")
    ax[pi].axhline(0.5,color="k",ls=":" if which=="B" else "--",lw=0.9)
    ax[pi].set_xticks(x); ax[pi].set_xticklabels(ests,rotation=18,ha="right"); ax[pi].set_ylim(0.2 if which=="Z" else 0.45,1.02)
    ax[pi].set_ylabel(f"AUROC({which} vs others)"+("  (want HIGH)" if which=="B" else "  (want ~0.5)"))
    ax[pi].set_title(f"({'a' if pi==0 else 'b'}) {'Causal B' if which=='B' else 'Neutral Z'}")
    if pi==0: ax[pi].legend(fontsize=8)
fig.suptitle(f"Structural beneficial carriage under MAE vs MSE loss ($\\lambda={lam}$): AUROC for causal B and spurious Z")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("fig_structural_mse.png"); print("\nsaved fig_structural_mse.png")
