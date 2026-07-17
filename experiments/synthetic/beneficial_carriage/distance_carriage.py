"""Distance-dependent functional & beneficial carriage curves (repo-faithful) on the lightweight GTs.

Faithful to src/graph_specialisation_metrics/beneficial_carriage.py:
  focal readout node i* (here node 0 = the model's readout); distance bands d={j: dist(i*,j)=d};
  resample source j's CONTENT on-manifold (matched-environment donor = same degree) and forward-pass:
     F(d) = E[(y_hat' - y_hat)^2]              functional  (label-free: the output moved)
     B(d) = E[ |y_hat' - y| - |y_hat - y| ]    beneficial  (>0 => distance-d content HELPS the task)
Single-source (necessity) resampling; 95% bootstrap CI over graphs. Applied to dense and 1-hop GTs
on both synthetic tasks. Content is iid N(0,1) here, so matched==marginal (no off-manifold risk);
the point is the DISTANCE decay: dense global-attention reads the target at any range (flat), the
1-hop bonded-mask must propagate through layers (decays / collapses past its reach).
"""
import os, math, time, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=10,6,4,32,4,4
G_TR,G_VA=(2000,500) if SMOKE else (6000,1200)
G_ME=200 if SMOKE else 1500
EPOCHS=4 if SMOKE else 30
BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; MAXD=5; NDON=6
MODELS=[("dense","semantic"),("1-hop","semantic"),("dense","structural"),("1-hop","structural")]

def rand_graph_adj(n):
    A=np.zeros((n,n),np.float32)
    for i in range(1,n): p=np.random.randint(0,i); A[i,p]=A[p,i]=1.0
    for _ in range(np.random.randint(0,n)):
        a,b=np.random.randint(0,n),np.random.randint(0,n)
        if a!=b: A[a,b]=A[b,a]=1.0
    return A
def bfs0(A):  # shortest-path hop distance from node 0; inf if unreachable
    d=np.full(N,np.inf,np.float32); d[0]=0; frontier=[0]
    for step in range(1,N):
        nxt=[]
        for u in frontier:
            for v in np.where(A[u]>0)[0]:
                if d[v]==np.inf: d[v]=step; nxt.append(v)
        frontier=nxt
        if not frontier: break
    return d
def rrwp(A):
    G=A.shape[0]; I=torch.eye(N).expand(G,N,N); Asl=A+I; M=Asl/Asl.sum(-1,keepdim=True)
    nd=[torch.ones(G,N)]; pr=[I.clone()]; cur=I.clone()
    for _ in range(1,Kw): cur=cur@M; nd.append(torch.diagonal(cur,dim1=1,dim2=2)); pr.append(cur.clone())
    return torch.stack(nd,-1), torch.stack(pr,-1)
def mk_mlp(d): return (np.random.randn(d,8).astype(np.float32)/np.sqrt(d),np.random.randn(8).astype(np.float32),np.random.randn(8,1).astype(np.float32)/np.sqrt(8))
def apply_g(x,g): W1,b1,W2=g; return (np.tanh(x@W1+b1)@W2)[...,0]
g_sem=mk_mlp(CV); g_str=mk_mlp(Kw)
def gen(task,G,ystat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); At=torch.tensor(A)
    nd,pr=rrwp(At); cont=np.random.randn(G,N,CV).astype(np.float32)
    B=np.random.randint(1,N,size=G)
    y=apply_g(cont[np.arange(G),B],g_sem) if task=="semantic" else apply_g(pr.numpy()[np.arange(G),0,B],g_str)
    if ystat is None: ystat=(y.mean(),y.std()+1e-6)
    y=(y-ystat[0])/ystat[1]
    flags=np.zeros((G,N,2),np.float32); flags[:,0,0]=1.0; flags[np.arange(G),B,1]=1.0
    dist=np.stack([bfs0(A[g]) for g in range(G)])
    return dict(feat=torch.tensor(np.concatenate([flags,cont],-1)),nd=nd,pr=pr,
                y=torch.tensor(y.astype(np.float32)),B=B,deg=A.sum(-1).astype(int),
                mask1=torch.tensor((A+np.eye(N))>0),dist=dist),ystat

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Linear(H,H);s.k=nn.Linear(H,H);s.v=nn.Linear(H,H);s.o=nn.Linear(H,H)
        s.bb=nn.Linear(Kw,HEADS); s.pv=nn.Linear(Kw,H)
        s.n1=nn.LayerNorm(H);s.n2=nn.LayerNorm(H);s.mlp=nn.Sequential(nn.Linear(H,2*H),nn.GELU(),nn.Linear(2*H,H))
    def forward(s,x,pr,mask):
        B=x.size(0);xn=s.n1(x)
        q=s.q(xn).view(B,N,HEADS,dh).transpose(1,2);k=s.k(xn).view(B,N,HEADS,dh).transpose(1,2);v=s.v(xn).view(B,N,HEADS,dh).transpose(1,2)
        a=((q@k.transpose(-2,-1))/math.sqrt(dh)+s.bb(pr).permute(0,3,1,2)).masked_fill(~mask,float("-inf")).softmax(-1)
        cv=a@v
        sv=torch.einsum('bhij,bhijd->bhid',a,s.pv(pr).view(B,N,N,HEADS,dh).permute(0,3,1,2,4))
        ho=cv+sv; x=x+s.o(ho.transpose(1,2).reshape(B,N,H)); x=x+s.mlp(s.n2(x)); return x
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=nn.Linear(2+CV+Kw,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
    def forward(s,feat,nd,pr,mask):
        x=s.enc(torch.cat([feat,nd],-1))
        for b in s.blocks: x=b(x,pr,mask)
        return s.head(x[:,0]).squeeze(-1)

def train(variant,task):
    torch.manual_seed(0); np.random.seed(1 if task=="structural" else 0)
    tr,st=gen(task,G_TR); va,_=gen(task,G_VA,st); me,_=gen(task,G_ME,st)
    dense=(variant=="dense")
    def MK(d): return torch.ones(len(d["y"]),1,N,N,dtype=torch.bool) if dense else d["mask1"][:,None]
    m=Net(); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(tr["y"]))
        for i in range(0,len(perm),BS):
            idx=perm[i:i+BS]; opt.zero_grad()
            lf(m(tr["feat"][idx],tr["nd"][idx],tr["pr"][idx],MK({"y":tr["y"][idx],"mask1":tr["mask1"][idx]})),tr["y"][idx]).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(va["feat"],va["nd"],va["pr"],MK(va))-va["y"])**2).mean().item()/va["y"].var().item()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); m.eval()
    return m,me,MK(me)

def carriage_curves(m,me,mask):
    G=len(me["y"]); y=me["y"].numpy(); deg=me["deg"]; dist=me["dist"]
    with torch.no_grad(): yh=m(me["feat"],me["nd"],me["pr"],mask).numpy()
    base=np.abs(yh-y)
    # matched (degree-signature) content donor bank from the SAME measurement pool
    cpool={}; allc=[]
    for g in range(G):
        for j in range(1,N):
            c=me["feat"][g,j,2:2+CV].numpy(); cpool.setdefault(int(deg[g,j]),[]).append(c); allc.append(c)
    rng=np.random.default_rng(0)
    accF=np.zeros((G,MAXD+1)); accB=np.zeros((G,MAXD+1)); cnt=np.zeros((G,MAXD+1))
    ar=np.arange(G)
    for j in range(1,N):
        keys=deg[:,j]; dd=dist[:,j]
        valid=(dd>=1)&(dd<=MAXD); dv=dd.astype(int)
        for _ in range(NDON):
            newc=np.zeros((G,CV),np.float32)
            for key in np.unique(keys):
                idx=np.where(keys==key)[0]; pool=cpool.get(int(key)) or allc
                pick=rng.integers(0,len(pool),size=len(idx)); newc[idx]=np.stack([pool[p] for p in pick])
            f2=me["feat"].clone(); f2[:,j,2:2+CV]=torch.tensor(newc)
            with torch.no_grad(): yp=m(f2,me["nd"],me["pr"],mask).numpy()
            F=(yp-yh)**2; Bn=np.abs(yp-y)-base
            np.add.at(accF,(ar[valid],dv[valid]),F[valid])
            np.add.at(accB,(ar[valid],dv[valid]),Bn[valid])
            np.add.at(cnt,(ar[valid],dv[valid]),1.0)
    return _bands(accF,accB,cnt)

def _bands(accF,accB,cnt):
    G=cnt.shape[0]
    pgF=np.where(cnt>0,accF/np.maximum(cnt,1),np.nan); pgB=np.where(cnt>0,accB/np.maximum(cnt,1),np.nan)
    def band_stat(pg):
        mu=np.full(MAXD+1,np.nan); lo=np.full(MAXD+1,np.nan); hi=np.full(MAXD+1,np.nan)
        for d in range(1,MAXD+1):
            v=pg[:,d]; v=v[np.isfinite(v)]
            if v.size<5: continue
            mu[d]=v.mean(); bs=np.array([np.random.default_rng(s).choice(v,v.size).mean() for s in range(400)])
            lo[d],hi[d]=np.percentile(bs,[2.5,97.5])
        return mu,lo,hi
    return band_stat(pgF),band_stat(pgB),(cnt>0).sum(0)

def carriage_curves_struct(m,me,mask):
    """STRUCTURAL donor swap: resample node j's node-RRWP + its readout-pair-RRWP pr[0,j] with a
    DEGREE-matched donor (on-manifold), holding content fixed. Same distance bands, F(d)/B(d)."""
    G=len(me["y"]); y=me["y"].numpy(); deg=me["deg"]; dist=me["dist"]
    with torch.no_grad(): yh=m(me["feat"],me["nd"],me["pr"],mask).numpy()
    base=np.abs(yh-y)
    npool={}; ppool={}                    # aligned by append order => same donor node
    for g in range(G):
        for j in range(1,N):
            npool.setdefault(int(deg[g,j]),[]).append(me["nd"][g,j].numpy())
            ppool.setdefault(int(deg[g,j]),[]).append(me["pr"][g,0,j].numpy())
    alln=[x for v in npool.values() for x in v]; allp=[x for v in ppool.values() for x in v]
    rng=np.random.default_rng(0)
    accF=np.zeros((G,MAXD+1)); accB=np.zeros((G,MAXD+1)); cnt=np.zeros((G,MAXD+1)); ar=np.arange(G)
    for j in range(1,N):
        keys=deg[:,j]; dd=dist[:,j]; valid=(dd>=1)&(dd<=MAXD); dv=dd.astype(int)
        for _ in range(NDON):
            ndn=np.zeros((G,Kw),np.float32); prn=np.zeros((G,Kw),np.float32)
            for key in np.unique(keys):
                idx=np.where(keys==key)[0]; pn=npool.get(int(key)) or alln; pp=ppool.get(int(key)) or allp
                pick=rng.integers(0,len(pn),size=len(idx))
                ndn[idx]=np.stack([pn[p] for p in pick]); prn[idx]=np.stack([pp[p] for p in pick])
            nd2=me["nd"].clone(); pr2=me["pr"].clone()
            nd2[:,j]=torch.tensor(ndn); pr2[:,0,j]=torch.tensor(prn); pr2[:,j,0]=torch.tensor(prn)
            with torch.no_grad(): yp=m(me["feat"],nd2,pr2,mask).numpy()
            F=(yp-yh)**2; Bn=np.abs(yp-y)-base
            np.add.at(accF,(ar[valid],dv[valid]),F[valid]); np.add.at(accB,(ar[valid],dv[valid]),Bn[valid])
            np.add.at(cnt,(ar[valid],dv[valid]),1.0)
    return _bands(accF,accB,cnt)

print(("SMOKE " if SMOKE else "")+"distance-dependent functional & beneficial carriage (repo-faithful) ...")
R={}
for v,t in MODELS:
    t0=time.time(); m,me,mask=train(v,t)
    with torch.no_grad(): sk=1-((m(me["feat"],me["nd"],me["pr"],mask)-me["y"])**2).mean().item()/me["y"].var().item()
    cF,cB,ng=carriage_curves(m,me,mask)               # CONTENT donor swap
    sF,sB,_=carriage_curves_struct(m,me,mask)         # STRUCTURAL donor swap
    R[(v,t)]=dict(skill=sk,F=cF,B=cB,Fs=sF,Bs=sB,ng=ng)
    print(f"[{v}/{t}] skill={sk:.3f}  content F(d)={np.array2string(cF[0][1:],precision=3,floatmode='fixed')}  "
          f"struct F(d)={np.array2string(sF[0][1:],precision=3,floatmode='fixed')}  ng={ng[1:].tolist()}  ({time.time()-t0:.0f}s)",flush=True)

ds=np.arange(1,MAXD+1)
STY={"dense":("-","o"),"1-hop":("--","^")}
def make_fig(fkey,bkey,cols,probe,fname):
    fig,ax=plt.subplots(2,2,figsize=(11.5,8),sharex=True)
    for ci,(task,role) in enumerate(cols):
        col="#1f77b4" if task=="semantic" else "#d62728"
        for ri,(chan,lab) in enumerate([(fkey,f"functional carriage  $F(d)=E[(\\hat y'-\\hat y)^2]$"),
                                         (bkey,f"beneficial carriage  $B(d)=E[|\\hat y'-y|-|\\hat y-y|]$")]):
            axi=ax[ri,ci]
            for v in ["dense","1-hop"]:
                d=R[(v,task)]; mu,lo,hi=d[chan]; ls,mk=STY[v]
                axi.plot(ds,mu[1:],ls,color=col,marker=mk,ms=5,label=f"{v}  (skill {d['skill']:.2f})")
                axi.fill_between(ds,lo[1:],hi[1:],color=col,alpha=0.15,linewidth=0)
            axi.axhline(0,color="k",lw=0.5); axi.grid(alpha=0.3); axi.set_xticks(ds)
            if ri==0: axi.set_title(f"{task} task  ({role})")
            if ri==1: axi.set_xlabel("shortest-path distance $d$ from readout node")
            if ci==0: axi.set_ylabel(lab,fontsize=9)
            axi.legend(fontsize=8)
    fig.suptitle(f"Distance-dependent functional & beneficial {probe} carriage (repo F(d)/B(d), degree-matched donor swaps, 95% bootstrap CI)")
    fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(fname,dpi=140); print(f"saved {fname}")

make_fig("F","B",[("semantic","signal"),("structural","control")],"content","fig_distance_carriage.png")
make_fig("Fs","Bs",[("structural","signal"),("semantic","control")],"structural","fig_distance_carriage_structural.png")
