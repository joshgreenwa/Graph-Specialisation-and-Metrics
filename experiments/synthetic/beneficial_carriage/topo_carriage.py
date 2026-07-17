"""Model-AGNOSTIC structural carriage via TOPOLOGY perturbation (not RRWP-encoding swap).

For a node j at distance d from readout 0, toggle ONE edge incident to j (add a non-neighbour or
remove a neighbour), GATED to stay connected and within the training max-degree -> the perturbed graph
is itself a valid draw from the generator (tree + random extra edges), so on-manifold. Recompute RRWP
AND the 1-hop mask from the new adjacency, forward-pass BOTH models:
    F(d)=E[(y'-y_hat)^2]   B(d)=E[|y'-y|-|y_hat-y|]        (binned by clean dist(0,j))
Since the actual structure changes, dense (reads pr) and 1-hop (reads mask) must BOTH respond.
Test: do their curves now AGREE? (model-agnostic) And is it on-manifold? (gated vs ungated).
"""
import os, math, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=10,6,4,32,4,4
G_TR,G_VA,G_ME=(2000,500,300) if SMOKE else (6000,1200,900)
EPOCHS=4 if SMOKE else 30; BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; MAXD=5; NDON=4
def rand_graph_adj(n):
    A=np.zeros((n,n),np.float32)
    for i in range(1,n): p=np.random.randint(0,i); A[i,p]=A[p,i]=1.0
    for _ in range(np.random.randint(0,n)):
        a,b=np.random.randint(0,n),np.random.randint(0,n)
        if a!=b: A[a,b]=A[b,a]=1.0
    return A
def bfs0(A):
    d=np.full(N,np.inf,np.float32); d[0]=0; fr=[0]
    for s in range(1,N):
        nx=[]
        for u in fr:
            for v in np.where(A[u]>0)[0]:
                if d[v]==np.inf: d[v]=s; nx.append(v)
        fr=nx
        if not fr: break
    return d
def rrwp_np(A):  # A [G,N,N] numpy -> nd,pr torch
    At=torch.tensor(A); G=At.shape[0]; I=torch.eye(N).expand(G,N,N); Asl=At+I; M=Asl/Asl.sum(-1,keepdim=True)
    nd=[torch.ones(G,N)]; pr=[I.clone()]; cur=I.clone()
    for _ in range(1,Kw): cur=cur@M; nd.append(torch.diagonal(cur,dim1=1,dim2=2)); pr.append(cur.clone())
    return torch.stack(nd,-1), torch.stack(pr,-1)
def mk_mlp(d): return (np.random.randn(d,8).astype(np.float32)/np.sqrt(d),np.random.randn(8).astype(np.float32),np.random.randn(8,1).astype(np.float32)/np.sqrt(8))
def apply_g(x,g): W1,b1,W2=g; return (np.tanh(x@W1+b1)@W2)[...,0]
g_str=mk_mlp(Kw)
def gen(G,ystat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); nd,pr=rrwp_np(A); cont=np.random.randn(G,N,CV).astype(np.float32); B=np.random.randint(1,N,size=G)
    y=apply_g(pr.numpy()[np.arange(G),0,B],g_str)
    if ystat is None: ystat=(y.mean(),y.std()+1e-6)
    y=(y-ystat[0])/ystat[1]
    flags=np.zeros((G,N,2),np.float32); flags[:,0,0]=1.0; flags[np.arange(G),B,1]=1.0
    dist=np.stack([bfs0(A[g]) for g in range(G)])
    return dict(A=A,feat=torch.tensor(np.concatenate([flags,cont],-1)),nd=nd,pr=pr,
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
        ho=a@v+torch.einsum('bhij,bhijd->bhid',a,s.pv(pr).view(B,N,N,HEADS,dh).permute(0,3,1,2,4))
        x=x+s.o(ho.transpose(1,2).reshape(B,N,H)); x=x+s.mlp(s.n2(x)); return x
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=nn.Linear(2+CV+Kw,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
    def forward(s,feat,nd,pr,mask):
        x=s.enc(torch.cat([feat,nd],-1))
        for b in s.blocks: x=b(x,pr,mask)
        return s.head(x[:,0]).squeeze(-1)
def train(variant):
    torch.manual_seed(0); np.random.seed(1)
    tr,st=gen(G_TR); va,_=gen(G_VA,st); me,_=gen(G_ME,st); dense=(variant=="dense")
    def MK(mask1,n): return torch.ones(n,1,N,N,dtype=torch.bool) if dense else mask1[:,None]
    m=Net(); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(tr["y"]))
        for i in range(0,len(perm),BS):
            idx=perm[i:i+BS]; opt.zero_grad()
            lf(m(tr["feat"][idx],tr["nd"][idx],tr["pr"][idx],MK(tr["mask1"][idx],len(idx))),tr["y"][idx]).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(va["feat"],va["nd"],va["pr"],MK(va["mask1"],len(va["y"])))-va["y"])**2).mean().item()/va["y"].var().item()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); m.eval()
    return m,me,dense,int(tr["deg"].max())

def toggle(A,j,rng,maxdeg,gated):
    G=A.shape[0]; A2=A.copy()
    for g in range(G):
        row=A[g,j]; nbrs=np.where(row>0)[0]; nbrs=nbrs[nbrs!=j]; non=np.where(row==0)[0]; non=non[non!=j]
        if rng.random()<0.5 and len(nbrs)>0:
            k=nbrs[rng.integers(len(nbrs))]; A2[g,j,k]=A2[g,k,j]=0.0
        elif len(non)>0:
            k=non[rng.integers(len(non))]
            if (not gated) or (A2[g,j].sum()+1<=maxdeg and A2[g,k].sum()+1<=maxdeg): A2[g,j,k]=A2[g,k,j]=1.0
    return A2

def curves(m,me,dense,maxdeg,gated=True):
    G=len(me["y"]); y=me["y"].numpy(); dist=me["dist"]; rng=np.random.default_rng(0); ar=np.arange(G)
    def MK(mask1): return torch.ones(G,1,N,N,dtype=torch.bool) if dense else mask1[:,None]
    with torch.no_grad(): yh=m(me["feat"],me["nd"],me["pr"],MK(me["mask1"])).numpy()
    base=np.abs(yh-y)
    accF=np.zeros((G,MAXD+1)); accB=np.zeros((G,MAXD+1)); cnt=np.zeros((G,MAXD+1)); rej=0; tot=0
    for j in range(1,N):
        dd=dist[:,j]; valid_d=(dd>=1)&(dd<=MAXD); dv=dd.astype(int)
        for _ in range(NDON):
            A2=toggle(me["A"],j,rng,maxdeg,gated)
            conn=np.array([np.isfinite(bfs0(A2[g])).all() for g in range(G)]); tot+=G; rej+=(~conn).sum()
            nd2,pr2=rrwp_np(A2); mask2=torch.tensor((A2+np.eye(N))>0)
            with torch.no_grad(): yp=m(me["feat"],nd2,pr2,MK(mask2)).numpy()
            F=(yp-yh)**2; Bn=np.abs(yp-y)-base; ok=valid_d&conn
            np.add.at(accF,(ar[ok],dv[ok]),F[ok]); np.add.at(accB,(ar[ok],dv[ok]),Bn[ok]); np.add.at(cnt,(ar[ok],dv[ok]),1.0)
    pgF=np.where(cnt>0,accF/np.maximum(cnt,1),np.nan); pgB=np.where(cnt>0,accB/np.maximum(cnt,1),np.nan)
    def bs(pg):
        mu=np.full(MAXD+1,np.nan); lo=mu.copy(); hi=mu.copy()
        for d in range(1,MAXD+1):
            v=pg[:,d]; v=v[np.isfinite(v)]
            if v.size<5: continue
            mu[d]=v.mean(); b=np.array([np.random.default_rng(s).choice(v,v.size).mean() for s in range(400)]); lo[d],hi[d]=np.percentile(b,[2.5,97.5])
        return mu,lo,hi
    return bs(pgF),bs(pgB),rej/max(tot,1)

print(("SMOKE " if SMOKE else "")+"model-agnostic structural carriage via topology perturbation ...")
R={}
for v in ["dense","1-hop"]:
    m,me,dense,maxdeg=train(v)
    with torch.no_grad():
        MK=torch.ones(len(me["y"]),1,N,N,dtype=torch.bool) if dense else me["mask1"][:,None]
        sk=1-((m(me["feat"],me["nd"],me["pr"],MK)-me["y"])**2).mean().item()/me["y"].numpy().var()
    (Fmu,Flo,Fhi),(Bmu,Blo,Bhi),rejrate=curves(m,me,dense,maxdeg,gated=True)
    R[v]=dict(skill=sk,F=(Fmu,Flo,Fhi),B=(Bmu,Blo,Bhi),rej=rejrate)
    print(f"[{v}] skill={sk:.3f}  F(d)={np.array2string(Fmu[1:],precision=3,floatmode='fixed')}  "
          f"B(d)={np.array2string(Bmu[1:],precision=3,floatmode='fixed')}  reject(disconn)={rejrate:.2%}",flush=True)

ds=np.arange(1,MAXD+1); STY={"dense":("-","o","#6a3d9a"),"1-hop":("--","^","#33a02c")}
fig,ax=plt.subplots(1,2,figsize=(11.5,4.8),sharex=True)
for pi,(chan,lab) in enumerate([("F","functional  $F(d)=E[(\\hat y'-\\hat y)^2]$"),("B","beneficial  $B(d)=E[|\\hat y'-y|-|\\hat y-y|]$")]):
    for v in ["dense","1-hop"]:
        mu,lo,hi=R[v][chan]; ls,mk,c=STY[v]
        ax[pi].plot(ds,mu[1:],ls,color=c,marker=mk,ms=5,label=f"{v}  (skill {R[v]['skill']:.2f})")
        ax[pi].fill_between(ds,lo[1:],hi[1:],color=c,alpha=0.15,lw=0)
    ax[pi].axhline(0,color="k",lw=0.5); ax[pi].grid(alpha=0.3); ax[pi].set_xticks(ds)
    ax[pi].set_xlabel("shortest-path distance $d$ from readout node"); ax[pi].set_ylabel(lab,fontsize=9); ax[pi].legend(fontsize=8)
fig.suptitle("Model-agnostic structural carriage via on-manifold TOPOLOGY perturbation (structural task; do dense & 1-hop now agree?)")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("fig_topo_carriage.png",dpi=140); print("saved fig_topo_carriage.png")
