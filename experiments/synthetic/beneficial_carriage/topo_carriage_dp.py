"""Model-agnostic structural carriage via DEGREE-PRESERVING double-edge swaps (configuration-model /
Maslov-Sneppen null) -- the literature-standard on-manifold local topology perturbation, replacing the
generator-specific +-1-edge toggle.

Anchored double-edge swap at a distance-d node j: remove edges (j,b),(c,d) -> add (j,d),(c,b). ALL node
degrees preserved exactly (stays on the configuration-model manifold); gated to stay simple + connected.
Each model recomputes its OWN node-RRWP, pair-RRWP, and k-hop mask from the swapped graph, so dense /
2-hop / 1-hop must all respond. Test: do their F(d)/B(d) still collapse onto one curve?
"""
import os, math, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=10,6,4,32,4,4
G_TR,G_VA,G_ME=(2000,500,300) if SMOKE else (6000,1200,800)
EPOCHS=4 if SMOKE else 30; BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; MAXD=5; NDON=4
VARIANTS=[("dense",0),("2-hop",2),("1-hop",1)]
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
def rrwp_np(A):
    At=torch.tensor(A); G=At.shape[0]; I=torch.eye(N).expand(G,N,N); Asl=At+I; M=Asl/Asl.sum(-1,keepdim=True)
    nd=[torch.ones(G,N)]; pr=[I.clone()]; cur=I.clone()
    for _ in range(1,Kw): cur=cur@M; nd.append(torch.diagonal(cur,dim1=1,dim2=2)); pr.append(cur.clone())
    return torch.stack(nd,-1), torch.stack(pr,-1)
def khop_mask(A,k):
    if k==0: return torch.ones(A.shape[0],N,N,dtype=torch.bool)
    R=((A+np.eye(N))>0).astype(np.float32); M=R.copy()
    for _ in range(k-1): M=((M@R)>0).astype(np.float32)
    return torch.tensor(M>0)
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
                y=torch.tensor(y.astype(np.float32)),deg=A.sum(-1).astype(int),dist=dist),ystat
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
def train(k):
    torch.manual_seed(0); np.random.seed(1)
    tr,st=gen(G_TR); va,_=gen(G_VA,st); me,_=gen(G_ME,st)
    trM=khop_mask(tr["A"],k); vaM=khop_mask(va["A"],k)
    m=Net(); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(tr["y"]))
        for i in range(0,len(perm),BS):
            idx=perm[i:i+BS]; opt.zero_grad()
            lf(m(tr["feat"][idx],tr["nd"][idx],tr["pr"][idx],trM[idx][:,None]),tr["y"][idx]).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(va["feat"],va["nd"],va["pr"],vaM[:,None])-va["y"])**2).mean().item()/va["y"].var().item()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); m.eval(); return m,me
def dp_swap(Ag,j,rng,tries=24):
    """anchored degree-preserving double-edge swap; returns (A2, applied?)"""
    nbrs=np.where(Ag[j]>0)[0]
    if len(nbrs)==0: return Ag,False
    ii,jj=np.where(np.triu(Ag,1)>0); edges=list(zip(ii.tolist(),jj.tolist()))
    for _ in range(tries):
        b=int(nbrs[rng.integers(len(nbrs))]); c,d=edges[rng.integers(len(edges))]
        if len({j,b,c,d})<4: continue
        if Ag[j,d]==0 and Ag[c,b]==0:
            A2=Ag.copy(); A2[j,b]=A2[b,j]=0; A2[c,d]=A2[d,c]=0; A2[j,d]=A2[d,j]=1; A2[c,b]=A2[b,c]=1; return A2,True
        if Ag[j,c]==0 and Ag[d,b]==0:
            A2=Ag.copy(); A2[j,b]=A2[b,j]=0; A2[c,d]=A2[d,c]=0; A2[j,c]=A2[c,j]=1; A2[d,b]=A2[b,d]=1; return A2,True
    return Ag,False
def curves(m,me,k):
    G=len(me["y"]); y=me["y"].numpy(); dist=me["dist"]; rng=np.random.default_rng(0); ar=np.arange(G)
    with torch.no_grad(): yh=m(me["feat"],me["nd"],me["pr"],khop_mask(me["A"],k)[:,None]).numpy()
    base=np.abs(yh-y); accF=np.zeros((G,MAXD+1)); accB=np.zeros((G,MAXD+1)); cnt=np.zeros((G,MAXD+1))
    degchg=0.0; napplied=0
    for j in range(1,N):
        dd=dist[:,j]; vd=(dd>=1)&(dd<=MAXD); dv=dd.astype(int)
        for _ in range(NDON):
            A2=me["A"].copy(); appl=np.zeros(G,bool)
            for g in range(G):
                a2,ok=dp_swap(me["A"][g],j,rng); A2[g]=a2; appl[g]=ok
            degchg=np.abs(A2.sum(1)-me["A"].sum(1)).max(); degchg=max(degchg,degchg); napplied+=appl.sum()
            conn=np.array([np.isfinite(bfs0(A2[g])).all() for g in range(G)])
            nd2,pr2=rrwp_np(A2)
            with torch.no_grad(): yp=m(me["feat"],nd2,pr2,khop_mask(A2,k)[:,None]).numpy()
            F=(yp-yh)**2; Bn=np.abs(yp-y)-base; okm=vd&conn&appl
            np.add.at(accF,(ar[okm],dv[okm]),F[okm]); np.add.at(accB,(ar[okm],dv[okm]),Bn[okm]); np.add.at(cnt,(ar[okm],dv[okm]),1.0)
    pgF=np.where(cnt>0,accF/np.maximum(cnt,1),np.nan); pgB=np.where(cnt>0,accB/np.maximum(cnt,1),np.nan)
    def bstat(pg):
        mu=np.full(MAXD+1,np.nan); lo=mu.copy(); hi=mu.copy()
        for d in range(1,MAXD+1):
            v=pg[:,d]; v=v[np.isfinite(v)]
            if v.size<5: continue
            mu[d]=v.mean(); bb=np.array([np.random.default_rng(s).choice(v,v.size).mean() for s in range(400)]); lo[d],hi[d]=np.percentile(bb,[2.5,97.5])
        return mu,lo,hi
    return bstat(pgF),bstat(pgB),float(degchg),napplied/(9*NDON*G)
print(("SMOKE " if SMOKE else "")+"model-agnostic structural carriage via DEGREE-PRESERVING double-edge swaps ...")
R={}
for name,k in VARIANTS:
    m,me=train(k)
    with torch.no_grad(): sk=1-((m(me["feat"],me["nd"],me["pr"],khop_mask(me["A"],k)[:,None])-me["y"])**2).mean().item()/me["y"].numpy().var()
    (Fmu,Flo,Fhi),(Bmu,Blo,Bhi),degchg,applied=curves(m,me,k); R[name]=dict(skill=sk,F=(Fmu,Flo,Fhi),B=(Bmu,Blo,Bhi))
    print(f"[{name}] skill={sk:.3f}  F(d)={np.array2string(Fmu[1:],precision=3,floatmode='fixed')}  "
          f"B(d)={np.array2string(Bmu[1:],precision=3,floatmode='fixed')}  [on-manifold: max|deg change|={degchg:.0f}, swap-applied={applied:.0%}]",flush=True)
ds=np.arange(1,MAXD+1); STY={"dense":("-","o","#6a3d9a"),"2-hop":("-.","s","#ff7f00"),"1-hop":("--","^","#33a02c")}
fig,ax=plt.subplots(1,2,figsize=(11.5,4.8),sharex=True)
for pi,(chan,lab) in enumerate([("F","functional  $F(d)=E[(\\hat y'-\\hat y)^2]$"),("B","beneficial  $B(d)=E[|\\hat y'-y|-|\\hat y-y|]$")]):
    for name,_ in VARIANTS:
        mu,lo,hi=R[name][chan]; ls,mk,c=STY[name]
        ax[pi].plot(ds,mu[1:],ls,color=c,marker=mk,ms=5,label=f"{name}  (skill {R[name]['skill']:.2f})")
        ax[pi].fill_between(ds,lo[1:],hi[1:],color=c,alpha=0.12,lw=0)
    ax[pi].axhline(0,color="k",lw=0.5); ax[pi].grid(alpha=0.3); ax[pi].set_xticks(ds)
    ax[pi].set_xlabel("shortest-path distance $d$ from readout node"); ax[pi].set_ylabel(lab,fontsize=9); ax[pi].legend(fontsize=8)
fig.suptitle("Structural carriage via DEGREE-PRESERVING double-edge swaps (configuration-model null): dense / 2-hop / 1-hop collapse?")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("fig_topo_carriage_dp.png",dpi=140); print("saved fig_topo_carriage_dp.png")
