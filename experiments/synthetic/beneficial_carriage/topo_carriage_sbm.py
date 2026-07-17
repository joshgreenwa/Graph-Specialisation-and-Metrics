"""Real-world-confidence test: degree-preserving topology carriage on STRUCTURED (community) graphs,
and the on-manifold LADDER. On an SBM-like family, an unconstrained degree-preserving double-edge swap
MIXES communities (off the community manifold); a BLOCK-CONSTRAINED swap additionally preserves each
node's intra/inter-block degree split (stays on the degree-corrected-SBM manifold). We ask:
  (1) does the dense/1-hop collapse (model-agnosticism) survive on structured graphs?
  (2) does the unconstrained swap go off-manifold (drop intra-block edge fraction) and inflate carriage,
      and does the block-constrained rung fix it?
"""
import os, math, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=12,6,4,32,4,4
NB=2; BLOCK=np.array([0 if i<N//2 else 1 for i in range(N)])   # node 0 in block 0
G_TR,G_VA,G_ME=(2000,500,300) if SMOKE else (6000,1200,800)
EPOCHS=4 if SMOKE else 30; BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; MAXD=4; NDON=4
VARIANTS=[("dense",0),("1-hop",1)]
def sbm_adj(rng):
    A=np.zeros((N,N),np.float32)
    for i in range(1,N):   # block-assortative spanning tree (guarantees connectivity)
        same=[k for k in range(i) if BLOCK[k]==BLOCK[i]]
        parent=int(rng.choice(same)) if (same and rng.random()<0.9) else int(rng.integers(0,i))
        A[i,parent]=A[parent,i]=1.0
    for _ in range(int(rng.integers(0,N))):   # extra edges, block-biased
        a,b=int(rng.integers(0,N)),int(rng.integers(0,N))
        if a==b: continue
        if (BLOCK[a]==BLOCK[b] and rng.random()<0.8) or (BLOCK[a]!=BLOCK[b] and rng.random()<0.08): A[a,b]=A[b,a]=1.0
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
def intra_frac(A):  # fraction of edges that are within-block (community assortativity)
    iu,ju=np.where(np.triu(A,1)>0)
    if len(iu)==0: return np.nan
    return float((BLOCK[iu]==BLOCK[ju]).mean())
def mk_mlp(d): return (np.random.randn(d,8).astype(np.float32)/np.sqrt(d),np.random.randn(8).astype(np.float32),np.random.randn(8,1).astype(np.float32)/np.sqrt(8))
def apply_g(x,g): W1,b1,W2=g; return (np.tanh(x@W1+b1)@W2)[...,0]
g_str=mk_mlp(Kw)
def gen(G,ystat=None):
    rng=np.random.default_rng(); A=np.stack([sbm_adj(np.random.default_rng(rng.integers(1<<30))) for _ in range(G)])
    nd,pr=rrwp_np(A); cont=np.random.randn(G,N,CV).astype(np.float32); B=np.random.randint(1,N,size=G)
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
def dp_swap(Ag,j,rng,preserve_block,tries=30):
    nbrs=np.where(Ag[j]>0)[0]
    if len(nbrs)==0: return Ag,False
    ii,jj=np.where(np.triu(Ag,1)>0); edges=list(zip(ii.tolist(),jj.tolist()))
    for _ in range(tries):
        b=int(nbrs[rng.integers(len(nbrs))]); c,d=edges[rng.integers(len(edges))]
        if len({j,b,c,d})<4: continue
        # orientation 1: ->(j,d),(c,b)
        if Ag[j,d]==0 and Ag[c,b]==0 and ((not preserve_block) or (BLOCK[b]==BLOCK[d] and BLOCK[c]==BLOCK[j])):
            A2=Ag.copy(); A2[j,b]=A2[b,j]=0; A2[c,d]=A2[d,c]=0; A2[j,d]=A2[d,j]=1; A2[c,b]=A2[b,c]=1; return A2,True
        # orientation 2: ->(j,c),(d,b)
        if Ag[j,c]==0 and Ag[d,b]==0 and ((not preserve_block) or (BLOCK[c]==BLOCK[b] and BLOCK[d]==BLOCK[j])):
            A2=Ag.copy(); A2[j,b]=A2[b,j]=0; A2[c,d]=A2[d,c]=0; A2[j,c]=A2[c,j]=1; A2[d,b]=A2[b,d]=1; return A2,True
    return Ag,False
def curves(m,me,k,preserve_block):
    G=len(me["y"]); y=me["y"].numpy(); dist=me["dist"]; rng=np.random.default_rng(0); ar=np.arange(G)
    with torch.no_grad(): yh=m(me["feat"],me["nd"],me["pr"],khop_mask(me["A"],k)[:,None]).numpy()
    base=np.abs(yh-y); accF=np.zeros((G,MAXD+1)); accB=np.zeros((G,MAXD+1)); cnt=np.zeros((G,MAXD+1))
    if0=np.nanmean([intra_frac(me["A"][g]) for g in range(G)]); if1=[]; applied=0
    for j in range(1,N):
        dd=dist[:,j]; vd=(dd>=1)&(dd<=MAXD); dv=dd.astype(int)
        for _ in range(NDON):
            A2=me["A"].copy(); appl=np.zeros(G,bool)
            for g in range(G):
                a2,ok=dp_swap(me["A"][g],j,rng,preserve_block); A2[g]=a2; appl[g]=ok
            applied+=appl.sum(); if1+= [intra_frac(A2[g]) for g in np.where(appl)[0]]
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
            mu[d]=v.mean(); bb=np.array([np.random.default_rng(s).choice(v,v.size).mean() for s in range(300)]); lo[d],hi[d]=np.percentile(bb,[2.5,97.5])
        return mu,lo,hi
    return bstat(pgF),bstat(pgB),if0,float(np.nanmean(if1)),applied/(( N-1)*NDON*G)
print(("SMOKE " if SMOKE else "")+"STRUCTURED (SBM) topology carriage + on-manifold ladder ...")
M={}
for name,k in VARIANTS: M[name]=train(k)
R={}
for name,k in VARIANTS:
    m,me=M[name]
    with torch.no_grad(): sk=1-((m(me["feat"],me["nd"],me["pr"],khop_mask(me["A"],k)[:,None])-me["y"])**2).mean().item()/me["y"].numpy().var()
    for pb,tag in [(False,"unconstrained"),(True,"block-constrained")]:
        (Fmu,Flo,Fhi),(Bmu,Blo,Bhi),if0,if1,appl=curves(m,me,k,pb)
        R[(name,tag)]=dict(skill=sk,F=(Fmu,Flo,Fhi),B=(Bmu,Blo,Bhi))
        print(f"[{name}/{tag}] skill={sk:.3f}  F(d)={np.array2string(Fmu[1:],precision=3,floatmode='fixed')}  "
              f"intra-block frac clean={if0:.2f}->pert={if1:.2f}  applied={appl:.0%}",flush=True)
ds=np.arange(1,MAXD+1); STY={"dense":("-","o","#6a3d9a"),"1-hop":("--","^","#33a02c")}
fig,ax=plt.subplots(1,2,figsize=(11.5,4.8),sharey=True,sharex=True)
for pi,tag in enumerate(["unconstrained","block-constrained"]):
    for name,_ in VARIANTS:
        mu,lo,hi=R[(name,tag)]["F"]; ls,mk,c=STY[name]
        ax[pi].plot(ds,mu[1:],ls,color=c,marker=mk,ms=5,label=f"{name}")
        ax[pi].fill_between(ds,lo[1:],hi[1:],color=c,alpha=0.13,lw=0)
    ax[pi].axhline(0,color="k",lw=0.5); ax[pi].grid(alpha=0.3); ax[pi].set_xticks(ds)
    ax[pi].set_title(f"{tag} degree-preserving swap"); ax[pi].set_xlabel("distance $d$ from readout"); ax[pi].legend(fontsize=8)
ax[0].set_ylabel("functional carriage $F(d)$")
fig.suptitle("Structured (SBM) graphs: model-agnostic collapse + on-manifold ladder (unconstrained vs block-constrained swap)")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("fig_topo_carriage_sbm.png",dpi=140); print("saved fig_topo_carriage_sbm.png")
