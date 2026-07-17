"""Localization v2: sharpen the distance-localization validation.
Fixes vs v1: (1) LOCAL task -- label depends on a short (<=2-hop) structural signature of the target,
so the ground-truth receptive field is small; (2) LOCALITY-CONSTRAINED degree-preserving swap -- both
swapped edges kept within `radius` hops of the anchor (no long-range compensator); (3) BIGGER graphs
(N=16) so a 2-hop RF is small vs the diameter; (4) BALANCED, larger target-distance samples (plant B
at each d*). Prediction: F(d) peaks sharply at d=d*, and peak/centre-of-mass tracks d* on the diagonal.
"""
import os, math, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=16,6,4,32,4,4
G_TR,G_VA,G_ME=(2500,600,600) if SMOKE else (9000,1800,3000)
EPOCHS=4 if SMOKE else 34; BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; MAXD=6; NDON=6; RADIUS=2; TSTEP=3
def rand_graph_adj(n):
    A=np.zeros((n,n),np.float32)
    for i in range(1,n): p=np.random.randint(0,i); A[i,p]=A[p,i]=1.0
    for _ in range(np.random.randint(0,n//2)):   # sparser -> larger diameter
        a,b=np.random.randint(0,n),np.random.randint(0,n)
        if a!=b: A[a,b]=A[b,a]=1.0
    return A
def bfs(A,src):
    d=np.full(N,np.inf,np.float32); d[src]=0; fr=[src]
    for s in range(1,N):
        nx=[]
        for u in fr:
            for v in np.where(A[u]>0)[0]:
                if d[v]==np.inf: d[v]=s; nx.append(v)
        fr=nx
        if not fr: break
    return d
def reach(A,r):  # nodes within r hops (bool [N,N])
    R=((A+np.eye(N))>0); M=R.copy()
    for _ in range(r-1): M=(M@R)>0
    return M
def rrwp_np(A):
    At=torch.tensor(A); G=At.shape[0]; I=torch.eye(N).expand(G,N,N); Asl=At+I; M=Asl/Asl.sum(-1,keepdim=True)
    nd=[torch.ones(G,N)]; pr=[I.clone()]; cur=I.clone()
    for _ in range(1,Kw): cur=cur@M; nd.append(torch.diagonal(cur,dim1=1,dim2=2)); pr.append(cur.clone())
    return torch.stack(nd,-1), torch.stack(pr,-1)
def mk_mlp(d): return (np.random.randn(d,8).astype(np.float32)/np.sqrt(d),np.random.randn(8).astype(np.float32),np.random.randn(8,1).astype(np.float32)/np.sqrt(8))
def apply_g(x,g): W1,b1,W2=g; return (np.tanh(x@W1+b1)@W2)[...,0]
g_loc=mk_mlp(TSTEP)   # label = g( first TSTEP RRWP-return steps of target ) -> ~ (TSTEP-1)-hop receptive field
def gen(G,ystat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); nd,pr=rrwp_np(A); cont=np.random.randn(G,N,CV).astype(np.float32)
    B=np.zeros(G,int); dB=np.zeros(G,int)
    for g in range(G):
        d0=bfs(A[g],0); want=1+(g%MAXD); cand=np.where(d0==want)[0]
        b=int(np.random.choice(cand)) if len(cand) else int(np.random.choice(np.where(d0>=1)[0]))
        B[g]=b; dB[g]=int(d0[b])
    ar=np.arange(G); y=apply_g(nd.numpy()[ar,B,:TSTEP],g_loc)
    if ystat is None: ystat=(y.mean(),y.std()+1e-6)
    y=(y-ystat[0])/ystat[1]
    flags=np.zeros((G,N,2),np.float32); flags[:,0,0]=1.0; flags[ar,B,1]=1.0
    return dict(A=A,feat=torch.tensor(np.concatenate([flags,cont],-1)),nd=nd,pr=pr,
                y=torch.tensor(y.astype(np.float32)),B=B,dB=dB),ystat
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
def train():
    torch.manual_seed(0); np.random.seed(1)
    tr,st=gen(G_TR); va,_=gen(G_VA,st); me,_=gen(G_ME,st); MK=lambda n: torch.ones(n,1,N,N,dtype=torch.bool)
    m=Net(); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(tr["y"]))
        for i in range(0,len(perm),BS):
            idx=perm[i:i+BS]; opt.zero_grad()
            lf(m(tr["feat"][idx],tr["nd"][idx],tr["pr"][idx],MK(len(idx))),tr["y"][idx]).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(va["feat"],va["nd"],va["pr"],MK(len(va["y"])))-va["y"])**2).mean().item()/va["y"].var().item()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); m.eval(); return m,me
def dp_swap_local(Ag,j,rng,radius=RADIUS,tries=40):
    """degree-preserving double-edge swap with BOTH swapped edges kept within `radius` hops of j"""
    loc=np.where(reach(Ag,radius)[j])[0]; locset=set(loc.tolist())
    nbrs=[b for b in np.where(Ag[j]>0)[0] if b!=j]
    ii,jj=np.where(np.triu(Ag,1)>0); edges=[(int(a),int(b)) for a,b in zip(ii,jj) if a in locset and b in locset]
    if not nbrs or not edges: return Ag
    for _ in range(tries):
        b=nbrs[rng.integers(len(nbrs))]; c,d=edges[rng.integers(len(edges))]
        if len({j,b,c,d})<4: continue
        if d in locset and Ag[j,d]==0 and Ag[c,b]==0:
            A2=Ag.copy(); A2[j,b]=A2[b,j]=0; A2[c,d]=A2[d,c]=0; A2[j,d]=A2[d,j]=1; A2[c,b]=A2[b,c]=1; return A2
        if c in locset and Ag[j,c]==0 and Ag[d,b]==0:
            A2=Ag.copy(); A2[j,b]=A2[b,j]=0; A2[c,d]=A2[d,c]=0; A2[j,c]=A2[c,j]=1; A2[d,b]=A2[b,d]=1; return A2
    return Ag
def carriage(m,me):
    G=len(me["y"]); y=me["y"].numpy(); rng=np.random.default_rng(0); ar=np.arange(G)
    dist=np.stack([bfs(me["A"][g],0) for g in range(G)])
    with torch.no_grad(): yh=m(me["feat"],me["nd"],me["pr"],torch.ones(G,1,N,N,dtype=torch.bool)).numpy()
    accF=np.zeros((G,MAXD+1)); cnt=np.zeros((G,MAXD+1))
    for j in range(1,N):
        dd=dist[:,j]; vd=(dd>=1)&(dd<=MAXD); dv=dd.astype(int)
        for _ in range(NDON):
            A2=np.stack([dp_swap_local(me["A"][g],j,rng) for g in range(G)])
            conn=np.array([np.isfinite(bfs(A2[g],0)).all() for g in range(G)])
            nd2,pr2=rrwp_np(A2)
            with torch.no_grad(): yp=m(me["feat"],nd2,pr2,torch.ones(G,1,N,N,dtype=torch.bool)).numpy()
            F=(yp-yh)**2; ok=vd&conn
            np.add.at(accF,(ar[ok],dv[ok]),F[ok]); np.add.at(cnt,(ar[ok],dv[ok]),1.0)
    return np.where(cnt>0,accF/np.maximum(cnt,1),np.nan)
print(("SMOKE " if SMOKE else "")+"localization v2 (local task + local swap + balanced planted d*) ...")
m,me=train()
with torch.no_grad(): sk=1-((m(me["feat"],me["nd"],me["pr"],torch.ones(len(me["y"]),1,N,N,dtype=torch.bool))-me["y"])**2).mean().item()/me["y"].numpy().var()
pg=carriage(m,me); ds=np.arange(1,MAXD+1)
print(f"[local nd[B][:{TSTEP}]] skill={sk:.3f}")
com_x=[]; com_y=[]; pk_y=[]
for dstar in range(1,MAXD+1):
    msk=me["dB"]==dstar; n=int(msk.sum())
    mu=np.array([np.nanmean(pg[msk,d]) if np.isfinite(pg[msk,d]).sum()>=5 else np.nan for d in range(MAXD+1)])
    if n<20 or not np.isfinite(mu[1:]).any(): continue
    v=np.nan_to_num(mu[1:]); com=(ds*v).sum()/(v.sum()+1e-12); pk=ds[np.argmax(v)]
    com_x.append(dstar); com_y.append(com); pk_y.append(pk)
    print(f"  d*={dstar} (n={n:4d}): F(d)={np.array2string(v,precision=3,floatmode='fixed')}  peak@d={pk}  CoM={com:.2f}")
# figure: normalized profiles + CoM-vs-d* diagonal check
fig,ax=plt.subplots(1,2,figsize=(11.5,4.7)); cmap=plt.cm.viridis(np.linspace(0.05,0.85,MAXD))
for i,dstar in enumerate(range(1,MAXD+1)):
    msk=me["dB"]==dstar; n=int(msk.sum())
    mu=np.array([np.nanmean(pg[msk,d]) if np.isfinite(pg[msk,d]).sum()>=5 else np.nan for d in range(MAXD+1)])
    if n<20 or not np.isfinite(mu[1:]).any(): continue
    v=np.nan_to_num(mu[1:]); vn=v/(v.max()+1e-12)
    ax[0].plot(ds,vn,marker="o",ms=4,color=cmap[dstar-1],label=f"d*={dstar} (n={n})")
    ax[0].axvline(dstar,color=cmap[dstar-1],ls=":",lw=1,alpha=0.6)
ax[0].set_title("normalised $F(d)$ profiles (dotted = planted $d^*$)"); ax[0].set_xlabel("perturbation distance $d$"); ax[0].set_ylabel("$F(d)$ / max"); ax[0].set_xticks(ds); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)
ax[1].plot([1,MAXD],[1,MAXD],"k:",lw=1,label="peak = $d^*$ (ideal)")
ax[1].plot(com_x,com_y,marker="o",ms=7,color="#d62728",label="centre-of-mass of $F(d)$")
ax[1].plot(com_x,pk_y,marker="s",ms=6,ls="--",color="#1f77b4",label="peak (argmax) of $F(d)$")
ax[1].set_title("does the carriage location track $d^*$?"); ax[1].set_xlabel("planted target distance $d^*$"); ax[1].set_ylabel("carriage location (d)"); ax[1].set_xticks(ds); ax[1].set_yticks(ds); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
fig.suptitle(f"Localization v2: local task (RF~{TSTEP-1} hops) + locality-constrained swap + balanced $d^*$  (skill {sk:.2f})")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("fig_localization_v2.png",dpi=140); print("saved fig_localization_v2.png")
