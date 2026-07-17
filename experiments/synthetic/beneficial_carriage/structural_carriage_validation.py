"""Controlled validation: does degree-preserving topology carriage F(d) recover the KNOWN ground-truth
structural-dependence shape of each synthetic task?

  (1) LOCALIZATION: y = g(nd[B]) (flagged target's LOCAL structural signature), B at distance d*.
      Expect F(d) to PEAK at d = d* (only structure near the target moves the answer). Split carriage
      by the target's distance -> the peak should track d*.
  (2) NULL: y = g(content[B]) (structure irrelevant). Expect F(d) ~ 0 everywhere (no false positive).
  (3) NULL-MODEL SEMANTICS: y = deg(B). Degree-preserving swaps hold deg(B) fixed -> expect F ~ 0
      (blind to degree); a non-degree-preserving edge toggle CAN change deg(B) -> expect F > 0.
Dense model (model-agnosticism already established); focus is measured-vs-expected SHAPE.
"""
import os, math, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=12,6,4,32,4,4
G_TR,G_VA,G_ME=(2000,500,400) if SMOKE else (7000,1500,2500)
EPOCHS=4 if SMOKE else 32; BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; MAXD=5; NDON=5
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
def mk_mlp(d): return (np.random.randn(d,8).astype(np.float32)/np.sqrt(d),np.random.randn(8).astype(np.float32),np.random.randn(8,1).astype(np.float32)/np.sqrt(8))
def apply_g(x,g): W1,b1,W2=g; return (np.tanh(x@W1+b1)@W2)[...,0]
g_nd=mk_mlp(Kw); g_cv=mk_mlp(CV)
def gen(task,G,ystat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); nd,pr=rrwp_np(A); cont=np.random.randn(G,N,CV).astype(np.float32)
    B=np.random.randint(1,N,size=G); ar=np.arange(G); dist=np.stack([bfs0(A[g]) for g in range(G)])
    if task=="target":   y=apply_g(nd.numpy()[ar,B],g_nd)          # local structural signature of target
    elif task=="null":   y=apply_g(cont[ar,B],g_cv)               # pure content
    elif task=="degree": y=A.sum(1)[ar,B].astype(np.float32)      # target degree
    if ystat is None: ystat=(y.mean(),y.std()+1e-6)
    y=(y-ystat[0])/ystat[1]
    flags=np.zeros((G,N,2),np.float32); flags[:,0,0]=1.0; flags[ar,B,1]=1.0
    return dict(A=A,feat=torch.tensor(np.concatenate([flags,cont],-1)),nd=nd,pr=pr,
                y=torch.tensor(y.astype(np.float32)),B=B,deg=A.sum(-1).astype(int),
                mask1=torch.tensor((A+np.eye(N))>0),dist=dist,dB=dist[ar,B]),ystat
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
def train(task):  # dense
    torch.manual_seed(0); np.random.seed(1)
    tr,st=gen(task,G_TR); va,_=gen(task,G_VA,st); me,_=gen(task,G_ME,st)
    MK=lambda n: torch.ones(n,1,N,N,dtype=torch.bool)
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
def dp_swap(Ag,j,rng,tries=30):
    nbrs=np.where(Ag[j]>0)[0]
    if len(nbrs)==0: return Ag
    ii,jj=np.where(np.triu(Ag,1)>0); edges=list(zip(ii.tolist(),jj.tolist()))
    for _ in range(tries):
        b=int(nbrs[rng.integers(len(nbrs))]); c,d=edges[rng.integers(len(edges))]
        if len({j,b,c,d})<4: continue
        if Ag[j,d]==0 and Ag[c,b]==0:
            A2=Ag.copy(); A2[j,b]=A2[b,j]=0; A2[c,d]=A2[d,c]=0; A2[j,d]=A2[d,j]=1; A2[c,b]=A2[b,c]=1; return A2
        if Ag[j,c]==0 and Ag[d,b]==0:
            A2=Ag.copy(); A2[j,b]=A2[b,j]=0; A2[c,d]=A2[d,c]=0; A2[j,c]=A2[c,j]=1; A2[d,b]=A2[b,d]=1; return A2
    return Ag
def toggle(Ag,j,rng):  # non-degree-preserving single-edge toggle at j
    row=Ag[j]; nbrs=np.where(row>0)[0]; nbrs=nbrs[nbrs!=j]; non=np.where(row==0)[0]; non=non[non!=j]; A2=Ag.copy()
    if rng.random()<0.5 and len(nbrs)>0:
        k=int(nbrs[rng.integers(len(nbrs))]); A2[j,k]=A2[k,j]=0.0
    elif len(non)>0:
        k=int(non[rng.integers(len(non))]); A2[j,k]=A2[k,j]=1.0
    return A2
def carriage(m,me,swap):
    G=len(me["y"]); y=me["y"].numpy(); dist=me["dist"]; rng=np.random.default_rng(0); ar=np.arange(G)
    with torch.no_grad(): yh=m(me["feat"],me["nd"],me["pr"],torch.ones(G,1,N,N,dtype=torch.bool)).numpy()
    accF=np.zeros((G,MAXD+1)); cnt=np.zeros((G,MAXD+1))
    for j in range(1,N):
        dd=dist[:,j]; vd=(dd>=1)&(dd<=MAXD); dv=dd.astype(int)
        for _ in range(NDON):
            A2=np.stack([swap(me["A"][g],j,rng) for g in range(G)])
            conn=np.array([np.isfinite(bfs0(A2[g])).all() for g in range(G)])
            nd2,pr2=rrwp_np(A2)
            with torch.no_grad(): yp=m(me["feat"],nd2,pr2,torch.ones(G,1,N,N,dtype=torch.bool)).numpy()
            F=(yp-yh)**2; ok=vd&conn
            np.add.at(accF,(ar[ok],dv[ok]),F[ok]); np.add.at(cnt,(ar[ok],dv[ok]),1.0)
    return np.where(cnt>0,accF/np.maximum(cnt,1),np.nan)   # per-graph, per-perturbation-distance F
def agg(pgF,mask=None):
    mu=np.full(MAXD+1,np.nan); se=np.full(MAXD+1,np.nan); rows=pgF if mask is None else pgF[mask]
    for d in range(1,MAXD+1):
        v=rows[:,d]; v=v[np.isfinite(v)]
        if v.size>=5: mu[d]=v.mean(); se[d]=v.std()/np.sqrt(v.size)
    return mu,se
ds=np.arange(1,MAXD+1)
print(("SMOKE " if SMOKE else "")+"controlled structural-carriage validation ...")
# (1) localization
mt,me=train("target")
with torch.no_grad(): skt=1-((mt(me["feat"],me["nd"],me["pr"],torch.ones(len(me["y"]),1,N,N,dtype=torch.bool))-me["y"])**2).mean().item()/me["y"].numpy().var()
pg=carriage(mt,me,dp_swap)
print(f"[target nd[B]] skill={skt:.3f}")
for dstar in range(1,MAXD+1):
    mu,se=agg(pg,me["dB"]==dstar); n=int((me["dB"]==dstar).sum())
    peak=np.nanargmax(mu) if np.isfinite(mu[1:]).any() else -1
    print(f"  target@d*={dstar} (n={n:4d}): F(d)={np.array2string(np.nan_to_num(mu[1:]),precision=3,floatmode='fixed')}  peak@d={peak}")
# (2) null  (3) degree
mn,men=train("null")
with torch.no_grad(): skn=1-((mn(men["feat"],men["nd"],men["pr"],torch.ones(len(men["y"]),1,N,N,dtype=torch.bool))-men["y"])**2).mean().item()/men["y"].numpy().var()
pgn=carriage(mn,men,dp_swap); mun,sen=agg(pgn)
md,med=train("degree")
with torch.no_grad(): skd=1-((md(med["feat"],med["nd"],med["pr"],torch.ones(len(med["y"]),1,N,N,dtype=torch.bool))-med["y"])**2).mean().item()/med["y"].numpy().var()
pgd_dp=carriage(md,med,dp_swap); mudp,sedp=agg(pgd_dp)
pgd_tg=carriage(md,med,toggle);  mutg,setg=agg(pgd_tg)
print(f"[null content] skill={skn:.3f}  F(d)={np.array2string(np.nan_to_num(mun[1:]),precision=4,floatmode='fixed')}  (expect ~0)")
print(f"[degree] skill={skd:.3f}  dp-swap F(d)={np.array2string(np.nan_to_num(mudp[1:]),precision=4,floatmode='fixed')} (expect ~0, blind)  "
      f"toggle F(d)={np.array2string(np.nan_to_num(mutg[1:]),precision=4,floatmode='fixed')} (expect >0)")
# ---- figure ----
fig,ax=plt.subplots(1,3,figsize=(15,4.6))
cmap=plt.cm.viridis(np.linspace(0.05,0.85,MAXD))
for dstar in range(1,MAXD+1):
    mu,se=agg(pg,me["dB"]==dstar); n=int((me["dB"]==dstar).sum())
    if n<20: continue
    ax[0].plot(ds,mu[1:],marker="o",ms=5,color=cmap[dstar-1],label=f"target @ d*={dstar}")
    ax[0].fill_between(ds,mu[1:]-1.96*se[1:],mu[1:]+1.96*se[1:],color=cmap[dstar-1],alpha=0.15,lw=0)
    pk=np.nanargmax(mu); ax[0].scatter([pk],[mu[pk]],marker="*",s=180,color=cmap[dstar-1],edgecolors="k",zorder=5,linewidths=0.5)
ax[0].set_title("(1) localization: does F(d) peak at target distance d*?"); ax[0].set_xlabel("perturbation distance $d$"); ax[0].set_ylabel("$F(d)$"); ax[0].set_xticks(ds); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
ax[1].plot(ds,mun[1:],marker="o",color="#1f77b4",label="content-null task")
ax[1].fill_between(ds,mun[1:]-1.96*sen[1:],mun[1:]+1.96*sen[1:],color="#1f77b4",alpha=0.15,lw=0)
ax[1].axhline(0,color="k",lw=0.5); ax[1].set_title("(2) null control: structure irrelevant → $F(d)\\approx0$"); ax[1].set_xlabel("perturbation distance $d$"); ax[1].set_ylabel("$F(d)$"); ax[1].set_xticks(ds); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
ax[2].plot(ds,mudp[1:],marker="o",color="#6a3d9a",label="degree-preserving swap (blind)")
ax[2].plot(ds,mutg[1:],marker="^",ls="--",color="#d62728",label="non-preserving toggle (sees it)")
for mu,se,c in [(mudp,sedp,"#6a3d9a"),(mutg,setg,"#d62728")]: ax[2].fill_between(ds,mu[1:]-1.96*se[1:],mu[1:]+1.96*se[1:],color=c,alpha=0.13,lw=0)
ax[2].axhline(0,color="k",lw=0.5); ax[2].set_title("(3) null-model semantics: degree task"); ax[2].set_xlabel("perturbation distance $d$"); ax[2].set_ylabel("$F(d)$"); ax[2].set_xticks(ds); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)
fig.suptitle("Controlled validation of degree-preserving structural carriage: measured $F(d)$ vs known ground-truth structural dependence")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("fig_struct_carriage_validation.png",dpi=140); print("saved fig_struct_carriage_validation.png")
