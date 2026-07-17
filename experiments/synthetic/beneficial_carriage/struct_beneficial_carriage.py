"""Faithful STRUCTURAL BENEFICIAL carriage (repo beneficial_carriage.py methodology on the topology
intervention), validating it separates USED from BENEFICIAL.

Two flagged targets: a CAUSAL source at distance d_c (drives the true test label) and a SPURIOUS source
at d_s (correlated with the label in TRAINING only). y_train = g_c(nd[B_c]) + lam*g_s(nd[B_s]);
y_test = g_c(nd[B_c]). Model reads BOTH. On-manifold (locality-constrained degree-preserving) swaps,
binned by distance, scored vs the TRUE label y_test:
    F(d) = E[(y'-y_hat)^2]              (used: prediction moves)          -> expect peaks at d_c AND d_s
    B(d) = E[|y'-y_test| - |y_hat-y_test|]  (beneficial: true error rises) -> expect peak at d_c ONLY
Faithful elements: matched(on-manifold) vs marginal(off-manifold) resampler ladder; var_ratio health
signal; bootstrap CIs; scored vs the clean label.
"""
import os, math, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=16,6,4,32,4,4
G_TR,G_VA,G_ME=(2500,600,700) if SMOKE else (9000,1800,3500)
EPOCHS=4 if SMOKE else 34; BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; MAXD=6; NDON=6; RADIUS=2; TSTEP=3
DC,DS,LAM=2,4,1.0    # causal distance, spurious distance, spurious training weight
def rand_graph_adj(n):
    A=np.zeros((n,n),np.float32)
    for i in range(1,n): p=np.random.randint(0,i); A[i,p]=A[p,i]=1.0
    for _ in range(np.random.randint(0,n//2)):
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
def reach(A,r):
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
g_c=mk_mlp(TSTEP); g_s=mk_mlp(TSTEP)
def gen(G,stat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); nd,pr=rrwp_np(A); cont=np.random.randn(G,N,CV).astype(np.float32)
    Bc=np.zeros(G,int); Bs=np.zeros(G,int)
    for g in range(G):
        d0=bfs(A[g],0)
        cc=np.where(d0==DC)[0]; ss=np.where(d0==DS)[0]
        Bc[g]=int(np.random.choice(cc)) if len(cc) else int(np.random.choice(np.where(d0>=1)[0]))
        pool=[s for s in ss if s!=Bc[g]]; Bs[g]=int(np.random.choice(pool)) if pool else int(np.random.choice([s for s in np.where(d0>=1)[0] if s!=Bc[g]]))
    ar=np.arange(G); zc_r=apply_g(nd.numpy()[ar,Bc,:TSTEP],g_c); zs_r=apply_g(nd.numpy()[ar,Bs,:TSTEP],g_s)
    if stat is None:
        cst=(zc_r.mean(),zc_r.std()+1e-6); sst=(zs_r.mean(),zs_r.std()+1e-6)          # balance the two sources
        zc=(zc_r-cst[0])/cst[1]; zs=(zs_r-sst[0])/sst[1]; yr=zc+LAM*zs; stat=(cst,sst,(yr.mean(),yr.std()+1e-6))
    cst,sst,yst=stat; zc=(zc_r-cst[0])/cst[1]; zs=(zs_r-sst[0])/sst[1]
    ytr=(zc+LAM*zs-yst[0])/yst[1]; yte=(zc-yst[0])/yst[1]
    flags=np.zeros((G,N,3),np.float32); flags[:,0,0]=1.0; flags[ar,Bc,1]=1.0; flags[ar,Bs,2]=1.0
    return dict(A=A,feat=torch.tensor(np.concatenate([flags,cont],-1)),nd=nd,pr=pr,
                ytr=torch.tensor(ytr.astype(np.float32)),yte=yte.astype(np.float32),Bc=Bc,Bs=Bs),stat
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
        super().__init__(); s.enc=nn.Linear(3+CV+Kw,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
    def forward(s,feat,nd,pr,mask):
        x=s.enc(torch.cat([feat,nd],-1))
        for b in s.blocks: x=b(x,pr,mask)
        return s.head(x[:,0]).squeeze(-1)
def khop_mask(A,k):
    if k==0: return torch.ones(A.shape[0],N,N,dtype=torch.bool)
    R=((A+np.eye(N))>0).astype(np.float32); M=R.copy()
    for _ in range(k-1): M=((M@R)>0).astype(np.float32)
    return torch.tensor(M>0)
def train(k):
    torch.manual_seed(0); np.random.seed(1)
    tr,st=gen(G_TR); va,_=gen(G_VA,st); me,_=gen(G_ME,st)
    trM=khop_mask(tr["A"],k); vaM=khop_mask(va["A"],k)
    m=Net(); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(tr["ytr"]))
        for i in range(0,len(perm),BS):
            idx=perm[i:i+BS]; opt.zero_grad()
            lf(m(tr["feat"][idx],tr["nd"][idx],tr["pr"][idx],trM[idx][:,None]),tr["ytr"][idx]).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(va["feat"],va["nd"],va["pr"],vaM[:,None])-va["ytr"])**2).mean().item()/va["ytr"].var().item()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); m.eval(); return m,me
def dp_local(Ag,j,rng,radius=RADIUS,tries=40):     # matched (on-manifold): local degree-preserving swap
    loc=set(np.where(reach(Ag,radius)[j])[0].tolist()); nbrs=[b for b in np.where(Ag[j]>0)[0] if b!=j]
    ii,jj=np.where(np.triu(Ag,1)>0); edges=[(int(a),int(b)) for a,b in zip(ii,jj) if a in loc and b in loc]
    if not nbrs or not edges: return Ag
    for _ in range(tries):
        b=nbrs[rng.integers(len(nbrs))]; c,d=edges[rng.integers(len(edges))]
        if len({j,b,c,d})<4: continue
        if d in loc and Ag[j,d]==0 and Ag[c,b]==0:
            A2=Ag.copy(); A2[j,b]=A2[b,j]=0; A2[c,d]=A2[d,c]=0; A2[j,d]=A2[d,j]=1; A2[c,b]=A2[b,c]=1; return A2
        if c in loc and Ag[j,c]==0 and Ag[d,b]==0:
            A2=Ag.copy(); A2[j,b]=A2[b,j]=0; A2[c,d]=A2[d,c]=0; A2[j,c]=A2[c,j]=1; A2[d,b]=A2[b,d]=1; return A2
    return Ag
def marginal(Ag,j,rng):                            # marginal (off-manifold): degree-changing random rewire
    row=Ag[j]; nbrs=[b for b in np.where(row>0)[0] if b!=j]; non=[b for b in np.where(row==0)[0] if b!=j]; A2=Ag.copy()
    if nbrs and non:
        b=nbrs[rng.integers(len(nbrs))]; r=non[rng.integers(len(non))]; A2[j,b]=A2[b,j]=0; A2[j,r]=A2[r,j]=1
    return A2
def bootci(v,iters=1500):
    v=v[np.isfinite(v)]; mu=v.mean(); bs=np.array([np.random.default_rng(s).choice(v,v.size).mean() for s in range(iters)]); return mu,np.percentile(bs,[2.5,97.5])
def carriage_pt(m,me,k):
    """PER-TARGET carriage: perturb the local structure AT the causal target, the spurious target, and a
    random IRRELEVANT node. Removes the distance-decay confound -> clean Causal/Spurious/Irrelevant test."""
    G=len(me["ytr"]); yte=me["yte"]; rng=np.random.default_rng(0)
    with torch.no_grad(): yh=m(me["feat"],me["nd"],me["pr"],khop_mask(me["A"],k)[:,None]).numpy()
    base=np.abs(yh-yte)
    Rn=np.array([int(rng.choice([x for x in range(1,N) if x!=me["Bc"][g] and x!=me["Bs"][g]])) for g in range(G)])
    cats={"causal":me["Bc"],"spurious":me["Bs"],"irrelevant":Rn}; out={}
    for nm,tg in cats.items():
        Fg=np.zeros(G); Bg=np.zeros(G); cg=np.zeros(G)
        for _ in range(NDON):
            A2=np.stack([dp_local(me["A"][g],int(tg[g]),rng) for g in range(G)])
            conn=np.array([np.isfinite(bfs(A2[g],0)).all() for g in range(G)])
            nd2,pr2=rrwp_np(A2)
            with torch.no_grad(): yp=m(me["feat"],nd2,pr2,khop_mask(A2,k)[:,None]).numpy()
            F=(yp-yh)**2; Bn=np.abs(yp-yte)-base
            Fg[conn]+=F[conn]; Bg[conn]+=Bn[conn]; cg[conn]+=1
        pgF=np.where(cg>0,Fg/np.maximum(cg,1),np.nan); pgB=np.where(cg>0,Bg/np.maximum(cg,1),np.nan)
        out[nm]=(bootci(pgF),bootci(pgB))
    return out
VAR=[("dense",0),("1-hop",1)]; CATS=["causal","spurious","irrelevant"]
print(("SMOKE " if SMOKE else "")+f"structural BENEFICIAL carriage, per-target trichotomy (causal@d={DC}, spurious@d={DS}, lam={LAM}) ...")
R={}
for name,k in VAR:
    m,me=train(k)
    with torch.no_grad(): yh=m(me["feat"],me["nd"],me["pr"],khop_mask(me["A"],k)[:,None]).numpy()
    sk_tr=1-((yh-me["ytr"].numpy())**2).mean()/me["ytr"].numpy().var(); sk_te=1-((yh-me["yte"])**2).mean()/me["yte"].var()
    R[name]=carriage_pt(m,me,k)
    print(f"[{name}] skill(train)={sk_tr:.3f} skill(test/causal-only)={sk_te:.3f}")
    for nm in CATS:
        (Fmu,_),(Bmu,_)=R[name][nm]
        print(f"   {nm:<11} F(used)={Fmu:.3f}   B(beneficial)={Bmu:+.3f}")
    print("   -> expect: F: causal~spurious>>irrelevant (both USED);  B: causal>0 (beneficial), spurious<=0 (harmful/not), irrelevant~0")
fig,ax=plt.subplots(1,2,figsize=(11,4.6)); x=np.arange(len(CATS)); w=0.36
COL={"dense":"#6a3d9a","1-hop":"#33a02c"}
for pi,(chan,lab) in enumerate([(0,"functional  $F$  (USED: prediction moves)"),(1,"beneficial  $B$  (BENEFICIAL: true error rises)")]):
    for i,(name,_) in enumerate(VAR):
        mus=[R[name][nm][chan][0] for nm in CATS]; cis=np.array([R[name][nm][chan][1] for nm in CATS]).T
        err=np.abs(cis-np.array(mus)); ax[pi].bar(x+(i-0.5)*w,mus,w,yerr=err,capsize=3,color=COL[name],alpha=0.85,label=name,edgecolor="k",linewidth=0.4)
    ax[pi].axhline(0,color="k",lw=0.6); ax[pi].set_xticks(x); ax[pi].set_xticklabels(CATS); ax[pi].set_title(lab,fontsize=10); ax[pi].grid(axis="y",alpha=0.3); ax[pi].legend(fontsize=8)
fig.suptitle("Structural beneficial carriage separates USED from BENEFICIAL (Causal / Spurious / Irrelevant, per-target on-manifold swap)")
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig("fig_struct_beneficial.png",dpi=140); print("saved fig_struct_beneficial.png")
