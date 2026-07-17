"""INSIGHT regime: does beneficial carriage REVEAL the reach-wall capability gap between dense and 1-hop?
Long-diameter graphs (chain-biased) so 1-hop (reach=L=4) fails on FAR targets. Task y=g(nd[B]) with the
target planted at controlled distance d*. Perturb AT the target; compute B_model=|yhat'-y|-|yhat-y| and
the model-agnostic B_task=|y'-y| (y'=true label of perturbed graph), split by d*.
Prediction: near d* -> B_dense~B_1hop~B_task (agree); FAR d* (beyond reach) -> B_dense~B_task but
B_1hop FALLS BELOW (can't read far structure -> output barely moves). The shortfall = GT niche.
"""
import os, math, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=18,8,4,32,4,4
G_TR,G_VA,G_ME=(2500,600,700) if SMOKE else (9000,1800,3000)
EPOCHS=4 if SMOKE else 34; BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; MAXD=8; NDON=6; RADIUS=2; TSTEP=3
def rand_graph_adj(n):   # chain-biased tree -> long diameter (node 0 at one end)
    A=np.zeros((n,n),np.float32)
    for i in range(1,n):
        parent=i-1 if np.random.rand()<0.6 else np.random.randint(0,i)
        A[i,parent]=A[parent,i]=1.0
    for _ in range(np.random.randint(0,n//5)):
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
def khop_mask(A,k):
    if k==0: return torch.ones(A.shape[0],N,N,dtype=torch.bool)
    R=((A+np.eye(N))>0).astype(np.float32); M=R.copy()
    for _ in range(k-1): M=((M@R)>0).astype(np.float32)
    return torch.tensor(M>0)
def mk_mlp(d): return (np.random.randn(d,8).astype(np.float32)/np.sqrt(d),np.random.randn(8).astype(np.float32),np.random.randn(8,1).astype(np.float32)/np.sqrt(8))
def apply_g(x,g): W1,b1,W2=g; return (np.tanh(x@W1+b1)@W2)[...,0]
g_node=mk_mlp(TSTEP)
def lab_raw(nd,B): ar=np.arange(len(B)); return apply_g(nd.numpy()[ar,B,:TSTEP],g_node)
def gen(G,stat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); nd,pr=rrwp_np(A); cont=np.random.randn(G,N,CV).astype(np.float32)
    B=np.zeros(G,int); dB=np.zeros(G,int)
    for g in range(G):
        d0=bfs(A[g],0); want=1+(g%MAXD); cand=np.where(d0==want)[0]
        b=int(np.random.choice(cand)) if len(cand) else int(np.random.choice(np.where(d0>=1)[0]))
        B[g]=b; dB[g]=int(d0[b])
    yr=lab_raw(nd,B)
    if stat is None: stat=(yr.mean(),yr.std()+1e-6)
    y=(yr-stat[0])/stat[1]
    flags=np.zeros((G,N,2),np.float32); flags[:,0,0]=1.0; flags[np.arange(G),B,1]=1.0
    return dict(A=A,feat=torch.tensor(np.concatenate([flags,cont],-1)),nd=nd,pr=pr,
                y=y.astype(np.float32),B=B,dB=dB),stat
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
            idx=perm[i:i+BS]; opt.zero_grad(); yb=torch.tensor(tr["y"][idx.numpy()])
            lf(m(tr["feat"][idx],tr["nd"][idx],tr["pr"][idx],trM[idx][:,None]),yb).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(va["feat"],va["nd"],va["pr"],vaM[:,None])-torch.tensor(va["y"]))**2).mean().item()/va["y"].var()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); m.eval(); return m,me,st
def dp_local(Ag,j,rng,radius=RADIUS,tries=40):
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
def bootci(v,it=500):
    v=v[np.isfinite(v)];
    if v.size<5: return np.nan,(np.nan,np.nan)
    mu=v.mean(); bs=np.array([np.random.default_rng(s).choice(v,v.size).mean() for s in range(it)]); return mu,tuple(np.percentile(bs,[2.5,97.5]))
def at_target(m,me,k,st):
    G=len(me["y"]); y=me["y"]; B=me["B"]; rng=np.random.default_rng(0)
    with torch.no_grad(): yh=m(me["feat"],me["nd"],me["pr"],khop_mask(me["A"],k)[:,None]).numpy()
    base=np.abs(yh-y); Bm=np.zeros(G); Bt=np.zeros(G); cg=np.zeros(G)
    for _ in range(NDON):
        A2=np.stack([dp_local(me["A"][g],int(B[g]),rng) for g in range(G)])
        conn=np.array([np.isfinite(bfs(A2[g],0)).all() for g in range(G)]); nd2,pr2=rrwp_np(A2)
        yprime=(lab_raw(nd2,B)-st[0])/st[1]
        with torch.no_grad(): yh2=m(me["feat"],nd2,pr2,khop_mask(A2,k)[:,None]).numpy()
        bm=np.abs(yh2-y)-base; bt=np.abs(yprime-y)
        Bm[conn]+=bm[conn]; Bt[conn]+=bt[conn]; cg[conn]+=1
    Bm=np.where(cg>0,Bm/np.maximum(cg,1),np.nan); Bt=np.where(cg>0,Bt/np.maximum(cg,1),np.nan)
    err=np.abs(yh-y)
    out={}
    for d in range(1,MAXD+1):
        msk=me["dB"]==d
        if msk.sum()<20: continue
        sk=1-((yh[msk]-y[msk])**2).mean()/max(y[msk].var(),1e-9)
        out[d]=(bootci(Bm[msk]),bootci(Bt[msk]),sk,int(msk.sum()))
    return out
ds=np.arange(1,MAXD+1)
print(("SMOKE " if SMOKE else "")+"reach-wall test: does B reveal the dense/1-hop capability gap at range? ...")
R={}
for name,k in [("dense",0),("1-hop",1)]:
    m,me,st=train(k)
    with torch.no_grad(): sk=1-((m(me["feat"],me["nd"],me["pr"],khop_mask(me["A"],k)[:,None])-torch.tensor(me["y"]))**2).mean().item()/me["y"].var()
    R[name]=at_target(m,me,k,st)
    print(f"[{name}] overall skill={sk:.3f}")
    for d in sorted(R[name]):
        (bm,_),(bt,_),skd,n=R[name][d]
        print(f"   d*={d} (n={n:4d}) skill={skd:+.2f}  B_model={bm:.3f}  B_task={bt:.3f}  gap={bm-bt:+.3f}")
fig,ax=plt.subplots(1,2,figsize=(12,4.7))
dd=sorted(set(R["dense"])|set(R["1-hop"]))
# skill(d*)
for name,col,mk in [("dense","#6a3d9a","o"),("1-hop","#33a02c","^")]:
    xs=[d for d in dd if d in R[name]]; sk=[R[name][d][2] for d in xs]; ax[0].plot(xs,sk,color=col,marker=mk,ms=6,label=name)
ax[0].axhline(0,color="k",lw=0.5); ax[0].set_title("skill vs target distance (1-hop fails past reach $L$=4)"); ax[0].set_xlabel("target distance $d^*$"); ax[0].set_ylabel("skill ($R^2$)"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8); ax[0].set_ylim(-0.1,1.05)
# B_task (reference, from dense which is accurate) + B_model curves
xs=[d for d in dd if d in R["dense"]]; bt=[R["dense"][d][1][0] for d in xs]
ax[1].plot(xs,bt,"k-",marker="D",ms=5,label="$B_{task}$ (true-label change)")
for name,col,mk in [("dense","#6a3d9a","o"),("1-hop","#33a02c","^")]:
    xs2=[d for d in dd if d in R[name]]; bm=[R[name][d][0][0] for d in xs2]; los=[R[name][d][0][1][0] for d in xs2]; his=[R[name][d][0][1][1] for d in xs2]
    ax[1].plot(xs2,bm,color=col,marker=mk,ms=5,label=f"$B_{{model}}$ {name}"); ax[1].fill_between(xs2,los,his,color=col,alpha=0.13,lw=0)
ax[1].axhline(0,color="k",lw=0.5); ax[1].set_title("beneficial carriage at the target vs $d^*$"); ax[1].set_xlabel("target distance $d^*$"); ax[1].set_ylabel("beneficial carriage $B$"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
fig.suptitle("Beneficial carriage reveals the reach wall: dense tracks $B_{task}$ at all $d^*$; 1-hop falls below past its reach (GT niche)")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig("fig_reachwall_carriage.png",dpi=140); print("saved fig_reachwall_carriage.png")
