"""Is 'hold the mask constant' (recompute RRWP from the topology edit, freeze original mask) a worthy
structural carriage? Test its task-dependence: on a FEATURE-payload task (y=g(nd[B]) -- answer in RRWP
node features, read by both archs) vs a TOPOLOGY-payload task (y=g(pr[0,B]) -- answer IS the structure,
which 1-hop reads via the MASK). Perturb the target's local structure; compare FULL vs HOLD-MASK carriage.
Prediction: hold-mask is model-agnostic on the feature task (dense~1-hop) but BLIND to 1-hop on the
topology task (1-hop hold-mask ~0, because its structural reading lives in the frozen mask).
"""
import os, math, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=16,6,4,32,4,4
G_TR,G_VA,G_ME=(2500,600,700) if SMOKE else (8000,1600,2000)
EPOCHS=4 if SMOKE else 32; BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; NDON=8; RADIUS=2; TSTEP=3
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
def khop_mask(A,k):
    if k==0: return torch.ones(A.shape[0],N,N,dtype=torch.bool)
    R=((A+np.eye(N))>0).astype(np.float32); M=R.copy()
    for _ in range(k-1): M=((M@R)>0).astype(np.float32)
    return torch.tensor(M>0)
def mk_mlp(d): return (np.random.randn(d,8).astype(np.float32)/np.sqrt(d),np.random.randn(8).astype(np.float32),np.random.randn(8,1).astype(np.float32)/np.sqrt(8))
def apply_g(x,g): W1,b1,W2=g; return (np.tanh(x@W1+b1)@W2)[...,0]
g_node=mk_mlp(TSTEP); g_topo=mk_mlp(Kw)
def gen(task,G,stat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); nd,pr=rrwp_np(A); cont=np.random.randn(G,N,CV).astype(np.float32)
    B=np.random.randint(1,N,size=G); ar=np.arange(G)
    if task=="node": y=apply_g(nd.numpy()[ar,B,:TSTEP],g_node)          # feature payload (RRWP node features)
    else:            y=apply_g(pr.numpy()[ar,0,B],g_topo)               # topology payload (0<->B relation)
    if stat is None: stat=(y.mean(),y.std()+1e-6)
    y=(y-stat[0])/stat[1]
    flags=np.zeros((G,N,2),np.float32); flags[:,0,0]=1.0; flags[ar,B,1]=1.0
    return dict(A=A,feat=torch.tensor(np.concatenate([flags,cont],-1)),nd=nd,pr=pr,
                y=y.astype(np.float32),B=B),stat
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
def train(task,k):
    torch.manual_seed(0); np.random.seed(1)
    tr,st=gen(task,G_TR); va,_=gen(task,G_VA,st); me,_=gen(task,G_ME,st)
    trM=khop_mask(tr["A"],k); vaM=khop_mask(va["A"],k)
    m=Net(); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(tr["y"]))
        for i in range(0,len(perm),BS):
            idx=perm[i:i+BS]; opt.zero_grad()
            yb=torch.tensor(tr["y"][idx.numpy()])
            lf(m(tr["feat"][idx],tr["nd"][idx],tr["pr"][idx],trM[idx][:,None]),yb).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(va["feat"],va["nd"],va["pr"],vaM[:,None])-torch.tensor(va["y"]))**2).mean().item()/va["y"].var()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); m.eval(); return m,me
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
def bootci(v,it=1200):
    v=v[np.isfinite(v)]; mu=v.mean(); bs=np.array([np.random.default_rng(s).choice(v,v.size).mean() for s in range(it)]); return mu,np.percentile(bs,[2.5,97.5])
def at_target(m,me,k):
    G=len(me["y"]); y=me["y"]; rng=np.random.default_rng(0); m0=khop_mask(me["A"],k); tg=me["B"]
    with torch.no_grad(): yh=m(me["feat"],me["nd"],me["pr"],m0[:,None]).numpy()
    base=np.abs(yh-y)
    acc={c:[np.zeros(G),np.zeros(G),np.zeros(G)] for c in ["full","holdmask"]}
    for _ in range(NDON):
        A2=np.stack([dp_local(me["A"][g],int(tg[g]),rng) for g in range(G)])
        conn=np.array([np.isfinite(bfs(A2[g],0)).all() for g in range(G)]); nd2,pr2=rrwp_np(A2); m2=khop_mask(A2,k)
        for c,mask in [("full",m2),("holdmask",m0)]:
            with torch.no_grad(): yp=m(me["feat"],nd2,pr2,mask[:,None]).numpy()
            F=(yp-yh)**2; Bn=np.abs(yp-y)-base; a=acc[c]
            a[0][conn]+=F[conn]; a[1][conn]+=Bn[conn]; a[2][conn]+=1
    return {c:(bootci(np.where(acc[c][2]>0,acc[c][0]/np.maximum(acc[c][2],1),np.nan)),
              bootci(np.where(acc[c][2]>0,acc[c][1]/np.maximum(acc[c][2],1),np.nan))) for c in acc}
print(("SMOKE " if SMOKE else "")+"hold-mask vs full carriage: feature-payload vs topology-payload task ...")
R={}
for task in ["node","topo"]:
    for name,k in [("dense",0),("1-hop",1)]:
        m,me=train(task,k)
        with torch.no_grad(): sk=1-((m(me["feat"],me["nd"],me["pr"],khop_mask(me["A"],k)[:,None])-torch.tensor(me["y"]))**2).mean().item()/me["y"].var()
        R[(task,name)]=at_target(m,me,k)
        f_full=R[(task,name)]["full"][0][0]; f_hm=R[(task,name)]["holdmask"][0][0]
        print(f"[{task:5}/{name:5}] skill={sk:.3f}  FULL F={f_full:.3f}  HOLD-MASK F={f_hm:.3f}  (hold-mask/full={f_hm/max(f_full,1e-9):.0%})")
fig,ax=plt.subplots(1,2,figsize=(11,4.6),sharey=False)
for pi,task in enumerate(["node","topo"]):
    labs=["dense","1-hop"]; x=np.arange(2); w=0.35
    for gi,ch in enumerate(["full","holdmask"]):
        mus=[R[(task,nm)][ch][0][0] for nm in labs]; cis=np.array([R[(task,nm)][ch][0][1] for nm in labs]).T
        err=np.abs(cis-np.array(mus)); ax[pi].bar(x+(gi-0.5)*w,mus,w,yerr=err,capsize=3,label=("full topology" if ch=="full" else "hold mask constant"),
              color=("#33a02c" if ch=="full" else "#a6cee3"),edgecolor="k",linewidth=0.4)
    ax[pi].set_xticks(x); ax[pi].set_xticklabels(labs); ax[pi].grid(axis="y",alpha=0.3); ax[pi].legend(fontsize=8)
    ax[pi].set_title(("FEATURE-payload task  y=g(nd[B])" if task=="node" else "TOPOLOGY-payload task  y=g(pr[0,B])"),fontsize=10); ax[pi].set_ylabel("functional carriage $F$ at target")
fig.suptitle("Hold-mask-constant is model-agnostic on FEATURE tasks (dense~1hop) but BLIND to 1-hop on TOPOLOGY tasks (reads via the mask)")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig("fig_holdmask_test.png",dpi=140); print("saved fig_holdmask_test.png")
