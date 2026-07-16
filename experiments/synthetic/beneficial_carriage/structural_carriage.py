"""Structural (RRWP) carriage: validating functional (i) and beneficial (ii) estimators.

Lightweight RRWP-augmented transformer (upgrade of the semantic toy): node-RRWP is
concatenated onto node features; pair-RRWP enters as an additive attention bias AND a
value augmentation (GRIT-like). RRWP_r = (D^{-1}(A+I))^r, r=0..K-1; node-RRWP = diagonal,
pair-RRWP = off-diagonal.

Two structural tasks on random trees (uninformative content -> label is purely structural),
each with a planted ground-truth source B:
  node task : y = w_node . node_rrwp[B]          (depends only on B's node-RRWP)
  pair task : y = w_pair . pair_rrwp[0,B]        (depends only on the readout<->B pair-RRWP)

Objectives:
  (i)  functional carriage -- does the estimator localise the RRWP the model USES to B?
       methods: grad sensitivity, IG (graph-mean baseline), channel-zero swap.
  (ii) beneficial carriage -- does it localise the RRWP that HELPS the task to B, under the
       OOD induced by RRWP swaps? methods: direct-mean (OOD), matched-donor (same
       degree/distance real block, ~on-manifold), topology-recompute (rewire+recompute RRWP,
       on-manifold reference), first-order. Score = AUROC(B vs other sources).
"""
import os, math, time, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"

N,Kw,H,HEADS,L=10,6,40,4,3
G_TR,G_VA=(2000,500) if SMOKE else (8000,1500)
G_ME=200 if SMOKE else 600
EPOCHS=4 if SMOKE else 40
BS,LR,WD=256,1e-3,1e-4
dh=H//HEADS

# ---------- graph + RRWP ----------
def rand_graph_adj(n):
    A=np.zeros((n,n),np.float32)
    for i in range(1,n):                       # spanning tree -> connected
        p=np.random.randint(0,i); A[i,p]=A[p,i]=1.0
    for _ in range(np.random.randint(0,n)):    # extra edges -> variable density (in-distribution rewiring)
        a,b=np.random.randint(0,n),np.random.randint(0,n)
        if a!=b: A[a,b]=A[b,a]=1.0
    return A
def rrwp_from_A(A):  # A [G,N,N] (no self loops); returns node[G,N,K], pair[G,N,N,K]
    G=A.shape[0]; I=torch.eye(N).expand(G,N,N)
    Asl=A+I; d=Asl.sum(-1,keepdim=True); M=Asl/d
    P=I.clone(); node=[torch.diagonal(P,dim1=1,dim2=2)]; pair=[P.clone()]
    cur=I.clone()
    for _ in range(1,Kw):
        cur=cur@M; node.append(torch.diagonal(cur,dim1=1,dim2=2)); pair.append(cur.clone())
    node=torch.stack(node,-1); pair=torch.stack(pair,-1)  # [G,N,K],[G,N,N,K]
    return node,pair
def bfs_dist(A):  # [N,N] adjacency -> distance from node 0 to each node
    n=A.shape[0]; dist=np.full(n,-1); dist[0]=0; q=[0]
    while q:
        u=q.pop(0)
        for v in range(n):
            if A[u,v]>0 and dist[v]<0: dist[v]=dist[u]+1; q.append(v)
    return dist

# fixed random 2-layer tanh readouts g(.) : R^K -> R  (NONLINEAR label -> can expose OOD)
def mk_mlp():
    return (np.random.randn(Kw,8).astype(np.float32)/np.sqrt(Kw), np.random.randn(8).astype(np.float32),
            np.random.randn(8,1).astype(np.float32)/np.sqrt(8))
g_node=mk_mlp(); g_pair=mk_mlp()
def apply_g(blk,g):  # blk [.,K] -> [.]
    W1,b1,W2=g; return (np.tanh(blk@W1+b1)@W2)[...,0]
def gen(G,task):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); At=torch.tensor(A)
    node,pair=rrwp_from_A(At)
    B=np.random.randint(1,N,size=G)
    flags=np.zeros((G,N,2),np.float32); flags[:,0,0]=1.0; flags[np.arange(G),B,1]=1.0
    if task=="node": y=apply_g(node.numpy()[np.arange(G),B],g_node)
    else:            y=apply_g(pair.numpy()[np.arange(G),0,B],g_pair)
    y=(y-y.mean())/(y.std()+1e-6)
    deg=A.sum(-1).astype(int)                       # [G,N]
    d0=np.stack([bfs_dist(A[g]) for g in range(G)]) # [G,N] dist from node 0
    return dict(A=At,node=node,pair=pair,flags=torch.tensor(flags),y=torch.tensor(y.astype(np.float32)),
                B=B,deg=deg,d0=d0)

# ---------- model ----------
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Linear(H,H); s.k=nn.Linear(H,H); s.v=nn.Linear(H,H); s.o=nn.Linear(H,H)
        s.bb=nn.Linear(Kw,HEADS); s.pv=nn.Linear(Kw,H)
        s.n1=nn.LayerNorm(H); s.n2=nn.LayerNorm(H); s.mlp=nn.Sequential(nn.Linear(H,2*H),nn.GELU(),nn.Linear(2*H,H))
    def forward(s,x,pair):
        Bsz=x.size(0); xn=s.n1(x)
        q=s.q(xn).view(Bsz,N,HEADS,dh).transpose(1,2); k=s.k(xn).view(Bsz,N,HEADS,dh).transpose(1,2)
        v=s.v(xn).view(Bsz,N,HEADS,dh).transpose(1,2)
        bias=s.bb(pair).permute(0,3,1,2)                          # [B,HEADS,N,N]
        a=((q@k.transpose(-2,-1))/math.sqrt(dh)+bias).softmax(-1) # [B,HEADS,N,N]
        pv=s.pv(pair).view(Bsz,N,N,HEADS,dh).permute(0,3,1,2,4)   # [B,HEADS,N,N,dh]
        out=(a@v)+torch.einsum('bhij,bhijd->bhid',a,pv)          # [B,HEADS,N,dh]
        x=x+s.o(out.transpose(1,2).reshape(Bsz,N,H)); return x+s.mlp(s.n2(x))
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=nn.Linear(2+Kw,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
    def forward(s,flags,node,pair):
        x=s.enc(torch.cat([flags,node],-1))
        for b in s.blocks: x=b(x,pair)
        return s.head(x[:,0]).squeeze(-1)

def train(task,D):
    t0=time.time(); m=Net(); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    Xtr=D["tr"]; Xva=D["va"]
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(Xtr["y"]))
        for i in range(0,len(perm),BS):
            idx=perm[i:i+BS]; opt.zero_grad()
            p=m(Xtr["flags"][idx],Xtr["node"][idx],Xtr["pair"][idx]); lf(p,Xtr["y"][idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(Xva["flags"],Xva["node"],Xva["pair"])-Xva["y"])**2).mean().item()/Xva["y"].var().item()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); print(f"[{task}] skill={best:.3f} ({time.time()-t0:.0f}s)",flush=True); return m

def auroc(sc,lb):
    o=np.argsort(sc); r=np.empty(len(sc)); r[o]=np.arange(1,len(sc)+1); p=lb.sum(); n=len(lb)-p
    return float((r[lb==1].sum()-p*(p+1)/2)/(p*n)) if p and n else float("nan")

# ---------- carriage methods ----------
def run_task(task):
    tr=gen(G_TR,task); va=gen(G_VA,task); me=gen(G_ME,task)
    m=train(task,{"tr":tr,"va":va}); m.eval()
    fl,nd,pr,y=me["flags"],me["node"],me["pair"],me["y"]; G=len(y)
    with torch.no_grad(): yh=m(fl,nd,pr); r=(yh-y); absr=r.abs().numpy(); yhn=yh.numpy(); yv=y.numpy()
    mean_node=tr["node"].mean((0,1)); mean_pair=tr["pair"].mean((0,1,2))     # dataset-mean RRWP block
    # donor pools (real on-manifold blocks) keyed by degree (node) / distance (pair)
    npool={}; ppool={}
    for g in range(min(2000,G_TR)):
        for j in range(1,N):
            npool.setdefault(int(tr["deg"][g,j]),[]).append(tr["node"][g,j].numpy())
            ppool.setdefault(int(tr["d0"][g,j]),[]).append(tr["pair"][g,0,j].numpy())
    lab=np.zeros((G,N)); lab[np.arange(G),me["B"]]=1.0
    src=list(range(1,N)); labels=lab[:,1:].reshape(-1)

    def set_block(node_t,pair_t,j,vec):
        n2=node_t.clone(); p2=pair_t.clone()
        if task=="node": n2[:,j,:]=vec
        else: p2[:,0,j,:]=vec; p2[:,j,0,:]=vec
        return n2,p2
    def fwd(n2,p2):
        with torch.no_grad(): return m(fl,n2,p2).numpy()
    def clean_block(j):
        return nd[:,j,:] if task=="node" else pr[:,0,j,:]

    # ----- functional -----
    grad=np.zeros((G,N)); ndg=nd.clone().requires_grad_(True); prg=pr.clone().requires_grad_(True)
    out=m(fl,ndg,prg).sum(); out.backward()
    gnode=ndg.grad; gpair=prg.grad
    for j in range(1,N):
        grad[:,j]=(gnode[:,j,:].norm(dim=-1).numpy() if task=="node" else gpair[:,0,j,:].norm(dim=-1).numpy())
    ig=np.zeros((G,N)); S=6
    for j in range(1,N):
        base=(mean_node if task=="node" else mean_pair); v=clean_block(j); acc=torch.zeros(G)
        for s in range(S):
            al=(s+0.5)/S; blk=(base+al*(v-base)).detach().requires_grad_(True)
            n2,p2=set_block(nd,pr,j,blk); yy=m(fl,n2,p2); g,=torch.autograd.grad(yy.sum(),blk); acc=acc+(g*(v-base)).sum(-1)
        ig[:,j]=(acc/S).abs().numpy()
    zero=np.zeros((G,N))
    for j in range(1,N):
        v=clean_block(j).clone(); v[:,2:]=0.0; n2,p2=set_block(nd,pr,j,v); zero[:,j]=np.abs(fwd(n2,p2)-yhn)

    # ----- beneficial (positive = beneficial) -----
    def loss_diff(newblk_fn,Kdraw):
        out=np.zeros((G,N))
        for j in range(1,N):
            acc=np.zeros(G)
            for kk in range(Kdraw):
                n2,p2=set_block(nd,pr,j,newblk_fn(j,kk)); acc+=np.abs(fwd(n2,p2)-yv)
            out[:,j]=acc/Kdraw-absr
        return out
    base_blk=(mean_node if task=="node" else mean_pair)
    direct_mean=loss_diff(lambda j,kk: base_blk.expand(G,Kw), 1)
    def donor(j,kk):
        keys=(me["deg"][:,j] if task=="node" else me["d0"][:,j]); pool=(npool if task=="node" else ppool)
        out=np.zeros((G,Kw),np.float32)
        for key in np.unique(keys):
            cand=pool.get(int(key)) or base_blk.numpy()[None]
            idx=np.where(keys==key)[0]; pick=np.random.randint(0,len(cand),size=len(idx))
            out[idx]=np.stack([cand[p] for p in pick])
        return torch.tensor(out)
    direct_donor=loss_diff(donor,4)
    first_order=np.zeros((G,N))
    for j in range(1,N):
        n2,p2=set_block(nd,pr,j,base_blk.expand(G,Kw)); yp=fwd(n2,p2)
        first_order[:,j]=np.sign(r.numpy())*(yp-yhn)
    # topology-recompute (on-manifold reference): add a random edge at j, recompute RRWP
    topo=np.zeros((G,N)); Kt=2
    Anp=me["A"].numpy()
    for j in range(1,N):
        acc=np.zeros(G)
        for _ in range(Kt):
            A2=Anp.copy()
            for g in range(G):
                choices=[c for c in range(N) if c!=j and A2[g,j,c]==0]
                if choices: c=np.random.choice(choices); A2[g,j,c]=A2[g,c,j]=1.0
            n2,p2=rrwp_from_A(torch.tensor(A2)); acc+=np.abs(fwd(n2,p2)-yv)
        topo[:,j]=acc/Kt-absr
    def AU(mat): return auroc(mat[:,1:].reshape(-1),labels)
    return dict(skill=float(1-((yhn-yv)**2).mean()/y.var().item()),
        functional={"grad":AU(grad),"IG_mean":AU(ig),"channel_zero":AU(zero)},
        beneficial={"direct_mean(OOD)":AU(direct_mean),"first_order(OOD)":AU(first_order),
                    "matched_donor":AU(direct_donor),"topo_recompute(on-mfld)":AU(topo)})

print(("SMOKE " if SMOKE else "")+"running structural carriage ...")
R={t:run_task(t) for t in ["node","pair"]}
for t in R:
    print(f"\n[{t}] skill={R[t]['skill']:.3f}")
    print("  functional:", {k:round(v,3) for k,v in R[t]['functional'].items()})
    print("  beneficial:", {k:round(v,3) for k,v in R[t]['beneficial'].items()})

# ---------- figure ----------
fig,ax=plt.subplots(2,2,figsize=(11,7.5))
fcol="#1f77b4"; bcol={"direct_mean(OOD)":"#d62728","first_order(OOD)":"#e377c2",
                      "matched_donor":"#9467bd","topo_recompute(on-mfld)":"#2ca02c"}
for ri,t in enumerate(["node","pair"]):
    fk=list(R[t]["functional"]); ax[ri,0].bar(range(len(fk)),[R[t]["functional"][k] for k in fk],color=fcol)
    ax[ri,0].set_xticks(range(len(fk))); ax[ri,0].set_xticklabels(fk,rotation=15); ax[ri,0].set_ylim(0.45,1.02)
    ax[ri,0].axhline(0.5,color="k",ls=":",lw=0.8); ax[ri,0].set_ylabel(f"{t}-RRWP task\nAUROC (B vs others)")
    if ri==0: ax[ri,0].set_title("(i) Functional structural carriage")
    bk=list(R[t]["beneficial"]); ax[ri,1].bar(range(len(bk)),[R[t]["beneficial"][k] for k in bk],color=[bcol[k] for k in bk])
    ax[ri,1].set_xticks(range(len(bk))); ax[ri,1].set_xticklabels(bk,rotation=15,ha="right"); ax[ri,1].set_ylim(0.45,1.02)
    ax[ri,1].axhline(0.5,color="k",ls=":",lw=0.8)
    if ri==0: ax[ri,1].set_title("(ii) Beneficial structural carriage (OOD vs on-manifold)")
fig.suptitle("Structural (RRWP) carriage: AUROC localising the planted source B, by estimator")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig("fig_structural.png"); print("\nsaved fig_structural.png")
