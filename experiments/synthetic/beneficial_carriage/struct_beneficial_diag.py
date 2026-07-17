"""Diagnose why 1-hop's structural carriage is inflated vs dense. Decompose the topology-edit effect
into an RRWP channel (recompute node/pair-RRWP, FREEZE original mask) and a MASK channel (freeze RRWP,
update mask). Prediction: 1-hop RRWP-channel ~ dense (model-agnostic on the shared input); the MASK
channel is the excess (inflated F, and the positive IRRELEVANT B = disrupted propagation paths).
Also: smarter irrelevant sampling (node far from readout & both targets) + more donor samples.
"""
import os, math, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=16,6,4,32,4,4
G_TR,G_VA,G_ME=(2500,600,700) if SMOKE else (9000,1800,3000)
EPOCHS=4 if SMOKE else 34; BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; NDON=8; RADIUS=2; TSTEP=3
DC,DS,LAM=2,4,1.0
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
g_c=mk_mlp(TSTEP); g_s=mk_mlp(TSTEP)
def gen(G,stat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); nd,pr=rrwp_np(A); cont=np.random.randn(G,N,CV).astype(np.float32)
    Bc=np.zeros(G,int); Bs=np.zeros(G,int)
    for g in range(G):
        d0=bfs(A[g],0); cc=np.where(d0==DC)[0]; ss=np.where(d0==DS)[0]
        Bc[g]=int(np.random.choice(cc)) if len(cc) else int(np.random.choice(np.where(d0>=1)[0]))
        pool=[s for s in ss if s!=Bc[g]]; Bs[g]=int(np.random.choice(pool)) if pool else int(np.random.choice([s for s in np.where(d0>=1)[0] if s!=Bc[g]]))
    ar=np.arange(G); zc_r=apply_g(nd.numpy()[ar,Bc,:TSTEP],g_c); zs_r=apply_g(nd.numpy()[ar,Bs,:TSTEP],g_s)
    if stat is None:
        cst=(zc_r.mean(),zc_r.std()+1e-6); sst=(zs_r.mean(),zs_r.std()+1e-6)
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
def carriage(m,me,k):
    G=len(me["ytr"]); yte=me["yte"]; rng=np.random.default_rng(0)
    m0=khop_mask(me["A"],k)
    with torch.no_grad(): yh=m(me["feat"],me["nd"],me["pr"],m0[:,None]).numpy()
    base=np.abs(yh-yte)
    Rn=np.zeros(G,int)   # smarter irrelevant: far from readout AND both targets
    for g in range(G):
        sc=np.minimum(np.minimum(bfs(me["A"][g],0),bfs(me["A"][g],me["Bc"][g])),bfs(me["A"][g],me["Bs"][g]))
        sc[[0,me["Bc"][g],me["Bs"][g]]]=-1; Rn[g]=int(np.argmax(sc))
    cats={"causal":me["Bc"],"spurious":me["Bs"],"irrelevant":Rn}
    # attribution channels of ONE topology edit: recompute one encoding, freeze the others at clean values
    chans=["full","node","pair"] if k==0 else ["full","node","pair","mask"]; out={}
    for nm,tg in cats.items():
        acc={c:[np.zeros(G),np.zeros(G),np.zeros(G)] for c in chans}
        for _ in range(NDON):
            A2=np.stack([dp_local(me["A"][g],int(tg[g]),rng) for g in range(G)])
            conn=np.array([np.isfinite(bfs(A2[g],0)).all() for g in range(G)]); nd2,pr2=rrwp_np(A2); m2=khop_mask(A2,k)
            for c in chans:
                if c=="full":   args=(me["feat"],nd2,pr2,m2)
                elif c=="node": args=(me["feat"],nd2,me["pr"],m0)   # node-RRWP from edit; pair-RRWP + mask frozen
                elif c=="pair": args=(me["feat"],me["nd"],pr2,m0)   # pair-RRWP from edit; node-RRWP + mask frozen
                else:           args=(me["feat"],me["nd"],me["pr"],m2)  # mask from edit; all RRWP frozen
                with torch.no_grad(): yp=m(args[0],args[1],args[2],args[3][:,None]).numpy()
                F=(yp-yh)**2; Bn=np.abs(yp-yte)-base; a=acc[c]
                a[0][conn]+=F[conn]; a[1][conn]+=Bn[conn]; a[2][conn]+=1
        out[nm]={c:(bootci(np.where(acc[c][2]>0,acc[c][0]/np.maximum(acc[c][2],1),np.nan)),
                    bootci(np.where(acc[c][2]>0,acc[c][1]/np.maximum(acc[c][2],1),np.nan))) for c in chans}
    return out
CATS=["causal","spurious","irrelevant"]
print(("SMOKE " if SMOKE else "")+"structural beneficial carriage -- node-RRWP / pair-RRWP / MASK channel decomposition ...")
R={}
for name,k in [("dense",0),("1-hop",1)]:
    m,me=train(k); R[(name,k)]=carriage(m,me,k)
    print(f"[{name}]")
    for nm in CATS:
        s=R[(name,k)][nm]
        line="   %-11s "%nm+"  ".join(f"{c}:F={s[c][0][0]:.3f} B={s[c][1][0]:+.3f}" for c in s)
        print(line)
# figure: beneficial B decomposed into node-RRWP / pair-RRWP / mask channels; dense (left) vs 1-hop (right)
CH_STY=[("node","#1f78b4"),("pair","#a6cee3"),("mask","#fb9a99"),("full","#33a02c")]
fig,ax=plt.subplots(1,2,figsize=(12,4.7),sharey=True); x=np.arange(len(CATS)); w=0.2
for pi,(name,k) in enumerate([("dense",0),("1-hop",1)]):
    chans=["node","pair","full"] if k==0 else ["node","pair","mask","full"]
    cmap={c:col for c,col in CH_STY}; nb=len(chans)
    for gi,ch in enumerate(chans):
        mus=[R[(name,k)][nm][ch][1][0] for nm in CATS]; cis=np.array([R[(name,k)][nm][ch][1][1] for nm in CATS]).T
        err=np.abs(cis-np.array(mus)); ax[pi].bar(x+(gi-(nb-1)/2)*w,mus,w,yerr=err,capsize=2,color=cmap[ch],label=ch,edgecolor="k",linewidth=0.3)
    ax[pi].axhline(0,color="k",lw=0.6); ax[pi].set_xticks(x); ax[pi].set_xticklabels(CATS)
    ax[pi].set_title(f"{name}: beneficial $B$ by channel",fontsize=10); ax[pi].grid(axis="y",alpha=0.3); ax[pi].legend(fontsize=8)
fig.suptitle("Beneficial carriage decomposed: node-RRWP + pair-RRWP = payload (model-agnostic); mask = routing (sparse-only, carries the irrelevant $B$)")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig("fig_struct_beneficial_decomp3.png",dpi=140); print("saved fig_struct_beneficial_decomp3.png")
