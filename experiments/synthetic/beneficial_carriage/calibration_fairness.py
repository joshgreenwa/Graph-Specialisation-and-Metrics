"""Fairness/calibration of per-head semantic vs structural TRANSPORT scores (head_scores.py style) with
the topology-perturbation structural intervention. Do dense and 1-hop get comparable-scale, calibrated
scores? Test: 3 tasks (pure-semantic, pure-structural, mixed), both archs at matched skill, targets
within reach (so no reach-wall confound). Per head, alpha-weighted ||d ho|| under:
  semantic  = content donor-swap of a node (structure held fixed)     -> sem score
  structural= topology degree-preserving swap (content held fixed; recompute RRWP + mask) -> str score
Report model-level mean sem/str and structural fraction per task. Fair iff: semantic task -> low struct
fraction, structural task -> high, mixed -> mid, and comparable-ish dense vs 1-hop (after a principled
normalisation). Watch for 1-hop mask-inflation making struct spuriously high on the SEMANTIC task.
"""
import os, math, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=10,6,4,32,4,4
G_TR,G_VA,G_ME=(2000,500,200) if SMOKE else (6000,1200,600)
EPOCHS=4 if SMOKE else 30; BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; RADIUS=2; TSTEP=3
TASKS=["semantic","structural","mixed"]; VARS=[("dense",0),("1-hop",1)]
def rand_graph_adj(n):
    A=np.zeros((n,n),np.float32)
    for i in range(1,n): p=np.random.randint(0,i); A[i,p]=A[p,i]=1.0
    for _ in range(np.random.randint(0,n)):
        a,b=np.random.randint(0,n),np.random.randint(0,n)
        if a!=b: A[a,b]=A[b,a]=1.0
    return A
def bfs(A,s):
    d=np.full(N,np.inf,np.float32); d[s]=0; fr=[s]; step=0
    while fr:
        step+=1; nx=[]
        for u in fr:
            for v in np.where(A[u]>0)[0]:
                if d[v]==np.inf: d[v]=step; nx.append(v)
        fr=nx
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
g_sem=mk_mlp(CV); g_str=mk_mlp(TSTEP)
def gen(task,G,stat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); nd,pr=rrwp_np(A); cont=np.random.randn(G,N,CV).astype(np.float32)
    B=np.zeros(G,int)
    for g in range(G):
        d0=bfs(A[g],0); cand=np.where((d0>=1)&(d0<=3))[0]; B[g]=int(np.random.choice(cand)) if len(cand) else int(np.random.choice(np.where(d0>=1)[0]))
    ar=np.arange(G); zc=apply_g(cont[ar,B],g_sem); zs=apply_g(nd.numpy()[ar,B,:TSTEP],g_str)
    zc=(zc-zc.mean())/(zc.std()+1e-6); zs=(zs-zs.mean())/(zs.std()+1e-6)
    yr={"semantic":zc,"structural":zs,"mixed":(zc+zs)/math.sqrt(2)}[task]
    if stat is None: stat=(yr.mean(),yr.std()+1e-6)
    y=(yr-stat[0])/stat[1]
    flags=np.zeros((G,N,2),np.float32); flags[:,0,0]=1.0; flags[ar,B,1]=1.0
    return dict(A=A,feat=torch.tensor(np.concatenate([flags,cont],-1)),nd=nd,pr=pr,
                y=y.astype(np.float32),B=B,deg=A.sum(-1).astype(int)),stat
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Linear(H,H);s.k=nn.Linear(H,H);s.v=nn.Linear(H,H);s.o=nn.Linear(H,H)
        s.bb=nn.Linear(Kw,HEADS); s.pv=nn.Linear(Kw,H)
        s.n1=nn.LayerNorm(H);s.n2=nn.LayerNorm(H);s.mlp=nn.Sequential(nn.Linear(H,2*H),nn.GELU(),nn.Linear(2*H,H))
    def forward(s,x,pr,mask,cap=False):
        B=x.size(0);xn=s.n1(x)
        q=s.q(xn).view(B,N,HEADS,dh).transpose(1,2);k=s.k(xn).view(B,N,HEADS,dh).transpose(1,2);v=s.v(xn).view(B,N,HEADS,dh).transpose(1,2)
        a=((q@k.transpose(-2,-1))/math.sqrt(dh)+s.bb(pr).permute(0,3,1,2)).masked_fill(~mask,float("-inf")).softmax(-1)
        ho=a@v+torch.einsum('bhij,bhijd->bhid',a,s.pv(pr).view(B,N,N,HEADS,dh).permute(0,3,1,2,4))
        x=x+s.o(ho.transpose(1,2).reshape(B,N,H)); x=x+s.mlp(s.n2(x))
        return (x,(a.detach(),ho.detach())) if cap else x
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=nn.Linear(2+CV+Kw,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
    def forward(s,feat,nd,pr,mask,cap=False):
        x=s.enc(torch.cat([feat,nd],-1)); caps=[]
        for b in s.blocks:
            if cap: x,c=b(x,pr,mask,cap=True); caps.append(c)
            else: x=b(x,pr,mask)
        pred=s.head(x[:,0]).squeeze(-1)
        return (pred,caps) if cap else pred
def khop_mask(A,k):
    if k==0: return torch.ones(A.shape[0],N,N,dtype=torch.bool)
    R=((A+np.eye(N))>0).astype(np.float32); M=R.copy()
    for _ in range(k-1): M=((M@R)>0).astype(np.float32)
    return torch.tensor(M>0)
def train(task,k):
    torch.manual_seed(0); np.random.seed(1)
    tr,st=gen(task,G_TR); va,_=gen(task,G_VA,st); me,_=gen(task,G_ME,st)
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
def scores(m,me,k):
    G=len(me["y"]); m0=khop_mask(me["A"],k); rng=np.random.default_rng(0)
    with torch.no_grad(): _,caps0=m(me["feat"],me["nd"],me["pr"],m0[:,None],cap=True)
    def hodelta(caps):
        vd=np.zeros((L,HEADS))
        for l in range(L): vd[l]=(caps[l][1]-caps0[l][1]).norm(dim=-1).mean((0,2)).numpy()  # alpha-weighted ||d ho||
        return vd
    sem=np.zeros((L,HEADS)); stru=np.zeros((L,HEADS))
    for j in range(1,N):
        f2=me["feat"].clone(); f2[:,j,2:2+CV]=torch.randn(G,CV)                       # semantic: content swap
        with torch.no_grad(): _,cc=m(f2,me["nd"],me["pr"],m0[:,None],cap=True)
        sem+=hodelta(cc)
        A2=np.stack([dp_local(me["A"][g],j,rng) for g in range(G)])                    # structural: topology swap (content fixed)
        nd2,pr2=rrwp_np(A2); m2=khop_mask(A2,k)
        with torch.no_grad(): _,cs=m(me["feat"],nd2,pr2,m2[:,None],cap=True)
        stru+=hodelta(cs)
    return sem/(N-1), stru/(N-1)
print(("SMOKE " if SMOKE else "")+"calibration/fairness of head transport scores (topology structural intervention) ...")
R={}
for task in TASKS:
    for name,k in VARS:
        m,me=train(task,k)
        with torch.no_grad(): sk=1-((m(me["feat"],me["nd"],me["pr"],khop_mask(me["A"],k)[:,None])-torch.tensor(me["y"]))**2).mean().item()/me["y"].var()
        sem,stru=scores(m,me,k); R[(task,name)]=dict(sk=sk,sem=sem,stru=stru)
        print(f"[{task:10}/{name:5}] skill={sk:.3f}  mean sem={sem.mean():.4f}  mean str={stru.mean():.4f}  raw str-frac={stru.mean()/(sem.mean()+stru.mean()):.2f}",flush=True)
# normalisations: (a) GLOBAL per-channel (head_scores style); (b) PER-MODEL per-channel (fairness)
gsem=np.mean([R[key]["sem"].mean() for key in R]); gstr=np.mean([R[key]["stru"].mean() for key in R])
print("\nstructural fraction by task x arch:")
print(f"  {'task':<11}{'arch':<7}{'raw':>7}{'globalNorm':>12}{'perModelNorm':>14}   (perModel = calibrated by each arch''s mean sem & str across tasks)")
permodel={}  # per-model channel means across tasks
for name,k in VARS:
    permodel[name]=(np.mean([R[(t,name)]["sem"].mean() for t in TASKS]), np.mean([R[(t,name)]["stru"].mean() for t in TASKS]))
rows={}
for task in TASKS:
    for name,k in VARS:
        s=R[(task,name)]["sem"].mean(); t=R[(task,name)]["stru"].mean()
        raw=t/(s+t)
        sn,tn=s/gsem,t/gstr; gN=tn/(sn+tn)
        psm,ptm=permodel[name]; sp,tp=s/psm,t/ptm; pmN=tp/(sp+tp)
        rows[(task,name)]=(raw,gN,pmN)
        print(f"  {task:<11}{name:<7}{raw:>7.2f}{gN:>12.2f}{pmN:>14.2f}")
# figure
fig,ax=plt.subplots(1,3,figsize=(14,4.4)); norms=[("raw",0),("global-norm",1),("per-model-norm",2)]
x=np.arange(len(TASKS)); w=0.36
for pi,(lab,ix) in enumerate(norms):
    for gi,(name,k) in enumerate(VARS):
        vals=[rows[(t,name)][ix] for t in TASKS]; ax[pi].bar(x+(gi-0.5)*w,vals,w,label=name,color=("#6a3d9a" if name=="dense" else "#33a02c"),edgecolor="k",linewidth=0.4)
    ax[pi].set_xticks(x); ax[pi].set_xticklabels(TASKS,rotation=12); ax[pi].set_ylim(0,1); ax[pi].axhline(0.5,color="k",lw=0.5,ls=":")
    ax[pi].set_title(f"structural fraction ({lab})",fontsize=10); ax[pi].grid(axis="y",alpha=0.3); ax[pi].legend(fontsize=8)
ax[0].set_ylabel("structural fraction  str/(sem+str)")
fig.suptitle("Fairness of head transport scores: does structural fraction track task type (low→high) comparably for dense & 1-hop?")
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig("fig_calibration_fairness.png",dpi=140); print("saved fig_calibration_fairness.png")
