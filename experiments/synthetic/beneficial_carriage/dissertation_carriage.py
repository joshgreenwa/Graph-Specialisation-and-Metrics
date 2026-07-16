"""Dissertation figures (semantic + structural), winning carriage methods.

Flagged tasks: causal source B + spurious source Z, y_train = payload_B + lam*payload_Z,
y_test = payload_B (model reads Z -> functional>0, but Z is task-irrelevant -> beneficial~0).
Winning beneficial estimator = on-manifold matched swap, MAE (semantic: resample i.i.d. value;
structural: degree-matched real node-RRWP donor). Functional = readout-grad norm on payload.
We also compare against ATTENTION (readout row alpha_{0,j}, mean over heads/layers) and RANDOM.

Two figures:
  (1) functional vs beneficial carriage, population-aggregated per class (95% CI over graphs).
  (2) faithfulness: AUROC for {beneficial, functional, attention, random} on B-vs-Irrelevant
      and B-vs-Spurious -- attention/functional detect USAGE (fail B-vs-Spurious); only
      beneficial is faithful to TASK BENEFIT.
"""
import os, math, time, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import paper_style
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,H,HEADS,L=12,6,40,4,3
G_TR,G_VA=(2000,500) if SMOKE else (9000,1500)
G_ME=200 if SMOKE else 1200
EPOCHS=4 if SMOKE else 45
BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS; Ksw=4 if SMOKE else 32; LAM=0.2

def reseed(s=0): torch.manual_seed(s); np.random.seed(s)
def rand_graph_adj(n):
    A=np.zeros((n,n),np.float32)
    for i in range(1,n): p=np.random.randint(0,i); A[i,p]=A[p,i]=1.0
    for _ in range(np.random.randint(0,n)):
        a,b=np.random.randint(0,n),np.random.randint(0,n)
        if a!=b: A[a,b]=A[b,a]=1.0
    return A
def node_rrwp(A):
    G=A.shape[0]; I=torch.eye(N).expand(G,N,N); Asl=A+I; M=Asl/Asl.sum(-1,keepdim=True)
    outs=[torch.ones(G,N)]; cur=I.clone()
    for _ in range(1,Kw): cur=cur@M; outs.append(torch.diagonal(cur,dim1=1,dim2=2))
    return torch.stack(outs,-1)
def mk_mlp(d): return (np.random.randn(d,8).astype(np.float32)/np.sqrt(d),np.random.randn(8).astype(np.float32),np.random.randn(8,1).astype(np.float32)/np.sqrt(8))
def apply_g(blk,g): W1,b1,W2=g; return (np.tanh(blk@W1+b1)@W2)[...,0]
def gen(modality,G,split,ystat=None,gfun=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)])
    B=np.random.randint(1,N,size=G); Z=np.array([np.random.choice([k for k in range(1,N) if k!=B[g]]) for g in range(G)])
    if modality=="semantic":
        pay=np.random.randn(G,N,1).astype(np.float32); pd=1
        yB=pay[np.arange(G),B,0]; yZ=pay[np.arange(G),Z,0]
    else:
        pay=node_rrwp(torch.tensor(A)).numpy(); pd=Kw
        yB=apply_g(pay[np.arange(G),B],gfun[0]); yZ=apply_g(pay[np.arange(G),Z],gfun[1])
    y=yB+(LAM*yZ if split=="train" else 0.0)
    if ystat is None: ystat=(y.mean(),y.std()+1e-6)
    y=(y-ystat[0])/ystat[1]
    flags=np.zeros((G,N,3),np.float32); flags[:,0,0]=1.0; flags[np.arange(G),B,1]=1.0; flags[np.arange(G),Z,2]=1.0
    return dict(flags=torch.tensor(flags),pay=torch.tensor(pay),y=torch.tensor(y.astype(np.float32)),
                B=B,Z=Z,deg=A.sum(-1).astype(int),pd=pd),ystat
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Linear(H,H);s.k=nn.Linear(H,H);s.v=nn.Linear(H,H);s.o=nn.Linear(H,H)
        s.n1=nn.LayerNorm(H);s.n2=nn.LayerNorm(H);s.mlp=nn.Sequential(nn.Linear(H,2*H),nn.GELU(),nn.Linear(2*H,H))
    def forward(s,x):
        Bsz=x.size(0);xn=s.n1(x)
        q=s.q(xn).view(Bsz,N,HEADS,dh).transpose(1,2);k=s.k(xn).view(Bsz,N,HEADS,dh).transpose(1,2);v=s.v(xn).view(Bsz,N,HEADS,dh).transpose(1,2)
        a=((q@k.transpose(-2,-1))/math.sqrt(dh)).softmax(-1)
        x=x+s.o((a@v).transpose(1,2).reshape(Bsz,N,H)); x=x+s.mlp(s.n2(x)); return x,a
class Net(nn.Module):
    def __init__(s,pd):
        super().__init__(); s.enc=nn.Linear(3+pd,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
    def forward(s,flags,pay,want_attn=False):
        x=s.enc(torch.cat([flags,pay],-1)); atts=[]
        for b in s.blocks: x,a=b(x); atts.append(a)
        pred=s.head(x[:,0]).squeeze(-1)
        if not want_attn: return pred
        attn=torch.stack([a[:,:,0,:].mean(1) for a in atts],0).mean(0)   # readout row, mean heads+layers [B,N]
        return pred,attn
def train(tr,va,pd):
    m=Net(pd); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(tr["y"]))
        for i in range(0,len(perm),BS):
            idx=perm[i:i+BS]; opt.zero_grad(); lf(m(tr["flags"][idx],tr["pay"][idx]),tr["y"][idx]).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(va["flags"],va["pay"])-va["y"])**2).mean().item()/va["y"].var().item()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); return m,best
def run(modality):
    reseed(0)
    gfun=(mk_mlp(Kw),mk_mlp(Kw)) if modality=="structural" else None
    tr,st=gen(modality,G_TR,"train",gfun=gfun); va,_=gen(modality,G_VA,"train",st,gfun=gfun); me,_=gen(modality,G_ME,"test",st,gfun=gfun)
    m,vs=train(tr,va,me["pd"]); m.eval()
    with torch.no_grad(): yh=m(me["flags"],me["pay"]).numpy(); _,attn=m(me["flags"],me["pay"],want_attn=True); attn=attn.numpy()
    yv=me["y"].numpy(); absr=np.abs(yh-yv); G=len(yv)
    npool={}
    if modality=="structural":
        for g in range(min(3000,G_TR)):
            for j in range(1,N): npool.setdefault(int(tr["deg"][g,j]),[]).append(tr["pay"][g,j].numpy())
    def fwd(p2):
        with torch.no_grad(): return m(me["flags"],p2).numpy()
    def setp(j,vec): p2=me["pay"].clone(); p2[:,j,:]=vec; return p2
    pg=me["pay"].clone().requires_grad_(True); m(me["flags"],pg).sum().backward()
    func=pg.grad.norm(dim=-1).numpy()
    def draw(j,kk):
        if modality=="semantic": return torch.randn(G,1)
        keys=me["deg"][:,j]; out=np.zeros((G,Kw),np.float32)
        for key in np.unique(keys):
            cand=npool.get(int(key)) or [np.zeros(Kw,np.float32)]; idx=np.where(keys==key)[0]
            out[idx]=np.stack([cand[p] for p in np.random.randint(0,len(cand),size=len(idx))])
        return torch.tensor(out)
    benef=np.zeros((G,N))
    for j in range(1,N):
        acc=np.zeros(G)
        for kk in range(Ksw): acc+=np.abs(fwd(setp(j,draw(j,kk)))-yv)
        benef[:,j]=acc/Ksw-absr
    ts=1-((yh-yv)**2).mean()/me["y"].var().item()
    return dict(skill=ts,func=func,benef=benef,attn=attn,B=me["B"],Z=me["Z"],G=G)

print(("SMOKE " if SMOKE else "")+"dissertation carriage + attention baseline ...")
R={}
for mod in ["semantic","structural"]:
    t0=time.time(); R[mod]=run(mod)
    d=R[mod]; om=np.ones((d["G"],N),bool); om[:,0]=False; om[np.arange(d["G"]),d["B"]]=False; om[np.arange(d["G"]),d["Z"]]=False
    R[mod]["om"]=om
    bc=lambda M: M[np.arange(d["G"]),d["B"]].mean(); zc=lambda M: M[np.arange(d["G"]),d["Z"]].mean(); ic=lambda M:(M*om).sum()/om.sum()
    print(f"[{mod}] skill={d['skill']:.3f}  func C/S/I={bc(d['func']):.2f}/{zc(d['func']):.2f}/{ic(d['func']):.2f}  "
          f"benef C/S/I={bc(d['benef']):.3f}/{zc(d['benef']):.3f}/{ic(d['benef']):.3f}  attn C/S/I={bc(d['attn']):.2f}/{zc(d['attn']):.2f}/{ic(d['attn']):.2f}",flush=True)

def auroc(p,n):
    s=np.r_[p,n]; lb=np.r_[np.ones(len(p)),np.zeros(len(n))]; o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    return (r[lb==1].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n))
def auroc_ci(posg,negg,it=800):
    G=len(posg); base=auroc(posg,negg.reshape(-1)); mus=[]
    for _ in range(it):
        idx=np.random.randint(0,G,G); mus.append(auroc(posg[idx],negg[idx].reshape(-1)))
    return base,np.percentile(mus,2.5),np.percentile(mus,97.5)

# ---------- Figure 1: functional vs beneficial by class (population aggregated) ----------
def boot_ci(v,it=2000):
    v=np.asarray(v); idx=np.random.randint(0,len(v),(it,len(v))); mu=v[idx].mean(1); return v.mean(),np.percentile(mu,2.5),np.percentile(mu,97.5)
CLASSES=["Causal","Spurious","Irrelevant"]
METHODS=[("functional","func","#1f77b4"),("attention","attn","#2ca02c"),("beneficial","benef","#ff7f0e")]
fig,ax=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
for pi,mod in enumerate(["semantic","structural"]):
    d=R[mod]; G=d["G"]; om=d["om"]
    def cls(M): return {"Causal":M[np.arange(G),d["B"]],"Spurious":M[np.arange(G),d["Z"]],"Irrelevant":(M*om).sum(1)/om.sum(1)}
    x=np.arange(3); w=0.27
    for gi,(lab,key,color) in enumerate(METHODS):
        dat=cls(d[key]); norm=dat["Causal"].mean(); mus=[];los=[];his=[]
        for c in CLASSES: mu,lo,hi=boot_ci(dat[c]/norm); mus.append(mu);los.append(mu-lo);his.append(hi-mu)
        ax[pi].bar(x+(gi-1)*w,mus,w,yerr=[los,his],capsize=2,label=lab,color=color,edgecolor="k",linewidth=0.5)
    ax[pi].axhline(0,color="k",lw=0.8); ax[pi].set_xticks(x); ax[pi].set_xticklabels(CLASSES)
    ax[pi].set_title(f"{mod}  (skill {d['skill']:.2f})")
    if pi==0: ax[pi].set_ylabel("importance, normalised to Causal=1\n(mean $\\pm$95% CI over graphs)"); ax[pi].legend(fontsize=9)
fig.suptitle("Per-class carriage importance by method\n(population mean, each method normalised to its Causal mean; $\\pm$95% CI over graphs)")
fig.tight_layout(rect=[0,0,1,0.9]); fig.savefig("fig_diss_func_vs_benef.png"); print("saved fig_diss_func_vs_benef.png")

# ---------- Figure 2: faithfulness -- identify beneficial (B vs Irr) + flag spurious (Z vs Irr) ----------
fig,ax=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
for pi,mod in enumerate(["semantic","structural"]):
    d=R[mod]; G=d["G"]; om=d["om"]; irr=lambda M:(M*om).sum(1)/om.sum(1)
    DISC=[("identify beneficial\n(B vs Irrelevant)", lambda M:M[np.arange(G),d["B"]], irr),
          ("flag spurious  (lower=faithful)\n(Spurious vs Irrelevant)", lambda M:M[np.arange(G),d["Z"]], irr)]
    x=np.arange(len(METHODS)+1); w=0.38
    for di,(disc,posfn,negfn) in enumerate(DISC):
        mus=[];los=[];his=[]
        for _,key,_c in METHODS:
            M=d[key]; mu,lo,hi=auroc_ci(posfn(M),negfn(M)); mus.append(mu);los.append(mu-lo);his.append(hi-mu)
        mus.append(0.5);los.append(0);his.append(0)   # random
        ax[pi].bar(x+(di-0.5)*w,mus,w,yerr=[los,his],capsize=3,
                   color=["#1f77b4","#2ca02c","#ff7f0e","#999999"],
                   edgecolor="k",linewidth=0.5,alpha=0.62 if di==0 else 1.0,hatch=None if di==0 else '//')
    ax[pi].axhline(0.5,color="k",ls=":",lw=1); ax[pi].set_ylim(0.4,1.02)
    ax[pi].set_xticks(x); ax[pi].set_xticklabels([m[0] for m in METHODS]+["random"],rotation=15)
    ax[pi].set_title(f"{mod}")
    if pi==0: ax[pi].set_ylabel("AUROC (95% CI over graphs)")
ax[1].legend(handles=[Patch(facecolor="#bbb",edgecolor="k",label="identify beneficial (B vs Irrelevant)"),
                      Patch(facecolor="#bbb",edgecolor="k",hatch='//',label="flag spurious (Spurious vs Irrelevant)")],fontsize=8,loc="lower left")
fig.suptitle("Identifying the beneficial source (B vs Irrelevant) and flagging the spurious source (Spurious vs Irrelevant)\n"
             "AUROC per method; mean $\\pm$95% CI over graphs (dotted line = chance)")
fig.tight_layout(rect=[0,0,1,0.92]); fig.savefig("fig_diss_faithfulness.png"); print("saved fig_diss_faithfulness.png")
