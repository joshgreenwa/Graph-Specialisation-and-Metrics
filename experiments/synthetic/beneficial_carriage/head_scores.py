"""Per-head semantic/structural scores (repo-faithful, alpha-weighted, intervention-based)
for lightweight GRIT-like GTs (dense + 1-hop) on the semantic and structural synthetic tasks.

Model (mirrors grit_attention_components): node states = enc([role flags, content, node-RRWP]);
per head the attention logit = content (Q.K) + structural bias Linear(pair-RRWP), and the head
value output ho = attn.V (content value) + attn.pv (structural value), pv=Linear(pair-RRWP).
Dense = global attention; 1-hop = attention masked to bonded neighbours (+self).

Per-head score = ALPHA-WEIGHTED downstream delta under a single-node input intervention:
  semantic  = E_{node j, graphs} || d(ho) || and || d(attn) ||  under a CONTENT donor-swap of j
  structural= same under an RRWP (degree-matched donor) swap of j
The value delta ||d ho|| is alpha-weighted by construction (ho = attn.(V+pv) = routed value).
Independent axes -> non-degenerate scatter; scores both attention and value.
Outputs: (1) per-head heatmaps, (2) semantic-vs-structural scatter by layer, (3) scores vs
per-head functional/beneficial carriage (full-head ablation).
"""
import os, math, time, copy, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import paper_style
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=10,6,4,32,4,4
G_TR,G_VA=(2000,500) if SMOKE else (6000,1200)
G_ME=200 if SMOKE else 700
EPOCHS=4 if SMOKE else 30
BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS
MODELS=[("dense","semantic"),("1-hop","semantic"),("dense","structural"),("1-hop","structural")]

def rand_graph_adj(n):
    A=np.zeros((n,n),np.float32)
    for i in range(1,n): p=np.random.randint(0,i); A[i,p]=A[p,i]=1.0
    for _ in range(np.random.randint(0,n)):
        a,b=np.random.randint(0,n),np.random.randint(0,n)
        if a!=b: A[a,b]=A[b,a]=1.0
    return A
def rrwp(A):
    G=A.shape[0]; I=torch.eye(N).expand(G,N,N); Asl=A+I; M=Asl/Asl.sum(-1,keepdim=True)
    nd=[torch.ones(G,N)]; pr=[I.clone()]; cur=I.clone()
    for _ in range(1,Kw): cur=cur@M; nd.append(torch.diagonal(cur,dim1=1,dim2=2)); pr.append(cur.clone())
    return torch.stack(nd,-1), torch.stack(pr,-1)
def mk_mlp(d): return (np.random.randn(d,8).astype(np.float32)/np.sqrt(d),np.random.randn(8).astype(np.float32),np.random.randn(8,1).astype(np.float32)/np.sqrt(8))
def apply_g(x,g): W1,b1,W2=g; return (np.tanh(x@W1+b1)@W2)[...,0]
g_sem=mk_mlp(CV); g_str=mk_mlp(Kw)
def gen(task,G,ystat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); At=torch.tensor(A)
    nd,pr=rrwp(At); cont=np.random.randn(G,N,CV).astype(np.float32)
    B=np.random.randint(1,N,size=G)
    y=apply_g(cont[np.arange(G),B],g_sem) if task=="semantic" else apply_g(pr.numpy()[np.arange(G),0,B],g_str)
    if ystat is None: ystat=(y.mean(),y.std()+1e-6)
    y=(y-ystat[0])/ystat[1]
    flags=np.zeros((G,N,2),np.float32); flags[:,0,0]=1.0; flags[np.arange(G),B,1]=1.0
    return dict(feat=torch.tensor(np.concatenate([flags,cont],-1)),nd=nd,pr=pr,
                y=torch.tensor(y.astype(np.float32)),B=B,deg=A.sum(-1).astype(int),
                mask1=torch.tensor((A+np.eye(N))>0)),ystat

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.q=nn.Linear(H,H);s.k=nn.Linear(H,H);s.v=nn.Linear(H,H);s.o=nn.Linear(H,H)
        s.bb=nn.Linear(Kw,HEADS); s.pv=nn.Linear(Kw,H)
        s.n1=nn.LayerNorm(H);s.n2=nn.LayerNorm(H);s.mlp=nn.Sequential(nn.Linear(H,2*H),nn.GELU(),nn.Linear(2*H,H))
    def forward(s,x,pr,mask,ablate=None,cap=False):
        B=x.size(0);xn=s.n1(x)
        q=s.q(xn).view(B,N,HEADS,dh).transpose(1,2);k=s.k(xn).view(B,N,HEADS,dh).transpose(1,2);v=s.v(xn).view(B,N,HEADS,dh).transpose(1,2)
        a=((q@k.transpose(-2,-1))/math.sqrt(dh)+s.bb(pr).permute(0,3,1,2)).masked_fill(~mask,float("-inf")).softmax(-1)
        cv=a@v
        sv=torch.einsum('bhij,bhijd->bhid',a,s.pv(pr).view(B,N,N,HEADS,dh).permute(0,3,1,2,4))
        if ablate is not None:
            head,mode=ablate; cv=cv.clone(); sv=sv.clone()
            if mode in (0,1): cv[:,head]=0.0
            if mode in (0,2): sv[:,head]=0.0
        ho=cv+sv
        x=x+s.o(ho.transpose(1,2).reshape(B,N,H)); x=x+s.mlp(s.n2(x))
        if cap: return x,(a.detach(), ho.detach())     # per-graph attn [B,HEADS,N,N], head value [B,HEADS,N,dh]
        return x
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=nn.Linear(2+CV+Kw,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
    def forward(s,feat,nd,pr,mask,ablate=None,cap=False):
        x=s.enc(torch.cat([feat,nd],-1)); caps=[]
        for li,b in enumerate(s.blocks):
            ab=(ablate[1],ablate[2]) if (ablate and ablate[0]==li) else None
            if cap: x,c=b(x,pr,mask,cap=True); caps.append(c)
            else:   x=b(x,pr,mask,ablate=ab)
        pred=s.head(x[:,0]).squeeze(-1)
        return (pred,caps) if cap else pred

def run(variant,task):
    torch.manual_seed(0); np.random.seed(1 if task=="structural" else 0)
    tr,st=gen(task,G_TR); va,_=gen(task,G_VA,st); me,_=gen(task,G_ME,st)
    dense=(variant=="dense")
    def MK(d): return torch.ones(len(d["y"]),1,N,N,dtype=torch.bool) if dense else d["mask1"][:,None]
    m=Net(); opt=torch.optim.Adam(m.parameters(),lr=LR,weight_decay=WD); lf=nn.MSELoss(); best=-1e9; bs=None
    for ep in range(EPOCHS):
        m.train(); perm=torch.randperm(len(tr["y"]))
        for i in range(0,len(perm),BS):
            idx=perm[i:i+BS]; opt.zero_grad()
            lf(m(tr["feat"][idx],tr["nd"][idx],tr["pr"][idx],MK({"y":tr["y"][idx],"mask1":tr["mask1"][idx]})),tr["y"][idx]).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): vs=1-((m(va["feat"],va["nd"],va["pr"],MK(va))-va["y"])**2).mean().item()/va["y"].var().item()
        if vs>best: best=vs; bs=copy.deepcopy(m.state_dict())
    m.load_state_dict(bs); m.eval(); Mme=MK(me); G=len(me["y"])
    # donor pools for the structural (RRWP) swap
    npool={}; ppool={}
    for g in range(min(2500,G_TR)):
        for j in range(1,N): npool.setdefault(int(tr["deg"][g,j]),[]).append(tr["nd"][g,j].numpy()); ppool.setdefault(int(tr["deg"][g,j]),[]).append(tr["pr"][g,0,j].numpy())
    def capf(feat,nd,pr):
        with torch.no_grad(): pred,caps=m(feat,nd,pr,Mme,cap=True)
        return pred.numpy(),caps
    yh,caps0=capf(me["feat"],me["nd"],me["pr"]); yv=me["y"].numpy(); absr=np.abs(yh-yv)
    def delta(caps):
        vd=np.zeros((L,HEADS)); ad=np.zeros((L,HEADS))
        for l in range(L):
            a0,h0=caps0[l]; a1,h1=caps[l]
            vd[l]=(h1-h0).norm(dim=-1).mean((0,2)).numpy()                 # alpha-weighted value-vector delta
            ad[l]=(a1-a0).reshape(G,HEADS,-1).norm(dim=-1).mean(0).numpy() # attention delta
        return vd,ad
    def content_swap(j):
        f2=me["feat"].clone(); f2[:,j,2:2+CV]=torch.randn(G,CV); return f2,me["nd"],me["pr"]
    def struct_swap(j):
        nd2=me["nd"].clone(); pr2=me["pr"].clone(); keys=me["deg"][:,j]
        ndn=np.zeros((G,Kw),np.float32); prn=np.zeros((G,Kw),np.float32)
        for key in np.unique(keys):
            idx=np.where(keys==key)[0]; cn=npool.get(int(key)) or [nd2[0,j].numpy()]; cp=ppool.get(int(key)) or [pr2[0,0,j].numpy()]
            pick=np.random.randint(0,len(cn),size=len(idx))
            ndn[idx]=np.stack([cn[p] for p in pick]); prn[idx]=np.stack([cp[p] for p in pick])
        nd2[:,j]=torch.tensor(ndn); pr2[:,0,j]=torch.tensor(prn); pr2[:,j,0]=torch.tensor(prn)
        return me["feat"],nd2,pr2
    sem_v=np.zeros((L,HEADS)); sem_a=np.zeros((L,HEADS)); str_v=np.zeros((L,HEADS)); str_a=np.zeros((L,HEADS))
    for j in range(1,N):
        _,cc=capf(*content_swap(j)); vd,ad=delta(cc); sem_v+=vd; sem_a+=ad
        _,cs=capf(*struct_swap(j)); vd,ad=delta(cs); str_v+=vd; str_a+=ad
    sem_v/=(N-1); sem_a/=(N-1); str_v/=(N-1); str_a/=(N-1)
    # ---- Method 1: content-transposition equivariance (semantic) / invariance (structural) ----
    # Transpose content of a node pair (i,j) at input; per query row look at the 2-vector of attention
    # RECEIVED by (i,j): before p=(a,b), after p'=(ap,bp), swap-target p^swap=(b,a).  Two INDEPENDENT
    # agreements (not a single fraction) -> genuine 2-D scatter, and a uniform head scores BOTH = 1:
    #   equivariance (semantic)   = 1 - ||p'-p^swap|| / (||p^swap-p|| + f)   (1 = attention swapped)
    #   invariance   (structural) = 1 - ||p'-p||      / (||p^swap-p|| + f)   (1 = attention unchanged)
    # per query, clamp to [0,1], then average over queries/samples/graphs.  RAW = alpha-weighted (w=a+b);
    # CENTRED = differential-weighted (w=|a-b|), which drops uniform-background queries (that trivially
    # score both=1) so only selectively-routing queries survive.
    S=6 if SMOKE else 40; ar=torch.arange(G); Hh=torch.arange(HEADS); Nn=torch.arange(N); F=0.02
    def col(A,c): return A[ar[:,None,None],Hh[None,:,None],Nn[None,None,:],c[:,None,None]]  # [G,heads,N(query)]
    # SAME transposition, but the equivariance/invariance can be read off the ATTENTION matrix (a) OR the
    # head VALUE output ho=attn.(V+pv) at the swapped nodes.  For ho the swap-target is (ho_j, ho_i): a head
    # whose OUTPUT swaps with content is content-carrying (semantic); one whose output is unchanged is
    # content-blind (structural).  NB V itself swaps under the content swap, so value-equivariance is the
    # expected behaviour of any content-carrying head -> directly comparable to Method 2's ||d ho||.
    eqv=np.zeros((L,HEADS)); inv=np.zeros((L,HEADS)); wr=np.zeros((L,HEADS))     # ATTENTION raw (alpha-weighted)
    eqvc=np.zeros((L,HEADS)); invc=np.zeros((L,HEADS)); wc=np.zeros((L,HEADS))   # ATTENTION centred (diff-weighted)
    veq=np.zeros((L,HEADS)); viv=np.zeros((L,HEADS)); vwr=np.zeros((L,HEADS))    # VALUE raw
    veqc=np.zeros((L,HEADS)); vivc=np.zeros((L,HEADS)); vwc=np.zeros((L,HEADS))  # VALUE centred
    for _ in range(S):
        ii=np.random.randint(1,N,size=G); jj=np.array([np.random.choice([x for x in range(1,N) if x!=ii[g]]) for g in range(G)])
        it=torch.tensor(ii); jt=torch.tensor(jj); f2=me["feat"].clone()
        ci=f2[ar,it,2:2+CV].clone(); f2[ar,it,2:2+CV]=f2[ar,jt,2:2+CV]; f2[ar,jt,2:2+CV]=ci
        _,capsT=capf(f2,me["nd"],me["pr"])
        for l in range(L):
            # -- attention channel (per query row) --
            a=col(caps0[l][0],it); b=col(caps0[l][0],jt); ap=col(capsT[l][0],it); bp=col(capsT[l][0],jt)
            e=torch.sqrt(2*(a-b)**2)                                # ||p^swap - p||  (swap displacement)
            d_eq=torch.sqrt((ap-b)**2+(bp-a)**2)                    # ||p' - p^swap|| (distance to swapped)
            d_in=torch.sqrt((ap-a)**2+(bp-b)**2)                    # ||p' - p||      (distance to original)
            eq=(1-d_eq/(e+F)).clamp(0,1); iv=(1-d_in/(e+F)).clamp(0,1)   # INDEPENDENT, each in [0,1]
            wa=a+b; wd=(a-b).abs()                                  # alpha weight  /  differential weight
            eqv[l]+=(wa*eq).sum((0,2)).numpy(); inv[l]+=(wa*iv).sum((0,2)).numpy(); wr[l]+=wa.sum((0,2)).numpy()
            eqvc[l]+=(wd*eq).sum((0,2)).numpy(); invc[l]+=(wd*iv).sum((0,2)).numpy(); wc[l]+=wd.sum((0,2)).numpy()
            # -- value channel (per node output ho, dh-dim vector at i,j) --
            hi=caps0[l][1][ar,:,it]; hj=caps0[l][1][ar,:,jt]        # [G,heads,dh] clean output at i,j
            hi2=capsT[l][1][ar,:,it]; hj2=capsT[l][1][ar,:,jt]      # after swap
            ev=torch.sqrt(((hj-hi)**2).sum(-1)+((hi-hj)**2).sum(-1))    # ||p^swap - p||  [G,heads]
            dqv=torch.sqrt(((hi2-hj)**2).sum(-1)+((hj2-hi)**2).sum(-1)) # to swapped target
            div=torch.sqrt(((hi2-hi)**2).sum(-1)+((hj2-hj)**2).sum(-1)) # to original
            Fv=0.1*ev.mean().clamp(min=1e-6)                        # scale-adaptive floor (value norms vary by head)
            eqV=(1-dqv/(ev+Fv)).clamp(0,1); ivV=(1-div/(ev+Fv)).clamp(0,1)
            wav=hi.norm(dim=-1)+hj.norm(dim=-1); wdv=(hi-hj).norm(dim=-1)   # value-magnitude / value-differential weight
            veq[l]+=(wav*eqV).sum(0).numpy(); viv[l]+=(wav*ivV).sum(0).numpy(); vwr[l]+=wav.sum(0).numpy()
            veqc[l]+=(wdv*eqV).sum(0).numpy(); vivc[l]+=(wdv*ivV).sum(0).numpy(); vwc[l]+=wdv.sum(0).numpy()
    sem1=eqv/(wr+1e-9); str1=inv/(wr+1e-9)                   # ATTENTION raw: independent equivariance / invariance
    sem1c=eqvc/(wc+1e-9); str1c=invc/(wc+1e-9)               # ATTENTION centred: differential-weighted
    semV=veq/(vwr+1e-9); strV=viv/(vwr+1e-9)                 # VALUE raw
    semVc=veqc/(vwc+1e-9); strVc=vivc/(vwc+1e-9)             # VALUE centred

    # per-head functional / beneficial carriage (full-head ablation)
    imp=np.zeros((L,HEADS)); ben=np.zeros((L,HEADS))
    with torch.no_grad():
        for l in range(L):
            for h in range(HEADS):
                yp=m(me["feat"],me["nd"],me["pr"],Mme,ablate=(l,h,0)).numpy()
                imp[l,h]=np.abs(yp-yh).mean(); ben[l,h]=(np.abs(yp-yv)-absr).mean()
    skill=1-((yh-yv)**2).mean()/me["y"].var().item()
    return dict(skill=skill,sem_v=sem_v,str_v=str_v,sem_a=sem_a,str_a=str_a,imp=imp,ben=ben,
                sem1=sem1,str1=str1,sem1c=sem1c,str1c=str1c,
                semV=semV,strV=strV,semVc=semVc,strVc=strVc)

print(("SMOKE " if SMOKE else "")+"per-head alpha-weighted semantic/structural scores ...")
R={}
for v,t in MODELS:
    t0=time.time(); R[(v,t)]=run(v,t); d=R[(v,t)]
    print(f"[{v}/{t}] skill={d['skill']:.3f}  M2 str_v={d['str_v'].mean():.3f}(sem {d['sem_v'].mean():.3f})  "
          f"M1 struct(invariance)={d['str1'].mean():.2f}  ({time.time()-t0:.0f}s)",flush=True)

# normalise each channel by its GLOBAL mean (content perturbations are inherently larger),
# so the semantic/structural comparison is on a common scale.
gsem=float(np.mean([R[m]["sem_v"] for m in R])); gstr=float(np.mean([R[m]["str_v"] for m in R]))
def SN(d): return d["sem_v"]/(gsem+1e-9)      # VALUE  · separate: semantic (content-swap ||d ho||, norm.)
def TN(d): return d["str_v"]/(gstr+1e-9)      # VALUE  · separate: structural (RRWP-swap ||d ho||, norm.)
def SF(d): return TN(d)/(SN(d)+TN(d)+1e-9)
gsem_a=float(np.mean([R[m]["sem_a"] for m in R])); gstr_a=float(np.mean([R[m]["str_a"] for m in R]))
def SNa(d): return d["sem_a"]/(gsem_a+1e-9)   # ATTN   · separate: semantic (content-swap ||d alpha||, norm.)
def TNa(d): return d["str_a"]/(gstr_a+1e-9)   # ATTN   · separate: structural (RRWP-swap ||d alpha||, norm.)

# ---------------- (1) heatmaps: structural fraction (value) per head ----------------
fig,ax=plt.subplots(2,2,figsize=(9.6,7.8),constrained_layout=True)
for (v,t),axi in zip(MODELS,ax.ravel()):
    d=R[(v,t)]; frac=SF(d)
    im=axi.imshow(frac,cmap="coolwarm",vmin=0,vmax=1,aspect="auto")
    axi.set_title(f"{v} / {t}  (skill {d['skill']:.2f})"); axi.set_xlabel("head"); axi.set_ylabel("layer")
    axi.set_xticks(range(HEADS)); axi.set_yticks(range(L))
    for l in range(L):
        for h in range(HEADS): axi.text(h,l,f"{frac[l,h]:.2f}",ha="center",va="center",fontsize=7)
fig.colorbar(im,ax=ax,fraction=0.04,label="structural value fraction  (0.5=balanced; scale-normalised)")
fig.suptitle("Per-head structural fraction of the value vector (content vs RRWP swap deltas, each scale-normalised)")
fig.savefig("fig_head_heatmap.png",dpi=140); print("saved fig_head_heatmap.png")

# ---------------- (2) scatter: semantic vs structural value score, by layer ----------------
fig,ax=plt.subplots(2,2,figsize=(9,7.5)); cmap=plt.cm.viridis(np.linspace(0,0.85,L))
for (v,t),axi in zip(MODELS,ax.ravel()):
    d=R[(v,t)]; xs=SN(d); ys=TN(d)
    for l in range(L): axi.scatter(xs[l],ys[l],color=cmap[l],s=40,edgecolors="k",linewidths=0.4,label=f"layer {l}")
    mx=max(xs.max(),ys.max())*1.05+1e-6; axi.plot([0,mx],[0,mx],"k:",lw=0.7)
    axi.set_title(f"{v} / {t}"); axi.set_xlabel("semantic score (content-swap, norm.)"); axi.set_ylabel("structural score (RRWP-swap, norm.)")
ax[0,0].legend(fontsize=7)
fig.suptitle("Per-head semantic vs structural score (alpha-weighted value-vector deltas, scale-normalised), by layer")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig("fig_head_scatter.png",dpi=140); print("saved fig_head_scatter.png")

# ---------------- (3) relate scores to functional/beneficial carriage ----------------
fig,ax=plt.subplots(1,2,figsize=(11,4.6))
for pi,(metric,lab) in enumerate([("imp","functional carriage  |$\\Delta\\hat y$| (head ablation)"),("ben","beneficial carriage  (loss change, head ablation)")]):
    for (v,t) in MODELS:
        d=R[(v,t)]; frac=SF(d).ravel()
        mk="o" if v=="dense" else "^"; col="#d62728" if t=="structural" else "#1f77b4"
        ax[pi].scatter(frac,d[metric].ravel(),marker=mk,color=col,s=45,edgecolors="k",linewidths=0.4,label=f"{v}/{t}",alpha=0.85)
    ax[pi].set_xlabel("head structural fraction"); ax[pi].set_ylabel(lab); ax[pi].axhline(0,color="k",lw=0.5)
ax[0].legend(fontsize=7)
fig.suptitle("Per-head structural fraction vs its functional & beneficial carriage\n(structural task=red, semantic task=blue; dense=circle, 1-hop=triangle)")
fig.tight_layout(rect=[0,0,1,0.9]); fig.savefig("fig_head_carriage.png",dpi=140); print("saved fig_head_carriage.png")

print("\ncorr(structural-value-fraction, functional carriage) per model:")
for (v,t) in MODELS:
    d=R[(v,t)]; frac=SF(d).ravel()
    print(f"  {v}/{t}: r={np.corrcoef(frac,d['imp'].ravel())[0,1]:+.2f}   mean struct-frac(norm)={frac.mean():.2f}")

# ---------------- (4) Method-1 heatmap: structural (invariance) fraction per head ----------------
fig,ax=plt.subplots(2,2,figsize=(9.6,7.8),constrained_layout=True)
for (v,t),axi in zip(MODELS,ax.ravel()):
    d=R[(v,t)]; im=axi.imshow(d["str1"],cmap="coolwarm",vmin=0,vmax=1,aspect="auto")
    axi.set_title(f"{v} / {t}  (skill {d['skill']:.2f})"); axi.set_xlabel("head"); axi.set_ylabel("layer")
    axi.set_xticks(range(HEADS)); axi.set_yticks(range(L))
    for l in range(L):
        for h in range(HEADS): axi.text(h,l,f"{d['str1'][l,h]:.2f}",ha="center",va="center",fontsize=7)
fig.colorbar(im,ax=ax,fraction=0.04,label="structural (invariance) fraction  (0=semantic, 1=structural)")
fig.suptitle("Method 1: per-head structural fraction via content transposition (alpha-weighted attention invariance)")
fig.savefig("fig_head_heatmap_equiv.png",dpi=140); print("saved fig_head_heatmap_equiv.png")

# ---------------- (5) compare Method 1 (raw & centred) vs Method 2 ----------------
fig,ax=plt.subplots(1,2,figsize=(11.5,5.4),sharey=True)
for pi,(xk,xlab) in enumerate([("str1","Method 1 RAW: structural fraction (transposition invariance)"),
                                ("str1c","Method 1 CENTRED: structural fraction (background removed)")]):
    ax_=ax[pi]; ax2=[]; ay2=[]
    for (v,t) in MODELS:
        d=R[(v,t)]; x=d[xk].ravel(); y=SF(d).ravel()
        mk="o" if v=="dense" else "^"; col="#d62728" if t=="structural" else "#1f77b4"
        ax_.scatter(x,y,marker=mk,color=col,s=42,edgecolors="k",linewidths=0.4,alpha=0.85,label=f"{v}/{t}")
        ax2+=list(x); ay2+=list(y)
    rr=np.corrcoef(ax2,ay2)[0,1]; ax_.set_xlabel(xlab); ax_.set_title(f"vs Method 2   (r={rr:+.2f})"); ax_.grid(alpha=0.3); ax_.set_xlim(0,1)
    if pi==0: ax_.set_ylabel("Method 2: structural fraction (RRWP-swap sensitivity)"); ax_.legend(fontsize=8)
fig.suptitle("Per-head structural scores: content-transposition (raw vs centred) against separate-intervention value sensitivity")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("fig_head_method_compare.png",dpi=140); print("saved fig_head_method_compare.png")

# ---------------- (6) three-method scatter: structural (x) vs semantic (y) per head ----------------
# separate intervention = independent axes; the two transposition methods give a single fraction,
# so semantic = 1 - structural (points lie on the anti-diagonal by construction).
def two_scores(d,method):
    if method=="sep":   return TN(d).ravel(),   SN(d).ravel()     # structural, semantic (scale-norm. value sensitivity)
    if method=="trans": return d["str1"].ravel(),  d["sem1"].ravel()    # invariance, equivariance (independent)
    return d["str1c"].ravel(), d["sem1c"].ravel()                       # centred invariance, equivariance (independent)
panels=[("sep","separate intervention"),("trans","node transposition"),("cent","centred node transposition")]
fig,ax=plt.subplots(1,3,figsize=(15,5.2))
for pi,(mkey,title) in enumerate(panels):
    axi=ax[pi]; allx=[]; ally=[]
    for (v,t) in MODELS:
        d=R[(v,t)]; xs,ys=two_scores(d,mkey)
        marker="o" if v=="dense" else "^"; c="#d62728" if t=="structural" else "#1f77b4"
        axi.scatter(xs,ys,marker=marker,color=c,s=42,edgecolors="k",linewidths=0.4,alpha=0.85,label=f"{v}/{t}")
        allx+=list(xs); ally+=list(ys)
    axi.set_title(title); axi.set_xlabel("structural score"); axi.grid(alpha=0.3)
    if mkey=="sep":
        mx=max(max(allx),max(ally))*1.05+1e-6; axi.plot([0,mx],[0,mx],"k:",lw=0.7); axi.set_xlim(0,mx); axi.set_ylim(0,mx)
    else:  # equivariance/invariance are independent -> genuine 2-D; y=x = equal semantic/structural, (1,1)=undifferentiated
        axi.plot([0,1],[0,1],"k:",lw=0.7); axi.set_xlim(-0.02,1.02); axi.set_ylim(-0.02,1.02)
ax[0].set_ylabel("semantic score"); ax[0].legend(fontsize=8)
fig.suptitle("Per-head structural vs semantic score under three scoring methods (structural task=red, semantic task=blue; dense=circle, 1-hop=triangle)")
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig("fig_head_three_methods.png",dpi=140); print("saved fig_head_three_methods.png")

print("\nper-model structural-fraction means and correlations with Method 2:")
print(f"  {'model':<18}{'M1 raw':>8}{'M1 centred':>12}{'M2':>7}{'r(raw,M2)':>11}{'r(cen,M2)':>11}")
for (v,t) in MODELS:
    d=R[(v,t)]; m2=SF(d).ravel()
    print(f"  {v+'/'+t:<18}{d['str1'].mean():>8.2f}{d['str1c'].mean():>12.2f}{SF(d).mean():>7.2f}"
          f"{np.corrcoef(d['str1'].ravel(),m2)[0,1]:>+11.2f}{np.corrcoef(d['str1c'].ravel(),m2)[0,1]:>+11.2f}")

MODEL_STYLE=lambda v,t:("o" if v=="dense" else "^","#d62728" if t=="structural" else "#1f77b4")

# ---------------- (7) NODE TRANSPOSITION: attention channel vs value channel ----------------
# equivariance (semantic) / invariance (structural) of alpha  vs  of the head output ho, each in [0,1].
fig,ax=plt.subplots(1,2,figsize=(11,5.2))
for pi,(sk,ek,title) in enumerate([("str1","sem1","attention  ·  node transposition"),
                                    ("strV","semV","value  ·  node transposition")]):
    axi=ax[pi]
    for (v,t) in MODELS:
        d=R[(v,t)]; mkr,c=MODEL_STYLE(v,t)
        axi.scatter(d[sk].ravel(),d[ek].ravel(),marker=mkr,color=c,s=42,edgecolors="k",linewidths=0.4,alpha=0.85,label=f"{v}/{t}")
    axi.plot([0,1],[0,1],"k:",lw=0.7); axi.set_xlim(-0.02,1.02); axi.set_ylim(-0.02,1.02)
    axi.set_title(title); axi.set_xlabel("structural score (invariance)"); axi.grid(alpha=0.3)
    if pi==0: axi.set_ylabel("semantic score (equivariance)"); axi.legend(fontsize=8)
fig.suptitle("Node transposition, scored on the attention vs the value vector (structural task=red, semantic task=blue; dense=circle, 1-hop=triangle)")
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig("fig_head_transposition_value.png",dpi=140); print("saved fig_head_transposition_value.png")

# ---------------- (8) SEPARATE INTERVENTION: attention channel vs value channel ----------------
# scale-normalised sensitivity ||d alpha|| (attention) or ||d ho|| (value) to a single-node content
# donor-swap (semantic) vs a degree-matched RRWP donor-swap (structural).
fig,ax=plt.subplots(1,2,figsize=(11,5.2))
sep_scores=[("attention  ·  separate intervention",TNa,SNa),("value  ·  separate intervention",TN,SN)]
for pi,(title,Xf,Yf) in enumerate(sep_scores):
    axi=ax[pi]; mx=0.0
    for (v,t) in MODELS:
        d=R[(v,t)]; mkr,c=MODEL_STYLE(v,t); xs=Xf(d).ravel(); ys=Yf(d).ravel(); mx=max(mx,xs.max(),ys.max())
        axi.scatter(xs,ys,marker=mkr,color=c,s=42,edgecolors="k",linewidths=0.4,alpha=0.85,label=f"{v}/{t}")
    mx*=1.06; axi.plot([0,mx],[0,mx],"k:",lw=0.7); axi.set_xlim(0,mx); axi.set_ylim(0,mx)
    axi.set_title(title); axi.set_xlabel("structural score (RRWP-swap sensitivity, norm.)"); axi.grid(alpha=0.3)
    if pi==0: axi.set_ylabel("semantic score (content-swap sensitivity, norm.)"); axi.legend(fontsize=8)
fig.suptitle("Separate single-node donor intervention, scored on the attention vs the value vector (structural task=red, semantic task=blue; dense=circle, 1-hop=triangle)")
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig("fig_head_separate_value.png",dpi=140); print("saved fig_head_separate_value.png")

print("\nnode-transposition structural score, ATTENTION channel vs VALUE channel (and value-vs-Method 2):")
print(f"  {'model':<18}{'attn str':>9}{'val str':>9}{'val cen':>9}{'r(attn,val)':>12}{'r(val,M2)':>11}")
for (v,t) in MODELS:
    d=R[(v,t)]; m2=SF(d).ravel()
    print(f"  {v+'/'+t:<18}{d['str1'].mean():>9.2f}{d['strV'].mean():>9.2f}{d['strVc'].mean():>9.2f}"
          f"{np.corrcoef(d['str1'].ravel(),d['strV'].ravel())[0,1]:>+12.2f}{np.corrcoef(d['strV'].ravel(),m2)[0,1]:>+11.2f}")

# ---------------- (9) per-head FUNCTIONAL & BENEFICIAL carriage on each task ----------------
# functional carriage = |dy_hat| when the head is ablated (how much the head's output is USED).
# beneficial carriage = change in |prediction error| when ablated (>0 => ablation hurts => head HELPS).
fig,ax=plt.subplots(2,4,figsize=(15,7.4),constrained_layout=True)
vI=max(R[m]["imp"].max() for m in R); vB=max(np.abs(R[m]["ben"]).max() for m in R)
for ci,(v,t) in enumerate(MODELS):
    d=R[(v,t)]
    im0=ax[0,ci].imshow(d["imp"],cmap="viridis",vmin=0,vmax=vI,aspect="auto")
    im1=ax[1,ci].imshow(d["ben"],cmap="coolwarm",vmin=-vB,vmax=vB,aspect="auto")
    ax[0,ci].set_title(f"{v} / {t}   (skill {d['skill']:.3f})")
    for r,M in [(0,"imp"),(1,"ben")]:
        ax[r,ci].set_xticks(range(HEADS)); ax[r,ci].set_yticks(range(L))
        for l in range(L):
            for h in range(HEADS): ax[r,ci].text(h,l,f"{d[M][l,h]:.2g}",ha="center",va="center",fontsize=6)
    ax[1,ci].set_xlabel("head")
ax[0,0].set_ylabel("FUNCTIONAL carriage\n|$\\Delta\\hat y$|   (layer)"); ax[1,0].set_ylabel("BENEFICIAL carriage\n$\\Delta$|error|   (layer)")
fig.colorbar(im0,ax=ax[0,:],fraction=0.02,pad=0.01,label="functional  |$\\Delta\\hat y$|  (usage)")
fig.colorbar(im1,ax=ax[1,:],fraction=0.02,pad=0.01,label="beneficial  $\\Delta$|error|  ( >0 = head helps )")
fig.suptitle("Per-head functional and beneficial carriage on each synthetic task (full-head ablation)")
fig.savefig("fig_head_carriage_maps.png",dpi=140); print("saved fig_head_carriage_maps.png")
