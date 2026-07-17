"""Diagnostic: is the structural-carriage dense/1-hop gap a REPRESENTATION effect (not task-intrinsic)?

Both models fit label=g_str(pr[0,B]) ~perfectly. We corrupt the TARGET node B's structure two ways
(degree-matched donor), and report the resulting skill drop for each model:
  (b) DIRECT encoding : swap only pr[0,B] (readout-pair-RRWP) + nd[B]   <- what dense reads globally
  (c) FULL footprint  : swap nd[B] + entire pr row & col of B          <- also hits the 1-hop path
Prediction if representation-dependent: dense breaks under BOTH; 1-hop survives (b) but breaks under (c).
"""
import os, math, copy, numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
SMOKE=os.environ.get("SMOKE","0")=="1"
N,Kw,CV,H,HEADS,L=10,6,4,32,4,4
G_TR,G_VA,G_ME=(2000,500,300) if SMOKE else (6000,1200,1500)
EPOCHS=4 if SMOKE else 30; BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS

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
    nd,pr=rrwp(At); cont=np.random.randn(G,N,CV).astype(np.float32); B=np.random.randint(1,N,size=G)
    y=apply_g(pr.numpy()[np.arange(G),0,B],g_str)
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
    def forward(s,x,pr,mask):
        B=x.size(0);xn=s.n1(x)
        q=s.q(xn).view(B,N,HEADS,dh).transpose(1,2);k=s.k(xn).view(B,N,HEADS,dh).transpose(1,2);v=s.v(xn).view(B,N,HEADS,dh).transpose(1,2)
        a=((q@k.transpose(-2,-1))/math.sqrt(dh)+s.bb(pr).permute(0,3,1,2)).masked_fill(~mask,float("-inf")).softmax(-1)
        cv=a@v; sv=torch.einsum('bhij,bhijd->bhid',a,s.pv(pr).view(B,N,N,HEADS,dh).permute(0,3,1,2,4))
        ho=cv+sv; x=x+s.o(ho.transpose(1,2).reshape(B,N,H)); x=x+s.mlp(s.n2(x)); return x
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=nn.Linear(2+CV+Kw,H); s.blocks=nn.ModuleList([Block() for _ in range(L)]); s.head=nn.Linear(H,1)
    def forward(s,feat,nd,pr,mask):
        x=s.enc(torch.cat([feat,nd],-1))
        for b in s.blocks: x=b(x,pr,mask)
        return s.head(x[:,0]).squeeze(-1)

def train(variant):
    torch.manual_seed(0); np.random.seed(1)
    tr,st=gen("structural",G_TR); va,_=gen("structural",G_VA,st); me,_=gen("structural",G_ME,st)
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
    m.load_state_dict(bs); m.eval(); return m,me,MK(me)

def skill(m,feat,nd,pr,mask,y):
    with torch.no_grad(): return 1-((m(feat,nd,pr,mask)-torch.tensor(y))**2).mean().item()/y.var()

for variant in ["dense","1-hop"]:
    m,me,mask=train(variant); G=len(me["y"]); y=me["y"].numpy(); deg=me["deg"]; Bt=me["B"]
    base=skill(m,me["feat"],me["nd"],me["pr"],mask,y)
    # degree-matched donor banks (node-RRWP, full pair-RRWP row) keyed by degree
    ndp={}; prrow={}
    for g in range(G):
        for j in range(1,N):
            ndp.setdefault(int(deg[g,j]),[]).append((g,j));
    rng=np.random.default_rng(0); ar=np.arange(G)
    # (b) DIRECT: swap pr[0,B]+nd[B]
    ndB=me["nd"].clone(); prB=me["pr"].clone()
    for g in range(G):
        b=Bt[g]; pool=ndp.get(int(deg[g,b])) or [(g,b)]
        dg,dj=pool[rng.integers(0,len(pool))]
        ndB[g,b]=me["nd"][dg,dj]; prB[g,0,b]=me["pr"][dg,0,dj]; prB[g,b,0]=me["pr"][dg,0,dj]
    sk_b=skill(m,me["feat"],ndB,prB,mask,y)
    # (c) FULL: swap nd[B] + entire pr row & col of B (donor node's row, aligned by index)
    ndC=me["nd"].clone(); prC=me["pr"].clone()
    for g in range(G):
        b=Bt[g]; pool=ndp.get(int(deg[g,b])) or [(g,b)]
        dg,dj=pool[rng.integers(0,len(pool))]
        ndC[g,b]=me["nd"][dg,dj]
        prC[g,b,:]=me["pr"][dg,dj,:]; prC[g,:,b]=me["pr"][dg,:,dj]
    sk_c=skill(m,me["feat"],ndC,prC,mask,y)
    print(f"[{variant}] baseline skill={base:.3f}   after DIRECT swap(pr[0,B]+nd[B])={sk_b:.3f}   "
          f"after FULL swap(nd[B]+pr row/col of B)={sk_c:.3f}",flush=True)
