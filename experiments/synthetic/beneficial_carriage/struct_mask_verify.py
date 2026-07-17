"""Verify 1-hop solves the structural task via the MASK (adjacency), not the RRWP values.
Strip ALL random-walk info from RRWP inputs (keep only the trivial 0-step: pr->I, nd->[1,0,..]),
globally, at test time. If 1-hop retains skill (mask=adjacency carries topology) while dense collapses
(all-ones mask carries no topology), the mask-learning hypothesis is confirmed.
"""
import os, math, copy, numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
N,Kw,CV,H,HEADS,L=10,6,4,32,4,4
G_TR,G_VA,G_ME=6000,1200,1500; EPOCHS=30; BS,LR,WD=256,1e-3,1e-4; dh=H//HEADS
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
g_str=mk_mlp(Kw)
def gen(G,ystat=None):
    A=np.stack([rand_graph_adj(N) for _ in range(G)]); At=torch.tensor(A)
    nd,pr=rrwp(At); cont=np.random.randn(G,N,CV).astype(np.float32); B=np.random.randint(1,N,size=G)
    y=apply_g(pr.numpy()[np.arange(G),0,B],g_str)
    if ystat is None: ystat=(y.mean(),y.std()+1e-6)
    y=(y-ystat[0])/ystat[1]
    flags=np.zeros((G,N,2),np.float32); flags[:,0,0]=1.0; flags[np.arange(G),B,1]=1.0
    return dict(feat=torch.tensor(np.concatenate([flags,cont],-1)),nd=nd,pr=pr,
                y=torch.tensor(y.astype(np.float32)),mask1=torch.tensor((A+np.eye(N))>0)),ystat
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
def strip_rw(nd,pr):  # keep only 0-step: nd->[1,0,..], pr->I  (all random-walk info removed)
    nd2=torch.zeros_like(nd); nd2[...,0]=1.0
    pr2=torch.zeros_like(pr); pr2[...,0]=torch.eye(N)
    return nd2,pr2
def train_eval(variant):
    torch.manual_seed(0); np.random.seed(1)
    tr,st=gen(G_TR); va,_=gen(G_VA,st); me,_=gen(G_ME,st); dense=(variant=="dense")
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
    m.load_state_dict(bs); m.eval(); mask=MK(me); yv=me["y"].numpy()
    def sk(nd,pr):
        with torch.no_grad(): return 1-((m(me["feat"],nd,pr,mask)-me["y"])**2).mean().item()/yv.var()
    nd0,pr0=strip_rw(me["nd"],me["pr"])
    print(f"[{variant}] baseline skill={sk(me['nd'],me['pr']):.3f}   RRWP stripped (mask only)={sk(nd0,pr0):.3f}",flush=True)
for v in ["dense","1-hop"]: train_eval(v)
