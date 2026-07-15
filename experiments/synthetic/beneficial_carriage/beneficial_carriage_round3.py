"""Round 3: WHY does data-driven matching fail, and is it rescuable?

Hypothesis from round 2: a data-driven matched swap must ESTIMATE the context
(environment) it conditions on, and that estimation error re-introduces exactly
the off-manifold-ness we were trying to avoid. So estimator quality should be a
smooth function of ONE thing: how well you can identify the context.

We parametrise it directly: resample node j as  x_j' = u' + z_hat,  where
z_hat = z + eps * noise. eps=0 is the oracle; eps large -> marginal. Sweep eps
for both a hand-built model and a TRAINED MLP, and watch (a) estimator quality
vs GT and (b) the var_ratio health gate -- does the gate ever notice?
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)
CLASS = np.array(["ben"]*4 + ["harm"]*3 + ["neu"]*3 + ["unused"]*2)
N = len(CLASS)
a = np.where(CLASS == "neu", 0.0, 1.0)
c = np.select([CLASS=="ben", CLASS=="harm", CLASS=="neu", CLASS=="unused"], [1.,-1.,1.,0.])
U_SCALE, Z_SCALE, SIGMA, THR, KAPPA = 0.7, 1.0, 0.3, 1.6, 0.8

def spearman(u, v):
    ru=np.argsort(np.argsort(u)).astype(float); rv=np.argsort(np.argsort(v)).astype(float)
    return 0.0 if ru.std()<1e-12 or rv.std()<1e-12 else float(np.corrcoef(ru,rv)[0,1])
def gen(G):
    z=Z_SCALE*rng.standard_normal((G,1)); u=U_SCALE*rng.standard_normal((G,N)); return u+z,z,u
def make_label(u): return u@a + SIGMA*rng.standard_normal(u.shape[0])
def model_analytic(X):
    uhat=X-X.mean(1,keepdims=True)
    pen=KAPPA*(np.maximum(np.abs(uhat)-THR,0.)**2*np.abs(c)).sum(1)
    return uhat@c+pen
class MLP:
    def __init__(s,nin,nh=48):
        sc=1/np.sqrt(nin); s.W1=rng.normal(0,sc,(nin,nh)); s.b1=np.zeros(nh)
        s.W2=rng.normal(0,1/np.sqrt(nh),(nh,1)); s.b2=np.zeros(1)
    def __call__(s,X): return (np.tanh(X@s.W1+s.b1)@s.W2+s.b2)[:,0]
    def fit(s,X,y,iters=4000,lr=0.05):
        mW1=mW2=mb1=mb2=0
        for _ in range(iters):
            h=np.tanh(X@s.W1+s.b1); yh=(h@s.W2+s.b2)[:,0]; g=(yh-y)[:,None]/len(y)
            gW2=h.T@g; gb2=g.sum(0); gh=(g@s.W2.T)*(1-h**2); gW1=X.T@gh; gb1=gh.sum(0)
            mW1=.9*mW1+gW1; mW2=.9*mW2+gW2; mb1=.9*mb1+gb1; mb2=.9*mb2+gb2
            s.W1-=lr*mW1; s.W2-=lr*mW2; s.b1-=lr*mb1; s.b2-=lr*mb2
        return s

def resample(X,j,z,eps,K):
    zhat = z + eps*rng.standard_normal((X.shape[0],1))
    return U_SCALE*rng.standard_normal((X.shape[0],K)) + zhat

def ground_truth(model,X,y,z,draws=120):
    base=np.abs(model(X)-y); gt=np.zeros(N)
    for j in range(N):
        v=resample(X,j,z,0.0,draws); acc=np.zeros_like(y)
        for k in range(draws):
            Xc=X.copy(); Xc[:,j]=v[:,k]; acc+=np.abs(model(Xc)-y)
        gt[j]=(acc/draws-base).mean()
    return gt

def finite_quality(model,X,y,z,gt,eps,K=16):
    r=model(X)-y; Lc=np.zeros((X.shape[0],N)); yps=[]
    for j in range(N):
        v=resample(X,j,z,eps,K); acc=np.zeros(X.shape[0])
        for k in range(K):
            Xc=X.copy(); Xc[:,j]=v[:,k]; yp=model(Xc); acc+=np.abs(yp-y); yps.append(yp)
        Lc[:,j]=acc/K
    est=(Lc-np.abs(r)[:,None]).mean(0)
    tol=0.1*np.max(np.abs(gt)); signed=np.abs(gt)>tol
    rho=spearman(est,gt)
    sacc=np.mean(np.sign(est[signed])==np.sign(gt[signed])) if signed.any() else np.nan
    leak=np.mean(np.abs(est[~signed]))/(np.mean(np.abs(est[CLASS=="ben"]))+1e-9)
    quality=0.4*(sacc if np.isfinite(sacc) else 0)+0.3*max(rho,0)+0.3*(1-min(leak,2)/2)
    vr=float(np.var(np.concatenate(yps))/(np.var(model(X))+1e-12))
    return quality, rho, leak, vr

G=1500; EPS=[0.0,0.05,0.1,0.15,0.2,0.3,0.45,0.7,1.0,1.5]
fig,axes=plt.subplots(1,2,figsize=(13,4.8))
for ax,mk in zip(axes,["analytic","mlp"]):
    Xp,zp,up=gen(6000); yp=make_label(up)
    model = model_analytic if mk=="analytic" else MLP(N).fit(Xp,yp)
    X,z,u=gen(G); y=make_label(u); gt=ground_truth(model,X,y,z)
    Q=[]; RHO=[]; LEAK=[]; VR=[]
    for e in EPS:
        q,rho,leak,vr=finite_quality(model,X,y,z,gt,e); Q.append(q);RHO.append(rho);LEAK.append(leak);VR.append(vr)
    # crude data-driven context error for reference: std of (mean of 11 others - z)
    ctx_err = float((X.mean(1,keepdims=True)-z).std())
    ax.plot(EPS,Q,"o-",label="quality vs GT",color="#2ca02c")
    ax.plot(EPS,RHO,"s--",label="rank-corr with GT",color="#1f77b4")
    ax.plot(EPS,[min(v,3)/3 for v in VR],"^:",label="var_ratio/3 (gate=1.0)",color="#d62728")
    ax.axhline(1.0,color="k",lw=0.5,ls=":")
    ax.axvline(ctx_err,color="gray",lw=1.2)
    ax.text(ctx_err,0.05,f" data-driven\n context err\n ~{ctx_err:.2f}",fontsize=7,color="gray")
    ax.set_xlabel("context-estimation error  eps  (0 = oracle, large = marginal)")
    ax.set_ylabel("score"); ax.set_ylim(-0.6,1.1); ax.set_title(f"model = {mk}")
    ax.legend(fontsize=8,loc="lower left")
    print(f"[{mk}] ctx_err~{ctx_err:.2f}  quality@eps0={Q[0]:.2f}  "
          f"quality@ctx_err~{Q[min(range(len(EPS)),key=lambda i:abs(EPS[i]-ctx_err))]:.2f}  "
          f"quality@marginal={Q[-1]:.2f}  max_var_ratio={max(VR):.2f}")
fig.suptitle("Round 3: beneficial-carriage quality collapses smoothly with context-estimation error.\n"
             "The health gate (var_ratio, red) stays flat ~1 across the whole collapse -> it does NOT detect it.",
             fontsize=10)
fig.tight_layout(rect=[0,0,1,0.9])
fig.savefig("fig_round3.png",dpi=130)
print("saved fig_round3.png")
