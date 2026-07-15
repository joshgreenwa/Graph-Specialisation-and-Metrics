"""Round 2: break the assumptions of round 1.

Round 1 concluded: matched (on-manifold) swap + direct loss difference wins.
But round 1 CHEATED -- it built matched swaps from the KNOWN generative model.
Here we break that and more:

  BREAK 1  We no longer know the manifold. Matched swaps must be built from DATA
           (a donor pool), via kNN on an environment key or a learned conditional.
           Q: does the winner survive? and what does its quality now depend on?
  BREAK 2  We no longer hand-design the model. It is a small MLP TRAINED on data,
           so its off-manifold behaviour is emergent, not engineered.
  BREAK 3  We stress the environment key: match on the RIGHT context, the WRONG
           context, or nothing -- to find where data-driven matching collapses.
  GATE     For every resampler we compute var_ratio = Var(yhat_corrupt)/Var(yhat_clean).
           Q: does a cheap health gate DETECT when matching has gone off-manifold,
           so we can refuse to report rather than report garbage?

Ground truth stays the oracle on-manifold leave-one-out loss change (we are the
designer, so we may compute it). Estimators may NOT use oracle knowledge.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)

# ---- node classes: 4 beneficial, 3 harmful, 3 neutral, 2 unused ----
CLASS = np.array(["ben"]*4 + ["harm"]*3 + ["neu"]*3 + ["unused"]*2)
N = len(CLASS)
a = np.where(CLASS == "neu", 0.0, 1.0)                       # task coef (0 for neutral)
c = np.select([CLASS == "ben", CLASS == "harm", CLASS == "neu", CLASS == "unused"],
              [1.0, -1.0, 1.0, 0.0])                          # model coef

def spearman(u, v):
    ru = np.argsort(np.argsort(u)).astype(float); rv = np.argsort(np.argsort(v)).astype(float)
    if ru.std() < 1e-12 or rv.std() < 1e-12: return 0.0
    return float(np.corrcoef(ru, rv)[0, 1])

# ================================================================================
# Manifold: x_j = u_j (independent signal) + z (shared context).  Task uses u.
# ================================================================================
U_SCALE, Z_SCALE, SIGMA = 0.7, 1.0, 0.3

def gen(G):
    z = Z_SCALE * rng.standard_normal((G, 1))
    u = U_SCALE * rng.standard_normal((G, N))
    return u + z, z, u

def make_label(u):
    return u @ a + SIGMA * rng.standard_normal(u.shape[0])

# ---- analytic model (BREAK-1 experiment): context-aware + off-manifold penalty --
THR, KAPPA = 1.6, 0.8
def model_analytic(X):
    zhat = X.mean(axis=1, keepdims=True)
    uhat = X - zhat
    pen = KAPPA * (np.maximum(np.abs(uhat) - THR, 0.0) ** 2 * np.abs(c)).sum(axis=1)
    return uhat @ c + pen

# ---- trained MLP (BREAK-2 experiment): learns context-dependence from data -------
class MLP:
    def __init__(self, nin, nh=48):
        s = 1.0/np.sqrt(nin)
        self.W1 = rng.normal(0, s, (nin, nh)); self.b1 = np.zeros(nh)
        self.W2 = rng.normal(0, 1/np.sqrt(nh), (nh, 1)); self.b2 = np.zeros(1)
    def __call__(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        return (h @ self.W2 + self.b2)[:, 0]
    def fit(self, X, y, iters=4000, lr=0.05):
        mW1=mW2=mb1=mb2=0
        for t in range(iters):
            h = np.tanh(X @ self.W1 + self.b1); yhat = (h @ self.W2 + self.b2)[:, 0]
            g = (yhat - y)[:, None] / len(y)
            gW2 = h.T @ g; gb2 = g.sum(0)
            gh = (g @ self.W2.T) * (1 - h**2)
            gW1 = X.T @ gh; gb1 = gh.sum(0)
            mom = 0.9
            mW1 = mom*mW1 + gW1; mW2 = mom*mW2 + gW2; mb1 = mom*mb1 + gb1; mb2 = mom*mb2 + gb2
            self.W1 -= lr*mW1; self.W2 -= lr*mW2; self.b1 -= lr*mb1; self.b2 -= lr*mb2
        return self

# ================================================================================
# Resamplers (return a copy of X with column j replaced). Only 'oracle' knows z.
# ================================================================================
class DonorBank:
    """1-D environment-keyed donor bank built purely from sample data."""
    def __init__(self, Xpool, key="context"):
        G, n = Xpool.shape
        keys, vals = [], []
        for j in range(n):
            others = np.delete(Xpool, j, axis=1)
            if key == "context":       k = others.mean(axis=1)          # ~ z : the RIGHT key
            elif key == "wrong":       k = others[:, 0]                 # one arbitrary neighbour
            elif key == "none":        k = np.zeros(G)                  # match on nothing == marginal
            else: raise ValueError(key)
            keys.append(k); vals.append(Xpool[:, j])
        self.keys = np.concatenate(keys); self.vals = np.concatenate(vals)
        order = np.argsort(self.keys)
        self.keys = self.keys[order]; self.vals = self.vals[order]
        self.key = key
    def draw(self, X, j, K):
        others = np.delete(X, j, axis=1)
        if self.key == "context": q = others.mean(axis=1)
        elif self.key == "wrong": q = others[:, 0]
        else: q = np.zeros(X.shape[0])
        idx = np.searchsorted(self.keys, q)
        out = np.empty((X.shape[0], K))
        W = 64
        for k in range(K):
            off = rng.integers(-W, W+1, size=X.shape[0])
            pick = np.clip(idx + off, 0, len(self.vals)-1)
            out[:, k] = self.vals[pick]
        return out   # [G,K] donor contents

class LearnedConditional:
    """Sample x_j | x_{-j} from a linear regression fit on the pool (+ resid noise)."""
    def __init__(self, Xpool):
        self.coef = []; self.b = []; self.sd = []
        for j in range(Xpool.shape[1]):
            others = np.delete(Xpool, j, axis=1)
            A = np.c_[others, np.ones(len(others))]
            w, *_ = np.linalg.lstsq(A, Xpool[:, j], rcond=None)
            pred = A @ w
            self.coef.append(w[:-1]); self.b.append(w[-1]); self.sd.append((Xpool[:, j]-pred).std())
    def draw(self, X, j, K):
        others = np.delete(X, j, axis=1)
        mu = others @ self.coef[j] + self.b[j]
        return mu[:, None] + self.sd[j] * rng.standard_normal((X.shape[0], K))

def resample_oracle(X, j, z, K):            # cheats: knows manifold
    return (U_SCALE*rng.standard_normal((X.shape[0], K)) + z[:, 0:1])
def resample_marginal(X, j, K):
    sd = np.sqrt(U_SCALE**2 + Z_SCALE**2)
    return sd * rng.standard_normal((X.shape[0], K))

# ================================================================================
# Ground truth (oracle) + estimator harness
# ================================================================================
def ground_truth(model, X, y, z, draws=120):
    yc = model(X); base = np.abs(yc - y); gt = np.zeros(N)
    for j in range(N):
        acc = np.zeros_like(y)
        vals = resample_oracle(X, j, z, draws)
        for k in range(draws):
            Xc = X.copy(); Xc[:, j] = vals[:, k]; acc += np.abs(model(Xc) - y)
        gt[j] = (acc/draws - base).mean()
    return gt

def run_estimators(model, X, y, z, banks, K=16):
    yc = model(X); r = yc - y
    G = X.shape[0]
    donor_sets = {
        "finite_oracle":   ("oracle", None),
        "finite_knn_ctx":  ("bank", banks["context"]),
        "finite_knn_wrong":("bank", banks["wrong"]),
        "finite_learned":  ("bank", banks["learned"]),
        "finite_marginal": ("marginal", None),
    }
    out = {}; varratio = {}
    # functional carriage (for the first-order baseline) via oracle-matched swaps -- generous
    F = np.zeros((G, N))
    for j in range(N):
        v = resample_oracle(X, j, z, K); f = np.zeros(G)
        for k in range(K):
            Xc = X.copy(); Xc[:, j] = v[:, k]; f += (yc - model(Xc))
        F[:, j] = f/K
    out["hard_sign"] = (-np.sign(r)[:, None]*F).mean(axis=0)

    for name, (kind, obj) in donor_sets.items():
        Lc = np.zeros((G, N)); yc_all = []
        for j in range(N):
            if kind == "oracle": vals = resample_oracle(X, j, z, K)
            elif kind == "marginal": vals = resample_marginal(X, j, K)
            else: vals = obj.draw(X, j, K)
            acc = np.zeros(G)
            for k in range(K):
                Xc = X.copy(); Xc[:, j] = vals[:, k]; yp = model(Xc)
                acc += np.abs(yp - y); yc_all.append(yp)
            Lc[:, j] = acc/K
        out[name] = (Lc - np.abs(r)[:, None]).mean(axis=0)
        varratio[name] = float(np.var(np.concatenate(yc_all)) / (np.var(yc) + 1e-12))
    return out, varratio

# ================================================================================
# Metrics
# ================================================================================
def metrics(gt, v):
    tol = 0.1*np.max(np.abs(gt)); signed = np.abs(gt) > tol
    rho = spearman(v, gt)
    sacc = np.mean(np.sign(v[signed]) == np.sign(gt[signed])) if signed.any() else np.nan
    leak = np.mean(np.abs(v[~signed])) / (np.mean(np.abs(v[CLASS == "ben"]))+1e-9)
    quality = 0.4*(sacc if np.isfinite(sacc) else 0) + 0.3*max(rho,0) + 0.3*(1-min(leak,2)/2)
    return dict(rho=rho, sign_acc=sacc, leak=leak, quality=quality)

# ================================================================================
# RUN
# ================================================================================
G_pool, G = 5000, 1500
ESTS = ["finite_oracle", "finite_knn_ctx", "finite_learned",
        "finite_knn_wrong", "finite_marginal", "hard_sign"]

def experiment(model_kind):
    Xp, zp, up = gen(G_pool); yp = make_label(up)
    if model_kind == "analytic":
        model = model_analytic
    else:
        model = MLP(N).fit(Xp, yp)
    X, z, u = gen(G); y = make_label(u)
    banks = {"context": DonorBank(Xp, "context"),
             "wrong":   DonorBank(Xp, "wrong"),
             "learned": LearnedConditional(Xp)}
    gt = ground_truth(model, X, y, z)
    est, vr = run_estimators(model, X, y, z, banks)
    m = {e: metrics(gt, est[e]) for e in ESTS}
    return gt, est, m, vr, model, (X, y, z)

print("="*90)
res = {}
for mk in ["analytic", "mlp"]:
    gt, est, m, vr, model, data = experiment(mk)
    res[mk] = (gt, est, m, vr)
    X, y, z = data
    print(f"\n#### MODEL = {mk}   (clean MAE = {np.abs(model(X)-y).mean():.3f}, "
          f"label std = {y.std():.2f})")
    print("GT:       " + "".join(f"{g:7.3f}" for g in gt) + "   <- ben>0 harm<0 neu~0")
    print("classes:  " + "".join(f"{k:>7}" for k in CLASS))
    print(f"{'estimator':<18}{'rho':>7}{'sign':>7}{'leak':>8}{'quality':>9}{'var_ratio':>11}")
    for e in ESTS:
        mm = m[e]; v = vr.get(e, float('nan'))
        gate = "" if not np.isfinite(v) else ("   <-- GATE TRIPS" if v > 3 else "")
        print(f"{e:<18}{mm['rho']:+7.2f}{mm['sign_acc']:7.2f}{mm['leak']:8.2f}"
              f"{mm['quality']:9.2f}{v:11.2f}{gate}")

# ================================================================================
# Figures
# ================================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for ax, mk in zip(axes, ["analytic", "mlp"]):
    gt, est, m, vr = res[mk]
    q = [m[e]["quality"] for e in ESTS]
    colors = ["#2ca02c", "#2ca02c", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    ax.bar(range(len(ESTS)), q, color=colors)
    ax.set_xticks(range(len(ESTS))); ax.set_xticklabels(ESTS, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("estimator quality (1=matches GT)")
    ax.set_title(f"model = {mk}")
    for i, e in enumerate(ESTS):
        v = vr.get(e)
        tag = f"\nvr={v:.1f}" if v is not None else ""
        ax.text(i, q[i]+0.02, f"{q[i]:.2f}{tag}", ha="center", va="bottom", fontsize=7)
fig.suptitle("Break the manifold assumption: data-driven matched swaps (green) vs wrong-key/marginal (orange/red) "
             "vs first-order (purple).\nvr = var_ratio health gate (trips when >3 => swap went off-manifold).", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig("fig_round2.png", dpi=130)
print("\nsaved fig_round2.png")
