"""Synthetic testbed for definitions of *beneficial semantic carriage*.

We control the generative process, so we KNOW each node's true task benefit
(the on-manifold leave-one-out loss change). We then ask which cheap estimator
-- computed from a clean forward pass + a handful of content swaps, exactly what
the real pipeline can afford -- best tracks that ground truth across regimes.

Node classes (planted):
  beneficial : task wants x_j, model uses x_j the same way  -> removing hurts (GT>0)
  harmful    : model uses x_j *against* the task            -> removing helps (GT<0)
  neutral    : task independent of x_j, model still uses it -> removing ~neutral (GT~0)
  unused     : model ignores x_j                            -> no functional, no benefit

Estimators (all oriented so POSITIVE = beneficial), with r = yhat_clean - y and
F_j = yhat_clean - yhat_corrupt (functional carriage, mean over swaps):
  hard_sign    :  -sign(r) * F_j            (current method: L1 loss-gradient x functional)
  huber        :  -clip(r/delta,-1,1) * F_j (soft sign, dead-zone ~delta around label)
  raw_resid    :  -r * F_j                  (L2 loss-gradient x functional)
  finite_marg  :  |yhat_corrupt - y| - |yhat_clean - y|   (direct loss diff, MARGINAL swap)
  finite_match :  same, but ON-MANIFOLD (matched) swap
The first three only ever touch the *clean* residual (never the corrupt loss);
finite_* evaluate the loss at the corrupt (possibly off-manifold) prediction.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style

def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])

# ----------------------------- node layout -------------------------------------
# 4 beneficial, 3 harmful, 3 neutral, 2 unused
CLASS = np.array(["ben"]*4 + ["harm"]*3 + ["neu"]*3 + ["unused"]*2)
N = len(CLASS)
COLOR = {"ben": "#2ca02c", "harm": "#d62728", "neu": "#7f7f7f", "unused": "#1f77b4"}

def base_coeffs():
    a = np.zeros(N); c = np.zeros(N)          # a = task coef, c = model coef
    for j, k in enumerate(CLASS):
        if k == "ben":    a[j], c[j] = 1.0,  1.0
        elif k == "harm": a[j], c[j] = 1.0, -1.0
        elif k == "neu":  a[j], c[j] = 0.0,  1.0
        elif k == "unused": a[j], c[j] = 1.0, 0.0
    return a, c

# ----------------------------- regimes -----------------------------------------
# each returns dict with generator + model + on/off-manifold resamplers
def make_regime(name):
    a, c = base_coeffs()
    cfg = dict(name=name, a=a, c=c, b=0.0, sigma=0.5, scale=1.0,
               nonlinear=False, ood=False, rho=0.0, kappa=0.0)
    if name == "baseline":
        pass
    elif name == "lowres":                # tiny residuals -> sign(r) is noise
        cfg["c"] = c * np.where(CLASS == "ben", 1.0, 0.3); cfg["sigma"] = 0.05
    elif name == "biased":                # systematic offset -> sign(r) coherent
        cfg["b"] = 3.0
    elif name == "noisy":                 # label noise dominates -> sign(r) random
        cfg["sigma"] = 2.0
    elif name == "overshoot":             # big swings -> swaps straddle the label
        cfg["scale"] = 3.0
    elif name == "nonlinear":
        cfg["nonlinear"] = True
    elif name == "ood":
        # Each node = independent signal u_j + shared context z; model reads u_j by
        # de-meaning the context, but PENALISES nodes whose de-contexted value is
        # implausibly large (a learned "this looks wrong" reaction). A matched swap
        # keeps u_j small -> benign; a marginal swap breaks the u_j<->z coupling ->
        # large de-contexted value -> penalty fires -> yhat_corrupt is garbage.
        cfg.update(sigma=0.3, ood=True, u_scale=0.7, z_scale=1.0, thr=1.6, kappa=0.8)
    else:
        raise ValueError(name)
    return cfg

def feat(x, cfg):
    return np.tanh(x) if cfg["nonlinear"] else x

def gen(cfg, G, rng):
    """Return X [G,N] and latent dict."""
    if cfg["ood"]:
        z = cfg["z_scale"] * rng.standard_normal((G, 1))
        u = cfg["u_scale"] * rng.standard_normal((G, N))
        return u + z, {"z": z, "u": u}
    return cfg["scale"] * rng.standard_normal((G, N)), {}

def model(X, cfg):
    if cfg["ood"]:
        zhat = X.mean(axis=1, keepdims=True)                 # consensus estimate of z
        uhat = X - zhat                                       # de-contexted node value ~ u_j
        s_j = np.abs(cfg["c"])
        pen = cfg["kappa"] * (np.maximum(np.abs(uhat) - cfg["thr"], 0.0) ** 2 * s_j).sum(axis=1)
        return uhat @ cfg["c"] + cfg["b"] + pen
    return feat(X, cfg) @ cfg["c"] + cfg["b"]

def label(X, cfg, rng, latent):
    core = latent["u"] @ cfg["a"] if cfg["ood"] else feat(X, cfg) @ cfg["a"]
    return core + cfg["sigma"] * rng.standard_normal(X.shape[0])

def resample_col(X, j, cfg, rng, matched, latent):
    """Copy of X with column j replaced. matched=True stays on-manifold."""
    Xc = X.copy(); G = X.shape[0]
    if cfg["ood"]:
        if matched:                          # keep context z, redraw this node's signal u_j
            Xc[:, j] = cfg["u_scale"] * rng.standard_normal(G) + latent["z"][:, 0]
        else:                                # marginal draw -> breaks u_j<->z coupling (OOD)
            sd = np.sqrt(cfg["u_scale"]**2 + cfg["z_scale"]**2)
            Xc[:, j] = sd * rng.standard_normal(G)
    else:
        Xc[:, j] = cfg["scale"] * rng.standard_normal(G)   # iid: matched == marginal
    return Xc

# ----------------------------- ground truth ------------------------------------
def ground_truth(cfg, X, y, latent, rng, draws=200):
    """GT_j = E_resample[ |yhat(resampled_j) - y| ] - |yhat_clean - y|, ON-MANIFOLD."""
    yhat = model(X, cfg); base = np.abs(yhat - y)
    gt = np.zeros(N)
    for j in range(N):
        acc = np.zeros_like(y)
        for _ in range(draws):
            Xc = resample_col(X, j, cfg, rng, matched=True, latent=latent)
            acc += np.abs(model(Xc, cfg) - y)
        gt[j] = (acc / draws - base).mean()
    return gt

# ----------------------------- estimators --------------------------------------
def estimators(cfg, X, y, latent, rng, K=24):
    yhat = model(X, cfg); r = yhat - y
    delta = max(np.median(np.abs(r)), 1e-6)
    G = X.shape[0]
    F = np.zeros((G, N))            # functional carriage (matched swaps: good practice)
    Lc_marg = np.zeros((G, N)); Lc_match = np.zeros((G, N))
    for j in range(N):
        fj = np.zeros(G); lm = np.zeros(G); lmt = np.zeros(G)
        for _ in range(K):
            Xmt = resample_col(X, j, cfg, rng, matched=True, latent=latent)
            ymt = model(Xmt, cfg)
            fj += (yhat - ymt); lmt += np.abs(ymt - y)
            Xmg = resample_col(X, j, cfg, rng, matched=False, latent=latent)
            lm += np.abs(model(Xmg, cfg) - y)
        F[:, j] = fj / K; Lc_marg[:, j] = lm / K; Lc_match[:, j] = lmt / K
    absr = np.abs(r)[:, None]
    est = {
        "functional":   np.abs(F).mean(axis=0),                     # magnitude only (ref)
        "hard_sign":    (-np.sign(r)[:, None] * F).mean(axis=0),
        "huber":        (-np.clip(r / delta, -1, 1)[:, None] * F).mean(axis=0),
        "raw_resid":    (-r[:, None] * F).mean(axis=0),
        "finite_marg":  (Lc_marg - absr).mean(axis=0),
        "finite_match": (Lc_match - absr).mean(axis=0),
    }
    return est

# ----------------------------- evaluation --------------------------------------
def score(gt, est):
    tol = 0.1 * np.max(np.abs(gt))
    truth_signed = np.abs(gt) > tol            # ben or harm (non-neutral)
    neutralish = ~truth_signed
    out = {}
    for name, v in est.items():
        if name == "functional":
            continue
        # rank correlation with GT across nodes
        rho = spearman(v, gt)
        rho = 0.0 if not np.isfinite(rho) else rho
        # sign accuracy on truly ben/harm nodes
        if truth_signed.any():
            sacc = np.mean(np.sign(v[truth_signed]) == np.sign(gt[truth_signed]))
        else:
            sacc = np.nan
        # neutral leakage: |value| on neutral nodes / |value| on ben nodes
        vscale = np.mean(np.abs(v[CLASS == "ben"])) + 1e-9
        leak = np.mean(np.abs(v[neutralish])) / vscale
        out[name] = dict(rho=rho, sign_acc=sacc, leak=leak)
    return out, truth_signed

# ----------------------------- run all -----------------------------------------
REGIMES = ["baseline", "lowres", "biased", "noisy", "overshoot", "nonlinear", "ood"]
ESTS = ["hard_sign", "huber", "raw_resid", "finite_marg", "finite_match"]
G = 1500
rng = np.random.default_rng(7)

results = {}   # regime -> (gt, est, scores)
for rn in REGIMES:
    cfg = make_regime(rn)
    X, latent = gen(cfg, G, rng)
    y = label(X, cfg, rng, latent)
    gt = ground_truth(cfg, X, y, latent, rng, draws=200)
    est = estimators(cfg, X, y, latent, rng, K=24)
    sc, truth_signed = score(gt, est)
    results[rn] = (gt, est, sc)

# ----------------------------- print summary -----------------------------------
print("\n================  GROUND TRUTH (mean loss change from removing node) ================")
hdr = "regime      " + "".join(f"{k:>8}" for k in CLASS)
print(hdr)
for rn in REGIMES:
    gt = results[rn][0]
    print(f"{rn:<11} " + "".join(f"{g:8.3f}" for g in gt))
print("classes:    " + "".join(f"{k:>8}" for k in CLASS))

def combined(sc):
    # bounded quality in [0,1]: sign correctness, rank agreement, neutral suppression
    sacc = sc["sign_acc"] if np.isfinite(sc["sign_acc"]) else 0.0
    return 0.4*sacc + 0.3*max(sc["rho"], 0.0) + 0.3*(1 - min(sc["leak"], 2.0)/2.0)

print("\n================  SCORES per regime  (rho | sign_acc | neutral_leak) ================")
for rn in REGIMES:
    sc = results[rn][2]
    print(f"\n[{rn}]")
    for e in ESTS:
        s = sc[e]
        print(f"   {e:<13} rho={s['rho']:+.2f}  sign_acc={s['sign_acc']:.2f}  leak={s['leak']:.2f}  score={combined(s):+.2f}")

print("\n================  MEAN score across regimes (winner ranking) ================")
means = {e: np.mean([combined(results[rn][2][e]) for rn in REGIMES]) for e in ESTS}
means_noood = {e: np.mean([combined(results[rn][2][e]) for rn in REGIMES if rn != "ood"]) for e in ESTS}
for e in sorted(means, key=lambda k: -means[k]):
    print(f"   {e:<13} mean_score={means[e]:+.3f}   (excl. ood: {means_noood[e]:+.3f})")

# ----------------------------- figure: scatter grid ----------------------------
cols = ["GT"] + ESTS
fig, axes = plt.subplots(len(REGIMES), len(cols), figsize=(2.9*len(cols), 2.7*len(REGIMES)),
                         squeeze=False)
for ri, rn in enumerate(REGIMES):
    gt, est, sc = results[rn]
    fmag = est["functional"]
    xs = fmag / (fmag.max() + 1e-9)
    for ci, col in enumerate(cols):
        ax = axes[ri][ci]
        v = gt if col == "GT" else est[col]
        vn = v / (np.std(v) + 1e-9)
        for k in ["ben", "harm", "neu", "unused"]:
            m = CLASS == k
            ax.scatter(xs[m], vn[m], c=COLOR[k], s=28, label=k, edgecolors="k", linewidths=0.3)
        ax.axhline(0, color="k", lw=0.6, ls=":")
        if ri == 0:
            ax.set_title(col, fontsize=10)
        if ci == 0:
            ax.set_ylabel(rn, fontsize=10)
        if col != "GT":
            ax.text(0.03, 0.9, f"score {combined(sc[col]):+.2f}", transform=ax.transAxes,
                    fontsize=7, va="top")
        ax.set_xticks([]); ax.set_yticks([])
axes[0][0].legend(fontsize=6, loc="lower right", framealpha=0.9)
fig.suptitle("Functional (x = |functional carriage|) vs Beneficial (y, per-panel z-scored).  "
             "Good estimator: green>0, red<0, grey~0 -- i.e. matches the GT column.", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("fig_scatter_grid.png", dpi=130)
print("\nsaved fig_scatter_grid.png")

# ----------------------------- figure: score heatmap ---------------------------
fig2, ax = plt.subplots(figsize=(1.1*len(REGIMES)+2, 0.6*len(ESTS)+2))
M = np.array([[combined(results[rn][2][e]) for rn in REGIMES] for e in ESTS])
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(REGIMES))); ax.set_xticklabels(REGIMES, rotation=35, ha="right")
ax.set_yticks(range(len(ESTS))); ax.set_yticklabels(ESTS)
for i in range(len(ESTS)):
    for j in range(len(REGIMES)):
        ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Beneficial-carriage estimator quality in [0,1]\n(0.4*sign_acc + 0.3*max(rankcorr,0) + 0.3*neutral_suppression)")
fig2.colorbar(im, fraction=0.046, pad=0.04)
fig2.tight_layout()
fig2.savefig("fig_score_heatmap.png", dpi=130)
print("saved fig_score_heatmap.png")

# ------------------- figure: the headline flaw (neutral leakage) ---------------
fig3, ax = plt.subplots(figsize=(7, 4))
leak_mean = {e: np.mean([results[rn][2][e]["leak"] for rn in REGIMES]) for e in ESTS}
bars = ax.bar(range(len(ESTS)), [leak_mean[e] for e in ESTS],
              color=["#d62728"]*3 + ["#2ca02c"]*2)
ax.axhline(1.0, color="k", ls="--", lw=1, label="|beneficial| on neutral == on genuinely-beneficial nodes")
ax.set_xticks(range(len(ESTS))); ax.set_xticklabels(ESTS, rotation=20)
ax.set_ylabel("neutral leakage  =  mean |B| on NEUTRAL nodes\n/ mean |B| on BENEFICIAL nodes")
ax.set_title("The core flaw of first-order estimators: they score task-irrelevant\n"
             "(spuriously-used) nodes as MORE 'beneficial' than genuinely useful ones")
for b, e in zip(bars, ESTS):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f"{leak_mean[e]:.1f}x",
            ha="center", fontsize=9)
ax.legend(fontsize=8)
fig3.tight_layout()
fig3.savefig("fig_neutral_leakage.png", dpi=130)
print("saved fig_neutral_leakage.png")
