"""
NoPE last-query specialisation — self-contained Colab script.
====================================================================================

We train small single-head NoPE (no positional encoding) causal transformers as teacher-students,
and — reading ONLY the last query position — score each attention head on a positional<->semantic
axis with three methods:

  local attention   : swap two content vectors at the head's OWN residual-stream input, score the
                      attention response (cosine following/invariance), weighted by moved mass.
  global attention  : swap two content vectors at the MODEL input, recover attention by a forward
                      pass, same cosine following/invariance.
  global transport  : same input swap, but score the readout-weighted delivered MESSAGE
                      (o = sum_k a_{q<-k} v_k), decomposed into a value-change (structural) part and
                      an attention-rerouting (semantic) part.

Each score is reported as a POSITIONAL FRACTION in [0, 1] (1 = positional, 0 = semantic).

We also probe the residual stream directly (linear probes for slot index vs input content) and
report a RESIDUAL POSITIONAL-REPRESENTATION FRACTION per layer.

Five synthetic last-query tasks (continuous Gaussian tokens; the last query does the scored op):
  first  : output token 0
  prev   : output the previous token
  ihalf  : output the token at floor(q/2)
  match  : output the earlier token most similar to the query (max content inner-product)
  posret : output the POSITION of the earlier token whose content equals the query
           (content-addressed position readout)

Six seeds per task, four layers per model. TASKS lists the models trained + cached; FIG_TASKS is the
(sub)set shown in figures (default: prev, ihalf, posret). Every trained model AND its scores are
cached to Drive, so all figures regenerate without retraining or re-scoring. Figures produced:
  fig_positional_fraction : per task, the three methods per layer (with error bars over seeds).
  fig_residual_posrep     : per task, the residual positional-representation fraction per layer.
  fig_attn_mean_<task>    : 1 x n_layers heatmaps of the mean attention across inputs (best-fit seed).
  fig_attn_std_<task>     : 1 x n_layers heatmaps of the std of attention across inputs (best-fit seed).
The attention figures load the cached model and do a single forward pass (no retraining/re-scoring),
and cache their result so subsequent regeneration is free.
"""

# %%
# ============================== 0. Config ==============================
import os
import copy
import json
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:  # pragma: no cover
    _HAVE_PANDAS = False

SMOKE = os.environ.get("NOPE_SMOKE", "") == "1"

# ---- tasks / model ----
TASKS = ["prev", "ihalf", "posret"]          # models trained + cached (add "first"/"match" to include)
FIG_TASKS = ["prev", "ihalf", "posret"]      # subset shown in figures (must be a subset of TASKS)
TASK_NAME = {                         # human-readable, identifies the operation (no claim of result)
    "first":  "Copy first token",
    "prev":   "Copy previous token",
    "ihalf":  "Copy token at ⌊q/2⌋",
    "match":  "Copy most-similar token",
    "posret": "Report position of content match",
}
TASK_KIND = {"first": "reg", "prev": "reg", "ihalf": "reg", "match": "reg", "posret": "cls"}

N_LAYERS   = 4
N_HEADS    = 1
SEQ_LEN    = 8
D_MODEL    = 24 if SMOKE else 32
D_FF       = 48 if SMOKE else 64
SEEDS      = [0, 1] if SMOKE else [0, 1, 2, 3, 4, 5]

# ---- training (early stopping: large budget, converge before scoring) ----
LR            = 1e-3
BATCH_SIZE    = 64 if SMOKE else 256
MAX_STEPS     = 800 if SMOKE else 40000     # generous ceiling
MIN_STEPS     = 200 if SMOKE else 8000      # never stop before this
EVAL_EVERY    = 200 if SMOKE else 1000      # convergence check cadence
PATIENCE      = 3   if SMOKE else 12        # stop after this many checks with no improvement
MIN_DELTA     = 1e-3                         # relative improvement needed to reset patience
FORCE_RETRAIN = False

# ---- scoring ----
N_EVAL     = 64  if SMOKE else 256          # inputs used for the swap scoring
N_PROBES   = 8   if SMOKE else 16           # R: readout probes for the transport functional magnitude
PROBE_N    = 300 if SMOKE else 1500         # samples for the residual probe
EVAL_SEED  = 20260721                        # seeds ALL scoring randomness (inputs, probes, split)
CONV_EVAL_N     = 256 if SMOKE else 1024    # held-out set that selects the best (saved) model
PROBE_RIDGE_LAM = 1e-1                       # residual-probe ridge regulariser
PROBE_SPLIT     = 0.8                        # residual-probe train fraction
FORCE_RESCORE = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
MASK = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), 1).bool().unsqueeze(0).unsqueeze(0)


# %%
# ============================== 1. Drive / cache ==============================
def _setup_cache_dir():
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        base = "/content/drive/MyDrive/nope_lastquery"
    except Exception:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__))
                            if "__file__" in globals() else ".", "nope_lastquery_cache")
    os.makedirs(base, exist_ok=True)
    print(f"[cache] using {base}")
    return base

CACHE_DIR = _setup_cache_dir()

def _hash(d):
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]

def _task_salt(t):                    # stable per-task seed offset (independent of the TASKS list order)
    return int(hashlib.md5(t.encode()).hexdigest()[:6], 16)

CFG = dict(T=SEQ_LEN, d=D_MODEL, dff=D_FF, heads=N_HEADS, layers=N_LAYERS, lr=LR, bs=BATCH_SIZE,
           max_steps=MAX_STEPS, min_steps=MIN_STEPS, eval_every=EVAL_EVERY, patience=PATIENCE,
           min_delta=MIN_DELTA, conv_eval=CONV_EVAL_N)
CFG_TAG = _hash(CFG)
EVAL_TAG = _hash(dict(cfg=CFG_TAG, ne=N_EVAL, r=N_PROBES, pn=PROBE_N, es=EVAL_SEED,
                      lam=PROBE_RIDGE_LAM, split=PROBE_SPLIT))
RUN_DIR = os.path.join(CACHE_DIR, f"L{N_LAYERS}_T{SEQ_LEN}_d{D_MODEL}")
os.makedirs(RUN_DIR, exist_ok=True)


# %%
# ============================== 2. Model ==============================
class Attn(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.qkv = nn.Linear(d, 3 * d, bias=False)
        s.o = nn.Linear(d, d, bias=False)

    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(x).reshape(B, T, 3, D).permute(2, 0, 1, 3)   # single head -> no head dim
        A = (q @ k.transpose(-2, -1) / D ** 0.5).masked_fill(MASK[0], float("-inf")).softmax(-1)
        msg = A @ v
        return s.o(msg), A, v, msg

class Block(nn.Module):
    def __init__(s, d, dff):
        super().__init__()
        s.ln1 = nn.LayerNorm(d); s.at = Attn(d); s.ln2 = nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, dff), nn.GELU(), nn.Linear(dff, d))

class Net(nn.Module):
    def __init__(s, d, dff, nl, out_dim):
        super().__init__()
        s.blocks = nn.ModuleList([Block(d, dff) for _ in range(nl)])
        s.lnf = nn.LayerNorm(d)
        s.head = nn.Linear(d, out_dim)

def forward(net, x, capture=False):
    """x:(B,T,D). Returns logits/preds (B,T,out); if capture also the per-layer A, v, message."""
    h = x; A_all, v_all, m_all = [], [], []
    for b in net.blocks:
        o, A, v, m = b.at(b.ln1(h))
        if capture:
            A_all.append(A); v_all.append(v); m_all.append(m)
        h = h + o
        h = h + b.ff(b.ln2(h))
    out = net.head(net.lnf(h))
    return (out, A_all, v_all, m_all) if capture else out

def residual_inputs(net, x):
    """Residual stream fed into each block (list length n_layers) plus block OUTPUTS (list)."""
    res_in, res_out, h = [], [], x
    for b in net.blocks:
        res_in.append(h)
        o, _, _, _ = b.at(b.ln1(h)); h = h + o; h = h + b.ff(b.ln2(h))
        res_out.append(h)
    return res_in, res_out


# %%
# ============================== 3. Teachers ==============================
def gen_batch(task, B, gen):
    """Returns (x, target, is_cls). x:(B,T,D). Only the LAST query's output is the scored target."""
    T, D = SEQ_LEN, D_MODEL
    x = torch.randn(B, T, D, device=DEVICE, dtype=DTYPE, generator=gen)
    if task == "posret":                         # last token duplicates key at pos r; target = r
        r = torch.randint(0, T - 1, (B,), device=DEVICE, generator=gen)
        x[torch.arange(B), -1] = x[torch.arange(B), r].clone()
        return x, r, True
    if task == "first":   yq = x[:, 0]
    elif task == "prev":  yq = x[:, -2]
    elif task == "ihalf": yq = x[:, (T - 1) // 2]
    elif task == "match":
        sim = torch.einsum("bd,bkd->bk", x[:, -1], x[:, :-1])
        yq = x[torch.arange(B), sim.argmax(-1)]
    else:
        raise ValueError(task)
    return x, yq, False

def _loss(pred_last, target, is_cls):
    return F.cross_entropy(pred_last, target) if is_cls else F.mse_loss(pred_last, target)


# %%
# ============================== 4. Train (early stopping) + cache ==============================
def _model_path(task, seed):
    return os.path.join(RUN_DIR, f"model_{task}_seed{seed}_{CFG_TAG}.pt")

def train_one(task, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    is_cls = TASK_KIND[task] == "cls"
    out_dim = SEQ_LEN if is_cls else D_MODEL
    net = Net(D_MODEL, D_FF, N_LAYERS, out_dim).to(DEVICE, DTYPE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    gtr = torch.Generator(device=DEVICE).manual_seed(seed + 1)
    gev = torch.Generator(device=DEVICE).manual_seed(seed + 777)
    x_ev, t_ev, _ = gen_batch(task, CONV_EVAL_N, gev)     # fixed held-out convergence set

    best = float("inf"); best_state = None; no_improve = 0; stopped_at = MAX_STEPS
    net.train()
    for step in range(1, MAX_STEPS + 1):
        x, t, _ = gen_batch(task, BATCH_SIZE, gtr)
        loss = _loss(forward(net, x)[:, -1], t, is_cls)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0:
            net.eval()
            with torch.no_grad():
                ev = float(_loss(forward(net, x_ev)[:, -1], t_ev, is_cls))
            net.train()
            if ev < best * (1 - MIN_DELTA):
                best = ev; best_state = copy.deepcopy(net.state_dict()); no_improve = 0
            else:
                no_improve += 1
            if step >= MIN_STEPS and no_improve >= PATIENCE:
                stopped_at = step; break
    converged = best_state is not None and np.isfinite(best)
    if converged:
        net.load_state_dict(best_state)
    else:                                       # loss never improved / diverged (e.g. NaN)
        print(f"[warn ] {task} seed{seed}: no finite improving eval — training may have diverged")
    net.eval()
    with torch.no_grad():
        pred = forward(net, x_ev)[:, -1]
        fit = float((pred.argmax(-1) == t_ev).float().mean()) if is_cls else float(F.mse_loss(pred, t_ev))
    return net, dict(fit=fit, fitkind=("acc" if is_cls else "mse"), best_loss=best,
                     steps=stopped_at, converged=bool(converged))

def get_model(task, seed):
    path = _model_path(task, seed)
    is_cls = TASK_KIND[task] == "cls"
    out_dim = SEQ_LEN if is_cls else D_MODEL
    if (not FORCE_RETRAIN) and os.path.exists(path):
        blob = torch.load(path, map_location=DEVICE, weights_only=False)
        net = Net(D_MODEL, D_FF, N_LAYERS, out_dim).to(DEVICE, DTYPE)
        net.load_state_dict(blob["state_dict"]); net.eval()
        print(f"[load ] {task:>7} seed{seed}  {blob['stats']['fitkind']}={blob['stats']['fit']:.3f} "
              f"steps={blob['stats']['steps']}  (cached)")
        return net, blob["stats"]
    net, stats = train_one(task, seed)
    torch.save({"state_dict": net.state_dict(), "stats": stats, "cfg": CFG}, path)
    print(f"[train] {task:>7} seed{seed}  {stats['fitkind']}={stats['fit']:.3f} steps={stats['steps']}  -> cached")
    return net, stats


# %%
# ============================== 5. Scoring (last query) ==============================
def _cos(a, b, eps=1e-8):
    return (a * b).sum(-1) / (a.norm(dim=-1).clamp(min=eps) * b.norm(dim=-1).clamp(min=eps))

def _att_pos_sem(Ac_q, Ap_q, i, j):
    """Last-query attention over keys, clean vs swapped -> mass-weighted (pos, sem) contributions."""
    vij = torch.stack([Ac_q[:, i], Ac_q[:, j]], -1)
    vji = torch.stack([Ac_q[:, j], Ac_q[:, i]], -1)
    vp = torch.stack([Ap_q[:, i], Ap_q[:, j]], -1)
    al = (Ac_q[:, i] - Ac_q[:, j]).abs()
    return al * _cos(vp, vij).clamp(min=0), al * _cos(vp, vji).clamp(min=0)

def eval_inputs(task, n):
    return gen_batch(task, n, torch.Generator(device=DEVICE).manual_seed(EVAL_SEED))[0]

def _scores_path(task, seed):
    return os.path.join(RUN_DIR, f"scores_{task}_seed{seed}_{CFG_TAG}_{EVAL_TAG}.pt")

def score_model(net, task, seed):
    """Per-layer positional fraction for local-attn, global-attn, global-transport at the last query."""
    path = _scores_path(task, seed)
    if (not FORCE_RESCORE) and os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)["scores"]

    L, T = N_LAYERS, SEQ_LEN
    x = eval_inputs(task, N_EVAL); B = x.shape[0]
    out, Ac, Vc, Mc = forward(net, x, capture=True)      # clean capture (no grad path needed here)
    res_in, _ = residual_inputs(net, x)

    # readout gradients phi_l at the last query, via R probes (needs a grad graph)
    x2 = x.detach()
    out2, _, _, Mc2 = forward(net, x2, capture=True)
    qy = out2[:, -1]                                      # (B, out_dim)
    sgen = torch.Generator(device=DEVICE).manual_seed(EVAL_SEED + 7919 * seed + _task_salt(task))
    U = torch.randn(N_PROBES, qy.shape[-1], device=DEVICE, generator=sgen)   # reproducible probes
    phis = []
    for l in range(L):
        Ub = U.unsqueeze(1).expand(N_PROBES, B, qy.shape[-1])
        g = torch.autograd.grad(qy, Mc2[l], grad_outputs=Ub, is_grads_batched=True,
                                retain_graph=True)[0]
        phis.append(g[:, :, -1, :].detach())             # (R, B, D)

    pairs = [(i, j) for i in range(T) for j in range(i + 1, T)]
    acc = {m: {"p": torch.zeros(L, device=DEVICE), "s": torch.zeros(L, device=DEVICE)}
           for m in ("local_att", "global_att", "global_tr")}

    def fmag(phi, d):                                    # sqrt(mean_r (phi.d)^2) -> (B,)
        return (torch.einsum("rbd,bd->rb", phi, d) ** 2).mean(0).sqrt()

    with torch.no_grad():
        for (i, j) in pairs:
            xs = x.clone(); xs[:, [i, j]] = xs[:, [j, i]]
            _, As, Vs, Ms = forward(net, xs, capture=True)
            for l in range(L):
                Acq = Ac[l][:, -1]
                p, s = _att_pos_sem(Acq, As[l][:, -1], i, j)         # global attention
                acc["global_att"]["p"][l] += p.sum(); acc["global_att"]["s"][l] += s.sum()
                hs = res_in[l].clone(); hs[:, [i, j]] = hs[:, [j, i]]
                _, Al, _, _ = net.blocks[l].at(net.blocks[l].ln1(hs))
                p, s = _att_pos_sem(Acq, Al[:, -1], i, j)            # local attention
                acc["local_att"]["p"][l] += p.sum(); acc["local_att"]["s"][l] += s.sum()
                o_clean = Mc[l][:, -1]; o_swap = Ms[l][:, -1]        # global transport
                o_pos = torch.einsum("bk,bkd->bd", Acq, Vs[l])
                al = (Acq[:, i] - Acq[:, j]).abs()
                acc["global_tr"]["p"][l] += (al * fmag(phis[l], o_swap - o_clean)).sum()
                acc["global_tr"]["s"][l] += (al * fmag(phis[l], o_swap - o_pos)).sum()

    scores = {m: [float(d["p"][l] / (d["p"][l] + d["s"][l] + 1e-8)) for l in range(L)]
              for m, d in acc.items()}
    scores["posrep"] = residual_probe(net, task)
    torch.save({"scores": scores, "cfg": CFG_TAG, "eval": EVAL_TAG}, path)
    return scores


# %%
# ============================== 6. Residual probe ==============================
def _ridge(Ftr, Ytr, Fte, lam=PROBE_RIDGE_LAM):
    Ftr = torch.cat([Ftr, torch.ones(Ftr.shape[0], 1, device=DEVICE)], 1)
    Fte = torch.cat([Fte, torch.ones(Fte.shape[0], 1, device=DEVICE)], 1)
    W = torch.linalg.solve(Ftr.T @ Ftr + lam * torch.eye(Ftr.shape[1], device=DEVICE), Ftr.T @ Ytr)
    return Fte @ W

@torch.no_grad()
def residual_probe(net, task):
    """Per-layer residual positional-rep fraction = pos_norm / (pos_norm + content_R2), where
    pos_norm normalises the linear position-decoding accuracy above chance, content_R2 is the linear
    reconstruction of the input. Probes each block's output, pooled over all positions."""
    T = SEQ_LEN
    x = eval_inputs(task, PROBE_N)
    _, res_out = residual_inputs(net, x)
    content = x.reshape(-1, D_MODEL); M = x.shape[0] * T
    rgen = torch.Generator(device=DEVICE).manual_seed(EVAL_SEED + _task_salt(task))
    idx = torch.randperm(M, device=DEVICE, generator=rgen)
    cut = int(PROBE_SPLIT * M); tr, te = idx[:cut], idx[cut:]
    pos = torch.arange(T, device=DEVICE).view(1, T).expand(x.shape[0], T).reshape(-1)
    poh = F.one_hot(pos, T).float(); chance = 1.0 / T
    out = []
    for r in res_out:
        Ff = r.reshape(-1, D_MODEL)
        acc = (_ridge(Ff[tr], poh[tr], Ff[te]).argmax(1) == pos[te]).float().mean().item()
        pc = _ridge(Ff[tr], content[tr], Ff[te])
        r2 = (1 - ((content[te] - pc) ** 2).sum(0)
              / ((content[te] - content[te].mean(0)) ** 2).sum(0).clamp(min=1e-8)).mean().item()
        pn = max(0.0, (acc - chance) / (1 - chance))
        out.append(pn / (pn + max(r2, 0.0) + 1e-9))
    return out


# %%
# ============================== 6b. Attention mean/std (cached; no retrain/rescore) ==============================
@torch.no_grad()
def attention_stats(task, seed):
    """Per-layer mean and std of attention across the eval inputs, for one cached model. Loads the
    cached model (no retraining) + one forward pass, and caches the result so re-generation is free."""
    path = os.path.join(RUN_DIR, f"attn_{task}_seed{seed}_{CFG_TAG}_{EVAL_TAG}.pt")
    if os.path.exists(path):
        b = torch.load(path, map_location="cpu", weights_only=False)
        return b["mean"], b["std"]
    net, _ = get_model(task, seed)
    _, A, _, _ = forward(net, eval_inputs(task, N_EVAL), capture=True)   # A[l]: (B,T,T)
    mean = [A[l].mean(0).cpu().numpy() for l in range(N_LAYERS)]
    std = [A[l].std(0).cpu().numpy() for l in range(N_LAYERS)]
    torch.save({"mean": mean, "std": std}, path)
    return mean, std

def best_seed_for(task, rows):
    rs = [r for r in rows if r["task"] == task]
    pick = max if TASK_KIND[task] == "cls" else min      # cls: highest acc; reg: lowest mse
    return pick(rs, key=lambda r: r["fit"])["seed"]


# %%
# ============================== 7. Run all ==============================
def run_all():
    rows = []
    agg = {t: {m: [] for m in ("local_att", "global_att", "global_tr", "posrep")} for t in TASKS}
    for task in TASKS:
        for seed in SEEDS:
            net, stats = get_model(task, seed)
            sc = score_model(net, task, seed)
            for m in agg[task]:
                agg[task][m].append(sc[m])
            rows.append(dict(task=task, seed=seed, **stats))
    return rows, agg

def _stack(list_of_lists):
    return np.array(list_of_lists, dtype=float)   # (seeds, layers)


# %%
# ============================== 8. Figures ==============================
plt.rcParams.update({
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "axes.linewidth": 0.8, "lines.linewidth": 2.0, "lines.markersize": 6,
})
# Validated categorical hues (data-viz reference palette, fixed order): the two faithful GLOBAL
# methods take the cool pair (blue, green); LOCAL — the method that diverges — takes the distinct
# magenta accent. The probe (separate figure) uses a calm violet; heatmaps use cool sequential ramps.
METHOD_COLOR = {"local_att": "#e87ba4", "global_att": "#2a78d6", "global_tr": "#008300"}
METHOD_LABEL = {"local_att": "local attention", "global_att": "global attention",
                "global_tr": "global transport"}
PROBE_COLOR = "#4a3aa7"
ATTN_CMAP = {"mean": "Blues", "std": "Purples"}
_MARK = dict(marker="o", markersize=7, markeredgecolor="white", markeredgewidth=0.7,
             capsize=3, elinewidth=1.2, capthick=1.2)

def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.6); ax.set_axisbelow(True)

def make_positional_fraction_figure(agg, out_path):
    layers = np.arange(N_LAYERS)
    dodge = {"local_att": -0.10, "global_att": 0.0, "global_tr": 0.10}
    fig, axes = plt.subplots(1, len(FIG_TASKS), figsize=(3.9 * len(FIG_TASKS), 4.3), squeeze=False)
    axes = axes[0]
    for ax, task in zip(axes, FIG_TASKS):
        for m in ("global_att", "global_tr", "local_att"):
            a = _stack(agg[task][m]); mu = a.mean(0); err = a.std(0)
            ax.errorbar(layers + dodge[m], mu, yerr=err, color=METHOD_COLOR[m], label=METHOD_LABEL[m],
                        **_MARK)
        ax.axhline(0.5, color="0.7", lw=0.8, ls=(0, (2, 3)))
        ax.set_title(TASK_NAME[task]); ax.set_xlabel("layer")
        ax.set_xticks(layers); ax.set_ylim(-0.03, 1.03)
        if ax is axes[0]:
            ax.set_ylabel("positional specialisation fraction\n(1 = positional, 0 = semantic)")
            ax.legend(loc="lower left", framealpha=0.92, edgecolor="0.85")
        _despine(ax)
    fig.suptitle(f"Per-layer positional specialisation fraction, by scoring method"
                 f"   ({N_LAYERS}-layer NoPE students, {len(SEEDS)} seeds)",
                 fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path); print(f"[fig  ] saved {out_path}")
    try: plt.show()
    except Exception: pass

def make_residual_posrep_figure(agg, out_path):
    layers = np.arange(N_LAYERS)
    fig, axes = plt.subplots(1, len(FIG_TASKS), figsize=(3.9 * len(FIG_TASKS), 4.3), squeeze=False)
    axes = axes[0]
    ymax = max(0.5, max((_stack(agg[t]["posrep"]).mean(0) + _stack(agg[t]["posrep"]).std(0)).max()
                        for t in FIG_TASKS) * 1.1)
    for ax, task in zip(axes, FIG_TASKS):
        a = _stack(agg[task]["posrep"]); mu = a.mean(0); err = a.std(0)
        ax.errorbar(layers, mu, yerr=err, color=PROBE_COLOR, **{**_MARK, "marker": "s"})
        ax.set_title(TASK_NAME[task]); ax.set_xlabel("layer")
        ax.set_xticks(layers); ax.set_ylim(0, ymax)
        if ax is axes[0]:
            ax.set_ylabel("residual positional representation\n(probe: position vs content)")
        _despine(ax)
    fig.suptitle(f"Per-layer residual positional representation, by linear probe"
                 f"   ({N_LAYERS}-layer NoPE students, {len(SEEDS)} seeds)",
                 fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path); print(f"[fig  ] saved {out_path}")
    try: plt.show()
    except Exception: pass

def make_attention_figure(task, seed, kind, out_path):
    """One figure, size 1 x n_layers: the per-layer attention 'kind' (mean or std) across inputs,
    for the best-fit model of `task`. mean and std are drawn as SEPARATE figures."""
    mean, std = attention_stats(task, seed)
    mats = mean if kind == "mean" else std
    fig, axes = plt.subplots(1, N_LAYERS, figsize=(2.9 * N_LAYERS, 3.4), squeeze=False)
    axes = axes[0]
    for l, ax in enumerate(axes):
        im = ax.imshow(mats[l], cmap=ATTN_CMAP[kind], vmin=0, vmax=max(float(mats[l].max()), 1e-8))
        ax.set_title(f"Layer {l}")
        ax.set_xlabel("key position")
        ax.set_ylabel("query position" if l == 0 else "")
        ax.set_xticks(range(SEQ_LEN)); ax.set_yticks(range(SEQ_LEN))
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    label = "mean attention" if kind == "mean" else "attention variability (std)"
    fig.suptitle(f"{TASK_NAME[task]} — {label} per layer", fontsize=13, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path); print(f"[fig  ] saved {out_path}")
    try: plt.show()
    except Exception: pass


# %%
# ============================== 9. Main ==============================
def print_fit_table(rows):
    print("\n===================== TRAINING SUMMARY =====================")
    if _HAVE_PANDAS:
        df = pd.DataFrame(rows)[["task", "seed", "fitkind", "fit", "steps"]]
        summ = df.groupby(["task", "fitkind"]).agg(fit_mean=("fit", "mean"),
                                                   fit_std=("fit", "std"),
                                                   steps_mean=("steps", "mean")).reset_index()
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print(summ.to_string(index=False))
    else:  # pragma: no cover
        for r in rows:
            print(f"  {r['task']:>7} seed{r['seed']} {r['fitkind']}={r['fit']:.3f} steps={r['steps']}")
    print("============================================================\n")

def main():
    print(f"[run  ] device={DEVICE} tasks={TASKS} fig_tasks={FIG_TASKS} seeds={SEEDS} "
          f"cfg={CFG_TAG} eval={EVAL_TAG} smoke={SMOKE}")
    rows, agg = run_all()
    print_fit_table(rows)
    tag = f"{CFG_TAG}_{EVAL_TAG}"
    make_positional_fraction_figure(agg, os.path.join(RUN_DIR, f"fig_positional_fraction_{tag}.png"))
    make_residual_posrep_figure(agg, os.path.join(RUN_DIR, f"fig_residual_posrep_{tag}.png"))
    # per-task attention mean & std (separate figures), best-fit seed, from cached models
    for task in FIG_TASKS:
        bs = best_seed_for(task, rows)
        make_attention_figure(task, bs, "mean",
                              os.path.join(RUN_DIR, f"fig_attn_mean_{task}_seed{bs}_{tag}.png"))
        make_attention_figure(task, bs, "std",
                              os.path.join(RUN_DIR, f"fig_attn_std_{task}_seed{bs}_{tag}.png"))
    return rows, agg


if __name__ == "__main__":
    main()
