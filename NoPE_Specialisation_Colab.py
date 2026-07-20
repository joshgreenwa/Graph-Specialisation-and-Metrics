"""
NoPE teacher-student SPECIALISATION scores -- self-contained Colab script.
====================================================================================

Task (from the `nope_index_retrieval` notebook + the /specialisation/ methodology):

  A TEACHER applies a fixed *positional* attention matrix P to the input:  Y = P X
  (default P = "previous_token": query q reads key q-1, with q=0 -> itself).
  A STUDENT is a small NoPE (no positional encoding) causal transformer, ONE head per
  layer, trained to imitate the teacher from random Gaussian inputs X.

  We treat the sequence as a 1D graph: a "node" = (slot, content). NoPE => the token
  representation is PURELY content; position is only the fixed causal-mask/slot scaffold.
  "positional/structural" = attention/message INVARIANT to a content swap (stays on slot);
  "semantic" = EQUIVARIANT (follows content).

This script trains 3 seeds of each of the 1-, 2- and 3-layer students (caching every
trained model to Drive so a later analysis run never retrains), prints a training-summary
table, and draws a 3x3 grid of scatter plots -- semantic (y) vs structural/positional (x) --
with one column per depth and one row per scoring method:

  Row 1  METHOD (1) LOCAL  : content swap at each head's OWN residual-stream input, score the
                             attention response (cosine following/invariance), alpha-weighted.
  Row 2  METHOD (2) GLOBAL : content swap (node transposition) at the MODEL input, recover all
                             attention by a forward pass, same cosine following/invariance.
  Row 3  METHOD (3) TRANSPORT : the /specialisation/ transport score ported to a single
                             node-transposition base unit, alpha- and readout-weighted (below).

Each scatter point is one (layer, query-position) unit, coloured by layer, averaged over seeds.
A companion 3x3 grid draws the D-J plane: normalising each channel by its mean (S~ = raw/mean),
J = (S~sem + S~str)/2 is joint strength (influential vs inert) and D = (S~sem - S~str)/2 is
selectivity (semantic right / structural left). Caveat (see make_jd_figure): here S_sem and S_str
are a split of ONE swap, not two independent interventions, so D is a within-swap balance rather
than a two-cause preference -- cleanest at the extremes.

--------------------------------------------------------------------------------------------
WHY METHOD (3) HAS NO SEPARATE "POSITIONAL INTERVENTION"
--------------------------------------------------------------------------------------------
In /specialisation/ (graphs) the SEMANTIC and STRUCTURAL scores come from two interventions on
two distinct input channels: a donor content swap (content channel) and an RRWP node
transposition with the attention mask frozen (structural channel). A NoPE sequence has NO
separate structural channel -- "swap position, hold content" is the SAME array op as "swap
content, hold slot". So a separate positional intervention does not exist; we use ONE base
unit (the content transposition i<->j at the model input, mask fixed) and read BOTH channels
from how the transported message responds:

  transport site      o^l_q   = sum_k a^l_{q<-k} v^l_k        (per-layer, per-query message)
  readout gradient    phi^l_q = d yhat_q / d o^l_q            (functional magnitude over the D
                                                               prediction dims, Hutchinson-
                                                               estimated with R random probes)
  clean / swap captures give  A_clean,V_clean,MSG_clean  and  A_swap,V_swap,MSG_swap.

  positional counterfactual   o_pos_q = A_clean (x) V_swap     (attention stays, swapped
                                                                content flows into the slots)

  d_net = o_swap_q - o_clean_q      NET delivered-message change  -> STRUCTURAL / positional
  d_sem = o_swap_q - o_pos_q        attention re-routing response -> SEMANTIC / equivariant

  S_str(l,q) = alpha-weighted mean_pairs  || phi^l_q . d_net ||        (Jensen: weight before |.|)
  S_sem(l,q) = alpha-weighted mean_pairs  || phi^l_q . d_sem ||
  alpha      = | A_clean[l,q,i] - A_clean[l,q,j] |   (attention mass the swap actually moves)

Sanity (verified below): a purely positional head has A_swap=A_clean so d_sem=0 (S_sem~0) while
d_net!=0 (S_str>0); a purely semantic head delivers an invariant message so o_swap~o_clean
(d_net~0, S_str~0) while its attention re-routes (d_sem!=0, S_sem>0); an inert head scores ~0 on
both. NB the label flips vs a single-site donor swap: under a content-PRESERVING transposition
the NET transport change is the POSITIONAL signal, not the semantic one.
"""

# %%
# ============================== 0. Config ==============================
import os
import math
import json
import hashlib
from collections import defaultdict

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

# Smoke mode (local CPU sanity run): NOPE_SMOKE=1 shrinks everything so the whole
# pipeline runs in a few seconds. Colab runs leave it unset for the full config.
SMOKE = os.environ.get("NOPE_SMOKE", "") == "1"

# ---- task / model geometry ----
TEACHER_TYPE = "previous_token"     # previous_token | diagonal | first_token | uniform_causal
SEQ_LEN      = 8  if SMOKE else 16
D_MODEL      = 32 if SMOKE else 64
D_FF         = 64 if SMOKE else 128
N_HEADS      = 1                     # "each layer just a single head"
DEPTHS       = [1, 2, 3]             # 1-, 2-, 3-layer students
SEEDS        = [0, 1] if SMOKE else [0, 1, 2]

# ---- training ----
# 30k fits previous_token well for all depths; it is a ONE-TIME cost (every trained model is
# cached to Drive, so later analysis runs never retrain). Raise for harder teachers; changing
# it invalidates the model cache automatically (the config hash is in the filename).
N_STEPS      = 300 if SMOKE else 30000
BATCH_SIZE   = 128 if SMOKE else 256
LR           = 1e-3
WEIGHT_DECAY = 0.0
FORCE_RETRAIN = False               # True to ignore the model cache and retrain

# ---- scoring ----
N_EVAL_ATTN  = 64  if SMOKE else 256   # samples for methods (1)&(2) (cheap, no grad)
N_EVAL_TRANS = 32  if SMOKE else 128   # samples for method (3) (readout grads)
N_PROBES     = 8   if SMOKE else 16    # R: Hutchinson probes for the functional magnitude
EVAL_SEED    = 20260720                # fixed so scores are comparable across models/seeds
FORCE_RESCORE = False               # True to ignore the score cache and recompute

# ---- optional per-layer mean/std attention figure (across inputs) ----
ATTN_VIZ_DEPTH = 3                   # which depth to visualise (must be in DEPTHS); None to skip
ATTN_VIZ_SEED  = None                # None -> SEEDS[0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


# %%
# ============================== 1. Drive / cache ==============================
def _setup_cache_dir():
    """Mount Drive on Colab; fall back to a local dir elsewhere. Returns the cache path."""
    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive", force_remount=False)
        base = "/content/drive/MyDrive/nope_specialisation"
    except Exception:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__))
                            if "__file__" in globals() else ".", "nope_specialisation_cache")
    os.makedirs(base, exist_ok=True)
    print(f"[cache] using {base}")
    return base

CACHE_DIR = _setup_cache_dir()

def _config_tag():
    """Hash of the hyperparameters that make a trained model comparable/cacheable."""
    cfg = dict(teacher=TEACHER_TYPE, T=SEQ_LEN, d=D_MODEL, dff=D_FF, heads=N_HEADS,
               steps=N_STEPS, bs=BATCH_SIZE, lr=LR, wd=WEIGHT_DECAY)
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8], cfg

CFG_TAG, CFG_DICT = _config_tag()

def _eval_tag():
    cfg = dict(cfg=CFG_TAG, na=N_EVAL_ATTN, nt=N_EVAL_TRANS, R=N_PROBES, es=EVAL_SEED)
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

EVAL_TAG = _eval_tag()


# %%
# ============================== 2. Teacher ==============================
def make_teacher_pattern(seq_len, pattern_type):
    P = torch.zeros(seq_len, seq_len)
    if pattern_type == "previous_token":
        P[0, 0] = 1.0
        for i in range(1, seq_len):
            P[i, i - 1] = 1.0
    elif pattern_type == "diagonal":
        for i in range(seq_len):
            P[i, i] = 1.0
    elif pattern_type == "first_token":
        P[:, 0] = 1.0
    elif pattern_type == "uniform_causal":
        P = torch.tril(torch.ones(seq_len, seq_len))
        P = P / P.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown teacher pattern: {pattern_type}")
    return P

TEACHER_P = make_teacher_pattern(SEQ_LEN, TEACHER_TYPE).to(DEVICE, DTYPE)

def teacher_forward(x):
    """x:(B,T,D) -> Y=P X:(B,T,D)."""
    return torch.einsum("qk,bkd->bqd", TEACHER_P, x)


# %%
# ============================== 3. Student model ==============================
class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask, return_internals=False):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # (B,H,T,hd)
        scale = self.head_dim ** 0.5
        logits = (q @ k.transpose(-2, -1)) / scale         # (B,H,T,T)
        logits = logits.masked_fill(mask, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        msg = attn @ v                                     # (B,H,T,hd)  <- transport site
        out = self.out(msg.transpose(1, 2).reshape(B, T, D))
        if return_internals:
            return out, attn, v, msg
        return out, attn


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, mask):
        attn_out, attn_w = self.attn(self.ln1(x), mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_w


class StudentTransformer(nn.Module):
    """Minimal NoPE causal transformer. Prediction = ln_f(h) (matches the teacher's D-vector)."""
    def __init__(self, d_model, n_heads, n_layers, d_ff):
        super().__init__()
        self.n_layers = n_layers
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        attn_weights = []
        for block in self.blocks:
            x, w = block(x, mask)
            attn_weights.append(w)
        return self.ln_f(x), attn_weights


def make_mask(seq_len):
    m = torch.triu(torch.ones(seq_len, seq_len, device=DEVICE), diagonal=1).bool()
    return m.unsqueeze(0).unsqueeze(0)                      # (1,1,T,T)

MASK = make_mask(SEQ_LEN)


# %%
# ============================== 4. Train / load ==============================
def train_student(n_layers, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = StudentTransformer(D_MODEL, N_HEADS, n_layers, D_FF).to(DEVICE, DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_STEPS)
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 10_000)

    model.train()
    last = 0.0
    for step in range(1, N_STEPS + 1):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=DEVICE, dtype=DTYPE, generator=gen)
        y_hat, _ = model(x, MASK)
        loss = F.mse_loss(y_hat, teacher_forward(x))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        last = loss.item()
    return model, last


@torch.no_grad()
def eval_mse(model, n=4096):
    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(EVAL_SEED)
    tot, cnt = 0.0, 0
    for start in range(0, n, BATCH_SIZE):
        bs = min(BATCH_SIZE, n - start)
        x = torch.randn(bs, SEQ_LEN, D_MODEL, device=DEVICE, dtype=DTYPE, generator=gen)
        y_hat, _ = model(x, MASK)
        tot += F.mse_loss(y_hat, teacher_forward(x), reduction="sum").item()
        cnt += bs * SEQ_LEN * D_MODEL
    return tot / cnt


def model_cache_path(n_layers, seed):
    return os.path.join(CACHE_DIR, f"student_L{n_layers}_seed{seed}_{CFG_TAG}.pt")


def get_student(n_layers, seed):
    """Load from cache if present & config matches, else train and cache."""
    path = model_cache_path(n_layers, seed)
    if (not FORCE_RETRAIN) and os.path.exists(path):
        blob = torch.load(path, map_location=DEVICE, weights_only=False)
        model = StudentTransformer(D_MODEL, N_HEADS, n_layers, D_FF).to(DEVICE, DTYPE)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        print(f"[load ] L{n_layers} seed{seed}  train_mse={blob['train_mse']:.3e} "
              f"eval_mse={blob['eval_mse']:.3e}  (cached)")
        return model, blob["train_mse"], blob["eval_mse"], True

    model, train_mse = train_student(n_layers, seed)
    e_mse = eval_mse(model)
    torch.save({"state_dict": model.state_dict(), "config": CFG_DICT,
                "train_mse": train_mse, "eval_mse": e_mse,
                "n_layers": n_layers, "seed": seed}, path)
    print(f"[train] L{n_layers} seed{seed}  train_mse={train_mse:.3e} "
          f"eval_mse={e_mse:.3e}  -> cached")
    model.eval()
    return model, train_mse, e_mse, False


# %%
# ============================== 5. Shared capture helpers ==============================
def forward_capture(model, x, grad=False):
    """Manual forward returning per-layer residual inputs, attention, values, messages, pred."""
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        res_in, A, V, MSG = [], [], [], []
        h = x
        for block in model.blocks:
            res_in.append(h)
            out, attn, v, msg = block.attn(block.ln1(h), MASK, return_internals=True)
            A.append(attn); V.append(v); MSG.append(msg)
            h = h + out
            h = h + block.ff(block.ln2(h))
        Y = model.ln_f(h)
    return dict(res_in=res_in, A=A, V=V, MSG=MSG, Y=Y)


def swap_positions(x, i, j):
    """Swap slots i,j in x:(B,T,D) -> content transposition / node transposition."""
    xs = x.clone()
    xs[:, i], xs[:, j] = x[:, j].clone(), x[:, i].clone()
    return xs


def _cossim_last2(a, b, eps=1e-8):
    """Cosine similarity over the last dim for (...,2) vectors."""
    dot = (a * b).sum(-1)
    return dot / (a.norm(dim=-1).clamp(min=eps) * b.norm(dim=-1).clamp(min=eps))


def eval_inputs(n):
    gen = torch.Generator(device=DEVICE).manual_seed(EVAL_SEED)
    return torch.randn(n, SEQ_LEN, D_MODEL, device=DEVICE, dtype=DTYPE, generator=gen)


def _all_pairs(T):
    return [(i, j) for i in range(T) for j in range(i + 1, T)]


# %%
# ============================== 6. Methods (1)&(2): attention cosine ==============================
def _accumulate_cosine(A_clean_l, Aprime_l, i, j, num_pos, num_sym, den, l):
    """Alpha-weighted cosine following/invariance for one layer, one (i,j) swap.

    A_clean_l, Aprime_l: (B,H,T,T). H=1. Accumulates per query q into (T,) buffers.
    """
    a_i = A_clean_l[:, :, :, i]                             # (B,H,T)
    a_j = A_clean_l[:, :, :, j]
    ap_i = Aprime_l[:, :, :, i]
    ap_j = Aprime_l[:, :, :, j]
    v_ij = torch.stack([a_i, a_j], dim=-1)                 # (B,H,T,2) original
    v_ji = torch.stack([a_j, a_i], dim=-1)                 # content-followed reference
    v_pr = torch.stack([ap_i, ap_j], dim=-1)               # after swap
    pos = _cossim_last2(v_pr, v_ij)                        # attention stayed -> positional
    sym = _cossim_last2(v_pr, v_ji)                        # attention followed -> semantic
    alpha = (a_i - a_j).abs()                              # (B,H,T) mass moved
    # sum over batch & heads -> (T,)
    num_pos[l] += (pos * alpha).sum(dim=(0, 1))
    num_sym[l] += (sym * alpha).sum(dim=(0, 1))
    den[l] += alpha.sum(dim=(0, 1))


def scores_attention(model, mode):
    """mode='local' (swap at each layer input) or 'global' (swap at model input).

    Returns pos[(l)]:(T,), sym[(l)]:(T,) -- alpha-weighted cosine scores per (layer, query).
    """
    n_layers = model.n_layers
    x = eval_inputs(N_EVAL_ATTN)
    clean = forward_capture(model, x, grad=False)
    A_clean = clean["A"]
    pairs = _all_pairs(SEQ_LEN)

    num_pos = {l: torch.zeros(SEQ_LEN, device=DEVICE) for l in range(n_layers)}
    num_sym = {l: torch.zeros(SEQ_LEN, device=DEVICE) for l in range(n_layers)}
    den = {l: torch.zeros(SEQ_LEN, device=DEVICE) for l in range(n_layers)}

    with torch.no_grad():
        for (i, j) in pairs:
            if mode == "global":
                x_sw = swap_positions(x, i, j)
                A_sw = forward_capture(model, x_sw, grad=False)["A"]
                for l in range(n_layers):
                    _accumulate_cosine(A_clean[l], A_sw[l], i, j, num_pos, num_sym, den, l)
            elif mode == "local":
                for l in range(n_layers):
                    h_sw = swap_positions(clean["res_in"][l], i, j)
                    _, A_sw_l, _, _ = model.blocks[l].attn(
                        model.blocks[l].ln1(h_sw), MASK, return_internals=True)
                    _accumulate_cosine(A_clean[l], A_sw_l, i, j, num_pos, num_sym, den, l)
            else:
                raise ValueError(mode)

    pos = {l: (num_pos[l] / den[l].clamp(min=1e-8)).cpu().numpy() for l in range(n_layers)}
    sym = {l: (num_sym[l] / den[l].clamp(min=1e-8)).cpu().numpy() for l in range(n_layers)}
    return pos, sym


# %%
# ============================== 7. Method (3): transport ==============================
def _readout_probe_grads(model, clean, probes):
    """g[(l,q)] = d(<u_r, yhat_q>)/d o^l_q for R probes, batched via is_grads_batched.

    Returns dict[(l,q)] -> (R,B,hd). clean must have been captured with grad=True.
    """
    n_layers = model.n_layers
    Y = clean["Y"]                                         # (B,T,D)  requires grad
    B = Y.shape[0]
    R = probes.shape[0]
    g = {}
    for l in range(n_layers):
        msg_l = clean["MSG"][l]                            # (B,H,T,hd)  graph node
        for q in range(SEQ_LEN):
            Yq = Y[:, q, :]                                # (B,D)
            # grad_outputs[r,b,:] = u_r  -> VJP gives per-sample d(<u_r,Yq>)/d msg_l
            gout = probes.unsqueeze(1).expand(R, B, Y.shape[-1]).contiguous()  # (R,B,D)
            grad = torch.autograd.grad(Yq, msg_l, grad_outputs=gout,
                                       is_grads_batched=True, retain_graph=True)[0]
            g[(l, q)] = grad[:, :, 0, q, :].detach()       # (R,B,hd)  (H=1)
    return g


def _func_mag(g_lq, delta):
    """Hutchinson functional magnitude ||J delta|| ~ sqrt(mean_r (g_r . delta)^2).

    g_lq:(R,B,hd), delta:(B,hd) -> (B,).
    """
    proj = torch.einsum("rbd,bd->rb", g_lq, delta)         # (R,B)
    return (proj.pow(2).mean(dim=0)).clamp(min=0).sqrt()   # (B,)


def scores_transport(model):
    """Transport following/invariance off the single node-transposition base unit.

    Returns S_sem[(l)]:(T,), S_str[(l)]:(T,) -- alpha-weighted readout-weighted magnitudes.
    """
    n_layers = model.n_layers
    x = eval_inputs(N_EVAL_TRANS)
    clean = forward_capture(model, x, grad=True)           # keep graph for readout grads
    A_clean = clean["A"]
    MSG_clean = clean["MSG"]

    pgen = torch.Generator(device=DEVICE).manual_seed(EVAL_SEED + 7)
    probes = torch.randn(N_PROBES, D_MODEL, device=DEVICE, dtype=DTYPE, generator=pgen)
    g = _readout_probe_grads(model, clean, probes)         # dict[(l,q)] -> (R,B,hd)

    pairs = _all_pairs(SEQ_LEN)
    num_sem = {l: torch.zeros(SEQ_LEN, device=DEVICE) for l in range(n_layers)}
    num_str = {l: torch.zeros(SEQ_LEN, device=DEVICE) for l in range(n_layers)}
    den = {l: torch.zeros(SEQ_LEN, device=DEVICE) for l in range(n_layers)}

    with torch.no_grad():
        for (i, j) in pairs:
            sw = forward_capture(model, swap_positions(x, i, j), grad=False)
            for l in range(n_layers):
                A_c = A_clean[l]                            # (B,H,T,T)
                V_sw = sw["V"][l]                           # (B,H,T,hd)
                o_pos = (A_c @ V_sw)                        # (B,H,T,hd) clean attn, swapped vals
                o_clean = MSG_clean[l]                      # (B,H,T,hd)
                o_swap = sw["MSG"][l]
                alpha = (A_c[:, 0, :, i] - A_c[:, 0, :, j]).abs()   # (B,T) mass moved
                for q in range(SEQ_LEN):
                    d_net = (o_swap[:, 0, q, :] - o_clean[:, 0, q, :])   # (B,hd) -> structural
                    d_sem = (o_swap[:, 0, q, :] - o_pos[:, 0, q, :])     # (B,hd) -> semantic
                    a_q = alpha[:, q]                                    # (B,)
                    num_str[l][q] += (a_q * _func_mag(g[(l, q)], d_net)).sum()
                    num_sem[l][q] += (a_q * _func_mag(g[(l, q)], d_sem)).sum()
                    den[l][q] += a_q.sum()

    S_sem = {l: (num_sem[l] / den[l].clamp(min=1e-8)).cpu().numpy() for l in range(n_layers)}
    S_str = {l: (num_str[l] / den[l].clamp(min=1e-8)).cpu().numpy() for l in range(n_layers)}
    return S_sem, S_str


# %%
# ============================== 8. Score orchestration + cache ==============================
def scores_cache_path(n_layers, seed):
    return os.path.join(CACHE_DIR, f"scores_L{n_layers}_seed{seed}_{CFG_TAG}_{EVAL_TAG}.pt")


def compute_scores(model, n_layers, seed):
    """All three methods for one model. Cached by (config, eval-config). Returns a dict:
       {method: {'pos': {l:(T,)}, 'sym': {l:(T,)}}} with method in {local,global,transport}.
    """
    path = scores_cache_path(n_layers, seed)
    if (not FORCE_RESCORE) and os.path.exists(path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return blob["scores"]

    loc_pos, loc_sym = scores_attention(model, "local")
    glo_pos, glo_sym = scores_attention(model, "global")
    tr_sem, tr_str = scores_transport(model)

    def _np(d):
        return {int(l): np.asarray(v) for l, v in d.items()}

    scores = {
        "local":     {"x": _np(loc_pos), "y": _np(loc_sym)},   # x=positional, y=semantic
        "global":    {"x": _np(glo_pos), "y": _np(glo_sym)},
        "transport": {"x": _np(tr_str),  "y": _np(tr_sem)},    # x=structural, y=semantic
    }
    torch.save({"scores": scores, "cfg": CFG_TAG, "eval": EVAL_TAG}, path)
    return scores


# %%
# ============================== 9. Run all models ==============================
METHOD_ORDER = ["local", "global", "transport"]
METHOD_TITLE = {
    "local":     "(1) LOCAL content swap  (per-head input, attention)",
    "global":    "(2) GLOBAL content swap  (model input, attention)",
    "transport": "(3) TRANSPORT following/invariance  (readout-weighted)",
}

def run_all():
    train_rows = []
    # scores_by[method][depth] -> list over seeds of dict{l:(T,)} for x and y
    agg = {m: {L: {"x": [], "y": []} for L in DEPTHS} for m in METHOD_ORDER}

    for L in DEPTHS:
        for s in SEEDS:
            model, tr_mse, ev_mse, cached = get_student(L, s)
            train_rows.append(dict(depth=L, seed=s, steps=N_STEPS,
                                   train_mse=tr_mse, eval_mse=ev_mse,
                                   source="cache" if cached else "trained"))
            sc = compute_scores(model, L, s)
            for m in METHOD_ORDER:
                agg[m][L]["x"].append(sc[m]["x"])
                agg[m][L]["y"].append(sc[m]["y"])
    return train_rows, agg


def seed_mean_points(list_of_dicts):
    """list over seeds of {l:(T,)} -> stacked mean over seeds. Returns {l:(T,)}."""
    layers = sorted(list_of_dicts[0].keys())
    out = {}
    for l in layers:
        out[l] = np.mean(np.stack([d[l] for d in list_of_dicts], axis=0), axis=0)  # (T,)
    return out


# %%
# ============================== 10. Report + figure ==============================
def print_training_table(train_rows):
    print("\n================= TRAINING SUMMARY =================")
    if _HAVE_PANDAS:
        df = pd.DataFrame(train_rows)
        df = df[["depth", "seed", "steps", "train_mse", "eval_mse", "source"]]
        with pd.option_context("display.float_format", lambda v: f"{v:.3e}"):
            print(df.to_string(index=False))
        summ = (df.groupby("depth")[["train_mse", "eval_mse"]]
                  .mean().reset_index())
        print("\n-- mean over seeds --")
        with pd.option_context("display.float_format", lambda v: f"{v:.3e}"):
            print(summ.to_string(index=False))
    else:  # pragma: no cover
        print(f"{'depth':>5} {'seed':>4} {'steps':>7} {'train_mse':>12} {'eval_mse':>12} {'src':>8}")
        for r in train_rows:
            print(f"{r['depth']:>5} {r['seed']:>4} {r['steps']:>7} "
                  f"{r['train_mse']:>12.3e} {r['eval_mse']:>12.3e} {r['source']:>8}")
    print("===================================================\n")


def make_figure(agg, out_path):
    nrows, ncols = len(METHOD_ORDER), len(DEPTHS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.9 * nrows))
    axes = np.atleast_2d(axes)
    cmap = plt.get_cmap("viridis")
    max_layer = max(DEPTHS) - 1

    for r, m in enumerate(METHOD_ORDER):
        for c, L in enumerate(DEPTHS):
            ax = axes[r, c]
            x_by_l = seed_mean_points(agg[m][L]["x"])
            y_by_l = seed_mean_points(agg[m][L]["y"])
            xs_all, ys_all = [], []
            for l in sorted(x_by_l.keys()):
                xs, ys = x_by_l[l], y_by_l[l]
                xs_all.append(xs); ys_all.append(ys)
                color = cmap(l / max(max_layer, 1))
                ax.scatter(xs, ys, s=46, color=color, edgecolors="black",
                           linewidth=0.4, alpha=0.9, zorder=3,
                           label=f"layer {l}")
            xs_all = np.concatenate(xs_all); ys_all = np.concatenate(ys_all)

            # equal-score reference + origin guides
            lo = float(min(xs_all.min(), ys_all.min()))
            hi = float(max(xs_all.max(), ys_all.max()))
            pad = 0.05 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--",
                    color="gray", lw=0.8, zorder=1)
            if m in ("local", "global"):     # cosines live in [-1,1]
                ax.axhline(0, color="gray", lw=0.5, ls=":")
                ax.axvline(0, color="gray", lw=0.5, ls=":")

            if r == 0:
                ax.set_title(f"{L}-layer student", fontsize=12, fontweight="bold")
            ax.set_xlabel("structural / positional  (x)")
            ax.set_ylabel("semantic  (y)")
            if L == DEPTHS[-1] and len(x_by_l) > 1:
                ax.legend(fontsize=7, loc="best", framealpha=0.8)
            # row label on the left
            if c == 0:
                ax.text(-0.30, 0.5, METHOD_TITLE[m], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10,
                        fontweight="bold")

    fig.suptitle("NoPE teacher-student specialisation: semantic (y) vs structural (x)\n"
                 f"teacher={TEACHER_TYPE}, T={SEQ_LEN}, d={D_MODEL}, "
                 f"{len(SEEDS)} seeds (points = per (layer, query), seed-averaged, coloured by layer)",
                 fontsize=13, y=1.005)
    fig.tight_layout(rect=[0.02, 0, 1, 0.99])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[fig  ] saved {out_path}")
    try:
        plt.show()
    except Exception:
        pass
    return fig


def _normalise_channel(vals_by_l):
    """S~ = raw / mean, with the mean taken over ALL (layer, query) points in the panel."""
    pooled = np.concatenate([vals_by_l[l] for l in sorted(vals_by_l.keys())])
    mean = float(pooled.mean())
    denom = mean if abs(mean) > 1e-8 else 1.0
    return {l: vals_by_l[l] / denom for l in vals_by_l}


def make_jd_figure(agg, out_path):
    """Companion plane: D = selectivity = (S~sem - S~str)/2 (x) vs J = joint strength =
    (S~sem + S~str)/2 (y), per (method, depth), points = per (layer, query), seed-averaged.

    NB (kept out of the axis labels by request): unlike the two-INTERVENTION case, S_sem and
    S_str here are an additive split of ONE swap's delta (d_net = d_valpart + d_sem), so they are
    algebraically linked and can share/cancel signal. D therefore reads as a net-delivery-vs-
    re-routing balance within one swap rather than a preference between two independent causes;
    the reading is cleanest at the extremes (pure structural D<0, pure semantic D>0).
    """
    nrows, ncols = len(METHOD_ORDER), len(DEPTHS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.9 * nrows))
    axes = np.atleast_2d(axes)
    cmap = plt.get_cmap("viridis")
    max_layer = max(DEPTHS) - 1

    for r, m in enumerate(METHOD_ORDER):
        for c, L in enumerate(DEPTHS):
            ax = axes[r, c]
            str_t = _normalise_channel(seed_mean_points(agg[m][L]["x"]))   # S~structural
            sem_t = _normalise_channel(seed_mean_points(agg[m][L]["y"]))   # S~semantic
            J_all, D_all = [], []
            for l in sorted(str_t.keys()):
                J = 0.5 * (sem_t[l] + str_t[l])
                D = 0.5 * (sem_t[l] - str_t[l])
                J_all.append(J); D_all.append(D)
                color = cmap(l / max(max_layer, 1))
                ax.scatter(D, J, s=46, color=color, edgecolors="black",
                           linewidth=0.4, alpha=0.9, zorder=3, label=f"layer {l}")
            J_all = np.concatenate(J_all); D_all = np.concatenate(D_all)

            ax.axvline(0, color="gray", lw=0.9, ls="--", zorder=1)     # no-preference boundary
            ax.axhline(1.0, color="gray", lw=0.6, ls=":", zorder=1)    # mean joint strength
            # annotate the two horizontal directions
            xmax = float(np.abs(D_all).max()) * 1.05 + 1e-9
            ax.set_xlim(-xmax, xmax)
            ax.text(0.98, 0.02, "semantic →", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="dimgray")
            ax.text(0.02, 0.02, "← structural", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=8, color="dimgray")

            if r == 0:
                ax.set_title(f"{L}-layer student", fontsize=12, fontweight="bold")
            ax.set_xlabel(r"selectivity  $D=(\tilde S_{sem}-\tilde S_{str})/2$")
            ax.set_ylabel(r"joint strength  $J=(\tilde S_{sem}+\tilde S_{str})/2$")
            if L == DEPTHS[-1] and len(str_t) > 1:
                ax.legend(fontsize=7, loc="best", framealpha=0.8)
            if c == 0:
                ax.text(-0.30, 0.5, METHOD_TITLE[m], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10,
                        fontweight="bold")

    fig.suptitle("NoPE teacher-student specialisation: D–J plane "
                 "(strength vs preference)\n"
                 f"teacher={TEACHER_TYPE}, T={SEQ_LEN}, d={D_MODEL}, "
                 f"{len(SEEDS)} seeds; S~ = raw/mean. High J = influential, low J = inert; "
                 "D sign = semantic (right) / structural (left)",
                 fontsize=13, y=1.005)
    fig.tight_layout(rect=[0.02, 0, 1, 0.99])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[fig  ] saved {out_path}")
    try:
        plt.show()
    except Exception:
        pass
    return fig


def make_attention_figure(depth=None, seed=None, out_path=None):
    """Per-layer MEAN and STD of attention across random inputs, for one model.

    A 2 x n_layers grid: row 1 = mean attention A[q,k] (positional if input-invariant),
    row 2 = std across inputs (high = content-dependent). depth is configurable (e.g. L2/L4);
    uses a single seed (attention solutions differ across seeds, so they are not averaged).
    """
    depth = depth if depth is not None else ATTN_VIZ_DEPTH
    seed = seed if seed is not None else (ATTN_VIZ_SEED if ATTN_VIZ_SEED is not None else SEEDS[0])
    model, *_ = get_student(depth, seed)
    x = eval_inputs(N_EVAL_ATTN)
    A = forward_capture(model, x, grad=False)["A"]         # list of (B,H,T,T), H=1
    means = [a[:, 0].mean(0).cpu().numpy() for a in A]     # (T,T) per layer
    stds = [a[:, 0].std(0).cpu().numpy() for a in A]

    nL = depth
    fig, axes = plt.subplots(2, nL, figsize=(4.3 * nL, 8.2))
    axes = np.atleast_2d(axes)
    for l in range(nL):
        ax = axes[0, l]
        im = ax.imshow(means[l], cmap="Blues", vmin=0, vmax=max(means[l].max(), 1e-8))
        ax.set_title(f"Layer {l}: mean attention", fontsize=11)
        ax.set_xlabel("key pos"); ax.set_ylabel("query pos" if l == 0 else "")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax = axes[1, l]
        im = ax.imshow(stds[l], cmap="Oranges", vmin=0)
        ax.set_title(f"Layer {l}: attention std\n(high = content-dependent)", fontsize=11)
        ax.set_xlabel("key pos"); ax.set_ylabel("query pos" if l == 0 else "")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"NoPE {depth}-layer student (seed {seed}): attention mean & std across "
                 f"{N_EVAL_ATTN} inputs\nlow std across inputs = positional (content-invariant) "
                 "attention",
                 fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    if out_path:
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        print(f"[fig  ] saved {out_path}")
    try:
        plt.show()
    except Exception:
        pass
    return fig


# %%
# ============================== 11. Main ==============================
def main():
    print(f"[run  ] device={DEVICE}  depths={DEPTHS}  seeds={SEEDS}  "
          f"cfg={CFG_TAG}  eval={EVAL_TAG}  smoke={SMOKE}")
    train_rows, agg = run_all()
    print_training_table(train_rows)
    scatter_path = os.path.join(CACHE_DIR, f"fig_specialisation_scatter_{CFG_TAG}_{EVAL_TAG}.png")
    jd_path = os.path.join(CACHE_DIR, f"fig_specialisation_JDplane_{CFG_TAG}_{EVAL_TAG}.png")
    make_figure(agg, scatter_path)
    make_jd_figure(agg, jd_path)
    if ATTN_VIZ_DEPTH is not None:
        if ATTN_VIZ_DEPTH in DEPTHS:
            attn_path = os.path.join(CACHE_DIR,
                                     f"fig_attention_L{ATTN_VIZ_DEPTH}_{CFG_TAG}_{EVAL_TAG}.png")
            make_attention_figure(depth=ATTN_VIZ_DEPTH, out_path=attn_path)
        else:
            print(f"[warn ] ATTN_VIZ_DEPTH={ATTN_VIZ_DEPTH} not in DEPTHS={DEPTHS}; skipping attn fig")
    return train_rows, agg


if __name__ == "__main__":
    main()
