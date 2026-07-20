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

This script trains 3 seeds of each 1-, 2- and 3-layer student across three FAMILIES (variant x
task) -- (nope, positional), (rope, positional), (nope, semantic) -- caching every model so a
later run never retrains, prints a training table, and draws FIVE figures: the 3x3 method x depth
scatter and the D-J plane (both for the PRIMARY nope/positional family), the per-layer mean/std
attention figure, a FAITHFULNESS figure vs a positional-ness oracle that overlays the families
(the RoPE control removes the local-vs-global gap; the semantic control is a genuine content task
where local should NOT mislabel), and a TRANSPORT-IMPORTANCE figure (lesion the final head inert
while keeping its attention pattern: global-attention score unchanged, transport J -> 0).

The 3x3 scatter has one column per depth and one row per scoring method:

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
import copy
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
DEPTHS       = [1, 2, 3, 4, 5]       # student depths (depth-scaling of the local mislabel)
SEEDS        = [0, 1] if SMOKE else [0, 1, 2]

# ---- experiment families: (variant, task) ----
#   variant "nope" = no positional encoding (position must be CONSTRUCTED in-stream over depth);
#           "rope" = rotary PE (position injected explicitly every layer -> faithful at any depth,
#                    the control that attributes the local-vs-global gap to in-stream construction).
#   task    "positional" = Y = P X (fixed positional teacher);
#           "semantic"   = content self-attention teacher (routing depends on content, not slot).
# The scatter / D-J / attention figures use PRIMARY; the faithfulness figure overlays all families.
FAMILIES = [("nope", "positional"), ("rope", "positional"), ("nope", "semantic")]
PRIMARY  = ("nope", "positional")

# Sharpness of the SEMANTIC teacher's content routing. Raw logits are ~unit-variance -> a nearly
# uniform (positional-averaging) softmax; SEM_SHARP>1 concentrates routing on the best-matching
# earlier token so the operation is genuinely CONTENT-selective (high attention variance).
SEM_SHARP = 6.0

# ---- training ----
# Base step count (used for any depth NOT overridden below). One-time cost -- every trained model
# is cached; changing steps for a depth invalidates ONLY that depth's cache (the per-depth step
# count is folded into that model's config hash).
N_STEPS      = int(os.environ.get("NOPE_STEPS", "0")) or (300 if SMOKE else 30000)
# Per-depth OVERRIDE: deeper NoPE students need more steps to converge (fixed steps undertrains
# them). Depths absent here use N_STEPS -- so keeping L1-L3 at N_STEPS preserves their cache while
# only L4/L5 (new step counts -> new hash) retrain.
STEPS_BY_DEPTH = {} if SMOKE else {4: 60000, 5: 90000}
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
ATTN_VIZ_DEPTH = 5                   # which depth to visualise (must be in DEPTHS); None to skip
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

def _config_dict(steps):
    """Hyperparameters that make a trained model comparable/cacheable (parametrised by steps)."""
    return dict(teacher=TEACHER_TYPE, sem=f"strict_earlier_sharp{SEM_SHARP}", T=SEQ_LEN, d=D_MODEL,
                dff=D_FF, heads=N_HEADS, steps=steps, bs=BATCH_SIZE, lr=LR, wd=WEIGHT_DECAY)

def _hash_cfg(cfg):
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]

def steps_for_depth(n_layers):
    return STEPS_BY_DEPTH.get(n_layers, N_STEPS)

def model_tag(n_layers):
    """Per-depth config hash: depths at N_STEPS share the base tag (their cache is preserved);
    an overridden depth gets a new tag and retrains."""
    return _hash_cfg(_config_dict(steps_for_depth(n_layers)))

# Base tag (steps=N_STEPS) -- used for figure/eval filenames and by any depth not overridden.
CFG_DICT = _config_dict(N_STEPS)
CFG_TAG = _hash_cfg(CFG_DICT)

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

# Fixed query-key map for the SEMANTIC teacher (content self-attention). Shared across all
# models/seeds so the semantic task is identical everywhere; random so it is a non-trivial
# content routing (not a copy-self identity).
_sem_gen = torch.Generator().manual_seed(1234)
TEACHER_W = (torch.randn(D_MODEL, D_MODEL, generator=_sem_gen) / (D_MODEL ** 0.5)).to(DEVICE, DTYPE)
# Semantic teacher attends to STRICTLY EARLIER tokens (self excluded), so the target is a
# content-selected *other* token and cannot be solved by identity/copy -> the student must learn
# a genuine content-routing head. q=0 has no earlier token, so it attends to itself.
_SEM_MASK = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), diagonal=0).bool()  # masks k>=q
_SEM_MASK[0, 0] = False

def teacher_forward(x, task="positional"):
    """x:(B,T,D) -> teacher target Y:(B,T,D).

    positional: Y = P X  (routing by SLOT; equivariant<->invariant flips are positional).
    semantic  : Y[q] = sum_{k<q} softmax_k(SEM_SHARP*(x[q] W).x[k]) x[k]  (CONTENT routing, self excl).
    """
    if task == "positional":
        return torch.einsum("qk,bkd->bqd", TEACHER_P, x)
    if task == "semantic":
        logits = SEM_SHARP * torch.einsum("bqd,bkd->bqk", x @ TEACHER_W, x) / (D_MODEL ** 0.5)
        logits = logits.masked_fill(_SEM_MASK.unsqueeze(0), float("-inf"))
        A = torch.softmax(logits, dim=-1)
        return torch.einsum("bqk,bkd->bqd", A, x)
    raise ValueError(f"Unknown task: {task}")


# %%
# ============================== 3. Student model ==============================
def _apply_rope(x):
    """Rotary positional encoding by ABSOLUTE slot index. x:(B,H,T,hd), hd even. Position enters
    Q/K explicitly here (not via content), so a RoPE head can be positional at any layer."""
    B, H, T, hd = x.shape
    half = hd // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device, dtype=x.dtype) / hd))
    t = torch.arange(T, device=x.device, dtype=x.dtype)
    freqs = torch.outer(t, inv_freq)                       # (T, half)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None, None]   # (1,1,T,hd)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None, None]
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rot * sin


class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_rope=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask, return_internals=False):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # (B,H,T,hd)
        if self.use_rope:
            q, k = _apply_rope(q), _apply_rope(k)          # position injected explicitly (not V)
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
    def __init__(self, d_model, n_heads, d_ff, use_rope=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads, use_rope=use_rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, mask):
        attn_out, attn_w = self.attn(self.ln1(x), mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_w


class StudentTransformer(nn.Module):
    """Minimal causal transformer (NoPE or RoPE). Prediction = ln_f(h) (teacher's D-vector)."""
    def __init__(self, d_model, n_heads, n_layers, d_ff, use_rope=False):
        super().__init__()
        self.n_layers = n_layers
        self.use_rope = use_rope
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, use_rope=use_rope)
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
def train_student(variant, task, n_layers, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    steps = steps_for_depth(n_layers)
    use_rope = (variant == "rope")
    model = StudentTransformer(D_MODEL, N_HEADS, n_layers, D_FF, use_rope=use_rope).to(DEVICE, DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator(device=DEVICE).manual_seed(seed + 10_000)

    model.train()
    last = 0.0
    for step in range(1, steps + 1):
        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=DEVICE, dtype=DTYPE, generator=gen)
        y_hat, _ = model(x, MASK)
        loss = F.mse_loss(y_hat, teacher_forward(x, task))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        last = loss.item()
    return model, last


@torch.no_grad()
def eval_mse(model, task, n=4096):
    model.eval()
    gen = torch.Generator(device=DEVICE).manual_seed(EVAL_SEED)
    tot, cnt = 0.0, 0
    for start in range(0, n, BATCH_SIZE):
        bs = min(BATCH_SIZE, n - start)
        x = torch.randn(bs, SEQ_LEN, D_MODEL, device=DEVICE, dtype=DTYPE, generator=gen)
        y_hat, _ = model(x, MASK)
        tot += F.mse_loss(y_hat, teacher_forward(x, task), reduction="sum").item()
        cnt += bs * SEQ_LEN * D_MODEL
    return tot / cnt


def model_cache_path(variant, task, n_layers, seed):
    return os.path.join(
        CACHE_DIR, f"student_{variant}_{task}_L{n_layers}_seed{seed}_{model_tag(n_layers)}.pt")


def get_student(variant, task, n_layers, seed, force=False):
    """Load from cache if present & config matches, else train and cache. Keyed by (variant,task).
    force=True (or the FORCE_RETRAIN global) ignores the cache and retrains this model."""
    path = model_cache_path(variant, task, n_layers, seed)
    tag = f"{variant}/{task} L{n_layers} seed{seed}"
    if (not FORCE_RETRAIN) and (not force) and os.path.exists(path):
        blob = torch.load(path, map_location=DEVICE, weights_only=False)
        model = StudentTransformer(D_MODEL, N_HEADS, n_layers, D_FF,
                                   use_rope=(variant == "rope")).to(DEVICE, DTYPE)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        print(f"[load ] {tag}  train_mse={blob['train_mse']:.3e} "
              f"eval_mse={blob['eval_mse']:.3e}  (cached)")
        return model, blob["train_mse"], blob["eval_mse"], True

    model, train_mse = train_student(variant, task, n_layers, seed)
    e_mse = eval_mse(model, task)
    torch.save({"state_dict": model.state_dict(), "config": _config_dict(steps_for_depth(n_layers)),
                "variant": variant, "task": task, "train_mse": train_mse, "eval_mse": e_mse,
                "n_layers": n_layers, "seed": seed, "steps": steps_for_depth(n_layers)}, path)
    print(f"[train] {tag}  steps={steps_for_depth(n_layers)} train_mse={train_mse:.3e} "
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
def scores_cache_path(variant, task, n_layers, seed):
    return os.path.join(
        CACHE_DIR,
        f"scores_{variant}_{task}_L{n_layers}_seed{seed}_{model_tag(n_layers)}_{EVAL_TAG}.pt")


def compute_scores(model, variant, task, n_layers, seed, force=False):
    """All three methods for one model. Cached by (variant, task, config, eval-config). Returns:
       {method: {'x': {l:(T,)}, 'y': {l:(T,)}}} with method in {local,global,transport}.
    force=True (or FORCE_RESCORE) ignores the score cache -- required when the model was retrained.
    """
    path = scores_cache_path(variant, task, n_layers, seed)
    if (not FORCE_RESCORE) and (not force) and os.path.exists(path):
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

def run_all(families=None, force_retrain_depths=None):
    """Train/score every (variant, task) family x depth x seed. Returns train_rows and
    agg[(variant,task)][method][depth] = list over seeds of {l:(T,)} for 'x' and 'y'.

    force_retrain_depths: a collection of depths to retrain AND rescore from scratch, ignoring
    their cache (other depths still load from cache). E.g. {4, 5} keeps L1-L3 but refreshes L4/L5.
    """
    families = families if families is not None else FAMILIES
    fdepths = set(force_retrain_depths or [])
    train_rows = []
    agg = {fam: {m: {L: {"x": [], "y": []} for L in DEPTHS} for m in METHOD_ORDER}
           for fam in families}

    for fam in families:
        variant, task = fam
        for L in DEPTHS:
            force = L in fdepths
            for s in SEEDS:
                model, tr_mse, ev_mse, cached = get_student(variant, task, L, s, force=force)
                train_rows.append(dict(variant=variant, task=task, depth=L, seed=s,
                                       steps=steps_for_depth(L),
                                       train_mse=tr_mse, eval_mse=ev_mse,
                                       source="cache" if cached else "trained"))
                sc = compute_scores(model, variant, task, L, s, force=force)
                for m in METHOD_ORDER:
                    agg[fam][m][L]["x"].append(sc[m]["x"])
                    agg[fam][m][L]["y"].append(sc[m]["y"])
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
        df = df[["variant", "task", "depth", "seed", "steps", "train_mse", "eval_mse", "source"]]
        with pd.option_context("display.float_format", lambda v: f"{v:.3e}"):
            print(df.to_string(index=False))
        summ = (df.groupby(["variant", "task", "depth", "steps"])[["train_mse", "eval_mse"]]
                  .mean().reset_index())
        print("\n-- mean over seeds --")
        with pd.option_context("display.float_format", lambda v: f"{v:.3e}"):
            print(summ.to_string(index=False))
    else:  # pragma: no cover
        print(f"{'variant':>8} {'task':>11} {'depth':>5} {'seed':>4} {'train_mse':>12} {'eval_mse':>12}")
        for r in train_rows:
            print(f"{r['variant']:>8} {r['task']:>11} {r['depth']:>5} {r['seed']:>4} "
                  f"{r['train_mse']:>12.3e} {r['eval_mse']:>12.3e}")
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


def make_attention_figure(depth=None, seed=None, family=None, out_path=None):
    """Per-layer MEAN and STD of attention across random inputs, for one model.

    A 2 x n_layers grid: row 1 = mean attention A[q,k] (positional if input-invariant),
    row 2 = std across inputs (high = content-dependent). depth is configurable (e.g. L2/L4);
    uses a single seed (attention solutions differ across seeds, so they are not averaged).
    """
    depth = depth if depth is not None else ATTN_VIZ_DEPTH
    seed = seed if seed is not None else (ATTN_VIZ_SEED if ATTN_VIZ_SEED is not None else SEEDS[0])
    variant, task = family if family is not None else PRIMARY
    model, *_ = get_student(variant, task, depth, seed)
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

    fig.suptitle(f"{variant}/{task} {depth}-layer student (seed {seed}): attention mean & std "
                 f"across {N_EVAL_ATTN} inputs\nlow std across inputs = positional "
                 "(content-invariant) attention",
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


def compute_oracle(variant, task, depth, seeds=None):
    """Ground-truth positional-ness per (layer, query): high = attention is content-INVARIANT
    across random inputs (low std) = positional. Independent of the swap-based scores. Seed-avg.
    Returns {l: (T,)} in [0, 1]. Normalised by a GLOBAL std ceiling so it is comparable across
    families (a semantic model has genuinely high std -> low positional fraction).
    """
    seeds = seeds if seeds is not None else SEEDS
    per = []
    for s in seeds:
        model, *_ = get_student(variant, task, depth, s)
        A = forward_capture(model, eval_inputs(N_EVAL_ATTN), grad=False)["A"]
        stds = {}
        for l, a in enumerate(A):
            sqk = a[:, 0].std(0).cpu().numpy()                       # (T,T) std across inputs
            stds[l] = np.array([sqk[q, :q + 1].mean() for q in range(SEQ_LEN)])  # avg over visible keys
        per.append(stds)
    std_avg = {l: np.mean([per[i][l] for i in range(len(seeds))], axis=0) for l in range(depth)}
    # Ceiling = std of a maximally content-dependent (uniform-random) attention row, so the
    # positional fraction is absolute (comparable across positional vs semantic families).
    ceil = 1.0 / np.sqrt(12.0)
    return {l: np.clip(1.0 - std_avg[l] / ceil, 0.0, 1.0) for l in range(depth)}


def _pos_fraction_byq(x_by_l, y_by_l):
    """Per (layer): positional fraction per query = structural / (structural + semantic) in [0,1]."""
    out = {}
    for l in x_by_l:
        xc = np.clip(x_by_l[l], 0, None); yc = np.clip(y_by_l[l], 0, None)
        out[l] = xc / (xc + yc + 1e-8)
    return out


def _final_pf_curve(agg, fam, method, depths):
    """Final-layer positional fraction vs depth for one (family, method)."""
    ys = []
    for L in depths:
        pf = _pos_fraction_byq(seed_mean_points(agg[fam][method][L]["x"]),
                               seed_mean_points(agg[fam][method][L]["y"]))
        ys.append(float(np.mean(pf[L - 1])))
    return ys


def _oracle_final_curve(fam, depths):
    return [float(np.mean(compute_oracle(fam[0], fam[1], L)[L - 1])) for L in depths]


def make_faithfulness_figure(agg, out_path):
    """Is each method FAITHFUL to a ground-truth positional-ness oracle (attention std across
    inputs)? Up to three panels, built from whichever families were trained:

      A RoPE control   : final-layer positional fraction vs depth, LOCAL method, NoPE vs RoPE
                         (+ oracles). The NoPE local curve falls with depth (position constructed
                         in-stream, then mislabelled); RoPE tracks its oracle (position explicit).
      B semantic ctrl  : final-layer positional fraction vs depth, LOCAL method, NoPE positional
                         vs NoPE semantic (+ oracles). Local diverges from the oracle only on the
                         positional task; on the semantic task it tracks it -> local is wrong
                         SPECIFICALLY when position is constructed in-stream.
      C localisation   : positional fraction per layer for the primary deepest model, all three
                         methods vs oracle -> global & transport track the oracle, local dips at
                         the late layers.
    """
    depths = sorted(DEPTHS)
    fams = list(agg.keys())
    panels = []
    if ("nope", "positional") in fams and ("rope", "positional") in fams:
        panels.append("rope")
    if ("nope", "positional") in fams and ("nope", "semantic") in fams:
        panels.append("semantic")
    if PRIMARY in fams:
        panels.append("localise")
    if not panels:
        panels = ["localise"]

    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2), squeeze=False)
    axes = axes[0]
    mcol = {"local": "#d62728", "global": "#1f77b4", "transport": "#2ca02c"}

    for ax, panel in zip(axes, panels):
        if panel == "rope":
            fam_n, fam_r = ("nope", "positional"), ("rope", "positional")
            ax.plot(depths, _final_pf_curve(agg, fam_n, "local", depths), "o-",
                    color="#d62728", label="NoPE local")
            ax.plot(depths, _oracle_final_curve(fam_n, depths), "^--", color="#d62728",
                    alpha=0.6, label="NoPE oracle")
            ax.plot(depths, _final_pf_curve(agg, fam_r, "local", depths), "o-",
                    color="#1f77b4", label="RoPE local")
            ax.plot(depths, _oracle_final_curve(fam_r, depths), "^--", color="#1f77b4",
                    alpha=0.6, label="RoPE oracle")
            ax.set_title("(A) RoPE control\nlocal mislabels only when position is in-stream")
            ax.set_xlabel("student depth L"); ax.set_xticks(depths)
        elif panel == "semantic":
            fam_p, fam_s = ("nope", "positional"), ("nope", "semantic")
            ax.plot(depths, _final_pf_curve(agg, fam_p, "local", depths), "o-",
                    color="#d62728", label="positional local")
            ax.plot(depths, _oracle_final_curve(fam_p, depths), "^--", color="#d62728",
                    alpha=0.6, label="positional oracle")
            ax.plot(depths, _final_pf_curve(agg, fam_s, "local", depths), "o-",
                    color="#9467bd", label="semantic local")
            ax.plot(depths, _oracle_final_curve(fam_s, depths), "^--", color="#9467bd",
                    alpha=0.6, label="semantic oracle")
            ax.set_title("(B) semantic control\nlocal is wrong only on the positional task")
            ax.set_xlabel("student depth L"); ax.set_xticks(depths)
        else:  # localise
            Ld = depths[-1]
            for m in METHOD_ORDER:
                ys = [float(np.mean(_pos_fraction_byq(
                    seed_mean_points(agg[PRIMARY][m][Ld]["x"]),
                    seed_mean_points(agg[PRIMARY][m][Ld]["y"]))[l])) for l in range(Ld)]
                ax.plot(range(Ld), ys, "o-", color=mcol[m], label=m)
            orc = compute_oracle(PRIMARY[0], PRIMARY[1], Ld)
            ax.plot(range(Ld), [float(np.mean(orc[l])) for l in range(Ld)], "k^--", label="oracle")
            ax.set_title(f"(C) {PRIMARY[0]}/{PRIMARY[1]} L{Ld}: per-layer\nlocal dips at late layers")
            ax.set_xlabel("layer"); ax.set_xticks(range(Ld))
        ax.axhline(0.5, color="gray", lw=0.6, ls=":")
        ax.set_ylabel("positional fraction  (1 = positional)")
        ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)

    fig.suptitle("Faithfulness to a positional-ness oracle: local attention mislabels position "
                 "constructed in-stream; global & transport stay faithful\n"
                 f"T={SEQ_LEN}, d={D_MODEL}, {len(SEEDS)} seeds",
                 fontsize=12, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[fig  ] saved {out_path}")
    try:
        plt.show()
    except Exception:
        pass
    return fig


def _lesion_final_head(model, kind):
    """Return a copy of `model` with its FINAL layer's single head made causally inert while its
    attention PATTERN is preserved: 'outproj0' zeros the output projection; 'value0' zeros V."""
    m = copy.deepcopy(model)
    attn = m.blocks[m.n_layers - 1].attn
    if kind == "outproj0":
        attn.out.weight.data.zero_()                        # delivered message never reaches ŷ
    elif kind == "value0":
        attn.qkv.weight.data[2 * D_MODEL:3 * D_MODEL].zero_()   # V=0 -> message is 0
    else:
        raise ValueError(kind)
    return m


def _final_head_scores(model):
    """(global-attention positional score, transport strength J) for the final-layer head."""
    L = model.n_layers - 1
    gpos, _ = scores_attention(model, "global")
    S_sem, S_str = scores_transport(model)
    attn_pos = float(np.mean(gpos[L]))
    J = float(np.mean((S_str[L] + S_sem[L]) / 2.0))
    return attn_pos, J


def make_transport_importance_figure(out_path):
    """Why transport, not (even global) attention: attention scores a head by what it SELECTS,
    so it cannot tell a load-bearing positional head from a causally inert one. Lesion the final
    head to be inert while keeping its attention pattern -> the global-attention positional score
    is unchanged, but the transport strength J collapses to ~0.
    """
    variant, task = PRIMARY
    Ld = DEPTHS[-1]
    model, *_ = get_student(variant, task, Ld, SEEDS[0])
    cases = [("intact", model),
             ("out_proj → 0\n(inert)", _lesion_final_head(model, "outproj0")),
             ("V → 0\n(inert)", _lesion_final_head(model, "value0"))]
    attn_vals, J_vals = [], []
    for _, mdl in cases:
        ap, J = _final_head_scores(mdl)
        attn_vals.append(ap); J_vals.append(J)
    a0 = attn_vals[0] if abs(attn_vals[0]) > 1e-8 else 1.0
    j0 = J_vals[0] if abs(J_vals[0]) > 1e-8 else 1.0
    attn_r = [a / a0 for a in attn_vals]
    J_r = [j / j0 for j in J_vals]

    fig, ax = plt.subplots(figsize=(8.2, 5))
    xs = np.arange(len(cases)); w = 0.38
    b1 = ax.bar(xs - w / 2, attn_r, w, color="#1f77b4", label="global ATTENTION positional score")
    b2 = ax.bar(xs + w / 2, J_r, w, color="#2ca02c", label="TRANSPORT strength  J")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cases])
    ax.set_ylabel("score (relative to the intact head)")
    ax.set_title(f"Why transport, not attention — final head of the {variant}/{task} L{Ld} model\n"
                 "attention scores an inert head as unchanged; transport J collapses to ~0")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[fig  ] saved {out_path}")
    try:
        plt.show()
    except Exception:
        pass
    return fig


# %%
# ============================== 11. Main ==============================
def main(force_retrain_depths=None):
    """force_retrain_depths: collection of depths to retrain+rescore from scratch (cache ignored),
    keeping all other depths cached. E.g. main(force_retrain_depths={4, 5})."""
    print(f"[run  ] device={DEVICE}  depths={DEPTHS}  seeds={SEEDS}  cfg={CFG_TAG}  eval={EVAL_TAG}"
          f"  smoke={SMOKE}  force_retrain_depths={sorted(set(force_retrain_depths or []))}")
    train_rows, agg = run_all(force_retrain_depths=force_retrain_depths)
    print_training_table(train_rows)
    # Primary scatter / D-J / attention figures use the PRIMARY family; faithfulness overlays all.
    agg_primary = agg[PRIMARY]
    scatter_path = os.path.join(CACHE_DIR, f"fig_specialisation_scatter_{CFG_TAG}_{EVAL_TAG}.png")
    jd_path = os.path.join(CACHE_DIR, f"fig_specialisation_JDplane_{CFG_TAG}_{EVAL_TAG}.png")
    make_figure(agg_primary, scatter_path)
    make_jd_figure(agg_primary, jd_path)
    faith_path = os.path.join(CACHE_DIR, f"fig_faithfulness_{CFG_TAG}_{EVAL_TAG}.png")
    make_faithfulness_figure(agg, faith_path)
    tr_imp_path = os.path.join(CACHE_DIR, f"fig_transport_importance_{CFG_TAG}_{EVAL_TAG}.png")
    make_transport_importance_figure(tr_imp_path)
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
