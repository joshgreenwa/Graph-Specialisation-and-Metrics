# Per-head semantic / structural specialisation scores

Formal reference for the **specialisation-score** methodology in this folder. It lifts the
`/carriage/` intervention vocabulary (`README.md`, `core.py`, `content.py`, `structural.py`) from
the pooled output to **individual attention heads**, giving each head a **semantic** and a
**structural** score. The equations below are exactly what `spec_head_scores.py` computes.

Scope: a GRIT-lite graph transformer — standard multi-head attention + a **static RRWP additive
bias**, content-only value, readout on a designated query node (node 0) — trained by
`spec_tasks_train.py`. The method is architecture-general within that family; the same estimator
skeleton mirrors `/carriage/`, so the two are kept consistent.

> **Historical estimator:** this spec-lite experiment retains coherent-gross (CG), averaging donor
> deltas before magnitude. The central production GRIT methodology in
> `src/graph_specialisation_metrics/specialisation/README.md` now uses graph-balanced
> eventwise-gross (EG). Do not treat the CG equations below as the current production default.

## Setup

A graph carries content `X` (role-flags + content columns) and structure `S` (node-RRWP `nd`,
pair-RRWP `pr`, and the k-hop attention **mask**). Layer `ℓ`, head `h` compute, per query node `i`:

    a^{ℓh}_{iq} = softmax_q ( q_i·k_q / √d_h  +  Linear(pr_{iq})[h] )      # attention (selection)
    o^{ℓh}_i    = Σ_q a^{ℓh}_{iq} · V x_q                                  # transport (routed value)

`o^{ℓh}_i` is the **transport site** — the message head `h` delivers to node `i`, the same object
`mechanistic_operator_analysis.py` reads as `Iᵢⱼ`. The model output is `ŷ = head(x^L_0)`.

Per-head readout gradient at the **clean** input (the per-head analogue of the pooled `g = ∂ŷ/∂h^L`):

    φ^{ℓh}_i = ∂ŷ / ∂o^{ℓh}_i                                             # [d_h], output-reach of the head's message

## The two interventions (from `/carriage/`, applied at the graph input)

**Semantic — donor swap** (`content.py`). Replace source node `j`'s **content** with a REAL donor
content row sampled from **another graph** (on-manifold), holding structure — node/pair-RRWP, mask,
**and the role-flags** — fixed. `K` donors are drawn per source and the transport **delta is
averaged over donors BEFORE any magnitude** (README Eq 3.5/3.6):

    X_{j→x̃},  S fixed        Δ̄o^{ℓh}_i(j) = (1/K) Σ_k [ o^{ℓh}_i(clean) − o^{ℓh}_i(X_{j→x̃_k}) ]

**Structural — node transposition at a fixed anchor** (`structural.py`). Conjugate the
structure-derived **RRWP features** by the transposition `P_(u v)` — swap node-RRWP rows `u↔v` and
swap rows AND columns `u↔v` of pair-RRWP (`r → P_π r P_πᵀ`) — holding content (incl. role-flags)
fixed. The anchor `u` is the source; the partner `v` is **degree-matched** and marginalised over `K`
draws (the donor-average recipe on the structural side):

    S_{u↔v}(RRWP),  X fixed   Δ̄o^{ℓh}_i(u) = (1/K) Σ_k [ o^{ℓh}_i(clean) − o^{ℓh}_i(S_{u↔v_k}) ]

> **The attention MASK is architecture and is held FIXED.** The k-hop mask defines the model's
> receptive field / computation graph; it is **not** input structure. `structural.py`'s full
> transposition would conjugate the mask (it derives it from the conjugated pair-RRWP) — that is the
> one thing to **override**: freeze the mask, conjugate only the RRWP features. Conjugating the mask
> confounds a sparse model's **wiring** reliance with its structural **payload** reliance and
> inflates its structural score task-independently. Freezing the mask (the model-agnostic
> RRWP-feature channel) is therefore the *correct definition* of the structural intervention here,
> not an optional channel. For dense models the mask is all-ones, so this is a no-op.

## Method A — per-head carriage score (refined, primary)

Project the donor/partner-averaged transport delta onto the per-head readout gradient — the carriage
estimator `C = g·Δh` opened per head at the transport site — and aggregate its magnitude:

    F^{ℓh}[i, s] = | φ^{ℓh}_i · Δ̄o^{ℓh}_i(s) |                            # functional carriage of head (ℓ,h), source s
    S_sem(ℓ,h)   = mean_{graph, j}  Σ_i F^{ℓh}[i, j]      under the semantic donor swap
    S_str(ℓ,h)   = mean_{graph, u}  Σ_i F^{ℓh}[i, u]      under the structural transposition

Properties: (i) **alpha-weighted by construction** — `o = a·V` is the attention-routed value, so
`Δo` carries the moved attention mass, and `φ·` additionally weights by how much reaches `ŷ`;
(ii) **positive evidence for both classes** — a structural head must respond to the structural
intervention (not merely be invariant to content), and an inert head scores low on both (no
entropy-saturation); (iii) donor-average **before** `|·|` (Jensen at the kink), matching the pooled
carriage. **Beneficial twin:** swap the basis `φ → ∂ℓ/∂o^{ℓh}` for a signed loss-carriage; the
functional side is sound, but a per-head *beneficial* score needs an **ablation-based** `dL`
(zero the head's `o`, take the exact `L(ablate)−L(clean)`, or the swap×ablate 2×2), **not** the
first-order loss projection — that linearisation fails at the L1/BCE kink on well-fit models.

## Method B — semantic node transposition (original, for comparison)

Swap the **content** of a random node pair `(a,b)` (structure fixed) and read, per head, whether the
transport output `ho` at those nodes **swaps** with content (equivariant = semantic) or **stays**
(invariant = structural), each a `[0,1]` agreement, alpha-weighted by value mass `w = ‖ho_a‖+‖ho_b‖`
(`head_scores.py` value channel):

    ev = ‖ho^swap − ho‖,   d_eq = ‖ho' − ho^swap‖,   d_in = ‖ho' − ho‖,   f = 0.1·mean(ev)
    B_sem(ℓ,h) = ⟨w·clamp(1 − d_eq/(ev+f), 0,1)⟩ / ⟨w⟩          # equivariance  (semantic)
    B_str(ℓ,h) = ⟨w·clamp(1 − d_in/(ev+f), 0,1)⟩ / ⟨w⟩          # invariance    (structural)

This is the method the refined score improves on: under masking/high entropy the equivariance
geometry breaks (a masked head cannot equivariate to content it never attends to), so it reads
**architecture** (dense→semantic, 1-hop→structural) rather than task — the reach-limited-equivariance
artifact. Kept as a diagnostic for the routing-*tracking* claim only.

## Normalisation and interpretation (the scatter)

Each point is one head; x = structural score, y = semantic score.
- **Method A** divides each axis by that channel's **global mean** across all scored models
  (`x/ḡ_str`, `y/ḡ_sem`). This de-biases the intrinsic amplitude gap (a content swap moves ~unit
  content; an RRWP transposition moves ~0.1 features) **once**, so the `y=x` diagonal is
  *amplitude-normalised equal*. **Above the diagonal = semantic-specialised, below = structural.**
  The task signal is the shift relative to the diagonal (structural-task heads move toward the
  structural axis); it is normalisation-robust (survives the constant-free `str/(sem+str)`).
- **Method B** axes are raw agreements in `[0,1]`.

## Verification (asserted every run)

- `forward_capture` reproduces `Net.forward` (`|Δpred| < 1e-5`) — the capture path is faithful.
- **No-op donor** `|Δo| = 0`: a same-content swap gives an exactly zero transport delta (printed
  `noop|dh|`); a nonzero value means the wrong row was written.
- **Conjugation correctness** (offline check, `verify` workflow): a FULL relabel (transpose RRWP
  *and* content rows `u↔v`, `u,v ≠ 0`) leaves `ŷ` invariant (`|Δpred| ≈ 7e-7`); the rows-only
  control breaks it — the column swap is present and required.
- **Per-graph gradient independence**: `∂ŷ_b/∂o_{b'} = 0` for `b'≠b`, so `pred.sum()` yields the
  correct per-graph `φ`.

## How to run

```bash
# 1. train the models (writes ckpts/<task>__<variant>.pt + eval_<task>.pt)
python spec_tasks_train.py                       # SMOKE=1 for a fast pass

# 2. score + plot (works on ANY subset of trained tasks; knobs are CLI args)
python spec_head_scores.py                                   # default: semantic structural mixed
python spec_head_scores.py --tasks semantic structural --gsub 200 --donors 8 --pairs 40 --out fig.png
```

Adding a new synthetic task = add its `y = g(...)` branch to `spec_tasks_train.gen`, retrain, then
pass `--tasks <name>`; the figure grid adapts to `2 × len(tasks)`. Output: `fig_head_scores_2x3.png`.

## Known limitations

- **Amplitude asymmetry.** The score is a transport *magnitude*, so a low-amplitude structural
  payload under-registers — a *mixed* task can read semantic-leaning even when it provably uses both
  channels. Report the relative (diagonal) shift, not the raw ratio.
- **First-order readout.** `φ·Δo` linearises the final readout (the `Δ` itself is a full nonlinear
  swap). Sound for the functional score; the beneficial score must use an ablation-based `dL`.
- **Single-site.** These scores read the transport (value) site. The selection (attention) site is a
  complementary readout; a head can dissociate (structural selection + content transport). Add the
  `α`-site readout when that distinction matters.
