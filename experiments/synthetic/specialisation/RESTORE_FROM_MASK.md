# Restore-from-mask: is dense content retrieval irreducible?

**Question.** Which heads carry a global-attention function that a 1-hop model *cannot* rebuild?

## Setup
- One trained **dense** GT (weights `θ`). No retraining.
- At inference, **mask its attention to 1-hop** → crippled model `M₁`; unmasked = `M₀` (dense).
- Rank all heads by the **raw** Method-A scores `S_sem(l,h)`, `S_str(l,h)` (separate-intervention, `‖φ·Δo‖`).

## Procedure (what is intervened)
Run the **1-hop-masked** forward pass, but for a chosen head **set** `S` put back those heads'
**original dense (unmasked, global) routed output** `oˡʰ = attn·V`; every other head stays local.
Same weights — only the restored heads regain global reach.

- **Restore set = top-q heads by score**, sweeping `q = 1,2,…` (a dose-response), for:
  - the **semantic** ranking (`S_sem`), and
  - the **structural** ranking (`S_str`).

## Measured
- **Gap recovery** `R(S) = [L(M₁) − L(M₁ ⊕ restore S)] / [L(M₁) − L(M₀)]` — fraction of the
  dense→1-hop performance gap closed by restoring set `S`. Plot `R` vs `q` (recovery curve).

## Controls / substitutability
- **Random-set** restore (same `q`) — does score-ranked beat random? (validates the score in the *restore* direction, mirroring the ablation result).
- **No-op null**: restore into `M₀` must do nothing.
- **Leakage null**: restoring the structural component must not move semantic-attributed benefit.
- **Substitution**: can 1-hop rebuild the recovered signal with **+depth / +hops / a VNode** instead of restoration? → gives the *irreducible* residual per channel.

## Prediction (raw scores)
- Semantic ranking: `R(q)` **rises fast**, closes most of the gap, beats random — and the residual **survives depth + VNode** and **grows with #distractors/N** ⇒ **irreducible**.
- Structural ranking: `R(q)` **barely moves**, and whatever it recovers is **closed by 1–2 hops or a VNode** ⇒ **locally substitutable**.
- Single-head restore is only a lower bound; the **head-set curve + the semantic-vs-structural and score-vs-random contrasts** are the result.
