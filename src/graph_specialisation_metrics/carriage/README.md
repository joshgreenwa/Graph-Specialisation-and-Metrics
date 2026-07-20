# Semantic carriage

Reference for the **semantic-carriage** methodology as implemented in this package. Scope is
deliberately narrow: this measures reliance on node **content** (a semantic donor-swap
intervention). *Structural* carriage (perturbing the topology / positional encoding) is a
separate work-in-progress; it will share this vocabulary and estimator skeleton, so this file
is the reference both are kept consistent with.

The equations below are exactly what the code computes (`core.py`, `grit_runner.py`).

## Setup

A graph carries content `X = (x_1,…,x_n)` and structure `S` (edges + RRWP, computed from
topology only). A GRIT model maps them to `ŷ = ρ(h^L_1,…,h^L_n) ∈ R^T`, where `h^L_i` is node
`i`'s state after the `L` transformer layers (the tensor entering the pooling head) and `ρ` is
add/mean pooling followed by an MLP. `d(i,j)` is the shortest-path hop distance on the pristine
graph. Node `j` is the **source** (perturbed); node `i` is the **carrier** (read).

Readout gradients, both evaluated at the clean input:

    g^out_{t,i} = ∂ŷ_t / ∂h^L_i        (t = 1..T)        # functional direction
    g^loss_i    = ∂ℓ  / ∂h^L_i                           # beneficial direction

`ℓ` is the per-graph task loss reduced over the `T` targets (`l1`: mean|ŷ−y|; multilabel:
mean BCEWithLogits). Add/mean pooling makes every `g_i` identical across `i` (checked: `[5]`).

## Intervention (donor swap)

A semantic intervention replaces one source node's content, holding `S` fixed:

    X_{j→x̃} = (x_1, …, x_{j-1}, x̃, x_{j+1}, …, x_n),   S unchanged

Donors `x̃` are **real node contents sampled from other graphs** of the same split (on-manifold).
`K` donors are drawn per source and averaged, marginalising donor identity. Because RRWP is a
topology-only pre-transform (`add_node_attr=False`), `S` — including any k-hop mask — is provably
invariant to the swap (checked: `[8]`).

## Carriage estimator

Transport of `j`'s content to `i` is the change in `i`'s final state, computed with a
**within-batch clean baseline** (clean and swapped replicas in the same forward pass, so the
batch-context float offset cancels):

    Δh_i(j,k) = h^L_i(X, S) − h^L_i(X_{j→x̃_k}, S)

Projecting onto a readout gradient and donor-averaging gives the carriage under that gradient:

    C^g[i,j] = (1/K) Σ_k  g_i · Δh_i(j,k)

## Functional carriage `F` (label-free)

Magnitude of the output movement that `j` induces at `i`, over the `T` outputs:

    F[i,j] = ‖ ( C^{out_t}[i,j] )_{t=1..T} ‖_2          # = |C^out[i,j]| when T = 1

`F(bin) = mean_{d(i,j)∈bin} F[i,j]` (aggregated as below). It answers *does the model use `j` at
`i`, and how far does that use reach* — no labels involved.

## Beneficial carriage `B` (does the transport help the task?)

Let `C_loss[i,j] = C^{g_loss}[i,j]` denote the clean-gradient (first-order) loss carriage. The
**exact** per-source loss change (donor-average the *loss*, not the prediction — Jensen matters
at the L1/BCE kink) is

    dL_j = ℓ(clean) − (1/K) Σ_k ℓ(swap_{j,k})

The signed finite-loss option maps this change to carriers by integrating the task-loss
gradient along each donor's **swapped-to-clean final-state path**:

    H_{j,k}(α) = H_swap(j,k) + α(H_clean − H_swap(j,k))

    b[i,j,k] = ∫₀¹ <∂ℓ(ρ(H_{j,k}(α)),y)/∂h_i, Δh_i(j,k)> dα

    B[i,j] = (1/K) Σ_k b[i,j,k]                                      # integrated

Each donor is integrated **before** donor-averaging; integrating from the mean swapped state
would decompose a different nonlinear loss. By the fundamental theorem of calculus,

    Σ_i B[i,j] = ℓ(clean) − (1/K) Σ_k ℓ(swap_{j,k}) = dL_j.

Here `clean` is the same-forward clean endpoint used by `Δh`, which cancels batch-context
floating-point drift. `[11c]` compares the donor-averaged source loss change with the existing
clean-alone `dL_j` target and aborts if the difference exceeds `tol`, so estimator comparisons
cannot silently use different loss changes.

Because add/mean pooling is linear, the implementation integrates only through GRIT's small
pooling-to-MLP readout and projects the resulting cotangent back onto each carrier's `Δh_i`.
Adaptive local Gauss–Kronrod quadrature resolves L1/ReLU kinks. It reports carrier-refinement
error, endpoint completeness, and exact-head replay. A rare path that reaches the interval cap
retains its best estimate (it is not deleted from the donor average) and emits a warning. The run
still aborts if the global capped-path fraction exceeds `0.1%`, if a capped path's completeness or
carrier error exceeds `5e-4`, or if donor-averaged source completeness fails. These thresholds are
configurable and recorded. There is **no ratio and no clipping**: positive and negative carrier
terms may legitimately exceed `|dL_j|` while cancelling in their sum.
This is an Aumann–Shapley allocation along the declared straight final-state path: complete and
signed for that path, but path-dependent rather than a unique Shapley decomposition.

The earlier estimators remain available for controlled comparison and backwards compatibility:

    B[i,j] = clip(dL_j / Σ_{i'} C_loss[i',j], −1, +1) · C_loss[i,j]   # slope (default API)
             dL_j · |C_loss[i,j]| / Σ_{i'} |C_loss[i',j]|             # magnitude
             dL_j ·  C_loss[i,j]  / Σ_{i'}  C_loss[i',j]              # signed (legacy)

`slope` is a fast clean-tangent approximation. Its clip prevents numerical blow-up, but clip
activation means that the frozen clean gradient does not represent the finite intervention; it
must not be interpreted as successful exact attribution. `magnitude` is complete and bounded but
forces every carrier to inherit the source-level sign. `signed` is complete but unstable when its
signed denominator cancels.

    B < 0  beneficial   (content reduced the error)
    B > 0  adverse      (content increased the error)
    B ≈ 0  dispensable  (moved the output, not the error)

## Distance profiles

Pairs are pooled into adaptive shortest-path bins (`log` default: `{0},{1},{2},{3},{4–7},
{8–15},{16–31},…`) and reported over **graphs**:

    F(bin), B(bin)  = robust central tendency (20%-trimmed mean; median/mean optional) of the
                      per-graph mean, with a graph-clustered bootstrap 95% CI; bins with < 50
                      pairs are dropped.
    S(bin)          = per-graph SUM of B in bin, mean over graphs (loss units; additive).
    B_far(k)        = per-graph SUM of B over d(i,j) > k, mean over graphs (at bin upper edges).

`S` telescopes to `B_far` at bin edges. Every raw per-pair `(graph, i, j, d, C_loss, B, F)` is
saved to `carriage_pairs.npz`, so every reported binning and aggregation can be recomputed.
Recomputing a different carrier estimator requires rerunning the checkpoint because donor-path
final states are intentionally not persisted.

## Key decisions (why)

- **Path-integrated attribution (recommended signed finite-loss option).** Integrates the exact
  task-loss gradient from each swapped final state back to the within-batch clean state. It keeps
  genuine carrier-level beneficial/adverse cancellation and reconstructs `dL_j` without a share
  denominator. Adaptive quadrature diagnostics replace clipping. `slope` remains the default only
  for notebook backwards compatibility; `magnitude` and `signed` remain comparison estimators.
- **Loss-gradient beneficial basis.** For `T = 1` regression `g^loss = sign(ŷ−y)·g^out`, so
  `C_loss = sign(ŷ−y)·C^out` recovers the dissertation's error-direction projection (Eq. 3.8) and
  `B` is its exact `dL_j`-attribution; for `T > 1` (multi-target regression, multilabel) it is the
  principled generalisation, and it keeps `F` label-free while `B` uses the loss.
- **Within-batch clean baseline.** Removes the batch-context float32 offset that, on large graphs,
  is comparable to the long-range signal; also makes no-op donors give ≈ 0.
- **Binning + robust central tendency + graph-clustered CI.** Per-hop means are dominated by a few
  far pairs (heavy diameter tail) with huge CIs and, on large graphs, an unreadable axis; binning
  pools pairs, weights graphs equally, and resists outliers.

## Verification (run every time; abort on failure)

`[5]` shared `g_i` (pooling) · `[6]` batch invariance (eval mode) · `[7]` no-op donors → 0 ·
`[8]` structure invariance under swap · `[9]` loss additivity `Σ_i C_loss` vs `dL_j` ·
`[10]` unreachable pairs excluded · `[11]` `Σ_i B = dL_j`. In `integrated` mode, `[11b]` reports
quadrature residual/refinement and interval counts, `[11c]` checks clean-target alignment, and
`[11d]` verifies that replaying the pooled head at captured endpoints reproduces the full model;
`[11e]` reports the capped-path count/rate and failure-only residual distributions.
Plus a checkpoint-load metric
(`[4]`: MAE / AP) recomputed on the eval split.

## Reproduction

- GRIT pinned to `6c988ea600a606fbb49a2246c64a2d37396b3ab5`; each task's config/params/metric are in
  `tasks.py` (`zinc`, `zinc_1hop`, `zinc_1hop_local`, `peptides_func`, `peptides_struct`).
- Entry point `carriage.colab.run(task=..., …)`; the notebook cells in `experiments/carriage/`
  clone this repo (via `dissertation_key`) and call it. For signed finite-loss attribution use
  `beneficial_denom="integrated"` (the argument name is retained for compatibility); available
  values are `integrated`|`slope`|`magnitude`|`signed`. Quadrature controls are
  `integrated_atol`, `integrated_rtol`, `integrated_max_intervals`,
  `integrated_max_unconverged_fraction`, and `integrated_unconverged_error_cap`. Other key args:
  `bin_strategy` (`log`|`hop`|`equal_count`), `central`
  (`trimmed`|`median`|`mean`), `num_graphs`, `donors` (K).
- Outputs per task under `…/carriage_figures/<task>/`: three figures, `carriage_pairs.npz`
  (raw pairs + curves), `carriage_summary.json` (curves + settings + checks).
