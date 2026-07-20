# Semantic and structural carriage

Reference for the two **carriage** methodologies implemented in this package:

- **semantic carriage** measures reliance on node content using an inter-graph donor swap; and
- **structural carriage** measures reliance on topology-derived signal using a within-graph
  structural transposition.

They differ only in the intervention. Both capture the same final node state, form the same
within-batch transport delta, and use the same functional `F`, beneficial `B`, distance-binning,
aggregation, and verification machinery. The primary structural intervention is implemented in
`structural.py` and `structural_runner.py` and is part of the methodology documented here (the API
may still label that execution path `BETA`).

The equations below are exactly what the code computes (`core.py`, `grit_runner.py`, and
`structural_runner.py`).

## Setup

A graph carries content `X = (x_1,…,x_n)` and structure `S` (edges + RRWP, computed from
topology only). A GRIT model maps them to `ŷ = ρ(h^L_1,…,h^L_n) ∈ R^T`, where `h^L_i` is node
`i`'s state after the `L` transformer layers (the tensor entering the pooling head) and `ρ` is
add/mean pooling followed by an MLP. `d(i,s)` is the shortest-path hop distance on the pristine
graph. Node `s` is the **source/anchor** (perturbed); node `i` is the **carrier** (read).

Readout gradients, both evaluated at the clean input:

    g^out_{t,i} = ∂ŷ_t / ∂h^L_i        (t = 1..T)        # functional direction
    g^loss_i    = ∂ℓ  / ∂h^L_i                           # beneficial direction

`ℓ` is the per-graph task loss reduced over the `T` targets (`l1`: mean|ŷ−y|; multilabel:
mean BCEWithLogits). Add/mean pooling makes every `g_i` identical across `i` (checked: `[5]`).
For semantic carriage the source is the content-swapped node `j`; for structural carriage the
source/anchor is the node `u` whose structural role is transposed. Distances are always measured
on the pristine graph.

## Interventions

### Semantic: inter-graph content donor swap

A semantic intervention replaces one source node's content, holding `S` fixed:

    X_{j→x̃} = (x_1, …, x_{j-1}, x̃, x_{j+1}, …, x_n),   S unchanged

Donors `x̃` are **real node contents sampled from other graphs** of the same split (on-manifold).
`K` donors are drawn per source and averaged, marginalising donor identity. Because RRWP is a
topology-only pre-transform (`add_node_attr=False`), `S` — including any k-hop mask — is provably
invariant to the swap (checked: `[8]`).

### Structural: within-graph structural transposition

The primary structural intervention fixes node content `X` and transposes the topology-derived
representation between an anchor `u` and partner `v`. Let `P_(u v)` be their node transposition:

    X unchanged
    S_{u↔v} = P_(u v) S P_(u v)^T

Concretely, every present topology-derived channel is transformed consistently:

- node-indexed encodings such as `rrwp`, `deg`, `log_deg`, `abs_pe`, and `pestat_RRWP` have rows
  `u` and `v` exchanged; and
- pair/support indices such as `rrwp_index`, `edge_index`, and `rrwp_local_edge_index` are
  relabelled by `u↔v`, with aligned values (`rrwp_val`, `edge_attr`) carried with their entries.

Thus the molecular/attention support is conjugated together with the positional encoding. This
is deliberate: **structural carriage measures the use and propagation of the complete
topology-derived signal, including architectural support when the model is sparse**. It differs
from the per-head specialisation score's mask-frozen structural intervention, which holds the
attention support fixed to isolate RRWP payload reliance at a head. The two measurements answer
different questions and must not be described as using an identical structural intervention.

Both `u` and `v` move under a transposition, but carriage is indexed by the declared anchor `u`
and pristine distance `d(i,u)`. The partner is a nuisance variable: `K` partners are sampled with
replacement and averaged before the magnitude is taken, exactly as semantic donor identity is
marginalised. `partner_match="degree"` is the primary setting, drawing `v` from nodes with the
same degree when possible, then from the nearest-degree bucket when no exact match exists;
`partner_match="any"` is an unmatched sensitivity control. A full structure-plus-content
transposition is a graph isomorphism and must leave the pooled prediction invariant; this is the
completeness check that all structure-derived channels were transformed.

`structural_mode="single_node"` is also implemented: it copies partner `v`'s structural
footprint onto `u` without moving `v`. This creates a duplicated, generally unrealizable role and
is therefore an **off-manifold functional-sensitivity control**, not the primary structural
carriage intervention and not a basis for task-intrinsic interpretation of beneficial `B`.

## Carriage estimator

Transport from source/anchor `s` to carrier `i` is the change in `i`'s final state, computed with a
**within-batch clean baseline** (clean and swapped replicas in the same forward pass, so the
batch-context float offset cancels):

    semantic:  Δh_i(j,k) = h^L_i(X, S) − h^L_i(X_{j→x̃_k}, S)
    structural: Δh_i(u,k) = h^L_i(X, S) − h^L_i(X, S_{u↔v_k})

Projecting onto a readout gradient and donor-averaging gives the carriage under that gradient:

    C^g[i,s] = (1/K) Σ_k  g_i · Δh_i(s,k)

Here `k` indexes semantic donors or structural partners. The nuisance average is taken in signed
transport space before `F` applies its magnitude.

## Functional carriage `F` (label-free)

Magnitude of the output movement that source/anchor `s` induces at `i`, over the `T` outputs:

    F[i,s] = ‖ ( C^{out_t}[i,s] )_{t=1..T} ‖_2          # = |C^out[i,s]| when T = 1

`F(bin) = mean_{d(i,s)∈bin} F[i,s]` (aggregated as below). It answers *does the model use the
intervened semantic/structural signal at `i`, and how far does that use reach* — no labels
involved. It measures the reach and magnitude of response, not the information richness of the
signal that arrived.

## Beneficial carriage `B` (does the transport help the task?)

Let `C_loss[i,s] = C^{g_loss}[i,s]` denote the clean-gradient (first-order) loss carriage. The
**exact** per-source loss change (average the *loss* over donors/partners, not the prediction —
Jensen matters at the L1/BCE kink) is

    dL_s = ℓ(clean) − (1/K) Σ_k ℓ(intervention_{s,k})

The signed finite-loss option maps this change to carriers by integrating the task-loss
gradient along each donor/partner's **intervened-to-clean final-state path**:

    H_{s,k}(α) = H_intervention(s,k) + α(H_clean − H_intervention(s,k))

    b[i,s,k] = ∫₀¹ <∂ℓ(ρ(H_{s,k}(α)),y)/∂h_i, Δh_i(s,k)> dα

    B[i,s] = (1/K) Σ_k b[i,s,k]                                      # integrated

Each donor/partner path is integrated **before** nuisance-averaging; integrating from the mean
intervened state would decompose a different nonlinear loss. By the fundamental theorem of
calculus,

    Σ_i B[i,s] = ℓ(clean) − (1/K) Σ_k ℓ(intervention_{s,k}) = dL_s.

Here `clean` is the same-forward clean endpoint used by `Δh`, which cancels batch-context
floating-point drift. `[11c]` compares the nuisance-averaged source loss change with the existing
clean-alone `dL_s` target and aborts if the difference exceeds `tol`, so estimator comparisons
cannot silently use different loss changes.

Because add/mean pooling is linear, the implementation integrates only through GRIT's small
pooling-to-MLP readout and projects the resulting cotangent back onto each carrier's `Δh_i`.
Adaptive local Gauss–Kronrod quadrature resolves L1/ReLU kinks. It reports carrier-refinement
error, endpoint completeness, and exact-head replay. A rare path that reaches the interval cap
retains its best estimate (it is not deleted from the donor average) and emits a warning. The run
still aborts if the global capped-path fraction exceeds `0.1%`, if a capped path's completeness or
carrier error exceeds `5e-4`, or if donor-averaged source completeness fails. These thresholds are
configurable and recorded. There is **no ratio and no clipping**: positive and negative carrier
terms may legitimately exceed `|dL_s|` while cancelling in their sum.
This is an Aumann–Shapley allocation along the declared straight final-state path: complete and
signed for that path, but path-dependent rather than a unique Shapley decomposition.

The earlier estimators remain available for controlled comparison and backwards compatibility:

    B[i,s] = clip(dL_s / Σ_{i'} C_loss[i',s], −1, +1) · C_loss[i,s]   # slope (default API)
             dL_s · |C_loss[i,s]| / Σ_{i'} |C_loss[i',s]|             # magnitude
             dL_s ·  C_loss[i,s]  / Σ_{i'}  C_loss[i',s]              # signed (legacy)

`slope` is a fast clean-tangent approximation. Its clip prevents numerical blow-up, but clip
activation means that the frozen clean gradient does not represent the finite intervention; it
must not be interpreted as successful exact attribution. `magnitude` is complete and bounded but
forces every carrier to inherit the source-level sign. `signed` is complete but unstable when its
signed denominator cancels.

    B < 0  beneficial   (the clean semantic/structural signal reduced the error)
    B > 0  adverse      (the clean semantic/structural signal increased the error)
    B ≈ 0  dispensable  (moved the output, not the error)

## Distance profiles

Pairs are pooled into adaptive shortest-path bins (`log` default: `{0},{1},{2},{3},{4–7},
{8–15},{16–31},…`) and reported over **graphs**:

    F(bin), B(bin)  = robust central tendency (20%-trimmed mean; median/mean optional) of the
                      per-graph mean, with a graph-clustered bootstrap 95% CI; bins with < 50
                      pairs are dropped.
    S(bin)          = per-graph SUM of B in bin, mean over graphs (loss units; additive).
    B_far(k)        = per-graph SUM of B over d(i,s) > k, mean over graphs (at bin upper edges).

`S` telescopes to `B_far` at bin edges. Every raw per-pair `(graph, i, j, d, C_loss, B, F)` is
saved to `carriage_pairs.npz` (`j` stores the semantic source or structural anchor), so every
reported binning and aggregation can be recomputed. Recomputing a different carrier estimator
requires rerunning the checkpoint because intervention-path final states are intentionally not
persisted.

## Key decisions (why)

- **Path-integrated attribution (recommended signed finite-loss option).** Integrates the exact
  task-loss gradient from each intervened final state back to the within-batch clean state. It keeps
  genuine carrier-level beneficial/adverse cancellation and reconstructs `dL_s` without a share
  denominator. Adaptive quadrature diagnostics replace clipping. `slope` remains the default only
  for notebook backwards compatibility; `magnitude` and `signed` remain comparison estimators.
- **Loss-gradient beneficial basis.** For `T = 1` regression `g^loss = sign(ŷ−y)·g^out`, so
  `C_loss = sign(ŷ−y)·C^out` recovers the dissertation's error-direction projection (Eq. 3.8) and
  `B` is its exact `dL_s`-attribution; for `T > 1` (multi-target regression, multilabel) it is the
  principled generalisation, and it keeps `F` label-free while `B` uses the loss.
- **Structural transposition with a marginalised partner.** Conjugates every structure-derived
  channel while holding content fixed. Degree matching reduces arbitrary partner variation; the
  average over `K` partners prevents a single second moved node from defining the result.
- **Within-batch clean baseline.** Removes the batch-context float32 offset that, on large graphs,
  is comparable to the long-range signal; also makes no-op donors/self-partners give ≈ 0.
- **Binning + robust central tendency + graph-clustered CI.** Per-hop means are dominated by a few
  far pairs (heavy diameter tail) with huge CIs and, on large graphs, an unreadable axis; binning
  pools pairs, weights graphs equally, and resists outliers.

## Verification (run every time; abort on failure)

Shared checks are `[5]` identical `g_i` under pooling · `[6]` batch invariance in eval mode ·
`[9]` loss additivity `Σ_i C_loss` vs `dL_s` · `[10]` unreachable pairs excluded · and `[11]`
`Σ_i B = dL_s`. Intervention-specific checks are:

- **semantic:** `[7]` same-content donor gives zero transport and `[8]` every structure-derived
  tensor is invariant under the content swap;
- **structural:** `[7]` self-partner (`v=u`) gives zero transport, `[8]` content is invariant and
  every structural tensor transforms equivariantly, and `[inv]` a full structure-plus-content
  relabelling leaves the pooled prediction invariant.

In `integrated` mode, `[11b]` reports
quadrature residual/refinement and interval counts, `[11c]` checks clean-target alignment, and
`[11d]` verifies that replaying the pooled head at captured endpoints reproduces the full model;
`[11e]` reports the capped-path count/rate and failure-only residual distributions.
Plus a checkpoint-load metric
(`[4]`: MAE / AP) recomputed on the eval split.

## Reproduction

- GRIT pinned to `6c988ea600a606fbb49a2246c64a2d37396b3ab5`; each task's config/params/metric are in
  `tasks.py` (`zinc`, `zinc_1hop`, `zinc_1hop_local`, `zinc_2hop`, `zinc_1hop_vnode`,
  `zinc_2hop_vnode`, `peptides_func`, `peptides_struct`). The `zinc_2hop` / `*_vnode` variants
  replay the exact k-hop + global-VNode training patch (`khop_env` -> `GRIT_khop_ZINC.apply_khop_patch`);
  their Colab-safe recovery checkpoints (`results/_recovery_checkpoints/seed0_<name_tag>/{best,latest}.ckpt`)
  are found by `env.find_checkpoint`'s recovery fallback. A global-VNode row is excluded from
  pooling, so add-pooling over the real nodes (and every carriage precondition) still holds; the
  `h^L` capture restricts to real nodes. Every run now also recomputes the **validation** metric
  next to the test metric (both in `meta`). Cross-model comparison lives in
  `graph_specialisation_metrics.comparison` (`run_all` / `build_figures`).
- Entry point `carriage.colab.run(task=..., …)`; the notebook cells in `experiments/carriage/`
  clone this repo (via `dissertation_key`) and call it. Select the channel with
  `intervention="semantic"` (default) or `intervention="structural"`. The primary structural run
  is:

      run(task="zinc", intervention="structural", structural_mode="transposition",
          partner_match="degree", beneficial_denom="integrated")

  `donors=K` means donor rows per source for semantic carriage and partner nodes per anchor for
  structural carriage. `structural_mode="single_node"` and `partner_match="any"` are secondary
  sensitivity controls. For signed finite-loss attribution use
  `beneficial_denom="integrated"` (the argument name is retained for compatibility); available
  values are `integrated`|`slope`|`magnitude`|`signed`. Quadrature controls are
  `integrated_atol`, `integrated_rtol`, `integrated_max_intervals`,
  `integrated_max_unconverged_fraction`, and `integrated_unconverged_error_cap`. Other key args:
  `bin_strategy` (`log`|`hop`|`equal_count`), `central`
  (`trimmed`|`median`|`mean`), `num_graphs`, `donors` (K).
- Outputs per task/intervention under `…/carriage_figures/<task>/`: three figures,
  `carriage_pairs.npz` (raw pairs + curves), and `carriage_summary.json` (curves + intervention
  settings + checks). The collation index uses distinct keys for semantic and structural modes,
  so running both does not overwrite their headline rows.
