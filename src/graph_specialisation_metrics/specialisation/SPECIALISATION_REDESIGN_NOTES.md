# Specialisation methodology redesign notes

Living notes for refining the semantic and structural head-specialisation methodology. The two
proposed structural interventions are recorded first; definitions and decisions will be updated as
the design develops.

## Proposed structural interventions

1. **PE intervention.** Hold node content, edge/bond attributes, and architectural attention support
   fixed while transposing the topology-derived PE/RRWP payload between matched nodes. This isolates
   reliance on structural encoding payload, but is intentionally not a topology intervention; the
   resulting PE can be inconsistent with the fixed graph. The score should therefore be named
   **PE/RRWP-payload specialisation**.

2. **Non-isomorphic matched structural donor.** Replace a graph's topology with a different,
   non-isomorphic topology drawn from a matched real donor while holding aligned node content fixed,
   then recompute PE/RRWP from the donor topology. This is the structural analogue of the semantic
   real-donor intervention. Donors should be matched for graph size, node/edge-attribute multisets,
   degree statistics and other task-relevant nuisance variables, with an explicit node alignment.
   Controlled synthetic tasks should make such donors easy to generate; molecular tasks can use
   matched constitutional isomers where available, with validity-filtered constrained rewiring as a
   fallback when real matches are sparse.

The former measures use of the encoded structural channel; the latter measures reliance on genuine
changes of topology/isomorphism class. Full PE-plus-mask transposition is not currently proposed as a
core third intervention because it preserves the graph's isomorphism class and primarily probes
content-to-structural-role binding or architecture-conditioned wiring.

## Current implemented methodology

For each graph, layer and head, the model captures the routed head output

    o_i^(l,h) = wV[i,h,:]

at every carrier node `i`. At the clean input it computes one downstream readout gradient per model
output:

    phi_(t,i)^(l,h) = d y_hat_t / d o_i^(l,h).

Two separate global input interventions are currently run through the complete model:

- **Semantic:** replace source node `s`'s complete content row with a real row sampled from another
  graph, while holding RRWP and attention support fixed.
- **Structural:** transpose RRWP/PE payload between anchor `s` and a degree-matched within-graph
  partner, while holding node content, edge/bond attributes and the attention mask fixed.

For `K` donors or partners, transport deltas are averaged before taking a magnitude:

    Delta_bar o_i(s) = (1/K) sum_k [o_i(clean) - o_i(intervention_(s,k))].

The per-carrier functional contribution and raw head scores are

    F[i,s] = sqrt(sum_t (phi_(t,i) . Delta_bar o_i(s))^2)

    S_sem(l,h) = mean_(graph,source) sum_i F_sem[i,source]
    S_str(l,h) = mean_(graph,anchor) sum_i F_str[i,anchor].

Thus the current scores measure **output-relevant gross transport response** under the declared
intervention distributions. They use global counterfactuals and full routed transport, but average
signed donor effects before magnitude and sum carrier magnitudes without carrier cancellation.

For comparison plots, each channel is divided by a shared reference mean, producing
`S_tilde_sem` and `S_tilde_str`, followed by

    D_rel = (S_tilde_sem - S_tilde_str) / (S_tilde_sem + S_tilde_str)
    J     = (S_tilde_sem + S_tilde_str) / 2.

`D_rel` is relative channel preference and `J` is total intervention-evoked strength. Together they
are an invertible re-expression of the two normalized raw scores; `J` is not, by itself, a measure
that both channels are jointly strong.

## Successor plan: validation before integration

This section specifies candidate experiments, not approved production changes. Existing outputs
must be retained while the candidates are compared on identical interventions and graph samples.

### 1. Four specialisation aggregations

For one head and channel, let

    q[k,i] = (phi[t,i] . Delta_o[k,i])_(t=1..T)

be the output-projected transport vector for intervention event `k` and carrier `i`. Compute the
complete two-by-two aggregation family:

    S_CG = mean_(graph,source) sum_i ||mean_k q[k,i]||       # coherent-gross; current score
    S_EG = mean_(graph,source,k) sum_i ||q[k,i]||            # eventwise-gross
    S_CN = mean_(graph,source) ||mean_k sum_i q[k,i]||       # coherent-net
    S_EN = mean_(graph,source,k) ||sum_i q[k,i]||            # eventwise-net

`S_EN` is the candidate primary functional-specialisation score. Taking magnitude per event avoids
donor/partner-direction cancellation; summing signed carriers first estimates the head's net
first-order graph-output effect. Its cost is that heterogeneous/noisy events contribute positively
and strong internal transport that cancels at the readout becomes small. Therefore retain `S_CG`
as legacy coherent gross transport and `S_EG` as the principal internal-transport companion.

Report donor and carrier coherence diagnostics, conceptually

    kappa_donor   = S_CN / S_EN
    kappa_carrier = S_EN / S_EG,

with event-level definitions and noise floors chosen so the ratios remain in `[0,1]` and are not
reported near zero. All four estimands can be accumulated from the existing intervention forwards.

Aggregation must be hierarchical and graph-balanced:

1. average donors/partners within source;
2. average sources within graph;
3. average graphs equally.

The existing source-pooled estimator should remain as a sensitivity analysis because it targets a
uniformly sampled source rather than a uniformly sampled graph.

### 2. Conditional specialisation

Conditional specialisation is a first-class target because trained models may contain few global
specialists but meaningful context-gated heads. For example, a head may become selective and
spatially concentrated when oxygen is present while becoming diffuse otherwise.

For a clean-input condition `C=r`, compute condition-specific versions of all four scores using the
same fixed reference normalisation across conditions. The primary inferential quantity is a
**channel-by-condition interaction**, not separate significance within each stratum. One useful
form is

    I_cond = [log(S_sem|C=1) - log(S_sem|C=0)]
           - [log(S_str|C=1) - log(S_str|C=0)].

Also report condition-specific selectivity, evoked strength and balanced dual-channel response.
Require adequate activity in both relevant strata so ratios of near-zero scores cannot define a
specialist.

Keep distinct conditional questions:

- **graph context:** e.g. oxygen is present in the clean molecule;
- **source identity/context:** e.g. the intervened source is oxygen, non-oxygen, or near oxygen;
- **donor transition:** e.g. O->non-O, non-O->O, or a swap not changing oxygen availability;
- **spatial phenotype:** concentrated versus diffuse carrier transport.

The condition must be defined from the clean input. In particular, a donor swap that introduces or
removes the only oxygen must not silently change the conditioning variable. Prefer conditions that
remain invariant under the intervention, or stratify the source-to-donor transition explicitly.

Spatial localisation is related mechanism evidence, not itself semantic/structural specialisation.
Record normalised carrier entropy/effective carrier count, near-source mass, expected radius and
distance profile, then test their condition interaction separately.

Predeclare a small set of scientifically motivated conditions. Conditions discovered from score or
activation data require a discovery split; freeze their exact rule, heads, direction, aggregation
and activity floor before confirmation. Balance or adjust clean pre-treatment nuisances such as
graph size, composition, degree profile and intervention dose across conditions. Oxygen-present
effects remain associative effect modification unless the condition itself is randomized.

### 3. Strength, selectivity and head labels

Recompute the derived coordinates for every candidate raw-score family:

    D_rel = (S_tilde_sem - S_tilde_str) / (S_tilde_sem + S_tilde_str)
    J     = (S_tilde_sem + S_tilde_str) / 2
    G     = sqrt(S_tilde_sem * S_tilde_str)
          = J * sqrt(1 - D_rel^2).

Rename `J` **evoked strength**. Name `G` **balanced joint-channel strength**: it shows that both
separate-intervention responses are strong, but does not demonstrate simultaneous co-use, synergy
or interaction.

Select specialists using extreme `D_rel` plus a raw-strength/SNR floor and uncertainty, not raw
`argmax(S_sem)` or `argmax(S_str)`. Those raw maxima are the strongest channel responders. A
per-channel share/log-ratio is mostly a re-centred monotonic expression of `D_rel`, not a new source
of discrimination. Empirical log-log residuals may rank off-line outliers secondarily, but must not
replace the raw scores.

### 4. Causal confirmation by bidirectional transport patching

On held-out graphs, patch the complete carrier-aligned `wV` tensor for a selected head at one layer:

- **restore:** clean head transport into the intervened run;
- **inject:** intervened head transport into the clean run.

Report both directions separately, their absolute output effects, and signed alignment with the
actual clean-intervened output delta. A large orthogonal movement is not successful rescue. Use
normalized recovery fractions only when the underlying intervention effect clears a noise floor.

Primary validation is a held-out head-by-channel interaction. Conditional specialists require a
head-by-channel-by-condition interaction: rescue/induction should be selectively stronger in the
predicted channel and context. Include same-layer, evoked-strength-matched generalists, inactive
heads, sham patches and wrong-donor controls. This supports the narrow claim that the head site
mediates the response; it does not prove that the computation originated in that head.

The existing swap-by-zero-ablation experiment remains a secondary necessity test. Fix it before
use: evaluate nonlinear loss donor-by-donor before averaging, use eventwise functional effects, and
separate its graph sample from score discovery.

### 5. Routing, message and wiring mechanism

On aligned fixed support, decompose the routed output using the exact symmetric bilinear split

    Delta_o = Delta_A * mean(message) + mean(A) * Delta_message.

`message` must include the complete GRIT node-value and relation/edge-enhanced packet. Preserve
signed components until carrier summation, assert numerical reconstruction of the total delta, and
then apply the same output projection and four aggregations. This distinguishes, for example,
structural routing from semantic message transport.

The algebraic split is descriptive mechanism evidence. For confirmed specialists, form observed
hybrids that patch routing only, message only, and both to test their causal contributions and
interaction.

For non-isomorphic or support-changing interventions, routing versus message is not uniquely
defined without an alignment convention. Treat support/wiring as a third component, or restrict
the two-way decomposition to the fixed-support PE intervention. Do not present an arbitrary
reindexing convention as an identified mechanism.

### 6. Functional and beneficial carriage

Carriage must retain individual carriers because their distance distribution is the estimand. Add
eventwise functional sensitivity

    F_sens[i,s] = mean_k ||q[k,i,s]||

beside current coherent functional carriage

    F_coh[i,s] = ||mean_k q[k,i,s]||.

`F_sens` asks whether a typical valid intervention reaches a carrier; `F_coh` asks whether the
population-average intervention moves it consistently. Add donor coherence, and optionally compare
gross versus signed-net contributions within each distance band. Do not globally sum carriers before
constructing the distance profile.

No replacement for integrated beneficial carriage is required. Its donor-wise signed path
attribution and subsequent averaging are precisely what preserve loss completeness:

    B[i,s] = mean_k b[i,s,k]
    sum_i B[i,s] = mean_k [loss(clean) - loss(intervention_k)].

Useful companions, not replacements, are `mean_k |b|`, sign coherence, and the distribution of the
donor-level total loss change. Report both the probability and conditional magnitude of helpful and
harmful donor outcomes; probabilities alone can hide rare large harms. Outcome-conditioned carrier
terms should sum back to helpful and harmful expected loss-change mass separately. Make integrated
`B` the headline/default; never replace it with an absolute or RMS quantity.

For pair transpositions, compute structural distance per partner event before partner averaging:

    d_changed(i;u,v) = min(d(i,u), d(i,v)).

Build distance profiles from these event-level distances, retain the number and effective number of
events supporting each distance band, and obtain confidence bands from the same nested
graph/source/event bootstrap. Do not assign an averaged intervention a single averaged distance.

A global non-isomorphic structural donor has no honest local-source distance unless the edits are
localized; define distance to the changed node/edge set where possible, otherwise omit a local
distance claim.

Path-integrated functional output carriage through the cached readout is a stronger later
confirmation because carrier terms exactly reconstruct the finite output delta. It is path-dependent
and costs additional backward work, so it does not block the cheap `F_sens` experiment.

### 7. Experimental validation sequence

1. **Cache-compatible discovery.** Version the score cache; retain legacy matrices and accumulate
   all four specialisation scores, `F_sens/F_coh`, coherence, per-graph summaries, clean condition
   metadata, intervention type/dose and localisation statistics without new intervention forwards.
2. **Reliability.** Use nested graph/source/event bootstrap, explicit deterministic no-op/sham noise
   floors, repeated donor seeds and `K`-convergence checks. Report score/selectivity CIs, ranking and
   top-k stability, source-cap sensitivity, and per-checkpoint results.
3. **Intervention comparability.** `D_rel` compares responses to two declared counterfactual
   distributions; it is not intervention-distribution-free. Record channel-appropriate pre-treatment
   dose measures, inspect score-versus-dose curves and repeat comparisons on common dose-support or
   matched dose quantiles. Do not divide by the realized model-output change, which is a
   post-intervention response and can erase genuine sensitivity. Treat selectivity that reverses
   under reasonable donor distributions as distribution-specific rather than intrinsic to the head.
4. **Score comparison.** Predeclare the winner criterion: prediction of held-out donor-wise causal
   patch/channel-ablation effects plus rank stability, not correlation with ordinary ablation alone.
   Reserve at least one task or training seed from method selection. On a held-out subset, compare
   the clean-gradient first-order prediction directly with the finite bidirectional patch effect;
   report alignment, calibration and failure rate, and audit path-integrated projections if the local
   approximation is poor.
5. **Conditional confirmation.** Freeze conditions and candidate heads on discovery data; test the
   channel-by-condition interaction and spatial phenotype on held-out graphs with adequate common
   support and multiplicity control.
6. **Mechanism and causality.** Run routing/message decomposition, bidirectional head patching and
   matched controls only for preregistered specialist, conditional-specialist, generalist and inactive
   families. Replicate architecture-level claims across independently trained seeds.

Eventwise magnitudes have a positive noise floor and may reward merely larger interventions. Record
same-content/self-transposition no-ops, the probability an event is nontrivial, and conditional
sensitivity given a nontrivial event. Do not force semantic/structural families by rank when no head
clears the predeclared selectivity, activity and uncertainty thresholds: the empirical absence of
global specialists is itself a valid result.

## Synthetic mixed-task beta implementation

The removable standalone Colab beta is
`experiments/synthetic/analysis/paper_synthetic_mixed_task_redesign_beta_colab.py`. It loads the
three existing `cycle_dual_v2` checkpoints read-only and writes only to
`cycle_dual_v2/redesign_beta_v1` on Drive. Its validation contract is:

- compute `CG/EG/CN/EN` from exhaustive event populations, with repeated donor-order/K checks;
- retain the matched semantic-task/semantic-factor and structural-task/structural-factor cells as
  candidate scores, while caching the two off-diagonal task-mode × intervention-factor cells as
  specificity controls;
- select an aggregation using seeds 0–1, then require seed-2 paired graph-bootstrap evidence for
  restore, inject, causal `D`, and channel ablation before replacing `CG`; failed candidates remain
  diagnostic and all headline figures revert to `CG`;
- define causal `D` from signed channel effects, causal `J` from absolute evoked magnitude and
  causal `G` from nonnegative balanced magnitudes; report anti-aligned heads rather than clipping;
- validate D-selected families with bidirectional whole-transport patching, zero-ablation,
  cross-graph mismatch and sham controls, plus family-by-channel interactions;
- decompose fixed-support transport into exact routing/message terms and causally patch routing,
  message, full transport and their finite interaction;
- report graph-balanced donor outcomes, support-aware `F_sens/F_coh`, event-specific structural
  distance, raw/J/G/D family ablations, and sample-split conditional rules with dose-overlap,
  sign-replication, spatial-phenotype and global-FDR checks.

This cycle task cannot approve non-isomorphic structural donors, multi-source hierarchy,
production integrated beneficial carriage, or the sensitivity of general conditional-specialist
discovery. Those remain explicit transfer-stage validations rather than synthetic claims.
