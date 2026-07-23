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

For intervention event `k`, define the output-projected transport vector

    q[k,i,s] = (phi_(t,i) . [o_i(clean) - o_i(intervention_(s,k))])_(t=1..T).

The production per-carrier sensitivity and raw head score are now

    F_sens[i,s] = (1/K) sum_k ||q[k,i,s]||

    S_sem(l,h) = mean_graph mean_source sum_i F_sens_sem[i,source]
    S_str(l,h) = mean_graph mean_anchor sum_i F_sens_str[i,anchor].

These are graph-balanced **eventwise-gross (EG)** scores: magnitude is taken per event before
averaging, so neither opposite donor/partner directions nor opposite carrier contributions cancel.
The graph-balanced coherent-gross matrices remain stored as `S_sem_CG/S_str_CG` diagnostics.
Production score-cache version 3 makes this choice explicit and invalidates older CG-default caches.

For comparison plots, each channel is divided by a shared reference mean, producing
`S_tilde_sem` and `S_tilde_str`, followed by

    D_rel = (S_tilde_sem - S_tilde_str) / (S_tilde_sem + S_tilde_str)
    J     = (S_tilde_sem + S_tilde_str) / 2.

`D_rel` is relative channel preference and `J` is total intervention-evoked strength. Together they
are an invertible re-expression of the two normalized raw scores; `J` is not, by itself, a measure
that both channels are jointly strong.

## Successor plan: validation before integration

This section records the validation design. The specialisation aggregation and functional-carriage
comparisons are now resolved in favour of EG and `F_sens`; unresolved components remain candidates
until separately approved.

### 1. Four specialisation aggregations — resolved

For one head and channel, let

    q[k,i] = (phi[t,i] . Delta_o[k,i])_(t=1..T)

be the output-projected transport vector for intervention event `k` and carrier `i`. Compute the
complete two-by-two aggregation family:

    S_CG = mean_(graph,source) sum_i ||mean_k q[k,i]||       # coherent-gross; legacy score
    S_EG = mean_(graph,source,k) sum_i ||q[k,i]||            # eventwise-gross
    S_CN = mean_(graph,source) ||mean_k sum_i q[k,i]||       # coherent-net
    S_EN = mean_(graph,source,k) ||sum_i q[k,i]||            # eventwise-net

**Firm decision: `S_EG` is the production functional-specialisation score.** Taking magnitude per
event avoids donor/partner-direction cancellation, while gross carrier aggregation preserves the
full internal transport signal that the methodology is designed to identify. CN/EN instead estimate
net first-order graph-output influence and can hide strong internal transport that cancels across
carriers; causal patching and ablation already test downstream use independently. CG remains the
coherent legacy diagnostic. Across the synthetic and ZINC validations all four methods were
qualitatively stable, with CG/EG and CN/EN especially close, so the choice is made by the cleaner EG
sensitivity estimand rather than a small empirical advantage on one dataset.

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

### 6. Functional and beneficial carriage — functional choice resolved

Carriage must retain individual carriers because their distance distribution is the estimand.
**Firm decision: eventwise functional sensitivity is the production functional-carriage field:**

    F_sens[i,s] = mean_k ||q[k,i,s]||

`F_sens` asks whether a typical valid intervention reaches a carrier and is aligned with production
EG specialisation. The coherent-response alternative was empirically indistinguishable in both the
synthetic and ZINC validations and is therefore retired from the beta pipeline rather than computed
as a duplicate curve. Existing `F`/`F_mean` fields mean `F_sens`. Functional-carriage version 2
invalidates old progress snapshots rather than silently mixing estimands. Optional donor-direction
coherence analyses must be separately motivated; they are not part of functional carriage. Do not
globally sum carriers before constructing the distance profile.

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
   all four specialisation scores, `F_sens`, per-graph summaries, clean condition
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
- report graph-balanced donor outcomes, support-aware `F_sens`, event-specific structural
  distance, raw/J/G/D family ablations, and sample-split conditional rules with dose-overlap,
  sign-replication, spatial-phenotype and global-FDR checks.

This cycle task cannot approve non-isomorphic structural donors, multi-source hierarchy,
production integrated beneficial carriage, or the sensitivity of general conditional-specialist
discovery. Those remain explicit transfer-stage validations rather than synthetic claims.

## Synthetic mixed-task beta findings

These are validation findings from the three trained `cycle_dual_v2` seeds, not yet production
methodology decisions. The figures currently retained in `Interesting Outputs From Synthetic`
support the following conclusions.

### Strong positive findings

- **The independent analyses agree on a semantic/structural double dissociation.** The aggregate
  task-mode × intervention-factor score matrix is strongly diagonal for every aggregation;
  D-selected family × intervention-channel patching is diagonal under restore, inject and
  necessity; and component patching finds semantic specialist families to be message-dominated
  while structural specialist families are routing-dominated. This convergence across score,
  causal and mechanism analyses is the central synthetic result.
- **The task-factor controls are stable across seeds and aggregation choices.** Relative to the
  off-diagonal task context, matched semantic-factor responses are approximately `2.8–3.8x`
  stronger and matched structural-factor responses approximately `7.7–8.5x` stronger. Net carrier
  aggregation modestly sharpens semantic specificity, but does not change the qualitative result.
- **Selected families retain the task-factor diagonal.** In
  `beta_fig1c_selected_family_specificity`, every raw-ranked and D-selected semantic, structural and
  joint role has positive matched-factor enrichment in every seed, with narrow graph-bootstrap
  intervals. D-selected enrichment is slightly weaker than raw-ranked enrichment: D preserves the
  control result but does not improve it.
- **Raw channel scores predict finite causal mediation.** Pooled across layers, median seed
  correlations between raw score and bidirectional desired-direction patch effect are `0.93`
  semantic and `0.94` structural. The corresponding within-layer association is weaker, roughly
  `0.63–0.66`, so layer organisation is a real part of the global relationship but not its whole
  explanation.
- **Evoked and balanced strength are strongly causally aligned.** Pooled `J -> causal strength` and
  `G -> balanced causal response` correlations are approximately `0.94` and `0.97`. Low-J families
  are correspondingly inert under cumulative ablation.
- **Causal family effects are bidirectional and necessary.** Semantic families show much stronger
  effects on semantic than structural events; structural families show the reverse. The interaction
  is present under clean-to-corrupt restoration, corrupt-to-clean injection and donor-wise
  zero-ablation necessity, so the result is not rescue-only.
- **Routing/message decomposition gives a clean mechanism result.** Algebraic reconstruction error
  is approximately `7.6e-6`; finite component interaction is near zero for the selected families;
  semantic rescue is almost entirely message-mediated and structural rescue almost entirely
  routing-mediated across the three seeds.
- **Functional-carriage choice is resolved.** The eventwise-sensitivity and coherent-response
  curves were effectively overlapping in both synthetic and ZINC validation. Retain only `F_sens`,
  which has the cleaner eventwise-sensitivity interpretation and aligns directly with EG.

### Negative findings and constraints

- **Ungated `D_rel` is not a valid general causal-role score.** Its pooled correlation with signed
  causal channel role is approximately `-0.07`, although the within-layer association is modestly
  positive (`0.35–0.42`). Near-inert early-layer heads can receive extreme D values because both
  channel scores are close to zero. The successful D-selected families already impose J/activity
  and uncertainty gates. The supported claim is therefore **conditional/gated D**, not standalone D.
- **The synthetic experiment alone did not justify replacing CG.** EG had the best observed top-k
  stability and small advantages on some ablation and D correlations, but no candidate supplied a
  decisive held-out causal improvement. This was the correct interim conclusion before molecular
  transfer; the later cross-validation decision adopts EG for its cleaner sensitivity estimand after
  ZINC showed that the qualitative results are insensitive to the aggregation choice.
- **Pooled correlations partly reflect layer hierarchy.** Every real-task analysis must report both
  global and within-layer validity, and should test incremental prediction after controlling for
  layer and head activity/norm.
- **The selected-family task-factor figure is not an independent confirmation.** Families are
  selected using the matched diagonal and the same matched score sample enters the enrichment
  statistic; its bootstrap conditions on the selected heads. Treat this as an off-diagonal
  specificity control. Independent held-out patching provides the causal confirmation.
- **Cross-graph mismatch patches are not a zero-effect control.** Some mismatched activation
  transplants move the output in the desired direction. Matched effects must be compared directly
  with mismatch, sham and same-layer strength-matched controls at family level.
- **Ablation reveals redundancy and task asymmetry.** Family ablation produces large functional
  logit displacement but much smaller behavioural degradation, especially on the semantic task;
  structural loss is more fragile and more seed-variable. Functional, loss and task-performance
  effects must remain separate endpoints.
- **The synthetic task does not validate topology intervention, general conditional specialists or
  molecular transfer.** It uses isomorphic cycles and the fixed-support PE/RRWP intervention. A
  topology donor here would be isomorphic or training-OOD.

## Priority ZINC transfer validation

ZINC is the next decision stage. Its purpose is to test which synthetic findings survive a larger
model, variable molecular topologies, a regression readout and less cleanly separated computation.
The task-mode × intervention-factor 2x2 from the mixed synthetic task has no literal analogue on a
single ZINC target. The transferable primary design is the **selected-family × intervention-channel
2x2**, evaluated on held-out molecules.

### Priority 1: decision-critical experiments

1. **Intervention validity and comparability.** Run the semantic real-donor intervention and the
   fixed-support PE/RRWP-payload intervention on identical held-out molecule samples. Add the
   non-isomorphic matched-topology donor as a separate beta intervention, not as a silent replacement
   for PE transposition. Match or balance size, atom/bond composition, degree statistics, target and
   intervention dose; record donor common support and chemistry validity.
2. **Raw score causal validity.** Compute CG/EG/CN/EN from identical events, then test each raw
   channel score against bidirectional whole-transport patching and independent channel ablation.
   Average donors within source, sources within molecule and molecules equally. Report both pooled
   and within-layer correlations across at least three independently trained seeds.
3. **Held-out family × channel causal 2x2.** Freeze semantic specialists, structural specialists,
   high-J/high-G generalists and low-J controls on discovery molecules. On disjoint molecules, run
   restore, inject and donor-wise necessity for both intervention channels. Compare matched effects
   with sham, cross-graph mismatch, same-layer J-matched generalists and inactive heads. This is the
   main transfer test of specialisation.
4. **D/J/G validation.** Plot ungated and gated D against signed causal role, explicitly stratified
   by layer and activity. Require D sign/role replication after a J/SNR floor and uncertainty gate.
   Test J and G against their intended causal magnitudes and test whether G adds information beyond J
   after controlling for layer and raw activity.

### Priority 2: mechanism-defining experiments

5. **Routing versus message component patching.** For the fixed-support PE intervention, reproduce
   the exact decomposition and finite routing-only/message-only/full patches. Test whether the
   synthetic semantic-message and structural-routing dissociation survives. For a support-changing
   topology donor, add an explicit wiring/support component rather than forcing the two-way split.
6. **Functional carriage.** Report production `F_sens` with graph-balanced distance profiles. Use
   shortest distance to the changed node/edge set, report support by distance and normalize
   comparisons for molecule size. The redundant coherent-response curve is no longer computed.
7. **Functional versus task-level necessity.** Report finite prediction displacement, loss/MAE
   change and any sign changes separately for individual heads and frozen families. A small MAE
   effect with a large functional effect is evidence of redundancy, not failed mediation.

### Priority 3: extension experiments

8. **Conditional specialisation.** After the unconditional pipeline is locked, predeclare a small
   number of molecular conditions such as oxygen presence, source atom class and source proximity to
   a functional group. Use discovery/confirmation splits, condition-invariant definitions, dose
   common support and channel-by-condition interactions. Do not infer a general conditional method
   from the cycle task.
9. **Integrated beneficial carriage.** Validate the finite loss-completeness identity and helpful
   versus harmful donor decomposition on the regression loss. This remains complementary to, not a
   replacement for, `F_sens`.

## ZINC transfer beta implementation

The removable standalone Colab is
`experiments/zinc/analysis/zinc_specialisation_redesign_beta_colab.py`. It analyses the existing
dense GRIT and parameter-matched 1-hop GRIT checkpoints read-only and confines every write to
`MyDrive/graph_specialisation_metrics/zinc_redesign_headline_v2`. The old ZINC richness beta and
the previous redesign output are not
modified. Expensive products are checkpoint-hash/config-fingerprint bound and restartable at graph,
event or chunk level.

The headline rerun protocol fixes `EG` and `F_sens` before looking at the new results:

- use deterministic, mutually disjoint ZINC test subsets for score discovery, whole-transport
  causal patching and ordinary/family ablation; use a further discovery/confirmation division for
  conditional rules and heads;
- force dense and 1-hop analyses to use identical graph IDs, semantic donor IDs, PE partners and
  topology donor IDs, and fail if molecular RRWP-derived topology, content or target alignment
  differs across architectures;
- compute all four aggregations as audit fields but use graph-balanced `EG` for every score, family
  and causal comparison; donors are averaged within source, sources within molecule, then molecules
  equally;
- retain semantic real-donor and fixed-mask PE transposition as distinct local channels and add
  matched-real non-isomorphic topology donors as a third channel. Headline topology comparisons use
  only strict/near donor tiers and pair the semantic or PE means to the exact same eligible graphs;
- add semantic-versus-topology score and D/J planes, activity-gated PE-versus-topology comparisons,
  pooled, layer-controlled and per-layer correlations, top-k overlap and paired graph-bootstrap
  rank stability;
- separate topology conclusions by donor tier and edit-dose tertile. Donors are first averaged
  within molecule before molecules are averaged; the same graph-balancing is used for held-out
  topology patch effects. Up to six donors are selected by interleaving the best available matches
  from each tier so the sensitivity analysis does not collapse to the easiest tier; relaxed donors
  remain excluded from headline scores. The expensive all-head causal sweep uses the best donor in
  each available tier per molecule, retaining tier coverage without pseudoreplicating near-identical
  donors;
- freeze semantic, PE-specific, topology-specific and PE/topology-shared families on score graphs,
  then patch all heads in each family simultaneously on held-out semantic, PE and topology events.
  Restore, inject and zero-ablation necessity are accompanied by sham and same-size cross-graph
  mismatch controls;
- run individual-head and frozen-family ablations on a third split, and relate all three raw EG
  scores to functional and loss impact without reopening aggregation selection;
- decompose transport under every intervention into exact common-support routing and message terms
  plus an exclusive-support wiring term. Patch each component and the full transport; abort on a
  failed reconstruction identity;
- compute final-state production `F_sens` and path-integrated signed beneficial carriage for
  semantic, PE and a separate local topology-reach intervention using event-specific changed-set
  distance. The whole-molecule matched topology donor remains the global score/causal probe and is
  excluded from reach plots because its changed-node set is not local. Functional curves use
  shared log limits; signed beneficial curves use shared symmetric-log limits. Endpoint head replay,
  quadrature convergence and loss completeness are saved diagnostics. The ZINC beta uses the
  established bounded integrated-carriage policy (`atol=5e-4`, at most 256 intervals): a capped path
  is retained only when both completeness and carrier errors are at most `5e-3`, and the complete
  run aborts if capped paths exceed 1%. This prevents one harmless L1/ReLU kink from discarding a
  long run while keeping numerical failures auditable and fail-closed.
  The local topology reach probe is a connected, degree-, edge-count- and bond-label-preserving
  two-edge switch with RRWP recomputed from the edited support;
- retain carrier distance before the production EG carrier sum. For every channel, compute
  `S_EG(l,h,b) = E_graph E_source E_event sum_{i in b} ||q_event(l,h,i)||`, and assert that summing
  distance buckets reconstructs the cached head EG score graph by graph. This is an exact
  decomposition, not a new score. Semantic bins use source distance, PE bins use distance to the
  source/partner changed set, and topology-reach bins use pristine molecular distance to the exact
  four endpoints of the local two-edge switch. Disconnected nodes
  use `unreachable`, and virtual-node transport uses a separate `hub` bucket. Report head-by-distance
  atlases, reach summaries and frozen-family distance curves aligned with final-state `F_sens` and
  signed beneficial carriage; interpret these as who implements reach, not direct performance
  prediction;
- retain that exact score-mass atlas unchanged, and add two explicit locality diagnostics. First,
  divide each graph's distance-bucket EG contribution by its matched mean event-carrier opportunity
  before graph averaging; this removes shell cardinality and estimates output-relevant response per
  available carrier without replacing the headline score. Second, bin each clean head's post-softmax
  attention mass by pristine molecular query-key distance. Report head atlases plus aggregate curves
  for exact EG mass, support-normalised EG and clean attention, with graph-bootstrap intervals and
  machine-readable per-head intervals. This comparison distinguishes global attention support from
  functionally effective transport; it does not equate query-key distance with a multi-layer causal
  path;
- expand the sample-split conditional screen to raw semantic/PE/topology EG scores, every pairwise
  D/J/G coordinate and three-channel J/G. Conditions cover clean graph composition/topology and
  source atom, neighbourhood, degree, cycle and articulation context; confirmation uses a global
  FDR and reports intervention-dose overlap and spatial concentration changes. A strict
  intervention-unit rule now excludes any source-only condition from a score containing the
  whole-graph topology donor. Such topology-containing scores can be conditioned only on graph
  context; source-conditioned topology claims require a future source-local topology intervention.
  Confirmed effects are classified as raw channel gain, joint-channel activation or relative
  selectivity shift, and as chemical, topological, routing-load, source or global graph context;
  score-confirmed effects are then tested once more on the disjoint causal split using gross
  restore magnitude, the frozen condition thresholds/head identity and a second global-FDR
  correction. Existing causal caches are enriched with molecular descriptors only, so this
  independent condition-by-mediation test does not repeat any model forward pass;
- repeat the continuous held-out differential-mediation analysis for semantic versus topology.
  This is reported separately from the primary semantic-versus-PE validation because the
  matched-real topology donor changes a whole graph rather than one source. Preserve raw semantic
  and topology causal strengths beside every ratio, and require activity gating;
- add a causal role map that keeps four questions separate: `D` supplies channel identity, `J`
  supplies evoked importance, matched restore minus mismatch/sham supplies specific rescue, and
  zero-ablation supplies necessity. The resulting necessary-and-rescuable,
  redundant/distributed-rescue, necessary-not-cleanly-rescued and weak/unresolved labels are
  explicitly exploratory relative-threshold summaries; the continuous raw effects remain primary.
  Also plot raw semantic/PE family effects and bootstrap intervals beside the normalised
  `D_causal` family controls so near-zero denominators cannot manufacture apparently decisive
  selectivity.

### Matched-real topology donor in the ZINC beta

Molecular bonds are recovered from the one-step RRWP transition channel, never from dense attention
support. Real train-set donors must have the same node count and be non-isomorphic as unlabelled
graphs. They are ranked in explicit tiers:

1. exact atom multiset, degree multiset, edge count and bond-label multiset;
2. exact atom multiset and edge count;
3. relaxed same-size nearest match, retained only for coverage diagnostics.

A Hungarian alignment prioritises atom identity, then degree and neighbour-atom context. The donor's
edge/bond tensors and already recomputed RRWP tensors are conjugated into base-node coordinates while
base `x` and `y` remain fixed. Unknown topology-derived fields, isomorphic donors or failed support
transplants stop the run. Primary topology figures and head scores use only tiers 1–2; relaxed donors
cannot silently enter the claim.

Topology remains a separate third structural axis. Its whole-graph dose and intervention unit differ
from local semantic/PE events, so it is not inserted into PE-based `D`. Adoption requires adequate
tier-1/2 coverage and held-out score-to-topology-patch validity; PE/topology head agreement is measured
but is not required, because a useful topology probe may expose wiring reliance that PE transposition
cannot. Topology events are never forced into a two-way fixed-support account: routing and message
are defined only on common directed pairs, with exclusive clean/corrupt support reported as an
explicit wiring contribution.

### Global topology score versus local topology reach

The matched-real donor changes topology throughout a molecule. Distance to the union of all edited
endpoints therefore often has support only through two or three hops; that is changed-set coverage,
not evidence that topology effects are local. It remains the preferred beta intervention for global
topology specialisation, patching and PE-versus-topology comparisons because it stays on the support
of real molecules.

Topology reach is now a separate estimand. For two disjoint bonds with identical bond labels, remove
both and reconnect the same four endpoints using a valid alternative pairing. Reject duplicates,
disconnection and no-ops. This preserves every node degree, edge count and bond-label multiset while
changing support locally; atom features and target remain fixed and all RRWP fields are recomputed.
Candidate groups prioritise compact edited sets with large observable pristine eccentricity. The run
records the planned maximum distance and fraction of events reaching beyond three hops, so limited
support cannot pass silently. This local probe supplies topology `F_sens`, beneficial-carriage and
distance-resolved EG panels only; it does not replace the matched-real topology score used for
`S_topology`, families or held-out causal validation.

The notebook emits twenty-three paper figure families (PNG and PDF), machine-readable head/causal/topology
tables and a predeclared decision report. It also records a current limitation: Drive contains one
checkpoint per architecture for this comparison. Graph bootstraps quantify evaluation-sample
uncertainty, not training-seed uncertainty, so decisions are concrete for this beta but remain subject
to revision if independently trained ZINC seeds disagree.

Score estimation is restartable at three levels: every completed source/event group is saved, then
every completed intervention channel, then the assembled graph. Local topology reach has its own
versioned graph/source cache and reuses completed global score caches. A cumulative integrated-carriage
audit is also written after each graph. Thus an interruption or later-channel error does not repeat
completed semantic/PE/topology integrations.

### Headline rerun decision priorities

1. Treat PE and topology as complementary only if strict/near donors have useful coverage, topology
   EG predicts held-out topology patching, and PE-specific/topology-specific/shared families are
   stable under graph bootstrap and show the expected three-channel patch interactions.
2. If PE and topology agree globally but stable off-diagonal families exist, retain two structural
   axes; high pooled correlation is not evidence that the interventions are interchangeable. If the
   purported specific families are unstable or fail held-out patching, retain topology only as an
   intervention-validity diagnostic.
3. Read tier/dose reversals as intervention-distribution dependence. Do not merge topology into the
   production PE score or D denominator unless conclusions survive common-tier/common-dose support.
4. Require exact support-aware reconstruction before interpreting routing/message/wiring fractions.
   Functional displacement, loss change and rescue remain separate endpoints because redundancy can
   preserve MAE after a mechanistically important family is removed.
5. Conditional effects are promoted only when the frozen feature/rule/head direction replicates on
   confirmation graphs with global-FDR control and acceptable dose overlap. Otherwise the expanded
   screen remains hypothesis generation for a later multi-seed experiment.

## Firm and remaining decisions after ZINC

- **Aggregation — firm:** use graph-balanced EG for production `S_sem/S_str`, and therefore for
  derived `D`, `J` and `G`. Save graph-balanced CG as a coherence/legacy diagnostic; CN and EN remain
  validation estimands rather than production alternatives. This decision is now implemented in the
  central scorer and methodology README.
- **Selectivity:** retain D only as a gated relative-role coordinate if its sign and causal-role
  association replicate within layers; otherwise use the two raw scores plus J/G and report D only
  descriptively.
- **Strength:** retain J as evoked strength if its causal relation transfers. Retain G as a distinct
  balanced-strength statistic only if it adds information beyond J.
- **Structural scope:** keep PE/RRWP-payload specialisation and topology specialisation as separate
  claims. Approve the topology score only if matched donors have adequate common support and its
  results survive nuisance/dose controls.
- **Topology reach:** do not interpret whole-molecule donor distance as propagation distance. Use the
  separate local degree-preserving two-edge switch for topology distance, functional carriage and
  beneficial carriage; keep matched-real donors for global topology scores and causal validation.
- **Causal confirmation:** make held-out family × channel bidirectional patching the required
  confirmation for specialist labels; ablation alone remains insufficient.
- **Relative-focus causal target — implemented in the ZINC beta:** validate `D_rel` against
  differential semantic-versus-PE whole-transport mediation, not ordinary head-ablation magnitude.
  The primary causal coordinate uses eventwise-gross restore magnitude to match EG, with donors
  averaged within source, sources within molecule and molecules equally. Report activity-gated
  pooled and within-layer rank association, graph-bootstrap uncertainty, a within-layer
  permutation test, and high-minus-low-D causal contrast. Net-aligned restore,
  intervention-effect-normalised restore, injection and necessity are robustness specifications.
  Semantic and PE causal validation use two independently planned donors/partners per source rather
  than a single swap. Because strong specialists may be absent, also patch frozen active
  high-`+D` and high-`-D` families continuously and compare them with same-layer, `J`-matched
  controls; call these D-extreme families, not confirmed specialists.
  A parallel semantic-versus-topology figure is complementary evidence only until the intervention
  units are matched. Raw channel strengths and mismatch/sham-calibrated rescue must accompany all
  ratio coordinates. Role, evoked importance, necessity and rescue are reported as distinct
  endpoints rather than collapsed into a single specialist-importance label.
- **Mechanism:** retain routing/message labels only when both exact reconstruction and finite component
  patching agree. Treat wiring as a third component for topology-changing interventions.
- **Carriage — firm and implemented:** use `F_sens` as the sole functional-carriage field aligned
  with EG. The empirically redundant coherent-response curve is not computed in the headline beta
  pipeline. Existing `F`/`F_mean` artifact keys are compatibility aliases for `F_sens`, and
  functional-carriage version 2 prevents reuse of coherent-default progress. Keep beneficial
  carriage as the complete signed loss attribution.
- **Distance-resolved specialisation — exact and implemented in beta:** decompose EG by the
  event-specific carrier distance before summing over carriers, with graphwise reconstruction as a
  fatal identity check. Use the resulting head/family reach profiles to explain which mechanisms
  implement `F_sens`/beneficial reach; do not treat them as an additional specialist score or as a
  direct performance predictor. Keep `unreachable` and virtual-node `hub` buckets explicit. Preserve
  the exact score-mass view as primary; use support-normalised EG to test whether apparent locality
  survives distance-shell opportunity, and clean attention-distance mass to test whether functional
  locality differs from raw routing locality. Neither diagnostic changes `S`, `D`, `J` or `G`.
- **Conditional labels:** approve only conditions that replicate under frozen rules, adequate support
  and multiplicity control. Whole-graph topology scores are never conditioned on source-only
  properties. Otherwise retain conditional analysis as exploratory.
