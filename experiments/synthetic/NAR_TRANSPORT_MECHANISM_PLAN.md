# NAR attention faithfulness and transport-mechanism experiment

## Purpose

This experiment extends the fixed-N Neighbor Associative Recall (NAR) study without changing
the trained models or the production specialisation metric. It asks:

> How does graph support determine when retrieval can become query-conditioned, and therefore
> whether task-relevant semantic transport is realised through attention routing or through
> message carriage/compression?

The intended dissertation contribution is a mechanistic explanation of the capacity curves, not
only another comparison of attention maps. The production per-head transport score remains the
common measure of specialisation; the routing/message decomposition explains how that transport
is produced.

## Fixed experimental decisions

- **Task:** fixed-N NAR with exactly two GRIT layers.
- **Supports:** 1-hop, 2-hop and dense.
- **Memory sizes:** `N = {4, 8, 16, 32, 64, 80}`.
- **Widths:** selectable with `--analysis-width {64,128}`. Run width 64 first; replay the identical
  pipeline at width 128 when training finishes.
- **Checkpoint selection:** each saved seed checkpoint is already the lowest-validation-loss
  training step. For primary mechanism plots, select the seed with the lowest validation loss
  within each `(width, support, N)` cell. Never select using held-out performance.
- **Performance uncertainty:** request the paper-matched seeds 0--2 and use every checkpoint present, not only the
  selected checkpoint. Missing seed cells are recorded and skipped; every support-by-N cell must
  retain at least one checkpoint.
- **Mechanism robustness:** repeat the main mechanism estimates for all seeds at
  `N = {4,16,64}`. The validation-selected trajectory over all six `N` values is descriptive;
  the all-seed anchor analysis supplies model-level uncertainty.
- **Evaluation mode:** dropout off and fixed running normalisation statistics.
- **GRIT version:** retain the pinned official GRIT commit already recorded in each checkpoint.
- **Core score:** retain the existing Method-A routed-`wV` semantic and structural scores. New
  address/payload analyses explain these scores; they do not redefine them.

The primary width-64 command will have the form:

```bash
python -m graph_specialisation_metrics.synthetic.nar_transport_mechanisms \
  --drive-root /content/drive/MyDrive/graph_specialisation_metrics/nar_grit \
  --run-name nar_grit_fixed_n_v3 \
  --analysis-width 64 \
  --phase all
```

Changing only `--analysis-width 128` will create a separate cache and reproduce every table and
figure for the larger model.

## Pre-registered mechanistic predictions

1. **Address-routing opportunity.** In a two-layer 1-hop model, the query has not reached the
   centre or records before final-layer attention is computed. The query can nevertheless change
   the competing intermediate-to-centre logit, so softmax may gate the *total* record-attention
   mass. What must remain unchanged is the attention profile normalised within the record set.
   In 2-hop and dense models, the query can reach the centre in layer 1, so layer-2 routing among
   records can become address-conditioned.
2. **Payload/message dominance.** Changing only the value stored at the queried record should act
   primarily through the message term in every support, although later-layer routing may also
   change after the payload has propagated.
3. **Local compression.** The 1-hop model must form a query-independent representation of all
   records at the centre before the query becomes available. As `N` grows, loss of useful payload
   transport should accompany its capacity failure even if late attention remains structured.
4. **Attention faithfulness is conditional.** Raw attention movement should predict functional
   transport when the intervention is address-like and the support permits query-conditioned
   routing. It should be a weaker proxy for payload transport and for 1-hop retrieval.
5. **Causal double dissociation.** Patching routing should preferentially rescue address
   interventions; patching messages should preferentially rescue payload interventions.

These predictions distinguish a useful result from the generic observation that attention and
value vectors are different objects.

## Task-specific intervention suite

All semantic interventions are within-graph, keep the topology/RRWP/support fixed, and use several
valid alternatives per graph. Signed intervention deltas are averaged over alternatives before
projection or magnitude, matching the production estimator.

| Intervention | Exact change | Label handling | Purpose |
|---|---|---|---|
| `target_payload` | Replace only the value token at the record named by the query; key and query stay fixed. Replacement value differs from the clean value. | Keep the clean label when measuring corruption/rescue. | Primary message intervention and the unchanged production `S_sem`. |
| `address_different_answer` | Replace only the query key with another key already present in memory whose stored value differs from the clean answer. Memory is unchanged. | Store both labels: the clean label for clean-target corruption/rescue and the new record value for counterfactual query-following accuracy. | Primary routing intervention. |
| `distractor_payload` | Replace the value of an unqueried record, matched to the target-payload intervention scale. | Clean label unchanged. | Negative control for message selectivity. |
| `address_same_answer` | Change the query to a different key whose stored value equals the clean answer. Memory is unchanged. | Correct label is unchanged. Use only eligible graphs and report eligibility. | Strong attention-faithfulness control: routing may change while the required output does not. |
| `identical` | Duplicate the clean graph. | Clean label unchanged. | Numerical no-op and within-batch baseline check. |
| `record_permutation` | Jointly permute complete key-value rows over the structurally symmetric record nodes, leaving the query fixed. | Clean label unchanged. | Equivariance/slot-memorisation implementation control. |

Use four alternatives per primary intervention by default. For address interventions, sample
uniformly from eligible in-memory keys without replacement where possible. Do not silently replace
ineligible same-answer graphs: retain a validity mask, report coverage by `N`, and omit invalid
graphs only from that control's estimator.

### Structural context intervention

Compute the existing mask-frozen structural specialisation score without changing its semantics:
hold content and sparse support fixed while transposing all topology-derived RRWP/node/pair payloads.
For NAR, use degree-matched query-record role transpositions as the structural signal. A
record-record transposition is a graph automorphism and serves as the structural no-op check.

NAR has no independent structural target, so `S_str` and all semantic-structural selectivity
plots are contextual diagnostics rather than evidence for a learned structural subtask.

## Core estimands

For head `h` at layer `l`, capture post-softmax routing `A` and the complete GRIT pre-attention
message field `m`, including the edge-enhance relation term:

```text
o_i = sum_j A_ij m_ij
```

For clean `c` and intervention replica `v`, use the symmetric exact decomposition with the same
clean-minus-intervention sign as the production score:

```text
Delta_route_i = sum_j (A_c - A_v)_ij (m_c + m_v)_ij / 2
Delta_msg_i   = sum_j (A_c + A_v)_ij (m_c - m_v)_ij / 2
Delta_o_i     = Delta_route_i + Delta_msg_i
```

The identity is exact at the routed node-head output `wV`; it is not a claim that the downstream
network is linear.

For every intervention family, average the signed `Delta_route` and `Delta_msg` over alternatives
before applying the clean output Jacobian `phi = d logits / d wV`:

```text
z_route[t] = sum_i phi[t,i] dot mean_v(Delta_route_i[v])
z_msg[t]   = sum_i phi[t,i] dot mean_v(Delta_msg_i[v])

S_route = ||z_route||_2
S_msg   = ||z_msg||_2
S_total = ||z_route + z_msg||_2
```

`S_total` must reproduce the existing per-head Method-A score for `target_payload`. Report:

```text
routing_share = S_route / (S_route + S_msg + eps)
mechanism_balance = (S_route - S_msg) / (S_route + S_msg + eps)
alignment = S_total / (S_route + S_msg + eps)
```

`alignment < 1` records cancellation between projected routing and message effects. Magnitudes are
not claimed to add; the signed output-space vectors do.

Two routing scopes will be saved:

- **whole-head:** all destination/source pairs contributing to `wV`, matching `S_total`;
- **retrieval edge set:** layer-2 edges from record sources into the centre. Under an address swap,
  this is the direct realised query-conditioned retrieval statistic.

Raw attention faithfulness uses the same replicas and donor averaging. Save both clean
target-to-background attention advantage/ratio and intervention-induced moved attention mass.
Store total and edge-count-normalised moved mass so within-cell head rankings are not confused
with the growth in dense edge count as `N` increases.

### Contextual semantic-structural plane

Always save raw `S_sem` and `S_str`. For the requested exploratory `D`-versus-`J` plot, remove only
within-checkpoint channel amplitude:

```text
s_sem = S_sem / mean_heads(S_sem)
s_str = S_str / mean_heads(S_str)
J = (s_sem + s_str) / 2
D = (s_sem - s_str) / 2
```

This `D/J` view describes head allocation after channel-wise normalisation; it is not presented as
an absolute comparison of semantic and structural intervention strength.

## Causal validation

### Head-family selection

Use a discovery graph set that is disjoint from all causal evaluation graphs.

- **Routing family:** top heads by address `S_route`.
- **Message family:** top heads by target-payload `S_msg`.
- **Random controls:** equal-size families matched for layer composition and approximately matched
  for clean `wV` throughput.
- **Size:** two heads per family by default. Record overlap between routing and message rankings;
  do not force disjointness silently. A disjoint sensitivity analysis may be reported separately.

### Pre-head ablation

Zero the selected heads at the routed `wV` site, retaining the existing causal intervention. Measure
clean cross-entropy increase, accuracy decrease, logit displacement, and change in each semantic
intervention effect. Compare routing, message and matched-random families on the same graphs.

Single-head ablation remains a validation of total head importance. It cannot by itself establish
whether the head acts through routing or messages.

### Routing/message hybrid patching

For one layer at a time, construct all four aligned head outputs from endpoint fields:

```text
o_cc = o(A_clean, m_clean)
o_cv = o(A_clean, m_variant)
o_vc = o(A_variant, m_clean)
o_vv = o(A_variant, m_variant)
```

Patch the chosen head family at `wV` during the variant forward and let every later operation and
layer recompute normally. Patch layers separately so a layer-1 intervention does not make a
precomputed layer-2 hybrid internally inconsistent.

Evaluate clean-target probability/loss rescue for payload and address corruption, plus new-target
query-following for the address intervention. Report:

- routing-only rescue (`o_cv` versus `o_vv`);
- message-only rescue (`o_vc` versus `o_vv`);
- full head-output rescue (`o_cc` versus `o_vv`);
- the finite downstream factorial interaction
  `I = f(o_cc) - f(o_cv) - f(o_vc) + f(o_vv)`.

The symmetric algebraic decomposition has zero residual at `wV`; this factorial term explicitly
tests nonlinear routing-message interaction after that site.

Causal experiments run on all validation-selected checkpoints at `N = {4,16,64}`. The all-seed
anchor replication is required for the headline double dissociation if compute permits; otherwise
it is reported as a stated limitation rather than replaced with graph-level pseudo-replication.

## Step-by-step implementation

### Step 1 - Add a non-destructive analysis entry point

Create:

- `src/graph_specialisation_metrics/synthetic/nar_transport_mechanisms.py` - central analysis,
  caching and plotting entry point;
- `experiments/synthetic/analysis/nar_transport_mechanisms_colab.py` - thin Drive/Colab
  launcher;
- `tests/test_nar_transport_mechanisms.py` - intervention, decomposition, selection and
  patching tests.

Do not add analysis options to the training `Config` or alter the existing training fingerprint.
The new entry point discovers checkpoints, validates their embedded metadata, and writes a
separate versioned analysis cache.

### Step 2 - Build and freeze the checkpoint manifest

Scan the existing checkpoint directory and write one row per checkpoint containing:

```text
path, experiment version, GRIT commit, width, support, N, seed,
best validation loss/accuracy, held-out loss/accuracy, parameter count, selected_for_analysis
```

Assertions and flexibility:

- at most one compatible checkpoint per requested `(width, support, N, seed)`;
- at least one compatible checkpoint per `(width, support, N)`; individual missing seeds are
  recorded in the manifest and skipped;
- supports are parameter matched within `(width, N)`;
- selection depends only on validation loss;
- every selected checkpoint reproduces its stored held-out metric within tolerance.

The manifest is cached per width and is the only source of checkpoint selection downstream.

### Step 3 - Implement and verify replicas

Add pure batch-transform functions for every intervention in the table above. Each returns the
replica plus explicit metadata: changed nodes, old/new query, old/new target, clean/new labels and
validity.

Unit tests must show that each intervention changes exactly its declared semantic field; adjacency,
RRWP and support remain bit-identical. Add the record-permutation equivariance test and verify that
address selection is determined only by graph contents and the fixed sampling seed, never by model
outputs or held-out performance.

### Step 4 - Capture fields and compute graph-level metrics

Adapt the existing GRIT field collector/decomposition rather than reimplementing GRIT attention.
Canonicalise sparse slots by `(graph, destination, source)` and abort if clean/variant support does
not align. Capture:

- `A` after softmax;
- the full node-plus-edge message `m` immediately before attention weighting;
- routed `wV`;
- clean logits and clean `d logits / d wV`.

Save graph-level, layer-level and head-level sufficient statistics, not full dense activation
fields. Required checks:

- softmax sums to one over valid sources;
- identical replicas give zero response within a preferred per-logit RMS tolerance (the raw
  output-vector L2 response is retained for auditability); small float32/CUDA exceedances are
  warned and recorded, while a five-times-larger hard guard terminates the run;
- `Delta_o = Delta_route + Delta_msg` at absolute and relative tolerances;
- decomposed target-payload `S_total` reproduces the existing `S_sem`;
- 1-hop address swaps give zero layer-2 within-record routing-profile change, while raw record
  mass gating is measured separately rather than assumed to be zero;
- record permutation leaves predictions invariant.

### Step 5 - Compute the structural context scores

Reuse the production mask-frozen structural transposition and the same clean-gradient projection.
Save raw `S_sem`, raw `S_str`, channel-normalised scores, `D` and `J`. Structural checks include
self-transposition zero, record-record automorphism zero, fixed semantic content and fixed
support.

### Step 6 - Run selected-family ablation and hybrid patching

Rank heads only from the discovery cache, freeze the family definitions, then evaluate them on an
independent causal set. Cache each endpoint and patch result by
`(width, support, N, seed, layer, family, intervention)` so figures can be regenerated without a
model forward.

### Step 7 - Generate final tables and figures

Every figure has a tidy CSV plus PNG and vector PDF. Figure code reads only cached tables; it never
loads a model.

### Step 8 - Execute width 64, then replay width 128

Run a CPU/tiny-model smoke test, one width-64 cell, the complete width-64 analysis, and finally the
figures-only phase. When width 128 training completes, change only the width flag. Produce a compact
cross-width replication table reporting whether each pre-registered directional prediction
replicates.

## Deliverables

### 1. Capacity figure

Held-out accuracy versus `N` for all supports, using all seeds. Show the mean with seed-level 95%
intervals and chance; do not overlay individual seed points. Widths 64 and 128 are separate
panels/files rather than pooled.

### 2. Specialisation-context figure

Three support columns with:

- raw structural score (x) versus raw semantic score (y);
- `D` (x) versus `J` (y).

Encode layer explicitly and distinguish `N`. These panels are labelled exploratory for NAR.

### 3. Attention-faithfulness and decomposition figure

For payload and address interventions, show both clean queried-record attention and matched
intervention-induced attention response versus output-grounded transport, followed by message
contribution versus routing contribution. Use 1-hop, 2-hop and dense columns; encode layer and `N`.
Report within-checkpoint correlations and the decomposition closure maximum in the caption/table
rather than treating heads as independent replicates.

### 4. Information-rendezvous figure

This is the headline mechanism figure:

- layerwise routing share for payload and address interventions across `N`;
- layer-2 record-channel gating and within-record address-selective routing across supports;
- the relationship between realised address routing, useful payload transport and recall accuracy.

Separate solved cells from failure cells. The decisive negative control is the predicted numerical
zero for 1-hop within-record routing-profile movement; non-zero raw centre-to-record attention
movement is allowed and identifies softmax-mediated gating of the whole record channel.

### 5. Targeted causal-validation figure

Show:

- clean-task impact of top-routing, top-message and matched-random family ablation;
- routing-only, message-only and full-output patch rescue under address and payload interventions;
- the downstream routing-message factorial interaction.

The headline test is the intervention-by-patched-component double dissociation, with uncertainty
clustered over graphs and replicated over training seeds at anchor `N` values.

### 6. Cumulative specialisation-ablation figure

Rank all heads by semantic transport on discovery graphs and ablate nested prefixes
`k = {1,2,4,8,all}`. Compare with nested random rankings matched for layer sequence and cumulative
clean throughput. Plot support-specific curves at every `N`; use all seeds at anchor `N` values.
This estimates causal concentration/redundancy rather than repeating the existing top-family
necessity result.

### 7. Support-stratified rescue figure

Repeat causal routing/message rescue for every seed at anchor `N`. Report selected-family minus
matched-random effects separately for support and `N`, with checkpoint-seed rather than graph-level
uncertainty. Include the downstream factorial interaction.

### 8. Attention-faithfulness inference figure

Compute head-rank correlations within checkpoint and layer, then bootstrap checkpoint cells.
Estimate clean-attention and intervention-response faithfulness separately for payload/address and
each support, plus the paired address-minus-payload contrast. This is the formal test of whether
attention faithfulness depends on graph support and intervention semantics.

### 9. Routing-message overlap and interaction figure

Report routing/message family Jaccard overlap, overlap-corrected joint ablation excess
`union - routing - message + overlap`, and union-family routing/message hybrid-patch interaction.
This distinguishes separate head roles, shared heads and nonlinear downstream cooperation.

### 10. Capacity-linkage figure

Replace the ceiling-saturated accuracy-only scatter with held-out cross-entropy against
address-selective within-record routing and realised address-routing contribution. Preserve the
validation-selected `N=4 -> 80` trajectories, add all-seed anchor cells, report within-support rank
associations, and cache the last-solved/first-failed transition table.

## Sampling, uncertainty and interpretation

- **Discovery:** 32 graphs per selected checkpoint, four alternatives per primary intervention.
- **Mechanism estimation:** 96 independent graphs per selected checkpoint at every `N`.
- **All-seed robustness:** 48 independent graphs per checkpoint at `N = {4,16,64}`.
- **Causal evaluation:** 256 independent graphs for every available seed checkpoint at anchor `N`
  values.
- **Follow-up evaluation:** 128 independent graphs for the selected checkpoint at every `N` and
  every available seed at anchor `N`; two intervention donors for union-family patching.
- Use deterministic, non-overlapping generator seeds recorded in the cache metadata.
- Average graphs and donors within checkpoint first. For anchor results, bootstrap checkpoint-seed
  estimates so individual graphs are never treated as independent training replicates.
- Heads are nested measurements, not independent experimental replicates. Compute head-level
  correlations within checkpoint, then summarise across checkpoints/seeds.
- Interpret mechanisms primarily in solved/matched-performance cells. Failure cells describe the
  trajectory into failure and are not pooled with solved cells to claim architectural equivalence.
- No claim about structural task specialisation is made from NAR's exploratory `S_str`/`D`/`J`.

## Compute reuse and cache design

Use the following cache tree under the existing run without touching checkpoints:

```text
nar_grit_fixed_n_v3/
  transport_mechanisms_v2/
    d64/                         # or d128
      checkpoint_manifest.csv
      config.json
      metrics/*.pt
      causal/*.pt
      followups/*.pt
      tables/*.csv
      figures/*.{png,pdf}
```

Efficiency requirements:

1. Bundle the clean graph and all semantic replicas into the same adaptive forward chunk so the
   within-batch baseline is shared.
2. Compute the clean output Jacobian once per clean batch/layer and reuse it for payload, address,
   distractor and same-answer decompositions.
3. Derive routing, message, total, raw-attention and retrieval-edge statistics from the same
   captured `A/m` endpoints; do not run one forward per metric.
4. Cache graph/head sufficient statistics immediately and release raw activations.
5. Run finite patching only for frozen selected families and only at anchor `N` values.
6. Batch different cumulative head-ablation sets as replicated graph blocks in one model forward.
7. Keep `analyze`, `causal`, `followups` and `figures` as resumable phases with per-cell atomic
   cache files.
8. Store analysis and follow-up fingerprints separately so changing fonts, bootstrap draws or figure
   layout never invalidates model forwards.

## Completion gates

The width-specific experiment is complete only when:

- the checkpoint manifest and validation-only selection audit pass;
- every intervention invariant passes;
- decomposition closure and no-op errors are below declared tolerances;
- `target_payload` decomposed totals reproduce the production semantic score;
- the exact 1-hop address-selective routing-profile negative control passes;
- family selection and causal evaluation sets are disjoint;
- all ten figure families, their CSVs and a machine-readable summary are generated;
- the summary reports every pre-registered hypothesis as supported, unsupported or inconclusive,
  without changing the hypothesis after held-out inspection.
