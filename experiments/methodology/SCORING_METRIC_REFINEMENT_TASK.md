# Final scoring-metric refinement experiment

## Objective

Build one small, reusable experiment that compares the main candidate definitions of
semantic/structural head specialisation on trained dense GRIT models. Run it on:

- dense GRIT+RRWP trained on ZINC regression; and
- dense GRIT+RRWP trained on the QM9 HOMO-LUMO gap.

The experiment must answer two separate questions:

1. **Head significance:** which score best identifies heads whose transport is causally important?
2. **Head role:** which score best distinguishes semantic from PE/structural mediation?

The implementation must live in a task-general methodology package, not in a ZINC- or QM9-specific
analysis file. A standalone Colab frontend must clone the repository, load either dense checkpoint,
resume Drive caches, and render every comparison.

## Terminology and scope

Use three channel names consistently:

| Channel | Meaning | Canonical intervention |
|---|---|---|
| **semantic** | node content, such as atom identity | change `x` while holding topology, PE/RRWP, bonds and attention support fixed |
| **PE** | topology-derived positional payload supplied to the model | change RRWP/PE while holding content, bonds and architectural attention support fixed |
| **topology** | the molecular graph/isomorphism class itself | use a matched non-isomorphic topology donor, align nodes, and recompute RRWP |

In particular, a node transposition of RRWP/PE is a **PE intervention**, not a topology
intervention. Genuine topology remains a separate third channel and is not silently pooled with
PE.

The semantic and PE channels each receive two intervention variants. The topology channel remains
fixed throughout this refinement experiment.

## Intervention catalogue

### Semantic intervention variants

#### `semantic_single_donor`

Replace the complete content row of one source node `u` with a real donor node's content:

```text
x[u] <- donor_x
```

Hold every topology-derived field, bond field, PE/RRWP field, target and attention-support index
fixed. Match donor nodes on clean pre-intervention structural context where possible, while
requiring a different content row so that the event is not a no-op.

This is the current semantic intervention.

#### `semantic_transposition`

Choose two distinct, structurally matched nodes `(u, v)` in the same graph and exchange their
complete semantic rows:

```text
x[u], x[v] <- x[v], x[u]
```

Hold PE/RRWP, bonds, target and attention support fixed. Prefer pairs with different content and
similar clean degree/local structural context. The operation preserves the graph's content
multiset and is an involution.

### PE intervention variants

#### `pe_single_donor`

Change one source node's PE/structural footprint using a matched within-graph donor node while
holding the donor node fixed. Operationally, this is the existing `single_node` structural mode:
copy the donor node's node-indexed PE fields and its pairwise structural incidences onto the source.
Freeze the trained attention mask/support exactly as in the current production PE score.

This one-sided operation duplicates a structural role and is therefore only **near-manifold**, not
an exact PE encoding of a real graph. It must be labelled as such. Record reconstruction
inconsistency, intervention dose and no-op rate; do not describe it as fully on-manifold.

The implementation may construct it as “transpose `(u, v)`, then restore `v`”, but the resulting
tensors must be identical to the explicit one-sided-copy definition and covered by tests.

#### `pe_transposition`

Choose a structurally matched pair `(u, v)` and conjugate every topology-derived PE/RRWP payload by
the transposition `P_(u v)`, while keeping content, bonds and architectural support fixed:

```text
node PE:  r[u], r[v] <- r[v], r[u]
pair PE:  R <- P_(u v) R P_(u v)^T
```

This is the current mask-frozen PE intervention. Both nodes are in the changed set.

### Fixed topology intervention

Retain the existing matched-real, non-isomorphic topology donor:

1. match a real donor graph on node count and molecular nuisance variables;
2. align donor nodes to base nodes;
3. copy donor topology/bond structure in aligned coordinates;
4. recompute all RRWP fields from the donor topology; and
5. keep base semantic content and target fixed.

Use the same topology intervention plan for every scoring method. It is a common comparator and
causal control, not another arm of the semantic/PE `2 × 2` factorial.

## Shared notation

For layer `l`, head `h`, query/carrier node `i` and key/sender node `j`:

```text
A[l,h,i,j]       = post-softmax attention probability
m[l,h,i,j,:]     = complete pre-attention message, including GRIT edge enhancement
o[l,h,i,:]       = sum_j A[l,h,i,j] m[l,h,i,j,:] = routed head output `wV`
phi[t,l,h,i,:]   = d y_hat[t] / d o[l,h,i,:] at the clean input
```

For an intervention event `k`:

```text
Delta o[k,l,h,i,:] = o_clean[l,h,i,:] - o_k[l,h,i,:]
q[k,l,h,i,:]       = (phi[t,l,h,i,:] dot Delta o[k,l,h,i,:]) over outputs t
```

All event-based estimators must be hierarchical and graph-balanced:

1. average valid events within source;
2. average sources within graph; and
3. average graphs equally.

Use the same deterministic graph, source, pair and donor manifests wherever two methods are
compared.

## Probability-mass-weighted follow/invariance protocol

The transposition methods compare the perturbed field with two clean references in fixed-query
coordinates. If `pi=(u,v)`:

```text
invariant reference: F_clean[i,:]
following reference: F_clean[i, pi(:)]
```

“Following” means that the key/sender field follows the transposed semantic or PE identity.
“Invariant” means that it remains in the original key coordinates.

### Attention agreement

For raw attention, use probability-distribution overlap:

```text
AttnInvariant = 1 - 0.5 * ||A_event[i,:] - A_clean[i,:]||_1
AttnFollow    = 1 - 0.5 * ||A_event[i,:] - A_clean[i,pi(:)]||_1
```

Evaluate only aligned valid support. Clamp numerical noise to `[0,1]`. For dense attention the
support is naturally aligned. Any later sparse-model extension must use the union/intersection
support policy explicitly and must not equate a support change with attention-mass movement.

### Transport/message agreement

For transport, compare the complete per-key message field `m`, because it preserves the sender
axis needed by the following reference. Weight the comparison by the attention probability mass
that actually routes those messages.

For reference `R` (clean-invariant or clean-following), use:

```text
w_invariant[i,j] = 0.5 * (A_event[i,j] + A_clean[i,j])
w_follow[i,j]    = 0.5 * (A_event[i,j] + A_clean[i,pi(j)])

WeightedCos(U,V;w) =
    sum_j w[j] <U[j],V[j]>
    / sqrt(sum_j w[j] ||U[j]||^2 * sum_j w[j] ||V[j]||^2)
```

The primary agreement is the uncentered weighted cosine clipped to `[0,1]`:

```text
TransportInvariant = WeightedCos(m_event, m_clean, w_invariant)
TransportFollow    = WeightedCos(m_event, m_clean[:,pi(:)], w_follow)
```

Also store the raw cosine and a centered-key companion, but do not create extra headline methods
from them. Record attention mass, valid key count and effective support for every query so a high
agreement on negligible mass is detectable.

This probability-mass weighting is distinct from the older softmax weighting over sampled
permutations. If the existing general metric engine retains that older weighting, expose it as an
optional diagnostic named `permutation_moved_mass_weighting`; do not call it attention-probability
weighting.

Aggregate query scores within graph, then graphs equally. Do not select transpositions by inspecting
their perturbed scores.

## Six headline scoring methods

Every method produces two raw per-head axes. Plot the raw axes before any `D_rel/J` transformation.

### M1 — separate-intervention output-projected transport

This is the current production EG methodology, evaluated for every semantic/PE intervention pair:

```text
S_sem = mean_graph mean_source mean_event sum_i ||q_sem[event,i]||
S_PE  = mean_graph mean_source mean_event sum_i ||q_PE[event,i]||
```

Run all four combinations:

| M1 arm | Semantic intervention | PE intervention |
|---|---|---|
| `M1_DD` | `semantic_single_donor` | `pe_single_donor` |
| `M1_DT` | `semantic_single_donor` | `pe_transposition` |
| `M1_TD` | `semantic_transposition` | `pe_single_donor` |
| `M1_TT` | `semantic_transposition` | `pe_transposition` |

`D` means single-node donor/copy and `T` means two-node transposition. Cache the semantic and PE raw
scores once per intervention variant; the four arms are combinations of those shared results, not
four redundant model runs.

For each arm, use fixed discovery-split channel references:

```text
S_tilde_sem = S_sem / reference_sem
S_tilde_PE  = S_PE  / reference_PE

D_rel = (S_tilde_sem - S_tilde_PE) / (S_tilde_sem + S_tilde_PE + eps)
J     = (S_tilde_sem + S_tilde_PE) / 2
```

### M2 — semantic-transposition transport following/invariance

Apply `semantic_transposition` and score the probability-mass-weighted message field:

```text
semantic axis = TransportFollow(content transposition)
PE axis       = TransportInvariant(content transposition)
```

Interpret the invariant axis cautiously: invariance to semantic transposition is not positive proof
of PE use. This method is a routing/representation-geometry diagnostic.

### M3 — PE-transposition transport following/invariance

Apply `pe_transposition` and score the probability-mass-weighted message field:

```text
semantic axis = TransportInvariant(PE transposition)
PE axis       = TransportFollow(PE transposition)
```

Again, invariance is not by itself positive evidence of semantic use.

### M4 — semantic-transposition attention following/invariance

Apply `semantic_transposition` and score raw attention distributions:

```text
semantic axis = AttnFollow(content transposition)
PE axis       = AttnInvariant(content transposition)
```

### M5 — PE-transposition attention following/invariance

Apply `pe_transposition` and score raw attention distributions:

```text
semantic axis = AttnInvariant(PE transposition)
PE axis       = AttnFollow(PE transposition)
```

### M6 — separate positive following scores

Use positive following evidence from two separate transposition interventions:

```text
S_sem = TransportFollow(content transposition)  # from M2
S_PE  = TransportFollow(PE transposition)       # from M3
```

Combine these two raw scores using fixed references and the same `D_rel/J` equations as M1. M6 is
the follow-score analogue of the current separate-intervention method.

### Derived coordinates for fair validation

M1 and M6 have scientific `D_rel/J` coordinates by construction. For M2–M5, also compute the same
two-axis transform so every method can enter the common validation code, but label those
coordinates **diagnostic** because one axis is invariance rather than positive channel evidence.
Always retain and plot the two raw scores.

## Topology reporting

Compute the fixed topology EG score once per head using the current output-projected transport
estimator. It is not folded into semantic/PE `D_rel`.

For every task report:

- raw topology score by layer/head;
- topology score versus each method's semantic axis;
- topology score versus each method's PE axis;
- correlation of topology score with topology restore/inject patching; and
- whether a semantic/PE method merely tracks general intervention sensitivity by controlling for
  topology score and clean head throughput.

## Required comparisons

### 1. Raw-score scatter plots

Produce separate, readable, layer-coloured panels for every raw score pair:

1. a `2 × 2` M1 grid for `DD`, `DT`, `TD`, and `TT`;
2. one panel each for M2, M3, M4, and M5;
3. one M6 semantic-follow versus PE-follow panel; and
4. topology companion panels.

Each panel must:

- show one point per `(layer, head)`;
- label both axes with the exact field, intervention and estimator;
- show raw score units/ranges;
- include a diagonal only when equality has a meaningful interpretation;
- use a shared layer colour scale;
- report pooled and within-layer Spearman correlations; and
- avoid overlays that hide individual methods.

Create normalized `D_rel/J` figures separately from the raw-score atlas.

### 2. Single-node donor versus node transposition

Compare the M1 raw intervention scores directly:

```text
semantic_single_donor  vs semantic_transposition
pe_single_donor        vs pe_transposition
```

Report:

- Pearson and Spearman correlation;
- within-layer Spearman correlation;
- layer-centred correlation;
- top-`k` overlap for `k in {3,5,10}`;
- rank stability under graph bootstrap;
- raw and log-scale difference plots;
- intervention dose, no-op rate and event variance; and
- the four M1 arms' `D_rel/J` rank agreement.

The semantic and PE score for a given intervention variant must be computed only once. Do not let
the other channel's selected variant alter its graph/source/event manifest.

### 3. Fast ablation and causal patching

Use disjoint deterministic splits:

- **score split:** estimate all methods and freeze references/rankings;
- **causal split:** restore/inject transport under held-out semantic, PE and topology events;
- **ablation split:** measure clean head necessity independently.

#### Individual-head significance

For every head, zero its full routed output `wV` at one layer and measure:

- mean absolute prediction movement;
- task-loss increase;
- graph-bootstrap confidence interval; and
- rank against the all-head single-ablation null.

Evaluate each method's strength coordinate against held-out significance using:

- Spearman correlation between `J` and ablation magnitude;
- partial correlation controlling for layer and clean transport throughput;
- top-`k` ablation enrichment; and
- stability across ZINC and QM9.

#### Separate semantic/PE role

For each held-out semantic and PE event, patch the full carrier-aligned head transport:

- **restore:** clean head transport into the intervened run;
- **inject:** intervened head transport into the clean run;
- **necessity:** compare clean/intervened effects after zeroing the head; and
- **sham:** patch the head with its own condition.

Construct graph-balanced semantic and PE mediation strengths and their differential coordinate.
Evaluate:

- Spearman correlation between score `D_rel` and differential semantic-versus-PE mediation;
- correlation between `J` and total mediation;
- within-layer permutation `p` values;
- graph-bootstrap intervals;
- top/bottom score-family patch interactions;
- comparison with same-layer, throughput- and `J`-matched controls; and
- sham and wrong-graph patch controls.

M2–M5 must not win the role comparison merely because invariant scores are high. The primary role
criterion is held-out **differential mediation**, not separation in the raw scatter.

#### Method selection

Select winners separately:

- **best significance method:** strongest stable held-out relation between `J` and ablation;
- **best role method:** strongest stable held-out relation between `D_rel` and differential
  semantic/PE mediation.

Report uncertainty and task heterogeneity. Do not force a single winner if ZINC and QM9 disagree.
An optional omnibus rank may average predeclared standardized significance and role statistics, but
the two scientific conclusions must remain separate.

## Default run sizes

Keep the experiment substantially cheaper than the full redesign analysis:

| Split/setting | Default | Fast development |
|---|---:|---:|
| score graphs | 48 | 6 |
| sources per graph/channel | 6 | 2 |
| donor/pair events per source | 6 | 2 |
| topology donor events | 3 | 1 |
| causal graphs | 24 | 4 |
| causal sources per channel | 1 | 1 |
| causal events per source | 2 | 1 |
| ablation graphs | 64 | 8 |
| graph bootstrap samples | 1000 | 40 |

Batch intervention replicas and head patches. Cache clean fields, clean gradients, intervention
fields and event manifests so M1–M6 reuse forwards whenever their required tensors coincide.

## Reusable implementation

Create a general package:

```text
src/graph_specialisation_metrics/scoring_refinement/
    README.md
    config.py
    interventions.py
    fields.py
    scores.py
    validation.py
    figures.py
    cache.py
    runner.py
```

Suggested public API:

```python
from graph_specialisation_metrics.scoring_refinement import run

run(
    tasks=["zinc", "qm9_gap_dense"],
    output_dir="...",
    methods="all",
    phase="all",
    force=False,
)
```

Requirements:

- use `GritTaskSpec`, existing checkpoint loading and task content adapters;
- reuse the official GRIT attention/message collector;
- reuse production EG, `D_rel/J`, hierarchical aggregation and patching utilities where possible;
- keep task-specific paths/configuration outside metric definitions;
- fingerprint caches by protocol version, task, checkpoint SHA, config and event manifest;
- permit `phase={scores,validation,figures,all}`;
- support cache-only figure rerendering; and
- never write inside checkpoint/training result directories.

The package `README.md` must define M1–M6, every intervention, the probability-mass weighting,
normalisation, causal estimands, caveats and output schema.

## Standalone Colab frontend

Create:

```text
experiments/methodology/colab_scoring_metric_refinement_dense.py
```

It must:

1. mount Google Drive;
2. clone/refresh the requested repository branch using the existing optional Colab secret;
3. install the repository and GRIT/PyG dependencies;
4. ignore only injected Jupyter `-f kernel-....json` arguments;
5. run dense ZINC and dense QM9 sequentially, releasing GPU memory between tasks;
6. auto-discover checkpoints, with explicit checkpoint overrides;
7. save resumable caches, tables and figures under:

   ```text
   /content/drive/MyDrive/graph_specialisation_metrics/scoring_metric_refinement_dense
   ```

8. expose `--phase`, `--task`, `--force`, `--resume-config` and `--fast-dev-run`; and
9. support a cache-only `figures` phase without loading either model.

The file must work both when uploaded/run with `%run` and when pasted into a Colab cell.

## Output contract

For each task, write:

```text
cache/
    manifests/
    clean_fields/
    intervention_fields/
    scores/
    causal/
    ablation/
tables/
    raw_head_scores.csv
    derived_head_coordinates.csv
    intervention_variant_agreement.csv
    causal_validation.csv
    ablation_validation.csv
    method_ranking.csv
figures/
    raw_score_atlas.{png,pdf}
    m1_intervention_factorial.{png,pdf}
    donor_vs_transposition.{png,pdf}
    derived_DJ_atlas.{png,pdf}
    significance_validation.{png,pdf}
    role_validation.{png,pdf}
    topology_companions.{png,pdf}
protocol.json
summary.json
```

`raw_head_scores.csv` must contain at least:

```text
task, checkpoint_sha, graph_split, layer, head, method,
semantic_intervention, pe_intervention, field,
semantic_score, pe_score, topology_score,
centered, probability_weighting, graphs, events
```

## Verification and tests

Assert or test:

- semantic and PE self-donor events are exact no-ops;
- semantic and PE transpositions are involutions;
- a complete content+structure relabel leaves graph prediction invariant;
- semantic interventions leave all PE/topology/support fields fixed;
- PE interventions leave content, target and architectural support fixed;
- topology interventions leave aligned content and target fixed and recompute RRWP;
- `wV` reconstructs from collected attention and complete messages;
- attention mass is non-negative and receiver-normalized;
- probability weights sum to one on valid support;
- following/invariance scores have documented bounds and finite-value behavior;
- event/source/graph hierarchy is graph-balanced and padding invariant;
- shared event manifests are identical across compared methods;
- score, causal and ablation graph splits are disjoint;
- patch sham effects are numerically zero;
- checkpoint loading reproduces a healthy task metric;
- ZINC and QM9 content encodings survive intervention/cache round trips;
- cache fingerprints invalidate protocol/checkpoint/config changes; and
- fast-development runs produce every required table and figure.

## Acceptance criteria

The task is complete when:

1. M1–M6 run from one task-general package on both dense checkpoints.
2. All four M1 semantic/PE intervention combinations are reported.
3. Every raw score pair has a clear standalone scatter panel.
4. Semantic and PE donor-versus-transposition agreement is quantified.
5. Every method enters the same held-out ablation and causal-patching comparison.
6. Significance insight and semantic/PE role insight are evaluated separately.
7. The fixed topology channel is retained and clearly separated from PE.
8. Colab execution is resumable and writes only to the dedicated Drive output root.
9. The package README and protocol metadata fully define the estimands.
10. Unit tests and a fast-development smoke run pass before a full Colab run.
