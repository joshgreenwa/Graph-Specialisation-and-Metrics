# Chapter 6 experiment design: finite carriage under learned categorical routing

**Status:** implemented and smoke-tested; four-seed production run pending  
**Proposed protocol ID:** `query-routing-carriage-v1`  
**Scientific role:** main-text methodological validation for Chapter 6  
**Primary channel:** semantic query-key donor-swap  
**Primary comparison:** direction-matched clean Jacobian versus finite Functional carriage  
**Independent reference:** hard-routing teacher carrier field

**Implementation:** `src/graph_specialisation_metrics/synthetic/query_routing_carriage.py`  
**Colab:** `experiments/synthetic/analysis/query_routing_carriage_colab.py`  
**Tests:** `tests/test_query_routing_carriage.py`

## 1. Decision summary

Build a graph-native key-value retrieval task on heterogeneous random graphs. Each graph contains
one categorical query and one record for every possible key. The model must emit:

1. a local query contribution; and
2. the value of the record whose key matches the query.

The graph prediction is the sum of the node contributions. A semantic donor-swap changes the
query key to a different real key drawn under the registered donor law. The correct local and
record carriers after that change are known exactly from the data-generating process.

The learned model uses an ordinary softmax key-value router. When routing is confident, its clean
Jacobian can become locally flat even though a valid categorical query replacement switches the
selected record. Functional carriage measures the complete finite state change and should continue
to recover the hard-routing teacher field.

This is preferable to a hand-constructed saturating scalar pathway because:

- categorical retrieval is a standard transformer computation;
- softmax saturation is an ordinary learned mechanism;
- a real donor query is the smallest meaningful on-manifold change;
- the full eventwise distance profile is known independently of either estimator; and
- the model uses a realistic GraphGPS-style encoder on variable random graphs.

The existing
`src/graph_specialisation_metrics/synthetic/softmax_routing_carriage.py` is the conceptual
prototype. The new experiment keeps its hard-routing oracle and confidence diagnostic, but replaces
the fixed near/far toy geometry and custom compatibility-only model with random graph geometry,
explicit task splits, a GraphGPS-style contextual encoder, canonical donor manifests, canonical
aggregation, and node-resolved carrier supervision.

## 2. Scientific question and claim boundary

### Primary question

When a trained graph transformer implements high-confidence categorical routing, does a local
Jacobian recover the distance-dependent finite behaviour caused by changing the category to
another valid value?

### Intended claim

> A clean Jacobian can under-report learned long-range dependence when a model routes discrete
> alternatives through a locally saturated softmax. Functional carriage recovers the realised
> dependence under valid finite donor replacements.

### Claims this experiment must not make

- It does not show that gradients are generally inferior to finite interventions.
- It does not establish that every trained Graph Transformer is saturated.
- It does not make finite carriage donor-law independent.
- It does not validate structural carriage; this is a semantic methodological test.
- It does not identify a unique internal message-passing path. Distance remains the pristine-graph
  SPD between source and carrier.
- It does not make Beneficial carriage an independent oracle. Beneficial-carriage completeness
  follows from its definition.

## 3. Predeclared hypotheses

### H1: linear agreement

For an analytic linear key matcher acting on one-hot keys, the direction-matched Jacobian,
Functional-carriage event field, and hard-routing oracle agree to numerical tolerance at every
carrier and donor event.

### H2: finite recovery in the learned model

On healthy native-temperature checkpoints, Functional carriage has lower eventwise error against
the hard-routing oracle than the direction-matched clean Jacobian.

Primary contrast:

```text
Delta_TV = TV(Jacobian, oracle) - TV(Functional carriage, oracle)
```

The result supports H2 when the registered hierarchical 95% interval for `Delta_TV` lies above
zero and the effect is visible for every trained seed.

### H3: confidence-linked Jacobian failure

The Jacobian-to-oracle error increases as the clean matched-record probability approaches one,
while finite-carriage error remains stable or decreases.

This is tested:

1. at the model's native temperature, across events and seeds; and
2. with a predeclared post-training logit-multiplier sweep as a mechanism diagnostic.

The native-temperature result is primary. The multiplier sweep cannot substitute for it.

### H4: distance-specific under-reporting

At high routing confidence, the Jacobian preserves the local query response but under-allocates
mass to the old and new record carriers. Its source-anchored expected distance and far-mass share
therefore fall below the oracle. Functional carriage retains both distant carrier changes.

## 4. Data-generating process

### 4.1 Graph population

Each example is a fresh connected graph with:

- one query node;
- `K_key = 8` record nodes, one for each key;
- ordinary context nodes;
- a sparse random backbone; and
- no virtual or graph-token carrier.

Default graph populations:

| split | nodes | required record-distance coverage | graph families |
|---|---:|---|---|
| train/validation/ID test | 32-64 | at least `{1, 2, 3, 4-5, 6+}` | sparse ER, small-world, preferential attachment, SBM-like |
| OOD test | 80-128 | at least `{1, 2, 3, 4-7, 8-11, 12+}` | same families with larger size and diameter |

Generation retries until:

1. the graph is connected;
2. the query has finite SPD to every record;
3. the requested distance strata are represented;
4. all record nodes are distinct from the query; and
5. record identity, key, value, and distance are mutually randomized.

The generator should reuse the graph-family machinery in
`experiments/synthetic/training/structural_symbolic_graphgps.py`, but use a query node on the
connected backbone. The existing GraphWorld readout node is not suitable unchanged because its
symbolic route can be global without a meaningful molecular/backbone SPD.

### 4.2 Node fields

Every node stores:

| field | role | intervention status |
|---|---|---|
| `role` | query, record, or context | immutable control |
| `x` | one-hot categorical key in `R^K_key` | semantic payload |
| `value` | continuous record value; zero off records | fixed for query-key intervention |
| `pe` | RWSE or other topology-derived node PE | structural payload, fixed |
| `edge_index` | graph support | fixed |
| `y_node` | teacher node contribution | target only |
| `y_graph` | sum of `y_node` | target only |

All nodes receive a valid one-hot categorical key so that a donor key sampled from any
degree-matched node is meaningful. Record nodes contain a random permutation of all eight keys.
Context-node keys are i.i.d. uniform and carry no predictive relationship to record distance or
value. The model consumes `x` directly; no unchanged integer-key field may provide a shortcut around
the intervention.

Record values are independently sampled as:

```text
sign * Uniform(0.5, 1.5),  sign ~ Uniform({-1, +1})
```

Bounding their absolute magnitude away from zero prevents nominally selected records from becoming
numerically invisible.

### 4.3 Teacher carrier field

Let:

- `s` be the query node;
- `q` be its key;
- `R` be the record-node set;
- `k_i` and `v_i` be record `i`'s key and value; and
- `b(q)` be a fixed local code.

Use:

```text
b(q) = linearly spaced values from -0.75 to +0.75 over the key vocabulary.
```

The teacher contribution at node `i` is:

```text
c*_i(q) =
    b(q)                         if i = s
    v_i * 1[k_i = q]             if i is a record
    0                             otherwise
```

The graph target is:

```text
y*(q) = sum_i c*_i(q) = b(q) + v_{match(q)}.
```

The primary training target is the complete node contribution vector `c*(q)`. The graph target is
an additional consistency target. Node supervision is necessary here: without it, a graph-level
model can move the retrieved value to an arbitrary node, so the task would determine output
dependence but not the ground-truth carrier.

### 4.4 Exact donor-event oracle

For a valid donor key `q' != q`, define:

```text
O[e,i] = |c*_i(q) - c*_i(q')|.
```

Exactly three carriers can be non-zero:

```text
query carrier:          |b(q) - b(q')|
old matched record:     |v_{match(q)}|
new matched record:     |v_{match(q')}|
```

All other carriers are exactly zero. Because every key has one record, every valid donor event has
a defined counterfactual target and two known distant record carriers.

The oracle is derived from the data-generating process, not from Functional carriage or model
activations.

## 5. Split and intervention contract

### 5.1 Splits

Use disjoint generated graph IDs:

| split | default size | purpose |
|---|---:|---|
| train | 12,288 | model fitting |
| validation | 1,024 | early stopping and checkpoint selection |
| ID test | 256 | primary measurement |
| OOD test | 256 | larger/longer graph generalization |
| semantic donor pool | 512 | donor payloads only |

The donor graph IDs are disjoint from every base split. Dataset generation seeds and graph IDs are
saved in the experiment contract.

### 5.2 Semantic source domain

The task declares query-role nodes as its semantic source domain. There is exactly one query source
per graph, so it is exhaustively enumerated and treated as fixed in the bootstrap. This restriction
is role-based, frozen before model training, and independent of labels and model responses.

The source role marker is immutable. Only its categorical key is replaced.

### 5.3 Content adapter

Implement a task-local `QueryKeyContentAdapter` satisfying the existing `ContentAdapter` protocol:

```text
rows(data)                    -> [num_nodes, K_key] one-hot keys
write_donors(x, row, donors)  -> replace the complete one-hot key only
num_symbols(...)              -> None
```

The adapter must leave `role`, `value`, `pe`, `edge_index`, `y_node`, and `y_graph` unchanged.

### 5.4 Donor law

Use `SemanticDonorPool` and `build_channel_events` without response- or target-based filtering:

1. donor graph differs from the base graph by split construction;
2. donor key differs from the clean query key;
3. retain every donor node at the minimum source-donor degree gap;
4. draw donor graph uniformly, then eligible donor node uniformly;
5. draw `K_donor = 16` events independently with replacement.

Because every possible key is represented by exactly one record in the base graph, a donor key
always specifies a valid alternative retrieval.

Save the complete `DonorEvent` record, event-manifest fingerprint, clean/donor key, old/new record,
degree gap, and one-hot donor dose. The expected dose is `sqrt(2)`; deviations are an audit error.

### 5.5 Target handling

The model is evaluated against the original clean target for loss and Beneficial carriage, exactly
as required by the canonical intervention. The counterfactual teacher field `c*(q')` is used only
as an external behavioural oracle; it never replaces the registered target during carriage.

## 6. Models

### 6.1 Analytic linear control

Represent keys as one-hot vectors. For query one-hot `u_q` and record key one-hot `u_i`, define:

```text
p_i(q) = u_q^T u_i
c_query(q) = b^T u_q
c_record_i(q) = p_i(q) * v_i
```

This map is affine in `u_q`. Therefore, for every donor direction
`delta_u = u_q - u_q'`, the clean directional Jacobian exactly equals the finite endpoint
difference.

This control does not need training. It validates the event construction, Jacobian direction,
carrier geometry, normalisation, distance accounting, and plotting code.

### 6.2 Learned GraphGPS-style router

Implement a pure-PyTorch dense GraphGPS-style model using the established components:

- role embedding;
- linear embedding of key one-hot vectors;
- linear value projection;
- RWSE projection;
- three GraphGPS layers;
- four attention heads;
- hidden width 96;
- local GINE-style branch;
- dense self-attention branch; and
- an explicit final query-to-record softmax router.

The explicit router is:

```text
z_i(q) = <W_q h_s, W_k h_i> / sqrt(d_head)
p_i(q; beta) = softmax_i(beta * z_i(q))       over record nodes only
```

The final scalar carrier state immediately before pooling is:

```text
h^L_s = local_head(u_q)
h^L_i = p_i(q; beta) * value_i               for record i
h^L_i = 0                                    otherwise
```

The readout is strictly linear:

```text
y_hat = sum_i h^L_i.
```

`h^L` in this experiment is therefore the scalar node-contribution vector, not the hidden
GraphGPS embedding before the contribution head. This registration makes:

```text
d y_hat / d h^L_i = 1
```

at every carrier and prevents final-readout saturation from contaminating the routing comparison.

The model returns a structured result containing:

```text
node_contributions    [batch, node]
graph_prediction      [batch]
final_hidden          [batch, node, hidden]
router_logits         [batch, record]
router_probability    [batch, record]
attention_by_layer    optional diagnostics
```

### 6.3 Training objective

Use:

```text
L_node  = mean over valid nodes (h^L_i - c*_i)^2
L_graph = mean (sum_i h^L_i - y*)^2
L_total = L_node + 0.25 * L_graph
```

No Jacobian, carriage, attention entropy, oracle range, or counterfactual event enters training or
checkpoint selection.

Training defaults:

```text
seeds                 = 0,1,2,3
optimizer             = AdamW
learning_rate         = 3e-4
weight_decay          = 1e-4
batch_size            = 32
maximum_epochs        = 100
early_stop_patience   = 12 validation epochs
gradient_clip_norm    = 1.0
dropout               = 0.10
attention_dropout     = 0.10
native beta           = 1.0
```

Dropout is disabled for every measurement. Save the minimum-validation-loss checkpoint per seed;
do not select a seed for the paper.

### 6.4 Model health gates

The main comparison is reported only when every seed satisfies on ID test:

```text
matched-record top-1 accuracy       >= 0.98
mean matched-record probability     >= 0.95
node-contribution MAE               <= 0.03
graph-output MAE                    <= 0.05
oracle finite-field relative error  <= 0.10
```

The last quantity compares the model's direct finite node-contribution change with the teacher
field before applying either range estimator. If it fails, the model has not learned the required
counterfactual behaviour and the run cannot validate an estimator.

Threshold failures remain visible and are reported as model failures, not converted into favourable
metric results.

## 7. Measurements

All primary fields are computed donor-event first, before donor averaging.

### 7.1 Oracle field

```text
M_O[e,i] = O[e,i] = |c*_i(q) - c*_i(q')|.
```

### 7.2 Direction-matched clean Jacobian

Let `u_q` be the clean one-hot query and `u_q'` the donor one-hot query. Compute a Jacobian-vector
product of the complete carrier vector with the exact donor direction:

```text
delta_u = u_q - u_q'
M_J[e,i] = |J_i(u_q) delta_u|.
```

The Jacobian is taken through the clean learned model at the native query input, with role, values,
PE, support, and every non-source node fixed.

Use `torch.func.jvp` when supported by the pinned PyTorch runtime. Provide an autograd JVP fallback
and test the two implementations against each other.

This direction-matched field is the primary Jacobian comparator because it shares:

- the same donor;
- the same displacement direction;
- the same carriers;
- the same absolute scalar output geometry; and
- the same distance aggregation as Functional carriage.

It isolates tangent-versus-finite behaviour.

### 7.3 Functional carriage

For event `e`:

```text
Delta h[e,i] = h^L_clean[i] - h^L_event[i]
g_out[i] = 1
M_F[e,i] = |g_out[i] Delta h[e,i]| = |Delta h[e,i]|.
```

Call the public production `functional_carriage_events` with:

```text
delta shape          [source=1, donor=K, carrier=N, width=1]
clean-gradient shape [output=1, carrier=N, width=1]
```

The production donor-averaged field is:

```text
F_sens[g,i,s] = mean_k M_F[(g,s,k),i].
```

Do not average signed changes before taking magnitude.

### 7.4 Supplementary entrywise Jacobian

Also retain the Bamberger-style entrywise Jacobian-block mass:

```text
M_J,L1[i,s] = sum_{a,b} |d output_i^a / d input_s^b|.
```

It is supplementary because it differs from Functional carriage in channel aggregation as well as
perturbation scale. It must not be used to attribute a gap specifically to saturation.

The main source-anchored first moment below is not identical to Bamberger et al.'s
carrier-anchored mean-over-sources estimator. With one task-declared query source per graph, the
original orientation is degenerate for many carriers. The paper must state the orientation rather
than call the two summaries numerically identical.

### 7.5 Beneficial carriage

Beneficial carriage is predeclared secondary. Use the graph MSE against the original clean target:

```text
ell = (sum_i h^L_i - y_clean)^2.
```

Run canonical `beneficial_carriage` over the straight path in scalar node-contribution space.
Require:

```text
sum_i B[e,i] = loss_event[e] - loss_clean
```

within the registered tolerance for every event.

Beneficial carriage is used to show where task-loss increase is allocated. It is not scored
against the teacher oracle as if it were an independent predictor of loss change.

## 8. Profiles and scalar summaries

### 8.1 Raw fields are primary

For each method `m in {O, J, F}`, retain the raw carrier field and report:

```text
raw mass by exact SPD
raw mass by canonical adaptive distance bin
total response mass
far response mass
```

The default adaptive bins remain:

```text
{0}, {1}, {2}, {3}, {4-7}, {8-15}, {16-31}, ...
```

`d = 0` is always included.

### 8.2 Event-normalised shape

For an event with denominator above the registered effect floor:

```text
M_tilde_m[e,i] = M_m[e,i] / sum_j M_m[e,j].
```

Record and exclude non-estimable events rather than stabilising a zero denominator.

The event-level source-anchored expected distance is:

```text
rho_m[e] = sum_i M_tilde_m[e,i] d_g(i,s).
```

This is a shape-only companion. It never replaces the raw field.

### 8.3 Primary error metrics

Against the hard-routing oracle:

```text
relative raw-field L1 error
    = sum_i |M_m[e,i] - O[e,i]| / sum_i O[e,i]

profile total variation
    = 0.5 * sum_i |M_tilde_m[e,i] - O_tilde[e,i]|

expected-distance absolute error
    = |rho_m[e] - rho_O[e]|

far-share absolute error at r=3
    = |sum_{d>3} M_tilde_m - sum_{d>3} O_tilde|
```

The primary inferential contrast is the within-event difference between Jacobian and Functional
carriage for profile total variation. Raw-field and range errors are supporting endpoints.

### 8.4 Confidence diagnostic

For every clean event source, save:

```text
p_match
normalised router entropy
clean matched-vs-runner-up logit margin
softmax derivative factor p_match * (1 - p_match)
```

Predeclare confidence strata from fixed probability thresholds rather than outcome-optimized
quantiles:

```text
low:       p_match < 0.80
moderate:  0.80 <= p_match < 0.95
high:      p_match >= 0.95
```

The high-confidence stratum is the principal mechanism test.

## 9. Aggregation and uncertainty

The hierarchy is:

```text
trained seed -> graph -> query source -> donor event.
```

There is one exhaustively enumerated query source per graph, so source resampling is disabled and
documented. Resample:

```text
seed -> graph -> donor
```

with:

- exactly 2,000 percentile-bootstrap replicates;
- a 20% trimmed mean across graph estimates;
- central 2.5% and 97.5% quantiles;
- RNG seed `17_071`; and
- seed-level estimates shown individually.

Use `Observation` and `nested_percentile_interval` from the production methodology package.
Set `BootstrapPolicy.resample_source=False`; retain the other production defaults.

Distance cells must satisfy the canonical reporting floor:

```text
at least 10 contributing graphs
at least 50 eligible (carrier, source) pairs
```

Within an eligible event, a distance containing no registered carrier has exactly zero mass, so
the eventwise profile remains a probability distribution. Every aggregate row also records graph
and carrier-source support; production figures mask cells below either reporting floor.

## 10. Experimental arms

### Arm A: analytic linear exactness

Run the full event, oracle, Jacobian, carriage, profile, aggregation, and caching pipeline on the
analytic linear router.

Required:

```text
max carrier-field error       <= 1e-10 in float64
max expected-distance error   <= 1e-10 in float64
```

### Arm B: learned native-temperature model

This is the headline arm. Measure every healthy seed on fixed ID and OOD base graphs with the
registered semantic donor events.

No temperature, dose, donor, or checkpoint is chosen in response to the observed Jacobian gap.

### Arm C: logit-multiplier mechanism sweep

Evaluate the frozen checkpoint at:

```text
beta multiplier = {0.25, 0.5, 1, 2, 4}.
```

`1` is the native model. This sweep asks whether increasing routing confidence suppresses the clean
Jacobian while preserving finite endpoint switching. It is a diagnostic model intervention, not a
new trained model and not the primary result.

Report task and counterfactual fidelity at every multiplier. A point at which the finite model no
longer follows the hard oracle cannot support the mechanism claim.

### Arm D: donor-dose ladder

As an explicitly off-protocol diagnostic, interpolate in one-hot space:

```text
u(alpha) = u_q + alpha (u_q' - u_q)
alpha = {1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1}.
```

The ladder should converge to the direction-matched Jacobian as `alpha -> 0` and to the registered
finite donor event at `alpha = 1`.

Only `alpha = 1` is a valid semantic donor-swap and may enter the primary analysis.

### Arm E: OOD distance generalization

Repeat native-temperature measurement on larger graphs with longer record distances. This tests
whether the estimator gap persists when the same learned categorical operation is applied farther
from the query.

OOD failure of the model is scientifically informative but cannot be used as estimator validation
unless direct finite counterfactual fidelity remains adequate.

## 11. Interpretation matrix

| Result | Interpretation |
|---|---|
| Linear control gives `J = F = oracle` | Event direction, carrier geometry, and aggregation are implemented correctly. |
| Native model is healthy; `F` matches oracle and `J` loses far mass | Evidence for locally flat learned routing missed by the clean tangent. |
| `J` and `F` both match oracle | The native model is not meaningfully saturated at the registered intervention site. |
| Both miss oracle and direct finite fidelity is poor | Model/task failure; no estimator conclusion. |
| Direct finite fidelity is good but `F` misses oracle | Carrier/readout registration or implementation is wrong. |
| Gap appears only after multiplying logits | Demonstrates a possible failure mode, not that the native model exhibits it. |
| `F` shows additional non-oracle carriers | The model performs collateral finite computation beyond the teacher route. |
| `F` is long-range but `B` has little far mass | Finite long-range response is active without corresponding signed loss benefit under the registered path. |

## 12. Cache and artifact contract

Default root:

```text
outputs/query_routing_carriage_v1/
```

Required layout:

```text
experiment.json
data/
  split_manifest.json
  generation_summary.json
checkpoints/
  seed_000.pt
  seed_001.pt
  seed_002.pt
  seed_003.pt
cache/
  events/
    id_manifest.json
    ood_manifest.json
  measurements/
    linear_control.parquet
    seed_000_id.parquet
    seed_000_ood.parquet
    ...
results/
  model_health.csv
  event_metrics.parquet
  distance_profiles.csv
  primary_contrasts.json
  beneficial_summary.json
figures/
  query_routing_carriage.png
  query_routing_carriage.pdf
  query_routing_carriage.metadata.json
```

Use CSV instead of Parquet only if the workspace dependency set does not include a stable Parquet
writer. The logical schemas remain fixed.

### Event-carrier measurement schema

At minimum:

```text
protocol_version
experiment_fingerprint
checkpoint_sha256
seed
split
graph_id
source
donor_draw
donor_graph_id
donor_node
clean_key
donor_key
old_record
new_record
carrier
carrier_role
distance
degree_gap
donor_dose
beta_multiplier
alpha
oracle_mass
jacobian_directional_mass
jacobian_entrywise_l1_mass
functional_event_mass
beneficial_event_mass
clean_match_probability
event_match_probability
clean_entropy
clean_logit_margin
clean_loss
event_loss
```

The full donor-resolved table is the source of every aggregate figure. Never cache only plotted
means.

### Fingerprint inputs

Bind every cache to:

- protocol ID;
- complete experiment configuration;
- data-generation code version;
- split graph IDs and seeds;
- donor manifest;
- model geometry;
- checkpoint SHA-256;
- task adapter version;
- registered carrier site;
- registered query input/Jacobian site;
- output scale;
- distance bins;
- effect floor;
- bootstrap policy; and
- numerical tolerances.

A mismatch fails with instructions to choose a new output directory. It must never silently
overwrite an incompatible cache.

## 13. Proposed implementation files

### New files

```text
src/graph_specialisation_metrics/synthetic/query_routing_carriage.py
experiments/synthetic/analysis/query_routing_carriage_colab.py
tests/test_query_routing_carriage.py
```

The Python module owns:

- frozen dataclass configuration and fingerprint;
- graph generator and split materialization;
- task data container and collation;
- `QueryKeyContentAdapter`;
- analytic linear router;
- learned GraphGPS router;
- training and checkpoint health;
- canonical donor-event construction;
- oracle/Jacobian/Functional/Beneficial measurements;
- aggregation;
- cache-only figure generation; and
- CLI phases.

CLI:

```text
--phase data
--phase train
--phase measure
--phase figures
--phase all
```

Figures must be regenerable with `--phase figures` without importing a checkpoint or running model
inference.

### Small production-methodology refactor

Promote event normalisation from the private runner helper to a public tested function in:

```text
src/graph_specialisation_metrics/methodology/carriage.py
```

Suggested interface:

```python
event_normalise_functional(event_field, *, effect_floor)
```

It returns the normalized field, eligible-event mask, and denominators. The existing methodology
runner calls the same public function so the experiment and production figures cannot drift.

Do not duplicate the canonical nested bootstrap, donor sampler, semantic swap, or shortest-path
implementation.

## 14. Required tests

### Generator

- every graph is connected;
- exactly one query exists;
- exactly one record exists per key;
- query and record nodes differ;
- required distance strata are present;
- record value, key, and distance are empirically independent within tolerance over a fixture;
- teacher node contributions sum to the graph target; and
- ID and OOD graph ID sets are disjoint.

### Intervention

- only the query key changes;
- role, value, PE, support, and targets remain bit-identical;
- donor key differs from the source key;
- every donor key resolves to exactly one base-graph record;
- donor dose equals `sqrt(2)` in one-hot geometry;
- event manifests are deterministic under their seed; and
- a no-op query replacement yields zero oracle and zero model response.

### Linear exactness

- eventwise Jacobian equals finite delta at every carrier;
- Functional carriage equals the finite delta magnitude;
- all three fields equal the teacher oracle;
- raw distance profiles reconstruct total mass; and
- normalized profiles sum to one for eligible events.

### Learned softmax mechanism

On a hand-set high-margin router:

- matched probability approaches one;
- remote directional-Jacobian mass approaches zero;
- finite old/new record mass remains non-zero;
- local Jacobian mass remains estimable; and
- Functional carriage equals the direct finite carrier change.

### Jacobian implementation

- `torch.func.jvp` agrees with the fallback;
- JVP agrees with a central finite difference at sufficiently small `alpha`;
- batch evaluation agrees with single-graph evaluation; and
- dropout is disabled during measurement.

### Carriage and loss

- `functional_carriage_events` receives the declared geometry;
- event magnitude is taken before donor averaging;
- Functional-carriage event mass equals direct scalar carrier change;
- Beneficial-carriage completeness holds per event;
- event normalisation excludes only denominators at or below the registered floor; and
- explicit `d=0` plus all numeric/special distance buckets reconstruct total mass.

### Aggregation and caching

- donor events nest within source, source within graph, graph within seed;
- source resampling is disabled and recorded;
- the production 2,000-draw bootstrap is used outside smoke mode;
- structural-zero distance bins retain zero mass and low-support aggregate cells are masked;
- cache fingerprints fail closed;
- figure generation is cache-only; and
- smoke artifacts cannot be mistaken for paper artifacts.

## 15. Figure plan

### Main figure: four panels

1. **Task and oracle.** One representative random graph showing query, old record, new record,
   pristine distances, and exact finite carrier changes.
2. **Method agreement control.** Linear-router oracle, Jacobian, and Functional distance profiles,
   visually coincident with an exactness annotation.
3. **Native learned model.** Raw and event-normalised oracle/Jacobian/Functional distance profiles
   with hierarchical intervals and seed curves.
4. **Failure mechanism.** Oracle error or far-mass share against routing confidence/logit
   multiplier, with native temperature explicitly marked.

### Supplementary figures

- ID versus OOD profiles;
- donor-dose ladder;
- full native seed panels;
- raw response scale alongside normalized shape;
- entrywise-L1 Jacobian comparison;
- Beneficial-carriage `S_B` and `B_far`;
- attention entropy/margin diagnostics; and
- distance support counts.

## 16. Implementation sequence and gates

### Stage 1: exact CPU fixture

Implement the generator, adapter, teacher, analytic router, measurement table, and tests. Do not
start learned training until linear exactness passes.

### Stage 2: learned smoke run

Use:

```text
2 seeds
16 train graphs per batch
256 optimization steps
16 ID graphs
16 donor graphs
2 donors per source
no production bootstrap
```

This is an engineering check only. Stamp all artifacts `smoke=true`.

### Stage 3: single-seed pilot

Train one full checkpoint and verify:

- the task is learnable;
- the native model becomes confident without logit manipulation;
- direct finite counterfactual fidelity passes;
- the Jacobian/finite gap, if present, has the predicted carrier pattern; and
- GPU memory and measurement runtime are acceptable.

Pilot results may set engineering batch sizes and memory chunking, but must not change scientific
endpoints or choose favourable thresholds.

### Stage 4: freeze `v1`

Freeze the complete configuration and fingerprint before running all four seeds. Any scientific
change after this point creates `v2`.

### Stage 5: production run and audit

Run data, training, measurement, 2,000-draw aggregation, cache-only figures, and the complete audit
suite. Retain all seed results whether or not they agree with the hypothesis.

## 17. Principal risks and controls

| Risk | Control |
|---|---|
| Model hides the selected value at the query node | Supervise the complete node contribution vector and register the scalar pre-sum contributions as carriers. |
| Finite carriage wins tautologically | Score both estimators against the independently generated hard-routing teacher, require model counterfactual fidelity, and include exact linear agreement. |
| Gap is caused by L1-versus-L2 aggregation | Make the exact donor-direction JVP the primary Jacobian; keep entrywise L1 supplementary. |
| Saturation occurs in the readout and defeats both methods | Use scalar node contributions followed by a strictly linear sum readout. |
| A distance or record-order shortcut solves the task | Randomize key-to-record assignment, record placement, graph family, value, and distance independently for every graph. |
| A donor query is invalid | Include one record for every key in every base graph and exclude identical-key donors. |
| Result exists only after artificial temperature scaling | Make the native checkpoint the headline condition; treat the multiplier sweep as a mechanism diagnostic. |
| One source per graph understates uncertainty | Use many independent graphs and donors, show all seeds, and explicitly condition on the exhaustively enumerated query source. |
| Expected distance hides scale failure | Make raw fields and raw distance profiles primary; normalized range is a companion. |
| Original Bamberger orientation is conflated with carriage | Label the primary summary source-anchored and report the entrywise/carrier-anchored variant separately where meaningful. |

## 18. Go/no-go criterion for the Chapter 6 main text

Promote the experiment to the main text only if:

1. the linear exactness control passes;
2. all native checkpoints pass the task and counterfactual health gates;
3. the Jacobian-versus-finite difference is present at native temperature, not only after
   multiplier manipulation;
4. the predicted discrepancy is localized to old/new distant record carriers;
5. Functional carriage is closer to the independent teacher oracle on every seed; and
6. the result survives ID/OOD, raw/normalized, and donor-event sensitivity reporting.

If the native model does not saturate, report agreement as a valid negative result and move the
temperature sweep to an appendix existence proof. If direct finite fidelity fails, improve the
model or task before interpreting either range estimator.
