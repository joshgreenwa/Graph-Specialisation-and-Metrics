# Final methodology: donor-swap specialisation and carriage

This document is the **normative, task-general methodology** for this repository. It fixes the
scientific definitions of:

1. semantic and structural donor-swap interventions;
2. raw per-head semantic and structural specialisation scores;
3. semantic and structural functional carriage using `F_sens`; and
4. semantic and structural beneficial carriage using signed, donor-wise finite-loss paths.

**Protocol version:** `donor-swap-specialisation-carriage-v3`.

The same two intervention distributions are used throughout. Semantic scoring, semantic
functional carriage, and semantic beneficial carriage use the same semantic donor-swap.
Structural scoring, structural functional carriage, and structural beneficial carriage use the
same structural donor-swap.

This is the only core methodology. The method has no subtype name. Older documents and code paths
describe the experiments used to choose it, including transpositions, coherent aggregation, and
topology-changing controls; those are not alternative production defaults.

The production workflow has three stages:

1. **Measure:** compute and cache raw semantic and structural head scores for each trained seed,
   retaining event-, source-, graph-, and distance-resolved sufficient statistics.
2. **Validate causally:** relate the derived head coordinates to held-out ablation, family
   ablation, restoration/injection patching, and aligned rescue/induction against matched controls.
3. **Map carriage:** compute final-state Functional carriage and signed path-integrated Beneficial
   carriage under the same two donor-swap event distributions.

The mandatory score figures are:

- Structural score on `x` against Semantic score on `y`; and
- Selectivity `D_rel` on `x` against Joint sensitivity `J` on `y`.

The exact quantities, labels, uncertainty treatment, and causal endpoints are fixed below.

## 1. Scientific object and terminology

A graph `g` has:

- semantic node payloads `C_g = (c_1, ..., c_n)`;
- immutable task/control fields `R_g`, such as query, source, or mode markers when a controlled
  task contains them;
- topology-derived structural payload `P_g`, such as node and pair RRWP/PE fields; and
- fixed architectural support `W_g`, such as molecular edges, bond attributes, and a sparse
  attention mask.

For ordinary molecular tasks, the semantic payload is normally the complete node `x` row. For a
controlled task whose `x` row also contains experimental role markers, the task's content adapter
must identify the replaceable semantic payload and leave those control fields fixed. This boundary
must be declared once per task; it must never be inferred from hard-coded feature indices inside
the scorer.

The structural channel in the core method means **topology-derived positional payload presented to
the model on fixed support**. It does not mean that the graph's molecular topology is replaced.
Consequently:

- `semantic` means reliance on node content;
- `structural` means reliance on node/pair PE or RRWP payload; and
- `topology` means a support- or isomorphism-changing intervention and is not a core channel.

A **source** `s` is the single node whose semantic or structural payload is replaced. A **donor**
`k` supplies a real alternative payload. A **carrier** `i` is a node at which the intervention's
transport response is read. Clean shortest-path distance is denoted `d_g(i,s)`.

“Donor-swap” is the established paper-facing term even though each event is a one-way replacement,
not a reciprocal exchange.

## 2. The two core interventions

### 2.1 Semantic donor-swap

For source `s` and donor node `v` from another graph in the declared donor pool:

```text
c'_s = c_donor(v)
c'_i = c_i                         for i != s
R'_g = R_g
P'_g = P_g
W'_g = W_g
y'_g = y_g
```

The donor supplies the complete task-declared semantic payload. Structural encodings, topology,
bond values, attention support, graph target, and immutable task controls remain exactly fixed.

Core donor eligibility is:

1. the donor graph is not the evaluated base graph;
2. the donor payload differs from the source payload;
3. compute the smallest available absolute source--donor degree gap over all remaining donor
   nodes; and
4. retain every donor node attaining that smallest gap, including all ties.

Donors are sampled without using the graph target or model response. The donor pool must come from
a predeclared data split representing the same task/data regime, and its graph IDs are disjoint
from every base-graph split. Section 9 fixes how this pool may be reused across analysis stages.

The semantic donor probability law is graph-balanced. For every draw, sample uniformly from donor
graphs containing at least one eligible minimum-gap node, then sample uniformly from the eligible
nodes in that graph. Draw the `K` donors independently with replacement. Thus large donor graphs
do not receive greater probability merely because they contain more nodes. Record the selected
graph, node, degree gap, and payload fingerprint for every draw.

### 2.2 Structural donor-swap

For source `s` and a different, degree-matched donor node `v` in the same graph, copy the donor's
complete topology-derived positional footprint onto the source. For a node PE `p` and a possibly
directed pair PE `r`, the operation is defined entrywise by:

```text
node PE:
    p'_s       = p_v
    p'_i       = p_i                         for i != s

pair PE:
    r'_{i,j}   = r_{v,j}                     if i = s and j != s
               = r_{i,v}                     if j = s and i != s
               = r_{v,v}                     if i = s and j = s
               = r_{i,j}                     otherwise

fixed:
    C'_g       = C_g
    R'_g       = R_g
    W'_g       = W_g
    y'_g       = y_g
```

This operation replaces only entries incident to the single declared source. It does not
reciprocally replace the donor, move a second node, relabel node identities, change graph topology,
or change the architectural attention mask.

This dense entrywise definition is normative even when a field is stored sparsely. A sparse
implementation must materialize exactly the same relation: remove the source's previous incident
entries, copy the donor's outgoing and incoming entries into the source coordinate, set the source
self-entry to the donor self-entry, sort the resulting indices, and coalesce duplicates. Retain
one copy only when all duplicate payloads agree within the registered tolerance; abort on
conflicting duplicate values. Duplicate payloads are never summed. Dense-versus-sparse equivalence
is a required intervention check.

For a PyG/GRIT task, every present topology-derived payload field must be handled consistently.
This normally includes node-indexed fields such as:

```text
rrwp, deg, log_deg, abs_pe, pestat_RRWP
```

and pairwise RRWP/PE fields such as:

```text
rrwp_index + rrwp_val
```

The following remain fixed:

```text
x, y, edge_index, edge_attr, rrwp_local_edge_index, architectural masks
```

The task adapter must declare any additional PE fields or support tensors. Unknown structural
fields are a fatal audit error rather than something to ignore silently.

Core donor eligibility is:

1. `v != s`;
2. the copied footprint differs from the source footprint;
3. compute the smallest available absolute source--donor degree gap over the remaining nodes; and
4. retain every donor node attaining that smallest gap, including all ties.

If no nontrivial eligible donor exists, the source is marked non-estimable; a self-donor is used
only as a deterministic no-op check. Structural donors are sampled uniformly over the eligible
minimum-gap nodes in the base graph. Draw the `K` donors independently with replacement so donor
identity is marginalised in the same manner as on the semantic side.

This is a fixed-support PE/RRWP intervention. Copying one real footprint into another node's
coordinate slot can duplicate a structural role and need not produce the RRWP of any realizable
graph. It is therefore a controlled structural-payload sensitivity intervention, not a claim to an
on-manifold topology edit. The limitation must be stated in papers and artifact metadata.

### 2.3 Shared event protocol

For both channels:

- sources are sampled uniformly without replacement within each graph, subject to a declared
  memory cap;
- `K` eligible donors are drawn independently with replacement for every source under the
  channel-specific probability law above;
- the clean graph and all events for one source are evaluated in the same forward batch whenever
  possible;
- the clean member of that batch is the numerical baseline for every event;
- source and donor identities are deterministic under a recorded seed; and
- aggregation is hierarchical: events within source, sources within graph, graphs equally.

The semantic and structural analyses should use the same base graph IDs and source IDs when task
geometry permits. Their event manifests, donor matching tiers, degree gaps, and nontrivial-event
rates must be saved.

Both interventions are controlled counterfactual corruptions. A semantic row sampled from a real
node need not be chemically or contextually valid after insertion into a different graph, and a
copied PE/RRWP footprint need not correspond to any realizable topology. The original target is
held fixed deliberately. Results therefore measure preservation, transport, and causal use of the
clean-task signal under the declared corruption distribution; they do not measure performance on a
valid alternative graph.

## 3. Shared transport notation

For layer `l`, head `h`, receiving node `i`, and sender `j`, let:

```text
A^{lh}_{ij}          post-softmax attention
m^{lh}_{ij}          complete pre-attention message, including edge enhancement
o^{lh}_i = wV^{lh}_i = sum_j A^{lh}_{ij} m^{lh}_{ij}
```

`o^{lh}_i` is the per-head **transport site**: the routed message that head `(l,h)` delivers to
carrier `i`. Let `H^L = (h^L_1, ..., h^L_n)` be the final node states and let `rho` denote graph
pooling plus the prediction head, so `ŷ = rho(H^L)`. The collector must use the actual hooked
tensor on the model's gradient path, not a reconstruction detached from the readout.

Every task registers one score/causal output representation `ŷ in R^T` and a fixed positive
diagonal scale:

```text
z_t = ŷ_t / sigma_t
||z|| = (sum_t z_t^2)^(1/2)
```

For multi-target regression, `ŷ` is in evaluation space and `sigma_t` is the training-set target
standard deviation. For classification, `ŷ` is the logit vector and `sigma_t=1` unless the task
registration explicitly fixes another representation and scale. Single-target and other tasks
must likewise register their representation and positive `sigma_t`. The scale is estimated from
training data only, saved with the task registration, and never refitted on analysis graphs.

Raw scores, Functional carriage, clean prediction movement, donor-wise necessity, patch
responses, alignment, and causal effect floors all use this same `z`-space geometry. Beneficial
carriage remains defined by the registered scalar task loss in Section 6.

At the clean input:

```text
phi^{lh}_{t,i} = d z_t / d o^{lh}_i
```

For channel `c in {semantic, structural}`, graph `g`, source `s`, and donor event `k`:

```text
Delta o^{lh,c}_{g,s,k,i}
    = o^{lh}_{g,clean,i} - o^{lh}_{g,c(s,k),i}

q^{lh,c}_{g,s,k,i,t}
    = phi^{lh}_{g,t,i} dot Delta o^{lh,c}_{g,s,k,i}
```

`q` is a `T`-dimensional first-order transport vector in registered `z`-space. The intervention is
finite and propagated through the complete nonlinear transformer; only the final projection from
the transport site to `z` is linearized at the clean input.

The sign convention is always `clean - intervention`.

## 4. Raw semantic and structural head scores

The production head score is graph-balanced **output-projected transport**:

```text
E^{lh,c}_{g,s,k}
    = sum_i ||q^{lh,c}_{g,s,k,i,:}||_2

S_{c,g}(l,h)
    = mean_{s in sources(g)} mean_{k=1..K} E^{lh,c}_{g,s,k}

S_c(l,h)
    = mean_g S_{c,g}(l,h)
```

Therefore:

```text
S_sem(l,h) = raw Semantic score under semantic donor-swaps
S_str(l,h) = raw Structural score under structural donor-swaps
```

These are the two raw production scores. They are non-negative and remain in the registered
`z`-space intervention-response units.

The order of operations is mandatory:

1. project each event at each carrier;
2. take the `L2` magnitude across output dimensions for that event and carrier;
3. sum carrier magnitudes;
4. average donor events within source;
5. average sources within graph; and
6. average graphs equally.

Magnitude is taken before donor averaging, so valid events with opposite directions do not cancel.
Carrier magnitudes are summed before graph averaging, so strong internal transport is not erased
when carriers have opposing output directions. Large graphs do not dominate merely because they
contain more sampled sources.

No normalization, ratio, selectivity coordinate, attention-only statistic, task loss, or causal
ablation is part of either raw score.

Because the clean readout gradient and norm use the registered representation and diagonal scale,
raw amplitudes are directly comparable across heads within a registered task. Cross-task amplitude
comparisons require a shared interpretation of the registered `z` coordinates; target-standardized
regression or unit-scaled logits do not by themselves make different tasks scientifically
commensurate.

### 4.1 Cache and aggregation contract

Raw scoring is run separately for every trained seed. A cache must retain, for both channels:

```text
E[g,s,k,l,h]           event score
S_graph[g,l,h]         source- and donor-averaged graph score
C_graph[g,l,h,d]       exact-distance contribution
O_graph[g,d]           event-carrier support
event/source/donor manifests and matching metadata
```

An implementation may omit the full event tensor only if the retained source-level sufficient
statistics exactly reproduce `S_graph` and `C_graph` and the donor audit table remains available.
The sequence `donor events -> sources -> graphs` is never flattened into one sample mean.

For multiple training seeds, compute a complete independent cache for each seed. Head index
`(l,h)` is not assumed to identify the same learned feature across seeds, so raw head scores are
not averaged index-wise across seeds unless an independently validated head-alignment procedure is
declared. Figures show seed-specific heads, using a fixed marker or facet for seed. Population
statistics use seeds as independent top-level replicates and graphs as clustered observations
within seed. With fewer than three trained seeds, report the seed estimates and their within-seed
graph intervals rather than presenting an unstable seed-population confidence interval.

### 4.2 Mandatory score plane and normalization

For each task, model, and trained seed, define the within-model channel means:

```text
bar(S_sem) = mean_{l,h} S_sem(l,h)
bar(S_str) = mean_{l,h} S_str(l,h)

s_sem(l,h) = S_sem(l,h) / bar(S_sem)
s_str(l,h) = S_str(l,h) / bar(S_str)
```

The means are computed over all registered real attention heads. They are visual calibration
constants, not new scientific measurements; the raw `S_sem`, `S_str`, and both means must always
be saved. Both means must be finite and exceed a registered numerical floor. If either channel
fails that check, its normalized plane, `J`, and `D_rel` are reported as non-estimable rather than
stabilized into an apparent result.

The primary head scatter has one point per head and uses exactly:

```text
x-axis: Structural score  s_str = S_str / bar(S_str)
y-axis: Semantic score    s_sem = S_sem / bar(S_sem)
```

Paper-facing axis labels must be:

```text
Structural score  $S_{str}/\overline{S}_{str}$
Semantic score  $S_{sem}/\overline{S}_{sem}$
```

These strings intentionally match the mixed-synthetic paper figure and are the repository-wide
figure convention.

Use equal axis limits and aspect ratio, include the identity line, colour by layer, and distinguish
training seeds without implying cross-seed head alignment. Every head point carries faint
horizontal and vertical 95% intervals computed by rerunning the complete graph-balanced
aggregation under the uncertainty hierarchy in Section 7. A clean inset or adjacent interval panel
may be used when direct bars would obscure the scatter, but a point-only figure without its
interval companion is incomplete. Panels must never silently pool task, model, or seed.

### 4.3 Exact distance decomposition and score heatmaps

The raw score can be decomposed without redefining it:

```text
C_{g,c}(l,h,d)
    = mean_s mean_k
      sum_{i: d_g(i,s)=d} ||q^{lh,c}_{g,s,k,i,:}||_2

S_{c,g}(l,h) = sum_d C_{g,c}(l,h,d)
S_c(l,h)     = sum_d mean_g C_{g,c}(l,h,d)
```

Distance is measured on the pristine graph. Because both core interventions change one declared
source, the distance anchor is `s` for both channels. For each channel, model, and task, the
mandatory exact-contribution matrix is head-resolved:

```text
H_c(l,h,d) = mean_g C_{g,c}(l,h,d)
H_c(l,d)   = sum_h H_c(l,h,d)
```

`H_c(l,h,d)` is what the heatmap displays: one row per head, blocked by layer, layer `0` in the top
block, columns the integer distances `0, 1, ..., d_max`. Heads are not summed for display, because a
layer's heads routinely peak at different distances and the head sum then reports every layer as
broader than any head inside it. The head-summed `H_c(l,d)` is retained as the aggregate
measurement and is what the accompanying distance-profile figure and its intervals are built from.
Save both before any display normalization.

Two normalizations are registered, and both figures are mandatory for a channel:

```text
H_c(l,h,d)                        (primary, absolute)
H_c(l,h,d) / sum_d H_c(l,h,d)     (companion, row-normalized)
```

The primary figure carries the between-layer and between-head magnitude differences. The companion
divides each head by its own total over the **whole** registered distance axis, so it reports profile
shape at fixed head magnitude, and blanking a column under the reporting floor can never inflate the
columns that remain — a displayed row sums to at most one. A head with no mass anywhere stays
non-estimable rather than becoming a uniform profile. Publishing the companion alone is incomplete:
it is the primary figure that shows which heads carry the mass being profiled.

The event-carrier support at distance `d` is the mean opportunity count within graph:

```text
O_{g,c}(d)
    = mean_s mean_k # {i : d_g(i,s)=d}
```

The corresponding per-opportunity response is computed **within graph before graph averaging**:

```text
R_c(l,h,d) = mean_{g: O_{g,c}(d)>0} [ C_{g,c}(l,h,d) / O_{g,c}(d) ]
R_c(l,d)   = sum_h R_c(l,h,d)
```

`R_c` occupies the second panel of both heatmap figures, at the same head-resolved geometry and
under the same two normalizations as `H_c`. It is the exact score contribution divided by
event-carrier support, not a replacement score. This within-graph division prevents large or
long-diameter graphs from supplying both the numerator and the weighting. Note that the two panels
are not conditioned on the same graphs: `H_c` averages every graph, counting a graph with no carrier
at `d` as a real zero, while `R_c` averages only the graphs that have support there. At the far
columns that subset is small and `O_{g,c}(d)` approaches one carrier, so `R_c` is both amplified and
conditioned on the longest-diameter graphs; the reporting floor of Section 7 is what keeps those
cells from being read as population estimates.

`unreachable` carriers and internal virtual/hub nodes use explicit columns after the numeric
distance columns and are never folded into `d_max`. Summing all numeric and explicit special
buckets must reconstruct the raw score graph by graph within tolerance.

Measurement is always at unit distance resolution. Presentation is not: on a long-diameter task a
figure would carry more distance columns than it can label, so every distance figure is drawn on a
grouped display axis of a bounded number of columns — unit resolution near the changed node, dyadic
widening in the tail, explicit columns never merged into a numeric range. Grouping is applied to the
per-graph and per-event statistics and the registered estimators are rerun on the grouped axis, so a
displayed value is always one the estimator would produce for that grouping, with its own interval
and its own reporting-floor decision. Because groups have unequal widths, additive quantities are
displayed per unit distance; support-normalized quantities need no such correction. Both reduce to
the ungrouped figure when every group is a single column.

Heatmap cells cannot display useful error bars. Therefore every heatmap is accompanied by a
distance-profile figure obtained by summing `H_c(l,d)` and `R_c(l,d)` over layers, with a 95% band
under the Section 7 hierarchy, plus a machine-readable cell table with corresponding intervals.
With multiple seeds, show seed-level curves and, with at least three seeds, use the registered
hierarchical population interval. Both semantic and structural panels use identical distance
columns and colour-scale policy within a task/model comparison.

This is an accounting view of `S_sem` or `S_str`, not an additional score.

Shortest-path distance is a common graph coordinate for organizing responses to changed node and
pair features; it is not necessarily the distance relation used internally by the model, nor the
number of computational propagation steps taken by a signal. This distinction is particularly
important for the structural intervention: topology-derived node and pair features can encode
global information, and in a dense transformer the changed pair entry `r_{i,s}` is directly
available to a distant receiving node `i`. The curves therefore quantify **SPD-dependence of
sensitivity**, not a uniquely identified message-passing path.

For every discovery-frozen leaning, central-responsive, or inactive family `A`, retain the
family-specific score profile:

```text
C_{g,c,A}(d)
    = mean_s mean_k
      sum_{(l,h) in A} sum_{i: d_g(i,s)=d}
      ||q^{lh,c}_{g,s,k,i,:}||_2
```

Plot its exact and support-normalized forms alongside final-state Functional carriage and
Beneficial carriage evaluated on the same channel and distance axis. These coupled profiles show
where family sensitivity is distributed and how it corresponds to final-state reach. They do not
establish that the family implements that reach, create a new score, or imply that a head score
must predict task loss directly. Distance-resolved causal analysis is not part of the core
methodology.

As a routing-locality comparison, also report clean attention mass by pristine sender--receiver
distance. Normalize within graph and head before averaging:

```text
A_A(d)
    = mean_g mean_{(l,h) in A}
      [ sum_{i,j: d_g(i,j)=d} A^{lh}_{ij}
        / sum_{i,j} A^{lh}_{ij} ]
```

This attention profile is descriptive. It tests whether functionally local score/carriage profiles
coexist with spatially broad raw routing, but it is not a substitute for transport sensitivity.
Virtual/hub-node attention uses its own explicit bucket and is never assigned an artificial
shortest-path distance.

## 5. Functional carriage

Head scores read transport at every layer's per-head `wV`. Carriage reads the intervention response
at final node states `h^L_i`, immediately before graph pooling/readout.

For output direction `t`:

```text
g^out_{t,i} = d z_t / d h^L_i                 evaluated at clean input

Delta h^c_{g,s,k,i}
    = h^L_{g,clean,i} - h^L_{g,c(s,k),i}

q^c_{g,s,k,i,t}
    = g^out_{g,t,i} dot Delta h^c_{g,s,k,i}
```

The sole production functional-carriage field is:

```text
F_sens^c[g,i,s]
    = mean_{k=1..K} ||q^c_{g,s,k,i,:}||_2
```

Use the scientific name **Functional carriage** and the implementation symbol `F_sens`. `F` may
be retained only as a compatibility alias pointing to the same values.

`F_sens` is label-free. It answers whether a typical valid donor-swap at source `s` produces
output-relevant transport at carrier `i`. It measures response magnitude and reach, not whether the
response improves task performance.

The eventwise magnitude must be computed before donor averaging. The following is not the core
functional field:

```text
||mean_k q^c_{g,s,k,i,:}||_2
```

## 6. Beneficial carriage

Beneficial carriage asks whether the clean signal carried from a source reduces or increases the
registered task loss.

Let `ell(ŷ,y)` be the task's scalar per-graph loss, reduced across all `T` outputs according to the
task registry. For every donor event, define a straight path in final-state space from the
intervened endpoint back to the clean endpoint:

```text
H^c_{g,s,k}(alpha)
    = H^L_{g,c(s,k)} + alpha (H^L_{g,clean} - H^L_{g,c(s,k)})
      for alpha in [0,1]
```

First define the donor-wise signed loss-path allocation:

```text
a_loss^c[g,i,s,k]
    = integral_0^1
      < d ell(rho(H^c_{g,s,k}(alpha)), y_g) / d h_i,
        Delta h^c_{g,s,k,i} >
      d alpha
```

By the fundamental theorem for line integrals:

```text
sum_i a_loss^c[g,i,s,k]
    = ell(ŷ_g,clean, y_g) - ell(ŷ_g,c(s,k), y_g)
```

The production **Beneficial carriage** field takes the negative of that allocation:

```text
B^c[g,i,s]
    = - mean_{k=1..K} a_loss^c[g,i,s,k]
```

This uses the same eventwise principle as `F_sens`: perform the nonlinear donor-specific operation
first, then average donors. Do not integrate a path from a donor-averaged state, average predictions
before applying the loss, or replace signed donor outcomes with their absolute/RMS magnitude.

For every source:

```text
sum_i B^c[g,i,s]
    = mean_k [
        ell(ŷ_g,c(s,k), y_g)
        - ell(ŷ_g,clean, y_g)
      ]
```

The sign convention is:

```text
B > 0    beneficial: the clean signal reduced loss
B < 0    adverse:    the clean signal increased loss
B ~ 0    little net loss allocation on this registered path and intervention distribution
```

This positive-is-beneficial convention is normative. Older artifacts with the opposite sign must
be negated and relabelled before comparison; silent mixing of sign conventions is prohibited.

Near-zero `B` does **not** by itself establish dispensability or absence of useful computation.
It can reflect a genuinely negligible task-loss effect, donor averaging of beneficial and adverse
events, cancellation across carriers or outputs, local flatness of the loss, redundancy elsewhere
in the network, or the chosen final-state path allocation. Interpret its scale against the
donor-event loss change and its interval. For example, `B=10^-5` is small compared with a
`10^-1` clean-to-event loss change, but it is not meaningfully "positive" unless its uncertainty
and the relevant aggregate loss mass exclude zero.

Only the donor-wise sum over carriers is fixed by the endpoint loss difference. Individual carrier
allocations depend on the declared straight path through final-state space and are not unique
causal effects or necessity estimates. Beneficial carriage localizes the signed task-loss
allocation under that path; necessity requires the separate ablation and patching programme in
Section 9.

For add or mean pooling, the path can be integrated through the smaller pooling-to-readout head and
projected exactly back to carriers. Adaptive Gauss-Kronrod quadrature is the production numerical
method because it resolves L1/ReLU kinks and exposes both carrier-refinement and completeness error.
No clipping, ratio rescaling, or forced completeness correction is permitted.

A capped path retains its best estimate only under predeclared per-path error limits. A
capped/unconverged fraction or donor-averaged completeness residual above the recorded tolerance is
a recorded audit failure for the whole run under the Section 10 policy.

## 7. Distance profiles and population aggregation

Distances are shortest-path distances on the pristine graph:

```text
d = d_g(i,s)
```

The default adaptive bins are:

```text
{0}, {1}, {2}, {3}, {4-7}, {8-15}, {16-31}, ...
```

Per-bin functional and beneficial profiles use a graph-balanced two-stage estimator:

1. compute the mean of eligible `(carrier, source)` pairs within each graph and bin;
2. combine graph-level values with the predeclared central tendency; and
3. obtain 95% intervals with the registered nested bootstrap below.

The repository default is a 20% trimmed mean across contributing graph values. For `G`
contributing graphs, sort the graph values and remove `floor(0.2 G)` observations from each tail;
if that count is zero, the estimator reduces to the ordinary mean. Median and ordinary mean are
sensitivity analyses, not silent replacements. A bin is reported only when it contains at least
10 contributing graphs and 50 eligible `(carrier, source)` pairs in total; otherwise it is marked
not estimable.

This floor governs score distance columns exactly as it governs carriage distance bins. The
distance axis is frozen from the all-pairs distances of the whole discovery split, so its far
columns can be populated by very few graphs; every distance figure therefore suppresses columns
below the floor, and each run records the per-column supporting-graph and eligible-pair counts
alongside a support figure. Cached measurements retain every column: the floor is a reporting rule,
not a measurement change.

Within a resample, a distance column with no carrier opportunity is *missing*, not zero. Support-
normalized quantities average only over the graphs that supply opportunity, and a bootstrap
replicate in which no resampled graph supplies any is excluded from that cell's percentiles rather
than entered as zero, which would bias the band toward zero and contradict the point estimate. Each
interval records how many replicates were estimable per cell.

Every plotted Functional carriage or Beneficial carriage summary includes uncertainty. Unless a
section explicitly fixes a level, the same hierarchy also governs raw scores, conditional
contrasts, and causal endpoints. The full inferential hierarchy is:

```text
trained seed -> graph -> eligible source -> donor event
```

Resample each stochastic level that the estimand treats as a population. Within one trained seed,
resample `graph -> source -> donor`; for a population result with at least three seeds, resample
`seed -> graph -> source -> donor`. If all eligible sources or all registered donors were
exhaustively enumerated and are treated as fixed, omit that level and state so. A graph-only
bootstrap is allowed only when the event manifest is explicitly frozen; label its interval as
conditional on those sampled sources and donors. Seed-level estimates must remain visible, and a
within-seed interval must never be presented as training-seed uncertainty.

Every interval uses exactly 2,000 percentile-bootstrap replicates and the central 2.5% and 97.5%
quantiles. The bootstrap RNG seed is fixed in the task registration and recorded in every artifact.
BCa, studentized, and asymptotic substitutions are not production alternatives.

For signed beneficial carriage, also report additive loss mass:

```text
S_B(b)    = mean_g sum_{(i,s) in bin b} B[g,i,s]
B_far(r)  = mean_g sum_{d_g(i,s) > r} B[g,i,s]
```

Mean `B` answers the typical pair question; `S_B` and `B_far` answer how much signed task-loss mass
is carried in a region. They must not be conflated.

At minimum, saved artifacts must retain:

```text
graph_id, carrier i, source s, clean distance,
F_sens, B,
channel, event-manifest fingerprint
```

Donor-level loss changes, matching metadata, and integration diagnostics must remain recoverable
from the event or audit tables.

Paper-facing panel and legend names are simply:

```text
Functional carriage
Beneficial carriage
```

The channel is identified separately as Semantic donor-swap or Structural donor-swap. Internal
symbols such as `F_sens` and `B` belong in methods text and artifact schemas, not as competing
metric names in figure titles.

## 8. Raw scores versus derived head coordinates

`S_sem` and `S_str` are the scientific measurements. Downstream coordinates may summarize them,
but never redefine them.

Using the within-model normalized scores fixed in Section 4.2:

```text
J     = 0.5 (s_sem + s_str)
D_rel = (s_sem - s_str) / (s_sem + s_str + eps)
```

`J` is formally named **Joint sensitivity**. It measures the mean normalized sensitivity to the
two core donor-swap channels. A large `J` may arise from one dominant channel; “Joint” names the
combined two-channel sensitivity summary, not a claim of balanced or conjunctive computation.
`D_rel` is formally named **Selectivity**. It measures relative channel balance:

```text
D_rel -> +1    predominantly semantic
D_rel =  0     balanced
D_rel -> -1    predominantly structural
```

This coordinate is protocol-relative. In particular:

```text
D_rel = 0
    iff S_sem / bar(S_sem) = S_str / bar(S_str)
```

It does not mean that the two raw sensitivities are equal, and `D_rel` is not an intrinsic
fraction of a head's semantic versus structural computation. Its sign, magnitude, and ranking are
defined relative to the registered semantic and structural interventions, donor pools, matching
rules, doses, and the within-model reference population of heads. Cross-model comparisons of
absolute `D_rel` values are interpretable only under the same registered protocol and with both
raw-score planes shown.

The usual empirical interpretation is therefore **relative leaning**, not an absolute specialist
type. A head can be more semantic-leaning than its peers even if no active head has strongly
positive `D_rel`, and analogously for structural leaning in a shifted distribution. The term
specialist is reserved for a frozen leaning family that subsequently shows the predicted
held-out family-by-channel causal interaction in Section 9.

`eps` is a registered numerical stabilizer only; it must be negligible relative to an active
head's denominator and may not be used to manufacture stable selectivity for inactive heads.

The second mandatory head scatter has one point per head and uses exactly:

```text
x-axis: Selectivity D_rel (structural <- 0 -> semantic)
y-axis: Joint sensitivity J
```

Paper-facing axis labels must be:

```text
Selectivity $D_{rel}$  (structural $\leftarrow$ 0 $\rightarrow$ semantic)
Joint sensitivity $J$
```

These strings are likewise the repository-wide figure convention.

Use a fixed horizontal activity/reliability threshold for interpreting selectivity. Heads below
that `J` threshold remain visible in grey but receive no semantic/structural family label.
The threshold is frozen from discovery data and reported with sensitivity analysis. Transform each
nested-bootstrap replicate through the complete normalization and `D_rel`/`J` formulas to obtain
horizontal and vertical 95% intervals; do not propagate only marginal raw-score standard errors.

Any scientific use of `D_rel` must:

- show the two raw scores alongside it;
- impose a predeclared activity/reliability floor;
- use the Section 4.2 within-model normalization without outcome-tuned rescaling;
- report pooled and within-layer results;
- test sensitivity to donor dose and matching; and
- validate role with held-out channel-specific restoration/injection patches and their aligned
  rescue/induction effects rather than ordinary ablation alone.

Axis-wise normalization is acceptable for visualization, but it must not be used to claim that raw
semantic and structural amplitudes are equal or directly calibrated.

## 9. Causal validation

Scores identify where output-relevant intervention responses are transported. They do not by
themselves prove that a head is necessary, uniquely responsible, or semantically/structurally
specialized. Causal validation is a held-out programme with three complementary endpoints.

### 9.1 Split discipline and frozen families

Use non-overlapping base-graph partitions:

1. **Discovery split:** compute `S_sem`, `S_str`, `s_sem`, `s_str`, `J`, and `D_rel`; freeze the
   activity floor and all tested heads/families.
2. **Causal donor-event split:** run donor-wise necessity and channel-specific
   restoration/injection patches on unseen semantic and structural donor events.
3. **Clean ablation split:** measure clean-task importance on independent unperturbed graphs.

Base-graph IDs are disjoint across all three splits. The semantic donor pool is also disjoint from
every base split. It may be reused across discovery, causal validation, and clean ablation to hold
the intervention distribution fixed, but donor RNG streams, draws, and event manifests are
independent across stages. Structural donors remain nodes of their stage-specific base graph.

When data are scarce, nested cross-fitting may replace one fixed partition, but candidate
selection and endpoint evaluation must remain out of fold. Never select a family and report its
causal effect on the same donor events.

The canonical discovery families are:

```text
semantic-leaning:       active heads in the upper D_rel tail
structural-leaning:     active heads in the lower D_rel tail
central-responsive:     high J, D_rel near the active-population median
inactive:               low J
```

These are relative, rank-based families: neither leaning family requires an absolute positive or
negative `D_rel` value. “Central” is likewise relative and means neither tail; it is called
balanced only when its interval lies within a predeclared equivalence region around `D_rel=0`.
Cutoffs or quantiles, family size, layer restrictions, and cumulative ordering are frozen before
causal evaluation. Report bootstrap membership/rank stability, and do not choose tail sizes by
optimizing a held-out endpoint. A leaning family is called
**specialized** only if its predicted family-by-channel interaction is confirmed on the causal
split. Otherwise report relative leaning without evidence of causal specialization; it is valid
for a model to contain no confirmed specialist family.

### 9.2 Ablation: clean necessity and donor-wise role

Ablation zeros the full routed head output `o^{lh}=wV^{lh}` at its actual downstream site. For a
head or frozen family `A`, measure on clean graphs:

```text
A_pred = ||z_clean - z_ablate(A)||_2
A_loss = ell(ŷ_ablate(A),y) - ell(ŷ_clean,y)
```

Report prediction movement, registered metric change, and loss change with nested 95% intervals.
The required head-level validation relates **Joint sensitivity `J`** to ablation impact using rank
correlation and a layer-adjusted regression. A positive relationship supports the claim that
jointly sensitive heads are functionally consequential, but does not validate channel identity.

`D_rel` must be related to a **channel contrast**, not to unsigned ablation impact. If the task has
independently registered semantic- and structural-role endpoints, correlate `D_rel` with:

```text
A_role = A_semantic_role - A_structural_role
```

The task-general channel-specific ablation endpoint is **donor-wise necessity**. For each held-out
channel event, rerun both its clean and intervened members with the same head/family ablated:

```text
Delta z           = z_clean - z_event
Delta z_ablate(A) = z_clean,ablate(A) - z_event,ablate(A)

N_{c,g,s,k}(A)
    = <Delta z - Delta z_ablate(A), Delta z>
      / (||Delta z||_2 + eps)

N_c(A)
    = mean_g mean_s mean_k N_{c,g,s,k}(A)
```

Positive `N_c` means ablating `A` removes model movement aligned with the original channel event.
Report its gross counterpart `||Delta z - Delta z_ablate(A)||_2` as well, because a large
orthogonal change is not channel mediation. The gross counterpart follows the same donor--source--
graph aggregation. Donor-wise necessity uses new donor events and the same matched controls as
patching.

The required ablation relationships are therefore:

```text
J      versus clean ablation impact and calibrated total donor-wise necessity
D_rel  versus the calibrated channel-specific necessity contrast
```

If registered task-role endpoints exist, `A_role` is an additional convergent test. Ordinary
unsigned clean-task ablation alone cannot establish semantic versus structural role.

Report Spearman correlation as the distribution-robust primary statistic, a nested 95% interval
under Section 7, the number of heads and seeds, and a layer-adjusted sensitivity analysis. Do not
pool head observations across seeds as though they were independent replicas of an aligned
feature.

### 9.3 Joint family ablation

Jointly ablate the frozen semantic-leaning, structural-leaning, central-responsive, and inactive
families. For every target family include controls matched as closely as possible on:

```text
layer composition, family size, J, clean-output throughput, and ablation dose
```

Required controls are active same-layer central-responsive, low-`J` inactive, and same-layer
random families. Where a task exposes separate semantic and structural role metrics, test the
preregistered family-by-role interaction. Retain cumulative prefix ablations in the frozen family
ranking; promote them to the paper-facing set when needed to distinguish a distributed effect from
one extreme head. All families and prefix lengths are chosen on discovery data, and all endpoint
plots include nested 95% intervals.

### 9.4 Bidirectional activation patching

For a held-out donor event, let:

```text
Delta z = z_clean - z_event
```

Patch the complete selected head/family output tensor at every carrier, at its native layer:

- **Restoration patch:** insert the clean head output into the donor-intervened run.
- **Injection patch:** insert the donor-intervened head output into the clean run.

These are directional tests and are always reported separately. For patch target `A`:

```text
m_restore = z_event+clean(A) - z_event
m_inject  = z_clean+event(A) - z_clean

R_gross = ||m_restore||_2
I_gross = ||m_inject||_2

R_align = <m_restore,  Delta z> / (||Delta z||_2 + eps)
I_align = <m_inject,  -Delta z> / (||Delta z||_2 + eps)
```

Positive `R_align` is **causal rescue**: movement from the intervened output toward the clean
output. Positive `I_align` is **causal induction**: movement from the clean output toward the
intervened output. `R_gross` and `I_gross` are **gross patch responses** and are the primary patch
endpoints for validating the raw magnitude score. They show how much output movement the patched
activation causes without claiming that the movement mediates the original event displacement.
Aligned movement is the mediation endpoint: a large orthogonal gross response is neither
mediation, rescue, nor induction.

Optional recovered fractions divide the same dot-product numerators by
`||Delta z||_2^2 + eps`, but are reported only when the original event effect exceeds a
predeclared noise floor. Fractions outside `[0,1]` are retained because over-restoration and
sign-reversal are scientifically meaningful.

Each patch test includes:

- a same-condition self patch, which should be numerically zero;
- same-layer, `J`- and throughput-matched central-responsive heads/families;
- low-`J` inactive and same-layer random controls;
- wrong-donor or wrong-graph activation patches;
- wrong-channel leaning families; and
- declared no-op donor events.

Patch one frozen family at a time and cumulative prefixes where feasible. Use the actual tensor
consumed downstream; never patch a detached reconstruction.

For event `e` in channel `c`, define the bidirectional gross response:

```text
P_gross[c,e,A] = 0.5 (R_gross[c,e,A] + I_gross[c,e,A])
```

Run the same patch with a mismatched activation under the same architecture, tensor shape, layer,
family size, donor tier, and intervention-dose constraints. Prefer a different donor event for the
same base graph, source, and channel; use a wrong-graph activation only when carrier alignment and
all matching constraints are explicitly defined. This is the nonspecific patch control. The
matched-control-adjusted gross patch response is:

```text
G_c(A)
    = mean_g mean_s mean_k [
        P_gross_matched[c,g,s,k,A] - P_gross_mismatch[c,g,s,k,A]
      ]
```

Donors are averaged within source, sources within graph, and graphs equally. `G_c` is a causal
patch-response summary, not a mediation or rescue measure.

The same-condition self patch is a numerical-zero audit; it is not a replacement for the mismatch
control. Same-layer central-responsive families are comparative importance controls and are shown
separately rather than silently subtracted.

An aligned mediation summary `M_align,c(A)` is defined with the same mismatch adjustment and
donor--source--graph aggregation from `0.5 (R_align + I_align)`, only when restoration and
injection have the preregistered concordant directions. Restoration and injection always remain
visible separately. Effect-normalized fractions are robustness summaries only for events above
the registered effect floor.

### 9.5 Primary causal hypotheses and interpretation

The raw scores measure response magnitude, so their direct causal validation target is held-out
gross patch response in the same intervention channel:

```text
S_sem  <-> G_sem
S_str  <-> G_str
```

Semantic and structural causal responses can have different intervention-distribution scales even
in the same registered `z` geometry. Therefore `D_rel` and `J` are never tested against an
uncalibrated semantic-minus-structural causal difference. For each causal endpoint, define a
positive channel reference scale over the same registered unit type:

```text
a_G[c] = reference-population mean of P_gross_matched[c]
a_N[c] = reference-population mean of
         ||Delta z - Delta z_ablate(A)||_2

g_c = G_c        / a_G[c]
n_c = N_c        / a_N[c]
```

Each reference member is aggregated `donor -> source -> graph` under the same graph-balanced rule
as its numerator, then the reference is averaged across all registered real heads for head analyses
or across the frozen same-size, same-layer-composition matched reference families for family
analyses. `a_G` deliberately uses the positive, unadjusted matched gross response even though its
numerator `G_c` is mismatch-adjusted. `a_N` analogously uses the positive gross donor-wise ablation
response. The reference rule is fixed before causal evaluation, never fitted to improve an
association, and both scales are recomputed inside each bootstrap replicate. If a reference scale
does not exceed its registered numerical floor, that calibrated causal contrast is non-estimable.

The derived causal targets are:

```text
J      <-> 0.5 (g_sem + g_str)    and 0.5 (n_sem + n_str)
D_rel  <->       g_sem - g_str    and       n_sem - n_str
```

This calibration makes the causal contrast a relative channel focus under the declared protocol,
matching the interpretation of `D_rel`; it still does not turn either quantity into an intrinsic
fraction of computation. Show the cross-channel raw-score relationships as controls. Clean
ablation remains the direct test of overall importance, not channel identity.

Gross agreement validates that the score locates channel-responsive patch effects. A stronger
causal-role claim additionally requires the predicted aligned rescue/induction direction and a
held-out family-by-channel interaction against matched controls. For semantic-leaning family
`A_sem` and structural-leaning family `A_str`, the canonical gross score-validation interaction is:

```text
[g_sem(A_sem) - g_str(A_sem)]
    - [g_sem(A_str) - g_str(A_str)]
```

For rescue/induction interpretation, apply the same interaction to `M_align,c/a_G[c]`, while
retaining restoration and injection separately. Significance of one family in isolation, or gross
movement without aligned movement, is insufficient evidence of specialization.

Test `J` relationships across all estimable heads. Activity-gate only `D_rel`, family assignment,
and selectivity claims; active-only `J` results are sensitivity analyses. Use Spearman correlation,
pooled and within-layer views, a layer-adjusted model, and within-layer permutation. Uncertainty
follows the complete `seed -> graph -> source -> donor` hierarchy; head observations within a
trained model are not independent training replicates. “Pooled” means pooled across layers within
one trained seed. Compute association coefficients per seed and summarize those coefficients
across seeds rather than concatenating head rows across independently trained models.

Interpret confirmed families with a compact endpoint taxonomy:

```text
necessary and rescuable:        positive donor-wise necessity and causal rescue
redundant/distributed:           causal rescue with weak clean or donor-wise necessity
necessary, not cleanly rescued: positive necessity with weak or unstable rescue
inducible, not necessary:       causal induction with weak necessity
weak or unresolved:             intervals include practically negligible effects
```

These labels are descriptive combinations of separately reported continuous endpoints, not a new
scalar score. Thresholds are frozen before confirmation. Do not collapse sensitivity, clean
necessity, donor-wise necessity, gross patch responses, rescue, and induction into a single
importance number.

### 9.6 Minimum causal figure set

The paper-facing validation is deliberately compact:

1. a same-channel/cross-channel calibration panel relating `S_sem` and `S_str` to held-out gross
   patch response `G_c`;
2. `J` against total gross patch response, donor-wise necessity, and clean ablation, plus `D_rel`
   against the corresponding channel contrasts, with pooled, within-layer, and layer-adjusted
   statistics;
3. channel-by-family gross restoration/injection patch responses, aligned rescue/induction, and
   donor-wise necessity panels for the frozen leaning, central-responsive, inactive, and
   matched-control families; and
4. where family effects are distributed, a cumulative frozen-prefix ablation/patching curve as a
   supplementary diagnostic.

Error bars or confidence bands are mandatory for aggregate points and curves. Matrices are paired
with machine-readable cell intervals and a compact interval companion where needed; colour alone
is not an uncertainty display. Gross endpoints are the headline score-validation view and aligned
endpoints are the causal-direction companion. Additional causal plots are included only when they
resolve a preregistered ambiguity; redundant or unstable controls remain in audit artifacts rather
than overloading the paper figure set.

### 9.7 Optional conditional specialization

Conditional specialization asks whether a head or frozen family changes its channel leaning or
sensitivity across a predeclared, input-defined context `C`, rather than requiring one global role.
The condition may use graph, source, or donor metadata available before model evaluation, but may
not be selected from the head response or causal outcome being tested.

For each sufficiently supported stratum, rerun the complete graph-balanced raw-score estimator.
To make conditional contrasts comparable, freeze the unconditional discovery-split channel means
as normalization constants rather than renormalizing each stratum:

```text
s_c | C = (S_c | C) / bar(S_c)_discovery

S_sem | C,  S_str | C,  J | C,  D_rel | C

Delta_C S_c    = S_c | C=1 - S_c | C=0
Delta_C J      = J   | C=1 - J   | C=0
Delta_C D_rel  = D_rel | C=1 - D_rel | C=0
```

Condition-specific renormalization may be shown only as a labelled sensitivity analysis because it
removes condition-wide channel shifts and changes the reference population.

Discovery may screen a registered condition library with global false-discovery-rate control.
Freeze the condition, head/family, contrast direction, activity floor, donor tier, and dose, then
confirm on disjoint graphs and donor events. Each stratum must have adequate graph, source, and
donor support and comparable matching/dose distributions. Report nested-bootstrap intervals for
the conditional contrast and bootstrap selection stability. A discovery-only or non-replicating
effect remains exploratory, and any resulting specialist label is explicitly condition-specific.

## 10. Required verification and failure policy

Every production run is fail-reported. Each check below is mandatory and its measured value is
recorded, but a numerical, invariance, or estimability breach does not terminate the run: it is
logged once, accumulated per task/seed, written to `audits.json` (and, for the model checks, to
`canonical_audits.failures` in `model.json`), and reported again in the run summary. Results
carrying a recorded breach are not eligible for headline claims until the breach is resolved or
explicitly argued to be immaterial. A verification run may re-enable fail-closed behaviour with
`strict_audits=True`, which raises `AuditError` on the first failed check.

Three conditions remain fatal because no quantity can be computed or the computed quantity would
not be the declared one: an unregistered or channel-crossing task field, a task/model disagreement
about registration (for example the virtual-node contract or a patch geometry mismatch), and a
stage in which no graph retains a source estimable under both channels.

### Model and gradient checks

- reload the exact checkpoint and reproduce the registered validation/test metric;
- run in evaluation mode and verify batch invariance within tolerance;
- replay the registered score/causal output representation and require every `sigma_t` to be
  finite and strictly positive;
- verify attention normalization on supported attention layers;
- require clean readout gradients to be present, finite, and nonzero in every scored layer; and
- verify that the hooked `wV` is the tensor actually used by the downstream model.

### Intervention checks

- semantic swap changes only the declared semantic payload at the source;
- structural swap changes only source-incident PE/RRWP payload;
- the structural source node field, outgoing pair row, incoming pair column, and self-entry exactly
  equal the declared donor fields after coordinate replacement;
- every non-source-incident structural entry and the donor's own coordinate remain unchanged;
- dense and sparse structural implementations agree entrywise, including duplicate coalescing;
- content, target, topology, bond data, and attention support remain fixed structurally;
- structural fields remain exactly fixed semantically;
- same-content semantic donors and structural self-donors produce numerical zero;
- every production event is nontrivial;
- source/donor matching tiers and degree gaps are recorded;
- a mismatch control drawn outside the frozen degree tier is recorded as an audit failure for that
  stratum, and an event with no admissible control at all is excluded from the causal record
  rather than reported against a self-control that would null its adjustment; and
- unknown task fields that could cross the channel boundary abort the run.

### Estimator checks

- use a within-batch clean baseline for all deltas;
- reconstruct raw head scores from source/event accumulators;
- reconstruct distance-resolved head scores from all distance buckets;
- verify every support-normalized cell was divided within graph before graph averaging;
- record per-column supporting-graph and eligible-pair counts, the fraction of bootstrap replicates
  in which each column had no support, and every column suppressed by the reporting floor;
- verify each integrated donor path against its finite endpoint loss change;
- verify donor-averaged `sum_i B = mean_k(loss_event_k - loss_clean)`;
- verify same-condition activation patches are numerical zero and replay the exact intervened
  tensor at the registered native site;
- verify mismatch patches satisfy the frozen donor-tier, dose, shape, layer, and family-size
  rules; and
- record quadrature residuals, interval counts, capped-path rates, and endpoint replay error.

### Cache checks

Every cache must bind:

```text
methodology version
task and task-adapter version
checkpoint SHA-256
training seed
model geometry
score/causal output representation and sigma vector
graph/source/donor split IDs
event-manifest hash
K and source cap
semantic graph-uniform/node-uniform donor law
structural node-uniform donor law
matching rule, minimum-gap ties, and fallback
raw-score aggregation
score normalization constants and activity floor
distance buckets, support counts, and heatmap aggregation
functional estimand = F_sens
beneficial sign = positive-is-beneficial
beneficial estimator and numerical tolerances
causal split and frozen family manifest
causal mismatch-control manifest and channel reference scales
bootstrap hierarchy, 2,000-replicate percentile rule, RNG seed, and resampled/fixed levels
conditional-specialisation registry, if used
```

Changing any intervention, aggregation order, field boundary, split, or numerical beneficial
estimator invalidates the corresponding cache. Training checkpoints remain read-only and are not
invalidated by analysis-only changes.

## 11. Non-core methods and diagnostics

The following may be retained for audits, historical comparisons, or explicitly labelled
sensitivity analyses. None is part of the main methodology.

| Method or quantity | Status |
|---|---|
| Magnitude after donor averaging | Legacy coherence diagnostic; not a production score. |
| Signed or norm-after-averaging head aggregation | Net output-influence diagnostic; not a raw specialisation score. |
| Semantic node transposition | Historical intervention comparison; not core. |
| Structural/PE transposition | Historical intervention comparison; not core. |
| Full structure-plus-content relabelling | Model invariance check only. |
| Non-isomorphic whole-graph topology donor | Separate topology comparator; never folded into `S_str`. |
| Local edge switch with recomputed RRWP | Separate topology-reach probe; not core carriage. |
| Support-changing structural intervention | Measures wiring plus payload; not the fixed-support structural channel. |
| Attention following/invariance | Routing diagnostics; not raw transport scores. |
| Attention-only specialisation score | Secondary descriptive field; not `S_sem` or `S_str`. |
| Input-space integrated gradients | Baseline-dependent attribution cross-check; not core carriage. |
| Slope, magnitude-share, or signed-share beneficial allocation | Backwards-compatible comparisons; not production `B`. |
| Method-family labels from refinement experiments | Historical selection vocabulary, not names for the final method. |

## 12. Task-registration contract

A new task joins this methodology by registering, not forking:

1. model/checkpoint loader and pinned environment;
2. graph split and donor split;
3. semantic payload adapter plus immutable control fields;
4. complete topology-derived PE/RRWP field list;
5. fixed support/bond field list and sparse duplicate-agreement tolerance;
6. score/causal output representation, positive `sigma` vector, output shape, and scalar per-graph
   task loss;
7. pooling rule and carrier set;
8. eligible source rule and memory cap;
9. semantic graph-uniform/node-uniform and structural node-uniform donor laws, minimum-gap tie
   handling, `K`, and intervention dose;
10. score/carriage graph counts, distance bins, and the 10-graph/50-pair reporting floor;
11. bootstrap RNG seed and which levels are resampled or fixed under the 2,000-replicate percentile
    rule;
12. activity floor, leaning-family quantiles, family sizes, and equivalence region around zero;
13. disjoint discovery, causal-event, and clean-ablation base splits plus the separately disjoint
    reusable semantic donor pool;
14. patch mismatch controls, causal effect floor, causal reference populations, and task-specific
    validation tolerances; and
15. any predeclared conditional-specialisation registry.

The scorer and carriage estimators contain no task-specific feature indices. A task that cannot
state or audit these boundaries is not yet eligible for headline semantic/structural results.

## 13. Minimal reproducibility record

Every paper table or figure must be traceable to:

- repository commit and methodology version;
- task registration and checkpoint hash;
- trained seed or seed set;
- registered output representation and `sigma` vector;
- base/donor/validation split IDs;
- graph IDs and source IDs;
- donor event manifest and RNG seed;
- `K`, source cap, donor probability law, degree gaps, tie counts, and fallback rates;
- raw `S_sem` and `S_str` before normalization;
- normalization means, `s_sem`, `s_str`, `J`, and `D_rel`;
- exact and support-normalized distance matrices;
- Functional carriage (`F_sens`) and positive-is-beneficial signed `B`;
- frozen causal families, controls, ablation, gross patch responses, rescue, and induction
  endpoints;
- causal channel reference scales and matched/mismatch event manifests;
- nested-bootstrap inputs, RNG seed, and all levels treated as fixed or resampled;
- conditional contrasts, selection stability, and confirmation split when used;
- every verification result; and
- any non-core diagnostic clearly labelled as such.

The methodology is correctly implemented only when the intervention used for a channel is
identical across its head score, functional carriage, beneficial carriage, and causal
role-validation event.
