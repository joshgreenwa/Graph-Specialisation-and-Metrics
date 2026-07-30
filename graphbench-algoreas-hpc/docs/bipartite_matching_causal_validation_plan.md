# Bipartite-matching causal validation

Status: implementation specification for the locked GraphBench method.

## Scope

This run is deliberately narrow. It analyses the four trained GRIT seeds
(`0,1,2,3`) on the `n=16` bipartite-matching validation split and asks whether
the complete-PE, coherent-output head scores identify heads with a causal role.
Carriage, distance breakdowns, raw-score calibration, top-k curves, and
Taylor-fidelity figures are out of scope.

The structural intervention is fixed-topology complete-PE donor-copy: copy all
model-visible RRWP row, column, and self channels and the donor's degree role,
then derive log-degree. Structural donors are not degree-matched. Semantic
edge donor-swaps are unchanged.

## Frozen head-selection rules

All selection is computed separately within every trained seed from discovery
scores only. Heads are not aligned across seeds and causal outcomes never
influence selection.

1. Require joint sensitivity \(J \ge 0.20\).
2. The continuous analysis includes every active head and relates raw
   \(D_{\mathrm{rel}}\) to semantic-minus-structural restoration, injection,
   and necessity contrasts. Report Spearman correlation and a standardized
   regression coefficient adjusted for \(J\) and layer.
3. For the key categorical test, semantic candidates have
   \(D_{\mathrm{rel}}>+0.10\) and structural candidates have
   \(D_{\mathrm{rel}}<-0.10\). Heads inside the point-estimate band never enter
   this test.
4. Within each direction select at most the six largest
   \(|D_{\mathrm{rel}}|\) candidates, then pair the selected groups one-to-one
   using the minimum-total-absolute-\(J\) assignment.
5. A seed-level strongest-candidate interaction requires at least three pairs.
   If fewer exist, report the categorical test as non-estimable while retaining
   the continuous analysis.
6. Record pair count, mean absolute \(J\) gap, standardized residual \(J\)
   difference, mean layer gap, and exact-layer pair fraction. Layer is a
   balance audit rather than a hard gate.
7. Separately label a candidate as 95%-confirmed only when its entire
   graph-bootstrap interval clears the same directional threshold. The
   confirmed-only population robustness result requires at least eight pairs
   across at least three trained seeds.

This provides two distinct claims: a high-power continuous relationship across
all responsive heads, and a direct causal double dissociation among the
strongest observed directional heads. The confidence label shows how much of
the latter survives the strict uncertainty criterion without making that
criterion a single point of failure.

## Primary outputs

### 1. Candidate and confidence map

Plot \(J\) against \(D_{\mathrm{rel}}\), with 95% \(D_{\mathrm{rel}}\)
intervals, the generalist band, the activity floor, the selected strongest
semantic and structural candidates, other directional candidates, and
inactive heads. Ring 95%-confirmed heads and connect the strongest candidates'
\(J\)-matched pairs. Use one panel per seed.

This figure defines the tested population before any causal result is examined.

### 2. Continuous raw-\(D_{\mathrm{rel}}\) causal validation

For all active heads, plot raw \(D_{\mathrm{rel}}\) against the
semantic-minus-structural contrast in mismatch-adjusted restoration, injection,
and donor-wise necessity fraction. Show the causal-bootstrap interval for the
Spearman correlation and for the standardized \(D_{\mathrm{rel}}\) coefficient
adjusted for \(J\) and layer.

### 3. Strongest-candidate restoration and injection double dissociation

For each retained individual head and donor event:

- restoration starts from the intervened graph and patches the clean head
  output; positive aligned movement is movement back toward the clean output;
- injection starts from the clean graph and patches the intervened head output;
  positive aligned movement is movement toward the intervened output.

Use mismatch-adjusted aligned restoration and injection as primary endpoints.
The matched alternative is the same source with an alternative donor matched
on full input dose. Gross response remains an audit, not the headline endpoint.

For the strongest \(J\)-matched candidate pairs, estimate separately for
restoration and injection:

\[
(\mathrm{SemCand}_{sem}-\mathrm{SemCand}_{str})
-
(\mathrm{StrCand}_{sem}-\mathrm{StrCand}_{str}).
\]

Show the four cells as well as the interaction. A positive interaction supports
preferential semantic mediation by semantic-direction heads and preferential
structural mediation by structural-direction heads. This is the key categorical
target. Repeat it for the 95%-confirmed subset only when that subset is
estimable under the frozen population rule.

### 4. Donor-wise necessity

Remove each head's donor-induced response and measure the signed fraction of
that donor event's output displacement that disappears:

\[
\frac{\langle \Delta_{\mathrm{removed}},\Delta_{\mathrm{event}}\rangle}
{\|\Delta_{\mathrm{event}}\|^2+\epsilon}.
\]

Aggregate donor to source to graph, then across graphs. Report semantic and
structural events for the matched candidate heads. Values near zero are
consistent with redundancy; negative values indicate compensation; values
above one indicate suppressive interactions. Also retain the gross removed
fraction as an audit for orthogonal movement.

### 5. Clean necessity

Plot \(J\) against clean individual-head ablation prediction movement for every
head. Report Spearman correlation and its graph-bootstrap interval within seed;
summarize seed estimates without treating heads as aligned across seeds.
Clean-loss change is a secondary table/audit.

## Estimation and reporting

- Individual heads are the causal intervention unit; candidate groups are
  CPU-side subsets of those cached results, not new joint multi-head patches.
- Donor events are nested in source nodes/edges, sources in graphs, and graphs
  in trained seeds. Aggregate at the graph level before uncertainty estimates.
- Show all four seed estimates. The cross-seed summary treats trained seed as
  the population unit and does not pool all heads as independent observations.
- Selection uses only score estimates and their bootstrap intervals. No causal
  endpoint may influence labels, matching, thresholds, or exclusions.
- Always report candidate, confirmed-head, and matched-pair counts. A missing estimand is an
  explicit result, not a reason to relax the rule.

## Cache and execution contract

GPU workers run `scores,causal` only. They atomically cache:

- coherent semantic and complete-PE structural scores, coordinates, bootstrap
  intervals, candidate/confirmation labels, rankings, and \(J\)-matched pairs;
- every individual-head restoration, injection, mismatch control, donor-wise
  necessity event, and clean-ablation graph result;
- consolidated per-seed score and causal artifacts.

Only the individual heads are patched/ablated on GPU. The previous rank-tail
families, cumulative prefix ladders, and matched family controls are not run:
they are outside these focused deliverables and cannot contribute to the
threshold-defined causal tests. Candidate groups and confidence sensitivities are
CPU-side subsets of the individual-head cache.

The CPU finalizer requires only the score and causal artifacts. Carriage is
optional. All listed figures, tables, classifications, pairings, interactions,
and cross-seed summaries must be reproducible from these caches without loading
a checkpoint or rerunning a GPU intervention.
