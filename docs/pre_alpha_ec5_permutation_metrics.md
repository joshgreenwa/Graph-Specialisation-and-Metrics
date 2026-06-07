# Pre-Alpha EC5 Permutation Metrics

This note describes how the GraphBench EC5 runner computed permutation metrics
before alpha re-weighting was added.

## What Was Permuted

For each sampled validation batch, the runner sampled a within-graph node
permutation `perm_pos[b, new_i] = old_i`.

Only the categorical node type channel was permuted:

```text
node_type_perm[b, new_i] = node_type[b, old_i]
```

Topology, degree, SPD bias, RWSE, RRWP, edge inputs, masks, duty, and targets
were left unchanged. For Graphormer, the graph token was fixed and only real
node positions were permuted.

## What Was Compared

The model was run twice:

```text
clean attention:   A(G)
variant attention: A(G with permuted node types)
```

For symbolic/equivariant comparison, the clean attention tensor was transported
on both query and key axes:

```text
A_ref[new_i, new_j] = A_clean[old_i, old_j]
```

With Graphormer, the graph-token row/column stayed at index `0`, and node
indices were shifted by `+1`.

## Scores

For each layer and head, the runner computed cosine similarity over the full
masked attention tensor, flattened across batch, query, and key axes:

```text
positional_score = cos(A_variant, A_clean)
symbolic_score   = cos(A_variant, A_ref)
```

It also computed row-centered variants, subtracting each query row mean before
cosine:

```text
centered_positional_score = cos(row_center(A_variant), row_center(A_clean))
centered_symbolic_score   = cos(row_center(A_variant), row_center(A_ref))
```

The mask came from the clean model support. In practice this was the real-node
mask, plus the graph token for Graphormer.

## Aggregation

Each sampled permutation produced one row per model, phase, batch, layer, head,
and metric. Final summaries were simple unweighted means and standard
deviations across sampled batches and permutations.

There was no per-query aggregation, no moved-mass calculation, and no alpha
weighting. Every sampled permutation contributed equally, even if it barely
moved any attention mass.

## What Was Not Included

The pre-alpha EC5 metrics did not compute:

- structural-feature permutation scores such as `pe_invariance` or
  `pe_equivariance`;
- joint node-type plus structural-feature permutation scores;
- interaction residual metrics;
- entropy or attention residual norm diagnostics;
- query-level metric CSVs.

## What Changed After Alpha Weighting

The current runner now computes paper-aligned alpha-weighted scores. Instead of
flattening all queries and averaging permutations uniformly, it computes
per-query cosine scores for each sampled permutation.

For each graph, layer, head, query, and permutation, it measures moved attention
mass:

```text
moved_mass = 0.5 * L1(A_clean, transported(A_clean))
```

It then weights sampled permutations with:

```text
alpha = softmax(moved_mass / metric_alpha_tau)
```

and aggregates scores as an alpha-weighted per-query average before averaging
queries into graph/head rows. The default is:

```text
--num-metric-perms 16
--metric-alpha-tau 0.1
```

New output files use the `alpha_permutation_*` prefix so old non-alpha CSVs are
not mistaken for current metrics.

## Other Recent Additions

The current runner also adds structural-feature interventions:

```text
pe_invariance      = cos(A_struct_variant, A_clean)
pe_equivariance    = cos(A_struct_variant, transported(A_clean))
joint_equivariance = cos(A_joint_variant, transported(A_clean))
```

Structural permutations transport degree/RWSE on the node axis and adjacency,
SPD edge input, and RRWP on both pair axes, while holding node types fixed.
Joint permutations transport both node types and structural features.

Centered versions are also written, along with:

- `interaction_residual_norm_centered`;
- `joint_equivariance_excess_centered`;
- `entropy_norm`;
- `attention_residual_norm`;
- query-level metric CSVs.

The Colab runner now also performs EC5-trained zero-shot evaluation on EC7 and
EC10 by default, writing `zero_shot_metrics` into `summary.json` plus
zero-shot diagnostic plots. This can be changed with:

```bash
--zero-shot-components 7
--zero-shot-components ""
```
