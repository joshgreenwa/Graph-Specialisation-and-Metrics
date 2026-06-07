# Global Permutation Attention Metrics

This document is the implementation contract for graph-transformer attention
specialisation metrics used in the synthetic teacher-head runs and the ZINC
analysis. Future agents should treat this as the source of truth.

The method adapts the positional/symbolic attention framework of Urrutia et al.,
*Decoupling Positional and Symbolic Attention Behavior in Transformers*, ICLR
2026, especially the block-permutation and mass-weighted scoring construction in
Appendix B.3: https://openreview.net/pdf?id=V38yAoqddQ.

The graph setting differs in one central way: our permutations are global
within-graph input permutations followed by real model forward passes. We do not
score local key swaps as the primary metric.

## Tensor Contract

For graph \(G\), layer \(\ell\), head \(h\), query node \(i\), and key node
\(j\), let

\[
Z^{\ell h}_{ij}(G)
\]

be the pre-softmax attention logit and let

\[
M_{ij}(G)\in\{0,1\}
\]

be the valid query-key mask. Dense full attention has \(M_{ij}=1\) for all real
nodes. Sparse, biased, or edge-index attention should use the actual valid
attention support exposed by the model.

The canonical metric tensor is post-softmax attention:

\[
A^{\ell h}_{ij}(G)
=
\operatorname{softmax}_{j:M_{ij}=1} Z^{\ell h}_{ij}(G),
\]

with invalid entries filled with zero. All core metrics use these post-softmax
attention rows and raw cosine similarity. The alpha weights are also computed
from post-softmax attention mass.

Optional diagnostic spaces may be implemented, but must not replace the default:

- `logits`: compare masked pre-softmax logit rows;
- `value_output`: compare per-head transported value outputs
  \(o_i=\sum_j A_{ij}V_j\), if the question is about information flow rather
  than attention selection.

## Permutation Convention

For each graph in a batch, sample a within-graph node permutation
\(\pi\in S_n\). In code, prefer a tensor `perm_pos[b, new_i] = old_i`. This is
the convention used by the existing synthetic and ZINC scripts.

For any pair tensor \(T\in\mathbb{R}^{B\times H\times N\times N}\), define the
transported clean reference:

\[
(P_\pi T)_{ij}=T_{\pi(i),\pi(j)}.
\]

With the `perm_pos[b, new_i] = old_i` convention, this is implemented by
gathering rows with `perm_pos` and then gathering columns with `perm_pos`.

For a mask, use the same transport:

\[
(P_\pi M)_{ij}=M_{\pi(i),\pi(j)}.
\]

## Actual Forward Passes

Every score must compare a real forward pass on a permuted graph input against a
clean reference. Transporting the clean tensor alone is only the reference, not
the measurement.

For node-content permutations:

\[
G^{x,\pi}
=
\text{same structure and structural features as }G,\quad
x_i(G^{x,\pi})=x_{\pi(i)}(G).
\]

Run the model on \(G^{x,\pi}\) and collect \(A(G^{x,\pi})\). This is the
symbolic-vs-positional plane.

For structural-feature permutations:

\[
G^{p,\pi}
=
\text{same node content and topology as }G,\quad
p_i(G^{p,\pi})=p_{\pi(i)}(G).
\]

Run the model on \(G^{p,\pi}\) and collect \(A(G^{p,\pi})\). This is the
PE-invariance-vs-PE-equivariance plane.

Examples of structural channels:

- node-level RWSE, LapPE, RRWP summaries, degree, or other structural features;
- pairwise SPD, RRWP, shortest-path, or attention-bias tensors transported on
  both axes when the channel is pairwise;
- other structural feature bundles, if they can be permuted cleanly while
  holding the intended non-target channels fixed.

For full graph relabeling sanity checks, relabel content, topology, edge
features, masks, and all structural features together, then compare to the
transported clean reference. This sanity check is separate from the core
specialisation metrics.

## Comparison Masks

For invariant comparisons, compare \(A(G^\pi)\) with \(A(G)\) on pairs valid in
both tensors:

\[
M_{\mathrm{stable}} = M(G)\cap M(G^\pi).
\]

For equivariant comparisons, compare \(A(G^\pi)\) with \(P_\pi A(G)\) on pairs
valid in both the permuted run and the transported clean support:

\[
M_{\mathrm{follow}} = M(G^\pi)\cap P_\pi M(G).
\]

When the intervention keeps support fixed, these masks are usually identical.
Still compute them explicitly, because sparse architectures and structural
counterfactuals can change support.

## Raw Cosine Score

For two masked vectors \(u_i,v_i\) for the same query row, define

\[
\operatorname{cos}(u_i,v_i)
=
\frac{\langle u_i,v_i\rangle}
{\|u_i\|\|v_i\|+\epsilon}.
\]

For the uncentred attention metrics, clip numerical output to \([0,1]\). Because
attention rows are nonnegative, valid cosines should already lie in \([0,1]\).

If a row has zero norm after masking, return score \(0\) for that query-row
comparison.

## Alpha Weighting

Sample a set of permutations \(\Pi\). Not all sampled permutations are equally
informative. For each graph \(G\), layer \(\ell\), head \(h\), query \(i\), and
permutation \(\pi\), compute how much clean attention probability mass the
permutation moves:

\[
m_{\pi,i}^{\ell h}(G)
=
\frac{1}{2}
\left\|
A_i^{\ell h}(G)-P_\pi A_i^{\ell h}(G)
\right\|_1.
\]

Use the union of the clean and transported-clean valid supports when computing
this moved mass. In fixed-support dense attention this is just the full row.

Convert moved masses into per-query, per-head permutation weights:

\[
\alpha_{\pi,i}^{\ell h}(G)
=
\frac{\exp(m_{\pi,i}^{\ell h}(G)/\tau)}
{\sum_{\sigma\in\Pi}\exp(m_{\sigma,i}^{\ell h}(G)/\tau)}.
\]

\(\tau>0\) is the alpha temperature. Lower \(\tau\) concentrates score mass on
the permutations that most disturb the clean attention pattern; higher \(\tau\)
approaches uniform averaging.

Important implementation details:

- alpha is per graph, layer, head, query, and permutation;
- alpha is computed from clean post-softmax attention only;
- the same alpha weights are reused for invariant, equivariant, and centred
  scores for that permutation plane;
- do not average permutation scores uniformly unless explicitly running an
  ablation.

## Node-Content Metrics

For node-content permutations \(G^{x,\pi}\), positional means invariant under
node content:

\[
S_{\mathrm{pos},i}^{\ell h}(G)
=
\sum_{\pi\in\Pi}
\alpha_{\pi,i}^{\ell h}(G)\,
\operatorname{cos}
\left(
A_i^{\ell h}(G^{x,\pi}),
A_i^{\ell h}(G)
\right).
\]

Symbolic means equivariant with node content:

\[
S_{\mathrm{sym},i}^{\ell h}(G)
=
\sum_{\pi\in\Pi}
\alpha_{\pi,i}^{\ell h}(G)\,
\operatorname{cos}
\left(
A_i^{\ell h}(G^{x,\pi}),
(P_\pi A)^{\ell h}_i(G)
\right).
\]

Metric names:

- `positional_score`
- `symbolic_score`

Interpretation:

- high `positional_score`: attention pattern is stable when node content is
  globally permuted, so the head is content-invariant;
- high `symbolic_score`: attention pattern follows globally permuted node
  content, so the head is content-following;
- high values on both uncentred scores can occur for diffuse attention, which is
  why entropy and centred scores are required.

## Structural-Feature Metrics

For structural-feature permutations \(G^{p,\pi}\), PE-invariance means invariant
under the structural channel:

\[
S_{\mathrm{PEinv},i}^{\ell h}(G)
=
\sum_{\pi\in\Pi}
\alpha_{\pi,i}^{\ell h}(G)\,
\operatorname{cos}
\left(
A_i^{\ell h}(G^{p,\pi}),
A_i^{\ell h}(G)
\right).
\]

PE-equivariance means the head follows the permuted structural channel:

\[
S_{\mathrm{PEeq},i}^{\ell h}(G)
=
\sum_{\pi\in\Pi}
\alpha_{\pi,i}^{\ell h}(G)\,
\operatorname{cos}
\left(
A_i^{\ell h}(G^{p,\pi}),
(P_\pi A)^{\ell h}_i(G)
\right).
\]

Metric names:

- `pe_invariance`
- `pe_equivariance`

Interpretation:

- high `pe_invariance`: the head ignores the permuted structural feature
  channel;
- high `pe_equivariance`: the head's attention pattern is controlled by that
  structural feature channel.

## Centered Scores

Centred scores are a required companion to the uncentred attention scores. They
use the same permutations, actual forward passes, transported references,
comparison masks, moved-mass alpha weights, and aggregation order. The only
difference is the row representation being compared.

The uniform background is removed at the attention-row level before computing
cosine similarity. Do not compute centred scores by subtracting a scalar
baseline from the final symbolic/structural scores.

For a valid attention row \(A_i\), define the uniform row on its valid support:

\[
U_{ij}
=
\begin{cases}
1/|M_i|, & M_{ij}=1,\\
0, & M_{ij}=0.
\end{cases}
\]

The centred attention residual is

\[
\widetilde{A}_i=A_i-U_i.
\]

The centred cosine is raw cosine on residual rows:

\[
\operatorname{ccos}(A_i,B_i)
=
\frac{
\langle \widetilde{A}_i,\widetilde{B}_i\rangle
}{
\|\widetilde{A}_i\|\|\widetilde{B}_i\|+\epsilon
}.
\]

If either residual norm is numerically zero, return \(0\) for that query-row
comparison. This makes uniform or near-uniform attention contribute near zero
rather than receiving an artificial positive background.

Centred scores are not shifted into \([0,1]\). Their natural range is
\([-1,1]\). In plots, use a diverging colormap with zero as the meaningful
background.

The centred node-content metrics are:

\[
S_{\mathrm{pos,ctr},i}^{\ell h}(G)
=
\sum_{\pi\in\Pi}
\alpha_{\pi,i}^{\ell h}(G)\,
\operatorname{ccos}
\left(
A_i^{\ell h}(G^{x,\pi}),
A_i^{\ell h}(G)
\right),
\]

\[
S_{\mathrm{sym,ctr},i}^{\ell h}(G)
=
\sum_{\pi\in\Pi}
\alpha_{\pi,i}^{\ell h}(G)\,
\operatorname{ccos}
\left(
A_i^{\ell h}(G^{x,\pi}),
(P_\pi A)^{\ell h}_i(G)
\right).
\]

The centred structural-feature metrics are:

\[
S_{\mathrm{PEinv,ctr},i}^{\ell h}(G)
=
\sum_{\pi\in\Pi}
\alpha_{\pi,i}^{\ell h}(G)\,
\operatorname{ccos}
\left(
A_i^{\ell h}(G^{p,\pi}),
A_i^{\ell h}(G)
\right),
\]

\[
S_{\mathrm{PEeq,ctr},i}^{\ell h}(G)
=
\sum_{\pi\in\Pi}
\alpha_{\pi,i}^{\ell h}(G)\,
\operatorname{ccos}
\left(
A_i^{\ell h}(G^{p,\pi}),
(P_\pi A)^{\ell h}_i(G)
\right).
\]

Metric names:

- `positional_score_centered`
- `symbolic_score_centered`
- `pe_invariance_centered`
- `pe_equivariance_centered`

These are not legacy logit-centred scores and are not `0.5 * (cos + 1)`.
They are also not `score - background`; the centering happens before the
cosine, by projecting each attention row away from the all-keys-uniform
direction.

## Aggregation

The required aggregation order is:

1. For every clean graph batch, run the clean model and cache attention tensors,
   masks, and clean post-softmax attention.
2. Sample \(K\) within-graph permutations.
3. For each permutation, construct the appropriate permuted input and run the
   model. Do this separately for `x` permutations and structural-feature
   permutations.
4. For each layer/head/query, compute invariant and equivariant row scores for
   every permutation.
5. Compute moved mass from clean attention and convert to alpha weights per
   graph/layer/head/query.
6. Alpha-weight over permutations, producing per-query scores.
7. Average over valid query nodes inside each graph, producing one row per
   graph/layer/head/metric.
8. Average those rows over graphs/batches/seeds only in downstream summaries.

For a valid-query set \(\mathcal{Q}(G)\), the graph-level head score is

\[
S^{\ell h}(G)
=
\frac{1}{|\mathcal{Q}(G)|}
\sum_{i\in\mathcal{Q}(G)}
S_i^{\ell h}(G).
\]

Dataset-level summaries average \(S^{\ell h}(G)\) over graph rows. Preserve
per-graph rows where possible, because subset-specialisation can be washed out
by dataset means.

## Minimal Pseudocode

```python
clean = forward_collect(batch)
clean_attn, clean_mask = masked_attention(clean[layer])

score_lists = {metric_name: [] for metric_name in metrics}
moved_masses = []

for perm in permutations:
    permuted_batch = make_x_permuted_batch(batch, perm)
    variant = forward_collect(permuted_batch)
    var_attn, var_mask = masked_attention(variant[layer])

    ref_attn = transport_pair_reference(clean_attn, perm)
    ref_mask = transport_pair_reference(clean_mask, perm)

    stable_mask = clean_mask & var_mask
    follow_mask = ref_mask & var_mask

    moved_masses.append(moved_mass(clean_attn, ref_attn, clean_mask | ref_mask))

    score_lists["positional_score"].append(
        cosine_by_query(var_attn, clean_attn, stable_mask)
    )
    score_lists["symbolic_score"].append(
        cosine_by_query(var_attn, ref_attn, follow_mask)
    )

    clean_ctr = row_center_attention(clean_attn, clean_mask)
    var_ctr = row_center_attention(var_attn, var_mask)
    ref_ctr = transport_pair_reference(clean_ctr, perm)

    score_lists["positional_score_centered"].append(
        raw_cosine_by_query(var_ctr, clean_ctr, stable_mask)
    )
    score_lists["symbolic_score_centered"].append(
        raw_cosine_by_query(var_ctr, ref_ctr, follow_mask)
    )

alpha = softmax(stack(moved_masses) / tau, dim="perm")
for metric, scores in score_lists.items():
    per_query = (alpha * stack(scores)).sum(dim="perm")
    per_graph_head = masked_mean(per_query, valid_queries, dim="query")
```

The PE-invariance and PE-equivariance scores follow the same pseudocode, using
structural-feature-permuted batches and metric names `pe_invariance`,
`pe_equivariance`, `pe_invariance_centered`, and `pe_equivariance_centered`.

## Mixed Symbolic-Structural Metrics

Mixed metrics ask whether symbolic and structural channels combine in a way that
is not explained by either channel alone. They require four forward passes for
the same permutation \(\pi\):

- clean: \(G\);
- node-content permuted: \(G^{x,\pi}\);
- structural-feature permuted: \(G^{p,\pi}\);
- jointly permuted: \(G^{x+p,\pi}\).

All mixed metrics below use centred post-softmax attention residuals:

\[
R_0=\widetilde{A}(G),\quad
R_x=\widetilde{A}(G^{x,\pi}),\quad
R_p=\widetilde{A}(G^{p,\pi}),\quad
R_{xp}=\widetilde{A}(G^{x+p,\pi}).
\]

The centred interaction residual is the two-factor finite difference

\[
I_\pi = R_{xp}-R_x-R_p+R_0.
\]

The per-query interaction residual norm is

\[
E_{\mathrm{int},i}^{\ell h}
=
\frac{\|I_{\pi,i}^{\ell h}\|_2}
{\|R_{xp,i}^{\ell h}-R_{0,i}^{\ell h}\|_2
 + \|R_{x,i}^{\ell h}-R_{0,i}^{\ell h}\|_2
 + \|R_{p,i}^{\ell h}-R_{0,i}^{\ell h}\|_2
 + \epsilon}.
\]

This is then alpha-weighted over permutations exactly like the core scores.
The metric name is `interaction_residual_norm_centered`. It lies in \([0,1]\)
up to numerical error. High values mean the joint x+structural perturbation is
not well approximated by adding the single-channel effects.

The centred joint-equivariance score for the joint intervention is

\[
S_{\mathrm{joint},i}^{\ell h}
=
\operatorname{ccos}
\left(
A_i^{\ell h}(G^{x+p,\pi}),
(P_\pi A)^{\ell h}_i(G)
\right).
\]

The joint equivariance excess compares this joint explanation to the best
single-channel centred explanation for the same permutation:

\[
E_{\mathrm{joint},i}^{\ell h}
=
S_{\mathrm{joint},i}^{\ell h}
-
\max\left(
S_{\mathrm{pos},i}^{ctr},
S_{\mathrm{sym},i}^{ctr},
S_{\mathrm{PEinv},i}^{ctr},
S_{\mathrm{PEeq},i}^{ctr}
\right).
\]

This is also alpha-weighted over permutations. The metric name is
`joint_equivariance_excess_centered`. Positive values mean the attention pattern
is better explained when symbolic and structural channels move together than by
any single-channel invariant/equivariant account.

Interpretation:

- high `interaction_residual_norm_centered`: non-additive symbolic-structural
  interaction exists, whether useful or not;
- high `joint_equivariance_excess_centered`: the head has a coherent joint
  symbolic-structural routing pattern;
- high both: strongest evidence for genuinely mixed attention behaviour;
- high interaction but low/negative joint excess: nonlinear disruption rather
  than a clean mixed routing rule.

## Causal Bias And Value-Transport Diagnostics

The variance-ratio diagnostics compare the scale of explicit additive bias
logits against content/QK logits. They should be supplemented by direct
softmax-space counterfactuals. For any layer/head with decomposed logits

\[
Z = C + B,
\]

where \(C\) is the dot-product/content logit and \(B\) is the learned explicit
bias logit, compute

\[
A_{\mathrm{full}} = \operatorname{softmax}(C+B),\quad
A_{\neg B} = \operatorname{softmax}(C),\quad
A_{\neg C} = \operatorname{softmax}(B).
\]

The attention-ablation metrics are query-averaged total variation distances:

\[
\mathrm{bias\_ablation\_attention\_tv}
=
\frac{1}{2}\lVert A_{\mathrm{full}}-A_{\neg B}\rVert_1,
\]

\[
\mathrm{dot\_product\_ablation\_attention\_tv}
=
\frac{1}{2}\lVert A_{\mathrm{full}}-A_{\neg C}\rVert_1.
\]

For architectures without explicit attention bias, set \(B=0\): bias ablation
should be zero, and dot-product ablation compares learned attention to uniform
valid-key attention.

For architectures with structural value transport, also cache counterfactual
per-head value outputs:

\[
O_{\mathrm{full}},\quad
O_{\neg B\_\mathrm{attn}},\quad
O_{\neg V_\mathrm{struct}},\quad
O_{\neg B,\neg V_\mathrm{struct}}.
\]

Then compute:

\[
\mathrm{bias\_value\_transport\_effect}
=
1-\cos(O_{\mathrm{full}},O_{\neg B\_\mathrm{attn}}),
\]

\[
\mathrm{struct\_value\_add\_effect}
=
1-\cos(O_{\mathrm{full}},O_{\neg V_\mathrm{struct}}),
\]

\[
\mathrm{total\_struct\_transport\_effect}
=
1-\cos(O_{\mathrm{full}},O_{\neg B,\neg V_\mathrm{struct}}).
\]

For each counterfactual, also report the relative norm effect

\[
\frac{\lVert O_{\mathrm{full}}-O_{\mathrm{cf}}\rVert_2}
{\lVert O_{\mathrm{full}}\rVert_2+\epsilon}.
\]

These are diagnostic metrics, not replacements for global permutation
specialisation scores. They answer whether explicit bias/value channels
causally shape attention or transported values in the trained forward pass.

## Output Expectations

Per-row metric CSVs should contain at least:

- model or architecture identifier;
- batch and graph index when available;
- layer;
- head;
- metric name;
- score;
- metric space, usually `attention`;
- centred flag where available;
- alpha temperature;
- mean moved mass;
- alpha concentration diagnostics such as `alpha_max_mean` and
  `effective_perms_mean`.

Required visualisations:

- uncentred heatmaps on \([0,1]\);
- centred heatmaps on \([-1,1]\) with a diverging colormap centred at zero;
- x-permutation plane: `positional_score` vs `symbolic_score`;
- centred x-permutation plane:
  `positional_score_centered` vs `symbolic_score_centered`;
- PE-permutation plane: `pe_invariance` vs `pe_equivariance`;
- centred PE-permutation plane:
  `pe_invariance_centered` vs `pe_equivariance_centered`;
- mixed-channel plots for `interaction_residual_norm_centered` and
  `joint_equivariance_excess_centered`;
- per-query versions for synthetic tasks where query-level behaviour matters;
- graph-subset variation summaries for ZINC, because specialised behaviour can
  appear only on subsets of graphs.

## Non-Negotiables

- Use actual forward passes on globally permuted graph inputs.
- Use post-softmax attention rows as the default comparison space.
- Compute alpha weights from moved clean post-softmax attention mass.
- Weight scores per graph/layer/head/query before query and graph averaging.
- Keep uncentred and centred scores separate.
- Do not implement centred scores as shifted signed cosine.
- Do not replace global input permutations with local key swaps for the baseline
  symbolic/positional and PE metrics.
