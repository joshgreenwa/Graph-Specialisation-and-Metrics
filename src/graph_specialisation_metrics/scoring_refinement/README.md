# Scoring-metric refinement

This package implements the task-general experiment specified in
`experiments/methodology/SCORING_METRIC_REFINEMENT_TASK.md`. Its public API is:

```python
from graph_specialisation_metrics.scoring_refinement import run

run(
    tasks=["zinc", "qm9_gap_dense"],
    output_dir="/content/drive/MyDrive/graph_specialisation_metrics/"
               "scoring_metric_refinement_dense",
    phase="all",
)
```

A new task requires a `GritTaskSpec` registration, not a scoring fork. Model loading,
dataset construction, checkpoint verification, and whole-row content access come from
`graph_specialisation_metrics.carriage`. The metric modules never contain ZINC- or
QM9-specific feature indices.

## Module boundaries

- `config.py`: immutable scientific settings, fast-development sizes, and disjoint splits.
- `interventions.py`: semantic donor/transposition, mask-frozen PE copy/transposition,
  and matched non-isomorphic topology donors.
- `fields.py`: official GRIT attention, complete edge-enhanced message, and routed `wV`
  collection.
- `scores.py`: EG output projection, probability-mass-weighted following/invariance,
  graph-balanced aggregation, and M1–M6 coordinates.
- `validation.py`: individual-head ablation, restore/inject patching, correlations,
  bootstrap intervals, permutation tests, and method rankings.
- `figures.py`: the raw, derived, agreement, significance, role, and topology atlases.
- `synthetic_validation.py`: task-independent M1/M4/M5/M7 score mapping and the four
  consolidated mixed-task causal-validation figures.
- `cache.py`: atomic, checkpoint- and protocol-fingerprinted caches.
- `runner.py`: phase orchestration only.

This separation is deliberate: the intervention and score functions accept tensors or PyG
`Data` objects and can be tested without loading a checkpoint. The GRIT-specific field
collector is the only component that knows how the official attention layer stores `attn`,
`V_h`, `wE`, and `VeRow`.

## Channels and interventions

Semantic content is the complete node `x` row.

- `semantic_single_donor`: replace one node row with a different real donor row, preferring an
  exact degree match and otherwise the nearest available degree.
- `semantic_transposition`: exchange the complete rows of two structurally matched nodes.

PE is the topology-derived payload supplied to the model while molecular bonds and attention
support remain frozen.

- `pe_single_donor`: copy one matched node's RRWP footprint onto the source. This duplicates
  a role and is labelled near-/off-manifold.
- `pe_transposition`: transpose all node and pair RRWP payloads for two nodes. This is an
  involution.

Topology is a separate comparator. A same-node-count, non-isomorphic real donor is aligned to
the base graph; its molecular edges, bond values, and already recomputed donor RRWP are copied
into base coordinates. Base semantic content and target remain fixed. Exact atom-multiset
matches are preferred; relaxed matches are marked by their tier.

## Score definitions

For layer `l`, head `h`, query `i`, and key `j`, the collector returns post-softmax attention
`A[l,h,i,j]`, complete pre-attention message `m[l,h,i,j,:]`, and
`wV[l,h,i,:] = sum_j A m`.

### Output-projected EG

At the clean input, the runner computes `phi = d y_hat / d wV`. For every event:

```text
q[event,l,h,i,t] = phi[t,l,h,i,:] · (wV_clean - wV_event)[l,h,i,:]
EG[event,l,h]    = sum_i ||q[event,l,h,i,:]||_2
```

Events are averaged within source, sources within graph, and graphs equally.

### Appendix A.3 cosine following/invariance

For a node transposition `pi=(u,v)`, nodes are singleton blocks. At each graph query, the event
pair at sender locations `(u,v)` is compared by cosine similarity with either the clean pair in
its original ordering (invariant) or its reversed ordering (equivariant/following). This is the
graph adaptation of Appendix A.3 in
[arXiv:2511.11579](https://arxiv.org/abs/2511.11579).

Attention compares the two attention masses directly. Transport compares the flattened pair of
realised complete-message contributions `A*m`. Sampled swaps are weighted per query using:

```text
softmax(|A_clean[i,u] - A_clean[i,v]| / temperature)
```

The default temperature is `0.1` and is protocol-fingerprinted. Query rows are subsequently
weighted by clean attention mass on the swapped pair. Attention scores lie in `[0,1]`; transport
scores retain the appendix cosine range `[-1,1]`.

## M1–M6

| Method | Semantic raw axis | PE raw axis | Coordinate status |
|---|---|---|---|
| `M1_DD` | semantic donor EG | PE single-copy EG | scientific |
| `M1_DT` | semantic donor EG | PE transposition EG | scientific/current |
| `M1_TD` | semantic transposition EG | PE single-copy EG | scientific |
| `M1_TT` | semantic transposition EG | PE transposition EG | scientific |
| `M2` | semantic transport-follow | semantic transport-invariant | diagnostic |
| `M3` | PE transport-invariant | PE transport-follow | diagnostic |
| `M4` | semantic attention-follow | semantic attention-invariant | diagnostic |
| `M5` | PE attention-invariant | PE attention-follow | diagnostic |
| `M6` | semantic transport-follow | PE transport-follow | scientific |

All M1 arms share the discovery-split mean references from the current `M1_DT` axes. Therefore
donor/transposition amplitude differences are not normalized away. M2–M6 use fixed
discovery-split axis means. For raw axes `S_sem`, `S_PE`:

```text
D_rel = (S_sem/ref_sem - S_PE/ref_PE) /
        (S_sem/ref_sem + S_PE/ref_PE + eps)
J     = 0.5 * (S_sem/ref_sem + S_PE/ref_PE)
```

M2–M5 coordinates are explicitly diagnostic because invariance is not positive evidence for the
other channel. Topology EG is always reported separately and never folded into semantic/PE
`D_rel`.

Every M1 arm is additionally compared against the role-aligned raw axes of M2–M6. For example,
an M1 structural/PE score is correlated with M2 semantic-transposition invariance, M3
PE-transposition following, M4 semantic-attention invariance, M5 PE-attention following, and the
raw M6 PE-following component. Semantic comparisons use the complementary raw axes.
`m1_cross_method_correlations.csv` contains pooled, within-layer, and layer-centred statistics;
four `m1_cross_method_correlations_m1_*.{png,pdf}` atlases show the underlying head scatters.

## Validation estimands

Significance is evaluated on a disjoint clean split. Each head's full `wV` is zeroed and the
runner records prediction movement, the registered model loss increase, and graph-bootstrap
intervals. `J` is compared with this held-out effect using pooled, within-layer, partial, and
top-k statistics. L1, MSE, and BCE-with-logits models use the shared task-loss adapter.

Role is evaluated on a second disjoint split balanced over semantic donor/transposition and PE
copy/transposition events. For each head:

- restore clean `wV` in the intervened run;
- inject intervened `wV` in the clean run; and
- patch each condition with itself as a numerical sham.

The mediation strength is the mean of restore reduction and injection magnitude. The primary role
target is semantic minus PE mediation; the total target is their mean. Methods are ranked
separately for significance (`J`) and role (`D_rel`).

## Caching and phases

`phase` is one of `scores`, `validation`, `figures`, or `all`. Score work checkpoints after every
graph. Caches include the protocol fingerprint, task, checkpoint SHA-256, and event-manifest hash.
`figures` reads CSV tables only and never imports or loads GRIT.

Protocol `scoring-refinement-v2-appendix-cosine` invalidates v1 score caches. The collector keeps
the exact hooked `wV` tensor on the clean gradient path; missing, non-finite, or layerwise-zero
readout gradients abort rather than being converted into silent zero M1/topology scores.

All donor/partner events belonging to one source are evaluated in a single batched GRIT forward.
The clean fields and each intervention field are collected once and shared by every method that
uses them; M1 factorial arms are table-level combinations, not redundant model runs.

`--resume-config` reuses the scientific settings in the root `protocol.json`. `--force` ignores
compatible caches. Checkpoint and training result directories are read-only; all generated files
live under the requested analysis output root.

The required tables and figures are written beneath each task directory. In addition to the
contracted files, `ablation_head_effects.csv` and `causal_head_effects.csv` retain the underlying
per-head validation measurements.

## Colab

Run:

```python
%run experiments/methodology/colab_scoring_metric_refinement_dense.py
```

For the final controlled comparison on the cached mixed synthetic task:

```python
%run experiments/methodology/colab_scoring_metric_refinement_synthetic.py
```

The synthetic comparison adds `M7`, whose semantic axis is M4 semantic-attention
following and whose structural axis is M5 PE-attention following. It does not
change the dense ZINC/QM9 M1–M6 protocol or invalidate those caches.

The frontend mounts Drive, refreshes this repository, installs it, runs ZINC and QM9 dense models
sequentially, and releases GPU memory between tasks. Use `--fast-dev-run` for an end-to-end smoke
run and `--phase figures --resume-config` to rerender without loading either model.
