# Per-head specialisation scores on GRIT (central methodology)

The productionised version of the specialisation-score methodology
(`experiments/synthetic/specialisation/SPECIALISATION_SCORES.md`), lifted from the spec-lite
`Net` onto the **real GRIT transformer**. A notebook clones the repo and calls `run(...)`;
everything below runs from there.

```python
from graph_specialisation_metrics.specialisation import run
run(
    tasks=["zinc", "zinc_1hop"],
    num_graphs=200,
    donors=8,
    causal_extension=True,   # default
)
```

Set `causal_extension=False` to recover the original score → zero-ablation → attention-grid
pipeline. `legacy_outputs=True` writes those appendix artefacts alongside the causal extension.
The default paper path intentionally promotes only two figures per task.

## Default paper extension: held-out causal evidence

The raw score is a discovery statistic, not itself a causal treatment. The default command now
uses the score graphs only for discovery and chooses confirmation molecules from their exact
complement. It freezes the atom donors, RRWP-role partners, specialised head rankings, and matched
controls before any confirmation measurement.

### 1. Channel-specific causal mediation

Positive channel scores are calibrated on discovery heads,

```
s_h = S_sem,h / mean_h(S_sem,h)        r_h = S_RRWP,h / mean_h(S_RRWP,h)
I_h = s_h + r_h                        q_h = (s_h - r_h) / (I_h + eps)
```

where `I` is overall path importance and `q` is semantic-versus-RRWP-role preference. Preference
ranking is restricted to an importance floor; the semantic and RRWP top-k sets are explicitly
disjoint. On held-out ZINC molecules the runner evaluates every single head and cumulative
`k = 1,2,4,8` groups with layer/importance-matched controls:

- **noising** — insert counterfactual `wV` head states into the clean computation;
- **denoising** — insert clean `wV` head states into the counterfactual computation.

For `delta = pred_cf - pred_clean` and patched displacement `m`, the primary estimator is the
stable, unclipped ratio of sums `beta = sum <m,delta> / sum ||delta||²`. Signed ZINC `delta MAE` is
stored separately. Inference resamples whole molecules and recomputes the paired semantic/RRWP
double dissociation. The output also contains group-minus-summed-single interactions, a continuous
`q × channel` test controlling for importance and layer, and a within-layer permutation p-value.

### 2. Static routing versus effective relational transport

Attention maps are not treated as transport. For every clean GRIT layer the extension reconstructs
the exact edge contribution, including the `VeRow` edge-enhancement term,

```
P_ijh = A_ijh * (V_jh + E_ijh VeRow_h)       wV_ih = sum_j P_ijh,
```

and blocks mechanistic estimates if the reconstruction does not equal the captured `wV`. A
maximum-entropy static-routing null is found on the **same support** with the same receiver and
source attention marginals. It is therefore valid for dense and 1-hop GRIT; on dense support it is
exactly the receiver-repeated column mean.

Effective head transport is read after GRIT's degree scaler and before `O_h`, then projected through
the exact per-head block of `O_h`. Within each molecule it is decomposed as `U = B + R`, where `B`
is receiver-broadcast and `R` is zero-mean receiver-specific transport. Causal confirmation runs
measure:

- support/marginal-matched routing staticisation;
- broadcast-only transport (`R` removed);
- deterministic within-molecule residual permutation (mean, norm and multiset preserved);
- full selected-head zeroing;
- separate content-residual and edge-residual removal.

Hard checks cover pair → `wV`, degree-scaled `U`, per-head `O_h`, broadcast/residual and component
energy identities, static-routing marginals, residual-permutation preservation, and parity with the
existing whole-head ablation. A failed check blocks every downstream mechanism estimate but still
writes the failure diagnostic and headline failure figure, so a real null and an invalid analysis
cannot be confused.

### Headline figures and reproducibility artefacts

The default command writes PNG (300 dpi) and vector PDF versions under `paper/`:

1. `fig_channel_causal_mediation_<task>` — discovery selection plus held-out noising and denoising
   double dissociation. A CI crossing zero is labelled as unresolved rather than hidden.
2. `fig_effective_relational_transport_<task>` — receiver-specific routing (distance from the
   support- and source-marginal-matched static null) versus effective
   projected relational transport, top-k necessity, and the component interventions. Invalid exact
   reconstruction is shown as a failed analysis, never as a mechanism result.

Per-head selection, the frozen intervention bank, every patched prediction, per-graph causal
endpoints, head metrics, checks, and summaries are stored under `<task>/causal_extension/`.

## Task-general — one central methodology, many models

This is the single place to edit the methodology; it runs out to **any** GRIT / 1-hop GRIT model
registered in `carriage/tasks.py`. Nothing here is ZINC-specific:

- **Model geometry** (layers, heads, per-head dim, pooling head) is read from the loaded GraphGym
  cfg, so it adapts to each checkpoint.
- **Output shape** is general: for `T > 1` (multi-target regression like peptides-struct, or
  multilabel classification like peptides-func) the per-head score uses the functional MAGNITUDE
  `sqrt(sum_t (phi_t . Dbar-o)^2)` over the T readout gradients (for `T = 1` this is `|phi . Dbar-o|`),
  and the ablation impact uses the task loss (`l1` / `mse` / BCE) via `carriage.metrics`.
- **Content shape** is general: the semantic swap uses the task's `content_adapter` (whole `x` row
  — ZINC atom type or OGB's 9 atom features); figures/features use the first content column.
- **Scale**: large-graph tasks (peptides, `n` up to ~450) auto-cap the sources scored per graph to a
  memory budget (logged; pass `max_sources` to override) and skip the `O(n²)` attention-routing
  score on big graphs — the transport score is unaffected.

**Add a new GRIT model** = one `GritTaskSpec` entry in `carriage/tasks.py` (config, checkpoint
`drive_dir`, content adapter, an env hook if it patches GRIT source), then `run(tasks=["<name>"])`.
Running two models in one process re-imports the (possibly patched) GRIT clone per task; the
registry is made overwrite-safe so the patched variant replaces the previous one.

```python
run(tasks=["peptides_struct"], max_sources=32)     # large graphs
run(tasks=["peptides_func", "peptides_struct"])     # peptides family (each gets its own clone)
```

## What it measures

Each attention head `(l, h)` gets a **semantic** and a **structural** score, read at the
per-head **transport site**

    o^{lh}_i = batch.wV[i, h, :]              # the message head (l,h) delivers to node i

(`grit/layer/grit_layer.py`; `wV = Σ_j a^{lh}_{i<-j} · V x_j` plus the edge-enhance term). The
per-head readout gradient `φ^{lh}_i = ∂ŷ/∂o^{lh}_i` is a single `autograd.grad` at the clean
input. Both interventions come straight from the carriage vocabulary and feed the identical
estimator (donor/partner-average the transport delta **before** the abs — Jensen at the kink):

- **Semantic** — donor swap (`carriage.content`): overwrite node `j`'s atom type with a real
  donor row from another molecule, structure (RRWP, mask) held fixed.
- **Structural** — node transposition (mask-frozen, `_perturb_mask_frozen`): conjugate the RRWP
  payload (`rrwp`, `rrwp_index`/`rrwp_val`, `deg`, `log_deg`) by `P_(u v)` with a degree-matched
  partner `v` (K draws, marginalised), content held fixed, and **freeze the attention mask** by
  restoring `edge_index`/`edge_attr`. SPECIALISATION_SCORES.md requires the k-hop mask be held
  fixed: the 1-hop encoder derives its support from `edge_index`, so relabelling it would conjugate
  the mask and confound a sparse head's *wiring* reliance with its structural *payload* reliance,
  inflating `S_str`. For the dense model the support is all-pairs, so freezing is a no-op there.

```
F^{lh}[i, s] = | φ^{lh}_i · Δ̄o^{lh}_i(s) |
S_sem(l,h)   = mean_{graph, j}  Σ_i F   under the semantic donor swap
S_str(l,h)   = mean_{graph, u}  Σ_i F   under the structural transposition
```

A complementary **attention-routing** (selection-site) score is reported for the **semantic**
intervention only — a content swap keeps the edge set fixed so the per-edge attention slots stay
aligned across replicas; a structural transposition relabels edges, so the selection score is not
slot-comparable and is omitted (transport is the primary readout on both sides).

The transport delta uses a **within-batch clean baseline** (replica 0 of every forward chunk),
exactly like `carriage.grit_runner`, so a no-op donor / self-transposition gives ~0 and the
batch-context float32 offset cancels.

## Legacy / appendix deliverables (per model, written to Drive)

1. `fig_scatter_<task>.png` — structural (x) vs semantic (y) score per head, coloured by layer;
   axes divided by each channel's global mean so the diagonal is amplitude-normalised-equal
   (above = semantic-leaning, below = structural). `fig_scatter_combined.png` overlays both models.
2. `fig_heatmaps_<task>.png` — `[layer × head]` heatmaps of `S_sem`, `S_str`, the attention score,
   and the layer profile.
3. `fig_attention_<task>.png` / `fig_attention_graph_<task>.png` — the attention maps
   `A[i,j] = a^{lh}_{i<-j}` of the interesting heads (top-semantic / top-structural / top-joint /
   inert) across several molecules, as matrices and as attention-weighted molecular graphs.
4. `fig_ablation_<task>.png` — zero a head's routed value and measure the eval-set error change:
   (a) each target head's rank / z-score / ratio vs the **random-head null** (all `L·H` single-head
   impacts) plus a random-pair null for joint ablation; (b) per-graph impact distributions;
   (c) impact-vs-graph-feature Spearman heatmap; (d) does the structural head matter more on
   ring-ier molecules?
5. `fig_score_impact_<task>.png` — **does the specialisation score predict a head's causal (ablation)
   importance?** Per-head scatter of each channel's score vs its mean ablation impact (coloured by
   layer) with Spearman ρ, plus a bar panel of **partial correlations**: raw ρ, ρ controlling for the
   OTHER channel, and ρ controlling for depth (layer). Because both the score and the impact scale
   with a head's overall output-reach/throughput, the raw ρ is expected to be high and largely
   reflects that shared factor; the partials isolate whether the semantic/structural distinction
   carries causal signal *beyond* that amplitude. (`score_impact_corr` in the saved JSON holds the
   full raw + partial correlations for functional and loss impact.)

## Verification (asserted every run)

- **softmax** — attention into each destination node sums to 1.
- **no-op** — a same-content swap / self-transposition moves ~0 transport (within-batch baseline).
- **full-relabel invariance** — a structure+content relabel is an isomorphism, so the pooled
  prediction is invariant (catches a missed structural channel in the transposition).
- **checkpoint load** — the eval metric is recomputed from the checkpoint and aborts if bad.

## Files

| file | role |
|------|------|
| `model.py` | load a checkpoint (mirrors `carriage.grit_runner`) + per-head capture / ablation hooks |
| `scores.py` | per-head `S_sem` / `S_str` (transport) + `S_attn_sem` (selection); `select_heads` |
| `ablation.py` | causal head ablation vs random-head null, per-graph, feature correlations |
| `attention_viz.py` | per-head attention maps across molecules |
| `figures.py` | the four legacy appendix deliverables + cross-model scatter |
| `causal_mediation.py` | held-out bidirectional channel-specific head mediation |
| `effective_transport.py` | exact static-routing and effective relational-transport decomposition |
| `paper_figures.py` | the two failure-aware headline figures (PNG + PDF) |
| `colab.py` | `run()`: default paper workflow plus explicit legacy reversion; collates on Drive |

Deleting this package leaves the carriage path untouched (it only imports *from* carriage).
