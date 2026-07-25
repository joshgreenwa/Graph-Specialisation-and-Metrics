# Per-head specialisation scores on GRIT (central methodology)

> **Normative status:** the final cross-task methodology is now
> [`../README.md`](../README.md). This file remains an implementation and historical reference.
> Its structural-transposition passages describe the previous production path and do not override
> the final structural donor-swap specification.

The production specialisation-score methodology on the **real GRIT transformer**. It originated
from the spec-lite method in `experiments/synthetic/specialisation/SPECIALISATION_SCORES.md`, but
the production estimator is now the graph-balanced **eventwise-gross (EG)** score defined below.
A notebook clones the repo and calls `run(...)`; everything below runs from there.

```python
from graph_specialisation_metrics.specialisation import run
run(tasks=["zinc", "zinc_1hop"], num_graphs=200, donors=8, ablation_graphs=256)
```

## Task-general — one central methodology, many models

This is the single place to edit the methodology; it runs out to **any** GRIT / 1-hop GRIT model
registered in `carriage/tasks.py`. Nothing here is ZINC-specific:

- **Model geometry** (layers, heads, per-head dim, pooling head) is read from the loaded GraphGym
  cfg, so it adapts to each checkpoint.
- **Output shape** is general: for `T > 1` (multi-target regression like peptides-struct, or
  multilabel classification like peptides-func) the per-head score uses the functional MAGNITUDE
  `sqrt(sum_t (phi_t . Delta-o_k)^2)` per event over the T readout gradients (for `T = 1` this is
  `|phi . Delta-o_k|`),
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
per-head readout gradients `φ^{lh}_{t,i} = ∂ŷ_t/∂o^{lh}_i` are computed at the clean
input. Both interventions come straight from the carriage vocabulary and feed the identical EG
estimator: take magnitude for each donor/partner event, then average events.

- **Semantic** — donor swap (`carriage.content`): overwrite node `j`'s atom type with a real
  donor row from another molecule, structure (RRWP, mask) held fixed.
- **Structural** — node transposition (mask-frozen, `_perturb_mask_frozen`): conjugate the RRWP
  payload (`rrwp`, `rrwp_index`/`rrwp_val`, `deg`, `log_deg`) by `P_(u v)` with a degree-matched
  partner `v` (K draws, marginalised), content held fixed, and **freeze the attention mask** by
  restoring `edge_index`/`edge_attr`. SPECIALISATION_SCORES.md requires the k-hop mask be held
  fixed: the 1-hop encoder derives its support from `edge_index`, so relabelling it would conjugate
  the mask and confound a sparse head's *wiring* reliance with its structural *payload* reliance,
  inflating `S_str`. For the dense model the support is all-pairs, so freezing is a no-op there.

This definition is unchanged for a trained **k-hop** model: the k-hop mask is fixed architecture,
while the score measures semantic/structural signal transported through the head on that support.
For a **global-VNode** model, the VNode is included as an internal carrier in `S_sem/S_str` (its
earlier-layer head output can affect later real nodes even though its final state is excluded from
pooling). Attention-routing scores remain disabled for VNode runs because virtual-edge slots are
not a comparable graph-edge support. Score-cache version 3 records graph-balanced EG production
scores; every older CG-default score cache must be recomputed.

```
q_k^{lh}[i,s] = (φ_{t,i}^{lh} · Δo_{k,i}^{lh}(s))_{t=1..T}

S_EG^{lh} = mean_graph mean_source mean_k Σ_i ||q_k^{lh}[i,s]||
S_CG^{lh} = mean_graph mean_source Σ_i ||mean_k q_k^{lh}[i,s]||
```

`S_sem` and `S_str` mean `S_EG` for their respective intervention channels. **EG is the firm
production choice.** It estimates the expected output-relevant transport response to a typical
valid event. Taking magnitude before event averaging prevents legitimate donor/partner responses
in opposite directions from cancelling; summing carrier magnitudes preserves internal signal even
when different carriers cancel at the final output. This matches the purpose of the score—identify
heads that carry channel-specific signal—while held-out patching and ablation test whether that
signal is causally used.

Aggregation is hierarchical and graph-balanced: average events within source, sources within
graph, and graphs equally. `S_sem_CG` and `S_str_CG` remain saved as coherence/legacy diagnostics;
CG/CN/EN are not alternative production defaults. Close EG and CG results mean the event
population is coherent, not that signed events should be averaged first.

The matching primary functional-carriage field is
`F_sens[i,s] = mean_k ||q_k[i,s]||`; coherent carriage
`F_coh[i,s] = ||mean_k q_k[i,s]||` remains a cancellation diagnostic.

### Exact distance decomposition of the production score

For reach analysis, retain carrier distance before EG sums over carriers:

```
S_EG^{lh}(b) = mean_graph mean_source mean_k
               sum_{i: d(i, changed_set_k) in b} ||q_k^{lh}[i,s]||

S_EG^{lh} = sum_b S_EG^{lh}(b)
```

This is an accounting decomposition of the established score, not a new score or an estimator
choice. The event magnitude must remain *inside* the donor average: replacing it with
`||mean_k q_k||` would instead decompose CG, not the production EG score. Semantic distance is to
the swapped source; PE-transposition distance is to the source/partner changed set; topology-donor
distance is to the edited-node set. Disconnected real carriers use an `unreachable` bucket and
VNode transport uses its own `hub` bucket, so summing all buckets reconstructs the full score.
Distance-resolved results explain which heads implement an intervention's reach profile; final-state
`F_sens` and signed beneficial carriage remain the functional-reach and task-usage endpoints.

A complementary **attention-routing** (selection-site) score is reported for the **semantic**
intervention only — a content swap keeps the edge set fixed so the per-edge attention slots stay
aligned across replicas; a structural transposition relabels edges, so the selection score is not
slot-comparable and is omitted (transport is the primary readout on both sides). This secondary
attention score remains coherent/donor-averaged and must not be confused with production EG
transport scores.

The transport delta uses a **within-batch clean baseline** (replica 0 of every forward chunk),
exactly like `carriage.grit_runner`, so a no-op donor / self-transposition gives ~0 and the
batch-context float32 offset cancels.

## The four deliverables (per model, written to Drive)

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

The cross-model comparison adds a separately cached **factorial family ablation** without
re-estimating these scores. It crosses signed selectivity (`semantic`, `structural`, `generalist`)
with joint strength (`high J`, `low J`), matches families for layer and clean pre-head throughput,
and cumulatively zeros their routed `wV` values. High-`J` generalists are the active scientific
null; low-`J` generalists are the inactive null. See `factorial_ablation.py` and
`comparison/README.md`.

It also contains a clearly marked **beta raw-channel-outlier test** in
`semantic_outlier_ablation.py`. This does not alter the score methodology: it consumes cached
`S_sem/S_str`, ablates each channel's six largest raw-score heads on independent validation
graphs, and compares them with disjoint layer-nearest throughput controls. Dense semantic static-
attention examples are descriptive only: for presentation they are chosen per head as the four
highest donor-swap `S_sem` estimates from a fixed small-molecule validation pool, before any
attention is inspected. See `comparison/README.md` for the estimand and cache contract.

The cross-model comparison also maintains a separate beta molecule gallery. In every model it
selects the three highest and three lowest signed `D_rel` heads, then ranks a fixed small-molecule
validation pool by the matching per-graph intervention score (`S_sem` for semantic heads, `S_str`
for structural heads). Thus neither the head examples nor their molecules are chosen from visual
inspection of attention.

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
| `factorial_ablation.py` | cached-score D × J family selection, matched cumulative ablation, cache writer |
| `semantic_outlier_ablation.py` | beta cached-score raw-channel outlier ablations + dense semantic attention cache |
| `attention_viz.py` | per-head attention maps across molecules |
| `figures.py` | the four deliverables + cross-model scatter |
| `colab.py` | `run()`: one-call orchestration for both models; figures collate on Drive |

Deleting this package leaves the carriage path untouched (it only imports *from* carriage).
