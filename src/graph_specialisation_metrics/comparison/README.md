# Cross-model comparison (dense / k-hop / VNode ZINC GRIT)

Evaluate, carriage-analyse, and specialise **every** registered ZINC GRIT variant, cache all of
it to Drive, and build side-by-side deliverables that can be re-subsetted (drop/include methods)
without any re-compute. This package only *orchestrates and plots* — the science lives in
`carriage/` and `specialisation/`; this is the single place to compare their outputs across
models.

```python
from graph_specialisation_metrics.comparison import run_all, build_figures

run_all()                       # cached stages + new held-out family ablations + figures
build_figures(drop_vnode=True)  # re-draw the deliverables from cache only (no GPU, no re-run)
build_figures(include=["zinc", "zinc_1hop", "zinc_2hop"])
```

## QM9 HOMO-LUMO gap suite

`experiments/carriage/notebook_qm9_gap_models_cell.py` runs this same programme for the
parameter-matched dense and 1-hop QM9 gap checkpoints. It passes
`comparison.data.QM9_GAP_TASKS`, pins epochs 294/295, and writes to an isolated
`graph_specialisation_metrics/qm9_gap/` Drive namespace.

Both task registrations replay `GRIT_QM9_gap.apply_qm9_patch` before importing GRIT and use the
training data contract: PyG QM9 target 4 (raw eV), `x=z`, categorical bonds, split
110000/10000/remainder with seed 42, mean pooling and L1/MAE. Their shared dataset cache is
`/content/drive/MyDrive/grit_qm9_gap_data`, represented by the task-level `dataset_dir` override
instead of the usual `<drive_dir>/datasets` convention. Dense-only example generation is selected
by method role rather than the literal ZINC task id, so the complete outlier/gallery deliverables
are produced for QM9 as well.

The five models (`comparison.data.DEFAULT_TASKS`): `zinc` (dense), `zinc_1hop`, `zinc_2hop`,
`zinc_1hop_vnode`, `zinc_2hop_vnode`. The k-hop / VNode three are auto-registered in
`carriage/tasks.py` and replay their exact training patch (`GRIT_khop_ZINC.apply_khop_patch`) so
the checkpoints load. `run_all` pins the recovery checkpoints the runner wrote
(`comparison.run.DEFAULT_CKPTS`); dense/1-hop auto-discover their GraphGym `ckpt/` dirs.

## What it does

`run_all` delegates to the existing orchestrators once per model, **skipping any stage whose
cached artefact already exists** (`force=True` overrides). It refuses to publish a partial
"all-model" overlay by default (`allow_partial=True` is the explicit diagnostic escape hatch):

1. **Evaluate** — val + test metric recomputed from each checkpoint (added to the carriage/spec
   load check; surfaced in every cached summary's `meta`).
2. **Carriage** — functional/beneficial **semantic** and **structural** carriage via
   `carriage.colab.run(intervention=...)`, integrated beneficial estimator by default
   (`num_graphs=128, donors=64, beneficial_denom="integrated", integrated_atol=1e-4`, matching
   the dissertation's structural command).
3. **Specialise** — per-head `S_sem` / `S_str` transport scores via `specialisation.colab.run`
   with `with_ablation=False, with_attention=False` (the deliverables need the scores, not the
   ablation/attention sweeps).
4. **Channel-split causal ablation** — enabled by default at 128 graphs, 32 sources/graph and 16
   donors/source; set `with_channel_ablation=False` explicitly for a lighter score/carriage-only
   diagnostic run.
5. **Factorial family ablation** — enabled by default and separately cached. It reads the existing
   score matrices, estimates clean pre-head throughput on validation graphs, then computes only
   the new simultaneous family-ablation forwards. Existing carriage and score stages are not
   repeated under `force=False`.
6. **Raw channel-outlier ablation (beta)** — enabled by default and separately cached. It takes
   the top six heads independently by raw `S_sem` and raw `S_str` in each model and measures nested
   and individual held-out validation-loss effects. Controls are disjoint heads selected by layer
   first and nearest clean `||wV||` throughput second; if six targets exhaust a layer, the nearest
   available layer is used and its offset is cached. `J` is deliberately *not* matched: this asks
   whether the visibly exceptional heads are important in total, not whether selectivity adds
   importance beyond activity. Reverse score order diagnoses cancellation/redundancy and a
   layer-nearest random band is secondary. When the family-stage graph contract matches, graph
   IDs, clean outputs, labels, and throughput are reused. Dense semantic heads also receive four
   head-specific examples: Method-A `S_sem` is estimated on a fixed pool of 32 validation
   molecules with at most 18 nodes, and each head's top four are selected before attention is
   inspected. Each key head gets its own four-row figure containing the indexed molecular
   topology, attention-key inflow, and raw receiver-by-sender matrix. These are explicitly
   descriptive rather than causal.

Within the expensive carriage and score stages, cumulative hidden snapshots are also written to
each task directory every four completed graphs. Rerunning an identical request after a Colab
disconnect or late verification failure restores the accumulators and RNG state from the last
snapshot. A changed checkpoint, graph sample, or analysis setting invalidates it automatically;
`resume=False` and `checkpoint_every=...` can be passed through `carriage_kwargs` / `spec_kwargs`.

Then `build_figures` reads the cache and writes the deliverables to `comparison_dir`.

## Deliverables (under `.../model_comparison/`)

| file | deliverable |
|------|-------------|
| `fig_spec_scatter_grid.png`          | (ii) side-by-side per-model scatter: structural `S_str` (x) vs semantic `S_sem` (y), layer-coloured, axes divided by each channel's global-mean, shared diagonal. |
| `fig_carriage_smallmult_<intv>.png`  | (iii) rows = {functional, beneficial}, columns = models; **standardised** (reference-fixed) y-axis per row so the scale never moves when methods are dropped. |
| `fig_carriage_overlay_<intv>.png`    | (iv) two panels (functional \| beneficial) overlaying every method for direct comparison. |
| `fig_DJ_family_ablation_curves.png` | (vi) cumulative validation-loss impact for semantic, structural and generalist families, split into high/low `J`; paired graph-bootstrap CIs and a secondary layer-matched random band. |
| `fig_DJ_family_ablation_contrasts.png` | (vi-b) full-family specialist-minus-strength-matched-generalist contrasts for functional movement and loss. |
| `fig_semantic_outlier_ablation.png` | (vii, beta) nested and single-head loss effects for the top six raw-`S_sem` heads versus layer-nearest throughput controls. |
| `fig_structural_outlier_ablation.png` | (vii-b, beta) identical test for the top six raw-`S_str` heads. |
| `molecule_examples_all/fig_semantic_outlier_attention_dense_LxHy.png` (six files) | (vii-c, beta) retained dense raw-`S_sem` outlier examples: per-head rows for the four highest-`S_sem` molecules in a fixed `n<=18` validation pool. |
| `molecule_examples_all/*.png` (six per model) | (viii, beta) separate gallery for the three highest-`D_rel` semantic and three lowest-`D_rel` structural heads in every model. Each head gets its own four highest matching-channel-score molecules from a fixed `n<=18` validation pool. |
| `fig_performance.png` + `performance.json` | (1) val/test bars + table. |
| `figures_manifest.json`, `run_status.json` | which methods were shown, figure paths, per-stage cache status. |

`<intv>` is `semantic` and `structural`. The **cache** (deliverable i) is the existing per-task
`carriage_summary.json` (+ `carriage_pairs.npz`) and `scores_<task>.npz` on Drive; `build_figures`
consumes only those, so figures are re-stylable/subsettable offline.

### Standardised y-axis

The stable-scale requirement is a *reference-fixed* limit: F/B y-limits (and the specialisation
`gsem`/`gstr` norms) are computed from the full reference set (`tasks=`), so drawing a subset via
`include`/`exclude`/`drop_vnode` never rescales the axis. The beneficial panels omit the self-pair
(`d=0`) bin by default (`include_self_B=True` to keep it) because it is ~100x the transport terms
and would dominate the shared symlog scale.

The specialisation scatter and D/J grids use the same `gsem` and `gstr` reference constants and
the same limits in every panel. Thus the larger dense semantic spread is not a panel-rescaling
artefact: absolute score-amplitude differences are deliberately retained. `D_rel` is the bounded
within-head semantic-vs-structural balance; `J` deliberately retains total output-relevant
transport strength. Raw `J` therefore supports cross-model comparison only as *influence
amplitude*, not as an architecture-free selectivity statistic; use `D_rel` for the latter and the
channel-split ablation figure to validate its causal interpretation.

### Factorial family-ablation programme

The score plane is converted into six **disjoint** within-model families:
`{semantic, structural, generalist} × {high J, low J}`. The highest-`D_rel` tail is relatively
semantic, the lowest-`D_rel` tail relatively structural, and the middle-`D_rel` band generalist.
These are within-model rankings, so the structural family need not have negative absolute
`D_rel`. Within each `J` stratum, the three families are
selected as matched triplets for layer, `J`, and clean pre-head `||wV||` throughput. Thus the
high-`J` generalist is the active null and the low-`J` generalist the inactive null; uniformly
random heads appear only as a secondary, exactly layer-count-matched reference band.

Families are selected from cached test-set scores and ablated on independent validation graphs.
The two primary outcomes are label-free output movement and signed task-loss change. The cache
retains per-graph outcomes so all specialist-minus-generalist confidence intervals are paired
graph bootstraps. High-|D|/low-`J` specialists below an activity floor are excluded because their
relative selectivity is ratio-noise prone. A model that still cannot form at least two matched
triplets is recorded as **not estimable** and omitted from this optional figure; that verdict does
not abort or invalidate the complete carriage/specialisation comparison.

## VNode note

A global-VNode model appends one virtual-node row per graph before the transformer layers and
strips it before the add-pooling head. Carriage remains defined over real graph-node carriers at
`h^L`; earlier VNode communication is present in their final states. Per-head specialisation,
however, includes the VNode's intermediate `wV` row because it can have nonzero downstream
readout gradient. The VNode's edge-based attention-routing score is disabled. See
`carriage/khop_env.py` and `specialisation/README.md`.

## Files

| file | role |
|------|------|
| `data.py`  | cache paths, loaders, method metadata, `select_methods` (drop/include) |
| `plots.py` | the deliverable figures (pure numpy/matplotlib, no torch) |
| `run.py`   | `run_all` (compute with cache-skip) + `build_figures` + `performance_table` |
