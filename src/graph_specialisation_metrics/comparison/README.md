# Cross-model comparison (dense / k-hop / VNode ZINC GRIT)

Evaluate, carriage-analyse, and specialise **every** registered ZINC GRIT variant, cache all of
it to Drive, and build side-by-side deliverables that can be re-subsetted (drop/include methods)
without any re-compute. This package only *orchestrates and plots* — the science lives in
`carriage/` and `specialisation/`; this is the single place to compare their outputs across
models.

```python
from graph_specialisation_metrics.comparison import run_all, build_figures

run_all()                       # 5 models x {semantic, structural} carriage + scores + figures
build_figures(drop_vnode=True)  # re-draw the deliverables from cache only (no GPU, no re-run)
build_figures(include=["zinc", "zinc_1hop", "zinc_2hop"])
```

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
   (`num_graphs=128, donors=64, beneficial_denom="integrated"`, matching the dissertation's
   structural command).
3. **Specialise** — per-head `S_sem` / `S_str` transport scores via `specialisation.colab.run`
   with `with_ablation=False, with_attention=False` (the deliverables need the scores, not the
   ablation/attention sweeps).

Then `build_figures` reads the cache and writes the deliverables to `comparison_dir`.

## Deliverables (under `.../model_comparison/`)

| file | deliverable |
|------|-------------|
| `fig_spec_scatter_grid.png`          | (ii) side-by-side per-model scatter: structural `S_str` (x) vs semantic `S_sem` (y), layer-coloured, axes divided by each channel's global-mean, shared diagonal. |
| `fig_carriage_smallmult_<intv>.png`  | (iii) rows = {functional, beneficial}, columns = models; **standardised** (reference-fixed) y-axis per row so the scale never moves when methods are dropped. |
| `fig_carriage_overlay_<intv>.png`    | (iv) two panels (functional \| beneficial) overlaying every method for direct comparison. |
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
