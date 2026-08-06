# Final paper figure plan

Status: third visual iteration complete using deterministic synthetic data. The
source notebook remains unchanged.

## Paper sequence and placement

The plotting code should export nine independent PDFs. Their canvases are
matched so LaTeX can place the related files on a common baseline without
rescaling one panel differently from its neighbours.

| Placement row | Component exports | Size per export |
|---|---|---:|
| Head metrics | `(a)` task-score plane; `(b)` sensitivity/selectivity | 3.315 x 3.12 in |
| Causal validation | `(a)` sensitivity/impact; `(b)` selectivity/ablation; `(c)` selectivity/rescue | 2.175 x 2.36 in |
| Joint ablation | one 2 x 2 full-width figure | 6.85 x 4.25 in |
| Family effects | `(a)` necessity; `(b)` rescue | 3.315 x 2.70 in |
| Attention | one 2 x 2 full-width figure | 6.85 x 4.75 in |

Use a 0.22-inch gap for paired components and a 0.16-inch gap for the causal
row. The narrative remains: define specialisation, validate it causally, test
group-level effects, summarise necessity/rescue, then show the mechanism.

## Visual language

- Layer is the primary identity for head-level points: L1 violet `#6B5B95`, L2
  teal `#2A8C82`, and L3 ochre `#B88720`. Seeds remain circle, square, and
  triangle. A thin white edge prevents overlap from muddying the colours.
- Vermillion `#D55E00` means semantic and blue `#0072B2` means structural.
  They are reserved for selection rings, task labels, and ablation curves.
- Selected heads retain their layer fill inside a white halo and coloured outer
  ring, so layer and semantic/structural identity remain simultaneously legible.
- Generalists are charcoal; inert controls are light grey. Dash and marker
  differences make the ablation figure independent of colour.
- Grid and reference lines are light neutral grey. Only the generalist region
  receives a faint grey band.
- Signed matrix effects use a conventional muted blue-white-red diverging map:
  `#3B75AF`, `#F7F7F5`, `#C84E3A`. The map denotes sign and magnitude only;
  matrix row and column labels remain black.
- Cividis is reserved exclusively for continuous attention magnitude. Query and
  source annotations remain charcoal circle/solid and grey square/dashed.
- DejaVu Sans remains the Colab-safe font. Essential text is at least 7 pt at
  final size.

## Simplification decisions

- The head-metric pair uses one footer band per file: selection rings under the
  task-score plane and the layer/seed key under the sensitivity/selectivity
  panel. This avoids a duplicated six-item legend while keeping the two axes
  identically sized and aligned.
- The causal components omit legends because they are intended to be placed as
  one row; their shared caption states that colour denotes layer and shape seed.
  Panel (a) keeps one all-head correlation; panels (b) and (c) report both the
  all-head correlation and the correlation for heads with `J >= 0.5`.
- Every causal point receives equal visual weight. Reliability is communicated
  through the two reported correlation subsets rather than opacity.
- The ablation figure shows one mean curve and a light observed seed-range band.
  Individual seed traces are omitted at manuscript size.
- Matrix cells show only the signed across-seed mean. Sample count and
  dispersion belong in the caption or supplement.
- Attention headers retain only role and layer/head identity. Only highlighted
  query/source nodes are numbered with compact numerals that remain inside the
  markers; matrix ticks are 1, 8, and 16; only the lower panels retain the
  `Source node` axis label. The redundant `Destination node` text is omitted.

## Notebook implementation

The implementation should remain a renderer-only change:

1. Preserve trained models, cached analyses, tables, result structures, and all
   numerical calculations.
2. Centralise the palette, fixed dimensions, ordering, axes style, panel labels,
   legend constructors, matrix helper, and export helper.
3. Export the nine component PDFs directly with the dimensions above. Do not
   create a composite raster before PDF export.
4. Keep the figure-numbering transition explicit in LaTeX captions and
   cross-references; related files can share one `figure` environment as
   subfigures while remaining separate assets.
5. Bump figure-renderer versions only; do not invalidate experiment or analysis
   caches.
6. Save vector-first PDF plus 300 dpi PNG previews, embed TrueType fonts, and
   emit unavoidable PDF raster gradients at 600 ppi. Use fixed canvases and
   close each figure after export.

## Caption responsibilities

Keep the artwork simple by putting these details in captions:

- the shared layer/seed and semantic/structural selection grammar for the first
  pair;
- reliability threshold, reliable correlation subset, and layer/seed grammar
  for the three causal panels;
- the min-max band definition and `n = 3` for joint ablation;
- across-seed aggregation and uncertainty for necessity/rescue;
- `J` and `D_rel` values and query/source selection rules for attention.

## QA gate

- Render every component PDF with Poppler at its intended placement width.
- Require exact matching page sizes within each pair/triplet, embedded fonts,
  no Type 3 fonts, and no clipped or overlapping text.
- Keep axes, text, marks, graph elements, matrix cells, and the four 16 x 16
  attention matrices vector. Only compact colourbar gradients may rasterise,
  and they must be embedded at 600 ppi.
- Confirm the task-score plane contains every positive point, ablation panels
  share row scales, matrix annotations match their arrays, and attention uses
  one global normalisation.
- Regenerate twice and compare hashes or rendered-page diffs for determinism.
