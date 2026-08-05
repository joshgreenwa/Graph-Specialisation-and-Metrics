# Final paper figure plan

Status: second visual iteration complete using deterministic synthetic data. The
source notebook remains unchanged.

## Paper sequence

The seven existing plot contents become five full-width paper figures. This
removes repeated legends and gives the section a clear reading order.

| Paper figure | Contents | Size |
|---|---|---:|
| 1 | Raw task scores + joint sensitivity/selectivity | 6.85 x 3.05 in |
| 2 | Three causal-validation panels | 6.85 x 2.52 in |
| 3 | Joint family-ablation tests | 6.85 x 4.25 in |
| 4 | Necessity + rescue matrices | 6.85 x 2.75 in |
| 5 | Four attention examples | 6.85 x 4.75 in |

The narrative is therefore: define specialisation, validate it causally, test
group-level effects, summarise necessity/rescue, then show the mechanism.

## Visual language

- Grey carries structure. Layers use dark, mid, and light grey; seeds use
  circle, square, and triangle.
- Vermillion `#D55E00` means semantic. Blue `#0072B2` means structural.
- Generalists are charcoal; inert controls are light grey. Dash and marker
  differences make the ablation figure independent of colour.
- Grid and reference lines are light neutral grey. No coloured plot backgrounds
  are used; only the generalist region receives a faint grey band.
- Signed matrix effects use a quiet cool-grey to warm-grey diverging scale.
- Cividis is reserved exclusively for continuous attention magnitude.
- Query and source attention roles are neutral and use circle/solid versus
  square/dashed encodings.
- DejaVu Sans remains the Colab-safe font. Essential plot text is at least 7 pt
  at final size.

## Simplification decisions

- One legend per composite figure, always outside the data region.
- No large internal figure titles. Short `(a) Title` headings navigate panels;
  manuscript captions carry interpretation.
- The causal-validation panels show one correlation each: all heads for joint
  sensitivity and reliable heads for the two selectivity tests.
- The ablation figure shows one mean curve and a light observed seed-range band.
  Individual seed traces are omitted at manuscript size.
- Matrix cells show only the signed across-seed mean. Sample count and
  dispersion belong in the caption or supplement.
- Attention headers retain only role and layer/head identity. J and D values
  move to the caption or the existing machine-readable table.
- Only highlighted query/source nodes are numbered. Matrix ticks are reduced to
  1, 8, and 16, with shared outer axis labels.

## Notebook implementation

The implementation should remain a renderer-only change:

1. Preserve trained models, cached analyses, tables, result structures, and all
   numerical calculations.
2. Centralise the palette, dimensions, ordering, axes style, panel labels,
   legends, matrix helper, and export helper.
3. Generate the five composite paper PDFs directly so paired panels genuinely
   share legends and alignment. The seven component plots may remain optional
   diagnostic exports, but the manuscript should use the five composites.
4. Keep the figure-numbering transition explicit in the LaTeX captions and
   cross-references.
5. Bump figure-renderer versions only; do not invalidate experiment or analysis
   caches.
6. Save vector-first PDF plus 300 dpi PNG previews, use fixed canvases, embed
   TrueType fonts, and close each figure after export.

## Caption responsibilities

Keep the artwork simple by putting these details in captions:

- semantic/structural selection-ring meaning in Figure 1;
- reliability and selectivity thresholds and correlation subsets in Figure 2;
- the min-max band definition and `n = 3` in Figure 3;
- across-seed aggregation and uncertainty for Figure 4;
- J and D values and query/source selection rules for Figure 5.

## QA gate

- Render every PDF with Poppler at its final 6.85-inch width.
- Require embedded fonts, no Type 3 fonts, and no clipped or overlapping text.
- Keep axes, text, marks, graph elements, and matrix cells vector. Only the four
  native 16 x 16 attention matrices and small colourbar gradients are raster.
- Confirm Figure 1 contains every positive point, Figure 3 shares row scales,
  Figure 4 annotations match the arrays, and Figure 5 uses one global attention
  normalisation.
- Regenerate twice and compare hashes or rendered-page diffs for determinism.
