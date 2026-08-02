# Canonical methodology launcher

[`canonical_methodology_colab.py`](canonical_methodology_colab.py) is the lightweight,
Drive-backed Colab entry point for the public methodology. It clones a selected repository
revision, reuses registered training checkpoints and datasets, and dispatches GRIT and official
Graphormer tasks through
`graph_specialisation_metrics.methodology.colab.run`.

The normative scientific specification is
[`../../src/graph_specialisation_metrics/README.md`](../../src/graph_specialisation_metrics/README.md).
The implementation boundary and output layout are documented in
[`../../src/graph_specialisation_metrics/methodology/README.md`](../../src/graph_specialisation_metrics/methodology/README.md).

Edit only the task list, training seeds, requested phases, checkpoint overrides, run sizes, and the
explicitly preregistered `FAMILIES` thresholds in the launcher. Methodological definitions belong
in the canonical package, not in the Colab cell. `FAMILIES` exposes separate discovery and causal
equivalence regions plus the stability and dual-channel response floors; set them before looking
at causal outcomes.

Canonical runs resume at the individual derived-artifact level. Repository commit IDs are retained
as provenance rather than used as cache-validity keys. If a file genuinely belongs to a different
scientific contract or is unreadable, the runner preserves it under `cache/_stale/` and recomputes
that miss; read-only downstream artifact loaders continue to reject incompatible inputs.

The initial production run is configured for the dense `zinc`, `qm9_gap_dense`,
`peptides_func`, and `peptides_struct` registrations.

## Focused PCQM4Mv2 causal analysis

[`graphormer_pcqm4mv2_causal_colab.ipynb`](graphormer_pcqm4mv2_causal_colab.ipynb)
is the Drive-backed frontend for the four focused Graphormer tests: the frozen specialist map,
restoration/injection with a same-source nearest-dose alternative donor, donor-wise necessity, and
`J` versus clean single-head ablation. Discovery, causal, and clean-ablation splits each contain
128 disjoint molecules. A directional specialist must clear `J >= 0.20` and the relevant
`D_rel = +/-0.10` margin jointly in at least 95% of the registered discovery bootstrap draws;
three `J`-matched semantic/structural pairs are required for the causal group panels.

Set `PHASE = "run"` to populate/resume graph shards, `"all"` to run and render, or `"figures"`
to redraw solely from Drive caches without loading the checkpoint or PCQM4Mv2. The exact endpoint,
control, cache, and figure contracts are recorded in
[`../../docs/graphormer_pcqm4mv2_causal_analysis_plan.md`](../../docs/graphormer_pcqm4mv2_causal_analysis_plan.md).

After the ZINC and QM9 score caches have completed under
`canonical_methodology_v4_zinc_qm9`, run
[`grit_zinc_qm9_figures_colab.ipynb`](grit_zinc_qm9_figures_colab.ipynb) for the focused
cross-task figure suite. It reads each task's `seed_42/cache/scores/raw.pt` artifact
read-only, verifies the matching `model.json` and checkpoint before model-forward
diagnostics, and writes task-separated supplemental caches and figure manifests. The
`TASK_SELECTION` Colab control runs `zinc`, `qm9`, or `both`; choosing one task does not
construct, validate, or render the other.
The score/coordinate figures reuse the PCQM presentation, while attention and routed-output
diagnostics attach to native GRIT sites. ZINC atom-type IDs are decoded with the exact source
vocabulary and bond dictionary, so the attention grids contain index-preserving RDKit molecules
rather than generic graph layouts; QM9 is reconstructed from its atomic-number and bond-class
fields. Routed-output PCAs use the same fixed chemical-group labels and colour identities as the
PCQM suite (plus an explicit-hydrogen category for QM9). Attention grids, individual routed-output
PCAs, and selected-head distance score breakdowns mirror the current PCQM publication layout,
typography, simplified titles, axis wording, and legend placement; task identity remains explicit
in the distance-figure subtitle, export names, and provenance. Like the PCQM notebook, each
attention figure uses four configurable molecule rows. ZINC and QM9 each expose independent
ordered graph-index lists for semantic, structural, and generalist attention figures. The notebook
caches the stable five-row union once, then slices the configured four-row family view for every
head in that family; unchanged defaults therefore continue to hit the existing shared cache.
Attention colour is stable by head family:
semantic specialists use orange, structural specialists use blue, and high-$J$ generalists use
purple. Each figure applies its family colour map consistently to the attention-weighted molecule,
node-conditioned matrix, and colour bar; this is render-only metadata and does not invalidate the
cached attention tensors. The notebook installs RDKit explicitly, writes every model-forward
diagnostic to its exact supplemental cache before rendering, and constructs the GRIT runtime only
if one of those artifacts is missing. Runtime reconstruction selects the immutable task/seed
`protocol.json` whose scientific fingerprint is bound to the canonical score cache; the shared
root record is accepted only as an exact-matching fallback, since finalisation or another run may
have rewritten it. Once populated, styling-only reruns do not rebuild RRWP,
reload the checkpoint, or execute model forwards. Each named semantic specialist, structural
specialist, and high-`J` generalist is displayed as an attention grid, routed-output PCA, and then
its mean clean-attention-mass versus shortest-path-distance companion. The notebook selects five
distinct heads in each family (semantic, structural, and high-`J` generalist), and emits no distance
curves for heads outside those 15 identified examples. Optional first-, second-, and final-layer
all-head routed-output PCA grids share one additional contract-cached sweep. GRIT's non-additive
relation conditioning is reported as such; its node-only versus relation-conditioned raw-logit
figure is not labelled as Graphormer dot-versus-bias. The same cached diagnostic sweep records
normalised clean-attention entropy and relates it separately to relative selectivity and joint
sensitivity. For ZINC, the notebook additionally reproduces the PCQM selected-head transport
response analysis for the registered heads `(L1,H2)`, `(L1,H7)`, `(L4,H7)`, `(L6,H0)`,
`(L6,H3)`, `(L7,H6)`, `(L9,H1)`, and `(L8,H4)`. Its upper row shows clean attention mass by
query-key shortest-path distance, while its lower row gives the exact additive semantic and
structural canonical-score contributions by source-carrier distance. The lower curves use the
same within-model channel normalisation as `D_rel` and `J`, reconstruct the selected heads'
normalised scores, and carry 95% registered nested-bootstrap intervals. The analysis is computed
from the canonical score cache's event sufficient statistics, written once to a versioned
supplemental cache for all eight heads, and rendered as two four-head paper-size pages without a
model forward. Individual PNG/PDF/JSON figure bundles remain
under each task's figure directory. The notebook also assembles ordered, multi-page PDFs in the
shared Drive `pdf_sections` directory: `zinc_*.pdf` and `qm9_*.pdf` files for main scores, selected
distance curves, semantic specialists, structural specialists, high-`J` generalists,
mechanism/logit diagnostics, and layer-PCA overviews. Specialist section PDFs keep each attention
grid, PCA, and companion SPD curve together in display order. Individual PNGs are lossless
600-DPI exports. PDFs keep typography, axes, curves, and annotations as vectors, render dense
heatmap/scatter layers at 1200 DPI and RDKit molecule line art at 600 DPI, and are merged into
section PDFs without recompression.

For the public PCQM model, set `TASKS = ("graphormer_pcqm4mv2",)` and leave `CHECKPOINTS`
empty. The registered `clefourrier/graphormer-base-pcqm4mv2@refs/pr/4` checkpoint is loaded
with its scalar head. Use `TASK_TRAIN_SEEDS = {"graphormer_pcqm4mv2": (0,)}` so its cache label
does not inherit GRIT training seeds. `TASK_OVERRIDES` can relocate the PCQM and Hugging Face
caches or enforce offline loading. The same front end accepts local Graphormer checkpoints once a
matching task/dataset registration (for example ZINC) is added.
