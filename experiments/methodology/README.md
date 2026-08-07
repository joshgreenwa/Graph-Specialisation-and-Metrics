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

## ZINC checkpoint trajectory

The seed-0 dense/1-hop training trajectory has a dedicated scores-only workflow. Put
`zinc_dense_1hop_seed0_trajectory.tar` and its `.sha256` companion directly in Drive at
`multi_seed_models/multiple_checkpoints_zinc/`, then open these two notebooks on separate Colab
GPUs and run all cells:

- [`zinc_dense_checkpoint_trajectory_colab.ipynb`](zinc_dense_checkpoint_trajectory_colab.ipynb)
- [`zinc_1hop_checkpoint_trajectory_colab.ipynb`](zinc_1hop_checkpoint_trajectory_colab.ipynb)

Each notebook defaults to `MODE="run"` and processes epochs 10, 100, 250, 500, 1000, and 1990
sequentially. Every epoch has an isolated canonical scores cache under
`score_trajectory_outputs/<architecture>/epoch_<epoch>/`; rerunning freshly validates and skips a
complete epoch or resumes its atomic graph shards. No carriage is computed. After all six epochs,
the notebook validates the shared split/geometry contract and writes per-head violin plots, a long
score table, a summary table, and a provenance manifest beneath
`score_trajectory_plots/<architecture>/`. `MODE="plot"` rebuilds those outputs without loading any
model, and `MODE="status"` provides a lightweight cache inventory.

The checked-in launcher is configured to complete `scores` and `carriage` for all six QM9 gap
controls: 1-hop, 1-hop with local-only RRWP, 2-hop, 1-hop+VNode, 2-hop+VNode, and dense. It uses
eight graphs per runtime batch with automatic CUDA OOM backoff and then validates both atomic
48-graph consolidated caches for every task. Existing compatible caches and graph shards resume;
the execution batch size does not alter their scientific contract.

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
PCQM suite (plus an explicit-hydrogen category for QM9), and every title names the dataset and
dense GRIT+RRWP model. The notebook installs RDKit explicitly, writes every model-forward
diagnostic to its exact supplemental cache before rendering, and constructs the GRIT runtime only
if one of those artifacts is missing. Once populated, styling-only reruns do not rebuild RRWP,
reload the checkpoint, or execute model forwards. Each named semantic specialist, structural
specialist, and high-`J` generalist is displayed as an attention grid, routed-output PCA, and then
its mean clean-attention-mass versus shortest-path-distance companion. The notebook selects five
distinct heads in each family (semantic, structural, and high-`J` generalist), and emits no distance
curves for heads outside those 15 identified examples. Optional first-, second-, and final-layer
all-head routed-output PCA grids share one additional contract-cached sweep. GRIT's non-additive
relation conditioning is reported as such; its node-only versus relation-conditioned raw-logit
figure is not labelled as Graphormer dot-versus-bias. Individual PNG/PDF/JSON figure bundles remain
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
