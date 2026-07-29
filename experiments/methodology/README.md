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

After the ZINC and QM9 score caches have completed under
`canonical_methodology_v4_zinc_qm9`, run
[`grit_zinc_qm9_figures_colab.ipynb`](grit_zinc_qm9_figures_colab.ipynb) for the focused
cross-task figure suite. It reads each task's `seed_42/cache/scores/raw.pt` artifact
read-only, verifies the matching `model.json` and checkpoint before model-forward
diagnostics, and writes task-separated supplemental caches and figure manifests.
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
its mean clean-attention-mass versus shortest-path-distance companion. Every other active head
with negative `D_rel` also gets the SPD plot from the canonical score cache afterward, preserving
complete structural coverage without duplicating named companions. GRIT's non-additive relation
conditioning is reported as such; its node-only versus relation-conditioned raw-logit figure is
not labelled as Graphormer dot-versus-bias. Individual PNG/PDF/JSON figure bundles remain under
each task's figure directory. The notebook also assembles ordered, multi-page PDFs in the shared
Drive `pdf_sections` directory: `zinc_*.pdf` and `qm9_*.pdf` files for main scores, all distance
curves, semantic specialists, structural specialists, high-`J` generalists, and mechanism/logit
diagnostics. Specialist section PDFs keep each attention grid, PCA, and companion SPD curve
together in display order.

For the public PCQM model, set `TASKS = ("graphormer_pcqm4mv2",)` and leave `CHECKPOINTS`
empty. The registered `clefourrier/graphormer-base-pcqm4mv2@refs/pr/4` checkpoint is loaded
with its scalar head. Use `TASK_TRAIN_SEEDS = {"graphormer_pcqm4mv2": (0,)}` so its cache label
does not inherit GRIT training seeds. `TASK_OVERRIDES` can relocate the PCQM and Hugging Face
caches or enforce offline loading. The same front end accepts local Graphormer checkpoints once a
matching task/dataset registration (for example ZINC) is added.
