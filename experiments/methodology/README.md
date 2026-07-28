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

Edit only the task list, training seeds, requested phases, checkpoint overrides, and run sizes in
the launcher. Methodological definitions belong in the canonical package, not in the Colab cell.

The initial production run is configured for the dense `zinc`, `qm9_gap_dense`,
`peptides_func`, and `peptides_struct` registrations.

For the public PCQM model, set `TASKS = ("graphormer_pcqm4mv2",)` and leave `CHECKPOINTS`
empty. The registered `clefourrier/graphormer-base-pcqm4mv2@refs/pr/4` checkpoint is loaded
with its scalar head. Use `TASK_TRAIN_SEEDS = {"graphormer_pcqm4mv2": (0,)}` so its cache label
does not inherit GRIT training seeds. `TASK_OVERRIDES` can relocate the PCQM and Hugging Face
caches or enforce offline loading. The same front end accepts local Graphormer checkpoints once a
matching task/dataset registration (for example ZINC) is added.

After a completed `graphormer_pcqm4mv2:seed0` score run, open
[`graphormer_pcqm4mv2_figures_colab.ipynb`](graphormer_pcqm4mv2_figures_colab.ipynb)
to produce the focused PCQM figure suite. It validates and reads the canonical score cache
without recomputing the methodology, selects the structural specialist from active-head
`D_rel`, reuses the cached clean attention-distance profile, and separately contract-caches
only selected-head attention, pooled `A@V`, and dot/bias logit diagnostics.
