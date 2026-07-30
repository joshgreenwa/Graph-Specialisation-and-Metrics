# Synthetic Experiments

This folder contains controlled graph tasks for studying symbolic and structural
attention behaviour outside ZINC.

## Finite-intervention range under nonlinear saturation

`analysis/saturation_carriage_colab.py` is a standalone Colab frontend for the focused
carriage validation experiment. It trains four tiny two-path models across five saturation
strengths, then compares:

- the expected range inferred from local Jacobian influence;
- the expected range measured by finite Functional carriage; and
- signed Beneficial carriage under an MAE target.

The far path preserves the same finite donor-swap response while its endpoint Jacobian
saturates. This gives a controlled test of the distinction between infinitesimal influence and
finite intervention. The near path is deliberately task-adverse and the far path task-beneficial,
so Beneficial-carriage signs and completeness are also directly checkable.

In Colab, paste the complete frontend into one cell and run it. Checkpoints, measurement caches,
CSV/JSON results, and the PNG/PDF figure are saved under
`MyDrive/graph_specialisation_metrics/saturation_carriage_v1/`. Change `PHASE = "figures"` to
regenerate the paper figure entirely from cached measurements.

Local equivalent:

```bash
python -m graph_specialisation_metrics.synthetic.saturation_carriage \
  --phase all \
  --output-dir outputs/saturation_carriage_v1 \
  --seeds 0,1,2,3
```

## Learned softmax routing: finite versus local carriage

`analysis/softmax_routing_carriage_colab.py` is the more realistic follow-up. Each graph contains
a categorical query source, a near carrier, and a randomly ordered record for every key at a
controlled far distance. Record payloads vary independently, and the regression target is the
sum of a query-local code and the payload at the uniquely matching record. A tiny compatibility
model must therefore learn key-value routing through an ordinary softmax.

Every semantic donor is another key already represented in the same graph. The hard-routing
teacher identifies the exact carriers that should change, providing an eventwise oracle. The
experiment compares this oracle with canonical finite Functional carriage and with the clean
Jacobian projected onto the same donor direction. Scaling the learned attention logits changes
routing confidence without changing the selected key.

The default run trains four seeds and caches checkpoints, measurements, CSV/JSON tables, and the
PNG/PDF paper figure under
`MyDrive/graph_specialisation_metrics/softmax_routing_carriage_v1/`. The Colab frontend displays
the generated PNG inline as well as saving it. Set `PHASE = "figures"` for model-free regeneration.

Local equivalent:

```bash
python -m graph_specialisation_metrics.synthetic.softmax_routing_carriage \
  --phase all \
  --output-dir outputs/softmax_routing_carriage_v1 \
  --seeds 0,1,2,3
```

## Fixed-N associative recall: canonical analysis

`training/nar_grit_colab.py` trains the three-seed 1-hop, 2-hop, and dense fixed-N GRIT
checkpoints. `analysis/nar_canonical_methodology_colab.py` is the standalone Colab frontend for
the repository's normative donor-swap specialisation, causal-validation, and Functional carriage
methodology.

The analysis frontend separates:

- `--all-ns`, always used for the three-seed accuracy-versus-N figure; and
- `--analysis-ns`, the subset receiving the expensive score, causal, and carriage analysis.

It caches checkpoint-bound raw results under the training run on Drive, regenerates PNG/PDF
figures from those caches, uses at most 14 grouped shortest-path columns, and computes carriage
only for the lowest-validation-loss seed in each support-by-N cell. Beneficial carriage is
intentionally disabled for this experiment.

Local CLI form, after the NAR checkpoints and official GRIT environment are available:

```bash
python -m graph_specialisation_metrics.synthetic.nar_canonical_analysis \
  --phase all \
  --all-ns 4,8,16,32,64,80 \
  --analysis-ns 4,16,64 \
  --seeds 0,1,2
```

## Fixed-N associative recall: methodology-paper extension

`analysis/nar_methodology_extension_colab.py` is a separate, cache-safe frontend for three
NAR-specific paper analyses:

- query-address versus requested-value head specialisation;
- exact in-distribution query/value counterfactual mediation by frozen head families; and
- changes in role separation at retrieval-capacity transitions.

The frontend never writes below the existing
`canonical_nar_analysis_d128/canonical/` directory. It reads the completed N=4,16,64 score
artifacts with an internally validated read-only loader, writes new score-only N=8,32 runs below
`extensions/nar_role_counterfactual_v1/transition_scores/`, and places all derived caches,
tables, and PNG/PDF figures below the same versioned extension directory. Existing extension
caches are also immutable: a contract mismatch fails with instructions to choose a new
`--extension-name`.

The three registered headline figures are:

- `06_head_role_specialisation_N16`: normalized content sensitivity versus normalized address
  sensitivity for every head, layer, and seed (the other transition Ns are saved as `S01_*`
  supplementary figures);
- `07_counterfactual_head_role_validation`: symmetric injection/restoration mediation of exact
  query and target-value counterfactual answers; and
- `08_specialisation_performance_transition`: chance-adjusted accuracy, total-variation role
  separation, and their adjacent-N changes.

Figure regeneration is model-free:

```bash
python -m graph_specialisation_metrics.synthetic.nar_methodology_extension \
  --phase figures
```

The full extension run is:

```bash
python -m graph_specialisation_metrics.synthetic.nar_methodology_extension \
  --phase all \
  --cached-ns 4,16,64 \
  --additional-score-ns 8,32 \
  --transition-ns 4,8,16,32,64 \
  --counterfactual-ns 4,8,16
```

Both score and counterfactual inference start at 48 base graphs per batch. CUDA OOM handling
automatically halves only a failing batch and continues from immutable graph shards. The runtime
controls are independently configurable with `--score-graphs-per-batch` and
`--counterfactual-graphs-per-batch`; changing them does not change the scientific cache contract.
Low host-RAM use is expected because NAR graphs are generated lazily and model activations live on
the accelerator.

## Fixed-N associative recall: publication synthesis

`analysis/nar_methodology_paper_colab.py` is the final, cache-only publication frontend. It keeps
the original canonical analysis and `nar_role_counterfactual_v1` extension immutable, performs no
checkpoint/GRIT inference, and writes a separate
`extensions/nar_methodology_paper_v2/` tree containing:

- `01_core_specialisation_N16`: the mandatory structural-versus-semantic and
  `D_rel`-versus-`J` head landscapes, with discrete layer colours, seed shapes, core-family
  outlines, model accuracy subtitles, and no whiskers over the main scatter;
- `02_core_causal_validation_N16`: same-/cross-channel raw-score calibration, `J` against clean
  ablation/total causal response, `D_rel` against signed channel contrasts, and the frozen
  semantic-family versus structural-family causal interaction against matched controls;
- `03_task_role_and_counterfactual_validation`: canonical-family query/record fingerprints using
  the unconditional core-channel normalisers, plus a correctness-gated, graph-paired,
  control-adjusted counterfactual double dissociation;
- `04_competence_and_causal_grounding`: the full accuracy curve next to causal-grounding summaries
  at the completed canonical `N=4,16,64` cells; and
- role-conditioned raw Functional carriage (`F_sens`) supplements for the
  lowest-validation-loss seed of each model, with the registered reporting floors and estimator.

The pointwise nested score intervals required by the normative README are moved out of the crowded
head scatters into model-specific interval companions below `supplementary/intervals/`. Full
machine-readable head intervals, causal estimates, conditional fingerprints, counterfactual
contrasts, and carriage profiles are saved under `tables/`.

Run locally (with the completed Drive tree mounted at the configured path):

```bash
python -m graph_specialisation_metrics.synthetic.nar_methodology_paper \
  --phase figures \
  --score-ns 4,8,16,32,64 \
  --causal-ns 4,16,64 \
  --counterfactual-ns 4,8,16 \
  --performance-ns 4,8,16,32,64,80
```

`N=80` is intentionally performance-only: no score cache was computed for it. Missing or
provenance-incompatible protected source artifacts fail closed.

### Publication v3: complete capacity and validity analysis

The v3 workflow adds causal validation to the already-computed N=8 and N=32 scores without
rerunning scoring. Run `analysis/nar_causal_transition_colab.py` once. It invokes the official
causal-validation implementation directly against the protected score payloads and writes
checkpoint- and score-hash-bound artifacts only below
`extensions/nar_causal_transition_v1/`. Existing canonical, score-extension, and paper-v2 files
are never overwritten. The frontend is safely resumable: it preflights the requested matrix,
validates every existing causal cell, and computes only missing causal cells. It updates
`completion_manifest.json` in Drive after each completed cell, then marks the manifest complete
only after the full requested matrix has been validated.

If a missing consolidated `validation.pt` is accompanied by protected partial causal shards from
an earlier contract, do not delete or overwrite them. Run only the affected cell with a fresh
`--causal-extension-name` such as `nar_causal_transition_repair_v1`, then pass that name to v3 via
`--causal-overlay-extension-names nar_causal_transition_repair_v1`. V3 searches the primary
namespace first and uses an overlay only where the primary consolidated artifact is absent.

Then run `analysis/nar_methodology_paper_v3_colab.py`. This second frontend is model-free and
writes only to `extensions/nar_methodology_paper_v3/`. Its targeted outputs are:

- complete structural-versus-semantic and `D_rel`-versus-`J` landscapes at every score-cached
  `N=4,8,16,32,64`;
- a compact two-row causal validation figure at N=16, with the unstable family-interaction
  endpoint retained as a diagnostic table rather than a headline result;
- query-versus-record localisation of the canonical families across N, paired with per-cell and
  hierarchically pooled exact-counterfactual double dissociation;
- accuracy plus complete all-N heatmaps for `J` versus clean ablation and `D_rel` versus channel
  contrast;
- a cache-only capacity-retention test asking whether collapse of the absolute semantic and
  structural raw-score magnitudes, and independently held-out gross-patching/necessity response,
  co-transitions with collapse of chance-adjusted retrieval accuracy. The figure uses within-seed
  retention relative to N=4 and an adjacent-N trajectory-cluster bootstrap;
- a focused mechanism-survival figure combining chance-adjusted competence, active-head
  population, interval-reliable `D_rel`, semantic/structural family layer organisation, absolute
  matched-control family causal specificity, and raw Functional-carriage magnitude. A companion
  figure shows semantic and structural Functional carriage jointly across cached N and
  shortest-path distance;
- supplementary sensitivity-concentration, frozen-family overlap, and same-minus-cross-channel
  discriminant-validity analyses, including a diagnostic that separates relative `D_rel`
  selectivity from absolute causal-response magnitude; and
- official raw `F_sens` role-conditioned Functional carriage supplements. Event-normalised
  carriage and the task-specific `R_role` score are not used.

`J` is deliberately not treated as an absolute cross-N response level: its mean over heads is one
by construction in every estimable checkpoint. The cross-N analysis therefore retains the
unnormalised `S_sem` and `S_str` means and the uncalibrated causal reference scales. It reports
fixed-N capacity co-transition across independently trained checkpoints, not strict cross-N OOD
generalisation. Since the registered output-projected response has an N-way output geometry, the
raw-score, gross-patching, and necessity endpoints are retained separately and should be interpreted
as convergent evidence rather than interchangeable measurements.

For mechanism survival, a `D_rel` sign is reliable only when the head is point-active under the
registered activity floor and its registered nested-bootstrap 95% interval excludes zero. Family
stability is population-level: layer occupancy and held-out causal phenotype are compared across
independently trained cells, never head identity. The family causal endpoint is the absolute raw
matched-control double contrast
`(family same - family cross) - (control same - control cross)`, avoiding the unstable calibrated
ratio retained only in the diagnostic table. Carriage remains raw `F_sens`; beneficial and
event-normalised carriage are not computed.

The first full run writes a source-fingerprinted derived cache below
`extensions/nar_methodology_paper_v3/cache/mechanism_survival/`. Once it exists, figure-only
iteration is:

```bash
python -m graph_specialisation_metrics.synthetic.nar_methodology_paper_v3 \
  --render-target mechanism \
  --mechanism-cache-mode require
```

`require` never recomputes the mechanism estimands: it fails if the exact protected score, causal,
carriage, performance, and bootstrap fingerprint is unavailable. Use the default `auto` for the
first run or when intentionally analysing a new source fingerprint.

Command-line equivalents:

```bash
python -m graph_specialisation_metrics.synthetic.nar_causal_transition \
  --causal-ns 8,32 \
  --graphs-per-batch 48

python -m graph_specialisation_metrics.synthetic.nar_methodology_paper_v3 \
  --phase figures \
  --score-ns 4,8,16,32,64 \
  --canonical-causal-ns 4,16,64 \
  --transition-causal-ns 8,32
```

## MarkedTreePath

`training/marked_tree_path_graphgps.py` trains small GraphGPS-style baselines on
random trees with two marked endpoints `S` and `T`. The node target is the
unique `S`-to-`T` path mask.

Default run:

```bash
python experiments/synthetic/training/marked_tree_path_graphgps.py
```

The default sweep trains 1-, 2-, and 3-layer models, evaluates fixed ID and OOD
graph-size splits, stops early when validation performance is perfect, and
writes x-only permutation attention metrics.

Useful variants:

```bash
python experiments/synthetic/training/marked_tree_path_graphgps.py --structural-channel rwse
python experiments/synthetic/training/marked_tree_path_graphgps.py --structural-channel none
python experiments/synthetic/training/marked_tree_path_graphgps.py --fast-dev-run
```

The training file is also compatible with direct Colab notebook use: paste the
whole file into an empty cell and run it. To override defaults in a pasted cell,
edit the `CELL_ARGS` line near the top of the file.

## StructuralSymbolicGraph

`training/structural_symbolic_graphgps.py` trains small self-contained
GraphGPS-style baselines on structural/positional, symbolic/semantic, and mixed
graph tasks. The default model is intentionally small: 2 GPS layers, 2 global
attention heads per layer, hidden dimension 64.

The default dataset style is GraphWorld-inspired and self-contained. Each sample
uses a fresh random graph backbone from ER, small-world,
preferential-attachment, SBM-like, or mixed families. A variable number of
candidate nodes are sampled from the backbone, each with a value leaf. Symbolic
runs put the primary query key on the readout, the primary record key on the
candidate node, and optional extra key leaves on hard partial-match cases.
Structural runs mark an anchor node and a structural target candidate on the
backbone.

The focused labels are:

- `structural`: classify the shortest-path distance shell between the anchor
  and a structurally marked target candidate. The default is a balanced 4-way
  distance-shell task with RWSE features; SPD attention bias can be enabled but
  usually makes this positional task too easy;
- `symbolic`: classify the value attached to the candidate whose key tuple
  matches the query key leaves on the readout node. Defaults mix easier one-key
  retrieval examples with harder partial-match distractors.

The `dual` task predicts both labels from the same graph, which gives the
2-head model pressure to develop different structural and symbolic routing
patterns. The structural head reads the marked target node; the symbolic head
reads the query/readout node. `--dual-symbolic-loss-weight` can be increased if
the semantic retrieval path learns more slowly than the positional shell label.
The older record/motif generator remains available with `--dataset-style
record`.

Default suite, suitable for Colab:

```bash
python experiments/synthetic/training/structural_symbolic_graphgps.py
```

In Google Colab the script mounts Drive and writes outputs to
`/content/drive/MyDrive/graph_specialisation_metrics/structural_symbolic_graphgps/<run-name>/`.
Locally it writes to `experiments/synthetic/results/structural_symbolic_graphgps/<run-name>/`.

Single-task baselines:

```bash
python experiments/synthetic/training/structural_symbolic_graphgps.py --task structural
python experiments/synthetic/training/structural_symbolic_graphgps.py --task symbolic
python experiments/synthetic/training/structural_symbolic_graphgps.py \
  --task structural \
  --graph-family sbm \
  --graph-min-nodes 32 \
  --graph-max-nodes 64
```

Specialisation-encouraged dual run:

```bash
python experiments/synthetic/training/structural_symbolic_graphgps.py \
  --task dual \
  --specialisation-loss-weight 0.01
python experiments/synthetic/training/structural_symbolic_graphgps.py \
  --task dual \
  --dual-symbolic-loss-weight 3.0
```

Outputs are written under
the suite directory and then split into `structural/`, `symbolic/`, and `dual/`
subdirectories. Each task directory includes:

- `summary.csv` and per-depth `train_log.csv`;
- `symbol_permutation_summary.csv`, including baseline and centered
  structural/symbolic head scores;
- `attention_target_summary.csv`, target-attention mass diagnostics;
- `task_example.png`, `training_curves.png`, `head_specialisation_id.png`, and
  `prediction_panel_id.png`.
- suite-level `suite_model_metric_comparison.png`, comparing metrics across the
  separately trained task models.

Fast smoke test:

```bash
python experiments/synthetic/training/structural_symbolic_graphgps.py --fast-dev-run
```

Custom suite example:

```bash
python experiments/synthetic/training/structural_symbolic_graphgps.py --tasks structural symbolic
```

## TeacherStudentStructuralKeys

`training/teacher_student_structural_keys_graphgps.py` runs the newer
teacher-student copy-routing benchmark. The synthetic teacher selects one
candidate source node for a query node, and the target label is the value
attached to that source. The student only sees input-output examples; teacher
source nodes are saved for attention diagnostics.

The source candidate directly carries its copied value as an input token
embedding. In the starter setup, value leaves are retained as blank graph
context/diagnostic nodes rather than carrying value tokens, so the easiest route
is to attend to the selected candidate itself.

The default `one_head_one_layer` run is the fast starter benchmark:

- tasks: `symbolic` and `structural`;
- structural key family: `anchor_distance`;
- feature set: `anchor_dist`;
- depth `1`, one attention head, hidden dimension `48`;
- key vocab size `16` and value vocab size `16`.

The structural teacher uses clipped shortest-path distances to two anchor nodes
as its hidden role key. The student receives those distances only through the PE
channel, while the structural task disables symbolic key tokens. This gives a
small, direct first test for PE-driven structural routing.

The broader `pilot` run is deliberately narrower than the full design matrix:

- tasks: `symbolic`, `structural_local`, and `mixed_local`;
- feature sets: `rwse` and `rwse_stats`;
- depths: `2 3`;
- key vocab size `64` and value vocab size `32`.

The default student uses a tied value decoder with fixed value input codes. This
is deliberate: the experiment is meant to test whether the model learns the
teacher's routing operation, not whether a tiny model can slowly align arbitrary
learned value embeddings with arbitrary classifier rows. The older untied
classifier remains available with `--value-decoder mlp --no-fixed-value-embeddings`.

```bash
python experiments/synthetic/training/teacher_student_structural_keys_graphgps.py
```

In Google Colab this mounts Drive and writes outputs to
`/content/drive/MyDrive/graph_specialisation_metrics/teacher_student_structural_keys/<run-name>/`.
Useful tuning variants:

```bash
python experiments/synthetic/training/teacher_student_structural_keys_graphgps.py \
  --suite-preset one_head_one_layer \
  --run-name one_head_one_layer_v1

python experiments/synthetic/training/teacher_student_structural_keys_graphgps.py \
  --suite-preset single_head_fast \
  --run-name single_head_fast_v1

python experiments/synthetic/training/teacher_student_structural_keys_graphgps.py \
  --dry-run

python experiments/synthetic/training/teacher_student_structural_keys_graphgps.py \
  --suite-preset family_scan

python experiments/synthetic/training/teacher_student_structural_keys_graphgps.py \
  --suite-preset full

python experiments/synthetic/training/teacher_student_structural_keys_graphgps.py \
  --suite-preset custom \
  --tasks symbolic structural mixed \
  --structural-key-families local diffusion \
  --feature-sets rwse rwse_stats \
  --depths 1 2 3

python experiments/synthetic/training/teacher_student_structural_keys_graphgps.py \
  --fast-dev-run \
  --no-save-checkpoints
```

`single_head_fast` remains a slightly broader 1-head preset with depth `2` and
the older local structural key. Use `one_head_one_layer` first when checking
whether the simplest PE-driven structural route is learnable.

Outputs include per-run summaries and training curves, teacher-source attention
metrics, prediction/error type counts, ZINC-style permutation specialisation
metrics, and suite-level comparison plots. Important figures are:

- `suite_accuracy_heatmaps.png`: validation, ID, and OOD accuracy by task,
  structural feature set, and depth;
- `suite_permutation_plane_x.png`: `positional_score` vs `symbolic_score` from
  symbolic-channel input permutations;
- `suite_permutation_plane_pe.png`: `pe_invariance` vs `pe_equivariance` from
  structural-PE input permutations;
- `suite_specialisation_by_depth.png`: best-head x-channel and PE-channel
  permutation scores as depth varies;
- `suite_teacher_source_attention.png`: best teacher-source attention MRR per
  run;
- per-run `head_role_diagnostics.png`: head-level attention to teacher source,
  source value, candidate distractor types, symbolic key leaves, anchors, and
  entropy.

By default each run also saves `model.pt` as the CPU best-validation state dict
and `checkpoint.pt` as a richer reloadable checkpoint containing model
constructor kwargs, run config, role IDs, summary metrics, and the training log.
Both files are written under the Drive-backed run directory in Colab. Pass
`--no-save-checkpoints` only when storage is more important than later analysis.

The default sweep samples fresh training graphs every epoch with 2048 graphs per
epoch, 512 validation graphs, and 1024 ID/OOD test graphs. ID graphs use 24-72
backbone nodes with 6-12 candidates; OOD graphs use 72-120 backbone nodes with
12-20 candidates. Use `--fast-dev-run` only for syntax/debug checks, or
increase `--train-graphs-per-epoch` after the pilot if learning curves are
noisy.

## TeacherHeadCalibration

`training/teacher_head_calibration_graphgps.py` runs a cleaner metric-calibration
suite. The teacher is a hand-coded attention operator over every query row, and
the value labels are copied through that operator. This is intended as a
positive control for the specialization metrics before moving back to less
constrained graph-level tasks.

Default run:

```bash
python experiments/synthetic/training/teacher_head_calibration_graphgps.py
```

Colab/notebook call after pasting the file:

```python
main([
    "--tasks",
    "symbolic_equality",
    "path_predecessor",
    "tree_parent",
    "mirror_node",
    "hop_then_key",
    "previous_same_key",
    "--run-name",
    "teacher_head_zoo_v1",
])
```

Teacher operators:

- `symbolic_equality`: pure symbolic key equality.
- `path_predecessor`: anchored path previous-node routing.
- `tree_parent`: rooted-tree parent routing.
- `mirror_node`: global structural mirror correspondence.
- `hop_then_key`: structural hop routing plus symbolic selection inside the
  routed set.
- `previous_same_key`: nearest previous node with matching key on an anchored
  path.

Outputs include `metric_summary_all_runs.csv`,
`teacher_attention_all_runs.csv`, `suite_permutation_plane_x.png`, and
`suite_permutation_plane_pe.png`.
