# Synthetic Experiments

This folder contains controlled graph tasks for studying symbolic and structural
attention behaviour outside ZINC.

## CausalSpecialisationDoubleDissociation

`training/causal_specialisation_double_dissociation_colab.py` is a single-cell,
official-GRIT experiment for causal validation of the production semantic and structural
head scores. A dense GRIT jointly learns marked-source value retrieval and structural
source-distance classification on the same cycle graphs. The shared source marker controls
addressing difficulty, leaving semantic payload versus RRWP relation as the task contrast. It then runs planted-source
scoring, every-head pre-output ablation, and clean-to-corrupt head-output patching.

Paste the complete file into Colab and run it. The default three-seed run mounts Drive,
pins official GRIT, caches checkpoints and analysis tensors, and writes PNG/PDF versions
of the specialisation plane, score-ablation correlations, and necessity-plus-rescue
double-dissociation figure. For an installation/plumbing check, change the final call to:

```python
main(["--fast-dev-run"])
```

Cached phases can be rerun independently with `--phase train`, `--phase analyze`, or
`--phase figures`; the figures-only phase does not reinstall GRIT.

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
