# Synthetic Experiments

This folder contains controlled graph tasks for studying symbolic and structural
attention behaviour outside ZINC.

## Paper-aligned Neighbor Associative Recall with trained GRIT support

`training/nar_grit_colab.py` launches the centrally maintained implementation in
`src/graph_specialisation_metrics/synthetic/nar_grit_fixed.py`. The task now follows the NAR
classification construction directly. For each fixed neighborhood size `N`, a separate model is
trained on graphs containing exactly `N+3` nodes: `N` key--value neighbors, a central node, an
intermediate node, and a query node. Every key occurs exactly once; values are sampled with
replacement from an `N`-value vocabulary; central and intermediate input features are zero; and
the central node predicts the queried value after exactly two layers. There is no cross-`N`
curriculum, motif gadget, structural auxiliary label, or task marker.

The intended experimental substitution is the architecture. Parameter-matched official GRITs
are trained with 1-hop, 2-hop, or dense attention support at widths 64 and 128 for
`N={4,8,16,32,64}` and three seeds. Key and value embeddings and the `N`-way classifier are
specific to each fixed-`N` checkpoint, as required by the task. Full mechanistic analysis is
restricted to width 128 and `N={4,16,64}` to keep the run tractable. The resulting figures show
the capacity curves, semantic score--ablation/carriage causality with equal-size random-family
controls, and aggregate semantic carriage plus queried-record attention selection.

Relative to the authors' released training protocol, this lightweight version samples fresh
training graphs online, uses 192 validation and 512 held-out graphs, and omits their `N={80,96}`
and width-256 settings. The graph distribution, token construction, learned null-padded key/value
embeddings, fixed-`N` training unit, batch size 64, learning rate `1e-3`, and three repeats match
the reference implementation. The A100 run allows at most 10,000 optimizer steps per checkpoint
and adopts the reference early-stopping rule exactly: stop when validation cross-entropy falls
below `0.001`. Fixed topology and RRWP tensors are cached in memory, so the larger budget is spent
on optimization rather than regenerating identical graph structure.

This benchmark is deliberately semantic-only, so it does not estimate structural scores or
`J/D_rel`; those require a genuinely independent structural factor and belong in the preceding
mixed-task validation experiment. The semantic score still uses the central Method-A routed-`wV`
transport site with within-forward clean/corrupt replicas and donor averaging before magnitude.
Use `--phase train`, `--phase analyze`, and `--phase figures` to separate cached stages;
`--fast-dev-run` is an installation and wiring check only.

### NAR attention faithfulness and transport mechanisms

`analysis/nar_transport_mechanisms_colab.py` is the standalone Colab access point for the
post-training mechanism experiment. It selects the best seed in each support-by-N cell using
validation loss only, supports `--analysis-width 64` or `128`, and never changes the training
fingerprint. The analysis adds target-payload, query-address, distractor-payload and same-answer
address interventions; decomposes routed `wV` exactly into routing and message terms; computes the
contextual semantic/structural and D/J planes; and runs score-selected family ablation plus finite
2x2 routing/message patching.

Expensive graph/head tensors are cached per checkpoint under
`transport_mechanisms_v1/d<width>/metrics/` and causal results under `causal/`. The `figures` phase
reads only those caches, so all tables and PNG/PDF figures can be regenerated without loading GRIT.
Metric and causal graph chunks are also committed atomically as they finish, allowing interrupted
large-N runs to resume at the next incomplete chunk.
The full design and registered checks are in `NAR_TRANSPORT_MECHANISM_PLAN.md`.

## ReachCarriageSpecialisation

`training/reach_carriage_specialisation_colab.py` is a standalone, single-cell official-GRIT
experiment connecting trained attention reach, semantic/structural carriage, and per-head
specialisation. Parameter-matched 1-hop, 2-hop, 3-hop, and dense GRITs are trained on one
colour-matched value-retrieval generator.

The generator uses 24-node degree-4 two-block graphs. Degree-preserving edge switches create a
thin two-edge cut or wide eight-edge cut without changing node count, degree, or edge count.
Rank-1 retrieval sweeps query--source distance 1--6; the oversquashing experiment fixes distance
three (reachable by all three-layer models) and sweeps 1, 2, 4, or 6 simultaneous cross-cut
retrievals. This separates finite receptive-field failure from bottleneck load.

The Colab caches every checkpoint and analysis tensor to Drive and writes five PNG/PDF figure
families: reachability, oversquashing plus routing lesions, semantic/structural functional and
beneficial carriage, score--carriage/ablation/rescue causality, and dense--masked carriage
similarity. Use `--phase train`, `--phase analyze`, and `--phase figures` to split the GPU and
plotting stages; use `--fast-dev-run` only to test installation and plumbing.
Structural carriage conjugates RRWP and trained sparse support together; the per-head structural
specialisation score separately freezes support, matching the distinction in the two production
methodologies.
If a run misses the reachable-cell gate, the notebook makes up to two recorded alternate
initialisation attempts while holding its task/data seed fixed. Selection uses validation only;
held-out accuracy remains a final gate, and successful checkpoints are reused unchanged.

### Analysis note: sensitivity versus selectivity

Keep the production scores `S_sem` and `S_str` unchanged: each remains a raw measure of how
strongly that factor's intervention reaches the output through a head. Do not interpret their
mean-normalised scatter as evidence that the two factors have equal total strength; normalising
each axis separately deliberately removes that comparison.

For cross-channel analysis, first calibrate each score against a matched null from the same
intervention family, retaining intervention scale while removing factor--task alignment. Denote
the resulting comparable effect sizes by `S_sem_tilde` and `S_str_tilde`, then report

```text
J = (S_sem_tilde + S_str_tilde) / 2   # shared/non-specific head response
D = (S_sem_tilde - S_str_tilde) / 2   # semantic-versus-structural selectivity
```

`J` distinguishes generally influential heads from specialists; the sign and magnitude of `D`
capture the off-diagonal preference. Report raw and calibrated scores alongside these derived
quantities rather than folding selectivity into the core score. Validate `D` with the full
cross-channel matrix (semantic/structural score against semantic/structural carriage loss), and
keep task usefulness separate through beneficial carriage, ablation, and rescue. The exact
matched-null construction must be fixed and verified before using `D` for headline claims.

## CausalSpecialisationDoubleDissociation

`training/causal_specialisation_double_dissociation_colab.py` is a single-cell,
official-GRIT experiment for causal validation of the production semantic and structural
head scores. A dense GRIT jointly learns marked-source value retrieval and structural
source-distance classification on the same cycle graphs. The shared source marker controls
addressing difficulty, leaving semantic payload versus RRWP relation as the task contrast. It then runs planted-source
scoring, every-head pre-output ablation, and clean-to-corrupt head-output patching.
It also jointly ablates score-selected head families and traces fixed score-ranked ablation
prefixes, testing whether necessity appears only after redundant specialised heads are removed.

Paste the complete file into Colab and run it. The default three-seed run mounts Drive,
pins official GRIT, caches checkpoints and analysis tensors, and writes PNG/PDF versions
of the specialisation plane, score-ablation correlations, and necessity-plus-rescue
double-dissociation figure, plus cumulative family-ablation curves. For an
installation/plumbing check, change the final call to:

```python
main(["--fast-dev-run"])
```

Cached phases can be rerun independently with `--phase train`, `--phase analyze`, or
`--phase figures`; the figures-only phase does not reinstall GRIT. When a cached v2 analysis
lacks the newer family-ablation block, `--phase analyze` augments it without recomputing scores,
single-head ablations, rescue results, or training.

The figure-only stage also derives a joint-strength/selectivity view from the cached per-head
tensors. Within each seed, semantic and structural scores are divided by their respective head
means before computing

```text
J     = (S_sem_norm + S_str_norm) / 2
D_rel = (S_sem_norm - S_str_norm) / (S_sem_norm + S_str_norm)
```

`fig5_joint_influence_selectivity` shows the joint-sensitivity/selectivity plane, tests whether
`J` predicts mean cross-task functional ablation impact, and tests whether `D_rel` predicts both
semantic-minus-structural ablation role and causal rescue role. Heads with `J < 0.5` are faded and
excluded from selectivity correlations because a ratio of two tiny scores is unstable. The
semantic-specialist, generalist, structural-specialist, and low-`J` quadrant summaries remain in
the cached analysis tables. This is a within-seed head-allocation analysis, not yet a claim that
raw semantic and structural score amplitudes are directly comparable across intervention families.

`--phase analyze` enriches older caches once with four additional fixed score-selected family
ablations: semantic specialists, structural specialists, high-`J` generalists, and low-`J`/inert
heads. `fig6_joint_selectivity_family_ablation` compares their cumulative functional and accuracy
effects on both tasks. It reuses the checkpoint and every existing score, single-head ablation,
and rescue tensor; no model is retrained.

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
