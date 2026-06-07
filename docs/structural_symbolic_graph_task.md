# Structural/Symbolic Graph Task

This task translates the sequence-level positional/symbolic split from Urrutia
et al., "Decoupling Positional and Symbolic Attention Behavior in Transformers,"
into a graph learning setting.

The paper's index task asks a model to retrieve the symbol at a specified
position, while its information-retrieval task asks a model to retrieve the
value associated with a matching query symbol. The graph version keeps that
operational distinction but replaces sequence position with graph position and
semantic retrieval over candidate nodes.

The default generator is now GraphWorld-style. Each example is a fresh synthetic
graph drawn from a family of random graph generators rather than a fixed motif
layout. This follows the GraphWorld benchmarking motivation: use controlled,
tunable synthetic graph populations to expose model sensitivity to graph
properties and task design.

## Graph Construction

The default `graphworld` style creates:

- one readout/query node;
- a random graph backbone sampled from ER, small-world, preferential-attachment,
  SBM-like, or mixed families;
- a variable number of candidate record nodes sampled from the backbone;
- one value leaf attached to each candidate;
- in symbolic and dual runs, the readout's primary symbol is the query key.
  Query-key leaves and candidate-key features provide optional extra components
  for harder partial-match cases;
- in structural and dual runs, one anchor node and one structurally marked
  target candidate on the backbone.

The number of backbone nodes and candidate records is sampled independently for
every graph, so input size is variable within train, validation, ID test, and
OOD test splits.

The default suite runs three experiments:

- **Structural/positional:** query/key symbols are disabled. An anchor node and
  a target candidate are marked by roles, and the label is their shortest-path
  distance shell. The default is a balanced 4-way shell label with RWSE
  features. SPD attention bias is available but tends to make this task too easy
  for the 2-layer baseline.
- **Symbolic/semantic:** the structural anchor selector is disabled. The readout
  carries the primary query key directly, and the target is the value attached
  to the candidate whose key tuple matches exactly. The default `mixed`
  distractor setting combines easier one-key retrieval with harder partial-match
  examples.
- **Dual:** both selectors are present on the same random graph. By default the
  structural and symbolic candidates differ, so the labels cannot collapse into
  the same route. `--dual-symbolic-loss-weight` can be increased if the semantic
  retrieval objective learns more slowly than the distance-shell label.

The older record/motif task is still available with `--dataset-style record`.

## Labels

- **Structural:** classify the anchor-distance shell of the marked target
  in the default `distance_bin` setting. `--graphworld-structural-label
  value_at_nearest` switches back to retrieving that candidate's value.
- **Symbolic:** output the value attached to the candidate whose key matches the
  readout query.
- **Dual:** predict both labels on the same graph.

The default baseline is a 2-layer, 2-head GraphGPS-style model. The structural
head reads from the marked target node, while the symbolic head reads from the
readout/query node. Layer 1 can move local key/value evidence into candidate
nodes and propagate anchor information over the random backbone. Layer 2 can
support either the positional target-node computation or the semantic
candidate-to-readout route. The dual task is intended to make separate routing
strategies useful enough that the two global heads can differentiate.

## Diagnostics

The runner reports two permutation-based head scores:

- `structural_score`: attention rows are invariant when node symbols are
  permuted while graph structure is fixed;
- `symbolic_score`: attention rows transform equivariantly with the symbol
  permutation.
- `centered_structural_score`: row-centered structural score from the PDF,
  computed after subtracting the per-row attention mean.
- `centered_symbolic_score`: row-centered symbolic score from the PDF.

The centered variants use the same global node-symbol permutations as the
existing synthetic runner rather than the PDF's local key transpositions.

It also reports task-aware attention mass:

- `cls_to_structural_record`;
- `cls_to_symbolic_record`;
- `structural_record_to_value`;
- `symbolic_record_to_value`;
- `cls_attention_entropy`.

These are plotted in `head_specialisation_id.png` alongside the
structural/symbolic score plane. The suite-level
`suite_model_metric_comparison.png` compares accuracy, centered score planes,
per-head permutation metrics, and final-layer target attention across the
structural-trained, symbolic-trained, and dual-trained models.

## Commands

```bash
python experiments/synthetic/training/structural_symbolic_graphgps.py
python experiments/synthetic/training/structural_symbolic_graphgps.py --task structural
python experiments/synthetic/training/structural_symbolic_graphgps.py --task symbolic
python experiments/synthetic/training/structural_symbolic_graphgps.py --include-dual
python experiments/synthetic/training/structural_symbolic_graphgps.py \
  --graph-family sbm \
  --graph-min-nodes 32 \
  --graph-max-nodes 64
python experiments/synthetic/training/structural_symbolic_graphgps.py \
  --task dual \
  --specialisation-loss-weight 0.01
python experiments/synthetic/training/structural_symbolic_graphgps.py \
  --task dual \
  --dual-symbolic-loss-weight 3.0
```

In Google Colab, the script mounts Drive by default and writes to
`/content/drive/MyDrive/graph_specialisation_metrics/structural_symbolic_graphgps`.
Use `--output-dir` to override the root or `--no-mount-drive` to keep outputs
inside the runtime.
