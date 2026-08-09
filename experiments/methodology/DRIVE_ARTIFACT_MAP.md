# Drive checkpoint and cache map

All paths below are absolute Colab paths. The common canonical root is:

```text
M = /content/drive/MyDrive/graph_specialisation_metrics/multi_seed_models
C = M/corpus/1d41d10b5e5c32fa1ce70535de646c762ba3ad4ad3393406c12b64c21cfced22/zinc_qm9_best_available_20260807_151904/checkpoints
O = M/canonical_outputs
```

Replace `M`, `C`, or `O` in the tables with the corresponding full value above.

The immutable source archive and checksum are `M/zinc_qm9_best_checkpoints.tar` and
`M/zinc_qm9_best_checkpoints.tar.sha256`.

## Canonical 30-checkpoint run

| Model task | Seed | Checkpoint | Scores cache | Carriage cache |
|---|---:|---|---|---|
| `zinc` (dense) | 0 | `C/zinc.dense.s0/best_available.ckpt` | `O/zinc/seed_0/cache/scores/raw.pt` | `O/zinc/seed_0/cache/carriage/fields.pt` |
| `zinc` (dense) | 1 | `C/zinc.dense.s1/best_available.ckpt` | `O/zinc/seed_1/cache/scores/raw.pt` | `O/zinc/seed_1/cache/carriage/fields.pt` |
| `zinc` (dense) | 2 | `C/zinc.dense.s2/best_available.ckpt` | `O/zinc/seed_2/cache/scores/raw.pt` | `O/zinc/seed_2/cache/carriage/fields.pt` |
| `zinc_1hop` | 0 | `C/zinc.1hop.s0/best_available.ckpt` | `O/zinc_1hop/seed_0/cache/scores/raw.pt` | `O/zinc_1hop/seed_0/cache/carriage/fields.pt` |
| `zinc_1hop` | 1 | `C/zinc.1hop.s1/best_available.ckpt` | `O/zinc_1hop/seed_1/cache/scores/raw.pt` | `O/zinc_1hop/seed_1/cache/carriage/fields.pt` |
| `zinc_1hop` | 2 | `C/zinc.1hop.s2/best_available.ckpt` | `O/zinc_1hop/seed_2/cache/scores/raw.pt` | `O/zinc_1hop/seed_2/cache/carriage/fields.pt` |
| `zinc_1hop_vnode` | 0 | `C/zinc.1hop_vnode.s0/best_available.ckpt` | `O/zinc_1hop_vnode/seed_0/cache/scores/raw.pt` | `O/zinc_1hop_vnode/seed_0/cache/carriage/fields.pt` |
| `zinc_1hop_vnode` | 1 | `C/zinc.1hop_vnode.s1/best_available.ckpt` | `O/zinc_1hop_vnode/seed_1/cache/scores/raw.pt` | `O/zinc_1hop_vnode/seed_1/cache/carriage/fields.pt` |
| `zinc_1hop_vnode` | 2 | `C/zinc.1hop_vnode.s2/best_available.ckpt` | `O/zinc_1hop_vnode/seed_2/cache/scores/raw.pt` | `O/zinc_1hop_vnode/seed_2/cache/carriage/fields.pt` |
| `zinc_2hop` | 0 | `C/zinc.2hop.s0/best_available.ckpt` | `O/zinc_2hop/seed_0/cache/scores/raw.pt` | `O/zinc_2hop/seed_0/cache/carriage/fields.pt` |
| `zinc_2hop` | 1 | `C/zinc.2hop.s1/best_available.ckpt` | `O/zinc_2hop/seed_1/cache/scores/raw.pt` | `O/zinc_2hop/seed_1/cache/carriage/fields.pt` |
| `zinc_2hop` | 2 | `C/zinc.2hop.s2/best_available.ckpt` | `O/zinc_2hop/seed_2/cache/scores/raw.pt` | `O/zinc_2hop/seed_2/cache/carriage/fields.pt` |
| `zinc_2hop_vnode` | 0 | `C/zinc.2hop_vnode.s0/best_available.ckpt` | `O/zinc_2hop_vnode/seed_0/cache/scores/raw.pt` | `O/zinc_2hop_vnode/seed_0/cache/carriage/fields.pt` |
| `zinc_2hop_vnode` | 1 | `C/zinc.2hop_vnode.s1/best_available.ckpt` | `O/zinc_2hop_vnode/seed_1/cache/scores/raw.pt` | `O/zinc_2hop_vnode/seed_1/cache/carriage/fields.pt` |
| `zinc_2hop_vnode` | 2 | `C/zinc.2hop_vnode.s2/best_available.ckpt` | `O/zinc_2hop_vnode/seed_2/cache/scores/raw.pt` | `O/zinc_2hop_vnode/seed_2/cache/carriage/fields.pt` |
| `qm9_gap_dense` | 0 | `C/qm9_gap.dense.s0/best_available.ckpt` | `O/qm9_gap_dense/seed_0/cache/scores/raw.pt` | `O/qm9_gap_dense/seed_0/cache/carriage/fields.pt` |
| `qm9_gap_dense` | 1 | `C/qm9_gap.dense.s1/best_available.ckpt` | `O/qm9_gap_dense/seed_1/cache/scores/raw.pt` | `O/qm9_gap_dense/seed_1/cache/carriage/fields.pt` |
| `qm9_gap_dense` | 2 | `C/qm9_gap.dense.s2/best_available.ckpt` | `O/qm9_gap_dense/seed_2/cache/scores/raw.pt` | `O/qm9_gap_dense/seed_2/cache/carriage/fields.pt` |
| `qm9_gap_1hop` | 0 | `C/qm9_gap.1hop.s0/best_available.ckpt` | `O/qm9_gap_1hop/seed_0/cache/scores/raw.pt` | `O/qm9_gap_1hop/seed_0/cache/carriage/fields.pt` |
| `qm9_gap_1hop` | 1 | `C/qm9_gap.1hop.s1/best_available.ckpt` | `O/qm9_gap_1hop/seed_1/cache/scores/raw.pt` | `O/qm9_gap_1hop/seed_1/cache/carriage/fields.pt` |
| `qm9_gap_1hop` | 2 | `C/qm9_gap.1hop.s2/best_available.ckpt` | `O/qm9_gap_1hop/seed_2/cache/scores/raw.pt` | `O/qm9_gap_1hop/seed_2/cache/carriage/fields.pt` |
| `qm9_gap_1hop_vnode` | 0 | `C/qm9_gap.1hop_vnode.s0/best_available.ckpt` | `O/qm9_gap_1hop_vnode/seed_0/cache/scores/raw.pt` | `O/qm9_gap_1hop_vnode/seed_0/cache/carriage/fields.pt` |
| `qm9_gap_1hop_vnode` | 1 | `C/qm9_gap.1hop_vnode.s1/best_available.ckpt` | `O/qm9_gap_1hop_vnode/seed_1/cache/scores/raw.pt` | `O/qm9_gap_1hop_vnode/seed_1/cache/carriage/fields.pt` |
| `qm9_gap_1hop_vnode` | 2 | `C/qm9_gap.1hop_vnode.s2/best_available.ckpt` | `O/qm9_gap_1hop_vnode/seed_2/cache/scores/raw.pt` | `O/qm9_gap_1hop_vnode/seed_2/cache/carriage/fields.pt` |
| `qm9_gap_2hop` | 0 | `C/qm9_gap.2hop.s0/best_available.ckpt` | `O/qm9_gap_2hop/seed_0/cache/scores/raw.pt` | `O/qm9_gap_2hop/seed_0/cache/carriage/fields.pt` |
| `qm9_gap_2hop` | 1 | `C/qm9_gap.2hop.s1/best_available.ckpt` | `O/qm9_gap_2hop/seed_1/cache/scores/raw.pt` | `O/qm9_gap_2hop/seed_1/cache/carriage/fields.pt` |
| `qm9_gap_2hop` | 2 | `C/qm9_gap.2hop.s2/best_available.ckpt` | `O/qm9_gap_2hop/seed_2/cache/scores/raw.pt` | `O/qm9_gap_2hop/seed_2/cache/carriage/fields.pt` |
| `qm9_gap_2hop_vnode` | 0 | `C/qm9_gap.2hop_vnode.s0/best_available.ckpt` | `O/qm9_gap_2hop_vnode/seed_0/cache/scores/raw.pt` | `O/qm9_gap_2hop_vnode/seed_0/cache/carriage/fields.pt` |
| `qm9_gap_2hop_vnode` | 1 | `C/qm9_gap.2hop_vnode.s1/best_available.ckpt` | `O/qm9_gap_2hop_vnode/seed_1/cache/scores/raw.pt` | `O/qm9_gap_2hop_vnode/seed_1/cache/carriage/fields.pt` |
| `qm9_gap_2hop_vnode` | 2 | `C/qm9_gap.2hop_vnode.s2/best_available.ckpt` | `O/qm9_gap_2hop_vnode/seed_2/cache/scores/raw.pt` | `O/qm9_gap_2hop_vnode/seed_2/cache/carriage/fields.pt` |

For any canonical row, its run directory is `O/<model task>/seed_<seed>/`. It also contains:

```text
protocol.json
model.json
audits.json
progress.jsonl
worker_complete.json
cache/clean_jacobians/graph_XXXXXX.pt
cache/scores/semantic/graph_XXXXXX.pt
cache/scores/structural/graph_XXXXXX.pt
cache/carriage/semantic/graph_XXXXXX.pt
cache/carriage/structural/graph_XXXXXX.pt
```

`raw.pt` and `fields.pt` are the consolidated caches to use for downstream analysis. The
`graph_XXXXXX.pt` files are resumable intermediate shards. Any rejected older artifact is retained
under the same run's `cache/_stale/` directory.

## Chapter 6 multi-seed analysis

Run `experiments/methodology/chapter6_multiseed_colab.ipynb` with `DATASET = "zinc"` or
`DATASET = "qm9"`. The two runs write independently to:

```text
M/chapter6_multiseed_analysis/zinc
M/chapter6_multiseed_analysis/qm9
```

Each run reads the five architectures and three seeds listed above. Score and carriage figures are
cache-only. On the first run for a dataset, the notebook also computes the clean single-head
ablation endpoint on 64 held-out graphs and writes resumable per-head shards under each canonical
run's `cache/causal/clean_ablation/` directory. Compact plotting summaries are written to:

```text
M/chapter6_clean_head_ablation/<model task>/seed_<seed>.json
M/chapter6_clean_head_ablation/zinc_complete.json
M/chapter6_clean_head_ablation/qm9_complete.json
```

Each dataset completion marker binds all 15 compact summaries by SHA-256. The summaries bind the
underlying score cache and checkpoint, while the canonical per-head shards retain the 64 graph
records. Use a T4 GPU for the first pass; subsequent figure-only reruns validate and reuse these
caches without loading a model.

### Controlled special-head diagnostics

The same frontend selects three distinct active heads per architecture across the three seeds:
the largest `D_rel`, the smallest `D_rel`, and the largest remaining `J`. It then caches ten
controlled molecule examples and a 500-molecule chemistry-labelled routed-output PCA for each
selected head. Outputs live beneath the dataset analysis root:

```text
M/chapter6_multiseed_analysis/<dataset>/special_head_analysis/selected_heads.csv
M/chapter6_multiseed_analysis/<dataset>/special_head_analysis/selected_heads.json
M/chapter6_multiseed_analysis/<dataset>/special_head_analysis/manifest.json
M/chapter6_multiseed_analysis/<dataset>/special_head_analysis/cache/<model task>/seed_<seed>/
M/chapter6_multiseed_analysis/<dataset>/special_head_analysis/figures/
```

Each selected head receives two five-molecule attention pages (ten examples in total), one
chemistry-labelled PCA, and one semantic/structural/attention distance-profile figure. The cache
contracts bind the score-cache SHA, checkpoint SHA, selected heads, graph indices, PCA graph
count, and chemistry-label version. Virtual-node models preserve the auxiliary node in the
attention matrix and distance profile; chemistry PCA remains defined over real molecular
receivers. Once these supplemental caches exist, figures and smaller paper subsets can be
regenerated without loading a checkpoint by setting `SPECIAL_HEAD_RENDER_GRAPH_INDICES` to any
subset of the cached ten indices.

## Seed-0 dense/1-hop checkpoint trajectory

The trajectory roots are:

```text
T  = M/multiple_checkpoints_zinc
TC = T/corpus/e8b37688bea8abefba11fe972345c1bf539cdb98d55b87fd625fe2f0e4d33634/zinc_dense_1hop_seed0_trajectory
TO = T/score_trajectory_outputs
```

This workflow computes scores only; there are no carriage caches.
Its immutable source archive and checksum are `T/zinc_dense_1hop_seed0_trajectory.tar` and
`T/zinc_dense_1hop_seed0_trajectory.tar.sha256`.

| Architecture | Seed | Epoch | Checkpoint | Scores cache |
|---|---:|---:|---|---|
| dense | 0 | 10 | `TC/dense/epoch10.ckpt` | `TO/dense/epoch_0010/zinc/seed_0/cache/scores/raw.pt` |
| dense | 0 | 100 | `TC/dense/epoch100.ckpt` | `TO/dense/epoch_0100/zinc/seed_0/cache/scores/raw.pt` |
| dense | 0 | 250 | `TC/dense/epoch250.ckpt` | `TO/dense/epoch_0250/zinc/seed_0/cache/scores/raw.pt` |
| dense | 0 | 500 | `TC/dense/epoch500.ckpt` | `TO/dense/epoch_0500/zinc/seed_0/cache/scores/raw.pt` |
| dense | 0 | 1000 | `TC/dense/epoch1000.ckpt` | `TO/dense/epoch_1000/zinc/seed_0/cache/scores/raw.pt` |
| dense | 0 | 1990 | `TC/dense/epoch1990.ckpt` | `TO/dense/epoch_1990/zinc/seed_0/cache/scores/raw.pt` |
| 1-hop | 0 | 10 | `TC/1hop/epoch10.ckpt` | `TO/1hop/epoch_0010/zinc_1hop/seed_0/cache/scores/raw.pt` |
| 1-hop | 0 | 100 | `TC/1hop/epoch100.ckpt` | `TO/1hop/epoch_0100/zinc_1hop/seed_0/cache/scores/raw.pt` |
| 1-hop | 0 | 250 | `TC/1hop/epoch250.ckpt` | `TO/1hop/epoch_0250/zinc_1hop/seed_0/cache/scores/raw.pt` |
| 1-hop | 0 | 500 | `TC/1hop/epoch500.ckpt` | `TO/1hop/epoch_0500/zinc_1hop/seed_0/cache/scores/raw.pt` |
| 1-hop | 0 | 1000 | `TC/1hop/epoch1000.ckpt` | `TO/1hop/epoch_1000/zinc_1hop/seed_0/cache/scores/raw.pt` |
| 1-hop | 0 | 1990 | `TC/1hop/epoch1990.ckpt` | `TO/1hop/epoch_1990/zinc_1hop/seed_0/cache/scores/raw.pt` |

Each trajectory run directory also contains `protocol.json`, `model.json`, `audits.json`,
`progress.jsonl`, `trajectory_score_complete.json`, clean-Jacobian shards, and semantic/structural
score shards in the same relative locations shown for the canonical run.

The final plots and machine-readable summaries are:

```text
T/score_trajectory_plots/dense/score_distributions.png
T/score_trajectory_plots/dense/score_distributions.pdf
T/score_trajectory_plots/dense/head_scores_long.csv
T/score_trajectory_plots/dense/score_summary.csv
T/score_trajectory_plots/dense/plot_manifest.json

T/score_trajectory_plots/1hop/score_distributions.png
T/score_trajectory_plots/1hop/score_distributions.pdf
T/score_trajectory_plots/1hop/head_scores_long.csv
T/score_trajectory_plots/1hop/score_summary.csv
T/score_trajectory_plots/1hop/plot_manifest.json
```

The ZINC Chapter 6 frontend validates the twelve `TO` score caches before plotting. If any are
missing, it uses the immutable `TC` checkpoint corpus to resume or compute only the affected
architecture; completed epochs are validated and skipped. Use a T4 GPU for this first pass.
Subsequent runs are cache-only. The frontend additionally writes:

```text
M/chapter6_multiseed_analysis/zinc/figures/11a_dense_score_trajectory.png
M/chapter6_multiseed_analysis/zinc/figures/11a_dense_score_trajectory.pdf
M/chapter6_multiseed_analysis/zinc/figures/11b_1hop_score_trajectory.png
M/chapter6_multiseed_analysis/zinc/figures/11b_1hop_score_trajectory.pdf
M/chapter6_multiseed_analysis/zinc/zinc_checkpoint_trajectory_heads.csv
```

Each architecture receives a 2x6 figure: raw semantic versus structural scores on the top row and
`J` versus `D_rel` on the bottom row. Axis limits are computed jointly over both architectures and
all twelve checkpoints, so the two figures use identical scales. These outputs are ZINC-only.

## Peptides-func / Peptides-struct, five architectures, three seeds

```text
P  = M/peptides_func_struct_checkpoints
PC = P/corpus/d6e5923249326f0d2498d7f5ca996ec2a25c4d213c41cbfdf9c7a28224ad3456/peptides_best_available_20260809_153204
PO = P/canonical_outputs
```

For each row below and each seed `s` in `0,1,2`:

```text
checkpoint = PC/checkpoints/<archive prefix>.s<s>/best_available.ckpt
scores     = PO/<canonical task>/seed_<s>/cache/scores/raw.pt
carriage   = PO/<canonical task>/seed_<s>/cache/carriage/fields.pt
```

| Dataset/model | Archive prefix | Canonical task |
|---|---|---|
| Peptides-func dense | `peptides_func.dense` | `peptides_func_dense` |
| Peptides-func 1-hop | `peptides_func.1hop` | `peptides_func_1hop` |
| Peptides-func 1-hop + VNode | `peptides_func.1hop_vnode` | `peptides_func_1hop_vnode` |
| Peptides-func 2-hop | `peptides_func.2hop` | `peptides_func_2hop` |
| Peptides-func 2-hop + VNode | `peptides_func.2hop_vnode` | `peptides_func_2hop_vnode` |
| Peptides-struct dense | `peptides_struct.dense` | `peptides_struct_dense` |
| Peptides-struct 1-hop | `peptides_struct.1hop` | `peptides_struct_1hop` |
| Peptides-struct 1-hop + VNode | `peptides_struct.1hop_vnode` | `peptides_struct_1hop_vnode` |
| Peptides-struct 2-hop | `peptides_struct.2hop` | `peptides_struct_2hop` |
| Peptides-struct 2-hop + VNode | `peptides_struct.2hop_vnode` | `peptides_struct_2hop_vnode` |

Every run directory also contains `protocol.json`, `model.json`, `audits.json`, `progress.jsonl`,
`worker_complete.json`, clean-Jacobian shards, and per-channel score/carriage graph shards. Rejected
incompatible artifacts are retained under that run's `cache/_stale/` directory.
