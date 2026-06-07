# GraphBench EC5 Fast Pilot

## Algorithmic Reasoning Hard-Base Runs

The base configuration for the dissertation/paper algorithmic-reasoning runs is
stored in:

```bash
experiments/graphbench/configs/algoreas_hard_base_v1.yaml
```

The matching runner defaults to that protocol. The intended production use is
one task/model/seed per A100 job. The default suite is five hard tasks:
`bipartite_matching_hard`, `flow_hard`, `mst_hard`, `maxclique_hard`, and
`bridges_hard`.

```bash
python experiments/graphbench/training/official_algoreas_screen_colab.py --print-jobs
python experiments/graphbench/training/official_algoreas_screen_colab.py --array-index 0
```

Preparing the shared official GraphBench subset caches can be run separately:

```bash
python experiments/graphbench/training/official_algoreas_screen_colab.py --prepare-data-only
```

Precompute reusable CPU positional/structural encodings before GPU training:

```bash
python experiments/graphbench/training/official_algoreas_screen_colab.py \
  --precompute-pe-only \
  --dataset-root "$DATASET_ROOT" \
  --pe-cache-root "$PE_CACHE_ROOT" \
  --device cpu \
  --allow-cpu
```

SLURM templates are provided under:

```bash
experiments/graphbench/slurm/precompute_pe.sbatch
experiments/graphbench/slurm/train_one_at_a_time.sbatch
experiments/graphbench/slurm/final_eval_one_at_a_time.sbatch
```

For a local smoke test that does not need GraphBench data:

```bash
python experiments/graphbench/training/official_algoreas_screen_colab.py --fast-dev-run --print-jobs
```

Protocol highlights:

- hard GraphBench algorithmic splits: `n=16` ER-only train and `n=128` no-ER
  validation/test;
- `40k/4k/8k` graph subsets, four model seeds, fixed task/model learning-rate
  policy;
- 6-layer parameter-matched models around `2.2M` trainable parameters:
  Graphormer `d=160`, GraphGPS `d=128`, GRIT `d=112`, static-GRIT `d=168`,
  GatedGCN+ `d=184`, GIN+ `d=224`, GCN+ `d=240`;
- Graphormer uses degree/SPD/direct edge bias and no RWSE/RRWP;
- GraphGPS/GNN+ use RWSE16 node PE;
- GRIT/static-GRIT use RRWP16 pair PE with no SPD;
- fixed `5k` training steps, `500` warmup, no model-dependent early
  termination, checkpoint selection only from step `2500` onward;
- default LR policy is family-fixed for fairness: `2e-4` for transformer-style
  models and `1e-3` for GNN+ models. The GraphBench task-specific table remains
  available via `--lr-policy graphbench_table`;
- base-training jobs skip the expensive full train/val/test sweep by default;
  pass `--final-eval` later to run it from the selected checkpoint;
- A100-oriented per-model physical batch sizes by default: train batch `1024`,
  eval batches `128/64/32/256` for Graphormer/GraphGPS/GRIT/GNN+ families;
- major checkpoints are saved at every eligible evaluation step
  (`checkpoint_step02500.pt`, ..., `checkpoint_step05000.pt`) plus `best.pt`.

The runner caches converted official GraphBench graph subsets and can separately
cache SPD/RWSE/RRWP tensors on CPU storage. Training jobs should use
`--require-pe-cache` after the PE precompute job to avoid accidental slow
on-the-fly PE collation. Base-training jobs attach PE caches only for train and
validation; the test PE cache is loaded only by the later `--final-eval` pass.

This folder contains the first GraphBench pilot target:
`electronic_circuits_5_vout`.

Run all three comparable-lite models:

```bash
python experiments/graphbench/training/ec5_vout_fast.py --model all
```

Useful Colab-style single-model runs:

```bash
python experiments/graphbench/training/ec5_vout_fast.py --model graphormer
python experiments/graphbench/training/ec5_vout_fast.py --model graphgps
python experiments/graphbench/training/ec5_vout_fast.py --model grit
```

The runner first tries `graphbench-lib` and falls back to the public EC5 JSON
archive if GraphBench is not installed. Results are written under
`experiments/graphbench/results/ec5_vout_fast/<model>/seed0/`.

For Colab, use the standalone file:

```bash
python experiments/graphbench/training/ec5_vout_fast_colab.py --model all
```

By default it requires a CUDA GPU, mounts Drive, and writes to
`/content/drive/MyDrive/graph_specialisation_metrics/graphbench_ec5/`.
If `best.pt` and `summary.json` already exist for a model/seed, the Colab
runner skips training, reloads the checkpoint, and fills in any missing metrics
or plots. Pass `--force-retrain` to overwrite a completed run, or
`--force-metrics` to recompute permutation diagnostics. For local smoke tests
without a GPU, pass `--allow-cpu`.

Each model directory includes training curves, final metric bars, prediction
diagnostics, EC5->EC7/EC10 zero-shot metrics, initial/best alpha-weighted
positional-symbolic permutation CSVs, query-level permutation CSVs, and
permutation-plane PNGs. The alpha-weighted metrics include node-feature
permutation, structural-feature permutation, joint equivariance, centered
variants, entropy, and attention residual norm.

Useful metric/eval switches:

```bash
python experiments/graphbench/training/ec5_vout_fast_colab.py --model all --force-metrics
python experiments/graphbench/training/ec5_vout_fast_colab.py --model all --zero-shot-components 7
python experiments/graphbench/training/ec5_vout_fast_colab.py --model all --zero-shot-components ""
```

For a quick integration check:

```bash
python experiments/graphbench/training/ec5_vout_fast.py --model graphormer --fast-dev-run
```
