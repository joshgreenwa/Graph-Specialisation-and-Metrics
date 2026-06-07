# GraphBench Algorithmic HPC Base Runs

This is the HPC/SLURM setup for compact hard-OOD GraphBench algorithmic base
training. It is independent of the older Colab runner.

## Files

```bash
experiments/graphbench/hpc/bin/algoreas_hpc.py
experiments/graphbench/hpc/bin/aggregate_hpc_results.py
experiments/graphbench/hpc/configs/algoreas_hpc_base_v1.yaml
experiments/graphbench/hpc/slurm/precompute_pe.sbatch
experiments/graphbench/hpc/slurm/train_base_array.sbatch
experiments/graphbench/hpc/slurm/final_eval_array.sbatch
experiments/graphbench/hpc/slurm/aggregate_results.sbatch
```

## Environment

Set these before submitting jobs:

```bash
export PROJECT_ROOT=/path/to/Graph\ Specialisation\ and\ Metrics
export GRAPHBENCH_DATASET_ROOT=/path/to/graphbench_datasets
export GRAPHBENCH_PE_CACHE_ROOT=/path/to/graphbench_pe_cache
export GRAPHBENCH_OUTPUT_ROOT=/path/to/graphbench_outputs
export ENV_ACTIVATE=/path/to/venv_or_conda_activate_script  # optional
export GRAPHBENCH_NUM_WORKERS=4
```

The environment must already provide `torch`, `graphbench-lib`, and the usual
scientific Python stack. The runner does not install packages inside jobs.

## Submit Order

Inspect the job table:

```bash
python experiments/graphbench/hpc/bin/algoreas_hpc.py --print-jobs
```

Precompute base PE caches. This is a CPU array over the five tasks and runs one
task at a time via `%1`:

```bash
sbatch experiments/graphbench/hpc/slurm/precompute_pe.sbatch
```

Train base models. This is a GPU array over 140 task/model/seed jobs and runs
one A100 job at a time via `%1`:

```bash
sbatch experiments/graphbench/hpc/slurm/train_base_array.sbatch
```

Optional full final evaluation from `best.pt`:

```bash
sbatch experiments/graphbench/hpc/slurm/final_eval_array.sbatch
```

Aggregate final eval summaries:

```bash
sbatch experiments/graphbench/hpc/slurm/aggregate_results.sbatch
```

## PE Cache Policy

Base-training PEs use:

```bash
--pe-cache-namespace base
```

Do not reuse this namespace for later size-generalisation runs. Use a separate
namespace such as `sizegen_n256` or `sizegen_grid_v1` so base PEs and
size-generalisation PEs cannot collide.

Training jobs use `--require-pe-cache`; if a base PE cache is missing, the job
fails instead of silently recomputing expensive PEs inside the GPU job.

## Base Protocol

- Tasks: `bipartite_matching_hard`, `flow_hard`, `mst_hard`,
  `maxclique_hard`, `bridges_hard`
- Splits: `40k/4k/8k`, hard OOD, `n=16` train and `n=128` val/test
- Models: Graphormer, GraphGPS, static-GRIT, GRIT, GatedGCN+, GIN+, GCN+
- Seeds: `0,1,2,3`
- Training: 5000 steps, batch 1024, 500 warmup, cosine decay
- Checkpoints: every 500 steps from 2500 to 5000 plus `best.pt`
- Default LR policy: family-fixed, `2e-4` for transformer-style models and
  `1e-3` for GNN+ models
- Full train/val/test final eval is separate from base training
