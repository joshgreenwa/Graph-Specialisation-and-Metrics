# GraphBench Algorithmic HPC Base Runs

This repository contains the HPC/SLURM setup for compact hard-OOD GraphBench
algorithmic base training.

## Model Fidelity

Paper runs should use official-backed model implementations. The current
`bin/algoreas_hpc.py` runner is useful for GraphBench loading, PE-cache,
checkpointing, logging, and SLURM scaffolding, but its local dense model classes
must not be reported as faithful Graphormer, GraphGPS, GRIT, or GNN+
implementations.

See `docs/model_fidelity_audit.md` before launching paper runs. The intended
implementation boundary is:

- official repos provide model/layer implementations
- this repo provides GraphBench adapters, cached PEs, task heads, metrics,
  checkpointing, W&B, and SLURM orchestration

## Files

```bash
bin/algoreas_hpc.py
bin/aggregate_hpc_results.py
bin/check_official_backends.py
configs/algoreas_hpc_base_v1.yaml
docs/hpc_environment.md
docs/model_fidelity_audit.md
slurm/precompute_pe.sbatch
slurm/train_base_array.sbatch
slurm/final_eval_array.sbatch
slurm/aggregate_results.sbatch
requirements.txt
env.example
```

## Environment

Set these before submitting jobs:

```bash
export PROJECT_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas-hpc
export GRAPHBENCH_DATASET_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas-hpc/datasets
export GRAPHBENCH_PE_CACHE_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas-hpc/pe_cache
export GRAPHBENCH_OUTPUT_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas-hpc/outputs
export ENV_ACTIVATE=/path/to/venv_or_conda_activate_script  # optional
export GRAPHBENCH_NUM_WORKERS=4
export WANDB_MODE=online
export WANDB_PROJECT=graphbench-algoreas-hpc
export WANDB_TAGS=base,hpc
export WANDB_API_KEY=replace_with_your_wandb_key
```

The SLURM scripts use these RDS paths as defaults, so exporting them is only
needed if you want to override the default locations.

The environment must already provide `torch`, `graphbench-lib`, `wandb`, and the usual
scientific Python stack. The runner does not install packages inside jobs.
W&B logging is enabled by default for train/eval jobs when `WANDB_API_KEY` is
set. If needed, disable it with `WANDB_MODE=disabled`.

## Submit Order

Inspect the job table:

```bash
python bin/algoreas_hpc.py --print-jobs
```

Check official backend imports:

```bash
python bin/check_official_backends.py
```

Precompute base PE caches. This is a CPU array over the five tasks and runs one
task at a time via `%1`:

```bash
sbatch slurm/precompute_pe.sbatch
```

Train base models. This is a GPU array over 140 task/model/seed jobs and runs
one A100 job at a time via `%1`:

```bash
sbatch slurm/train_base_array.sbatch
```

At present this command is intentionally blocked by the Python runner until
official model backends are wired in. To run only a local smoke test of the
scaffold, pass `--allow-local-style-models` manually; do not use that flag for
paper results.

Optional full final evaluation from `best.pt`:

```bash
sbatch slurm/final_eval_array.sbatch
```

Aggregate final eval summaries:

```bash
sbatch slurm/aggregate_results.sbatch
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

## Official Model Backends

Use pinned official sources in the HPC environment:

```bash
git clone --recurse-submodules https://github.com/microsoft/Graphormer.git external/Graphormer
git clone https://github.com/rampasek/GraphGPS.git external/GraphGPS
git clone https://github.com/LiamMa/GRIT.git external/GRIT
git clone https://github.com/LUOyk1999/tunedGNN-G.git external/GNNPlus
```

Checked commits are recorded in `docs/model_fidelity_audit.md`. Pinning exact
commits is recommended for paper runs.

Before launching training arrays, run:

```bash
python bin/check_official_backends.py
```

All rows must pass. Environment notes and likely import failures are listed in
`docs/hpc_environment.md`.
