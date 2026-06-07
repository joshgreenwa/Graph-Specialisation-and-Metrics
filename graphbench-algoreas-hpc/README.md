# GraphBench Algorithmic HPC Base Runs

This repository contains the HPC/SLURM setup for compact hard-OOD GraphBench
algorithmic base training.

## Model Fidelity

Paper runs should use official-backed model implementations. The current
`bin/algoreas_hpc.py` runner now exposes official-backed GRIT, static-GRIT,
GCN+, GIN+, and GatedGCN+ paths through `--model-backend official`. Its local
dense model classes remain available only for explicit smoke tests and must not
be reported as faithful Graphormer, GraphGPS, GRIT, or GNN+ implementations.

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
export PROJECT_ROOT=/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics/graphbench-algoreas-hpc
export GRAPHBENCH_DATASET_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets
export GRAPHBENCH_PE_CACHE_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache
export GRAPHBENCH_OUTPUT_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs
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
python bin/check_official_backends.py --models static_grit,grit,gatedgcn_plus,gin_plus,gcn_plus
```

Precompute base PE caches. This is a CPU array over the five tasks and runs one
task at a time via `%1`:

```bash
sbatch slurm/precompute_pe.sbatch
```

Train base official-backed GRIT/GNN+ models. This is a GPU array over 100
task/model/seed jobs and runs one A100 job at a time via `%1`:

```bash
sbatch slurm/train_base_array.sbatch
```

The script defaults to `MODELS=static_grit,grit,gatedgcn_plus,gin_plus,gcn_plus`.
Graphormer and GraphGPS official wrappers are not wired into this runner yet;
local smoke-test variants require `--model-backend local --allow-local-style-models`
and should not be used for paper results.

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
--pe-cache-namespace base_40k4k4k_n64
```

Do not reuse this namespace for later size-generalisation runs. Use a separate
namespace such as `sizegen_n256` or `sizegen_grid_v1` so base PEs and
size-generalisation PEs cannot collide.

Training jobs use `--require-pe-cache`; if a base PE cache is missing, the job
fails instead of silently recomputing expensive PEs inside the GPU job.

PE precompute writes resumable partial shards under `*.pt.parts/` while a split
is in progress. If a CPU job times out, rerun the same command without
`--force-recompute-pe`; completed shards are reused and only missing graphs are
computed. Once a split finishes, the final `.pt` cache is written and the
temporary shard directory is removed.

## Base Protocol

- Tasks: `bipartite_matching_hard`, `flow_hard`, `mst_hard`,
  `maxclique_hard`, `bridges_hard`
- Splits: `40k/4k/4k`, hard OOD, `n=16/16/64` train/val/test
- Note: GraphBench's normal hard AlgoReas loader validates at `n=16` and tests
  most tasks at `n=128` (`flow` at `n=64`). This base run uses a compact
  `n=64` test for all tasks to reduce PE cost; full `n=128` evaluation can be
  run later from saved checkpoints.
- For compact `n=64` test splits not present in the official tar files, the
  runner generates exactly the requested test count with GraphBench's own
  AlgoReas generator and a fixed split seed.
- Default official-backed models: static-GRIT, GRIT, GatedGCN+, GIN+, GCN+
- Graphormer/GraphGPS: preflight-audited but not wired into the official runner
  path yet
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
git clone https://github.com/LUOyk1999/GNNPlus.git external/GNNPlus
```

Checked commits are recorded in `docs/model_fidelity_audit.md`. Pinning exact
commits is recommended for paper runs.

Before launching training arrays, run:

```bash
python bin/check_official_backends.py --models static_grit,grit,gatedgcn_plus,gin_plus,gcn_plus
```

All rows must pass. Environment notes and likely import failures are listed in
`docs/hpc_environment.md`.
