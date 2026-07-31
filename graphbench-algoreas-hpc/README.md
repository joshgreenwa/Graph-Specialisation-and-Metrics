# GraphBench Algorithmic HPC Base Runs

This repository contains the HPC/SLURM setup for compact hard-OOD GraphBench
algorithmic base training.

## Model Fidelity

Paper runs should use official-backed model implementations. The current
`bin/algoreas_hpc.py` runner now exposes official-backed GraphGPS, GRIT,
static-GRIT, GCN+, GIN+, and GatedGCN+ paths through `--model-backend official`. Its local
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
python bin/check_official_backends.py --models graphgps,static_grit,grit,gatedgcn_plus,gin_plus,gcn_plus
```

Precompute base PE caches. This is a CPU array over the five tasks and runs one
task at a time via `%1`:

```bash
sbatch slurm/precompute_pe.sbatch
```

Train base official-backed GraphGPS/GRIT/GNN+ models. This is a GPU array over 120
task/model/seed jobs and runs one A100 job at a time via `%1`:

```bash
sbatch slurm/train_base_array.sbatch
```

The script defaults to `MODELS=graphgps,static_grit,grit,gatedgcn_plus,gin_plus,gcn_plus`.
Graphormer official wrappers are not wired into this runner yet; its local
smoke-test variant requires `--model-backend local --allow-local-style-models`
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

Training jobs use `--require-subset-cache`, `--no-build-missing-pe-cache`, and
`--require-pe-cache`; if a converted GraphBench split cache or base PE cache is
missing, the job fails instead of loading/generating data or recomputing
expensive PEs inside the GPU job.

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
- Default official-backed models: GraphGPS, static-GRIT, GRIT, GatedGCN+, GIN+, GCN+
- Graphormer: preflight-audited but not wired into the official runner path yet
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
python bin/check_official_backends.py --models graphgps,static_grit,grit,gatedgcn_plus,gin_plus,gcn_plus
```

All rows must pass. Environment notes and likely import failures are listed in
`docs/hpc_environment.md`.

# GRIT specialisation and carriage

The trained GRIT checkpoints for `bipartite_matching_hard` and `flow_hard` are supported by the
canonical `donor-swap-specialisation-carriage-v4` implementation through the explicitly labelled
`graphbench-complete-pe-coherent-v2` extension. GCN+ checkpoints are intentionally rejected: there is no
GCN+ transport-site adapter in the canonical methodology.

GraphBench has no swappable node-content row for these tasks. The semantic source is therefore one
normalized weighted edge from the model input. Flow uses one ordered directed edge; matching treats
a reciprocal pair as one semantic unit when both orientations are present. The external training
donor law is graph-balanced and matches the source's endpoint-degree signature before drawing an
edge value. The structural intervention is the locked fixed-support complete-PE donor-copy: it
copies one model-visible RRWP row/column/self footprint plus degree, from which the official
adapter re-derives log-degree. It does not degree-match donors and never changes topology, weights,
labels, or flow source/sink markers. GraphBench raw scores use coherent output movement; transport
mass remains a secondary diagnostic.

Production defaults are recorded in
[`configs/grit_specialisation_graphbench_edge_v1.yaml`](configs/grit_specialisation_graphbench_edge_v1.yaml):
64 discovery graphs, 32 causal graphs, 96 held-out clean-ablation graphs, up to 16 sources per
channel, 8 donors per source, and 2,000 bootstrap draws. All 16 structural node sources are
enumerated on the primary `n=16` validation split, so structural-source resampling is disabled;
semantic edge sources are sampled when more than 16 are eligible.

The production GPU envelope requests 32 base graphs per forward, allows up to 2,000,000
`replicas * nodes^2` units in an event batch, and computes matching-output VJPs in chunks of 64.
Graph batches halve and retain the smaller size after a CUDA OOM. Heartbeats record process
allocated/reserved/peak CUDA memory plus device utilization, total VRAM use, and power from
`nvidia-smi`; these execution settings do not alter the scientific fingerprint.

Run all models sequentially in one process:

```bash
python graphbench-algoreas-hpc/bin/grit_specialisation.py \
  --tasks bipartite_matching_hard,flow_hard \
  --seeds 0,1,2,3 \
  --profile production
```

For HPC production, use isolated task/seed GPU workers and one CPU finalizer:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics

ENV_ACTIVATE=$PWD/graphbench-algoreas-hpc/activate_graphbench_algoreas \
PROFILE=production \
ANALYSIS_TASKS=bipartite_matching_hard \
PHASES=scores,causal \
MAX_PARALLEL=4 \
bash graphbench-algoreas-hpc/bin/submit_grit_specialisation.sh
```

The focused default submits one four-seed matching array. Set
`ANALYSIS_TASKS=bipartite_matching_hard,flow_hard` only when both registered tasks are intended.
Each worker writes only
`<analysis-root>/<task>/seed_<seed>/`, so workers never race on task or root summaries. One
`afterok` CPU job verifies the selected seeds' cache contracts, renders every figure, and writes
the task `population.json` plus the root summaries. Each GPU array element requests six hours;
interrupted runs resume from completed atomic graph and target shards. The focused run caches
scores and causal results only; carriage is optional and is not required by the finalizer.
The launcher validates `GRIT_ROOT` as the pinned official Git checkout and transports the four
seeds through Slurm as a colon-separated list, avoiding the comma semantics of `--export`.

Every long component writes atomic graph/target shards and a consolidated cache. Re-running the
same command resumes missing work. `progress.jsonl` and stdout include the active
task/seed/component/channel, cache use, graph counts, elapsed time, heartbeat, and CUDA memory.
Numerical/no-op/attention/replay/completeness gates are soft by default and make the run
`headline_eligible=false`; checkpoint geometry, event-manifest alignment, patch geometry, and
cache corruption remain hard failures. Add `--strict-audits` for release verification.
Non-finite projected-transport entries and impossible negative/norm-rounding score artefacts are
recorded, conservatively repaired, and allowed to finish rather than terminating a queued worker.

The finalizer can be rerun manually on CPU without importing GRIT, loading a checkpoint, or
reopening the dataset:

```bash
python graphbench-algoreas-hpc/bin/grit_specialisation.py \
  --tasks bipartite_matching_hard \
  --seeds 0,1,2,3 \
  --phases figures \
  --accelerator cpu
```

This requires the four consolidated score and causal caches; carriage is loaded only when present.
It regenerates figure artifacts and atomically rebuilds the complete four-seed population and root
indexes.

For bipartite matching, the key categorical test selects active heads with point-estimate
`D_rel > +0.10` or `< -0.10`, keeps at most the six strongest per direction, and optimally matches
them on `J`; a seed requires three pairs. The continuous all-active-head analysis remains
estimable when this count fails. Heads whose complete 95% interval clears the same threshold form
a separately labelled robustness tier, promoted at population level only with at least eight
pairs across at least three seeds. All of these are CPU-side subsets/statistics of the cached
individual-head causal events.

## Bipartite structural-PE refinement

The focused `graphbench-matching-pe-refinement-v2` experiment is separate from the preceding
score/carriage run. The official GraphBench identifier is `bipartite_matching_hard`, but the
released generator applies maximum-weight matching to ordinary graphs and supplies no
bipartition. The experiment compares RRWP donor copy, RRWP node transposition, complete-PE donor
copy, and complete-PE node transposition on that matching task. Each arm is evaluated with
transport-mass and coherent output-movement scores, matched/mismatch causal patching, `J`/`D_rel`
validation, cancellation, and Taylor-fidelity audits across seeds `0,1,2,3`.

The complete protocol and lockbox are recorded in
[`docs/bipartite_matching_specialisation_rerun_plan.md`](docs/bipartite_matching_specialisation_rerun_plan.md).
Validate local/HPC paths before submission:

```bash
python graphbench-algoreas-hpc/bin/check_official_backends.py --models grit

python graphbench-algoreas-hpc/bin/grit_pe_refinement.py preflight \
  --profile production
```

Submit the common, arm, and model-free refinement-finalizer DAG:

```bash
ENV_ACTIVATE=$PWD/graphbench-algoreas-hpc/activate_graphbench_algoreas \
PROFILE=production \
MAX_PARALLEL=4 \
bash graphbench-algoreas-hpc/bin/submit_grit_pe_refinement.sh
```

If all v1 common components completed but the structural arms failed, reuse only those
donor-law-invariant common products and place all corrected arm tasks into the GPU queue
immediately:

```bash
GPU_STAGE=arms-only \
GRAPHBENCH_ANALYSIS_OUTPUT_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_bipartite_pe_refinement_v1 \
ENV_ACTIVATE=$PWD/graphbench-algoreas-hpc/activate_graphbench_algoreas \
PROFILE=production \
MAX_PARALLEL=4 \
bash graphbench-algoreas-hpc/bin/submit_grit_pe_refinement.sh
```

This mode hard-validates every required common shard and accepts the legacy contract only in the
`common` namespace. It submits one 16-element GPU arm array with no GPU dependency, followed by
the dependent CPU finalizer. Legacy structural-arm caches are always rejected.

The submitter repeats both official-backend and path/cache preflights before calling `sbatch`.
The latter loads the exact registered validation graphs and PE tensors, verifies donor feasibility,
and constructs all four interventions before any job is queued.
Every GPU array element and the one-CPU finalizer have a six-hour safety limit. The common array has 12 elements (three reusable
components by four seeds); the dependent arm array has 16 elements (four interventions by four
seeds). Production starts at 16 score-event graph groups and 24 independently patched heads per
causal forward, with automatic OOM backoff and graph-level resume. Confirmation event caches are
computed blindly, but the confirmation finalizer refuses to read them until one refinement
candidate is explicitly locked:

```bash
python graphbench-algoreas-hpc/bin/grit_pe_refinement.py lock \
  --arm complete_pe_copy \
  --score-system coherent

sbatch -A mlmi-jgg45-sl2-cpu -p sapphire --qos=cpu1 \
  --nodes=1 --ntasks=1 \
  --job-name=gb-pe-confirm \
  --chdir="$PWD" \
  --output="$PWD/graphbench-algoreas-hpc/logs/%x-%j.out" \
  --error="$PWD/graphbench-algoreas-hpc/logs/%x-%j.err" \
  --export=ALL,ENV_ACTIVATE="$PWD/graphbench-algoreas-hpc/activate_graphbench_algoreas",FINALIZE_SPLIT=confirmation \
  graphbench-algoreas-hpc/slurm/grit_pe_refinement_finalize.sbatch
```
