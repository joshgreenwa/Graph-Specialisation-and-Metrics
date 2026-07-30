# GraphBench AlgoReas HPC State

This note records the current confirmed HPC state for the compact GraphBench
AlgoReas runs and mechanistic-analysis workflow.

## Key Locations

Repo:

```text
/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics
```

HPC runner:

```text
/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics/graphbench-algoreas-hpc/bin/algoreas_hpc.py
```

Mechanistic analysis:

```text
/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics/src/graph_specialisation_metrics/mechanistic_operator_analysis.py
```

Training outputs:

```text
/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1
```

PE cache:

```text
/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache/hpc_base_v1_5task_5k_pe_cache_matched_params/base_40k4k4k_n64
```

## Environment

Activate conda on login or compute nodes:

```bash
source /usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh
conda activate graphbench-algoreas
```

For Slurm jobs, create this once from the HPC runner directory:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics/graphbench-algoreas-hpc

cat > activate_graphbench_algoreas <<'EOF'
source /usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh
conda activate graphbench-algoreas
export PROJECT_ROOT=/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics
export GRIT_ROOT=$PROJECT_ROOT/external/GRIT
export PYTHONPATH="$PROJECT_ROOT/src:$GRIT_ROOT:$PROJECT_ROOT/external/GNNPlus:${PYTHONPATH:-}"
export GRAPHBENCH_DATASET_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets
export GRAPHBENCH_PE_CACHE_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache
export GRAPHBENCH_OUTPUT_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs
EOF
```

## Training Protocol

Compact hard-OOD protocol:

| split | graphs | nodes |
|---|---:|---:|
| train | 40,000 | 16 |
| val | 4,000 | 16 |
| test | 4,000 | 64 |

Other defaults:

- Seeds: `0,1,2,3`
- Backend: official-backed model components integrated into our GraphBench runner
- Bipartite primary metric: F1, higher is better
- Flow primary metric: MAE, lower is better

## Confirmed Trained Models

| task | model | seeds | params |
|---|---|---:|---:|
| `bipartite_matching_hard` | `grit` | `0-3` | `956,373` |
| `bipartite_matching_hard` | `gcn_plus` | `0-3` | `2,163,173` |
| `flow_hard` | `grit` | `0-3` | `956,373` |
| `flow_hard` | `gcn_plus` | `0-3` | `2,163,173` |

Queued or awaiting completion:

| task | model | seeds | status |
|---|---|---:|---|
| `maxclique_hard` | `grit` | `0-2` | GPU training queued/pending after PE completion |
| `maxclique_hard` | `gcn_plus` | `0-2` | GPU training queued/pending after PE completion |

GRIT and GCN+ are official-backed: upstream official model components/layers are
used, while the GraphBench data wrapper, prediction heads, compact training
protocol, and evaluation harness are implemented in this repository.

## Performance Summary

| task | model | train primary mean±sd | val primary mean±sd | test primary mean±sd |
|---|---|---:|---:|---:|
| `bipartite_matching_hard` | `grit` | `0.9797±0.0036` | `0.9564±0.0023` | `0.3390±0.1084` |
| `bipartite_matching_hard` | `gcn_plus` | `0.7035±0.0837` | `0.6998±0.0808` | `0.2960±0.0847` |
| `flow_hard` | `grit` | `0.0810±0.0024` | `0.1078±0.0019` | `6.1056±0.9187` |
| `flow_hard` | `gcn_plus` | `0.6759±0.0249` | `0.7492±0.0090` | `7.7519±1.2002` |

## Checkpoints

Best checkpoint path pattern:

```text
/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1/{task}/{model}/seed{seed}/best.pt
```

Example:

```text
/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1/flow_hard/grit/seed1/best.pt
```

## PE Cache Status

Last verified after PE job `32159444_3`:

| task | train | val | test |
|---|---|---|---|
| `bipartite_matching_hard` | complete | complete | complete |
| `flow_hard` | complete | complete | complete |
| `mst_hard` | complete | complete | complete |
| `maxclique_hard` | complete | complete | complete |

Verify current PE cache state:

```bash
find /rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache -type f -name "*.pt" -print | sort
```

Summarise cached task/split files:

```bash
find /rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache -type f -name "*.pt" -printf "%f\n" \
  | sed -E 's/_(train|val|test)_graphs[0-9]+_nodes[0-9]+_.*/ \1/' \
  | sort | uniq -c
```

## Useful Verification Commands

List trained summaries:

```bash
find /rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1 \
  -path "*/seed*/summary.json" -print | sort
```

Aggregate run summaries:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics

python graphbench-algoreas-hpc/bin/aggregate_hpc_results.py \
  --output-root /rds/user/jgg45/hpc-work/graphbench-algoreas/outputs \
  --run-name graphbench_algoreas_hpc_base_v1
```

## Working Slurm Patterns

Move to the HPC runner directory:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics/graphbench-algoreas-hpc
mkdir -p logs
```

Resume PE cache for `maxclique_hard` if it is ever missing or partial:

```bash
sbatch -A mlmi-jgg45-sl2-cpu -p sapphire --qos=cpu1 \
  --array=3 --cpus-per-task=2 --mem=16G --time=02:00:00 \
  --export=ALL,ENV_ACTIVATE=$PWD/activate_graphbench_algoreas,GRAPHBENCH_PE_WORKERS=1,GRAPHBENCH_PE_SAVE_EVERY=250 \
  slurm/precompute_pe.sbatch
```

Train `mst_hard` for GRIT and GCN+, seeds `0,1,2`, one GPU job at a time:

```bash
sbatch -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
  --nodes=1 --ntasks=1 --gres=gpu:1 \
  --array=0-5%1 \
  --export=ALL,ENV_ACTIVATE=$PWD/activate_graphbench_algoreas,TASKS=mst_hard,MODELS=grit,gcn_plus,SEEDS=0,1,2,WANDB_MODE=online \
  slurm/train_base_array.sbatch
```

Train `maxclique_hard` for GRIT and GCN+, seeds `0,1,2`, one GPU job at a time:

```bash
sbatch -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
  --nodes=1 --ntasks=1 --gres=gpu:1 \
  --array=0-5%1 \
  --export=ALL,ENV_ACTIVATE=$PWD/activate_graphbench_algoreas,TASKS=maxclique_hard,MODELS=grit,gcn_plus,SEEDS=0,1,2,WANDB_MODE=online \
  slurm/train_base_array.sbatch --final-eval
```

Train with dependency on a PE job, using the PE array master id:

```bash
sbatch -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
  --nodes=1 --ntasks=1 --gres=gpu:1 \
  --array=0-5%1 --dependency=afterok:${PE_JOBID} \
  --export=ALL,ENV_ACTIVATE=$PWD/activate_graphbench_algoreas,TASKS=maxclique_hard,MODELS=grit,gcn_plus,SEEDS=0,1,2,WANDB_MODE=online \
  slurm/train_base_array.sbatch --final-eval
```

Check queued/running maxclique training:

```bash
squeue -u jgg45 -o "%.18i %.9P %.24j %.8T %.10M %.6D %R" | grep -E "gb-hpc-train|JOBID"
```

Check logs:

```bash
ls -lh logs/*JOBID*
tail -200 logs/*JOBID*.err
tail -200 logs/*JOBID*.out
```

Check Slurm accounting:

```bash
sacct -j JOBID --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES%40
```

## GRIT Specialisation and Carriage Launch

The canonical GraphBench analysis uses the `expansion/graphormer_specialisation` branch, official
GRIT commit `6c988ea600a606fbb49a2246c64a2d37396b3ab5`, all four trained seeds, and the
`n=16` validation split. Do not update the checkout between starting a production worker and
finishing its dependent finalizer: cache contracts bind the repository commit.

Refresh an existing clean checkout:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics
git status --short
git fetch origin
git switch expansion/graphormer_specialisation
git pull --ff-only origin expansion/graphormer_specialisation
git rev-parse HEAD
```

If that checkout has local changes or cannot fast-forward, preserve it and make a fresh clone:

```bash
cd /rds/user/jgg45/hpc-work
STAMP="$(date +%Y%m%d-%H%M%S)"
mv Graph-Specialisation-and-Metrics "Graph-Specialisation-and-Metrics.backup-${STAMP}"
git clone --branch expansion/graphormer_specialisation --single-branch \
  https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git \
  Graph-Specialisation-and-Metrics
cd Graph-Specialisation-and-Metrics
mkdir -p external
git clone https://github.com/LiamMa/GRIT.git external/GRIT
git -C external/GRIT checkout 6c988ea600a606fbb49a2246c64a2d37396b3ab5
```

For a fresh clone, recreate `graphbench-algoreas-hpc/activate_graphbench_algoreas` using the
environment block near the top of this document. Then run the preflight and verify the four
matching checkpoints:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics
source graphbench-algoreas-hpc/activate_graphbench_algoreas
python graphbench-algoreas-hpc/bin/check_official_backends.py --models grit

for TASK in bipartite_matching_hard; do
  for SEED in 0 1 2 3; do
    test -s "/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1/${TASK}/grit/seed${SEED}/best.pt"
  done
done
```

Run an isolated smoke submission first. Its output root must differ from production because cache
contracts are immutable:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics

ENV_ACTIVATE=$PWD/graphbench-algoreas-hpc/activate_graphbench_algoreas \
PROFILE=smoke \
ANALYSIS_TASKS=bipartite_matching_hard \
PHASES=scores,causal \
MAX_PARALLEL=1 \
GRAPHBENCH_ANALYSIS_OUTPUT_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_graphbench_complete_pe_v2_smoke \
bash graphbench-algoreas-hpc/bin/submit_grit_specialisation.sh
```

After the smoke finalizer succeeds, submit production:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics

ENV_ACTIVATE=$PWD/graphbench-algoreas-hpc/activate_graphbench_algoreas \
PROFILE=production \
ANALYSIS_TASKS=bipartite_matching_hard \
PHASES=scores,causal \
MAX_PARALLEL=4 \
GRAPHBENCH_ANALYSIS_OUTPUT_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_graphbench_complete_pe_v2 \
bash graphbench-algoreas-hpc/bin/submit_grit_specialisation.sh
```

The focused launcher submits one `0-3%4` matching array, then one CPU finalizer with an `afterok`
dependency on the complete array. Set
`ANALYSIS_TASKS=bipartite_matching_hard,flow_hard` only when both tasks are intended. Every submission
explicitly requests `--nodes=1 --ntasks=1 --gres=gpu:1`, as required by the Cambridge site
wrapper; the `ampere` partition selects the GPU class. Each seed worker requests six hours and
can be resubmitted against the same contract-bound output root to resume completed shards.
Production workers use 32 base graphs per batch, a 2,000,000 dense-replica budget, and
64-output matching VJPs. CUDA OOM automatically halves graph batches and keeps the stable smaller
size. `progress.jsonl` and heartbeat log lines expose utilization, VRAM, and power for live tuning.
The launcher exports the four seeds internally as colon-separated `SEED_LIST=0:1:2:3`; commas
cannot be embedded directly in Slurm's `--export` list because Slurm treats them as variable
separators. It also refuses submission unless `GRIT_ROOT` is a Git checkout at the pinned commit.

Monitor the matching array and CPU-finalizer job IDs printed by the launcher:

```bash
squeue -u jgg45 -o "%.18i %.9P %.24j %.8T %.10M %.6D %R" \
  | grep -E "gb-grit-match|gb-grit-finalize|JOBID"
tail -f graphbench-algoreas-hpc/logs/gb-grit-match-*.out
```

After completion, verify the finalizer and four-seed population summaries:

```bash
ANALYSIS_ROOT=/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_graphbench_complete_pe_v2
test -s "${ANALYSIS_ROOT}/index.json"
test -s "${ANALYSIS_ROOT}/graphbench_bipartite_matching_hard/population.json"
find "${ANALYSIS_ROOT}" -path "*/seed_*/figures.json" -print | sort
```

## Mechanistic Analysis

Primary script:

```text
/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics/src/graph_specialisation_metrics/mechanistic_operator_analysis.py
```

The script supports:

- operator-transport atlas
- mechanism knockouts
- metric-ranked causal ablations
- graph-level operator specificity against matched random controls
- compact crux dashboard figures

Example flow run on validation graphs:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics

sbatch -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
  --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=80G --time=06:00:00 \
  -J grit-flow-mech512 \
  -o logs/grit-flow-mech512-%j.out -e logs/grit-flow-mech512-%j.err \
  --wrap 'source /usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh; conda activate graphbench-algoreas; export PYTHONPATH=$PWD/src:$PYTHONPATH; python src/graph_specialisation_metrics/mechanistic_operator_analysis.py --checkpoint /rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1/flow_hard/grit/seed1/best.pt --task flow_hard --model grit --model-backend official --experiments all --split val --num-graphs 512 --batch-size 8 --eval-batch-size 16 --device cuda --baseline-primary 7.7519 --output-dir /rds/user/jgg45/hpc-work/graphbench-algoreas/metrics/grit_flow_seed1_val512_mechanistic_full'
```

View figures from VSCode:

```bash
code /rds/user/jgg45/hpc-work/graphbench-algoreas/metrics/grit_flow_seed1_val512_mechanistic_full/figures
```
