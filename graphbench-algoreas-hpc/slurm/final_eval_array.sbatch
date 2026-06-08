#!/bin/bash
#SBATCH --job-name=gb-hpc-eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-99%1
#SBATCH --output=logs/gb-hpc-eval-%A-%a.out
#SBATCH --error=logs/gb-hpc-eval-%A-%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics/graphbench-algoreas-hpc}"
GRAPHBENCH_DATASET_ROOT="${GRAPHBENCH_DATASET_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets}"
GRAPHBENCH_PE_CACHE_ROOT="${GRAPHBENCH_PE_CACHE_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache}"
GRAPHBENCH_OUTPUT_ROOT="${GRAPHBENCH_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${GRAPHBENCH_DATASET_ROOT}" "${GRAPHBENCH_PE_CACHE_ROOT}" "${GRAPHBENCH_OUTPUT_ROOT}"

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

MODELS="${MODELS:-static_grit,grit,gatedgcn_plus,gin_plus,gcn_plus}"
TASKS="${TASKS:-bipartite_matching_hard,flow_hard,mst_hard,maxclique_hard,bridges_hard}"
SEEDS="${SEEDS:-0,1,2,3}"

python bin/check_official_backends.py --models "${MODELS}"

python bin/algoreas_hpc.py \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --tasks "${TASKS}" \
  --models "${MODELS}" \
  --seeds "${SEEDS}" \
  --model-backend official \
  --dataset-root "${GRAPHBENCH_DATASET_ROOT}" \
  --pe-cache-root "${GRAPHBENCH_PE_CACHE_ROOT}" \
  --pe-cache-namespace base_40k4k4k_n64 \
  --output-root "${GRAPHBENCH_OUTPUT_ROOT}" \
  --train-size 40000 \
  --val-size 4000 \
  --test-size 4000 \
  --train-node-size 16 \
  --val-node-size 16 \
  --test-node-size 64 \
  --pe-cache-dtype float32 \
  --no-build-missing-pe-cache \
  --require-pe-cache \
  --final-eval \
  --num-workers "${GRAPHBENCH_NUM_WORKERS:-4}" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --wandb-project "${WANDB_PROJECT:-graphbench-algoreas-hpc}" \
  --wandb-tags "${WANDB_TAGS:-base,hpc,final_eval}" \
  "$@"
