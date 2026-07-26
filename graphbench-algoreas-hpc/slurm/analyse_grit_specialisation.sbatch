#!/bin/bash
#SBATCH --job-name=gb-grit-spec
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/gb-grit-spec-%A-%a.out
#SBATCH --error=logs/gb-grit-spec-%A-%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
GRAPHBENCH_DATASET_ROOT="${GRAPHBENCH_DATASET_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets}"
GRAPHBENCH_PE_CACHE_ROOT="${GRAPHBENCH_PE_CACHE_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache}"
GRAPHBENCH_TRAINING_OUTPUT_ROOT="${GRAPHBENCH_TRAINING_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1}"
GRAPHBENCH_ANALYSIS_OUTPUT_ROOT="${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_graphbench_edge_v1}"

cd "${PROJECT_ROOT}"
mkdir -p graphbench-algoreas-hpc/logs "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}"

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

TASKS=(bipartite_matching_hard flow_hard)
TASK="${TASKS[${SLURM_ARRAY_TASK_ID}]}"

python graphbench-algoreas-hpc/bin/check_official_backends.py --models grit

python graphbench-algoreas-hpc/bin/grit_specialisation.py \
  --tasks "${TASK}" \
  --seeds "${SEEDS:-0,1,2}" \
  --phases "${PHASES:-scores,causal,carriage,figures}" \
  --profile "${PROFILE:-production}" \
  --training-output-root "${GRAPHBENCH_TRAINING_OUTPUT_ROOT}" \
  --analysis-output-root "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}" \
  --dataset-root "${GRAPHBENCH_DATASET_ROOT}" \
  --pe-cache-root "${GRAPHBENCH_PE_CACHE_ROOT}" \
  --pe-cache-namespace base_40k4k4k_n64 \
  --accelerator cuda:0 \
  "$@"
