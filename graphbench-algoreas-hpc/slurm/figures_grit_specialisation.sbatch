#!/bin/bash
#SBATCH --job-name=gb-grit-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
GRAPHBENCH_ANALYSIS_OUTPUT_ROOT="${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_graphbench_edge_v1}"
GRAPHBENCH_DATASET_ROOT="${GRAPHBENCH_DATASET_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets}"
GRAPHBENCH_PE_CACHE_ROOT="${GRAPHBENCH_PE_CACHE_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache}"
GRAPHBENCH_TRAINING_OUTPUT_ROOT="${GRAPHBENCH_TRAINING_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1}"

cd "${PROJECT_ROOT}"
mkdir -p graphbench-algoreas-hpc/logs

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

python graphbench-algoreas-hpc/bin/grit_specialisation.py \
  --tasks bipartite_matching_hard,flow_hard \
  --seeds "${SEEDS:-0,1,2,3}" \
  --phases figures \
  --profile "${PROFILE:-production}" \
  --analysis-output-root "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}" \
  --training-output-root "${GRAPHBENCH_TRAINING_OUTPUT_ROOT}" \
  --dataset-root "${GRAPHBENCH_DATASET_ROOT}" \
  --pe-cache-root "${GRAPHBENCH_PE_CACHE_ROOT}" \
  --pe-cache-namespace base_40k4k4k_n64 \
  --accelerator cpu \
  "$@"
