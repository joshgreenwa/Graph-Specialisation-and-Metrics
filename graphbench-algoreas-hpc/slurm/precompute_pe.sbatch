#!/bin/bash
#SBATCH --job-name=gb-hpc-pe
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%1
#SBATCH --output=logs/gb-hpc-pe-%A-%a.out
#SBATCH --error=logs/gb-hpc-pe-%A-%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas-hpc}"
GRAPHBENCH_DATASET_ROOT="${GRAPHBENCH_DATASET_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas-hpc/datasets}"
GRAPHBENCH_PE_CACHE_ROOT="${GRAPHBENCH_PE_CACHE_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas-hpc/pe_cache}"
GRAPHBENCH_OUTPUT_ROOT="${GRAPHBENCH_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas-hpc/outputs}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${GRAPHBENCH_DATASET_ROOT}" "${GRAPHBENCH_PE_CACHE_ROOT}" "${GRAPHBENCH_OUTPUT_ROOT}"

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

TASKS=(bipartite_matching_hard flow_hard mst_hard maxclique_hard bridges_hard)
TASK="${TASKS[${SLURM_ARRAY_TASK_ID}]}"

python bin/algoreas_hpc.py \
  --tasks "${TASK}" \
  --precompute-pe-only \
  --dataset-root "${GRAPHBENCH_DATASET_ROOT}" \
  --pe-cache-root "${GRAPHBENCH_PE_CACHE_ROOT}" \
  --pe-cache-namespace base \
  --output-root "${GRAPHBENCH_OUTPUT_ROOT}" \
  --pe-cache-dtype float32 \
  --device cpu \
  --allow-cpu \
  --num-workers 0 \
  --no-pin-memory
