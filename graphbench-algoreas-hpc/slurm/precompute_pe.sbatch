#!/bin/bash
#SBATCH --job-name=gb-hpc-pe
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%1
#SBATCH --output=logs/gb-hpc-pe-%A-%a.out
#SBATCH --error=logs/gb-hpc-pe-%A-%a.err

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

TASKS=(bipartite_matching_hard flow_hard mst_hard maxclique_hard bridges_hard)
TASK="${TASKS[${SLURM_ARRAY_TASK_ID}]}"

python bin/algoreas_hpc.py \
  --tasks "${TASK}" \
  --precompute-pe-only \
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
  --pe-workers "${GRAPHBENCH_PE_WORKERS:-1}" \
  --pe-save-every "${GRAPHBENCH_PE_SAVE_EVERY:-250}" \
  --device cpu \
  --allow-cpu \
  --num-workers 0 \
  --no-pin-memory
