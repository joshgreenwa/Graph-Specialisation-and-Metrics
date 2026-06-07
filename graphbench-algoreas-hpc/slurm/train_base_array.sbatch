#!/bin/bash
#SBATCH --job-name=gb-hpc-train
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-139%1
#SBATCH --output=logs/gb-hpc-train-%A-%a.out
#SBATCH --error=logs/gb-hpc-train-%A-%a.err

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

python bin/algoreas_hpc.py \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --dataset-root "${GRAPHBENCH_DATASET_ROOT}" \
  --pe-cache-root "${GRAPHBENCH_PE_CACHE_ROOT}" \
  --pe-cache-namespace base \
  --output-root "${GRAPHBENCH_OUTPUT_ROOT}" \
  --pe-cache-dtype float32 \
  --no-build-missing-pe-cache \
  --require-pe-cache \
  --num-workers "${GRAPHBENCH_NUM_WORKERS:-4}"
