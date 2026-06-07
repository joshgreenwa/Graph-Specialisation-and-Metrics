#!/bin/bash
#SBATCH --job-name=gb-eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --array=0-139%1
#SBATCH --output=logs/gb-eval-%A-%a.out
#SBATCH --error=logs/gb-eval-%A-%a.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the GraphBench dataset/cache directory}"
: "${PE_CACHE_ROOT:?Set PE_CACHE_ROOT to the precomputed PE cache directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the run output/checkpoint directory}"

cd "${PROJECT_ROOT}"
mkdir -p logs

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

python experiments/graphbench/training/official_algoreas_screen_colab.py \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --dataset-root "${DATASET_ROOT}" \
  --pe-cache-root "${PE_CACHE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --no-build-missing-pe-cache \
  --require-pe-cache \
  --final-eval \
  --no-mount-drive
