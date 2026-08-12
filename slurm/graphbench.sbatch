#!/bin/bash
#SBATCH --job-name=gsm-graphbench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=graphbench-%A-%a.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
OUTPUT_ROOT="${OUTPUT_DIR:-${PROJECT_ROOT}/results/graphbench}"
if [[ -n "${ENV_ACTIVATE:-}" ]]; then source "${ENV_ACTIVATE}"; fi
cd "${PROJECT_ROOT}"
exec gsm train \
  --config configs/graphbench.yaml \
  --job-index "${SLURM_ARRAY_TASK_ID}" \
  --output-dir "${OUTPUT_ROOT}/job_${SLURM_ARRAY_TASK_ID}"
