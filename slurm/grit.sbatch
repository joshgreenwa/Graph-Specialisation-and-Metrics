#!/bin/bash
#SBATCH --job-name=gsm-grit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-29
#SBATCH --output=grit-%A-%a.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
OUTPUT_ROOT="${OUTPUT_DIR:-${PROJECT_ROOT}/results/grit}"
if [[ -n "${ENV_ACTIVATE:-}" ]]; then source "${ENV_ACTIVATE}"; fi
cd "${PROJECT_ROOT}"

if (( SLURM_ARRAY_TASK_ID < 15 )); then
  CONFIG="configs/grit_zinc.yaml"
  JOB_INDEX="${SLURM_ARRAY_TASK_ID}"
else
  CONFIG="configs/grit_qm9.yaml"
  JOB_INDEX="$((SLURM_ARRAY_TASK_ID - 15))"
fi

exec gsm train \
  --config "${CONFIG}" \
  --job-index "${JOB_INDEX}" \
  --output-dir "${OUTPUT_ROOT}/job_${SLURM_ARRAY_TASK_ID}"
