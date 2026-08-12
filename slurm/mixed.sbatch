#!/bin/bash
#SBATCH --job-name=gsm-mixed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=mixed-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/mixed}"
if [[ -n "${ENV_ACTIVATE:-}" ]]; then source "${ENV_ACTIVATE}"; fi
cd "${PROJECT_ROOT}"
exec gsm run \
  --config configs/mixed.yaml \
  --output-dir "${OUTPUT_DIR}"
