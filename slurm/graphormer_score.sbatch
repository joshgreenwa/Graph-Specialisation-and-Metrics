#!/bin/bash
#SBATCH --job-name=gsm-graphormer
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=graphormer-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/graphormer_pcqm4mv2}"
if [[ -n "${ENV_ACTIVATE:-}" ]]; then source "${ENV_ACTIVATE}"; fi
cd "${PROJECT_ROOT}"
exec gsm score \
  --config configs/graphormer_pcqm4mv2.yaml \
  --output-dir "${OUTPUT_DIR}"
