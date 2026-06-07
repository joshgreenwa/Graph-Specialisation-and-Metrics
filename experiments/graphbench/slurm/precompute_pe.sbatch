#!/bin/bash
#SBATCH --job-name=gb-pe-cache
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gb-pe-cache-%j.out
#SBATCH --error=logs/gb-pe-cache-%j.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the GraphBench dataset/cache directory}"
: "${PE_CACHE_ROOT:?Set PE_CACHE_ROOT to a large shared PE cache directory}"

cd "${PROJECT_ROOT}"
mkdir -p logs

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

python experiments/graphbench/training/official_algoreas_screen_colab.py \
  --tasks bipartite_matching_hard,flow_hard,mst_hard,maxclique_hard,bridges_hard \
  --precompute-pe-only \
  --dataset-root "${DATASET_ROOT}" \
  --pe-cache-root "${PE_CACHE_ROOT}" \
  --pe-cache-dtype float32 \
  --device cpu \
  --allow-cpu \
  --no-mount-drive
