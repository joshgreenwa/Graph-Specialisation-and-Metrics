#!/bin/bash
#SBATCH --job-name=gb-hpc-agg
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gb-hpc-agg-%j.out
#SBATCH --error=logs/gb-hpc-agg-%j.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
: "${GRAPHBENCH_OUTPUT_ROOT:?Set GRAPHBENCH_OUTPUT_ROOT to the shared output/checkpoint directory}"

cd "${PROJECT_ROOT}"
mkdir -p logs

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

python experiments/graphbench/hpc/bin/aggregate_hpc_results.py \
  --output-root "${GRAPHBENCH_OUTPUT_ROOT}"
