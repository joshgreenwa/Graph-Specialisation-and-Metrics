#!/bin/bash
#SBATCH --job-name=gb-hpc-agg
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gb-hpc-agg-%j.out
#SBATCH --error=logs/gb-hpc-agg-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas-hpc}"
GRAPHBENCH_OUTPUT_ROOT="${GRAPHBENCH_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas-hpc/outputs}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${GRAPHBENCH_OUTPUT_ROOT}"

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

python bin/aggregate_hpc_results.py \
  --output-root "${GRAPHBENCH_OUTPUT_ROOT}"
