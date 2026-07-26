#!/bin/bash
#SBATCH --job-name=gb-grit-fig
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/gb-grit-fig-%A-%a.out
#SBATCH --error=logs/gb-grit-fig-%A-%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
GRAPHBENCH_ANALYSIS_OUTPUT_ROOT="${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_graphbench_edge_v1}"

cd "${PROJECT_ROOT}"
mkdir -p graphbench-algoreas-hpc/logs

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

TASKS=(bipartite_matching_hard flow_hard)
TASK="${TASKS[${SLURM_ARRAY_TASK_ID}]}"

python graphbench-algoreas-hpc/bin/grit_specialisation.py \
  --tasks "${TASK}" \
  --seeds "${SEEDS:-0,1,2}" \
  --phases figures \
  --profile "${PROFILE:-production}" \
  --analysis-output-root "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}" \
  --accelerator cpu \
  "$@"
