#!/bin/bash
#SBATCH --job-name=gb-pe-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
GRAPHBENCH_ANALYSIS_OUTPUT_ROOT="${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_bipartite_pe_refinement_v1}"
GRAPHBENCH_DATASET_ROOT="${GRAPHBENCH_DATASET_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets}"
GRAPHBENCH_PE_CACHE_ROOT="${GRAPHBENCH_PE_CACHE_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache}"
GRAPHBENCH_TRAINING_OUTPUT_ROOT="${GRAPHBENCH_TRAINING_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1}"
FINALIZE_SPLIT="${FINALIZE_SPLIT:-refinement}"

if [[ "${FINALIZE_SPLIT}" != "refinement" && "${FINALIZE_SPLIT}" != "confirmation" ]]; then
  echo "FINALIZE_SPLIT must be refinement or confirmation" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p graphbench-algoreas-hpc/logs
if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

python graphbench-algoreas-hpc/bin/grit_pe_refinement.py finalize \
  --split "${FINALIZE_SPLIT}" \
  --profile "${PROFILE:-production}" \
  --analysis-output-root "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}" \
  --training-output-root "${GRAPHBENCH_TRAINING_OUTPUT_ROOT}" \
  --dataset-root "${GRAPHBENCH_DATASET_ROOT}" \
  --pe-cache-root "${GRAPHBENCH_PE_CACHE_ROOT}" \
  --pe-cache-namespace "${GRAPHBENCH_PE_CACHE_NAMESPACE:-base_40k4k4k_n64}" \
  --pe-cache-dtype "${GRAPHBENCH_PE_CACHE_DTYPE:-float32}" \
  --accelerator cpu \
  "$@"
