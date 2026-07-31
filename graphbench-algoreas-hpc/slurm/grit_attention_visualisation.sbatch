#!/bin/bash
#SBATCH --job-name=gb-grit-attn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
GRIT_ROOT="${GRIT_ROOT:-${PROJECT_ROOT}/external/GRIT}"
GRAPHBENCH_ANALYSIS_OUTPUT_ROOT="${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_bipartite_complete_pe_causal_v3}"
GRAPHBENCH_DATASET_ROOT="${GRAPHBENCH_DATASET_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets}"
GRAPHBENCH_PE_CACHE_ROOT="${GRAPHBENCH_PE_CACHE_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache}"
GRAPHBENCH_TRAINING_OUTPUT_ROOT="${GRAPHBENCH_TRAINING_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1}"

cd "${PROJECT_ROOT}"
mkdir -p graphbench-algoreas-hpc/logs

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi
export PROJECT_ROOT GRIT_ROOT
export PYTHONPATH="${PROJECT_ROOT}/src:${GRIT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

python graphbench-algoreas-hpc/bin/grit_specialisation.py \
  --tasks bipartite_matching_hard \
  --seeds "${ATTENTION_SEED:-0}" \
  --phases figures \
  --profile production \
  --analysis-output-root "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}" \
  --training-output-root "${GRAPHBENCH_TRAINING_OUTPUT_ROOT}" \
  --dataset-root "${GRAPHBENCH_DATASET_ROOT}" \
  --pe-cache-root "${GRAPHBENCH_PE_CACHE_ROOT}" \
  --pe-cache-namespace base_40k4k4k_n64 \
  --accelerator cpu \
  --attention-visualisation \
  --attention-seed "${ATTENTION_SEED:-0}" \
  --attention-graph "${ATTENTION_GRAPH_ID:-0}" \
  --attention-top-k "${ATTENTION_TOP_K:-2}" \
  "$@"
