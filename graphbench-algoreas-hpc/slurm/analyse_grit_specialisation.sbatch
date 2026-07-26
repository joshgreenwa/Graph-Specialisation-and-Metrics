#!/bin/bash
#SBATCH --job-name=gb-grit-worker
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
GRIT_ROOT="${GRIT_ROOT:-${PROJECT_ROOT}/external/GRIT}"
GRAPHBENCH_DATASET_ROOT="${GRAPHBENCH_DATASET_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets}"
GRAPHBENCH_PE_CACHE_ROOT="${GRAPHBENCH_PE_CACHE_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache}"
GRAPHBENCH_TRAINING_OUTPUT_ROOT="${GRAPHBENCH_TRAINING_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1}"
GRAPHBENCH_ANALYSIS_OUTPUT_ROOT="${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_graphbench_edge_v1}"

cd "${PROJECT_ROOT}"
mkdir -p graphbench-algoreas-hpc/logs "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}"

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi
export PROJECT_ROOT GRIT_ROOT
export PYTHONPATH="${PROJECT_ROOT}/src:${GRIT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

ANALYSIS_TASK="${ANALYSIS_TASK:?Set ANALYSIS_TASK to bipartite_matching_hard or flow_hard}"
case "${ANALYSIS_TASK}" in
  bipartite_matching_hard|flow_hard) ;;
  *) echo "Unsupported ANALYSIS_TASK=${ANALYSIS_TASK}" >&2; exit 2 ;;
esac
IFS=':' read -r -a SEED_VALUES <<< "${SEED_LIST:-0:1:2:3}"
if (( SLURM_ARRAY_TASK_ID >= ${#SEED_VALUES[@]} )); then
  echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds SEED_LIST=${SEED_LIST:-0:1:2:3}" >&2
  exit 2
fi
SEED="${SEED_VALUES[${SLURM_ARRAY_TASK_ID}]}"
SEEDS_CSV="$(IFS=,; echo "${SEED_VALUES[*]}")"

python graphbench-algoreas-hpc/bin/check_official_backends.py --models grit

python graphbench-algoreas-hpc/bin/grit_specialisation.py \
  --tasks bipartite_matching_hard,flow_hard \
  --seeds "${SEEDS_CSV}" \
  --worker-task "${ANALYSIS_TASK}" \
  --worker-seed "${SEED}" \
  --phases "${PHASES:-scores,causal,carriage}" \
  --profile "${PROFILE:-production}" \
  --training-output-root "${GRAPHBENCH_TRAINING_OUTPUT_ROOT}" \
  --analysis-output-root "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}" \
  --dataset-root "${GRAPHBENCH_DATASET_ROOT}" \
  --pe-cache-root "${GRAPHBENCH_PE_CACHE_ROOT}" \
  --pe-cache-namespace base_40k4k4k_n64 \
  --accelerator cuda:0 \
  --graphs-per-batch "${GRAPHS_PER_BATCH:-32}" \
  --replica-pair-budget "${REPLICA_PAIR_BUDGET:-2000000}" \
  --jacobian-output-chunk "${JACOBIAN_OUTPUT_CHUNK:-64}" \
  "$@"
