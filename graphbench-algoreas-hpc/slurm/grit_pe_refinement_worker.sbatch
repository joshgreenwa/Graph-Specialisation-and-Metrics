#!/bin/bash
#SBATCH --job-name=gb-pe-refine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
GRIT_ROOT="${GRIT_ROOT:-${PROJECT_ROOT}/external/GRIT}"
GRAPHBENCH_DATASET_ROOT="${GRAPHBENCH_DATASET_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets}"
GRAPHBENCH_PE_CACHE_ROOT="${GRAPHBENCH_PE_CACHE_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache}"
GRAPHBENCH_TRAINING_OUTPUT_ROOT="${GRAPHBENCH_TRAINING_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1}"
GRAPHBENCH_ANALYSIS_OUTPUT_ROOT="${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_bipartite_pe_refinement_v1}"

cd "${PROJECT_ROOT}"
mkdir -p graphbench-algoreas-hpc/logs "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}"

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi
export PROJECT_ROOT GRIT_ROOT
export PYTHONPATH="${PROJECT_ROOT}/src:${GRIT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

IFS=':' read -r -a SEED_VALUES <<< "${SEED_LIST:-0:1:2:3}"
if (( ${#SEED_VALUES[@]} != 4 )); then
  echo "SEED_LIST must contain exactly 0:1:2:3-compatible four entries" >&2
  exit 2
fi

WORKER_GROUP="${WORKER_GROUP:?Set WORKER_GROUP=common or arm}"
if [[ "${WORKER_GROUP}" == "common" ]]; then
  COMPONENTS=(common-scores common-causal common-ablation)
  COMPONENT_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
  SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
  if (( COMPONENT_INDEX >= ${#COMPONENTS[@]} )); then
    echo "Common array index ${SLURM_ARRAY_TASK_ID} is outside 0-11" >&2
    exit 2
  fi
  COMPONENT="${COMPONENTS[${COMPONENT_INDEX}]}"
  ARM_ARGS=()
elif [[ "${WORKER_GROUP}" == "arm" ]]; then
  ARMS=(rrwp_copy rrwp_transpose complete_pe_copy complete_pe_transpose)
  ARM_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
  SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
  if (( ARM_INDEX >= ${#ARMS[@]} )); then
    echo "Arm array index ${SLURM_ARRAY_TASK_ID} is outside 0-15" >&2
    exit 2
  fi
  COMPONENT="arm"
  ARM_ARGS=(--arm "${ARMS[${ARM_INDEX}]}")
else
  echo "Unsupported WORKER_GROUP=${WORKER_GROUP}" >&2
  exit 2
fi
SEED="${SEED_VALUES[${SEED_INDEX}]}"

python graphbench-algoreas-hpc/bin/check_official_backends.py --models grit

python graphbench-algoreas-hpc/bin/grit_pe_refinement.py worker \
  --component "${COMPONENT}" \
  --seed "${SEED}" \
  "${ARM_ARGS[@]}" \
  --profile "${PROFILE:-production}" \
  --training-output-root "${GRAPHBENCH_TRAINING_OUTPUT_ROOT}" \
  --analysis-output-root "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}" \
  --dataset-root "${GRAPHBENCH_DATASET_ROOT}" \
  --pe-cache-root "${GRAPHBENCH_PE_CACHE_ROOT}" \
  --pe-cache-namespace "${GRAPHBENCH_PE_CACHE_NAMESPACE:-base_40k4k4k_n64}" \
  --pe-cache-dtype "${GRAPHBENCH_PE_CACHE_DTYPE:-float32}" \
  --accelerator cuda:0 \
  --graphs-per-batch "${GRAPHS_PER_BATCH:-16}" \
  --head-batch-size "${HEAD_BATCH_SIZE:-24}" \
  --replica-pair-budget "${REPLICA_PAIR_BUDGET:-2000000}" \
  --jacobian-output-chunk "${JACOBIAN_OUTPUT_CHUNK:-64}" \
  "$@"
