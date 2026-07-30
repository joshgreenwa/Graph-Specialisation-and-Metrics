#!/bin/bash
# Submit the bipartite-only GRIT PE refinement as common, arm, and CPU-finalizer stages.

set -euo pipefail

HPC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${HPC_ROOT}/.." && pwd)"
ENV_ACTIVATE="${ENV_ACTIVATE:-${HPC_ROOT}/activate_graphbench_algoreas}"
PROFILE="${PROFILE:-production}"
ANALYSIS_SPLIT="${ANALYSIS_SPLIT:-refinement}"
SEEDS="${SEEDS:-0,1,2,3}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
COMMON_ARRAY="${COMMON_ARRAY:-0-11}"
GRAPHBENCH_ANALYSIS_OUTPUT_ROOT="${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_bipartite_pe_refinement_v1}"
GRAPHBENCH_TRAINING_OUTPUT_ROOT="${GRAPHBENCH_TRAINING_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1}"
GRAPHBENCH_DATASET_ROOT="${GRAPHBENCH_DATASET_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets}"
GRAPHBENCH_PE_CACHE_ROOT="${GRAPHBENCH_PE_CACHE_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache}"
GRAPHBENCH_PE_CACHE_NAMESPACE="${GRAPHBENCH_PE_CACHE_NAMESPACE:-base_40k4k4k_n64}"
GRAPHBENCH_PE_CACHE_DTYPE="${GRAPHBENCH_PE_CACHE_DTYPE:-float32}"
GRAPHS_PER_BATCH="${GRAPHS_PER_BATCH:-16}"
HEAD_BATCH_SIZE="${HEAD_BATCH_SIZE:-24}"
REPLICA_PAIR_BUDGET="${REPLICA_PAIR_BUDGET:-2000000}"
JACOBIAN_OUTPUT_CHUNK="${JACOBIAN_OUTPUT_CHUNK:-64}"
GRIT_ROOT="${GRIT_ROOT:-${PROJECT_ROOT}/external/GRIT}"
EXPECTED_GRIT_COMMIT="6c988ea600a606fbb49a2246c64a2d37396b3ab5"

if [[ ! -f "${ENV_ACTIVATE}" ]]; then
  echo "Missing environment activation file: ${ENV_ACTIVATE}" >&2
  exit 2
fi
if [[ ! -d "${GRIT_ROOT}/.git" ]]; then
  echo "Missing official GRIT Git checkout: ${GRIT_ROOT}" >&2
  exit 2
fi
OBSERVED_GRIT_COMMIT="$(git -C "${GRIT_ROOT}" rev-parse HEAD)"
if [[ "${OBSERVED_GRIT_COMMIT}" != "${EXPECTED_GRIT_COMMIT}" ]]; then
  echo "Official GRIT commit is ${OBSERVED_GRIT_COMMIT}; expected ${EXPECTED_GRIT_COMMIT}" >&2
  exit 2
fi
if [[ "${PROFILE}" != "production" && "${PROFILE}" != "smoke" ]]; then
  echo "PROFILE must be production or smoke" >&2
  exit 2
fi
if [[ "${ANALYSIS_SPLIT}" != "refinement" ]]; then
  echo "The four-arm selection launcher is refinement-only" >&2
  exit 2
fi
if [[ ! "${MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_PARALLEL must be a positive integer" >&2
  exit 2
fi
if [[ "${COMMON_ARRAY}" != "0-11" && "${COMMON_ARRAY}" != "4-7" ]]; then
  echo "COMMON_ARRAY must be 0-11 (full run) or 4-7 (causal recovery)" >&2
  exit 2
fi
for VALUE_NAME in GRAPHS_PER_BATCH HEAD_BATCH_SIZE REPLICA_PAIR_BUDGET JACOBIAN_OUTPUT_CHUNK; do
  VALUE="${!VALUE_NAME}"
  if [[ ! "${VALUE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${VALUE_NAME} must be a positive integer" >&2
    exit 2
  fi
done
IFS=',' read -r -a SEED_VALUES <<< "${SEEDS}"
if [[ "${SEEDS}" != "0,1,2,3" ]] || (( ${#SEED_VALUES[@]} != 4 )); then
  echo "This registered launcher requires SEEDS=0,1,2,3" >&2
  exit 2
fi
SEED_LIST="${SEEDS//,/:}"
PREFLIGHT_ARGS=()
if [[ "${COMMON_ARRAY}" == "4-7" ]]; then
  PREFLIGHT_ARGS+=(--require-causal-recovery-prerequisites)
fi

for SEED in "${SEED_VALUES[@]}"; do
  CHECKPOINT="${GRAPHBENCH_TRAINING_OUTPUT_ROOT}/bipartite_matching_hard/grit/seed${SEED}/best.pt"
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 2
  fi
done

cd "${PROJECT_ROOT}"
mkdir -p graphbench-algoreas-hpc/logs "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}"

(
  # Fail on the login node before spending queue time on a missing package, path, or cache.
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
  export PROJECT_ROOT GRIT_ROOT
  export PYTHONPATH="${PROJECT_ROOT}/src:${GRIT_ROOT}:${PYTHONPATH:-}"
  python graphbench-algoreas-hpc/bin/check_official_backends.py --models grit
  python graphbench-algoreas-hpc/bin/grit_pe_refinement.py preflight \
    --profile "${PROFILE}" \
    --analysis-output-root "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}" \
    --training-output-root "${GRAPHBENCH_TRAINING_OUTPUT_ROOT}" \
    --dataset-root "${GRAPHBENCH_DATASET_ROOT}" \
    --pe-cache-root "${GRAPHBENCH_PE_CACHE_ROOT}" \
    --pe-cache-namespace "${GRAPHBENCH_PE_CACHE_NAMESPACE}" \
    --pe-cache-dtype "${GRAPHBENCH_PE_CACHE_DTYPE}" \
    --accelerator cpu \
    --graphs-per-batch "${GRAPHS_PER_BATCH}" \
    --head-batch-size "${HEAD_BATCH_SIZE}" \
    --replica-pair-budget "${REPLICA_PAIR_BUDGET}" \
    --jacobian-output-chunk "${JACOBIAN_OUTPUT_CHUNK}" \
    "${PREFLIGHT_ARGS[@]}"
)

COMMON_EXPORT="ALL,PROJECT_ROOT=${PROJECT_ROOT},ENV_ACTIVATE=${ENV_ACTIVATE},PROFILE=${PROFILE},ANALYSIS_SPLIT=${ANALYSIS_SPLIT},FINALIZE_SPLIT=${ANALYSIS_SPLIT},SEED_LIST=${SEED_LIST},GRIT_ROOT=${GRIT_ROOT},GRAPHBENCH_ANALYSIS_OUTPUT_ROOT=${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT},GRAPHBENCH_TRAINING_OUTPUT_ROOT=${GRAPHBENCH_TRAINING_OUTPUT_ROOT},GRAPHBENCH_DATASET_ROOT=${GRAPHBENCH_DATASET_ROOT},GRAPHBENCH_PE_CACHE_ROOT=${GRAPHBENCH_PE_CACHE_ROOT},GRAPHBENCH_PE_CACHE_NAMESPACE=${GRAPHBENCH_PE_CACHE_NAMESPACE},GRAPHBENCH_PE_CACHE_DTYPE=${GRAPHBENCH_PE_CACHE_DTYPE},GRAPHS_PER_BATCH=${GRAPHS_PER_BATCH},HEAD_BATCH_SIZE=${HEAD_BATCH_SIZE},REPLICA_PAIR_BUDGET=${REPLICA_PAIR_BUDGET},JACOBIAN_OUTPUT_CHUNK=${JACOBIAN_OUTPUT_CHUNK}"

COMMON_JOB="$(
  sbatch --parsable \
    -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
    --nodes=1 --ntasks=1 --gres=gpu:1 \
    --array="${COMMON_ARRAY}%${MAX_PARALLEL}" \
    --job-name=gb-pe-common \
    --chdir="${PROJECT_ROOT}" \
    --output="${HPC_ROOT}/logs/%x-%A-%a.out" \
    --error="${HPC_ROOT}/logs/%x-%A-%a.err" \
    --export="${COMMON_EXPORT},WORKER_GROUP=common" \
    graphbench-algoreas-hpc/slurm/grit_pe_refinement_worker.sbatch
)"
COMMON_JOB="${COMMON_JOB%%;*}"
echo "common_array=${COMMON_JOB}"

ARM_JOB="$(
  sbatch --parsable \
    -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
    --nodes=1 --ntasks=1 --gres=gpu:1 \
    --array="0-15%${MAX_PARALLEL}" \
    --dependency="afterok:${COMMON_JOB}" \
    --job-name=gb-pe-arms \
    --chdir="${PROJECT_ROOT}" \
    --output="${HPC_ROOT}/logs/%x-%A-%a.out" \
    --error="${HPC_ROOT}/logs/%x-%A-%a.err" \
    --export="${COMMON_EXPORT},WORKER_GROUP=arm" \
    graphbench-algoreas-hpc/slurm/grit_pe_refinement_worker.sbatch
)"
ARM_JOB="${ARM_JOB%%;*}"
echo "arm_array=${ARM_JOB}"

FINAL_JOB="$(
  sbatch --parsable \
    -A mlmi-jgg45-sl2-cpu -p sapphire --qos=cpu1 \
    --nodes=1 --ntasks=1 \
    --dependency="afterok:${ARM_JOB}" \
    --job-name=gb-pe-finalize \
    --chdir="${PROJECT_ROOT}" \
    --output="${HPC_ROOT}/logs/%x-%j.out" \
    --error="${HPC_ROOT}/logs/%x-%j.err" \
    --export="${COMMON_EXPORT}" \
    graphbench-algoreas-hpc/slurm/grit_pe_refinement_finalize.sbatch
)"
FINAL_JOB="${FINAL_JOB%%;*}"

echo "refinement_finalizer=${FINAL_JOB}"
echo "analysis_root=${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}"
echo "monitor: squeue -j ${COMMON_JOB},${ARM_JOB},${FINAL_JOB}"
