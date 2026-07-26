#!/bin/bash
# Submit isolated GRIT seed workers and one dependency-gated CPU finalizer.

set -euo pipefail

HPC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${HPC_ROOT}/.." && pwd)"
ENV_ACTIVATE="${ENV_ACTIVATE:-${HPC_ROOT}/activate_graphbench_algoreas}"
PROFILE="${PROFILE:-production}"
SEEDS="${SEEDS:-0,1,2,3}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
GRAPHBENCH_ANALYSIS_OUTPUT_ROOT="${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_graphbench_edge_v1}"
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
if [[ "${PROFILE}" != "production" && "${PROFILE}" != "sensitivity" && "${PROFILE}" != "smoke" ]]; then
  echo "PROFILE must be production, sensitivity, or smoke" >&2
  exit 2
fi
if [[ ! "${MAX_PARALLEL}" =~ ^[1-4]$ ]]; then
  echo "MAX_PARALLEL must be an integer from 1 to 4" >&2
  exit 2
fi
IFS=',' read -r -a SEED_VALUES <<< "${SEEDS}"
if (( ${#SEED_VALUES[@]} != 4 )); then
  echo "This production launcher requires exactly four comma-separated seeds" >&2
  exit 2
fi
SEED_LIST="${SEEDS//,/:}"

cd "${PROJECT_ROOT}"
mkdir -p graphbench-algoreas-hpc/logs "${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}"

COMMON_EXPORT="ALL,ENV_ACTIVATE=${ENV_ACTIVATE},PROFILE=${PROFILE},SEED_LIST=${SEED_LIST},GRIT_ROOT=${GRIT_ROOT},GRAPHBENCH_ANALYSIS_OUTPUT_ROOT=${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}"
ARRAY_SPEC="0-3%${MAX_PARALLEL}"

MATCH_JOB="$(
  sbatch --parsable \
    -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
    --nodes=1 --ntasks=1 --gres=gpu:1 \
    --array="${ARRAY_SPEC}" \
    --job-name=gb-grit-match \
    --chdir="${PROJECT_ROOT}" \
    --output="${HPC_ROOT}/logs/%x-%A-%a.out" \
    --error="${HPC_ROOT}/logs/%x-%A-%a.err" \
    --export="${COMMON_EXPORT},ANALYSIS_TASK=bipartite_matching_hard" \
    graphbench-algoreas-hpc/slurm/analyse_grit_specialisation.sbatch
)"
MATCH_JOB="${MATCH_JOB%%;*}"

FLOW_JOB="$(
  sbatch --parsable \
    -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
    --nodes=1 --ntasks=1 --gres=gpu:1 \
    --array="${ARRAY_SPEC}" \
    --job-name=gb-grit-flow \
    --chdir="${PROJECT_ROOT}" \
    --output="${HPC_ROOT}/logs/%x-%A-%a.out" \
    --error="${HPC_ROOT}/logs/%x-%A-%a.err" \
    --export="${COMMON_EXPORT},ANALYSIS_TASK=flow_hard" \
    graphbench-algoreas-hpc/slurm/analyse_grit_specialisation.sbatch
)"
FLOW_JOB="${FLOW_JOB%%;*}"

FINAL_JOB="$(
  sbatch --parsable \
    -A mlmi-jgg45-sl2-cpu -p sapphire --qos=cpu1 \
    --nodes=1 --ntasks=1 \
    --dependency="afterok:${MATCH_JOB}:${FLOW_JOB}" \
    --job-name=gb-grit-finalize \
    --chdir="${PROJECT_ROOT}" \
    --output="${HPC_ROOT}/logs/%x-%j.out" \
    --error="${HPC_ROOT}/logs/%x-%j.err" \
    --export="${COMMON_EXPORT}" \
    graphbench-algoreas-hpc/slurm/figures_grit_specialisation.sbatch
)"
FINAL_JOB="${FINAL_JOB%%;*}"

echo "matching_array=${MATCH_JOB}"
echo "flow_array=${FLOW_JOB}"
echo "cpu_finalizer=${FINAL_JOB}"
echo "analysis_root=${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT}"
echo "monitor: squeue -j ${MATCH_JOB},${FLOW_JOB},${FINAL_JOB}"
