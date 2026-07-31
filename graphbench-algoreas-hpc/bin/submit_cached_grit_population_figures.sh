#!/bin/bash
# Locate one complete four-seed matching cache and submit only its CPU figure finalizer.

set -euo pipefail

HPC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${HPC_ROOT}/.." && pwd)"
ENV_ACTIVATE="${ENV_ACTIVATE:-${HPC_ROOT}/activate_graphbench_algoreas}"
OUTPUT_BASE="${GRAPHBENCH_OUTPUT_BASE:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs}"
REQUESTED_ROOT="${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT:-}"
PROFILE="${PROFILE:-production}"
SEED_LIST="${SEED_LIST:-0:1:2:3}"
TASK_LIST="bipartite_matching_hard"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

if [[ ! -f "${ENV_ACTIVATE}" ]]; then
  echo "Missing environment activation file: ${ENV_ACTIVATE}" >&2
  exit 2
fi
if [[ ! -d "${OUTPUT_BASE}" ]]; then
  echo "Missing GraphBench output base: ${OUTPUT_BASE}" >&2
  exit 2
fi
if [[ "${SEED_LIST}" != "0:1:2:3" ]]; then
  echo "The population finalizer requires SEED_LIST=0:1:2:3" >&2
  exit 2
fi

cache_root_complete() {
  local root="$1"
  local seed
  local relative
  for seed in 0 1 2 3; do
    for relative in \
      "cache/scores/raw.pt" \
      "cache/causal/validation.pt" \
      "audits.json"; do
      if [[ ! -s "${root}/graphbench_bipartite_matching_hard/seed_${seed}/${relative}" ]]; then
        return 1
      fi
    done
  done
  return 0
}

declare -a COMPLETE_ROOTS=()

add_complete_root() {
  local candidate="$1"
  local existing
  cache_root_complete "${candidate}" || return 0
  for existing in "${COMPLETE_ROOTS[@]:-}"; do
    if [[ "${existing}" == "${candidate}" ]]; then
      return 0
    fi
  done
  COMPLETE_ROOTS+=("${candidate}")
}

if [[ -n "${REQUESTED_ROOT}" ]]; then
  add_complete_root "${REQUESTED_ROOT}"
fi

CACHE_SUFFIX="/graphbench_bipartite_matching_hard/seed_0/cache/scores/raw.pt"
while IFS= read -r score_cache; do
  add_complete_root "${score_cache%${CACHE_SUFFIX}}"
done < <(
  find "${OUTPUT_BASE}" \
    -type f \
    -path "*${CACHE_SUFFIX}" \
    -print \
    2>/dev/null \
    | sort
)

if (( ${#COMPLETE_ROOTS[@]} == 0 )); then
  echo "No complete four-seed matching cache root was found under ${OUTPUT_BASE}." >&2
  if [[ -n "${REQUESTED_ROOT}" ]]; then
    echo "Requested but incomplete: ${REQUESTED_ROOT}" >&2
  fi
  echo "Required per seed: raw.pt, validation.pt, and audits.json." >&2
  exit 2
fi

if (( ${#COMPLETE_ROOTS[@]} > 1 )); then
  echo "Multiple complete matching cache roots were found; no job was submitted." >&2
  printf '  %s\n' "${COMPLETE_ROOTS[@]}" >&2
  echo "Set GRAPHBENCH_ANALYSIS_OUTPUT_ROOT to the intended root and rerun." >&2
  exit 2
fi

ANALYSIS_ROOT="${COMPLETE_ROOTS[0]}"
if [[ -n "${REQUESTED_ROOT}" && "${REQUESTED_ROOT}" != "${ANALYSIS_ROOT}" ]]; then
  echo "[WARN] requested root is incomplete: ${REQUESTED_ROOT}"
  echo "[OK] using uniquely complete cache root: ${ANALYSIS_ROOT}"
else
  echo "[OK] complete cache root: ${ANALYSIS_ROOT}"
fi

for seed in 0 1 2 3; do
  echo "[OK] seed ${seed}: scores + causal validation + audits"
done

cd "${PROJECT_ROOT}"
mkdir -p "${HPC_ROOT}/logs"

COMMON_EXPORT="ALL,ENV_ACTIVATE=${ENV_ACTIVATE},PROFILE=${PROFILE},SEED_LIST=${SEED_LIST},TASK_LIST=${TASK_LIST},GRAPHBENCH_ANALYSIS_OUTPUT_ROOT=${ANALYSIS_ROOT},OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1"

FINAL_JOB="$(
  "${SBATCH_BIN}" --parsable \
    -A mlmi-jgg45-sl2-cpu \
    -p sapphire \
    --qos=cpu1 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=32G \
    --time=01:00:00 \
    --job-name=gb-grit-finalize \
    --chdir="${PROJECT_ROOT}" \
    --output="${HPC_ROOT}/logs/%x-%j.out" \
    --error="${HPC_ROOT}/logs/%x-%j.err" \
    --export="${COMMON_EXPORT}" \
    graphbench-algoreas-hpc/slurm/figures_grit_specialisation.sbatch
)"
FINAL_JOB="${FINAL_JOB%%;*}"

echo "cpu_finalizer=${FINAL_JOB}"
echo "analysis_root=${ANALYSIS_ROOT}"
echo "monitor: squeue -j ${FINAL_JOB}"
