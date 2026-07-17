#!/usr/bin/env bash
#SBATCH --job-name=grit-peptides-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=grit-peptides-array-%A_%a.out
#SBATCH --error=grit-peptides-array-%A_%a.err

set -euo pipefail

if [[ -n "${PROJECT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${PROJECT_DIR}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/experiments/peptides/grit_official_slurm" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
fi
SCRIPT_DIR="${REPO_ROOT}/experiments/peptides/grit_official_slurm/slurm"

IFS=' ' read -r -a TASK_LIST <<< "${PEPTIDES_TASKS:-struct func}"
IFS=' ' read -r -a VARIANT_LIST <<< "${GRIT_VARIANTS:-official 1hop}"
IFS=' ' read -r -a SEED_LIST <<< "${SEEDS:-42}"

N_SEEDS="${#SEED_LIST[@]}"
N_VARIANTS="${#VARIANT_LIST[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
TASK_INDEX=$((TASK_ID / (N_VARIANTS * N_SEEDS)))
REM=$((TASK_ID % (N_VARIANTS * N_SEEDS)))
VARIANT_INDEX=$((REM / N_SEEDS))
SEED_INDEX=$((REM % N_SEEDS))

if (( TASK_INDEX >= ${#TASK_LIST[@]} )); then
  echo "Array task ${TASK_ID} is outside ${#TASK_LIST[@]} tasks x ${N_VARIANTS} variants x ${N_SEEDS} seeds." >&2
  exit 2
fi

export PEPTIDES_TASK="${TASK_LIST[${TASK_INDEX}]}"
export GRIT_VARIANT="${VARIANT_LIST[${VARIANT_INDEX}]}"
export SEED="${SEED_LIST[${SEED_INDEX}]}"

bash "${SCRIPT_DIR}/train_grit_peptides_single.sbatch"
