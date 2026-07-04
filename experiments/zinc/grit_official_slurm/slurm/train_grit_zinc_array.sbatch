#!/usr/bin/env bash
#SBATCH --job-name=grit-zinc-array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --output=grit-zinc-array-%A_%a.out
#SBATCH --error=grit-zinc-array-%A_%a.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IFS=' ' read -r -a SEED_LIST <<< "${SEEDS:-41 42 43 44}"
IFS=' ' read -r -a VARIANT_LIST <<< "${GRIT_VARIANTS:-official 1hop}"
N_SEEDS="${#SEED_LIST[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT_INDEX=$((TASK_ID / N_SEEDS))
SEED_INDEX=$((TASK_ID % N_SEEDS))

if (( VARIANT_INDEX >= ${#VARIANT_LIST[@]} )); then
  echo "Array task ${TASK_ID} is outside ${#VARIANT_LIST[@]} variants x ${N_SEEDS} seeds." >&2
  exit 2
fi

export GRIT_VARIANT="${VARIANT_LIST[${VARIANT_INDEX}]}"
export SEED="${SEED_LIST[${SEED_INDEX}]}"

bash "${SCRIPT_DIR}/train_grit_zinc_single.sbatch"
