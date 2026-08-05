#!/bin/bash
#SBATCH --job-name=grit-molecules
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/grit-molecules-%A-%a.out
#SBATCH --error=logs/grit-molecules-%A-%a.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
: "${GRIT_MANIFEST:?Set GRIT_MANIFEST to an absolute JSONL manifest path}"
: "${GRIT_DATA_ROOT:?Set GRIT_DATA_ROOT to the staged shared dataset root}"
: "${GRIT_OUTPUT_ROOT:?Set GRIT_OUTPUT_ROOT to the shared output/checkpoint root}"

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONWARNINGS="ignore::FutureWarning"

SCRATCH_BASE="${GRIT_SCRATCH_ROOT:-${SLURM_TMPDIR:-/tmp}/grit_hpc}"
mkdir -p "${SCRATCH_BASE}" "${GRIT_OUTPUT_ROOT}"

EXTRA_ARGS=()
if [[ "${GRIT_WANDB:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--wandb)
  if [[ -n "${WANDB_PROJECT:-}" ]]; then
    EXTRA_ARGS+=(--wandb-project "${WANDB_PROJECT}")
  fi
fi

exec python experiments/grit_hpc/bin/grit_hpc.py run \
  --manifest "${GRIT_MANIFEST}" \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --dataset-root "${GRIT_DATA_ROOT}" \
  --output-root "${GRIT_OUTPUT_ROOT}" \
  --scratch-root "${SCRATCH_BASE}" \
  --grit-source "${GRIT_SOURCE_REPO:-https://github.com/LiamMa/GRIT.git}" \
  --num-threads "${SLURM_CPUS_PER_TASK:-8}" \
  --accelerator cuda:0 \
  --recovery-ckpt-period "${GRIT_RECOVERY_CKPT_PERIOD:-10}" \
  "${EXTRA_ARGS[@]}"
