#!/bin/bash
#SBATCH --job-name=grit-stage-data
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/grit-stage-%j.out
#SBATCH --error=logs/grit-stage-%j.err

set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
: "${GRIT_DATA_ROOT:?Set GRIT_DATA_ROOT to the shared dataset root}"

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONWARNINGS="ignore::FutureWarning"

SCRATCH_BASE="${GRIT_SCRATCH_ROOT:-${SLURM_TMPDIR:-/tmp}/grit_hpc_stage}"
mkdir -p "${SCRATCH_BASE}" "${GRIT_DATA_ROOT}"

exec python experiments/grit_hpc/bin/grit_hpc.py stage-datasets \
  --tasks zinc,qm9_gap,peptides_func,peptides_struct \
  --dataset-root "${GRIT_DATA_ROOT}" \
  --scratch-root "${SCRATCH_BASE}" \
  --grit-source "${GRIT_SOURCE_REPO:-https://github.com/LiamMa/GRIT.git}"
