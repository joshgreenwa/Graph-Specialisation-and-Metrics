#!/bin/bash
#SBATCH --job-name=cfim-grit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-1%1
#SBATCH --output=logs/cfim-grit-%A-%a.out
#SBATCH --error=logs/cfim-grit-%A-%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
CFIM_ARTIFACT_ROOT="${CFIM_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/cfim}"
CFIM_BACKEND="${CFIM_BACKEND:-official}"
CFIM_DEVICE="${CFIM_DEVICE:-cuda}"
CFIM_SKIP_TRAINING="${CFIM_SKIP_TRAINING:-0}"

if [[ -n "${ENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_ACTIVATE}"
fi

cd "${PROJECT_ROOT}"
mkdir -p logs "${CFIM_ARTIFACT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CFIM_ARTIFACT_ROOT

TASKS=(ppr_diffusion nearest_anchor_voronoi)
TASK="${TASKS[${SLURM_ARRAY_TASK_ID}]}"
CONFIG="experiments/synthetic/cfim/configs/grit_${TASK}.yaml"

ARGS=(
  -m graph_specialisation_metrics.counterfactual_interchange_mediation
  run-sequence
  --config "${CONFIG}"
  --task "${TASK}"
  --device "${CFIM_DEVICE}"
  --backend "${CFIM_BACKEND}"
)

if [[ "${CFIM_SKIP_TRAINING}" == "1" ]]; then
  ARGS+=(--skip-training)
fi

python "${ARGS[@]}"
