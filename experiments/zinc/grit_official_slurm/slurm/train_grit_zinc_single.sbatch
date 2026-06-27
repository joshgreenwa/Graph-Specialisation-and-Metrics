#!/usr/bin/env bash
#SBATCH --job-name=grit-zinc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=grit-zinc-%j.out
#SBATCH --error=grit-zinc-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

if [[ -n "${VENV:-}" ]]; then
  source "${VENV}/bin/activate"
fi

GRIT_VARIANT="${GRIT_VARIANT:-official}"
SEED="${SEED:-41}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
RESULTS_DIR="${RESULTS_DIR:-${EXP_DIR}/results}"
ACCELERATOR="${ACCELERATOR:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export GRIT_WORK_ROOT="${GRIT_WORK_ROOT:-${SLURM_TMPDIR:-${REPO_ROOT}/tmp}/grit_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}}"

GRIT_REPO_DIR="$("${EXP_DIR}/scripts/prepare_grit_repo.sh" | tail -n 1)"

RUN_ARGS=(
  --repo-dir "${GRIT_REPO_DIR}"
  --variant "${GRIT_VARIANT}"
  --seed "${SEED}"
  --data-dir "${DATA_DIR}"
  --out-dir "${RESULTS_DIR}"
  --accelerator "${ACCELERATOR}"
  --python "${PYTHON_BIN}"
)

if [[ -n "${MAX_EPOCH:-}" ]]; then
  RUN_ARGS+=(--max-epoch "${MAX_EPOCH}")
fi

if [[ -n "${NAME_TAG:-}" ]]; then
  RUN_ARGS+=(--name-tag "${NAME_TAG}")
fi

if [[ "${ALLOW_PARAM_COUNT_DRIFT:-0}" == "1" ]]; then
  RUN_ARGS+=(--allow-param-count-drift)
fi

if [[ -n "${GRIT_EXTRA_CFG:-}" ]]; then
  read -r -a EXTRA_CFG <<< "${GRIT_EXTRA_CFG}"
  RUN_ARGS+=("${EXTRA_CFG[@]}")
fi

"${PYTHON_BIN}" "${EXP_DIR}/scripts/run_grit_zinc_slurm.py" "${RUN_ARGS[@]}"
