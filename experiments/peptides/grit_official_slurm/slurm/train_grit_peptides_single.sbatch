#!/usr/bin/env bash
#SBATCH --job-name=grit-peptides
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=grit-peptides-%j.out
#SBATCH --error=grit-peptides-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  if [[ -f "/usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh" ]]; then
    # CSD3's legacy conda does not reliably support `conda shell.bash hook`
    # inside non-interactive Slurm shells.
    source "/usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV}"
    else
      source activate "${CONDA_ENV}"
    fi
  else
    source activate "${CONDA_ENV}"
  fi
fi

if [[ -n "${VENV:-}" ]]; then
  source "${VENV}/bin/activate"
fi

PEPTIDES_TASK="${PEPTIDES_TASK:-struct}"
GRIT_VARIANT="${GRIT_VARIANT:-official}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/peptides_${PEPTIDES_TASK}_${GRIT_VARIANT}}"
RESULTS_DIR="${RESULTS_DIR:-${EXP_DIR}/results/${PEPTIDES_TASK}/${GRIT_VARIANT}}"
ACCELERATOR="${ACCELERATOR:-cuda:0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export GRIT_WORK_ROOT="${GRIT_WORK_ROOT:-${SLURM_TMPDIR:-${REPO_ROOT}/tmp}/grit_peptides_${PEPTIDES_TASK}_${GRIT_VARIANT}_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}}"

GRIT_REPO_DIR="$("${EXP_DIR}/scripts/prepare_grit_repo.sh" | tail -n 1)"

RUN_ARGS=(
  --repo-dir "${GRIT_REPO_DIR}"
  --task "${PEPTIDES_TASK}"
  --variant "${GRIT_VARIANT}"
  --seed "${SEED}"
  --data-dir "${DATA_DIR}"
  --out-dir "${RESULTS_DIR}"
  --accelerator "${ACCELERATOR}"
  --python "${PYTHON_BIN}"
  --num-threads "${OMP_NUM_THREADS}"
  --rrwp-stream-chunk-size "${GRIT_PE_STREAM_CHUNK_SIZE:-32}"
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

if [[ -n "${EXPECTED_PARAMS:-}" ]]; then
  RUN_ARGS+=(--expected-params "${EXPECTED_PARAMS}")
fi

if [[ "${NO_RECOVERY_PATCH:-0}" == "1" ]]; then
  RUN_ARGS+=(--no-recovery-patch)
fi

if [[ -n "${GRIT_EXTRA_CFG:-}" ]]; then
  read -r -a EXTRA_CFG <<< "${GRIT_EXTRA_CFG}"
  RUN_ARGS+=("${EXTRA_CFG[@]}")
fi

"${PYTHON_BIN}" "${EXP_DIR}/scripts/run_grit_peptides_slurm.py" "${RUN_ARGS[@]}"
