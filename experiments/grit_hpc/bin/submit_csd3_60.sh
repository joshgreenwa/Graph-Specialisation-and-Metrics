#!/bin/bash
set -euo pipefail

# Budget-bounded CSD3 launch: 4 tasks x 5 variants x 3 seeds = 60 GPUs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
HPC_WORK_ROOT="${HPC_WORK_ROOT:-/rds/user/jgg45/hpc-work}"

export PROJECT_ROOT
export GRIT_DATA_ROOT="${GRIT_DATA_ROOT:-${HPC_WORK_ROOT}/grit_shared_datasets}"
export GRIT_OUTPUT_ROOT="${GRIT_OUTPUT_ROOT:-${HPC_WORK_ROOT}/grit_checkpoints/grit_all_3seeds}"
export GRIT_MANIFEST="${GRIT_MANIFEST:-${HPC_WORK_ROOT}/grit_manifests/grit_all_3seeds.jsonl}"
export GRIT_TRACKING_FILE="${GRIT_TRACKING_FILE:-${HPC_WORK_ROOT}/grit_manifests/grit_all_3seeds.jobs.tsv}"
export GRIT_STAGE_TRACKING_FILE="${GRIT_STAGE_TRACKING_FILE:-${HPC_WORK_ROOT}/grit_manifests/grit_dataset_staging.jobs.tsv}"
export GRIT_SOURCE_REPO="${GRIT_SOURCE_REPO:-${HPC_WORK_ROOT}/GRIT_pristine}"
export GRIT_RECOVERY_CKPT_PERIOD="${GRIT_RECOVERY_CKPT_PERIOD:-10}"
# This batch is intentionally W&B-free. Do not inherit a stale interactive
# GRIT_WANDB=1 from the submission shell.
export GRIT_WANDB=0

TRAIN_TIME_LIMIT="${GRIT_JOB_TIME_LIMIT:-06:00:00}"
STAGE_TIME_LIMIT="${GRIT_STAGE_TIME_LIMIT:-01:00:00}"

time_to_seconds() {
  local value="$1" hours minutes seconds
  IFS=: read -r hours minutes seconds <<<"${value}"
  if [[ -z "${hours:-}" || -z "${minutes:-}" || -z "${seconds:-}" ]] \
      || ! [[ "${hours}" =~ ^[0-9]+$ && "${minutes}" =~ ^[0-9]+$ && "${seconds}" =~ ^[0-9]+$ ]] \
      || (( 10#${minutes} >= 60 || 10#${seconds} >= 60 )); then
    echo "Invalid Slurm time value (expected HH:MM:SS): ${value}" >&2
    exit 2
  fi
  echo $((10#${hours} * 3600 + 10#${minutes} * 60 + 10#${seconds}))
}

TRAIN_SECONDS="$(time_to_seconds "${TRAIN_TIME_LIMIT}")"
STAGE_SECONDS="$(time_to_seconds "${STAGE_TIME_LIMIT}")"
REQUESTED_SECONDS=$((60 * TRAIN_SECONDS))
BUDGET_SECONDS=$((400 * 3600))
TRAIN_REQUEST_HOURS="$(awk "BEGIN { printf \"%.2f\", 60 * ${TRAIN_SECONDS} / 3600 }")"
STAGE_REQUEST_HOURS="$(awk "BEGIN { printf \"%.2f\", 2 * ${STAGE_SECONDS} / 3600 }")"
TOTAL_REQUEST_HOURS="$(awk "BEGIN { printf \"%.2f\", ${REQUESTED_SECONDS} / 3600 }")"
if (( REQUESTED_SECONDS > BUDGET_SECONDS )); then
  printf 'Refusing submission: requested maximum is %.2f GPU-hours, above the 400-hour budget.\n' \
    "$(awk "BEGIN { print ${REQUESTED_SECONDS} / 3600 }")" >&2
  exit 2
fi

mkdir -p \
  "${PROJECT_ROOT}/logs" \
  "${GRIT_DATA_ROOT}" \
  "${GRIT_OUTPUT_ROOT}" \
  "$(dirname "${GRIT_MANIFEST}")"

if [[ ! -d "${GRIT_SOURCE_REPO}/.git" ]]; then
  if [[ -e "${GRIT_SOURCE_REPO}" ]]; then
    echo "GRIT_SOURCE_REPO exists but is not a Git checkout: ${GRIT_SOURCE_REPO}" >&2
    exit 2
  fi
  git clone https://github.com/LiamMa/GRIT.git "${GRIT_SOURCE_REPO}"
fi

PINNED_GRIT_COMMIT="6c988ea600a606fbb49a2246c64a2d37396b3ab5"
if ! git -C "${GRIT_SOURCE_REPO}" cat-file -e "${PINNED_GRIT_COMMIT}^{commit}"; then
  git -C "${GRIT_SOURCE_REPO}" fetch origin "${PINNED_GRIT_COMMIT}"
fi

python "${PROJECT_ROOT}/experiments/grit_hpc/bin/grit_hpc.py" make-manifest \
  --tasks zinc,qm9_gap,peptides_func,peptides_struct \
  --variants dense,1hop,1hop_vnode,2hop,2hop_vnode \
  --seeds 0,1,2 \
  --output "${GRIT_MANIFEST}"

python "${PROJECT_ROOT}/experiments/grit_hpc/bin/grit_hpc.py" print-jobs \
  --manifest "${GRIT_MANIFEST}"

cd "${PROJECT_ROOT}"

# INTR permits only one submitted job per user. This single CPU allocation
# stages all four datasets sequentially before releasing the GPU dependency.
STAGE_JOB="$(sbatch --parsable \
  --export=ALL \
  -A mlmi-jgg45-sl2-cpu -p sapphire --qos=intr \
  -N 1 --ntasks=1 --cpus-per-task=2 --mem=16G \
  --time="${STAGE_TIME_LIMIT}" \
  experiments/grit_hpc/slurm/stage_datasets.sbatch)"
STAGE_JOB="${STAGE_JOB%%;*}"
{
  printf 'job_id\tdataset\n'
  printf '%s\tzinc,qm9_gap,peptides_func,peptides_struct\n' "${STAGE_JOB}"
} >"${GRIT_STAGE_TRACKING_FILE}"

# 60 x 6 hours = 360 requested GPU-hours against the 400-hour GPU balance.
TRAIN_JOB="$(sbatch --parsable \
  --export=ALL \
  -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
  -N 1 --ntasks=1 --gres=gpu:1 \
  --time="${TRAIN_TIME_LIMIT}" \
  --array=0-59%60 \
  --dependency="afterok:${STAGE_JOB}" \
  experiments/grit_hpc/slurm/train_array.sbatch)"
TRAIN_JOB="${TRAIN_JOB%%;*}"
echo "Training array submitted with parent JobID: ${TRAIN_JOB}"

python "${PROJECT_ROOT}/experiments/grit_hpc/bin/grit_hpc.py" write-tracking \
  --manifest "${GRIT_MANIFEST}" \
  --slurm-array-job-id "${TRAIN_JOB}" \
  --output-root "${GRIT_OUTPUT_ROOT}" \
  --log-root "${PROJECT_ROOT}/logs" \
  --output "${GRIT_TRACKING_FILE}"

echo "Dataset staging job:   ${STAGE_JOB} (${STAGE_TIME_LIMIT}; ${STAGE_REQUEST_HOURS} CPU-hours max)"
echo "Training array:        ${TRAIN_JOB} (${TRAIN_TIME_LIMIT} each; ${TRAIN_REQUEST_HOURS} GPU-hours max)"
echo "Initial GPU request:   ${TOTAL_REQUEST_HOURS} GPU-hours maximum"
echo "Persistent outputs:    ${GRIT_OUTPUT_ROOT}"
echo "Manifest:              ${GRIT_MANIFEST}"
echo "Job tracking ledger:   ${GRIT_TRACKING_FILE}"
echo "Staging job ledger:    ${GRIT_STAGE_TRACKING_FILE}"
