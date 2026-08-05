#!/bin/bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 ARRAY_SPEC" >&2
  echo "Examples: $0 4    or    $0 4,17,39-42%8" >&2
  exit 2
fi

ARRAY_SPEC="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
HPC_WORK_ROOT="${HPC_WORK_ROOT:-/rds/user/jgg45/hpc-work}"

export PROJECT_ROOT
export GRIT_DATA_ROOT="${GRIT_DATA_ROOT:-${HPC_WORK_ROOT}/grit_shared_datasets}"
export GRIT_OUTPUT_ROOT="${GRIT_OUTPUT_ROOT:-${HPC_WORK_ROOT}/grit_checkpoints/grit_all_3seeds}"
export GRIT_MANIFEST="${GRIT_MANIFEST:-${HPC_WORK_ROOT}/grit_manifests/grit_all_3seeds.jsonl}"
export GRIT_RESUBMIT_LEDGER="${GRIT_RESUBMIT_LEDGER:-${HPC_WORK_ROOT}/grit_manifests/grit_all_3seeds.resubmissions.tsv}"
export GRIT_SOURCE_REPO="${GRIT_SOURCE_REPO:-${HPC_WORK_ROOT}/GRIT_pristine}"
export GRIT_RECOVERY_CKPT_PERIOD="${GRIT_RECOVERY_CKPT_PERIOD:-10}"
export GRIT_WANDB=0
TRAIN_TIME_LIMIT="${GRIT_JOB_TIME_LIMIT:-06:00:00}"

if [[ ! -f "${GRIT_MANIFEST}" ]]; then
  echo "Manifest is missing; run submit_csd3_60.sh first: ${GRIT_MANIFEST}" >&2
  exit 2
fi

python "${PROJECT_ROOT}/experiments/grit_hpc/bin/grit_hpc.py" check-datasets \
  --tasks zinc,qm9_gap,peptides_func,peptides_struct \
  --dataset-root "${GRIT_DATA_ROOT}"

cd "${PROJECT_ROOT}"
RESUBMIT_JOB="$(sbatch --parsable \
  --export=ALL \
  -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
  -N 1 --ntasks=1 --gres=gpu:1 \
  --time="${TRAIN_TIME_LIMIT}" \
  --array="${ARRAY_SPEC}" \
  experiments/grit_hpc/slurm/train_array.sbatch)"
RESUBMIT_JOB="${RESUBMIT_JOB%%;*}"

if [[ ! -f "${GRIT_RESUBMIT_LEDGER}" ]]; then
  printf 'submitted_at\tarray_job_id\tarray_spec\ttime_limit\n' >"${GRIT_RESUBMIT_LEDGER}"
fi
printf '%s\t%s\t%s\t%s\n' \
  "$(date --iso-8601=seconds)" "${RESUBMIT_JOB}" "${ARRAY_SPEC}" "${TRAIN_TIME_LIMIT}" \
  >>"${GRIT_RESUBMIT_LEDGER}"

echo "Resubmission array: ${RESUBMIT_JOB}"
echo "Rows:               ${ARRAY_SPEC}"
echo "Attempt ledger:     ${GRIT_RESUBMIT_LEDGER}"
echo "Track this attempt: bash experiments/grit_hpc/bin/status_csd3.sh ${RESUBMIT_JOB}"
