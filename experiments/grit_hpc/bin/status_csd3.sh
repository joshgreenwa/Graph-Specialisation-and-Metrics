#!/bin/bash
set -euo pipefail

HPC_WORK_ROOT="${HPC_WORK_ROOT:-/rds/user/jgg45/hpc-work}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
TRACKING_FILE="${GRIT_TRACKING_FILE:-${HPC_WORK_ROOT}/grit_manifests/grit_all_3seeds.jobs.tsv}"
RESUBMIT_LEDGER="${GRIT_RESUBMIT_LEDGER:-${HPC_WORK_ROOT}/grit_manifests/grit_all_3seeds.resubmissions.tsv}"
STAGE_TRACKING_FILE="${GRIT_STAGE_TRACKING_FILE:-${HPC_WORK_ROOT}/grit_manifests/grit_dataset_staging.jobs.tsv}"

if [[ ! -f "${TRACKING_FILE}" ]]; then
  echo "Tracking ledger not found: ${TRACKING_FILE}" >&2
  echo "Run submit_csd3_60.sh first." >&2
  exit 2
fi

ARRAY_JOB_ID="${1:-$(awk -F '\t' 'NR == 2 { print $2 }' "${TRACKING_FILE}")}"
if [[ -z "${ARRAY_JOB_ID}" ]]; then
  echo "Could not read the array job ID from ${TRACKING_FILE}" >&2
  exit 2
fi

if [[ -f "${RESUBMIT_LEDGER}" ]]; then
  echo
  echo "Resubmission history: ${RESUBMIT_LEDGER}"
  if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "${RESUBMIT_LEDGER}"
  else
    cat "${RESUBMIT_LEDGER}"
  fi
fi

if [[ -f "${STAGE_TRACKING_FILE}" ]]; then
  echo
  echo "Dataset-staging mapping: ${STAGE_TRACKING_FILE}"
  if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "${STAGE_TRACKING_FILE}"
  else
    cat "${STAGE_TRACKING_FILE}"
  fi
  STAGE_ARRAY_JOB_ID="$(awk -F '\t' 'NR == 2 { print $2 }' "${STAGE_TRACKING_FILE}")"
  echo "Current staging queue state:"
  squeue -r -j "${STAGE_ARRAY_JOB_ID}" \
    -o '%.20i %.10T %.10M %.10l %.24R' || true
  echo "Staging accounting state:"
  sacct -X -j "${STAGE_ARRAY_JOB_ID}" \
    --format=JobID,JobName%24,State,Elapsed,Timelimit,ExitCode || true
fi

echo
STATUS_ARGS=()
if [[ "$#" -gt 0 ]]; then
  STATUS_ARGS+=(--slurm-array-job-id "${ARRAY_JOB_ID}" --only-known)
fi
python "${PROJECT_ROOT}/experiments/grit_hpc/bin/grit_hpc.py" status-tracking \
  --tracking "${TRACKING_FILE}" \
  "${STATUS_ARGS[@]}"
