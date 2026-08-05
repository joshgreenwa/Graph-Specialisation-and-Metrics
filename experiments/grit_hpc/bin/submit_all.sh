#!/bin/bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
: "${GRIT_MANIFEST:?Set GRIT_MANIFEST to an absolute JSONL manifest path}"
: "${GRIT_DATA_ROOT:?Set GRIT_DATA_ROOT to the shared dataset root}"
: "${GRIT_OUTPUT_ROOT:?Set GRIT_OUTPUT_ROOT to the shared output/checkpoint root}"

JOB_COUNT="$(awk 'NF && $1 !~ /^#/' "${GRIT_MANIFEST}" | wc -l | tr -d ' ')"
if [[ "${JOB_COUNT}" -lt 1 ]]; then
  echo "Manifest contains no jobs: ${GRIT_MANIFEST}" >&2
  exit 2
fi

MAX_PARALLEL="${GRIT_MAX_PARALLEL:-${JOB_COUNT}}"
mkdir -p "${PROJECT_ROOT}/logs" "${GRIT_DATA_ROOT}" "${GRIT_OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

STAGE_JOB="$(sbatch --parsable \
  "${PROJECT_ROOT}/experiments/grit_hpc/slurm/stage_datasets.sbatch")"
STAGE_JOB="${STAGE_JOB%%;*}"

TRAIN_JOB="$(sbatch --parsable \
  --dependency="afterok:${STAGE_JOB}" \
  --array="0-$((JOB_COUNT - 1))%${MAX_PARALLEL}" \
  "${PROJECT_ROOT}/experiments/grit_hpc/slurm/train_array.sbatch")"

echo "Dataset staging job: ${STAGE_JOB}"
echo "GPU training array:  ${TRAIN_JOB} (starts after successful staging)"
