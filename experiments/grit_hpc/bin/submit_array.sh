#!/bin/bash
set -euo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the repository root}"
: "${GRIT_MANIFEST:?Set GRIT_MANIFEST to an absolute JSONL manifest path}"

JOB_COUNT="$(awk 'NF && $1 !~ /^#/' "${GRIT_MANIFEST}" | wc -l | tr -d ' ')"
if [[ "${JOB_COUNT}" -lt 1 ]]; then
  echo "Manifest contains no jobs: ${GRIT_MANIFEST}" >&2
  exit 2
fi

MAX_PARALLEL="${GRIT_MAX_PARALLEL:-${JOB_COUNT}}"
mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

exec sbatch \
  --array="0-$((JOB_COUNT - 1))%${MAX_PARALLEL}" \
  "${PROJECT_ROOT}/experiments/grit_hpc/slurm/train_array.sbatch"
