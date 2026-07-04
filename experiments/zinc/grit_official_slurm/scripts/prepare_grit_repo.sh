#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../.." && pwd)"

GRIT_REPO_URL="${GRIT_REPO_URL:-https://github.com/LiamMa/GRIT.git}"
GRIT_COMMIT="${GRIT_COMMIT:-6c988ea600a606fbb49a2246c64a2d37396b3ab5}"
GRIT_WORK_ROOT="${GRIT_WORK_ROOT:-${EXP_DIR}/work}"
GRIT_REPO_DIR="${GRIT_REPO_DIR:-${GRIT_WORK_ROOT}/GRIT}"
GRIT_PATCH="${GRIT_PATCH:-${EXP_DIR}/patches/0001-add-zinc-grit-1hop-control.patch}"

mkdir -p "${GRIT_WORK_ROOT}"

if [[ ! -d "${GRIT_REPO_DIR}/.git" ]]; then
  git clone "${GRIT_REPO_URL}" "${GRIT_REPO_DIR}"
fi

git -C "${GRIT_REPO_DIR}" fetch --all --tags --prune
git -C "${GRIT_REPO_DIR}" reset --hard
git -C "${GRIT_REPO_DIR}" clean -fd
git -C "${GRIT_REPO_DIR}" checkout --detach "${GRIT_COMMIT}"

if git -C "${GRIT_REPO_DIR}" apply --ignore-whitespace --whitespace=nowarn --check "${GRIT_PATCH}"; then
  git -C "${GRIT_REPO_DIR}" apply --ignore-whitespace --whitespace=nowarn "${GRIT_PATCH}"
else
  echo "Patch failed to apply cleanly: ${GRIT_PATCH}" >&2
  exit 2
fi

if [[ "${GRIT_ENABLE_LOCAL_RRWP_VARIANT:-1}" == "1" ]]; then
  PATCH_NOTE_DIR="${GRIT_LOCAL_RRWP_PATCH_NOTE_DIR:-${GRIT_WORK_ROOT}/patch_notes}"
  PYTHON_BIN="${PYTHON_BIN:-python}"
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" - "${GRIT_REPO_DIR}" "${PATCH_NOTE_DIR}" <<'PY'
from pathlib import Path
import sys

from experiments.zinc.training.grit_zinc_1hop_localrrwp_core import (
    apply_parameter_matched_onehop_localrrwp_patch,
)

apply_parameter_matched_onehop_localrrwp_patch(Path(sys.argv[1]), Path(sys.argv[2]))
PY
fi

git -C "${GRIT_REPO_DIR}" rev-parse HEAD >&2
printf '%s\n' "${GRIT_REPO_DIR}"
