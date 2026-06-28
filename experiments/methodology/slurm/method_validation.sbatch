#!/bin/bash
#SBATCH --job-name=method-validation
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd "${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"

"${PYTHON_BIN}" -u -m graph_specialisation_metrics.method_validation run-all \
  --config "${CONFIG_PATH:-experiments/methodology/configs/method_validation_default.yaml}" \
  --output-root "${OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/method_validation_artifacts}" \
  --device "${DEVICE:-cuda}" \
  ${FORCE_FLAG:-}
