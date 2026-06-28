#!/bin/bash
#SBATCH --job-name=render-figures
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd "${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"

"${PYTHON_BIN}" -u -m graph_specialisation_metrics.figures render \
  --artifact-root "${ARTIFACT_ROOT:?set ARTIFACT_ROOT to an existing artifact directory}"
