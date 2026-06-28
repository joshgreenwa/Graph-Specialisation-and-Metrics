#!/bin/bash
#SBATCH --job-name=main-procedure
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-5

set -euo pipefail

cd "${PROJECT_ROOT:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
STEP="${STEP_OVERRIDE:-${SLURM_ARRAY_TASK_ID}}"

"${PYTHON_BIN}" -u -m graph_specialisation_metrics.main_procedure run \
  --config "${CONFIG_PATH:-experiments/methodology/configs/zinc_main_procedure.yaml}" \
  --steps "${STEP}" \
  --output-root "${OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/main_procedure_artifacts}" \
  ${DRY_RUN_FLAG:-} \
  ${FORCE_FLAG:-}
