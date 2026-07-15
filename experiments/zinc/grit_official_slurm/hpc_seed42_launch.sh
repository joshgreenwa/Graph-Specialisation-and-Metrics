#!/usr/bin/env bash
# Submit seed-42 official GRIT and parameter-matched 1-hop sparse GRIT on ZINC.
#
# Intended HPC location after git pull:
#   /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics/experiments/zinc/grit_official_slurm/hpc_seed42_launch.sh
#
# Run from the repository root on login.hpc.cam.ac.uk:
#   bash experiments/zinc/grit_official_slurm/hpc_seed42_launch.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
CONDA_ENV="${CONDA_ENV:-graphbench-algoreas}"
SEED="${SEED:-42}"

# Separate DATA_DIRs avoid concurrent PyG processed-cache writes when dense and
# 1-hop jobs are submitted together.
DATA_ROOT="${DATA_ROOT:-/rds/user/jgg45/hpc-work/grit_zinc_seed42_data}"
RESULTS_ROOT="${RESULTS_ROOT:-/rds/user/jgg45/hpc-work/grit_zinc_seed42_results}"

ACCOUNT="${ACCOUNT:-mlmi-jgg45-sl2-gpu}"
PARTITION="${PARTITION:-ampere}"
QOS="${QOS:-gpu1}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"

cd "${PROJECT_DIR}"
git pull --ff-only

mkdir -p \
  "${DATA_ROOT}/official" \
  "${DATA_ROOT}/1hop" \
  "${RESULTS_ROOT}/official" \
  "${RESULTS_ROOT}/1hop" \
  "${RESULTS_ROOT}/slurm_submit_logs"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-grit-zinc}"

GRIT_DENSE_JOB_ID="$(
  sbatch --parsable \
    --export=ALL,WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,CONDA_ENV="${CONDA_ENV}",GRIT_VARIANT=official,SEED="${SEED}",DATA_DIR="${DATA_ROOT}/official",RESULTS_DIR="${RESULTS_ROOT}/official" \
    -A "${ACCOUNT}" -p "${PARTITION}" --qos="${QOS}" \
    -N 1 --ntasks=1 --gres=gpu:1 --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --time="${TIME_LIMIT}" \
    --job-name="grit-zinc-dense-s${SEED}" \
    --output="${RESULTS_ROOT}/slurm_submit_logs/%x-%j.out" \
    --error="${RESULTS_ROOT}/slurm_submit_logs/%x-%j.err" \
    experiments/zinc/grit_official_slurm/slurm/train_grit_zinc_single.sbatch
)"

GRIT_1HOP_JOB_ID="$(
  sbatch --parsable \
    --export=ALL,WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,CONDA_ENV="${CONDA_ENV}",GRIT_VARIANT=1hop,SEED="${SEED}",DATA_DIR="${DATA_ROOT}/1hop",RESULTS_DIR="${RESULTS_ROOT}/1hop" \
    -A "${ACCOUNT}" -p "${PARTITION}" --qos="${QOS}" \
    -N 1 --ntasks=1 --gres=gpu:1 --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --time="${TIME_LIMIT}" \
    --job-name="grit-zinc-1hop-s${SEED}" \
    --output="${RESULTS_ROOT}/slurm_submit_logs/%x-%j.out" \
    --error="${RESULTS_ROOT}/slurm_submit_logs/%x-%j.err" \
    experiments/zinc/grit_official_slurm/slurm/train_grit_zinc_single.sbatch
)"

cat <<EOF
Submitted:
  GRIT_DENSE_JOB_ID=${GRIT_DENSE_JOB_ID}
  GRIT_1HOP_JOB_ID=${GRIT_1HOP_JOB_ID}

Save these in your shell if needed:
  export GRIT_DENSE_JOB_ID=${GRIT_DENSE_JOB_ID}
  export GRIT_1HOP_JOB_ID=${GRIT_1HOP_JOB_ID}
  export GRIT_ZINC_RESULTS_ROOT=${RESULTS_ROOT}

Queue:
  squeue -j "\${GRIT_DENSE_JOB_ID},\${GRIT_1HOP_JOB_ID}"

Tail:
  tail -f \\
    "\${GRIT_ZINC_RESULTS_ROOT}/slurm_submit_logs/grit-zinc-dense-s${SEED}-\${GRIT_DENSE_JOB_ID}.out" \\
    "\${GRIT_ZINC_RESULTS_ROOT}/slurm_submit_logs/grit-zinc-dense-s${SEED}-\${GRIT_DENSE_JOB_ID}.err" \\
    "\${GRIT_ZINC_RESULTS_ROOT}/slurm_submit_logs/grit-zinc-1hop-s${SEED}-\${GRIT_1HOP_JOB_ID}.out" \\
    "\${GRIT_ZINC_RESULTS_ROOT}/slurm_submit_logs/grit-zinc-1hop-s${SEED}-\${GRIT_1HOP_JOB_ID}.err"

Accounting after completion:
  sacct -j "\${GRIT_DENSE_JOB_ID},\${GRIT_1HOP_JOB_ID}" --format=JobID,JobName%32,State,ExitCode,Elapsed,Timelimit,ReqTRES%80
EOF
