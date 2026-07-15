#!/usr/bin/env bash
# Submit seed-42 official GRIT and parameter-matched 1-hop sparse GRIT on:
#   - Peptides-struct
#   - Peptides-func
#
# Intended HPC location after git pull:
#   /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics/experiments/peptides/grit_official_slurm/hpc_seed42_launch.sh
#
# Run from the repository root on login.hpc.cam.ac.uk:
#   bash experiments/peptides/grit_official_slurm/hpc_seed42_launch.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics}"
CONDA_ENV="${CONDA_ENV:-graphbench-algoreas}"
SEED="${SEED:-42}"

DATA_ROOT="${DATA_ROOT:-/rds/user/jgg45/hpc-work/grit_peptides_seed42_data}"
RESULTS_ROOT="${RESULTS_ROOT:-/rds/user/jgg45/hpc-work/grit_peptides_seed42_results}"

ACCOUNT="${ACCOUNT:-mlmi-jgg45-sl2-gpu}"
PARTITION="${PARTITION:-ampere}"
QOS="${QOS:-gpu1}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-120G}"

cd "${PROJECT_DIR}"
git pull --ff-only

mkdir -p "${RESULTS_ROOT}/slurm_submit_logs"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-grit-peptides}"
export GRIT_PE_STREAM_CHUNK_SIZE="${GRIT_PE_STREAM_CHUNK_SIZE:-32}"

declare -A JOB_IDS=()

submit_one() {
  local task="$1"
  local variant="$2"
  local label="$3"
  local data_dir="${DATA_ROOT}/${task}/${variant}"
  local results_dir="${RESULTS_ROOT}/${task}/${variant}"
  mkdir -p "${data_dir}" "${results_dir}"

  local job_id
  job_id="$(
    sbatch --parsable \
      --export=ALL,WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,CONDA_ENV="${CONDA_ENV}",PEPTIDES_TASK="${task}",GRIT_VARIANT="${variant}",SEED="${SEED}",DATA_DIR="${data_dir}",RESULTS_DIR="${results_dir}",GRIT_PE_STREAM_CHUNK_SIZE="${GRIT_PE_STREAM_CHUNK_SIZE}" \
      -A "${ACCOUNT}" -p "${PARTITION}" --qos="${QOS}" \
      -N 1 --ntasks=1 --gres=gpu:1 --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEM}" --time="${TIME_LIMIT}" \
      --job-name="grit-pep-${label}-s${SEED}" \
      --output="${RESULTS_ROOT}/slurm_submit_logs/%x-%j.out" \
      --error="${RESULTS_ROOT}/slurm_submit_logs/%x-%j.err" \
      experiments/peptides/grit_official_slurm/slurm/train_grit_peptides_single.sbatch
  )"
  JOB_IDS["${label}"]="${job_id}"
}

submit_one "struct" "official" "struct-dense"
submit_one "struct" "1hop" "struct-1hop"
submit_one "func" "official" "func-dense"
submit_one "func" "1hop" "func-1hop"

cat <<EOF
Submitted:
  GRIT_PEPTIDES_STRUCT_DENSE_JOB_ID=${JOB_IDS[struct-dense]}
  GRIT_PEPTIDES_STRUCT_1HOP_JOB_ID=${JOB_IDS[struct-1hop]}
  GRIT_PEPTIDES_FUNC_DENSE_JOB_ID=${JOB_IDS[func-dense]}
  GRIT_PEPTIDES_FUNC_1HOP_JOB_ID=${JOB_IDS[func-1hop]}

Save these in your shell if needed:
  export GRIT_PEPTIDES_STRUCT_DENSE_JOB_ID=${JOB_IDS[struct-dense]}
  export GRIT_PEPTIDES_STRUCT_1HOP_JOB_ID=${JOB_IDS[struct-1hop]}
  export GRIT_PEPTIDES_FUNC_DENSE_JOB_ID=${JOB_IDS[func-dense]}
  export GRIT_PEPTIDES_FUNC_1HOP_JOB_ID=${JOB_IDS[func-1hop]}
  export GRIT_PEPTIDES_RESULTS_ROOT=${RESULTS_ROOT}

Queue:
  squeue -j "\${GRIT_PEPTIDES_STRUCT_DENSE_JOB_ID},\${GRIT_PEPTIDES_STRUCT_1HOP_JOB_ID},\${GRIT_PEPTIDES_FUNC_DENSE_JOB_ID},\${GRIT_PEPTIDES_FUNC_1HOP_JOB_ID}"

Tail:
  tail -f \\
    "\${GRIT_PEPTIDES_RESULTS_ROOT}/slurm_submit_logs/grit-pep-struct-dense-s${SEED}-\${GRIT_PEPTIDES_STRUCT_DENSE_JOB_ID}.out" \\
    "\${GRIT_PEPTIDES_RESULTS_ROOT}/slurm_submit_logs/grit-pep-struct-dense-s${SEED}-\${GRIT_PEPTIDES_STRUCT_DENSE_JOB_ID}.err" \\
    "\${GRIT_PEPTIDES_RESULTS_ROOT}/slurm_submit_logs/grit-pep-struct-1hop-s${SEED}-\${GRIT_PEPTIDES_STRUCT_1HOP_JOB_ID}.out" \\
    "\${GRIT_PEPTIDES_RESULTS_ROOT}/slurm_submit_logs/grit-pep-struct-1hop-s${SEED}-\${GRIT_PEPTIDES_STRUCT_1HOP_JOB_ID}.err" \\
    "\${GRIT_PEPTIDES_RESULTS_ROOT}/slurm_submit_logs/grit-pep-func-dense-s${SEED}-\${GRIT_PEPTIDES_FUNC_DENSE_JOB_ID}.out" \\
    "\${GRIT_PEPTIDES_RESULTS_ROOT}/slurm_submit_logs/grit-pep-func-dense-s${SEED}-\${GRIT_PEPTIDES_FUNC_DENSE_JOB_ID}.err" \\
    "\${GRIT_PEPTIDES_RESULTS_ROOT}/slurm_submit_logs/grit-pep-func-1hop-s${SEED}-\${GRIT_PEPTIDES_FUNC_1HOP_JOB_ID}.out" \\
    "\${GRIT_PEPTIDES_RESULTS_ROOT}/slurm_submit_logs/grit-pep-func-1hop-s${SEED}-\${GRIT_PEPTIDES_FUNC_1HOP_JOB_ID}.err"

Accounting after completion:
  sacct -j "\${GRIT_PEPTIDES_STRUCT_DENSE_JOB_ID},\${GRIT_PEPTIDES_STRUCT_1HOP_JOB_ID},\${GRIT_PEPTIDES_FUNC_DENSE_JOB_ID},\${GRIT_PEPTIDES_FUNC_1HOP_JOB_ID}" --format=JobID,JobName%36,State,ExitCode,Elapsed,Timelimit,ReqTRES%80
EOF
