#!/bin/bash
#SBATCH -J transport-rank-trainability
#SBATCH -o logs/transport-rank-trainability-%j.out
#SBATCH -e logs/transport-rank-trainability-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00

set -euo pipefail

: "${TRAINABILITY_OUTPUT_ROOT:=${PWD}/artifacts/transport_rank_trainability}"
: "${TRAINABILITY_CONDA_ENV:=graphbench-algoreas}"
: "${TRAINABILITY_STAGE:=run}"
: "${TRAINABILITY_SMOKE:=0}"

source /usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh
conda activate "${TRAINABILITY_CONDA_ENV}"

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
mkdir -p logs

COMMON_FLAGS=(
  --output-root "${TRAINABILITY_OUTPUT_ROOT}"
  --device "${TRAINABILITY_DEVICE:-cuda}"
  --n-nodes "${TRAINABILITY_N_NODES:-256}"
  --heads "${TRAINABILITY_HEADS:-4}"
  --content-dim "${TRAINABILITY_CONTENT_DIM:-16}"
  --transport-dim "${TRAINABILITY_TRANSPORT_DIM:-8}"
  --relation-types "${TRAINABILITY_RELATION_TYPES:-32}"
  --batch-size "${TRAINABILITY_BATCH_SIZE:-128}"
  --eval-batch-size "${TRAINABILITY_EVAL_BATCH_SIZE:-256}"
  --steps "${TRAINABILITY_STEPS:-6000}"
  --smoke-steps "${TRAINABILITY_SMOKE_STEPS:-2000}"
  --lr "${TRAINABILITY_LR:-0.003}"
  --weight-decay "${TRAINABILITY_WEIGHT_DECAY:-0.0}"
  --seeds "${TRAINABILITY_SEEDS:-0,1,2}"
  --rho-g-sweep "${TRAINABILITY_RHO_G_SWEEP:-1,2,3,4,5,6,7,8}"
  --smoke-rho-g-sweep "${TRAINABILITY_SMOKE_RHO_G_SWEEP:-1,4,8}"
  --dynamics-rho-g "${TRAINABILITY_DYNAMICS_RHO_G:-8}"
  --eval-every "${TRAINABILITY_EVAL_EVERY:-250}"
  --log-every "${TRAINABILITY_LOG_EVERY:-250}"
  --rank-energy "${TRAINABILITY_RANK_ENERGY:-0.99}"
)

if [[ "${TRAINABILITY_FORCE:-0}" == "1" || "${TRAINABILITY_FORCE:-false}" == "true" ]]; then
  COMMON_FLAGS+=(--force)
fi

if [[ "${TRAINABILITY_SMOKE}" == "1" || "${TRAINABILITY_SMOKE}" == "true" ]]; then
  COMMON_FLAGS+=(--smoke)
fi

case "${TRAINABILITY_STAGE}" in
  run)
    python -u -m graph_specialisation_metrics.transport_rank_trainability run "${COMMON_FLAGS[@]}"
    ;;
  plot)
    python -u -m graph_specialisation_metrics.transport_rank_trainability plot "${COMMON_FLAGS[@]}"
    ;;
  *)
    echo "Unknown TRAINABILITY_STAGE=${TRAINABILITY_STAGE}. Use run or plot." >&2
    exit 2
    ;;
esac
