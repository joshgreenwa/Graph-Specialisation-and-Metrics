#!/bin/bash
#SBATCH -J min-order-v3
#SBATCH -o logs/min-order-v3-%j.out
#SBATCH -e logs/min-order-v3-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00

set -euo pipefail

: "${MOD_OUTPUT_ROOT:?Set MOD_OUTPUT_ROOT to the minimal-order artifact directory}"
: "${MOD_CONDA_ENV:=graphbench-algoreas}"

source /usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh
conda activate "${MOD_CONDA_ENV}"

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
mkdir -p logs

EXTRA_FLAGS=()
if [[ "${MOD_RUN_K_SWEEP:-0}" == "1" || "${MOD_RUN_K_SWEEP:-false}" == "true" ]]; then
  EXTRA_FLAGS+=(--run-k-sweep)
fi
if [[ "${MOD_RUN_METHODOLOGY:-1}" == "0" || "${MOD_RUN_METHODOLOGY:-true}" == "false" ]]; then
  EXTRA_FLAGS+=(--no-run-methodology)
fi

python -u -m graph_specialisation_metrics.minimal_order_dissociation_experiment run-all \
  --output-root "${MOD_OUTPUT_ROOT}" \
  --tasks "${MOD_TASKS:-add,prod,gap,relmode}" \
  --models "${MOD_MODELS:-mpnn,mpnn_vn,nope_gt,gt_route,gt_full}" \
  --seeds "${MOD_SEEDS:-1001,1002,1003}" \
  --n-nodes "${MOD_N_NODES:-32}" \
  --receptive-radius "${MOD_RECEPTIVE_RADIUS:-4}" \
  --mode-count "${MOD_MODE_COUNT:-6}" \
  --train-size "${MOD_TRAIN_SIZE:-5000}" \
  --val-size "${MOD_VAL_SIZE:-1000}" \
  --test-size "${MOD_TEST_SIZE:-1000}" \
  --data-seed "${MOD_DATA_SEED:-7001}" \
  --hidden-dim "${MOD_HIDDEN_DIM:-128}" \
  --gt-heads "${MOD_GT_HEADS:-4}" \
  --batch-size "${MOD_BATCH_SIZE:-1024}" \
  --eval-batch-size "${MOD_EVAL_BATCH_SIZE:-2048}" \
  --max-epochs "${MOD_MAX_EPOCHS:-800}" \
  --patience "${MOD_PATIENCE:-100}" \
  --lr "${MOD_LR:-0.002}" \
  --weight-decay "${MOD_WEIGHT_DECAY:-0.0001}" \
  --resamples-per-site "${MOD_RESAMPLES_PER_SITE:-4}" \
  --rq3-seed "${MOD_RQ3_SEED:-8101}" \
  --methodology-graphs "${MOD_METHODOLOGY_GRAPHS:-200}" \
  --k-sweep-values "${MOD_K_SWEEP_VALUES:-1,2,3,4,5,6,7,8}" \
  --k-sweep-models "${MOD_K_SWEEP_MODELS:-mpnn,mpnn_vn,nope_gt,gt_route,gt_full}" \
  --k-sweep-train-size "${MOD_K_SWEEP_TRAIN_SIZE:-3000}" \
  --k-sweep-max-epochs "${MOD_K_SWEEP_MAX_EPOCHS:-400}" \
  --k-sweep-patience "${MOD_K_SWEEP_PATIENCE:-80}" \
  --log-every "${MOD_LOG_EVERY:-20}" \
  --device cuda \
  "${EXTRA_FLAGS[@]}"
