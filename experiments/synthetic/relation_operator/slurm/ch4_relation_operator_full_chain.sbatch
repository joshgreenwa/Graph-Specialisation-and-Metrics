#!/bin/bash
#SBATCH -J ch4-relop
#SBATCH -o logs/ch4-relop-%j.out
#SBATCH -e logs/ch4-relop-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00

set -euo pipefail

: "${CH4_OUTPUT_ROOT:=${PWD}/artifacts/ch4_relation_operator_capacity}"
: "${CH4_CONDA_ENV:=graphbench-algoreas}"
: "${CH4_STAGE:=full}"

source /usr/local/Cluster-Apps/miniconda3/4.5.1/etc/profile.d/conda.sh
conda activate "${CH4_CONDA_ENV}"

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p logs

COMMON_FLAGS=(
  --output-root "${CH4_OUTPUT_ROOT}"
  --seeds "${CH4_SEEDS:-1001,1002,1003}"
  --r-sweep "${CH4_R_SWEEP:-1,2,4,8,12,16}"
  --noise-sweep "${CH4_NOISE_SWEEP:-0,0.25,0.5,1,2,4}"
  --relation-types "${CH4_RELATION_TYPES:-16}"
  --train-size "${CH4_TRAIN_SIZE:-8192}"
  --val-size "${CH4_VAL_SIZE:-2048}"
  --test-size "${CH4_TEST_SIZE:-2048}"
  --n-nodes "${CH4_N_NODES:-32}"
  --input-dim "${CH4_INPUT_DIM:-32}"
  --target-dim "${CH4_TARGET_DIM:-32}"
  --hidden-dim "${CH4_HIDDEN_DIM:-64}"
  --heads "${CH4_HEADS:-4}"
  --transport-bases "${CH4_TRANSPORT_BASES:-4}"
  --batch-size "${CH4_BATCH_SIZE:-2048}"
  --eval-batch-size "${CH4_EVAL_BATCH_SIZE:-4096}"
  --max-epochs "${CH4_MAX_EPOCHS:-250}"
  --patience "${CH4_PATIENCE:-25}"
  --lr "${CH4_LR:-0.002}"
  --weight-decay "${CH4_WEIGHT_DECAY:-0.0001}"
  --num-workers "${CH4_NUM_WORKERS:-8}"
  --device "${CH4_DEVICE:-cuda}"
)

if [[ "${CH4_USE_AMP:-1}" == "1" || "${CH4_USE_AMP:-true}" == "true" ]]; then
  COMMON_FLAGS+=(--use-amp)
else
  COMMON_FLAGS+=(--no-amp)
fi

PRACTICAL_FLAGS=()
if [[ "${CH4_INCLUDE_PRACTICAL:-0}" == "1" || "${CH4_INCLUDE_PRACTICAL:-false}" == "true" ]]; then
  PRACTICAL_FLAGS+=(--include-practical)
fi

case "${CH4_STAGE}" in
  full)
    python -u -m graph_specialisation_metrics.relation_operator_capacity run-all \
      "${COMMON_FLAGS[@]}" "${PRACTICAL_FLAGS[@]}"
    ;;
  controlled)
    python -u -m graph_specialisation_metrics.relation_operator_capacity run-all \
      "${COMMON_FLAGS[@]}"
    ;;
  practical)
    python -u -m graph_specialisation_metrics.relation_operator_capacity run-transport-support \
      "${COMMON_FLAGS[@]}" --include-practical
    python -u -m graph_specialisation_metrics.relation_operator_capacity run-overglobalisation \
      "${COMMON_FLAGS[@]}" --include-practical
    python -u -m graph_specialisation_metrics.relation_operator_capacity plot \
      --output-root "${CH4_OUTPUT_ROOT}"
    ;;
  plot)
    python -u -m graph_specialisation_metrics.relation_operator_capacity evaluate \
      --output-root "${CH4_OUTPUT_ROOT}"
    python -u -m graph_specialisation_metrics.relation_operator_capacity plot \
      --output-root "${CH4_OUTPUT_ROOT}"
    ;;
  crossover)
    python -u -m graph_specialisation_metrics.relation_operator_capacity run-crossover \
      "${COMMON_FLAGS[@]}"
    ;;
  capacity-crossover)
    python -u -m graph_specialisation_metrics.relation_operator_capacity run-capacity-crossover \
      "${COMMON_FLAGS[@]}"
    python -u -m graph_specialisation_metrics.relation_operator_capacity plot \
      --output-root "${CH4_OUTPUT_ROOT}"
    ;;
  transport-support)
    python -u -m graph_specialisation_metrics.relation_operator_capacity run-transport-support \
      "${COMMON_FLAGS[@]}" "${PRACTICAL_FLAGS[@]}"
    ;;
  transport-support-official)
    python -u -m graph_specialisation_metrics.relation_operator_capacity run-transport-support-official \
      "${COMMON_FLAGS[@]}"
    python -u -m graph_specialisation_metrics.relation_operator_capacity plot \
      --output-root "${CH4_OUTPUT_ROOT}"
    ;;
  overglobalisation)
    python -u -m graph_specialisation_metrics.relation_operator_capacity run-overglobalisation \
      "${COMMON_FLAGS[@]}" "${PRACTICAL_FLAGS[@]}"
    ;;
  overglobalisation-grit)
    python -u -m graph_specialisation_metrics.relation_operator_capacity run-overglobalisation-grit \
      "${COMMON_FLAGS[@]}"
    python -u -m graph_specialisation_metrics.relation_operator_capacity plot \
      --output-root "${CH4_OUTPUT_ROOT}"
    ;;
  depth-escape)
    python -u -m graph_specialisation_metrics.relation_operator_capacity run-depth-escape \
      "${COMMON_FLAGS[@]}"
    ;;
  *)
    echo "Unknown CH4_STAGE=${CH4_STAGE}. Use full, controlled, practical, plot, crossover, capacity-crossover, transport-support, transport-support-official, overglobalisation, overglobalisation-grit, or depth-escape." >&2
    exit 2
    ;;
esac
