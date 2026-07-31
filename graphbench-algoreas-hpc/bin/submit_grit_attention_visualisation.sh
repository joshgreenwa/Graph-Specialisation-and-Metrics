#!/bin/bash
# Submit one CPU-only clean-attention extraction and publication render.

set -euo pipefail

HPC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${HPC_ROOT}/.." && pwd)"
ENV_ACTIVATE="${ENV_ACTIVATE:-${HPC_ROOT}/activate_graphbench_algoreas}"
ANALYSIS_ROOT="${GRAPHBENCH_ANALYSIS_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/grit_specialisation_bipartite_complete_pe_causal_v3}"
TRAINING_ROOT="${GRAPHBENCH_TRAINING_OUTPUT_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/outputs/graphbench_algoreas_hpc_base_v1}"
DATASET_ROOT="${GRAPHBENCH_DATASET_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/datasets}"
PE_ROOT="${GRAPHBENCH_PE_CACHE_ROOT:-/rds/user/jgg45/hpc-work/graphbench-algoreas/pe_cache}"
GRIT_ROOT="${GRIT_ROOT:-${PROJECT_ROOT}/external/GRIT}"
ATTENTION_SEED="${ATTENTION_SEED:-0}"
ATTENTION_GRAPH_ID="${ATTENTION_GRAPH_ID:-0}"
ATTENTION_TOP_K="${ATTENTION_TOP_K:-2}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

if [[ ! -f "${ENV_ACTIVATE}" ]]; then
  echo "Missing environment activation file: ${ENV_ACTIVATE}" >&2
  exit 2
fi
SCORE_CACHE="${ANALYSIS_ROOT}/graphbench_bipartite_matching_hard/seed_${ATTENTION_SEED}/cache/scores/raw.pt"
MODEL_RECORD="${ANALYSIS_ROOT}/graphbench_bipartite_matching_hard/seed_${ATTENTION_SEED}/model.json"
if [[ ! -s "${SCORE_CACHE}" || ! -s "${MODEL_RECORD}" ]]; then
  echo "Missing seed-${ATTENTION_SEED} score cache or model record under ${ANALYSIS_ROOT}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "${HPC_ROOT}/logs"
COMMON_EXPORT="ALL,PROJECT_ROOT=${PROJECT_ROOT},GRIT_ROOT=${GRIT_ROOT},ENV_ACTIVATE=${ENV_ACTIVATE},GRAPHBENCH_ANALYSIS_OUTPUT_ROOT=${ANALYSIS_ROOT},GRAPHBENCH_TRAINING_OUTPUT_ROOT=${TRAINING_ROOT},GRAPHBENCH_DATASET_ROOT=${DATASET_ROOT},GRAPHBENCH_PE_CACHE_ROOT=${PE_ROOT},ATTENTION_SEED=${ATTENTION_SEED},ATTENTION_GRAPH_ID=${ATTENTION_GRAPH_ID},ATTENTION_TOP_K=${ATTENTION_TOP_K}"

JOB_ID="$(
  "${SBATCH_BIN}" --parsable \
    -A mlmi-jgg45-sl2-cpu \
    -p sapphire \
    --qos=cpu1 \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=32G \
    --time=01:00:00 \
    --job-name=gb-grit-attn \
    --chdir="${PROJECT_ROOT}" \
    --output="${HPC_ROOT}/logs/%x-%j.out" \
    --error="${HPC_ROOT}/logs/%x-%j.err" \
    --export="${COMMON_EXPORT}" \
    graphbench-algoreas-hpc/slurm/grit_attention_visualisation.sbatch
)"
JOB_ID="${JOB_ID%%;*}"

echo "attention_job=${JOB_ID}"
echo "analysis_root=${ANALYSIS_ROOT}"
echo "seed=${ATTENTION_SEED} graph=${ATTENTION_GRAPH_ID} top_k=${ATTENTION_TOP_K}"
echo "monitor: squeue -j ${JOB_ID}"
