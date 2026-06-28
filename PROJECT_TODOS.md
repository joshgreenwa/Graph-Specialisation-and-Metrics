# Project To-Dos

HPC is currently down. Use this file as the launch order once it returns.

## Fidelity Audit Before Running Paper-Claim Experiments

- [ ] Resolve the Carriage Check observability issue before treating the validation suite as an exact reproduction of `method_validation.md`.
  - Markdown: each graph randomly chooses a planted set `S` of 3 atoms, node features are independent `x in R^8`, and the model is trained from graph to `y = sum_{a in S} w.x_a`.
  - Current implementation: adds a validation-only `selector_mask` so the small GT can know which atoms are in `S`.
  - Reason this was added: with random unobserved `S`, the target is not learnable from `x` and topology alone.
  - Action needed: either update the markdown to state that `S` is provided as a validation-only selector/marker, or redesign the task so random `S` is observable without changing the planned method.

- [ ] Align the Carriage Check reconstruction perturbation exactly.
  - Markdown: compare predicted influence `sum_i C[i,j]` against measured finite content swap `yhat(swap x_j) - yhat`.
  - Current implementation: compares IG against baseline-replacement influence so the reconstruction identity is exact for the validation-only model.
  - Action needed: implement the swap-measured panel exactly, or explicitly add a second baseline-replacement panel and keep the markdown's swap panel as the pass/fail figure.

- [ ] Confirm the Rank Check null definition.
  - Markdown says "randomly permuting the matrix's columns"; a single global column permutation leaves singular values unchanged.
  - Current implementation uses row-wise source shuffling, which matches the main procedure's distance-preserving source-label null.
  - Action needed: confirm that row-wise/distance-preserving source shuffling is the intended null for both validation and Step 5.

- [ ] Implement official GRIT forward/attention/patch hooks before running main Steps 0 and 2-5 as paper-claim analysis.
  - Current main runner is ready for artifact discovery, parameter-count checking, manifest/config/status outputs, and Step 1 plots from training stats.
  - It intentionally writes `requires_trained_model_intervention_hooks` for Steps 0 and 2-5 rather than inventing results.

- [ ] Verify official ZINC training outputs include all later analysis inputs.
  - Required later: checkpoints, per-epoch train/val stats, test metrics, config copies, dense and 1-hop parameter counts, seed, split identity, and checkpoint path.
  - Keep `train.ckpt_clean False`.

## 0. Common HPC Setup

Run once per shell on the login node, from the active `graphbench-algoreas` conda env:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics
git pull --ff-only

export PROJECT_ROOT="$PWD"
export PYTHON_BIN="$(which python)"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export DATA_DIR=/rds/user/jgg45/hpc-work/grit_zinc_data
export RESULTS_DIR=/rds/user/jgg45/hpc-work/grit_zinc_results
export METHOD_RESULTS_DIR=/rds/user/jgg45/hpc-work/method_validation_artifacts
export MAIN_RESULTS_DIR=/rds/user/jgg45/hpc-work/main_procedure_artifacts

mkdir -p \
  "${DATA_DIR}" \
  "${RESULTS_DIR}/slurm_submit_logs" \
  "${METHOD_RESULTS_DIR}/slurm_submit_logs" \
  "${MAIN_RESULTS_DIR}/slurm_submit_logs"

echo "PYTHON_BIN=${PYTHON_BIN}"
```

Use this status command template. `ReqGRES` is not valid on this cluster; use `ReqTRES`.

```bash
sacct -j "${JOB_ID}" \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Timelimit,ReqTRES%80
```

## 1. Method Validation Suite

- [ ] Do not treat this as final/pass-fail until the fidelity items above are resolved.
- [ ] Run once first as a full GPU job to verify artifacts, figures, and logs.

```bash
export METHOD_VALIDATION_JOB_ID=$(
  sbatch --parsable \
    --export=ALL,PYTHON_BIN="${PYTHON_BIN}",PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_PATH=experiments/methodology/configs/method_validation_default.yaml,OUTPUT_ROOT="${METHOD_RESULTS_DIR}",DEVICE=cuda \
    -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
    -N 1 --ntasks=1 --gres=gpu:1 \
    --time=04:00:00 \
    --job-name=method-validation \
    --output="${METHOD_RESULTS_DIR}/slurm_submit_logs/%x-%j.out" \
    --error="${METHOD_RESULTS_DIR}/slurm_submit_logs/%x-%j.err" \
    experiments/methodology/slurm/method_validation.sbatch
)
echo "METHOD_VALIDATION_JOB_ID=${METHOD_VALIDATION_JOB_ID}"
```

Tail and inspect:

```bash
squeue -j "${METHOD_VALIDATION_JOB_ID}"
tail -f \
  "${METHOD_RESULTS_DIR}/slurm_submit_logs/method-validation-${METHOD_VALIDATION_JOB_ID}.out" \
  "${METHOD_RESULTS_DIR}/slurm_submit_logs/method-validation-${METHOD_VALIDATION_JOB_ID}.err"
```

Find the latest artifact directory and rerender figures from cached metrics if needed:

```bash
export METHOD_ARTIFACT_ROOT="$(find "${METHOD_RESULTS_DIR}" -mindepth 1 -maxdepth 1 -type d ! -name slurm_submit_logs -print | sort | tail -n 1)"
echo "METHOD_ARTIFACT_ROOT=${METHOD_ARTIFACT_ROOT}"

"${PYTHON_BIN}" -m graph_specialisation_metrics.figures render \
  --artifact-root "${METHOD_ARTIFACT_ROOT}"
```

## 2. Main Procedure Dry-Run / Discovery

- [ ] Run after method-validation artifacts are confirmed.
- [ ] This is safe before ZINC training finishes; it should discover missing artifacts and write status files.

```bash
"${PYTHON_BIN}" -m graph_specialisation_metrics.main_procedure run \
  --config experiments/methodology/configs/zinc_main_procedure.yaml \
  --steps 0,1,2,3,4,5 \
  --output-root "${MAIN_RESULTS_DIR}/zinc" \
  --dry-run \
  --force
```

## 3. ZINC Training Smoke: Official GRIT and 1-Hop GRIT, Seed 41

- [ ] Run short smoke jobs first to verify conda/Python, official GRIT checkout, ZINC data, parameter count, checkpoint/stat output paths, and 1-hop patch.
- [ ] These are not scientific runs because `MAX_EPOCH=2`.
- [ ] Use short walltime because `12:00:00` previously hit `QOSMaxWallDurationPerJobLimit`.

```bash
export GRIT_EXTRA_CFG="train.ckpt_clean False num_workers 16"

export GRIT_DENSE_SMOKE_JOB_ID=$(
  sbatch --parsable \
    --export=ALL,WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,GRIT_VARIANT=official,SEED=41,MAX_EPOCH=2,DATA_DIR="${DATA_DIR}",RESULTS_DIR="${RESULTS_DIR}",GRIT_EXTRA_CFG="${GRIT_EXTRA_CFG}",PYTHON_BIN="${PYTHON_BIN}" \
    -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
    -N 1 --ntasks=1 --gres=gpu:1 \
    --time=00:45:00 \
    --job-name=grit-zinc-dense-smoke-s41 \
    --output="${RESULTS_DIR}/slurm_submit_logs/%x-%j.out" \
    --error="${RESULTS_DIR}/slurm_submit_logs/%x-%j.err" \
    experiments/zinc/grit_official_slurm/slurm/train_grit_zinc_single.sbatch
)
echo "GRIT_DENSE_SMOKE_JOB_ID=${GRIT_DENSE_SMOKE_JOB_ID}"

export GRIT_1HOP_SMOKE_JOB_ID=$(
  sbatch --parsable \
    --export=ALL,WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,GRIT_VARIANT=1hop,SEED=41,MAX_EPOCH=2,DATA_DIR="${DATA_DIR}",RESULTS_DIR="${RESULTS_DIR}",GRIT_EXTRA_CFG="${GRIT_EXTRA_CFG}",PYTHON_BIN="${PYTHON_BIN}" \
    -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
    -N 1 --ntasks=1 --gres=gpu:1 \
    --time=00:45:00 \
    --job-name=grit-zinc-1hop-smoke-s41 \
    --output="${RESULTS_DIR}/slurm_submit_logs/%x-%j.out" \
    --error="${RESULTS_DIR}/slurm_submit_logs/%x-%j.err" \
    experiments/zinc/grit_official_slurm/slurm/train_grit_zinc_single.sbatch
)
echo "GRIT_1HOP_SMOKE_JOB_ID=${GRIT_1HOP_SMOKE_JOB_ID}"
```

Tail smoke logs:

```bash
squeue -j "${GRIT_DENSE_SMOKE_JOB_ID},${GRIT_1HOP_SMOKE_JOB_ID}"
tail -f \
  "${RESULTS_DIR}/slurm_submit_logs/grit-zinc-dense-smoke-s41-${GRIT_DENSE_SMOKE_JOB_ID}.out" \
  "${RESULTS_DIR}/slurm_submit_logs/grit-zinc-dense-smoke-s41-${GRIT_DENSE_SMOKE_JOB_ID}.err" \
  "${RESULTS_DIR}/slurm_submit_logs/grit-zinc-1hop-smoke-s41-${GRIT_1HOP_SMOKE_JOB_ID}.out" \
  "${RESULTS_DIR}/slurm_submit_logs/grit-zinc-1hop-smoke-s41-${GRIT_1HOP_SMOKE_JOB_ID}.err"
```

## 4. ZINC Single-Seed Training: Official GRIT and 1-Hop GRIT, Seed 41

- [ ] Launch only after both smoke jobs finish cleanly and report matching `Num parameters: 473,473`.
- [ ] These are the first real one-seed training runs.
- [ ] If 4 hours is too short for full convergence, use the checkpoint/stat outputs for diagnosis and move to an allowed longer QOS/partition rather than changing back to the rejected 12-hour command.

Dense GRIT:

```bash
export GRIT_JOB_ID=$(
  sbatch --parsable \
    --export=ALL,WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,GRIT_VARIANT=official,SEED=41,DATA_DIR="${DATA_DIR}",RESULTS_DIR="${RESULTS_DIR}",GRIT_EXTRA_CFG="${GRIT_EXTRA_CFG}",PYTHON_BIN="${PYTHON_BIN}" \
    -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
    -N 1 --ntasks=1 --gres=gpu:1 \
    --time=04:00:00 \
    --job-name=grit-zinc-dense-s41 \
    --output="${RESULTS_DIR}/slurm_submit_logs/%x-%j.out" \
    --error="${RESULTS_DIR}/slurm_submit_logs/%x-%j.err" \
    experiments/zinc/grit_official_slurm/slurm/train_grit_zinc_single.sbatch
)
echo "GRIT_JOB_ID=${GRIT_JOB_ID}"
```

1-hop GRIT:

```bash
export GRIT_1HOP_JOB_ID=$(
  sbatch --parsable \
    --export=ALL,WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,GRIT_VARIANT=1hop,SEED=41,DATA_DIR="${DATA_DIR}",RESULTS_DIR="${RESULTS_DIR}",GRIT_EXTRA_CFG="${GRIT_EXTRA_CFG}",PYTHON_BIN="${PYTHON_BIN}" \
    -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
    -N 1 --ntasks=1 --gres=gpu:1 \
    --time=04:00:00 \
    --job-name=grit-zinc-1hop-s41 \
    --output="${RESULTS_DIR}/slurm_submit_logs/%x-%j.out" \
    --error="${RESULTS_DIR}/slurm_submit_logs/%x-%j.err" \
    experiments/zinc/grit_official_slurm/slurm/train_grit_zinc_single.sbatch
)
echo "GRIT_1HOP_JOB_ID=${GRIT_1HOP_JOB_ID}"
```

Tail both real one-seed logs:

```bash
squeue -j "${GRIT_JOB_ID},${GRIT_1HOP_JOB_ID}"
tail -f \
  "${RESULTS_DIR}/slurm_submit_logs/grit-zinc-dense-s41-${GRIT_JOB_ID}.out" \
  "${RESULTS_DIR}/slurm_submit_logs/grit-zinc-dense-s41-${GRIT_JOB_ID}.err" \
  "${RESULTS_DIR}/slurm_submit_logs/grit-zinc-1hop-s41-${GRIT_1HOP_JOB_ID}.out" \
  "${RESULTS_DIR}/slurm_submit_logs/grit-zinc-1hop-s41-${GRIT_1HOP_JOB_ID}.err"
```

## 5. Main Procedure Step 1 After One-Seed ZINC Stats Exist

- [ ] Run after the dense and 1-hop seed-41 runs produce stats/checkpoints.
- [ ] This should create artifact discovery, adapter checks, and Step 1 plots if stats are discoverable.

```bash
export MAIN_STEP1_JOB_ID=$(
  sbatch --parsable \
    --export=ALL,PYTHON_BIN="${PYTHON_BIN}",PROJECT_ROOT="${PROJECT_ROOT}",CONFIG_PATH=experiments/methodology/configs/zinc_main_procedure.yaml,OUTPUT_ROOT="${MAIN_RESULTS_DIR}/zinc",STEP_OVERRIDE=1 \
    -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
    -N 1 --ntasks=1 --gres=gpu:1 \
    --time=01:00:00 \
    --array=1 \
    --job-name=main-zinc-step1 \
    --output="${MAIN_RESULTS_DIR}/slurm_submit_logs/%x-%A_%a.out" \
    --error="${MAIN_RESULTS_DIR}/slurm_submit_logs/%x-%A_%a.err" \
    experiments/methodology/slurm/main_procedure_array.sbatch
)
echo "MAIN_STEP1_JOB_ID=${MAIN_STEP1_JOB_ID}"
```

Tail Step 1:

```bash
tail -f \
  "${MAIN_RESULTS_DIR}/slurm_submit_logs/main-zinc-step1-${MAIN_STEP1_JOB_ID}_1.out" \
  "${MAIN_RESULTS_DIR}/slurm_submit_logs/main-zinc-step1-${MAIN_STEP1_JOB_ID}_1.err"
```

## 6. ZINC Full Three-Seed Variant

- [ ] Launch only after seed-41 dense and 1-hop are clean enough to trust.
- [ ] This launches 6 array tasks: `{official, 1hop} x {41, 42, 43}` with at most 2 concurrent jobs.

```bash
export SEEDS="41 42 43"

export GRIT_3SEED_JOB_ID=$(
  sbatch --parsable \
    --export=ALL,WANDB_API_KEY,WANDB_MODE,WANDB_PROJECT,SEEDS="${SEEDS}",DATA_DIR="${DATA_DIR}",RESULTS_DIR="${RESULTS_DIR}",GRIT_EXTRA_CFG="${GRIT_EXTRA_CFG}",PYTHON_BIN="${PYTHON_BIN}" \
    -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
    -N 1 --ntasks=1 --gres=gpu:1 \
    --array=0-5%2 \
    --time=04:00:00 \
    --job-name=grit-zinc-3seed \
    --output="${RESULTS_DIR}/slurm_submit_logs/%x-%A_%a.out" \
    --error="${RESULTS_DIR}/slurm_submit_logs/%x-%A_%a.err" \
    experiments/zinc/grit_official_slurm/slurm/train_grit_zinc_array.sbatch
)
echo "GRIT_3SEED_JOB_ID=${GRIT_3SEED_JOB_ID}"
```

Tail the three-seed array logs:

```bash
squeue -j "${GRIT_3SEED_JOB_ID}"
tail -f \
  "${RESULTS_DIR}/slurm_submit_logs/grit-zinc-3seed-${GRIT_3SEED_JOB_ID}"_*.out \
  "${RESULTS_DIR}/slurm_submit_logs/grit-zinc-3seed-${GRIT_3SEED_JOB_ID}"_*.err
```

## 7. Later Main Procedure Steps

- [ ] Do not launch Steps 0 and 2-5 as scientific runs until official GRIT intervention hooks are implemented and tested.
- [ ] Once hooks are implemented, run in order: Step 0 gate, Step 1 final aggregation, Step 2, Step 3, Step 4, Step 5.
- [ ] Re-render figures from cached metrics after each methodology/plotting update:

```bash
export MAIN_ARTIFACT_ROOT="$(find "${MAIN_RESULTS_DIR}/zinc" -mindepth 1 -maxdepth 1 -type d ! -name slurm_submit_logs -print | sort | tail -n 1)"
echo "MAIN_ARTIFACT_ROOT=${MAIN_ARTIFACT_ROOT}"

"${PYTHON_BIN}" -m graph_specialisation_metrics.figures render \
  --artifact-root "${MAIN_ARTIFACT_ROOT}"
```
