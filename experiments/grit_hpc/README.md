# GRIT multi-dataset Slurm runner

This runner launches one model per Slurm array task (and therefore one model
per GPU) while keeping the validated task-specific settings already used by
the repository:

| Task | Target / metric | Training configuration |
|---|---|---|
| `zinc` | ZINC subset regression | Official GRIT ZINC config |
| `qm9_gap` | QM9 HOMO-LUMO gap (`y[:, 4]`) | Existing QM9-gap + RRWP config |
| `peptides_func` | Peptides-func | Official GRIT Peptides-func config |
| `peptides_struct` | Peptides-struct | Official GRIT Peptides-struct config |

Each task accepts dense attention or arbitrary k-hop attention, optionally
with a global VNode. QM9 and both Peptides tasks also accept local RRWP
variants. ZINC local RRWP is intentionally rejected because it has not been
combined with the general ZINC runner.

## jgg45 CSD3 launch: 60 GPUs within a 400-hour balance

The prepared CSD3 launcher creates exactly this grid:

- Tasks: ZINC, QM9 gap, Peptides-func, Peptides-struct
- Models: dense, 1-hop, 1-hop+VNode, 2-hop, 2-hop+VNode
- Seeds: 0, 1, 2
- Total: 60 independent array rows, requested as `0-59%60`
- Training limit: 6 hours per row (`360` requested GPU-hours)
- Dataset staging limit: one Sapphire/INTR job, 2 CPUs for 1 hour (`2` CPU-core-hours)
- Initial GPU request total: at most `360` GPU-hours

From the repository checkout on RDS:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics

# Optional if the current shell environment is not sufficient on compute nodes:
export ENV_ACTIVATE=/rds/user/jgg45/hpc-work/venvs/grit/bin/activate

bash experiments/grit_hpc/bin/submit_csd3_60.sh
```

The prepared CSD3 launcher deliberately forces W&B off so an inherited
`GRIT_WANDB` or stale API key cannot make a training job fail. Slurm job IDs,
logs, checkpoints, and progress remain available through the local tracking
ledger and status command below.

The script uses:

```text
account:    mlmi-jgg45-sl2-gpu
partition:  ampere
qos:        gpu1
GPU/task:   1
array:      0-59%60
time:       06:00:00 per training row
```

Persistent state is stored at:

```text
/rds/user/jgg45/hpc-work/grit_shared_datasets/
/rds/user/jgg45/hpc-work/grit_manifests/grit_all_3seeds.jsonl
/rds/user/jgg45/hpc-work/grit_manifests/grit_all_3seeds.jobs.tsv
/rds/user/jgg45/hpc-work/grit_checkpoints/grit_all_3seeds/<run_id>/
```

Immediately after `sbatch`, `grit_all_3seeds.jobs.tsv` records every model's
array index, Slurm handle (`<array-job-id>_<index>`), run ID, task, variant,
seed, output directory, and stdout/stderr paths. The single dataset-staging ID
is written to `grit_dataset_staging.jobs.tsv`. Each model also records the actual
`SLURM_JOB_ID`, array ID, task ID, and node under its `hpc_attempts/` directory.

Display the mappings plus current and historical Slurm states with:

```bash
bash experiments/grit_hpc/bin/status_csd3.sh
```

For continuous progress updates:

```bash
watch -n 30 bash experiments/grit_hpc/bin/status_csd3.sh
```

Pass a resubmission's parent JobID to inspect that attempt instead of the
original array:

```bash
bash experiments/grit_hpc/bin/status_csd3.sh 12345678
```

Recovery checkpoints are written after the first epoch, whenever validation
improves, and every 10 epochs. Both stable `latest.ckpt` and `best.ckpt` copies
are retained alongside GraphGym's resumable checkpoints. Scientific model and
optimizer settings are unchanged.

## Export validation-selected saved checkpoints

Completed ZINC and QM9 runs can be exported without loading a GPU. The exporter
joins wrapper-log metrics to checkpoint epochs that still exist, selects by
validation MAE only, copies the 30 selected files, and records source paths,
checksums, selected metrics, and whether each file is the exact logged-global
best or the best available saved snapshot:

```bash
STAMP="$(date +%Y%m%d_%H%M%S)"
EXPORT_PARENT=/rds/user/jgg45/hpc-work/grit_exports
EXPORT_NAME="zinc_qm9_best_available_${STAMP}"
mkdir -p "$EXPORT_PARENT"

python experiments/grit_hpc/bin/export_best_available.py \
  --input-root /rds/user/jgg45/hpc-work/grit_checkpoints/grit_all_3seeds \
  --output-dir "$EXPORT_PARENT/$EXPORT_NAME"

tar -C "$EXPORT_PARENT" -cf "$EXPORT_PARENT/$EXPORT_NAME.tar" "$EXPORT_NAME"
sha256sum "$EXPORT_PARENT/$EXPORT_NAME.tar" | tee "$EXPORT_PARENT/$EXPORT_NAME.tar.sha256"
```

Inspect `manifest.tsv` before treating the resulting tar as canonical. The
archive calls non-exact selections *best available validation-selected saved
checkpoints* and never represents them as exact global-best snapshots.

If array row 4 is interrupted, resubmit only that model with:

```bash
bash experiments/grit_hpc/bin/resubmit_csd3.sh 4
```

Multiple rows and a concurrency cap use normal Slurm array syntax:

```bash
bash experiments/grit_hpc/bin/resubmit_csd3.sh '4,17,39-42%8'
```

The original manifest and output locations are reused, and auto-resume is on.
The index blocks are ZINC `0-14`, QM9 `15-29`, Peptides-func `30-44`, and
Peptides-struct `45-59`. Run `grit_hpc.py print-jobs` for the exact row mapping.

The 360-hour figure is a cap for the initial GPU submission, not a guarantee that
all models finish within six hours. Any resubmission consumes additional
allocation. The first run therefore also provides exact per-task runtime data;
use that before spending the remaining 40 GPU-hours.

## One-time cluster setup

Create a Python environment containing the dependencies required by the
existing GRIT runners. Array jobs pass `--skip-install` and
`--skip-editable-install`: no job mutates a shared Python environment.

Copy and edit the environment example:

```bash
cp experiments/grit_hpc/env.example experiments/grit_hpc/env.local
source experiments/grit_hpc/env.local
mkdir -p "$PROJECT_ROOT/logs" "$(dirname "$GRIT_MANIFEST")"
```

`GRIT_DATA_ROOT`, `GRIT_OUTPUT_ROOT`, and `GRIT_MANIFEST` must live on storage
visible from every compute node. `GRIT_SCRATCH_ROOT` should be node-local when
possible. A shared pristine GRIT clone can reduce GitHub traffic:

```bash
git clone https://github.com/LiamMa/GRIT.git "$GRIT_SOURCE_REPO"
git -C "$GRIT_SOURCE_REPO" checkout 6c988ea600a606fbb49a2246c64a2d37396b3ab5
```

Every GPU job makes and patches its own checkout under scratch. This avoids
concurrent edits to either the official source clone or another model's code.

## Create a model manifest

The default architecture set is dense, 1-hop, 1-hop+VNode, 2-hop, and
2-hop+VNode. For example, two seeds across all four tasks produce 40 rows:

```bash
python "$PROJECT_ROOT/experiments/grit_hpc/bin/grit_hpc.py" make-manifest \
  --tasks zinc,qm9_gap,peptides_func,peptides_struct \
  --variants dense,1hop,1hop_vnode,2hop,2hop_vnode \
  --seeds 0,1 \
  --output "$GRIT_MANIFEST"

python "$PROJECT_ROOT/experiments/grit_hpc/bin/grit_hpc.py" print-jobs \
  --manifest "$GRIT_MANIFEST"
```

Variant syntax is:

```text
dense
dense_vnode
1hop
1hop_vnode
2hop
2hop_vnode
1hop_localrrwp_h1
1hop_vnode_localrrwp_h1
```

The final number is the RRWP horizon. Omit `_localrrwp_hH` to retain the
existing global RRWP channels. Any valid k can be used within the task's RRWP
limit (ZINC/QM9 20, Peptides-func 16, Peptides-struct 23).

## Launch everything

The simplest launch submits dataset staging and the GPU array with an `afterok`
dependency:

```bash
bash "$PROJECT_ROOT/experiments/grit_hpc/bin/submit_all.sh"
```

The four datasets are staged sequentially in one INTR job. After it succeeds, the model array
starts with one model per GPU. Existing readiness markers make later staging
submissions cheap no-ops.

## Or stage and train separately

Do this before the GPU array:

```bash
(cd "$PROJECT_ROOT" && sbatch experiments/grit_hpc/slurm/stage_datasets.sbatch)
```

The staging job processes the four datasets sequentially and writes a
versioned readiness marker for each. ZINC and QM9 are staged at the nested PyG
cache roots used by GRIT (`zinc/ZINC` and `qm9_gap/QM9`). GPU workers fail early
if a marker is absent or stale, rather than racing to download/process the same
shared cache. RRWP statistics are still computed by each model using that
model's configured horizon.

Then launch the GPU array:

```bash
bash "$PROJECT_ROOT/experiments/grit_hpc/bin/submit_array.sh"
```

The submit helper reads the manifest length and supplies the exact Slurm array
range. Set `GRIT_MAX_PARALLEL` to cap simultaneous GPUs; unset it to permit the
whole manifest to run concurrently. Cluster-specific partition/account/QOS
options can be added to `train_array.sbatch` or supplied through local Slurm
defaults.

Each run is isolated at:

```text
$GRIT_OUTPUT_ROOT/<run_id>/
```

That directory contains `hpc_job.json`, wrapper logs, patch provenance,
results, recovery checkpoints, and checkpoint audit/inventory files. Repeating
the same manifest automatically resumes from that run's existing checkpoints;
a changed job is not allowed to reuse the same `run_id` directory.

## Dry-run one row

After staging data, inspect the exact task-runner arguments without launching
training:

```bash
python "$PROJECT_ROOT/experiments/grit_hpc/bin/grit_hpc.py" run \
  --manifest "$GRIT_MANIFEST" \
  --array-index 0 \
  --dataset-root "$GRIT_DATA_ROOT" \
  --output-root "$GRIT_OUTPUT_ROOT" \
  --scratch-root "${GRIT_SCRATCH_ROOT:-/tmp/grit_hpc}" \
  --grit-source "${GRIT_SOURCE_REPO:-https://github.com/LiamMa/GRIT.git}" \
  --dry-run
```
