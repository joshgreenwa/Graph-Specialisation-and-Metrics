# Official GRIT ZINC Slurm Setup

This setup trains the official GRIT ZINC subset configuration and a 1-hop sparse-control variant from the official GRIT repo:

- Official source: <https://github.com/LiamMa/GRIT>
- Pinned commit: `6c988ea600a606fbb49a2246c64a2d37396b3ab5`
- Official config used unchanged for the baseline: `configs/GRIT/zinc-GRIT-RRWP.yaml`
- Expected trainable parameters for both runs: `473,473`

The official baseline uses complete-graph GRIT attention after RRWP edge padding. The `1hop` control keeps the same ZINC subset config, depth, hidden width, heads, RRWP dimensions, optimizer, schedule, and trainable modules, but switches the RRWP edge encoder to an existing masked path so each layer attends only over original molecular bonds plus self-loops. This is a parameter-matched control for global complete-graph attention, not a new tuned model.

The stricter `1hop-localrrwp` control additionally truncates RRWP values to local information only: it keeps the official 21-dimensional RRWP encoder for parameter matching, but zeroes channels above identity plus one-step random walks and keeps relative RRWP support on molecular bonds plus self only. Use a separate `DATA_DIR` for this variant so PyG does not reuse processed full-RRWP tensors from another run.

## Files

- `patches/0001-add-zinc-grit-1hop-control.patch`: patch applied to the pinned official GRIT repo.
- `scripts/prepare_grit_repo.sh`: clones the official repo, checks out the pinned commit, and applies the 1-hop plus local-RRWP patches.
- `scripts/run_grit_zinc_slurm.py`: launches GRIT and aborts if the logged parameter count differs from `473,473`.
- `slurm/train_grit_zinc_single.sbatch`: one GPU job for one variant/seed.
- `slurm/train_grit_zinc_array.sbatch`: 8-job array for `official` and `1hop` across seeds `41 42 43 44`.
- `env/create_grit_conda_env.sh`: conda environment matching the official README era: Python 3.9, Torch 1.12.1 CUDA 11.3, PyG 2.2.0.

## Environment

Create the environment once on the cluster login node or an interactive GPU node:

```bash
cd /Users/joshgreen/Documents/Graph\ Specialisation\ and\ Metrics
bash experiments/zinc/grit_official_slurm/env/create_grit_conda_env.sh
```

If your cluster uses a different CUDA module, edit the Torch/PyG wheel versions in `env/create_grit_conda_env.sh` before installing. The Slurm examples below activate `grit-zinc-py39-cu113` through `CONDA_ENV`; if `CONDA_ENV` is omitted, the scripts use the current Python environment.

## Submit Jobs

Official GRIT, one seed:

```bash
sbatch --export=ALL,CONDA_ENV=grit-zinc-py39-cu113,GRIT_VARIANT=official,SEED=41,DATA_DIR=/path/to/datasets,RESULTS_DIR=/path/to/results experiments/zinc/grit_official_slurm/slurm/train_grit_zinc_single.sbatch
```

1-hop sparse control, one seed:

```bash
sbatch --export=ALL,CONDA_ENV=grit-zinc-py39-cu113,GRIT_VARIANT=1hop,SEED=41,DATA_DIR=/path/to/datasets,RESULTS_DIR=/path/to/results experiments/zinc/grit_official_slurm/slurm/train_grit_zinc_single.sbatch
```

Strict 1-hop local-RRWP control, one seed:

```bash
sbatch --export=ALL,CONDA_ENV=grit-zinc-py39-cu113,GRIT_VARIANT=1hop-localrrwp,SEED=41,DATA_DIR=/path/to/localrrwp-datasets,RESULTS_DIR=/path/to/localrrwp-results experiments/zinc/grit_official_slurm/slurm/train_grit_zinc_single.sbatch
```

Both variants across four seeds:

```bash
sbatch --export=ALL,CONDA_ENV=grit-zinc-py39-cu113,DATA_DIR=/path/to/datasets,RESULTS_DIR=/path/to/results experiments/zinc/grit_official_slurm/slurm/train_grit_zinc_array.sbatch
```

Only the strict local-RRWP variant across three seeds:

```bash
sbatch --export=ALL,CONDA_ENV=grit-zinc-py39-cu113,GRIT_VARIANTS="1hop-localrrwp",SEEDS="41 42 43",DATA_DIR=/path/to/localrrwp-datasets,RESULTS_DIR=/path/to/localrrwp-results experiments/zinc/grit_official_slurm/slurm/train_grit_zinc_array.sbatch
```

`DATA_DIR` is passed to GRIT as `dataset.dir`; PyG will place the ZINC data under that root. `RESULTS_DIR` defaults to `experiments/zinc/grit_official_slurm/results`, which is ignored by the repository.

For a smoke test, add `MAX_EPOCH=1` to the exported variables. The runner still checks that both variants log `Num parameters: 473473`.

## Slurm Notes

The scripts request `--partition=gpu`, `--gres=gpu:1`, `8` CPUs, `48G` RAM, and `48:00:00`. Edit those `#SBATCH` lines for your cluster. Each job clones and patches GRIT under `${SLURM_TMPDIR}` by default to avoid races between array tasks.
