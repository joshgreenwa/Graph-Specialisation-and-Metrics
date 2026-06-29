# Methodology Experiments

This directory contains config and Slurm entry points for the dissertation core
procedure and the method-validation suite.

## Local smoke runs

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m graph_specialisation_metrics.method_validation run-all \
  --config experiments/methodology/configs/method_validation_default.yaml \
  --fast-dev-run --device cpu --force

python -m graph_specialisation_metrics.main_procedure run \
  --config experiments/methodology/configs/main_procedure_default.yaml \
  --steps 0,1 --dry-run --force
```

## Colab synthetic validation

The standalone Colab runner mounts Drive, clones this GitHub repo with the Colab
secret `dissertation_key`, caches prepared synthetic data and the trained validation
model on Drive, and writes all artifacts/figures to Drive.

```bash
python experiments/methodology/colab_method_validation_core.py --mode core
```

If the methodology branch lives on a fork or feature branch, point Colab at that
repository/branch explicitly:

```bash
python experiments/methodology/colab_method_validation_core.py --mode core \
  --repo-url https://github.com/YOUR_USER/Graph-Specialisation-and-Metrics.git \
  --branch YOUR_BRANCH \
  --github-username YOUR_USER
```

For a quick smoke test:

```bash
python experiments/methodology/colab_method_validation_core.py --mode fast-dev
```

## HPC launch pattern

Run from the already-activated `graphbench-algoreas` environment:

```bash
cd /rds/user/jgg45/hpc-work/Graph-Specialisation-and-Metrics
git pull --ff-only
export PYTHON_BIN="$(which python)"

sbatch --export=ALL,PYTHON_BIN,PROJECT_ROOT=$PWD \
  -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
  -N 1 --ntasks=1 --gres=gpu:1 \
  experiments/methodology/slurm/method_validation.sbatch

sbatch --export=ALL,PYTHON_BIN,PROJECT_ROOT=$PWD \
  -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
  -N 1 --ntasks=1 --gres=gpu:1 \
  experiments/methodology/slurm/main_procedure_array.sbatch
```

Figure-only rerenders use cached metrics:

```bash
sbatch --export=ALL,PYTHON_BIN,PROJECT_ROOT=$PWD,ARTIFACT_ROOT=/path/to/artifacts/hash \
  -A mlmi-jgg45-sl2-gpu -p ampere --qos=gpu1 \
  -N 1 --ntasks=1 \
  experiments/methodology/slurm/render_figures.sbatch
```
