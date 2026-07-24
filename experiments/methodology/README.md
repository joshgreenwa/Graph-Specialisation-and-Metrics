# Methodology Experiments

This directory contains config and Slurm entry points for the dissertation core
procedure and the method-validation suite.

The implementation specification for the final dense-model scoring comparison is
[SCORING_METRIC_REFINEMENT_TASK.md](SCORING_METRIC_REFINEMENT_TASK.md).
Its standalone ZINC/QM9 frontend is
[colab_scoring_metric_refinement_dense.py](colab_scoring_metric_refinement_dense.py);
the reusable implementation and estimands are documented in
[`scoring_refinement/README.md`](../../src/graph_specialisation_metrics/scoring_refinement/README.md).

The final controlled comparison on the already-trained mixed synthetic task uses
[colab_scoring_metric_refinement_synthetic.py](colab_scoring_metric_refinement_synthetic.py).
Upload it to Colab and run:

```python
%run colab_scoring_metric_refinement_synthetic.py
```

It loads the existing `cycle_dual_v2` seeds from Drive, evaluates
`M1_DD`, `M1_DT`, `M1_TD`, `M1_TT`, `M4`, `M5`, and `M7`, and writes
new caches beneath:

```text
/content/drive/MyDrive/graph_specialisation_metrics/
  causal_specialisation_double_dissociation/
  cycle_dual_v2/scoring_refinement_m1_m4_m5_m7_v1
```

The original checkpoints and causal-analysis caches remain unchanged. The four
figure families consolidate score planes, head necessity/task role, causal rescue
role, and score-selected family necessity plus rescue across every method.

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
