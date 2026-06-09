# Counterfactual Interchange Mediation GRIT Runs

This directory prepares the first CFIM teacher-student experiments:

- `ppr_diffusion`
- `nearest_anchor_voronoi`

The configs use a 2-layer official-backed GRIT-RRWP model with continuous node
inputs and a 16-dimensional node-regression head. The run path writes the
artifact layout requested in the experiment plan under `artifacts/cfim/`.

## Required HPC Environment

Set these before submitting jobs:

```bash
export PROJECT_ROOT=/path/to/Graph-Specialisation-and-Metrics
export GRIT_ROOT=/path/to/GRIT
export CFIM_ARTIFACT_ROOT=/shared/path/artifacts/cfim
export ENV_ACTIVATE=/path/to/activate_script  # optional
```

The official GRIT repository should be pinned to:

```text
6c988ea600a606fbb49a2246c64a2d37396b3ab5
```

The job uses `PYTHONPATH=${PROJECT_ROOT}/src` and launches:

```bash
python -m graph_specialisation_metrics.counterfactual_interchange_mediation run-sequence \
  --config experiments/synthetic/cfim/configs/grit_ppr_diffusion.yaml \
  --task ppr_diffusion \
  --device cuda \
  --backend official
```

and the corresponding Voronoi config.

## Submit

From the repository root:

```bash
sbatch experiments/synthetic/cfim/slurm/cfim_a100_sequence.sbatch
```

The array is `0-1%1`, so only one task uses the A100 at a time. To run only
post-training experiments from cached `best.pt` checkpoints:

```bash
sbatch --export=ALL,CFIM_SKIP_TRAINING=1 experiments/synthetic/cfim/slurm/cfim_a100_sequence.sbatch
```

## Local Smoke Test

This verifies data generation, local fallback training, intervention caching,
and counterfactual metric plumbing without the official GRIT install:

```bash
PYTHONPATH=src python -m graph_specialisation_metrics.counterfactual_interchange_mediation run-sequence \
  --config experiments/synthetic/cfim/configs/grit_ppr_diffusion.yaml \
  --task ppr_diffusion \
  --backend local \
  --device cpu \
  --fast-dev-run
```

The local backend is not paper-faithful and should not be reported as GRIT.
