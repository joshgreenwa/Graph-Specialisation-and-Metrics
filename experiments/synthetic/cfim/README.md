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

For GNN+ baselines, expose the official GNNPlus repository:

```bash
git clone https://github.com/LUOyk1999/GNNPlus "$PROJECT_ROOT/external/GNNPlus"
cd "$PROJECT_ROOT/external/GNNPlus"
git checkout 0e02ad9acc2f1e54b5ad71c051bf5dfb1fcb4f28
export GNNPLUS_ROOT="$PROJECT_ROOT/external/GNNPlus"
```

The `gcn_plus_*.yaml` configs use the official GNNPlus `GCNConvLayer` class
with the CFIM node-regression data and training loop. They keep the GRIT depth
fixed at two layers and set `hidden_dim=192` to approximate the parameter budget
of the two-layer GRIT-128 baseline while preserving RWSE/degree inputs,
BatchNorm, dropout, residual connections, and FFN blocks.

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

## Official GCN+ Baseline

Train parameter-matched official GCN+ on the two teacher-student tasks:

```bash
python -m graph_specialisation_metrics.counterfactual_interchange_mediation train \
  --config experiments/synthetic/cfim/configs/gcn_plus_ppr_diffusion.yaml \
  --task ppr_diffusion \
  --device cuda \
  --backend official_gnnplus

python -m graph_specialisation_metrics.counterfactual_interchange_mediation train \
  --config experiments/synthetic/cfim/configs/gcn_plus_nearest_anchor_voronoi.yaml \
  --task nearest_anchor_voronoi \
  --device cuda \
  --backend official_gnnplus
```

Checkpoints are written separately from GRIT:

```text
artifacts/cfim/checkpoints/gcn_plus/<task>/seed_1001/best.pt
```

After training, evaluate clean and counterfactual performance with:

```bash
python -m graph_specialisation_metrics.counterfactual_interchange_mediation evaluate-counterfactuals \
  --config experiments/synthetic/cfim/configs/gcn_plus_ppr_diffusion.yaml \
  --task ppr_diffusion \
  --device cuda \
  --backend official_gnnplus
```
