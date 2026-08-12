# Graph Specialisation and Metrics

Code for Josh Green's dissertation on semantic and structural head specialisation in graph transformers.

## Install

Python 3.10–3.12 is required. Replace `mixed` with `graphbench`, `graphormer`, or `grit` as needed:

```bash
git clone --branch main --single-branch https://github.com/joshgreenwa/Graph-Specialisation-and-Metrics.git
cd Graph-Specialisation-and-Metrics
python -m pip install ".[mixed]"
```

## Experiments

| Config | Experiment |
| --- | --- |
| `configs/mixed.yaml` | Mixed semantic–structural synthetic task |
| `configs/graphbench.yaml` | GraphBench bipartite matching |
| `configs/graphormer_pcqm4mv2.yaml` | Public Graphormer PCQM4Mv2 checkpoint |
| `configs/grit_zinc.yaml` | GRIT on ZINC |
| `configs/grit_qm9.yaml` | GRIT on QM9 |

```bash
gsm run   --config configs/mixed.yaml --output-dir results/mixed
gsm run   --config configs/graphbench.yaml --job-index 0 --output-dir results/graphbench/job_0
gsm score --config configs/graphormer_pcqm4mv2.yaml --output-dir results/graphormer
gsm run   --config configs/grit_zinc.yaml --job-index 0 --output-dir results/grit_zinc/job_0
gsm run   --config configs/grit_qm9.yaml --job-index 0 --output-dir results/grit_qm9/job_0
```

Use `--fast` for a quick run. GraphBench uses job indices 0–3; each GRIT config uses 0–14. Slurm scripts are in [`slurm/`](slurm/):

```bash
sbatch slurm/mixed.sbatch
sbatch slurm/graphbench.sbatch
sbatch slurm/graphormer_score.sbatch
sbatch slurm/grit.sbatch
```

## Notebooks and output

Run the mixed [notebook](notebooks/mixed_demo.ipynb) / [Colab](https://colab.research.google.com/github/joshgreenwa/Graph-Specialisation-and-Metrics/blob/main/notebooks/mixed_demo.ipynb) or Graphormer [notebook](notebooks/graphormer_pcqm4mv2_demo.ipynb) / [Colab](https://colab.research.google.com/github/joshgreenwa/Graph-Specialisation-and-Metrics/blob/main/notebooks/graphormer_pcqm4mv2_demo.ipynb).

Each run writes `scores.npz` with `semantic_scores` ($S_{\mathrm{sem}}$), `structural_scores` ($S_{\mathrm{str}}$), and their distance-resolved score contributions. The notebooks also plot $J$ and $D_{\mathrm{rel}}$.
