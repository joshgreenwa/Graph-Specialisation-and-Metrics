# Graph Specialisation and Metrics

This repository explores how graph transformer models process symbolic and structural information, with an emphasis on architecture-specific head specialisation.

The long-term project scope covers:

- training and evaluation files for Graphormer, GraphGPS, GRIT, CSA, and Exphormer;
- datasets and preprocessing artifacts for symbolic and structural graph tasks;
- checkpoints from trained model runs;
- metrics for quantifying specialisation across attention heads and model layers;
- visualisations for comparing specialisation patterns across architectures.

## Current Contents

The first component is a ZINC test case with four training notebooks and extracted Python scripts:

- `experiments/zinc/notebooks/grit_ZINC_core.ipynb`
- `experiments/zinc/notebooks/graphormer_ZINC_core.ipynb`
- `experiments/zinc/notebooks/CSA_ZINC_core.ipynb`
- `experiments/zinc/notebooks/graphgps_ZINC_core.ipynb`
- `experiments/zinc/training/grit_zinc_core.py`
- `experiments/zinc/training/graphormer_zinc_core.py`
- `experiments/zinc/training/csa_zinc_core.py`
- `experiments/zinc/training/graphgps_zinc_core.py`

The repository also includes controlled synthetic graph tasks:

- `experiments/synthetic/training/marked_tree_path_graphgps.py`
- `experiments/synthetic/training/structural_symbolic_graphgps.py`

## Repository Layout

```text
.
├── checkpoints/                 # Model checkpoints, grouped by task/model
├── data/                        # Dataset notes and optional small metadata files
├── docs/                        # Project notes and design docs
├── experiments/
│   ├── synthetic/
│   │   └── training/            # Controlled synthetic task runners
│   └── zinc/
│       ├── notebooks/           # Original ZINC notebooks
│       └── training/            # Extracted scripts from the notebooks
├── metrics/                     # Specialisation metric implementations
├── src/
│   └── graph_specialisation_metrics/
├── tests/
└── visualisations/              # Plotting and analysis utilities
```

Large datasets and checkpoints should stay out of git unless they are intentionally small reproducibility fixtures. Use the directory README files to document where artifacts came from and how to regenerate or download them.

## Planned Components

- Add Exphormer training artifacts for ZINC.
- Standardise run metadata across model families.
- Implement attention-head and layer-level specialisation metrics.
- Add structural and symbolic probing tasks beyond ZINC.
- Add visualisations for comparing model architectures and training stages.
