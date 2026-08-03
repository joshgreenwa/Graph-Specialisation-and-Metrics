# Graph Specialisation and Metrics

This repository explores how graph transformer models process symbolic and structural information, with an emphasis on architecture-specific head specialisation.

## Canonical methodology

The final, task-general scientific specification is
[`src/graph_specialisation_metrics/README.md`](src/graph_specialisation_metrics/README.md).
It fixes donor-swaps for both semantic and structural channels, graph-balanced output-projected
head scores, `F_sens` Functional carriage, and positive-is-beneficial donor-wise path-integrated
Beneficial carriage.
The public implementation is
[`src/graph_specialisation_metrics/methodology/`](src/graph_specialisation_metrics/methodology/README.md),
exposed through `graph_specialisation_metrics.main()`. The Drive-backed production launcher is
[`experiments/methodology/canonical_methodology_colab.py`](experiments/methodology/canonical_methodology_colab.py).
Module-level READMEs document implementations and historical refinement experiments; where their
older alternatives differ, the canonical methodology takes precedence.

## NAR reach analysis

The standalone launcher
[`experiments/synthetic/analysis/nar_reach_analysis_colab.py`](experiments/synthetic/analysis/nar_reach_analysis_colab.py)
reports the original Bamberger semantic encoded-input/central-output Jacobian range, then compares
a matched local donor-direction Jacobian with finite Functional carriage on the same semantic or
structural donor event and native routed-message carrier sites. The original estimator has no
canonical structural donor-swap analogue, so it is not relabelled as one. The launcher reads the
existing `nar_grit_fixed_n_v3` checkpoints for 1-hop, 2-hop, and dense GRIT at `N=8,16,64`, caches
graph-level measurements under
`canonical_nar_analysis_d128/extensions/nar_reach_analysis_v1`, saves PNG/PDF figures, and displays
them in Colab. Set `PHASE = "figures"` for a checkpoint-free figure rerun.

Layer-1 distance curves are the primary learned-reach result and expected layer-1 carrier distance
is secondary. Layer 2 is retained as an arrival/readout check: because NAR classifies only the
central node, output-projected final-layer mass should collapse to the central carrier at the known
query/readout (`d=2`) or record/readout (`d=1`) distance. NAR otherwise supplies architecture
support ceilings but no unique ground-truth learned carrier distribution. Structural RRWP swaps are
diagnostic counterfactuals because fixed-N NAR has the same topology and RRWP in every training
example. Beneficial carriage is shown separately as signed task benefit at the central final-state
readout, not as a distributed reach oracle.

The long-term project scope covers:

- training and evaluation files for Graphormer, GraphGPS, GRIT, CSA, and Exphormer;
- datasets and preprocessing artifacts for symbolic and structural graph tasks;
- checkpoints from trained model runs;
- metrics for quantifying specialisation across attention heads and model layers;
- visualisations for comparing specialisation patterns across architectures.

## Current Contents

The first component is a ZINC test case with five training notebooks and extracted Python scripts:

- `experiments/zinc/notebooks/grit_ZINC_core.ipynb`
- `experiments/zinc/notebooks/grit_ZINC_dense_localrrwp.ipynb`
- `experiments/zinc/notebooks/graphormer_ZINC_core.ipynb`
- `experiments/zinc/notebooks/CSA_ZINC_core.ipynb`
- `experiments/zinc/notebooks/graphgps_ZINC_core.ipynb`
- `experiments/zinc/training/grit_zinc_core.py`
- `experiments/zinc/training/graphormer_zinc_core.py`
- `experiments/zinc/training/csa_zinc_core.py`
- `experiments/zinc/training/graphgps_zinc_core.py`

The dense GRIT runner accepts `--rrwp-horizon 1` to retain official dense
attention while limiting RRWP to identity and one-step random-walk information.
The dedicated `grit_ZINC_dense_localrrwp.ipynb` notebook launches that control
with isolated GRIT, dataset-cache, result, and checkpoint directories.

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

### Dense-ZINC head mediation check

`experiments/zinc/analysis/zinc_head_mediation_colab.py` is a lightweight,
figures-only comparison of canonical internal head specialisation
\(S_x(h)\) with symmetric finite head mediation \(M_x(h)\) for semantic and
structural donor swaps. It reads the completed dense-ZINC score and causal
caches, loads no model or dataset, and writes two paper-oriented figures plus
head-level CSV tables to Drive.

`experiments/methodology/causal_spatial_support_colab.py` extends that check to
the causal spatial support of specialist heads on dense ZINC and dense QM9. It
separates source-conditioned attention access, canonical internal response,
and held-out shell-specific output mediation; jointly patched frozen families
quantify overlap among realised head pathways. Graph/channel shards are
resumable, while a figures-only rerun loads no model or dataset.
The frontend defaults to a four-graph, ZINC-only pilot; setting `PILOT = False`
enables the denser ZINC--QM9 comparison after the direction is validated.
The complete estimand and interpretation are documented in
`experiments/methodology/causal_spatial_support.md`.
