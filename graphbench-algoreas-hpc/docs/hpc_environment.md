# HPC Environment Preflight

The highest-risk failure mode is not the SLURM wrapper; it is official backend
imports failing after the job starts. Run this preflight before submitting
training arrays.

## Recommended Environment

Use Python 3.10. Avoid Python 3.13 for paper runs because the PyG compiled wheel
stack and older Graphormer/Fairseq code are much more likely to fail there.

A practical unified starting point is the GNN+ stack:

```bash
conda create -n graphbench-algoreas python=3.10 -y
conda activate graphbench-algoreas

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" "torch_geometric>=2.5,<2.7"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

pip install yacs torchmetrics pytorch-lightning performer-pytorch tensorboardX ogb wandb
pip install opt_einsum networkx scikit-learn matplotlib pyyaml
pip install graphbench-lib
```

Then install the official repos editable:

```bash
mkdir -p external
git clone --recurse-submodules https://github.com/microsoft/Graphormer.git external/Graphormer
git clone https://github.com/rampasek/GraphGPS.git external/GraphGPS
git clone https://github.com/LiamMa/GRIT.git external/GRIT
git clone https://github.com/LUOyk1999/tunedGNN-G.git external/GNNPlus

git -C external/Graphormer checkout ac154fe4253d076a1c294f14be20dad0351cff3c
git -C external/Graphormer submodule update --init --recursive
git -C external/GraphGPS checkout 28015707cbab7f8ad72bed0ee872d068ea59c94b
git -C external/GRIT checkout 6c988ea600a606fbb49a2246c64a2d37396b3ab5
git -C external/GNNPlus checkout 0e02ad9acc2f1e54b5ad71c051bf5dfb1fcb4f28

pip install -e external/GraphGPS
pip install -e external/GRIT
pip install -e external/GNNPlus
```

Graphormer is the most fragile dependency because its encoder imports Fairseq
modules. First try installing its checked-out submodule:

```bash
pip install -e external/Graphormer/fairseq
```

If this fails under the unified Torch/PyG environment, do not silently replace
Graphormer. Either fix the Fairseq install, or make an explicit compatibility
shim and label it as such in the method section.

## Required Preflight

From `graphbench-algoreas-hpc`:

```bash
python bin/check_official_backends.py
```

All rows must be `OK` before paper training arrays are launched.

Expected local failure modes if the environment is not ready:

- `torch_geometric`, `torch_scatter`, `torch_sparse`, or `pyg_lib` missing:
  PyG stack is not installed.
- `fairseq` missing: Graphormer encoder will not import.
- `graphormer.fairseq_submodule` failed: Graphormer was cloned without
  `--recurse-submodules`.
- `graphbench` missing: GraphBench loader is not installed.
- Python 3.13: use a Python 3.10 environment for HPC runs.
