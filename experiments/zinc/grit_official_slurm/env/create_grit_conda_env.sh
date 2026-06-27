#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${CONDA_ENV_NAME:-grit-zinc-py39-cu113}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to create the official GRIT environment." >&2
  exit 1
fi

conda create -y -n "${ENV_NAME}" python=3.9
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install \
  torch==1.12.1+cu113 \
  torchvision==0.13.1+cu113 \
  torchaudio==0.12.1 \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --trusted-host download.pytorch.org

python -m pip install \
  torch-scatter \
  torch-sparse \
  torch-cluster \
  torch-spline-conv \
  torch-geometric==2.2.0 \
  -f https://data.pyg.org/whl/torch-1.12.1+cu113.html \
  --trusted-host data.pyg.org

python -m pip install \
  rdkit \
  torchmetrics==0.9.1 \
  ogb \
  tensorboardX \
  yacs \
  opt_einsum \
  graphgym \
  "pytorch-lightning<2" \
  setuptools==59.5.0 \
  scikit-learn

echo "Created conda env: ${ENV_NAME}"
