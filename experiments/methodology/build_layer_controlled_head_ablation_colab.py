"""Build the small multi-cell launcher for the layer-controlled correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SOURCE = Path(__file__).with_name("layer_controlled_head_ablation_colab.py")
DEFAULT_OUTPUT = Path(__file__).with_name("layer_controlled_head_ablation_colab.ipynb")


def build(source: Path, output: Path) -> Path:
    code = source.read_text(encoding="utf-8")
    marker_start = "# ============================ paste from here ============================\n"
    marker_end = "# ============================ paste to here ============================\n"
    if marker_start not in code or marker_end not in code:
        raise ValueError("frontend paste markers are missing")
    executable = code.split(marker_start, 1)[1].rsplit(marker_end, 1)[0]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": output.name, "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "CPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Layer-controlled joint-sensitivity correction\n",
                    "\n",
                    (
                        "This cache-only frontend regenerates dissertation Figures 4.5(a) and "
                        "5.6 with an all-layer, within-layer Spearman estimate. It searches "
                        "Drive for the exact dissertation caches, resamples held-out molecules "
                        "for the molecular confidence intervals, and preserves the original "
                        "panel geometry and visual grammar. GraphBench is intentionally "
                        "deferred to the HPC pass.\n"
                    ),
                    "\n",
                    (
                        "Use a CPU runtime and choose **Runtime -> Run all**. Start with "
                        '`MODE = "inventory"` if any cache may have moved. No model, '
                        "dataset, checkpoint, GRIT, Graphormer, GPU, or forward pass is used.\n"
                    ),
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"colab": {"base_uri": "https://localhost:8080/"}},
                "outputs": [],
                "source": executable.splitlines(keepends=True),
            },
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build(args.source, args.output))


if __name__ == "__main__":
    main()
