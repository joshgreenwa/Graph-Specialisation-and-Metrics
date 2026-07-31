from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from graph_specialisation_metrics.methodology.figures import FigureTheme
from graph_specialisation_metrics.methodology.graphbench_attention_figures import (
    plot_attention_head_grid,
    select_attention_heads,
)


def _scores():
    joint = np.asarray(
        [
            [0.8, 0.7, 0.4, 0.6],
            [0.9, 0.75, 1.2, 0.5],
        ]
    )
    selectivity = np.asarray(
        [
            [0.8, -0.7, 0.3, -0.2],
            [0.6, 0.02, -0.03, -0.4],
        ]
    )
    return {
        "coordinates": SimpleNamespace(
            joint_sensitivity=joint,
            selectivity=selectivity,
            active=np.ones_like(joint, dtype=bool),
        ),
        "specialist_classification": {
            "preference_threshold": 0.10,
            "heads": {"generalist": ((1, 1), (1, 2))},
            "strength_ranking": {
                "semantic_candidates": (
                    {"head": (0, 0)},
                    {"head": (1, 0)},
                    {"head": (0, 2)},
                ),
                "structural_candidates": (
                    {"head": (0, 1)},
                    {"head": (1, 3)},
                ),
            },
        },
    }


def test_attention_head_selection_uses_directional_ranking_and_high_j_generalist():
    selected = select_attention_heads(_scores())

    assert [(row["layer"], row["head"]) for row in selected] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 2),
    ]
    assert [row["role"] for row in selected] == [
        "semantic_1",
        "semantic_2",
        "structural",
        "generalist",
    ]


def test_attention_grid_contains_complete_matrices_and_graph_overlays(tmp_path):
    selected = select_attention_heads(_scores())
    rng = np.random.default_rng(17)
    attention = rng.uniform(0.01, 1.0, size=(4, 4, 4))
    attention /= attention.sum(axis=-1, keepdims=True)
    payload = {
        "attention": attention,
        "edge_index": np.asarray(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
            dtype=np.int64,
        ),
        "edge_value": np.asarray([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.float64),
        "edge_target": np.zeros(8, dtype=np.float64),
        "node_type": np.zeros(4, dtype=np.int64),
        "selected_heads": selected,
        "num_nodes": 4,
        "seed": 0,
        "graph_id": 0,
    }
    theme = FigureTheme(width=9.2, height=12.4, dpi=72)

    figure, axes = plot_attention_head_grid(
        payload,
        theme,
        top_k_per_receiver=2,
    )
    output = tmp_path / "attention.pdf"
    figure.savefig(output)

    assert axes.shape == (4, 2)
    assert output.is_file() and output.stat().st_size > 0
    assert axes[0, 0].get_title().startswith("Complete attention matrix")
    assert "top 2 senders" in axes[0, 1].get_title()
    plt.close(figure)
