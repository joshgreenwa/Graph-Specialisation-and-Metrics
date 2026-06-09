from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from graph_specialisation_metrics.task_causal_specialisation_protocol import (
    average_precision,
    flow_sensitivity_rows,
    matching_sensitivity_rows,
)


def base_args(**kwargs):
    values = {
        "selection_fraction": 0.5,
        "solver_edge_value_source": "loaded",
        "solver_edge_value_mean": None,
        "solver_edge_value_std": None,
        "solver_capacity_clamp_min": 0.0,
        "flow_eta": 1.0,
        "critical_threshold": 0.0,
        "matching_nonedges_per_edge": 1,
        "random_seed": 0,
        "matching_added_edge_value": 1.0,
    }
    values.update(kwargs)
    return Namespace(**values)


def test_flow_sensitivity_marks_capacity_decrease_critical_on_chain():
    graph = SimpleNamespace(
        node_type=torch.tensor([1, 0, 2]),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_value=torch.tensor([1.0, 1.0]),
        target=torch.tensor([1.0]),
        task_type="graph_regression",
        num_nodes=3,
    )

    rows = flow_sensitivity_rows(graph, graph_position=0, graph_index=7, args=base_args())

    assert len(rows) == 2
    assert {row["critical"] for row in rows} == {1}
    assert {row["perturbation"] for row in rows} == {"decrease"}
    assert {row["sensitivity"] for row in rows} == {1.0}
    crossing = {row["sender"]: row["mincut_crossing_e"] for row in rows}
    assert crossing[0] == 1
    assert crossing[1] == 0


def test_matching_sensitivity_marks_delete_critical_existing_edge():
    graph = SimpleNamespace(
        node_type=torch.zeros(2, dtype=torch.long),
        edge_index=torch.tensor([[0], [1]]),
        edge_value=torch.tensor([1.0]),
        target=torch.tensor([1.0]),
        task_type="edge_binary",
        num_nodes=2,
    )

    rows = matching_sensitivity_rows(graph, graph_position=0, graph_index=11, args=base_args())

    edge_rows = [row for row in rows if row["unit_kind"] == "edge"]
    assert len(edge_rows) == 1
    assert edge_rows[0]["critical"] == 1
    assert edge_rows[0]["perturbation"] == "delete"
    assert edge_rows[0]["sensitivity"] == 1.0


def test_average_precision_handles_ranked_binary_labels():
    assert average_precision([1, 0, 1], [0.9, 0.8, 0.1]) == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
