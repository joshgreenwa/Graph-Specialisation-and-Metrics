import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from graph_specialisation_metrics.experiments import ExperimentSetupError
from graph_specialisation_metrics.experiments._grit import grit_transformer_layer
from graph_specialisation_metrics.experiments.graphbench import (
    _data_root,
    _EdgeDonorPool,
    _load_splits,
    _model,
    _prepared_score_splits,
    _semantic_donor_swap,
    _settings,
    _structural_candidates,
    _structural_donors,
)


def _graph(edges, values=None, *, node_types=None, rrwp=None):
    directed = [
        (left, right) for left, right in edges for left, right in ((left, right), (right, left))
    ]
    edge_index = torch.tensor(directed, dtype=torch.long).T
    nodes = 1 + int(edge_index.max())
    degree = torch.bincount(edge_index[0], minlength=nodes).float()
    values = values or list(range(1, len(edges) + 1))
    edge_values = torch.tensor([value for value in values for _ in range(2)], dtype=torch.float32)
    return SimpleNamespace(
        num_nodes=nodes,
        orig_edge_index=edge_index,
        orig_edge_value=edge_values,
        edge_attr=edge_values[:, None].clone(),
        deg=degree,
        log_deg=torch.log1p(degree),
        x=torch.tensor(node_types or [0] * nodes),
        rrwp_dense=torch.as_tensor(rrwp, dtype=torch.float32) if rrwp is not None else None,
        clone=lambda: None,
    )


def test_scoring_prepares_only_checkpoint_donors_and_evaluation_graphs(
    monkeypatch, tmp_path
) -> None:
    from graph_specialisation_metrics.experiments import graphbench

    class TrackedSplit:
        def __init__(self, name, size):
            self.name, self.size, self.read = name, size, []

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            self.read.append(index)
            return f"{self.name}-{index}"

    splits = {
        "train": TrackedSplit("train", 300),
        "val": TrackedSplit("val", 120),
        "test": TrackedSplit("test", 120),
    }
    roots, prepared = [], []

    def load_dataset(root):
        roots.append(root)
        return splits

    def prepare_graph(graph, *, mean, std, steps):
        prepared.append((graph, mean, std, steps))
        return graph

    monkeypatch.setattr(graphbench, "_load_dataset", load_dataset)
    monkeypatch.setattr(graphbench, "_prepare_graph", prepare_graph)
    config = {
        "data": {
            "root": str(tmp_path / "graphbench"),
            "train_graphs": 200,
            "validation_graphs": 80,
        }
    }
    model_settings = _settings(config, fast=False)
    score_settings = _settings(config, fast=True)
    selected = _prepared_score_splits(tmp_path, config, model_settings, score_settings, (7.5, 2.25))

    train_population = sorted(random.Random(101).sample(range(300), 200))
    semantic_pool_positions = np.random.default_rng(31_416).choice(200, size=16, replace=False)
    expected_train = [train_population[index] for index in sorted(semantic_pool_positions)]
    validation_population = sorted(random.Random(211).sample(range(120), 80))
    evaluation_positions = np.random.default_rng(31_415).permutation(80)[:2]
    expected_val = [validation_population[index] for index in sorted(evaluation_positions)]
    assert splits["train"].read == expected_train
    assert splits["val"].read == expected_val
    assert splits["test"].read == []
    assert selected == {
        "train": [f"train-{index}" for index in expected_train],
        "val": [f"val-{index}" for index in expected_val],
    }
    assert roots == [tmp_path / "graphbench"]
    assert len(prepared) == 18
    assert all((mean, std, steps) == (7.5, 2.25, 16) for _, mean, std, steps in prepared)


def test_graphbench_jobs_share_the_parent_data_cache_by_default(tmp_path) -> None:
    output_dir = tmp_path / "graphbench" / "job_2"
    assert _data_root(output_dir, {}) == tmp_path / "graphbench" / "data"
    assert _data_root(output_dir, {"data": {"root": "~/datasets/graphbench"}}) == (
        tmp_path.home() / "datasets" / "graphbench"
    )


def test_training_does_not_read_the_unused_test_split(monkeypatch) -> None:
    from graph_specialisation_metrics.experiments import graphbench

    class TestSplit:
        def __len__(self):
            return 1

        def __getitem__(self, _index):  # pragma: no cover - assertion helper
            raise AssertionError("training must not read GraphBench test graphs")

    monkeypatch.setattr(
        graphbench,
        "_load_dataset",
        lambda _root: {"train": ["train"], "val": ["val"], "test": TestSplit()},
    )
    settings = _settings(
        {"data": {"train_graphs": 1, "validation_graphs": 1}},
        fast=False,
    )

    assert _load_splits(Path("unused"), settings) == {"train": ["train"], "val": ["val"]}


def test_semantic_edge_donors_are_degree_matched_and_graph_balanced() -> None:
    source = _graph([(0, 1), (1, 2)], [1.0, 5.0])
    donors = [
        _graph([(0, 1), (1, 2)], [2.0, 3.0]),
        _graph([(0, 1), (1, 2)], [4.0, 6.0]),
        _graph([(0, 1), (1, 2), (2, 0)], [7.0, 8.0, 9.0]),
    ]
    pool = _EdgeDonorPool(donors)
    eligible = pool.eligible(source, 0)
    assert set(eligible) == {0, 1}
    assert all(donor.degree == (1, 2) for rows in eligible.values() for donor in rows)

    class LastGraph:
        @staticmethod
        def integers(high):
            return high - 1

    assert pool.sample(source, 0, 1, LastGraph())[0].graph == 1


def test_semantic_donor_swap_changes_the_reciprocal_edge_unit_only() -> None:
    graph = _graph([(0, 1), (1, 2)], [1.0, 5.0])
    graph.clone = lambda: SimpleNamespace(
        **{
            **graph.__dict__,
            "orig_edge_value": graph.orig_edge_value.clone(),
            "edge_attr": graph.edge_attr.clone(),
        }
    )
    donor_swap = _semantic_donor_swap(graph, 0, 9.0)
    assert donor_swap.orig_edge_value.tolist() == [9.0, 9.0, 5.0, 5.0]
    assert donor_swap.edge_attr[:, 0].tolist() == donor_swap.orig_edge_value.tolist()


def test_structural_donors_are_uniform_unique_and_exhaust_eligible_nodes() -> None:
    rrwp = np.zeros((8, 8, 1), dtype=np.float32)
    for node in range(8):
        rrwp[node, :, 0] = node
        rrwp[:, node, 0] += node / 10
    graph = _graph(
        [(node, node + 1) for node in range(7)],
        node_types=[0] * 8,
        rrwp=rrwp,
    )
    assert _structural_candidates(graph, 0) == [1, 2, 3, 4, 5, 6, 7]

    class RecordingGenerator:
        def choice(self, candidates, *, size, replace):
            assert list(candidates) == [1, 2, 3, 4, 5, 6, 7]
            assert size == 3 and replace is False
            return np.asarray([7, 2, 5])

    assert _structural_donors(graph, 0, 3, RecordingGenerator()) == (7, 2, 5)
    assert _structural_donors(graph, 0, 99, np.random.default_rng(7)) == tuple(range(1, 8))

    # Semantic node type is not part of structural-donor eligibility.
    graph.x[4] = 1
    assert _structural_candidates(graph, 0) == [1, 2, 3, 4, 5, 6, 7]


def test_graphbench_capture_preserves_the_layer_forward() -> None:
    from pathlib import Path

    import yaml
    from torch_geometric.data import Data

    try:
        grit_transformer_layer()
    except ExperimentSetupError as exc:
        pytest.skip(str(exc))

    root = Path(__file__).resolve().parents[1]
    settings = _settings(
        yaml.safe_load((root / "configs" / "graphbench.yaml").read_text()), fast=True
    )
    model = _model(settings).eval()
    source = _graph([(0, 1), (1, 2)], [1.0, 5.0])
    graph = Data(
        x=source.x,
        edge_index=source.orig_edge_index,
        orig_edge_index=source.orig_edge_index,
        orig_edge_value=source.orig_edge_value,
        edge_attr=source.edge_attr,
        deg=source.deg,
        log_deg=source.log_deg,
        num_nodes=source.num_nodes,
    )
    graph.rrwp_dense = torch.eye(3).unsqueeze(-1).repeat(1, 1, settings.rrwp_steps)
    graph.rrwp = torch.diagonal(graph.rrwp_dense, dim1=0, dim2=1).T
    nodes = torch.arange(3)
    graph.rrwp_index = torch.stack((nodes.repeat_interleave(3), nodes.repeat(3)))
    graph.rrwp_val = graph.rrwp_dense.reshape(-1, settings.rrwp_steps)
    graph.y = torch.zeros(graph.orig_edge_index.shape[1])

    from torch_geometric.data import Batch

    ordinary = model(Batch.from_data_list([graph.clone()]))
    model_output, head_outputs = model(Batch.from_data_list([graph.clone()]), capture=True)

    torch.testing.assert_close(model_output, ordinary)
    assert len(head_outputs) == settings.layers
    gradient = torch.autograd.grad(model_output[0], head_outputs)[0]
    assert torch.isfinite(gradient).all() and float(gradient.abs().sum()) > 0
