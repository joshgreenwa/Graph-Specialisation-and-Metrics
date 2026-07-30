import csv
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from graph_specialisation_metrics.synthetic.nar_reach_analysis import (
    ReachConfig,
    _clean_reach_jacobians,
    _encoded_transport_jvp,
    architectural_reach,
    nar_known_references,
    run,
    summarise_bamberger,
    summarise_profiles,
)


def test_nar_reach_references_separate_known_constraints_from_learned_route():
    references = nar_known_references()

    assert references["learned_carrier_ground_truth"] is None
    assert references["semantic_sources"]["query"]["distance_to_readout"] == 2
    assert references["semantic_sources"]["requested_record"]["distance_to_readout"] == 1
    assert architectural_reach("1hop", 1) == 1
    assert architectural_reach("1hop", 2) == 2
    assert architectural_reach("2hop", 2) == 3
    assert architectural_reach("dense", 1) == 3


def test_profile_summary_normalises_each_event_before_aggregation():
    rows = []
    for seed in (0, 1):
        for distance, local, finite in zip(
            range(4),
            (0.0, 2.0, 2.0, 0.0),
            (0.0, 1.0, 3.0, 0.0),
        ):
            rows.append(
                {
                    "model": "1hop",
                    "N": 8,
                    "seed": seed,
                    "graph": 5,
                    "channel": "semantic",
                    "source": 2,
                    "source_role": "query",
                    "donor_graph": 11,
                    "donor_node": 2,
                    "draw": 0,
                    "layer": 2,
                    "carrier": distance,
                    "distance": distance,
                    "local_jacobian": local,
                    "functional_carriage": finite,
                }
            )

    profiles, expected = summarise_profiles(rows, effect_floor=1.0e-12, layer=2)
    functional = {
        row["distance"]: row["mean"]
        for row in profiles
        if row["method"] == "functional_carriage"
    }
    local_range = next(
        row["mean"] for row in expected if row["method"] == "local_jacobian"
    )
    functional_range = next(
        row["mean"] for row in expected if row["method"] == "functional_carriage"
    )

    assert functional == pytest.approx({0: 0.0, 1: 0.25, 2: 0.75, 3: 0.0})
    assert local_range == pytest.approx(1.5)
    assert functional_range == pytest.approx(1.75)


class _ToyAttention(torch.nn.Module):
    def forward(self, data):
        routed = (
            data.x.square() + 0.0 * data.edge_attr.sum()
        ).reshape(-1, 1, 1)
        return routed, data.edge_attr


class _ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _ToyAttention()

    def forward(self, data):
        routed, _ = self.attention(data)
        output = copy(data)
        output.x = data.x + routed.reshape_as(data.x)
        return output


class _ToyModel(torch.nn.Module):
    L = 1
    H = 1
    dh = 1
    width = 1

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.layers = torch.nn.ModuleList([_ToyLayer()])

    @property
    def attention_layers(self):
        return [layer.attention for layer in self.layers]

    def _pyg_batch(self, batch):
        nodes = int(batch.x.shape[1])
        replicas = int(batch.x.shape[0])
        data = SimpleNamespace()
        data.x = (
            batch.x[..., :1]
            .reshape(replicas * nodes, 1)
            .float()
            .requires_grad_()
        )
        data.edge_attr = torch.zeros(
            replicas,
            1,
            device=data.x.device,
            requires_grad=True,
        )
        data.edge_index = torch.tensor([[0], [0]], device=data.x.device)
        return data

    def forward(self, batch):
        data = self._pyg_batch(batch)
        for layer in self.layers:
            data = layer(data)
        states = data.x.reshape(int(batch.x.shape[0]), int(batch.x.shape[1]), 1)
        central = states[:, 0]
        return torch.cat((central, 2.0 * central), dim=-1)


def _toy_graph(first_value):
    return SimpleNamespace(
        x=torch.tensor([[first_value, 0], [1, 0]], dtype=torch.long),
        adj=torch.eye(2),
        rrwp=torch.zeros(2, 2, 1),
        central_idx=torch.tensor(0),
        intermediate_idx=torch.tensor(0),
        query_idx=torch.tensor(0),
        target_idx=torch.tensor(1),
        record_mask=torch.tensor([False, True]),
        y=torch.tensor([0]),
        n_records=torch.tensor(2),
        num_nodes=2,
    )


def test_local_comparator_is_clean_directional_jvp_not_finite_difference():
    tangent = _encoded_transport_jvp(
        _ToyModel(),
        [_toy_graph(2)],
        [_toy_graph(1)],
    )

    # d(x^2)/dx at clean x=2, along clean-event direction +1, is 4.
    assert tangent.method == "exact_autograd_jvp"
    assert tangent.values[0][0, 0, 0, 0].item() == pytest.approx(4.0)
    # The second unchanged node has zero directional response.
    assert tangent.values[0][0, 1, 0, 0].item() == pytest.approx(0.0)


def test_clean_reach_capture_reuses_one_graph_for_carrier_and_input_jacobians():
    model = _ToyModel()

    captured = _clean_reach_jacobians(model, _toy_graph(2), torch.device("cpu"))

    assert captured.transport.shape == (2, 1, 2, 1, 1)
    assert captured.input_node.shape == (2, 2, 1)
    assert captured.capture.transport[0].shape == (2, 1, 1)
    assert torch.isfinite(captured.transport).all()


def test_bamberger_summary_is_expected_input_to_output_distance():
    rows = [
        {
            "model": "dense",
            "N": 8,
            "seed": seed,
            "graph": 0,
            "distance_to_central_output": distance,
            "influence": influence,
        }
        for seed in (0, 1)
        for distance, influence in ((0, 0.0), (1, 1.0), (2, 3.0))
    ]

    profiles, ranges = summarise_bamberger(rows, effect_floor=1.0e-12)

    assert {
        row["distance"]: row["mean"] for row in profiles
    } == pytest.approx({0: 0.0, 1: 0.25, 2: 0.75})
    assert ranges[0]["mean"] == pytest.approx(1.75)


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_figure_only_reach_run_uses_cached_csv_without_grit(tmp_path):
    reach_rows = []
    beneficial_rows = []
    bamberger_rows = []
    for channel in ("semantic", "structural"):
        for layer in (1, 2):
            for distance in range(4):
                reach_rows.append(
                    {
                        "model": "1hop",
                        "N": 8,
                        "seed": 0,
                        "graph": 0,
                        "channel": channel,
                        "source": 2,
                        "source_role": "query",
                        "donor_graph": 1,
                        "donor_node": 2,
                        "draw": 0,
                        "layer": layer,
                        "carrier": distance,
                        "distance": distance,
                        "local_jacobian": (1.0, 2.0, 1.0, 0.0)[distance],
                        "functional_carriage": (1.0, 1.0, 2.0, 0.0)[distance],
                    }
                )
        for role in ("query", "requested_record"):
            beneficial_rows.append(
                {
                    "model": "1hop",
                    "N": 8,
                    "seed": 0,
                    "graph": 0,
                    "channel": channel,
                    "source_role": role,
                    "beneficial_carriage": 0.2,
                }
            )
    for distance, influence in ((0, 0.1), (1, 0.3), (2, 0.6)):
        bamberger_rows.append(
            {
                "model": "1hop",
                "N": 8,
                "seed": 0,
                "graph": 0,
                "distance_to_central_output": distance,
                "influence": influence,
            }
        )
    _write_rows(tmp_path / "results" / "reach_carriers.csv", reach_rows)
    _write_rows(
        tmp_path / "results" / "beneficial_carriage.csv",
        beneficial_rows,
    )
    _write_rows(
        tmp_path / "results" / "bamberger_semantic_input.csv",
        bamberger_rows,
    )
    config = ReachConfig(
        models=("1hop",),
        ns=(8,),
        seeds=(0,),
        graphs=1,
        donors_per_source=1,
        semantic_donor_graphs=2,
    )

    result = run(config, output_dir=tmp_path, phase="figures")

    assert set(result["figures"]) == {
        "semantic_reach_layer1",
        "semantic_arrival_layer2",
        "structural_reach_layer1",
        "structural_arrival_layer2",
        "expected_reach_layer1",
        "bamberger_semantic_range",
        "beneficial",
    }
    for paths in result["figures"].values():
        assert set(paths) == {"png", "pdf"}
        assert all(Path(path).is_file() for path in paths.values())
