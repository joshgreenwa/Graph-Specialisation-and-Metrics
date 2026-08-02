from pathlib import Path

import pytest
import torch

from graph_specialisation_metrics.synthetic.molecular_nonlinear_reach import (
    Config,
    MolecularSample,
    RadialGraphRegressor,
    exact_target,
    measure_model,
    saturated_channel,
    shortest_path_matrix,
    summarise,
)


def _path_edge_index(nodes: int) -> torch.Tensor:
    left = torch.arange(nodes - 1, dtype=torch.long)
    right = left + 1
    return torch.stack(
        (torch.cat((left, right)), torch.cat((right, left))),
        dim=0,
    )


def test_saturated_encoder_preserves_binary_payload_but_flattens_derivative():
    binary = torch.tensor([-1.0, 1.0], dtype=torch.float64, requires_grad=True)
    encoded = saturated_channel(binary, 8.0)
    assert torch.equal(encoded.detach(), binary.detach())
    encoded.sum().backward()
    assert float(binary.grad.abs().max()) < 1.0e-4


def test_finite_range_tracks_known_molecular_support_when_jacobian_flattens(
    tmp_path: Path,
):
    nodes = 7
    spd = shortest_path_matrix(nodes, _path_edge_index(nodes))
    semantic = torch.tensor(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    config = Config(
        output_dir=tmp_path,
        data_root=tmp_path,
        train_graphs=1,
        val_graphs=1,
        test_graphs=1,
        target_distance=3,
        max_distance=3,
        kappas=(0.0, 8.0),
        seeds=(0,),
        bootstrap_replicates=10,
    )
    sample = MolecularSample(
        graph_id=0,
        atom_types=torch.zeros(nodes, dtype=torch.long),
        spd=spd,
        semantic=semantic,
        target=exact_target(
            semantic,
            spd,
            target_distance=3,
            local_weight=0.5,
            far_weight=1.0,
            kappa=4.0,
        ),
    )
    model = RadialGraphRegressor(max_distance=3).to(dtype=torch.float64)
    with torch.no_grad():
        model.radial_weight.zero_()
        model.radial_weight[0, 0] = 0.5
        model.radial_weight[3, 1] = 1.0
        model.bias.zero_()
    profiles, performance = measure_model(config, model, [sample], seed=0)
    _, expected, performance_summary = summarise(config, profiles, performance)

    lookup = {
        (float(row["kappa"]), str(row["method"])): float(row["mean"])
        for row in expected
    }
    assert lookup[(0.0, "bamberger")] == pytest.approx(
        lookup[(0.0, "finite")], abs=1.0e-10
    )
    assert lookup[(8.0, "finite")] == pytest.approx(
        lookup[(8.0, "oracle")], abs=1.0e-10
    )
    assert lookup[(8.0, "bamberger")] < 1.0e-3
    assert max(float(row["mean"]) for row in performance_summary) < 1.0e-12
