from pathlib import Path

import pytest
import torch

from graph_specialisation_metrics.synthetic.molecular_nonlinear_reach import (
    shortest_path_matrix,
)
from graph_specialisation_metrics.synthetic.molecular_redundancy_reach import (
    AnchoredMolecule,
    AnchoredRadialRegressor,
    Config,
    evaluate_performance,
    measure_reach,
)


def _path_edge_index(nodes: int) -> torch.Tensor:
    left = torch.arange(nodes - 1, dtype=torch.long)
    right = left + 1
    return torch.stack(
        (torch.cat((left, right)), torch.cat((right, left))),
        dim=0,
    )


def _sample() -> AnchoredMolecule:
    return AnchoredMolecule(
        graph_id=0,
        spd=shortest_path_matrix(7, _path_edge_index(7)),
        anchor=0,
        copy_nodes={2: 2, 4: 4, 6: 6},
        target=1.0,
    )


def _model(weights: dict[int, float]) -> AnchoredRadialRegressor:
    model = AnchoredRadialRegressor(max_distance=6).to(dtype=torch.float64)
    with torch.no_grad():
        model.radial_weight.zero_()
        for distance, weight in weights.items():
            model.radial_weight[int(distance)] = float(weight)
        model.bias.zero_()
    model.eval()
    return model


def test_jacobian_and_finite_carriage_agree_for_linear_redundant_cues(
    tmp_path: Path,
):
    config = Config(
        output_dir=tmp_path,
        data_root=tmp_path,
        train_graphs=1,
        val_graphs=1,
        test_graphs=1,
        seeds=(0,),
        bootstrap_replicates=10,
    )
    rows = measure_reach(
        config,
        _model({0: 0.25, 2: 0.25, 4: 0.25, 6: 0.25}),
        [_sample()],
        task="redundant",
        variant="through_d6",
        active_distances=(0, 2, 4, 6),
        seed=0,
    )
    profiles = {
        method: [
            float(row["normalised_mass"])
            for row in rows
            if row["method"] == method
        ]
        for method in ("bamberger", "finite")
    }
    assert profiles["finite"] == pytest.approx(profiles["bamberger"])
    assert sum(
        float(row["expected_distance"])
        for row in rows
        if row["method"] == "finite" and int(row["distance"]) == 0
    ) == pytest.approx(3.0)


def test_redundant_reliance_is_not_necessity_but_essential_control_is(
    tmp_path: Path,
):
    config = Config(
        output_dir=tmp_path,
        data_root=tmp_path,
        train_graphs=1,
        val_graphs=1,
        test_graphs=1,
        seeds=(0,),
        bootstrap_replicates=10,
    )
    redundant = evaluate_performance(
        config,
        _model({0: 0.25, 2: 0.25, 4: 0.25, 6: 0.25}),
        _model({0: 1.0}),
        [_sample()],
        task="redundant",
        variant="through_d6",
        active_distances=(0, 2, 4, 6),
        seed=0,
    )[0]
    essential = evaluate_performance(
        config,
        _model({6: 1.0}),
        _model({}),
        [_sample()],
        task="essential",
        variant="essential_d6",
        active_distances=(0, 6),
        seed=0,
    )[0]

    assert redundant["full_mae"] == pytest.approx(0.0)
    assert redundant["frozen_far_removed_mae"] == pytest.approx(0.75)
    assert redundant["local_refit_mae"] == pytest.approx(0.0)
    assert essential["full_mae"] == pytest.approx(0.0)
    assert essential["frozen_far_removed_mae"] == pytest.approx(1.0)
    assert essential["local_refit_mae"] == pytest.approx(1.0)
