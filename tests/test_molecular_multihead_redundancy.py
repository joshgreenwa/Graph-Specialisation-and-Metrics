from pathlib import Path

import torch

from graph_specialisation_metrics.synthetic.molecular_multihead_redundancy import (
    Config,
    TinyMultiHeadGraphTransformer,
    measure_internal,
)
from graph_specialisation_metrics.synthetic.molecular_redundant_record_router import (
    MolecularSupport,
    make_router_dataset,
)


def _dataset():
    supports = [
        MolecularSupport(
            graph_id=index,
            anchor=0,
            record_nodes=(1, 4, 5, 6),
            record_distances=(1, 4, 4, 6),
        )
        for index in range(4)
    ]
    return make_router_dataset(supports, examples_per_graph=4, seed=1)


def test_tiny_transformer_has_registered_multihead_and_carrier_geometry(tmp_path: Path):
    config = Config(
        output_dir=tmp_path,
        data_root=tmp_path,
        train_graphs=1,
        val_graphs=1,
        test_graphs=1,
        examples_per_graph=4,
        seeds=(0,),
        train_local_corruption=(0.0,),
        hidden_dim=16,
        layers=2,
        heads=4,
        max_steps=1,
    )
    dataset = _dataset()
    model = TinyMultiHeadGraphTransformer(config, seed=0)
    output, states, attention = model(dataset, return_details=True)

    assert tuple(output.shape) == (len(dataset),)
    assert tuple(states.shape) == (len(dataset), 5, 16)
    assert model.readout == "sum"
    assert len(attention) == 2
    assert tuple(attention[0].shape) == (len(dataset), 4, 5, 5)
    measured = measure_internal(model, dataset)
    assert len(measured["heads"]) == 8
    assert measured["interaction_effect_mass"] >= 0.0
    assert 0.0 <= measured["head_score_alignment"] <= 1.0
    assert 0.0 <= measured["head_relative_imbalance"] <= 1.0
    assert torch.isfinite(output).all()
