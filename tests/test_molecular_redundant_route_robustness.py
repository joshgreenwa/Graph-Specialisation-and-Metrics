import pytest
import torch

from graph_specialisation_metrics.synthetic.molecular_redundant_record_router import (
    MolecularSupport,
    RedundantRecordRouter,
    make_router_dataset,
)
from graph_specialisation_metrics.synthetic.molecular_redundant_route_robustness import (
    corrupt_local_copy,
    corrupt_selected_distant_record,
    route_failure_metrics,
)


def _dataset():
    supports = [
        MolecularSupport(
            graph_id=index,
            anchor=0,
            record_nodes=(1, 4, 5, 6),
            record_distances=(1, 4, 4, 6),
        )
        for index in range(8)
    ]
    return make_router_dataset(supports, examples_per_graph=8, seed=7)


def _oracle_router(local_mix: float) -> RedundantRecordRouter:
    model = RedundantRecordRouter(local_mix, seed=0)
    with torch.no_grad():
        model.semantic_logits.copy_(torch.tensor(((8.0, -8.0), (-8.0, 8.0))))
        model.structural_logits.copy_(torch.tensor(((8.0, -8.0), (-8.0, 8.0))))
        model.distance_bias.zero_()
    return model


def test_corruptions_target_opposite_redundant_routes():
    dataset = _dataset()
    local_failed = corrupt_local_copy(dataset, probability=1.0, seed=0)
    distant_failed = corrupt_selected_distant_record(dataset)
    torch.testing.assert_close(local_failed.local_value, -dataset.target)
    selected = 2 * dataset.structure + dataset.query
    rows = torch.arange(len(dataset))
    torch.testing.assert_close(
        distant_failed.record_values[rows, selected], -dataset.target
    )
    torch.testing.assert_close(distant_failed.local_value, dataset.target)


def test_more_redundant_global_use_rescues_local_failure_but_adds_far_vulnerability():
    dataset = _dataset()
    global_heavy = route_failure_metrics(_oracle_router(0.25), dataset)
    local_heavy = route_failure_metrics(_oracle_router(0.75), dataset)

    assert global_heavy["clean_mae"] < 1.0e-5
    assert local_heavy["clean_mae"] < 1.0e-5
    assert global_heavy["local_failure_mae"] < local_heavy["local_failure_mae"]
    assert global_heavy["local_failure_rescue"] > local_heavy["local_failure_rescue"]
    assert global_heavy["distant_failure_damage"] > local_heavy["distant_failure_damage"]
    assert global_heavy["local_only_failure_mae"] == pytest.approx(
        local_heavy["local_only_failure_mae"]
    )
