from pathlib import Path

import torch

from graph_specialisation_metrics.synthetic.molecular_correlated_clues import (
    Config,
    drop_training_clues,
    make_correlated_clue_dataset,
    reference_fits,
)
from graph_specialisation_metrics.synthetic.molecular_redundant_record_router import (
    MolecularSupport,
)


def _supports(count: int = 32):
    return [
        MolecularSupport(
            graph_id=index,
            anchor=0,
            record_nodes=(1, 4, 5, 6),
            record_distances=(1, 4, 4, 6),
        )
        for index in range(count)
    ]


def test_shared_noise_one_makes_local_and_selected_clues_identical():
    dataset = make_correlated_clue_dataset(
        _supports(),
        examples_per_graph=8,
        shared_noise=1.0,
        clue_noise=0.5,
        seed=3,
    )
    rows = torch.arange(len(dataset))
    selected = 2 * dataset.structure + dataset.query

    assert torch.equal(dataset.local_value, dataset.record_values[rows, selected])
    assert not torch.equal(dataset.local_value, dataset.target)


def test_two_clues_lose_unique_gain_when_all_noise_is_shared():
    supports = _supports(128)
    independent_train = make_correlated_clue_dataset(
        supports,
        examples_per_graph=8,
        shared_noise=0.0,
        clue_noise=0.5,
        seed=4,
    )
    independent_test = make_correlated_clue_dataset(
        supports,
        examples_per_graph=8,
        shared_noise=0.0,
        clue_noise=0.5,
        seed=5,
    )
    shared_train = make_correlated_clue_dataset(
        supports,
        examples_per_graph=8,
        shared_noise=1.0,
        clue_noise=0.5,
        seed=4,
    )
    shared_test = make_correlated_clue_dataset(
        supports,
        examples_per_graph=8,
        shared_noise=1.0,
        clue_noise=0.5,
        seed=5,
    )

    independent = reference_fits(independent_train, independent_test)
    shared = reference_fits(shared_train, shared_test)

    assert independent["available_two_clue_gain"] > 0.04
    assert abs(shared["available_two_clue_gain"]) < 1.0e-8


def test_training_dropout_hides_only_clues():
    dataset = make_correlated_clue_dataset(
        _supports(),
        examples_per_graph=8,
        shared_noise=1.0,
        clue_noise=0.5,
        seed=3,
    )
    dropped = drop_training_clues(dataset, probability=0.25, seed=8)

    assert torch.equal(dropped.target, dataset.target)
    assert torch.equal(dropped.query, dataset.query)
    assert int(torch.count_nonzero(dropped.local_value == 0.0)) > 0
    assert int(torch.count_nonzero(dropped.record_values == 0.0)) > 0


def test_config_rejects_invalid_shared_noise(tmp_path: Path):
    config = Config(
        output_dir=tmp_path,
        data_root=tmp_path,
        seeds=(0,),
        shared_noise=(0.0, 1.1),
    )

    try:
        config.validate()
    except ValueError as error:
        assert "shared_noise" in str(error)
    else:
        raise AssertionError("invalid shared noise should be rejected")
