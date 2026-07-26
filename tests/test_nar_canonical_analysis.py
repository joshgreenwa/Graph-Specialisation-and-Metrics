from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from graph_specialisation_metrics.methodology.figures import carriage_profiles
from graph_specialisation_metrics.methodology.protocol import MethodologyConfig
from graph_specialisation_metrics.methodology.tasks import TASKS
from graph_specialisation_metrics.synthetic.nar_canonical_analysis import (
    best_seed_by_validation,
    categorical_accuracy_metric,
    NarSemanticDonorPool,
    register_nar_tasks,
    task_name,
)


def test_nar_registration_declares_canonical_channel_boundaries(tmp_path: Path) -> None:
    name = task_name("dense", 997)
    try:
        names = register_nar_tasks(
            models=("dense",),
            analysis_ns=(997,),
            width=32,
            training_run_dir=tmp_path,
        )
        assert names == (name,)
        task = TASKS[name]
        assert task.backend_kind == "nar_grit"
        assert callable(task.metric_fn)
        assert task.semantic_fields == ("x",)
        assert task.dense_pair_structural_fields == ("rrwp",)
        assert "adj" in task.fixed_support_fields
        assert task.adapter_version.endswith("role-aligned-donors")
        logits = np.asarray([[0.1, 1.2, -0.5], [2.0, 0.3, -1.0]])
        labels = np.asarray([[1], [2]])
        assert task.metric_fn(logits, labels) == 0.5
    finally:
        TASKS.pop(name, None)


def test_nar_categorical_accuracy_validates_graph_alignment() -> None:
    with np.testing.assert_raises_regex(ValueError, "graph counts differ"):
        categorical_accuracy_metric(
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            np.asarray([0]),
        )


def test_best_seed_selection_uses_validation_not_heldout() -> None:
    rows = [
        {
            "model": "dense",
            "N": 16,
            "seed": 0,
            "validation_loss": 0.30,
            "validation_accuracy": 0.95,
            "accuracy": 1.00,
        },
        {
            "model": "dense",
            "N": 16,
            "seed": 1,
            "validation_loss": 0.10,
            "validation_accuracy": 0.91,
            "accuracy": 0.80,
        },
        {
            "model": "dense",
            "N": 16,
            "seed": 2,
            "validation_loss": 0.20,
            "validation_accuracy": 0.94,
            "accuracy": 0.99,
        },
    ]
    assert best_seed_by_validation(rows) == {("dense", 16): 1}


def test_nar_semantic_donors_preserve_query_or_record_role() -> None:
    def graph(query_key: int, values: tuple[int, int, int, int]):
        x = torch.tensor(
            [
                [4, 4],  # centre
                [4, 4],  # intermediate
                [query_key, 4],
                *[[key, value] for key, value in enumerate(values)],
            ],
            dtype=torch.long,
        )
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 0, 3, 0, 4, 0, 5, 0, 6],
                [1, 0, 2, 1, 3, 0, 4, 0, 5, 0, 6, 0],
            ],
            dtype=torch.long,
        )
        return SimpleNamespace(
            x=x,
            edge_index=edge_index,
            num_nodes=7,
            query_idx=torch.tensor(2),
            record_mask=torch.tensor([False, False, False, True, True, True, True]),
        )

    pool = NarSemanticDonorPool(
        ((10, graph(1, (0, 3, 2, 1))), (11, graph(2, (1, 0, 3, 2)))),
        records=4,
    )
    query = pool.draw((0, 4), 1, 8, np.random.default_rng(5))
    assert all(donor.payload[1] == 4 for donor in query)
    assert all(donor.node == 2 for donor in query)

    record = pool.draw((2, 0), 1, 8, np.random.default_rng(7))
    assert all(donor.payload[0] == 2 for donor in record)
    assert all(donor.payload != (2, 0) for donor in record)


def test_functional_only_carriage_figure_has_one_panel() -> None:
    fig, axes = carriage_profiles(
        ("0", "1", "2"),
        (0.9, 0.4, 0.1),
        None,
        functional_interval=((0.8, 0.3, 0.05), (1.0, 0.5, 0.2)),
        channel="semantic",
    )
    try:
        assert len(axes) == 1
        assert axes[0].get_title() == "Functional carriage"
    finally:
        plt.close(fig)


def test_beneficial_carriage_choice_is_cache_fingerprinted() -> None:
    included = MethodologyConfig(compute_beneficial_carriage=True)
    omitted = MethodologyConfig(compute_beneficial_carriage=False)
    assert included.fingerprint != omitted.fingerprint
    assert omitted.scientific_record["beneficial_sign"] == "not_computed"
