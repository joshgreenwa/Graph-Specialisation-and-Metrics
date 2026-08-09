import csv
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

from graph_specialisation_metrics.chapter6_peptides import (
    FIGURE_FILENAMES,
    PEPTIDE_DISTANCE_BINS,
    dataset_spec,
    peptide_carriage_profile,
    read_manifest,
    run,
)

matplotlib.use("Agg")


def _score(*, vnode: bool, seed: int) -> dict:
    axis = (0, 1, 2, 3, 4, 5, 7, 10, 15) + (("virtual",) if vnode else ())
    width = len(axis)
    semantic = np.zeros((2, 2, width), dtype=np.float64)
    structural = np.zeros_like(semantic)
    attention = np.zeros_like(semantic)
    for position in range(width):
        semantic[..., position] = 0.2 + 0.03 * position + 0.01 * seed
        structural[..., position] = 0.15 + 0.04 * (width - position)
        attention[..., position] = 0.1 + 0.02 * position
    semantic[0, 0] *= 1.5
    structural[1, 1] *= 1.6

    def channel(values: np.ndarray) -> dict:
        return {
            "raw": values.sum(axis=-1),
            "heatmap_exact_head": values,
            "heatmap_per_opportunity_head": values,
            "graph_distance_contribution": {0: values, 1: values * 1.1},
            "graph_distance_support": {
                0: np.ones(width, dtype=np.float64),
                1: np.ones(width, dtype=np.float64),
            },
            "events": [
                {"graph_id": graph, "source": 0, "draw": 0} for graph in (0, 1)
            ],
            "distance_support": {
                "reportable": np.ones(width, dtype=bool),
                "minimum_graphs": 1,
                "minimum_pairs": 1,
            },
        }

    raw_semantic = semantic.sum(axis=-1)
    raw_structural = structural.sum(axis=-1)
    joint = raw_semantic + raw_structural
    return {
        "axis": axis,
        "channels": {
            "semantic": channel(semantic),
            "structural": channel(structural),
        },
        "coordinates": {
            "normalized_semantic": raw_semantic / raw_semantic.mean(),
            "normalized_structural": raw_structural / raw_structural.mean(),
            "joint_sensitivity": joint,
            "selectivity": (raw_semantic - raw_structural) / joint,
        },
        "clean_attention_distance": attention,
    }


def _carriage(*, vnode: bool, seed: int) -> dict:
    distances: list[tuple[float, str]] = [
        (0.0, "molecular_node"),
        (1.0, "molecular_node"),
        (2.0, "molecular_node"),
        (3.0, "molecular_node"),
        (4.0, "molecular_node"),
        (5.0, "molecular_node"),
        (6.0, "molecular_node"),
        (8.0, "molecular_node"),
        (12.0, "molecular_node"),
        (18.0, "molecular_node"),
    ]
    if vnode:
        distances.append((float("nan"), "virtual_node"))
    channels = {}
    for channel_index, channel in enumerate(("semantic", "structural")):
        rows = []
        for graph in range(2):
            for carrier, (distance, kind) in enumerate(distances):
                rows.append(
                    {
                        "seed": seed,
                        "graph_id": graph,
                        "source": 0,
                        "donor": 1,
                        "carrier": carrier,
                        "distance": distance,
                        "carrier_kind": kind,
                        "F_sens": float(1.0 + channel_index + 0.1 * carrier),
                    }
                )
        channels[channel] = {"pairs": rows}
    return {"channels": channels}


def _write_run(root: Path, task: str, seed: int) -> None:
    task_dir = root / task / f"seed_{seed}"
    vnode = "vnode" in task
    score_path = task_dir / "cache/scores/raw.pt"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": {
                "protocol_version": "donor-swap-specialisation-carriage-v4",
                "contract": {"task": task},
            },
            "value": _score(vnode=vnode, seed=seed),
        },
        score_path,
    )
    carriage_path = task_dir / "cache/carriage/fields.pt"
    carriage_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"metadata": {}, "value": _carriage(vnode=vnode, seed=seed)},
        carriage_path,
    )


def test_peptide_specs_and_distance_bins_are_bounded():
    assert len(PEPTIDE_DISTANCE_BINS) == 9
    assert PEPTIDE_DISTANCE_BINS[-1].label == "15+"
    for dataset in ("peptides_func", "peptides_struct"):
        spec = dataset_spec(dataset)
        assert len(spec.tasks) == 5
        assert sum("vnode" in task for task in spec.tasks) == 2
        assert spec.tasks[-1] == f"{dataset}_dense"


def test_peptide_carriage_profile_preserves_exact_weighted_distance():
    labels, response, weighted = peptide_carriage_profile(
        _carriage(vnode=True, seed=0),
        "semantic",
        minimum_graphs=1,
        minimum_pairs=1,
    )
    assert labels == tuple(group.label for group in PEPTIDE_DISTANCE_BINS) + ("virtual",)
    assert np.isfinite(response).all()
    assert np.isfinite(weighted[:-1]).all()
    assert np.isnan(weighted[-1])
    position = labels.index("7-9")
    assert weighted[position] / response[position] == pytest.approx(8.0)


@pytest.mark.parametrize("dataset", ("peptides_func", "peptides_struct"))
def test_peptide_run_emits_only_requested_pdf_figures(tmp_path: Path, dataset: str):
    canonical_root = tmp_path / "canonical"
    for task in dataset_spec(dataset).tasks:
        for seed in (0, 1, 2):
            _write_run(canonical_root, task, seed)

    output_dir = tmp_path / "analysis" / dataset
    manifest = run(
        canonical_root,
        output_dir,
        dataset=dataset,
        minimum_graphs=1,
        minimum_pairs=1,
        verbose=False,
    )
    assert manifest["runs_loaded"] == 15
    figure_files = {path.name for path in (output_dir / "figures").iterdir()}
    assert figure_files == set(FIGURE_FILENAMES)
    assert all(path.suffix == ".pdf" and path.stat().st_size > 0 for path in (output_dir / "figures").iterdir())
    assert read_manifest(output_dir) is not None

    with (output_dir / "final_state_response_variants.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        response_rows = list(csv.DictReader(handle))
    for task in dataset_spec(dataset).tasks:
        normalised = [
            float(row["response_mean"])
            for row in response_rows
            if row["task"] == task
            and row["variant"] == "normalised"
            and np.isfinite(float(row["response_mean"]))
        ]
        assert sum(normalised) == pytest.approx(1.0)

    with (output_dir / "final_state_response_expected_distance.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        expected_rows = list(csv.DictReader(handle))
    assert len(expected_rows) == 10
    assert all(np.isfinite(float(row["expected_distance_mean"])) for row in expected_rows)
