import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

from graph_specialisation_metrics.chapter6_multiseed import (
    alignment_summary_rows,
    cache_inventory,
    dataset_spec,
    per_opportunity_head_profile,
    raw_carriage_strength_profile,
    run,
)

matplotlib.use("Agg")


def _score(*, vnode: bool, seed: int) -> dict:
    axis = (0, 1, 2, "virtual") if vnode else (0, 1, 2)
    semantic = np.asarray(
        [
            [[1.0, 0.3, 0.0], [0.2, 0.7, 0.1]],
            [[0.1, 0.7, 0.2], [0.0, 0.3, 0.9]],
        ],
        dtype=np.float64,
    )
    structural = np.asarray(
        [
            [[0.6, 0.6, 0.1], [0.1, 0.7, 0.4]],
            [[0.0, 0.6, 0.7], [0.0, 0.2, 1.1]],
        ],
        dtype=np.float64,
    )
    if vnode:
        semantic = np.concatenate(
            [semantic, np.full((*semantic.shape[:2], 1), 0.2 + 0.01 * seed)], axis=-1
        )
        structural = np.concatenate(
            [structural, np.full((*structural.shape[:2], 1), 0.08)], axis=-1
        )

    def channel(values: np.ndarray) -> dict:
        raw = values.sum(axis=-1)
        return {
            "raw": raw,
            "heatmap_exact_head": values,
            "heatmap_per_opportunity_head": values,
            "graph_distance_contribution": {0: values, 1: values * 1.1},
            "graph_distance_support": {
                0: np.ones(len(axis), dtype=np.float64),
                1: np.ones(len(axis), dtype=np.float64),
            },
            "events": [{"graph_id": graph, "source": 0, "draw": 0} for graph in (0, 1)],
            "distance_support": {
                "reportable": np.ones(len(axis), dtype=bool),
                "minimum_graphs": 1,
                "minimum_pairs": 1,
            },
        }

    raw_semantic = semantic.sum(axis=-1)
    raw_structural = structural.sum(axis=-1)
    joint = raw_semantic + raw_structural
    attention = np.full_like(semantic, 0.1)
    attention[..., 1] = 0.8
    if vnode:
        attention[..., -1] = 0.3
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
        "families": {
            "semantic_leaning": ((0, 0),),
            "structural_leaning": ((1, 1),),
            "central_responsive": ((0, 1), (1, 0)),
            "inactive": (),
        },
        "clean_attention_distance": attention,
    }


def _carriage(seed: int, *, vnode: bool) -> dict:
    distances: list[tuple[float, str, float]] = [
        (0.0, "molecular_node", 1.0),
        (1.0, "molecular_node", 2.0),
        (2.0, "molecular_node", 1.0),
    ]
    if vnode:
        distances.append((float("nan"), "virtual", 0.5))
    rows = []
    for graph in range(10):
        for source in range(5):
            for carrier, (distance, kind, value) in enumerate(distances):
                rows.append(
                    {
                        "seed": seed,
                        "graph_id": graph,
                        "source": source,
                        "donor": 1,
                        "carrier": carrier,
                        "distance": distance,
                        "carrier_kind": kind,
                        "F_sens": value,
                    }
                )
    return {"channels": {channel: {"pairs": rows} for channel in ("semantic", "structural")}}


def _write_run(root: Path, task: str, seed: int) -> None:
    task_dir = root / task / f"seed_{seed}"
    score_path = task_dir / "cache/scores/raw.pt"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    vnode = "vnode" in task
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
    torch.save({"metadata": {}, "value": _carriage(seed, vnode=vnode)}, carriage_path)
    (task_dir / "model.json").write_text(
        json.dumps({"train_seed": seed, "test_metric": 0.1}), encoding="utf-8"
    )


@pytest.mark.parametrize("dataset", ("zinc", "qm9"))
def test_dataset_spec_has_five_models(dataset):
    spec = dataset_spec(dataset)
    assert len(spec.tasks) == 5
    assert len(spec.labels) == 5
    assert sum("vnode" in task for task in spec.tasks) == 2


def test_alignment_summary_reports_coarse_agreement():
    rows = [
        {
            "task": "zinc",
            "seed": seed,
            "joint_sensitivity": 1.0 + head,
            "semantic_expected_distance": float(head),
            "structural_expected_distance": float(head),
            "semantic_peak_distance": head,
            "structural_peak_distance": head,
        }
        for seed in (0, 1, 2)
        for head in (0, 1, 2)
    ]
    summary = alignment_summary_rows(rows, activity_quantile=0.0)
    assert summary[0]["spearman_rho"] == pytest.approx(1.0)
    assert summary[0]["same_peak_fraction"] == pytest.approx(1.0)
    assert summary[0]["same_or_adjacent_peak_fraction"] == pytest.approx(1.0)


def test_matched_profiles_control_for_opportunity_and_use_raw_carriage():
    score = _score(vnode=False, seed=0)
    labels, score_profile = per_opportunity_head_profile(score, "semantic")
    assert labels[:3] == ("0", "1", "2")
    assert np.nansum(score_profile) == pytest.approx(1.0)

    labels, carriage_profile = raw_carriage_strength_profile(
        _carriage(0, vnode=False),
        "semantic",
        minimum_graphs=1,
        minimum_pairs=1,
    )
    assert labels[:3] == ("0", "1", "2")
    np.testing.assert_allclose(carriage_profile[:3], (0.25, 0.5, 0.25))
    assert np.nansum(carriage_profile) == pytest.approx(1.0)

    _, raw_carriage = raw_carriage_strength_profile(
        _carriage(0, vnode=False),
        "semantic",
        minimum_graphs=1,
        minimum_pairs=1,
        normalise=False,
    )
    np.testing.assert_allclose(raw_carriage[:3], (1.0, 2.0, 1.0))


@pytest.mark.parametrize("dataset", ("zinc", "qm9"))
def test_run_builds_dataset_specific_multiseed_suite(tmp_path, dataset):
    spec = dataset_spec(dataset)
    canonical_root = tmp_path / "canonical_outputs"
    for task in spec.tasks:
        for seed in (0, 1, 2):
            _write_run(canonical_root, task, seed)

    inventory_rows = cache_inventory(canonical_root, dataset=dataset)
    assert len(inventory_rows) == 15
    assert all(row["score_exists"] and row["carriage_exists"] for row in inventory_rows)

    output_dir = tmp_path / "analysis" / dataset
    manifest = run(canonical_root, output_dir, dataset=dataset, verbose=False)
    assert manifest["runs_loaded"] == 15
    assert manifest["dataset"] == dataset
    assert len([path for path in manifest["figures"] if path.endswith(".png")]) == 9
    assert (output_dir / "figures/01_spatial_organisation.png").is_file()
    assert (output_dir / "figures/07_score_and_final_state_response.pdf").is_file()
    assert (output_dir / "figures/08_matched_score_and_final_state_response.pdf").is_file()
    assert (output_dir / "figures/09_final_state_response_variants.pdf").is_file()
    assert (output_dir / "final_state_response_variants.csv").is_file()
    assert (output_dir / "head_metrics.csv").is_file()
    assert (output_dir / "manifest.json").is_file()
