import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
import torch

from graph_specialisation_metrics.chapter6_clean_ablation import (
    SUMMARY_SCHEMA,
    ensure_completion_manifest,
)
from graph_specialisation_metrics.chapter6_clean_ablation import (
    completion_path as ablation_completion_path,
)
from graph_specialisation_metrics.chapter6_clean_ablation import (
    load_rows as load_ablation_rows,
)
from graph_specialisation_metrics.chapter6_clean_ablation import (
    missing_runs as missing_ablation_runs,
)
from graph_specialisation_metrics.chapter6_clean_ablation import (
    summary_path as ablation_summary_path,
)
from graph_specialisation_metrics.chapter6_multiseed import (
    alignment_summary_rows,
    cache_inventory,
    dataset_spec,
    per_opportunity_head_profile,
    raw_carriage_strength_profile,
    run,
)
from graph_specialisation_metrics.chapter6_score_trajectory import (
    cache_inventory as trajectory_cache_inventory,
)
from graph_specialisation_metrics.chapter6_score_trajectory import (
    load_rows as load_trajectory_rows,
)
from graph_specialisation_metrics.chapter6_score_trajectory import (
    missing_architectures as missing_trajectory_architectures,
)
from graph_specialisation_metrics.chapter6_score_trajectory import (
    score_cache_path as trajectory_score_cache_path,
)
from graph_specialisation_metrics.methodology.protocol import stable_hash

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


def _write_ablation(root: Path, task: str, seed: int) -> None:
    path = ablation_summary_path(root, task, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": SUMMARY_SCHEMA,
                "task": task,
                "seed": seed,
                "layers": 2,
                "heads_per_layer": 2,
                "clean_ablation_graphs": 64,
                "heads": [
                    {
                        "layer": layer,
                        "head": head,
                        "joint_sensitivity": float(1 + 2 * layer + head),
                        "prediction_movement": float(0.1 + 0.2 * layer + 0.05 * head),
                        "loss_change": 0.0,
                    }
                    for layer in range(2)
                    for head in range(2)
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_trajectory(root: Path, architecture: str, epoch: int) -> None:
    task = "zinc" if architecture == "dense" else "zinc_1hop"
    path = trajectory_score_cache_path(root, architecture, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "task": task,
        "train_seed": 0,
        "checkpoint_sha256": f"{architecture}-{epoch}",
    }
    score = _score(vnode=False, seed=0)
    scale = 1.0 + epoch / 1_000.0 + (0.25 if architecture == "dense" else 0.0)
    for channel in ("semantic", "structural"):
        score["channels"][channel]["raw"] = np.asarray(score["channels"][channel]["raw"]) * scale
    torch.save(
        {
            "metadata": {
                "protocol_version": "donor-swap-specialisation-carriage-v4",
                "contract": contract,
                "contract_fingerprint": stable_hash(contract),
            },
            "value": score,
        },
        path,
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
            "joint_sensitivity": 0.0 if head == 0 else 1.0 + head,
            "semantic_expected_distance": float(head),
            "structural_expected_distance": float(head),
            "semantic_peak_distance": head,
            "structural_peak_distance": head,
        }
        for seed in (0, 1, 2)
        for head in (0, 1, 2)
    ]
    summary = alignment_summary_rows(rows)
    assert summary[0]["heads"] == 9
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


def test_clean_ablation_summary_inventory_and_loading(tmp_path):
    spec = dataset_spec("zinc")
    root = tmp_path / "ablations"
    for task in spec.tasks:
        for seed in (0, 1, 2):
            _write_ablation(root, task, seed)
    assert missing_ablation_runs(root, dataset="zinc") == []
    rows = load_ablation_rows(root, dataset="zinc", strict=True)
    assert len(rows) == 5 * 3 * 4
    assert {row["clean_ablation_graphs"] for row in rows} == {64}
    marker = ensure_completion_manifest(root, dataset="zinc")
    assert marker == ablation_completion_path(root, "zinc")
    assert json.loads(marker.read_text())["runs"] == 15


def test_zinc_checkpoint_trajectory_loads_twelve_cached_epochs(tmp_path):
    root = tmp_path / "trajectory"
    assert missing_trajectory_architectures(root) == ("dense", "1hop")
    for architecture in ("dense", "1hop"):
        for epoch in (10, 100, 250, 500, 1_000, 1_990):
            _write_trajectory(root, architecture, epoch)
        expected_missing = ("1hop",) if architecture == "dense" else ()
        assert missing_trajectory_architectures(root) == expected_missing
    inventory = trajectory_cache_inventory(root)
    assert len(inventory) == 12
    assert all(row["score_exists"] for row in inventory)
    rows = load_trajectory_rows(root, strict=True)
    assert len(rows) == 12 * 4
    assert {row["architecture"] for row in rows} == {"dense", "1hop"}
    assert all(
        np.isfinite(row["normalized_semantic"])
        and np.isfinite(row["normalized_structural"])
        for row in rows
    )


@pytest.mark.parametrize("dataset", ("zinc", "qm9"))
def test_run_builds_dataset_specific_multiseed_suite(tmp_path, dataset):
    spec = dataset_spec(dataset)
    canonical_root = tmp_path / "canonical_outputs"
    ablation_root = tmp_path / "clean_head_ablation"
    trajectory_root = tmp_path / "trajectory"
    for task in spec.tasks:
        for seed in (0, 1, 2):
            _write_run(canonical_root, task, seed)
            _write_ablation(ablation_root, task, seed)
    if dataset == "zinc":
        for architecture in ("dense", "1hop"):
            for epoch in (10, 100, 250, 500, 1_000, 1_990):
                _write_trajectory(trajectory_root, architecture, epoch)

    inventory_rows = cache_inventory(canonical_root, dataset=dataset)
    assert len(inventory_rows) == 15
    assert all(row["score_exists"] and row["carriage_exists"] for row in inventory_rows)

    output_dir = tmp_path / "analysis" / dataset
    manifest = run(
        canonical_root,
        output_dir,
        dataset=dataset,
        ablation_root=ablation_root,
        strict_ablation=True,
        trajectory_root=trajectory_root if dataset == "zinc" else None,
        strict_trajectory=dataset == "zinc",
        verbose=False,
    )
    assert manifest["runs_loaded"] == 15
    assert manifest["dataset"] == dataset
    expected_pngs = 15 if dataset == "zinc" else 13
    assert len([path for path in manifest["figures"] if path.endswith(".png")]) == expected_pngs
    assert (output_dir / "figures/01_spatial_organisation.png").is_file()
    assert (output_dir / "figures/01b_expected_graph_distance.pdf").is_file()
    assert (
        output_dir / "figures/02b_dense_one_hop_attention_specialisation.pdf"
    ).is_file()
    assert (output_dir / "figures/07_score_and_final_state_response.pdf").is_file()
    assert (output_dir / "figures/08_matched_score_and_final_state_response.pdf").is_file()
    assert (output_dir / "figures/09_final_state_response_variants.pdf").is_file()
    assert (output_dir / "figures/09b_final_state_response.pdf").is_file()
    assert (output_dir / "figures/10_joint_sensitivity_head_ablation.pdf").is_file()
    if dataset == "zinc":
        assert (output_dir / "figures/11a_dense_score_trajectory.pdf").is_file()
        assert (output_dir / "figures/11b_1hop_score_trajectory.pdf").is_file()
        assert (output_dir / "zinc_checkpoint_trajectory_heads.csv").is_file()
    else:
        assert not (output_dir / "zinc_checkpoint_trajectory_heads.csv").exists()
    assert (output_dir / "final_state_response_variants.csv").is_file()
    assert (output_dir / "joint_sensitivity_head_ablation.csv").is_file()
    assert (output_dir / "head_metrics.csv").is_file()
    assert (output_dir / "manifest.json").is_file()
